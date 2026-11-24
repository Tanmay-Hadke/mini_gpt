# src/run_instruct.py
import argparse
import json
import os
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llama_model import LlamaConfig, MiniLlama


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync_device(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def load_hf_into_minillama(model_id: str, device: str) -> MiniLlama:
    print(f"Downloading/loading '{model_id}'...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_cfg = hf_model.config
    sd_hf = hf_model.state_dict()

    has_qkv_bias = "model.layers.0.self_attn.q_proj.bias" in sd_hf

    config = LlamaConfig(
        vocab_size=hf_cfg.vocab_size,
        block_size=min(getattr(hf_cfg, "max_position_embeddings", 2048), 4096),
        n_layer=hf_cfg.num_hidden_layers,
        n_head=hf_cfg.num_attention_heads,
        n_kv_head=hf_cfg.num_key_value_heads,
        n_embd=hf_cfg.hidden_size,
        intermediate_size=hf_cfg.intermediate_size,
        rms_norm_eps=hf_cfg.rms_norm_eps,
        rope_theta=getattr(hf_cfg, "rope_theta", 10000.0),
        qkv_bias=has_qkv_bias,
        tie_word_embeddings=getattr(hf_cfg, "tie_word_embeddings", True),
    )

    model = MiniLlama(config)
    sd = model.state_dict()

    with torch.no_grad():
        for k in sd.keys():
            hf_key = k if k == "lm_head.weight" else f"model.{k}"
            if hf_key not in sd_hf and k == "lm_head.weight":
                hf_key = "model.embed_tokens.weight"
            assert sd_hf[hf_key].shape == sd[k].shape, f"Shape mismatch on {k}"
            sd[k].copy_(sd_hf[hf_key])

    del hf_model, sd_hf
    return model.to(device).eval()


def load_chat_history(path: str, system_msg: dict) -> list[dict]:
    if not os.path.exists(path):
        print(f"[Warning] History file not found at '{path}'. Starting fresh.")
        return [system_msg]

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not all(
            isinstance(m, dict) and "role" in m and "content" in m for m in data
        ):
            raise ValueError("Invalid message format")
        if not data or data[0]["role"] != "system":
            data.insert(0, system_msg)
        print(f"[Loaded {len(data) - 1} messages from '{path}']")
        return data
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[Warning] Could not parse '{path}' ({e}). Starting fresh.")
        return [system_msg]


def build_prompt_ids(tokenizer, messages: list[dict], max_prompt_tokens: int) -> list[int]:
    """Formats conversation history and drops oldest user/assistant pairs if context is full."""
    while True:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.encode(formatted)
        if len(token_ids) <= max_prompt_tokens or len(messages) <= 2:
            return token_ids
        del messages[1:3]


def longest_common_prefix(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def slice_kv_cache(past_key_values: list | None, length: int) -> list | None:
    if past_key_values is None or length == 0:
        return None
    return [(k[:, :, :length, :], v[:, :, :length, :]) for k, v in past_key_values]


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float | None) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def main():
    parser = argparse.ArgumentParser(description="Multi-turn Instruct Chat with Persistent KV Cache")
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM2-360M-Instruct",
        help="e.g. HuggingFaceTB/SmolLM2-360M-Instruct or Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--load-history",
        type=str,
        default=None,
        help="Path to a saved chat history JSON file (e.g. outputs/chat_history.json)",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_hf_into_minillama(args.model, device=device)

    stop_ids = {tokenizer.eos_token_id}
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        for tok in ("<|im_end|>", "<|endoftext|>"):
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int):
                stop_ids.add(tid)

    system_msg = {"role": "system", "content": "You are a helpful, concise assistant."}
    messages = load_chat_history(args.load_history, system_msg) if args.load_history else [system_msg]
    max_prompt_tokens = model.config.block_size - args.max_tokens

    # Persistent KV cache state across turns
    past_key_values = None
    cached_ids: list[int] = []

    print(f"\nMiniLlama ({args.model}) ready!")
    print("Commands: 'history' | 'save' | 'clear' | 'quit'\n")

    while True:
        try:
            user_prompt = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_prompt:
            continue
        if user_prompt.lower() in {"quit", "exit"}:
            break
        if user_prompt.lower() in {"clear", "reset"}:
            messages = [system_msg]
            past_key_values = None
            cached_ids = []
            print("[Conversation history & KV cache cleared]\n")
            continue
        if user_prompt.lower() == "history":
            print("\n--- Conversation History ---")
            for i, msg in enumerate(messages):
                print(f"[{i}] {msg['role'].capitalize()}: {msg['content']}")
            print("----------------------------\n")
            continue
        if user_prompt.lower() == "save":
            os.makedirs("outputs", exist_ok=True)
            save_path = args.load_history or "outputs/chat_history.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2)
            print(f"[Saved history to '{save_path}']\n")
            continue

        messages.append({"role": "user", "content": user_prompt})
        full_ids = build_prompt_ids(tokenizer, messages, max_prompt_tokens)

        # Reuse as much of the existing KV cache as matches the new prompt prefix
        prefix_len = longest_common_prefix(cached_ids, full_ids)
        if prefix_len == len(full_ids):
            prefix_len = max(0, prefix_len - 1)  # Always feed at least 1 token to produce logits

        past_key_values = slice_kv_cache(past_key_values, prefix_len)
        new_prompt_ids = full_ids[prefix_len:]
        prefill_count = len(new_prompt_ids)

        print("Assistant > ", end="", flush=True)
        sync_device(device)
        t0 = time.perf_counter()
        ttft = None
        generated_ids: list[int] = []
        printed_len = 0

        with torch.no_grad():
            # 1. Incremental prefill (only new tokens since last turn)
            idx_chunk = torch.tensor([new_prompt_ids], dtype=torch.long, device=device)
            logits, _, past_key_values = model(idx_chunk, past_key_values=past_key_values, use_cache=True)
            cached_ids = list(full_ids)

            # 2. Autoregressive decode loop
            for _ in range(args.max_tokens):
                next_token = sample_next_token(logits[:, -1, :], args.temperature, args.top_p)
                token_id = next_token.item()

                if ttft is None:
                    sync_device(device)
                    ttft = (time.perf_counter() - t0) * 1000

                if token_id in stop_ids:
                    break

                generated_ids.append(token_id)
                decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
                if not decoded.endswith("\ufffd"):
                    print(decoded[printed_len:], end="", flush=True)
                    printed_len = len(decoded)

                # Guard against exceeding block_size mid-generation
                if len(cached_ids) + 1 >= model.config.block_size:
                    break

                logits, _, past_key_values = model(next_token, past_key_values=past_key_values, use_cache=True)
                cached_ids.append(token_id)

        sync_device(device)
        elapsed = time.perf_counter() - t0
        tok_per_sec = len(generated_ids) / elapsed if elapsed > 0 else 0.0

        assistant_reply = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        messages.append({"role": "assistant", "content": assistant_reply})

        print(
            f"\n\n[{len(generated_ids)} tokens in {elapsed:.2f}s | "
            f"TTFT: {ttft:.1f} ms | {tok_per_sec:.1f} tok/s | "
            f"prefill: {prefill_count}/{len(full_ids)} tok | "
            f"ctx: {len(cached_ids)}/{model.config.block_size}]\n"
        )


if __name__ == "__main__":
    main()