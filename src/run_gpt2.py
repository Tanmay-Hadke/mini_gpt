# src/run_gpt2.py
import argparse
import time
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src.model import GPTConfig, MiniGPT
from src.generate import generate_stream


GPT2_CONFIGS = {
    "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
    "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),
}


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


def load_gpt2_into_minigpt(model_type: str = "gpt2", device: str = "cpu") -> MiniGPT:
    assert model_type in GPT2_CONFIGS, f"Choose from {list(GPT2_CONFIGS.keys())}"
    print(f"Downloading/loading HuggingFace '{model_type}' weights...")

    hf_model = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = hf_model.state_dict()

    cfg = GPT2_CONFIGS[model_type]
    config = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=0.0,
        bias=True,
        use_sdpa=True,
    )

    model = MiniGPT(config)
    sd = model.state_dict()

    transposed = (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    )

    with torch.no_grad():
        for k in sd.keys():
            if k.endswith(transposed):
                sd[k].copy_(sd_hf[k].t())
            else:
                sd[k].copy_(sd_hf[k])

    del hf_model, sd_hf
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description="Interactive streaming GPT-2 using MiniGPT + KV Cache")
    parser.add_argument("--model", type=str, default="gpt2", choices=list(GPT2_CONFIGS.keys()))
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-cache", action="store_true", help="Disable KV caching for comparison")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    model = load_gpt2_into_minigpt(args.model, device=device)

    print("\nMiniGPT (GPT-2 weights) ready! Type a prompt and press Enter (or 'quit' to exit).\n")

    while True:
        try:
            prompt = input("Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit"}:
            break

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        print(f"\n{prompt}", end="", flush=True)

        sync_device(device)
        t0 = time.perf_counter()
        ttft = None
        generated_ids = []
        printed_len = 0

        for next_token in generate_stream(
            model,
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=not args.no_cache,
        ):
            token_id = next_token.item()
            if ttft is None:
                sync_device(device)
                ttft = (time.perf_counter() - t0) * 1000  # Prefill latency in ms

            generated_ids.append(token_id)
            # Incremental UTF-8-safe decode: wait if a multi-byte character is incomplete
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if not decoded.endswith("\ufffd"):
                print(decoded[printed_len:], end="", flush=True)
                printed_len = len(decoded)

        sync_device(device)
        elapsed = time.perf_counter() - t0
        tok_per_sec = len(generated_ids) / elapsed if elapsed > 0 else 0.0

        print(
            f"\n\n[{len(generated_ids)} tokens in {elapsed:.2f}s | "
            f"TTFT: {ttft:.1f} ms | {tok_per_sec:.1f} tok/s | "
            f"cache={'ON' if not args.no_cache else 'OFF'}]\n"
        )


if __name__ == "__main__":
    main()