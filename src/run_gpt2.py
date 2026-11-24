# src/run_gpt2.py
import argparse
import time
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src.model import GPTConfig, MiniGPT
from src.generate import generate


GPT2_CONFIGS = {
    "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
        bias=True,      # GPT-2 uses bias in Linear and LayerNorm layers
        use_sdpa=True,
    )

    model = MiniGPT(config)
    sd = model.state_dict()

    # HuggingFace GPT-2 uses Conv1D layers whose weights are transposed vs nn.Linear
    transposed = (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    )

    with torch.no_grad():
        for k in sd.keys():
            if k.endswith(transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"Shape mismatch on {k}"
                sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch on {k}"
                sd[k].copy_(sd_hf[k])

    del hf_model, sd_hf  # Free HF model memory
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description="Interactive GPT-2 inference using MiniGPT + KV Cache")
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

        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()
        output_ids = generate(
            model,
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=not args.no_cache,
        )

        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

        elapsed = time.perf_counter() - t0
        generated_tokens = output_ids.size(1) - input_ids.size(1)
        tok_per_sec = generated_tokens / elapsed

        text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        print(f"\n{text}\n")
        print(f"[{generated_tokens} tokens in {elapsed:.2f}s | {tok_per_sec:.1f} tok/s | cache={'ON' if not args.no_cache else 'OFF'}]\n")


if __name__ == "__main__":
    main()