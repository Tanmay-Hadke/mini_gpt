# tests/test_kv_cache.py
import time
import torch
from src.model import GPTConfig, MiniGPT
from src.generate import generate


def test_kv_cache_token_equivalence():
    """1. Verify generated token IDs match bit-for-bit under greedy decoding."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPTConfig(vocab_size=65, block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.0)
    model = MiniGPT(config).to(device).eval()

    prompt = torch.randint(0, config.vocab_size, (1, 16), device=device)
    max_new_tokens = 50

    out_no_cache = generate(model, prompt.clone(), max_new_tokens=max_new_tokens, temperature=0.0, use_cache=False)
    out_cached = generate(model, prompt.clone(), max_new_tokens=max_new_tokens, temperature=0.0, use_cache=True)

    assert torch.equal(out_no_cache, out_cached), (
        f"Mismatch detected!\nNo cache: {out_no_cache}\nCached:   {out_cached}"
    )
    print("PASS: Token IDs match bit-for-bit.")


def test_kv_cache_logit_parity():
    """2. Verify raw logits match within floating-point tolerance (catches subtle mask/pos bugs)."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPTConfig(vocab_size=65, block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.0)
    model = MiniGPT(config).to(device).eval()

    prompt = torch.randint(0, config.vocab_size, (1, 16), device=device)
    next_token = torch.randint(0, config.vocab_size, (1, 1), device=device)
    full_seq = torch.cat([prompt, next_token], dim=1)

    with torch.no_grad():
        # Full forward pass without cache
        logits_full, _, _ = model(full_seq, use_cache=False)
        last_logit_no_cache = logits_full[:, -1, :]

        # Prefill prompt, then decode single next_token using cache
        _, _, past_kv = model(prompt, use_cache=True)
        logits_step, _, _ = model(next_token, past_key_values=past_kv, use_cache=True)
        last_logit_cached = logits_step[:, -1, :]

    max_diff = (last_logit_no_cache - last_logit_cached).abs().max().item()
    assert torch.allclose(last_logit_no_cache, last_logit_cached, atol=1e-5, rtol=1e-4), (
        f"Logit divergence too high: {max_diff:.2e}"
    )
    print(f"PASS: Logits match within tolerance (max abs diff: {max_diff:.2e}).")


def benchmark_kv_cache_speedup():
    """3. Measure wall-clock speedup over a longer generation horizon."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPTConfig(vocab_size=65, block_size=512, n_layer=6, n_head=6, n_embd=384, dropout=0.0)
    model = MiniGPT(config).to(device).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 32), device=device)
    max_new_tokens = 200

    # Warmup
    _ = generate(model, prompt.clone(), max_new_tokens=5, temperature=0.0, use_cache=True)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = generate(model, prompt.clone(), max_new_tokens=max_new_tokens, temperature=0.0, use_cache=False)
    if device == "cuda":
        torch.cuda.synchronize()
    t_no_cache = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _ = generate(model, prompt.clone(), max_new_tokens=max_new_tokens, temperature=0.0, use_cache=True)
    if device == "cuda":
        torch.cuda.synchronize()
    t_cached = (time.perf_counter() - t0) * 1000

    print(f"No Cache: {t_no_cache:.1f} ms | Cached: {t_cached:.1f} ms | Speedup: {t_no_cache / t_cached:.2f}x")


if __name__ == "__main__":
    test_kv_cache_token_equivalence()
    test_kv_cache_logit_parity()
    benchmark_kv_cache_speedup()