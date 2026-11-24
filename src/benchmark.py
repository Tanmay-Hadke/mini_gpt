# src/benchmark.py
import os
import time
import torch
from transformers import GPT2LMHeadModel

from src.model import GPTConfig, MiniGPT


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


def reset_memory(device: str):
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device == "mps":
        torch.mps.empty_cache()


def get_memory_mb(device: str) -> float | None:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    if device == "mps":
        # MPS exposes current/driver allocated memory rather than peak stats
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return None


def benchmark_attention(device: str, seq_lens=(256, 512, 1024), batch_size=4, iters=15):
    lines = []
    header = f"=== Mastery Challenge: Manual vs. Fused SDPA Attention ({device.upper()}) ==="
    print(header)
    lines.append(header)

    for seq_len in seq_lens:
        torch.manual_seed(0)
        cfg_manual = GPTConfig(
            vocab_size=1000, block_size=seq_len, n_layer=4, n_head=8, n_embd=512, dropout=0.0, use_sdpa=False
        )
        cfg_sdpa = GPTConfig(
            vocab_size=1000, block_size=seq_len, n_layer=4, n_head=8, n_embd=512, dropout=0.0, use_sdpa=True
        )

        model_manual = MiniGPT(cfg_manual).to(device).eval()
        model_sdpa = MiniGPT(cfg_sdpa).to(device).eval()
        model_sdpa.load_state_dict(model_manual.state_dict())

        x = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        # 1. Numerical Agreement Check
        with torch.no_grad():
            logits_manual, _, _ = model_manual(x)
            logits_sdpa, _, _ = model_sdpa(x)

        max_diff = (logits_manual - logits_sdpa).abs().max().item()
        is_close = torch.allclose(logits_manual, logits_sdpa, atol=1e-5, rtol=1e-4)

        # 2. Latency & Memory Profiling
        def profile(m):
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = m(x)
            sync_device(device)
            reset_memory(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(iters):
                    _ = m(x)
            sync_device(device)
            elapsed_ms = (time.perf_counter() - t0) / iters * 1000
            mem_mb = get_memory_mb(device)
            return elapsed_ms, mem_mb

        t_manual, mem_manual = profile(model_manual)
        t_sdpa, mem_sdpa = profile(model_sdpa)
        speedup = t_manual / t_sdpa

        mem_str_m = f"{mem_manual:.1f} MB" if mem_manual is not None else "N/A"
        mem_str_s = f"{mem_sdpa:.1f} MB" if mem_sdpa is not None else "N/A"

        block = (
            f"\n[Batch={batch_size}, SeqLen={seq_len}]\n"
            f"  Numerical Parity (atol=1e-5): {is_close} | Max Abs Diff: {max_diff:.2e}\n"
            f"  Manual Attention : {t_manual:7.2f} ms | Mem: {mem_str_m}\n"
            f"  Fused SDPA       : {t_sdpa:7.2f} ms | Mem: {mem_str_s} | Speedup: {speedup:.2f}x"
        )
        print(block)
        lines.append(block)

    return "\n".join(lines)


def verify_against_huggingface(device: str):
    header = f"\n=== HuggingFace GPT-2 Logit Parity Check ({device.upper()}) ==="
    print(header)

    hf_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    sd_hf = hf_model.state_dict()

    config = GPTConfig(
        vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True, use_sdpa=True
    )
    my_model = MiniGPT(config).to(device).eval()
    sd = my_model.state_dict()

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

    # "Hello, I'm a language model,"
    idx = torch.tensor([[15496, 11, 314, 1101, 257, 3303, 2746, 11]], device=device)

    with torch.no_grad():
        my_logits, _, _ = my_model(idx)
        hf_logits = hf_model(idx).logits

    max_err = (my_logits - hf_logits).abs().max().item()
    is_close = torch.allclose(my_logits, hf_logits, atol=1e-4, rtol=1e-4)

    result = f"  Parity Match (atol=1e-4): {is_close} | Max Logit Error: {max_err:.2e}"
    print(result)
    return f"{header}\n{result}"


def main():
    device = get_device()
    attn_report = benchmark_attention(device)
    hf_report = verify_against_huggingface(device)

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "benchmark_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(attn_report + "\n" + hf_report + "\n")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()