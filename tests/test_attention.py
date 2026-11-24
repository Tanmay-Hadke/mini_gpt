# tests/test_attention.py
import pytest
import torch
from src.model import GPTConfig, CausalSelfAttention, MiniGPT


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_no_future_token_leakage_forward(use_sdpa: bool):
    """Mutating future tokens must not affect outputs at earlier positions."""
    torch.manual_seed(42)
    config = GPTConfig(
        vocab_size=100, block_size=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0, use_sdpa=use_sdpa
    )
    model = MiniGPT(config).eval()

    x1 = torch.randint(0, 100, (2, 16))
    x2 = x1.clone()
    cutoff = 8
    # Corrupt every token after the cutoff
    x2[:, cutoff:] = (x2[:, cutoff:] + 1) % 100

    with torch.no_grad():
        logits1, _, _ = model(x1)
        logits2, _, _ = model(x2)

    # All positions strictly before cutoff must remain identical
    assert torch.allclose(logits1[:, :cutoff, :], logits2[:, :cutoff, :], atol=1e-6), (
        f"Future token leakage detected (use_sdpa={use_sdpa})!"
    )
    # Sanity check: positions at/after cutoff should differ
    assert not torch.allclose(logits1[:, cutoff:, :], logits2[:, cutoff:, :], atol=1e-4)


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_no_future_token_leakage_gradients(use_sdpa: bool):
    """Gradient of output[t] w.r.t. input[j] must be strictly 0 for all j > t."""
    torch.manual_seed(42)
    config = GPTConfig(
        vocab_size=100, block_size=32, n_layer=1, n_head=4, n_embd=64, dropout=0.0, use_sdpa=use_sdpa
    )
    attn = CausalSelfAttention(config).eval()

    x = torch.randn(1, 12, 64, requires_grad=True)
    out, _ = attn(x)

    target_pos = 5
    loss = out[0, target_pos].sum()
    loss.backward()

    # Gradients for future positions (6..11) must be identically zero
    future_grad_norm = x.grad[0, target_pos + 1 :].abs().max().item()
    assert future_grad_norm == 0.0, f"Gradient leaked from future positions: {future_grad_norm}"

    # Gradients for current/past positions (0..5) should be non-zero
    past_grad_norm = x.grad[0, : target_pos + 1].abs().max().item()
    assert past_grad_norm > 0.0


def test_manual_and_sdpa_attention_parity():
    """Manual causal attention and PyTorch SDPA must agree within float32 tolerance."""
    torch.manual_seed(0)
    cfg_manual = GPTConfig(block_size=32, n_head=4, n_embd=64, dropout=0.0, use_sdpa=False)
    cfg_sdpa = GPTConfig(block_size=32, n_head=4, n_embd=64, dropout=0.0, use_sdpa=True)

    attn_manual = CausalSelfAttention(cfg_manual).eval()
    attn_sdpa = CausalSelfAttention(cfg_sdpa).eval()
    attn_sdpa.load_state_dict(attn_manual.state_dict())

    x = torch.randn(2, 20, 64)
    with torch.no_grad():
        y_manual, _ = attn_manual(x)
        y_sdpa, _ = attn_sdpa(x)

    assert torch.allclose(y_manual, y_sdpa, atol=1e-5, rtol=1e-4)