# src/llama_model.py
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LlamaConfig:
    vocab_size: int = 49152
    block_size: int = 2048
    n_layer: int = 32
    n_head: int = 15
    n_kv_head: int = 5
    n_embd: int = 960
    intermediate_size: int = 2560
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    qkv_bias: bool = False          # False for SmolLM2/Llama, True for Qwen2.5
    tie_word_embeddings: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability, then cast back
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * normed.to(x.dtype)


def precompute_rope(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute HuggingFace-compatible half-rotation cos/sin tables."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)          # (T, head_dim // 2)
    emb = torch.cat((freqs, freqs), dim=-1)           # (T, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # cos, sin: (1, 1, T, head_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = config.n_head // config.n_kv_head
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to current positions BEFORE caching
        q, k = apply_rope(q, k, cos, sin)

        # Cache the compact n_kv_head tensors (saves memory!)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None
        T_k = k.size(2)

        # Expand KV heads to match Q heads for SDPA
        if self.n_rep > 1:
            k_attn = k.repeat_interleave(self.n_rep, dim=1)
            v_attn = v.repeat_interleave(self.n_rep, dim=1)
        else:
            k_attn, v_attn = k, v

        is_causal = (T == T_k) and (T > 1)
        y = F.scaled_dot_product_attention(q, k_attn, v_attn, is_causal=is_causal)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y), present_kv


class SwiGLUMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, kv_cache=None, use_cache: bool = False):
        attn_out, present_kv = self.self_attn(
            self.input_layernorm(x), cos, sin, kv_cache=kv_cache, use_cache=use_cache
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present_kv


class MiniLlama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.embed_tokens.weight = self.lm_head.weight

        cos, sin = precompute_rope(config.n_embd // config.n_head, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: list | None = None,
        use_cache: bool = False,
    ):
        B, T = idx.size()
        past_len = past_key_values[0][0].size(2) if past_key_values is not None else 0
        assert past_len + T <= self.config.block_size, "Sequence exceeds block_size"

        x = self.embed_tokens(idx)
        cos = self.rope_cos[past_len : past_len + T].to(x.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin[past_len : past_len + T].to(x.dtype).unsqueeze(0).unsqueeze(0)

        new_kv_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present_kv = layer(x, cos, sin, kv_cache=layer_past, use_cache=use_cache)
            if use_cache:
                new_kv_cache.append(present_kv)

        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, new_kv_cache