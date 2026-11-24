# src/generate.py
from collections.abc import Iterator
import torch
import torch.nn.functional as F
from src.model import MiniGPT


@torch.no_grad()
def generate_stream(
    model: MiniGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = True,
) -> Iterator[torch.Tensor]:
    """Yields each newly generated token tensor of shape (B, 1) as it is produced."""
    model.eval()
    past_key_values = None

    for step in range(max_new_tokens):
        if use_cache:
            idx_cond = idx if step == 0 else idx[:, -1:]
            if past_key_values is not None and past_key_values[0][0].size(2) >= model.config.block_size:
                idx_cond = idx[:, -model.config.block_size :]
                past_key_values = None
        else:
            idx_cond = idx[:, -model.config.block_size :]

        logits, _, past_key_values = model(
            idx_cond, past_key_values=past_key_values, use_cache=use_cache
        )
        logits = logits[:, -1, :]

        # 1. Greedy Decoding
        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
            yield next_token
            continue

        # 2. Temperature Scaling
        logits = logits / temperature

        # 3. Top-k Filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # 4. Top-p (Nucleus) Filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
        yield next_token


@torch.no_grad()
def generate(
    model: MiniGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """Generates tokens and returns the full sequence (prompt + generated tokens)."""
    new_tokens = list(
        generate_stream(
            model,
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=use_cache,
        )
    )
    if not new_tokens:
        return idx
    return torch.cat([idx, *new_tokens], dim=1)