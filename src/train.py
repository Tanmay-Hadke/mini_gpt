# train.py
import os
import random
import urllib.request
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.model import GPTConfig, MiniGPT
from src.tokenizer import CharTokenizer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters=25):
    model.eval()
    losses = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        split_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            _, loss, _ = model(X, Y)
            split_losses[k] = loss.item()
        losses[split] = split_losses.mean().item()
    model.train()
    return losses


def configure_optimizers(model: MiniGPT, weight_decay: float, lr: float):
    # Decay only 2D parameters (weights); skip 1D parameters (biases, LayerNorm)
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download TinyShakespeare (~1.1 MB) if not present
    data_path = "input.txt"
    if not os.path.exists(data_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
    )
    model = MiniGPT(config).to(device)
    optimizer = configure_optimizers(model, weight_decay=0.1, lr=3e-4)

    max_iters = 1500
    eval_interval = 150
    batch_size = 64
    history = {"step": [], "train": [], "val": []}

    for step in range(max_iters + 1):
        if step % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, config.block_size, batch_size, device)
            history["step"].append(step)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            print(f"Step {step:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

        xb, yb = get_batch(train_data, config.block_size, batch_size, device)
        _, loss, _ = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping for stability
        optimizer.step()

    torch.save({"model": model.state_dict(), "config": config, "text_vocab": text}, "minigpt.pt")

    # Plot training vs validation loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(history["step"], history["train"], label="Train Loss")
    plt.plot(history["step"], history["val"], label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("MiniGPT Training & Overfitting Diagnostics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("loss_curve.png", dpi=150)


if __name__ == "__main__":
    main()