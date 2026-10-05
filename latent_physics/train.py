"""Teacher-forced one-step training."""
from __future__ import annotations

import copy
import math
import time

import torch
from torch import nn


@torch.no_grad()
def mse(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, batch_size: int = 1024) -> float:
    model.eval()
    total = 0.0
    for i in range(0, len(X), batch_size):
        total += ((model(X[i:i + batch_size]) - Y[i:i + batch_size]) ** 2).sum().item()
    return total / Y.numel()


def fit(model: nn.Module, X_tr: torch.Tensor, Y_tr: torch.Tensor, X_va: torch.Tensor, Y_va: torch.Tensor,
        *, epochs: int, batch_size: int, lr: float, name: str = "model", log=print) -> list[dict]:
    """Train on full trajectories with MSE on standardized deltas; keeps the best-validation weights."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(X_tr) / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    best, best_state, history = math.inf, None, []
    for epoch in range(epochs):
        model.train()
        t0, running = time.time(), 0.0
        perm = torch.randperm(len(X_tr), device=X_tr.device)
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i:i + batch_size]
            loss = ((model(X_tr[idx]) - Y_tr[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item() * len(idx)
        val = mse(model, X_va, Y_va)
        history.append({"epoch": epoch + 1, "train": running / len(X_tr), "val": val})
        if val < best:
            best, best_state = val, copy.deepcopy(model.state_dict())
        log(f"[{name}] epoch {epoch + 1:3d}/{epochs}  train {running / len(X_tr):.5f}  "
            f"val {val:.5f}  ({time.time() - t0:.1f}s)")
    model.load_state_dict(best_state)
    model.eval()
    return history
