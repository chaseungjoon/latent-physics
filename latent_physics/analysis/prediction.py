"""Level 1: prediction error over history length and in open-loop rollouts."""
from __future__ import annotations

import numpy as np
import torch

from ..data import Normalizer


@torch.no_grad()
def per_t_mse(model, X: torch.Tensor, Y: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    """One-step teacher-forced MSE at each timestep, shape (T,). Late t = more evidence about phi."""
    model.eval()
    se = torch.zeros(Y.shape[1], device=Y.device)
    for i in range(0, len(X), batch_size):
        se += ((model(X[i:i + batch_size]) - Y[i:i + batch_size]) ** 2).mean(-1).sum(0)
    return (se / len(X)).cpu().numpy()


@torch.no_grad()
def rollout_mse(model, norm: Normalizer, states: np.ndarray, actions: np.ndarray, extra: np.ndarray | None,
                t0: int, horizon: int, device) -> np.ndarray:
    """Teacher-force t0 steps of context, then roll out `horizon` steps on the model's own predictions
    (true actions given). Returns MSE in standardized state units for each horizon step."""
    model.eval()
    nt = norm.torch(device)
    S = torch.as_tensor(states, dtype=torch.float32, device=device)
    A = torch.as_tensor(actions, dtype=torch.float32, device=device)
    E = None if extra is None else torch.as_tensor(extra, dtype=torch.float32, device=device)

    def inp(s, a):
        parts = [((s - nt["s_mean"]) / nt["s_std"])[:, nt["in_dims"]], a / nt["a_std"]]
        return torch.cat(parts + ([] if E is None else [E]), -1)

    carry = None
    for t in range(t0):
        _, carry = model.step(inp(S[:, t], A[:, t]), carry)
    s, errs = S[:, t0].clone(), []
    for h in range(horizon):
        y, carry = model.step(inp(s, A[:, t0 + h]), carry)
        s = s + y * nt["d_std"] + nt["d_mean"]
        errs.append((((s - S[:, t0 + h + 1]) / nt["s_std"]) ** 2).mean().item())
    return np.array(errs)
