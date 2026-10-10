"""k-dimensional subspaces of the GRU memory, for subspace swaps (see swaps.py).

Each function returns one (H, k) matrix with orthonormal columns per GRU layer.

random      -- k random directions drawn like the hidden states themselves (N(0, Cov)), orthonormalized
encode_pca  -- top-k principal directions of the part of the memory explained by the params, with
               nonlinear param terms and param x recent-input interactions, recent inputs held fixed
das         -- learned by distributed alignment search: trained so that swapping the subspace from B into
               A moves A's predictions the way the simulator moves when A takes B's params
"""
from __future__ import annotations

import numpy as np
import torch

from ..models import GRUWorldModel
from .probes import pooled
from .swaps import SwapSet, rollout, swap_response


def _orthonormal(M: np.ndarray) -> np.ndarray:
    return np.linalg.qr(M)[0]


def random_subspace(hid: dict[str, np.ndarray], layers: list[str], idx, t_lo: int, k: int, rng) -> list[np.ndarray]:
    out = []
    for name in layers:
        H = hid[name][idx, t_lo:].reshape(-1, hid[name].shape[-1])
        rows = H[rng.integers(len(H), size=512)] - H.mean(0)
        out.append(_orthonormal(rows.T @ rng.normal(size=(len(rows), k))))
    return out


def param_features(z: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    """Param terms used to explain the memory: z, z^2, pairwise z_i z_j, and z x recent inputs."""
    P = z.shape[1]
    pairs = [z[:, i] * z[:, j] for i in range(P) for j in range(i + 1, P)]
    inter = (z[:, :, None] * ctx[:, None, :]).reshape(len(z), -1)
    return np.concatenate([z, z ** 2] + ([np.stack(pairs, -1)] if pairs else []) + [inter], -1)


def encode_pca(hid: dict[str, np.ndarray], z: np.ndarray, X: np.ndarray, layers: list[str], idx, t_lo: int,
               k: int, rng, max_rows: int = 60000) -> list[np.ndarray] | None:
    """Regress the memory on [param features, recent inputs], keep the fitted param part, and return its top-k
    principal directions. None if k exceeds the number of param features (the param part's maximum rank)."""
    T = X.shape[1]
    n = len(idx)
    ctx = np.concatenate([X[idx, t_lo:], X[idx, t_lo - 1:T - 1]], -1).reshape(n * (T - t_lo), -1)
    zz = np.repeat(z[idx], T - t_lo, axis=0)
    rows = rng.choice(len(ctx), size=min(max_rows, len(ctx)), replace=False)
    F = param_features(zz[rows], ctx[rows])
    if k > F.shape[1]:
        return None
    R = np.concatenate([F, ctx[rows]], -1)
    R = (R - R.mean(0)) / (R.std(0) + 1e-8)
    out = []
    for name in layers:
        H, _, _ = pooled(hid[name], z, idx, t_lo)
        H = H[rows] - H[rows].mean(0)
        B = np.linalg.lstsq(R, H, rcond=None)[0]
        part = R[:, :F.shape[1]] @ B[:F.shape[1]]
        out.append(np.linalg.svd(part, full_matrices=False)[2][:k].T)
    return out


def train_das(model: GRUWorldModel, s: SwapSet, target: torch.Tensor, k: int, mode: str, k_from: int = 1,
              steps: int = 1000, batch: int = 256, lr: float = 1e-2, seed: int = 0) -> list[torch.Tensor]:
    """Distributed alignment search: learn one orthonormal (H, k) subspace per GRU layer such that swapping it
    from the partner into A (mode "once" or "clamp", see swaps.py) makes the change of A's predictions match
    `target` (K, N, S), the simulator's change, on steps k >= k_from. The model itself is frozen."""
    device = s.xs.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    Ws = [torch.randn(c.shape[-1], k, generator=gen).to(device).requires_grad_() for c in s.cA]
    opt = torch.optim.Adam(Ws, lr=lr)
    with torch.no_grad():
        base = rollout(model, s.xs, s.cA)
    # cuDNN only backpropagates through an RNN in training mode; the native kernel works in eval mode
    with torch.backends.cudnn.flags(enabled=False):
        for _ in range(steps):
            idx = torch.randint(len(s), (batch,), generator=gen).numpy()
            sub = s.subset(idx)
            Us = [torch.linalg.qr(W)[0] for W in Ws]
            pred = swap_response(model, sub, Us, mode, base[:, idx])
            loss = ((pred[k_from:] - target[k_from:, idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return [torch.linalg.qr(W)[0].detach() for W in Ws]


def subspace_profile(hid: dict[str, np.ndarray], layers: list[str], Us: list, targets: np.ndarray, train_idx, test_idx,
                     t_lo: int) -> dict:
    """How natural a subspace is: the share of the memory's variance (late t) that lies in it (chance = k/H),
    and the held-out linear R^2 of per-trajectory `targets` (n, K) from the memory's coordinates in it."""
    var, f_tr, f_te = [], [], []
    for name, U in zip(layers, Us):
        U = np.asarray(torch.as_tensor(U).cpu(), dtype=np.float64)
        Htr, _, _ = pooled(hid[name], targets, train_idx, t_lo)
        Hte, _, _ = pooled(hid[name], targets, test_idx, t_lo)
        mu = Htr.mean(0)
        C = np.cov(Htr, rowvar=False)
        var.append(float(np.trace(U.T @ C @ U) / np.trace(C)))
        f_tr.append((Htr - mu) @ U)
        f_te.append((Hte - mu) @ U)
    _, Ytr, _ = pooled(hid[layers[0]], targets, train_idx, t_lo)
    _, Yte, _ = pooled(hid[layers[0]], targets, test_idx, t_lo)
    Ftr = np.concatenate(f_tr + [np.ones((len(Ytr), 1))], -1)
    Fte = np.concatenate(f_te + [np.ones((len(Yte), 1))], -1)
    W = np.linalg.lstsq(Ftr, Ytr, rcond=None)[0]
    res = Yte - Fte @ W
    r2 = 1 - (res ** 2).sum(0) / (((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12)
    return {"var_frac": float(np.mean(var)), "r2": r2.tolist()}
