"""Candidate parameter directions in the GRU memory, and random controls."""
from __future__ import annotations

import numpy as np

from .probes import RidgeProbe, pooled


def param_directions(hid: dict[str, np.ndarray], z: np.ndarray, X: np.ndarray, probes: dict[str, RidgeProbe],
                     layers: list[str], train_idx, t_lo: int) -> dict[str, dict[str, np.ndarray]]:
    """Two candidate unit directions per (layer, param), each (P, D):

    decode -- the linear probe's readout weights: the direction the *probe* reads the param from.
    encode -- regression of the hidden state on all params *and the two most recent inputs*: the direction
              the state actually moves when the param changes, with the current state held fixed. (Without
              the state control, e.g. "heavy" would also pick up "moving slowly".)
    The two differ whenever hidden dims are correlated; which one the model uses is an empirical question.
    """
    T, n = X.shape[1], len(train_idx)
    ctx = np.concatenate([X[train_idx, t_lo:], X[train_idx, t_lo - 1:T - 1]], -1).reshape(n * (T - t_lo), -1)
    out = {"decode": {}, "encode": {}}
    for name in layers:
        H, Z, _ = pooled(hid[name], z, train_idx, t_lo)
        dec = np.stack([probes[name].direction(k) for k in range(z.shape[1])])
        R = np.concatenate([Z, ctx], -1)
        enc = np.linalg.lstsq(R - R.mean(0), H - H.mean(0), rcond=None)[0][: z.shape[1]]
        out["decode"][name] = dec / np.linalg.norm(dec, axis=1, keepdims=True)
        out["encode"][name] = enc / np.linalg.norm(enc, axis=1, keepdims=True)
    return out


def natural_random_directions(hid: dict[str, np.ndarray], layers: list[str], train_idx, t_lo: int, n: int, rng):
    """Random unit directions distributed like the hidden states themselves (N(0, Cov) draws), so the
    control moves the state by a typical natural amount rather than along near-empty dimensions."""
    out = []
    for _ in range(n):
        us = []
        for name in layers:
            H = hid[name][train_idx, t_lo:].reshape(-1, hid[name].shape[-1])
            rows = H[rng.integers(len(H), size=512)] - H.mean(0)
            u = rows.T @ rng.normal(size=len(rows))
            us.append(u / np.linalg.norm(u))
        out.append(us)
    return out
