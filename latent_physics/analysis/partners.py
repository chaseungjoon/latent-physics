"""Swap partners for the interchange test.

A partner for (trajectory A, time t) is the memory of another trajectory B whose most recent inputs match
A's, so the two memories differ mainly in what they hold about the hidden parameters. Two pools:

standard  -- other trajectories of the same split (B has its own parameters)
same-phi  -- fresh trajectories simulated with exactly A's parameters (true swap effect is zero, so
             whatever the swap does to the prediction is noise from the imperfect history match)
"""
from __future__ import annotations

import numpy as np
import torch

from ..data import Normalizer
from ..envs import Env
from ..models import GRUWorldModel
from .probes import hidden_states


def history(X: np.ndarray, i, t, match_steps: int) -> np.ndarray:
    """The inputs the memory at t-1 has most recently consumed: X[i, t-1], ..., X[i, t-match_steps]."""
    return np.concatenate([X[i, t - j] for j in range(1, match_steps + 1)], -1)


@torch.no_grad()
def match_partners(X: np.ndarray, ia: np.ndarray, ts: np.ndarray, lo: int, device, match_steps: int = 2,
                   chunk: int = 256, exclude: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest neighbour of each query (ia, ts) among all (i, t') with t' in [lo, T) and i != ia
    (and i != exclude, if given: one more pool row per query to skip, e.g. to get a second partner).

    X: (n, T, D) inputs of the pool; ia indexes its rows. Returns partner rows ib, partner times tb (the
    partner memory is the state at tb-1), and the match distances.
    """
    T = X.shape[1]
    ci, ct = np.meshgrid(np.arange(len(X)), np.arange(lo, T), indexing="ij")
    ci, ct = ci.ravel(), ct.ravel()
    cand = torch.as_tensor(history(X, ci, ct, match_steps), device=device)
    query = torch.as_tensor(history(X, ia, ts, match_steps), device=device)
    ci_t = torch.as_tensor(ci, device=device)
    ib, tb, dist = np.empty_like(ia), np.empty_like(ts), np.empty(len(ia))
    for s0 in range(0, len(ia), chunk):
        d = torch.cdist(query[s0:s0 + chunk], cand)
        d[torch.as_tensor(ia[s0:s0 + chunk], device=device)[:, None] == ci_t[None]] = float("inf")
        if exclude is not None:
            d[torch.as_tensor(exclude[s0:s0 + chunk], device=device)[:, None] == ci_t[None]] = float("inf")
        j = d.argmin(1)
        dist[s0:s0 + chunk] = d.gather(1, j[:, None])[:, 0].cpu().numpy()
        j = j.cpu().numpy()
        ib[s0:s0 + chunk], tb[s0:s0 + chunk] = ci[j], ct[j]
    return ib, tb, dist


@torch.no_grad()
def same_phi_partners(model: GRUWorldModel, env: Env, norm: Normalizer, params: np.ndarray, query: np.ndarray,
                      T: int, lo: int, force_prob: float, rng: np.random.Generator, pool: int, device,
                      match_steps: int = 2, chunk: int = 25) -> tuple[list[np.ndarray], np.ndarray]:
    """For each query (params[i], history query[i]), simulate `pool` new trajectories with exactly those
    params and return the memory (per GRU layer, (N, H)) of the best history match, plus the distances."""
    n = len(params)
    carries = None
    dist = np.empty(n)
    q = torch.as_tensor(query, device=device)
    for s0 in range(0, n, chunk):
        idx = np.arange(s0, min(s0 + chunk, n))
        p = np.repeat(params[idx], pool, 0)
        st, ac = env.simulate_episodes(env.sample_init(len(p), rng), p, T, rng, force_prob)
        Xs = norm.inputs(st, ac).reshape(len(idx), pool, T, -1)
        hs = np.concatenate([Xs[:, :, lo - k:T - k] for k in range(1, match_steps + 1)], -1)
        cand = torch.as_tensor(hs.reshape(len(idx), -1, hs.shape[-1]), device=device)
        dd = torch.cdist(q[idx][:, None], cand)[:, 0]
        dmin, jj = dd.min(1)
        dist[idx] = dmin.cpu().numpy()
        jj = jj.cpu().numpy()
        sib, tsib = jj // (T - lo), lo + jj % (T - lo)
        hid = hidden_states(model, torch.as_tensor(Xs[np.arange(len(idx)), sib], device=device))
        if carries is None:
            carries = [np.empty((n, hid[k].shape[-1]), np.float32) for k in model.gru_names]
        for li, k in enumerate(model.gru_names):
            carries[li][idx] = hid[k][np.arange(len(idx)), tsib - 1]
    return carries, dist
