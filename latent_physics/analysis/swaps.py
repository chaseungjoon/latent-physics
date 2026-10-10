"""Swap rollouts: put (part of) a partner's memory into trajectory A, then follow A's real inputs.

A subspace is given per GRU layer as an (H, k) matrix with orthonormal columns; `None` means the whole
memory. Two modes:

once   -- replace the subspace component of A's memory with B's at the swap step only; afterwards the
          model updates its memory from A's inputs as usual (the classic interchange intervention)
clamp  -- additionally hold the subspace component at B's value at every following step, so A's
          evidence cannot overwrite it. Separates "not used" from "used but quickly overwritten".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data import Normalizer
from ..envs import Env
from ..models import GRUWorldModel
from .stats import corr


@dataclass
class SwapSet:
    """N swaps: A's memory before the swap, partner memories, and A's next `horizon` inputs."""

    env: Env
    d_std: np.ndarray
    xs: torch.Tensor              # (K, N, D) A's model inputs after the swap
    cA: list[torch.Tensor]        # per GRU layer (N, H): A's memory just before the swap
    cB: list[torch.Tensor]        # per GRU layer (N, H): partner memory
    states: np.ndarray            # (K, N, S) A's states at the K steps
    actions: np.ndarray           # (K, N, A)
    params_a: np.ndarray          # (N, P) A's params
    z_b: np.ndarray               # (N, P) partner params, probe space

    @property
    def horizon(self) -> int:
        return self.xs.shape[0]

    def __len__(self) -> int:
        return self.xs.shape[1]

    def subset(self, idx) -> "SwapSet":
        it = torch.as_tensor(idx, device=self.xs.device)
        return SwapSet(self.env, self.d_std, self.xs[:, it], [c[it] for c in self.cA], [c[it] for c in self.cB],
                       self.states[:, idx], self.actions[:, idx], self.params_a[idx], self.z_b[idx])

    def true_effect(self, ks: list[int], z_b: np.ndarray | None = None) -> np.ndarray:
        """Simulator's change of A's next state (in delta std units) when A's params `ks` take the partner's
        values (or those in z_b), at each of the K steps on A's real states. Shape (K, N, S)."""
        env, K = self.env, self.horizon
        z_a = env.to_probe(self.params_a)
        z = z_a.copy()
        z[:, ks] = (self.z_b if z_b is None else z_b)[:, ks]
        states, actions = self.states.reshape(-1, self.states.shape[-1]), self.actions.reshape(-1, self.actions.shape[-1])
        s_next = env.step(states, actions, np.tile(self.params_a, (K, 1)))
        eff = (env.step(states, actions, np.tile(env.from_probe(z), (K, 1))) - s_next) / self.d_std
        return eff.reshape(K, len(z_a), -1)


def make_swap_set(env: Env, norm: Normalizer, data: dict, hid: dict[str, np.ndarray], layers: list[str],
                  a_rows: np.ndarray, ts: np.ndarray, b_rows: np.ndarray, tb: np.ndarray, horizon: int, device,
                  cB: list[np.ndarray] | None = None) -> SwapSet:
    """Swaps at (a_rows, ts) with partner memories hid[b_rows, tb - 1] (or the given cB). Rows index the
    split `data` and `hid` were computed on."""
    X = norm.inputs(data["states"][a_rows], data["actions"][a_rows])
    n = np.arange(len(a_rows))
    xs = torch.as_tensor(np.stack([X[n, ts + k] for k in range(horizon)]), device=device)
    cA = [torch.as_tensor(hid[name][a_rows, ts - 1], device=device) for name in layers]
    if cB is None:
        cB = [hid[name][b_rows, tb - 1] for name in layers]
    cB = [torch.as_tensor(c, device=device) for c in cB]
    states = np.stack([data["states"][a_rows, ts + k] for k in range(horizon)])
    actions = np.stack([data["actions"][a_rows, ts + k] for k in range(horizon)])
    return SwapSet(env, norm.d_std, xs, cA, cB, states, actions, data["params"][a_rows],
                   env.to_probe(data["params"][b_rows]))


def rollout(model: GRUWorldModel, xs: torch.Tensor, carry: list[torch.Tensor], edit=None) -> torch.Tensor:
    """Predictions (K, N, S) while reading inputs xs from the given memory."""
    ys = []
    for x in xs:
        y, carry = model.step(x, carry, edit)
        ys.append(y)
    return torch.stack(ys)


def swapped_carry(cA: list[torch.Tensor], cB: list[torch.Tensor], Us: list[torch.Tensor] | None) -> list[torch.Tensor]:
    if Us is None:
        return list(cB)
    return [a + ((b - a) @ U) @ U.T for a, b, U in zip(cA, cB, Us)]


def swap_response(model: GRUWorldModel, s: SwapSet, Us: list[torch.Tensor] | None = None, mode: str = "once",
                  base: torch.Tensor | None = None) -> torch.Tensor:
    """Change of A's predictions (K, N, S) caused by the swap. Differentiable w.r.t. Us."""
    if base is None:
        with torch.no_grad():
            base = rollout(model, s.xs, s.cA)
    edit = None
    if mode == "clamp":
        if Us is None:
            edit = lambda i, h: s.cB[i]  # noqa: E731
        else:
            edit = lambda i, h: h + ((s.cB[i] - h) @ Us[i]) @ Us[i].T  # noqa: E731
    elif mode != "once":
        raise ValueError(mode)
    return rollout(model, s.xs, swapped_carry(s.cA, s.cB, Us), edit) - base


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x ** 2).mean()))


def swap_stats(pred: np.ndarray, true: np.ndarray, k_from: int = 0) -> dict:
    """corr / slope pooled over steps k >= k_from after the swap, plus per-step correlations.
    slope 1 = the model fully adopts the partner's value."""
    p, t = pred[k_from:], true[k_from:]
    return {"corr": corr(p, t), "corr_k0": corr(pred[0], true[0]),
            "corr_by_k": [corr(pred[k], true[k]) for k in range(len(pred))],
            "slope": float((p * t).sum() / ((t ** 2).sum() + 1e-12)), "rms_pred": rms(p), "rms_true": rms(t)}
