"""How much each hidden parameter physically matters for the next step."""
from __future__ import annotations

import numpy as np

from ..envs import Env


def param_sensitivity(env: Env, data: dict, idx: np.ndarray, z_std: np.ndarray, t_lo: int) -> np.ndarray:
    """How much each param physically matters at each step: squared change of the next state (in units of
    the delta std) if the param moved by +1 std, for steps t >= t_lo. Returns (len(idx), T - t_lo, P)."""
    states, actions, params = data["states"][idx], data["actions"][idx], data["params"][idx]
    T = actions.shape[1]
    n, Tl = len(idx), T - t_lo
    s = states[:, t_lo:T].reshape(n * Tl, -1)
    a = actions[:, t_lo:].reshape(n * Tl, -1)
    z = np.repeat(env.to_probe(params), Tl, axis=0)
    s_next = env.step(s, a, env.from_probe(z))
    d_std = norm_d_std(data, env)
    out = []
    for k in range(len(env.params)):
        z2 = z.copy()
        z2[:, k] += z_std[k]
        out.append((((env.step(s, a, env.from_probe(z2)) - s_next) / d_std) ** 2).sum(-1).reshape(n, Tl))
    return np.stack(out, -1)


def norm_d_std(data: dict, env: Env) -> np.ndarray:
    """Scale for state changes: std of one-step deltas in this split."""
    return np.diff(data["states"], axis=1).reshape(-1, len(env.state_names)).std(0) + 1e-6
