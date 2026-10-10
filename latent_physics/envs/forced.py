"""Forced 1-D motion with Coulomb friction."""
from __future__ import annotations

import numpy as np

from .base import G, Env, Param


class ForcedMotion(Env):
    """1D block pushed by an external force, with hidden mass m and Coulomb friction mu.

    m*dv/dt = F - mu*m*g*sign(v) while sliding; static friction holds the block while |F| <= mu*m*g.
    Mass shows up only through the response to F (coasting deceleration mu*g is mass-independent),
    while mu shows up in coasting and in the stick/slip threshold.
    """

    name = "forced"
    state_names = ("x", "v")
    params = (
        Param("mass", 1.0, 5.0, log=True, extrap_high=10.0),
        Param("friction", 0.05, 0.3),
    )
    dt = 0.05
    substeps = 10
    f_max = 30.0
    # x is irrelevant to the dynamics and random-walks far outside its early range, so feeding it only
    # adds a drifting nuisance input; the model sees v (and F) and still predicts both dx and dv.
    input_dims = (1,)
    v_soft_max = 15.0

    def sample_init(self, n, rng):
        return np.stack([rng.uniform(-1, 1, n), rng.uniform(-5, 5, n)], -1)

    def step(self, s, a, p):
        x, v = s[:, 0].copy(), s[:, 1].copy()
        F = a[:, 0]
        m, mu = p[:, 0], p[:, 1]
        f_fric = mu * m * G
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            moving = v != 0.0
            # kinetic friction opposes motion; static friction cancels F up to f_fric
            fric = np.where(moving, -f_fric * np.sign(v), -np.clip(F, -f_fric, f_fric))
            v_new = v + (F + fric) / m * h
            v_new = np.where(moving & (v_new * v < 0), 0.0, v_new)  # friction stops, never reverses
            x = x + v_new * h
            v = v_new
        return np.stack([x, v], -1)
