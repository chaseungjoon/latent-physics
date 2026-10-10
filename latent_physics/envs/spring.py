"""Driven mass-spring-damper."""
from __future__ import annotations

import numpy as np

from .base import Env, Param


class SpringDamper(Env):
    """Driven mass-spring-damper m*x'' + c*x' + k*x = F with hidden m, k, c.

    Without forcing only k/m and c/m are identifiable; the force reveals m itself.
    """

    name = "spring"
    state_names = ("x", "v")
    params = (
        Param("mass", 0.5, 2.0, log=True),
        Param("stiffness", 1.0, 10.0, log=True, extrap_high=25.0),
        Param("damping", 0.1, 1.0, log=True),
    )
    dt = 0.1
    substeps = 20
    f_max = 5.0
    extrap_param = 1

    def derived_targets(self, z):
        # without forcing only these ratios are identifiable (m alone is not)
        return {"k/m": z[:, 1] - z[:, 0], "c/m": z[:, 2] - z[:, 0]}

    def sample_init(self, n, rng):
        return np.stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)], -1)

    def step(self, s, a, p):
        x, v = s[:, 0].copy(), s[:, 1].copy()
        F = a[:, 0]
        m, k, c = p[:, 0], p[:, 1], p[:, 2]
        h = self.dt / self.substeps
        for _ in range(self.substeps):  # semi-implicit Euler
            v = v + (F - c * v - k * x) / m * h
            x = x + v * h
        return np.stack([x, v], -1)
