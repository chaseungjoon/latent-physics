"""Simulated dynamical systems with hidden, per-trajectory physical parameters.

Every environment follows s_{t+1} = F(s_t, a_t; phi). The world model sees (s_t, a_t) only;
phi is fixed within a trajectory, varies across trajectories, and is used only for analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass(frozen=True)
class Param:
    name: str
    low: float
    high: float
    log: bool = False  # sample log-uniformly and probe in log space
    extrap_high: float | None = None  # upper end of the extrapolation range (starts at `high`)

    def to_probe(self, x):
        return np.log(x) if self.log else x

    def from_probe(self, z):
        return np.exp(z) if self.log else z


class Env:
    name: str
    state_names: tuple[str, ...]
    params: tuple[Param, ...]
    dt: float
    substeps: int
    f_max: float
    seg_len: tuple[int, int] = (5, 20)  # actions are piecewise constant over segments of this length
    comp_params: tuple[int, int] = (0, 1)  # the high-high corner of these two params is held out
    comp_threshold: float = 0.7  # corner = both params above this quantile of their (probe-space) range
    extrap_param: int = 0
    input_dims: tuple[int, ...] | None = None  # state dims fed to the model (None = all); all dims are predicted

    # ---- parameters -------------------------------------------------------------------------
    def to_probe(self, p: np.ndarray) -> np.ndarray:
        return np.stack([prm.to_probe(p[:, i]) for i, prm in enumerate(self.params)], -1)

    def from_probe(self, z: np.ndarray) -> np.ndarray:
        return np.stack([prm.from_probe(z[:, i]) for i, prm in enumerate(self.params)], -1)

    def in_comp_corner(self, p: np.ndarray) -> np.ndarray:
        u = self._unit(p)
        i, j = self.comp_params
        return (u[:, i] > self.comp_threshold) & (u[:, j] > self.comp_threshold)

    def _unit(self, p: np.ndarray) -> np.ndarray:
        lo = np.array([prm.to_probe(prm.low) for prm in self.params])
        hi = np.array([prm.to_probe(prm.high) for prm in self.params])
        return (self.to_probe(p) - lo) / (hi - lo)

    def sample_params(self, n: int, rng: np.random.Generator, split: str) -> np.ndarray:
        """split: 'id' (training distribution, comp corner excluded), 'comp' (corner only),
        or 'extrap' (extrap_param beyond its training range, others in range)."""
        P = len(self.params)
        i, j = self.comp_params
        thr = self.comp_threshold
        u = rng.uniform(size=(n, P))
        if split == "id":
            corner = (u[:, i] > thr) & (u[:, j] > thr)
            while corner.any():
                u[corner] = rng.uniform(size=(corner.sum(), P))
                corner = (u[:, i] > thr) & (u[:, j] > thr)
        elif split == "comp":
            u[:, [i, j]] = thr + (1 - thr) * u[:, [i, j]]
        elif split != "extrap":
            raise ValueError(split)
        lo = np.array([prm.to_probe(prm.low) for prm in self.params])
        hi = np.array([prm.to_probe(prm.high) for prm in self.params])
        z = lo + u * (hi - lo)
        if split == "extrap":
            prm = self.params[self.extrap_param]
            z[:, self.extrap_param] = rng.uniform(prm.to_probe(prm.high), prm.to_probe(prm.extrap_high), size=n)
        return self.from_probe(z)

    # ---- actions / initial states -------------------------------------------------------------
    def sample_actions(self, n: int, T: int, rng: np.random.Generator, force_prob: float) -> np.ndarray:
        """Piecewise-constant forces. Each segment is pushed with probability `force_prob`, else zero.
        force_prob is the identifiability knob: with 0, mass never affects the trajectory."""
        a = np.zeros((n, T, 1))
        for k in range(n):
            t = 0
            while t < T:
                L = rng.integers(self.seg_len[0], self.seg_len[1] + 1)
                if rng.random() < force_prob:
                    a[k, t:t + L, 0] = rng.uniform(-self.f_max, self.f_max)
                t += L
        return a

    def sample_init(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    # ---- dynamics -------------------------------------------------------------------------------
    def step(self, s: np.ndarray, a: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Vectorized one-step transition. s: (n, S), a: (n, A), p: (n, P) -> (n, S)."""
        raise NotImplementedError

    def simulate(self, s0: np.ndarray, actions: np.ndarray, p: np.ndarray) -> np.ndarray:
        n, T, _ = actions.shape
        states = np.empty((n, T + 1, s0.shape[1]))
        states[:, 0] = s0
        for t in range(T):
            states[:, t + 1] = self.step(states[:, t], actions[:, t], p)
        return states


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


ENVS = {e.name: e for e in (ForcedMotion, SpringDamper)}


def make_env(name: str) -> Env:
    return ENVS[name]()
