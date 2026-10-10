"""Base classes: a hidden physical parameter and a simulated environment."""
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
    v_soft_max: float | None = None  # above this speed pushes become brakes (see simulate_episodes)

    def derived_targets(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """Extra probe-only targets (functions of probe-space params), e.g. identifiable ratios."""
        return {}

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
    def simulate_episodes(self, s0: np.ndarray, p: np.ndarray, T: int, rng: np.random.Generator,
                          force_prob: float) -> tuple[np.ndarray, np.ndarray]:
        """Simulate with piecewise-constant forces chosen as the episode unfolds.

        Each segment is pushed with probability `force_prob` (else zero force) with a uniform random
        force. force_prob is the identifiability knob: with 0, mass never affects a `forced` trajectory.
        If `v_soft_max` is set and the body is already faster than it, a push is turned into a brake
        (same magnitude, opposite to v), which keeps the velocity distribution stationary over time
        instead of letting a few trajectories run away. Returns states (n, T+1, S), actions (n, T, 1).
        """
        n = len(s0)
        vi = self.state_names.index("v")
        states = np.empty((n, T + 1, s0.shape[1]))
        actions = np.zeros((n, T, 1))
        states[:, 0] = s0
        seg_left, force = np.zeros(n, dtype=int), np.zeros(n)
        for t in range(T):
            new = seg_left == 0
            if new.any():
                k = int(new.sum())
                seg_left[new] = rng.integers(self.seg_len[0], self.seg_len[1] + 1, size=k)
                f = rng.uniform(-self.f_max, self.f_max, size=k)
                if self.v_soft_max is not None:
                    v = states[new, t, vi]
                    f = np.where(np.abs(v) > self.v_soft_max, -np.sign(v) * np.abs(f), f)
                force[new] = np.where(rng.random(k) < force_prob, f, 0.0)
            actions[:, t, 0] = force
            seg_left -= 1
            states[:, t + 1] = self.step(states[:, t], actions[:, t], p)
        return states, actions

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
