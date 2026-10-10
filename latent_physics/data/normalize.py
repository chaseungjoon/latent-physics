"""Normalization of model inputs and targets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..envs import Env


@dataclass
class Normalizer:
    """Model inputs are [(s_t - mean)/std, a_t/std]; targets are standardized deltas s_{t+1} - s_t."""

    s_mean: np.ndarray
    s_std: np.ndarray
    a_std: np.ndarray
    d_mean: np.ndarray
    d_std: np.ndarray
    p_mean: np.ndarray  # oracle-only: standardized probe-space params
    p_std: np.ndarray
    in_dims: np.ndarray  # observed state dims fed to the model

    @classmethod
    def fit(cls, train: dict, env: Env) -> "Normalizer":
        s = train["states"].reshape(-1, train["states"].shape[-1])
        d = np.diff(train["states"], axis=1).reshape(-1, s.shape[-1])
        a = train["actions"].reshape(-1, train["actions"].shape[-1])
        z = env.to_probe(train["params"])
        eps = 1e-6
        in_dims = np.arange(s.shape[-1]) if env.input_dims is None else np.array(env.input_dims)
        return cls(s.mean(0), s.std(0) + eps, a.std(0) + eps, d.mean(0), d.std(0) + eps,
                   z.mean(0), z.std(0) + eps, in_dims)

    def inputs(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        s = ((states[:, :-1] - self.s_mean) / self.s_std)[..., self.in_dims]
        return np.concatenate([s, actions / self.a_std], -1).astype(np.float32)

    def targets(self, states: np.ndarray) -> np.ndarray:
        return ((np.diff(states, axis=1) - self.d_mean) / self.d_std).astype(np.float32)

    def params(self, params: np.ndarray, env: Env) -> np.ndarray:
        return ((env.to_probe(params) - self.p_mean) / self.p_std).astype(np.float32)

    def torch(self, device) -> dict[str, torch.Tensor]:
        out = {k: torch.as_tensor(np.asarray(v, dtype=np.float32), device=device)
               for k, v in self.__dict__.items() if k != "in_dims"}
        out["in_dims"] = torch.as_tensor(self.in_dims, device=device)
        return out


def oracle_inputs(X: np.ndarray, p_norm: np.ndarray) -> np.ndarray:
    """Append the (standardized) true parameters to every timestep: the privileged upper bound."""
    return np.concatenate([X, np.broadcast_to(p_norm[:, None], (*X.shape[:2], p_norm.shape[-1]))], -1)
