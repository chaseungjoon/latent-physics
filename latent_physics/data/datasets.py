"""Trajectory datasets: one deterministic random stream per (seed, split)."""
from __future__ import annotations

import numpy as np

from ..envs import Env

# train/val/id share the training distribution; comp holds out a parameter corner; extrap leaves the range
SPLIT_KIND = {"train": "id", "val": "id", "id": "id", "comp": "comp", "extrap": "extrap"}
SPLIT_STREAM = {name: i for i, name in enumerate(SPLIT_KIND)}


def generate_split(env: Env, split: str, n: int, T: int, force_prob: float, seed: int) -> dict:
    rng = np.random.default_rng([seed, SPLIT_STREAM[split]])
    params = env.sample_params(n, rng, SPLIT_KIND[split])
    states, actions = env.simulate_episodes(env.sample_init(n, rng), params, T, rng, force_prob)
    return {"states": states, "actions": actions, "params": params}


def make_datasets(env: Env, sizes: dict[str, int], T: int, force_prob: float, seed: int) -> dict[str, dict]:
    return {split: generate_split(env, split, n, T, force_prob, seed) for split, n in sizes.items()}
