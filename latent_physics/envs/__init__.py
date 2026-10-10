"""Simulated dynamical systems with hidden, per-trajectory physical parameters.

Every environment follows s_{t+1} = F(s_t, a_t; phi). The world model sees (s_t, a_t) only;
phi is fixed within a trajectory, varies across trajectories, and is used only for analysis.
"""
from __future__ import annotations

from .base import G, Env, Param
from .forced import ForcedMotion
from .spring import SpringDamper

ENVS = {e.name: e for e in (ForcedMotion, SpringDamper)}


def make_env(name: str) -> Env:
    return ENVS[name]()


__all__ = ["G", "Env", "Param", "ForcedMotion", "SpringDamper", "ENVS", "make_env"]
