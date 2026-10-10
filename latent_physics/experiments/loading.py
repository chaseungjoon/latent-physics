"""Load a finished baseline run for a follow-up analysis.

Data are regenerated from the run's seed (identically to the baseline), the GRU is loaded from gru.pt,
and the probe train/test trajectory split is redrawn exactly as the baseline drew it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..analysis.probes import hidden_states
from ..data import Normalizer, generate_split
from ..envs import Env, make_env
from ..models import GRUWorldModel


def pick_device(name: str = "auto") -> torch.device:
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name)


def probe_split(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """70/30 split of the id-test trajectories into probe-train and probe-test (first draw from rng)."""
    perm = rng.permutation(n)
    return np.sort(perm[: int(0.7 * n)]), np.sort(perm[int(0.7 * n):])


@dataclass
class LoadedRun:
    dir: Path
    cfg: dict
    env: Env
    norm: Normalizer
    data: dict                    # the id-test split
    X: np.ndarray                 # its model inputs (n, T, D)
    train_idx: np.ndarray         # probe-train trajectories
    test_idx: np.ndarray          # probe-test trajectories
    model: GRUWorldModel
    hid: dict[str, np.ndarray]    # hidden states on the id-test split
    t_lo: int                     # first "late" timestep
    rng: np.random.Generator      # seeded, already advanced past the probe split
    device: torch.device

    @property
    def layers(self) -> list[str]:
        return self.model.gru_names


def load_run(run_dir: Path, device: torch.device, untrained: bool = False) -> LoadedRun:
    """untrained=True gives the same GRU at initialization instead of the trained weights (the baseline seeds
    torch with the run seed right before creating it, so this is exactly the baseline's untrained control)."""
    cfg = json.loads((run_dir / "metrics.json").read_text())["config"]
    env, seed, fp, T = make_env(cfg["env"]), cfg["seed"], cfg["force_prob"], cfg["T"]
    norm = Normalizer.fit(generate_split(env, "train", cfg["n_train"], T, fp, seed), env)
    data = generate_split(env, "id", cfg["n_test"], T, fp, seed)
    rng = np.random.default_rng(seed)
    train_idx, test_idx = probe_split(cfg["n_test"], rng)
    X = norm.inputs(data["states"], data["actions"])
    torch.manual_seed(seed)
    model = GRUWorldModel(X.shape[-1], len(env.state_names), hidden=cfg["hidden"]).to(device)
    if not untrained:
        model.load_state_dict(torch.load(run_dir / "gru.pt", map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hid = hidden_states(model, torch.as_tensor(X, device=device))
    return LoadedRun(run_dir, cfg, env, norm, data, X, train_idx, test_idx, model, hid, T // 5, rng, device)


def find_runs(paths: list[str]) -> list[Path]:
    """The given run dirs, or every runs/*/ that has a trained gru.pt."""
    return [Path(p) for p in paths] if paths else sorted(p.parent for p in Path("runs").glob("*/gru.pt"))
