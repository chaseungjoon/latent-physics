"""Collate many runs into one table (mean ± std over seeds per env × force_prob).

    uv run python -m latent_physics.summarize            # all runs under runs/
    uv run python -m latent_physics.summarize runs/foo*  # a subset
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def run_row(res: dict) -> dict[str, float]:
    late = res["prediction"]["late_mse"]
    em, rel = res["probe"]["emergence"], res["probe"]["relevant_r2"]
    trained = [k for k in em if k != "input" and not k.startswith("untrained/")]
    controls = [k for k in em if k.startswith("untrained/")] + ["input"]
    ic = res["interchange"]
    row = {"GRU/MLP": late["gru"]["id"] / late["mlp"]["id"],
           "GRU/oracle": late["gru"]["id"] / late["oracle"]["id"],
           "comp/ID": late["gru"]["comp"] / late["gru"]["id"],
           "extrap/ID": late["gru"]["extrap"] / late["gru"]["id"],
           "swap whole": ic["full"]["corr"], "swap whole k=0": ic["full"]["corr_k0"]}
    nan = float("nan")
    for k, name in enumerate(res["env"]["params"]):
        vals = [rel[l][k] for l in trained]
        row[f"R2 {name} (when it matters)"] = nan if vals[0] is None else max(vals)
        row[f"R2ctl {name} (when it matters)"] = nan if vals[0] is None else max(rel[l][k] for l in controls)
    for k, name in enumerate(res["probe"]["targets"]):
        if not name.startswith("v "):
            row[f"R2 {name} (last t)"] = max(em[l][-1][k] for l in trained)
    for name in res["env"]["params"]:
        best = max(("decode", "encode"), key=lambda d: ic[f"{d}/{name}"]["corr"])
        row[f"swap {name}"] = ic[f"{best}/{name}"]["corr"]
        row[f"swap {name} k=0"] = ic[f"{best}/{name}"]["corr_k0"]
        row[f"swap {name} random"] = ic[f"{best}/{name}"]["rand_corr"]
    return row


def main(paths: list[str]):
    runs = [Path(p) for p in paths] if paths else sorted(Path("runs").glob("*/"))
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in runs:
        f = d / "metrics.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())
        if "relevant_r2" not in res["probe"]:  # produced by an older version: re-run with --reuse-models
            continue
        cfg = res["config"]
        groups[(cfg["env"], cfg["preset"], cfg["force_prob"])].append({"seed": cfg["seed"], **run_row(res)})
    lines = []
    for (env, preset, fp), rows in sorted(groups.items()):
        cols = [c for c in rows[0] if c != "seed"]
        seeds = sorted(r["seed"] for r in rows)
        lines += [f"## {env} ({preset}), force_prob={fp}, seeds={seeds}", "", "| metric | mean | std |", "|---|---|---|"]
        for c in cols:
            v = np.array([r[c] for r in rows], dtype=float)
            mean, std = ("n/a", "n/a") if np.isnan(v).all() else (f"{np.nanmean(v):.3f}", f"{np.nanstd(v):.3f}")
            lines.append(f"| {c} | {mean} | {std} |")
        lines.append("")
    text = "\n".join(lines) if lines else "no runs with metrics.json found"
    print(text)
    out = Path("runs/summary.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text(text + "\n")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
