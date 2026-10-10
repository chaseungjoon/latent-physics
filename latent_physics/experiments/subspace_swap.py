"""Subspace swaps: does a k-dim part of the GRU memory carry the physics belief?

For trajectory A, the component of its memory in a k-dim subspace (per GRU layer) is replaced by that of a
history-matched partner B. The change of A's predictions is scored against the simulator's change when A
takes all of B's params, on steps k >= 1 after the swap (k = 0 is dominated by history mismatch; see
PROGRESS.md §3.4).

Subspaces (see analysis/subspaces.py):
  random        control: random directions with the memory's own covariance
  encode_pca    principal directions of the param-explained part of the memory
  das           learned by distributed alignment search on probe-train trajectories, scored on probe-test ones
  das_shuffled  DAS trained toward the effect of a *random other* trajectory's params, which B's memory does
                not hold. Its plain `corr` shows how much of that score erasing A's own belief alone can reach;
                its `corr_import` is zero by construction. The control for an over-expressive search is
                --untrained: the same DAS on the GRU at initialization.
Modes (see analysis/swaps.py): once (swap, then A's inputs update the memory) and clamp (hold the subspace
at B's value for the whole horizon). Every swap is repeated with same-phi partners to give its noise floor.

Two scores. `corr` compares the change with f(phi_B) - f(phi_A). Its -f(phi_A) half depends on A alone, so a
swap that only erases A's own belief already scores on it. `corr_import` removes that half: each test swap
point gets a second partner B2, and the difference of the two responses is compared with
f(phi_B) - f(phi_B2). Only what the partner's memory brings in can explain that difference.

    uv run python -m latent_physics.experiments.subspace_swap runs/spring_full_fp1_s0 [more runs]
    uv run python -m latent_physics.experiments.subspace_swap --summary      # -> runs/subspace_summary.md
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..analysis.partners import history, match_partners, same_phi_partners
from ..analysis.subspaces import encode_pca, random_subspace, subspace_profile, train_das
from ..analysis.swaps import SwapSet, make_swap_set, rms, swap_response, swap_stats
from .loading import LoadedRun, find_runs, load_run, pick_device

KS = (1, 2, 4, 8, 16, 32)
MODES = ("once", "clamp")
HORIZON, MATCH_STEPS, K_FROM = 10, 2, 1
TRAIN_PER_TRAJ, TEST_PER_TRAJ, N_RANDOM = 8, 2, 3


def build_sets(run: LoadedRun) -> tuple[SwapSet, SwapSet, SwapSet, SwapSet]:
    """Swaps on probe-train trajectories (for DAS); on probe-test ones; the same test swap points with a
    second, different partner (for corr_import); and with same-phi partners (noise floor). Partners are
    matched within the same pool."""
    T = run.cfg["T"]
    lo = max(run.t_lo, MATCH_STEPS)

    def swaps(idx, per_traj, cB=None):
        Xp = run.X[idx]
        ia = np.repeat(np.arange(len(idx)), per_traj)
        ts = run.rng.integers(lo, T - HORIZON + 1, size=len(ia))
        ib, tb, _ = match_partners(Xp, ia, ts, lo, run.device, MATCH_STEPS)
        s = make_swap_set(run.env, run.norm, run.data, run.hid, run.layers, idx[ia], ts, idx[ib], tb, HORIZON, run.device)
        return s, Xp, ia, ts, ib, tb

    train, *_ = swaps(run.train_idx, TRAIN_PER_TRAJ)
    test, Xte, ia, ts, ib, tb = swaps(run.test_idx, TEST_PER_TRAJ)
    ib2, tb2, _ = match_partners(Xte, ia, ts, lo, run.device, MATCH_STEPS, exclude=ib)
    test2 = make_swap_set(run.env, run.norm, run.data, run.hid, run.layers, run.test_idx[ia], ts, run.test_idx[ib2],
                          tb2, HORIZON, run.device)
    cB_same, _ = same_phi_partners(
        run.model, run.env, run.norm, run.data["params"][run.test_idx[ia]], history(Xte, ia, ts, MATCH_STEPS), T, lo,
        run.cfg["force_prob"], np.random.default_rng([run.cfg["seed"], 4321]), len(run.test_idx) - 1, run.device,
        MATCH_STEPS)
    same = make_swap_set(run.env, run.norm, run.data, run.hid, run.layers, run.test_idx[ia], ts, run.test_idx[ib], tb,
                         HORIZON, run.device, cB=cB_same)
    return train, test, test2, same


@torch.no_grad()
def evaluate(run: LoadedRun, sets: tuple, Us, mode: str, true: tuple[np.ndarray, np.ndarray]) -> dict:
    """sets = (test, test2, same); true = simulator effects for test and test2."""
    test, test2, same = sets
    pred, pred2, noise = (swap_response(run.model, s, Us, mode).cpu().numpy() for s in (test, test2, same))
    st = swap_stats(pred, true[0], K_FROM)
    imp = swap_stats(pred - pred2, true[0] - true[1], K_FROM)
    nz = rms(noise[K_FROM:])
    return {**st, "corr_import": imp["corr"], "slope_import": imp["slope"], "corr_import_by_k": imp["corr_by_k"],
            "rms_noise": nz, "corr_ceiling": st["rms_true"] / np.hypot(st["rms_true"], nz)}


def overlap(U1: list, U2: list) -> float:
    """Mean over layers of ||U1^T U2||_F^2 / k: 1 = same subspace, k/H = chance."""
    vals = []
    for a, b in zip(U1, U2):
        a, b = np.asarray(torch.as_tensor(a).cpu()), np.asarray(torch.as_tensor(b).cpu())
        vals.append(float(((a.T @ b) ** 2).sum() / a.shape[1]))
    return float(np.mean(vals))


def subspace_swap(run: LoadedRun, ks=KS, das_steps: int = 1000, log=print) -> tuple[dict, dict]:
    """Returns the results and the learned DAS subspaces ({'<method>_<mode>_k<k>_<layer>': (H, k)})."""
    t0 = time.time()
    env, P = run.env, len(run.env.params)
    z = env.to_probe(run.data["params"])
    derived = env.derived_targets(z)
    targets = np.concatenate([z] + ([np.stack(list(derived.values()), -1)] if derived else []), -1)
    target_names = [p.name for p in env.params] + list(derived)
    profile = lambda Us: subspace_profile(run.hid, run.layers, Us, targets, run.train_idx, run.test_idx, run.t_lo)  # noqa: E731
    saved = {}
    train, test, test2, same = build_sets(run)
    sets = (test, test2, same)
    every = list(range(P))
    # targets: the simulator's effect of B's params, and (control) of a random other swap's params
    perm_tr, perm_te, perm_te2 = (run.rng.permutation(len(x)) for x in (train, test, test2))
    to_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=run.device)  # noqa: E731
    tgt = {"das": to_t(train.true_effect(every)), "das_shuffled": to_t(train.true_effect(every, train.z_b[perm_tr]))}
    true = {"das": (test.true_effect(every), test2.true_effect(every)),
            "das_shuffled": (test.true_effect(every, test.z_b[perm_te]), test2.true_effect(every, test2.z_b[perm_te2]))}
    fit_idx = np.arange(min(2000, len(train)))
    fit_set = train.subset(fit_idx)
    log(f"  sets: train {len(train)}, test {len(test)} swaps ({time.time() - t0:.0f}s)")

    out = {"config": run.cfg, "n_train": len(train), "n_test": len(test), "k_from": K_FROM, "profile_targets": target_names,
           "whole": {"once": evaluate(run, sets, None, "once", true["das"])}, "subspaces": defaultdict(dict)}
    for k in ks:
        to_u = lambda Us: [to_t(U) for U in Us]  # noqa: E731
        rand = [to_u(random_subspace(run.hid, run.layers, run.train_idx, run.t_lo, k, run.rng)) for _ in range(N_RANDOM)]
        enc = encode_pca(run.hid, z, run.X, run.layers, run.train_idx, run.t_lo, k, run.rng)
        for mode in MODES:
            res = {}
            rs = [evaluate(run, sets, U, mode, true["das"]) for U in rand]
            res["random"] = {key: float(np.mean([r[key] for r in rs])) for key in ("corr", "corr_import", "slope", "rms_pred", "rms_noise", "corr_ceiling")}
            res["random"]["profile"] = profile(rand[0])
            if enc is not None:
                res["encode_pca"] = evaluate(run, sets, to_u(enc), mode, true["das"])
                res["encode_pca"]["profile"] = profile(enc)
            for name in ("das", "das_shuffled"):
                Us = train_das(run.model, train, tgt[name], k, mode, K_FROM, steps=das_steps, seed=k)
                r = evaluate(run, sets, Us, mode, true[name])
                with torch.no_grad():
                    fit = swap_response(run.model, fit_set, Us, mode).cpu().numpy()
                r["train_corr"] = swap_stats(fit, tgt[name][:, fit_idx].cpu().numpy(), K_FROM)["corr"]
                r["profile"] = profile(Us)
                saved.update({f"{name}_{mode}_k{k}_{layer}": U.cpu().numpy() for layer, U in zip(run.layers, Us)})
                if name == "das":
                    r["overlap_random"] = overlap(Us, rand[0])
                    if enc is not None:
                        r["overlap_encode_pca"] = overlap(Us, enc)
                    das_us = Us
                else:
                    r["corr_vs_true_params"] = evaluate(run, sets, Us, mode, true["das"])["corr"]
                    r["overlap_das"] = overlap(Us, das_us)
                res[name] = r
            out["subspaces"][str(k)][mode] = res
            log(f"  k={k:2d} {mode:5s}: " + "  ".join(f"{m} {r['corr']:+.3f}/{r['corr_import']:+.3f}" for m, r in res.items())
                + f"  ({time.time() - t0:.0f}s)")
    out["subspaces"] = dict(out["subspaces"])
    out["runtime_s"] = time.time() - t0
    return out, saved


# ---- summary ------------------------------------------------------------------------------------------

METHODS = ("random", "encode_pca", "das", "das_shuffled")


def load_results(paths: list[str], tag: str = "") -> dict:
    groups = defaultdict(list)
    for d in find_runs(paths):
        f = d / f"subspace_swap{tag}.json"
        if f.exists():
            r = json.loads(f.read_text())
            groups[(r["config"]["env"], r["config"]["force_prob"])].append(r)
    return groups


def summarize(groups: dict) -> str:
    """One table per (env, force_prob, mode): rows are (method, score), columns k. Mean over seeds (± std)."""
    rows = [(m, "corr_import") for m in METHODS] + [(m, "corr") for m in METHODS] + \
        [("das", "slope_import"), ("das", "slope"), ("das", "corr_ceiling"), ("das", "train_corr"),
         ("das", "overlap_encode_pca"), ("das_shuffled", "corr_vs_true_params")]
    L = ["# Subspace swaps", "",
         f"Scores on steps k ≥ {K_FROM} after the swap, test trajectories; mean over seeds (± std when > 1 seed). "
         "`corr_import` = paired score (A-only part cancelled), `corr` = plain correlation with f(φ_B) − f(φ_A). "
         "`das_shuffled` is scored against its own shuffled target. `corr_ceiling` = best corr possible on top of "
         "the same-φ noise floor. `overlap_encode_pca` = 1 for the same subspace, k/128 by chance.", ""]
    for (env, fp), runs in sorted(groups.items()):
        seeds = sorted(r["config"]["seed"] for r in runs)
        w = {f: np.mean([r["whole"]["once"][f] for r in runs]) for f in ("corr", "corr_import", "slope")}
        L += [f"## {env}, force_prob={fp}, seeds={seeds}", "",
              f"Whole-memory swap (once): corr_import {w['corr_import']:+.3f}, corr {w['corr']:+.3f}, slope {w['slope']:+.3f}", ""]
        ks = list(runs[0]["subspaces"])
        for mode in MODES:
            L += [f"**{mode}**", "", "| method | score | " + " | ".join(f"k={k}" for k in ks) + " |",
                  "|---|---|" + "---|" * len(ks)]
            for name, field in rows:
                cells = []
                for k in ks:
                    v = [r["subspaces"][k][mode][name][field] for r in runs
                         if name in r["subspaces"][k][mode] and field in r["subspaces"][k][mode][name]]
                    cells.append("—" if not v else f"{np.mean(v):+.3f}" + (f" ± {np.std(v):.3f}" if len(v) > 1 else ""))
                L.append(f"| {name} | {field} | " + " | ".join(cells) + " |")
            L.append("")
    return "\n".join(L)


# categorical slots 1-4 of the reference palette, validated (CVD and normal vision pass); two slots are
# below 3:1 contrast, so every series also gets its own marker and controls are dashed
STYLE = {"das": ("#2a78d6", "o", "-", "DAS (learned)"), "encode_pca": ("#eb6834", "s", "-", "encode-PCA"),
         "random": ("#1baf7a", "^", "--", "random (control)"),
         "untrained": ("#eda100", "D", "--", "DAS on the untrained GRU (control)")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style_axes(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.axhline(0, color=MUTED, lw=0.8)


def plot_summary(groups: dict, untrained: dict, out: Path, fp_focus: float = 1.0, k_focus: str = "8"):
    """Two figures: corr_import vs k at one force_prob (rows env, cols mode), and DAS / encode-PCA corr_import
    vs force_prob at one k. `untrained` holds the --untrained control runs (may be empty)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    envs = sorted({e for e, _ in groups})
    fig, axes = plt.subplots(len(envs), len(MODES), figsize=(4.2 * len(MODES), 3.2 * len(envs)), squeeze=False,
                             sharex=True, sharey=True)
    for i, env in enumerate(envs):
        runs = groups.get((env, fp_focus), [])
        if not runs:
            continue
        ks = list(runs[0]["subspaces"])
        x = np.array([int(k) for k in ks])
        for j, mode in enumerate(MODES):
            ax = axes[i, j]
            _style_axes(ax)
            for m, (c, mk, ls, label) in STYLE.items():
                src, meth = (untrained.get((env, fp_focus), []), "das") if m == "untrained" else (runs, m)
                if not src:
                    continue
                v = np.array([[r["subspaces"][k][mode][meth]["corr_import"]
                               if k in r["subspaces"] and meth in r["subspaces"][k][mode] else np.nan
                               for k in ks] for r in src])
                ok = ~np.isnan(v).all(0)
                mu, sd = np.nanmean(v[:, ok], 0), np.nanstd(v[:, ok], 0)
                ax.plot(x[ok], mu, color=c, marker=mk, ls=ls, lw=2, ms=5.5, label=label)
                if len(src) > 1:
                    ax.fill_between(x[ok], mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
            if mode == "once":
                whole = np.mean([r["whole"]["once"]["corr_import"] for r in runs])
                ax.axhline(whole, color=MUTED, ls=":", lw=1.2)
                ax.annotate("whole memory", (x[-1], whole), textcoords="offset points", xytext=(0, 4),
                            ha="right", fontsize=7.5, color=MUTED)
            ax.set_xscale("log", base=2)
            ax.set_xticks(x, [str(k) for k in x])
            ax.set_ylim(-0.15, 1.0)
            ax.set_title(f"{env} · {mode}", fontsize=10, color=INK, loc="left")
            if i == len(envs) - 1:
                ax.set_xlabel("subspace dimension k (per GRU layer)", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel("corr_import (steps ≥ 1)", fontsize=9, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center", ncol=len(labels))
    fig.suptitle(f"Subspace swaps at force_prob = {fp_focus:g}: does the swapped part carry B's physics?",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "subspace_swap.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(envs), figsize=(4.2 * len(envs), 3.2), squeeze=False, sharey=True)
    for i, env in enumerate(envs):
        ax = axes[0, i]
        _style_axes(ax)
        fps = sorted(fp for e, fp in groups if e == env)
        for m in ("das", "encode_pca"):
            c, mk, _, label = STYLE[m]
            for mode, ls in (("once", "-"), ("clamp", "--")):
                vals = [np.mean([r["subspaces"][k_focus][mode][m]["corr_import"] for r in groups[(env, fp)]
                                 if m in r["subspaces"][k_focus][mode]] or [np.nan]) for fp in fps]
                ax.plot(fps, vals, color=c, marker=mk, ls=ls, lw=2, ms=5.5, label=f"{label}, {mode}")
        whole = [np.mean([r["whole"]["once"]["corr_import"] for r in groups[(env, fp)]]) for fp in fps]
        ax.plot(fps, whole, color=MUTED, ls=":", marker="x", lw=1.2, ms=5, label="whole memory, once")
        ax.set_ylim(-0.15, 1.0)
        ax.set_title(env, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("force_prob", fontsize=9, color=INK)
        if i == 0:
            ax.set_ylabel(f"corr_import at k = {k_focus}", fontsize=9, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out / "subspace_vs_force_prob.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run dirs (default: every runs/*/ with a gru.pt)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--das-steps", type=int, default=1000)
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS), help="subspace dimensions per GRU layer")
    ap.add_argument("--tag", default="", help="suffix for the output files (e.g. a partial k grid)")
    ap.add_argument("--untrained", action="store_true",
                    help="control: run on the GRU at initialization (use with a --tag)")
    ap.add_argument("--summary", action="store_true", help="only collate existing subspace_swap.json files")
    args = ap.parse_args()
    if not args.summary:
        device = pick_device(args.device)
        for run_dir in find_runs(args.runs):
            print(run_dir.name, flush=True)
            res, saved = subspace_swap(load_run(run_dir, device, args.untrained), tuple(args.ks), das_steps=args.das_steps,
                                       log=lambda m: print(m, flush=True))
            (run_dir / f"subspace_swap{args.tag}.json").write_text(json.dumps(res, indent=2))
            np.savez_compressed(run_dir / f"subspace_das{args.tag}.npz", **saved)
    paths = args.runs if args.summary else []
    groups, untrained = load_results(paths), load_results(paths, "_untrained")
    Path("runs/subspace_summary.md").write_text(summarize(groups) + "\n")
    if untrained:
        Path("runs/subspace_summary_untrained.md").write_text(summarize(untrained) + "\n")
    plot_summary(groups, untrained, Path("runs"))
    print("-> runs/subspace_summary.md")


if __name__ == "__main__":
    main()
