"""Figures for a baseline run directory."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODEL_LABELS = {"gru": "GRU world model", "mlp": "memoryless MLP", "oracle": "oracle MLP (given phi)"}


def plot_loss_vs_t(out: Path, res: dict):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, curve in res["prediction"]["per_t"].items():
        ax.plot(curve, label=MODEL_LABELS[name])
    ax.set(yscale="log", xlabel="timestep t (history length)", ylabel="one-step MSE (standardized)",
           title="Prediction error vs. context (ID test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_vs_t.png", dpi=120)
    plt.close(fig)


def plot_emergence(out: Path, res: dict):
    p = res["probe"]
    ts = np.array(p["timesteps"]) + 1
    names = p["targets"]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6), sharey=True)
    for k, (ax, target) in enumerate(zip(axes, names)):
        for layer, r in p["emergence"].items():
            r = np.array(r)[:, k]
            style = dict(ls="--", alpha=0.6) if layer.startswith("untrained/") else \
                dict(ls=":", color="k") if layer == "input" else dict(lw=2)
            ax.plot(ts, r, marker="o", ms=3, label=layer, **style)
        ax.set(xscale="log", xlabel="timestep t+1", title=target, ylim=(-0.2, 1.02))
        ax.axhline(0, color="gray", lw=0.5)
    axes[0].set_ylabel("held-out linear probe R²")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "probe_emergence.png", dpi=120)
    plt.close(fig)


def plot_interchange(out: Path, interchange: dict, params: list[str]):
    """Row per direction kind; first column is the whole-memory swap."""
    kinds = ["decode", "encode"]
    fig, axes = plt.subplots(len(kinds), 1 + len(params), figsize=(3.6 * (1 + len(params)), 3.4 * len(kinds)),
                             squeeze=False)
    for row, kind in enumerate(kinds):
        for col, key in enumerate(["full"] + [f"{kind}/{p}" for p in params]):
            ax = axes[row, col]
            if key == "full" and row > 0:
                ax.axis("off")
                continue
            r = interchange[key]
            ax.scatter(r["_true_v"], r["_pred_v"], s=3, alpha=0.3)
            lim = np.abs(np.concatenate([r["_true_v"], r["_pred_v"]])).max() * 1.05 + 1e-9
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), title=f"{'whole memory' if key == 'full' else key}  corr={r['corr']:.2f}",
                   xlabel="true Δ(next v)", ylabel="model Δ(next v)")
    fig.suptitle("Swap test: model's reaction vs. physics' reaction (diagonal = perfect)")
    fig.tight_layout()
    fig.savefig(out / "interchange.png", dpi=120)
    plt.close(fig)


def plot_swap_horizon(out: Path, res: dict):
    """Correlation with the physical effect at each step after the swap, per param (best direction kind)."""
    ic = res["interchange"]
    params = res["env"]["params"]
    fig, axes = plt.subplots(1, 1 + len(params), figsize=(3.6 * (1 + len(params)), 3.2), sharey=True)
    k = np.arange(len(ic["full"]["corr_by_k"]))
    axes[0].plot(k, ic["full"]["corr_by_k"], marker="o", ms=3)
    axes[0].set(title="whole memory", xlabel="steps after swap", ylabel="corr(model Δ, physics Δ)")
    for ax, prm in zip(axes[1:], params):
        for kind in ("decode", "encode"):
            ax.plot(k, ic[f"{kind}/{prm}"]["corr_by_k"], marker="o", ms=3, label=kind)
        ax.plot(k, ic[f"decode/{prm}"]["rand_corr_by_k"], ls="--", color="gray", label="random |corr|")
        ax.set(title=prm, xlabel="steps after swap")
    for ax in axes:
        ax.axhline(0, color="k", lw=0.5)
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "swap_horizon.png", dpi=120)
    plt.close(fig)
