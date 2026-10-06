"""Markdown report + figures for a run directory."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODEL_LABELS = {"gru": "GRU world model", "mlp": "memoryless MLP", "oracle": "oracle MLP (given phi)"}


def _plot_loss_vs_t(out: Path, res: dict):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, curve in res["prediction"]["per_t"].items():
        ax.plot(curve, label=MODEL_LABELS[name])
    ax.set(yscale="log", xlabel="timestep t (history length)", ylabel="one-step MSE (standardized)",
           title="Prediction error vs. context (ID test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_vs_t.png", dpi=120)
    plt.close(fig)


def _plot_emergence(out: Path, res: dict):
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


def _plot_interchange(out: Path, interchange: dict, params: list[str]):
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


def _plot_swap_horizon(out: Path, res: dict):
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


def write_report(out: Path, res: dict, interchange: dict):
    _plot_loss_vs_t(out, res)
    _plot_emergence(out, res)
    _plot_interchange(out, interchange, res["env"]["params"])
    _plot_swap_horizon(out, res)

    cfg, pred, probe = res["config"], res["prediction"], res["probe"]
    L = [f"# Baseline report — `{cfg['env']}` ({cfg['preset']}, force_prob={cfg['force_prob']}, seed={cfg['seed']})", ""]
    L += ["## Verdict", "", "| status | check | detail |", "|---|---|---|"]
    L += [f"| **{c['status']}** | {c['check']} | {c['detail']} |" for c in res["verdict"]]
    L += ["", f"Hidden params: {', '.join(res['env']['params'])}. Compositional holdout: high-high corner of "
          f"{' × '.join(res['env']['comp_params'])}. Extrapolation: {res['env']['extrap_param']} beyond training range.", ""]

    L += ["## Level 1 — prediction & generalization", "",
          "One-step MSE (standardized deltas), averaged over late timesteps; rollout = standardized state MSE "
          f"after {len(next(iter(pred['rollout'].values()))['id'])} open-loop steps.", "",
          "| model | ID | comp-OOD | extrap-OOD | rollout ID | rollout comp | rollout extrap |", "|---|---|---|---|---|---|---|"]
    for name in pred["late_mse"]:
        m, ro = pred["late_mse"][name], pred["rollout"][name]
        L.append(f"| {MODEL_LABELS[name]} | {m['id']:.5f} | {m['comp']:.5f} | {m['extrap']:.5f} | "
                 f"{ro['id'][-1]:.4f} | {ro['comp'][-1]:.4f} | {ro['extrap'][-1]:.4f} |")
    L += ["", "![loss vs t](loss_vs_t.png)", ""]

    L += ["## Level 2 — linear decodability", "",
          "Separate ridge probe per (layer, timestep), split by trajectory. `input` = raw (s_t, a_t); "
          "`untrained/*` = same architecture at initialization.", "",
          "| layer | " + " | ".join(probe["targets"]) + " |", "|---|" + "---|" * len(probe["targets"])]
    for layer, r in probe["emergence"].items():
        L.append(f"| {layer} | " + " | ".join(f"{x:.3f}" for x in r[-1]) + " |")
    L.append("| shuffled-label control (top layer) | " + " | ".join(f"{x:.3f}" for x in probe["shuffled_label_r2_top_last_t"]) + " |")
    params = res["env"]["params"]
    L += ["", "**R² when it matters**: pooled over t ≥ T/5 but only on steps where the param physically affects the "
          "next step (simulator sensitivity > 10% of its mean). `—` = the param (almost) never matters here (not identifiable).", "",
          "| layer | " + " | ".join(params) + " |", "|---|" + "---|" * len(params),
          "| (share of steps where it matters) | " + " | ".join(f"{f * 100:.0f}%" for f in probe["relevant_frac"]) + " |"]
    L += [f"| {layer} | " + " | ".join("—" if x is None else f"{x:.3f}" for x in r) + " |" for layer, r in probe["relevant_r2"].items()]
    L += ["", "Nonlinear (MLP) probe at the last timestep — present but not linearly readable?", "",
          "| layer | " + " | ".join(res["env"]["params"]) + " |", "|---|" + "---|" * len(res["env"]["params"])]
    L += [f"| {layer} | " + " | ".join(f"{x:.3f}" for x in r) + " |" for layer, r in probe["mlp_probe_r2_last_t"].items()]
    L += ["", "Linear R² at the last timestep is in the first table. Emergence over time:", "", "![probe emergence](probe_emergence.png)", ""]
    L += ["Pooled late-t probe RMSE (probe space) — does the readout transfer to OOD parameters?", "",
          "| layer | split | " + " | ".join(res["env"]["params"]) + " |", "|---|---|" + "---|" * len(res["env"]["params"])]
    for layer, d in probe["pooled_rmse"].items():
        for split, v in d.items():
            L.append(f"| {layer} | {split} | " + " | ".join(f"{x:.3f}" for x in v) + " |")

    ic, ab = res["interchange"], res["ablations"]
    L += ["", "## Level 4 — causal use", "",
          "**Swap test.** For trajectory A at time t, copy from another trajectory B only the component of the GRU "
          "memory along a param's direction (all GRU layers), and compare the change in A's predicted next state with "
          "the simulator's change when A gets B's value of that param. `decode` = probe readout direction, "
          "`encode` = direction the memory actually moves with the param. `corr` 1 = reacts exactly like physics; "
          "`slope` 1 = fully adopts B's value; `other` = strongest |corr| with swapping a *different* param; "
          "`random` = same swap along random natural directions. After the swap the model keeps reading A's real "
          f"observations for {len(ic['full']['corr_by_k'])} steps; `corr` pools those steps, `k=0` is the immediate step only.", "",
          f"Whole-memory swap: corr {ic['full']['corr']:.3f} (k=0: {ic['full']['corr_k0']:.3f}), slope {ic['full']['slope']:.3f}.", "",
          "| direction/param | corr | k=0 | slope | other | random | ablation Δloss | var frac | sens corr | random: Δloss / var frac / sens corr |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in ic.items():
        if key == "full":
            continue
        a, rd = ab[key], ab[key]["random"]
        L.append(f"| {key} | {r['corr']:.3f} | {r['corr_k0']:.3f} | {r['slope']:.3f} | {r['corr_other']:.3f} | {r['rand_corr']:.3f} "
                 f"| {a['rel_increase'] * 100:+.1f}% | {a['var_frac']:.3f} | {a['sens_corr']:.3f} "
                 f"| {rd['rel_increase'] * 100:+.1f}% / {rd['var_frac']:.3f} / {rd['sens_corr']:.3f} |")
    L += ["", "**Ablation** (right columns): mean-ablate the direction in every GRU layer at every step. "
          "`var frac` = share of memory variance removed (compare damage only at similar var frac); "
          "`sens corr` = does the extra error land on the steps where this param physically matters?", "",
          "![swap test](interchange.png)", "", "![swap over time](swap_horizon.png)", "",
          f"_Runtime {res['runtime_s'] / 60:.1f} min on {res['device']}._", ""]
    (out / "report.md").write_text("\n".join(L))
