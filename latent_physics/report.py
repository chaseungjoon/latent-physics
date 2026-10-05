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


def _plot_interventions(out: Path, interventions: dict, layer: str):
    keys = [k for k in interventions if k.startswith(layer + "/")]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.8))
    for ax, key in zip(np.atleast_1d(axes), keys):
        r = interventions[key]
        ax.scatter(r["_true_v"], r["_pred_v"], s=3, alpha=0.3)
        lim = np.abs(np.concatenate([r["_true_v"], r["_pred_v"]])).max() * 1.05 + 1e-9
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="true Δ(next v) from changing param",
               ylabel="model Δ(next v) from latent shift", title=f"{key}  corr={r['corr']:.2f}")
    fig.tight_layout()
    fig.savefig(out / "interventions.png", dpi=120)
    plt.close(fig)


def write_report(out: Path, res: dict, interventions: dict):
    _plot_loss_vs_t(out, res)
    _plot_emergence(out, res)
    top = max({v["layer"] for v in res["interventions"].values()})
    _plot_interventions(out, interventions, top)

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
    L += ["", "Nonlinear (MLP) probe at the last timestep — present but not linearly readable?", "",
          "| layer | " + " | ".join(res["env"]["params"]) + " |", "|---|" + "---|" * len(res["env"]["params"])]
    L += [f"| {layer} | " + " | ".join(f"{x:.3f}" for x in r) + " |" for layer, r in probe["mlp_probe_r2_last_t"].items()]
    L += ["", "Linear R² at the last timestep is in the first table. Emergence over time:", "", "![probe emergence](probe_emergence.png)", ""]
    L += ["Pooled late-t probe RMSE (probe space) — does the readout transfer to OOD parameters?", "",
          "| layer | split | " + " | ".join(res["env"]["params"]) + " |", "|---|---|" + "---|" * len(res["env"]["params"])]
    for layer, d in probe["pooled_rmse"].items():
        for split, v in d.items():
            L.append(f"| {layer} | {split} | " + " | ".join(f"{x:.3f}" for x in v) + " |")

    L += ["", "## Level 4 — causal use", "",
          "Intervention: shift the GRU state along a param's probe direction (±0.5 std in probe units) and correlate "
          "the change in predicted next-state with the simulator's change when that param is truly moved. "
          "Ablation: mean-ablate that 1-D direction at every step; relative increase in late one-step loss.", "",
          "| layer/param | corr | slope | random-dir |corr| | shift / state spread | ablation Δloss | random ablation Δloss |",
          "|---|---|---|---|---|---|---|"]
    for key, r in res["interventions"].items():
        a = res["ablations"][key]
        L.append(f"| {key} | {r['corr']:.3f} | {r['slope']:.3f} | {r['rand_corr']:.3f} | {r['shift_norm_over_spread']:.3f} "
                 f"| {a['rel_increase'] * 100:+.1f}% | {a['rand_rel_increase'] * 100:+.1f}% |")
    L += ["", f"![interventions]({'interventions.png'})", "",
          f"_Runtime {res['runtime_s'] / 60:.1f} min on {res['device']}._", ""]
    (out / "report.md").write_text("\n".join(L))
