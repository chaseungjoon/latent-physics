"""End-to-end baseline: simulate -> train world models -> evaluate -> probe -> intervene -> verdict.

    uv run python -m latent_physics.run --env forced --preset quick
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import analysis as A
from .data import Normalizer, make_datasets, oracle_inputs
from .envs import ENVS, make_env
from .models import GRUWorldModel, MLPWorldModel
from .report import write_report
from .train import fit

PRESETS = {
    # CPU smoke test: checks the pipeline end to end in a few minutes; numbers are not meaningful
    "quick": dict(n_train=1000, n_val=200, n_test=400, T=100, hidden=64, epochs=8, mlp_epochs=8,
                  batch_size=64, lr=2e-3),
    # the actual baseline
    "full": dict(n_train=8000, n_val=1000, n_test=2000, T=100, hidden=128, epochs=60, mlp_epochs=30,
                 batch_size=128, lr=2e-3),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", choices=sorted(ENVS), default="forced")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="full")
    ap.add_argument("--force-prob", type=float, default=0.7,
                    help="P(an action segment applies force). 0 makes mass unidentifiable (identifiability knob).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--out", default=None, help="output dir (default runs/<env>_<preset>_fp<force-prob>_s<seed>)")
    for k in ("n_train", "n_test", "hidden", "epochs", "mlp_epochs", "batch_size"):
        ap.add_argument(f"--{k.replace('_', '-')}", type=int, default=None, help="override preset")
    ap.add_argument("--lr", type=float, default=None, help="override preset")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = dict(PRESETS[args.preset])
    for k in cfg:
        if getattr(args, k, None) is not None:
            cfg[k] = getattr(args, k)
    cfg.update(env=args.env, preset=args.preset, force_prob=args.force_prob, seed=args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out or f"runs/{args.env}_{args.preset}_fp{args.force_prob:g}_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "log.txt", "w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"config: {json.dumps(cfg)}  device: {device}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    # ---- data ------------------------------------------------------------------------------------
    env = make_env(args.env)
    sizes = {"train": cfg["n_train"], "val": cfg["n_val"], "id": cfg["n_test"],
             "comp": cfg["n_test"], "extrap": cfg["n_test"]}
    data = make_datasets(env, sizes, cfg["T"], args.force_prob, args.seed)
    norm = Normalizer.fit(data["train"], env)
    assert not env.in_comp_corner(data["train"]["params"]).any()
    log(f"data: {sizes}  ({time.time() - t_start:.1f}s)")

    def tensors(split):
        d = data[split]
        X = norm.inputs(d["states"], d["actions"])
        return {"X": X, "Y": norm.targets(d["states"]), "Xo": oracle_inputs(X, norm.params(d["params"], env))}

    np_t = {s: tensors(s) for s in sizes}
    T_ = {s: {k: torch.as_tensor(v, device=device) for k, v in d.items()} for s, d in np_t.items()}
    S, D = len(env.state_names), np_t["train"]["X"].shape[-1]

    # ---- models ----------------------------------------------------------------------------------
    gru = GRUWorldModel(D, S, hidden=cfg["hidden"]).to(device)
    gru_untrained = copy.deepcopy(gru).eval()  # same init, never trained: probing control
    models = {"gru": (gru, "X", cfg["epochs"]),
              "mlp": (MLPWorldModel(D, S).to(device), "X", cfg["mlp_epochs"]),
              "oracle": (MLPWorldModel(np_t["train"]["Xo"].shape[-1], S).to(device), "Xo", cfg["mlp_epochs"])}
    history = {}
    for name, (model, key, epochs) in models.items():
        history[name] = fit(model, T_["train"][key], T_["train"]["Y"], T_["val"][key], T_["val"]["Y"],
                            epochs=epochs, batch_size=cfg["batch_size"], lr=cfg["lr"], name=name, log=log)
        torch.save(model.state_dict(), out / f"{name}.pt")

    # ---- prediction & generalization ---------------------------------------------------------------
    T = cfg["T"]
    t_lo = T // 5  # "late" timesteps: enough history to have seen the hidden params act
    t0, horizon = T // 2, T // 4
    pred = {"per_t": {}, "late_mse": {}, "early_mse": {}, "rollout": {}}
    for name, (model, key, _) in models.items():
        for split in ("id", "comp", "extrap"):
            curve = A.per_t_mse(model, T_[split][key], T_[split]["Y"])
            pred["late_mse"].setdefault(name, {})[split] = float(curve[t_lo:].mean())
            pred["early_mse"].setdefault(name, {})[split] = float(curve[:5].mean())
            if split == "id":
                pred["per_t"][name] = curve.tolist()
            d = data[split]
            extra = norm.params(d["params"], env) if name == "oracle" else None
            pred["rollout"].setdefault(name, {})[split] = A.rollout_mse(
                model, norm, d["states"], d["actions"], extra, t0, horizon, device).tolist()
    log(f"late one-step MSE: {json.dumps(pred['late_mse'])}")

    # ---- probing -------------------------------------------------------------------------------------
    perm = rng.permutation(cfg["n_test"])
    train_idx, test_idx = np.sort(perm[: int(0.7 * len(perm))]), np.sort(perm[int(0.7 * len(perm)):])
    hid = A.hidden_states(gru, T_["id"]["X"])
    hid_untrained = A.hidden_states(gru_untrained, T_["id"]["X"])
    targets, target_names = A.probe_targets(env, data["id"])
    timesteps = sorted({t for t in (0, 1, 2, 4, 8, 16, 32, 64, T - 1) if t < T})
    layers = {"input": np_t["id"]["X"], **hid, **{f"untrained/{k}": v for k, v in hid_untrained.items()}}
    emergence = A.probe_emergence(layers, targets, timesteps, train_idx, test_idx, rng)
    top = gru.gru_names[-1]
    shuffled = A.shuffled_label_r2(hid[top], targets, T - 1, train_idx, test_idx, rng)
    log("probe R^2 at last t: " + ", ".join(
        f"{layer}={np.round(r[-1], 3).tolist()}" for layer, r in emergence.items()))
    z = {s: env.to_probe(data[s]["params"]) for s in sizes}
    nonlinear = {layer: A.mlp_probe_r2(hid[layer][:, T - 1], z["id"], train_idx, test_idx, device).tolist()
                 for layer in gru.gru_names}
    log(f"MLP-probe R^2 at last t: {nonlinear}")

    # pooled late-t probes: used for OOD readout, interventions and ablations
    probes = A.fit_pooled_probes(hid, z["id"], train_idx, t_lo, rng)
    probe_ood = {}
    for split in ("id", "comp", "extrap"):
        h = hid if split == "id" else A.hidden_states(gru, T_[split]["X"])
        idx = test_idx if split == "id" else np.arange(cfg["n_test"])
        for layer, probe in probes.items():
            Xp, Yp, _ = A.pooled(h[layer], z[split], idx, t_lo)
            probe_ood.setdefault(layer, {})[split] = np.sqrt(((probe.predict(Xp) - Yp) ** 2).mean(0)).tolist()

    # ---- causal tests --------------------------------------------------------------------------------
    z_std = z["id"][train_idx].std(0)
    interventions = A.intervention_test(gru, env, norm, data["id"], hid, probes, z_std, test_idx, t_lo, rng, device)
    ablations = A.ablation_test(gru, env, T_["id"]["X"], T_["id"]["Y"], hid, probes, train_idx, test_idx,
                                t_lo, rng, device)
    for key, r in interventions.items():
        log(f"intervention {key}: corr {r['corr']:.3f} slope {r['slope']:.3f} rand {r['rand_corr']:.3f}  "
            f"| ablation +{ablations[key]['rel_increase'] * 100:.1f}% (rand +{ablations[key]['rand_rel_increase'] * 100:.1f}%)")

    # ---- verdict & report -----------------------------------------------------------------------------
    results = {
        "config": cfg, "device": str(device), "env": {"params": [p.name for p in env.params],
                                                      "extrap_param": env.params[env.extrap_param].name,
                                                      "comp_params": [env.params[i].name for i in env.comp_params]},
        "history": history, "prediction": pred,
        "probe": {"timesteps": timesteps, "targets": target_names,
                  "emergence": {k: v.tolist() for k, v in emergence.items()},
                  "shuffled_label_r2_top_last_t": shuffled.tolist(), "mlp_probe_r2_last_t": nonlinear,
                  "pooled_rmse": probe_ood},
        "interventions": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in interventions.items()},
        "ablations": ablations,
        "runtime_s": time.time() - t_start,
    }
    results["verdict"] = verdict(results, env, gru.gru_names)
    write_report(out, results, interventions)
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    log("\n" + "\n".join(f"[{c['status']}] {c['check']}: {c['detail']}" for c in results["verdict"]))
    log(f"\ndone in {results['runtime_s'] / 60:.1f} min -> {out}/report.md")


def verdict(res: dict, env, gru_names: list[str]) -> list[dict]:
    """Heuristic viability checks. Thresholds are rules of thumb, not statistical tests."""
    checks = []
    late = res["prediction"]["late_mse"]
    ratio = late["gru"]["id"] / late["mlp"]["id"]
    gap = late["gru"]["id"] / late["oracle"]["id"]
    checks.append({"check": "Hidden physics matters & is inferred in-context",
                   "status": "PASS" if ratio < 0.5 else "FAIL",
                   "detail": f"GRU/memoryless-MLP late MSE = {ratio:.3f} (want < 0.5); GRU/oracle = {gap:.2f} (1 = matches privileged model)"})
    em, names = res["probe"]["emergence"], res["probe"]["targets"]
    trained = [k for k in em if k != "input" and not k.startswith("untrained/")]
    untrained = [k for k in em if k.startswith("untrained/")] + ["input"]
    for k, prm in enumerate(env.params):
        best = max(trained, key=lambda l: em[l][-1][k])
        r_best = em[best][-1][k]
        r_ctrl = max(em[l][-1][k] for l in untrained)
        r_first = em[best][0][k]
        checks.append({"check": f"{prm.name}: linearly decodable (Level 2)",
                       "status": "PASS" if r_best > 0.5 and r_best - r_ctrl > 0.2 else "FAIL",
                       "detail": f"best layer {best} R2={r_best:.3f} vs best control R2={r_ctrl:.3f}"})
        checks.append({"check": f"{prm.name}: emerges as evidence accumulates",
                       "status": "PASS" if r_best - r_first > 0.2 else "FAIL",
                       "detail": f"{best} R2 t=0: {r_first:.3f} -> t=T-1: {r_best:.3f}"})
        iv = [res["interventions"][f"{l}/{prm.name}"] for l in gru_names]
        b = max(iv, key=lambda r: r["corr"])
        checks.append({"check": f"{prm.name}: probe direction is causally used (Level 4)",
                       "status": "PASS" if b["corr"] > 0.3 and b["corr"] > b["rand_corr"] + 0.2 else "FAIL",
                       "detail": f"{b['layer']}: corr(pred, true effect)={b['corr']:.3f}, slope={b['slope']:.3f}, "
                                 f"random-direction |corr|={b['rand_corr']:.3f}"})
    return checks


if __name__ == "__main__":
    main()
