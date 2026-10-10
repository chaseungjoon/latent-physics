"""End-to-end baseline: simulate -> train world models -> evaluate -> probe -> intervene -> verdict.

    uv run python -m latent_physics.experiments.baseline --env forced --preset quick

The stages share one seeded numpy generator; they must run in this order for a run to be reproducible.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from ...analysis import ablation, directions, interchange, prediction, probes, sensitivity
from ...data import Normalizer, make_datasets, oracle_inputs
from ...envs import ENVS, make_env
from ...models import GRUWorldModel, MLPWorldModel, fit
from ..loading import pick_device, probe_split
from .report import write_report
from .verdict import verdict

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
    ap.add_argument("--reuse-models", action="store_true",
                    help="load gru/mlp/oracle .pt from the output dir instead of training when they exist "
                         "(re-runs only the analysis; data is regenerated identically from the seed)")
    return ap.parse_args()


def build_data(env, cfg: dict, device) -> tuple[dict, Normalizer, dict, dict]:
    """Simulate all splits. Returns raw data, normalizer, numpy tensors and torch tensors per split
    (X = model inputs, Y = standardized deltas, Xo = oracle inputs with the true params appended)."""
    sizes = {"train": cfg["n_train"], "val": cfg["n_val"], "id": cfg["n_test"],
             "comp": cfg["n_test"], "extrap": cfg["n_test"]}
    data = make_datasets(env, sizes, cfg["T"], cfg["force_prob"], cfg["seed"])
    norm = Normalizer.fit(data["train"], env)
    assert not env.in_comp_corner(data["train"]["params"]).any()

    def tensors(split):
        d = data[split]
        X = norm.inputs(d["states"], d["actions"])
        return {"X": X, "Y": norm.targets(d["states"]), "Xo": oracle_inputs(X, norm.params(d["params"], env))}

    np_t = {s: tensors(s) for s in sizes}
    T_ = {s: {k: torch.as_tensor(v, device=device) for k, v in d.items()} for s, d in np_t.items()}
    return data, norm, np_t, T_


def train_models(env, cfg: dict, np_t: dict, T_: dict, out: Path, reuse: bool, device, log):
    """Train (or load) the GRU, the memoryless MLP and the oracle; also keep the GRU at initialization."""
    S, D = len(env.state_names), np_t["train"]["X"].shape[-1]
    gru = GRUWorldModel(D, S, hidden=cfg["hidden"]).to(device)
    gru_untrained = copy.deepcopy(gru).eval()  # same init, never trained: probing control
    models = {"gru": (gru, "X", cfg["epochs"]),
              "mlp": (MLPWorldModel(D, S).to(device), "X", cfg["mlp_epochs"]),
              "oracle": (MLPWorldModel(np_t["train"]["Xo"].shape[-1], S).to(device), "Xo", cfg["mlp_epochs"])}
    history, old_history = {}, {}
    if reuse and (out / "metrics.json").exists():
        old_history = json.loads((out / "metrics.json").read_text()).get("history", {})
    for name, (model, key, epochs) in models.items():
        ckpt = out / f"{name}.pt"
        if reuse and ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            history[name] = old_history.get(name, [])
            log(f"[{name}] loaded {ckpt} (training skipped)")
            continue
        history[name] = fit(model, T_["train"][key], T_["train"]["Y"], T_["val"][key], T_["val"]["Y"],
                            epochs=epochs, batch_size=cfg["batch_size"], lr=cfg["lr"], name=name, log=log)
        torch.save(model.state_dict(), ckpt)
    return models, gru_untrained, history


def evaluate_prediction(env, models: dict, data: dict, norm: Normalizer, T_: dict, T: int, device) -> dict:
    """Level 1 (and 5/6): late one-step MSE and open-loop rollouts on id / comp / extrap."""
    t_lo, t0, horizon = T // 5, T // 2, T // 4
    pred = {"per_t": {}, "late_mse": {}, "early_mse": {}, "rollout": {}}
    for name, (model, key, _) in models.items():
        for split in ("id", "comp", "extrap"):
            curve = prediction.per_t_mse(model, T_[split][key], T_[split]["Y"])
            pred["late_mse"].setdefault(name, {})[split] = float(curve[t_lo:].mean())
            pred["early_mse"].setdefault(name, {})[split] = float(curve[:5].mean())
            if split == "id":
                pred["per_t"][name] = curve.tolist()
            d = data[split]
            extra = norm.params(d["params"], env) if name == "oracle" else None
            pred["rollout"].setdefault(name, {})[split] = prediction.rollout_mse(
                model, norm, d["states"], d["actions"], extra, t0, horizon, device).tolist()
    return pred


def run_probes(env, cfg: dict, gru, gru_untrained, data: dict, np_t: dict, T_: dict, rng, device, log) -> dict:
    """Level 2: emergence curves, controls, nonlinear probe, R^2 when it matters, pooled probes and OOD readout."""
    T, n = cfg["T"], cfg["n_test"]
    t_lo = T // 5
    train_idx, test_idx = probe_split(n, rng)
    hid = probes.hidden_states(gru, T_["id"]["X"])
    hid_untrained = probes.hidden_states(gru_untrained, T_["id"]["X"])
    targets, target_names = probes.probe_targets(env, data["id"])
    timesteps = sorted({t for t in (0, 1, 2, 4, 8, 16, 32, 64, T - 1) if t < T})
    layers = {"input": np_t["id"]["X"], **hid, **{f"untrained/{k}": v for k, v in hid_untrained.items()}}
    emergence = probes.probe_emergence(layers, targets, timesteps, train_idx, test_idx, rng)
    shuffled = probes.shuffled_label_r2(hid[gru.gru_names[-1]], targets, T - 1, train_idx, test_idx, rng)
    log("probe R^2 at last t: " + ", ".join(
        f"{layer}={np.round(r[-1], 3).tolist()}" for layer, r in emergence.items()))
    z = {s: env.to_probe(data[s]["params"]) for s in ("id", "comp", "extrap")}
    nonlinear = {layer: probes.mlp_probe_r2(hid[layer][:, T - 1], z["id"], train_idx, test_idx, device).tolist()
                 for layer in gru.gru_names}
    log(f"MLP-probe R^2 at last t: {nonlinear}")
    # R^2 only at the steps where each param physically matters for the next step
    z_std = z["id"][train_idx].std(0)
    sens = sensitivity.param_sensitivity(env, data["id"], np.arange(n), z_std, t_lo)
    relevant, relevant_frac = probes.relevant_r2(layers, z["id"], sens, train_idx, test_idx, t_lo, rng)
    log(f"probe R^2 when relevant (relevant fraction {np.round(relevant_frac, 3).tolist()}): "
        + ", ".join(f"{layer}={[None if x is None else round(x, 3) for x in r]}" for layer, r in relevant.items()))

    # pooled late-t probes: used for OOD readout, swap tests and ablations
    pooled_probes = probes.fit_pooled_probes(hid, z["id"], train_idx, t_lo, rng)
    probe_ood = {}
    for split in ("id", "comp", "extrap"):
        h = hid if split == "id" else probes.hidden_states(gru, T_[split]["X"])
        idx = test_idx if split == "id" else np.arange(n)
        for layer, probe in pooled_probes.items():
            Xp, Yp, _ = probes.pooled(h[layer], z[split], idx, t_lo)
            probe_ood.setdefault(layer, {})[split] = np.sqrt(((probe.predict(Xp) - Yp) ** 2).mean(0)).tolist()
    results = {"timesteps": timesteps, "targets": target_names,
               "emergence": {k: v.tolist() for k, v in emergence.items()},
               "shuffled_label_r2_top_last_t": shuffled.tolist(), "mlp_probe_r2_last_t": nonlinear,
               "relevant_r2": relevant, "relevant_frac": relevant_frac, "pooled_rmse": probe_ood}
    state = {"hid": hid, "z_id": z["id"], "train_idx": train_idx, "test_idx": test_idx, "probes": pooled_probes}
    return results, state


def run_causal(env, gru, data: dict, norm: Normalizer, np_t: dict, T_: dict, st: dict, t_lo: int, rng, device, log):
    """Level 4: 1-D swap test and mean-ablation along decode / encode directions."""
    hid, train_idx, test_idx = st["hid"], st["train_idx"], st["test_idx"]
    dirs = directions.param_directions(hid, st["z_id"], np_t["id"]["X"], st["probes"], gru.gru_names, train_idx, t_lo)
    rand_dirs = directions.natural_random_directions(hid, gru.gru_names, train_idx, t_lo, 5, rng)
    ic = interchange.interchange_test(gru, env, norm, data["id"], hid, dirs, rand_dirs, test_idx, t_lo, rng, device)
    ab = ablation.ablation_test(gru, env, data["id"], T_["id"]["X"], T_["id"]["Y"], hid, dirs, rand_dirs[:3],
                                train_idx, test_idx, t_lo, device)
    log(f"swap whole memory: corr {ic['full']['corr']:.3f} (k=0: {ic['full']['corr_k0']:.3f}) "
        f"slope {ic['full']['slope']:.3f}")
    for key, r in ic.items():
        if key != "full":
            a = ab[key]
            log(f"swap {key}: corr {r['corr']:.3f} (k=0: {r['corr_k0']:.3f}) slope {r['slope']:.3f} rand {r['rand_corr']:.3f} "
                f"other {r['corr_other']:.3f} | ablation {a['rel_increase'] * 100:+.1f}% "
                f"(var {a['var_frac']:.3f}, sens_corr {a['sens_corr']:.3f}; random {a['random']['rel_increase'] * 100:+.1f}%, "
                f"var {a['random']['var_frac']:.3f}, sens_corr {a['random']['sens_corr']:.3f})")
    return ic, ab


def main():
    args = parse_args()
    cfg = dict(PRESETS[args.preset])
    for k in cfg:
        if getattr(args, k, None) is not None:
            cfg[k] = getattr(args, k)
    cfg.update(env=args.env, preset=args.preset, force_prob=args.force_prob, seed=args.seed)
    device = pick_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out or f"runs/{args.env}_{args.preset}_fp{args.force_prob:g}_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "log.txt", "a" if args.reuse_models else "w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"config: {json.dumps(cfg)}  device: {device}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    env = make_env(args.env)
    data, norm, np_t, T_ = build_data(env, cfg, device)
    log(f"data: { {s: len(d['params']) for s, d in data.items()} }  ({time.time() - t_start:.1f}s)")
    models, gru_untrained, history = train_models(env, cfg, np_t, T_, out, args.reuse_models, device, log)
    gru = models["gru"][0]

    pred = evaluate_prediction(env, models, data, norm, T_, cfg["T"], device)
    log(f"late one-step MSE: {json.dumps(pred['late_mse'])}")
    probe_res, st = run_probes(env, cfg, gru, gru_untrained, data, np_t, T_, rng, device, log)
    ic, ab = run_causal(env, gru, data, norm, np_t, T_, st, cfg["T"] // 5, rng, device, log)

    results = {
        "config": cfg, "device": str(device), "env": {"params": [p.name for p in env.params],
                                                      "extrap_param": env.params[env.extrap_param].name,
                                                      "comp_params": [env.params[i].name for i in env.comp_params]},
        "history": history, "prediction": pred, "probe": probe_res,
        "interchange": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in ic.items()},
        "ablations": ab,
        "runtime_s": time.time() - t_start,
    }
    results["verdict"] = verdict(results, env)
    write_report(out, results, ic)
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    log("\n" + "\n".join(f"[{c['status']}] {c['check']}: {c['detail']}" for c in results["verdict"]))
    log(f"\ndone in {results['runtime_s'] / 60:.1f} min -> {out}/report.md")
