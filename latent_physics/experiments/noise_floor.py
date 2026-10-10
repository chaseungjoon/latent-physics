"""Noise floor of the interchange (swap) test.

The swap test copies the memory of a partner B, matched to A on the last few inputs, into A and reads
the change of A's predictions as the model's response to B's physics belief. But the two memories also
differ because the matched histories are not identical, and the model reacts to that too. This script
measures that part on its own: partners are fresh trajectories simulated with *exactly A's parameters*,
matched the same way from an equally large pool. Their true effect is zero, so whatever the swap does
to the prediction is noise.

    uv run python -m latent_physics.experiments.noise_floor                     # every run under runs/
    uv run python -m latent_physics.experiments.noise_floor runs/spring_full_fp0.7_s0
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ..analysis import directions, probes
from ..analysis.partners import history, match_partners, same_phi_partners
from ..analysis.swaps import make_swap_set, rms, rollout, swap_response, swap_stats
from .loading import LoadedRun, find_runs, load_run, pick_device

HORIZON, MATCH_STEPS = 10, 2


@torch.no_grad()
def noise_floor(run: LoadedRun) -> dict:
    env, layers, T = run.env, run.layers, run.cfg["T"]
    z = env.to_probe(run.data["params"])
    pooled_probes = probes.fit_pooled_probes({k: run.hid[k] for k in layers}, z, run.train_idx, run.t_lo, run.rng)
    dirs = directions.param_directions(run.hid, z, run.X, pooled_probes, layers, run.train_idx, run.t_lo)

    lo = max(run.t_lo, MATCH_STEPS)
    na = len(run.test_idx)
    ia = np.arange(na)
    ts = run.rng.integers(lo, T - HORIZON + 1, size=na)  # one swap point per test trajectory
    Xt = run.X[run.test_idx]
    ib, tb, dist_std = match_partners(Xt, ia, ts, lo, run.device, MATCH_STEPS)
    a_rows, b_rows = run.test_idx, run.test_idx[ib]
    cB_same, dist_same = same_phi_partners(
        run.model, env, run.norm, run.data["params"][a_rows], history(Xt, ia, ts, MATCH_STEPS), T, lo,
        run.cfg["force_prob"], np.random.default_rng([run.cfg["seed"], 1234]), na - 1, run.device, MATCH_STEPS)

    std = make_swap_set(env, run.norm, run.data, run.hid, layers, a_rows, ts, b_rows, tb, HORIZON, run.device)
    same = make_swap_set(env, run.norm, run.data, run.hid, layers, a_rows, ts, b_rows, tb, HORIZON, run.device,
                         cB=cB_same)
    base = rollout(run.model, std.xs, std.cA)

    def response(s, us):
        Us = None if us is None else [torch.as_tensor(u, dtype=torch.float32, device=run.device)[:, None] for u in us]
        return swap_response(run.model, s, Us, "once", base).cpu().numpy()

    # same-phi partners match a bit closer than standard ones (similar dynamics); the noise of the ones
    # matched no better than a typical standard partner checks that this does not hide the noise
    far = dist_same >= np.median(dist_std)
    rms_k = lambda x: np.sqrt((x ** 2).mean((1, 2))).tolist()  # noqa: E731

    def summarize(pred_std, pred_same, true):
        st = swap_stats(pred_std, true)
        sig, noise = rms(true), rms(pred_same)
        return {
            "corr": st["corr"], "corr_k0": st["corr_k0"], "slope": st["slope"],
            "rms_true": sig, "rms_pred": rms(pred_std), "rms_noise": noise,
            "rms_noise_far": rms(pred_same[:, far]), "n_far": int(far.sum()),
            "noise_over_true": noise / (sig + 1e-12), "noise_over_pred": noise / (rms(pred_std) + 1e-12),
            # best corr a model that fully adopts B's belief (slope 1) could show on top of this noise
            "corr_ceiling": sig / np.sqrt(sig ** 2 + noise ** 2),
            "rms_true_by_k": rms_k(true), "rms_pred_by_k": rms_k(pred_std), "rms_noise_by_k": rms_k(pred_same),
        }

    out = {"config": run.cfg, "match_dist": {"standard_median": float(np.median(dist_std)),
                                             "same_phi_median": float(np.median(dist_same))},
           "full": summarize(response(std, None), response(same, None), std.true_effect(list(range(len(env.params)))))}
    for k, prm in enumerate(env.params):
        true = std.true_effect([k])
        for kind, d in dirs.items():
            us = [d[name][k] for name in layers]
            out[f"{kind}/{prm.name}"] = summarize(response(std, us), response(same, us), true)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run dirs (default: every runs/*/ with a gru.pt)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)
    for run_dir in find_runs(args.runs):
        res = noise_floor(load_run(run_dir, device))
        (run_dir / "noise_floor.json").write_text(json.dumps(res, indent=2))
        f = res["full"]
        print(f"{run_dir.name}: whole corr {f['corr']:.3f} ceiling {f['corr_ceiling']:.3f} "
              f"noise/true {f['noise_over_true']:.2f} noise/pred {f['noise_over_pred']:.2f} "
              f"match {res['match_dist']['standard_median']:.3f}/{res['match_dist']['same_phi_median']:.3f}", flush=True)


if __name__ == "__main__":
    main()
