"""Heuristic PASS/FAIL checks for a baseline run."""
from __future__ import annotations


def verdict(res: dict, env) -> list[dict]:
    """Heuristic viability checks. Thresholds are rules of thumb, not statistical tests."""
    checks = []
    late = res["prediction"]["late_mse"]
    ratio = late["gru"]["id"] / late["mlp"]["id"]
    gap = late["gru"]["id"] / late["oracle"]["id"]
    checks.append({"check": "Hidden physics matters & is inferred in-context",
                   "status": "PASS" if ratio < 0.5 else "FAIL",
                   "detail": f"GRU/memoryless-MLP late MSE = {ratio:.3f} (want < 0.5); GRU/oracle = {gap:.2f} (1 = matches privileged model)"})
    full = res["interchange"]["full"]
    checks.append({"check": "Memory is used as the physics belief (Level 4, exploratory: whole-memory swap)",
                   "status": "PASS" if full["corr"] > 0.5 else "FAIL",
                   "detail": f"swap memory with another trajectory: corr(pred, true effect)={full['corr']:.3f}, slope={full['slope']:.2f}"})
    em, rel = res["probe"]["emergence"], res["probe"]["relevant_r2"]
    trained = [k for k in em if k != "input" and not k.startswith("untrained/")]
    untrained = [k for k in em if k.startswith("untrained/")] + ["input"]
    for k, prm in enumerate(env.params):
        if rel[trained[0]][k] is None:
            checks.append({"check": f"{prm.name}: linearly decodable when it matters (Level 2)", "status": "N/A",
                           "detail": f"{prm.name} (almost) never affects the next step in this data: not identifiable"})
        else:
            best = max(trained, key=lambda l: rel[l][k])
            r_best, r_ctrl = rel[best][k], max(rel[l][k] for l in untrained)
            checks.append({"check": f"{prm.name}: linearly decodable when it matters (Level 2)",
                           "status": "PASS" if r_best > 0.5 and r_best - r_ctrl > 0.2 else "FAIL",
                           "detail": f"best layer {best} R2={r_best:.3f} vs best control R2={r_ctrl:.3f} "
                                     f"(on the {res['probe']['relevant_frac'][k] * 100:.0f}% of steps where it matters)"})
        best = max(trained, key=lambda l: max(row[k] for row in em[l]))
        peak = max(range(len(em[best])), key=lambda i: em[best][i][k])
        r_peak, r_first = em[best][peak][k], em[best][0][k]
        checks.append({"check": f"{prm.name}: emerges as evidence accumulates",
                       "status": "PASS" if r_peak - r_first > 0.2 else "FAIL",
                       "detail": f"{best} R2 t=0: {r_first:.3f} -> peak {r_peak:.3f} at t={res['probe']['timesteps'][peak]} "
                                 f"(last t: {em[best][-1][k]:.3f})"})
        ic = res["interchange"]
        dec, enc = ic[f"decode/{prm.name}"], ic[f"encode/{prm.name}"]
        b = max(dec, enc, key=lambda r: r["corr"])
        checks.append({"check": f"{prm.name}: its direction is causally used (Level 4, exploratory: swap test)",
                       "status": "PASS" if b["corr"] > 0.3 and b["corr"] > b["rand_corr"] + 0.2 else "FAIL",
                       "detail": f"corr over {len(dec['corr_by_k'])} steps after swap: decode={dec['corr']:.3f} / encode={enc['corr']:.3f} "
                                 f"(slope {dec['slope']:.2f} / {enc['slope']:.2f}); random natural direction |corr|={b['rand_corr']:.3f}"})
    return checks
