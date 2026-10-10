"""Level 4: the interchange (swap) test along 1-D parameter directions."""
from __future__ import annotations

import numpy as np
import torch

from ..data import Normalizer
from ..envs import Env
from ..models import GRUWorldModel
from .partners import match_partners
from .stats import corr
from .swaps import make_swap_set, rollout, swap_response, swap_stats


@torch.no_grad()
def interchange_test(model: GRUWorldModel, env: Env, norm: Normalizer, data: dict, hid: dict[str, np.ndarray],
                     dirs: dict, rand_dirs: list, test_idx: np.ndarray, t_lo: int, rng, device,
                     samples_per_traj: int = 4, match_steps: int = 2, horizon: int = 10) -> dict:
    """Swap ("interchange") test of causal use, on real hidden states only.

    For trajectory A at time t, take the recurrent state of a different trajectory B whose recent history
    matches A's (nearest neighbour on the last `match_steps` inputs, any late t') and copy over only its
    component along a param's direction (in every GRU layer), then predict A's next step. Matching makes
    the two memories differ mainly in what they believe about the hidden params, not in recent motion.

    After the swap the model keeps reading A's real observations for `horizon` steps. At every step k the
    change in its prediction is compared with the simulator's change when A's param is replaced by B's
    (same states and actions of A). A filtering model may first react to the swap as a "surprise" (k=0)
    and only express the swapped belief over the next steps, until A's evidence overwrites it.

    full        -- swap the whole memory: does the model use its memory as its physics belief at all?
    corr        -- correlation of predicted vs. true effect pooled over the horizon (1 = like physics)
    corr_by_k   -- the same per step after the swap; corr_k0 = immediate step only
    slope       -- 1 = the model fully adopts B's value along that direction
    corr_other  -- strongest |corr| with the effect of swapping a *different* param (entanglement)
    """
    layers = model.gru_names
    T = data["actions"].shape[1]
    Xt = norm.inputs(data["states"][test_idx], data["actions"][test_idx])  # rows aligned with test_idx
    lo = max(t_lo, match_steps)
    ia = np.repeat(np.arange(len(test_idx)), samples_per_traj)
    ts = rng.integers(lo, T - horizon + 1, size=len(ia))
    ib, tb, _ = match_partners(Xt, ia, ts, lo, device, match_steps)
    s = make_swap_set(env, norm, data, hid, layers, test_idx[ia], ts, test_idx[ib], tb, horizon, device)
    base = rollout(model, s.xs, s.cA)

    def patched(us):
        Us = None if us is None else [torch.as_tensor(u, dtype=torch.float32, device=device)[:, None] for u in us]
        return swap_response(model, s, Us, "once", base).cpu().numpy()

    def stats(pred, true):
        r = swap_stats(pred, true)
        return {k: r[k] for k in ("corr", "corr_k0", "corr_by_k", "slope")}

    vi = env.state_names.index("v")
    P = len(env.params)
    true = [s.true_effect([k]) for k in range(P)]
    pred_full, true_full = patched(None), s.true_effect(list(range(P)))
    out = {"full": {**stats(pred_full, true_full), "_pred_v": pred_full[..., vi].ravel(), "_true_v": true_full[..., vi].ravel()}}
    rand_preds = [patched(us) for us in rand_dirs]
    for k, prm in enumerate(env.params):
        rand = [stats(rp, true[k]) for rp in rand_preds]
        for kind, d in dirs.items():
            pred = patched([d[n][k] for n in layers])
            out[f"{kind}/{prm.name}"] = {
                **stats(pred, true[k]), "kind": kind, "param": prm.name,
                "rand_corr": float(np.mean([abs(r["corr"]) for r in rand])),
                "rand_corr_by_k": np.mean([np.abs(r["corr_by_k"]) for r in rand], 0).tolist(),
                "corr_other": max([abs(corr(pred, true[j])) for j in range(P) if j != k], default=0.0),
                "_pred_v": pred[..., vi].ravel(), "_true_v": true[k][..., vi].ravel(),
            }
    return out
