"""Level 4: mean-ablation of parameter directions."""
from __future__ import annotations

import numpy as np
import torch

from ..envs import Env
from ..models import GRUWorldModel
from .sensitivity import param_sensitivity
from .stats import corr


@torch.no_grad()
def ablation_test(model: GRUWorldModel, env: Env, data: dict, X: torch.Tensor, Y: torch.Tensor, hid: dict[str, np.ndarray],
                  dirs: dict, rand_dirs: list, train_idx, test_idx, t_lo: int, device) -> dict:
    """Mean-ablate one direction per GRU layer at every timestep and measure the damage.

    rel_increase -- relative increase of late one-step loss
    var_frac     -- fraction of the hidden variance that lives along the ablated directions; damage
                    should be compared between directions of similar var_frac
    sens_corr    -- correlation, over (trajectory, t), between the extra error and how much that param
                    physically matters at that step (|change of next state| if the param moved by 1 std).
                    High = the damage lands exactly where this param matters: a specific, not generic, role.
    """
    layers = model.gru_names
    Xt, Yt = X[test_idx], Y[test_idx]
    Hs = [hid[n][train_idx, t_lo:].reshape(-1, hid[n].shape[-1]) for n in layers]
    mus = [torch.as_tensor(H.mean(0), dtype=torch.float32, device=device) for H in Hs]
    covs = [np.cov(H, rowvar=False) for H in Hs]

    def run(us):
        edit = None
        if us is not None:
            ut = [torch.as_tensor(u, dtype=torch.float32, device=device) for u in us]
            edit = lambda i, h: h - ((h - mus[i]) @ ut[i])[:, None] * ut[i]  # noqa: E731
        return ((model.forward_stepwise(Xt, edit) - Yt) ** 2).mean(-1)[:, t_lo:].cpu().numpy()

    def var_frac(us):
        return float(np.mean([u @ C @ u / np.trace(C) for u, C in zip(us, covs)]))

    z_std = env.to_probe(data["params"][train_idx]).std(0)
    sens = param_sensitivity(env, data, test_idx, z_std, t_lo)

    base = run(None)
    out = {"base_loss": float(base.mean())}
    rand_err = [run(us) for us in rand_dirs]
    for k, prm in enumerate(env.params):
        rand = {"rel_increase": float(np.mean([(e.mean() - base.mean()) / base.mean() for e in rand_err])),
                "var_frac": float(np.mean([var_frac(us) for us in rand_dirs])),
                "sens_corr": float(np.mean([corr(e - base, sens[..., k]) for e in rand_err]))}
        for kind, d in dirs.items():
            us = [d[n][k] for n in layers]
            err = run(us)
            out[f"{kind}/{prm.name}"] = {"rel_increase": float((err.mean() - base.mean()) / base.mean()),
                                         "var_frac": var_frac(us), "sens_corr": corr(err - base, sens[..., k]),
                                         "random": rand}
    return out
