"""Prediction, probing, intervention, and ablation analyses."""
from __future__ import annotations

import numpy as np
import torch

from .data import Normalizer
from .envs import Env
from .models import GRUWorldModel

# ---------------------------------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------------------------------


@torch.no_grad()
def per_t_mse(model, X: torch.Tensor, Y: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    """One-step teacher-forced MSE at each timestep, shape (T,). Late t = more evidence about phi."""
    model.eval()
    se = torch.zeros(Y.shape[1], device=Y.device)
    for i in range(0, len(X), batch_size):
        se += ((model(X[i:i + batch_size]) - Y[i:i + batch_size]) ** 2).mean(-1).sum(0)
    return (se / len(X)).cpu().numpy()


@torch.no_grad()
def rollout_mse(model, norm: Normalizer, states: np.ndarray, actions: np.ndarray, extra: np.ndarray | None,
                t0: int, horizon: int, device) -> np.ndarray:
    """Teacher-force t0 steps of context, then roll out `horizon` steps on the model's own predictions
    (true actions given). Returns MSE in standardized state units for each horizon step."""
    model.eval()
    nt = norm.torch(device)
    S = torch.as_tensor(states, dtype=torch.float32, device=device)
    A = torch.as_tensor(actions, dtype=torch.float32, device=device)
    E = None if extra is None else torch.as_tensor(extra, dtype=torch.float32, device=device)

    def inp(s, a):
        parts = [((s - nt["s_mean"]) / nt["s_std"])[:, nt["in_dims"]], a / nt["a_std"]]
        return torch.cat(parts + ([] if E is None else [E]), -1)

    carry = None
    for t in range(t0):
        _, carry = model.step(inp(S[:, t], A[:, t]), carry)
    s, errs = S[:, t0].clone(), []
    for h in range(horizon):
        y, carry = model.step(inp(s, A[:, t0 + h]), carry)
        s = s + y * nt["d_std"] + nt["d_mean"]
        errs.append((((s - S[:, t0 + h + 1]) / nt["s_std"]) ** 2).mean().item())
    return np.array(errs)


# ---------------------------------------------------------------------------------------------------
# linear probes
# ---------------------------------------------------------------------------------------------------


class RidgeProbe:
    """Multi-output ridge regression on standardized features; alpha picked per output on a
    held-out group (trajectory) split so pooled-over-time rows never leak across the split."""

    alphas = np.logspace(-2, 4, 7)

    def fit(self, X: np.ndarray, Y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> "RidgeProbe":
        X, Y = X.astype(np.float64), Y.astype(np.float64)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xs, self.ym = (X - self.mu) / self.sd, Y.mean(0)
        Yc = Y - self.ym
        uniq = np.unique(groups)
        va = np.isin(groups, rng.choice(uniq, size=max(1, len(uniq) // 5), replace=False))
        U, s, Vt = np.linalg.svd(Xs[~va], full_matrices=False)
        UtY = U.T @ Yc[~va]
        errs = np.stack([((Xs[va] @ (Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY)) - Yc[va]) ** 2).mean(0)
                         for a in self.alphas])
        self.alpha = self.alphas[errs.argmin(0)]
        U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
        UtY = U.T @ Yc
        self.W = np.stack([Vt.T @ ((s / (s ** 2 + a)) * UtY[:, k]) for k, a in enumerate(self.alpha)], 1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.W + self.ym

    def direction(self, k: int) -> np.ndarray:
        """Raw-feature-space vector w with probe(h + d) - probe(h) = d . w for output k."""
        return self.W[:, k] / self.sd


def r2(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return 1 - ((y - yhat) ** 2).sum(0) / (((y - y.mean(0)) ** 2).sum(0) + 1e-12)


@torch.no_grad()
def hidden_states(model: GRUWorldModel, X: torch.Tensor, batch_size: int = 1024) -> dict[str, np.ndarray]:
    model.eval()
    chunks: dict[str, list] = {}
    for i in range(0, len(X), batch_size):
        _, hid = model(X[i:i + batch_size], return_hidden=True)
        for k, v in hid.items():
            chunks.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in chunks.items()}


def probe_targets(env: Env, data: dict) -> tuple[np.ndarray, list[str]]:
    """(n, T, K) targets: hidden params in probe space (constant in t), env-specific derived quantities
    (e.g. identifiable ratios), and the observable velocity v_t as a sanity check."""
    z = env.to_probe(data["params"])
    derived = env.derived_targets(z)
    if derived:
        z = np.concatenate([z, np.stack(list(derived.values()), -1)], -1)
    T = data["actions"].shape[1]
    v = data["states"][:, :T, env.state_names.index("v")]
    targets = np.concatenate([np.broadcast_to(z[:, None], (len(z), T, z.shape[1])), v[..., None]], -1)
    return targets, [p.name for p in env.params] + list(derived) + ["v (observable)"]


def probe_emergence(layers: dict[str, np.ndarray], targets: np.ndarray, timesteps: list[int],
                    train_idx: np.ndarray, test_idx: np.ndarray, rng) -> dict[str, np.ndarray]:
    """Held-out R^2 of a separate linear probe per (layer, timestep): {layer: (len(timesteps), K)}."""
    out = {}
    for name, H in layers.items():
        res = np.zeros((len(timesteps), targets.shape[-1]))
        for i, t in enumerate(timesteps):
            probe = RidgeProbe().fit(H[train_idx, t], targets[train_idx, t], train_idx, rng)
            res[i] = r2(targets[test_idx, t], probe.predict(H[test_idx, t]))
        out[name] = res
    return out


def shuffled_label_r2(H: np.ndarray, targets: np.ndarray, t: int, train_idx, test_idx, rng) -> np.ndarray:
    """Control: probe fit on labels permuted across trajectories should give R^2 <= 0."""
    Y = targets[train_idx, t][rng.permutation(len(train_idx))]
    probe = RidgeProbe().fit(H[train_idx, t], Y, train_idx, rng)
    return r2(targets[test_idx, t], probe.predict(H[test_idx, t]))


def mlp_probe_r2(H: np.ndarray, Y: np.ndarray, train_idx, test_idx, device, steps: int = 2000, seed: int = 0) -> np.ndarray:
    """Nonlinear (1-hidden-layer MLP) probe at a single timestep. Secondary diagnostic only: it tells
    "absent" apart from "present but not linearly readable" when the linear probe is weak."""
    g = torch.Generator().manual_seed(seed)
    Xtr = torch.as_tensor(H[train_idx], dtype=torch.float32, device=device)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ytr = torch.as_tensor(Y[train_idx], dtype=torch.float32, device=device)
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(H.shape[-1], 256), torch.nn.GELU(), torch.nn.Linear(256, Y.shape[-1]))
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * (p.shape[-1] ** -0.5 if p.dim() > 1 else 0.0))
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = ((net((Xtr - mu) / sd) - (Ytr - ym) / ys) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        Xte = torch.as_tensor(H[test_idx], dtype=torch.float32, device=device)
        pred = (net((Xte - mu) / sd) * ys + ym).cpu().numpy()
    return r2(Y[test_idx], pred)


def pooled(H: np.ndarray, Y: np.ndarray, idx: np.ndarray, t_lo: int):
    """Rows (trajectory, t) for t >= t_lo. Y is per-trajectory (n, K)."""
    T = H.shape[1] - t_lo
    return (H[idx, t_lo:].reshape(-1, H.shape[-1]), np.repeat(Y[idx], T, axis=0), np.repeat(idx, T))


def fit_pooled_probes(hid: dict[str, np.ndarray], z: np.ndarray, train_idx, t_lo: int, rng) -> dict[str, RidgeProbe]:
    """One probe per layer for the hidden params, pooled over late timesteps (enough evidence seen)."""
    probes = {}
    for name, H in hid.items():
        Xp, Yp, g = pooled(H, z, train_idx, t_lo)
        probes[name] = RidgeProbe().fit(Xp, Yp, g, rng)
    return probes


# ---------------------------------------------------------------------------------------------------
# causal tests
# ---------------------------------------------------------------------------------------------------


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def param_directions(hid: dict[str, np.ndarray], z: np.ndarray, X: np.ndarray, probes: dict[str, RidgeProbe],
                     layers: list[str], train_idx, t_lo: int) -> dict[str, dict[str, np.ndarray]]:
    """Two candidate unit directions per (layer, param), each (P, D):

    decode -- the linear probe's readout weights: the direction the *probe* reads the param from.
    encode -- regression of the hidden state on all params *and the two most recent inputs*: the direction
              the state actually moves when the param changes, with the current state held fixed. (Without
              the state control, e.g. "heavy" would also pick up "moving slowly".)
    The two differ whenever hidden dims are correlated; which one the model uses is an empirical question.
    """
    T, n = X.shape[1], len(train_idx)
    ctx = np.concatenate([X[train_idx, t_lo:], X[train_idx, t_lo - 1:T - 1]], -1).reshape(n * (T - t_lo), -1)
    out = {"decode": {}, "encode": {}}
    for name in layers:
        H, Z, _ = pooled(hid[name], z, train_idx, t_lo)
        dec = np.stack([probes[name].direction(k) for k in range(z.shape[1])])
        R = np.concatenate([Z, ctx], -1)
        enc = np.linalg.lstsq(R - R.mean(0), H - H.mean(0), rcond=None)[0][: z.shape[1]]
        out["decode"][name] = dec / np.linalg.norm(dec, axis=1, keepdims=True)
        out["encode"][name] = enc / np.linalg.norm(enc, axis=1, keepdims=True)
    return out


def natural_random_directions(hid: dict[str, np.ndarray], layers: list[str], train_idx, t_lo: int, n: int, rng):
    """Random unit directions distributed like the hidden states themselves (N(0, Cov) draws), so the
    control moves the state by a typical natural amount rather than along near-empty dimensions."""
    out = []
    for _ in range(n):
        us = []
        for name in layers:
            H = hid[name][train_idx, t_lo:].reshape(-1, hid[name].shape[-1])
            rows = H[rng.integers(len(H), size=512)] - H.mean(0)
            u = rows.T @ rng.normal(size=len(rows))
            us.append(u / np.linalg.norm(u))
        out.append(us)
    return out


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
    states, actions, params = data["states"], data["actions"], data["params"]
    T = actions.shape[1]
    Xall = norm.inputs(states[test_idx], actions[test_idx])  # (n_test, T, D), rows aligned with test_idx
    lo = max(t_lo, match_steps)
    ia = np.repeat(np.arange(len(test_idx)), samples_per_traj)
    ts = rng.integers(lo, T - horizon + 1, size=len(ia))

    def history(i, t):  # the inputs the memory at t-1 has most recently consumed
        return np.concatenate([Xall[i, t - j] for j in range(1, match_steps + 1)], -1)

    ci, ct = np.meshgrid(np.arange(len(test_idx)), np.arange(lo, T), indexing="ij")
    ci, ct = ci.ravel(), ct.ravel()
    cand = torch.as_tensor(history(ci, ct), device=device)
    query = torch.as_tensor(history(ia, ts), device=device)
    ib, tb = np.empty_like(ia), np.empty_like(ts)
    for s0 in range(0, len(ia), 256):
        d = torch.cdist(query[s0:s0 + 256], cand)
        d[torch.as_tensor(ia[s0:s0 + 256], device=device)[:, None] == torch.as_tensor(ci, device=device)[None]] = float("inf")
        j = d.argmin(1).cpu().numpy()
        ib[s0:s0 + 256], tb[s0:s0 + 256] = ci[j], ct[j]
    A_, B_ = test_idx[ia], test_idx[ib]
    xs = [torch.as_tensor(Xall[ia, ts + k], device=device) for k in range(horizon)]
    cA = [torch.as_tensor(hid[n][A_, ts - 1], device=device) for n in layers]
    cB = [torch.as_tensor(hid[n][B_, tb - 1], device=device) for n in layers]

    def run(carry):  # (horizon, N, S) predictions while reading A's real inputs
        ys = []
        for x in xs:
            y, carry = model.step(x, carry)
            ys.append(y)
        return torch.stack(ys).cpu().numpy()

    base = run(cA)
    s_k = np.stack([states[A_, ts + k] for k in range(horizon)])
    a_k = np.stack([actions[A_, ts + k] for k in range(horizon)])
    flat = lambda x: x.reshape(-1, x.shape[-1])  # noqa: E731
    pA = np.tile(params[A_], (horizon, 1))
    s_next = env.step(flat(s_k), flat(a_k), pA)
    zA, zB = env.to_probe(params[A_]), env.to_probe(params[B_])

    def true_effect(ks):
        z = zA.copy()
        z[:, ks] = zB[:, ks]
        eff = (env.step(flat(s_k), flat(a_k), np.tile(env.from_probe(z), (horizon, 1))) - s_next) / norm.d_std
        return eff.reshape(base.shape)

    def patched(us):
        c = []
        for li in range(len(layers)):
            if us is None:
                c.append(cB[li])
            else:
                u = torch.as_tensor(us[li], dtype=torch.float32, device=device)
                c.append(cA[li] + ((cB[li] - cA[li]) @ u)[:, None] * u)
        return run(c) - base

    def stats(pred, true):
        by_k = [_corr(pred[k], true[k]) for k in range(horizon)]
        return {"corr": _corr(pred, true), "corr_k0": by_k[0], "corr_by_k": by_k,
                "slope": float((pred * true).sum() / ((true ** 2).sum() + 1e-12))}

    vi = env.state_names.index("v")
    P = len(env.params)
    true = [true_effect([k]) for k in range(P)]
    pred_full, true_full = patched(None), true_effect(list(range(P)))
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
                "corr_other": max([abs(_corr(pred, true[j])) for j in range(P) if j != k], default=0.0),
                "_pred_v": pred[..., vi].ravel(), "_true_v": true[k][..., vi].ravel(),
            }
    return out


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
                "sens_corr": float(np.mean([_corr(e - base, sens[..., k]) for e in rand_err]))}
        for kind, d in dirs.items():
            us = [d[n][k] for n in layers]
            err = run(us)
            out[f"{kind}/{prm.name}"] = {"rel_increase": float((err.mean() - base.mean()) / base.mean()),
                                         "var_frac": var_frac(us), "sens_corr": _corr(err - base, sens[..., k]),
                                         "random": rand}
    return out


def param_sensitivity(env: Env, data: dict, idx: np.ndarray, z_std: np.ndarray, t_lo: int) -> np.ndarray:
    """How much each param physically matters at each step: squared change of the next state (in units of
    the delta std) if the param moved by +1 std, for steps t >= t_lo. Returns (len(idx), T - t_lo, P)."""
    states, actions, params = data["states"][idx], data["actions"][idx], data["params"][idx]
    T = actions.shape[1]
    n, Tl = len(idx), T - t_lo
    s = states[:, t_lo:T].reshape(n * Tl, -1)
    a = actions[:, t_lo:].reshape(n * Tl, -1)
    z = np.repeat(env.to_probe(params), Tl, axis=0)
    s_next = env.step(s, a, env.from_probe(z))
    d_std = norm_d_std(data, env)
    out = []
    for k in range(len(env.params)):
        z2 = z.copy()
        z2[:, k] += z_std[k]
        out.append((((env.step(s, a, env.from_probe(z2)) - s_next) / d_std) ** 2).sum(-1).reshape(n, Tl))
    return np.stack(out, -1)


def relevant_r2(layers: dict[str, np.ndarray], z: np.ndarray, sens: np.ndarray, train_idx, test_idx, t_lo: int, rng,
                rel_frac: float = 0.1) -> tuple[dict[str, list], list[float]]:
    """Linear-probe R^2 measured only at the steps where the param currently matters for the next step
    (sensitivity > rel_frac * its mean), pooled over t >= t_lo. Answers "does the model know the param
    when it needs it?"; the last-timestep R^2 instead penalizes forgetting a param that no longer matters
    (e.g. friction after the block has stopped). `sens` is param_sensitivity over the whole split.

    Returns ({layer: [R^2 or None per param]}, fraction of relevant steps per param). None = the param
    (almost) never matters in this data, i.e. it is not identifiable from it."""
    P = z.shape[1]
    out = {name: [None] * P for name in layers}
    frac = []
    for k in range(P):
        sk = sens[..., k]
        rel = sk > rel_frac * sk.mean()
        frac.append(float(rel.mean()))
        if sk.mean() < 1e-10 or rel[train_idx].sum() < 50 or rel[test_idx].sum() < 50:
            continue
        for name, H in layers.items():
            Hl = H[:, t_lo:]
            tr, te = rel[train_idx], rel[test_idx]
            Xtr, Xte = Hl[train_idx][tr], Hl[test_idx][te]
            ytr = np.broadcast_to(z[train_idx, None, k], tr.shape)[tr][:, None]
            yte = np.broadcast_to(z[test_idx, None, k], te.shape)[te][:, None]
            groups = np.broadcast_to(train_idx[:, None], tr.shape)[tr]
            probe = RidgeProbe().fit(Xtr, ytr, groups, rng)
            out[name][k] = float(r2(yte, probe.predict(Xte))[0])
    return out, frac


def norm_d_std(data: dict, env: Env) -> np.ndarray:
    """Scale for state changes: std of one-step deltas in this split."""
    return np.diff(data["states"], axis=1).reshape(-1, len(env.state_names)).std(0) + 1e-6
