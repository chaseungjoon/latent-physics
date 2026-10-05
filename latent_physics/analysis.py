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
    """(n, T, K) targets: hidden params in probe space (constant in t) + observable velocity v_t."""
    z = env.to_probe(data["params"])
    T = data["actions"].shape[1]
    v = data["states"][:, :T, env.state_names.index("v")]
    targets = np.concatenate([np.broadcast_to(z[:, None], (len(z), T, z.shape[1])), v[..., None]], -1)
    return targets, [p.name for p in env.params] + ["v (observable)"]


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


@torch.no_grad()
def intervention_test(model: GRUWorldModel, env: Env, norm: Normalizer, data: dict, hid: dict[str, np.ndarray],
                      probes: dict[str, RidgeProbe], z_std: np.ndarray, test_idx: np.ndarray, t_lo: int, rng,
                      device, scale: float = 0.5, samples_per_traj: int = 4, n_random: int = 5) -> dict:
    """Shift a GRU layer's state along a param's probe direction so the probe reading moves by
    +-scale*std, and compare the change in the predicted next-state with the simulator's change when
    that param is actually moved by the same amount (same s_t, a_t, other params fixed).

    corr  -- Pearson correlation between predicted and true effects (1 = physically faithful)
    slope -- regression of predicted on true effect (1 = right magnitude)
    rand_corr -- same with random directions of equal norm (control)
    """
    states, actions, params = data["states"], data["actions"], data["params"]
    T = actions.shape[1]
    traj = np.repeat(test_idx, samples_per_traj)
    ts = rng.integers(max(t_lo, 1), T, size=len(traj))
    X = norm.inputs(states[traj], actions[traj])[np.arange(len(traj)), ts]
    x_t = torch.as_tensor(X, device=device)
    names = model.gru_names
    carry = [torch.as_tensor(hid[n][traj, ts - 1], device=device) for n in names]
    base, _ = model.step(x_t, carry)

    s_t, a_t, p = states[traj, ts], actions[traj, ts], params[traj]
    s_next = env.step(s_t, a_t, p)
    z = env.to_probe(p)

    def shifted(li, dh):
        c = list(carry)
        c[li] = c[li] + torch.as_tensor(dh, dtype=torch.float32, device=device)
        return (model.step(x_t, c)[0] - base).cpu().numpy()

    out = {}
    for li, name in enumerate(names):
        H = hid[name][test_idx, t_lo:].reshape(-1, hid[name].shape[-1])
        spread = np.sqrt(H.var(0).sum())
        for k, prm in enumerate(env.params):
            w = probes[name].direction(k)
            pred, true, rand = [], [], []
            for sign in (1.0, -1.0):
                delta = sign * scale * z_std[k]
                dh = delta * w / (w @ w)
                pred.append(shifted(li, dh))
                rand.append(np.stack([shifted(li, r / np.linalg.norm(r) * np.linalg.norm(dh))
                                      for r in rng.normal(size=(n_random, len(w)))]))
                z2 = z.copy()
                z2[:, k] += delta
                p2 = np.maximum(env.from_probe(z2), 1e-6)
                true.append((env.step(s_t, a_t, p2) - s_next) / norm.d_std)
            pred, true, rand = np.stack(pred), np.stack(true), np.stack(rand, 1)  # rand: (n_random, 2, N, S)
            out[f"{name}/{prm.name}"] = {
                "layer": name, "param": prm.name,
                "corr": _corr(pred, true),
                "slope": float((pred * true).sum() / ((true ** 2).sum() + 1e-12)),
                "rand_corr": float(np.mean([abs(_corr(r, true)) for r in rand])),
                "shift_norm_over_spread": float(np.linalg.norm(dh) / spread),
                # velocity-dimension samples for plotting
                "_pred_v": pred[..., env.state_names.index("v")].ravel(),
                "_true_v": true[..., env.state_names.index("v")].ravel(),
            }
    return out


@torch.no_grad()
def ablation_test(model: GRUWorldModel, env: Env, X: torch.Tensor, Y: torch.Tensor, hid: dict[str, np.ndarray],
                  probes: dict[str, RidgeProbe], train_idx, test_idx, t_lo: int, rng, device, n_random: int = 3) -> dict:
    """Mean-ablate a param's 1-D probe direction in one GRU layer at every timestep and measure the
    relative increase in late-timestep one-step loss, versus ablating random directions."""
    Xt, Yt = X[test_idx], Y[test_idx]

    def late_loss(edit=None):
        return ((model.forward_stepwise(Xt, edit) - Yt) ** 2)[:, t_lo:].mean().item()

    def ablate(li, u_np, mu_np):
        u = torch.as_tensor(u_np, dtype=torch.float32, device=device)
        mu = torch.as_tensor(mu_np, dtype=torch.float32, device=device)
        return lambda i, h: h - ((h - mu) @ u)[:, None] * u if i == li else h

    base = late_loss()
    out = {"base_loss": base}
    for li, name in enumerate(model.gru_names):
        mu = hid[name][train_idx].reshape(-1, hid[name].shape[-1]).mean(0)
        rand = np.mean([late_loss(ablate(li, r / np.linalg.norm(r), mu))
                        for r in rng.normal(size=(n_random, len(mu)))])
        for k, prm in enumerate(env.params):
            w = probes[name].direction(k)
            loss = late_loss(ablate(li, w / np.linalg.norm(w), mu))
            out[f"{name}/{prm.name}"] = {"rel_increase": (loss - base) / base,
                                         "rand_rel_increase": (rand - base) / base}
    return out
