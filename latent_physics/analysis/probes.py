"""Level 2: linear (and one nonlinear) probes for hidden parameters in the model's states."""
from __future__ import annotations

import numpy as np
import torch

from ..envs import Env
from ..models import GRUWorldModel
from .stats import r2


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
