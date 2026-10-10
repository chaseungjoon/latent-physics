"""Consistency checks for the swap machinery used by the Level-4 tests."""
import numpy as np
import torch

from latent_physics.analysis.partners import match_partners
from latent_physics.analysis.swaps import make_swap_set, swap_response
from latent_physics.analysis.probes import hidden_states
from latent_physics.data import Normalizer, generate_split
from latent_physics.envs import make_env
from latent_physics.models import GRUWorldModel


def _setup(n=40, T=30, k=5):
    torch.manual_seed(0)
    env = make_env("spring")
    data = generate_split(env, "id", n, T, 0.5, seed=0)
    norm = Normalizer.fit(data, env)
    X = norm.inputs(data["states"], data["actions"])
    model = GRUWorldModel(X.shape[-1], 2, hidden=16).eval()
    hid = hidden_states(model, torch.as_tensor(X))
    rng = np.random.default_rng(0)
    ia = np.arange(n)
    ts = rng.integers(5, T - k + 1, size=n)
    ib, tb, _ = match_partners(X, ia, ts, 5, "cpu")
    s = make_swap_set(env, norm, data, hid, model.gru_names, ia, ts, ib, tb, k, "cpu")
    return model, s, ia, ib


def test_partners_are_other_trajectories():
    _, _, ia, ib = _setup()
    assert (ia != ib).all()


def test_full_subspace_equals_whole_memory_swap():
    model, s, _, _ = _setup()
    eye = [torch.eye(16) for _ in model.gru_names]
    with torch.no_grad():
        whole = swap_response(model, s, None, "once")
        full = swap_response(model, s, eye, "once")
    torch.testing.assert_close(whole, full, atol=1e-5, rtol=1e-4)


def test_empty_swap_changes_nothing_and_same_params_have_no_true_effect():
    model, s, _, _ = _setup()
    zero = [torch.zeros(16, 1) for _ in model.gru_names]
    with torch.no_grad():
        for mode in ("once", "clamp"):
            assert swap_response(model, s, zero, mode).abs().max() < 1e-6
    z_a = s.env.to_probe(s.params_a)
    np.testing.assert_allclose(s.true_effect([0, 1, 2], z_b=z_a), 0, atol=1e-8)


def test_subset_matches_full_set():
    model, s, _, _ = _setup()
    idx = np.array([3, 7, 11])
    with torch.no_grad():
        a = swap_response(model, s, None, "once")[:, idx]
        b = swap_response(model, s.subset(idx), None, "once")
    torch.testing.assert_close(a, b)
    np.testing.assert_allclose(s.true_effect([1])[:, idx], s.subset(idx).true_effect([1]))
