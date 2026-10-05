"""Physics sanity checks: if the simulator is wrong, every downstream probe result is meaningless."""
import numpy as np

from latent_physics.envs import G, ForcedMotion, SpringDamper


def _const(n, T, F):
    return np.full((n, T, 1), F, dtype=float)


def test_coasting_deceleration_is_mu_g_and_mass_independent():
    env = ForcedMotion()
    p = np.array([[1.0, 0.2], [5.0, 0.2]])  # different mass, same friction
    s0 = np.array([[0.0, 3.0], [0.0, 3.0]])
    states = env.simulate(s0, _const(2, 10, 0.0), p)
    dv = states[:, 1, 1] - states[:, 0, 1]
    np.testing.assert_allclose(dv, -0.2 * G * env.dt, rtol=1e-6)
    np.testing.assert_allclose(states[0], states[1])  # mass is invisible without force


def test_friction_stops_block_and_does_not_reverse():
    env = ForcedMotion()
    states = env.simulate(np.array([[0.0, 1.0]]), _const(1, 40, 0.0), np.array([[2.0, 0.3]]))
    v = states[0, :, 1]
    assert (v >= 0).all() and v[-1] == 0.0


def test_static_friction_holds_below_threshold():
    env = ForcedMotion()
    m, mu = 2.0, 0.3
    F = 0.9 * mu * m * G
    states = env.simulate(np.array([[0.0, 0.0]]), _const(1, 20, F), np.array([[m, mu]]))
    assert np.all(states[0, :, 1] == 0.0)


def test_force_response_scales_with_inverse_mass():
    env = ForcedMotion()
    p = np.array([[1.0, 0.05], [4.0, 0.05]])
    s0 = np.array([[0.0, 2.0], [0.0, 2.0]])
    a = env.step(s0, np.array([[20.0], [20.0]]), p) - env.step(s0, np.zeros((2, 1)), p)
    np.testing.assert_allclose(a[:, 1], 20.0 / p[:, 0] * env.dt, rtol=1e-6)


def test_spring_conserves_energy_without_damping_or_force():
    env = SpringDamper()
    m, k = 1.0, 4.0
    states = env.simulate(np.array([[1.0, 0.0]]), _const(1, 200, 0.0), np.array([[m, k, 1e-12]]))
    E = 0.5 * m * states[0, :, 1] ** 2 + 0.5 * k * states[0, :, 0] ** 2
    assert np.ptp(E) / E[0] < 0.02


def test_splits_respect_holdouts():
    for env in (ForcedMotion(), SpringDamper()):
        rng = np.random.default_rng(0)
        assert not env.in_comp_corner(env.sample_params(5000, rng, "id")).any()
        assert env.in_comp_corner(env.sample_params(500, rng, "comp")).all()
        prm = env.params[env.extrap_param]
        ex = env.sample_params(500, rng, "extrap")[:, env.extrap_param]
        assert (ex >= prm.high).all() and (ex <= prm.extrap_high).all()


def test_zero_force_prob_gives_no_actions():
    env = ForcedMotion()
    assert not env.sample_actions(10, 50, np.random.default_rng(0), force_prob=0.0).any()
