"""LCP certificates, coupled-contact response and reset determinism."""

import itertools
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_env import activate_dependencies
activate_dependencies()

import mujoco
import numpy as np
import pytest
from lcp_backend import (solve_lcp, solve_lcp_active_set, coulomb_problem, solve_coulomb_lcp,
                         pyramid_contacts)
from compare_prediction import make_model, fresh_data, backend_for, argument_parser


def _point_contact_rows(mu):
    normal = np.array([0., 0., 1.])
    t1, t2 = np.array([1., 0., 0.]), np.array([0., 1., 0.])
    return np.array([normal + mu*t1, normal - mu*t1, normal + mu*t2, normal - mu*t2])


@pytest.mark.parametrize("vx", [1.0, 0.002, 0.0])
def test_coulomb_facet_lcp_point_mass_matches_analytic_coulomb(vx):
    h, mu = .002, .5
    rows = _point_contact_rows(mu)
    free = np.array([vx, 0., -9.81*h])
    contacts = pyramid_contacts([3], [0], 4)
    problem = coulomb_problem(rows, -1e-3*np.ones(4), free, np.eye(3), h, 1e-8, contacts, nonnegative_gap=True)
    z, certificate = solve_coulomb_lcp(problem["M"], problem["q"])
    post = free + rows.T @ problem["decode"](z)
    # Non-penetration without pop-out, normal impulse m g h, Coulomb bound sum(beta) <= c_n.
    assert certificate["residual"] <= 1e-8
    assert post[2] == pytest.approx(0., abs=1e-9)
    assert z[0] == pytest.approx(9.81*h, abs=1e-9)
    assert z[1:5].sum() <= z[0] + 1e-12
    expected = max(vx - mu*9.81*h, 0.)
    assert post[0] == pytest.approx(expected, abs=1e-9)
    if expected > 0:
        assert z[1:5].sum() == pytest.approx(z[0], abs=1e-9)  # sliding: on the friction bound


def test_coulomb_facet_lcp_signed_gap_pops_out_and_nonnegative_does_not():
    h = .002
    rows = _point_contact_rows(.5)
    free = np.array([0., 0., -9.81*h])
    contacts = pyramid_contacts([3], [0], 4)
    signed = coulomb_problem(rows, -1e-3*np.ones(4), free, np.eye(3), h, 1e-8, contacts, nonnegative_gap=False)
    z, _ = solve_coulomb_lcp(signed["M"], signed["q"])
    assert (free + rows.T @ signed["decode"](z))[2] == pytest.approx(1e-3/h, abs=1e-7)  # r=1e-8 compliance
    clipped = coulomb_problem(rows, -1e-3*np.ones(4), free, np.eye(3), h, 1e-8, contacts, nonnegative_gap=True)
    z, _ = solve_coulomb_lcp(clipped["M"], clipped["q"])
    assert (free + rows.T @ clipped["decode"](z))[2] == pytest.approx(0., abs=1e-9)


def test_coulomb_decoder_preserves_generalized_impulse_and_pyramid_structure():
    rows = _point_contact_rows(.5)
    contacts = pyramid_contacts([3], [0], 4)
    problem = coulomb_problem(rows, np.zeros(4), np.array([.3, -.2, -.02]), np.eye(3), .002, 1e-8, contacts)
    z, _ = solve_coulomb_lcp(problem["M"], problem["q"])
    p = problem["decode"](z)
    assert p.min() >= 0
    C, F = problem["normals"], problem["facets"]
    expected = problem["Jn"].T @ z[:C] + problem["Jd"].T @ z[C:C+F]
    np.testing.assert_allclose(rows.T @ p, expected, atol=1e-12)
    assert p.sum() == pytest.approx(z[0], abs=1e-12)
    with pytest.raises(ValueError, match="contact row"):
        pyramid_contacts([3], [0], 5)


@pytest.mark.parametrize("seed", range(8))
def test_coupled_lcp_matches_exhaustive_active_sets(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(6, 6))
    W = A.T @ A + .2 * np.eye(6)
    g = rng.normal(size=6)
    expected = None
    for bits in itertools.product((False, True), repeat=6):
        ids = np.flatnonzero(bits)
        candidate = np.zeros(6)
        candidate[ids] = np.linalg.solve(W[np.ix_(ids, ids)], -g[ids])
        slack = W @ candidate + g
        if np.min(candidate) >= -1e-10 and np.min(slack) >= -1e-10:
            expected = candidate
            break
    assert expected is not None
    impulse, certificate = solve_lcp(W, g)
    np.testing.assert_allclose(impulse, expected, atol=1e-9)
    independent, _ = solve_lcp_active_set(W, g)
    np.testing.assert_allclose(impulse, independent, atol=1e-9)
    assert certificate["solver"] == "quantecon-lemke"
    assert certificate["residual"] <= 1e-8
    assert np.linalg.norm(W - np.diag(np.diag(W))) > 0


def test_empty_separation_and_failed_solve_are_explicit():
    impulse, certificate = solve_lcp(np.zeros((0, 0)), np.zeros(0))
    assert impulse.size == 0 and certificate["residual"] == 0
    with pytest.raises(FloatingPointError, match="iteration limit"):
        solve_lcp(np.eye(3), -np.ones(3), max_iterations=1)
    with pytest.raises(ValueError, match="symmetric"):
        solve_lcp(np.array([[1., 2.], [0., 1.]]), np.ones(2))


@pytest.mark.parametrize("shape", ["sphere", "box", "elephant"])
def test_lcp_gpu_pipeline_preserves_coupled_impulse_and_reset(shape):
    import warp as wp
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA required by common collision/integration pipeline")
    model = make_model(shape, .5, .002, .15)
    initial = fresh_data(model)
    initial.qpos[2] -= .00505
    initial.qvel[:] = [-.3, .1, -.02, .02, -.01, .1]
    mujoco.mj_forward(model, initial)
    args = argument_parser().parse_args(["--headless", "--engines", "mjwarp", "lcp"])
    backend = backend_for("lcp", model, initial, args)
    states = []
    for _ in range(2):
        data = fresh_data(model, initial.qpos, initial.qvel)
        backend.reset(data)
        backend.step(data)
        count = int(backend.wdata.nefc.numpy()[0])
        assert count > 0
        J = backend.wdata.efc.J.numpy()[0, :count, :6].astype(float)
        gap = backend.wdata.efc.efc_dist.numpy()[0, :count].astype(float)
        free = backend.wdata.qvel_smooth_pred.numpy()[0].astype(float)
        inverse = (np.diag(backend.api.inverse_diagonal.numpy().astype(float)) if backend.api.use_diagonal
                   else backend.api.inverse_dense.numpy().astype(float))
        impulse, certificate = solve_lcp(J @ inverse @ J.T + 1e-8 * np.eye(count), J @ free + gap / .002)
        np.testing.assert_allclose(data.qvel, free + inverse @ J.T @ impulse, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(backend.wdata.efc.force.numpy()[0, :count] * .002, impulse, rtol=2e-6, atol=2e-8)
        assert backend.api.last_lcp["residual"] <= args.lcp_tolerance
        states.append(data.qvel.copy())
    np.testing.assert_array_equal(states[0], states[1])
