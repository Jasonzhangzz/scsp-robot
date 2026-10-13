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
from lcp_backend import solve_lcp, solve_lcp_active_set
from compare_prediction import make_model, fresh_data, backend_for, argument_parser


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
