"""Closed-form checks for the spectral and Jacobi residual SCM updates."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_env import activate_dependencies
activate_dependencies()

import numpy as np
import pytest

from residual_scm import (
    apply_delassus, coupled_delassus, environment_impulses, normalized_gershgorin,
    softplus, spectral_radius,
)


def quadratic(impulse, matrix, linear):
    return 0.5 * impulse @ matrix @ impulse + linear @ impulse


def random_contact(seed, rows=6, dofs=4, regularization=1e-4):
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(dofs, dofs))
    inverse = factor @ factor.T + np.eye(dofs)
    jacobian = rng.normal(size=(rows, dofs))
    matrix = coupled_delassus(jacobian, inverse, regularization)
    linear = rng.normal(size=rows)
    return jacobian, inverse, regularization, matrix, linear


def hard_step(impulse, matrix, linear, diagonal, lipschitz, correction):
    residual = matrix @ impulse + linear
    step = 1.0 / lipschitz
    if correction == "spectral":
        trial = impulse - step * residual
    else:
        trial = impulse - step * residual / diagonal
    return np.maximum(trial, 0.0)


@pytest.mark.parametrize("temperature", [1e-4, 1e-3])
def test_softplus_matches_logaddexp_and_hard_limit(temperature):
    values = np.array([-20., -1e-6, 0., 1e-6, 2., 40.])
    expected = temperature * np.logaddexp(0., values / temperature)
    np.testing.assert_allclose(softplus(values, temperature), expected, rtol=0, atol=0)
    np.testing.assert_allclose(softplus(values, 0.0), np.maximum(values, 0), atol=0)


def test_matvec_matches_dense_delassus_and_power_iteration_finds_largest_eigenvalue():
    jacobian, inverse, regularization, matrix, _ = random_contact(3)
    vector = np.linspace(-0.2, 1.0, matrix.shape[0])
    np.testing.assert_allclose(apply_delassus(jacobian, inverse, regularization, vector), matrix @ vector)
    value, iterations, source = spectral_radius(jacobian, inverse, regularization)
    exact = float(np.linalg.eigvalsh(matrix)[-1])
    assert source == "power"
    assert 0 < iterations <= 200
    assert value == pytest.approx(exact, rel=1e-8, abs=1e-10)


def test_power_iteration_sees_eigenvector_orthogonal_to_ones():
    """Ones lies on the smaller eigenpair of [[2, -1], [-1, 2]]; lambda_max is 3."""
    regularization = 1e-8
    inverse = np.array([[2., -1.], [-1., 2.]]) - regularization * np.eye(2)
    value, _, source = spectral_radius(np.eye(2), inverse, regularization)
    assert source == "power"
    assert value == pytest.approx(3.0, rel=1e-10, abs=1e-12)


def test_gershgorin_upper_bounds_normalized_spectrum():
    for seed in range(12):
        _, _, _, matrix, _ = random_contact(seed)
        diagonal = np.diag(matrix)
        scale = np.sqrt(diagonal)
        normalized = matrix / scale[:, None] / scale[None, :]
        bound = normalized_gershgorin(matrix, diagonal)
        assert bound + 1e-9 >= float(np.linalg.eigvalsh(normalized)[-1])
        assert bound + 1e-12 >= 1.0


def test_preconditioned_step_is_identity_on_a_diagonal_delassus():
    inverse = np.diag([1.0, 4.0])
    beta, info = environment_impulses(
        np.eye(2), inverse, np.array([-1.0, -1e-8]), np.zeros(2), 1.0, 1e-8, 1e-4,
        "preconditioned", 1)
    np.testing.assert_allclose(beta, info["initial"], rtol=0, atol=1e-12)
    assert info["step_size"] == pytest.approx(1.0)


def test_residual_changes_allocation_between_unequal_contacts():
    jacobian = np.array([[1.0, 0.2], [0.15, 1.0]])
    inverse = np.diag([1.0, 5.0])
    projected = np.array([-0.4, -0.05])
    common = dict(jacobian=jacobian, inverse_mass=inverse, projected_velocity=projected,
                  gap=np.zeros(2), timestep=1.0, regularization=1e-8, temperature=1e-4, steps=1)
    spectral, spectral_info = environment_impulses(correction="spectral", **common)
    preconditioned, pre_info = environment_impulses(correction="preconditioned", **common)
    assert np.linalg.norm(spectral - spectral_info["initial"]) > 1e-4
    assert np.linalg.norm(preconditioned - pre_info["initial"]) > 1e-4
    assert np.linalg.norm(spectral - preconditioned) > 1e-4
    for corrected, initial in ((spectral, spectral_info["initial"]), (preconditioned, pre_info["initial"])):
        ratio = corrected / initial
        assert np.max(ratio) - np.min(ratio) > 1e-3


def test_zero_correction_steps_return_the_diagonal_estimate():
    jacobian, inverse, regularization, matrix, linear = random_contact(5)
    beta, info = environment_impulses(
        jacobian, inverse, np.zeros(inverse.shape[0]), linear, 1.0, regularization, 1e-4,
        "spectral", 0, nonnegative_gap=False)
    expected = softplus(-linear / np.diag(matrix), 1e-4)
    np.testing.assert_allclose(beta, expected, rtol=1e-12, atol=1e-12)
    assert info["lipschitz"] is None


@pytest.mark.parametrize("correction", ["spectral", "preconditioned"])
def test_hard_step_decreases_and_stays_closer_to_the_lcp_solution(correction):
    from lcp_backend import solve_lcp
    for seed in range(8):
        rng = np.random.default_rng(10 + seed)
        factor = rng.normal(size=(5, 5))
        matrix = factor @ factor.T / 5.0 + 0.4 * np.eye(5)
        linear = rng.normal(size=5)
        diagonal = np.diag(matrix).copy()
        beta0 = np.maximum(-linear / diagonal, 0.0)
        if correction == "spectral":
            lipschitz = float(np.linalg.eigvalsh(matrix)[-1])
            distance = lambda left, right: np.linalg.norm(left - right)
        else:
            lipschitz = normalized_gershgorin(matrix, diagonal)
            distance = lambda left, right: np.sqrt((left - right) @ (diagonal * (left - right)))
        solution, _ = solve_lcp(matrix, linear, tolerance=1e-8, max_iterations=4000, solver="active-set")
        fixed = hard_step(solution, matrix, linear, diagonal, lipschitz, correction)
        np.testing.assert_allclose(fixed, solution, rtol=0, atol=1e-6)
        updated = hard_step(beta0, matrix, linear, diagonal, lipschitz, correction)
        assert quadratic(updated, matrix, linear) <= quadratic(beta0, matrix, linear) + 1e-8
        assert distance(updated, solution) <= distance(beta0, solution) + 1e-6
        second = hard_step(updated, matrix, linear, diagonal, lipschitz, correction)
        assert quadratic(second, matrix, linear) <= quadratic(updated, matrix, linear) + 1e-8


def test_negative_gap_is_dropped_from_the_linear_term():
    jacobian = np.eye(1)
    beta, _ = environment_impulses(
        jacobian, np.eye(1), np.array([0.2]), np.array([-0.01]), 0.01, 1e-8, 1e-4, "spectral", 1)
    # Separating velocity and a negative gap must not create a restoring impulse.
    assert beta[0] < 1e-8


@pytest.mark.parametrize("engine", ["scm_residual", "scm_preconditioned"])
@pytest.mark.parametrize("motion", ["dynamic", "increment"])
def test_gpu_prediction_matches_residual_formula(engine, motion):
    import warp as wp
    import mujoco
    from benchmark_utils import Backend
    from compare_prediction import fresh_data, make_model
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA required by pinned upstream")
    model = make_model("box", .5, .002, .15)
    state = fresh_data(model)
    state.qpos[2] = .025 - .00005
    state.qvel[:] = [-.3, .1, -.02, 0., 0., .1]
    mujoco.mj_forward(model, state)
    backend = Backend(engine, model, state, scm_motion=motion, nconmax=128, njmax=2048,
                      scm_softplus_temperature=1e-4, scm_correction_steps=1)
    query = fresh_data(model, state.qpos, state.qvel)
    backend.reset(query)
    backend.step(query)
    count = int(backend.wdata.nefc.numpy()[0])
    assert count == 16
    jacobian = backend.wdata.efc.J.numpy()[0, :count, :model.nv].astype(float)
    gap = backend.wdata.efc.efc_dist.numpy()[0, :count].astype(float)
    free = backend.wdata.qvel_smooth_pred.numpy()[0, :model.nv].astype(float)
    # qvel on the data object is already the integrated result. The increment
    # law subtracts the float32 velocity that entered the step.
    incoming = np.asarray(state.qvel, dtype=np.float32).astype(float)
    projected = free if motion == "dynamic" else free - incoming
    correction = "spectral" if engine == "scm_residual" else "preconditioned"
    beta, info = environment_impulses(
        jacobian, backend.api.inverse_mass(), projected, gap, .002, backend.api.regularization,
        1e-4, correction, 1)
    np.testing.assert_allclose(backend.wdata.efc.force.numpy()[0, :count] * .002, beta, rtol=1e-5, atol=1e-8)
    assert backend.last_scm_lipschitz == pytest.approx(info["lipschitz"], rel=1e-12, abs=1e-12)
    assert backend.last_scm_step_size == pytest.approx(info["step_size"], rel=1e-12, abs=1e-12)
    assert backend.last_scm_spectral_source == info["spectral_source"]
    expected = free + backend.api.inverse_mass() @ jacobian.T @ beta
    np.testing.assert_allclose(query.qvel, expected, rtol=2e-5, atol=1e-6)
    assert np.min(beta) >= -1e-12


@pytest.mark.parametrize("engine", ["scm_residual", "scm_preconditioned"])
def test_gpu_free_motion_when_separated(engine):
    import warp as wp
    import mujoco
    from benchmark_utils import Backend
    from compare_prediction import fresh_data, make_model
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA required by pinned upstream")
    model = make_model("sphere", .5, .002, .15)
    state = fresh_data(model)
    state.qpos[2] = .5
    state.qvel[:] = [.4, -.2, .1, 0., 0., 0.]
    mujoco.mj_forward(model, state)
    backend = Backend(engine, model, state, nconmax=128, njmax=2048)
    query = fresh_data(model, state.qpos, state.qvel)
    backend.reset(query)
    backend.step(query)
    assert int(backend.wdata.nefc.numpy()[0]) == 0
    assert backend.last_scm_lipschitz is None
    np.testing.assert_allclose(query.qvel, backend.wdata.qvel_smooth_pred.numpy()[0], atol=1e-8)


def test_invalid_residual_settings_are_rejected():
    with pytest.raises(ValueError, match="temperature"):
        from residual_scm import ResidualScmEngine
        import mujoco
        from compare_prediction import fresh_data, make_model
        model = make_model("box", .5, .002, .15)
        ResidualScmEngine(model, fresh_data(model), softplus_temperature=0.0)
