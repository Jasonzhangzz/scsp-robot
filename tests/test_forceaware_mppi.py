"""MPPI updates for native FA losses, including invalid rollout handling."""

import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import patch


def test_weights_are_stable_normalized_fa_cost_weights():
    from planning.dexforge_forceaware import mppi_weights

    costs = np.array([12.0, 12.5, 13.0, 14.0])
    temperature = 0.5
    expected = np.exp(-(costs - costs.min()) / temperature)
    expected /= expected.sum()
    weights = mppi_weights(costs, temperature)
    np.testing.assert_allclose(weights, expected, rtol=1.e-14)
    np.testing.assert_allclose(mppi_weights(costs + 1.e12, temperature), expected, rtol=1.e-14)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(np.diff(weights) < 0.0)


def test_invalid_rollouts_have_zero_weight_even_with_low_cost():
    from planning.dexforge_forceaware import mppi_weights

    costs = np.array([3.0, -1.e9, np.nan, np.inf, -np.inf, 4.0])
    valid = np.array([True, False, True, True, True, True])
    weights = mppi_weights(costs, 1.0, valid=valid)
    np.testing.assert_array_equal(weights[1:5], np.zeros(4))
    np.testing.assert_allclose(weights[[0, 5]], [1 / (1 + np.exp(-1)), 1 / (1 + np.exp(1))])
    np.testing.assert_allclose(mppi_weights([2.0, 2.0, 2.0], 0.1), np.full(3, 1 / 3))


@pytest.mark.parametrize("temperature", [0.0, -1.0, np.nan, np.inf])
def test_invalid_mppi_temperature_rejected(temperature):
    from planning.dexforge_forceaware import mppi_weights

    with pytest.raises(ValueError):
        mppi_weights([1.0, 2.0], temperature)


@pytest.mark.parametrize("costs,valid", [
    ([np.nan, np.inf, -np.inf], None),
    ([1.0, 2.0], [False, False]),
])
def test_no_finite_valid_rollout_is_reported_as_failure(costs, valid):
    from planning.dexforge_forceaware import mppi_weights

    with pytest.raises(FloatingPointError, match="all MPPI candidates are invalid"):
        mppi_weights(costs, 0.5, valid=valid)


def test_weighted_update_preserves_knot_shape_and_ignores_failed_samples():
    from planning.dexforge_forceaware import mppi_weighted_mean, mppi_weights

    samples = np.arange(4 * 3 * 14, dtype=float).reshape(4, 3, 14) / 100.0
    costs = np.array([1.0, 1.4, 2.0, np.nan])
    samples[3] = np.nan
    expected = np.tensordot(mppi_weights(costs[:3], 0.3), samples[:3], axes=(0, 0))
    update = mppi_weighted_mean(samples, costs, temperature=0.3)
    assert update.shape == (3, 14)
    assert np.isfinite(update).all()
    np.testing.assert_allclose(update, expected)
    # Reordering worlds cannot change the update or privilege a world index.
    reorder = [3, 1, 0, 2]
    np.testing.assert_allclose(
        mppi_weighted_mean(samples[reorder], costs[reorder], temperature=0.3), expected)


def test_weighted_mean_does_not_include_masked_outlier_controls():
    from planning.dexforge_forceaware import mppi_weighted_mean

    samples = np.array([[0.02, -0.03], [1.e9, -1.e9], [0.02, -0.03]])
    update = mppi_weighted_mean(samples, [2.0, -1.e12, 2.0], temperature=0.2,
                               valid=[True, False, True])
    np.testing.assert_allclose(update, samples[0])


def _warp_optimizer():
    """Run real sampling/update kernels with a small quadratic rollout cost."""
    pytest.importorskip("warp")
    from planning.dexforge_forceaware import _classes

    try:
        api = _classes()
    except ImportError as error:
        pytest.skip(f"DexForge runtime unavailable: {error}")
    wp = api.wp
    wp.init()
    solver = api.PandaMppiWindowOptimizer.__new__(api.PandaMppiWindowOptimizer)
    solver.device = "cpu"
    worlds, actions, horizon = 8, 14, 2
    width = actions * (horizon + 1)
    solver.index = SimpleNamespace(action_dim=actions)
    solver.settings = SimpleNamespace(worlds=worlds, horizon=horizon, iterations=3, execution_substeps=2)
    solver.control = SimpleNamespace(
        raw=wp.zeros((worlds, width), dtype=float, device="cpu"),
        target_actions=tuple(wp.zeros(actions, dtype=float, device="cpu") for _ in range(horizon)),
        gradient=wp.zeros((worlds, width), dtype=float, device="cpu"),
        gradient_norm=wp.zeros(worlds, dtype=float, device="cpu"),
        valid=wp.zeros(worlds, dtype=int, device="cpu"),
        failure_iteration=wp.zeros(worlds, dtype=int, device="cpu"),
        failure_code=wp.zeros(worlds, dtype=int, device="cpu"),
        best_raw=wp.zeros((worlds, width), dtype=float, device="cpu"),
        best_available=wp.zeros(worlds, dtype=int, device="cpu"),
        best_loss=wp.empty(worlds, dtype=float, device="cpu"),
        best_grad_norm=wp.zeros(worlds, dtype=float, device="cpu"),
        last_evaluated_raw=wp.zeros((worlds, width), dtype=float, device="cpu"),
        last_gradient_norm=wp.zeros(worlds, dtype=float, device="cpu"))
    solver.loss = SimpleNamespace(total=wp.zeros(worlds, dtype=float, device="cpu"))
    solver.mean = wp.zeros(width, dtype=float, device="cpu")
    solver.iteration = wp.zeros(1, dtype=int, device="cpu")
    solver.random_seed = wp.zeros(1, dtype=int, device="cpu")
    solver.baseline = wp.zeros(1, dtype=float, device="cpu")
    solver.weights = wp.zeros(worlds, dtype=float, device="cpu")
    solver.statistics = wp.zeros(3, dtype=float, device="cpu")
    solver.action_lower = wp.array(np.full((horizon, actions), -0.04, np.float32), device="cpu")
    solver.action_upper = wp.array(np.full((horizon, actions), 0.04, np.float32), device="cpu")
    solver.mppi_iterations, solver.mppi_init_iterations = 2, 3
    solver.temperature, solver.noise_scale, solver.noise_decay = 0.003, 0.025, 0.75
    solver.seed, solver.solve_count = 31, 0
    solver.rng = np.random.default_rng(solver.seed)
    solver.graphs = {False: object()}
    solver.evaluations = []
    solver.final_failure = None
    solver.final_mean_penalty = 0.0
    solver.sample_failure = None
    solver._clear_contact_history = lambda: None
    solver._commit_regularization_history = lambda: None
    solver._ensure_graph = lambda **kwargs: False
    solver._candidate_drifts = lambda *args: np.zeros(worlds, np.float32)
    solver._hard_simulator_valid = lambda: np.ones(worlds, bool)

    def prepare(qpos, qvel, targets, **kwargs):
        solver.control.best_loss.fill_(1.e30)
        solver.control.best_available.zero_()

    def evaluate():
        raw = solver.control.raw.numpy()
        # A target inside the position-control bounds creates unequal costs.
        costs = np.square(raw[:, actions:] - 0.015).sum(axis=1).astype(np.float32)
        valid = np.ones(worlds, np.int32)
        is_final = int(solver.iteration.numpy()[0]) == (
            solver.mppi_init_iterations if solver.solve_count == 0 else solver.mppi_iterations)
        if is_final:
            # Model a nonlinear contact loss at the average trajectory while
            # keeping every state/control finite and therefore valid.
            costs[0] += solver.final_mean_penalty
        if not is_final and solver.sample_failure is not None:
            valid[solver.sample_failure], costs[solver.sample_failure] = 0, -1.e6
        if is_final and solver.final_failure is not None:
            indices = [0] if solver.final_failure == "mean" else np.arange(worlds)
            valid[indices], costs[indices] = 0, np.nan
        solver.control.valid.assign(valid)
        solver.control.last_evaluated_raw.assign(raw)
        solver.loss.total.assign(costs)
        solver.evaluations.append((raw.copy(), costs.copy(), valid.copy()))

    def plan(selected, started, *, evaluation, execution_steps):
        raw = solver.control.raw.numpy()[selected]
        return SimpleNamespace(raw=raw.copy(), loss=float(evaluation.scores[selected]),
                               action=raw[actions:2 * actions].copy(), diag={"selected": selected})

    solver._prepare_window = prepare
    solver._evaluate_batch = evaluate
    solver._plan = plan
    return solver, api


def _solve_cpu(solver, api, warm):
    # CPU executes the graph body directly; the production CUDA path captures
    # precisely this same _iteration body. All its Warp kernels remain real.
    with api.wp.ScopedDevice("cpu"), patch.object(
            api.wp, "capture_launch", side_effect=lambda graph: solver._iteration(allow_init=False)):
        return solver.solve(np.zeros(21), np.zeros(20), None, warm=warm,
                            previous_action=np.zeros(14), allow_init=False, seed=0,
                            execution_steps=2, start_time=0.0)


def test_native_warp_mppi_update_is_bounded_warm_and_seeded():
    solver, api = _warp_optimizer()
    warm = np.r_[np.full(14, 0.9), np.full(14, 0.08), np.full(14, -0.015)].astype(np.float32)
    first = _solve_cpu(solver, api, warm)
    first_evaluations = [values.copy() for values, _, _ in solver.evaluations]
    assert first.diag["mppi_iterations"] == 3
    assert first.diag["selected_mean"]
    assert not first.diag["adjoint_used"]
    assert solver.solve_count == 1
    for values in first_evaluations:
        np.testing.assert_array_equal(values[:, :14], np.zeros((8, 14)))
        assert np.all(np.abs(values[:, 14:]) <= 0.040001)
    # World zero starts at the warm center, after physical action projection.
    np.testing.assert_allclose(first_evaluations[0][0, 14:28], 0.04)
    np.testing.assert_allclose(first_evaluations[0][0, 28:], -0.015)
    assert np.any(first_evaluations[0][1:, 14:] != first_evaluations[0][0, 14:])
    np.testing.assert_allclose(first.diag["mppi_weights"].sum(), 1.0, atol=1.e-6)
    from planning.dexforge_forceaware import mppi_weighted_mean, mppi_weights
    np.testing.assert_allclose(first.diag["mppi_weights"],
                               mppi_weights(first.diag["sampled_costs"], solver.temperature),
                               atol=1.e-7, rtol=1.e-5)
    expected_mean = mppi_weighted_mean(first_evaluations[-2], first.diag["sampled_costs"],
                                       temperature=solver.temperature)
    np.testing.assert_allclose(first.raw, expected_mean, atol=1.e-7, rtol=1.e-5)
    np.testing.assert_array_equal(first.raw, solver.evaluations[-1][0][0])
    assert first.loss == pytest.approx(solver.evaluations[-1][1][0])
    second = _solve_cpu(solver, api, warm)
    assert second.diag["mppi_iterations"] == 2
    assert solver.solve_count == 2
    solver.reset()
    assert solver.solve_count == 0
    np.testing.assert_array_equal(solver.mean.numpy(), np.zeros(42))
    solver.evaluations.clear()
    repeat = _solve_cpu(solver, api, warm)
    np.testing.assert_array_equal(repeat.raw, first.raw)
    for values, expected in zip(solver.evaluations, first_evaluations):
        np.testing.assert_array_equal(values[0], expected)


def test_native_warp_invalid_final_mean_uses_evaluated_valid_fallback():
    solver, api = _warp_optimizer()
    solver.final_failure = "mean"
    plan = _solve_cpu(solver, api, np.zeros(42, np.float32))
    selected = plan.diag["selected"]
    assert selected != 0
    assert not plan.diag["selected_mean"]
    assert np.isfinite(plan.loss)
    final_raw, final_costs, final_valid = solver.evaluations[-1]
    assert not final_valid[0]
    assert final_valid[selected]
    np.testing.assert_array_equal(plan.raw, final_raw[selected])
    assert plan.loss == pytest.approx(final_costs[selected])


def test_native_warp_valid_but_worse_mean_does_not_discard_lower_cost_contact_candidate():
    solver, api = _warp_optimizer()
    solver.final_mean_penalty = 100.0
    plan = _solve_cpu(solver, api, np.zeros(42, np.float32))
    selected = plan.diag["selected"]
    final_raw, final_costs, final_valid = solver.evaluations[-1]
    assert final_valid[0]  # Finiteness alone must not privilege the mean.
    assert selected != 0
    assert not plan.diag["selected_mean"]
    assert plan.loss == pytest.approx(np.min(final_costs[final_valid.astype(bool)]))
    assert plan.loss < plan.diag["mean_cost"]
    assert plan.diag["best_candidate_cost"] == pytest.approx(plan.loss)
    np.testing.assert_array_equal(plan.raw, final_raw[selected])


def test_native_warp_all_invalid_final_candidates_fail_clearly():
    solver, api = _warp_optimizer()
    solver.final_failure = "all"
    with pytest.raises(FloatingPointError, match="all multistart candidates"):
        _solve_cpu(solver, api, np.zeros(42, np.float32))


def test_native_warp_sampling_ignores_low_cost_invalid_rollout():
    from planning.dexforge_forceaware import mppi_weights

    solver, api = _warp_optimizer()
    solver.sample_failure = 1
    plan = _solve_cpu(solver, api, np.zeros(42, np.float32))
    assert plan.diag["mppi_weights"][1] == 0.0
    assert plan.diag["valid_samples"] == 7
    np.testing.assert_allclose(plan.diag["mppi_weights"],
                               mppi_weights(plan.diag["sampled_costs"], solver.temperature,
                                            valid=np.arange(8) != 1),
                               atol=1.e-7, rtol=1.e-5)


def test_native_warp_far_ca_reference_does_not_collapse_bounded_exploration():
    solver, api = _warp_optimizer()
    solver.control.target_actions = tuple(
        api.wp.array(np.full(14, 2.0, np.float32), device="cpu") for _ in range(2))
    solver.action_lower.assign(np.full((2, 14), -0.15, np.float32))
    solver.action_upper.assign(np.full((2, 14), 0.15, np.float32))
    solver.noise_scale = 0.005
    _solve_cpu(solver, api, np.zeros(42, np.float32))
    absolute = solver.evaluations[0][0][:, 14:] + 2.0
    assert np.all(absolute >= -0.150001)
    assert np.all(absolute <= 0.150001)
    np.testing.assert_allclose(absolute[0], 0.15, atol=1.e-7)
    assert np.any(absolute[1:] < 0.149)
    assert np.max(np.ptp(absolute, axis=0)) > 0.001


@pytest.mark.parametrize("cost_scale", [1.e-6, 1.0])
def test_native_warp_softmax_uses_shared_loss_scale_without_scaling_reported_cost(cost_scale):
    from planning.dexforge_forceaware import mppi_weights

    solver, api = _warp_optimizer()
    from forceaware.numerics import LOSS_GRADIENT_SCALE

    assert float(LOSS_GRADIENT_SCALE) == pytest.approx(1.e-6)
    solver.cost_scale = cost_scale
    solver.temperature = 0.003
    plan = _solve_cpu(solver, api, np.zeros(42, np.float32))
    expected = mppi_weights(plan.diag["sampled_costs"] * solver.cost_scale, solver.temperature)
    np.testing.assert_allclose(plan.diag["mppi_weights"], expected, rtol=1.e-5, atol=1.e-7)
    if cost_scale == 1.e-6:
        assert plan.diag["effective_samples"] > 7.5
    assert plan.diag["mppi_cost_scale"] == pytest.approx(cost_scale)
    assert plan.diag["mppi_effective_temperature"] == pytest.approx(0.003 / cost_scale)
    # The returned loss remains the native FA objective of the executed mean.
    raw_cost = np.square(plan.raw[14:] - 0.015).sum()
    assert raw_cost > 1.e-4
    assert plan.loss == pytest.approx(raw_cost, rel=1.e-6)
    if cost_scale != 1.0:
        assert plan.loss != pytest.approx(raw_cost * solver.cost_scale, rel=1.e-3, abs=1.e-12)
