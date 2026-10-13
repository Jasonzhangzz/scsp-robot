"""Goal-driven dual contact predictions and the action actually sent to OSC."""

from types import SimpleNamespace

import numpy as np
import pytest

from models.support_geometry import support_jacobians
from planning.bimanual_online_mpc import BimanualOnlineMPC


def params(solver="ipopt", horizon=3):
    return SimpleNamespace(h_=.02, mpc_horizon_=horizon, planner_solver_=solver,
        planner_cmd_limit=.01, online_task_stiffness=600., contact_command_force_limit_=12.,
        mpc_tip_effective_mass_=2.5, mpc_tip_damping_=40., soft_scm_response_time=.05,
        soft_scm_damping_ratio=1., soft_scm_regularization=1e-8, mpc_patch_radius_=.03,
        support_surface_point_=np.zeros(3), support_surface_half_extents_=np.ones(2),
        online_contact_offset=.007, tip_radius=.01, planner_ee_position_weight_=80.,
        planner_object_lateral_weight_=300., planner_object_target_weight_=300.,
        planner_object_orientation_weight_=15., planner_synchronization_weight_=25.,
        online_object_velocity_weight=.05, planner_contact_gate_sync_weight_=25.,
        planner_contact_gate_scale_=.004, planner_force_tracking_weight_=12.,
        planner_workspace_lower_=(-1., -1., 0.), planner_workspace_upper_=(2., 1., 2.),
        planner_workspace_weight_=50., planner_action_weight_=2.,
        planner_smooth_action_weight_=3., online_terminal_weight=2., model_params=1.,
        ipopt_max_iter_=100)


def state(left=-.06, right=.06):
    return np.r_[[0., 0., .3, 1., 0., 0., 0.], [left, 0., .3], [right, 0., .3], np.zeros(12)]


POINTS = np.array([[-.05, 0., 0.], [.05, 0., 0.]])
NORMALS = np.array([[1., 0., 0.], [-1., 0., 0.]])
INERTIA = np.diag([.1, .1, .1, .003, .003, .003])


def patches(x, support=False):
    phi, jac = np.ones(24), np.zeros((24, 12))
    for side in range(2):
        point = x[:3] + POINTS[side]
        block, _ = support_jacobians(x[:3], np.eye(3), point, 12, .9, NORMALS[side])
        block[:, 6 + 3 * side:9 + 3 * side] = -block[:, :3]
        jac[4 * side:4 * side + 4] = block
        phi[4 * side:4 * side + 4] = abs(x[7 + 3 * side]) - .05 - .01
    if support:
        point = np.array([0., 0., 0.])
        jac[8:12], _ = support_jacobians(x[:3], np.eye(3), point, 12, .9)
        phi[8:12] = 0.
    return phi, jac


@pytest.fixture(scope="module")
def mpc():
    return BimanualOnlineMPC(params())


def request(mpc, x, established=False, target=None, phi=None, jac=None, **kwargs):
    if phi is None:
        phi, jac = patches(x)
    return mpc.plan_once(np.array([0., 0., .35]) if target is None else np.asarray(target),
        np.array([1., 0., 0., 0.]), x, phi, jac, contact_points_local=POINTS,
        normals_local=NORMALS, desired_force_local=.5 * NORMALS,
        object_inertia=INERTIA, external_wrench=np.zeros(6),
        contact_established=established, selected_contact_mask=(established, established),
        approach_origin=x[:3], **kwargs)


def test_no_contact_transmits_no_force_and_free_body_carries_gravity(mpc):
    x = state(-.2, .2)
    mpc.model.set_observation(x, INERTIA, [0., 0., -.981, 0., 0., 0.])
    phi, jac = patches(x)
    command = np.r_[[.001, 0., 0.], [-.001, 0., 0.]]
    predicted = np.asarray(mpc.model.prediction_fn(x, command, phi, jac, mpc.model.runtime_parameters())).ravel()
    np.testing.assert_allclose(predicted[15], -.1962, atol=1e-10)
    np.testing.assert_allclose(predicted[2], .3 - .003924, atol=1e-10)
    np.testing.assert_allclose(mpc.model.contact_force_fn(x, command, phi, jac, mpc.model.runtime_parameters()), 0.)


def test_unilateral_push_predicts_horizontal_motion_and_other_arm_cannot_support(mpc):
    x = state(-.06, .2)
    mpc.model.set_observation(x, INERTIA, np.zeros(6))
    phi, jac = patches(x)
    command = np.array([.005, 0., 0., 0., 0., 0.])
    predicted = np.asarray(mpc.model.prediction_fn(x, command, phi, jac, mpc.model.runtime_parameters())).ravel()
    force = np.asarray(mpc.model.contact_force_fn(x, command, phi, jac, mpc.model.runtime_parameters())).reshape(2, 3)
    assert predicted[0] > 1e-9 and force[0, 0] > 0
    # The shared unilateral OSC cap reduces this push without pretending
    # that the physically active contact cannot displace the object at all.
    import copy
    from models.bimanual_rigid_contact import BimanualRigidContactModel
    unrestricted_params = copy.copy(mpc.model.param_)
    unrestricted_params.online_unilateral_force_limit = 12.
    unrestricted = BimanualRigidContactModel(unrestricted_params)
    unrestricted.set_observation(x, INERTIA, np.zeros(6))
    unbounded = np.asarray(unrestricted.prediction_fn(x, command, phi, jac, unrestricted.runtime_parameters())).ravel()
    assert predicted[0] < unbounded[0]
    np.testing.assert_allclose(force[1], 0., atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(predicted[3:7]), 1.)


def test_tangential_miss_deactivates_transported_pad_patch(mpc):
    x = state()
    phi, jac = patches(x)
    mpc.model.set_observation(x, INERTIA, np.zeros(6))
    missed = x.copy()
    missed[8] += .1
    force = mpc.model.contact_force_fn(missed, [.005, 0., 0., 0., 0., 0.], phi, jac, mpc.model.runtime_parameters())
    np.testing.assert_allclose(np.asarray(force)[:3], 0.)


def test_finite_support_stops_supporting_after_object_leaves_table(mpc):
    x = state(-.2, .2)
    phi, jac = patches(x, support=True)
    mpc.model.set_observation(x, INERTIA, [0., 0., -.981, 0., 0., 0.])
    mp = mpc.model.runtime_parameters()
    supported = np.asarray(mpc.model.prediction_fn(x, np.zeros(6), phi, jac, mp)).ravel()
    outside = x.copy()
    outside[0] = 1.2
    outside[7] += 1.2
    outside[10] += 1.2
    fallen = np.asarray(mpc.model.prediction_fn(outside, np.zeros(6), phi, jac, mp)).ravel()
    assert supported[15] > fallen[15]
    assert fallen[15] == pytest.approx(-.1962)


def test_online_approach_and_x_plus_match_executed_action(mpc):
    mpc.reset()
    x = state(-.15, .15)
    result = request(mpc, x)
    assert result["solve_status"] in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
    assert result["action"][0] > 0 and result["action"][3] < 0
    phi, jac = patches(x)
    executed_prediction = np.asarray(mpc.model.prediction_fn(x, result["action"], phi, jac,
                                                            mpc.model.runtime_parameters())).ravel()
    np.testing.assert_allclose(result["x_plus_opt"], executed_prediction[:7])
    np.testing.assert_allclose(result["rollout_q"][0], executed_prediction)
    assert np.linalg.norm(result["action"] @ mpc.param_.robot_stiff_.T) <= 12. + 1e-8
    assert result["diagnostics"]["phase"] == "approach"
    assert result["diagnostics"]["predicted_cost_reduction"] == pytest.approx(
        result["diagnostics"]["object_cost_now"] - result["diagnostics"]["object_cost_next"])


@pytest.mark.parametrize("established", [False, True])
def test_online_mpc_responds_to_final_goal_without_reference_trajectory(mpc, established):
    x = state()
    actions = []
    for sign in (1., -1.):
        mpc.reset()
        result = request(mpc, x, established=established, target=[0., sign * .03, .3])
        assert result["solve_status"] in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        actions.append(result["action"])
        assert result["diagnostics"]["predicted_cost_reduction"] > 0
        assert result["x_plus_opt"][1] * sign > 0
    assert actions[0][1] > 0 and actions[1][1] < 0
    assert actions[0][4] > 0 and actions[1][4] < 0


def test_native_acados_uses_bimanual_prediction_and_bounded_first_action(tmp_path):
    from planning.acados_env import ensure_acados_env

    ensure_acados_env()
    pytest.importorskip("acados_template")
    planner = BimanualOnlineMPC(params(solver="acados", horizon=3))
    assert planner.acados_solver_ is not None, planner._acados_init_error
    result = request(planner, state(-.15, .15))
    assert result["solver_backend"] == "acados"
    assert result["action"][0] > 0 and result["action"][3] < 0
    assert np.linalg.norm(result["action"] @ planner.param_.robot_stiff_.T) <= 12. + 1e-8
    phi, jac = patches(state(-.15, .15))
    prediction = np.asarray(planner.model.prediction_fn(state(-.15, .15), result["action"], phi, jac,
                                                       planner.model.runtime_parameters())).ravel()
    np.testing.assert_allclose(result["x_plus_opt"], prediction[:7])
    # Read the native QP matrices and preserve a replayable snapshot. A NaN
    # solver status alone must not be reported as NaN model input.
    diagnostics, _ = planner._inspect_acados_qp()
    assert diagnostics["available"] and diagnostics["finite"], diagnostics
    snapshot = tmp_path / "failure.npz"
    planner.param_.mpc_failure_snapshot_ = str(snapshot)
    planner._capture_acados_nan(state(-.15, .15), phi, jac, planner._context,
        result["u_traj"], np.vstack((state(-.15, .15), result["rollout_q"])),
        planner.param_.mpc_u_lb_, planner.param_.mpc_u_ub_)
    import casadi as cs
    with np.load(snapshot, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["curr_x"], state(-.15, .15))
        restored = cs.Function.deserialize(str(saved["prediction_fn"]))
        replay = np.asarray(restored(saved["curr_x"], result["action"], saved["phi"],
                                     saved["jac"], saved["model_params"])).ravel()
        np.testing.assert_allclose(replay, prediction)


def test_uncontacted_tip_prediction_integrates_the_executors_force_norm_cap(mpc):
    x = state(-.2, .2)
    x[19:22] = [-1., 0., 0.]
    mpc.model.set_observation(x, INERTIA, np.zeros(6))
    predicted = np.asarray(mpc.model.prediction_fn(x, np.zeros(6), np.ones(24),
        np.zeros((24, 12)), mpc.model.runtime_parameters())).ravel()
    # D*v=40 N exceeds the 12 N cap for the whole .02 s interval.
    acceleration = 12. / 2.5
    assert predicted[19] == pytest.approx(-1 + acceleration * .02)
    assert predicted[7] == pytest.approx(x[7] - .02 + .5 * acceleration * .02**2)


def test_free_space_velocity_feedforward_matches_capped_osc_and_increases_speed():
    from scipy.integrate import solve_ivp
    from models.bimanual_rigid_contact import online_tip_velocity_reference

    planner = BimanualOnlineMPC(params())
    x, command = state(-.2, .2), np.array([.006, 0., 0., -.006, 0., 0.])
    planner.model.set_observation(x, INERTIA, np.zeros(6))
    mp = planner.model.runtime_parameters()
    phi, jac = np.ones(24), np.zeros((24, 12))
    predicted = np.asarray(planner.model.prediction_fn(x, command, phi, jac, mp)).ravel()
    reference = online_tip_velocity_reference(command[:3], 1., False, .02, .25)

    def dynamics(t, value):
        velocity = value[3:]
        force = 600. * command[:3] - 40. * (velocity - reference)
        force *= min(1., 12. / max(np.linalg.norm(force), 1e-10))
        return np.r_[velocity, force / 2.5]

    integrated = solve_ivp(dynamics, [0., .02], np.r_[x[7:10], x[19:22]],
                          rtol=1e-10, atol=1e-12).y[:, -1]
    np.testing.assert_allclose(predicted[7:10], integrated[:3], atol=1e-10)
    np.testing.assert_allclose(predicted[19:22], integrated[3:], atol=1e-9)
    for _ in range(25):
        x = np.asarray(planner.model.prediction_fn(x, command, phi, jac, mp)).ravel()
    old_terminal_speed = 600. * command[0] / 40.
    assert x[19] > 3 * old_terminal_speed


@pytest.mark.parametrize("gap,grasp", [(1., False), (.001, False), (0., False), (-.002, False), (1., True)])
def test_contact_and_grasp_disable_feedforward_in_symbolic_and_numpy_laws(gap, grasp):
    import casadi as cs
    from models.bimanual_rigid_contact import online_tip_velocity_reference

    command = np.array([.006, -.008, 0.])
    numeric = online_tip_velocity_reference(command, gap, grasp, .02, .25)
    symbolic = np.asarray(online_tip_velocity_reference(cs.DM(command), gap, float(grasp), .02, .25)).ravel()
    np.testing.assert_allclose(symbolic, numeric)
    assert np.linalg.norm(numeric) <= .25 + 1e-10
    if grasp or gap <= 0:
        np.testing.assert_array_equal(numeric, np.zeros(3))


def test_approach_cannot_buy_contact_progress_by_dragging_object_sideways():
    p = params(horizon=4)
    p.ipopt_max_iter_ = 120  # narrow 0.1 mm soft corridor needs additional L-BFGS iterations
    p.online_max_horizontal_error = .0001
    planner = BimanualOnlineMPC(p)
    result = request(planner, state(-.06, .2))
    assert result["diagnostics"]["solver_success"]
    bounds = result["diagnostics"]["horizontal_bounds"]
    assert np.all(result["rollout_q"][:, :2] >= bounds[0] - 1e-5)
    assert np.all(result["rollout_q"][:, :2] <= bounds[1] + 1e-5)


def test_loaded_fallback_retains_current_primal_seed_and_reports_solver_failure(monkeypatch):
    import planning.mpc_explicit as explicit

    p = params()
    p.online_ipopt_fallback = True
    planner = BimanualOnlineMPC(p)
    planner.planner_solver_ = "acados"
    command = np.array([.001, 0., .001, -.001, 0., .001])
    planner._warm_controls = np.tile(command, (planner.horizon, 1))
    attempts, fallback_seeds = [], []

    def fail_acados(curr_x, phi, jac, cp, guess, lower, upper):
        attempts.append(guess)
        planner.last_acados_error = "acados status 4 HPIPM_QP=MAX_ITER"
        return None

    original_ipopt = planner._plan_once_ipopt

    def capture_ipopt(curr_x, phi, jac, cp, guess, lower, upper, **kwargs):
        fallback_seeds.append(guess)
        return original_ipopt(curr_x, phi, jac, cp, guess, lower, upper, **kwargs)

    monkeypatch.setattr(planner, "_plan_once_acados", fail_acados)
    monkeypatch.setattr(planner, "_plan_once_ipopt", capture_ipopt)
    monkeypatch.setattr(explicit, "_ACADOS_IPOPT_FALLBACK_WARNED", False)
    with pytest.warns(RuntimeWarning, match="HPIPM_QP=MAX_ITER"):
        result = request(planner, state(), established=True)
    assert len(attempts) == 2 and len(fallback_seeds) == 1
    for seed in (attempts[1], fallback_seeds[0]):
        assert seed is not None
        np.testing.assert_allclose(seed["u_traj"], np.tile(command, (planner.horizon, 1)))
        np.testing.assert_allclose(seed["x_traj"][0], state())
        assert "lam_x0" not in seed and "lam_g0" not in seed
    assert result["solver_backend"] == "ipopt" and result["diagnostics"]["solver_success"]
    assert result["x_plus_opt"][2] > state()[2]
    diagnostics = result["diagnostics"]
    assert diagnostics["acados_error"].endswith("HPIPM_QP=MAX_ITER")
    assert len(diagnostics["acados_attempt_errors"]) == 2
    assert diagnostics["ipopt_iterations"] > 0 and diagnostics["ipopt_solve_time"] > 0


def test_solver_switch_preserves_primal_without_other_backends_multipliers():
    from planning.mpc_explicit import _primal_sol_guess

    source = dict(x0=np.arange(6), u_traj=np.ones((2, 3)), x_traj=np.ones((3, 4)),
                  lam_x0=np.full(6, 1e9), lam_g0=np.full(4, -1e9), solver_backend="acados")
    seed = _primal_sol_guess(source)
    assert set(seed) == {"x0", "u_traj", "x_traj"}
    np.testing.assert_array_equal(seed["x0"], source["x0"])
    assert _primal_sol_guess(None) is None


@pytest.mark.parametrize("bad_input", [False, True])
def test_qp_nan_diagnostic_distinguishes_bad_input_from_solver_failure(bad_input):
    from planning.mpc_explicit import MPCExplicit

    class FailedSolver:
        def get_from_qp_in(self, stage, field):
            return np.array([[np.nan if bad_input and field == "A" else 1.]])

    planner = MPCExplicit.__new__(MPCExplicit)
    planner.acados_solver_ = FailedSolver()
    planner.param_ = SimpleNamespace(mpc_horizon_=1)
    info, arrays = planner._inspect_acados_qp()
    assert info["finite"] == (not bad_input)
    assert info["nonfinite_fields"] == (["qp_0_A"] if bad_input else [])
    assert "qp_0_A" in arrays


def test_loaded_grasp_disables_micrometre_gap_force_cap(mpc):
    x = state(-.059998, .060002)
    phi, jac = patches(x)
    command = [.003, 0., .003, -.003, 0., .003]
    forces = []
    for grasp in (False, True):
        mpc.model.set_observation(x, INERTIA, np.zeros(6), grasp_established=grasp)
        forces.append(np.asarray(mpc.model.contact_force_fn(
            x, command, phi, jac, mpc.model.runtime_parameters())).reshape(2, 3))
    assert forces[1][0, 0] > forces[0][0, 0] * 1.1


def test_off_anchor_grasp_and_solver_failure_keep_pressure_without_blocking_ipopt(monkeypatch):
    planner = BimanualOnlineMPC(params())
    planner.planner_solver_ = "acados"

    def fail(*args):
        planner.last_acados_error = "acados status 4 HPIPM_QP=NAN_SOL"
        return None

    monkeypatch.setattr(planner, "_plan_once_acados", fail)
    monkeypatch.setattr(planner, "_plan_once_ipopt", lambda *args, **kwargs: pytest.fail("Blocking IPOPT fallback"))
    x = state(-.057, .057)
    result = request(planner, x, grasp_established=True, measured_normal_force=[.19, .19])
    diag = result["diagnostics"]
    assert not diag["solver_success"] and diag["phase"] == "grasp_recovery"
    assert diag["grasp_established"] and not diag["selected_contact_established"]
    assert diag["fallback_control"] in {"force_hold", "shifted_plan", "previous_action"}
    assert planner._warm_controls is not None
    pressure = np.sum(result["action"].reshape(2, 3) * NORMALS, axis=1) * 600
    assert np.all(pressure > .01)
    phi, jac = patches(x)
    prediction = np.asarray(planner.model.prediction_fn(x, result["action"], phi, jac,
                                                      planner.model.runtime_parameters())).ravel()
    np.testing.assert_allclose(result["x_plus_opt"], prediction[:7])
    assert np.linalg.norm(result["action"] * 600) <= 12. + 1e-8


def test_observed_outward_momentum_outside_corridor_remains_solvable(mpc):
    mpc.reset()
    x = state(-.2, .2)
    x[0], x[13] = .02, .03
    phi, jac = patches(x)
    result = mpc.plan_once([0., 0., .3], [1., 0., 0., 0.], x, phi, jac,
        contact_points_local=POINTS, normals_local=NORMALS, desired_force_local=.5 * NORMALS,
        object_inertia=INERTIA, external_wrench=np.zeros(6), approach_origin=[0., 0., .3])
    assert result["diagnostics"]["solver_success"]
    # No contact can instantaneously cancel this measured velocity. The
    # former hard upper bound x_next <= x_current made the NLP infeasible.
    assert result["x_plus_opt"][0] == pytest.approx(.02 + .03 * mpc.param_.h_)
    assert result["diagnostics"]["predicted_horizontal_bound_violation"] > 0
