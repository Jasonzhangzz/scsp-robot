from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
import pytest

from planning.franka_contact_motion import FrankaApproachVia, UprightRelease, bounded_step
from planning.object_pose_cost import ObjectCostStages


def make_obs(t, tilt=0, tip=(.08, 0, .05), angular=0):
    q = Rotation.from_euler('y', tilt, degrees=True).as_quat()[[3, 0, 1, 2]]
    return dict(sim_time=t, policy_q=np.r_[[0., 0., .04], q, tip],
                object_linear_velocity=np.zeros(3), object_angular_velocity=[0, angular, 0])


def stages():
    return ObjectCostStages(SimpleNamespace(two_stage_object_cost=True, pos_coef=500, ori_coef=20))


def policy():
    return dict(p_arm_surface_world=np.array([.06, 0, .05]),
                p_arm_track_world=np.array([.07, 0, .05]))


def test_opposite_face_has_finite_lift_cross_drop_route_without_orbit():
    via = FrankaApproachVia(max_step=.005, max_lead=.005)
    tip, obj, goal = np.array([-.08, 0, .04]), np.array([0., 0, .04]), np.array([.07, 0, .05])
    phases = []
    for i in range(150):
        _, point, phase = via.update(tip, obj, goal, goal, .08, .09, False, goal)
        phases.append(phase)
        assert np.linalg.norm(point - tip) <= .005 + 1e-12
        assert point[1] == pytest.approx(0)
        if phase == 'lift':
            np.testing.assert_allclose(point[:2], tip[:2])
        if phase == 'cross':
            assert point[0] >= tip[0] - 1e-9
            assert point[2] >= .101 - 1e-9
        tip = point
        if np.linalg.norm(tip - goal) < .003:
            break
    assert i < 100 and set(phases) == {'lift', 'cross', 'drop'}


def test_crossing_target_is_latched_against_candidate_ranking_churn():
    via = FrankaApproachVia()
    obj, tip, right = np.array([0., 0, .04]), np.array([-.08, 0, .11]), np.array([.08, 0, .05])
    via.update(tip, obj, right, right, .08, .09, False, right)
    for i in range(15):
        other = np.array([-.08, .04 if i % 2 else -.04, .05])
        _, point, phase = via.update(tip, obj, other, other, .08, .09, False, other)
        assert phase == 'cross'
        np.testing.assert_allclose(via.goal, right)
        assert point[0] > tip[0] and point[1] == pytest.approx(0)
        tip = point
    via.reset()
    assert via.goal is None and via.via is None


def test_near_upright_withdrawal_waits_for_real_stability_and_uses_fixed_goal():
    stage, release = stages(), UprightRelease()
    first = make_obs(0, tilt=5, angular=1)
    stage.update(first)
    goal = release.update(stage, first, policy())
    assert stage.stage == 'flip'  # high velocity cannot complete the flip
    assert goal[0] > first['policy_q'][7] and goal[2] >= first['policy_q'][9]
    for t in [.1, .2]:
        current = make_obs(t, tilt=5, tip=(.09, 0, .05))
        stage.update(current)
        np.testing.assert_allclose(release.update(stage, current, policy()), goal)
    current = make_obs(.4, tilt=5)
    assert stage.update(current) and stage.stage == 'pose'
    np.testing.assert_allclose(release.update(stage, current, policy()), goal)
    assert release.update(stage, make_obs(.71, tilt=5), policy()) is None
    assert release.releases == 1


def test_lost_upright_condition_releases_settle_latch_to_resume_flip():
    stage, release = stages(), UprightRelease()
    stage.update(make_obs(0, tilt=5))
    assert release.update(stage, make_obs(0, tilt=5), policy()) is not None
    stage.update(make_obs(.1, tilt=16))
    assert release.update(stage, make_obs(.1, tilt=16), policy()) is None
    assert release.goal is None and stage.stage == 'flip'


def test_pose_guard_rejects_tipping_but_accepts_yaw_and_small_corrective_tilt():
    stage, release = stages(), UprightRelease()
    stage.update(make_obs(0, tilt=2)); stage.update(make_obs(.3, tilt=2))
    assert stage.stage == 'pose'
    assert release.unsafe_prediction(stage, [make_obs(.4, tilt=30)['policy_q']])
    upright = make_obs(.4, tilt=3)['policy_q']
    upright[3:7] = Rotation.from_euler('zy', [130, 3], degrees=True).as_quat()[[3, 0, 1, 2]]
    assert not release.unsafe_prediction(stage, [upright])
    assert release.rejected_tilts == 1
    stage.enabled = False
    assert release.update(stage, make_obs(.4), policy()) is None
    assert not release.unsafe_prediction(stage, [make_obs(.4, tilt=60)['policy_q']])


def test_step_is_bounded_and_zero_at_fixed_withdrawal_goal():
    np.testing.assert_allclose(bounded_step([0, 0, 0], [1, 0, 0], .005), [.005, 0, 0])
    np.testing.assert_allclose(bounded_step([.1, 0, 0], [.1, 0, 0], .005), 0)


def test_withdrawal_uses_occupied_surface_even_when_candidate_is_opposite():
    stage, release = stages(), UprightRelease()
    current = make_obs(0, tilt=5, tip=(-.08, 0, .05))
    stage.update(current)
    target = policy()
    target['withdrawal_axis'] = [-1., 0., 0.]
    goal = release.update(stage, current, target)
    assert goal[0] < current['policy_q'][7]


def test_measured_tilt_growth_withdraws_before_fallback_angle():
    stage, release = stages(), UprightRelease()
    stage.update(make_obs(0)); stage.update(make_obs(.3))
    release.last_stage = 'pose'
    assert release.update(stage, make_obs(.4), policy()) is None
    current = make_obs(.42, tilt=4)
    stage.update(current)
    assert release.update(stage, current, policy()) is not None
    assert stage.stage == 'pose' and stage.fallbacks == 0
    assert release.until == pytest.approx(.72)


def osc_stub():
    from examples.mpc.franka.ik2.test_mpc_isaac import IsaacFrankaOSCSimulator
    sim = IsaacFrankaOSCSimulator.__new__(IsaacFrankaOSCSimulator)
    sim._refresh_osc_tensors = lambda: None
    sim.get_current_joint_position = lambda: np.zeros(7)
    sim.get_current_joint_velocity = lambda: np.zeros(7)
    sim.get_end_effector_pos = lambda: (np.array([.4, 0, .4]), np.eye(3))
    sim._get_position_jacobian = lambda q: np.column_stack([np.eye(3), np.zeros((3, 4))])
    sim._fingertip_contact_info = lambda: (True, np.array([1., 0, 0]))
    sim._fingertip_in_physical_contact = lambda: True
    sim.get_state = lambda: np.r_[[.38, 0, .4], [1., 0, 0, 0], np.zeros(7)]
    sim.q_d_nullspace = np.zeros(7)
    sim.nullspace_stiffness = 0.
    sim.activate_tool_compensation = False
    sim.torque_limits = np.full(7, 100.)
    sim.position_d = np.array([.4, 0, .4])
    sim._mpc_action = np.array([.005, 0, 0])
    sim._last_mpc_action = np.array([.005, 0, 0])
    sim._mpc_p0 = np.array([.395, 0, .4])
    sim._mpc_T = .02
    sim._mpc_press = np.array([.4, 0, .4])
    sim._path_blocked = False
    sim.param_ = SimpleNamespace(mpc_u_ub_=.005, isaac_via_max_step_=.005,
                                isaac_action_slew_=.0025, table_height=.35)
    sim.R_d_hold = np.eye(3)
    return sim


def test_release_removes_active_contact_wrench():
    sim = osc_stub()
    sim._compute_osc_torques()
    assert np.linalg.norm(sim._last_osc_force) == pytest.approx(.8)
    sim._release_pressure = True
    sim._compute_osc_torques()
    np.testing.assert_allclose(sim._last_osc_force, 0, atol=1e-8)


def test_pose_contact_cap_is_applied_at_actual_osc_wrench():
    sim = osc_stub()
    sim._preserve_upright = True
    sim._compute_osc_torques()
    assert np.linalg.norm(sim._last_osc_force) == pytest.approx(.25)


def test_release_bypasses_slew_of_previous_inward_press():
    sim = osc_stub()
    sim._start_via_action([.395, 0, .4], action=[-.005, 0, 0],
                          press=[.4, 0, .4], release_pressure=True)
    assert sim._mpc_action[0] < 0
    assert sim._release_pressure
    assert np.linalg.norm(sim._mpc_action) <= .005 + 1e-8


@pytest.mark.parametrize('invalid', [{'policy_q': np.zeros(10)}, make_obs(float('nan'))])
def test_missing_or_nonfinite_time_interrupts_release_without_crashing(invalid):
    stage, release = stages(), UprightRelease()
    stage.update(make_obs(0, tilt=5))
    assert release.update(stage, make_obs(0, tilt=5), policy()) is not None
    assert release.update(stage, invalid, policy()) is None
    assert release.goal is None and release.last_time is None


def test_completed_action_does_not_replay_old_wrench_or_velocity_reference():
    sim = osc_stub()
    sim._mpc_action = None
    sim._compute_osc_torques()
    np.testing.assert_allclose(sim._last_osc_force, 0, atol=1e-8)
    # A held target can still correct displacement; its wrench is position PD,
    # independent of the direction/magnitude of the finished increment.
    sim.position_d = np.array([.399, 0, .4])
    sim._compute_osc_torques()
    force = sim._last_osc_force.copy()
    assert force[0] < 0
    sim._last_mpc_action = np.array([0, .005, 0])
    sim._compute_osc_torques()
    np.testing.assert_allclose(sim._last_osc_force, force, atol=1e-8)


def test_completed_free_space_action_holds_without_stale_feedforward():
    sim = osc_stub()
    sim._mpc_action = None
    sim._fingertip_contact_info = lambda: (False, np.array([1., 0, 0]))
    sim._mpc_press = None
    sim._compute_osc_torques()
    np.testing.assert_allclose(sim._last_osc_force, 0, atol=1e-8)


def test_physics_guard_reacts_before_next_planner_observation():
    from planning.franka_contact_motion import UprightExecutionGuard
    guard = UprightExecutionGuard([0., 0, 1])
    current = make_obs(0, tilt=4)
    tip = current['policy_q'][7:10]
    goal = guard.update(current['policy_q'][:7], [0, 1., 0], tip, [1., 0, 0], True, 0., True)
    assert goal is not None and goal[0] > tip[0]
    # Same goal persists on the next 2 ms physics frame, even without a new plan.
    np.testing.assert_allclose(guard.update(current['policy_q'][:7], [0, 0, 0], tip+.001,
        None, False, .002, True), goal)
    assert guard.releases == 1
    assert guard.update(current['policy_q'][:7], [0, 0, 0], goal, None, False, .31, True) is None


def test_physics_guard_ignores_world_yaw_and_uses_target_derived_axis():
    from planning.franka_contact_motion import UprightExecutionGuard
    from planning.object_pose_cost import target_up_axis
    # The goal has local +X, rather than +Z, pointing up.
    target = Rotation.from_euler('y', -90, degrees=True).as_quat()[[3, 0, 1, 2]]
    guard = UprightExecutionGuard(target_up_axis(target))
    current = np.r_[np.zeros(3), target]
    assert guard.update(current, [0, 0, 3], [.1, 0, .1], [1, 0, 0], True, 0., True) is None
    assert guard.update(current, [0, 0, 0], [.1, 0, .1], [1, 0, 0], True, .002, False) is not None


def test_outward_release_can_brake_without_normal_press_or_tangential_drag():
    from planning.franka_contact_motion import outward_release_force
    np.testing.assert_allclose(outward_release_force([6., 4, 2], [1., 0, 0]), [6., 0, 0])
    np.testing.assert_allclose(outward_release_force([-6., 4, 2], [1., 0, 0]), 0)
    np.testing.assert_allclose(outward_release_force([6., 4, 2], None), 0)


def test_actual_osc_uses_target_axis_and_outward_braking_above_press_cap():
    from planning.object_pose_cost import target_up_axis
    sim = osc_stub()
    target = Rotation.from_euler('y', -90, degrees=True).as_quat()[[3,0,1,2]]
    sim.param_.runtime_object_cost_ = True
    sim.param_.object_up_axis_ = target_up_axis(target)
    current = Rotation.from_euler('y', -86, degrees=True).as_quat()[[3,0,1,2]]
    sim.get_state = lambda: np.r_[[.38, 0, .4], current, np.zeros(7)]
    sim.get_object_velocity_world = lambda: (np.zeros(3), np.array([0, 1., 0]))
    sim.gym = SimpleNamespace(get_sim_time=lambda _sim: 0.)
    sim.sim = object()
    sim._preserve_upright = True
    sim._mpc_action = np.array([-.005, 0, 0])
    sim._mpc_p0 = np.array([.405, 0, .4])
    sim._compute_osc_torques()
    assert sim._osc_upright_guard_active
    assert sim._last_osc_force[0] > .25
    np.testing.assert_allclose(sim._last_osc_force[1:], 0, atol=1e-8)
    assert sim._osc_effective_target[0] > sim.position_d[0]
