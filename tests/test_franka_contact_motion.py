from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
import pytest

from planning.franka_contact_motion import FrankaApproachVia, SurfaceNormalApproachVia, bounded_step


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


def test_step_is_bounded_and_zero_at_goal():
    np.testing.assert_allclose(bounded_step([0, 0, 0], [1, 0, 0], .005), [.005, 0, 0])
    np.testing.assert_allclose(bounded_step([.1, 0, 0], [.1, 0, 0], .005), 0)


def test_recessed_patch_is_reached_from_outward_standoff():
    via = SurfaceNormalApproachVia(floor_z=.012)
    tip, obj = np.array([-.08, 0., .04]), np.array([0., 0., .04])
    surface, track = np.array([.05, 0., .04]), np.array([.06, 0., .04])
    press = np.array([.0595, 0., .04])
    phases = []
    for i in range(160):
        _, point, phase = via.update(tip, obj, surface, track, .08, .09, False, press)
        phases.append(phase)
        assert np.linalg.norm(point-tip) <= .005+1e-12
        if phase == 'normal':
            assert point[0] < tip[0]
            # The entry tolerance permits a small lateral tracking error;
            # the last leg must predominantly follow the outward-normal axis.
            assert abs(point[2]-tip[2]) < abs(point[0]-tip[0])
        tip = point
        if np.linalg.norm(tip-press) < .003:
            break
    assert i < 120 and 'normal' in phases


def test_completed_normal_approach_latch_allows_a_new_destination():
    via = SurfaceNormalApproachVia(floor_z=.012)
    obj, old = np.array([0., 0., .04]), np.array([.06, 0., .04])
    via.press_goal, via.anchor = old.copy(), obj.copy()
    via.entry, via.normal_approach = old+np.array([.035, 0., 0.]), True
    surface, track = np.array([0., .05, .04]), np.array([0., .06, .04])
    _, point, _ = via.update(old, obj, surface, track, .08, .09, False, track)
    np.testing.assert_allclose(via.press_goal, track)
    assert not via.normal_approach
    assert np.linalg.norm(point-old) > .001


def test_normal_approach_preserves_a_reached_moving_patch_and_clears_floor():
    via = SurfaceNormalApproachVia(floor_z=.362)
    obj = np.array([0., 0., .4])
    surface, track = np.array([.05, 0., .37]), np.array([.056, 0., .362])
    tip = np.array([.1, 0., .45])
    via.update(tip, obj, surface, track, .43, .09, False, track)
    assert via.entry[2] >= .362
    tip = track+np.array([.002, 0., 0.])
    blocked, point, _ = via.update(tip, obj, surface, track, .43, .09, True, track)
    assert not blocked
    np.testing.assert_allclose(point, track)


def osc_stub():
    import os
    # Skip the native Isaac import. These stubs never step PhysX, and a
    # prior test may already have imported torch.
    os.environ["SCSP_PLANNER_ONLY"] = "1"
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
                                isaac_action_slew_=.0025, table_height=.35, contact_force_limit_=.8)
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
    # A configured small cap applies in either objective stage.
    assert np.linalg.norm(sim._last_osc_force) == pytest.approx(.8)


@pytest.mark.parametrize('limit,expected', [(.8, .8), (10., 3.)])
def test_contact_pressure_uses_configured_cap_and_actual_cartesian_stiffness(limit, expected):
    sim = osc_stub()
    sim.param_.contact_force_limit_ = limit
    sim._compute_osc_torques()
    assert sim._last_osc_force[0] == pytest.approx(expected)
    np.testing.assert_allclose(sim._last_osc_force[1:], 0., atol=1e-8)


@pytest.mark.parametrize('tip_z,press_allowed', [(.364, True), (.361, False)])
def test_downward_pressure_is_not_disabled_three_millimetres_above_floor(tip_z, press_allowed):
    sim = osc_stub()
    tip = np.array([.4, 0., tip_z])
    sim.get_end_effector_pos = lambda: (tip, np.eye(3))
    sim._mpc_action = np.array([0., 0., -.005])
    sim._mpc_p0 = tip
    sim._mpc_press = tip
    sim.position_d = tip + sim._mpc_action
    sim.param_.table_clearance_ = .012
    sim._compute_osc_torques()
    if press_allowed:
        assert sim._last_osc_force[2] < 0.
    else:
        assert sim._last_osc_force[2] >= 0.


def test_release_bypasses_slew_of_previous_inward_press():
    sim = osc_stub()
    sim._start_via_action([.395, 0, .4], action=[-.005, 0, 0],
                          press=[.4, 0, .4], release_pressure=True)
    assert sim._mpc_action[0] < 0
    assert sim._release_pressure
    assert np.linalg.norm(sim._mpc_action) <= .005 + 1e-8


def test_completed_action_keeps_developer_velocity_reference():
    sim = osc_stub()
    sim._mpc_action = None
    sim._fingertip_contact_info = lambda: (False, None)
    sim._mpc_press = None
    sim.get_state = lambda: np.r_[np.zeros(3), [1., 0, 0, 0], np.zeros(7)]
    sim._compute_osc_torques()
    # 5 mm / 20 ms is 0.25 m/s. D * v_ref = 10 N, so the arm does not stop
    # while the next plan is still in flight.
    assert sim._last_osc_force[0] == pytest.approx(10., abs=1e-3)
    np.testing.assert_allclose(sim._last_osc_force[1:], 0, atol=1e-6)


def test_completed_free_space_action_holds_without_stale_feedforward():
    sim = osc_stub()
    sim._mpc_action = None
    sim._last_mpc_action = None
    sim._mpc_p0 = None
    sim._fingertip_contact_info = lambda: (False, np.array([1., 0, 0]))
    sim._mpc_press = None
    sim._compute_osc_torques()
    np.testing.assert_allclose(sim._last_osc_force, 0, atol=1e-8)


def test_outward_release_can_brake_without_normal_press_or_tangential_drag():
    from planning.franka_contact_motion import outward_release_force
    np.testing.assert_allclose(outward_release_force([6., 4, 2], [1., 0, 0]), [6., 0, 0])
    np.testing.assert_allclose(outward_release_force([-6., 4, 2], [1., 0, 0]), 0)
    np.testing.assert_allclose(outward_release_force([6., 4, 2], None), 0)


def test_actual_osc_executes_planned_action_without_tilt_withdrawal():
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
    assert not sim._osc_upright_guard_active
    assert sim._last_osc_force[0] < 0
    assert np.linalg.norm(sim._last_osc_force) == pytest.approx(.8)
    np.testing.assert_allclose(sim._last_osc_force[1:], 0, atol=1e-8)
    np.testing.assert_allclose(sim._osc_effective_target, sim.position_d)


@pytest.mark.parametrize('blocked', [False, True])
def test_real_pose_contact_tracks_the_five_millimetre_step(blocked):
    sim = osc_stub()
    sim._preserve_upright = True
    sim._path_blocked = blocked
    sim.position_d = np.array([.405, 0., .4])
    sim._mpc_press = np.array([.45, .1, .4])  # 11 cm from an actual touch
    sim._compute_osc_torques()
    assert sim._last_osc_near_press
    # A travel waypoint cannot bypass the force cap at a real touch.
    assert np.linalg.norm(sim._last_osc_force) == pytest.approx(.8, abs=1e-3)


def test_pose_approach_does_not_brake_before_physical_touch():
    sim = osc_stub()
    sim._preserve_upright = True
    sim._fingertip_contact_info = lambda: (False, None)
    sim._mpc_press = np.array([.45, .1, .4])
    sim.param_.runtime_object_cost_ = True
    sim.param_.object_hull_normals_ = np.r_[np.eye(3), -np.eye(3)]
    sim.param_.object_hull_offsets_ = np.array([.011]*6)
    sim.get_object_velocity_world = lambda: (np.zeros(3), np.zeros(3))
    sim.get_state = lambda: np.r_[[.2, 0, .2], [1., 0, 0, 0], np.zeros(7)]
    sim.param_.object_up_axis_ = np.array([0., 0., 1.])
    sim.gym = SimpleNamespace(get_sim_time=lambda _: 1.)
    sim.sim = None
    sim._compute_osc_torques()
    assert not sim._last_osc_near_press
    assert np.linalg.norm(sim._last_osc_force) > 2.


def test_sphere_contact_gap_derivative_predicts_press_and_release_correctly():
    from examples.mpc.franka.ik2.physx_contact import object_outward_normal
    from examples.mpc.franka.ik2.contact_frames import _contact_jacobian_np, tangent_basis_from_normal
    tip, point = np.array([.05,0,0]), np.array([.04,0,0])
    for raw in [[1,0,0],[-1,0,0]]:
        n_out = object_outward_normal(raw, tip, point)
        np.testing.assert_allclose(n_out,[1,0,0])
        obj_minus_tip = np.c_[np.eye(3),np.zeros((3,3)),-np.eye(3)]
        n,t1,t2 = tangent_basis_from_normal(-n_out)
        jac = _contact_jacobian_np(n,t1,t2,obj_minus_tip,.5)
        press = np.r_[np.zeros(6),[-.1,0,0]]
        assert (jac@press).mean() == pytest.approx(-.1)
        assert (jac@(-press)).mean() == pytest.approx(.1)
        from models.soft_scm_contact import grouped_contact_velocity_numpy, soft_scm_coefficients
        a,b = soft_scm_coefficients(.02)
        qi = np.diag([.02]*3+[20.]*3+[1/600]*3)
        after = grouped_contact_velocity_numpy(press,jac,np.zeros(4),qi,.02,a,b,1e-8)
        assert after[0] < 0  # pressing the right face moves the object left
