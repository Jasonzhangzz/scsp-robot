"""Isaac flip fixes: OSC press cap, table plane, raised-table ranking."""

import numpy as np
from scipy.spatial.transform import Rotation

import inspect

from examples.mpc.fingertips.test.test_0902 import (
    _should_hold_occupied_contact,
    _travel_verify_cost,
    compute_rollout_contact_via,
)
from examples.mpc.franka.ik2.contact_frames import (
    FRANKA_QD_NULLSPACE,
    _contact_jacobian_np,
    clip_via_target,
    contact_aware_task_force,
    diagnose_contact_pose_source,
    franka_nullspace_posture_torque,
    mpc_action_track_accel,
    mpc_ball_contact_force,
    mpc_ball_trajectory,
    mpc_ball_velocity_ref,
    near_press_force_mode,
    planar_table_jacobians,
    planar_table_support_local,
    press_normal_outward,
    strip_inward_press,
    rollout_task_accel,
    tangent_basis_from_normal,
    wrap_joint_error,
)
from examples.mpc.franka.ik2.params import (
    _actor_quat_from_body,
    _box_inertia_diag,
    _ground_rotation_target_q,
    _source_standing_quat_wxyz,
    _xml_obj_geom_quat_wxyz,
)
from utils import rotations


def test_compute_rollout_contact_via_accepts_support_plane():
    # handle_mpc_request always forwards support_point/support_normal,
    # including None on a flat Isaac table.  Missing kwargs raise TypeError
    # in the planner worker before ranking starts.
    params = inspect.signature(compute_rollout_contact_via).parameters
    assert "support_point" in params
    assert "support_normal" in params
    inspect.signature(compute_rollout_contact_via).bind_partial(
        floor_ground=0.4,
        floor_z=0.4,
        support_point=np.array([0.0, 0.0, 0.4]),
        support_normal=np.array([0.0, 0.0, 1.0]),
    )


def test_blocked_travel_zeros_contact_verify():
    assert _travel_verify_cost(1.0, path_blocked=True) == 0.0
    assert _travel_verify_cost(0.7, path_blocked=False) == 0.7


def test_strip_inward_press_keeps_lift_and_tangent():
    n_out = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    punch = strip_inward_press(
        np.array([1.0, 0.0, -2.0], dtype=np.float32), n_out)
    assert abs(float(punch[2])) < 1e-6
    assert abs(float(punch[0]) - 1.0) < 1e-5
    lift = strip_inward_press(
        np.array([0.5, 0.0, 2.0], dtype=np.float32), n_out)
    assert lift[2] > 1.9
    assert abs(float(lift[0]) - 0.5) < 1e-5


def test_hold_occupied_when_best_is_across_the_com():
    class _Opt:
        point_curvature = None

        @staticmethod
        def pose_delta_for_sample(idx):
            return {14: 1.6, 18: 0.0, 19: 3.05, 30: 3.07, 31: 4.5}.get(int(idx))

    tip = np.array([0.296, 0.149, 0.381])
    obj = np.array([0.358, 0.138, 0.380])
    far = np.array([0.399, 0.107, 0.368])
    near = np.array([0.319, 0.181, 0.381])
    # Table-J near-tie: stay on the occupied graze.
    assert _should_hold_occupied_contact(
        _Opt(), tip, obj, far, 19, best_idx=30)
    # Clearly better far contact must be allowed to switch.
    assert not _should_hold_occupied_contact(
        _Opt(), tip, obj, far, 14, best_idx=31)
    assert not _should_hold_occupied_contact(_Opt(), tip, obj, near, 14)
    assert not _should_hold_occupied_contact(_Opt(), tip, obj, far, 18)


def test_numpy_contact_jacobian_is_4x_nv_and_avoids_jax():
    n, t1, t2 = tangent_basis_from_normal([0.0, 0.0, 1.0])
    assert np.allclose(n, [0.0, 0.0, 1.0], atol=1e-6)
    assert abs(float(np.dot(t1, t2))) < 1e-6
    j_rel = np.zeros((3, 9), dtype=np.float64)
    j_rel[:, :3] = np.eye(3)
    jac = _contact_jacobian_np(n, t1, t2, j_rel, 0.5)
    assert jac.shape == (4, 9)
    assert np.isfinite(jac).all()


def test_near_press_uses_contact_scale_before_physx_touch():
    tip = np.array([0.50, 0.10, 0.37])
    press = np.array([0.48, 0.10, 0.37])
    assert near_press_force_mode(False, tip, press=press)
    assert not near_press_force_mode(
        False, tip, press=np.array([0.40, 0.10, 0.37]))
    # Orbiting far from press must keep the air-force budget.
    assert not near_press_force_mode(
        False, np.array([0.30, 0.10, 0.42]), press=press)
    n_out = press_normal_outward(tip, press)
    punched = contact_aware_task_force(
        np.array([-40.0, 0.0, 0.0], dtype=np.float32),
        np.array([-0.5, 0.0, 0.0], dtype=np.float32),
        n_out,
    )
    assert punched[0] >= -2.0 - 1e-5
    assert abs(float(punched[0]) + 0.5) < 1e-4


def test_contact_force_caps_inward_press_not_lift_or_tangent():
    n_out = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    force_contact = np.array([0.3, 0.0, -0.4], dtype=np.float32)

    # A lift via must follow the track force.  Rewriting it into min_press
    # pins the tip and kills flip follow-through.
    lift = contact_aware_task_force(
        np.array([20.0, 0.0, 80.0], dtype=np.float32), force_contact, n_out)
    assert lift[2] > 0.0
    assert abs(float(lift[0]) - 20.0) < 1e-5

    # Inward punch is capped to the ball scale; tangent stays on track.
    press = contact_aware_task_force(
        np.array([20.0, 0.0, -80.0], dtype=np.float32), force_contact, n_out)
    assert -2.0 - 1e-5 <= press[2] <= -0.4 + 1e-5
    assert abs(float(press[0]) - 20.0) < 1e-5

    # A reached via still keeps a small inward press so the tip does not
    # drop off mid-flip.
    hold = contact_aware_task_force(np.zeros(3, dtype=np.float32), np.zeros(3), n_out)
    assert hold[2] < 0.0
    assert abs(float(hold[2])) >= 0.5 - 1e-5

    # Arrived + ball command: push in the action direction, not only -n.
    arrived = contact_aware_task_force(
        np.zeros(3, dtype=np.float32),
        np.array([0.3, 0.0, 0.4], dtype=np.float32),
        n_out,
    )
    assert arrived[0] > 0.2
    assert arrived[2] > 0.0


def test_table_support_uses_world_plane_not_mesh_minus_z():
    obj_pos = np.array([0.4, 0.0, 0.375], dtype=np.float64)
    table_height = 0.35
    identity = np.eye(3)
    local_z_up = planar_table_support_local(obj_pos, identity, table_height)
    assert np.allclose(local_z_up, [0.0, 0.0, -0.025], atol=1e-6)

    # Source-STL standing: Ry(-90) maps +X to +Z, so the table is -X.
    r_x_up = Rotation.from_euler("y", -90.0, degrees=True).as_matrix()
    local_x_up = planar_table_support_local(obj_pos, r_x_up, table_height)
    assert np.allclose(local_x_up, [-0.025, 0.0, 0.0], atol=1e-6)
    assert abs(float(local_x_up[2])) < 1e-6


def test_planar_table_jacobian_is_a_single_world_up_constraint():
    obj_pos = np.array([0.4, 0.05, 0.375], dtype=np.float32)
    rot = Rotation.from_euler("y", -90.0, degrees=True).as_matrix()
    con_jac, con_jac_body, local = planar_table_jacobians(
        obj_pos, rot, 0.35, nv=9, mu=0.5)
    assert con_jac.shape[0] == 4
    assert con_jac_body.shape == (4, 6)
    assert np.isfinite(con_jac).all()
    assert np.isfinite(con_jac_body).all()
    assert np.allclose(local, [-0.025, 0.0, 0.0], atol=1e-5)


def test_source_standing_quat_matches_xml_geom_plus_90y():
    geom = _xml_obj_geom_quat_wxyz("envs/xmls/env_fingertips_foam_brick.xml")
    standing = _source_standing_quat_wxyz("envs/xmls/env_fingertips_foam_brick.xml")
    assert np.allclose(standing, geom, atol=1e-8)
    rot = Rotation.from_quat([standing[1], standing[2], standing[3], standing[0]]).as_matrix()
    # XML +90 Y sends authored +X to world -Z (feet down).  Ry(-90) inverts it.
    assert np.allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 0.0, -1.0], atol=1e-6)
    upside_down = rotations.rpy_to_quaternion(np.array([0.0, -0.5 * np.pi, 0.0]))
    assert float(np.abs(np.dot(standing, upside_down))) < 0.1


def test_ground_rotation_target_matches_fingertips_0902():
    roll = 0.25
    body_q = rotations.rpy_to_quaternion(np.hstack([0.0, -0.5 * np.pi, np.pi * roll - 0.5 * np.pi]))
    goal_geom = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])
    expected = rotations.quaternion_multiply(body_q, goal_geom)
    got = _ground_rotation_target_q(roll)
    assert np.allclose(np.abs(np.dot(got, expected)), 1.0, atol=1e-8)
    # A standing-only yaw target stays near identity after the geom offset.
    # The flip target must be about 90 deg away from that family.
    standing = rotations.quaternion_multiply(
        rotations.rpy_to_quaternion(np.array([0.0, 0.0, 0.0])), goal_geom
    )
    quat_err = 1.0 - float(np.dot(got, standing)) ** 2
    assert quat_err > 0.4


def test_extracted_mesh_keeps_mujoco_body_quat():
    body_q = _ground_rotation_target_q(0.4)
    standing = _source_standing_quat_wxyz("envs/xmls/env_fingertips_foam_brick.xml")
    assert np.allclose(_actor_quat_from_body(body_q, standing, True), body_q)
    mapped = _actor_quat_from_body(body_q, standing, False)
    expected = rotations.quaternion_multiply(body_q, standing)
    assert np.allclose(np.abs(np.dot(mapped, expected)), 1.0, atol=1e-8)


def test_mujoco_hull_extracts_foam_brick_body_frame():
    from examples.mpc.fingertips.test.params import _mujoco_collision_mesh

    source = "envs/assets/objects/foam_brick.stl"
    extracted = _mujoco_collision_mesh(source, "envs/xmls/env_fingertips_foam_brick.xml")
    assert extracted != source


def test_box_inertia_scales_with_mass_not_urdf_1e4():
    inertia = _box_inertia_diag(
        0.01,
        np.array([-0.025, -0.02, -0.02]),
        np.array([0.025, 0.02, 0.02]),
    )
    assert np.all(inertia < 5e-6)
    assert np.all(inertia > 1e-7)
    double = _box_inertia_diag(
        0.02,
        np.array([-0.025, -0.02, -0.02]),
        np.array([0.025, 0.02, 0.02]),
    )
    assert np.allclose(double, 2.0 * inertia, atol=1e-12)


def test_mpc_ball_velocity_is_action_over_horizon():
    action = np.array([0.005, 0.0, 0.0], dtype=np.float64)
    v_ref = mpc_ball_velocity_ref(action, 0.02)
    assert np.allclose(v_ref, [0.25, 0.0, 0.0], atol=1e-8)


def test_mpc_ball_contact_force_dumps_planned_momentum():
    action = np.array([0.005, 0.0, 0.0], dtype=np.float64)
    v_ref = mpc_ball_velocity_ref(action, 0.02)
    at_speed = mpc_ball_contact_force(action, v_ref, 100.0, 2.0)
    stalled = mpc_ball_contact_force(action, np.zeros(3), 100.0, 2.0)
    assert np.allclose(at_speed, [0.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(stalled, [0.5, 0.0, 0.0], atol=1e-6)
    # Killing v_ref produces the same impulse rate as K u.
    assert np.allclose(stalled - at_speed, 2.0 * v_ref, atol=1e-6)


def test_mpc_ball_trajectory_is_constant_velocity_then_hold():
    p0 = np.zeros(3)
    action = np.array([0.005, 0.0, 0.0], dtype=np.float64)
    p_mid, v_mid = mpc_ball_trajectory(p0, action, 0.01, 0.02)
    assert np.allclose(p_mid, [0.0025, 0.0, 0.0], atol=1e-6)
    assert np.allclose(v_mid, [0.25, 0.0, 0.0], atol=1e-6)
    p_end, v_end = mpc_ball_trajectory(p0, action, 0.02, 0.02)
    assert np.allclose(p_end, [0.005, 0.0, 0.0], atol=1e-6)
    assert np.allclose(v_end, [0.0, 0.0, 0.0], atol=1e-6)


def test_mpc_action_track_accel_is_finite_for_5mm_step():
    acc = mpc_action_track_accel([0.0025, 0.0, 0.0], [0.0, 0.0, 0.0], 0.02)
    assert acc.shape == (3,)
    assert np.isfinite(acc).all()
    assert float(acc[0]) > 0.0


def test_franka_nullspace_posture_pulls_toward_start_config():
    # Official start: {0, -π/4, 0, -3π/4, 0, π/2, π/4}.
    q_d = FRANKA_QD_NULLSPACE
    assert np.allclose(
        q_d, [0.0, -0.25 * np.pi, 0.0, -0.75 * np.pi, 0.0, 0.5 * np.pi, 0.25 * np.pi]
    )
    jac = np.zeros((3, 7), dtype=np.float64)
    jac[0, 0] = 1.0
    jac[1, 1] = 1.0
    jac[2, 2] = 1.0
    at_home = franka_nullspace_posture_torque(jac, q_d, np.zeros(7), q_d, 10.0)
    assert np.allclose(at_home, 0.0, atol=1e-6)

    q = q_d.copy()
    q[6] += 0.3
    tau = franka_nullspace_posture_torque(jac, q, np.zeros(7), q_d, 10.0)
    # Wrist is in the position nullspace, so it must restore q7 toward q_d.
    assert float(tau[6]) < 0.0
    # Task joints must not be driven by the posture term.
    assert np.allclose(tau[:3], 0.0, atol=1e-6)
    assert np.allclose(wrap_joint_error(q_d, q)[6], -0.3, atol=1e-8)


def test_clip_via_target_uses_live_tip_not_stale_via():
    tip = np.array([0.40, 0.00, 0.36])
    stale_via = np.array([0.37, 0.00, 0.36])
    target, action = clip_via_target(tip, stale_via, max_step=0.005)
    assert abs(float(np.linalg.norm(target - tip)) - 0.005) < 1e-6
    assert abs(float(np.linalg.norm(action)) - 0.005) < 1e-6
    assert target[0] < tip[0]
    near = tip + np.array([0.002, 0.0, 0.0])
    target_near, _ = clip_via_target(tip, near, max_step=0.005)
    assert np.allclose(target_near, near)


def test_clip_via_target_tracks_planner_action():
    tip = np.array([0.40, 0.00, 0.36])
    via = tip + np.array([0.008, 0.0, 0.0])
    action = np.array([-0.003, 0.004, 0.0])
    target, increment = clip_via_target(tip, via, action=action, max_step=0.005)
    assert np.allclose(increment, action, atol=1e-6)
    assert np.allclose(target, tip + action, atol=1e-6)
    big = np.array([0.02, 0.0, 0.0])
    target_big, increment_big = clip_via_target(
        tip, via, action=big, max_step=0.005)
    assert abs(float(np.linalg.norm(increment_big)) - 0.005) < 1e-6
    assert increment_big[0] > 0.0
    assert np.allclose(target_big, tip + increment_big, atol=1e-6)
    target_via, _ = clip_via_target(tip, via, max_step=0.005)
    assert abs(float(np.linalg.norm(target_via - tip)) - 0.005) < 1e-6
    assert target_via[0] > tip[0]


def test_rollout_task_accel_does_not_fight_the_increment_velocity():
    err = np.array([0.005, 0.0, 0.0])
    v_ref = mpc_ball_velocity_ref(err, 0.02)
    cruise = rollout_task_accel(err, v_ref, v_ref=v_ref, k_task=100.0, d_task=2.0)
    fight = rollout_task_accel(err, v_ref, v_ref=None, k_task=100.0, d_task=2.0)
    assert cruise[0] > 0.0
    assert fight[0] < cruise[0]
    done = rollout_task_accel(np.zeros(3), v_ref, v_ref=v_ref, k_task=100.0, d_task=2.0)
    assert np.allclose(done, 0.0, atol=1e-6)


def test_mpc_via_reference_is_interpolated_across_the_horizon():
    p0 = np.array([0.40, 0.00, 0.36])
    action = np.array([0.005, 0.0, 0.0])
    p_mid, v_mid = mpc_ball_trajectory(p0, action, 0.01, 0.02)
    assert np.allclose(p_mid, [0.4025, 0.00, 0.36], atol=1e-6)
    assert np.allclose(v_mid, [0.25, 0.0, 0.0], atol=1e-6)


def test_contact_pose_source_is_lambda_when_occupied_improves():
    tip = np.array([0.38, 0.00, 0.36])
    nearest = tip.copy()
    best = np.array([0.45, 0.02, 0.36])
    diag = diagnose_contact_pose_source(
        occupied_delta=0.04, best_delta=0.12, executed_delta=0.04,
        tip=tip, p_arm_world=nearest, best_contact_world=best,
    )
    assert diag["source"] == "lambda_feasible"
    assert diag["occupied_improves"]


def test_contact_pose_source_is_planner_when_nearest_cannot_improve():
    tip = np.array([0.38, 0.00, 0.36])
    nearest = tip.copy()
    best = np.array([0.45, 0.02, 0.36])
    diag = diagnose_contact_pose_source(
        occupied_delta=-0.01, best_delta=0.12, executed_delta=-0.01,
        tip=tip, p_arm_world=nearest, best_contact_world=best,
        osc_target=nearest,
    )
    assert diag["source"] == "planner_attract"
    assert not diag["occupied_improves"]
    assert diag["best_improves"]


def test_contact_pose_source_is_osc_when_tracking_misses_both_patches():
    tip = np.array([0.38, 0.00, 0.36])
    p_arm = np.array([0.45, 0.02, 0.36])
    best = np.array([0.45, 0.02, 0.36])
    via = np.array([0.45, 0.02, 0.36])
    diag = diagnose_contact_pose_source(
        occupied_delta=-0.02, best_delta=0.12, executed_delta=0.12,
        tip=tip, p_arm_world=p_arm, best_contact_world=best, osc_target=via,
    )
    assert diag["source"] == "osc_tracking"
    assert diag["osc_track"] > 0.015
