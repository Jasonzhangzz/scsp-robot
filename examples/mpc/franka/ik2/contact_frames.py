"""Isaac / MuJoCo contact-frame helpers that do not import Isaac Gym."""

import numpy as np


def contact_jacobian_body_frame(jacobian, body_mat):
    """Right-multiply object columns by diag(R, R) into the body frame."""
    jacobian = np.asarray(jacobian, dtype=np.float64).copy()
    if jacobian.ndim != 2 or jacobian.shape[1] < 6:
        return jacobian
    rot = np.asarray(body_mat, dtype=np.float64).reshape(3, 3)
    frame = np.zeros((6, 6), dtype=np.float64)
    frame[:3, :3] = rot
    frame[3:, 3:] = rot
    jacobian[:, :6] = jacobian[:, :6] @ frame
    return jacobian


def clip_mpc_action(action, max_step=0.005):
    action = np.asarray(action, dtype=np.float64).reshape(3)
    limit = abs(float(max_step))
    nrm = float(np.linalg.norm(action))
    if nrm > limit and nrm > 1e-9:
        action = action * (limit / nrm)
    return action.astype(np.float32)


def clip_via_target(origin, via, action=None, max_step=0.005):
    """One MPC increment from the live tip.

    When ``action`` is set, OSC tracks that clipped increment — the
    acados command.  Otherwise take one step toward ``via``.
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    if action is not None:
        increment = clip_mpc_action(action, max_step)
    else:
        via = np.asarray(via, dtype=np.float64).reshape(3)
        increment = clip_mpc_action(via - origin, max_step)
    target = origin + np.asarray(increment, dtype=np.float64).reshape(3)
    return target.astype(np.float32), increment.astype(np.float32)


def rollout_task_accel(err, vel, v_ref=None, k_task=100.0, d_task=2.0, mass=0.01):
    """Same PD as the floating ball, with velocity feedforward.

    Damping against raw ``vel`` fights the 0.25 m/s increment and
    slams the arm.  Damp ``vel - v_ref`` instead.
    """
    err = np.asarray(err, dtype=np.float64).reshape(3)
    vel = np.asarray(vel, dtype=np.float64).reshape(3)
    if v_ref is None:
        v_ref = np.zeros(3, dtype=np.float64)
    else:
        v_ref = np.asarray(v_ref, dtype=np.float64).reshape(3)
    mass = max(float(mass), 1e-6)
    return (
        (float(k_task) / mass) * err - (float(d_task) / mass) * (vel - v_ref)
    ).astype(np.float32)


def diagnose_contact_pose_source(
    occupied_delta,
    best_delta,
    executed_delta,
    tip,
    p_arm_world,
    best_contact_world,
    osc_target=None,
    improve_eps=1e-9,
    patch_radius=0.03,
    track_tol=0.015,
):
    """Classify a physical contact from lambda pose-cost reduction.

    ``lambda_feasible``: the occupied sample's ``x_plus`` lowers pose cost,
    so ranking / MPC believed this point could help.
    ``planner_attract``: the tip is on the executed attract / p_arm, but
    that sample does not reduce pose cost.
    ``osc_tracking``: neither ranked nor executed patch is nearby, or the
    OSC target is farther than a single increment.
    """
    tip = np.asarray(tip, dtype=float).reshape(3)
    p_arm = np.asarray(p_arm_world, dtype=float).reshape(3)
    best = np.asarray(best_contact_world, dtype=float).reshape(3)
    dist_arm = float(np.linalg.norm(tip - p_arm))
    dist_best = float(np.linalg.norm(tip - best))
    track_err = None
    if osc_target is not None:
        track_err = float(np.linalg.norm(
            tip - np.asarray(osc_target, dtype=float).reshape(3)))

    def _improves(delta):
        return delta is not None and np.isfinite(float(delta)) and float(delta) > float(improve_eps)

    occ_improves = _improves(occupied_delta)
    exec_improves = _improves(executed_delta)
    best_improves = _improves(best_delta)
    if occ_improves:
        source = "lambda_feasible"
    elif dist_arm <= float(patch_radius) and not exec_improves:
        source = "planner_attract"
    elif track_err is not None and track_err > float(track_tol):
        source = "osc_tracking"
    elif dist_best > float(patch_radius) and dist_arm > float(patch_radius):
        source = "osc_tracking"
    else:
        source = "planner_attract"
    return {
        "source": source,
        "occupied_delta": None if occupied_delta is None or not np.isfinite(float(occupied_delta))
        else float(occupied_delta),
        "best_delta": None if best_delta is None or not np.isfinite(float(best_delta))
        else float(best_delta),
        "executed_delta": None if executed_delta is None or not np.isfinite(float(executed_delta))
        else float(executed_delta),
        "tip_to_p_arm": dist_arm,
        "tip_to_best": dist_best,
        "osc_track": track_err,
        "occupied_improves": bool(occ_improves),
        "best_improves": bool(best_improves),
    }


def mpc_ball_velocity_ref(action, policy_dt=0.02):
    """Average velocity of the MPC sphere over one policy interval."""
    action = np.asarray(action, dtype=np.float64).reshape(3)
    return (action / max(float(policy_dt), 1e-6)).astype(np.float32)


def mpc_ball_trajectory(p0, action, t, policy_dt=0.02):
    """Constant-velocity fingertip path the 3-DoF sphere would take.

    After the horizon the reference holds ``p0 + action`` with zero speed so a
    late planner does not keep coasting.
    """
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    action = np.asarray(action, dtype=np.float64).reshape(3)
    horizon = max(float(policy_dt), 1e-6)
    t = float(np.clip(t, 0.0, horizon))
    alpha = t / horizon
    p_ref = p0 + action * alpha
    v_ref = action / horizon if t < horizon - 1e-9 else np.zeros(3, dtype=np.float64)
    return p_ref.astype(np.float32), v_ref.astype(np.float32)


def mpc_ball_contact_force(action, vel, k_task=100.0, d_task=2.0):
    """Same PD as the floating sphere: ``F = K u - D v``.

    At the free-space equilibrium ``v = (K/D) u = u / dt`` the force is 0.
    Impact kills that speed and the damping term becomes the contact
    momentum dump.
    """
    action = np.asarray(action, dtype=np.float64).reshape(3)
    vel = np.asarray(vel, dtype=np.float64).reshape(3)
    return (float(k_task) * action - float(d_task) * vel).astype(np.float32)


def mpc_action_track_accel(e_p, e_v, policy_dt=0.02):
    """Task-space accel that finishes one increment in ``policy_dt``."""
    e_p = np.asarray(e_p, dtype=np.float64).reshape(3)
    e_v = np.asarray(e_v, dtype=np.float64).reshape(3)
    wn = 2.0 / max(float(policy_dt), 1e-6)
    return (wn * wn * e_p + 2.0 * wn * e_v).astype(np.float32)


NEAR_PRESS_SWITCH = 0.030


def near_press_force_mode(in_contact, tip, press=None, obj=None, obj_radius=0.06,
                          near=NEAR_PRESS_SWITCH):
    """True when OSC must use the ball press scale, not the 40 N air cap.

    The Isaac table's near edge is at x=0.2 and the object starts at
    x=0.30--0.40.  A free-space punch into the far face shoves the
    object back off the table before PhysX reports contact.
    """
    if bool(in_contact):
        return True
    tip = np.asarray(tip, dtype=np.float64).reshape(3)
    if press is not None:
        press = np.asarray(press, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(tip - press)) <= float(near):
            return True
    if obj is not None and press is None:
        obj = np.asarray(obj, dtype=np.float64).reshape(3)
        gap = float(np.linalg.norm(tip - obj)) - max(float(obj_radius), 1e-6)
        if gap <= 0.015:
            return True
    return False


def press_normal_outward(tip, press, fallback=None):
    """Outward (object → tip) direction used to cap the inward press."""
    if fallback is not None:
        n = np.asarray(fallback, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(n)) > 1e-8:
            return n
    if press is None:
        return None
    n = np.asarray(tip, dtype=np.float64).reshape(3) - np.asarray(
        press, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(n)) < 1e-8:
        return None
    return n.astype(np.float32)


def contact_aware_task_force(
    force_track, force_contact, contact_n_outward, min_press=0.5, max_press=2.0,
    mu=0.9,
):
    """Track the via; only the inward punch is ball-scale.

    ``force_track`` (~40 N) is what finishes a 5 mm OSC increment.  The
    floating ball in ``test_0902_isaac`` applies ``F = K u − D v`` in the
    command direction and lets PhysX build the cone — it does not rewrite
    the wrench onto the contact normal.  Clamping tangent to ``μ N`` or
    turning a lift via into ``min_press`` pins the tip and kills flip
    torque.  Inward is the only term capped to ``max_press``.  After the
    cap, the ball-law command is a floor so a reached via still pushes
    in the MPC action direction.  ``mu`` is unused; kept so existing
    callers do not break.
    """
    del mu
    force = np.asarray(force_track, dtype=np.float32).reshape(3).copy()
    ball = np.asarray(force_contact, dtype=np.float32).reshape(3)
    if contact_n_outward is None:
        return force
    n = np.asarray(contact_n_outward, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(n))
    if nrm < 1e-8:
        return force
    n = n / nrm
    inward = -n
    f_in = float(np.dot(force, inward))
    f_ball = float(np.dot(ball.astype(np.float64), inward))
    press_hi = float(np.clip(max(f_ball, float(min_press)), 0.0, float(max_press)))
    if f_in > press_hi:
        force = force + np.float32(press_hi - f_in) * inward.astype(np.float32)
    elif 0.0 <= f_in < float(min_press):
        # Reached via: keep a small press so the tip does not drop off.
        # A lift command (f_in < 0) stays on force_track so the tip can
        # follow a flip, matching the 3-DoF ball.
        force = force + np.float32(float(min_press) - f_in) * inward.astype(np.float32)
    # Tracking arrived: force_track is ~0, so push like the 3-DoF ball
    # in the MPC action direction instead of only along the normal.
    track_n = float(np.linalg.norm(np.asarray(force_track, dtype=np.float64).reshape(3)))
    ball_n = float(np.linalg.norm(ball))
    if track_n < max(ball_n, float(min_press)) + 1e-6 and ball_n > 1e-9:
        b_dir = ball.astype(np.float64) / ball_n
        along = float(np.dot(force.astype(np.float64), b_dir))
        if along < ball_n:
            force = force + np.float32(ball_n - along) * b_dir.astype(np.float32)
    return force


def strip_inward_press(force, contact_n_outward):
    """Drop the component that punches into the object.

    Used on a blocked leave: the via is an orbit / lift, so an inward
    residual from ``F = K u − D v`` would graze the body and shove it
    away while the fingertip is trying to hook around.
    """
    force = np.asarray(force, dtype=np.float32).reshape(3).copy()
    if contact_n_outward is None:
        return force
    n = np.asarray(contact_n_outward, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(n))
    if nrm < 1e-8:
        return force
    inward = -n / nrm
    f_in = float(np.dot(force.astype(np.float64), inward))
    if f_in > 0.0:
        force = force - np.float32(f_in) * inward.astype(np.float32)
    return force


def planar_table_support_local(obj_pos, r_obj_to_world, table_height):
    """Body-frame point under the COM on the world table plane.

    Do not hardcode [0, 0, -0.025] in the mesh frame: foam_brick / piggy
    source STLs are X-up, so the table is -X, not -Z.
    """
    obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3)
    rot = np.asarray(r_obj_to_world, dtype=np.float64).reshape(3, 3)
    support_world = np.array([obj_pos[0], obj_pos[1], float(table_height)], dtype=np.float64)
    return (rot.T @ (support_world - obj_pos)).astype(np.float32)


def _skew(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def tangent_basis_from_normal(n):
    """Orthonormal (n, t1, t2).  NumPy so Isaac contact never touches JAX/GPU."""
    n = np.asarray(n, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(n))
    n = n / nrm if nrm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ref = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(n[2])) < 0.9
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    t1 = np.cross(n, ref)
    t1 = t1 / max(float(np.linalg.norm(t1)), 1e-8)
    t2 = np.cross(n, t1)
    t2 = t2 / max(float(np.linalg.norm(t2)), 1e-8)
    return n.astype(np.float32), t1.astype(np.float32), t2.astype(np.float32)


def _contact_jacobian_np(n, t1, t2, j_rel, mu):
    n = np.asarray(n, dtype=np.float64).reshape(3)
    t1 = np.asarray(t1, dtype=np.float64).reshape(3)
    t2 = np.asarray(t2, dtype=np.float64).reshape(3)
    con_frame = np.stack([n, t1, t2], axis=1)
    con_frame_pmd = np.concatenate([con_frame, -con_frame[:, 1:]], axis=1)
    con_jacp = con_frame_pmd.T @ np.asarray(j_rel, dtype=np.float64)
    return con_jacp[0] + float(mu) * con_jacp[1:]


def planar_table_jacobians(obj_pos, r_obj_to_world, table_height, nv, mu, skew_fn=None):
    """One world-up table plane at the COM projection, plus its body-frame J."""
    if skew_fn is None:
        skew_fn = _skew
    obj_pos = np.asarray(obj_pos, dtype=np.float32).reshape(3)
    support_world = np.array([obj_pos[0], obj_pos[1], float(table_height)], dtype=np.float32)
    r_obj = support_world - obj_pos
    j_rel = np.zeros((3, int(nv)), dtype=np.float32)
    j_rel[:, 0:3] = np.eye(3, dtype=np.float32)
    j_rel[:, 3:6] = -np.asarray(skew_fn(r_obj), dtype=np.float32)
    n_t = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    t1_t = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t2_t = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    con_jac = np.asarray(_contact_jacobian_np(n_t, t1_t, t2_t, j_rel, mu), dtype=np.float32)
    con_jac_body = contact_jacobian_body_frame(con_jac[:, :6], r_obj_to_world)
    return con_jac, con_jac_body, planar_table_support_local(obj_pos, r_obj_to_world, table_height)


def _quat_wxyz_to_R(quat_wxyz):
    qw, qx, qy, qz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    nrm = float(np.linalg.norm([qw, qx, qy, qz]))
    if nrm < 1e-9:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / nrm, qx / nrm, qy / nrm, qz / nrm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def table_jac_mat_env(obj_pos, obj_quat_wxyz, table_height, nv=9, mu=0.5, max_ncon=10):
    """Planner-side table J from object pose.  Isaac does not need to send it."""
    rot = _quat_wxyz_to_R(obj_quat_wxyz)
    _, con_jac_body, _ = planar_table_jacobians(obj_pos, rot, table_height, nv, mu)
    jac = np.zeros((int(max_ncon) * 4, int(nv)), dtype=np.float64)
    jac[0:4, :6] = con_jac_body
    return jac


# Official Franka ``move_to_start`` / cartesian example home:
# {0, -π/4, 0, -3π/4, 0, π/2, π/4}.  Flange / fingertip Z is world -Z.
FRANKA_QD_NULLSPACE = np.array(
    [0.0, -0.25 * np.pi, 0.0, -0.75 * np.pi, 0.0, 0.5 * np.pi, 0.25 * np.pi],
    dtype=np.float64,
)


def wrap_joint_error(q_d, q):
    return ((np.asarray(q_d, dtype=np.float64) - np.asarray(q, dtype=np.float64) + np.pi) % (2.0 * np.pi)) - np.pi


def franka_nullspace_projector(jacobian, rcond=1e-6):
    """``I - J^T (J^T)^+`` from cartesian_impedance_example_controller."""
    j = np.asarray(jacobian, dtype=np.float64)
    n = int(j.shape[1])
    return (np.eye(n, dtype=np.float64) - j.T @ np.linalg.pinv(j.T, rcond=rcond)).astype(np.float32)


def franka_nullspace_posture_torque(jacobian, q, dq, q_d, stiffness):
    """Penalize ``q - q_d`` in the Cartesian-task nullspace.

    Matches franka_ros cartesian_impedance_example_controller::update:

        τ_null = (I - Jᵀ (Jᵀ)⁺) (K (q_d - q) - 2 √K q̇)

    Position-only J leaves orientation in that nullspace, so the home
    configuration keeps the fingertip normal along world -Z without a
    stiff orientation task.
    """
    k = float(stiffness)
    damp = 2.0 * np.sqrt(max(k, 1e-6))
    tau0 = k * wrap_joint_error(q_d, q) - damp * np.asarray(dq, dtype=np.float64).reshape(-1)
    return (franka_nullspace_projector(jacobian) @ tau0).astype(np.float32)
