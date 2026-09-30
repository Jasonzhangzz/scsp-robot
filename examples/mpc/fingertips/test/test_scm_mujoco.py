"""Compare SCM lambda predictions with MuJoCo, then apply the optimal force.

``physics`` drives the fingertip onto the ranked contact.  ``force`` holds
the fingertip still and applies the solved contact wrench on the object;
table reaction then comes only from MuJoCo.  Both modes print the gap
between ``x_plus_opt`` and the simulated pose, and between the predicted
environment impulse and the measured table contact wrench.
"""

import argparse
import os
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(current_dir)
while os.path.basename(parent_dir) != "scsp-robot":
    _next_dir = os.path.dirname(parent_dir)
    if _next_dir == parent_dir:
        raise RuntimeError("scsp-robot repo root not found from %s" % current_dir)
    parent_dir = _next_dir
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from planning.acados_env import ensure_acados_env
ensure_acados_env()

from contact.fingertips_collision_detection2 import Contact
from envs.fingertips_env import MjSimulator
from examples.mpc.fingertips.test.params import ExplicitMPCParams
from examples.mpc.fingertips.test.test_0902 import (
    _pose_mismatch_diagnostics,
    add_rollout_via_args,
)
from utils import metrics, rotations


SUCCESS_POS = 0.02
SUCCESS_QUAT = 0.015


def _object_frame(qpos):
    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    quat_xyzw = [qpos[4], qpos[5], qpos[6], qpos[3]]
    return Rotation.from_quat(quat_xyzw).as_matrix()


def _predict_env_impulse(optimizer, lam, p_local, n_arm, t1, t2, tau_o):
    """Replay the environment projection inside ``_solve_optimization_acados``."""
    lam = np.asarray(lam, dtype=np.float64).reshape(3)
    p_local = np.asarray(p_local, dtype=np.float64).reshape(3)
    n_arm = np.asarray(n_arm, dtype=np.float64).reshape(3)
    t1 = np.asarray(t1, dtype=np.float64).reshape(3)
    t2 = np.asarray(t2, dtype=np.float64).reshape(3)
    jac = np.zeros((3, 6), dtype=np.float64)
    jac[:, :3] = np.eye(3)
    jac[0, 4], jac[0, 5] = p_local[2], -p_local[1]
    jac[1, 3], jac[1, 5] = -p_local[2], p_local[0]
    jac[2, 3], jac[2, 4] = p_local[1], -p_local[0]
    contact_frame = np.column_stack((n_arm, t1, t2))
    wrench_scale = optimizer.h if optimizer.wrench_is_force else 1.0
    bias = optimizer.h * np.asarray(tau_o, dtype=np.float64).reshape(6)
    bias = bias + wrench_scale * jac.T @ (contact_frame @ lam)
    qib = optimizer.Q_inv @ bias
    j_env = np.asarray(optimizer.J_tilde, dtype=np.float64)
    d_inv = np.asarray(optimizer.compute_env_diag_inverse(j_env), dtype=np.float64)
    impulses = []
    wrench = np.zeros(6, dtype=np.float64)
    for i in range(int(optimizer.max_contacts)):
        rows = slice(4 * i, 4 * (i + 1))
        fi = np.maximum(-(d_inv[rows] @ (j_env[rows] @ qib)), 0.0)
        impulses.append(fi)
        wrench += j_env[rows].T @ fi
    return np.asarray(impulses, dtype=np.float64), wrench


def _table_contact_wrench(model, data, obj_body_id, obj_pos):
    """World wrench about the object origin from table contacts."""
    force = np.zeros(3, dtype=np.float64)
    torque = np.zeros(3, dtype=np.float64)
    table_id = model.geom("table").id
    obj_geoms = {
        int(gid) for gid in range(model.ngeom)
        if int(model.geom_bodyid[gid]) == int(obj_body_id)
    }
    for i in range(int(data.ncon)):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        on_obj = g1 in obj_geoms or g2 in obj_geoms
        on_table = g1 == table_id or g2 == table_id
        if not (on_obj and on_table):
            continue
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, i, wrench)
        # Contact frame force/torque.  The normal points from geom1 to geom2.
        normal = np.asarray(con.frame[:3], dtype=np.float64)
        contact_force = wrench[0] * normal
        if g1 in obj_geoms:
            contact_force = -contact_force
        point = np.asarray(con.pos, dtype=np.float64)
        force += contact_force
        torque += np.cross(point - obj_pos, contact_force)
    return force, torque


def _solve_step(param, args, env, contact):
    curr_q = env.get_state()
    _, _, _, jac_mat_env, _ = contact.detect_once(env)
    r_obj = _object_frame(curr_q)
    gravity = np.hstack([
        r_obj.T @ param.gravity_[:3] * param.obj_mass_,
        np.zeros(3),
    ])
    target_quat_local = rotations.quaternion_multiply(
        rotations.quaternion_conjugate(curr_q[3:7]), param.target_q_)
    target_pose = np.hstack([
        r_obj.T @ (param.target_p_ - curr_q[:3]),
        target_quat_local,
    ])
    current_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    tip_local = r_obj.T @ (curr_q[7:10] - curr_q[:3])
    optimizer = param.lambda_optimizer
    optimizer.update_Jacobian(jac_mat_env)
    visible = optimizer.get_availble_point_idx(
        curr_q[:3], r_obj, param.target_p_, args.ground_height_threshold,
        viewpoint_local=None, heading_filter=False)
    visible = optimizer.filter_rankable_indices(visible)
    point, normal, _, _, _ = optimizer.choose_contact_points(
        target_pose, current_pose, gravity, visible,
        v_last=None, force_required=True, query_local=tip_local)
    idx = int(getattr(optimizer, "last_selected_idx", 0))
    lam = np.asarray(getattr(optimizer, "last_best_force", np.zeros(3)), dtype=np.float64).reshape(3)
    x_plus = getattr(optimizer, "last_best_x_plus", None)
    n_arm = np.asarray(optimizer.normal[idx], dtype=np.float64)
    t1 = np.asarray(optimizer.t1[idx], dtype=np.float64)
    t2 = np.asarray(optimizer.t2[idx], dtype=np.float64)
    p_local = np.asarray(optimizer.sample_point[idx], dtype=np.float64)
    impulses, env_wrench = _predict_env_impulse(
        optimizer, lam, p_local, n_arm, t1, t2, gravity)
    contact_frame = np.column_stack((n_arm, t1, t2))
    force_local = contact_frame @ lam
    return {
        "qpos": curr_q.copy(),
        "R": r_obj,
        "x_plus": None if x_plus is None else np.asarray(x_plus, dtype=np.float64).reshape(7),
        "lam": lam,
        "force_local": force_local,
        "p_local": p_local,
        "point": np.asarray(point, dtype=np.float64),
        "normal": np.asarray(normal, dtype=np.float64),
        "impulses": impulses,
        "env_wrench": env_wrench,
        "target_p": np.asarray(param.target_p_, dtype=np.float64),
        "target_q": np.asarray(param.target_q_, dtype=np.float64),
    }


def _goal_errors(qpos, target_p, target_q):
    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    return (
        float(metrics.comp_pos_error(qpos[:3], target_p)),
        float(metrics.comp_quat_error(qpos[3:7], target_q)),
    )


def _measure(env, before, solved):
    after = env.get_state()
    mismatch = _pose_mismatch_diagnostics(before, after, solved["x_plus"])
    obj_body = int(env.model_.geom("obj").bodyid)
    force, torque = _table_contact_wrench(
        env.model_, env.data_, obj_body, after[:3])
    r_after = _object_frame(after)
    measured = np.hstack([r_after.T @ force, r_after.T @ torque])
    pred = np.asarray(solved["env_wrench"], dtype=np.float64).reshape(6)
    # Predicted wrench is an impulse over h; MuJoCo reports a force.
    h = float(env.model_.opt.timestep * env.param_.frame_skip_)
    pred_force = pred / max(h, 1e-8)
    return {
        "pose_pos_error": float(mismatch.get("pose_pos_error", np.nan)),
        "pose_rot_error_rad": float(mismatch.get("pose_rot_error_rad", np.nan)),
        "lambda_env_error": float(np.linalg.norm(measured - pred_force)),
        "pred_env_force": float(np.linalg.norm(pred_force[:3])),
        "measured_env_force": float(np.linalg.norm(measured[:3])),
        "goal_pos": _goal_errors(after, solved["target_p"], solved["target_q"])[0],
        "goal_quat": _goal_errors(after, solved["target_p"], solved["target_q"])[1],
    }


def _step_physics(env, param, solved):
    tip = np.asarray(solved["qpos"][7:10], dtype=np.float64)
    contact_world = solved["R"] @ solved["p_local"] + solved["qpos"][:3]
    step = max(1e-4, float(getattr(param, "mpc_u_ub_", 0.005)))
    delta = contact_world - tip
    norm = float(np.linalg.norm(delta))
    if norm > step:
        delta = delta * (step / norm)
    env.step(delta)


def _step_force(env, solved):
    """Hold the fingertip and apply the solved contact wrench for one lambda step."""
    param = env.param_
    curr_q = env.get_state()
    desired = curr_q[7:].copy()
    r_obj = _object_frame(curr_q)
    force_world = r_obj @ np.asarray(solved["force_local"], dtype=np.float64)
    point_world = r_obj @ np.asarray(solved["p_local"], dtype=np.float64) + curr_q[:3]
    torque_world = np.cross(point_world - curr_q[:3], force_world)
    obj_body = int(env.model_.geom("obj").bodyid)
    for _ in range(int(param.frame_skip_)):
        env.data_.xfrc_applied[:] = 0
        env.data_.xfrc_applied[env.fingertip_body_id, :3] = (
            -env.fingertip_mass * env.gravity_vec)
        env.data_.xfrc_applied[obj_body, :3] = force_world
        env.data_.xfrc_applied[obj_body, 3:] = torque_world
        q = env.get_state()
        dpos = q[7:] - desired
        dvel = env.data_.qvel[6:]
        env.data_.ctrl[:] = -100.0 * dpos - 2.0 * dvel
        mujoco.mj_step(env.model_, env.data_, nstep=1)
        env._sync_viewer()


def _run_mode(mode, args, steps):
    os.environ["MUJOCO_HEADLESS"] = "0" if args.viewer else "1"
    args.rollout = True
    args.solver = "acados"
    param = ExplicitMPCParams(
        args, rand_seed=0, target_type="ground-rotation", model="explicit")
    param.torch_solver = "acados"
    param.lambda_optimizer.solver = "acados"
    contact = Contact(param)
    env = MjSimulator(param)
    rows = []
    try:
        for step in range(int(steps)):
            solved = _solve_step(param, args, env, contact)
            before = env.get_state().copy()
            if mode == "physics":
                _step_physics(env, param, solved)
            else:
                _step_force(env, solved)
            row = _measure(env, before, solved)
            row["step"] = step
            rows.append(row)
            print("%s step %d: pose_pos=%.6f pose_rot=%.6f env_force_err=%.4f goal_pos=%.4f goal_quat=%.4f" % (
                mode, step, row["pose_pos_error"], row["pose_rot_error_rad"],
                row["lambda_env_error"], row["goal_pos"], row["goal_quat"]))
            if mode == "force" and row["goal_pos"] < SUCCESS_POS and row["goal_quat"] < SUCCESS_QUAT:
                break
    finally:
        if env.viewer_ is not None:
            env.viewer_.close()
    return rows


def _summary(mode, rows):
    if not rows:
        print("%s: no steps" % mode)
        return
    pose = np.array([row["pose_pos_error"] for row in rows], dtype=np.float64)
    rot = np.array([row["pose_rot_error_rad"] for row in rows], dtype=np.float64)
    env_err = np.array([row["lambda_env_error"] for row in rows], dtype=np.float64)
    last = rows[-1]
    reached = last["goal_pos"] < SUCCESS_POS and last["goal_quat"] < SUCCESS_QUAT
    print("%s summary: steps=%d mean_pose_pos=%.6f mean_pose_rot=%.6f mean_lambda_env_err=%.4f success=%s" % (
        mode, len(rows), float(np.nanmean(pose)), float(np.nanmean(rot)),
        float(np.nanmean(env_err)), reached if mode == "force" else "n/a"))


def main():
    parser = argparse.ArgumentParser(description="Compare SCM lambda predictions with MuJoCo.")
    add_rollout_via_args(parser)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--mode", choices=("both", "physics", "force"), default="both")
    args = parser.parse_args()
    modes = ("physics", "force") if args.mode == "both" else (args.mode,)
    for mode in modes:
        _summary(mode, _run_mode(mode, args, args.steps))


if __name__ == "__main__":
    main()
