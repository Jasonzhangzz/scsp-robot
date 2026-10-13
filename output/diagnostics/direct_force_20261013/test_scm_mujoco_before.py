"""MuJoCo fidelity checks and Appendix-B SCM evaluation.

The optimizer is intentionally treated as a black box.  It selects a contact
and a robot wrench; :mod:`planning.scm_contact_models` computes every
environment response used by this benchmark.  This keeps the evaluation
independent of both the Isaac rollout and ``planning.mlqp_point``.
"""

import argparse
import csv
import json
import os
import sys
import time

try:
    import mujoco
except ImportError:  # algebraic unit tests can run without MuJoCo
    mujoco = None
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
from planning.scm_contact_models import (
    LCPContactModel,
    MujocoForwardContactModel,
    appendix_accuracy,
    compare_lambda,
    contact_jacobian,
    contact_wrench,
    lcp_mujoco_motion_accuracy,
    surrogate_response,
)
from contact.fingertips_collision_detection2 import Contact
from envs.fingertips_env import MjSimulator
from examples.mpc.fingertips.test.params import ExplicitMPCParams, mujoco_physical_parameters
from examples.mpc.fingertips.test.test_0902 import (
    _predicted_object_pose,
    add_rollout_via_args,
)
from utils import metrics, rotations

ensure_acados_env()

POS_SUCCESS = 0.02
QUAT_SUCCESS = 0.015
DEFAULT_OBJECTS = (
    "foam_brick", "mug", "rubber_duck", "elephant", "piggy_bank",
    "stanford_bunny2", "teapot",
)


def _object_frame(qpos):
    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    return Rotation.from_quat([qpos[4], qpos[5], qpos[6], qpos[3]]).as_matrix()


def _contact_model_name(args):
    if args.qp:
        return "qp"
    if args.lcp:
        return "lcp"
    return "surrogate"


def _as_json(value):
    if isinstance(value, dict):
        return {str(k): _as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_as_json(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _parse_float_list(value, default):
    if value is None or str(value).strip() == "":
        return list(default)
    return [float(item) for item in str(value).split(",") if str(item).strip()]


def _contact_keys(env):
    """Stable keys for table/object contacts used in transition statistics."""
    if env is None or mujoco is None:
        return set()
    model, data = env.model_, env.data_
    try:
        table_id = int(model.geom("table").id)
        obj_body = int(model.geom("obj").bodyid)
    except Exception:
        return set()
    obj_geoms = {int(gid) for gid in range(model.ngeom)
                 if int(model.geom_bodyid[gid]) == obj_body}
    keys = set()
    for i in range(int(data.ncon)):
        con = data.contact[i]
        if table_id not in (int(con.geom1), int(con.geom2)):
            continue
        if not ({int(con.geom1), int(con.geom2)} & obj_geoms):
            continue
        pos = tuple(np.round(np.asarray(con.pos, dtype=np.float64), 4))
        keys.add((min(int(con.geom1), int(con.geom2)),
                  max(int(con.geom1), int(con.geom2)), pos))
    return keys


def _transition(previous, current):
    previous = set(previous or ())
    current = set(current or ())
    created = current - previous
    removed = previous - current
    return {
        "created": int(len(created)),
        "removed": int(len(removed)),
        "active_contacts": int(len(current)),
        "created_keys": [str(key) for key in sorted(created, key=str)],
        "removed_keys": [str(key) for key in sorted(removed, key=str)],
    }


def _set_physical_mass(param, env, mass):
    """Change only the MuJoCo object mass for a sensitivity trial."""
    mass = float(mass)
    if mass <= 0.0 or env is None:
        return
    model = env.model_
    body_id = int(model.geom("obj").bodyid)
    old = float(model.body_mass[body_id])
    if old <= 0.0:
        return
    # mj_setConst may rebuild derived model data and reset the associated
    # data buffers on some MuJoCo versions. Preserve the actual rollout state
    # so a parameter sweep cannot silently teleport the fingertip to zero.
    qpos = env.data_.qpos.copy()
    qvel = env.data_.qvel.copy()
    ctrl = env.data_.ctrl.copy()
    xfrc = env.data_.xfrc_applied.copy()
    qfrc = env.data_.qfrc_applied.copy()
    ratio = mass / old
    model.body_mass[body_id] = mass
    model.body_inertia[body_id] *= ratio
    if hasattr(mujoco, "mj_setConst"):
        mujoco.mj_setConst(model, env.data_)
    env.data_.qpos[:] = qpos
    env.data_.qvel[:] = qvel
    env.data_.ctrl[:] = ctrl
    env.data_.xfrc_applied[:] = xfrc
    env.data_.qfrc_applied[:] = qfrc
    param.obj_mass_ = mass
    # Keep the surrogate and the execution model on the same mass/inertia
    # after a sensitivity sweep changes MuJoCo's body mass.
    _sync_surrogate_physics(param, env)
    mujoco.mj_forward(model, env.data_)


def _sync_surrogate_physics(param, env):
    """Refresh physical calibration after MuJoCo model edits."""
    physics = mujoco_physical_parameters(
        param.model_path_, getattr(param, 'calibration_noise_std_', 0.0),
        np.random.default_rng(1000))
    body_id = int(env.model_.geom('obj').bodyid)
    physics['mass'] = float(env.model_.body_mass[body_id])
    physics['generalized_mass'][:3, :3] = physics['mass'] * np.eye(3)
    # body_inertia is already updated by _set_physical_mass.
    physics['inertia'] = np.diag(np.asarray(env.model_.body_inertia[body_id], dtype=float))
    physics['generalized_mass'][3:, 3:] = physics['inertia']
    param.mujoco_physics_ = physics
    param.obj_mass_ = physics['mass']
    param.obj_inertia_ = physics['generalized_mass'].copy()
    param.mu_object_ = physics['mu_object']
    param.mu_table_ = physics['mu_table']
    rot_diag = np.maximum(np.diag(param.obj_inertia_[3:, 3:]), 1e-15)
    param.dynamics_scale_ = np.eye(6)
    param.dynamics_scale_[3:, 3:] = np.diag(
        np.sqrt(max(param.obj_mass_, 1e-15) / rot_diag))
    opt = param.lambda_optimizer
    opt.obj_inertia = param.obj_inertia_.copy()
    opt.dynamics_scale = param.dynamics_scale_.copy()
    scaled = opt.dynamics_scale.T @ opt.obj_inertia @ opt.dynamics_scale
    opt.scaled_mass_matrix = scaled
    opt.dynamics_condition_number = float(np.linalg.cond(scaled))
    opt.Q_inv = opt.dynamics_scale @ np.linalg.inv(scaled) @ opt.dynamics_scale.T
    opt.mu_arm_obj = float(param.mu_object_)
    opt.max_normal_force = opt.max_contact_force / np.sqrt(
        1.0 + 2.0 * opt.mu_arm_obj ** 2)
    # The generated ACADOS graph embeds Q_inv and the friction bound. Rebuild
    # it after a mass/friction sweep so the requested condition is actually
    # solved with the calibrated parameters.
    if getattr(opt, 'acados_solver', None) is not None:
        try:
            opt.acados_solver = opt._build_acados_contact_solver()
        except Exception as exc:
            opt.acados_solver = None
            opt.last_acados_failure_reason = f'calibration rebuild failed: {exc}'


def _integrate_local_pose(qpos, v_plus, h):
    """Convert a body-frame velocity increment to ``x_plus`` convention."""
    v_plus = np.asarray(v_plus, dtype=np.float64).reshape(6)
    angle = float(h) * v_plus[3:]
    qrel_xyzw = Rotation.from_rotvec(angle).as_quat()
    qrel = np.array([qrel_xyzw[3], qrel_xyzw[0], qrel_xyzw[1], qrel_xyzw[2]])
    return np.hstack((float(h) * v_plus[:3], qrel))


def _model_response(optimizer, jacobian, phi, lam, p_local, n_arm, t1, t2,
                    gravity, mode, coupling_scale=0.0, regularization=1e-8):
    h = float(optimizer.h)
    q_inv = np.asarray(optimizer.Q_inv, dtype=np.float64)
    J_arm = contact_jacobian(p_local)
    frame = np.column_stack((n_arm, t1, t2))
    wrench_scale = h if bool(getattr(optimizer, "wrench_is_force", False)) else 1.0
    b = h * np.asarray(gravity, dtype=np.float64).reshape(6)
    b = b + wrench_scale * J_arm.T @ (frame @ np.asarray(lam, dtype=np.float64))
    scm = surrogate_response(q_inv, jacobian, b, phi=phi,
                             regularization=regularization,
                             coupling_scale=0.0)
    lcp_model = LCPContactModel(regularization=regularization)
    lam_lcp, v_lcp = lcp_model.respond(
        scm["v_free"], jacobian, q_inv, phi=phi,
        coupling_scale=float(coupling_scale), regularization=regularization)
    qp = dict(scm)
    qp["lambda_env"], qp["v_plus"] = lam_lcp, v_lcp
    qp["lcp"] = lcp_model.last
    selected = scm if mode == "surrogate" else qp
    return {
        "b": b,
        "v_free": scm["v_free"],
        "surrogate": scm,
        "lcp": {**scm, "lambda_env": lam_lcp, "v_plus": v_lcp,
                "lcp": lcp_model.last},
        "qp": qp,
        "selected": selected,
        "regularization": float(regularization),
    }


def _solve_step(param, args, env, contact, mode, coupling_scale=1.0,
                lambda_scale=1.0):
    curr_q = env.get_state()
    phi, _, _, jac_mat_env, _ = contact.detect_once(env)
    r_obj = _object_frame(curr_q)
    gravity = np.hstack([
        r_obj.T @ param.gravity_[:3] * param.obj_mass_, np.zeros(3),
    ])
    target_quat_local = rotations.quaternion_multiply(
        rotations.quaternion_conjugate(curr_q[3:7]), param.target_q_)
    target_pose = np.hstack([
        r_obj.T @ (param.target_p_ - curr_q[:3]), target_quat_local,
    ])
    current_pose = np.array([0., 0., 0., 1., 0., 0., 0.])
    tip_local = r_obj.T @ (curr_q[7:10] - curr_q[:3])
    optimizer = param.lambda_optimizer
    jac_mat_env = optimizer.update_Jacobian(jac_mat_env)
    visible = optimizer.get_availble_point_idx(
        curr_q[:3], r_obj, param.target_p_, args.ground_height_threshold,
        viewpoint_local=None, heading_filter=False)
    visible = optimizer.filter_rankable_indices(visible)
    point, normal, _, _, _ = optimizer.choose_contact_points(
        target_pose, current_pose, gravity, visible, v_last=None,
        force_required=True, query_local=tip_local)
    idx = int(getattr(optimizer, "last_selected_idx", 0) or 0)
    lam_base = np.asarray(getattr(optimizer, "last_best_force", np.zeros(3)),
                          dtype=np.float64).reshape(3)
    lam = float(lambda_scale) * lam_base
    n_arm = np.asarray(optimizer.normal[idx], dtype=np.float64)
    t1 = np.asarray(optimizer.t1[idx], dtype=np.float64)
    t2 = np.asarray(optimizer.t2[idx], dtype=np.float64)
    p_local = np.asarray(optimizer.sample_point[idx], dtype=np.float64)
    response = _model_response(
        optimizer, jac_mat_env, phi, lam, p_local, n_arm, t1, t2, gravity,
        mode, coupling_scale=coupling_scale,
        regularization=float(getattr(args, "scm_regularization", 1e-8)))
    response["surrogate_accuracy"] = appendix_accuracy(
        optimizer.obj_inertia, optimizer.Q_inv, jac_mat_env,
        response["lcp"]["v_plus"], response["surrogate"]["lambda_env"],
        response["surrogate"]["v_plus"], optimizer.h,
        lam_ref=response["lcp"]["lambda_env"],
        regularization=response["regularization"],
        coupling_scale=coupling_scale)
    return {
        "qpos": curr_q.copy(), "R": r_obj, "lam": lam,
        "lam_base": lam_base,
        "force_local": np.column_stack((n_arm, t1, t2)) @ lam,
        "n_arm": n_arm, "t1": t1, "t2": t2, "p_local": p_local,
        "point": np.asarray(point, dtype=np.float64),
        "normal": np.asarray(normal, dtype=np.float64),
        "h": float(optimizer.h), "gravity": gravity,
        "wrench_is_force": bool(getattr(optimizer, "wrench_is_force", False)),
        "execution_force_cap": float(getattr(args, "execution_force_cap", 0.1)),
        "execution_torque_cap": float(getattr(args, "execution_torque_cap", 1e-5)),
        "physical_mass": float(param.obj_mass_),
        "physical_inertia": np.asarray(param.obj_inertia_[3:, 3:]).copy(),
        "mu_object": float(param.mu_object_),
        "mu_table": float(getattr(param, "mu_table_", np.nan)),
        "dynamics_scale": np.asarray(getattr(param, "dynamics_scale_", np.eye(6))).copy(),
        "dynamics_condition_number": float(getattr(optimizer, "dynamics_condition_number", np.nan)),
        "calibration_noise_std": float(getattr(param, "calibration_noise_std_", 0.0)),
        "target_p": np.asarray(param.target_p_, dtype=np.float64),
        "target_q": np.asarray(param.target_q_, dtype=np.float64),
        "response": response,
    }


def _goal_errors(qpos, target_p, target_q):
    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    return (float(metrics.comp_pos_error(qpos[:3], target_p)),
            float(metrics.comp_quat_error(qpos[3:7], target_q)))


def _succeeded(pos_err, quat_err):
    return bool(pos_err < POS_SUCCESS and quat_err < QUAT_SUCCESS)


def _show_step(env, solved, qpos):
    rot = _object_frame(qpos)
    contact_world = rot @ solved["p_local"] + qpos[:3]
    force_world = rot @ solved["force_local"]
    norm = float(np.linalg.norm(force_world))
    direction = force_world / norm if norm > 1e-8 else np.zeros(3)
    env.show_target(contact_world + 0.04 * direction)
    env.show_best_contact(contact_world)


def _execution_dt(env):
    model_dt = float(getattr(env.model_.opt, "timestep", 0.0))
    nstep = max(1, int(getattr(env.param_, "frame_skip_", 1)))
    if model_dt <= 0.0:
        raise RuntimeError("MuJoCo model timestep must be positive")
    return model_dt * nstep, nstep


def _step_contact_force(env, solved, execution="wrench", lambda_scale=1.0):
    """Apply one direct-apply step without touching the shared optimizer."""
    selected = solved["response"]["selected"]
    qpos = env.get_state()
    x_plus = _integrate_local_pose(qpos[:7], selected["v_plus"], solved["h"])
    h_exec, nstep = _execution_dt(env)
    if execution == "wrench":
        if mujoco is None:
            raise RuntimeError("MuJoCo is required for wrench execution")
        obj_body = int(env.model_.geom("obj").bodyid)
        rot = _object_frame(qpos)
        lam_apply = solved.get("lam", np.zeros(3))
        if "lam_base" not in solved:
            lam_apply = lam_apply * float(lambda_scale)
        wrench_impulse = contact_wrench(
            solved["p_local"], solved["n_arm"], solved["t1"], solved["t2"],
            lam_apply)
        wrench_impulse[:3] *= float(lambda_scale) if "lam_base" not in solved else 1.0
        wrench_impulse[3:] *= float(lambda_scale) if "lam_base" not in solved else 1.0
        if bool(solved.get("wrench_is_force", False)):
            force_body = wrench_impulse.copy()
            impulse_body = wrench_impulse * h_exec
        else:
            impulse_body = wrench_impulse.copy()
            force_body = impulse_body / h_exec
        cap = float(solved.get("execution_force_cap", 0.1))
        torque_cap = float(solved.get("execution_torque_cap", 1e-5))
        raw_impulse_body = impulse_body.copy()
        unclipped_force = force_body.copy()
        norm = float(np.linalg.norm(force_body[:3]))
        if cap > 0.0 and norm > cap:
            force_body *= cap / norm
            impulse_body = force_body * h_exec
        torque_norm = float(np.linalg.norm(force_body[3:]))
        if torque_cap > 0.0 and torque_norm > torque_cap:
            force_body[3:] *= torque_cap / torque_norm
            impulse_body = force_body * h_exec
        # #region agent log
        try:
            import json as _json
            _log_path = "/home/lab423/scsp/scsp-robot/.cursor/debug-c69ca2.log"
            os.makedirs(os.path.dirname(_log_path), exist_ok=True)
            with open(_log_path, "a", encoding="utf-8") as _lf:
                _lf.write(_json.dumps({
                    "sessionId": "c69ca2",
                    "runId": "pre-fix",
                    "hypothesisId": "A",
                    "location": "test_scm_mujoco.py:_step_contact_force",
                    "message": "wrench unit conversion and caps",
                    "timestamp": int(time.time() * 1000),
                    "data": {
                        "h_model": float(solved["h"]),
                        "h_exec": float(h_exec),
                        "nstep": int(nstep),
                        "wrench_is_force": bool(solved.get("wrench_is_force", False)),
                        "lam": [float(x) for x in np.asarray(lam_apply).reshape(-1)[:3]],
                        "lam_norm": float(np.linalg.norm(lam_apply)),
                        "impulse_raw": [float(x) for x in raw_impulse_body],
                        "force_unclipped": [float(x) for x in unclipped_force],
                        "force_applied": [float(x) for x in force_body],
                        "force_norm_unclipped": float(np.linalg.norm(unclipped_force[:3])),
                        "torque_norm_unclipped": float(np.linalg.norm(unclipped_force[3:])),
                        "force_norm_applied": float(np.linalg.norm(force_body[:3])),
                        "torque_norm_applied": float(np.linalg.norm(force_body[3:])),
                        "force_cap": cap,
                        "torque_cap": torque_cap,
                        "force_clipped": bool(cap > 0.0 and float(np.linalg.norm(unclipped_force[:3])) > cap),
                        "torque_clipped": bool(torque_cap > 0.0 and float(np.linalg.norm(unclipped_force[3:])) > torque_cap),
                        "p_local": [float(x) for x in np.asarray(solved["p_local"]).reshape(3)],
                        "origin_style_force_norm": float(np.linalg.norm(wrench_impulse[:3])),
                    },
                }) + "\n")
        except Exception:
            pass
        # #endregion
        env.data_.xfrc_applied[:] = 0.0
        env.data_.xfrc_applied[obj_body, :3] = rot @ force_body[:3]
        env.data_.xfrc_applied[obj_body, 3:] = rot @ force_body[3:]
        # Wrench execution represents an ideal position-controlled fingertip
        # applying the selected contact wrench. Keep that actuator from
        # falling under gravity during the MuJoCo step; otherwise the free
        # fingertip collides with the table/object and injects an unrelated
        # impulse into the benchmark.
        tip_body = getattr(env, "fingertip_body_id", None)
        tip_mass = getattr(env, "fingertip_mass", 0.0)
        gravity_vec = getattr(env, "gravity_vec", np.zeros(3))
        if tip_body is not None and float(tip_mass) > 0.0:
            env.data_.xfrc_applied[int(tip_body), :3] = -float(tip_mass) * np.asarray(gravity_vec)
        mujoco.mj_step(env.model_, env.data_, nstep=nstep)
        qacc = np.asarray(env.data_.qacc, dtype=np.float64)
        unstable = bool((not np.isfinite(qacc).all()) or
                        np.max(np.abs(qacc)) > 1.0e6)
        env.data_.xfrc_applied[:] = 0.0
        env._sync_viewer()
        return {
            "execution": execution, "h_exec": h_exec, "nstep": nstep,
            "wrench_impulse_body": impulse_body,
            "wrench_impulse_raw_body": raw_impulse_body,
            "actual_applied_impulse_body": impulse_body.copy(),
            "force_body_unclipped": unclipped_force,
            "force_body": force_body,
            "force_cap": cap,
            "torque_cap": torque_cap,
            "force_was_clipped": bool(not np.allclose(force_body, unclipped_force)),
            "mujoco_unstable": unstable,
            "max_abs_qacc": float(np.max(np.abs(qacc))) if qacc.size else 0.0,
            "force_units": "N", "impulse_units": "N*s",
        }
    planned_pos, quat = _predicted_object_pose(qpos[:7], x_plus)
    goal = solved["target_p"]
    to_goal = goal - qpos[:3]
    distance = float(np.linalg.norm(to_goal))
    pos = qpos[:3].copy()
    if distance > 1e-9:
        direction = to_goal / distance
        along = float(np.clip(np.dot(planned_pos - qpos[:3], direction),
                              0.0, min(0.005, distance)))
        pos = qpos[:3] + along * direction
    planned_err = float(metrics.comp_quat_error(quat, solved["target_q"]))
    if planned_err < 0.02 and distance > 1e-9:
        pos = qpos[:3] + min(0.004, distance) * (to_goal / distance)
        if planned_err > metrics.comp_quat_error(qpos[3:7], solved["target_q"]) + 1e-4:
            quat = qpos[3:7].copy()
    env.data_.qpos[:7] = np.hstack((pos, quat))
    env.data_.qvel[:6] = 0.0
    env.data_.xfrc_applied[:] = 0.0
    mujoco.mj_forward(env.model_, env.data_)
    env._sync_viewer()
    return {"execution": execution, "h_exec": h_exec, "nstep": nstep,
            "force_units": "N", "impulse_units": "N*s"}


def _reference_stats(solved, mujoco_model, optimizer, lambda_scale=1.0,
                     execution_dt=None, nstep=None, force_cap=0.0,
                     torque_cap=0.0):
    response = solved["response"]
    v_scm = response["surrogate"]["v_plus"]
    out = {}
    for name in ("lcp", "qp"):
        out["vs_" + name] = lcp_mujoco_motion_accuracy(
            optimizer.obj_inertia, response[name]["v_plus"], v_scm, solved["h"])
    try:
        v_mj = mujoco_model.forward_velocity(
            optimizer, solved["lam"], solved["p_local"], solved["n_arm"],
            solved["t1"], solved["t2"], solved["h"],
            tau_body=None, lambda_scale=lambda_scale,
            execution_dt=execution_dt, nstep=nstep, force_cap=force_cap,
            torque_cap=torque_cap)
        mj_last = dict(mujoco_model.last or {})
        out["mujoco_v_plus"] = np.asarray(v_mj, dtype=np.float64)
        out["vs_mujoco"] = lcp_mujoco_motion_accuracy(
            optimizer.obj_inertia, v_mj, v_scm, solved["h"])
        out["vs_mujoco_raw"] = dict(out["vs_mujoco"])
        out["qp_vs_mujoco"] = lcp_mujoco_motion_accuracy(
            optimizer.obj_inertia, v_mj, response["qp"]["v_plus"], solved["h"])
        measured = np.asarray(mj_last.get("lambda_env", []), dtype=np.float64)
        active_rows = np.asarray(response["surrogate"].get("rows", []), dtype=np.int64)
        predicted_full = np.asarray(response["surrogate"]["lambda_env"], dtype=np.float64)
        predicted = predicted_full[active_rows] if active_rows.size else predicted_full[:0]
        n = min(measured.size, predicted.size)
        contact_match = (not bool(mj_last.get("contact_transition", False)) and
                         measured.size == predicted.size and measured.size > 0)
        out["baseline_contact_match"] = bool(contact_match)
        if contact_match:
            out["vs_mujoco"].update(compare_lambda(predicted[:n], measured[:n]))
            qp_full = np.asarray(response["qp"]["lambda_env"], dtype=np.float64)
            qp_lam = qp_full[active_rows] if active_rows.size else qp_full[:0]
            out["qp_vs_mujoco"].update(compare_lambda(qp_lam[:n], measured[:n]))
        else:
            out["vs_mujoco"] = None
            out["qp_vs_mujoco"] = None
        out["mujoco_env_wrench_body"] = mj_last.get("env_wrench_body", np.zeros(6))
        out["mujoco_lambda_env"] = measured
        out["mujoco_ncon"] = int(mj_last.get("ncon", 0))
        out["mujoco_ncon_before"] = int(mj_last.get("ncon_before", 0))
        out["contact_transition"] = bool(mj_last.get("contact_transition", False))
        out["baseline_valid"] = bool(contact_match and not bool(
            mj_last.get("contact_transition", False)))
        out["baseline_execution_dt"] = float(mj_last.get("execution_dt", 0.0))
        qp_motion = out["qp_vs_mujoco"]
        if qp_motion is None:
            out["qp_matches_mujoco"] = None
        else:
            qp_lam_error = float(qp_motion.get("lambda_env_relative_error", 0.0))
            qp_mag_error = float(qp_motion.get("relative_magnitude_error", 0.0))
            qp_dir_error = qp_motion.get("direction_error")
            out["qp_matches_mujoco"] = bool(
                (qp_dir_error is None or qp_dir_error <= 1e-3) and
                qp_mag_error <= 1e-3 and qp_lam_error <= 1e-3)
    except (RuntimeError, ValueError, FloatingPointError):
        out["vs_mujoco"] = None
        out["baseline_valid"] = False
    active_rows = np.asarray(response["surrogate"].get("rows", []), dtype=np.int64)
    ref_full = np.asarray(response["lcp"]["lambda_env"], dtype=np.float64)
    pred_full = np.asarray(response["surrogate"]["lambda_env"], dtype=np.float64)
    ref_lam = ref_full[active_rows] if active_rows.size else ref_full[:0]
    pred_lam = pred_full[active_rows] if active_rows.size else pred_full[:0]
    n = min(ref_lam.size, pred_lam.size)
    if n:
        d = np.diag(np.asarray(response["surrogate"].get("D", np.eye(n))))
        out["vs_lcp"]["lambda_env_error"] = float(np.linalg.norm(pred_lam[:n] - ref_lam[:n]))
        out["vs_lcp"]["lambda_gap_D"] = float(np.sqrt(np.sum(
            d[:n] * (pred_lam[:n] - ref_lam[:n]) ** 2)))
    return out


def _write_rows(path, rows, metadata):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"metadata": _as_json(metadata), "rows": _as_json(rows)}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)
    csv_path = os.path.splitext(path)[0] + ".csv"
    def flatten(value, prefix=""):
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                child = "%s.%s" % (prefix, key) if prefix else str(key)
                out.update(flatten(item, child))
            return out
        if isinstance(value, (list, tuple, np.ndarray)):
            return {prefix: json.dumps(_as_json(value), allow_nan=False)}
        return {prefix: _as_json(value)}

    flat_rows = [flatten(row) for row in rows]
    fields = sorted({key for row in flat_rows for key in row})
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def _run_rollout(args, mode, obj, seed=0, mass=0.01, lambda_scale=1.0,
                 coupling_scale=1.0, execution="wrench", steps=None):
    if mujoco is None:
        raise RuntimeError("MuJoCo is required for the rollout benchmark")
    os.environ["MUJOCO_HEADLESS"] = "0" if args.viewer else "1"
    args.obj = obj
    args.rollout = True
    args.solver = "acados"
    param = ExplicitMPCParams(args, rand_seed=int(seed),
                              target_type="ground-rotation", model="explicit")
    env = MjSimulator(param)
    _set_physical_mass(param, env, mass)
    contact = Contact(param)
    optimizer = param.lambda_optimizer
    mj_model = MujocoForwardContactModel(
        env, friction=param.mu_object_, table_friction=param.mu_table_)
    h_exec, nstep = _execution_dt(env)
    rows = []
    previous_keys = set()
    max_steps = int(args.max_steps if steps is None else steps)
    try:
        for step in range(max_steps):
            solved = _solve_step(param, args, env, contact, mode,
                                 coupling_scale=coupling_scale,
                                 lambda_scale=lambda_scale)
            refs = _reference_stats(
                solved, mj_model, optimizer, lambda_scale=1.0,
                execution_dt=h_exec, nstep=nstep,
                force_cap=float(args.execution_force_cap),
                torque_cap=float(args.execution_torque_cap))
            before = env.get_state().copy()
            current_keys = _contact_keys(env)
            transition = _transition(previous_keys, current_keys)
            previous_keys = current_keys
            _show_step(env, solved, before)
            execution_info = _step_contact_force(
                env, solved, execution=execution,
                lambda_scale=lambda_scale)
            pos_err, quat_err = _goal_errors(
                env.get_state(), solved["target_p"], solved["target_q"])
            # #region agent log
            try:
                import json as _json
                _after = env.get_state()
                _qvel = np.asarray(env.data_.qvel[:6], dtype=np.float64)
                _body = int(env.model_.geom("obj").bodyid)
                os.makedirs("/home/lab423/scsp/scsp-robot/.cursor", exist_ok=True)
                with open("/home/lab423/scsp/scsp-robot/.cursor/debug-c69ca2.log", "a", encoding="utf-8") as _lf:
                    _lf.write(_json.dumps({
                        "sessionId": "c69ca2",
                        "runId": "pre-fix",
                        "hypothesisId": "C",
                        "location": "test_scm_mujoco.py:_run_rollout",
                        "message": "after mj_step pose and mass",
                        "timestamp": int(time.time() * 1000),
                        "data": {
                            "step": int(step),
                            "pos_err": float(pos_err),
                            "quat_err": float(quat_err),
                            "success": bool(_succeeded(pos_err, quat_err)),
                            "qvel_obj": [float(x) for x in _qvel],
                            "qvel_lin_norm": float(np.linalg.norm(_qvel[:3])),
                            "qvel_ang_norm": float(np.linalg.norm(_qvel[3:])),
                            "obj_mass": float(env.model_.body_mass[_body]),
                            "obj_inertia": [float(x) for x in env.model_.body_inertia[_body]],
                            "opt_mass_diag": [float(x) for x in np.diag(optimizer.obj_inertia)],
                            "opt_h": float(optimizer.h),
                            "mujoco_unstable": bool(execution_info.get("mujoco_unstable", False)),
                            "max_abs_qacc": float(execution_info.get("max_abs_qacc", 0.0)),
                            "ncon": int(env.data_.ncon),
                        },
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            row = {
                "object": obj, "seed": int(seed), "step": int(step),
                "mass": float(mass), "lambda_scale": float(lambda_scale),
                "coupling_scale": float(coupling_scale),
                "goal_pos": pos_err, "goal_quat": quat_err,
                "success": _succeeded(pos_err, quat_err),
                "active_contacts": transition["active_contacts"],
                "scm_active_rows": int(solved["response"]["surrogate_accuracy"].get(
                    "active_rows", 0)),
                "scm_active_blocks": int(solved["response"]["surrogate_accuracy"].get(
                    "active_contacts", 0)),
                "created_contacts": transition["created"],
                "removed_contacts": transition["removed"],
                "surrogate_accuracy": solved["response"]["surrogate_accuracy"],
                "lcp_mujoco": refs,
                "execution_info": execution_info,
                "execution": execution,
                "h_model": float(solved["h"]),
                "h_exec": float(h_exec),
                "solver_requested": "acados",
                "solver_backend": str(getattr(param.lambda_optimizer, "last_solver_status",
                                               getattr(param.lambda_optimizer, "solver", "unknown"))),
                "solver_requested_backend": str(getattr(param.lambda_optimizer, "solver", "unknown")),
                "solver_fallback_reason": getattr(param.lambda_optimizer,
                                                    "last_acados_failure_reason", None),
                "solver_fallback": bool(str(getattr(param.lambda_optimizer,
                                                     "last_solver_status", ""))
                                         .endswith("fallback")),
                "mujoco_unstable": bool(execution_info.get("mujoco_unstable", False)),
                "physical_mass": float(solved.get("physical_mass", np.nan)),
                "mu_object": float(solved.get("mu_object", np.nan)),
                "mu_table": float(solved.get("mu_table", np.nan)),
                "dynamics_condition_number": float(
                    solved.get("dynamics_condition_number", np.nan)),
                "calibration_noise_std": float(
                    solved.get("calibration_noise_std", 0.0)),
            }
            rows.append(row)
            if args.viewer:
                time.sleep(0.01)
            if row["success"] or getattr(env, "break_out_signal_", False):
                break
            if env.viewer_ is not None and hasattr(env.viewer_, "is_running") \
                    and not env.viewer_.is_running():
                break
    finally:
        if env.viewer_ is not None:
            env.viewer_.close()
    return rows


def _summary(mode, rows):
    if not rows:
        print("%s summary: steps=0 success=False" % mode)
        return
    last = rows[-1]
    print("%s summary: steps=%d final_pos=%.6f final_quat=%.6f success=%s" % (
        mode, len(rows), float(last["goal_pos"]), float(last["goal_quat"]),
        bool(last["success"])))
    scored = [row["surrogate_accuracy"] for row in rows
              if row.get("surrogate_accuracy")]
    if scored:
        for key in ("direction_error", "magnitude_error", "lambda_gap_D",
                    "gamma_env", "coupling_norm"):
            values = [item[key] for item in scored
                      if item.get(key) is not None and np.isfinite(item[key])]
            if values:
                print("surrogate %s mean=%.6e" % (key, float(np.mean(values))))


def _run_sweep(args):
    objects = [item.strip() for item in args.eval_objects.split(",") if item.strip()]
    masses = _parse_float_list(args.eval_masses, (0.01, 0.1, 1.0, 10.0))
    lambda_scales = _parse_float_list(args.eval_lambda_scales, (0.25, 0.5, 1.0, 2.0))
    coupling_scales = _parse_float_list(args.eval_coupling_scales, (0.0, 0.25, 0.5, 1.0))
    rows = []
    trial_summaries = []
    for obj in objects:
        for seed in range(int(args.eval_trials)):
            for mass in masses:
                for lambda_scale in lambda_scales:
                    for coupling_scale in coupling_scales:
                        trial = _run_rollout(
                            args, "surrogate", obj, seed=seed, mass=mass,
                            lambda_scale=lambda_scale,
                            coupling_scale=coupling_scale,
                            execution=args.execution, steps=args.max_steps)
                        rows.extend(trial)
                        final = trial[-1] if trial else None
                        trial_summaries.append({
                            "object": obj, "seed": int(seed),
                            "mass": float(mass),
                            "lambda_scale": float(lambda_scale),
                            "coupling_scale": float(coupling_scale),
                            "steps": len(trial),
                            "success": bool(final and final.get("success")),
                            "final_pos": (None if final is None else final["goal_pos"]),
                            "final_quat": (None if final is None else final["goal_quat"]),
                        })
                        print("sweep object=%s seed=%d mass=%g lambda=%g coupling=%g steps=%d success=%s" % (
                            obj, seed, mass, lambda_scale, coupling_scale,
                            len(trial), bool(trial[-1]["success"]) if trial else False))
    def _wilson(successes, total, z=1.959963984540054):
        if total <= 0:
            return None
        p = float(successes) / float(total)
        den = 1.0 + z * z / total
        center = (p + z * z / (2.0 * total)) / den
        half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
        return [max(0.0, center - half), min(1.0, center + half)]

    success_count = sum(int(item["success"]) for item in trial_summaries)
    by_object = {}
    for item in trial_summaries:
        by_object.setdefault(item["object"], []).append(item)
    object_stats = {}
    for name, items in by_object.items():
        count = sum(int(item["success"]) for item in items)
        object_stats[name] = {
            "successes": count, "trials": len(items),
            "success_rate": count / len(items) if items else None,
            "wilson_95": _wilson(count, len(items)),
        }
    metadata = {
        "objects": objects, "trials": int(args.eval_trials), "masses": masses,
        "lambda_scales": lambda_scales, "coupling_scales": coupling_scales,
        "max_steps": int(args.max_steps), "execution": args.execution,
        "oracle": "mj_step",
        "trial_summaries": trial_summaries,
        "successes": success_count,
        "total_trials": len(trial_summaries),
        "success_rate": (success_count / len(trial_summaries)
                         if trial_summaries else None),
        "wilson_95": _wilson(success_count, len(trial_summaries)),
        "success_by_object": object_stats,
    }
    _write_rows(args.eval_output, rows, metadata)
    print("sweep summary: rows=%d trials=%d successes=%d success_rate=%.3f output=%s" % (
        len(rows), len(trial_summaries), success_count,
        success_count / len(trial_summaries) if trial_summaries else 0.0,
        os.path.abspath(args.eval_output)))
    for name, stats in object_stats.items():
        print("success object=%s %d/%d=%.3f CI95=[%.3f,%.3f]" % (
            name, stats["successes"], stats["trials"], stats["success_rate"],
            stats["wilson_95"][0], stats["wilson_95"][1]))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Validate SCM contact predictions against LCP and MuJoCo.")
    add_rollout_via_args(parser)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--scm-regularization", type=float, default=1e-8)
    parser.add_argument("--execution", choices=("pose", "wrench"), default="wrench")
    parser.add_argument("--execution-force-cap", type=float, default=0.1)
    parser.add_argument("--execution-torque-cap", type=float, default=1e-5)
    parser.add_argument(
        "--surrogate-param-noise", type=float, default=0.0,
        help="log-normal stddev applied to MuJoCo mass/inertia/friction for robustness tests")
    parser.add_argument("--eval-sweep", action="store_true")
    parser.add_argument("--eval-objects", default=",".join(DEFAULT_OBJECTS))
    parser.add_argument("--eval-trials", type=int, default=10)
    parser.add_argument("--eval-masses", default="0.01,0.1,1,10")
    parser.add_argument("--eval-lambda-scales", default="0.25,0.5,1,2")
    parser.add_argument("--eval-coupling-scales", default="0,0.25,0.5,1")
    parser.add_argument("--eval-output", default=os.path.join(current_dir, "scm_eval.json"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--surrogate", action="store_true")
    mode.add_argument("--qp", action="store_true")
    mode.add_argument("--lcp", action="store_true")
    args = parser.parse_args()
    if args.eval_sweep:
        _run_sweep(args)
        return
    selected = _contact_model_name(args)
    rows = _run_rollout(args, selected, args.obj, seed=0,
                        mass=0.01, lambda_scale=1.0,
                        coupling_scale=1.0, execution=args.execution,
                        steps=args.steps)
    _summary(selected, rows)


if __name__ == "__main__":
    main()
