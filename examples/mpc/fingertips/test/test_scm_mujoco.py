"""MuJoCo execution using the reference Isaac lambda contact model.

The coupled SCM rollout shares the Isaac factory and its exact prediction
graph. Optimal point forces drive MuJoCo physics; successful wrench rollouts
never assign predicted object poses. Legacy algebraic comparison helpers
remain available separately.

Choose the SCM force solver with --solver acados|ipopt|cma-es|cem.
"""

import argparse
import copy
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
from planning.rollout_timing import TIMING_DEFINITIONS, trial_timing, aggregate_trials
from planning.direct_contact_force import (
    add_direct_force_args, configure_direct_force_optimizer,
    environment_contacts_body, gravity_wrench_body, object_velocity_body,
    step_point_force,
    step_actuator_contact_force,
)
from planning.scm_contact_models import (
    LCPContactModel,
    MujocoForwardContactModel,
    appendix_accuracy,
    compare_lambda,
    contact_jacobian,
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


class RolloutPlanningFailure(RuntimeError):
    """Carry attempted rounds so a failed solve remains in timing statistics."""
    def __init__(self, cause, rows):
        super().__init__('%s: %s'%(type(cause).__name__,cause))
        self.rows = rows


class RolloutExecutionFailure(RolloutPlanningFailure):
    """Preserve completed planning rounds when the physical rollout fails."""


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
        obj_body = model.body("obj").id
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
    body_id = model.body("obj").id
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
    body_id = env.model_.body('obj').id
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
                    gravity, mode, coupling_scale=0.0, regularization=1e-8,
                    velocity=None, current_pose=None, reference_predictor=None):
    h = float(optimizer.h)
    q_inv = np.asarray(optimizer.Q_inv, dtype=np.float64)
    J_arm = contact_jacobian(p_local)
    frame = np.column_stack((n_arm, t1, t2))
    if optimizer.contact_model == 'comfree':
        smooth = (np.zeros(6) if velocity is None else np.asarray(velocity)) + q_inv @ (
            optimizer.gravity_velocity_scale * gravity +
            optimizer.wrench_velocity_scale * J_arm.T @ (frame @ lam))
        weights = optimizer.compute_env_diag_inverse(jacobian)
        v_plus = optimizer._contact_velocity_numpy(smooth, jacobian, weights, phi)
        x_plus = np.asarray(optimizer.cs_qposInteg_(current_pose, v_plus)).reshape(7)
        response = dict(v_plus=v_plus, x_plus=x_plus, model='comfree')
        return dict(selected=response, surrogate=response, regularization=regularization)
    if reference_predictor is not None:
        response = reference_predictor.predict(lam, p_local, frame)
        return dict(selected=response, surrogate_accuracy={}, regularization=regularization)
    if optimizer.robot_stiffness is not None:
        if mode != 'surrogate':
            raise ValueError('Isaac-compatible execution uses coupled soft-SCM; independent QP/LCP is a different model')
        response = optimizer.predict_contact_response(
            lam, p_local, frame, gravity, velocity=velocity, current_pose=current_pose)
        return dict(selected=response, surrogate=response,
                    regularization=optimizer.soft_scm_regularization)
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
    direct = bool(getattr(env, 'direct_force_', False))
    velocity = None
    if direct:
        source = param.lambda_source_param_
        jac_mat_env, phi = environment_contacts_body(env, source.mu_table_object_)
        from examples.mpc.franka.runtime.contact import gravity_wrench_object_frame
        gravity = gravity_wrench_object_frame(source.gravity_[:3], source.lambda_obj_mass_, r_obj)
        # Isaac's default quasi-dynamic ranking starts each prediction at
        # rest. Only shared-response variants carry measured velocity.
        velocity = optimizer.rank_object_velocity.copy()
        if optimizer.contact_model == 'diagonal-contact':
            velocity = object_velocity_body(env)
        stages = param.force_cost_stages_
        observed_linear = env.data_.qvel[:3].copy()
        observed_angular = r_obj @ env.data_.qvel[3:6]
        previous = getattr(param, 'force_observation_pose_', None)
        if previous is not None and env.data_.time > previous[0]:
            dt = float(env.data_.time - previous[0])
            observed_linear = (curr_q[:3] - previous[1][:3]) / dt
            now_rot = Rotation.from_quat(curr_q[[4, 5, 6, 3]])
            old_rot = Rotation.from_quat(previous[1][[4, 5, 6, 3]])
            observed_angular = (now_rot * old_rot.inv()).as_rotvec() / dt
        # Observe motion over one policy interval. Instantaneous mesh-contact
        # solver chatter otherwise prevents the shared 0.3 s flip-stability
        # gate from recognizing an object whose pose is already stationary.
        param.force_observation_pose_ = (float(env.data_.time), curr_q[:7].copy())
        changed = stages.update(dict(sim_time=env.data_.time, policy_q=curr_q,
                                     object_linear_velocity=observed_linear,
                                     object_angular_velocity=observed_angular))
        if changed:
            optimizer._dwell_idx = optimizer._dwell_best_cost = None
            optimizer._dwell_steps = 0
            optimizer._dwell_blocked = False
            optimizer._blocked_contact_indices.clear()
            optimizer.set_contact_switch_confidence(1.)
        from planning.object_pose_cost import pose_only_config
        config = stages.config(r_obj)
        if optimizer.cso_guidance == 'pose':
            config = pose_only_config(stages.pos_weight, stages.ori_weight,
                                      stages.body_axis, r_obj.T @ np.array([0., 0., 1.]))
        optimizer.set_object_cost_config(config)
    jac_mat_env = optimizer.update_Jacobian(jac_mat_env, phi=phi if direct else None)
    if direct:
        phi = optimizer.env_phi.copy()
    visible = optimizer.get_availble_point_idx(
        curr_q[:3], r_obj, param.target_p_, args.ground_height_threshold,
        viewpoint_local=None, heading_filter=False)
    visible = optimizer.filter_rankable_indices(visible)
    predictor = getattr(param, 'force_prediction_model_', None)
    if predictor is not None:
        predictor.prepare()
    point, normal, _, _, _ = optimizer.choose_contact_points(
        target_pose, current_pose, gravity, visible, v_last=velocity,
        # This benchmark places the ideal actuator directly on the selected
        # patch. Match Isaac's made-contact ranking: a gravity-only/zero-drive
        # prediction must not masquerade as useful executed contact.
        force_required=True, query_local=tip_local)
    if not len(visible) or optimizer.last_selected_idx is None:
        raise RuntimeError('No feasible lambda contact point above the table')
    if direct and (optimizer.last_best_x_plus is None or
                   not np.isfinite(optimizer.last_best_cost)):
        raise RuntimeError('Lambda failed to produce a finite optimal force')
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
        regularization=float(getattr(args, "scm_regularization", 1e-8)),
        velocity=velocity, current_pose=current_pose, reference_predictor=predictor)
    response["surrogate_accuracy"] = {} if direct else appendix_accuracy(
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
        "actuator_force": bool(direct),
        "prediction_velocity": velocity,
        "object_cost_stage": param.force_cost_stages_.stage if direct else 'pose',
        "h": float(optimizer.h), "gravity": gravity,
        "wrench_is_force": bool(getattr(optimizer, "wrench_is_force", False)),
        "execution_force_cap": float(getattr(args, "execution_force_cap", 0.2)),
        "execution_torque_cap": float(getattr(args, "execution_torque_cap", 0.0)),
        "physical_mass": float(param.obj_mass_),
        "physical_inertia": np.diag(env.model_.body_inertia[env.model_.body('obj').id]),
        "mu_object": float(optimizer.mu_arm_obj),
        "mu_table": float(param.lambda_source_param_.mu_table_object_) if direct else float(param.mu_table_),
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
    qpos = env.get_state()
    h_exec, nstep = _execution_dt(env)
    if execution == "wrench":
        if solved.get('actuator_force', False):
            return step_actuator_contact_force(env, solved)
        # _solve_step already scales lambda once. Older callers without
        # lam_base supply an unscaled lambda; scale it exactly once here.
        lam = np.asarray(solved.get("lam", np.zeros(3)), dtype=float).copy()
        if "lam_base" not in solved:
            lam *= float(lambda_scale)
        frame = np.column_stack((solved["n_arm"], solved["t1"], solved["t2"]))
        force = frame @ lam
        if not solved.get("wrench_is_force", False):
            force /= h_exec
        com = env.model_.body_ipos[env.model_.body('obj').id]
        torque = np.cross(np.asarray(solved["p_local"]) - com, force)
        unclipped = np.r_[force, torque]
        cap = float(solved.get("execution_force_cap", 0.0))
        torque_cap = float(solved.get("execution_torque_cap", 0.0))
        scale = 1.0
        if cap > 0 and np.linalg.norm(force) > cap:
            scale = min(scale, cap / np.linalg.norm(force))
        if torque_cap > 0 and np.linalg.norm(torque) > torque_cap:
            scale = min(scale, torque_cap / np.linalg.norm(torque))
        # A moment bound must scale the whole force; clipping only torque
        # changes the requested surface point into an impossible wrench.
        info = step_point_force(env, solved["p_local"], scale * force, nstep)
        return dict(info, execution=execution,
                    wrench_impulse_body=info["actual_applied_impulse_body"].copy(),
                    wrench_impulse_raw_body=unclipped*h_exec,
                    force_body_unclipped=unclipped,
                    force_cap=cap, torque_cap=torque_cap,
                    force_was_clipped=bool(scale < 1.0 - 1e-6),
                    force_scale=scale,
                    actuator_force_contact=lam,
                    contact_reaction_body=None,
                    force_units="N", impulse_units="N*s")
    selected = solved["response"]["selected"]
    x_plus = _integrate_local_pose(qpos[:7], selected["v_plus"], solved["h"])
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
    args.solver = getattr(args, 'solver', 'acados')
    construction_args = copy.copy(args)
    if args.solver == 'batched-lm':
        construction_args.solver = 'torch-gn'
    param = ExplicitMPCParams(construction_args, rand_seed=int(seed),
                              target_type="ground-rotation", model="explicit")
    param.trial_seed_ = int(seed)
    env = MjSimulator(param)
    env.force_args_ = args
    if mass is None:
        mass = float(env.model_.body_mass[env.model_.body('obj').id])
    _set_physical_mass(param, env, mass)
    configure_direct_force_optimizer(param, env, construction_args)
    if args.solver == 'batched-lm':
        from planning.scm_batched_lm import GPUExactBatchLM
        lm = GPUExactBatchLM(param.lambda_optimizer, iterations=args.lm_iterations)
        lm.install()
        # Verify the CUDA contact map and objective against the current
        # CasADi SCM graph before accepting any benchmark result.
        lm.validate = True
        _solve_step(param, args, env, Contact(param), mode)
        lm.validate = False
        param.scm_lm_validation_ = lm.validations
    if mode in ('qp', 'lcp'):
        from planning.point_contact_baselines import install_point_contact_baseline
        install_point_contact_baseline(env, mode, maxiter=getattr(args, 'baseline_maxiter', 12))
    contact = Contact(param)
    optimizer = param.lambda_optimizer
    h_exec, nstep = _execution_dt(env)
    rows = []
    previous_keys = set()
    max_steps = int(args.max_steps if steps is None else steps)
    success_hold = 0
    hold_steps = max(1, int(getattr(args, 'success_hold_steps', 10)))
    async_planner = None
    if getattr(args, 'async_planner', False):
        if execution != 'wrench' or mode != 'surrogate':
            raise ValueError('Async planner requires SCM wrench execution')
        from planning.direct_force_async import AsyncForcePlanner
        async_planner = AsyncForcePlanner(env, param, args, _solve_step, Contact,
                                         mode, coupling_scale, lambda_scale)
    # Scene/model construction and compilation are deliberately outside this
    # clock. Lazy work triggered by a planning round remains inside its timing.
    rollout_start = time.perf_counter()
    simulation_start = float(env.data_.time)
    try:
        for step in range(max_steps):
            planning_start = time.perf_counter()
            try:
                if async_planner is None:
                    solved = _solve_step(param, args, env, contact, mode,
                                         coupling_scale=coupling_scale,
                                         lambda_scale=lambda_scale)
                    plan_updated = True
                else:
                    solved, plan_updated = async_planner.poll(env)
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                pos,quat = _goal_errors(env.get_state(),param.target_p_,param.target_q_)
                rows.append(dict(object=obj,seed=int(seed),step=step,success=False,
                                 settled=False,goal_pos=pos,goal_quat=quat,
                                 planning_time_ms=1000.*(time.perf_counter()-planning_start),
                                 elapsed_wall_time_s=time.perf_counter()-rollout_start,
                                 simulation_time_s=float(env.data_.time)-simulation_start,
                                 execution_time_s=None,planning_error=str(exc),
                                 initial_object_pose=env.get_state()[:7].copy() if step==0 else None))
                raise RolloutPlanningFailure(exc,rows) from exc
            planning_time_ms = 1000.*(time.perf_counter()-planning_start)
            if async_planner is not None:
                planning_time_ms = solved['planning_time_ms'] if plan_updated else 0.
            # The old diagonal-SCM/LCP oracle uses a different mobility map.
            # Record the shared predictor and measured motion directly.
            refs = {'prediction_model': solved['response']['selected']['model']}
            before = env.get_state().copy()
            current_keys = _contact_keys(env)
            transition = _transition(previous_keys, current_keys)
            previous_keys = current_keys
            _show_step(env, solved, before)
            # Reaching the tolerance starts an unforced settling interval.
            # If inertia carries the object outside it, resume optimization.
            before_errors = _goal_errors(before, solved['target_p'], solved['target_q'])
            stages = solved['stages'] if async_planner is not None else param.force_cost_stages_
            flip_park = bool(getattr(param, 'force_flip_settling_', False))
            if stages.stage != 'flip' or stages.angle is None or stages.angle > stages.enter_angle + 2.:
                flip_park = False
            elif stages.angle <= stages.enter_angle:
                flip_park = True
            param.force_flip_settling_ = flip_park
            settling = (execution == 'wrench' and before_errors[0] < 0.016 and
                        before_errors[1] < 0.010) or (execution == 'wrench' and
                        success_hold > 0 and _succeeded(*before_errors))
            if settling or flip_park:
                solved = dict(solved, lam=np.zeros(3), settling=True)
            execution_info = _step_contact_force(
                env, solved, execution=execution,
                lambda_scale=lambda_scale)
            pos_err, quat_err = _goal_errors(
                env.get_state(), solved["target_p"], solved["target_q"])
            pose_success = _succeeded(pos_err, quat_err)
            success_hold = success_hold + 1 if pose_success and settling else 0
            velocity = object_velocity_body(env)
            settled = bool(np.linalg.norm(velocity[:3]) < 0.05 and
                           np.linalg.norm(velocity[3:]) < 0.5)
            success = (pose_success if execution == 'pose' else
                       pose_success and success_hold >= hold_steps and settled)
            row = {
                "object": obj, "seed": int(seed), "step": int(step),
                "planning_time_ms": planning_time_ms,
                "simulation_time_s": float(env.data_.time)-simulation_start,
                "mass": float(mass), "lambda_scale": float(lambda_scale),
                "coupling_scale": float(coupling_scale),
                "goal_pos": pos_err, "goal_quat": quat_err,
                "success": bool(success), "pose_success": bool(pose_success),
                "settling": bool(settling), "success_hold": success_hold,
                "velocity_body": velocity.copy(), "settled": settled,
                "object_pose": env.get_state()[:7].copy(),
                "initial_object_pose": before[:7].copy() if step == 0 else None,
                "best_contact_point": solved['p_local'].copy(),
                "optimal_force_contact": solved['lam_base'].copy(),
                "selected_idx": int(solved.get('selected_idx', optimizer.last_selected_idx)),
                "async_planner": async_planner is not None,
                "planning_updated": plan_updated,
                "command_age_s": float(env.data_.time - solved.get('observation_time', env.data_.time)),
                "contact_impratio": float(env.model_.opt.impratio),
                "contact_noslip_iterations": int(env.model_.opt.noslip_iterations),
                "lm_validation": getattr(param, 'scm_lm_validation_', []) if step == 0 else None,
                "predicted_x_plus": solved['response']['selected']['x_plus'].copy(),
                "object_cost_stage": solved['object_cost_stage'],
                "optimizer_model": solved['response']['selected']['model'],
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
                "solver_requested": args.solver if mode == 'surrogate' else 'slsqp',
                "solver_backend": solved.get('solver_status', str(getattr(param.lambda_optimizer, "last_solver_status",
                                               getattr(param.lambda_optimizer, "solver", "unknown")))),
                "solver_requested_backend": (str(getattr(param.lambda_optimizer, "solver", "unknown"))
                                             if mode == 'surrogate' else 'slsqp-local-linearization'),
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
            if async_planner is None and not solved.get('settling', False):
                from planning.object_pose_cost import object_cost_np
                progress = object_cost_np(env.get_state()[:7],
                                          np.r_[param.target_p_, param.target_q_],
                                          stages.config()) / stages.progress_scale()
                optimizer.note_contact_progress(
                    row['selected_idx'], progress, active=True,
                    gamma=float(args.contact_dwell_gamma),
                    min_dwell_steps=max(80, int(args.contact_dwell_steps)),
                    improve_eps=1e-3, merge_radius=.03, block_radius=.03,
                    block_cycles=80, time_decay=True)
                row['blocked_contact_count'] = len(optimizer._blocked_contact_indices)
            row['elapsed_wall_time_s'] = time.perf_counter()-rollout_start
            row['execution_time_s'] = row['elapsed_wall_time_s'] if success else None
            predictor = getattr(param, 'force_prediction_model_', None)
            if predictor is not None:
                row['model_prediction_evaluations'] = predictor.evaluations
                row['inner_force_solves'] = predictor.inner_solves
                row['inner_nonconverged'] = predictor.inner_nonconverged
            if step % 50 == 0 or success:
                print('direct_force solver=%s object=%s seed=%d step=%d pos=%.6f quat=%.6f hold=%d success=%s' % (
                    args.solver, obj, seed, step, pos_err, quat_err, success_hold, success), flush=True)
            if args.viewer:
                time.sleep(0.01)
            if async_planner is not None:
                # Common real-time physics clock: continue applying the latest
                # command while inference runs, without waiting for each solve.
                time.sleep(max(0., rollout_start + (step+1)*h_exec - time.perf_counter()))
            if row['mujoco_unstable']:
                row['success'] = False
                row['execution_time_s'] = None
                raise RolloutExecutionFailure(
                    RuntimeError('MuJoCo became unstable during direct force execution'),rows)
            if row["success"] or getattr(env, "break_out_signal_", False):
                break
            if env.viewer_ is not None and hasattr(env.viewer_, "is_running") \
                    and not env.viewer_.is_running():
                break
    finally:
        if async_planner is not None:
            async_planner.close()
        if env.viewer_ is not None:
            env.viewer_.close()
    return rows


def run_direct_force(args):
    """test_0902's direct-force mode: replan from measured state until settled."""
    args.skip_model_comparison = True
    all_rows, trials = [], []
    for seed in range(max(1, int(args.trial_num))):
        rows = _run_rollout(args, 'surrogate', args.obj, seed=seed, mass=None,
                            execution='wrench', steps=max(1, int(args.max_rollout_length)))
        all_rows.extend(rows)
        last = rows[-1]
        trials.append(dict(seed=seed, steps=len(rows), success=last['success'],
                           final_pos=last['goal_pos'], final_quat=last['goal_quat'],
                           settled=last['settled'], **trial_timing(rows)))
    successes = sum(t['success'] for t in trials)
    metadata = dict(object=args.obj, execution='wrench', pose_overwrites=0,
                    position_threshold=POS_SUCCESS, quaternion_threshold=QUAT_SUCCESS,
                    success_hold_steps=args.success_hold_steps,
                    successes=successes, total_trials=len(trials),
                    trial_summaries=trials, args=vars(args).copy(),
                    timing_definitions=TIMING_DEFINITIONS,
                    timing_statistics=aggregate_trials(trials, all_rows))
    if args.force_output:
        _write_rows(args.force_output, all_rows, metadata)
    print('direct_force summary: %d/%d succeeded' % (successes, len(trials)), flush=True)
    return successes == len(trials)


def _summary(mode, rows):
    if not rows:
        print("%s summary: steps=0 success=False" % mode)
        return
    last = rows[-1]
    print("%s summary: steps=%d final_pos=%.6f final_quat=%.6f success=%s" % (
        mode, len(rows), float(last["goal_pos"]), float(last["goal_quat"]),
        bool(last["success"])))
    if 'planning_time_ms' in last:
        timing = trial_timing(rows)
        print('%s timing: execution_time_s=%s planning_time_mean_ms=%.3f planning_time_p95_ms=%.3f simulation_time_s=%.3f' % (
            mode, timing['execution_time_s'], timing['planning_time_ms']['mean'],
            timing['planning_time_ms']['p95'], timing['simulation_time_s']))
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
                            **trial_timing(trial),
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
        "timing_definitions": TIMING_DEFINITIONS,
        "timing_statistics": aggregate_trials(trial_summaries, rows),
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


def build_parser():
    parser = argparse.ArgumentParser(
        description="Execute the shared Isaac SCM optimizer with MuJoCo point forces.")
    add_rollout_via_args(parser)
    add_direct_force_args(parser)
    parser.add_argument('--solver', '--solver-backend', dest='solver',
                        choices=('acados', 'ipopt', 'cma-es', 'cem', 'batched-lm'), default='acados',
                        help='SCM per-contact force optimizer (default: acados).')
    parser.add_argument('--async-planner', action='store_true')
    parser.add_argument('--lm-iterations', type=int, default=20)
    parser.add_argument('--population-size', type=int, default=24,
                        help='CMA-ES/CEM samples per generation per contact.')
    parser.add_argument('--population-iterations', type=int, default=12,
                        help='CMA-ES/CEM maximum generations per contact.')
    parser.add_argument('--solver-seed', type=int, default=0,
                        help='CMA-ES/CEM random seed offset added to the trial seed.')
    parser.add_argument('--cem-elite-fraction', type=float, default=0.25,
                        help='CEM fraction of lowest-cost samples used to update its distribution.')
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--scm-regularization", type=float, default=1e-8)
    parser.add_argument('--baseline-maxiter', type=int, default=12,
                        help='QP/LCP per-candidate SLSQP force iteration budget.')
    parser.add_argument("--execution", choices=("pose", "wrench"), default="wrench")
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
    return parser


def main():
    args = build_parser().parse_args()
    if args.eval_sweep:
        _run_sweep(args)
        return
    selected = _contact_model_name(args)
    rows = _run_rollout(args, selected, args.obj, seed=0,
                        mass=0.01, lambda_scale=1.0,
                        coupling_scale=1.0, execution=args.execution,
                        steps=args.steps)
    _summary(selected, rows)
    if args.force_output:
        _write_rows(args.force_output, rows, dict(object=args.obj, execution=args.execution,
                    successes=int(bool(rows and rows[-1]['success'])), total_trials=1,
                    args=vars(args).copy(), timing_definitions=TIMING_DEFINITIONS,
                    **trial_timing(rows)))
    if not rows or not rows[-1]['success']:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
