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
from examples.mpc.fingertips.test.params import ExplicitMPCParams
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
    ratio = mass / old
    model.body_mass[body_id] = mass
    model.body_inertia[body_id] *= ratio
    if hasattr(mujoco, "mj_setConst"):
        mujoco.mj_setConst(model, env.data_)
    param.obj_mass_ = mass
    mujoco.mj_forward(model, env.data_)


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
        "execution_force_cap": float(getattr(args, "execution_force_cap", 2.0)),
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


def _step_contact_force(env, solved, execution="pose", lambda_scale=1.0):
    """Apply one direct-apply step without touching the shared optimizer."""
    selected = solved["response"]["selected"]
    qpos = env.get_state()
    x_plus = _integrate_local_pose(qpos[:7], selected["v_plus"], solved["h"])
    if execution == "wrench":
        if mujoco is None:
            raise RuntimeError("MuJoCo is required for wrench execution")
        obj_body = int(env.model_.geom("obj").bodyid)
        rot = _object_frame(qpos)
        lam_apply = solved.get("lam", np.zeros(3))
        if "lam_base" not in solved:
            lam_apply = lam_apply * float(lambda_scale)
        wrench = contact_wrench(
            solved["p_local"], solved["n_arm"], solved["t1"], solved["t2"],
            lam_apply)
        cap = float(solved.get("execution_force_cap", 2.0))
        norm = float(np.linalg.norm(wrench[:3]))
        if cap > 0.0 and norm > cap:
            wrench *= cap / norm
        env.data_.xfrc_applied[:] = 0.0
        env.data_.xfrc_applied[obj_body, :3] = rot @ wrench[:3]
        env.data_.xfrc_applied[obj_body, 3:] = rot @ wrench[3:]
        mujoco.mj_step(env.model_, env.data_, nstep=max(1, int(env.param_.frame_skip_)))
        env.data_.xfrc_applied[:] = 0.0
        env._sync_viewer()
        return
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


def _reference_stats(solved, mujoco_model, optimizer, lambda_scale=1.0):
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
            tau_body=None, lambda_scale=lambda_scale)
        mj_last = dict(mujoco_model.last or {})
        out["vs_mujoco"] = lcp_mujoco_motion_accuracy(
            optimizer.obj_inertia, v_mj, v_scm, solved["h"])
        out["qp_vs_mujoco"] = lcp_mujoco_motion_accuracy(
            optimizer.obj_inertia, v_mj, response["qp"]["v_plus"], solved["h"])
        measured = np.asarray(mj_last.get("lambda_env", []), dtype=np.float64)
        active_rows = np.asarray(response["surrogate"].get("rows", []), dtype=np.int64)
        predicted_full = np.asarray(response["surrogate"]["lambda_env"], dtype=np.float64)
        predicted = predicted_full[active_rows] if active_rows.size else predicted_full[:0]
        n = min(measured.size, predicted.size)
        if n:
            out["vs_mujoco"].update(compare_lambda(predicted[:n], measured[:n]))
            qp_full = np.asarray(response["qp"]["lambda_env"], dtype=np.float64)
            qp_lam = qp_full[active_rows] if active_rows.size else qp_full[:0]
            out["qp_vs_mujoco"].update(compare_lambda(qp_lam[:n], measured[:n]))
        out["mujoco_env_wrench_body"] = mj_last.get("env_wrench_body", np.zeros(6))
        out["mujoco_lambda_env"] = measured
        out["mujoco_ncon"] = int(mj_last.get("ncon", 0))
        qp_motion = out["qp_vs_mujoco"]
        qp_lam_error = float(qp_motion.get("lambda_env_relative_error", 0.0))
        qp_mag_error = float(qp_motion.get("relative_magnitude_error", 0.0))
        qp_dir_error = qp_motion.get("direction_error")
        out["qp_matches_mujoco"] = bool(
            (qp_dir_error is None or qp_dir_error <= 1e-3) and
            qp_mag_error <= 1e-3 and qp_lam_error <= 1e-3)
    except (RuntimeError, ValueError, FloatingPointError):
        out["vs_mujoco"] = None
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
                 coupling_scale=1.0, execution="pose", steps=None):
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
    mj_model = MujocoForwardContactModel(env, friction=param.mu_object_)
    rows = []
    previous_keys = set()
    max_steps = int(args.max_steps if steps is None else steps)
    try:
        for step in range(max_steps):
            solved = _solve_step(param, args, env, contact, mode,
                                 coupling_scale=coupling_scale,
                                 lambda_scale=lambda_scale)
            refs = _reference_stats(solved, mj_model, optimizer,
                                    lambda_scale=1.0)
            before = env.get_state().copy()
            current_keys = _contact_keys(env)
            transition = _transition(previous_keys, current_keys)
            previous_keys = current_keys
            _show_step(env, solved, before)
            _step_contact_force(env, solved, execution=execution,
                                lambda_scale=lambda_scale)
            pos_err, quat_err = _goal_errors(
                env.get_state(), solved["target_p"], solved["target_q"])
            row = {
                "object": obj, "seed": int(seed), "step": int(step),
                "mass": float(mass), "lambda_scale": float(lambda_scale),
                "coupling_scale": float(coupling_scale),
                "goal_pos": pos_err, "goal_quat": quat_err,
                "success": _succeeded(pos_err, quat_err),
                "active_contacts": transition["active_contacts"],
                "scm_contacts": int(solved["response"]["surrogate_accuracy"].get(
                    "active_contacts", 0)),
                "contact_set_mismatch": bool(
                    transition["active_contacts"] != int(
                        solved["response"]["surrogate_accuracy"].get(
                            "active_contacts", 0))),
                "created_contacts": transition["created"],
                "removed_contacts": transition["removed"],
                "surrogate_accuracy": solved["response"]["surrogate_accuracy"],
                "lcp_mujoco": refs,
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
                        print("sweep object=%s seed=%d mass=%g lambda=%g coupling=%g steps=%d success=%s" % (
                            obj, seed, mass, lambda_scale, coupling_scale,
                            len(trial), bool(trial[-1]["success"]) if trial else False))
    metadata = {
        "objects": objects, "trials": int(args.eval_trials), "masses": masses,
        "lambda_scales": lambda_scales, "coupling_scales": coupling_scales,
        "max_steps": int(args.max_steps), "execution": args.execution,
        "oracle": "mj_forward",
    }
    _write_rows(args.eval_output, rows, metadata)
    successes = [row for row in rows if row.get("success")]
    print("sweep summary: rows=%d success_rows=%d output=%s" % (
        len(rows), len(successes), os.path.abspath(args.eval_output)))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Validate SCM contact predictions against LCP and MuJoCo.")
    add_rollout_via_args(parser)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--scm-regularization", type=float, default=1e-8)
    parser.add_argument("--execution", choices=("pose", "wrench"), default="pose")
    parser.add_argument("--execution-force-cap", type=float, default=2.0)
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
