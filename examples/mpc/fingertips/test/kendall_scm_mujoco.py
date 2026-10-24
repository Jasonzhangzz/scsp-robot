#!/usr/bin/env python3
"""Matched-state Kendall ranking consistency for CSO contact candidates.

For each frozen MuJoCo state, the script reuses the candidate set and force
solutions produced by ``test_scm_mujoco.py``.  The paper SCM and MuJoCo then
score the same candidate point/force pairs.  Kendall's tau-b is aggregated
over states; the reported similarity is ``S_tau=(tau+1)/2`` so that 1 means
identical ordering, 0.5 means no ordinal association, and 0 means reversed
ordering.  Invalid or tied-only states are excluded from the mean.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.mpc.fingertips.test import test_scm_mujoco as tsm
from examples.mpc.fingertips.test.params import ExplicitMPCParams
from examples.mpc.fingertips.test.test_0902 import _predicted_object_pose
from examples.mpc.fingertips.test.test_scm_mujoco import (
    _model_response,
    _paper_scm_response,
    _object_frame,
    _set_physical_mass,
    _solve_step,
    _execution_dt,
)
from contact.fingertips_collision_detection2 import Contact
from envs.fingertips_env import MjSimulator
from planning.direct_contact_force import (
    configure_direct_force_optimizer,
    environment_contacts_body,
)
from planning.scm_contact_models import contact_wrench
from examples.mpc.franka.runtime.contact import gravity_wrench_object_frame


DEFAULT_OBJECTS = (
    "foam_brick", "mug", "rubber_duck", "elephant", "piggy_bank",
    "stanford_bunny2", "teapot",
)


def pose_cost(qpos, target_p, target_q, pos_coef=500.0, ori_coef=20.0):
    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    target_p = np.asarray(target_p, dtype=np.float64).reshape(3)
    target_q = np.asarray(target_q, dtype=np.float64).reshape(4)
    q = qpos[3:7] / max(float(np.linalg.norm(qpos[3:7])), 1e-12)
    target_q = target_q / max(float(np.linalg.norm(target_q)), 1e-12)
    return float(pos_coef) * float(np.sum((qpos[:3] - target_p) ** 2)) + \
        float(ori_coef) * (1.0 - float(np.dot(q, target_q)) ** 2)


def kendall_similarity(a, b):
    """Return tau-b and its [0,1] similarity transform for two cost maps."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(valid)) < 2:
        return None
    tau, pvalue = kendalltau(a[valid], b[valid], variant="b")
    if not np.isfinite(tau):
        return None
    return {"tau": float(tau), "similarity": float((tau + 1.0) / 2.0),
            "pvalue": float(pvalue), "n_candidates": int(np.count_nonzero(valid))}


def _candidate_pose_mujoco(env, optimizer, point, normal, tangent1, tangent2, lam):
    """Apply one candidate point impulse in MuJoCo and return the next pose."""
    model, data = env.model_, env.data_
    body = int(model.body("obj").id)
    qpos, qvel, ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    xfrc, qfrc, time = data.xfrc_applied.copy(), data.qfrc_applied.copy(), float(data.time)
    try:
        body_rot = np.asarray(data.xmat[body], dtype=np.float64).reshape(3, 3)
        wrench = contact_wrench(point, normal, tangent1, tangent2, lam)
        execution_dt, nstep = _execution_dt(env)
        if bool(getattr(optimizer, "wrench_is_force", False)):
            force_body = wrench
        else:
            force_body = wrench / execution_dt
        data.xfrc_applied[:] = 0.0
        data.xfrc_applied[body, :3] = body_rot @ force_body[:3]
        data.xfrc_applied[body, 3:] = body_rot @ force_body[3:]
        mujoco.mj_step(model, data, nstep=nstep)
        return np.asarray(data.qpos[:7], dtype=np.float64).copy()
    finally:
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = ctrl
        data.xfrc_applied[:] = xfrc
        data.qfrc_applied[:] = qfrc
        data.time = time
        mujoco.mj_forward(model, data)


def _state_context(param, env):
    """Rebuild the direct-force state inputs used by ``_solve_step``."""
    curr_q = env.get_state()
    R = _object_frame(curr_q)
    if getattr(env, "direct_force_", False):
        source = param.lambda_source_param_
        jacobian, phi = environment_contacts_body(env, source.mu_table_object_)
        gravity = gravity_wrench_object_frame(source.gravity_[:3], source.lambda_obj_mass_, R)
    else:
        jacobian = np.asarray(param.lambda_optimizer.J_tilde, dtype=np.float64)
        phi = np.asarray(param.lambda_optimizer.env_phi, dtype=np.float64)
        gravity = np.hstack([R.T @ param.gravity_[:3] * param.obj_mass_, np.zeros(3)])
    velocity = param.lambda_optimizer.rank_object_velocity.copy()
    return curr_q, jacobian, phi, gravity, velocity


def score_state(param, env, args, contact):
    """Solve one CSO state and compare all candidate cost rankings."""
    solved = _solve_step(param, args, env, contact, "surrogate")
    optimizer = param.lambda_optimizer
    ids = np.asarray(getattr(optimizer, "last_candidate_ids", []), dtype=np.int64).reshape(-1)
    forces = getattr(optimizer, "last_candidate_forces", None)
    if ids.size == 0 or forces is None or len(forces) != ids.size:
        return None
    curr_q, jacobian, phi, gravity, velocity = _state_context(param, env)
    target = np.r_[param.target_p_, param.target_q_]
    scm_costs, mujoco_costs = [], []
    for local, idx in enumerate(ids):
        lam = np.asarray(forces[local], dtype=np.float64).reshape(3)
        if not np.isfinite(lam).all():
            scm_costs.append(np.nan); mujoco_costs.append(np.nan); continue
        point = np.asarray(optimizer.sample_point[int(idx)], dtype=np.float64)
        normal = np.asarray(optimizer.normal[int(idx)], dtype=np.float64)
        tangent1 = np.asarray(optimizer.t1[int(idx)], dtype=np.float64)
        tangent2 = np.asarray(optimizer.t2[int(idx)], dtype=np.float64)
        response = _paper_scm_response(
            optimizer, jacobian, phi, lam, point, normal, tangent1, tangent2,
            gravity, regularization=float(args.scm_regularization),
            velocity=velocity, current_pose=np.array([0., 0., 0., 1., 0., 0., 0.]),
            temperature=float(args.scm_softplus_temperature))
        predicted_pos, predicted_quat = _predicted_object_pose(
            curr_q[:7], response["selected"]["x_plus"])
        scm_costs.append(pose_cost(np.r_[predicted_pos, predicted_quat],
                                   param.target_p_, param.target_q_,
                                   args.pos_coef, args.ori_coef))
        try:
            measured = _candidate_pose_mujoco(
                env, optimizer, point, normal, tangent1, tangent2, lam)
            mujoco_costs.append(pose_cost(measured, param.target_p_, param.target_q_,
                                          args.pos_coef, args.ori_coef))
        except (RuntimeError, FloatingPointError, ValueError):
            mujoco_costs.append(np.nan)
    result = kendall_similarity(scm_costs, mujoco_costs)
    if result is None:
        return None
    result.update(object=args.obj, seed=int(getattr(param, "trial_seed_", 0)),
                  solver=str(args.solver), selected_idx=int(optimizer.last_selected_idx),
                  n_candidates=int(len(ids)))
    return result


def run_solver(args, solver, objects, states_per_object):
    rows = []
    for obj in objects:
        for seed in range(states_per_object):
            trial_args = copy.copy(args)
            trial_args.obj = obj
            trial_args.solver = solver
            trial_args.rollout = True
            trial_args.viewer = False
            trial_args.headless = True
            trial_args.direct_force = True
            os.environ["MUJOCO_HEADLESS"] = "1"
            param = ExplicitMPCParams(trial_args, rand_seed=seed,
                                      target_type="ground-rotation", model="explicit")
            param.trial_seed_ = seed
            env = MjSimulator(param)
            env.force_args_ = trial_args
            _set_physical_mass(param, env, 0.01)
            # Candidate evaluation applies an ideal point wrench directly to
            # the object. Disable the fingertip collision geometry so the
            # MuJoCo replay contains only object--table contact, matching the
            # CSO model and the direct-force benchmark.
            for geom_id in range(env.model_.ngeom):
                name = mujoco.mj_id2name(env.model_, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                if "fingertip" in name or "finger" in name:
                    env.model_.geom_contype[geom_id] = 0
                    env.model_.geom_conaffinity[geom_id] = 0
            env.direct_force_ = False
            mujoco.mj_forward(env.model_, env.data_)
            contact = Contact(param)
            try:
                row = score_state(param, env, trial_args, contact)
                if row is not None:
                    rows.append(row)
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                print(f"skip object={obj} seed={seed} solver={solver}: {exc}", flush=True)
            finally:
                if getattr(env, "viewer_", None) is not None:
                    env.viewer_.close()
    return rows


def aggregate(rows):
    values = np.asarray([r["similarity"] for r in rows], dtype=np.float64)
    taus = np.asarray([r["tau"] for r in rows], dtype=np.float64)
    if not len(values):
        return {"n": 0, "similarity_mean": None, "similarity_std": None,
                "tau_mean": None, "tau_std": None}
    return {"n": int(len(values)),
            "similarity_mean": float(values.mean()),
            "similarity_std": float(values.std()),
            "tau_mean": float(taus.mean()), "tau_std": float(taus.std())}


def build_parser():
    parser = tsm.build_parser()
    parser.add_argument("--objects", default=",".join(DEFAULT_OBJECTS))
    parser.add_argument("--states-per-object", type=int, default=10)
    parser.add_argument("--solvers", default="acados,ipopt,cem",
                        help="Comma-separated solver backends; acados=SQP, ipopt=IPOPT, cem=Ours.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output" / "kendall_scm_mujoco.json")
    return parser


def main():
    args = build_parser().parse_args()
    if args.states_per_object < 1:
        raise SystemExit("--states-per-object must be positive")
    objects = [x.strip() for x in args.objects.split(",") if x.strip()]
    solvers = [x.strip() for x in args.solvers.split(",") if x.strip()]
    rows = []
    for solver in solvers:
        rows.extend(run_solver(args, solver, objects, args.states_per_object))
    summary = {solver: aggregate([r for r in rows if r["solver"] == solver])
               for solver in solvers}
    payload = {"protocol": {"objects": objects, "states_per_object": args.states_per_object,
                             "solvers": solvers, "similarity": "(Kendall tau-b + 1)/2",
                             "reference": "MuJoCo point-force execution from the same frozen state",
                             "scm": "compare_prediction.py paper SCM: four pyramid rows, softplus diagonal impulse, one spectral residual step"},
               "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("object", "seed", "solver", "tau", "similarity", "pvalue", "n_candidates", "selected_idx"), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    for solver in solvers:
        print(solver, json.dumps(summary[solver], sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
