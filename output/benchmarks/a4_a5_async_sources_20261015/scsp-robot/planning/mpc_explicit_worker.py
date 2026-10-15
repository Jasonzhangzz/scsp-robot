"""MPC planner process helpers kept out of the Isaac MPCExplicit module.

Isaac's ``test_mpc_isaac.py`` imports ``MPCExplicit`` from
``planning.mpc_explicit``.  Ranking payloads, support-plane kwargs, and the
spawned planner loop live here so that import does not pull them in.
"""

import numpy as np

from planning.mpc_explicit import MPCExplicit

def planner_init_payload(args, param, trial_count, device=None):
    return {
        "args": {k: v for k, v in vars(args).items() if not callable(v)},
        "trial_count": int(trial_count),
        "device": device,
        "target_p_": np.asarray(param.target_p_, dtype=np.float64),
        "target_q_": np.asarray(param.target_q_, dtype=np.float64),
        "mesh_path_": param.mesh_path_,
        "table_height": float(param.table_height),
        "lambda_obj_mass_": float(getattr(param, "lambda_obj_mass_", param.obj_mass_)),
        "gravity_": np.asarray(param.gravity_, dtype=np.float64),
        "attract_coef": float(param.attract_coef),
        "contact_coef": float(param.contact_coef),
    }


def _build_mpc_planner_runtime(init):
    import argparse
    from planning.acados_env import ensure_acados_env
    ensure_acados_env()
    from examples.mpc.franka.ik2.params import ExplicitMPCParams
    from examples.mpc.fingertips.test.test_0902 import (
        ContactValueTracker,
        ModelCostConfidence,
        SmoothedApproachVia,
    )
    from planning.MPPIWarp import (
        adapt_param_for_cartesian_ranking,
        configure_mppi_rollout_param,
    )

    args = argparse.Namespace(**init["args"])
    args.solver = "acados"
    args.rollout = True
    trial_count = int(init["trial_count"])
    param = ExplicitMPCParams(
        args,
        rand_seed=trial_count,
        target_type=getattr(args, "target_type", "ground-rotation"),
        mpc_model="explicit",
    )
    param = adapt_param_for_cartesian_ranking(param, args)
    param = configure_mppi_rollout_param(param, args)
    for key in (
        "target_p_", "target_q_", "mesh_path_", "table_height",
        "lambda_obj_mass_", "gravity_", "attract_coef", "contact_coef",
    ):
        if key in init:
            setattr(param, key, init[key])
    mpc = MPCExplicit(param)
    mpc_step = max(1e-4, float(getattr(args, "mpc_step_limit", 0.005)))
    via_step = getattr(args, "via_max_step", None)
    via_step = mpc_step if via_step is None else max(1e-4, float(via_step))
    via_lead = max(1e-4, float(getattr(args, "via_max_lead", via_step)))
    trackers = {
        "value_tracker": ContactValueTracker(
            tau=float(args.value_tau),
            rel_scale=float(args.value_rel_scale),
            rho=float(args.value_rho),
            alpha=float(args.value_alpha),
            beta=float(args.verify_beta),
            window_size=int(getattr(args, "verify_window_size", 20)),
            confirm_steps=int(getattr(args, "verify_enter_steps", 5)),
            min_hold_steps=int(getattr(args, "verify_hold_steps", 30)),
            release_steps=int(getattr(args, "verify_release_steps", 8)),
            accept_margin_ratio=0.05,
            accept_margin_abs=0.02,
        ),
        "model_cost_conf": ModelCostConfidence(
            threshold=float(getattr(args, "model_cost_error_threshold", 6.0)),
            eps=float(getattr(args, "model_cost_error_eps", 1e-6)),
            min_steps=int(getattr(args, "model_cost_error_min_steps", 3)),
        ),
        "approach_via": SmoothedApproachVia(
            rate=float(getattr(args, "via_smooth_rate", 0.05)),
            max_step=via_step,
            max_lead=via_lead,
        ),
        "arrived_hold": False,
        "arrived_dest_idx": None,
        "sol_guess": None,
        "last_verify_cost": None,
    }
    return args, param, mpc, trackers


def _supported_call_kwargs(fn, kwargs):
    """Drop kwargs the live callee does not accept.

    Isaac / tilted-push payloads always carry support_point.  An older
    ``compute_rollout_contact_via`` on disk raises TypeError on that name
    before ranking starts.  New signatures still receive the plane.
    """
    import inspect
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in params}


def _call_rollout_contact_via(fn, *args, **kwargs):
    return fn(*args, **_supported_call_kwargs(fn, kwargs))


def handle_mpc_request(args, param, mpc, trackers, msg):
    import time
    from examples.mpc.fingertips.test.test_0902 import (
        compute_rollout_contact_via,
        _verify_is_chatter,
    )
    from examples.mpc.franka.ik2.contact_frames import gravity_wrench_object_frame
    from planning.MPPIWarp import (
        _apply_dwell_payload,
        _opt_snapshot,
        _pickle_safe,
        clamp_via_to_tip,
    )
    from scipy.spatial.transform import Rotation

    dwell = msg.get("dwell")
    if dwell is not None:
        dwell = dict(dwell)
        dwell["model_cost_conf"] = trackers["model_cost_conf"]
        _apply_dwell_payload(param, args, dwell)

    curr_q = np.asarray(msg["policy_q"], dtype=np.float32)
    max_ncon = int(getattr(param, "max_ncon_", 10))
    nv = int(getattr(param, "n_qvel_", 9))
    phi_vec = (
        np.asarray(msg["phi_vec"])
        if msg.get("phi_vec") is not None
        else np.ones((max_ncon * 4,), dtype=np.float64)
    )
    jac_mat = (
        np.asarray(msg["jac_mat"])
        if msg.get("jac_mat") is not None
        else np.zeros((max_ncon * 4, nv), dtype=np.float64)
    )
    if msg.get("jac_mat_env") is not None:
        jac_mat_env = np.asarray(msg["jac_mat_env"])
    else:
        from examples.mpc.franka.ik2.contact_frames import table_jac_mat_env
        jac_mat_env = table_jac_mat_env(
            curr_q[:3], curr_q[3:7], float(param.table_height),
            nv=nv, mu=float(getattr(param, "mu_object_", 0.5)), max_ncon=max_ncon,
        )
    table_ground = float(msg["table_ground"])
    floor_z = float(msg["floor_z"]) if msg.get("floor_z") is not None else float(param.table_height)
    support_point = msg.get("support_point")
    support_normal = msg.get("support_normal")
    if support_point is not None:
        support_point = np.asarray(support_point, dtype=np.float64).reshape(3)
    if support_normal is not None:
        support_normal = np.asarray(support_normal, dtype=np.float64).reshape(3)
    r_obj_to_world = Rotation.from_quat([curr_q[4], curr_q[5], curr_q[6], curr_q[3]]).as_matrix()
    ranking_mass = float(getattr(param, "lambda_obj_mass_", param.lambda_optimizer.m))
    gravity = gravity_wrench_object_frame(
        param.gravity_[:3], ranking_mass, r_obj_to_world, support_normal,
    )
    t0 = time.perf_counter()
    policy = _call_rollout_contact_via(
        compute_rollout_contact_via,
        param, args, curr_q, r_obj_to_world, gravity, jac_mat_env,
        0.01, trackers["value_tracker"], trackers["model_cost_conf"],
        trackers["approach_via"], trackers["arrived_hold"], trackers["arrived_dest_idx"],
        floor_ground=table_ground,
        floor_z=floor_z,
        support_point=support_point,
        support_normal=support_normal,
    )
    rank_dt = time.perf_counter() - t0
    trackers["arrived_hold"] = policy["arrived_hold"]
    trackers["arrived_dest_idx"] = policy["arrived_dest_idx"]
    verify_cost = policy["verify_cost"]
    verify_chatter = _verify_is_chatter(trackers["last_verify_cost"], verify_cost)
    trackers["last_verify_cost"] = float(verify_cost)

    tip = np.asarray(curr_q[7:10], dtype=np.float64)
    max_lead = max(1e-4, float(getattr(args, "via_max_lead", getattr(args, "via_max_step", getattr(args, "mpc_step_limit", 0.005)))))
    value_info = policy.get("value_info") or {}
    if bool(value_info.get("path_blocked", False)):
        trackers["sol_guess"] = None
    # A blocked orbit already walks via at max_step.  Clamping it back
    # to a 3 mm tip lead whenever the tip crosses the keep-out is what
    # made the rim hop.
    if not bool(value_info.get("path_blocked", False)):
        policy["mpc_virtual_point"] = clamp_via_to_tip(tip, policy["mpc_virtual_point"], max_lead)
        policy["mpc_contact_point"] = clamp_via_to_tip(tip, policy["mpc_contact_point"], max_lead)

    t1 = time.perf_counter()
    sol = mpc.plan_once(
        param.target_p_,
        param.target_q_,
        curr_q,
        phi_vec,
        jac_mat,
        verify_cost_param=verify_cost,
        virtual_point=policy["mpc_virtual_point"],
        contact_point=policy["mpc_contact_point"],
        sol_guess=trackers["sol_guess"],
    )
    plan_dt = time.perf_counter() - t1
    trackers["sol_guess"] = sol["sol_guess"]
    policy["choose_dt"] = rank_dt
    return {
        "action": np.asarray(sol["action"], dtype=np.float64).reshape(3),
        "sol_guess": sol["sol_guess"],
        "cost_opt": sol.get("cost_opt"),
        "policy": _pickle_safe(policy),
        "verify_cost": float(verify_cost),
        "verify_chatter": bool(verify_chatter),
        "if_contact": bool(msg.get("if_contact", False)),
        "opt_snapshot": _opt_snapshot(param.lambda_optimizer),
        "model_tightness": float(trackers["model_cost_conf"].tightness()),
        "model_accum": float(trackers["model_cost_conf"].accum),
        "rank_dt": float(rank_dt),
        "plan_dt": float(plan_dt),
        "mppi_dt": float(plan_dt),
        "lambda_failures": int(getattr(param.lambda_optimizer, "acados_failure_count", 0)),
        "mpc_failures": int(getattr(mpc, "acados_failure_count", 0)),
        "lambda_solves": int(getattr(param.lambda_optimizer, "acados_solve_count", 0)),
        "mpc_solves": int(getattr(mpc, "acados_solve_count", 0)),
        "mpc_init_error": str(getattr(mpc, "_acados_init_error", "") or ""),
    }


def mpc_planner_worker(state_q, action_q, ready_q, init):
    """Independent planner loop.  Must not import isaacgym."""
    import traceback
    from planning.MPPIWarp import drain_latest, put_latest

    try:
        args, param, mpc, trackers = _build_mpc_planner_runtime(init)
        ready_q.put({"ok": True})
    except Exception:
        ready_q.put({"ok": False, "error": traceback.format_exc()})
        return
    while True:
        try:
            msg = state_q.get()
        except (EOFError, KeyboardInterrupt):
            break
        if msg is None or (isinstance(msg, dict) and msg.get("cmd") == "stop"):
            break
        msg = drain_latest(state_q, msg)
        if msg is None:
            break
        try:
            put_latest(action_q, handle_mpc_request(args, param, mpc, trackers, msg))
        except Exception:
            put_latest(action_q, {"ok": False, "error": traceback.format_exc()})


class AsyncMpcPlanner:
    """ROS-style latest-only state/action bus.  Isaac never waits for a plan."""

    def __init__(self, state_q, action_q, proc):
        self.state_q = state_q
        self.action_q = action_q
        self.proc = proc

    @classmethod
    def start(cls, init, timeout=180.0):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        state_q = ctx.Queue(maxsize=1)
        action_q = ctx.Queue(maxsize=1)
        ready_q = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=mpc_planner_worker,
            args=(state_q, action_q, ready_q, init),
            daemon=True,
        )
        proc.start()
        ready = ready_q.get(timeout=timeout)
        if not ready.get("ok", False):
            proc.join(timeout=2.0)
            raise RuntimeError(ready.get("error", "MPC planner process failed to start"))
        return cls(state_q, action_q, proc)

    def publish_state(self, payload):
        from planning.MPPIWarp import put_latest

        return put_latest(self.state_q, payload)

    def take_action(self):
        from planning.MPPIWarp import take_latest

        out = take_latest(self.action_q)
        if isinstance(out, dict) and out.get("ok") is False:
            raise RuntimeError(out.get("error", "MPC planner process failed"))
        return out

    def close(self):
        from planning.MPPIWarp import put_latest

        put_latest(self.state_q, None)
        if self.proc is not None:
            self.proc.join(timeout=5.0)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=2.0)
