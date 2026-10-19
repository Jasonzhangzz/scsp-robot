"""Legacy joint MPPI experiment; the maintained MPC entry is ik2/test_mpc_isaac.py."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
if __name__ == '__main__':
    os.environ.setdefault('SCSP_PLANNER_ONLY', '1')

if os.environ.get('SCSP_PLANNER_ONLY') != '1':
    from isaacgym import gymapi, gymtorch

import numpy as np
from examples.mpc.franka.runtime.params import ExplicitMPCParams
from examples.mpc.fingertips.test.test_0902 import add_rollout_via_args
from utils import metrics
from examples.mpc.franka.runtime.simulator import (
    _parse_bool_arg,
    adapt_param_for_joint_mppi,
    _apply_plan_result,
    _plan_from_obs,
    _pred_reduction_from_policy,
    _dwell_payload,
)


def _configure_rollout_param(param, args):
    param.rollout_press_patch = True
    param.quadratic_contact_track = True
    param.attract_coef = max(float(param.attract_coef), 20.0)
    param.field_cost_weight = 0.0
    param.contact_coef = max(float(param.contact_coef), float(param.attract_coef))
    param.lambda_optimizer.lock_contact_patch = False
    param.lambda_optimizer.contact_switch_confirm_steps = max(
        1, int(args.contact_switch_confirm_steps)
    )
    return param


def _add_mppi_policy_args(parser):
    add_rollout_via_args(parser)
    parser.add_argument("--use-xml-texture", action="store_true")
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--viewer", dest="headless", action="store_false")
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--show-ghost-object", type=_parse_bool_arg, default=False)
    parser.add_argument("--cartesian-joint-stiffness", type=float, default=100.0)
    parser.add_argument("--osc-pos-stiffness", type=float, default=12000.0)
    parser.add_argument("--osc-ori-stiffness", type=float, default=0.0)
    parser.add_argument("--nullspace-stiffness", type=float, default=10.0)
    parser.add_argument("--svg-screenshot-dir", type=str, default="")
    parser.add_argument("--svg-screenshot-interval", type=float, default=0.2)
    parser.add_argument("--svg-screenshot-width", type=int, default=1280)
    parser.add_argument("--svg-screenshot-height", type=int, default=960)
    parser.add_argument("--cartesian_stiffness", type=float, nargs="+", default=None)
    parser.add_argument("--cartesian_damping", type=float, nargs="+", default=None)
    parser.add_argument("--effort-joint-damping", type=float, default=0.0)
    parser.add_argument(
        "--target-type",
        dest="target_type",
        type=str,
        default="ground-rotation",
        choices=("ground-rotation", "rotation"),
        help="ground-rotation matches fingertips --rollout (90 deg pitch flip).",
    )
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument(
        "--control-substeps",
        type=int,
        default=0,
        help="OSC frames per MPPI joint update.  0 uses the 20 ms --rollout interval.",
    )
    parser.add_argument(
        "--joint-step",
        type=float,
        default=0.02,
        help="MPPI joint-delta limit per 20 ms (rad).  Smaller is smoother.",
    )
    parser.add_argument(
        "--action-smooth",
        type=float,
        default=0.35,
        help="EMA blend of the executed joint action.  1 is raw MPPI.",
    )
    parser.add_argument(
        "--via-max-lead",
        type=float,
        default=0.008,
        help="Max via/press offset from the current fingertip (m).",
    )
    parser.add_argument(
        "--via-smooth-rate",
        type=float,
        default=0.05,
        help="SmoothedApproachVia lerp rate.  Smaller interpolates more.",
    )
    parser.add_argument("--joint-model-stiffness", type=float, default=300.0)
    parser.add_argument("--mppi_Nsample", type=int, default=256)
    parser.add_argument("--mppi_Hsample", type=int, default=16)
    parser.add_argument("--mppi_Ndiffuse", type=int, default=2)
    parser.add_argument("--mppi_Ndiffuse_init", type=int, default=3)
    parser.add_argument("--mppi_temp_sample", type=float, default=0.5)
    parser.add_argument("--mppi_sigma_scale", type=float, default=0.6)
    parser.add_argument("--mppi_traj_diffuse_factor", type=float, default=0.5)
    parser.add_argument("--mppi_seed", type=int, default=0)
    parser.add_argument(
        "--mppi_w_energy",
        type=float,
        default=50.0,
        help="0902 plan_once control weight on ||u||^2.",
    )
    parser.add_argument("--mppi_w_nopen", type=float, default=80.0)
    parser.add_argument("--mppi_w_joint_limit", type=float, default=100.0)
    parser.add_argument("--mppi_nopen_margin", type=float, default=0.002)
    parser.add_argument(
        "--async-planner",
        dest="async_planner",
        action="store_true",
        help="ROS-style latest-only state/action topics in a second process.",
    )
    parser.add_argument(
        "--sync-planner",
        dest="async_planner",
        action="store_false",
        help="Rank+MPPI in the Isaac process (debug).",
    )
    parser.add_argument("--viewer-hz", type=float, default=60.0)
    parser.add_argument(
        "--gpu-physx",
        action="store_true",
        help="GPU PhysX in the viewer (fights Warp MPPI on the same device).",
    )
    parser.set_defaults(headless=False, async_planner=True)
    return parser


def run_mppi_planner(bus, args):
    from examples.mpc.franka.runtime.bus import wait_trial_obs
    from examples.mpc.franka.runtime.mpc import (
        DEFAULT_CARTESIAN_STIFFNESS,
        adapt_param_for_cartesian_solver,
        _apply_dywa_physics_to_param as _apply_mpc_physics_to_param,
        _configure_rollout_param as _configure_mpc_rollout_param,
        _print_rollout_step,
    )
    from planning.MPPIWarp import (
        _build_planner_runtime,
        blend_action,
        handle_planner_request,
        planner_init_payload,
    )

    if args.cartesian_stiffness is None:
        args.cartesian_stiffness = DEFAULT_CARTESIAN_STIFFNESS.tolist()
    if args.trial_start < 0:
        raise ValueError(f"trial_start must be non-negative, got {args.trial_start}")
    if int(args.trial_num) <= 0:
        raise ValueError(f"trial_num must be positive, got {args.trial_num}")

    trial_start = int(args.trial_start)
    trial_num = max(1, int(args.trial_num))
    trial_stop = trial_start + trial_num
    success_pos_threshold = 0.02
    success_quat_threshold = 0.015
    consecutive_success_time_threshold = 0
    max_rollout_length = max(1, int(args.max_rollout_length))
    success_rate = 0
    viewer_quit = False

    for trial_count in range(trial_start, trial_stop):
        param = ExplicitMPCParams(
            args,
            rand_seed=trial_count,
            target_type=getattr(args, "target_type", "ground-rotation"),
            mpc_model="explicit",
        )
        param = _apply_mpc_physics_to_param(param)
        param.use_jax_contact_ = False
        param = adapt_param_for_cartesian_solver(param, args)
        param = _configure_mpc_rollout_param(param, args)
        param = adapt_param_for_joint_mppi(param, args)
        param.control_substeps_ = int(args.control_substeps)
        init = planner_init_payload(args, param, trial_count, args.sim_device)
        _, plan_param, mpc, trackers = _build_planner_runtime(init)

        obs = wait_trial_obs(bus, trial_count)
        if obs is None:
            viewer_quit = True
            break

        prev_action = None
        policy = None
        c_now_cost = None
        pred_reduction = None
        last_accept_p_arm = False
        escape_on = False
        verify_chatter = False
        rollout_step = 0
        consecutive_success_time = 0
        min_pos_err = float("inf")
        min_quat_err = float("inf")
        pose_apply_count = 0
        choose_times = []
        pos_err_now = None
        quat_err_now = None
        action = np.zeros(7, dtype=np.float32)

        while rollout_step < max_rollout_length:
            if obs.get("break_out") or obs.get("viewer_closed") or obs.get("cmd") == "stop":
                viewer_quit = True
                break
            curr_q = np.asarray(obs["policy_q"], dtype=np.float32)
            pos_err_now = float(metrics.comp_pos_error(curr_q[0:3], param.target_p_))
            quat_err_now = float(metrics.comp_quat_error(curr_q[3:7], param.target_q_))
            min_pos_err = min(min_pos_err, pos_err_now)
            min_quat_err = min(min_quat_err, quat_err_now)
            if pos_err_now < success_pos_threshold and quat_err_now < success_quat_threshold:
                consecutive_success_time += 1
            else:
                consecutive_success_time = 0
            if consecutive_success_time > consecutive_success_time_threshold:
                break

            contact_distance = float(obs.get("contact_distance", float("inf")))
            dwell = None
            if policy is not None:
                dwell = _dwell_payload(
                    None, args, None, policy, last_accept_p_arm, escape_on,
                    verify_chatter, c_now_cost, pred_reduction, contact_distance,
                    pos_err_now, curr_q=curr_q, param=param,
                )
            payload, curr_q = _plan_from_obs(obs, dwell)
            result = handle_planner_request(args, plan_param, mpc, trackers, payload)
            policy = _apply_plan_result(param, None, result, curr_q, _print_rollout_step)
            choose_times.append(float(result["rank_dt"]))
            c_now_cost, pred_reduction = _pred_reduction_from_policy(param, curr_q, policy)
            last_accept_p_arm = bool(policy["value_info"].get("accept_p_arm", False))
            escape_on = bool(policy["escape_on"])
            verify_chatter = bool(result["verify_chatter"])
            raw = np.asarray(result["action"], dtype=np.float32).reshape(7)
            action = blend_action(prev_action, raw, float(getattr(args, "action_smooth", 0.35)))
            prev_action = action.copy()

            on_exec_contact = False
            if np.isfinite(contact_distance) and contact_distance <= 0.003:
                post_tip = np.asarray(curr_q[7:10], dtype=float)
                on_exec_contact = (
                    float(np.linalg.norm(post_tip - np.asarray(policy["p_arm_world"], dtype=float))) <= 0.03
                )
                if on_exec_contact:
                    pose_apply_count += 1
            print(
                "contact_distance:",
                None if not np.isfinite(contact_distance) else round(contact_distance, 6),
                "physics_contact:", int(on_exec_contact),
                "mppi_action_norm:", round(float(np.linalg.norm(action)), 6),
            )
            bus.publish_cmd({
                "kind": "joint",
                "action": action,
                "markers": {
                    "target": policy["mpc_virtual_point"],
                    "best_contact": policy["best_contact_world"],
                },
            })
            rollout_step += 1
            obs = bus.wait_obs(timeout=180.0)
            if obs is None:
                viewer_quit = True
                break

        bus.publish_cmd({"cmd": "end_trial"})
        choose_arr = np.asarray(choose_times, dtype=np.float64) if choose_times else np.array([0.0])
        print("trial_summary:", {
            "trial": trial_count,
            "mode": "mppi_planner",
            "success": int(rollout_step < max_rollout_length),
            "steps": rollout_step,
            "pose_applies": pose_apply_count,
            "final_pos_err": None if pos_err_now is None else round(float(pos_err_now), 5),
            "final_quat_err": None if quat_err_now is None else round(float(quat_err_now), 5),
            "min_pos_err": None if not np.isfinite(min_pos_err) else round(float(min_pos_err), 5),
            "min_quat_err": None if not np.isfinite(min_quat_err) else round(float(min_quat_err), 5),
            "choose_dt_mean": round(float(np.mean(choose_arr)), 5),
            "choose_dt_max": round(float(np.max(choose_arr)), 5),
            "choose_dt_p95": round(float(np.percentile(choose_arr, 95)), 5),
        })
        success_rate += 1 if rollout_step < max_rollout_length else 0
        if viewer_quit:
            break

    print(
        f"Success rate over {trial_num} trials "
        f"(trial ids {trial_start} to {trial_stop - 1}): "
        f"{success_rate}/{trial_num} = {success_rate / trial_num:.2%}"
    )
    return viewer_quit


def planner_worker(cmd_q, obs_q, ready_q, init):
    from examples.mpc.franka.runtime.bus import IsaacBus, args_from_init

    if ready_q is not None:
        ready_q.put({"ok": True, "pid": os.getpid()})
    run_mppi_planner(IsaacBus(cmd_q, obs_q), args_from_init(init))


def main():
    script = str(ROOT / "examples/mpc/franka/runtime/isaac.py")
    os.environ.pop("SCSP_PLANNER_ONLY", None)
    os.execv(sys.executable, [sys.executable, script, "--planner", "mppi", *sys.argv[1:]])


if __name__ == "__main__":
    main()
