"""Baseline Franka executor and options, with the ablation planner loop.

Scene creation, OSC, contact geometry and argument defaults are imported from
runtime.mpc. Only the trial loop is local so spawned planners use our hooks.
"""
import os
import sys
import time
import json

import numpy as np

from examples.mpc.franka.runtime import mpc as baseline
from examples.mpc.franka.runtime.mpc import (
    torch, POLICY_INTERVAL, DYWA_SIM_DT, ISAAC_TABLE_CLEARANCE,
    _apply_dywa_physics_to_param, _configure_rollout_param,
    _dwell_payload, _plan_from_obs, _apply_plan_result, _print_rollout_step,
    _pred_reduction_from_policy, _clip_mpc_action, _print_contact_pose_diag,
    _contact_pose_diag, _print_rank_cost_diag, _analyze_edge_pose_source,
    _print_edge_pose_source, resolve_nullspace_stiffness,
)
from planning.runtime_compat import (
    resolve_solver_backend, print_runtime_profile, make_runtime_profile,
)
from utils import metrics
from utils.ranking_metrics import discounted_terminal_reward
from .params import ExplicitMPCParams, build_lambda_optimizer
from .config import MODES, add_ablation_args, configure_ablation


def __getattr__(name):
    return getattr(baseline, name)


def _add_rollout_policy_args(parser):
    baseline._add_rollout_policy_args(parser)
    add_ablation_args(parser)
    for action in parser._actions:
        if '--sync-planner' in action.option_strings:
            action.help = 'Use the baseline synchronous planner protocol (default).'
        elif '--async-planner' in action.option_strings:
            action.help = 'Use the baseline asynchronous planner protocol.'
    return parser


def _apply_execution_mode(args):
    configure_ablation(args)
    return baseline._apply_execution_mode(args)


def adapt_param_for_cartesian_solver(param, args, build_optimizer=True):
    param = baseline.adapt_param_for_cartesian_solver(param, args, build_optimizer=False)
    if build_optimizer and param.lambda_optimizer is None:
        param.lambda_optimizer = build_lambda_optimizer(param, args)
    if param.lambda_optimizer is not None and param.ablation_mode == 'a5':
        param.lambda_optimizer.solver = 'cma-es'
    return param


def run_mpc_planner(bus, args):
    if getattr(args, 'ablation_mode', 'full') == 'a6':
        from .adamw import run_adamw_planner
        return run_adamw_planner(bus, args)
    from examples.mpc.franka.experiments.mpc_ablation.bus import wait_trial_obs
    from examples.mpc.franka.experiments.mpc_ablation.planner import (
        _build_mpc_planner_runtime,
        handle_mpc_request,
        planner_init_payload,
    )

    if args.trial_start < 0:
        raise ValueError(f"trial_start must be non-negative, got {args.trial_start}")
    if int(args.trial_num) <= 0:
        raise ValueError(f"trial_num must be positive, got {args.trial_num}")
    metric_gamma = float(getattr(args, 'ranking_metric_gamma', 0.9))
    metric_eta = float(getattr(args, 'ranking_metric_eta', 1.0))
    discounted_terminal_reward(False, gamma=metric_gamma, eta=metric_eta)
    # Graphics must not implicitly enable thousands of diagnostic prints:
    # their cost changes the async feedback delay relative to headless.
    verbose = bool(getattr(args, 'rollout_verbose', False))
    metric_path = getattr(args, 'ranking_metrics_jsonl', '')
    if metric_path:
        os.makedirs(os.path.dirname(os.path.abspath(metric_path)), exist_ok=True)
    trace_path = getattr(args, 'rollout_trace_jsonl', '')
    if trace_path:
        os.makedirs(os.path.dirname(os.path.abspath(trace_path)), exist_ok=True)
    args.solver = resolve_solver_backend(getattr(args, "solver_backend", "auto"))
    print_runtime_profile(
        make_runtime_profile(
            getattr(args, "solver_backend", "auto"),
            policy_dt=POLICY_INTERVAL,
            sim_dt=DYWA_SIM_DT,
            table_clearance=float(getattr(args, "table_clearance", ISAAC_TABLE_CLEARANCE)),
        ),
        prefix="[mpc-runtime]",
    )

    trial_start = int(args.trial_start)
    trial_num = max(1, int(args.trial_num))
    trial_stop = trial_start + trial_num
    success_pos_threshold = 0.02
    success_quat_threshold = 0.015
    consecutive_success_time_threshold = 0
    max_rollout_length = max(1, int(args.max_rollout_length))
    mpc_step = max(1e-4, float(getattr(args, "mpc_step_limit", 0.005)))
    # OSC clips the executed increment near the patch.  Clipping the
    # published action to via_max_step made every approach 2.5 mm.
    exec_step = mpc_step
    success_rate = 0
    viewer_quit = False
    rollout_started = time.perf_counter()
    trial_summaries = []

    for trial_count in range(trial_start, trial_stop):
        if getattr(args, 'rollout_cache_dir', ''):
            import random
            random.seed(100 + trial_count)
            np.random.seed(100 + trial_count)
            if torch is not None:
                torch.manual_seed(100 + trial_count)
        startup_started = time.perf_counter()
        print('[mpc-startup] trial=%d building contact/MPC runtime' % trial_count,
              flush=True, file=sys.__stdout__)
        param = ExplicitMPCParams(
            args,
            rand_seed=trial_count,
            target_type=getattr(args, "target_type", "ground-rotation"),
            mpc_model="explicit",
        )
        param = _apply_dywa_physics_to_param(param)
        param.use_jax_contact_ = False
        param = adapt_param_for_cartesian_solver(param, args)
        param = _configure_rollout_param(param, args)
        param.control_substeps_ = int(args.control_substeps)
        sampling_controller = None
        if bool(getattr(args, "sampling_c3", False)):
            from planning.sampling_c3 import SamplingC3Controller, config_from_args
            sampling_controller = SamplingC3Controller(config_from_args(args, trial_count))
        init = planner_init_payload(args, param, trial_count)
        _, plan_param, mpc, trackers = _build_mpc_planner_runtime(init, param)
        # Optional one-decision counterfactual evaluator; ordinary runs have no hook.
        plan_param.rs_decision_hook = getattr(bus, 'rs_decision_hook', None)
        print('[mpc-startup] trial=%d ready in %.2fs (contact=%s mpc=%s)' % (
            trial_count, time.perf_counter()-startup_started,
            getattr(plan_param.lambda_optimizer, 'solver', 'unknown'),
            getattr(mpc, 'planner_solver_', 'unknown')), flush=True, file=sys.__stdout__)

        obs = wait_trial_obs(bus, trial_count)
        if obs is None:
            viewer_quit = True
            break
        trial_record = metrics.open_trial_recorder(
            args, trial_count, param.target_p_, param.target_q_)
        if trace_path:
            opt = plan_param.lambda_optimizer
            obb = getattr(opt, 'obb_geometry', None)
            with open(trace_path, 'a') as trace:
                trace.write(json.dumps(dict(event='setup', trial=trial_count,
                    samples=np.asarray(opt.sample_point).tolist(),
                    curvature=np.asarray(opt.point_curvature).tolist(),
                    high_curvature=np.asarray(opt._high_curvature_mask()).tolist(),
                    target_p=np.asarray(param.target_p_).tolist(),
                    target_q=np.asarray(param.target_q_).tolist(),
                    **({} if obb is None else dict(lambda_geometry=obb.diagnostics(),
                        target_obb_world=obb.world_transform(np.r_[param.target_p_, param.target_q_]).tolist()))),
                    allow_nan=False)+'\n')

        policy = None
        c_now_cost = None
        pred_reduction = None
        last_accept_p_arm = False
        escape_on = False
        verify_chatter = False
        last_result = None
        previous_trace_pose = None
        pose_snapshot_saved = False
        rollout_step = 0
        trial_success = False
        consecutive_success_time = 0
        min_pos_err = float("inf")
        min_quat_err = float("inf")
        pose_apply_count = 0
        choose_times = []
        plan_times = []
        planning_wall_times = []
        pos_err_now = None
        quat_err_now = None
        prev_pos_err = None
        prev_quat_err = None
        edge_verdicts = []
        iteration_wall_times = []
        max_floor_recovery = 0.0
        observation_gaps = []
        trial_wall_start = time.perf_counter()

        # Check the observation after the final allowed action as well.
        while rollout_step <= max_rollout_length:
            iteration_wall_start = time.perf_counter()
            if obs.get("break_out") or obs.get("viewer_closed") or obs.get("cmd") == "stop":
                viewer_quit = True
                break
            curr_q = np.asarray(obs["policy_q"], dtype=np.float32)
            if trace_path:
                with open(trace_path, 'a') as trace:
                    trace.write(json.dumps(dict(event='pose', trial=trial_count, step=int(rollout_step),
                        policy_q=np.asarray(curr_q, dtype=float).reshape(-1).tolist()), allow_nan=False)+'\n')
            pos_err_now = float(metrics.comp_pos_error(curr_q[0:3], param.target_p_))
            quat_err_now = float(metrics.comp_quat_error(curr_q[3:7], param.target_q_))
            min_pos_err = min(min_pos_err, pos_err_now)
            min_quat_err = min(min_quat_err, quat_err_now)
            if trial_record is not None:
                trial_record.observe(
                    rollout_step, curr_q[:7], metrics.robot_q_from_observation(obs),
                    sim_time=obs.get("sim_time"))
            stages = trackers.get('object_cost_stages')
            # Process the terminal observation too, before declaring success.
            # handle_mpc_request sees the same timestamp and cannot count it twice.
            from planning.object_pose_cost import reset_objective_history
            if stages is not None and stages.update(obs):
                reset_objective_history(plan_param.lambda_optimizer, mpc, trackers)
                policy = None
                c_now_cost = pred_reduction = None
                print('[object-cost] transition:', stages.diagnostics(), flush=True)
            if verbose and stages is not None and rollout_step % 25 == 0:
                print('[object-cost] state:', stages.diagnostics(), flush=True)
            if stages is not None and stages.enabled:
                plan_param.object_cost_config_ = stages.config()
            if (pos_err_now <= success_pos_threshold and quat_err_now <= success_quat_threshold
                    and (stages is None or not stages.enabled or stages.stage == 'pose')):
                consecutive_success_time += 1
            else:
                consecutive_success_time = 0
            if consecutive_success_time > consecutive_success_time_threshold:
                trial_success = True
                break
            if rollout_step >= max_rollout_length:
                break

            contact_distance = float(obs.get("contact_distance", float("inf")))
            dwell = None
            if policy is not None and not bool(obs.get('osc_upright_guard_active', False)):
                dwell = _dwell_payload(
                    None, args, None, policy, last_accept_p_arm, escape_on,
                    verify_chatter, c_now_cost, pred_reduction, contact_distance,
                    pos_err_now, curr_q=curr_q, param=param,
                )
            payload, curr_q = _plan_from_obs(obs, dwell)
            planning_started = time.perf_counter()
            if sampling_controller is None:
                result = handle_mpc_request(args, plan_param, mpc, trackers, payload)
            else:
                from planning.sampling_c3 import sampling_c3_step
                result = sampling_c3_step(
                    sampling_controller, args, plan_param, mpc, trackers, payload,
                    exec_step, pos_err_now, quat_err_now,
                )
            planning_wall_times.append(time.perf_counter() - planning_started)
            if trial_record is not None:
                trial_record.note_planning(planning_wall_times[-1])
            if (getattr(args, 'rollout_planner_snapshots', '') and not pose_snapshot_saved
                    and stages is not None and stages.enabled and stages.stage == 'pose'
                    and payload.get('actual_contact') is not None
                    and (result.get('policy', {}).get('value_info') or {}).get('using_observed_contact')):
                snapshot_dir = os.path.abspath(args.rollout_planner_snapshots)
                os.makedirs(snapshot_dir, exist_ok=True)
                # Static replay has no preceding command/dwell history.
                snapshot_request = dict(payload, dwell=None)
                snapshot = dict(planner_init_payload(args, plan_param, trial_count),
                                msg=snapshot_request, config=stages.config())
                path = os.path.join(snapshot_dir, '%s_trial%d_pose.json' % (args.obj, trial_count))
                with open(path, 'w') as output:
                    json.dump(snapshot, output,
                              default=lambda value: np.asarray(value).tolist(), allow_nan=False)
                pose_snapshot_saved = True
            last_result = result
            max_floor_recovery = max(max_floor_recovery, float(result.get('mpc_floor_recovery_m', 0.)))
            policy = _apply_plan_result(param, None, result, curr_q, _print_rollout_step, verbose=verbose)
            choose_times.append(float(result["rank_dt"]))
            plan_times.append(float(result.get("plan_dt", 0.0)))
            c_now_cost, pred_reduction = _pred_reduction_from_policy(param, curr_q, policy)
            last_accept_p_arm = bool(policy["value_info"].get("accept_p_arm", False))
            escape_on = bool(policy["escape_on"])
            verify_chatter = bool(result["verify_chatter"])
            action = np.asarray(result["action"], dtype=np.float64).reshape(3)
            tip_now = np.asarray(curr_q[7:10], dtype=np.float64)
            action = np.asarray(_clip_mpc_action(action, exec_step), dtype=np.float64)
            exec_via = np.asarray(policy["mpc_virtual_point"], dtype=np.float64).reshape(3)
            if trace_path:
                opt = plan_param.lambda_optimizer
                selected = getattr(opt, 'last_selected_idx', None)
                executed = getattr(opt, 'last_executed_idx', None)
                with open(trace_path, 'a') as trace:
                    trace.write(json.dumps(dict(event='cycle', trial=trial_count, step=rollout_step,
                        observation_seq=obs.get('seq'), sim_time=obs.get('sim_time'),
                        policy_q=curr_q.tolist(), action=action.tolist(),
                        planning_ms=planning_wall_times[-1]*1000.,
                        selected_idx=None if selected is None else int(selected),
                        executed_idx=None if executed is None else int(executed),
                        selected_curvature=None if selected is None else float(opt.point_curvature[selected]),
                        physical_contact=bool(np.isfinite(contact_distance) and contact_distance <= 0.),
                        contact_distance=float(contact_distance) if np.isfinite(contact_distance) else None,
                        via=exec_via.tolist(), p_arm=np.asarray(policy['p_arm_world']).tolist(),
                        approach_phase=(policy.get('value_info') or {}).get('via_phase'),
                        force_approach_active=bool((policy.get('value_info') or {}).get('force_approach_active', False)),
                        mpc_contact_seed_costs=policy.get('mpc_contact_seed_costs'),
                        mpc_contact_seed_selected=policy.get('mpc_contact_seed_selected'),
                        tool_rotation_world=np.asarray(policy.get('tool_rotation_world', np.zeros(3))).tolist(),
                        tool_rotation_action=(None if policy.get('tool_rotation_action') is None else
                                              np.asarray(policy['tool_rotation_action']).tolist()),
                        best_contact=np.asarray(policy['best_contact_world']).tolist(),
                        contact_predictions=policy.get('contact_prediction_diagnostics'),
                        reported_executed_action=(None if obs.get('executed_action') is None else
                            np.asarray(obs['executed_action']).tolist()),
                        executor_floor_constraints=result.get('mpc_executor_floor'),
                        executor_edge_constraint=result.get('mpc_executor_edge', False),
                        observed_position_reduction_mm=(None if previous_trace_pose is None else
                            1000*(np.linalg.norm(previous_trace_pose[:3]-param.target_p_)
                                  - np.linalg.norm(curr_q[:3]-param.target_p_))),
                        **({} if stages is None else stages.diagnostics()),
                        **({} if getattr(opt, 'obb_geometry', None) is None else
                           dict(current_obb_world=opt.obb_geometry.world_transform(curr_q[:7]).tolist()))),
                        allow_nan=False)+'\n')
                previous_trace_pose = curr_q.copy()

            on_exec_contact = False
            physical_contact = bool(np.isfinite(contact_distance) and contact_distance <= 0.003)
            if physical_contact:
                on_exec_contact = float(np.linalg.norm(tip_now - np.asarray(policy["p_arm_world"], dtype=float))) <= 0.03
                if on_exec_contact:
                    pose_apply_count += 1
                if verbose:
                    _print_contact_pose_diag(
                        _contact_pose_diag(
                            param.lambda_optimizer, policy, tip_now,
                            osc_target=tip_now + action,
                        )
                    )
            if verbose:
                print(
                    "contact_distance:",
                    None if not np.isfinite(contact_distance) else round(contact_distance, 6),
                    "physics_contact:", int(on_exec_contact),
                )
                _print_rank_cost_diag(param, curr_q, policy["best_contact_world"])
                osc_force = obs.get("osc_force")
                osc_pd = obs.get("osc_pd")
                edge_info = _analyze_edge_pose_source(
                    param, curr_q, policy, result, osc_force, physical_contact,
                    pos_err_now, quat_err_now, prev_pos_err, prev_quat_err,
                )
                if edge_info["near_edge"] or edge_info["if_contact"] or float(pos_err_now) > 0.05:
                    _print_edge_pose_source(edge_info)
                    edge_verdicts.append((rollout_step, edge_info["verdict"], edge_info))
                prev_pos_err = float(pos_err_now)
                prev_quat_err = float(quat_err_now)
                print(
                    "diag_push:",
                    "step:", rollout_step,
                    "obj:", np.round(np.asarray(curr_q[:3], dtype=float), 4).tolist(),
                    "obj_z:", round(float(curr_q[2]), 4),
                    "table:", round(float(param.table_height), 4),
                    "below_table:", int(float(curr_q[2]) < float(param.table_height) - 0.01),
                    "tip:", np.round(tip_now, 4).tolist(),
                    "via:", np.round(exec_via, 4).tolist(),
                    "virtual:", np.round(np.asarray(policy["mpc_virtual_point"], dtype=float), 4).tolist(),
                    "p_arm:", np.round(np.asarray(policy["p_arm_world"], dtype=float), 4).tolist(),
                    "best:", np.round(np.asarray(policy["best_contact_world"], dtype=float), 4).tolist(),
                    "action:", np.round(action, 4).tolist(),
                    "osc_pd:", None if osc_pd is None else np.round(np.asarray(osc_pd, dtype=float), 4).tolist(),
                    "osc_f:", None if osc_force is None else np.round(np.asarray(osc_force, dtype=float), 3).tolist(),
                    "osc_fn:", None if osc_force is None else round(float(np.linalg.norm(osc_force)), 3),
                    "near_press:", int(bool(obs.get("osc_near_press", False))),
                    "verify:", None if result.get("verify_cost") is None else round(float(result["verify_cost"]), 4),
                    "conf:", round(float(param.lambda_optimizer.contact_switch_confidence), 3),
                    "tight:", round(float(result.get("model_tightness", 0.0)), 3),
                    "accept:", int(bool(policy["value_info"].get("accept_p_arm", False))),
                    "pose_tie:", int(bool(policy["value_info"].get("pose_near_tie", True))),
                    "blocked:", int(bool((policy.get("value_info") or {}).get("path_blocked", False))),
                    "phase:", (policy.get("value_info") or {}).get("via_phase"),
                    "opposite:", edge_info["opposite"],
                    "keepout:", round(float(edge_info["keepout"]), 4),
                    "tip_r:", round(float(edge_info["tip_xy_r"]), 4),
                    "via_r:", round(float(edge_info["via_xy_r"]), 4),
                    "orbit_clips:", edge_info["orbit_clips"],
                    "dwell:", int(getattr(param.lambda_optimizer, "_dwell_steps", 0)),
                    "blacklisted:", len(getattr(param.lambda_optimizer, "_blocked_contact_indices", {})),
                )
            bus.publish_cmd({
                "kind": "via",
                "seq": int(rollout_step),
                "via": exec_via,
                "action": action,
                "policy_dt": POLICY_INTERVAL,
                "planner_dt": float(result.get("plan_dt", result.get("mppi_dt", 0.0))),
                "submitted_at": time.monotonic(),
                "solver_backend": result.get("solver_backend"),
                "press": policy["p_arm_world"],
                "release_pressure": bool(policy.get("release_pressure", False)),
                "preserve_upright": bool(policy.get("preserve_upright", False)),
                "object_cost_stage": policy.get("object_cost_stage"),
                "tool_rotation_world": policy.get("tool_rotation_world"),
                "approach_phase": (policy.get("value_info") or {}).get("via_phase"),
                "tool_rotation_action": policy.get("tool_rotation_action"),
                "tool_force_world": policy.get("tool_contact_force_world"),
                "tool_normal_world": policy.get("tool_contact_normal_world"),
                "path_blocked": bool(policy.get("execution_path_blocked",
                    (policy.get("value_info") or {}).get("path_blocked", False))),
                "markers": {
                    "target": policy["mpc_virtual_point"],
                    "best_contact": policy["best_contact_world"],
                },
            })
            rollout_step += 1
            previous_seq = obs.get('seq')
            obs = bus.wait_obs(timeout=180.0)
            if obs is not None and previous_seq is not None and obs.get('seq') is not None:
                observation_gaps.append(max(0, int(obs['seq'])-int(previous_seq)))
            iteration_wall_times.append(time.perf_counter() - iteration_wall_start)
            if obs is None:
                viewer_quit = True
                break

        trial_wall_time = time.perf_counter() - trial_wall_start
        if trial_record is not None:
            trial_record.finish(
                success=bool(trial_success), aborted=viewer_quit,
                execution_steps=rollout_step, policy_interval_s=POLICY_INTERVAL)
        identifier = trackers.get('scm_identifier')
        if identifier is not None:
            if obs is not None:
                identifier.observe(obs.get('scm_transitions', []), learn=False)
            if identifier.log_path:
                identifier.save(identifier.log_path.parent/'transitions.npz')
            identifier.learning_enabled = bool(getattr(args,'online_scm',False))
            print('[online-scm] final:', identifier.diagnostics())
        bus.publish_cmd({"cmd": "end_trial"})
        if last_result is not None:
            lambda_failures = int(last_result.get("lambda_failures", 0))
            mpc_failures = int(last_result.get("mpc_failures", 0))
            if lambda_failures or mpc_failures or last_result.get("mpc_init_error"):
                print("acados diagnostics:", {
                    "lambda_solves": int(last_result.get("lambda_solves", 0)),
                    "lambda_failures": lambda_failures,
                    "mpc_solves": int(last_result.get("mpc_solves", 0)),
                    "mpc_failures": mpc_failures,
                    "mpc_init_error": last_result.get("mpc_init_error") or None,
                })
        if edge_verdicts:
            counts = {}
            worse = [item for item in edge_verdicts if item[2].get("d_pos") is not None and item[2]["d_pos"] > 5e-4]
            for _, verdict, _ in edge_verdicts:
                counts[verdict] = counts.get(verdict, 0) + 1
            worse_counts = {}
            for _, verdict, _ in worse:
                worse_counts[verdict] = worse_counts.get(verdict, 0) + 1
            print("edge_verdict_summary:", {
                "n": len(edge_verdicts),
                "n_pos_worse": len(worse),
                "all": counts,
                "pos_worse": worse_counts,
            })
        choose_arr = np.asarray(choose_times, dtype=np.float64) if choose_times else np.array([0.0])
        plan_arr = np.asarray(plan_times, dtype=np.float64) if plan_times else np.array([0.0])
        solve_mean_s = float(np.mean(choose_arr + plan_arr))
        trial_summary = {
            "trial": trial_count,
            "seed": trial_count,
            "numpy_seed": 100 + trial_count,
            "object": str(args.obj),
            "mode": "mpc_planner",
            "physics_backend": str(getattr(args, 'physics_backend', 'isaac')),
            "tool_orientation_control": str(getattr(args, 'tool_orientation_control', 'free')),
            "nullspace_stiffness": resolve_nullspace_stiffness(args),
            "mpc_horizon": int(plan_param.mpc_horizon_),
            "nullspace_damping_model": ('projected-inertia' if plan_param.nullspace_inertia_damping_ else 'legacy'),
            "num_envs": int(getattr(args, 'num_envs', 1)),
            "mjwarp_scene": (getattr(args, 'mjwarp_scene', 'legacy')
                             if getattr(args, 'physics_backend', 'isaac') == 'mjwarp' else None),
            "headless": bool(getattr(args, 'headless', False)),
            "async_planner": bool(getattr(args, 'async_planner', True)),
            "headless_realtime": bool(getattr(args, 'headless_realtime', True)),
            "pose_executor_constraints": bool(getattr(args, 'pose_executor_constraints', False)),
            "flip_executor_constraints": bool(getattr(args, 'flip_executor_constraints', True)),
            "physics_policy_ticks": None if obs is None else obs.get('seq'),
            "sim_end_time_s": None if obs is None else obs.get('sim_time'),
            "observation_gap_mean": float(np.mean(observation_gaps)) if observation_gaps else None,
            "observation_gap_max": max(observation_gaps) if observation_gaps else None,
            "physics_scene_sha256": None if obs is None else obs.get('physics_scene_sha256'),
            "mjwarp_nconmax": int(getattr(args, 'mjwarp_nconmax', 512)),
            "mjwarp_njmax": int(getattr(args, 'mjwarp_njmax', 2048)),
            "joint_velocity_limiter": ('poststep_projection'
                if getattr(args, 'physics_backend', 'isaac') == 'mjwarp'
                and getattr(args, 'mjwarp_scene', 'legacy') == 'isaac' else None),
            "joint_velocity_projections": 0 if obs is None else int(obs.get('joint_velocity_projections', 0)),
            "joint_velocity_max_unlimited": 0. if obs is None else float(obs.get('joint_velocity_max_unlimited', 0.)),
            "contact_dynamics_variant": getattr(args, "contact_dynamics_variant", "legacy"),
            "shared_contact_dynamics": bool(getattr(args, "shared_contact_dynamics", False)),
            "solver_requested": str(getattr(args, 'solver_backend', 'auto')),
            "solver_backend": (getattr(mpc, 'planner_solver_', None) if last_result is None
                               else last_result.get('solver_backend')),
            "planner_solver_backend": str(getattr(mpc, 'planner_solver_', 'ipopt')),
            "lambda_solver_backend": str(getattr(plan_param.lambda_optimizer, 'solver', 'ipopt')),
            "mpc_solves": int(getattr(mpc, 'acados_solve_count', 0)),
            "lambda_solves": int(getattr(plan_param.lambda_optimizer, 'acados_solve_count', 0)),
            "mpc_init_error": str(getattr(mpc, '_acados_init_error', '') or ''),
            "lambda_init_error": str(getattr(plan_param.lambda_optimizer, 'acados_error', '') or ''),
            "mpc_fallbacks": int(getattr(mpc, 'acados_fallback_count', 0)),
            "mpc_contact_resets": int(trackers.get('mpc_contact_resets', 0)),
            "mpc_floor_recovery_max_m": max_floor_recovery,
            "lambda_fallbacks": int(getattr(plan_param.lambda_optimizer, 'acados_fallback_count', 0)),
            # Wall timing excludes reset/model construction and final reporting.
            # Iterations include policy work, diagnostic output and bus/physics wait.
            "trial_wall_time_s": trial_wall_time,
            "completion_wall_time_s": trial_wall_time if trial_success else None,
            # Full CSO/approach/MPC call; excludes diagnostics, IPC and physics.
            "planning_wall_count": len(planning_wall_times),
            "planning_wall_total_s": float(sum(planning_wall_times)),
            "planning_wall_sum_squares_s2": float(np.dot(planning_wall_times, planning_wall_times)),
            "planning_wall_mean_ms": (float(np.mean(planning_wall_times))*1000
                                      if planning_wall_times else None),
            "planning_wall_std_ms": (float(np.std(planning_wall_times, ddof=1))*1000
                                     if len(planning_wall_times) > 1 else None),
            "iteration_wall_count": len(iteration_wall_times),
            "iteration_wall_total_s": float(sum(iteration_wall_times)),
            "iteration_wall_mean_s": (float(np.mean(iteration_wall_times))
                                      if iteration_wall_times else None),
            "iteration_wall_p95_s": (float(np.percentile(iteration_wall_times, 95))
                                     if iteration_wall_times else None),
            "iteration_wall_max_s": (float(max(iteration_wall_times))
                                     if iteration_wall_times else None),
            "success": int(trial_success),
            "steps": rollout_step,
            "completion_steps": rollout_step if trial_success else None,
            "completion_time_s": (float(obs['sim_time']) - trackers['object_cost_stages'].start_time
                if trial_success and obs is not None and obs.get('sim_time') is not None
                and trackers.get('object_cost_stages') is not None
                and trackers['object_cost_stages'].start_time is not None else None),
            "terminal_reward": discounted_terminal_reward(
                trial_success, rollout_step, gamma=metric_gamma, eta=metric_eta,
            ),
            "metric_gamma": metric_gamma,
            "metric_eta": metric_eta,
            "policy_interval_s": POLICY_INTERVAL,
            "aborted": viewer_quit,
            "reward_origin": "trial_start",
            "pose_applies": pose_apply_count,
            "physics_pressure_releases": 0 if obs is None else int(obs.get('physics_pressure_releases', 0)),
            "pressure_releases": 0 if trackers.get('upright_release') is None else trackers['upright_release'].releases,
            "upright_action_rejections": 0 if trackers.get('upright_release') is None else trackers['upright_release'].rejected_tilts,
            **({} if trackers.get('object_cost_stages') is None else trackers['object_cost_stages'].diagnostics()),
            "lambda_failures": 0 if last_result is None else last_result.get('lambda_failures', 0),
            "mpc_failures": 0 if last_result is None else last_result.get('mpc_failures', 0),
            "final_pos_err": None if pos_err_now is None else round(float(pos_err_now), 5),
            "final_quat_err": None if quat_err_now is None else round(float(quat_err_now), 5),
            "min_pos_err": None if not np.isfinite(min_pos_err) else round(float(min_pos_err), 5),
            "min_quat_err": None if not np.isfinite(min_quat_err) else round(float(min_quat_err), 5),
            "choose_dt_mean": round(float(np.mean(choose_arr)), 5),
            "choose_dt_max": round(float(np.max(choose_arr)), 5),
            "choose_dt_p95": round(float(np.percentile(choose_arr, 95)), 5),
            "plan_dt_mean": round(float(np.mean(plan_arr)), 5),
            "solve_dt_mean_s": round(solve_mean_s, 5),
            "solve_dt_mean_ms": round(solve_mean_s * 1000.0, 3),
            "cso_guidance": str(getattr(args, "cso_guidance", "cso")),
            "cpo_variant": str(getattr(args, "cpo_variant", "full")),
            "ablation_variant": str(getattr(args, "ablation_variant", "") or ""),
            "ablation_mode": args.ablation_mode,
            "ablation_description": MODES[args.ablation_mode],
            "verify_beta": float(args.verify_beta),
            "verify_accept_threshold": float(args.verify_accept_threshold),
            "verify_release_threshold": float(args.verify_release_threshold),
            "cso_backend": (plan_param.cso_ablation_.diagnostics() if getattr(plan_param, "cso_ablation_", None) is not None else None),
            "target_type": str(getattr(args, "target_type", "")),
            "friction_reg_coef": float(args.friction_reg_coef),
            "lambda_friction_cone": str(getattr(param.lambda_optimizer, 'friction_cone', 'box')),
            "fingertip_object_friction": float(param.mu_fingertip_object_),
            "table_object_friction": float(param.mu_table_object_),
            "collision_geometry": getattr(param, 'collision_geometry_', 'hull'),
            "collision_part_count": len(getattr(param, 'collision_mesh_paths_', [])),
            "contact_sampling": getattr(param, 'contact_sampling_', 'vertices'),
            "contact_sample_count": len(param.lambda_optimizer.sample_point),
            "lambda_contact_geometry": getattr(plan_param.lambda_optimizer, 'contact_geometry', 'mesh'),
            **({} if getattr(plan_param.lambda_optimizer, 'obb_geometry', None) is None else
               dict(a3_obb=plan_param.lambda_optimizer.obb_geometry.diagnostics())),
        }
        print("trial_summary: " + repr(trial_summary))
        trial_summaries.append(trial_summary)
        if hasattr(bus, 'record_summary'):
            bus.record_summary(trial_summary)
        if metric_path:
            with open(metric_path, 'a', encoding='utf-8') as metric_file:
                metric_file.write(json.dumps(trial_summary, allow_nan=False) + '\n')
        backend = getattr(plan_param, 'cso_ablation_', None)
        if backend is not None and hasattr(backend, 'close'):
            backend.close()
        success_rate += int(trial_success)
        if viewer_quit:
            break

    print(
        f"Success rate over {trial_num} trials "
        f"(trial ids {trial_start} to {trial_stop - 1}): "
        f"{success_rate}/{trial_num} = {success_rate / trial_num:.2%}"
    )
    if (getattr(args, 'headless', False) and getattr(args, 'rollout_summary_json', '')
            and not hasattr(bus, 'record_summary')):
        # The original async Isaac bus keeps its latest-only command/obs
        # protocol. Write statistics in the planner without adding barriers.
        from examples.mpc.franka.experiments.mpc_ablation.rollout import report
        report(args, trial_summaries, rollout_started)
    return viewer_quit


def planner_worker(cmd_q, obs_q, ready_q, init):
    from examples.mpc.franka.experiments.mpc_ablation.bus import IsaacBus, args_from_init

    if ready_q is not None:
        ready_q.put({"ok": True, "pid": os.getpid()})
    run_mpc_planner(IsaacBus(cmd_q, obs_q), args_from_init(init))
