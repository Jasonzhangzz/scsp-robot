"""Independent A6 trajectory planner and trial loop for the Isaac Gym bus.

Observations -> ComFree shooting -> AdamW controls -> Isaac OSC. This path
does not construct Lambda, a contact policy, ranking/verification trackers,
or approach/dwell decisions.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from planning.franka_adamw_ablation import FrankaComFreeAdamW


def build_runtime(init, param=None):
    from .config import configure_ablation
    from .params import ExplicitMPCParams
    from .mpc import adapt_param_for_cartesian_solver, _configure_rollout_param
    from .stages import make_object_cost_stages
    from planning.object_pose_cost import target_up_axis

    args = configure_ablation(argparse.Namespace(**init['args']))
    args.rollout = True
    if param is None:
        param = ExplicitMPCParams(args, rand_seed=int(init['trial_count']),
            target_type=getattr(args, 'target_type', 'ground-rotation'), build_optimizer=False)
    param = adapt_param_for_cartesian_solver(param, args, build_optimizer=False)
    param = _configure_rollout_param(param, args)
    for name in ('target_p_', 'target_q_', 'mesh_path_', 'table_height',
                 'gravity_', 'attract_coef', 'contact_coef'):
        if name in init:
            setattr(param, name, init[name])
    stages = make_object_cost_stages(args, target_up_axis(param.target_q_))
    if stages.enabled:
        param.object_cost_config_ = stages.config()
    controller = FrankaComFreeAdamW(param, args)
    return args, param, controller, dict(object_cost_stages=stages, sol_guess=None)


def handle_request(args, param, controller, trackers, observation):
    """Optimize every observation, whether or not contact has been reached."""
    from planning.execution_constraints import floor_command_bounds, table_edge_command_bounds

    stages = trackers['object_cost_stages']
    if stages.update(observation):
        trackers['sol_guess'] = None
    if stages.enabled:
        param.object_cost_config_ = stages.config()
    state = np.asarray(observation['policy_q'], dtype=float).reshape(10)
    phi = (np.ones(4*param.max_ncon_) if observation.get('phi_vec') is None else
           np.asarray(observation['phi_vec'], dtype=float))
    jac = (np.zeros((4*param.max_ncon_, param.n_qvel_)) if observation.get('jac_mat') is None else
           np.asarray(observation['jac_mat'], dtype=float))
    lower, upper = param.mpc_u_lb_.copy(), param.mpc_u_ub_.copy()
    # Physical endpoint bounds are independent of any contact-point choice.
    align = ((stages.stage == 'flip' and getattr(args, 'flip_executor_constraints', True))
             or (stages.stage == 'pose' and getattr(args, 'pose_executor_constraints', False)))
    edge_applied = False
    if align:
        from .mpc import _table_edge_gap_and_normal, ISAAC_TABLE_EDGE_GUARD_DISTANCE
        gap, normal = _table_edge_gap_and_normal(state, param)
        lower, upper, edge_applied = table_edge_command_bounds(lower, upper, gap, normal,
            ISAAC_TABLE_EDGE_GUARD_DISTANCE, bool(observation.get('if_contact', False)))
    previous = observation.get('executed_action')
    first_bounds = None
    radius = float(np.max(np.abs(param.mpc_u_ub_)))
    if controller.contact_force_limit is not None:
        radius = min(radius, controller.contact_force_limit/np.linalg.norm(param.robot_stiff_, 2))
    slew = float(param.isaac_action_slew_)/np.sqrt(3.)
    if previous is not None:
        previous = np.clip(np.asarray(previous, dtype=float).reshape(3), lower, upper)
        previous *= min(1., radius/max(np.linalg.norm(previous), 1e-12))
        first_bounds = (np.maximum(lower, previous-slew), np.minimum(upper, previous+slew))
    controller.first_action_bounds, floor_info = floor_command_bounds(
        state[9], param.mpc_q_lb_[9], lower, upper, first_bounds=first_bounds,
        previous=previous, slew=slew, max_norm=radius)
    result = controller.plan_once(param.target_p_, param.target_q_, state, phi, jac,
        sol_guess=trackers['sol_guess'], u_lb=lower, u_ub=upper)
    trackers['sol_guess'] = result['sol_guess']
    result.update(seq=observation.get('seq'), physics_backend='isaac',
        robot_trajectory_world=result['rollout_q'][:, 7:10].copy(),
        object_cost_stage=stages.stage, object_cost_diagnostics=stages.diagnostics(),
        executor_floor=floor_info, executor_edge=edge_applied,
        mpc_solves=controller.adamw_solve_count)
    return result


def trajectory_command(result, seq):
    """The Isaac executor receives only the optimized first displacement."""
    return dict(kind='trajectory', seq=int(seq), action=np.asarray(result['action']).copy(),
        policy_dt=.02, planner_dt=float(result['solve_time']),
        submitted_at=time.monotonic(), solver_backend='adamw',
        object_cost_stage=result['object_cost_stage'])


def _append_json(path, record):
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as output:
            output.write(json.dumps(record, allow_nan=False,
                default=lambda value: np.asarray(value).tolist())+'\n')


def run_adamw_planner(bus, args):
    from .bus import wait_trial_obs
    from .config import configure_ablation
    from .params import ExplicitMPCParams
    from .mpc import _apply_dywa_physics_to_param
    from utils import metrics

    configure_ablation(args)
    if args.trial_start < 0 or args.trial_num < 1:
        raise ValueError('trial-start must be nonnegative and trial_num positive')
    summaries, viewer_quit = [], False
    run_started = time.perf_counter()
    for trial in range(args.trial_start, args.trial_start+args.trial_num):
        param = ExplicitMPCParams(args, rand_seed=trial, target_type=args.target_type,
                                  build_optimizer=False)
        param = _apply_dywa_physics_to_param(param)
        _, param, controller, trackers = build_runtime(dict(args=vars(args), trial_count=trial), param)
        print('[a6-startup] trial=%d physics=isaac predictor=comfree solver=adamw horizon=%d' %
              (trial, controller.horizon), flush=True)
        observation = wait_trial_obs(bus, trial)
        if observation is None:
            viewer_quit = True
            break
        trial_record = metrics.open_trial_recorder(
            args, trial, param.target_p_, param.target_q_)
        _append_json(getattr(args, 'rollout_trace_jsonl', ''), dict(
            event='setup', trial=trial, target_p=np.asarray(param.target_p_).tolist(),
            target_q=np.asarray(param.target_q_).tolist()))
        started = time.perf_counter()
        durations, pos_errors, quat_errors = [], [], []
        result, success, step = None, False, 0
        for step in range(max(1, args.max_rollout_length)+1):
            if observation is None or any(observation.get(key) for key in ('break_out', 'viewer_closed')) or observation.get('cmd') == 'stop':
                viewer_quit = True
                break
            state = np.asarray(observation['policy_q'], dtype=float)
            _append_json(getattr(args, 'rollout_trace_jsonl', ''), dict(
                event='pose', trial=trial, step=int(step), policy_q=state.tolist()))
            pos_errors.append(float(metrics.comp_pos_error(state[:3], param.target_p_)))
            quat_errors.append(float(metrics.comp_quat_error(state[3:7], param.target_q_)))
            if trial_record is not None:
                trial_record.observe(step, state[:7], metrics.robot_q_from_observation(observation),
                                     sim_time=observation.get('sim_time'))
            stages = trackers['object_cost_stages']
            if stages.update(observation):
                trackers['sol_guess'] = None
            success = (pos_errors[-1] <= .02 and quat_errors[-1] <= .015
                       and (not stages.enabled or stages.stage == 'pose'))
            if success or step >= max(1, args.max_rollout_length):
                break
            result = handle_request(args, param, controller, trackers, observation)
            durations.append(result['solve_time'])
            if trial_record is not None:
                trial_record.note_planning(result['solve_time'])
            if getattr(args, 'rollout_verbose', False):
                print('[a6] step=%d cost=%.6g solve_ms=%.2f action=%s' % (
                    step, result['cost_opt'][0], result['solve_time']*1000,
                    np.array2string(result['action'], precision=5)), flush=True)
            _append_json(getattr(args, 'rollout_trace_jsonl', ''), dict(
                event='trajectory', trial=trial, step=step, observation_seq=observation.get('seq'),
                policy_q=state, action=result['action'], controls=result['u_traj'],
                robot_trajectory_world=result['robot_trajectory_world'],
                cost=float(result['cost_opt'][0]), initial_cost=result['initial_cost'],
                solver_backend='adamw', gradient_method=result['gradient_method'],
                planning_ms=result['solve_time']*1000))
            bus.publish_cmd(trajectory_command(result, step))
            observation = bus.wait_obs(timeout=180.)
        bus.publish_cmd(dict(cmd='end_trial'))
        stages = trackers['object_cost_stages']
        summary = dict(trial=trial, seed=trial, object=args.obj, ablation_mode='a6',
            physics_backend='isaac', solver_backend='adamw', planner_solver_backend='adamw',
            contact_model='comfree', gradient_method='central_difference', success=bool(success),
            aborted=viewer_quit, rollout_steps=len(durations), mpc_solves=controller.adamw_solve_count,
            mpc_fallbacks=0, final_pos_err=pos_errors[-1] if pos_errors else None,
            final_quat_err=quat_errors[-1] if quat_errors else None,
            min_pos_err=min(pos_errors) if pos_errors else None,
            min_quat_err=min(quat_errors) if quat_errors else None,
            solve_dt_mean_ms=float(np.mean(durations))*1000 if durations else 0.,
            trial_wall_time_s=time.perf_counter()-started,
            a6_adamw=dict(iterations=controller.iterations, learning_rate=controller.learning_rate,
                weight_decay=controller.weight_decay, fd_epsilon=controller.fd_epsilon,
                center_weight=param.a6_center_weight),
            **stages.diagnostics())
        if trial_record is not None:
            from examples.mpc.franka.runtime.mpc import POLICY_INTERVAL
            trial_record.finish(success=bool(success), aborted=viewer_quit,
                                execution_steps=step, policy_interval_s=POLICY_INTERVAL)
        summaries.append(summary)
        _append_json(getattr(args, 'ranking_metrics_jsonl', ''), summary)
        if hasattr(bus, 'record_summary'):
            bus.record_summary(summary)
        else:
            print('trial_summary: '+json.dumps(summary, allow_nan=False), flush=True)
        if viewer_quit:
            break
    if (getattr(args, 'headless', False) and getattr(args, 'rollout_summary_json', '')
            and not hasattr(bus, 'record_summary') and not viewer_quit):
        from .rollout import report
        report(args, summaries, run_started)
    return viewer_quit
