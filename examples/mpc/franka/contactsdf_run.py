"""Run the official ContactSDF MPC in the Franka task through a single-tip adapter."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def validate_options(args, argv):
    """Refuse options that would silently reintroduce another controller."""
    forbidden = ('--verify', '--ema', '--threshold', '--adaptive', '--ideal-contact',
                 '--cso-', '--sampling-c3', '--two-stage-object-cost', '--online-scm',
                 '--mpc-contact-multistart', '--tool-orientation-control')
    for flag in argv:
        if flag.startswith(forbidden):
            raise ValueError('Official ContactSDF does not use local strategy option '+flag)
    if '--async-planner' in argv or '--trial' in argv:
        raise ValueError('Official ContactSDF uses one synchronous plan/position step')
    if args.physics_backend != 'isaac' or args.num_envs != 1:
        raise ValueError('The single-tip ContactSDF interface uses one native Isaac scene per process')
    backend_named = any(flag == '--solver-backend' or flag.startswith('--solver-backend=') for flag in argv)
    if backend_named and args.solver_backend not in ('portable', 'auto', 'ipopt'):
        raise ValueError('Official ContactSDF uses IPOPT')
    if args.mpc_horizon not in (None, 4):
        raise ValueError('Official ContactSDF uses horizon 4')
    args.solver_backend = 'ipopt'
    args.contactsdf = True
    args.contact_model = 'contactsdf'
    args.two_stage_object_cost = False
    args.rollout = True
    args.async_planner = False
    args.tool_orientation_control = 'free'


def load_cached_task(args, trial):
    """Read a fixed task interface; no controller or strategy parameters."""
    import numpy as np
    path = getattr(args, 'contactsdf_task_json', None)
    if not path:
        return None
    task = json.loads(Path(path).read_text())
    if (task['object'] != args.obj or task['seed'] != trial
            or task['target_type'] != args.target_type):
        raise ValueError('Cached task object, seed or task type differs from the command')
    for key, size in [('goal_xyz', 3), ('goal_quat_wxyz', 4), ('init_policy_state', 10),
                      ('init_joint_position', 7)]:
        value = np.asarray(task[key], float)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError('Invalid cached task field '+key)
    return task


def apply_cached_task(scene, task):
    """Restore the measured initial pose and original goal, before simulation."""
    import numpy as np
    if task is not None:
        scene.target_p_ = np.asarray(task['goal_xyz'], float)
        scene.target_q_ = np.asarray(task['goal_quat_wxyz'], float)
        state = np.asarray(task['init_policy_state'], float)
        scene.init_obj_qpos_ = state[:7].copy()
        scene.init_fingertip_pos_ = state[7:].copy()
        scene.init_robot_qpos_ = np.asarray(task['init_joint_position'], float)
        if task.get('collision_surface'):
            # Preserve a cached concave object's original decomposition even
            # when the current scene factory uses a newer decomposition recipe.
            scene.mesh_path_ = task['collision_surface']
            scene.collision_mesh_paths_ = list(task['collision_parts'])
            scene.collision_geometry_ = 'decomposed'


def run(args):
    import numpy as np
    from planning.contactsdf_franka import FrankaContactSDF
    from examples.mpc.franka.runtime.isaac import build_osc_param, viewer_closed
    from examples.mpc.franka.runtime.mpc import IsaacFrankaOSCSimulator
    from utils import metrics

    summaries = []
    output = Path(args.rollout_summary_json).parent if args.rollout_summary_json else Path('output/contactsdf')
    geometry_directory = output/'contactsdf_geometry'
    # Keep the cached Franka task's duration, while retaining upstream h=0.1.
    task_time_limit = int(args.max_rollout_length)*.02
    for trial in range(int(args.trial_start), int(args.trial_start)+int(args.trial_num)):
        scene = build_osc_param(args, trial)
        assert scene.lambda_optimizer is None
        cached_task = load_cached_task(args, trial)
        apply_cached_task(scene, cached_task)
        controller = FrankaContactSDF(scene, geometry_directory)
        h = controller.param.h_
        max_actions = int(np.floor(task_time_limit/h+1e-9))
        env = IsaacFrankaOSCSimulator(scene, headless=args.headless,
            sim_device=args.sim_device, graphics_device_id=args.graphics_device_id)
        record = metrics.open_trial_recorder(args, trial, scene.target_p_, scene.target_q_, method='contactsdf')
        timings, detection_timings, full_timings, statuses = [], [], [], []
        success = aborted = False
        start = time.perf_counter()
        initial_time = float(env.gym.get_sim_time(env.sim))
        try:
            initial_difference = None
            if cached_task is not None:
                # Set the cached joint configuration once; never replay actions.
                env._set_arm_qpos(cached_task['init_joint_position'])
                env._sync_desired_pose_with_current()
                measured = np.r_[env.get_policy_state(), env.get_current_joint_position()]
                expected = np.r_[cached_task['init_policy_state'], cached_task['init_joint_position']]
                initial_difference = float(np.max(np.abs(measured-expected)))
                if initial_difference > 2e-6:
                    raise ValueError('Cached initial state does not match: max abs difference %g' % initial_difference)
            env.show_target_object_pose(scene.target_p_, scene.target_q_)
            frames = max(1, int(round(h/env.sim_dt_)))
            if not np.isclose(frames*env.sim_dt_, h):
                raise ValueError('Isaac timestep must divide the upstream ContactSDF period 0.1s')
            for step in range(max_actions+1):
                state = np.asarray(env.get_policy_state(), float)
                pos_error = float(metrics.comp_pos_error(state[:3], scene.target_p_))
                quat_error = float(metrics.comp_quat_error(state[3:7], scene.target_q_))
                if record is not None:
                    record.observe(step, state[:7], env.get_current_joint_position(),
                        sim_time=float(env.gym.get_sim_time(env.sim)))
                if viewer_closed(env) or env.break_out_signal_:
                    aborted = True
                    break
                # Evaluation only: this check never changes controller costs or modes.
                if pos_error <= .02 and quat_error <= .015:
                    success = True
                    break
                if step == max_actions:
                    break
                before = time.perf_counter()
                result = controller.plan(state)
                full_timings.append(time.perf_counter()-before)
                timings.append(result['planning_callback_time_s'])
                detection_timings.append(result['contact_preparation_time_s'])
                statuses.append(result['solve_status'])
                if record is not None:
                    record.note_planning(timings[-1])
                action = np.asarray(result['action'], float).reshape(3)
                if not np.isfinite(action).all():
                    raise FloatingPointError('Upstream ContactSDF returned a nonfinite action')
                # Actuator interface only: track the returned displacement using
                # plain Cartesian impedance, bypassing via/press/OSC contact guards.
                env.set_desired_pose(state[7:10]+action, env.R_d_hold)
                for _ in range(frames):
                    env.set_control_torque(env.compute_cartesian_impedance_control())
                    env._simulate_once(sync_realtime=bool(args.headless_realtime), draw=not args.headless)
            elapsed = time.perf_counter()-start
            if record is not None:
                record.finish(success=success, aborted=aborted, execution_steps=len(timings), policy_interval_s=h)
            summary = dict(trial=trial, seed=trial, object=args.obj, success=int(success), aborted=aborted,
                target_type=args.target_type, cached_task_json=getattr(args, 'contactsdf_task_json', None),
                cached_initial_max_abs_diff=initial_difference,
                steps=len(timings), completion_steps=len(timings) if success else None,
                completion_time_s=float(env.gym.get_sim_time(env.sim))-initial_time if success else None,
                trial_wall_time_s=elapsed, completion_wall_time_s=elapsed if success else None,
                final_pos_err=pos_error, final_quat_err=quat_error,
                controller='contactsdf_official_mpc_single_tip', contact_model='contactsdf',
                verify=False, cso=False, approach=False, staged_cost=False, multistart=False,
                physics_backend='isaac', headless=args.headless, async_planner=False,
                planner_solver_backend='ipopt', lambda_solver_backend=None,
                policy_interval_s=h, task_time_limit_s=task_time_limit, max_mpc_actions=max_actions,
                contactsdf_theta=controller.theta.tolist(), contactsdf_theta_source=scene.contactsdf_theta_source_,
                planning_wall_count=len(timings), planning_wall_total_s=sum(timings),
                planning_wall_sum_squares_s2=float(np.dot(timings, timings)),
                planning_wall_mean_ms=float(np.mean(timings)*1000) if timings else None,
                planning_wall_std_ms=float(np.std(timings, ddof=1)*1000) if len(timings)>1 else None,
                planning_timing_scope='callback_after_contact_preparation_v1',
                planning_time_s=timings,
                contact_preparation_time_s=detection_timings,
                detection_and_planning_time_s=full_timings,
                solve_status=statuses[-1] if statuses else None,
                mpc_failures=sum(s not in ('Solve_Succeeded', 'Solved_To_Acceptable_Level') for s in statuses),
                lambda_failures=0, mpc_fallbacks=0, lambda_fallbacks=0)
            summaries.append(summary)
            if args.ranking_metrics_jsonl:
                path = Path(args.ranking_metrics_jsonl)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a') as f:
                    f.write(json.dumps(summary, allow_nan=False)+'\n')
        finally:
            env.close()
        if aborted:
            raise RuntimeError('ContactSDF trial aborted; evaluation is incomplete')
    aggregate = dict(controller='contactsdf_official_mpc_single_tip', trials=len(summaries),
        successes=sum(s['success'] for s in summaries),
        success_rate=sum(s['success'] for s in summaries)/len(summaries),
        verify=False, cso=False, approach=False, cases=summaries)
    if args.rollout_summary_json:
        path = Path(args.rollout_summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(aggregate, indent=2, allow_nan=False)+'\n')
    print('rollout_summary: '+json.dumps(aggregate, allow_nan=False), flush=True)
    return aggregate


def main():
    from planning.cpu_runtime import configure_compute_threads
    configure_compute_threads()
    if '--help' in sys.argv or '-h' in sys.argv:
        os.environ['SCSP_PLANNER_ONLY'] = '1'
    else:
        os.environ.pop('SCSP_PLANNER_ONLY', None)
        from isaacgym import gymapi, gymtorch  # Must precede Torch in runtime imports.
    from examples.mpc.franka.runtime.mpc import _add_rollout_policy_args
    parser = _add_rollout_policy_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--contactsdf-task-json', help='Fixed cached task pose/goal and initial joint configuration.')
    args = parser.parse_args()
    try:
        validate_options(args, sys.argv[1:])
    except ValueError as exc:
        parser.error(str(exc))
    run(args)


if __name__ == '__main__':
    main()
