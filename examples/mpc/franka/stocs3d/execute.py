"""Track a STOCS-3D object trajectory with the Franka Isaac OSC executor."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

# Same termination as run_mpc_planner in runtime/mpc.py.
SUCCESS_POS = 0.02
SUCCESS_QUAT = 0.015


def validate_options(args):
    if getattr(args, 'contactsdf', False) or getattr(args, 'contact_model', '') == 'contactsdf':
        raise ValueError('STOCS-3D execution does not use ContactSDF')
    if args.physics_backend != 'isaac' or int(args.num_envs) != 1:
        raise ValueError('STOCS-3D execution uses one Isaac scene')
    if args.floating_fingertip:
        raise ValueError('STOCS-3D execution uses the Franka OSC scene')


def load_trial(directory, trial):
    path = Path(directory) / ('trial_%03d.json' % int(trial))
    if not path.is_file():
        raise ValueError('No STOCS-3D trajectory for trial %d at %s' % (trial, path))
    return json.loads(path.read_text())


def run(args):
    import numpy as np
    from examples.mpc.franka.runtime.isaac import IsaacFrankaOSCSimulator, build_osc_param, viewer_closed
    from examples.mpc.franka.stocs3d.trajectory import position_at
    from utils import metrics

    summaries = []
    limit = max(1, int(args.max_rollout_length)) * 0.02
    for trial in range(int(args.trial_start), int(args.trial_start) + max(1, int(args.trial_num))):
        plan = load_trial(args.stocs_trajectory, trial)
        scene = build_osc_param(args, trial)
        env = IsaacFrankaOSCSimulator(
            scene, headless=args.headless, sim_device=args.sim_device,
            graphics_device_id=args.graphics_device_id)
        record = metrics.open_trial_recorder(args, trial, scene.target_p_, scene.target_q_, method='stocs3d')
        knot_times = np.asarray(plan['times'], dtype=float)
        knots = np.asarray(plan['fingertip_xyz'], dtype=float)
        times = np.concatenate([[0.0], float(plan['approach_s']) + knot_times])
        points = np.vstack([np.asarray(scene.init_fingertip_pos_, dtype=float), knots])
        success = aborted = False
        start = time.perf_counter()
        initial = float(env.gym.get_sim_time(env.sim))
        pos_error = quat_error = float('inf')
        steps = 0
        try:
            env.show_target_object_pose(scene.target_p_, scene.target_q_)
            while True:
                state = np.asarray(env.get_policy_state(), dtype=float)
                pos_error = float(metrics.comp_pos_error(state[:3], scene.target_p_))
                quat_error = float(metrics.comp_quat_error(state[3:7], scene.target_q_))
                now = float(env.gym.get_sim_time(env.sim)) - initial
                if record is not None:
                    record.observe(steps, state[:7], env.get_current_joint_position(), sim_time=now + initial)
                if viewer_closed(env) or env.break_out_signal_:
                    aborted = True
                    break
                if pos_error <= SUCCESS_POS and quat_error <= SUCCESS_QUAT:
                    success = True
                    break
                if now >= limit:
                    break
                env.set_desired_pose(position_at(times, points, now), env.R_d_hold)
                env.set_control_torque(env.compute_cartesian_impedance_control())
                env._simulate_once(sync_realtime=bool(args.headless_realtime), draw=not args.headless)
                steps += 1
        finally:
            elapsed = time.perf_counter() - start
            if record is not None:
                record.finish(success=success, aborted=aborted, execution_steps=steps, policy_interval_s=float(env.sim_dt_))
            env.close()
        summaries.append(dict(
            trial=trial, object=args.obj, success=int(success), aborted=aborted,
            stocs_solved=bool(plan.get('solved', False)), stocs_iterations=plan.get('iterations'),
            planning_time_s=plan.get('planning_time_s'),
            execution_time_s=(now if success else None),
            simulated_time_s=now,
            final_pos_err=pos_error, final_quat_err=quat_error, steps=steps,
            trial_wall_time_s=elapsed, controller='stocs3d',
        ))
        print('[stocs3d] trial %d success=%s pos=%.4f quat=%.4f' % (
            trial, success, pos_error, quat_error), flush=True)
    if args.rollout_summary_json:
        path = Path(args.rollout_summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(controller='stocs3d', trials=summaries), indent=2) + '\n')
    return summaries


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        os.environ['SCSP_PLANNER_ONLY'] = '1'
    else:
        os.environ.pop('SCSP_PLANNER_ONLY', None)
        from isaacgym import gymapi, gymtorch  # Must precede Torch.
    from examples.mpc.franka.runtime.mpc import _add_rollout_policy_args
    parser = _add_rollout_policy_args(argparse.ArgumentParser(
        description='Execute a STOCS-3D upper trajectory in Isaac Gym.'))
    parser.add_argument('--stocs-trajectory', required=True,
                        help='Directory of trial_XXX.json files written by the offline STOCS-3D solve.')
    args = parser.parse_args()
    try:
        validate_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    run(args)


if __name__ == '__main__':
    main()
