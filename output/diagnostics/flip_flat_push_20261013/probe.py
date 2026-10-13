"""Read-only instrumentation of the normal Isaac planner/physics protocol."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path('/home/zz/scsp-robot')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def install_probe(prefix):
    import numpy as np
    import planning.mpc_explicit as planner
    from examples.mpc.franka.ik2 import test_mpc_isaac as entry
    original_payload = entry._plan_from_obs

    def observed_payload(obs, *args, **kwargs):
        payload, state = original_payload(obs, *args, **kwargs)
        for key in ('osc_force', 'policy_schedule', 'probe_joint_velocity', 'probe_force_samples'):
            payload[key] = obs.get(key)
        return payload, state

    entry._plan_from_obs = observed_payload
    original = planner.handle_mpc_request
    rows, snapshots = [], []

    def plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def inspected(args, param, mpc, trackers, msg):
        result = original(args, param, mpc, trackers, msg)
        policy = result['policy']
        row = dict(step=len(rows), sim_time=msg.get('sim_time'), seq=msg.get('seq'),
            state=msg['policy_q'], action=result['action'], executed_action=msg.get('executed_action'),
            force=msg.get('osc_force'), fingertip_velocity=msg.get('fingertip_linear_velocity'),
            full_q=msg.get('full_q'), joint_velocity=msg.get('probe_joint_velocity'),
            force_samples=msg.get('probe_force_samples'),
            linear_velocity=msg.get('object_linear_velocity'), angular_velocity=msg.get('object_angular_velocity'),
            actual_contact=msg.get('actual_contact'), if_contact=msg.get('if_contact'),
            config=param.object_cost_config_, target=np.r_[param.target_p_, param.target_q_],
            stage=policy.get('object_cost_stage'), mode=policy.get('execution_mode'),
            predictions=policy.get('contact_prediction_diagnostics'), value=policy.get('value_info'),
            press=policy['p_arm_world'], best=policy['best_contact_world'],
            confidence=param.lambda_optimizer.contact_switch_confidence,
            dwell=param.lambda_optimizer._dwell_steps,
            floor=result.get('mpc_executor_floor'), floor_recovery=result.get('mpc_floor_recovery_m'),
            first_action_bounds=mpc.first_action_bounds,
            u_traj=None if result['sol_guess'] is None else result['sol_guess'].get('u_traj'),
            rank_dt=result['rank_dt'], plan_dt=result['plan_dt'])
        rows.append(plain(row))
        if (row['if_contact'] and row['mode'] == 'contact' and
                (not snapshots or len(rows) - snapshots[-1][0] >= 20) and len(snapshots) < 16):
            state = mpc.observation_state(np.asarray(msg['policy_q']), msg,
                np.asarray(msg.get('fingertip_linear_velocity', np.zeros(3))))
            snapshot = dict(state=state, phi=np.asarray(msg['phi_vec']), jac=np.asarray(msg['jac_mat']),
                env_phi=np.asarray(msg['env_phi']), env_jac=np.asarray(msg['jac_mat_env']),
                config=np.asarray(param.object_cost_config_), target=np.r_[param.target_p_, param.target_q_],
                action=np.asarray(result['action']), virtual_point=np.asarray(policy['mpc_virtual_point']),
                contact_point=np.asarray(policy['mpc_contact_point']),
                first_lower=(np.full(3,-.005) if mpc.first_action_bounds is None else mpc.first_action_bounds[0]),
                first_upper=(np.full(3,.005) if mpc.first_action_bounds is None else mpc.first_action_bounds[1]),
                verify=float(not (policy.get('value_info') or {}).get('path_blocked',False)),
                args_json=np.array(json.dumps(plain(vars(args)))),
                row_json=np.array(json.dumps(plain(row))),
                u_traj=(np.zeros((5,3)) if result['sol_guess'] is None else np.asarray(result['sol_guess'].get('u_traj'))),
                full_q=np.asarray(msg['full_q']), joint_velocity=np.asarray(msg.get('probe_joint_velocity')))
            snapshots.append((len(rows), snapshot))
        return result

    planner.handle_mpc_request = inspected

    def finish():
        with (OUT / (prefix + '_probe.jsonl')).open('w') as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
        directory = OUT / (prefix + '_snapshots')
        directory.mkdir(exist_ok=True)
        for index, snapshot in snapshots:
            np.savez_compressed(directory / ('step_%03d.npz' % (index-1)), **snapshot)
        print('Probe saved %s cycles and %s contact snapshots' % (len(rows),len(snapshots)),flush=True)

    return finish


def instrumented_planner(cmd_q, obs_q, ready_q, init):
    os.environ['SCSP_PLANNER_ONLY'] = '1'
    from planning.cpu_runtime import configure_compute_threads
    configure_compute_threads()
    prefix = init['args']['obj'] + '_' + os.environ.get('SCSP_DIAG_TAG', 'async')
    finish = install_probe(prefix)
    try:
        from examples.mpc.franka.ik2.test_mpc_isaac import planner_worker
        planner_worker(cmd_q, obs_q, ready_q, init)
    finally:
        finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', required=True)
    parser.add_argument('--sync', action='store_true')
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--tag')
    config = parser.parse_args()
    from planning.cpu_runtime import configure_compute_threads
    configure_compute_threads()
    os.environ.pop('SCSP_PLANNER_ONLY', None)
    from isaacgym import gymapi, gymtorch
    from examples.mpc.franka.ik2.test_mpc_isaac import _add_rollout_policy_args, run_mpc_planner, IsaacFrankaOSCSimulator
    import numpy as np
    original_osc = IsaacFrankaOSCSimulator._compute_osc_torques

    def measured_osc(env):
        active = None if env._mpc_action is None else np.asarray(env._mpc_action).tolist()
        q = env.get_current_joint_position()
        velocity = env._get_position_jacobian(q) @ np.clip(env.get_current_joint_velocity(), -8, 8)
        error = np.asarray(env.position_d) - env.get_end_effector_pos()[0]
        result = original_osc(env)
        frames = getattr(env, '_probe_force_frames', [])
        frames.append(dict(action=active, force=np.asarray(env._last_osc_force).tolist(),
                           velocity=np.asarray(velocity).tolist(), error=error.tolist(),
                           contact=bool(env._fingertip_contact_info()[0])))
        env._probe_force_frames = frames
        return result

    IsaacFrankaOSCSimulator._compute_osc_torques = measured_osc
    from examples.mpc.franka.ik2.run import SynchronousMPCBus, isaac_loop
    import examples.mpc.franka.ik2.run as runtime
    original_pack = runtime.pack_state

    def measured_state(env, *args, **kwargs):
        state = original_pack(env, *args, **kwargs)
        state['probe_joint_velocity'] = env.get_current_joint_velocity()
        state['probe_force_samples'] = getattr(env, '_probe_force_frames', [])
        env._probe_force_frames = []
        return state

    runtime.pack_state = measured_state
    from examples.mpc.franka.ik2 import isaac_bus
    flags = ['--obj',config.obj,'--flipping','--headless','--trial_num','1',
             '--max_rollout_length',str(config.steps)]
    tag = config.tag or ('sync' if config.sync else 'async')
    os.environ['SCSP_DIAG_TAG'] = tag
    prefix=config.obj + '_' + tag
    flags += ['--ranking-metrics-jsonl',str(OUT / (prefix + '_trials.jsonl'))]
    flags += ['--rollout-trace-jsonl',str(OUT / (prefix + '_probe_trace.jsonl'))]
    if config.sync:
        flags += ['--sync-planner']
    args = _add_rollout_policy_args(argparse.ArgumentParser()).parse_args(flags)
    args.rollout=True
    if config.sync:
        finish=install_probe(prefix)
        bus=SynchronousMPCBus(args)
        try:
            run_mpc_planner(bus,args)
        finally:
            bus.close()
            finish()
    else:
        isaac_bus.mpc_planner_entry=instrumented_planner
        bus,process=isaac_bus.start_planner(args,'mpc')
        try:
            isaac_loop(bus,args,'mpc')
        finally:
            isaac_bus.stop_peer(bus,process)


if __name__ == '__main__':
    main()
