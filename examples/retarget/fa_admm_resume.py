"""Recover executor state by exact control replay, then resume the MPC clock."""
import json
import numpy as np


RECORDER_FIELDS={
    'qpos':'qpos_traj','qvel':'qvel_traj','qpos_frames':'qpos_frames',
    'raw_warm':'raw_warm_history','ctrl':'ctrl_traj','target_frames':'target_frames',
    'drift':'drift','rotation':'rot_err_deg','step_seconds':'mpc_step_sec',
    'step_sizes':'mpc_execution_steps','scores':'multistart_scores',
    'candidate_drifts':'multistart_drifts','has_finite_best':'multistart_has_finite_best',
    'candidate_valid':'multistart_valid','training_failure_iteration':'multistart_training_failure_iteration',
    'training_failure_code':'multistart_training_failure_code','rejection_code':'multistart_rejection_code',
    'selected':'multistart_selected',
}


def install(workflow,prefix,case):
    windows=len(prefix['mpc_execution_steps'])
    physics_steps=int(np.sum(prefix['mpc_execution_steps']))
    assert len(prefix['ctrl_traj'])==physics_steps

    def run(ctx):
        qpos=prefix['qpos_traj'][0].astype(float);qvel=prefix['qvel_traj'][0].astype(float)
        cursor=0;max_q=0.;max_v=0.
        for count in prefix['mpc_execution_steps']:
            count=int(count)
            execution=ctx.executor.execute_action_sequence(qpos,qvel,actions=prefix['ctrl_traj'][cursor:cursor+count])
            expected_q=prefix['qpos_traj'][cursor+1:cursor+count+1]
            expected_v=prefix['qvel_traj'][cursor+1:cursor+count+1]
            max_q=max(max_q,float(np.max(np.abs(execution.qpos_sequence.astype(np.float32)-expected_q))))
            max_v=max(max_v,float(np.max(np.abs(execution.qvel_sequence.astype(np.float32)-expected_v))))
            qpos,qvel=execution.next_qpos,execution.next_qvel;cursor+=count
        # Exact replay retains MuJoCo's qacc_warmstart and all internal data;
        # resetting q/v to the float32 archive would lose that state.
        if max_q!=0. or max_v!=0.:
            raise RuntimeError(f'Executor checkpoint replay is not exact: q={max_q}, v={max_v}')
        for attribute,key in RECORDER_FIELDS.items():
            setattr(ctx.recorder,attribute,[value.copy() if hasattr(value,'copy') else value for value in prefix[key]])
        warm=prefix['raw_warm_history'][-1].copy()
        previous=prefix['ctrl_traj'][-1].copy()
        ctx.optimizer.objective.regularizer._previous_previous_cpu=prefix['ctrl_traj'][physics_steps-4].copy()
        audit=dict(resume_window=windows,resume_physics_steps=physics_steps,
            executor_replay_max_qpos_error=max_q,executor_replay_max_qvel_error=max_v,
            executor_internal_state_recovered_by_replay=True,original_prefix_retained=True,
            optimizer_workspace_rebuilt=True)
        (case/'resume_audit.json').write_text(json.dumps(audit,indent=2))
        print('[resume] '+json.dumps(audit),flush=True)
        executed_steps=physics_steps
        for window in range(windows,len(ctx.initial.request.execution_steps_per_window)):
            count=ctx.initial.request.execution_steps_per_window[window]
            targets=workflow._step_targets(ctx,window,executed_steps=executed_steps)
            plan=ctx.optimizer.solve(qpos,qvel,targets,warm=warm,previous_action=previous,
                allow_init=False,seed=ctx.cfg.optimizer.multistart_seed+window,
                execution_steps=count,start_time=ctx.executor.elapsed)
            execution=ctx.executor.execute_action_sequence(qpos,qvel,actions=plan.control_sequence)
            warm=workflow._shift_warm(plan.raw,ctx.index.action_dim,ctx.cfg,execution_steps=count)
            ctx.recorder.record(plan,targets,qpos_sequence=execution.qpos_sequence,
                qvel_sequence=execution.qvel_sequence,index=ctx.index,warm=warm,
                display_frames=workflow._display_frames(ctx.cfg,ctx.initial.start_frame,executed_steps,execution_steps=count))
            qpos,qvel=execution.next_qpos,execution.next_qvel;previous=plan.action;executed_steps+=count
            workflow._print_step(window,len(ctx.initial.request.execution_steps_per_window),plan,recorder=ctx.recorder)
        assert executed_steps==ctx.initial.request.total_physics_steps

    workflow._run_steps=run
