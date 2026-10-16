"""Compare old FA and direct-OCP costs on identical successful control sequences."""
from pathlib import Path
import json
import numpy as np


def main():
    from examples.retarget.matrix_comfree import ComFreeEngine
    from examples.retarget.fa_direct_mpc import DirectComFreeMPC
    e=ComFreeEngine(Path('outputs/fa_admm_solver_20261016'),Path('thirdparty/retarget_assets/DexForge'))
    c=DirectComFreeMPC(e)
    from examples.retarget.fa_matched_ablation import expand
    from forceaware.targets import make_window
    from forceaware.optimizer import ControlModel,DenseControlJob,_build_dense_control_kernel
    import warp as wp
    model=ControlModel();model.action_dim=26;model.hand_ctrl=wp.array(e.index.hand_ctrl.astype(np.int32),device=e.device)
    root=Path('outputs/fa_matched_ablation_20261016');history=np.load('outputs/fa_admm_solver_20261016/comfree/lbfgs_fixed/rollout.npz')
    rows=[]
    for probe in map(json.loads,(root/'probes/baseline/probes.jsonl').read_text().splitlines()):
        dense=probe['window']*3;saved=np.load(root/f'probes/baseline/window_{probe["window"]:04d}.npz')
        targets=make_window(e.sequence,executed_steps=dense,horizon=5,time_grid=e.cfg.simulator.time_grid,start_frame=0.,end_frame=71.)
        previous=history['ctrl_traj'][dense-1] if dense else history['qpos_traj'][0,:26]
        pp=history['ctrl_traj'][dense-2] if dense>=3 else previous
        nodes=targets.actions+saved['raw'][26:].reshape(5,26)
        left=wp.empty((1,26),device=e.device);right=wp.empty_like(left);out=wp.empty_like(left);commands=[]
        for k in range(5):
            left.assign((previous if k==0 else nodes[k-1])[None].astype(np.float32));right.assign(nodes[k][None].astype(np.float32))
            for sub in range(6):
                job=DenseControlJob();job.model=model;job.left_action=left;job.right_action=right;job.control=out;job.fraction=(sub+1)/6
                wp.launch(_build_dense_control_kernel,(1,26),inputs=[job],device=e.device);commands.append(out.numpy()[0])
        commands=np.asarray(commands)
        c.native.prepare(targets);states=c.rollout(np.r_[history['qpos_traj'][dense],history['qvel_traj'][dense]],commands,previous,pp)
        parameters=[np.r_[k,targets.dense_actions[min(k,29)],targets.object_qpos[min(k,29)],2. if k==30 else 0. if k==29 else 1.] for k in range(31)]
        cost=c.trajectory_cost(states,commands,parameters)
        tracking_pre=0.;tracking_post=0.
        for k in range(30):
            before=np.asarray(c.pose_tracking_function(states[k],parameters[k])).ravel()
            after=np.asarray(c.pose_tracking_function(states[k+1],parameters[k])).ravel()
            tracking_pre+=float(before@before);tracking_post+=float(after@after)
        rm=c.native.regularization;indices=e.model.nq+e.dadr
        v0=(states[0,:c.nphysical]*c.physical_scale)[indices]
        vN=(states[-1,:c.nphysical]*c.physical_scale)[indices]
        velocity_shift=1e-6*e.cfg.smoothing.hand_state_weight*np.sum(rm.state_component_ratio.numpy()*(v0*v0-vN*vN)/rm.state_velocity_scale.numpy()**2)
        physical=states[1:,:c.nphysical]*c.physical_scale
        qerror=float(np.max(abs(physical[:,:e.model.nq]-saved['predicted_qpos'])))
        row=dict(time=probe['time'],old_FA_cost=probe['final_FA_cost'],direct_OCP_cost=cost,
            relative_difference=(cost/probe['final_FA_cost']-1),predicted_qpos_max_difference=qerror)
        row.update(tracking_pre_cost=tracking_pre,tracking_post_cost=tracking_post,
            tracking_timing_difference=tracking_pre-tracking_post,state_velocity_timing_difference=float(velocity_shift),
            control_smoothing_difference_plus_roundoff=float(cost-probe['final_FA_cost']-(tracking_pre-tracking_post)-velocity_shift),
            direct_cost_after_fixing_tracking_timing=cost-(tracking_pre-tracking_post))
        assert qerror<3e-5,row
        rows.append(row);print(json.dumps(row),flush=True)
    (root/'objective_discretization_audit.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
