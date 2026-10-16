"""Check the reindexed tracking-time ablation against the literal pre-step loss."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np


def main():
    from examples.retarget.matrix_comfree import ComFreeEngine
    e=ComFreeEngine(Path('outputs/fa_admm_solver_20261016'),Path('thirdparty/retarget_assets/DexForge'))
    from examples.retarget.fa_contact_admm import ADMMSettings,optimizer_class
    from examples.retarget.fa_matched_ablation import install,constant_tracking
    from forceaware.targets import make_window
    from forceaware import mpc_loss as fa
    import mujoco,warp as wp
    native=mujoco.MjModel.from_xml_path('outputs/fa_admm_solver_20261016/sequence/scene/wuji_right/sugar_box_mesh.xml')
    cfg=replace(e.cfg,optimizer=replace(e.cfg.optimizer,multistart_count=1,n_iter=40))
    factory=optimizer_class(native,ADMMSettings(optimize_contacts=False))
    original=factory(e.model,e.device_model,e.index,cfg=cfg,guidance_count=e.sequence.contact.mask.shape[1])
    changed=factory(e.model,e.device_model,e.index,cfg=cfg,guidance_count=e.sequence.contact.mask.shape[1])
    root=Path('outputs/fa_matched_ablation_20261016');install(changed,'tracking-time',native,root,sequence=e.sequence)
    saved=np.load('outputs/fa_admm_solver_20261016/comfree/lbfgs_fixed/rollout.npz');out=wp.zeros(1,device=e.device);results=[]
    for dense in (0,240,360):
        q=saved['qpos_traj'][dense];v=saved['qvel_traj'][dense]
        previous=saved['ctrl_traj'][dense-1] if dense else q[:26];pp=saved['ctrl_traj'][dense-4] if dense>=6 else previous
        target=make_window(e.sequence,executed_steps=dense,horizon=5,time_grid=e.cfg.simulator.time_grid,start_frame=0.,end_frame=71.)
        values=[];gradients=[]
        for solver in (original,changed):
            solver.objective.regularizer._previous_previous_cpu=pp.copy()
            solver.backend._prepare_window(q,v,target,warm=saved['raw_warm_history'][dense],previous_action=previous,seed=dense//3,start_time=dense*.01)
            x=solver.control.raw.numpy()[:,26:].astype(float)/solver.variable_scale
            solver.evaluations=0;solver._forward_variables=None;f,g=solver._evaluate(x,True)
            values.append(float(f[0]));gradients.append(g)
        difference=0.
        for k in range(29):
            old=original.loss.immediate[k].job
            wp.launch(constant_tracking,1,inputs=[old,out],device=e.device);after=float(out.numpy()[0])
            state=fa.LossState();state.qpos=original.steps[k].inputs.dynamics.qpos
            job=fa.LossJob();job.state=state;job.model=old.model;job.weights=old.weights;job.target=old.target
            wp.launch(constant_tracking,1,inputs=[job,out],device=e.device);before=float(out.numpy()[0]);difference+=before-after
        error=values[1]-values[0]-difference
        result=dict(dense=dense,original=values[0],changed=values[1],literal_tracking_difference=difference,
            cost_identity_error=error,analytic_gradients_finite=bool(all(np.isfinite(g).all() for g in gradients)))
        assert abs(error)<3e-6,result;assert result['analytic_gradients_finite'],result
        results.append(result);print(json.dumps(result),flush=True)
    (root/'tracking_time_validation.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
