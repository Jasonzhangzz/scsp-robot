"""Check value and analytic-gradient equivalence under the dense control embedding."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np


def main():
    from examples.retarget.matrix_comfree import ComFreeEngine
    e=ComFreeEngine(Path('outputs/fa_admm_solver_20261016'),Path('thirdparty/retarget_assets/DexForge'))
    from examples.retarget.fa_contact_admm import ADMMSettings,optimizer_class
    from examples.retarget.fa_matched_ablation import install,expand,shift_dense_matched
    from forceaware.control import resample_warm_corrections
    from forceaware.targets import make_window
    import mujoco
    native=mujoco.MjModel.from_xml_path('outputs/fa_admm_solver_20261016/sequence/scene/wuji_right/sugar_box_mesh.xml')
    cfg=replace(e.cfg,optimizer=replace(e.cfg.optimizer,multistart_count=1,n_iter=40))
    dense_cfg=replace(cfg,optimizer=replace(cfg.optimizer,horizon=30),simulator=replace(cfg.simulator,time_grid=replace(cfg.simulator.time_grid,knot_dt=.01,knot_substeps=1)))
    factory=optimizer_class(native,ADMMSettings(optimize_contacts=False))
    first=factory(e.model,e.device_model,e.index,cfg=cfg,guidance_count=e.sequence.contact.mask.shape[1])
    second=factory(e.model,e.device_model,e.index,cfg=dense_cfg,guidance_count=e.sequence.contact.mask.shape[1])
    output=Path('outputs/fa_matched_ablation_20261016');output.mkdir(exist_ok=True)
    install(second,'dense-controls',native,output,sequence=e.sequence)
    saved=np.load('outputs/fa_admm_solver_20261016/comfree/lbfgs_fixed/rollout.npz');results=[]
    P=expand(np.eye(5).T[...,None],np.zeros((5,1)))[:,:,0].T
    for dense in (0,240,360):
        q=saved['qpos_traj'][dense];v=saved['qvel_traj'][dense];previous=saved['ctrl_traj'][dense-1] if dense else q[:26]
        pp=saved['ctrl_traj'][dense-4] if dense>=6 else previous;warm=saved['raw_warm_history'][dense]
        values=[];gradients=[];positions=[];control_sequences=[]
        for solver,c in [(first,cfg),(second,dense_cfg)]:
            solver.objective.regularizer._previous_previous_cpu=pp.copy()
            target=make_window(e.sequence,executed_steps=dense,horizon=c.optimizer.horizon,time_grid=c.simulator.time_grid,start_frame=0.,end_frame=71.)
            w=warm if c.optimizer.horizon==5 else np.r_[warm[:26],expand(warm[26:].reshape(5,26),np.zeros(26)).ravel()].astype(np.float32)
            solver.backend._prepare_window(q,v,target,warm=w,previous_action=previous,seed=dense//3,start_time=dense*.01)
            x=solver.control.raw.numpy()[:,26:].astype(float)/solver.variable_scale
            solver.evaluations=0;solver._forward_variables=None
            f,g=solver._evaluate(x,True);values.append(f[0]);gradients.append(g[0])
            positions.append(solver.steps[-1].recorded.result.qpos.numpy()[0]);control_sequences.append(np.stack([u.numpy()[0] for u in solver.control.controls]))
        pulled=(P.T@gradients[1].reshape(30,26)).ravel()
        row=dict(dense=dense,value_relative_error=float(abs(values[0]-values[1])/max(abs(values[0]),1e-8)),
            analytic_gradient_relative_error=float(np.linalg.norm(pulled-gradients[0])/max(np.linalg.norm(gradients[0]),1e-12)),
            terminal_qpos_max_error=float(np.max(abs(positions[0]-positions[1]))),
            dense_control_max_error=float(np.max(abs(control_sequences[0]-control_sequences[1]))),values=values)
        warm_errors=[]
        embedded=np.r_[warm[:26],expand(warm[26:].reshape(5,26),np.zeros(26)).ravel()].astype(np.float32)
        for execution_steps in (1,2,3,6):
            coarse_shift=resample_warm_corrections(warm,action_dim=26,horizon=5,executed_dense_steps=execution_steps,
                physics_steps_per_knot=6,warm_scale=cfg.optimizer.warm_shift_scale)
            expected=np.r_[coarse_shift[:26],expand(coarse_shift[26:].reshape(5,26),np.zeros(26)).ravel()].astype(np.float32)
            actual=shift_dense_matched(embedded,26,execution_steps=execution_steps,warm_scale=cfg.optimizer.warm_shift_scale)
            warm_errors.append(float(np.max(abs(actual-expected))))
        row['warm_shift_embedding_max_error']=max(warm_errors)
        assert row['warm_shift_embedding_max_error']<1e-7,row
        results.append(row);print(json.dumps(row),flush=True)
        assert row['dense_control_max_error']<3e-6,row
        assert row['value_relative_error']<.002,row
        assert row['analytic_gradient_relative_error']<.02,row
    (output/'embedding_validation.json').write_text(json.dumps(dict(checks=results,finite_difference_jacobians=False),indent=2)+'\n')


if __name__=='__main__':main()
