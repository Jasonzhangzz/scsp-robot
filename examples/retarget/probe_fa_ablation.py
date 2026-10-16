"""One-factor optimizations from identical states/warm starts of successful FA."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--factor',required=True,choices=['baseline','native-cache','baseline-contiguous','budget3','budget3-contiguous','budget8-contiguous','single-start','slsqp','slsqp-contiguous','acados','acados-contiguous',
        'absolute-warm','full-contact-jacobian','dense-smoothing','control-bounds','slow-templates','dense-controls','tracking-time'])
    p.add_argument('--windows',type=int,nargs='+',default=[0,30,80,120,150])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'probes.jsonl').exists():raise FileExistsError(a.output/'probes.jsonl')
    from examples.retarget.matrix_comfree import ComFreeEngine
    e=ComFreeEngine(Path('outputs/fa_admm_solver_20261016'),Path('thirdparty/retarget_assets/DexForge'))
    from examples.retarget.fa_contact_admm import ADMMSettings,optimizer_class
    from examples.retarget.fa_matched_ablation import install,expand
    from forceaware.targets import make_window
    iterations=3 if a.factor.startswith('budget3') else 8 if a.factor.startswith('budget8') else 40
    cfg=replace(e.cfg,optimizer=replace(e.cfg.optimizer,n_iter=iterations,multistart_count=1 if a.factor=='single-start' else 8))
    if a.factor=='dense-controls':cfg=replace(cfg,optimizer=replace(cfg.optimizer,horizon=30),simulator=replace(cfg.simulator,
        time_grid=replace(cfg.simulator.time_grid,knot_dt=.01,knot_substeps=1)))
    import mujoco
    native=mujoco.MjModel.from_xml_path('outputs/fa_admm_solver_20261016/sequence/scene/wuji_right/sugar_box_mesh.xml')
    algorithm=a.factor.split('-')[0]
    settings=ADMMSettings(optimize_contacts=False,hand_solver=algorithm if algorithm in ('slsqp','acados') else 'lbfgs',hand_rti_iterations=40,
        fixed_hand_single_block=a.factor.endswith('-contiguous'),memoize_evaluations=a.factor=='native-cache')
    solver=optimizer_class(native,settings)(e.model,e.device_model,e.index,cfg=cfg,guidance_count=e.sequence.contact.mask.shape[1])
    if a.factor not in ('baseline','native-cache','single-start') and not a.factor.startswith(('budget','slsqp','acados','baseline-')):
        install(solver,a.factor,native,a.output,sequence=e.sequence)
    reference=np.load('outputs/fa_admm_solver_20261016/comfree/lbfgs_fixed/rollout.npz');original=solver._evaluate;trace=[];capture={}
    def evaluate(x,gradient):
        f,g=original(x,gradient)
        if not trace or (gradient and not any(t['gradient'] for t in trace)):
            trace.append(dict(gradient=gradient,values=f.tolist(),gradient_norm=np.linalg.norm(g,axis=1).tolist() if gradient else None))
            if gradient:capture.update(initial_variables=x.copy(),initial_costs=f.copy(),initial_gradients=g.copy())
        return f,g
    solver._evaluate=evaluate
    for window in a.windows:
        dense=window*3;q=reference['qpos_traj'][dense];v=reference['qvel_traj'][dense]
        previous=reference['ctrl_traj'][dense-1] if dense else q[:26]
        pp=reference['ctrl_traj'][dense-4] if dense>=6 else reference['qpos_traj'][0,:26]
        solver.objective.regularizer._previous_previous_cpu=pp.copy()
        warm=reference['raw_warm_history'][dense].copy()
        target=make_window(e.sequence,executed_steps=dense,horizon=cfg.optimizer.horizon,time_grid=cfg.simulator.time_grid,start_frame=0.,end_frame=71.)
        if a.factor=='dense-controls':warm=np.r_[warm[:26],expand(warm[26:].reshape(5,26),np.zeros(26)).ravel()].astype(np.float32)
        if a.factor=='absolute-warm' and dense:
            old_target=make_window(e.sequence,executed_steps=dense-3,horizon=5,time_grid=e.cfg.simulator.time_grid,start_frame=0.,end_frame=71.)
            shifted=warm[26:].reshape(5,26);old_delta=shifted.copy()
            for k in range(3,-1,-1):old_delta[k]=2*shifted[k]-old_delta[k+1]
            solver._ablation_state['absolute']=expand(old_target.actions+old_delta,pp)
        if a.factor=='dense-smoothing':solver._ablation_state['previous_previous']=reference['ctrl_traj'][dense-2] if dense>=3 else previous
        if a.factor=='slow-templates':
            template=solver._ablation_state['templates'];template.memory.clear();template.window=window
        trace.clear();capture.clear();started=perf_counter()
        plan=solver.solve(q,v,target,warm=warm,previous_action=previous,allow_init=False,seed=window,execution_steps=3,start_time=dense*.01)
        selected=plan.diag['selected'];predq=solver.steps[-1].recorded.result.qpos.numpy()[selected]
        error=1000*np.linalg.norm(predq[e.oq:e.oq+3]-target.object_qpos[-1,:3])
        # Same physical first three commands, to inspect immediate corrective action.
        q1,v1,_=e.step(q,v,plan.control_sequence[0],dt=.01)
        row=dict(factor=a.factor,window=window,time=dense*.01,seconds=perf_counter()-started,solver_seconds=plan.seconds,
            final_FA_cost=plan.loss*1e-6,terminal_position_error_mm=float(error),terminal_height_mm=float(1000*predq[e.oq+2]),
            selected=int(selected),first_three_controls=plan.control_sequence.tolist(),initial_evaluations=list(trace),solver=solver.last_solver_diag)
        np.savez_compressed(a.output/f'window_{window:04d}.npz',raw=plan.raw,controls=plan.control_sequence,
            predicted_qpos=np.stack([s.recorded.result.qpos.numpy()[selected] for s in solver.steps]),**capture)
        with (a.output/'probes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ('first_three_controls','initial_evaluations','solver')}),flush=True)


if __name__=='__main__':main()
