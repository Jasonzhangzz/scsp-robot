"""Compare generated FA equations/cost/analytic control gradients to native FA.

Uses common successful-FA snapshots and identical raw candidates and history.
No finite differences, no change to the physical executor or optimizer.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/fa_compiled_evaluator_20261016'))
    p.add_argument('--windows',nargs='+',type=int,default=[0,80,120]);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--solve',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    from examples.retarget.matrix_comfree import ComFreeEngine
    root=Path('outputs/fa_admm_solver_20261016');e=ComFreeEngine(root,Path('thirdparty/retarget_assets/DexForge'))
    from examples.retarget.fa_contact_admm import ADMMSettings,optimizer_class
    from examples.retarget.fa_compiled_evaluator import CompiledFAEvaluator
    from forceaware.targets import make_window
    import mujoco
    native=mujoco.MjModel.from_xml_path(str(root/'sequence/scene/wuji_right/sugar_box_mesh.xml'))
    cfg=replace(e.cfg,optimizer=replace(e.cfg.optimizer,multistart_count=8,n_iter=40))
    s=optimizer_class(native,ADMMSettings(optimize_contacts=False))(e.model,e.device_model,e.index,cfg=cfg,guidance_count=e.sequence.contact.mask.shape[1])
    ce=CompiledFAEvaluator(s.backend,a.output/'cache',threads=a.threads)
    reference=np.load(root/'comfree/lbfgs_fixed/rollout.npz');results=[]
    for window in a.windows:
        dense=window*3;q=reference['qpos_traj'][dense];v=reference['qvel_traj'][dense]
        prev=reference['ctrl_traj'][dense-1] if dense else q[:26]
        pp=reference['ctrl_traj'][dense-4] if dense>=6 else reference['qpos_traj'][0,:26]
        s.objective.regularizer._previous_previous_cpu=pp.copy();warm=reference['raw_warm_history'][dense]
        target=make_window(e.sequence,executed_steps=dense,horizon=cfg.optimizer.horizon,time_grid=cfg.simulator.time_grid,start_frame=0.,end_frame=71.)
        kw=dict(warm=warm,previous_action=prev,allow_init=False,seed=window,execution_steps=3,start_time=dense*.01)
        s.backend._prepare_window(q,v,target,**{k:w for k,w in kw.items() if k not in ('allow_init','execution_steps')})
        raw=s.control.raw.numpy()[:,26:];ce.prepare()
        # Capture before reading times: graph/JIT initialization is excluded.
        s.evaluations=0;s.forward_calls=0;s.backward_calls=0;s.backward_reuses=0;s.value_reuses=0;s._forward_variables=None
        values,grads=s._evaluate(raw/s.variable_scale,True)
        truth_states=np.stack([np.c_[st.recorded.result.qpos.numpy(),st.recorded.result.qvel.numpy()] for st in s.steps],axis=1)
        cv,cg=ce.evaluate_raw(raw,True)
        state_error=ce.last['states'][:,1:]-truth_states
        scalar_scale=np.maximum(np.max(np.abs(grads),axis=1),1e-10)
        diff=cg*s.variable_scale-grads
        row=dict(window=window,time=dense*.01,build_seconds=ce.build_seconds,
            cost_max_relative=float(np.max(np.abs(cv-values)/np.maximum(np.abs(values),1e-8))),
            cost_native=values.tolist(),cost_compiled=cv.tolist(),
            gradient_max_absolute=float(np.max(np.abs(diff))),
            gradient_relative_per_world=(np.max(np.abs(diff),axis=1)/scalar_scale).tolist(),
            qpos_max_absolute=float(np.max(np.abs(state_error[:,:,:e.model.nq]))),
            qvel_max_absolute=float(np.max(np.abs(state_error[:,:,e.model.nq:]))))
        np.savez_compressed(a.output/f'compare_{window:04d}.npz',raw=raw,native_values=values,compiled_values=cv,
            native_gradient=grads,compiled_gradient=cg*s.variable_scale,native_states=truth_states,compiled_states=ce.last['states'][:,1:])
        def elapsed(f):
            start=perf_counter();f();return perf_counter()-start
        # Restore the same input for each repeat; native reuse is kept as an
        # explicit separate measurement rather than silently changing work.
        row['compiled_value_s']=[elapsed(lambda:ce.evaluate_raw(raw,False)) for _ in range(3)]
        def compiled_gradient():ce.last=None;ce.evaluate_raw(raw,True)
        row['compiled_value_gradient_s']=[elapsed(compiled_gradient) for _ in range(3)]
        row['compiled_backward_reuse_s']=[elapsed(lambda:ce.evaluate_raw(raw,True)) for _ in range(3)]
        def native_value():s._forward_variables=None;s._evaluate(raw/s.variable_scale,False)
        def native_gradient():s._forward_variables=None;s._evaluate(raw/s.variable_scale,True)
        native_value()
        row['native_value_s']=[elapsed(native_value) for _ in range(3)]
        row['native_value_gradient_s']=[elapsed(native_gradient) for _ in range(3)]
        if a.solve:
            result=s.solve(q,v,target,**kw);row['native_solve_s']=result.seconds
            compiled_settings=ADMMSettings(optimize_contacts=False,evaluation_backend='compiled',compiled_cache=str(a.output/'cache'),compiled_threads=a.threads)
            other=optimizer_class(native,compiled_settings)(e.model,e.device_model,e.index,cfg=cfg,guidance_count=e.sequence.contact.mask.shape[1])
            other.objective.regularizer._previous_previous_cpu=pp.copy()
            result2=other.solve(q,v,target,**kw);row['compiled_solve_s']=result2.seconds
            row['native_solve_cost']=result.loss*1e-6;row['compiled_solve_cost']=result2.loss*1e-6
            row['control_max_absolute']=float(np.max(np.abs(result.control_sequence-result2.control_sequence)))
            row['compiled_solver']=other.last_solver_diag
        results.append(row);(a.output/'validation.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)


if __name__=='__main__':main()
