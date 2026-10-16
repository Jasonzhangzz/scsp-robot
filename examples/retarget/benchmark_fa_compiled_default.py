"""Matched warm full-MPC wall time for the entry point's 2x20 configuration."""
from pathlib import Path
from dataclasses import replace
import argparse
import json
from time import perf_counter
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    from examples.retarget.matrix_comfree import ComFreeEngine
    root=Path('outputs/fa_admm_solver_20261016');e=ComFreeEngine(root,Path('thirdparty/retarget_assets/DexForge'))
    from examples.retarget.fa_contact_admm import ADMMSettings,optimizer_class
    from examples.retarget.comfree_contact_search import write_object_scene
    from forceaware.targets import make_window
    import mujoco
    native=mujoco.MjModel.from_xml_path(str(root/'sequence/scene/wuji_right/sugar_box_mesh.xml'))
    scene=write_object_scene(e.cfg.sequence.xml,a.output/'object_contact_scene.xml')
    cfg=replace(e.cfg,optimizer=replace(e.cfg.optimizer,multistart_count=2,n_iter=20))
    reference=np.load(root/'comfree/lbfgs_fixed/rollout.npz');dense=360
    q=reference['qpos_traj'][dense];v=reference['qvel_traj'][dense];warm=reference['raw_warm_history'][dense]
    prev=reference['ctrl_traj'][dense-1];pp=reference['ctrl_traj'][dense-4]
    target=make_window(e.sequence,executed_steps=dense,horizon=5,time_grid=cfg.simulator.time_grid,start_frame=0.,end_frame=71.)
    request=dict(warm=warm,previous_action=prev,seed=120,start_time=3.6)
    results=[]
    for backend,memo in [('native',False),('native',True),('compiled',True)]:
        settings=ADMMSettings(outer=2,optimize_contacts=True,contact_mode='comfree-force',object_scene=str(scene),
            contact_solver='batch-lm',evaluation_backend=backend,memoize_evaluations=memo,
            compiled_cache='outputs/fa_compiled_evaluator_20261016/cache')
        solver=optimizer_class(native,settings)(e.model,e.device_model,e.index,cfg=cfg,guidance_count=e.sequence.contact.mask.shape[1])
        # Capture native forward/backward/materialization and force-LM graphs
        # without committing warm starts, contact memory or action history.
        solver.objective.regularizer._previous_previous_cpu=pp.copy()
        solver.backend._prepare_window(q,v,target,**request)
        solver.evaluations=solver.forward_calls=solver.backward_calls=solver.backward_reuses=solver.value_reuses=0
        solver.native_materializations=0;solver._forward_variables=None
        if solver.compiled_evaluator is not None:solver.compiled_evaluator.prepare()
        raw=solver.control.raw.numpy();raw[:,:26]=0.;solver.control.raw.assign(raw)
        variables=raw[:,26:].astype(float)/solver.variable_scale
        solver._evaluate(variables,False);solver._evaluate(variables,True)
        solver._forward_variables=None;solver._evaluate(variables,True)
        solver._materialize_native()
        solver.objective.contact_search.begin(q,v,target,120,3);solver.objective.contact_search.update()
        for repeat in range(a.repeats):
            solver.objective.contact_memory=None;solver.objective.force_memory=None
            solver.objective.regularizer._previous_previous_cpu=pp.copy()
            started=perf_counter();plan=solver.solve(q,v,target,**request,allow_init=False,execution_steps=3)
            row=dict(backend=backend,memo=memo,repeat=repeat,wall_s=perf_counter()-started,solver_s=plan.seconds,
                cost=plan.loss*1e-6,selected=plan.diag['selected'],solver=solver.last_solver_diag)
            results.append(row);(a.output/'timings.json').write_text(json.dumps(results,indent=2))
            np.savez_compressed(a.output/f'{backend}_{int(memo)}_{repeat}.npz',raw=plan.raw,controls=plan.control_sequence,
                qpos=plan.qpos_sequence,qvel=plan.qvel_sequence)
            print(json.dumps({k:v for k,v in row.items() if k!='solver'}),flush=True)


if __name__=='__main__':main()
