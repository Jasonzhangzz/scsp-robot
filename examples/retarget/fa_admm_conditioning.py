"""Same frozen state: check physical-unit scaling of the hand subproblem."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--forge',type=Path,required=True);a=p.parse_args()
    from examples.retarget.matrix_comfree import ComFreeEngine
    e=ComFreeEngine(a.experiment,a.forge)
    import mujoco
    from forceaware.targets import make_window
    from forceaware.servo_prior import ServoPriorOptimizer
    from examples.retarget.fa_contact_admm import ADMMSettings,optimizer_class
    native=mujoco.MjModel.from_xml_path(str(a.experiment/'sequence/scene/wuji_right/sugar_box_mesh.xml'))
    path=a.experiment/'comfree/fa_fixed/rollout.npz'
    if not path.exists():path=path.with_name('partial.npz')
    saved=np.load(path);i=90
    cfg=replace(e.cfg,optimizer=replace(e.cfg.optimizer,multistart_count=2,n_iter=40))
    target=make_window(e.sequence,executed_steps=i,horizon=5,time_grid=cfg.simulator.time_grid,start_frame=0.,end_frame=71.)
    specs=[('FA_Adam',None),('fixed_uniform',ADMMSettings(control_scale=.003,balanced_coordinates=False,optimize_contacts=False)),
           ('fixed_balanced',ADMMSettings(optimize_contacts=False)),
           ('ADMM_uniform',ADMMSettings(control_scale=.003,balanced_coordinates=False)),('ADMM_balanced',ADMMSettings())]
    results=[]
    for name,settings in specs:
        cls=ServoPriorOptimizer if settings is None else optimizer_class(native,settings)
        solver=cls(e.model,e.device_model,e.index,guidance_count=e.sequence.contact.mask.shape[1],cfg=cfg)
        solver.objective.regularizer._previous_previous_cpu=saved['ctrl_traj'][i-4].copy()
        plan=solver.solve(saved['qpos_traj'][i],saved['qvel_traj'][i],target,
            warm=saved['raw_warm_history'][i],previous_action=saved['ctrl_traj'][i-1],
            allow_init=False,seed=i//3,execution_steps=3,start_time=i*.01)
        terminal=solver.steps[-1].recorded.result.qpos.numpy()[plan.diag['selected'],e.oq:e.oq+7]
        record=dict(variant=name,time_s=i*.01,FA_loss=plan.loss,seconds=plan.seconds,
            next_position_error_mm=float(np.linalg.norm(plan.next_qpos[e.oq:e.oq+3]-target.object_qpos[2,:3])*1000),
            terminal_position_error_mm=float(np.linalg.norm(terminal[:3]-target.object_qpos[-1,:3])*1000))
        results.append(record);print(json.dumps(record),flush=True)
        (a.experiment/'conditioning_probe.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
