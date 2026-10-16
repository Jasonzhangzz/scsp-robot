"""Independent FA identity/proximal/solver-replacement checks at an active frame."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--forge',type=Path,required=True);args=parser.parse_args()
    from examples.retarget.common.engine import ComFreeEngine
    e=ComFreeEngine(args.experiment,args.forge)
    import mujoco
    from forceaware.targets import make_window
    from forceaware.servo_prior import ServoPriorOptimizer
    from forceaware import optimizer as original
    from examples.retarget.fa.solver import ADMMSettings,optimizer_class
    native=mujoco.MjModel.from_xml_path(str(args.experiment/'sequence/scene/wuji_right/sugar_box_mesh.xml'))
    cfg=replace(e.cfg,optimizer=replace(e.cfg.optimizer,multistart_count=2,n_iter=10))
    common=dict(guidance_count=e.sequence.contact.mask.shape[1],cfg=cfg)
    targets=make_window(e.sequence,executed_steps=100,horizon=5,time_grid=cfg.simulator.time_grid,start_frame=0.,end_frame=71.)
    q=e.model.qpos0.copy();q[e.qadr]=e.sequence.hand[30];q[e.oq:e.oq+7]=targets.object_qpos[0]
    kw=dict(warm=np.zeros(26*6),previous_action=targets.actions[0],seed=0,start_time=1.)
    baseline=ServoPriorOptimizer(e.model,e.device_model,e.index,**common)
    solver=optimizer_class(native,ADMMSettings(optimize_contacts=False))(e.model,e.device_model,e.index,**common)
    for b in [baseline,solver.backend]:
        b._prepare_window(q,np.zeros(e.model.nv),targets,**kw)
    baseline._forward(allow_init=False);baseline.objective.evaluate(baseline.control,allow_init=False)
    baseline._backward(allow_init=False)
    truth=baseline.loss.total.numpy().astype(float);gradient=baseline.control.gradient.numpy()[:,26:].astype(float)
    variables=solver.backend.control.raw.numpy()[:,26:].astype(float)/solver.variable_scale
    solver.evaluations=0;value,grad=solver._evaluate(variables,True)
    relative=float(np.max(np.abs(value*1e6-truth)/np.maximum(np.abs(truth),1.)))
    grad_error=float(np.max(np.abs(grad/solver.variable_scale-gradient))/max(np.max(np.abs(gradient)),1.))
    assert relative<2e-6,(relative,truth,value)
    assert grad_error<2e-5,grad_error
    workspace=solver.loss.terminal;points=solver.objective.points[-1];job=workspace.job
    bp=job.state.body_position.numpy();bm=job.state.body_matrix.numpy();body=job.target.query_body.numpy()
    local=job.target.query_local.numpy();obj=e.index.object_body
    h=bp[:,body]+np.einsum('wnij,nj->wni',bm[:,body],local)
    x=np.einsum('wni,wij->wnj',h-bp[:,obj,None],bm[:,obj])
    c0=points.c.numpy().copy();z=points.z.numpy();dual=points.dual.numpy()
    mask=job.target.query_mask.numpy()*job.target.query_weight.numpy()
    marked=job.auxiliary.marked_query_weight.numpy()
    target=bp[:,obj,None]+np.einsum('wij,wnj->wni',bm[:,obj],c0)
    expected=2.*job.weights.marked_query*mask[None,:,None]*marked[:,:,None]*np.einsum('wni,wij->wnj',target-h,bm[:,obj])
    point_gradient_error=float(np.max(np.abs(expected-points.c.grad.numpy())))
    assert point_gradient_error<1e-5,point_gradient_error
    solver.objective.update_contacts();c=points.c.numpy();v=z-dual
    from scipy.optimize import minimize_scalar
    prox_errors=[]
    for world,slot in zip(*np.nonzero(np.broadcast_to(mask>0,(2,len(mask))))):
        distance=float(np.linalg.norm(x[world,slot]-v[world,slot])*1000)
        def f(r):
            sigmoid=1/(1+np.exp(-np.clip((np.sqrt(r*r+1e-12)-10),-700,700)))
            return sigmoid*r*r+.5*(distance-r)**2
        grid=np.linspace(0,distance,10001);idx=int(np.argmin([f(t) for t in grid]))
        lo=grid[max(idx-1,0)];hi=grid[min(idx+1,len(grid)-1)]
        optimum=minimize_scalar(f,bounds=(lo,hi),method='bounded',options={'xatol':1e-10})
        r=float(np.linalg.norm(c[world,slot]-x[world,slot])*1000)
        prox_errors.append(abs(f(r)-optimum.fun)/max(optimum.fun,1.))
    assert max(prox_errors)<2e-4,max(prox_errors)
    # Explicit guards make a silent fall back to the original FA optimizer fail.
    def forbidden(*a,**k):raise AssertionError('Original FA solve/Adam was called')
    old_solve=original.WindowOptimizer.solve;old_adam=original._adam_update
    original.WindowOptimizer.solve=forbidden;original._adam_update=forbidden
    try:
        active_solver=optimizer_class(native,ADMMSettings())(e.model,e.device_model,e.index,**common)
        plan=active_solver.solve(q,np.zeros(e.model.nv),targets,**kw,allow_init=False,execution_steps=3)
    finally:
        original.WindowOptimizer.solve=old_solve;original._adam_update=old_adam
    snap=active_solver.objective.snapshot(plan.diag['selected']);active=snap['mask']>0
    final_fa=float(active_solver.loss.total.numpy()[plan.diag['selected']])
    final_score_error=abs(final_fa-plan.loss)/max(abs(final_fa),1.)
    assert final_score_error<2e-6,final_score_error
    assert np.array_equal(snap['c'],snap['z'])
    shift=np.linalg.norm(snap['z']-snap['template'],axis=-1)[active]*1000
    result=dict(frame=30,FA_formula_value_max_relative_error=relative,
        FA_hand_gradient_max_relative_error=grad_error,contact_point_gradient_max_absolute_error=point_gradient_error,
        contact_prox_global_grid_comparison_max_relative_error=max(prox_errors),
        original_FA_solve_and_Adam_forbidden_check_passed=True,final_score_pure_FA_relative_error=final_score_error,
        MANO_role='initialization_only',active_contact_shift_mean_mm=float(shift.mean()),
        solver_diagnostics=active_solver.last_solver_diag)
    (args.experiment/'admm_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
