"""Analytic consistency checks for the real ComFree OCP external functions."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs/fa_direct_mpc_20261016/validation.json'))
    p.add_argument('--rollout-dir',type=Path)
    a=p.parse_args()
    from examples.retarget.matrix_comfree import ComFreeEngine
    from examples.retarget.fa_direct_mpc import DirectComFreeMPC
    root=Path('outputs/fa_admm_solver_20261016')
    e=ComFreeEngine(root,Path('thirdparty/retarget_assets/DexForge'));s=DirectComFreeMPC(e)
    from forceaware.targets import make_window
    from forceaware import mpc_loss as fa
    import warp as wp
    saved=np.load(root/'mujoco/force_contact_mujoco_verified/rollout.npz')
    q=saved['qpos_traj'][90];v=saved['qvel_traj'][90];u=saved['ctrl_traj'][90];x=np.r_[q,v]
    targets=make_window(e.sequence,executed_steps=90,horizon=5,time_grid=e.cfg.simulator.time_grid,start_frame=0.,end_frame=71.)
    n=s.native;n.prepare(targets)
    f,J=n.physics(x,u,True)
    q1,v1,_=e.step(q,v,u,dt=.01)
    forward_error=float(np.max(np.abs(f[:n.nx]-np.r_[q1[0],v1[0]])))
    assert forward_error<2e-5,forward_error
    external=s.symbolic_physics.jacobian()(x,u,5,f)
    bridge_error=float(max(np.max(np.abs(np.asarray(external[0])-J[:,:n.nx])),np.max(np.abs(np.asarray(external[1])-J[:,n.nx:]))))
    assert bridge_error==0.,bridge_error
    # Independently seed a weighted output combination. This checks the full
    # assembled matrix against one native VJP, without any finite differences.
    rng=np.random.default_rng(731);weight=rng.normal(size=n.nout);b=n._dynamics_buffer(True)
    for name,section in [('qpos',slice(0,n.nq)),('qvel',slice(n.nq,n.nx)),('qacc',slice(n.nx,n.nout))]:
        seed=np.zeros(getattr(b['seed'],name).shape,np.float32);seed[0]=weight[section]
        getattr(b['seed'],name).assign(seed)
    n.cache.clear();_,mixed=n.physics(x,u,True)
    expected=weight@J
    vjp_error=float(np.max(np.abs(mixed[0]-expected))/max(np.max(np.abs(expected)),1.))
    assert vjp_error<2e-5,(vjp_error,float(np.linalg.norm(expected)),float(np.linalg.norm(mixed[0])),weight.shape,J.shape)
    from examples.retarget.comfree_ocp_native import seed_outputs
    wp.launch(seed_outputs,(n.nout,n.nout),inputs=[b['seed'].qpos,b['seed'].qvel,b['seed'].qacc],device=e.device)
    n.cache.clear()
    cost_errors=[];gradient_errors=[];tracking_errors=[]
    for stage in (5,20):
        value,g=n.cost(q,stage,True);r,R=n.residual(q,stage,True)
        cost_errors.append(float(abs(r@r-value)/max(abs(value),1e-8)))
        gradient_errors.append(float(np.max(np.abs(2*r@R-g))/max(np.max(np.abs(g)),1.)))
        cb=n._cost_buffer(stage);cb['loss'].job.reference_scale=1.
        fa.evaluate(cb['loss']);original=float(cb['loss'].loss.numpy()[0])*1e-6
        parameter=np.r_[stage,targets.dense_actions[stage],targets.object_qpos[stage],1.]
        scaled=np.r_[x,u,u]/s.xscale
        residual=np.asarray(s.geometry_function(scaled,parameter)).ravel()
        tracking_errors.append(float(abs(residual@residual-original)/max(abs(original),1e-8)))
        cb['loss'].job.reference_scale=0.
    assert max(cost_errors)<2e-6,cost_errors
    assert max(gradient_errors)<2e-5,gradient_errors
    assert max(tracking_errors)<2e-5,tracking_errors
    f0,_=n.physics(x,u)
    points=targets.contact_position_object.copy();points[...,0]+=.005
    changed=replace(targets,contact_position_object=points)
    old_r,_=n.residual(q,5);n.prepare(changed)
    f2,_=n.physics(x,u);new_r,_=n.residual(q,5)
    assert np.array_equal(f0,f2),'Template target changed physical dynamics'
    response=float(np.linalg.norm(old_r-new_r));assert response>1e-6,response
    result=dict(native_forward_max_error=forward_error,external_dynamics_Jacobian_max_error=bridge_error,
        analytic_matrix_vs_independent_VJP_relative_error=vjp_error,
        FA_residual_value_relative_error=max(cost_errors),FA_residual_gradient_relative_error=max(gradient_errors),
        full_FA_tracking_geometry_value_relative_error=max(tracking_errors),
        template_change_dynamics_bitwise_equal=True,template_change_cost_response=response,
        dynamics_contact_frame_derivatives=True,finite_differences_used=False)
    if a.rollout_dir:
        if (a.rollout_dir/'INVALIDATED.json').exists():raise ValueError('Cannot certify an invalidated rollout')
        rows=[json.loads(line) for line in (a.rollout_dir/'windows.jsonl').read_text().splitlines()]
        accepted=0;refresh=[]
        for i,row in enumerate(rows):
            costs=row.get('true_FA_costs',[])
            assert np.all(np.diff(costs)<=0.),(i,costs)
            accepted+=row.get('accepted_SQP_steps',0)
            update=row.get('template_update',{})
            if update.get('updated'):
                assert i%update['period_windows']==0,(i,update)
                assert update['merit_improvement']>0
                assert update['anchor_weight']>0
                refresh.append(i)
        result.update(checked_closed_loop_windows=len(rows),accepted_SQP_steps=accepted,
            true_FA_accepted_steps_nonincreasing=True,template_update_windows=refresh,
            low_frequency_soft_anchored_updates=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
