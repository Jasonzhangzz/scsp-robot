"""Active-contact checks for force shooting, unchanged FA and contact memory."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import mujoco


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,default=Path('outputs/fa_admm_solver_20261016'))
    p.add_argument('--forge',type=Path,default=Path('thirdparty/retarget_assets/DexForge'))
    p.add_argument('--contact-solver',choices=['shooting','batch-lm'],default='shooting')
    p.add_argument('--hand-solver',choices=['lbfgs','slsqp','acados'],default='lbfgs')
    p.add_argument('--output',type=Path,help='Separate validation output directory')
    a=p.parse_args();root=a.experiment
    output=a.output or root;output.mkdir(parents=True,exist_ok=True)
    from examples.retarget.matrix_comfree import ComFreeEngine
    engine=ComFreeEngine(root,a.forge)
    from examples.retarget.fa_contact_admm import ADMMSettings,optimizer_class
    from examples.retarget.comfree_contact_search import write_object_scene
    from forceaware.targets import make_window
    from forceaware.servo_prior import ServoPriorOptimizer
    from forceaware import optimizer as original
    native=mujoco.MjModel.from_xml_path(str(root/'sequence/scene/wuji_right/sugar_box_mesh.xml'))
    scene=write_object_scene(root/'sequence/scene/wuji_right/sugar_box_sdf.xml',output/'validation_object_scene.xml')
    cfg=replace(engine.cfg,optimizer=replace(engine.cfg.optimizer,multistart_count=2,n_iter=10))
    settings=ADMMSettings(contact_mode='comfree-force',object_scene=str(scene),outer=2,
        contact_samples=5,contact_force_iterations=2,contact_solver=a.contact_solver,hand_solver=a.hand_solver)
    solver=optimizer_class(native,settings)(engine.model,engine.device_model,engine.index,
        guidance_count=engine.sequence.contact.mask.shape[1],cfg=cfg)
    saved=np.load(root/'comfree/admm/rollout.npz')
    dense=90;q=saved['qpos_traj'][dense].astype(float);v=saved['qvel_traj'][dense].astype(float)
    targets=make_window(engine.sequence,executed_steps=dense,horizon=5,
        time_grid=cfg.simulator.time_grid,start_frame=0.,end_frame=71.)
    kw=dict(warm=saved['raw_warm_history'][30],previous_action=saved['ctrl_traj'][dense-1],seed=30,start_time=.9)
    # Compare hand objective at exactly the same (unprojected) C and raw U.
    baseline=ServoPriorOptimizer(engine.model,engine.device_model,engine.index,
        guidance_count=engine.sequence.contact.mask.shape[1],cfg=cfg)
    for backend in (baseline,solver.backend):backend._prepare_window(q,v,targets,**kw)
    for workspace,points in zip((*solver.loss.immediate,solver.loss.terminal),solver.objective.points):
        initial=workspace.target.query_position_object.numpy()
        points.c.assign(np.broadcast_to(initial,points.c.shape+(3,)).copy())
    baseline._forward(allow_init=False);baseline.objective.evaluate(baseline.control,allow_init=False);baseline._backward(allow_init=False)
    raw=solver.control.raw.numpy()[:,26:].astype(float)/solver.variable_scale
    solver.evaluations=0;value,gradient=solver._evaluate(raw,True)
    truth=baseline.loss.total.numpy().astype(float)
    value_error=float(np.max(np.abs(value*1e6-truth)/np.maximum(np.abs(truth),1.)))
    expected=baseline.control.gradient.numpy()[:,26:]
    gradient_error=float(np.max(np.abs(gradient/solver.variable_scale-expected))/max(np.max(np.abs(expected)),1.))
    assert value_error<2e-6 and gradient_error<2e-5,(value_error,gradient_error)
    cached_value,cached_gradient=solver._evaluate(raw,True)
    cached_error=float(np.max(np.abs(cached_gradient-gradient))/max(np.max(np.abs(gradient)),1.))
    assert cached_error<2e-6 and np.array_equal(value,cached_value),(cached_error,value,cached_value)
    cached_value_only,_=solver._evaluate(raw,False)
    assert np.array_equal(value,cached_value_only)
    # Prove the independent solver does not silently call the old FA Adam.
    old_solve,old_adam=original.WindowOptimizer.solve,original._adam_update
    def forbidden(*args,**kwargs):raise AssertionError('Original FA Adam was called')
    original.WindowOptimizer.solve=forbidden;original._adam_update=forbidden
    try:plan=solver.solve(q,v,targets,**kw,allow_init=False,execution_steps=3)
    finally:original.WindowOptimizer.solve=old_solve;original._adam_update=old_adam
    selected=plan.diag['selected'];snap=solver.objective.snapshot(selected);search=solver.objective.contact_search
    np.savez_compressed(output/'force_validation_snapshot.npz',**snap)
    last=search.last
    assert np.all(last['object_cost']<=last['initial_cost']+1e-5)
    assert np.max(np.abs(snap['search_forces']))>1e-4
    coeff=snap['search_coefficients'];force=snap['search_forces'];normal=snap['search_normals']
    fn=np.sum(force*(-normal),axis=-1);ft=force-fn[...,None]*(-normal)
    assert coeff.min()>=-1e-8 and np.max(coeff.sum(axis=-1))<=settings.contact_force_cap+1e-6
    assert np.all(np.linalg.norm(ft,axis=-1)<=settings.contact_friction*fn+1e-5)
    projected,_=search.project(snap['search_candidates'])
    surface_error=float(np.max(np.linalg.norm(projected-snap['search_candidates'],axis=-1)))
    assert surface_error<1e-6
    assert np.array_equal(snap['c'],snap['z'])
    score_error=abs(plan.loss-(solver.last_solver_diag['FA_loss']+settings.contact_object_weight*solver.last_solver_diag['object_contact_cost']))/max(abs(plan.loss),1.)
    assert score_error<2e-6
    # Force/pose prediction must respond to a changed tracking target.
    target=targets.object_qpos[np.arange(5,30,6)].copy();shifted=target.copy();shifted[:,0]+=.01
    points=snap['search_points'][None];normals=normal[None]
    active=snap['mask'][np.arange(5,30,6)][None]>0
    pose=q[-7:];velocity=v[-6:]
    changed=search.force_solver.solve(pose,velocity,points,normals,active,shifted)
    response=float(np.linalg.norm(changed['forces'][0]-force))
    assert response>1e-4,response
    # Retain allocation owners while interleaving graph batch sizes.
    model=search.force_solver.model
    ob=engine.index.object_body
    for field in ('body_mass','body_inertia','body_ipos'):
        assert np.allclose(getattr(model.model,field)[1],getattr(engine.model,field)[ob],atol=1e-9),field
    for field in ('dof_armature','dof_damping','dof_frictionloss'):
        assert np.allclose(getattr(model.model,field),getattr(engine.model,field)[-6:],atol=1e-9),field
    assert np.array_equal(model.model.opt.gravity,engine.model.opt.gravity)
    for batch in (1,3,1,3):
        poses=model.rollout(pose,velocity,np.zeros((batch,5,6)))
        assert np.isfinite(poses).all()
    # Verify warm start is the shifted selected contact solution, including Y.
    memory=solver.objective.contact_memory
    next_targets=make_window(engine.sequence,executed_steps=dense+3,horizon=5,
        time_grid=cfg.simulator.time_grid,start_frame=0.,end_frame=71.)
    solver.backend._prepare_window(q,v,next_targets,**kw)
    restored=solver.objective.warm_contacts_restored;assert restored>0
    exact=0
    enabled=[(w,p) for w,p in zip((*solver.loss.immediate,solver.loss.terminal),solver.objective.points) if w.job.query_enabled]
    for i,(workspace,point) in enumerate(enabled):
        j=min(i+3,len(enabled)-1)
        active=(workspace.target.query_mask.numpy()>0)&(memory['mask'][j]>0)
        active&=workspace.target.query_body.numpy()==memory['body'][j]
        if active.any():
            for name in ('c','z','dual'):
                assert np.array_equal(getattr(point,name).numpy()[:,active],np.broadcast_to(memory[name][j,active],getattr(point,name).numpy()[:,active].shape))
            exact+=int(active.sum())
    result=dict(FA_value_relative_error=value_error,FA_hand_gradient_relative_error=gradient_error,
        hand_solver=a.hand_solver,cached_backward_gradient_relative_error=cached_error,
        original_FA_Adam_forbidden=True,final_combined_score_relative_error=score_error,
        native_ComFree_force_optimization_nonincreasing=True,
        selected_object_cost=snap['search_object_cost'].item(),initial_object_cost=snap['search_initial_cost'].item(),
        changed_target_force_response_N=response,surface_error_m=surface_error,
        max_normal_force_N=float(fn.max()),friction_cone_satisfied=True,
        restored_contact_slots_exact=exact,interleaved_CUDA_graph_buffers_finite=True,
        actual_execution_external_forces=False)
    (output/'force_contact_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
