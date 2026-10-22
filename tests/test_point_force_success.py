"""New SCM law, model parity, objective parity and physical execution."""
import copy
import numpy as np
import pytest
import casadi as cs
import torch
from models.corrected_scm import corrected_scm_velocity,corrected_scm_velocity_cs,corrected_scm_velocity_torch
from planning.point_force_success import SuccessPlanner
from planning.point_force_benchmark import scene,surface_points,run_rollout
from planning.object_pose_cost import ObjectCostStages,target_up_axis
from examples.mpc.fingertips.test.run_point_force_success import parser


@pytest.mark.parametrize('rows',[0,1,8])
def test_new_scm_zero_distance_line_search_and_all_implementations(rows):
    rng=np.random.default_rng(7)
    A=rng.normal(size=(6,6));inverse=np.linalg.inv(A.T@A+np.eye(6))
    jac=rng.normal(size=(rows,6));gap=rng.uniform(-.001,.002,rows)
    free=rng.normal(size=(5,6))
    actual,info=corrected_scm_velocity(free,jac,gap,inverse,.02)
    g=free@jac.T+np.maximum(gap,0)/.02
    W=jac@inverse@jac.T+1e-8*np.eye(rows)
    initial=info['initial_impulse'];impulse=info['impulse']
    objective=lambda p:.5*np.einsum('ni,ij,nj->n',p,W,p)+np.sum(g*p,axis=1)
    assert np.all(objective(impulse)<=np.minimum(objective(initial),0)+1e-10)
    clamped=corrected_scm_velocity(free,jac,np.maximum(gap,0),inverse,.02)[0]
    np.testing.assert_array_equal(actual,clamped)
    tensors=[torch.tensor(x,dtype=torch.float64) for x in (free,jac,gap,inverse)]
    np.testing.assert_allclose(corrected_scm_velocity_torch(*tensors,.02).numpy(),actual,atol=1e-12)
    for k in range(len(free)):
        result=corrected_scm_velocity_cs(*[cs.DM(x) for x in (free[k],jac,gap,inverse)],.02)
        np.testing.assert_allclose(np.asarray(result).reshape(6),actual[k],atol=1e-12)


@pytest.mark.parametrize('name',['scm_corrected','comfree','contactsdf','qp'])
def test_loaded_state_contact_law_and_least_squares_objective(name):
    args=parser().parse_args(['--sample_num','12','--lm-iterations','2'])
    args.async_planner=False;args.obj='foam_brick';args.solver='lm'
    m,d,target,_=scene(args,'foam_brick',0);points,frames=surface_points(m,d,12,0)
    planner=SuccessPlanner(m,points,frames,target,args,name,0)
    pose=d.qpos.copy();pose[2]-=.001
    snapshot=(pose,np.array([.03,-.02,.01,.1,-.2,.15]),0.,0)
    stages=ObjectCostStages(args,target_up_axis(target[3:]))
    for stage in ('flip','pose'):
        stages.stage=stage
        assert planner.validate(snapshot,stages.config())['passed']
        ids=np.flatnonzero(planner.valid);raw=np.tile([.15,.55,.45],(len(ids),1))
        cmd,v,p,cost=planner._evaluate(raw,ids)
        residual=planner.residual(cmd,v,p)
        np.testing.assert_allclose(np.sum(residual**2,axis=1),cost,atol=1e-10)


@pytest.mark.parametrize('solver',['lm','cma-es','cem','ipopt'])
def test_solver_executes_continuous_point_force_with_shared_initial_state(solver):
    args=parser().parse_args(['--sample_num','12','--lm-iterations','2',
        '--population-size','8','--population-iterations','2'])
    args.async_planner=False;args.obj='foam_brick';args.solver=solver
    m,d,target,_=scene(args,'foam_brick',0)
    rows=run_rollout(args,'scm_corrected','foam_brick',seed=0,steps=2,planner_class=SuccessPlanner)
    np.testing.assert_array_equal(rows.initialization['initial_pose'],d.qpos)
    np.testing.assert_array_equal(rows.initialization['target_pose'],target)
    assert len(rows)==2
    for row in rows:
        assert row['execution_info']['pose_overwrites']==0
        assert row['execution_info']['active_substeps']==40
        np.testing.assert_allclose(row['execution_info']['actual_applied_impulse_body'][:3],
                                   .02*np.asarray(row['applied_force_body']),atol=1e-12)


def test_qp_adamw_uses_adam_updates_instead_of_cem(monkeypatch):
    args=parser().parse_args(['--sample_num','12','--adamw-iterations','2'])
    args.async_planner=False;args.obj='foam_brick';args.solver='adamw'
    from planning.point_force_gpu import GPUForceCEM
    monkeypatch.setattr(GPUForceCEM,'solve',lambda *a,**kw:pytest.fail('AdamW must not dispatch to CEM'))
    rows=run_rollout(args,'qp','foam_brick',seed=0,steps=2,planner_class=SuccessPlanner)
    assert len(rows)==2
    for row in rows:
        assert row['solver_backend']=='cuda-contact-adamw'
        assert row['search_iterations']==2
        assert row['execution_info']['active_substeps']==40


@pytest.mark.parametrize('name',['scm_corrected','comfree','contactsdf'])
@pytest.mark.parametrize('iterations',[1,3,20])
def test_cuda_batch_lm_matches_cpu_residual_step_acceptance_and_warm_start(name,iterations):
    args=parser().parse_args(['--sample_num','12','--lm-iterations',str(iterations)])
    args.async_planner=False;args.obj='foam_brick';args.solver='lm'
    m,d,target,_=scene(args,'foam_brick',0);points,frames=surface_points(m,d,12,0)
    cpu=SuccessPlanner(m,points,frames,target,args,name,0)
    gpu_args=copy.copy(args);gpu_args.solver='gpu-lm'
    gpu=SuccessPlanner(m,points,frames,target,gpu_args,name,0)
    stages=ObjectCostStages(args,target_up_axis(target[3:]))
    for stage in ('flip','pose'):
        stages.stage=stage
        pose=d.qpos.copy();pose[2]-=.001
        snapshot=(pose,np.array([.03,-.02,.01,.1,-.2,.15]),0.,0)
        assert gpu.validate(snapshot,stages.config())['passed']
        expected=cpu.solve(snapshot,stages.config())
        actual=gpu.solve(snapshot,stages.config())
        # At full budget, CPU/CUDA finite-difference roundoff can change late
        # accept/reject decisions in flat directions. Compare physical forces
        # within one FD force resolution, and objectives, rather than requiring
        # identical raw tangent coordinates when normal force is nearly zero.
        full_budget=iterations==20
        np.testing.assert_allclose(actual['cost'],expected['cost'],atol=1e-5 if full_budget else 2e-8,rtol=2e-8)
        np.testing.assert_allclose(actual['lam'],expected['lam'],atol=args.max_contact_force*1e-4 if full_budget else 2e-6,rtol=2e-6)
        ids=np.flatnonzero(gpu.valid)
        if full_budget:
            cpu_warm=cpu._evaluate(cpu.warm[ids],ids)
            gpu_warm=cpu._evaluate(gpu.gpu.warm.numpy()[ids],ids)
            np.testing.assert_allclose(gpu_warm[0],cpu_warm[0],atol=args.max_contact_force*1e-4,rtol=1e-5)
            np.testing.assert_allclose(gpu_warm[3],cpu_warm[3],atol=1e-5,rtol=1e-8)
        else:
            np.testing.assert_allclose(gpu.gpu.warm.numpy()[ids],cpu.warm[ids],atol=3e-6,rtol=3e-6)
        assert actual['solver_backend']=='cuda-batched-lm-finite-difference'
        assert actual['gpu_parallel_prediction']
        assert actual['search_iterations']==iterations
        assert actual['evaluations']==len(ids)*(3+8*iterations)


def test_gpu_batch_lm_execution_never_dispatches_to_cpu_lm_or_cem(monkeypatch):
    args=parser().parse_args(['--sample_num','12','--lm-iterations','2'])
    args.async_planner=False;args.obj='foam_brick';args.solver='gpu-lm'
    from planning.point_force_gpu import GPUForceCEM
    monkeypatch.setattr(SuccessPlanner,'_lm',lambda *a,**kw:pytest.fail('GPU LM must not dispatch to CPU LM'))
    monkeypatch.setattr(GPUForceCEM,'solve',lambda *a,**kw:pytest.fail('GPU LM must not dispatch to CEM'))
    rows=run_rollout(args,'contactsdf','foam_brick',seed=0,steps=2,planner_class=SuccessPlanner)
    assert len(rows)==2
    for row in rows:
        assert row['solver_backend']=='cuda-batched-lm-finite-difference'
        assert row['search_iterations']==2
        assert row['execution_info']['active_substeps']==40
        np.testing.assert_allclose(row['execution_info']['actual_applied_impulse_body'][:3],
                                   .02*np.asarray(row['applied_force_body']),atol=1e-12)
