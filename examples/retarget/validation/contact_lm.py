"""Check analytic derivatives, exact costs, force feasibility and batch replay.

Numerical directional probes below are independent validation only; the LM
solver never estimates a Jacobian with finite differences.
"""
import argparse
from pathlib import Path
import json
import os
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=Path('outputs/fa_admm_solver_20261016'))
    parser.add_argument('--forge',type=Path,default=Path('thirdparty/retarget_assets/DexForge'))
    parser.add_argument('--output',type=Path,default=Path('outputs/fa_contact_timing/validation.json'))
    args=parser.parse_args();forge=args.forge.resolve()
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('COMFREE_WARP_ROOT',str(forge/'forceaware/third_party/comfree_warp'))
    os.environ.setdefault('HAND_ASSETS_ROOT',str(forge/'assets'))
    sys.path[:0]=[str(forge/'forceaware'),str(forge),os.environ['COMFREE_WARP_ROOT']]
    import numpy as np
    import torch
    import warp as wp
    from examples.retarget.common.contact_search import ObjectComFree,pose_residual,force_maps,write_object_scene
    from examples.retarget.common.contact_lm import ConeForceBatchLM,pose_terms,AnalyticObjectRollout
    from examples.retarget.common.engine import ComFreeEngine
    from forceaware.targets import make_window
    e=ComFreeEngine(args.experiment,forge)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    scene=write_object_scene(e.cfg.sequence.xml,args.output.parent/'validation_object_scene.xml')
    model=ObjectComFree(scene)
    lm=ConeForceBatchLM(model,iterations=2)
    api=lm.analytic
    saved=np.load(args.experiment/'mujoco/force_contact_mujoco_verified/rollout.npz')
    snapshot=np.load(args.experiment/'force_validation_snapshot.npz')
    pose=saved['qpos_traj'][90,-7:].copy();velocity=saved['qvel_traj'][90,-6:].copy()
    targets=make_window(e.sequence,executed_steps=90,horizon=5,time_grid=e.cfg.simulator.time_grid,start_frame=0.,end_frame=71.).object_qpos[np.arange(5,30,6)]
    rng=np.random.default_rng(710)
    reports=[]
    with torch.cuda.stream(api.stream):
        # Check residual value and closed-form pose Jacobian independently.
        samples=np.tile(targets,(3,1,1));samples[...,:3]+=rng.normal(0,.01,(3,5,3))
        samples[...,3:]+=rng.normal(0,.1,(3,5,4))
        tensor=api.tensor(samples).requires_grad_(True);target=api.tensor(targets)
        actual,jac=pose_terms(torch,tensor,target,lm.pose_weights)
        expected=pose_residual(samples,targets,*lm.pose_weights)
        value_error=float(np.max(np.abs(actual.detach().cpu().numpy()-expected)))
        # Reverse autodiff of the residual expression, never of an FD estimate.
        probe=torch.randn_like(actual)
        autodiff=torch.autograd.grad((actual*probe).sum(),tensor)[0]
        analytic=(jac.transpose(-1,-2)@probe.reshape(3,5,6,1)).squeeze(-1)
        jac_error=float((autodiff-analytic).detach().abs().max().cpu())
        assert value_error<.02 and jac_error<.01,(value_error,jac_error)
        for label,p in [('air',pose.copy()),('floor',pose.copy())]:
            if label=='air':p[2]+=.5
            if label=='floor':
                from scipy.spatial.transform import Rotation
                vertices=np.asarray(model.model.mesh_vert)
                bottom=np.min(Rotation.from_quat(p[3:],scalar_first=True).apply(vertices)[:,2])
                p[2]=-bottom+.001
            wrench=np.zeros((2,5,6));wrench[...,2]=float(model.model.body_mass[1]*9.81)*.4
            wrench+=rng.normal(size=wrench.shape)*np.array([.05,.05,.05,.001,.001,.001])
            wt=api.tensor(wrench)
            predicted,q,v=api.rollout(p,velocity,wt,states=True)
            J=api.trajectory_jacobian(q,v,wt).cpu().numpy()
            repeat=api.trajectory_jacobian(q,v,wt).cpu().numpy()
            assert np.allclose(J,repeat,atol=1.e-7),label
            errors=[];branch_changes=[]
            for _ in range(3):
                direction=rng.normal(size=wrench.shape)*np.array([.1,.1,.1,.001,.001,.001])
                h=.002
                positive=model.rollout(p,velocity,wrench+h*direction)
                positive_branch=np.stack([w.contact.final.rows.slope.numpy() for w in model._buffer(2)['work']])
                negative=model.rollout(p,velocity,wrench-h*direction)
                negative_branch=np.stack([w.contact.final.rows.slope.numpy() for w in model._buffer(2)['work']])
                branch_changes.append(int(np.count_nonzero(positive_branch!=negative_branch)))
                numeric=(positive-negative)/(2*h)
                analytic=np.einsum('bspk,bk->bsp',J,direction.reshape(2,-1))
                relative=np.linalg.norm(numeric-analytic)/max(np.linalg.norm(numeric),1.e-8)
                errors.append(float(relative))
            if max(errors)>=.035 and not any(branch_changes):
                local_A,local_B=api.local_jacobians(q,v,wt.repeat_interleave(6,dim=1))
                np.savez(args.output.parent/'derivative_debug.npz',J=J,numeric=numeric,analytic=analytic,
                    A=local_A.cpu().numpy(),B=local_B.cpu().numpy(),q=q.cpu().numpy(),v=v.cpu().numpy(),wrench=wrench,pose=p)
                print('numeric',numeric[0,-1], 'analytic',analytic[0,-1], 'B',local_B[0,0].cpu().numpy(),flush=True)
            assert all(error<.035 or changes>0 for error,changes in zip(errors,branch_changes)),(label,errors,branch_changes)
            reports.append(dict(state=label,directional_relative_errors=errors,hard_contact_branch_changes=branch_changes))
        # A contact-active single step avoids the long rollout's active-set
        # switches and independently checks the native contact force adjoint.
        short=ObjectComFree(scene,horizon=1,substeps=1)
        short_api=AnalyticObjectRollout(short)
        p[2]-=.002
        local_force=wrench[:,:1].copy()
        local_q=short_api.tensor(np.broadcast_to(p,(2,1,7)))
        local_v=short_api.tensor(np.broadcast_to(velocity,(2,1,6)))
        _,B=short_api.local_jacobians(local_q,local_v,short_api.tensor(local_force))
        B=B.cpu().numpy()[:,0,:7]
        direction=rng.normal(size=(2,1,6))*np.array([.1,.1,.1,.001,.001,.001])
        numeric=(short.rollout(p,velocity,local_force+.02*direction)-short.rollout(p,velocity,local_force-.02*direction))/.04
        analytic=np.einsum('bij,bj->bi',B,direction[:,0])[:,None]
        local_error=float(np.linalg.norm(numeric-analytic)/np.linalg.norm(numeric))
        assert local_error<.035,local_error
        reports.append(dict(state='contact_active_single_step',directional_relative_error=local_error))
        active=np.broadcast_to(snapshot['mask'][np.arange(5,30,6)]>0,(2,5,19))
        points=np.broadcast_to(snapshot['search_points'],(2,5,19,3))
        normals=np.broadcast_to(snapshot['search_normals'],points.shape)
        _,mapping=force_maps(points,normals,lm.friction)
        support,tracking=lm.initializations(pose,velocity,points,mapping,active,targets)
        support_reference=lm.support_initialization(pose,points,mapping,active)
        tracking_reference=lm.support_initialization(pose,points,mapping,active,
            desired_wrench=lm.trajectory_initialization(pose,velocity,targets))
        seed_error=float(max(np.max(np.abs(support.cpu().numpy()-support_reference)),
                             np.max(np.abs(tracking.cpu().numpy()-tracking_reference))))
        assert seed_error<1.e-3,seed_error
        result=lm.solve(pose,velocity,points,normals,active,targets)
        coeff=result['coefficients'];force=result['forces']
        assert coeff.min()>=0 and coeff.sum(-1).max()<=lm.cap+1.e-4
        assert np.all(coeff[~active]==0)
        assert np.all(result['cost']<=result['initial_cost']+1.e-2)
        normal_force=np.sum(force*(-normals),axis=-1)
        tangent=force+normal_force[...,None]*normals
        assert np.all(np.linalg.norm(tangent,axis=-1)<=lm.friction*normal_force+1.e-4)
        _,mapping=force_maps(points,normals,lm.friction)
        net=np.einsum('bkwd,bkd->bkw',mapping,coeff.reshape(2,5,-1))
        poses=model.rollout(pose,velocity,net)
        truth=np.sum(pose_residual(poses,targets,*lm.pose_weights)**2,axis=-1)+lm.reg*np.sum(force**2,axis=(1,2,3))
        cost_error=float(np.max(np.abs(truth-result['cost'])/np.maximum(truth,1.)))
        assert cost_error<2.e-4,cost_error
        assert np.allclose(result['pose'],poses,atol=2.e-6)
        assert np.allclose(result['cost'][0],result['cost'][1],rtol=1.e-5)
        changed=targets.copy();changed[:,0]+=.01
        response=lm.solve(pose,velocity,points,normals,active,changed)
        force_response=float(np.linalg.norm(result['forces']-response['forces']))
        assert force_response>1.e-4,force_response
        zero=lm.solve(pose,velocity,points,normals,np.zeros_like(active),targets)
        assert np.all(zero['forces']==0) and zero['model_evaluations_per_candidate']==1
    report=dict(residual_value_max_error=value_error,pose_jacobian_autodiff_max_error=jac_error,
        support_seed_max_error_N=seed_error,
        dynamics=reports,objective_relative_error=cost_error,feasible_forces=True,nonincreasing_cost=True,
        changed_target_force_response=force_response,jacobian='analytic native adjoint; no solver finite differences',
        cost=result['cost'].tolist(),initial_cost=result['initial_cost'].tolist())
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
