"""Numerical and provenance checks for the executor/model ablations."""
from pathlib import Path
import json
import sys
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from examples.retarget.experiment_matrix import environment
    import os
    os.environ.update(environment())
    from examples.retarget.matrix_comfree import ComFreeEngine
    p=ROOT/'outputs/retarget_matrix_20261016'
    e=ComFreeEngine(p,ROOT/'thirdparty/retarget_assets/DexForge')
    from forceaware.executor import WarpActionExecutor
    native=mujoco.MjModel.from_xml_path(str(p/'sequence/scene/wuji_right/sugar_box_mesh.xml'))
    common={name:float(np.max(np.abs(np.asarray(getattr(native,name),float)-np.asarray(getattr(e.model,name),float)))) for name in
        ['body_mass','body_inertia','body_ipos','jnt_range','jnt_dofadr','jnt_actfrcrange','dof_armature','dof_damping','dof_frictionloss','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange']}
    if max(common.values())>1e-7:raise RuntimeError(common)
    object_native=int(np.flatnonzero((native.geom_bodyid==e.index.object_body)&(native.geom_contype!=0))[0])
    object_name=mujoco.mj_id2name(native,mujoco.mjtObj.mjOBJ_GEOM,object_native)
    # ComFree's CPU model keeps disabled proxy geoms; its contact layout
    # owns the actual SDF collisions and corresponding friction parameters.
    object_comfree=mujoco.mj_name2id(e.model,mujoco.mjtObj.mjOBJ_GEOM,object_name)
    assert object_comfree>=0
    common['object_geom_friction']=float(np.max(np.abs(native.geom_friction[object_native]-e.model.geom_friction[object_comfree])))
    nf=mujoco.mj_name2id(native,mujoco.mjtObj.mjOBJ_GEOM,'floor');cf=mujoco.mj_name2id(e.model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    common['floor_friction']=float(np.max(np.abs(native.geom_friction[nf]-e.model.geom_friction[cf])))
    common['gravity']=float(np.max(np.abs(native.opt.gravity-e.model.opt.gravity)))
    if max(common.values())>1e-7:raise RuntimeError(common)
    reference=WarpActionExecutor(e.model,e.device_model,index=e.index,cfg=e.cfg)
    hand=np.load(p/'contactaware/wuji_right_qpos.npy');objects=np.load(p/'contactaware/object_pose_7.npy')[:,[4,5,6,3,0,1,2]]
    maximum=0.;cases=[];fd=[]
    from planning.wuji_retarget_mpc import WujiMPCParams
    from examples.retarget.matrix_comfree import model_factory
    from models.wuji_hand import WujiHand,ObjectSurface
    h=WujiHand(p/'sequence/scene/wuji_right/sugar_box_mesh.xml',ObjectSurface(p/'sequence/model/textured_simple.obj',convex=True))
    f=model_factory(e)(WujiMPCParams(h,19),19)
    for frame in [0,31,71]:
        q=e.model.qpos0.copy();q[e.qadr]=hand[frame];q[e.oq:e.oq+7]=objects[frame]
        v=np.zeros(e.model.nv);target=np.clip(hand[frame]+np.r_[np.zeros(6),np.full(20,.002)],h.lower,h.upper)
        qn,vn,b=e.step(q,v,target)
        truth=reference.execute_action_sequence(q,v,actions=np.repeat(target[None],3,axis=0))
        error=max(np.max(np.abs(qn[0]-truth.next_qpos)),np.max(np.abs(vn[0]-truth.next_qvel)))
        cases.append(dict(frame=frame,maximum_state_error=float(error)));maximum=max(maximum,error)
        contacts=e.contacts(b);total=sum((c['force'] for c in contacts),np.zeros(3))
        for contact in contacts:
            basis=np.vstack([contact['normal'],contact['tangents']])
            assert np.allclose(basis@basis.T,np.eye(3),atol=1e-12,rtol=0)
        force=b['work'][-1].contact.output.constraint_force.numpy()[0,e.ov:e.ov+3]
        difference=np.max(np.abs(total-force))
        if difference>1e-3:raise RuntimeError(f'ComFree contact force reconstruction: {difference}')
        x=e.pack(q[None],v[None],hand[frame][None])[0];u=target-hand[frame]
        mp=f.prepare_stage(x,u,None,None,f.runtime_parameters(1.))
        # Independently verify selected columns with the same central scheme,
        # including both measured state and carried servo/command coordinates.
        eps={0:1e-4,7:1e-4,12:1e-3,40:1e-3,66:1e-4,91:1e-4,98:1e-3}
        start=f.base_dim+91+26+91
        J=np.c_[mp[start:start+91*91].reshape(91,91,order='F'),mp[start+91*91:].reshape(91,26,order='F')]
        z=np.r_[x,u]
        for column,delta in eps.items():
            plus=z.copy();minus=z.copy();plus[column]+=delta;minus[column]-=delta
            independent=(f.forward(plus[:91],plus[91:])-f.forward(minus[:91],minus[91:]))/(2*delta)
            fd.append(float(np.max(np.abs(independent-J[:,column]))))
    if maximum>1e-6:raise RuntimeError(f'Batched/graph ComFree step differs from official FA executor: {maximum}')
    if max(fd)>1e-5:raise RuntimeError(f'ComFree local Jacobian mismatch: {max(fd)}')
    audit=dict(physical_parameter_max_differences=common,graph_and_batch_vs_original_executor=cases,
        maximum_state_error=maximum,maximum_independent_jacobian_column_error=max(fd),
        force_reconstruction_tolerance_N=1e-3,checked_frames=[0,31,71])
    (p/'implementation_validation.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
