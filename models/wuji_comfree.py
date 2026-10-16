"""Full ComFree forward dynamics for the Wuji retarget MPC.

Only prediction changes: measured execution, MANO attractors and the objective
remain owned by the caller. RTI uses a refreshed per-stage affine model, while
all numerical rollouts and reported forces call the official ComFree step.
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import os
import sys
import xml.etree.ElementTree as ET

import casadi as cs
import mujoco
import numpy as np


def build_comfree_scene(native_scene, query_hand, output, timestep=.01):
    """Replace hand collision clouds, preserving the executed physical tuning."""
    native_scene=Path(native_scene).resolve();query_hand=Path(query_hand).resolve()
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    root=ET.parse(native_scene).getroot();robot=ET.parse(query_hand).getroot()
    native_bodies={b.get('name'):b for b in root.findall('.//body')}
    # Query geometry changes, but body frames, joints, inertias and actuator
    # parameters must be those of the actual generated/executed scene.
    for body in robot.findall('.//body'):
        source=native_bodies.get(body.get('name'))
        if source is None:raise ValueError(f'ComFree hand body absent in execution scene: {body.get("name")}')
        body.attrib.clear();body.attrib.update(source.attrib)
        for tag in ('inertial','joint','site'):
            for old in list(body.findall(tag)):body.remove(old)
            for child in reversed(source.findall(tag)):body.insert(0,copy.deepcopy(child))
        colliders=[g for g in source.findall('geom') if g.get('contype','1')!='0']
        if colliders:
            for geom in body.findall("geom[@type='querypoint']"):
                for key in ('friction','solref','solimp','solmix','priority','condim','margin','gap'):
                    if key in colliders[0].attrib:geom.set(key,colliders[0].get(key))
    asset=root.find('asset')
    for node in robot.findall('asset/*'):
        item=copy.deepcopy(node)
        if item.get('file'):item.set('file',str((query_hand.parent/item.get('file')).resolve()))
        old=asset.find(f"*[@name='{item.get('name')}']")
        if old is not None:asset.remove(old)
        asset.append(item)
    world=root.find('worldbody')
    for body in robot.findall('worldbody/body'):
        old=world.find(f"body[@name='{body.get('name')}']")
        if old is None:raise ValueError('ComFree hand root does not match execution scene')
        index=list(world).index(old);world.remove(old)
        for node in body.iter():
            if node.get('file'):node.set('file',str((query_hand.parent/node.get('file')).resolve()))
        world.insert(index,copy.deepcopy(body))
    obj=world.find("body[@name='obj']")
    if obj is None:raise ValueError('ComFree retargeting requires body obj')
    geoms=[g for g in obj.findall('geom') if g.get('contype','1')!='0']
    if len(geoms)!=1 or geoms[0].get('type','mesh')!='mesh':
        raise ValueError('ComFree retargeting requires one mesh object collider')
    geom=geoms[0];mesh=asset.find(f"mesh[@name='{geom.get('mesh')}']")
    if mesh is None:raise ValueError('Object collider mesh asset is missing')
    if mesh.get('refpos') or mesh.get('refquat'):
        raise ValueError('Bake object mesh refpos/refquat before ComFree prediction')
    path=Path(mesh.get('file'))
    if not path.is_absolute():path=(native_scene.parent/path).resolve()
    scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
    if not np.array_equal(scale,np.ones(3)):
        import trimesh
        baked=trimesh.load_mesh(path,process=False);baked.apply_scale(scale)
        path=output.with_name(output.stem+'_object.obj');baked.export(path)
    geom.attrib.pop('mesh',None);geom.set('type','sdfmesh');geom.set('file',str(path))
    root.find('option').set('timestep',str(timestep))
    ET.ElementTree(root).write(output)
    return output


class NumericFunction:
    """CasADi for code generation, official nonlinear forward for observations."""
    def __init__(self,symbolic,numeric):self.symbolic=symbolic;self.numeric=numeric
    def __getattr__(self,key):return getattr(self.symbolic,key)
    def __call__(self,*args):
        return self.symbolic(*args) if any(isinstance(a,(cs.SX,cs.MX)) for a in args) else self.numeric(*args)


class ComFreeEngine:
    def __init__(self,scene,forge,*,timestep=.01,control_dt=.03,device='cuda:0',contact_topk=1):
        forge=Path(forge).resolve();warp_root=forge/'forceaware/third_party/comfree_warp'
        if not (warp_root/'comfree_warp').is_dir():
            raise FileNotFoundError(f'ComFree runtime missing: {warp_root}; run retarget.py --sync-assets')
        sys.path.insert(0,str(warp_root))
        import warp as wp
        wp.init();wp.set_device(device)
        import comfree_warp
        from comfree_warp.collision_config import CollisionConfig,configure_collision
        from comfree_warp.native_adjoint.fast_step import compile_fast_step
        self.device_model,self.model=comfree_warp.load_model(str(Path(scene).resolve()))
        self.timestep=float(timestep);self.control_dt=float(control_dt)
        self.model.opt.timestep=self.timestep;self.device_model.opt.timestep.fill_(self.timestep)
        configure_collision(self.device_model,CollisionConfig(int(contact_topk),0.))
        self.compiled=compile_fast_step(self.model,self.device_model)
        self.wp=wp;self.device=self.compiled.base.device;self.buffers={}
        self.qadr=self.model.jnt_qposadr[self.model.actuator_trnid[:,0]].astype(int)
        self.dadr=self.model.jnt_dofadr[self.model.actuator_trnid[:,0]].astype(int)
        joint=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'obj_joint')
        self.oq=int(self.model.jnt_qposadr[joint]);self.ov=int(self.model.jnt_dofadr[joint])
        self.index=SimpleNamespace(object_body=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'obj'))
        if (self.model.nq,self.model.nv,self.model.nu)!=(33,32,26):
            raise ValueError('ComFree retarget MPC requires the 26-DOF Wuji hand and one free object')
        self._substeps(self.control_dt)
        params=self.compiled.base.contact_rows.parameters
        self.pairs=params.contact_geom.numpy();self.bodies=self.model.geom_bodyid[self.pairs]
        self.row_ids=params.row_contact.numpy()[:params.contact_row_count]
        self.edges=params.row_edge.numpy()[:params.contact_row_count]
        self.friction=params.contact_friction.numpy();self.friction_scale=float(params.friction_scale.numpy()[0])
        self.object_contacts=np.flatnonzero(np.any(self.bodies==self.index.object_body,axis=1))
        self.row_count=params.contact_row_count

    def validate_physics(self,hand):
        fields=('body_mass','body_inertia','body_ipos','body_pos','body_quat','jnt_range','jnt_actfrcrange',
            'dof_armature','dof_damping','dof_frictionloss','actuator_gainprm','actuator_biasprm',
            'actuator_ctrlrange','actuator_forcerange')
        differences={name:float(np.max(np.abs(getattr(hand.model,name)-getattr(self.model,name)))) for name in fields}
        for kind,count in [(mujoco.mjtObj.mjOBJ_BODY,hand.model.nbody),(mujoco.mjtObj.mjOBJ_JOINT,hand.model.njnt)]:
            if any(mujoco.mj_id2name(hand.model,kind,i)!=mujoco.mj_id2name(self.model,kind,i) for i in range(count)):
                raise ValueError('ComFree and execution scene have different body/joint indexing')
        differences['gravity']=float(np.max(np.abs(hand.model.opt.gravity-self.model.opt.gravity)))
        if max(differences.values())>1e-7:raise ValueError(f'ComFree changed execution physics: {differences}')
        return differences

    def _substeps(self,dt):
        steps=int(round(dt/self.timestep))
        if steps<1 or abs(steps*self.timestep-dt)>1e-9:
            raise ValueError('Control dt must be a positive integer multiple of ComFree timestep')
        return steps

    def _buffer(self,worlds,substeps):
        key=(worlds,substeps)
        if key in self.buffers:return self.buffers[key]
        wp=self.wp
        from comfree_warp.native_adjoint.fast_step import allocate_workspace,step
        from comfree_warp.native_adjoint.dynamics import DynamicsInput
        from comfree_warp.native_adjoint.runtime import StepInput
        q=wp.array(np.tile(self.model.qpos0,(worlds,1)).astype(np.float32),device=self.device)
        v=wp.zeros((worlds,self.model.nv),device=self.device);u=wp.zeros((worlds,self.model.nu),device=self.device)
        force=wp.zeros_like(v);time=wp.zeros(worlds,device=self.device);soft=wp.zeros_like(time)
        work=[allocate_workspace(self.compiled,worlds) for _ in range(substeps)]
        def launch():
            pos,vel,clock=q,v,time
            for w in work:
                d=DynamicsInput();d.qpos=pos;d.qvel=vel;d.ctrl=u;d.qfrc_applied=force
                result=step(self.compiled,StepInput(d,clock,soft),w)
                pos,vel,clock=result.qpos,result.qvel,result.time
            return result
        result=launch();wp.synchronize()
        graph=None
        if self.device.is_cuda:
            with wp.ScopedCapture(device=self.device) as capture:result=launch()
            graph=capture.graph
        b=dict(q=q,v=v,u=u,launch=launch,result=result,graph=graph,work=work)
        self.buffers[key]=b;return b

    def step(self,qpos,qvel,control,dt=None):
        qpos=np.atleast_2d(qpos);qvel=np.atleast_2d(qvel);control=np.atleast_2d(control)
        if not all(np.isfinite(x).all() for x in (qpos,qvel,control)):raise ValueError('Nonfinite ComFree input')
        count=len(qpos)
        if qpos.shape!=(count,33) or qvel.shape!=(count,32) or control.shape!=(count,26):
            raise ValueError('ComFree batch state/control shape mismatch')
        b=self._buffer(count,self._substeps(self.control_dt if dt is None else dt))
        for key,value in [('q',qpos),('v',qvel),('u',control)]:b[key].assign(value.astype(np.float32))
        if b['graph'] is None:b['result']=b['launch']()
        else:self.wp.capture_launch(b['graph'])
        result=b['result'];q=result.qpos.numpy().astype(float);v=result.qvel.numpy().astype(float)
        if not np.isfinite(q).all() or not np.isfinite(v).all():raise RuntimeError('ComFree produced nonfinite state')
        return q,v,b

    def unpack(self,states):
        states=np.atleast_2d(states);q=np.tile(self.model.qpos0,(len(states),1));v=np.zeros((len(states),32))
        q[:,self.oq:self.oq+7]=states[:,:7];q[:,self.qadr]=states[:,7:33]
        v[:,self.ov:self.ov+6]=states[:,33:39];v[:,self.dadr]=states[:,39:65]
        return q,v

    def pack(self,q,v,target):
        return np.c_[q[:,self.oq:self.oq+7],q[:,self.qadr],v[:,self.ov:self.ov+6],v[:,self.dadr],target]

    def predict(self,x,u,projection,bias,limits,lower,upper,*,return_buffer=False):
        x=np.atleast_2d(x);u=np.atleast_2d(u)
        target=np.clip(x[:,-26:]+np.clip(u@projection.T+bias,-limits,limits),lower,upper)
        q,v=self.unpack(x);q,v,b=self.step(q,v,target)
        result=self.pack(q,v,target)
        return (result,b) if return_buffer else result

    def contact_forces(self,buffer,materials,active):
        """Forces on the object, from real ComFree constraint rows, per slot."""
        workspace=buffer['work'][-1];collision=workspace.base.collision.contacts
        frames=collision.frame.numpy().astype(float);enabled=collision.active.numpy().astype(bool)
        coefficient=workspace.contact.output.force.numpy()[:,:self.row_count].astype(float)
        position=collision.position.numpy().astype(float)
        body_p=workspace.base.pose.body_position.numpy().astype(float)
        body_r=workspace.base.pose.body_matrix.numpy().astype(float)
        worlds=len(frames);result=np.zeros((worlds,len(materials),3));world=np.arange(worlds)
        for c in self.object_contacts:
            sign=-1. if self.bodies[c,0]==self.index.object_body else 1.
            body=int(self.bodies[c,1] if sign<0 else self.bodies[c,0])
            ids=[k for k,m in enumerate(materials) if m.body==body]
            physical=[k for k in ids if materials[k].contact_patch]
            preferred=physical or [k for k in ids if active[k]]
            ids=preferred or ids
            if not ids:continue
            rows=np.flatnonzero(self.row_ids==c);directions=np.repeat(frames[:,c,0,None,:],len(rows),axis=1)
            for j,row in enumerate(rows):
                axis=int(self.edges[row])//2;side=1. if int(self.edges[row])%2==0 else -1.
                if axis<2:directions[:,j]+=side*self.friction[c,axis]*self.friction_scale*frames[:,c,axis+1]
            force=sign*np.sum(coefficient[:,rows,None]*directions,axis=1)*enabled[:,c,None]
            points=body_p[:,body,None,:]+np.einsum('wij,kj->wki',body_r[:,body],np.array([materials[k].local for k in ids]))
            distance=np.linalg.norm(points-position[:,c,None,:],axis=2)
            if physical:
                normal=sign*frames[:,c,0];normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
                direction=np.einsum('wij,kj->wki',body_r[:,body],np.array([materials[k].normal for k in ids]))
                distance+=.01*(1-np.einsum('wki,wi->wk',direction,normal))
            selected=np.array(ids)[np.argmin(distance,axis=1)]
            result[world,selected]+=force
        return result.reshape(worlds,-1)


def comfree_model_factory(engine):
    # Import lazily: controller definitions can be used without Warp installed.
    from planning.wuji_retarget_mpc import WujiContactContext
    class WujiComFreeModel(WujiContactContext):
        contact_model='comfree'
        uses_scm_grasp_response=False
        def __init__(self,p,slots):
            super().__init__(p,slots)
            nx=p.n_qpos_;nu=p.n_cmd_
            self.model_parameter_dim=nx+nu+nx+nx*nx+nx*nu
            self.extra=np.r_[np.zeros(nx+nu+nx),np.eye(nx).ravel(order='F'),np.zeros(nx*nu)]
            self.materials=[];self.active=np.zeros(slots)
            x=cs.SX.sym('x',nx);u=cs.SX.sym('u',nu);phi=cs.SX.sym('phi',4*p.max_ncon_)
            jac=cs.SX.sym('jac',4*p.max_ncon_,p.n_qvel_);mp=cs.SX.sym('mp',self.model_parameter_dim)
            i=0;x0=mp[i:i+nx];i+=nx;u0=mp[i:i+nu];i+=nu;f0=mp[i:i+nx];i+=nx
            A=cs.reshape(mp[i:i+nx*nx],nx,nx);i+=nx*nx;B=cs.reshape(mp[i:],nx,nu)
            symbolic=cs.Function('wuji_comfree_local_step',[x,u,phi,jac,mp],[f0+A@(x-x0)+B@(u-u0)])
            self.prediction_fn=NumericFunction(symbolic,lambda x,u,*_:self.predict_numeric(x,u))
            self.contact_force_fn=self.contact_forces;self.force_linearization_fn=self.force_linearization
        def runtime_parameters(self,sigma):return self.extra.copy()
        def set_contact_materials(self,materials,active):
            self.materials=list(materials);self.active=np.asarray(active).copy()
        def _predict(self,x,u,forces=False):
            result,buffer=engine.predict(x,u,self.projection,self.pressure_bias,self.command_limits,
                self.param_.hand_lower,self.param_.hand_upper,return_buffer=True)
            return engine.contact_forces(buffer,self.materials,self.active) if forces else result
        def predict_numeric(self,x,u):return self._predict(x,u)[0]
        def contact_forces(self,x,u,*_):
            if not self.slots:return np.zeros(0)
            if len(self.materials)!=self.slots:raise RuntimeError('Set ComFree contact materials before requesting forces')
            return self._predict(x,u,forces=True)[0]
        def _batch(self,x,u):
            z=np.r_[x,u]
            epsilon=np.r_[np.full(3,1e-4),np.full(4,2e-4),np.full(3,1e-4),np.full(23,1e-3),
                np.full(32,1e-3),np.full(3,1e-4),np.full(23,1e-3),np.full(3,1e-4),np.full(23,1e-3)]
            batch=np.tile(z,(1+2*len(z),1))
            for axis in range(len(z)):
                batch[1+2*axis,axis]+=epsilon[axis];batch[2+2*axis,axis]-=epsilon[axis]
            return batch,epsilon
        def force_linearization(self,x,u,*_):
            if not self.slots:return np.zeros(0),np.zeros((0,len(x))),np.zeros((0,len(u)))
            batch,epsilon=self._batch(x,u);nx=len(x)
            values=self._predict(batch[:,:nx],batch[:,nx:],forces=True)
            J=((values[1::2]-values[2::2])/(2*epsilon[:,None])).T
            return values[0],J[:,:nx],J[:,nx:]
        def prepare_stage(self,x,u,phi,jac,mp):
            batch,epsilon=self._batch(x,u);nx=len(x)
            values=self._predict(batch[:,:nx],batch[:,nx:])
            J=((values[1::2]-values[2::2])/(2*epsilon[:,None])).T
            self.extra=np.r_[x,u,values[0],J[:,:nx].ravel(order='F'),J[:,nx:].ravel(order='F')]
            return self.extra.copy()
    return WujiComFreeModel
