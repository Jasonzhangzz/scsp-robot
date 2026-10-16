"""Native ComFree engine shared by retained MPC and validation tools."""
from pathlib import Path
import sys
import numpy as np
import mujoco


class ComFreeEngine:
    def __init__(self,experiment,forge,device='cuda:0'):
        self.root=Path(experiment).resolve();forge=Path(forge).resolve()
        sys.path[:0]=[str(forge/'forceaware'),str(forge)]
        from forceaware.config import load_retarget_config
        from forceaware.runtime import setup_runtime
        import os
        os.environ['FORCEAWARE_DEVICE']=device
        self.cfg=load_retarget_config(self.root/'sequence',self.root/'forceaware.yaml',hand='wuji_right',contactaware_dir=self.root/'contactaware')
        setup_runtime(self.cfg)
        import warp as wp
        import comfree_warp
        from forceaware.workflow import _load_runtime_models,_collision_output_layout,_object_surface_spheres
        from forceaware.targets import build_index,load_sequence
        from comfree_warp.native_adjoint.fast_step import compile_fast_step
        runtime=_load_runtime_models(self.cfg,comfree_warp)
        self.model=runtime.planner_cpu;self.device_model=runtime.planner_device
        self.index=build_index(self.model);self.compiled=compile_fast_step(self.model,self.device_model)
        self.device=self.compiled.base.device;self.wp=wp;self.buffers={};self.observers={}
        bodies,enabled=_collision_output_layout(self.device_model)
        self.sequence=load_sequence(self.cfg,output_body_ids=bodies,enabled_output_indices=enabled,
            object_surface_spheres=_object_surface_spheres(self.device_model,self.index.object_body))
        self.qadr=self.index.hand_qpos;self.oq=self.index.object_qpos
        self.dadr=self.model.jnt_dofadr[self.model.actuator_trnid[:,0]].astype(int)
        self.ov=int(self.model.jnt_dofadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'obj_joint')])

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

    def step(self,qpos,qvel,control,dt=.03):
        qpos=np.atleast_2d(qpos);qvel=np.atleast_2d(qvel);control=np.atleast_2d(control)
        count=len(qpos);substeps=round(dt/.01)
        assert abs(substeps*.01-dt)<1e-9
        b=self._buffer(count,substeps)
        for key,value in [('q',qpos),('v',qvel),('u',control)]:b[key].assign(value.astype(np.float32))
        if b['graph'] is None:b['result']=b['launch']()
        else:self.wp.capture_launch(b['graph'])
        result=b['result']
        return result.qpos.numpy().astype(float),result.qvel.numpy().astype(float),b

    def unpack(self,states):
        states=np.atleast_2d(states);q=np.tile(self.model.qpos0,(len(states),1));v=np.zeros((len(states),self.model.nv))
        q[:,self.oq:self.oq+7]=states[:,:7];q[:,self.qadr]=states[:,7:33]
        v[:,self.ov:self.ov+6]=states[:,33:39];v[:,self.dadr]=states[:,39:65]
        return q,v

    def pack(self,q,v,target):
        return np.c_[q[:,self.oq:self.oq+7],q[:,self.qadr],v[:,self.ov:self.ov+6],v[:,self.dadr],target]

    def predict(self,x,u,projection,bias,limits,lower,upper):
        x=np.atleast_2d(x);u=np.atleast_2d(u)
        command=np.clip(u@projection.T+bias,-limits,limits)
        target=np.clip(x[:,-26:]+command,lower,upper)
        q,v=self.unpack(x);q,v,_=self.step(q,v,target)
        return self.pack(q,v,target)

    def contacts(self,buffer,world=0):
        params=self.compiled.base.contact_rows.parameters;workspace=buffer['work'][-1]
        collision=workspace.base.collision.contacts
        pairs=params.contact_geom.numpy();bodies=self.model.geom_bodyid[pairs]
        rows=params.row_contact.numpy()[:params.contact_row_count]
        edges=params.row_edge.numpy()[:params.contact_row_count]
        mu=params.contact_friction.numpy();scale=float(params.friction_scale.numpy()[0])
        distances=collision.distance.numpy()[world];active=collision.active.numpy()[world]
        positions=collision.position.numpy()[world];frames=collision.frame.numpy()[world]
        coefficients=workspace.contact.output.force.numpy()[world][:params.contact_row_count]
        items=[]
        for c in np.flatnonzero(np.any(bodies==self.index.object_body,axis=1)&active.astype(bool)):
            sign=-1. if bodies[c,0]==self.index.object_body else 1.
            other=int(bodies[c,1] if bodies[c,0]==self.index.object_body else bodies[c,0])
            force=np.zeros(3)
            for row in np.flatnonzero(rows==c):
                axis=int(edges[row])//2;side=1. if int(edges[row])%2==0 else -1.
                direction=frames[c,0].copy()
                if axis<2:direction+=side*mu[c,axis]*scale*frames[c,axis+1]
                force+=sign*coefficients[row]*direction
            if distances[c]>.0001 and np.linalg.norm(force)<.001:continue
            # Warp frames are float32. Re-orthogonalize their existing axes
            # before double-precision diagnostics (which require 1e-7 accuracy).
            # Force reconstruction above retains the original solver frame.
            from planning.contact_friction import native_tangents
            normal=sign*frames[c,0].astype(float);normal/=np.linalg.norm(normal)
            tangents=native_tangents(normal,frames[c,1:].astype(float))
            items.append(dict(point=positions[c].astype(float),normal=normal,
                tangents=tangents,force=force,body=other,gap=float(distances[c])))
        return items
