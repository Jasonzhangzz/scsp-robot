"""Official ComFree dynamics, batched numerical linearization and executor adapter.

The CasADi graph is the current per-stage local model; numerical rollouts always
call ComFree. Its local derivatives are central differences of the identical
hard-contact forward step, recomputed at every SQP_RTI observation/stage.
"""
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


class NumericFunction:
    def __init__(self,symbolic,numeric):self.symbolic=symbolic;self.numeric=numeric
    def __getattr__(self,key):return getattr(self.symbolic,key)
    def __call__(self,*args):
        import casadi as cs
        return self.symbolic(*args) if any(isinstance(a,(cs.SX,cs.MX)) for a in args) else self.numeric(*args)


def model_factory(engine):
    from planning.wuji_retarget_mpc import WujiContactModel
    import casadi as cs
    class ComFreeModel(WujiContactModel):
        contact_model='comfree'
        def __init__(self,p,slots):
            super().__init__(p,slots)
            self.base_dim=self.model_parameter_dim;nx=p.n_qpos_;nu=p.n_cmd_
            self.model_parameter_dim+=nx+nu+nx+nx*nx+nx*nu
            x=cs.SX.sym('x',nx);u=cs.SX.sym('u',nu);phi=cs.SX.sym('phi',4*p.max_ncon_)
            jac=cs.SX.sym('jac',4*p.max_ncon_,p.n_qvel_);mp=cs.SX.sym('mp',self.model_parameter_dim)
            i=self.base_dim;x0=mp[i:i+nx];i+=nx;u0=mp[i:i+nu];i+=nu
            f0=mp[i:i+nx];i+=nx;A=cs.reshape(mp[i:i+nx*nx],nx,nx);i+=nx*nx;B=cs.reshape(mp[i:],nx,nu)
            symbolic=cs.Function('comfree_local_step',[x,u,phi,jac,mp],[f0+A@(x-x0)+B@(u-u0)])
            self.prediction_fn=NumericFunction(symbolic,lambda x,u,*_:self.forward(x,u))
            # The current objective has no force residual; these diagnostic
            # derivatives are not used to inject forces into the dynamics.
            self.contact_force_fn=NumericFunction(self.contact_force_fn,lambda *a:np.zeros(3*slots))
            self.force_linearization_fn=NumericFunction(self.force_linearization_fn,
                lambda *a:(np.zeros(3*slots),np.zeros((3*slots,nx)),np.zeros((3*slots,nu))))
            self.extra=np.r_[np.zeros(nx+nu+nx),np.eye(nx).ravel(order='F'),np.zeros(nx*nu)]
            self.tag='comfree_forward_fd_rti'
        def runtime_parameters(self,sigma):return np.r_[super().runtime_parameters(sigma),self.extra]
        def forward(self,x,u):
            return engine.predict(x,u,self.projection,self.pressure_bias,self.command_limits,self.param_.hand_lower,self.param_.hand_upper)[0]
        def prepare_stage(self,x,u,phi,jac,mp):
            nx=len(x);nu=len(u);z=np.r_[x,u]
            # Float32 dynamics need resolved perturbations, not 1e-8 deltas.
            epsilon=np.r_[np.full(3,1e-4),np.full(4,2e-4),np.full(3,1e-4),np.full(23,1e-3),np.full(32,1e-3),
                np.full(3,1e-4),np.full(23,1e-3),np.full(3,1e-4),np.full(23,1e-3)]
            assert len(epsilon)==nx+nu
            batch=np.tile(z,(1+2*len(z),1))
            for axis in range(len(z)):
                batch[1+2*axis,axis]+=epsilon[axis];batch[2+2*axis,axis]-=epsilon[axis]
            values=engine.predict(batch[:,:nx],batch[:,nx:],self.projection,self.pressure_bias,self.command_limits,self.param_.hand_lower,self.param_.hand_upper)
            J=((values[1::2]-values[2::2])/(2*epsilon[:,None])).T
            self.extra=np.r_[x,u,values[0],J[:,:nx].ravel(order='F'),J[:,nx:].ravel(order='F')]
            return np.r_[np.asarray(mp)[:self.base_dim],self.extra]
    return ComFreeModel


def attach_executor(hand,engine):
    """Keep MuJoCo solely for FK/mesh guards; step and forces use ComFree."""
    import types
    cache={};hand.execution_backend='comfree'
    def update(data,q,v,target,dt,buffer):
        data.qpos[:]=q;data.qvel[:]=v;data.ctrl[:]=target;data.time+=dt
        mujoco.mj_forward(hand.model,data)
        cache[id(data)]=engine.contacts(buffer)
    def execute(self,target,dt):
        self.servo=np.clip(target,self.lower,self.upper)
        q,v,b=engine.step(self.data.qpos,self.data.qvel,self.servo,dt)
        update(self.data,q[0],v[0],self.servo,dt,b)
    def preview_sequence(self,targets,dt,observer=None):
        mujoco.mj_copyData(self.preview,self.model,self.data)
        for target in targets:
            target=np.clip(target,self.lower,self.upper)
            q,v,b=engine.step(self.preview.qpos,self.preview.qvel,target,dt)
            update(self.preview,q[0],v[0],target,dt,b)
            if observer is not None:observer(self.preview)
        return self.preview
    def observed(self,materials,active=None,data=None):
        data=self.data if data is None else data
        points=self.kinematics(materials,data)[0];forces=np.zeros((len(materials),3));actual=[]
        for c in cache.get(id(data),[]):
            if c['body'] not in self.hand_bodies:continue
            ids=[k for k,m in enumerate(materials) if m.body==c['body']]
            physical=[k for k in ids if materials[k].contact_patch]
            preferred=physical or [k for k in ids if active is not None and active[k]]
            if preferred:ids=preferred
            def distance(k):
                result=np.linalg.norm(points[k]-c['point'])
                if physical:result+=.01*(1-(data.xmat[materials[k].body].reshape(3,3)@materials[k].normal)@c['normal'])
                return result
            slot=min(ids,key=distance) if ids else -1
            item=dict(c,slot=slot)
            if slot>=0:forces[slot]+=c['force']
            actual.append(item)
        return forces,actual
    hand.execute=types.MethodType(execute,hand)
    hand.preview_sequence=types.MethodType(preview_sequence,hand)
    hand.observed_contacts=types.MethodType(observed,hand)
    hand.comfree_contacts=lambda data=None:cache.get(id(hand.data if data is None else data),[])
    return hand
