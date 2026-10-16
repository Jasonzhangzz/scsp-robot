"""Native ComFree stage functions for the direct acados OCP.

No finite differences: dynamics Jacobians are batches of native reverse-mode
seeds; FA gradients use its original loss and observation adjoints.
"""
from time import perf_counter
from dataclasses import replace
import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def seed_outputs(q: wp.array2d(dtype=float), v: wp.array2d(dtype=float), a: wp.array2d(dtype=float)):
    world, col = wp.tid()
    if col < q.shape[1]:
        q[world, col] = float(world == col)
    elif col < q.shape[1] + v.shape[1]:
        v[world, col-q.shape[1]] = float(world == col)
    else:
        a[world, col-q.shape[1]-v.shape[1]] = float(world == col)


@wp.kernel(enable_backward=False)
def add_q(left: wp.array2d(dtype=float), right: wp.array2d(dtype=float)):
    i, j = wp.tid()
    right[i, j] += left[i, j]


@wp.kernel(enable_backward=False)
def residual_seeds(seed: wp.array2d(dtype=float)):
    i,j=wp.tid();seed[i,j]=.001*float(i==j)


class NativeStages:
    def __init__(self, engine, horizon, *, full_contact_jacobian=True):
        self.engine=engine;self.horizon=horizon;self.device=engine.device
        self.nq=engine.model.nq;self.nv=engine.model.nv;self.nu=engine.model.nu
        self.nx=self.nq+self.nv;self.nout=self.nx+self.nv
        self.full_contact_jacobian=full_contact_jacobian
        self.dynamics_buffers={};self.cost_buffers={};self.cache={};self.cost_cache={}
        self.timing={};self.calls={};self.error=None
        from forceaware.mpc_loss import compile_model,make_weights,_compile_model
        self.loss_model=compile_model(engine.model,engine.index,engine.compiled.base.collision,
            hand='wuji_right',guidance_count=engine.sequence.contact.mask.shape[1],device=self.device)
        self.weights=make_weights(engine.cfg)
        self.regularization=_compile_model(engine.model,engine.index,engine.cfg,self.device)
        self.nr=30+4*self.loss_model.guidance_count+2*self.loss_model.output_count+self.loss_model.finger_joint_pair_count
        self.residual_buffers={};self.residual_jobs={};self.residual_cache={}

    def _dynamics_buffer(self, derivative):
        if derivative in self.dynamics_buffers:return self.dynamics_buffers[derivative]
        from comfree_warp.native_adjoint.dynamics import DynamicsInput
        from comfree_warp.native_adjoint.runtime import StepInput
        from comfree_warp.native_adjoint.fast_step_adjoint import allocate_workspace,record,forward,backward,BackwardCall
        from forceaware.optimizer import _allocate_cotangent
        from comfree_warp.collision_config import FREEZE_TANGENT_GAUGE_VJP
        n=self.nout if derivative else 1
        q=wp.array(np.tile(self.engine.model.qpos0,(n,1)).astype(np.float32),device=self.device,requires_grad=True)
        v=wp.zeros((n,self.nv),device=self.device,requires_grad=True)
        u=wp.zeros((n,self.nu),device=self.device,requires_grad=True)
        d=DynamicsInput();d.qpos=q;d.qvel=v;d.ctrl=u
        d.qfrc_applied=wp.zeros_like(v,requires_grad=True)
        inputs=StepInput(d,wp.zeros(n,device=self.device,requires_grad=True),wp.zeros(n,device=self.device,requires_grad=True))
        work=allocate_workspace(self.engine.compiled,n)
        recorded=record(self.engine.compiled,inputs,work,freeze_frame_vjp=FREEZE_TANGENT_GAUGE_VJP)
        if derivative and self.full_contact_jacobian:
            from comfree_warp.native_adjoint.geometry_vjp import GeometryTape
            from comfree_warp.native_adjoint import gaussian_collision
            from comfree_warp.native_adjoint.runtime import collision_contacts
            class FullContactTape(GeometryTape):
                def record_launch(self,kernel,dim,max_blocks,inputs,outputs,device,block_dim=0,metadata=None):
                    if kernel in (gaussian_collision._narrow_pairs,gaussian_collision._narrow_plane):
                        inputs[1].stop_frame_vjp=0
                    super().record_launch(kernel,dim,max_blocks,inputs,outputs,device,block_dim=block_dim,metadata=metadata)
            tape=FullContactTape()
            with tape:collision_contacts(self.engine.compiled.base,work.forward.base,freeze_frame_vjp=False)
            recorded=replace(recorded,contact_tape=tape)
        cotangent=_allocate_cotangent(self.engine.compiled,n)
        if derivative:wp.launch(seed_outputs,(n,self.nout),inputs=[cotangent.qpos,cotangent.qvel,cotangent.qacc],device=self.device)
        def launch():
            result=forward(self.engine.compiled,inputs,work)
            if derivative:backward(BackwardCall(self.engine.compiled,inputs,work,recorded,cotangent))
            return result
        launch();wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as capture:result=launch()
        # CUDA graphs keep device addresses, not Python allocation owners.
        # Retain StepInput (applied forces, clock, softness) as well as all
        # workspaces; later observer/executor allocations may reuse freed data.
        b=dict(q=q,v=v,u=u,inputs=inputs,launch=launch,work=work,recorded=recorded,
            seed=cotangent,result=result,graph=capture.graph)
        self.dynamics_buffers[derivative]=b;return b

    def physics(self,x,u,derivative=False):
        t=perf_counter();x=np.asarray(x,dtype=np.float32);u=np.asarray(u,dtype=np.float32)
        key=(x.tobytes(),u.tobytes());cached=self.cache.get(key)
        if cached is not None and (not derivative or cached[1] is not None):return cached
        b=self._dynamics_buffer(derivative);n=len(b['q'])
        for name,value in [('q',x[:self.nq]),('v',x[self.nq:]),('u',u)]:
            b[name].assign(np.broadcast_to(value,b[name].shape).copy())
        wp.capture_launch(b['graph'])
        r=b['result'];f=np.r_[r.qpos.numpy()[0],r.qvel.numpy()[0],r.qacc.numpy()[0]].astype(float)
        jac=None
        if derivative:jac=np.c_[b['q'].grad.numpy(),b['v'].grad.numpy(),b['u'].grad.numpy()].astype(float)
        if not np.isfinite(f).all() or (jac is not None and not np.isfinite(jac).all()):
            raise FloatingPointError('Nonfinite native ComFree stage')
        self.cache[key]=(f,jac)
        if len(self.cache)>4*(self.horizon+1):self.cache.pop(next(iter(self.cache)))
        label='dynamics_jacobian' if derivative else 'dynamics_forward'
        self.timing[label]=self.timing.get(label,0.)+perf_counter()-t
        self.calls[label]=self.calls.get(label,0)+1
        return f,jac

    def _cost_buffer(self,stage):
        if stage in self.cost_buffers:return self.cost_buffers[stage]
        from comfree_warp.native_adjoint import observation_adjoint as obs
        from forceaware import mpc_loss as fa
        from comfree_warp.collision_config import FREEZE_TANGENT_GAUGE_VJP
        q=wp.array(self.engine.model.qpos0[None].astype(np.float32),device=self.device,requires_grad=True)
        work=obs.allocate_workspace(self.engine.compiled.base,1)
        recorded=obs.record(self.engine.compiled.base,q,work,freeze_frame_vjp=FREEZE_TANGENT_GAUGE_VJP)
        state=fa.LossState();state.qpos=q
        for name in ('body_position','body_matrix','contact_distance','contact_position','contact_frame'):
            setattr(state,name,getattr(recorded.result,name))
        state.contact_active=work.collision.contacts.active
        job=fa.LossJob();job.model=self.loss_model;job.weights=self.weights;job.state=state
        job.target=fa.allocate_target(self.nu,self.loss_model.guidance_count,self.device)
        job.guidance_enabled=int(stage>0);job.query_enabled=int(stage>0);job.penetration_enabled=1
        # Tracking residuals are symbolic in the OCP; this external term is
        # exactly the remaining native contact/penetration/capsule loss.
        job.reference_scale=0.
        loss=fa.build_workspace(job,1,self.device)
        cotangent=obs.allocate_cotangent(self.engine.compiled.base,1)
        direct=wp.zeros_like(q);gradient=wp.zeros_like(q)
        def forward():
            obs.forward(self.engine.compiled.base,q,work);fa.evaluate(loss)
        def backward():
            fa.backward(loss)
            wp.copy(direct,q.grad)
            for name in ('body_position','body_matrix','contact_distance','contact_position','contact_frame'):
                wp.copy(getattr(cotangent,name),getattr(state,name).grad)
            obs.backward(q,work,recorded=recorded,cotangent=cotangent)
            wp.launch(add_q,q.shape,inputs=[direct,q.grad],device=self.device)
        # Graph construction is deferred until valid window targets exist.
        b=dict(q=q,work=work,recorded=recorded,loss=loss,seed=cotangent,direct=direct,
            gradient=gradient,forward=forward,backward=backward,graphs={},previous=None,H=None)
        self.cost_buffers[stage]=b;return b

    def prepare(self,targets):
        from forceaware.mpc_loss import assign_target
        self.targets=targets;self.cache.clear();self.cost_cache.clear();self.residual_cache.clear();self.timing={};self.calls={}
        for stage in range(self.horizon+1):
            b=self._cost_buffer(stage);k=min(stage,self.horizon-1)
            assign_target(b['loss'].target,targets,k,contact_step=max(k-1,0) if stage<self.horizon else k,grasp=False)

    def _residual_buffer(self,derivative):
        if derivative in self.residual_buffers:return self.residual_buffers[derivative]
        from comfree_warp.native_adjoint import observation_adjoint as obs
        from comfree_warp.collision_config import FREEZE_TANGENT_GAUGE_VJP
        from examples.retarget.direct.residual import residual
        n=self.nr if derivative else 1
        q=wp.array(np.tile(self.engine.model.qpos0,(n,1)).astype(np.float32),device=self.device,requires_grad=True)
        work=obs.allocate_workspace(self.engine.compiled.base,n)
        recorded=obs.record(self.engine.compiled.base,q,work,freeze_frame_vjp=FREEZE_TANGENT_GAUGE_VJP)
        cotangent=obs.allocate_cotangent(self.engine.compiled.base,n)
        r=wp.zeros((n,self.nr),device=self.device,requires_grad=True)
        seeds=wp.zeros_like(r)
        if derivative:wp.launch(residual_seeds,seeds.shape,inputs=[seeds],device=self.device)
        direct=wp.zeros_like(q)
        analytic=wp.kernel(residual.func,enable_backward=True,module='unique')
        b=dict(q=q,work=work,recorded=recorded,seed=cotangent,r=r,seeds=seeds,direct=direct,kernel=analytic)
        self.residual_buffers[derivative]=b;return b

    def residual(self,q,stage,derivative=False):
        from comfree_warp.native_adjoint import observation_adjoint as obs
        from forceaware import mpc_loss as fa
        t=perf_counter();q=np.asarray(q,dtype=np.float32);key=(stage,q.tobytes())
        cached=self.residual_cache.get(key)
        if cached is not None and (not derivative or cached[1] is not None):return cached
        b=self._residual_buffer(derivative);n=len(b['q']);jobkey=(stage,derivative)
        if jobkey not in self.residual_jobs:
            state=fa.LossState();state.qpos=b['q']
            for name in ('body_position','body_matrix','contact_distance','contact_position','contact_frame'):
                setattr(state,name,getattr(b['recorded'].result,name))
            state.contact_active=b['work'].collision.contacts.active
            job=fa.LossJob();job.model=self.loss_model;job.weights=self.weights;job.state=state
            job.target=self._cost_buffer(stage)['loss'].target
            job.reference_scale=0.;job.guidance_enabled=int(stage>0);job.query_enabled=int(stage>0);job.penetration_enabled=1
            job.auxiliary=fa.LossAuxiliary();job.auxiliary.marked_query_weight=wp.zeros((n,self.loss_model.guidance_count),device=self.device)
            tape=wp.Tape();tape.record_launch(b['kernel'],n,0,[job],[b['r']],wp.get_device(self.device),block_dim=256)
            def launch():
                obs.forward(self.engine.compiled.base,b['q'],b['work'])
                wp.launch(fa._prepare_auxiliary,(n,self.loss_model.guidance_count),inputs=[job],device=self.device)
                wp.launch(b['kernel'],n,inputs=[job],outputs=[b['r']],device=self.device)
                if derivative:
                    tape.zero();fa._zero_state_gradients(state)
                    tape.backward(grads={b['r']:b['seeds']})
                    wp.copy(b['direct'],b['q'].grad)
                    for name in ('body_position','body_matrix','contact_distance','contact_position','contact_frame'):
                        wp.copy(getattr(b['seed'],name),getattr(state,name).grad)
                    obs.backward(b['q'],b['work'],recorded=b['recorded'],cotangent=b['seed'])
                    wp.launch(add_q,b['q'].shape,inputs=[b['direct'],b['q'].grad],device=self.device)
            launch();wp.synchronize_device(self.device)
            with wp.ScopedCapture(device=self.device) as capture:launch()
            self.residual_jobs[jobkey]=dict(job=job,tape=tape,graph=capture.graph)
        b['q'].assign(np.broadcast_to(q,b['q'].shape).copy())
        wp.capture_launch(self.residual_jobs[jobkey]['graph'])
        r=b['r'].numpy()[0].astype(float)*.001
        jac=b['q'].grad.numpy().astype(float) if derivative else None
        if not np.isfinite(r).all() or (jac is not None and not np.isfinite(jac).all()):raise FloatingPointError('Nonfinite analytic FA residual')
        self.residual_cache[key]=(r,jac)
        label='FA_residual_jacobian' if derivative else 'FA_residual'
        self.timing[label]=self.timing.get(label,0.)+perf_counter()-t
        return r,jac

    def cost(self,q,stage,derivative=False):
        t=perf_counter();q=np.asarray(q,dtype=np.float32);key=(stage,q.tobytes())
        cached=self.cost_cache.get(key)
        if cached is not None and (not derivative or cached[1] is not None):return cached
        b=self._cost_buffer(stage);b['q'].assign(q[None])
        if derivative not in b['graphs']:
            b['forward']()
            if derivative:b['backward']()
            wp.synchronize_device(self.device)
            with wp.ScopedCapture(device=self.device) as capture:
                b['forward']()
                if derivative:b['backward']()
            b['graphs'][derivative]=capture.graph
        wp.capture_launch(b['graphs'][derivative])
        f=float(b['loss'].loss.numpy()[0])*1e-6
        g=b['q'].grad.numpy()[0].astype(float) if derivative else None
        if not np.isfinite(f) or (g is not None and not np.isfinite(g).all()):raise FloatingPointError('Nonfinite FA stage')
        self.cost_cache[key]=(f,g)
        self.timing['FA_cost_gradient' if derivative else 'FA_cost']=self.timing.get('FA_cost_gradient' if derivative else 'FA_cost',0.)+perf_counter()-t
        return f,g

