"""Batched projected LM with analytic native ComFree trajectory Jacobians.

The independent damping/trust/acceptance policy follows lambda_batched_lm.
Dynamics are the real object/environment ComFree program, not soft SCM.
"""
import numpy as np
import warp as wp
import torch
from dataclasses import replace
from functools import cache

from examples.retarget.common.contact_search import ConeForceShooting, force_maps


@wp.kernel
def applied_force(q: wp.array2d(dtype=float), wrench: wp.array2d(dtype=float),
                  force: wp.array2d(dtype=float)):
    i=wp.tid()
    quat=wp.quat(q[i,4],q[i,5],q[i,6],q[i,3])
    f=wp.quat_rotate(quat,wp.vec3(wrench[i,0],wrench[i,1],wrench[i,2]))
    for j in range(3):
        force[i,j]=f[j]
        force[i,j+3]=wrench[i,j+3]


@wp.kernel(enable_backward=False)
def identity_seeds(q: wp.array2d(dtype=float), v: wp.array2d(dtype=float)):
    i=wp.tid();axis=i%13
    if axis<7:q[i,axis]=1.
    else:v[i,axis-7]=1.


@cache
def support_kernel(slots):
    """Fuse the existing 128-step support-seed fit, without changing its cost."""
    Vector=wp.types.vector(length=slots*4,dtype=wp.float32)
    Vec6=wp.types.vector(length=6,dtype=wp.float32)
    @wp.kernel(enable_backward=False,module='unique')
    def fit(matrix: wp.array3d(dtype=float), goal: wp.array2d(dtype=float),
            active: wp.array2d(dtype=float), lipschitz: wp.array(dtype=float),
            cap: float, output: wp.array3d(dtype=float)):
        b=wp.tid();x=Vector(0.);y=Vector(0.);momentum=float(1.)
        for iteration in range(128):
            error=Vec6(0.)
            for row in range(6):
                value=-goal[b,row]
                for col in range(wp.static(slots*4)):value+=matrix[b,row,col]*y[col]
                error[row]=value
            new=Vector(0.)
            for slot in range(wp.static(slots)):
                proposal=wp.vec4(0.)
                for ray in range(4):
                    col=slot*4+ray;gradient=float(0.)
                    for row in range(6):gradient+=matrix[b,row,col]*error[row]
                    proposal[ray]=wp.max(y[col]-2.*gradient/lipschitz[b],0.)
                ordered=wp.vec4(proposal)
                # Fixed-size descending insertion sort.
                for i in range(1,4):
                    for reverse in range(i):
                        j=i-reverse
                        if ordered[j]>ordered[j-1]:
                            temp=ordered[j-1];ordered[j-1]=ordered[j];ordered[j]=temp
                total=float(0.);theta=float(0.)
                for i in range(4):
                    total+=ordered[i]
                    test=(total-cap)/float(i+1)
                    if ordered[i]>test:theta=test
                if total<=cap:theta=0.
                for ray in range(4):new[slot*4+ray]=wp.max(proposal[ray]-theta,0.)*active[b,slot]
            next_momentum=(1.+wp.sqrt(1.+4.*momentum*momentum))/2.
            y=new+(momentum-1.)/next_momentum*(new-x);x=new;momentum=next_momentum
        for slot in range(wp.static(slots)):
            for ray in range(4):output[b,slot,ray]=x[slot*4+ray]
    return fit


class AnalyticObjectRollout:
    """One native reverse sweep per output, batched over all trajectory steps.

    First run the true trajectory. Evaluate its local 13x19 step Jacobians
    in parallel, then compose state sensitivities along the trajectory.
    All local derivatives come from ComFree's native segmented adjoint.
    """
    def __init__(self, model):
        import torch
        self.torch=torch;self.model=model;self.buffers={}
        self.device=torch.device(str(model.device))
        # Both libraries must enqueue work on the same CUDA stream.
        self.stream=torch.cuda.ExternalStream(wp.get_stream(model.device).cuda_stream,device=self.device)

    def tensor(self, value):
        return self.torch.as_tensor(np.array(value,copy=True),dtype=self.torch.float32,device=self.device)

    @torch.no_grad()
    def rollout(self, pose, velocity, wrench, states=False):
        torch=self.torch;batch=len(wrench);b=self.model._buffer(batch)
        b['q'].assign(np.broadcast_to(pose,(batch,7)).astype(np.float32))
        b['v'].assign(np.broadcast_to(velocity,(batch,6)).astype(np.float32))
        wp.copy(b['wrench'],wp.from_torch(wrench.contiguous()))
        wp.capture_launch(b['graph'])
        poses=torch.stack([wp.to_torch(q) for q in b['poses']],dim=1)
        if not states:return poses
        q=torch.stack([wp.to_torch(s[0]) for s in b['states']],dim=1)
        v=torch.stack([wp.to_torch(s[1]) for s in b['states']],dim=1)
        return poses,q,v

    @torch.no_grad()
    def _buffer(self, count):
        if count in self.buffers:return self.buffers[count]
        from comfree_warp.native_adjoint.dynamics import DynamicsInput
        from comfree_warp.native_adjoint.runtime import StepInput
        from comfree_warp.native_adjoint.fast_step_adjoint import allocate_workspace,record,forward,backward,BackwardCall
        from forceaware.optimizer import _allocate_cotangent
        from comfree_warp.native_adjoint.geometry_vjp import GeometryTape
        from comfree_warp.native_adjoint import gaussian_collision
        from comfree_warp.native_adjoint.runtime import collision_contacts
        class FullContactGeometryTape(GeometryTape):
            def record_launch(self,kernel,dim,max_blocks,inputs,outputs,device,block_dim=0,metadata=None):
                if kernel in (gaussian_collision._narrow_pairs,gaussian_collision._narrow_plane):
                    # The FA hand optimizer intentionally stops hard-GS normal
                    # gradients. LM needs the full piecewise trajectory Jacobian.
                    # Only this private derivative tape enables that path;
                    # forward collisions and the hand's tapes stay untouched.
                    inputs[1].stop_frame_vjp=0
                super().record_launch(kernel,dim,max_blocks,inputs,outputs,device,block_dim=block_dim,metadata=metadata)
        q=wp.zeros((count,7),device=self.model.device,requires_grad=True)
        v=wp.zeros((count,6),device=self.model.device,requires_grad=True)
        wrench=wp.zeros_like(v,requires_grad=True);force=wp.zeros_like(v,requires_grad=True)
        wp.to_torch(q)[:,3]=1.
        u=wp.zeros((count,0),device=self.model.device,requires_grad=True)
        clock=wp.zeros(count,device=self.model.device,requires_grad=True)
        soft=wp.zeros_like(clock,requires_grad=True)
        d=DynamicsInput();d.qpos=q;d.qvel=v;d.ctrl=u;d.qfrc_applied=force
        inputs=StepInput(d,clock,soft);work=allocate_workspace(self.model.compiled,count)
        tape=wp.Tape()
        with tape:wp.launch(applied_force,count,inputs=[q,wrench],outputs=[force],device=self.model.device)
        recorded=record(self.model.compiled,inputs,work)
        full_geometry=FullContactGeometryTape()
        with full_geometry:collision_contacts(self.model.compiled.base,work.forward.base,freeze_frame_vjp=False)
        recorded=replace(recorded,contact_tape=full_geometry)
        cotangent=_allocate_cotangent(self.model.compiled,count)
        wp.launch(identity_seeds,count,inputs=[cotangent.qpos,cotangent.qvel],device=self.model.device)
        def launch():
            tape.zero()
            wp.launch(applied_force,count,inputs=[q,wrench],outputs=[force],device=self.model.device)
            forward(self.model.compiled,inputs,work)
            backward(BackwardCall(self.model.compiled,inputs,work,recorded,cotangent))
            tape.backward()
        launch();wp.synchronize_device(self.model.device)
        with wp.ScopedCapture(device=self.model.device) as capture:launch()
        result=dict(q=q,v=v,wrench=wrench,force=force,u=u,clock=clock,soft=soft,
                    inputs=inputs,work=work,tape=tape,recorded=recorded,cotangent=cotangent,graph=capture.graph)
        self.buffers[count]=result
        return result

    @torch.no_grad()
    def local_jacobians(self, q, v, wrench):
        torch=self.torch;batch,steps,_=q.shape
        b=self._buffer(batch*steps*13)
        for key,value in [('q',q),('v',v),('wrench',wrench)]:
            expanded=value[:,:,None,:].expand(-1,-1,13,-1).contiguous().reshape(-1,value.shape[-1])
            wp.copy(b[key],wp.from_torch(expanded))
        wp.capture_launch(b['graph'])
        qj=wp.to_torch(b['q'].grad).reshape(batch,steps,13,7)
        vj=wp.to_torch(b['v'].grad).reshape(batch,steps,13,6)
        fj=wp.to_torch(b['wrench'].grad).reshape(batch,steps,13,6)
        return torch.cat((qj,vj),dim=-1),fj

    @torch.no_grad()
    def trajectory_jacobian(self, q, v, wrench):
        torch=self.torch;batch,steps,_=q.shape;stages=self.model.horizon;sub=self.model.substeps
        dense=wrench.repeat_interleave(sub,dim=1)
        A,B=self.local_jacobians(q,v,dense)
        sensitivity=torch.zeros((batch,13,stages*6),device=self.device)
        output=[]
        for step in range(steps):
            sensitivity=A[:,step]@sensitivity
            stage=step//sub
            sensitivity[:,:,stage*6:(stage+1)*6]+=B[:,step]
            if (step+1)%sub==0:output.append(sensitivity[:,:7].clone())
        return torch.stack(output,dim=1)


def pose_terms(torch, poses, target, weights):
    """Original geodesic FA pose residual and its closed-form pose Jacobian."""
    # Small-angle quaternion costs are sensitive to cancellation in FP32.
    # Keep these small residual calculations in FP64; physics/LM stay FP32.
    poses=poses.to(torch.float64);target=target.to(torch.float64)
    position,rotation,reference,terminal=weights
    batch,stages,_=poses.shape
    q=poses[...,3:];norm=torch.linalg.vector_norm(q,dim=-1,keepdim=True)
    q=q/norm.clamp_min(1.e-12)
    d=target[...,3:];d=d/torch.linalg.vector_norm(d,dim=-1,keepdim=True).clamp_min(1.e-12)
    w,x,y,z=d.unbind(-1)
    # Left quaternion product conj(target) * current, scalar first.
    left=torch.stack((w,x,y,z,-x,w,z,-y,-y,-z,w,x,-z,y,-x,w),dim=-1).reshape(stages,4,4)
    relative=(left[None]@q[...,None]).squeeze(-1)
    sign=torch.where(relative[...,:1]<0.,-1.,1.)
    relative=relative*sign
    scalar=relative[...,:1];vector=relative[...,1:]
    length=torch.linalg.vector_norm(vector,dim=-1,keepdim=True)
    angle=2*torch.atan2(length,scalar)
    small=length<1.e-4
    factor=torch.where(small,2/scalar.clamp_min(1.e-12)-2*length**2/(3*scalar.clamp_min(1.e-12)**3),angle/length.clamp_min(1.e-12))
    denominator=(scalar**2+length**2).clamp_min(1.e-12)
    extra=torch.where(small,-4/(3*scalar.clamp_min(1.e-12)**3),(2*scalar/denominator-factor)/length.square().clamp_min(1.e-12))
    eye3=torch.eye(3,device=poses.device,dtype=poses.dtype)
    dv=factor[...,None]*eye3+extra[...,None]*vector[...,None]*vector[...,None,:]
    dw=-2*vector/denominator
    derivative=torch.cat((dw[...,None],dv),dim=-1)
    normalize=(torch.eye(4,device=poses.device,dtype=poses.dtype)-q[...,None]*q[...,None,:])/norm[...,None].clamp_min(1.e-12)
    dq=(derivative*sign[...,None])@left[None]@normalize
    stage_weights=torch.full((stages,),6.,device=poses.device,dtype=poses.dtype);stage_weights[-1]+=terminal
    scale=1000*(reference*stage_weights).sqrt()
    residual=torch.cat((position*(poses[...,:3]-target[...,:3]),rotation*factor*vector),dim=-1)*scale[None,:,None]
    jac=torch.zeros((batch,stages,6,7),device=poses.device,dtype=poses.dtype)
    jac[...,:3,:3]=position*eye3
    jac[...,3:,3:]=rotation*dq
    jac*=scale[None,:,None,None]
    return residual.reshape(batch,-1),jac


class ConeForceBatchLM(ConeForceShooting):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.analytic=AnalyticObjectRollout(self.model);self.support_buffers={}

    def initializations(self,pose,velocity,points,mapping,active,targets):
        from scipy.spatial.transform import Rotation
        batch,stages,slots=active.shape
        radius=np.maximum(np.max(np.linalg.norm(points,axis=-1),axis=-1),.01)
        scale=np.concatenate((np.ones((batch,stages,3)),np.repeat(1/radius[...,None],3,axis=-1)),axis=-1)
        matrix=mapping*scale[...,None]*np.repeat(active,4,axis=-1)[:,:,None]
        support=np.zeros((batch,stages,6))
        support[...,:3]=Rotation.from_quat(pose[3:],scalar_first=True).inv().apply(-self.model.model.body_mass[1]*self.model.model.opt.gravity)
        desired=self.trajectory_initialization(pose,velocity,targets)
        tracking=np.broadcast_to(desired,(batch,stages,6))
        goals=np.concatenate((support*scale,tracking*scale)).reshape(-1,6)
        matrices=np.concatenate((matrix,matrix)).reshape(-1,6,slots*4)
        masks=np.concatenate((active,active)).reshape(-1,slots)
        L=2*np.maximum(np.linalg.eigvalsh(matrix@matrix.swapaxes(-1,-2))[...,-1],1.e-8)
        L=np.concatenate((L,L)).reshape(-1)
        key=(batch,stages,slots)
        if key not in self.support_buffers:
            def allocate(value):return wp.empty(value.shape,dtype=float,device=self.model.device)
            self.support_buffers[key]=dict(matrix=allocate(matrices),goal=allocate(goals),active=allocate(masks),
                L=allocate(L),output=wp.empty((len(goals),slots,4),device=self.model.device))
        b=self.support_buffers[key]
        for name,value in [('matrix',matrices),('goal',goals),('active',masks),('L',L)]:b[name].assign(value.astype(np.float32))
        wp.launch(support_kernel(slots),len(goals),inputs=[b['matrix'],b['goal'],b['active'],b['L'],self.cap,b['output']],device=self.model.device)
        seeds=wp.to_torch(b['output']).reshape(2,batch,stages,slots,4)
        return seeds[0],seeds[1]

    def solve(self,pose,velocity,points,normals,active,targets,warm=None):
        import torch
        # Native Warp graphs and Torch linear algebra share one stream.
        with torch.cuda.stream(self.analytic.stream),torch.no_grad():
            return self._solve(pose,velocity,points,normals,active,targets,warm)

    def _solve(self,pose,velocity,points,normals,active,targets,warm):
        import torch
        api=self.analytic;batch,stages,slots=active.shape
        rays_np,maps_np=force_maps(points,normals,self.friction)
        rays=api.tensor(rays_np);maps=api.tensor(maps_np);mask=api.tensor(active)
        target=torch.as_tensor(np.array(targets,copy=True),device=api.device,dtype=torch.float64)
        def project(x):
            x=x.clamp_min(0.)
            ordered=x.sort(dim=-1,descending=True).values
            threshold=(ordered.cumsum(-1)-self.cap)/torch.arange(1,5,device=api.device)
            index=(ordered>threshold).sum(-1).clamp_min(1)-1
            theta=threshold.gather(-1,index[...,None]).clamp_min(0.)
            return (x-theta).clamp_min(0.)*mask[...,None]
        def wrench(x):return (maps@x.reshape(batch,stages,-1,1)).squeeze(-1)
        def force(x):return torch.einsum('bknrj,bknr->bknj',rays,x)
        def evaluate(x,states=False):
            result=api.rollout(pose,velocity,wrench(x),states=states)
            poses=result[0] if states else result
            residual,_=pose_terms(torch,poses,target,self.pose_weights)
            cost=residual.square().sum(-1)+self.reg*force(x).square().sum((1,2,3))
            return cost,result,residual
        coeff=project(api.tensor(np.zeros((*active.shape,4)) if warm is None else warm))
        cost,predicted,_=evaluate(coeff);initial=cost.clone();evaluations=1
        accepted=torch.zeros(batch,device=api.device,dtype=torch.int32)
        if active.any():
            # Preserve the existing initial guesses for a controlled solver comparison.
            support,tracking=self.initializations(pose,velocity,points,maps_np,active,targets)
            for seed in (support,tracking,torch.zeros_like(coeff)):
                newcost,newpose,_=evaluate(seed);evaluations+=1
                better=newcost<cost
                coeff=torch.where(better[:,None,None,None],seed,coeff)
                cost=torch.minimum(cost,newcost)
                predicted=torch.where(better[:,None,None],newpose,predicted)
            # Remove exactly inactive variables, preserving each candidate's mask.
            indices=np.flatnonzero(np.repeat(active.any(axis=0).reshape(-1),4))
            ids=torch.as_tensor(indices,device=api.device)
            dimension=stages*slots*4
            mapping=torch.einsum('kl,bkwd->bkwld',torch.eye(stages,device=api.device),maps).reshape(batch,stages*6,dimension)[:,:,ids]
            ray_gram=torch.einsum('bknrj,bknsj->bknrs',rays,rays).reshape(batch,stages*slots,4,4)
            regularizer=self.reg*torch.einsum('ij,birs->birjs',torch.eye(stages*slots,device=api.device),ray_gram).reshape(batch,dimension,dimension)
            regularizer=regularizer[:,ids][:,:,ids]
            damping=torch.full((batch,),1.e-3,device=api.device)
            trust=torch.full((batch,),4.,device=api.device)
            for _ in range(self.iterations):
                net=wrench(coeff)
                cost,current,r=evaluate(coeff,states=True);evaluations+=1
                predicted,q,v=current
                pose_jac=api.trajectory_jacobian(q,v,net)
                _,residual_jac=pose_terms(torch,predicted,target,self.pose_weights)
                J=(residual_jac.float()@pose_jac).reshape(batch,stages*6,stages*6)@mapping
                x=coeff.reshape(batch,-1)[:,ids]
                gram=J.transpose(-1,-2)@J+regularizer
                gradient=(J.transpose(-1,-2)@r.float()[...,None]).squeeze(-1)+(regularizer@x[...,None]).squeeze(-1)
                matrix=gram+torch.diag_embed(damping[:,None]*gram.diagonal(dim1=-2,dim2=-1).clamp_min(1.e-6)+1.e-7)
                factor,info=torch.linalg.cholesky_ex(matrix,check_errors=False)
                delta=torch.linalg.solve_triangular(factor,-gradient[...,None],upper=False)
                delta=torch.linalg.solve_triangular(factor.transpose(-1,-2),delta,upper=True).squeeze(-1)
                good=(info==0)&torch.isfinite(delta).all(-1)
                delta=torch.where(good[:,None],delta,0.)
                delta*=torch.minimum(torch.ones_like(trust),trust/torch.linalg.vector_norm(delta,dim=-1).clamp_min(1.e-12))[:,None]
                proposal=coeff.reshape(batch,-1).clone();proposal[:,ids]=x+delta
                proposal=project(proposal.reshape_as(coeff))
                step=proposal.reshape(batch,-1)[:,ids]-x
                reduction=-2*(gradient*step).sum(-1)-(step*(gram@step[...,None]).squeeze(-1)).sum(-1)
                nextcost,nextpose,_=evaluate(proposal);evaluations+=1
                actual=cost-nextcost
                accept=good&torch.isfinite(nextcost)&(actual>0)&(reduction>0)
                ratio=actual/reduction.clamp_min(1.e-20)
                coeff=torch.where(accept[:,None,None,None],proposal,coeff)
                predicted=torch.where(accept[:,None,None],nextpose,predicted)
                cost=torch.where(accept,nextcost,cost);accepted+=accept.to(torch.int32)
                damping=torch.where(accept&(ratio>.75),damping/3,torch.where(accept&(ratio>=.25),damping,damping*10)).clamp(1.e-9,1.e12)
                trust=torch.where(accept&(ratio>.75),trust*2,torch.where(accept&(ratio>=.25),trust,trust*.5)).clamp(1.e-6,2*self.cap)
        def cpu(x):return x.cpu().numpy().astype(float)
        return dict(cost=cpu(cost),pose=cpu(predicted),coefficients=cpu(coeff),forces=cpu(force(coeff)),
            initial_cost=cpu(initial),model_evaluations_per_candidate=evaluations,
            jacobian='analytic_native_ComFree_adjoint',accepted_lm_steps=cpu(accepted))
