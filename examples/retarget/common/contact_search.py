"""Object-side contact shooting: surface samples, cone forces, native ComFree.

This is an ideal contact planner, not the physical hand executor. Its forces
are never injected into the physical rollout. Environment reactions are solved
by the same native hard ComFree implementation used by FA.
"""
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation


def write_object_scene(source, output):
    tree=ET.parse(source);root=tree.getroot()
    # The hand lives in the include. Preserve the original object inertia,
    # free joint, SDF collider, floor, gravity and contact materials.
    for node in list(root):
        if node.tag in ('include','contact','actuator','tendon','sensor','keyframe','equality'):
            root.remove(node)
    world=root.find('worldbody')
    for node in list(world):
        if node.tag=='body' and node.get('name')!='obj':world.remove(node)
    for node in root.iter():
        if node.get('file'):
            path=Path(node.get('file'))
            if not path.is_absolute():node.set('file',str((Path(source).resolve().parent/path).resolve()))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    tree.write(output)
    return output


@wp.kernel(enable_backward=False)
def local_wrench(q: wp.array2d(dtype=float), wrench: wp.array3d(dtype=float),
                 stage: int, output: wp.array2d(dtype=float)):
    world=wp.tid()
    quat=wp.quat(q[world,4],q[world,5],q[world,6],q[world,3])
    force=wp.quat_rotate(quat,wp.vec3(wrench[world,stage,0],wrench[world,stage,1],wrench[world,stage,2]))
    for axis in range(3):
        output[world,axis]=force[axis]
        output[world,axis+3]=wrench[world,stage,axis+3]


class ObjectComFree:
    def __init__(self, scene, *, dt=.01, substeps=6, horizon=5):
        import comfree_warp
        from comfree_warp.collision_config import CollisionConfig,configure_collision
        from comfree_warp.native_adjoint.fast_step import compile_fast_step
        self.device_model,self.model=comfree_warp.load_model(str(Path(scene).resolve()))
        self.model.opt.timestep=dt;self.device_model.opt.timestep.fill_(dt)
        configure_collision(self.device_model,CollisionConfig(1,0.))
        if (self.model.nq,self.model.nv,self.model.nu)!=(7,6,0):
            raise ValueError('Contact shooting requires one free object and fixed environment')
        if np.max(np.abs(self.model.body_ipos[1]))>1e-8:
            raise ValueError('Contact wrench currently requires the object origin at its COM')
        self.compiled=compile_fast_step(self.model,self.device_model)
        self.device=self.compiled.base.device;self.substeps=substeps;self.horizon=horizon;self.buffers={}

    def _buffer(self, worlds):
        if worlds in self.buffers:return self.buffers[worlds]
        from comfree_warp.native_adjoint.fast_step import allocate_workspace,step
        from comfree_warp.native_adjoint.dynamics import DynamicsInput
        from comfree_warp.native_adjoint.runtime import StepInput
        q=wp.array(np.tile(self.model.qpos0,(worlds,1)).astype(np.float32),device=self.device)
        v=wp.zeros((worlds,6),device=self.device)
        u=wp.zeros((worlds,0),device=self.device)
        force=wp.zeros((worlds,6),device=self.device)
        wrench=wp.zeros((worlds,self.horizon,6),device=self.device)
        clock=wp.zeros(worlds,device=self.device);soft=wp.zeros_like(clock)
        work=[allocate_workspace(self.compiled,worlds) for _ in range(self.horizon*self.substeps)]
        def launch():
            pos,vel,time=q,v,clock;poses=[];states=[]
            for i,w in enumerate(work):
                states.append((pos,vel))
                stage=i//self.substeps
                wp.launch(local_wrench,worlds,inputs=[pos,wrench,stage],outputs=[force],device=self.device)
                d=DynamicsInput();d.qpos=pos;d.qvel=vel;d.ctrl=u;d.qfrc_applied=force
                result=step(self.compiled,StepInput(d,time,soft),w)
                pos,vel,time=result.qpos,result.qvel,result.time
                if (i+1)%self.substeps==0:poses.append(pos)
            return poses,states
        poses,states=launch();wp.synchronize()
        with wp.ScopedCapture(device=self.device) as capture:poses,states=launch()
        # CUDA graphs retain device addresses, not Python allocation owners.
        # Keep every workspace/input alive when buffers of different batch
        # sizes are interleaved by finite differences and line searches.
        b=dict(q=q,v=v,wrench=wrench,graph=capture.graph,poses=poses,
               work=work,u=u,force=force,clock=clock,soft=soft,launch=launch,states=states)
        self.buffers[worlds]=b
        return b

    def rollout(self, pose, velocity, wrench):
        wrench=np.asarray(wrench);n=len(wrench);b=self._buffer(n)
        b['q'].assign(np.broadcast_to(pose,(n,7)).astype(np.float32))
        b['v'].assign(np.broadcast_to(velocity,(n,6)).astype(np.float32))
        b['wrench'].assign(wrench.astype(np.float32));wp.capture_launch(b['graph'])
        return np.stack([q.numpy() for q in b['poses']],axis=1).astype(float)


def pose_residual(poses, target, position_weight=1., rotation_weight=.3, reference_scale=100., terminal=2.):
    q=poses[...,3:];q=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
    desired=np.broadcast_to(target[...,3:],q.shape)
    rotation=Rotation.from_quat(q.reshape(-1,4),scalar_first=True)
    goal=Rotation.from_quat(desired.reshape(-1,4),scalar_first=True)
    angle=(goal.inv()*rotation).as_rotvec().reshape(*q.shape[:-1],3)
    result=np.concatenate((position_weight*(poses[...,:3]-target[...,:3]),rotation_weight*angle),axis=-1)
    weights=np.full(poses.shape[-2],6.);weights[-1]+=terminal
    return (1000*np.sqrt(reference_scale)*result*np.sqrt(weights)[None,:,None]).reshape(len(poses),-1)


def project_coefficients(value, active, cap):
    """Nonnegative friction rays, with a per-contact sum/normal-force cap."""
    value=np.maximum(value,0.)
    ordered=np.sort(value,axis=-1)[...,::-1];sums=np.cumsum(ordered,axis=-1)
    k=np.arange(1,5);valid=ordered-(sums-cap)/k>0
    index=np.maximum(valid.sum(axis=-1)-1,0)
    theta=np.take_along_axis((sums-cap)/k,index[...,None],axis=-1)[...,0]
    theta=np.where(value.sum(axis=-1)>cap,theta,0.)
    return np.maximum(value-theta[...,None],0.)*active[...,None]


def force_maps(points, outward, friction):
    inward=-outward
    axis=np.eye(3)[np.argmin(np.abs(inward),axis=-1)]
    t1=np.cross(inward,axis);t1/=np.maximum(np.linalg.norm(t1,axis=-1,keepdims=True),1e-12)
    t2=np.cross(inward,t1)
    rays=inward[...,None,:]+friction*np.stack((t1,t2,-t1,-t2),axis=-2)
    torque=np.cross(points[...,None,:],rays)
    # [batch, stage, generalized wrench, contact * ray]
    mapping=np.concatenate((rays,torque),axis=-1)
    mapping=mapping.reshape(*points.shape[:2],-1,6).swapaxes(-1,-2)
    return rays,mapping


class ConeForceShooting:
    """Sequential quadratic shooting, finite differences through real ComFree.

    Linearize the entire pose trajectory wrt the 6*K body wrenches, then solve
    a projected cone-force quadratic subproblem. Re-evaluate each line-search
    candidate with nonlinear ComFree. Finite iterations give a local solution.
    """
    def __init__(self, model, *, friction=1., force_cap=40., force_weight=.01,
                 iterations=2, qp_iterations=64, pose_weights=(1.,.3,100.,2.)):
        self.model=model;self.friction=friction;self.cap=force_cap;self.reg=force_weight
        self.iterations=iterations;self.qp_iterations=qp_iterations;self.pose_weights=pose_weights
    def support_initialization(self, pose, points, A, active, desired_wrench=None):
        """A gravity-compensating initial guess, never a feasibility test.

        Starting from free fall can put every late finite-difference rollout
        on the floor, hiding the benefit of supporting/lifting the object.
        Evaluate this seed, zero, and warm forces in the true trajectory cost.
        """
        batch,stages,slots=active.shape
        radius=np.maximum(np.max(np.linalg.norm(points,axis=-1),axis=-1),.01)
        scale=np.concatenate((np.ones((batch,stages,3)),np.repeat(1/radius[...,None],3,axis=-1)),axis=-1)
        matrix=A*scale[...,None]*np.repeat(active,4,axis=-1)[:,:,None]
        force=Rotation.from_quat(pose[3:],scalar_first=True).inv().apply(-self.model.model.body_mass[1]*self.model.model.opt.gravity)
        desired=np.zeros((batch,stages,6));desired[:,:,:3]=force
        if desired_wrench is not None:desired[:]=desired_wrench
        desired*=scale
        L=2*np.maximum(np.linalg.eigvalsh(matrix@matrix.swapaxes(-1,-2))[...,-1],1e-8)
        x=np.zeros((batch,stages,slots,4));y=x.copy();momentum=1.
        for _ in range(128):
            error=np.einsum('bkwd,bkd->bkw',matrix,y.reshape(batch,stages,-1))-desired
            gradient=2*np.einsum('bkwd,bkw->bkd',matrix,error).reshape(x.shape)
            new=project_coefficients(y-gradient/L[:,:,None,None],active,self.cap)
            next_momentum=(1+np.sqrt(1+4*momentum*momentum))/2
            y=new+(momentum-1)/next_momentum*(new-x);x=new;momentum=next_momentum
        return x

    def trajectory_initialization(self, pose, velocity, targets):
        """Inverse-dynamics shooting seed; only the true ComFree cost ranks it."""
        dt=self.model.model.opt.timestep*self.model.substeps
        subdt=self.model.model.opt.timestep
        mass=self.model.model.body_mass[1]
        inertia_rotation=Rotation.from_quat(self.model.model.body_iquat[1],scalar_first=True).as_matrix()
        inertia=inertia_rotation@np.diag(self.model.model.body_inertia[1])@inertia_rotation.T
        effective=inertia+np.diag(self.model.model.dof_armature[3:])
        p=pose[:3].copy();r=Rotation.from_quat(pose[3:],scalar_first=True)
        v=velocity[:3].copy();omega=r.apply(velocity[3:]);wrenches=[]
        for target in targets:
            acceleration=2*(target[:3]-p-v*dt)/(dt*(dt+subdt))
            goal=Rotation.from_quat(target[3:],scalar_first=True)
            angular_acceleration=2*((goal*r.inv()).as_rotvec()-omega*dt)/(dt*(dt+subdt))
            local_omega=r.inv().apply(omega)
            force=(mass+self.model.model.dof_armature[:3])*acceleration-mass*self.model.model.opt.gravity
            torque=effective@r.inv().apply(angular_acceleration)+np.cross(local_omega,inertia@local_omega)
            wrenches.append(np.r_[r.inv().apply(force),torque])
            v+=acceleration*dt;omega+=angular_acceleration*dt;p=target[:3];r=goal
        return np.array(wrenches)

    def solve(self, pose, velocity, points, normals, active, targets, warm=None):
        rays,A=force_maps(points,normals,self.friction)
        batch,stages,slots=active.shape
        coeff=np.zeros((batch,stages,slots,4)) if warm is None else np.array(warm,copy=True)
        coeff=project_coefficients(coeff,active,self.cap)
        def wrench(x):return np.einsum('bkwd,bkd->bkw',A,x.reshape(batch,stages,-1))
        def residual(poses):return pose_residual(poses,targets,*self.pose_weights)
        def effort(x):return self.reg*np.sum(np.einsum('bknrj,bknr->bknj',rays,x)**2,axis=(1,2,3))
        def evaluate(x):
            poses=self.model.rollout(pose,velocity,wrench(x))
            r=residual(poses);return np.sum(r*r,axis=1)+effort(x),poses
        cost,predicted=evaluate(coeff);initial=cost.copy();evaluations=1
        if not active.any():
            # All force variables are exactly zero by the demonstration mask.
            # Keep every contact candidate and its real passive pose cost.
            return dict(cost=cost,pose=predicted,coefficients=coeff,
                forces=np.zeros((*active.shape,3)),initial_cost=initial,
                model_evaluations_per_candidate=evaluations)
        support=self.support_initialization(pose,points,A,active)
        desired_wrench=self.trajectory_initialization(pose,velocity,targets)
        tracking=self.support_initialization(pose,points,A,active,desired_wrench=desired_wrench)
        for seed in (support,tracking,np.zeros_like(coeff)):
            seed_cost,seed_pose=evaluate(seed);evaluations+=1
            better=seed_cost<cost
            coeff[better]=seed[better];cost[better]=seed_cost[better];predicted[better]=seed_pose[better]
        dimension=stages*6
        delta=np.tile([.1,.1,.1,.001,.001,.001],stages)
        perturb=np.concatenate((np.zeros((1,dimension)),np.diag(delta),-np.diag(delta))).reshape(-1,stages,6)
        for _ in range(self.iterations):
            net=wrench(coeff)
            all_wrench=(net[:,None]+perturb[None]).reshape(-1,stages,6)
            poses=self.model.rollout(pose,velocity,all_wrench)
            rr=residual(poses).reshape(batch,1+2*dimension,-1);evaluations+=1+2*dimension
            r0=rr[:,0]
            J=((rr[:,1:1+dimension]-rr[:,1+dimension:])/(2*delta[None,:,None])).swapaxes(1,2)
            B=np.einsum('bokw,bkwd->bokd',J.reshape(batch,stages*6,stages,6),A).reshape(batch,stages*6,-1)
            rhs=r0-np.einsum('bod,bd->bo',B,coeff.reshape(batch,-1))
            if not np.isfinite(B).all():
                raise FloatingPointError(f'Contact linearization nonfinite: A={np.isfinite(A).all()}, J={np.isfinite(J).all()}, poses={np.isfinite(poses).all()}, points={np.isfinite(points).all()}, normals={np.isfinite(normals).all()}')
            spectral=np.linalg.eigvalsh(B@B.swapaxes(1,2))[:,-1]
            ray_bound=np.max(np.sum(rays*rays,axis=(-1,-2)),axis=(1,2))
            L=2*np.maximum(spectral+self.reg*ray_bound,1e-8)
            x=coeff.copy();y=x.copy();momentum=1.
            for _ in range(self.qp_iterations):
                error=np.einsum('bod,bd->bo',B,y.reshape(batch,-1))+rhs
                gradient=2*np.einsum('bod,bo->bd',B,error).reshape(coeff.shape)
                f=np.einsum('bknrj,bknr->bknj',rays,y)
                gradient+=2*self.reg*np.einsum('bknrj,bknj->bknr',rays,f)
                new=project_coefficients(y-gradient/L[:,None,None,None],active,self.cap)
                next_momentum=(1+np.sqrt(1+4*momentum*momentum))/2
                y=new+(momentum-1)/next_momentum*(new-x);x=new;momentum=next_momentum
            trials=np.stack([coeff+a*(x-coeff) for a in (0.,1.,.5,.25)],axis=1)
            net_trials=np.einsum('bkwd,bakd->bakw',A,trials.reshape(batch,4,stages,-1))
            qp=self.model.rollout(pose,velocity,net_trials.reshape(-1,stages,6));evaluations+=4
            values=np.sum(residual(qp).reshape(batch,4,-1)**2,axis=-1)
            forces=np.einsum('bknrj,baknr->baknj',rays,trials)
            values+=self.reg*np.sum(forces*forces,axis=(2,3,4))
            best=values.argmin(axis=1);row=np.arange(batch)
            coeff=trials[row,best];cost=values[row,best];predicted=qp.reshape(batch,4,stages,7)[row,best]
        forces=np.einsum('bknrj,bknr->bknj',rays,coeff)
        return dict(cost=cost,pose=predicted,coefficients=coeff,forces=forces,
                    initial_cost=initial,model_evaluations_per_candidate=evaluations)


@wp.kernel(enable_backward=False)
def surface_queries(mesh: wp.uint64, query: wp.array(dtype=wp.vec3),
                    position: wp.array(dtype=wp.vec3), normal: wp.array(dtype=wp.vec3)):
    i=wp.tid();hit=wp.mesh_query_point(mesh,query[i],100.)
    position[i]=query[i];normal[i]=wp.vec3(0.0)
    if hit.result:
        position[i]=wp.mesh_eval_position(mesh,hit.face,hit.u,hit.v)
        normal[i]=wp.mesh_eval_face_normal(mesh,hit.face)


@wp.kernel(enable_backward=False)
def pack_search_state(c: wp.array2d(dtype=wp.vec3), z: wp.array2d(dtype=wp.vec3),
                      dual: wp.array2d(dtype=wp.vec3), template: wp.array(dtype=wp.vec3),
                      mask: wp.array(dtype=float), weight: wp.array(dtype=float), body: wp.array(dtype=int),
                      stage: int, pc: wp.array3d(dtype=wp.vec3), pz: wp.array3d(dtype=wp.vec3),
                      py: wp.array3d(dtype=wp.vec3), pt: wp.array2d(dtype=wp.vec3),
                      pm: wp.array2d(dtype=float), pb: wp.array2d(dtype=int)):
    world,slot=wp.tid()
    pc[world,stage,slot]=c[world,slot];pz[world,stage,slot]=z[world,slot];py[world,stage,slot]=dual[world,slot]
    if world==0:
        pt[stage,slot]=template[slot];pm[stage,slot]=mask[slot]*weight[slot];pb[stage,slot]=body[slot]


@wp.kernel(enable_backward=False)
def apply_search_choice(chosen: wp.array3d(dtype=wp.vec3), mask: wp.array2d(dtype=float), stage: int,
                        c: wp.array2d(dtype=wp.vec3), z: wp.array2d(dtype=wp.vec3),
                        old_z: wp.array2d(dtype=wp.vec3), dual: wp.array2d(dtype=wp.vec3)):
    world,slot=wp.tid()
    old_z[world,slot]=z[world,slot]
    if mask[stage,slot]>0.:
        z[world,slot]=chosen[world,stage,slot]
        dual[world,slot]+=c[world,slot]-z[world,slot]


class ObjectContactSearch:
    def __init__(self, objective, native, cfg, settings):
        self.objective=objective;self.native=native;self.data=mujoco.MjData(native)
        self.cfg=cfg;self.settings=settings;self.last=None;self.phase=0
        model=ObjectComFree(settings.object_scene,dt=cfg.simulator.time_grid.mpc_dt,
            substeps=cfg.simulator.time_grid.knot_substeps,horizon=cfg.optimizer.horizon)
        loss=cfg.loss
        solver=ConeForceShooting
        if settings.contact_solver=='batch-lm':
            from examples.retarget.common.contact_lm import ConeForceBatchLM
            solver=ConeForceBatchLM
        elif settings.contact_solver!='shooting':raise ValueError(f'Unknown contact solver: {settings.contact_solver}')
        self.force_solver=solver(model,friction=settings.contact_friction,
            force_cap=settings.contact_force_cap,force_weight=settings.contact_force_weight,
            iterations=settings.contact_force_iterations,qp_iterations=settings.contact_qp_iterations,
            pose_weights=(loss.obj_pos,loss.obj_rot,loss.reference_pose_scale,loss.terminal))
        self.projection_buffers={};self.search_buffers={}

    def packed_state(self, enabled):
        worlds=self.objective.worlds;dense=len(enabled);slots=enabled[0][1].c.shape[1]
        shape=(worlds,dense,slots)
        if shape not in self.search_buffers:
            device=self.objective.device
            self.search_buffers[shape]=dict(
                **{k:wp.empty(shape,dtype=wp.vec3,device=device) for k in ('c','z','dual','chosen')},
                template=wp.empty((dense,slots),dtype=wp.vec3,device=device),
                mask=wp.empty((dense,slots),dtype=float,device=device),body=wp.empty((dense,slots),dtype=int,device=device))
        b=self.search_buffers[shape]
        for stage,(w,p) in enumerate(enabled):
            t=w.target
            wp.launch(pack_search_state,p.c.shape,inputs=[p.c,p.z,p.dual,t.query_position_object,
                t.query_mask,t.query_weight,t.query_body,stage,b['c'],b['z'],b['dual'],b['template'],b['mask'],b['body']],device=self.objective.device)
        return b

    def project(self, points):
        shape=points.shape;n=points.size//3
        if n not in self.projection_buffers:
            self.projection_buffers[n]=[wp.empty(n,dtype=wp.vec3,device=self.objective.device) for _ in range(3)]
        q,p,normal=self.projection_buffers[n];q.assign(np.asarray(points,np.float32).reshape(-1,3))
        wp.launch(surface_queries,n,inputs=[self.objective.mesh.id,q],outputs=[p,normal],device=self.objective.device)
        positions=p.numpy().reshape(shape).astype(float);normals=normal.numpy().reshape(shape).astype(float)
        if not np.isfinite(positions).all() or np.any(np.linalg.norm(normals,axis=-1)<.5):
            raise ValueError('Object mesh projection did not yield finite points and nondegenerate surface normals')
        return positions,normals

    def begin(self, qpos, qvel, targets, seed, execution_steps):
        self.qpos=np.array(qpos);self.qvel=np.array(qvel);self.targets=targets
        self.seed=seed;self.phase=0;self.execution_steps=execution_steps
        self.last=None
        # Native contact geometry at the measured execution state. Used as an
        # additional candidate, never as a force/closure feasibility gate.
        self.data.qpos[:]=qpos;self.data.qvel[:]=qvel;mujoco.mj_forward(self.native,self.data)
        obj=self.objective.index.object_body
        self.observed=[];R=self.data.xmat[obj].reshape(3,3);origin=self.data.xpos[obj]
        for c in self.data.contact:
            bodies=self.native.geom_bodyid[[c.geom1,c.geom2]]
            if obj not in bodies or c.dist>.001:continue
            other=int(bodies[1] if bodies[0]==obj else bodies[0])
            if other==0:continue
            self.observed.append((other,(c.pos-origin)@R))

    def update(self):
        started=perf_counter()
        obj=self.objective;settings=self.settings
        if self.phase==0 and self.cfg.simulator.exec_backend=='comfree':
            collision=obj.steps[0].workspace.forward.base.collision.contacts
            params=obj.compiled.base.contact_rows.parameters
            pairs=params.contact_geom.numpy();bodies=obj.cpu_model.geom_bodyid[pairs]
            positions=collision.position.numpy()[0];distance=collision.distance.numpy()[0]
            active_collision=collision.active.numpy()[0]
            ob=obj.index.object_body;R=Rotation.from_quat(self.qpos[-4:],scalar_first=True).as_matrix()
            self.observed=[]
            for k in np.flatnonzero(active_collision.astype(bool)&(distance<=.001)):
                if ob not in bodies[k]:continue
                other=int(bodies[k,1] if bodies[k,0]==ob else bodies[k,0])
                if other:self.observed.append((other,(positions[k]-self.qpos[-7:-4])@R))
        enabled=[(w,p) for w,p in zip((*obj.training.immediate,obj.training.terminal),obj.points) if w.job.query_enabled]
        packed=self.packed_state(enabled)
        c=packed['c'].numpy().astype(float);z=packed['z'].numpy().astype(float);dual=packed['dual'].numpy().astype(float)
        template=packed['template'].numpy();mask=packed['mask'].numpy();body=packed['body'].numpy()
        worlds,dense,slots,_=c.shape
        sub=self.cfg.simulator.time_grid.knot_substeps;stages=self.cfg.optimizer.horizon
        end=np.arange(sub-1,dense,sub)
        base,normals=self.project(z[:,end]);original,_=self.project(template[end])
        # Every search contains incumbent, original MANO, a consensus step,
        # measured-contact step and random samples within the local radius.
        samples=settings.contact_samples
        candidates=np.repeat(base[:,None],samples,axis=1)
        candidates[:,1]=original
        center=(c+dual)[:,end]
        delta=center-base;length=np.linalg.norm(delta,axis=-1,keepdims=True)
        candidates[:,2]=base+delta*np.minimum(1.,settings.contact_radius/np.maximum(length,1e-12))
        measured=base.copy()
        for stage in range(stages):
            for slot,b in enumerate(body[end[stage]]):
                points=[p for other,p in self.observed if other==b]
                if not points:continue
                points=np.asarray(points)
                for world in range(worlds):
                    nearest=points[np.argmin(np.linalg.norm(points-base[world,stage,slot],axis=1))]
                    d=nearest-base[world,stage,slot]
                    measured[world,stage,slot]+=d*min(1.,settings.contact_radius/max(np.linalg.norm(d),1e-12))
        candidates[:,3]=measured
        rng=np.random.default_rng(self.seed+10007*self.phase)
        for sample in range(4,samples):
            delta=rng.normal(size=base.shape)
            delta-=np.sum(delta*normals,axis=-1,keepdims=True)*normals
            delta/=np.maximum(np.linalg.norm(delta,axis=-1,keepdims=True),1e-12)
            delta*=settings.contact_radius*rng.random((*base.shape[:-1],1))**.5
            candidates[:,sample]=base+delta
        candidates,outward=self.project(candidates)
        # Projection can cross an edge. Keep local samples within the requested
        # Euclidean neighborhood; the original MANO candidate stays available.
        too_far=np.linalg.norm(candidates-base[:,None],axis=-1)>settings.contact_radius+1e-6
        too_far[:,1]=False
        candidates=np.where(too_far[...,None],base[:,None],candidates)
        candidates,outward=self.project(candidates)
        expanded=np.repeat(candidates,sub,axis=2)
        expander=getattr(self,'candidate_expander',None)
        if expander is not None:
            expanded=expander(candidates)
            candidates=expanded[:,:,end]
            _,outward=self.project(candidates)
        flat=candidates.reshape(worlds*samples,stages,slots,3)
        active=np.broadcast_to(mask[end][None,None]>0,(worlds,samples,stages,slots)).reshape(worlds*samples,stages,slots)
        pose=self.qpos[obj.index.object_qpos:obj.index.object_qpos+7]
        velocity=self.qvel[-6:]
        warm=None
        if self.last is not None:
            memory=self.last['coefficients']
            warm=np.broadcast_to(memory[:,None],(worlds,samples,*memory.shape[1:])).reshape(worlds*samples,stages,slots,4).copy()
        elif getattr(obj,'force_memory',None) is not None:
            memory=obj.force_memory
            warm=np.broadcast_to(memory[None,None],(worlds,samples,*memory.shape)).reshape(worlds*samples,stages,slots,4).copy()
        force_started=perf_counter()
        result=self.force_solver.solve(pose,velocity,flat,outward.reshape(flat.shape),active,
            self.targets.object_qpos[end],warm=warm)
        force_seconds=perf_counter()-force_started
        object_cost=result['cost'].reshape(worlds,samples)
        consensus=.5*settings.rho*np.sum(mask[None,None,...]*np.sum((1000*(expanded-(c+dual)[:,None]))**2,axis=-1),axis=(2,3))
        template_cost=settings.contact_template_weight*np.sum(mask[None,None,...]*
            np.sum((1000*(expanded-template[None,None]))**2,axis=-1),axis=(2,3))
        total=settings.contact_object_weight*object_cost+consensus+template_cost
        choice=np.argmin(total,axis=1);row=np.arange(worlds)
        chosen=expanded[row,choice];chosen_z=candidates[row,choice]
        packed['chosen'].assign(chosen.astype(np.float32))
        for i,(workspace,point) in enumerate(enabled):
            wp.launch(apply_search_choice,point.c.shape,inputs=[packed['chosen'],packed['mask'],i,
                point.c,point.z,point.old_z,point.dual],device=obj.device)
        self.last=dict(object_cost=object_cost[row,choice],sample_cost=object_cost,
            template_shift_cost=template_cost[row,choice],sample_template_shift_cost=template_cost,
            sample_merit=total,chosen=choice,points=chosen_z,candidates=candidates,dense_points=chosen,
            normals=outward[row,choice],forces=result['forces'].reshape(worlds,samples,stages,slots,3)[row,choice],
            coefficients=result['coefficients'].reshape(worlds,samples,stages,slots,4)[row,choice],
            predicted_pose=result['pose'].reshape(worlds,samples,stages,7)[row,choice],
            initial_cost=result['initial_cost'].reshape(worlds,samples)[row,choice],
            model_evaluations_per_candidate=result['model_evaluations_per_candidate'])
        self.last['jacobian']=result.get('jacobian','central_differences')
        self.last['force_seconds']=force_seconds
        residual=np.linalg.norm(c-chosen.astype(np.float32),axis=-1)[:,mask>0]
        self.last['consensus_mean_mm']=float(residual.mean()*1000) if residual.size else 0.
        self.phase+=1
        wp.synchronize_device(obj.device)
        self.last['search_seconds']=perf_counter()-started

    def snapshot(self, selected):
        if self.last is None:return {}
        keys=('object_cost','sample_cost','sample_merit','chosen','points','candidates','normals','forces','coefficients','predicted_pose','initial_cost')
        result={'search_'+k:np.asarray(self.last[k][selected]) for k in keys}
        result['observed_points']=np.zeros((64,3),np.float32)
        result['observed_body']=np.zeros(64,np.int32)
        result['observed_mask']=np.zeros(64,np.int32)
        for i,(body,point) in enumerate(self.observed[:64]):
            result['observed_points'][i]=point;result['observed_body'][i]=body;result['observed_mask'][i]=1
        return result
