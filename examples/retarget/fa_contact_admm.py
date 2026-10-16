"""ADMM solver for FA with variable object-side template contact points.

Geometric mode: min FA(u,c) + I_surface(z), subject to c=z.
Force mode adds the object-side ComFree trajectory/force cost G(z).
MANO initializes c,z and remains a candidate, without an attraction cost.
The hand block supports batched L-BFGS, SciPy SLSQP and condensed acados SQP/RTI.
Original FA functions, ComFree dynamics and native adjoints are reused.
"""
from dataclasses import dataclass, replace
from time import perf_counter
import mujoco
import numpy as np
import warp as wp

from forceaware import mpc_loss as fa
from forceaware.optimizer import WindowOptimizer, Plan
from forceaware.servo_prior import ServoPriorOptimizer
from forceaware.optimizer_contract import finalize_candidate_evaluation


@dataclass(frozen=True)
class ADMMSettings:
    rho: float = 1.0
    outer: int = 4
    history: int = 5
    line_search: int = 8
    control_scale: float = .001
    balanced_coordinates: bool = True
    optimize_contacts: bool = True
    contact_mode: str = 'geometric'
    object_scene: str = ''
    contact_samples: int = 7
    contact_radius: float = .005
    contact_friction: float = 1.
    contact_force_cap: float = 40.
    contact_force_weight: float = .01
    contact_force_iterations: int = 2
    contact_qp_iterations: int = 64
    contact_object_weight: float = 1.
    contact_template_weight: float = 0.
    contact_solver: str = 'shooting'
    hand_solver: str = 'lbfgs'
    hand_ftol: float = 1.e-6
    reuse_forward: bool = True
    hand_rti_iterations: int = 1
    fuse_adjoint_zero: bool = True
    fixed_hand_single_block: bool = False
    evaluation_backend: str = 'native'
    compiled_cache: str = ''
    compiled_threads: int = 8
    memoize_evaluations: bool = False


@wp.struct
class PointState:
    c: wp.array2d(dtype=wp.vec3)
    z: wp.array2d(dtype=wp.vec3)
    dual: wp.array2d(dtype=wp.vec3)
    old_z: wp.array2d(dtype=wp.vec3)
    last_c: wp.array2d(dtype=wp.vec3)
    mesh: wp.uint64
    rho: float


@wp.func
def _project(mesh: wp.uint64, point: wp.vec3):
    query=wp.mesh_query_point(mesh,point,100.0)
    result=point
    if query.result:
        result=wp.mesh_eval_position(mesh,query.face,query.u,query.v)
    return result


@wp.kernel(enable_backward=False)
def initialize_points(job: fa.LossJob, points: PointState, project: int):
    world,slot=wp.tid()
    initial=job.target.query_position_object[slot]
    if project==1:initial=_project(points.mesh,initial)
    points.c[world,slot]=initial
    points.z[world,slot]=initial
    points.old_z[world,slot]=initial
    points.last_c[world,slot]=initial
    points.dual[world,slot]=wp.vec3(0.0)


@wp.func
def _hand_point(job: fa.LossJob, world: int, slot: int):
    body=job.target.query_body[slot]
    return job.state.body_position[world,body]+job.state.body_matrix[world,body]@job.target.query_local[slot]


@wp.kernel(enable_backward=False)
def variable_auxiliary(job: fa.LossJob, points: PointState):
    world,slot=wp.tid()
    obj=job.model.object_body
    target=job.state.body_position[world,obj]+job.state.body_matrix[world,obj]@points.c[world,slot]
    error=(_hand_point(job,world,slot)-target)*fa.MILLIMETERS
    distance=wp.sqrt(wp.dot(error,error)+fa.LOSS_EPSILON)
    argument=(distance-job.weights.marked_distance_mm)/wp.max(job.weights.marked_temperature_mm,fa.TEMPERATURE_EPSILON)
    job.auxiliary.marked_query_weight[world,slot]=fa._sigmoid(argument)


@wp.kernel
def variable_FA_loss(job: fa.LossJob, points: PointState, output: wp.array(dtype=float)):
    world=wp.tid()
    value=job.reference_scale*fa._reference_pose_loss(job,world)
    obj=job.model.object_body
    for slot in range(job.model.guidance_count):
        if job.guidance_enabled==1:
            value+=fa._physical_contact_loss(job,world,slot)
        if job.query_enabled==1:
            target=job.state.body_position[world,obj]+job.state.body_matrix[world,obj]@points.c[world,slot]
            error=(_hand_point(job,world,slot)-target)*fa.MILLIMETERS
            mask=job.target.query_mask[slot]*job.target.query_weight[slot]
            marked=job.auxiliary.marked_query_weight[world,slot]
            value+=job.weights.marked_query*mask*marked*wp.dot(error,error)
    if job.penetration_enabled==1:
        value+=fa._penetration_loss(job,world)
    output[world]=value+fa._finger_joint_collision_loss(job,world)


@wp.func
def _point_prox_value(r: float, distance: float, weight: float, rho: float,
                      marked_distance: float, temperature: float):
    s=fa._sigmoid((wp.sqrt(r*r+fa.LOSS_EPSILON)-marked_distance)/temperature)
    return weight*s*r*r+0.5*rho*(distance-r)*(distance-r)


@wp.kernel(enable_backward=False)
def point_subproblem(job: fa.LossJob, points: PointState):
    """Global 1-D grid plus local refinement of the FA marked-cost proximal map."""
    world,slot=wp.tid()
    mask=job.target.query_mask[slot]*job.target.query_weight[slot]
    if mask>0.0 and job.query_enabled==1:
        obj=job.model.object_body
        x=wp.transpose(job.state.body_matrix[world,obj])@(_hand_point(job,world,slot)-job.state.body_position[world,obj])
        v=points.z[world,slot]-points.dual[world,slot]
        direction=v-x
        distance=wp.length(direction)*1000.0
        weight=job.weights.marked_query
        temp=wp.max(job.weights.marked_temperature_mm,fa.TEMPERATURE_EPSILON)
        best=int(0);best_value=float(1.e30)
        for k in range(129):
            r=distance*float(k)/128.0
            value=_point_prox_value(r,distance,weight,points.rho,job.weights.marked_distance_mm,temp)
            if value<best_value:
                best_value=value;best=k
        lo=distance*float(wp.max(best-1,0))/128.0
        hi=distance*float(wp.min(best+1,128))/128.0
        for k in range(24):
            a=lo+(hi-lo)*0.38196601125
            b=lo+(hi-lo)*0.61803398875
            va=_point_prox_value(a,distance,weight,points.rho,job.weights.marked_distance_mm,temp)
            vb=_point_prox_value(b,distance,weight,points.rho,job.weights.marked_distance_mm,temp)
            if va<vb:hi=b
            else:lo=a
        r=0.5*(lo+hi)
        fraction=float(0.0)
        if distance>1.e-10:fraction=r/distance
        points.c[world,slot]=x+fraction*direction


@wp.kernel(enable_backward=False)
def surface_dual_update(job: fa.LossJob, points: PointState):
    world,slot=wp.tid()
    if job.target.query_mask[slot]*job.target.query_weight[slot]>0.0:
        c=points.c[world,slot];y=points.dual[world,slot]
        z=_project(points.mesh,c+y)
        points.old_z[world,slot]=points.z[world,slot]
        points.z[world,slot]=z
        points.dual[world,slot]=y+c-z


@wp.kernel(enable_backward=False)
def use_surface_points(points: PointState):
    world,slot=wp.tid()
    points.last_c[world,slot]=points.c[world,slot]
    points.c[world,slot]=points.z[world,slot]


def object_mesh(native,index,device):
    from scipy.spatial.transform import Rotation
    vertices=[];faces=[]
    for geom in np.flatnonzero((native.geom_bodyid==index.object_body)&
            (native.geom_type==mujoco.mjtGeom.mjGEOM_MESH)&
            ((native.geom_contype!=0)|(native.geom_conaffinity!=0))):
        mid=native.geom_dataid[geom];va=native.mesh_vertadr[mid];vn=native.mesh_vertnum[mid]
        fa0=native.mesh_faceadr[mid];fn=native.mesh_facenum[mid]
        rot=Rotation.from_quat(native.geom_quat[geom],scalar_first=True).as_matrix()
        faces.append(native.mesh_face[fa0:fa0+fn]+sum(len(v) for v in vertices))
        vertices.append(native.mesh_vert[va:va+vn]@rot.T+native.geom_pos[geom])
    if not vertices:raise ValueError('No object triangle collision mesh')
    return wp.Mesh(wp.array(np.concatenate(vertices).astype(np.float32),dtype=wp.vec3,device=device),
                   wp.array(np.concatenate(faces).astype(np.int32).reshape(-1),dtype=int,device=device))


class VariableContactObjective(fa.MpcObjective):
    def __init__(self,*args,native_model,admm,**kwargs):
        super().__init__(*args,**kwargs)
        self.admm=admm;self.mesh=object_mesh(native_model,self.index,self.device);self.points=[];workspaces=[]
        for workspace in (*self.training.immediate,self.training.terminal):
            points=PointState();shape=(self.worlds,workspace.job.model.guidance_count)
            for name in ['c','z','dual','old_z','last_c']:
                setattr(points,name,wp.zeros(shape,dtype=wp.vec3,device=self.device,requires_grad=name=='c'))
            points.mesh=self.mesh.id;points.rho=admm.rho;self.points.append(points)
            # Reuse the exact FA backward plumbing with the variable-target loss.
            tape=wp.Tape()
            tape.record_launch(variable_FA_loss,self.worlds,0,[workspace.job,points],[workspace.loss],wp.get_device(self.device),block_dim=256)
            workspaces.append(replace(workspace,tape=tape))
        self.training=replace(self.training,immediate=tuple(workspaces[:-1]),terminal=workspaces[-1])
        self.contact_memory=None;self.force_memory=None;self.contact_search=None
        if admm.contact_mode not in ('geometric','comfree-force'):
            raise ValueError(f'Unknown contact mode: {admm.contact_mode}')
        if admm.contact_mode=='comfree-force':
            if not admm.optimize_contacts:raise ValueError('ComFree force mode requires contact optimization')
            if admm.contact_samples<4:raise ValueError('Need incumbent, MANO, consensus and observed-contact candidates')
            if not np.isfinite([admm.contact_radius,admm.contact_force_cap,admm.contact_friction,admm.contact_force_weight,admm.contact_object_weight]).all():
                raise ValueError('Contact search parameters must be finite')
            if admm.contact_radius<=0 or admm.contact_force_cap<=0:raise ValueError('Positive contact radius and force cap required')
            if not np.isfinite(admm.contact_template_weight) or admm.contact_template_weight<0:
                raise ValueError('Template displacement weight must be finite and nonnegative')
            if min(admm.contact_friction,admm.contact_force_weight,admm.contact_object_weight)<0:
                raise ValueError('Contact friction and objective weights must be nonnegative')
            if min(admm.contact_force_iterations,admm.contact_qp_iterations)<1:
                raise ValueError('Positive force-shooting and quadratic iteration counts required')
            from examples.retarget.comfree_contact_search import ObjectContactSearch
            self.contact_search=ObjectContactSearch(self,native_model,self.cfg,admm)

    def prepare_window(self,*args,**kwargs):
        super().prepare_window(*args,**kwargs)
        for workspace,points in zip((*self.training.immediate,self.training.terminal),self.points):
            wp.launch(initialize_points,dim=points.c.shape,inputs=[workspace.job,points,int(self.admm.optimize_contacts)],device=self.device)
        self.warm_contacts_restored=0
        if self.contact_search is not None and self.contact_memory is not None:
            memory=self.contact_memory
            enabled=[(w,p) for w,p in zip((*self.training.immediate,self.training.terminal),self.points) if w.job.query_enabled]
            for i,(w,p) in enumerate(enabled):
                j=min(i+self.memory_shift,len(enabled)-1)
                keep=(w.target.query_mask.numpy()>0)&(memory['mask'][j]>0)&(w.target.query_body.numpy()==memory['body'][j])
                keep&=np.linalg.norm(w.target.query_local.numpy()-memory['hand_local'][j],axis=-1)<1e-6
                for name in ('c','z','dual'):
                    value=getattr(p,name).numpy();value[:,keep]=memory[name][j,keep]
                    getattr(p,name).assign(value)
                self.warm_contacts_restored+=int(keep.sum())

    def remember_contacts(self, selected, execution_steps):
        if self.contact_search is None:return
        self.contact_memory=self.snapshot(selected);self.memory_shift=execution_steps
        # Force knots are shifted at the actual 30 ms execution interval,
        # independently of the 60 ms force/control knot spacing.
        force=self.contact_search.last['coefficients'][selected]
        grid=np.arange(len(force));shift=execution_steps/self.cfg.simulator.time_grid.knot_substeps
        self.force_memory=np.stack([np.interp(grid+shift,grid,force[:,slot,ray])
            for slot in range(force.shape[1]) for ray in range(4)],axis=-1).reshape(force.shape)

    def _evaluate_trajectory(self,runtime):
        for column,(workspace,points) in enumerate(zip((*runtime.immediate,runtime.terminal),self.points)):
            wp.launch(variable_auxiliary,dim=points.c.shape,inputs=[workspace.job,points],device=self.device)
            wp.launch(variable_FA_loss,dim=self.worlds,inputs=[workspace.job,points],outputs=[workspace.loss],device=self.device)
            wp.launch(fa._store_trajectory_loss,dim=self.worlds,inputs=[workspace.loss,column],outputs=[runtime.trajectory_terms],device=self.device)

    def update_contacts(self):
        for workspace,points in zip((*self.training.immediate,self.training.terminal),self.points):
            if workspace.job.query_enabled:
                wp.launch(point_subproblem,dim=points.c.shape,inputs=[workspace.job,points],device=self.device)

    def project_dual(self):
        if self.contact_search is not None:
            self.contact_search.update();return
        for workspace,points in zip((*self.training.immediate,self.training.terminal),self.points):
            if workspace.job.query_enabled:
                wp.launch(surface_dual_update,dim=points.c.shape,inputs=[workspace.job,points],device=self.device)

    def polish_surface(self):
        for points in self.points:
            wp.launch(use_surface_points,dim=points.c.shape,inputs=[points],device=self.device)

    def snapshot(self,selected):
        rows=[]
        for workspace,points in zip((*self.training.immediate,self.training.terminal),self.points):
            if not workspace.job.query_enabled:continue
            job=workspace.job;bp=job.state.body_position.numpy()[selected];bm=job.state.body_matrix.numpy()[selected]
            body=job.target.query_body.numpy();local=job.target.query_local.numpy()
            h=bp[body]+np.einsum('nij,nj->ni',bm[body],local)
            x=(h-bp[self.index.object_body])@bm[self.index.object_body]
            rows.append(dict(c=points.c.numpy()[selected],z=points.z.numpy()[selected],x=x,
                c_before_surface_polish=points.last_c.numpy()[selected],
                dual=points.dual.numpy()[selected],body=body,hand_local=local,
                template=job.target.query_position_object.numpy(),template_hand_local=local,
                mask=job.target.query_mask.numpy()*job.target.query_weight.numpy()))
        result={key:np.stack([r[key] for r in rows]) for key in rows[0]}
        if self.contact_search is not None:result.update(self.contact_search.snapshot(selected))
        return result


def _lbfgs_direction(gradient,history):
    q=gradient.copy();alphas=[]
    for s,y in reversed(history):
        sy=np.sum(s*y,axis=1);valid=sy>1e-12
        inverse=np.divide(1.,sy,out=np.zeros_like(sy),where=valid)
        alpha=inverse*np.sum(s*q,axis=1);q-=alpha[:,None]*y;alphas.append((alpha,inverse))
    if history:
        s,y=history[-1];yy=np.sum(y*y,axis=1);sy=np.sum(s*y,axis=1)
        gamma=np.clip(np.divide(sy,yy,out=np.ones_like(sy),where=(yy>1e-12)&(sy>1e-12)),1e-8,1e4)
        q*=gamma[:,None]
    else:q/=np.maximum(np.max(np.abs(q),axis=1),1.)[:,None]
    for (s,y),(alpha,inverse) in zip(history,reversed(alphas)):
        beta=inverse*np.sum(y*q,axis=1);q+=s*(alpha-beta)[:,None]
    direction=-q
    bad=np.sum(direction*gradient,axis=1)>=0
    direction[bad]=-gradient[bad]/np.maximum(np.max(np.abs(gradient[bad]),axis=1),1.)[:,None]
    return direction


def optimizer_class(native_model,settings):
    class EvaluationBackend(ServoPriorOptimizer):
        """Allocation and native FA derivatives only. Its optimizer is disabled."""
        def _make_objective(self,cpu_model,index,*,cfg,guidance_count):
            return VariableContactObjective(cpu_model,self.compiled,index,self.steps,self.terminal_observation,
                cfg=cfg,guidance_count=guidance_count,worlds=self.settings.worlds,
                native_model=native_model,admm=settings)

        def solve(self,*args,**kwargs):
            raise RuntimeError('Original FA solve is disabled: use FAContactADMMSolver')

        def _iteration(self,*args,**kwargs):
            raise RuntimeError('Original FA Adam iteration is disabled')

    class FAContactADMMSolver:
        def __init__(self,*args,**kwargs):
            if settings.hand_solver not in ('lbfgs','slsqp','acados'):
                raise ValueError(f'Unknown hand solver: {settings.hand_solver}')
            self.backend=EvaluationBackend(*args,**kwargs);self.graphs={};self.last_solver_diag={}
            if settings.fuse_adjoint_zero:
                from examples.retarget.fa_adjoint_fusion import install
                install(self.backend)
            scale=np.ones(self.backend.index.action_dim)
            if settings.balanced_coordinates:scale[3:6]=10.;scale[6:]=50.
            self.variable_scale=np.tile(scale*settings.control_scale,self.backend.settings.horizon)
            self.compiled_evaluator=None
            if settings.evaluation_backend not in ('native','compiled'):
                raise ValueError('FA evaluation backend must be native or compiled')
            if settings.evaluation_backend=='compiled':
                from pathlib import Path
                from examples.retarget.fa_compiled_evaluator import CompiledFAEvaluator
                directory=Path(settings.compiled_cache) if settings.compiled_cache else Path.home()/'.cache/scsp/fa_compiled'
                self.compiled_evaluator=CompiledFAEvaluator(self.backend,directory,threads=settings.compiled_threads)
            self.hand_qp=None
            if settings.hand_solver=='acados':
                if settings.hand_rti_iterations<1:raise ValueError('Positive hand RTI iteration count required')
                from examples.retarget.fa_hand_acados import CondensedHandQP
                self.hand_qp=CondensedHandQP(len(self.variable_scale))
            self.value_memo=None
            if settings.memoize_evaluations:
                from examples.retarget.fa_value_cache import FAValueCache
                self.value_memo=FAValueCache(self,self._evaluate_uncached)

        def __getattr__(self,key):return getattr(self.backend,key)

        def _evaluate(self,variables,gradient):
            if self.value_memo is not None:return self.value_memo.evaluate(variables,gradient)
            return self._evaluate_uncached(variables,gradient)

        def _evaluate_uncached(self,variables,gradient):
            if self.compiled_evaluator is not None:return self._evaluate_compiled(variables,gradient)
            b=self.backend
            same=settings.reuse_forward and np.array_equal(variables,getattr(self,'_forward_variables',None))
            if same and not gradient:
                self.evaluations+=1
                if hasattr(self,'value_reuses'):self.value_reuses+=1
                return self._forward_value.copy(),None
            reuse=gradient and same
            key='backward' if reuse else gradient
            if not reuse:
                raw=b.control.raw.numpy();raw[:,b.index.action_dim:]=variables*self.variable_scale
                b.control.raw.assign(raw);b.control.valid.fill_(1)
            def launch():
                if reuse:b._backward(allow_init=False)
                else:self._launch(gradient)
            if key not in self.graphs:
                launch();wp.synchronize()
                with wp.ScopedCapture(device=b.device) as capture:launch()
                self.graphs[key]=capture.graph
            wp.capture_launch(self.graphs[key]);wp.synchronize()
            self._forward_variables=variables.copy()
            if hasattr(self,'forward_calls'):
                self.forward_calls+=int(not reuse);self.backward_calls+=int(gradient);self.backward_reuses+=int(reuse)
            value=b.loss.total.numpy().astype(float)*1e-6
            valid=b.control.valid.numpy().astype(bool)&np.isfinite(value)
            value[~valid]=np.inf
            self.evaluations+=1
            if not gradient:
                self._forward_value=value.copy()
                return value,None
            g=b.control.gradient.numpy()[:,b.index.action_dim:].astype(float)*self.variable_scale
            valid&=np.isfinite(g).all(axis=1);value[~valid]=np.inf;g[~valid]=0.
            self._forward_value=value.copy()
            return value,g

        def _evaluate_compiled(self,variables,gradient):
            b=self.backend;e=self.compiled_evaluator
            same=settings.reuse_forward and np.array_equal(variables,getattr(self,'_forward_variables',None))
            self.evaluations+=1
            if same and not gradient:
                self.value_reuses+=1
                return self._forward_value.copy(),None
            raw=np.zeros((len(variables),b.index.action_dim+variables.shape[1]),np.float32)
            raw[:,b.index.action_dim:]=variables*self.variable_scale
            b.control.raw.assign(raw);b.control.valid.fill_(1)
            value,g=e.evaluate_raw(raw[:,b.index.action_dim:],gradient)
            b.loss.total.assign((value*1e6).astype(np.float32))
            if gradient:
                full=np.zeros_like(raw);full[:,b.index.action_dim:]=g
                b.control.gradient.assign(full);g=g*self.variable_scale
            self.forward_calls+=int(not (gradient and same));self.backward_calls+=int(gradient)
            self.backward_reuses+=int(gradient and same)
            self._forward_variables=variables.copy();self._forward_value=value.copy()
            return value,g

        def _materialize_native(self):
            # Contact ADMM and plan export consume native buffers. Refresh only
            # at these boundaries, rather than at every line-search trial.
            dirty=self.value_memo is not None and self.value_memo.dirty
            if self.compiled_evaluator is not None or dirty:
                b=self.backend
                def launch():
                    b._clear_contact_history();b._set_hard_contact();b._forward(allow_init=False);b._validate_simulator_candidates()
                    if self.compiled_evaluator is None:b.objective.evaluate(b.control,allow_init=False)
                if 'materialize' not in self.graphs:
                    launch();wp.synchronize_device(b.device)
                    with wp.ScopedCapture(device=b.device) as capture:launch()
                    self.graphs['materialize']=capture.graph
                wp.capture_launch(self.graphs['materialize'])
                self.native_materializations+=1
                if self.value_memo is not None:self.value_memo.dirty=False

        def _launch(self,gradient):
            b=self.backend;b._clear_contact_history();b._set_hard_contact();b._forward(allow_init=False)
            b._validate_simulator_candidates();b.objective.evaluate(b.control,allow_init=False)
            if gradient:b._backward(allow_init=False)

        def _hand_block(self,variables,iterations):
            # Contact points/marked weights can change between ADMM blocks.
            self._forward_variables=None
            if self.compiled_evaluator is not None and settings.optimize_contacts:self.compiled_evaluator.cache=None
            if self.value_memo is not None and settings.optimize_contacts:self.value_memo.invalidate()
            if iterations==0:return variables,0
            if settings.hand_solver=='acados':
                from examples.retarget.fa_hand_acados import solve_batched
                variables,accepted,diag=solve_batched(self._evaluate,variables,min(iterations,settings.hand_rti_iterations),
                    self.hand_qp,line_search=settings.line_search)
                self.hand_diagnostics.append(diag)
                return variables,accepted
            if settings.hand_solver=='slsqp':
                from examples.retarget.fa_hand_slsqp import solve_batched
                variables,accepted,diag=solve_batched(self._evaluate,variables,iterations,settings.hand_ftol)
                self.hand_diagnostics.append(diag)
                return variables,accepted
            value,gradient=self._evaluate(variables,True);history=[];accepted_steps=0
            for _ in range(iterations):
                direction=_lbfgs_direction(gradient,history)
                slope=np.sum(gradient*direction,axis=1)
                pending=np.isfinite(value)&(slope<0);accepted=np.zeros(len(value),bool)
                alpha=np.ones(len(value));trial=variables.copy();new=variables.copy()
                for _ in range(settings.line_search):
                    if not pending.any():break
                    trial[pending]=variables[pending]+alpha[pending,None]*direction[pending]
                    candidate,_=self._evaluate(trial,False)
                    good=pending&(candidate<=value+1e-4*alpha*slope)
                    new[good]=trial[good];accepted|=good;pending&=~good
                    alpha[pending]*=.5
                next_value,next_gradient=self._evaluate(new,True)
                s=new-variables;y=next_gradient-gradient
                sy=np.sum(s*y,axis=1);keep=(sy>1e-12)&accepted
                s[~keep]=0.;y[~keep]=0.;history.append((s,y))
                if len(history)>settings.history:history.pop(0)
                variables,value,gradient=new,next_value,next_gradient
                accepted_steps+=int(accepted.sum())
                if not accepted.any():break
            return variables,accepted_steps

        def solve(self,qpos,qvel,targets,*,warm,previous_action,allow_init,seed,execution_steps,start_time):
            started=perf_counter()
            b=self.backend
            if allow_init:raise ValueError('This experiment fixes the common CA initial pose')
            if settings.rho<=0 or settings.outer<1:raise ValueError('Positive rho and outer count required')
            phases=settings.outer
            if b.settings.iterations<1 or (settings.optimize_contacts and b.settings.iterations<phases+1):
                raise ValueError('Positive hand budget; contact ADMM needs at least one iteration per phase and polish')
            b._prepare_window(qpos,qvel,targets,warm=warm,previous_action=previous_action,seed=seed,start_time=start_time)
            if b.objective.contact_search is not None:
                b.objective.contact_search.begin(qpos,qvel,targets,seed,execution_steps)
            raw=b.control.raw.numpy();raw[:,:b.index.action_dim]=0.;b.control.raw.assign(raw)
            variables=raw[:,b.index.action_dim:].astype(float)/self.variable_scale
            prepared=perf_counter();self.evaluations=0;accepted=0;outer_diag=[];hand_seconds=0.;self.hand_diagnostics=[]
            self._forward_variables=None;self.forward_calls=0;self.backward_calls=0;self.backward_reuses=0;self.value_reuses=0
            self.native_materializations=0
            if self.value_memo is not None:self.value_memo.reset()
            if self.compiled_evaluator is not None:self.compiled_evaluator.prepare()
            counts=np.full(phases+1,b.settings.iterations//(phases+1),int)
            counts[:b.settings.iterations%(phases+1)]+=1
            if settings.fixed_hand_single_block:
                if settings.optimize_contacts:raise ValueError('Single-block ablation requires fixed contacts')
                counts[:]=0;counts[0]=b.settings.iterations
            self._evaluate(variables,False)
            for phase in range(phases):
                if settings.optimize_contacts:
                    self._materialize_native();b.objective.update_contacts()
                hand_started=perf_counter()
                variables,n=self._hand_block(variables,int(counts[phase]));accepted+=n
                hand_seconds+=perf_counter()-hand_started
                if settings.optimize_contacts:
                    self._materialize_native();b.objective.project_dual()
                residual=[]
                search=b.objective.contact_search
                if search is not None:
                    consensus=search.last['consensus_mean_mm']
                else:
                    for workspace,point in zip((*b.loss.immediate,b.loss.terminal),b.objective.points):
                        if workspace.job.query_enabled:
                            mask=workspace.target.query_mask.numpy()>0
                            error=np.linalg.norm(point.c.numpy()-point.z.numpy(),axis=-1)[:,mask]
                            if error.size:residual.extend(error.reshape(-1).tolist())
                    consensus=float(np.mean(residual)*1000) if residual else 0.
                outer_diag.append(dict(phase=phase,contact_consensus_mean_mm=consensus))
                if search is not None:
                    outer_diag[-1].update(force_seconds=search.last['force_seconds'],contact_search_seconds=search.last['search_seconds'])
            # Execute controls optimized for actual surface points and compare
            # pure FA values. ADMM rho never enters the final objective/ranking.
            if settings.optimize_contacts:b.objective.polish_surface()
            hand_started=perf_counter()
            variables,n=self._hand_block(variables,int(counts[-1]));accepted+=n
            hand_seconds+=perf_counter()-hand_started
            values,_=self._evaluate(variables,False)
            self._materialize_native()
            if self.compiled_evaluator is not None:
                values[~b.control.valid.numpy().astype(bool)]=np.inf
            fa_values=values.copy()
            if b.objective.contact_search is not None:
                values+=settings.contact_object_weight*b.objective.contact_search.last['object_cost']*1e-6
            valid=np.isfinite(values);selected=int(np.argmin(values))
            if not valid.any():raise FloatingPointError('All ADMM hand candidates are invalid')
            scores=(values*1e6).astype(np.float32);drifts=b._candidate_drifts(targets,execution_steps)
            b.control.best_raw.assign(b.control.raw);b.control.best_loss.assign(scores)
            b.control.best_available.assign(valid.astype(np.int32))
            b.control.best_grad_norm.assign(np.linalg.norm(b.control.gradient.numpy(),axis=1).astype(np.float32))
            evaluation=finalize_candidate_evaluation(scores,drifts,has_finite_best=valid,hard_simulator_valid=valid,
                training_failure_iteration=np.full(len(valid),-1,np.int32),training_failure_code=np.zeros(len(valid),np.int32))
            plan=b._plan(selected,started,evaluation=evaluation,execution_steps=execution_steps)
            b._commit_regularization_history()
            b.objective.remember_contacts(selected,execution_steps)
            self.last_solver_diag=dict(solver='contact_ADMM_batched_'+settings.hand_solver,original_FA_Adam_called=False,
                hand_iteration_budget=int(counts.sum()),evaluations=self.evaluations,accepted_candidate_steps=accepted,
                hand_subproblem_iteration_budgets=counts.tolist(),
                outer=outer_diag,final_ranking='pure_FA',variable_scales=self.variable_scale[:b.index.action_dim].tolist(),
                MANO_role='initialization_only' if settings.optimize_contacts else 'fixed_template',
                prepare_seconds=prepared-started,hand_seconds=hand_seconds,contact_solver=settings.contact_solver,
                hand_solver=settings.hand_solver,hand_subproblems=self.hand_diagnostics,
                forward_calls=self.forward_calls,backward_calls=self.backward_calls,backward_reuses=self.backward_reuses,
                value_reuses=self.value_reuses,fused_adjoint_zero=settings.fuse_adjoint_zero)
            self.last_solver_diag.update(evaluation_backend=settings.evaluation_backend)
            self.last_solver_diag.update(memoize_evaluations=settings.memoize_evaluations,
                memoized_world_values=self.value_memo.hits if self.value_memo is not None else 0,
                skipped_value_batches=self.value_memo.skipped_batches if self.value_memo is not None else 0)
            if self.compiled_evaluator is not None:
                self.last_solver_diag.update(compiled_build_seconds=self.compiled_evaluator.build_seconds,
                    native_materializations=self.native_materializations,compiled_threads=self.compiled_evaluator.threads,
                    compiled_library=str(self.compiled_evaluator.library))
            self.last_solver_diag['hand_rti_iterations']=settings.hand_rti_iterations if settings.hand_solver=='acados' else None
            self.last_solver_diag['grad_norm_at_final_controls']=settings.hand_solver=='lbfgs'
            if b.objective.contact_search is not None:
                self.last_solver_diag.update(final_ranking='FA_plus_object_contact_cost',
                    FA_loss=float(fa_values[selected]*1e6),
                    object_contact_cost=float(b.objective.contact_search.last['object_cost'][selected]),
                    contact_samples=settings.contact_samples,contact_radius_m=settings.contact_radius,
                    warm_contact_slots_restored=b.objective.warm_contacts_restored,
                    force_model='native hard ComFree object/environment, ideal friction-cone hand forces',
                    force_jacobian=b.objective.contact_search.last['jacobian'],
                    force_evaluations_per_candidate=b.objective.contact_search.last['model_evaluations_per_candidate'])
            wp.synchronize_device(b.device)
            elapsed=perf_counter()-started
            self.last_solver_diag['wall_seconds']=elapsed
            return replace(plan,seconds=elapsed)

    return FAContactADMMSolver
