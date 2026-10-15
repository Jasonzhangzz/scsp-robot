"""Material-point Wuji MPC on the shared grouped soft-SCM contact response.

State = object pose, measured hand q, mixed velocity, carried servo target.
Predicted hand-object contact impulses come from SCM, never from applied Lambda
forces. Actual contacts remain controlled independently of Lambda validity.
"""
from dataclasses import dataclass
import hashlib

import casadi as cs
import numpy as np
from models.contact_geometry import rotation_cs
from models.soft_scm_contact import grouped_contact_velocity_cs,soft_scm_coefficients
from planning.mpc_explicit import MPCExplicit


def select_contact_candidate(candidates,hold_cost):
    """Prefer physically predicted closure AND task/pose descent when available.

    Before such a candidate exists, contact acquisition can still improve the
    task cost; it is explicitly reported as acquisition, never certified grasp.
    """
    improving=[c for c in candidates if np.isfinite(c['score']) and c['score']<hold_cost-1e-12]
    certified=[c for c in improving if c['certificate']['force_closure'] and
        (c['object_pose_reduction']>1e-12 or c['object_pose_cost']<=1e-12)]
    if certified:return min(certified,key=lambda c:c['score']),'closed_pose_descent'
    if improving:return min(improving,key=lambda c:c['score']),'contact_acquisition'
    holds=[c for c in candidates if c['source']=='hold' and np.isfinite(c['score'])]
    if holds:return holds[0],'hold_pending_closure_or_descent'
    return (min(candidates,key=lambda c:c['score']),'collision_recovery') if candidates else (None,'unsafe_hold')


def internal_preload_floor(weight,friction,made_count,force_limit):
    """Mass-scaled soft compression preference for a native pyramidal cone.

    Its inscribed tangential disk has radius mu/sqrt(2). Poor opposition must
    not LOWER the requested load budget. Force balance and native task ranking
    decide whether applying this preference is useful for the measured grasp.
    """
    required=.6 if made_count<2 else max(.6,1.3*np.sqrt(2.)*weight/(friction*made_count))
    return min(required,force_limit/np.sqrt(1+friction**2))


@dataclass(frozen=True)
class WujiRetargetCosts:
    """Online trigrasp weights, with persistent soft MANO correspondence.

    trigrasp releases its IK attraction at contact. Retargeting retains a small
    nonzero attraction to every frame's pose, without a shape-error constraint.
    """
    contact: float = 5000.
    original_contact: float = 150.
    existing_contact_scale: float = .1
    normal: float = .1
    object_position: float = 120.
    object_lateral: float = 300.
    object_rotation: float = 50.
    horizontal_recovery: float = 2e6
    horizontal_tolerance: float = .005
    transport: float = 1000.
    normal_support: float = 1.
    preload: float = 2.
    load_balance: float = 10.
    posture_base: float = 1.
    posture_rotation: float = .5
    posture_fingers: float = 5.
    posture_contact_scale: float = .05
    action_base: float = .03
    action_rotation: float = 50.
    action_fingers: float = 50.
    collision: float = 1e7
    terminal: float = 4.

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not 0 < self.posture_contact_scale <= 1:
            raise ValueError('posture_contact_scale must be in (0, 1]')
        if self.existing_contact_scale>1:raise ValueError('existing_contact_scale must be in [0, 1]')


class WujiContactModel:
    transports_contacts=True
    def __init__(self,param,slots):
        self.param_=param;self.slots=slots;n=param.n_cmd_;nv=param.n_qvel_;nx=param.n_qpos_;self.pose_size=7+n
        self.physical_parameter_dim=1+self.pose_size+36+n+6+n
        self.command_parameter_dim=self.physical_parameter_dim+n*n+2*n
        self.model_parameter_dim=self.command_parameter_dim+115+40*slots+3*slots*n+param.max_ncon_
        self.contact_friction=param.contact_friction.copy()
        self.projection=np.eye(n);self.pressure_bias=np.zeros(n)
        self.command_limits=param.mpc_u_ub_.copy()
        self.grasp_map=np.zeros((6,3*slots));self.grasp_robot=np.zeros((3*slots,n))
        self.object_mass=np.eye(6);self.grasp_inverse=np.eye(6);self.grasp_mobility=np.zeros((6,6))
        self.grasp_internal=np.zeros((6,3*slots));self.grasp_reference=np.zeros(3*slots)
        self.grasp_limits=np.zeros(slots);self.other_wrench=np.zeros(6);self.grasp_active=False
        self.reference=np.r_[np.zeros(3),1.,np.zeros(3+n)]
        self.inverse=np.eye(6);self.robot_mass=np.ones(n);self.external=np.zeros(6)
        self.robot_external=np.zeros(n)
        x=cs.SX.sym('wuji_state',nx);u=cs.SX.sym('wuji_command',n)
        phi=cs.SX.sym('wuji_gap',4*param.max_ncon_);jac=cs.SX.sym('wuji_contact',4*param.max_ncon_,nv)
        mp=cs.SX.sym('wuji_physics',self.model_parameter_dim)
        reference=mp[1:1+self.pose_size];o=1+self.pose_size
        object_inverse=cs.reshape(mp[o:o+36],6,6)
        hand_inverse=mp[o+36:o+36+n]
        inverse=cs.diagcat(object_inverse,cs.diag(hand_inverse))
        external=mp[o+36+n:o+42+n]
        robot_external=mp[o+42+n:o+42+2*n]
        cursor=self.physical_parameter_dim
        projection=cs.reshape(mp[cursor:cursor+n*n],n,n);cursor+=n*n
        pressure_bias=mp[cursor:cursor+n];limits=mp[cursor+n:cursor+2*n]
        command=cs.fmin(limits,cs.fmax(-limits,projection@u+pressure_bias))
        qr,q=reference[3:7],x[3:7]
        displacement=cs.vertcat(x[:3]-reference[:3],2*(qr[0]*q[1:]-q[0]*qr[1:]-cs.cross(qr[1:],q[1:])),x[7:self.pose_size]-reference[7:])
        gaps=phi+cs.vertcat(*[cs.repmat((cs.sum1(jac[i:i+4,:])/4)@displacement,4,1) for i in range(0,phi.numel(),4)])
        self.contact_gap_fn=cs.Function('wuji_contact_gaps',[x,phi,jac,mp],[gaps])
        velocity=x[self.pose_size:self.pose_size+nv];h=param.h_
        target=cs.fmin(cs.DM(param.hand_upper),cs.fmax(cs.DM(param.hand_lower),x[-n:]+command))
        response=h*param.robot_damping_+h*h*np.diag(param.robot_stiff_)
        hand_free=velocity[6:]+mp[o+36:o+36+n]*(h*cs.DM(param.robot_stiff_)@(target-x[7:self.pose_size])-cs.DM(response)*velocity[6:])
        robot_mass=1/mp[o+36:o+36+n]-cs.DM(response)
        effort=(hand_free-velocity[6:])*robot_mass/h
        effort=cs.fmin(cs.DM(param.actuator_upper),cs.fmax(cs.DM(param.actuator_lower),effort))
        hand_free=velocity[6:]+h*(effort+robot_external)/robot_mass
        free=cs.vertcat(velocity[:6]+h*inverse[:6,:6]@external,hand_free)
        a,b=soft_scm_coefficients(h,.02,1.,1e-8)
        nxt,impulses=grouped_contact_velocity_cs(free,jac,gaps,inverse,h,a,b,1e-8,
            return_impulses=True,friction=mp[-param.max_ncon_:],friction_cone='pyramidal')
        forces=[]
        for i in range(slots):
            block=jac[4*i:4*i+4,:3];normal=cs.sum1(block).T/4
            mu=cs.sqrt(cs.sumsqr((block[0,:]-block[2,:])/2)+1e-20)
            tangents=cs.horzcat((block[0,:]-block[2,:]).T,(block[1,:]-block[3,:]).T)/(2*cs.fmax(mu,1e-10))
            fn,ft=impulses[i];forces.append((normal*fn+tangents@ft)/h)
        force=cs.vertcat(*forces) if forces else cs.SX.zeros(0)
        if slots:
            cursor=self.command_parameter_dim
            def matrix(rows,columns):
                nonlocal cursor
                value=cs.reshape(mp[cursor:cursor+rows*columns],rows,columns)
                cursor+=rows*columns;return value
            velocity_map,robot_jac=matrix(6,3*slots),matrix(3*slots,n)
            object_mass,grasp_inverse,mobility=matrix(6,6),matrix(6,6),matrix(6,6)
            internal,reference_force,normal_limits,other_wrench=matrix(6,3*slots),matrix(3*slots,1),matrix(slots,1),matrix(6,1)
            from models.allegro_grasp_response import force_limited_grasp_cs
            current=cs.vertcat(velocity[:3],rotation_cs(qr).T@rotation_cs(q)@velocity[3:6])
            supported,supported_force=force_limited_grasp_cs(nxt,force,current,hand_inverse,
                object_inverse,external,jac,gaps,h=h,friction=param.friction,
                velocity_map=velocity_map,robot_jac=robot_jac,object_mass=object_mass,
                grasp_inverse=grasp_inverse,grasp_mobility=mobility,internal_map=internal,
                reference_force=reference_force,normal_limits=normal_limits,other_wrench=other_wrench,
                friction_cone='pyramidal',allow_preload_growth=True,
                force_limit=param.contact_force_limit)
            supported=cs.vertcat(supported[:3],rotation_cs(q).T@rotation_cs(qr)@supported[3:6],supported[6:])
            nxt=cs.if_else(mp[cursor]>.5,supported,nxt)
            force=cs.if_else(mp[cursor]>.5,supported_force,force)
        self.contact_force_fn=cs.Function('wuji_contact_forces',[x,u,phi,jac,mp],[force],{'cse':True})
        # Force gradients are refreshed at the measured state for each RTI step.
        self.force_linearization_fn=cs.Function('wuji_force_linearization',[x,u,phi,jac,mp],
            [force,cs.jacobian(force,x),cs.jacobian(force,u)],{'cse':True})
        omega=nxt[3:6];quat=q+.5*h*cs.vertcat(-cs.dot(q[1:],omega),q[0]*omega+cs.cross(q[1:],omega));quat/=cs.sqrt(cs.sumsqr(quat)+1e-20)
        # The physical asset has joint stops. Include them in the reduced
        # predictor instead of asking the QP to satisfy bounds that its free
        # dynamics immediately violate under gravity.
        hand_position=cs.fmin(cs.DM(param.hand_upper),cs.fmax(cs.DM(param.hand_lower),x[7:self.pose_size]+h*nxt[6:]))
        hand_velocity=(hand_position-x[7:self.pose_size])/h
        next_pose=cs.vertcat(x[:3]+h*nxt[:3],quat,hand_position)
        next_velocity=cs.vertcat(nxt[:3],rotation_cs(quat).T@rotation_cs(q)@omega,hand_velocity)
        self.prediction_fn=cs.Function('wuji_contact_step',[x,u,phi,jac,mp],[cs.vertcat(next_pose,next_velocity,target)],{'cse':True})

    def runtime_parameters(self,sigma):
        h=self.param_.h_;response=h*self.param_.robot_damping_+h*h*np.diag(self.param_.robot_stiff_)
        return np.r_[sigma,self.reference,self.inverse.reshape(-1,order='F'),1/(self.robot_mass+response),self.external,self.robot_external,
            self.projection.reshape(-1,order='F'),self.pressure_bias,self.command_limits,
            self.grasp_map.reshape(-1,order='F'),self.grasp_robot.reshape(-1,order='F'),
            self.object_mass.reshape(-1,order='F'),self.grasp_inverse.reshape(-1,order='F'),
            self.grasp_mobility.reshape(-1,order='F'),self.grasp_internal.reshape(-1,order='F'),
            self.grasp_reference,self.grasp_limits,self.other_wrench,float(self.grasp_active),self.contact_friction]


class WujiMPCParams:
    def __init__(self,hand,slots,dt=.03,horizon=4,solver='acados',rates=None):
        self.h_=dt;self.n_cmd_=self.n_robot_qpos_=hand.nu;self.n_qvel_=6+hand.nu
        self.n_qpos_=7+hand.nu+self.n_qvel_+hand.nu
        self.max_ncon_=slots+9;self.num_contacts=slots
        self.hand_lower=hand.lower;self.hand_upper=hand.upper;self.friction=hand.friction
        self.contact_force_limit=10.
        self.contact_friction=np.r_[np.full(slots,hand.friction),hand.floor_friction,np.full(8,hand.friction)]
        self.actuator_lower=np.where(hand.model.actuator_forcelimited,hand.model.actuator_forcerange[:,0],-1e6)
        self.actuator_upper=np.where(hand.model.actuator_forcelimited,hand.model.actuator_forcerange[:,1],1e6)
        self.robot_stiff_=np.diag(hand.model.actuator_gainprm[:,0]);self.robot_damping_=-hand.model.actuator_biasprm[:,2]
        self.mpc_horizon_=horizon;self.planner_solver_=solver;self.mpc_model='explicit';self.runtime_object_cost_=True
        self.model_params=1.;self.ipopt_max_iter_=60;self.mpc_ipopt_fallback_=False
        self.mpc_compact_dynamics_=True;self.mpc_compact_cost_=True;self.mpc_prune_codegen_=True
        self.mpc_acados_profile_=dict(nlp_solver_type='SQP_RTI',qp_solver='FULL_CONDENSING_HPIPM',nlp_solver_max_iter=1,levenberg_marquardt=.01)
        self.mpc_u_ub_=dt*(np.r_[np.full(3,.15),np.full(3,.8),np.full(hand.nu-6,1.5)] if rates is None else np.asarray(rates));self.mpc_u_lb_=-self.mpc_u_ub_
        low=np.r_[np.full(3,-3.),np.full(4,-2.),hand.lower,np.full(self.n_qvel_,-100.),hand.lower]
        high=np.r_[np.full(3,3.),np.full(4,2.),hand.upper,np.full(self.n_qvel_,100.),hand.upper]
        self.mpc_q_lb_=low;self.mpc_q_ub_=high
        self.tag=hashlib.sha1(np.r_[low,high,self.mpc_u_lb_,self.mpc_u_ub_,dt,slots,horizon].tobytes()).hexdigest()[:10]

    def init_cost_fns(self): return self.costs


class WujiOnlineMPC(MPCExplicit):
    def __init__(self,hand,slots,dt=.03,horizon=4,solver='acados',rates=None,costs=None,force_limit=10.,native_force_feedback=False,regrasp=False):
        self.hand=hand;self.slots=slots;p=WujiMPCParams(hand,slots,dt,horizon,solver,rates)
        self.weights=costs if isinstance(costs,WujiRetargetCosts) else WujiRetargetCosts(**(costs or {}))
        self.force_limit=float(force_limit)
        self.native_force_feedback=bool(native_force_feedback)
        self.load_ready_streak=0
        self.transport_released=False
        from planning.wuji_regrasp import OnlineRegrasp
        self.regrasp=OnlineRegrasp() if regrasp else None
        if not np.isfinite(self.force_limit) or self.force_limit<=0:raise ValueError('force_limit must be positive')
        p.contact_force_limit=self.force_limit
        self.force_scale=1/max(float(hand.model.body_mass[hand.obj]*np.linalg.norm(hand.model.opt.gravity)),1e-6)
        p.tag=hashlib.sha1((p.tag+repr(self.weights)+'online_recovery_v1').encode()).hexdigest()[:10]
        physics=WujiContactModel(p,slots);self.physics=physics
        self._build_cost(p,physics)
        super().__init__(p,cost_kind='param',model=physics)
        self.stage_cost_on_next=False
        self.warm=None
        self.previous_reference=None;self.previous_active=None
        self.corridor_origin=hand.object_pose()[:2].copy()
        self.sticking_anchors=np.full((slots,3),np.nan)
        self.sticking_materials=[None]*slots
        self.physical_materials=[None]*slots
        self.support_servo=None
        self.support_origin=None

    def _build_cost(self,p,physics):
        n,s,nx=p.n_cmd_,self.slots,p.n_qpos_
        # Goal/posture, measured kinematics and both object anchor references,
        # masks, force linearization, and self-collision linearizations.
        size=7+3*n+15*s+9*s*n+6*s+2*s+1+5*s+16+nx+n+3*s+3*s*(nx+n)+8*(1+n)
        cp=cs.SX.sym('wuji_semantic_targets',size);self.cp_size=size
        x=cs.SX.sym('wuji_cost_state',nx);u=cs.SX.sym('wuji_cost_command',n);i=0
        def take(k):
            nonlocal i
            value=cp[i:i+k];i+=k;return value
        goal,qref,servo_ref,q0=take(7),take(n),take(n),take(n)
        points=cs.reshape(take(3*s),3,s);normals=cs.reshape(take(3*s),3,s)
        jac=cs.reshape(take(3*s*n),3*s,n)
        normal_jac=cs.reshape(take(3*s*n),3*s,n)
        original_points=cs.reshape(take(3*s),3,s)
        original_jac=cs.reshape(take(3*s*n),3*s,n)
        anchors=cs.reshape(take(3*s),3,s);original=cs.reshape(take(3*s),3,s)
        desired_normals=cs.reshape(take(3*s),3,s)
        active,stick=take(s),take(s);approach=take(1)
        contact_normals=cs.reshape(take(3*s),3,s)
        live_arms=cs.reshape(take(3*s),3,s)
        self.attraction_index=i+1
        load_objective_scale,attraction=take(1),take(1)
        semantic_active=take(s)
        release_mask=take(s)
        lower,upper=take(2),take(2)
        transport_origin=take(3)
        self.preload_floor_index=i
        preload_floor=take(1)
        self.other_load_index=i
        other_load=take(6)
        self.force_offset=i
        fx,fu=take(nx),take(n);fv=take(3*s);jx=cs.reshape(take(3*s*nx),3*s,nx);ju=cs.reshape(take(3*s*n),3*s,n)
        force=cs.reshape(fv+jx@(x-fx)+ju@(u-fu),3,s)
        predicted=points+cs.reshape(jac@(x[7:7+n]-q0),3,s)
        original_predicted=original_points+cs.reshape(original_jac@(x[7:7+n]-q0),3,s)
        predicted_normals=normals+cs.reshape(normal_jac@(x[7:7+n]-q0),3,s)
        world=cs.repmat(x[:3],1,s)+rotation_cs(x[3:7])@anchors
        original_world=cs.repmat(x[:3],1,s)+rotation_cs(x[3:7])@original
        w=self.weights
        grasp_intent=cs.fmin(cs.sum1(semantic_active),1.)*load_objective_scale
        # Same object/task-space objective and weight conventions as
        # planning.allegro_mpc_costs, used by trigrasp's online controller.
        terms=[np.sqrt(w.object_lateral)*(x[:2]-goal[:2]),
            np.sqrt(w.object_position)*(x[2]-goal[2]),
            np.sqrt(w.object_rotation/8)*cs.reshape(rotation_cs(goal[3:7]).T@rotation_cs(x[3:7])-cs.DM.eye(3),9,1),
            np.sqrt(w.horizontal_recovery)*cs.fmax(lower-x[:2],0),
            np.sqrt(w.horizontal_recovery)*cs.fmax(x[:2]-upper,0)]
        support_forces=[];support_moments=[]
        transport_delta=goal[:3]-transport_origin
        transport_scale=cs.fmin(1.,(.15*p.h_*p.mpc_horizon_)/cs.sqrt(cs.sumsqr(transport_delta)+1e-20))
        transport_position=transport_origin+transport_scale*transport_delta
        self.transport_residual_slices=[]
        for k in range(s):
            gate=active[k];loaded=stick[k]
            error=predicted[:,k]-world[:,k]
            # Contact dynamics and a soft internal-preload objective select
            # indentation. Material correspondence preserves tangential points
            # after touch without fixing the normal force or servo depth.
            normal=contact_normals[:,k]
            alignment_normal=rotation_cs(x[3:7])@desired_normals[:,k]
            tangent=error-alignment_normal*cs.dot(alignment_normal,error)
            alignment=(1-loaded)*error+loaded*tangent
            original_error=original_predicted[:,k]-original_world[:,k]
            original_alignment=original_error
            semantic_gate=semantic_active[k]
            offset=sum(int(term.numel()) for term in terms)+6
            self.transport_residual_slices.append(slice(offset,offset+3))
            terms.extend([np.sqrt(w.contact)*gate*alignment,
                np.sqrt(w.original_contact)*semantic_gate*original_alignment,
                np.sqrt(w.transport)*gate*loaded*(predicted[:,k]-transport_position-rotation_cs(goal[3:7])@anchors[:,k]),
                np.sqrt(w.normal)*semantic_gate*(predicted_normals[:,k]-rotation_cs(x[3:7])@desired_normals[:,k])])
            support_normal=predicted_normals[:,k]/cs.sqrt(cs.sumsqr(predicted_normals[:,k])+1e-12)
            support_forces.append(gate*support_normal)
            support_moments.append(gate*cs.cross((predicted[:,k]-x[:3])/.05,support_normal))
        if s:
            count=cs.fmax(cs.sum1(active),1.)
            terms.extend([np.sqrt(w.normal_support)*sum(support_forces)/count,
                np.sqrt(w.normal_support)*sum(support_moments)/count])
            # Internal load support, not a wrench supplied by Lambda. Geometry
            # alone cannot tell a weightless touch from a load-bearing grasp.
            # The force variables remain SCM responses to bounded joint servo
            # commands; native rollout ranking uses measured physical forces.
            gravity=cs.DM(self.hand.model.body_mass[self.hand.obj]*self.hand.model.opt.gravity)
            com=rotation_cs(x[3:7])@cs.DM(self.hand.model.body_ipos[self.hand.obj])
            moment=cs.cross(com,gravity)+other_load[3:]
            for k in range(s):moment+=cs.cross(live_arms[:,k],force[:,k])
            terms.extend([grasp_intent*self.force_scale*np.sqrt(w.load_balance)*(cs.sum2(force)+other_load[:3]+gravity),
                grasp_intent*self.force_scale*np.sqrt(w.load_balance)*moment/.05])
        for _ in range(8):
            gap,row=take(1),take(n)
            terms.append(np.sqrt(w.collision)*cs.fmax(.001-gap-row.T@(x[7:7+n]-q0),0))
        if i!=size: raise RuntimeError((i,size))
        # Preload belongs in the terminal objective too. A running-only term
        # lets a short receding horizon postpone all compression beyond the
        # command actually executed. This is an internal soft preference, not
        # Lambda force tracking or an externally supplied wrench.
        for k in range(s):
            terms.append(grasp_intent*self.force_scale*np.sqrt(w.preload)*stick[k]*
                cs.fmax(preload_floor-cs.dot(contact_normals[:,k],force[:,k]),0))
            terms.append(release_mask[k]*self.force_scale*np.sqrt(w.load_balance)*
                cs.dot(contact_normals[:,k],force[:,k]))
        state_residual=cs.vertcat(*terms)
        # Shape is a running soft preference at EVERY frame, including loaded
        # contact. It never enters the terminal grasp objective or feasibility.
        posture_weights=np.r_[np.full(3,w.posture_base),np.full(3,w.posture_rotation),np.full(n-6,w.posture_fingers)]
        posture=cs.sqrt(cs.fmax(attraction,w.posture_contact_scale))*cs.DM(np.sqrt(posture_weights))*(x[7:7+n]-qref)
        self.posture_residual_slice=slice(int(state_residual.numel()),int(state_residual.numel())+n)
        running=[state_residual,posture]
        action_weights=np.r_[np.full(3,w.action_base),np.full(3,w.action_rotation),np.full(n-6,w.action_fingers)]
        residual=cs.vertcat(*running,cs.DM(np.sqrt(action_weights))*u)
        self.residual_fn=cs.Function('wuji_residual',[x,u,cp],[residual])
        terminal=np.sqrt(w.terminal)*cs.substitute(state_residual,u,cs.SX.zeros(n))
        self.terminal_residual_fn=cs.Function('wuji_terminal_residual',[x,cp],[terminal])
        p.costs=(cs.Function('wuji_stage_'+p.tag,[x,u,cp],[cs.sumsqr(residual)]),cs.Function('wuji_terminal_'+p.tag,[x,cp],[cs.sumsqr(terminal)]))

    def _cost_params(self,**kwargs): return self.current_cp

    def state(self):
        h=self.hand
        return np.r_[h.object_pose(),h.data.qpos[h.qadr],h.data.qvel[h.ov:h.ov+6],h.data.qvel[h.dadr],h.servo]

    def plan(self,goal,qref,materials,anchors,original,inward,active,*,original_materials=None,contact_qref=None,approach=False,lambda_valid=True,execution_mask=None):
        from examples.retarget.contact_geometry import contact_geometry
        from models.wuji_hand import rotation,MaterialPoint
        state=self.state();p=self.param_;hand=self.hand
        semantic_active=np.asarray(active,float).copy()
        planning_active=semantic_active if execution_mask is None else np.asarray(execution_mask,float)
        requested_materials=list(materials)
        if original_materials is None:original_materials=requested_materials
        # A touch on a link does NOT mean its requested MANO material point
        # touched. With non-spherical Wuji links those can be centimetres apart.
        # As in trigrasp's adopted local pad patch, use the real hand-side
        # surface material for execution; retain MANO materials as soft priors.
        _,patches=hand.observed_contacts(requested_materials,planning_active)
        materials=list(requested_materials);used=set();unrepresented_patches=0
        for body in sorted({c['body'] for c in patches}):
            available=[k for k,m in enumerate(requested_materials) if m.body==body]
            groups=[]
            for patch in [c for c in patches if c['body']==body and c['normal']@c['force']>.01]:
                matching=next((g for g in groups if g[0]['normal']@patch['normal']>.98 and
                    np.linalg.norm(g[0]['point']-patch['point'])<.003),None)
                if matching is None:groups.append([patch])
                else:matching.append(patch)
            # Duplicate colliders on the same patch share a force slot. Two
            # different box faces on ONE link must keep distinct SCM normals.
            for group in groups:
                patch=max(group,key=lambda c:c['normal']@c['force'])
                body_rotation=hand.data.xmat[body].reshape(3,3)
                def match_key(k):
                    cached=self.physical_materials[k]
                    same=cached is not None and cached.body==body and (body_rotation@cached.normal)@patch['normal']>.98
                    return (not same,not bool(semantic_active[k]),
                        np.linalg.norm(hand.data.xpos[body]+body_rotation@requested_materials[k].local-patch['point']))
                free=[k for k in available if k not in used]
                if not free:
                    unrepresented_patches+=1;continue
                k=min(free,key=match_key);used.add(k);requested=requested_materials[k]
                cached=self.physical_materials[k]
                if cached is None or cached.body!=body or (body_rotation@cached.normal)@patch['normal']<.98:
                    hand_point=patch['point']-.5*patch['gap']*patch['normal']
                    cached=MaterialPoint(body,body_rotation.T@(hand_point-hand.data.xpos[body]),
                        body_rotation.T@patch['normal'],requested.finger,requested.region,requested.local,True)
                    self.physical_materials[k]=cached
                materials[k]=cached
        for k in range(self.slots):
            if k not in used:self.physical_materials[k]=None
        phi,jac,points,normals,point_jac,stick,collision,coefficients=contact_geometry(
            hand,materials,planning_active,p.max_ncon_,return_friction=True)
        self.physics.contact_friction=coefficients
        active=np.maximum(planning_active,stick*(1. if execution_mask is None else self.weights.existing_contact_scale))
        anchors=np.asarray(anchors,float).copy();inward=np.asarray(inward,float).copy()
        inertia,robot_mass,external=hand.dynamics()
        self.physics.reference=state[:7+hand.nu].copy();self.physics.inverse=np.linalg.inv(inertia);self.physics.robot_mass=robot_mass;self.physics.external=external
        self.physics.robot_external=-hand.data.qfrc_bias[hand.dadr].copy()
        self.physics.object_mass=inertia.copy();self.physics.projection=np.eye(hand.nu);self.physics.pressure_bias=np.zeros(hand.nu)
        measured,actual=hand.observed_contacts(materials,active)
        normal_rows=np.asarray(jac).reshape(-1,4,p.n_qvel_)[:self.slots,:,:3].mean(axis=1)
        measured_normal=np.einsum('ij,ij->i',normal_rows,measured)
        made=(stick>.5)&(measured_normal>.01)
        ids=np.flatnonzero(made)
        physical_jac=point_jac.copy()
        for k in ids:
            block=jac[4*k:4*k+4]
            physical_jac[k]=np.linalg.lstsq(block[:,:3],-block[:,6:],rcond=None)[0]
        r=rotation(state[:7])
        projected=hand.surface.query((points-state[:3])@r)[0] if self.slots else np.empty((0,3))
        live_points=state[:3]+projected@r.T
        for k in np.flatnonzero(stick):
            matching=[c for c in actual if c['slot']==k]
            weights=[max(c['normal']@c['force'],1e-9) for c in matching]
            live_points[k]=np.average([c['point'] for c in matching],axis=0,weights=weights)
            previous=self.sticking_materials[k]
            changed=previous is None or previous.body!=materials[k].body or not np.allclose(previous.local,materials[k].local,atol=1e-7,rtol=0)
            if changed or not np.isfinite(self.sticking_anchors[k]).all():
                # Match trigrasp: a made hand material point keeps its object
                # coordinate until separation, rather than chasing each slip.
                self.sticking_anchors[k]=(points[k]-state[:3])@r
                self.sticking_materials[k]=materials[k]
            if not lambda_valid or not planning_active[k]:
                # Original semantic anchors remain a separate soft preference.
                anchors[k]=self.sticking_anchors[k]
                inward[k]=normal_rows[k]@r
        self.sticking_anchors[stick<.5]=np.nan
        self.physics.grasp_active=False
        self.physics.other_wrench=np.zeros(6)
        from planning.grasp_verification import verify_grasp
        from planning.contact_friction import native_tangents
        observed=[c for c in actual if c['normal']@c['force']>=.01]
        actual_points=np.array([(c['point']-state[:3])@r for c in observed]).reshape(-1,3)
        actual_normals=np.array([c['normal']@r for c in observed]).reshape(-1,3)
        actual_tangents=np.array([c.get('tangents',native_tangents(c['normal']))@r for c in observed]).reshape(-1,2,3)
        certificate=verify_grasp(actual_points,actual_normals,hand.friction,edges=4,
            tangent_directions=actual_tangents,external_wrench=np.r_[r.T@external[:3],external[3:]],
            normal_force_limits=np.array([c['normal']@c['force'] for c in observed]))
        regrasp_report=dict(phase='disabled',moving_slots=[])
        moving_materials={};releasing_fingers=set();semantic_cost_active=semantic_active.copy()
        if self.regrasp is not None:
            anchors,inward,regrasp_report,moving_materials=self.regrasp.update(
                hand,materials,anchors,inward,planning_active,actual,certificate,goal,original)
            moving=regrasp_report['moving_slots']
            if moving:
                active[moving]=0.
                active[regrasp_report['slot']]=1.
                semantic_cost_active[moving]*=.1
                if regrasp_report['phase']!='settle':
                    releasing_fingers={materials[k].finger for k in moving}
                self.transport_released=False;self.load_ready_streak=0
        # Force closure stays a SOFT objective/candidate preference. Sustained
        # finite-load capacity can release transport even when a tiny patch
        # crosses the certificate's Fn cutoff; it never gates contact physics.
        load_ready=bool(certificate['load_feasible']) and not regrasp_report['moving_slots']
        self.load_ready_streak=self.load_ready_streak+1 if load_ready else 0
        if self.load_ready_streak>=6:self.transport_released=True
        if not np.any(semantic_active):self.transport_released=False;self.load_ready_streak=0
        requested_goal=np.asarray(goal).copy()
        if np.any(semantic_active) and not self.transport_released:
            # trigrasp establishes sustained, load-bearing compression before
            # advancing lift. This is a REFERENCE-phase pause, never a gate on
            # partial-contact dynamics, pressure buildup or per-frame soft IK.
            goal=state[:7].copy()
        # The force-limited response can describe partial grasps. Neither a
        # geometric closure Boolean nor CURRENT gravity support may prevent
        # predicting transport and the buildup of physically generated preload.
        self.physics.grasp_active=bool(len(ids)>=2)
        # Forces are chosen through contact dynamics by the object-pose and
        # contact objectives. Lambda contributes contact locations, never a
        # force/wrench target. As in trigrasp, the model and executor share an
        # internal pressure-feedback command map; its load is solved HERE.
        preload_budget=internal_preload_floor(np.linalg.norm(external[:3]),max(hand.friction,1e-6),len(ids),self.force_limit)
        # A small internal preload floor is distinct from the mass-dependent
        # force allocation. Uniform full-weight pressure at two one-sided
        # patches would shove the box away before an opposing finger arrives.
        preload_floor=min(.6,self.force_limit/np.sqrt(1+hand.friction**2))
        from planning.wuji_load_support import static_preload_seed
        selected_bodies={m.body for k,m in enumerate(requested_materials) if planning_active[k]}
        releasing_bodies={m.body for m in requested_materials if m.finger in releasing_fingers}
        support_actual=[c for c in actual if c['body'] not in releasing_bodies]
        load_seed=static_preload_seed(hand,support_actual,normal_rows,self.force_limit,
            preferred_bodies=None if execution_mask is None else selected_bodies)
        requested_fingers={m.finger for k,m in enumerate(requested_materials) if semantic_active[k]}
        pressure_ids=np.array([k for k in ids if materials[k].finger in requested_fingers and materials[k].finger not in releasing_fingers],dtype=int)
        pressure_active=len(pressure_ids)>=2
        if pressure_active:
            from planning.wuji_load_support import pressure_command_map
            pressure_reference=np.full(len(pressure_ids),preload_floor)
            floors=np.array([.2 if execution_mask is None or materials[k].body in selected_bodies else 0. for k in pressure_ids])
            if load_seed['feasible']:pressure_reference=np.maximum(floors,load_seed['normal_forces'][pressure_ids])
            elif execution_mask is not None:
                pressure_reference=np.array([preload_floor if materials[k].body in selected_bodies else min(.2,measured_normal[k]) for k in pressure_ids])
            self.physics.projection,self.physics.pressure_bias=pressure_command_map(
                physical_jac[pressure_ids],normal_rows[pressure_ids],measured_normal[pressure_ids],
                pressure_reference,limit=min(.0015,float(p.mpc_u_ub_[:3].min())),
                stiffness=hand.model.actuator_gainprm[:,0],
                grasp_established=bool(certificate['force_closure']))
        else:pressure_reference=np.empty(0)
        for contact in actual:
            if contact['slot'] not in ids:
                self.physics.other_wrench[:3]+=contact['force']
                self.physics.other_wrench[3:]+=r.T@np.cross(contact['point']-state[:3],contact['force'])
        if self.physics.grasp_active:
            from models.allegro_grasp_response import grasp_operators
            grasp_jac=jac.copy()
            for k in np.flatnonzero(~made):grasp_jac[4*k:4*k+4]=0
            response=p.h_*p.robot_damping_+p.h_**2*np.diag(p.robot_stiff_)
            operators=grasp_operators(grasp_jac,1/(robot_mass+response),inertia,self.slots)
            (self.physics.grasp_map,self.physics.grasp_robot,self.physics.grasp_inverse,
                self.physics.grasp_mobility,self.physics.grasp_internal)=operators
            self.physics.grasp_reference=measured.reshape(-1).copy()
            self.physics.grasp_reference.reshape(-1,3)[~made]=0
            self.physics.grasp_limits=np.maximum(measured_normal,0)*made
        mp=self.physics.runtime_parameters(1.)
        # Dynamics use the currently touching patch; the planned contact cost
        # must use Lambda's requested hand material. Moving an unrelated made
        # patch to Lambda's object anchor can make a valid contact IK look bad.
        cost_materials=list(materials)
        if lambda_valid:
            for k in np.flatnonzero(planning_active):cost_materials[k]=requested_materials[k]
        for k,material in moving_materials.items():cost_materials[k]=material
        cost_points,cost_normals,cost_jac=hand.kinematics(cost_materials)
        cost_stick=stick.copy()
        if self.slots:
            requested_projected,cost_gaps,requested_inward=hand.surface.query((cost_points-state[:3])@r)
        else:requested_projected=np.empty((0,3));cost_gaps=np.empty(0);requested_inward=np.empty((0,3))
        if lambda_valid and self.slots:
            for k in np.flatnonzero(planning_active):
                cost_stick[k]*=float(cost_gaps[k]<=.0001 and cost_normals[k]@(r@inward[k])>0.)
        if regrasp_report['moving_slots'] and regrasp_report['phase']!='settle':
            cost_stick[regrasp_report['moving_slots']]=0.
        original_points,_,original_jac=hand.kinematics(materials if original_materials is None else original_materials)
        force,jx,ju=self.physics.force_linearization_fn(state,np.zeros(hand.nu),phi,jac,mp)
        # Match trigrasp's measured-output correction. In particular, zero
        # control must predict the pressure actually present, rather than an
        # arbitrary SCM preload. The motion model remains force-limited SCM.
        force_baseline=np.asarray(force).reshape(-1).copy()
        raw_normal=np.einsum('ij,ij->i',normal_rows,force_baseline.reshape(-1,3))
        force_gain=np.repeat(np.where(made,np.clip(measured_normal/np.maximum(raw_normal,.01),.2,50.),1.),3)
        force=np.asarray(measured).reshape(-1)
        jx=force_gain[:,None]*np.asarray(jx);ju=force_gain[:,None]*np.asarray(ju)
        pack=lambda a:np.asarray(a).reshape(-1,order='F')
        servo_ref=hand.servo_reference(qref)
        gaps=phi.reshape(-1,4)[:self.slots,0]
        attraction=float(np.clip((np.min(gaps[active>0])-.002)/.018,0.,1.)) if np.any(active) else 1.
        tolerance=self.weights.horizontal_tolerance
        coverage=float(np.clip(1-np.linalg.norm(np.mean(normal_rows[ids],axis=0)),0,1)) if len(ids) else 0.
        if load_seed['feasible']:
            bias=hand.model.actuator_biasprm
            servo_bias=bias[:,0]+bias[:,1]*hand.data.actuator_length+bias[:,2]*hand.data.actuator_velocity
            self.support_servo=np.clip((load_seed['required_joint_torque']-servo_bias)/
                np.maximum(hand.model.actuator_gainprm[:,0],1e-12),hand.lower,hand.upper)
            self.support_origin=state[:3].copy()
        # One link can touch two box faces. Its dominant SCM normal cannot
        # represent both pressures; generate the balanced seed per REAL patch,
        # including currently unmapped physical contacts on the palm/hand.
        pressure_patches=[c for c in actual if c['normal']@c['force']>.01]
        patch_jac=[]
        if load_seed['feasible']:
            import mujoco
            for contact in pressure_patches:
                jp,jr=np.zeros((3,hand.model.nv)),np.zeros((3,hand.model.nv))
                mujoco.mj_jac(hand.model,hand.data,jp,jr,contact['point'],contact['body'])
                patch_jac.append(contact['normal']@jp[:,hand.dadr[6:]])
        def unassigned_load(contacts,origin):
            load=np.zeros(6)
            for contact in contacts:
                if contact['slot']<0:
                    load[:3]+=contact['force']
                    load[3:]+=np.cross(contact['point']-origin,contact['force'])
            return load
        other_load=unassigned_load(actual,state[:3])
        lower=np.minimum(state[:2],np.asarray(goal)[:2])-tolerance
        upper=np.maximum(state[:2],np.asarray(goal)[:2])+tolerance
        release_mask=np.array([float(m.finger in releasing_fingers) for m in materials])
        load_objective_scale=.1 if releasing_fingers else 1.
        self.current_cp=np.r_[goal,qref,servo_ref,hand.data.qpos[hand.qadr],pack(cost_points.T),pack(cost_normals.T),pack(cost_jac.reshape(-1,hand.nu)),pack(hand.normal_jacobians(cost_materials).reshape(-1,hand.nu)),pack(original_points.T),pack(original_jac.reshape(-1,hand.nu)),
            pack(anchors.T),pack(original.T),pack(inward.T),active,cost_stick,float(approach),
            pack(normal_rows.T),pack((live_points-state[:3]).T),load_objective_scale,attraction,semantic_cost_active,release_mask,lower,upper,state[:3],preload_floor,other_load,
            state,np.zeros(hand.nu),pack(force),pack(jx),pack(ju),collision.reshape(-1)]
        if len(self.current_cp)!=self.cp_size: raise RuntimeError((len(self.current_cp),self.cp_size))
        self.last_problem=dict(state=state.copy(),gap=phi.copy(),jacobian=jac.copy(),
            physics=mp.copy(),cost=self.current_cp.copy())
        if self.previous_reference is None or np.max(np.abs(qref-self.previous_reference))>.1 or not np.array_equal(active,self.previous_active):
            self.warm=None
        self.previous_reference=np.asarray(qref).copy();self.previous_active=np.asarray(active).copy()
        result=self.plan_once(curr_x=state,phi_vec=phi,jac_mat=jac,sol_guess=self.warm)
        action=np.asarray(result['action']).reshape(-1)
        raw_commands=np.asarray(result['u_traj']).reshape(p.mpc_horizon_,hand.nu)
        _,actual_contacts=hand.observed_contacts(materials,active)
        allowed={m.body for k,m in enumerate(materials) if active[k]}|{c['body'] for c in actual_contacts}
        baseline=hand.collision_report(hand.data,allowed)
        baseline_pairs={tuple(item[:2]) for item in baseline['collisions']}

        def exact_cost_parameters(current,command):
            cp=self.current_cp.copy();offset=self.force_offset
            cp[offset:offset+p.n_qpos_]=current;offset+=p.n_qpos_
            cp[offset:offset+hand.nu]=command;offset+=hand.nu
            raw=np.asarray(self.physics.contact_force_fn(current,command,phi,jac,mp)).reshape(-1)
            cp[offset:offset+3*self.slots]=measured.reshape(-1)+force_gain*(raw-force_baseline)
            offset+=3*self.slots
            cp[offset:offset+3*self.slots*(p.n_qpos_+hand.nu)]=0.
            return cp

        def rollout(commands,first_state=None):
            current=state.copy();score=0.;rows=[]
            for stage,command in enumerate(commands):
                score+=float(self.path_cost_fn(current,command,exact_cost_parameters(current,command)))
                current=first_state.copy() if stage==0 and first_state is not None else np.asarray(self.physics.prediction_fn(current,command,phi,jac,mp)).reshape(-1)
                if not np.isfinite(current).all():return np.inf,[]
                rows.append(np.r_[command,current])
            score+=float(self.final_cost_fn(current,exact_cost_parameters(current,np.zeros(hand.nu))))
            return score,rows

        recovery_joints=[]
        if regrasp_report['moving_slots']:
            import mujoco
            from planning.wuji_retarget import WUJI_FINGERS
            prefix=f'right_finger{WUJI_FINGERS[regrasp_report["finger"]]}_'
            recovery_joints=[j for j,joint in enumerate(hand.model.actuator_trnid[:,0]) if
                mujoco.mj_id2name(hand.model,mujoco.mjtObj.mjOBJ_BODY,int(hand.model.jnt_bodyid[joint])).startswith(prefix)]
        seeds=[]
        for source in ('ik_seed','coordinated_ik_seed','contact_ik_seed','coordinated_contact_ik_seed','material_seed','projection_seed',
                       'regrasp_open_seed',
                       'preload_seed','transport_seed','preload_transport_seed',
                       'balanced_preload_seed','balanced_transport_seed',
                       'equilibrium_seed','equilibrium_transport_seed'):
            if regrasp_report['moving_slots']:
                recovery_sources=('material_seed','transport_seed','projection_seed') if regrasp_report['phase'] in ('approach','settle') else ('material_seed','regrasp_open_seed')
                if source not in recovery_sources:continue
            if source=='regrasp_open_seed' and regrasp_report['phase']!='release':continue
            if pressure_active and source in ('ik_seed','coordinated_ik_seed','preload_seed','preload_transport_seed',
                'balanced_preload_seed','balanced_transport_seed'):continue
            if self.transport_released and certificate['force_closure'] and source in (
                'contact_ik_seed','equilibrium_seed','equilibrium_transport_seed'):continue
            if source.startswith('balanced_') and not load_seed['feasible']:continue
            if source.startswith('equilibrium_') and self.support_servo is None:continue
            current=state.copy();commands=[]
            for _ in range(p.mpc_horizon_):
                seed_reference=hand.servo_reference(contact_qref) if source in ('contact_ik_seed','coordinated_contact_ik_seed') and contact_qref is not None else servo_ref
                delta=seed_reference-current[-hand.nu:]
                if source=='regrasp_open_seed':
                    delta=np.zeros(hand.nu)
                    neutral=np.clip(np.zeros(hand.nu),hand.lower,hand.upper)
                    delta[recovery_joints]=neutral[recovery_joints]-current[-hand.nu:][recovery_joints]
                if source.startswith('equilibrium_'):
                    # Convert MPC's internally feasible static drive torques to
                    # the unchanged position actuators' affine control law.
                    # This is only a native-cost-checked position candidate;
                    # no forces are applied or supplied by Lambda. Retaining
                    # it through temporary patch loss can finish acquisition.
                    support_target=self.support_servo.copy()
                    if source=='equilibrium_transport_seed':
                        motion=np.asarray(goal)[:3]-self.support_origin
                        support_target[:3]+=motion*min(1.,.05*p.h_*(len(commands)+1)/max(np.linalg.norm(motion),1e-12))
                    delta=support_target-current[-hand.nu:]
                if source in ('preload_seed','preload_transport_seed','transport_seed',
                              'balanced_preload_seed','balanced_transport_seed'):
                    # A measured-pressure candidate, never an overriding
                    # command map. MPC accepts it only on native task-cost
                    # improvement. Leave tangential motion to its other seeds.
                    rows=physical_jac[ids,:,6:].reshape(-1,hand.nu-6)
                    delta=np.zeros(hand.nu)
                    if source.startswith('balanced_'):
                        rows=np.asarray(patch_jac)
                        observed_pressure=np.array([c['normal']@c['force'] for c in pressure_patches])
                        indentation=.0003*(load_seed['patch_normal_forces']-observed_pressure)
                        delta[6:]=rows.T@np.linalg.solve(rows@rows.T+1e-6*np.eye(len(rows)),indentation)
                    elif len(ids):
                        indentation=.0003*(preload_floor-measured_normal[ids])
                        displacement=(normal_rows[ids]*indentation[:,None]).reshape(-1)
                        delta[6:]=rows.T@np.linalg.solve(rows@rows.T+1e-5*np.eye(len(displacement)),displacement)
                    if source=='transport_seed':delta[:]=0.
                    if source in ('transport_seed','preload_transport_seed','balanced_transport_seed'):
                        # trigrasp explicitly evaluates rigid grasp transport
                        # because reduced SCM can underestimate tangential load
                        # transfer. It is a candidate, not a lift mode override.
                        motion=np.asarray(goal)[:3]-current[:3]
                        delta[:3]=motion*min(1.,.05*p.h_/max(np.linalg.norm(motion),1e-12))
                target_ids=np.flatnonzero(active)
                if source in ('material_seed','projection_seed') and len(target_ids) and not approach:
                    predicted=cost_points+(cost_jac.reshape(-1,hand.nu)@(current[7:7+hand.nu]-state[7:7+hand.nu])).reshape(-1,3)
                    from models.wuji_hand import rotation
                    desired_points=current[:3]+anchors@rotation(current[:7]).T
                    if source=='projection_seed':
                        nearest,_,inward_nearest=hand.surface.query((predicted-current[:3])@rotation(current[:7]))
                        desired_points=current[:3]+(nearest+.0005*inward_nearest)@rotation(current[:7]).T
                    rows=cost_jac[target_ids].reshape(-1,hand.nu)
                    error=(desired_points[target_ids]-predicted[target_ids]).reshape(-1)
                    if regrasp_report['moving_slots'] and regrasp_report['phase'] in ('release','transfer'):
                        rows=rows.copy();rows[:,[j for j in range(hand.nu) if j not in recovery_joints]]=0.
                    delta=rows.T@np.linalg.solve(rows@rows.T+1e-5*np.eye(len(error)),error)
                if source in ('coordinated_ik_seed','coordinated_contact_ik_seed'):
                    scale=min(1.,float(np.min(p.mpc_u_ub_/np.maximum(np.abs(delta),1e-12))))
                    command=scale*delta
                else:command=np.clip(delta,p.mpc_u_lb_,p.mpc_u_ub_)
                commands.append(command)
                current=np.asarray(self.physics.prediction_fn(current,command,phi,jac,mp)).reshape(-1)
            seeds.append((source,np.asarray(commands)))
        native_sensitivity=False
        if self.native_force_feedback and len(ids)>=2 and self.weights.load_balance>0:
            # Output-bias scaling cannot recover missing cross-patch pressure
            # sensitivities. Refresh a bounded Gauss-Newton seed through the
            # ACTUAL position-servo executor; acados/SCM remains the main solve.
            # No force or wrench target is supplied to this feedback step.
            from scipy.optimize import lsq_linear
            def native_residual(command):
                effective=np.clip(self.physics.projection@command+self.physics.pressure_bias,
                    -self.physics.command_limits,self.physics.command_limits)
                targets=np.clip(hand.servo+np.arange(1,p.mpc_horizon_+1)[:,None]*effective,hand.lower,hand.upper)
                observations=[]
                def observe(data):
                    force,contacts=hand.observed_contacts(materials,active,data=data)
                    x=np.r_[data.qpos[hand.oq:hand.oq+7],data.qpos[hand.qadr],data.qvel[hand.ov:hand.ov+6],
                        data.qvel[hand.dadr],data.ctrl].copy()
                    observations.append((x,force.reshape(-1).copy(),unassigned_load(contacts,x[:3])))
                hand.preview_sequence(targets,p.h_,observer=observe)
                values=[];current=state;current_force=measured.reshape(-1);current_other=other_load
                def parameters(x,u,force,other):
                    cp=self.current_cp.copy();o=self.force_offset
                    cp[o:o+p.n_qpos_]=x;o+=p.n_qpos_
                    cp[o:o+hand.nu]=u;o+=hand.nu
                    cp[o:o+3*self.slots]=force;o+=3*self.slots
                    cp[o:o+3*self.slots*(p.n_qpos_+hand.nu)]=0.
                    cp[self.other_load_index:self.other_load_index+6]=other
                    return cp
                for next_state,next_force,next_other in observations:
                    values.append(np.asarray(self.residual_fn(current,command,
                        parameters(current,command,current_force,current_other))).reshape(-1))
                    current,current_force,current_other=next_state,next_force,next_other
                values.append(np.asarray(self.terminal_residual_fn(current,
                    parameters(current,np.zeros(hand.nu),current_force,current_other))).reshape(-1))
                return np.concatenate(values)
            zero=np.zeros(hand.nu);baseline_residual=native_residual(zero)
            epsilon=np.minimum(.1*p.mpc_u_ub_,np.r_[np.full(3,.0001),np.full(3,.001),np.full(hand.nu-6,.002)])
            columns=[]
            for axis in range(hand.nu):
                perturbation=zero.copy();perturbation[axis]=epsilon[axis]
                columns.append((native_residual(perturbation)-native_residual(-perturbation))/(2*epsilon[axis]))
            sensitivity=np.column_stack(columns)
            if np.isfinite(sensitivity).all() and np.isfinite(baseline_residual).all():
                correction=lsq_linear(np.r_[sensitivity,.1*np.eye(hand.nu)],
                    np.r_[-baseline_residual,np.zeros(hand.nu)],bounds=(p.mpc_u_lb_,p.mpc_u_ub_),max_iter=50)
                if np.isfinite(correction.x).all():
                    seeds.append(('native_force_feedback_seed',np.tile(correction.x,(p.mpc_horizon_,1))))
                    native_sensitivity=True
        candidates=[('solver',raw_commands),*seeds,('hold',np.zeros_like(raw_commands))]
        from examples.retarget.contact_geometry import native_grasp_certificate
        def object_pose_cost(pose):
            error=rotation(goal).T@rotation(pose)-np.eye(3)
            return float(self.weights.object_lateral*np.sum((pose[:2]-goal[:2])**2)+
                self.weights.object_position*(pose[2]-goal[2])**2+self.weights.object_rotation/8*np.sum(error**2))
        initial_pose_cost=object_pose_cost(state[:7])
        evaluated=[];hold_cost=np.inf
        for source,commands in candidates:
            if not np.isfinite(commands).all():continue
            for fraction in (1.,.5,.25,.125,.0625):
                candidate=commands*fraction
                effective=np.clip(self.physics.projection@candidate[0]+self.physics.pressure_bias,-self.physics.command_limits,self.physics.command_limits)
                target=np.clip(hand.servo+effective,hand.lower,hand.upper)
                shadow=hand.preview_control(target,p.h_)
                predicted=np.r_[shadow.qpos[hand.oq:hand.oq+7],shadow.qpos[hand.qadr],
                    shadow.qvel[hand.ov:hand.ov+6],shadow.qvel[hand.dadr],target]
                if not np.isfinite(predicted).all():continue
                check=hand.collision_report(shadow,allowed)
                pairs={tuple(item[:2]) for item in check['collisions']}
                recovering=(pairs<=baseline_pairs and check['max_penetration']<baseline['max_penetration']-1e-6)
                if not check['valid'] and not recovering:continue
                score,rows=rollout(candidate,first_state=predicted)
                if not np.isfinite(score):continue
                # Validate the whole candidate's carried position targets using
                # the native executor. Only these physical patches certify
                # closure; projected/desired points cannot masquerade as made.
                targets=[np.asarray(row)[-hand.nu:] for row in rows]
                # Ranking must use the SAME physics that validates closure.
                # Previously only stage one was native: later SCM forces could
                # claim compression and win the ranking despite no real grip.
                native_states=[];native_forces=[];native_other=[]
                def observe(data):
                    native_states.append(np.r_[data.qpos[hand.oq:hand.oq+7],data.qpos[hand.qadr],
                        data.qvel[hand.ov:hand.ov+6],data.qvel[hand.dadr],data.ctrl].copy())
                    forces,contacts=hand.observed_contacts(materials,active,data=data)
                    native_forces.append(forces.reshape(-1).copy())
                    native_other.append(unassigned_load(contacts,data.qpos[hand.oq:hand.oq+3]))
                preview=hand.preview_sequence(targets,p.h_,observer=observe)
                previous=state;previous_force=measured.reshape(-1);previous_other=other_load;score=0.
                for command,next_state,next_force,next_other in zip(candidate,native_states,native_forces,native_other):
                    cp=exact_cost_parameters(previous,command)
                    start=self.force_offset+p.n_qpos_+hand.nu
                    cp[start:start+3*self.slots]=previous_force
                    cp[self.other_load_index:self.other_load_index+6]=previous_other
                    score+=float(self.path_cost_fn(previous,command,cp))
                    previous,previous_force,previous_other=next_state,next_force,next_other
                cp=exact_cost_parameters(previous,np.zeros(hand.nu))
                cp[start:start+3*self.slots]=previous_force
                cp[self.other_load_index:self.other_load_index+6]=previous_other
                score+=float(self.final_cost_fn(previous,cp))
                final_check=hand.collision_report(preview,allowed)
                final_pairs={tuple(item[:2]) for item in final_check['collisions']}
                final_recovering=(final_pairs<=baseline_pairs and
                    final_check['max_penetration']<baseline['max_penetration']-1e-6)
                if not final_check['valid'] and not final_recovering:continue
                preview_certificate=native_grasp_certificate(hand,materials,active,preview)
                end_pose=preview.qpos[hand.oq:hand.oq+7]
                pose_cost=object_pose_cost(end_pose)
                evaluated.append(dict(score=score,target=target.copy(),source=source,fraction=fraction,
                    commands=candidate.copy(),rows=rows,certificate=preview_certificate,
                    force_balance_N=float(np.linalg.norm(previous_force.reshape(-1,3).sum(axis=0)+
                        previous_other[:3]+hand.model.body_mass[hand.obj]*hand.model.opt.gravity)),
                    object_pose_cost=pose_cost,object_pose_reduction=initial_pose_cost-pose_cost))
                if source=='hold':hold_cost=score
                # For a hold command every fraction is identical.
                if source=='hold':break
        self.warm=None
        winner,contact_plan_status=select_contact_candidate(evaluated,hold_cost)
        if winner is None:
            target=hand.servo.copy();source='unsafe_hold';fraction=0.;score=None
            executed_prediction=[]
        else:
            score,target,source,fraction,commands,rows=(winner[k] for k in
                ('score','target','source','fraction','commands','rows'))
            executed_prediction=np.asarray(self.physics.contact_force_fn(state,commands[0],phi,jac,mp)).reshape(-1,3).tolist()
            if source!='hold':
                commands=np.vstack((commands[1:],commands[-1:]))
                _,rows=rollout(commands)
                if rows:self.warm=dict(x0=np.concatenate(rows),u_traj=commands)
        status=result.get('solve_status','unknown')
        if source!='solver':status='BOUNDED_ROLLOUT_'+source.upper()+'; '+status
        nearby=[c for c in evaluated if c['source']=='projection_seed']
        return target,dict(status=status,backend=result.get('solver_backend',self.planner_solver_),
            predicted_force=executed_prediction,stick=stick.tolist(),
            lambda_valid=bool(lambda_valid),made_contact_count=len(ids),
            execution_anchors=anchors.tolist(),horizontal_bounds=np.array([lower,upper]).tolist(),
            semantic_mask=semantic_active.tolist(),execution_mask=planning_active.tolist(),
            force_source='contact_dynamics',external_wrench_target=False,
            candidate_cost_physics='native_MuJoCo',force_feedback=True,
            native_force_sensitivity_seed=native_sensitivity,
            force_objective='internal_gravity_force_and_moment_balance',
            internal_pressure_control=pressure_active,internal_pressure_reference_N=pressure_reference.tolist(),
            internal_pressure_slots=pressure_ids.tolist() if pressure_active else [],
            internal_load_seed={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in load_seed.items()},
            candidate_scores=[dict(source=c['source'],fraction=c['fraction'],cost=c['score'],
                force_balance_N=c['force_balance_N'],force_closure=c['certificate']['force_closure'],
                load_feasible=c['certificate']['load_feasible'],pose_reduction=c['object_pose_reduction']) for c in evaluated],
            physical_material_count=sum(m is not None for m in self.physical_materials),
            unrepresented_physical_patches=unrepresented_patches,
            execution_materials=[dict(body=m.body,local=m.local.tolist(),normal=m.normal.tolist()) for m in cost_materials],
            physics_materials=[dict(body=m.body,local=m.local.tolist(),normal=m.normal.tolist()) for m in materials],
            planned_material_contact_mask=cost_stick.tolist(),
            internal_preload_floor_N=float(preload_floor),internal_preload_budget_N=float(preload_budget),opposing_coverage=coverage,
            contact_plan_status=contact_plan_status,
            preview_grasp_certificate=None if winner is None else winner['certificate'],
            object_pose_reduction=None if winner is None else winner['object_pose_reduction'],
            task_cost_reduction_vs_hold=None if winner is None or not np.isfinite(hold_cost) else hold_cost-score,
            raw_action=action.tolist(),guard_scale=fraction,executed_action=(target-hand.servo).tolist(),
            selected_control=source,verified_rollout_cost=score,grasp_response=self.physics.grasp_active,
            transport_active=bool(len(ids) and self.transport_released),load_ready_streak=self.load_ready_streak,
            requested_object_goal=requested_goal.tolist(),executed_object_goal=np.asarray(goal).tolist(),
            regrasp=regrasp_report,
            release_mask=release_mask.tolist(),load_objective_scale=load_objective_scale,
            nearest_contact_candidates=dict(points_local=requested_projected.tolist(),gaps_m=cost_gaps.tolist(),
                inward_local=requested_inward.tolist(),source_materials='current_execution_materials',
                selected=(source=='projection_seed'),native_feasible=bool(nearby),
                evaluated_count=len(nearby),
                best_cost=None if not nearby else min(c['score'] for c in nearby),
                closed_candidates=sum(bool(c['certificate']['force_closure']) for c in nearby),
                cost_improving_candidates=sum(c['score']<hold_cost-1e-12 for c in nearby),
                closed_pose_descent_candidates=sum(bool(c['certificate']['force_closure']) and
                    (c['object_pose_reduction']>1e-12 or c['object_pose_cost']<=1e-12) for c in nearby),
                validation='native_horizon_cost_and_collision'),
            grasp_certificate=certificate,friction_cone='pyramidal')
