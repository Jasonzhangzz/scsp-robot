"""Direct multiple-shooting ComFree MPC with native analytic stage functions.

Experimental path: 30 dense controls over the same 300 ms horizon (the old FA
path interpolates five knots). FA geometry and tracking enter as individual
residuals with native analytic Jacobians and Gauss-Newton curvature. A separate
optional slow ADMM planner updates contact targets; physics contacts stay native.
"""
from pathlib import Path
from time import perf_counter
import numpy as np
import casadi as cs


class DirectComFreeMPC:
    def __init__(self,engine,*,horizon=30,rti_steps=1,cache=None,trust=.5,full_contact_jacobian=True,line_search_steps=4,post_step_tracking=False,levenberg_marquardt=.01,relative_warm=False):
        if horizon<1 or rti_steps<1 or line_search_steps<1 or not np.isfinite(trust) or trust<=0:
            raise ValueError('Positive horizon, RTI iteration budget and trust radius required')
        from examples.retarget.comfree_ocp_native import NativeStages
        from examples.retarget.comfree_ocp_bridge import build_bridge
        from planning.mpc_explicit import _load_acados,_set_acados_code_export_directory
        from planning.acados_env import acados_cache_directory,build_cached_solver
        Model,Ocp,_=_load_acados()
        class FirstOrderCostOcp(Ocp):
            def _setup_code_generation_context(this,context,ignore_initial=False,ignore_terminal=False):
                # The installed acados template differentiates NLS residuals
                # twice even in GAUSS_NEWTON mode. Native VJPs expose first
                # derivatives only. Generate exactly the two entry points
                # requested by GN; do not invent second derivatives.
                if this.solver_options.hessian_approx!='GAUSS_NEWTON':
                    raise ValueError('Native stage functions require Gauss-Newton')
                names=('cost_type_0','cost_type','cost_type_e')
                original=[getattr(this.cost,n) for n in names]
                try:
                    for name in names:setattr(this.cost,name,'LINEAR_LS')
                    context=super()._setup_code_generation_context(context,ignore_initial,ignore_terminal)
                finally:
                    for name,value in zip(names,original):setattr(this.cost,name,value)
                m=this.model
                for suffix,skip in [('_0',ignore_initial),('',False),('_e',ignore_terminal)]:
                    if skip:continue
                    y=getattr(m,'cost_y_expr'+suffix)
                    v=m.u if suffix!='_e' else cs.MX.sym('empty_u',0)
                    directory=str(Path(context.opts.code_export_directory)/(m.name+'_cost'))
                    base=m.name+'_cost_y'+suffix
                    context.add_function_definition(base+'_fun',[m.x,v,m.z,m.t,m.p],[y],directory,'cost')
                    jac=cs.jacobian(y,cs.vertcat(v,m.x)).T
                    context.add_function_definition(base+'_fun_jac_ut_xt',[m.x,v,m.z,m.t,m.p],
                        [y,jac,cs.jacobian(y,m.z)],directory,'cost')
                return context
        self.engine=engine;self.N=horizon;self.rti_steps=rti_steps;self.dt=.01
        self.trust=trust
        self.line_search_steps=line_search_steps
        self.post_step_tracking=post_step_tracking
        if not np.isfinite(levenberg_marquardt) or levenberg_marquardt<0:raise ValueError('Finite nonnegative LM damping required')
        self.native=NativeStages(engine,horizon,full_contact_jacobian=full_contact_jacobian);self.nu=engine.model.nu;self.nphysical=self.native.nx
        self.nx=self.nphysical+2*self.nu
        self.hand_scale=np.r_[np.full(3,.01),np.full(3,.1),np.full(self.nu-6,.5)]
        qscale=np.ones(self.native.nq);qscale[engine.index.hand_qpos]=self.hand_scale
        qscale[engine.oq:engine.oq+3]=.01
        vscale=np.ones(self.native.nv);vscale[engine.dadr]=np.r_[np.ones(3),np.full(3,3.),np.full(self.nu-6,4.)]
        vscale[engine.ov+3:engine.ov+6]=3.
        self.physical_scale=np.r_[qscale,vscale]
        self.xscale=np.r_[self.physical_scale,self.hand_scale,self.hand_scale]
        version='v5_post' if post_step_tracking else 'v3'
        if levenberg_marquardt!=.01:version+='_lm'+format(levenberg_marquardt,'.0e').replace('-','m').replace('+','p')
        self.cache=Path(cache or acados_cache_directory('fa_direct_comfree_'+version)).resolve()
        library,self.shared,self.callback=build_bridge(self.cache,self.native)
        physics=cs.external('native_physics',str(library),{'enable_fd':False})
        cost=cs.external('native_fa',str(library),{'enable_fd':False})
        x=cs.MX.sym('x',self.nx);u=cs.MX.sym('u',self.nu)
        p=cs.MX.sym('stage',35)
        physical=x[:self.nphysical]*cs.DM(self.physical_scale);servo=u*cs.DM(self.hand_scale)
        result=physics(physical,servo,p[0]);next_physical=result[:self.nphysical]
        previous=x[self.nphysical:self.nphysical+self.nu]*cs.DM(self.hand_scale)
        previous_previous=x[self.nphysical+self.nu:]*cs.DM(self.hand_scale)
        rm=self.native.regularization;cfg=engine.cfg
        dv=(servo-previous)/self.dt;ddv=(servo-2*previous+previous_previous)/self.dt**2
        hv=physical[self.native.nq+engine.dadr];ha=result[self.nphysical+engine.dadr]
        reg=cs.vertcat(cs.DM(np.sqrt(rm.control_velocity_weight.numpy()))*dv/cs.DM(rm.control_velocity_scale.numpy()),
            np.sqrt(cfg.smoothing.control_acceleration_weight)*ddv/cs.DM(rm.control_acceleration_scale.numpy()),
            cs.DM(np.sqrt(cfg.smoothing.hand_state_weight*rm.state_component_ratio.numpy()))*hv/cs.DM(rm.state_velocity_scale.numpy()),
            cs.DM(np.sqrt(2*cfg.smoothing.hand_state_weight*rm.state_component_ratio.numpy()))*ha/cs.DM(rm.state_acceleration_scale.numpy()))
        reg*=.001
        weights=self.native.weights
        hand_weights=np.r_[np.full(3,weights.action_base_position),np.full(3,weights.action_base_rotation),np.full(self.nu-6,weights.action_finger)]
        oq=engine.oq
        quat=physical[oq+3:oq+7];target=p[30:34]
        quat=quat/(cs.norm_2(quat)+1e-8);target=target/(cs.norm_2(target)+1e-8)
        target=cs.if_else(cs.dot(quat,target)<0,-target,target)
        relative=cs.vertcat(target[0]*quat[1]-target[1]*quat[0]-target[2]*quat[3]+target[3]*quat[2],
            target[0]*quat[2]+target[1]*quat[3]-target[2]*quat[0]-target[3]*quat[1],
            target[0]*quat[3]-target[1]*quat[2]+target[2]*quat[1]-target[3]*quat[0])
        angle=2*cs.atan2(cs.sqrt(cs.fmax(cs.sumsqr(relative),1e-12)),cs.dot(target,quat))
        tracking=cs.vertcat(cs.DM(hand_weights)*(physical[engine.index.hand_qpos]-p[1:27]),
            np.sqrt(weights.reference_pose_scale)*weights.object_position*(physical[oq:oq+3]-p[27:30]),
            np.sqrt(weights.reference_pose_scale)*weights.object_rotation*angle)*cs.sqrt(p[34])
        geometry=cs.vertcat(tracking,cost(physical[:self.native.nq],p[0]))
        if post_step_tracking:
            future_x=cs.vertcat(next_physical/cs.DM(self.physical_scale),x[self.nphysical:])
            stage_geometry=cs.vertcat(cs.substitute(tracking,x,future_x),cost(physical[:self.native.nq],p[0]))
        else:stage_geometry=geometry
        residual=cs.vertcat(stage_geometry,reg)
        model=Model();model.name=f'fa_native_comfree_ocp_{version}_n{horizon}'
        model.x=x;model.u=u;model.p=p
        model.disc_dyn_expr=cs.vertcat(next_physical/cs.DM(self.physical_scale),u,x[self.nphysical:self.nphysical+self.nu])
        model.cost_y_expr=residual;model.cost_y_expr_e=geometry
        ocp=FirstOrderCostOcp();ocp.model=model;ocp.parameter_values=np.r_[np.zeros(30),1.,0.,0.,0.,1.]
        ocp.cost.cost_type='NONLINEAR_LS';ocp.cost.cost_type_e='NONLINEAR_LS'
        ocp.cost.W=2*np.eye(residual.numel());ocp.cost.W_e=2*np.eye(geometry.numel())
        ocp.cost.yref=np.zeros(residual.numel());ocp.cost.yref_e=np.zeros(geometry.numel())
        ocp.constraints.x0=np.zeros(self.nx)
        ocp.constraints.idxbu=np.arange(self.nu)
        lower=engine.model.actuator_ctrlrange[:,0].copy();upper=engine.model.actuator_ctrlrange[:,1].copy()
        unlimited=~engine.model.actuator_ctrllimited.astype(bool)
        margin=np.r_[np.full(3,.1),np.full(3,.5),np.ones(self.nu-6)]
        lower[unlimited]=np.min(engine.sequence.hand,axis=0)[unlimited]-margin[unlimited]
        upper[unlimited]=np.max(engine.sequence.hand,axis=0)[unlimited]+margin[unlimited]
        self.lower=lower;self.upper=upper
        ocp.constraints.lbu=lower/self.hand_scale;ocp.constraints.ubu=upper/self.hand_scale
        opts=ocp.solver_options;opts.N_horizon=horizon;opts.tf=horizon*self.dt
        opts.integrator_type='DISCRETE';opts.nlp_solver_type='SQP_RTI'
        opts.qp_solver='PARTIAL_CONDENSING_HPIPM';opts.qp_solver_cond_N=min(5,horizon)
        opts.hessian_approx='GAUSS_NEWTON'
        opts.levenberg_marquardt=levenberg_marquardt;opts.qp_solver_iter_max=100;opts.print_level=0
        opts.cost_scaling=np.ones(horizon+1)
        ocp.code_gen_options.model_external_shared_lib_dir=str(self.cache)
        ocp.code_gen_options.model_external_shared_lib_name='fa_native_bridge'
        self.cache.mkdir(parents=True,exist_ok=True)
        _set_acados_code_export_directory(ocp,str(self.cache/'solver'))
        directory=self.cache/'solver'
        self.solver=build_cached_solver(ocp,str(directory/(model.name+'.json')),
            str(directory/('libacados_ocp_solver_'+model.name+'.so')),prune_generated=False)
        self.symbolic_physics=physics;self.symbolic_cost=cost
        self.residual_function=cs.Function('fa_stage_residual',[x,u,p],[residual])
        self.geometry_function=cs.Function('fa_geometry_residual',[x,p],[geometry])
        self.pose_tracking_function=cs.Function('fa_pose_tracking_value',[x,p],[tracking])
        self.commands=None;self.previous_previous=None;self.previous=None
        self.relative_warm=relative_warm;self.command_reference=None
        self.templates=None
        self.save_prediction=False

    def rollout(self,physical,commands,previous,previous_previous):
        states=[np.r_[physical,previous,previous_previous]/self.xscale]
        for servo in commands:
            f,_=self.native.physics(physical,servo)
            physical=f[:self.nphysical]
            states.append(np.r_[physical,servo,previous]/self.xscale)
            previous_previous,previous=previous,servo
        return np.asarray(states)

    def trajectory_cost(self,states,commands,parameters):
        value=0.
        for k in range(self.N):
            r=np.asarray(self.residual_function(states[k],commands[k]/self.hand_scale,parameters[k])).ravel()
            value+=float(r@r)
        terminal=np.asarray(self.geometry_function(states[-1],parameters[-1])).ravel()
        value+=float(terminal@terminal)
        return value

    def solve(self,qpos,qvel,targets,previous,*,execution_steps=3):
        from planning.acados_env import quiet_acados_stderr
        t=perf_counter()
        original_targets=targets
        if self.templates is not None:targets=self.templates.apply(targets)
        self.native.prepare(targets);physical=np.r_[qpos,qvel]
        previous_previous=previous if self.previous_previous is None else self.previous_previous
        if self.commands is None:
            commands=np.asarray(targets.dense_actions).copy()
            lag=-self.engine.model.actuator_biasprm[:,2]/self.engine.model.actuator_gainprm[:,0]
            commands+=np.gradient(commands,self.dt,axis=0)*lag
        else:
            correction=self.commands-self.command_reference if self.relative_warm else self.commands
            commands=np.concatenate([correction[execution_steps:],np.repeat(correction[-1:],execution_steps,axis=0)])
            if self.relative_warm:commands=commands+targets.dense_actions
        commands=np.clip(commands,self.lower,self.upper)
        states=self.rollout(physical,commands,previous,previous_previous)
        if self.templates is not None:
            targets=self.templates.update(qpos,qvel,targets,states)
            physics_cache=self.native.cache.copy();timings=self.native.timing.copy();calls=self.native.calls.copy()
            self.native.prepare(targets)
            self.native.cache=physics_cache;self.native.timing=timings;self.native.calls=calls
        prepared=perf_counter();s=self.solver
        statuses=[];qp_statuses=[];qp_iterations=[];solver_times=[]
        initial_commands=commands.copy()
        parameters=[]
        for k in range(self.N+1):
            target_step=min(k,self.N-1)
            reference_scale=self.engine.cfg.loss.terminal if k==self.N else (0. if k==self.N-1 else 1.)
            parameters.append(np.r_[k,targets.dense_actions[target_step],targets.object_qpos[target_step],reference_scale])
        costs=[self.trajectory_cost(states,commands,parameters)]
        accepted=0;trial_details=[]
        for iteration in range(self.rti_steps):
            for k in range(self.N+1):
                s.set(k,'x',states[k]);s.set(k,'p',parameters[k])
                if k<self.N:
                    s.set(k,'u',commands[k]/self.hand_scale)
                    s.constraints_set(k,'lbu',np.maximum(self.lower/self.hand_scale,commands[k]/self.hand_scale-self.trust))
                    s.constraints_set(k,'ubu',np.minimum(self.upper/self.hand_scale,commands[k]/self.hand_scale+self.trust))
            s.set(0,'lbx',states[0]);s.set(0,'ubx',states[0])
            with quiet_acados_stderr():status=s.solve()
            if self.native.error is not None:raise RuntimeError('Native external-function callback failed') from self.native.error
            statuses.append(int(status));solver_times.append(float(s.get_stats('time_tot')))
            stats=s.get_stats('statistics')
            qp_statuses.append(int(stats[1,-1]));qp_iterations.append(int(stats[2,-1]))
            if status:break
            trial=np.asarray([s.get(k,'u') for k in range(self.N)])*self.hand_scale
            trial=np.clip(trial,self.lower,self.upper)
            # Hard contacts can invalidate a large local SQP step. Restore
            # nonlinear dynamics feasibility and accept on the same true FA
            # objective before executing; no sampled forces enter this rollout.
            improved=False
            for alpha in (2.**-j for j in range(self.line_search_steps)):
                candidate=commands+alpha*(trial-commands)
                try:
                    proposed=self.rollout(physical,candidate,previous,previous_previous)
                    value=self.trajectory_cost(proposed,candidate,parameters)
                except FloatingPointError:value=float('inf')
                trial_details.append(dict(iteration=iteration,alpha=alpha,cost=float(value),baseline_cost=float(costs[-1])))
                if np.isfinite(value) and value<costs[-1]-1e-9:
                    commands=candidate;states=proposed;costs.append(value);accepted+=1;improved=True;break
            if not improved:break
        # Every accepted control sequence has passed a feasible nonlinear
        # rollout. The executor receives servo targets, never ideal forces.
        self.commands=commands.copy()
        self.command_reference=np.asarray(targets.dense_actions).copy()
        self.previous_previous=(commands[execution_steps-2] if execution_steps>1 else previous).copy()
        elapsed=perf_counter()-t
        self.last=dict(wall_s=elapsed,prepare_rollout_s=prepared-t,statuses=statuses,
            qp_statuses=qp_statuses,qp_iterations=qp_iterations,
            control_update_max=float(np.max(np.abs(commands-initial_commands))),
            accepted_SQP_steps=accepted,true_FA_costs=costs,
            acados_s=solver_times,native_seconds=dict(self.native.timing),native_calls=dict(self.native.calls),
            solver='direct_multiple_shooting_native_ComFree',analytic_jacobians=True,
            full_contact_frame_jacobian=self.native.full_contact_jacobian,
            cost_Hessian='Gauss-Newton; individual FA residuals',horizon_steps=self.N)
        self.last['line_search_steps']=self.line_search_steps
        self.last['post_step_tracking']=self.post_step_tracking
        self.last['reference_relative_warm']=self.relative_warm
        if self.templates is not None:self.last['template_update']=self.templates.last.copy()
        if self.save_prediction:
            self.last['trial_details']=trial_details
            self.last_prediction=dict(states=states.copy(),commands=commands.copy(),parameters=np.asarray(parameters),
                previous=np.asarray(previous).copy(),previous_previous=np.asarray(previous_previous).copy(),
                targets=targets,original_targets=original_targets)
        return commands,self.last
