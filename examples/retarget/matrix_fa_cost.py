"""Complete FA objective on the current controller's state/control time grid.

One RTI QP, unchanged SCM and candidate family. An added command-history state
exists solely for the acceleration penalty. Geometry residuals use the official
FA collision/FK observation and are linearized at each nominal stage.
"""
from pathlib import Path
import types
import hashlib
import numpy as np
import casadi as cs


class FACost:
    def __init__(self,mpc,engine):
        self.mpc=mpc;self.engine=engine;self.wp=engine.wp;self.buffers={};self.previous_command=np.zeros(26)
        self.validation_max_relative=0.;self.solver=None
        from forceaware.mpc_loss import compile_model,make_weights
        from forceaware.time_grid import TimeGrid
        self.model=compile_model(engine.model,engine.index,engine.compiled.base.collision,hand='wuji_right',guidance_count=engine.sequence.contact.mask.shape[1],device=engine.device)
        self.weights=make_weights(engine.cfg)
        self.nr=30+4*self.model.guidance_count+2*self.model.output_count+self.model.finger_joint_pair_count
        h=mpc.param_.h_
        self.grid=TimeGrid.create(mpc_dt=h,action_dt=h,knot_dt=h,exec_dt=h,ref_dt=1/30)
        self.velocity_scale=np.r_[np.full(3,10.),np.full(3,3.),np.ones(20)]
        self.acceleration_scale=np.r_[np.full(6,50.),np.full(20,500.)]
        self.component_ratio=np.r_[np.ones(6),np.full(20,.3)]
        self.state_vscale=np.r_[np.ones(3),np.full(3,3.),np.full(20,4.)]
        self.state_ascale=np.r_[np.full(3,50.),np.full(23,100.)]
        self.targets=None

    def window(self):
        from forceaware.targets import make_window
        self.targets=make_window(self.engine.sequence,executed_steps=int(round(self.mpc.hand.data.time/self.grid.mpc_dt)),
            horizon=self.mpc.param_.mpc_horizon_,time_grid=self.grid,start_frame=0.,end_frame=len(self.engine.sequence.hand)-1)

    def _buffer(self,worlds):
        if worlds in self.buffers:return self.buffers[worlds]
        wp=self.wp;e=self.engine
        from comfree_warp.native_adjoint.observation_adjoint import allocate_workspace,forward
        from forceaware.mpc_loss import LossState,LossJob,allocate_target,build_workspace
        q=wp.array(np.tile(e.model.qpos0,(worlds,1)).astype(np.float32),device=e.device)
        obs=allocate_workspace(e.compiled.base,worlds,requires_grad=False)
        result=forward(e.compiled.base,q,obs)
        state=LossState();state.qpos=q
        for k in ['body_position','body_matrix','contact_distance','contact_position','contact_frame']:setattr(state,k,getattr(result,k))
        state.contact_active=obs.collision.contacts.active
        job=LossJob();job.model=self.model;job.weights=self.weights;job.state=state
        job.target=allocate_target(26,self.model.guidance_count,e.device)
        workspace=build_workspace(job,worlds,e.device)
        r=wp.empty((worlds,self.nr),device=e.device)
        b=dict(q=q,obs=obs,loss=workspace,residual=r)
        self.buffers[worlds]=b;return b

    def residual(self,states,stage,*,terminal=False,freeze=False):
        from comfree_warp.native_adjoint.observation_adjoint import forward
        from forceaware.mpc_loss import assign_target,_prepare_auxiliary,_loss
        from examples.retarget.matrix_fa_residual import residual,freeze_marked
        states=np.atleast_2d(states);count=len(states);capacity=128 if count<=128 else 256
        if count>capacity:raise ValueError('FA cost batch capacity exceeded')
        b=self._buffer(capacity);q,_=self.engine.unpack(states)
        q=np.concatenate([q,np.repeat(q[:1],capacity-count,axis=0)])
        b['q'].assign(q.astype(np.float32));forward(self.engine.compiled.base,b['q'],b['obs'])
        workspace=b['loss'];job=workspace.job
        last=self.mpc.param_.mpc_horizon_-1
        assign_target(workspace.target,self.targets,last if terminal else stage,contact_step=None if terminal else max(stage-1,0),grasp=False)
        job.reference_scale=2. if terminal else (0. if stage==last else 1.)
        job.guidance_enabled=int(terminal or stage>0);job.query_enabled=job.guidance_enabled;job.penetration_enabled=1
        wp=self.wp
        wp.launch(_prepare_auxiliary,dim=(capacity,self.model.guidance_count),inputs=[job],device=self.engine.device)
        if freeze:wp.launch(freeze_marked,dim=(capacity,self.model.guidance_count),inputs=[job.auxiliary.marked_query_weight],device=self.engine.device)
        wp.launch(residual,dim=capacity,inputs=[job],outputs=[b['residual']],device=self.engine.device)
        result=b['residual'].numpy()[:count].astype(float)
        if not freeze:
            wp.launch(_loss,dim=capacity,inputs=[job],outputs=[workspace.loss],device=self.engine.device)
            original=workspace.loss.numpy()[:count]
            relative=np.max(np.abs(np.sum(result**2,axis=1)-original)/np.maximum(original,1.))
            self.validation_max_relative=max(self.validation_max_relative,float(relative))
            if relative>2e-5:raise RuntimeError(f'FA residual differs from original loss: {relative}')
        return result

    def linearization(self,x,stage,terminal=False):
        epsilon=np.r_[np.full(3,1e-4),np.full(4,2e-4),np.full(3,1e-4),np.full(23,1e-3)]
        batch=np.tile(x,(67,1))
        for k in range(33):batch[1+2*k,k]+=epsilon[k];batch[2+2*k,k]-=epsilon[k]
        values=self.residual(batch,stage,terminal=terminal,freeze=True)
        J=((values[1::2]-values[2::2])/(2*epsilon[:,None])).T
        return values[0],J

    def trajectory_scores(self,initial,candidates):
        p=self.mpc.param_;h=p.h_;count=len(candidates);scores=np.zeros(count)
        if not count:return scores
        histories=np.array([np.vstack([initial,c['native_states']]) for c in candidates])
        commands=np.array([c['commands'] for c in candidates])
        # The physical carried target increments are the controls in our
        # parameterization. Acceleration history couples successive stages.
        targets=histories[:,:,-26:];increments=np.diff(targets,axis=1)
        previous=np.broadcast_to(self.previous_command,(count,26))
        for k in range(p.mpc_horizon_):
            r=self.residual(histories[:,k],k);scores+=np.sum(r*r,axis=1)
            dv=increments[:,k]
            scores+=np.sum(self.component_ratio*(dv/(h*self.velocity_scale))**2,axis=1)
            scores+=np.sum(((dv-previous)/(h*h*self.acceleration_scale))**2,axis=1)
            velocity=histories[:,k+1,39:65];acceleration=(velocity-histories[:,k,39:65])/h
            scores+=np.sum(self.component_ratio*((velocity/self.state_vscale)**2+2*(acceleration/self.state_ascale)**2),axis=1)
            previous=dv
        r=self.residual(histories[:,-1],p.mpc_horizon_-1,terminal=True);scores+=np.sum(r*r,axis=1)
        return scores

    def build_solver(self):
        from planning.mpc_explicit import _load_acados,_set_acados_code_export_directory
        from planning.acados_env import acados_cache_directory,build_cached_solver
        AcadosModel,AcadosOcp,_=_load_acados();p=self.mpc.param_;n=p.mpc_horizon_;nx=91;nu=26;h=p.h_
        x=cs.MX.sym('x',117);u=cs.MX.sym('u',26)
        # Affine local dynamics keep model and cost linearizations separate.
        # The current SCM's exact Jacobians are used for the dynamics rows.
        size=nx+nu+nx+nx*nx+nx*nu+33+self.nr+self.nr*33
        cp=cs.MX.sym('p',size);i=0
        def take(k):
            nonlocal i
            a=cp[i:i+k];i+=k;return a
        x0=take(nx);u0=take(nu);f0=take(nx);A=cs.reshape(take(nx*nx),nx,nx);B=cs.reshape(take(nx*nu),nx,nu)
        obs0=take(33);r0=take(self.nr);J=cs.reshape(take(self.nr*33),self.nr,33)
        nxt=f0+A@(x[:nx]-x0)+B@(u-u0)
        increment=nxt[-26:]-x[65:91]
        velocity=nxt[39:65];acceleration=(velocity-x[39:65])/h
        smoothing=cs.vertcat(cs.DM(np.sqrt(self.component_ratio))*(increment/(h*cs.DM(self.velocity_scale))),
            (increment-x[91:])/(h*h*cs.DM(self.acceleration_scale)),
            cs.DM(np.sqrt(self.component_ratio))*(velocity/cs.DM(self.state_vscale)),
            cs.DM(np.sqrt(2*self.component_ratio))*(acceleration/cs.DM(self.state_ascale)))
        geometry=r0+J@(x[:33]-obs0);r=cs.vertcat(geometry,smoothing)
        model=AcadosModel();key=hashlib.sha1((str(self.nr)+'FA_full_original_units_v5_unscaled_sum').encode()).hexdigest()[:10]
        model.name='retarget_fa_cost_'+key;model.x=x;model.u=u;model.p=cp
        model.disc_dyn_expr=cs.vertcat(nxt,increment)
        model.cost_y_expr=r;model.cost_y_expr_e=geometry
        ocp=AcadosOcp();ocp.model=model;ocp.parameter_values=np.zeros(size)
        ocp.cost.cost_type='NONLINEAR_LS';ocp.cost.cost_type_e='NONLINEAR_LS'
        ocp.cost.W=2*np.eye(self.nr+104);ocp.cost.W_e=2*np.eye(self.nr)
        ocp.cost.yref=np.zeros(self.nr+104);ocp.cost.yref_e=np.zeros(self.nr)
        ocp.constraints.idxbu=np.arange(26);ocp.constraints.lbu=p.mpc_u_lb_;ocp.constraints.ubu=p.mpc_u_ub_
        # History is a cost state; it has no additional physical constraint.
        ocp.constraints.idxbx=np.arange(91);ocp.constraints.lbx=p.mpc_q_lb_;ocp.constraints.ubx=p.mpc_q_ub_
        ocp.constraints.idxbx_e=np.arange(91);ocp.constraints.lbx_e=p.mpc_q_lb_;ocp.constraints.ubx_e=p.mpc_q_ub_
        ocp.constraints.x0=np.zeros(117)
        ocp.solver_options.N_horizon=n;ocp.solver_options.tf=h*n
        ocp.solver_options.integrator_type='DISCRETE';ocp.solver_options.nlp_solver_type='SQP_RTI'
        ocp.solver_options.qp_solver='FULL_CONDENSING_HPIPM';ocp.solver_options.levenberg_marquardt=.01
        ocp.solver_options.hessian_approx='GAUSS_NEWTON';ocp.solver_options.print_level=0
        # FA sums its residuals without an extra stage-duration multiplier.
        ocp.solver_options.cost_scaling=np.ones(n+1)
        directory=acados_cache_directory(model.name);directory.mkdir(exist_ok=True,parents=True)
        _set_acados_code_export_directory(ocp,str(directory))
        self.solver=build_cached_solver(ocp,str(directory/(model.name+'.json')),str(directory/('libacados_ocp_solver_'+model.name+'.so')),prune_generated=True)
        self.step_jacobian=cs.Function('scm_fa_dyn_jac',self.mpc.physics.prediction_fn.sx_in(),
            [self.mpc.physics.prediction_fn(*self.mpc.physics.prediction_fn.sx_in()),
             cs.jacobian(self.mpc.physics.prediction_fn(*self.mpc.physics.prediction_fn.sx_in()),self.mpc.physics.prediction_fn.sx_in(0)),
             cs.jacobian(self.mpc.physics.prediction_fn(*self.mpc.physics.prediction_fn.sx_in()),self.mpc.physics.prediction_fn.sx_in(1))])

    def solve(self,current,phi,jac,cost,guess,u_lb,u_ub):
        from planning.acados_env import quiet_acados_stderr
        if self.solver is None:self.build_solver()
        s=self.solver;p=self.mpc.param_;n=p.mpc_horizon_
        if guess is None:s.reset()
        commands=np.zeros((n,26)) if guess is None else np.asarray(guess['u_traj']).copy()
        commands=np.clip(commands,u_lb,u_ub)
        states=[current];aug=[np.r_[current,self.previous_command]];parameters=[]
        mp=self.mpc.physics.runtime_parameters(1.)
        for k in range(n):
            f,A,B=self.step_jacobian(states[-1],commands[k],phi,jac,mp)
            f=np.array(f).ravel();A=np.array(A);B=np.array(B)
            r,J=self.linearization(states[-1],k)
            parameters.append(np.r_[states[-1],commands[k],f,A.ravel(order='F'),B.ravel(order='F'),states[-1][:33],r,J.ravel(order='F')])
            states.append(f);aug.append(np.r_[f,f[-26:]-aug[-1][65:91]])
        r,J=self.linearization(states[-1],n-1,terminal=True)
        terminal=parameters[-1].copy();terminal[-(33+self.nr+self.nr*33):]=np.r_[states[-1][:33],r,J.ravel(order='F')]
        for k in range(n):
            s.set(k,'x',aug[k]);s.set(k,'u',commands[k]);s.set(k,'p',parameters[k])
            s.constraints_set(k,'lbu',u_lb);s.constraints_set(k,'ubu',u_ub)
        s.set(n,'x',aug[-1]);s.set(n,'p',terminal)
        s.set(0,'lbx',aug[0]);s.set(0,'ubx',aug[0])
        self.mpc.acados_solve_count+=1
        with quiet_acados_stderr():status=s.solve()
        if status:
            self.mpc.last_acados_error=f'FA cost acados status {status}'
            self.mpc.acados_failure_count+=1;return None
        u=np.array([s.get(k,'u') for k in range(n)]);u=np.clip(u,u_lb,u_ub)
        x=[current]
        for command in u:x.append(np.asarray(self.mpc.physics.prediction_fn(x[-1],command,phi,jac,mp)).ravel())
        x=np.array(x)
        return dict(action=u[0],u_traj=u,rollout_q=x[1:],sol_guess=dict(u_traj=u,x_traj=x),
            cost_opt=np.array([0.]),solve_status='ACADOS_SUCCESS (FA cost, sqp_iter=1)',solver_backend='acados',requested_solver='acados')


def attach_fa_cost(mpc,engine):
    objective=FACost(mpc,engine);original_plan=mpc.plan
    def plan(self,*args,**kwargs):
        objective.window();before=self.hand.servo.copy()
        result=original_plan(*args,**kwargs)
        objective.previous_command=result[0]-before
        result[1]['cost_profile']='full_FA_formulas_current_grid'
        result[1]['FA_loss_value_validation_max_relative']=objective.validation_max_relative
        return result
    mpc.plan=types.MethodType(plan,mpc)
    mpc._plan_once_acados=types.MethodType(lambda self,*args,**kwargs:objective.solve(*args,**kwargs),mpc)
    mpc.trajectory_objective=objective.trajectory_scores
    mpc.fa_objective=objective
    return mpc
