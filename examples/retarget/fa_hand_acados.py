"""Condensed single-shooting SQP/RTI hand updates using acados HPIPM.

The local control QP uses native ComFree/FA analytic gradients and a positive
BFGS Hessian. ComFree remains the nonlinear evaluation oracle; this module
does not claim to compile the Warp simulator into acados dynamics.
"""
from time import perf_counter
import numpy as np


class CondensedHandQP:
    def __init__(self,dimension):
        import casadi as cs
        from planning.mpc_explicit import _load_acados,_set_acados_code_export_directory
        from planning.acados_env import acados_cache_directory,build_cached_solver
        Model,Ocp,_=_load_acados()
        n=dimension;x=cs.MX.sym('dummy_state',1);u=cs.MX.sym('control_step',n)
        p=cs.MX.sym('local_model',n+n*n);gradient=p[:n];H=cs.reshape(p[n:],n,n)
        model=Model();model.name=f'fa_condensed_hand_qp_v2_n{n}'
        model.x=x;model.u=u;model.p=p;model.disc_dyn_expr=0*x
        model.cost_expr_ext_cost=cs.dot(gradient,u)+.5*cs.mtimes([u.T,H,u])
        model.cost_expr_ext_cost_custom_hess=cs.diagcat(H,cs.MX.zeros(1,1))
        model.cost_expr_ext_cost_e=0*x[0]
        ocp=Ocp();ocp.model=model;ocp.parameter_values=np.r_[np.zeros(n),np.eye(n).ravel(order='F')]
        ocp.cost.cost_type='EXTERNAL';ocp.cost.cost_type_e='EXTERNAL'
        ocp.constraints.x0=np.zeros(1)
        ocp.constraints.idxbu=np.arange(n);ocp.constraints.lbu=-np.ones(n);ocp.constraints.ubu=np.ones(n)
        opts=ocp.solver_options;opts.N_horizon=1;opts.tf=1.
        opts.integrator_type='DISCRETE';opts.nlp_solver_type='SQP_RTI';opts.qp_solver='FULL_CONDENSING_HPIPM'
        opts.hessian_approx='EXACT';opts.exact_hess_dyn=False;opts.print_level=0
        opts.cost_scaling=np.ones(2);opts.qp_solver_iter_max=100
        directory=acados_cache_directory(model.name);directory.mkdir(parents=True,exist_ok=True)
        _set_acados_code_export_directory(ocp,str(directory))
        self.solver=build_cached_solver(ocp,str(directory/(model.name+'.json')),
            str(directory/('libacados_ocp_solver_'+model.name+'.so')))
        self.last_seconds=0.;self.last_native_seconds=0.

    def solve(self,gradient,H,trust):
        started=perf_counter();s=self.solver;n=len(gradient)
        s.reset();s.set(0,'x',np.zeros(1));s.set(1,'x',np.zeros(1));s.set(0,'u',np.zeros(n))
        s.set(0,'p',np.r_[gradient,H.ravel(order='F')])
        s.constraints_set(0,'lbu',-np.full(n,trust));s.constraints_set(0,'ubu',np.full(n,trust))
        status=int(s.solve());step=np.asarray(s.get(0,'u')).reshape(-1)
        self.last_native_seconds=float(np.asarray(s.get_stats('time_tot')).reshape(-1)[0])
        self.last_seconds=perf_counter()-started
        return step,status


def solve_batched(evaluate,variables,iterations,qp,*,trust=1.,line_search=8):
    value,gradient=evaluate(variables,True);initial=value.copy();worlds,n=variables.shape
    H=np.array([np.eye(n)*max(np.max(np.abs(g)),1.) for g in gradient])
    accepted_steps=0;statuses=[];qp_wall=0.;qp_native=0.
    for iteration in range(iterations):
        direction=np.zeros_like(variables);status=[]
        for world in range(worlds):
            step,code=qp.solve(gradient[world],H[world],trust)
            if code==0 and np.isfinite(step).all():direction[world]=step
            status.append(code);qp_wall+=qp.last_seconds;qp_native+=qp.last_native_seconds
        statuses.append(status)
        slope=np.sum(gradient*direction,axis=1);pending=np.isfinite(value)&(slope<0)
        alpha=np.ones(worlds);new=variables.copy();trial=variables.copy();accepted=np.zeros(worlds,bool)
        for _ in range(line_search):
            if not pending.any():break
            trial[pending]=variables[pending]+alpha[pending,None]*direction[pending]
            candidate,_=evaluate(trial,False)
            good=pending&(candidate<=value+1.e-4*alpha*slope)
            new[good]=trial[good];accepted|=good;pending&=~good;alpha[pending]*=.5
        accepted_steps+=int(accepted.sum())
        if not accepted.any():break
        if iteration+1==iterations:
            variables=new
            break
        next_value,next_gradient=evaluate(new,True)
        for world in np.flatnonzero(accepted):
            step=new[world]-variables[world];change=next_gradient[world]-gradient[world]
            hs=H[world]@step;shs=step@hs;sy=step@change
            if shs>1.e-12:
                # Powell damping preserves positive definiteness.
                if sy<.2*shs:
                    theta=.8*shs/(shs-sy);change=theta*change+(1-theta)*hs;sy=step@change
                if sy>1.e-12:H[world]+=np.outer(change,change)/sy-np.outer(hs,hs)/shs
        variables,value,gradient=new,next_value,next_gradient
    final,_=evaluate(variables,False)
    return variables,accepted_steps,dict(backend='acados_condensed_SQP_RTI',updates=len(statuses),
        statuses=statuses,qp_wall_seconds=qp_wall,qp_native_seconds=qp_native,
        initial_values=initial.tolist(),final_values=final.tolist(),
        model='native_ComFree_oracle_analytic_gradient_BFGS_control_QP')
