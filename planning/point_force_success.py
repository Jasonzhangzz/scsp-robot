"""Paired CSO success experiment: new SCM, pinned ComFree law, native QP.

The common object-centric prediction is one 20 ms step with a frozen measured
contact set. Execution is independent MuJoCo, with continuous point force.
"""
import copy
import time
import numpy as np
from scipy.spatial.transform import Rotation
from planning.point_force_benchmark import PointForcePlanner


class SuccessPlanner(PointForcePlanner):
    def __init__(self,model,points,frames,target,args,model_name,seed):
        cfg=copy.copy(args);cfg.solver='adamw'
        super().__init__(model,points,frames,target,cfg,model_name,seed)
        self.args=args;self.solver=args.solver
        self.backend={'lm':'numpy-batched-lm','gpu-lm':'cuda-batched-lm-finite-difference','cma-es':'numpy-batched-full-covariance-cma-es',
                      'ipopt':'casadi-ipopt-exact-ad-batched','cem':'cuda-cem',
                      'adamw':'cpu-adamw-finite-difference'}[args.solver]
        self.warm=np.tile([.1,.5,.5],(len(points),1))
        # The QP evaluator is the native pyramidal primal objective on CUDA.
        # Its predictions are checked against CPU mj_forward before every trial.
        if model_name=='qp' or args.solver=='cem':
            from planning.point_force_gpu import GPUForceCEM
            self.gpu=GPUForceCEM(points,frames,args,model_name)
        if args.solver=='gpu-lm':
            from planning.point_force_lm_gpu import GPUForceLM
            self.gpu=GPUForceLM(points,frames,args,model_name)
        if args.solver=='cem':self.backend='cuda-cem'
        elif model_name=='qp':self.backend='cuda-contact-'+args.solver
        self.validation=None

    def _evaluate(self,raw,indices):
        raw=np.clip(np.asarray(raw,dtype=float).reshape(-1,3),0,1)
        indices=np.asarray(indices,dtype=int)
        if self.name=='qp':
            return self.gpu.evaluate_raw(raw,indices)
        return super()._evaluate(raw,indices)

    def validate(self,snapshot,config):
        self.prepare(snapshot,config)
        indices=np.flatnonzero(self.valid)
        raw=np.tile([.1,.55,.45],(len(indices),1))
        if self.gpu is None:
            _,v,p,c=self._evaluate(raw,indices)
            assert np.isfinite(np.r_[v.ravel(),p.ravel(),c]).all()
            result=dict(passed=True,backend=self.backend,law=self.name)
        else:
            cmd,v,p,c=self.gpu.evaluate_raw(raw,indices)
            ev,ep,ec,_=self.predict_numpy(cmd,indices)
            delta=float(np.max(np.abs(v-ev)))
            cost_delta=float(np.max(np.abs(c-ec)))
            tolerance=2e-3 if self.name=='qp' else 2e-9
            result=dict(passed=delta<tolerance and cost_delta<(2e-3 if self.name=='qp' else 2e-8),
                        backend=self.backend,max_velocity_error=delta,max_cost_error=cost_delta)
            if not result['passed']:raise AssertionError(f'Contact model parity failed: {result}')
        self.validation=result
        return result

    def residual(self,command,velocity,poses):
        c=np.asarray(self.config)
        q=poses[:,3:];q=q/np.linalg.norm(q,axis=1,keepdims=True)
        qd=self.target[3:]/np.linalg.norm(self.target[3:])
        delta=poses[:,:3]-self.target[:3]
        if len(c)>9 and c[9]>0:
            planar=delta-np.sum(delta*c[6:9],axis=1,keepdims=True)*c[6:9]
            norm=np.linalg.norm(planar,axis=1,keepdims=True)
            delta=planar*np.maximum(norm-c[9],0)/np.maximum(norm,1e-30)
        axis=c[3:6]+2*np.cross(q[:,1:],np.cross(q[:,1:],c[3:6])+q[:,:1]*c[3:6])
        barrier=np.zeros((len(q),1)) if len(c)<12 else np.sqrt(c[11])*np.maximum(1-np.sum(axis*c[6:9],axis=1,keepdims=True)-c[10],0)
        return np.column_stack((np.sqrt(c[0])*delta,
            np.sqrt(c[1])*(q-np.sum(q*qd,axis=1,keepdims=True)*qd),
            np.sqrt(c[2]/4)*(axis-c[6:9]),barrier,
            np.sqrt(self.cost_args.linear_velocity_coef)*velocity[:,:3],
            np.sqrt(self.cost_args.angular_velocity_coef)*velocity[:,3:],
            np.sqrt(self.args.force_reg_coef)*command,
            np.sqrt(self.args.friction_reg_coef)*command[:,1:]))

    def _lm(self,iterations):
        indices=np.flatnonzero(self.valid)
        raw=self.warm[indices].copy()
        damping=np.full(len(indices),1e-3)
        evaluations=0;best=None
        # Include zero drive and a full normal-force seed at every contact.
        starts=np.stack((raw,np.tile([0.,.5,.5],(len(indices),1)),np.tile([1.,.5,.5],(len(indices),1))),axis=1)
        cmd,vel,pose,cost=self._evaluate(starts.reshape(-1,3),np.repeat(indices,3))
        cost=cost.reshape(-1,3);winner=np.argmin(cost,axis=1)
        raw=starts[np.arange(len(indices)),winner]
        for iteration in range(iterations):
            eps=1e-4
            batch=np.repeat(raw[:,None,:],7,axis=1)
            for axis in range(3):
                batch[:,1+2*axis,axis]+=eps;batch[:,2+2*axis,axis]-=eps
            cmd,vel,pose,cost=self._evaluate(batch.reshape(-1,3),np.repeat(indices,7))
            residual=self.residual(cmd,vel,pose).reshape(len(indices),7,-1)
            base=residual[:,0];cost=cost.reshape(-1,7)[:,0]
            candidate=self._best_record(indices,cmd.reshape(-1,7,3)[:,0],vel.reshape(-1,7,6)[:,0],pose.reshape(-1,7,7)[:,0],cost)
            if best is None or candidate['cost']<best['cost']:best=candidate
            jac=np.stack([(residual[:,1+2*k]-residual[:,2+2*k])/(2*eps) for k in range(3)],axis=-1)
            hessian=np.einsum('nri,nrj->nij',jac,jac)
            grad=np.einsum('nri,nr->ni',jac,base)
            diagonal=np.maximum(np.diagonal(hessian,axis1=1,axis2=2),1.)
            system=hessian+np.eye(3)[None]*damping[:,None,None]*diagonal[:,None,:]
            step=np.linalg.solve(system,-grad[...,None])[...,0]
            proposed=np.clip(raw+step,0,1)
            pc,pv,pp,score=self._evaluate(proposed,indices)
            evaluations+=8*len(indices)
            accept=score<cost
            raw[accept]=proposed[accept]
            damping=np.clip(damping*np.where(accept,.4,4.),1e-9,1e9)
            candidate=self._best_record(indices,pc,pv,pp,score)
            if candidate['cost']<best['cost']:best=candidate
            if np.max(np.linalg.norm(proposed-self.warm[indices],axis=1))<1e-8:break
        self.warm[indices]=raw
        best.update(alpha=None,residual=None,evaluations=evaluations+3*len(indices),
                    solver_backend=self.backend,gpu_parallel_prediction=self.name=='qp',search_iterations=iteration+1)
        return best

    def solve(self,snapshot,config,iterations=None):
        if self.solver not in ('lm','gpu-lm','adamw'):return super().solve(snapshot,config,iterations)
        started=time.perf_counter()
        self.prepare(snapshot,config)
        if self.solver=='gpu-lm':
            result=self.gpu.solve(self.args.lm_iterations if iterations is None else iterations)
        elif self.solver=='lm':
            result=self._lm(self.args.lm_iterations)
        else:
            result=self._adamw(getattr(self.args,'adamw_iterations',20) if iterations is None else iterations)
            result.update(solver_backend=self.backend,gpu_parallel_prediction=self.name=='qp')
        index=result['selected_idx']
        result.update(p_local=self.points[index].copy(),frame=self.frames[index].copy(),
            force_local=self.frames[index]@result['lam'],observation_time=snapshot[2],observation_step=snapshot[3],
            planning_state_pose=self.pose.copy(),objective_config=np.asarray(config).copy(),
            velocity_cost_coefficients=[self.cost_args.linear_velocity_coef,self.cost_args.angular_velocity_coef],
            model=self.name,planning_round=self.round+1,planning_time_ms=1000*(time.perf_counter()-started),
            solver_fallback_reason=None,search_budget_ms=0.,control_deadline_exceeded=False,
            contact_solver=self.name,prediction_substep_evaluations=result['evaluations'])
        self.round+=1
        return result

    def _ipopt(self,iterations):
        """One sparse, separable NLP solves all candidate forces with exact AD."""
        import casadi as cs
        from examples.mpc.franka.scm.residual_scm import spectral_residual_velocity_cs
        from planning.object_pose_cost import object_cost_cs,extended_cost_config
        rows=len(self.problem['gap']);count=len(self.points)
        key=(rows,int(iterations))
        if key not in self._ipopt_solvers:
            fields=[('pose',7,1),('target',7,1),('inverse',6,6),('free',6,1),
                    ('jac',rows,6),('gap',rows,1),('config',12,1),('vcoef',2,1),('lipschitz',1,1)]
            symbols={name:cs.MX.sym(name,n,m) for name,n,m in fields}
            packed=cs.vertcat(*[cs.vec(symbols[name]) for name,_,_ in fields])
            raw=cs.MX.sym('raw',3);frame=cs.MX.sym('frame',3,3);point=cs.MX.sym('point',3)
            # Same feasible set as force_commands/bounded_force: Coulomb cone
            # cap further limited by the per-interval object delta-v caps.
            body=self.model.body('obj').id
            rotation=Rotation.from_quat(self.model.body_iquat[body][[1,2,3,0]]).as_matrix()
            inverse_inertia=np.linalg.inv(rotation@np.diag(self.model.body_inertia[body])@rotation.T)
            tangent=2*raw[1:]-1
            tangent/=cs.fmax(1,cs.sum1(cs.fabs(tangent)))
            direction=cs.vertcat(1,self.args.arm_friction*tangent)
            body_force=frame@direction
            angular=cs.DM(inverse_inertia)@cs.cross(point-cs.DM(self.model.body_ipos[body]),body_force)
            cap=cs.fmin(self.args.max_contact_force/np.sqrt(1+self.args.arm_friction**2),
                float(self.model.body_mass[body])*getattr(self.args,'max_linear_force_delta_v',1e6)/
                (self.args.policy_dt*cs.fmax(cs.norm_2(body_force),1e-30)))
            cap=cs.fmin(cap,getattr(self.args,'max_angular_force_delta_v',1e6)/
                (self.args.policy_dt*cs.fmax(cs.norm_2(angular),1e-30)))
            command=direction*(raw[0]*cap)
            force=frame@command
            free=symbols['free']+self.args.policy_dt*symbols['inverse']@cs.vertcat(force,cs.cross(point,force))
            velocity=spectral_residual_velocity_cs(free,symbols['jac'],symbols['gap'],symbols['inverse'],
                self.args.policy_dt,self.args.scm_regularization,self.args.softplus_temperature,
                symbols['lipschitz'])
            pose=symbols['pose'];q=pose[3:];q=q/cs.sqrt(cs.sumsqr(q))
            delta=self.args.policy_dt*velocity[:3]
            pos=pose[:3]+delta+2*cs.cross(q[1:],cs.cross(q[1:],delta)+q[0]*delta)
            speed=cs.sqrt(cs.sumsqr(velocity[3:])+1e-24)
            half=.5*self.args.policy_dt*speed
            dw=cs.cos(half);dv=cs.sin(half)/speed*velocity[3:]
            quat=cs.vertcat(q[0]*dw-cs.dot(q[1:],dv),q[0]*dv+dw*q[1:]+cs.cross(q[1:],dv))
            result=cs.vertcat(pos,quat)
            cost=(object_cost_cs(result,symbols['target'],symbols['config'])+
                self.args.force_reg_coef*cs.sumsqr(command)+self.args.friction_reg_coef*cs.sumsqr(command[1:])+
                symbols['vcoef'][0]*cs.sumsqr(velocity[:3])+symbols['vcoef'][1]*cs.sumsqr(velocity[3:]))
            fn=cs.Function(f'candidate_{rows}',[raw,frame,point,packed],[cost])
            X=cs.MX.sym('X',3,count);P=cs.MX.sym('P',packed.numel())
            costs=fn.map(count)(X,cs.horzcat(*[cs.DM(frame) for frame in self.frames]),
                               cs.DM(self.points.T),cs.repmat(P,1,count))
            # Invalid contacts have fixed zero force and zero objective weight.
            mask=cs.MX.sym('mask',count)
            solver=cs.nlpsol(f'batched_ipopt_{rows}','ipopt',dict(x=cs.vec(X),p=cs.vertcat(P,mask),f=costs@mask),
                {'ipopt.print_level':0,'ipopt.sb':'yes','print_time':0,
                 'ipopt.max_iter':int(iterations),'ipopt.tol':1e-4,'ipopt.acceptable_tol':1e-3,
                 'ipopt.acceptable_iter':1,'error_on_fail':False})
            self._ipopt_solvers[key]=(solver,fields)
        solver,fields=self._ipopt_solvers[key]
        values=dict(pose=self.pose,target=self.target,inverse=self.problem['inverse'],free=self.free,
            jac=self.problem['jac'],gap=self.problem['gap'],config=extended_cost_config(self.config),
            vcoef=[self.cost_args.linear_velocity_coef,self.cost_args.angular_velocity_coef],
            lipschitz=self.scm_lipschitz)
        packed=np.concatenate([np.asarray(values[name]).reshape(-1,order='F') for name,_,_ in fields]+[self.valid.astype(float)])
        lb=np.zeros((count,3));ub=np.ones((count,3));ub[~self.valid]=0
        solved=solver(x0=self.warm.reshape(-1),lbx=lb.reshape(-1),ubx=ub.reshape(-1),p=packed)
        raw=np.asarray(solved['x']).reshape(count,3)
        self.warm=np.clip(raw,0,1)
        ids=np.flatnonzero(self.valid)
        cmd,vel,poses,cost=self._evaluate(raw[ids],ids)
        np.testing.assert_allclose(float(solved['f']),np.sum(cost),rtol=1e-7,atol=1e-7,
                                   err_msg='IPOPT objective differs from the common model/objective')
        best=self._best_record(ids,cmd,vel,poses,cost)
        stats=solver.stats()
        best.update(alpha=None,residual=None,evaluations=int(stats.get('n_call_nlp_f',0))*len(ids),
            solver_backend='casadi-ipopt-exact-ad-batched',gpu_parallel_prediction=False,
            search_iterations=int(stats.get('iter_count',0)),nlp_status=stats.get('return_status'),
            nlp_success=bool(stats.get('success',False)))
        return best

    def _cmaes(self,iterations):
        """Positive-weight full-covariance CMA-ES, batched across contacts.

        Standard cumulative step-size and covariance path adaptation in 3D;
        out-of-box samples are repaired by clipping before selection/update.
        """
        ids=np.flatnonzero(self.valid);count=len(ids);n=3;pop=self.args.population_size
        mu=pop//2;weights=np.log(mu+.5)-np.log(np.arange(1,mu+1));weights/=weights.sum()
        mueff=1/np.sum(weights**2)
        cc=(4+mueff/n)/(n+4+2*mueff/n);cs=(mueff+2)/(n+mueff+5)
        c1=2/((n+1.3)**2+mueff);cmu=min(1-c1,2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        damping=1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs
        chi=np.sqrt(n)*(1-1/(4*n)+1/(21*n*n))
        mean=self.warm[ids].copy();sigma=np.full(count,.25)
        cov=np.tile(np.eye(n),(count,1,1));pc=np.zeros_like(mean);ps=np.zeros_like(mean)
        rng=np.random.default_rng(1000003*self.seed+self.args.solver_seed+7919*self.round+1)
        best=None;best_cost=np.full(count,np.inf);best_raw=mean.copy()
        for generation in range(iterations):
            eigen,basis=np.linalg.eigh(cov);eigen=np.maximum(eigen,1e-20)
            factor=basis*np.sqrt(eigen)[:,None,:]
            z=rng.standard_normal((count,pop,n))
            raw=np.clip(mean[:,None,:]+sigma[:,None,None]*np.einsum('npj,nij->npi',z,factor),0,1)
            cmd,vel,poses,cost=self._evaluate(raw.reshape(-1,3),np.repeat(ids,pop))
            score=cost.reshape(count,pop);order=np.argsort(score,axis=1)
            chosen=np.take_along_axis(raw,order[:,:mu,None],axis=1)
            old=mean.copy();mean=np.sum(weights[None,:,None]*chosen,axis=1)
            y=(mean-old)/sigma[:,None]
            invsqrt=(basis/np.sqrt(eigen)[:,None,:])@np.swapaxes(basis,1,2)
            ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*np.einsum('nij,nj->ni',invsqrt,y)
            hs=np.linalg.norm(ps,axis=1)/np.sqrt(1-(1-cs)**(2*(generation+1)))<(1.4+2/(n+1))*chi
            pc=(1-cc)*pc+hs[:,None]*np.sqrt(cc*(2-cc)*mueff)*y
            ys=(chosen-old[:,None,:])/sigma[:,None,None]
            rankmu=np.einsum('p,npi,npj->nij',weights,ys,ys)
            cov=((1-c1-cmu)*cov+c1*(pc[:,:,None]*pc[:,None,:]+(~hs)[:,None,None]*cc*(2-cc)*cov)+cmu*rankmu)
            cov=.5*(cov+np.swapaxes(cov,1,2))
            sigma*=np.exp(np.clip(cs/damping*(np.linalg.norm(ps,axis=1)/chi-1),-1,1))
            local=score[np.arange(count),order[:,0]];improved=local<best_cost
            best_cost[improved]=local[improved];best_raw[improved]=raw[np.arange(count),order[:,0]][improved]
            candidate=self._best_record(np.repeat(ids,pop),cmd,vel,poses,cost)
            if best is None or candidate['cost']<best['cost']:best=candidate
        self.warm[ids]=best_raw
        best.update(alpha=None,residual=None,evaluations=count*pop*iterations,
            solver_backend='numpy-batched-full-covariance-cma-es',gpu_parallel_prediction=False,
            search_iterations=iterations)
        return best
