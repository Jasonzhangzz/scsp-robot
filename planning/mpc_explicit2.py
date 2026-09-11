import casadi as cs
import numpy as np
import time
try:
    import torch
except ImportError:  # IPOPT must remain usable without the optional Torch stack
    torch = None

from models.explicit_model import ExplicitModel
# cs.PK_VERBOSE = 0  # 全局关闭 CasADi 内部调试信息

class MPCExplicit:
    def __init__(self, param):
        self.param_ = param

        # cost function
        self.path_cost_fn, self.final_cost_fn = self.param_.init_cost_fns()

        # parse mpc model
        self.mpc_model = self.param_.mpc_model

        if self.mpc_model == 'explicit':
            self.model = ExplicitModel(param)
            self.init_MPC()
        else:
            raise ValueError('Invalid model type')

    def plan_once(self, target_p, target_q, curr_x, phi_vec, jac_mat, verify_cost_param, virtual_point, contact_point, curr_ori_coef=None, sol_guess=None):
        if sol_guess is None:
            sol_guess = dict(x0=self.nlp_w0_, lam_x0=self.nlp_lam_x0_, lam_g0=self.nlp_lam_g0_)

        # ``init_cost_fns`` defines the exact cost-parameter vector.  The
        # rollout also passes ``curr_ori_coef`` for compatibility with other
        # MPC variants, but this parameter is not used by the fingertip cost
        # function.  CasADi requires an exact shape, so normalize here rather
        # than allowing an occasional one-element mismatch to reach the
        # generated function (e.g. 415 vs 414 elements).
        # The reference fingertip cost has no orientation-coefficient
        # parameter; ``curr_ori_coef`` is accepted for compatibility with
        # newer callers but intentionally ignored.
        cost_params = cs.vvcat([target_p, target_q, phi_vec, jac_mat,
                                verify_cost_param, virtual_point,
                                contact_point])
        expected_rows, expected_cols = self.path_cost_fn.size_in(2)
        expected_dim = int(expected_rows * expected_cols)
        actual_dim = int(np.prod(cost_params.shape))
        if actual_dim > expected_dim:
            cost_params = cost_params[:expected_dim]
        elif actual_dim < expected_dim:
            cost_params = cs.vertcat(cost_params,
                                     cs.DM.zeros(expected_dim - actual_dim, 1))

        nlp_param = self.nlp_params_fn_(curr_x, phi_vec, jac_mat, cost_params, self.param_.model_params)

        nlp_lbw, nlp_ubw = self.nlp_bounds_fn_(self.param_.mpc_u_lb_, self.param_.mpc_u_ub_,
                                               self.param_.mpc_q_lb_,
                                               self.param_.mpc_q_ub_)

        if getattr(self.param_, 'torch_solver', 'ipopt') != 'ipopt':
            try:
                return self._plan_torch(curr_x, phi_vec, jac_mat, cost_params, sol_guess)
            except Exception as exc:
                # IPOPT remains an explicit safety net for numerical failures
                # (e.g. an ill-conditioned contact Jacobian on a rollout step).
                print(f'Torch MPC failed ({type(exc).__name__}: {exc}); falling back to IPOPT')
        st = time.time()
        raw_sol = self.ipopt_solver(x0=sol_guess['x0'],
                                    lam_x0=sol_guess['lam_x0'],
                                    lam_g0=sol_guess['lam_g0'],
                                    lbx=nlp_lbw, ubx=nlp_ubw,
                                    lbg=0.0, ubg=0.0,
                                    p=nlp_param)
        # print("mpc solve time:", time.time() - st)
        # print('mpc solve status = ', self.ipopt_solver.stats()['return_status'])

        w_opt = raw_sol['x'].full().flatten()
        cost_opt = raw_sol['f'].full().flatten()

        # extract the solution from the raw solution
        sol_traj = np.reshape(w_opt, (self.param_.mpc_horizon_, -1))
        opt_u_traj = sol_traj[:, 0:self.param_.n_cmd_]   # [u, q]

        return dict(action=opt_u_traj[0, :],
                    sol_guess=dict(x0=w_opt,
                                   lam_x0=raw_sol['lam_x'],
                                   lam_g0=raw_sol['lam_g'],
                                   opt_cost=raw_sol['f'].full().item()),
                    cost_opt=cost_opt,
                                   solve_status=self.ipopt_solver.stats()['return_status'])

    def _plan_torch(self, curr_x, phi_vec, jac_mat, cost_params, sol_guess):
        """GPU Torch replacement for the CasADi/IPOPT NLP (same objective)."""
        if torch is None:
            raise RuntimeError('Torch is not installed')
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float32
        p = torch.as_tensor(np.asarray(cost_params).reshape(-1), device=dev, dtype=dtype)
        phi = torch.as_tensor(np.asarray(phi_vec).reshape(-1), device=dev, dtype=dtype)
        jac = torch.as_tensor(np.asarray(jac_mat), device=dev, dtype=dtype)
        q0 = torch.as_tensor(np.asarray(curr_x).reshape(-1), device=dev, dtype=dtype)
        n, nu, nq = self.param_.mpc_horizon_, self.param_.n_cmd_, self.param_.n_qpos_
        qlb = torch.as_tensor(self.param_.mpc_q_lb_, device=dev, dtype=dtype)
        qub = torch.as_tensor(self.param_.mpc_q_ub_, device=dev, dtype=dtype)
        ulb = torch.as_tensor(self.param_.mpc_u_lb_, device=dev, dtype=dtype)
        uub = torch.as_tensor(self.param_.mpc_u_ub_, device=dev, dtype=dtype)
        # Warm start uses the previous flattened [u,q] trajectory when available.
        z0 = np.asarray(sol_guess.get('x0', self.nlp_w0_)).reshape(-1)
        if z0.size != n * (nu + nq):
            raise ValueError(f'invalid Torch MPC warm start size {z0.size}, expected {n * (nu + nq)}')
        # ``nlp_w0_`` initializes all states to zero, including an invalid
        # zero quaternion.  Seed the first Torch solve with the measured state
        # so LBFGS does not create a large artificial correction impulse.
        if not np.isfinite(z0).all() or np.linalg.norm(z0) < 1e-12:
            for k in range(n):
                z0[k * (nu + nq) + nu:(k + 1) * (nu + nq)] = np.asarray(curr_x).reshape(-1)
        z = torch.nn.Parameter(torch.as_tensor(z0, device=dev, dtype=dtype))
        opt = torch.optim.LBFGS([z], max_iter=int(self.param_.ipopt_max_iter_), tolerance_grad=1e-5,
                                tolerance_change=1e-9, line_search_fn='strong_wolfe')
        Qinv = torch.as_tensor(np.linalg.inv(self.param_.Q), device=dev, dtype=dtype)
        h = float(self.param_.h_); sigma = float(self.param_.model_params)
        target, quat = p[:3], p[3:7]; verify = p[-7]; virt, cp = p[-6:-3], p[-3:]
        def closure():
            opt.zero_grad(); q = q0; J = torch.zeros((), device=dev)
            off = 0
            for k in range(n):
                u = torch.clamp(z[off:off+nu], ulb, uub); off += nu
                qnext_var = z[off:off+nq]; off += nq
                b = torch.cat((torch.as_tensor(self.param_.obj_mass_ * self.param_.gravity_, device=dev, dtype=dtype),
                               torch.as_tensor(self.param_.robot_stiff_, device=dev, dtype=dtype) @ u))
                cf = torch.clamp(-sigma * (jac @ Qinv @ b + phi), min=0)
                v = (Qinv @ b + Qinv @ jac.T @ cf) / h
                qpred = torch.cat((q[:3] + h*v[:3], q[3:7] + 0.5*h*torch.stack((-q[4]*v[3]-q[5]*v[4]-q[6]*v[5], q[3]*v[3]+q[6]*v[4]-q[5]*v[5], q[3]*v[4]-q[6]*v[3]+q[4]*v[5], q[3]*v[5]+q[5]*v[3]-q[4]*v[4])), q[-self.param_.n_robot_qpos_:] + h*v[-self.param_.n_robot_qpos_:]))
                q = torch.clamp(qnext_var, qlb, qub); J = J + 1e4*torch.sum((q-qpred)**2) + (1-verify)*(self.param_.attract_coef*torch.log(torch.sum((q[7:10]-virt)**2)+1e-3)+self.param_.reject_coef*torch.where(torch.sum((q[:2]-q[7:9])**2)+1e-3 < self.param_.reject_dis, 1/(torch.sum((q[:2]-q[7:9])**2)+1e-3), torch.zeros((),device=dev))) + self.param_.contact_coef*verify*(self.param_.contact_cost_param*torch.sum((q[:3]-q[7:10])**2)+(1-self.param_.contact_cost_param)*torch.log(torch.sum((q[7:10]-cp)**2)+1e-3)) + 50*torch.sum(u*u)
            J = J + 10*(500*torch.sum((q[:3]-target)**2)+5*(1-torch.dot(q[3:7],quat)**2)); J.backward(); return J
        opt.step(closure)
        out = z.detach().cpu().numpy()
        traj = out.reshape(n, -1)
        # ``z`` is an unconstrained optimization variable; the objective uses
        # clamped controls.  Returning raw z was the source of very large
        # fingertip jumps (the environment interprets action as a position
        # increment).  Return and warm-start with the physically bounded
        # controls/states instead.
        traj[:, :nu] = np.clip(traj[:, :nu],
                               np.asarray(self.param_.mpc_u_lb_),
                               np.asarray(self.param_.mpc_u_ub_))
        traj[:, nu:] = np.clip(traj[:, nu:],
                               np.asarray(self.param_.mpc_q_lb_),
                               np.asarray(self.param_.mpc_q_ub_))
        if not np.isfinite(traj).all():
            raise FloatingPointError('non-finite Torch MPC trajectory')
        out = traj.reshape(-1)
        final_cost = float(closure().detach().cpu())
        return dict(action=traj[0, :nu].copy(),
                    sol_guess=dict(x0=out,
                                   lam_x0=np.zeros_like(out),
                                   lam_g0=np.zeros(n*self.param_.n_qpos_),
                                   opt_cost=final_cost),
                    cost_opt=np.array([final_cost]), solve_status='torch-lbfgs')

    def init_MPC(self):
        model_params = cs.SX.sym('model_param', 1)

        phi_vec = cs.SX.sym('phi_vec', self.param_.max_ncon_ * 4)
        jac_mat = cs.SX.sym('jac_mat', self.param_.max_ncon_ * 4, self.param_.n_qvel_)

        cost_params = cs.SX.sym('cost_params', self.path_cost_fn.size_in(2))

        lbu = cs.SX.sym('lbu', self.param_.n_cmd_)
        ubu = cs.SX.sym('ubu', self.param_.n_cmd_)

        lbq = cs.SX.sym('lbq', self.param_.n_qpos_)
        ubq = cs.SX.sym('ubq', self.param_.n_qpos_)

        # start with empty NLP
        w, w0, lbw, ubw, g = [], [], [], [], []
        J = 0.0
        q0 = cs.SX.sym('q', self.param_.n_qpos_)
        qk = q0
        for k in range(self.param_.mpc_horizon_):
            # control at time k
            uk = cs.SX.sym('u' + str(k), self.param_.n_cmd_)
            w += [uk]
            lbw += [lbu]
            ubw += [ubu]
            w0 += [cs.DM.zeros(self.param_.n_cmd_)]

            # lse dyn function
            pred_q = self.model.step_once_fn(qk, uk, phi_vec, jac_mat, model_params)

            # compute the cost function
            J += self.path_cost_fn(qk, uk, cost_params)

            # q at time k+1 .... q_new
            qk = cs.SX.sym('q' + str(k + 1), self.param_.n_qpos_)
            w += [qk]
            w0 += [cs.DM.zeros(self.param_.n_qpos_)]
            lbw += [lbq]
            ubw += [ubq]

            # add the concatenation constraint 等式约束，动力学约束，g=0，q_k+1 = f(q_k, u_k)
            # qk = pred_q
            g += [pred_q - qk]  

        # compute the final cost
        J += self.final_cost_fn(qk, cost_params)

        # create an NLP solver
        nlp_params = cs.vvcat([q0, phi_vec, jac_mat, cost_params, model_params])
        nlp_prog = {'f': J, 'x': cs.vcat(w), 'g': cs.vcat(g), 'p': nlp_params}
        nlp_opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0,
                    'ipopt.max_iter': self.param_.ipopt_max_iter_,     
                    "ipopt.tol": 1e-4,
                    "ipopt.linear_solver": "mumps",}
        self.ipopt_solver = cs.nlpsol('solver', 'ipopt', nlp_prog, nlp_opts)

        # useful mappings
        self.nlp_w0_ = cs.vcat(w0)
        self.nlp_lam_x0_ = cs.DM.zeros(self.nlp_w0_.shape)
        self.nlp_lam_g0_ = cs.DM.zeros(cs.vcat(g).shape)
        self.nlp_bounds_fn_ = cs.Function('nlp_bounds_fn', [lbu, ubu, lbq, ubq], [cs.vcat(lbw), cs.vvcat(ubw)])
        self.nlp_params_fn_ = cs.Function('nlp_params_fn',
                                          [q0, phi_vec, jac_mat, cost_params, model_params], [nlp_params])
