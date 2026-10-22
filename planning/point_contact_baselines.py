"""Independent QP/LCP predictors for the ideal point-force comparison.

Only benchmark instances replace their per-candidate inner solve. The reference
Isaac SCM factory, contact selection policy and physical executor are retained.
QP is MuJoCo's native pyramidal soft-contact QP, not an alias for the LCP.
LCP is the repository's regularized, coupled nonnegative friction-ray LCP.
Both physical predictors use the optimized force over the full policy step.
"""
import copy
from pathlib import Path
import sys
from types import MethodType

import mujoco
import casadi as cs
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from planning.direct_contact_force import (
    environment_contacts_body, gravity_wrench_body, object_frame,
    object_mass_body, object_velocity_body, continuous_force_duration,
)
from planning.object_pose_cost import object_cost_np, object_cost_cs
from planning.scm_contact_models import contact_jacobian
# Existing standalone SCM runs cache Numba dependencies under "vendor".
# Make that original import path resolvable when reusing the same vendored
# solver through its fully qualified package path; leave the vendor unchanged.
_vendor_parent = str(Path(__file__).resolve().parents[1]/'examples/mpc/franka/scm')
if _vendor_parent not in sys.path:
    sys.path.append(_vendor_parent)
def _load_lemke():
    # Native MuJoCo QP does not require the independent Numba/LCP backend.
    from examples.mpc.franka.scm.vendor.quantecon_lcp import lcp_lemke
    from examples.mpc.franka.scm.vendor.quantecon_lcp.linprog_simplex import PivOptions
    return lcp_lemke, PivOptions


def solve_physical_lcp(free, jac, inverse, gaps, h, regularization=1e-8):
    """Scaled Lemke solve, certified against the original unscaled system."""
    if not len(jac):
        return np.asarray(free).copy(), 0.
    lcp_lemke, PivOptions = _load_lemke()
    matrix = jac @ inverse @ jac.T + regularization*np.eye(len(jac))
    matrix = (matrix+matrix.T)/2
    offset = jac @ free + gaps/h
    scale = np.sqrt(np.diag(matrix))
    result = lcp_lemke(np.ascontiguousarray(matrix/scale[:,None]/scale[None,:]),
                       np.ascontiguousarray(offset/scale),max_iter=2000,
                       piv_options=PivOptions(tol_piv=1e-12,tol_ratio_diff=1e-15))
    if not result.success:
        raise FloatingPointError('Lemke LCP failed, status=%d'%result.status)
    dual = result.z/scale
    slack = matrix@dual+offset
    residual = max(float(np.maximum(-dual,0).max(initial=0)),
                   float(np.maximum(-slack,0).max(initial=0)),
                   float(np.abs(dual*slack).max(initial=0)))
    if residual > 1e-6:
        raise FloatingPointError('LCP failed unscaled certificate: %.3e'%residual)
    return free + inverse @ jac.T @ dual, residual


class PointContactBaseline:
    def __init__(self, env, mode, maxiter=12):
        if mode not in ('qp', 'lcp'):
            raise ValueError('baseline must be qp or lcp')
        self.env, self.mode = env, mode
        self.opt = env.param_.lambda_optimizer
        self.maxiter = int(maxiter)
        if self.maxiter < 1:
            raise ValueError('baseline maxiter must be positive')
        self.model_name = ('mujoco-native-pyramidal-qp' if mode == 'qp'
                           else 'coupled-physical-friction-ray-lcp')
        self.evaluations = self.inner_solves = self.inner_nonconverged = 0
        if mode == 'lcp':
            # Compile the vendored Numba Lemke implementation during setup,
            # consistently excluding code compilation from execution time.
            lcp_lemke, PivOptions = _load_lemke()
            lcp_lemke(np.eye(4),-np.ones(4),piv_options=PivOptions(tol_piv=1e-12,tol_ratio_diff=1e-15))
        # Optimize a first-order local pose response, then score the feasible
        # command with the actual contact model. Native QP/LCP active-set
        # changes are nonsmooth; no SCM derivative is substituted here.
        f, x0, tangent = cs.SX.sym('f',3), cs.SX.sym('x0',7), cs.SX.sym('T',7,3)
        xd, config = cs.SX.sym('target',7), cs.SX.sym('config',12)
        z = x0 + tangent@f
        cost = (object_cost_cs(z, xd, config) +
                self.opt.friction_reg_coef*cs.sumsqr(f[1:]) +
                self.opt.force_reg_coef*cs.sumsqr(f))
        self.local_objective = cs.Function('baseline_local_cost', [f,x0,tangent,xd,config],
                                            [cost,cs.gradient(cost,f)])
        if mode == 'qp':
            self.model = copy.copy(env.model_)
            self.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
            self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
            self.model.opt.timestep = self.opt.h
            self.data = mujoco.MjData(self.model)
            self.signature = mujoco.mjtState.mjSTATE_INTEGRATION
            self.state = np.zeros(mujoco.mj_stateSize(env.model_, self.signature))

    def prepare(self):
        """Snapshot measured plant state without changing the execution data."""
        env = self.env
        self.pose = env.data_.qpos[:7].copy()
        self.rotation = object_frame(env)
        self.duration = continuous_force_duration(env)
        self.velocity = object_velocity_body(env)
        self.inverse = np.linalg.inv(object_mass_body(env))
        self.gravity = gravity_wrench_body(env)
        jac, gaps = environment_contacts_body(env, env.param_.lambda_source_param_.mu_table_object_)
        self.jac, self.gaps = jac, gaps
        if self.mode == 'qp':
            mujoco.mj_getState(env.model_, env.data_, self.state, self.signature)
            mujoco.mj_setState(self.model, self.data, self.state, self.signature)
            self.data.qfrc_applied[:] = 0.; self.data.xfrc_applied[:] = 0.
            mujoco.mj_forward(self.model, self.data)
            self.warmstart = self.data.qacc_warmstart.copy()
        self.evaluations = self.inner_solves = self.inner_nonconverged = 0

    def predict(self, command, point, frame):
        self.evaluations += 1
        command, point = np.asarray(command), np.asarray(point)
        force = np.asarray(frame) @ command
        if self.mode == 'qp':
            m, d = self.model, self.data
            d.qfrc_applied[:] = 0.; d.xfrc_applied[:] = 0.
            d.qacc_warmstart[:] = self.warmstart
            body = m.body('obj').id
            mujoco.mj_applyFT(m, d, self.rotation @ force,
                             np.zeros(3), d.xpos[body]+self.rotation @ point,
                             body, d.qfrc_applied)
            tip = getattr(self.env, 'fingertip_body_id', None)
            if tip is not None:
                d.xfrc_applied[tip, :3] = -m.body_mass[tip]*m.opt.gravity
            # Every candidate starts at the SAME measured pose and velocity.
            # Reuse collision/velocity terms; rerun native force/constraint
            # dynamics, including the QP, for every different applied force.
            mujoco.mj_forwardSkip(m, d, mujoco.mjtStage.mjSTAGE_VEL, 1)
            next_velocity = d.qvel + self.opt.h*d.qacc
            qpos = d.qpos.copy()
            mujoco.mj_integratePos(m, qpos, next_velocity, self.opt.h)
            q = qpos[:7]
            rel = Rotation.from_matrix(self.rotation.T) * Rotation.from_quat(q[[4,5,6,3]])
            xyzw = rel.as_quat()
            x = np.r_[self.rotation.T @ (q[:3]-self.pose[:3]), xyzw[[3,0,1,2]]]
            v = np.r_[self.rotation.T @ next_velocity[:3], next_velocity[3:6]]
            residual = None
        else:
            free = self.velocity + self.inverse @ (
                self.opt.h*self.gravity + self.duration*(contact_jacobian(point).T @ force))
            v,residual = solve_physical_lcp(free,self.jac,self.inverse,self.gaps,self.opt.h)
            x = np.asarray(self.opt.cs_qposInteg_(np.r_[0.,0.,0.,1.,0.,0.,0.], v)).reshape(7)
        if not np.isfinite(x).all():
            raise FloatingPointError('Nonfinite baseline prediction')
        return dict(x_plus=x, v_plus=v, model=self.model_name, residual=residual)

    def solve_candidate(self, **kwargs):
        """Constrained local force solve; enumerate the same sampled contacts."""
        self.inner_solves += 1
        opt = self.opt
        frame = np.column_stack((kwargs['n_arm'], kwargs['t1'], kwargs['t2']))
        point, target = kwargs['p_arm'], kwargs['x_d']
        config = opt.object_cost_config()
        cap = opt.max_normal_force
        mu = opt.mu_arm_obj
        initial = np.array([.9*cap, 0., 0.])
        base = self.predict(initial, point, frame)['x_plus']
        tangent = np.zeros((7,3))
        eps = 1e-4
        for k in range(3):
            delta = np.zeros(3); delta[k] = eps
            p = self.predict(initial+delta, point, frame)['x_plus']
            if np.dot(p[3:],base[3:]) < 0: p[3:] *= -1
            tangent[:,k] = (p-base)/eps
        intercept = base-tangent@initial

        def local(f):
            cost, grad = self.local_objective(f, intercept, tangent, target, config)
            return float(cost), np.asarray(grad).reshape(3)

        def cone(f):
            return np.array([mu*f[0]-s1*f[1]-s2*f[2]
                             for s1,s2 in ((1,1),(1,-1),(-1,1),(-1,-1))])

        result = minimize(local, initial, jac=True, method='SLSQP',
                          bounds=((0., cap), (-mu*cap, mu*cap), (-mu*cap, mu*cap)),
                          constraints=({'type':'ineq', 'fun':cone},),
                          options={'maxiter':self.maxiter, 'ftol':1e-7, 'eps':1e-4})
        if not result.success:
            self.inner_nonconverged += 1
        # A stopped local solve may yield a feasible incumbent. Report it
        # explicitly, preserving its own model rather than falling back to SCM.
        f = opt._project_contact_force(result.x, opt.max_contact_force)
        pred = self.predict(f, point, frame)
        cost = (object_cost_np(pred['x_plus'], target, config) +
                opt.friction_reg_coef*np.dot(f[1:],f[1:]) + opt.force_reg_coef*np.dot(f,f))
        opt.last_solver_status = 'slsqp' if result.success else 'slsqp-iteration-or-linesearch-stop'
        return dict(lam_arm_opt=f, x_plus_opt=pred['x_plus'], cost=cost)


def install_point_contact_baseline(env, mode, maxiter=12):
    predictor = PointContactBaseline(env, mode, maxiter=maxiter)
    opt = env.param_.lambda_optimizer
    # Benchmark-only override: the standard selection still enumerates every
    # visible contact, ranks its own model predictions, and applies hysteresis.
    def solve(instance, **kwargs):
        return predictor.solve_candidate(**kwargs)
    opt._solve_optimization = MethodType(solve, opt)
    env.param_.force_prediction_model_ = predictor
    return predictor
