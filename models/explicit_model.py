"""Reduced rigid-body contact model with CasADi derivatives.

step_state_fn carries generalized velocity. The five-input step_once_fn keeps
its historical position-only interface and starts each step at zero velocity.
In full-dynamic mode Q (or comfree_mass_matrix) supplies generalized mass.
In quasi-dynamic mode Q is the legacy impedance metric and cmd is a
displacement: v_free = Q^-1 [external; K_robot cmd] / h. These units must
not be replaced by a force integration when Q_robot = K_robot.
Collision supplies active phi/J rows; padding rows must have a zero Jacobian.
soft-SCM is the default; param.contact_model='comfree' restores ComFree.
The historical sigma input controls ComFree stiffness only. soft-SCM uses
param.soft_scm_response_time, soft_scm_damping_ratio and regularization.
With online_scm_=True, prediction_fn accepts [sigma; theta(6)] as its last
input. The legacy step_once_fn/step_state_fn remain fixed-prior interfaces.
"""
import numpy as np
import casadi as cs
from models.soft_scm_contact import (
    SOFT_SCM_RESPONSE_TIME, SOFT_SCM_DAMPING_RATIO, SOFT_SCM_REGULARIZATION,
    normalize_contact_model, soft_scm_coefficients,
    inverse_diagonal_cs, inverse_diagonal_torch,
)

from models.scm_parameters import (scaled_inverse_cs, scaled_inverse_numpy,
    friction_jacobian_cs, response_cs, validate_theta)

np.set_printoptions(suppress=True)


class ExplicitModel:
    def __init__(self, param):
        self.param_ = param
        self.contact_model = normalize_contact_model(getattr(param, 'contact_model', 'soft-scm'))
        self.soft_scm_response_time = float(getattr(param, 'soft_scm_response_time', SOFT_SCM_RESPONSE_TIME))
        self.soft_scm_damping_ratio = float(getattr(param, 'soft_scm_damping_ratio', SOFT_SCM_DAMPING_RATIO))
        self.soft_scm_regularization = float(getattr(param, 'soft_scm_regularization', SOFT_SCM_REGULARIZATION))
        self.soft_scm_a, self.soft_scm_b = soft_scm_coefficients(
            param.h_, self.soft_scm_response_time, self.soft_scm_damping_ratio,
            self.soft_scm_regularization)
        self.contact_dynamics = str(getattr(param, 'contact_dynamics', 'full-dynamic'))
        if self.contact_dynamics not in ('full-dynamic', 'quasi-dynamic'):
            raise ValueError("contact_dynamics must be 'full-dynamic' or 'quasi-dynamic'")
        self.force_velocity_scale = param.h_ if self.contact_dynamics == 'full-dynamic' else 1.0 / param.h_

        self.scm_theta = np.zeros(6)
        self.online_scm = bool(getattr(param, "online_scm_", False)) and self.contact_model == "soft-scm"
        self.model_parameter_dim = 7 if self.online_scm else 1
        self.grouped_contact_response = bool(getattr(param, 'grouped_contact_response_', False))
        self.init_utils()
        self.init_model()

    def init_utils(self):
        # -------------------------------
        #    quaternion integration fn
        # -------------------------------
        quat = cs.SX.sym('quat', 4)
        H_q_body = cs.vertcat(cs.horzcat(-quat[1], quat[0], quat[3], -quat[2]),
                              cs.horzcat(-quat[2], -quat[3], quat[0], quat[1]),
                              cs.horzcat(-quat[3], quat[2], -quat[1], quat[0]))
        self.cs_qmat_body_fn_ = cs.Function('cs_qmat_body_fn', [quat], [H_q_body.T])

        # -------------------------------
        #    state integration fn
        # -------------------------------
        qvel = cs.SX.sym('qvel', self.param_.n_qvel_)
        qpos = cs.SX.sym('qpos', self.param_.n_qpos_)
        next_obj_pos = qpos[0:3] + self.param_.h_ * qvel[0:3]
        next_robot_qpos = qpos[-(self.param_.n_robot_qpos_):] + self.param_.h_ * qvel[-(self.param_.n_robot_qpos_):]
        next_obj_quat = (qpos[3:7] + 0.5 * self.param_.h_ * self.cs_qmat_body_fn_(qpos[3:7]) @ qvel[3:6])
        next_obj_quat = next_obj_quat / cs.norm_2(next_obj_quat)
        next_qpos = cs.vertcat(next_obj_pos, next_obj_quat, next_robot_qpos)
        self.cs_qposInteg_ = cs.Function('cs_qposInte', [qpos, qvel], [next_qpos])

    def init_model(self):
        curr_q = cs.SX.sym('qvel', self.param_.n_qpos_)
        curr_v = cs.SX.sym('curr_v', self.param_.n_qvel_)
        cmd = cs.SX.sym('cmd', self.param_.n_cmd_)
        phi_vec = cs.SX.sym('phi_vec', self.param_.max_ncon_ * 4)
        jac_mat = cs.SX.sym('jac_mat', self.param_.max_ncon_ * 4, self.param_.n_qvel_)

        # Full dynamics needs physical mass; quasi dynamics retains the
        # existing impedance metric and displacement-control convention.
        mass = np.asarray(getattr(self.param_, "comfree_mass_matrix", self.param_.Q))
        if (mass.shape != (self.param_.n_qvel_, self.param_.n_qvel_) or
                not np.isfinite(mass).all() or not np.allclose(mass, mass.T) or
                np.linalg.eigvalsh(mass).min() <= 0):
            raise ValueError("Generalized mass must be finite, symmetric and positive definite")
        inverse_np = np.linalg.inv(mass)
        self.base_inverse = inverse_np.copy()
        inverse = cs.DM(inverse_np)
        force = cs.vertcat(cs.DM(self.param_.obj_mass_ * self.param_.gravity_),
                           cs.DM(self.param_.robot_stiff_) @ cmd)
        h = self.param_.h_
        model_params = cs.SX.sym('sigma', 1)  # Legacy interface: ComFree stiffness only
        damping = float(getattr(self.param_, "comfree_damping", 0.001))
        if not np.isfinite(h) or h <= 0 or not np.isfinite(damping) or damping < 0:
            raise ValueError("ComFree timestep must be positive and damping nonnegative")
        v_smooth = curr_v + self.force_velocity_scale * inverse @ force
        speed = jac_mat @ v_smooth
        if self.contact_model == 'soft-scm':
            # Includes object rotation and robot mobility in every active row.
            weights = inverse_diagonal_cs(jac_mat, inverse, self.soft_scm_regularization)
            residual = self.soft_scm_a * speed + self.soft_scm_b * phi_vec / h
            impulse = cs.fmax(-weights * residual, 0)
            v = v_smooth + inverse @ jac_mat.T @ impulse
            if self.grouped_contact_response:
                from models.soft_scm_contact import grouped_contact_velocity_cs
                v = grouped_contact_velocity_cs(v_smooth, jac_mat, phi_vec, inverse, h, self.soft_scm_a, self.soft_scm_b, self.soft_scm_regularization)
        else:
            invweight = float(np.trace(inverse_np[:3, :3]) / 3.0)
            row_masses = []
            for row in range(self.param_.max_ncon_ * 4):
                x = cs.fmin(cs.fabs(phi_vec[row]) / 0.01, 1)
                curve = cs.if_else(x < 0.5, 2*x*x, 1-2*(1-x)**2)
                imp = 0.9 + 0.05*curve
                row_invweight = invweight
                # Each Cartesian end effector is a second moving body.
                for start in range(6, self.param_.n_qvel_ - 2, 3):
                    robot_weight = float(np.trace(inverse_np[start:start+3, start:start+3])/3)
                    row_invweight += cs.if_else(cs.sumsqr(jac_mat[row, start:start+3]) > 0, robot_weight, 0)
                weight = imp / (row_invweight * (1-imp))
                row_masses.append(cs.if_else(cs.sumsqr(jac_mat[row, :]) > 0, weight, 0))
            acceleration = -(damping/h)*speed - (model_params/h)*(h*speed + phi_vec)
            contact_force = cs.fmax(cs.vertcat(*row_masses) * acceleration, 0)
            v = v_smooth + h * inverse @ jac_mat.T @ contact_force

        # time integration
        next_qpos = self.cs_qposInteg_(curr_q, v)

        # assemble the casadi function
        self.step_state_fn = cs.Function(self.contact_model.replace('-', '_') + '_step_state',
            [curr_q, curr_v, cmd, phi_vec, jac_mat, model_params], [next_qpos, v])
        zero_velocity = cs.substitute(next_qpos, curr_v, cs.SX.zeros(self.param_.n_qvel_))
        self.step_once_fn = cs.Function('step_once', [curr_q, cmd, phi_vec, jac_mat, model_params], [zero_velocity])
        if self.contact_model == 'soft-scm':
            theta = cs.SX.sym('scm_theta', 6)
            qi = scaled_inverse_cs(inverse_np, theta)
            j = friction_jacobian_cs(jac_mat, theta)
            a, b = response_cs(h, self.soft_scm_response_time, self.soft_scm_damping_ratio, theta)
            free = self.force_velocity_scale * qi @ force
            weights = inverse_diagonal_cs(j, qi, self.soft_scm_regularization)
            vel = free + qi @ j.T @ cs.fmax(-weights*(a*(j@free)+b*phi_vec/h), 0)
            if self.grouped_contact_response:
                from models.soft_scm_contact import grouped_contact_velocity_cs
                vel = grouped_contact_velocity_cs(free, j, phi_vec, qi, h, a, b, self.soft_scm_regularization)
            pred = self.cs_qposInteg_(curr_q, vel)
            self.step_learn_fn = cs.Function('scm_step_learn', [curr_q, cmd, phi_vec, jac_mat, theta], [pred])
            mp = cs.SX.sym('scm_model_parameters', 7)
            self.step_adaptive_fn = cs.Function('scm_step_adaptive',
                [curr_q, cmd, phi_vec, jac_mat, mp],
                [self.step_learn_fn(curr_q, cmd, phi_vec, jac_mat, mp[1:])])
        self.prediction_fn = self.step_adaptive_fn if self.online_scm else self.step_once_fn

    def runtime_parameters(self, sigma):
        return np.r_[float(sigma), self.scm_theta] if self.online_scm else float(sigma)

    def set_scm_parameters(self, theta):
        if self.contact_model != 'soft-scm':
            raise ValueError('Online calibration requires soft-SCM')
        self.scm_theta = validate_theta(theta)

    def current_inverse(self):
        return scaled_inverse_numpy(self.base_inverse, self.scm_theta)


    def step(self, curr_q, cmd, phi_vec, jac_mat, sigma, qvel=None):
        if qvel is None:
            return self.prediction_fn(curr_q, cmd, phi_vec, jac_mat, self.runtime_parameters(sigma))
        return self.step_state_fn(curr_q, qvel, cmd, phi_vec, jac_mat, sigma)[0]

    def contact_velocity_torch(self, smooth, jacobian, phi, inverse, sigma):
        """Same row law as step_state_fn, used by the Torch MPC backend."""
        import torch
        h = float(self.param_.h_)
        if self.online_scm:
            blocks = jacobian.reshape(-1, 4, jacobian.shape[1])
            normal = blocks.mean(dim=1, keepdim=True)
            jacobian = (normal + float(np.exp(self.scm_theta[5]))*(blocks-normal)).reshape(jacobian.shape)
        speed = jacobian @ smooth
        if self.contact_model == 'soft-scm':
            weights = inverse_diagonal_torch(jacobian, inverse, self.soft_scm_regularization)
            a, b = soft_scm_coefficients(h, self.soft_scm_response_time*np.exp(self.scm_theta[4]),
                                         self.soft_scm_damping_ratio, self.soft_scm_regularization)
            if self.grouped_contact_response:
                from models.soft_scm_contact import grouped_contact_velocity_torch
                return grouped_contact_velocity_torch(smooth, jacobian, phi, inverse, h, a, b, self.soft_scm_regularization)
            residual = a * speed + b * phi / h
            impulse = torch.clamp(-weights * residual, min=0)
            return smooth + inverse @ jacobian.T @ impulse
        x = (phi.abs()/0.01).clamp(max=1)
        curve = torch.where(x < .5, 2*x*x, 1-2*(1-x)**2)
        imp = .9 + .05*curve
        row_invweight = torch.ones_like(imp) * torch.trace(inverse[:3, :3])/3
        for start in range(6, self.param_.n_qvel_-2, 3):
            robot_weight = torch.trace(inverse[start:start+3, start:start+3])/3
            row_invweight = row_invweight + torch.where(
                torch.sum(jacobian[:, start:start+3]**2, dim=1)>0, robot_weight, 0)
        masses = imp/(row_invweight*(1-imp))
        masses = torch.where(torch.sum(jacobian**2, dim=1)>0, masses, 0)
        damping = float(getattr(self.param_, 'comfree_damping', .001))
        force = torch.clamp(masses*(-damping/h*speed-sigma/h*(h*speed+phi)), min=0)
        return smooth + h * (inverse @ jacobian.T @ force)
