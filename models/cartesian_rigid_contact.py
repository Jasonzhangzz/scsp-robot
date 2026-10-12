"""Velocity-state rigid object and finite-response Cartesian fingertip.

State = [object pose(7), fingertip position(3), world linear velocity(3),
body angular velocity(3), world fingertip velocity(3)]. Commands remain
Cartesian position increments, not forces or joint velocities.
"""
import numpy as np
import casadi as cs

from models.contact_geometry import rotation_cs
from models.soft_scm_contact import (
    soft_scm_coefficients, grouped_contact_velocity_cs,
)


class CartesianRigidContactModel:
    model_parameter_dim = 1
    contact_model = 'soft-scm'
    online_scm = False

    def __init__(self, param):
        self.param_ = param
        h = float(param.h_)
        if not np.isfinite(h) or h <= 0:
            raise ValueError('MPC timestep must be positive')
        inertia = np.asarray(param.sim_obj_inertia_diag_, dtype=float)
        # The fingertip's effective mass is an actuator response parameter,
        # separate from the object's physical mass and the old impedance Q.
        tip_mass = float(getattr(param, 'mpc_tip_effective_mass_', 2.5))
        mass = np.r_[np.full(3, param.sim_obj_mass_), inertia, np.full(3, tip_mass)]
        if not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError('physical object and effective fingertip masses must be positive')
        self.base_inverse = np.diag(1/mass)
        self.scm_theta = np.zeros(6)
        self.force_velocity_scale = h
        damping = float(getattr(param, 'mpc_tip_damping_', 40.))
        if not np.isfinite(damping) or damping <= 0:
            raise ValueError('Cartesian damping must be positive')
        self.response_time = tip_mass/damping
        self.decay = np.exp(-h/self.response_time)
        self.velocity_memory = self.response_time/h*(1-self.decay)
        # OSC in contact applies K*increment - D*tip_velocity throughout
        # the interval (see _compute_osc_torques), rather than a position
        # servo toward a linearly interpolated endpoint. Integrate that PD
        # wrench exactly; carry the resulting tip velocity between stages.
        self.command_velocity = np.asarray(param.robot_stiff_)/damping
        self.response = h*(1-self.velocity_memory)*np.mean(np.diag(self.command_velocity))
        self.soft_scm_regularization = float(param.soft_scm_regularization)
        self.soft_scm_response_time = float(getattr(param, 'mpc_rigid_contact_time_', .01))
        self.soft_scm_damping_ratio = float(param.soft_scm_damping_ratio)
        a, b = soft_scm_coefficients(h, self.soft_scm_response_time,
                                     self.soft_scm_damping_ratio, self.soft_scm_regularization)
        x, u = cs.SX.sym('rigid_state', 19), cs.SX.sym('increment', 3)
        phi, jac = cs.SX.sym('gap', 4*param.max_ncon_), cs.SX.sym('jac', 4*param.max_ncon_, 9)
        sigma = cs.SX.sym('sigma')
        self.response_options = dict(h=h, tip_mass=tip_mass, damping=damping,
            response_time=param.soft_scm_response_time,
            rigid_response_time=self.soft_scm_response_time,
            damping_ratio=self.soft_scm_damping_ratio,
            regularization=self.soft_scm_regularization,
            blend_speed=float(getattr(param, 'mpc_contact_blend_speed_', .005)),
            variant=getattr(param, 'contact_dynamics_variant_', 'legacy'))
        self.use_shared_response = bool(getattr(param, 'shared_contact_dynamics_', False)
                                        or self.response_options['variant'] != 'legacy')
        if not np.isfinite(self.response_options['blend_speed']) or self.response_options['blend_speed'] <= 0:
            raise ValueError('contact blend speed must be positive')
        if self.use_shared_response:
            from models.cartesian_contact_response import contact_response_cs
            velocity, next_tip_velocity = contact_response_cs(
                x[10:16], x[16:19], cs.DM(param.robot_stiff_) @ u,
                cs.DM(param.obj_mass_ * param.gravity_), jac, phi,
                cs.DM(self.base_inverse), cs.DM(np.linalg.inv(np.asarray(param.Q))),
                **self.response_options)
            self.contact_blend_speed = self.response_options['blend_speed']
        else:
            free_tip = self.velocity_memory*x[16:19]+(1-self.velocity_memory)*cs.DM(self.command_velocity)@u
            end_tip = self.decay*x[16:19]+(1-self.decay)*cs.DM(self.command_velocity)@u
            free = cs.vertcat(x[10:13]+h*cs.DM(param.gravity_[:3]), x[13:16], free_tip)
            velocity = grouped_contact_velocity_cs(free, jac, phi, cs.DM(self.base_inverse),
                h, a, b, self.soft_scm_regularization)
            # A made Cartesian contact is an impedance-controlled manipulation,
            # not an impulsive collision with a free robot mass. Keep its measured
            # steady contact mobility, while an unsupported object uses physical
            # inertia and carries its velocity into the falling trajectory.
            mobility = cs.DM(np.linalg.inv(np.asarray(param.Q)))
            ca, cb = soft_scm_coefficients(h, param.soft_scm_response_time,
                self.soft_scm_damping_ratio, self.soft_scm_regularization)
            supported_free = cs.vertcat(.5*x[10:16]+mobility[:6,:6]@cs.DM(param.obj_mass_*param.gravity_)/h,
                                       free_tip)
            supported = grouped_contact_velocity_cs(supported_free,jac,phi,mobility,h,ca,cb,self.soft_scm_regularization)
            loading = 0
            for start in range(0,4*param.max_ncon_,4):
                normal = cs.sum1(jac[start:start+4,:])/4
                gap = cs.sum1(phi[start:start+4])/4
                loading = cs.fmax(loading,cs.if_else(cs.sumsqr(normal[6:9]) > 0,
                    cs.fmax(-(normal@supported_free)-gap/h,0),0))
            blend_speed = float(getattr(param, 'mpc_contact_blend_speed_', .005))
            if not np.isfinite(blend_speed) or blend_speed <= 0:
                raise ValueError('contact blend speed must be positive')
            self.contact_blend_speed = blend_speed
            fraction = cs.fmin(1.,cs.fmax(0.,loading/blend_speed))
            # Smooth endpoints bound the RTI derivatives at zero pressure. A
            # reciprocal blend has arbitrarily large derivatives there and turns
            # a contact switch into an ill-conditioned QP / IPOPT fallback.
            support_fraction = fraction**3*(10-15*fraction+6*fraction**2)
            velocity += support_fraction*(supported-velocity)
        q, w = x[3:7], velocity[3:6]
        qnext = q+.5*h*cs.vertcat(-cs.dot(q[1:4], w), q[0]*w+cs.cross(q[1:4],w))
        qnext /= cs.sqrt(cs.sumsqr(qnext)+1e-20)
        # Re-express angular momentum's velocity coordinates in the next
        # body frame, so a changing orientation does not rotate world omega.
        next_w = rotation_cs(qnext).T@rotation_cs(q)@w
        xn = cs.vertcat(x[:3]+h*velocity[:3], qnext, x[7:10]+h*velocity[6:9],
                       velocity[:3], next_w, (next_tip_velocity if self.use_shared_response else end_tip+velocity[6:9]-free_tip))
        self.prediction_fn = cs.Function('cartesian_rigid_step', [x,u,phi,jac,sigma], [xn])

    def runtime_parameters(self, sigma):
        return float(sigma)

    def current_inverse(self):
        return self.base_inverse

    def observation_state(self, pose, linear=None, angular=None, fingertip=None):
        pose = np.asarray(pose, dtype=float).reshape(10)
        linear = np.zeros(3) if linear is None else np.asarray(linear, dtype=float).reshape(3)
        angular = np.zeros(3) if angular is None else np.asarray(angular, dtype=float).reshape(3)
        fingertip = np.zeros(3) if fingertip is None else np.asarray(fingertip, dtype=float).reshape(3)
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_quat(pose[[4,5,6,3]])
        return np.r_[pose, linear, rotation.inv().apply(angular), fingertip]

    def prediction_torch(self, x, command, phi, jac):
        import torch
        from models.contact_geometry import rotation_torch
        from models.soft_scm_contact import grouped_contact_velocity_torch
        h = float(self.param_.h_)
        inverse = torch.as_tensor(self.base_inverse,dtype=x.dtype,device=x.device)
        mobility = torch.as_tensor(np.linalg.inv(self.param_.Q),dtype=x.dtype,device=x.device)
        gravity = torch.as_tensor(self.param_.gravity_,dtype=x.dtype,device=x.device)
        if self.use_shared_response:
            from models.cartesian_contact_response import contact_response_torch
            velocity, next_tip_velocity = contact_response_torch(
                x[10:16], x[16:19], torch.as_tensor(self.param_.robot_stiff_, dtype=x.dtype, device=x.device) @ command,
                self.param_.obj_mass_ * gravity, jac, phi, inverse, mobility,
                **self.response_options)
        else:
            free_tip = self.velocity_memory*x[16:19]+(1-self.velocity_memory)*torch.as_tensor(self.command_velocity,dtype=x.dtype,device=x.device)@command
            end_tip = self.decay*x[16:19]+(1-self.decay)*torch.as_tensor(self.command_velocity,dtype=x.dtype,device=x.device)@command
            free = torch.cat([x[10:13]+h*gravity[:3],x[13:16],free_tip])
            a,b = soft_scm_coefficients(h,self.soft_scm_response_time,self.soft_scm_damping_ratio,self.soft_scm_regularization)
            velocity = grouped_contact_velocity_torch(free,jac,phi,inverse,h,a,b,self.soft_scm_regularization)
            supported_free = torch.cat([.5*x[10:16]+mobility[:6,:6]@(self.param_.obj_mass_*gravity)/h,free_tip])
            a,b = soft_scm_coefficients(h,self.param_.soft_scm_response_time,self.soft_scm_damping_ratio,self.soft_scm_regularization)
            supported = grouped_contact_velocity_torch(supported_free,jac,phi,mobility,h,a,b,self.soft_scm_regularization)
            normals = jac.reshape(-1,4,9).mean(dim=1)
            gaps = phi.reshape(-1,4).mean(dim=1)
            loading = torch.where(normals[:,6:].square().sum(dim=1)>0,
                (-normals@supported_free-gaps/h).clamp_min(0),torch.zeros_like(gaps)).max()
            fraction = (loading/self.contact_blend_speed).clamp(0,1)
            support_fraction = fraction**3*(10-15*fraction+6*fraction**2)
            velocity = velocity+support_fraction*(supported-velocity)
        q,w = x[3:7],velocity[3:6]
        qnext = q+.5*h*torch.cat([-(q[1:]*w).sum().reshape(1),q[0]*w+torch.linalg.cross(q[1:],w)])
        qnext = qnext/torch.linalg.vector_norm(qnext).clamp_min(1e-10)
        next_w = rotation_torch(qnext).T@rotation_torch(q)@w
        return torch.cat([x[:3]+h*velocity[:3],qnext,x[7:10]+h*velocity[6:],
                          velocity[:3],next_w,(next_tip_velocity if self.use_shared_response else end_tip+velocity[6:]-free_tip)])
