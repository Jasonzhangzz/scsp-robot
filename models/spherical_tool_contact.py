"""Finite-response spherical tool with optimized world rotational increments.

State: CartesianRigidContactModel's 19 states + tool quaternion(wxyz) and
world angular velocity. The observed contact geometry stays in the existing
nine-velocity format; rotational contact columns are constructed at each step.
"""
import casadi as cs
import numpy as np
from models.contact_geometry import rotation_cs
from models.soft_scm_contact import grouped_contact_velocity_cs, soft_scm_coefficients


def sphere_contact_jacobian(jac, radius):
    """Append angular columns for v_surface = v_center + omega x lever."""
    blocks = []
    for start in range(0, jac.size1(), 4):
        rows = jac[start:start+4, :]
        outward = cs.sum1(rows[:, 6:9]).T/4
        outward /= cs.sqrt(cs.sumsqr(outward)+1e-20)
        lever = -float(radius)*outward
        blocks.append(cs.horzcat(rows, -rows[:, 6:9]@cs.skew(lever)))
    return cs.vertcat(*blocks)


def integrate_world_rotation(q, omega, dt):
    qnext = q+.5*dt*cs.vertcat(-cs.dot(q[1:4], omega), q[0]*omega+cs.cross(omega,q[1:4]))
    return qnext/cs.sqrt(cs.sumsqr(qnext)+1e-20)


class SphericalToolContactModel:
    model_parameter_dim = 1
    contact_model = 'soft-scm'
    online_scm = False
    tool_rotation_step = .04  # rad per policy interval
    tool_tilt_limit = np.deg2rad(35.)
    angular_stiffness = 10.
    angular_inertia = .02
    angular_damping = 2*np.sqrt(angular_stiffness*angular_inertia)
    radius = .01

    def __init__(self, param):
        self.param_ = param
        if getattr(param,'shared_contact_dynamics_',False) or getattr(param,'contact_dynamics_variant_','legacy') != 'legacy':
            raise ValueError('Tool-pose experiment currently requires legacy contact dynamics')
        h = float(param.h_)
        mass = np.r_[[param.sim_obj_mass_]*3,param.sim_obj_inertia_diag_,
                     [getattr(param,'mpc_tip_effective_mass_',2.5)]*3,[self.angular_inertia]*3]
        self.base_inverse = np.diag(1/mass)
        self.scm_theta = np.zeros(6)
        x,u = cs.SX.sym('tool_state',26),cs.SX.sym('pose_increment',6)
        phi = cs.SX.sym('gap',4*param.max_ncon_)
        jac = cs.SX.sym('jac',4*param.max_ncon_,9)
        sigma = cs.SX.sym('sigma')
        rows = sphere_contact_jacobian(jac,self.radius)
        damping = float(getattr(param,'mpc_tip_damping_',40.))
        tip_mass = float(getattr(param,'mpc_tip_effective_mass_',2.5))
        decay = np.exp(-h*damping/tip_mass)
        memory = tip_mass/damping/h*(1-decay)
        self.response = h*(1-memory)*np.mean(np.diag(np.asarray(param.robot_stiff_)/damping))
        free_tip = memory*x[16:19]+(1-memory)*cs.DM(param.robot_stiff_)/damping@u[:3]
        end_tip = decay*x[16:19]+(1-decay)*cs.DM(param.robot_stiff_)/damping@u[:3]
        adecay = np.exp(-h*self.angular_damping/self.angular_inertia)
        amemory = self.angular_inertia/self.angular_damping/h*(1-adecay)
        free_angular = amemory*x[23:26]+(1-amemory)*self.angular_stiffness/self.angular_damping*u[3:6]
        end_angular = adecay*x[23:26]+(1-adecay)*self.angular_stiffness/self.angular_damping*u[3:6]
        free = cs.vertcat(x[10:13]+h*cs.DM(param.gravity_[:3]),x[13:16],free_tip,free_angular)
        a,b = soft_scm_coefficients(h,float(getattr(param,'mpc_rigid_contact_time_',.01)),
                                    param.soft_scm_damping_ratio,param.soft_scm_regularization)
        velocity = grouped_contact_velocity_cs(free,rows,phi,cs.DM(self.base_inverse),h,a,b,param.soft_scm_regularization)
        mobility = np.zeros((12,12));mobility[:9,:9] = np.linalg.inv(param.Q)
        mobility[9:,9:] = np.eye(3)/self.angular_stiffness
        supported_free = cs.vertcat(.5*x[10:16]+cs.DM(mobility[:6,:6])@cs.DM(param.obj_mass_*param.gravity_)/h,
                                    free_tip,free_angular)
        ca,cb = soft_scm_coefficients(h,param.soft_scm_response_time,param.soft_scm_damping_ratio,param.soft_scm_regularization)
        supported = grouped_contact_velocity_cs(supported_free,rows,phi,cs.DM(mobility),h,ca,cb,param.soft_scm_regularization)
        loading = 0
        for start in range(0,4*param.max_ncon_,4):
            normal = cs.sum1(rows[start:start+4,:])/4
            gap = cs.sum1(phi[start:start+4])/4
            loading = cs.fmax(loading,cs.if_else(cs.sumsqr(normal[6:9])>0,
                cs.fmax(-(normal@supported_free)-gap/h,0),0))
        fraction = cs.fmin(1.,cs.fmax(0.,loading/float(getattr(param,'mpc_contact_blend_speed_',.005))))
        velocity += fraction**3*(10-15*fraction+6*fraction**2)*(supported-velocity)
        q,w = x[3:7],velocity[3:6]
        qnext = q+.5*h*cs.vertcat(-cs.dot(q[1:4],w),q[0]*w+cs.cross(q[1:4],w))
        qnext /= cs.sqrt(cs.sumsqr(qnext)+1e-20)
        next_w = rotation_cs(qnext).T@rotation_cs(q)@w
        tool_qnext = integrate_world_rotation(x[19:23],velocity[9:12],h)
        xn = cs.vertcat(x[:3]+h*velocity[:3],qnext,x[7:10]+h*velocity[6:9],velocity[:3],next_w,
            end_tip+velocity[6:9]-free_tip,tool_qnext,end_angular+velocity[9:12]-free_angular)
        self.prediction_fn = cs.Function('spherical_tool_step',[x,u,phi,jac,sigma],[xn])

    def runtime_parameters(self,sigma): return float(sigma)
    def current_inverse(self): return self.base_inverse

    def observation_state(self,pose,linear=None,angular=None,fingertip=None,tool_quaternion=None,tool_angular=None):
        from models.cartesian_rigid_contact import CartesianRigidContactModel
        base = CartesianRigidContactModel.observation_state(self,pose,linear,angular,fingertip)
        quaternion = np.array([0.,1.,0.,0.]) if tool_quaternion is None else np.asarray(tool_quaternion,dtype=float)
        quaternion /= max(np.linalg.norm(quaternion),1e-12)
        omega = np.zeros(3) if tool_angular is None else np.asarray(tool_angular,dtype=float)
        return np.r_[base,quaternion,omega]
