"""Contact-pair selection by the same coupled soft-SCM dynamics as MPC.

Lambda's applied-force model can instantaneously cancel a bad squeeze with
independent tangential forces. Here commands act through finite-mass pads,
OSC damping, unilateral force caps and Coulomb contact before moving the
object. Candidate quality is an optimized rollout cost, not geometric rules.
"""
from types import SimpleNamespace

import numpy as np
import casadi as cs
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from models.bimanual_rigid_contact import BimanualRigidContactModel
from models.support_geometry import support_jacobians
from models.contact_geometry import rotation_cs


class BimanualLambdaPrediction:
    def __init__(self, *, step=.01, horizon=8, stiffness=600., damping=40., tip_mass=2.5,
                 force_limit=12., unilateral_limit=.05, response_time=.05,
                 patch_radius=.03, approach_speed=.25, support_point=(0., 0., 0.),
                 support_half_extents=(1., 1.), position_weight=300., orientation_weight=15.,
                 velocity_weight=.05, pressure_weight=12., force_weight=.01, tangent_weight=.03,
                 tangent_impedance=None, spin_friction=0., roll_friction=0.):
        self.param = SimpleNamespace(h_=step, mpc_tip_effective_mass_=tip_mass,
            mpc_tip_damping_=damping, online_approach_speed=approach_speed, max_ncon_=6,
            robot_stiff_=stiffness*np.eye(6), contact_command_force_limit_=force_limit,
            online_unilateral_force_limit=unilateral_limit, soft_scm_response_time=response_time,
            soft_scm_damping_ratio=1., soft_scm_regularization=1e-8, mpc_patch_radius_=patch_radius,
            support_surface_point_=np.asarray(support_point), support_surface_half_extents_=np.asarray(support_half_extents))
        if tangent_impedance is not None:
            if not 0 < tangent_impedance <= 1:
                raise ValueError('Tangent impedance must be in (0,1]')
            self.param.mpc_tangent_impedance_ = float(tangent_impedance)
        self.param.mpc_spin_friction_ = np.broadcast_to(spin_friction, (2,)).copy()
        self.param.mpc_roll_friction_ = np.broadcast_to(roll_friction, (2,)).copy()
        if (not np.isfinite(np.r_[self.param.mpc_spin_friction_, self.param.mpc_roll_friction_]).all()
                or np.any(self.param.mpc_spin_friction_ < 0) or np.any(self.param.mpc_roll_friction_ < 0)):
            raise ValueError('Rotational friction must be finite and nonnegative')
        self.model = BimanualRigidContactModel(self.param)
        self.horizon, self.stiffness = int(horizon), float(stiffness)
        if self.horizon < 1 or self.stiffness <= 0 or step <= 0:
            raise ValueError('Prediction horizon, step and stiffness must be positive')
        self.bound = force_limit/(stiffness*np.sqrt(3))
        self.position_weight, self.orientation_weight = position_weight, orientation_weight
        self.velocity_weight, self.pressure_weight = velocity_weight, pressure_weight
        self.force_weight, self.tangent_weight = force_weight, tangent_weight
        self._build_rollout()

    def _build_rollout(self):
        """One cached native evaluation and exact Jacobian per LM iteration."""
        x, u = cs.MX.sym('x', 25), cs.MX.sym('u', 6)
        phi, jac = cs.MX.sym('phi', 24), cs.MX.sym('jac', 24, 12)
        parameters, target = cs.MX.sym('parameters', self.model.model_parameter_dim), cs.MX.sym('target', 7)
        pressure, normals = cs.MX.sym('pressure'), cs.MX.sym('normals', 3, 2)
        state = x
        residuals, states, forces = [], [], []
        first_moment = None
        for _ in range(self.horizon):
            world_normals = rotation_cs(state[3:7]) @ normals
            next_state, force_vector, moment = self.model.step_and_wrench_fn(state, u, phi, jac, parameters)
            force = cs.reshape(force_vector, 3, 2)
            if first_moment is None:
                first_moment = moment
            fn = cs.vertcat(*[cs.dot(force[:, side], world_normals[:, side]) for side in range(2)])
            state = next_state
            difference = target[3]*state[4:7]-state[3]*target[4:7]-cs.cross(target[4:7], state[4:7])
            residuals.extend((np.sqrt(self.position_weight)*(state[:3]-target[:3]),
                np.sqrt(8*self.orientation_weight)*difference,
                np.sqrt(self.velocity_weight)*state[13:19],
                # Price insufficient grip, not deviation from an invented
                # spring force K_osc*squeeze_depth. Extra useful pressure is
                # permitted; its realised contact force is regularized below.
                np.sqrt(self.pressure_weight)*cs.fmax(pressure-fn, 0)))
            tangent_force = cs.horzcat(*[force[:, side]-fn[side]*world_normals[:, side] for side in range(2)])
            residuals.extend((np.sqrt(self.force_weight)*cs.reshape(force, 6, 1),
                              np.sqrt(self.tangent_weight)*cs.reshape(tangent_force, 6, 1)))
            states.append(state)
            forces.append(cs.reshape(force, 6, 1))
        # OSC increments are lengths. A force penalty on K*u was punishing
        # actuator effort as though it were contact force, especially when the
        # model predicts a weak pressure response. Keep this regularizer in
        # Cartesian command units, as in online MPC.
        residuals.append(np.sqrt(self.horizon*self.force_weight)*u)
        residual = cs.vertcat(*residuals)
        inputs = [u, x, phi, jac, parameters, target, pressure, normals]
        self.rollout_fn = cs.Function('lambda_coupled_rollout', inputs,
            [residual, cs.horzcat(*states), cs.horzcat(*forces), first_moment])
        self.residual_fn = cs.Function('lambda_coupled_residual', inputs, [residual])
        self.residual_jac_fn = cs.Function('lambda_coupled_jacobian', inputs, [cs.jacobian(residual, u)])

    def set_observation(self, pose, target, velocity, inertia, external, *,
                        friction, support_friction, support_points=(), tip_radius=.01,
                        squeeze_depth=.003, minimum_normal=.035, tips_world=None, tip_velocity=None,
                        measured_normal=None, observed_tip_mask=None):
        self.pose, self.target = np.asarray(pose), np.asarray(target)
        self.velocity, self.inertia, self.external = np.asarray(velocity), np.asarray(inertia), np.asarray(external)
        self.rotation = Rotation.from_quat(self.pose[[4, 5, 6, 3]]).as_matrix()
        self.friction, self.support_friction = float(friction), float(support_friction)
        self.support_points = np.asarray(support_points).reshape(-1, 3)[:4]
        self.tip_radius, self.squeeze_depth = float(tip_radius), float(squeeze_depth)
        # A load-related pressure floor is a reference, not a claim that
        # two equal Fn alone balance the object's force and torque.
        self.pressure = max(float(minimum_normal),
            np.linalg.norm(self.external[:3])/(2*max(.8*self.friction, .05)))
        self.tips_world = None if tips_world is None else np.asarray(tips_world).reshape(2, 3).copy()
        self.observed_tip_mask = np.ones(2, bool) if observed_tip_mask is None else np.asarray(observed_tip_mask, bool).reshape(2)
        self.tip_velocity = np.zeros(6) if tip_velocity is None else np.asarray(tip_velocity).reshape(6).copy()
        self.measured_normal = None if measured_normal is None else np.asarray(measured_normal).reshape(2).copy()

    def _geometry(self, points, normals):
        surface = points@self.rotation.T+self.pose[:3]
        inward = normals@self.rotation.T
        tips = surface-self.tip_radius*inward
        if self.tips_world is not None:
            tips = np.where(self.observed_tip_mask[:, None], self.tips_world, tips)
        state = np.r_[self.pose, tips.ravel(), self.velocity, self.tip_velocity]
        phi, jac = np.ones(24), np.zeros((24, 12))
        for side in range(2):
            rows, _ = support_jacobians(self.pose[:3], self.rotation, surface[side], 12,
                                        self.friction, inward[side])
            rows[:, 6+3*side:9+3*side] = -rows[:, :3]
            phi[4*side:4*side+4] = (surface[side]-tips[side])@inward[side]-self.tip_radius
            jac[4*side:4*side+4] = rows
        for side, point in enumerate(self.support_points, start=2):
            rows, _ = support_jacobians(self.pose[:3], self.rotation, point, 12,
                self.support_friction, [0., 0., 1.])
            jac[4*side:4*side+4] = rows
            phi[4*side:4*side+4] = point[2]-self.param.support_surface_point_[2]
        self.model.set_observation(state, self.inertia, self.external)
        if self.measured_normal is not None:
            self.model.set_contact_preload(state, phi, jac, self.measured_normal)
        return state, phi, jac, self.model.runtime_parameters(), inward

    def _rollout(self, command, geometry, *, details=False):
        state, phi, jac, parameters, inward = geometry
        args = (command, state, phi, jac, parameters, self.target, self.pressure, (inward@self.rotation).T)
        if not details:
            return np.asarray(self.residual_fn(*args)).ravel()
        residual, trajectory, world_forces, moment = self.rollout_fn(*args)
        result, states = np.asarray(residual).ravel(), np.asarray(trajectory).T
        forces = np.asarray(world_forces).T.reshape(self.horizon, 2, 3)
        if details:
            stage = result[:26*self.horizon].reshape(self.horizon, 26)
            components = dict(position=float(np.sum(stage[:, :3]**2)),
                orientation=float(np.sum(stage[:, 3:6]**2)), velocity=float(np.sum(stage[:, 6:12]**2)),
                pressure=float(np.sum(stage[:, 12:14]**2)),
                contact_force=float(np.sum(stage[:, 14:20]**2)), tangent_force=float(np.sum(stage[:, 20:26]**2)),
                command=float(np.sum(result[26*self.horizon:]**2)))
            return dict(cost=float(result@result), x_plus=np.asarray(states[0])[:7],
                        velocity_next=np.asarray(states[0])[13:19], rollout=np.asarray(states),
                        force_vectors_local=forces[0]@self.rotation,
                        moment_vectors_local=np.asarray(moment).reshape(2, 3)@self.rotation,
                        command_delta_world=np.asarray(command).reshape(2, 3),
                        next_position_cost=float(self.position_weight*np.sum((states[0][:3]-self.target[:3])**2)),
                        next_orientation_cost=float(8*self.orientation_weight*max(0., 1-(states[0][3:7]@self.target[3:7])**2)),
                        pose_cost_now=float(self.position_weight*np.sum((state[:3]-self.target[:3])**2)+
                            8*self.orientation_weight*max(0., 1-(state[3:7]@self.target[3:7])**2)),
                        cost_components=components, prediction_initial_state=state.copy(),
                        prediction_dt=float(self.param.h_))

    def evaluate(self, points, normals, *, optimize=False):
        geometry = self._geometry(np.asarray(points), np.asarray(normals))
        initial = np.clip(self.pressure/self.stiffness*geometry[-1].ravel(), -self.bound, self.bound)
        command = initial
        if optimize:
            result = least_squares(lambda u: self._rollout(u, geometry), initial,
                jac=lambda u: np.asarray(self.residual_jac_fn(u, *geometry[:4], self.target, self.pressure,
                                                             (geometry[-1]@self.rotation).T)),
                bounds=(-self.bound, self.bound), max_nfev=15, ftol=1e-5, xtol=1e-5, gtol=1e-5)
            if np.isfinite(result.x).all() and np.sum(self._rollout(result.x, geometry)**2) <= np.sum(self._rollout(initial, geometry)**2):
                command = result.x
        return self._rollout(command, geometry, details=True)

    def rank(self, points, normals, valid, *, refinements=12):
        if refinements < 0:
            raise ValueError('Predictive refinements must be nonnegative')
        self.results = {}
        scores = np.full(len(points), np.inf)
        for row in np.flatnonzero(valid):
            result = self.evaluate(points[row], normals[row])
            if np.isfinite(result['cost']):
                scores[row], self.results[int(row)] = result['cost'], result
        ranked = np.argsort(scores, kind='stable')
        for row in ranked[:refinements]:
            if not np.isfinite(scores[row]):
                break
            result = self.evaluate(points[row], normals[row], optimize=True)
            scores[row], self.results[int(row)] = result['cost'], result
        return scores
