"""Isaac-style velocity-state predictor for two Cartesian fingertips.

State: object XYZ/WXYZ, left/right XYZ, object world-linear/body-angular
velocity, left/right world velocity (25). Action: two OSC increments (6).
The first two patch blocks belong to the left/right pad; remaining blocks
are finite table supports. All velocities and forces use the same mixed
coordinates as a MuJoCo free joint. No reference trajectory is required.
"""

import casadi as cs
import numpy as np

from models.contact_geometry import rotation_cs
from models.soft_scm_contact import grouped_contact_velocity_cs, soft_scm_coefficients


def online_tip_velocity_reference(action, gap, grasp_established, interval, speed_limit):
    """Isaac-style free-space feedforward; touching/loaded pads use raw damping.

    Accept both NumPy and CasADi vectors so prediction and OSC share the law.
    The limit caps the *reference* speed, not the measured tip velocity.
    """
    if isinstance(action, (cs.SX, cs.MX, cs.DM)):
        reference = action / interval
        length = cs.sqrt(cs.sumsqr(reference) + 1e-20)
        reference *= cs.fmin(1., speed_limit / cs.fmax(length, 1e-10))
        # Brake over the last 2 mm, rather than removing feedforward in the
        # centimetre-scale approach region (which makes the tip crawl).
        reference *= cs.fmin(1., cs.fmax(gap / .002, 0.))
        return cs.if_else(grasp_established < .5, reference, cs.DM.zeros(3))
    reference = np.asarray(action, float) / interval
    reference *= min(1., speed_limit / max(np.linalg.norm(reference), 1e-10))
    reference *= np.clip(gap / .002, 0., 1.)
    return reference if not grasp_established else np.zeros(3)


class BimanualRigidContactModel:
    model_parameter_dim = 58  # reference(13), inverse mass(36), wrench(6), latch(1), pad preload bias(2)
    contact_model = "soft-scm-bimanual"
    transports_contacts = True

    def __init__(self, param):
        self.param_ = param
        self.h = float(param.h_)
        self.mass = float(param.mpc_tip_effective_mass_)
        self.damping = float(param.mpc_tip_damping_)
        self.approach_speed = float(getattr(param, "online_approach_speed", .25))
        if not np.isfinite(self.approach_speed) or self.approach_speed < 0:
            raise ValueError("Online approach speed must be finite and nonnegative")
        if not np.isfinite([self.h, self.mass, self.damping]).all() or min(self.h, self.mass, self.damping) <= 0:
            raise ValueError("MPC timestep, fingertip mass and damping must be positive")
        self.reference = np.r_[np.zeros(3), 1., np.zeros(9)]
        self.inverse = np.eye(6)
        self.external = np.zeros(6)
        self.grasp_established = False
        self.normal_preload_bias = np.zeros(2)
        x, u = cs.SX.sym("dual_state", 25), cs.SX.sym("dual_increment", 6)
        phi = cs.SX.sym("patch_gap", 4 * param.max_ncon_)
        jac = cs.SX.sym("patch_rows", 4 * param.max_ncon_, 12)
        mp = cs.SX.sym("dual_physics", self.model_parameter_dim)
        gaps, rows = self.transport(x, mp[:13], phi, jac)
        inverse = cs.diagcat(cs.reshape(mp[13:49], 6, 6), cs.DM.eye(6) / self.mass)
        target_velocity = cs.DM(param.robot_stiff_) @ u / self.damping
        free_tips, end_tips = [], []
        # The OSC clips K*u-D*(v-v_ref), not just K*u. With an isotropic
        # effective mass the free velocity stays on this error direction:
        # integrate the constant-force saturation followed by exponential PD.
        tau = self.mass / self.damping
        force_limit = float(param.contact_command_force_limit_)
        unilateral_limit = float(getattr(param, "online_unilateral_force_limit", .05))
        if not np.isfinite([force_limit, unilateral_limit]).all() or min(force_limit, unilateral_limit) <= 0:
            raise ValueError("OSC force limit must be finite and positive")
        for arm in range(2):
            current = x[19 + 3 * arm:22 + 3 * arm]
            gap = cs.sum1(gaps[4 * arm:4 * arm + 4]) / 4
            reference_velocity = online_tip_velocity_reference(
                u[3 * arm:3 * arm + 3], gap, mp[55], self.h, self.approach_speed)
            error = target_velocity[3 * arm:3 * arm + 3] + reference_velocity - current
            other = 1 - arm
            other_gap = cs.sum1(gaps[4 * other:4 * other + 4]) / 4
            inward = cs.sum1(rows[4 * arm:4 * arm + 4, :3]).T / 4
            unilateral_push = cs.logic_and(cs.logic_and(gap <= 1e-6, other_gap > 1e-6),
                                           cs.dot(error, inward) > 0)
            unilateral_push = cs.logic_and(mp[55] < .5, unilateral_push)
            cap = cs.if_else(unilateral_push, cs.fmin(force_limit, unilateral_limit), force_limit)
            acceleration = cap / self.mass
            length = cs.sqrt(cs.sumsqr(error) + 1e-20)
            direction = error / cs.fmax(length, 1e-10)
            saturated_time = cs.fmin(self.h, cs.fmax((length - cap / self.damping) / acceleration, 0))
            remaining = self.h - saturated_time
            after = length - acceleration * saturated_time
            end_change = acceleration * saturated_time + after * (1 - cs.exp(-remaining / tau))
            displacement = .5 * acceleration * saturated_time ** 2
            displacement += acceleration * saturated_time * remaining
            displacement += after * (remaining - tau * (1 - cs.exp(-remaining / tau)))
            free_tips.append(current + direction * displacement / self.h)
            end_tips.append(current + direction * end_change)
        free_tip, end_tip = cs.vertcat(*free_tips), cs.vertcat(*end_tips)
        free = cs.vertcat(x[13:19] + self.h * inverse[:6, :6] @ mp[49:55], free_tip)
        a, b = soft_scm_coefficients(self.h, param.soft_scm_response_time,
                                     param.soft_scm_damping_ratio, param.soft_scm_regularization)
        spin = np.broadcast_to(getattr(param, 'mpc_spin_friction_', 0.), (2,))
        roll = np.broadcast_to(getattr(param, 'mpc_roll_friction_', 0.), (2,))
        moment_rows, moment_coefficients, moment_axes = [], [], []
        for arm in range(param.max_ncon_):
            if arm >= 2 or (spin[arm] == 0 and roll[arm] == 0):
                moment_rows.append(None)
                moment_coefficients.append((0., 0., 0.))
                if arm < 2:
                    moment_axes.append(cs.SX.zeros(3, 3))
                continue
            block = rows[4*arm:4*arm+4, :3]
            normal = cs.sum1(block).T/4
            mu = cs.sqrt(cs.sumsqr((block[0, :]-block[2, :])/2)+1e-20)
            axes = cs.horzcat(normal, (block[0, :]-block[2, :]).T/(2*cs.fmax(mu, 1e-10)),
                             (block[1, :]-block[3, :]).T/(2*cs.fmax(mu, 1e-10)))
            helper = cs.if_else(cs.fabs(normal[0]) < .9, cs.DM([1., 0., 0.]), cs.DM([0., 1., 0.]))
            first = cs.cross(normal, helper)
            first /= cs.sqrt(cs.sumsqr(first)+1e-20)
            # Zero sliding friction erases tangent directions in the four
            # force rows; it must not erase nonzero rolling friction as well.
            axes = cs.if_else(mu > 1e-8, axes, cs.horzcat(normal, first, cs.cross(normal, first)))
            body_axes = rotation_cs(x[3:7]).T@axes
            moment_rows.append(cs.horzcat(cs.SX.zeros(3, 3), body_axes.T, cs.SX.zeros(3, 6)))
            moment_coefficients.append((spin[arm], roll[arm], roll[arm]))
            moment_axes.append(axes)
        # Solve force and rotational friction in the same contact sweeps.
        # Applying a moment only after the force solve artificially made an
        # immovable wrist able to cancel rotation without resisting pad motion.
        velocity, impulses, moment_impulses = grouped_contact_velocity_cs(
            free, rows, gaps, inverse, self.h, a, b, param.soft_scm_regularization,
            normal_bias=cs.vertcat(mp[56:58], cs.DM.zeros(param.max_ncon_-2)),
            tangent_impedance=getattr(param, 'mpc_tangent_impedance_', None),
            moment_rows=moment_rows, moment_friction=moment_coefficients, return_moments=True)
        moments = [moment_axes[arm]@cs.vertcat(*moment_impulses[arm])/self.h for arm in range(2)]
        q, omega = x[3:7], velocity[3:6]
        next_q = q + .5 * self.h * cs.vertcat(-cs.dot(q[1:], omega),
                                              q[0] * omega + cs.cross(q[1:], omega))
        next_q /= cs.sqrt(cs.sumsqr(next_q) + 1e-20)
        next_omega = rotation_cs(next_q).T @ rotation_cs(q) @ omega
        next_x = cs.vertcat(x[:3] + self.h * velocity[:3], next_q,
                           x[7:13] + self.h * velocity[6:12], velocity[:3], next_omega,
                           end_tip + velocity[6:12] - free_tip)
        forces = []
        for arm in range(2):
            block = rows[4 * arm:4 * arm + 4, :3]
            normal = cs.sum1(block).T / 4
            mu = cs.sqrt(cs.sumsqr((block[0, :] - block[2, :]) / 2) + 1e-20)
            tangents = cs.horzcat((block[0, :] - block[2, :]).T,
                                  (block[1, :] - block[3, :]).T) / (2 * cs.fmax(mu, 1e-10))
            fn, ft = impulses[arm]
            forces.append((normal * fn + tangents @ ft) / self.h)
        inputs = [x, u, phi, jac, mp]
        self.prediction_fn = cs.Function("dual_rigid_step", inputs, [next_x], {"cse": True})
        self.contact_force_fn = cs.Function("dual_contact_forces", inputs, [cs.vertcat(*forces)], {"cse": True})
        self.contact_moment_fn = cs.Function('dual_contact_moments', inputs, [cs.vertcat(*moments)], {"cse": True})
        self.step_and_wrench_fn = cs.Function('dual_step_and_wrench', inputs,
            [next_x, cs.vertcat(*forces), cs.vertcat(*moments)], {"cse": True})
        normal_forces = cs.vertcat(*[cs.dot(cs.sum1(rows[4*arm:4*arm+4, :3]).T/4, forces[arm])
                                    for arm in range(2)])
        self.preload_jac_fn = cs.Function('dual_preload_jacobian', inputs,
            [cs.jacobian(normal_forces, mp)[:, 56:58]], {"cse": True})
        self.geometry_fn = cs.Function("dual_patch_geometry", [x, phi, jac, mp], [gaps, rows])

    def transport(self, state, reference, phi, jac):
        rotation, observed = rotation_cs(state[3:7]), rotation_cs(reference[3:7])
        gaps, rows = [], []
        for patch in range(self.param_.max_ncon_):
            block = jac[4 * patch:4 * patch + 4, :]
            normal, tangent = cs.sum1(block) / 4, (block[0, :] - block[2, :]) / 2
            ns = cs.sqrt(cs.sumsqr(normal[:3]) + 1e-20)
            ts = cs.sqrt(cs.sumsqr(tangent[:3]) + 1e-20)
            n, t = observed.T @ normal[:3].T / ns, observed.T @ tangent[:3].T / ts
            lever = cs.cross(n, normal[3:6].T / ns)
            lever += n * cs.dot(cs.cross(t, tangent[3:6].T / ts), n)
            point = state[:3] + rotation @ lever
            robot = cs.SX.zeros(4, 6)
            if patch < 2:
                directions = block[:, :3] @ observed @ rotation.T
                robot[:, 3 * patch:3 * patch + 3] = -directions
                tip = state[7 + 3 * patch:10 + 3 * patch]
                old_tip = reference[7 + 3 * patch:10 + 3 * patch]
                new_normal = cs.sum1(directions).T / 4
                change = cs.dot(new_normal, point - tip) - cs.dot(
                    normal[:3].T, reference[:3] + observed @ lever - old_tip)
                # A transported patch cannot push a finger that has missed it
                # tangentially. Re-query the actual nearest mesh patch next cycle.
                relative = tip - point
                tangent_distance = cs.sumsqr(relative - new_normal * cs.dot(new_normal, relative))
                active = tangent_distance <= self.param_.mpc_patch_radius_ ** 2
            else:
                directions = block[:, :3]
                change = cs.dot(normal[:3].T, point - reference[:3] - observed @ lever)
                uv = point[:2] - cs.DM(self.param_.support_surface_point_[:2])
                bounds = self.param_.support_surface_half_extents_
                active = cs.logic_and(cs.fabs(uv[0]) <= bounds[0], cs.fabs(uv[1]) <= bounds[1])
            angular = cs.vertcat(*[cs.cross(lever, rotation.T @ directions[i, :].T).T for i in range(4)])
            row = cs.horzcat(directions, angular, robot)
            active = cs.logic_and(active, cs.sumsqr(normal[:3]) > .5)
            rows.append(cs.if_else(active, row, cs.DM.zeros(4, 12)))
            gaps.append(cs.if_else(active, phi[4 * patch:4 * patch + 4] + change, cs.DM.ones(4)))
        return cs.vertcat(*gaps), cs.vertcat(*rows)

    def set_observation(self, state, inertia, external_wrench, grasp_established=False):
        inertia = np.asarray(inertia, float).reshape(6, 6)
        state = np.asarray(state, float).reshape(25)
        external = np.asarray(external_wrench, float).reshape(6)
        if not np.isfinite(inertia).all() or not np.isfinite(state).all() or not np.isfinite(external).all():
            raise ValueError("MPC observation and dynamics must be finite")
        if not np.allclose(inertia, inertia.T, atol=1e-10):
            raise ValueError("Object inertia must be symmetric positive definite")
        np.linalg.cholesky(inertia)
        self.inverse = np.linalg.inv(inertia)
        self.reference = state[:13].copy()
        self.grasp_established = bool(grasp_established)
        self.normal_preload_bias = np.zeros(2)
        # _lambda_object_dynamics supplies world torque; free-joint omega is body-frame.
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_quat(state[[4, 5, 6, 3]]).as_matrix()
        self.external = external.copy()
        self.external[3:] = rotation.T @ external[3:]

    def set_contact_preload(self, state, phi, jac, measured_normal):
        """Reconcile observed pad compression with the nominal soft response.

        This is a contact stress observation, separate from signed geometry.
        Calibrate the coupled normal RHS, including off-diagonal Delassus
        terms, so equal opposing preload cannot accelerate a free object.
        Newly proposed or separated contacts have no observed preload.
        """
        measured = np.asarray(measured_normal, float).reshape(2)
        if not np.isfinite(measured).all() or np.any(measured < 0):
            raise ValueError('Measured pad normal force must be finite and nonnegative')
        self.normal_preload_bias[:] = 0.
        jac = np.asarray(jac).reshape(self.param_.max_ncon_*4, 12)
        normals = jac[:8].reshape(2, 4, 12).mean(axis=1)
        gaps = np.asarray(phi)[:8].reshape(2, 4).mean(axis=1)
        active = (gaps <= 1e-6) & (np.linalg.norm(normals[:, :3], axis=1) > .5) & (measured > 0)
        if not active.any():
            return
        nominal = np.asarray(self.contact_force_fn(state, np.zeros(6), phi, jac,
                              self.runtime_parameters())).reshape(2, 3)
        fn = np.sum(nominal*normals[:, :3], axis=1)
        delta = np.where(active, measured-fn, 0.)
        inverse = np.zeros((12, 12))
        inverse[:6, :6] = self.inverse
        inverse[6:, 6:] = np.eye(6)/self.mass
        gram = normals @ inverse @ normals.T
        a, _ = soft_scm_coefficients(self.h, self.param_.soft_scm_response_time,
                                     self.param_.soft_scm_damping_ratio, self.param_.soft_scm_regularization)
        regularizer = (np.diag(gram)+self.param_.soft_scm_regularization)/a-np.diag(gram)
        correction = -self.h*(gram@delta+regularizer*delta)
        self.normal_preload_bias = np.where(active, correction, 0.)
        # Rotational friction and table reaction also couple normal rows.
        # Refine their two-variable stress observation using the exact local
        # force response, without changing signed gaps or friction limits.
        indices = np.flatnonzero(active)
        for _ in range(2):
            parameters = self.runtime_parameters()
            force = np.asarray(self.contact_force_fn(state, np.zeros(6), phi, jac, parameters)).reshape(2, 3)
            error = measured-np.sum(force*normals[:, :3], axis=1)
            if np.max(np.abs(error[indices])) < 1e-5:
                break
            response = np.asarray(self.preload_jac_fn(state, np.zeros(6), phi, jac, parameters))[np.ix_(indices, indices)]
            if not np.isfinite(response).all() or np.linalg.cond(response) > 1e8:
                break
            self.normal_preload_bias[indices] += np.linalg.solve(response, error[indices])

    def runtime_parameters(self, unused=None):
        return np.r_[self.reference, self.inverse.reshape(-1, order="F"), self.external,
                     float(self.grasp_established), self.normal_preload_bias]
