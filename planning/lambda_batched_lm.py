"""Independent, bounded contact-force LM solves for discrete contact pairs.

All dynamics, residuals, finite differences, matrix factorizations and LM
updates run in Warp.  Poses use ``[xyz, qw, qx, qy, qz]``; velocities and the
full inertia matrix use world linear and body angular coordinates.  External
wrenches are world-frame about the body origin.  Omitting the wrench supplies
uniform gravity, including the COM offset encoded by the full inertia matrix.
An optional generalized bias wrench is already in the mixed coordinates.
Returned device buffers are reused.
"""

from types import SimpleNamespace
from time import perf_counter

import numpy as np


_KERNELS = None


def ensure_kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import warp as wp

    wp.init()
    Vec6 = wp.types.vector(length=6, dtype=wp.float32)
    Vec12 = wp.types.vector(length=12, dtype=wp.float32)
    Mat6 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
    Jac = wp.types.matrix(shape=(12, 6), dtype=wp.float32)

    @wp.struct
    class Physics:
        position: wp.vec3
        quaternion: wp.quat
        target_position: wp.vec3
        target_quaternion: wp.quat
        inverse: Mat6
        free_velocity: Vec6
        support_row: Vec6
        support_gap: float
        support_mass: float
        stiffness: float
        damping: float
        dt: float
        valid: int

    @wp.struct
    class Job:
        points: wp.array(dtype=wp.vec3)
        normals: wp.array(dtype=wp.vec3)
        mask: wp.array(dtype=int)
        initial: wp.array2d(dtype=float)
        use_initial: wp.array(dtype=int)
        physics: wp.array(dtype=Physics)
        pose: wp.array(dtype=float)
        target: wp.array(dtype=float)
        velocity: wp.array(dtype=float)
        external: wp.array(dtype=float)
        use_default_gravity: wp.array(dtype=int)
        bias: wp.array(dtype=float)
        inertia: wp.array2d(dtype=float)
        support_row: wp.array(dtype=float)
        support_gap: wp.array(dtype=float)
        dt: wp.array(dtype=float)
        forces: wp.array2d(dtype=float)
        poses: wp.array2d(dtype=float)
        velocities: wp.array2d(dtype=float)
        force_vectors_local: wp.array(dtype=wp.vec3)
        costs: wp.array(dtype=float)
        pose_costs: wp.array(dtype=float)
        reductions: wp.array(dtype=float)
        baseline_reductions: wp.array(dtype=float)
        valid: wp.array(dtype=int)
        damping: wp.array(dtype=float)
        trust: wp.array(dtype=float)
        accepted: wp.array(dtype=int)
        baseline_pose: wp.array(dtype=float)
        baseline_cost: wp.array(dtype=float)
        current_pose_cost: wp.array(dtype=float)
        selected: wp.array(dtype=int)
        selection_override: wp.array(dtype=int)
        selected_statistics: wp.array(dtype=float)
        selected_points: wp.array(dtype=wp.vec3)
        selected_normals: wp.array(dtype=wp.vec3)
        selected_forces: wp.array(dtype=float)
        selected_vectors: wp.array(dtype=wp.vec3)
        selected_pose: wp.array(dtype=float)
        selected_velocity: wp.array(dtype=float)
        selected_wrench: wp.array(dtype=float)
        mu: float
        force_limit: float
        initial_normal: float
        pos_weight: float
        ori_weight: float
        force_weight: float
        tangent_weight: float
        initial_damping: float
        initial_trust: float
        fd_step: float
        support_stiffness: float
        support_damping: float

    @wp.func
    def cholesky(matrix: Mat6):
        lower = Mat6(0.0)
        valid = int(1)
        for row in range(6):
            for col in range(row + 1):
                value = matrix[row, col]
                for k in range(col):
                    value -= lower[row, k] * lower[col, k]
                if row == col:
                    if not wp.isfinite(value) or value <= 1.0e-25:
                        valid = 0
                    lower[row, col] = wp.sqrt(wp.max(value, 1.0e-25))
                else:
                    lower[row, col] = value / lower[col, col]
        return lower, valid

    @wp.func
    def triangular_solve(lower: Mat6, rhs: Vec6):
        interim = Vec6(0.0)
        result = Vec6(0.0)
        for row in range(6):
            value = rhs[row]
            for col in range(row):
                value -= lower[row, col] * interim[col]
            interim[row] = value / lower[row, row]
        for reverse in range(6):
            row = 5 - reverse
            value = interim[row]
            for col in range(row + 1, 6):
                value -= lower[col, row] * result[col]
            result[row] = value / lower[row, row]
        return result

    @wp.func
    def frame(normal: wp.vec3):
        n = wp.normalize(normal)
        helper = wp.vec3(1.0, 0.0, 0.0)
        if wp.abs(n[0]) > 0.9:
            helper = wp.vec3(0.0, 1.0, 0.0)
        tangent = wp.normalize(wp.cross(n, helper))
        return n, tangent, wp.cross(n, tangent)

    @wp.func
    def projected_force(force: Vec6, mu: float, limit: float):
        result = Vec6(0.0)
        normal_limit = limit / wp.sqrt(1.0 + 2.0 * mu * mu)
        for contact in range(2):
            offset = 3 * contact
            normal = force[offset]
            a = wp.abs(force[offset + 1])
            b = wp.abs(force[offset + 2])
            lo = float(0.0)
            hi = normal_limit
            # Euclidean projection onto the bounded square friction pyramid.
            # A tangential proposal can therefore produce a positive normal.
            for iteration in range(25):
                mid = 0.5 * (lo + hi)
                derivative = mid - normal - mu * wp.max(a - mu * mid, 0.0)
                derivative -= mu * wp.max(b - mu * mid, 0.0)
                if derivative > 0.0:
                    hi = mid
                else:
                    lo = mid
            fn = wp.clamp(0.5 * (lo + hi), 0.0, normal_limit)
            if normal <= 0.0 and a == 0.0 and b == 0.0:
                fn = 0.0
            result[offset] = fn
            result[offset + 1] = wp.clamp(force[offset + 1], -mu * fn, mu * fn)
            result[offset + 2] = wp.clamp(force[offset + 2], -mu * fn, mu * fn)
        return result

    @wp.func
    def predict(physics: Physics, force: Vec6, p0: wp.vec3, p1: wp.vec3,
                n0: wp.vec3, n1: wp.vec3):
        a, b, c = frame(n0)
        d, e, f = frame(n1)
        force0 = a * force[0] + b * force[1] + c * force[2]
        force1 = d * force[3] + e * force[4] + f * force[5]
        linear = wp.quat_rotate(physics.quaternion, force0 + force1)
        angular = wp.cross(p0, force0) + wp.cross(p1, force1)
        wrench = Vec6(linear[0], linear[1], linear[2], angular[0], angular[1], angular[2])
        velocity = physics.free_velocity + physics.dt * (physics.inverse * wrench)
        # Official ComFree: evaluate at smooth predicted velocity, then
        # apply h M^-1 J^T f once. Lambda is an applied contact wrench.
        if physics.support_gap < 0.0:
            speed = wp.dot(physics.support_row, velocity)
            acc = -(physics.damping / physics.dt) * speed
            acc -= (physics.stiffness / physics.dt) * (physics.dt * speed + physics.support_gap)
            reaction = wp.max(physics.support_mass * acc, 0.0)
            velocity += physics.dt * (physics.inverse * physics.support_row) * reaction
        position = physics.position + physics.dt * wp.vec3(velocity[0], velocity[1], velocity[2])
        q = physics.quaternion
        omega = wp.quat(velocity[3], velocity[4], velocity[5], 0.0)
        # Right multiplication: MuJoCo free-joint angular velocity is body-frame.
        increment = q * omega
        next_q = wp.normalize(q + (0.5 * physics.dt) * increment)
        return position, next_q, velocity

    @wp.func
    def pose_cost(physics: Physics, position: wp.vec3, quaternion: wp.quat,
                  pos_weight: float, ori_weight: float):
        difference = wp.quat_inverse(physics.target_quaternion) * quaternion
        orientation = wp.vec3(difference[0], difference[1], difference[2])
        return pos_weight * wp.length_sq(position - physics.target_position) + ori_weight * wp.length_sq(orientation)

    @wp.func
    def residual(job: Job, force: Vec6, candidate: int):
        result = Vec12(0.0)
        physics = job.physics[0]
        p, q, velocity = predict(physics, force, job.points[2 * candidate],
                                job.points[2 * candidate + 1], job.normals[2 * candidate],
                                job.normals[2 * candidate + 1])
        difference = wp.quat_inverse(physics.target_quaternion) * q
        position = p - physics.target_position
        for component in range(3):
            result[component] = wp.sqrt(job.pos_weight) * position[component]
            result[3 + component] = wp.sqrt(job.ori_weight) * difference[component]
        for component in range(6):
            weight = job.force_weight
            if component % 3 != 0:
                weight += job.tangent_weight
            result[6 + component] = wp.sqrt(weight) * force[component]
        return result

    @wp.kernel(enable_backward=False)
    def prepare(job: Job):
        physics = Physics()
        physics.position = wp.vec3(job.pose[0], job.pose[1], job.pose[2])
        physics.quaternion = wp.normalize(wp.quat(job.pose[4], job.pose[5], job.pose[6], job.pose[3]))
        physics.target_position = wp.vec3(job.target[0], job.target[1], job.target[2])
        physics.target_quaternion = wp.normalize(wp.quat(job.target[4], job.target[5], job.target[6], job.target[3]))
        physics.dt = job.dt[0]
        matrix = Mat6(0.0)
        for row in range(6):
            for col in range(6):
                matrix[row, col] = job.inertia[row, col]
        lower, valid = cholesky(matrix)
        input_q = wp.quat(job.pose[4], job.pose[5], job.pose[6], job.pose[3])
        target_q = wp.quat(job.target[4], job.target[5], job.target[6], job.target[3])
        if wp.length(input_q) < 1.0e-12 or wp.length(target_q) < 1.0e-12:
            valid = 0
        for component in range(7):
            if not wp.isfinite(job.pose[component]) or not wp.isfinite(job.target[component]):
                valid = 0
        for component in range(6):
            if not wp.isfinite(job.velocity[component]) or not wp.isfinite(job.external[component]):
                valid = 0
            if not wp.isfinite(job.bias[component]) or not wp.isfinite(job.support_row[component]):
                valid = 0
        if not wp.isfinite(job.support_gap[0]):
            valid = 0
        for row in range(6):
            for col in range(6):
                if not wp.isfinite(matrix[row, col]):
                    valid = 0
                difference = wp.abs(matrix[row, col] - matrix[col, row])
                if difference > 1.0e-7 * wp.max(wp.abs(matrix[row, col]), wp.abs(matrix[col, row])) + 1.0e-10:
                    valid = 0
        inverse = Mat6(0.0)
        for col in range(6):
            rhs = Vec6(0.0)
            rhs[col] = 1.0
            column = triangular_solve(lower, rhs)
            for row in range(6):
                inverse[row, col] = column[row]
        physics.inverse = inverse
        torque = wp.quat_rotate_inv(physics.quaternion, wp.vec3(job.external[3], job.external[4], job.external[5]))
        external = Vec6(job.external[0], job.external[1], job.external[2], torque[0], torque[1], torque[2])
        if job.use_default_gravity[0] != 0:
            # Uniform gravity acts at the COM.  The full mixed-coordinate mass
            # matrix supplies its torque about an offset body-frame origin.
            external = matrix * Vec6(0.0, 0.0, -9.81, 0.0, 0.0, 0.0)
        velocity = Vec6(0.0)
        support = Vec6(0.0)
        for component in range(6):
            external[component] -= job.bias[component]
            velocity[component] = job.velocity[component]
            support[component] = job.support_row[component]
        physics.free_velocity = velocity + physics.dt * (inverse * external)
        physics.support_row = support
        physics.support_gap = job.support_gap[0]
        invweight = (inverse[0, 0] + inverse[1, 1] + inverse[2, 2]) / 3.0
        x = wp.min(wp.abs(physics.support_gap) / 0.01, 1.0)
        curve = wp.where(x < 0.5, 2.0*x*x, 1.0-2.0*(1.0-x)*(1.0-x))
        imp = 0.9 + 0.05*curve
        physics.support_mass = 1.0 / wp.max(invweight*(1.0-imp)/imp, 1.0e-15)
        physics.stiffness = job.support_stiffness
        physics.damping = job.support_damping
        physics.valid = valid
        job.physics[0] = physics
        # A zero force prediction still includes velocity, external wrench and
        # environment response, using exactly the same physics as every pair.
        n = wp.vec3(1.0, 0.0, 0.0)
        position, quaternion, next_velocity = predict(physics, Vec6(0.0), wp.vec3(0.0), wp.vec3(0.0), n, n)
        for component in range(3):
            job.baseline_pose[component] = position[component]
        job.baseline_pose[3] = quaternion[3]
        for component in range(3):
            job.baseline_pose[4 + component] = quaternion[component]
        job.baseline_cost[0] = pose_cost(physics, position, quaternion, job.pos_weight, job.ori_weight)
        job.current_pose_cost[0] = pose_cost(physics, physics.position, physics.quaternion, job.pos_weight, job.ori_weight)

    @wp.kernel(enable_backward=False)
    def initialize(job: Job):
        candidate = wp.tid()
        job.valid[candidate] = 0
        job.costs[candidate] = wp.inf
        job.accepted[candidate] = 0
        job.damping[candidate] = job.initial_damping
        job.trust[candidate] = job.initial_trust
        if job.mask[candidate] == 0 or job.physics[0].valid == 0:
            return
        for contact in range(2):
            if wp.length(job.normals[2 * candidate + contact]) < 1.0e-8:
                return
            for component in range(3):
                if not wp.isfinite(job.points[2 * candidate + contact][component]):
                    return
                if not wp.isfinite(job.normals[2 * candidate + contact][component]):
                    return
        value = Vec6(0.0)
        for component in range(6):
            if job.use_initial[0] != 0:
                value[component] = job.initial[candidate, component]
            elif component % 3 == 0:
                value[component] = job.initial_normal
        value = projected_force(value, job.mu, job.force_limit)
        terms = residual(job, value, candidate)
        cost = wp.dot(terms, terms)
        if not wp.isfinite(cost):
            return
        for component in range(6):
            job.forces[candidate, component] = value[component]
        job.costs[candidate] = cost
        job.valid[candidate] = 1

    @wp.kernel(enable_backward=False)
    def iteration(job: Job):
        candidate = wp.tid()
        if job.valid[candidate] == 0:
            return
        force = Vec6(0.0)
        for component in range(6):
            force[component] = job.forces[candidate, component]
        terms = residual(job, force, candidate)
        jacobian = Jac(0.0)
        for parameter in range(6):
            h = job.fd_step * (1.0 + wp.abs(force[parameter]))
            positive = Vec6(force)
            negative = Vec6(force)
            positive[parameter] += h
            negative[parameter] -= h
            rp = residual(job, positive, candidate)
            rn = residual(job, negative, candidate)
            for row in range(12):
                jacobian[row, parameter] = (rp[row] - rn[row]) / (2.0 * h)
        matrix = Mat6(0.0)
        gradient = Vec6(0.0)
        for row in range(6):
            for residual_row in range(12):
                gradient[row] += jacobian[residual_row, row] * terms[residual_row]
            for col in range(6):
                for residual_row in range(12):
                    matrix[row, col] += jacobian[residual_row, row] * jacobian[residual_row, col]
            matrix[row, row] += job.damping[candidate] * wp.max(matrix[row, row], 1.0e-12)
        lower, good = cholesky(matrix)
        delta = triangular_solve(lower, -gradient)
        if good == 0 or not wp.isfinite(wp.length(delta)):
            job.damping[candidate] = wp.min(job.damping[candidate] * 10.0, 1.0e12)
            return
        length = wp.length(delta)
        if length > job.trust[candidate]:
            delta *= job.trust[candidate] / length
        proposal = projected_force(force + delta, job.mu, job.force_limit)
        actual_step = proposal - force
        predicted_residual = terms + jacobian * actual_step
        predicted_reduction = wp.dot(terms, terms) - wp.dot(predicted_residual, predicted_residual)
        trial = residual(job, proposal, candidate)
        trial_cost = wp.dot(trial, trial)
        actual_reduction = job.costs[candidate] - trial_cost
        if wp.isfinite(trial_cost) and actual_reduction > 0.0 and predicted_reduction > 0.0:
            ratio = actual_reduction / predicted_reduction
            for component in range(6):
                job.forces[candidate, component] = proposal[component]
            job.costs[candidate] = trial_cost
            job.accepted[candidate] = job.accepted[candidate] + 1
            if ratio > 0.75:
                job.damping[candidate] = wp.max(job.damping[candidate] / 3.0, 1.0e-9)
                job.trust[candidate] = wp.min(job.trust[candidate] * 2.0, 2.0 * job.force_limit)
            elif ratio < 0.25:
                job.damping[candidate] = wp.min(job.damping[candidate] * 3.0, 1.0e12)
                job.trust[candidate] = job.trust[candidate] * 0.5
        else:
            job.damping[candidate] = wp.min(job.damping[candidate] * 10.0, 1.0e12)
            job.trust[candidate] = wp.max(job.trust[candidate] * 0.5, 1.0e-6)

    @wp.kernel(enable_backward=False)
    def finalize(job: Job):
        candidate = wp.tid()
        if job.valid[candidate] == 0:
            job.pose_costs[candidate] = wp.inf
            job.reductions[candidate] = -wp.inf
            job.baseline_reductions[candidate] = -wp.inf
            return
        force = Vec6(0.0)
        for component in range(6):
            force[component] = job.forces[candidate, component]
        # A feasible no-contact fallback prevents a failed warm/LM solution
        # from being worse than doing nothing under the same physical rollout.
        zero = Vec6(0.0)
        terms = residual(job, zero, candidate)
        zero_cost = wp.dot(terms, terms)
        if zero_cost < job.costs[candidate]:
            force = zero
        physics = job.physics[0]
        position, quaternion, velocity = predict(physics, force, job.points[2 * candidate], job.points[2 * candidate + 1],
                                                job.normals[2 * candidate], job.normals[2 * candidate + 1])
        terms = residual(job, force, candidate)
        cost = wp.dot(terms, terms)
        task_cost = pose_cost(physics, position, quaternion, job.pos_weight, job.ori_weight)
        if not wp.isfinite(cost):
            job.valid[candidate] = 0
            job.costs[candidate] = wp.inf
            job.pose_costs[candidate] = wp.inf
            job.reductions[candidate] = -wp.inf
            job.baseline_reductions[candidate] = -wp.inf
            return
        job.costs[candidate] = cost
        job.pose_costs[candidate] = task_cost
        job.reductions[candidate] = job.current_pose_cost[0] - task_cost
        job.baseline_reductions[candidate] = job.baseline_cost[0] - task_cost
        for contact in range(2):
            n, t1, t2 = frame(job.normals[2 * candidate + contact])
            vector = n * force[3 * contact] + t1 * force[3 * contact + 1] + t2 * force[3 * contact + 2]
            job.force_vectors_local[2 * candidate + contact] = vector
        for component in range(6):
            job.forces[candidate, component] = force[component]
            job.velocities[candidate, component] = velocity[component]
        for component in range(3):
            job.poses[candidate, component] = position[component]
            job.poses[candidate, 4 + component] = quaternion[component]
        job.poses[candidate, 3] = quaternion[3]

    @wp.kernel(enable_backward=False)
    def select(job: Job):
        best = job.selection_override[0]
        if best < 0:
            for candidate in range(job.costs.shape[0]):
                if job.valid[candidate] != 0 and wp.isfinite(job.costs[candidate]):
                    if best < 0:
                        best = candidate
                    elif job.costs[candidate] < job.costs[best]:
                        best = candidate
        elif best >= job.costs.shape[0]:
            best = -1
        elif job.valid[best] == 0 or not wp.isfinite(job.costs[best]):
            best = -1
        job.selected[0] = best
        if best < 0:
            return
        total_force = wp.vec3(0.0)
        total_torque = wp.vec3(0.0)
        for contact in range(2):
            point = job.points[2 * best + contact]
            normal = job.normals[2 * best + contact]
            n, t1, t2 = frame(normal)
            force = n * job.forces[best, 3 * contact] + t1 * job.forces[best, 3 * contact + 1]
            force += t2 * job.forces[best, 3 * contact + 2]
            job.selected_points[contact] = point
            job.selected_normals[contact] = n
            job.selected_vectors[contact] = force
            total_force += force
            total_torque += wp.cross(point, force)
        physics = job.physics[0]
        total_force = wp.quat_rotate(physics.quaternion, total_force)
        total_torque = wp.quat_rotate(physics.quaternion, total_torque)
        for component in range(3):
            job.selected_wrench[component] = total_force[component]
            job.selected_wrench[3 + component] = total_torque[component]
        for component in range(6):
            job.selected_forces[component] = job.forces[best, component]
            job.selected_velocity[component] = job.velocities[best, component]
        for component in range(7):
            job.selected_pose[component] = job.poses[best, component]
        job.selected_statistics[0] = job.costs[best]
        job.selected_statistics[1] = job.pose_costs[best]
        job.selected_statistics[2] = job.reductions[best]
        job.selected_statistics[3] = job.baseline_reductions[best]
        job.selected_statistics[4] = float(job.accepted[best])

    _KERNELS = SimpleNamespace(wp=wp, Job=Job, Physics=Physics, prepare=prepare,
                              initialize=initialize, iteration=iteration,
                              finalize=finalize, select=select, projected_force=projected_force)
    return _KERNELS


class BatchedLambdaLM:
    """Batch independent LM solves over six physical contact-force components."""

    solver_backend = "lambda_batched_lm_warp"

    def __init__(self, *, device="cuda:0", iterations=20, mu=0.9,
                 max_contact_force=10.0, mass=0.2, pos_coef=1.0, ori_coef=0.0005,
                 force_reg_coef=0.0, friction_reg_coef=0.0, damping=1.0e-3,
                 trust_radius=2.0, fd_step=1.0e-3, initial_normal_force=0.01):
        self.kernels = ensure_kernels()
        self.device = self.kernels.wp.get_device(device)
        self.iterations = int(iterations)
        if self.iterations < 1:
            raise ValueError("LM iterations must be positive")
        positive = dict(max_contact_force=max_contact_force, mass=mass, damping=damping,
                        trust_radius=trust_radius, fd_step=fd_step)
        nonnegative = dict(mu=mu, pos_coef=pos_coef, ori_coef=ori_coef,
                           force_reg_coef=force_reg_coef, friction_reg_coef=friction_reg_coef,
                           initial_normal_force=initial_normal_force)
        for name, value in {**positive, **nonnegative}.items():
            if not np.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"{name} must be finite and {'positive' if name in positive else 'nonnegative'}")
        if pos_coef + ori_coef <= 0:
            raise ValueError("at least one pose weight must be positive")
        self.mass = float(mass)
        self.parameters = dict(mu=mu, force_limit=max_contact_force, initial_normal=initial_normal_force,
                               pos_weight=pos_coef, ori_weight=ori_coef,
                               force_weight=force_reg_coef, tangent_weight=friction_reg_coef,
                               initial_damping=damping, initial_trust=trust_radius, fd_step=fd_step)
        self.buffers = None
        self.graph = None

    def reset(self):
        """Discard reusable buffers; explicit warm forces belong to the caller."""
        self.buffers = None
        self.graph = None

    def _allocate(self, count):
        wp, kernels = self.kernels.wp, self.kernels
        if self.buffers is not None and self.buffers.costs.shape[0] == count:
            return self.buffers
        self.graph = None
        job = kernels.Job()
        scalar = lambda n: wp.zeros(n, dtype=float, device=self.device)
        vector = lambda n: wp.zeros(n, dtype=wp.vec3, device=self.device)
        integer = lambda n: wp.zeros(n, dtype=int, device=self.device)
        matrix = lambda rows, cols: wp.zeros((rows, cols), dtype=float, device=self.device)
        job.points, job.normals = vector(2 * count), vector(2 * count)
        job.mask, job.initial, job.use_initial = integer(count), matrix(count, 6), integer(1)
        job.physics = wp.empty(1, dtype=kernels.Physics, device=self.device)
        job.pose, job.target, job.velocity = scalar(7), scalar(7), scalar(6)
        job.external, job.bias, job.inertia = scalar(6), scalar(6), matrix(6, 6)
        job.use_default_gravity = integer(1)
        job.support_row, job.support_gap, job.dt = scalar(6), scalar(1), scalar(1)
        job.forces, job.poses, job.velocities = matrix(count, 6), matrix(count, 7), matrix(count, 6)
        job.force_vectors_local = vector(2 * count)
        for name in ("costs", "pose_costs", "reductions", "baseline_reductions", "damping", "trust"):
            setattr(job, name, scalar(count))
        job.valid, job.accepted = integer(count), integer(count)
        job.baseline_pose, job.baseline_cost, job.current_pose_cost = scalar(7), scalar(1), scalar(1)
        job.selected = integer(1)
        job.selection_override, job.selected_statistics = integer(1), scalar(5)
        job.selected_points, job.selected_normals, job.selected_vectors = vector(2), vector(2), vector(2)
        job.selected_forces, job.selected_pose, job.selected_velocity = scalar(6), scalar(7), scalar(6)
        job.selected_wrench = scalar(6)
        for name, value in self.parameters.items():
            setattr(job, name, float(value))
        self.buffers = job
        return job

    def _copy(self, target, value, shape, name):
        wp = self.kernels.wp
        if isinstance(value, wp.array):
            if tuple(value.shape) != tuple(shape):
                raise ValueError(f"{name} must have shape {shape}")
            if value.dtype != target.dtype:
                raise ValueError(f"{name} device array must have dtype {target.dtype}")
            wp.copy(target, value)
        else:
            array = np.asarray(value, dtype=np.float32)
            if array.shape != tuple(shape) or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            target.assign(array)

    def _run(self):
        wp, kernels, job = self.kernels.wp, self.kernels, self.buffers
        wp.launch(kernels.prepare, dim=1, inputs=[job], device=self.device)
        wp.launch(kernels.initialize, dim=job.costs.shape[0], inputs=[job], device=self.device)
        for _ in range(self.iterations):
            wp.launch(kernels.iteration, dim=job.costs.shape[0], inputs=[job], device=self.device)
        wp.launch(kernels.finalize, dim=job.costs.shape[0], inputs=[job], device=self.device)
        wp.launch(kernels.select, dim=1, inputs=[job], device=self.device)

    def solve_device(self, points, normals, *, current_x, x_d, prediction_dt,
                     object_inertia, v_obj=None, tau_o=None, initial_forces=None,
                     valid_mask=None, valid=None, support_row=None, support_gap=0.0,
                     support_stiffness=0.2, support_damping=0.001, bias_wrench=None):
        """Return device buffers; accept flat Warp vec3 arrays or host N×2×3.

        ``reductions`` compares the current pose with the resulting pose, as
        Isaac's contact-quality policy does.  ``baseline_reductions`` instead
        measures the benefit over a no-contact physical prediction.  Selection
        minimizes the full exact objective; the caller owns contact switching.
        """
        wp = self.kernels.wp
        if isinstance(points, wp.array):
            if points.dtype != wp.vec3 or len(points.shape) != 1 or points.shape[0] % 2:
                raise ValueError("device points must be a flat Warp vec3 array with two points per pair")
            count = points.shape[0] // 2
        else:
            points = np.asarray(points, dtype=np.float32)
            if points.ndim != 3 or points.shape[1:] != (2, 3):
                raise ValueError("points must have shape (N, 2, 3)")
            count = points.shape[0]
            points = points.reshape(-1, 3)
        if count < 1:
            raise ValueError("at least one candidate pair is required")
        if not isinstance(normals, wp.array):
            normals = np.asarray(normals, dtype=np.float32).reshape(2 * count, 3)
        if not isinstance(object_inertia, wp.array):
            inertia = np.asarray(object_inertia, dtype=np.float32)
            if inertia.shape != (6, 6) or not np.isfinite(inertia).all() or not np.allclose(inertia, inertia.T, atol=1.e-7):
                raise ValueError("object_inertia must be finite symmetric positive definite 6x6")
        for name, value in (("prediction_dt", prediction_dt), ("support_stiffness", support_stiffness)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(current_x, wp.array) and np.linalg.norm(np.asarray(current_x)[3:7]) < 1.e-12:
            raise ValueError("current_x quaternion must be nonzero")
        if not isinstance(x_d, wp.array) and np.linalg.norm(np.asarray(x_d)[3:7]) < 1.e-12:
            raise ValueError("x_d quaternion must be nonzero")
        if valid is not None and valid_mask is not None:
            raise ValueError("specify only one of valid or valid_mask")
        job = self._allocate(count)
        self._copy(job.points, points, (2 * count, 3) if not isinstance(points, wp.array) else (2 * count,), "points")
        self._copy(job.normals, normals, (2 * count, 3) if not isinstance(normals, wp.array) else (2 * count,), "normals")
        self._copy(job.pose, current_x, (7,), "current_x")
        self._copy(job.target, x_d, (7,), "x_d")
        self._copy(job.inertia, object_inertia, (6, 6), "object_inertia")
        self._copy(job.velocity, np.zeros(6) if v_obj is None else v_obj, (6,), "v_obj")
        self._copy(job.external, np.zeros(6) if tau_o is None else tau_o, (6,), "tau_o")
        job.use_default_gravity.fill_(int(tau_o is None))
        self._copy(job.bias, np.zeros(6) if bias_wrench is None else bias_wrench, (6,), "bias_wrench")
        self._copy(job.support_row, np.zeros(6) if support_row is None else support_row, (6,), "support_row")
        gap = support_gap if isinstance(support_gap, wp.array) else np.asarray([support_gap])
        self._copy(job.support_gap, gap, (1,), "support_gap")
        self._copy(job.dt, np.asarray([prediction_dt]), (1,), "prediction_dt")
        if not np.isfinite(support_damping) or support_damping < 0:
            raise ValueError("support_damping must be finite and nonnegative")
        if job.support_stiffness != float(support_stiffness) or job.support_damping != float(support_damping):
            self.graph = None
        job.support_stiffness = float(support_stiffness)
        job.support_damping = float(support_damping)
        mask = valid if valid_mask is None else valid_mask
        if mask is None:
            job.mask.fill_(1)
        elif isinstance(mask, wp.array):
            if mask.dtype != wp.int32 or mask.shape != (count,):
                raise ValueError("valid_mask must be a Warp int array of candidate count")
            wp.copy(job.mask, mask)
        else:
            mask = np.asarray(mask, dtype=np.int32)
            if mask.shape != (count,):
                raise ValueError("valid_mask must have one entry per candidate")
            job.mask.assign(mask)
        job.use_initial.fill_(int(initial_forces is not None))
        job.selection_override.fill_(-1)
        if initial_forces is not None:
            if not isinstance(initial_forces, wp.array):
                initial_forces = np.asarray(initial_forces, np.float32).reshape(count, 6)
            self._copy(job.initial, initial_forces, (count, 6), "initial_forces")
        with wp.ScopedDevice(self.device):
            if self.device.is_cuda:
                if self.graph is None:
                    self._run()  # Load modules before capturing the fixed LM loop.
                    wp.synchronize_device(self.device)
                    with wp.ScopedCapture(device=self.device) as capture:
                        self._run()
                    self.graph = capture.graph
                wp.capture_launch(self.graph)
            else:
                self._run()
        result = SimpleNamespace(**{name: getattr(job, name) for name in (
            "forces", "poses", "velocities", "costs", "pose_costs", "reductions",
            "baseline_reductions", "valid", "accepted", "damping", "trust",
            "force_vectors_local",
            "baseline_pose", "baseline_cost", "current_pose_cost", "selected",
            "selected_points", "selected_normals", "selected_forces", "selected_vectors",
            "selected_pose", "selected_velocity", "selected_wrench", "selected_statistics", "physics")})
        result.x_plus, result.velocity_next = result.poses, result.velocities
        result.baseline_pose_cost = result.baseline_cost
        return result

    def solve(self, *args, **kwargs):
        """Read the selected result at the host edge; retain all batch arrays."""
        started = perf_counter()
        batch = self.solve_device(*args, **kwargs)
        result = self.gather_selected(batch=batch)
        result["solve_time"] = perf_counter() - started
        return result

    def gather_selected(self, selected_index=None, *, batch=None):
        """Gather any valid row chosen by the caller's contact-switch policy."""
        if self.buffers is None:
            raise RuntimeError("solve a candidate batch before gathering a row")
        job, wp = self.buffers, self.kernels.wp
        if batch is None:
            batch = SimpleNamespace(**{name: getattr(job, name) for name in (
                "selected", "selected_points", "selected_normals", "selected_forces",
                "selected_vectors", "selected_pose", "selected_velocity", "selected_wrench",
                "selected_statistics", "baseline_pose", "baseline_cost", "current_pose_cost", "physics")})
        if selected_index is not None:
            if not 0 <= int(selected_index) < job.costs.shape[0]:
                raise ValueError("selected_index must index the current candidate batch")
            job.selection_override.fill_(int(selected_index))
            wp.launch(self.kernels.select, dim=1, inputs=[job], device=self.device)
        self.kernels.wp.synchronize_device(self.device)
        selected = int(batch.selected.numpy()[0])
        if selected < 0:
            if int(batch.physics.numpy()["valid"][0]) == 0:
                raise ValueError("lambda physics requires finite inputs, nonzero quaternions and positive definite inertia")
            raise FloatingPointError("all lambda contact candidates are invalid")
        wrench = batch.selected_wrench.numpy().copy()
        statistics = batch.selected_statistics.numpy().copy()
        return dict(selected_index=selected, points=batch.selected_points.numpy().copy(),
                    normals=batch.selected_normals.numpy().copy(),
                    control_input=batch.selected_forces.numpy().reshape(2, 3).copy(),
                    force_vectors_local=batch.selected_vectors.numpy().copy(),
                    contact_force_world=wrench[:3], contact_torque_world=wrench[3:],
                    x_plus_opt=batch.selected_pose.numpy().copy(),
                    velocity_next=batch.selected_velocity.numpy().copy(),
                    cost=float(statistics[0]), pose_cost=float(statistics[1]),
                    pose_reduction=float(statistics[2]), baseline_reduction=float(statistics[3]),
                    baseline_pose=batch.baseline_pose.numpy().copy(),
                    baseline_cost=float(batch.baseline_cost.numpy()[0]),
                    current_pose_cost=float(batch.current_pose_cost.numpy()[0]),
                    solver_backend=self.solver_backend, solver_failed=False,
                    solver_converged=bool(statistics[4] > 0), batch=batch)
