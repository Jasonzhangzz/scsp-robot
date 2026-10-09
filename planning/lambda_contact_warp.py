"""Warp CUDA-graph Adam contact selector on an object Gaussian cloud."""

from types import SimpleNamespace

import numpy as np

from planning.mpc_explicit_adam import LambdaContactAdamOptimizer
from planning.warp_adam import ensure_kernels as ensure_adam, ensure_warp, launch_adam


_KERNELS = None


def pca_opposite_seeds(points):
    """Return the two samples at opposite ends of the longest PCA axis."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 2:
        raise RuntimeError("Need at least two Gaussian samples to seed contacts")
    mean = points.mean(axis=0)
    centered = points - mean
    cov = centered.T @ centered
    _, vecs = np.linalg.eigh(cov)
    axis = vecs[:, -1]
    proj = centered @ axis
    return points[int(np.argmin(proj))].copy(), points[int(np.argmax(proj))].copy()


def ensure_kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    wp = ensure_warp()

    @wp.func
    def _softplus(x: float) -> float:
        ax = wp.abs(x)
        return wp.max(x, 0.0) + wp.log(1.0 + wp.exp(-ax))

    @wp.func
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + wp.exp(-x))

    @wp.struct
    class GsJob:
        points: wp.array(dtype=wp.vec3)
        spheres: wp.array(dtype=wp.vec4)
        phi: wp.array(dtype=float)
        position: wp.array(dtype=wp.vec3)
        inward: wp.array(dtype=wp.vec3)
        query_radius: float
        tau: float
        sphere_count: int

    @wp.kernel(enable_backward=False)
    def gs_features(job: GsJob):
        tip = wp.tid()
        query = job.points[tip]
        tau = wp.max(job.tau, 1.0e-6)
        min_d = float(1.0e6)
        for index in range(job.sphere_count):
            sphere = job.spheres[index]
            center = wp.vec3(sphere[0], sphere[1], sphere[2])
            delta = center - query
            pair = wp.length(delta)
            distance = pair - sphere[3] - job.query_radius
            min_d = wp.min(min_d, distance)
        weight_sum = float(0.0)
        phi = float(0.0)
        position = wp.vec3(0.0, 0.0, 0.0)
        normal = wp.vec3(0.0, 0.0, 0.0)
        for index in range(job.sphere_count):
            sphere = job.spheres[index]
            center = wp.vec3(sphere[0], sphere[1], sphere[2])
            delta = center - query
            pair = wp.max(wp.length(delta), 1.0e-8)
            distance = pair - sphere[3] - job.query_radius
            weight = wp.exp(-(distance - min_d) / tau)
            inward = delta / pair
            surface = center - inward * sphere[3]
            weight_sum += weight
            phi += weight * distance
            position += weight * surface
            normal += weight * inward
        inv = 1.0 / wp.max(weight_sum, 1.0e-12)
        phi = phi * inv
        position = position * inv
        normal = wp.normalize(normal * inv)
        job.phi[tip] = phi
        job.position[tip] = position
        job.inward[tip] = normal

    @wp.struct
    class LambdaLossJob:
        phi: wp.array(dtype=float)
        position: wp.array(dtype=wp.vec3)
        inward: wp.array(dtype=wp.vec3)
        force: wp.array(dtype=wp.vec3)
        loss: wp.array(dtype=float)
        g_phi: wp.array(dtype=float)
        g_position: wp.array(dtype=wp.vec3)
        g_inward: wp.array(dtype=wp.vec3)
        gravity: wp.vec3
        stiffness: float
        min_pair: float

    @wp.kernel(enable_backward=False)
    def lambda_loss(job: LambdaLossJob):
        phi0 = job.phi[0]
        phi1 = job.phi[1]
        pos0 = job.position[0]
        pos1 = job.position[1]
        n0 = job.inward[0]
        n1 = job.inward[1]
        s0 = _softplus(-phi0)
        s1 = _softplus(-phi1)
        f0 = -n0 * (job.stiffness * s0)
        f1 = -n1 * (job.stiffness * s1)
        job.force[0] = f0
        job.force[1] = f1
        surface = phi0 * phi0 + phi1 * phi1
        antipodal = wp.length_sq(n0 + n1)
        pair = wp.length(pos0 - pos1)
        sep = _softplus(job.min_pair - pair)
        net = f0 + f1 - job.gravity
        torque = wp.cross(pos0, f0) + wp.cross(pos1, f1)
        loss = 25.0 * surface + 8.0 * antipodal + 40.0 * sep + 0.4 * wp.length_sq(net) + 0.1 * wp.length_sq(torque)
        job.loss[0] = loss

        d_net = 0.8 * net
        d_torque = 0.2 * torque
        d_f0 = d_net + wp.cross(pos0, d_torque)
        d_f1 = d_net + wp.cross(pos1, d_torque)
        d_pos0 = wp.cross(d_torque, f0)
        d_pos1 = wp.cross(d_torque, f1)
        d_pair = -40.0 * _sigmoid(job.min_pair - pair)
        axis = (pos0 - pos1) / wp.max(pair, 1.0e-8)
        d_pos0 += d_pair * axis
        d_pos1 -= d_pair * axis
        d_n0 = 16.0 * (n0 + n1)
        d_n1 = 16.0 * (n0 + n1)
        d_n0 += d_f0 * (-job.stiffness * s0)
        d_n1 += d_f1 * (-job.stiffness * s1)
        ds0 = -_sigmoid(-phi0)
        ds1 = -_sigmoid(-phi1)
        d_phi0 = 50.0 * phi0 + wp.dot(d_f0, -n0 * (job.stiffness * ds0))
        d_phi1 = 50.0 * phi1 + wp.dot(d_f1, -n1 * (job.stiffness * ds1))
        job.g_phi[0] = d_phi0
        job.g_phi[1] = d_phi1
        job.g_position[0] = d_pos0
        job.g_position[1] = d_pos1
        job.g_inward[0] = d_n0
        job.g_inward[1] = d_n1

    @wp.struct
    class GsVjpJob:
        points: wp.array(dtype=wp.vec3)
        spheres: wp.array(dtype=wp.vec4)
        phi: wp.array(dtype=float)
        inward: wp.array(dtype=wp.vec3)
        g_phi: wp.array(dtype=float)
        g_position: wp.array(dtype=wp.vec3)
        g_inward: wp.array(dtype=wp.vec3)
        gradient: wp.array2d(dtype=float)
        query_radius: float
        tau: float
        sphere_count: int

    @wp.kernel(enable_backward=False)
    def gs_features_vjp(job: GsVjpJob):
        tip = wp.tid()
        query = job.points[tip]
        tau = wp.max(job.tau, 1.0e-6)
        min_d = float(1.0e6)
        for index in range(job.sphere_count):
            sphere = job.spheres[index]
            center = wp.vec3(sphere[0], sphere[1], sphere[2])
            pair = wp.length(center - query)
            min_d = wp.min(min_d, pair - sphere[3] - job.query_radius)
        weight_sum = float(0.0)
        n_soft = wp.vec3(0.0, 0.0, 0.0)
        for index in range(job.sphere_count):
            sphere = job.spheres[index]
            center = wp.vec3(sphere[0], sphere[1], sphere[2])
            delta = center - query
            pair = wp.max(wp.length(delta), 1.0e-8)
            distance = pair - sphere[3] - job.query_radius
            weight = wp.exp(-(distance - min_d) / tau)
            weight_sum += weight
            n_soft += weight * (delta / pair)
        inv = 1.0 / wp.max(weight_sum, 1.0e-12)
        n_soft = n_soft * inv
        phi = job.phi[tip]
        fused = job.inward[tip]
        g_phi = job.g_phi[tip]
        g_pos = job.g_position[tip]
        g_n = job.g_inward[tip]
        nlen = wp.max(wp.length(n_soft), 1.0e-8)
        g_unnorm = (g_n - fused * wp.dot(g_n, fused)) / nlen
        eye = wp.identity(n=3, dtype=float)
        grad = wp.vec3(0.0, 0.0, 0.0)
        for index in range(job.sphere_count):
            sphere = job.spheres[index]
            center = wp.vec3(sphere[0], sphere[1], sphere[2])
            delta = center - query
            pair = wp.max(wp.length(delta), 1.0e-8)
            inward = delta / pair
            distance = pair - sphere[3] - job.query_radius
            weight = wp.exp(-(distance - min_d) / tau) * inv
            surface = center - inward * sphere[3]
            d_phi_dd = weight * (1.0 - (distance - phi) / tau)
            dw_dq = (weight / tau) * (inward - n_soft)
            dinward_dq = -(eye / pair) + wp.outer(inward, inward) / pair
            grad += -inward * (g_phi * d_phi_dd)
            grad += dw_dq * wp.dot(g_pos, surface)
            grad += dinward_dq * (weight * g_pos * (-sphere[3]))
            grad += dw_dq * wp.dot(g_unnorm, inward)
            grad += dinward_dq * (weight * g_unnorm)
        job.gradient[0, 3 * tip + 0] = grad[0]
        job.gradient[0, 3 * tip + 1] = grad[1]
        job.gradient[0, 3 * tip + 2] = grad[2]

    @wp.kernel(enable_backward=False)
    def pack_points(raw: wp.array2d(dtype=float), points: wp.array(dtype=wp.vec3)):
        tip = wp.tid()
        points[tip] = wp.vec3(raw[0, 3 * tip + 0], raw[0, 3 * tip + 1], raw[0, 3 * tip + 2])

    @wp.kernel(enable_backward=False)
    def pack_points_flat(raw: wp.array2d(dtype=float), points: wp.array(dtype=wp.vec3)):
        sample, tip = wp.tid()
        base = sample * 2 + tip
        points[base] = wp.vec3(raw[sample, 3 * tip + 0], raw[sample, 3 * tip + 1], raw[sample, 3 * tip + 2])

    @wp.kernel(enable_backward=False)
    def lambda_loss_batch(job: LambdaLossJob):
        sample = wp.tid()
        i0 = sample * 2
        i1 = i0 + 1
        phi0 = job.phi[i0]
        phi1 = job.phi[i1]
        pos0 = job.position[i0]
        pos1 = job.position[i1]
        n0 = job.inward[i0]
        n1 = job.inward[i1]
        s0 = _softplus(-phi0)
        s1 = _softplus(-phi1)
        f0 = -n0 * (job.stiffness * s0)
        f1 = -n1 * (job.stiffness * s1)
        job.force[i0] = f0
        job.force[i1] = f1
        surface = phi0 * phi0 + phi1 * phi1
        antipodal = wp.length_sq(n0 + n1)
        pair = wp.length(pos0 - pos1)
        sep = _softplus(job.min_pair - pair)
        net = f0 + f1 - job.gravity
        torque = wp.cross(pos0, f0) + wp.cross(pos1, f1)
        job.loss[sample] = (
            25.0 * surface + 8.0 * antipodal + 40.0 * sep + 0.4 * wp.length_sq(net) + 0.1 * wp.length_sq(torque)
        )

    _KERNELS = SimpleNamespace(
        wp=wp,
        GsJob=GsJob,
        LambdaLossJob=LambdaLossJob,
        GsVjpJob=GsVjpJob,
        gs_features=gs_features,
        lambda_loss=lambda_loss,
        lambda_loss_batch=lambda_loss_batch,
        gs_features_vjp=gs_features_vjp,
        pack_points=pack_points,
        pack_points_flat=pack_points_flat,
    )
    return _KERNELS


class LambdaContactWarp(LambdaContactAdamOptimizer):
    """Once-per-run Warp Adam contact picker.  Host I/O only at the edges."""

    def __init__(self, device=None, **kwargs):
        requested = None if device is None else str(device)
        self.pos_coef = float(kwargs.get("pos_coef", 1.0))
        self.ori_coef = float(kwargs.get("ori_coef", 0.0005))
        self.max_contact_force = float(kwargs.get("max_contact_force", 10.0))
        self.comfree_damping = float(kwargs.pop("comfree_damping", 0.001))
        self.force_reg_coef = float(kwargs.get("force_reg_coef", 0.0))
        self.friction_reg_coef = float(kwargs.get("friction_reg_coef", 0.0))
        super().__init__(device="cpu", **kwargs)
        self.warp_device = self._resolve_device(requested)
        self._graph = None
        self._buffers = None
        self._captured_n = None
        self._control_force_warm = None

    def optimize_control_input(
        self, x_d, current_x, tau_o=None, p_arm=None, v_obj=None,
        r_obj_to_world=None, return_world_torque=False, *,
        prediction_dt=None, object_inertia=None,
    ):
        """Predict one pose from bounded forces at the selected lambda pair.

        Poses are world position followed by a WXYZ quaternion. ``p_arm`` is
        in the object frame; ``v_obj`` uses MuJoCo free-joint coordinates
        (world linear velocity, body angular velocity). ``tau_o`` is a world
        force/torque wrench; when omitted it supplies gravity. Contact forces
        are physical forces, converted to impulses over ``prediction_dt``.
        This solve leaves the Warp contact selector and its pair unchanged.
        """
        import time
        from scipy.optimize import Bounds, LinearConstraint, minimize
        from scipy.spatial.transform import Rotation
        from models.comfree_gs_torch import gs_sphere_features

        started = time.perf_counter()
        pose = np.asarray(current_x, dtype=np.float64).reshape(7).copy()
        target = np.asarray(x_d, dtype=np.float64).reshape(7).copy()
        dt = float(self.h if prediction_dt is None else prediction_dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("prediction_dt must be finite and positive")
        if not np.isfinite(pose).all() or not np.isfinite(target).all():
            raise ValueError("lambda poses must be finite")
        for value in (pose, target):
            qnorm = float(np.linalg.norm(value[3:7]))
            if qnorm <= 1.0e-12:
                raise ValueError("lambda pose quaternion must be nonzero")
            value[3:7] /= qnorm
        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        if r_obj_to_world is not None:
            supplied = np.asarray(r_obj_to_world, dtype=np.float64).reshape(3, 3)
            if not np.allclose(supplied, rotation, atol=1.0e-5):
                raise ValueError("r_obj_to_world must match current_x quaternion")
        grasp = self.last_grasp_result or {}
        if p_arm is None:
            p_arm = grasp.get("contact_points")
            if p_arm is None:
                raise ValueError("select lambda contacts or provide p_arm before predicting a pose")
        points = np.asarray(p_arm, dtype=np.float64).reshape(-1, 3)
        if points.shape != (2, 3) or not np.isfinite(points).all():
            raise ValueError("lambda pose prediction requires two finite object-frame contacts")
        selected = grasp.get("contact_points")
        selected_normals = grasp.get("contact_normals")
        if selected is not None and selected_normals is not None:
            selected = np.asarray(selected, dtype=np.float64).reshape(-1, 3)
            order = np.linalg.norm(points[:, None] - selected[None], axis=-1).argmin(axis=1)
            normals = np.asarray(selected_normals, dtype=np.float64).reshape(-1, 3)[order].copy()
        else:
            import torch
            _, _, inward = gs_sphere_features(
                torch.as_tensor(points, dtype=torch.float32), self.spheres,
                query_radius=0.0, tau=self.gs_tau,
            )
            normals = inward.detach().cpu().numpy().astype(np.float64)
        normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
        if np.any(normal_norms <= 1.0e-8):
            raise ValueError("lambda contacts must have nonzero inward normals")
        normals /= normal_norms
        helper = np.tile([1.0, 0.0, 0.0], (2, 1))
        helper[np.abs(normals[:, 0]) > 0.9] = [0.0, 1.0, 0.0]
        tangent1 = np.cross(normals, helper)
        tangent1 /= np.linalg.norm(tangent1, axis=1, keepdims=True)
        frames = np.stack((normals, tangent1, np.cross(normals, tangent1)), axis=-1)
        # Force translation is world-frame, while torque/omega are body-frame.
        grasp_map = np.zeros((6, 6), dtype=np.float64)
        for index in range(2):
            grasp_map[:3, 3 * index:3 * index + 3] = rotation @ frames[index]
            grasp_map[3:, 3 * index:3 * index + 3] = np.cross(
                points[index], frames[index].T,
            ).T
        if object_inertia is None:
            centered = self.cloud.centers.astype(np.float64)
            covariance = centered.T @ centered / len(centered)
            moment = self.m * (np.trace(covariance) * np.eye(3) - covariance)
            moment += self.m * 0.4 * float(np.mean(self.cloud.radii ** 2)) * np.eye(3)
            inertia = np.zeros((6, 6), dtype=np.float64)
            inertia[:3, :3] = max(self.m, 1.0e-8) * np.eye(3)
            inertia[3:, 3:] = moment + 1.0e-8 * np.eye(3)
        else:
            inertia = np.asarray(object_inertia, dtype=np.float64).reshape(6, 6)
        if (not np.isfinite(inertia).all() or
                not np.allclose(inertia, inertia.T, atol=1.0e-10) or
                np.min(np.linalg.eigvalsh(inertia)) <= 0.0):
            raise ValueError("object_inertia must be finite, symmetric and positive definite")
        inverse = np.linalg.inv(inertia)
        velocity = np.zeros(6) if v_obj is None else np.asarray(v_obj, dtype=np.float64).reshape(6)
        external = np.array([0.0, 0.0, -self.m * 9.81, 0.0, 0.0, 0.0]) if tau_o is None else np.asarray(tau_o, dtype=np.float64).reshape(6).copy()
        if not np.isfinite(velocity).all() or not np.isfinite(external).all():
            raise ValueError("lambda velocity and external wrench must be finite")
        external[3:] = rotation.T @ external[3:]
        free_velocity = velocity + dt * (inverse @ external)
        force_velocity = dt * inverse @ grasp_map
        support = None
        if self.support_surface_point is not None and self.support_surface_normal is not None:
            support_normal = np.asarray(self.support_surface_normal, dtype=np.float64).reshape(3)
            support_normal /= max(float(np.linalg.norm(support_normal)), 1.0e-12)
            centers_world = self.cloud.centers @ rotation.T + pose[:3]
            gaps = (centers_world - self.support_surface_point) @ support_normal - self.cloud.radii
            index = int(np.argmin(gaps))
            local_normal = rotation.T @ support_normal
            row = np.r_[support_normal, np.cross(self.cloud.centers[index], local_normal)]
            from models.comfree_contact import row_mass
            weight = np.trace(inverse[:3, :3])/3
            mass = float(row_mass(weight, gaps[index]))
            support = (row, float(gaps[index]), mass)

        def predict_velocity(forces):
            value = free_velocity + force_velocity @ forces
            derivative = force_velocity
            if support is not None:
                row, gap, mass = support
                speed = float(row @ value)
                acceleration = -(self.comfree_damping/dt)*speed - (self.K_contact/dt)*(dt*speed+gap)
                force = mass*acceleration
                if gap < 0.0 and force > 0.0:
                    response = dt * inverse @ row
                    value = value + response*force
                    derivative = force_velocity - np.outer(response, mass*(self.comfree_damping/dt+self.K_contact)*(row @ force_velocity))
            return value, derivative

        target_rotation = Rotation.from_quat(target[[4, 5, 6, 3]])
        rotation_delta = (Rotation.from_matrix(rotation).inv() * target_rotation).as_rotvec()
        desired_delta = np.r_[target[:3] - pose[:3], rotation_delta]
        task_weights = np.r_[np.full(3, np.sqrt(max(self.pos_coef, 0.0))),
                             np.full(3, 0.5 * np.sqrt(max(self.ori_coef, 0.0)))]
        scale = max(float(np.linalg.norm(task_weights * desired_delta)),
                    float(np.linalg.norm(task_weights * dt * free_velocity)), 1.0e-4)
        regularization = np.tile([max(self.force_reg_coef, 0.0),
                                  max(self.force_reg_coef + self.friction_reg_coef, 0.0),
                                  max(self.force_reg_coef + self.friction_reg_coef, 0.0)], 2)

        def objective(forces):
            value, derivative = predict_velocity(forces)
            residual = task_weights * (dt * value - desired_delta)
            jacobian = task_weights[:, None] * dt * derivative
            cost = (residual @ residual + (regularization * forces) @ forces) / (scale * scale)
            gradient = 2.0 * (jacobian.T @ residual + regularization * forces) / (scale * scale)
            return float(cost), gradient

        mu = float(self.mu_arm_obj)
        limit = float(self.max_contact_force)
        if not np.isfinite(mu) or mu < 0.0 or not np.isfinite(limit) or limit <= 0.0:
            raise ValueError("lambda friction must be nonnegative and force limit positive")
        normal_limit = limit / np.sqrt(1.0 + 2.0 * mu * mu)
        lower = np.tile([0.0, -mu * normal_limit, -mu * normal_limit], 2)
        upper = np.tile([normal_limit, mu * normal_limit, mu * normal_limit], 2)
        cone = np.zeros((8, 6))
        for index in range(2):
            for tangent in range(2):
                cone[4 * index + 2 * tangent:4 * index + 2 * tangent + 2, 3 * index] = mu
                cone[4 * index + 2 * tangent, 3 * index + tangent + 1] = 1.0
                cone[4 * index + 2 * tangent + 1, 3 * index + tangent + 1] = -1.0
        initial = np.zeros(6) if self._control_force_warm is None else self._control_force_warm.copy()
        initial = np.clip(initial, lower, upper).reshape(2, 3)
        initial[:, 1:] = np.clip(initial[:, 1:], -mu * initial[:, :1], mu * initial[:, :1])
        initial = initial.reshape(6)
        solution = minimize(objective, initial, jac=True, method="SLSQP",
                            bounds=Bounds(lower, upper),
                            constraints=LinearConstraint(cone, 0.0, np.inf),
                            options={"maxiter": 80, "ftol": 1.0e-10})
        candidates = [np.zeros(6), initial]
        if np.isfinite(solution.x).all():
            candidate = np.clip(solution.x, lower, upper).reshape(2, 3)
            candidate[:, 1:] = np.clip(candidate[:, 1:], -mu * candidate[:, :1], mu * candidate[:, :1])
            candidates.append(candidate.reshape(6))
        forces = min(candidates, key=lambda value: objective(value)[0]).copy()
        self._control_force_warm = forces.copy()
        next_velocity = predict_velocity(forces)[0]
        w, x, y, z = pose[3:7]
        quaternion_map = np.array([[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]])
        next_quaternion = pose[3:7] + 0.5 * dt * quaternion_map @ next_velocity[3:]
        next_quaternion /= np.linalg.norm(next_quaternion)
        x_plus = np.r_[pose[:3] + dt * next_velocity[:3], next_quaternion]
        coefficients = forces.reshape(2, 3)
        vectors_local = np.einsum("nij,nj->ni", frames, coefficients)
        force_local = vectors_local.sum(axis=0)
        torque_local = np.cross(points, vectors_local).sum(axis=0)
        cost = (self.pos_coef * np.sum((x_plus[:3] - target[:3]) ** 2)
                + self.ori_coef * max(1.0 - float(x_plus[3:7] @ target[3:7]) ** 2, 0.0)
                + float((regularization * forces) @ forces))
        info = {
            "solver_failed": not np.isfinite(x_plus).all(),
            "solver_converged": bool(solution.success),
            "solver_status": str(solution.message),
            "solver_backend": "lambda_contact_force_slsqp",
            "solve_time": float(time.perf_counter() - started),
            "prediction_dt": dt,
            "query_contact_points_local": points.copy(),
            "contact_normals": normals.copy(),
            "control_input": coefficients.copy(),
            "contact_forces_local": coefficients.copy(),
            "force_vectors_local": vectors_local.copy(),
            "contact_force_local": force_local.copy(),
            "contact_torque_local": torque_local.copy(),
            "contact_force_world": rotation @ force_local,
            "contact_torque_world": rotation @ torque_local,
            "velocity_next": next_velocity.copy(),
            "resulting_pose": x_plus.copy(),
            "x_plus_opt": x_plus.copy(),
        }
        self.last_grasp_result = dict(grasp, contact_points=points.copy(),
                                      contact_normals=normals.copy(),
                                      x_plus_opt=x_plus.copy(),
                                      witness_contact_forces_local=coefficients.copy(),
                                      witness_force_vectors_local=vectors_local.copy())
        result = (points.copy(), normals.copy(), x_plus, float(cost), info)
        if return_world_torque:
            return (*result, info["contact_torque_world"].copy())
        return result

    def _resolve_device(self, requested):
        wp = ensure_warp()
        devices = [str(item) for item in wp.get_devices()]
        if requested and requested in devices:
            return requested
        if requested and requested.startswith("cuda") and "cuda:0" in devices:
            return "cuda:0"
        if "cuda:0" in devices:
            return "cuda:0"
        return "cpu"

    def _seed_pair(self, visible_idx):
        return pca_opposite_seeds(self.sample_point[visible_idx])

    def _ensure_buffers(self):
        if self._buffers is not None:
            return self._buffers
        wp = ensure_warp()
        k = ensure_kernels()
        ensure_adam()
        device = self.warp_device
        n = int(self.spheres.shape[0])
        spheres = np.asarray(self.cloud.as_spheres(), dtype=np.float32)
        buffers = SimpleNamespace(
            raw=wp.zeros((1, 6), dtype=float, device=device),
            gradient=wp.zeros((1, 6), dtype=float, device=device),
            first=wp.zeros((1, 6), dtype=float, device=device),
            second=wp.zeros((1, 6), dtype=float, device=device),
            grad_norm=wp.zeros(1, dtype=float, device=device),
            valid=wp.zeros(1, dtype=int, device=device),
            iteration=wp.zeros(1, dtype=int, device=device),
            best_raw=wp.zeros((1, 6), dtype=float, device=device),
            best_loss=wp.zeros(1, dtype=float, device=device),
            best_available=wp.zeros(1, dtype=int, device=device),
            points=wp.zeros(2, dtype=wp.vec3, device=device),
            spheres=wp.array(spheres, dtype=wp.vec4, device=device),
            phi=wp.zeros(2, dtype=float, device=device),
            position=wp.zeros(2, dtype=wp.vec3, device=device),
            inward=wp.zeros(2, dtype=wp.vec3, device=device),
            force=wp.zeros(2, dtype=wp.vec3, device=device),
            loss=wp.zeros(1, dtype=float, device=device),
            g_phi=wp.zeros(2, dtype=float, device=device),
            g_position=wp.zeros(2, dtype=wp.vec3, device=device),
            g_inward=wp.zeros(2, dtype=wp.vec3, device=device),
            sphere_count=n,
        )
        buffers.valid.fill_(1)
        self._buffers = buffers
        self._captured_n = n
        k  # keep imported
        return buffers

    def _iteration(self):
        k = ensure_kernels()
        b = self._buffers
        wp = k.wp
        wp.launch(k.pack_points, dim=2, inputs=[b.raw, b.points])
        gs = k.GsJob()
        gs.points = b.points
        gs.spheres = b.spheres
        gs.phi = b.phi
        gs.position = b.position
        gs.inward = b.inward
        gs.query_radius = float(self.tip_radius)
        gs.tau = float(self.gs_tau)
        gs.sphere_count = int(b.sphere_count)
        wp.launch(k.gs_features, dim=2, inputs=[gs])
        loss = k.LambdaLossJob()
        loss.phi = b.phi
        loss.position = b.position
        loss.inward = b.inward
        loss.force = b.force
        loss.loss = b.loss
        loss.g_phi = b.g_phi
        loss.g_position = b.g_position
        loss.g_inward = b.g_inward
        loss.gravity = wp.vec3(0.0, 0.0, -float(self.m) * 9.81)
        loss.stiffness = float(self.K_contact)
        loss.min_pair = float(self.min_pair_distance)
        wp.launch(k.lambda_loss, dim=1, inputs=[loss])
        vjp = k.GsVjpJob()
        vjp.points = b.points
        vjp.spheres = b.spheres
        vjp.phi = b.phi
        vjp.inward = b.inward
        vjp.g_phi = b.g_phi
        vjp.g_position = b.g_position
        vjp.g_inward = b.g_inward
        vjp.gradient = b.gradient
        vjp.query_radius = float(self.tip_radius)
        vjp.tau = float(self.gs_tau)
        vjp.sphere_count = int(b.sphere_count)
        wp.launch(k.gs_features_vjp, dim=2, inputs=[vjp])
        adam = ensure_adam()
        wp.launch(
            adam.copy_if_better,
            dim=b.raw.shape,
            inputs=[b.loss, b.raw, b.best_loss, b.best_raw, b.best_available],
        )
        launch_adam(
            b.raw,
            b.gradient,
            b.first,
            b.second,
            b.grad_norm,
            b.valid,
            b.iteration,
            iterations=self.adam_steps,
            learning_rate=self.adam_lr,
        )

    def _ensure_graph(self):
        if self._graph is not None or not str(self.warp_device).startswith("cuda"):
            return
        wp = ensure_warp()
        self._iteration()
        wp.synchronize()
        self._buffers.iteration.zero_()
        with wp.ScopedCapture(device=self.warp_device) as capture:
            self._iteration()
        self._graph = capture.graph

    def choose_contact_set(
        self,
        visible_face_idx=None,
        object_pos=None,
        object_rot=None,
        **_unused,
    ):
        del object_pos, object_rot
        visible = self.get_contact_candidate_indices(visible_face_idx)
        seed_a, seed_b = self._seed_pair(visible)
        raw = np.concatenate((seed_a, seed_b)).astype(np.float32).reshape(1, 6)
        b = self._ensure_buffers()
        b.raw.assign(raw)
        b.best_raw.assign(raw)
        b.best_loss.fill_(1.0e30)
        b.best_available.zero_()
        b.first.zero_()
        b.second.zero_()
        b.iteration.zero_()
        b.gradient.zero_()
        wp = ensure_warp()
        self._ensure_graph()
        steps = max(int(self.adam_steps), 1) + 1
        if self._graph is not None:
            for _ in range(steps):
                wp.capture_launch(self._graph)
        else:
            for _ in range(steps):
                self._iteration()
        wp.synchronize()
        adam = ensure_adam()
        wp.launch(adam.restore_best, dim=b.raw.shape, inputs=[b.raw, b.best_raw, b.best_available])
        wp.launch(ensure_kernels().pack_points, dim=2, inputs=[b.raw, b.points])
        gs = ensure_kernels().GsJob()
        gs.points = b.points
        gs.spheres = b.spheres
        gs.phi = b.phi
        gs.position = b.position
        gs.inward = b.inward
        gs.query_radius = float(self.tip_radius)
        gs.tau = float(self.gs_tau)
        gs.sphere_count = int(b.sphere_count)
        wp.launch(ensure_kernels().gs_features, dim=2, inputs=[gs])
        loss = ensure_kernels().LambdaLossJob()
        loss.phi = b.phi
        loss.position = b.position
        loss.inward = b.inward
        loss.force = b.force
        loss.loss = b.loss
        loss.g_phi = b.g_phi
        loss.g_position = b.g_position
        loss.g_inward = b.g_inward
        loss.gravity = wp.vec3(0.0, 0.0, -float(self.m) * 9.81)
        loss.stiffness = float(self.K_contact)
        loss.min_pair = float(self.min_pair_distance)
        wp.launch(ensure_kernels().lambda_loss, dim=1, inputs=[loss])
        wp.synchronize()
        position = np.asarray(b.position.numpy(), dtype=np.float64).reshape(2, 3)
        inward = np.asarray(b.inward.numpy(), dtype=np.float64).reshape(2, 3)
        force = np.asarray(b.force.numpy(), dtype=np.float64).reshape(2, 3)
        phi = np.asarray(b.phi.numpy(), dtype=np.float64).reshape(2)
        antipodal_margin = float(-np.dot(inward[0], inward[1]))
        self.last_grasp_result = {
            "contact_indices": np.array([0, 1], dtype=int),
            "contact_points": position.copy(),
            "contact_normals": inward.copy(),
            "witness_contact_forces_local": force.copy(),
            "witness_force_vectors_local": force.copy(),
            "phi": phi,
        }
        return position, inward, float(np.sum(phi ** 2)), 0.0, antipodal_margin


class LambdaContactMPPI(LambdaContactWarp):
    """Once-per-run Warp MPPI contact picker on the same GS features as Adam."""

    def __init__(
        self,
        device=None,
        num_samples=32,
        iterations=4,
        noise=0.008,
        temperature=0.1,
        seed=None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.mppi_samples = max(int(num_samples), 2)
        self.mppi_iterations = max(int(iterations), 1)
        self.mppi_noise = max(float(noise), 1.0e-4)
        self.mppi_temperature = max(float(temperature), 1.0e-4)
        self._mppi_rng = np.random.default_rng(seed)
        self._mppi_buffers = None

    def _ensure_mppi_buffers(self):
        if self._mppi_buffers is not None:
            return self._mppi_buffers
        wp = ensure_warp()
        k = ensure_kernels()
        device = self.warp_device
        samples = int(self.mppi_samples)
        spheres = np.asarray(self.cloud.as_spheres(), dtype=np.float32)
        buffers = SimpleNamespace(
            raw=wp.zeros((samples, 6), dtype=float, device=device),
            points=wp.zeros(samples * 2, dtype=wp.vec3, device=device),
            spheres=wp.array(spheres, dtype=wp.vec4, device=device),
            phi=wp.zeros(samples * 2, dtype=float, device=device),
            position=wp.zeros(samples * 2, dtype=wp.vec3, device=device),
            inward=wp.zeros(samples * 2, dtype=wp.vec3, device=device),
            force=wp.zeros(samples * 2, dtype=wp.vec3, device=device),
            loss=wp.zeros(samples, dtype=float, device=device),
            g_phi=wp.zeros(samples * 2, dtype=float, device=device),
            g_position=wp.zeros(samples * 2, dtype=wp.vec3, device=device),
            g_inward=wp.zeros(samples * 2, dtype=wp.vec3, device=device),
            sphere_count=int(spheres.shape[0]),
        )
        self._mppi_buffers = buffers
        k  # keep imported
        return buffers

    def _score_samples(self, samples):
        import torch

        from planning.spider_mppi import compute_spider_weights

        k = ensure_kernels()
        wp = k.wp
        b = self._ensure_mppi_buffers()
        samples = np.asarray(samples, dtype=np.float32).reshape(self.mppi_samples, 6)
        b.raw.assign(samples)
        wp.launch(k.pack_points_flat, dim=(self.mppi_samples, 2), inputs=[b.raw, b.points])
        gs = k.GsJob()
        gs.points = b.points
        gs.spheres = b.spheres
        gs.phi = b.phi
        gs.position = b.position
        gs.inward = b.inward
        gs.query_radius = float(self.tip_radius)
        gs.tau = float(self.gs_tau)
        gs.sphere_count = int(b.sphere_count)
        wp.launch(k.gs_features, dim=self.mppi_samples * 2, inputs=[gs])
        loss = k.LambdaLossJob()
        loss.phi = b.phi
        loss.position = b.position
        loss.inward = b.inward
        loss.force = b.force
        loss.loss = b.loss
        loss.g_phi = b.g_phi
        loss.g_position = b.g_position
        loss.g_inward = b.g_inward
        loss.gravity = wp.vec3(0.0, 0.0, -float(self.m) * 9.81)
        loss.stiffness = float(self.K_contact)
        loss.min_pair = float(self.min_pair_distance)
        wp.launch(k.lambda_loss_batch, dim=self.mppi_samples, inputs=[loss])
        wp.synchronize()
        costs = np.asarray(b.loss.numpy(), dtype=np.float64).reshape(self.mppi_samples)
        rewards = torch.as_tensor(-costs, dtype=torch.float32)
        weights = compute_spider_weights(rewards, self.mppi_temperature).detach().cpu().numpy()
        return weights

    def choose_contact_set(
        self,
        visible_face_idx=None,
        object_pos=None,
        object_rot=None,
        **_unused,
    ):
        del object_pos, object_rot
        visible = self.get_contact_candidate_indices(visible_face_idx)
        seed_a, seed_b = self._seed_pair(visible)
        mean = np.concatenate((seed_a, seed_b)).astype(np.float64).reshape(6)
        for iteration in range(self.mppi_iterations):
            scale = self.mppi_noise * (0.85 ** iteration)
            samples = mean.reshape(1, 6) + scale * self._mppi_rng.standard_normal((self.mppi_samples, 6))
            samples[0] = mean
            weights = self._score_samples(samples)
            mean = np.sum(weights.reshape(-1, 1) * samples, axis=0)
        b = self._ensure_buffers()
        raw = np.asarray(mean, dtype=np.float32).reshape(1, 6)
        b.raw.assign(raw)
        wp = ensure_warp()
        k = ensure_kernels()
        wp.launch(k.pack_points, dim=2, inputs=[b.raw, b.points])
        gs = k.GsJob()
        gs.points = b.points
        gs.spheres = b.spheres
        gs.phi = b.phi
        gs.position = b.position
        gs.inward = b.inward
        gs.query_radius = float(self.tip_radius)
        gs.tau = float(self.gs_tau)
        gs.sphere_count = int(b.sphere_count)
        wp.launch(k.gs_features, dim=2, inputs=[gs])
        loss = k.LambdaLossJob()
        loss.phi = b.phi
        loss.position = b.position
        loss.inward = b.inward
        loss.force = b.force
        loss.loss = b.loss
        loss.g_phi = b.g_phi
        loss.g_position = b.g_position
        loss.g_inward = b.g_inward
        loss.gravity = wp.vec3(0.0, 0.0, -float(self.m) * 9.81)
        loss.stiffness = float(self.K_contact)
        loss.min_pair = float(self.min_pair_distance)
        wp.launch(k.lambda_loss, dim=1, inputs=[loss])
        wp.synchronize()
        position = np.asarray(b.position.numpy(), dtype=np.float64).reshape(2, 3)
        inward = np.asarray(b.inward.numpy(), dtype=np.float64).reshape(2, 3)
        force = np.asarray(b.force.numpy(), dtype=np.float64).reshape(2, 3)
        phi = np.asarray(b.phi.numpy(), dtype=np.float64).reshape(2)
        self.last_grasp_result = {
            "contact_indices": np.array([0, 1], dtype=int),
            "contact_points": position.copy(),
            "contact_normals": inward.copy(),
            "witness_contact_forces_local": force.copy(),
            "witness_force_vectors_local": force.copy(),
            "phi": phi,
        }
        return position, inward, float(np.sum(phi ** 2)), 0.0, float(-np.dot(inward[0], inward[1]))
