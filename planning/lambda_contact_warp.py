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

    _KERNELS = SimpleNamespace(
        wp=wp,
        GsJob=GsJob,
        LambdaLossJob=LambdaLossJob,
        GsVjpJob=GsVjpJob,
        gs_features=gs_features,
        lambda_loss=lambda_loss,
        gs_features_vjp=gs_features_vjp,
        pack_points=pack_points,
    )
    return _KERNELS


class LambdaContactWarp(LambdaContactAdamOptimizer):
    """Once-per-run Warp Adam contact picker.  Host I/O only at the edges."""

    def __init__(self, device=None, **kwargs):
        requested = None if device is None else str(device)
        super().__init__(device="cpu", **kwargs)
        self.warp_device = self._resolve_device(requested)
        self._graph = None
        self._buffers = None
        self._captured_n = None

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
            "witness_contact_forces_local": force.copy(),
            "witness_force_vectors_local": force.copy(),
            "phi": phi,
        }
        return position, inward, float(np.sum(phi ** 2)), 0.0, antipodal_margin
