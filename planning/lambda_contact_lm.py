"""Offline/global and live/cone contact selection using batched Warp LM.

Only mesh preprocessing and small simulator input/output transfers use the
host. Candidate filtering, dynamics, LM and contact quality selection run on
the selected Warp device.
"""

from types import SimpleNamespace
import time

import numpy as np


_SELECTION = None


def ensure_selection_kernels():
    global _SELECTION
    if _SELECTION is not None:
        return _SELECTION
    import warp as wp

    @wp.func
    def pose_cost(pose: wp.array(dtype=float), target: wp.array(dtype=float),
                  position_weight: float, rotation_weight: float) -> float:
        position = wp.vec3(pose[0] - target[0], pose[1] - target[1], pose[2] - target[2])
        dot = float(0.0)
        for i in range(4):
            dot += pose[i + 3] * target[i + 3]
        return position_weight * wp.length_sq(position) + rotation_weight * wp.max(1.0 - dot * dot, 0.0)

    @wp.func
    def in_cone(point: wp.vec3, inward: wp.vec3, tip: wp.vec3,
                center: wp.vec3, radius: float, cosine: float) -> bool:
        heading = center - tip
        delta = point - tip
        hlen = wp.length(heading)
        dlen = wp.length(delta)
        keep = bool(False)
        if hlen > 1.e-8 and dlen > 1.e-8:
            along = wp.dot(delta, heading / hlen)
            keep = along > 0.0 and along <= hlen + radius and along / dlen >= cosine
            keep = keep and wp.dot(-inward, tip - point) > 0.0
        return keep

    @wp.kernel(enable_backward=False)
    def select_global(reductions: wp.array(dtype=float), valid: wp.array(dtype=int),
                      ids: wp.array2d(dtype=int), points: wp.array(dtype=wp.vec3),
                      curvature: wp.array(dtype=float), center: wp.vec3, radius: float,
                      current: wp.array(dtype=float), target: wp.array(dtype=float),
                      tips: wp.array(dtype=wp.vec3), position_weight: float,
                      rotation_weight: float, quality_fraction: float, curvature_weight: float,
                      selected: wp.array(dtype=int)):
        best = float(-1.e30)
        for row in range(valid.shape[0]):
            if valid[row] != 0 and wp.isfinite(reductions[row]):
                best = wp.max(best, reductions[row])
        cost_now = pose_cost(current, target, position_weight, rotation_weight)
        margin = wp.max(quality_fraction * wp.abs(best), 1.e-6 * wp.max(cost_now, 1.e-8))
        winner = int(-1)
        best_geometry = float(1.e30)
        for row in range(valid.shape[0]):
            if valid[row] != 0 and wp.isfinite(reductions[row]) and reductions[row] >= best - margin:
                first, second = int(ids[row, 0]), int(ids[row, 1])
                midpoint = 0.5 * (points[2 * row] + points[2 * row + 1])
                geometry = curvature_weight * (curvature[first] + curvature[second])
                geometry += wp.length_sq(midpoint - center) / wp.max(radius * radius, 1.e-10)
                if geometry < best_geometry:
                    best_geometry, winner = geometry, row
        selected[0] = winner
        selected[1] = 0
        if winner >= 0:
            p0, p1 = wp.vec3(points[2 * winner]), wp.vec3(points[2 * winner + 1])
            keep = wp.length(p0 - tips[0]) + wp.length(p1 - tips[1])
            swap = wp.length(p1 - tips[0]) + wp.length(p0 - tips[1])
            if swap < keep:
                selected[1] = 1

    @wp.func
    def assemble_row(row: int, source_points: wp.array(dtype=wp.vec3),
                        source_normals: wp.array(dtype=wp.vec3),
                        offline: wp.array(dtype=int), active: wp.array(dtype=int),
                        candidate_points: wp.array(dtype=wp.vec3),
                        candidate_normals: wp.array(dtype=wp.vec3),
                        candidate_ids: wp.array2d(dtype=int), candidate_valid: wp.array(dtype=int),
                        tips: wp.array(dtype=wp.vec3), center: wp.vec3,
                        radius: float, cosine: float, support_enabled: int,
                        support_point: wp.vec3, support_normal: wp.vec3,
                        pad_radius: float, support_clearance: float,
                        points: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
                        ids: wp.array2d(dtype=int), valid: wp.array(dtype=int)):
        first, second = int(-1), int(-1)
        p0, p1, n0, n1 = wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)
        keep = bool(False)
        if row < 2:
            first, second = int(offline[0]), int(offline[1])
            if row == 1:
                first, second = int(active[0]), int(active[1])
            if first >= 0 and second >= 0:
                p0, p1 = wp.vec3(source_points[first]), wp.vec3(source_points[second])
                n0, n1 = wp.vec3(source_normals[first]), wp.vec3(source_normals[second])
                keep = True
                if row == 0:
                    direct = wp.length(p0 - tips[0]) + wp.length(p1 - tips[1])
                    crossed = wp.length(p1 - tips[0]) + wp.length(p0 - tips[1])
                    if crossed < direct:
                        first, second = second, first
                        p0, p1 = wp.vec3(source_points[first]), wp.vec3(source_points[second])
                        n0, n1 = wp.vec3(source_normals[first]), wp.vec3(source_normals[second])
                else:
                    keep = in_cone(p0, n0, tips[0], center, radius, cosine)
                    keep = keep and in_cone(p1, n1, tips[1], center, radius, cosine)
                if support_enabled != 0:
                    gap0 = wp.dot(p0 - pad_radius * n0 - support_point, support_normal) - pad_radius
                    gap1 = wp.dot(p1 - pad_radius * n1 - support_point, support_normal) - pad_radius
                    keep = keep and wp.min(gap0, gap1) >= support_clearance
        else:
            index = row - 2
            first, second = int(candidate_ids[index, 0]), int(candidate_ids[index, 1])
            p0, p1 = wp.vec3(candidate_points[2 * index]), wp.vec3(candidate_points[2 * index + 1])
            n0, n1 = wp.vec3(candidate_normals[2 * index]), wp.vec3(candidate_normals[2 * index + 1])
            keep = candidate_valid[index] != 0
        points[2 * row] = p0
        points[2 * row + 1] = p1
        normals[2 * row] = n0
        normals[2 * row + 1] = n1
        ids[row, 0] = first
        ids[row, 1] = second
        valid[row] = int(keep)

    @wp.kernel(enable_backward=False)
    def assemble_online(source_points: wp.array(dtype=wp.vec3), source_normals: wp.array(dtype=wp.vec3),
                        offline: wp.array(dtype=int), active: wp.array(dtype=int),
                        candidate_points: wp.array(dtype=wp.vec3), candidate_normals: wp.array(dtype=wp.vec3),
                        candidate_ids: wp.array2d(dtype=int), candidate_valid: wp.array(dtype=int),
                        tips: wp.array(dtype=wp.vec3), center: wp.vec3, radius: float, cosine: float,
                        support_enabled: int, support_point: wp.vec3, support_normal: wp.vec3,
                        pad_radius: float, support_clearance: float, points: wp.array(dtype=wp.vec3),
                        normals: wp.array(dtype=wp.vec3), ids: wp.array2d(dtype=int), valid: wp.array(dtype=int)):
        assemble_row(wp.tid(), source_points, source_normals, offline, active, candidate_points,
            candidate_normals, candidate_ids, candidate_valid, tips, center, radius, cosine,
            support_enabled, support_point, support_normal, pad_radius, support_clearance,
            points, normals, ids, valid)

    @wp.kernel(enable_backward=False)
    def assemble_device(source_points: wp.array(dtype=wp.vec3), source_normals: wp.array(dtype=wp.vec3),
                        offline: wp.array(dtype=int), active: wp.array(dtype=int),
                        candidate_points: wp.array(dtype=wp.vec3), candidate_normals: wp.array(dtype=wp.vec3),
                        candidate_ids: wp.array2d(dtype=int), candidate_valid: wp.array(dtype=int),
                        tips: wp.array(dtype=wp.vec3), center: wp.vec3, radius: float, cosine: float,
                        support_enabled: int, plane: wp.array(dtype=wp.vec3),
                        pad_radius: float, support_clearance: float, points: wp.array(dtype=wp.vec3),
                        normals: wp.array(dtype=wp.vec3), ids: wp.array2d(dtype=int), valid: wp.array(dtype=int)):
        assemble_row(wp.tid(), source_points, source_normals, offline, active, candidate_points,
            candidate_normals, candidate_ids, candidate_valid, tips, center, radius, cosine,
            support_enabled, wp.vec3(plane[0]), wp.vec3(plane[1]), pad_radius, support_clearance,
            points, normals, ids, valid)

    @wp.kernel(enable_backward=False)
    def prepare_inputs(current: wp.array(dtype=float), target: wp.array(dtype=float),
                       tips_world: wp.array(dtype=wp.vec3), plane_world: wp.array(dtype=wp.vec3),
                       tips_local: wp.array(dtype=wp.vec3), plane_local: wp.array(dtype=wp.vec3)):
        position = wp.vec3(current[0], current[1], current[2])
        rotation = wp.normalize(wp.quat(current[4], current[5], current[6], current[3]))
        goal = wp.normalize(wp.quat(target[4], target[5], target[6], target[3]))
        current[3] = rotation[3]
        target[3] = goal[3]
        for axis in range(3):
            current[axis + 4] = rotation[axis]
            target[axis + 4] = goal[axis]
        for arm in range(2):
            tips_local[arm] = wp.quat_rotate_inv(rotation, tips_world[arm] - position)
        plane_local[0] = wp.quat_rotate_inv(rotation, plane_world[0] - position)
        plane_local[1] = wp.quat_rotate_inv(rotation, plane_world[1])

    @wp.kernel(enable_backward=False)
    def select_online(reductions: wp.array(dtype=float), valid: wp.array(dtype=int),
                      points: wp.array(dtype=wp.vec3), tips: wp.array(dtype=wp.vec3),
                      current: wp.array(dtype=float), target: wp.array(dtype=float),
                      position_weight: float, rotation_weight: float, quality_fraction: float,
                      switch_margin: float, selected: wp.array(dtype=int),
                      diagnostics: wp.array(dtype=float)):
        reference = float(-1.e30)
        if valid[0] != 0 and wp.isfinite(reductions[0]):
            reference = reductions[0]
        for row in range(1, valid.shape[0]):
            if valid[row] != 0 and wp.isfinite(reductions[row]):
                reference = wp.max(reference, reductions[row])
        now = pose_cost(current, target, position_weight, rotation_weight)
        margin = wp.max(quality_fraction * wp.abs(reference), 1.e-6 * wp.max(now, 1.e-8))
        winner = int(-1)
        closest = float(1.e30)
        eligible = int(0)
        incumbent_distance = float(1.e30)
        for row in range(1, valid.shape[0]):
            if valid[row] != 0 and wp.isfinite(reductions[row]) and reductions[row] >= reference - margin:
                if reference <= 1.e-9 or reductions[row] > 1.e-9:
                    eligible += 1
                    distance = wp.length(points[2 * row] - tips[0]) + wp.length(points[2 * row + 1] - tips[1])
                    if row == 1:
                        incumbent_distance = distance
                    if distance < closest:
                        closest, winner = distance, row
        if winner >= 0 and incumbent_distance <= closest + switch_margin:
            winner = 1
        if winner < 0 and valid[0] != 0 and wp.isfinite(reductions[0]):
            winner = 0
        selected[0] = winner
        selected[1] = 0
        diagnostics[0] = reference
        diagnostics[1] = margin
        diagnostics[2] = float(eligible)
        diagnostics[3] = float(winner == 0)
        diagnostics[4] = now

    @wp.func
    def support_geometry_impl(spheres: wp.array(dtype=wp.vec4), rotation: wp.mat33,
                         position: wp.vec3, support_point: wp.vec3, support_normal: wp.vec3,
                         row: wp.array(dtype=float), gap: wp.array(dtype=float)):
        best = float(1.e30)
        point = wp.vec3(0.0)
        for index in range(spheres.shape[0]):
            sphere = spheres[index]
            local = wp.vec3(sphere[0], sphere[1], sphere[2])
            distance = wp.dot(rotation * local + position - support_point, support_normal) - sphere[3]
            if distance < best:
                best, point = distance, local
        angular = wp.cross(point, wp.transpose(rotation) * support_normal)
        for axis in range(3):
            row[axis] = support_normal[axis]
            row[axis + 3] = angular[axis]
        gap[0] = best

    @wp.kernel(enable_backward=False)
    def support_geometry(spheres: wp.array(dtype=wp.vec4), rotation: wp.mat33,
                         position: wp.vec3, support_point: wp.vec3, support_normal: wp.vec3,
                         row: wp.array(dtype=float), gap: wp.array(dtype=float)):
        support_geometry_impl(spheres, rotation, position, support_point, support_normal, row, gap)

    @wp.kernel(enable_backward=False)
    def support_device(spheres: wp.array(dtype=wp.vec4), pose: wp.array(dtype=float),
                       plane_world: wp.array(dtype=wp.vec3), row: wp.array(dtype=float),
                       gap: wp.array(dtype=float)):
        quaternion = wp.quat(pose[4], pose[5], pose[6], pose[3])
        support_geometry_impl(spheres, wp.quat_to_matrix(quaternion),
            wp.vec3(pose[0], pose[1], pose[2]), wp.vec3(plane_world[0]), wp.vec3(plane_world[1]), row, gap)

    @wp.kernel(enable_backward=False)
    def gather(selected: wp.array(dtype=int), points: wp.array(dtype=wp.vec3),
               normals: wp.array(dtype=wp.vec3), ids: wp.array2d(dtype=int),
               forces: wp.array2d(dtype=float), poses: wp.array2d(dtype=float),
               velocities: wp.array2d(dtype=float), costs: wp.array(dtype=float),
               pose_costs: wp.array(dtype=float), reductions: wp.array(dtype=float),
               summary: wp.array(dtype=float), chosen_ids: wp.array(dtype=int)):
        row, flip = int(selected[0]), int(selected[1])
        if row >= 0:
            for contact in range(2):
                slot = contact
                if flip != 0:
                    slot = 1 - contact
                chosen_ids[contact] = ids[row, slot]
                p, n = wp.vec3(points[2 * row + slot]), wp.vec3(normals[2 * row + slot])
                for axis in range(3):
                    summary[3 * contact + axis] = p[axis]
                    summary[6 + 3 * contact + axis] = n[axis]
                    summary[12 + 3 * contact + axis] = forces[row, 3 * slot + axis]
                helper = wp.vec3(1.0, 0.0, 0.0)
                if wp.abs(n[0]) > 0.9:
                    helper = wp.vec3(0.0, 1.0, 0.0)
                t1 = wp.normalize(wp.cross(n, helper))
                t2 = wp.cross(n, t1)
                vector = n * forces[row, 3 * slot] + t1 * forces[row, 3 * slot + 1] + t2 * forces[row, 3 * slot + 2]
                for axis in range(3):
                    summary[18 + 3 * contact + axis] = vector[axis]
            for axis in range(7):
                summary[24 + axis] = poses[row, axis]
            for axis in range(6):
                summary[31 + axis] = velocities[row, axis]
            summary[37] = costs[row]
            summary[38] = pose_costs[row]
            summary[39] = reductions[row]

    _SELECTION = SimpleNamespace(wp=wp, select_global=select_global,
        assemble_online=assemble_online, select_online=select_online,
        support_geometry=support_geometry, gather=gather, prepare_inputs=prepare_inputs,
        assemble_device=assemble_device, support_device=support_device)
    return _SELECTION


class LambdaContactBatchedLM:
    """Two-arm lambda contact policy with an offline reference and live cones."""

    def __init__(self, mesh_path, *, scale_factors=(1.0, 1.0, 1.0), sample_num=70,
                 obj_mass=0.1, arm_friction=0.9, contact_stiffness=0.2,
                 time_step=0.01, pos_coef=1.0, ori_coef=0.0005,
                 tip_radius=0.01, device="cuda:0", seed=0, lm_iterations=20,
                 cone_half_angle_deg=45.0, quality_fraction=0.015,
                 max_per_arm=32, switch_margin=0.002, curvature_penalty_weight=1.5,
                 max_point_curvature=0.25, max_mean_curvature=0.12,
                 max_contact_force=10.0, force_reg_coef=0.0, friction_reg_coef=0.0,
                 support_surface_point=None, support_surface_normal=None,
                 support_surface_clearance=0.0, body_sides=False,
                 body_core_fraction=0.5, body_side_normal_cos=0.85, comfree_damping=0.001, **unused):
        from models.comfree_gs_torch import GaussianCloud
        from planning.contact_candidates_warp import SurfaceContactCandidates
        from planning.lambda_batched_lm import BatchedLambdaLM

        if not 0 < cone_half_angle_deg < 90 or not 0 <= quality_fraction < 1:
            raise ValueError("lambda cone half angle must be in (0,90), quality fraction in [0,1)")
        if max_per_arm < 1 or not np.isfinite(switch_margin) or switch_margin < 0:
            raise ValueError("lambda max_per_arm must be positive and switch_margin finite/nonnegative")
        self.comfree_damping = float(comfree_damping)
        self.device = str(device)
        self.m, self.mu_arm_obj, self.K_contact, self.h = map(float, (obj_mass, arm_friction, contact_stiffness, time_step))
        self.pos_coef, self.ori_coef, self.tip_radius = float(pos_coef), float(ori_coef), float(tip_radius)
        self.quality_fraction, self.cone_half_angle_deg = float(quality_fraction), float(cone_half_angle_deg)
        self.max_per_arm, self.switch_margin = int(max_per_arm), float(switch_margin)
        self.curvature_penalty_weight = float(curvature_penalty_weight)
        self.surface = SurfaceContactCandidates(mesh_path, scale_factors, sample_num=sample_num,
            seed=seed, patch_radius=tip_radius, max_point_curvature=max_point_curvature,
            max_mean_curvature=max_mean_curvature, device=device, body_sides=body_sides,
            body_core_fraction=body_core_fraction, body_side_normal_cos=body_side_normal_cos)
        self.sample_point, self.normal = self.surface.points_local, self.surface.normals_local
        self.sample_num, self.point_idx = self.surface.sample_num, np.arange(self.surface.sample_num)
        self.point_curvature = self.surface.curvature_max
        self.point_curvature_mean = self.surface.curvature_mean
        # Every contact candidate is an exact GS centre; additional surface
        # samples cover the rest of the object for native collision dynamics.
        dense = self.surface.dense_points_local
        stride = max(1, len(dense) // 1024)
        self.cloud = GaussianCloud(np.vstack((self.sample_point, dense[::stride])), radius=0.004)
        self.lm = BatchedLambdaLM(device=device, iterations=lm_iterations,
            mass=obj_mass, mu=arm_friction, max_contact_force=max_contact_force, pos_coef=pos_coef,
            ori_coef=ori_coef, force_reg_coef=force_reg_coef, friction_reg_coef=friction_reg_coef)
        self.k = ensure_selection_kernels()
        wp = self.k.wp
        self._tips = wp.zeros(2, dtype=wp.vec3, device=device)
        self._tips_world = wp.zeros(2, dtype=wp.vec3, device=device)
        self._support_world = wp.zeros(2, dtype=wp.vec3, device=device)
        self._support_local = wp.zeros(2, dtype=wp.vec3, device=device)
        self._current = wp.zeros(7, dtype=float, device=device)
        self._target = wp.zeros(7, dtype=float, device=device)
        self._choice = wp.zeros(2, dtype=int, device=device)
        self._diagnostics = wp.zeros(5, dtype=float, device=device)
        self._summary = wp.zeros(40, dtype=float, device=device)
        self._chosen_ids = wp.zeros(2, dtype=int, device=device)
        self._offline_ids = wp.full(2, -1, dtype=int, device=device)
        self._active_ids = wp.full(2, -1, dtype=int, device=device)
        self._curvature = wp.array(self.surface.curvature_mean, dtype=float, device=device)
        self._spheres = wp.array(self.cloud.as_spheres(), dtype=wp.vec4, device=device)
        self._support_row = wp.zeros(6, dtype=float, device=device)
        self._support_gap = wp.zeros(1, dtype=float, device=device)
        self.support_surface_point = None
        self.support_surface_normal = None
        self.support_surface_clearance = float(support_surface_clearance)
        self.set_support_surface(support_surface_point, support_surface_normal)
        self.last_grasp_result = None
        self.enable_timing_prints = False
        self._online = None
        self._initialized = False
        self.offline_solve_count = 0
        self.online_solve_count = 0

    def reset(self):
        self._initialized = False
        self.last_grasp_result = None
        self._offline_ids.fill_(-1)
        self._active_ids.fill_(-1)
        self.offline_solve_count = self.online_solve_count = 0

    def set_timing_print_enabled(self, enabled):
        self.enable_timing_prints = bool(enabled)

    def set_support_surface(self, support_surface_point=None, support_surface_normal=None, **unused):
        if (support_surface_point is None) != (support_surface_normal is None):
            raise ValueError("support point and normal must be provided together")
        self.support_surface_point = None if support_surface_point is None else np.asarray(support_surface_point, float).reshape(3)
        self.support_surface_normal = None if support_surface_normal is None else np.asarray(support_surface_normal, float).reshape(3)
        if self.support_surface_normal is not None:
            length = np.linalg.norm(self.support_surface_normal)
            if not np.isfinite(self.support_surface_point).all() or not np.isfinite(length) or length < 1.e-12:
                raise ValueError("support point/normal must be finite and normal nonzero")
            self.support_surface_normal = self.support_surface_normal / length
            self._support_world.assign(np.asarray([self.support_surface_point, self.support_surface_normal], np.float32))
        else:
            self._support_world.zero_()

    def get_contact_candidate_indices(self, visible_face_idx=None, **unused):
        return self.point_idx.copy()

    def _inputs(self, x_d, current_x, tips_world):
        pose, target = np.asarray(current_x, float).reshape(7).copy(), np.asarray(x_d, float).reshape(7).copy()
        tips = np.asarray(tips_world, float).reshape(2, 3)
        if not np.isfinite(pose).all() or not np.isfinite(target).all() or not np.isfinite(tips).all():
            raise ValueError("lambda poses and tips must be finite")
        for value in (pose, target):
            if np.linalg.norm(value[3:]) < 1.e-10:
                raise ValueError("lambda quaternion must be nonzero")
        self._tips_world.assign(tips.astype(np.float32))
        self._current.assign(pose.astype(np.float32))
        self._target.assign(target.astype(np.float32))
        self.k.wp.launch(self.k.prepare_inputs, dim=1, inputs=[self._current, self._target,
            self._tips_world, self._support_world, self._tips, self._support_local], device=self.device)
        constraint = dict(arm_friction=self.mu_arm_obj, min_separation=0.03,
            fingertip_radius=self.tip_radius, support_clearance=self.support_surface_clearance)
        if self.support_surface_normal is not None:
            constraint["support_plane_device"] = self._support_local
        return self._current, self._target, self._tips, constraint

    def _solve(self, batch, pose, target, *, object_inertia, v_obj, tau_o, prediction_dt):
        kwargs = dict(current_x=pose, x_d=target, prediction_dt=prediction_dt,
                      object_inertia=object_inertia, v_obj=v_obj, tau_o=tau_o, valid_mask=batch.valid)
        if self.support_surface_normal is not None:
            wp = self.k.wp
            wp.launch(self.k.support_device, dim=1, inputs=[self._spheres,
                self._current, self._support_world,
                self._support_row, self._support_gap], device=self.device)
            kwargs.update(support_row=self._support_row, support_gap=self._support_gap,
                          support_stiffness=self.K_contact, support_damping=self.comfree_damping)
        return self.lm.solve_device(batch.points, batch.normals, **kwargs)

    def _result(self, batch, result, prediction_dt, started, offline):
        wp = self.k.wp
        row = int(self._choice.numpy()[0])
        if row < 0:
            raise FloatingPointError("No admissible low-curvature lambda contact pair")
        wp.launch(self.k.gather, dim=1, inputs=[self._choice, batch.points, batch.normals,
            batch.ids, result.forces, result.x_plus, result.velocity_next, result.costs,
            result.pose_costs, result.reductions, self._summary, self._chosen_ids], device=self.device)
        values, ids = self._summary.numpy().astype(float), self._chosen_ids.numpy().copy()
        points, normals = values[:6].reshape(2, 3), values[6:12].reshape(2, 3)
        coefficients, vectors = values[12:18].reshape(2, 3), values[18:24].reshape(2, 3)
        self._active_ids.assign(ids)
        if offline:
            self._offline_ids.assign(ids)
        self._initialized = True
        diag = self._diagnostics.numpy().astype(float)
        info = dict(solver_backend="lambda_warp_batched_lm", solver_failed=False,
            contact_model="asu_comfree", comfree_stiffness=self.K_contact,
            comfree_damping=self.comfree_damping,
            solve_time=time.perf_counter() - started, prediction_dt=float(prediction_dt),
            contact_indices=ids.copy(), query_contact_points_local=points.copy(),
            contact_normals=normals.copy(), control_input=coefficients.copy(),
            contact_forces_local=coefficients.copy(), force_vectors_local=vectors.copy(),
            velocity_next=values[31:37].copy(), resulting_pose=values[24:31].copy(),
            x_plus_opt=values[24:31].copy(), pose_cost=float(values[38]),
            cost_reduction=float(values[39]), candidate_capacity=batch.capacity,
            source="offline_global" if offline else ("offline_fallback" if row == 0 else "online_cones"),
            geometric_center_local=self.surface.geometric_center_local.copy(),
            cone_half_angle_deg=self.cone_half_angle_deg, quality_fraction=self.quality_fraction,
            reference_cost_reduction=float(diag[0]) if not offline else float(values[39]),
            quality_margin=float(diag[1]) if not offline else 0.0,
            near_quality_pairs=int(diag[2]) if not offline else 0,
            curvature_max=self.surface.curvature_max[ids].copy(),
            curvature_mean=self.surface.curvature_mean[ids].copy(),
            contact_region="body_sides" if self.surface.body_sides else "all",
            offline_solve_count=self.offline_solve_count, online_solve_count=self.online_solve_count)
        self.last_grasp_result = dict(info, contact_points=points.copy(),
            witness_contact_forces_local=coefficients.copy(), witness_force_vectors_local=vectors.copy(),
            total_cost=float(values[37]), valid=True)
        if self.enable_timing_prints:
            print(f"lambda LM {info['source']}: candidates={batch.capacity} delta={values[39]:.6g} time={info['solve_time']*1000:.1f}ms")
        return points, normals, values[24:31].copy(), float(values[37]), info

    def initialize_offline(self, x_d, current_x, *, tips_world, object_inertia,
                           v_obj=None, tau_o=None, prediction_dt=None):
        started = time.perf_counter()
        dt = self.h if prediction_dt is None else float(prediction_dt)
        pose, target, tips, constraints = self._inputs(x_d, current_x, tips_world)
        batch = self.surface.global_pairs(**constraints)
        result = self._solve(batch, pose, target, object_inertia=object_inertia,
                             v_obj=v_obj, tau_o=tau_o, prediction_dt=dt)
        wp = self.k.wp
        wp.launch(self.k.select_global, dim=1, inputs=[result.reductions, result.valid, batch.ids,
            batch.points, self._curvature, wp.vec3(*self.surface.geometric_center_local),
            self.surface.object_radius, self._current, self._target, self._tips,
            self.pos_coef, self.ori_coef, self.quality_fraction, self.curvature_penalty_weight,
            self._choice], device=self.device)
        self.offline_solve_count += 1
        return self._result(batch, result, dt, started, True)

    def update_contacts(self, x_d, current_x, *, tips_world, object_inertia,
                        v_obj=None, tau_o=None, prediction_dt=None):
        if not self._initialized:
            return self.initialize_offline(x_d, current_x, tips_world=tips_world,
                object_inertia=object_inertia, v_obj=v_obj, tau_o=tau_o, prediction_dt=prediction_dt)
        from planning.contact_candidates_warp import PairCandidateBatch
        started = time.perf_counter()
        dt = self.h if prediction_dt is None else float(prediction_dt)
        pose, target, tips, constraints = self._inputs(x_d, current_x, tips_world)
        candidates = self.surface.online_pairs(tips, cone_half_angle_deg=self.cone_half_angle_deg,
            max_per_arm=self.max_per_arm, max_pairs=self.max_per_arm**2, **constraints)
        wp = self.k.wp
        if self._online is None or self._online.capacity != candidates.capacity + 2:
            size = candidates.capacity + 2
            self._online = PairCandidateBatch(
                wp.zeros(2*size, dtype=wp.vec3, device=self.device),
                wp.zeros(2*size, dtype=wp.vec3, device=self.device),
                wp.zeros((size, 2), dtype=int, device=self.device),
                wp.zeros(size, dtype=int, device=self.device),
                wp.zeros(size, dtype=float, device=self.device), size)
        batch = self._online
        wp.launch(self.k.assemble_device, dim=batch.capacity, inputs=[
            self.surface.points_device, self.surface.normals_device, self._offline_ids, self._active_ids,
            candidates.points, candidates.normals, candidates.ids, candidates.valid,
            self._tips, wp.vec3(*self.surface.geometric_center_local), self.surface.object_radius,
            float(np.cos(np.deg2rad(self.cone_half_angle_deg))), int(self.support_surface_normal is not None),
            self._support_local, self.tip_radius,
            self.support_surface_clearance, batch.points, batch.normals, batch.ids, batch.valid], device=self.device)
        result = self._solve(batch, pose, target, object_inertia=object_inertia,
                             v_obj=v_obj, tau_o=tau_o, prediction_dt=dt)
        wp.launch(self.k.select_online, dim=1, inputs=[result.reductions, result.valid, batch.points,
            self._tips, self._current, self._target, self.pos_coef, self.ori_coef,
            self.quality_fraction, self.switch_margin, self._choice, self._diagnostics], device=self.device)
        self.online_solve_count += 1
        return self._result(batch, result, dt, started, False)
