"""Offline/global and live/cone contact selection using batched Warp LM.

The two-contact policy keeps candidate filtering and selection on Warp.
The n-contact policy builds bounded, assigned candidates on the host. Both
policies use the same Warp dynamics, force projection and LM iterations.
"""

from types import SimpleNamespace
import copy
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

    @wp.kernel(enable_backward=False)
    def keep_loaded_pair(valid: wp.array(dtype=int), selected: wp.array(dtype=int)):
        if valid[1] != 0:
            selected[0] = 1
            selected[1] = 0

    @wp.kernel(enable_backward=False)
    def retain_incumbent(valid: wp.array(dtype=int)):
        valid[1] = 1

    @wp.kernel(enable_backward=False)
    def verify_pair(points: wp.array(dtype=wp.vec3), tips: wp.array(dtype=wp.vec3),
                    reductions: wp.array(dtype=float), margin: float,
                    selected: wp.array(dtype=int), verification: wp.array(dtype=float)):
        """A summed distance gain cannot conceal the other arm's loss."""
        row = int(selected[0])
        accept = bool(row >= 0)
        for arm in range(2):
            gain = float(-1.e30)
            if row >= 0:
                gain = wp.length(points[2 + arm] - tips[arm]) - wp.length(points[2 * row + arm] - tips[arm])
            verification[arm] = gain
            accept = accept and wp.isfinite(gain) and gain >= -margin
        # Both arms share the object's goal cost; never buy shorter reaches
        # with a worse predicted object response.
        if row >= 0:
            accept = accept and wp.isfinite(reductions[row]) and reductions[row] >= reductions[1] - 1.e-9
        verification[2] = float(accept)
        if not accept:
            selected[0] = 1
        selected[1] = 0

    _SELECTION = SimpleNamespace(wp=wp, select_global=select_global,
        assemble_online=assemble_online, select_online=select_online,
        support_geometry=support_geometry, gather=gather, prepare_inputs=prepare_inputs,
        assemble_device=assemble_device, support_device=support_device, keep_loaded_pair=keep_loaded_pair,
        retain_incumbent=retain_incumbent, verify_pair=verify_pair)
    return _SELECTION


class LambdaContactBatchedLM:
    """Shared Lambda policy; num_contacts defaults to the existing two-pad path."""

    def __new__(cls, *args, num_contacts=2, **kwargs):
        if cls is LambdaContactBatchedLM and num_contacts != 2:
            return object.__new__(LambdaContactMultiLM)
        return object.__new__(cls)

    def __init__(self, mesh_path, *, num_contacts=2, scale_factors=(1.0, 1.0, 1.0), sample_num=70,
                 obj_mass=0.1, arm_friction=0.9, contact_stiffness=0.2,
                 time_step=0.01, pos_coef=1.0, ori_coef=0.0005,
                 tip_radius=0.01, device="cuda:0", seed=0, lm_iterations=20,
                 cone_half_angle_deg=45.0, quality_fraction=0.015,
                 max_per_arm=32, switch_margin=0.002, curvature_penalty_weight=1.5,
                 max_point_curvature=0.25, max_mean_curvature=0.12,
                 max_contact_force=10.0, force_reg_coef=0.0, friction_reg_coef=0.0,
                 support_surface_point=None, support_surface_normal=None,
                 support_surface_clearance=0.0, body_sides=False,
                 body_core_fraction=0.5, body_side_normal_cos=0.85, comfree_damping=0.001,
                 friction_cone="box", paired_center_local=None, paired_up_local=(0., 0., 1.), **unused):
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
            body_core_fraction=body_core_fraction, body_side_normal_cos=body_side_normal_cos,
            paired_center_local=paired_center_local, paired_up_local=paired_up_local)
        self.sample_point, self.normal = self.surface.points_local, self.surface.normals_local
        self.sample_num, self.point_idx = self.surface.sample_num, np.arange(self.surface.sample_num)
        self.point_curvature = self.surface.curvature_max
        self.point_curvature_mean = self.surface.curvature_mean
        # Every contact candidate is an exact GS centre; additional surface
        # samples cover the rest of the object for native collision dynamics.
        dense = self.surface.dense_points_local
        stride = max(1, len(dense) // 1024)
        self.cloud = GaussianCloud(np.vstack((self.sample_point, dense[::stride])), radius=0.004)
        self.lm = BatchedLambdaLM(device=device, iterations=lm_iterations, num_contacts=num_contacts,
            mass=obj_mass, mu=arm_friction, max_contact_force=max_contact_force, pos_coef=pos_coef,
            ori_coef=ori_coef, force_reg_coef=force_reg_coef, friction_reg_coef=friction_reg_coef,
            friction_cone=friction_cone)
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
        self._verification = wp.zeros(3, dtype=float, device=device)
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
        self._fixed = None
        self._initialized = False
        self.offline_solve_count = 0
        self.online_solve_count = 0

    def reset(self):
        self._initialized = False
        self.last_grasp_result = None
        self._fixed = None
        self._offline_ids.fill_(-1)
        self._active_ids.fill_(-1)
        self.offline_solve_count = self.online_solve_count = 0

    def fork_for_search(self):
        """Share immutable candidates/kernels, isolate all search work buffers."""
        other = copy.copy(self)
        other.surface = copy.copy(self.surface)
        other.surface._buffers = {}
        other.lm = copy.copy(self.lm)
        other.lm.reset()
        wp = self.k.wp
        for name, value in vars(self).items():
            if name.startswith('_') and isinstance(value, wp.array):
                setattr(other, name, wp.clone(value))
        other.reset()
        other._online = None
        return other

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

    def _result(self, batch, result, prediction_dt, started, offline, locked=False):
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
            selection_failed=False,
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
        info["contact_pair_locked"] = bool(locked and row == 1)
        self.last_grasp_result = dict(info, contact_points=points.copy(),
            witness_contact_forces_local=coefficients.copy(), witness_force_vectors_local=vectors.copy(),
            total_cost=float(values[37]), valid=True)
        if self.enable_timing_prints:
            print(f"lambda LM {info['source']}: candidates={batch.capacity} delta={values[39]:.6g} time={info['solve_time']*1000:.1f}ms")
        return points, normals, values[24:31].copy(), float(values[37]), info

    def initialize_offline(self, x_d, current_x, *, tips_world, object_inertia,
                           v_obj=None, tau_o=None, prediction_dt=None,
                           candidate_ranker=None, candidate_validator=None, max_candidate_checks=32):
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
        selection = None
        if candidate_ranker is not None:
            # A bounded host shortlist is used only when robot feasibility is
            # supplied. Other clients retain the entirely device-side policy.
            ids = batch.ids.numpy()
            # CPU numpy() aliases Warp storage; assignment ranking must not
            # mutate the batch later gathered with its original IDs/forces.
            points = batch.points.numpy().reshape(-1, 2, 3).copy()
            normals = batch.normals.numpy().reshape(-1, 2, 3).copy()
            local_tips = self._tips.numpy()
            direct = np.linalg.norm(points-local_tips[None], axis=2).sum(axis=1)
            crossed = np.linalg.norm(points[:, ::-1]-local_tips[None], axis=2).sum(axis=1)
            flipped = crossed < direct
            points[flipped], normals[flipped] = points[flipped, ::-1], normals[flipped, ::-1]
            scores, diagnostics = candidate_ranker(points, normals, local_tips,
                self.surface.curvature_mean[ids], result.reductions.numpy())
            admissible = (result.valid.numpy() != 0) & np.isfinite(scores)
            ranked = np.flatnonzero(admissible)
            ranked = ranked[np.argsort(scores[ranked], kind='stable')]
            selected = -1
            rejected = []
            for row in ranked[:max_candidate_checks]:
                check = {} if candidate_validator is None else candidate_validator(points[row], normals[row])
                if check is not None and check.get('valid', True):
                    selected = int(row)
                    selection = dict(check, candidate_checks=len(rejected)+1,
                        rejected_candidates=rejected, prediction_cost=float(scores[row]))
                    if isinstance(diagnostics, dict):
                        selection.update(diagnostics[int(row)])
                    else:
                        selection['unsupported_load_ratio'] = float(diagnostics[row])
                    break
                rejected.append(dict(row=int(row), reason=(check or {}).get('reason', 'robot infeasible')))
            self._choice.assign(np.array([selected, int(flipped[selected]) if selected >= 0 else 0], np.int32))
            if selected < 0:
                raise FloatingPointError(f'No reachable predicted-cost Lambda pair: '
                    f'{len(ranked)} candidates, {len(rejected)} robot checks; {rejected[:3]}')
        self.offline_solve_count += 1
        output = self._result(batch, result, dt, started, True)
        if selection is not None:
            output[4]['contact_selection'] = selection
            self.last_grasp_result['contact_selection'] = selection
            if 'static_equilibrium' in selection:
                self.last_grasp_result['static_equilibrium'] = selection['static_equilibrium']
            if 'x_plus' in selection:
                # The contact-aware predictor's forces and state replace the
                # instantaneous applied-force warm start for online clients.
                vectors = np.asarray(selection['force_vectors_local']).copy()
                normals = output[1]
                coefficients = np.zeros((2, 3))
                for side, normal in enumerate(normals):
                    helper = np.array([1., 0., 0.]) if abs(normal[0]) <= .9 else np.array([0., 1., 0.])
                    tangent = np.cross(normal, helper)
                    tangent /= np.linalg.norm(tangent)
                    coefficients[side] = [normal@vectors[side], tangent@vectors[side], np.cross(normal, tangent)@vectors[side]]
                info = output[4]
                info['applied_force_warm_start_pose_cost'] = info['pose_cost']
                info['applied_force_warm_start_cost_reduction'] = info['cost_reduction']
                info.update(selection_backend='coupled_soft_scm', force_vectors_local=vectors,
                    contact_forces_local=coefficients, control_input=coefficients,
                    x_plus_opt=selection['x_plus'].copy(), resulting_pose=selection['x_plus'].copy(),
                    velocity_next=selection['velocity_next'].copy(), contact_model='coupled_soft_scm',
                    pose_cost=selection['next_position_cost']+selection['next_orientation_cost'],
                    cost_reduction=selection['pose_cost_now']-selection['next_position_cost']-selection['next_orientation_cost'],
                    cost=float(selection['cost']))
                if 'moment_vectors_local' in selection:
                    info['moment_vectors_local'] = np.asarray(selection['moment_vectors_local']).copy()
                self.last_grasp_result.update(info, witness_contact_forces_local=coefficients.copy(),
                    witness_force_vectors_local=vectors.copy(), total_cost=float(selection['cost']))
                output = (output[0], output[1], selection['x_plus'].copy(), float(selection['cost']), info)
        return output

    def update_contacts(self, x_d, current_x, *, tips_world, object_inertia,
                        v_obj=None, tau_o=None, prediction_dt=None, lock_contact_pair=False,
                        freeze_contact_pair=False, joint_verify=False):
        if not self._initialized:
            return self.initialize_offline(x_d, current_x, tips_world=tips_world,
                object_inertia=object_inertia, v_obj=v_obj, tau_o=tau_o, prediction_dt=prediction_dt)
        if freeze_contact_pair:
            return self.update_fixed_contacts(x_d, current_x, tips_world=tips_world,
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
        if joint_verify:
            # The existing pair remains the comparison baseline even outside
            # the search cones. Losing a cone alone does not authorize a switch.
            wp.launch(self.k.retain_incumbent, dim=1, inputs=[batch.valid], device=self.device)
        result = self._solve(batch, pose, target, object_inertia=object_inertia,
                             v_obj=v_obj, tau_o=tau_o, prediction_dt=dt)
        wp.launch(self.k.select_online, dim=1, inputs=[result.reductions, result.valid, batch.points,
            self._tips, self._current, self._target, self.pos_coef, self.ori_coef,
            self.quality_fraction, self.switch_margin, self._choice, self._diagnostics], device=self.device)
        if joint_verify:
            wp.launch(self.k.verify_pair, dim=1, inputs=[batch.points, self._tips,
                result.reductions, self.switch_margin, self._choice, self._verification], device=self.device)
        if lock_contact_pair:
            wp.launch(self.k.keep_loaded_pair, dim=1, inputs=[result.valid, self._choice], device=self.device)
        chosen = int(self._choice.numpy()[0])
        if chosen < 0 or not bool(result.valid.numpy()[chosen]):
            # Live cone/support filtering may reject every row, including
            # row 0. Preserve the current assignment rather than inventing a
            # new pair or returning a stale pose/force prediction.
            output = self._update_pair_forces(self._active_ids.numpy(), x_d, current_x,
                tips_world=tips_world, object_inertia=object_inertia, v_obj=v_obj,
                tau_o=tau_o, prediction_dt=dt)
            output[4].update(source="incumbent_fallback", selection_failed=True,
                selection_failure_reason="no_admissible_online_pair", contact_pair_frozen=False,
                pair_switch_approved=False, rejected_candidate_capacity=batch.capacity,
                solve_time=time.perf_counter() - started)
            self.last_grasp_result.update(output[4])
            return output
        previous_ids = self._active_ids.numpy().copy() if joint_verify else None
        self.online_solve_count += 1
        output = self._result(batch, result, dt, started, False, locked=lock_contact_pair)
        if joint_verify:
            verification = self._verification.numpy().astype(float)
            switched = not np.array_equal(previous_ids, output[4]["contact_indices"])
            output[4].update(verify_cost=verification[:2],
                             pair_switch_approved=bool(verification[2]) and switched,
                             contact_pair_frozen=False)
            self.last_grasp_result.update(output[4])
        return output

    def update_fixed_contacts(self, x_d, current_x, *, tips_world, object_inertia,
                              v_obj=None, tau_o=None, prediction_dt=None):
        """Reoptimize forces at the initial assigned pair without reselection.

        Cone membership and measured contact loss cannot change either anchor
        or the left/right assignment in this diagnostic mode.
        """
        if not self._initialized:
            return self.initialize_offline(x_d, current_x, tips_world=tips_world,
                object_inertia=object_inertia, v_obj=v_obj, tau_o=tau_o, prediction_dt=prediction_dt)
        return self._update_pair_forces(self._offline_ids.numpy(), x_d, current_x,
            tips_world=tips_world, object_inertia=object_inertia, v_obj=v_obj,
            tau_o=tau_o, prediction_dt=prediction_dt)

    def _update_pair_forces(self, ids, x_d, current_x, *, tips_world, object_inertia,
                            v_obj=None, tau_o=None, prediction_dt=None):
        """Refresh forces/prediction at an already assigned, ordered pair."""
        from planning.contact_candidates_warp import PairCandidateBatch
        started = time.perf_counter()
        dt = self.h if prediction_dt is None else float(prediction_dt)
        pose, target, _, _ = self._inputs(x_d, current_x, tips_world)
        wp = self.k.wp
        ids = np.asarray(ids, dtype=int).reshape(2).copy()
        if self._fixed is None or not np.array_equal(self._fixed.ids.numpy()[0], ids):
            self._fixed = PairCandidateBatch(
                wp.array(self.sample_point[ids], dtype=wp.vec3, device=self.device),
                wp.array(self.normal[ids], dtype=wp.vec3, device=self.device),
                wp.array(ids.reshape(1, 2), dtype=int, device=self.device),
                wp.ones(1, dtype=int, device=self.device),
                wp.zeros(1, dtype=float, device=self.device), 1)
        result = self._solve(self._fixed, pose, target, object_inertia=object_inertia,
                             v_obj=v_obj, tau_o=tau_o, prediction_dt=dt)
        if not bool(result.valid.numpy()[0]):
            raise FloatingPointError("Lambda force optimization failed for assigned pair %s" % ids.tolist())
        self._choice.zero_()
        self._diagnostics.zero_()
        self.online_solve_count += 1
        points, normals, predicted, cost, info = self._result(
            self._fixed, result, dt, started, False)
        info.update(source="fixed_pair", contact_pair_locked=True, contact_pair_frozen=True,
                    verify_cost=np.zeros(2), pair_switch_approved=False)
        self.last_grasp_result.update(info)
        return points, normals, predicted, cost, info


class LambdaContactMultiLM(LambdaContactBatchedLM):
    """n-contact policy using the very same Warp force LM as the two-pad policy.

    Bounded candidate generation and assignment run on the host; dynamics,
    force projection and every LM iteration run on the requested Warp device.
    Contacts remain in fingertip order, including when their forces are updated.
    """

    def __init__(self, *args, num_contacts=4, candidate_capacity=256, seed=0,
                 require_force_closure=False, **kwargs):
        super().__init__(*args, num_contacts=num_contacts, seed=seed, **kwargs)
        self.num_contacts = int(num_contacts)
        self.candidate_capacity = int(candidate_capacity)
        if self.num_contacts < 1 or self.candidate_capacity < 1:
            raise ValueError("Contact count and candidate capacity must be positive")
        self.rng = np.random.default_rng(seed)
        self.active_ids = self.offline_ids = None
        self.last_batch_ids = None
        self.require_force_closure = bool(require_force_closure)
        self._closure_cache = {}

    def reset(self):
        self.lm.reset()
        self.active_ids = self.offline_ids = None
        self.last_grasp_result = None
        self.offline_solve_count = self.online_solve_count = 0

    def _inputs(self, x_d, current_x, tips_world):
        from scipy.spatial.transform import Rotation
        pose = np.asarray(current_x, float).reshape(7).copy()
        target = np.asarray(x_d, float).reshape(7).copy()
        tips = np.asarray(tips_world, float).reshape(self.num_contacts, 3)
        if not all(np.isfinite(a).all() for a in (pose, target, tips)):
            raise ValueError("Lambda poses and fingertips must be finite")
        for q in (pose[3:], target[3:]):
            length = np.linalg.norm(q)
            if length < 1e-12:
                raise ValueError("Lambda quaternion must be nonzero")
            q /= length
        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        return pose, target, (tips - pose[:3]) @ rotation, rotation

    def _candidate_ids(self, pose, rotation, tips, *, offline):
        from scipy.optimize import linear_sum_assignment
        points, normals = self.sample_point, self.normal
        eligible = np.ones(self.sample_num, dtype=bool)
        if self.support_surface_point is not None:
            centers = pose[:3] + (points - self.tip_radius * normals) @ rotation.T
            clearance = (centers - self.support_surface_point) @ self.support_surface_normal - self.tip_radius
            eligible &= clearance >= self.support_surface_clearance
        pools = []
        for tip in tips:
            delta = points - tip
            distance = np.linalg.norm(delta, axis=1)
            keep = eligible.copy()
            if not offline:
                heading = self.surface.geometric_center_local - tip
                length = np.linalg.norm(heading)
                if length > 1e-8:
                    along = delta @ (heading / length)
                    keep &= (along > 0) & (along <= length + self.surface.object_radius)
                    keep &= along / np.maximum(distance, 1e-10) >= np.cos(np.deg2rad(self.cone_half_angle_deg))
                    keep &= np.einsum("ij,ij->i", -normals, tip - points) > 0
                else:
                    keep[:] = False
            ids = np.flatnonzero(keep)
            ids = ids[np.argsort(distance[ids])]
            pools.append(ids if offline else ids[:self.max_per_arm])
        candidates = []
        if self.active_ids is not None:
            candidates.append(self.active_ids.copy())
        if any(len(pool) == 0 for pool in pools):
            if candidates:
                return np.asarray(candidates)
            raise RuntimeError("No admissible Lambda surface patches above the support")
        if offline:
            ids = pools[0]
            if len(ids) < self.num_contacts:
                raise RuntimeError("Too few surface patches for the requested contacts")
            # Also consider the nearest distinct assignment as a reachable seed.
            _, columns = linear_sum_assignment(np.linalg.norm(tips[:, None] - points[ids], axis=2))
            candidates.append(ids[columns])
        attempts = max(self.candidate_capacity * 20, 100)
        seen = {tuple(row) for row in candidates}
        for _ in range(attempts):
            if len(candidates) >= self.candidate_capacity:
                break
            if offline:
                row = self.rng.choice(pools[0], self.num_contacts, replace=False)
                _, assignment = linear_sum_assignment(np.linalg.norm(tips[:, None] - points[row], axis=2))
                row = row[assignment]
            else:
                row = np.array([self.rng.choice(pool) for pool in pools])
            if len(set(row)) != self.num_contacts or tuple(row) in seen:
                continue
            distances = np.linalg.norm(points[row, None] - points[row][None], axis=2)
            distances += np.eye(self.num_contacts) * 1e6
            if distances.min() < 2 * self.tip_radius:
                continue
            candidates.append(row)
            seen.add(tuple(row))
        if not candidates:
            raise RuntimeError("No separated Lambda contact set is available")
        return np.asarray(candidates, dtype=int)

    def _solve_multi(self, ids, x_d, current_x, *, tips_world, object_inertia,
                     v_obj=None, tau_o=None, prediction_dt=None, offline=False):
        started = time.perf_counter()
        pose, target, tips, rotation = self._inputs(x_d, current_x, tips_world)
        ids = np.asarray(ids, int).reshape(-1, self.num_contacts)
        if np.any(ids < 0) or np.any(ids >= self.sample_num):
            raise ValueError("Contact indices must index the sampled surface")
        dt = self.h if prediction_dt is None else float(prediction_dt)
        self.last_batch_ids = ids.copy()
        batch = self.lm.solve_device(self.sample_point[ids], self.normal[ids], current_x=pose,
            x_d=target, object_inertia=object_inertia, v_obj=v_obj, tau_o=tau_o,
            prediction_dt=dt)
        reductions, valid = batch.reductions.numpy(), batch.valid.numpy().astype(bool)
        closure = np.ones(len(ids), dtype=bool)
        if self.require_force_closure:
            from planning.grasp_verification import verify_grasp
            for i, row in enumerate(ids):
                key = tuple(sorted(row.tolist()))
                if key not in self._closure_cache:
                    self._closure_cache[key] = verify_grasp(self.sample_point[row], self.normal[row], self.mu_arm_obj)
                closure[i] = self._closure_cache[key]["force_closure"]
            valid &= closure
        self.last_candidate_force_closure = closure
        if not valid.any():
            raise FloatingPointError("No valid n-contact Lambda result with the required force closure")
        best = np.max(reductions[valid])
        margin = max(self.quality_fraction * abs(float(best)), 1e-9)
        quality = valid & (reductions >= best - margin)
        distances = np.linalg.norm(self.sample_point[ids] - tips[None], axis=2)
        geometry = distances.sum(axis=1) + self.curvature_penalty_weight * self.surface.curvature_mean[ids].sum(axis=1)
        # Allegro has one thumb opposing a bank of fingers. Expose the
        # quality shortlist so a caller can check its coupled kinematics.
        self.last_candidate_quality = quality.copy()
        self.last_candidate_reductions = reductions.copy()
        self.last_candidate_valid = valid.copy()
        self.last_candidate_geometry = geometry.copy()
        selected = int(np.argmin(np.where(quality, geometry, np.inf)))
        if not offline and self.active_ids is not None and len(ids) > 1:
            # A shorter total reach must not hide a loss for another fingertip.
            gains = distances[0] - distances[selected]
            if reductions[selected] < reductions[0] - 1e-9 or np.any(gains < -self.switch_margin):
                selected = 0
        result = self.lm.gather_selected(selected, batch=batch)
        self.active_ids = ids[selected].copy()
        if offline:
            self.offline_ids = self.active_ids.copy()
            self.offline_solve_count += 1
        else:
            self.online_solve_count += 1
        info = dict(result, contact_indices=self.active_ids.copy(),
            contact_points=result["points"].copy(), contact_normals=result["normals"].copy(),
            witness_contact_forces_local=result["control_input"].copy(),
            witness_force_vectors_local=result["force_vectors_local"].copy(),
            total_cost=result["cost"], cost_reduction=result["pose_reduction"],
            candidate_capacity=len(ids), valid=True,
            source="offline_global" if offline else "online_cones",
            solve_time=time.perf_counter() - started, prediction_dt=dt)
        self.last_grasp_result = info
        if self.require_force_closure:
            info["force_closure_certificate"] = self._closure_cache[tuple(sorted(self.active_ids.tolist()))].copy()
        if self.enable_timing_prints:
            print(f"lambda LM n={self.num_contacts}: candidates={len(ids)} time={info['solve_time']*1000:.1f}ms")
        return result["points"], result["normals"], result["x_plus_opt"], result["cost"], info

    def initialize_offline(self, x_d, current_x, *, tips_world, **kwargs):
        pose, _, tips, rotation = self._inputs(x_d, current_x, tips_world)
        ids = self._candidate_ids(pose, rotation, tips, offline=True)
        return self._solve_multi(ids, x_d, current_x, tips_world=tips_world, offline=True, **kwargs)

    def update_contacts(self, x_d, current_x, *, tips_world, freeze_contact_pair=True,
                        lock_contact_pair=False, joint_verify=True, **kwargs):
        if self.active_ids is None:
            return self.initialize_offline(x_d, current_x, tips_world=tips_world, **kwargs)
        if freeze_contact_pair or lock_contact_pair:
            ids = self.active_ids[None]
        else:
            pose, _, tips, rotation = self._inputs(x_d, current_x, tips_world)
            ids = self._candidate_ids(pose, rotation, tips, offline=False)
        return self._solve_multi(ids, x_d, current_x, tips_world=tips_world, **kwargs)

    def update_fixed_contacts(self, contact_indices, x_d, current_x, *, tips_world, **kwargs):
        return self._solve_multi(np.asarray(contact_indices).reshape(1, self.num_contacts),
            x_d, current_x, tips_world=tips_world, **kwargs)
