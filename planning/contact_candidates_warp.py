"""Original-mesh surface patches and Warp contact-pair candidate batches.

Mesh loading, area sampling and local patch analysis run once on the host.
All per-cycle cone, facing, support-plane and pair filtering runs in Warp.
Positions and inward normals are in the object frame throughout this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import warp as wp

    @wp.func
    def point_clearance(point: wp.vec3, inward: wp.vec3, support_point: wp.vec3,
                        support_normal: wp.vec3, radius: float) -> float:
        # Contact is on the surface; the spherical pad centre is outside it.
        return wp.dot(point - radius * inward - support_point, support_normal) - radius

    @wp.kernel(enable_backward=False)
    def point_filter(points: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
                     eligible: wp.array(dtype=int), tips: wp.array(dtype=wp.vec3),
                     center: wp.vec3, object_radius: float, cone_cos: float,
                     use_cones: int, use_support: int,
                     support_plane: wp.array(dtype=wp.vec3), pad_radius: float, clearance: float,
                     scores: wp.array2d(dtype=float), accepted: wp.array2d(dtype=int)):
        arm, index = wp.tid()
        point, inward = points[index], normals[index]
        keep = eligible[index] != 0
        score = float(0.0)
        if use_support != 0:
            support_point = support_plane[0]
            support_normal = support_plane[1]
            normal_length = wp.length(support_normal)
            finite = bool(True)
            for axis in range(3):
                finite = finite and wp.isfinite(support_point[axis]) and wp.isfinite(support_normal[axis])
            if not finite or normal_length < 1.0e-8:
                keep = False
            else:
                keep = keep and point_clearance(point, inward, support_point,
                                               support_normal / normal_length, pad_radius) >= clearance
        if use_cones != 0:
            delta = point - tips[arm]
            heading = center - tips[arm]
            length = wp.length(heading)
            distance = wp.length(delta)
            if length <= 1.0e-8 or distance <= 1.0e-8:
                keep = False
            else:
                along = wp.dot(delta, heading / length)
                keep = keep and along > 0.0 and along <= length + object_radius
                keep = keep and along / distance >= cone_cos
                # A cone also contains the far side. Reject patches facing
                # away from this arm rather than crossing through the object.
                keep = keep and wp.dot(-inward, tips[arm] - point) > 0.0
            score = wp.length_sq(delta)
        accepted[arm, index] = int(keep)
        scores[arm, index] = score

    @wp.kernel(enable_backward=False)
    def nearest_per_arm(scores: wp.array2d(dtype=float), accepted: wp.array2d(dtype=int),
                        selected: wp.array2d(dtype=int)):
        arm = wp.tid()
        for rank in range(selected.shape[1]):
            best = int(-1)
            best_score = float(1.0e30)
            for index in range(scores.shape[1]):
                used = bool(False)
                for earlier in range(rank):
                    used = used or selected[arm, earlier] == index
                if accepted[arm, index] != 0 and not used:
                    value = scores[arm, index]
                    if value < best_score:
                        best, best_score = index, value
            selected[arm, rank] = best

    @wp.kernel(enable_backward=False)
    def online_ids(selected: wp.array2d(dtype=int), ids: wp.array2d(dtype=int)):
        pair = wp.tid()
        width = selected.shape[1]
        ids[pair, 0] = selected[0, pair // width]
        ids[pair, 1] = selected[1, pair % width]

    @wp.kernel(enable_backward=False)
    def pack_pairs(source_points: wp.array(dtype=wp.vec3),
                   source_normals: wp.array(dtype=wp.vec3),
                   ids: wp.array2d(dtype=int), accepted: wp.array2d(dtype=int),
                   scores: wp.array2d(dtype=float), min_separation: float,
                   friction_cos: float, points: wp.array(dtype=wp.vec3),
                   normals: wp.array(dtype=wp.vec3), valid: wp.array(dtype=int),
                   approach: wp.array(dtype=float)):
        pair = wp.tid()
        first, second = ids[pair, 0], ids[pair, 1]
        valid[pair] = 0
        approach[pair] = 1.0e30
        points[2 * pair] = wp.vec3(0.0)
        points[2 * pair + 1] = wp.vec3(0.0)
        normals[2 * pair] = wp.vec3(0.0)
        normals[2 * pair + 1] = wp.vec3(0.0)
        if first >= 0 and second >= 0 and first != second:
            p0, p1 = source_points[first], source_points[second]
            n0, n1 = source_normals[first], source_normals[second]
            points[2 * pair] = p0
            points[2 * pair + 1] = p1
            normals[2 * pair] = n0
            normals[2 * pair + 1] = n1
            chord = p1 - p0
            distance = wp.length(chord)
            if accepted[0, first] != 0 and accepted[1, second] != 0 and distance >= min_separation:
                axis = chord / wp.max(distance, 1.0e-8)
                if wp.dot(n0, axis) >= friction_cos and wp.dot(n1, -axis) >= friction_cos:
                    valid[pair] = 1
                    approach[pair] = scores[0, first] + scores[1, second]

    _KERNELS = SimpleNamespace(wp=wp, point_filter=point_filter,
                              nearest_per_arm=nearest_per_arm,
                              online_ids=online_ids, pack_pairs=pack_pairs)
    return _KERNELS


@dataclass
class PairCandidateBatch:
    """Fixed-capacity pair batch accepted directly by the Warp lambda solver."""

    points: object
    normals: object
    ids: object
    valid: object
    approach_cost: object
    capacity: int

    def to_host(self) -> dict:
        return dict(points=self.points.numpy().reshape(self.capacity, 2, 3),
                    normals=self.normals.numpy().reshape(self.capacity, 2, 3),
                    ids=self.ids.numpy(), valid=self.valid.numpy().astype(bool),
                    approach_cost=self.approach_cost.numpy())

    @property
    def count(self) -> int:
        """Number of admissible pairs; querying this copies the small mask."""
        return int(np.count_nonzero(self.valid.numpy()))


def _area_samples(mesh, count, rng):
    """One random draw per equal-area stratum, independent of triangulation."""
    area = np.asarray(mesh.area_faces, np.float64)
    cumulative = np.cumsum(area)
    strata = (np.arange(count) + rng.random(count)) * cumulative[-1] / count
    faces = np.searchsorted(cumulative, strata, side="right")
    root = np.sqrt(rng.random(count))
    second = rng.random(count)
    barycentric = np.column_stack((1.0 - root, root * (1.0 - second), root * second))
    points = np.einsum("ni,nij->nj", barycentric, mesh.triangles[faces])
    permutation = rng.permutation(count)
    return points[permutation], faces[permutation]


def _patch_statistics(mesh, points, faces, dense_points, dense_faces, radius):
    """Physical-radius normal variation on a connected mesh patch.

    A Euclidean neighbourhood alone confuses opposite sides of thin plates.
    Flooding only across mesh edges within the footprint keeps those sides
    distinct, while retaining real nearby creases and corners.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(dense_points)
    neighbours = tree.query_ball_point(points, radius)
    adjacency = [[] for _ in mesh.faces]
    for pair, edge in zip(mesh.face_adjacency, mesh.face_adjacency_edges):
        a, b = map(int, pair)
        edge = tuple(map(int, edge))
        adjacency[a].append((b, edge))
        adjacency[b].append((a, edge))
    vertices = np.asarray(mesh.vertices)
    face_normals = np.asarray(mesh.face_normals)
    area_per_sample = float(mesh.area) / len(dense_points)
    maximum = np.zeros(len(points))
    mean = np.zeros(len(points))
    support = np.zeros(len(points))
    radius_sq = radius * radius
    for index, (point, face, nearby) in enumerate(zip(points, faces, neighbours)):
        reached = {int(face)}
        frontier = [int(face)]
        while frontier:
            for other, edge in adjacency[frontier.pop()]:
                if other in reached:
                    continue
                start, end = vertices[list(edge)]
                delta = end - start
                fraction = np.clip((point - start) @ delta / max(delta @ delta, 1.e-20), 0.0, 1.0)
                if np.sum((start + fraction * delta - point) ** 2) <= radius_sq:
                    reached.add(other)
                    frontier.append(other)
        nearby = np.asarray(nearby, np.int64)
        nearby = nearby[np.fromiter((int(dense_faces[item]) in reached for item in nearby), bool,
                                    count=len(nearby))]
        if len(nearby):
            dot = np.clip(face_normals[dense_faces[nearby]] @ face_normals[face], -1.0, 1.0)
            disagreement = 0.5 * (1.0 - dot)
            maximum[index] = float(disagreement.max())
            mean[index] = float(disagreement.mean())
            support[index] = len(nearby) * area_per_sample
        else:
            maximum[index], mean[index] = 1.0, 1.0
    return maximum, mean, support


class SurfaceContactCandidates:
    """Area-balanced low-curvature candidates on the original scaled mesh."""

    def __init__(self, mesh_path, scale_factors=(1.0, 1.0, 1.0), *, sample_num=70,
                 seed=0, patch_radius=0.01, max_point_curvature=0.25,
                 max_mean_curvature=0.12, min_patch_area_ratio=0.30,
                 dense_sample_num=8192, device="cuda:0", body_sides=False,
                 body_core_fraction=0.5, body_side_normal_cos=0.85):
        import trimesh

        if sample_num < 2 or dense_sample_num < 32:
            raise ValueError("sample_num must be at least two; dense_sample_num at least 32")
        if not np.isfinite(patch_radius) or patch_radius <= 0:
            raise ValueError("patch_radius must be positive and finite")
        if not 0.0 <= max_point_curvature <= 1.0 or not 0.0 <= max_mean_curvature <= 1.0:
            raise ValueError("curvature thresholds must be in [0, 1]")
        if not np.isfinite(min_patch_area_ratio) or not 0.0 < min_patch_area_ratio <= 1.0:
            raise ValueError("min_patch_area_ratio must be in (0, 1]")
        if not 0 < body_core_fraction <= 1 or not 0 < body_side_normal_cos <= 1:
            raise ValueError("body core fraction and side normal cosine must be in (0,1]")
        scale = np.asarray(scale_factors, np.float64)
        if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("scale_factors must contain three positive finite scales")
        mesh = mesh_path.copy() if isinstance(mesh_path, trimesh.Trimesh) else trimesh.load_mesh(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        mesh.apply_scale(scale)
        mesh.update_faces(np.asarray(mesh.area_faces) > 1.e-16)
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals(multibody=True)
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all() or mesh.area <= 0:
            raise ValueError("mesh must contain a finite, nonempty surface")
        self.mesh = mesh
        self.patch_radius = float(patch_radius)
        self.geometric_center_local = np.average(mesh.triangles_center, axis=0, weights=mesh.area_faces)
        self.object_radius = float(np.linalg.norm(mesh.vertices - self.geometric_center_local, axis=1).max())
        rng = np.random.default_rng(seed)
        dense_count = max(int(dense_sample_num), 64 * int(sample_num))
        dense_points, dense_faces = _area_samples(mesh, dense_count, rng)
        self.dense_points_local = dense_points.astype(np.float32)
        self.dense_normals_local = (-mesh.face_normals[dense_faces]).astype(np.float32)
        candidate_count = max((16 if body_sides else 8) * int(sample_num), 512)
        candidates, candidate_faces = _area_samples(mesh, candidate_count, rng)
        maximum, mean, support = _patch_statistics(
            mesh, candidates, candidate_faces, dense_points, dense_faces, self.patch_radius)
        support_threshold = float(min_patch_area_ratio) * np.pi * self.patch_radius ** 2
        eligible = ((maximum <= max_point_curvature) & (mean <= max_mean_curvature)
                    & (support >= support_threshold))
        self.body_sides = bool(body_sides)
        self.body_long_axis = self.body_side_axis = None
        self.body_core_half_length = 0.0
        region = np.ones(len(candidates), bool)
        if body_sides:
            # Upright mesh: infer the longitudinal axis from area-weighted
            # horizontal geometry, then keep the broad transverse flanks.
            # Curvature alone also accepts a rounded head or tail root.
            xy = mesh.triangles_center[:, :2] - self.geometric_center_local[:2]
            covariance = np.einsum("n,ni,nj->ij", mesh.area_faces, xy, xy) / mesh.area
            _, axes = np.linalg.eigh(covariance)
            long_axis = np.r_[axes[:, -1], 0.0]
            if long_axis[np.argmax(np.abs(long_axis))] < 0:
                long_axis = -long_axis
            side_axis = np.array([-long_axis[1], long_axis[0], 0.0])
            relative_vertices = mesh.vertices - self.geometric_center_local
            half_length = np.max(np.abs(relative_vertices @ long_axis))
            half_width = np.max(np.abs(relative_vertices @ side_axis))
            displacement = candidates - self.geometric_center_local
            lateral = displacement @ side_axis
            outward_alignment = mesh.face_normals[candidate_faces] @ side_axis
            self.body_long_axis, self.body_side_axis = long_axis, side_axis
            self.body_core_half_length = float(body_core_fraction * half_length)
            region = ((np.abs(displacement @ long_axis) <= self.body_core_half_length)
                      & (np.abs(outward_alignment) >= body_side_normal_cos)
                      & (np.abs(lateral) >= 0.35 * half_width)
                      & (lateral * outward_alignment > 0))
            eligible &= region
        usable = np.flatnonzero(eligible)
        if len(usable) < 2:
            raise ValueError("Mesh has fewer than two locally supported low-curvature contact patches")
        # Area strata were shuffled before filtering; truncation retains an
        # unbiased area distribution among admissible patches, not PCA tips.
        chosen = usable[:int(sample_num)]
        self.points_local = candidates[chosen].astype(np.float32)
        self.normals_local = (-mesh.face_normals[candidate_faces[chosen]]).astype(np.float32)
        self.curvature_max = maximum[chosen].astype(np.float32)
        self.curvature_mean = mean[chosen].astype(np.float32)
        self.patch_support_area = support[chosen].astype(np.float32)
        self.eligible = np.ones(len(chosen), np.int32)
        self.sample_num = len(chosen)
        self.preprocessing_diagnostics = dict(
            original_faces=len(mesh.faces), dense_samples=dense_count,
            tested_patches=candidate_count, eligible_patches=len(usable),
            rejected_curvature=int(np.count_nonzero((maximum > max_point_curvature) | (mean > max_mean_curvature))),
            rejected_support=int(np.count_nonzero(support < support_threshold)),
            rejected_body_region=int(np.count_nonzero(~region)),
            body_sides=self.body_sides,
            support_area_threshold=support_threshold)
        k = _kernels()
        k.wp.init()
        self.device = str(k.wp.get_device(device))
        self.points_device = k.wp.array(self.points_local, dtype=k.wp.vec3, device=self.device)
        self.normals_device = k.wp.array(self.normals_local, dtype=k.wp.vec3, device=self.device)
        self.eligible_device = k.wp.array(self.eligible, dtype=int, device=self.device)
        self.dense_points_device = k.wp.array(self.dense_points_local, dtype=k.wp.vec3, device=self.device)
        self.dense_normals_device = k.wp.array(self.dense_normals_local, dtype=k.wp.vec3, device=self.device)
        self._buffers = {}

    def _batch(self, key, ids, width):
        k = _kernels()
        if key not in self._buffers:
            capacity = len(ids)
            batch = PairCandidateBatch(
                points=k.wp.zeros(2 * capacity, dtype=k.wp.vec3, device=self.device),
                normals=k.wp.zeros(2 * capacity, dtype=k.wp.vec3, device=self.device),
                ids=k.wp.array(ids, dtype=int, device=self.device),
                valid=k.wp.zeros(capacity, dtype=int, device=self.device),
                approach_cost=k.wp.zeros(capacity, dtype=float, device=self.device), capacity=capacity)
            self._buffers[key] = SimpleNamespace(
                batch=batch, tips=k.wp.zeros(2, dtype=k.wp.vec3, device=self.device),
                support_plane=k.wp.array([[0., 0., 0.], [0., 0., 1.]], dtype=k.wp.vec3, device=self.device),
                scores=k.wp.zeros((2, self.sample_num), dtype=float, device=self.device),
                accepted=k.wp.zeros((2, self.sample_num), dtype=int, device=self.device),
                selected=k.wp.zeros((2, width), dtype=int, device=self.device))
        return self._buffers[key]

    def _filter(self, buffers, tips, *, cone_half_angle_deg, arm_friction, min_separation,
                support_point_local, support_normal_local, support_plane_device,
                fingertip_radius, support_clearance):
        k = _kernels()
        if not np.isfinite(arm_friction) or arm_friction < 0:
            raise ValueError("arm_friction must be nonnegative and finite")
        if not np.isfinite(min_separation) or min_separation <= 0:
            raise ValueError("min_separation must be positive and finite")
        if not np.isfinite(fingertip_radius) or fingertip_radius < 0 or not np.isfinite(support_clearance):
            raise ValueError("fingertip radius and support clearance must be finite; radius nonnegative")
        use_support = support_normal_local is not None or support_plane_device is not None
        support_normal = np.array([0.0, 0.0, 1.0])
        support_point = np.zeros(3)
        support_plane = buffers.support_plane
        if support_plane_device is not None:
            if support_point_local is not None or support_normal_local is not None:
                raise ValueError("Use either support_plane_device or the host support plane")
            self._device_vec3_input(support_plane_device, 2, "support_plane_device")
            support_plane = support_plane_device
        elif use_support:
            support_normal = np.asarray(support_normal_local, np.float64).reshape(3)
            support_point = np.asarray(support_point_local, np.float64).reshape(3)
            length = np.linalg.norm(support_normal)
            if not np.isfinite(support_normal).all() or not np.isfinite(support_point).all() or length < 1.e-8:
                raise ValueError("support plane must be finite with a nonzero normal")
            support_normal /= length
            buffers.support_plane.assign(np.stack((support_point, support_normal)).astype(np.float32))
        tip_input = buffers.tips
        if tips is not None:
            if isinstance(tips, k.wp.array):
                self._device_vec3_input(tips, 2, "tips_local")
                tip_input = tips
            else:
                tips = np.asarray(tips, np.float32)
                if tips.shape != (2, 3) or not np.isfinite(tips).all():
                    raise ValueError("tips_local must be a finite (2, 3) array")
                buffers.tips.assign(tips)
            if not np.isfinite(cone_half_angle_deg) or not 0 < cone_half_angle_deg < 90:
                raise ValueError("cone_half_angle_deg must be in (0, 90)")
        k.wp.launch(k.point_filter, dim=(2, self.sample_num), inputs=[
            self.points_device, self.normals_device, self.eligible_device, tip_input,
            k.wp.vec3(*self.geometric_center_local), self.object_radius,
            float(np.cos(np.deg2rad(cone_half_angle_deg))), int(tips is not None), int(use_support),
            support_plane, float(fingertip_radius),
            float(support_clearance), buffers.scores, buffers.accepted], device=self.device)
        return 1.0 / np.sqrt(1.0 + arm_friction * arm_friction)

    def _device_vec3_input(self, value, count, name):
        k = _kernels()
        if not isinstance(value, k.wp.array) or value.dtype != k.wp.vec3 or value.shape != (count,):
            raise ValueError(f"{name} must be a Warp vec3 array of shape ({count},)")
        if value.device != k.wp.get_device(self.device):
            raise ValueError(f"{name} must be on device {self.device}")

    def global_pairs(self, *, arm_friction=0.9, min_separation=0.03,
                     support_point_local=None, support_normal_local=None,
                     support_plane_device=None,
                     fingertip_radius=0.01, support_clearance=0.0):
        """Every unordered pair in the finite surface pool, with Warp validity."""
        first, second = np.triu_indices(self.sample_num, 1)
        ids = np.column_stack((first, second)).astype(np.int32)
        buffers = self._batch("global", ids, 1)
        cosine = self._filter(
            buffers, None, cone_half_angle_deg=45.0, arm_friction=arm_friction,
            min_separation=min_separation, support_point_local=support_point_local,
            support_normal_local=support_normal_local, support_plane_device=support_plane_device,
            fingertip_radius=fingertip_radius,
            support_clearance=support_clearance)
        self._pack(buffers, min_separation, cosine)
        return buffers.batch

    def _pack(self, buffers, min_separation, cosine):
        k = _kernels()
        batch = buffers.batch
        k.wp.launch(k.pack_pairs, dim=batch.capacity, inputs=[
            self.points_device, self.normals_device, batch.ids, buffers.accepted, buffers.scores,
            float(min_separation), float(cosine), batch.points, batch.normals, batch.valid,
            batch.approach_cost], device=self.device)

    def online_pairs(self, tips_local, *, cone_half_angle_deg=45.0, max_per_arm=32,
                     max_pairs=1024, arm_friction=0.9, min_separation=0.03,
                     support_point_local=None, support_normal_local=None,
                     support_plane_device=None,
                     fingertip_radius=0.01, support_clearance=0.0):
        """Ordered left/right pairs from finite cones toward the mesh centroid."""
        if int(max_per_arm) < 1 or int(max_pairs) < 1:
            raise ValueError("max_per_arm and max_pairs must be positive")
        width = min(int(max_per_arm), self.sample_num, int(np.sqrt(int(max_pairs))))
        capacity = width * width
        buffers = self._batch(("online", width), np.full((capacity, 2), -1, np.int32), width)
        cosine = self._filter(
            buffers, tips_local, cone_half_angle_deg=cone_half_angle_deg, arm_friction=arm_friction,
            min_separation=min_separation, support_point_local=support_point_local,
            support_normal_local=support_normal_local, support_plane_device=support_plane_device,
            fingertip_radius=fingertip_radius,
            support_clearance=support_clearance)
        k = _kernels()
        k.wp.launch(k.nearest_per_arm, dim=2, inputs=[buffers.scores, buffers.accepted,
                                                   buffers.selected], device=self.device)
        k.wp.launch(k.online_ids, dim=capacity, inputs=[buffers.selected, buffers.batch.ids], device=self.device)
        self._pack(buffers, min_separation, cosine)
        return buffers.batch

    def batch_pairs(self, tips_local=None, **kwargs):
        return self.global_pairs(**kwargs) if tips_local is None else self.online_pairs(tips_local, **kwargs)
