import argparse
import importlib.util
import itertools
import time
from pathlib import Path

import casadi as cs
import numpy as np
import trimesh
from scipy.spatial import cKDTree

try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    from project_point import ProjectionPoint
except ImportError:
    from planning.project_point import ProjectionPoint


OBJECT_ASSET_DIR = Path(__file__).resolve().parents[1] / "envs" / "assets" / "objects"
_COACD_PACKAGE_DIR = Path(__file__).resolve().parents[2] / "thirdparty" / "CoACD" / "python" / "package"
_EPS = 1e-9
_O3D_OBJECT_BASE_COLOR = np.array([0.82, 0.82, 0.85], dtype=np.float64)
_O3D_OBJECT_TRANSPARENCY = 0.30
_O3D_OBJECT_OPACITY = 1.0 - _O3D_OBJECT_TRANSPARENCY
_O3D_BACKGROUND_COLOR = np.array([1.0, 1.0, 1.0], dtype=np.float64)


def _as_numpy(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float64)
    if hasattr(value, "full"):
        return np.asarray(value.full(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _normalize(vec, axis=None):
    vec = np.asarray(vec, dtype=np.float64)
    if axis is None:
        norm = float(np.linalg.norm(vec))
        if norm < _EPS:
            return np.zeros_like(vec)
        return vec / norm

    norm = np.linalg.norm(vec, axis=axis, keepdims=True)
    norm = np.where(norm < _EPS, 1.0, norm)
    return vec / norm


def _farthest_point_sampling(points, num_samples, seed_index=None):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or num_samples <= 0:
        return np.zeros((0,), dtype=int)

    num_samples = min(int(num_samples), int(points.shape[0]))
    selected = np.empty(num_samples, dtype=int)

    if seed_index is None:
        center = np.mean(points, axis=0, keepdims=True)
        seed_index = int(np.argmax(np.linalg.norm(points - center, axis=1)))
    seed_index = int(np.clip(seed_index, 0, points.shape[0] - 1))

    selected[0] = seed_index
    min_dist2 = np.sum((points - points[seed_index]) ** 2, axis=1)
    min_dist2[seed_index] = -1.0

    for i in range(1, num_samples):
        next_idx = int(np.argmax(min_dist2))
        selected[i] = next_idx
        dist2 = np.sum((points - points[next_idx]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
        min_dist2[selected[: i + 1]] = -1.0

    return selected


def _rotation_from_z(direction):
    direction = _normalize(direction)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot_val = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))

    if dot_val > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float64)
    if dot_val < -1.0 + 1e-8:
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )

    axis = np.cross(z_axis, direction)
    axis = _normalize(axis)
    angle = np.arccos(dot_val)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _orthonormal_frame_from_normal(normal):
    normal = _normalize(normal)
    arbitrary_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(arbitrary_dir, normal))) > 0.9:
        arbitrary_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    tangent1 = arbitrary_dir - np.dot(arbitrary_dir, normal) * normal
    tangent1 = _normalize(tangent1)
    tangent2 = _normalize(np.cross(normal, tangent1))
    return normal, tangent1, tangent2


def _build_frames_from_outward_normals(points, outward_normals):
    points = np.asarray(points, dtype=np.float64)
    outward_normals = np.asarray(outward_normals, dtype=np.float64)
    inward_normals = np.zeros_like(points)
    tangent1 = np.zeros_like(points)
    tangent2 = np.zeros_like(points)

    for idx in range(points.shape[0]):
        inward_normal, t1, t2 = _orthonormal_frame_from_normal(-outward_normals[idx])
        inward_normals[idx] = np.asarray(inward_normal, dtype=np.float64)
        tangent1[idx] = np.asarray(t1, dtype=np.float64)
        tangent2[idx] = np.asarray(t2, dtype=np.float64)

    return inward_normals, tangent1, tangent2


class LambdaContactControlOptimizer:
    def __init__(
        self,
        mesh_path,
        obj_mass=0.01,
        arm_friction=0.9,
        contact_stiffness=12.5,
        time_step=0.01,
        max_contacts=10,
        sample_num=70,
        pos_coef=1.0,
        ori_coef=0.0005,
        scale_factors=(1.0, 1.0, 1.0),
        mppi_samples=96,
        mppi_iterations=4,
        mppi_horizon=4,
        mppi_lambda=1.0,
        mppi_noise_sigma=0.01,
        mppi_noise_decay=0.85,
        mppi_elite_frac=0.1,
        neighbor_k=12,
        min_pair_distance=0.015,
        path_tracking_weight=5.0,
        overlap_penalty_weight=1500.0,
        antipodal_penalty_weight=25.0,
        distance_reward_weight=0.6,
        force_reg_weight=1e-5,
        device=None,
        num_grasp_contacts=2,
        region_anchor_count=200,
        region_radius=0.08,
        region_max_points=256,
        region_contact_samples=5,
        top_region_pairs=3,
        preselect_region_pairs=200,
        friction_cone_edges=8,
        gwb_wrench_count=1000,
        beta=1.0,
        gamma=0.2,
        concavity_tol=None,
        normal_consistency_min=0.6,
        accessibility_alignment_min=-0.2,
        max_region_curvature_deg=35.0,
        proxy_preselect_mode="auto",
        coacd_threshold=0.05,
        coacd_max_convex_hull=12,
        coacd_prep_resolution=50,
        proxy_projection_neighbors=8,
        proxy_normal_alignment_min=0.2,
        max_point_combination_eval=256,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=0.0,
        support_surface_normal_alignment_threshold=0.25,
    ):
        self.mesh_path = str(mesh_path)
        self.m = float(obj_mass)
        self.mu_arm_obj = float(arm_friction)
        self.K_contact = float(contact_stiffness)
        self.h = float(time_step)
        self.max_contacts = int(max_contacts)
        self.sample_budget = int(sample_num)
        self.pos_coef = float(pos_coef)
        self.ori_coef = float(ori_coef)
        self.force_reg_weight = float(force_reg_weight)
        self.min_pair_distance = float(min_pair_distance)
        self.overlap_penalty_weight = float(overlap_penalty_weight)
        self.antipodal_penalty_weight = float(antipodal_penalty_weight)
        self.distance_reward_weight = float(distance_reward_weight)

        # Legacy parameters are kept in the signature so the class remains easy to
        # drop into the existing codebase, but the new selection logic no longer
        # relies on the old MPPI rollout.
        self.mppi_samples = int(mppi_samples)
        self.mppi_iterations = int(mppi_iterations)
        self.mppi_horizon = int(mppi_horizon)
        self.mppi_lambda = float(mppi_lambda)
        self.mppi_noise_sigma = float(mppi_noise_sigma)
        self.mppi_noise_decay = float(mppi_noise_decay)
        self.mppi_elite_frac = float(mppi_elite_frac)
        self.neighbor_k = int(neighbor_k)
        self.path_tracking_weight = float(path_tracking_weight)
        self.device_ = device
        self.num_grasp_contacts = max(2, int(num_grasp_contacts))
        self.max_point_combination_eval = max(1, int(max_point_combination_eval))

        self.pp = ProjectionPoint(self.mesh_path, scale_factors)
        self.mesh = self.pp.scaled_mesh
        self.mesh_centroid = np.asarray(self.mesh.centroid, dtype=np.float64)
        bounds = np.asarray(self.mesh.bounds, dtype=np.float64)
        self.mesh_diag = float(np.linalg.norm(bounds[1] - bounds[0]))
        self.mesh_diag = max(self.mesh_diag, 1e-3)
        self.characteristic_length = max(0.5 * self.mesh_diag, 1e-3)
        self.wrench_scale = np.array(
            [1.0, 1.0, 1.0, 1.0 / self.characteristic_length, 1.0 / self.characteristic_length, 1.0 / self.characteristic_length],
            dtype=np.float64,
        )

        self.region_anchor_count = int(region_anchor_count)
        self.region_radius = min(float(region_radius), 0.35 * self.mesh_diag)
        self.region_radius = max(self.region_radius, 0.02 * self.mesh_diag)
        self.region_max_points = int(region_max_points)
        self.region_contact_samples = int(region_contact_samples)
        self.top_region_pairs = int(top_region_pairs)
        self.preselect_region_pairs = int(max(preselect_region_pairs, top_region_pairs))
        self.friction_cone_edges = int(max(4, friction_cone_edges))
        self.gwb_wrench_count = int(max(64, gwb_wrench_count))
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.normal_consistency_min = float(normal_consistency_min)
        self.accessibility_alignment_min = float(accessibility_alignment_min)
        self.max_region_curvature = np.deg2rad(float(max_region_curvature_deg))
        self.proxy_preselect_mode = str(proxy_preselect_mode).strip().lower()
        self.coacd_threshold = float(coacd_threshold)
        self.coacd_max_convex_hull = int(coacd_max_convex_hull)
        self.coacd_prep_resolution = int(coacd_prep_resolution)
        self.proxy_projection_neighbors = max(1, int(proxy_projection_neighbors))
        self.proxy_normal_alignment_min = float(np.clip(proxy_normal_alignment_min, -1.0, 1.0))
        self.concavity_tol = (
            float(concavity_tol) if concavity_tol is not None else 0.25 * self.region_radius
        )
        self.region_pair_min_distance = max(self.min_pair_distance, 0.5 * self.region_radius)
        self.support_surface_clearance = max(0.0, float(support_surface_clearance))
        self.support_surface_normal_alignment_threshold = float(
            np.clip(support_surface_normal_alignment_threshold, 0.0, 1.0)
        )
        self.support_surface_point = None
        self.support_surface_normal = None
        self.set_support_surface(
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
        )

        self.sampling_frame = self._build_surface_frames()
        self.sample_point = np.asarray(self.sampling_frame["points"], dtype=np.float64)
        self.normal = np.asarray(self.sampling_frame["normals"], dtype=np.float64)  # inward normals
        self.t1 = np.asarray(self.sampling_frame["tangent1"], dtype=np.float64)
        self.t2 = np.asarray(self.sampling_frame["tangent2"], dtype=np.float64)
        self.outward_normal = -self.normal
        self.sample_num = int(self.sample_point.shape[0])
        self.point_idx = np.arange(self.sample_num, dtype=int)
        self.sample_kdtree = cKDTree(self.sample_point)
        self.sample_curvature = self._estimate_local_surface_curvature()

        self.convex_hull = self.mesh.convex_hull
        self.convex_hull_points = self._sample_convex_hull_surface()
        self.convex_hull_tree = cKDTree(self.convex_hull_points)

        self.J_tilde = np.zeros((4 * self.max_contacts, 6), dtype=np.float64)
        self.friction_primitives = self._build_friction_primitives()
        self.disturbance_wrenches, self.disturbance_labels = self._build_disturbance_wrenches()

        self.last_region_results = []
        self.last_available_idx = self.point_idx.copy()
        self.last_candidate_point_groups = []
        self.last_ranked_grasp_results = []
        self.last_proxy_preselection_info = {}
        self.last_grasp_result = None
        self.last_static_equilibrium_result = None
        self.last_object_pos = None
        self.last_object_rot = None
        self._coacd_module = None
        self._coacd_module_loaded = False
        self._proxy_surface_cache = None

        self._precompile_optimization_function()
        self._precompile_static_equilibrium_function()

    def _build_surface_frames(self):
        face_vertices = np.asarray(self.mesh.triangles, dtype=np.float64)
        points = np.asarray(face_vertices.mean(axis=1), dtype=np.float64)
        outward_normals = np.asarray(self.pp.face_normals, dtype=np.float64)
        normals, tangent1, tangent2 = _build_frames_from_outward_normals(points, outward_normals)

        return {
            "points": points,
            "normals": normals,
            "tangent1": tangent1,
            "tangent2": tangent2,
        }

    def _estimate_local_surface_curvature(self):
        curvature = np.zeros((self.sample_num,), dtype=np.float64)
        neighborhood_radius = max(float(self.region_radius), 1e-6)

        for idx in range(self.sample_num):
            neighbor_idx = np.asarray(
                self.sample_kdtree.query_ball_point(self.sample_point[idx], r=neighborhood_radius),
                dtype=int,
            ).reshape(-1)
            if neighbor_idx.size <= 1:
                curvature[idx] = 0.0
                continue

            outward_neighbors = self.outward_normal[neighbor_idx]
            mean_outward = _normalize(np.mean(outward_neighbors, axis=0))
            if np.linalg.norm(mean_outward) < 1e-8:
                curvature[idx] = 0.5 * np.pi
                continue

            cos_vals = np.clip(outward_neighbors @ mean_outward, -1.0, 1.0)
            curvature[idx] = float(np.mean(np.arccos(cos_vals)))

        return curvature

    def _load_coacd_module(self):
        if self._coacd_module_loaded:
            return self._coacd_module

        self._coacd_module_loaded = True

        try:
            import coacd  # type: ignore

            self._coacd_module = coacd
            return self._coacd_module
        except Exception:
            pass

        init_path = _COACD_PACKAGE_DIR / "__init__.py"
        has_local_lib = any(_COACD_PACKAGE_DIR.glob("lib_coacd*"))
        if not init_path.exists() or not has_local_lib:
            self._coacd_module = None
            return None

        try:
            spec = importlib.util.spec_from_file_location("_local_coacd_package", init_path)
            if spec is None or spec.loader is None:
                self._coacd_module = None
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._coacd_module = module
            return self._coacd_module
        except Exception:
            self._coacd_module = None
            return None

    def _build_proxy_surface_cache(self):
        if self._proxy_surface_cache is not None:
            return self._proxy_surface_cache

        mode = self.proxy_preselect_mode
        if mode not in {"auto", "coacd", "convex_hull", "off"}:
            mode = "auto"

        proxy_parts = []
        proxy_source = "none"

        if mode in {"auto", "coacd"}:
            coacd_module = self._load_coacd_module()
            if coacd_module is not None:
                try:
                    if hasattr(coacd_module, "set_log_level"):
                        coacd_module.set_log_level("error")
                    coacd_mesh = coacd_module.Mesh(self.mesh.vertices, self.mesh.faces)
                    proxy_result = coacd_module.run_coacd(
                        coacd_mesh,
                        threshold=self.coacd_threshold,
                        max_convex_hull=self.coacd_max_convex_hull,
                        preprocess_mode="auto",
                        preprocess_resolution=self.coacd_prep_resolution,
                    )
                    for vertices, faces in proxy_result:
                        if len(vertices) == 0 or len(faces) == 0:
                            continue
                        proxy_parts.append(
                            trimesh.Trimesh(
                                np.asarray(vertices, dtype=np.float64),
                                np.asarray(faces, dtype=np.int64),
                                process=False,
                            )
                        )
                    if proxy_parts:
                        proxy_source = "coacd"
                except Exception:
                    proxy_parts = []

        if not proxy_parts and mode in {"auto", "coacd", "convex_hull"}:
            proxy_parts = [self.mesh.convex_hull]
            proxy_source = "convex_hull"

        if not proxy_parts:
            self._proxy_surface_cache = {
                "source": "none",
                "parts": [],
                "points": np.zeros((0, 3), dtype=np.float64),
                "outward_normals": np.zeros((0, 3), dtype=np.float64),
            }
            return self._proxy_surface_cache

        proxy_points = []
        proxy_outward_normals = []
        proxy_part_ids = []
        for part_idx, part in enumerate(proxy_parts):
            triangles = np.asarray(part.triangles, dtype=np.float64)
            if triangles.size == 0:
                continue
            face_centers = np.asarray(triangles.mean(axis=1), dtype=np.float64)
            face_normals = np.asarray(part.face_normals, dtype=np.float64)
            if face_centers.shape[0] != face_normals.shape[0]:
                continue
            proxy_points.append(face_centers)
            proxy_outward_normals.append(face_normals)
            proxy_part_ids.append(np.full((face_centers.shape[0],), part_idx, dtype=int))

        if proxy_points:
            proxy_points = np.vstack(proxy_points)
            proxy_outward_normals = np.vstack(proxy_outward_normals)
            proxy_part_ids = np.concatenate(proxy_part_ids)
        else:
            proxy_points = np.zeros((0, 3), dtype=np.float64)
            proxy_outward_normals = np.zeros((0, 3), dtype=np.float64)
            proxy_part_ids = np.zeros((0,), dtype=int)

        self._proxy_surface_cache = {
            "source": proxy_source,
            "parts": proxy_parts,
            "points": np.asarray(proxy_points, dtype=np.float64),
            "outward_normals": np.asarray(proxy_outward_normals, dtype=np.float64),
            "part_ids": np.asarray(proxy_part_ids, dtype=int),
        }
        return self._proxy_surface_cache

    def _project_proxy_points_to_original_indices(self, base_candidate_idx):
        base_candidate_idx = self._sanitize_point_indices(base_candidate_idx)
        if base_candidate_idx.size == 0:
            self.last_proxy_preselection_info = {
                "enabled": False,
                "source": "none",
                "base_candidate_count": 0,
                "projected_candidate_count": 0,
            }
            return base_candidate_idx

        proxy_cache = self._build_proxy_surface_cache()
        proxy_points = np.asarray(proxy_cache["points"], dtype=np.float64)
        proxy_outward_normals = np.asarray(proxy_cache["outward_normals"], dtype=np.float64)
        if proxy_points.shape[0] == 0:
            self.last_proxy_preselection_info = {
                "enabled": False,
                "source": proxy_cache["source"],
                "base_candidate_count": int(base_candidate_idx.size),
                "projected_candidate_count": int(base_candidate_idx.size),
            }
            return base_candidate_idx

        candidate_points = self.sample_point[base_candidate_idx]
        candidate_outward = self.outward_normal[base_candidate_idx]
        candidate_tree = cKDTree(candidate_points)
        neighbor_k = min(self.proxy_projection_neighbors, base_candidate_idx.size)
        query_dist, query_local_idx = candidate_tree.query(proxy_points, k=neighbor_k)
        if neighbor_k == 1:
            query_dist = query_dist[:, None]
            query_local_idx = query_local_idx[:, None]

        projected_indices = []
        distance_scale = max(self.mesh_diag, 1e-6)
        for proxy_idx in range(proxy_points.shape[0]):
            local_neighbors = np.asarray(query_local_idx[proxy_idx], dtype=int).reshape(-1)
            neighbor_dist = np.asarray(query_dist[proxy_idx], dtype=np.float64).reshape(-1)
            original_idx = base_candidate_idx[local_neighbors]
            normal_alignment = np.clip(candidate_outward[local_neighbors] @ proxy_outward_normals[proxy_idx], -1.0, 1.0)

            preferred_mask = normal_alignment >= self.proxy_normal_alignment_min
            if np.any(preferred_mask):
                original_idx = original_idx[preferred_mask]
                neighbor_dist = neighbor_dist[preferred_mask]
                normal_alignment = normal_alignment[preferred_mask]

            score = neighbor_dist / distance_scale + 0.25 * (1.0 - normal_alignment)
            best_idx = int(original_idx[int(np.argmin(score))])
            projected_indices.append(best_idx)

        projected_indices = np.unique(np.asarray(projected_indices, dtype=int))
        if projected_indices.size < self.num_grasp_contacts:
            projected_indices = base_candidate_idx

        self.last_proxy_preselection_info = {
            "enabled": True,
            "source": proxy_cache["source"],
            "base_candidate_count": int(base_candidate_idx.size),
            "proxy_face_count": int(proxy_points.shape[0]),
            "projected_candidate_count": int(projected_indices.size),
        }
        return projected_indices

    def _sample_convex_hull_surface(self):
        target_count = int(np.clip(max(1024, 2 * len(self.convex_hull.vertices)), 1024, 4096))
        try:
            hull_points, _ = trimesh.sample.sample_surface_even(self.convex_hull, target_count)
        except Exception:
            hull_points = np.asarray(self.convex_hull.vertices, dtype=np.float64)
        return np.asarray(hull_points, dtype=np.float64)

    @staticmethod
    def _get_solver_config():
        p_opts = {"print_time": False, "jit": False}
        solver_name = "ipopt"
        s_opts = {
            "max_iter": 300,
            "tol": 1e-6,
            "acceptable_tol": 1e-5,
            "linear_solver": "mumps",
            "print_level": 0,
            "sb": "yes",
        }
        return solver_name, p_opts, s_opts

    def _configure_solver(self, opti):
        solver_name, p_opts, s_opts = self._get_solver_config()
        opti.solver(solver_name, p_opts, s_opts)

    def update_Jacobian(self, J_tilde=None):
        required_rows = 4 * self.max_contacts
        if J_tilde is None:
            return self.J_tilde

        J_tilde = np.asarray(J_tilde, dtype=np.float64)[:, :6]
        padded = np.zeros((required_rows, 6), dtype=np.float64)
        valid_rows = min(required_rows, J_tilde.shape[0])
        padded[:valid_rows] = J_tilde[:valid_rows]
        self.J_tilde = padded
        return self.J_tilde

    def _build_friction_primitives(self):
        theta = np.linspace(0.0, 2.0 * np.pi, self.friction_cone_edges, endpoint=False, dtype=np.float64)
        tangential = self.mu_arm_obj * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        primitives = np.column_stack([np.ones_like(theta), tangential])
        return np.asarray(primitives, dtype=np.float64)

    def _build_disturbance_wrenches(self):
        base = np.eye(6, dtype=np.float64)
        disturbances = np.vstack([base, -base])
        labels = [
            "+Fx",
            "+Fy",
            "+Fz",
            "+Tx",
            "+Ty",
            "+Tz",
            "-Fx",
            "-Fy",
            "-Fz",
            "-Tx",
            "-Ty",
            "-Tz",
        ]
        return disturbances, labels

    def _precompile_optimization_function(self):
        opti = cs.Opti()

        n_dist = int(self.disturbance_wrenches.shape[0])
        G = opti.parameter(6, 3 * self.num_grasp_contacts)
        f = opti.variable(3 * self.num_grasp_contacts, n_dist)
        disturbance_matrix = cs.DM(self.disturbance_wrenches.T)

        opti.set_initial(f, 0.1)

        cost_terms = []
        response_terms = []
        objective = 0

        for j in range(n_dist):
            wrench_response = G @ f[:, j]
            residual = self.beta * disturbance_matrix[:, j] - wrench_response
            response_terms.append(wrench_response)
            cost_terms.append(cs.sumsqr(residual))
            objective += cost_terms[-1]

            normal_force_sum = 0
            for contact_idx in range(self.num_grasp_contacts):
                f_contact = f[3 * contact_idx : 3 * (contact_idx + 1), j]
                self._add_friction_cone_constraints(opti, f_contact)
                normal_force_sum += f_contact[0]
            opti.subject_to(normal_force_sum >= self.gamma)

        objective += self.force_reg_weight * cs.sumsqr(f)
        opti.minimize(objective)
        self._configure_solver(opti)

        self.force_closure_fn = opti.to_function(
            "force_closure_fn",
            [G],
            [f, objective, cs.vertcat(*cost_terms), cs.hcat(response_terms)],
            ["G"],
            ["f_opt", "cost", "cost_terms", "wrench_response"],
        )

    def _precompile_static_equilibrium_function(self):
        opti = cs.Opti()

        G = opti.parameter(6, 3 * self.num_grasp_contacts)
        wrench_ext = opti.parameter(6)

        f = opti.variable(3 * self.num_grasp_contacts)

        opti.set_initial(f, 0.05)

        residual = wrench_ext + G @ f
        objective = cs.sumsqr(residual) + self.force_reg_weight * cs.sumsqr(f)
        opti.minimize(objective)

        normal_force_sum = 0
        for contact_idx in range(self.num_grasp_contacts):
            f_contact = f[3 * contact_idx : 3 * (contact_idx + 1)]
            self._add_static_friction_cone_constraints(opti, f_contact)
            normal_force_sum += f_contact[0]
        opti.subject_to(normal_force_sum >= self.gamma)

        self._configure_solver(opti)

        self.static_equilibrium_fn = opti.to_function(
            "static_equilibrium_fn",
            [G, wrench_ext],
            [f, residual, objective],
            ["G", "wrench_ext"],
            ["f_opt", "residual", "cost"],
        )

    def _add_friction_cone_constraints(self, opti, force_local):
        opti.subject_to(force_local[0] >= 0.0)
        opti.subject_to(force_local[0] <= 1.0)
        opti.subject_to(cs.sumsqr(force_local[1:3]) <= (self.mu_arm_obj * force_local[0]) ** 2)

    def _add_static_friction_cone_constraints(self, opti, force_local):
        opti.subject_to(force_local[0] >= 0.001)
        opti.subject_to(force_local[0] <= 2.0)
        opti.subject_to(cs.sumsqr(force_local[1:3]) <= (self.mu_arm_obj * force_local[0]) ** 2)

    def _sanitize_point_indices(self, indices):
        if indices is None:
            return self.point_idx.copy()

        indices = np.asarray(indices, dtype=int).reshape(-1)
        if indices.size == 0:
            return np.zeros((0,), dtype=int)
        indices = indices[(indices >= 0) & (indices < self.sample_num)]
        if indices.size == 0:
            return np.zeros((0,), dtype=int)
        return np.unique(indices)

    def _filter_low_curvature_indices(self, indices):
        indices = self._sanitize_point_indices(indices)
        if indices.size == 0:
            return indices

        curvature = self.sample_curvature[indices]
        keep_mask = curvature <= self.max_region_curvature
        if np.any(keep_mask):
            return indices[keep_mask]

        keep_count = min(indices.size, max(self.num_grasp_contacts, 32))
        order = np.argsort(curvature)
        return indices[order[:keep_count]]

    def set_support_surface(
        self,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        if support_surface_clearance is not None:
            self.support_surface_clearance = max(0.0, float(support_surface_clearance))
        if support_surface_normal_alignment_threshold is not None:
            self.support_surface_normal_alignment_threshold = float(
                np.clip(support_surface_normal_alignment_threshold, 0.0, 1.0)
            )

        if support_surface_point is None and support_surface_normal is None:
            if (
                support_surface_clearance is None
                and support_surface_normal_alignment_threshold is None
            ):
                self.support_surface_point = None
                self.support_surface_normal = None
            return

        normal = self.support_surface_normal
        if support_surface_normal is not None:
            normal = _normalize(_as_numpy(support_surface_normal).reshape(3))
        if normal is None:
            raise ValueError("support_surface_normal is required when configuring a support surface.")
        if np.linalg.norm(normal) < 1e-8:
            raise ValueError("support_surface_normal must be a non-zero 3D vector.")

        self.support_surface_point = (
            None if support_surface_point is None else _as_numpy(support_surface_point).reshape(3)
        )
        self.support_surface_normal = normal

    def _resolve_support_surface(
        self,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        point = self.support_surface_point
        if support_surface_point is not None:
            point = _as_numpy(support_surface_point).reshape(3)

        normal = self.support_surface_normal
        if support_surface_normal is not None:
            normal = _normalize(_as_numpy(support_surface_normal).reshape(3))
        if normal is None:
            if support_surface_point is not None:
                raise ValueError("support_surface_normal is required when support_surface_point is provided.")
            return None
        if np.linalg.norm(normal) < 1e-8:
            raise ValueError("support_surface_normal must be a non-zero 3D vector.")

        clearance = self.support_surface_clearance
        if support_surface_clearance is not None:
            clearance = max(0.0, float(support_surface_clearance))

        normal_alignment_threshold = self.support_surface_normal_alignment_threshold
        if support_surface_normal_alignment_threshold is not None:
            normal_alignment_threshold = float(
                np.clip(support_surface_normal_alignment_threshold, 0.0, 1.0)
            )

        return {
            "point": None if point is None else np.asarray(point, dtype=np.float64),
            "normal": np.asarray(normal, dtype=np.float64),
            "clearance": float(clearance),
            "normal_alignment_threshold": float(normal_alignment_threshold),
        }

    def get_contact_candidate_indices(
        self,
        visible_face_idx=None,
        object_pos=None,
        object_rot=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        candidate_idx = self._sanitize_point_indices(visible_face_idx)
        if candidate_idx.size == 0:
            return candidate_idx

        support_surface = self._resolve_support_surface(
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
            support_surface_clearance=support_surface_clearance,
            support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
        )
        if support_surface is None:
            return self._filter_low_curvature_indices(candidate_idx)

        if object_pos is not None and object_rot is not None:
            self.last_object_pos = _as_numpy(object_pos).reshape(3)
            self.last_object_rot = _as_numpy(object_rot).reshape(3, 3)
        elif self.last_object_pos is not None and self.last_object_rot is not None:
            object_pos = self.last_object_pos
            object_rot = self.last_object_rot

        if object_pos is None or object_rot is None:
            raise ValueError(
                "object_pos and object_rot are required when filtering contact candidates with a support surface."
            )

        object_pos = _as_numpy(object_pos).reshape(3)
        object_rot = _as_numpy(object_rot).reshape(3, 3)
        support_normal = support_surface["normal"]

        candidate_points_world = (object_rot @ self.sample_point[candidate_idx].T).T + object_pos[None, :]
        candidate_outward_world = (object_rot @ self.outward_normal[candidate_idx].T).T

        if support_surface["point"] is None:
            # If no plane point is given, treat the lowest points along the support
            # normal as the current support band.
            all_points_world = (object_rot @ self.sample_point.T).T + object_pos[None, :]
            support_level = float(np.min(all_points_world @ support_normal))
        else:
            support_level = float(np.dot(support_surface["point"], support_normal))

        signed_height = candidate_points_world @ support_normal - support_level
        support_facing = (
            candidate_outward_world @ support_normal
        ) <= -support_surface["normal_alignment_threshold"]
        blocked_mask = (signed_height <= support_surface["clearance"] + _EPS) & support_facing
        return self._filter_low_curvature_indices(candidate_idx[~blocked_mask])

    def _build_region_from_anchor(self, anchor_idx, candidate_idx, candidate_tree, region_radius, max_points):
        anchor_idx = int(anchor_idx)
        anchor_local = int(np.where(candidate_idx == anchor_idx)[0][0])
        neighbor_local = candidate_tree.query_ball_point(self.sample_point[anchor_idx], r=region_radius)
        neighbor_local = np.asarray(neighbor_local, dtype=int).reshape(-1)
        if neighbor_local.size == 0:
            neighbor_local = np.array([anchor_local], dtype=int)

        region_idx = candidate_idx[neighbor_local]
        region_points = self.sample_point[region_idx]
        dist2 = np.sum((region_points - self.sample_point[anchor_idx][None]) ** 2, axis=1)
        order = np.argsort(dist2)
        region_idx = region_idx[order[: max_points]]
        region_points = self.sample_point[region_idx]

        region_outward = self.outward_normal[region_idx]
        mean_outward = _normalize(np.mean(region_outward, axis=0))
        mean_inward = -mean_outward
        normal_consistency = float(np.mean(np.clip(region_outward @ mean_outward, -1.0, 1.0)))
        curvature_angles = np.arccos(np.clip(region_outward @ mean_outward, -1.0, 1.0))
        mean_curvature = float(np.mean(curvature_angles))
        center = np.mean(region_points, axis=0)
        radial = _normalize(center - self.mesh_centroid)
        accessibility = float(np.clip(np.dot(mean_outward, radial), -1.0, 1.0))
        hull_distance = float(np.mean(self.convex_hull_tree.query(region_points, k=1)[0]))

        is_high_curvature = mean_curvature > self.max_region_curvature
        is_concave = (
            hull_distance > self.concavity_tol
            or normal_consistency < self.normal_consistency_min
            or accessibility < self.accessibility_alignment_min
        )
        quality = (
            1.25 * normal_consistency
            + max(accessibility, 0.0)
            - hull_distance / max(self.concavity_tol, 1e-6)
            - mean_curvature / max(self.max_region_curvature, 1e-6)
        )

        return {
            "anchor_idx": anchor_idx,
            "point_indices": np.asarray(region_idx, dtype=int),
            "center": np.asarray(center, dtype=np.float64),
            "mean_inward_normal": np.asarray(mean_inward, dtype=np.float64),
            "mean_outward_normal": np.asarray(mean_outward, dtype=np.float64),
            "normal_consistency": normal_consistency,
            "mean_curvature": mean_curvature,
            "accessibility": accessibility,
            "hull_distance": hull_distance,
            "quality": float(quality),
            "is_high_curvature": bool(is_high_curvature),
            "is_concave": bool(is_concave),
        }

    def _deduplicate_regions(self, regions):
        if not regions:
            return []

        sorted_regions = sorted(regions, key=lambda item: item["quality"], reverse=True)
        selected = []
        for region in sorted_regions:
            keep = True
            region_points = region["point_indices"]
            for chosen in selected:
                center_distance = np.linalg.norm(region["center"] - chosen["center"])
                normal_dot = float(
                    np.clip(
                        np.dot(region["mean_inward_normal"], chosen["mean_inward_normal"]),
                        -1.0,
                        1.0,
                    )
                )
                overlap = np.intersect1d(region_points, chosen["point_indices"]).size
                overlap_ratio = overlap / max(1, min(region_points.size, chosen["point_indices"].size))
                if center_distance < 0.35 * self.region_radius and normal_dot > 0.9 and overlap_ratio > 0.6:
                    keep = False
                    break
            if keep:
                selected.append(region)
        return selected

    def sample_candidate_region_on_surface(
        self,
        visible_face_idx=None,
        anchor_count=None,
        region_radius=None,
        max_points_per_region=None,
        object_pos=None,
        object_rot=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        candidate_idx = self.get_contact_candidate_indices(
            visible_face_idx=visible_face_idx,
            object_pos=object_pos,
            object_rot=object_rot,
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
            support_surface_clearance=support_surface_clearance,
            support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
        )
        if candidate_idx.size == 0:
            return []

        region_radius = self.region_radius if region_radius is None else float(region_radius)
        max_points_per_region = self.region_max_points if max_points_per_region is None else int(max_points_per_region)
        anchor_count = self.region_anchor_count if anchor_count is None else int(anchor_count)
        anchor_count = min(anchor_count, candidate_idx.size)

        candidate_points = self.sample_point[candidate_idx]
        candidate_tree = cKDTree(candidate_points)
        anchor_local_idx = _farthest_point_sampling(candidate_points, anchor_count)

        raw_regions = []
        valid_regions = []
        for local_anchor in anchor_local_idx:
            anchor_idx = int(candidate_idx[int(local_anchor)])
            region = self._build_region_from_anchor(
                anchor_idx,
                candidate_idx,
                candidate_tree,
                region_radius,
                max_points_per_region,
            )
            raw_regions.append(region)
            if not region["is_concave"] and not region["is_high_curvature"]:
                valid_regions.append(region)

        regions = valid_regions
        regions = self._deduplicate_regions(regions)
        regions.sort(key=lambda item: item["quality"], reverse=True)
        return regions

    def _select_farthest_point_set(self, indices, num_points=None):
        indices = self._sanitize_point_indices(indices)
        if indices.size == 0:
            return None

        num_points = self.num_grasp_contacts if num_points is None else int(num_points)
        num_points = min(max(1, num_points), indices.size)
        local_idx = _farthest_point_sampling(self.sample_point[indices], num_points)
        return indices[np.asarray(local_idx, dtype=int)]

    def sample_points_from_region(self, region, num_points=None):
        if region is None:
            return np.zeros((0,), dtype=int)

        indices = np.asarray(region["point_indices"], dtype=int)
        if indices.size == 0:
            return indices

        num_points = self.region_contact_samples if num_points is None else int(num_points)
        num_points = min(num_points, indices.size)
        anchor_hits = np.where(indices == int(region["anchor_idx"]))[0]
        seed_index = int(anchor_hits[0]) if anchor_hits.size else None
        sampled_local = _farthest_point_sampling(self.sample_point[indices], num_points, seed_index=seed_index)
        return indices[sampled_local]

    def _scale_wrench(self, wrench):
        wrench = _as_numpy(wrench).reshape(6)
        return self.wrench_scale * wrench

    def _unscale_wrench(self, wrench):
        wrench = _as_numpy(wrench).reshape(6)
        return wrench / np.maximum(self.wrench_scale, _EPS)

    def _default_gravity_wrench_local(self):
        return np.array([0.0, 0.0, -self.m * 9.81, 0.0, 0.0, 0.0], dtype=np.float64)

    def _stack_grasp_matrices(self, contact_indices, scaled=True):
        contact_indices = np.asarray(contact_indices, dtype=int).reshape(-1)
        return np.hstack([self._grasp_matrix(int(idx), scaled=scaled) for idx in contact_indices])

    def _grasp_matrix(self, idx, scaled=True):
        idx = int(idx)
        p = self.sample_point[idx]
        n = self.normal[idx]
        d = self.t1[idx]
        e = self.t2[idx]

        G = np.zeros((6, 3), dtype=np.float64)
        G[:3, 0] = n
        G[:3, 1] = d
        G[:3, 2] = e
        G[3:, 0] = np.cross(p, n)
        G[3:, 1] = np.cross(p, d)
        G[3:, 2] = np.cross(p, e)
        if scaled:
            return self.wrench_scale[:, None] * G
        return G

    def _primitive_wrenches_for_indices(self, indices):
        indices = np.asarray(indices, dtype=int).reshape(-1)
        primitive_wrenches = np.zeros((indices.size, self.friction_primitives.shape[0], 6), dtype=np.float64)
        for row, idx in enumerate(indices):
            primitive_wrenches[row] = self.friction_primitives @ self._grasp_matrix(int(idx)).T
        return primitive_wrenches

    def _sample_boundary_wrenches(self, primitive_sets, sample_count, seed):
        primitive_sets = [np.asarray(ps, dtype=np.float64) for ps in primitive_sets]
        if not primitive_sets:
            return np.zeros((0, 6), dtype=np.float64)

        total_combo_count = 1
        counts = []
        for primitive_set in primitive_sets:
            count = int(primitive_set.shape[0])
            counts.append(count)
            total_combo_count *= max(count, 1)

        if total_combo_count <= sample_count:
            boundary_wrenches = []
            for combo in itertools.product(*[range(count) for count in counts]):
                wrench = np.zeros(6, dtype=np.float64)
                for primitive_set, combo_idx in zip(primitive_sets, combo):
                    wrench += primitive_set[int(combo_idx)]
                boundary_wrenches.append(wrench)
            return np.asarray(boundary_wrenches, dtype=np.float64)

        rng = np.random.default_rng(int(seed) % (2**32 - 1))
        boundary_wrenches = np.zeros((sample_count, 6), dtype=np.float64)
        for sample_idx in range(sample_count):
            wrench = np.zeros(6, dtype=np.float64)
            for primitive_set in primitive_sets:
                primitive_idx = int(rng.integers(primitive_set.shape[0]))
                wrench += primitive_set[primitive_idx]
            boundary_wrenches[sample_idx] = wrench
        return boundary_wrenches

    def _estimate_region_group_gwb(self, region_group):
        region_group = list(region_group)
        region_sample_indices = [
            self.sample_points_from_region(region, self.region_contact_samples)
            for region in region_group
        ]
        primitive_sets = [
            self._primitive_wrenches_for_indices(sample_idx).reshape(-1, 6)
            for sample_idx in region_sample_indices
        ]
        seed = sum(int(region["anchor_idx"]) for region in region_group) + 97 * len(region_group)
        boundary_wrenches = self._sample_boundary_wrenches(
            primitive_sets,
            self.gwb_wrench_count,
            seed=seed,
        )

        projections = self.disturbance_wrenches @ boundary_wrenches.T
        disturbance_scores = np.max(projections, axis=1)
        stability_score = float(np.min(disturbance_scores))

        return {
            "regions": region_group,
            "region_sample_indices": region_sample_indices,
            "boundary_wrenches": boundary_wrenches,
            "disturbance_scores": disturbance_scores,
            "stability_score": stability_score,
        }

    def _compute_region_group_coarse_score(self, region_group):
        centers = np.stack([region["center"] for region in region_group], axis=0)
        normals = np.stack([region["mean_inward_normal"] for region in region_group], axis=0)
        pairwise_distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        triu = np.triu_indices(len(region_group), k=1)
        if triu[0].size == 0:
            return None

        pair_distances = pairwise_distances[triu]
        if np.any(pair_distances < self.region_pair_min_distance):
            return None

        normal_dots = np.clip(normals @ normals.T, -1.0, 1.0)
        normal_diversity = float(np.mean(1.0 - normal_dots[triu]))
        mean_distance = float(np.mean(pair_distances))
        min_distance = float(np.min(pair_distances))
        mean_quality = float(np.mean([region["quality"] for region in region_group]))
        mean_accessibility = float(np.mean([max(region["accessibility"], 0.0) for region in region_group]))

        coarse_score = min_distance * mean_distance * max(mean_quality, 0.1) * (
            0.5 + normal_diversity + 0.5 * mean_accessibility
        )
        return float(coarse_score)

    def _fallback_region_group(self, visible_face_idx):
        visible_face_idx = self._sanitize_point_indices(visible_face_idx)
        if visible_face_idx.size == 0:
            return []

        seed_indices = self._select_farthest_point_set(visible_face_idx, self.num_grasp_contacts)
        if seed_indices is None or len(seed_indices) == 0:
            return []

        candidate_tree = cKDTree(self.sample_point[visible_face_idx])
        regions = [
            self._build_region_from_anchor(
                int(anchor_idx),
                visible_face_idx,
                candidate_tree,
                self.region_radius,
                self.region_max_points,
            )
            for anchor_idx in np.asarray(seed_indices, dtype=int)
        ]
        regions = [
            region for region in regions
            if not region["is_concave"] and not region["is_high_curvature"]
        ]
        if len(regions) < self.num_grasp_contacts:
            return []
        fallback = self._estimate_region_group_gwb(regions)
        fallback["rank"] = 1
        fallback["coarse_score"] = 0.0
        return [fallback]

    def get_best_regions(
        self,
        visible_face_idx=None,
        top_k=3,
        object_pos=None,
        object_rot=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        candidate_idx = self.get_contact_candidate_indices(
            visible_face_idx=visible_face_idx,
            object_pos=object_pos,
            object_rot=object_rot,
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
            support_surface_clearance=support_surface_clearance,
            support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
        )
        if candidate_idx.size == 0:
            self.last_region_results = []
            return []

        candidate_idx = self._project_proxy_points_to_original_indices(candidate_idx)
        self.last_available_idx = np.asarray(candidate_idx, dtype=int)
        if candidate_idx.size == 0:
            self.last_region_results = []
            return []

        candidate_regions = self.sample_candidate_region_on_surface(visible_face_idx=candidate_idx)
        if len(candidate_regions) < self.num_grasp_contacts:
            self.last_region_results = self._fallback_region_group(candidate_idx)
            return self.last_region_results[: int(top_k)]

        candidate_region_limit = min(
            len(candidate_regions),
            max(
                self.num_grasp_contacts,
                8,
                2 * self.num_grasp_contacts * self.top_region_pairs,
            ),
        )
        candidate_regions = candidate_regions[:candidate_region_limit]

        coarse_groups = []
        for combo in itertools.combinations(range(len(candidate_regions)), self.num_grasp_contacts):
            region_group = [candidate_regions[idx] for idx in combo]
            coarse_score = self._compute_region_group_coarse_score(region_group)
            if coarse_score is None:
                continue
            coarse_groups.append((coarse_score, combo))

        if not coarse_groups:
            self.last_region_results = self._fallback_region_group(candidate_idx)
            return self.last_region_results[: int(top_k)]

        coarse_groups.sort(key=lambda item: item[0], reverse=True)
        coarse_groups = coarse_groups[: self.preselect_region_pairs]

        scored_groups = []
        for coarse_score, combo in coarse_groups:
            region_group = [candidate_regions[idx] for idx in combo]
            group_info = self._estimate_region_group_gwb(region_group)
            group_info["coarse_score"] = float(coarse_score)
            scored_groups.append(group_info)

        scored_groups.sort(
            key=lambda item: (item["stability_score"], item["coarse_score"]),
            reverse=True,
        )

        best_groups = scored_groups[: int(top_k)]
        for rank, group in enumerate(best_groups, start=1):
            group["rank"] = rank
            for region_idx, region in enumerate(group["regions"]):
                group[f"region{region_idx + 1}"] = region
                group[f"region{region_idx + 1}_sample_idx"] = np.asarray(
                    group["region_sample_indices"][region_idx],
                    dtype=int,
                )

        self.last_region_results = best_groups
        return best_groups

    def compute_two_point_force_closure_margin(self, idx1, idx2):
        idx1 = int(idx1)
        idx2 = int(idx2)
        if idx1 == idx2:
            return -np.inf

        p1 = self.sample_point[idx1]
        p2 = self.sample_point[idx2]
        delta = p2 - p1
        distance = float(np.linalg.norm(delta))
        if distance < max(self.min_pair_distance, 1e-6):
            return -np.inf

        line_12 = delta / distance
        friction_angle = np.arctan(self.mu_arm_obj)
        cos_phi = float(np.cos(friction_angle))

        margin_1 = float(np.dot(self.normal[idx1], line_12) - cos_phi)
        margin_2 = float(np.dot(self.normal[idx2], -line_12) - cos_phi)
        return min(margin_1, margin_2)

    def _compute_contact_set_metrics(self, contact_indices):
        contact_indices = np.asarray(contact_indices, dtype=int).reshape(-1)
        points = self.sample_point[contact_indices]
        normals = self.normal[contact_indices]
        pairwise_distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        triu = np.triu_indices(contact_indices.size, k=1)

        overlap_penalty = 0.0
        if triu[0].size:
            too_close = pairwise_distances[triu] < self.min_pair_distance
            if np.any(too_close):
                overlap_penalty = self.overlap_penalty_weight * float(
                    np.sum((self.min_pair_distance - pairwise_distances[triu][too_close]) ** 2)
                )

        if contact_indices.size == 2:
            geometry_margin = self.compute_two_point_force_closure_margin(contact_indices[0], contact_indices[1])
            geometry_penalty = 0.0
            if not np.isfinite(geometry_margin) or geometry_margin < 0.0:
                geometry_penalty = self.antipodal_penalty_weight * (
                    max(-geometry_margin, 1e-4) ** 2 if np.isfinite(geometry_margin) else 1.0
                )
            normal_diversity = float(1.0 - np.clip(np.dot(normals[0], normals[1]), -1.0, 1.0))
            mean_distance = float(np.mean(pairwise_distances[triu])) if triu[0].size else 0.0
        else:
            pair_margins = [
                self.compute_two_point_force_closure_margin(contact_indices[i], contact_indices[j])
                for i, j in zip(*triu)
            ]
            geometry_margin = float(np.max(pair_margins)) if pair_margins else -np.inf
            normal_dots = np.clip(normals @ normals.T, -1.0, 1.0)
            normal_diversity = float(np.mean(1.0 - normal_dots[triu])) if triu[0].size else 0.0
            target_diversity = 0.8
            geometry_penalty = self.antipodal_penalty_weight * max(0.0, target_diversity - normal_diversity) ** 2
            mean_distance = float(np.mean(pairwise_distances[triu])) if triu[0].size else 0.0

        min_distance = float(np.min(pairwise_distances[triu])) if triu[0].size else 0.0
        distance_reward = 0.0
        if triu[0].size:
            distance_reward = 0.5 * (min_distance + mean_distance) / max(self.mesh_diag, 1e-6)
        return {
            "geometry_margin": float(geometry_margin),
            "geometry_penalty": float(geometry_penalty),
            "overlap_penalty": float(overlap_penalty),
            "normal_diversity": float(normal_diversity),
            "min_contact_distance": float(min_distance),
            "mean_contact_distance": float(mean_distance),
            "distance_reward": float(distance_reward),
        }

    def _local_force_to_object_force(self, idx, force_local):
        idx = int(idx)
        force_local = _as_numpy(force_local).reshape(3)
        return (
            self.normal[idx] * force_local[0]
            + self.t1[idx] * force_local[1]
            + self.t2[idx] * force_local[2]
        )

    def _forces_from_solution_matrix(self, force_matrix):
        force_matrix = _as_numpy(force_matrix).reshape(3 * self.num_grasp_contacts, -1)
        return np.stack(
            [
                force_matrix[3 * contact_idx : 3 * (contact_idx + 1)]
                for contact_idx in range(self.num_grasp_contacts)
            ],
            axis=0,
        )

    def _forces_from_solution_vector(self, force_vector):
        force_vector = _as_numpy(force_vector).reshape(3 * self.num_grasp_contacts)
        return np.stack(
            [
                force_vector[3 * contact_idx : 3 * (contact_idx + 1)]
                for contact_idx in range(self.num_grasp_contacts)
            ],
            axis=0,
        )

    def _local_force_batch_to_object(self, contact_indices, contact_forces_local):
        contact_indices = np.asarray(contact_indices, dtype=int).reshape(-1)
        contact_forces_local = _as_numpy(contact_forces_local)
        return np.stack(
            [
                self._local_force_to_object_force(int(idx), contact_forces_local[i])
                for i, idx in enumerate(contact_indices)
            ],
            axis=0,
        )

    def _solve_force_closure(self, G):
        try:
            return self.force_closure_fn(G=G)
        except RuntimeError:
            return None

    def _solve_static_equilibrium(self, G, wrench_ext):
        try:
            return self.static_equilibrium_fn(G=G, wrench_ext=wrench_ext)
        except RuntimeError:
            return None

    def solve_static_equilibrium(self, result=None, external_wrench=None):
        if result is None:
            result = self.last_grasp_result
        if result is None:
            raise ValueError("No grasp result available. Call get_best_grasp() first or pass a result dict.")

        if isinstance(result, dict):
            contact_indices = np.asarray(result["contact_indices"], dtype=int).reshape(-1)
        else:
            contact_indices = np.asarray(result, dtype=int).reshape(-1)

        if external_wrench is None:
            external_wrench = self._default_gravity_wrench_local()
        external_wrench = _as_numpy(external_wrench).reshape(6)
        scaled_wrench_ext = self._scale_wrench(external_wrench)

        start_time = time.time()
        sol = self._solve_static_equilibrium(
            self._stack_grasp_matrices(contact_indices, scaled=True),
            scaled_wrench_ext,
        )
        solve_time = time.time() - start_time

        if sol is None:
            contact_forces_local = np.zeros((contact_indices.size, 3), dtype=np.float64)
            scaled_residual = np.full(6, np.inf, dtype=np.float64)
            cost = float("inf")
            valid = False
        else:
            contact_forces_local = self._forces_from_solution_vector(sol["f_opt"])
            scaled_residual = _as_numpy(sol["residual"]).reshape(6)
            cost = float(_as_numpy(sol["cost"]).reshape(-1)[0])
            valid = np.isfinite(cost)

        residual_unscaled = self._unscale_wrench(scaled_residual)
        force_vectors = self._local_force_batch_to_object(contact_indices, contact_forces_local)

        static_result = {
            "valid": bool(valid),
            "contact_indices": np.asarray(contact_indices, dtype=int),
            "contact_points": self.sample_point[contact_indices],
            "contact_normals": self.normal[contact_indices],
            "external_wrench": external_wrench,
            "scaled_external_wrench": scaled_wrench_ext,
            "contact_forces_local": contact_forces_local,
            "force_vectors_local": force_vectors,
            "residual_wrench": residual_unscaled,
            "scaled_residual_wrench": scaled_residual,
            "residual_norm": float(np.linalg.norm(residual_unscaled)),
            "scaled_residual_norm": float(np.linalg.norm(scaled_residual)),
            "cost": cost,
            "solve_time": float(solve_time),
        }

        self.last_static_equilibrium_result = static_result
        if isinstance(result, dict):
            result["static_equilibrium"] = static_result
        return static_result

    def _get_visualization_result(self, result=None, use_static_equilibrium=False, external_wrench=None):
        if result is None:
            result = self.last_grasp_result
        if result is None:
            raise ValueError("No grasp result available for visualization.")

        if not use_static_equilibrium:
            return result

        static_result = None
        if isinstance(result, dict):
            static_result = result.get("static_equilibrium", None)
            if static_result is not None and external_wrench is not None:
                cached_wrench = _as_numpy(static_result.get("external_wrench", np.zeros(6))).reshape(6)
                requested_wrench = _as_numpy(external_wrench).reshape(6)
                if not np.allclose(cached_wrench, requested_wrench):
                    static_result = None

        if static_result is None:
            static_result = self.solve_static_equilibrium(result=result, external_wrench=external_wrench)

        vis_result = dict(result) if isinstance(result, dict) else {}
        vis_result.update(static_result)
        return vis_result

    def _get_visualization_results(self, result=None, use_static_equilibrium=False, external_wrench=None, top_k=3):
        if result is None:
            result = self.last_grasp_result
        if result is None:
            raise ValueError("No grasp result available for visualization.")

        candidate_results = []
        if isinstance(result, dict):
            top_grasp_results = list(result.get("top_grasp_results", []))
            if top_grasp_results:
                candidate_results = top_grasp_results[: max(1, int(top_k))]

        if not candidate_results:
            candidate_results = [result]

        return [
            self._get_visualization_result(
                result=candidate_result,
                use_static_equilibrium=use_static_equilibrium,
                external_wrench=external_wrench,
            )
            for candidate_result in candidate_results
        ]

    def _evaluate_force_closure_candidate(self, contact_indices, region_group=None):
        contact_indices = np.asarray(contact_indices, dtype=int).reshape(-1)
        metrics = self._compute_contact_set_metrics(contact_indices)

        start_time = time.time()
        sol = self._solve_force_closure(self._stack_grasp_matrices(contact_indices, scaled=True))
        solve_time = time.time() - start_time

        if sol is None:
            force_closure_cost = float("inf")
            cost_terms = np.full((self.disturbance_wrenches.shape[0],), np.inf, dtype=np.float64)
            local_contact_forces = np.zeros(
                (contact_indices.size, 3, self.disturbance_wrenches.shape[0]),
                dtype=np.float64,
            )
            wrench_response = np.zeros((6, self.disturbance_wrenches.shape[0]), dtype=np.float64)
            valid = False
        else:
            force_closure_cost = float(_as_numpy(sol["cost"]).reshape(-1)[0])
            cost_terms = _as_numpy(sol["cost_terms"]).reshape(-1)
            local_contact_forces = self._forces_from_solution_matrix(sol["f_opt"])
            wrench_response = _as_numpy(sol["wrench_response"]).reshape(6, -1)
            valid = np.isfinite(force_closure_cost)

        worst_disturbance_idx = int(np.argmax(cost_terms)) if cost_terms.size else 0
        chosen_contact_force_local = (
            local_contact_forces[:, :, worst_disturbance_idx]
            if local_contact_forces.size
            else np.zeros((contact_indices.size, 3), dtype=np.float64)
        )
        force_vectors = self._local_force_batch_to_object(contact_indices, chosen_contact_force_local)

        total_cost = (
            force_closure_cost
            + metrics["geometry_penalty"]
            + metrics["overlap_penalty"]
            - self.distance_reward_weight * metrics["distance_reward"]
        )
        result = {
            "valid": bool(valid and np.isfinite(total_cost)),
            "contact_indices": np.asarray(contact_indices, dtype=int),
            "contact_points": self.sample_point[contact_indices],
            "contact_normals": self.normal[contact_indices],
            "contact_tangent1": self.t1[contact_indices],
            "contact_tangent2": self.t2[contact_indices],
            "min_contact_distance": metrics["min_contact_distance"],
            "mean_contact_distance": metrics["mean_contact_distance"],
            "pair_distance": metrics["min_contact_distance"],
            "antipodal_margin": float(metrics["geometry_margin"]),
            "antipodal_penalty": float(metrics["geometry_penalty"]),
            "normal_diversity": float(metrics["normal_diversity"]),
            "overlap_penalty": float(metrics["overlap_penalty"]),
            "distance_reward": float(metrics["distance_reward"]),
            "force_closure_cost": float(force_closure_cost),
            "total_cost": float(total_cost),
            "solve_time": float(solve_time),
            "local_contact_forces": local_contact_forces,
            "cost_terms": cost_terms,
            "wrench_response": wrench_response,
            "worst_disturbance_idx": worst_disturbance_idx,
            "worst_disturbance_label": self.disturbance_labels[worst_disturbance_idx],
            "worst_disturbance_wrench": self.disturbance_wrenches[worst_disturbance_idx],
            "force_vectors_local": force_vectors,
            "chosen_contact_force_local": chosen_contact_force_local,
        }

        if region_group is not None:
            result["region_rank"] = int(region_group.get("rank", 0))
            result["region_score"] = float(region_group.get("stability_score", 0.0))
            result["regions"] = region_group["regions"]
            result["region_sample_indices"] = [
                np.asarray(sample_idx, dtype=int)
                for sample_idx in region_group["region_sample_indices"]
            ]
            for region_idx, region in enumerate(region_group["regions"]):
                result[f"region{region_idx + 1}"] = region
                result[f"region{region_idx + 1}_sample_idx"] = np.asarray(
                    region_group["region_sample_indices"][region_idx],
                    dtype=int,
                )

        return result

    def _evaluate_force_closure_pair(self, idx1, idx2, region_pair=None):
        return self._evaluate_force_closure_candidate([idx1, idx2], region_group=region_pair)

    def _prepare_contact_sample_groups(self, region_sample_groups):
        sample_groups = [np.asarray(group, dtype=int).reshape(-1) for group in region_sample_groups]
        if not sample_groups:
            return sample_groups

        per_group_cap = max(1, int(round(self.max_point_combination_eval ** (1.0 / len(sample_groups)))))
        sample_groups = [group[: min(group.size, per_group_cap)] for group in sample_groups]

        while np.prod([max(group.size, 1) for group in sample_groups], dtype=np.int64) > self.max_point_combination_eval:
            largest_group_idx = int(np.argmax([group.size for group in sample_groups]))
            if sample_groups[largest_group_idx].size <= 1:
                break
            sample_groups[largest_group_idx] = sample_groups[largest_group_idx][:-1]
        return sample_groups

    @staticmethod
    def _grasp_candidate_sort_key(item):
        return (
            item["total_cost"],
            -item.get("region_score", 0.0),
            -item["min_contact_distance"],
        )

    def _collect_top_grasp_results(self, ranked_candidates, top_k=3):
        top_results = []
        for rank, candidate in enumerate(ranked_candidates[: max(1, int(top_k))], start=1):
            candidate_copy = dict(candidate)
            candidate_copy["grasp_rank"] = rank
            top_results.append(candidate_copy)
        self.last_ranked_grasp_results = top_results
        return top_results

    def get_best_grasp(
        self,
        visible_face_idx=None,
        object_pos=None,
        object_rot=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        region_groups = self.get_best_regions(
            visible_face_idx=visible_face_idx,
            top_k=self.top_region_pairs,
            object_pos=object_pos,
            object_rot=object_rot,
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
            support_surface_clearance=support_surface_clearance,
            support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
        )
        if not region_groups:
            self.last_ranked_grasp_results = []
            self.last_grasp_result = None
            return None

        evaluated_candidates = []
        for region_group in region_groups:
            sample_groups = self._prepare_contact_sample_groups(region_group["region_sample_indices"])
            for contact_indices in itertools.product(*[group.tolist() for group in sample_groups]):
                evaluated_candidates.append(
                    self._evaluate_force_closure_candidate(contact_indices, region_group=region_group)
                )

        if not evaluated_candidates:
            self.last_ranked_grasp_results = []
            self.last_grasp_result = None
            return None

        ranked_candidates = sorted(evaluated_candidates, key=self._grasp_candidate_sort_key)
        best_result = ranked_candidates[0]
        top_grasp_results = self._collect_top_grasp_results(ranked_candidates, top_k=3)
        best_result["evaluated_candidate_count"] = len(evaluated_candidates)
        best_result["grasp_rank"] = 1
        best_result["top_grasp_results"] = top_grasp_results
        best_result["top_region_pairs"] = region_groups
        best_result["proxy_preselection"] = dict(self.last_proxy_preselection_info)
        self.last_grasp_result = best_result
        return best_result

    def _normalize_region_groups(self, region_groups):
        if region_groups is None:
            return []
        if isinstance(region_groups, dict):
            return [region_groups]
        return [group for group in list(region_groups) if group is not None]

    def _prepare_fixed_region_group(self, region_group, candidate_idx):
        candidate_idx = self._sanitize_point_indices(candidate_idx)
        prepared_regions = []
        prepared_sample_groups = []
        raw_sample_groups = list(region_group.get("region_sample_indices", []))

        for region_idx, region in enumerate(region_group.get("regions", [])):
            region_point_idx = np.asarray(region.get("point_indices", []), dtype=int).reshape(-1)
            filtered_point_idx = (
                region_point_idx[np.isin(region_point_idx, candidate_idx)]
                if candidate_idx.size
                else np.zeros((0,), dtype=int)
            )

            if filtered_point_idx.size == 0:
                fallback_group = (
                    np.asarray(raw_sample_groups[region_idx], dtype=int).reshape(-1)
                    if region_idx < len(raw_sample_groups)
                    else np.zeros((0,), dtype=int)
                )
                filtered_point_idx = fallback_group if fallback_group.size else region_point_idx

            if filtered_point_idx.size == 0:
                return None

            prepared_region = dict(region)
            prepared_region["point_indices"] = np.asarray(filtered_point_idx, dtype=int)
            prepared_regions.append(prepared_region)

            sampled_idx = self.sample_points_from_region(prepared_region, self.region_contact_samples)
            if sampled_idx.size == 0:
                return None
            prepared_sample_groups.append(np.asarray(sampled_idx, dtype=int))

        if not prepared_regions:
            return None

        prepared_group = dict(region_group)
        prepared_group["regions"] = prepared_regions
        prepared_group["region_sample_indices"] = prepared_sample_groups
        for region_idx, region in enumerate(prepared_regions):
            prepared_group[f"region{region_idx + 1}"] = region
            prepared_group[f"region{region_idx + 1}_sample_idx"] = np.asarray(
                prepared_sample_groups[region_idx],
                dtype=int,
            )
        return prepared_group

    def get_best_grasp_from_regions(
        self,
        region_groups,
        visible_face_idx=None,
        object_pos=None,
        object_rot=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        region_groups = self._normalize_region_groups(region_groups)
        if not region_groups:
            self.last_ranked_grasp_results = []
            self.last_grasp_result = None
            return None

        candidate_idx = self.get_contact_candidate_indices(
            visible_face_idx=visible_face_idx,
            object_pos=object_pos,
            object_rot=object_rot,
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
            support_surface_clearance=support_surface_clearance,
            support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
        )
        if candidate_idx.size == 0:
            candidate_idx = self._sanitize_point_indices(visible_face_idx)
        if candidate_idx.size == 0:
            candidate_idx = self.point_idx.copy()

        self.last_available_idx = np.asarray(candidate_idx, dtype=int)
        self.last_proxy_preselection_info = {
            "enabled": False,
            "source": "fixed_regions",
            "base_candidate_count": int(candidate_idx.size),
            "projected_candidate_count": int(candidate_idx.size),
        }

        prepared_region_groups = []
        evaluated_candidates = []
        for region_group in region_groups:
            prepared_group = self._prepare_fixed_region_group(region_group, candidate_idx)
            if prepared_group is None:
                continue

            prepared_region_groups.append(prepared_group)
            sample_groups = self._prepare_contact_sample_groups(prepared_group["region_sample_indices"])
            if not sample_groups or any(group.size == 0 for group in sample_groups):
                continue

            for contact_indices in itertools.product(*[group.tolist() for group in sample_groups]):
                evaluated_candidates.append(
                    self._evaluate_force_closure_candidate(contact_indices, region_group=prepared_group)
                )

        self.last_region_results = prepared_region_groups if prepared_region_groups else region_groups
        self.last_candidate_point_groups = [
            [np.asarray(group, dtype=int) for group in prepared_group["region_sample_indices"]]
            for prepared_group in prepared_region_groups
        ]

        if not evaluated_candidates:
            self.last_ranked_grasp_results = []
            self.last_grasp_result = None
            return None

        ranked_candidates = sorted(evaluated_candidates, key=self._grasp_candidate_sort_key)
        best_result = ranked_candidates[0]
        top_grasp_results = self._collect_top_grasp_results(ranked_candidates, top_k=3)
        best_result["evaluated_candidate_count"] = len(evaluated_candidates)
        best_result["grasp_rank"] = 1
        best_result["top_grasp_results"] = top_grasp_results
        best_result["top_region_pairs"] = prepared_region_groups if prepared_region_groups else region_groups
        best_result["fixed_region_groups"] = True
        best_result["proxy_preselection"] = dict(self.last_proxy_preselection_info)
        self.last_grasp_result = best_result
        return best_result

    def get_availble_point_idx(
        self,
        pos,
        R,
        target_pos,
        threshold=0.025,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        pos = _as_numpy(pos).reshape(3)
        R = _as_numpy(R).reshape(3, 3)

        centers_world = (R @ self.sample_point.T).T + pos
        visible_mask = centers_world[:, 2] > float(threshold)
        visible_idx = np.where(visible_mask)[0]
        if visible_idx.size == 0:
            visible_idx = self.point_idx.copy()

        visible_idx = self.get_contact_candidate_indices(
            visible_face_idx=visible_idx,
            object_pos=pos,
            object_rot=R,
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
            support_surface_clearance=support_surface_clearance,
            support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
        )
        if visible_idx.size == 0:
            visible_idx = self.get_contact_candidate_indices(
                visible_face_idx=self.point_idx,
                object_pos=pos,
                object_rot=R,
                support_surface_point=support_surface_point,
                support_surface_normal=support_surface_normal,
                support_surface_clearance=support_surface_clearance,
                support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
            )

        self.last_available_idx = visible_idx
        best_regions = self.get_best_regions(
            visible_face_idx=visible_idx,
            top_k=self.top_region_pairs,
            object_pos=pos,
            object_rot=R,
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
            support_surface_clearance=support_surface_clearance,
            support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
        )

        grouped_samples = []
        flat_samples = []
        for region_group in best_regions:
            groups = [
                np.asarray(group, dtype=int)
                for group in region_group["region_sample_indices"]
            ]
            grouped_samples.append(groups)
            for group in groups:
                flat_samples.extend(group.tolist())

        self.last_candidate_point_groups = grouped_samples
        if not flat_samples:
            return visible_idx
        return np.unique(np.asarray(flat_samples, dtype=int))

    def choose_contact_points(
        self,
        x_d=None,
        current_x=None,
        tau_o=None,
        visible_face_idx=None,
        v_obj=None,
        object_pos=None,
        object_rot=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
        fixed_region_groups=None,
    ):
        if fixed_region_groups is None:
            result = self.get_best_grasp(
                visible_face_idx=visible_face_idx,
                object_pos=object_pos,
                object_rot=object_rot,
                support_surface_point=support_surface_point,
                support_surface_normal=support_surface_normal,
                support_surface_clearance=support_surface_clearance,
                support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
            )
        else:
            result = self.get_best_grasp_from_regions(
                fixed_region_groups,
                visible_face_idx=visible_face_idx,
                object_pos=object_pos,
                object_rot=object_rot,
                support_surface_point=support_surface_point,
                support_surface_normal=support_surface_normal,
                support_surface_clearance=support_surface_clearance,
                support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
            )
        if result is None:
            fallback_idx = int(self.point_idx[0])
            return (
                self.sample_point[fallback_idx],
                self.normal[fallback_idx],
                float("inf"),
                float("inf"),
                0.0,
            )

        return (
            result["contact_points"][0],
            result["contact_normals"][0],
            float(result["total_cost"]),
            float(result.get("region_score", 0.0)),
            float(result["antipodal_margin"]),
        )

    def choose_contact_set(
        self,
        x_d=None,
        current_x=None,
        tau_o=None,
        visible_face_idx=None,
        v_obj=None,
        object_pos=None,
        object_rot=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
        fixed_region_groups=None,
    ):
        del x_d
        del current_x
        del tau_o
        del v_obj
        if fixed_region_groups is None:
            result = self.get_best_grasp(
                visible_face_idx=visible_face_idx,
                object_pos=object_pos,
                object_rot=object_rot,
                support_surface_point=support_surface_point,
                support_surface_normal=support_surface_normal,
                support_surface_clearance=support_surface_clearance,
                support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
            )
        else:
            result = self.get_best_grasp_from_regions(
                fixed_region_groups,
                visible_face_idx=visible_face_idx,
                object_pos=object_pos,
                object_rot=object_rot,
                support_surface_point=support_surface_point,
                support_surface_normal=support_surface_normal,
                support_surface_clearance=support_surface_clearance,
                support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
            )
        if result is None:
            fallback_idx = int(self.point_idx[0])
            return (
                self.sample_point[[fallback_idx]],
                self.normal[[fallback_idx]],
                float("inf"),
                float("inf"),
                0.0,
            )

        return (
            result["contact_points"],
            result["contact_normals"],
            float(result["total_cost"]),
            float(result.get("region_score", 0.0)),
            float(result["antipodal_margin"]),
        )

    def optimize_dual_contact_input(self, x_d, current_x, tau_o, p_arm1, p_arm2, v_obj=None):
        idx1 = int(self.sample_kdtree.query(_as_numpy(p_arm1).reshape(3), k=1)[1])
        idx2 = int(self.sample_kdtree.query(_as_numpy(p_arm2).reshape(3), k=1)[1])
        result = self._evaluate_force_closure_candidate([idx1, idx2])
        self.last_grasp_result = result
        return result

    def optimize_multi_contact_input(self, contact_points, x_d=None, current_x=None, tau_o=None, v_obj=None):
        del x_d
        del current_x
        del tau_o
        del v_obj
        contact_points = _as_numpy(contact_points).reshape(-1, 3)
        contact_indices = np.array(
            [int(self.sample_kdtree.query(point, k=1)[1]) for point in contact_points],
            dtype=int,
        )
        result = self._evaluate_force_closure_candidate(contact_indices)
        self.last_grasp_result = result
        return result

    def optimize_control_input(self, x_d, current_x, tau_o, p_arm=None, v_obj=None):
        if p_arm is None:
            best_grasp = self.get_best_grasp(self.last_available_idx)
            if best_grasp is None:
                point = self.sample_point[0]
                normal = self.normal[0]
                return point, normal, _as_numpy(current_x).reshape(7), float("inf"), {"solver_failed": True}
            point = best_grasp["contact_points"][0]
            normal = best_grasp["contact_normals"][0]
            return point, normal, _as_numpy(current_x).reshape(7), float(best_grasp["total_cost"]), {
                "solver_failed": False,
                "selected_contacts": best_grasp["contact_indices"],
            }

        idx = int(self.sample_kdtree.query(_as_numpy(p_arm).reshape(3), k=1)[1])
        point = self.sample_point[idx]
        normal = self.normal[idx]
        return point, normal, _as_numpy(current_x).reshape(7), 0.0, {
            "solver_failed": False,
            "nearest_surface_idx": idx,
        }

    @staticmethod
    def _make_colored_sphere(center, radius, color):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color)
        sphere.translate(np.asarray(center, dtype=np.float64))
        return sphere

    @staticmethod
    def _make_arrow(start, vector, color, cylinder_radius, cone_radius):
        length = float(np.linalg.norm(vector))
        if length < 1e-8:
            return None

        cylinder_height = max(0.7 * length, 1e-4)
        cone_height = max(length - cylinder_height, 1e-4)
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=float(cylinder_radius),
            cone_radius=float(cone_radius),
            cylinder_height=float(cylinder_height),
            cone_height=float(cone_height),
        )
        arrow.compute_vertex_normals()
        arrow.paint_uniform_color(color)
        arrow.rotate(_rotation_from_z(vector), center=np.zeros(3, dtype=np.float64))
        arrow.translate(np.asarray(start, dtype=np.float64))
        return arrow

    @staticmethod
    def _contact_palette():
        return [
            np.array([0.95, 0.45, 0.45]),
            np.array([0.25, 0.45, 0.95]),
            np.array([0.95, 0.75, 0.25]),
            np.array([0.20, 0.72, 0.78]),
            np.array([0.85, 0.35, 0.80]),
            np.array([0.55, 0.78, 0.30]),
        ]

    @staticmethod
    def _build_object_mesh_material():
        if o3d is None:
            return None
        if not hasattr(o3d.visualization, "rendering"):
            return None

        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"
        material.base_color = [
            float(_O3D_OBJECT_BASE_COLOR[0]),
            float(_O3D_OBJECT_BASE_COLOR[1]),
            float(_O3D_OBJECT_BASE_COLOR[2]),
            float(_O3D_OBJECT_OPACITY),
        ]
        material.has_alpha = True
        return material

    def _build_draw_items(self, geometries):
        draw_items = []
        mesh_material = self._build_object_mesh_material()

        for geometry_idx, geometry in enumerate(geometries):
            draw_item = {
                "name": f"grasp_geometry_{geometry_idx:03d}",
                "geometry": geometry,
            }
            if geometry_idx == 0 and mesh_material is not None:
                draw_item["material"] = mesh_material
            draw_items.append(draw_item)

        return draw_items

    @staticmethod
    def _draw_geometries_with_white_background(geometries):
        if o3d is None:
            raise ImportError("open3d is required for visualization but is not installed.")

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        try:
            for geometry in geometries:
                vis.add_geometry(geometry)

            render_option = vis.get_render_option()
            render_option.background_color = _O3D_BACKGROUND_COLOR.copy()
            render_option.mesh_show_back_face = True

            vis.run()
        finally:
            vis.destroy_window()

    def build_grasp_visualization_geometries(
        self,
        result=None,
        point_radius=None,
        force_scale=None,
        include_normals=False,
        include_region_samples=True,
        use_static_equilibrium=False,
        external_wrench=None,
    ):
        if o3d is None:
            raise ImportError("open3d is required for visualization but is not installed.")

        vis_results = self._get_visualization_results(
            result=result,
            use_static_equilibrium=use_static_equilibrium,
            external_wrench=external_wrench,
            top_k=3,
        )
        primary_result = vis_results[0]

        mesh_o3d = self.pp.to_open3d_mesh()
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color(_O3D_OBJECT_BASE_COLOR.tolist())

        bbox = mesh_o3d.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        diag = max(diag, 1e-3)
        if point_radius is None:
            point_radius = 0.015 * diag

        all_force_vectors = []
        for vis_result in vis_results:
            contact_points = np.asarray(vis_result["contact_points"], dtype=np.float64)
            force_vectors = np.asarray(
                vis_result.get("force_vectors_local", np.zeros_like(contact_points)),
                dtype=np.float64,
            )
            all_force_vectors.append(force_vectors)
        max_force = max(
            [float(np.max(np.linalg.norm(force_vectors, axis=1))) for force_vectors in all_force_vectors if force_vectors.size]
            + [1e-6]
        )
        if force_scale is None:
            force_scale = 0.18 * diag / max_force

        geometries = [mesh_o3d]
        geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2 * diag, origin=[0.0, 0.0, 0.0])
        )

        grasp_colors = self._contact_palette()
        region_sample_groups = primary_result.get("region_sample_indices", None)
        if region_sample_groups is None:
            region_sample_groups = []
            region_idx = 1
            while f"region{region_idx}_sample_idx" in primary_result:
                region_sample_groups.append(np.asarray(primary_result[f"region{region_idx}_sample_idx"], dtype=int))
                region_idx += 1
        if include_region_samples:
            for i, region_sample_idx in enumerate(region_sample_groups):
                color = grasp_colors[i % len(grasp_colors)]
                region_points = self.sample_point[np.asarray(region_sample_idx, dtype=int)]
                if region_points.size == 0:
                    continue
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(region_points)
                pcd.paint_uniform_color((0.6 * color + 0.4).clip(0.0, 1.0))
                geometries.append(pcd)

        for grasp_idx, vis_result in enumerate(vis_results):
            contact_points = np.asarray(vis_result["contact_points"], dtype=np.float64)
            contact_normals = np.asarray(vis_result["contact_normals"], dtype=np.float64)
            force_vectors = all_force_vectors[grasp_idx]
            color = grasp_colors[grasp_idx % len(grasp_colors)]

            for contact_idx in range(contact_points.shape[0]):
                point = contact_points[contact_idx]
                force_vec = force_vectors[contact_idx] * force_scale

                geometries.append(self._make_colored_sphere(point, point_radius, color))

                arrow = self._make_arrow(
                    point,
                    force_vec,
                    color,
                    cylinder_radius=0.22 * point_radius,
                    cone_radius=0.42 * point_radius,
                )
                if arrow is not None:
                    geometries.append(arrow)

                if include_normals:
                    normal_arrow = self._make_arrow(
                        point,
                        0.18 * diag * contact_normals[contact_idx],
                        color,
                        cylinder_radius=0.12 * point_radius,
                        cone_radius=0.26 * point_radius,
                    )
                    if normal_arrow is not None:
                        geometries.append(normal_arrow)

        return geometries

    def visual_grasp(
        self,
        result=None,
        point_radius=None,
        force_scale=None,
        include_normals=False,
        include_region_samples=True,
        use_static_equilibrium=False,
        external_wrench=None,
    ):
        geometries = self.build_grasp_visualization_geometries(
            result=result,
            point_radius=point_radius,
            force_scale=force_scale,
            include_normals=include_normals,
            include_region_samples=include_region_samples,
            use_static_equilibrium=use_static_equilibrium,
            external_wrench=external_wrench,
        )
        if hasattr(o3d.visualization, "draw") and hasattr(o3d.visualization, "rendering"):
            o3d.visualization.draw(
                self._build_draw_items(geometries),
                bg_color=(*_O3D_BACKGROUND_COLOR.tolist(), 1.0),
                show_skybox=False,
            )
            return
        self._draw_geometries_with_white_background(geometries)

    def visualize_grasp_result(
        self,
        result=None,
        force_scale=None,
        normal_scale=None,
        point_radius=None,
        include_normals=False,
        include_target=True,
        use_static_equilibrium=False,
        external_wrench=None,
    ):
        del normal_scale
        del include_target
        self.visual_grasp(
            result=result,
            point_radius=point_radius,
            force_scale=force_scale,
            include_normals=include_normals,
            include_region_samples=True,
            use_static_equilibrium=use_static_equilibrium,
            external_wrench=external_wrench,
        )


def find_object_mesh(object_name, objects_dir=OBJECT_ASSET_DIR):
    objects_dir = Path(objects_dir)
    if not objects_dir.exists():
        raise FileNotFoundError(f"Objects directory does not exist: {objects_dir}")

    object_name = str(object_name).strip().lower()
    candidates = []
    for mesh_path in objects_dir.rglob("*"):
        if mesh_path.suffix.lower() not in {".stl", ".obj"}:
            continue
        stem = mesh_path.stem.lower()
        filename = mesh_path.name.lower()
        if stem == object_name or filename == object_name:
            score = 0
        elif object_name in stem or object_name in filename:
            score = 1
        else:
            continue
        format_bias = 0 if mesh_path.suffix.lower() == ".stl" else 1
        candidates.append((score, format_bias, len(mesh_path.name), mesh_path))

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find mesh for object '{object_name}' under {objects_dir}."
        )
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def build_default_target_pose(current_pose, lift_distance=0.2):
    current_pose = _as_numpy(current_pose).reshape(7).copy()
    target_pose = current_pose.copy()
    target_pose[2] += float(lift_distance)
    return target_pose


def run_grasp_prediction(args):
    mesh_path = find_object_mesh(args.object_name, args.objects_dir)
    optimizer = LambdaContactControlOptimizer(
        mesh_path=mesh_path,
        obj_mass=args.obj_mass,
        arm_friction=args.arm_friction,
        contact_stiffness=args.contact_stiffness,
        time_step=args.time_step,
        max_contacts=args.max_contacts,
        sample_num=args.sample_num,
        pos_coef=args.pos_coef,
        ori_coef=args.ori_coef,
        num_grasp_contacts=2,
        region_anchor_count=args.region_anchor_count,
        region_radius=args.region_radius,
        region_max_points=args.region_max_points,
        region_contact_samples=args.region_contact_samples,
        top_region_pairs=args.top_region_pairs,
        preselect_region_pairs=args.preselect_region_pairs,
        friction_cone_edges=args.friction_cone_edges,
        gwb_wrench_count=args.gwb_wrench_count,
        beta=args.beta,
        gamma=args.gamma,
        force_reg_weight=args.force_reg_weight,
        concavity_tol=args.concavity_tol,
        proxy_preselect_mode=args.proxy_preselect_mode,
        coacd_threshold=args.coacd_threshold,
        coacd_max_convex_hull=args.coacd_max_convex_hull,
        coacd_prep_resolution=args.coacd_prep_resolution,
        max_point_combination_eval=args.max_point_combination_eval,
    )

    current_pose = np.asarray(args.current_pose, dtype=np.float64)
    target_pose = build_default_target_pose(current_pose, args.lift_distance)
    start_time = time.time()
    result = optimizer.get_best_grasp(np.arange(optimizer.sample_num, dtype=int))
    wall_time = time.time() - start_time
    if result is None:
        raise RuntimeError("Failed to produce a valid grasp candidate.")

    result["mesh_path"] = str(mesh_path)
    result["current_pose"] = current_pose
    result["target_pose"] = target_pose
    result["wall_time"] = wall_time
    return optimizer, result


def build_argparser():
    parser = argparse.ArgumentParser(description="Region-based dual-point grasp selection with GWB ranking and force-closure optimization.")
    parser.add_argument("object_name", type=str, help="Object name used to search envs/assets/objects/*.stl or *.obj")
    parser.add_argument("--objects-dir", type=str, default=str(OBJECT_ASSET_DIR))
    parser.add_argument("--current-pose", nargs=7, type=float, default=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    parser.add_argument("--lift-distance", type=float, default=0.2)
    parser.add_argument("--obj-mass", type=float, default=0.01)
    parser.add_argument("--arm-friction", type=float, default=0.9)
    parser.add_argument("--contact-stiffness", type=float, default=12.5)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--max-contacts", type=int, default=10)
    parser.add_argument("--sample-num", type=int, default=70, help="Legacy parameter kept for compatibility")
    parser.add_argument("--pos-coef", type=float, default=1.0, help="Legacy parameter kept for compatibility")
    parser.add_argument("--ori-coef", type=float, default=0.0005, help="Legacy parameter kept for compatibility")
    parser.add_argument("--num-grasp-contacts", type=int, default=3, help="Number of contact points used for grasp synthesis")
    parser.add_argument("--region-anchor-count", type=int, default=200)
    parser.add_argument("--region-radius", type=float, default=0.08)
    parser.add_argument("--region-max-points", type=int, default=256)
    parser.add_argument("--region-contact-samples", type=int, default=5)
    parser.add_argument("--top-region-pairs", type=int, default=3)
    parser.add_argument("--preselect-region-pairs", type=int, default=200)
    parser.add_argument("--friction-cone-edges", type=int, default=8)
    parser.add_argument("--gwb-wrench-count", type=int, default=1000)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--force-reg-weight", type=float, default=1e-5)
    parser.add_argument("--concavity-tol", type=float, default=None)
    parser.add_argument(
        "--proxy-preselect-mode",
        type=str,
        default="auto",
        choices=["auto", "coacd", "convex_hull", "off"],
        help="Reduce initial contact faces using a convex proxy before region selection",
    )
    parser.add_argument("--coacd-threshold", type=float, default=0.05)
    parser.add_argument("--coacd-max-convex-hull", type=int, default=12)
    parser.add_argument("--coacd-prep-resolution", type=int, default=50)
    parser.add_argument("--max-point-combination-eval", type=int, default=256)
    parser.add_argument("--visualize", action="store_true", help="Show object mesh and top-3 grasp candidates in Open3D")
    parser.add_argument("--force-scale", type=float, default=None)
    parser.add_argument("--point-radius", type=float, default=None)
    parser.add_argument("--show-normals", action="store_true", help="Show contact normal arrows in Open3D")
    parser.add_argument("--hide-normals", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--hide-region-samples", action="store_true")
    parser.add_argument(
        "--use-static-equilibrium-force",
        action="store_true",
        help="Visualize the static-equilibrium contact forces instead of the worst-disturbance witness forces",
    )
    return parser


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    cli_args = build_argparser().parse_args()
    optimizer, grasp_result = run_grasp_prediction(cli_args)

    print("\nBest grasp result")
    print(f"mesh_path: {grasp_result['mesh_path']}")
    print(f"num_grasp_contacts: {grasp_result['contact_indices'].shape[0]}")
    print(f"contact_indices: {grasp_result['contact_indices']}")
    print(f"contact_points_local:\n{grasp_result['contact_points']}")
    print(f"contact_normals_local:\n{grasp_result['contact_normals']}")
    print(f"region_rank: {grasp_result.get('region_rank', 0)}")
    print(f"region_score(GWB): {grasp_result.get('region_score', 0.0):.6f}")
    print(f"force_closure_cost: {grasp_result['force_closure_cost']:.6f}")
    print(f"total_cost: {grasp_result['total_cost']:.6f}")
    print(f"min_contact_distance: {grasp_result['min_contact_distance']:.6f}")
    print(f"antipodal_margin: {grasp_result['antipodal_margin']:.6f}")
    print(f"worst_disturbance: {grasp_result['worst_disturbance_label']}")
    print(f"selected_local_contact_forces:\n{grasp_result['chosen_contact_force_local']}")
    print(f"force_vectors_local:\n{grasp_result['force_vectors_local']}")
    print(f"solve_time: {grasp_result['solve_time']:.6f}s")
    print(f"wall_time: {grasp_result['wall_time']:.6f}s")
    if grasp_result.get("proxy_preselection"):
        print(f"proxy_preselection: {grasp_result['proxy_preselection']}")
    top_grasp_results = list(grasp_result.get("top_grasp_results", []))
    if top_grasp_results:
        print("\nTop grasp candidates")
        for candidate in top_grasp_results[:3]:
            print(
                "rank={rank} total_cost={cost:.6f} region_rank={region_rank} contact_indices={contact_indices}".format(
                    rank=int(candidate.get("grasp_rank", 0)),
                    cost=float(candidate["total_cost"]),
                    region_rank=int(candidate.get("region_rank", 0)),
                    contact_indices=np.asarray(candidate["contact_indices"], dtype=int),
                )
            )

    # if cli_args.visualize:
    optimizer.visual_grasp(
        result=grasp_result,
        point_radius=cli_args.point_radius,
        force_scale=cli_args.force_scale,
        include_normals=bool(cli_args.show_normals) and not bool(cli_args.hide_normals),
        include_region_samples=not cli_args.hide_region_samples,
        use_static_equilibrium=True,
    )
