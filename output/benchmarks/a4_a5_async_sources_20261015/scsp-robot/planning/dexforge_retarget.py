"""Singleton lambda -> ContactAware -> ForceAware retargeting for two Panda arms.

The dataset pipelines assume a MANO hand and a six-DOF mounted wrist. Here the
original CA contact residuals and constrained QP operate on 14 revolute joints;
the resulting one-frame hand/object/contact sequence goes to the original FA
window optimizer. All public object poses are [position, WXYZ quaternion].
"""

import time
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from planning.dexforge_imports import ensure_dexforge


def singleton_pose(pose):
    pose = np.asarray(pose, dtype=np.float64).reshape(1, 7).copy()
    length = np.linalg.norm(pose[0, 3:])
    if not np.isfinite(pose).all() or length < 1.e-10:
        raise ValueError("x_plus_opt must be a finite position/WXYZ pose with a nonzero quaternion")
    pose[0, 3:] /= length
    return pose


def pose_xyzw(pose):
    """DexForge uses [XYZW quaternion, position], unlike the planner state."""
    return singleton_pose(pose)[:, [4, 5, 6, 3, 0, 1, 2]]


def _nonredundant_box_rows(rows, row_lower, lower, upper):
    """Retain inequalities that can be violated anywhere in the QP box."""
    minimum = np.minimum(rows * lower, rows * upper).sum(axis=1)
    # Keep boundary cases conservatively. Only inequalities guaranteed over
    # the entire trust/joint-limit box are omitted from the convex solve.
    return minimum < row_lower + 1.e-8


class _PandaQueries:
    """MuJoCo position and normal Jacobians in CA's scalar-coordinate contract."""

    def __init__(self, fast_step, tip_radius):
        self.model = fast_step.cpu_model
        self.data = mujoco.MjData(self.model)
        ids = fast_step._ids
        self.qadr = np.r_[ids["left_qpos"], ids["right_qpos"]].astype(int)
        self.dadr = np.asarray([
            self.model.jnt_dofadr[np.flatnonzero(self.model.jnt_qposadr == q)[0]]
            for q in self.qadr
        ], dtype=int)
        self.body = np.array([ids["left_tip"], ids["right_tip"]], dtype=int)
        self.tip_offset = np.asarray([ids["left_tip_offset"], ids["right_tip_offset"]])
        self.query_local = self.tip_offset.copy()
        self.local_normal = np.zeros((2, 3))
        self.query_surface_radii = np.full(2, float(tip_radius))
        self.qpos_dim = 14
        uadr = np.r_[ids["left_act"], ids["right_act"]].astype(int)
        joint = np.array([np.flatnonzero(self.model.jnt_qposadr == q)[0] for q in self.qadr])
        joint_limits = np.where(self.model.jnt_limited[joint, None], self.model.jnt_range[joint], [-np.inf, np.inf])
        control_limits = np.where(self.model.actuator_ctrllimited[uadr, None], self.model.actuator_ctrlrange[uadr], [-np.inf, np.inf])
        self.lower = np.maximum(joint_limits[:, 0], control_limits[:, 0])
        self.upper = np.minimum(joint_limits[:, 1], control_limits[:, 1])
        self.base_qpos = self.model.qpos0.copy()
        self._current = None

    def forward(self, q):
        if self._current is not None and np.array_equal(q, self._current):
            return
        self.data.qpos[:] = self.base_qpos
        self.data.qpos[self.qadr] = q
        # CA only needs transforms and Jacobians. Full mj_forward also solves
        # contacts, dynamics and actuators at every line-search point.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        self._current = np.array(q, copy=True)

    def point(self, body, local):
        rotation = self.data.xmat[body].reshape(3, 3)
        position = self.data.xpos[body] + rotation @ local
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jp, jr, position, int(body))
        return position, jp[:, self.dadr], jr[:, self.dadr]

    def query_positions_jacobian(self, q, query_ids):
        self.forward(q)
        values = [self.point(self.body[i], self.query_local[i]) for i in query_ids]
        return np.array([v[0] for v in values]), np.array([v[1] for v in values])

    def query_normals_jacobian(self, q, query_ids):
        self.forward(q)
        normals, rows = [], []
        for i in query_ids:
            body = self.body[i]
            n = self.data.xmat[body].reshape(3, 3) @ self.local_normal[i]
            jr = self.point(body, self.query_local[i])[2]
            normals.append(n)
            rows.append(np.cross(jr.T, n).T)
        return np.array(normals), np.array(rows)


class PandaContactAware:
    """Original CA anchor/normal residuals and sparse QP, without MANO tracking."""

    def __init__(self, fast_step, *, tip_radius=0.01, iterations=20, trust=0.15):
        ensure_dexforge()
        from contactaware.solver.qp import QPSolverOptions
        from contactaware.solver.sparse_qp import make_joint_contact_qp

        self.hand = _PandaQueries(fast_step, tip_radius)
        self.fast_step = fast_step
        self.iterations = max(1, int(iterations))
        self.trust = float(trust)
        if not np.isfinite(self.trust) or self.trust <= 0:
            raise ValueError("CA trust region must be finite and positive")
        self.cfg = SimpleNamespace(contact_anchor_weight=1.0, contact_normal_weight_scale=1.0)
        self.qp = make_joint_contact_qp(QPSolverOptions(1.e-8, 1.e-8, 1.e-6, 200))
        self._reference = None
        self.collision_spheres = None
        if hasattr(fast_step, "compiled") and hasattr(fast_step.compiled.base.collision, "batches"):
            from forceaware.mpc_loss import _collision_source_clouds

            self.collision_spheres = _collision_source_clouds(fast_step.compiled.base.collision)

    def reset(self):
        self._reference = None

    def solve(self, qpos, *, object_pose_wxyz, contact_points_local, normals_local, approach_offset=None):
        from contactaware.types import ContactAnchorSet, ContactRuntime
        from contactaware.solver.contact import anchor_constraint_linearization, anchor_normal_linearization
        from contactaware.solver.qp import QPProblem, SolverFailure

        started = time.perf_counter()
        pose = singleton_pose(object_pose_wxyz)
        contacts = np.asarray(contact_points_local, dtype=np.float64).reshape(1, 2, 3)
        normals = np.asarray(normals_local, dtype=np.float64).reshape(1, 2, 3).copy()
        lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
        if not np.isfinite(contacts).all() or not np.isfinite(normals).all() or np.any(lengths < 1.e-10):
            raise ValueError("Lambda contacts and inward normals must be finite; normals must be nonzero")
        normals /= lengths
        h = self.hand
        h.base_qpos = np.asarray(qpos, dtype=np.float64).reshape(h.model.nq).copy()
        obj = int(self.fast_step._ids["obj_qpos"])
        h.base_qpos[obj:obj + 7] = pose[0]
        h._current = None
        initial = h.base_qpos[h.qadr].copy()
        h.forward(initial)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, pose[0, 3:])
        inward_world = normals[0] @ rotation.reshape(3, 3).T
        # A spherical pad may select any point on its surface. Freeze that
        # body-local query through CA and FA so both stages track the same point.
        for i, body in enumerate(h.body):
            h.local_normal[i] = h.data.xmat[body].reshape(3, 3).T @ inward_world[i]
        radius = h.query_surface_radii[0] if approach_offset is None else float(approach_offset)
        if not np.isfinite(radius) or not 0.0 < radius <= h.query_surface_radii[0]:
            raise ValueError("CA approach_offset must be positive and no larger than the fingertip radius")
        h.query_local = h.tip_offset + radius * h.local_normal
        anchors = ContactAnchorSet(np.ones((1, 2), bool), contacts, -normals,
                                   np.arange(2), np.full(2, -1), {"source": "lambda_optimizer"})
        runtime = ContactRuntime(anchors, np.array([[0, 1]], np.int32), np.ones((1, 2)),
                                 contacts.copy(), np.ones((1, 2)))
        arguments = dict(cfg=self.cfg, runtime=runtime, frame_id=0)
        dex_pose = pose_xyzw(pose)[0]

        def objective(q):
            points, jac, targets, _ = anchor_constraint_linearization(h, q, dex_pose, **arguments)
            normal, normal_jac, _ = anchor_normal_linearization(h, q, dex_pose, **arguments)
            # Uniform radian proximal term: Panda has no translating wrist.
            residual = np.r_[(points - targets).ravel(), normal.ravel(), 0.002 * (q - initial)]
            rows = np.vstack((jac.reshape(-1, 14), normal_jac.reshape(-1, 14), 0.002 * np.eye(14)))
            return residual, rows

        # Geometry and pair topology are constant through this CA solve.
        ids = self.fast_step._ids
        bodies = [ids["left_forearm"], ids["left_wrist"], ids["left_tip"],
                  ids["right_forearm"], ids["right_wrist"], ids["right_tip"]]
        offsets = [np.zeros(3), np.zeros(3), h.tip_offset[0],
                   np.zeros(3), np.zeros(3), h.tip_offset[1]]
        radii = [0.045, 0.035, h.query_surface_radii[0]] * 2
        sides = [0, 0, 0, 1, 1, 1]
        if self.collision_spheres is not None:
            bodies, offsets, radii, sides = [], [], [], []
            for body, cloud in sorted(self.collision_spheres.items()):
                name = mujoco.mj_id2name(h.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
                if name.startswith("left_") or name.startswith("right_"):
                    for sphere in cloud:
                        bodies.append(body)
                        offsets.append(sphere[:3])
                        radii.append(sphere[3])
                        sides.append(0 if name.startswith("left_") else 1)
        radii = np.asarray(radii)
        pairs = np.asarray([(i, j) for i in range(len(bodies)) for j in range(i + 1, len(bodies))
                            if sides[i] != sides[j]], dtype=int).reshape(-1, 2)
        spheres = getattr(self.fast_step, "spheres_local", None)
        object_tree = None
        if spheres is not None:
            spheres = spheres.detach().cpu().numpy() if hasattr(spheres, "detach") else np.asarray(spheres)
            centers = spheres[:, :3] @ rotation.reshape(3, 3).T + pose[0, :3]
            # Equal-radius GS reduces the exact nearest surface query to a
            # nearest-centre query. Unequal radii retain the full gap search.
            if len(spheres) and np.ptp(spheres[:, 3]) == 0:
                object_tree = cKDTree(centers)
        terminal = np.isin(bodies, h.body)

        def collision_rows(q):
            h.forward(q)
            values = [h.point(b, off) for b, off in zip(bodies, offsets)]
            positions = np.asarray([v[0] for v in values])
            jacobians = np.asarray([v[1] for v in values])
            left, right = pairs.T
            delta = positions[left] - positions[right]
            length = np.maximum(np.linalg.norm(delta, axis=1), 1.e-10)
            distances = length - radii[left] - radii[right] - 0.002
            rows = np.einsum("ni,nij->nj", delta / length[:, None], jacobians[left] - jacobians[right])
            if spheres is not None:
                index = np.arange(len(bodies))
                if object_tree is None:
                    delta = positions[:, None] - centers
                    length = np.maximum(np.linalg.norm(delta, axis=2), 1.e-10)
                    gap = length - spheres[None, :, 3] - radii[:, None]
                    support = np.argmin(gap, axis=1)
                    selected_delta, selected_length, selected_gap = delta[index, support], length[index, support], gap[index, support]
                else:
                    nearest_distance, nearest = object_tree.query(positions, k=2)
                    support = nearest[:, 0]
                    # Match np.argmin's first-index tie rule at symmetric or
                    # duplicate centres so the selected constraint is stable.
                    tied = nearest_distance[:, 1] - nearest_distance[:, 0] < 1.e-12
                    for i in np.flatnonzero(tied):
                        support[i] = np.argmin(np.linalg.norm(positions[i] - centers, axis=1))
                    selected_delta = positions - centers[support]
                    selected_length = np.maximum(np.linalg.norm(selected_delta, axis=1), 1.e-10)
                    selected_gap = selected_length - spheres[support, 3] - radii
                # Exactly the former nearest-support constraint, including GS
                # inflation at pads and positive clearance at wrists/forearms.
                clearance = np.where(terminal, radius - h.query_surface_radii[0] - spheres[support, 3], 0.002)
                distances = np.r_[distances, selected_gap - clearance]
                direction = selected_delta / selected_length[:, None]
                rows = np.vstack((rows, np.einsum("ni,nij->nj", direction, jacobians)))
            return distances, rows

        q = np.clip(initial, h.lower, h.upper)
        if self._reference is not None:
            candidate = np.clip(self._reference, h.lower, h.upper)
            candidate_clearance = collision_rows(candidate)[0]
            initial_clearance = collision_rows(q)[0]
            if (np.all(candidate_clearance >= np.minimum(initial_clearance, 0.0) - 1.e-6)
                    and np.linalg.norm(objective(candidate)[0]) < np.linalg.norm(objective(q)[0])):
                q = candidate
        status, iterations = "iteration_limit", 0
        constraint_counts = []
        for iterations in range(1, self.iterations + 1):
            residual, jac = objective(q)
            distances, rows = collision_rows(q)
            lower, upper = np.maximum(h.lower - q, -self.trust), np.minimum(h.upper - q, self.trust)
            row_lower = -np.maximum(distances, 0.0)
            keep = _nonredundant_box_rows(rows, row_lower, lower, upper)
            constraint_counts.append(int(keep.sum()))
            problem = QPProblem(jac.T @ jac + 1.e-6 * np.eye(14), jac.T @ residual,
                                rows[keep], row_lower[keep], np.full(int(keep.sum()), np.inf), lower, upper)
            try:
                delta = self.qp.solve(problem, label="singleton Panda CA").solution
            except SolverFailure:
                status = "qp_failed"
                break
            if np.max(np.abs(delta)) < 1.e-6:
                status = "step_tolerance"
                break
            accepted = False
            for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = np.clip(q + fraction * delta, h.lower, h.upper)
                next_distances, _ = collision_rows(candidate)
                if (np.all(next_distances >= np.minimum(distances, 0.0) - 1.e-6)
                        and np.linalg.norm(objective(candidate)[0]) < np.linalg.norm(residual)):
                    q, accepted = candidate, True
                    break
            if not accepted:
                status = "line_search_tolerance"
                break
        self._reference = q.copy()
        positions, _, targets, _ = anchor_constraint_linearization(h, q, dex_pose, **arguments)
        clearance = float(np.min(collision_rows(q)[0]))
        return dict(hand_reference=q.reshape(1, 14), object_pose_wxyz=pose,
                    contact_points_local=contacts.copy(), normals_local=normals.copy(),
                    query_local=h.query_local.reshape(1, 2, 3).copy(), contact_runtime=runtime,
                    diagnostics=dict(status=status, iterations=iterations,
                                     contact_error=np.linalg.norm(positions - targets, axis=1),
                                     collision_clearance=clearance,
                                     collision_constraints=len(pairs) + (len(bodies) if spheres is not None else 0),
                                     qp_constraint_counts=constraint_counts,
                                     object_search="kdtree" if object_tree is not None else "full_gap",
                                     solve_time=time.perf_counter() - started))


class DexForgeRetargetingMPC:
    """Receding-horizon facade: one lambda frame, CA solve, then native FA."""

    def __init__(self, param, warp_model, *, solver="adam"):
        from planning.dexforge_forceaware import PandaForceAware

        if solver not in {"adam", "mppi"}:
            raise ValueError(f"Unsupported FA solver: {solver}")
        self.solver = solver
        self.warp_model = warp_model
        self.horizon = max(1, int(param.adam_horizon))
        self.knot_substeps = max(1, int(param.adam_knot_substeps))
        self.iters, self.lr = int(param.adam_iters), float(param.adam_lr)
        self.u_limit = float(param.planner_joint_delta_limit)
        self.object_speed = float(getattr(param, "retarget_object_speed", np.inf))
        self.object_angular_speed = float(getattr(param, "retarget_object_angular_speed", np.inf))
        if any(np.isnan(value) or value <= 0 for value in (self.object_speed, self.object_angular_speed)):
            raise ValueError("Retarget object speed limits must be positive")
        self.device = warp_model.device_name
        self.ca = PandaContactAware(warp_model, tip_radius=float(param.tip_radius),
                                    iterations=int(getattr(param, "retarget_ca_iters", 20)))
        options = dict(adam_multistart_count=int(getattr(param, "adam_multistart_count", 2)))
        if solver == "mppi":
            self.iters = int(param.mppi_iterations_)
            self.num_samples = max(2, int(param.mppi_samples_))
            options = dict(num_samples=self.num_samples, mppi_iterations=self.iters,
                           mppi_init_iterations=int(param.mppi_init_iterations_),
                           temperature=float(param.mppi_lambda_),
                           noise_scale=float(param.mppi_noise_sigma_),
                           noise_decay=float(param.mppi_noise_decay_), seed=param.mppi_seed_)
        self.fa = PandaForceAware(warp_model, horizon=self.horizon, iters=self.iters,
                                 lr=self.lr, u_limit=self.u_limit, knot_substeps=self.knot_substeps,
                                 solver=solver, **options)

    def reset(self):
        self.ca.reset()
        reset = getattr(self.fa, "reset", None)
        if callable(reset):
            reset()

    def plan_once(self, target_p=None, target_q=None, curr_x=None, *, x_plus_opt,
                  contact_points_local, normals_local, qpos0, qvel0, approach_offset=None,
                  contact_established=True, **kwargs):
        del target_p, target_q, curr_x, kwargs
        started = time.perf_counter()
        prediction = singleton_pose(x_plus_opt)
        reference = prediction.copy()
        obj = int(self.warp_model._ids["obj_qpos"])
        measured = singleton_pose(np.asarray(qpos0)[obj:obj + 7])
        if not contact_established:
            reference = measured.copy()
        else:
            dt = float(self.warp_model.cpu_model.opt.timestep) * self.knot_substeps
            displacement = prediction[0, :3] - measured[0, :3]
            fraction = min(1.0, self.object_speed * dt / max(np.linalg.norm(displacement), 1.e-12))
            reference[0, :3] = measured[0, :3] + fraction * displacement
            start, goal = measured[0, 3:], prediction[0, 3:].copy()
            if start @ goal < 0:
                goal *= -1
            half_angle = np.arccos(np.clip(start @ goal, -1, 1))
            if half_angle > 1.e-8:
                fraction = min(1.0, self.object_angular_speed * dt / (2 * half_angle))
                reference[0, 3:] = (np.sin((1 - fraction) * half_angle) * start
                                    + np.sin(fraction * half_angle) * goal) / np.sin(half_angle)
        ca = self.ca.solve(qpos0, object_pose_wxyz=reference,
                           contact_points_local=contact_points_local, normals_local=normals_local,
                           approach_offset=approach_offset)
        ca_seconds = time.perf_counter() - started
        fa_started = time.perf_counter()
        result = self.fa.solve(qpos0, qvel0, **{name: ca[name] for name in (
            "hand_reference", "object_pose_wxyz", "contact_points_local", "normals_local", "query_local")})
        fa_seconds = time.perf_counter() - fa_started
        result["retargeting"] = dict(source_frames=1, x_plus_opt=ca["object_pose_wxyz"].copy(),
                                     lambda_x_plus_opt=prediction.copy(),
                                     phase="manipulate" if contact_established else "approach",
                                     hand_reference=ca["hand_reference"].copy(), ca=ca["diagnostics"],
                                     ca_time=ca_seconds, fa_time=fa_seconds)
        result["solve_time"] = time.perf_counter() - started
        return result
