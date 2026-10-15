import casadi as cs
import numpy as np
import os

from utils import rotations


current_dir = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(current_dir)
while os.path.basename(REPO_ROOT) != "scsp-robot":
    _next_dir = os.path.dirname(REPO_ROOT)
    if _next_dir == REPO_ROOT:
        raise RuntimeError("scsp-robot repo root not found from %s" % current_dir)
    REPO_ROOT = _next_dir


def _normalize_quaternion_wxyz(quat_wxyz):
    quat_norm = cs.sqrt(cs.dot(quat_wxyz, quat_wxyz) + 1e-12)
    return quat_wxyz / quat_norm


def _safe_normalize(vec):
    return vec / (cs.norm_2(vec) + 1e-8)


def _quaternion_alignment_cost(curr_quat_wxyz, target_quat_wxyz):
    curr_quat_wxyz = _normalize_quaternion_wxyz(curr_quat_wxyz)
    target_quat_wxyz = _normalize_quaternion_wxyz(target_quat_wxyz)
    return 1.0 - cs.dot(curr_quat_wxyz, target_quat_wxyz) ** 2


def quaternion_conjugate_casadi(q):
    return cs.vertcat(q[0], -q[1], -q[2], -q[3])


def quaternion_multiply_casadi(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return cs.vertcat(
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _resolve_mesh_path(mesh_path, repo_root=REPO_ROOT):
    if mesh_path is None:
        return None

    candidate_paths = [mesh_path] if os.path.isabs(mesh_path) else [
        os.path.join(repo_root, mesh_path),
        os.path.join(os.getcwd(), mesh_path),
        os.path.abspath(mesh_path),
    ]
    for candidate_path in candidate_paths:
        candidate_path = os.path.abspath(candidate_path)
        if os.path.isfile(candidate_path):
            return candidate_path
    return None


def _estimate_approach_offset_from_mesh(mesh_path):
    mesh_abs_path = _resolve_mesh_path(mesh_path)
    if mesh_abs_path is None:
        return 0.05

    try:
        import trimesh
    except ImportError:
        return 0.05

    try:
        mesh = trimesh.load(mesh_abs_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        bounds = np.asarray(mesh.bounds, dtype=np.float32)
        extents = bounds[1] - bounds[0]
        half_max_extent = 0.5 * float(np.max(extents))
        return float(np.clip(half_max_extent + 0.012, 0.03, 0.12))
    except Exception:
        return 0.05


class ExplicitMPCParams:
    def __init__(self, args, rand_seed=1, target_type="rotation", mpc_model="explicit"):
        self.contact_cost_param = float(np.clip(getattr(args, "contact_cost_param", 0.0), 0.0, 1.0))
        self.model_path_ = f"./envs/xmls/env_fingertips_{args.obj}.xml"
        self.mesh_path_ = f"envs/assets/objects/{args.obj}.stl"
        self.object_names_ = ["obj"]

        self.h_ = 0.01
        self.frame_skip_ = 10

        # Planner state layout:
        # [object xyz(3), object quat_wxyz(4), end-effector point xyz(3)].
        self.n_robot_qpos_ = 3
        self.n_qpos_ = 10
        self.n_qvel_ = 9
        self.n_cmd_ = 3

        self.jc_kp_ = 200
        self.jc_damping_ = 10
        self.proximity_threshold_ = 0.1
        self.fingertip_geoms = [
            "left_finger_tip_pad_1",
            "left_finger_tip_pad_2",
            "left_finger_tip_pad_3",
            "left_finger_tip_pad_4",
            "left_finger_tip_pad_5",
            "right_finger_tip_pad_1",
            "right_finger_tip_pad_2",
            "right_finger_tip_pad_3",
            "right_finger_tip_pad_4",
            "right_finger_tip_pad_5",
        ]

        seed_base = int(getattr(args, "seed", getattr(args, "init_rand_seed", 100)))
        self.init_rand_seed_base_ = seed_base
        self.random_seed_ = seed_base + int(rand_seed)
        self.random_generator_ = np.random.default_rng(self.random_seed_)

        self.table_height = 0.35
        init_height = 0.05 + self.table_height

        init_xy_rand = getattr(args, "init_xy_rand", None)
        if init_xy_rand is None:
            init_xy_rand = 0.1 * self.random_generator_.random(2)
            init_xy_rand[0] += 0.3
        init_xy_rand = np.asarray(init_xy_rand, dtype=np.float32).reshape(2)

        init_obj_quat_rand = getattr(args, "init_obj_quat_rand", None)
        if init_obj_quat_rand is None:
            yaw_angle = -np.pi * float(self.random_generator_.random()) + np.pi / 2
            init_obj_quat_rand = rotations.rpy_to_quaternion(
                np.array([yaw_angle, np.pi / 2, -np.pi / 2], dtype=np.float32)
            )
        init_obj_quat_rand = np.asarray(init_obj_quat_rand, dtype=np.float32).reshape(4)
        init_obj_quat_rand = init_obj_quat_rand / max(np.linalg.norm(init_obj_quat_rand), 1e-8)

        self.init_xy_rand_ = init_xy_rand.astype(np.float32).copy()
        self.init_obj_quat_rand_ = np.asarray(init_obj_quat_rand, dtype=np.float32).copy()
        self.init_obj_qpos_ = np.hstack((init_xy_rand, init_height, init_obj_quat_rand)).astype(np.float32)
        self.init_robot_qpos_ = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)

        if target_type != "rotation":
            raise ValueError(f"Target type {target_type} not supported")

        target_p = getattr(args, "target_p", None)
        if target_p is None:
            target_p = np.array([0.31, 0.0, init_height - 0.02], dtype=np.float32)
        self.target_p_ = np.asarray(target_p, dtype=np.float32).reshape(3)

        target_q = getattr(args, "target_q", None)
        if target_q is None:
            target_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        target_q = np.asarray(target_q, dtype=np.float32).reshape(4)
        self.target_q_ = target_q / max(np.linalg.norm(target_q), 1e-8)

        self.mu_object_ = 1.2
        self.fingertip_friction_ = self.mu_object_
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_
        self.max_ncon_ = 10

        self.obj_inertia_ = np.identity(6)
        self.obj_inertia_[0:3, 0:3] = 50.0 * np.eye(3)
        self.obj_inertia_[3:, 3:] = 0.05 * np.eye(3)

        point_action_stiffness = float(
            getattr(args, "joint_action_stiffness", getattr(args, "cartesian_joint_stiffness", 300.0))
        )
        self.robot_stiff_ = np.diag(self.n_cmd_ * [point_action_stiffness])

        q_matrix = np.zeros((self.n_qvel_, self.n_qvel_))
        q_matrix[:6, :6] = self.obj_inertia_
        q_matrix[6:, 6:] = self.robot_stiff_
        self.Q = q_matrix

        self.obj_mass_ = 0.1
        self.gravity_ = np.array([0.0, 0.0, -9.8, 0.0, 0.0, 0.0], dtype=np.float32)
        self.model_params = float(args.model_param)
        approach_offset_arg = getattr(args, "approach_offset", None)
        if approach_offset_arg is None:
            self.point_approach_offset_ = float(_estimate_approach_offset_from_mesh(self.mesh_path_))
        else:
            self.point_approach_offset_ = float(approach_offset_arg)
        self.point_path_cost_weight_ = float(getattr(args, "point_path_cost_weight", 60.0))
        self.point_final_cost_weight_ = float(getattr(args, "point_final_cost_weight", 200.0))
        self.point_control_cost_weight_ = float(getattr(args, "point_control_cost_weight", 0.5))

        self.mpc_horizon_ = 5
        self.ipopt_max_iter_ = 100
        self.mpc_model = mpc_model

        point_step = float(getattr(args, "joint_step", 0.1))
        self.mpc_u_lb_ = -point_step * np.ones((self.n_cmd_,), dtype=np.float32)
        self.mpc_u_ub_ = point_step * np.ones((self.n_cmd_,), dtype=np.float32)

        point_lb = np.array([-1.0, -1.0, self.table_height + 0.02], dtype=np.float32)
        point_ub = np.array([1.5, 1.0, 1.5], dtype=np.float32)
        self.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7, dtype=np.float32), point_lb)).astype(np.float32)
        self.mpc_q_ub_ = np.hstack((1e7 * np.ones(7, dtype=np.float32), point_ub)).astype(np.float32)
        self.sol_guess_ = None
        self.comple_relax = 0.1
        self.max_env_contacts_ = 4

    def init_cost_fns(self):
        x = cs.SX.sym("x", self.n_qpos_)
        u = cs.SX.sym("u", self.n_cmd_)

        obj_pose = x[0:7]
        point_pos = x[7:10]

        target_position = cs.SX.sym("target_position", 3)
        target_quaternion = cs.SX.sym("target_quaternion", 4)
        phi_vec = cs.SX.sym("phi_vec", self.max_ncon_ * 4)
        jac_mat = cs.SX.sym("jac_mat", self.max_ncon_ * 4, self.n_qvel_)
        cost_param = cs.vvcat([target_position, target_quaternion, phi_vec, jac_mat])

        position_cost = cs.sumsqr(obj_pose[0:3] - target_position)
        quaternion_cost = _quaternion_alignment_cost(obj_pose[3:7], target_quaternion)
        contact_cost = cs.sumsqr(obj_pose[0:3] - point_pos)
        approach_direction = _safe_normalize(obj_pose[0:3] - target_position)
        approach_point = obj_pose[0:3] + self.point_approach_offset_ * approach_direction
        approach_cost = cs.sumsqr(point_pos - approach_point)
        point_tracking_cost = self.contact_cost_param * contact_cost + (1.0 - self.contact_cost_param) * approach_cost

        obj_dirmat = rotations.quat2dcm_fn(obj_pose[3:7])
        obj_to_point_world = point_pos - obj_pose[0:3]
        obj_v0 = obj_dirmat.T @ obj_to_point_world
        grasp_closure = cs.sumsqr(_safe_normalize(obj_v0))

        control_cost = cs.sumsqr(u)

        obj_tar_vec = _safe_normalize(obj_pose[0:3] - target_position)
        poi_tar_vec = _safe_normalize(obj_to_point_world)
        alignment_cost = -(cs.dot(obj_tar_vec, poi_tar_vec) + 1.0) / 2.0

        q_diff = quaternion_multiply_casadi(target_quaternion, quaternion_conjugate_casadi(obj_pose[3:7]))
        poi_dir_local = obj_dirmat.T @ poi_tar_vec
        rot_alignment_cost = cs.dot(poi_dir_local, q_diff[1:4]) ** 2

        base_cost = (
            self.point_path_cost_weight_ * point_tracking_cost
            + 2.0 * alignment_cost
            + 0.0 * grasp_closure
            + 0.0 * rot_alignment_cost
        )
        final_cost = (
            500.0 * position_cost
            + 5.0 * quaternion_cost
            + self.point_final_cost_weight_ * point_tracking_cost
        )

        path_cost_fn = cs.Function(
            "path_cost_fn",
            [x, u, cost_param],
            [base_cost + self.point_control_cost_weight_ * control_cost],
        )
        final_cost_fn = cs.Function("final_cost_fn", [x, cost_param], [10.0 * final_cost])
        return path_cost_fn, final_cost_fn
