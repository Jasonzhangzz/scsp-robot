import casadi as cs
import numpy as np

import envs.panda_fkin as panda_fkin
from utils import rotations


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


class ExplicitMPCParams:
    def __init__(self, args, rand_seed=1, target_type="rotation", mpc_model="explicit"):
        self.model_path_ = f"./envs/xmls/env_fingertips_{args.obj}.xml"
        self.mesh_path_ = f"envs/assets/objects/{args.obj}.stl"
        self.object_names_ = ["obj"]

        # Keep the stage-2 explicit solver numerically aligned with
        # examples/mpc/franka/direct_control/params.py.
        self.h_ = 0.1
        self.frame_skip_ = 20

        # State layout follows the MuJoCo explicit model:
        # [object xyz(3), object quat_wxyz(4), franka q(7)].
        self.n_robot_qpos_ = 7
        self.n_qpos_ = 14
        self.n_qvel_ = 13
        self.n_cmd_ = 7

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

        self.mu_object_ = 0.5
        self.fingertip_friction_ = self.mu_object_
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_
        self.max_ncon_ = 15

        self.obj_inertia_ = np.identity(6)
        self.obj_inertia_[0:3, 0:3] = 50.0 * np.eye(3)
        self.obj_inertia_[3:, 3:] = 0.1 * np.eye(3)

        joint_action_stiffness = float(
            getattr(args, "joint_action_stiffness", getattr(args, "cartesian_joint_stiffness", 200.0))
        )
        self.robot_stiff_ = np.diag(self.n_cmd_ * [joint_action_stiffness])

        q_matrix = np.zeros((self.n_qvel_, self.n_qvel_))
        q_matrix[:6, :6] = self.obj_inertia_
        q_matrix[6:, 6:] = self.robot_stiff_
        self.Q = q_matrix

        self.obj_mass_ = 0.01
        self.gravity_ = np.array([0.0, 0.0, -9.8, 0.0, 0.0, 0.0], dtype=np.float32)
        self.model_params = float(args.model_param)

        self.mpc_horizon_ = 4
        self.ipopt_max_iter_ = 100
        self.mpc_model = mpc_model

        joint_step = float(getattr(args, "joint_step", 0.2))
        self.mpc_u_lb_ = -joint_step * np.ones((self.n_cmd_,), dtype=np.float32)
        self.mpc_u_ub_ = joint_step * np.ones((self.n_cmd_,), dtype=np.float32)

        obj_pos_lb = np.array([-10.99, -10.99, self.table_height], dtype=np.float32)
        obj_pos_ub = np.array([10.99, 10.99, 0.99], dtype=np.float32)
        self.mpc_q_lb_ = np.hstack((obj_pos_lb, -1e7 * np.ones(4), -1e7 * np.ones(7))).astype(np.float32)
        self.mpc_q_ub_ = np.hstack((obj_pos_ub, 1e7 * np.ones(4), 1e7 * np.ones(7))).astype(np.float32)
        self.sol_guess_ = None
        self.comple_relax = 0.1
        self.max_env_contacts_ = 4

    def init_cost_fns(self):
        x = cs.SX.sym("x", self.n_qpos_)
        u = cs.SX.sym("u", self.n_cmd_)

        obj_pose = x[0:7]
        robot_qpos = x[7:]
        panda_position = panda_fkin.franka_fk(robot_qpos)

        target_position = cs.SX.sym("target_position", 3)
        target_quaternion = cs.SX.sym("target_quaternion", 4)
        phi_vec = cs.SX.sym("phi_vec", self.max_ncon_ * 4)
        jac_mat = cs.SX.sym("jac_mat", self.max_ncon_ * 4, self.n_qvel_)
        cost_param = cs.vvcat([target_position, target_quaternion, phi_vec, jac_mat])

        position_cost = cs.sumsqr(obj_pose[0:3] - target_position)
        quaternion_cost = 1 - cs.dot(obj_pose[3:7], target_quaternion) ** 2
        contact_cost = cs.sumsqr(obj_pose[0:3] - panda_position)

        obj_dirmat = rotations.quat2dcm_fn(obj_pose[3:7])
        obj_v0 = obj_dirmat.T @ (panda_position - obj_pose[0:3])
        grasp_closure = cs.sumsqr(obj_v0 / cs.norm_2(obj_v0))

        control_cost = cs.sumsqr(u)

        obj_tar_vec = obj_pose[0:3] - target_position
        obj_tar_vec = obj_tar_vec / cs.norm_2(obj_tar_vec)
        poi_tar_vec = panda_position - obj_pose[0:3]
        poi_tar_vec = poi_tar_vec / cs.norm_2(poi_tar_vec)
        alignment_cost = -(cs.dot(obj_tar_vec, poi_tar_vec) + 1.0) / 2.0

        q_diff = quaternion_multiply_casadi(target_quaternion, quaternion_conjugate_casadi(obj_pose[3:7]))
        poi_dir_local = cs.mtimes(obj_dirmat.T, poi_tar_vec)
        cos_theta = cs.dot(poi_dir_local, q_diff[1:4])
        rot_alignment_cost = cos_theta ** 2

        base_cost = 0.5 * contact_cost + 0.0 * grasp_closure + 0.0 * alignment_cost + 0.0 * rot_alignment_cost
        final_cost = 500.0 * position_cost + 5.0 * quaternion_cost

        path_cost_fn = cs.Function("path_cost_fn", [x, u, cost_param], [base_cost + 10.0 * control_cost])
        final_cost_fn = cs.Function("final_cost_fn", [x, cost_param], [10.0 * final_cost])
        return path_cost_fn, final_cost_fn
