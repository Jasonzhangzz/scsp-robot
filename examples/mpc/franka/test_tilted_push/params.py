import casadi as cs
import numpy as np
import trimesh

from examples.mpc.franka.ik2.params import build_lambda_optimizer
from planning.attract_function import compute_scalar_potential_and_gradient
from utils import rotations


class ExplicitMPCParams:
    def __init__(self, args, rand_seed=1, target_type="rotation", mpc_model="explicit"):
        self.contact_cost_param = float(args.contact_cost_param)
        self.attract_coef = float(args.attract_coef)
        self.field_cost_weight = 0.05
        self.quadratic_contact_track = False
        self.reject_coef = args.reject_coef
        self.spline_escape_cost = bool(getattr(args, "spline_escape_cost", 0))
        self.contact_coef = args.contact_coef
        self.reject_dis = args.reject_dis
        self.pos_coef = float(getattr(args, "pos_coef", 500.0))
        self.ori_coef = float(getattr(args, "ori_coef", 20.0))
        self.smooth_contact_detour = bool(getattr(args, "ideal_contact_pose", False))
        self.detour_attract_coef = float(getattr(args, "detour_attract_coef", 80.0))
        self.detour_repel_coef = float(getattr(args, "detour_repel_coef", 40.0))
        self.detour_lift_coef = float(getattr(args, "detour_lift_coef", 25.0))
        self.detour_align_thresh = float(getattr(args, "detour_align_thresh", 0.50))
        self.detour_align_sharpness = float(getattr(args, "detour_align_sharpness", 8.0))
        self.object_circumradius = 0.08
        self.object_aabb_lo = np.array([-0.06, -0.04, -0.04], dtype=np.float64)
        self.object_aabb_hi = np.array([0.06, 0.04, 0.06], dtype=np.float64)

        self.model_path_ = "./envs/xmls/env_fingertips_" + args.obj + ".xml"
        self.source_mesh_path_ = "envs/assets/objects/" + args.obj + ".stl"
        self.mesh_path_ = self.source_mesh_path_
        self.visual_mesh_path_ = self.source_mesh_path_
        self.object_names_ = ["obj"]
        requested_hull = getattr(args, "collision_hull", None)
        self.collision_hull = True if requested_hull is None else bool(requested_hull)
        self._collision_mesh_extracted = False
        try:
            bounds = np.asarray(
                trimesh.load_mesh(self.mesh_path_, process=False).bounds,
                dtype=np.float64,
            )
            self.object_aabb_lo = bounds[0].copy()
            self.object_aabb_hi = bounds[1].copy()
            self.object_circumradius = float(np.linalg.norm(0.5 * (bounds[1] - bounds[0])))
        except Exception:
            self.object_circumradius = 0.08

        self.frame_skip_ = int(10)
        self.h_ = 0.02
        self.lambda_h_ = 0.05

        self.n_robot_qpos_ = 3
        self.n_qpos_ = 10
        self.n_qvel_ = 9
        self.n_cmd_ = 3

        self.jc_kp_ = 200
        self.jc_damping_ = 10
        self.proximity_threshold_ = 0.1
        self.fingertip_geoms = [
            "left_finger_tip_pad_1", "left_finger_tip_pad_2", "left_finger_tip_pad_3",
            "left_finger_tip_pad_4", "left_finger_tip_pad_5",
            "right_finger_tip_pad_1", "right_finger_tip_pad_2", "right_finger_tip_pad_3",
            "right_finger_tip_pad_4", "right_finger_tip_pad_5",
        ]

        seed_base = int(getattr(args, "seed", getattr(args, "init_rand_seed", 100)))
        self.init_rand_seed_base_ = seed_base
        self.random_seed_ = seed_base + int(rand_seed)
        self.random_generator_ = np.random.default_rng(self.random_seed_)
        self.set_to_goal_pose_ = bool(getattr(args, "set_to_goal_pose", False))
        self.set_to_goal_pose_xyz_noise_ = np.asarray(
            getattr(args, "set_to_goal_pose_xyz_noise", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        ).reshape(3)
        self.set_to_goal_pose_yaw_noise_deg_ = float(getattr(args, "set_to_goal_pose_yaw_noise_deg", 0.0))

        self.table_height = 0.35
        init_height = 0.05 + self.table_height
        init_xy_rand = 0.1 * self.random_generator_.random(2)
        init_xy_rand[0] += 0.3

        yaw_angle = -np.pi * float(self.random_generator_.random()) + np.pi / 2
        init_obj_quat_rand = rotations.rpy_to_quaternion(
            np.array([yaw_angle + np.pi / 3, 0, 0], dtype=np.float32)
        )
        self.init_xy_rand_ = init_xy_rand.astype(np.float32).copy()
        self.init_obj_quat_rand_ = np.asarray(init_obj_quat_rand, dtype=np.float32).copy()
        self.init_obj_qpos_ = np.hstack((init_xy_rand, init_height, init_obj_quat_rand))
        self.init_robot_qpos_ = np.array([0.0, -0.785, 0.0, -2.356, 0, 1.571, 0.785])

        if target_type == "rotation":
            target_p = getattr(args, "target_p", None)
            if target_p is None:
                target_p = np.array([0.31, 0.0, init_height - 0.02], dtype=np.float32)
            self.target_p_ = np.asarray(target_p, dtype=np.float32).reshape(3)

            target_q = getattr(args, "target_q", None)
            if target_q is None:
                target_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            target_q = np.asarray(target_q, dtype=np.float32).reshape(4)
            self.target_q_ = target_q / max(np.linalg.norm(target_q), 1e-8)
        else:
            raise ValueError(f"Target type {target_type} not supported")

        self.mu_object_ = 0.5
        self.fingertip_friction_ = self.mu_object_
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_
        self.max_ncon_ = 10

        self.obj_inertia_ = np.identity(6)
        self.obj_inertia_[0:3, 0:3] = 50 * np.eye(3)
        self.obj_inertia_[3:, 3:] = 0.05 * np.eye(3)
        self.robot_stiff_ = np.diag(self.n_cmd_ * [100])

        Q = np.zeros((self.n_qvel_, self.n_qvel_))
        Q[:6, :6] = self.obj_inertia_
        Q[6:, 6:] = self.robot_stiff_
        self.Q = Q

        self.obj_mass_ = 0.01
        self.lambda_obj_mass_ = 0.01
        self.sim_obj_mass_ = 0.01
        self.gravity_ = np.array([0.00, 0.00, -9.8, 0.0, 0.0, 0.0])
        self.model_params = args.model_param

        self.mpc_model = mpc_model
        self.torch_solver = getattr(args, "solver", "acados")
        self.planner_solver_ = "acados"
        self.mpc_horizon_ = 5
        self.ipopt_max_iter_ = 100
        self.comple_relax = 0.01

        self.mpc_u_lb_ = -float(getattr(args, "mpc_step_limit", 0.005))
        self.mpc_u_ub_ = -self.mpc_u_lb_
        fts_q_lb = np.array([-10, -10, self.table_height - 0.01])
        fts_q_ub = np.array([10, 10, self.table_height + 1.0])
        self.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7), fts_q_lb))
        self.mpc_q_ub_ = np.hstack((1e7 * np.ones(7), fts_q_ub))
        self.sol_guess_ = None
        self.max_env_contacts_ = 4
        self.lambda_optimizer = build_lambda_optimizer(self, args)

    @staticmethod
    def calculate_rotation_quaternion(x, target_position):
        direction = x[:2] - target_position[:2]
        direction = direction / cs.sqrt(cs.sumsqr(direction) + 1e-9)
        angle = cs.arctan2(direction[1], direction[0])
        half_angle = angle / 2.0
        return [cs.cos(half_angle), 0, 0, cs.sin(half_angle)]

    def init_cost_fns(self):
        x = cs.SX.sym("x", self.n_qpos_)
        u = cs.SX.sym("u", self.n_cmd_)

        target_position = cs.SX.sym("target_position", 3)
        target_quaternion = cs.SX.sym("target_quaternion", 4)
        position_cost = cs.sumsqr(x[0:3] - target_position)
        quaternion_cost = 1 - cs.dot(x[3:7], target_quaternion) ** 2
        contact_cost = cs.sumsqr(x[0:3] - x[7:10])
        control_cost = cs.sumsqr(u)
        virtual_point = cs.SX.sym("virtual_point", 3)
        contact_point = cs.SX.sym("contact point", 3)

        phi_vec = cs.SX.sym("phi_vec", self.max_ncon_ * 4)
        jac_mat = cs.SX.sym("jac_mat", self.max_ncon_ * 4, self.n_qvel_)
        verify_cost_param = cs.SX.sym("verify_cost", 1)
        cost_param = cs.vvcat(
            [target_position, target_quaternion, phi_vec, jac_mat, verify_cost_param, virtual_point, contact_point]
        )
        if getattr(self, "smooth_contact_detour", False):
            base_cost = self._smooth_contact_detour_cost(x, virtual_point, contact_point)
            final_cost = (
                500 * position_cost
                + 20.0 * quaternion_cost
                + 20.0 * self.detour_attract_coef * cs.sumsqr(x[7:10] - virtual_point)
            )
            control_weight = 8.0
        else:
            direction_quat = self.calculate_rotation_quaternion(x, target_position)
            field_cost = compute_scalar_potential_and_gradient(
                x[7], x[8], x[9],
                center=x[:3],
                quaternion=direction_quat,
                distance=0.5,
                m_magnitude=0.5,
            )[0]
            if getattr(self, "quadratic_contact_track", False):
                virtual_point_cost = cs.sumsqr(x[7:10] - virtual_point)
            else:
                virtual_point_cost = self.log_barrier_function(x, virtual_point)

            reject_distance = cs.sumsqr(x[0:2] - x[7:9]) + 1e-3
            obstacle_cost = (
                cs.DM(0)
                if self.spline_escape_cost
                else cs.if_else(reject_distance < self.reject_dis, 1 / reject_distance, 0.0)
            )
            attract_cost = (
                self.attract_coef * virtual_point_cost
                + float(getattr(self, "field_cost_weight", 0.05)) * field_cost
                + self.reject_coef * obstacle_cost
            )
            contact_point_cost = (
                cs.sumsqr(x[7:10] - contact_point)
                if getattr(self, "quadratic_contact_track", False)
                else self.log_barrier_function(x, contact_point)
            )
            press_cost = contact_point_cost
            if float(getattr(self, "contact_cost_param", 0.0)) > 0.0 and not getattr(
                self, "rollout_press_patch", False
            ):
                press_cost = (
                    self.contact_cost_param * contact_cost
                    + (1 - self.contact_cost_param) * contact_point_cost
                )
            base_cost = (1 - verify_cost_param) * attract_cost + self.contact_coef * verify_cost_param * press_cost
            final_cost = (
                float(getattr(self, "pos_coef", 500.0)) * position_cost
                + float(getattr(self, "ori_coef", 20.0)) * quaternion_cost
            )
            control_weight = 50.0

        path_cost_fn = cs.Function("path_cost_fn", [x, u, cost_param], [base_cost + control_weight * control_cost])
        final_cost_fn = cs.Function("final_cost_fn", [x, cost_param], [10 * final_cost])
        return path_cost_fn, final_cost_fn

    @staticmethod
    def _smooth_relu(z, eps=1e-4):
        return 0.5 * (z + cs.sqrt(z * z + eps))

    @staticmethod
    def _smooth_gate(value, threshold, sharpness=8.0):
        return 0.5 * (1.0 + cs.tanh(float(sharpness) * (value - threshold)))

    @staticmethod
    def _quat_wxyz_to_rot(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        return cs.vertcat(
            cs.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            cs.horzcat(2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            cs.horzcat(2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )

    def _object_top_z(self, obj_pos, obj_quat):
        lo = np.asarray(self.object_aabb_lo, dtype=np.float64)
        hi = np.asarray(self.object_aabb_hi, dtype=np.float64)
        half = cs.DM(0.5 * (hi - lo))
        center_local = cs.DM(0.5 * (hi + lo))
        R = self._quat_wxyz_to_rot(obj_quat)
        center_world = obj_pos + R @ center_local
        z_extent = (
            cs.fabs(R[2, 0]) * half[0]
            + cs.fabs(R[2, 1]) * half[1]
            + cs.fabs(R[2, 2]) * half[2]
        )
        return center_world[2] + z_extent

    def _smooth_contact_detour_cost(self, x, virtual_point, contact_point):
        tip = x[7:10]
        obj = x[0:3]
        goal = virtual_point
        surface = contact_point

        patch_n = goal - surface
        patch_n = patch_n / cs.sqrt(cs.sumsqr(patch_n) + 1e-9)
        radial = tip - obj
        r = cs.sqrt(cs.sumsqr(radial) + 1e-9)
        align = cs.dot(radial / r, patch_n)
        dist_goal = cs.sqrt(cs.sumsqr(tip - goal) + 1e-12)
        arrive = self._smooth_gate(0.03 - dist_goal, 0.0, 20.0)
        approach_gate = cs.fmax(
            self._smooth_gate(align, self.detour_align_thresh, self.detour_align_sharpness),
            arrive,
        )

        r_core = 0.025
        lo = np.asarray(self.object_aabb_lo, dtype=np.float64)
        hi = np.asarray(self.object_aabb_hi, dtype=np.float64)
        half = cs.DM(0.5 * (hi - lo) + 0.008)
        center_local = cs.DM(0.5 * (hi + lo))
        R = self._quat_wxyz_to_rot(x[3:7])
        tip_local = R.T @ (tip - obj) - center_local
        rho = cs.sqrt(cs.sumsqr(tip_local / half) + 1e-9)
        U_clear = (1.0 - approach_gate) * (
            self._smooth_relu(1.0 - rho) ** 2 + self._smooth_relu(r_core - r) ** 2
        )

        chord = goal - tip
        chord_len = cs.sqrt(cs.sumsqr(chord) + 1e-9)
        d_line = cs.sqrt(cs.sumsqr(cs.cross(chord, obj - tip)) + 1e-12) / chord_len
        blocked = (1.0 - arrive) * self._smooth_gate(0.04 - d_line, 0.0, 12.0)
        z_clear = self._object_top_z(obj, x[3:7]) + 0.015
        U_att = cs.sumsqr(tip - goal)
        U_lift = blocked * self._smooth_relu(z_clear - tip[2]) ** 2
        return (
            self.detour_attract_coef * U_att
            + self.detour_repel_coef * U_clear
            + self.detour_lift_coef * U_lift
        )

    @staticmethod
    def log_barrier_function(x, virtual_point, epsilon=1e-3):
        diff = x[7:10] - virtual_point
        squared_norm = cs.dot(diff, diff) + epsilon
        return cs.log(squared_norm)
