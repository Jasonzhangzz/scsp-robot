import casadi as cs
import re
import numpy as np
import trimesh

from utils import rotations
from planning.attract_function import compute_scalar_potential_and_gradient
from planning.mlqp_point import LambdaContactControlOptimizer
from examples.mpc.fingertips.test.params import (
    _mujoco_collision_mesh,
    _mujoco_visual_mesh,
)

# Historical elephant goal-geom offset, same as fingertips --rollout.
_GOAL_GEOM_Q = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0], dtype=np.float64)


def add_via_init_pose_args(parser):
    """Same tilted-start flags as examples/mpc/fingertips/test/test_0902.py."""
    parser.add_argument(
        "--random_init_tilt",
        dest="random_init_tilt",
        action="store_true",
        help="Randomize the initial object tilt so flip starts are not upright.",
    )
    parser.add_argument("--no_random_init_tilt", dest="random_init_tilt", action="store_false")
    parser.set_defaults(random_init_tilt=True)
    parser.add_argument("--init_tilt_deg", type=float, default=75.0)
    parser.add_argument("--init_tilt_min_deg", type=float, default=35.0)
    return parser


def _tilted_init_quaternion(args, yaw_angle):
    yaw = float(np.asarray(yaw_angle, dtype=np.float64).reshape(-1)[0])
    pitch_angle = 0.0
    roll_angle = 0.0
    if getattr(args, "random_init_tilt", False):
        max_tilt = np.deg2rad(max(0.0, float(getattr(args, "init_tilt_deg", 65.0))))
        min_tilt = np.deg2rad(max(0.0, float(getattr(args, "init_tilt_min_deg", 0.0))))
        min_tilt = min(min_tilt, max_tilt)
        tilt = np.sqrt(min_tilt ** 2 + np.random.rand() * (max_tilt ** 2 - min_tilt ** 2))
        axis_angle = 2.0 * np.pi * np.random.rand()
        pitch_angle = float(tilt * np.cos(axis_angle))
        roll_angle = float(tilt * np.sin(axis_angle))
    return rotations.rpy_to_quaternion(np.hstack([yaw, pitch_angle, roll_angle]))


def _quat_wxyz_to_R(quat_wxyz):
    qw, qx, qy, qz = np.asarray(quat_wxyz, dtype=float).reshape(4)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _lift_init_height_for_tilt(init_height, quat_wxyz, local_bounds, table_height):
    rot = _quat_wxyz_to_R(quat_wxyz)
    corners = np.array(np.meshgrid(*zip(local_bounds[0], local_bounds[1]))).T.reshape(-1, 3)
    min_world_z = float(np.min(corners @ rot[2, :]))
    return max(float(init_height), float(table_height) + 0.002 - min_world_z)


def _xml_obj_geom_quat_wxyz(model_path):
    """Standing offset baked into the MuJoCo obj geom (wxyz).

    piggy_bank / mug / teapot / foam_brick / rubber_duck use +90 deg about Y
    so the raw STL rests on its feet.  Elephant / bunny are already identity.
    """
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    try:
        text = open(model_path, "r", encoding="ascii").read()
    except OSError:
        return identity
    tag = None
    for match in re.finditer(r"<geom\b[^>]*>", text):
        chunk = match.group(0)
        if re.search(r'\bname="obj"', chunk):
            tag = chunk
            break
    if tag is None:
        return identity
    quat_m = re.search(r'\bquat="([^"]+)"', tag)
    if quat_m is None:
        return identity
    quat = np.fromstring(quat_m.group(1), sep=" ", dtype=np.float64)
    if quat.size != 4:
        return identity
    nrm = float(np.linalg.norm(quat))
    if nrm < 1e-9:
        return identity
    return quat / nrm


def _source_standing_quat_wxyz(model_path):
    """Stand the *source STL* using the XML obj geom quat.

    piggy_bank / mug / teapot / foam_brick / rubber_duck bake +90 deg about
    Y so authored +X becomes world +Z.  Ry(-90) is the inverse and stands
    those meshes on their heads.  Elephant / bunny already use identity.
    """
    return _xml_obj_geom_quat_wxyz(model_path)


def _ground_rotation_target_q(target_roll_sample):
    """Same 90 deg pitch flip as fingertips test_0902 ground-rotation."""
    body_target_q = rotations.rpy_to_quaternion(
        np.hstack([0.0, -0.5 * np.pi, np.pi * float(target_roll_sample) - 0.5 * np.pi])
    )
    return rotations.quaternion_multiply(body_target_q, _GOAL_GEOM_Q)


def _actor_quat_from_body(body_q, standing_q, extracted):
    """Map a MuJoCo body quaternion into the Isaac actor frame."""
    body_q = np.asarray(body_q, dtype=np.float64).reshape(4)
    if extracted:
        return body_q
    return rotations.quaternion_multiply(body_q, np.asarray(standing_q, dtype=np.float64).reshape(4))


def _box_inertia_diag(mass, aabb_lo, aabb_hi):
    """Principal box inertia for a uniform solid of the given AABB."""
    extents = np.asarray(aabb_hi, dtype=np.float64).reshape(3) - np.asarray(aabb_lo, dtype=np.float64).reshape(3)
    extents = np.maximum(np.abs(extents), 1e-4)
    mass = float(mass)
    return np.array(
        [
            mass / 12.0 * (extents[1] ** 2 + extents[2] ** 2),
            mass / 12.0 * (extents[0] ** 2 + extents[2] ** 2),
            mass / 12.0 * (extents[0] ** 2 + extents[1] ** 2),
        ],
        dtype=np.float64,
    )


def build_lambda_optimizer(param, args):
    """Same acados contact NLP as examples/mpc/fingertips/test/params.py."""
    extracted = bool(getattr(param, "_collision_mesh_extracted", False))
    return LambdaContactControlOptimizer(
        mesh_path=param.mesh_path_,
        obj_mass=float(getattr(param, "lambda_obj_mass_", 0.01)),
        arm_friction=param.mu_object_,
        contact_stiffness=param.model_params,
        time_step=getattr(param, "lambda_h_", 0.05),
        sample_num=args.sample_num,
        top_k=getattr(args, "top_k", 2),
        pos_coef=args.pos_coef,
        ori_coef=args.ori_coef,
        friction_reg_coef=getattr(args, "friction_reg_coef", 0.0),
        force_reg_coef=getattr(args, "force_reg_coef", 0.01),
        max_contact_force=float(getattr(args, "max_contact_force", 10.0)),
        contact_switch_radius=getattr(args, "contact_switch_radius", 0.03),
        contact_switch_margin_ratio=getattr(args, "contact_switch_margin_ratio", 0.2),
        contact_switch_margin_abs=getattr(args, "contact_switch_margin_abs", 1e-3),
        fingertip_clearance=getattr(args, "fingertip_clearance", 0.011),
        normal_stability_cos=getattr(args, "normal_stability_cos", 0.95),
        solver=getattr(args, "solver", "acados"),
        torch_max_iter=getattr(args, "torch_max_iter", 100),
        obj_inertia=None,
        wrench_is_force=False,
        collision_hull=(bool(getattr(param, "collision_hull", True)) and not extracted),
    )


class ExplicitMPCParams:
    def __init__(self, args, rand_seed=1, target_type="ground-rotation", mpc_model="explicit"):
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
        # Sample the same compiled convex hull as fingertips --rollout so
        # ranking, PhysX, and the pose metric share the MuJoCo body frame.
        self._collision_mesh_extracted = False
        if self.collision_hull:
            extracted = _mujoco_collision_mesh(self.source_mesh_path_, self.model_path_)
            if extracted != self.source_mesh_path_:
                self.mesh_path_ = extracted
                self.visual_mesh_path_ = _mujoco_visual_mesh(
                    self.source_mesh_path_, self.model_path_
                )
                self._collision_mesh_extracted = True
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
        # Match the fingertip --rollout planner interval.  Isaac physics keeps
        # its own sim_dt_ (DyWA 12.5 ms); this h_ is only the acados model step.
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

        np.random.seed(100 + rand_seed)

        self.table_height = 0.35
        local_bounds = np.stack([self.object_aabb_lo, self.object_aabb_hi], axis=0)
        extracted = bool(self._collision_mesh_extracted)
        # Extracted hull is already in the MuJoCo body frame (geom baked).
        self.standing_q_ = (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            if extracted
            else _source_standing_quat_wxyz(self.model_path_)
        )
        init_xy_rand = 0.1 * np.random.rand(2)
        init_xy_rand[0] += 0.3

        yaw_angle = float(2.0 * np.pi * np.random.rand() - np.pi)
        # Draw the target roll before the optional tilt so enabling
        # --random_init_tilt only changes the initial state.
        target_roll_sample = float(np.random.rand())
        target_type = str(getattr(args, "target_type", target_type) or target_type)
        if target_type == "ground-rotation":
            # Same flip as test_0902, shifted into the Franka workspace.
            self.target_q_ = _actor_quat_from_body(
                _ground_rotation_target_q(target_roll_sample),
                self.standing_q_,
                extracted,
            )
            self.target_p_ = np.array(
                [0.40, 0.10, float(self.table_height) + 0.03], dtype=np.float64
            )
        elif target_type == "rotation":
            target_xy_rand = 0.1 * np.random.rand(2)
            target_xy_rand[0] += 0.35
            target_yaw = np.pi * np.random.rand(1) - np.pi / 2
            target_yaw_q = rotations.rpy_to_quaternion(np.hstack([target_yaw, 0, 0]))
            self.target_q_ = _actor_quat_from_body(
                target_yaw_q, self.standing_q_, extracted
            )
            target_height = _lift_init_height_for_tilt(
                0.0, self.target_q_, local_bounds, self.table_height
            )
            self.target_p_ = np.hstack([target_xy_rand, target_height])
        else:
            raise ValueError(f"Target type {target_type} not supported")

        init_tilt_q = _tilted_init_quaternion(args, yaw_angle)
        init_obj_quat_rand = _actor_quat_from_body(
            init_tilt_q, self.standing_q_, extracted
        )
        init_height = _lift_init_height_for_tilt(
            0.0, init_obj_quat_rand, local_bounds, self.table_height
        )

        self.init_obj_qpos_ = np.hstack((init_xy_rand, init_height, init_obj_quat_rand))
        self.init_robot_qpos_ = np.array([0.0, -0.785, 0.0, -2.356, 0, 1.571, 0.785])
        # Same start as fingertips --rollout: beside the object, just above
        # the table.  The Franka ready pose parks the sphere above the
        # elephant, so the keep-out via hovers and never drops onto press.
        toward_base = -np.asarray(self.init_obj_qpos_[:2], dtype=np.float64)
        toward_norm = float(np.linalg.norm(toward_base))
        if toward_norm < 1e-6:
            toward_base = np.array([-1.0, 0.0], dtype=np.float64)
        else:
            toward_base = toward_base / toward_norm
        self.init_fingertip_pos_ = np.array(
            [
                float(self.init_obj_qpos_[0] + 0.10 * toward_base[0]),
                float(self.init_obj_qpos_[1] + 0.10 * toward_base[1]),
                float(self.table_height + 0.02),
            ],
            dtype=np.float64,
        )
        if getattr(args, "random_init_tilt", False):
            self.init_fingertip_pos_[:2] += 0.06 * (2.0 * np.random.rand(2) - 1.0)
            self.init_fingertip_pos_[2] += 0.03 * float(np.random.rand())

        self.mu_object_ = 0.5
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
        # Planner, ranking, and PhysX actor mass all match fingertips --rollout.
        self.lambda_obj_mass_ = 0.01
        self.sim_obj_mass_ = 0.01
        self.sim_obj_inertia_diag_ = _box_inertia_diag(
            self.sim_obj_mass_, self.object_aabb_lo, self.object_aabb_hi
        )
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
        # Same fingertip box as fingertips --rollout, shifted by the table.
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
            # Lambda pose: pos_coef||Δp||² + ori_coef(1 − q·q*)².
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
