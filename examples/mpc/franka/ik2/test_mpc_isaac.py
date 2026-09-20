import argparse
import os
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation

if __name__ == "__main__" and "--sync-planner" not in sys.argv:
    os.environ.setdefault("SCSP_PLANNER_ONLY", "1")

if os.environ.get("SCSP_PLANNER_ONLY") == "1":
    gymapi = None
    gymtorch = None
    torch = None
else:
    from isaacgym import gymapi, gymtorch
    import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(current_dir)
while os.path.basename(parent_dir) != "scsp-robot":
    _next_dir = os.path.dirname(parent_dir)
    if _next_dir == parent_dir:
        raise RuntimeError("scsp-robot repo root not found from %s" % current_dir)
    parent_dir = _next_dir
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from planning.acados_env import ensure_acados_env
ensure_acados_env()

from examples.mpc.franka.ik2.params import (
    ExplicitMPCParams,
    build_lambda_optimizer,
    _box_inertia_diag,
)
from examples.mpc.franka.ik2.test_mppi_isaac import (
    IsaacFrankaSimulator,
    _apply_plan_result,
    _dwell_payload,
    _extract_quat_xyzw,
    _extract_vec3,
    _parse_bool_arg,
    _plan_from_obs,
    _plan_payload,
    _pred_reduction_from_policy,
)
from examples.mpc.fingertips.test.test_0902 import (
    add_rollout_via_args,
    _keepout_radius,
    _on_opposite_sides,
    _press_path_blocked,
    _segment_hits_core,
    _travel_orbit_radius,
)
from planning.screenshot import create_isaacgym_svg_screenshot_recorder
from examples.mpc.franka.ik2.contact_frames import (
    FRANKA_QD_NULLSPACE,
    clip_mpc_action as _clip_mpc_action,
    clip_via_target as _clip_via_target,
    diagnose_contact_pose_source as _diagnose_contact_pose_source,
    franka_nullspace_posture_torque as _franka_nullspace_posture_torque,
    isaac_task_force as _isaac_task_force,
    mpc_ball_trajectory as _mpc_ball_trajectory,
    near_press_force_mode as _near_press_force_mode,
    NEAR_PRESS_SWITCH as CONTACT_NEAR_PRESS,
    osc_action_task_force as _osc_action_task_force,
    slew_mpc_action as _slew_mpc_action,
    planar_table_jacobians as _planar_table_jacobians,
    press_normal_outward as _press_normal_outward,
    project_along_action as _project_along_action,
    remaining_along_action as _remaining_along_action,
    tangent_basis_from_normal as _tangent_basis_from_normal,
    _contact_jacobian_np as _contact_jacobian,
)
from examples.mpc.franka.ik2.isaac_bus import joint_hold_target
from examples.mpc.franka.ik2.physx_contact import (
    DEFAULT_CONTACT_OFFSET,
    contact_impulse,
    contact_overlap,
    end_effector_position,
    fingertip_body_index,
    fingertip_radius,
    physx_signed_gap,
)
from utils import metrics

# DyWA used 12.5 ms; that only allowed ~2 OSC updates per 20 ms policy
# step, so the arm finished a small fraction of each 5 mm increment.
# Match the MuJoCo --rollout inner step (default 2 ms, frame_skip 10).
DYWA_SIM_DT = 0.002
DYWA_SIM_SUBSTEPS = 1
POLICY_INTERVAL = 0.02
DYWA_PHYSX_SOLVER_TYPE = 1
DYWA_PHYSX_POSITION_ITERATIONS = 8
DYWA_PHYSX_VELOCITY_ITERATIONS = 1
DYWA_PHYSX_CONTACT_OFFSET = 0.001
DYWA_PHYSX_REST_OFFSET = 0.0
DYWA_PHYSX_FRICTION_OFFSET_THRESHOLD = 0.001
DYWA_PHYSX_FRICTION_CORRELATION_DISTANCE = 0.0005
DYWA_PHYSX_MAX_DEPENETRATION_VELOCITY = 0.25
DYWA_TABLE_FRICTION_RANGE = (0.3, 0.8)
DYWA_OBJECT_FRICTION_RANGE = (0.2, 1.0)
DYWA_OBJECT_MASS_RANGE = (0.1, 0.5)

DEFAULT_CARTESIAN_STIFFNESS = np.array([2000.0, 2000.0, 2000.0, 50.0, 50.0, 50.0], dtype=np.float32)
# Official Isaac OSC uses 0.  PhysX joint viscosity on top of task-space
# damping made the tip crawl.  Contact force is set in task space, not here.
DEFAULT_EFFORT_JOINT_DAMPING = 0.0
# Fingertips --rollout PD: ctrl = -100 dpos - 2 dvel on an ~0.01 kg sphere.
# Isaac / Franka cannot use that law: the task mass is ~2.5 kg and PhysX
# contact is softer than MuJoCo, so the same 5 mm / 20 ms carrot slams.
ROLLOUT_FINGERTIP_MASS = 0.01
ROLLOUT_TASK_KP = 100.0
ROLLOUT_TASK_KD = 2.0
ISAAC_TASK_KP = 600.0
ISAAC_TASK_KD = 40.0
# Enough to walk a 2.5 mm via without saturating, far below the 40 N
# punch that finished a 5 mm ball increment in 20 ms.
FREE_SPACE_FORCE_LIMIT = 12.0
# PhysX needs a lighter press than the MuJoCo 2 N ball cap.
CONTACT_FORCE_LIMIT = 0.8
# Shorter carrot than --rollout.  MPC may still emit 5 mm; OSC slews it.
AIR_VIA_STEP = 0.0025
ISAAC_VIA_MAX_STEP = 0.0025
ISAAC_VIA_MAX_LEAD = 0.003
ISAAC_VIA_SMOOTH_RATE = 0.03
ISAAC_ACTION_SLEW = 0.0015
# Same keep-out as path_blocked / verify.  An extra 8 cm circle
# sent the Franka tip 15 cm from the COM before it could drop.
ISAAC_ORBIT_EXTRA = 0.0
NEAR_PRESS_SWITCH = CONTACT_NEAR_PRESS
ROLLOUT_TABLE_FRICTION = 0.5
ROLLOUT_OBJECT_FRICTION = 0.9


def policy_control_substeps(control_substeps, sim_dt, policy_interval=POLICY_INTERVAL):
    """Frames per planner action.  0 means one --rollout interval (20 ms)."""
    n = int(control_substeps)
    if n <= 0:
        return max(1, int(round(float(policy_interval) / max(float(sim_dt), 1e-6))))
    return n


ROLLOUT_FINGERTIP_FRICTION = 1.5
SVG_SCREENSHOT_CAMERA_POSITION = np.array([0.7, 0.00, 0.63], dtype=np.float32)
SVG_SCREENSHOT_CAMERA_TARGET = np.array([0.1, 0.00, 0.32], dtype=np.float32)




def _quat_xyzw_to_matrix(quat_xyzw):
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    nrm = float(np.linalg.norm(quat))
    if not np.isfinite(nrm) or nrm < 1e-8:
        return np.eye(3, dtype=np.float32)
    return Rotation.from_quat(quat / nrm).as_matrix().astype(np.float32)


def _mjcf_quat_wxyz_to_urdf_rpy(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    return Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)


class VelocityKalmanFilter6D:
    def __init__(self, process_variance, measurement_variance, initial_covariance=1.0):
        process_variance = np.asarray(process_variance, dtype=np.float32).reshape(6)
        measurement_variance = np.asarray(measurement_variance, dtype=np.float32).reshape(6)
        self.Q = np.diag(np.maximum(process_variance, 1e-9))
        self.R = np.diag(np.maximum(measurement_variance, 1e-9))
        self.initial_covariance = float(initial_covariance)
        self._I = np.eye(6, dtype=np.float32)
        self.reset()

    def reset(self):
        self.x = np.zeros(6, dtype=np.float32)
        self.P = self.initial_covariance * np.eye(6, dtype=np.float32)
        self.initialized = False

    def update(self, measurement):
        z = np.asarray(measurement, dtype=np.float32).reshape(6)
        if not self.initialized:
            self.x = z.copy()
            self.initialized = True
            return self.x.copy()

        # Constant-velocity random walk model:
        # x_k|k-1 = x_k-1|k-1, P_k|k-1 = P_k-1|k-1 + Q
        self.P = self.P + self.Q

        innovation = z - self.x
        innovation_covariance = self.P + self.R
        kalman_gain = self.P @ np.linalg.inv(innovation_covariance)
        self.x = self.x + kalman_gain @ innovation
        self.P = (self._I - kalman_gain) @ self.P
        return self.x.copy()


def _apply_velocity_deadband(v_obj_local, linear_deadband, angular_deadband):
    v_obj_local = np.asarray(v_obj_local, dtype=np.float32).reshape(6).copy()
    v_obj_local[:3] = np.where(np.abs(v_obj_local[:3]) < linear_deadband, 0.0, v_obj_local[:3])
    v_obj_local[3:] = np.where(np.abs(v_obj_local[3:]) < angular_deadband, 0.0, v_obj_local[3:])
    return v_obj_local


def _set_actor_friction(gym, env, actor_handle, friction):
    shape_props = gym.get_actor_rigid_shape_properties(env, actor_handle)
    for prop in shape_props:
        prop.friction = float(friction)
        prop.torsion_friction = float(friction)
        prop.rolling_friction = float(friction)
    gym.set_actor_rigid_shape_properties(env, actor_handle, shape_props)


def _inertia_diag_from_prop(prop):
    inertia = getattr(prop, "inertia", None)
    if inertia is None:
        return None
    row0 = getattr(inertia, "x", None)
    if row0 is not None and hasattr(row0, "x"):
        return np.array(
            [float(inertia.x.x), float(inertia.y.y), float(inertia.z.z)],
            dtype=np.float64,
        )
    if row0 is not None:
        return np.array([float(inertia.x), float(inertia.y), float(inertia.z)], dtype=np.float64)
    return None


def _set_actor_inertia(prop, inertia_diag):
    """Isaac Gym stores rigid-body inertia as Mat33, not a Vec3."""
    ixx, iyy, izz = [float(v) for v in np.asarray(inertia_diag, dtype=np.float64).reshape(3)]
    inertia = getattr(prop, "inertia", None)
    if inertia is None:
        return
    row0 = getattr(inertia, "x", None)
    if row0 is not None and hasattr(row0, "x"):
        inertia.x = gymapi.Vec3(ixx, 0.0, 0.0)
        inertia.y = gymapi.Vec3(0.0, iyy, 0.0)
        inertia.z = gymapi.Vec3(0.0, 0.0, izz)
        return
    if row0 is not None:
        inertia.x = ixx
        inertia.y = iyy
        inertia.z = izz
        return
    prop.inertia = gymapi.Vec3(ixx, iyy, izz)


def _set_actor_mass(gym, env, actor_handle, mass, inertia_diag=None):
    body_props = gym.get_actor_rigid_body_properties(env, actor_handle)
    mass = float(mass)
    for prop in body_props:
        old_mass = float(prop.mass)
        prop.mass = mass
        if inertia_diag is not None:
            _set_actor_inertia(prop, inertia_diag)
        elif old_mass > 1e-9:
            old_diag = _inertia_diag_from_prop(prop)
            if old_diag is not None:
                _set_actor_inertia(prop, old_diag * (mass / old_mass))
    # Keep the explicit brick inertia.  recomputeInertia=True would
    # restore the URDF 1e-4 values after the 0.05 -> 0.01 mass change.
    gym.set_actor_rigid_body_properties(env, actor_handle, body_props, False)


def _apply_dywa_physics_to_param(param):
    param.sim_dt_ = DYWA_SIM_DT
    param.sim_substeps_ = DYWA_SIM_SUBSTEPS
    param.physx_solver_type_ = DYWA_PHYSX_SOLVER_TYPE
    param.physx_position_iterations_ = DYWA_PHYSX_POSITION_ITERATIONS
    param.physx_velocity_iterations_ = DYWA_PHYSX_VELOCITY_ITERATIONS
    param.physx_contact_offset_ = DYWA_PHYSX_CONTACT_OFFSET
    param.physx_rest_offset_ = DYWA_PHYSX_REST_OFFSET
    param.physx_friction_offset_threshold_ = DYWA_PHYSX_FRICTION_OFFSET_THRESHOLD
    param.physx_friction_correlation_distance_ = DYWA_PHYSX_FRICTION_CORRELATION_DISTANCE
    param.physx_max_depenetration_velocity_ = DYWA_PHYSX_MAX_DEPENETRATION_VELOCITY
    # Same Coulomb values as env_fingertips_*.xml.  DyWA randomization
    # (table 0.3--0.8, object 0.2--1.0) made some trials unflippable.
    param.table_friction_ = ROLLOUT_TABLE_FRICTION
    param.object_friction_ = ROLLOUT_OBJECT_FRICTION
    rollout_mass = float(getattr(param, "lambda_obj_mass_", 0.01))
    param.sim_obj_mass_ = rollout_mass
    param.obj_mass_ = rollout_mass
    param.sim_obj_inertia_diag_ = _box_inertia_diag(
        rollout_mass,
        getattr(param, "object_aabb_lo", np.array([-0.03, -0.03, -0.03])),
        getattr(param, "object_aabb_hi", np.array([0.03, 0.03, 0.03])),
    )
    return param


_FRANKA_DH = np.array(
    [
        [0.0, 0.0, 0.333],
        [0.0, -np.pi / 2, 0.0],
        [0.0, np.pi / 2, 0.316],
        [0.0825, np.pi / 2, 0.0],
        [-0.0825, -np.pi / 2, 0.384],
        [0.0, np.pi / 2, 0.0],
        [0.088, np.pi / 2, 0.0],
    ],
    dtype=np.float64,
)
_ATTACH_R = Rotation.from_quat([0.0, 0.0, 0.9238795, 0.3826834]).as_matrix()
_ATTACH_POS = np.array([0.0, 0.0, 0.107], dtype=np.float64)
_TIP_POS = np.array([0.0, 0.0, 0.06], dtype=np.float64)


def _mdh_np(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _franka_fk_T_np(q):
    T = np.eye(4, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).reshape(7)
    for i in range(7):
        a, alpha, d = _FRANKA_DH[i]
        T = T @ _mdh_np(a, alpha, d, q[i])
    T_attach = np.eye(4, dtype=np.float64)
    T_attach[:3, :3] = _ATTACH_R
    T_attach[:3, 3] = _ATTACH_POS
    T_tip = np.eye(4, dtype=np.float64)
    T_tip[:3, 3] = _TIP_POS
    return T @ T_attach @ T_tip


def _franka_jacobian_pos_np(q, eps=1e-4):
    q = np.asarray(q, dtype=np.float64).reshape(7)
    jac = np.zeros((3, 7), dtype=np.float64)
    for i in range(7):
        dq = np.zeros(7, dtype=np.float64)
        dq[i] = eps
        jac[:, i] = (_franka_fk_T_np(q + dq)[:3, 3] - _franka_fk_T_np(q - dq)[:3, 3]) / (2.0 * eps)
    return jac


class IsaacFrankaOSCSimulator(IsaacFrankaSimulator):
    def __init__(self, param, headless=False, sim_device="cuda:0", graphics_device_id=0):
        self.param_ = param
        self.break_out_signal_ = False
        self.dyn_paused_ = False
        self.viewer_ = None
        self.svg_screenshot_recorder_ = None
        self.show_ghost_object_ = bool(getattr(self.param_, "show_ghost_object_", False))

        self.gym = gymapi.acquire_gym()

        compute_id = int(sim_device.split(":")[-1]) if ":" in sim_device else 0
        self.sim_dt_ = float(getattr(self.param_, "sim_dt_", DYWA_SIM_DT))
        sim_substeps = int(getattr(self.param_, "sim_substeps_", DYWA_SIM_SUBSTEPS))

        sim_params = gymapi.SimParams()
        sim_params.dt = self.sim_dt_
        sim_params.substeps = sim_substeps
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        # This script uses CPU-style state APIs (get/set_actor_rigid_body_states),
        # so disable GPU pipeline to avoid invalid resource handle errors.
        sim_params.use_gpu_pipeline = False
        sim_params.physx.solver_type = int(getattr(self.param_, "physx_solver_type_", DYWA_PHYSX_SOLVER_TYPE))
        sim_params.physx.num_position_iterations = int(
            getattr(self.param_, "physx_position_iterations_", DYWA_PHYSX_POSITION_ITERATIONS)
        )
        sim_params.physx.num_velocity_iterations = int(
            getattr(self.param_, "physx_velocity_iterations_", DYWA_PHYSX_VELOCITY_ITERATIONS)
        )
        sim_params.physx.contact_offset = float(
            getattr(self.param_, "physx_contact_offset_", DYWA_PHYSX_CONTACT_OFFSET)
        )
        sim_params.physx.rest_offset = float(getattr(self.param_, "physx_rest_offset_", DYWA_PHYSX_REST_OFFSET))
        sim_params.physx.friction_offset_threshold = float(
            getattr(self.param_, "physx_friction_offset_threshold_", DYWA_PHYSX_FRICTION_OFFSET_THRESHOLD)
        )
        sim_params.physx.friction_correlation_distance = float(
            getattr(self.param_, "physx_friction_correlation_distance_", DYWA_PHYSX_FRICTION_CORRELATION_DISTANCE)
        )
        sim_params.physx.bounce_threshold_velocity = 2.0 * 9.81 * self.sim_dt_ / max(sim_substeps, 1)
        sim_params.physx.max_depenetration_velocity = float(
            getattr(self.param_, "physx_max_depenetration_velocity_", DYWA_PHYSX_MAX_DEPENETRATION_VELOCITY)
        )
        # One-env viewer + Warp MPPI share a GPU.  CPU PhysX avoids that hitch.
        sim_params.physx.use_gpu = bool(getattr(self.param_, "physx_use_gpu_", False))

        self.sim = self.gym.create_sim(compute_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym sim")

        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane)

        env_lower = gymapi.Vec3(-2.0, -2.0, 0.0)
        env_upper = gymapi.Vec3(2.0, 2.0, 2.0)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1)

        self._create_scene_actors()
        self._apply_scene_physics_settings()
        self._configure_franka()
        self.gym.prepare_sim(self.sim)
        self._build_body_index_cache()
        self.reset_mj_env()

        if not headless:
            self.viewer_ = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer_ is not None:
                cam_pos = gymapi.Vec3(
                    float(SVG_SCREENSHOT_CAMERA_POSITION[0]),
                    float(SVG_SCREENSHOT_CAMERA_POSITION[1]),
                    float(SVG_SCREENSHOT_CAMERA_POSITION[2]),
                )
                cam_target = gymapi.Vec3(
                    float(SVG_SCREENSHOT_CAMERA_TARGET[0]),
                    float(SVG_SCREENSHOT_CAMERA_TARGET[1]),
                    float(SVG_SCREENSHOT_CAMERA_TARGET[2]),
                )
                self.gym.viewer_camera_look_at(self.viewer_, self.env, cam_pos, cam_target)

        screenshot_dir = getattr(self.param_, "svg_screenshot_dir_", None)
        if screenshot_dir:
            self.svg_screenshot_recorder_ = create_isaacgym_svg_screenshot_recorder(
                gym=self.gym,
                sim=self.sim,
                env=self.env,
                output_dir=screenshot_dir,
                camera_position=SVG_SCREENSHOT_CAMERA_POSITION,
                camera_target=SVG_SCREENSHOT_CAMERA_TARGET,
                interval_seconds=float(getattr(self.param_, "svg_screenshot_interval_", 1.0)),
                width=int(getattr(self.param_, "svg_screenshot_width_", 1280)),
                height=int(getattr(self.param_, "svg_screenshot_height_", 960)),
                filename_prefix=str(getattr(self.param_, "svg_screenshot_prefix_", "frame")),
                capture_on_start=True,
            )
            print(
                "[IsaacFrankaOSCSimulator] SVG screenshot capture enabled: "
                f"dir={screenshot_dir}, every={float(getattr(self.param_, 'svg_screenshot_interval_', 1.0)):.2f}s, "
                f"size={int(getattr(self.param_, 'svg_screenshot_width_', 1280))}x"
                f"{int(getattr(self.param_, 'svg_screenshot_height_', 960))}"
            )

        self.R_d = np.eye(3)
        cartesian_stiffness = getattr(self.param_, "cartesian_stiffness_", DEFAULT_CARTESIAN_STIFFNESS)
        cartesian_damping = getattr(self.param_, "cartesian_damping_", None)
        self.cartesian_stiffness = self._format_cartesian_gain(cartesian_stiffness)
        if cartesian_damping is None:
            self.cartesian_damping = (2.0 * np.sqrt(self.cartesian_stiffness)).astype(np.float32)
        else:
            self.cartesian_damping = self._format_cartesian_gain(cartesian_damping)
        self._sync_desired_pose_with_current()

    def _create_scene_actors(self):
        super()._create_scene_actors()

        # Override the inherited table visual with a silver-gray finish.
        table_color = gymapi.Vec3(0.45, 0.47, 0.50)
        num_table_bodies = self.gym.get_actor_rigid_body_count(self.env, self.table_actor)
        for body_idx in range(num_table_bodies):
            self.gym.set_rigid_body_color(
                self.env,
                self.table_actor,
                body_idx,
                gymapi.MESH_VISUAL,
                table_color,
            )

        p_arm_marker_pose = gymapi.Transform()
        p_arm_marker_pose.p = gymapi.Vec3(0.3, 0.0, 0.4)
        p_arm_marker_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.p_arm_marker_actor = self.gym.create_actor(
            self.env, self.marker_asset, p_arm_marker_pose, "p_arm_marker", 0, 0
        )
        self.gym.set_rigid_body_color(
            self.env,
            self.p_arm_marker_actor,
            0,
            gymapi.MESH_VISUAL,
            gymapi.Vec3(1.0, 0.0, 0.0),
        )

        best_marker_pose = gymapi.Transform()
        best_marker_pose.p = gymapi.Vec3(0.3, 0.05, 0.4)
        best_marker_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.best_contact_marker_actor = self.gym.create_actor(
            self.env, self.marker_asset, best_marker_pose, "best_contact_marker", 0, 0
        )
        self.gym.set_rigid_body_color(
            self.env,
            self.best_contact_marker_actor,
            0,
            gymapi.MESH_VISUAL,
            gymapi.Vec3(1.0, 1.0, 0.0),
        )

    @staticmethod
    def _get_franka_asset_info(repo_root):
        asset_root = os.path.join(repo_root, "envs/robots/assets/urdf")
        src_urdf = os.path.join(asset_root, "franka_description", "robots", "franka_panda.urdf")
        dst_rel = os.path.join("franka_description", "robots", "franka_panda_nohand_sphere_tmp.urdf")
        dst_urdf = os.path.join(asset_root, dst_rel)
        attachment_rpy = _mjcf_quat_wxyz_to_urdf_rpy([0.3826834, 0.0, 0.0, 0.9238795])

        with open(src_urdf, "r", encoding="ascii") as f:
            urdf_text = f.read()

        # Strip the default hand/finger subtree so Isaac loads a true no-hand arm.
        strip_patterns = [
            r'\s*<joint name="panda_hand_joint"[\s\S]*?</joint>',
            r'\s*<link name="panda_hand">[\s\S]*?</link>',
            r'\s*<link name="panda_leftfinger">[\s\S]*?</link>',
            r'\s*<link name="panda_rightfinger">[\s\S]*?</link>',
            r'\s*<joint name="panda_finger_joint1"[\s\S]*?</joint>',
            r'\s*<joint name="panda_finger_joint2"[\s\S]*?</joint>',
        ]
        for pattern in strip_patterns:
            urdf_text = re.sub(pattern, "\n", urdf_text, flags=re.MULTILINE)

        attachment_block = f"""
  <link name="attachment">
    <visual>
      <origin xyz="0 0 0.03" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.005" length="0.06"/>
      </geometry>
      <material name="attachment_dark">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.03" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.005" length="0.06"/>
      </geometry>
    </collision>
  </link>
  <joint name="attachment_joint" type="fixed">
    <parent link="panda_link7"/>
    <child link="attachment"/>
    <origin xyz="0 0 0.107" rpy="{attachment_rpy[0]} {attachment_rpy[1]} {attachment_rpy[2]}"/>
  </joint>
  <link name="fingertip">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.01"/>
      </geometry>
      <material name="fingertip_red">
        <color rgba="0.8 0.2 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.01"/>
      </geometry>
    </collision>
  </link>
  <joint name="fingertip_joint" type="fixed">
    <parent link="attachment"/>
    <child link="fingertip"/>
    <origin xyz="0 0 0.06" rpy="0 0 0"/>
  </joint>
"""
        urdf_text = urdf_text.replace("</robot>", attachment_block + "\n</robot>")
        with open(dst_urdf, "w", encoding="ascii") as f:
            f.write(urdf_text)

        return asset_root, dst_rel

    def _apply_scene_physics_settings(self):
        self.table_friction_ = float(getattr(self.param_, "table_friction_", 0.5))
        self.object_friction_ = float(getattr(self.param_, "object_friction_", 0.5))
        self.obj_mass_ = float(getattr(self.param_, "sim_obj_mass_", getattr(self.param_, "obj_mass_", 0.01)))
        inertia_diag = getattr(self.param_, "sim_obj_inertia_diag_", None)
        _set_actor_friction(self.gym, self.env, self.table_actor, self.table_friction_)
        _set_actor_friction(self.gym, self.env, self.obj_actor, self.object_friction_)
        _set_actor_mass(self.gym, self.env, self.obj_actor, self.obj_mass_, inertia_diag)
        self._apply_fingertip_only_object_collision()

    def _apply_fingertip_only_object_collision(self):
        """Only the sphere touches the object, same as the MuJoCo fingertip.

        Arm-link / elephant collisions park the tip on a hover via above
        the mesh; 0902 never has those links.
        Isaac filter convention: shapes collide iff (filterA & filterB) == 0.
        """
        body_names = list(self.gym.get_actor_rigid_body_names(self.env, self.franka_actor))
        shape_props = self.gym.get_actor_rigid_shape_properties(self.env, self.franka_actor)
        assigned = False
        try:
            index_data = self.gym.get_actor_rigid_body_shape_indices(self.env, self.franka_actor)
            for body_i, name in enumerate(body_names):
                start = int(getattr(index_data[body_i], "start", index_data[body_i][0]))
                count = int(getattr(index_data[body_i], "count", index_data[body_i][1]))
                filt = 0 if name == "fingertip" else 1
                for k in range(start, start + count):
                    if 0 <= k < len(shape_props):
                        shape_props[k].filter = filt
                        if name == "fingertip":
                            shape_props[k].friction = ROLLOUT_FINGERTIP_FRICTION
                            shape_props[k].torsion_friction = ROLLOUT_FINGERTIP_FRICTION
                            shape_props[k].rolling_friction = ROLLOUT_FINGERTIP_FRICTION
                        assigned = True
        except Exception:
            assigned = False
        if not assigned:
            fingertip_idx = body_names.index("fingertip") if "fingertip" in body_names else len(shape_props) - 1
            for i, prop in enumerate(shape_props):
                prop.filter = 0 if i == fingertip_idx or i == len(shape_props) - 1 else 1
        self.gym.set_actor_rigid_shape_properties(self.env, self.franka_actor, shape_props)

        obj_props = self.gym.get_actor_rigid_shape_properties(self.env, self.obj_actor)
        for prop in obj_props:
            prop.filter = 1
        self.gym.set_actor_rigid_shape_properties(self.env, self.obj_actor, obj_props)

        table_props = self.gym.get_actor_rigid_shape_properties(self.env, self.table_actor)
        for prop in table_props:
            prop.filter = 0
        self.gym.set_actor_rigid_shape_properties(self.env, self.table_actor, table_props)

    def _configure_franka(self):
        dof_props = self.gym.get_actor_dof_properties(self.env, self.franka_actor)
        effort_joint_damping = float(
            getattr(self.param_, "effort_joint_damping_", 1.0)
        )
        dof_props["driveMode"][:7] = gymapi.DOF_MODE_EFFORT
        dof_props["stiffness"][:7] = 0.0
        dof_props["damping"][:7] = effort_joint_damping

        if dof_props["driveMode"].shape[0] >= 9:
            dof_props["driveMode"][7:] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][7:] = 1.0e6
            dof_props["damping"][7:] = 1.0e2

        self.gym.set_actor_dof_properties(self.env, self.franka_actor, dof_props)

        self.franka_dof_count = self.gym.get_actor_dof_count(self.env, self.franka_actor)
        self._joint_targets = np.zeros(self.franka_dof_count, dtype=np.float32)
        self._joint_targets[:7] = np.array(self.param_.init_robot_qpos_, dtype=np.float32)
        if self.franka_dof_count >= 9:
            self._joint_targets[7] = 0.04
            self._joint_targets[8] = 0.04
        self.torque_limits = np.array(dof_props["effort"][:7], dtype=np.float32)
        self.dof_lower_ = np.array(dof_props["lower"][:7], dtype=np.float32)
        self.dof_upper_ = np.array(dof_props["upper"][:7], dtype=np.float32)

        pos_stiffness = float(getattr(self.param_, "osc_pos_stiffness_", 12000.0))
        ori_stiffness = float(getattr(self.param_, "osc_ori_stiffness_", 0.0))
        self.osc_task_kp = np.array(
            [pos_stiffness, pos_stiffness, pos_stiffness, ori_stiffness, ori_stiffness, ori_stiffness],
            dtype=np.float32,
        )
        # Slightly under-damped so a 5 mm increment can finish inside 20 ms.
        self.osc_task_kd = (1.4 * np.sqrt(self.osc_task_kp)).astype(np.float32)
        self.nullspace_stiffness = float(getattr(self.param_, "nullspace_stiffness_", 10.0))
        self.home_q = np.array(self.param_.init_robot_qpos_, dtype=np.float32)
        self.q_d_nullspace = FRANKA_QD_NULLSPACE.copy()
        self.low_height = float(self.param_.table_height + 0.02)
        self.activate_tool_compensation = False
        self.tool_compensation_force = np.zeros(6, dtype=np.float32)

    def _build_body_index_cache(self):
        super()._build_body_index_cache()
        self.franka_body_count = len(self.franka_body_names)
        self._jacobian_body_offset = self.franka_body_count - int(self._jacobian.shape[1])
        if self._jacobian_body_offset not in (0, 1):
            raise RuntimeError(
                f"Unexpected Franka Jacobian body shape: rigid_bodies={self.franka_body_count}, "
                f"jacobian_bodies={int(self._jacobian.shape[1])}"
            )
        self.ee_body_name = "attachment" if "attachment" in self.franka_body_names else "panda_link7"
        self.ee_body_local_idx = self.franka_body_names.index(self.ee_body_name)
        self.fingertip_body_idx = self.franka_body_names.index("fingertip") if "fingertip" in self.franka_body_names else self.ee_body_local_idx
        self.task_body_local_idx = self.fingertip_body_idx

        rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        mm = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        self._rigid_body_states = gymtorch.wrap_tensor(rb_states)
        self._dof_state = gymtorch.wrap_tensor(dof_states)
        self._mm = gymtorch.wrap_tensor(mm)
        self._effort_control = torch.zeros((self.franka_dof_count,), dtype=torch.float32, device=self._dof_state.device)

    def _jacobian_body_index(self, local_body_idx):
        jac_idx = int(local_body_idx) - int(self._jacobian_body_offset)
        if jac_idx < 0 or jac_idx >= int(self._jacobian.shape[1]):
            raise IndexError(
                f"Rigid body index {local_body_idx} maps to invalid jacobian index {jac_idx}; "
                f"offset={self._jacobian_body_offset}, jacobian_bodies={int(self._jacobian.shape[1])}"
            )
        return jac_idx

    def _refresh_osc_tensors(self):
        # Pose / joints stay on the CPU actor APIs.  Refreshing the rigid-body
        # state tensor here can zero those CPU reads when the GPU pipeline is off.
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

    def reset_mj_env(self):
        super().reset_mj_env()
        self._place_fingertip_for_rollout()
        self._sync_desired_pose_with_current()
        self._hold_q0 = np.asarray(self.get_current_joint_position(), dtype=np.float32)
        self._hold_dq = None
        self._hold_i = 0
        self._hold_n = 1
        self._contact_info_cache = None
        self._clear_mpc_action()

    def _rollout_fingertip_start(self):
        custom = getattr(self.param_, "init_fingertip_pos_", None)
        if custom is not None:
            return np.asarray(custom, dtype=np.float32).reshape(3)
        obj = np.asarray(self.param_.init_obj_qpos_[:3], dtype=np.float64)
        toward = -obj[:2]
        nrm = float(np.linalg.norm(toward))
        if nrm < 1e-6:
            toward = np.array([-1.0, 0.0], dtype=np.float64)
        else:
            toward = toward / nrm
        return np.array(
            [obj[0] + 0.10 * toward[0], obj[1] + 0.10 * toward[1], float(self.param_.table_height) + 0.02],
            dtype=np.float32,
        )

    def _set_arm_qpos(self, q):
        q = np.asarray(q, dtype=np.float32).reshape(7)
        lower = getattr(self, "dof_lower_", None)
        upper = getattr(self, "dof_upper_", None)
        if lower is not None and upper is not None:
            q = np.clip(q, lower, upper)
        dof_states = self.gym.get_actor_dof_states(self.env, self.franka_actor, gymapi.STATE_ALL)
        dof_states["pos"][:7] = q
        dof_states["vel"][:7] = 0.0
        self.gym.set_actor_dof_states(self.env, self.franka_actor, dof_states, gymapi.STATE_ALL)
        self._joint_targets[:7] = q
        if self.franka_dof_count >= 9:
            self._joint_targets[7] = 0.04
            self._joint_targets[8] = 0.04
            self.gym.set_actor_dof_position_targets(self.env, self.franka_actor, self._joint_targets)

    def _place_fingertip_for_rollout(self, steps=80):
        target = self._rollout_fingertip_start()
        yaw = float(np.arctan2(target[1], max(target[0], 0.05)))
        q = np.array(FRANKA_QD_NULLSPACE, dtype=np.float32)
        q[0] = yaw
        self._set_arm_qpos(q)
        for _ in range(int(steps)):
            self._refresh_osc_tensors()
            p_curr, _ = self.get_end_effector_pos()
            err = target - p_curr
            if float(np.linalg.norm(err)) < 0.008:
                break
            jac = np.asarray(self._get_task_jacobian()[:3], dtype=np.float32)
            damp = 1e-3 * np.eye(3, dtype=np.float32)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damp, 0.45 * err)
            q = q + dq
            self._set_arm_qpos(q)
        self.p_d = target.copy()
        self.position_d = target.copy()

    def _sync_desired_pose_with_current(self):
        self._refresh_osc_tensors()
        p, r = self.get_end_effector_pos()
        self.position_d = p.copy()
        self.orientation_d = r.copy()
        self.p_d = p.copy()
        self.R_d = r.copy()
        self.R_d_hold = r.copy()
        # Franka starting(): freeze nullspace at the pose we actually
        # reached.  Pulling toward the high home config from the table
        # leaks through J mismatch and swims the elbow / wrist.
        self.q_d_nullspace = self.get_current_joint_position().copy()

    def _simulate_once(self, sync_realtime=False, draw=True):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if draw:
            self._contact_info_cache = None

        graphics_stepped = False
        if draw and (self.viewer_ is not None or self.svg_screenshot_recorder_ is not None):
            self._sync_pose_axes()
            self.gym.step_graphics(self.sim)
            graphics_stepped = True

        if draw and self.svg_screenshot_recorder_ is not None:
            self.svg_screenshot_recorder_.capture_if_due(
                sim_time=float(self.gym.get_sim_time(self.sim)),
                step_graphics=not graphics_stepped,
            )

        if draw and self.viewer_ is not None:
            self.gym.draw_viewer(self.viewer_, self.sim, True)
            if sync_realtime:
                self.gym.sync_frame_time(self.sim)

    def sync_realtime(self):
        """Wait for wall time to catch the last simulated frame.  CPU-only."""
        if self.viewer_ is not None:
            self.gym.sync_frame_time(self.sim)

    def hold_and_sync(self):
        """Keep the current OSC target and pump the viewer for one frame."""
        self._track_desired_pose(
            preserve_nullspace_target=True, sync_realtime=True, draw=True
        )

    def _get_body_pose(self, local_idx):
        states = self.gym.get_actor_rigid_body_states(self.env, self.franka_actor, gymapi.STATE_POS)
        p = _extract_vec3(states["pose"]["p"][local_idx])
        # Isaac rigid-body states store world-frame orientations as quaternions in xyzw order.
        quat_xyzw = _extract_quat_xyzw(states["pose"]["r"][local_idx])
        quat_ok = np.isfinite(quat_xyzw).all() and float(np.linalg.norm(quat_xyzw)) > 1e-8
        pos_ok = np.isfinite(p).all()
        if not quat_ok or not pos_ok:
            T = _franka_fk_T_np(self.get_current_joint_position())
            return T[:3, 3].astype(np.float32), T[:3, :3].astype(np.float32)
        return p, _quat_xyzw_to_matrix(quat_xyzw)

    def get_end_effector_pos(self):
        p_task, r_task = self._get_body_pose(self.task_body_local_idx)
        return p_task.astype(np.float32), r_task

    def get_policy_state(self):
        full_q = self.get_state()
        ee_pos, _ = self.get_end_effector_pos()
        return np.hstack([full_q[:7], ee_pos]).astype(np.float32)

    def get_object_velocity_world(self):
        obj_state = self.gym.get_actor_rigid_body_states(self.env, self.obj_actor, gymapi.STATE_ALL)
        linear_velocity = _extract_vec3(obj_state["vel"]["linear"][0])
        angular_velocity = _extract_vec3(obj_state["vel"]["angular"][0])
        return linear_velocity.astype(np.float32), angular_velocity.astype(np.float32)

    def get_current_joint_velocity(self):
        dof_states = self.gym.get_actor_dof_states(self.env, self.franka_actor, gymapi.STATE_VEL)
        return np.array(dof_states["vel"][:7], dtype=np.float32)

    def set_desired_pose(self, position, orientation):
        self.position_d = np.asarray(position, dtype=np.float32).copy()
        self.orientation_d = np.asarray(orientation, dtype=np.float32).copy()

    def set_nullspace_stiffness(self, stiffness):
        self.nullspace_stiffness = float(stiffness)

    def set_tool_compensation(self, force, activate=True):
        self.tool_compensation_force = np.asarray(force, dtype=np.float32).copy()
        self.activate_tool_compensation = bool(activate)

    def get_R(self):
        return self.R_d

    def _set_marker_pos(self, actor, goal_pos=None):
        marker_state = self.gym.get_actor_rigid_body_states(self.env, actor, gymapi.STATE_ALL)
        if goal_pos is not None:
            marker_state["pose"]["p"][0] = (float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2]))
        marker_state["vel"]["linear"][0] = (0.0, 0.0, 0.0)
        marker_state["vel"]["angular"][0] = (0.0, 0.0, 0.0)
        self.gym.set_actor_rigid_body_states(self.env, actor, marker_state, gymapi.STATE_ALL)

    def show_point(self, goal_pos=None):
        self._set_marker_pos(self.p_arm_marker_actor, goal_pos)

    def show_target(self, goal_pos=None, goal_quat=None):
        self.show_point(goal_pos)

    def show_best_contact(self, goal_pos=None):
        marker = getattr(self, "best_contact_marker_actor", self.p_arm_marker_actor)
        self._set_marker_pos(marker, goal_pos)

    def _get_task_jacobian(self):
        jac_task_idx = self._jacobian_body_index(self.task_body_local_idx)
        return np.array(self._jacobian[0, jac_task_idx, :, :7], dtype=np.float32)

    def _get_position_jacobian(self, q=None):
        """Analytic fingertip J.  Isaac's tensor J is for a different body
        index on the no-hand URDF and was producing weak task torques."""
        if q is None:
            q = self.get_current_joint_position()
        return _franka_jacobian_pos_np(q).astype(np.float32)

    def _get_arm_mass_matrix(self):
        mm = self._mm
        if mm.ndim == 3:
            mm = mm[0]
        return np.array(mm[:7, :7], dtype=np.float32)

    def _task_mass_matrix(self, jac_p):
        jac_p = np.asarray(jac_p, dtype=np.float64)
        mm = self._get_arm_mass_matrix()
        mm_ok = np.isfinite(mm).all() and float(np.trace(np.asarray(mm, dtype=np.float64))) > 0.5
        if mm_ok:
            mm_inv = self._safe_inverse(mm)
            lambda_inv = jac_p @ np.asarray(mm_inv, dtype=np.float64) @ jac_p.T
            if np.isfinite(lambda_inv).all() and abs(float(np.linalg.det(lambda_inv))) > 1e-10:
                return self._safe_inverse(lambda_inv.astype(np.float32))
        # CPU-pipeline mass tensor is often empty; a 2.5 kg task mass is
        # enough to finish a 5 mm increment in 20 ms without saturating.
        return (2.5 * np.eye(3, dtype=np.float32))

    def _rollout_task_gains(self):
        stiff = getattr(self.param_, "robot_stiff_", None)
        if stiff is not None:
            k = float(np.asarray(stiff).reshape(-1)[0])
        else:
            k = ROLLOUT_TASK_KP
        return max(k, 1e-3), ROLLOUT_TASK_KD

    def _fingertip_contact_info(self):
        """Physical fingertip/object contact and the outward (object → tip) normal."""
        cached = getattr(self, "_contact_info_cache", None)
        if cached is not None:
            return cached
        in_contact, best_n, _sep = self.fingertip_object_contact()
        self._contact_info_cache = (in_contact, best_n)
        return in_contact, best_n

    def _fingertip_in_physical_contact(self):
        in_contact, _ = self._fingertip_contact_info()
        return in_contact

    def _clip_task_force(self, force, limit):
        force = np.asarray(force, dtype=np.float32).reshape(3)
        nrm = float(np.linalg.norm(force))
        if nrm > float(limit) and nrm > 1e-9:
            force = force * (float(limit) / nrm)
        return force

    @staticmethod
    def _orientation_error(r_current, r_desired):
        r_error = r_current.T @ r_desired
        # SciPy Rotation.as_quat() uses xyzw order, which matches Isaac Gym rigid-body states.
        # Keep the scalar part non-negative so we always take the shortest quaternion branch.
        err_quat_xyzw = Rotation.from_matrix(r_error).as_quat()
        if err_quat_xyzw[3] < 0.0:
            err_quat_xyzw = -err_quat_xyzw
        # OSC expects desired-current directly because the task torque uses +Kp * dpose.
        # The MuJoCo impedance controller uses the negated quantity and then multiplies by -K.
        return (r_current @ err_quat_xyzw[:3]).astype(np.float32)

    @staticmethod
    def _project_to_rotation_matrix(rotation_matrix):
        u, _, vh = np.linalg.svd(np.asarray(rotation_matrix, dtype=np.float64))
        r_proj = u @ vh
        if np.linalg.det(r_proj) < 0.0:
            u[:, -1] *= -1.0
            r_proj = u @ vh
        return r_proj.astype(np.float32)

    @staticmethod
    def _safe_inverse(matrix, rcond=1e-5):
        matrix = np.asarray(matrix, dtype=np.float64)
        try:
            inv = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(matrix, rcond=rcond)
        return inv.astype(np.float32)

    @staticmethod
    def _format_cartesian_gain(gain):
        gain = np.asarray(gain, dtype=np.float32)
        if gain.ndim == 0:
            return np.diag(np.full(6, float(gain), dtype=np.float32))
        if gain.ndim == 1:
            if gain.size == 1:
                return np.diag(np.full(6, float(gain.item()), dtype=np.float32))
            if gain.size == 6:
                return np.diag(gain.astype(np.float32))
            if gain.size == 2:
                return np.diag(np.array([gain[0], gain[0], gain[0], gain[1], gain[1], gain[1]], dtype=np.float32))
            raise ValueError(f"Invalid cartesian gain length: {gain.size}. Expected 1, 2, or 6.")
        if gain.shape == (6, 6):
            return gain.astype(np.float32)
        raise ValueError(f"Invalid cartesian gain shape: {gain.shape}. Expected scalar, (1,), (2,), (6,), or (6, 6).")

    def _compute_task_space_error(self, p_curr, r_curr):
        dpose = np.zeros(6, dtype=np.float32)
        dpose[:3] = self.position_d - p_curr
        dpose[3:] = self._orientation_error(r_curr, self.orientation_d)
        return dpose

    def _compute_osc_torques(self):
        self._refresh_osc_tensors()

        q = self.get_current_joint_position()
        qd = np.clip(np.nan_to_num(self.get_current_joint_velocity(), nan=0.0), -8.0, 8.0)
        p_curr, _ = self.get_end_effector_pos()
        jac_p = self._get_position_jacobian(q)
        vel = np.clip(jac_p @ qd, -2.0, 2.0)
        err = np.asarray(self.position_d - p_curr, dtype=np.float32).reshape(3)
        action = getattr(self, "_mpc_action", None)
        if action is None:
            action = getattr(self, "_last_mpc_action", None)
        p0 = getattr(self, "_mpc_p0", None)
        blocked = bool(getattr(self, "_path_blocked", False))
        if action is not None and p0 is not None:
            remain = _remaining_along_action(
                p_curr, p0, action, keep_lateral=blocked)
            horizon = float(getattr(self, "_mpc_T", None) or POLICY_INTERVAL)
            v_ref = _project_along_action(
                np.asarray(action, dtype=np.float64) / max(horizon, 1e-6), action)
        else:
            remain = err
            v_ref = None
        force_track = _isaac_task_force(
            remain, vel, v_ref=v_ref,
            k_task=ISAAC_TASK_KP, d_task=ISAAC_TASK_KD,
            limit=FREE_SPACE_FORCE_LIMIT,
        )
        if action is not None and not blocked:
            force_track = _project_along_action(force_track, action)
            # Damp raw velocity on contact so PhysX is not driven at v_ref.
            force_contact = _isaac_task_force(
                action, vel, v_ref=None,
                k_task=ISAAC_TASK_KP, d_task=ISAAC_TASK_KD,
                limit=CONTACT_FORCE_LIMIT,
            )
        else:
            force_contact = _isaac_task_force(
                err, vel, v_ref=None,
                k_task=ISAAC_TASK_KP, d_task=ISAAC_TASK_KD,
                limit=CONTACT_FORCE_LIMIT,
            )
        in_contact, contact_n = self._fingertip_contact_info()
        press = getattr(self, "_mpc_press", None)
        state_fn = getattr(self, "get_state", None)
        obj = None if state_fn is None else np.asarray(state_fn()[:3], dtype=np.float64)
        near_press = _near_press_force_mode(in_contact, p_curr, press=press, obj=obj)
        force = _osc_action_task_force(
            force_track, force_contact, near_press,
            bool(getattr(self, "_path_blocked", False)),
            contact_n_outward=_press_normal_outward(
                p_curr, press, fallback=contact_n),
            contact_limit=CONTACT_FORCE_LIMIT,
        )
        self._last_osc_force = np.asarray(force, dtype=np.float32).reshape(3)
        self._last_osc_pd = np.asarray(self.position_d, dtype=np.float32).reshape(3)
        self._last_osc_near_press = bool(near_press)
        tau_task = jac_p.T @ force

        # Position-only J: orientation lives in the nullspace and is held
        # at the post-placement configuration (fingertip normal along -Z
        # if placement started from the Franka home wrist).
        tau_nullspace = _franka_nullspace_posture_torque(
            jac_p, q, qd, self.q_d_nullspace, self.nullspace_stiffness
        )

        if self.activate_tool_compensation:
            tau_tool = jac_p.T @ self.tool_compensation_force[:3]
        else:
            tau_tool = np.zeros(7, dtype=np.float32)

        tau_d = np.nan_to_num(tau_task + tau_nullspace + tau_tool, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(tau_d, -self.torque_limits, self.torque_limits)

    def compute_cartesian_impedance_control(self):
        # Get current state
        self._refresh_osc_tensors()
        q = self.get_current_joint_position()
        dq = self.get_current_joint_velocity()
        jacobian = self._get_task_jacobian()
        p, R_current = self.get_end_effector_pos()
        cartesian_stiffness = self._format_cartesian_gain(self.cartesian_stiffness)
        cartesian_damping = self._format_cartesian_gain(self.cartesian_damping)
        
        # Compute error to desired pose
        error = np.zeros(6, dtype=np.float32)
        
        # Position error
        error[:3] = p - self.position_d
        
        # Orientation error
        R_d = self.orientation_d
        R_error = R_current.T @ R_d
        error_quat = Rotation.from_matrix(R_error).as_quat()
        
        # Check quaternion sign to ensure shortest path
        if error_quat[3] < 0:
            error_quat = -error_quat
            
        error[3:] = -R_current @ error_quat[:3]  # Transform to base frame
        
        # Compute end-effector velocity
        velocity = jacobian @ dq
        
        # Cartesian PD control
        F_ee_des = -cartesian_stiffness @ error - cartesian_damping @ velocity
        tau_task = jacobian.T @ F_ee_des
        
        # Same Franka posture term as OSC.  Position-only stiffness leaves
        # orientation in the nullspace so q_d can restore the -Z fingertip.
        position_only = float(np.diag(cartesian_stiffness)[3]) <= 1e-6
        task_jac = jacobian[:3] if position_only else jacobian
        tau_nullspace = _franka_nullspace_posture_torque(
            task_jac, q, dq, self.q_d_nullspace, self.nullspace_stiffness
        )
        
        # Tool compensation
        if self.activate_tool_compensation:
            tau_tool = jacobian.T @ self.tool_compensation_force
        else:
            tau_tool = np.zeros(7)
            
        tau_d = np.nan_to_num(tau_task + tau_nullspace + tau_tool, nan=0.0, posinf=0.0, neginf=0.0)
        tau_d = np.clip(tau_d, -self.torque_limits, self.torque_limits)
        return tau_d

    def _apply_arm_torque(self, tau):
        tau = np.asarray(tau, dtype=np.float32).reshape(-1)
        if tau.size != 7:
            raise ValueError(f"Invalid torque dimension: {tau.size}. Expected 7.")
        tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
        tau = np.clip(tau, -self.torque_limits, self.torque_limits)
        forces = np.zeros(self.franka_dof_count, dtype=np.float32)
        forces[:7] = tau
        if hasattr(self.gym, "set_actor_dof_actuation_force"):
            self.gym.set_actor_dof_actuation_force(self.env, self.franka_actor, forces)
        self._effort_control.zero_()
        self._effort_control[:7] = torch.as_tensor(
            tau, dtype=torch.float32, device=self._effort_control.device
        )
        if self.franka_dof_count >= 9:
            self.gym.set_actor_dof_position_targets(self.env, self.franka_actor, self._joint_targets)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
        return tau

    def set_control_torque(self, control_torque):
        self._apply_arm_torque(control_torque)

    def _track_desired_pose(self, preserve_nullspace_target=False, sync_realtime=False, draw=True):
        p_target = np.asarray(self.p_d, dtype=np.float32).copy()
        r_target = self._project_to_rotation_matrix(self.R_d_hold)

        if not preserve_nullspace_target:
            self.q_d_nullspace = self.get_current_joint_position().copy()
        self.set_desired_pose(p_target, r_target)
        tau = self._compute_osc_torques()
        self._apply_arm_torque(tau)
        self._simulate_once(sync_realtime=sync_realtime, draw=draw)
        return np.clip(tau, -self.torque_limits, self.torque_limits)

    def _clear_mpc_action(self):
        self._mpc_action = None
        self._last_mpc_action = None
        self._mpc_v_ref = None
        self._mpc_p0 = None
        self._mpc_press = None
        self._mpc_via = None
        self._path_blocked = False
        self._mpc_t = 0.0
        self._mpc_T = POLICY_INTERVAL

    def _advance_mpc_action_target(self):
        """Slide ``p_d`` along the MPC increment, same as the 3-DoF ball."""
        if self._mpc_action is None or self._mpc_p0 is None:
            return
        self._mpc_t = min(self._mpc_t + float(self.sim_dt_), float(self._mpc_T))
        self.p_d, self._mpc_v_ref = _mpc_ball_trajectory(
            self._mpc_p0, self._mpc_action, self._mpc_t, self._mpc_T
        )
        if self._mpc_t >= float(self._mpc_T) - 1e-9:
            self._mpc_v_ref = np.zeros(3, dtype=np.float32)
            if self._mpc_action is not None:
                self._last_mpc_action = np.asarray(self._mpc_action, dtype=np.float32).reshape(3)
            self._mpc_action = None
        self.R_d = self.R_d_hold.copy()

    def track_via(self, via_pos, n_substeps=None, sync_realtime=True):
        """Execute one MPC increment (action + velocity), not a pose hold."""
        self.set_via_action(via_pos)
        if n_substeps is None:
            n_substeps = policy_control_substeps(
                getattr(self.param_, "control_substeps_", 0), self.sim_dt_
            )
        applied_tau = None
        for i in range(n_substeps):
            last = (i + 1 == n_substeps)
            applied_tau = self.step_control_frame(
                draw=last, sync_realtime=bool(sync_realtime) and last
            )
        return applied_tau

    def step_joint_delta(self, dq, sync_realtime=True):
        q = self.get_current_joint_position()
        dq = np.asarray(dq, dtype=np.float32).reshape(7)

        # One MPPI joint action is one 20 ms --rollout interval.  Interpolate
        # the FK target across the OSC frames so the tip does not jump.
        n_substeps = policy_control_substeps(
            getattr(self.param_, "control_substeps_", 0), self.sim_dt_
        )
        applied_tau = None
        for i in range(n_substeps):
            frac = float(i + 1) / float(n_substeps)
            q_i = q + dq * frac
            T_i = _franka_fk_T_np(q_i).astype(np.float32)
            self.q_d_nullspace = q_i.copy()
            self.p_d = T_i[:3, 3].copy()
            self.R_d = T_i[:3, :3].copy()
            last = (i + 1 == n_substeps)
            applied_tau = self._track_desired_pose(
                preserve_nullspace_target=True,
                sync_realtime=bool(sync_realtime) and last,
                draw=last,
            )
        return applied_tau

    def step(self, cmd, sync_realtime=True):
        cmd = np.asarray(cmd, dtype=np.float32).reshape(-1)
        if cmd.size == 3:
            p_curr, _ = self.get_end_effector_pos()
            self.set_via_action(np.asarray(p_curr, dtype=np.float64) + cmd, action=cmd)
            return self.step_control_frame(draw=True, sync_realtime=sync_realtime)

        if cmd.size == 7:
            self.step_joint_delta(cmd, sync_realtime=sync_realtime)
            return

        raise ValueError(f"Invalid action dimension: {cmd.size}. Expected 3 or 7.")

    def set_joint_action(self, dq):
        """Start a 20 ms joint increment.  Isaac interpolates one frame at a time."""
        self._hold_q0 = np.asarray(self.get_current_joint_position(), dtype=np.float32)
        self._hold_dq = np.asarray(dq, dtype=np.float32).reshape(7)
        self._hold_n = policy_control_substeps(
            getattr(self.param_, "control_substeps_", 0), self.sim_dt_
        )
        self._hold_i = 0

    def set_via_action(self, via_pos, action=None, policy_dt=None,
                       press=None, path_blocked=None):
        """Execute the MPC action as a 20 ms fingertip trajectory."""
        p_curr, _ = self.get_end_effector_pos()
        p_curr = np.asarray(p_curr, dtype=np.float64).reshape(3)
        via = np.asarray(via_pos, dtype=np.float64).reshape(3)
        max_step = min(
            abs(float(getattr(self.param_, "mpc_u_ub_", 0.005))),
            abs(float(getattr(self.param_, "isaac_via_max_step_", AIR_VIA_STEP))),
        )
        max_slew = abs(float(getattr(
            self.param_, "isaac_action_slew_", ISAAC_ACTION_SLEW)))
        if bool(path_blocked):
            max_step = abs(float(getattr(self.param_, "mpc_u_ub_", 0.005)))
            max_slew = max(max_slew, max_step)
        target, increment = _clip_via_target(
            p_curr, via, action=action, max_step=max_step)
        increment = _slew_mpc_action(
            getattr(self, "_last_mpc_action", None), increment, max_step,
            max_slew=max_slew,
        )
        target = (p_curr + np.asarray(increment, dtype=np.float64).reshape(3)).astype(np.float32)
        horizon = float(policy_dt if policy_dt is not None else POLICY_INTERVAL)
        self._hold_dq = None
        self._hold_i = 0
        self._mpc_p0 = p_curr.astype(np.float32)
        self._mpc_action = np.asarray(increment, dtype=np.float32).reshape(3)
        self._last_mpc_action = self._mpc_action.copy()
        self._mpc_T = max(horizon, 1e-6)
        self._mpc_t = 0.0
        self._mpc_press = None if press is None else np.asarray(
            press, dtype=np.float64).reshape(3)
        self._mpc_via = via.copy()
        self._path_blocked = bool(path_blocked)
        self.p_d, self._mpc_v_ref = _mpc_ball_trajectory(
            self._mpc_p0, self._mpc_action, 0.0, self._mpc_T
        )
        self.R_d = self.R_d_hold.copy()

    def hold_current_pose(self):
        self._hold_dq = None
        self._hold_i = 0
        self._clear_mpc_action()
        p, r = self.get_end_effector_pos()
        self.p_d = np.asarray(p, dtype=np.float32).reshape(3).copy()
        self.R_d = np.asarray(r, dtype=np.float32).reshape(3, 3).copy()

    def step_control_frame(self, draw=True, sync_realtime=False):
        """One PhysX/OSC frame.  Holds the last target after the increment ends."""
        if self._hold_dq is not None and self._hold_i < self._hold_n:
            self._hold_i += 1
            q_i = np.asarray(
                joint_hold_target(self._hold_q0, self._hold_dq, self._hold_i, self._hold_n),
                dtype=np.float32,
            )
            T_i = _franka_fk_T_np(q_i).astype(np.float32)
            self.q_d_nullspace = q_i.copy()
            self.p_d = T_i[:3, 3].copy()
            self.R_d = T_i[:3, :3].copy()
            if self._hold_i >= self._hold_n:
                self._hold_dq = None
        elif self._mpc_action is not None:
            self._advance_mpc_action_target()
        tau = self._track_desired_pose(
            preserve_nullspace_target=True,
            sync_realtime=sync_realtime,
            draw=draw,
        )
        return tau

    def close(self):
        if self.svg_screenshot_recorder_ is not None:
            self.svg_screenshot_recorder_.close()
            self.svg_screenshot_recorder_ = None
        super().close()


class ContactIsaacCartesian:
    def __init__(self, param):
        self.param_ = param
        self._last_fingertip_contact = None

    def get_actual_fingertip_contact(self):
        return self._last_fingertip_contact

    @staticmethod
    def _contact_jacobian_body_frame(jacobian, body_mat):
        """Match contact.fingertips_collision_detection2.Contact."""
        jacobian = np.asarray(jacobian, dtype=np.float64).copy()
        if jacobian.ndim != 2 or jacobian.shape[1] < 6:
            return jacobian
        rot = np.asarray(body_mat, dtype=np.float64).reshape(3, 3)
        frame = np.zeros((6, 6), dtype=np.float64)
        frame[:3, :3] = rot
        frame[3:, 3:] = rot
        jacobian[:, :6] = jacobian[:, :6] @ frame
        return jacobian

    def detect_once(self, simulator):
        full_q = np.asarray(simulator.get_state(), dtype=np.float32)
        obj_pos = full_q[0:3]
        obj_quat_wxyz = full_q[3:7]
        tip_pos = end_effector_position(simulator, fallback=full_q[7:10])
        radius = fingertip_radius(simulator)
        contact_offset = float(getattr(self.param_, "physx_contact_offset_", DEFAULT_CONTACT_OFFSET))
        self._last_fingertip_contact = None

        nv = self.param_.n_qvel_
        max_ncon = self.param_.max_ncon_
        phi_vec = np.ones((max_ncon * 4,), dtype=np.float32)
        jac_mat = np.zeros((max_ncon * 4, nv), dtype=np.float32)
        jac_mat_env = np.zeros((max_ncon * 4, nv), dtype=np.float32)
        con_pos_list = []
        if_contact = False

        contacts_raw = simulator.gym.get_env_rigid_contacts(simulator.env)
        mu = float(self.param_.mu_object_)
        fingertip_sim_idx = fingertip_body_index(simulator)
        quat_xyzw = np.array(
            [obj_quat_wxyz[1], obj_quat_wxyz[2], obj_quat_wxyz[3], obj_quat_wxyz[0]],
            dtype=np.float32,
        )
        r_obj_to_world = Rotation.from_quat(quat_xyzw).as_matrix()
        table_height = float(self.param_.table_height)
        row_idx = 0
        table_seen = False
        best_ft_sep = float("inf")
        pose_cache = {}
        if contacts_raw is not None:
            for c in contacts_raw:
                b0, b1 = simulator._contact_bodies(c)
                if b0 is None:
                    continue
                involves_obj = (b0 == simulator.obj_body_idx) or (b1 == simulator.obj_body_idx)
                if not involves_obj:
                    continue
                other_sim_idx = b1 if b0 == simulator.obj_body_idx else b0
                other_is_table = (b0 == simulator.table_body_idx) or (b1 == simulator.table_body_idx)
                if other_is_table:
                    table_seen = True
                    continue
                other_is_fingertip = fingertip_sim_idx is not None and other_sim_idx == fingertip_sim_idx
                if not other_is_fingertip:
                    continue
                _sep, _speculative, n_raw = simulator._contact_sep_normal(c)
                if b1 == simulator.obj_body_idx:
                    n_raw = -n_raw
                cpos = simulator._contact_world_pos(c, b0, b1, pose_cache)
                overlap = contact_overlap(c)
                if cpos is None:
                    if overlap <= 1e-8 or tip_pos is None:
                        continue
                    cpos = np.asarray(tip_pos, dtype=np.float32) - radius * np.asarray(n_raw, dtype=np.float32)
                else:
                    cpos = np.asarray(cpos, dtype=np.float32)
                dist = physx_signed_gap(overlap, cpos, tip_pos, n_raw, radius)
                if contact_impulse(c) > 1e-6:
                    dist = min(float(dist), 0.0) if np.isfinite(dist) else 0.0
                # MuJoCo keeps margin contacts in the buffer.  Physical
                # contact for verify/pose-apply is still ``dist <= 0``.
                if not np.isfinite(dist) or dist > contact_offset + 1e-8:
                    continue
                if_contact = True
                surface = np.asarray(cpos, dtype=np.float32) - 0.5 * float(dist) * np.asarray(n_raw, dtype=np.float32)
                if dist < best_ft_sep:
                    best_ft_sep = dist
                    self._last_fingertip_contact = {
                        "dist": float(dist),
                        "point_world": surface,
                        "midpoint_world": np.asarray(cpos, dtype=np.float32),
                        "normal_world": np.asarray(n_raw, dtype=np.float32),
                    }
                if row_idx < max_ncon:
                    r_obj = surface - obj_pos
                    j_obj = np.zeros((3, nv), dtype=np.float32)
                    j_obj[:, 0:3] = np.eye(3, dtype=np.float32)
                    j_obj[:, 3:6] = -simulator._skew(r_obj)
                    j_other = np.zeros((3, nv), dtype=np.float32)
                    j_other[:, 6:9] = np.eye(3, dtype=np.float32)
                    n, t1, t2 = _tangent_basis_from_normal(n_raw)
                    con_jac = np.array(
                        _contact_jacobian(n, t1, t2, j_obj - j_other, mu),
                        dtype=np.float32,
                    )
                    phi_vec[4 * row_idx : 4 * row_idx + 4] = dist
                    jac_mat[4 * row_idx : 4 * row_idx + 4, :] = con_jac
                    row_idx += 1
        dist_table = float(obj_pos[2] - table_height)
        if table_seen or dist_table < 0.02:
            # One world-up plane.  Dumping every PhysX manifold point into
            # J_tilde treats each as independently supporting the object
            # weight and kills downward-press flip candidates.
            con_jac_table, con_jac_body, con_pos_local = _planar_table_jacobians(
                obj_pos, r_obj_to_world, table_height, nv, mu, simulator._skew
            )
            if row_idx < max_ncon:
                phi_vec[4 * row_idx : 4 * row_idx + 4] = min(max(dist_table, 0.0), 0.02)
                jac_mat[4 * row_idx : 4 * row_idx + 4, :] = con_jac_table
            jac_mat_env[0:4, :6] = con_jac_body
            con_pos_list.append(np.asarray(con_pos_local, dtype=np.float32))

        return phi_vec, jac_mat, con_pos_list, jac_mat_env, if_contact


def adapt_param_for_cartesian_solver(param, args):
    table_height = float(param.table_height)
    param.n_cmd_ = 3
    param.n_robot_qpos_ = 3
    param.n_qpos_ = 10
    param.n_qvel_ = 9
    param.robot_stiff_ = np.diag(3 * [float(args.cartesian_joint_stiffness)])
    q = np.zeros((param.n_qvel_, param.n_qvel_))
    q[:6, :6] = param.obj_inertia_
    q[6:, 6:] = param.robot_stiff_
    param.Q = q
    param.mpc_u_lb_ = -float(getattr(args, "mpc_step_limit", 0.005))
    param.mpc_u_ub_ = -param.mpc_u_lb_
    # Same fingertip box as test_0902 --rollout, shifted by the Isaac table.
    param.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7), np.array([-10.0, -10.0, table_height - 0.01])))
    param.mpc_q_ub_ = np.hstack((1e7 * np.ones(7), np.array([10.0, 10.0, table_height + 1.0])))
    args.solver = "acados"
    param.torch_solver = "acados"
    param.planner_solver_ = "acados"
    # Keep the ranking NLP built with the fingertip --rollout mass / hull.
    # Rebuilding after DyWA would score patches with the randomized sim mass.
    if getattr(param, "lambda_optimizer", None) is None:
        param.lambda_optimizer = build_lambda_optimizer(param, args)
    param.lambda_optimizer.solver = "acados"
    param.sol_guess_ = None
    return param


def _isaac_contact_distance(contact, env):
    contact.detect_once(env)
    measured = contact.get_actual_fingertip_contact()
    if measured is None:
        return float("inf")
    return float(measured.get("dist", float("inf")))


def _clip_toward(origin, target, max_step):
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    delta = target - origin
    dist = float(np.linalg.norm(delta))
    if dist <= float(max_step) or dist < 1e-9:
        return target.copy()
    return origin + delta * (float(max_step) / dist)


def _set_via_step(approach_via, step):
    step = max(1e-6, float(step))
    approach_via.max_step = step
    approach_via.max_lead = step


def _free_space_air_mode(if_contact, tip, press, path_blocked):
    """True when a larger via/OSC step cannot change contact force."""
    if bool(if_contact):
        return False
    tip = np.asarray(tip, dtype=np.float64).reshape(3)
    press = np.asarray(press, dtype=np.float64).reshape(3)
    dist = float(np.linalg.norm(tip - press))
    if dist <= 0.015:
        return False
    if dist <= NEAR_PRESS_SWITCH and not bool(path_blocked):
        return False
    return True


def _configure_rollout_param(param, args):
    param.rollout_press_patch = True
    param.quadratic_contact_track = True
    param.attract_coef = max(float(param.attract_coef), 20.0)
    param.field_cost_weight = 0.0
    param.contact_coef = max(float(param.contact_coef), float(param.attract_coef))
    param.lambda_optimizer.lock_contact_patch = False
    param.lambda_optimizer.contact_switch_confirm_steps = max(
        1, int(args.contact_switch_confirm_steps)
    )
    via_step = getattr(args, "via_max_step", None)
    param.isaac_via_max_step_ = (
        ISAAC_VIA_MAX_STEP if via_step is None else max(1e-4, float(via_step))
    )
    param.isaac_action_slew_ = max(
        1e-4, float(getattr(args, "action_slew", ISAAC_ACTION_SLEW))
    )
    param.isaac_orbit_extra_ = max(
        0.0, float(getattr(args, "orbit_extra", ISAAC_ORBIT_EXTRA))
    )
    return param


def _sample_pose_delta(opt, sample_idx):
    lookup = getattr(opt, "pose_delta_for_sample", None)
    if callable(lookup):
        return lookup(sample_idx)
    return None


def _contact_pose_diag(opt, policy, tip, osc_target=None):
    value_info = policy.get("value_info") or {}
    occupied_idx = value_info.get("occupied_idx")
    best_idx = getattr(opt, "last_global_idx", None)
    if best_idx is None:
        best_idx = getattr(opt, "last_selected_idx", None)
    executed_idx = getattr(opt, "last_executed_idx", None)
    diag = _diagnose_contact_pose_source(
        _sample_pose_delta(opt, occupied_idx),
        _sample_pose_delta(opt, best_idx),
        _sample_pose_delta(opt, executed_idx),
        tip,
        policy["p_arm_world"],
        policy["best_contact_world"],
        osc_target=osc_target if osc_target is not None else policy.get("mpc_virtual_point"),
    )
    diag["occupied_idx"] = occupied_idx
    diag["best_idx"] = best_idx
    diag["executed_idx"] = executed_idx
    return diag


def _rank_cost_rows(param, curr_q):
    """Per-sample lambda costs plus world height for ranking diagnostics."""
    opt = param.lambda_optimizer
    ids = getattr(opt, "last_candidate_ids", None)
    if ids is None:
        return []
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    if ids.size == 0:
        return []
    pos = np.asarray(curr_q[:3], dtype=np.float64).reshape(3)
    quat = np.asarray(curr_q[3:7], dtype=np.float64).reshape(4)
    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    table = float(getattr(param, "table_height", 0.0))
    pts = np.asarray(opt.sample_point, dtype=np.float64)
    nrm = np.asarray(opt.normal, dtype=np.float64)
    world = (rot @ pts[ids].T).T + pos
    n_world = (rot @ nrm[ids].T).T
    costs = np.asarray(getattr(opt, "last_candidate_costs", np.full(ids.size, np.nan)), dtype=np.float64).reshape(-1)
    raw = getattr(opt, "last_candidate_raw_costs", None)
    raw = np.full(ids.size, np.nan) if raw is None else np.asarray(raw, dtype=np.float64).reshape(-1)
    deltas = getattr(opt, "last_candidate_deltas", None)
    deltas = np.full(ids.size, np.nan) if deltas is None else np.asarray(deltas, dtype=np.float64).reshape(-1)
    curv = getattr(opt, "point_curvature", None)
    rows = []
    for i, idx in enumerate(ids):
        rows.append({
            "idx": int(idx),
            "z": float(world[i, 2]),
            "h": float(world[i, 2] - table),
            "xyz": world[i].astype(float),
            "nz": float(n_world[i, 2]),
            "cost": float(costs[i]) if i < costs.size else float("nan"),
            "raw": float(raw[i]) if i < raw.size else float("nan"),
            "dC": float(deltas[i]) if i < deltas.size else float("nan"),
            "curv": None if curv is None else float(curv[int(idx)]),
        })
    return rows


def _print_rank_cost_diag(param, curr_q, best_contact_world):
    """Show why last_global won, and whether a higher patch had better cost."""
    rows = _rank_cost_rows(param, curr_q)
    if not rows:
        print("rank_cost_diag: no candidates")
        return
    table = float(getattr(param, "table_height", 0.0))
    global_idx = getattr(param.lambda_optimizer, "last_global_idx", None)
    hs = np.asarray([r["h"] for r in rows], dtype=np.float64)
    improving = [r for r in rows if np.isfinite(r["dC"]) and r["dC"] > 1e-9]
    foot_band = 0.018
    n_foot = int(np.sum(hs <= foot_band))
    n_high = int(np.sum(hs > 0.03))
    best_high = None
    if improving:
        high_imp = [r for r in improving if r["h"] > foot_band]
        if high_imp:
            best_high = max(high_imp, key=lambda r: r["dC"])
    global_row = next((r for r in rows if r["idx"] == global_idx), None)
    top = sorted(
        [r for r in rows if np.isfinite(r["cost"])],
        key=lambda r: r["cost"],
    )[:5]
    print(
        "rank_cost_diag:",
        "n:", len(rows),
        "n_foot<=18mm:", n_foot,
        "n_high>30mm:", n_high,
        "best_z:", None if best_contact_world is None else round(float(best_contact_world[2]), 4),
        "best_h:", None if best_contact_world is None else round(float(best_contact_world[2] - table), 4),
        "global_idx:", global_idx,
        "global_h:", None if global_row is None else round(global_row["h"], 4),
        "global_dC:", None if global_row is None else round(global_row["dC"], 6),
        "global_cost:", None if global_row is None else round(global_row["cost"], 6),
        "global_nz:", None if global_row is None else round(global_row["nz"], 3),
        "best_high_idx:", None if best_high is None else best_high["idx"],
        "best_high_h:", None if best_high is None else round(best_high["h"], 4),
        "best_high_dC:", None if best_high is None else round(best_high["dC"], 6),
        "best_high_cost:", None if best_high is None else round(best_high["cost"], 6),
        "top5:", [
            (r["idx"], round(r["h"], 3), round(r["dC"], 5), round(r["cost"], 5), round(r["nz"], 2))
            for r in top
        ],
    )


def _print_contact_pose_diag(diag):
    print(
        "contact_pose_diag:",
        "source:", diag["source"],
        "occupied_idx:", diag["occupied_idx"],
        "occupied_dC:", None if diag["occupied_delta"] is None else round(float(diag["occupied_delta"]), 6),
        "best_idx:", diag["best_idx"],
        "best_dC:", None if diag["best_delta"] is None else round(float(diag["best_delta"]), 6),
        "executed_idx:", diag["executed_idx"],
        "executed_dC:", None if diag["executed_delta"] is None else round(float(diag["executed_delta"]), 6),
        "tip_to_best:", round(float(diag["tip_to_best"]), 4),
        "tip_to_p_arm:", round(float(diag["tip_to_p_arm"]), 4),
        "osc_track:", None if diag["osc_track"] is None else round(float(diag["osc_track"]), 4),
        "occupied_improves:", int(bool(diag["occupied_improves"])),
        "best_improves:", int(bool(diag["best_improves"])),
    )


# Isaac table is a 2 x 2 x 0.35 box at (1.2, 0, 0.175).
_TABLE_XY_MIN = np.array([0.20, -1.00], dtype=np.float64)
_TABLE_XY_MAX = np.array([2.20, 1.00], dtype=np.float64)


def _table_edge_dist_xy(obj_xy):
    xy = np.asarray(obj_xy, dtype=np.float64).reshape(2)
    return float(np.min(np.concatenate([xy - _TABLE_XY_MIN, _TABLE_XY_MAX - xy])))


def _finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _sample_cost(opt, sample_idx):
    if sample_idx is None:
        return None
    ids = getattr(opt, "last_candidate_ids", None)
    costs = getattr(opt, "last_candidate_costs", None)
    if ids is None or costs is None:
        return None
    hits = np.flatnonzero(np.asarray(ids, dtype=np.int32).reshape(-1) == int(sample_idx))
    if not hits.size:
        return None
    return _finite_or_none(np.asarray(costs, dtype=np.float64).reshape(-1)[int(hits[0])])


def _analyze_edge_pose_source(
    param, curr_q, policy, result, osc_force, if_contact, pos_err, quat_err,
    prev_pos_err, prev_quat_err,
):
    """Say whether a worsening desk-edge contact is lambda, verify, or OSC."""
    opt = param.lambda_optimizer
    value_info = policy.get("value_info") or {}
    p_arm = np.asarray(policy["p_arm_world"], dtype=np.float64).reshape(3)
    best = np.asarray(policy["best_contact_world"], dtype=np.float64).reshape(3)
    via = np.asarray(policy["mpc_virtual_point"], dtype=np.float64).reshape(3)
    tip = np.asarray(curr_q[7:10], dtype=np.float64).reshape(3)
    obj = np.asarray(curr_q[:3], dtype=np.float64).reshape(3)
    target = np.asarray(param.target_p_, dtype=np.float64).reshape(3)
    to_target = target - obj
    to_target_n = float(np.linalg.norm(to_target))
    to_target_u = to_target / max(to_target_n, 1e-9)
    edge = _table_edge_dist_xy(obj[:2])
    verify = _finite_or_none(result.get("verify_cost"))
    accept = bool(value_info.get("accept_p_arm", False))
    occupied_idx = value_info.get("occupied_idx")
    best_idx = getattr(opt, "last_global_idx", None)
    executed_idx = getattr(opt, "last_executed_idx", None)
    occ_dC = _sample_pose_delta(opt, occupied_idx)
    best_dC = _sample_pose_delta(opt, best_idx)
    exec_dC = _sample_pose_delta(opt, executed_idx)
    occ_cost = _sample_cost(opt, occupied_idx)
    best_cost = _sample_cost(opt, best_idx)
    exec_cost = _sample_cost(opt, executed_idx)
    if exec_cost is None:
        exec_cost = _finite_or_none(getattr(opt, "last_executed_cost", None))
    p_arm_dC = exec_dC if exec_dC is not None else occ_dC
    p_arm_cost = exec_cost if exec_cost is not None else occ_cost
    lambda_now = _finite_or_none(getattr(opt, "last_pose_cost_now", None))
    force = None if osc_force is None else np.asarray(osc_force, dtype=np.float64).reshape(3)
    force_n = 0.0 if force is None else float(np.linalg.norm(force))
    force_align = None
    if force is not None and force_n > 1e-6 and to_target_n > 1e-6:
        # Task force is the EE push into the world.  On contact the object
        # is shoved along +F.  Negative alignment increases position error.
        force_align = float(np.dot(force, to_target_u) / force_n)
    contact_align = float(np.dot(p_arm - obj, to_target_u))
    best_align = float(np.dot(best - obj, to_target_u))
    d_pos = None if prev_pos_err is None else float(pos_err - prev_pos_err)
    d_quat = None if prev_quat_err is None else float(quat_err - prev_quat_err)
    actual_worse = (d_pos is not None and d_pos > 5e-4) or (
        d_quat is not None and d_quat > 5e-4)
    lambda_improves = p_arm_dC is not None and float(p_arm_dC) > 1e-9
    best_improves = best_dC is not None and float(best_dC) > 1e-9
    near_p_arm = float(np.linalg.norm(tip - p_arm)) <= 0.03
    via_to_best = float(np.linalg.norm(via - best))
    via_to_parm = float(np.linalg.norm(via - p_arm))
    osc_away = bool(if_contact) and force_align is not None and force_align < -0.15
    verify_hold = accept or (verify is not None and float(verify) >= 0.25 and near_p_arm)
    planner_wants_best = (not accept) and via_to_best + 1e-6 < via_to_parm
    keepout = _keepout_radius(
        getattr(param, "object_aabb_lo", (-0.06, -0.04, -0.04)),
        getattr(param, "object_aabb_hi", (0.06, 0.04, 0.06)),
        getattr(param, "object_circumradius", None),
    )
    lo = np.asarray(getattr(param, "object_aabb_lo", (-0.06, -0.04, -0.04)), dtype=np.float64)
    hi = np.asarray(getattr(param, "object_aabb_hi", (0.06, 0.04, 0.06)), dtype=np.float64)
    half = 0.5 * (hi - lo)
    obj_r_xy = float(np.hypot(half[0], half[1]))
    tip_xy_r = float(np.linalg.norm(tip[:2] - obj[:2]))
    via_xy_r = float(np.linalg.norm(via[:2] - obj[:2]))
    opposite = bool(_on_opposite_sides(tip, obj, best))
    blocked_geom = bool(_press_path_blocked(tip, obj, best, keepout))
    core_r = max(0.028, 0.45 * float(keepout))
    chord_hits = bool(_segment_hits_core(tip, best, obj, radius=core_r))
    orbit_extra = float(getattr(param, "isaac_orbit_extra_", ISAAC_ORBIT_EXTRA))
    isaac_keepout = _travel_orbit_radius(keepout, orbit_extra)
    planner_wants_orbit = (
        (not accept)
        and (bool(value_info.get("path_blocked", False)) or opposite)
        and (verify is None or float(verify) < 0.25)
    )
    orbit_clips = planner_wants_orbit and (
        bool(if_contact) or tip_xy_r <= isaac_keepout
    ) and tip_xy_r < isaac_keepout + 0.01
    if lambda_improves and actual_worse:
        verdict = "lambda_false_improve"
    elif (not lambda_improves) and verify_hold and actual_worse:
        verdict = "verify_holds_bad_parm"
    elif orbit_clips and actual_worse:
        verdict = "osc_orbit_radius_clips"
    elif planner_wants_best and osc_away:
        verdict = "osc_presses_despite_planner"
    elif (not lambda_improves) and osc_away and not planner_wants_best:
        verdict = "osc_and_planner_push_away"
    elif lambda_improves and not actual_worse:
        verdict = "lambda_ok"
    else:
        verdict = "ambiguous"
    return {
        "verdict": verdict,
        "edge": edge,
        "pos_err": float(pos_err),
        "quat_err": float(quat_err),
        "d_pos": d_pos,
        "d_quat": d_quat,
        "verify": verify,
        "accept": int(accept),
        "phase": value_info.get("via_phase"),
        "blocked": int(bool(value_info.get("path_blocked", False))),
        "p_arm_cost": p_arm_cost,
        "p_arm_dC": None if p_arm_dC is None else float(p_arm_dC),
        "occ_cost": occ_cost,
        "occ_dC": None if occ_dC is None else float(occ_dC),
        "best_cost": best_cost,
        "best_dC": None if best_dC is None else float(best_dC),
        "lambda_now": lambda_now,
        "lambda_improves": int(lambda_improves),
        "best_improves": int(best_improves),
        "contact_align": contact_align,
        "best_align": best_align,
        "force_align": force_align,
        "force_n": force_n,
        "tip_to_p_arm": float(np.linalg.norm(tip - p_arm)),
        "tip_to_best": float(np.linalg.norm(tip - best)),
        "p_arm_to_best": float(np.linalg.norm(p_arm - best)),
        "via_to_best": via_to_best,
        "via_to_parm": via_to_parm,
        "keepout": float(keepout),
        "isaac_keepout": float(isaac_keepout),
        "obj_r_xy": float(obj_r_xy),
        "tip_xy_r": float(tip_xy_r),
        "via_xy_r": float(via_xy_r),
        "opposite": int(opposite),
        "blocked_geom": int(blocked_geom),
        "chord_hits": int(chord_hits),
        "orbit_clips": int(orbit_clips),
        "near_edge": int(edge <= 0.18),
        "if_contact": int(bool(if_contact)),
    }


def _print_edge_pose_source(info):
    print(
        "edge_source:",
        "verdict:", info["verdict"],
        "edge:", round(float(info["edge"]), 4),
        "pos_err:", round(float(info["pos_err"]), 5),
        "d_pos:", None if info["d_pos"] is None else round(float(info["d_pos"]), 5),
        "quat_err:", round(float(info["quat_err"]), 5),
        "d_quat:", None if info["d_quat"] is None else round(float(info["d_quat"]), 5),
        "verify:", None if info["verify"] is None else round(float(info["verify"]), 4),
        "accept:", info["accept"],
        "phase:", info["phase"],
        "blocked:", info["blocked"],
        "p_arm_cost:", None if info["p_arm_cost"] is None else round(float(info["p_arm_cost"]), 6),
        "p_arm_dC:", None if info["p_arm_dC"] is None else round(float(info["p_arm_dC"]), 6),
        "occ_dC:", None if info["occ_dC"] is None else round(float(info["occ_dC"]), 6),
        "best_dC:", None if info["best_dC"] is None else round(float(info["best_dC"]), 6),
        "best_cost:", None if info["best_cost"] is None else round(float(info["best_cost"]), 6),
        "lambda_now:", None if info["lambda_now"] is None else round(float(info["lambda_now"]), 6),
        "lambda_improves:", info["lambda_improves"],
        "best_improves:", info["best_improves"],
        "contact_align:", round(float(info["contact_align"]), 3),
        "best_align:", round(float(info["best_align"]), 3),
        "osc_align:", None if info["force_align"] is None else round(float(info["force_align"]), 3),
        "osc_fn:", round(float(info["force_n"]), 3),
        "tip_p_arm:", round(float(info["tip_to_p_arm"]), 4),
        "tip_best:", round(float(info["tip_to_best"]), 4),
        "p_arm_best:", round(float(info["p_arm_to_best"]), 4),
        "via_best:", round(float(info["via_to_best"]), 4),
        "via_parm:", round(float(info["via_to_parm"]), 4),
        "keepout:", round(float(info["keepout"]), 4),
        "isaac_ko:", round(float(info["isaac_keepout"]), 4),
        "obj_r:", round(float(info["obj_r_xy"]), 4),
        "tip_r:", round(float(info["tip_xy_r"]), 4),
        "via_r:", round(float(info["via_xy_r"]), 4),
        "opposite:", info["opposite"],
        "chord:", info["chord_hits"],
        "orbit_clips:", info["orbit_clips"],
        "contact:", info["if_contact"],
    )


def _print_rollout_step(
    param, curr_q, policy, value_info, verify_cost, verify_chatter,
    model_cost_conf, pred_reduction, act_reduction, if_contact, escape_on,
):
    """Same per-step fields as test_0902.py --rollout."""
    min_error = policy["min_error"]
    max_error = policy["max_error"]
    error = policy["error"]
    cached_cost = policy["cached_cost"]
    choose_dt = policy["choose_dt"]
    mpc_virtual_point = policy["mpc_virtual_point"]
    p_arm_world = policy["p_arm_world"]
    best_contact_world = policy["best_contact_world"]
    selected_idx = getattr(param.lambda_optimizer, "last_selected_idx",
                           getattr(param.lambda_optimizer, "last_best_idx", None))
    executed_idx = getattr(param.lambda_optimizer, "last_executed_idx", None)
    global_idx = getattr(param.lambda_optimizer, "last_global_idx", None)
    occupied_idx = None if not value_info else value_info.get("occupied_idx")
    occupied_dC = _sample_pose_delta(param.lambda_optimizer, occupied_idx)
    best_dC = _sample_pose_delta(param.lambda_optimizer, global_idx)
    executed_dC = _sample_pose_delta(param.lambda_optimizer, executed_idx)
    global_cost = getattr(param.lambda_optimizer, "last_global_total_cost", None)
    curv = getattr(param.lambda_optimizer, "point_curvature", None)
    curv_lim = float(getattr(param.lambda_optimizer, "region_max_point_curvature", 0.25))

    def _curv_of(idx):
        if curv is None or idx is None:
            return None
        try:
            return round(float(curv[int(idx)]), 4)
        except (IndexError, TypeError, ValueError):
            return None

    print(f"花费时间: {choose_dt:.4f}")
    raw_costs = getattr(param.lambda_optimizer, "last_candidate_raw_costs", None)
    if raw_costs is not None:
        raw_finite = np.asarray(raw_costs, dtype=np.float64).reshape(-1)
        raw_finite = raw_finite[np.isfinite(raw_finite)]
    else:
        raw_finite = np.zeros(0, dtype=np.float64)
    print(
        "min error:", min_error, "max error", max_error, "actual error:", error,
        "raw_min:", None if raw_finite.size == 0 else round(float(np.min(raw_finite)), 6),
        "raw_max:", None if raw_finite.size == 0 else round(float(np.max(raw_finite)), 6),
    )
    print(
        "verify cost:", None if verify_cost is None else round(float(verify_cost), 4),
        "verify_chatter:", int(bool(verify_chatter)),
        "p_arm_quality:", None if not value_info else round(float(value_info.get("quality", 0.0)), 4),
        "accept_p_arm:", None if not value_info else int(bool(value_info.get("accept_p_arm", False))),
        "pose_near_tie:", None if not value_info else int(bool(value_info.get("pose_near_tie", True))),
        "same_patch:", None if not value_info else int(bool(value_info.get("same_patch", False))),
        "q_dist:", None if not value_info else round(float(value_info.get("q_dist", 0.0)), 4),
        "verify_window:", None if not value_info else round(float(value_info.get("window_mean", 0.0)), 4),
        "verify_active:", None if not value_info else int(bool(value_info.get("contact_active", False))),
        "pose_pos_err:", float(metrics.comp_pos_error(curr_q[0:3], param.target_p_)),
        "pose_rot_err:", float(metrics.comp_quat_error(curr_q[3:7], param.target_q_)),
        "ball_to_best_contact:", round(float(np.linalg.norm(curr_q[7:10] - best_contact_world)), 6),
        "ball_to_p_arm:", round(float(np.linalg.norm(curr_q[7:10] - p_arm_world)), 6),
        "ball_to_virtual:", round(float(np.linalg.norm(curr_q[7:10] - mpc_virtual_point)), 6),
        "selected_idx:", selected_idx,
        "executed_idx:", executed_idx,
        "global_idx:", global_idx,
        "occupied_dC:", None if occupied_dC is None else round(float(occupied_dC), 6),
        "best_dC:", None if best_dC is None else round(float(best_dC), 6),
        "executed_dC:", None if executed_dC is None else round(float(executed_dC), 6),
        "topk:", np.asarray(getattr(param.lambda_optimizer, "last_topk_ids", []), dtype=int).tolist(),
        "selected_cost:", None if cached_cost is None else round(float(cached_cost), 6),
        "global_cost:", None if global_cost is None else round(float(global_cost), 6),
        "locked:", int(bool(param.lambda_optimizer.lock_contact_patch)),
        "confidence:", round(float(param.lambda_optimizer.contact_switch_confidence), 3),
        "model_tightness:", round(float(model_cost_conf.tightness()), 3),
        "model_cost_accum:", round(float(model_cost_conf.accum), 4),
        "pred_dcost:", None if pred_reduction is None else round(float(pred_reduction), 6),
        "act_dcost:", None if act_reduction is None else round(float(act_reduction), 6),
        "dwell:", int(getattr(param.lambda_optimizer, "_dwell_steps", 0)),
        "occupied_idx:", None if not value_info else value_info.get("occupied_idx"),
        "arrived:", None if not value_info else int(bool(value_info.get("arrived_at_best", False))),
        "via_phase:", None if not value_info else value_info.get("via_phase"),
        "path_blocked:", None if not value_info else int(bool(value_info.get("path_blocked", False))),
        "escape:", int(bool(escape_on)),
        "best_curv:", _curv_of(global_idx),
        "high_curv:", int(bool(_curv_of(global_idx) is not None and _curv_of(global_idx) > curv_lim)),
        "if_contact:", int(if_contact),
    )


def _add_rollout_policy_args(parser):
    add_rollout_via_args(parser)
    parser.add_argument("--use-xml-texture", action="store_true")
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--viewer", dest="headless", action="store_false")
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--show-ghost-object", type=_parse_bool_arg, default=False)
    parser.add_argument("--cartesian-joint-stiffness", type=float, default=100.0)
    parser.add_argument("--osc-pos-stiffness", type=float, default=12000.0)
    parser.add_argument("--osc-ori-stiffness", type=float, default=0.0)
    parser.add_argument("--nullspace-stiffness", type=float, default=10.0)
    parser.add_argument("--svg-screenshot-dir", type=str, default="")
    parser.add_argument("--svg-screenshot-interval", type=float, default=0.2)
    parser.add_argument("--svg-screenshot-width", type=int, default=1280)
    parser.add_argument("--svg-screenshot-height", type=int, default=960)
    parser.add_argument("--cartesian_stiffness", type=float, nargs="+", default=DEFAULT_CARTESIAN_STIFFNESS.tolist())
    parser.add_argument("--cartesian_damping", type=float, nargs="+", default=None)
    parser.add_argument("--effort-joint-damping", type=float, default=DEFAULT_EFFORT_JOINT_DAMPING)
    parser.add_argument(
        "--target-type",
        dest="target_type",
        type=str,
        default="ground-rotation",
        choices=("ground-rotation", "rotation"),
        help="ground-rotation matches fingertips --rollout (90 deg pitch flip).",
    )
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument(
        "--control-substeps",
        type=int,
        default=0,
        help="OSC frames per via update.  0 uses the 20 ms --rollout interval.",
    )
    parser.add_argument(
        "--via-max-lead",
        type=float,
        default=ISAAC_VIA_MAX_LEAD,
        help="Max via/press offset from the current fingertip (m).  "
             "Shorter than the MuJoCo 5 mm carrot so Franka actions stay small.",
    )
    parser.add_argument(
        "--via-max-step",
        type=float,
        default=ISAAC_VIA_MAX_STEP,
        help="SmoothedApproachVia and OSC increment cap (m).",
    )
    parser.add_argument(
        "--via-smooth-rate",
        type=float,
        default=ISAAC_VIA_SMOOTH_RATE,
        help="SmoothedApproachVia lerp rate.  Smaller interpolates more.",
    )
    parser.add_argument(
        "--action-slew",
        type=float,
        default=ISAAC_ACTION_SLEW,
        help="Max change of the executed increment between policy steps (m).",
    )
    parser.add_argument(
        "--orbit-extra",
        type=float,
        default=ISAAC_ORBIT_EXTRA,
        help="Added only to the via orbit radius (m).  "
             "Default 0: same keep-out as path_blocked / verify_cost.",
    )
    parser.add_argument(
        "--async-planner",
        dest="async_planner",
        action="store_true",
        help="ROS-style latest-only state/action topics in a second process.",
    )
    parser.add_argument(
        "--sync-planner",
        dest="async_planner",
        action="store_false",
        help="Rank+MPC in the Isaac process (debug).",
    )
    parser.add_argument("--viewer-hz", type=float, default=60.0)
    parser.add_argument(
        "--floating-fingertip",
        action="store_true",
        help="Stage-1 PhysX 3-DoF sphere.  No Franka / OSC.",
    )
    parser.add_argument(
        "--gpu-physx",
        action="store_true",
        help="GPU PhysX in the viewer (fights Warp MPPI on the same device).",
    )
    parser.set_defaults(headless=False, async_planner=True)
    return parser


def run_mpc_planner(bus, args):
    from examples.mpc.franka.ik2.isaac_bus import wait_trial_obs
    from planning.mpc_explicit import (
        _build_mpc_planner_runtime,
        handle_mpc_request,
        planner_init_payload,
    )

    if args.trial_start < 0:
        raise ValueError(f"trial_start must be non-negative, got {args.trial_start}")
    if int(args.trial_num) <= 0:
        raise ValueError(f"trial_num must be positive, got {args.trial_num}")

    trial_start = int(args.trial_start)
    trial_num = max(1, int(args.trial_num))
    trial_stop = trial_start + trial_num
    success_pos_threshold = 0.02
    success_quat_threshold = 0.015
    consecutive_success_time_threshold = 0
    max_rollout_length = max(1, int(args.max_rollout_length))
    mpc_step = max(1e-4, float(getattr(args, "mpc_step_limit", 0.005)))
    via_step = getattr(args, "via_max_step", None)
    exec_step = mpc_step if via_step is None else min(mpc_step, max(1e-4, float(via_step)))
    success_rate = 0
    viewer_quit = False

    for trial_count in range(trial_start, trial_stop):
        param = ExplicitMPCParams(
            args,
            rand_seed=trial_count,
            target_type=getattr(args, "target_type", "ground-rotation"),
            mpc_model="explicit",
        )
        param = _apply_dywa_physics_to_param(param)
        param.use_jax_contact_ = False
        param = adapt_param_for_cartesian_solver(param, args)
        param = _configure_rollout_param(param, args)
        param.control_substeps_ = int(args.control_substeps)
        init = planner_init_payload(args, param, trial_count)
        _, plan_param, mpc, trackers = _build_mpc_planner_runtime(init)

        obs = wait_trial_obs(bus, trial_count)
        if obs is None:
            viewer_quit = True
            break

        policy = None
        c_now_cost = None
        pred_reduction = None
        last_accept_p_arm = False
        escape_on = False
        verify_chatter = False
        last_result = None
        rollout_step = 0
        consecutive_success_time = 0
        min_pos_err = float("inf")
        min_quat_err = float("inf")
        pose_apply_count = 0
        choose_times = []
        pos_err_now = None
        quat_err_now = None
        prev_pos_err = None
        prev_quat_err = None
        edge_verdicts = []

        while rollout_step < max_rollout_length:
            if obs.get("break_out") or obs.get("viewer_closed") or obs.get("cmd") == "stop":
                viewer_quit = True
                break
            curr_q = np.asarray(obs["policy_q"], dtype=np.float32)
            pos_err_now = float(metrics.comp_pos_error(curr_q[0:3], param.target_p_))
            quat_err_now = float(metrics.comp_quat_error(curr_q[3:7], param.target_q_))
            min_pos_err = min(min_pos_err, pos_err_now)
            min_quat_err = min(min_quat_err, quat_err_now)
            if pos_err_now < success_pos_threshold and quat_err_now < success_quat_threshold:
                consecutive_success_time += 1
            else:
                consecutive_success_time = 0
            if consecutive_success_time > consecutive_success_time_threshold:
                break

            contact_distance = float(obs.get("contact_distance", float("inf")))
            dwell = None
            if policy is not None:
                dwell = _dwell_payload(
                    None, args, None, policy, last_accept_p_arm, escape_on,
                    verify_chatter, c_now_cost, pred_reduction, contact_distance,
                    pos_err_now, curr_q=curr_q, param=param,
                )
            payload, curr_q = _plan_from_obs(obs, dwell)
            result = handle_mpc_request(args, plan_param, mpc, trackers, payload)
            last_result = result
            policy = _apply_plan_result(param, None, result, curr_q, _print_rollout_step)
            choose_times.append(float(result["rank_dt"]))
            c_now_cost, pred_reduction = _pred_reduction_from_policy(param, curr_q, policy)
            last_accept_p_arm = bool(policy["value_info"].get("accept_p_arm", False))
            escape_on = bool(policy["escape_on"])
            verify_chatter = bool(result["verify_chatter"])
            action = np.asarray(result["action"], dtype=np.float64).reshape(3)
            tip_now = np.asarray(curr_q[7:10], dtype=np.float64)
            action = np.asarray(_clip_mpc_action(action, exec_step), dtype=np.float64)
            exec_via = np.asarray(policy["mpc_virtual_point"], dtype=np.float64).reshape(3)

            on_exec_contact = False
            physical_contact = bool(np.isfinite(contact_distance) and contact_distance <= 0.003)
            if physical_contact:
                on_exec_contact = float(np.linalg.norm(tip_now - np.asarray(policy["p_arm_world"], dtype=float))) <= 0.03
                if on_exec_contact:
                    pose_apply_count += 1
                _print_contact_pose_diag(
                    _contact_pose_diag(
                        param.lambda_optimizer, policy, tip_now,
                        osc_target=tip_now + action,
                    )
                )
            print(
                "contact_distance:",
                None if not np.isfinite(contact_distance) else round(contact_distance, 6),
                "physics_contact:", int(on_exec_contact),
            )
            _print_rank_cost_diag(param, curr_q, policy["best_contact_world"])
            osc_force = obs.get("osc_force")
            osc_pd = obs.get("osc_pd")
            edge_info = _analyze_edge_pose_source(
                param, curr_q, policy, result, osc_force, physical_contact,
                pos_err_now, quat_err_now, prev_pos_err, prev_quat_err,
            )
            if edge_info["near_edge"] or edge_info["if_contact"] or float(pos_err_now) > 0.05:
                _print_edge_pose_source(edge_info)
                edge_verdicts.append((rollout_step, edge_info["verdict"], edge_info))
            prev_pos_err = float(pos_err_now)
            prev_quat_err = float(quat_err_now)
            print(
                "diag_push:",
                "step:", rollout_step,
                "obj:", np.round(np.asarray(curr_q[:3], dtype=float), 4).tolist(),
                "obj_z:", round(float(curr_q[2]), 4),
                "table:", round(float(param.table_height), 4),
                "below_table:", int(float(curr_q[2]) < float(param.table_height) - 0.01),
                "tip:", np.round(tip_now, 4).tolist(),
                "via:", np.round(exec_via, 4).tolist(),
                "virtual:", np.round(np.asarray(policy["mpc_virtual_point"], dtype=float), 4).tolist(),
                "p_arm:", np.round(np.asarray(policy["p_arm_world"], dtype=float), 4).tolist(),
                "best:", np.round(np.asarray(policy["best_contact_world"], dtype=float), 4).tolist(),
                "action:", np.round(action, 4).tolist(),
                "osc_pd:", None if osc_pd is None else np.round(np.asarray(osc_pd, dtype=float), 4).tolist(),
                "osc_f:", None if osc_force is None else np.round(np.asarray(osc_force, dtype=float), 3).tolist(),
                "osc_fn:", None if osc_force is None else round(float(np.linalg.norm(osc_force)), 3),
                "near_press:", int(bool(obs.get("osc_near_press", False))),
                "verify:", None if result.get("verify_cost") is None else round(float(result["verify_cost"]), 4),
                "conf:", round(float(param.lambda_optimizer.contact_switch_confidence), 3),
                "tight:", round(float(result.get("model_tightness", 0.0)), 3),
                "accept:", int(bool(policy["value_info"].get("accept_p_arm", False))),
                "pose_tie:", int(bool(policy["value_info"].get("pose_near_tie", True))),
                "blocked:", int(bool((policy.get("value_info") or {}).get("path_blocked", False))),
                "phase:", (policy.get("value_info") or {}).get("via_phase"),
                "opposite:", edge_info["opposite"],
                "keepout:", round(float(edge_info["keepout"]), 4),
                "tip_r:", round(float(edge_info["tip_xy_r"]), 4),
                "via_r:", round(float(edge_info["via_xy_r"]), 4),
                "orbit_clips:", edge_info["orbit_clips"],
                "dwell:", int(getattr(param.lambda_optimizer, "_dwell_steps", 0)),
                "blacklisted:", len(getattr(param.lambda_optimizer, "_blocked_contact_indices", {})),
            )
            bus.publish_cmd({
                "kind": "via",
                "via": exec_via,
                "action": action,
                "policy_dt": POLICY_INTERVAL,
                "press": policy["p_arm_world"],
                "path_blocked": bool((policy.get("value_info") or {}).get(
                    "path_blocked", False)),
                "markers": {
                    "target": policy["mpc_virtual_point"],
                    "best_contact": policy["best_contact_world"],
                },
            })
            rollout_step += 1
            obs = bus.wait_obs(timeout=180.0)
            if obs is None:
                viewer_quit = True
                break

        bus.publish_cmd({"cmd": "end_trial"})
        if last_result is not None:
            lambda_failures = int(last_result.get("lambda_failures", 0))
            mpc_failures = int(last_result.get("mpc_failures", 0))
            if lambda_failures or mpc_failures or last_result.get("mpc_init_error"):
                print("acados diagnostics:", {
                    "lambda_solves": int(last_result.get("lambda_solves", 0)),
                    "lambda_failures": lambda_failures,
                    "mpc_solves": int(last_result.get("mpc_solves", 0)),
                    "mpc_failures": mpc_failures,
                    "mpc_init_error": last_result.get("mpc_init_error") or None,
                })
        if edge_verdicts:
            counts = {}
            worse = [item for item in edge_verdicts if item[2].get("d_pos") is not None and item[2]["d_pos"] > 5e-4]
            for _, verdict, _ in edge_verdicts:
                counts[verdict] = counts.get(verdict, 0) + 1
            worse_counts = {}
            for _, verdict, _ in worse:
                worse_counts[verdict] = worse_counts.get(verdict, 0) + 1
            print("edge_verdict_summary:", {
                "n": len(edge_verdicts),
                "n_pos_worse": len(worse),
                "all": counts,
                "pos_worse": worse_counts,
            })
        choose_arr = np.asarray(choose_times, dtype=np.float64) if choose_times else np.array([0.0])
        print("trial_summary:", {
            "trial": trial_count,
            "mode": "mpc_planner",
            "success": int(rollout_step < max_rollout_length),
            "steps": rollout_step,
            "pose_applies": pose_apply_count,
            "final_pos_err": None if pos_err_now is None else round(float(pos_err_now), 5),
            "final_quat_err": None if quat_err_now is None else round(float(quat_err_now), 5),
            "min_pos_err": None if not np.isfinite(min_pos_err) else round(float(min_pos_err), 5),
            "min_quat_err": None if not np.isfinite(min_quat_err) else round(float(min_quat_err), 5),
            "choose_dt_mean": round(float(np.mean(choose_arr)), 5),
            "choose_dt_max": round(float(np.max(choose_arr)), 5),
            "choose_dt_p95": round(float(np.percentile(choose_arr, 95)), 5),
        })
        success_rate += 1 if rollout_step < max_rollout_length else 0
        if viewer_quit:
            break

    print(
        f"Success rate over {trial_num} trials "
        f"(trial ids {trial_start} to {trial_stop - 1}): "
        f"{success_rate}/{trial_num} = {success_rate / trial_num:.2%}"
    )
    return viewer_quit


def planner_worker(cmd_q, obs_q, ready_q, init):
    from examples.mpc.franka.ik2.isaac_bus import IsaacBus, args_from_init

    if ready_q is not None:
        ready_q.put({"ok": True, "pid": os.getpid()})
    run_mpc_planner(IsaacBus(cmd_q, obs_q), args_from_init(init))


def main():
    if "--floating-fingertip" in sys.argv:
        script = os.path.join(
            parent_dir, "examples", "mpc", "fingertips", "test", "test_0902_isaac.py"
        )
        argv = [sys.executable, script] + [
            arg for arg in sys.argv[1:] if arg != "--floating-fingertip"
        ]
        os.execv(sys.executable, argv)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")
    os.environ.pop("SCSP_PLANNER_ONLY", None)
    os.execv(sys.executable, [sys.executable, script, "--planner", "mpc", *sys.argv[1:]])


if __name__ == "__main__":
    main()
