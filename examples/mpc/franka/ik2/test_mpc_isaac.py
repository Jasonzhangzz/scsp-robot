import argparse
import os
import re
import sys
import time

import numpy as np
from scipy.linalg import pinv
from scipy.spatial.transform import Rotation
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

from examples.mpc.franka.ik2.params import ExplicitMPCParams, build_lambda_optimizer
from examples.mpc.franka.ik2.test_mppi_isaac import (
    IsaacFrankaSimulator,
    _parse_bool_arg,
    _extract_quat_xyzw,
    _extract_vec3,
)
from examples.mpc.fingertips.test.test_0902 import (
    ContactValueTracker,
    ModelCostConfidence,
    SmoothedApproachVia,
    _arrived_at_best_contact,
    _blend_travel_to_press,
    _candidate_solution,
    _floor_slide_away_from_patch,
    _keepout_radius,
    _lambda_pose_cost,
    _line_in_best_fov_and_cone,
    _nearest_sample_idx,
    _object_top_z_world,
    _on_opposite_sides,
    _patch_press_point,
    _patch_proximity,
    _predicted_object_pose,
    _protect_destination_dwell,
    _rollout_dwell_assignment,
    _rollout_verify_cost,
    _same_contact_patch,
    _sample_world,
    _should_observe_model_cost,
    _sphere_center_on_patch,
    _verify_distance,
    _verify_is_chatter,
    _x_plus_is_usable,
)
from planning.mpc_explicit import MPCExplicit
from planning.MPPIExplicit import _contact_jacobian, _franka_fk_T_jax, _tangent_basis_from_normal
from planning.mpc_implicit import MPCImplicit
from planning.screenshot import create_isaacgym_svg_screenshot_recorder
from utils import metrics, rotations

DYWA_SIM_DT = 0.0125
DYWA_SIM_SUBSTEPS = 1
DYWA_PHYSX_SOLVER_TYPE = 1
DYWA_PHYSX_POSITION_ITERATIONS = 8
DYWA_PHYSX_VELOCITY_ITERATIONS = 1
DYWA_PHYSX_CONTACT_OFFSET = 0.001
DYWA_PHYSX_REST_OFFSET = 0.0
DYWA_PHYSX_FRICTION_OFFSET_THRESHOLD = 0.001
DYWA_PHYSX_FRICTION_CORRELATION_DISTANCE = 0.0005
DYWA_PHYSX_MAX_DEPENETRATION_VELOCITY = 10.0
DYWA_TABLE_FRICTION_RANGE = (0.3, 0.8)
DYWA_OBJECT_FRICTION_RANGE = (0.2, 1.0)
DYWA_OBJECT_MASS_RANGE = (0.1, 0.5)

DEFAULT_CARTESIAN_STIFFNESS = np.array([2000.0, 2000.0, 2000.0, 50.0, 50.0, 50.0], dtype=np.float32)
DEFAULT_EFFORT_JOINT_DAMPING = 10.0
SVG_SCREENSHOT_CAMERA_POSITION = np.array([0.7, 0.00, 0.63], dtype=np.float32)
SVG_SCREENSHOT_CAMERA_TARGET = np.array([0.1, 0.00, 0.32], dtype=np.float32)


# DEFAULT_ELEPHANT_TRIAL_REPLAY = {
#     # Matches the elephant parameter-search entry whose trial 1 succeeded.
#     "obj": "cube",
#     "attract_coef": 0.5,
#     "reject_coef": 0.001,
#     "contact_coef": 0.5,
#     "contact_cost_param": 0.0,
#     "model_param": 6.0,
#     "reject_dis": 0.01,
#     "attract_point_comp": 0.1,
#     "ground_height_threshold": 0.012,
#     "sample_num": 70,
#     "pos_coef": 1,
#     "ori_coef": 0.02,
#     "low_err_coef": 0.75,
#     "upper_err_coef": 0.95,
#     "sim_device": "cuda:0",
#     "graphics_device_id": 0,
#     "cartesian_step": 0.05,
#     "cartesian_joint_stiffness": 100.0,
#     "cartesian_dls_lambda": 0.0001,
#     "osc_pos_stiffness": 2000.0,
#     "osc_ori_stiffness": 400.0,
#     # "headless": True,
#     "trial_start": 5,
#     "trial_count": 1,
# }
DEFAULT_ELEPHANT_TRIAL_REPLAY = {
    "obj": "foam_brick",
    "attract_coef": 0.5,
    "reject_coef": 0.001,
    "contact_coef": 0.7,
    "contact_cost_param": 1.0,
    "model_param": 7.0,
    "reject_dis": 0.02,
    "attract_point_comp": 0.1,
    "ground_height_threshold": 0.012,
    "sample_num": 70,
    "pos_coef": 500,
    "ori_coef": 20,
    "sim_device": "cuda:0",
    "graphics_device_id": 0,
    "cartesian_joint_stiffness": 100.0,
    "osc_pos_stiffness": 2000.0,
    "osc_ori_stiffness": 400.0,
    "trial_start": 3,
    "trial_count": 1,
}


def _quat_xyzw_to_matrix(quat_xyzw):
    return Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix().astype(np.float32)


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


def _set_actor_mass(gym, env, actor_handle, mass):
    body_props = gym.get_actor_rigid_body_properties(env, actor_handle)
    for prop in body_props:
        prop.mass = float(mass)
    gym.set_actor_rigid_body_properties(env, actor_handle, body_props, True)


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
    param.table_friction_ = float(np.random.uniform(*DYWA_TABLE_FRICTION_RANGE))
    param.object_friction_ = float(np.random.uniform(*DYWA_OBJECT_FRICTION_RANGE))
    param.obj_mass_ = float(np.random.uniform(*DYWA_OBJECT_MASS_RANGE))
    param.mu_object_ = float(param.object_friction_)
    param.gravity_[2] = -9.81
    return param


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
        sim_params.physx.use_gpu = (compute_id >= 0)

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
            gymapi.Vec3(1.0, 0.9, 0.1),
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
            gymapi.Vec3(1.0, 0.1, 0.1),
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

    @staticmethod
    def _prepare_mesh_urdf_assets(repo_root, mesh_path):
        if mesh_path is None:
            raise ValueError("param.mesh_path_ is required for Isaac object asset loading")

        mesh_abs_path = mesh_path if os.path.isabs(mesh_path) else os.path.join(repo_root, mesh_path)
        mesh_abs_path = os.path.abspath(mesh_abs_path)
        if not os.path.isfile(mesh_abs_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_abs_path}")

        mesh_asset_root = os.path.join(repo_root, "envs", "assets", "objects", "_isaac_tmp")
        os.makedirs(mesh_asset_root, exist_ok=True)
        mesh_rel_to_urdf = os.path.relpath(mesh_abs_path, mesh_asset_root)
        mesh_scale = "1 1 1"
        obj_urdf_rel = "obj_mesh_dynamic.urdf"
        obj_urdf_abs = os.path.join(mesh_asset_root, obj_urdf_rel)
        target_urdf_rel = "obj_mesh_target_ghost.urdf"
        target_urdf_abs = os.path.join(mesh_asset_root, target_urdf_rel)

        obj_urdf = f"""<?xml version="1.0"?>
<robot name="mesh_obj">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.05"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_rel_to_urdf}" scale="{mesh_scale}"/>
      </geometry>
      <material name="obj_color">
        <color rgba="0.2 0.6 1.0 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_rel_to_urdf}" scale="{mesh_scale}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
        ghost_urdf = f"""<?xml version="1.0"?>
<robot name="mesh_target_ghost">
  <link name="base">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_rel_to_urdf}" scale="{mesh_scale}"/>
      </geometry>
      <material name="ghost_color">
        <color rgba="0.95 0.95 0.95 0.35"/>
      </material>
    </visual>
  </link>
</robot>
"""
        with open(obj_urdf_abs, "w", encoding="ascii") as f:
            f.write(obj_urdf)
        with open(target_urdf_abs, "w", encoding="ascii") as f:
            f.write(ghost_urdf)

        return obj_urdf_rel, target_urdf_rel, mesh_asset_root

    def _apply_scene_physics_settings(self):
        self.table_friction_ = float(getattr(self.param_, "table_friction_", 0.5))
        self.object_friction_ = float(getattr(self.param_, "object_friction_", 0.5))
        self.obj_mass_ = float(getattr(self.param_, "obj_mass_", 0.1))
        _set_actor_friction(self.gym, self.env, self.table_actor, self.table_friction_)
        _set_actor_friction(self.gym, self.env, self.obj_actor, self.object_friction_)
        _set_actor_mass(self.gym, self.env, self.obj_actor, self.obj_mass_)

    def _configure_franka(self):
        dof_props = self.gym.get_actor_dof_properties(self.env, self.franka_actor)
        effort_joint_damping = float(
            getattr(self.param_, "effort_joint_damping_", DEFAULT_EFFORT_JOINT_DAMPING)
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

        pos_stiffness = float(getattr(self.param_, "osc_pos_stiffness_", 150.0))
        ori_stiffness = float(getattr(self.param_, "osc_ori_stiffness_", 400.0))
        self.osc_task_kp = np.array(
            [pos_stiffness, pos_stiffness, pos_stiffness, ori_stiffness, ori_stiffness, ori_stiffness],
            dtype=np.float32,
        )
        self.osc_task_kd = (2.0 * np.sqrt(self.osc_task_kp)).astype(np.float32)
        self.nullspace_stiffness = 10.0
        self.home_q = np.array(self.param_.init_robot_qpos_, dtype=np.float32)
        self.q_d_nullspace = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
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
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

    def reset_mj_env(self):
        super().reset_mj_env()
        self._sync_desired_pose_with_current()

    def _sync_desired_pose_with_current(self):
        self._refresh_osc_tensors()
        p, r = self.get_end_effector_pos()
        self.position_d = p.copy()
        self.orientation_d = r.copy()
        self.p_d = p.copy()
        self.R_d = r.copy()
        self.R_d_hold = r.copy()
        # self.q_d_nullspace = self.get_current_joint_position().copy()
        self.q_d_nullspace = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    def _simulate_once(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self._sync_pose_axes()

        graphics_stepped = False
        if self.viewer_ is not None or self.svg_screenshot_recorder_ is not None:
            self.gym.step_graphics(self.sim)
            graphics_stepped = True

        if self.svg_screenshot_recorder_ is not None:
            self.svg_screenshot_recorder_.capture_if_due(
                sim_time=float(self.gym.get_sim_time(self.sim)),
                step_graphics=not graphics_stepped,
            )

        if self.viewer_ is not None:
            self.gym.draw_viewer(self.viewer_, self.sim, True)
            self.gym.sync_frame_time(self.sim)

    def _get_body_pose(self, local_idx):
        states = self.gym.get_actor_rigid_body_states(self.env, self.franka_actor, gymapi.STATE_POS)
        p = _extract_vec3(states["pose"]["p"][local_idx])
        # Isaac rigid-body states store world-frame orientations as quaternions in xyzw order.
        quat_xyzw = _extract_quat_xyzw(states["pose"]["r"][local_idx])
        r = _quat_xyzw_to_matrix(quat_xyzw)
        return p, r

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

    def _get_arm_mass_matrix(self):
        mm = self._mm
        if mm.ndim == 3:
            mm = mm[0]
        return np.array(mm[:7, :7], dtype=np.float32)

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
        qd = self.get_current_joint_velocity()
        p_curr, r_curr = self.get_end_effector_pos()
        jac = self._get_task_jacobian()
        mm = self._get_arm_mass_matrix()
        mm_inv = self._safe_inverse(mm)
        m_task_inv = jac @ mm_inv @ jac.T
        m_task = self._safe_inverse(m_task_inv)

        dpose = self._compute_task_space_error(p_curr, r_curr)
        ee_velocity = jac @ qd
        tau_task = jac.T @ (m_task @ (self.osc_task_kp * dpose - self.osc_task_kd * ee_velocity))

        j_task_inv = m_task @ jac @ mm_inv
        q_error = (self.q_d_nullspace - q + np.pi) % (2.0 * np.pi) - np.pi
        tau_nullspace = (
            2.0 * np.sqrt(self.nullspace_stiffness) * (-qd)
            + self.nullspace_stiffness * q_error
        ).astype(np.float32)
        tau_nullspace = mm @ tau_nullspace
        tau_nullspace = (np.eye(7, dtype=np.float32) - jac.T @ j_task_inv) @ tau_nullspace

        if self.activate_tool_compensation:
            tau_tool = jac.T @ self.tool_compensation_force
        else:
            tau_tool = np.zeros(7, dtype=np.float32)

        tau_d = tau_task + tau_nullspace + tau_tool
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
        
        # Nullspace control
        jacobian_pinv = pinv(jacobian.T)
        nullspace_proj = np.eye(7) - jacobian.T @ jacobian_pinv
        tau_nullspace = nullspace_proj @ (self.nullspace_stiffness * (self.q_d_nullspace - q) - 
                                         2 * np.sqrt(self.nullspace_stiffness) * dq)
        
        # Tool compensation
        if self.activate_tool_compensation:
            tau_tool = jacobian.T @ self.tool_compensation_force
        else:
            tau_tool = np.zeros(7)
            
        # Total desired torque
        tau_d = tau_task + tau_nullspace + tau_tool
        
        # Saturate torque
        tau_d = np.clip(tau_d, -self.torque_limits, self.torque_limits)
        
        return tau_d

    def _apply_arm_torque(self, tau):
        tau = np.asarray(tau, dtype=np.float32).reshape(-1)
        if tau.size != 7:
            raise ValueError(f"Invalid torque dimension: {tau.size}. Expected 7.")
        tau = np.clip(tau, -self.torque_limits, self.torque_limits)
        self._effort_control.zero_()
        self._effort_control[:7] = torch.as_tensor(tau, dtype=torch.float32, device=self._effort_control.device)
        if self.franka_dof_count >= 9:
            self.gym.set_actor_dof_position_targets(self.env, self.franka_actor, self._joint_targets)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
        return tau

    def set_control_torque(self, control_torque):
        self._apply_arm_torque(control_torque)

    def _track_desired_pose(self, preserve_nullspace_target=False):
        p_target = np.asarray(self.p_d, dtype=np.float32).copy()
        r_target = self._project_to_rotation_matrix(self.R_d_hold)

        if not preserve_nullspace_target:
            # self.q_d_nullspace = self.get_current_joint_position().copy()
            self.q_d_nullspace = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.set_desired_pose(p_target, r_target)
        target_quat = Rotation.from_matrix(r_target.astype(np.float64)).as_quat()
        # tau = self._compute_osc_torques()
        tau = self.compute_cartesian_impedance_control()
        self._apply_arm_torque(tau)
        # self.show_target(goal_pos=p_target, goal_quat=target_quat)
        self._simulate_once()
        return np.clip(tau, -self.torque_limits, self.torque_limits)

    def step_joint_delta(self, dq):
        q = self.get_current_joint_position()
        dq = np.asarray(dq, dtype=np.float32).reshape(7)
        q_des = q + dq

        T_des = np.array(_franka_fk_T_jax(q_des), dtype=np.float32)
        p_des = T_des[:3, 3]
        r_des = T_des[:3, :3]

        self.q_d_nullspace = q_des.copy()
        self.p_d = p_des.copy()
        self.R_d = r_des.copy()
        # A planner action represents one complete joint-space delta.  One
        # PhysX frame is generally too short for the effort-controlled OSC to
        # track that target, especially once the fingertip is loaded by a
        # contact.  Holding the same target for a few frames reduces the
        # rollout/execution mismatch and permits sustained contact force.  The
        # shared MPC environment keeps its previous behavior unless a caller
        # explicitly configures control_substeps_.
        control_substeps = max(int(getattr(self.param_, "control_substeps_", 1)), 1)
        applied_tau = None
        for _ in range(control_substeps):
            applied_tau = self._track_desired_pose(preserve_nullspace_target=True)
        return applied_tau

    def step(self, cmd):
        cmd = np.asarray(cmd, dtype=np.float32).reshape(-1)
        if cmd.size == 3:
            p_curr, _ = self.get_end_effector_pos()
            # print("p_curr = ", p_curr)
            self.p_d = p_curr + cmd
            # self.p_d[2] = max(self.p_d[2], self.low_height)
            self.R_d = self.R_d_hold.copy()
            return self._track_desired_pose()

        if cmd.size == 7:
            self.step_joint_delta(cmd)
            return

        raise ValueError(f"Invalid action dimension: {cmd.size}. Expected 3 or 7.")

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

    def detect_once(self, simulator: IsaacFrankaOSCSimulator):
        full_q = simulator.get_state()
        full_q = np.asarray(full_q, dtype=np.float32)
        obj_pos = full_q[0:3]
        obj_quat_wxyz = full_q[3:7]
        self._last_fingertip_contact = None

        nv = self.param_.n_qvel_
        max_ncon = self.param_.max_ncon_
        phi_vec = np.ones((max_ncon * 4,), dtype=np.float32)
        jac_mat = np.zeros((max_ncon * 4, nv), dtype=np.float32)
        jac_mat_env = np.zeros((max_ncon * 4, nv), dtype=np.float32)
        con_pos_list = []
        if_contact = False

        contacts = simulator.get_physx_contacts()
        mu = float(self.param_.mu_object_)
        contact_sep_threshold = float(getattr(self.param_, "if_contact_separation_threshold_", 0.0))
        fingertip_sim_idx = getattr(simulator, "franka_body_name_to_index", {}).get("fingertip")
        row_idx = 0
        row_env_idx = 0
        best_ft_sep = float("inf")
        for c in contacts:
            b0 = c["body0"]
            b1 = c["body1"]
            sep = float(c["separation"])
            n_raw = c["normal"]
            cpos = c["pos"]

            involves_obj = (b0 == simulator.obj_body_idx) or (b1 == simulator.obj_body_idx)
            if not involves_obj:
                continue

            if b1 == simulator.obj_body_idx:
                n_raw = -n_raw

            if cpos is None:
                cpos = obj_pos + 0.01 * n_raw
            else:
                cpos = np.asarray(cpos, dtype=np.float32)

            r_obj = cpos - obj_pos
            j_obj = np.zeros((3, nv), dtype=np.float32)
            j_obj[:, 0:3] = np.eye(3, dtype=np.float32)
            j_obj[:, 3:6] = -simulator._skew(r_obj)

            j_other = np.zeros((3, nv), dtype=np.float32)
            other_sim_idx = b1 if b0 == simulator.obj_body_idx else b0
            if other_sim_idx in simulator.franka_body_indices:
                j_other[:, 6:9] = np.eye(3, dtype=np.float32)

            j_rel_point = j_obj - j_other
            n, t1, t2 = _tangent_basis_from_normal(n_raw)
            con_jac = np.array(_contact_jacobian(n, t1, t2, j_rel_point, mu), dtype=np.float32)

            other_is_franka = (b0 in simulator.franka_body_indices) or (b1 in simulator.franka_body_indices)
            other_is_fingertip = fingertip_sim_idx is not None and other_sim_idx == fingertip_sim_idx
            # Isaac rigid contacts may include proximity pairs with positive separation.
            # We only report a true robot-object contact when the pair is actually
            # touching / penetrating according to PhysX's separation convention.
            if other_is_franka and sep <= contact_sep_threshold:
                if_contact = True
            if other_is_fingertip and sep < best_ft_sep:
                best_ft_sep = sep
                self._last_fingertip_contact = {"dist": float(sep), "point_world": cpos}
            if other_is_franka and row_idx < max_ncon:
                phi_vec[4 * row_idx : 4 * row_idx + 4] = 0.5 * sep
                jac_mat[4 * row_idx : 4 * row_idx + 4, :] = con_jac
                row_idx += 1

            other_is_table = (b0 == simulator.table_body_idx) or (b1 == simulator.table_body_idx)
            if other_is_table and row_idx < max_ncon:
                phi_vec[4 * row_idx : 4 * row_idx + 4] = 0.5 * sep
                jac_mat[4 * row_idx : 4 * row_idx + 4, :] = con_jac
                row_idx += 1
            if other_is_table and row_env_idx < max_ncon:
                jac_mat_env[4 * row_env_idx : 4 * row_env_idx + 4, :6] = con_jac[:, :6]
                row_env_idx += 1
                quat_xyzw = np.array(
                    [obj_quat_wxyz[1], obj_quat_wxyz[2], obj_quat_wxyz[3], obj_quat_wxyz[0]],
                    dtype=np.float32,
                )
                r_obj_to_world = Rotation.from_quat(quat_xyzw).as_matrix()
                con_pos_local = r_obj_to_world.T @ (cpos - obj_pos)
                con_pos_list.append(np.array(con_pos_local, dtype=np.float32))
        if row_env_idx == 0:
            dist_table = float(obj_pos[2] - float(self.param_.table_height))
            if dist_table < 0.02:
                n_t = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                t1_t = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                t2_t = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                j_rel_table = np.concatenate([np.eye(3), np.zeros((3, 6))], axis=1)
                con_jac_table = np.array(_contact_jacobian(n_t, t1_t, t2_t, j_rel_table, mu), dtype=np.float32)
                jac_mat_env[0:4, :6] = con_jac_table[:, :6]
                con_pos_list.append(np.array([0.0, 0.0, -0.025], dtype=np.float32))

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
    param.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7), np.array([-1.0, -1.0, table_height + 0.02])))
    param.mpc_q_ub_ = np.hstack((1e7 * np.ones(7), np.array([1.5, 1.0, 1.5])))
    args.solver = "acados"
    param.torch_solver = "acados"
    param.planner_solver_ = "acados"
    param.lambda_optimizer = build_lambda_optimizer(param, args)
    param.lambda_optimizer.solver = "acados"
    param.sol_guess_ = None
    return param


def _isaac_contact_distance(contact, env):
    contact.detect_once(env)
    measured = contact.get_actual_fingertip_contact()
    if measured is None:
        return float("inf")
    return abs(float(measured.get("dist", float("inf"))))


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
    return param


def _add_rollout_policy_args(parser):
    parser.add_argument("--obj", type=str, default="stanford_bunny2")
    parser.add_argument("--use-xml-texture", action="store_true")
    parser.add_argument("--attract_coef", type=float, default=0.5)
    parser.add_argument("--reject_coef", type=float, default=0.001)
    parser.add_argument("--contact_coef", type=float, default=0.7)
    parser.add_argument("--contact_cost_param", type=float, default=1)
    parser.add_argument("--model_param", type=float, default=7)
    parser.add_argument("--reject_dis", type=float, default=0.02)
    parser.add_argument("--attract_point_comp", type=float, default=0.1)
    parser.add_argument("--ground_height_threshold", type=float, default=0.012)
    parser.add_argument("--fingertip_clearance", type=float, default=0.011)
    parser.add_argument("--sample_num", type=int, default=70)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--normal_stability_cos", type=float, default=0.95)
    parser.add_argument("--pos_coef", type=float, default=500)
    parser.add_argument("--ori_coef", type=float, default=20)
    parser.add_argument("--mpc_step_limit", type=float, default=0.005)
    parser.add_argument("--friction_reg_coef", type=float, default=1.0)
    parser.add_argument("--force_reg_coef", type=float, default=0.01)
    parser.add_argument("--max_contact_force", type=float, default=10.0)
    parser.add_argument("--contact_switch_radius", type=float, default=0.03)
    parser.add_argument("--contact_switch_margin_ratio", type=float, default=0.08)
    parser.add_argument("--contact_switch_margin_abs", type=float, default=0.001)
    parser.add_argument("--contact_switch_confirm_steps", type=int, default=5)
    parser.add_argument("--contact_dwell_gamma", type=float, default=0.70)
    parser.add_argument("--contact_dwell_steps", type=int, default=4)
    parser.add_argument("--model_cost_error_threshold", type=float, default=6.0)
    parser.add_argument("--model_cost_error_eps", type=float, default=1e-6)
    parser.add_argument("--model_cost_error_min_steps", type=int, default=3)
    parser.add_argument("--value_tau", type=float, default=1.0)
    parser.add_argument("--value_rel_scale", type=float, default=0.08)
    parser.add_argument("--value_rho", type=float, default=0.08)
    parser.add_argument("--value_alpha", type=float, default=0.25)
    parser.add_argument("--verify_beta", type=float, default=0.18)
    parser.add_argument("--verify_window", "--verify_window_size", dest="verify_window_size", type=int, default=5)
    parser.add_argument("--verify_enter_steps", type=int, default=5)
    parser.add_argument("--verify_hold_steps", type=int, default=30)
    parser.add_argument("--verify_release_steps", type=int, default=8)
    parser.add_argument("--ideal_contact_surface_margin", type=float, default=-0.0005)
    parser.add_argument("--spline_escape_cost", type=int, default=1)
    parser.add_argument("--detour_attract_coef", type=float, default=80.0)
    parser.add_argument("--detour_repel_coef", type=float, default=40.0)
    parser.add_argument("--detour_lift_coef", type=float, default=25.0)
    parser.add_argument("--detour_align_thresh", type=float, default=0.50)
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--viewer", dest="headless", action="store_false")
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--show-ghost-object", type=_parse_bool_arg, default=False)
    parser.add_argument("--cartesian-joint-stiffness", type=float, default=100.0)
    parser.add_argument("--osc-pos-stiffness", type=float, default=1000.0)
    parser.add_argument("--osc-ori-stiffness", type=float, default=100.0)
    parser.add_argument("--svg-screenshot-dir", type=str, default="")
    parser.add_argument("--svg-screenshot-interval", type=float, default=0.2)
    parser.add_argument("--svg-screenshot-width", type=int, default=1280)
    parser.add_argument("--svg-screenshot-height", type=int, default=960)
    parser.add_argument("--cartesian_stiffness", type=float, nargs="+", default=DEFAULT_CARTESIAN_STIFFNESS.tolist())
    parser.add_argument("--cartesian_damping", type=float, nargs="+", default=None)
    parser.add_argument("--effort-joint-damping", type=float, default=DEFAULT_EFFORT_JOINT_DAMPING)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--trial-count", type=int, default=20)
    parser.add_argument("--max_rollout_length", type=int, default=5000)
    parser.set_defaults(headless=False, **DEFAULT_ELEPHANT_TRIAL_REPLAY)
    return parser


def main():
    parser = argparse.ArgumentParser()
    _add_rollout_policy_args(parser)
    args = parser.parse_args()
    args.solver = "acados"
    args.rollout = True

    if args.trial_start < 0:
        raise ValueError(f"trial_start must be non-negative, got {args.trial_start}")
    if args.trial_count <= 0:
        raise ValueError(f"trial_count must be positive, got {args.trial_count}")

    trial_start = int(args.trial_start)
    trial_num = int(args.trial_count)
    trial_stop = trial_start + trial_num
    success_pos_threshold = 0.02
    success_quat_threshold = 0.04
    consecutive_success_time_threshold = 20
    max_rollout_length = max(1, int(args.max_rollout_length))
    fingertip_radius = 0.01
    success_rate = 0

    for trial_count in range(trial_start, trial_stop):
        param = ExplicitMPCParams(args, rand_seed=trial_count, target_type="rotation", mpc_model="explicit")
        param = _apply_dywa_physics_to_param(param)
        param.use_jax_contact_ = False
        param = adapt_param_for_cartesian_solver(param, args)
        param = _configure_rollout_param(param, args)
        param.osc_pos_stiffness_ = float(args.osc_pos_stiffness)
        param.osc_ori_stiffness_ = float(args.osc_ori_stiffness)
        param.cartesian_stiffness_ = np.array(args.cartesian_stiffness, dtype=np.float32)
        param.cartesian_damping_ = (
            None if args.cartesian_damping is None else np.array(args.cartesian_damping, dtype=np.float32)
        )
        param.effort_joint_damping_ = float(args.effort_joint_damping)
        param.show_ghost_object_ = bool(args.show_ghost_object)
        param.svg_screenshot_dir_ = args.svg_screenshot_dir or None
        param.svg_screenshot_interval_ = float(args.svg_screenshot_interval)
        param.svg_screenshot_width_ = int(args.svg_screenshot_width)
        param.svg_screenshot_height_ = int(args.svg_screenshot_height)
        param.svg_screenshot_prefix_ = f"trial_{trial_count:03d}"

        contact = ContactIsaacCartesian(param)
        env = IsaacFrankaOSCSimulator(
            param,
            headless=args.headless,
            sim_device=args.sim_device,
            graphics_device_id=args.graphics_device_id,
        )
        env.show_target_object_pose(param.target_p_, param.target_q_)

        mpc = MPCExplicit(param) if param.mpc_model == "explicit" else MPCImplicit(param)
        table_ground = float(param.table_height) + 0.012
        height_threshold = float(param.table_height) + float(args.ground_height_threshold)

        rollout_step = 0
        consecutive_success_time = 0
        verify_cost = 0.0
        value_tracker = ContactValueTracker(
            tau=float(args.value_tau),
            rel_scale=float(args.value_rel_scale),
            rho=float(args.value_rho),
            alpha=float(args.value_alpha),
            beta=float(args.verify_beta),
            window_size=int(args.verify_window_size),
            confirm_steps=int(args.verify_enter_steps),
            min_hold_steps=int(args.verify_hold_steps),
            release_steps=int(args.verify_release_steps),
            accept_margin_ratio=0.05,
            accept_margin_abs=0.02,
        )
        value_info = {}
        arrived_hold = False
        arrived_dest_idx = None
        approach_via = SmoothedApproachVia(
            max_step=max(1e-4, float(args.mpc_step_limit) + 0.001)
        )
        last_verify_cost = None
        last_accept_p_arm = None
        verify_chatter = False
        model_cost_conf = ModelCostConfidence(
            threshold=float(args.model_cost_error_threshold),
            eps=float(args.model_cost_error_eps),
            min_steps=int(args.model_cost_error_min_steps),
        )
        pred_reduction = None
        act_reduction = None
        c_now_cost = None
        choose_times = []

        while rollout_step < max_rollout_length:
            if env.dyn_paused_:
                env._simulate_once()
                continue

            curr_q = env.get_policy_state()
            phi_vec, jac_mat, _con_point, jac_mat_env, if_contact = contact.detect_once(env)
            r_obj_to_world = Rotation.from_quat([curr_q[4], curr_q[5], curr_q[6], curr_q[3]]).as_matrix()
            gravity = np.hstack([r_obj_to_world.T @ param.gravity_[:3] * param.obj_mass_, np.zeros(3)])
            target_quat_local = rotations.quaternion_multiply(
                rotations.quaternion_conjugate(curr_q[3:7]), param.target_q_
            )
            target_pose_eval = np.hstack([r_obj_to_world.T @ (param.target_p_ - curr_q[:3]), target_quat_local])
            current_pose_eval = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            current_tip_local = r_obj_to_world.T @ (curr_q[7:10] - curr_q[:3])

            param.lambda_optimizer.update_Jacobian(jac_mat_env)
            visible_point_idx = param.lambda_optimizer.get_availble_point_idx(
                curr_q[0:3], r_obj_to_world, param.target_p_, height_threshold,
                viewpoint_local=None, heading_filter=False,
            )
            visible_point_idx = param.lambda_optimizer.filter_rankable_indices(visible_point_idx)

            last_idx = getattr(param.lambda_optimizer, "last_selected_idx", None)
            last_exec_idx = getattr(param.lambda_optimizer, "last_executed_idx", None)
            blocked = getattr(param.lambda_optimizer, "_blocked_contact_indices", {})
            visible_point_idx = np.asarray(visible_point_idx, dtype=np.int32)
            incumbents = [int(idx) for idx in (last_idx, last_exec_idx) if idx is not None]
            param.lambda_optimizer.lock_contact_patch = bool(
                value_tracker.contact_active
                and value_tracker._holding_p_arm
                and param.lambda_optimizer.contact_switch_confidence >= (1.0 - 1e-9)
                and any(idx in visible_point_idx and idx not in blocked for idx in incumbents)
            )

            start_time = time.time()
            rank_anchor_local = None
            last_global = getattr(param.lambda_optimizer, "last_global_idx", None)
            if last_idx is not None:
                last_best_world = _sample_world(param.lambda_optimizer, last_idx, curr_q[:3], r_obj_to_world)
                near_selected = float(np.linalg.norm(curr_q[7:10] - last_best_world)) <= float(
                    param.lambda_optimizer.contact_switch_radius
                )
                same_as_global = last_global is None or _same_contact_patch(
                    param.lambda_optimizer, last_idx, last_global, radius=0.03
                )
                if near_selected and same_as_global:
                    rank_anchor_local = current_tip_local

            best_contact_point, normal, min_error, max_error, _ = param.lambda_optimizer.choose_contact_points(
                target_pose_eval,
                current_pose_eval,
                gravity,
                visible_point_idx,
                contact_anchor_local=rank_anchor_local,
                v_last=None,
                force_required=True,
            )
            cached_x_plus = getattr(param.lambda_optimizer, "last_best_x_plus", None)
            cached_force = getattr(param.lambda_optimizer, "last_best_force", None)
            cached_cost = getattr(param.lambda_optimizer, "last_best_cost", None)
            global_cost = getattr(param.lambda_optimizer, "last_global_total_cost", None)
            candidate_costs = getattr(param.lambda_optimizer, "last_candidate_costs", None)
            reference_candidates = [global_cost, min_error]
            reference_candidates.extend(
                np.asarray(candidate_costs if candidate_costs is not None else [], dtype=np.float64).reshape(-1).tolist()
            )
            reference_candidates = [
                float(v) for v in reference_candidates if v is not None and np.isfinite(float(v))
            ]
            reference_error = min(reference_candidates) if reference_candidates else None
            choose_dt = time.time() - start_time
            choose_times.append(choose_dt)

            guide_idx = param.lambda_optimizer.choose_nearby_topk_idx(current_tip_local)
            if guide_idx is None:
                guide_idx = getattr(param.lambda_optimizer, "last_global_idx", None)
            if guide_idx is not None:
                guide_idx = int(guide_idx)
                best_contact_point = param.lambda_optimizer.sample_point[guide_idx]
                normal = param.lambda_optimizer.normal[guide_idx]
                cached_x_plus, cached_force, cached_cost = _candidate_solution(param.lambda_optimizer, guide_idx)
                if cached_x_plus is not None:
                    param.lambda_optimizer.last_best_x_plus = cached_x_plus
                if cached_force is not None:
                    param.lambda_optimizer.last_best_force = cached_force
                if cached_cost is not None:
                    param.lambda_optimizer.last_best_cost = cached_cost

            best_contact_track_world, best_contact_world, best_normal_world = _sphere_center_on_patch(
                curr_q[:3], r_obj_to_world, best_contact_point, normal,
                fingertip_radius, args.ideal_contact_surface_margin,
            )
            p_arm_local, p_arm_normal_out, x_plus_opt, error, info = param.lambda_optimizer.resolve_executed_contact(
                current_tip_local, visible_point_idx,
                target_pose_eval, current_pose_eval, gravity,
                v_last=None,
                sphere_radius=max(1e-4, float(fingertip_radius) + float(args.ideal_contact_surface_margin)),
            )
            p_arm_surface_world = r_obj_to_world @ p_arm_local + curr_q[:3]
            p_arm_inward_local = -np.asarray(p_arm_normal_out, dtype=np.float64)
            p_arm_inward_world = r_obj_to_world @ p_arm_inward_local
            p_arm_inward_world /= max(float(np.linalg.norm(p_arm_inward_world)), 1e-9)
            p_arm_track_world = p_arm_surface_world - max(
                1e-4, float(fingertip_radius) + float(args.ideal_contact_surface_margin)
            ) * p_arm_inward_world
            p_arm_world = p_arm_track_world

            p_arm_force = np.asarray(info.get("control_input", np.zeros(3)), dtype=np.float64).reshape(-1)
            solver_ok = (
                not bool(info.get("solver_failed", False))
                and np.isfinite(float(error))
                and p_arm_force.size >= 3
                and np.isfinite(p_arm_force[:3]).all()
                and float(np.linalg.norm(p_arm_force[:3])) > 1e-3
            )
            p_arm_idx = getattr(param.lambda_optimizer, "last_executed_idx", None)
            best_idx = getattr(param.lambda_optimizer, "last_global_idx", None)
            if best_idx is None:
                best_idx = getattr(param.lambda_optimizer, "last_selected_idx", None)
            occupied_idx = _nearest_sample_idx(param.lambda_optimizer, current_tip_local)
            arrived_at_best = _arrived_at_best_contact(
                curr_q[7:10], best_contact_world, best_contact_track_world,
                occupied_idx, best_idx, param.lambda_optimizer,
            )
            dist_best_now = min(
                float(np.linalg.norm(curr_q[7:10] - best_contact_world)),
                float(np.linalg.norm(curr_q[7:10] - best_contact_track_world)),
            )
            if arrived_at_best:
                arrived_hold = True
                if arrived_dest_idx is None and best_idx is not None:
                    arrived_dest_idx = int(best_idx)
            elif arrived_hold and dist_best_now > 0.035:
                arrived_hold = False
                arrived_dest_idx = None
            arrived_at_best = bool(arrived_at_best or arrived_hold)
            if (
                _on_opposite_sides(curr_q[7:10], curr_q[:3], best_contact_world)
                or _floor_slide_away_from_patch(curr_q[7:10], best_contact_world, ground=table_ground)
            ):
                arrived_at_best = False
                arrived_hold = False
                arrived_dest_idx = None
            if arrived_at_best and best_idx is not None:
                param.lambda_optimizer.last_executed_idx = int(best_idx)
                param.lambda_optimizer.last_executed_x_plus = cached_x_plus
                param.lambda_optimizer.last_executed_cost = cached_cost
                param.lambda_optimizer.last_executed_force = cached_force
                p_arm_idx = int(best_idx)
                p_arm_world = best_contact_track_world
                p_arm_track_world = best_contact_track_world
                p_arm_surface_world = best_contact_world
                x_plus_opt = cached_x_plus
                error = float(cached_cost) if cached_cost is not None else float(min_error)
                info = {
                    "control_input": cached_force if cached_force is not None else np.zeros(3),
                    "solver_failed": cached_x_plus is None,
                }
            same_patch = _same_contact_patch(param.lambda_optimizer, p_arm_idx, best_idx, radius=0.03)
            if arrived_at_best:
                same_patch = True
            dist_arm = float(np.linalg.norm(curr_q[7:10] - p_arm_world))
            near_arm = dist_arm <= float(param.lambda_optimizer.contact_switch_radius)
            is_best_sample = best_idx is not None and p_arm_idx is not None and int(best_idx) == int(p_arm_idx)
            if arrived_at_best:
                is_best_sample = True
            value_tracker.reset_arm(p_arm_idx)
            model_cost_conf.note_sample(
                param.lambda_optimizer, best_idx if best_idx is not None else p_arm_idx
            )
            model_tightness = model_cost_conf.tightness()
            value_info = value_tracker.update_values(
                reference_error, error, solver_ok=solver_ok,
                candidate_costs=candidate_costs, same_patch=same_patch,
                near_arm=near_arm, is_best_sample=is_best_sample,
                confidence=param.lambda_optimizer.contact_switch_confidence,
                stagnant_steps=getattr(param.lambda_optimizer, "_dwell_steps", 0),
                min_error=min_error, max_error=max_error, tightness=model_tightness,
            )
            if not value_info["accept_p_arm"]:
                p_arm_world = best_contact_track_world
                p_arm_track_world = best_contact_track_world
                p_arm_surface_world = best_contact_world
                x_plus_opt = cached_x_plus
                error = float(cached_cost) if cached_cost is not None else float(min_error)
                info = {
                    "control_input": cached_force if cached_force is not None else np.zeros(3),
                    "solver_failed": cached_x_plus is None,
                }
            dist_surface = float(np.linalg.norm(curr_q[7:10] - best_contact_world))
            dist_track = float(np.linalg.norm(curr_q[7:10] - best_contact_track_world))
            dist_to_exec = float(np.linalg.norm(curr_q[7:10] - p_arm_world))
            align_info = _line_in_best_fov_and_cone(
                curr_q[7:10], p_arm_world, p_arm_surface_world, best_normal_world,
                mu=float(getattr(param.lambda_optimizer, "mu_arm_obj", 0.9)),
            )
            near_patch = _patch_proximity(dist_surface, dist_track, dist_to_exec) <= 0.03
            if (
                align_info["ok"] and near_patch
                and not _on_opposite_sides(curr_q[7:10], curr_q[:3], best_contact_world)
                and not _floor_slide_away_from_patch(curr_q[7:10], best_contact_world, ground=table_ground)
            ):
                arrived_at_best = True
            dist_to_exec = _verify_distance(dist_to_exec, dist_surface, dist_track, arrived=arrived_at_best)
            verify_now, q_dist = value_tracker.update_verify(
                dist_exec=dist_to_exec, tightness=model_tightness
            )
            value_info.update({
                "occupied_idx": occupied_idx,
                "on_target": bool(arrived_at_best or dist_to_exec <= 0.03 or (align_info["ok"] and near_patch)),
                "arrived_at_best": bool(arrived_at_best),
                "approach_fov": bool(align_info["fov"]),
                "approach_cone": bool(align_info["cone"]),
                "verify": verify_now,
                "q_dist": q_dist,
                "same_patch": bool(same_patch),
                "contact_active": bool(value_tracker.contact_active),
            })

            verify_cost = _rollout_verify_cost(value_info.get("verify", 0.0))
            exec_press = _patch_press_point(p_arm_track_world, p_arm_surface_world)
            opposite = _on_opposite_sides(curr_q[7:10], curr_q[:3], best_contact_world)
            floor_slide = _floor_slide_away_from_patch(curr_q[7:10], best_contact_world, ground=table_ground)
            top_z = _object_top_z_world(
                curr_q[:3], r_obj_to_world,
                getattr(param, "object_aabb_lo", (-0.06, -0.04, -0.04)),
                getattr(param, "object_aabb_hi", (0.06, 0.04, 0.06)),
            )
            keepout = _keepout_radius(
                getattr(param, "object_aabb_lo", (-0.06, -0.04, -0.04)),
                getattr(param, "object_aabb_hi", (0.06, 0.04, 0.06)),
                getattr(param, "object_circumradius", None),
            )
            use_via, via_pos, via_phase = approach_via.update(
                curr_q[7:10], curr_q[:3], p_arm_surface_world, p_arm_track_world,
                top_z, keepout, bool(value_info.get("arrived_at_best", False)), exec_press,
            )
            travel, escape_on = _blend_travel_to_press(
                via_pos, exec_press, model_cost_conf.tightness(), use_via,
                via_phase=via_phase, opposite=opposite,
            )
            mpc_virtual_point = travel
            path_blocked = bool(getattr(approach_via, "blocked", opposite))
            mpc_contact_point = travel if path_blocked else exec_press
            accept_now = bool(value_info.get("accept_p_arm", False))
            verify_chatter = _verify_is_chatter(last_verify_cost, verify_cost)
            last_verify_cost = float(verify_cost)
            last_accept_p_arm = accept_now

            print(
                f"choose_dt={choose_dt:.4f} verify={float(verify_cost):.3f} "
                f"tight={model_cost_conf.tightness():.3f} conf={param.lambda_optimizer.contact_switch_confidence:.3f} "
                f"accept={int(accept_now)} arrived={int(arrived_at_best)} via={via_phase} "
                f"contact={int(if_contact)}"
            )
            env.show_target(mpc_virtual_point)
            env.show_best_contact(best_contact_world)

            c_now_cost = _lambda_pose_cost(
                curr_q[:3], curr_q[3:7], param.target_p_, param.target_q_,
                param.lambda_optimizer.pos_coef, param.lambda_optimizer.ori_coef,
            )
            pred_reduction = None
            if x_plus_opt is not None and _x_plus_is_usable(x_plus_opt, info):
                pred_pos, pred_quat = _predicted_object_pose(curr_q[:7], x_plus_opt)
                c_pred = _lambda_pose_cost(
                    pred_pos, pred_quat, param.target_p_, param.target_q_,
                    param.lambda_optimizer.pos_coef, param.lambda_optimizer.ori_coef,
                )
                pred_reduction = c_now_cost - c_pred

            sol = mpc.plan_once(
                param.target_p_,
                param.target_q_,
                curr_q,
                phi_vec,
                jac_mat,
                verify_cost_param=verify_cost,
                virtual_point=mpc_virtual_point,
                contact_point=mpc_contact_point,
                sol_guess=param.sol_guess_,
            )
            param.sol_guess_ = sol["sol_guess"]
            env.step(np.asarray(sol["action"], dtype=np.float32))
            contact_distance = _isaac_contact_distance(contact, env)
            measured_contact = contact.get_actual_fingertip_contact()
            on_exec_contact = bool(
                measured_contact is not None
                and float(measured_contact["dist"]) <= 0.003
                and float(np.linalg.norm(env.get_policy_state()[7:10] - np.asarray(p_arm_world, dtype=float))) <= 0.03
            )
            rollout_step += 1

            curr_q = env.get_policy_state()
            pos_err_now = float(metrics.comp_pos_error(curr_q[0:3], param.target_p_))
            quat_err_now = float(metrics.comp_quat_error(curr_q[3:7], param.target_q_))
            if c_now_cost is not None:
                c_after = _lambda_pose_cost(
                    curr_q[:3], curr_q[3:7], param.target_p_, param.target_q_,
                    param.lambda_optimizer.pos_coef, param.lambda_optimizer.ori_coef,
                )
                act_reduction = c_now_cost - c_after
                opt = param.lambda_optimizer
                if _should_observe_model_cost(opt.has_delta_span(), getattr(opt, "last_pose_cost_now", None)):
                    pred_delta = pred_reduction
                    if pred_delta is None or not np.isfinite(float(pred_delta)):
                        pred_delta = getattr(opt, "last_best_delta", None)
                    if pred_delta is None or not np.isfinite(float(pred_delta)):
                        finite = np.asarray(getattr(opt, "last_candidate_deltas", []), dtype=np.float64).reshape(-1)
                        finite = finite[np.isfinite(finite)]
                        pred_delta = float(np.max(finite)) if finite.size else None
                    if pred_delta is None:
                        model_cost_conf.observe_unusable_prediction()
                    else:
                        pred_n = opt.normalize_cost_delta(pred_delta)
                        act_n = opt.normalize_cost_delta(act_reduction)
                        if pred_n is None or act_n is None:
                            model_cost_conf.observe_unusable_prediction()
                        else:
                            model_cost_conf.observe(pred_n, act_n)

            if pos_err_now < success_pos_threshold and quat_err_now < success_quat_threshold:
                consecutive_success_time += 1
            else:
                consecutive_success_time = 0

            tip_now = np.asarray(curr_q[7:10], dtype=float)
            r_now = Rotation.from_quat([curr_q[4], curr_q[5], curr_q[6], curr_q[3]]).as_matrix()
            tip_local_now = r_now.T @ (tip_now - curr_q[:3])
            post_physical = bool(np.isfinite(contact_distance) and contact_distance <= 0.003)
            (progress_idx, dwell_active, dwell_dead, occupied_for_log,
             on_exec_patch, _dist_exec_now) = _rollout_dwell_assignment(
                param.lambda_optimizer, tip_local_now, tip_now,
                curr_q[:3], r_now,
                getattr(param.lambda_optimizer, "last_executed_idx", None),
                p_arm_world, post_physical,
                prev_dwell=getattr(param.lambda_optimizer, "_dwell_idx", None),
            )
            if dwell_active and not on_exec_patch:
                dwell_dead = True
            elif dwell_active and on_exec_patch:
                dwell_dead = False
                dwell_active = False
            if (not last_accept_p_arm) and not on_exec_patch and escape_on:
                dwell_dead = False
                dwell_active = False
            if progress_idx is None:
                progress_idx = (
                    getattr(param.lambda_optimizer, "last_global_idx", None)
                    or getattr(param.lambda_optimizer, "last_executed_idx", None)
                )
            if verify_chatter:
                progress_idx = occupied_for_log or progress_idx
                dwell_active = True
                dwell_dead = True
            dest_idx = getattr(param.lambda_optimizer, "last_global_idx", None)
            dest_protected = bool(
                dest_idx is not None and progress_idx is not None
                and _same_contact_patch(param.lambda_optimizer, progress_idx, dest_idx, radius=0.03)
            )
            dwell_active, dwell_dead = _protect_destination_dwell(
                param.lambda_optimizer, progress_idx, dest_idx, dwell_active, dwell_dead
            )
            param.lambda_optimizer.note_contact_progress(
                progress_idx,
                pos_err_now,
                active=dwell_active,
                gamma=float(args.contact_dwell_gamma),
                min_dwell_steps=int(args.contact_dwell_steps),
                improve_eps=0.002,
                dead_increment=dwell_dead,
                merge_radius=0.03,
                block_radius=0.03,
                block_cycles=80,
                time_decay=bool(last_accept_p_arm and not dest_protected),
            )
            if consecutive_success_time > consecutive_success_time_threshold:
                break

        lambda_failures = int(getattr(param.lambda_optimizer, "acados_failure_count", 0))
        mpc_failures = int(getattr(mpc, "acados_failure_count", 0))
        if lambda_failures or mpc_failures or getattr(mpc, "acados_solver_", None) is None:
            print("acados diagnostics:", {
                "lambda_solves": int(getattr(param.lambda_optimizer, "acados_solve_count", 0)),
                "lambda_failures": lambda_failures,
                "mpc_solves": int(getattr(mpc, "acados_solve_count", 0)),
                "mpc_failures": mpc_failures,
                "mpc_init_error": str(getattr(mpc, "_acados_init_error", "")) or None,
            })
        print("trial_summary:", {
            "trial": trial_count,
            "mode": "rollout",
            "success": int(rollout_step < max_rollout_length),
            "steps": rollout_step,
            "final_pos_err": round(pos_err_now, 5) if rollout_step else None,
            "final_quat_err": round(quat_err_now, 5) if rollout_step else None,
            "choose_dt_mean": round(float(np.mean(choose_times)), 5) if choose_times else None,
        })
        success_rate += 1 if rollout_step < max_rollout_length else 0
        env.close()

    print(
        f"Success rate over {trial_num} trials "
        f"(trial ids {trial_start} to {trial_stop - 1}): "
        f"{success_rate}/{trial_num} = {success_rate / trial_num:.2%}"
    )


if __name__ == "__main__":
    main()
