import argparse
import json
import os
import re
import sys

import numpy as np
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

from examples.mpc.fingertips.test.test_0902 import (
    ContactValueTracker,
    ModelCostConfidence,
    SmoothedApproachVia,
    add_rollout_via_args,
)
from examples.mpc.franka.ik2.contact_frames import (
    FRANKA_QD_NULLSPACE,
    clip_mpc_action as _clip_mpc_action,
    clip_via_target as _clip_via_target,
    franka_nullspace_posture_torque as _franka_nullspace_posture_torque,
    gravity_accel_with_support,
    isaac_task_force as _isaac_task_force,
    mpc_ball_trajectory as _mpc_ball_trajectory,
    near_press_force_mode as _near_press_force_mode,
    NEAR_PRESS_SWITCH as CONTACT_NEAR_PRESS,
    osc_action_task_force as _osc_action_task_force,
    planar_support_jacobians,
    slew_mpc_action as _slew_mpc_action,
    press_normal_outward as _press_normal_outward,
    project_along_action as _project_along_action,
    remaining_along_action as _remaining_along_action,
)
from examples.mpc.franka.ik2.isaac_bus import joint_hold_target
from examples.mpc.franka.ik2.params import _box_inertia_diag, build_lambda_optimizer
from examples.mpc.franka.ik2.test_mpc_isaac import (
    AIR_VIA_STEP,
    CONTACT_FORCE_LIMIT,
    ContactIsaacCartesian,
    FREE_SPACE_FORCE_LIMIT,
    ISAAC_ACTION_SLEW,
    ISAAC_ORBIT_EXTRA,
    ISAAC_REF_HORIZON,
    ISAAC_TASK_KD,
    ISAAC_TASK_KP,
    ISAAC_VIA_MAX_LEAD,
    ISAAC_VIA_MAX_STEP,
    ISAAC_VIA_SMOOTH_RATE,
    POLICY_INTERVAL,
    ROLLOUT_FINGERTIP_FRICTION,
    _franka_fk_T_np,
    _franka_jacobian_pos_np,
    _print_rollout_step,
    adapt_param_for_cartesian_solver as _adapt_ik2_cartesian_solver,
    policy_control_substeps,
)
from examples.mpc.franka.ik2.test_mppi_isaac import (
    IsaacFrankaSimulator,
    _apply_plan_result,
    _dwell_payload,
    _extract_quat_xyzw,
    _extract_vec3,
    _parse_bool_arg,
    _plan_payload,
    _pred_reduction_from_policy,
)
from examples.mpc.franka.test_tilted_push.params import ExplicitMPCParams
from planning.mpc_explicit import MPCExplicitIsaac, handle_mpc_request
from planning.screenshot import create_isaacgym_mp4_recorder, create_isaacgym_svg_screenshot_recorder
from utils import metrics, rotations

DYWA_SIM_DT = 0.002
DYWA_SIM_SUBSTEPS = 1
DYWA_PHYSX_SOLVER_TYPE = 1
DYWA_PHYSX_POSITION_ITERATIONS = 8
DYWA_PHYSX_VELOCITY_ITERATIONS = 1
DYWA_PHYSX_CONTACT_OFFSET = 0.001
DYWA_PHYSX_REST_OFFSET = 0.0
DYWA_PHYSX_FRICTION_OFFSET_THRESHOLD = 0.001
DYWA_PHYSX_FRICTION_CORRELATION_DISTANCE = 0.0005
DYWA_PHYSX_MAX_DEPENETRATION_VELOCITY = 0.25
DYWA_TABLE_FRICTION_RANGE = (0.3, 0.8)
DYWA_RAMP_FRICTION_SCALE = 0.4
DYWA_OBJECT_FRICTION_RANGE = (0.2, 1.0)
DYWA_OBJECT_MASS_RANGE = (0.1, 0.5)
ROLLOUT_TABLE_FRICTION = 0.5
ROLLOUT_OBJECT_FRICTION = 0.9
NEAR_PRESS_SWITCH = CONTACT_NEAR_PRESS

DEFAULT_CARTESIAN_STIFFNESS = np.array([2000.0, 2000.0, 2000.0, 50.0, 50.0, 50.0], dtype=np.float32)
DEFAULT_EFFORT_JOINT_DAMPING = 0.0
SVG_SCREENSHOT_CAMERA_POSITION = np.array([0.7, 0.00, 0.63], dtype=np.float32)
SVG_SCREENSHOT_CAMERA_TARGET = np.array([0.1, 0.00, 0.32], dtype=np.float32)
FORCE_VIS_HIDDEN_POSITION = np.array([0.0, 0.0, -10.0], dtype=np.float32)
FORCE_CYLINDER_BASE_LENGTH = 0.08
FORCE_CYLINDER_RADIUS = 0.004
FORCE_VECTOR_LENGTH_SCALE = 0.05
FORCE_VECTOR_MIN_LENGTH = 0.025
FORCE_VECTOR_MAX_LENGTH = 0.12
FORCE_ARROW_HEAD_LENGTH = 0.018
FORCE_ARROW_HEAD_WIDTH = 0.012
ATTACHMENT_RADIUS = 0.005
FINGERTIP_RADIUS = 0.006
ATTRACT_POINT_MAX_INWARD_NORMAL_WORLD_Z = 0.2
DEFAULT_RECORD_OUTPUT_DIR = "/home/lab423/scsp/scsp-robot/outputs/videos_tilted_pushing"
DEFAULT_AXIS_POSE_WXYZ = np.array(
    [0.42849, 0.11289, 0.38614, 0.62768, -0.10893, -0.10086, 0.76419],
    dtype=np.float32,
)
TILTED_RAMP_ANGLE_DEG = 15.0
TILTED_RAMP_ROLL_RAD = np.deg2rad(-TILTED_RAMP_ANGLE_DEG)
TILTED_RAMP_SIZE = np.array([0.60, 0.60, 0.03], dtype=np.float32)
TILTED_RAMP_CENTER_X = 0.60
TILTED_RAMP_CENTER_Y = 0.0
TILTED_RAMP_BASE_CLEARANCE = 0.002
TILTED_RAMP_EDGE_MARGIN = 0.04
TILTED_RAMP_INIT_CLEARANCE = 0.05
TILTED_RAMP_TARGET_CLEARANCE = 0.03
TILTED_RAMP_HOLD_FRICTION_MARGIN = 0.05
TILTED_RAMP_SETTLE_STEPS = 80
TILTED_RAMP_SETTLE_CYCLES = 2
DEFAULT_TARGET_RAMP_LOCAL_XY = np.array([-0.18, 0.0], dtype=np.float32)
DEFAULT_TARGET_RAMP_LOCAL_YAW_DEG = 0.0
DEFAULT_TARGET_LOCAL_Z_ROTATION_OFFSET_RAD = np.pi / 3.0
DEFAULT_INIT_RAMP_LOCAL_XY = np.array([-0.24, -0.14], dtype=np.float32)
DEFAULT_INIT_RAMP_LOCAL_XY_JITTER = np.array([0.02, 0.02], dtype=np.float32)
DEFAULT_SPHERE_MESH_SCALE = np.array([0.08, 0.08, 0.08], dtype=np.float32)
DEFAULT_TARGET_WORLD_OFFSET_FROM_INIT = np.array([0.0, -0.10, 0.10], dtype=np.float32)
DEFAULT_POSITION_ONLY_GOAL = True
DEFAULT_SET_TO_GOAL_POSE = False
DEFAULT_SET_TO_GOAL_POSE_XYZ_NOISE = np.array([0.01, 0.01, 0.005], dtype=np.float32)
DEFAULT_SET_TO_GOAL_POSE_YAW_NOISE_DEG = 10.0


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
# DEFAULT_ELEPHANT_TRIAL_REPLAY = {
#     "obj": "elephant",
#     "attract_coef": 0.5,
#     "reject_coef": 0.001,
#     "contact_coef": 0.5,
#     "contact_cost_param": 0.0,
#     "model_param": 7.0,
#     "reject_dis": 0.01,
#     "attract_point_comp": 0.05,
#     "ground_height_threshold": 0.012,
#     "sample_num": 70,
#     "pos_coef": 1,
#     "ori_coef": 0.005,
#     "low_err_coef": 0.75,
#     "upper_err_coef": 0.95,
#     "sim_device": "cuda:0",
#     "graphics_device_id": 0,
#     "cartesian_step": 0.05,
#     "cartesian_joint_stiffness": 100.0,
#     "cartesian_dls_lambda": 0.0001,
#     "osc_pos_stiffness": 2000.0,
#     "osc_ori_stiffness": 400.0,
#     "trial_start": 3,
#     "trial_count": 1,
# }


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


def _normalize_quaternion_wxyz_np(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return quat_wxyz / max(np.linalg.norm(quat_wxyz), 1e-12)


def _format_array_for_print(array, precision=5):
    return np.array2string(
        np.asarray(array, dtype=np.float32),
        precision=precision,
        suppress_small=False,
        floatmode="fixed",
    )


def _parse_cli_array_arg(raw_value, expected_dim, arg_name):
    if raw_value is None:
        return None
    if not isinstance(raw_value, (list, tuple)):
        raw_value = [raw_value]

    joined = " ".join(str(v) for v in raw_value).strip()
    if "[" in joined and "]" in joined:
        joined = joined[joined.find("[") + 1 : joined.rfind("]")]
    cleaned = joined.replace(",", " ")
    values = np.fromstring(cleaned, sep=" ", dtype=np.float64)
    if values.size != expected_dim:
        raise ValueError(f"{arg_name} must contain exactly {expected_dim} numeric values, got {values.size}: {raw_value}")
    return values.astype(np.float32)


def _parse_pose_wxyz_arg(raw_value, arg_name):
    pose = _parse_cli_array_arg(raw_value, expected_dim=7, arg_name=arg_name)
    quat_wxyz = _normalize_quaternion_wxyz_np(pose[3:7]).astype(np.float32)
    return np.hstack([pose[:3].astype(np.float32), quat_wxyz]).astype(np.float32)


def _quat_wxyz_to_xyzw_np(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)


def _quat_xyzw_to_wxyz_np(quat_xyzw):
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def _compose_quat_wxyz(left_quat_wxyz, right_quat_wxyz):
    left_xyzw = _quat_wxyz_to_xyzw_np(left_quat_wxyz)
    right_xyzw = _quat_wxyz_to_xyzw_np(right_quat_wxyz)
    composed_xyzw = (Rotation.from_quat(left_xyzw) * Rotation.from_quat(right_xyzw)).as_quat()
    return _normalize_quaternion_wxyz_np(_quat_xyzw_to_wxyz_np(composed_xyzw)).astype(np.float32)


def _quat_from_world_z_rotation_wxyz(yaw_rad):
    quat_xyzw = Rotation.from_euler("z", float(yaw_rad), degrees=False).as_quat().astype(np.float32)
    return _quat_xyzw_to_wxyz_np(quat_xyzw)


def _quat_from_local_z_rotation_wxyz(yaw_rad):
    return rotations.angle_dir_to_quat(float(yaw_rad), np.array([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)


def _compute_tilted_ramp_config(table_height):
    ramp_size = np.asarray(TILTED_RAMP_SIZE, dtype=np.float32).reshape(3)
    half_vertical_extent = 0.5 * (
        abs(float(ramp_size[1]) * np.sin(float(TILTED_RAMP_ROLL_RAD)))
        + abs(float(ramp_size[2]) * np.cos(float(TILTED_RAMP_ROLL_RAD)))
    )
    ramp_center = np.array(
        [
            TILTED_RAMP_CENTER_X,
            TILTED_RAMP_CENTER_Y,
            float(table_height) + half_vertical_extent + TILTED_RAMP_BASE_CLEARANCE,
        ],
        dtype=np.float32,
    )
    ramp_quat_xyzw = Rotation.from_euler("x", float(TILTED_RAMP_ROLL_RAD), degrees=False).as_quat().astype(np.float32)
    ramp_rotation = Rotation.from_quat(ramp_quat_xyzw)
    ramp_top_center = ramp_center + ramp_rotation.apply(
        np.array([0.0, 0.0, 0.5 * float(ramp_size[2])], dtype=np.float32)
    )
    ramp_top_normal = ramp_rotation.apply(np.array([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)
    ramp_top_normal /= max(np.linalg.norm(ramp_top_normal), 1e-8)
    x_limit = 0.5 * float(ramp_size[0])
    y_limit = 0.5 * float(ramp_size[1])
    top_corners_local = np.array(
        [
            [-x_limit, -y_limit, 0.0],
            [-x_limit, y_limit, 0.0],
            [x_limit, -y_limit, 0.0],
            [x_limit, y_limit, 0.0],
        ],
        dtype=np.float32,
    )
    top_corners_world = ramp_top_center[None, :] + ramp_rotation.apply(top_corners_local)
    return {
        "size": ramp_size,
        "center": ramp_center.astype(np.float32),
        "quat_xyzw": ramp_quat_xyzw.astype(np.float32),
        "quat_wxyz": _quat_xyzw_to_wxyz_np(ramp_quat_xyzw).astype(np.float32),
        "surface_center": ramp_top_center.astype(np.float32),
        "surface_normal": ramp_top_normal.astype(np.float32),
        "surface_min_height": float(np.min(top_corners_world[:, 2])),
        "rotation": ramp_rotation,
    }


def _support_plane(param):
    table_z = float(getattr(param, "table_height", 0.35))
    point = np.asarray(
        getattr(param, "support_surface_point_", np.array([0.0, 0.0, table_z])),
        dtype=np.float64,
    ).reshape(3)
    normal = np.asarray(
        getattr(param, "support_surface_normal_", np.array([0.0, 0.0, 1.0])),
        dtype=np.float64,
    ).reshape(3)
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-8:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        normal = normal / nrm
    return point, normal


def _support_surface_z(param, xy):
    point, normal = _support_plane(param)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1)
    if abs(float(normal[2])) < 1e-8:
        return float(point[2])
    return float((np.dot(normal, point) - normal[0] * xy[0] - normal[1] * xy[1]) / normal[2])


def _support_signed_distance(param, pos):
    point, normal = _support_plane(param)
    return float(np.dot(np.asarray(pos, dtype=np.float64).reshape(3) - point, normal))


def _support_air_floor(param, xy, margin=0.012):
    return _support_surface_z(param, xy) + float(margin)


def _clip_tilted_ramp_local_xy(ramp_config, local_xy):
    local_xy = np.asarray(local_xy, dtype=np.float32).reshape(2)
    ramp_size = np.asarray(ramp_config["size"], dtype=np.float32).reshape(3)
    x_limit = max(0.0, 0.5 * float(ramp_size[0]) - TILTED_RAMP_EDGE_MARGIN)
    y_limit = max(0.0, 0.5 * float(ramp_size[1]) - TILTED_RAMP_EDGE_MARGIN)
    local_xy[0] = np.clip(local_xy[0], -x_limit, x_limit)
    local_xy[1] = np.clip(local_xy[1], -y_limit, y_limit)
    return local_xy.astype(np.float32)


def _place_pose_on_tilted_ramp_local(ramp_config, local_xy, clearance_along_normal):
    ramp_rotation = ramp_config["rotation"]
    surface_center = np.asarray(ramp_config["surface_center"], dtype=np.float32).reshape(3)
    surface_normal = np.asarray(ramp_config["surface_normal"], dtype=np.float32).reshape(3)
    clipped_local_xy = _clip_tilted_ramp_local_xy(ramp_config, local_xy)
    local_surface = np.array([clipped_local_xy[0], clipped_local_xy[1], 0.0], dtype=np.float32)
    surface_point = surface_center + ramp_rotation.apply(local_surface)
    pose_position = surface_point + float(clearance_along_normal) * surface_normal
    return pose_position.astype(np.float32), surface_point.astype(np.float32), clipped_local_xy.astype(np.float32)


def _place_pose_on_tilted_ramp(ramp_config, desired_xy, clearance_along_normal):
    desired_xy = np.asarray(desired_xy, dtype=np.float32).reshape(2)
    ramp_rotation = ramp_config["rotation"]
    surface_center = np.asarray(ramp_config["surface_center"], dtype=np.float32).reshape(3)
    surface_normal = np.asarray(ramp_config["surface_normal"], dtype=np.float32).reshape(3)

    plane_z = float(surface_center[2]) - (
        float(surface_normal[0]) * float(desired_xy[0] - surface_center[0])
        + float(surface_normal[1]) * float(desired_xy[1] - surface_center[1])
    ) / max(float(surface_normal[2]), 1e-8)
    guess_world = np.array([desired_xy[0], desired_xy[1], plane_z], dtype=np.float32)
    local_surface = ramp_rotation.inv().apply(guess_world - surface_center).astype(np.float32)
    pose_position, surface_point, clipped_local_xy = _place_pose_on_tilted_ramp_local(
        ramp_config,
        local_surface[:2],
        clearance_along_normal,
    )
    return pose_position.astype(np.float32), surface_point.astype(np.float32), clipped_local_xy.astype(np.float32)


def _project_world_offset_to_tilted_ramp_local_xy(ramp_config, world_offset):
    world_offset = np.asarray(world_offset, dtype=np.float32).reshape(3)
    surface_normal = np.asarray(ramp_config["surface_normal"], dtype=np.float32).reshape(3)
    ramp_rotation = ramp_config["rotation"]
    tangent_world_offset = world_offset - float(np.dot(world_offset, surface_normal)) * surface_normal
    tangent_local_offset = ramp_rotation.inv().apply(tangent_world_offset).astype(np.float32)
    return tangent_local_offset[:2].astype(np.float32), tangent_world_offset.astype(np.float32)


def _build_goal_pose_on_tilted_ramp(ramp_config, local_xy, local_yaw_deg, clearance_along_normal):
    goal_pos_world, goal_surface_point, clipped_local_xy = _place_pose_on_tilted_ramp_local(
        ramp_config,
        local_xy,
        clearance_along_normal,
    )
    goal_local_quat_wxyz = rotations.rpy_to_quaternion(
        np.array([np.deg2rad(float(local_yaw_deg)), 0.0, 0.0], dtype=np.float32)
    ).astype(np.float32)
    goal_world_quat_wxyz = _compose_quat_wxyz(ramp_config["quat_wxyz"], goal_local_quat_wxyz)
    return {
        "world_pos": goal_pos_world.astype(np.float32),
        "world_quat_wxyz": goal_world_quat_wxyz.astype(np.float32),
        "surface_point": goal_surface_point.astype(np.float32),
        "local_xy": clipped_local_xy.astype(np.float32),
        "local_quat_wxyz": goal_local_quat_wxyz.astype(np.float32),
    }


def _apply_tilted_ramp_scene_to_param(
    param,
    use_default_target_position,
    use_default_target_orientation,
    target_ramp_local_xy,
    target_ramp_local_yaw_deg,
    target_world_offset,
    position_only_goal,
    target_local_z_rotation_offset_rad=DEFAULT_TARGET_LOCAL_Z_ROTATION_OFFSET_RAD,
):
    ramp_config = _compute_tilted_ramp_config(float(param.table_height))
    object_surface_clearance = float(getattr(param, "object_surface_clearance_", TILTED_RAMP_INIT_CLEARANCE))
    target_surface_clearance = float(getattr(param, "target_surface_clearance_", TILTED_RAMP_TARGET_CLEARANCE))
    param.tilted_ramp_angle_deg_ = float(TILTED_RAMP_ANGLE_DEG)
    param.tilted_ramp_size_ = np.asarray(ramp_config["size"], dtype=np.float32).copy()
    param.tilted_ramp_center_ = np.asarray(ramp_config["center"], dtype=np.float32).copy()
    param.tilted_ramp_quat_xyzw_ = np.asarray(ramp_config["quat_xyzw"], dtype=np.float32).copy()
    param.tilted_ramp_quat_wxyz_ = np.asarray(ramp_config["quat_wxyz"], dtype=np.float32).copy()
    param.support_surface_point_ = np.asarray(ramp_config["surface_center"], dtype=np.float32).copy()
    param.support_surface_normal_ = np.asarray(ramp_config["surface_normal"], dtype=np.float32).copy()
    param.support_surface_min_height_ = float(ramp_config["surface_min_height"])
    param.project_mpc_action_to_support_tangent_ = False
    param.mpc_action_support_normal_ = np.asarray(ramp_config["surface_normal"], dtype=np.float32).copy()
    gravity = np.asarray(getattr(param, "gravity_", np.zeros(6)), dtype=np.float64).reshape(-1)
    if gravity.size < 6:
        gravity = np.hstack([gravity[:3], np.zeros(max(0, 6 - int(gravity.size)), dtype=np.float64)])
    gravity[:3] = gravity_accel_with_support(gravity[:3], param.support_surface_normal_)
    param.gravity_ = gravity.astype(np.float64)

    init_ramp_local_xy_seed = DEFAULT_INIT_RAMP_LOCAL_XY.astype(np.float32).copy()
    init_ramp_local_xy_jitter = DEFAULT_INIT_RAMP_LOCAL_XY_JITTER.astype(np.float32).copy()
    random_generator = getattr(param, "random_generator_", None)
    if random_generator is not None:
        init_ramp_local_xy_seed = init_ramp_local_xy_seed + random_generator.uniform(
            low=-init_ramp_local_xy_jitter,
            high=init_ramp_local_xy_jitter,
        ).astype(np.float32)
    init_pos_world, init_surface_point, init_ramp_local_xy = _place_pose_on_tilted_ramp_local(
        ramp_config,
        init_ramp_local_xy_seed,
        object_surface_clearance,
    )
    init_quat_wxyz = _compose_quat_wxyz(ramp_config["quat_wxyz"], param.init_obj_qpos_[3:7])
    param.init_obj_qpos_ = np.hstack([init_pos_world, init_quat_wxyz]).astype(np.float32)
    param.init_xy_rand_ = init_surface_point[:2].astype(np.float32).copy()
    param.init_ramp_local_xy_seed_ = np.asarray(init_ramp_local_xy_seed, dtype=np.float32).copy()
    param.init_ramp_local_xy_ = np.asarray(init_ramp_local_xy, dtype=np.float32).copy()
    param.init_obj_quat_rand_ = init_quat_wxyz.astype(np.float32).copy()
    param.target_local_z_rotation_offset_rad_ = float(target_local_z_rotation_offset_rad)
    param.position_only_goal_ = bool(position_only_goal)
    param.target_world_offset_ = np.asarray(target_world_offset, dtype=np.float32).reshape(3).copy()
    target_ramp_local_offset, projected_target_world_offset = _project_world_offset_to_tilted_ramp_local_xy(
        ramp_config,
        param.target_world_offset_,
    )
    param.target_world_offset_projected_ = np.asarray(projected_target_world_offset, dtype=np.float32).copy()
    param.target_ramp_local_offset_ = np.asarray(target_ramp_local_offset, dtype=np.float32).copy()
    resolved_goal_local_xy = (
        param.init_ramp_local_xy_ + param.target_ramp_local_offset_
        if param.position_only_goal_
        else np.asarray(target_ramp_local_xy, dtype=np.float32).reshape(2)
    )
    goal_pose_on_ramp = _build_goal_pose_on_tilted_ramp(
        ramp_config,
        resolved_goal_local_xy,
        target_ramp_local_yaw_deg,
        target_surface_clearance,
    )
    param.target_ramp_local_xy_ = np.asarray(goal_pose_on_ramp["local_xy"], dtype=np.float32).copy()
    param.target_ramp_local_yaw_deg_ = float(target_ramp_local_yaw_deg)
    param.target_ramp_local_quat_wxyz_ = np.asarray(goal_pose_on_ramp["local_quat_wxyz"], dtype=np.float32).copy()
    param.target_surface_point_ = np.asarray(goal_pose_on_ramp["surface_point"], dtype=np.float32).copy()

    if use_default_target_position:
        param.target_p_ = np.asarray(goal_pose_on_ramp["world_pos"], dtype=np.float32).copy()
    if use_default_target_orientation:
        local_z_offset_quat_wxyz = _quat_from_local_z_rotation_wxyz(param.target_local_z_rotation_offset_rad_)
        param.target_q_ = _compose_quat_wxyz(init_quat_wxyz, local_z_offset_quat_wxyz).astype(np.float32)
        param.target_ramp_local_yaw_deg_ = float(
            np.rad2deg(param.target_local_z_rotation_offset_rad_)
        )
        param.target_ramp_local_quat_wxyz_ = _compose_quat_wxyz(
            goal_pose_on_ramp["local_quat_wxyz"],
            local_z_offset_quat_wxyz,
        ).astype(np.float32)

    if bool(getattr(param, "set_to_goal_pose_", False)):
        random_generator = getattr(param, "random_generator_", None)
        if random_generator is None:
            random_generator = np.random.default_rng()
        goal_pos_world = np.asarray(param.target_p_, dtype=np.float32).reshape(3)
        goal_quat_wxyz = _normalize_quaternion_wxyz_np(np.asarray(param.target_q_, dtype=np.float32).reshape(4))
        xyz_noise_bound = np.asarray(
            getattr(param, "set_to_goal_pose_xyz_noise_", DEFAULT_SET_TO_GOAL_POSE_XYZ_NOISE),
            dtype=np.float32,
        ).reshape(3)
        xyz_noise = random_generator.uniform(low=-xyz_noise_bound, high=xyz_noise_bound).astype(np.float32)
        yaw_noise_deg = float(getattr(param, "set_to_goal_pose_yaw_noise_deg_", DEFAULT_SET_TO_GOAL_POSE_YAW_NOISE_DEG))
        yaw_noise_rad = np.deg2rad(float(random_generator.uniform(low=-yaw_noise_deg, high=yaw_noise_deg)))
        yaw_noise_quat_wxyz = _quat_from_world_z_rotation_wxyz(yaw_noise_rad)

        init_pos_world = goal_pos_world + xyz_noise
        init_quat_wxyz = _compose_quat_wxyz(yaw_noise_quat_wxyz, goal_quat_wxyz)
        param.init_obj_qpos_ = np.hstack([init_pos_world, init_quat_wxyz]).astype(np.float32)
        param.init_xy_rand_ = init_pos_world[:2].astype(np.float32).copy()
        param.init_obj_quat_rand_ = init_quat_wxyz.astype(np.float32).copy()
        param.init_from_goal_pose_xyz_noise_ = xyz_noise.astype(np.float32).copy()
        param.init_from_goal_pose_yaw_noise_deg_ = float(np.rad2deg(yaw_noise_rad))
        if use_default_target_orientation:
            local_z_offset_quat_wxyz = _quat_from_local_z_rotation_wxyz(param.target_local_z_rotation_offset_rad_)
            param.target_q_ = _compose_quat_wxyz(init_quat_wxyz, local_z_offset_quat_wxyz).astype(np.float32)

    return param


def _pose_wxyz_to_xyzw_components(pose_wxyz):
    pose_wxyz = np.asarray(pose_wxyz, dtype=np.float32).reshape(7)
    return pose_wxyz[:3].copy(), _quat_wxyz_to_xyzw_np(pose_wxyz[3:7])


def _print_object_pose(trial_id, step_id, obj_pos, obj_quat_wxyz):
    print(
        f"[trial {trial_id:03d} step {step_id:04d}] "
        f"object xyz={_format_array_for_print(obj_pos)} "
        f"quat_wxyz={_format_array_for_print(obj_quat_wxyz)}"
    )


def _append_object_pose_sample(traj_qpos, qpos, atol=1e-9):
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if traj_qpos and np.allclose(np.asarray(traj_qpos[-1], dtype=np.float32), qpos, atol=atol, rtol=0.0):
        return
    traj_qpos.append(qpos.copy())


def _get_next_available_numbered_path(output_path):
    output_path = os.path.abspath(output_path)
    if not os.path.exists(output_path):
        return output_path

    output_dir = os.path.dirname(output_path)
    base_name = os.path.basename(output_path)
    stem, ext = os.path.splitext(base_name)
    match = re.match(r"^(.*?)(\d+)$", stem)

    if match:
        prefix, number_text = match.groups()
        next_index = int(number_text)
        width = len(number_text)
    else:
        prefix = f"{stem}_"
        next_index = 0
        width = 1

    while True:
        next_index += 1
        candidate_name = f"{prefix}{next_index:0{width}d}{ext}"
        candidate_path = os.path.join(output_dir, candidate_name)
        if not os.path.exists(candidate_path):
            return candidate_path


def _allocate_trial_record_dir(base_output_dir, obj_name, width=3):
    if not base_output_dir:
        return None

    base_output_dir = os.path.abspath(base_output_dir)
    safe_obj_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(obj_name).strip()).strip("._-")
    if not safe_obj_name:
        safe_obj_name = "object"

    obj_output_dir = os.path.join(base_output_dir, safe_obj_name)
    os.makedirs(obj_output_dir, exist_ok=True)

    record_index = 0
    while True:
        record_dir = os.path.join(obj_output_dir, f"{record_index:0{width}d}")
        if not os.path.exists(record_dir):
            os.makedirs(record_dir, exist_ok=False)
            return record_dir
        record_index += 1


def _build_traj_payload(
    traj_qpos,
    trial_id=None,
    success=None,
    interrupted=False,
    target_pose_wxyz=None,
):
    pose_records = []
    for step_id, qpos in enumerate(traj_qpos):
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        pose_records.append(
            {
                "step": int(step_id),
                "xyz": [float(v) for v in qpos[:3]],
                "quat_wxyz": [float(v) for v in qpos[3:7]],
            }
        )

    payload = {
        "trial_id": None if trial_id is None else int(trial_id),
        "success": None if success is None else bool(success),
        "interrupted": bool(interrupted),
        "num_samples": int(len(pose_records)),
        "object_pose_wxyz": pose_records,
    }
    if target_pose_wxyz is not None:
        target_pose_wxyz = np.asarray(target_pose_wxyz, dtype=np.float32).reshape(7)
        payload["target_pose_wxyz"] = {
            "xyz": [float(v) for v in target_pose_wxyz[:3]],
            "quat_wxyz": [float(v) for v in target_pose_wxyz[3:7]],
        }
    return payload


def record_traj(
    traj_qpos,
    output_path,
    trial_id=None,
    success=None,
    interrupted=False,
    target_pose_wxyz=None,
    allow_overwrite=False,
):
    if output_path is None:
        return None

    output_path = os.path.abspath(output_path)
    if not allow_overwrite:
        output_path = _get_next_available_numbered_path(output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    payload = _build_traj_payload(
        traj_qpos,
        trial_id=trial_id,
        success=success,
        interrupted=interrupted,
        target_pose_wxyz=target_pose_wxyz,
    )

    tmp_output_path = f"{output_path}.tmp"
    with open(tmp_output_path, "w", encoding="ascii") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    os.replace(tmp_output_path, output_path)

    return output_path


def _quat_wxyz_to_z_axis_np(quat_wxyz):
    w, x, y, z = _normalize_quaternion_wxyz_np(quat_wxyz)
    return np.array(
        [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dtype=np.float64,
    )


def _filter_point_indices_to_upper_third(optimizer, pos, rotation_matrix, candidate_idx, sampling_region=-0.5):
    candidate_idx = np.asarray(candidate_idx, dtype=np.int32).reshape(-1)
    if candidate_idx.size == 0:
        return candidate_idx

    sampling_region = float(sampling_region)
    if abs(sampling_region) <= 1e-6:
        return candidate_idx

    sample_points = np.asarray(optimizer.sample_point, dtype=np.float32)
    centers_world_all = (np.asarray(rotation_matrix, dtype=np.float32) @ sample_points.T).T + np.asarray(
        pos, dtype=np.float32
    )
    z_coords_all = centers_world_all[:, 2]
    z_min = float(np.min(z_coords_all))
    z_max = float(np.max(z_coords_all))
    z_span = z_max - z_min
    if z_span <= 1e-6:
        return candidate_idx

    region_fraction = min(abs(sampling_region), 1.0)
    candidate_z = centers_world_all[candidate_idx, 2]
    if sampling_region > 0.0:
        z_cutoff = z_max - region_fraction * z_span
        region_mask = candidate_z >= z_cutoff
    else:
        z_cutoff = z_min + region_fraction * z_span
        region_mask = candidate_z <= z_cutoff

    region_idx = candidate_idx[region_mask]
    return region_idx if region_idx.size > 0 else candidate_idx


def _filter_point_indices_to_local_positive_z(optimizer, candidate_idx):
    candidate_idx = np.asarray(candidate_idx, dtype=np.int32).reshape(-1)
    if candidate_idx.size == 0:
        return candidate_idx

    sample_points = np.asarray(optimizer.sample_point, dtype=np.float32)
    keep_mask = sample_points[candidate_idx, 2] > 0.0
    filtered_idx = candidate_idx[keep_mask]
    return filtered_idx if filtered_idx.size > 0 else candidate_idx


def _filter_point_indices_to_local_upper_half(optimizer, candidate_idx):
    candidate_idx = np.asarray(candidate_idx, dtype=np.int32).reshape(-1)
    if candidate_idx.size == 0:
        return candidate_idx

    sample_points = np.asarray(optimizer.sample_point, dtype=np.float32)
    local_z_all = sample_points[:, 2]
    z_min = float(np.min(local_z_all))
    z_max = float(np.max(local_z_all))
    z_span = z_max - z_min
    if z_span <= 1e-6:
        return candidate_idx

    z_cutoff = z_min + 0.5 * z_span
    keep_mask = sample_points[candidate_idx, 2] >= z_cutoff
    filtered_idx = candidate_idx[keep_mask]
    return filtered_idx if filtered_idx.size > 0 else candidate_idx


def _filter_point_indices_to_world_right_half(optimizer, pos, rotation_matrix, candidate_idx):
    candidate_idx = np.asarray(candidate_idx, dtype=np.int32).reshape(-1)
    if candidate_idx.size == 0:
        return candidate_idx

    sample_points = np.asarray(optimizer.sample_point, dtype=np.float32)
    centers_world_all = (np.asarray(rotation_matrix, dtype=np.float32) @ sample_points.T).T + np.asarray(
        pos, dtype=np.float32
    ).reshape(3)
    object_center_x_world = float(np.asarray(pos, dtype=np.float32).reshape(3)[1])
    keep_mask = centers_world_all[candidate_idx, 1] > object_center_x_world
    filtered_idx = candidate_idx[keep_mask]
    return filtered_idx if filtered_idx.size > 0 else candidate_idx


def _filter_point_indices_to_side_or_top(optimizer, rotation_matrix, candidate_idx):
    candidate_idx = np.asarray(candidate_idx, dtype=np.int32).reshape(-1)
    if candidate_idx.size == 0:
        return candidate_idx

    # optimizer.normal stores inward-facing local normals, so large positive
    # world-z means the surface is the object's underside. Reject those points
    # to avoid planning an "under the object / through the table" approach.
    normals_local = np.asarray(optimizer.normal, dtype=np.float32)
    normals_world = (np.asarray(rotation_matrix, dtype=np.float32) @ normals_local[candidate_idx].T).T
    keep_mask = normals_world[:, 2] <= float(ATTRACT_POINT_MAX_INWARD_NORMAL_WORLD_Z)
    filtered_idx = candidate_idx[keep_mask]
    return filtered_idx if filtered_idx.size > 0 else candidate_idx


def _normalize_np(vec, fallback=None):
    vec = np.asarray(vec, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        if fallback is None:
            return None
        return np.asarray(fallback, dtype=np.float32).reshape(3)
    return (vec / norm).astype(np.float32)


def _project_vector_to_support_tangent(vec, support_surface_normal):
    vec = np.asarray(vec, dtype=np.float32).reshape(3)
    support_surface_normal = _normalize_np(support_surface_normal, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    return (vec - float(np.dot(vec, support_surface_normal)) * support_surface_normal).astype(np.float32)


def _support_tangent_distance(point_a, point_b, support_surface_normal):
    tangent_delta = _project_vector_to_support_tangent(
        np.asarray(point_a, dtype=np.float32).reshape(3) - np.asarray(point_b, dtype=np.float32).reshape(3),
        support_surface_normal,
    )
    return float(np.linalg.norm(tangent_delta))


def _compute_safe_attract_point_world(
    contact_point_local,
    normal_local,
    pos_world,
    rotation_matrix,
    attract_point_comp,
    support_surface_normal=None,
):
    contact_point_world = (
        np.asarray(rotation_matrix, dtype=np.float32) @ np.asarray(contact_point_local, dtype=np.float32).reshape(3)
    ) + np.asarray(pos_world, dtype=np.float32).reshape(3)
    offset_world = -float(attract_point_comp) * (
        np.asarray(rotation_matrix, dtype=np.float32) @ np.asarray(normal_local, dtype=np.float32).reshape(3)
    )
    offset_world = np.asarray(offset_world, dtype=np.float32).reshape(3)
    support_surface_normal = _normalize_np(
        np.array([0.0, 0.0, 1.0], dtype=np.float32) if support_surface_normal is None else support_surface_normal,
        fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    inward_support_component = float(np.dot(offset_world, support_surface_normal))
    if inward_support_component < 0.0:
        offset_world = offset_world - inward_support_component * support_surface_normal
    attract_point_world = contact_point_world + offset_world
    attract_support_delta = float(np.dot(attract_point_world - contact_point_world, support_surface_normal))
    if attract_support_delta < 0.0:
        attract_point_world = attract_point_world - attract_support_delta * support_surface_normal
    return attract_point_world.astype(np.float32)


def _quat_xyzw_from_z_axis(direction):
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    direction = direction / norm
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if dot < -1.0 + 1e-8:
        return Rotation.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0], dtype=np.float64)).as_quat().astype(np.float32)

    axis = np.cross(z_axis, direction)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)
    return Rotation.from_rotvec(axis * angle).as_quat().astype(np.float32)


def _z_axis_alignment_error_wxyz(curr_quat_wxyz, target_quat_wxyz):
    curr_z_axis = _quat_wxyz_to_z_axis_np(curr_quat_wxyz)
    target_z_axis = _quat_wxyz_to_z_axis_np(target_quat_wxyz)
    axis_alignment = float(np.clip(np.dot(curr_z_axis, target_z_axis), -1.0, 1.0))
    return float(np.arccos(axis_alignment))


def _z_axis_alignment_cost_wxyz(curr_quat_wxyz, target_quat_wxyz):
    curr_z_axis = _quat_wxyz_to_z_axis_np(curr_quat_wxyz)
    target_z_axis = _quat_wxyz_to_z_axis_np(target_quat_wxyz)
    axis_alignment = float(np.clip(np.dot(curr_z_axis, target_z_axis), -1.0, 1.0))
    return float(1.0 - axis_alignment)


def _contact_wrench_world_from_optimizer_info(info, r_obj_to_world):
    r_obj_to_world = np.asarray(r_obj_to_world, dtype=np.float32).reshape(3, 3)
    force_local = np.asarray(
        info.get("contact_force_local", np.zeros(3, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(3)
    force_world = (r_obj_to_world @ force_local).astype(np.float32)

    contact_point_local = info.get("contact_point_local", None)
    if contact_point_local is not None:
        # Build the world-frame moment directly from the transformed contact force
        # so the tilted support-frame geometry stays explicit.
        moment_arm_world = (
            r_obj_to_world
            @ np.asarray(contact_point_local, dtype=np.float32).reshape(3)
        ).astype(np.float32)
        torque_world = np.cross(moment_arm_world, force_world).astype(np.float32)
    else:
        torque_local = np.asarray(
            info.get("contact_torque_local", np.zeros(3, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(3)
        torque_world = (r_obj_to_world @ torque_local).astype(np.float32)
    return force_world, torque_world


def _quat_wxyz_to_rotvec_np(quat_wxyz):
    quat_xyzw = _quat_wxyz_to_xyzw_np(_normalize_quaternion_wxyz_np(quat_wxyz))
    return Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_rotvec().astype(np.float32)


def _compute_stage1_pd_torque_scale(x_plus_local, v_plus_local, current_v_local, args):
    x_plus_local = np.asarray(x_plus_local, dtype=np.float32).reshape(7)
    v_plus_local = np.asarray(v_plus_local, dtype=np.float32).reshape(6)
    current_v_local = np.asarray(current_v_local, dtype=np.float32).reshape(6)

    desired_pos_error = float(np.linalg.norm(x_plus_local[:3]))
    desired_ori_error = float(np.linalg.norm(_quat_wxyz_to_rotvec_np(x_plus_local[3:7])))
    linear_velocity_error = float(np.linalg.norm(v_plus_local[:3] - current_v_local[:3]))
    angular_velocity_error = float(np.linalg.norm(v_plus_local[3:] - current_v_local[3:]))

    scale = (
        float(args.stage1_torque_pd_kp_pos) * desired_pos_error
        + float(args.stage1_torque_pd_kp_ori) * desired_ori_error
        + float(args.stage1_torque_pd_kd_lin) * linear_velocity_error
        + float(args.stage1_torque_pd_kd_ang) * angular_velocity_error
    )
    return float(np.clip(scale, float(args.stage1_torque_scale_min), float(args.stage1_torque_scale_max)))


def _contact_force_edge_directions_world(n, t1, t2, mu):
    n = np.asarray(n, dtype=np.float32).reshape(3)
    t1 = np.asarray(t1, dtype=np.float32).reshape(3)
    t2 = np.asarray(t2, dtype=np.float32).reshape(3)
    return np.column_stack(
        [
            n + mu * t1,
            n + mu * t2,
            n - mu * t1,
            n - mu * t2,
        ]
    ).astype(np.float32)


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
    param.table_friction_ = ROLLOUT_TABLE_FRICTION
    hold_mu = abs(np.tan(float(TILTED_RAMP_ROLL_RAD))) + TILTED_RAMP_HOLD_FRICTION_MARGIN
    param.ramp_friction_ = max(float(param.table_friction_), hold_mu)
    param.object_friction_ = ROLLOUT_OBJECT_FRICTION
    param.fingertip_friction_ = float(ROLLOUT_FINGERTIP_FRICTION)
    rollout_mass = float(getattr(param, "lambda_obj_mass_", 0.01))
    param.sim_obj_mass_ = rollout_mass
    param.obj_mass_ = rollout_mass
    param.mu_object_ = float(param.object_friction_)
    param.sim_obj_inertia_diag_ = _box_inertia_diag(
        rollout_mass,
        getattr(param, "object_aabb_lo", np.array([-0.03, -0.03, -0.03])),
        getattr(param, "object_aabb_hi", np.array([0.03, 0.03, 0.03])),
    )
    param.gravity_[2] = -9.81
    return param


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


def _build_mpc_trackers(args):
    mpc_step = max(1e-4, float(getattr(args, "mpc_step_limit", 0.005)))
    via_step = getattr(args, "via_max_step", None)
    via_step = mpc_step if via_step is None else max(1e-4, float(via_step))
    via_lead = max(1e-4, float(getattr(args, "via_max_lead", via_step)))
    return {
        "value_tracker": ContactValueTracker(
            tau=float(args.value_tau),
            rel_scale=float(args.value_rel_scale),
            rho=float(args.value_rho),
            alpha=float(args.value_alpha),
            beta=float(args.verify_beta),
            window_size=int(getattr(args, "verify_window_size", 5)),
            confirm_steps=int(getattr(args, "verify_enter_steps", 5)),
            min_hold_steps=int(getattr(args, "verify_hold_steps", 30)),
            release_steps=int(getattr(args, "verify_release_steps", 8)),
            accept_margin_ratio=0.05,
            accept_margin_abs=0.02,
        ),
        "model_cost_conf": ModelCostConfidence(
            threshold=float(getattr(args, "model_cost_error_threshold", 6.0)),
            eps=float(getattr(args, "model_cost_error_eps", 1e-6)),
            min_steps=int(getattr(args, "model_cost_error_min_steps", 3)),
        ),
        "approach_via": SmoothedApproachVia(
            rate=float(getattr(args, "via_smooth_rate", ISAAC_VIA_SMOOTH_RATE)),
            max_step=via_step,
            max_lead=via_lead,
        ),
        "arrived_hold": False,
        "arrived_dest_idx": None,
        "sol_guess": None,
        "last_verify_cost": None,
    }


def _scalar_bound(value, default=0.005):
    arr = np.asarray(value if value is not None else default, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr[0]):
        return abs(float(default))
    return abs(float(arr[0]))


class IsaacFrankaOSCSimulator(IsaacFrankaSimulator):
    def __init__(self, param, headless=False, sim_device="cuda:0", graphics_device_id=0):
        self.param_ = param
        self.break_out_signal_ = False
        self.dyn_paused_ = False
        self.viewer_ = None
        self.svg_screenshot_recorder_ = None
        self.video_recorder_ = None
        self.show_goal_object_ = bool(
            getattr(
                self.param_,
                "show_goal_object_",
                getattr(self.param_, "show_ghost_object_", getattr(self.param_, "show_goal_pose_", False)),
            )
        )
        self.show_goal_pose_ = bool(
            getattr(self.param_, "show_goal_pose_", getattr(self.param_, "show_ghost_object_", False))
        )
        self.show_point_ = bool(getattr(self.param_, "show_point_", True))
        # Keep the legacy field in sync because inherited helpers still read it.
        self.show_ghost_object_ = self.show_goal_object_

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
        axis_pose = getattr(self.param_, "axis_pose_", None)
        if axis_pose is None:
            axis_pose = np.hstack([self.param_.target_p_, self.param_.target_q_]).astype(np.float32)
        axis_position, axis_quat_xyzw = _pose_wxyz_to_xyzw_components(axis_pose)
        self._axis_pose_pos_cache = axis_position
        self._axis_pose_quat_xyzw_cache = axis_quat_xyzw
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

        video_output_path = getattr(self.param_, "video_output_path_", None)
        if video_output_path:
            try:
                self.video_recorder_ = create_isaacgym_mp4_recorder(
                    gym=self.gym,
                    sim=self.sim,
                    env=self.env,
                    output_path=video_output_path,
                    camera_position=SVG_SCREENSHOT_CAMERA_POSITION,
                    camera_target=SVG_SCREENSHOT_CAMERA_TARGET,
                    fps=float(getattr(self.param_, "video_fps_", 20.0)),
                    width=int(getattr(self.param_, "video_width_", 1280)),
                    height=int(getattr(self.param_, "video_height_", 960)),
                    capture_on_start=True,
                )
                print(
                    "[IsaacFrankaOSCSimulator] MP4 capture enabled: "
                    f"path={video_output_path}, fps={float(getattr(self.param_, 'video_fps_', 20.0)):.2f}, "
                    f"size={int(getattr(self.param_, 'video_width_', 1280))}x"
                    f"{int(getattr(self.param_, 'video_height_', 960))}"
                )
            except RuntimeError as exc:
                self.video_recorder_ = None
                print(f"[IsaacFrankaOSCSimulator] MP4 capture disabled: {exc}")

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

        # Keep the inherited large box as the desk, then add a tilted ramp as
        # the actual support surface for the object.
        self.desk_actor = self.table_actor
        desk_color = gymapi.Vec3(0.45, 0.47, 0.50)
        num_desk_bodies = self.gym.get_actor_rigid_body_count(self.env, self.desk_actor)
        for body_idx in range(num_desk_bodies):
            self.gym.set_rigid_body_color(
                self.env,
                self.desk_actor,
                body_idx,
                gymapi.MESH_VISUAL,
                desk_color,
            )

        ramp_defaults = _compute_tilted_ramp_config(float(self.param_.table_height))
        ramp_size = np.asarray(
            getattr(self.param_, "tilted_ramp_size_", ramp_defaults["size"]),
            dtype=np.float32,
        ).reshape(3)
        ramp_center = np.asarray(
            getattr(self.param_, "tilted_ramp_center_", ramp_defaults["center"]),
            dtype=np.float32,
        ).reshape(3)
        ramp_quat_xyzw = np.asarray(
            getattr(self.param_, "tilted_ramp_quat_xyzw_", ramp_defaults["quat_xyzw"]),
            dtype=np.float32,
        ).reshape(4)

        ramp_opts = gymapi.AssetOptions()
        ramp_opts.fix_base_link = True
        self.ramp_asset = self.gym.create_box(
            self.sim,
            float(ramp_size[0]),
            float(ramp_size[1]),
            float(ramp_size[2]),
            ramp_opts,
        )
        self.table_asset = self.ramp_asset

        ramp_pose = gymapi.Transform()
        ramp_pose.p = gymapi.Vec3(float(ramp_center[0]), float(ramp_center[1]), float(ramp_center[2]))
        ramp_pose.r = gymapi.Quat(
            float(ramp_quat_xyzw[0]),
            float(ramp_quat_xyzw[1]),
            float(ramp_quat_xyzw[2]),
            float(ramp_quat_xyzw[3]),
        )
        self.ramp_actor = self.gym.create_actor(self.env, self.ramp_asset, ramp_pose, "tilted_ramp", 0, 0)
        self.table_actor = self.ramp_actor

        ramp_color = gymapi.Vec3(0.60, 0.61, 0.65)
        num_ramp_bodies = self.gym.get_actor_rigid_body_count(self.env, self.ramp_actor)
        for body_idx in range(num_ramp_bodies):
            self.gym.set_rigid_body_color(
                self.env,
                self.ramp_actor,
                body_idx,
                gymapi.MESH_VISUAL,
                ramp_color,
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

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
        force_cylinder_urdf_rel, force_asset_root = self._prepare_force_cylinder_urdf_asset(repo_root)
        force_opts = gymapi.AssetOptions()
        force_opts.fix_base_link = True
        force_opts.disable_gravity = True
        self.force_cylinder_asset = self.gym.load_asset(self.sim, force_asset_root, force_cylinder_urdf_rel, force_opts)

        force_pose = gymapi.Transform()
        force_pose.p = gymapi.Vec3(
            float(FORCE_VIS_HIDDEN_POSITION[0]),
            float(FORCE_VIS_HIDDEN_POSITION[1]),
            float(FORCE_VIS_HIDDEN_POSITION[2]),
        )
        force_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.force_cylinder_actor = self.gym.create_actor(
            self.env,
            self.force_cylinder_asset,
            force_pose,
            "best_contact_force_cylinder",
            0,
            0,
        )
        self.gym.set_actor_scale(self.env, self.force_cylinder_actor, 1.0)

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
        <cylinder radius="{ATTACHMENT_RADIUS}" length="0.06"/>
      </geometry>
      <material name="attachment_dark">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.03" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{ATTACHMENT_RADIUS}" length="0.06"/>
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
        <sphere radius="{FINGERTIP_RADIUS}"/>
      </geometry>
      <material name="fingertip_red">
        <color rgba="0.8 0.2 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="{FINGERTIP_RADIUS}"/>
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

    def _prepare_mesh_urdf_assets(
        self,
        repo_root,
        mesh_path,
        visual_mesh_path=None,
        mass=0.01,
        inertia_diag=None,
    ):
        if mesh_path is None:
            raise ValueError("param.mesh_path_ is required for Isaac object asset loading")

        mesh_abs_path = mesh_path if os.path.isabs(mesh_path) else os.path.join(repo_root, mesh_path)
        mesh_abs_path = os.path.abspath(mesh_abs_path)
        if not os.path.isfile(mesh_abs_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_abs_path}")
        visual_src = visual_mesh_path or mesh_path
        visual_abs_path = visual_src if os.path.isabs(visual_src) else os.path.join(repo_root, visual_src)
        visual_abs_path = os.path.abspath(visual_abs_path)
        if not os.path.isfile(visual_abs_path):
            visual_abs_path = mesh_abs_path

        mesh_asset_root = os.path.join(repo_root, "envs", "assets", "objects", "_isaac_tmp")
        os.makedirs(mesh_asset_root, exist_ok=True)
        mesh_rel_to_urdf = os.path.relpath(mesh_abs_path, mesh_asset_root)
        visual_rel_to_urdf = os.path.relpath(visual_abs_path, mesh_asset_root)
        mesh_scale_vec = np.asarray(
            getattr(self.param_, "obj_mesh_scale_", np.ones(3, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(3)
        mesh_scale = " ".join(f"{float(v):.8g}" for v in mesh_scale_vec)
        obj_urdf_rel = "obj_mesh_dynamic.urdf"
        obj_urdf_abs = os.path.join(mesh_asset_root, obj_urdf_rel)
        target_urdf_rel = "obj_mesh_target_ghost.urdf"
        target_urdf_abs = os.path.join(mesh_asset_root, target_urdf_rel)
        if inertia_diag is None:
            inertia_diag = (1.5e-6, 1.5e-6, 1.5e-6)
        ixx, iyy, izz = [float(v) for v in np.asarray(inertia_diag, dtype=np.float64).reshape(3)]
        mass = float(mass)

        obj_urdf = f"""<?xml version="1.0"?>
<robot name="mesh_obj">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{visual_rel_to_urdf}" scale="{mesh_scale}"/>
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
        <mesh filename="{visual_rel_to_urdf}" scale="{mesh_scale}"/>
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

    @staticmethod
    def _prepare_force_cylinder_urdf_asset(repo_root):
        marker_asset_root = os.path.join(repo_root, "envs", "assets", "objects", "_isaac_tmp")
        os.makedirs(marker_asset_root, exist_ok=True)
        force_urdf_rel = "force_cylinder_visual_only.urdf"
        force_urdf_abs = os.path.join(marker_asset_root, force_urdf_rel)

        force_urdf = f"""<?xml version="1.0"?>
<robot name="force_cylinder_visual_only">
  <link name="base">
    <visual>
      <origin xyz="0 0 {0.5 * FORCE_CYLINDER_BASE_LENGTH}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{FORCE_CYLINDER_RADIUS}" length="{FORCE_CYLINDER_BASE_LENGTH}"/>
      </geometry>
      <material name="force_yellow">
        <color rgba="1.0 0.92 0.15 1.0"/>
      </material>
    </visual>
  </link>
</robot>
"""
        with open(force_urdf_abs, "w", encoding="ascii") as f:
            f.write(force_urdf)

        return force_urdf_rel, marker_asset_root

    def _apply_scene_physics_settings(self):
        self.table_friction_ = float(getattr(self.param_, "table_friction_", 0.5))
        self.ramp_friction_ = float(getattr(self.param_, "ramp_friction_", self.table_friction_))
        self.object_friction_ = float(getattr(self.param_, "object_friction_", 0.5))
        self.fingertip_friction_ = float(
            getattr(self.param_, "fingertip_friction_", ROLLOUT_FINGERTIP_FRICTION)
        )
        self.obj_mass_ = float(
            getattr(self.param_, "sim_obj_mass_", getattr(self.param_, "obj_mass_", 0.01))
        )
        inertia_diag = getattr(self.param_, "sim_obj_inertia_diag_", None)
        _set_actor_friction(self.gym, self.env, self.table_actor, self.ramp_friction_)
        if hasattr(self, "desk_actor"):
            _set_actor_friction(self.gym, self.env, self.desk_actor, self.table_friction_)
        _set_actor_friction(self.gym, self.env, self.obj_actor, self.object_friction_)
        _set_actor_mass(self.gym, self.env, self.obj_actor, self.obj_mass_, inertia_diag)
        self._apply_fingertip_only_object_collision()

    def _apply_fingertip_only_object_collision(self):
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
                            shape_props[k].friction = float(self.fingertip_friction_)
                            shape_props[k].torsion_friction = float(self.fingertip_friction_)
                            shape_props[k].rolling_friction = float(self.fingertip_friction_)
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
        if hasattr(self, "desk_actor"):
            desk_props = self.gym.get_actor_rigid_shape_properties(self.env, self.desk_actor)
            for prop in desk_props:
                prop.filter = 0
            self.gym.set_actor_rigid_shape_properties(self.env, self.desk_actor, desk_props)

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
        self.dof_lower_ = np.array(dof_props["lower"][:7], dtype=np.float32)
        self.dof_upper_ = np.array(dof_props["upper"][:7], dtype=np.float32)

        pos_stiffness = float(getattr(self.param_, "osc_pos_stiffness_", 12000.0))
        ori_stiffness = float(getattr(self.param_, "osc_ori_stiffness_", 0.0))
        self.osc_task_kp = np.array(
            [pos_stiffness, pos_stiffness, pos_stiffness, ori_stiffness, ori_stiffness, ori_stiffness],
            dtype=np.float32,
        )
        self.osc_task_kd = (1.4 * np.sqrt(self.osc_task_kp)).astype(np.float32)
        self.nullspace_stiffness = float(getattr(self.param_, "nullspace_stiffness_", 10.0))
        self.home_q = np.array(self.param_.init_robot_qpos_, dtype=np.float32)
        self.q_d_nullspace = FRANKA_QD_NULLSPACE.copy()
        self.low_height = float(
            getattr(self.param_, "support_surface_min_height_", self.param_.table_height) + 0.02
        )
        self.activate_tool_compensation = False
        self.tool_compensation_force = np.zeros(6, dtype=np.float32)

    def _build_body_index_cache(self):
        super()._build_body_index_cache()
        self.ramp_body_idx = int(self.table_body_idx)
        self.support_surface_body_indices = {int(self.table_body_idx)}
        if hasattr(self, "desk_actor"):
            self.desk_body_idx = self.gym.get_actor_rigid_body_index(
                self.env,
                self.desk_actor,
                0,
                gymapi.DOMAIN_SIM,
            )
            self.support_surface_body_indices.add(int(self.desk_body_idx))
            self.sim_body_to_actor[int(self.desk_body_idx)] = (self.desk_actor, 0, "desk")
        else:
            self.desk_body_idx = None
        self.obj_body_env_idx = self.gym.get_actor_rigid_body_index(
            self.env, self.obj_actor, 0, gymapi.DOMAIN_ENV
        )
        self.env_rigid_body_count = self.gym.get_env_rigid_body_count(self.env)
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
        self.fingertip_body_sim_idx = self.gym.get_actor_rigid_body_index(
            self.env,
            self.franka_actor,
            self.fingertip_body_idx,
            gymapi.DOMAIN_SIM,
        )
        self.task_body_local_idx = self.fingertip_body_idx

        rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        mm = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        self._rigid_body_states = gymtorch.wrap_tensor(rb_states)
        self._dof_state = gymtorch.wrap_tensor(dof_states)
        self._mm = gymtorch.wrap_tensor(mm)
        self._effort_control = torch.zeros((self.franka_dof_count,), dtype=torch.float32, device=self._dof_state.device)
        self._rb_forces = torch.zeros(
            (1, self.env_rigid_body_count, 3),
            dtype=torch.float32,
            device=self._dof_state.device,
        )
        self._rb_torques = torch.zeros(
            (1, self.env_rigid_body_count, 3),
            dtype=torch.float32,
            device=self._dof_state.device,
        )

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
        self._hide_best_contact_force_visual()
        self._sync_point_marker()
        self._place_fingertip_for_rollout()
        self._settle_object_on_ramp()
        self._sync_desired_pose_with_current()
        self._hold_q0 = np.asarray(self.get_current_joint_position(), dtype=np.float32)
        self._hold_dq = None
        self._hold_i = 0
        self._hold_n = 1
        self._contact_info_cache = None
        self._clear_mpc_action()

    def _zero_object_velocity(self):
        obj_state = self.gym.get_actor_rigid_body_states(self.env, self.obj_actor, gymapi.STATE_ALL)
        obj_state["vel"]["linear"][0] = (0.0, 0.0, 0.0)
        obj_state["vel"]["angular"][0] = (0.0, 0.0, 0.0)
        self.gym.set_actor_rigid_body_states(self.env, self.obj_actor, obj_state, gymapi.STATE_ALL)

    def _settle_object_on_ramp(self):
        # Parent reset only drops ~1 cm in 8 steps. Keep the landed pose and
        # pin the arm so extra settle time does not let effort joints sag.
        hold_q = np.asarray(self.get_current_joint_position(), dtype=np.float32)
        for _ in range(TILTED_RAMP_SETTLE_CYCLES):
            for _ in range(TILTED_RAMP_SETTLE_STEPS):
                self._set_arm_qpos(hold_q)
                self._simulate_once(draw=False)
            self._zero_object_velocity()

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
        start_xy = np.array([obj[0] + 0.10 * toward[0], obj[1] + 0.10 * toward[1]], dtype=np.float64)
        side_clearance = max(0.025, 0.45 * float(getattr(self.param_, "object_circumradius", 0.06)))
        return np.array(
            [start_xy[0], start_xy[1], _support_surface_z(self.param_, start_xy) + side_clearance],
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
        self.q_d_nullspace = self.get_current_joint_position().copy()

    def _simulate_once(self, sync_realtime=False, draw=True):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if draw:
            self._contact_info_cache = None
            self._sync_pose_axes()

        graphics_stepped = False
        need_graphics = draw and (
            self.viewer_ is not None
            or self.svg_screenshot_recorder_ is not None
            or self.video_recorder_ is not None
        )
        if need_graphics:
            self.gym.step_graphics(self.sim)
            graphics_stepped = True

        if draw and self.svg_screenshot_recorder_ is not None:
            self.svg_screenshot_recorder_.capture_if_due(
                sim_time=float(self.gym.get_sim_time(self.sim)),
                step_graphics=not graphics_stepped,
            )

        if draw and self.video_recorder_ is not None:
            self.video_recorder_.capture_if_due(
                sim_time=float(self.gym.get_sim_time(self.sim)),
                step_graphics=not graphics_stepped,
            )

        if draw and self.viewer_ is not None:
            self.gym.draw_viewer(self.viewer_, self.sim, True)
            if sync_realtime:
                self.gym.sync_frame_time(self.sim)

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

    def get_object_pose(self):
        obj_state = self.gym.get_actor_rigid_body_states(self.env, self.obj_actor, gymapi.STATE_POS)
        obj_pos = _extract_vec3(obj_state["pose"]["p"][0]).astype(np.float32)
        quat_xyzw = _extract_quat_xyzw(obj_state["pose"]["r"][0]).astype(np.float32)
        obj_quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
        return obj_pos, obj_quat_wxyz

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

    def set_tool_compensation_wrench(self, force_world=None, torque_world=None):
        wrench = np.zeros(6, dtype=np.float32)
        has_target = False

        if force_world is not None:
            force_world = np.asarray(force_world, dtype=np.float32).reshape(3)
            wrench[:3] = force_world
            has_target = has_target or bool(np.linalg.norm(force_world) > 1e-6)

        if torque_world is not None:
            torque_world = np.asarray(torque_world, dtype=np.float32).reshape(3)
            wrench[3:] = torque_world
            has_target = has_target or bool(np.linalg.norm(torque_world) > 1e-6)

        # In this scene the Franka base frame is aligned with the world frame,
        # so the OSC wrench term can consume the world-frame target directly.
        self.set_tool_compensation(wrench, activate=has_target)

    def get_R(self):
        return self.R_d

    def _sync_point_marker(self, goal_pos=None):
        marker_state = self.gym.get_actor_rigid_body_states(self.env, self.p_arm_marker_actor, gymapi.STATE_ALL)
        if not self.show_point_:
            marker_state["pose"]["p"][0] = (
                float(FORCE_VIS_HIDDEN_POSITION[0]),
                float(FORCE_VIS_HIDDEN_POSITION[1]),
                float(FORCE_VIS_HIDDEN_POSITION[2]),
            )
        elif goal_pos is not None:
            marker_state["pose"]["p"][0] = (float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2]))
        marker_state["vel"]["linear"][0] = (0.0, 0.0, 0.0)
        marker_state["vel"]["angular"][0] = (0.0, 0.0, 0.0)
        self.gym.set_actor_rigid_body_states(self.env, self.p_arm_marker_actor, marker_state, gymapi.STATE_ALL)

    def show_point(self, goal_pos=None):
        self._sync_point_marker(goal_pos)

    def show_target_object_pose(self, goal_pos=None, goal_quat_wxyz=None):
        if goal_pos is not None:
            self._target_obj_pos_cache = np.asarray(goal_pos, dtype=np.float32).copy()
        if goal_quat_wxyz is not None:
            self._target_obj_quat_xyzw_cache = _quat_wxyz_to_xyzw_np(goal_quat_wxyz)
        if bool(getattr(self.param_, "lock_axis_pose_to_goal_pose_", True)):
            self._axis_pose_pos_cache = self._target_obj_pos_cache.copy()
            self._axis_pose_quat_xyzw_cache = self._target_obj_quat_xyzw_cache.copy()
        self._sync_target_object_visual()
        self._sync_pose_axes()

    def set_goal_pose_visibility(self, visible):
        visible = bool(visible)
        self.show_goal_pose_ = visible
        self._sync_pose_axes()

    def set_ghost_object_visibility(self, visible):
        visible = bool(visible)
        self.show_goal_object_ = visible
        self.show_ghost_object_ = visible
        self._sync_target_object_visual()

    def set_axis_pose(self, pose_wxyz=None):
        if pose_wxyz is None:
            return
        pose_wxyz = np.asarray(pose_wxyz, dtype=np.float32).reshape(7)
        self._axis_pose_pos_cache = pose_wxyz[:3].copy()
        self._axis_pose_quat_xyzw_cache = _quat_wxyz_to_xyzw_np(pose_wxyz[3:7])
        self._sync_pose_axes()

    def _sync_pose_axes(self):
        if not hasattr(self, "obj_pose_axes_actor") or not hasattr(self, "target_pose_axes_actor"):
            return
        if self.show_point_:
            obj_pos, obj_quat_xyzw = self._get_actor_body_pose(self.obj_actor, 0)
        else:
            obj_pos = FORCE_VIS_HIDDEN_POSITION
            obj_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._set_visual_actor_pose_xyzw(self.obj_pose_axes_actor, obj_pos, obj_quat_xyzw)
        if self.show_goal_pose_:
            axis_pos = getattr(self, "_axis_pose_pos_cache", self._target_obj_pos_cache)
            axis_quat_xyzw = getattr(self, "_axis_pose_quat_xyzw_cache", self._target_obj_quat_xyzw_cache)
        else:
            axis_pos = FORCE_VIS_HIDDEN_POSITION
            axis_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._set_visual_actor_pose_xyzw(self.target_pose_axes_actor, axis_pos, axis_quat_xyzw)

    def _hide_best_contact_force_visual(self):
        if hasattr(self, "force_cylinder_actor"):
            self.gym.set_actor_scale(self.env, self.force_cylinder_actor, 1.0)
            self._set_visual_actor_pose_xyzw(
                self.force_cylinder_actor,
                FORCE_VIS_HIDDEN_POSITION,
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )
        if self.viewer_ is not None:
            self.gym.clear_lines(self.viewer_)

    def finalize_video_recording(self):
        if self.video_recorder_ is None:
            return None
        saved_path = self.video_recorder_.close()
        self.video_recorder_ = None
        return saved_path

    def show_best_contact_force(self, contact_point_world=None, force_world=None):
        if contact_point_world is None or force_world is None or not hasattr(self, "force_cylinder_actor"):
            self._hide_best_contact_force_visual()
            return

        contact_point_world = np.asarray(contact_point_world, dtype=np.float32).reshape(3)
        force_world = np.asarray(force_world, dtype=np.float32).reshape(3)
        force_norm = float(np.linalg.norm(force_world))
        if force_norm < 1e-6:
            self._hide_best_contact_force_visual()
            return

        direction = force_world / force_norm
        display_length = float(
            np.clip(force_norm * FORCE_VECTOR_LENGTH_SCALE, FORCE_VECTOR_MIN_LENGTH, FORCE_VECTOR_MAX_LENGTH)
        )
        shaft_quat_xyzw = _quat_xyzw_from_z_axis(direction)
        self.gym.set_actor_scale(self.env, self.force_cylinder_actor, display_length / FORCE_CYLINDER_BASE_LENGTH)
        self._set_visual_actor_pose_xyzw(self.force_cylinder_actor, contact_point_world, shaft_quat_xyzw)

        if self.viewer_ is None:
            return

        self.gym.clear_lines(self.viewer_)
        tip = contact_point_world + display_length * direction
        ref_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(direction, ref_axis))) > 0.9:
            ref_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        side_axis_1 = np.cross(direction, ref_axis)
        side_axis_1 = side_axis_1 / max(np.linalg.norm(side_axis_1), 1e-6)
        side_axis_2 = np.cross(direction, side_axis_1)
        side_axis_2 = side_axis_2 / max(np.linalg.norm(side_axis_2), 1e-6)
        arrow_base = tip - FORCE_ARROW_HEAD_LENGTH * direction
        arrow_color = [1.0, 0.92, 0.15]

        for side_dir in (side_axis_1, -side_axis_1, side_axis_2, -side_axis_2):
            arrow_point = arrow_base + 0.5 * FORCE_ARROW_HEAD_WIDTH * side_dir
            self.gym.add_lines(
                self.viewer_,
                self.env,
                1,
                [
                    float(tip[0]),
                    float(tip[1]),
                    float(tip[2]),
                    float(arrow_point[0]),
                    float(arrow_point[1]),
                    float(arrow_point[2]),
                ],
                arrow_color,
            )

    def _get_task_jacobian(self):
        jac_task_idx = self._jacobian_body_index(self.task_body_local_idx)
        return np.array(self._jacobian[0, jac_task_idx, :, :7], dtype=np.float32)

    def _get_position_jacobian(self, q=None):
        if q is None:
            q = self.get_current_joint_position()
        return _franka_jacobian_pos_np(q).astype(np.float32)

    def _get_arm_mass_matrix(self):
        mm = self._mm
        if mm.ndim == 3:
            mm = mm[0]
        return np.array(mm[:7, :7], dtype=np.float32)

    def _fingertip_contact_info(self):
        cached = getattr(self, "_contact_info_cache", None)
        if cached is not None:
            return cached
        in_contact, best_n, _sep = self.fingertip_object_contact()
        self._contact_info_cache = (in_contact, best_n)
        return in_contact, best_n

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
        if action is not None and p0 is not None:
            remain = _remaining_along_action(p_curr, p0, action)
            v_ref = _project_along_action(
                np.asarray(action, dtype=np.float64) / ISAAC_REF_HORIZON, action)
        else:
            remain = err
            v_ref = None
        force_track = _isaac_task_force(
            remain, vel, v_ref=v_ref,
            k_task=ISAAC_TASK_KP, d_task=ISAAC_TASK_KD,
            limit=FREE_SPACE_FORCE_LIMIT,
        )
        if action is not None:
            force_track = _project_along_action(force_track, action)
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
        self._refresh_osc_tensors()
        q = self.get_current_joint_position()
        dq = self.get_current_joint_velocity()
        jacobian = self._get_task_jacobian()
        p, R_current = self.get_end_effector_pos()
        cartesian_stiffness = self._format_cartesian_gain(self.cartesian_stiffness)
        cartesian_damping = self._format_cartesian_gain(self.cartesian_damping)

        error = np.zeros(6, dtype=np.float32)
        error[:3] = p - self.position_d
        R_d = self.orientation_d
        R_error = R_current.T @ R_d
        error_quat = Rotation.from_matrix(R_error).as_quat()
        if error_quat[3] < 0:
            error_quat = -error_quat
        error[3:] = -R_current @ error_quat[:3]

        velocity = jacobian @ dq
        F_ee_des = -cartesian_stiffness @ error - cartesian_damping @ velocity
        tau_task = jacobian.T @ F_ee_des

        position_only = float(np.diag(cartesian_stiffness)[3]) <= 1e-6
        task_jac = jacobian[:3] if position_only else jacobian
        tau_nullspace = _franka_nullspace_posture_torque(
            task_jac, q, dq, self.q_d_nullspace, self.nullspace_stiffness
        )

        if self.activate_tool_compensation:
            tau_tool = jacobian.T @ self.tool_compensation_force
        else:
            tau_tool = np.zeros(7)

        tau_d = np.nan_to_num(tau_task + tau_nullspace + tau_tool, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(tau_d, -self.torque_limits, self.torque_limits)

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

    def apply_object_wrench_world(self, force_world=None, torque_world=None):
        self._rb_forces.zero_()
        self._rb_torques.zero_()

        if force_world is not None:
            force_world = np.asarray(force_world, dtype=np.float32).reshape(3)
            self._rb_forces[0, self.obj_body_env_idx, :] = torch.as_tensor(
                force_world,
                dtype=torch.float32,
                device=self._rb_forces.device,
            )

        if torque_world is not None:
            torque_world = np.asarray(torque_world, dtype=np.float32).reshape(3)
            self._rb_torques[0, self.obj_body_env_idx, :] = torch.as_tensor(
                torque_world,
                dtype=torch.float32,
                device=self._rb_torques.device,
            )

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self._rb_forces),
            gymtorch.unwrap_tensor(self._rb_torques),
            gymapi.ENV_SPACE,
        )

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
        self.set_via_action(via_pos)
        if n_substeps is None:
            n_substeps = policy_control_substeps(
                getattr(self.param_, "control_substeps_", 0), self.sim_dt_
            )
        applied_tau = None
        for i in range(n_substeps):
            last = (i + 1 == n_substeps)
            applied_tau = self.step_control_frame(
                draw=last or self.video_recorder_ is not None,
                sync_realtime=bool(sync_realtime) and last,
            )
        return applied_tau

    def step_joint_delta(self, dq, sync_realtime=True):
        q = self.get_current_joint_position()
        dq = np.asarray(dq, dtype=np.float32).reshape(7)
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
                draw=last or self.video_recorder_ is not None,
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

    def set_via_action(self, via_pos, action=None, policy_dt=None,
                       press=None, path_blocked=None):
        p_curr, _ = self.get_end_effector_pos()
        p_curr = np.asarray(p_curr, dtype=np.float64).reshape(3)
        via = np.asarray(via_pos, dtype=np.float64).reshape(3)
        max_step = min(
            _scalar_bound(getattr(self.param_, "mpc_u_ub_", 0.005), 0.005),
            _scalar_bound(getattr(self.param_, "isaac_via_max_step_", AIR_VIA_STEP), AIR_VIA_STEP),
        )
        max_slew = _scalar_bound(
            getattr(self.param_, "isaac_action_slew_", ISAAC_ACTION_SLEW),
            ISAAC_ACTION_SLEW,
        )
        if bool(path_blocked):
            max_step = _scalar_bound(getattr(self.param_, "mpc_u_ub_", 0.005), 0.005)
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
        self.finalize_video_recording()
        if self.svg_screenshot_recorder_ is not None:
            self.svg_screenshot_recorder_.close()
            self.svg_screenshot_recorder_ = None
        super().close()


class ContactIsaacRamp(ContactIsaacCartesian):
    """ik2 fingertip signed-gap detector, with one ramp plane instead of a table."""

    @staticmethod
    def _first_free_contact_row(phi_vec, max_ncon):
        for row_idx in range(int(max_ncon)):
            if np.allclose(phi_vec[4 * row_idx : 4 * row_idx + 4], 1.0):
                return row_idx
        return int(max_ncon)

    def _planar_ramp_jacobians(self, obj_pos, r_obj_to_world, nv, mu, skew_fn):
        point, normal = _support_plane(self.param_)
        return planar_support_jacobians(
            obj_pos, r_obj_to_world, point, normal, nv, mu, skew_fn=skew_fn
        )

    def detect_once(self, simulator):
        saved_table_idx = simulator.table_body_idx
        saved_table_height = self.param_.table_height
        try:
            # Hide the flat-table plane so ik2 only fills fingertip signed-gap rows.
            simulator.table_body_idx = -1
            self.param_.table_height = -1.0e6
            phi_vec, jac_mat, con_pos_list, jac_mat_env, if_contact = super().detect_once(simulator)
        finally:
            simulator.table_body_idx = saved_table_idx
            self.param_.table_height = saved_table_height

        full_q = np.asarray(simulator.get_state(), dtype=np.float32)
        obj_pos = full_q[0:3]
        obj_quat_wxyz = full_q[3:7]
        quat_xyzw = np.array(
            [obj_quat_wxyz[1], obj_quat_wxyz[2], obj_quat_wxyz[3], obj_quat_wxyz[0]],
            dtype=np.float32,
        )
        r_obj_to_world = Rotation.from_quat(quat_xyzw).as_matrix()
        dist_ramp = _support_signed_distance(self.param_, obj_pos)
        # COM-to-plane is typically 2–4 cm on foam_brick, so the old
        # ``dist < 0.02`` gate dropped the ramp row and left J_tilde = 0.
        # Lambda then ranked top / edge samples as a gravity jack.
        con_jac_table, con_jac_body, con_pos_local, _dist = self._planar_ramp_jacobians(
            obj_pos, r_obj_to_world, self.param_.n_qvel_, float(self.param_.mu_object_), simulator._skew
        )
        jac_mat_env[0:4, :6] = con_jac_body
        con_pos_list.append(np.asarray(con_pos_local, dtype=np.float32))
        # Same plane row for MPC ``jac_mat`` (world frame) as lambda
        # ``jac_mat_env`` (body frame).  Do not gate on COM height: foam_brick
        # sits 2–4 cm above the plane and would otherwise fall in the model.
        row_idx = self._first_free_contact_row(phi_vec, self.param_.max_ncon_)
        if row_idx < self.param_.max_ncon_:
            phi_vec[4 * row_idx : 4 * row_idx + 4] = min(max(dist_ramp, 0.0), 0.02)
            jac_mat[4 * row_idx : 4 * row_idx + 4, :] = con_jac_table
        return phi_vec, jac_mat, con_pos_list, jac_mat_env, if_contact


def adapt_param_for_cartesian_solver(param, args):
    args.solver = "acados"
    if getattr(param, "lambda_optimizer", None) is None:
        param.lambda_optimizer = build_lambda_optimizer(param, args)
    param = _adapt_ik2_cartesian_solver(param, args)
    z_lo = float(getattr(param, "support_surface_min_height_", param.table_height)) - 0.01
    param.mpc_q_lb_ = np.hstack((
        np.array([-1e7, -1e7, z_lo - 0.02], dtype=np.float64),
        -1e7 * np.ones(4, dtype=np.float64),
        np.array([-10.0, -10.0, z_lo], dtype=np.float64),
    ))
    param.mpc_q_ub_ = np.hstack((1e7 * np.ones(7, dtype=np.float64), np.array([10.0, 10.0, z_lo + 1.01])))
    print("param.mesh_path_ = ", param.mesh_path_)
    return param


def main():
    parser = argparse.ArgumentParser()
    add_rollout_via_args(parser)
    parser.set_defaults(obj="elephant")
    parser.add_argument(
        "--mesh-scale",
        dest="mesh_scale",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help=(
            "mesh scale factors sx sy sz used for Isaac URDF generation and contact-point optimization. "
            "Defaults to a larger sphere scale when --obj sphere."
        ),
    )
    parser.add_argument("--use-xml-texture", action="store_true", help="apply object texture parsed from env_fingertips_*.xml")
    parser.add_argument(
        "--sampling-region",
        type=float,
        default=0.5,
        help=(
            "Filter candidate contact points by world-frame object height. "
            "Positive values keep the top fraction of the object's current height range "
            "(e.g. 0.5 -> top half); negative values keep the bottom fraction "
            "(e.g. -0.5 -> bottom half); 0 disables this filter."
        ),
    )
    parser.add_argument(
        "--use_vertices",
        "--use-vertices",
        dest="use_vertices",
        type=_parse_bool_arg,
        default=False,
        help="when true, sample and project contacts on mesh vertices; otherwise use triangle face centers (true/false)",
    )
    parser.add_argument(
        "--target-p",
        dest="target_p",
        type=float,
        nargs=3,
        default=None,
        help="fixed target object position in world xyz; when omitted, this script uses the default uphill target projected onto the ramp",
    )
    parser.add_argument(
        "--target-q",
        dest="target_q",
        type=float,
        nargs=4,
        default=None,
        help="fixed target object orientation as a world-frame wxyz quaternion; ignored by default in position-only mode",
    )
    parser.add_argument(
        "--position-only-goal",
        dest="position_only_goal",
        type=_parse_bool_arg,
        default=DEFAULT_POSITION_ONLY_GOAL,
        help="when true, plan and terminate using only the target position; goal quaternion is ignored (true/false)",
    )
    parser.add_argument(
        "--set-to-goal-pose",
        dest="set_to_goal_pose",
        type=_parse_bool_arg,
        default=False,
        help="when true, initialize the object near the goal pose with small xyz and world-z yaw perturbations (true/false)",
    )
    parser.add_argument(
        "--set-to-goal-pose-xyz-noise",
        dest="set_to_goal_pose_xyz_noise",
        type=float,
        nargs=3,
        default=DEFAULT_SET_TO_GOAL_POSE_XYZ_NOISE.tolist(),
        help="per-axis world-frame xyz perturbation bounds used when --set-to-goal-pose is true",
    )
    parser.add_argument(
        "--set-to-goal-pose-yaw-noise-deg",
        dest="set_to_goal_pose_yaw_noise_deg",
        type=float,
        default=DEFAULT_SET_TO_GOAL_POSE_YAW_NOISE_DEG,
        help="max absolute world-z yaw perturbation in degrees used when --set-to-goal-pose is true",
    )
    parser.add_argument(
        "--target-offset-world",
        dest="target_offset_world",
        type=float,
        nargs=3,
        default=DEFAULT_TARGET_WORLD_OFFSET_FROM_INIT.tolist(),
        help=(
            "world-frame xyz offset from the initial object pose used to define the desired target direction. "
            "In position-only mode this offset is projected onto the ramp surface so the goal stays on the ramp."
        ),
    )
    parser.add_argument(
        "--target-ramp-local-xy",
        dest="target_ramp_local_xy",
        type=float,
        nargs=2,
        default=DEFAULT_TARGET_RAMP_LOCAL_XY.tolist(),
        help=(
            "target goal pose position in the tilted-ramp local surface frame [x, y]. "
            "Used when --target-p is not provided and --position-only-goal is false."
        ),
    )
    parser.add_argument(
        "--target-ramp-yaw-deg",
        dest="target_ramp_yaw_deg",
        type=float,
        default=DEFAULT_TARGET_RAMP_LOCAL_YAW_DEG,
        help=(
            "goal pose yaw in degrees about the ramp surface normal. "
            "Only used when --target-q is not provided."
        ),
    )
    parser.add_argument(
        "--target-local-z-rotation-offset",
        dest="target_local_z_rotation_offset",
        type=float,
        default=DEFAULT_TARGET_LOCAL_Z_ROTATION_OFFSET_RAD,
        help=(
            "default target orientation offset from the initial object pose, in radians, "
            "applied about the object's local z axis when --target-q is not provided"
        ),
    )
    parser.add_argument(
        "--set_axis_pose",
        "--set-axis-pose",
        dest="set_axis_pose",
        nargs="+",
        default=DEFAULT_AXIS_POSE_WXYZ.tolist(),
        help=(
            "7D axis pose [x, y, z, qw, qx, qy, qz] in world frame. "
            "Accepts either 7 separate numbers or a single numpy-style string."
        ),
    )
    parser.add_argument(
        "--z-axis-cost-switch-threshold",
        type=float,
        default=0.1,
        help="legacy compatibility flag; z-axis stage switching is disabled in this script",
    )
    parser.add_argument(
        "--z-axis-cost-switch-back-threshold",
        type=float,
        default=0.6,
        help="legacy compatibility flag; z-axis stage switching is disabled in this script",
    )
    parser.add_argument(
        "--anti-tip-force-max",
        type=float,
        default=0.1,
        help="upper bound on the optimizer contact force used only for the anti-tip stabilizing torque",
    )
    parser.add_argument(
        "--anti-tip-torque-max",
        type=float,
        default=0.1,
        help="norm cap on the anti-tip stabilizing torque applied to the object in world frame",
    )
    parser.add_argument(
        "--stage1-torque-pd-kp-pos",
        type=float,
        default=0.0,
        help="P gain on the optimizer-predicted local position offset x_plus during the z-axis flip stage",
    )
    parser.add_argument(
        "--stage1-torque-pd-kp-ori",
        type=float,
        default=0.2,
        help="P gain on the optimizer-predicted local orientation offset x_plus during the z-axis flip stage",
    )
    parser.add_argument(
        "--stage1-torque-pd-kd-lin",
        type=float,
        default=0.0,
        help="D gain on local linear velocity tracking error v_plus - v during the z-axis flip stage",
    )
    parser.add_argument(
        "--stage1-torque-pd-kd-ang",
        type=float,
        default=0.08,
        help="D gain on local angular velocity tracking error v_plus - v during the z-axis flip stage",
    )
    parser.add_argument(
        "--stage1-torque-scale-min",
        type=float,
        default=0.9,
        help="lower clamp for the PD-based torque scale applied to optimizer torque during the z-axis flip stage",
    )
    parser.add_argument(
        "--stage1-torque-scale-max",
        type=float,
        default=1.2,
        help="upper clamp for the PD-based torque scale applied to optimizer torque during the z-axis flip stage",
    )
    parser.add_argument(
        "--low-err-coef-flag0",
        type=float,
        default=0.75,
        help="low_err_coef used when best_contact_torque_scale_flag=0; defaults to --low_err_coef",
    )
    parser.add_argument(
        "--low-err-coef-flag1",
        type=float,
        default=0.75,
        help="low_err_coef used when best_contact_torque_scale_flag=1; defaults to --low_err_coef",
    )
    parser.add_argument(
        "--upper-err-coef-flag0",
        type=float,
        default=1.0,
        help="upper_err_coef used when best_contact_torque_scale_flag=0; defaults to --upper_err_coef",
    )
    parser.add_argument(
        "--upper-err-coef-flag1",
        type=float,
        default=1.0,
        help="upper_err_coef used when best_contact_torque_scale_flag=1; defaults to --upper_err_coef",
    )
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--viewer", dest="headless", action="store_false")
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument(
        "--show-goal-object",
        type=_parse_bool_arg,
        default=False,
        help="whether to render the semi-transparent ghost object at the goal pose (true/false)",
    )
    parser.add_argument(
        "--show-goal-pose",
        type=_parse_bool_arg,
        default=True,
        help="whether to render the goal-pose axes (true/false)",
    )
    parser.add_argument(
        "--show_point",
        "--show-point",
        dest="show_point",
        type=_parse_bool_arg,
        default=True,
        help="whether to render the yellow contact point marker and the object-frame axes (true/false)",
    )
    parser.add_argument(
        "--test",
        type=_parse_bool_arg,
        default=True,
        help="when true, keep the arm fixed and directly apply the optimizer torque to the object (true/false)",
    )
    parser.add_argument(
        "--execute-best-force",
        type=_parse_bool_arg,
        default=True,
        help="when true, turn best_force_lam into an end-effector desired force target during real execution (true/false)",
    )
    parser.add_argument("--cartesian-step", type=float, default=0.1, help="bound of xyz delta action per MPC step")
    parser.add_argument("--cartesian-joint-stiffness", type=float, default=100.0, help="internal stiffness used in explicit model")
    parser.add_argument("--cartesian-dls-lambda", type=float, default=1e-4, help="damped least squares term for J^+")
    parser.add_argument("--osc-pos-stiffness", type=float, default=12000.0, help="Cartesian position stiffness used by the Isaac OSC tracker")
    parser.add_argument("--osc-ori-stiffness", type=float, default=0.0, help="Cartesian orientation stiffness used by the Isaac OSC tracker")
    parser.add_argument("--nullspace-stiffness", type=float, default=10.0)
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
             "Same 3 mm carrot as ik2 test_mpc_isaac.py.",
    )
    parser.add_argument("--via-max-step", type=float, default=ISAAC_VIA_MAX_STEP)
    parser.add_argument("--via-smooth-rate", type=float, default=ISAAC_VIA_SMOOTH_RATE)
    parser.add_argument("--action-slew", type=float, default=ISAAC_ACTION_SLEW)
    parser.add_argument(
        "--orbit-extra",
        type=float,
        default=ISAAC_ORBIT_EXTRA,
        help="Added only to the via orbit radius (m).  "
             "path_blocked / verify_cost still use the MuJoCo keep-out.",
    )
    parser.add_argument(
        "--gpu-physx",
        action="store_true",
        help="GPU PhysX in the viewer.  Default matches ik2 CPU PhysX.",
    )
    parser.add_argument("--obj-vel-linear-process-var", type=float, default=1e-4, help="Kalman process variance for object linear velocity in world frame")
    parser.add_argument("--obj-vel-angular-process-var", type=float, default=5e-4, help="Kalman process variance for object angular velocity in world frame")
    parser.add_argument("--obj-vel-linear-measurement-var", type=float, default=5e-3, help="Kalman measurement variance for object linear velocity in world frame")
    parser.add_argument("--obj-vel-angular-measurement-var", type=float, default=2e-2, help="Kalman measurement variance for object angular velocity in world frame")
    parser.add_argument("--obj-vel-linear-deadband", type=float, default=1e-2, help="Zero out filtered object local linear velocity components below this threshold (m/s)")
    parser.add_argument("--obj-vel-angular-deadband", type=float, default=5e-2, help="Zero out filtered object local angular velocity components below this threshold (rad/s)")
    parser.add_argument(
        "--svg-screenshot-dir",
        type=str,
        default="/home/lab423/scsp/scsp-robot/examples/mpc/franka/ik2/figs",
        help="Directory to save Isaac Gym SVG screenshots. Leave empty to disable periodic capture.",
    )
    parser.add_argument(
        "--svg-screenshot-interval",
        type=float,
        default=0.2,
        help="Simulation-time seconds between Isaac Gym SVG screenshots.",
    )
    parser.add_argument(
        "--svg-screenshot-width",
        type=int,
        default=1280,
        help="Width of Isaac Gym SVG screenshots in pixels.",
    )
    parser.add_argument(
        "--svg-screenshot-height",
        type=int,
        default=960,
        help="Height of Isaac Gym SVG screenshots in pixels.",
    )
    parser.add_argument(
        "--video-output-dir",
        type=str,
        default=DEFAULT_RECORD_OUTPUT_DIR,
        help="Base directory for per-trial recording folders. Each trial saves video.mp4 and traj.json together under <obj>/<NNN>/.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=20.0,
        help="Output MP4 frame rate.",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=1280,
        help="Width of Isaac Gym MP4 frames in pixels.",
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=960,
        help="Height of Isaac Gym MP4 frames in pixels.",
    )
    parser.add_argument(
        "--traj-output-dir",
        type=str,
        default=DEFAULT_RECORD_OUTPUT_DIR,
        help="Legacy fallback base directory for per-trial recording folders when --video-output-dir is empty.",
    )
    parser.add_argument(
        "--print-object-pose",
        type=_parse_bool_arg,
        default=True,
        help="Whether to print the object xyz and quaternion during rollout (true/false).",
    )
    parser.add_argument(
        "--print-object-pose-interval",
        type=int,
        default=1,
        help="Print object pose every N rollout steps.",
    )
    parser.add_argument(
        "--success_pos_threshold",
        "--success-pos-threshold",
        dest="success_pos_threshold",
        type=float,
        default=0.02,
        help="Stop the rollout once object position error to the task target pose drops below this threshold.",
    )
    parser.add_argument(
        "--success_quat_threshold",
        "--success-quat-threshold",
        dest="success_quat_threshold",
        type=float,
        default=0.04,
        help="Stop the rollout once object quaternion error to the task target pose drops below this threshold.",
    )

    parser.add_argument(
        "--cartesian_stiffness",
        type=float,
        nargs="+",
        default=DEFAULT_CARTESIAN_STIFFNESS.tolist(),
        help="Cartesian stiffness gains; pass 1, 2, or 6 values. Default matches MuJoCo.",
    )
    parser.add_argument(
        "--cartesian_damping",
        type=float,
        nargs="+",
        default=None,
        help="Cartesian damping gains; pass 1, 2, or 6 values. Default uses critical damping from stiffness.",
    )
    parser.add_argument(
        "--effort-joint-damping",
        type=float,
        default=DEFAULT_EFFORT_JOINT_DAMPING,
        help="Passive damping applied to the 7 arm joints in Isaac effort mode.",
    )
    parser.add_argument(
        "--seed",
        "--init-rand-seed",
        dest="seed",
        type=int,
        default=15,
        help="Base random seed used to reproducibly sample init_xy_rand and init_obj_quat_rand.",
    )
    parser.add_argument("--trial-start", type=int, default=0, help="first trial id / random seed to evaluate")
    parser.add_argument("--trial-count", type=int, default=20, help="number of consecutive trials to run")

    # parser.set_defaults(**DEFAULT_ELEPHANT_TRIAL_REPLAY)
    args = parser.parse_args()
    args.solver = "acados"
    args.rollout = True
    args.init_rand_seed = int(args.seed)
    args.set_axis_pose = _parse_pose_wxyz_arg(args.set_axis_pose, "--set_axis_pose")
    args.set_to_goal_pose_xyz_noise = np.asarray(args.set_to_goal_pose_xyz_noise, dtype=np.float32).reshape(3)
    args.target_offset_world = np.asarray(args.target_offset_world, dtype=np.float32).reshape(3)
    args.target_ramp_local_xy = np.asarray(args.target_ramp_local_xy, dtype=np.float32).reshape(2)
    args.mesh_scale = None if args.mesh_scale is None else np.asarray(args.mesh_scale, dtype=np.float32).reshape(3)
    use_default_target_position = args.target_p is None
    use_default_target_orientation = args.target_q is None
    use_default_axis_pose = bool(
        args.set_axis_pose is not None
        and np.allclose(
            np.asarray(args.set_axis_pose, dtype=np.float32),
            DEFAULT_AXIS_POSE_WXYZ,
            atol=1e-6,
            rtol=0.0,
        )
    )

    if args.low_err_coef_flag0 is None:
        args.low_err_coef_flag0 = float(args.low_err_coef)
    if args.low_err_coef_flag1 is None:
        args.low_err_coef_flag1 = float(args.low_err_coef)
    if args.upper_err_coef_flag0 is None:
        args.upper_err_coef_flag0 = float(args.upper_err_coef)
    if args.upper_err_coef_flag1 is None:
        args.upper_err_coef_flag1 = float(args.upper_err_coef)

    if args.trial_start < 0:
        raise ValueError(f"trial_start must be non-negative, got {args.trial_start}")
    if args.trial_count <= 0:
        raise ValueError(f"trial_count must be positive, got {args.trial_count}")
    if args.print_object_pose_interval <= 0:
        raise ValueError(
            f"print_object_pose_interval must be positive, got {args.print_object_pose_interval}"
        )
    if args.success_pos_threshold < 0.0:
        raise ValueError(f"success_pos_threshold must be non-negative, got {args.success_pos_threshold}")
    if args.success_quat_threshold < 0.0:
        raise ValueError(f"success_quat_threshold must be non-negative, got {args.success_quat_threshold}")
    if float(args.stage1_torque_scale_min) < 0.0:
        raise ValueError(f"stage1_torque_scale_min must be non-negative, got {args.stage1_torque_scale_min}")
    if float(args.stage1_torque_scale_max) < float(args.stage1_torque_scale_min):
        raise ValueError(
            "stage1_torque_scale_max must be greater than or equal to stage1_torque_scale_min"
        )

    args.trial_num = int(args.trial_count)
    trial_start = int(args.trial_start)
    trial_num = int(args.trial_count)
    trial_stop = trial_start + trial_num
    max_rollout_length = max(1, int(getattr(args, "max_rollout_length", 5000)))
    success_rate = 0
    for trial_count in range(trial_start, trial_stop):
        param = ExplicitMPCParams(args, rand_seed=trial_count, target_type="rotation", mpc_model="explicit")
        if args.mesh_scale is not None:
            param.obj_mesh_scale_ = np.asarray(args.mesh_scale, dtype=np.float32).copy()
        elif str(args.obj).strip().lower() == "sphere":
            param.obj_mesh_scale_ = DEFAULT_SPHERE_MESH_SCALE.copy()
        else:
            param.obj_mesh_scale_ = np.ones(3, dtype=np.float32)
        if str(args.obj).strip().lower() == "sphere":
            inferred_sphere_radius = float(np.max(param.obj_mesh_scale_))
            param.object_surface_clearance_ = max(TILTED_RAMP_INIT_CLEARANCE, inferred_sphere_radius)
            param.target_surface_clearance_ = max(TILTED_RAMP_TARGET_CLEARANCE, inferred_sphere_radius)
        else:
            param.object_surface_clearance_ = float(TILTED_RAMP_INIT_CLEARANCE)
            param.target_surface_clearance_ = float(TILTED_RAMP_TARGET_CLEARANCE)
        param = _apply_dywa_physics_to_param(param)
        param = _apply_tilted_ramp_scene_to_param(
            param,
            use_default_target_position=use_default_target_position,
            use_default_target_orientation=use_default_target_orientation,
            target_ramp_local_xy=args.target_ramp_local_xy,
            target_ramp_local_yaw_deg=float(args.target_ramp_yaw_deg),
            target_world_offset=args.target_offset_world,
            position_only_goal=bool(args.position_only_goal),
            target_local_z_rotation_offset_rad=float(args.target_local_z_rotation_offset),
        )
        param.use_jax_contact_ = False
        param = adapt_param_for_cartesian_solver(param, args)
        param = _configure_rollout_param(param, args)
        param.control_substeps_ = int(args.control_substeps)
        param.nullspace_stiffness_ = float(args.nullspace_stiffness)
        param.physx_use_gpu_ = bool(getattr(args, "gpu_physx", False))
        obj_xy = np.asarray(param.init_obj_qpos_[:2], dtype=np.float64)
        toward = -obj_xy
        toward_norm = float(np.linalg.norm(toward))
        if toward_norm < 1e-6:
            toward = np.array([-1.0, 0.0], dtype=np.float64)
        else:
            toward = toward / toward_norm
        start_xy = np.array(
            [
                float(param.init_obj_qpos_[0] + 0.10 * toward[0]),
                float(param.init_obj_qpos_[1] + 0.10 * toward[1]),
            ],
            dtype=np.float64,
        )
        side_clearance = max(0.025, 0.45 * float(getattr(param, "object_circumradius", 0.06)))
        param.init_fingertip_pos_ = np.array(
            [start_xy[0], start_xy[1], _support_surface_z(param, start_xy) + side_clearance],
            dtype=np.float64,
        )
        param.osc_pos_stiffness_ = float(args.osc_pos_stiffness)
        param.osc_ori_stiffness_ = float(args.osc_ori_stiffness)
        param.cartesian_stiffness_ = np.array(args.cartesian_stiffness, dtype=np.float32)
        param.cartesian_damping_ = (
            None if args.cartesian_damping is None else np.array(args.cartesian_damping, dtype=np.float32)
        )
        param.effort_joint_damping_ = float(args.effort_joint_damping)
        param.show_goal_object_ = bool(args.show_goal_object)
        param.show_goal_pose_ = bool(args.show_goal_pose)
        param.show_point_ = bool(args.show_point)
        param.show_ghost_object_ = param.show_goal_object_
        param.test_ = bool(args.test)
        param.execute_best_force_ = bool(args.execute_best_force)
        param.svg_screenshot_dir_ = args.svg_screenshot_dir or None
        param.svg_screenshot_interval_ = float(args.svg_screenshot_interval)
        param.svg_screenshot_width_ = int(args.svg_screenshot_width)
        param.svg_screenshot_height_ = int(args.svg_screenshot_height)
        param.svg_screenshot_prefix_ = f"trial_{trial_count:03d}"
        axis_pose_wxyz = np.hstack([param.target_p_, param.target_q_]).astype(np.float32)
        param.axis_pose_ = axis_pose_wxyz.copy()
        param.lock_axis_pose_to_goal_pose_ = True
        record_output_root = args.video_output_dir or args.traj_output_dir
        record_output_dir = _allocate_trial_record_dir(record_output_root, args.obj)
        param.video_output_path_ = (
            os.path.join(record_output_dir, "video.mp4")
            if record_output_dir is not None and args.video_output_dir
            else None
        )
        traj_output_path = (
            os.path.join(record_output_dir, "traj.json")
            if record_output_dir is not None and args.traj_output_dir
            else None
        )
        param.video_fps_ = float(args.video_fps)
        param.video_width_ = int(args.video_width)
        param.video_height_ = int(args.video_height)

        if record_output_dir is not None:
            print(f"[trial {trial_count:03d}] record_dir={record_output_dir}")
        print(
            f"[trial {trial_count:03d}] tilted_ramp angle_deg={float(param.tilted_ramp_angle_deg_):.2f} "
            f"center={_format_array_for_print(param.tilted_ramp_center_)} "
            f"normal={_format_array_for_print(param.support_surface_normal_)}"
        )
        print(
            f"[trial {trial_count:03d}] obj={args.obj} "
            f"mesh_scale={_format_array_for_print(getattr(param, 'obj_mesh_scale_', np.ones(3, dtype=np.float32)))} "
            f"position_only_goal={bool(getattr(param, 'position_only_goal_', False))}"
        )

        print(
            f"[trial {trial_count:03d}] init_seed={int(param.random_seed_)} "
            f"init_xyz={_format_array_for_print(param.init_obj_qpos_[:3])} "
            f"init_quat_wxyz={_format_array_for_print(param.init_obj_qpos_[3:7])}"
        )
        if bool(getattr(param, "set_to_goal_pose_", False)):
            print(
                f"[trial {trial_count:03d}] init_from_goal_pose="
                f"{bool(param.set_to_goal_pose_)} "
                f"xyz_noise={_format_array_for_print(getattr(param, 'init_from_goal_pose_xyz_noise_', np.zeros(3, dtype=np.float32)))} "
                f"yaw_noise_deg={float(getattr(param, 'init_from_goal_pose_yaw_noise_deg_', 0.0)):.5f}"
            )
        print(
            f"[trial {trial_count:03d}] init_ramp_local_xy_seed="
            f"{_format_array_for_print(getattr(param, 'init_ramp_local_xy_seed_', np.zeros(2, dtype=np.float32)))} "
            f"resolved_init_ramp_local_xy={_format_array_for_print(getattr(param, 'init_ramp_local_xy_', np.zeros(2, dtype=np.float32)))}"
        )
        print(
            f"[trial {trial_count:03d}] target_pose xyz={_format_array_for_print(param.target_p_)} "
            f"quat_wxyz={_format_array_for_print(param.target_q_)}"
        )
        if use_default_target_orientation:
            print(
                f"[trial {trial_count:03d}] target_local_z_rotation_offset_rad="
                f"{float(param.target_local_z_rotation_offset_rad_):.5f}"
            )
        if bool(getattr(param, "position_only_goal_", False)):
            print(
                f"[trial {trial_count:03d}] target_offset_world="
                f"{_format_array_for_print(getattr(param, 'target_world_offset_', np.zeros(3, dtype=np.float32)))} "
                f"projected_to_ramp={_format_array_for_print(getattr(param, 'target_world_offset_projected_', np.zeros(3, dtype=np.float32)))}"
            )
            print(
                f"[trial {trial_count:03d}] init_ramp_local_xy={_format_array_for_print(getattr(param, 'init_ramp_local_xy_', np.zeros(2, dtype=np.float32)))} "
                f"target_ramp_local_offset={_format_array_for_print(getattr(param, 'target_ramp_local_offset_', np.zeros(2, dtype=np.float32)))} "
                f"resolved_target_ramp_local_xy={_format_array_for_print(param.target_ramp_local_xy_)}"
            )
        else:
            print(
                f"[trial {trial_count:03d}] target_ramp_local_xy={_format_array_for_print(param.target_ramp_local_xy_)} "
                f"target_ramp_yaw_deg={float(param.target_ramp_local_yaw_deg_):.2f}"
            )
        print(
            f"[trial {trial_count:03d}] visual_axis_pose xyz={_format_array_for_print(axis_pose_wxyz[:3])} "
            f"quat_wxyz={_format_array_for_print(axis_pose_wxyz[3:7])}"
        )

        env = None
        saved_video_path = None
        saved_traj_path = None
        interrupted_trial = False
        trial_success = False
        rollout_q_traj = []
        last_traj_snapshot_sample_count = -1
        target_pose_wxyz = np.hstack([param.target_p_, param.target_q_]).astype(np.float32)

        def _save_traj_snapshot(force=False):
            nonlocal saved_traj_path, last_traj_snapshot_sample_count
            if traj_output_path is None:
                return None
            if (not force) and last_traj_snapshot_sample_count == len(rollout_q_traj):
                return saved_traj_path

            saved_traj_path = record_traj(
                rollout_q_traj,
                traj_output_path,
                trial_id=trial_count,
                success=trial_success,
                interrupted=interrupted_trial,
                target_pose_wxyz=target_pose_wxyz,
                allow_overwrite=True,
            )
            last_traj_snapshot_sample_count = len(rollout_q_traj)
            if saved_traj_path is not None:
                print(
                    f"[trial {trial_count:03d}] updated traj json: {saved_traj_path} "
                    f"(samples={len(rollout_q_traj)})"
                )
            return saved_traj_path

        try:
            contact = ContactIsaacRamp(param)
            env = IsaacFrankaOSCSimulator(
                param,
                headless=args.headless,
                sim_device=args.sim_device,
                graphics_device_id=args.graphics_device_id,
            )
            env.show_target_object_pose(param.target_p_, param.target_q_)
            env.set_axis_pose(axis_pose_wxyz)
            env.hold_current_pose()

            mpc = MPCExplicitIsaac(param)
            trackers = _build_mpc_trackers(args)
            mpc_step = max(1e-4, float(getattr(args, "mpc_step_limit", 0.005)))
            via_step = getattr(args, "via_max_step", None)
            exec_step = mpc_step if via_step is None else min(mpc_step, max(1e-4, float(via_step)))

            rollout_step = 0
            policy = None
            c_now_cost = None
            pred_reduction = None
            last_accept_p_arm = False
            escape_on = False
            verify_chatter = False
            last_result = None
            position_only_goal = bool(getattr(param, "position_only_goal_", False))

            while rollout_step < max_rollout_length and not env.break_out_signal_:
                if env.dyn_paused_:
                    env.step_control_frame(draw=True, sync_realtime=True)
                    continue

                curr_q = env.get_policy_state()
                if args.print_object_pose and rollout_step % int(args.print_object_pose_interval) == 0:
                    _print_object_pose(trial_count, rollout_step, curr_q[:3], curr_q[3:7])
                _append_object_pose_sample(rollout_q_traj, curr_q)
                _save_traj_snapshot()

                pos_err_now = float(metrics.comp_pos_error(curr_q[0:3], param.target_p_))
                quat_err_now = float(metrics.comp_quat_error(curr_q[3:7], param.target_q_))
                success_reached = (
                    pos_err_now < float(args.success_pos_threshold)
                    if position_only_goal
                    else (
                        pos_err_now < float(args.success_pos_threshold)
                        and quat_err_now < float(args.success_quat_threshold)
                    )
                )
                if success_reached:
                    trial_success = True
                    saved_video_path = env.finalize_video_recording()
                    if position_only_goal:
                        print(f"[trial {trial_count:03d}] success: pos_error={pos_err_now:.6f}")
                    else:
                        print(
                            f"[trial {trial_count:03d}] success: pos_error={pos_err_now:.6f}, "
                            f"quat_error={quat_err_now:.6f}"
                        )
                    if saved_video_path is not None:
                        print(f"[trial {trial_count:03d}] saved mp4: {saved_video_path}")
                    break

                floor_z = float(param.table_height)
                table_ground = _support_air_floor(
                    param, curr_q[7:10], args.ground_height_threshold
                )
                payload, curr_q, if_contact, contact_distance = _plan_payload(
                    env, contact, table_ground,
                )
                payload["floor_z"] = floor_z
                payload["support_point"] = np.asarray(
                    getattr(param, "support_surface_point_", np.array([0.0, 0.0, floor_z])),
                    dtype=np.float64,
                ).reshape(3)
                payload["support_normal"] = np.asarray(
                    getattr(param, "support_surface_normal_", np.array([0.0, 0.0, 1.0])),
                    dtype=np.float64,
                ).reshape(3)
                if policy is not None:
                    payload["dwell"] = _dwell_payload(
                        env, args, contact, policy, last_accept_p_arm, escape_on,
                        verify_chatter, c_now_cost, pred_reduction, contact_distance,
                        pos_err_now, curr_q=curr_q, param=param,
                    )
                result = handle_mpc_request(args, param, mpc, trackers, payload)
                last_result = result
                policy = _apply_plan_result(param, None, result, curr_q, _print_rollout_step)
                c_now_cost, pred_reduction = _pred_reduction_from_policy(param, curr_q, policy)
                last_accept_p_arm = bool(policy["value_info"].get("accept_p_arm", False))
                escape_on = bool(policy["escape_on"])
                verify_chatter = bool(result["verify_chatter"])
                action = np.asarray(_clip_mpc_action(result["action"], exec_step), dtype=np.float64)
                exec_via = np.asarray(policy["mpc_virtual_point"], dtype=np.float64).reshape(3)
                env.show_point(policy["best_contact_world"])
                env.set_via_action(
                    exec_via,
                    action=action,
                    policy_dt=POLICY_INTERVAL,
                    press=policy["p_arm_world"],
                    path_blocked=bool((policy.get("value_info") or {}).get("path_blocked", False)),
                )
                n_substeps = policy_control_substeps(
                    getattr(param, "control_substeps_", 0), env.sim_dt_
                )
                for i in range(n_substeps):
                    last = (i + 1 == n_substeps)
                    env.step_control_frame(
                        draw=last or env.video_recorder_ is not None,
                        sync_realtime=last,
                    )
                print(
                    "diag_push:",
                    "step:", rollout_step,
                    "obj:", np.round(np.asarray(curr_q[:3], dtype=float), 4).tolist(),
                    "tip:", np.round(np.asarray(curr_q[7:10], dtype=float), 4).tolist(),
                    "via:", np.round(exec_via, 4).tolist(),
                    "p_arm:", np.round(np.asarray(policy["p_arm_world"], dtype=float), 4).tolist(),
                    "best:", np.round(np.asarray(policy["best_contact_world"], dtype=float), 4).tolist(),
                    "action:", np.round(action, 4).tolist(),
                    "floor_z:", round(float(floor_z), 4),
                    "ramp_tip:", round(_support_surface_z(param, curr_q[7:10]), 4),
                    "verify:", None if result.get("verify_cost") is None else round(float(result["verify_cost"]), 4),
                    "conf:", round(float(param.lambda_optimizer.contact_switch_confidence), 3),
                    "tight:", round(float(result.get("model_tightness", 0.0)), 3),
                    "accept:", int(bool(policy["value_info"].get("accept_p_arm", False))),
                    "blocked:", int(bool((policy.get("value_info") or {}).get("path_blocked", False))),
                    "phase:", (policy.get("value_info") or {}).get("via_phase"),
                    "dwell:", int(getattr(param.lambda_optimizer, "_dwell_steps", 0)),
                )
                rollout_step += 1

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

            if not trial_success:
                saved_video_path = env.finalize_video_recording()
                if saved_video_path is not None:
                    print(f"[trial {trial_count:03d}] saved mp4: {saved_video_path}")

            success_rate += 1 if trial_success else 0
        except Exception as exc:
            print(
                f"[trial {trial_count:03d}] error before completion. "
                "Saving latest trajectory snapshot before re-raising..."
            )
            if env is not None:
                try:
                    _append_object_pose_sample(rollout_q_traj, env.get_state())
                except Exception as capture_exc:
                    print(
                        f"[trial {trial_count:03d}] warning: failed to capture final pose on exception: "
                        f"{capture_exc}"
                    )
            try:
                _save_traj_snapshot(force=True)
            except Exception as save_exc:
                print(f"[trial {trial_count:03d}] warning: failed to save traj json on exception: {save_exc}")
            raise
        except KeyboardInterrupt:
            interrupted_trial = True
            print(f"\n[trial {trial_count:03d}] Ctrl+C received. Saving trajectory JSON and finalizing MP4...")
            if env is not None:
                try:
                    _append_object_pose_sample(rollout_q_traj, env.get_state())
                except Exception as exc:
                    print(f"[trial {trial_count:03d}] warning: failed to capture final pose on Ctrl+C: {exc}")
                try:
                    env.break_out_signal_ = True
                except Exception:
                    pass

            _save_traj_snapshot(force=True)

            if env is not None:
                try:
                    saved_video_path = env.finalize_video_recording()
                except Exception as exc:
                    print(f"[trial {trial_count:03d}] warning: failed to finalize MP4 on Ctrl+C: {exc}")
                    saved_video_path = None
                if saved_video_path is not None:
                    print(f"[trial {trial_count:03d}] saved mp4: {saved_video_path}")
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    print(f"[trial {trial_count:03d}] warning: cleanup failed: {exc}")

        _save_traj_snapshot(force=True)

        if interrupted_trial:
            break

    print(
        f"Success rate over {trial_num} trials "
        f"(trial ids {trial_start} to {trial_stop - 1}): "
        f"{success_rate}/{trial_num} = {success_rate/trial_num:.2%}"
    )


if __name__ == "__main__":
    main()
