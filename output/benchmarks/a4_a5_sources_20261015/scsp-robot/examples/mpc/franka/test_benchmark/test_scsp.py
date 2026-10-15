import argparse
import json
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

from examples.mpc.franka.test_flip_rot.params import ExplicitMPCParams
from examples.mpc.franka.ik2.test_mppi_isaac import (
    IsaacFrankaSimulator,
    _parse_bool_arg,
    _extract_quat_xyzw,
    _extract_vec3,
)
from planning.mpc_explicit import MPCExplicitTiltedPush as MPCExplicitIsaac
from planning.MPPIExplicit import _contact_jacobian, _franka_fk_T_jax, _tangent_basis_from_normal
from planning.mlqp_point_test_rot import LambdaContactControlOptimizer
from planning.mpc_implicit import MPCImplicit
from planning.screenshot import create_isaacgym_mp4_recorder, create_isaacgym_svg_screenshot_recorder
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
DYWA_TABLE_FRICTION_RANGE = (0.8, 1.2)
DYWA_OBJECT_FRICTION_RANGE = (1.0, 1.5)
DYWA_OBJECT_MASS_RANGE = (0.1, 0.5)

DEFAULT_CARTESIAN_STIFFNESS = np.array([2000.0, 2000.0, 2000.0, 50.0, 50.0, 50.0], dtype=np.float32)
DEFAULT_EFFORT_JOINT_DAMPING = 10.0
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
DEFAULT_RECORD_OUTPUT_DIR = "/home/lab423/scsp/scsp-robot/outputs/videos_panda"


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
    return Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix().astype(np.float32)


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


def _compute_safe_attract_point_world(contact_point_local, normal_local, pos_world, rotation_matrix, attract_point_comp):
    contact_point_world = (
        np.asarray(rotation_matrix, dtype=np.float32) @ np.asarray(contact_point_local, dtype=np.float32).reshape(3)
    ) + np.asarray(pos_world, dtype=np.float32).reshape(3)
    offset_world = -float(attract_point_comp) * (
        np.asarray(rotation_matrix, dtype=np.float32) @ np.asarray(normal_local, dtype=np.float32).reshape(3)
    )
    offset_world = np.asarray(offset_world, dtype=np.float32).reshape(3)
    offset_world[2] = max(float(offset_world[2]), 0.0)
    attract_point_world = contact_point_world + offset_world
    attract_point_world[2] = max(float(attract_point_world[2]), float(contact_point_world[2]))
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
    force_local = np.asarray(
        info.get("contact_force_local", np.zeros(3, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(3)
    torque_local = np.asarray(
        info.get("contact_torque_local", np.zeros(3, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(3)
    force_world = (np.asarray(r_obj_to_world, dtype=np.float32) @ force_local).astype(np.float32)
    torque_world = (np.asarray(r_obj_to_world, dtype=np.float32) @ torque_local).astype(np.float32)
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


def _set_actor_mass(gym, env, actor_handle, mass):
    body_props = gym.get_actor_rigid_body_properties(env, actor_handle)
    for prop in body_props:
        prop.mass = float(mass)
    gym.set_actor_rigid_body_properties(env, actor_handle, body_props, True)


def _apply_dywa_physics_to_param(param):
    rng = getattr(param, "random_generator_", None)
    if rng is None:
        rng = np.random.default_rng()
    param.h_ = DYWA_SIM_DT
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
    param.table_friction_ = float(rng.uniform(*DYWA_TABLE_FRICTION_RANGE))
    param.object_friction_ = float(rng.uniform(*DYWA_OBJECT_FRICTION_RANGE))
    param.fingertip_friction_ = float(param.object_friction_)
    param.obj_mass_ = float(rng.uniform(*DYWA_OBJECT_MASS_RANGE))
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
        self.video_recorder_ = None
        self.show_goal_pose_ = bool(
            getattr(self.param_, "show_goal_pose_", getattr(self.param_, "show_ghost_object_", False))
        )
        # Keep the legacy field in sync because inherited helpers still read it.
        self.show_ghost_object_ = self.show_goal_pose_

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
        self.object_friction_ = float(getattr(self.param_, "object_friction_", 0.5))
        self.fingertip_friction_ = float(getattr(self.param_, "fingertip_friction_", self.object_friction_))
        self.obj_mass_ = float(getattr(self.param_, "obj_mass_", 0.1))
        _set_actor_friction(self.gym, self.env, self.table_actor, self.table_friction_)
        _set_actor_friction(self.gym, self.env, self.obj_actor, self.object_friction_)
        _set_actor_friction(self.gym, self.env, self.franka_actor, self.fingertip_friction_)
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
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

    def reset_mj_env(self):
        super().reset_mj_env()
        self._hide_best_contact_force_visual()
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
        if self.viewer_ is not None or self.svg_screenshot_recorder_ is not None or self.video_recorder_ is not None:
            self.gym.step_graphics(self.sim)
            graphics_stepped = True

        if self.svg_screenshot_recorder_ is not None:
            self.svg_screenshot_recorder_.capture_if_due(
                sim_time=float(self.gym.get_sim_time(self.sim)),
                step_graphics=not graphics_stepped,
            )

        if self.video_recorder_ is not None:
            self.video_recorder_.capture_if_due(
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

    def show_point(self, goal_pos=None):
        marker_state = self.gym.get_actor_rigid_body_states(self.env, self.p_arm_marker_actor, gymapi.STATE_ALL)
        if goal_pos is not None:
            marker_state["pose"]["p"][0] = (float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2]))
        marker_state["vel"]["linear"][0] = (0.0, 0.0, 0.0)
        marker_state["vel"]["angular"][0] = (0.0, 0.0, 0.0)
        self.gym.set_actor_rigid_body_states(self.env, self.p_arm_marker_actor, marker_state, gymapi.STATE_ALL)

    def show_target_object_pose(self, goal_pos=None, goal_quat_wxyz=None):
        if goal_pos is not None:
            self._target_obj_pos_cache = np.asarray(goal_pos, dtype=np.float32).copy()
        if goal_quat_wxyz is not None:
            self._target_obj_quat_xyzw_cache = _quat_wxyz_to_xyzw_np(goal_quat_wxyz)
        self._sync_target_object_visual()
        self._sync_pose_axes()

    def set_goal_pose_visibility(self, visible):
        visible = bool(visible)
        self.show_goal_pose_ = visible
        self.show_ghost_object_ = visible
        self._sync_target_object_visual()
        self._sync_pose_axes()

    def set_ghost_object_visibility(self, visible):
        self.set_goal_pose_visibility(visible)

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
        obj_pos, obj_quat_xyzw = self._get_actor_body_pose(self.obj_actor, 0)
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
        return self._track_desired_pose(preserve_nullspace_target=True)

    def step_hold(self, desired_force_world=None, desired_torque_world=None):
        p_curr, r_curr = self.get_end_effector_pos()
        self.p_d = p_curr.copy()
        self.R_d = r_curr.copy()
        self.R_d_hold = r_curr.copy()
        self.set_tool_compensation_wrench(
            force_world=desired_force_world,
            torque_world=desired_torque_world,
        )
        return self._track_desired_pose(preserve_nullspace_target=True)

    def step_passive(self):
        self.set_tool_compensation_wrench(force_world=None, torque_world=None)
        self._effort_control.zero_()
        if self.franka_dof_count >= 9:
            self.gym.set_actor_dof_position_targets(self.env, self.franka_actor, self._joint_targets)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
        self._simulate_once()
        return np.zeros(7, dtype=np.float32)

    def step(self, cmd, desired_force_world=None, desired_torque_world=None):
        cmd = np.asarray(cmd, dtype=np.float32).reshape(-1)
        self.set_tool_compensation_wrench(
            force_world=desired_force_world,
            torque_world=desired_torque_world,
        )
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
        self.finalize_video_recording()
        if self.svg_screenshot_recorder_ is not None:
            self.svg_screenshot_recorder_.close()
            self.svg_screenshot_recorder_ = None
        super().close()


class ContactIsaacCartesian:
    def __init__(self, param):
        self.param_ = param

    def detect_once(self, simulator: IsaacFrankaOSCSimulator):
        full_q = simulator.get_state()
        full_q = np.asarray(full_q, dtype=np.float32)
        obj_pos = full_q[0:3]
        obj_quat_wxyz = full_q[3:7]

        nv = self.param_.n_qvel_
        max_ncon = self.param_.max_ncon_
        phi_vec = np.ones((max_ncon * 4,), dtype=np.float32)
        jac_mat = np.zeros((max_ncon * 4, nv), dtype=np.float32)
        jac_mat_env = np.zeros((max_ncon * 4, nv), dtype=np.float32)
        robot_contact_force_map = np.zeros((3, max_ncon * 4), dtype=np.float32)
        con_pos_list = []
        if_contact = False
        if_fingertip_contact = False

        contacts = simulator.get_physx_contacts()
        mu = float(self.param_.mu_object_)
        contact_sep_threshold = float(getattr(self.param_, "if_contact_separation_threshold_", 0.0))
        row_idx = 0
        row_env_idx = 0
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
            other_is_fingertip = other_sim_idx == simulator.fingertip_body_sim_idx
            # Isaac rigid contacts may include proximity pairs with positive separation.
            # We only report a true robot-object contact when the pair is actually
            # touching / penetrating according to PhysX's separation convention.
            if other_is_franka and sep <= contact_sep_threshold:
                if_contact = True
            if other_is_fingertip and sep <= contact_sep_threshold:
                if_fingertip_contact = True
            if other_is_franka and row_idx < max_ncon:
                phi_vec[4 * row_idx : 4 * row_idx + 4] = 0.5 * sep
                jac_mat[4 * row_idx : 4 * row_idx + 4, :] = con_jac
                robot_contact_force_map[:, 4 * row_idx : 4 * row_idx + 4] = _contact_force_edge_directions_world(
                    n,
                    t1,
                    t2,
                    mu,
                )
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

        return phi_vec, jac_mat, con_pos_list, jac_mat_env, robot_contact_force_map, if_contact, if_fingertip_contact


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
    param.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7), np.array([-1.0, -1.0, table_height + 0.02])))
    param.mpc_q_ub_ = np.hstack((1e7 * np.ones(7), np.array([1.5, 1.0, 1.5])))
    # Match the MuJoCo object geometry scale so the interaction point optimizer and
    # visible avoidance distance are comparable across simulators.
    print("param.mesh_path_ = ", param.mesh_path_ )
    param.lambda_optimizer = LambdaContactControlOptimizer(
        mesh_path=param.mesh_path_,
        obj_mass=param.obj_mass_,
        arm_friction=param.mu_object_,
        contact_stiffness=param.model_params,
        time_step=param.h_,
        sample_num=args.sample_num,
        pos_coef=args.pos_coef,
        ori_coef=args.ori_coef,
        nlp_solver=args.mlqp_solver,
        scale_factors=[1.0]*3,
        use_vertices=args.use_vertices,
    )
    param.sol_guess_ = None
    return param


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default="teapot", help="name of obj")
    parser.add_argument("--use-xml-texture", action="store_true", help="apply object texture parsed from env_fingertips_*.xml")
    parser.add_argument("--attract_coef", type=float, default=0.5, help="coef of attract function")
    parser.add_argument("--reject_coef", type=float, default=0.01, help="coef of reject function")
    parser.add_argument("--contact_coef", type=float, default=0.5, help="coef of contact function")
    parser.add_argument("--contact_cost_param", type=float, default=0.0, help="mass center or project point attract")
    parser.add_argument("--model_param", type=float, default=7, help="model param")
    parser.add_argument("--reject_dis", type=float, default=0.01, help="reject radius")
    parser.add_argument("--attract_point_comp", type=float, default=0.08, help="distance compensation of attract point")
    parser.add_argument("--ground_height_threshold", type=float, default=0.33, help="threshold of sample points height")
    parser.add_argument(
        "--sampling-region",
        type=float,
        default=-0.5,
        help=(
            "Filter candidate contact points by world-frame object height. "
            "Positive values keep the top fraction of the object's current height range "
            "(e.g. 0.5 -> top half); negative values keep the bottom fraction "
            "(e.g. -0.5 -> bottom half); 0 disables this filter."
        ),
    )
    parser.add_argument("--sample_num", type=int, default=70, help="number of sample point")
    parser.add_argument(
        "--use_vertices",
        "--use-vertices",
        dest="use_vertices",
        type=_parse_bool_arg,
        default=False,
        help="when true, sample and project contacts on mesh vertices; otherwise use triangle face centers (true/false)",
    )
    parser.add_argument("--pos_coef", type=float, default=1, help="coef of position cost in mlqp_point")
    parser.add_argument("--ori_coef", type=float, default=0.005, help="coef of orientation cost in mlqp_point")
    parser.add_argument(
        "--mlqp-solver",
        type=str,
        choices=["acados", "ipopt"],
        default="acados",
        help=(
            "Solver backend used inside planning/mlqp_point_test_rot.py; "
            "'acados' runs a single-stage nonlinear solve through acados_template."
        ),
    )
    parser.add_argument(
        "--target-p",
        dest="target_p",
        type=float,
        nargs=3,
        default=None,
        help="fixed target object position in world xyz; defaults to [0.45, 0.0, init_height - 0.02]",
    )
    parser.add_argument(
        "--target-q",
        dest="target_q",
        type=float,
        nargs=4,
        default=None,
        help="fixed target object orientation as a world-frame wxyz quaternion; defaults to [1, 0, 0, 0]",
    )
    parser.add_argument(
        "--set_axis_pose",
        "--set-axis-pose",
        dest="set_axis_pose",
        nargs="+",
        default=[0.42849, 0.11289, 0.38614, 0.62768, -0.10893, -0.10086,  0.76419],
        help=(
            "7D axis pose [x, y, z, qw, qx, qy, qz] in world frame. "
            "Accepts either 7 separate numbers or a single numpy-style string."
        ),
    )
    parser.add_argument(
        "--z-axis-cost-switch-threshold",
        type=float,
        default=0.1,
        help="enter the position + full-quaternion refinement stage once z_axis_cost drops below this value",
    )
    parser.add_argument(
        "--z-axis-cost-switch-back-threshold",
        type=float,
        default=0.6,
        help=(
            "leave the position + full-quaternion refinement stage only after z_axis_cost rises above this value; "
            "defaults to 2x --z-axis-cost-switch-threshold"
        ),
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
    parser.add_argument("--low_err_coef", type=float, default=0.75, help="coef of delta error")
    parser.add_argument("--upper_err_coef", type=float, default=0.75, help="coef of delta error")
    parser.add_argument(
        "--low-err-coef-flag0",
        type=float,
        default=0.75,
        help="low_err_coef used when best_contact_torque_scale_flag=0; defaults to --low_err_coef",
    )
    parser.add_argument(
        "--low-err-coef-flag1",
        type=float,
        default=0.3,
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
        default=0.75,
        help="upper_err_coef used when best_contact_torque_scale_flag=1; defaults to --upper_err_coef",
    )
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--viewer", dest="headless", action="store_false")
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument(
        "--show-goal-pose",
        type=_parse_bool_arg,
        default=False,
        help="whether to render the goal-pose axes and semi-transparent ghost object (true/false)",
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
    parser.add_argument("--cartesian-joint-stiffness", type=float, default=2000.0, help="internal stiffness used in explicit model")
    parser.add_argument("--cartesian-dls-lambda", type=float, default=1e-4, help="damped least squares term for J^+")
    parser.add_argument("--osc-pos-stiffness", type=float, default=1000.0, help="Cartesian position stiffness used by the Isaac OSC tracker")
    parser.add_argument("--osc-ori-stiffness", type=float, default=100.0, help="Cartesian orientation stiffness used by the Isaac OSC tracker")
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
    args.init_rand_seed = int(args.seed)
    args.set_axis_pose = _parse_pose_wxyz_arg(args.set_axis_pose, "--set_axis_pose")

    if args.z_axis_cost_switch_back_threshold is None:
        args.z_axis_cost_switch_back_threshold = 2.0 * float(args.z_axis_cost_switch_threshold)
    if float(args.z_axis_cost_switch_back_threshold) < float(args.z_axis_cost_switch_threshold):
        raise ValueError(
            "z_axis_cost_switch_back_threshold must be greater than or equal to "
            "z_axis_cost_switch_threshold to create hysteresis."
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

    trial_start = int(args.trial_start)
    trial_num = int(args.trial_count)
    trial_stop = trial_start + trial_num
    max_rollout_length = 5000
    success_rate = 0
    for trial_count in range(trial_start, trial_stop):
        param = ExplicitMPCParams(args, rand_seed=trial_count, target_type="rotation", mpc_model="explicit")
        param = _apply_dywa_physics_to_param(param)
        param.use_jax_contact_ = False
        param = adapt_param_for_cartesian_solver(param, args)
        param.osc_pos_stiffness_ = float(args.osc_pos_stiffness)
        param.osc_ori_stiffness_ = float(args.osc_ori_stiffness)
        param.cartesian_stiffness_ = np.array(args.cartesian_stiffness, dtype=np.float32)
        param.cartesian_damping_ = (
            None if args.cartesian_damping is None else np.array(args.cartesian_damping, dtype=np.float32)
        )
        param.effort_joint_damping_ = float(args.effort_joint_damping)
        param.show_goal_pose_ = bool(args.show_goal_pose)
        param.show_ghost_object_ = param.show_goal_pose_
        param.test_ = bool(args.test)
        param.execute_best_force_ = bool(args.execute_best_force)
        param.svg_screenshot_dir_ = args.svg_screenshot_dir or None
        param.svg_screenshot_interval_ = float(args.svg_screenshot_interval)
        param.svg_screenshot_width_ = int(args.svg_screenshot_width)
        param.svg_screenshot_height_ = int(args.svg_screenshot_height)
        param.svg_screenshot_prefix_ = f"trial_{trial_count:03d}"
        axis_pose_wxyz = (
            args.set_axis_pose.copy()
            if args.set_axis_pose is not None
            else np.hstack([param.target_p_, param.target_q_]).astype(np.float32)
        )
        param.axis_pose_ = axis_pose_wxyz.copy()
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
            f"[trial {trial_count:03d}] init_seed={int(param.random_seed_)} "
            f"init_xyz={_format_array_for_print(param.init_obj_qpos_[:3])} "
            f"init_quat_wxyz={_format_array_for_print(param.init_obj_qpos_[3:7])}"
        )
        print(
            f"[trial {trial_count:03d}] target_pose xyz={_format_array_for_print(param.target_p_)} "
            f"quat_wxyz={_format_array_for_print(param.target_q_)}"
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
            contact = ContactIsaacCartesian(param)
            env = IsaacFrankaOSCSimulator(
                param,
                headless=args.headless,
                sim_device=args.sim_device,
                graphics_device_id=args.graphics_device_id,
            )
            env.show_target_object_pose(param.target_p_, param.target_q_)
            env.set_axis_pose(axis_pose_wxyz)

            mpc = MPCExplicitIsaac(param) if param.mpc_model == "explicit" else MPCImplicit(param)
            obj_velocity_filter = VelocityKalmanFilter6D(
                process_variance=np.array(
                    3 * [args.obj_vel_linear_process_var] + 3 * [args.obj_vel_angular_process_var],
                    dtype=np.float32,
                ),
                measurement_variance=np.array(
                    3 * [args.obj_vel_linear_measurement_var] + 3 * [args.obj_vel_angular_measurement_var],
                    dtype=np.float32,
                ),
            )

            rollout_step = 0
            verify_cost = 0
            current_x = np.zeros(7, dtype=np.float32)
            current_x[3] = 1.0

            prev_action = np.zeros((param.n_cmd_,), dtype=np.float32)
            consecutive_detect_time = 0
            consecutive_contact_time = 0
            use_full_pose_stage = False
            best_contact_torque_scale_flag = 0

            while rollout_step < max_rollout_length and not env.break_out_signal_:
                    if not env.dyn_paused_:
                        curr_q = env.get_state()
                        if args.print_object_pose and rollout_step % int(args.print_object_pose_interval) == 0:
                            _print_object_pose(trial_count, rollout_step, curr_q[:3], curr_q[3:7])
                        _append_object_pose_sample(rollout_q_traj, curr_q)
                        _save_traj_snapshot()
                        ee_pos = env.get_end_effector_pos()[0].copy()
                        curr_x_solver = np.hstack([curr_q[:7], ee_pos]).astype(np.float32)
                        (
                            phi_vec,
                            jac_mat,
                            con_point,
                            jac_mat_env,
                            robot_contact_force_map,
                            if_contact,
                            if_fingertip_contact,
                        ) = contact.detect_once(env)
                        quaternion = [curr_q[4], curr_q[5], curr_q[6], curr_q[3]]
                        r_obj_to_world = Rotation.from_quat(quaternion).as_matrix()
                        obj_linear_velocity_world, obj_angular_velocity_world = env.get_object_velocity_world()
                        v_obj_world_raw = np.hstack([obj_linear_velocity_world, obj_angular_velocity_world]).astype(np.float32)
                        v_obj_world_filtered = obj_velocity_filter.update(v_obj_world_raw)
                        # This Isaac Gym setup uses a Z-up world frame and rigid-body quaternions in xyzw.
                        # Rotation.from_quat(quaternion) gives R_obj_to_world, so world-frame linear/angular
                        # velocities must be rotated back with R^T to match the optimizer's object-local frame.
                        v_obj = np.hstack([
                            r_obj_to_world.T @ v_obj_world_filtered[:3],
                            r_obj_to_world.T @ v_obj_world_filtered[3:],
                        ]).astype(np.float32)
                        v_obj = _apply_velocity_deadband(
                            v_obj,
                            linear_deadband=args.obj_vel_linear_deadband,
                            angular_deadband=args.obj_vel_angular_deadband,
                        )
                        gravity = np.hstack([r_obj_to_world.T @ param.gravity_[:3] * param.obj_mass_, np.zeros(3)])
                        effective_target_p = param.target_p_.copy()

                        target_pos_ = effective_target_p - curr_q[0:3]
                        target_quat_local = rotations.quaternion_multiply(
                            rotations.quaternion_conjugate(curr_q[3:7]),
                            param.target_q_,
                        )
                        target_pose_local = np.hstack([r_obj_to_world.T @ target_pos_, target_quat_local])
                        z_axis_cost = _z_axis_alignment_cost_wxyz(curr_q[3:7], param.target_q_)
                        if use_full_pose_stage:
                            if z_axis_cost >= float(args.z_axis_cost_switch_back_threshold):
                                use_full_pose_stage = False
                                best_contact_torque_scale_flag = 0
                        elif z_axis_cost <= float(args.z_axis_cost_switch_threshold):
                            use_full_pose_stage = True
                            best_contact_torque_scale_flag = 1

                        param.lambda_optimizer.update_Jacobian(jac_mat_env)
                        visible_point_idx = param.lambda_optimizer.get_availble_point_idx(
                            curr_q[0:3], r_obj_to_world, effective_target_p, args.ground_height_threshold
                        )
                        visible_point_idx = _filter_point_indices_to_side_or_top(
                            param.lambda_optimizer,
                            r_obj_to_world,
                            visible_point_idx,
                        )
                        if not use_full_pose_stage:
                            # visible_point_idx = _filter_point_indices_to_upper_third(
                            #     param.lambda_optimizer,
                            #     curr_q[0:3],
                            #     r_obj_to_world,
                            #     visible_point_idx,
                            #     sampling_region=args.sampling_region,
                            # )
                            visible_point_idx = _filter_point_indices_to_local_positive_z(
                                param.lambda_optimizer,
                                visible_point_idx,
                            )

                        if use_full_pose_stage:
                            visible_point_idx = _filter_point_indices_to_upper_third(
                                param.lambda_optimizer,
                                curr_q[0:3],
                                r_obj_to_world,
                                visible_point_idx,
                                sampling_region=args.sampling_region,
                            )

                        st = time.time()
                        best_contact_point, normal, min_error, max_error, curr_ori_coef = param.lambda_optimizer.choose_contact_points(
                            target_pose_local,
                            current_x,
                            gravity,
                            visible_point_idx,
                            use_full_pose_objective=use_full_pose_stage,
                        )
                        # print("time = ", time.time() - st)
                        _, _, _, _, best_contact_force_info, _ = param.lambda_optimizer.optimize_control_input(
                            target_pose_local,
                            current_x,
                            gravity,
                            best_contact_point,
                            use_full_pose_objective=use_full_pose_stage,
                            r_obj_to_world=r_obj_to_world,
                        )
                        best_contact_point_world = r_obj_to_world @ best_contact_point + curr_q[0:3]
                        best_contact_force_world, _ = _contact_wrench_world_from_optimizer_info(
                            best_contact_force_info,
                            r_obj_to_world,
                        )
                        best_contact_torque_world = None
                        # if use_full_pose_stage:
                            # Keep a separate z-axis-only stabilizing torque active during
                            # the refinement stage so the object is less likely to tip back over.
                        _, _, _, _, _, best_contact_torque_world = param.lambda_optimizer.optimize_stabilizing_torque(
                            target_pose_local,
                            current_x,
                            gravity,
                            best_contact_point,
                            r_obj_to_world=r_obj_to_world,
                            lam_upper_bound=args.anti_tip_force_max,
                            torque_norm_upper_bound=args.anti_tip_torque_max,
                        )

                        attract_point = best_contact_point.copy()
                        attract_point_world = _compute_safe_attract_point_world(
                            contact_point_local=attract_point,
                            normal_local=normal,
                            pos_world=curr_q[0:3],
                            rotation_matrix=r_obj_to_world,
                            attract_point_comp=args.attract_point_comp,
                        )

                        local_point = r_obj_to_world.T @ (ee_pos - curr_q[0:3])
                        p_arm_local, _, x_plus_opt, error, info, p_arm_torque_world = param.lambda_optimizer.optimize_control_input(
                            target_pose_local,
                            current_x,
                            gravity,
                            local_point,
                            use_full_pose_objective=use_full_pose_stage,
                            r_obj_to_world=r_obj_to_world,
                        )
                        env.show_best_contact_force(p_arm_torque_world, p_arm_torque_world)
                        p_arm_world = r_obj_to_world @ p_arm_local + curr_q[:3]
                        p_arm_force_world, _ = _contact_wrench_world_from_optimizer_info(
                            info,
                            r_obj_to_world,
                        )

                        visual_point = attract_point_world if not verify_cost else p_arm_world
                        env.show_point(visual_point)
                        if best_contact_torque_scale_flag:
                            low_err_coef = float(args.low_err_coef_flag1)
                            upper_err_coef = float(args.upper_err_coef_flag1)
                        else:
                            low_err_coef = float(args.low_err_coef_flag0)
                            upper_err_coef = float(args.upper_err_coef_flag0)
                        if not verify_cost and np.linalg.norm(ee_pos - attract_point_world) < 5e-2:
                            low_err_coef *= 1.1

                        eps = 1e-6
                        delta_error = max_error - min_error
                        delta_error = max(delta_error, eps)

                        # norm_err = (error - min_error) / delta_error
                        rel_impr = (max_error - error)

                        # 条件1：相对于未接触有一定提升
                        impr_th = (max_error - min_error)
                        if impr_th == 0:
                            improvement = 0
                        else:
                            improvement = rel_impr / impr_th

                        if not verify_cost:
                            upper_err_coef = max(upper_err_coef, 0.7)
                            if np.linalg.norm(ee_pos[:2] - attract_point_world[:2]) < 5e-2:
                                upper_err_coef *= 0.8

                        consecutive_contact_time = consecutive_contact_time + int(if_contact) if verify_cost else 0
                        if (not verify_cost and improvement > upper_err_coef) or (verify_cost and improvement <= low_err_coef):
                            consecutive_detect_time += 1
                        else:
                            consecutive_detect_time = 0

                        if not verify_cost and consecutive_detect_time >= 5:
                            verify_cost = 1
                        elif verify_cost and consecutive_detect_time >= 5 and consecutive_contact_time >= 10:
                            verify_cost = 0

                        plan_like_ik2 = bool(best_contact_torque_scale_flag)
                        execute_best_force_now = bool(args.execute_best_force and (not use_full_pose_stage))
                        plan_desired_force_world = None if plan_like_ik2 else p_arm_force_world
                        plan_robot_contact_force_map = None if plan_like_ik2 else robot_contact_force_map
                        plan_execute_best_force = False if plan_like_ik2 else execute_best_force_now
                        plan_u_lb = -0.01 if plan_like_ik2 else -0.05
                        plan_u_ub = 0.01 if plan_like_ik2 else 0.05
                        sol = mpc.plan_once(
                            effective_target_p,
                            param.target_q_,
                            curr_x_solver,
                            phi_vec,
                            jac_mat,
                            verify_cost_param=verify_cost,
                            virtual_point=attract_point_world,
                            contact_point=p_arm_world,
                            curr_ori_coef=curr_ori_coef,
                            use_full_pose_terminal_cost=plan_like_ik2,
                            sol_guess=param.sol_guess_,
                            desired_force_world=plan_desired_force_world,
                            robot_contact_force_map=plan_robot_contact_force_map,
                            execute_best_force=plan_execute_best_force,
                            u_lb=plan_u_lb,
                            u_ub=plan_u_ub,
                        )
                        param.sol_guess_ = sol["sol_guess"]
                        raw_action = np.asarray(sol["action"], dtype=np.float32)
                        action = raw_action.copy()
                        st1 = time.time()

                        p_arm_v_plus_local = np.asarray(
                            info.get("resulting_velocity", np.zeros(6, dtype=np.float32)),
                            dtype=np.float32,
                        ).reshape(6)

                        desired_force_world = None
                        if execute_best_force_now:
                            desired_force_world = p_arm_force_world.copy()
                            if np.linalg.norm(desired_force_world) < 1e-6:
                                desired_force_world = best_contact_force_world.copy()
                        applied_object_torque_world = None
                        if best_contact_torque_world is not None:
                            best_contact_torque_scale = 0.2 if best_contact_torque_scale_flag else 1.0
                            applied_object_torque_world = (
                                best_contact_torque_scale
                                * np.asarray(best_contact_torque_world, dtype=np.float32).copy()
                            )
                        if args.test and p_arm_torque_world is not None and if_fingertip_contact:
                            p_arm_torque_world = np.asarray(p_arm_torque_world, dtype=np.float32)
                            if not use_full_pose_stage:
                                stage1_torque_scale = _compute_stage1_pd_torque_scale(
                                    x_plus_local=x_plus_opt,
                                    v_plus_local=p_arm_v_plus_local,
                                    current_v_local=v_obj,
                                    args=args,
                                )
                                p_arm_torque_world = stage1_torque_scale * p_arm_torque_world
                            if applied_object_torque_world is None:
                                applied_object_torque_world = p_arm_torque_world.copy()
                            else:
                                applied_object_torque_world = applied_object_torque_world + p_arm_torque_world
                        env.apply_object_wrench_world(torque_world=applied_object_torque_world)
                        # env.apply_object_wrench_world(
                        #         torque_world=60
                        #         * np.asarray(best_contact_torque_world, dtype=np.float32).copy()
                        #     )
                        # action = np.ones(3) * 0.1
                        env.step(action, desired_force_world=desired_force_world)
                        # print("action = ", action, time.time() - st1)
                        rollout_step += 1

                        curr_q = env.get_state()
                        _append_object_pose_sample(rollout_q_traj, curr_q)
                        _save_traj_snapshot()
                        pos_error = metrics.comp_pos_error(curr_q[0:3], param.target_p_)
                        quat_error = metrics.comp_quat_error(curr_q[3:7], param.target_q_)
                        if pos_error < float(args.success_pos_threshold) and quat_error < float(args.success_quat_threshold):
                            trial_success = True
                            saved_video_path = env.finalize_video_recording()
                            print(
                                f"[trial {trial_count:03d}] success: pos_error={pos_error:.6f}, "
                                f"quat_error={quat_error:.6f}"
                            )
                            if saved_video_path is not None:
                                print(f"[trial {trial_count:03d}] saved mp4: {saved_video_path}")
                            break

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
