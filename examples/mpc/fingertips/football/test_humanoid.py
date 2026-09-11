import argparse
import os
import sys
import tempfile
import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(parent_dir)
mujoco_mpc_python_dir = os.path.join(parent_dir, "mujoco_mpc/python")
if mujoco_mpc_python_dir not in sys.path:
    sys.path.append(mujoco_mpc_python_dir)

from examples.mpc.fingertips.football.params import ExplicitMPCParams
from examples.mpc.fingertips.football.params_2 import ExplicitMPCParams2
from mujoco_mpc.demos.predictive_sampling import predictive_sampling
from planning.mpc_explicit_foorball import (
    HumanoidFootActionLimiter,
    MPCExplicitFootBall,
    compute_limited_foot_yaws,
)
from planning.mlqp_point import LambdaContactControlOptimizer
from utils import rotations

BALL_RADIUS = 0.07
BALL_START_POS = np.array([0.95, 0.0, BALL_RADIUS], dtype=np.float64)
GOAL_START_POS = np.array([1.45, 0.0, BALL_RADIUS], dtype=np.float64)
MARKER_START_POS = np.array([BALL_START_POS[0], BALL_START_POS[1], BALL_START_POS[2] + 0.05], dtype=np.float64)
INITIAL_ROOT_HEIGHT = 1.60
RECOVERY_ROOT_HEIGHT = 1.68
TRACKING_POINT_NAMES = (
    "pelvis",
    "head",
    "left_shin",
    "right_shin",
    "left_hand",
    "right_hand",
    "left_upper_arm",
    "right_upper_arm",
    "left_foot",
    "right_foot",
)


def quat_wxyz_to_mat(quat_wxyz):
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    return Rotation.from_quat(quat_xyzw).as_matrix()


def mat_to_quat_wxyz(mat):
    quat_xyzw = Rotation.from_matrix(mat).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def orientation_error(current_mat, desired_mat):
    rotvec = Rotation.from_matrix(desired_mat @ current_mat.T).as_rotvec()
    return rotvec.astype(np.float64)


def yaw_from_matrix(rot_mat):
    x_axis = rot_mat[:, 0]
    return float(np.arctan2(x_axis[1], x_axis[0]))


class HumanoidPredictiveController:
    def __init__(self, env):
        self.env = env
        self.nominal_pelvis_height = float(env.data_.body("pelvis").xpos[2])
        support_center = 0.5 * (env.data_.body("left_foot").xpos + env.data_.body("right_foot").xpos)
        self.nominal_pelvis_rel_support = float(env.data_.body("pelvis").xpos[2] - support_center[2])
        self.nominal_pelvis_xy_offset = env.data_.body("pelvis").xpos[:2] - support_center[:2]
        self.nominal_offsets = {
            name: env.data_.body(name).xpos.copy() - env.data_.body("pelvis").xpos.copy()
            for name in TRACKING_POINT_NAMES
            if name not in {"left_foot", "right_foot", "pelvis"}
        }
        self.reference_pos = {}
        self.reference_vel = {name: np.zeros(3, dtype=np.float64) for name in TRACKING_POINT_NAMES}
        self.prev_left_target = env.left_foot_target.copy()
        self.prev_right_target = env.right_foot_target.copy()
        ctrl_span = np.maximum(env.ctrl_high - env.ctrl_low, 1e-3)
        self.planner = predictive_sampling.Planner(
            env.model_,
            reward=self._reward,
            horizon=0.28,
            splinestep=0.04,
            planstep=env.model_.opt.timestep,
            nsample=18,
            noise_scale=float(np.mean(ctrl_span) * 0.08),
            nimprove=2,
            interp="zero",
            limits=True,
        )
        self.update_references(env.left_foot_target, env.right_foot_target)

    def update_references(self, left_target, right_target):
        dt = self.env.model_.opt.timestep * max(self.env.param_.frame_skip_, 1)
        pelvis_target = 0.5 * (left_target + right_target)
        pelvis_target[:2] += self.nominal_pelvis_xy_offset
        pelvis_target[2] = 0.5 * (left_target[2] + right_target[2]) + self.nominal_pelvis_rel_support

        self.reference_pos["pelvis"] = pelvis_target
        for name, offset in self.nominal_offsets.items():
            self.reference_pos[name] = pelvis_target + offset
        self.reference_pos["left_foot"] = left_target.copy()
        self.reference_pos["right_foot"] = right_target.copy()

        self.reference_vel["left_foot"] = (left_target - self.prev_left_target) / max(dt, 1e-6)
        self.reference_vel["right_foot"] = (right_target - self.prev_right_target) / max(dt, 1e-6)
        for name in TRACKING_POINT_NAMES:
            if name not in {"left_foot", "right_foot"}:
                self.reference_vel[name].fill(0.0)
        self.prev_left_target = left_target.copy()
        self.prev_right_target = right_target.copy()

    def _body_linear_velocity(self, data, body_name):
        return data.body(body_name).cvel[3:].copy()

    def _reward(self, model, data):
        joint_vel_cost = 0.001 * np.dot(data.qvel[6:], data.qvel[6:])
        ctrl_cost = 0.05 * np.dot(data.ctrl, data.ctrl)

        current_pos = {name: data.body(name).xpos.copy() for name in TRACKING_POINT_NAMES}
        avg_ref = np.mean([self.reference_pos[name] for name in TRACKING_POINT_NAMES], axis=0)
        avg_cur = np.mean([current_pos[name] for name in TRACKING_POINT_NAMES], axis=0)
        pos_cost = 100.0 * np.dot(avg_ref - avg_cur, avg_ref - avg_cur)

        body_weights = {
            "pelvis": 30.0,
            "head": 5.0,
            "left_shin": 20.0,
            "right_shin": 20.0,
            "left_hand": 4.0,
            "right_hand": 4.0,
            "left_upper_arm": 4.0,
            "right_upper_arm": 4.0,
            "left_foot": 45.0,
            "right_foot": 45.0,
        }
        vel_weights = {
            "pelvis": 0.1,
            "head": 0.05,
            "left_shin": 0.1,
            "right_shin": 0.1,
            "left_hand": 0.05,
            "right_hand": 0.05,
            "left_upper_arm": 0.05,
            "right_upper_arm": 0.05,
            "left_foot": 0.25,
            "right_foot": 0.25,
        }
        vel_cost = 0.0
        for name in TRACKING_POINT_NAMES:
            ref_rel = self.reference_pos[name] - avg_ref
            cur_rel = current_pos[name] - avg_cur
            rel_err = ref_rel - cur_rel
            pos_cost += body_weights[name] * np.dot(rel_err, rel_err)

            vel_err = self.reference_vel[name] - self._body_linear_velocity(data, name)
            vel_cost += vel_weights[name] * np.dot(vel_err, vel_err)

        foot_cost = 120.0 * np.dot(current_pos["left_foot"] - self.reference_pos["left_foot"], current_pos["left_foot"] - self.reference_pos["left_foot"])
        foot_cost += 120.0 * np.dot(current_pos["right_foot"] - self.reference_pos["right_foot"], current_pos["right_foot"] - self.reference_pos["right_foot"])

        pelvis_pos = current_pos["pelvis"]
        support_center = 0.5 * (current_pos["left_foot"] + current_pos["right_foot"])
        balance_err = pelvis_pos[:2] - support_center[:2]
        balance_cost = 60.0 * np.dot(balance_err, balance_err)
        balance_cost += 80.0 * (pelvis_pos[2] - self.reference_pos["pelvis"][2]) ** 2

        pelvis_mat = data.body("pelvis").xmat.reshape(3, 3)
        upright_cost = 45.0 * np.dot(pelvis_mat[:2, 2], pelvis_mat[:2, 2])
        left_foot_mat = data.body("left_foot").xmat.reshape(3, 3)
        right_foot_mat = data.body("right_foot").xmat.reshape(3, 3)
        upright_cost += 8.0 * np.dot(left_foot_mat[:2, 2], left_foot_mat[:2, 2])
        upright_cost += 8.0 * np.dot(right_foot_mat[:2, 2], right_foot_mat[:2, 2])

        posture_cost = 0.0
        for joint_name, q_des in self.env.nominal_qpos.items():
            q = data.joint(joint_name).qpos[0]
            posture_cost += 4.0 * (q - q_des) ** 2

        total_cost = joint_vel_cost + ctrl_cost + pos_cost + vel_cost + foot_cost + balance_cost + upright_cost + posture_cost
        return -float(total_cost)

    def replan(self):
        baseline = self.env._compute_taskspace_ctrl().copy()
        self.planner.policy._parameters[:] = baseline[:, None]
        act = self.env.data_.act.copy() if self.env.model_.na else np.zeros(0, dtype=np.float64)
        mocap_pos = self.env.data_.mocap_pos.copy() if self.env.model_.nmocap else np.zeros((0, 3), dtype=np.float64)
        mocap_quat = self.env.data_.mocap_quat.copy() if self.env.model_.nmocap else np.zeros((0, 4), dtype=np.float64)
        self.planner.improve_policy(
            self.env.data_.qpos.copy(),
            self.env.data_.qvel.copy(),
            act,
            float(self.env.data_.time),
            mocap_pos,
            mocap_quat,
        )

    def action(self):
        return self.planner.action_from_policy(float(self.env.data_.time)).copy()


def build_humanoid_football_xml():
    base_path = os.path.join(parent_dir, "envs/robots/assets/mjcf/nv_humanoid.xml")
    with open(base_path, "r", encoding="ascii") as f:
        xml = f.read()

    asset_block = """
  <asset>
    <texture type="2d" name="groundplane" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.12 0.18 0.22" width="300" height="300"/>
    <material name="grid" texture="groundplane" texrepeat="8 8" texuniform="true" reflectance="0.1"/>
    <material name="self" rgba="0.72 0.58 0.42 1"/>
    <material name="football_mat" rgba="0.92 0.75 0.28 1" specular="0.2" shininess="0.1"/>
    <material name="goal_mat" rgba="0 1 0 0.25"/>
    <material name="marker_mat" rgba="1 0 0 1"/>
  </asset>
"""
    world_insert = """
    <body name="goal" pos="1.45 0.0 0.07">
      <geom name="goal_geom" type="sphere" size="0.07" material="goal_mat" contype="0" conaffinity="0"/>
    </body>
    <body name="obj" pos="0.95 0.0 0.07">
      <freejoint name="football_root"/>
      <geom name="obj" type="sphere" size="0.07" material="football_mat" mass="0.12" condim="6" friction="1.2 0.05 0.01"/>
    </body>
    <body name="marker" pos="0.95 0.0 0.12">
      <geom name="marker_geom" type="sphere" size="0.02" material="marker_mat" contype="0" conaffinity="0"/>
    </body>
"""
    xml = xml.replace("<default>", asset_block + "\n  <default>", 1)
    xml = xml.replace("</worldbody>", world_insert + "\n  </worldbody>", 1)
    return xml


class HumanoidFootballEnv:
    def __init__(self, param, render=True):
        self.param_ = param
        self.render = render
        self.limiter = HumanoidFootActionLimiter()

        xml_text = build_humanoid_football_xml()
        tmp = tempfile.NamedTemporaryFile("w", suffix="_humanoid_football.xml", delete=False, dir="/tmp", encoding="ascii")
        tmp.write(xml_text)
        tmp.close()
        self.model_path = tmp.name

        self.model_ = mujoco.MjModel.from_xml_path(self.model_path)
        self.data_ = mujoco.MjData(self.model_)
        self.viewer_ = None

        self.left_foot_body_id = self.model_.body("left_foot").id
        self.right_foot_body_id = self.model_.body("right_foot").id
        self.pelvis_body_id = self.model_.body("pelvis").id
        self.obj_body_id = self.model_.body("obj").id
        self.obj_joint_id = self.model_.joint("football_root").id
        self.obj_dof_adr = int(self.model_.jnt_dofadr[self.obj_joint_id])
        self.obj_qpos_adr = int(self.model_.jnt_qposadr[self.obj_joint_id])

        self.actuator_joint_ids = self.model_.actuator_trnid[:, 0].astype(np.int32)
        self.actuator_dof_ids = np.array([self.model_.jnt_dofadr[jid] for jid in self.actuator_joint_ids], dtype=np.int32)
        self.actuator_gears = self.model_.actuator_gear[:, 0].copy()
        self.ctrl_low = self.model_.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model_.actuator_ctrlrange[:, 1].copy()

        self.nominal_qpos = {}
        self.posture_kp = {}
        self.posture_kd = {}
        self.nominal_left_foot_mat = np.eye(3)
        self.nominal_right_foot_mat = np.eye(3)
        self.foot_geom_names = {"left_foot", "right_foot"}
        self.floor_geom_names = {"floor"}
        self.controller = None
        self.step_count = 0
        self.warmup_steps = 40

        self.reset_mj_env()

        self.pelvis_target_pos = self.data_.body("pelvis").xpos.copy()
        self.pelvis_target_mat = self.data_.body("pelvis").xmat.reshape(3, 3).copy()
        self.left_foot_target = self.data_.body("left_foot").xpos.copy()
        self.right_foot_target = self.data_.body("right_foot").xpos.copy()
        self.left_foot_target_mat = self.data_.body("left_foot").xmat.reshape(3, 3).copy()
        self.right_foot_target_mat = self.data_.body("right_foot").xmat.reshape(3, 3).copy()
        self.controller = HumanoidPredictiveController(self)
        if render:
            self.viewer_ = mujoco.viewer.launch_passive(self.model_, self.data_)
            self.viewer_.sync()

    def reset_mj_env(self):
        mujoco.mj_resetData(self.model_, self.data_)
        self.step_count = 0

        self.data_.joint("root").qpos[:] = np.array([0.0, 0.0, INITIAL_ROOT_HEIGHT, 1.0, 0.0, 0.0, 0.0])
        self.data_.joint("abdomen_x").qpos[:] = 0.0
        self.data_.joint("abdomen_y").qpos[:] = -0.04
        self.data_.joint("abdomen_z").qpos[:] = 0.0

        stance = {
            "right_hip_x": -0.02,
            "right_hip_y": -0.30,
            "right_hip_z": 0.0,
            "right_knee": -0.62,
            "right_ankle_x": 0.01,
            "right_ankle_y": 0.32,
            "left_hip_x": 0.02,
            "left_hip_y": -0.30,
            "left_hip_z": 0.0,
            "left_knee": -0.62,
            "left_ankle_x": -0.01,
            "left_ankle_y": 0.32,
            "right_shoulder1": 0.10,
            "right_shoulder2": -0.20,
            "right_elbow": -0.45,
            "left_shoulder1": -0.10,
            "left_shoulder2": -0.20,
            "left_elbow": -0.45,
        }
        for joint_name, value in stance.items():
            self.data_.joint(joint_name).qpos[:] = value

        self.data_.joint("football_root").qpos[:] = np.hstack([BALL_START_POS, [1.0, 0.0, 0.0, 0.0]])
        self.data_.qvel[:] = 0.0
        mujoco.mj_forward(self.model_, self.data_)

        self.nominal_left_foot_mat = self.data_.body("left_foot").xmat.reshape(3, 3).copy()
        self.nominal_right_foot_mat = self.data_.body("right_foot").xmat.reshape(3, 3).copy()

        self.nominal_qpos = {
            joint_name: float(self.data_.joint(joint_name).qpos)
            for joint_name in stance.keys() | {"abdomen_x", "abdomen_y", "abdomen_z"}
        }

        for joint_name in self.nominal_qpos:
            if "hip" in joint_name:
                self.posture_kp[joint_name] = 180.0
            elif "knee" in joint_name:
                self.posture_kp[joint_name] = 220.0
            elif "ankle" in joint_name:
                self.posture_kp[joint_name] = 90.0
            elif "abdomen" in joint_name:
                self.posture_kp[joint_name] = 140.0
            else:
                self.posture_kp[joint_name] = 40.0
            self.posture_kd[joint_name] = 2.0 * np.sqrt(self.posture_kp[joint_name])

        self.pelvis_target_pos = self.data_.body("pelvis").xpos.copy()
        self.pelvis_target_mat = self.data_.body("pelvis").xmat.reshape(3, 3).copy()
        self.left_foot_target = self.data_.body("left_foot").xpos.copy()
        self.right_foot_target = self.data_.body("right_foot").xpos.copy()
        self.left_foot_target_mat = self.nominal_left_foot_mat.copy()
        self.right_foot_target_mat = self.nominal_right_foot_mat.copy()
        self._raise_root_for_clearance(min_pelvis_height=0.98, extra_clearance=0.03)
        self._stabilize_stance(steps=240)
        self._recover_initial_stance()
        if self.controller is not None:
            self.controller.update_references(self.left_foot_target, self.right_foot_target)

    def get_state(self):
        ball_qpos = self.data_.qpos[self.obj_qpos_adr:self.obj_qpos_adr + 7].copy()
        left_pos = self.data_.body("left_foot").xpos.copy()
        right_pos = self.data_.body("right_foot").xpos.copy()
        return np.hstack([ball_qpos, left_pos, right_pos])

    def set_goal(self, goal_pos=None, goal_quat=None):
        if goal_pos is not None:
            self.model_.body("goal").pos = goal_pos
        if goal_quat is not None:
            self.model_.body("goal").quat = goal_quat
        mujoco.mj_forward(self.model_, self.data_)

    def show_target(self, goal_pos=None):
        if goal_pos is not None:
            self.model_.body("marker").pos = goal_pos
        mujoco.mj_forward(self.model_, self.data_)

    def _body_pose(self, body_name):
        body = self.data_.body(body_name)
        return body.xpos.copy(), body.xmat.reshape(3, 3).copy()

    def _body_jacobian(self, body_id, point):
        jacp = np.zeros((3, self.model_.nv))
        jacr = np.zeros((3, self.model_.nv))
        mujoco.mj_jac(self.model_, self.data_, jacp=jacp, jacr=jacr, point=point, body=body_id)
        return jacp, jacr

    def _joint_index(self, joint_name):
        return int(self.model_.jnt_dofadr[self.model_.joint(joint_name).id])

    def _apply_posture_pd(self, tau):
        for joint_name, q_des in self.nominal_qpos.items():
            dof_id = self._joint_index(joint_name)
            q = self.data_.joint(joint_name).qpos[0]
            qd = self.data_.qvel[dof_id]
            tau[dof_id] += self.posture_kp[joint_name] * (q_des - q) - self.posture_kd[joint_name] * qd

    def _compute_taskspace_ctrl(self):
        tau = np.zeros(self.model_.nv, dtype=np.float64)
        self._apply_posture_pd(tau)
        self._apply_pelvis_task(tau)
        self._apply_foot_task(tau, "left_foot", self.left_foot_target, self.left_foot_target_mat)
        self._apply_foot_task(tau, "right_foot", self.right_foot_target, self.right_foot_target_mat)
        ctrl = tau[self.actuator_dof_ids] / np.maximum(self.actuator_gears, 1e-6)
        return np.clip(ctrl, self.ctrl_low, self.ctrl_high)

    def _stabilize_stance(self, steps=120):
        for _ in range(steps):
            self.data_.ctrl[:] = self._compute_taskspace_ctrl()
            mujoco.mj_step(self.model_, self.data_)
            if self.viewer_ is not None:
                self.viewer_.sync()

    def _raise_root_for_clearance(self, min_pelvis_height=0.95, extra_clearance=0.02):
        pelvis_pos = self.data_.body("pelvis").xpos.copy()
        left_foot_pos = self.data_.body("left_foot").xpos.copy()
        right_foot_pos = self.data_.body("right_foot").xpos.copy()
        support_height = 0.5 * (left_foot_pos[2] + right_foot_pos[2])
        desired_pelvis_height = max(min_pelvis_height, support_height + 0.86)
        delta_z = desired_pelvis_height - pelvis_pos[2] + extra_clearance
        if delta_z <= 0.0:
            return
        root_qpos = self.data_.joint("root").qpos.copy()
        root_qpos[2] += delta_z
        self.data_.joint("root").qpos[:] = root_qpos
        mujoco.mj_forward(self.model_, self.data_)
        self.pelvis_target_pos = self.data_.body("pelvis").xpos.copy()
        self.left_foot_target = self.data_.body("left_foot").xpos.copy()
        self.right_foot_target = self.data_.body("right_foot").xpos.copy()

    def _recover_initial_stance(self):
        if self.data_.body("pelvis").xpos[2] > 0.98:
            return
        root_qpos = self.data_.joint("root").qpos.copy()
        root_qpos[:3] = np.array([0.0, 0.0, RECOVERY_ROOT_HEIGHT], dtype=np.float64)
        self.data_.joint("root").qpos[:] = root_qpos
        self.data_.qvel[:] = 0.0
        mujoco.mj_forward(self.model_, self.data_)
        self._raise_root_for_clearance(min_pelvis_height=1.02, extra_clearance=0.05)
        self.pelvis_target_pos = self.data_.body("pelvis").xpos.copy()
        self.pelvis_target_mat = self.data_.body("pelvis").xmat.reshape(3, 3).copy()
        self.left_foot_target = self.data_.body("left_foot").xpos.copy()
        self.right_foot_target = self.data_.body("right_foot").xpos.copy()
        self.left_foot_target_mat = self.data_.body("left_foot").xmat.reshape(3, 3).copy()
        self.right_foot_target_mat = self.data_.body("right_foot").xmat.reshape(3, 3).copy()
        self._stabilize_stance(steps=180)

    def _apply_pelvis_task(self, tau):
        pelvis_pos, pelvis_mat = self._body_pose("pelvis")
        jacp, jacr = self._body_jacobian(self.pelvis_body_id, pelvis_pos)
        vel_p = jacp @ self.data_.qvel
        vel_r = jacr @ self.data_.qvel
        pos_err = self.pelvis_target_pos - pelvis_pos
        rot_err = orientation_error(pelvis_mat, self.pelvis_target_mat)
        pelvis_force = 420.0 * pos_err - 55.0 * vel_p
        pelvis_torque = 240.0 * rot_err - 30.0 * vel_r
        tau += jacp.T @ pelvis_force + jacr.T @ pelvis_torque

    def _is_unstable(self):
        pelvis_pos, pelvis_mat = self._body_pose("pelvis")
        support_center = 0.5 * (self.data_.body("left_foot").xpos + self.data_.body("right_foot").xpos)
        tilt = np.linalg.norm(pelvis_mat[:2, 2])
        return (
            pelvis_pos[2] < 0.72
            or pelvis_pos[2] - support_center[2] < 0.58
            or tilt > 0.45
        )

    def _apply_foot_task(self, tau, foot_name, foot_target, foot_target_mat):
        body_id = self.model_.body(foot_name).id
        foot_pos, foot_mat = self._body_pose(foot_name)
        jacp, jacr = self._body_jacobian(body_id, foot_pos)
        vel_p = jacp @ self.data_.qvel
        vel_r = jacr @ self.data_.qvel
        pos_err = foot_target - foot_pos
        rot_err = orientation_error(foot_mat, foot_target_mat)
        foot_force = 520.0 * pos_err - 55.0 * vel_p
        foot_torque = 140.0 * rot_err - 16.0 * vel_r
        tau += jacp.T @ foot_force + jacr.T @ foot_torque

    def _set_limited_foot_orientations(self):
        pelvis_mat = self.data_.body("pelvis").xmat.reshape(3, 3).copy()
        pelvis_yaw = yaw_from_matrix(pelvis_mat)
        ball_pos = self.data_.body("obj").xpos.copy()
        left_pos = self.data_.body("left_foot").xpos.copy()
        right_pos = self.data_.body("right_foot").xpos.copy()
        left_yaw, right_yaw = compute_limited_foot_yaws(ball_pos, pelvis_yaw, left_pos, right_pos)

        left_nominal_yaw = yaw_from_matrix(self.nominal_left_foot_mat)
        right_nominal_yaw = yaw_from_matrix(self.nominal_right_foot_mat)
        left_delta = left_yaw - left_nominal_yaw
        right_delta = right_yaw - right_nominal_yaw

        self.left_foot_target_mat = Rotation.from_euler("z", left_delta).as_matrix() @ self.nominal_left_foot_mat
        self.right_foot_target_mat = Rotation.from_euler("z", right_delta).as_matrix() @ self.nominal_right_foot_mat

    def step(self, foot_delta_cmd):
        foot_delta_cmd = self.limiter.clip_delta(foot_delta_cmd)
        if self.step_count < self.warmup_steps:
            foot_delta_cmd *= float(self.step_count + 1) / float(self.warmup_steps)
        curr_state = self.get_state()
        pelvis_pos = self.data_.body("pelvis").xpos.copy()

        left_target = curr_state[7:10] + foot_delta_cmd[:3]
        right_target = curr_state[10:13] + foot_delta_cmd[3:]
        left_target, right_target = self.limiter.clip_world_targets(left_target, right_target, pelvis_pos)
        self.left_foot_target = left_target
        self.right_foot_target = right_target
        self._set_limited_foot_orientations()
        self.controller.update_references(self.left_foot_target, self.right_foot_target)
        use_sampling = not self._is_unstable()
        if use_sampling:
            self.controller.replan()

        for _ in range(self.param_.frame_skip_):
            if use_sampling:
                self.data_.ctrl[:] = self.controller.action()
            else:
                self.data_.ctrl[:] = self._compute_taskspace_ctrl()
            mujoco.mj_step(self.model_, self.data_)
            if self.viewer_ is not None:
                self.viewer_.sync()
        self.step_count += 1


class ReducedHumanoidContact:
    def __init__(self, param, env):
        self.param_ = param
        self.env = env

    def _virtual_foot_jacobian(self, body_name):
        jac = np.zeros((3, self.param_.n_qvel_))
        if body_name == "left_foot":
            jac[:, 6:9] = np.eye(3)
        elif body_name == "right_foot":
            jac[:, 9:12] = np.eye(3)
        return jac

    def _ball_reduced_jacobian(self, point):
        jacp = np.zeros((3, self.env.model_.nv))
        mujoco.mj_jac(self.env.model_, self.env.data_, jacp=jacp, jacr=None, point=point, body=self.env.obj_body_id)
        reduced = np.zeros((3, self.param_.n_qvel_))
        reduced[:, :6] = jacp[:, self.env.obj_dof_adr:self.env.obj_dof_adr + 6]
        return reduced

    def detect_once(self, simulator):
        mujoco.mj_forward(simulator.model_, simulator.data_)
        mujoco.mj_collision(simulator.model_, simulator.data_)

        con_phi_list = []
        con_pos_list = []
        con_jac_list = []
        con_jac_env_list = []
        con_phi_env_list = []

        n_con = simulator.data_.ncon
        contacts = simulator.data_.contact
        ball_pos = simulator.data_.body("obj").xpos.copy()
        ball_mat = simulator.data_.body("obj").xmat.reshape(3, 3).copy()

        for i in range(n_con):
            contact_i = contacts[i]
            geom1_name = mujoco.mj_id2name(simulator.model_, mujoco.mjtObj.mjOBJ_GEOM, contact_i.geom1)
            geom2_name = mujoco.mj_id2name(simulator.model_, mujoco.mjtObj.mjOBJ_GEOM, contact_i.geom2)
            body1_name = simulator.model_.body(simulator.model_.geom_bodyid[contact_i.geom1]).name
            body2_name = simulator.model_.body(simulator.model_.geom_bodyid[contact_i.geom2]).name

            if geom1_name != "obj" and geom2_name != "obj":
                continue

            con_pos = contact_i.pos.copy()
            con_dist = contact_i.dist * 0.5
            con_mu = self.param_.mu_object_
            con_frame = contact_i.frame.reshape((-1, 3)).T
            con_frame_pmd = np.hstack((con_frame, -con_frame[:, -2:]))

            ball_jac = self._ball_reduced_jacobian(con_pos)
            other_jac = np.zeros((3, self.param_.n_qvel_))
            other_name = body2_name if geom1_name == "obj" else body1_name
            other_geom_name = geom2_name if geom1_name == "obj" else geom1_name
            if other_name in {"left_foot", "right_foot"}:
                other_jac = self._virtual_foot_jacobian(other_name)

            if geom1_name == "obj":
                con_jacp = -(con_frame_pmd.T @ (other_jac - ball_jac))
            else:
                con_jacp = con_frame_pmd.T @ (ball_jac - other_jac)

            con_jac = con_jacp[0] + con_mu * con_jacp[1:]
            con_phi_list.append(con_dist)
            con_jac_list.append(con_jac)

            con_pos_body = ball_mat.T @ (con_pos - ball_pos)
            if other_geom_name in {"floor"}:
                con_phi_env_list.append(con_dist)
                con_jac_env_list.append(con_jac)
                con_pos_list.append(con_pos_body)
            else:
                con_pos_list.append(con_pos)

        phi_vec, jac_mat = self.reformat(con_phi_list, con_jac_list)
        _, jac_mat_env = self.reformat(con_phi_env_list, con_jac_env_list)
        return phi_vec, jac_mat, con_pos_list, jac_mat_env

    def reformat(self, con_phi_list, con_jac_list):
        phi_vec = np.ones((self.param_.max_ncon_ * 4,))
        jac_mat = np.zeros((self.param_.max_ncon_ * 4, self.param_.n_mj_v_))
        for i in range(len(con_phi_list)):
            phi_vec[4 * i:4 * i + 4] = con_phi_list[i]
            jac_mat[4 * i:4 * i + 4] = con_jac_list[i]
        return phi_vec, jac_mat


def calculate_verify_cost(
    optimizer,
    target_pose,
    current_x,
    gravity,
    arm_point,
    verify_cost,
    low_err_coef,
    upper_err_coef,
    args,
    max_error,
    min_error,
    world_rot,
    state,
    attract_point_world,
):
    p_arm_local, x_plus_opt, error, info = optimizer.optimize_control_input(target_pose, current_x, gravity, arm_point)
    p_arm_world = world_rot @ p_arm_local + state[:3]

    if verify_cost:
        low_err_coef = args.low_err_coef
    elif np.linalg.norm(state[7:10] - attract_point_world) < 5e-2:
        low_err_coef *= 1.1

    delta_error = max_error - min_error
    adaptive = delta_error * upper_err_coef if verify_cost else delta_error * low_err_coef
    verify_cost = 1 if error < (min_error + adaptive) else 0
    return verify_cost, p_arm_world, error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default="football")
    parser.add_argument("--attract_coef", type=int, default=3)
    parser.add_argument("--reject_coef", type=float, default=1)
    parser.add_argument("--contact_coef", type=float, default=0.7)
    parser.add_argument("--contact_cost_param", type=float, default=0)
    parser.add_argument("--model_param", type=float, default=7)
    parser.add_argument("--reject_dis", type=float, default=0.01)
    parser.add_argument("--attract_point_comp", type=float, default=0.04)
    parser.add_argument("--ground_height_threshold", type=float, default=0.012)
    parser.add_argument("--sample_num", type=int, default=70)
    parser.add_argument("--pos_coef", type=float, default=1)
    parser.add_argument("--ori_coef", type=float, default=0)
    parser.add_argument("--low_err_coef", type=float, default=0.6)
    parser.add_argument("--upper_err_coef", type=float, default=0.9)
    parser.add_argument("--trial_num", type=int, default=1)
    parser.add_argument("--max_rollout_length", type=int, default=1500)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    target_pos_set = [
        [1.45, 0.0, BALL_RADIUS],
        [1.35, 0.18, BALL_RADIUS],
        [1.35, -0.18, BALL_RADIUS],
    ]

    for trial_count in range(args.trial_num):
        param = ExplicitMPCParams(args, rand_seed=trial_count, target_type="ground-rotation", model="explicit")
        param_2 = ExplicitMPCParams2(args, rand_seed=trial_count, target_type="ground-rotation", model="explicit")
        env = HumanoidFootballEnv(param, render=args.render)
        contact = ReducedHumanoidContact(param, env)
        mpc = MPCExplicitFootBall(param)
        mpc_2 = MPCExplicitFootBall(param_2)

        current_x = np.zeros(7)
        current_x[3] = 1.0
        vc1 = 0
        vc2 = 0
        point_idx = 0
        param.target_p_ = target_pos_set[point_idx]
        param_2.target_p_ = target_pos_set[point_idx]
        env.set_goal(param.target_p_, param.target_q_)

        low_err_coef = args.low_err_coef
        upper_err_coef = args.upper_err_coef

        for rollout_step in range(args.max_rollout_length):
            curr_q = env.get_state()
            phi_vec, jac_mat, con_point, jac_mat_env = contact.detect_once(env)

            quat_wxyz = curr_q[3:7]
            r_obj_to_world = quat_wxyz_to_mat(quat_wxyz)
            gravity = np.hstack([r_obj_to_world.T @ param.gravity_[:3] * param.obj_mass_, np.zeros(3)])

            target_pos_ = np.asarray(param.target_p_) - curr_q[0:3]
            target_quat_local = rotations.quaternion_multiply(
                rotations.quaternion_conjugate(curr_q[3:7]),
                param.target_q_,
            )
            target_pose_local = np.hstack([r_obj_to_world.T @ target_pos_, target_quat_local])

            param.lambda_optimizer.update_Jacobian(jac_mat_env)
            visible_point_idx = param.lambda_optimizer.get_availble_point_idx(
                curr_q[0:3],
                r_obj_to_world,
                np.asarray(param.target_p_),
                args.ground_height_threshold,
            )
            best_contact_point, normal, min_error, max_error, curr_ori_coef = param.lambda_optimizer.choose_contact_points(
                target_pose_local,
                current_x,
                gravity,
                visible_point_idx,
            )

            attract_point_world = r_obj_to_world @ best_contact_point + curr_q[0:3]
            attract_point_world -= args.attract_point_comp * (r_obj_to_world @ normal)
            env.show_target(attract_point_world)

            local_point1 = r_obj_to_world.T @ (curr_q[7:10] - curr_q[0:3])
            local_point2 = r_obj_to_world.T @ (curr_q[10:13] - curr_q[0:3])

            vc1, pw1, err1 = calculate_verify_cost(
                param.lambda_optimizer,
                target_pose_local,
                current_x,
                gravity,
                local_point1,
                vc1,
                low_err_coef,
                upper_err_coef,
                args,
                max_error,
                min_error,
                r_obj_to_world,
                curr_q,
                attract_point_world,
            )
            vc2, pw2, err2 = calculate_verify_cost(
                param.lambda_optimizer,
                target_pose_local,
                current_x,
                gravity,
                local_point2,
                vc2,
                low_err_coef,
                upper_err_coef,
                args,
                max_error,
                min_error,
                r_obj_to_world,
                curr_q,
                attract_point_world,
            )

            if vc1:
                alter_point = True
            elif vc2:
                alter_point = False
            else:
                alter_point = np.sum((curr_q[7:10] - curr_q[:3]) ** 2) < np.sum((curr_q[10:13] - curr_q[:3]) ** 2)

            sol = mpc.plan_once(
                param.target_p_,
                param.target_q_,
                curr_q,
                phi_vec,
                jac_mat,
                verify_cost_param=vc1,
                virtual_point=attract_point_world,
                contact_point=pw1,
                curr_ori_coef=curr_ori_coef,
                sol_guess=param.sol_guess_,
            )
            param.sol_guess_ = sol["sol_guess"]
            action_1 = sol["action"]

            sol_2 = mpc_2.plan_once(
                param.target_p_,
                param.target_q_,
                curr_q,
                phi_vec,
                jac_mat,
                verify_cost_param=vc2,
                virtual_point=attract_point_world,
                contact_point=pw2,
                curr_ori_coef=curr_ori_coef,
                sol_guess=param_2.sol_guess_,
            )
            param_2.sol_guess_ = sol_2["sol_guess"]
            action_2 = sol_2["action"]

            action = action_1.copy() if alter_point else action_2.copy()

            if alter_point:
                normal_vec = curr_q[0:2] - curr_q[7:9]
                if np.linalg.norm(normal_vec) > 1e-6:
                    normal_vec = normal_vec / np.linalg.norm(normal_vec)
                    x_d = curr_q[0:2] + 0.18 * normal_vec
                    force = -0.45 * (curr_q[10:12] - x_d)
                    action[3:5] = force
            else:
                normal_vec = curr_q[0:2] - curr_q[10:12]
                if np.linalg.norm(normal_vec) > 1e-6:
                    normal_vec = normal_vec / np.linalg.norm(normal_vec)
                    x_d = curr_q[0:2] + 0.18 * normal_vec
                    force = -0.45 * (curr_q[7:9] - x_d)
                    action[0:2] = force

            env.step(action)

            if rollout_step % 50 == 0:
                print(
                    f"step={rollout_step} "
                    f"ball={np.round(curr_q[:3], 3)} "
                    f"left={np.round(curr_q[7:10], 3)} "
                    f"right={np.round(curr_q[10:13], 3)}"
                )


if __name__ == "__main__":
    main()
