import argparse
import os
import re
import sys

import numpy as np

from scipy.spatial.transform import Rotation

from isaacgym import gymapi, gymtorch
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(parent_dir)

from examples.mpc.franka.ik2.params import ExplicitMPCParams
from examples.mpc.franka.ik2.test_mppi_isaac import (
    IsaacFrankaSimulator,
    _extract_quat_xyzw,
    _extract_vec3,
)
from planning.MPCExplicit_isaac import MPCExplicitIsaac
from planning.MPPIExplicit import _contact_jacobian, _tangent_basis_from_normal
from planning.mlqp_point1 import LambdaContactControlOptimizer
from planning.mpc_implicit import MPCImplicit
from utils import metrics, rotations
import time 

def _quat_xyzw_to_matrix(quat_xyzw):
    return Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix().astype(np.float32)


def _mjcf_quat_wxyz_to_urdf_rpy(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    return Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)


class IsaacFrankaOSCSimulator(IsaacFrankaSimulator):
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

    def _configure_franka(self):
        dof_props = self.gym.get_actor_dof_properties(self.env, self.franka_actor)
        dof_props["driveMode"][:7] = gymapi.DOF_MODE_EFFORT
        dof_props["stiffness"][:7] = 0.0
        dof_props["damping"][:7] = 0.0

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
        self._effort_limits = np.array(dof_props["effort"][:7], dtype=np.float32)

        self.cartesian_stiffness = np.diag([800.0, 800.0, 800.0, 50.0, 50.0, 50.0]).astype(np.float32)
        self.cartesian_damping = (2.0 * np.sqrt(self.cartesian_stiffness)).astype(np.float32)
        self.kp_null = float(getattr(self.param_, "osc_kp_null_", 10.0))
        self.kd_null = 2.0 * np.sqrt(self.kp_null)
        self.home_q = np.array(self.param_.init_robot_qpos_, dtype=np.float32)
        self.low_height = float(self.param_.table_height + 0.02)
        self.world_down_axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.tilt_stiffness = float(getattr(self.param_, "osc_tilt_stiffness_", 30.0))
        self.tilt_damping = float(getattr(self.param_, "osc_tilt_damping_", 2.0 * np.sqrt(self.tilt_stiffness)))

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
        self._refresh_osc_tensors()
        p, r = self.get_end_effector_pos()
        self.position_d = p.copy()
        self.orientation_d = r.copy()
        self.p_d = p.copy()
        self.R_d = r.copy()
        self._init_downward_constraint(r)

    def _init_downward_constraint(self, rotation_matrix):
        axis_scores = rotation_matrix.T @ self.world_down_axis
        best_axis = int(np.argmax(np.abs(axis_scores)))
        axis_sign = 1.0 if axis_scores[best_axis] >= 0.0 else -1.0
        self.ee_down_axis_local = np.zeros(3, dtype=np.float32)
        self.ee_down_axis_local[best_axis] = axis_sign

    def _get_body_pose(self, local_idx):
        states = self.gym.get_actor_rigid_body_states(self.env, self.franka_actor, gymapi.STATE_POS)
        p = _extract_vec3(states["pose"]["p"][local_idx])
        quat_xyzw = _extract_quat_xyzw(states["pose"]["r"][local_idx])
        r = _quat_xyzw_to_matrix(quat_xyzw)
        return p, r

    def get_end_effector_pos(self):
        p_tip, r_tip = self._get_body_pose(self.fingertip_body_idx)
        _, r_hand = self._get_body_pose(self.ee_body_local_idx)
        return p_tip.astype(np.float32), r_hand

    def get_current_joint_velocity(self):
        dof_states = self.gym.get_actor_dof_states(self.env, self.franka_actor, gymapi.STATE_VEL)
        return np.array(dof_states["vel"][:7], dtype=np.float32)

    @staticmethod
    def _orientation_error(r_current, r_desired):
        r_error = r_current.T @ r_desired
        err_quat = Rotation.from_matrix(r_error).as_quat()
        if err_quat[3] < 0:
            err_quat = -err_quat
        return (-r_current @ err_quat[:3]).astype(np.float32)

    def _downward_axis_error(self, r_current):
        axis_world = r_current @ self.ee_down_axis_local
        axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-8)
        return np.cross(axis_world, self.world_down_axis).astype(np.float32)

    def _compute_osc_torques(self):
        self._refresh_osc_tensors()

        q = self.get_current_joint_position()
        qd = self.get_current_joint_velocity()
        p_curr, r_curr = self.get_end_effector_pos()

        jac_hand_idx = self._jacobian_body_index(self.ee_body_local_idx)
        jac_tip_idx = self._jacobian_body_index(self.fingertip_body_idx)
        jac_hand = np.array(self._jacobian[0, jac_hand_idx, :, :7], dtype=np.float32)
        jac_tip = np.array(self._jacobian[0, jac_tip_idx, :, :7], dtype=np.float32)
        jac = np.zeros((6, 7), dtype=np.float32)
        jac[:3, :] = jac_tip[:3, :]
        jac[3:, :] = jac_hand[3:, :]
        mm = np.array(self._mm[0, :7, :7], dtype=np.float32)
        mm_inv = np.linalg.inv(mm + 1e-6 * np.eye(7, dtype=np.float32))

        m_eef_inv = jac @ mm_inv @ jac.T
        m_eef = np.linalg.inv(m_eef_inv + 1e-6 * np.eye(6, dtype=np.float32))

        error = np.zeros(6, dtype=np.float32)
        error[:3] = self.position_d - p_curr
        error[3:] = self._orientation_error(r_curr, self.orientation_d)

        ee_vel = jac @ qd
        tilt_error = self._downward_axis_error(r_curr)
        omega = ee_vel[3:]
        omega_tilt = omega - np.dot(omega, self.world_down_axis) * self.world_down_axis
        desired_wrench = self.cartesian_stiffness @ error - self.cartesian_damping @ ee_vel
        desired_wrench[3:] += self.tilt_stiffness * tilt_error - self.tilt_damping * omega_tilt
        tau = jac.T @ (m_eef @ desired_wrench)

        j_eef_inv = m_eef @ jac @ mm_inv
        q_error = (self.home_q - q + np.pi) % (2 * np.pi) - np.pi
        u_null = self.kp_null * q_error - self.kd_null * qd
        tau_null = mm @ u_null
        tau = tau + (np.eye(7, dtype=np.float32) - jac.T @ j_eef_inv) @ tau_null

        return np.clip(tau, -self._effort_limits, self._effort_limits)

    def step(self, cmd):
        cmd = np.asarray(cmd, dtype=np.float32).reshape(-1)
        if cmd.size == 3:
            self.p_d = self.p_d + cmd
            self.p_d[2] = max(self.p_d[2], self.low_height)
            self.position_d = self.p_d.copy()
            tau = self._compute_osc_torques()

            self._effort_control.zero_()
            self._effort_control[:7] = torch.as_tensor(tau, dtype=torch.float32, device=self._effort_control.device)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
            if self.franka_dof_count >= 9:
                self.gym.set_actor_dof_position_targets(self.env, self.franka_actor, self._joint_targets)
            self.show_target(goal_pos=self.p_d, goal_quat=Rotation.from_matrix(self.R_d).as_quat())
            self._simulate_once()
            return

        if cmd.size == 7:
            self.step_joint_delta(cmd)
            return

        raise ValueError(f"Invalid action dimension: {cmd.size}. Expected 3 or 7.")


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
        con_pos_list = []

        contacts = simulator.get_physx_contacts()
        mu = float(self.param_.mu_object_)
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

        return phi_vec, jac_mat, con_pos_list, jac_mat_env


def adapt_param_for_cartesian_solver(param, args):
    table_height = float(param.table_height)
    param.n_cmd_ = 3
    param.n_robot_qpos_ = 3
    param.n_qpos_ = 10
    param.n_qvel_ = 9
    param.mpc_u_lb_ = np.full((3,), -float(args.cartesian_step), dtype=np.float32)
    param.mpc_u_ub_ = np.full((3,), float(args.cartesian_step), dtype=np.float32)
    param.robot_stiff_ = np.diag(3 * [float(args.cartesian_joint_stiffness)])
    q = np.zeros((param.n_qvel_, param.n_qvel_))
    q[:6, :6] = param.obj_inertia_
    q[6:, 6:] = param.robot_stiff_
    param.Q = q
    param.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7), np.array([-1.0, -1.0, table_height + 0.02])))
    param.mpc_q_ub_ = np.hstack((1e7 * np.ones(7), np.array([1.5, 1.0, 1.5])))
    # Isaac object asset uses the raw mesh at scale 1.0, so rebuild the contact-point optimizer
    # with the same physical scale. The MuJoCo pipeline uses a separate 0.0025 scale.
    param.lambda_optimizer = LambdaContactControlOptimizer(
        mesh_path=param.mesh_path_,
        obj_mass=param.obj_mass_,
        arm_friction=param.mu_object_,
        contact_stiffness=param.model_params,
        time_step=param.h_ * 10,
        sample_num=args.sample_num,
        pos_coef=args.pos_coef,
        ori_coef=args.ori_coef,
        scale_factors=[1.0, 1.0, 1.0],
    )
    param.sol_guess_ = None
    return param


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default="piggy_bank", help="name of obj")
    parser.add_argument("--use-xml-texture", action="store_true", help="apply object texture parsed from env_fingertips_*.xml")
    parser.add_argument("--attract_coef", type=float, default=0.5, help="coef of attract function")
    parser.add_argument("--reject_coef", type=float, default=0.001, help="coef of reject function")
    parser.add_argument("--contact_coef", type=float, default=0.5, help="coef of contact function")
    parser.add_argument("--contact_cost_param", type=float, default=1, help="mass center or project point attract")
    parser.add_argument("--model_param", type=float, default=7, help="model param")
    parser.add_argument("--reject_dis", type=float, default=0.01, help="reject radius")
    parser.add_argument("--attract_point_comp", type=float, default=0.05, help="distance compensation of attract point")
    parser.add_argument("--ground_height_threshold", type=float, default=0.012, help="threshold of sample points height")
    parser.add_argument("--sample_num", type=int, default=70, help="number of sample point")
    parser.add_argument("--pos_coef", type=float, default=1, help="coef of position cost in mlqp_point")
    parser.add_argument("--ori_coef", type=float, default=0.001, help="coef of orientation cost in mlqp_point")
    parser.add_argument("--low_err_coef", type=float, default=0.1, help="coef of delta error")
    parser.add_argument("--upper_err_coef", type=float, default=1, help="coef of delta error")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--cartesian-step", type=float, default=0.01, help="bound of xyz delta action per MPC step")
    parser.add_argument("--cartesian-joint-stiffness", type=float, default=300.0, help="internal stiffness used in explicit model")
    parser.add_argument("--cartesian-dls-lambda", type=float, default=1e-4, help="damped least squares term for J^+")

    args = parser.parse_args()

    trial_num = 20
    success_pos_threshold = 0.02
    success_quat_threshold = 0.04
    consecutive_success_time_threshold = 20
    max_rollout_length = 5000
    success_rate = 0

    trial_count = 0
    while trial_count < trial_num:
        param = ExplicitMPCParams(args, rand_seed=trial_count, target_type="rotation", mpc_model="explicit")
        param.use_jax_contact_ = False
        param = adapt_param_for_cartesian_solver(param, args)

        contact = ContactIsaacCartesian(param)
        env = IsaacFrankaOSCSimulator(
            param,
            headless=args.headless,
            sim_device=args.sim_device,
            graphics_device_id=args.graphics_device_id,
        )
        env.show_target_object_pose(param.target_p_, param.target_q_)

        mpc = MPCExplicitIsaac(param) if param.mpc_model == "explicit" else MPCImplicit(param)

        rollout_step = 0
        consecutive_success_time = 0
        verify_cost = 0
        current_x = np.zeros(7, dtype=np.float32)
        current_x[3] = 1.0

        low_err_coef = args.low_err_coef
        upper_err_coef = args.upper_err_coef

        rollout_q_traj = []
        while rollout_step < max_rollout_length:
            if not env.dyn_paused_:
                curr_q = env.get_state()
                rollout_q_traj.append(curr_q)
                ee_pos = env.get_end_effector_pos()[0].copy()
                curr_x_solver = np.hstack([curr_q[:7], ee_pos]).astype(np.float32)

                phi_vec, jac_mat, con_point, jac_mat_env = contact.detect_once(env)
                quaternion = [curr_q[4], curr_q[5], curr_q[6], curr_q[3]]
                r_obj_to_world = Rotation.from_quat(quaternion).as_matrix()
                gravity = np.hstack([r_obj_to_world.T @ param.gravity_[:3] * param.obj_mass_, np.zeros(3)])

                target_pos_ = param.target_p_ - curr_q[0:3]
                target_pos_[2] = 0.0
                target_quat_local = rotations.quaternion_multiply(
                    rotations.quaternion_conjugate(curr_q[3:7]),
                    param.target_q_,
                )
                target_pose_local = np.hstack([r_obj_to_world.T @ target_pos_, target_quat_local])

                param.lambda_optimizer.update_Jacobian(jac_mat_env)
                visible_point_idx = param.lambda_optimizer.get_availble_point_idx(
                    curr_q[0:3], r_obj_to_world, param.target_p_, args.ground_height_threshold
                )
                st = time.time()
                best_contact_point, normal, min_error, max_error, curr_ori_coef = param.lambda_optimizer.choose_contact_points(
                    target_pose_local, current_x, gravity, visible_point_idx, phi_vec
                )
                print("time = ", time.time() - st)
                attract_point = best_contact_point.copy()
                attract_point_world = r_obj_to_world @ attract_point + curr_q[0:3]
                original_height = attract_point_world[2]
                attract_point_world -= args.attract_point_comp * r_obj_to_world @ normal
                attract_point_world[2] = max(attract_point_world[2], original_height)

                local_point = r_obj_to_world.T @ (ee_pos - curr_q[0:3])
                p_arm_local, _, x_plus_opt, error, info = param.lambda_optimizer.optimize_control_input(
                    target_pose_local, current_x, gravity, local_point
                )
                p_arm_world = r_obj_to_world @ p_arm_local + curr_q[:3]

                if verify_cost:
                    low_err_coef = args.low_err_coef
                else:
                    if np.linalg.norm(ee_pos - attract_point_world) < 5e-2:
                        low_err_coef *= 1.1
                upper_err_coef = max(args.upper_err_coef if not verify_cost else upper_err_coef - 0.002, 0.7)

                delta_error = max_error - min_error
                adaptive = delta_error * upper_err_coef if verify_cost else delta_error * low_err_coef
                verify_cost = 1 if error < (min_error + adaptive) else 0

                sol = mpc.plan_once(
                    param.target_p_,
                    param.target_q_,
                    curr_x_solver,
                    phi_vec,
                    jac_mat,
                    verify_cost_param=verify_cost,
                    virtual_point=attract_point_world,
                    contact_point=p_arm_world,
                    curr_ori_coef=curr_ori_coef,
                    sol_guess=param.sol_guess_,
                )
                param.sol_guess_ = sol["sol_guess"]
                action = sol["action"]

                env.step(action)
                rollout_step += 1

                curr_q = env.get_state()
                if (
                    metrics.comp_pos_error(curr_q[0:3], param.target_p_) < success_pos_threshold
                    and metrics.comp_quat_error(curr_q[3:7], param.target_q_) < success_quat_threshold
                ):
                    consecutive_success_time += 1
                else:
                    consecutive_success_time = 0

                if consecutive_success_time > consecutive_success_time_threshold:
                    break

        success_rate += 1 if rollout_step < max_rollout_length else 0
        env.close()
        trial_count += 1

    print(f"Success rate over {trial_num} trials: {success_rate}/{trial_num} = {success_rate/trial_num:.2%}")


if __name__ == "__main__":
    main()
