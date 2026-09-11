'''
python examples/mpc/franka/mppi2/test_mppi.py --obj elephant --curobo-robot-cfg franka.yml

'''
import argparse
import os
import re
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
from isaacgym import gymapi


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from examples.mpc.franka.ik2.params_curobo import ExplicitMPCParamsCurobo
from examples.mpc.franka.ik2.test_mppi_isaac import (
    ContactIsaac,
    IsaacFrankaSimulator,
    _extract_quat_xyzw,
    _extract_vec3,
    _parse_bool_arg,
)
from examples.mpc.franka.mppi2.Curobosolver import CuroboSolver
from utils import metrics, rotations


def _mjcf_quat_wxyz_to_urdf_rpy(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    return Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)


class IsaacFrankaFingertipSimulator(IsaacFrankaSimulator):
    @staticmethod
    def _get_franka_asset_info(repo_root):
        asset_root = os.path.join(repo_root, "envs/robots/assets/urdf")
        src_urdf = os.path.join(asset_root, "franka_description", "robots", "franka_panda.urdf")
        dst_rel = os.path.join("franka_description", "robots", "franka_panda_nohand_sphere_tmp.urdf")
        dst_urdf = os.path.join(asset_root, dst_rel)
        attachment_rpy = _mjcf_quat_wxyz_to_urdf_rpy([0.3826834, 0.0, 0.0, 0.9238795])

        with open(src_urdf, "r", encoding="ascii") as f:
            urdf_text = f.read()

        # Strip the stock hand/fingers so the loaded robot ends at the custom fingertip.
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

    def _build_body_index_cache(self):
        super()._build_body_index_cache()
        self.franka_body_count = len(self.franka_body_names)
        self._jacobian_body_offset = self.franka_body_count - int(self._jacobian.shape[1])
        if self._jacobian_body_offset < 0:
            raise RuntimeError(
                f"Unexpected Franka Jacobian shape: rigid_bodies={self.franka_body_count}, "
                f"jacobian_bodies={int(self._jacobian.shape[1])}"
            )
        self.ee_body_name = "attachment" if "attachment" in self.franka_body_names else "panda_link7"
        self.ee_body_local_idx = self.franka_body_names.index(self.ee_body_name)
        self.fingertip_body_idx = (
            self.franka_body_names.index("fingertip")
            if "fingertip" in self.franka_body_names
            else self.ee_body_local_idx
        )
        self.task_body_local_idx = self.fingertip_body_idx

    def _jacobian_body_index(self, local_body_idx):
        jac_idx = int(local_body_idx) - int(self._jacobian_body_offset)
        if jac_idx < 0 or jac_idx >= int(self._jacobian.shape[1]):
            raise IndexError(
                f"Rigid body index {local_body_idx} maps to invalid jacobian index {jac_idx}; "
                f"offset={self._jacobian_body_offset}, jacobian_bodies={int(self._jacobian.shape[1])}"
            )
        return jac_idx

    def _get_franka_body_pose(self, local_idx):
        states = self.gym.get_actor_rigid_body_states(self.env, self.franka_actor, gymapi.STATE_POS)
        pos = _extract_vec3(states["pose"]["p"][local_idx])
        quat_xyzw = _extract_quat_xyzw(states["pose"]["r"][local_idx])
        rot = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix().astype(np.float32)
        return pos.astype(np.float32), rot

    def get_end_effector_pos(self):
        return self._get_franka_body_pose(self.task_body_local_idx)

    def get_body_point_jacobian(self, sim_body_idx, point_world):
        info = self.sim_body_to_actor.get(int(sim_body_idx), None)
        if info is None:
            return np.zeros((3, self.param_.n_robot_qpos_), dtype=np.float32)
        _, local_idx, actor_kind = info
        if actor_kind != "franka":
            return np.zeros((3, self.param_.n_robot_qpos_), dtype=np.float32)

        self.gym.refresh_jacobian_tensors(self.sim)
        jac_body_idx = self._jacobian_body_index(local_idx)
        jac_body = self._jacobian[0, jac_body_idx]
        jv = np.array(jac_body[:3, : self.param_.n_robot_qpos_], dtype=np.float32)
        jw = np.array(jac_body[3:6, : self.param_.n_robot_qpos_], dtype=np.float32)

        p_body, _ = self.get_body_pose_by_sim_index(sim_body_idx)
        if p_body is None:
            return jv
        r = np.asarray(point_world, dtype=np.float32) - np.asarray(p_body, dtype=np.float32)
        return jv - self._skew(r) @ jw


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default="elephant", help="name of object mesh")
    parser.add_argument("--attract_coef", type=float, default=0.5, help="coef of attract function")
    parser.add_argument("--reject_coef", type=float, default=0.001, help="coef of reject function")
    parser.add_argument("--contact_coef", type=float, default=0.5, help="coef of contact function")
    parser.add_argument("--contact_cost_param", type=float, default=1.0, help="mass center or project point attract")
    parser.add_argument("--model_param", type=float, default=7.0, help="contact model parameter")
    parser.add_argument("--reject_dis", type=float, default=0.01, help="reject radius")
    parser.add_argument(
        "--attract_point_comp",
        type=float,
        default=0.05,
        help="distance compensation of attract point",
    )
    parser.add_argument(
        "--ground_height_threshold",
        type=float,
        default=0.012,
        help="threshold of sample point height",
    )
    parser.add_argument("--sample_num", type=int, default=70, help="number of sampled contact points")
    parser.add_argument("--pos_coef", type=float, default=1.0, help="coef of position cost in mlqp_point")
    parser.add_argument("--ori_coef", type=float, default=0.001, help="coef of orientation cost in mlqp_point")
    parser.add_argument("--low_err_coef", type=float, default=0.1, help="coef of delta error")
    parser.add_argument("--upper_err_coef", type=float, default=1.0, help="coef of delta error")

    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument(
        "--show-ghost-object",
        type=_parse_bool_arg,
        default=False,
        help="whether to render the semi-transparent ghost object mesh at the target pose",
    )
    parser.add_argument("--draw-rollout", action="store_true", help="draw the preview rollout in Isaac viewer")

    parser.add_argument("--trial-num", type=int, default=20)
    parser.add_argument("--max-rollout-length", type=int, default=5000)
    parser.add_argument("--success-pos-threshold", type=float, default=0.02)
    parser.add_argument("--success-quat-threshold", type=float, default=0.04)
    parser.add_argument("--consecutive-success-steps", type=int, default=20)

    parser.add_argument("--curobo-robot-cfg", type=str, default="franka.yml")
    parser.add_argument("--curobo-step-dt", type=float, default=None)
    parser.add_argument("--curobo-max-attempts", type=int, default=1)
    parser.add_argument("--curobo-preview-steps", type=int, default=8)
    parser.add_argument(
        "--curobo-use-goal-orientation",
        action="store_true",
        help="track target object orientation instead of keeping the current EE orientation",
    )
    parser.add_argument(
        "--disable-midpoint-goal",
        action="store_true",
        help="disable the midpoint candidate between virtual point and contact point",
    )
    parser.add_argument(
        "--enable-midpoint-goal",
        action="store_true",
        help="re-enable the midpoint candidate between virtual point and contact point",
    )
    return parser


def _apply_param_overrides(param, args) -> None:
    param.use_jax_contact_ = False
    param.show_ghost_object_ = bool(args.show_ghost_object)

    param.curobo_robot_cfg_ = args.curobo_robot_cfg
    param.curobo_step_dt_ = float(args.curobo_step_dt) if args.curobo_step_dt is not None else float(param.h_)
    param.curobo_store_rollouts_ = True
    param.curobo_max_attempts_ = int(args.curobo_max_attempts)
    param.curobo_preview_steps_ = int(args.curobo_preview_steps)
    param.curobo_use_goal_orientation_ = bool(args.curobo_use_goal_orientation)
    param.curobo_use_midpoint_goal_ = bool(args.enable_midpoint_goal) and not bool(args.disable_midpoint_goal)


def _print_solver_summary(step_idx: int, elapsed: float, verify_cost: int, sol: dict) -> None:
    cost_opt = float(np.asarray(sol.get("cost_opt", [np.nan]), dtype=np.float64).reshape(-1)[0])
    goal_pos = np.asarray(sol.get("goal_pos", np.full((3,), np.nan)), dtype=np.float64).reshape(3)
    pose_error = float(sol.get("pose_error", np.nan))
    goal_source = str(sol.get("goal_source", "unknown"))
    print(
        f"step={step_idx:04d} "
        f"time_cost={elapsed:.4f}s "
        f"verify_cost={int(verify_cost)} "
        f"goal_source={goal_source} "
        f"pose_error={pose_error:.6f} "
        f"opt_cost={cost_opt:.6f}"
    )
    print("goal_pos =", np.round(goal_pos, 4))

    breakdown = sol.get("cost_breakdown", {})
    if breakdown:
        print(
            "cost_breakdown =",
            {
                "base_cost": round(float(breakdown.get("base_cost", np.nan)), 6),
                "final_cost": round(float(breakdown.get("contact_final_cost", np.nan)), 6),
                "contact_path_cost": round(float(breakdown.get("contact_path_cost", np.nan)), 6),
                "combined_cost": round(float(breakdown.get("combined_cost", np.nan)), 6),
                "curobo_motion_cost": round(float(breakdown.get("curobo_motion_cost", np.nan)), 6),
            },
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    success_rate = 0
    trial_count = 0
    while trial_count < args.trial_num:
        param = ExplicitMPCParamsCurobo(args, rand_seed=trial_count, target_type="rotation", mpc_model="explicit")
        _apply_param_overrides(param, args)

        contact = ContactIsaac(param)
        env = IsaacFrankaFingertipSimulator(
            param,
            headless=args.headless,
            sim_device=args.sim_device,
            graphics_device_id=args.graphics_device_id,
        )
        env.show_target_object_pose(param.target_p_, param.target_q_)

        mpc = CuroboSolver(param)

        rollout_step = 0
        consecutive_success_time = 0
        verify_cost = 0
        current_x = np.zeros(7, dtype=np.float32)
        current_x[3] = 1.0

        low_err_coef = float(args.low_err_coef)
        upper_err_coef = float(args.upper_err_coef)

        try:
            while rollout_step < args.max_rollout_length:
                if env.dyn_paused_:
                    continue

                curr_q = env.get_state()
                phi_vec, jac_mat, _, jac_mat_env = contact.detect_once(env)

                quat_xyzw = np.array([curr_q[4], curr_q[5], curr_q[6], curr_q[3]], dtype=np.float32)
                r_obj_to_world = Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float32)
                gravity = np.hstack(
                    [r_obj_to_world.T @ param.gravity_[:3] * param.obj_mass_, np.zeros(3, dtype=np.float32)]
                )

                target_pos_local = param.target_p_ - curr_q[0:3]
                target_pos_local[2] = 0.0
                target_quat_local = rotations.quaternion_multiply(
                    rotations.quaternion_conjugate(curr_q[3:7]),
                    param.target_q_,
                )
                target_pose_local = np.hstack([r_obj_to_world.T @ target_pos_local, target_quat_local]).astype(
                    np.float32
                )

                param.lambda_optimizer.update_Jacobian(jac_mat_env)
                visible_point_idx = param.lambda_optimizer.get_availble_point_idx(
                    curr_q[0:3],
                    r_obj_to_world,
                    param.target_p_,
                    args.ground_height_threshold,
                )
                best_contact_point, normal, min_error, max_error, curr_ori_coef = (
                    param.lambda_optimizer.choose_contact_points(
                        target_pose_local,
                        current_x,
                        gravity,
                        visible_point_idx,
                    )
                )

                attract_point_world = r_obj_to_world @ best_contact_point + curr_q[0:3]
                original_height = float(attract_point_world[2])
                attract_point_world = attract_point_world - args.attract_point_comp * (r_obj_to_world @ normal)
                attract_point_world[2] = max(float(attract_point_world[2]), original_height)

                ee_pos = env.get_end_effector_pos()[0].copy()
                local_point = r_obj_to_world.T @ (ee_pos - curr_q[0:3])
                p_arm_local, _, _, error, _ = param.lambda_optimizer.optimize_control_input(
                    target_pose_local,
                    current_x,
                    gravity,
                    local_point,
                )
                p_arm_world = r_obj_to_world @ p_arm_local + curr_q[:3]

                if verify_cost:
                    low_err_coef = float(args.low_err_coef)
                elif np.linalg.norm(ee_pos - attract_point_world) < 5e-2:
                    low_err_coef *= 1.1

                upper_err_coef = max(
                    float(args.upper_err_coef) if not verify_cost else upper_err_coef - 0.002,
                    0.7,
                )
                delta_error = float(max_error - min_error)
                adaptive = delta_error * (upper_err_coef if verify_cost else low_err_coef)
                verify_cost = 1 if float(error) < (float(min_error) + adaptive) else 0
                start_time = time.time()
                sol = mpc.plan_once(
                    param.target_p_,
                    param.target_q_,
                    curr_q,
                    phi_vec,
                    jac_mat,
                    verify_cost_param=verify_cost,
                    virtual_point=attract_point_world,
                    contact_point=p_arm_world,
                    curr_ori_coef=curr_ori_coef,
                    sol_guess=param.sol_guess_,
                )
                elapsed = time.time() - start_time
                print("elapsed = ", elapsed)
                param.sol_guess_ = sol["sol_guess"]

                if args.draw_rollout:
                    env.draw_rollout_lines(sol.get("rollout_q", None))

                _print_solver_summary(rollout_step, elapsed, verify_cost, sol)
                print("attract_point_world =", np.round(attract_point_world, 4))
                print("contact_point_world =", np.round(p_arm_world, 4))

                visual_point = np.asarray(sol.get("goal_pos", attract_point_world), dtype=np.float32).reshape(3)
                env.show_point(visual_point)
                env.step(sol["action"])

                rollout_step += 1
                curr_q_post = env.get_state()
                pos_success = metrics.comp_pos_error(curr_q_post[0:3], param.target_p_) < args.success_pos_threshold
                quat_success = metrics.comp_quat_error(curr_q_post[3:7], param.target_q_) < args.success_quat_threshold
                if pos_success and quat_success:
                    consecutive_success_time += 1
                else:
                    consecutive_success_time = 0

                if consecutive_success_time > args.consecutive_success_steps:
                    break
        finally:
            env.close()

        if rollout_step < args.max_rollout_length:
            success_rate += 1
        trial_count += 1

    print(
        f"Success rate over {args.trial_num} trials: "
        f"{success_rate}/{args.trial_num} = {success_rate / max(args.trial_num, 1):.2%}"
    )


if __name__ == "__main__":
    main()
