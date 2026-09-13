import argparse
import os
import sys
import time

import numpy as np
from isaacgym import gymapi

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(current_dir)
while os.path.basename(parent_dir) != "scsp-robot":
    _next_dir = os.path.dirname(parent_dir)
    if _next_dir == parent_dir:
        raise RuntimeError("scsp-robot repo root not found from %s" % current_dir)
    parent_dir = _next_dir
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from examples.mpc.franka.ik2.test_mppi_isaac import (  # noqa: E402
    IsaacFrankaSimulator,
    _parse_bool_arg,
    _extract_vec3,
)
from examples.mpc.franka.mppi.mpc_solver import MPCSolver  # noqa: E402
from examples.mpc.franka.mppi.param_mppi import IDTOMPPIParams  # noqa: E402
from utils import metrics  # noqa: E402


DEFAULT_CITO_TRIAL_REPLAY = {
    "obj": "piggy_bank",
    "contact_cost_param": 0.0,
    "attract_coef": 0.5,
    "reject_coef": 0.001,
    "contact_coef": 0.5,
    "reject_dis": 0.01,
    "model_param": 7.0,
    "sim_device": "cuda:0",
    "graphics_device_id": 0,
    "trial_start": 0,
    "trial_count": 1,
}


class CITOIsaacFrankaSimulator(IsaacFrankaSimulator):
    def get_object_velocity_world(self):
        obj_state = self.gym.get_actor_rigid_body_states(
            self.env, self.obj_actor, gymapi.STATE_ALL
        )
        linear_velocity = _extract_vec3(obj_state["vel"]["linear"][0])
        angular_velocity = _extract_vec3(obj_state["vel"]["angular"][0])
        return linear_velocity.astype(np.float32), angular_velocity.astype(np.float32)

    def get_current_joint_velocity(self):
        dof_states = self.gym.get_actor_dof_states(
            self.env, self.franka_actor, gymapi.STATE_VEL
        )
        return np.array(dof_states["vel"][:7], dtype=np.float32)

    def show_point(self, goal_pos=None):
        self.show_target(goal_pos=goal_pos)

    def step_robot_q_target(self, q_robot_target):
        q_robot_target = np.asarray(q_robot_target, dtype=np.float32).reshape(7)
        self._joint_targets[:7] = q_robot_target
        if self.franka_dof_count >= 9:
            self._joint_targets[7] = 0.04
            self._joint_targets[8] = 0.04
        self.gym.set_actor_dof_position_targets(
            self.env, self.franka_actor, self._joint_targets
        )
        self._simulate_once()
        return q_robot_target

    def step_solution(self, solution):
        q_robot_target = np.asarray(
            solution["robot_q_target"], dtype=np.float32
        ).reshape(7)
        self.step_robot_q_target(q_robot_target)
        return q_robot_target


class ContactIsaacCITO:
    def __init__(self, param):
        self.param_ = param

    def detect_once(self, simulator):
        return self.param_.extract_isaac_contact_observation(simulator)


def build_param(args, rand_seed):
    param = IDTOMPPIParams(
        horizon=args.horizon,
        dt=args.mpc_dt,
        max_contacts=args.max_contacts,
        args=args,
        rand_seed=rand_seed,
        target_type="rotation",
        mpc_model="q_trajectory",
    )
    param.ipopt_max_iter_ = int(args.ipopt_max_iter)
    param.sim_dt_ = float(args.sim_dt)
    param.show_ghost_object_ = bool(args.show_ghost_object)
    param.use_xml_texture_ = bool(args.use_xml_texture)
    param.sol_guess_ = None
    return param


def _maybe_show_contact(env, contact_obs):
    signed_distances = np.asarray(contact_obs["signed_distances"], dtype=np.float64)
    valid_idx = np.where(np.isfinite(signed_distances) & (signed_distances < 1e2))[0]
    if valid_idx.size == 0:
        return
    best_idx = int(valid_idx[np.argmin(signed_distances[valid_idx])])
    env.show_point(contact_obs["contact_points_W"][best_idx])


def run_trial(args, trial_id):
    param = build_param(args, rand_seed=trial_id)
    env = CITOIsaacFrankaSimulator(
        param,
        headless=args.headless,
        sim_device=args.sim_device,
        graphics_device_id=args.graphics_device_id,
    )
    env.show_target_object_pose(param.target_p_, param.target_q_)

    contact_monitor = ContactIsaacCITO(param)
    solver = MPCSolver(param)

    rollout_step = 0
    consecutive_success_time = 0
    success = False
    rollout_q_traj = []

    try:
        while rollout_step < args.max_rollout_length:
            if env.dyn_paused_:
                continue

            curr_q = env.get_state()
            rollout_q_traj.append(np.asarray(curr_q, dtype=np.float32).copy())

            contact_obs = contact_monitor.detect_once(env)
            param.set_initial_state(
                contact_obs["q_curr"],
                contact_obs["v_curr"],
                update_nominal=True,
                update_robot_goal=True,
            )
            cost_param = param.default_cost_param()
            contact_schedule = param.extract_isaac_contact_schedule(
                env,
                repeat=bool(args.repeat_contact_schedule),
                contact_step=contact_obs["contact_step"],
            )

            solve_start = time.time()
            try:
                solution = solver.plan_once(
                    cost_param,
                    contact_schedule,
                    sol_guess=param.sol_guess_,
                )
            except RuntimeError as exc:
                print(
                    f"[trial {trial_id:03d} step {rollout_step:04d}] "
                    f"IPOPT solve failed: {exc}"
                )
                break
            param.sol_guess_ = solution["sol_guess"]
            solve_time = time.time() - solve_start

            q_robot_target = env.step_solution(solution)
            rollout_step += 1

            if args.show_rollout:
                env.draw_rollout_lines(solution["rollout_q"])
            if args.show_contact_point:
                _maybe_show_contact(env, contact_obs)

            curr_q = env.get_state()
            pos_error = metrics.comp_pos_error(curr_q[0:3], param.target_p_)
            quat_error = metrics.comp_quat_error(curr_q[3:7], param.target_q_)
            if (
                pos_error < args.success_pos_threshold
                and quat_error < args.success_quat_threshold
            ):
                consecutive_success_time += 1
            else:
                consecutive_success_time = 0

            if args.print_every > 0 and rollout_step % args.print_every == 0:
                min_contact_distance = float(np.min(contact_obs["signed_distances"]))
                equality_violation = float(
                    np.max(np.abs(solution["rollout"]["h"]))
                    if solution["rollout"]["h"].size
                    else 0.0
                )
                print(
                    f"[trial {trial_id:03d} step {rollout_step:04d}] "
                    f"cost={solution['cost_opt']:.4f} "
                    f"solve_time={solve_time:.3f}s "
                    f"status={solution['solve_status']} "
                    f"|q1-q0|={np.linalg.norm(solution['robot_dq_target']):.4f} "
                    f"|q_target|={np.linalg.norm(q_robot_target):.4f} "
                    f"min_signed_distance={min_contact_distance:.4f} "
                    f"eq_inf={equality_violation:.4e} "
                    f"pos_err={pos_error:.4f} "
                    f"quat_err={quat_error:.4f}"
                )

            if consecutive_success_time >= args.consecutive_success_time:
                success = True
                break
    finally:
        env.close()

    return {
        "success": success,
        "rollout_step": rollout_step,
        "rollout_q_traj": rollout_q_traj,
        "target_p": np.asarray(param.target_p_, dtype=np.float32),
        "target_q": np.asarray(param.target_q_, dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default="piggy_bank", help="name of obj")
    parser.add_argument(
        "--use-xml-texture",
        action="store_true",
        help="apply object texture parsed from env_fingertips_*.xml",
    )
    parser.add_argument(
        "--contact_cost_param",
        type=float,
        default=0.0,
        help="contact objective interpolation factor",
    )
    parser.add_argument("--attract_coef", type=float, default=0.5)
    parser.add_argument("--reject_coef", type=float, default=0.001)
    parser.add_argument("--contact_coef", type=float, default=0.5)
    parser.add_argument("--reject_dis", type=float, default=0.01)
    parser.add_argument("--model_param", type=float, default=7.0)
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--viewer", dest="headless", action="store_false")
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument(
        "--show-ghost-object",
        type=_parse_bool_arg,
        default=False,
        help="whether to render the target object ghost mesh",
    )
    parser.add_argument(
        "--show-rollout",
        type=_parse_bool_arg,
        default=True,
        help="draw the optimized EE rollout in the viewer",
    )
    parser.add_argument(
        "--show-contact-point",
        type=_parse_bool_arg,
        default=True,
        help="visualize the closest detected contact point",
    )
    parser.add_argument(
        "--repeat-contact-schedule",
        type=_parse_bool_arg,
        default=True,
        help="repeat the latest Isaac contact estimate across the whole horizon",
    )
    parser.add_argument("--horizon", type=int, default=20, help="CITO horizon")
    parser.add_argument("--mpc-dt", type=float, default=0.05, help="planner time step")
    parser.add_argument("--sim-dt", type=float, default=0.01, help="Isaac sim dt")
    parser.add_argument("--max-contacts", type=int, default=4)
    parser.add_argument("--ipopt-max-iter", type=int, default=100)
    parser.add_argument("--max-rollout-length", type=int, default=400)
    parser.add_argument("--success-pos-threshold", type=float, default=0.02)
    parser.add_argument("--success-quat-threshold", type=float, default=0.04)
    parser.add_argument("--consecutive-success-time", type=int, default=20)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--trial-count", type=int, default=1)
    parser.set_defaults(**DEFAULT_CITO_TRIAL_REPLAY)
    parser.set_defaults(headless=False)
    args = parser.parse_args()

    if args.trial_start < 0:
        raise ValueError(f"trial_start must be non-negative, got {args.trial_start}")
    if args.trial_count <= 0:
        raise ValueError(f"trial_count must be positive, got {args.trial_count}")

    success_count = 0
    trial_stop = int(args.trial_start) + int(args.trial_count)
    for trial_id in range(int(args.trial_start), trial_stop):
        result = run_trial(args, trial_id)
        success_count += int(result["success"])

    print(
        f"Success rate over {args.trial_count} trials "
        f"(trial ids {args.trial_start} to {trial_stop - 1}): "
        f"{success_count}/{args.trial_count} = {success_count/max(args.trial_count, 1):.2%}"
    )


if __name__ == "__main__":
    main()
