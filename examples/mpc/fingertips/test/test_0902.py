import time
import json
import numpy as np
import os
import sys
import faulthandler
faulthandler.enable()
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(current_dir)
while os.path.basename(parent_dir) != "scsp-robot":
    _next_dir = os.path.dirname(parent_dir)
    if _next_dir == parent_dir:
        raise RuntimeError("scsp-robot repo root not found from %s" % current_dir)
    parent_dir = _next_dir
sys.path.insert(0, parent_dir)
from planning.acados_env import ensure_acados_env
ensure_acados_env()
from examples.mpc.fingertips.test.params import ExplicitMPCParams
from planning.mpc_explicit import MPCExplicit
from planning.mpc_implicit import MPCImplicit

from envs.fingertips_env import MjSimulator
from contact.fingertips_collision_detection2 import Contact
from scipy.spatial.transform import Rotation
from utils import metrics, rotations
import argparse
import mujoco


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', type=str, default='foam_brick')
    parser.add_argument('--attract_coef', type=float, default=0.5)
    parser.add_argument('--reject_coef', type=float, default=0.001)
    parser.add_argument('--contact_coef', type=float, default=0.7)
    parser.add_argument('--contact_cost_param', type=float, default=1)
    parser.add_argument('--model_param', type=float, default=7)
    parser.add_argument('--reject_dis', type=float, default=0.02)
    parser.add_argument('--attract_point_comp', type=float, default=0.1)
    parser.add_argument('--ground_height_threshold', type=float, default=0.012)
    parser.add_argument('--fingertip_clearance', type=float, default=0.011)
    parser.add_argument('--sample_num', type=int, default=70)
    parser.add_argument('--normal_stability_cos', type=float, default=0.95)
    parser.add_argument('--random_init_tilt', dest='random_init_tilt', action='store_true',
                        help='Randomize the initial object tilt so flip starts are not upright.')
    parser.add_argument('--no_random_init_tilt', dest='random_init_tilt', action='store_false')
    parser.set_defaults(random_init_tilt=True)
    parser.add_argument('--init_tilt_deg', type=float, default=75.0)
    parser.add_argument('--init_tilt_min_deg', type=float, default=35.0)
    parser.add_argument('--pos_coef', type=float, default=500)
    parser.add_argument('--ori_coef', type=float, default=20)
    parser.add_argument('--mpc_step_limit', type=float, default=0.005)
    parser.add_argument('--low_err_coef', type=float, default=0.1)
    parser.add_argument('--upper_err_coef', type=float, default=1)
    parser.add_argument('--friction_reg_coef', type=float, default=1.0)
    parser.add_argument('--force_reg_coef', type=float, default=0.01)
    parser.add_argument('--max_contact_force', type=float, default=10.0)
    parser.add_argument('--contact_switch_radius', type=float, default=0.03)
    parser.add_argument('--contact_switch_margin_ratio', type=float, default=0.2)
    parser.add_argument('--contact_switch_margin_abs', type=float, default=0.001)
    parser.add_argument('--contact_switch_confirm_steps', type=int, default=5)
    parser.add_argument('--contact_dwell_gamma', type=float, default=0.85,
                        help='Multiply switch-confidence each stagnant on-patch step.')
    parser.add_argument('--contact_dwell_steps', type=int, default=6,
                        help='Grace steps on a patch before confidence starts decaying.')
    parser.add_argument('--value_tau', type=float, default=1.0,
                        help='Temperature for p_arm advantage → quality (value-style).')
    parser.add_argument('--value_rel_scale', type=float, default=0.08,
                        help='Advantage scale as a fraction of |V_best|; avoids raw-cost gates.')
    parser.add_argument('--value_rho', type=float, default=0.08,
                        help='Polyak rate for the V_best target network.')
    parser.add_argument('--value_alpha', type=float, default=0.25,
                        help='Online EMA rate for V_arm.')
    parser.add_argument('--verify_beta', type=float, default=0.18,
                        help='Soft update rate for verify_cost in [0, 1].')
    parser.add_argument('--ideal_contact_surface_margin', type=float, default=-0.0005)
    parser.add_argument('--spline_escape_cost', type=int, default=1)
    parser.add_argument('--ideal_contact_distance', type=float, default=0.006)
    parser.add_argument('--detour_attract_coef', type=float, default=80.0)
    parser.add_argument('--detour_repel_coef', type=float, default=40.0)
    parser.add_argument('--detour_lift_coef', type=float, default=25.0)
    parser.add_argument('--detour_align_thresh', type=float, default=0.50)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--trial_num', type=int, default=100)
    parser.add_argument('--max_rollout_length', type=int, default=5000)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--ideal_object_pose', action='store_true',
                      help='Set object pose from lambda_optimizer x_plus_opt every cycle.')
    mode.add_argument('--ideal_contact_pose', action='store_true',
                      help='Set object pose from x_plus_opt only when contact distance is near 0.')
    mode.add_argument('--ideal_contact_switch', action='store_true',
                      help='Same contact policy as --rollout, but apply x_plus '
                           'when contact distance is near 0.')
    mode.add_argument('--rollout', action='store_true',
                      help='Same policy as --ideal_contact_switch; object motion '
                           'comes from MuJoCo contact, not set-pose.')
    return parser


def _scale_local_increment(x_plus_local, scale):
    x = np.asarray(x_plus_local, dtype=np.float64).reshape(7)
    scale = float(scale)
    dp = scale * x[:3]
    qrel = x[3:7].copy()
    qrel = qrel / max(np.linalg.norm(qrel), 1e-9)
    rotvec = Rotation.from_quat([qrel[1], qrel[2], qrel[3], qrel[0]]).as_rotvec()
    q_s = Rotation.from_rotvec(scale * rotvec).as_quat()
    return np.hstack((dp, np.array([q_s[3], q_s[0], q_s[1], q_s[2]], dtype=np.float64)))


def _predicted_object_pose(q_obj, x_plus_local):
    R_obj = Rotation.from_quat([q_obj[4], q_obj[5], q_obj[6], q_obj[3]]).as_matrix()
    ideal_pos = q_obj[:3] + R_obj @ np.asarray(x_plus_local[:3], dtype=np.float64).reshape(3)
    ideal_qrel = np.asarray(x_plus_local[3:7], dtype=np.float64).reshape(4)
    ideal_quat = rotations.quaternion_multiply(q_obj[3:7], ideal_qrel)
    ideal_quat = ideal_quat / max(np.linalg.norm(ideal_quat), 1e-9)
    return ideal_pos, ideal_quat


def _apply_local_pose_increment(env, q_after, x_plus_local):
    ideal_pos, ideal_quat = _predicted_object_pose(q_after, x_plus_local)
    env.data_.qpos[:7] = np.hstack((ideal_pos, ideal_quat))
    env.data_.qvel[:6] = 0.0
    mujoco.mj_forward(env.model_, env.data_)
    return ideal_pos, ideal_quat


def _sphere_center_on_patch(obj_pos, obj_rot, local_point, local_normal, radius, margin):
    world = np.asarray(obj_pos, dtype=np.float64).reshape(3) + (
        np.asarray(obj_rot, dtype=np.float64).reshape(3, 3) @ np.asarray(local_point, dtype=np.float64).reshape(3))
    normal = np.asarray(obj_rot, dtype=np.float64).reshape(3, 3) @ np.asarray(local_normal, dtype=np.float64).reshape(3)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
    return world - max(1e-4, float(radius) + float(margin)) * normal, world, normal


def _x_plus_is_usable(x_plus_opt, info):
    if info.get('solver_failed', False) or x_plus_opt is None:
        return False
    x_plus = np.asarray(x_plus_opt, dtype=np.float64)
    if not np.isfinite(x_plus).all():
        return False
    x_plus_step = (float(np.linalg.norm(x_plus[:3])) +
                   float(np.linalg.norm(x_plus[3:7] - np.array([1., 0., 0., 0.]))))
    lambda_for_pose = np.asarray(info.get('control_input', np.zeros(3)), dtype=np.float64).reshape(-1)
    force_norm = float(np.linalg.norm(lambda_for_pose[:3])) if lambda_for_pose.size >= 3 else 0.0
    lambda_valid = (lambda_for_pose.size >= 3 and
                    np.isfinite(lambda_for_pose[:3]).all() and
                    force_norm > 1e-3)
    return x_plus_step > 1e-8 and lambda_valid


def _contact_distance_after_step(contact, env):
    contact.detect_once(env)
    measured = contact.get_actual_fingertip_contact()
    if measured is None:
        return float('inf')
    return abs(float(measured.get('dist', float('inf'))))


class ContactValueTracker:
    """Value-network style estimates of lambda costs (lower cost = higher value).

    Raw lambda costs are a bad gate: a loose margin executes a bad nearest
    ``p_arm`` and the ranked best is never used; a tight margin never turns
    ``verify_cost`` on.  This tracker keeps a slow target for the best
    cost, a conservative online estimate for ``p_arm``, and a soft
    ``verify`` that only rises when the executed point is both near-optimal
    and near the fingertip.
    """

    def __init__(self, tau=1.0, rel_scale=0.08, rho=0.08, alpha=0.25,
                 beta=0.18, dist_mid=0.022, dist_width=0.006):
        self.tau = float(tau)
        self.rel_scale = float(rel_scale)
        self.rho = float(rho)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.dist_mid = float(dist_mid)
        self.dist_width = float(dist_width)
        self.v_best = None
        self.v_arm = None
        self.verify = 0.0
        self.last_arm_idx = None

    @staticmethod
    def _ema(old, new, rate):
        new = float(new)
        if old is None or not np.isfinite(old):
            return new
        return (1.0 - rate) * float(old) + rate * new

    def reset_arm(self, sample_idx):
        if sample_idx is None:
            return
        idx = int(sample_idx)
        if self.last_arm_idx is None or idx != int(self.last_arm_idx):
            self.v_arm = None
            self.verify *= 0.5
            self.last_arm_idx = idx

    def update_values(self, c_best, c_arm, solver_ok=True):
        """Update V_best / V_arm and score whether p_arm is near-optimal."""
        info = {
            'quality': 0.0,
            'verify': float(self.verify),
            'accept_p_arm': False,
            'v_best': self.v_best,
            'v_arm': self.v_arm,
            'c_arm_cons': None,
            'scale': None,
            'q_dist': 0.0,
            'adv': None,
        }
        if c_best is None or not np.isfinite(float(c_best)):
            return info

        c_best = float(c_best)
        self.v_best = self._ema(self.v_best, c_best, self.rho)
        scale = max(self.rel_scale * abs(self.v_best), 0.05)
        info['scale'] = scale
        info['v_best'] = self.v_best

        arm_ok = bool(solver_ok) and c_arm is not None and np.isfinite(float(c_arm))
        quality = 0.0
        if arm_ok:
            c_arm = float(c_arm)
            self.v_arm = self._ema(self.v_arm, c_arm, self.alpha)
            # Clipped-double-Q analogue for costs: do not overestimate
            # value (underestimate cost) from a single lucky solve.
            c_arm_cons = max(c_arm, float(self.v_arm))
            adv = c_arm_cons - float(self.v_best)
            quality = float(np.exp(-max(adv, 0.0) / (self.tau * scale)))
            info['c_arm_cons'] = c_arm_cons
            info['adv'] = adv
            info['v_arm'] = self.v_arm
        # Hysteresis: once verify is committed, keep p_arm through noise;
        # before that, require a clearer value match to start using it.
        accept = arm_ok and quality >= (0.40 if self.verify > 0.35 else 0.55)
        info['quality'] = quality
        info['accept_p_arm'] = bool(accept)
        return info

    def update_verify(self, quality, dist_exec):
        """Soft policy: rise only when the executed point is near and good."""
        dist = float(dist_exec) if np.isfinite(float(dist_exec)) else 1.0
        q_dist = 1.0 / (1.0 + np.exp((dist - self.dist_mid) / self.dist_width))
        target = float(np.clip(quality, 0.0, 1.0)) * float(q_dist)
        # Rise like an online critic; decay like a slow target network so
        # a one-frame gap does not slam verify_cost back to zero.
        rate = self.beta if target >= self.verify else 0.35 * self.beta
        self.verify = (1.0 - rate) * self.verify + rate * target
        return float(self.verify), float(q_dist)


def main(args=None):
    if args is None:
        args = build_parser().parse_args()
    os.environ['MUJOCO_HEADLESS'] = '0' if args.viewer else '1'
    use_ideal_object_pose = bool(args.ideal_object_pose)
    use_ideal_contact_pose = bool(args.ideal_contact_pose)
    use_ideal_contact_switch = bool(getattr(args, 'ideal_contact_switch', False))
    use_rollout = bool(args.rollout) or not (
        use_ideal_object_pose or use_ideal_contact_pose or use_ideal_contact_switch)
    # Switch and rollout share one contact policy.  The only difference is
    # how a made contact moves the object: set-pose vs MuJoCo physics.
    use_switch_policy = use_ideal_contact_switch or use_rollout
    use_gated_contact_policy = use_ideal_contact_pose or use_switch_policy
    use_set_pose = use_ideal_object_pose or use_ideal_contact_pose or use_ideal_contact_switch

    save_flag = False
    success_rate = 0
    if save_flag:
        save_dir = './examples/mpc/franka/trail/cube'
        prefix_data_name = 'ours_'

    trial_num = max(1, int(args.trial_num))
    success_pos_threshold = 0.02
    success_quat_threshold = 0.015
    consecutive_success_time_threshold = 0
    max_rollout_length = max(1, int(args.max_rollout_length))
    trial_count = 0
    env = None
    contact = None
    fingertip_radius = None

    while trial_count < trial_num:
        args.solver = 'acados'
        param = ExplicitMPCParams(args, rand_seed=trial_count, target_type='ground-rotation', model='explicit')
        param.torch_solver = 'acados'
        param.lambda_optimizer.solver = 'acados'
        param.lambda_optimizer.lock_contact_patch = False
        param.lambda_optimizer.contact_switch_confirm_steps = max(
            1, int(args.contact_switch_confirm_steps))
        mpc = MPCExplicit(param) if param.mpc_model == 'explicit' else MPCImplicit(param)
        pose_apply_count = 0
        choose_times = []

        if env is None:
            contact = Contact(param)
            env = MjSimulator(param)
            fingertip_radius = float(np.asarray(env.model_.geom('fingertip0').size).reshape(-1)[0])
        else:
            contact.param_ = param
            env.param_ = param
            env.set_goal(param.target_p_, param.target_q_)
            env.reset_mj_env()
            print(f'next scene: trial {trial_count}')

        rollout_step = 0
        consecutive_success_time = 0
        min_pos_err = float('inf')
        min_quat_err = float('inf')
        verify_cost = 0.0
        value_tracker = ContactValueTracker(
            tau=float(args.value_tau),
            rel_scale=float(args.value_rel_scale),
            rho=float(args.value_rho),
            alpha=float(args.value_alpha),
            beta=float(args.verify_beta),
        )
        value_info = {}
        f_c = 1.0
        dt = env.model_.opt.timestep * env.param_.frame_skip_
        tau = 1.0 / (2.0 * np.pi * f_c)
        alpha = dt / (tau + dt)
        filtered_attract = None

        while rollout_step < max_rollout_length:
            curr_q = env.get_state()
            phi_vec, jac_mat, con_point, jac_mat_env, if_contact = contact.detect_once(env)
            quanternion = [curr_q[4], curr_q[5], curr_q[6], curr_q[3]]
            R_obj_to_world = Rotation.from_quat(quanternion).as_matrix()
            gravity = np.hstack([R_obj_to_world.T @ param.gravity_[:3] * param.obj_mass_, np.zeros(3)])

            target_quat_local = rotations.quaternion_multiply(
                rotations.quaternion_conjugate(curr_q[3:7]), param.target_q_)
            target_pose_eval = np.hstack([R_obj_to_world.T @ (param.target_p_ - curr_q[:3]), target_quat_local])
            current_pose_eval = np.array([0., 0., 0., 1., 0., 0., 0.])
            current_tip_local = R_obj_to_world.T @ (curr_q[7:10] - curr_q[:3])

            param.lambda_optimizer.update_Jacobian(jac_mat_env)
            # Lambda ranks every sampled surface point whose fingertip
            # centre would clear the table.  Visibility / heading gates are
            # MPC travel heuristics and must not shrink the contact set.
            visible_point_idx = param.lambda_optimizer.get_availble_point_idx(
                curr_q[0:3], R_obj_to_world, param.target_p_, args.ground_height_threshold,
                viewpoint_local=None,
                heading_filter=not use_gated_contact_policy)

            last_idx = getattr(param.lambda_optimizer, 'last_selected_idx', None)
            last_exec_idx = getattr(param.lambda_optimizer, 'last_executed_idx', None)
            blocked = getattr(param.lambda_optimizer, '_blocked_contact_indices', {})
            # Keep the incumbent ranked / executed patches in the candidate
            # pool after the first apply (rotation can drop them below the
            # floor gate).  Hard-lock only while dwell-confidence still
            # trusts the patch; once confidence collapses the ranker and
            # the nearest p_arm selector may explore.  Do not force-append
            # a blacklisted trunk/ear sample.  The same rules apply to
            # best_contact and to p_arm_world.
            if use_gated_contact_policy:
                visible_point_idx = np.asarray(visible_point_idx, dtype=np.int32)
                for idx in (last_idx, last_exec_idx):
                    if idx is None:
                        continue
                    if (int(idx) not in visible_point_idx
                            and int(idx) not in blocked):
                        visible_point_idx = np.append(visible_point_idx, int(idx))
                incumbents = [int(idx) for idx in (last_idx, last_exec_idx)
                              if idx is not None]
                if ((pose_apply_count > 0 or value_tracker.verify > 0.5) and
                        param.lambda_optimizer.contact_switch_confidence > 0.05
                        and any(idx not in blocked for idx in incumbents)):
                    param.lambda_optimizer.lock_contact_patch = True

            start_time = time.time()
            best_contact_point, normal, min_error, max_error, _ = param.lambda_optimizer.choose_contact_points(
                target_pose_eval,
                current_pose_eval,
                gravity,
                visible_point_idx,
                contact_anchor_local=current_tip_local,
                # A gated-contact mode must not rank a zero-wrench local
                # optimum as a successful contact.  This is especially
                # important for the nearest p_arm branch, which reuses the
                # candidate buffers produced here.
                force_required=bool(use_gated_contact_policy),
            )
            cached_x_plus = getattr(param.lambda_optimizer, 'last_best_x_plus', None)
            cached_force = getattr(param.lambda_optimizer, 'last_best_force', None)
            cached_cost = getattr(param.lambda_optimizer, 'last_best_cost', None)
            reference_error = (float(cached_cost) if cached_cost is not None and
                               np.isfinite(float(cached_cost)) else float(min_error))
            choose_dt = time.time() - start_time
            choose_times.append(choose_dt)

            best_contact_track_world, best_contact_world, best_normal_world = _sphere_center_on_patch(
                curr_q[:3], R_obj_to_world, best_contact_point, normal,
                fingertip_radius, args.ideal_contact_surface_margin)
            attract_point_world = best_contact_world - args.attract_point_comp * best_normal_world
            attract_point_world[2] = max(attract_point_world[2], best_contact_world[2])
            p_arm_track_world = best_contact_track_world
            p_arm_surface_world = best_contact_world

            if use_ideal_object_pose or use_ideal_contact_pose:
                p_arm_world = best_contact_world
                x_plus_opt = cached_x_plus
                error = float(cached_cost) if cached_cost is not None else float(min_error)
                info = {
                    'control_input': cached_force if cached_force is not None else np.zeros(3),
                    'solver_failed': cached_x_plus is None,
                }
                selected_for_exec = getattr(param.lambda_optimizer, 'last_selected_idx', None)
                if selected_for_exec is not None:
                    param.lambda_optimizer.last_executed_idx = int(selected_for_exec)
                    param.lambda_optimizer.last_executed_x_plus = cached_x_plus
                    param.lambda_optimizer.last_executed_cost = cached_cost
                    param.lambda_optimizer.last_executed_force = cached_force
            else:
                # rollout and --ideal_contact_switch: execute the nearest
                # *sampled* contact, under the same lock / confidence /
                # block / curvature gates used for best_contact_world.
                p_arm_local, p_arm_normal_out, x_plus_opt, error, info = param.lambda_optimizer.resolve_executed_contact(
                    current_tip_local, visible_point_idx,
                    target_pose_eval, current_pose_eval, gravity,
                    sphere_radius=max(1e-4, float(fingertip_radius) +
                                      float(args.ideal_contact_surface_margin)))
                # Lambda optimizes the mesh surface point, while the MPC and
                # MuJoCo fingertip state use the fingertip sphere centre.  The
                # old switch path passed the surface point directly, placing
                # the MPC target about one fingertip radius inside the object.
                # Keep the same sphere-centre construction as ideal_contact_pose.
                p_arm_surface_world = R_obj_to_world @ p_arm_local + curr_q[:3]
                p_arm_inward_local = -np.asarray(p_arm_normal_out, dtype=np.float64)
                p_arm_inward_world = R_obj_to_world @ p_arm_inward_local
                p_arm_inward_world /= max(float(np.linalg.norm(p_arm_inward_world)), 1e-9)
                p_arm_track_world = p_arm_surface_world - max(
                    1e-4, float(fingertip_radius) + float(args.ideal_contact_surface_margin)
                ) * p_arm_inward_world
                p_arm_world = p_arm_track_world

                p_arm_force = np.asarray(info.get('control_input', np.zeros(3)), dtype=np.float64).reshape(-1)
                solver_ok = (
                    not bool(info.get('solver_failed', False)) and
                    np.isfinite(float(error)) and
                    p_arm_force.size >= 3 and np.isfinite(p_arm_force[:3]).all() and
                    float(np.linalg.norm(p_arm_force[:3])) > 1e-3
                )
                value_tracker.reset_arm(getattr(param.lambda_optimizer, 'last_executed_idx', None))
                value_info = value_tracker.update_values(
                    reference_error, error, solver_ok=solver_ok)
                # Conservative value: a p_arm that overestimates how good it
                # is (underestimates cost) is not executed.  Fall back to
                # the ranked best so high-quality contacts are not skipped.
                if not value_info['accept_p_arm']:
                    p_arm_world = best_contact_track_world
                    p_arm_track_world = best_contact_track_world
                    p_arm_surface_world = best_contact_world
                    x_plus_opt = cached_x_plus
                    error = float(cached_cost) if cached_cost is not None else float(min_error)
                    info = {
                        'control_input': cached_force if cached_force is not None else np.zeros(3),
                        'solver_failed': cached_x_plus is None,
                    }
                    selected_for_exec = getattr(param.lambda_optimizer, 'last_selected_idx', None)
                    if selected_for_exec is not None:
                        param.lambda_optimizer.last_executed_idx = int(selected_for_exec)
                        param.lambda_optimizer.last_executed_x_plus = cached_x_plus
                        param.lambda_optimizer.last_executed_cost = cached_cost
                        param.lambda_optimizer.last_executed_force = cached_force
                    exec_quality = 1.0
                else:
                    exec_quality = float(value_info['quality'])
                verify_now, q_dist = value_tracker.update_verify(
                    exec_quality, float(np.linalg.norm(curr_q[7:10] - p_arm_world)))
                value_info['verify'] = verify_now
                value_info['q_dist'] = q_dist
                value_info['exec_quality'] = exec_quality

            exec_inward = p_arm_surface_world - p_arm_track_world
            exec_inward = exec_inward / max(float(np.linalg.norm(exec_inward)), 1e-9)
            attract_point_world = p_arm_surface_world - args.attract_point_comp * exec_inward
            attract_point_world[2] = max(attract_point_world[2], p_arm_surface_world[2])
            if filtered_attract is None:
                filtered_attract = attract_point_world.copy()
            else:
                filtered_attract = alpha * attract_point_world + (1.0 - alpha) * filtered_attract

            if use_ideal_contact_pose:
                # Keep one objective: sit on the selected patch or go around
                # the object.  Press 1.5 mm along the inward normal so the
                # quadratic equilibrium is in contact, not 1 mm outside.
                verify_cost = 0.0
                patch_out = best_contact_track_world - best_contact_world
                patch_out = patch_out / max(float(np.linalg.norm(patch_out)), 1e-9)
                mpc_virtual_point = best_contact_track_world - 0.0015 * patch_out
                mpc_contact_point = best_contact_world
            elif use_switch_policy:
                # Soft verify only blends the contact term.  The attract
                # target is the executed press point (same as pose), not
                # the 10 cm waypoint: otherwise the ball sits on the via
                # and q_dist never rises, so verify stays 0 forever.
                verify_cost = float(value_info.get('verify', 0.0))
                patch_out = p_arm_track_world - p_arm_surface_world
                patch_out = patch_out / max(float(np.linalg.norm(patch_out)), 1e-9)
                exec_press = p_arm_track_world - 0.0015 * patch_out
                mpc_virtual_point = exec_press
                mpc_contact_point = exec_press
            else:
                verify_cost = 1.0
                mpc_virtual_point = filtered_attract
                mpc_contact_point = p_arm_world

            selected_idx = getattr(param.lambda_optimizer, 'last_selected_idx',
                                   getattr(param.lambda_optimizer, 'last_best_idx', None))
            executed_idx = getattr(param.lambda_optimizer, 'last_executed_idx', None)
            global_idx = getattr(param.lambda_optimizer, 'last_global_idx', None)
            global_cost = getattr(param.lambda_optimizer, 'last_global_total_cost', None)
            print(f'花费时间: {choose_dt:.4f}')
            print('min error:', min_error, 'max error', max_error, 'actual error:', error)
            print("verify cost:", None if verify_cost is None else round(float(verify_cost), 4),
                  "p_arm_quality:", None if not value_info else round(float(value_info.get('quality', 0.0)), 4),
                  "accept_p_arm:", None if not value_info else int(bool(value_info.get('accept_p_arm', False))),
                  "v_best:", None if not value_info or value_info.get('v_best') is None else round(float(value_info['v_best']), 4),
                  "v_arm:", None if not value_info or value_info.get('v_arm') is None else round(float(value_info['v_arm']), 4),
                  "adv:", None if not value_info or value_info.get('adv') is None else round(float(value_info['adv']), 4),
                  "q_dist:", None if not value_info else round(float(value_info.get('q_dist', 0.0)), 4),
                  "pose_pos_err:", float(metrics.comp_pos_error(curr_q[0:3], param.target_p_)),
                  "pose_pos_vec:", np.round(np.asarray(curr_q[0:3], dtype=float) - param.target_p_, 4).tolist(),
                  "pose_rot_err:", float(metrics.comp_quat_error(curr_q[3:7], param.target_q_)),
                  "ball_to_best_contact:", round(float(np.linalg.norm(curr_q[7:10] - best_contact_world)), 6),
                  "ball_to_p_arm:", round(float(np.linalg.norm(curr_q[7:10] - p_arm_world)), 6),
                  "ball_to_virtual:", round(float(np.linalg.norm(curr_q[7:10] - mpc_virtual_point)), 6),
                  "best_contact_world:", np.round(best_contact_world, 4).tolist(),
                  "selected_idx:", selected_idx,
                  "executed_idx:", executed_idx,
                  "global_idx:", global_idx,
                  "selected_cost:", None if cached_cost is None else round(float(cached_cost), 6),
                  "global_cost:", None if global_cost is None else round(float(global_cost), 6),
                  "locked:", int(bool(param.lambda_optimizer.lock_contact_patch)),
                  "confidence:", round(float(param.lambda_optimizer.contact_switch_confidence), 3),
                  "dwell:", int(getattr(param.lambda_optimizer, '_dwell_steps', 0)),
                  "blocked:", len(getattr(param.lambda_optimizer, '_blocked_contact_indices', {})),
                  "lambda_backend:", getattr(param.lambda_optimizer, 'last_solver_status', 'unknown'),
                  "if_contact:", int(if_contact))
            if use_ideal_contact_pose:
                tip = np.asarray(curr_q[7:10], dtype=np.float64)
                obj = np.asarray(curr_q[:3], dtype=np.float64)
                patch_n = best_contact_track_world - best_contact_world
                patch_n = patch_n / max(float(np.linalg.norm(patch_n)), 1e-9)
                radial = tip - obj
                r = float(np.linalg.norm(radial))
                align = float(np.dot(radial / max(r, 1e-9), patch_n))
                chord = best_contact_track_world - tip
                chord_len = max(float(np.linalg.norm(chord)), 1e-9)
                d_line = float(np.linalg.norm(np.cross(chord, obj - tip))) / chord_len
                print("detour:", {
                    "align": round(align, 4),
                    "d_line": round(d_line, 4),
                    "r": round(r, 4),
                    "dist_goal": round(float(np.linalg.norm(tip - mpc_virtual_point)), 4),
                    "tip_z": round(float(tip[2]), 4),
                    "goal_z": round(float(mpc_virtual_point[2]), 4),
                    "if_contact": int(if_contact),
                    "selected_idx": selected_idx,
                    "global_idx": global_idx,
                    "locked": int(bool(param.lambda_optimizer.lock_contact_patch)),
                    "confidence": round(float(param.lambda_optimizer.contact_switch_confidence), 3),
                    "dwell": int(getattr(param.lambda_optimizer, '_dwell_steps', 0)),
                    "pose_applies": int(pose_apply_count),
                })

            env.show_target(mpc_virtual_point)
            env.show_best_contact(best_contact_world)

            sol = mpc.plan_once(
                param.target_p_,
                param.target_q_,
                curr_q,
                phi_vec,
                jac_mat,
                verify_cost_param=verify_cost,
                virtual_point=mpc_virtual_point,
                contact_point=mpc_contact_point,
                sol_guess=param.sol_guess_)
            param.sol_guess_ = sol['sol_guess']
            obj_qpos_before = env.data_.qpos[:7].copy()
            env.step(sol['action'])
            ideal_pose_applied = False
            contact_distance = float('inf')
            apply_scale = None
            if use_rollout:
                mujoco.mj_forward(env.model_, env.data_)
                contact_distance = _contact_distance_after_step(contact, env)
                if contact_distance <= float(args.ideal_contact_distance):
                    pose_apply_count += 1
                    ideal_pose_applied = True
                print('contact_distance:', None if not np.isfinite(contact_distance) else round(contact_distance, 6),
                      'physics_contact:', int(ideal_pose_applied))
            elif use_set_pose:
                q_after = env.get_state()
                ideal_pose_applied = _x_plus_is_usable(x_plus_opt, info)
                if use_ideal_contact_pose or use_ideal_contact_switch:
                    contact_distance = _contact_distance_after_step(contact, env)
                    if contact_distance > float(args.ideal_contact_distance):
                        ideal_pose_applied = False
                if ideal_pose_applied:
                    q_for_apply = q_after.copy()
                    q_for_apply[:7] = obj_qpos_before
                    cur_pos_err = float(metrics.comp_pos_error(q_for_apply[:3], param.target_p_))
                    cur_quat_err = float(metrics.comp_quat_error(q_for_apply[3:7], param.target_q_))
                    cur_score = (cur_pos_err / success_pos_threshold +
                                 cur_quat_err / success_quat_threshold)
                    near_goal = cur_pos_err < 0.04
                    chosen = None
                    chosen_score = cur_score
                    for scale in (1.0, 0.5, 0.25):
                        x_try = _scale_local_increment(x_plus_opt, scale)
                        pred_pos, pred_quat = _predicted_object_pose(q_for_apply, x_try)
                        pred_pos_err = float(metrics.comp_pos_error(pred_pos, param.target_p_))
                        pred_quat_err = float(metrics.comp_quat_error(pred_quat, param.target_q_))
                        # Near the goal, patch 5's full increment finishes
                        # rotation by shoving the object past the target.
                        if near_goal and pred_pos_err > cur_pos_err + 1e-4:
                            continue
                        pred_score = (pred_pos_err / success_pos_threshold +
                                      pred_quat_err / success_quat_threshold)
                        if pred_score < chosen_score - 1e-4:
                            chosen_score = pred_score
                            chosen = x_try
                            apply_scale = scale
                    if chosen is None:
                        ideal_pose_applied = False
                        if near_goal:
                            param.lambda_optimizer.lock_contact_patch = False
                    else:
                        _apply_local_pose_increment(env, q_for_apply, chosen)
                        pose_apply_count += 1
                print('contact_distance:', None if not np.isfinite(contact_distance) else round(contact_distance, 6),
                      'ideal_pose_applied:', int(ideal_pose_applied),
                      'apply_scale:', apply_scale)

            if args.viewer:
                time.sleep(0.01)
            if getattr(env, 'break_out_signal_', False):
                break
            if env.viewer_ is not None and hasattr(env.viewer_, 'is_running') and not env.viewer_.is_running():
                env.break_out_signal_ = True
                break
            rollout_step += 1

            curr_q = env.get_state()
            pos_err_now = float(metrics.comp_pos_error(curr_q[0:3], param.target_p_))
            quat_err_now = float(metrics.comp_quat_error(curr_q[3:7], param.target_q_))
            min_pos_err = min(min_pos_err, pos_err_now)
            min_quat_err = min(min_quat_err, quat_err_now)
            if (pos_err_now < success_pos_threshold) and (quat_err_now < success_quat_threshold):
                consecutive_success_time += 1
            else:
                consecutive_success_time = 0
            pose_score = (pos_err_now / success_pos_threshold +
                          quat_err_now / success_quat_threshold)
            tip_now = np.asarray(curr_q[7:10], dtype=float)
            near_best = (
                float(np.linalg.norm(tip_now - mpc_virtual_point)) < 0.03
                or float(np.linalg.norm(tip_now - best_contact_world)) < 0.03)
            near_p_arm = float(np.linalg.norm(tip_now - p_arm_world)) < 0.03
            if use_ideal_contact_pose:
                progress_idx = getattr(param.lambda_optimizer, 'last_selected_idx', None)
                near_patch = near_best
            else:
                progress_idx = getattr(param.lambda_optimizer, 'last_executed_idx', None)
                # The switch dwell schedule must follow the actually executed
                # nearest patch.  Reaching the ranked best/virtual point is not
                # evidence that p_arm made contact or made progress.
                near_patch = near_p_arm
            # Approach time still does not count.  Once the ball is on the
            # selected / executed patch, a frozen pose or an unusable x_plus
            # is a failed dwell even if set-pose never fired (trunk local
            # optimum).  The same schedule now governs p_arm_world.
            use_dwell = use_gated_contact_policy
            if use_rollout:
                dwell_active = bool(use_dwell and pose_apply_count > 0)
                dwell_dead = False
            else:
                dwell_active = bool(use_dwell and (pose_apply_count > 0 or near_patch))
                dwell_dead = bool(use_dwell and near_patch and not ideal_pose_applied)
            param.lambda_optimizer.note_contact_progress(
                progress_idx,
                pose_score,
                active=dwell_active,
                gamma=float(args.contact_dwell_gamma),
                min_dwell_steps=int(args.contact_dwell_steps),
                dead_increment=dwell_dead,
            )
            if consecutive_success_time > consecutive_success_time_threshold:
                # Last step() already synced the viewer.  Sleep only; an extra
                # viewer.sync() after success was segfaulting.
                if args.viewer:
                    time.sleep(0.35)
                break

        lambda_failures = int(getattr(param.lambda_optimizer, 'acados_failure_count', 0))
        mpc_failures = int(getattr(mpc, 'acados_failure_count', 0))
        if lambda_failures or mpc_failures or getattr(mpc, 'acados_solver_', None) is None:
            print('acados diagnostics:', {
                'lambda_solves': int(getattr(param.lambda_optimizer, 'acados_solve_count', 0)),
                'lambda_failures': lambda_failures,
                'lambda_qp_failures_projected': int(getattr(param.lambda_optimizer, 'acados_qp_failure_count', 0)),
                'lambda_ipopt_fallbacks': int(getattr(param.lambda_optimizer, 'acados_fallback_count', 0)),
                'mpc_solves': int(getattr(mpc, 'acados_solve_count', 0)),
                'mpc_failures': mpc_failures,
                'mpc_init_error': str(getattr(mpc, '_acados_init_error', '')) or None,
            })
        trial_success = rollout_step < max_rollout_length
        choose_arr = np.asarray(choose_times, dtype=np.float64) if choose_times else np.array([0.0])
        print('trial_summary:', {
            'trial': trial_count,
            'mode': ('ideal_object_pose' if use_ideal_object_pose
                     else 'ideal_contact_pose' if use_ideal_contact_pose
                     else 'ideal_contact_switch' if use_ideal_contact_switch
                     else 'rollout' if use_rollout else 'unknown'),
            'success': int(trial_success),
            'steps': rollout_step,
            'pose_applies': pose_apply_count,
            'final_pos_err': round(float(metrics.comp_pos_error(curr_q[0:3], param.target_p_)), 5),
            'final_quat_err': round(float(metrics.comp_quat_error(curr_q[3:7], param.target_q_)), 5),
            'min_pos_err': None if not np.isfinite(min_pos_err) else round(float(min_pos_err), 5),
            'min_quat_err': None if not np.isfinite(min_quat_err) else round(float(min_quat_err), 5),
            'choose_dt_mean': round(float(np.mean(choose_arr)), 5),
            'choose_dt_max': round(float(np.max(choose_arr)), 5),
            'choose_dt_p95': round(float(np.percentile(choose_arr, 95)), 5),
        })

        if save_flag:
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, f"{prefix_data_name}trial_{trial_count}_rollout.json")
            with open(filename, 'w') as f:
                json.dump({
                    "target_obj_pos": param.target_p_.tolist(),
                    "target_obj_quat": param.target_q_.tolist(),
                    "success": rollout_step < max_rollout_length,
                    "trial_number": trial_count,
                    "rollout_steps": rollout_step,
                }, f, indent=4)
        success_rate += 1 if rollout_step < max_rollout_length else 0
        trial_count += 1
        if getattr(env, 'break_out_signal_', False):
            break
        if env.viewer_ is not None and hasattr(env.viewer_, 'is_running') and not env.viewer_.is_running():
            break
        # Scene reset happens at the top of the following loop.

    if env is not None and env.viewer_ is not None:
        env.viewer_.close()
    print(f"Success rate over {trial_num} trials: {success_rate}/{trial_num} = {success_rate/trial_num:.2%}")


if __name__ == '__main__':
    main()
