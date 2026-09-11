import time
import json
import numpy as np
import os
import sys
import trimesh
os.environ.setdefault('ACADOS_SOURCE_DIR', '/home/lab423/scsp/thirdparty/acados')
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.insert(0, parent_dir)
from examples.mpc.fingertips.test.params import ExplicitMPCParams
from planning.mpc_explicit2_acados import MPCExplicitAcados as MPCExplicit
from planning.mpc_implicit import MPCImplicit

from envs.fingertips_env import MjSimulator
from contact.fingertips_collision_detection2 import Contact
from scipy.spatial.transform import Rotation
from utils import metrics, rotations
import argparse


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
    parser.add_argument('--normal_stability_cos', type=float, default=0.90)
    parser.add_argument('--random_init_tilt', action='store_true')
    parser.add_argument('--init_tilt_deg', type=float, default=65.0)
    parser.add_argument('--init_tilt_min_deg', type=float, default=0.0)
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
    parser.add_argument('--escape_clearance', type=float, default=0.012)
    parser.add_argument('--escape_step', type=float, default=0.08)
    parser.add_argument('--max_escape_steps', type=int, default=80)
    parser.add_argument('--escape_route_threshold', type=float, default=0.03)
    parser.add_argument('--ideal_contact_surface_margin', type=float, default=-0.0005)
    parser.add_argument('--spline_escape_cost', type=int, default=1)
    parser.add_argument('--ideal_object_pose', action='store_true')
    parser.add_argument('--ideal_contact_pose', action='store_true')
    parser.add_argument('--ideal_contact_distance', type=float, default=0.006)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--ideal_contact_best_contact', '--ideal_contact_best_contact_pose',
                        dest='ideal_contact_best_contact', action='store_true')
    parser.add_argument('--ideal_contact_tracking_coef', type=float, default=10.0)
    parser.add_argument('--trial_num', type=int, default=100)
    parser.add_argument('--max_rollout_length', type=int, default=5000)
    parser.add_argument('--contact_hold_stall_steps', type=int, default=250)
    return parser


class _SurfaceClearanceSpline:
    """Lift/side waypoint around the object AABB. No acados compile in the loop."""

    def __init__(self, mesh_path, clearance=0.012, samples=17):
        self.mesh = trimesh.load_mesh(mesh_path, process=False)
        self.clearance = float(clearance)
        self.samples = max(7, int(samples))
        self.vertices = np.asarray(self.mesh.vertices, dtype=np.float64)
        from scipy.spatial import cKDTree
        self._vertex_tree = cKDTree(self.vertices)
        bounds = np.asarray(self.mesh.bounds, dtype=np.float64)
        lo, hi = bounds[0], bounds[1]
        self._hull_equations = np.asarray([
            [1.0, 0.0, 0.0, -hi[0]], [-1.0, 0.0, 0.0, lo[0]],
            [0.0, 1.0, 0.0, -hi[1]], [0.0, -1.0, 0.0, lo[1]],
            [0.0, 0.0, 1.0, -hi[2]], [0.0, 0.0, -1.0, lo[2]],
        ])
        self._hull_norms = np.ones(6, dtype=np.float64)

    def _signed_distance(self, points_local):
        points_local = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
        try:
            return np.asarray(trimesh.proximity.signed_distance(self.mesh, points_local), dtype=np.float64)
        except Exception:
            d = self._vertex_tree.query(points_local, k=1, workers=1)[0]
            values = (points_local @ self._hull_equations[:, :3].T
                      + self._hull_equations[:, 3])
            inside = np.all(values <= 1e-8, axis=1)
            hull_clearance = np.min(-values / self._hull_norms, axis=1)
            d = np.where(inside, hull_clearance, d)
            return np.where(inside, d, -d)

    @staticmethod
    def _hermite(p0, p1, p2, t):
        t = float(np.clip(t, 0.0, 1.0))
        if t <= 0.5:
            u = 2.0 * t
            a, b = np.asarray(p0), np.asarray(p1)
            m0 = b - a
            m1 = 0.5 * (np.asarray(p2) - np.asarray(p0))
        else:
            u = 2.0 * t - 1.0
            a, b = np.asarray(p1), np.asarray(p2)
            m0 = 0.5 * (np.asarray(p2) - np.asarray(p0))
            m1 = b - a
        h00 = 2*u**3 - 3*u**2 + 1
        h10 = u**3 - 2*u**2 + u
        h01 = -2*u**3 + 3*u**2
        h11 = u**3 - u**2
        return h00*a + h10*m0 + h01*b + h11*m1

    def _local_points(self, points_world, obj_pos, obj_rot):
        return (np.asarray(obj_rot, dtype=np.float64).T @
                (np.asarray(points_world, dtype=np.float64) - np.asarray(obj_pos, dtype=np.float64)).T).T

    def _path_penalty(self, middle, start, goal, obj_pos, obj_rot):
        ts = np.linspace(0.0, 1.0, max(self.samples, 65))
        path = np.asarray([self._hermite(start, middle, goal, t) for t in ts])
        sdf = self._signed_distance(self._local_points(path, obj_pos, obj_rot))
        interior = sdf[1:-1]
        penetration = np.maximum(self.clearance - (-interior), 0.0)
        return 2.0 * np.sum((middle - 0.5*(start + goal))**2) + 2.0e3 * np.sum(penetration**2)

    def segment_blocked(self, start_world, goal_world, obj_pos, obj_rot):
        start = np.asarray(start_world, dtype=np.float64).reshape(3)
        goal = np.asarray(goal_world, dtype=np.float64).reshape(3)
        samples = np.linspace(0.0, 1.0, 9)[1:-1]
        pts = start + samples[:, None] * (goal - start)
        sdf = self._signed_distance(self._local_points(pts, obj_pos, obj_rot))
        return bool(np.any(sdf > -self.clearance))

    def _candidate_middle(self, start, goal, obj_pos, obj_rot):
        midpoint = 0.5 * (start + goal)
        direction = goal - start
        side = np.cross(direction, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(side) < 1e-8:
            side = np.cross(direction, np.array([0.0, 1.0, 0.0]))
        side /= max(np.linalg.norm(side), 1e-8)
        obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3)
        obj_rot = np.asarray(obj_rot, dtype=np.float64).reshape(3, 3)
        lo, hi = self.mesh.bounds
        corners = np.asarray([[x, y, z] for x in (lo[0], hi[0])
                              for y in (lo[1], hi[1])
                              for z in (lo[2], hi[2])])
        corners_world = (obj_rot @ corners.T).T + obj_pos
        top = float(np.max(corners_world[:, 2]))
        horizontal_extent = float(np.max(np.linalg.norm(
            corners_world[:, :2] - obj_pos[:2], axis=1)))
        middle = midpoint.copy()
        middle[:2] += side[:2] * (horizontal_extent + self.clearance)
        middle[2] = max(float(start[2]), float(goal[2]),
                        top + self.clearance + 0.03)
        return middle

    def plan(self, start_world, goal_world, obj_pos, obj_rot):
        start = np.asarray(start_world, dtype=np.float64).reshape(3)
        goal = np.asarray(goal_world, dtype=np.float64).reshape(3)
        middle = self._candidate_middle(start, goal, obj_pos, obj_rot)
        candidates = [middle]
        for lift in (0.03, 0.06, 0.10):
            alt = middle.copy(); alt[2] += lift; candidates.append(alt)
        direction = goal - start
        side = np.cross(direction, np.array([0., 0., 1.]))
        side /= max(np.linalg.norm(side), 1e-8)
        for sign in (-1.0, 1.0):
            alt = middle.copy(); alt[:2] += sign * side[:2] * (2.0 * self.clearance)
            candidates.append(alt)
        scores = [self._path_penalty(c, start, goal, obj_pos, obj_rot)
                  for c in candidates]
        middle = candidates[int(np.argmin(scores))]
        return np.vstack([start, middle, goal]).astype(np.float32)


def _apply_local_pose_increment(env, q_after, x_plus_local):
    import mujoco
    R_after = Rotation.from_quat([q_after[4], q_after[5], q_after[6], q_after[3]]).as_matrix()
    ideal_pos = q_after[:3] + R_after @ np.asarray(x_plus_local[:3], dtype=np.float64).reshape(3)
    ideal_qrel = np.asarray(x_plus_local[3:7], dtype=np.float64).reshape(4)
    ideal_quat = rotations.quaternion_multiply(q_after[3:7], ideal_qrel)
    ideal_quat = ideal_quat / max(np.linalg.norm(ideal_quat), 1e-9)
    env.data_.qpos[:7] = np.hstack((ideal_pos, ideal_quat))
    env.data_.qvel[:6] = 0.0
    mujoco.mj_forward(env.model_, env.data_)
    return ideal_pos, ideal_quat


def _set_fingertip(env, xyz):
    """Place the sphere centre, used to keep perfect contact after a pose update."""
    import mujoco
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    xyz[2] = max(float(xyz[2]), 0.0)
    env.data_.qpos[7:10] = xyz
    env.data_.qvel[6:] = 0.0
    mujoco.mj_forward(env.model_, env.data_)


def _sphere_center_on_patch(obj_pos, obj_rot, local_point, local_normal, radius, margin):
    world = np.asarray(obj_pos, dtype=np.float64).reshape(3) + (
        np.asarray(obj_rot, dtype=np.float64).reshape(3, 3) @ np.asarray(local_point, dtype=np.float64).reshape(3))
    normal = np.asarray(obj_rot, dtype=np.float64).reshape(3, 3) @ np.asarray(local_normal, dtype=np.float64).reshape(3)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
    return world - max(1e-4, float(radius) + float(margin)) * normal, world, normal


def main(args=None):
    if args is None:
        args = build_parser().parse_args()
    os.environ['MUJOCO_HEADLESS'] = '0' if args.viewer else '1'
    # A 5 mm step makes the first approach to a far-side patch take hundreds
    # of cycles.  The diagnostic still uses MPC to *establish* contact; after
    # that the fingertip stays on the selected patch.
    if args.ideal_contact_best_contact and abs(float(args.mpc_step_limit) - 0.005) < 1e-9:
        args.mpc_step_limit = 0.012

    save_flag = False
    success_rate = 0
    if save_flag:
        save_dir = './examples/mpc/franka/trail/cube'
        prefix_data_name = 'ours_'

    trial_num = max(1, int(args.trial_num))
    success_pos_threshold = 0.02
    success_quat_threshold = 0.015
    consecutive_success_time_threshold = 20
    max_rollout_length = max(1, int(args.max_rollout_length))
    trial_count = 0

    while trial_count < trial_num:
        args.solver = 'acados'
        param = ExplicitMPCParams(args, rand_seed=trial_count, target_type='ground-rotation', model='explicit')
        param.torch_solver = 'acados'
        param.lambda_optimizer.solver = 'acados'
        param.lambda_optimizer.lock_contact_patch = False
        param.lambda_optimizer.contact_switch_confirm_steps = max(
            1, int(args.contact_switch_confirm_steps))
        mpc = MPCExplicit(param) if param.mpc_model == 'explicit' else MPCImplicit(param)
        # After the gate opens, drive the fingertip with the same MPC gains
        # as --ideal_object_pose.  The diagnostic controller slams onto the
        # selected face and then hysteresis cannot switch (14 stays 14).
        mpc_pose = mpc
        fresh_lambda = None
        if args.ideal_contact_best_contact:
            pose_args = argparse.Namespace(**vars(args))
            pose_args.ideal_contact_best_contact = False
            pose_args.mpc_step_limit = 0.005
            pose_args.attract_coef = float(args.attract_coef)
            pose_param = ExplicitMPCParams(
                pose_args, rand_seed=trial_count,
                target_type='ground-rotation', model='explicit')
            pose_param.torch_solver = 'acados'
            pose_param.lambda_optimizer.solver = 'acados'
            pose_param.lambda_optimizer.lock_contact_patch = False
            pose_param.lambda_optimizer.contact_switch_confirm_steps = max(
                1, int(args.contact_switch_confirm_steps))
            fresh_lambda = pose_param.lambda_optimizer
            mpc_pose = MPCExplicit(pose_param) if pose_param.mpc_model == 'explicit' else mpc
        escape_planner = _SurfaceClearanceSpline(param.mesh_path_, clearance=args.escape_clearance)
        escape_control_points = None
        escape_goal = None
        reached_lift = False
        held_patch = None
        hold_steps = 0
        refresh_held_x_plus = False
        contact_established = False
        rank_anchor_world = None
        choose_times = []
        pose_apply_count = 0
        switch_count = 0
        min_track_dist = float('inf')

        contact = Contact(param)
        env = MjSimulator(param)
        fingertip_radius = float(np.asarray(env.model_.geom('fingertip0').size).reshape(-1)[0])

        rollout_step = 0
        consecutive_success_time = 0
        min_pos_err = float('inf')
        min_quat_err = float('inf')
        verify_cost = 0
        low_err_coef = args.low_err_coef
        upper_err_coef = args.upper_err_coef
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
            # Before the first gated apply, prefer the visible hemisphere so
            # the first approach is not aimed through the object.  After
            # contact is established, rank like --ideal_object_pose: full
            # candidate set, default switch hysteresis, and a *stationary*
            # fingertip anchor.  Using the snapped sphere as the anchor
            # locks the incumbent patch (geodesic + 20% cost margin).
            rank_like_set_pose = bool(args.ideal_contact_best_contact and contact_established)
            if rank_anchor_world is None:
                rank_anchor_world = np.asarray(curr_q[7:10], dtype=np.float64).copy()
            # After the reset on first contact, the fingertip is back at the
            # set-pose start and MPC moves it.  Use that live tip as the
            # ranking anchor so the same 14→17 style switches can occur.
            rank_anchor_local = current_tip_local
            visible_point_idx = param.lambda_optimizer.get_availble_point_idx(
                curr_q[0:3], R_obj_to_world, param.target_p_, args.ground_height_threshold,
                viewpoint_local=(None if rank_like_set_pose else (
                    current_tip_local if args.ideal_contact_best_contact else None)))

            reuse_held = (
                args.ideal_contact_best_contact and held_patch is not None
                and (not contact_established)
                and hold_steps < int(args.contact_hold_stall_steps))
            stalled_idx = None
            if (args.ideal_contact_best_contact and held_patch is not None
                    and (not contact_established)
                    and hold_steps >= int(args.contact_hold_stall_steps)):
                stalled_idx = held_patch.get('idx')
                param.lambda_optimizer.block_contact_patch(
                    held_patch.get('idx'), cycles=200, radius=0.05)
                held_patch = None
                reached_lift = False
                escape_control_points = None
                escape_goal = None
                reuse_held = False
            start_time = time.time()
            if reuse_held:
                best_contact_point = held_patch['local'].copy()
                normal = held_patch['normal'].copy()
                min_error = held_patch['min_error']
                max_error = held_patch['max_error']
                if refresh_held_x_plus:
                    p_loc, _, x_plus_opt, cost, info = param.lambda_optimizer.optimize_control_input(
                        target_pose_eval, current_pose_eval, gravity, best_contact_point)
                    held_patch['x_plus'] = np.asarray(x_plus_opt, dtype=np.float64).copy()
                    held_patch['force'] = np.asarray(info.get('control_input', np.zeros(3)), dtype=np.float64)
                    held_patch['cost'] = float(cost)
                    refresh_held_x_plus = False
                cached_x_plus = held_patch['x_plus']
                cached_force = held_patch['force']
                cached_cost = held_patch['cost']
                hold_steps += 1
            else:
                best_contact_point, normal, min_error, max_error, _ = param.lambda_optimizer.choose_contact_points(
                    target_pose_eval,
                    current_pose_eval,
                    gravity,
                    visible_point_idx,
                    contact_anchor_local=rank_anchor_local,
                )
                cached_x_plus = getattr(param.lambda_optimizer, 'last_best_x_plus', None)
                cached_force = getattr(param.lambda_optimizer, 'last_best_force', None)
                cached_cost = getattr(param.lambda_optimizer, 'last_best_cost', None)
                new_idx = getattr(param.lambda_optimizer, 'last_selected_idx', None)
                if held_patch is None or held_patch.get('idx') != new_idx:
                    prev_idx = None if held_patch is None else held_patch.get('idx')
                    if prev_idx is not None and prev_idx != new_idx:
                        switch_count += 1
                    elif stalled_idx is not None and stalled_idx != new_idx:
                        switch_count += 1
                    held_patch = {
                        'idx': new_idx,
                        'local': np.asarray(best_contact_point, dtype=np.float64).copy(),
                        'normal': np.asarray(normal, dtype=np.float64).copy(),
                        'min_error': float(min_error),
                        'max_error': float(max_error),
                        'x_plus': None if cached_x_plus is None else np.asarray(cached_x_plus, dtype=np.float64).copy(),
                        'force': None if cached_force is None else np.asarray(cached_force, dtype=np.float64).copy(),
                        'cost': cached_cost,
                    }
                hold_steps = 0
            choose_dt = time.time() - start_time
            choose_times.append(choose_dt)

            best_contact_track_world, best_contact_world, best_normal_world = _sphere_center_on_patch(
                curr_q[:3], R_obj_to_world, best_contact_point, normal,
                fingertip_radius, args.ideal_contact_surface_margin)
            attract_point_world = best_contact_world - args.attract_point_comp * best_normal_world
            attract_point_world[2] = max(attract_point_world[2], best_contact_world[2])

            if args.ideal_contact_best_contact or args.ideal_object_pose:
                p_arm_world = best_contact_world
                x_plus_opt = cached_x_plus
                error = float(cached_cost) if cached_cost is not None else float(min_error)
                info = {
                    'control_input': cached_force if cached_force is not None else np.zeros(3),
                    'solver_failed': cached_x_plus is None,
                }
            else:
                p_arm_local, _, x_plus_opt, error, info = param.lambda_optimizer.optimize_control_input(
                    target_pose_eval, current_pose_eval, gravity, current_tip_local)
                p_arm_world = R_obj_to_world @ p_arm_local + curr_q[:3]

            if filtered_attract is None:
                filtered_attract = attract_point_world.copy()
            else:
                filtered_attract = alpha * attract_point_world + (1.0 - alpha) * filtered_attract

            if args.ideal_contact_best_contact and not contact_established:
                # Track the selected sphere centre.  verify_cost=1 uses the
                # contact-point term (no field / center-reject).  A blocked
                # approach holds one lift waypoint until the fingertip arrives.
                verify_cost = 1
                tip_to_track = float(np.linalg.norm(curr_q[7:10] - best_contact_track_world))
                need_escape = (
                    tip_to_track > float(args.escape_route_threshold) and
                    escape_planner.segment_blocked(
                        curr_q[7:10], best_contact_track_world, curr_q[:3], R_obj_to_world))
                goal_changed = (
                    escape_goal is None or
                    float(np.linalg.norm(best_contact_track_world - escape_goal)) > 0.02)
                if need_escape and (escape_control_points is None or goal_changed):
                    escape_goal = best_contact_track_world.copy()
                    escape_control_points = escape_planner.plan(
                        curr_q[7:10], escape_goal, curr_q[:3], R_obj_to_world)
                    reached_lift = False
                if escape_control_points is not None and need_escape and not reached_lift:
                    lift = np.asarray(escape_control_points[1], dtype=np.float64)
                    if float(np.linalg.norm(curr_q[7:10] - lift)) > 0.025:
                        mpc_virtual_point = lift
                    else:
                        reached_lift = True
                        mpc_virtual_point = best_contact_track_world.copy()
                else:
                    mpc_virtual_point = best_contact_track_world.copy()
                    if not need_escape:
                        escape_control_points = None
                        escape_goal = None
                        reached_lift = False
                mpc_contact_point = mpc_virtual_point.copy()
            else:
                if verify_cost:
                    low_err_coef = args.low_err_coef
                elif float(np.linalg.norm(curr_q[7:10] - filtered_attract)) < 5e-2:
                    low_err_coef *= 1.1
                upper_err_coef = max(args.upper_err_coef if not verify_cost else upper_err_coef - 0.002, 0.7)
                delta_error = max(float(max_error - min_error), 1e-6)
                adaptive = delta_error * upper_err_coef if verify_cost else delta_error * low_err_coef
                verify_cost = 1 if float(error) < (float(min_error) + adaptive) else 0
                mpc_virtual_point = filtered_attract
                mpc_contact_point = p_arm_world

            print(f'花费时间: {choose_dt:.4f}')
            print('min error:', min_error, 'max error', max_error, 'actual error:', error)
            print("verify cost:", verify_cost,
                  "pose_pos_err:", float(metrics.comp_pos_error(curr_q[0:3], param.target_p_)),
                  "pose_rot_err:", float(metrics.comp_quat_error(curr_q[3:7], param.target_q_)),
                  "ball_to_best_contact:", round(float(np.linalg.norm(curr_q[7:10] - best_contact_world)), 6),
                  "ball_to_track_contact:", round(float(np.linalg.norm(curr_q[7:10] - best_contact_track_world)), 6),
                  "ball_to_virtual:", round(float(np.linalg.norm(curr_q[7:10] - mpc_virtual_point)), 6),
                  "best_contact_world:", np.round(best_contact_world, 4).tolist(),
                  "selected_idx:", (held_patch.get('idx') if held_patch is not None
                                    else getattr(param.lambda_optimizer, 'last_selected_idx', None)),
                  "lambda_backend:", getattr(param.lambda_optimizer, 'last_solver_status', 'unknown'),
                  "if_contact:", int(if_contact))

            env.show_target(mpc_virtual_point)
            env.show_best_contact(best_contact_world)

            sol = (mpc_pose if contact_established else mpc).plan_once(
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

            if args.ideal_object_pose or args.ideal_contact_pose or args.ideal_contact_best_contact:
                q_after = env.get_state()
                x_plus_for_pose = None if info.get('solver_failed', False) else x_plus_opt
                x_plus_finite = (x_plus_for_pose is not None and
                                 np.isfinite(np.asarray(x_plus_for_pose)).all())
                track_contact_distance = float(np.linalg.norm(q_after[7:10] - best_contact_track_world))
                measured = contact.get_actual_fingertip_contact()
                if measured is not None and measured.get('point_world') is not None:
                    actual_contact_world = np.asarray(measured['point_world'], dtype=np.float64)
                    ideal_contact_distance = float(np.linalg.norm(q_after[7:10] - actual_contact_world))
                else:
                    ideal_contact_distance = float('inf')
                ideal_contact_gap = abs(ideal_contact_distance - fingertip_radius)
                lambda_for_pose = np.asarray(info.get('control_input', np.zeros(3)), dtype=np.float64).reshape(-1)
                force_norm = float(np.linalg.norm(lambda_for_pose[:3])) if lambda_for_pose.size >= 3 else 0.0
                lambda_valid = (lambda_for_pose.size >= 3 and
                                np.isfinite(lambda_for_pose[:3]).all() and
                                force_norm > 1e-3)
                x_plus_step = 0.0
                if x_plus_finite:
                    x_plus_step = (float(np.linalg.norm(np.asarray(x_plus_for_pose)[:3])) +
                                   float(np.linalg.norm(np.asarray(x_plus_for_pose)[3:7] -
                                                        np.array([1., 0., 0., 0.]))))

                if args.ideal_contact_best_contact:
                    selected_target_near = (
                        contact_established or
                        track_contact_distance <= max(float(args.ideal_contact_distance),
                                                      2.5 * fingertip_radius))
                    # First arrival only opens the gate.  The approach patch
                    # is visibility-filtered and is often not the set-pose
                    # optimum (trial 0: idx 11 vs 14).  Drop the incumbent
                    # and blacklist so the next cycle ranks like
                    # --ideal_object_pose from the same frozen object pose.
                    if selected_target_near and not contact_established:
                        contact_established = True
                        held_patch = None
                        hold_steps = 0
                        refresh_held_x_plus = False
                        if fresh_lambda is not None:
                            jac_now = np.asarray(param.lambda_optimizer.J_tilde, dtype=np.float64).copy()
                            param.lambda_optimizer = fresh_lambda
                            param.lambda_optimizer.update_Jacobian(jac_now)
                            fresh_lambda = None
                        param.lambda_optimizer.last_selected_idx = None
                        param.lambda_optimizer.last_selected_local = None
                        param.lambda_optimizer.last_best_idx = None
                        param.lambda_optimizer._pending_selected_idx = None
                        param.lambda_optimizer._pending_selected_count = 0
                        if getattr(param.lambda_optimizer, '_blocked_contact_indices', None):
                            param.lambda_optimizer._blocked_contact_indices.clear()
                        # Replay --ideal_object_pose from the same initial
                        # fingertip/object state.  Leaving the sphere on the
                        # approach patch kept the incumbent face and blocked
                        # the 14→17 switch that finishes translation.
                        if rank_anchor_world is not None:
                            _set_fingertip(env, rank_anchor_world)
                        filtered_attract = None
                        verify_cost = 0
                        low_err_coef = args.low_err_coef
                        upper_err_coef = args.upper_err_coef
                        param.sol_guess_ = None
                        ideal_pose_applied = False
                        print('contact_established: re-rank like set-pose next step')
                    else:
                        # Match --ideal_object_pose: skip a near-zero force
                        # increment so residual physics can leave a dead
                        # patch (the 14→17 switch).  Always-on applies were
                        # pinning the object on idx 14's local minimum.
                        ideal_pose_applied = (
                            selected_target_near and x_plus_finite
                            and x_plus_step > 1e-8 and lambda_valid)
                        if selected_target_near and not ideal_pose_applied:
                            print('gated_apply_blocked:', {
                                'x_plus_finite': int(x_plus_finite),
                                'x_plus_step': round(float(x_plus_step), 8),
                                'force_norm': round(force_norm, 6),
                                'lambda_valid': int(lambda_valid),
                                'solver_failed': int(bool(info.get('solver_failed', False))),
                            })
                elif args.ideal_object_pose:
                    ideal_pose_applied = x_plus_finite and x_plus_step > 1e-8 and lambda_valid
                else:
                    ideal_pose_applied = (
                        args.ideal_contact_pose and
                        ideal_contact_gap <= max(float(args.ideal_contact_distance), 0.005) and
                        x_plus_finite and x_plus_step > 1e-8 and lambda_valid)
                # Apply lambda on the pre-step object pose it was computed from.
                q_for_apply = q_after.copy()
                q_for_apply[:7] = obj_qpos_before
                in_success_band = False
                if (args.ideal_contact_best_contact or args.ideal_object_pose):
                    pos_before = float(metrics.comp_pos_error(obj_qpos_before[:3], param.target_p_))
                    quat_before = float(metrics.comp_quat_error(obj_qpos_before[3:7], param.target_q_))
                    in_success_band = (pos_before < success_pos_threshold and
                                       quat_before < success_quat_threshold)
                    if ideal_pose_applied and in_success_band:
                        ideal_pose_applied = False

                min_track_dist = min(min_track_dist, track_contact_distance)
                if ideal_pose_applied:
                    _apply_local_pose_increment(env, q_for_apply, x_plus_for_pose)
                    pose_apply_count += 1
                    refresh_held_x_plus = True
                    hold_steps = 0
                    escape_control_points = None
                    escape_goal = None
                    reached_lift = False
                    if args.ideal_contact_best_contact:
                        contact_established = True
                elif ((args.ideal_contact_best_contact and not contact_established)
                      or in_success_band):
                    import mujoco
                    env.data_.qpos[:7] = obj_qpos_before
                    env.data_.qvel[:6] = 0.0
                    mujoco.mj_forward(env.model_, env.data_)
                print('track_contact_distance:', round(track_contact_distance, 6),
                      'ideal_contact_gap:', round(ideal_contact_gap, 6),
                      'ideal_pose_applied:', int(ideal_pose_applied),
                      'hold_steps:', int(hold_steps),
                      'held_idx:', None if held_patch is None else held_patch.get('idx'))

            if args.viewer:
                time.sleep(0.01)
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
            if consecutive_success_time > consecutive_success_time_threshold:
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
        if env.viewer_ is not None:
            env.viewer_.close()
        trial_success = rollout_step < max_rollout_length
        choose_arr = np.asarray(choose_times, dtype=np.float64) if choose_times else np.array([0.0])
        print('trial_summary:', {
            'trial': trial_count,
            'success': int(trial_success),
            'steps': rollout_step,
            'pose_applies': pose_apply_count,
            'contact_established': int(contact_established),
            'patch_switches': switch_count,
            'final_pos_err': round(float(metrics.comp_pos_error(curr_q[0:3], param.target_p_)), 5),
            'final_quat_err': round(float(metrics.comp_quat_error(curr_q[3:7], param.target_q_)), 5),
            'min_pos_err': None if not np.isfinite(min_pos_err) else round(float(min_pos_err), 5),
            'min_quat_err': None if not np.isfinite(min_quat_err) else round(float(min_quat_err), 5),
            'min_track_dist': None if not np.isfinite(min_track_dist) else round(float(min_track_dist), 6),
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

    print(f"Success rate over {trial_num} trials: {success_rate}/{trial_num} = {success_rate/trial_num:.2%}")


if __name__ == '__main__':
    main()
