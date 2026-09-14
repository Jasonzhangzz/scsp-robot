import time
import json
from collections import deque
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
    parser.add_argument('--contact_switch_margin_ratio', type=float, default=0.08)
    parser.add_argument('--contact_switch_margin_abs', type=float, default=0.001)
    parser.add_argument('--contact_switch_confirm_steps', type=int, default=5)
    parser.add_argument('--contact_dwell_gamma', type=float, default=0.70,
                        help='Multiply switch-confidence each stagnant on-patch step.')
    parser.add_argument('--contact_dwell_steps', type=int, default=4,
                        help='Grace steps on a patch before confidence starts decaying.')
    parser.add_argument('--model_cost_error_threshold', type=float, default=6.0,
                        help='Accumulate (predicted-actual) lambda pose-cost reduction '
                             'until this value, then fully tighten verify.')
    parser.add_argument('--model_cost_error_eps', type=float, default=1e-6,
                        help='Ignore predicted-vs-actual cost-reduction noise below this.')
    parser.add_argument('--model_cost_error_min_steps', type=int, default=3,
                        help='Minimum contact steps to reach full tightness '
                             '(caps the per-step miss).')
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
    parser.add_argument('--verify_window', '--verify_window_size',
                        dest='verify_window_size', type=int, default=5,
                        help='Number of recent contact-quality samples used to smooth verify_cost.')
    parser.add_argument('--verify_enter_steps', type=int, default=5,
                        help='Consecutive good window samples required to enter contact mode.')
    parser.add_argument('--verify_hold_steps', type=int, default=30,
                        help='Minimum contact-mode dwell before a release is allowed.')
    parser.add_argument('--verify_release_steps', type=int, default=8,
                        help='Consecutive bad window samples required to release contact mode.')
    parser.add_argument('--ideal_contact_surface_margin', type=float, default=-0.0005)
    parser.add_argument('--spline_escape_cost', type=int, default=1)
    parser.add_argument('--ideal_contact_distance', type=float, default=0.006)
    parser.add_argument('--detour_attract_coef', type=float, default=80.0)
    parser.add_argument('--detour_repel_coef', type=float, default=40.0)
    parser.add_argument('--detour_lift_coef', type=float, default=25.0)
    parser.add_argument('--detour_align_thresh', type=float, default=0.50)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--diagnose_rollout_model', action='store_true',
                        help='Print lambda x_plus versus MuJoCo object/contact displacement each rollout step.')
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


def _lambda_pose_cost(pos, quat, target_p, target_q, pos_coef, ori_coef):
    """Pose term of the lambda objective, in the same units as sol['cost']."""
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    target_p = np.asarray(target_p, dtype=np.float64).reshape(3)
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    target_q = np.asarray(target_q, dtype=np.float64).reshape(4)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-9)
    target_q = target_q / max(float(np.linalg.norm(target_q)), 1e-9)
    dpos = pos - target_p
    ori = 1.0 - float(np.clip(np.dot(quat, target_q), -1.0, 1.0)) ** 2
    return float(pos_coef) * float(np.dot(dpos, dpos)) + float(ori_coef) * ori


def _verify_cost_threshold(min_error, max_error, tightness=0.0):
    """Cost a p_arm must beat to keep verify / drop.

    tightness=0 keeps the full [min_error, max_error] band.  tightness=1
    collapses the gate onto min_error, which is exactly best_contact.
    """
    try:
        lo = float(min_error)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(lo):
        return None
    try:
        hi = float(max_error)
    except (TypeError, ValueError):
        hi = lo
    if not np.isfinite(hi) or hi < lo:
        hi = lo
    tight = float(np.clip(tightness, 0.0, 1.0))
    return lo + (1.0 - tight) * (hi - lo)


def _cost_span_quality(cost, min_error, max_error):
    """1 at min_error, 0 at max_error.  best_contact is always 1."""
    try:
        value = float(cost)
        lo = float(min_error)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value) or not np.isfinite(lo):
        return 0.0
    try:
        hi = float(max_error)
    except (TypeError, ValueError):
        hi = lo
    if not np.isfinite(hi) or hi <= lo + 1e-12:
        return 1.0 if value <= lo + 1e-9 else 0.0
    return float(np.clip((hi - value) / (hi - lo), 0.0, 1.0))


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


def _verify_is_chatter(prev_verify, verify_now, prev_accept=None, accept_now=None,
                       jump=0.25):
    """True when verify/p_arm flips hard enough to waste a control step."""
    if prev_verify is not None and abs(float(verify_now) - float(prev_verify)) >= float(jump):
        return True
    if prev_accept is not None and bool(prev_accept) != bool(accept_now):
        return True
    return False


def _nearest_sample_idx(optimizer, query_local, ids=None):
    points = np.asarray(optimizer.sample_point, dtype=np.float64)
    if ids is None:
        ids = np.arange(len(points), dtype=np.int32)
    else:
        ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    if ids.size == 0:
        return None
    query = np.asarray(query_local, dtype=np.float64).reshape(3)
    return int(ids[int(np.argmin(np.linalg.norm(points[ids] - query[None, :], axis=1)))])


def _sample_world(optimizer, sample_idx, obj_pos, obj_rot):
    local = np.asarray(optimizer.sample_point[int(sample_idx)], dtype=np.float64).reshape(3)
    return np.asarray(obj_pos, dtype=np.float64).reshape(3) + (
        np.asarray(obj_rot, dtype=np.float64).reshape(3, 3) @ local)


def _same_contact_patch(optimizer, idx_a, idx_b, radius=None):
    if idx_a is None or idx_b is None:
        return False
    if int(idx_a) == int(idx_b):
        return True
    try:
        limit = float(optimizer.contact_switch_radius if radius is None else radius)
        return float(optimizer.sample_geodesic[int(idx_a), int(idx_b)]) <= limit
    except (AttributeError, IndexError, TypeError):
        return False


def _rollout_dwell_assignment(optimizer, tip_local, tip_world, obj_pos, obj_rot,
                              executed_idx, executed_world, physical_contact,
                              prev_dwell=None):
    """Charge a dwell only when the fingertip is stuck or on the target.

    Free-space travel is inactive.  A contact that is not on the executed
    point is a failed local patch; neighbouring mesh samples stay one
    cluster while the tip remains within 3 cm of that cluster's sample.
    """
    occupied_idx = _nearest_sample_idx(optimizer, tip_local)
    if occupied_idx is None:
        return executed_idx, False, False, None, False, float('inf')
    dist_exec = float('inf')
    if executed_world is not None:
        dist_exec = float(np.linalg.norm(
            np.asarray(tip_world, dtype=float) - np.asarray(executed_world, dtype=float)))
    on_executed = dist_exec <= 0.03
    near_prev = False
    if prev_dwell is not None:
        prev_world = _sample_world(optimizer, prev_dwell, obj_pos, obj_rot)
        near_prev = float(np.linalg.norm(
            np.asarray(tip_world, dtype=float) - prev_world)) <= 0.03
    if on_executed:
        progress_idx = executed_idx if executed_idx is not None else occupied_idx
        return progress_idx, True, False, occupied_idx, True, dist_exec
    # One-frame contact dropouts still count as stuck if the ball has not
    # left a *failed* cluster.  Do not use this hold for the current
    # executed target: that would start decaying the best point during
    # the last centimetres of travel.
    hold_failed = bool(
        near_prev and prev_dwell is not None and
        (executed_idx is None or int(prev_dwell) != int(executed_idx)))
    if physical_contact or hold_failed:
        progress_idx = int(prev_dwell) if hold_failed else occupied_idx
        return progress_idx, True, True, occupied_idx, False, dist_exec
    return executed_idx, False, False, occupied_idx, False, dist_exec


def _object_top_z_world(obj_pos, obj_rot, aabb_lo, aabb_hi):
    lo = np.asarray(aabb_lo, dtype=np.float64).reshape(3)
    hi = np.asarray(aabb_hi, dtype=np.float64).reshape(3)
    half = 0.5 * (hi - lo)
    center_local = 0.5 * (hi + lo)
    rot = np.asarray(obj_rot, dtype=np.float64).reshape(3, 3)
    center_world = np.asarray(obj_pos, dtype=np.float64).reshape(3) + rot @ center_local
    z_extent = (abs(rot[2, 0]) * half[0] + abs(rot[2, 1]) * half[1] + abs(rot[2, 2]) * half[2])
    return float(center_world[2] + z_extent)


def _chord_clears_object(tip, goal, obj, clearance=0.045):
    tip = np.asarray(tip, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    obj = np.asarray(obj, dtype=float).reshape(3)
    chord = goal - tip
    clen = float(np.linalg.norm(chord))
    if clen < 1e-9:
        return True
    d_line = float(np.linalg.norm(np.cross(chord, obj - tip))) / clen
    return d_line >= float(clearance)


def _near_blocked_cluster(optimizer, tip_world, obj_pos, obj_rot, radius=0.03):
    blocked = getattr(optimizer, '_blocked_contact_indices', {})
    if not blocked:
        return None
    tip = np.asarray(tip_world, dtype=float).reshape(3)
    best_idx = None
    best_dist = float('inf')
    for idx in blocked:
        try:
            world = _sample_world(optimizer, idx, obj_pos, obj_rot)
        except (AttributeError, IndexError, TypeError):
            continue
        dist = float(np.linalg.norm(tip - world))
        if dist < best_dist:
            best_dist = dist
            best_idx = int(idx)
    if best_idx is not None and best_dist <= float(radius):
        return best_idx
    return None


def _blocked_escape_via(tip, obj, goal, outward_normal, object_top_z=None,
                        lift=0.03, slide=0.03):
    """Fixed-height over-the-top via.  Do not chase the fingertip upward."""
    tip = np.asarray(tip, dtype=float).reshape(3)
    obj = np.asarray(obj, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    n = np.asarray(outward_normal, dtype=float).reshape(3)
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    if float(np.dot(n, tip - obj)) < 0.0:
        n = -n
    away = np.array([n[0], n[1], 0.0], dtype=float)
    if float(np.linalg.norm(away)) < 1e-6:
        away = np.array([tip[0] - obj[0], tip[1] - obj[1], 0.0], dtype=float)
    if float(np.linalg.norm(away)) < 1e-6:
        away = np.array([1.0, 0.0, 0.0], dtype=float)
    away = away / float(np.linalg.norm(away))
    via = np.asarray(tip, dtype=float).copy() + float(lift) * away
    # Fixed height: never chase tip_z.  max(tip_z+2cm, top) lifted the
    # waypoint with the ball and created a mid-air equilibrium.
    if object_top_z is not None:
        via[2] = max(0.02, float(object_top_z) + 0.015)
    else:
        via[2] = max(float(tip[2]) + 0.02, 0.02)
    to_goal = goal - via
    to_goal[2] = 0.0
    inward = -away
    into = float(np.dot(to_goal, inward))
    if into > 0.0:
        to_goal = to_goal - into * inward
    gn = float(np.linalg.norm(to_goal))
    if gn > 1e-6:
        via = via + float(slide) * (to_goal / gn)
    if object_top_z is not None:
        via[2] = max(0.02, float(object_top_z) + 0.015)
    else:
        via[2] = max(float(via[2]), 0.02)
    # Never chase the tip outward.  Rebuilding the via from tip+away
    # each frame is what sent the ball to infinity.
    tip_r = float(np.linalg.norm(tip[:2] - obj[:2]))
    via_r = float(np.linalg.norm(via[:2] - obj[:2]))
    max_r = max(0.04, min(tip_r, 0.06))
    if via_r > max_r + 1e-9:
        via[:2] = obj[:2] + (max_r / via_r) * (via[:2] - obj[:2])
    return via


def _should_escape_blocked(optimizer, occupied_idx, tip_world, obj_pos, obj_rot,
                           executed_world, on_target, object_top_z=None,
                           hold_idx=None, physical_contact=False, released=False):
    """Climb only while glued to a blocked patch; never from mid-air."""
    if on_target:
        return False, None
    tip = np.asarray(tip_world, dtype=float).reshape(3)
    if object_top_z is not None and float(tip[2]) >= float(object_top_z) + 0.012:
        return False, None
    holding = hold_idx is not None
    near_idx = _near_blocked_cluster(
        optimizer, tip, obj_pos, obj_rot, radius=0.04 if holding else 0.03)
    if released:
        glued = bool(physical_contact) and near_idx is not None
        well_below = (object_top_z is None or
                      float(tip[2]) < float(object_top_z) - 0.02)
        if not (glued and well_below):
            return False, None
        holding = False
    if holding:
        if near_idx is None and not physical_contact:
            return False, None
        clearance = 0.055
    else:
        if near_idx is None:
            return False, None
        clearance = 0.045
    if executed_world is not None and _chord_clears_object(
            tip, executed_world, obj_pos, clearance=clearance):
        return False, None
    escape_idx = near_idx if near_idx is not None else (
        int(hold_idx) if hold_idx is not None else occupied_idx)
    if escape_idx is None:
        return False, None
    return True, int(escape_idx)


def _track_best_via(obj, goal, object_top_z):
    """Object-anchored over-the-top waypoint toward best_contact.

    The waypoint is fixed on the object, not on the fingertip.  A
    tip-relative via grows every frame and the ball runs away.
    """
    obj = np.asarray(obj, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    via = obj.copy()
    toward = goal - obj
    toward[2] = 0.0
    tn = float(np.linalg.norm(toward))
    if tn > 1e-6:
        via[:2] = obj[:2] + min(0.025, tn) * (toward[:2] / tn)
    via[2] = max(0.02, float(object_top_z) + 0.015)
    return via


def _arrived_at_best_contact(tip_world, best_surface_world, best_track_world,
                             occupied_idx=None, best_idx=None, optimizer=None,
                             radius=0.03):
    """True when the fingertip has reached the ranked best patch.

    The MPC target is the sphere centre, but a curved tail makes that
    point sit ~1 cm from the mesh sample the ball is already touching.
    Arrival therefore uses the closer of surface / track, and also
    accepts a fingertip that occupies the best geodesic patch.
    """
    tip = np.asarray(tip_world, dtype=float).reshape(3)
    dists = []
    if best_surface_world is not None:
        dists.append(float(np.linalg.norm(
            tip - np.asarray(best_surface_world, dtype=float).reshape(3))))
    if best_track_world is not None:
        dists.append(float(np.linalg.norm(
            tip - np.asarray(best_track_world, dtype=float).reshape(3))))
    if dists and min(dists) <= float(radius):
        return True
    if (optimizer is not None and occupied_idx is not None and
            best_idx is not None and dists and min(dists) <= 0.05):
        return _same_contact_patch(
            optimizer, occupied_idx, best_idx, radius=radius)
    return False


def _protect_destination_dwell(optimizer, progress_idx, dest_idx,
                               active, dead, radius=0.03):
    """Grazing the ranked best must not blacklist that patch."""
    if dest_idx is None or progress_idx is None:
        return bool(active), bool(dead)
    if _same_contact_patch(optimizer, progress_idx, dest_idx, radius=radius):
        return False, False
    return bool(active), bool(dead)


def _should_track_best_via(tip_world, executed_world, obj_pos, on_target,
                           object_top_z=None, holding=False):
    """Go over the object toward best_contact when no quality p_arm exists."""
    if on_target or executed_world is None:
        return False
    tip = np.asarray(tip_world, dtype=float).reshape(3)
    obj = np.asarray(obj_pos, dtype=float).reshape(3)
    goal = np.asarray(executed_world, dtype=float).reshape(3)
    dist_goal = float(np.linalg.norm(tip - goal))
    # A tail-like best_contact sits far from the COM.  The far-drift
    # pull-back must not keep the via once the tip is already there.
    if dist_goal <= 0.03:
        return False
    horiz = float(np.linalg.norm(tip[:2] - obj[:2]))
    # Already drifted away: pull back to the object top, do not disable.
    if horiz > 0.08:
        if dist_goal <= 0.05:
            return False
        return True
    # Over the object: drop onto best_contact.  The via sits at
    # top+1.5 cm; requiring top+1.2 cm left the ball 2 mm short and
    # chasing the waypoint forever.
    if (object_top_z is not None and float(tip[2]) >= float(object_top_z) + 0.005
            and horiz <= 0.04):
        return False
    clearance = 0.055 if holding else 0.045
    return not _chord_clears_object(
        tip, executed_world, obj, clearance=clearance)


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
    # Reject numerically finite but physically impossible one-step solutions.
    # With MuJoCo's 20 ms control interval a fingertip cannot create tens of
    # centimetres of object translation or a near-180-degree spin in one
    # contact solve.  Treating such an acados/IPOPT result as good evidence
    # makes verify_cost accept a divergent critic and leaves the real rollout
    # with a stale, usually vertical, target.
    if float(np.linalg.norm(x_plus[:3])) > 0.03:
        return False
    qrel = x_plus[3:7] / max(float(np.linalg.norm(x_plus[3:7])), 1e-9)
    rot_angle = 2.0 * float(np.arccos(np.clip(abs(qrel[0]), 0.0, 1.0)))
    if rot_angle > 0.6:
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


def _pose_mismatch_diagnostics(q_before, q_after, x_plus_opt,
                               p_arm_surface_world=None, contact=None):
    """Compare lambda's local one-step pose prediction with MuJoCo.

    ``x_plus_opt`` is an increment because the lambda call uses an identity
    local pose.  MuJoCo's free-joint qpos is in the world/body convention, so
    compare both the predicted absolute pose and the actual displacement
    expressed in the pre-step object frame.
    """
    out = {}
    before = np.asarray(q_before, dtype=np.float64).reshape(-1)
    after = np.asarray(q_after, dtype=np.float64).reshape(-1)
    if before.size < 7 or after.size < 7:
        return out
    try:
        R_before = Rotation.from_quat(
            [before[4], before[5], before[6], before[3]]).as_matrix()
        actual_local_dpos = R_before.T @ (after[:3] - before[:3])
        out['actual_dpos_world'] = float(np.linalg.norm(after[:3] - before[:3]))
        out['actual_dpos_local'] = float(np.linalg.norm(actual_local_dpos))
        out['actual_dpos_local_vec'] = actual_local_dpos.tolist()
        if x_plus_opt is not None:
            x_plus = np.asarray(x_plus_opt, dtype=np.float64).reshape(7)
            pred_pos, pred_quat = _predicted_object_pose(before[:7], x_plus)
            pred_qrel = x_plus[3:7] / max(float(np.linalg.norm(x_plus[3:7])), 1e-9)
            actual_qrel = rotations.quaternion_multiply(
                rotations.quaternion_conjugate(before[3:7]), after[3:7])
            actual_qrel = actual_qrel / max(float(np.linalg.norm(actual_qrel)), 1e-9)
            quat_dot = float(np.clip(abs(np.dot(pred_qrel, actual_qrel)), -1.0, 1.0))
            out.update({
                'pred_dpos_local': float(np.linalg.norm(x_plus[:3])),
                'pred_dpos_local_vec': x_plus[:3].tolist(),
                'pred_dpos_world': float(np.linalg.norm(R_before @ x_plus[:3])),
                'pose_pos_error': float(np.linalg.norm(after[:3] - pred_pos)),
                'pose_rot_error_rad': float(2.0 * np.arccos(quat_dot)),
                'pred_qrel': pred_qrel.tolist(),
                'actual_qrel': actual_qrel.tolist(),
            })
        if p_arm_surface_world is not None and contact is not None:
            measured = contact.get_actual_fingertip_contact()
            if measured is not None and measured.get('point_world') is not None:
                out['contact_point_error'] = float(np.linalg.norm(
                    np.asarray(measured['point_world'], dtype=np.float64)
                    - np.asarray(p_arm_surface_world, dtype=np.float64)))
                out['contact_dist_signed'] = float(measured.get('dist', np.nan))
    except (TypeError, ValueError, FloatingPointError):
        return out
    return out


class ModelCostConfidence:
    """Tighten verify from accumulated lambda-vs-reality cost-reduction error.

    When the fingertip actually executes a contact, compare the pose-cost
    decrease predicted by ``x_plus`` with the decrease MuJoCo produced.
    Only the under-delivery ``max(0, pred - actual)`` is accumulated.
    ``tightness`` in [0, 1] then shrinks the verify cost gate from
    ``max_error`` down to ``min_error``; the sample is never blacklisted.
    """

    def __init__(self, threshold=6.0, eps=1e-6, min_steps=3):
        self.threshold = max(float(threshold), 1e-8)
        self.eps = max(float(eps), 0.0)
        self.min_steps = max(1, int(min_steps))
        self.accum = 0.0
        self.last_pred = None
        self.last_act = None
        self._idx = None

    def tightness(self):
        return float(np.clip(self.accum / self.threshold, 0.0, 1.0))

    def _step_cap(self):
        return self.threshold / float(self.min_steps)

    def observe(self, pred_reduction, act_reduction):
        try:
            pred = float(pred_reduction)
            act = float(act_reduction)
        except (TypeError, ValueError):
            return self.tightness()
        self.last_pred = pred
        self.last_act = act
        if not np.isfinite(pred) or not np.isfinite(act):
            return self.tightness()
        if pred > self.eps and act < pred - self.eps:
            # One optimistic x_plus (pos_coef=500) can miss by >2 cost units.
            # Cap so tightness needs several failed contacts, not one bounce.
            self.accum += min(pred - act, self._step_cap())
        return self.tightness()

    def observe_unusable_prediction(self):
        """A contact with no usable x_plus still failed to deliver a model step."""
        self.accum += min(0.25 * self.threshold, self._step_cap())
        self.last_pred = None
        self.last_act = 0.0
        return self.tightness()

    def note_sample(self, optimizer, sample_idx, merge_radius=0.03):
        """Reset the gate when ranking moves to a new geodesic patch."""
        if sample_idx is None:
            return self.tightness()
        idx = int(sample_idx)
        if self._idx is not None and idx != int(self._idx):
            try:
                geo = float(optimizer.sample_geodesic[int(self._idx), idx])
            except (AttributeError, IndexError, TypeError):
                geo = float('inf')
            if geo > float(merge_radius):
                self.reset()
        self._idx = idx
        return self.tightness()

    def reset(self):
        self.accum = 0.0
        self.last_pred = None
        self.last_act = None


class ContactValueTracker:
    """Value-network style estimates of lambda costs (lower cost = higher value).

    ``best_contact`` is the default executed target.  ``p_arm`` is only a
    lazy hold of that same patch: accept it when it still sits on the
    ranked-best neighbourhood *and* its lambda cost beats the verify
    gate.  In rollout the gate is ``min_error + (1-tightness)*(max_error
    - min_error)``; tightness comes from accumulated model-vs-reality
    cost-reduction error, so a failed local hold is released without
    blacklisting the sample.  At ``best_contact`` the cost is
    ``min_error``, so the ball still drops.  A fingertip-nearest sample
    on another patch is never a reason to turn ``verify_cost`` on.
    """

    def __init__(self, tau=1.0, rel_scale=0.08, rho=0.08, alpha=0.25,
                 beta=0.18, dist_mid=0.022, dist_width=0.006,
                 window_size=5, enter_threshold=0.55, exit_threshold=0.35,
                 confirm_steps=5, min_hold_steps=30, release_steps=8,
                 accept_margin_ratio=0.2, accept_margin_abs=0.001):
        self.tau = float(tau)
        self.rel_scale = float(rel_scale)
        self.rho = float(rho)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.dist_mid = float(dist_mid)
        self.dist_width = float(dist_width)
        self.window_size = max(1, int(window_size))
        self.enter_threshold = float(np.clip(enter_threshold, 0.0, 1.0))
        self.exit_threshold = float(np.clip(exit_threshold, 0.0, 1.0))
        if self.exit_threshold >= self.enter_threshold:
            raise ValueError('verify exit threshold must be below enter threshold')
        self.confirm_steps = max(1, int(confirm_steps))
        self.min_hold_steps = max(0, int(min_hold_steps))
        self.release_steps = max(1, int(release_steps))
        self.accept_margin_ratio = max(0.0, float(accept_margin_ratio))
        self.accept_margin_abs = max(0.0, float(accept_margin_abs))
        self.v_best = None
        self.v_arm = None
        self.verify = 0.0
        self.last_arm_idx = None
        # Keep evidence separate from the EMA.  The EMA damps amplitude,
        # while this window damps the binary decision that selects the
        # contact cost.  A single bad solve therefore cannot turn contact
        # off, just as a single lucky solve cannot turn it on.
        self._target_window = deque(maxlen=self.window_size)
        self.contact_active = False
        self.good_streak = 0
        self.bad_streak = 0
        self.contact_phase_steps = 0
        self._holding_p_arm = False
        self._last_verify_target = 0.0

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
            # A neighbouring mesh sample is a new critic input, so its
            # running cost estimate must be reinitialized.  Do not touch
            # verify or the enter/hold streaks: those are patch-level
            # state.  Decaying them on every best↔neighbour idx change
            # is what made verify_cost chatter.
            self.v_arm = None
            self.last_arm_idx = idx

    def reset_contact(self):
        """Discard contact evidence when execution moves to another patch."""
        self._target_window.clear()
        self.verify = 0.0
        self.contact_active = False
        self.good_streak = 0
        self.bad_streak = 0
        self.contact_phase_steps = 0
        self._holding_p_arm = False
        self._last_verify_target = 0.0

    def update_values(self, c_best, c_arm, solver_ok=True, candidate_costs=None,
                      same_patch=False, near_arm=False, is_best_sample=False,
                      confidence=1.0, stagnant_steps=0, margin_gamma=0.8,
                      min_error=None, max_error=None, tightness=0.0):
        """Score whether p_arm is a lazy hold of the current ranked best."""
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
            'window_mean': (float(np.mean(self._target_window))
                            if self._target_window else 0.0),
            'contact_active': bool(self.contact_active),
            'candidate_scale': None,
            'same_patch': bool(same_patch),
            'cost_ok': False,
            'accept_scale': 1.0,
            'cost_thresh': None,
            'tightness': float(np.clip(tightness, 0.0, 1.0)),
        }
        if c_best is None or not np.isfinite(float(c_best)):
            return info

        c_best = float(c_best)
        self.v_best = self._ema(self.v_best, c_best, self.rho)
        finite_costs = np.asarray(candidate_costs if candidate_costs is not None
                                   else [], dtype=np.float64).reshape(-1)
        finite_costs = finite_costs[np.isfinite(finite_costs)]
        if finite_costs.size >= 2:
            q25, q75 = np.percentile(finite_costs, [25.0, 75.0])
            robust_spread = float((q75 - q25) / 1.349)
        else:
            robust_spread = 0.0
        # Scale quality against the current best, not the spread of every
        # sample.  The full-set IQR is large on a mesh, so a far, weak
        # nearest contact used to look competitive.
        scale = max(self.rel_scale * max(abs(c_best), abs(self.v_best), 1e-4),
                    1e-4)
        info['scale'] = scale
        info['candidate_scale'] = robust_spread
        info['v_best'] = self.v_best

        arm_ok = bool(solver_ok) and c_arm is not None and np.isfinite(float(c_arm))
        quality = 0.0
        cost_ok = False
        use_span_gate = min_error is not None
        cost_thresh = _verify_cost_threshold(
            min_error, max_error, tightness) if use_span_gate else None
        info['cost_thresh'] = cost_thresh
        if arm_ok:
            c_arm = float(c_arm)
            self.v_arm = self._ema(self.v_arm, c_arm, self.alpha)
            # Clipped-double-Q analogue for costs: do not overestimate
            # value (underestimate cost) from a single lucky solve.
            c_arm_cons = max(c_arm, float(self.v_arm))
            adv = c_arm_cons - c_best
            if use_span_gate:
                quality = _cost_span_quality(c_arm, min_error, max_error)
                tighten = 1.0 - float(np.clip(tightness, 0.0, 1.0))
                # Compare the raw lambda cost to the min/max gate.  The
                # EMA (c_arm_cons) lags while min_error falls, and would
                # reject even the current best_contact after one step.
                at_min = float(c_arm) <= float(min_error) + 1e-6
                cost_ok = bool(is_best_sample) or at_min or (
                    cost_thresh is not None and
                    float(c_arm) <= float(cost_thresh) + 1e-6)
            else:
                quality = float(np.exp(-max(adv, 0.0) / (self.tau * scale)))
                tighten = float(np.clip(margin_gamma, 0.0, 1.0)) ** max(
                    0, int(stagnant_steps))
                margin = tighten * (self.accept_margin_abs +
                                    self.accept_margin_ratio * max(abs(c_best), 1e-4))
                cost_ok = c_arm_cons <= c_best + margin
            info['accept_scale'] = tighten
            info['c_arm_cons'] = c_arm_cons
            info['adv'] = adv
            info['v_arm'] = self.v_arm
        # Lazy update only.  A same-patch neighbour that the fingertip
        # has not reached yet must not steal the target from best_contact:
        # that swap made q_dist (and therefore verify) jump every cycle.
        # Once a lazy hold is committed, keep it through cost noise until
        # the sample leaves the patch or blows a looser release margin.
        on_best_or_here = bool(is_best_sample) or bool(near_arm)
        # Model-error tightness is a verify gate, not a blacklist.  The
        # older confidence<1 path still hard-rejects when the caller has
        # not supplied a min/max span.
        trusted = use_span_gate or float(confidence) >= (1.0 - 1e-9)
        if not trusted:
            accept = False
        elif self._holding_p_arm:
            if arm_ok:
                if use_span_gate:
                    at_min = float(info.get('c_arm_cons', c_arm)) <= float(min_error) + 1e-6
                    still_cheap = bool(is_best_sample) or at_min or (
                        cost_thresh is not None and
                        float(c_arm) <= float(cost_thresh) + 1e-6)
                else:
                    tighten = float(info.get('accept_scale', 1.0))
                    release_margin = tighten * (self.accept_margin_abs +
                        2.0 * self.accept_margin_ratio * max(abs(c_best), 1e-4))
                    still_cheap = float(info['c_arm_cons']) <= c_best + release_margin
            else:
                still_cheap = False
            accept = arm_ok and bool(same_patch) and still_cheap
        else:
            accept = arm_ok and bool(same_patch) and cost_ok and on_best_or_here
        self._holding_p_arm = bool(accept)
        info['quality'] = quality
        info['cost_ok'] = bool(cost_ok)
        info['accept_p_arm'] = bool(accept)
        return info

    def update_verify(self, quality, dist_exec, physical_contact=None,
                      on_target=True, stagnant_steps=0, margin_gamma=0.8,
                      tightness=0.0, allow_approach_press=False):
        """Soft policy: rise only when the executed best-patch target is near."""
        dist = float(dist_exec) if np.isfinite(float(dist_exec)) else 1.0
        q_dist = 1.0 / (1.0 + np.exp((dist - self.dist_mid) / self.dist_width))
        quality = float(np.clip(quality, 0.0, 1.0))
        raw = quality * float(q_dist)
        stale = 1.0 - float(np.clip(margin_gamma, 0.0, 1.0)) ** max(
            0, int(stagnant_steps))
        tight = max(float(np.clip(tightness, 0.0, 1.0)), stale)
        # The real switch gate is the min/max cost threshold.  Only lift
        # verify-enter modestly: raising it to 1.0 would block even
        # min_error because q_dist is never exactly one.
        enter = self.enter_threshold + (0.85 - self.enter_threshold) * tight
        # Before contact mode, ignore mediocre / unconfirmed evidence so a
        # rejected nearest sample cannot enter.  After contact mode, keep
        # the continuous score: hard-zeroing one noisy frame emptied the
        # window and slammed verify_cost.
        if not self.contact_active and quality < enter:
            target = 0.0
        else:
            target = raw
        if self.contact_active and quality < enter:
            target = max(target, float(self._last_verify_target), self.exit_threshold)
        # Contact on the wrong patch (elephant foot while the ranked
        # target is on the back) is not evidence.  Do not hold the last
        # target in that case, or verify stays on and pins the ball.
        if not on_target:
            target = 0.0
        elif physical_contact is not None and not physical_contact:
            # Near a feasible min_error target the ball must be allowed to
            # press down.  Requiring physical contact first is what left
            # verify at 0 while the tip chattered 1 cm above best_contact.
            if allow_approach_press and quality >= enter:
                target = raw
            elif self.contact_active:
                target = max(float(self._last_verify_target), self.exit_threshold)
            else:
                target = 0.0
        self._last_verify_target = float(target)
        self._target_window.append(target)
        window_mean = float(np.mean(self._target_window))

        # Windowed enter/hold/release logic follows the stable policy used by
        # the MPPI experiment, but retains a continuous verify_cost for MPC.
        # Use the tightened enter gate so a mediocre local p_arm cannot
        # confirm contact after the model-error threshold has collapsed
        # onto min_error.
        if (len(self._target_window) == self.window_size and
                window_mean >= enter):
            self.good_streak += 1
        else:
            self.good_streak = 0
        if window_mean <= self.exit_threshold:
            self.bad_streak += 1
        else:
            self.bad_streak = 0

        if not self.contact_active:
            if self.good_streak >= self.confirm_steps:
                self.contact_active = True
                self.contact_phase_steps = 0
                self.bad_streak = 0
        else:
            self.contact_phase_steps += 1
            if (self.contact_phase_steps >= self.min_hold_steps and
                    self.bad_streak >= self.release_steps):
                self.contact_active = False
                self.contact_phase_steps = 0
                self.good_streak = 0

        # Rise like an online critic; decay like a slow target network so a
        # one-frame gap does not slam verify_cost back to zero.  During the
        # hold phase, never feed a zero target to the MPC solely because the
        # latest frame was noisy.
        # Do not blend in contact cost before the five-window confirmation.
        effective_target = window_mean if self.contact_active else 0.0
        if self.contact_active and self.contact_phase_steps < self.min_hold_steps:
            effective_target = max(effective_target, self.exit_threshold)
        rate = self.beta if effective_target >= self.verify else 0.35 * self.beta
        self.verify = (1.0 - rate) * self.verify + rate * effective_target
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
    # The default (no explicit mode flag) is also the physical rollout mode.
    # Propagate the resolved mode to ExplicitMPCParams so its MuJoCo-aligned
    # h/Q/actuator configuration is used in that case as well.
    args.rollout = bool(use_rollout)
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

        if bool(getattr(args, 'diagnose_rollout_model', False)):
            full_mass = np.zeros((env.model_.nv, env.model_.nv), dtype=np.float64)
            mujoco.mj_fullM(env.model_, env.data_, full_mass)
            lambda_q = np.asarray(getattr(param.lambda_optimizer, 'obj_inertia', np.zeros((6, 6))))[:6, :6]
            print('rollout_model_config:', {
                'mujoco_timestep': float(env.model_.opt.timestep),
                'frame_skip': int(param.frame_skip_),
                'mujoco_control_dt': float(env.model_.opt.timestep * param.frame_skip_),
                'lambda_h': float(param.lambda_optimizer.h),
                'lambda_Q_diag': np.diag(lambda_q).tolist(),
                'mujoco_object_M_diag': np.diag(full_mass[:6, :6]).tolist(),
                'mujoco_object_mass': float(env.model_.body_mass[env.model_.body('obj').id]),
                'mesh_path': str(getattr(param, 'mesh_path_', 'unknown')),
                'collision_hull': bool(getattr(param, 'collision_hull', False)),
            })

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
            window_size=int(getattr(args, 'verify_window_size', 5)),
            confirm_steps=int(getattr(args, 'verify_enter_steps', 5)),
            min_hold_steps=int(getattr(args, 'verify_hold_steps', 30)),
            release_steps=int(getattr(args, 'verify_release_steps', 8)),
            accept_margin_ratio=0.05,
            accept_margin_abs=0.02,
        )
        value_info = {}
        f_c = 1.0
        dt = env.model_.opt.timestep * env.param_.frame_skip_
        tau = 1.0 / (2.0 * np.pi * f_c)
        alpha = dt / (tau + dt)
        filtered_attract = None
        last_verify_idx = None
        pos_err_before = None
        quat_err_before = None
        escape_hold_idx = None
        escape_released = False
        travel_via_hold = False
        last_verify_cost = None
        last_accept_p_arm = None
        verify_chatter = False
        model_cost_conf = ModelCostConfidence(
            threshold=float(getattr(args, 'model_cost_error_threshold', 6.0)),
            eps=float(getattr(args, 'model_cost_error_eps', 1e-6)),
            min_steps=int(getattr(args, 'model_cost_error_min_steps', 3)))
        pred_reduction = None
        act_reduction = None
        c_now_cost = None

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
            # Hysteresis must not resurrect a patch below the table after the
            # object rotates.  Both selectors already retain an incumbent
            # while it belongs to this reachable candidate set.
            if use_gated_contact_policy:
                visible_point_idx = np.asarray(visible_point_idx, dtype=np.int32)
                incumbents = [int(idx) for idx in (last_idx, last_exec_idx)
                              if idx is not None]
                param.lambda_optimizer.lock_contact_patch = bool(
                    value_tracker.contact_active and
                    value_tracker._holding_p_arm and
                    param.lambda_optimizer.contact_switch_confidence >= (1.0 - 1e-9) and
                    any(idx in visible_point_idx and idx not in blocked
                        for idx in incumbents))

            start_time = time.time()
            # Only treat the fingertip as a lazy-ranking anchor while it
            # already sits on the incumbent patch.  Anchoring to a foot
            # that the ball is bouncing on pulls the local set onto that
            # failed neighbourhood even when the global best is on the back.
            rank_anchor_local = None
            last_global = getattr(param.lambda_optimizer, 'last_global_idx', None)
            if last_idx is not None:
                last_best_world = _sample_world(
                    param.lambda_optimizer, last_idx, curr_q[:3], R_obj_to_world)
                near_selected = float(np.linalg.norm(
                    curr_q[7:10] - last_best_world)) <= float(
                        param.lambda_optimizer.contact_switch_radius)
                # Only lazy-rank around the current *global* patch.  Sitting
                # on a leftover local incumbent must not become the anchor.
                same_as_global = (last_global is None or _same_contact_patch(
                    param.lambda_optimizer, last_idx, last_global, radius=0.03))
                if near_selected and same_as_global:
                    rank_anchor_local = current_tip_local
            best_contact_point, normal, min_error, max_error, _ = param.lambda_optimizer.choose_contact_points(
                target_pose_eval,
                current_pose_eval,
                gravity,
                visible_point_idx,
                contact_anchor_local=rank_anchor_local,
                # Rank from rest, same as --ideal_contact_switch.  Feeding
                # the measured object velocity into every candidate makes
                # coasting dominate the tiny physical-scale wrench and
                # flattens the ranking used to reject a nearby local patch.
                v_last=None,
                # A gated-contact mode must not rank a zero-wrench local
                # optimum as a successful contact.  This is especially
                # important for the nearest p_arm branch, which reuses the
                # candidate buffers produced here.
                force_required=bool(use_gated_contact_policy),
            )
            cached_x_plus = getattr(param.lambda_optimizer, 'last_best_x_plus', None)
            cached_force = getattr(param.lambda_optimizer, 'last_best_force', None)
            cached_cost = getattr(param.lambda_optimizer, 'last_best_cost', None)
            global_cost = getattr(param.lambda_optimizer, 'last_global_total_cost', None)
            candidate_costs = getattr(param.lambda_optimizer, 'last_candidate_costs', None)
            # The target value must describe the best physical candidate, not
            # the incumbent retained by transition hysteresis.  Otherwise a
            # stale/low-quality incumbent raises V_best and makes p_arm look
            # artificially competitive.
            reference_candidates = [global_cost, min_error]
            reference_candidates.extend(np.asarray(
                candidate_costs if candidate_costs is not None else [],
                dtype=np.float64).reshape(-1).tolist())
            reference_candidates = [float(v) for v in reference_candidates
                                    if v is not None and np.isfinite(float(v))]
            reference_error = min(reference_candidates) if reference_candidates else None
            choose_dt = time.time() - start_time
            choose_times.append(choose_dt)

            # Rollout travel is guided by the raw lambda optimum, not the
            # hysteresis-selected incumbent.
            if use_rollout:
                guide_idx = getattr(param.lambda_optimizer, 'last_global_idx', None)
                if guide_idx is not None:
                    guide_idx = int(guide_idx)
                    best_contact_point = param.lambda_optimizer.sample_point[guide_idx]
                    normal = param.lambda_optimizer.normal[guide_idx]
                    cand_ids = getattr(param.lambda_optimizer, 'last_candidate_ids', None)
                    if cand_ids is not None:
                        hits = np.flatnonzero(np.asarray(cand_ids) == guide_idx)
                        if hits.size:
                            loc = int(hits[0])
                            x_buf = getattr(param.lambda_optimizer, 'last_candidate_x_plus', None)
                            f_buf = getattr(param.lambda_optimizer, 'last_candidate_forces', None)
                            c_buf = getattr(param.lambda_optimizer, 'last_candidate_costs', None)
                            if x_buf is not None and loc < len(x_buf):
                                cached_x_plus = x_buf[loc]
                                param.lambda_optimizer.last_best_x_plus = cached_x_plus
                            if f_buf is not None and loc < len(f_buf):
                                cached_force = f_buf[loc]
                                param.lambda_optimizer.last_best_force = cached_force
                            if c_buf is not None and loc < len(c_buf):
                                cached_cost = float(c_buf[loc])
                                param.lambda_optimizer.last_best_cost = cached_cost

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
                    v_last=None,
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
                p_arm_idx = getattr(param.lambda_optimizer, 'last_executed_idx', None)
                best_idx = getattr(param.lambda_optimizer, 'last_global_idx', None)
                if best_idx is None:
                    best_idx = getattr(param.lambda_optimizer, 'last_selected_idx', None)
                occupied_idx = _nearest_sample_idx(param.lambda_optimizer, current_tip_local)
                arrived_at_best = bool(use_rollout) and _arrived_at_best_contact(
                    curr_q[7:10], best_contact_world, best_contact_track_world,
                    occupied_idx, best_idx, param.lambda_optimizer)
                # resolve_executed_contact writes the fingertip-nearest
                # leftover.  On the ranked-best patch that leftover is
                # not the sample we should drop on.
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
                        'control_input': cached_force if cached_force is not None else np.zeros(3),
                        'solver_failed': cached_x_plus is None,
                    }
                same_patch = _same_contact_patch(
                    param.lambda_optimizer, p_arm_idx, best_idx,
                    radius=0.03 if use_rollout else 0.01)
                if arrived_at_best:
                    same_patch = True
                dist_arm = float(np.linalg.norm(curr_q[7:10] - p_arm_world))
                near_arm = dist_arm <= float(param.lambda_optimizer.contact_switch_radius)
                is_best_sample = (best_idx is not None and p_arm_idx is not None and
                                  int(best_idx) == int(p_arm_idx))
                if arrived_at_best:
                    is_best_sample = True
                    near_arm = True
                value_tracker.reset_arm(p_arm_idx)
                if use_rollout:
                    model_cost_conf.note_sample(
                        param.lambda_optimizer,
                        best_idx if best_idx is not None else p_arm_idx)
                model_tightness = model_cost_conf.tightness()
                value_info = value_tracker.update_values(
                    reference_error, error, solver_ok=solver_ok,
                    candidate_costs=candidate_costs, same_patch=same_patch,
                    near_arm=near_arm, is_best_sample=is_best_sample,
                    confidence=param.lambda_optimizer.contact_switch_confidence,
                    stagnant_steps=getattr(
                        param.lambda_optimizer, '_dwell_steps', 0),
                    min_error=min_error if use_rollout else None,
                    max_error=max_error if use_rollout else None,
                    tightness=model_tightness if use_rollout else 0.0)
                # Default execution is the ranked best.  Keep p_arm only as
                # a lazy hold of that same patch so the MPC target does not
                # chatter among neighbouring mesh samples.
                if not value_info['accept_p_arm']:
                    # Do not slam verify here.  A one-frame p_arm reject
                    # (ear-tip normal flip) used to reset the window and
                    # make verify_cost chatter.  Patch changes still
                    # reset below via last_verify_idx.
                    p_arm_world = best_contact_track_world
                    p_arm_track_world = best_contact_track_world
                    p_arm_surface_world = best_contact_world
                    x_plus_opt = cached_x_plus
                    error = float(cached_cost) if cached_cost is not None else float(min_error)
                    info = {
                        'control_input': cached_force if cached_force is not None else np.zeros(3),
                        'solver_failed': cached_x_plus is None,
                    }
                    exec_quality = 1.0
                else:
                    # Valid lazy hold of the ranked-best patch: do not let a
                    # small neighbour-cost gap drop verify and reopen
                    # nearest-point hunting.
                    exec_quality = 1.0
                verify_idx = getattr(param.lambda_optimizer, 'last_executed_idx', None)
                if last_verify_idx is not None and verify_idx is not None:
                    patch_distance = param.lambda_optimizer.sample_geodesic[
                        int(last_verify_idx), int(verify_idx)]
                    if (patch_distance > param.lambda_optimizer.contact_switch_radius or
                            int(last_verify_idx) in blocked):
                        value_tracker.reset_contact()
                last_verify_idx = verify_idx
                measured_contact = contact.get_actual_fingertip_contact()
                physical_contact = bool(measured_contact is not None and
                                        float(measured_contact['dist']) <= 0.0)
                dist_surface = float(np.linalg.norm(curr_q[7:10] - best_contact_world))
                dist_track = float(np.linalg.norm(curr_q[7:10] - best_contact_track_world))
                dist_to_exec = float(np.linalg.norm(curr_q[7:10] - p_arm_world))
                if use_rollout and arrived_at_best:
                    # Curved tails put the sphere-centre target ~1 cm from
                    # the mesh sample.  q_dist uses dist_mid=2.2 cm, so a
                    # 3.1 cm track miss never reaches the enter gate.
                    dist_to_exec = min(dist_to_exec, dist_surface, dist_track, 0.01)
                on_verify_target = bool(
                    arrived_at_best or dist_to_exec <= 0.03)
                if use_rollout and physical_contact and not on_verify_target:
                    value_tracker.reset_contact()
                verify_now, q_dist = value_tracker.update_verify(
                    exec_quality, dist_to_exec,
                    physical_contact=physical_contact if use_rollout else None,
                    on_target=on_verify_target if use_rollout else True,
                    stagnant_steps=getattr(
                        param.lambda_optimizer, '_dwell_steps', 0),
                    tightness=model_tightness if use_rollout else 0.0,
                    allow_approach_press=bool(use_rollout and on_verify_target))
                value_info['occupied_idx'] = occupied_idx
                value_info['on_target'] = on_verify_target
                value_info['arrived_at_best'] = bool(arrived_at_best)
                value_info['verify'] = verify_now
                value_info['q_dist'] = q_dist
                value_info['exec_quality'] = exec_quality
                value_info['same_patch'] = bool(same_patch)
                value_info['window_mean'] = (
                    float(np.mean(value_tracker._target_window))
                    if value_tracker._target_window else 0.0)
                value_info['contact_active'] = bool(value_tracker.contact_active)

            exec_inward = p_arm_surface_world - p_arm_track_world
            exec_inward = exec_inward / max(float(np.linalg.norm(exec_inward)), 1e-9)
            attract_point_world = p_arm_surface_world - args.attract_point_comp * exec_inward
            attract_point_world[2] = max(attract_point_world[2], p_arm_surface_world[2])
            if filtered_attract is None:
                filtered_attract = attract_point_world.copy()
            else:
                filtered_attract = alpha * attract_point_world + (1.0 - alpha) * filtered_attract

            escape_on = False
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
                if use_rollout:
                    top_z = _object_top_z_world(
                        curr_q[:3], R_obj_to_world,
                        getattr(param, 'object_aabb_lo', (-0.06, -0.04, -0.04)),
                        getattr(param, 'object_aabb_hi', (0.06, 0.04, 0.06)))
                    tracking_best = not bool(value_info.get('accept_p_arm', False))
                    if tracking_best:
                        escape_hold_idx = None
                        travel_via_on = _should_track_best_via(
                            curr_q[7:10], p_arm_world, curr_q[:3],
                            bool(value_info.get('on_target', False)),
                            object_top_z=top_z, holding=travel_via_hold)
                        travel_via_hold = bool(travel_via_on)
                        escape_on = bool(travel_via_on)
                        if travel_via_on:
                            mpc_virtual_point = _track_best_via(
                                curr_q[:3], exec_press, top_z)
                            mpc_contact_point = mpc_virtual_point
                            verify_cost = 0.0
                    else:
                        travel_via_hold = False
                        escape_on, escape_idx = _should_escape_blocked(
                            param.lambda_optimizer, value_info.get('occupied_idx'),
                            curr_q[7:10], curr_q[:3], R_obj_to_world, p_arm_world,
                            bool(value_info.get('on_target', False)),
                            object_top_z=top_z, hold_idx=escape_hold_idx,
                            physical_contact=bool(if_contact),
                            released=escape_released)
                        if escape_hold_idx is not None and not escape_on:
                            tip_now = np.asarray(curr_q[7:10], dtype=float)
                            left_cluster = _near_blocked_cluster(
                                param.lambda_optimizer, tip_now, curr_q[:3],
                                R_obj_to_world, radius=0.04) is None
                            height_clear = float(tip_now[2]) >= float(top_z) + 0.012
                            if height_clear or (left_cluster and not if_contact):
                                escape_released = True
                        escape_hold_idx = escape_idx if escape_on else None
                        if escape_on:
                            n_local = np.asarray(
                                param.lambda_optimizer.normal[int(escape_idx)],
                                dtype=float).reshape(3)
                            n_world = R_obj_to_world @ n_local
                            mpc_virtual_point = _blocked_escape_via(
                                curr_q[7:10], curr_q[:3], exec_press, n_world,
                                object_top_z=top_z)
                            mpc_contact_point = mpc_virtual_point
                            verify_cost = 0.0
                            # Keep the failed neighbourhood excluded for the
                            # whole detour.  The per-cycle decrement otherwise
                            # expires mid-climb and ranks the leg again.
                            for key in list(param.lambda_optimizer._blocked_contact_indices):
                                param.lambda_optimizer._blocked_contact_indices[key] = max(
                                    80, int(param.lambda_optimizer._blocked_contact_indices[key]))
            else:
                verify_cost = 1.0
                mpc_virtual_point = filtered_attract
                mpc_contact_point = p_arm_world

            accept_now = bool(value_info.get('accept_p_arm', False)) if value_info else False
            # Accept flips are the designed verify-threshold switch.  Treating
            # them as chatter blacklisted the tail the first time we arrived.
            if use_rollout:
                verify_chatter = _verify_is_chatter(last_verify_cost, verify_cost)
            else:
                verify_chatter = _verify_is_chatter(
                    last_verify_cost, verify_cost, last_accept_p_arm, accept_now)
            last_verify_cost = float(verify_cost)
            last_accept_p_arm = accept_now

            selected_idx = getattr(param.lambda_optimizer, 'last_selected_idx',
                                   getattr(param.lambda_optimizer, 'last_best_idx', None))
            executed_idx = getattr(param.lambda_optimizer, 'last_executed_idx', None)
            global_idx = getattr(param.lambda_optimizer, 'last_global_idx', None)
            global_cost = getattr(param.lambda_optimizer, 'last_global_total_cost', None)
            print(f'花费时间: {choose_dt:.4f}')
            print('min error:', min_error, 'max error', max_error, 'actual error:', error)
            print("verify cost:", None if verify_cost is None else round(float(verify_cost), 4),
                  "verify_chatter:", int(bool(verify_chatter)),
                  "p_arm_quality:", None if not value_info else round(float(value_info.get('quality', 0.0)), 4),
                  "accept_p_arm:", None if not value_info else int(bool(value_info.get('accept_p_arm', False))),
                  "accept_scale:", None if not value_info or value_info.get('accept_scale') is None else round(float(value_info.get('accept_scale', 1.0)), 3),
                  "same_patch:", None if not value_info else int(bool(value_info.get('same_patch', False))),
                  "cost_ok:", None if not value_info else int(bool(value_info.get('cost_ok', False))),
                  "v_best:", None if not value_info or value_info.get('v_best') is None else round(float(value_info['v_best']), 4),
                  "v_arm:", None if not value_info or value_info.get('v_arm') is None else round(float(value_info['v_arm']), 4),
                  "adv:", None if not value_info or value_info.get('adv') is None else round(float(value_info['adv']), 4),
                  "q_dist:", None if not value_info else round(float(value_info.get('q_dist', 0.0)), 4),
                  "verify_window:", None if not value_info else round(float(value_info.get('window_mean', 0.0)), 4),
                  "verify_active:", None if not value_info else int(bool(value_info.get('contact_active', False))),
                  "cost_scale:", None if not value_info or value_info.get('scale') is None else round(float(value_info['scale']), 6),
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
                  "model_tightness:", round(float(model_cost_conf.tightness()), 3),
                  "model_cost_accum:", round(float(model_cost_conf.accum), 4),
                  "cost_thresh:", None if not value_info or value_info.get('cost_thresh') is None
                  else round(float(value_info['cost_thresh']), 6),
                  "pred_dcost:", None if pred_reduction is None else round(float(pred_reduction), 6),
                  "act_dcost:", None if act_reduction is None else round(float(act_reduction), 6),
                  "dwell:", int(getattr(param.lambda_optimizer, '_dwell_steps', 0)),
                  "dwell_idx:", getattr(param.lambda_optimizer, '_dwell_idx', None),
                  "occupied_idx:", None if not value_info else value_info.get('occupied_idx'),
                  "on_target:", None if not value_info else int(bool(value_info.get('on_target', True))),
                  "arrived:", None if not value_info else int(bool(value_info.get('arrived_at_best', False))),
                  "blocked:", len(getattr(param.lambda_optimizer, '_blocked_contact_indices', {})),
                  "escape:", int(bool(escape_on)),
                  "esc_rel:", int(bool(escape_released)),
                  "tip_z:", round(float(curr_q[9]), 4),
                  "virt_z:", round(float(mpc_virtual_point[2]), 4),
                  "exec_z:", round(float(p_arm_world[2]), 4),
                  "obj_z:", round(float(curr_q[2]), 4),
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
            pos_err_before = float(metrics.comp_pos_error(curr_q[0:3], param.target_p_))
            quat_err_before = float(metrics.comp_quat_error(curr_q[3:7], param.target_q_))
            pred_reduction = None
            act_reduction = None
            c_now_cost = None
            if use_rollout:
                c_now_cost = _lambda_pose_cost(
                    curr_q[:3], curr_q[3:7], param.target_p_, param.target_q_,
                    param.lambda_optimizer.pos_coef, param.lambda_optimizer.ori_coef)
                if x_plus_opt is not None and _x_plus_is_usable(x_plus_opt, info):
                    pred_pos, pred_quat = _predicted_object_pose(curr_q[:7], x_plus_opt)
                    c_pred = _lambda_pose_cost(
                        pred_pos, pred_quat, param.target_p_, param.target_q_,
                        param.lambda_optimizer.pos_coef, param.lambda_optimizer.ori_coef)
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
                sol_guess=param.sol_guess_)
            param.sol_guess_ = sol['sol_guess']
            obj_qpos_before = env.data_.qpos[:7].copy()
            env.step(sol['action'])
            ideal_pose_applied = False
            contact_distance = float('inf')
            apply_scale = None
            on_exec_contact = False
            if use_rollout:
                mujoco.mj_forward(env.model_, env.data_)
                contact_distance = _contact_distance_after_step(contact, env)
                measured_contact = contact.get_actual_fingertip_contact()
                if measured_contact is not None and float(measured_contact['dist']) <= 0.003:
                    # Touching the executed point is not itself a successful
                    # pose apply.  A foot/back local contact that does not
                    # move the object toward the goal is a dead dwell.
                    post_tip = np.asarray(env.get_state()[7:10], dtype=float)
                    on_exec_contact = float(np.linalg.norm(post_tip - np.asarray(
                        p_arm_world, dtype=float))) <= 0.03
                    if on_exec_contact:
                        pose_apply_count += 1
                if bool(getattr(args, 'diagnose_rollout_model', False)):
                    mismatch = _pose_mismatch_diagnostics(
                        obj_qpos_before, env.get_state(), x_plus_opt,
                        p_arm_surface_world=p_arm_surface_world, contact=contact)
                    mismatch['object_pos_after'] = np.asarray(env.get_state()[:3], dtype=float).tolist()
                    mismatch['fingertip_pos_after'] = np.asarray(env.get_state()[7:10], dtype=float).tolist()
                    mismatch['command'] = np.asarray(sol['action'], dtype=float).tolist()
                    mismatch['mpc_target'] = np.asarray(mpc_virtual_point, dtype=float).tolist()
                    print('rollout_model_mismatch:', {
                        key: (round(float(value), 7) if isinstance(value, (float, np.floating))
                              else value)
                        for key, value in mismatch.items()
                    })
                print('contact_distance:', None if not np.isfinite(contact_distance) else round(contact_distance, 6),
                      'physics_contact:', int(on_exec_contact))
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
            if use_rollout:
                if c_now_cost is not None:
                    c_after = _lambda_pose_cost(
                        curr_q[:3], curr_q[3:7], param.target_p_, param.target_q_,
                        param.lambda_optimizer.pos_coef, param.lambda_optimizer.ori_coef)
                    act_reduction = c_now_cost - c_after
                    if on_exec_contact and last_accept_p_arm:
                        if pred_reduction is None:
                            model_cost_conf.observe_unusable_prediction()
                        else:
                            model_cost_conf.observe(pred_reduction, act_reduction)
                ideal_pose_applied = bool(
                    on_exec_contact and
                    act_reduction is not None and pred_reduction is not None and
                    act_reduction + float(getattr(args, 'model_cost_error_eps', 1e-6))
                    >= pred_reduction)
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
            occupied_for_log = None
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
                R_now = Rotation.from_quat(
                    [curr_q[4], curr_q[5], curr_q[6], curr_q[3]]).as_matrix()
                tip_local_now = R_now.T @ (tip_now - curr_q[:3])
                post_physical = bool(
                    np.isfinite(contact_distance) and contact_distance <= 0.003)
                (progress_idx, dwell_active, dwell_dead, occupied_for_log,
                 on_exec_patch, _dist_exec_now) = _rollout_dwell_assignment(
                    param.lambda_optimizer, tip_local_now, tip_now,
                    curr_q[:3], R_now,
                    getattr(param.lambda_optimizer, 'last_executed_idx', None),
                    p_arm_world, post_physical,
                    prev_dwell=getattr(param.lambda_optimizer, '_dwell_idx', None))
                if dwell_active and not on_exec_patch:
                    dwell_dead = True
                elif dwell_active and on_exec_patch:
                    # Model-error tightness handles a local hold that
                    # under-delivers.  Do not blacklist this neighbourhood:
                    # best_contact on the same patch (the tail) must stay
                    # visible so the tightened min_error gate can drop.
                    dwell_dead = False
                    dwell_active = False
                dwell_active = bool(use_dwell and dwell_active)
                dwell_dead = bool(use_dwell and dwell_dead)
                if (not last_accept_p_arm) and not on_exec_patch and escape_on:
                    # Over-the-top travel toward best_contact.  Do not
                    # charge the destination; a stuck wrong-patch contact
                    # still keeps its dead increment above.
                    dwell_dead = False
                    dwell_active = False
            else:
                dwell_active = bool(use_dwell and (pose_apply_count > 0 or near_patch))
                dwell_dead = bool(use_dwell and near_patch and not ideal_pose_applied)
            if use_rollout and progress_idx is None:
                progress_idx = (
                    getattr(param.lambda_optimizer, 'last_global_idx', None)
                    or getattr(param.lambda_optimizer, 'last_executed_idx', None))
            if use_rollout and verify_chatter:
                # Tip-normal flips waste steps.  Charge the occupied
                # neighbourhood and decay faster; below the unlock
                # threshold the patch is blacklisted.
                progress_idx = occupied_for_log or progress_idx
                dwell_active = True
                dwell_dead = True
            dest_protected = False
            if use_rollout:
                dest_idx = getattr(param.lambda_optimizer, 'last_global_idx', None)
                dest_protected = bool(
                    dest_idx is not None and progress_idx is not None and
                    _same_contact_patch(
                        param.lambda_optimizer, progress_idx, dest_idx,
                        radius=0.03))
                dwell_active, dwell_dead = _protect_destination_dwell(
                    param.lambda_optimizer, progress_idx, dest_idx,
                    dwell_active, dwell_dead)
            param.lambda_optimizer.note_contact_progress(
                progress_idx,
                pos_err_now if use_rollout else pose_score,
                active=dwell_active,
                gamma=float(args.contact_dwell_gamma),
                min_dwell_steps=int(args.contact_dwell_steps),
                improve_eps=0.002 if use_rollout else 1e-3,
                dead_increment=dwell_dead,
                merge_radius=0.03 if use_rollout else None,
                block_radius=0.03 if use_rollout else None,
                block_cycles=80 if use_rollout else 20,
                time_decay=bool(use_rollout and last_accept_p_arm and not dest_protected),
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
                'lambda_failure_reasons': dict(getattr(param.lambda_optimizer, 'acados_failure_reasons', {})),
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
