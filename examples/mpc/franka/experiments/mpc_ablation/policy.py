"""Direct contact targets for A1/A2."""
import time

import numpy as np

from examples.mpc.fingertips.test.test_0902 import (
    compute_rollout_contact_via as _baseline_policy,
    _candidate_solution,
    _sphere_center_on_patch,
)
from utils import rotations


def compute_ablation_contact_policy(param, args, curr_q, r_obj_to_world, gravity,
        jac_mat_env, fingertip_radius, value_tracker, model_cost_conf, approach_via,
        arrived_hold, arrived_dest_idx, floor_ground=.012, floor_z=0.,
        support_point=None, support_normal=None):
    if param.ablation_mode == 'a6':
        raise ValueError('A6 uses the independent AdamW trajectory planner')
    if param.ablation_mode not in ('a1', 'a2'):
        return _baseline_policy(param, args, curr_q, r_obj_to_world, gravity,
            jac_mat_env, fingertip_radius, value_tracker, model_cost_conf,
            approach_via, arrived_hold, arrived_dest_idx, floor_ground=floor_ground,
            floor_z=floor_z, support_point=support_point, support_normal=support_normal)
    started = time.perf_counter()
    opt = param.lambda_optimizer
    tip_local = r_obj_to_world.T @ (curr_q[7:10] - curr_q[:3])
    target_quat = rotations.quaternion_multiply(
        rotations.quaternion_conjugate(curr_q[3:7]), param.target_q_)
    target = np.r_[r_obj_to_world.T @ (param.target_p_ - curr_q[:3]), target_quat]
    current = np.array([0., 0., 0., 1., 0., 0., 0.])
    opt.update_Jacobian(jac_mat_env, phi=getattr(param, 'lambda_env_phi_', None))
    ids = opt.get_availble_point_idx(curr_q[:3], r_obj_to_world, param.target_p_,
        args.ground_height_threshold, viewpoint_local=None, heading_filter=False,
        floor_z=floor_z, support_point=support_point, support_normal=support_normal)
    ids = opt.filter_rankable_indices(ids)
    opt.lock_contact_patch = False
    cached_pose = cached_force = cached_cost = None
    best_idx = None
    if param.ablation_mode == 'a1':
        point, inward, min_error, max_error, _ = opt.choose_contact_points(
            target, current, gravity, ids, contact_anchor_local=None,
            v_last=None, force_required=True, query_local=tip_local)
        # Match the baseline's best-contact guide, including its top-k rule.
        best_idx = opt.choose_nearby_topk_idx(tip_local)
        if best_idx is None:
            best_idx = opt.last_global_idx
        if best_idx is not None:
            best_idx = int(best_idx)
            point, inward = opt.sample_point[best_idx], opt.normal[best_idx]
            cached_pose, cached_force, cached_cost = _candidate_solution(opt, best_idx)
        track, surface, _ = _sphere_center_on_patch(curr_q[:3], r_obj_to_world,
            point, inward, fingertip_radius, args.ideal_contact_surface_margin)
        opt.last_executed_idx = best_idx
        opt.last_executed_x_plus = cached_pose
        opt.last_executed_force = cached_force
        opt.last_executed_cost = cached_cost
        pose, error = cached_pose, cached_cost
        info = dict(control_input=np.zeros(3) if cached_force is None else cached_force,
                    solver_failed=cached_pose is None)
        source = 'best_contact_point_world'
    else:
        # No ranking, best-target acceptance, arrival override, or transition
        # predictor is consulted. p_arm is solved at the nearest eligible patch.
        opt._acados_fallbacks_this_cycle = 0
        radius = max(1e-4, fingertip_radius + args.ideal_contact_surface_margin)
        point, outward, pose, error, info = opt.resolve_executed_contact(
            tip_local, ids, target, current, gravity, v_last=None, sphere_radius=radius)
        actual = getattr(param, 'lambda_actual_contact_', None)
        if actual is not None:
            point = r_obj_to_world.T @ (np.asarray(actual['point_world']) - curr_q[:3])
            outward = r_obj_to_world.T @ np.asarray(actual['normal_world'])
            from planning.measured_contact import measured_patch_index
            idx = measured_patch_index(opt, point, outward, opt.last_executed_idx)
            opt.last_executed_idx = int(idx)
            point, outward, pose, error, info = opt.optimize_control_input(
                target, current, gravity, p_arm=point, sample_idx=idx,
                contact_point_local=point, contact_normal_inward=-outward)
        track, surface, _ = _sphere_center_on_patch(curr_q[:3], r_obj_to_world,
            point, -np.asarray(outward), fingertip_radius, args.ideal_contact_surface_margin)
        min_error = max_error = error
        source = 'p_arm_world'
    # Executor metadata follows the same direct target. The A2 best fields are
    # visualization placeholders; no best-contact computation feeds this path.
    return dict(verify_cost=1., mpc_virtual_point=np.asarray(track).copy(),
        mpc_contact_point=np.asarray(track).copy(), p_arm_world=np.asarray(track).copy(),
        p_arm_track_world=np.asarray(track).copy(), p_arm_surface_world=np.asarray(surface).copy(),
        best_contact_world=np.asarray(surface).copy(), best_contact_point_world=np.asarray(track).copy(),
        best_contact_track_world=np.asarray(track).copy(), best_contact_idx=best_idx,
        x_plus_opt=pose, error=error, info=info,
        min_error=min_error, max_error=max_error, cached_cost=cached_cost,
        cached_x_plus=cached_pose, cached_force=cached_force,
        choose_dt=time.perf_counter() - started, escape_on=False,
        arrived_hold=False, arrived_dest_idx=None, arrived_at_best=False,
        value_info=dict(verify=1., path_blocked=False, via_phase='direct',
            contact_is_via=False, switch_observed_contact=False,
            accept_p_arm=param.ablation_mode == 'a2', target_source=source,
            actual_contact_idx=opt.last_executed_idx,
            using_observed_contact=getattr(param, 'lambda_actual_contact_', None) is not None))
