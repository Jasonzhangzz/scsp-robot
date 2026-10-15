"""Isolated contact policies for the paper's A1/A2/A3 experiments.

Lambda's model, candidate evaluation and solvers remain the baseline ones.
A1 ignores its predictions; A2 uses its finite cost argmin each cycle.
A3 keeps baseline target selection and omits the lift/place via construction.
MPPI uses the observed object center without surface selection or RS.
"""
import time
import numpy as np
from utils import rotations
from examples.mpc.fingertips.test.test_0902 import (
    compute_rollout_contact_via as _full_contact_policy,
    _arrived_at_best_contact,
    _candidate_solution,
    _floor_slide_away_from_patch,
    _nearest_sample_idx,
    _on_opposite_sides,
    _same_contact_patch,
    _sample_world,
    _should_hold_occupied_contact,
    _sphere_center_on_patch,
)


def _direct_policy(target, surface, best_track, best_surface, best_idx,
                   value_info=None, **diagnostics):
    """Keep the executor payload contract with a direct MPC target."""
    value = dict(value_info or {})
    value.update(path_blocked=False, via_phase='direct', contact_is_via=False,
                 switch_observed_contact=False)
    value.setdefault('accept_p_arm', False)
    value.setdefault('verify', 1.0)
    policy = dict(verify_cost=float(value['verify']),
        mpc_virtual_point=np.asarray(target).copy(),
        mpc_contact_point=np.asarray(target).copy(),
        p_arm_world=np.asarray(target).copy(), p_arm_track_world=np.asarray(target).copy(),
        p_arm_surface_world=np.asarray(surface).copy(),
        best_contact_world=np.asarray(best_surface).copy(),
        best_contact_point_world=np.asarray(best_track).copy(),
        best_contact_track_world=np.asarray(best_track).copy(),
        best_contact_idx=best_idx, value_info=value, escape_on=False,
        arrived_hold=False, arrived_dest_idx=None, arrived_at_best=False,
        x_plus_opt=None, error=0.0, info=dict(control_input=np.zeros(3), solver_failed=False),
        min_error=0.0, max_error=0.0, cached_cost=None, cached_x_plus=None,
        cached_force=None, choose_dt=0.0)
    policy.update(diagnostics)
    return policy


def _nearest_or_best_policy(param, args, curr_q, r_obj_to_world, gravity,
                            jac_mat_env, fingertip_radius, floor_z,
                            support_point, support_normal):
    opt = param.lambda_optimizer
    tip_local = r_obj_to_world.T @ (curr_q[7:10] - curr_q[:3])
    target_quat = rotations.quaternion_multiply(
        rotations.quaternion_conjugate(curr_q[3:7]), param.target_q_)
    target_pose = np.r_[r_obj_to_world.T @ (param.target_p_ - curr_q[:3]), target_quat]
    current_pose = np.array([0., 0., 0., 1., 0., 0., 0.])
    opt.update_Jacobian(jac_mat_env, phi=getattr(param, 'lambda_env_phi_', None))
    ids = opt.get_availble_point_idx(curr_q[:3], r_obj_to_world, param.target_p_,
        args.ground_height_threshold, viewpoint_local=None, heading_filter=False,
        floor_z=floor_z, support_point=support_point, support_normal=support_normal)
    ids = np.asarray(opt.filter_rankable_indices(ids), dtype=np.int32).reshape(-1)
    if not len(ids):
        # Baseline Lambda also falls back to sample zero for an empty set.
        ids = np.array([0], dtype=np.int32)
    radius = max(1e-4, float(fingertip_radius) + float(args.ideal_contact_surface_margin))
    centers = opt.sample_point[ids] - radius * opt.normal[ids]
    near_idx = int(ids[np.argmin(np.linalg.norm(centers - tip_local, axis=1))])
    # Evaluate the same Lambda NLP even in A1 for comparable planning timings.
    # None of the returned scores, forces, poses or selected IDs enters A1's policy.
    opt.lock_contact_patch = False
    opt._blocked_contact_indices.clear()
    started = time.perf_counter()
    opt.choose_contact_points(target_pose, current_pose, gravity, ids,
        contact_anchor_local=None, v_last=None, force_required=True, query_local=tip_local)
    choose_dt = time.perf_counter() - started
    selected = near_idx
    if param.ablation_mode == 'a2':
        candidate_ids = np.asarray(opt.last_candidate_ids, dtype=np.int32).reshape(-1)
        scores = np.asarray(opt.last_candidate_costs, dtype=float).reshape(-1)
        eligible = np.isin(candidate_ids, ids) & np.isfinite(scores)
        if eligible.any():
            locations = np.flatnonzero(eligible)
            selected = int(candidate_ids[locations[np.argmin(scores[locations])]])
        # With no finite Lambda solution, use the geometric nearest fallback.
    track, surface, _ = _sphere_center_on_patch(curr_q[:3], r_obj_to_world,
        opt.sample_point[selected], opt.normal[selected], fingertip_radius,
        args.ideal_contact_surface_margin)
    opt.last_executed_idx = selected
    opt.last_selected_idx = selected
    return _direct_policy(track, surface, track, surface, selected,
        value_info=dict(verify=1.0, accept_p_arm=False, occupied_idx=near_idx),
        choose_dt=choose_dt)


def compute_ablation_contact_policy(param, args, curr_q, r_obj_to_world, gravity,
        jac_mat_env, fingertip_radius, value_tracker, model_cost_conf, approach_via,
        arrived_hold, arrived_dest_idx, floor_ground=0.012, floor_z=0.0,
        support_point=None, support_normal=None):
    if param.ablation_mode == 'mppi':
        center = np.asarray(curr_q[:3], dtype=float)
        # This reference is only executor metadata. The cost uses x[:3] so
        # attraction follows the moving object throughout the MPC horizon.
        return _direct_policy(center, center, center, center, None)
    if param.ablation_mode in ('a1', 'a2'):
        return _nearest_or_best_policy(param, args, curr_q, r_obj_to_world, gravity,
            jac_mat_env, fingertip_radius, floor_z, support_point, support_normal)
    fn = _a3_contact_policy if param.ablation_mode == 'a3' else _full_contact_policy
    return fn(param, args, curr_q, r_obj_to_world, gravity, jac_mat_env,
        fingertip_radius, value_tracker, model_cost_conf, approach_via,
        arrived_hold, arrived_dest_idx, floor_ground=floor_ground, floor_z=floor_z,
        support_point=support_point, support_normal=support_normal)


def _a3_contact_policy(
    param,
    args,
    curr_q,
    r_obj_to_world,
    gravity,
    jac_mat_env,
    fingertip_radius,
    value_tracker,
    model_cost_conf,
    approach_via,
    arrived_hold,
    arrived_dest_idx,
    floor_ground=0.012,
    floor_z=0.0,
    support_point=None,
    support_normal=None,
):
    """Baseline contact selection and RS, ending before via construction."""
    current_tip_local = r_obj_to_world.T @ (curr_q[7:10] - curr_q[:3])
    target_quat_local = rotations.quaternion_multiply(
        rotations.quaternion_conjugate(curr_q[3:7]), param.target_q_)
    target_pose_eval = np.hstack(
        [r_obj_to_world.T @ (param.target_p_ - curr_q[:3]), target_quat_local])
    current_pose_eval = np.array([0., 0., 0., 1., 0., 0., 0.])

    param.lambda_optimizer.update_Jacobian(jac_mat_env, phi=getattr(param, 'lambda_env_phi_', None))
    actual_contact = getattr(param, 'lambda_actual_contact_', None)
    object_cost_stage = getattr(param, 'object_cost_stage_', None)
    post_flip_pose = object_cost_stage == 'pose'
    staged_object_cost = object_cost_stage in ('flip', 'pose')
    visible_point_idx = param.lambda_optimizer.get_availble_point_idx(
        curr_q[0:3], r_obj_to_world, param.target_p_, args.ground_height_threshold,
        viewpoint_local=None, heading_filter=False, floor_z=floor_z,
        support_point=support_point, support_normal=support_normal)
    visible_point_idx = param.lambda_optimizer.filter_rankable_indices(
        visible_point_idx)

    last_idx = getattr(param.lambda_optimizer, 'last_selected_idx', None)
    last_exec_idx = getattr(param.lambda_optimizer, 'last_executed_idx', None)
    blocked = getattr(param.lambda_optimizer, '_blocked_contact_indices', {})
    visible_point_idx = np.asarray(visible_point_idx, dtype=np.int32)
    incumbents = [int(idx) for idx in (last_idx, last_exec_idx) if idx is not None]
    lock_from_tracker = bool(
        (actual_contact is not None if staged_object_cost else value_tracker.contact_active) and
        value_tracker._holding_p_arm and
        param.lambda_optimizer.contact_switch_confidence >= (1.0 - 1e-9) and
        any(idx in visible_point_idx and idx not in blocked for idx in incumbents))

    actual_contact_idx = None
    if actual_contact is not None:
        try:
            actual_point_local = r_obj_to_world.T @ (
                np.asarray(actual_contact['point_world'], dtype=float).reshape(3)
                - curr_q[:3])
            actual_contact_idx = _nearest_sample_idx(
                param.lambda_optimizer, actual_point_local)
        except (KeyError, TypeError, ValueError):
            actual_contact_idx = None
    # Keep developer's trusted-patch continuity during a flip. Ranking a
    # hypothetical far patch each frame must not interrupt a supported turn.
    # The existing measured-progress confidence releases a stalled patch.
    param.lambda_optimizer.lock_contact_patch = bool(lock_from_tracker and not post_flip_pose)

    start_time = time.time()
    rank_anchor_local = None
    last_global = getattr(param.lambda_optimizer, 'last_global_idx', None)
    if last_idx is not None and not post_flip_pose:
        last_best_world = _sample_world(
            param.lambda_optimizer, last_idx, curr_q[:3], r_obj_to_world)
        near_selected = float(np.linalg.norm(
            curr_q[7:10] - last_best_world)) <= float(
                param.lambda_optimizer.contact_switch_radius)
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
        v_last=None,
        force_required=True,
        query_local=current_tip_local,
    )
    cached_x_plus = getattr(param.lambda_optimizer, 'last_best_x_plus', None)
    cached_force = getattr(param.lambda_optimizer, 'last_best_force', None)
    cached_cost = getattr(param.lambda_optimizer, 'last_best_cost', None)
    global_cost = getattr(param.lambda_optimizer, 'last_global_total_cost', None)
    candidate_costs = getattr(param.lambda_optimizer, 'last_candidate_costs', None)
    reference_candidates = [global_cost, min_error]
    reference_candidates.extend(np.asarray(
        candidate_costs if candidate_costs is not None else [],
        dtype=np.float64).reshape(-1).tolist())
    reference_candidates = [float(v) for v in reference_candidates
                            if v is not None and np.isfinite(float(v))]
    reference_error = min(reference_candidates) if reference_candidates else None
    choose_dt = time.time() - start_time

    guide_idx = param.lambda_optimizer.choose_nearby_topk_idx(current_tip_local)
    if guide_idx is None:
        guide_idx = getattr(param.lambda_optimizer, 'last_global_idx', None)
    if guide_idx is not None:
        guide_idx = int(guide_idx)
        best_contact_point = param.lambda_optimizer.sample_point[guide_idx]
        normal = param.lambda_optimizer.normal[guide_idx]
        cached_x_plus, cached_force, cached_cost = _candidate_solution(
            param.lambda_optimizer, guide_idx)
        if cached_x_plus is not None:
            param.lambda_optimizer.last_best_x_plus = cached_x_plus
        if cached_force is not None:
            param.lambda_optimizer.last_best_force = cached_force
        if cached_cost is not None:
            param.lambda_optimizer.last_best_cost = cached_cost

    best_contact_track_world, best_contact_world, best_normal_world = _sphere_center_on_patch(
        curr_q[:3], r_obj_to_world, best_contact_point, normal,
        fingertip_radius, args.ideal_contact_surface_margin)
    p_arm_local, p_arm_normal_out, x_plus_opt, error, info = param.lambda_optimizer.resolve_executed_contact(
        current_tip_local, visible_point_idx,
        target_pose_eval, current_pose_eval, gravity,
        v_last=None,
        sphere_radius=max(1e-4, float(fingertip_radius) +
                          float(args.ideal_contact_surface_margin)))
    if actual_contact is not None:
        point_local = r_obj_to_world.T @ (np.asarray(actual_contact['point_world']) - curr_q[:3])
        inward_local = -r_obj_to_world.T @ np.asarray(actual_contact['normal_world'])
        actual_idx = actual_contact_idx if actual_contact_idx is not None else _nearest_sample_idx(
            param.lambda_optimizer, point_local)
        if staged_object_cost:
            from planning.measured_contact import measured_patch_index
            actual_idx = measured_patch_index(
                param.lambda_optimizer, point_local, -inward_local, actual_idx)
            actual_contact_idx = actual_idx
        if post_flip_pose and int(actual_idx) in visible_point_idx and int(actual_idx) not in blocked:
            previous_idx = getattr(param.lambda_optimizer, 'last_executed_idx', None)
            if (previous_idx is not None and int(previous_idx) in visible_point_idx
                    and int(previous_idx) not in blocked):
                try:
                    if float(param.lambda_optimizer.sample_geodesic[
                            int(previous_idx), int(actual_idx)]) <= float(
                                param.lambda_optimizer.contact_switch_radius):
                        actual_idx = int(previous_idx)
                except (AttributeError, IndexError, TypeError, ValueError):
                    pass
        # Keep the execution state aligned with the measured patch.  Without
        # this, the next cycle locks one sample but the current cycle solves
        # at another, making the marker and press point disagree.
        param.lambda_optimizer.last_executed_idx = int(actual_idx)
        p_arm_local, p_arm_normal_out, x_plus_opt, error, info = param.lambda_optimizer.optimize_control_input(
            target_pose_eval, current_pose_eval, gravity, p_arm=point_local, sample_idx=actual_idx,
            contact_point_local=point_local, contact_normal_inward=inward_local)
        # Ranking caches contain normalized pose cost, whereas this separate
        # NLP solve returns raw pose + force regularization. Compare like units.
        if not info.get('solver_failed', False):
            info['objective_cost'] = float(error)
            error = param.lambda_optimizer.normalize_pose_cost(
                param.lambda_optimizer._pose_cost(np.asarray(x_plus_opt).reshape(7), target_pose_eval))
            if error is None:
                error = float('inf')
            param.lambda_optimizer.last_executed_cost = error
    # Keep baseline RS target selection; MPC handles the complete direct
    # trajectory. There is no geometric via/transition predictor in A3.
    transition = None
    p_arm_surface_world = r_obj_to_world @ p_arm_local + curr_q[:3]
    p_arm_inward_local = -np.asarray(p_arm_normal_out, dtype=np.float64)
    p_arm_inward_world = r_obj_to_world @ p_arm_inward_local
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
    # A guide may be a near-quality runner-up or a measured locked patch.
    # Arrival/execution must use the sample that supplied the target and
    # cached prediction, rather than the optimizer's global winner.
    best_idx = guide_idx
    if best_idx is None:
        best_idx = getattr(param.lambda_optimizer, 'last_selected_idx', None)
    occupied_idx = _nearest_sample_idx(param.lambda_optimizer, current_tip_local)
    arrived_at_best = _arrived_at_best_contact(
        curr_q[7:10], best_contact_world, best_contact_track_world,
        occupied_idx, best_idx, param.lambda_optimizer)
    dist_best_now = min(
        float(np.linalg.norm(curr_q[7:10] - best_contact_world)),
        float(np.linalg.norm(curr_q[7:10] - best_contact_track_world)))
    if arrived_at_best:
        arrived_hold = True
        if arrived_dest_idx is None and best_idx is not None:
            arrived_dest_idx = int(best_idx)
    elif arrived_hold and dist_best_now > 0.035:
        arrived_hold = False
        arrived_dest_idx = None
    arrived_at_best = bool(arrived_at_best or arrived_hold)
    if (_on_opposite_sides(curr_q[7:10], curr_q[:3], best_contact_world) or
            _floor_slide_away_from_patch(
                curr_q[7:10], best_contact_world, ground=floor_ground)):
        arrived_at_best = False
        arrived_hold = False
        arrived_dest_idx = None
    if arrived_at_best and best_idx is not None and actual_contact is None:
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
        param.lambda_optimizer, p_arm_idx, best_idx, radius=0.03)
    if arrived_at_best:
        same_patch = True
    dist_arm = float(np.linalg.norm(curr_q[7:10] - p_arm_world))
    near_arm = dist_arm <= float(param.lambda_optimizer.contact_switch_radius)
    is_best_sample = (best_idx is not None and p_arm_idx is not None and
                      int(best_idx) == int(p_arm_idx))
    if arrived_at_best:
        is_best_sample = True
    value_tracker.reset_arm(p_arm_idx)
    if not staged_object_cost or actual_contact is not None:
        model_cost_conf.note_sample(
            param.lambda_optimizer, p_arm_idx if staged_object_cost
            else (best_idx if best_idx is not None else p_arm_idx))
    model_tightness = model_cost_conf.tightness()
    value_info = value_tracker.update_values(
        reference_error, error, solver_ok=solver_ok,
        candidate_costs=candidate_costs, same_patch=same_patch,
        near_arm=near_arm, is_best_sample=is_best_sample,
        confidence=param.lambda_optimizer.contact_switch_confidence,
        stagnant_steps=getattr(param.lambda_optimizer, '_dwell_steps', 0),
        min_error=min_error, max_error=max_error, tightness=model_tightness)
    hold_occupied = _should_hold_occupied_contact(
        param.lambda_optimizer, curr_q[7:10], curr_q[:3],
        best_contact_world, occupied_idx, best_idx=best_idx)
    # Use the prediction at the measured geometry, not the nearest mesh
    # sample's cached gain. Arrival within 3 cm can describe different normals
    # and lever arms; it must not turn an unhelpful contact into the winner.
    measured_gain = None
    switch_observed_contact = False
    if staged_object_cost and actual_contact is not None:
        if solver_ok:
            measured_gain = (
                param.lambda_optimizer._pose_cost(current_pose_eval, target_pose_eval)
                - param.lambda_optimizer._pose_cost(np.asarray(x_plus_opt).reshape(7), target_pose_eval))
        # A sampled one-step wrench assumes its patch has already been reached.
        # Comparing it with the measured manifold is a diagnostic, not a reason
        # to discard MPC's horizon and withdraw from a supporting contact.
        # Ordinary relocation uses the ranked target and the existing via;
        # an explicitly supplied transition predictor can still request travel.
        switch_observed_contact = bool(transition is not None and transition['switch'])
        if switch_observed_contact:
            hold_occupied = False
            value_info['accept_p_arm'] = value_info['cost_ok'] = False
            value_tracker._holding_p_arm = False
        elif near_arm or transition is not None:
            # Sparse ranking labels do not identify the measured manifold.
            # Retain a made contact while its actual progress still earns
            # confidence; the MPC objective decides how to load that patch.
            useful = (param.lambda_optimizer.contact_switch_confidence > .05
                      and int(actual_contact_idx) not in blocked)
            value_info['accept_p_arm'] = value_info['cost_ok'] = useful
            value_tracker._holding_p_arm = useful
    value_info['measured_contact_gain'] = measured_gain
    value_info['switch_observed_contact'] = switch_observed_contact
    if transition is not None:
        value_info.update(contact_transition_stay_cost=transition['stay_cost'],
            contact_transition_switch_cost=transition['switch_cost'],
            contact_transition_steps=transition['steps'],
            contact_transition_travel_steps=transition['travel_steps'])
    if (not value_info['accept_p_arm'] or switch_observed_contact) and not hold_occupied:
        p_arm_world = best_contact_track_world
        p_arm_track_world = best_contact_track_world
        p_arm_surface_world = best_contact_world
        x_plus_opt = cached_x_plus if actual_contact is None else None
        error = float(cached_cost) if cached_cost is not None else float(min_error)
        info = {
            'control_input': cached_force if cached_force is not None else np.zeros(3),
            'solver_failed': cached_x_plus is None,
        }
        exec_quality = 1.0
    else:
        exec_quality = 1.0
    verify_now, q_dist = value_tracker.update_verify(
        dist_exec=float(np.linalg.norm(curr_q[7:10] - p_arm_world)),
        tightness=model_tightness)
    value_info.update(verify=verify_now, q_dist=q_dist,
        using_observed_contact=actual_contact is not None,
        actual_contact_idx=actual_contact_idx, occupied_idx=occupied_idx,
        contact_patch_locked=bool(param.lambda_optimizer.lock_contact_patch),
        arrived_at_best=bool(arrived_at_best), exec_quality=exec_quality)
    return _direct_policy(p_arm_world, p_arm_surface_world,
        best_contact_track_world, best_contact_world, best_idx,
        value_info=value_info, x_plus_opt=x_plus_opt, error=error, info=info,
        min_error=min_error, max_error=max_error, cached_cost=cached_cost,
        cached_x_plus=cached_x_plus, cached_force=cached_force,
        choose_dt=choose_dt)
