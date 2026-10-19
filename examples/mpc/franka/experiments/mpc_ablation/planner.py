"""Current MPC request lifecycle, with isolated ablation hooks.

The dynamics, solver and executor constraints are shared with the baseline.
"""
import numpy as np
from planning.mpc_explicit import MPCExplicit, planner_init_payload


def _build_mpc_planner_runtime(init, param=None):
    if init['args'].get('ablation_mode') == 'a6':
        from .adamw import build_runtime
        return build_runtime(init, param)
    import argparse
    from planning.acados_env import ensure_acados_env
    ensure_acados_env()
    from examples.mpc.franka.experiments.mpc_ablation.params import ExplicitMPCParams
    from examples.mpc.fingertips.test.test_0902 import (
        ContactValueTracker,
        ModelCostConfidence,
        SmoothedApproachVia,
    )
    from examples.mpc.franka.experiments.mpc_ablation.mpc import (
        adapt_param_for_cartesian_solver, _configure_rollout_param,
    )

    args = argparse.Namespace(**init["args"])
    from .config import configure_ablation
    configure_ablation(args)
    from planning.runtime_compat import resolve_solver_backend
    args.solver_backend = getattr(args, "solver_backend", "auto")
    args.solver = resolve_solver_backend(args.solver_backend)
    args.rollout = True
    trial_count = int(init["trial_count"])
    if param is None:
        param = ExplicitMPCParams(
            args,
            rand_seed=trial_count,
            target_type=getattr(args, "target_type", "ground-rotation"),
            mpc_model="explicit",
        )
    param = adapt_param_for_cartesian_solver(param, args)
    param = _configure_rollout_param(param, args)
    for key in (
        "target_p_", "target_q_", "mesh_path_", "table_height",
        "lambda_obj_mass_", "gravity_", "attract_coef", "contact_coef",
    ):
        if key in init:
            setattr(param, key, init[key])
    from planning.object_pose_cost import target_up_axis
    # Payload may override the generated target; derive the axis afterwards.
    param.object_up_axis_ = target_up_axis(param.target_q_)
    from .stages import make_object_cost_stages
    stages = make_object_cost_stages(args, getattr(param, 'object_up_axis_', (0., 0., 1.)))
    if stages.enabled:
        param.object_cost_config_ = stages.config()
    param.mpc_carry_velocity_ = bool(getattr(args, 'mpc_carry_velocity', True))
    param.mpc_tip_effective_mass_ = float(getattr(args, 'mpc_tip_effective_mass', 2.5))
    param.mpc_tip_damping_ = float(getattr(args, 'mpc_tip_damping', 40.))
    param.mpc_tool_pose_ = getattr(args,'tool_orientation_control','free') == 'mpc'
    if param.mpc_tool_pose_ and (getattr(args,'physics_backend','isaac') != 'isaac' or param.online_scm_):
        raise ValueError('Tool-pose MPC currently requires Isaac with online SCM disabled')
    if param.ablation_mode in ('a4', 'a5'):
        from .cso import install_cso_ablation
        install_cso_ablation(param, args)
    mpc = MPCExplicit(param)
    if mpc.tool_pose_control:
        import copy
        position_param = copy.copy(param); position_param.mpc_tool_pose_ = False
        mpc.position_mpc = MPCExplicit(position_param)
        mpc.position_mpc.source_param_ = param
    mpc_step = max(1e-4, float(getattr(args, "mpc_step_limit", 0.005)))
    via_step = getattr(args, "via_max_step", None)
    via_step = mpc_step if via_step is None else max(1e-4, float(via_step))
    via_lead = max(1e-4, float(getattr(args, "via_max_lead", via_step)))
    from planning.franka_contact_motion import FrankaApproachVia, SurfaceNormalApproachVia
    approach_class = (SurfaceNormalApproachVia if getattr(param, 'collision_mesh_paths_', [])
                      else FrankaApproachVia)
    approach_options = (dict(floor_z=float(param.table_height)+float(args.table_clearance))
                        if approach_class is SurfaceNormalApproachVia else {})
    collision_parts = getattr(param.lambda_optimizer.pp, 'collision_parts', [])
    if param.ablation_mode == 'a3' and getattr(args, 'obj', '') == 'mug':
        # The travel path still avoids the physical mug walls. Lambda's box
        # must not replace the execution geometry used by this baseline hook.
        import trimesh
        collision_parts = [trimesh.load_mesh(path) for path in getattr(param, 'collision_mesh_paths_', [])]
    if getattr(args, 'obj', '') == 'mug' and collision_parts:
        from planning.compound_contact_path import CompoundApproachVia
        approach_class = CompoundApproachVia
        approach_options = dict(parts=collision_parts,
            floor_z=float(param.table_height)+float(args.table_clearance))
    trackers = {
        "value_tracker": ContactValueTracker(
            tau=float(args.value_tau),
            rel_scale=float(args.value_rel_scale),
            rho=float(args.value_rho),
            alpha=float(args.value_alpha),
            beta=float(args.verify_beta),
            window_size=int(getattr(args, "verify_window_size", 20)),
            confirm_steps=int(getattr(args, "verify_enter_steps", 5)),
            min_hold_steps=int(getattr(args, "verify_hold_steps", 30)),
            release_steps=int(getattr(args, "verify_release_steps", 8)),
            accept_margin_ratio=0.05,
            accept_margin_abs=0.02,
            accept_enter_tightness=float(args.verify_accept_threshold),
            accept_release_tightness=float(args.verify_release_threshold),
        ),
        "model_cost_conf": ModelCostConfidence(
            threshold=float(getattr(args, "model_cost_error_threshold", 6.0)),
            eps=float(getattr(args, "model_cost_error_eps", 1e-6)),
            min_steps=int(getattr(args, "model_cost_error_min_steps", 3)),
        ),
        "approach_via": approach_class(
            rate=float(getattr(args, "via_smooth_rate", 0.05)),
            max_step=via_step,
            max_lead=via_lead,
            **approach_options,
        ),
        "arrived_hold": False,
        "arrived_dest_idx": None,
        "sol_guess": None,
        "last_verify_cost": None,
    }
    trackers['object_cost_stages'] = stages
    if param.contact_model == 'soft-scm' and (getattr(args, 'online_scm', False) or getattr(args, 'scm_record_data', False) or getattr(args, 'scm_parameters', '')):
        from models.scm_online import OnlineSCMIdentifier
        from pathlib import Path
        import datetime
        out = getattr(args, 'scm_learning_dir', '')
        if not out:
            out = 'examples/mpc/franka/scm/outputs/online_scm_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log = Path(out) / (str(args.obj)+'_trial_%03d'%trial_count) / 'learning.jsonl'
        trackers['scm_identifier'] = OnlineSCMIdentifier(mpc.model,
            min_samples=int(getattr(args,'scm_learning_min_samples',32)),
            update_every=int(getattr(args,'scm_learning_update_every',32)), log_path=log)
        identifier = trackers['scm_identifier']
        checkpoint = getattr(args,'scm_parameters','')
        if checkpoint:
            # Mesh identity comes from the object-specific output directory;
            # numeric convention/configuration compatibility is checked in load.
            if Path(checkpoint).parent.name.split('_trial_')[0] != str(args.obj):
                raise ValueError('SCM checkpoint belongs to a different object')
            identifier.load(checkpoint)
            mpc.model.set_scm_parameters(identifier.theta)
            param.lambda_optimizer.set_scm_parameters(identifier.theta)
        print('[online-scm] enabled=%s data=%s' % (bool(getattr(args,'online_scm',False)), log.parent))
    return args, param, mpc, trackers


def _supported_call_kwargs(fn, kwargs):
    """Drop kwargs the live callee does not accept.

    Isaac / tilted-push payloads always carry support_point.  An older
    ``compute_rollout_contact_via`` on disk raises TypeError on that name
    before ranking starts.  New signatures still receive the plane.
    """
    import inspect
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in params}


def _call_rollout_contact_via(fn, *args, **kwargs):
    return fn(*args, **_supported_call_kwargs(fn, kwargs))


def handle_mpc_request(args, param, mpc, trackers, msg):
    if param.ablation_mode == 'a6':
        from .adamw import handle_request
        return handle_request(args, param, mpc, trackers, msg)
    import time
    request_started = time.perf_counter()
    root_mpc = mpc
    from examples.mpc.fingertips.test.test_0902 import (
        _verify_is_chatter,
    )
    from .policy import compute_ablation_contact_policy as compute_rollout_contact_via
    from examples.mpc.franka.runtime.contact import gravity_wrench_object_frame
    from planning.MPPIWarp import (
        _apply_dwell_payload,
        _opt_snapshot,
        _pickle_safe,
        clamp_via_to_tip,
    )
    from scipy.spatial.transform import Rotation

    online_started = time.perf_counter()
    identifier = trackers.get('scm_identifier')
    if identifier is not None:
        update = identifier.observe(msg.get('scm_transitions', []), learn=bool(getattr(args,'online_scm',False)))
        if update is not None and update['accepted']:
            mpc.model.set_scm_parameters(identifier.theta)
            param.lambda_optimizer.set_scm_parameters(identifier.theta)
        if update is not None:
            print('[online-scm] samples=%d accepted=%s loss=%.4g->%.4g time=%.2fms scales=%s' %
                  (update['samples'], update['accepted'], update['validation_loss_before'],
                   update['validation_loss_after'], update['update_ms'], update['scales']))

    online_dt = time.perf_counter() - online_started
    from planning.object_pose_cost import reset_objective_history
    stages = trackers.get('object_cost_stages')
    changed = stages.update(msg) if stages is not None else False
    if changed:
        reset_objective_history(param.lambda_optimizer, mpc, trackers)
        print('[object-cost] transition:', stages.diagnostics(), flush=True)
    if stages is not None and stages.enabled:
        # CPO keeps the staged terminal objective. Pose-only CSO ranks with
        # ell_pose; the other CSO modes keep ell_cso.
        param.object_cost_stage_ = stages.stage
        param.object_cost_config_ = stages.config()
        q = np.asarray(msg['policy_q'])
        r = Rotation.from_quat([q[4], q[5], q[6], q[3]]).as_matrix()
        from planning.object_pose_cost import pose_only_config
        cso_config = stages.config(r)
        if str(getattr(param.lambda_optimizer, 'cso_guidance', 'cso')) == 'pose':
            cso_config = pose_only_config(
                stages.pos_weight, stages.ori_weight, stages.body_axis,
                r.T @ np.array([0., 0., 1.]))
        param.lambda_optimizer.set_object_cost_config(cso_config)
        from planning.contact_limits import contact_command_force_limit
        param.lambda_optimizer.set_contact_force_limit(float(getattr(param, 'contact_command_force_limit_', contact_command_force_limit(args, 600.))))
    elif stages is not None:
        param.object_cost_stage_ = stages.stage
    elif getattr(param, 'object_cost_config_', None) is not None:
        # Ramp pushing has a fixed support-relative objective, without a
        # world-up flip/pose stage machine. Lambda works in the observed body.
        q = np.asarray(msg['policy_q'])
        r = Rotation.from_quat(q[[4, 5, 6, 3]]).as_matrix()
        config = np.asarray(param.object_cost_config_, dtype=float).copy()
        config[6:9] = r.T @ config[6:9]
        param.lambda_optimizer.set_object_cost_config(config)

    direct = param.ablation_mode in ('a1', 'a2')
    if getattr(param, 'cso_ablation_', None) is not None:
        param.cso_ablation_.prepare_observation(msg)
    dwell = None if direct or msg.get("osc_upright_guard_active", False) else msg.get("dwell")
    if msg.get("osc_upright_guard_active", False):
        trackers["sol_guess"] = None
    if (dwell is not None and not changed and
            (stages is None or not stages.enabled or dwell.get('object_cost_stage') == stages.stage)):
        dwell = dict(dwell)
        dwell["model_cost_conf"] = trackers["model_cost_conf"]
        _apply_dwell_payload(param, args, dwell)

    curr_q = np.asarray(msg["policy_q"], dtype=np.float32)
    max_ncon = int(getattr(param, "max_ncon_", 10))
    nv = int(getattr(param, "n_qvel_", 9))
    phi_vec = (
        np.asarray(msg["phi_vec"])
        if msg.get("phi_vec") is not None
        else np.ones((max_ncon * 4,), dtype=np.float64)
    )
    jac_mat = (
        np.asarray(msg["jac_mat"])
        if msg.get("jac_mat") is not None
        else np.zeros((max_ncon * 4, nv), dtype=np.float64)
    )
    if msg.get("jac_mat_env") is not None:
        jac_mat_env = np.asarray(msg["jac_mat_env"])
    else:
        from examples.mpc.franka.runtime.contact import table_jac_mat_env
        jac_mat_env = table_jac_mat_env(
            curr_q[:3], curr_q[3:7], float(param.table_height),
            nv=nv, mu=float(getattr(param, "mu_object_", 0.5)), max_ncon=max_ncon,
        )
    table_ground = float(msg["table_ground"])
    floor_z = float(msg["floor_z"]) if msg.get("floor_z") is not None else float(param.table_height)
    support_point = msg.get("support_point")
    support_normal = msg.get("support_normal")
    if support_point is not None:
        support_point = np.asarray(support_point, dtype=np.float64).reshape(3)
    if support_normal is not None:
        support_normal = np.asarray(support_normal, dtype=np.float64).reshape(3)
    r_obj_to_world = Rotation.from_quat([curr_q[4], curr_q[5], curr_q[6], curr_q[3]]).as_matrix()
    ranking_mass = float(getattr(param, "lambda_obj_mass_", param.lambda_optimizer.m))
    gravity = gravity_wrench_object_frame(
        param.gravity_[:3], ranking_mass, r_obj_to_world, support_normal,
    )
    if getattr(param.lambda_optimizer, 'shared_response', None) is not None:
        param.lambda_optimizer.rank_object_velocity = np.r_[
            r_obj_to_world.T @ np.asarray(msg.get('object_linear_velocity', np.zeros(3))),
            r_obj_to_world.T @ np.asarray(msg.get('object_angular_velocity', np.zeros(3)))]
        param.lambda_optimizer.rank_tip_velocity = r_obj_to_world.T @ np.asarray(
            msg.get('fingertip_linear_velocity', np.zeros(3)))
    param.lambda_env_phi_ = msg.get('env_phi')
    # PhysX reports speculative patches before their signed gap reaches zero.
    # Keep that measured geometry: the dynamics must decide whether a bounded
    # command closes the gap, rather than switching on the binary impulse flag.
    measured_contact = msg.get('actual_contact')
    measured_gap = float(msg.get('contact_distance', float('inf')))
    if measured_contact is not None:
        measured_gap = float(measured_contact.get('dist', measured_gap))
    reachable_contact = bool(msg.get('if_contact', False) or (
        measured_contact is not None and np.isfinite(measured_gap)
        and measured_gap <= float(np.max(np.abs(param.mpc_u_ub_)))))
    param.lambda_actual_contact_ = (measured_contact if
        ((stages is not None and stages.enabled) or getattr(param, 'runtime_object_cost_', False))
        and reachable_contact else None)
    if getattr(mpc,'tool_pose_control',False) and not (
            stages is not None and stages.enabled and stages.stage == 'flip' and reachable_contact):
        mpc = mpc.position_mpc
    # Finite-difference fingertip motion is in world coordinates. Object
    # angular velocity is transformed to body coordinates by the model.
    previous = trackers.get('motion_observation')
    now = float(msg.get('sim_time', 0.))
    tip_velocity = np.zeros(3)
    if previous is not None and now > previous[0]:
        tip_velocity = (curr_q[7:10]-previous[1])/(now-previous[0])
    if msg.get('fingertip_linear_velocity') is not None:
        tip_velocity = np.asarray(msg['fingertip_linear_velocity'], dtype=float)
    trackers['motion_observation'] = (now, curr_q[7:10].copy())
    planner_state = (mpc.observation_state(curr_q, msg, tip_velocity)
        if hasattr(mpc, 'observation_state') else curr_q)
    predictor = getattr(param, 'contact_transition_predictor_', None)
    if predictor is not None and hasattr(predictor, 'set_observation'):
        predictor.set_observation(planner_state,param.lambda_optimizer)
    t0 = time.perf_counter()
    policy = _call_rollout_contact_via(
        compute_rollout_contact_via,
        param, args, curr_q, r_obj_to_world, gravity, jac_mat_env,
        0.01, trackers["value_tracker"], trackers["model_cost_conf"],
        trackers["approach_via"], trackers["arrived_hold"], trackers["arrived_dest_idx"],
        floor_ground=table_ground,
        floor_z=floor_z,
        support_point=support_point,
        support_normal=support_normal,
    )
    rank_dt = time.perf_counter() - t0
    trackers["arrived_hold"] = policy["arrived_hold"]
    trackers["arrived_dest_idx"] = policy["arrived_dest_idx"]
    # Travel bypasses MPC, so it does not need to overwrite the critic EMA.
    # Preserve its continuous value across lift/drop and impulse flag changes.
    verify_cost = float((policy.get("value_info") or {}).get("verify", policy["verify_cost"]))
    policy["verify_cost"] = verify_cost
    verify_chatter = _verify_is_chatter(trackers["last_verify_cost"], verify_cost)
    trackers["last_verify_cost"] = float(verify_cost)

    tip = np.asarray(curr_q[7:10], dtype=np.float64)
    max_lead = max(1e-4, float(getattr(args, "via_max_lead", getattr(args, "via_max_step", getattr(args, "mpc_step_limit", 0.005)))))
    value_info = policy.get("value_info") or {}
    contact_id = getattr(param.lambda_optimizer, 'last_executed_idx', None)
    context = (getattr(mpc,'tool_pose_control',False), contact_id, bool(reachable_contact),
               tuple(np.flatnonzero(np.linalg.norm(jac_mat.reshape(-1, 4, nv), axis=(1, 2)) > 1e-10)))
    previous_context = trackers.get('mpc_contact_context')
    if previous_context is not None and context != previous_context:
        trackers['sol_guess'] = None
        if hasattr(mpc, '_reset_acados_solver'):
            mpc._reset_acados_solver()
        trackers['mpc_contact_resets'] = trackers.get('mpc_contact_resets', 0) + 1
    trackers['mpc_contact_context'] = context
    if bool(value_info.get("path_blocked", False)) and not reachable_contact:
        trackers["sol_guess"] = None
    # A blocked orbit already walks via at max_step.  Clamping it back
    # to a 3 mm tip lead whenever the tip crosses the keep-out is what
    # made the rim hop.
    if not direct and not bool(value_info.get("path_blocked", False)):
        policy["mpc_virtual_point"] = clamp_via_to_tip(tip, policy["mpc_virtual_point"], max_lead)
        policy["mpc_contact_point"] = clamp_via_to_tip(tip, policy["mpc_contact_point"], max_lead)

    from planning.franka_contact_motion import bounded_step
    pose_stage = bool(stages is not None and stages.enabled and stages.stage == 'pose')
    flip_stage = bool(stages is not None and stages.enabled and stages.stage == 'flip')
    # OSC clips endpoint commands at the table in both stages. Predicting
    # only the finite-response fingertip state above the floor is insufficient:
    # a downward increment can still be replaced by an upward executor command.
    align_executor = bool(
        (pose_stage and getattr(args, 'pose_executor_constraints', False))
        or (flip_stage and getattr(args, 'flip_executor_constraints', True)))
    # Keep the developer Cartesian step (mpc_u_ub_, 5 mm). Dividing the
    # contact-force cap by K_task boxes every increment under 1 mm and the
    # arm crawls. The object cost and its force limit are unchanged.
    contact_bound = float(np.max(np.abs(getattr(param, "mpc_u_ub_", max_lead))))
    approach_action = (bool(value_info.get('switch_observed_contact', False))
        or (not reachable_contact and (bool(value_info.get('path_blocked', False))
            or (stages is not None and stages.enabled)
            or np.linalg.norm(tip - np.asarray(policy.get('p_arm_world', tip))) > .025)))
    if getattr(param, 'approach_requires_contact_', False) and not reachable_contact:
        # Without a reachable measured finger row the object objective cannot
        # drive an approach. Use the bounded via until contact geometry exists.
        approach_action = True
    if bool(getattr(param, 'cpo_direct_trajectory', False)):
        # Direct trajectory optimization solves the dynamics-constrained
        # problem. The geometric lift/place path is the phase potential
        # this variant removes.
        approach_action = False
    execution_mode = 'approach' if approach_action else 'contact'
    mpc.first_action_bounds = None
    trackers['mpc_slew_rebased'] = False
    if (getattr(mpc, 'velocity_state', False) and msg.get('executed_action') is not None
            and not approach_action):
        previous_command = np.clip(np.asarray(msg['executed_action'], dtype=float).reshape(3),
                                   param.mpc_u_lb_, param.mpc_u_ub_)
        force_cap = getattr(mpc, 'contact_force_limit', None)
        if force_cap is not None:
            force = np.linalg.norm(np.asarray(param.robot_stiff_) @ previous_command)
            if force > force_cap:
                # A free-space command can exceed the contact cap. Centering
                # the slew box on it can make its intersection with the force
                # ball empty. Rebase on a feasible command at contact entry.
                previous_command *= force_cap / force
                trackers['mpc_slew_rebased'] = True
        slew = float(getattr(param, 'isaac_action_slew_', .0025))/np.sqrt(3.)
        mpc.first_action_bounds = (np.maximum(param.mpc_u_lb_, previous_command-slew),
                                   np.minimum(param.mpc_u_ub_, previous_command+slew))
    command_lower = getattr(param, 'mpc_u_lb_', -contact_bound)
    command_upper = getattr(param, 'mpc_u_ub_', contact_bound)
    trackers['mpc_executor_edge'] = False
    if (align_executor
            and getattr(mpc, 'velocity_state', False) and not approach_action):
        from planning.execution_constraints import table_edge_command_bounds
        from examples.mpc.franka.experiments.mpc_ablation.mpc import (
            _table_edge_gap_and_normal, ISAAC_TABLE_EDGE_GUARD_DISTANCE)
        gap, outward = _table_edge_gap_and_normal(curr_q, param)
        guard_active = bool(msg.get('if_contact', False) or (
            not value_info.get('path_blocked', False)
            and np.linalg.norm(tip - policy['p_arm_world']) <= .015))
        command_lower, command_upper, edge_applied = table_edge_command_bounds(
            command_lower, command_upper, gap, outward, ISAAC_TABLE_EDGE_GUARD_DISTANCE, guard_active)
        trackers['mpc_executor_edge'] = edge_applied
    trackers['mpc_executor_floor'] = dict(applied=False, rebased=False, emergency=False)
    if (align_executor
            and getattr(mpc, 'velocity_state', False) and not approach_action):
        from planning.execution_constraints import floor_command_bounds
        force_cap = getattr(mpc, 'contact_force_limit', None)
        radius = (None if force_cap is None else force_cap / np.linalg.norm(param.robot_stiff_, 2))
        previous_command = msg.get('executed_action')
        bounds, floor_info = floor_command_bounds(tip[2], param.mpc_q_lb_[9],
            command_lower, command_upper, first_bounds=mpc.first_action_bounds,
            previous=previous_command,
            slew=float(getattr(param, 'isaac_action_slew_', .0025))/np.sqrt(3.), max_norm=radius)
        mpc.first_action_bounds = bounds
        trackers['mpc_executor_floor'] = floor_info
    if getattr(mpc,'tool_pose_control',False):
        angle = mpc.model.tool_rotation_step/np.sqrt(3.)
        angular_bounds = (np.full(3,-angle),np.full(3,angle))
        if mpc.first_action_bounds is not None:
            mpc.first_action_bounds = tuple(np.r_[b,a] for b,a in zip(mpc.first_action_bounds,angular_bounds))
        command_lower = np.r_[np.broadcast_to(command_lower,3),angular_bounds[0]]
        command_upper = np.r_[np.broadcast_to(command_upper,3),angular_bounds[1]]
    t1 = time.perf_counter()
    if approach_action:
        # The geometric approach replaces MPC's command. Solving that discarded
        # horizon can trigger a slow IPOPT fallback at contact-mode changes.
        action = bounded_step(tip, policy['mpc_virtual_point'], max_lead)
        sol = dict(sol_guess=None, cost_opt=None, solver_backend='approach')
    else:
        if (trackers["sol_guess"] is None and getattr(param, 'runtime_object_cost_', False)
                and param.lambda_actual_contact_ is not None and value_info.get('accept_p_arm', False)):
            # Start on the measured Lambda solution's active contact branch.
            # MPC still optimizes the same objective and all command constraints.
            inward = -r_obj_to_world.T @ np.asarray(param.lambda_actual_contact_['normal_world'], dtype=float)
            inward /= max(np.linalg.norm(inward), 1e-12)
            ref = np.array([0., 0., 1.]) if abs(inward[2]) < .9 else np.array([0., 1., 0.])
            tangent = np.cross(inward, ref); tangent /= np.linalg.norm(tangent)
            frame = np.column_stack([inward, tangent, np.cross(inward, tangent)])
            force = np.asarray(policy['info']['control_input'], dtype=float).reshape(3)
            command = np.linalg.solve(param.robot_stiff_, r_obj_to_world @ frame @ force)
            if getattr(mpc,'tool_pose_control',False):
                command = np.r_[command,np.zeros(3)]
            state = np.asarray(planner_state, dtype=float)
            states, rows = [state], []
            for _ in range(param.mpc_horizon_):
                state = np.asarray(mpc.stage_prediction_fn(state, command, phi_vec, jac_mat,
                    mpc.model.runtime_parameters(param.model_params), planner_state)).reshape(-1)
                states.append(state); rows.append(np.r_[command, state])
            trackers['sol_guess'] = dict(x0=np.concatenate(rows),
                u_traj=np.tile(command, (param.mpc_horizon_, 1)), x_traj=np.array(states))
        plan_kwargs = dict(
            target_p=param.target_p_, target_q=param.target_q_, curr_x=planner_state,
            phi_vec=phi_vec, jac_mat=jac_mat,
            verify_cost_param=(1.0 if direct else float(not value_info.get('path_blocked', False))
                if getattr(param, 'runtime_object_cost_', False) and reachable_contact else verify_cost),
            virtual_point=policy["mpc_virtual_point"],
            contact_point=policy["mpc_contact_point"],
            sol_guess=trackers["sol_guess"],
            **(dict(u_lb=command_lower, u_ub=command_upper)
               if trackers['mpc_executor_edge'] or getattr(mpc,'tool_pose_control',False) else {}),
        )
        # Lambda supplies initial guesses only. Every executed contact action
        # comes from optimization under the original objective and constraints.
        reseed = (getattr(args, 'mpc_contact_multistart', False)
            and getattr(args, 'physics_backend', 'isaac') == 'isaac'
            and flip_stage and reachable_contact and getattr(mpc, 'velocity_state', False)
            and not getattr(mpc,'tool_pose_control',False)
            and (trackers['sol_guess'] is None or context != previous_context
                 or trackers.get('mpc_contact_seed_age', 0) >= 25))
        seeds = []
        if reseed and param.lambda_actual_contact_ is not None:
            from planning.franka_mpc_seeds import contact_loading_seeds, rollout_guess
            actual = param.lambda_actual_contact_
            outward = np.asarray(actual['normal_world'], dtype=float)
            inward = -r_obj_to_world.T@outward
            inward /= max(np.linalg.norm(inward), 1e-12)
            ref = np.array([0., 0., 1.]) if abs(inward[2]) < .9 else np.array([0., 1., 0.])
            tangent = np.cross(inward, ref); tangent /= np.linalg.norm(tangent)
            frame = np.column_stack([inward, tangent, np.cross(inward, tangent)])
            force = r_obj_to_world@frame@np.asarray(param.lambda_optimizer.last_executed_force).reshape(3)
            commands = contact_loading_seeds(force, outward, param.robot_stiff_, param.mpc_horizon_)
            seeds = [rollout_guess(mpc, planner_state, u, phi_vec, jac_mat) for u in commands]
        if seeds:
            from planning.franka_mpc_seeds import best_contact_plan
            sol = best_contact_plan(mpc, plan_kwargs, seeds)
            trackers['mpc_contact_seed_age'] = 0
        else:
            sol = mpc.plan_once(**plan_kwargs)
            trackers['mpc_contact_seed_age'] = trackers.get('mpc_contact_seed_age', 0)+1
        policy['mpc_contact_seed_costs'] = sol.get('contact_seed_costs')
        policy['mpc_contact_seed_selected'] = sol.get('contact_seed_selected', 0)
        full_action = np.asarray(sol['action'], dtype=float).reshape(-1)
        action = full_action[:3]
        if getattr(mpc,'tool_pose_control',False):
            policy['tool_rotation_action'] = full_action[3:6].copy()
    plan_dt = time.perf_counter() - t1
    trackers["sol_guess"] = sol["sol_guess"]
    if execution_mode != 'contact' or msg.get('osc_upright_guard_active', False):
        trackers['sol_guess'] = None
        policy['x_plus_opt'] = None
        # Candidate force predictions no longer describe the executed action.
        # Do not feed their gains into confidence/dwell on the next frame.
        policy['execution_override'] = True
    if getattr(param, 'runtime_object_cost_', False):
        # Confidence must assess the action MPC actually sent, rather than
        # Lambda's separately optimized wrench at a hypothetical mesh sample.
        predicted_command = (np.r_[action,policy.get('tool_rotation_action',np.zeros(3))]
                             if getattr(mpc,'tool_pose_control',False) else action)
        predicted = np.asarray(mpc.stage_prediction_fn(planner_state, predicted_command, phi_vec, jac_mat,
            mpc.model.runtime_parameters(param.model_params), planner_state)).reshape(-1)
        policy['execution_object_pose_world'] = predicted[:7]
        if stages is not None and stages.enabled:
            from planning.object_pose_cost import object_cost_np
            target = np.r_[param.target_p_, param.target_q_]
            policy['pred_dcost'] = object_cost_np(curr_q[:7], target, stages.config()) - object_cost_np(predicted[:7], target, stages.config())
            policy['pred_dcost_prev'] = None if dwell is None else dwell.get('pred_reduction')
            policy['act_dcost_prev'] = (None if dwell is None else float(dwell['c_now_cost'])-float(dwell['c_after']))
            policy['object_cost_now'] = object_cost_np(curr_q[:7], target, stages.config())
    policy['preserve_upright'] = False
    policy['release_pressure'] = False
    policy['execution_mode'] = execution_mode
    if getattr(args, 'rollout_trace_jsonl', '') and getattr(param, 'runtime_object_cost_', False):
        from planning.contact_prediction_diagnostics import contact_prediction_diagnostics
        policy['contact_prediction_diagnostics'] = contact_prediction_diagnostics(
            param, mpc, curr_q, policy, sol)
    policy['execution_path_blocked'] = bool(value_info.get('path_blocked', False) and approach_action)
    policy["choose_dt"] = rank_dt
    if stages is not None:
        policy['object_cost_stage'] = stages.stage
        policy['object_cost_config'] = stages.config() if stages.enabled else None
        policy['object_progress_scale'] = stages.progress_scale() if stages.enabled else None
        policy['object_feedback_scale'] = max(.05, float(getattr(stages, 'ori_weight', 20.))*.025) if stages.enabled else None
    elif getattr(param, 'object_cost_config_', None) is not None:
        policy['object_cost_config'] = np.asarray(param.object_cost_config_).copy()
    mpc_instances = [root_mpc]
    if getattr(root_mpc, "position_mpc", None) is not None:
        mpc_instances.append(root_mpc.position_mpc)
    def mpc_count(name):
        return sum(int(getattr(instance, name, 0)) for instance in mpc_instances)
    return {
        "object_cost_diagnostics": dict({} if stages is None else stages.diagnostics(),
            pressure_releases=0, upright_action_rejections=0),
        "seq": None if msg.get("seq") is None else int(msg["seq"]),
        "action": np.asarray(action, dtype=np.float64).reshape(3),
        "sol_guess": sol["sol_guess"],
        "cost_opt": sol.get("cost_opt"),
        "policy": _pickle_safe(policy),
        "verify_cost": float(verify_cost),
        "verify_chatter": bool(verify_chatter),
        "if_contact": bool(msg.get("if_contact", False)),
        "opt_snapshot": _opt_snapshot(param.lambda_optimizer),
        "scm_identification": None if identifier is None else identifier.diagnostics(),
        "model_tightness": float(trackers["model_cost_conf"].tightness()),
        "model_accum": float(trackers["model_cost_conf"].accum),
        "rank_dt": float(rank_dt),
        "online_dt": float(online_dt),
        "total_dt": float(time.perf_counter() - request_started),
        "mpc_skipped": bool(approach_action),
        "mpc_executor_floor": dict(trackers['mpc_executor_floor']),
        "mpc_executor_edge": bool(trackers['mpc_executor_edge']),
        "mpc_fallbacks": mpc_count("acados_fallback_count"),
        "mpc_contact_resets": int(trackers.get('mpc_contact_resets', 0)),
        "mpc_slew_rebased": bool(trackers.get('mpc_slew_rebased', False)),
        "mpc_floor_recovery_m": max(0., float(param.mpc_q_lb_[9]) - float(np.min(sol['rollout_q'][:, 9])))
            if getattr(mpc, 'recoverable_floor', False) and sol.get('rollout_q') is not None else 0.,
        "mpc_last_error": str(getattr(mpc, "last_acados_error", "") or ""),
        "mpc_rti_iterations": 0 if approach_action else int(getattr(mpc, "last_acados_rti_iterations", 0)),
        "plan_dt": float(plan_dt),
        "mppi_dt": float(plan_dt),
        "solver_backend": str(sol.get("solver_backend", getattr(mpc, "planner_solver_", "ipopt"))),
        "lambda_failures": int(getattr(param.lambda_optimizer, "acados_failure_count", 0)),
        "mpc_failures": mpc_count("acados_failure_count"),
        "lambda_solves": int(getattr(param.lambda_optimizer, "acados_solve_count", 0)),
        "mpc_solves": mpc_count("acados_solve_count"),
        "mpc_init_error": str(getattr(mpc, "_acados_init_error", "") or ""),
    }

