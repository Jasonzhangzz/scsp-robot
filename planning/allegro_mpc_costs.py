"""Allegro pad-to-object alignment and contact-force MPC objectives.

Tip kinematics are linearized at the measured hand every command. Object
anchors move with the predicted object pose, as in the bigrasp EE cost.
"""

import casadi as cs
import numpy as np

from models.contact_geometry import rotation_cs


def build_allegro_contact_costs(param):
    nq, n = param.n_cmd_, param.num_contacts
    x = cs.SX.sym("allegro_state", param.n_qpos_)
    u = cs.SX.sym("allegro_action", nq)
    # Goal, weak IK attraction, loaded, XY bounds, measured hand/tips,
    # stacked tip Jacobian, object-frame pad targets and attraction gate.
    head_size = 13+2*nq+6*n+3*n*nq
    physics = getattr(param, "online_model", None)
    force_cost = physics is not None and (param.force_tracking_weight > 0 or param.pressure_sync_weight > 0)
    linear_force = force_cost and bool(getattr(param, 'linearize_force_cost', False))
    # Geometry also gates IK attraction along the predicted horizon, even
    # when force tracking is disabled.
    extra_size = 0 if physics is None else 3*n+4*param.max_ncon_*(1+param.n_qvel_)+physics.model_parameter_dim
    param.cost_force_linearization_offset = head_size+extra_size if linear_force else None
    if linear_force:
        extra_size += param.n_qpos_+nq+3*n*(1+param.n_qpos_+nq)
    feedback = linear_force and bool(getattr(param, 'force_feedback', False))
    param.cost_force_feedback_offset = head_size+extra_size if feedback else None
    if feedback:
        extra_size += 7*n  # per-pad gain, raw baseline and measured world forces
    cp = cs.SX.sym("allegro_contact_objective", head_size+extra_size)
    goal, ik_q = cp[:7], cp[7:7+nq]
    loaded = cp[7+nq]
    lower, upper = cp[8+nq:10+nq], cp[10+nq:12+nq]
    offset = 12+nq
    hand_reference = cp[offset:offset+nq]
    offset += nq
    tips = cp[offset:offset+3*n]
    offset += 3*n
    tip_jacobian = cs.reshape(cp[offset:offset+3*n*nq], 3*n, nq)
    offset += 3*n*nq
    anchors = cs.reshape(cp[offset:offset+3*n], 3, n)
    attraction = cp[offset+3*n]
    predicted_tips = tips + tip_jacobian @ (x[7:7+nq]-hand_reference)
    target_tips = cs.repmat(x[:3], 1, n) + rotation_cs(x[3:7]) @ anchors
    pad_error = predicted_tips-cs.reshape(target_tips, 3*n, 1)
    if physics is not None and getattr(param, 'pressure_control', False):
        gap_offset = head_size+3*n
        row_offset = gap_offset+4*param.max_ncon_
        contact_rows = cs.reshape(cp[row_offset:row_offset+4*param.max_ncon_*param.n_qvel_],
            4*param.max_ncon_, param.n_qvel_)
        errors = []
        for i in range(n):
            error = pad_error[3*i:3*i+3]
            normal = cs.sum1(contact_rows[4*i:4*i+4, :3]).T/4
            # After contact, preserve the object-frame region against slip.
            # Normal indentation is regulated from measured pressure, so
            # this cost must not keep demanding the original IK depth.
            errors.append(cs.if_else(cp[gap_offset+4*i] <= 0,
                error-normal*cs.dot(normal, error), error))
        pad_error = cs.vertcat(*errors)
    contact_residual = np.sqrt(param.tip_weight)*pad_error
    terms = [contact_residual,
        np.sqrt(param.lateral_weight)*(x[:2]-goal[:2]),
        np.sqrt(param.object_weight)*(x[2]-goal[2]),
        np.sqrt(param.quat_weight/8)*cs.reshape(
            rotation_cs(goal[3:7]).T@rotation_cs(x[3:7])-cs.DM.eye(3), 9, 1),
        np.sqrt(param.horizontal_recovery_weight)*cs.fmax(lower-x[:2], 0),
        np.sqrt(param.horizontal_recovery_weight)*cs.fmax(x[:2]-upper, 0)]
    transport_weight = float(getattr(param, 'contact_transport_weight', 0.))
    if transport_weight > 0:
        transport_position = goal[:3]
        distance_limit = getattr(param, 'contact_transport_distance', None)
        if physics is not None and distance_limit is not None:
            # A 20 cm object goal cannot be attained by the hand in a 100 ms
            # horizon. Advance pad references at the configured lift rate,
            # while retaining the full object position/orientation objective.
            reference_offset = head_size+3*n+4*param.max_ncon_*(1+param.n_qvel_)+1
            reference_position = cp[reference_offset:reference_offset+3]
            delta = goal[:3]-reference_position
            fraction = cs.fmin(1., float(distance_limit)/cs.sqrt(cs.sumsqr(delta)+1e-20))
            transport_position = reference_position+fraction*delta
        goal_tips = cs.repmat(transport_position, 1, n) + rotation_cs(goal[3:7]) @ anchors
        # Once a physical grasp is confirmed, carry its contact regions to
        # the object goal in task space. This avoids a stationary hand when
        # reduced contact dynamics underestimate tangential load transfer.
        terms.append(np.sqrt(transport_weight)*loaded*(predicted_tips-cs.reshape(goal_tips, 3*n, 1)))
    # IK only attracts the approach. Its runtime gate reaches zero before
    # contact, and it never appears in the terminal grasp objective.
    ik_gate = cs.sqrt(cs.fmax(attraction, 0))
    if physics is not None:
        reference = cs.reshape(cp[head_size:head_size+3*n], 3, n)
        offset = head_size+3*n
        gaps = cp[offset:offset+4*param.max_ncon_]
        offset += 4*param.max_ncon_
        rows = cs.reshape(cp[offset:offset+4*param.max_ncon_*param.n_qvel_],
            4*param.max_ncon_, param.n_qvel_)
        offset += 4*param.max_ncon_*param.n_qvel_
        predicted_gaps = physics.contact_gap_fn(x, gaps, rows, cp[offset:offset+physics.model_parameter_dim])
        predicted_gate = cs.fmin(cs.fmax((cs.mmin(predicted_gaps[:4*n])-.002)/.018, 0), 1)
        ik_gate = cs.fmin(ik_gate, predicted_gate)
        collision_weight = float(getattr(param, 'self_collision_weight', 0.))
        if collision_weight > 0:
            for i in range(n+4, param.max_ncon_):
                block = rows[4*i:4*i+4, :]
                robot_only = cs.logic_and(cs.sumsqr(block[:, :6]) == 0, cs.sumsqr(block[:, 6:]) > 0)
                terms.append(np.sqrt(collision_weight)*cs.if_else(robot_only,
                    cs.fmax(param.self_collision_clearance-predicted_gaps[4*i], 0), 0))
                # The position-servo target can lead measured joints by
                # several commands. Protect its geometric posture too;
                # otherwise soft predicted recovery masks a colliding target.
                commanded_gap = gaps[4*i] + (cs.sum1(block[:, 6:])/4)@(x[-nq:]-hand_reference)
                terms.append(np.sqrt(collision_weight)*cs.if_else(robot_only,
                    cs.fmax(param.self_collision_clearance-commanded_gap, 0), 0))
    state_residual = cs.vertcat(*terms)
    stage_terms = [state_residual, cs.DM(np.sqrt(param.action_weights))*u,
        ik_gate*(1-loaded)*cs.DM(np.sqrt(param.posture_weights))*(x[7:7+nq]-ik_q)]
    if force_cost:
        if linear_force:
            offset += physics.model_parameter_dim
            force_state = cp[offset:offset+param.n_qpos_]
            offset += param.n_qpos_
            force_action = cp[offset:offset+nq]
            offset += nq
            force_value = cp[offset:offset+3*n]
            offset += 3*n
            force_jx = cs.reshape(cp[offset:offset+3*n*param.n_qpos_], 3*n, param.n_qpos_)
            offset += 3*n*param.n_qpos_
            force_ju = cs.reshape(cp[offset:offset+3*n*nq], 3*n, nq)
            # RTI's force sensitivity is refreshed at every measured command.
            # Keep the full SCM map in dynamics, outside the cost codegen.
            force = force_value + force_jx@(x-force_state) + force_ju@(u-force_action)
        else:
            force = physics.contact_force_fn(x, u, gaps, rows, cp[offset:])
        force_scale = float(getattr(param, 'force_residual_scale', 1.))
        force_error = force-cs.reshape(rotation_cs(x[3:7])@reference, 3*n, 1)
        if getattr(param, 'force_cost_as_wrench', False):
            from models.allegro_grasp_response import object_point_jacobian_cs
            # Internal preload need not match an arbitrary per-pad split.
            # Track the resulting lifting force AND moment, while the normal
            # pressure loop retains compression. Necessary tangential forces
            # are not penalized merely for being large at individual pads.
            wrench_error = object_point_jacobian_cs(rows, n).T@force_error
            force_error = cs.vertcat(wrench_error[:3], wrench_error[3:]/.05)
        stage_terms.append(force_scale*np.sqrt(param.force_tracking_weight)*force_error)
        normal = cs.vertcat(*[cs.dot(cs.sum1(rows[4*i:4*i+4, :3]).T/4,
            force[3*i:3*i+3]) for i in range(n)])
        stage_terms.append(force_scale*np.sqrt(2*param.pressure_sync_weight/max(n-1, 1))*(1-loaded)*
            (normal-cs.sum1(normal)/n))
    residual = cs.vertcat(*stage_terms)
    terminal = np.sqrt(param.terminal_weight)*state_residual
    param.cost_param_head_size = head_size
    param.cost_uses_contact_geometry = physics is not None
    param.cost_uses_force_reference = force_cost
    param.cost_linearizes_force = linear_force
    param.residual_fn = cs.Function("allegro_contact_residual", [x, u, cp], [residual])
    param.terminal_residual_fn = cs.Function("allegro_contact_terminal_residual", [x, cp], [terminal])
    tag = getattr(param, "cost_tag", "test")
    return (cs.Function("allegro_contact_cost_v2_"+tag, [x, u, cp], [cs.sumsqr(residual)]),
            cs.Function("allegro_contact_terminal_v2_"+tag, [x, cp], [cs.sumsqr(terminal)]))


def contact_cost_parameters_at_state(parameters, state, action, physics, phi, jac,
                                     model_parameters, linearization_offset, feedback_offset=None):
    """Evaluate the RTI objective with exact forces at a rollout state.

    The QP keeps its cheap force linearization. Backtracking and fallback
    selection must instead score the nonlinear SCM forces, especially when
    a pad separates or a friction constraint changes mode. Re-centering the
    affine residual leaves every other objective term unchanged.
    """
    if linearization_offset is None:
        return parameters
    values = np.asarray(parameters, float).copy()
    state, action = np.asarray(state).reshape(-1), np.asarray(action).reshape(-1)
    offset = int(linearization_offset)
    values[offset:offset+state.size] = state
    offset += state.size
    values[offset:offset+action.size] = action
    offset += action.size
    force = np.asarray(physics.contact_force_fn(state, action, phi, jac, model_parameters)).reshape(-1)
    if feedback_offset is not None:
        start = int(feedback_offset)
        n = force.size//3
        gain = values[start:start+n]
        baseline = values[start+n:start+4*n].reshape(n, 3)
        measured = values[start+4*n:start+7*n].reshape(n, 3)
        normal = np.asarray(jac).reshape(-1, 4, np.asarray(jac).shape[1])[:n, :, :3].mean(axis=1)
        gaps = np.asarray(physics.contact_gap_fn(state, phi, jac, model_parameters)).reshape(-1, 4)[:n].mean(axis=1)
        force = calibrated_pad_forces(force.reshape(n, 3), baseline, measured, gain,
            normal, gaps, getattr(physics.param_, 'mu_object_', 0.)).reshape(-1)
    values[offset:offset+force.size] = force
    return values


def calibrated_pad_forces(raw, baseline, measured, gain, normals, gaps, friction):
    """Measured-output correction, with unilateral Coulomb feasibility.

    This corrects the force observable used in tracking, without replacing
    the softSCM motion predictor. Each control cycle refreshes the measured
    bias and a bounded preload gain. A separated, non-impacting patch cannot
    retain its previous measured force.
    """
    corrected = measured+np.asarray(gain)[:, None]*(raw-baseline)
    result = np.zeros_like(corrected)
    for i, normal in enumerate(normals):
        raw_normal = float(normal@raw[i])
        if gaps[i] > 0 and raw_normal <= 1e-8:
            continue
        fn = max(float(normal@corrected[i]), 0.)
        tangent = corrected[i]-(normal@corrected[i])*normal
        tangent *= min(1., friction*fn/max(np.linalg.norm(tangent), 1e-12))
        result[i] = fn*normal+tangent
    return result


def measured_pad_targets(initial, points, normals, gaps, measured_normal, desired_forces,
                         *, radius, max_depth, minimum_normal):
    """Pad-centre targets from actual patches and measured normal preload.

    All points/vectors use object coordinates. Penetrating contact points
    are surface midpoints; shifting by half the signed gap recovers the
    object side. Separated pads keep their approach initial guesses.
    """
    targets = np.asarray(initial, float).copy()
    for i, gap in enumerate(np.asarray(gaps).reshape(-1)):
        if gap > 0 or not np.isfinite(points[i]).all() or not np.isfinite(normals[i]).all():
            continue
        desired_normal = max(minimum_normal, float(normals[i]@desired_forces[i]))
        depth = np.clip(-gap*desired_normal/max(measured_normal[i], .02),
            min(5e-5, max_depth), max_depth)
        targets[i] = points[i]+(.5*gap-radius+depth)*normals[i]
    return targets
