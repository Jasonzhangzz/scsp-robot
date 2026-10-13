"""Allegro-only MPC objectives: joint IK pose plus object pose.

This module is independent of the bimanual MPC and its Cartesian contact
targets. Contact gaps/Jacobians still drive the Allegro dynamics; they are
not used as Cartesian position targets in this objective.
"""

import casadi as cs
import numpy as np

from models.contact_geometry import rotation_cs


def build_allegro_ik_costs(param):
    nq, n = param.n_cmd_, param.num_contacts
    x = cs.SX.sym("allegro_state", param.n_qpos_)
    u = cs.SX.sym("allegro_action", nq)
    # Object goal(7), IK joint target(nq), grasp loaded(1), XY bounds(4).
    head_size = 12+nq
    physics = getattr(param, "online_model", None)
    force_cost = physics is not None and (param.force_tracking_weight > 0 or param.pressure_sync_weight > 0)
    extra_size = 0 if not force_cost else 3*n+4*param.max_ncon_*(1+param.n_qvel_)+physics.model_parameter_dim
    cp = cs.SX.sym("allegro_ik_objective", head_size+extra_size)
    goal, ik_q = cp[:7], cp[7:7+nq]
    loaded = cp[7+nq]
    lower, upper = cp[8+nq:10+nq], cp[10+nq:12+nq]
    terms = [cs.DM(np.sqrt(param.posture_weights))*(x[7:7+nq]-ik_q),
        np.sqrt(param.lateral_weight)*(x[:2]-goal[:2]),
        np.sqrt(param.object_weight)*(x[2]-goal[2]),
        np.sqrt(param.quat_weight/8)*cs.reshape(
            rotation_cs(goal[3:7]).T@rotation_cs(x[3:7])-cs.DM.eye(3), 9, 1),
        np.sqrt(param.horizontal_recovery_weight)*cs.fmax(lower-x[:2], 0),
        np.sqrt(param.horizontal_recovery_weight)*cs.fmax(x[:2]-upper, 0)]
    state_residual = cs.vertcat(*terms)
    stage_terms = [state_residual, cs.DM(np.sqrt(param.action_weights))*u]
    if force_cost:
        reference = cs.reshape(cp[head_size:head_size+3*n], 3, n)
        offset = head_size+3*n
        gaps = cp[offset:offset+4*param.max_ncon_]
        offset += 4*param.max_ncon_
        rows = cs.reshape(cp[offset:offset+4*param.max_ncon_*param.n_qvel_],
            4*param.max_ncon_, param.n_qvel_)
        offset += 4*param.max_ncon_*param.n_qvel_
        force = physics.contact_force_fn(x, u, gaps, rows, cp[offset:])
        stage_terms.append(np.sqrt(param.force_tracking_weight)*loaded*(force-cs.reshape(
            rotation_cs(x[3:7])@reference, 3*n, 1)))
        normal = cs.vertcat(*[cs.dot(cs.sum1(rows[4*i:4*i+4, :3]).T/4,
            force[3*i:3*i+3]) for i in range(n)])
        stage_terms.append(np.sqrt(2*param.pressure_sync_weight/max(n-1, 1))*(1-loaded)*
            (normal-cs.sum1(normal)/n))
    residual = cs.vertcat(*stage_terms)
    terminal = np.sqrt(param.terminal_weight)*state_residual
    param.cost_param_head_size = head_size
    param.cost_uses_force_reference = force_cost
    param.residual_fn = cs.Function("allegro_ik_residual", [x, u, cp], [residual])
    param.terminal_residual_fn = cs.Function("allegro_ik_terminal_residual", [x, cp], [terminal])
    tag = getattr(param, "cost_tag", "test")
    return (cs.Function("allegro_ik_pose_cost_v1_"+tag, [x, u, cp], [cs.sumsqr(residual)]),
            cs.Function("allegro_ik_pose_terminal_v1_"+tag, [x, cp], [cs.sumsqr(terminal)]))
