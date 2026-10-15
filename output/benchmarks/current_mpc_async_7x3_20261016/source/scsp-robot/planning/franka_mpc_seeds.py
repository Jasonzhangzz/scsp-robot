"""Contact-loading initial guesses. Only optimized MPC results are executable."""
import numpy as np


def contact_loading_seeds(force_world, normal_outward, stiffness, horizon):
    """Yaw-equivariant guesses for loading a patch before lifting along it.

    These are local-solver initial guesses, not action overrides or reference
    paths. The original objective and all constraints decide their outcome.
    """
    force, normal = [np.asarray(x, dtype=float).reshape(3) for x in (force_world, normal_outward)]
    if not np.isfinite(np.r_[force, normal]).all() or np.linalg.norm(normal) < 1e-9:
        return []
    normal = normal/np.linalg.norm(normal)
    load = max(0., -float(force@normal))*(-normal)
    tangent = force-load
    if np.linalg.norm(load) < 1e-4 or tangent[2] <= .05*max(np.linalg.norm(force), 1e-9):
        return []
    ramp = np.linspace(0., 1., int(horizon))[:, None]
    schedules = [load+ramp*tangent, load+(2*ramp-1)*tangent]
    return [np.linalg.solve(stiffness, forces.T).T for forces in schedules]


def rollout_guess(mpc, state, commands, phi, jac):
    """Roll a seed through the same predictor, respecting command bounds."""
    state = np.asarray(state, dtype=float).copy()
    reference = state.copy()
    lower, upper = mpc.param_.mpc_u_lb_, mpc.param_.mpc_u_ub_
    stiffness = np.asarray(mpc.param_.robot_stiff_, dtype=float)
    states, actions = [state.copy()], []
    for k, command in enumerate(commands):
        lo, hi = lower, upper
        if k == 0 and mpc.first_action_bounds is not None:
            lo, hi = mpc.first_action_bounds
        command = np.clip(command, lo, hi)
        if mpc.contact_force_limit is not None:
            size = np.linalg.norm(stiffness@command)
            command *= min(1., mpc.contact_force_limit/max(size, 1e-12))
            command = np.clip(command, lo, hi)
            if np.linalg.norm(stiffness@command) > mpc.contact_force_limit+1e-7:
                return None
        state = np.asarray(mpc.stage_prediction_fn(state, command, phi, jac,
            mpc.model.runtime_parameters(mpc.param_.model_params), reference)).reshape(-1)
        if not np.isfinite(state).all():
            return None
        actions.append(command); states.append(state.copy())
    actions, states = np.asarray(actions), np.asarray(states)
    return dict(x0=np.column_stack([actions, states[1:]]).reshape(-1),
                u_traj=actions, x_traj=states)


def best_contact_plan(mpc, plan_kwargs, seeds):
    """Compare solved trajectories under the unchanged MPC objective.

    Always retain the ordinary solve as a candidate. Never execute a seed,
    and never replace a valid ordinary result with a failed/non-finite solve.
    """
    best = mpc.plan_once(**plan_kwargs)
    costs = [float(np.asarray(best.get('cost_opt', np.inf)).reshape(-1)[0])]
    selected = 0
    if best.get('solver_backend') != 'acados':
        return best
    for seed in seeds:
        if seed is None:
            continue
        mpc._reset_acados_solver()
        candidate = mpc.plan_once(**dict(plan_kwargs, sol_guess=seed))
        cost = float(np.asarray(candidate.get('cost_opt', np.inf)).reshape(-1)[0])
        commands = np.asarray(candidate.get('u_traj', []), dtype=float)
        valid = (candidate.get('solver_backend') == 'acados' and np.isfinite(cost)
                 and commands.size > 0 and np.isfinite(commands).all())
        if valid:
            lo, hi = plan_kwargs.get('u_lb', mpc.param_.mpc_u_lb_), plan_kwargs.get('u_ub', mpc.param_.mpc_u_ub_)
            valid = bool(np.all(commands >= np.asarray(lo)-1e-7) and np.all(commands <= np.asarray(hi)+1e-7))
            if mpc.first_action_bounds is not None:
                valid &= bool(np.all(commands[0] >= mpc.first_action_bounds[0]-1e-7)
                              and np.all(commands[0] <= mpc.first_action_bounds[1]+1e-7))
            if mpc.contact_force_limit is not None:
                valid &= bool(np.all(np.linalg.norm(commands@np.asarray(mpc.param_.robot_stiff_).T,axis=1)
                                      <= mpc.contact_force_limit+1e-7))
        costs.append(cost if valid else float('inf'))
        if valid and cost < costs[selected]-1e-9:
            best, selected = candidate, len(costs)-1
    # Clear dual memory belonging to a rejected branch. The selected primal
    # trajectory is supplied explicitly on the next observation.
    if len(costs) > 1:
        mpc._reset_acados_solver()
    best['contact_seed_costs'] = [c if np.isfinite(c) else None for c in costs]
    best['contact_seed_selected'] = selected
    return best
