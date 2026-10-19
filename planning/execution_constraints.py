"""First-command bounds matching the Cartesian executor's floor clipping."""
import numpy as np


def supported_flip_command_bounds(gap, lower, upper, max_norm, tolerance=.003):
    """Return a contact command ceiling when a tabletop pivot loses support.

    Keep horizontal torque authority while recovering at most half the
    command radius vertically. The floor/slew intersection is resolved by
    ``floor_command_bounds`` afterwards, before optimization.
    """
    lower, upper = [np.broadcast_to(np.asarray(x, dtype=float), (3,)).copy()
                    for x in (lower, upper)]
    active = bool(np.isfinite(gap) and gap > tolerance)
    if active:
        recovery = min(.25*(float(gap)-tolerance), .5*float(max_norm))
        upper[2] = min(upper[2], -recovery)
    return lower, upper, active


def table_edge_command_bounds(lower, upper, gap, outward, guard_distance, active):
    """Match the axis-aligned tabletop's outward-command projection."""
    lower, upper = [np.broadcast_to(np.asarray(x, dtype=float), (3,)).copy() for x in (lower, upper)]
    applied = bool(active and np.isfinite(gap) and gap <= guard_distance)
    if applied:
        outward = np.asarray(outward, dtype=float).reshape(2)
        axis = int(np.argmax(np.abs(outward)))
        if outward[axis] > 0:
            upper[axis] = min(upper[axis], 0.)
        elif outward[axis] < 0:
            lower[axis] = max(lower[axis], 0.)
    return lower, upper, applied


def floor_command_bounds(tip_z, floor_z, lower, upper, first_bounds=None,
                         previous=None, slew=None, max_norm=None):
    """Keep a feasible command above the executor's endpoint floor.

    The rigid predictor's actual fingertip displacement is smaller than its
    impedance command. Its predicted-state floor alone cannot constrain the
    endpoint that the executor clips. Rebase a stale slew box only if its
    intersection with the floor/force bounds is empty. Emergency recovery
    outside the force ball retains the existing bounded planner/safety brake.
    """
    lower, upper = [np.broadcast_to(np.asarray(x, dtype=float), (3,)).copy() for x in (lower, upper)]
    required = float(floor_z) - float(tip_z)
    radius = np.inf if max_norm is None else float(max_norm)
    if required > min(upper[2], radius) + 1e-10:
        # Keep horizontal edge bounds even when endpoint floor recovery is
        # impossible inside the force ball. Otherwise an old slew interval
        # can override the new edge restriction on acados' first stage.
        bounds, info = floor_command_bounds(float(floor_z)-lower[2], floor_z,
            lower, upper, first_bounds, previous, slew, max_norm)
        if first_bounds is not None and all(np.array_equal(a, b) for a, b in zip(bounds, first_bounds)):
            bounds = first_bounds
        return bounds, dict(applied=False, rebased=info['rebased'], emergency=True)
    lower[2] = max(lower[2], required)
    base_lower, base_upper = lower.copy(), upper.copy()
    if first_bounds is not None:
        lower = np.maximum(lower, first_bounds[0])
        upper = np.minimum(upper, first_bounds[1])
    valid = (np.all(lower <= upper)
             and np.linalg.norm(np.clip(np.zeros(3), lower, upper)) <= radius + 1e-10)
    rebased = False
    if not valid:
        center = np.clip(np.zeros(3) if previous is None else previous, base_lower, base_upper)
        center[2] = np.clip(center[2], max(base_lower[2], -radius), min(base_upper[2], radius))
        if np.linalg.norm(center) > radius:
            # Preserve mandatory vertical recovery; fit XY inside the ball.
            xy_radius = np.sqrt(max(0., radius**2 - center[2]**2))
            center[:2] *= min(1., xy_radius / max(np.linalg.norm(center[:2]), 1e-20))
        if slew is None:
            lower, upper = base_lower, base_upper
        else:
            lower = np.maximum(base_lower, center - float(slew))
            upper = np.minimum(base_upper, center + float(slew))
        rebased = True
    return (lower, upper), dict(applied=True, rebased=rebased, emergency=False)
