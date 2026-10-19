"""C3 with a mixed-integer projection, solved by CasADi's QRQP.

This is the MIQP projection in dairlib's ``C3MIQP``: each contact is either
separated (force fixed at zero) or sustained (gap fixed at zero). Gurobi is
not required. The horizon QP is the same consensus QP as C3; only the
projection changes.
"""

from __future__ import annotations

import numpy as np


def solve(lcs, costs, x0, x_desired, admm_iterations=3, rho_scale=3.0,
          u_lower=None, u_upper=None, **_ignored):
    sequence = lcs if isinstance(lcs, dict) else lcs.sequence(len(costs[1]))
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    horizon = len(sequence["A"])
    desired = _rows(x_desired, x0.size, horizon + 1)
    scaled, force_scale = _scale(sequence)
    Q, R, G, U = costs
    width = x0.size + scaled["D"][0].shape[1] + scaled["B"][0].shape[1]
    delta = [np.zeros(width) for _ in range(horizon)]
    for knot in delta:
        knot[: x0.size] = x0
    dual = [np.zeros(width) for _ in range(horizon)]
    weight = [np.array(matrix, dtype=float).copy() for matrix in G]
    lower, upper = _bounds(u_lower, u_upper, scaled["B"][0].shape[1])

    def horizon_qp(target):
        hessian, gradient, bounds = _assemble(
            scaled, x0, Q, R, weight, desired, target, lower, upper,
            nonnegative_force=True,
        )
        decision = _qrqp(hessian, gradient, bounds=bounds)
        return _unpack(scaled, x0, decision)

    for _ in range(int(admm_iterations)):
        target = [delta[t] - dual[t] for t in range(horizon)]
        z = horizon_qp(target)
        for t in range(horizon):
            delta[t] = _project(U[t], dual[t] + z[t], scaled, t)
            dual[t] = (dual[t] + z[t] - delta[t]) / float(rho_scale)
            weight[t] = weight[t] * float(rho_scale)
    z = horizon_qp([delta[t] - dual[t] for t in range(horizon)])
    return _rollout(sequence, x0, z, force_scale)


def _project(weight, target, scaled, knot):
    """Minimize ||z - target||_U subject to complementarity on one knot."""
    E = scaled["E"][knot]
    F = scaled["F"][knot]
    H = scaled["H"][knot]
    c = scaled["c"][knot]
    n, m = E.shape[1], E.shape[0]
    k = H.shape[1]
    width = n + m + k
    metric = np.asarray(weight, dtype=float)
    hessian = 2.0 * 0.5 * (metric + metric.T)
    gradient = -2.0 * metric.T @ target
    contact = np.hstack([E, F, H])
    best = None
    for mask in range(1 << m):
        rows = [np.eye(width)]
        lower = [-1e3 * np.ones(width)]
        upper = [1e3 * np.ones(width)]
        lam_eye = np.zeros((m, width))
        lam_eye[:, n : n + m] = np.eye(m)
        rows.append(lam_eye)
        lower.append(np.zeros(m))
        upper.append(np.full(m, np.inf))
        rows.append(contact)
        lower.append(-c)
        upper.append(np.full(m, np.inf))
        for index in range(m):
            if mask & (1 << index):
                row = lam_eye[index]
                value = 0.0
            else:
                row = contact[index]
                value = -c[index]
            rows.append(row.reshape(1, -1))
            lower.append(np.array([value]))
            upper.append(np.array([value]))
        decision = _qrqp(
            hessian, gradient, np.vstack(rows), np.concatenate(lower), np.concatenate(upper)
        )
        if decision is None:
            continue
        gap = contact @ decision + c
        force = decision[n : n + m]
        if np.any(force < -1e-6) or np.any(gap < -1e-6):
            continue
        objective = float((decision - target) @ metric @ (decision - target))
        if best is None or objective < best[0]:
            best = (objective, decision)
    if best is None:
        fallback = target.copy()
        fallback[n : n + m] = np.maximum(fallback[n : n + m], 0.0)
        return fallback
    return best[1]


def _assemble(model, x0, Q, R, G, desired, consensus, u_lower, u_upper,
              nonnegative_force=False, return_maps=False):
    horizon = len(model["A"])
    n = model["A"][0].shape[0]
    m = model["D"][0].shape[1]
    k = model["B"][0].shape[1]
    nv = horizon * (m + k)
    hessian = np.zeros((nv, nv))
    gradient = np.zeros(nv)
    state_map = [np.zeros((n, nv))]
    state_offset = [x0.copy()]
    for t in range(horizon):
        base = t * (m + k)
        force_map = np.zeros((m, nv))
        input_map = np.zeros((k, nv))
        force_map[:, base : base + m] = np.eye(m)
        input_map[:, base + m : base + m + k] = np.eye(k)
        state_map.append(
            model["A"][t] @ state_map[t]
            + model["B"][t] @ input_map
            + model["D"][t] @ force_map
        )
        state_offset.append(model["A"][t] @ state_offset[t] + model["d"][t])
        stacked = np.vstack([state_map[t], force_map, input_map])
        offset = np.concatenate(
            [
                state_offset[t] - consensus[t][:n],
                -consensus[t][n : n + m],
                -consensus[t][n + m :],
            ]
        )
        _add_quad(hessian, gradient, stacked, offset, G[t])
        _add_quad(hessian, gradient, state_map[t], state_offset[t] - desired[t], Q[t])
        _add_quad(hessian, gradient, input_map, np.zeros(k), R[t])
    _add_quad(hessian, gradient, state_map[-1], state_offset[-1] - desired[-1], Q[-1])
    lower = np.full(nv, -np.inf)
    upper = np.full(nv, np.inf)
    for t in range(horizon):
        base = t * (m + k)
        if nonnegative_force:
            lower[base : base + m] = 0.0
        if u_lower is not None:
            lower[base + m : base + m + k] = u_lower
            upper[base + m : base + m + k] = u_upper
    if return_maps:
        return hessian, gradient, (lower, upper), (state_map, state_offset)
    return hessian, gradient, (lower, upper)


def _unpack(model, x0, decision):
    horizon = len(model["A"])
    n = x0.size
    m = model["D"][0].shape[1]
    k = model["B"][0].shape[1]
    state = x0.copy()
    knots = []
    for t in range(horizon):
        base = t * (m + k)
        force = decision[base : base + m]
        control = decision[base + m : base + m + k]
        knots.append(np.concatenate([state, force, control]))
        state = model["A"][t] @ state + model["B"][t] @ control + model["D"][t] @ force + model["d"][t]
    return knots


def _rollout(sequence, x0, knots, force_scale):
    horizon = len(sequence["A"])
    n = x0.size
    k = sequence["B"][0].shape[1]
    m = sequence["D"][0].shape[1]
    states = np.zeros((horizon + 1, n))
    inputs = np.zeros((horizon, k))
    forces = np.zeros((horizon, m))
    states[0] = x0
    for t in range(horizon):
        inputs[t] = knots[t][-k:]
        forces[t] = force_scale * knots[t][n : n + m]
        states[t + 1] = (
            sequence["A"][t] @ states[t]
            + sequence["B"][t] @ inputs[t]
            + sequence["D"][t] @ forces[t]
            + sequence["d"][t]
        )
    if not np.isfinite(states).all():
        raise RuntimeError("MIQP returned a non-finite plan")
    return {"x": states, "u": inputs, "lambda": forces, "solver": "miqp"}


def _qrqp(hessian, gradient, constraints=None, lower=None, upper=None, bounds=None):
    import casadi as ca

    hessian = 0.5 * (hessian + hessian.T) + 1e-8 * np.eye(hessian.shape[0])
    gradient = np.asarray(gradient, dtype=float).reshape(-1)
    n = gradient.size
    if constraints is None:
        constraints = np.zeros((0, n))
        lower = np.zeros(0)
        upper = np.zeros(0)
    if bounds is None:
        variable_lower = -1e6 * np.ones(n)
        variable_upper = 1e6 * np.ones(n)
    else:
        variable_lower, variable_upper = bounds
    solver = ca.conic(
        "sampling_qrqp",
        "qrqp",
        {"h": ca.DM(hessian).sparsity(), "a": ca.DM(constraints).sparsity()},
        {"print_iter": False, "print_header": False, "print_info": False, "error_on_fail": False},
    )
    result = solver(
        h=hessian,
        g=gradient,
        a=constraints,
        lba=lower,
        uba=upper,
        lbx=variable_lower,
        ubx=variable_upper,
    )
    decision = np.asarray(result["x"], dtype=float).reshape(-1)
    if not np.isfinite(decision).all():
        return None
    return decision


def _add_quad(hessian, gradient, mapping, offset, weight):
    weight = np.asarray(weight, dtype=float)
    hessian += 2.0 * mapping.T @ weight @ mapping
    gradient += 2.0 * mapping.T @ weight @ np.asarray(offset, dtype=float).reshape(-1)


def _scale(sequence):
    dynamics = np.linalg.norm(sequence["A"][0])
    contact = np.linalg.norm(sequence["D"][0])
    scale = dynamics / contact if contact > 1e-12 else 1.0
    scaled = {
        "A": sequence["A"],
        "B": sequence["B"],
        "d": sequence["d"],
        "F": sequence["F"],
        "D": [matrix * scale for matrix in sequence["D"]],
        "E": [matrix / scale for matrix in sequence["E"]],
        "H": [matrix / scale for matrix in sequence["H"]],
        "c": [vector / scale for vector in sequence["c"]],
    }
    return scaled, scale


def _rows(values, width, count):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return [array.reshape(width).copy() for _ in range(count)]
    return [array[i].copy() for i in range(count)]


def _bounds(lower, upper, width):
    if lower is None and upper is None:
        return None, None
    return np.full(width, -50.0 if lower is None else float(lower)), np.full(
        width, 50.0 if upper is None else float(upper)
    )
