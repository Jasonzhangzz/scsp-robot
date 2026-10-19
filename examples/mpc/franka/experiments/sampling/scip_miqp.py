"""C3MIQP with SCIP in place of Gurobi.

Each projection has one binary per contact ray. Indicator constraints encode
lambda=0 OR gap=0, with both nonnegative. The quadratic objective is put in
epigraph form because SCIP's Python interface accepts linear objectives.
This is the same MIQP, not a complementarity penalty or convex relaxation.
"""

import numpy as np

from .c3 import _extension
from .miqp import _assemble, _rows, _scale, _unpack, _rollout


def project(weight, target, E, F, H, c, *, time_limit=.1, relative_gap=1e-3):
    try:
        from pyscipopt import Model, quicksum
    except ImportError as error:
        raise ImportError('Rigid sampling C3 requires SCIP: python -m pip install "pyscipopt>=5.2,<6"') from error
    n, m, k = E.shape[1], E.shape[0], H.shape[1]
    width = n + m + k
    metric = .5 * (np.asarray(weight) + np.asarray(weight).T)
    if np.linalg.eigvalsh(metric).min() < -1e-10:
        raise ValueError('MIQP projection metric must be positive semidefinite')
    scip = Model('c3_complementarity_projection')
    scip.hideOutput()
    scip.setRealParam('limits/time', float(time_limit))
    scip.setRealParam('limits/gap', float(relative_gap))
    scip.setIntParam('parallel/maxnthreads', 1)
    scip.setRealParam('numerics/feastol', 1e-7)
    scip.setBoolParam('constraints/nonlinear/tightenlpfeastol', False)
    z = [scip.addVar(lb=0. if n <= i < n+m else -100., ub=100., name='z%d' % i)
         for i in range(width)]
    contact = np.hstack([E, F, H])
    gaps, binaries = [], []
    for i in range(m):
        gap = scip.addVar(lb=0., name='gap%d' % i)
        scip.addCons(gap == quicksum(float(contact[i, j]) * z[j]
                                    for j in np.flatnonzero(contact[i])) + float(c[i]))
        separated = scip.addVar(vtype='B', name='separated%d' % i)
        gaps.append(gap)
        binaries.append(separated)
        scip.addConsIndicator(z[n+i] <= 0., binvar=separated, activeone=True)
        scip.addConsIndicator(gap <= 0., binvar=separated, activeone=False)
    objective = scip.addVar(lb=0., name='quadratic_objective')
    # Keep cross terms as well, so a non-diagonal metric is not silently lost.
    difference = [z[i] - float(target[i]) for i in range(width)]
    quadratic = quicksum(float(metric[i, i]) * difference[i]**2 for i in range(width))
    quadratic += quicksum(2 * float(metric[i, j]) * difference[i] * difference[j]
                          for i in range(width) for j in range(i+1, width) if metric[i, j] != 0.)
    scip.addCons(objective >= quadratic)
    scip.setObjective(objective, 'minimize')
    # Supply an exactly complementary LCP solution as an incumbent. SCIP
    # still optimizes the MIQP; this just makes a time-limited solve useful
    # immediately rather than returning no feasible contact mode.
    affine = E @ target[:n] + H @ target[n+m:] + c
    try:
        force = np.asarray(_extension().solve_qp(F, affine, np.eye(m), np.zeros(m))).reshape(-1)
        seed = target.copy()
        seed[n:n+m] = np.maximum(force, 0.)
        slack = contact @ seed + c
        if (np.max(np.abs(seed)) <= 100. and slack.min() >= -1e-7
                and np.max(np.abs(seed[n:n+m]*slack)) < 1e-6):
            incumbent = scip.createSol()
            for variable, value in zip(z, seed):
                scip.setSolVal(incumbent, variable, float(value))
            for i in range(m):
                scip.setSolVal(incumbent, gaps[i], float(max(0., slack[i])))
                scip.setSolVal(incumbent, binaries[i], float(force[i] < 1e-7))
            error = seed-target
            scip.setSolVal(incumbent, objective, float(error @ metric @ error) + 1e-7)
            scip.addSol(incumbent, free=True)
    except RuntimeError:
        pass  # No seed: SCIP must find a feasible integer incumbent itself.
    scip.optimize()
    # A budget may expire before the first integer incumbent. Resume the
    # SAME MIQP with a larger budget; never substitute a relaxed projection.
    for retry_limit in (max(.5, 4*time_limit), max(2., 8*time_limit)):
        if scip.getNSols() > 0:
            break
        scip.setRealParam('limits/time', float(retry_limit))
        scip.optimize()
    if scip.getNSols() == 0:
        status = str(scip.getStatus())
        scip.freeProb()
        raise RuntimeError('SCIP MIQP projection has no feasible incumbent (%s)' % status)
    solution = scip.getBestSol()
    value = np.array([scip.getSolVal(solution, variable) for variable in z])
    modes = np.array([scip.getSolVal(solution, variable) > .5 for variable in binaries])
    info = dict(status=str(scip.getStatus()), mip_gap=float(scip.getGap()), nodes=int(scip.getNNodes()),
                solve_time=float(scip.getSolvingTime()))
    scip.freeProb()
    # Polish the selected INTEGER mode as an equality-constrained QP. This
    # removes tiny indicator tolerances before force*gap is evaluated in SI
    # units; it does not change the discrete mode or relax complementarity.
    from scipy.linalg import null_space
    force_rows = np.zeros((m, width))
    force_rows[:, n:n+m] = np.eye(m)
    equality = np.where(modes[:, None], force_rows, contact)
    rhs = np.where(modes, 0., -c)
    particular = np.linalg.lstsq(equality, rhs, rcond=1e-10)[0]
    null = null_space(equality, rcond=1e-10)
    if null.shape[1]:
        inequalities = np.vstack([force_rows, contact, np.eye(width), -np.eye(width)])
        bounds = np.r_[np.zeros(m), -c, np.full(width, -100.), np.full(width, -100.)]
        try:
            reduced = np.asarray(_extension().solve_qp(
                2 * null.T @ metric @ null,
                2 * null.T @ metric @ (particular-target),
                inequalities @ null, bounds-inequalities @ particular)).reshape(-1)
            value = particular + null @ reduced
        except RuntimeError:
            pass  # Keep the SCIP incumbent, which is checked below.
    slack, force = contact @ value + c, value[n:n+m]
    if min(slack.min(), force.min()) < -1e-6 or np.max(np.abs(force * slack)) > 1e-4:
        raise RuntimeError('SCIP MIQP incumbent violates complementarity')
    return value, info


def solve(lcs, costs, x0, desired, *, admm_iterations=3, rho_scale=3.,
          u_limit=50., projection_time_limit=.1):
    sequence = lcs.sequence(len(costs[1]))
    model, scale = _scale(sequence)
    x0 = np.asarray(x0, dtype=float)
    N, n, m, k = len(sequence['A']), lcs.n, lcs.m, lcs.k
    desired = _rows(desired, n, N+1)
    Q, R, G, U = costs
    weights = [g.copy() for g in G]
    delta = [np.r_[x0, np.zeros(m+k)] for _ in range(N)]
    dual = [np.zeros(n+m+k) for _ in range(N)]
    lower, upper = np.full(k, -u_limit), np.full(k, u_limit)
    core = _extension()
    statistics = []

    def horizon_qp(target):
        hessian, gradient, bounds, maps = _assemble(model, x0, Q, R, weights, desired, target,
                                               lower, upper, nonnegative_force=False, return_maps=True)
        lo, hi = bounds
        identity = np.eye(len(lo))
        il, iu = np.isfinite(lo), np.isfinite(hi)
        constraints = np.vstack([identity[il], -identity[iu]])
        bound = np.r_[lo[il], -hi[iu]]
        if hasattr(lcs, 'ee_floor_z'):
            state_map, offset = maps
            # Replacing EE-ground contacts by this workspace constraint is
            # the native jack's contact reduction. It saves four binaries
            # while keeping every future EE knot above the physical floor.
            constraints = np.vstack([constraints, *[mapping[2:3] for mapping in state_map[1:]]])
            bound = np.r_[bound, [lcs.ee_floor_z-value[2] for value in offset[1:]]]
        decision = np.asarray(core.solve_qp(hessian, gradient, constraints, bound)).reshape(-1)
        return _unpack(model, x0, decision)

    for _ in range(admm_iterations):
        z = horizon_qp([delta[t]-dual[t] for t in range(N)])
        for t in range(N):
            delta[t], info = project(U[t], dual[t]+z[t], model['E'][t], model['F'][t],
                                     model['H'][t], model['c'][t], time_limit=projection_time_limit)
            statistics.append(info)
            dual[t] = (dual[t]+z[t]-delta[t]) / rho_scale
            weights[t] *= rho_scale
    z = horizon_qp([delta[t]-dual[t] for t in range(N)])
    result = _rollout(sequence, x0, z, scale)
    result['projected_u'] = np.asarray([knot[-k:] for knot in delta])
    result['projected_lambda'] = scale*np.asarray([knot[n:n+m] for knot in delta])
    result['solver'] = 'c3-miqp-scip'
    result['projection_statistics'] = statistics
    return result
