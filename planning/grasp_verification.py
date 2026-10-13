"""Object-only certificates for frictional point-contact grasps.

The circular Coulomb cones are approximated from the INSIDE by polygonal
rays. Full wrench rank plus a strictly positive zero-wrench combination
certifies force closure for those rays, hence for the circular cones.
This does not certify hand reachability or that physical contacts exist.
"""

import numpy as np
from scipy.optimize import linprog


def contact_wrench_rays(points, inward_normals, friction, edges=16):
    points = np.asarray(points, float).reshape(-1, 3)
    normals = np.asarray(inward_normals, float).reshape(points.shape)
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError("Contact geometry must be finite")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-12) or not np.isfinite(friction) or friction < 0 or edges < 4:
        raise ValueError("Nonzero normals, nonnegative friction and at least four rays are required")
    normals = normals / lengths[:, None]
    theta = np.arange(edges) * (2*np.pi/edges)
    columns = []
    for point, normal in zip(points, normals):
        helper = np.array([1., 0., 0.]) if abs(normal[0]) <= .9 else np.array([0., 1., 0.])
        tangent = np.cross(normal, helper)
        tangent /= np.linalg.norm(tangent)
        second = np.cross(normal, tangent)
        force = normal + friction * (np.cos(theta)[:, None]*tangent + np.sin(theta)[:, None]*second)
        columns.append(np.c_[force, np.cross(np.broadcast_to(point, force.shape), force)].T)
    return np.hstack(columns) if columns else np.empty((6, 0))


def verify_grasp(points, inward_normals, friction, *, external_wrench=None,
                 force_limit=None, normal_force_limits=None, edges=16):
    """Return a conservative force-closure and bounded load certificate.

All geometry and external wrenches use one frame and moments about the
same origin. The load LP bounds each contact's sum of ray coefficients;
its forces consequently obey both Coulomb friction and the force ball.
"""
    points = np.asarray(points, float).reshape(-1, 3)
    rays = contact_wrench_rays(points, inward_normals, friction, edges)
    scale = max(float(np.linalg.norm(points-points.mean(0), axis=1).max()), .001) if len(points) else 1.
    scaling = np.r_[np.ones(3), np.full(3, 1/scale)]
    matrix = scaling[:, None]*rays
    rank = int(np.linalg.matrix_rank(matrix, tol=1e-8))
    count = matrix.shape[1]
    margin = 0.
    if rank == 6:
        # maximize eps: W*a=0, sum(a)=1, and every a_i >= eps.
        objective = np.r_[np.zeros(count), -1.]
        equality = np.vstack((np.c_[matrix, np.zeros(6)], np.r_[np.ones(count), 0.]))
        result = linprog(objective, A_ub=np.c_[-np.eye(count), np.ones(count)],
            b_ub=np.zeros(count), A_eq=equality, b_eq=np.r_[np.zeros(6), 1.],
            bounds=[(0., None)]*(count+1), method="highs")
        if result.success:
            margin = float(result.x[-1])
    report = dict(force_closure=bool(rank == 6 and margin > 1e-7), wrench_rank=rank,
        positive_span_margin=margin, cone="inscribed_coulomb_polygon", cone_edges=edges,
        load_feasible=None, load_residual=None)
    if external_wrench is not None and count:
        external = np.asarray(external_wrench, float).reshape(6)
        bounds = None
        if force_limit is not None or normal_force_limits is not None:
            if normal_force_limits is not None:
                normal_limits = np.asarray(normal_force_limits, float).reshape(len(points))
                if not np.isfinite(normal_limits).all() or np.any(normal_limits < 0):
                    raise ValueError("normal_force_limits must be finite and nonnegative")
            else:
                normal_limits = np.full(len(points), float(force_limit)/np.sqrt(1+friction**2))
            if force_limit is not None and (not np.isfinite(force_limit) or force_limit <= 0):
                raise ValueError("force_limit must be positive")
            bounds = np.zeros((len(points), count))
            for i in range(len(points)):
                bounds[i, i*edges:(i+1)*edges] = 1.
        result = linprog(np.ones(count), A_eq=matrix, b_eq=-scaling*external,
            A_ub=bounds, b_ub=None if bounds is None else
            normal_limits,
            bounds=(0., None), method="highs")
        report["load_feasible"] = bool(result.success)
        if result.success:
            report["load_residual"] = float(np.linalg.norm(rays@result.x+external))
    return report
