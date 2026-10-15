"""Bounded force allocation at the current physical pad contact patches."""
import numpy as np
from scipy.optimize import linprog, minimize

from planning.grasp_verification import contact_wrench_rays


def allocate_contact_wrench(points, normals, wrench, friction, *, minimum_normal=.2,
                            force_limit=10., friction_margin=.8, normal_limits=None, rotational_friction=None,
                            friction_cone='coulomb', tangent_directions=None):
    """Allocate a wrench while retaining compressive preload on every pad.

    Geometry, forces and torque origins must use one frame. The feasibility
    LP uses an inscribed polygon; the refinement penalizes force magnitudes
    only after the requested wrench is imposed as an equality. It therefore
    cannot trade necessary tangential lifting force for a smaller force cost.
    This certifies force allocation, not joint-space actuator reachability.
    """
    points, normals = np.asarray(points, float).reshape(-1, 3), np.asarray(normals, float).reshape(-1, 3)
    wrench = np.asarray(wrench, float).reshape(6)
    if len(points) != len(normals) or not len(points):
        raise ValueError('A nonempty matching set of contact points and normals is required')
    minimums = np.broadcast_to(np.asarray(minimum_normal, float), (len(points),)).copy()
    if (not np.isfinite(np.r_[points.ravel(), normals.ravel(), wrench,
                             friction, minimums, force_limit, friction_margin]).all()
            or friction < 0 or np.any(minimums < 0) or force_limit <= 0
            or not 0 < friction_margin <= 1):
        raise ValueError('Invalid contact force allocation parameters')
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError('Contact normals must be nonzero')
    normals = normals/lengths[:, None]
    mu = float(friction)*float(friction_margin)
    if friction_cone not in ('coulomb','pyramidal'):raise ValueError('Invalid friction cone')
    edges, n = (4 if friction_cone=='pyramidal' else 16), len(points)
    if friction_cone=='pyramidal' and tangent_directions is None:
        from planning.contact_friction import native_tangents
        tangent_directions=native_tangents(normals)
    moment_coefficients = (None if rotational_friction is None else
                           tuple(friction_margin*np.broadcast_to(np.asarray(value, float), (n,))
                                 for value in rotational_friction))
    rays = contact_wrench_rays(points, normals, mu, edges, rotational_friction=moment_coefficients,
        tangent_directions=tangent_directions)
    edges = rays.shape[1]//n
    normal_map = np.kron(np.eye(n), np.ones((1, edges)))
    # Bounding the sum of rays by this value also bounds the force norm.
    max_normal = np.full(n, force_limit/np.sqrt(1+mu*mu))
    if normal_limits is not None:
        caps = np.asarray(normal_limits, float).reshape(n)
        if not np.isfinite(caps).all() or np.any(caps < 0):
            raise ValueError('Normal limits must be finite and nonnegative')
        max_normal = np.minimum(max_normal, caps)
    scale = max(np.linalg.norm(points, axis=1).max(), .001)
    scaling = np.r_[np.ones(3), np.full(3, 1/scale)]
    lp = linprog(np.ones(n*edges), A_eq=scaling[:, None]*rays, b_eq=scaling*wrench,
        A_ub=np.r_[normal_map, -normal_map],
        b_ub=np.r_[max_normal, -minimums],
        bounds=(0., None), method='highs')
    if not lp.success:
        return dict(feasible=False, forces=None, residual=None)
    initial = np.array([rays[:3, i*edges:(i+1)*edges]@lp.x[i*edges:(i+1)*edges]
                        for i in range(n)]).reshape(-1)
    if edges == 10:
        forces = initial.reshape(n, 3)
        torques = np.array([rays[3:, i*edges:(i+1)*edges]@lp.x[i*edges:(i+1)*edges]
                           for i in range(n)])-np.cross(points, forces)
        return dict(feasible=True, forces=forces, moments=torques,
                    residual=float(np.linalg.norm(rays@lp.x-wrench)),
                    normal_forces=np.sum(forces*normals, axis=1), requested_wrench=wrench.copy())
    matrix = np.zeros((6, 3*n))
    weights = []
    for i, (point, normal) in enumerate(zip(points, normals)):
        matrix[:3, 3*i:3*i+3] = np.eye(3)
        px, py, pz = point
        matrix[3:, 3*i:3*i+3] = [[0, -pz, py], [pz, 0, -px], [-py, px, 0]]
        weights.append(.01*np.eye(3)+.03*(np.eye(3)-np.outer(normal, normal)))
    weighted = np.zeros((3*n, 3*n))
    for i, block in enumerate(weights):
        weighted[3*i:3*i+3, 3*i:3*i+3] = block
    pyramid_rows=np.zeros((4*n,3*n))
    if friction_cone=='pyramidal':
        for i,(normal,tangents) in enumerate(zip(normals,np.asarray(tangent_directions))):
            for j,(s1,s2) in enumerate(((1,1),(1,-1),(-1,1),(-1,-1))):
                pyramid_rows[4*i+j,3*i:3*i+3]=mu*normal-s1*tangents[0]-s2*tangents[1]
    def inequality(flat):
        forces = flat.reshape(n, 3)
        normal = np.sum(normals*forces, axis=1)
        squared = np.sum(forces*forces, axis=1)
        cone=(pyramid_rows@flat if friction_cone=='pyramidal' else (1+mu*mu)*normal*normal-squared)
        return np.r_[normal-minimums, max_normal-normal, cone,
                     force_limit*force_limit-squared]
    def inequality_jac(flat):
        forces = flat.reshape(n, 3)
        normal = np.sum(normals*forces, axis=1)
        cone_rows=4*n if friction_cone=='pyramidal' else n
        jac = np.zeros((3*n+cone_rows, 3*n))
        for i in range(n):
            jac[i, 3*i:3*i+3] = normals[i]
            jac[n+i, 3*i:3*i+3] = -normals[i]
            if friction_cone!='pyramidal':
                jac[2*n+i, 3*i:3*i+3] = 2*((1+mu*mu)*normal[i]*normals[i]-forces[i])
            jac[2*n+cone_rows+i, 3*i:3*i+3] = -2*forces[i]
        if friction_cone=='pyramidal':jac[2*n:2*n+cone_rows]=pyramid_rows
        return jac
    scaled = scaling[:, None]*matrix
    result = minimize(lambda f: (float(f@weighted@f), 2*weighted@f), initial,
        method='SLSQP', jac=True, constraints=[
            dict(type='eq', fun=lambda f: scaled@f-scaling*wrench, jac=lambda f: scaled),
            dict(type='ineq', fun=inequality, jac=inequality_jac)],
        options=dict(maxiter=30, ftol=1e-9))
    candidate = result.x
    # The feasible LP remains a valid fallback when the refinement stops early.
    if (not np.isfinite(candidate).all() or np.min(inequality(candidate)) < -1e-7
            or np.linalg.norm(matrix@candidate-wrench) > 1e-7):
        candidate = initial
    return dict(feasible=True, forces=candidate.reshape(n, 3),
        residual=float(np.linalg.norm(matrix@candidate-wrench)),
        normal_forces=np.sum(candidate.reshape(n, 3)*normals, axis=1),
        requested_wrench=wrench.copy())


def normal_pressure_command(tip_jacobian, normals, measured, reference, admittance, limit,
                            *, grasp_established=False):
    """Split finger normal preload from MPC tangential/rigid-hand motion.

    The six base commands are preserved. Before grasp, finger motion cancels
    base motion relative to a stationary object. With a held object, rigid
    base motion carries both surfaces and must not be cancelled again.
    Pressure tracks the allocated preload; MPC retains the finger normal
    nullspace. Prediction and execution share this affine command map.
    """
    normals = np.asarray(normals, float).reshape(-1, 3)
    jac = np.asarray(tip_jacobian, float).reshape(len(normals), 3, -1)
    normal_jac = np.einsum('ij,ijk->ik', normals, jac)
    finger = normal_jac[:, 6:]
    point_jac = jac[:, :, 6:].reshape(3*len(normals), -1)
    normal_motion = np.zeros((3*len(normals), len(normals)))
    for i, normal in enumerate(normals):
        normal_motion[3*i:3*i+3, i] = normal
    # Control actual pad normal displacement without introducing tangential
    # motion through an anisotropic finger Jacobian (the old normal-only
    # inverse could roll the object while trying to squeeze it).
    inverse = point_jac.T@np.linalg.solve(point_jac@point_jac.T+1e-5*np.eye(3*len(normals)), normal_motion)
    projection = np.eye(jac.shape[-1])
    if not grasp_established:
        projection[6:, :6] = -inverse@normal_jac[:, :6]
    projection[6:, 6:] -= inverse@finger
    motion = np.clip(admittance*(np.asarray(reference)-np.asarray(measured)), -limit, limit)
    bias = np.r_[np.zeros(6), inverse@motion]
    return projection, bias
