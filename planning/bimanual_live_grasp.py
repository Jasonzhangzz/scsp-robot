"""Measured patch certificates, local replacements and balanced wrench targets.

Two frictional point contacts cannot close all six wrench directions. Their
rank-five positive span and the requested load are certified separately;
Additional real manifold patches, or physical condim=4/6 friction moments,
may provide full six-dimensional closure.
"""
from itertools import combinations, product

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from planning.grasp_force_allocation import allocate_contact_wrench
from planning.grasp_verification import contact_wrench_rays, verify_grasp


def task_wrench_in_contact_span(points, wrench):
    """Keep the requested force; expose torque the point contacts cannot supply."""
    points = np.asarray(points, float).reshape(-1, 3)
    requested = np.asarray(wrench, float).reshape(6)
    matrix = np.zeros((6, 3*len(points)))
    for i, (x, y, z) in enumerate(points):
        matrix[:3, 3*i:3*i+3] = np.eye(3)
        matrix[3:, 3*i:3*i+3] = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    projected = requested.copy()
    if len(points) >= 2 and np.linalg.matrix_rank(matrix, tol=1e-8) == 5:
        _, _, vh = np.linalg.svd(matrix.T, full_matrices=True)
        null = vh[-1]
        projected[3:] -= null[3:] * (null @ requested) / (null[3:] @ null[3:])
    return projected, requested-projected


def contact_certificate(points, normals, friction, external, *, normal_limits=None, force_limit=12.,
                        rotational_friction=None):
    points = np.asarray(points, float).reshape(-1, 3)
    normals = np.asarray(normals, float).reshape(-1, 3)
    report = verify_grasp(points, normals, friction, external_wrench=external,
                          normal_force_limits=normal_limits, force_limit=force_limit,
                          rotational_friction=rotational_friction)
    margin = report['positive_span_margin']
    if report['wrench_rank'] == 5:
        rays = contact_wrench_rays(points, normals, friction, rotational_friction=rotational_friction)
        radius = max(np.linalg.norm(points - points.mean(0), axis=1).max(), .001)
        matrix = np.r_[np.ones(3), np.full(3, 1/radius)][:, None] * rays
        count = matrix.shape[1]
        lp = linprog(np.r_[np.zeros(count), -1.],
            A_eq=np.vstack((np.c_[matrix, np.zeros(6)], np.r_[np.ones(count), 0.])),
            b_eq=np.r_[np.zeros(6), 1.], A_ub=np.c_[-np.eye(count), np.ones(count)],
            b_ub=np.zeros(count), bounds=(0., None), method='highs')
        margin = float(lp.x[-1]) if lp.success else 0.
    report.update(force_closure_in_contact_span=bool(report['wrench_rank'] >= 5 and margin > 1e-7),
                  contact_span_margin=margin, full_wrench_force_closure=report['force_closure'])
    supported, unsupported = task_wrench_in_contact_span(points, -np.asarray(external))
    if report['wrench_rank'] == 6:
        supported, unsupported = -np.asarray(external), np.zeros(6)
    task = (verify_grasp(points, normals, friction, external_wrench=-supported,
                         normal_force_limits=normal_limits, force_limit=force_limit,
                         rotational_friction=rotational_friction)
            if report['wrench_rank'] == 5 else report)
    report.update(task_load_feasible=bool(task['load_feasible']),
                  uncontrolled_wrench=unsupported)
    return report


def choose_nearby_pair(points, normals, measured_patches, candidates, friction, external,
                       *, force_limit=12., pair_quality=None, rotational_friction=None):
    """Keep every loaded patch; certify only those plus selected new patches.

    candidates[arm] contains reachable nearby (point, normal) pairs, in distance
    order. Remote unselected anchors never contribute to a grasp certificate.
    Returned replacements are future targets, never measured force capacity.
    """
    points, normals = np.asarray(points, float).copy(), np.asarray(normals, float).copy()
    touching = np.array([bool(patches) for patches in measured_patches])
    real = [patch for patches in measured_patches for patch in patches]
    for arm, patches in enumerate(measured_patches):
        if patches:
            primary = max(patches, key=lambda p: p['normal_force'])
            points[arm], normals[arm] = primary['point'], primary['normal']
    missing = [arm for arm in range(2) if not touching[arm] and candidates[arm]]
    for size in range(len(missing), 0, -1):
        best = None
        for arms in combinations(missing, size):
            for patches in product(*(candidates[arm] for arm in arms)):
                future_points, future_normals = points.copy(), normals.copy()
                for arm, (point, normal) in zip(arms, patches):
                    future_points[arm], future_normals[arm] = point, normal
                quality = 0. if pair_quality is None else float(pair_quality(future_points, future_normals))
                if not np.isfinite(quality):
                    continue
                geometry = real + [dict(point=p, normal=n) for p, n in patches]
                caps = [p['normal_force'] for p in real] + [force_limit / np.sqrt(1+friction**2)] * size
                moment_limits = None
                if rotational_friction is not None:
                    moment_limits = [np.r_[np.concatenate([np.full(len(patches), np.broadcast_to(value, (2,))[arm])
                        for arm, patches in enumerate(measured_patches)]),
                        [np.broadcast_to(value, (2,))[arm] for arm in arms]] for value in rotational_friction]
                cert = contact_certificate([p['point'] for p in geometry], [p['normal'] for p in geometry],
                    friction, external, normal_limits=caps, force_limit=force_limit,
                    rotational_friction=moment_limits)
                if cert['force_closure_in_contact_span'] and cert['task_load_feasible']:
                    geometry_points = np.asarray([p['point'] for p in geometry])
                    radius = max(np.linalg.norm(geometry_points-geometry_points.mean(0), axis=1).max(), .001)
                    travel = sum(np.linalg.norm(point-points[arm]) for arm, (point, _) in zip(arms, patches))
                    score = quality if pair_quality is not None else .02*travel/radius
                    if best is None or score < best[0]:
                        best = (score, arms, patches, cert)
        if best is not None:
            _, arms, patches, cert = best
            for arm, (point, normal) in zip(arms, patches):
                points[arm], normals[arm] = point, normal
            return points, normals, np.asarray(arms, int), cert
    return points, normals, np.empty(0, int), None


def requested_contact_wrench(pose, goal, velocity, inertia, external, *,
                             lift_speed=.03, rotation_speed=.2):
    """Return a body-frame wrench about the measured free-joint origin."""
    rotation = Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]]).as_matrix()
    target_rotation = Rotation.from_quat(np.asarray(goal)[[4, 5, 6, 3]]).as_matrix()
    linear = (np.asarray(goal)[:3]-np.asarray(pose)[:3])/.1
    linear *= min(1., lift_speed/max(np.linalg.norm(linear), 1e-10))
    angular = Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()/.25
    angular *= min(1., rotation_speed/max(np.linalg.norm(angular), 1e-10))
    frame = np.eye(6)
    frame[3:, 3:] = rotation
    acceleration = (np.r_[linear, angular]-frame @ np.asarray(velocity))/.1
    wrench = frame @ np.asarray(inertia) @ frame.T @ acceleration - np.asarray(external)
    return np.r_[rotation.T @ wrench[:3], rotation.T @ wrench[3:]]


def live_force_target(points, normals, friction, requested, external_local, *,
                      minimum_normal=.05, force_limit=12., friction_margin=.8, rotational_friction=None):
    """Try the task wrench, then brake to gravity support if it is infeasible.

    The current measured normal limits are deliberately not used to cap a
    FUTURE pressure target. A separate certificate checks capacity available now.
    """
    support = -np.asarray(external_local)
    for scale in (1., .5, 0.):
        target = support + scale*(np.asarray(requested)-support)
        allocation = allocate_contact_wrench(points, normals, target, friction,
            minimum_normal=minimum_normal, force_limit=force_limit, friction_margin=friction_margin,
            rotational_friction=rotational_friction)
        if allocation['feasible']:
            return dict(allocation, task_wrench_scale=scale, uncontrolled_wrench=np.zeros(6))
        projected, unsupported = task_wrench_in_contact_span(points, target)
        if np.linalg.norm(unsupported) > 1e-10:
            allocation = allocate_contact_wrench(points, normals, projected, friction,
                minimum_normal=minimum_normal, force_limit=force_limit, friction_margin=friction_margin,
                rotational_friction=rotational_friction)
            if allocation['feasible']:
                return dict(allocation, task_wrench_scale=scale, uncontrolled_wrench=unsupported)
    return dict(feasible=False, forces=None, residual=None, task_wrench_scale=None)
