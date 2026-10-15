"""Certify nearby finger patches together with the contacts already made."""
from itertools import combinations

import numpy as np

from planning.grasp_verification import verify_grasp


def choose_local_contact_updates(points, normals, nearest_points, nearest_normals,
                                 touching, reachable, normal_limits, friction,
                                 external_wrench):
    """Return the largest feasible subset of reachable finger substitutions.

    Untouched, unselected fingers never contribute to the certificate. New
    contacts have a prospective pressure limit; this is a planning certificate,
    not permission to apply their forces before physical contact occurs.
    """
    points, normals = np.asarray(points).copy(), np.asarray(normals).copy()
    touching, reachable = np.asarray(touching, bool), np.asarray(reachable, bool)
    points[touching] = np.asarray(nearest_points)[touching]
    normals[touching] = np.asarray(nearest_normals)[touching]
    candidates = np.flatnonzero(reachable & ~touching)
    for size in range(len(candidates), 0, -1):
        for subset in combinations(candidates, size):
            chosen = np.asarray(subset, int)
            participating = touching.copy()
            participating[chosen] = True
            proposal, directions = points.copy(), normals.copy()
            proposal[chosen] = np.asarray(nearest_points)[chosen]
            directions[chosen] = np.asarray(nearest_normals)[chosen]
            certificate = verify_grasp(proposal[participating], directions[participating],
                friction, external_wrench=external_wrench,
                normal_force_limits=np.asarray(normal_limits)[participating])
            if certificate['force_closure'] and certificate['load_feasible']:
                return proposal, directions, chosen, certificate
    return points, normals, np.empty(0, int), None
