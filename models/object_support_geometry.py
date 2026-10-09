"""Convex mesh pieces for finite tabletop support of nonconvex objects."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh


@lru_cache(maxsize=8)
def _parts(path, modification_time):
    del modification_time  # Cache invalidation when the input mesh changes.
    mesh = trimesh.load(path, force="mesh")
    points = np.asarray(mesh.vertices, dtype=np.float64)
    center = np.asarray(mesh.bounds).mean(axis=0)
    pieces = []
    for x_positive in (False, True):
        for y_positive in (False, True):
            part = points[((points[:, 0] >= center[0]) == x_positive)
                          & ((points[:, 1] >= center[1]) == y_positive)]
            if len(part) < 4 or np.linalg.matrix_rank(part - part.mean(axis=0)) < 3:
                # Sparse meshes such as an eight-vertex box cannot be split
                # this way without losing their support surface.
                pieces = [points]
                break
            pieces.append(part)
        if len(pieces) == 1 and pieces[0] is points:
            break
    result = []
    for part in pieces:
        hull = trimesh.convex.convex_hull(part)
        vertices = np.asarray(hull.vertices)
        faces = np.asarray(hull.faces)
        vertices.setflags(write=False)
        faces.setflags(write=False)
        result.append((vertices, faces))
    return tuple(result)


def support_mesh_parts(mesh_path):
    """Return body-local hulls without rescaling or changing the object mass.

    Each XY quadrant gets its own support manifold, so one deepest foot cannot
    displace the contacts at the other feet. The pieces remain approximations
    of the mesh, and collide only with the finite pedestal in the scene.
    """
    path = Path(mesh_path).resolve()
    return _parts(str(path), path.stat().st_mtime_ns)
