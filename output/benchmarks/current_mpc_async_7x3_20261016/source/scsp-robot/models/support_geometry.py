"""Finite planar support patches shared by collision and MPC linearization."""
import numpy as np


def support_frame(normal, tangents=None):
    normal = np.asarray(normal, dtype=float).reshape(3)
    normal = normal / np.linalg.norm(normal)
    if tangents is None:
        seed = np.eye(3)[np.argmin(np.abs(normal))]
        first = seed - normal * np.dot(seed, normal)
        first /= np.linalg.norm(first)
        tangents = np.stack([first, np.cross(normal, first)])
    return normal, np.asarray(tangents, dtype=float).reshape(2, 3)


def mesh_support_points(vertices, position, rotation, point, normal, *,
                        tangents=None, half_extents=None, edges=None,
                        contact_band=.001, max_points=4):
    """Lowest separated hull points over the finite support rectangle.

    Clipping hull edges also retains support at the rim when the lowest
    original vertex has overhung it. Heights are signed normal distances,
    rather than COM heights or world Z coordinates.
    """
    normal, tangents = support_frame(normal, tangents)
    point = np.asarray(point, dtype=float)
    world = np.asarray(vertices, dtype=float) @ np.asarray(rotation).T + position
    cloud = world
    if half_extents is not None:
        bounds = np.asarray(half_extents, dtype=float).reshape(2)
        uv = (world - point) @ tangents.T
        inside = np.all(np.abs(uv) <= bounds + 1e-9, axis=1)
        cloud = world[inside]
        if not inside.all() and edges is not None and len(edges):
            segments = world[np.asarray(edges, dtype=int)]
            starts = (segments[:, 0] - point) @ tangents.T
            delta = (segments[:, 1] - segments[:, 0]) @ tangents.T
            lo, hi = np.zeros(len(segments)), np.ones(len(segments))
            valid = np.ones(len(segments), dtype=bool)
            for axis in range(2):
                moving = np.abs(delta[:, axis]) > 1e-12
                valid &= moving | (np.abs(starts[:, axis]) <= bounds[axis] + 1e-9)
                denom = np.where(moving, delta[:, axis], 1.)
                a = (-bounds[axis] - starts[:, axis]) / denom
                b = (bounds[axis] - starts[:, axis]) / denom
                lo = np.maximum(lo, np.where(moving, np.minimum(a, b), 0.))
                hi = np.minimum(hi, np.where(moving, np.maximum(a, b), 1.))
            valid &= lo <= hi + 1e-12
            vector = segments[:, 1] - segments[:, 0]
            clipped = np.concatenate([segments[:, 0] + lo[:, None]*vector,
                                      segments[:, 0] + hi[:, None]*vector])
            cloud = np.concatenate([cloud, clipped[np.r_[valid, valid]]])
    if not len(cloud) or int(max_points) <= 0:
        return np.empty((0, 3))
    height = (cloud - point) @ normal
    cloud = cloud[height <= height.min() + float(contact_band)]
    selected = [int(np.argmin((cloud-point) @ normal))]
    uv = cloud @ tangents.T
    distances = np.linalg.norm(uv - uv[selected[0]], axis=1)
    while len(selected) < min(int(max_points), len(cloud)):
        idx = int(np.argmax(distances))
        if distances[idx] <= .002:
            break
        selected.append(idx)
        distances = np.minimum(distances, np.linalg.norm(uv - uv[idx], axis=1))
    return cloud[selected]


def support_jacobians(position, rotation, point, nv, mu, normal=(0., 0., 1.)):
    """Mixed world-linear/body-angular MPC rows and all-body Lambda rows."""
    normal, tangents = support_frame(normal)
    directions = np.array([normal + mu*tangents[0], normal + mu*tangents[1],
                           normal - mu*tangents[0], normal - mu*tangents[1]])
    lever = np.asarray(point) - position
    body_lever = np.asarray(rotation).T @ lever
    rows = np.zeros((4, int(nv)))
    rows[:, :3] = directions
    rows[:, 3:6] = np.cross(body_lever, directions @ rotation)
    body = rows[:, :6].copy()
    body[:, :3] = directions @ rotation
    return rows, body
