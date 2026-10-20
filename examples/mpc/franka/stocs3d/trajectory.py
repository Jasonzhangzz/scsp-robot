"""Turn one unmodified STOCS result into the fingertip path Isaac tracks."""
import numpy as np
from klampt.math import so3, se3

APPROACH_S = 0.6


def _unit(vector):
    vector = np.asarray(vector, dtype=float).reshape(3)
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        raise ValueError('STOCS contact normal vanished')
    return vector / length


def fingertip_knots(result, params, radius):
    """World position of the sphere center along the solved object trajectory."""
    iteration = int(result['total_iter'])
    if iteration < 1 or ('iter_%d' % iteration) not in result:
        raise RuntimeError('STOCS did not return a trajectory')
    knot = result['iter_%d' % iteration]
    orientation = np.asarray(knot['q'], dtype=float).reshape(-1, 4)
    position = np.asarray(knot['x'], dtype=float).reshape(-1, 3)
    steps = np.asarray(knot['dt'], dtype=float).reshape(-1)
    points = params['task_params']['manipulation_contact_points']
    normals = params['task_params']['manipulation_contact_normals']
    times = []
    tips = []
    clock = 0.0
    for i in range(orientation.shape[0]):
        if i:
            clock += float(steps[i])
        rotation = so3.from_quaternion(orientation[i].tolist())
        origin = position[i].tolist()
        world_points = [se3.apply((rotation, origin), list(map(float, point))) for point in points]
        world_normals = [so3.apply(rotation, list(map(float, normal))) for normal in normals]
        center = np.mean(np.asarray(world_points, dtype=float), axis=0)
        normal = _unit(np.mean(np.asarray(world_normals, dtype=float), axis=0))
        times.append(clock)
        tips.append(center - float(radius) * normal)
    return dict(
        solved=bool(result.get('is_success', False)),
        iterations=iteration,
        q_wxyz=orientation.tolist(),
        x=position.tolist(),
        dt=steps.tolist(),
        times=times,
        fingertip_xyz=np.asarray(tips, dtype=float).tolist(),
        manipulation_contact_points=points,
        manipulation_contact_normals=normals,
    )


def position_at(times, points, when):
    times = np.asarray(times, dtype=float).reshape(-1)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if when <= times[0]:
        return points[0].copy()
    if when >= times[-1]:
        return points[-1].copy()
    index = int(np.searchsorted(times, when)) - 1
    span = float(times[index + 1] - times[index])
    alpha = 0.0 if span <= 0.0 else (float(when) - float(times[index])) / span
    return (1.0 - alpha) * points[index] + alpha * points[index + 1]
