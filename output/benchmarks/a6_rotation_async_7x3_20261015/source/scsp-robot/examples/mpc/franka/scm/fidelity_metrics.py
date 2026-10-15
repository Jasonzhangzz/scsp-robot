"""Paper Section IV statistics and additional friction/one-step diagnostics."""

import math
import mujoco
import numpy as np


class Moments:
    """Population mean/std over ALL contact samples, with unequal frame counts."""

    def __init__(self):
        self.count, self.mean, self.m2 = 0, 0.0, 0.0
        self.maximum = 0.0

    def add(self, values):
        values = np.asarray(values, dtype=float).reshape(-1)
        if not values.size:
            return
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite contact statistics")
        n = values.size
        mean = float(values.mean())
        delta = mean - self.mean
        total = self.count + n
        self.m2 += float(np.sum((values - mean) ** 2)) + delta ** 2 * self.count * n / total
        self.mean += delta * n / total
        self.count = total
        self.maximum = max(self.maximum, float(values.max()))

    def report(self, scale=1.0):
        return {"count": self.count,
                "mean": self.mean * scale if self.count else None,
                "std": math.sqrt(max(0, self.m2 / self.count)) * scale if self.count else None,
                "max": self.maximum * scale if self.count else None}


def body_velocities(model, pose_data, qvel):
    """World angular and body-origin linear velocity at the SOLVE pose."""
    linear = np.zeros((model.nbody, 3))
    angular = np.zeros((model.nbody, 3))
    for body in range(1, model.nbody):
        joint = int(model.body_jntadr[body])
        if model.body_jntnum[body] == 0:
            continue
        adr = int(model.jnt_dofadr[joint])
        kind = model.jnt_type[joint]
        if model.body_jntnum[body] == 1 and kind == mujoco.mjtJoint.mjJNT_FREE:
            linear[body] = qvel[adr:adr + 3]
            angular[body] = pose_data.xmat[body].reshape(3, 3) @ qvel[adr + 3:adr + 6]
        elif model.body_jntnum[body] == 1 and kind == mujoco.mjtJoint.mjJNT_HINGE:
            angular[body] = pose_data.xaxis[joint] * qvel[adr]
            linear[body] = np.cross(angular[body], pose_data.xpos[body] - pose_data.xanchor[joint])
        else:
            jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, pose_data, jacp, jacr, body)
            linear[body], angular[body] = jacp @ qvel, jacr @ qvel
    return linear, angular


def contact_observations(model, pose_data, qvel_after, native):
    """Native forces with post-impulse velocity, expressed at pre-step pose.

    Forces are on geom2 from geom1; relative motion is geom2 minus geom1.
    No force from CPU mj_forward is substituted into GPU friction diagnostics.
    """
    geom = native["geom"].astype(int)
    count = len(geom)
    if not count:
        return {"depth": np.zeros(0), "relative_velocity": np.zeros((0, 6)),
                "power": np.zeros(0), "utilization": np.zeros((0, 3))}
    body = model.geom_bodyid[geom]
    linear, angular = body_velocities(model, pose_data, qvel_after)
    point = native["position"]
    velocities = linear[body] + np.cross(angular[body], point[:, None, :] - pose_data.xpos[body])
    relative_linear = velocities[:, 1] - velocities[:, 0]
    relative_angular = angular[body[:, 1]] - angular[body[:, 0]]
    frame = native["frame"]
    v = np.einsum("nij,nj->ni", frame, relative_linear)
    omega = np.einsum("nij,nj->ni", frame, relative_angular)
    relative = np.concatenate([v, omega], axis=1)
    force = native["force"]
    power = np.sum(force[:, 1:3] * v[:, 1:3], axis=1) + np.sum(force[:, 3:] * omega, axis=1)
    normal = np.maximum(force[:, 0], 1e-12)
    coefficients = np.maximum(native["friction"], 1e-12)
    utilization = np.column_stack([
        np.linalg.norm(force[:, 1:3] / coefficients[:, :2], axis=1) / normal,
        np.abs(force[:, 3]) / coefficients[:, 2] / normal,
        np.linalg.norm(force[:, 4:6] / coefficients[:, 3:5], axis=1) / normal,
    ])
    return {"depth": np.maximum(-native["gap"], 0), "relative_velocity": relative,
            "power": power, "utilization": utilization}


def motion_metrics(reference, prediction, mass, tolerance=1e-8):
    """Energy-weighted motion direction and magnitude; zero directions undefined."""
    def inner(a, b):
        return float(np.dot(a * mass, b) if mass.ndim == 1 else a @ mass @ b)
    refnorm = math.sqrt(max(0, inner(reference, reference)))
    prednorm = math.sqrt(max(0, inner(prediction, prediction)))
    error = math.sqrt(max(0, inner(prediction - reference, prediction - reference)))
    defined = refnorm > tolerance and prednorm > tolerance
    cosine = float(np.clip(inner(reference, prediction) / (refnorm * prednorm), -1, 1)) if defined else None
    return {"reference_norm": refnorm, "prediction_norm": prednorm, "weighted_error": error,
            "relative_error": error / refnorm if refnorm > tolerance else None,
            "cosine": cosine, "angle_deg": math.degrees(math.acos(cosine)) if defined else None,
            "same_direction": cosine > 0 if defined else None}
