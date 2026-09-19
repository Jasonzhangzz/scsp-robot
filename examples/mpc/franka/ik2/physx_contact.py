"""MuJoCo-style signed contact gap from PhysX rigid contacts.

Isaac Gym's RigidContact has no ``separation``.  ``initial_overlap`` is
positive for penetration and 0 for both a made touch and the 1 mm
``contact_offset`` pair.  Reconstruct the fingertip-sphere signed gap so
``dist <= 0`` means physical contact, matching
``contact/fingertips_collision_detection2.py``.
"""

from __future__ import annotations

import numpy as np


DEFAULT_FINGERTIP_RADIUS = 0.01
DEFAULT_CONTACT_OFFSET = 0.001


def _contact_names(c):
    return getattr(getattr(c, "dtype", None), "names", None) or ()


def _contact_value(c, *names, default=None):
    fields = _contact_names(c)
    for name in names:
        if name in fields:
            return c[name]
    return default


def contact_overlap(c):
    raw = _contact_value(c, "initialOverlap", "initial_overlap", default=0.0)
    return float(raw if raw is not None else 0.0)


def contact_impulse(c):
    raw = _contact_value(c, "lambda", default=0.0)
    return float(raw if raw is not None else 0.0)


def contact_min_dist(c):
    raw = _contact_value(c, "minDist", "separation", "distance", default=None)
    if raw is None:
        return None
    return float(raw)


def contact_local_pos(c, index):
    if int(index) == 0:
        return _contact_value(c, "localPos0", "local_pos0", default=None)
    return _contact_value(c, "localPos1", "local_pos1", default=None)


def fingertip_body_index(simulator):
    mapping = getattr(simulator, "franka_body_name_to_index", None) or {}
    idx = mapping.get("fingertip")
    if idx is not None:
        return int(idx)
    tip = getattr(simulator, "tip_body_idx", None)
    return None if tip is None else int(tip)


def fingertip_radius(simulator, default=DEFAULT_FINGERTIP_RADIUS):
    radius = getattr(simulator, "fingertip_radius", None)
    if radius is not None:
        return float(radius)
    param = getattr(simulator, "param_", None)
    return float(getattr(param, "fts_radius_", default) or default)


def end_effector_position(simulator, fallback=None):
    getter = getattr(simulator, "get_end_effector_pos", None)
    if getter is not None:
        pose = getter()
        if isinstance(pose, tuple):
            return np.asarray(pose[0], dtype=np.float64).reshape(3)
        return np.asarray(pose, dtype=np.float64).reshape(3)
    if fallback is None:
        return None
    return np.asarray(fallback, dtype=np.float64).reshape(3)


def physx_signed_gap(overlap, cpos, tip_pos, normal_obj_to_other, radius,
                     fallback=DEFAULT_CONTACT_OFFSET):
    """Signed surface gap.  Negative is penetration (MuJoCo ``contact.dist``).

    Prefer ``||tip - cpos|| - radius``.  The normal-axis formula can report
    several centimetres of fake penetration when PhysX ``pos`` is missing and
    the caller invented a point on the far side of the object.
    """
    del fallback
    overlap = float(overlap or 0.0)
    if overlap > 1e-8:
        return -overlap
    if cpos is None or tip_pos is None:
        return float("inf")
    tip = np.asarray(tip_pos, dtype=np.float64).reshape(3)
    pos = np.asarray(cpos, dtype=np.float64).reshape(3)
    radial = float(np.linalg.norm(tip - pos)) - float(radius)
    normal = np.asarray(normal_obj_to_other, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm > 1e-8:
        axial = float(np.dot(tip - pos, normal / norm)) - float(radius)
        # Keep the radial gap; only use the axis value when it agrees that
        # the point is on the near side of the sphere.
        if axial > 0.0 and radial > 0.0:
            return min(radial, axial)
    return radial
