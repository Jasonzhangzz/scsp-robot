"""Keep measured contact identity independent of sparse ranking samples."""
import numpy as np


def measured_patch_index(optimizer, point, outward_normal, nearest_idx,
                         radius=.03, normal_cos=.8):
    point = np.asarray(point, dtype=float).reshape(3)
    normal = np.asarray(outward_normal, dtype=float).reshape(3)
    normal /= max(np.linalg.norm(normal), 1e-12)
    anchor = getattr(optimizer, '_measured_patch_anchor', None)
    if anchor is not None:
        old_point, old_normal, old_idx = anchor
        # Compare with a fixed anchor, not the previous frame: sliding across
        # an entire object must eventually become a genuinely new patch.
        if (np.linalg.norm(point-old_point) <= radius
                and np.dot(normal, old_normal) >= normal_cos):
            return int(old_idx)
    optimizer._measured_patch_anchor = (point.copy(), normal.copy(), int(nearest_idx))
    return int(nearest_idx)
