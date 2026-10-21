"""Historical diagnostics. Never imported or installed by DyWA evaluation.

These functions are retained only for old numerical regression tests.
"""
import sys
import numpy as np
from planning.dywa.runtime import look_at_rotation_and_translation

def _alignment_types():
    from collections import namedtuple

    return (
        namedtuple("SimilarityTransform", "R T s"),
        namedtuple("ICPSolution", "converged rmse Xt RTs t_history"),
    )


def _iterative_closest_point(X, Y, init_transform=None, max_iterations=100,
                             relative_rmse_thr=1e-6, estimate_scale=False,
                             allow_reflection=False, verbose=False):
    """Rigid ICP with PyTorch3D's row-vector transform convention, X @ R + T."""
    import torch

    if estimate_scale:
        raise NotImplementedError("DyWA's rigid ICP shim does not estimate scale")
    if int(max_iterations) <= 0:
        raise ValueError("ICP max_iterations must be positive")
    SimilarityTransform, ICPSolution = _alignment_types()
    batch = X.shape[0]
    rotation = torch.eye(3, device=X.device, dtype=X.dtype).expand(batch, 3, 3).clone()
    translation = torch.zeros(batch, 3, device=X.device, dtype=X.dtype)
    scale = torch.ones(batch, device=X.device, dtype=X.dtype)
    if init_transform is not None:
        rotation = init_transform.R.clone()
        translation = init_transform.T.clone()
        scale = init_transform.s.clone()
    current = scale[:, None, None] * (X @ rotation) + translation[:, None, :]
    previous = None
    converged = False
    history = []
    for _ in range(int(max_iterations)):
        neighbor = torch.cdist(current, Y).argmin(dim=-1)
        matched = Y.gather(1, neighbor[..., None].expand(-1, -1, Y.shape[-1]))
        source_mean = current.mean(dim=1)
        target_mean = matched.mean(dim=1)
        covariance = (current - source_mean[:, None]).transpose(1, 2) @ (matched - target_mean[:, None])
        left, _, right = torch.linalg.svd(covariance)
        right = right.clone()
        if not allow_reflection:
            reflection = torch.det(left @ right) < 0
            right[reflection, -1] *= -1
        delta = left @ right
        shift = target_mean - (source_mean[:, None, :] @ delta).squeeze(1)
        current = current @ delta + shift[:, None, :]
        rotation = rotation @ delta
        translation = (translation[:, None, :] @ delta).squeeze(1) + shift
        rmse = (current - matched).square().sum(dim=-1).mean(dim=-1).sqrt()
        history.append(SimilarityTransform(rotation.clone(), translation.clone(), scale.clone()))
        if previous is not None:
            improvement = (previous - rmse) / previous.clamp_min(1e-12)
            converged = bool(torch.all((improvement >= 0) & (improvement <= relative_rmse_thr)).item())
            converged = converged or bool(torch.all(rmse < 1e-12).item())
            if converged:
                break
        previous = rmse.detach()
    return ICPSolution(
        converged, rmse, current, SimilarityTransform(rotation, translation, scale), history
    )


