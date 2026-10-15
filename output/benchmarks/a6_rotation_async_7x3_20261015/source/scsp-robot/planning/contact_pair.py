"""Contact-pair data shared by lambda grasp selection and physical MPPI.

The lambda optimizer historically returned a loose dictionary with several
different force-vector names.  This module gives the receding-horizon planner
one small, validated representation while keeping conversion from the legacy
dictionary at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


def _unit(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-10:
        return np.asarray(fallback, dtype=np.float64).reshape(3).copy()
    return value / norm


def _tangent_basis(normal: np.ndarray, tangent_hint: Optional[np.ndarray] = None) -> np.ndarray:
    normal = _unit(normal, np.array([0.0, 0.0, 1.0]))
    if tangent_hint is None:
        ref = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        tangent_hint = np.cross(normal, ref)
    tangent = np.asarray(tangent_hint, dtype=np.float64).reshape(3)
    tangent = tangent - np.dot(tangent, normal) * normal
    tangent = _unit(tangent, np.array([1.0, 0.0, 0.0]))
    bitangent = _unit(np.cross(normal, tangent), np.array([0.0, 1.0, 0.0]))
    tangent = _unit(np.cross(bitangent, normal), tangent)
    return np.stack((tangent, bitangent), axis=-1)


@dataclass(frozen=True)
class ContactPair:
    """Two object-frame contacts and the force-closure data from lambda.

    ``normals_local`` follows the convention used by ``mlqp_point_v2`` (the
    inward object-surface normal).  The physical EE target is consequently
    formed with ``-normal`` plus the configured fingertip offset.
    """

    contact_points_local: np.ndarray
    normals_local: np.ndarray
    tangent_basis_local: np.ndarray
    witness_force_local: np.ndarray
    desired_force_local: np.ndarray
    score: float = float("inf")
    feasible: bool = True

    def __post_init__(self):
        points = np.asarray(self.contact_points_local, dtype=np.float64).reshape(2, 3)
        normals = np.asarray(self.normals_local, dtype=np.float64).reshape(2, 3)
        tangents = np.asarray(self.tangent_basis_local, dtype=np.float64)
        if tangents.size == 0:
            tangents = np.stack([_tangent_basis(n) for n in normals], axis=0)
        tangents = tangents.reshape(2, 3, 2)
        normals = np.stack([_unit(n, np.array([0.0, 0.0, 1.0])) for n in normals], axis=0)
        # Re-orthogonalize user supplied tangent frames so numerical noise in
        # optimizer output cannot bias the world-frame projection.
        tangents = np.stack([_tangent_basis(normals[i], tangents[i, :, 0]) for i in range(2)], axis=0)

        witness = np.asarray(self.witness_force_local, dtype=np.float64)
        desired = np.asarray(self.desired_force_local, dtype=np.float64)
        witness = np.zeros((2, 3), dtype=np.float64) if witness.size == 0 else witness.reshape(2, 3)
        desired = witness.copy() if desired.size == 0 else desired.reshape(2, 3)
        object.__setattr__(self, "contact_points_local", points)
        object.__setattr__(self, "normals_local", normals)
        object.__setattr__(self, "tangent_basis_local", tangents)
        object.__setattr__(self, "witness_force_local", witness)
        object.__setattr__(self, "desired_force_local", desired)
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "feasible", bool(self.feasible))

    @classmethod
    def from_lambda_targets(cls, targets: dict) -> "ContactPair":
        """Convert the dictionary returned by ``_get_live_contact_targets``."""
        points = targets.get("contact_points_local", targets.get("raw_contact_points_local"))
        normals = targets.get("normals_local", targets.get("raw_normals_local"))
        if points is None or normals is None:
            raise ValueError("lambda targets must contain two contact points and normals")
        witness = targets.get("witness_force_vectors_local", targets.get("witness_contact_forces_local", []))
        desired = targets.get("desired_force_vectors_local", targets.get("desired_contact_forces_local", witness))
        return cls(
            contact_points_local=points,
            normals_local=normals,
            tangent_basis_local=targets.get("tangent_basis_local", np.zeros((0,))),
            witness_force_local=witness,
            desired_force_local=desired,
            score=targets.get("total_cost", float("inf")),
            feasible=bool(targets.get("feasible", True)),
        )

    def project_world(self, object_pos, object_quat):
        """Project points, inward normals and tangent frames to world space."""
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        quat = np.asarray(object_quat, dtype=np.float64).reshape(4)
        quat = quat / max(float(np.linalg.norm(quat)), 1.0e-12)
        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        points = object_pos[None, :] + (rotation @ self.contact_points_local.T).T
        normals = (rotation @ self.normals_local.T).T
        tangents = np.einsum("ij,njk->nik", rotation, self.tangent_basis_local)
        witness = (rotation @ self.witness_force_local.T).T
        desired = (rotation @ self.desired_force_local.T).T
        return {
            "contact_points_world": points,
            "normals_world": normals,
            "tangent_basis_world": tangents,
            "witness_force_world": witness,
            "desired_force_world": desired,
        }
