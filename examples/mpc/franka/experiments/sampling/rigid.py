"""Rigid-object Anitescu LCS used by the Isaac sampling controller.

The matrix construction follows dairlib/solvers/lcs_factory.cc (Anitescu
branch). State order matches its jack plant: ee position, object quaternion
(wxyz), object origin, ee velocity, WORLD angular velocity, COM velocity.
Collision parts and inertial/material properties come from the Isaac scene.
"""

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .c3 import _extension
from .lcs import LCS

EE = slice(0, 3)
QUAT = slice(3, 7)
OBJ = slice(7, 10)
VEL = slice(10, 19)
NX = 19


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def unit_quat(q):
    q = np.asarray(q, dtype=float).reshape(4)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-10:
        raise ValueError("object quaternion must be finite and nonzero")
    return q / np.linalg.norm(q)


def rotation(q):
    return Rotation.from_quat(unit_quat(q)[[1, 2, 3, 0]]).as_matrix()


def orientation_error(q, target):
    return float(2 * np.arccos(np.clip(abs(unit_quat(q) @ unit_quat(target)), 0., 1.)))


def pack_state(tip, position, quaternion, tip_velocity=None, angular=None, linear=None):
    x = np.zeros(NX)
    x[EE], x[QUAT], x[OBJ] = tip, unit_quat(quaternion), position
    for slot, value in ((slice(10, 13), tip_velocity), (slice(13, 16), angular),
                        (slice(16, 19), linear)):
        if value is not None:
            x[slot] = value
    return x


@dataclass
class Contact:
    kind: str
    gap: float
    point: np.ndarray
    normal: np.ndarray
    mu: float


class RigidScene:
    def __init__(self, mesh_paths, mass, inertia, table_height, *, tip_radius=.01,
                 tip_mass=.1, mu_tip=.5, mu_table=.5, com=None, vertices=None):
        self.parts = [trimesh.load_mesh(str(path), process=False).convex_hull
                      for path in mesh_paths]
        if not self.parts:
            raise ValueError("sampling C3 requires the Isaac collision mesh")
        self.vertices = np.vstack([part.vertices for part in self.parts])
        self.mass = float(mass)
        self.inertia = np.asarray(inertia, dtype=float).reshape(3, 3)
        if self.mass <= 0 or np.linalg.eigvalsh(self.inertia).min() <= 0:
            raise ValueError("object mass and central inertia must be positive")
        self.com = np.zeros(3) if com is None else np.asarray(com, dtype=float)
        self.table_height = float(table_height)
        self.tip_radius, self.tip_mass = float(tip_radius), float(tip_mass)
        self.mu_tip, self.mu_table = float(mu_tip), float(mu_table)
        self.radius = float(np.linalg.norm(self.vertices, axis=1).max()) + self.tip_radius

    def surface(self, point_local, part):
        """Signed sphere/convex-part distance; retain the actual witness point."""
        mesh = self.parts[part]
        closest, distance, triangle = trimesh.proximity.closest_point_naive(
            mesh, np.asarray(point_local).reshape(1, 3))
        surface = closest[0]
        inside = np.max(np.einsum('ij,ij->i', mesh.face_normals,
                                 point_local - mesh.triangles_center)) <= 1e-9
        delta = np.asarray(point_local) - surface
        if distance[0] > 1e-9:
            normal = delta / distance[0] * (-1 if inside else 1)
        else:
            normal = mesh.face_normals[int(triangle[0])]
        return (-distance[0] if inside else distance[0]), surface, normal

    def contacts(self, x, max_support=4, max_finger=3, include_tip_table=True):
        R = rotation(x[QUAT])
        position, tip = x[OBJ], x[EE]
        local = R.T @ (tip - position)
        finger = []
        for index in range(len(self.parts)):
            distance, point, normal = self.surface(local, index)
            finger.append(Contact('finger', float(distance - self.tip_radius),
                                  position + R @ point, R @ normal, self.mu_tip))
        finger.sort(key=lambda c: c.gap)
        # Keep several potential contacts for a compound object, without
        # filling concave cavities with one global convex hull.
        contacts = finger[:min(max_finger, len(finger))]
        world = self.vertices @ R.T + position
        # Include vertices within a prospective support band, not only the
        # current minimum. These contacts can become the next tipping pivot.
        lowest = float(world[:, 2].min())
        cloud = world[world[:, 2] <= lowest + .012]
        selected = [int(np.argmin(cloud[:, 2]))]
        distances = np.linalg.norm(cloud[:, :2] - cloud[selected[0], :2], axis=1)
        while len(selected) < min(max_support, len(cloud)):
            index = int(np.argmax(distances))
            if distances[index] < .003:
                break
            selected.append(index)
            distances = np.minimum(distances, np.linalg.norm(cloud[:, :2] - cloud[index, :2], axis=1))
        for point in cloud[selected]:
            contacts.append(Contact('table', float(point[2] - self.table_height), point,
                                    np.array([0., 0., 1.]), self.mu_table))
        if include_tip_table:
            contacts.append(Contact('tip_table', float(tip[2] - self.table_height - self.tip_radius),
                                    tip - [0., 0., self.tip_radius], np.array([0., 0., 1.]), self.mu_table))
        return contacts

    def linearize(self, x, dt=.05, for_planning=False):
        x = np.asarray(x, dtype=float)
        R, q = rotation(x[QUAT]), unit_quat(x[QUAT])
        N = np.zeros((10, 9))
        N[:3, :3] = np.eye(3)
        N[3:7, 3:6] = .5 * np.vstack([-q[1:], q[0] * np.eye(3) - skew(q[1:])])
        N[7:10, 6:9] = np.eye(3)
        N[7:10, 3:6] = skew(R @ self.com)
        inertia_world = R @ self.inertia @ R.T
        inverse_mass = np.zeros((9, 9))
        inverse_mass[:3, :3] = np.eye(3) / self.tip_mass
        inverse_mass[3:6, 3:6] = np.linalg.inv(inertia_world)
        inverse_mass[6:9, 6:9] = np.eye(3) / self.mass
        actuation = np.zeros((9, 3))
        actuation[:3] = np.eye(3) / self.tip_mass
        acceleration = np.zeros(9)
        acceleration[6:9] = [0., 0., -9.81]
        omega = x[13:16]
        acceleration[3:6] = -np.linalg.solve(inertia_world, np.cross(omega, inertia_world @ omega))
        A = np.eye(NX)
        A[:10, 10:] = dt * N
        B = np.vstack([dt**2 * N @ actuation, dt * actuation])
        d = np.r_[dt**2 * N @ acceleration, dt * acceleration]
        normals, rays, gaps = [], [], []
        # Native jack resolves [0 EE-ground, 1 EE-object, 3 ground-object]
        # for optimization and uses more contacts for physical cost rollout.
        contacts = (self.contacts(x, max_support=3, max_finger=1, include_tip_table=False)
                    if for_planning else self.contacts(x))
        center = x[OBJ] + R @ self.com
        for contact in contacts:
            relative = np.zeros((3, 9))
            if contact.kind == 'finger':
                relative[:, :3] = np.eye(3)
                relative[:, 3:6] = skew(contact.point - center)
                relative[:, 6:9] = -np.eye(3)
            elif contact.kind == 'table':
                relative[:, 3:6] = -skew(contact.point - center)
                relative[:, 6:9] = np.eye(3)
            else:
                relative[:, :3] = np.eye(3)
            n = contact.normal
            axis = np.eye(3)[int(np.argmin(abs(n)))]
            t1 = np.cross(n, axis)
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(n, t1)
            normal = n @ relative
            for tangent in (t1, t2, -t1, -t2):
                normals.append(normal)
                rays.append((n + contact.mu * tangent) @ relative)
                gaps.append(contact.gap)
        Jn, Jc = np.asarray(normals), np.asarray(rays)
        # dairlib Anitescu: gap/dt + Jc*v_next. Only NORMAL
        # displacement enters gap; tangents enter relative velocity.
        D = np.vstack([dt**2 * N @ inverse_mass @ Jc.T, dt * inverse_mass @ Jc.T])
        E = np.hstack([Jn @ np.linalg.pinv(N) / dt, Jc])
        F = dt * Jc @ inverse_mass @ Jc.T
        H = dt * Jc @ actuation
        c = np.asarray(gaps) / dt + dt * Jc @ acceleration - E[:, :10] @ x[:10]
        model = RigidLCS(A, B, D, d, E, F, H, c, dt)
        if for_planning:
            model.ee_floor_z = self.table_height+self.tip_radius+.002
        model.contacts, model.N, model.Jc, model.Jn = contacts, N, Jc, Jn
        return model


class RigidLCS(LCS):
    def resolve_force(self, state, control):
        affine = self.E @ state + self.H @ control + self.c
        force = np.asarray(_extension().solve_qp(
            self.F, affine, np.eye(self.m), np.zeros(self.m))).reshape(-1)
        slack = self.F @ force + affine
        if np.min(slack) < -1e-5 or abs(force @ slack) > 1e-4 * (1 + np.linalg.norm(force)):
            raise RuntimeError("rigid contact LCP did not satisfy complementarity")
        return force

    def simulate(self, state, control):
        nxt, force = super().simulate(state, control)
        nxt[QUAT] = unit_quat(nxt[QUAT])
        return nxt, force


def quaternion_angle_hessian(current, target):
    """Hessian of (2*acos(dot(q,r)/norm(q)))**2, as in dairlib's cost.

    The closest quaternion representative is selected before differentiating.
    The closed form below avoids the long componentwise symbolic expansion.
    """
    q, r = unit_quat(current), unit_quat(target)
    if q @ r < 0:
        r = -r
    s = float(np.clip(q @ r, 0., 1.))
    if s > 1.-1e-7:
        return 8 * (np.eye(4)-np.outer(r, r))
    angle, sine = np.arccos(s), np.sqrt(1.-s*s)
    first = -8*angle/sine
    second = 8/(sine*sine)-8*s*angle/(sine**3)
    derivative = r-s*q
    curvature = -np.outer(r, q)-np.outer(q, r)-s*np.eye(4)+3*s*np.outer(q, q)
    return second*np.outer(derivative, derivative)+first*curvature


def stage_costs(horizon, contacts, target_quaternion, *, pose_weight=2500., current_quaternion=None):
    """Sign-invariant orientation loss, with the jack pose/velocity weights."""
    q = np.zeros(NX)
    q[EE], q[OBJ] = .45, 45 * np.array([200., 200., 120.])
    q[10:13], q[13:16], q[16:19] = 225., 2.25, 2.25
    Q = np.diag(q)
    target = unit_quat(target_quaternion)
    hessian = quaternion_angle_hessian(target if current_quaternion is None else current_quaternion, target)
    # Match native jack's minimum-eigenvalue regularization to keep the
    # quaternion-dependent Q positive semidefinite.
    hessian += max(0., -np.linalg.eigvalsh(hessian).min()) * np.eye(4)
    Q[QUAT, QUAT] = pose_weight*hessian
    width = NX + contacts + 3
    gx = np.r_[np.full(3, 800.), np.full(4, 100.), np.full(3, 10.), np.full(9, .1)]
    g = .25 * np.r_[gx, np.full(contacts, .005), np.full(3, 30.)]
    ux = np.r_[np.full(3, 10.), np.full(7, 50.), np.full(3, 8.), np.ones(6)]
    u = .26 * np.r_[ux, np.full(contacts, 10.), np.full(3, .01)]
    assert len(g) == width
    return ([Q.copy() for _ in range(horizon + 1)],
            [.01 * np.eye(3) for _ in range(horizon)],
            [np.diag(g) for _ in range(horizon)],
            [np.diag(u) for _ in range(horizon)])
