"""Point-contact linear complementarity model.

The local model is the same object C3, MIQP, and the acados MPC all consume:
a 3D fingertip and a translating object, discretized with a fixed contact
normal. It replaces dairlib's Drake plant linearization. One normal contact
uses the Anitescu gap at the next position,

    0 <= lambda  perpendicular to  n·(p_ee+ - p_obj+) - radius.

The normal is frozen at the sampled fingertip location, which is the local
model used inside each sampling-C3 candidate.
"""

from __future__ import annotations

import numpy as np

NX = 12
NU = 3
EE = slice(0, 3)
OBJ = slice(3, 6)
EE_V = slice(6, 9)
OBJ_V = slice(9, 12)


class LCS:
    """Time-invariant linear complementarity system over one horizon."""

    def __init__(self, A, B, D, d, E, F, H, c, dt):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.D = np.asarray(D, dtype=float)
        self.d = np.asarray(d, dtype=float).reshape(-1)
        self.E = np.asarray(E, dtype=float)
        self.F = np.asarray(F, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.c = np.asarray(c, dtype=float).reshape(-1)
        self.dt = float(dt)
        self.n = int(self.A.shape[0])
        self.k = int(self.B.shape[1])
        self.m = int(self.D.shape[1])

    def step_affine(self, state, control, force):
        state = np.asarray(state, dtype=float).reshape(self.n)
        control = np.asarray(control, dtype=float).reshape(self.k)
        force = np.asarray(force, dtype=float).reshape(self.m)
        return self.A @ state + self.B @ control + self.D @ force + self.d

    def gap(self, state, control, force):
        state = np.asarray(state, dtype=float).reshape(self.n)
        control = np.asarray(control, dtype=float).reshape(self.k)
        force = np.asarray(force, dtype=float).reshape(self.m)
        return self.E @ state + self.F @ force + self.H @ control + self.c

    def resolve_force(self, state, control):
        """Exact mixed-integer resolution of the one-step LCP, mode by mode."""
        state = np.asarray(state, dtype=float).reshape(self.n)
        control = np.asarray(control, dtype=float).reshape(self.k)
        affine = self.E @ state + self.H @ control + self.c
        best = None
        for mask in range(1 << self.m):
            # Bit 1: separated, force is zero. Bit 0: sustained contact, gap is zero.
            rows = []
            rhs = []
            for index in range(self.m):
                row = np.zeros(self.m)
                if mask & (1 << index):
                    row[index] = 1.0
                    rows.append(row)
                    rhs.append(0.0)
                else:
                    rows.append(self.F[index].copy())
                    rhs.append(-affine[index])
            force = np.linalg.solve(np.vstack(rows), np.asarray(rhs))
            gap = affine + self.F @ force
            if np.all(force >= -1e-8) and np.all(gap >= -1e-8):
                complementarity = float(np.dot(np.maximum(force, 0.0), np.maximum(gap, 0.0)))
                if best is None or complementarity < best[0]:
                    best = (complementarity, np.maximum(force, 0.0))
        if best is None:
            raise RuntimeError("contact LCP has no complementary force")
        return best[1]

    def simulate(self, state, control):
        force = self.resolve_force(state, control)
        return self.step_affine(state, control, force), force

    def sequence(self, horizon):
        return {
            "A": [self.A.copy() for _ in range(horizon)],
            "B": [self.B.copy() for _ in range(horizon)],
            "D": [self.D.copy() for _ in range(horizon)],
            "d": [self.d.copy() for _ in range(horizon)],
            "E": [self.E.copy() for _ in range(horizon)],
            "F": [self.F.copy() for _ in range(horizon)],
            "H": [self.H.copy() for _ in range(horizon)],
            "c": [self.c.copy() for _ in range(horizon)],
        }


def unit_normal(direction):
    direction = np.asarray(direction, dtype=float).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return direction / norm


def point_push_lcs(normal, dt=0.05, ee_mass=0.4, object_mass=0.2, radius=0.05):
    """Fingertip/object LCS linearized about a unit contact normal."""
    normal = unit_normal(normal)
    eye = np.eye(3)
    A = np.eye(NX)
    A[EE, EE_V] = dt * eye
    A[OBJ, OBJ_V] = dt * eye
    B = np.zeros((NX, NU))
    B[EE_V, :] = (dt / ee_mass) * eye
    B[EE, :] = (dt ** 2 / ee_mass) * eye
    D = np.zeros((NX, 1))
    D[EE, 0] = (dt ** 2 / ee_mass) * normal
    D[OBJ, 0] = -(dt ** 2 / object_mass) * normal
    D[EE_V, 0] = (dt / ee_mass) * normal
    D[OBJ_V, 0] = -(dt / object_mass) * normal
    E = np.zeros((1, NX))
    E[0, EE] = normal
    E[0, OBJ] = -normal
    E[0, EE_V] = dt * normal
    E[0, OBJ_V] = -dt * normal
    F = np.array([[dt ** 2 * (1.0 / ee_mass + 1.0 / object_mass)]])
    H = (dt ** 2 / ee_mass) * normal.reshape(1, 3)
    return LCS(A, B, D, np.zeros(NX), E, F, H, np.array([-radius]), dt)


def pack_state(ee_position, object_position, ee_velocity=None, object_velocity=None):
    state = np.zeros(NX)
    state[EE] = np.asarray(ee_position, dtype=float).reshape(3)
    state[OBJ] = np.asarray(object_position, dtype=float).reshape(3)
    if ee_velocity is not None:
        state[EE_V] = np.asarray(ee_velocity, dtype=float).reshape(3)
    if object_velocity is not None:
        state[OBJ_V] = np.asarray(object_velocity, dtype=float).reshape(3)
    return state


def desired_trajectory(ee_position, object_position, horizon):
    target = pack_state(ee_position, object_position)
    return [target.copy() for _ in range(horizon + 1)]


def stage_costs(horizon, object_weight=9000.0, ee_weight=0.45, velocity_weight=5.0,
                input_weight=0.01, consensus_weight=0.25, projection_weight=1.0):
    q = np.zeros(NX)
    q[EE] = ee_weight
    q[OBJ] = object_weight
    q[EE_V] = velocity_weight
    q[OBJ_V] = velocity_weight
    Q = [np.diag(q) for _ in range(horizon + 1)]
    R = [input_weight * np.eye(NU) for _ in range(horizon)]
    width = NX + 1 + NU
    G = [consensus_weight * np.eye(width) for _ in range(horizon)]
    U = [projection_weight * np.eye(width) for _ in range(horizon)]
    return Q, R, G, U
