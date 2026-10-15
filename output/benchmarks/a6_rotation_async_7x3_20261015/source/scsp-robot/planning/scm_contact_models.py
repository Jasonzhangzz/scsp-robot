"""Independent contact-model references for the MuJoCo SCM benchmark.

This module deliberately does not import or modify ``LambdaContactControlOptimizer``.
The optimizer supplies a contact Jacobian, an object inertia and a selected
robot wrench; all contact-model calculations used for the benchmark live here.
MuJoCo is imported lazily so the algebraic references remain unit-testable on
machines without the simulator installed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


_EPS = 1.0e-10


def _array(value, shape=None):
    out = np.asarray(value, dtype=np.float64)
    if shape is not None:
        out = out.reshape(shape)
    return out


def pyramid_matrix(mu: float) -> np.ndarray:
    """Map four non-negative pyramid multipliers to ``[fn, ft1, ft2]``."""
    mu = float(mu)
    return np.array(
        [[1.0, 1.0, 1.0, 1.0],
         [mu, 0.0, -mu, 0.0],
         [0.0, mu, 0.0, -mu]], dtype=np.float64)


def contact_jacobian(point_local) -> np.ndarray:
    """Object body-frame Jacobian ``[I, -skew(point)]``."""
    p = _array(point_local, (3,))
    return np.array(
        [[1.0, 0.0, 0.0, 0.0, p[2], -p[1]],
         [0.0, 1.0, 0.0, -p[2], 0.0, p[0]],
         [0.0, 0.0, 1.0, p[1], -p[0], 0.0]], dtype=np.float64)


def contact_wrench(point_local, normal, tangent1, tangent2, lam) -> np.ndarray:
    frame = np.column_stack((_array(normal, (3,)), _array(tangent1, (3,)),
                             _array(tangent2, (3,))))
    return contact_jacobian(point_local).T @ (frame @ _array(lam, (3,)))


def _active_rows(jacobian, block_size=4, tolerance=1.0e-12):
    jacobian = _array(jacobian)
    if jacobian.ndim != 2 or jacobian.shape[1] != 6:
        raise ValueError("environment Jacobian must have shape (n, 6)")
    rows = []
    blocks = []
    for start in range(0, jacobian.shape[0], block_size):
        stop = min(start + block_size, jacobian.shape[0])
        if stop - start != block_size:
            continue
        if np.linalg.norm(jacobian[start:stop]) > tolerance:
            rows.extend(range(start, stop))
            blocks.append((start, stop))
    return np.asarray(rows, dtype=np.int64), blocks


def _regularization(size: int, value: float) -> np.ndarray:
    value = max(float(value), 0.0)
    return value * np.eye(size, dtype=np.float64)


def _block_diagonal_part(matrix: np.ndarray, blocks) -> np.ndarray:
    out = np.zeros_like(matrix)
    for start, stop in blocks:
        out[start:stop, start:stop] = matrix[start:stop, start:stop]
    return out


def _solve_monotone_lcp(W, g, tolerance=1.0e-9, max_iter=None):
    """Solve a regularized monotone LCP with a small active-set method.

    The benchmark matrices are positive definite after regularization.  The
    active-set iterations are therefore the KKT solve of the equivalent
    strictly convex non-negative QP.  The returned residual is always checked
    by the caller; a non-converged result is never silently accepted.
    """
    W = _array(W)
    g = _array(g, (-1,))
    n = int(g.size)
    if W.shape != (n, n):
        raise ValueError("LCP matrix and vector dimensions do not match")
    W = 0.5 * (W + W.T)
    if max_iter is None:
        max_iter = max(32, 8 * n + 16)
    active = set()
    lam = np.zeros(n, dtype=np.float64)
    for _ in range(int(max_iter)):
        if active:
            ids = np.asarray(sorted(active), dtype=np.int64)
            try:
                sol = np.linalg.solve(W[np.ix_(ids, ids)], -g[ids])
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(W[np.ix_(ids, ids)], -g[ids], rcond=None)[0]
            lam[:] = 0.0
            lam[ids] = sol
        else:
            lam[:] = 0.0
        w = W @ lam + g
        negative = [(float(lam[i]), int(i)) for i in active if lam[i] < -tolerance]
        if negative:
            _, idx = min(negative)
            active.remove(idx)
            continue
        violated = [(float(w[i]), int(i)) for i in range(n)
                    if i not in active and w[i] < -tolerance]
        if violated:
            _, idx = min(violated)
            active.add(idx)
            continue
        lam = np.maximum(lam, 0.0)
        w = W @ lam + g
        residual = max(float(np.max(np.maximum(-lam, 0.0))),
                       float(np.max(np.maximum(-w, 0.0))),
                       float(np.max(np.abs(lam * w))) if n else 0.0)
        return lam, w, residual, True
    lam = np.maximum(lam, 0.0)
    w = W @ lam + g
    residual = max(float(np.max(np.maximum(-lam, 0.0))),
                   float(np.max(np.maximum(-w, 0.0))),
                   float(np.max(np.abs(lam * w))) if n else 0.0)
    return lam, w, residual, False


def _weighted_norm(vector, matrix):
    vector = _array(vector, (-1,))
    matrix = 0.5 * (_array(matrix) + _array(matrix).T)
    value = float(vector @ matrix @ vector)
    return math.sqrt(max(value, 0.0))


def motion_accuracy(inertia, reference, predicted, h=1.0) -> Dict[str, Any]:
    """Appendix-B direction and magnitude errors for velocity increments."""
    M = 0.5 * (_array(inertia) + _array(inertia).T)
    ref = _array(reference, (-1,))
    pred = _array(predicted, ref.shape)
    ref_norm = _weighted_norm(ref, M)
    pred_norm = _weighted_norm(pred, M)
    if ref_norm <= _EPS or pred_norm <= _EPS:
        cosine = None
        direction = None
    else:
        cosine = float(np.clip((ref @ M @ pred) / (ref_norm * pred_norm), -1.0, 1.0))
        direction = float(math.acos(cosine))
    return {
        "direction_error": direction,
        "cos_theta": cosine,
        "magnitude_error": abs(ref_norm - pred_norm),
        "relative_magnitude_error": abs(ref_norm - pred_norm) / max(ref_norm, _EPS),
        "v_ref_norm": ref_norm,
        "v_pred_norm": pred_norm,
        "h": float(h),
    }


def lcp_mujoco_motion_accuracy(inertia, reference, predicted, h=1.0):
    """Compatibility wrapper used by the original test script."""
    return motion_accuracy(inertia, reference, predicted, h)


def appendix_accuracy(inertia, q_inv, jacobian, v_free, lam_hat, v_hat, h,
                      lam_ref=None, regularization=0.0,
                      coupling_scale=1.0) -> Dict[str, Any]:
    """Compute the complete local metrics from Appendix B."""
    J = _array(jacobian)
    q_inv = _array(q_inv)
    vf = _array(v_free, (-1,))
    lh = _array(lam_hat, (-1,))
    rows, blocks = _active_rows(J)
    if rows.size:
        Ja = J[rows]
        lam_active = lh[rows]
        W = Ja @ q_inv @ Ja.T
    else:
        Ja = np.zeros((0, 6), dtype=np.float64)
        lam_active = np.zeros(0, dtype=np.float64)
        W = np.zeros((0, 0), dtype=np.float64)
    if W.size:
        W = 0.5 * (W + W.T) + _regularization(W.shape[0], regularization)
        D = np.diag(np.diag(W))
        E = float(coupling_scale) * (W - D)
        D_safe = D + _regularization(D.shape[0], max(regularization, 1.0e-12))
        d_sqrt = np.sqrt(np.maximum(np.diag(D_safe), _EPS))
        d_invhalf = np.diag(1.0 / d_sqrt)
        loading = lam_active
        if lam_ref is not None:
            ref_full = _array(lam_ref, (-1,))
            if ref_full.size == J.shape[0] or ref_full.size == rows.size:
                loading = ref_full[rows] if ref_full.size == J.shape[0] else ref_full
        coupling_vec = d_invhalf @ (E @ loading)
        coupling_norm = float(np.linalg.norm(coupling_vec))
        m = 0.5 * (_array(inertia) + _array(inertia).T)
        evals, evecs = np.linalg.eigh(m)
        m_invhalf = (evecs * (1.0 / np.sqrt(np.maximum(evals, _EPS)))) @ evecs.T
        gamma = float(np.linalg.norm(m_invhalf @ Ja.T @ d_invhalf, 2))
        lambda_gap = (None if lam_ref is None else
                      _weighted_norm(lh[rows] - _array(lam_ref, (-1,))[rows], D_safe))
        regularization_term = float(lam_active @ (_regularization(W.shape[0], regularization) @ lam_active))
    else:
        D = E = np.zeros((0, 0), dtype=np.float64)
        coupling_norm = 0.0
        gamma = 0.0
        lambda_gap = None if lam_ref is None else 0.0
        regularization_term = 0.0
    motion = motion_accuracy(inertia, vf, _array(v_hat, (-1,)), h)
    motion.update({
        "v_norm": motion["v_ref_norm"],
        "v_hat_norm": motion["v_pred_norm"],
        "lambda_gap_D": lambda_gap,
        "regularization_term": regularization_term,
        "gamma_env": gamma,
        "coupling_norm": coupling_norm,
        "coupling_relative": (float(np.linalg.norm(E)) /
                               max(float(np.linalg.norm(D)), _EPS)),
        "active_rows": int(rows.size),
        "active_contacts": int(len(blocks)),
        "W_norm": float(np.linalg.norm(W)),
        "D_norm": float(np.linalg.norm(D)),
    })
    return motion


class LCPContactModel:
    """Rigid non-negative pyramid LCP reference."""

    def __init__(self, regularization=1.0e-8, tolerance=1.0e-8):
        self.regularization = float(regularization)
        self.tolerance = float(tolerance)
        self.last = None

    def respond(self, v_free, jacobian, q_inv, phi=0.0, coupling_scale=1.0,
                regularization=None):
        v_free = _array(v_free, (-1,))
        Jfull = _array(jacobian)
        q_inv = _array(q_inv)
        rows, blocks = _active_rows(Jfull)
        lam_full = np.zeros(Jfull.shape[0], dtype=np.float64)
        if rows.size == 0:
            self.last = {"lambda": lam_full, "residual": 0.0, "converged": True,
                         "W": np.zeros((0, 0)), "g": np.zeros(0), "rows": rows}
            return lam_full, v_free.copy()
        J = Jfull[rows]
        phi_vec = np.zeros(rows.size, dtype=np.float64)
        if np.ndim(phi) == 0:
            phi_vec.fill(float(phi))
        else:
            phi_arr = _array(phi, (-1,))
            phi_vec[:] = phi_arr[rows] if phi_arr.size == Jfull.shape[0] else phi_arr[:rows.size]
        reg = self.regularization if regularization is None else float(regularization)
        W = J @ q_inv @ J.T + _regularization(rows.size, reg)
        D = np.diag(np.diag(W))
        W = D + float(coupling_scale) * (W - D)
        g = J @ v_free + phi_vec
        lam, slack, residual, converged = _solve_monotone_lcp(
            W, g, tolerance=self.tolerance)
        lam_full[rows] = lam
        v_plus = v_free + q_inv @ J.T @ lam
        self.last = {"lambda": lam_full, "lambda_active": lam, "slack": slack,
                     "residual": residual, "converged": converged, "W": W,
                     "g": g, "D": D, "E": W - D, "rows": rows,
                     "blocks": blocks}
        if not converged and residual > 100.0 * self.tolerance:
            raise FloatingPointError("LCP active-set residual %.3e" % residual)
        return lam_full, v_plus


def surrogate_response(q_inv, jacobian, b, phi=0.0, regularization=1.0e-8,
                       coupling_scale=0.0) -> Dict[str, Any]:
    """Evaluate the diagonal SCM without touching the optimizer instance."""
    q_inv = _array(q_inv)
    Jfull = _array(jacobian)
    b = _array(b, (-1,))
    rows, blocks = _active_rows(Jfull)
    lam_full = np.zeros(Jfull.shape[0], dtype=np.float64)
    v_free = q_inv @ b
    if rows.size == 0:
        return {"lambda_env": lam_full, "v_free": v_free, "v_plus": v_free.copy(),
                "W": np.zeros((0, 0)), "D": np.zeros((0, 0)), "E": np.zeros((0, 0)),
                "g": np.zeros(0), "rows": rows, "blocks": blocks,
                "regularization": float(regularization), "residual": 0.0}
    J = Jfull[rows]
    phi_vec = np.zeros(rows.size, dtype=np.float64)
    if np.ndim(phi) == 0:
        phi_vec.fill(float(phi))
    else:
        phi_arr = _array(phi, (-1,))
        phi_vec[:] = phi_arr[rows] if phi_arr.size == Jfull.shape[0] else phi_arr[:rows.size]
    W = 0.5 * (J @ q_inv @ J.T + (J @ q_inv @ J.T).T)
    W += _regularization(rows.size, regularization)
    D = np.diag(np.diag(W))
    E = float(coupling_scale) * (W - D)
    g = J @ v_free + phi_vec
    d = np.maximum(np.diag(D), _EPS)
    lam = np.maximum(-g / d, 0.0)
    lam_full[rows] = lam
    v_plus = v_free + q_inv @ J.T @ lam
    slack = (D + E) @ lam + g
    return {"lambda_env": lam_full, "lambda_active": lam, "v_free": v_free,
            "v_plus": v_plus, "W": D + E, "D": D, "E": E, "g": g,
            "slack": slack, "rows": rows, "blocks": blocks,
            "regularization": float(regularization),
            "residual": float(np.max(np.maximum(-slack, 0.0))) if slack.size else 0.0}


class MujocoForwardContactModel:
    """MuJoCo forward oracle for the environment contact response."""

    def __init__(self, env=None, friction=0.5, table_friction=None):
        self.env = env
        self.friction = float(friction)
        self.table_friction = (self.friction if table_friction is None
                               else float(table_friction))
        self.last = None

    def set_mujoco_env(self, env):
        self.env = env

    @staticmethod
    def _object_frame(data, body_id):
        return np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)

    @staticmethod
    def _object_free_dofs(model, body_id):
        """Return the six qvel/qacc indices belonging to the object's free joint."""
        for joint_id in range(int(model.njnt)):
            if int(model.jnt_bodyid[joint_id]) != int(body_id):
                continue
            # MuJoCo's free joint has six velocity degrees of freedom.
            if int(model.jnt_type[joint_id]) == 0:
                start = int(model.jnt_dofadr[joint_id])
                return np.arange(start, start + 6, dtype=np.int64)
        raise ValueError("object body has no free joint")

    @staticmethod
    def _contact_signature(model, data, obj_body_id, table_id):
        """Stable contact signature used only to identify contact transitions."""
        obj_geoms = {int(gid) for gid in range(int(model.ngeom))
                     if int(model.geom_bodyid[gid]) == int(obj_body_id)}
        signature = []
        for i in range(int(data.ncon)):
            con = data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if table_id not in (g1, g2) or not ({g1, g2} & obj_geoms):
                continue
            signature.append((min(g1, g2), max(g1, g2),
                              tuple(np.round(np.asarray(con.pos), 5))))
        return tuple(sorted(signature))

    def _table_contacts(self, model, data, obj_body_id, obj_geom_ids, table_id,
                        body_pos, body_rot, mu):
        rows = []
        wrench_body = np.zeros(6, dtype=np.float64)
        for i in range(int(data.ncon)):
            con = data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if table_id not in (g1, g2) or not ({g1, g2} & obj_geom_ids):
                continue
            force6 = np.zeros(6, dtype=np.float64)
            try:
                import mujoco
                mujoco.mj_contactForce(model, data, i, force6)
            except Exception:
                continue
            frame = np.asarray(con.frame, dtype=np.float64).reshape(3, 3)
            force_world = frame @ force6[:3]
            # mj_contactForce returns the force on geom1.
            force_contact = force6[:3].copy()
            if g2 in obj_geom_ids:
                force_world = -force_world
                force_contact = -force_contact
            force_local = body_rot.T @ force_world
            point_body = body_rot.T @ (np.asarray(con.pos) - body_pos)
            wrench_body[:3] += force_local
            wrench_body[3:] += np.cross(point_body, force_local)
            fn = max(float(force_contact[0]), 0.0)
            A = pyramid_matrix(mu)
            try:
                from scipy.optimize import nnls
                multiplier, _ = nnls(A, np.array(
                    [fn, force_contact[1], force_contact[2]]))
            except Exception:
                multiplier = np.maximum(np.linalg.lstsq(A, np.array(
                    [fn, force_contact[1], force_contact[2]]), rcond=None)[0], 0.0)
            rows.extend(multiplier.tolist())
        return np.asarray(rows, dtype=np.float64), wrench_body

    def forward_velocity(self, optimizer, lam_r, point_local, normal, tangent1,
                         tangent2, h, tau_body=None, lambda_scale=1.0,
                         execution_dt=None, nstep=None, force_cap=0.0,
                         torque_cap=0.0):
        if self.env is None:
            raise ValueError("MuJoCo environment has not been set")
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("MuJoCo is required for the forward oracle") from exc
        env = self.env
        model, data = env.model_, env.data_
        qpos = data.qpos.copy(); qvel = data.qvel.copy()
        xfrc = data.xfrc_applied.copy(); qfrc = data.qfrc_applied.copy()
        ctrl = data.ctrl.copy()
        try:
            obj_body = int(model.geom("obj").bodyid)
            object_dofs = self._object_free_dofs(model, obj_body)
            execution_dt = (float(execution_dt) if execution_dt is not None else
                            float(h))
            if execution_dt <= 0.0:
                raise ValueError("execution_dt must be positive")
            nstep = max(1, int(nstep if nstep is not None else
                               getattr(env.param_, "frame_skip_", 1)))
            table_id = int(model.geom("table").id)
            contacts_before = self._contact_signature(
                model, data, obj_body, table_id)
            body_rot = self._object_frame(data, obj_body)
            wrench_value = contact_wrench(
                point_local, normal, tangent1, tangent2, _array(lam_r, (3,)))
            wrench_value[:3] *= float(lambda_scale)
            wrench_value[3:] *= float(lambda_scale)
            # The reduced optimizer uses an impulse-like lambda by default.
            # Convert it using the actual MuJoCo execution interval so that
            # one mj_step applies the same impulse represented by the model.
            if bool(getattr(optimizer, "wrench_is_force", False)):
                force_body = wrench_value.copy()
                wrench_impulse = wrench_value * execution_dt
            else:
                wrench_impulse = wrench_value.copy()
                force_body = wrench_impulse / execution_dt
            force_body_unclipped = force_body.copy()
            force_norm = float(np.linalg.norm(force_body[:3]))
            if float(force_cap) > 0.0 and force_norm > float(force_cap):
                force_body *= float(force_cap) / force_norm
                wrench_impulse = force_body * execution_dt
            torque_norm = float(np.linalg.norm(force_body[3:]))
            if float(torque_cap) > 0.0 and torque_norm > float(torque_cap):
                force_body[3:] *= float(torque_cap) / torque_norm
                wrench_impulse = force_body * execution_dt
            force_world = body_rot @ force_body[:3]
            torque_world = body_rot @ force_body[3:]
            data.xfrc_applied[:] = 0.0
            data.xfrc_applied[obj_body, :3] = force_world
            data.xfrc_applied[obj_body, 3:] = torque_world
            # Match the physical wrench rollout: the fingertip is an ideal
            # position-controlled actuator and must not fall into the table
            # while the object wrench is being measured.
            tip_body = getattr(env, "fingertip_body_id", None)
            tip_mass = getattr(env, "fingertip_mass", 0.0)
            gravity_vec = getattr(env, "gravity_vec", np.zeros(3))
            if tip_body is not None and float(tip_mass) > 0.0:
                data.xfrc_applied[int(tip_body), :3] = (
                    -float(tip_mass) * np.asarray(gravity_vec))
            if tau_body is not None:
                tau = _array(tau_body, (6,))
                data.xfrc_applied[obj_body, :3] += body_rot @ tau[:3]
                data.xfrc_applied[obj_body, 3:] += body_rot @ tau[3:]
            mujoco.mj_step(model, data, nstep=nstep)
            qvel_after = np.asarray(data.qvel[object_dofs], dtype=np.float64).copy()
            qvel_before = np.asarray(qvel[object_dofs], dtype=np.float64).copy()
            delta_world = qvel_after - qvel_before
            v_plus = np.hstack((body_rot.T @ delta_world[:3],
                                body_rot.T @ delta_world[3:]))
            obj_geom_ids = {int(g) for g in range(model.ngeom)
                            if int(model.geom_bodyid[g]) == obj_body}
            measured_lam_force, measured_wrench = self._table_contacts(
                model, data, obj_body, obj_geom_ids, table_id,
                np.asarray(data.xpos[obj_body]), body_rot, self.table_friction)
            measured_lam = measured_lam_force.copy()
            if not bool(getattr(optimizer, "wrench_is_force", False)):
                measured_lam *= execution_dt
            contacts_after = self._contact_signature(
                model, data, obj_body, table_id)
            self.last = {"v_plus": v_plus, "lambda_env": measured_lam,
                         "lambda_env_force": measured_lam_force,
                         "env_wrench_body": measured_wrench,
                         "qvel_before": qvel_before,
                         "qvel_after": qvel_after,
                         "delta_v_world": delta_world,
                         "wrench_impulse_body": wrench_impulse,
                         "force_body_unclipped": force_body_unclipped,
                         "force_body": force_body,
                         "force_cap": float(force_cap),
                         "torque_cap": float(torque_cap),
                         "force_was_clipped": bool(
                             not np.allclose(force_body, force_body_unclipped)),
                         "execution_dt": execution_dt,
                         "nstep": nstep,
                         "ncon": int(data.ncon), "oracle": "mj_step"}
            self.last["ncon_before"] = int(len(contacts_before))
            self.last["contact_transition"] = contacts_before != contacts_after
            self.last["contacts_before"] = contacts_before
            self.last["contacts_after"] = contacts_after
            return v_plus
        finally:
            data.qpos[:] = qpos; data.qvel[:] = qvel
            data.xfrc_applied[:] = xfrc; data.qfrc_applied[:] = qfrc
            data.ctrl[:] = ctrl
            mujoco.mj_forward(model, data)


# Name retained for callers of the earlier prototype.
MujocoQPContactModel = MujocoForwardContactModel


def compare_lambda(predicted, reference, weight=None):
    predicted = _array(predicted, (-1,))
    reference = _array(reference, predicted.shape)
    residual = predicted - reference
    if weight is None:
        value = float(np.linalg.norm(residual))
    else:
        value = _weighted_norm(residual, weight)
    return {"lambda_env_error": value,
            "lambda_env_norm": float(np.linalg.norm(reference)),
            "lambda_env_relative_error": value / max(float(np.linalg.norm(reference)), _EPS)}
