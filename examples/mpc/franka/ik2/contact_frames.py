"""Isaac / MuJoCo contact-frame helpers that do not import Isaac Gym."""

import numpy as np


def contact_jacobian_body_frame(jacobian, body_mat):
    """Right-multiply object columns by diag(R, R) into the body frame."""
    jacobian = np.asarray(jacobian, dtype=np.float64).copy()
    if jacobian.ndim != 2 or jacobian.shape[1] < 6:
        return jacobian
    rot = np.asarray(body_mat, dtype=np.float64).reshape(3, 3)
    frame = np.zeros((6, 6), dtype=np.float64)
    frame[:3, :3] = rot
    frame[3:, 3:] = rot
    jacobian[:, :6] = jacobian[:, :6] @ frame
    return jacobian


def contact_aware_task_force(force_track, force_contact, contact_n_outward):
    """Keep arm tracking; cap only the inward press at the MuJoCo scale.

    ``force_track`` is the task-mass force that finishes a 5 mm increment.
    ``force_contact`` is K e - D v (~0.5 N for 5 mm).  ``contact_n_outward``
    points object → fingertip.  Tangential / lifting components stay on
    ``force_track`` so a diagonal flip still has a +Z increment.
    """
    force_track = np.asarray(force_track, dtype=np.float32).reshape(3).copy()
    if contact_n_outward is None:
        return force_track
    n = np.asarray(contact_n_outward, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(n))
    if nrm < 1e-8:
        return force_track
    n = n / nrm
    inward = -n
    f_in_track = float(np.dot(force_track, inward))
    f_in_contact = float(np.dot(np.asarray(force_contact, dtype=np.float64).reshape(3), inward))
    cap = max(f_in_contact, 0.0)
    if f_in_track > cap:
        force_track = force_track - np.float32(f_in_track - cap) * inward.astype(np.float32)
    return force_track


def planar_table_support_local(obj_pos, r_obj_to_world, table_height):
    """Body-frame point under the COM on the world table plane.

    Do not hardcode [0, 0, -0.025] in the mesh frame: foam_brick / piggy
    source STLs are X-up, so the table is -X, not -Z.
    """
    obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3)
    rot = np.asarray(r_obj_to_world, dtype=np.float64).reshape(3, 3)
    support_world = np.array([obj_pos[0], obj_pos[1], float(table_height)], dtype=np.float64)
    return (rot.T @ (support_world - obj_pos)).astype(np.float32)


def _skew(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def _contact_jacobian_np(n, t1, t2, j_rel, mu):
    con_frame = np.stack([n, t1, t2], axis=1)
    con_frame_pmd = np.concatenate([con_frame, -con_frame[:, 1:]], axis=1)
    con_jacp = con_frame_pmd.T @ j_rel
    return con_jacp[0] + float(mu) * con_jacp[1:]


def planar_table_jacobians(obj_pos, r_obj_to_world, table_height, nv, mu, skew_fn=None):
    """One world-up table plane at the COM projection, plus its body-frame J."""
    if skew_fn is None:
        skew_fn = _skew
    obj_pos = np.asarray(obj_pos, dtype=np.float32).reshape(3)
    support_world = np.array([obj_pos[0], obj_pos[1], float(table_height)], dtype=np.float32)
    r_obj = support_world - obj_pos
    j_rel = np.zeros((3, int(nv)), dtype=np.float32)
    j_rel[:, 0:3] = np.eye(3, dtype=np.float32)
    j_rel[:, 3:6] = -np.asarray(skew_fn(r_obj), dtype=np.float32)
    n_t = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    t1_t = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t2_t = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    con_jac = np.asarray(_contact_jacobian_np(n_t, t1_t, t2_t, j_rel, mu), dtype=np.float32)
    con_jac_body = contact_jacobian_body_frame(con_jac[:, :6], r_obj_to_world)
    return con_jac, con_jac_body, planar_table_support_local(obj_pos, r_obj_to_world, table_height)


def _quat_wxyz_to_R(quat_wxyz):
    qw, qx, qy, qz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    nrm = float(np.linalg.norm([qw, qx, qy, qz]))
    if nrm < 1e-9:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / nrm, qx / nrm, qy / nrm, qz / nrm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def table_jac_mat_env(obj_pos, obj_quat_wxyz, table_height, nv=9, mu=0.5, max_ncon=10):
    """Planner-side table J from object pose.  Isaac does not need to send it."""
    rot = _quat_wxyz_to_R(obj_quat_wxyz)
    _, con_jac_body, _ = planar_table_jacobians(obj_pos, rot, table_height, nv, mu)
    jac = np.zeros((int(max_ncon) * 4, int(nv)), dtype=np.float64)
    jac[0:4, :6] = con_jac_body
    return jac
