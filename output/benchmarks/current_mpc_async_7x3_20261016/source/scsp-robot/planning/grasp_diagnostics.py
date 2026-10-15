"""Measured object wrenches and independently bounded grasp lift capacity."""
import numpy as np
from scipy.optimize import linprog

from planning.grasp_verification import contact_wrench_rays


def object_contact_wrenches(model, data, object_body, pad_geoms, hand_geoms=()):
    """Resolve mj_contactForce onto the object, about its free-body origin.

    MuJoCo returns the force on geom2; contact-frame axes are matrix rows.
    Support reactions are kept separate from forces supplied by the hand.
    """
    import mujoco
    pad_geoms, hand_geoms = set(pad_geoms), set(hand_geoms)
    origin = np.asarray(data.xpos[object_body])
    totals = {key: np.zeros(6) for key in ('pads', 'other_hand', 'support', 'all')}
    contacts = []
    for index, contact in enumerate(data.contact):
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        if (b1 == object_body) == (b2 == object_body):
            continue
        other, sign = (g1, 1.) if b2 == object_body else (g2, -1.)
        frame = np.asarray(contact.frame).reshape(3, 3)
        local = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, local)
        force = sign*frame.T@local[:3]
        moment = sign*frame.T@local[3:] + np.cross(contact.pos-origin, force)
        wrench = np.r_[force, moment]
        group = 'pads' if other in pad_geoms else 'other_hand' if other in hand_geoms else 'support'
        totals[group] += wrench
        totals['all'] += wrench
        contacts.append(dict(group=group, geom=other, point=np.asarray(contact.pos).copy(),
            normal=sign*frame[0], tangents=sign*frame[1:], normal_force=float(local[0]),
            force=force, wrench=wrench, mu=float(contact.friction[0]), gap=float(contact.dist)))
    return totals, contacts


def bounded_vertical_acceleration(points, normals, friction, mass_world, external_world,
                                  normal_limits, *, tangents=None):
    """Maximize upward acceleration with zero horizontal/angular acceleration.

    Limits bound each contact's normal force. With measured MuJoCo tangents,
    use its four-ray pyramid; otherwise use an inscribed Coulomb polygon.
    This is a contact-force capacity bound, not a hand-actuation certificate.
    """
    points, normals = np.asarray(points).reshape(-1, 3), np.asarray(normals).reshape(-1, 3)
    mu = np.broadcast_to(np.asarray(friction), (len(points),))
    limits = np.asarray(normal_limits).reshape(len(points))
    columns, groups = [], []
    for i, (point, normal) in enumerate(zip(points, normals)):
        if tangents is None:
            rays = contact_wrench_rays(point[None], normal[None], mu[i])
        else:
            t = np.asarray(tangents)[i]
            forces = normal+mu[i]*np.array([t[0], t[1], -t[0], -t[1]])
            rays = np.c_[forces, np.cross(point, forces)].T
        groups.append(rays.shape[1])
        columns.append(rays)
    w = np.hstack(columns) if columns else np.empty((6, 0))
    closure = False
    if w.shape[1] and np.linalg.matrix_rank(w) == 6:
        count = w.shape[1]
        span = linprog(np.r_[np.zeros(count), -1.],
            A_eq=np.vstack((np.c_[w, np.zeros(6)], np.r_[np.ones(count), 0.])),
            b_eq=np.r_[np.zeros(6), 1.], A_ub=np.c_[-np.eye(count), np.ones(count)],
            b_ub=np.zeros(count), bounds=(0., None), method='highs')
        closure = bool(span.success and span.x[-1] > 1e-7)
    constraints = np.zeros((len(points), w.shape[1]+1))
    offset = 0
    for i, count in enumerate(groups):
        constraints[i, offset:offset+count] = 1.
        offset += count
    # W*f + external = M*[0, 0, az, 0, 0, 0]. Allow free fall
    # as well as upward acceleration, avoiding a biased feasibility check.
    result = linprog(np.r_[np.zeros(w.shape[1]), -1.],
        A_eq=np.c_[w, -np.asarray(mass_world)[:, 2]], b_eq=-np.asarray(external_world),
        A_ub=constraints if len(points) else None, b_ub=limits if len(points) else None,
        bounds=[(0., None)]*w.shape[1]+[(None, None)], method='highs')
    return dict(feasible=bool(result.success),
        force_closure=closure,
        max_upward_acceleration=float(result.x[-1]) if result.success else None,
        contact_wrench=(w@result.x[:-1]).tolist() if result.success else None,
        cone='measured_four_ray_pyramid' if tangents is not None else 'inscribed_coulomb_polygon')
