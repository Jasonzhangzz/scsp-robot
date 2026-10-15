"""Propagate measured normal gaps within a short MPC prediction window.

Object translations use world coordinates; angular displacement is in the
observed object frame, matching the Jacobian supplied by Isaac. Friction
pyramid rows share a *normal* gap, never a tangential displacement bias.
"""


def rotation_cs(quaternion):
    import casadi as cs
    q = quaternion / cs.sqrt(cs.sumsqr(quaternion) + 1e-20)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return cs.vertcat(
        cs.horzcat(1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
        cs.horzcat(2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
        cs.horzcat(2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))


def attached_point_cs(state, reference_pose, point_world):
    """World target attached to the observed object, including sphere offset."""
    return state[:3] + rotation_cs(state[3:7]) @ (
        rotation_cs(reference_pose[3:7]).T @ (point_world-reference_pose[:3]))


def contact_geometry_cs(state, reference, phi, jacobian, *, support_point=None,
                        support_tangents=None, support_half_extents=None):
    """Refresh measured patch gaps AND moment arms inside the dynamics graph.

    Linear velocities are world-frame, angular velocities body-frame. Table
    directions stay in world axes; finger directions rotate with the object.
    The friction rows recover the body-fixed lever arm without extra IPC data.
    Padding remains inactive. This is short-horizon patch transport, not a
    collision query or an acceptance test on a solved trajectory.
    """
    import casadi as cs
    rotation, observed = rotation_cs(state[3:7]), rotation_cs(reference[3:7])
    gaps, rows = [], []
    for start in range(0, int(jacobian.size1()), 4):
        block = jacobian[start:start+4, :]
        normal = cs.sum1(block)/4
        tangent = (block[0, :]-block[2, :])/2
        ns = cs.sqrt(cs.sumsqr(normal[:3]) + 1e-20)
        ts = cs.sqrt(cs.sumsqr(tangent[:3]) + 1e-20)
        n, an = observed.T @ normal[:3].T/ns, normal[3:6].T/ns
        t, at = observed.T @ tangent[:3].T/ts, tangent[3:6].T/ts
        lever = cs.cross(n, an) + n*cs.dot(cs.cross(t, at), n)
        finger = cs.sumsqr(block[:, 6:]) > 0
        transported = block[:, :3] @ observed @ rotation.T
        directions = cs.if_else(finger, transported, block[:, :3])
        angular = cs.vertcat(*[cs.cross(lever, rotation.T @ directions[i, :].T).T
                              for i in range(4)])
        robot = cs.if_else(finger, -directions, block[:, 6:])
        rows.append(cs.horzcat(directions, angular, robot))
        world_lever = rotation @ lever
        delta = state[:3]-reference[:3] + world_lever-observed@lever
        # Relative normal distance to a body-attached sphere patch. Table
        # contact instead measures the height of its transported support point.
        new_normal = cs.sum1(directions).T/4
        finger_gap = cs.dot(new_normal, state[:3]+world_lever-state[7:10]) - cs.dot(
            normal[:3].T, reference[:3]+observed@lever-reference[7:10])
        change = cs.if_else(finger, finger_gap, cs.dot(normal[:3].T, delta))
        gap = phi[start:start+4] + cs.repmat(change, 4, 1)
        if support_half_extents is not None:
            uv = cs.DM(support_tangents) @ (state[:3] + world_lever - cs.DM(support_point))
            inside = cs.logic_and(cs.fabs(uv[0]) <= support_half_extents[0],
                                  cs.fabs(uv[1]) <= support_half_extents[1])
            active = cs.logic_or(finger, inside)
            rows[-1] = cs.if_else(active, rows[-1], cs.DM.zeros(4, jacobian.size2()))
            gap = cs.if_else(active, gap, cs.DM.ones(4))
        gaps.append(gap)
    return cs.vertcat(*gaps), cs.vertcat(*rows)


def contact_gaps_cs(state, reference, phi, jacobian):
    import casadi as cs
    q = state[3:7] / cs.sqrt(cs.sumsqr(state[3:7]) + 1e-20)
    r = reference[3:7] / cs.sqrt(cs.sumsqr(reference[3:7]) + 1e-20)
    q *= cs.if_else(cs.dot(q,r) < 0, -1, 1)
    # First-order body rotation of conj(q_reference) * q. This remains smooth
    # at zero rotation and is invariant to quaternion sign.
    angular = 2*(r[0]*q[1:4] - q[0]*r[1:4] - cs.cross(r[1:4],q[1:4]))
    displacement = cs.vertcat(state[:3]-reference[:3], angular, state[7:]-reference[7:])
    gaps = []
    for start in range(0,int(jacobian.size1()),4):
        normal = cs.sum1(jacobian[start:start+4,:])/4
        gaps.append(phi[start:start+4] + cs.repmat(normal@displacement,4,1))
    return cs.vertcat(*gaps)


def joint_contact_geometry_cs(state, reference, phi, jacobian):
    """Transport object patches with linearized, coupled robot kinematics.

    Recover a world point Jacobian from the measured normal/tangent rows.
    This permits any number of robot DOFs, unlike the Cartesian tip helper.
    Object angular columns use the predicted body frame. Support directions
    stay world fixed; robot-only collision rows retain their linearization.
    """
    import casadi as cs
    rotation, observed = rotation_cs(state[3:7]), rotation_cs(reference[3:7])
    joint_delta = state[7:reference.numel()]-reference[7:]
    gaps, rows = [], []
    for start in range(0, int(jacobian.size1()), 4):
        block = jacobian[start:start+4, :]
        normal = cs.sum1(block)/4
        tangents = [(block[i, :]-block[i+2, :])/2 for i in (0, 1)]
        ns = cs.sqrt(cs.sumsqr(normal[:3])+1e-20)
        ts = cs.sqrt(cs.sumsqr(tangents[0][:3])+1e-20)
        n = observed.T@normal[:3].T/ns
        t = observed.T@tangents[0][:3].T/ts
        lever = cs.cross(n, normal[3:6].T/ns) + n*cs.dot(
            cs.cross(t, tangents[0][3:6].T/ts), n)
        object_contact = cs.sumsqr(block[:, :3]) > 0
        hand_contact = cs.sumsqr(block[:, 6:]) > 0
        attached = cs.logic_and(object_contact, hand_contact)
        directions = cs.if_else(attached, block[:, :3]@observed@rotation.T, block[:, :3])
        angular = cs.vertcat(*[cs.cross(lever, rotation.T@directions[i, :].T).T
                              for i in range(4)])
        # Rows are d.(v_object - J_point*v_robot). The three orthogonal
        # directions span the contact frame, including its friction scale.
        point_jac = -normal[:3].T@normal[6:]/(ns*ns)
        for tangent in tangents:
            point_jac -= tangent[:3].T@tangent[6:]/(cs.sumsqr(tangent[:3])+1e-20)
        robot = cs.if_else(attached, -directions@point_jac, block[:, 6:])
        rows.append(cs.if_else(object_contact, cs.horzcat(directions, angular, robot), block))
        point_delta = state[:3]-reference[:3]+(rotation-observed)@lever
        gap_change = cs.if_else(attached,
            cs.dot(cs.sum1(directions).T/4, point_delta-point_jac@joint_delta),
            cs.if_else(object_contact, cs.dot(normal[:3].T, point_delta),
                       normal[6:]@joint_delta))
        gaps.append(phi[start:start+4]+cs.repmat(gap_change, 4, 1))
    return cs.vertcat(*gaps), cs.vertcat(*rows)


def contact_gaps_torch(state, reference, phi, jacobian):
    import torch
    q = state[3:7] / torch.linalg.vector_norm(state[3:7]).clamp_min(1e-10)
    r = reference[3:7] / torch.linalg.vector_norm(reference[3:7]).clamp_min(1e-10)
    q = q*torch.where(torch.dot(q,r) < 0,-1.,1.)
    angular = 2*(r[0]*q[1:] - q[0]*r[1:] - torch.linalg.cross(r[1:],q[1:]))
    displacement = torch.cat([state[:3]-reference[:3],angular,state[7:]-reference[7:]])
    normals = jacobian.reshape(-1,4,jacobian.shape[1]).mean(dim=1)
    return phi + torch.repeat_interleave(normals@displacement,4)


def rotation_torch(quaternion):
    import torch
    q = quaternion / torch.linalg.vector_norm(quaternion).clamp_min(1e-10)
    w, x, y, z = q.unbind()
    return torch.stack((
        torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w))),
        torch.stack((2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w))),
        torch.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))))


def attached_point_torch(state, reference_pose, point_world):
    return state[:3] + rotation_torch(state[3:7]) @ (
        rotation_torch(reference_pose[3:7]).T @ (point_world-reference_pose[:3]))


def contact_geometry_torch(state, reference, phi, jacobian, *, support_point=None,
                           support_tangents=None, support_half_extents=None):
    import torch
    rotation, observed = rotation_torch(state[3:7]), rotation_torch(reference[3:7])
    gaps, rows = [], []
    for start in range(0, jacobian.shape[0], 4):
        block = jacobian[start:start+4]
        normal, tangent = block.mean(dim=0), (block[0]-block[2])/2
        ns = torch.linalg.vector_norm(normal[:3]).clamp_min(1e-10)
        ts = torch.linalg.vector_norm(tangent[:3]).clamp_min(1e-10)
        n, an = observed.T @ normal[:3]/ns, normal[3:6]/ns
        t, at = observed.T @ tangent[:3]/ts, tangent[3:6]/ts
        lever = torch.linalg.cross(n, an) + n*torch.dot(torch.linalg.cross(t, at), n)
        finger = block[:, 6:].square().sum() > 0
        directions = torch.where(finger, block[:, :3] @ observed @ rotation.T, block[:, :3])
        angular = torch.linalg.cross(lever.expand(4, 3), directions @ rotation)
        robot = torch.where(finger, -directions, block[:, 6:])
        rows.append(torch.cat((directions, angular, robot), dim=1))
        world_lever = rotation @ lever
        delta = state[:3]-reference[:3] + world_lever-observed@lever
        finger_gap = torch.dot(directions.mean(dim=0), state[:3]+world_lever-state[7:10]) - torch.dot(
            normal[:3], reference[:3]+observed@lever-reference[7:10])
        change = torch.where(finger, finger_gap, torch.dot(normal[:3], delta))
        gap = phi[start:start+4] + change
        if support_half_extents is not None:
            point = torch.as_tensor(support_point, device=state.device, dtype=state.dtype)
            tangents = torch.as_tensor(support_tangents, device=state.device, dtype=state.dtype)
            bounds = torch.as_tensor(support_half_extents, device=state.device, dtype=state.dtype)
            inside = torch.all(torch.abs(tangents @ (state[:3]+world_lever-point)) <= bounds)
            active = finger | inside
            rows[-1] = torch.where(active, rows[-1], torch.zeros_like(rows[-1]))
            gap = torch.where(active, gap, torch.ones_like(gap))
        gaps.append(gap)
    return torch.cat(gaps), torch.cat(rows)


def refresh_table_support(vertices, state, phi, jacobian, floor_z, mu, *,
                          support_point=None, support_normal=(0., 0., 1.),
                          support_tangents=None, support_half_extents=None, support_edges=None):
    """Refresh the collision-hull footprint on a horizontal or finite tilted plane."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from models.support_geometry import mesh_support_points, support_jacobians
    state = np.asarray(state)
    rotation = Rotation.from_quat(state[[4, 5, 6, 3]]).as_matrix()
    point = np.array([0., 0., floor_z]) if support_point is None else np.asarray(support_point)
    normal = np.asarray(support_normal, dtype=float)
    normal /= np.linalg.norm(normal)
    blocks = np.asarray(jacobian).reshape(-1, 4, 9)
    slots = np.flatnonzero((blocks[:, :, 6:]**2).sum(axis=(1, 2)) == 0)
    rows, gaps = np.asarray(jacobian).copy(), np.asarray(phi).copy()
    for slot in slots:
        rows[4*slot:4*slot+4] = 0.
        gaps[4*slot:4*slot+4] = 1.
    patches = mesh_support_points(vertices, state[:3], rotation, point, normal,
        tangents=support_tangents, half_extents=support_half_extents,
        edges=support_edges, max_points=min(4, len(slots)))
    for slot, patch in zip(slots, patches):
        block, _ = support_jacobians(state[:3], rotation, patch, 9, mu, normal)
        rows[4*slot:4*slot+4] = block
        gaps[4*slot:4*slot+4] = np.dot(patch-point, normal)
    return gaps, rows
