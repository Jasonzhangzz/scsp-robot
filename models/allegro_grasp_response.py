"""Force-limited made-contact response for a verified multi-pad grasp.

The unsupported phase retains softSCM. In a measured force-closed grasp,
couple object inertia to the robot's finite servo response, then bound the
allocated forces by actual normal preload and the physical Coulomb cones.
"""
import numpy as np


def grasp_operators(jacobian, hand_inverse, object_mass, num_contacts):
    """Recover point velocities from the measured normal/tangent frame."""
    blocks = np.asarray(jacobian).reshape(-1, 4, np.asarray(jacobian).shape[1])
    points = []
    for block in blocks[:num_contacts]:
        normal = block.mean(axis=0)
        tangent = [(block[i]-block[i+2])/2 for i in (0, 1)]
        point = np.outer(normal[:3], normal)
        for row in tangent:
            norm = np.linalg.norm(row[:3])
            if norm > 1e-10:
                point += np.outer(row[:3]/norm, row/norm)
        points.append(point)
    point = np.vstack(points)
    obj, robot = point[:, :6], -point[:, 6:]
    tangent = np.eye(3*num_contacts)
    for i, block in enumerate(blocks[:num_contacts]):
        normal = block[:, :3].mean(axis=0)
        tangent[3*i:3*i+3, 3*i:3*i+3] -= np.outer(normal, normal)
    tangent_obj = tangent@obj
    # Normal preload is controlled separately; made contact constrains
    # tangential slip. Retain the full map for degenerate tangential layouts.
    velocity_map = np.linalg.pinv(tangent_obj if np.linalg.matrix_rank(tangent_obj)==6 else obj, rcond=1e-8)
    mobility = (robot*np.asarray(hand_inverse)[None, :])@robot.T
    internal = velocity_map@mobility
    coupled = internal@velocity_map.T
    inverse = np.linalg.inv(np.eye(6)+coupled@object_mass)
    return velocity_map, robot, inverse, coupled, internal


def object_point_jacobian_cs(jac, num_contacts):
    import casadi as cs
    point_rows = []
    for i in range(int(num_contacts)):
        block = jac[4*i:4*i+4, :6]
        normal = cs.sum1(block)/4
        tangent = [(block[k, :]-block[k+2, :])/2 for k in (0, 1)]
        point = normal[:3].T@normal
        for row in tangent:
            point += row[:3].T@row/(cs.sumsqr(row[:3])+1e-20)
        point_rows.append(point)
    return cs.vertcat(*point_rows)


def force_limited_grasp_cs(velocity, raw_forces, current_velocity, hand_inverse,
                           object_inverse, external, jac, gaps, *, h, friction,
                           velocity_map, robot_jac, object_mass, grasp_inverse,
                           grasp_mobility, internal_map, reference_force, normal_limits,
                           other_wrench, friction_cone='coulomb', allow_preload_growth=False,
                           force_limit=None, retain_other_contacts=False):
    """Mixed world-linear/body-angular response, Newton reaction included."""
    import casadi as cs
    if friction_cone not in ('coulomb','pyramidal'):raise ValueError('Invalid friction cone')
    if allow_preload_growth and (force_limit is None or force_limit<=0):
        raise ValueError('Predicting preload growth requires a positive force limit')
    # V is pinv(J_object). Its transpose distributes a wrench correction.
    # Recover point wrench rows from the existing measured contact Jacobian;
    # all inverses are computed outside the SX derivative/codegen graph.
    object_jac = object_point_jacobian_cs(jac, normal_limits.numel())
    wrench_map = object_jac.T
    if retain_other_contacts:
        # The coupled solve replaces ONLY the pad response. Recover all other
        # contact reactions (including the floor) from the SCM momentum balance;
        # measured external wrenches must not be added a second time.
        other_wrench = object_mass@(velocity[:6]-current_velocity)/h-external-wrench_map@raw_forces
    # Undo raw pad impulses, retain robot-only/other-contact effects, then
    # solve the coupled robot/object velocity rather than imposing u/h.
    hand_free = velocity[6:]+h*hand_inverse*(robot_jac.T@raw_forces)
    rhs = velocity_map@robot_jac@hand_free + grasp_mobility@object_mass@current_velocity
    rhs += h*(grasp_mobility@(external+other_wrench+wrench_map@reference_force)
              -internal_map@reference_force)
    rigid_velocity = grasp_inverse@rhs
    required = object_mass@(rigid_velocity-current_velocity)/h-external-other_wrench
    allocated = reference_force+velocity_map.T@(required-wrench_map@reference_force)
    forces = []
    for i in range(int(normal_limits.numel())):
        normal = cs.sum1(jac[4*i:4*i+4, :3]).T/4
        value = allocated[3*i:3*i+3]
        limit=normal_limits[i]
        if allow_preload_growth:
            # Only the contact-dynamics prediction can add future preload;
            # a desired Lambda force alone cannot create support.
            raw_normal=cs.fmax(cs.dot(normal,raw_forces[3*i:3*i+3]),0)
            limit=cs.fmin(force_limit/cs.sqrt(1+friction**2),cs.fmax(limit,raw_normal))
        pressure = cs.fmin(limit, cs.fmax(cs.dot(normal, value), 0))
        tangent = value-normal*cs.dot(normal, value)
        if friction_cone=='pyramidal':
            block=jac[4*i:4*i+4,:3]
            t1=(block[0,:]-block[2,:]).T
            t2=(block[1,:]-block[3,:]).T
            t1/=cs.sqrt(cs.sumsqr(t1)+1e-20);t2/=cs.sqrt(cs.sumsqr(t2)+1e-20)
            length=cs.fabs(cs.dot(t1,tangent))+cs.fabs(cs.dot(t2,tangent))
        else:length=cs.sqrt(cs.sumsqr(tangent)+1e-20)
        tangent *= cs.fmin(1., friction*pressure/(length+1e-20))
        # No tensile or speculative support after a predicted separation.
        active = gaps[4*i] <= 1e-5
        forces.append(cs.if_else(active, normal*pressure+tangent, cs.DM.zeros(3)))
    forces = cs.vertcat(*forces)
    obj_velocity = (velocity[:6]+h*object_inverse@wrench_map@(forces-raw_forces)
                    if retain_other_contacts else
                    current_velocity+h*object_inverse@(external+other_wrench+wrench_map@forces))
    hand_velocity = velocity[6:]-h*hand_inverse*(robot_jac.T@(forces-raw_forces))
    return cs.vertcat(obj_velocity, hand_velocity), forces
