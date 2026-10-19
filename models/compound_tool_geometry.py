"""Finite-tool distances and unloading against attached convex mug solids."""
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


CAVITY_TOOL_ROTATION = np.array([[0.,0.,-1.],[1.,0.,0.],[0.,-1.,0.]])


def cavity_entry_rotation(object_quaternion):
    """Tilt a sideways entry upward to leave room for the real wrist."""
    body_up = Rotation.from_quat(np.asarray(object_quaternion)[[1,2,3,0]]).inv().apply([0.,0.,1.])
    radial = body_up.copy();radial[0] = 0.
    norm = np.linalg.norm(radial)
    if norm < 1e-6:
        return CAVITY_TOOL_ROTATION.copy()
    axis = np.cross([1.,0.,0.],radial/norm)
    return Rotation.from_rotvec(np.deg2rad(25.)*norm*axis).as_matrix()@CAVITY_TOOL_ROTATION


def cavity_tool_solids(rotation=None):
    """Shaft and cooked wrist in the sphere frame, aligned through the mouth."""
    import trimesh
    shaft = trimesh.creation.cylinder(radius=.005/np.cos(np.pi/16), height=.06, sections=16)
    shaft.apply_translation([0.,0.,-.03])
    wrist = trimesh.load_mesh(str(Path(__file__).resolve().parents[1]/
        'envs/robots/assets/urdf/franka_description/meshes/collision/link7.obj')).convex_hull
    inverse = Rotation.from_quat([0.,0.,-.9238795,.3826834]).as_matrix()
    wrist.vertices = wrist.vertices @ inverse.T + [0.,0.,-.167]
    rotation = CAVITY_TOOL_ROTATION if rotation is None else np.asarray(rotation)
    for solid in (shaft,wrist):
        solid.vertices = solid.vertices @ rotation.T
    return shaft,wrist


def cavity_configuration_parts(parts, rotation=None):
    """Minkowski obstacles for translating a finite shaft and wrist."""
    import trimesh
    solids = cavity_tool_solids(rotation)
    obstacles = []
    for part in parts:
        for solid in solids:
            vertices = (np.asarray(part.vertices)[:,None,:]
                        - np.asarray(solid.vertices)[None,:,:]).reshape(-1,3)
            obstacles.append(trimesh.convex.convex_hull(vertices))
    return obstacles,np.concatenate([solid.vertices for solid in solids])


def collision_torque_projection(torque, inverse_mass, jacobian, qvel, tip, contacts, force_limit,
                                object_motion=None):
    """Stop task/posture torques squeezing a shaft or wrist into the cup.

    Project using the contact-point Jacobian and joint inertia, including
    rotation about the sphere center. Sphere loading remains physical; only
    excess measured penetration requests its withdrawal.
    """
    torque = np.asarray(torque,dtype=float).copy()
    active = 0
    for contact in sorted(contacts,key=lambda c:c['gap']):
        allowed = -.0002 if contact['kind'] == 'sphere' else .0005
        activation = .0001 if contact['kind'] == 'sphere' else .0005
        if contact['gap'] >= allowed+activation:
            continue
        normal = contact['normal']
        row = normal@jacobian[:3]+np.cross(contact['point']-tip,normal)@jacobian[3:]
        inverse = inverse_mass@row
        effective = float(row@inverse)
        if effective < 1e-8:
            continue
        speed = float(row@qvel)
        if object_motion is not None:
            linear,angular,position = object_motion
            speed -= float(normal@(linear+np.cross(angular,contact['point']-position)))
        desired = -80.*speed-4000.*(contact['gap']-allowed)
        increment = max(0., (desired-float(inverse@torque))/effective)
        if increment > 0.:
            torque += row*min(float(force_limit),increment)
            active += 1
    return torque,active


class MugUnloadingGuard:
    """Unload a suspended cup instead of maintaining its interior grip."""
    def __init__(self):
        self.active = False
        self.settled = 0

    def update(self, gap, object_velocity, contacts, tip_velocity,
               support_velocity=0., floor_gap=np.inf, escape_direction=None, contact_velocity=None):
        if gap > .003 or gap+.03*max(0.,float(support_velocity)) > .005:
            self.active = True
        velocity = np.asarray(object_velocity)
        quiet = gap < .001 and np.linalg.norm(velocity[:3]) < .02 and np.linalg.norm(velocity[3:]) < .3
        self.settled = self.settled+1 if quiet else 0
        if self.settled >= 25:
            self.active = False
        if not self.active:
            return None
        nearby = [c for c in contacts if c['gap'] < .002]
        sphere = [c for c in nearby if c['kind'] == 'sphere']
        chosen = min(sphere or nearby,key=lambda c:c['gap']) if nearby else None
        velocity = tip_velocity if chosen is None or contact_velocity is None else contact_velocity
        force = -40.*np.asarray(velocity)
        if chosen is not None:
            force += 4.*chosen['normal']
            if floor_gap < .004 and chosen['normal'][2] < -.2 and escape_direction is not None:
                # A ball under the cup cannot withdraw down through the
                # table. Slide it out instead of supporting the cup forever.
                direction = np.asarray(escape_direction,dtype=float).copy()
                direction[2] = 0.
                direction /= max(np.linalg.norm(direction),1e-12)
                force += 40.*.12*direction
                force[2] = 0.
        # No inward pressing at any currently touched tool piece.
        for _ in range(3):
            for c in nearby:
                force -= min(0.,float(force@c['normal']))*c['normal']
        force *= min(1.,4./max(np.linalg.norm(force),1e-12))
        return force


class CompoundToolGeometry:
    def __init__(self, paths, wrist=True):
        root = ET.Element('mujoco')
        option = ET.SubElement(root, 'option', ccd_tolerance='1e-9', ccd_iterations='100')
        ET.SubElement(option, 'flag', nativeccd='enable')
        assets, world = ET.SubElement(root, 'asset'), ET.SubElement(root, 'worldbody')
        obj = ET.SubElement(world, 'body')
        for i,path in enumerate(paths):
            ET.SubElement(assets, 'mesh', name='m%d'%i, file=str(path))
            ET.SubElement(obj, 'geom', type='mesh', mesh='m%d'%i)
        tool = ET.SubElement(world, 'body')
        ET.SubElement(tool, 'freejoint')
        ET.SubElement(tool, 'geom', name='sphere', type='sphere', size='.01')
        ET.SubElement(tool, 'geom', name='shaft', type='cylinder', size='.005 .03', pos='0 0 -.03')
        if wrist:
            mesh = Path(__file__).resolve().parents[1]/'envs/robots/assets/urdf/franka_description/meshes/collision/link7.obj'
            ET.SubElement(assets, 'mesh', name='wrist_mesh', file=str(mesh))
            # Invert the authored link7 -> attachment -> sphere transform.
            ET.SubElement(tool, 'geom', name='wrist', type='mesh', mesh='wrist_mesh',
                          pos='0 0 -.167', quat='.3826834 0 0 -.9238795')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.parts = len(paths)
        self.ids = {name:self.model.geom(name).id for name in ('sphere','shaft')}
        if wrist:
            self.ids['wrist'] = self.model.geom('wrist').id

    def contacts(self, position, rotation, tip, tool_rotation, margin=.003):
        rotation = np.asarray(rotation)
        local = rotation.T @ (np.asarray(tip)-np.asarray(position))
        orientation = Rotation.from_matrix(rotation.T @ tool_rotation).as_quat()[[3,0,1,2]]
        self.data.qpos[:] = np.r_[local,orientation]
        mujoco.mj_kinematics(self.model,self.data)
        contacts, gaps = [], {}
        for name,geom in self.ids.items():
            gaps[name] = float(margin)
            for part in range(self.parts):
                segment = np.zeros(6)
                gap = float(mujoco.mj_geomDistance(self.model,self.data,part,geom,margin,segment))
                gaps[name] = min(gaps[name],gap)
                if gap >= margin:
                    continue
                # Signed division preserves outward normals under penetration.
                if abs(gap) < 1e-10:
                    if name != 'sphere':
                        continue
                    normal = rotation@(local-segment[:3])
                else:
                    normal = rotation @ ((segment[3:]-segment[:3])/gap)
                length = np.linalg.norm(normal)
                if not np.isfinite(length) or length < 1e-8:
                    continue
                normal /= length
                point = np.asarray(position)+rotation@segment[3:]
                contacts.append(dict(kind=name,gap=gap,normal=normal,point=point))
        return contacts,gaps
