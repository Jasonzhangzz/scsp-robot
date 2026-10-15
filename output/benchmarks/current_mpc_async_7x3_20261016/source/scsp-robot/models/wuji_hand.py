"""Wuji material-point kinematics and a physical MuJoCo retarget scene.

The asset owns joint order and geometry; the generated scene records explicit
servo/torque scaling. Hand points are body-local physical surface points.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class MaterialPoint:
    body: int
    local: np.ndarray
    normal: np.ndarray
    finger: str
    region: str
    source_local: np.ndarray = None
    contact_patch: bool = False


def rotation(pose):
    return Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]]).as_matrix()


def normalize(value):
    value = np.asarray(value, float)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


class ObjectSurface:
    def __init__(self, mesh, seed=0, samples=12000, scale=None, convex=False):
        self.mesh = mesh if isinstance(mesh, trimesh.Trimesh) else trimesh.load_mesh(mesh, process=False)
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = self.mesh.to_geometry()
        if scale is not None:
            self.mesh=self.mesh.copy();self.mesh.apply_scale(np.asarray(scale,float))
        if convex:self.mesh=self.mesh.convex_hull
        # Local RNG, unlike trimesh.sample's process-global random seed.
        rng = np.random.default_rng(seed)
        faces = rng.choice(len(self.mesh.faces), samples, p=self.mesh.area_faces / self.mesh.area)
        uv = rng.random((samples, 2))
        uv[uv.sum(1) > 1] = 1 - uv[uv.sum(1) > 1]
        tri = self.mesh.triangles[faces]
        self.points = tri[:, 0] + uv[:, :1]*(tri[:, 1]-tri[:, 0]) + uv[:, 1:]*(tri[:, 2]-tri[:, 0])
        self.normals = -self.mesh.face_normals[faces]

    def query(self, points):
        """Closest mesh point, signed gap and inward normal, object coordinates."""
        points = np.asarray(points, float).reshape(-1, 3)
        closest, distance, faces = trimesh.proximity.closest_point(self.mesh, points)
        outward = self.mesh.face_normals[faces]
        sign = np.where(np.einsum('ij,ij->i', points-closest, outward) >= 0, 1., -1.)
        return closest, sign*distance, -outward


def build_scene(hand_xml, object_scene, output, initial_pose, dt=None,exclude_mounts=True,
                *,finger_stiffness_scale=1.,finger_force_scale=1.):
    """Replace the source robot, with explicit finger servo tuning.

    Scaling kp also scales kv by sqrt(scale), retaining its damping ratio.
    Both actuator and joint torque clamps must agree with the requested force
    scale; changing just one leaves the other clamp in effect.
    """
    if not np.isfinite([finger_stiffness_scale,finger_force_scale]).all() or min(finger_stiffness_scale,finger_force_scale)<=0:
        raise ValueError('finger servo scales must be positive and finite')
    hand_xml, object_scene = Path(hand_xml).resolve(), Path(object_scene).resolve()
    hand, scene = ET.parse(hand_xml).getroot(), ET.parse(object_scene).getroot()
    finger_joints={joint.get('name') for joint in hand.findall('.//joint')
        if (joint.get('name') or '').startswith('right_finger')}
    for joint in hand.findall('.//joint'):
        if joint.get('name') in finger_joints and joint.get('actuatorfrcrange'):
            joint.set('actuatorfrcrange',' '.join(str(float(v)*finger_force_scale) for v in joint.get('actuatorfrcrange').split()))
    for actuator in hand.findall('actuator/position'):
        if actuator.get('joint') not in finger_joints:continue
        actuator.set('kp',str(float(actuator.get('kp','1'))*finger_stiffness_scale))
        if actuator.get('kv') is not None:
            actuator.set('kv',str(float(actuator.get('kv'))*np.sqrt(finger_stiffness_scale)))
        if actuator.get('forcerange'):
            actuator.set('forcerange',' '.join(str(float(v)*finger_force_scale) for v in actuator.get('forcerange').split()))
    obj = copy.deepcopy(scene.find("worldbody/body[@name='obj']"))
    if obj is None:
        raise ValueError('source scene must contain body obj')
    root = ET.Element('mujoco', model='wuji_semantic_retarget')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true')
    # DexForge hand options, then scene overrides. Unspecified fields retain
    # MuJoCo defaults; do not introduce a different contact/solver profile.
    option=ET.SubElement(root,'option')
    for source in (hand,scene):
        source_option=source.find('option')
        if source_option is not None:
            option.attrib.update(source_option.attrib)
            for child in source_option:option.append(copy.deepcopy(child))
    if dt is not None:option.set('timestep',str(dt))
    asset = ET.SubElement(root, 'asset')
    for source, directory in ((hand, hand_xml.parent), (scene, object_scene.parent)):
        for item in source.findall('asset/*'):
            item = copy.deepcopy(item)
            if item.get('file'):
                item.set('file', str((directory/item.get('file')).resolve()))
            asset.append(item)
    default = hand.find('default')
    if default is not None:
        root.append(copy.deepcopy(default))
    world = ET.SubElement(root, 'worldbody')
    for body in hand.findall('worldbody/body'):
        body = copy.deepcopy(body)
        # Preserve the upstream physical geoms, collision masks, contact
        # defaults and duplicated colliders, as well as inertias and servos.
        world.append(body)
    obj.set('pos', ' '.join(map(str, initial_pose[:3])))
    obj.set('quat', ' '.join(map(str, initial_pose[3:])))
    world.append(obj)
    floor = copy.deepcopy(scene.find("worldbody/geom[@name='floor']"))
    if floor is not None:
        world.append(floor)
    ET.SubElement(world, 'light', pos='0 0 2', dir='0 0 -1', diffuse='.8 .8 .8')
    for tag in ('actuator', 'contact', 'equality'):
        element = hand.find(tag)
        if element is not None:
            root.append(copy.deepcopy(element))
    # Convex mesh hulls overlap at articulated mounting joints (including
    # palm/proximal grandchildren) for every hand pose. Those are structural
    # geometry overlaps, not finger/object contacts. Keep all source physical
    # coefficients and masks, but exclude these impossible mount collisions.
    # The unmodified source pairing is available for physics diagnostics.
    if exclude_mounts:
        contact=root.find('contact')
        if contact is None:contact=ET.SubElement(root,'contact')
        def mounts(body,ancestors):
            for ancestor in ancestors[-2:]:
                ET.SubElement(contact,'exclude',body1=ancestor,body2=body.get('name'))
            for child in body.findall('body'):mounts(child,ancestors+[body.get('name')])
        for body in world.findall('body'):
            if body.get('name')!='obj':mounts(body,[])
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding='utf-8', xml_declaration=True)
    return output


class WujiHand:
    def __init__(self, scene, surface):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.mount_exclusions={(entry.get('body1'),entry.get('body2')) for entry in ET.parse(scene).getroot().findall('contact/exclude')}
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)
        self.preview = mujoco.MjData(self.model)
        self.collision_probe = mujoco.MjData(self.model)
        self.surface = surface
        self._body_hulls = {}
        self._body_meshes = {}
        self.obj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'obj')
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'obj_joint')
        self.oq = int(self.model.jnt_qposadr[joint]); self.ov = int(self.model.jnt_dofadr[joint])
        joints = self.model.actuator_trnid[:, 0]
        self.qadr = self.model.jnt_qposadr[joints].astype(int)
        self.dadr = self.model.jnt_dofadr[joints].astype(int)
        self.limits = np.where(self.model.jnt_limited[joints, None], self.model.jnt_range[joints], [-3., 3.])
        self.lower, self.upper = self.limits.T.copy()
        self.lower = np.maximum(self.lower, np.where(self.model.actuator_ctrllimited, self.model.actuator_ctrlrange[:, 0], -np.inf))
        self.upper = np.minimum(self.upper, np.where(self.model.actuator_ctrllimited, self.model.actuator_ctrlrange[:, 1], np.inf))
        self.nu = self.model.nu
        colliders=np.flatnonzero((self.model.geom_bodyid==self.obj)&(self.model.geom_contype!=0))
        self.friction=float(self.model.geom_friction[colliders[0],0])
        floor=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
        object_geom=colliders[0]
        if floor>=0:
            if self.model.geom_priority[floor]>self.model.geom_priority[object_geom]:
                self.floor_friction=float(self.model.geom_friction[floor,0])
            elif self.model.geom_priority[floor]<self.model.geom_priority[object_geom]:
                self.floor_friction=self.friction
            else:self.floor_friction=max(self.friction,float(self.model.geom_friction[floor,0]))
        else:self.floor_friction=self.friction
        self.palm = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_palm_link')
        if self.palm < 0:
            raise ValueError('right Wuji palm not found')
        self.hand_bodies = {i for i in range(1,self.model.nbody) if i != self.obj and not self._descendant(i,self.obj)}
        self.servo = self.data.qpos[self.qadr].copy()
        mujoco.mj_forward(self.model, self.data)

    def _descendant(self, body, ancestor):
        while body > 0:
            if body == ancestor: return True
            body = self.model.body_parentid[body]
        return False

    def structural_pair(self, first, second):
        if first<=0 or second<=0:return False
        name1=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,int(first))
        name2=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,int(second))
        if (name1,name2) in self.mount_exclusions or (name2,name1) in self.mount_exclusions:return True
        for body,ancestor in ((first,second),(second,first)):
            for _ in range(1):
                body=int(self.model.body_parentid[body])
                if body==ancestor:return True
        return False

    def set_state(self, q, object_pose, data=None,kinematic=False):
        data = self.scratch if data is None else data
        data.qpos[:] = self.model.qpos0
        data.qpos[self.qadr] = q
        data.qpos[self.oq:self.oq+7] = object_pose
        data.qvel[:] = 0
        if kinematic:
            mujoco.mj_kinematics(self.model,data)
            mujoco.mj_comPos(self.model,data)
            mujoco.mj_collision(self.model,data)
        else:mujoco.mj_forward(self.model, data)
        return data

    def kinematics(self, materials, data=None):
        data = self.data if data is None else data
        points, normals, jacobians = [], [], []
        for material in materials:
            r = data.xmat[material.body].reshape(3,3)
            p = data.xpos[material.body] + r@material.local
            jp, jr = np.zeros((3,self.model.nv)),np.zeros((3,self.model.nv))
            mujoco.mj_jac(self.model,data,jp,jr,p,material.body)
            points.append(p); normals.append(r@material.normal); jacobians.append(jp[:,self.dadr])
        return (np.asarray(points).reshape(-1,3),np.asarray(normals).reshape(-1,3),
                np.asarray(jacobians).reshape(-1,3,self.nu))

    def material_on_collider(self, body, local, normal, finger, region):
        """Lift a semantic query onto the executed convex mesh union.

        Upstream query points follow visual concavities. MuJoCo mesh collision
        uses convex hulls: keeping an internal query as a contact point can make
        geometrically coincident IK physically impossible (or hide real touch).
        Project only on the same body's outward ray, retaining finger/region.
        """
        if body not in self._body_hulls:
            hulls=[];meshes=[];seen=set()
            for geom in np.flatnonzero(self.model.geom_bodyid==body):
                if self.model.geom_contype[geom]==0 or self.model.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_MESH:continue
                key=(int(self.model.geom_dataid[geom]),tuple(self.model.geom_pos[geom]),tuple(self.model.geom_quat[geom]))
                if key in seen:continue
                seen.add(key)
                mesh=int(self.model.geom_dataid[geom]);start=int(self.model.mesh_vertadr[mesh]);count=int(self.model.mesh_vertnum[mesh])
                vertices=np.asarray(self.model.mesh_vert[start:start+count],float)
                r=Rotation.from_quat(self.model.geom_quat[geom][[1,2,3,0]]).as_matrix()
                hull=trimesh.convex.convex_hull(vertices@r.T+self.model.geom_pos[geom])
                normals=hull.face_normals
                bounds=np.einsum('ij,ij->i',normals,hull.triangles_center)
                hulls.append((normals,bounds));meshes.append(hull)
            self._body_hulls[body]=hulls;self._body_meshes[body]=meshes
        local=np.asarray(local,float);normal=normalize(normal)
        intervals=[]
        for normals,bounds in self._body_hulls[body]:
            gaps=bounds-normals@local;slopes=normals@normal
            if np.any((np.abs(slopes)<1e-10)&(gaps<-1e-6)):continue
            positives=np.flatnonzero(slopes>1e-10);negatives=np.flatnonzero(slopes<-1e-10)
            if not len(positives):continue
            exit_face=int(positives[np.argmin(gaps[positives]/slopes[positives])])
            upper=float(gaps[exit_face]/slopes[exit_face])
            lower=float(np.max(gaps[negatives]/slopes[negatives])) if len(negatives) else -np.inf
            if lower<=upper:intervals.append((lower,upper,normals[exit_face]))
        end=0.;surface_normal=normal
        changed=True
        while changed:
            changed=False
            for lower,upper,face_normal in intervals:
                if lower<=end+1e-6 and upper>end+1e-9:
                    end=upper;surface_normal=face_normal;changed=True
        return MaterialPoint(body,local+end*normal,surface_normal.copy(),finger,region,local.copy())

    def nearest_material(self, original, target_world, data, radius=.01):
        """A physical patch in the same region; original remains the soft prior."""
        if original.body not in self._body_meshes:
            self.material_on_collider(original.body,original.local,original.normal,original.finger,original.region)
        r=data.xmat[original.body].reshape(3,3)
        local=(target_world-data.xpos[original.body])@r
        best=None
        for hull,(normals,bounds) in zip(self._body_meshes[original.body],self._body_hulls[original.body]):
            # A face projection satisfying every convex halfspace is the exact
            # closest point. Most pad queries take this path; edges/corners use
            # the full triangle query. Physical duplicate geoms remain intact.
            violation=normals@local-bounds;face=int(np.argmax(violation))
            normal=normals[face];point=local-violation[face]*normal
            if np.max(normals@point-bounds)>1e-9:
                projected,_,faces=trimesh.proximity.closest_point(hull,local[None])
                point=projected[0];normal=hull.face_normals[faces[0]]
            distance=float(np.linalg.norm(point-local))
            offset=float(np.linalg.norm(point-original.local))
            if offset>radius or normal@original.normal<.2:continue
            score=float(distance**2+.1*offset**2)
            if best is None or score<best[0]:best=(score,point,normal)
        if best is None:return original
        return MaterialPoint(original.body,best[1],best[2],original.finger,original.region,original.local.copy())

    def normal_jacobians(self, materials, data=None):
        data=self.data if data is None else data
        rows=[]
        points,normals,_=self.kinematics(materials,data)
        for i,m in enumerate(materials):
            jp,jr=np.zeros((3,self.model.nv)),np.zeros((3,self.model.nv))
            mujoco.mj_jac(self.model,data,jp,jr,points[i],m.body)
            rows.append(np.cross(jr[:,self.dadr].T,normals[i]).T)
        return np.asarray(rows).reshape(-1,3,self.nu)

    def collision_report(self, data, allowed_bodies=(), penetration=.0015):
        worst = 0.; bad = []
        for c in data.contact:
            b1,b2 = self.model.geom_bodyid[[c.geom1,c.geom2]]
            # Adjacent physical links contain intentional mesh overlaps.
            related = self.structural_pair(b1,b2)
            allowed = ((b1 == self.obj and b2 in allowed_bodies) or (b2 == self.obj and b1 in allowed_bodies))
            if (b1 in self.hand_bodies or b2 in self.hand_bodies) and not related:
                worst = max(worst, -float(c.dist))
                limit = getattr(self,'contact_penetration_limit',.003) if allowed else penetration
                if c.dist < -limit: bad.append((int(b1),int(b2),float(c.dist)))
        return dict(valid=not bad,max_penetration=worst,collisions=bad)

    def solve_ik(self, q0, pose, materials, targets, target_normals=None, shape_materials=(),
                 shape_targets=None, iterations=60, tolerance=.012, proximity=.035,additional_contact_bodies=(),
                 surface_priors=None,surface_radius=.03,material_weight=1.,shape_weight=.3):
        """Coupled bounded IK, including physical collision residuals."""
        q0 = np.clip(np.asarray(q0,float),self.lower+1e-7,self.upper-1e-7)
        materials = tuple(materials); shape_materials = tuple(shape_materials)
        targets = np.asarray(targets).reshape(-1,3)
        allowed = {m.body for m in materials}|set(additional_contact_bodies)
        position_scale=min(.01,tolerance/2)
        cache = {}
        collision_signs={}
        def selected_materials(data):
            if surface_priors is None:return materials
            return tuple(self.nearest_material(prior,target,data,radius=surface_radius)
                for prior,target in zip(surface_priors,targets))
        def evaluate(q):
            if 'q' in cache and np.array_equal(q,cache['q']): return cache['residual'],cache['jac']
            data = self.set_state(q,pose,kinematic=True)
            selected=selected_materials(data)
            p,n,j = self.kinematics(selected,data)
            if surface_priors is not None:
                # Eliminate surface coordinates by closest projection at
                # each joint iterate. Tangential pad motion changes the
                # selected material, so only normal motion changes distance
                # on a face. Use the full point Jacobian at patch boundaries.
                residual=p-targets
                tangential=residual-n*np.einsum('ij,ij->i',residual,n)[:,None]
                interior=np.linalg.norm(tangential,axis=1)<1e-5
                normal_jac=np.einsum('ij,ijk->ik',n,j)
                j[interior]=np.einsum('ij,ik->ijk',n[interior],normal_jac[interior])
            terms = [material_weight*(p-targets).ravel()/position_scale, proximity*(q-q0)]
            rows = [material_weight*j.reshape(-1,self.nu)/position_scale,proximity*np.eye(self.nu)]
            if target_normals is not None:
                terms.append(.2*(n-target_normals).ravel())
                nr=[]
                for k,m in enumerate(selected):
                    jp,jr=np.zeros((3,self.model.nv)),np.zeros((3,self.model.nv))
                    mujoco.mj_jac(self.model,data,jp,jr,p[k],m.body)
                    nr.append(np.cross(jr[:,self.dadr].T,n[k]).T)
                rows.append(.2*np.array(nr).reshape(-1,self.nu))
            if shape_materials:
                sp,_,sj = self.kinematics(shape_materials,data)
                terms.append(shape_weight*(sp-shape_targets).ravel()/.02)
                rows.append(shape_weight*sj.reshape(-1,self.nu)/.02)
            violation = np.zeros(self.model.ngeom)
            vjac=np.zeros((self.model.ngeom,self.nu))
            for c in data.contact:
                b1,b2 = self.model.geom_bodyid[[c.geom1,c.geom2]]
                if self.structural_pair(b1,b2): continue
                allowed_contact = (b1 == self.obj and b2 in allowed) or (b2 == self.obj and b1 in allowed)
                threshold = getattr(self,'contact_target_depth',.0015) if allowed_contact else .0005
                if b1 in self.hand_bodies or b2 in self.hand_bodies:
                    index=min(c.geom1,c.geom2);depth=-c.dist-threshold
                    if depth>violation[index]:
                        violation[index]=depth
                        jp1,jp2,jr=np.zeros((3,self.model.nv)),np.zeros((3,self.model.nv)),np.zeros((3,self.model.nv))
                        mujoco.mj_jac(self.model,data,jp1,jr,c.pos,int(b1))
                        mujoco.mj_jac(self.model,data,jp2,jr,c.pos,int(b2))
                        row=-c.frame[:3]@(jp2-jp1)[:,self.dadr]
                        key=(int(c.geom1),int(c.geom2))
                        if key not in collision_signs and np.max(np.abs(row))>1e-8:
                            # Collision distance is the IK residual, whereas
                            # the force frame belongs to the contact solver.
                            # Verify its distance-gradient orientation for the
                            # current mesh pair before using it in least squares.
                            axis=int(np.argmax(np.abs(row)));epsilon=1e-6
                            probe=q.copy();probe[axis]+=epsilon
                            trial=self.set_state(probe,pose,self.collision_probe,kinematic=True)
                            before=max(-cc.dist for cc in data.contact if (cc.geom1,cc.geom2)==key)
                            after=[-cc.dist for cc in trial.contact if (cc.geom1,cc.geom2)==key]
                            numeric=(max(after)-before)/epsilon if after else row[axis]
                            collision_signs[key]=1. if numeric*row[axis]>=0 else -1.
                        vjac[index]=collision_signs.get(key,1.)*row
            terms.append(3*violation/.002);rows.append(3*vjac/.002)
            value,jacobian=np.concatenate(terms),np.vstack(rows)
            cache.update(q=q.copy(),residual=value,jac=jacobian)
            return value,jacobian
        result = least_squares(lambda q:evaluate(q)[0],q0,jac=lambda q:evaluate(q)[1],
            bounds=(self.lower+1e-8,self.upper-1e-8),max_nfev=iterations,ftol=1e-5,xtol=1e-5,gtol=1e-5)
        data = self.set_state(result.x,pose,kinematic=True)
        selected=selected_materials(data)
        p,_,_ = self.kinematics(selected,data)
        error = np.linalg.norm(p-targets,axis=1)
        collision = self.collision_report(data,allowed)
        shape_error=[]
        if shape_materials:
            sp,_,_=self.kinematics(shape_materials,data)
            shape_error=np.linalg.norm(sp-shape_targets,axis=1)
        report=dict(collision,valid=bool(collision['valid'] and (not len(error) or error.max()<=tolerance)),
            max_contact_error=float(error.max()) if len(error) else 0.,iterations=int(result.nfev),
            max_shape_error=float(np.max(shape_error)) if len(shape_error) else None)
        if surface_priors is not None:report['materials']=list(selected)
        return result.x,report

    def dynamics(self, data=None):
        data = self.data if data is None else data
        mass = np.empty((self.model.nv,self.model.nv))
        try: mujoco.mj_fullM(self.model,data,mass)
        except TypeError: mujoco.mj_fullM(self.model,mass,data.qM)
        oi = np.arange(self.ov,self.ov+6)
        return mass[np.ix_(oi,oi)],np.diag(mass)[self.dadr],-data.qfrc_bias[oi].copy()

    def execute(self, target, dt):
        self.servo = np.clip(target,self.lower,self.upper)
        self.data.ctrl[:] = self.servo
        # DexForge executes only position actuators, with zero applied forces.
        for _ in range(max(1,int(round(dt/self.model.opt.timestep)))):
            self.data.qfrc_applied[:] = 0
            mujoco.mj_step(self.model,self.data)
        mujoco.mj_forward(self.model,self.data)

    def servo_reference(self,q):
        """Gravity feedforward through position targets, with upstream limits.

        This changes the control command, never physics gains or applied forces.
        The optimizer still tracks the geometric joint pose itself.
        """
        gain=self.model.actuator_gainprm[:,0]
        offset=self.data.qfrc_bias[self.dadr]/np.maximum(gain,1e-12)
        return np.clip(np.asarray(q)+offset,self.lower,self.upper)

    def transport_configuration(self,q,source_pose,target_pose):
        """Transport a nominal contact-IK seed to the measured object pose.

        The asset's wrist is world XYZ translation followed by intrinsic XYZ
        hinges. Finger coordinates and material identities stay unchanged.
        This changes only a candidate reference; native preview validates it.
        """
        result=np.asarray(q,float).copy()
        delta=rotation(target_pose)@rotation(source_pose).T
        result[:3]=np.asarray(target_pose)[:3]+delta@(result[:3]-np.asarray(source_pose)[:3])
        angles=Rotation.from_matrix(delta@Rotation.from_euler('XYZ',result[3:6]).as_matrix()).as_euler('XYZ')
        alternative=np.array([angles[0]+np.pi,np.pi-angles[1],angles[2]+np.pi])
        candidates=[]
        for value in (angles,alternative):
            value=value+2*np.pi*np.round((result[3:6]-value)/(2*np.pi))
            if np.all(value>=self.lower[3:6]) and np.all(value<=self.upper[3:6]):candidates.append(value)
        result[3:6]=min(candidates,key=lambda value:np.linalg.norm(value-result[3:6])) if candidates else angles
        return np.clip(result,self.lower,self.upper)

    def preview_control(self,target,dt):
        """Predict one control cycle with the executed MuJoCo physics.

        This shadow state neither changes measured state nor applies an ideal
        contact force. It checks reduced-model proposals against real contacts.
        """
        return self.preview_sequence([target],dt)

    def preview_sequence(self,targets,dt,observer=None):
        """Roll out position controls, optionally observing each physical step.

        The observer must copy any arrays it retains: the shadow data is reused.
        """
        specification=mujoco.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mujoco.mj_stateSize(self.model,specification))
        mujoco.mj_getState(self.model,self.data,state,specification)
        mujoco.mj_setState(self.model,self.preview,state,specification)
        self.preview.qfrc_applied[:]=0.
        mujoco.mj_forward(self.model,self.preview)
        for target in targets:
            self.preview.ctrl[:]=np.clip(target,self.lower,self.upper)
            for _ in range(max(1,int(round(dt/self.model.opt.timestep)))):
                mujoco.mj_step(self.model,self.preview)
            mujoco.mj_forward(self.model,self.preview)
            if observer is not None:observer(self.preview)
        return self.preview

    def object_pose(self):
        return self.data.qpos[self.oq:self.oq+7].copy()

    def finger_contact_gaps(self):
        """Native finger/object gaps, including patches with nearly zero force."""
        result={}
        for contact in self.data.contact:
            b1,b2=self.model.geom_bodyid[[contact.geom1,contact.geom2]]
            if self.obj not in (b1,b2):continue
            other=int(b2 if b1==self.obj else b1)
            name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,other) or ''
            for number,finger in enumerate(('thumb','index','middle','ring','pinky'),1):
                if name.startswith(f'right_finger{number}_'):
                    result[finger]=min(result.get(finger,np.inf),float(contact.dist))
        return result

    def observed_contacts(self, materials, active=None, data=None):
        """Actual surface contacts and forces on the object, per semantic slot."""
        data=self.data if data is None else data
        p,_,_ = self.kinematics(materials,data)
        forces = np.zeros((len(materials),3)); actual=[]
        for contact_index,c in enumerate(data.contact):
            b1,b2 = self.model.geom_bodyid[[c.geom1,c.geom2]]
            if self.obj not in (b1,b2): continue
            other = b2 if b1 == self.obj else b1
            if other not in self.hand_bodies: continue
            local = np.zeros(6); mujoco.mj_contactForce(self.model,data,contact_index,local)
            f = c.frame.reshape(3,3).T@local[:3]
            if b1 == self.obj: f = -f
            normal = c.frame[:3].copy() * (-1 if b1 == self.obj else 1)
            ids = [i for i,m in enumerate(materials) if m.body==other]
            physical=[i for i in ids if materials[i].contact_patch]
            preferred=physical or [i for i in ids if active is not None and active[i]]
            if preferred:ids=preferred
            def patch_distance(i):
                distance=np.linalg.norm(p[i]-c.pos)
                if physical:
                    direction=data.xmat[materials[i].body].reshape(3,3)@materials[i].normal
                    distance+=.01*(1-float(direction@normal))
                return distance
            slot = min(ids,key=patch_distance) if ids else -1
            if slot>=0: forces[slot]+=f
            if c.dist<=.0001 or np.linalg.norm(f)>.001: actual.append(dict(point=np.array(c.pos),normal=normal,
                tangents=c.frame.reshape(3,3)[1:].copy(),force=f,body=int(other),slot=slot,gap=float(c.dist)))
        return forces,actual
