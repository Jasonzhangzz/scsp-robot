"""Render-only MANO/object ghosts; never add them to the execution model."""
from pathlib import Path
import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import cKDTree


class DemonstrationOverlay:
    def __init__(self, scene, vertices, faces, poses, alpha=.3):
        if not 0 < alpha < 1:
            raise ValueError('ghost alpha must be between zero and one')
        self.vertices = np.asarray(vertices)
        self.poses = np.asarray(poses)
        tree = ET.parse(scene)
        root = tree.getroot()
        # Resolve assets when the visualization XML is compiled from a string.
        for asset in root.findall('asset/*'):
            if asset.get('file'):
                asset.set('file', str((Path(scene).resolve().parent / asset.get('file')).resolve()))
        asset = root.find('asset')
        ET.SubElement(asset, 'mesh', name='demo_mano_mesh',
            vertex=' '.join(map(str, self.vertices[0].ravel())),
            face=' '.join(map(str, np.asarray(faces, int).ravel())))
        world = root.find('worldbody')
        body = ET.SubElement(world, 'body', name='demo_mano')
        ET.SubElement(body, 'geom', name='demo_mano_geom', type='mesh', mesh='demo_mano_mesh',
            contype='0', conaffinity='0', group='2', density='0',
            rgba=f'1 .55 .25 {alpha}', material='')
        obj = root.find("worldbody/body[@name='obj']")
        ghost = ET.SubElement(world, 'body', name='demo_object', mocap='true')
        visual = [g for g in obj.findall('geom') if g.get('contype', '1') == '0']
        for i, geom in enumerate(visual or obj.findall('geom')[:1]):
            geom = copy.deepcopy(geom)
            geom.attrib.update(name=f'demo_object_geom_{i}', contype='0', conaffinity='0',
                group='2', density='0', rgba=f'.1 .9 .8 {alpha}')
            geom.attrib.pop('material', None)
            ghost.append(geom)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.mesh = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MESH, 'demo_mano_mesh')
        self.geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'demo_mano_geom')
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'demo_object')
        self.mocap = int(self.model.body_mocapid[b])
        self.va = int(self.model.mesh_vertadr[self.mesh])
        self.vn = int(self.model.mesh_vertnum[self.mesh])
        self.fa = int(self.model.mesh_faceadr[self.mesh])
        self.fn = int(self.model.mesh_facenum[self.mesh])
        self.na = int(self.model.mesh_normaladr[self.mesh])
        self.nn = int(self.model.mesh_normalnum[self.mesh])
        mujoco.mj_forward(self.model, self.data)
        self.origin = self.data.geom_xpos[self.geom].copy()
        self.rotation = self.data.geom_xmat[self.geom].reshape(3, 3).copy()
        compiled = self.model.mesh_vert[self.va:self.va+self.vn] @ self.rotation.T + self.origin
        distances, self.vertex_ids = cKDTree(self.vertices[0]).query(compiled)
        if distances.max() > 1e-5 or len(np.unique(self.vertex_ids)) != self.vn:
            raise ValueError('MANO compiled mesh correspondence could not be recovered')
        self.frame = None

    def update(self, physical, frame):
        """Copy measured state for display; no writes to the physical MjData."""
        frame = float(np.clip(frame, 0, len(self.poses)-1))
        lower = int(np.floor(frame));upper = min(lower+1,len(self.poses)-1)
        fraction = frame-lower
        pose = (1-fraction)*self.poses[lower]+fraction*self.poses[upper]
        other = self.poses[upper,3:].copy()
        if np.dot(self.poses[lower,3:],other)<0:other *= -1
        pose[3:] = (1-fraction)*self.poses[lower,3:]+fraction*other
        pose[3:] /= np.linalg.norm(pose[3:])
        if physical.qpos.shape != self.data.qpos.shape:
            raise ValueError('Overlay must preserve execution degrees of freedom')
        for name in ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart', 'qfrc_applied', 'xfrc_applied'):
            target = getattr(self.data, name)
            source = getattr(physical, name)
            # Extra render bodies have no applied force.
            target[:] = 0
            target[:len(source)] = source
        self.data.time = physical.time
        self.data.mocap_pos[self.mocap] = pose[:3]
        self.data.mocap_quat[self.mocap] = pose[3:]
        changed = frame != self.frame
        if changed:
            interpolated = (1-fraction)*self.vertices[lower]+fraction*self.vertices[upper]
            self.origin = interpolated.mean(axis=0)
            self.model.geom_pos[self.geom] = self.origin
            vertices = (interpolated[self.vertex_ids]-self.origin) @ self.rotation
            self.model.mesh_vert[self.va:self.va+self.vn] = vertices
            self.model.geom_rbound[self.geom] = np.linalg.norm(vertices,axis=1).max()
            faces = self.model.mesh_face[self.fa:self.fa+self.fn]
            triangles = vertices[faces]
            normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            accumulated = np.zeros((self.nn, 3))
            ids = self.model.mesh_facenormal[self.fa:self.fa+self.fn]
            for corner in range(3):
                np.add.at(accumulated, ids[:, corner], normals)
            accumulated /= np.maximum(np.linalg.norm(accumulated, axis=1, keepdims=True), 1e-12)
            self.model.mesh_normal[self.na:self.na+self.nn] = accumulated
            self.frame = frame
        mujoco.mj_forward(self.model, self.data)
        return changed

    def set_camera(self, camera):
        points=np.concatenate((self.vertices.reshape(-1,3),self.poses[:,:3]))
        low,high=points.min(0),points.max(0)
        camera.lookat[:]=(low+high)/2
        camera.distance=max(.65,2.3*float(np.max(high-low)))
        camera.azimuth=135;camera.elevation=-25

    def upload(self, renderer):
        # Renderer owns its GL context; upload after every changed MANO frame.
        if renderer._gl_context:
            renderer._gl_context.make_current()
        mujoco.mjr_uploadMesh(self.model, renderer._mjr_context, self.mesh)
