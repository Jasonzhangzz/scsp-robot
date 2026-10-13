"""Calibrated glove FK and recorded transforms used to locate its wrist."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def glove_fk(urdf, side, finger_flexion=0.0):
    """FK in wrist coordinates."""
    root = ET.parse(Path(urdf)).getroot()
    transforms = {'wrist': np.eye(4)}
    remaining = list(root.findall('joint'))
    while remaining:
        progressed = False
        for joint in remaining[:]:
            parent = joint.find('parent').get('link')
            if parent not in transforms:
                continue
            transform = np.eye(4)
            origin = joint.find('origin')
            if origin is not None:
                transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                transform[:3, :3] = Rotation.from_euler(
                    'xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
                ).as_matrix()
            name = joint.get('name')
            angle = 0.0
            if (joint.get('type') == 'revolute' and not name.startswith('thumb')
                    and ('flex' in name or name.endswith(('_pip', '_dip')))):
                angle = float(finger_flexion) * (1.0 if side == 'right' else -1.0)
                limit = joint.find('limit')
                if limit is not None:
                    angle = np.clip(angle, float(limit.get('lower')), float(limit.get('upper')))
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(axis * angle).as_matrix()
            transforms[joint.find('child').get('link')] = transforms[parent] @ transform
            remaining.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError(f'Disconnected glove URDF: {urdf}')
    return transforms


def world_from_emf(local_message, world_message):
    """Recover the recorded emitter's world pose from corresponding RX poses."""
    from wuji_retarget.emf_offline import json_emf_payload
    p, q, confidence = json_emf_payload(local_message)
    wp, wq, _ = json_emf_payload(world_message)
    index = int(np.argmax(confidence))
    local_rotation = Rotation.from_quat(q[index]).as_matrix()
    world_rotation = Rotation.from_quat(wq[index]).as_matrix()
    rotation = world_rotation @ local_rotation.T
    translation = np.asarray(wp[index]) - rotation @ np.asarray(p[index])
    # Both messages must describe one rigid transform, not unrelated sensor frames.
    residual = np.linalg.norm(np.asarray(p) @ rotation.T + translation - wp, axis=1)
    if not np.isfinite(residual).all() or residual.max() > 1e-4:
        raise ValueError('Local/world EMF messages do not share a rigid emitter pose')
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def transform_skeleton(positions, quaternions, world_from_wrist):
    rotation = world_from_wrist[:3, :3]
    positions_world = np.asarray(positions) @ rotation.T + world_from_wrist[:3, 3]
    rotations = Rotation.from_quat(np.asarray(quaternions)).as_matrix()
    quaternions_world = Rotation.from_matrix(rotation @ rotations).as_quat()
    return positions_world, quaternions_world


def recorded_world_from_wrist(episode, group, side, timestamps, tx_from_wrist=None):
    """Recover capture TX poses from synchronized H5 wrist poses and capture TF.

    Camera JSON topics may be independently sampled. Native H5 derivatives share
    the exact EMF source header, so they provide the emitter pose at the solved frame.
    The captured wrist frame is used only to recover TX; current calibrated bones
    and current wrist->TX extrinsics determine the new wrist and fingers.
    """
    import h5py
    if 'hand_skeleton/world/position' not in group:
        return None
    names = group['hand_skeleton/joint_names'].asstr()[:].tolist()
    index = names.index('wrist')
    captured = np.repeat(np.eye(4)[None], len(timestamps), axis=0)
    captured[:, :3, 3] = group['hand_skeleton/world/position'][:, index]
    captured[:, :3, :3] = Rotation.from_quat(
        group['hand_skeleton/world/orientation_xyzw'][:, index]).as_matrix()
    if tx_from_wrist is None:
        return captured
    tf_path = Path(episode) / 'wuji_glove_tf.h5'
    if not tf_path.is_file():
        return None
    with h5py.File(tf_path, 'r', locking=False) as h:
        tf = h['tf_static']
        match = ((tf['parent_frame'].asstr()[:] == side[0] + '_wrist') &
                 (tf['child_frame'].asstr()[:] == side[0] + '_hand_emf_tx'))
        selected = np.flatnonzero(match)
        if not len(selected):
            raise ValueError(f'{side}: capture wrist->EMF TX transform is missing')
        times = tf['timestamps'][:][selected]
        order = np.argsort(times, kind='stable')
        selected, times = selected[order], times[order]
        rows = np.clip(np.searchsorted(times, timestamps, side='right') - 1,
                       0, len(times) - 1)
        capture_wrist_from_tx = np.repeat(np.eye(4)[None], len(rows), axis=0)
        capture_wrist_from_tx[:, :3, 3] = tf['translation'][:][selected][rows]
        capture_wrist_from_tx[:, :3, :3] = Rotation.from_quat(
            tf['rotation'][:][selected][rows]).as_matrix()
    return captured @ capture_wrist_from_tx @ tx_from_wrist
