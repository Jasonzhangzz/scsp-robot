"""Use DexForge ContactAware itself, then transfer FK across wrist charts.

CA supplies a posture reference only. Its optimized anchors, contact filtering,
and force certificates never replace the original MANO targets used by MPC.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from models.wuji_hand import rotation,build_scene
from planning.wuji_retarget import xyzw_xyz_to_pose


def transfer_postures(hand,source_model,states,source_poses,target_poses):
    """Preserve every body transform in object coordinates, not Euler angles."""
    source=mujoco.MjData(source_model);result=[];reports=[]
    joints=[mujoco.mj_id2name(hand.model,mujoco.mjtObj.mjOBJ_JOINT,int(j))
            for j in hand.model.actuator_trnid[:,0]]
    ids=[mujoco.mj_name2id(source_model,mujoco.mjtObj.mjOBJ_JOINT,name) for name in joints]
    if min(ids)<0:raise ValueError('CA scene joint names do not match the Wuji executor')
    qids=[int(source_model.jnt_qposadr[j]) for j in ids]
    palm=mujoco.mj_name2id(source_model,mujoco.mjtObj.mjOBJ_BODY,'right_palm_link')
    for state,src,dst in zip(states,source_poses,target_poses):
        source.qpos[:len(state)]=state;mujoco.mj_forward(source_model,source)
        delta=rotation(dst)@rotation(src).T
        desired_r=delta@source.xmat[palm].reshape(3,3)
        desired_p=dst[:3]+delta@(source.xpos[palm]-src[:3])
        q=np.asarray(source.qpos[qids],float).copy();q[:6]=0.
        zero=hand.set_state(q,dst,kinematic=True)
        angles=Rotation.from_matrix(desired_r@zero.xmat[hand.palm].reshape(3,3).T).as_euler('XYZ')
        alternative=np.array([angles[0]+np.pi,np.pi-angles[1],angles[2]+np.pi])
        choices=[]
        for a in (angles,alternative):
            a=(a+np.pi)%(2*np.pi)-np.pi
            if np.all(a>=hand.lower[3:6]) and np.all(a<=hand.upper[3:6]):choices.append(a)
        if not choices:raise ValueError('CA wrist pose cannot be represented within the executor joint bounds')
        q[3:6]=choices[0]
        data=hand.set_state(q,dst,kinematic=True);q[:3]+=desired_p-data.xpos[hand.palm]
        if np.any(q<hand.lower-1e-6) or np.any(q>hand.upper+1e-6):raise ValueError('CA posture exceeds executor joint limits')
        data=hand.set_state(q,dst,kinematic=True);position_error=rotation_error=0.
        for body in hand.hand_bodies:
            name=mujoco.mj_id2name(hand.model,mujoco.mjtObj.mjOBJ_BODY,body)
            other=mujoco.mj_name2id(source_model,mujoco.mjtObj.mjOBJ_BODY,name)
            # Virtual wrist links belong to different coordinate charts.
            if name.startswith('base_'):continue
            if other<0:raise ValueError(f'CA scene is missing hand body {name}')
            wanted=dst[:3]+delta@(source.xpos[other]-src[:3])
            position_error=max(position_error,float(np.linalg.norm(data.xpos[body]-wanted)))
            relative=data.xmat[body].reshape(3,3).T@delta@source.xmat[other].reshape(3,3)
            rotation_error=max(rotation_error,float(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec())))
        if position_error>2e-6 or rotation_error>2e-5:raise ValueError(('CA FK transfer mismatch',position_error,rotation_error))
        reports.append(dict(valid=True,geometric_pose_valid=True,backend='dexforge_contactaware',
            max_body_position_error_m=position_error,max_body_rotation_error_rad=rotation_error,
            max_contact_error=None,contact_targets_replaced=False))
        result.append(q)
    return np.asarray(result),reports


def solve_contactaware(hand,demo,args):
    root=Path(args.dexforge_root or args.asset_cache/'DexForge').resolve()
    output=args.output/'contactaware';sequence=args.output/'ca_sequence'
    supplied=getattr(args,'ca_result',None)
    if supplied is not None:
        output=Path(supplied).resolve()
        if getattr(args,'ca_scene',None) is None:raise ValueError('--ca-result requires --ca-scene (the mounted CA mesh XML)')
        scene=Path(args.ca_scene).resolve()
    else:
        scene=sequence/'scene/wuji_right/sugar_box_mesh.xml'
        meta=json.loads((args.sequence/'mano_raw/meta.json').read_text())
        label=meta['object'].partition('_')[2] if meta['object'].partition('_')[0].isdigit() else meta['object']
        scene=scene.with_name(f'{label}_mesh.xml')
        sequence.mkdir(parents=True,exist_ok=True)
        if (sequence/'mano_raw').exists():shutil.rmtree(sequence/'mano_raw')
        shutil.copytree(args.sequence/'mano_raw',sequence/'mano_raw',dirs_exist_ok=True)
        scene.parent.mkdir(parents=True,exist_ok=True)
        # Build fresh CA scenes from the same physical assets. CA owns and may
        # remount its private copy; the runtime scene remains independent.
        build_scene(args.hand_assets/'right_mesh.xml',args.sequence/'scene/leaphand'/f'{label}_mesh.xml',
            scene,demo.poses[0],args.sim_dt,physics_reference=args.physics_reference)
        tree=ET.parse(scene);document=tree.getroot();robot=ET.parse(args.hand_assets/'right_mesh.xml').getroot()
        for node in robot.findall('asset/*'):
            old=document.find(f"asset/*[@name='{node.get('name')}']")
            if old is not None:document.find('asset').remove(old)
        for node in robot.findall('worldbody/body'):
            old=document.find(f"worldbody/body[@name='{node.get('name')}']")
            if old is not None:document.find('worldbody').remove(old)
        for tag in ('default','actuator','equality'):
            old=document.find(tag)
            if old is not None:document.remove(old)
        for node in robot.iter():
            if node.get('file'):node.set('file',str((args.hand_assets/node.get('file')).resolve()))
        ET.ElementTree(robot).write(scene.parent/'right_mesh.xml')
        document.insert(0,ET.Element('include',file='right_mesh.xml'));tree.write(scene)
        sdf=ET.parse(scene);obj=sdf.getroot().find("worldbody/body[@name='obj']")
        collider=next(g for g in obj.findall('geom') if g.get('contype','1')!='0')
        mesh=sdf.getroot().find(f"asset/mesh[@name='{collider.get('mesh')}']")
        collider.set('type','sdfmesh');collider.set('file',mesh.get('file'));collider.attrib.pop('mesh',None)
        sdf.write(scene.with_name(f'{label}_sdf.xml'))
        environment=dict(os.environ,MANO_MODEL_ROOT=str(args.mano_models),HAND_ASSETS_ROOT=str(root/'assets'),
            OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl')
        for key,candidates in {
            'MANOPTH_ROOT':[root/'contactaware/third_party/manopth'],
            'COMFREE_WARP_ROOT':[root/'forceaware/third_party/comfree_warp'],
        }.items():
            path=next((p for p in candidates if p.exists()),None)
            if path is None:raise FileNotFoundError(f'{key} dependency missing; synchronize the complete DexForge dependencies')
            environment[key]=str(path)
        output.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',str(root),str(sequence.resolve()),str(output.resolve())]
        try:
            with (output/'solve.log').open('w') as log:subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'DexForge CA failed; see {output/"solve.log"}') from exc
        # mount_trajectory rewrites the scene and records the new wrist chart.
    states=np.load(output/'wuji_right_qpos.npy')
    poses=xyzw_xyz_to_pose(np.load(output/'object_pose_7.npy'))
    raw=np.load(args.sequence/'mano_raw/hand_pose_51.npy').reshape(-1,51)
    config=json.loads((output/'config.json').read_text())
    source_raw=Path(config['sequence_dir'])/'mano_raw'
    if not source_raw.exists():raise ValueError('Cannot verify CA input; the original CA sequence/mano_raw directory is missing')
    for name in ('hand_pose_51.npy','hand_shape_10.npy','extrinsics_4x4.npy','object_pose_camera_7.npy'):
        if not np.array_equal(np.load(source_raw/name),np.load(args.sequence/'mano_raw'/name)):
            raise ValueError(f'CA result input differs: {name}')
    valid=np.abs(raw).sum(axis=1)>1e-5
    if (args.sequence/'mano_raw/frame_valid.npy').exists():valid &= np.load(args.sequence/'mano_raw/frame_valid.npy').astype(bool)
    source_valid=np.abs(raw).sum(axis=1)>1e-5
    if (source_raw/'frame_valid.npy').exists():source_valid &= np.load(source_raw/'frame_valid.npy').astype(bool)
    if not np.array_equal(source_valid,valid):raise ValueError('CA result uses a different valid-frame mask')
    ids=np.flatnonzero(valid)
    if len(states)!=len(ids):raise ValueError('CA results must cover the same complete valid input sequence')
    selection=np.searchsorted(ids,demo.frame_ids)
    source=mujoco.MjModel.from_xml_path(str(scene))
    q,reports=transfer_postures(hand,source,states[selection],poses[selection],demo.poses)
    fingerprint=hashlib.sha256((args.sequence/'mano_raw/hand_pose_51.npy').read_bytes()).hexdigest()
    (args.output/'ca_ik_audit.json').write_text(json.dumps(dict(backend='DexForge original CA pipeline',
        source_result=str(output),source_scene=str(scene),mano_hand_pose_sha256=fingerprint,
        selected_frame_ids=demo.frame_ids.tolist(),max_body_position_error_m=max(r['max_body_position_error_m'] for r in reports),
        max_body_rotation_error_rad=max(r['max_body_rotation_error_rad'] for r in reports),
        mpc_contact_source='original_MANO_unchanged'),indent=2))
    return q,reports


if __name__=='__main__':
    # Isolate upstream imports and their global NumPy aliases from the MPC.
    _,root,sequence,output=sys.argv[1:]
    sys.path[:0]=[str(Path(root)/'contactaware'),root]
    sys.argv=['contactaware','--sequence-dir',sequence,'--hand','wuji_right','--output-dir',output,'--set','scene=sdf','--set','contact_topk=1']
    from contactaware.settings import parse_args
    from contactaware import pipeline
    # Rendering has no bearing on the numerical CA result.
    import contactaware.output as ca_output
    pipeline.render_keyframes=lambda *a,**kw:None
    ca_output.render_retarget_video=lambda *a,**kw:None
    pipeline.run(parse_args())
