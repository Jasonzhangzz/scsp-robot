"""Prepare/run the requested executor × retarget-controller ablations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]


def prepare(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    source=ROOT/'outputs/wuji_contact_fix_rollout_20261016'
    sequence=output/'sequence';ca=output/'contactaware'
    sequence.mkdir(exist_ok=True);ca.mkdir(exist_ok=True)
    shutil.copytree(ROOT/'outputs/retarget_shared_20261016/mano_raw',sequence/'mano_raw',dirs_exist_ok=True)
    shutil.copytree(ROOT/'outputs/retarget_shared_20261016/scene/leaphand',sequence/'scene/leaphand',dirs_exist_ok=True)
    scene=sequence/'scene/wuji_right';scene.mkdir(exist_ok=True)
    tree=ET.parse(source/'scene.xml');root=tree.getroot()
    model_dir=sequence/'model';model_dir.mkdir(exist_ok=True)
    for node in root.findall('asset/*'):
        if node.get('file') and not node.get('file').lower().endswith('.stl'):
            source_file=Path(node.get('file'));target=model_dir/source_file.name
            shutil.copy2(source_file,target);node.set('file',str(target))
    tree.write(scene/'sugar_box_mesh.xml')
    geom=next(g for g in root.findall("worldbody/body[@name='obj']/geom") if g.get('contype','1')!='0')
    mesh=root.find(f"asset/mesh[@name='{geom.get('mesh')}']")
    geom.set('type','sdfmesh');geom.set('file',mesh.get('file'));geom.attrib.pop('mesh',None)
    # Official FA uses the hand's querypoint cloud against the object SDF.
    # MuJoCo's mesh counterpart has the same bodies, joints and inertias.
    robot=ET.parse(ROOT/'thirdparty/retarget_assets/DexForge/assets/wuji/right.xml').getroot()
    for node in robot.findall('asset/*'):
        old=root.find(f"asset/*[@name='{node.get('name')}']")
        if old is not None:root.find('asset').remove(old)
    for node in robot.findall('worldbody/body'):
        old=root.find(f"worldbody/body[@name='{node.get('name')}']")
        if old is not None:root.find('worldbody').remove(old)
    for tag in ('default','actuator','equality'):
        old=root.find(tag)
        if old is not None:root.remove(old)
    root.insert(0,ET.Element('include',file=str((ROOT/'thirdparty/retarget_assets/DexForge/assets/wuji/right.xml').resolve())))
    root.find('option').set('timestep','.01');tree.write(scene/'sugar_box_sdf.xml')
    q=np.load(source/'wuji_q.npy');semantic=np.load(source/'semantic_contacts.npz')
    np.save(ca/'wuji_right_qpos.npy',q)
    np.save(ca/'object_pose_7.npy',semantic['object_pose'][:,[4,5,6,3,0,1,2]])
    original=ROOT/'outputs/wuji_contact_fix_20261016/contactaware'
    for name in ['contact_guidance.npz','camera_pose_7.npy']:shutil.copy2(original/name,ca/name)
    cfg=json.loads((original/'config.json').read_text());cfg.update(sequence_dir=str(sequence),render_xml=str(scene/'sugar_box_mesh.xml'),scene_xml=str(scene/'sugar_box_sdf.xml'))
    (ca/'config.json').write_text(json.dumps(cfg,indent=2))
    forge=ROOT/'thirdparty/retarget_assets/DexForge'
    config=yaml.safe_load((forge/'forceaware/configs/forceaware.yaml').read_text())
    config['sequence'].update(hand='wuji_right',scene='sdf')
    config['simulator']['exec_dt']=.01;config['contact_physics']['contact_topk']=1
    (output/'forceaware.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    cases={f'{executor}/{method}':dict(status='pending',executor=executor,controller=method) for executor in ['mujoco','comfree'] for method in ['fa_cost','comfree_model','dexforge','current']}
    manifest=dict(sequence='20200709_142802',frames=len(q),common_CA_array=str(ca/'wuji_right_qpos.npy'),
        ca_sha256=hashlib.sha256((ca/'wuji_right_qpos.npy').read_bytes()).hexdigest(),
        definition=dict(fa_cost='Current controller with complete FA objective',comfree_model='Current controller and current cost; ComFree prediction only',dexforge='Original FA optimizer with the corresponding closed-loop executor',current='Current soft-SCM controller and cost'),
        timestep=dict(control=.03,mujoco_execution=.002,comfree_execution=.01,comfree_prediction=.01),cases=cases)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return output


def environment():
    forge=ROOT/'thirdparty/retarget_assets/DexForge'
    return dict(os.environ,COMFREE_WARP_ROOT=str(forge/'forceaware/third_party/comfree_warp'),
        MANO_MODEL_ROOT='/home/zz/GraspSONIC/thirdparty/DexForge/assets/mano',HAND_ASSETS_ROOT=str(forge/'assets'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONUNBUFFERED='1')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    args=p.parse_args()
    if args.prepare:print(prepare(args.output))


if __name__=='__main__':main()
