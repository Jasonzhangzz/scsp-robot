"""Read official DexYCB labels from the authorized GraspSonic data tree.

Only labels, calibration and the grasped object's mesh are downloaded; RGB and
depth streams are unnecessary for MANO retargeting. Physics comes from an
explicit DexForge object-scene template, never from guessed mesh densities.
"""
from pathlib import Path
import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

YCB_OBJECTS=('002_master_chef_can','003_cracker_box','004_sugar_box',
    '005_tomato_soup_can','006_mustard_bottle','007_tuna_fish_can',
    '008_pudding_box','009_gelatin_box','010_potted_meat_can','011_banana',
    '019_pitcher_base','021_bleach_cleanser','024_bowl','025_mug',
    '035_power_drill','036_wood_block','037_scissors','040_large_marker',
    '051_large_clamp','052_extra_large_clamp','061_foam_brick')
REMOTE_DATA='/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/grasp_sonic/data'


class CalibrationLoader(yaml.SafeLoader):
    pass


CalibrationLoader.add_constructor('tag:yaml.org,2002:python/tuple',
    lambda loader,node:loader.construct_sequence(node))


def read_yaml(path):
    return yaml.load(Path(path).read_text(),Loader=CalibrationLoader)


def sync_sequence(cache,subject,sequence,camera=None,host='8.147.67.185',port=1016):
    for name in (subject,sequence):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name):raise ValueError('invalid DexYCB sequence identifier')
    cache=Path(cache).resolve();relative=f'{subject}/{sequence}'
    def copy(source,destination,filters=()):
        destination=cache/destination;destination.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-az',*filters,'-e',f'ssh -p {int(port)} -o BatchMode=yes',
            f'{host}:{REMOTE_DATA}/{source}',str(destination)+'/'],check=True)
    copy(f'{relative}/meta.yml',relative)
    meta=read_yaml(cache/relative/'meta.yml');camera=str(camera or meta['serials'][0])
    if camera not in meta['serials']:raise ValueError('camera not listed in sequence metadata')
    copy(f'{relative}/{camera}/',f'{relative}/{camera}',('--include=labels_*.npz','--exclude=*'))
    copy(f'{relative}/{camera}/aligned_depth_to_color_000000.png',f'{relative}/{camera}')
    copy(f'calibration/intrinsics/{camera}_640x480.yml','calibration/intrinsics')
    for calibration in (f'extrinsics_{meta["extrinsics"]}',f'mano_{meta["mano_calib"][0]}'):
        copy(f'calibration/{calibration}/',f'calibration/{calibration}')
    label=YCB_OBJECTS[int(meta['ycb_ids'][int(meta['ycb_grasp_ind'])])-1]
    copy(f'models/{label}/',f'models/{label}',('--include=*.obj','--include=*.mtl','--include=texture_map.png','--exclude=*'))
    return cache/relative,camera


def fit_table_plane(points,object_vertices,seed=0):
    """Fit the background plane that actually supports the initial object.

    Plane support disambiguates walls from the tabletop. Camera-facing normals
    establish the upward sign; no object-motion target enters this estimation.
    """
    points=np.asarray(points,float);vertices=np.asarray(object_vertices,float)
    rng=np.random.default_rng(seed)
    if len(points)>6000:points=points[rng.choice(len(points),6000,replace=False)]
    best=None
    for _ in range(250):
        tri=points[rng.choice(len(points),3,replace=False)]
        normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);length=np.linalg.norm(normal)
        if length<1e-8:continue
        normal/=length
        if normal@tri.mean(0)>0:normal=-normal
        height=float(tri[0]@normal)
        gap=float(np.min(vertices@normal)-height)
        if abs(gap)>.015:continue
        mask=np.abs(points@normal-height)<.004
        score=int(mask.sum())
        if best is None or score>best[0]:best=(score,mask,normal)
    if best is None or best[0]<100:raise ValueError('cannot identify a depth plane supporting the initial object')
    inliers=points[best[1]];center=inliers.mean(0)
    _,_,vt=np.linalg.svd(inliers-center,full_matrices=False);normal=vt[-1]
    if normal@best[2]<0:normal=-normal
    return dict(normal_camera=normal.tolist(),offset_camera=float(center@normal),
        inliers=len(inliers),rms_m=float(np.sqrt(np.mean(((inliers-center)@normal)**2))))


def table_plane(dataset,sequence,camera,label,initial_object,index):
    import imageio.v3 as imageio
    import trimesh
    depth_path=sequence/camera/'aligned_depth_to_color_000000.png'
    if not depth_path.exists():return None
    depth=np.asarray(imageio.imread(depth_path),float)/1000.
    intrinsics=read_yaml(dataset/f'calibration/intrinsics/{camera}_640x480.yml')['color']
    with np.load(sequence/camera/'labels_000000.npz') as first:mask=first['seg']==0
    v,u=np.mgrid[:depth.shape[0],:depth.shape[1]]
    valid=mask&(depth>.1)&(depth<2.)
    points=np.c_[((u-intrinsics['ppx'])*depth/intrinsics['fx'])[valid],
        ((v-intrinsics['ppy'])*depth/intrinsics['fy'])[valid],depth[valid]]
    mesh=trimesh.load_mesh(dataset/f'models/{label}/textured_simple.obj',process=False)
    pose=initial_object[index]
    vertices=mesh.vertices@Rotation.from_quat(pose[:4]).as_matrix().T+pose[4:]
    return fit_table_plane(points,vertices)


def prepare_sequence(dataset,sequence,camera,scene_template,output):
    dataset=Path(dataset).resolve();sequence=Path(sequence).resolve();output=Path(output).resolve()
    meta=read_yaml(sequence/'meta.yml');camera=str(camera or meta['serials'][0])
    if meta['mano_sides']!=['right']:raise ValueError('only single right-hand DexYCB sequences supported')
    index=int(meta['ycb_grasp_ind']);label=YCB_OBJECTS[int(meta['ycb_ids'][index])-1]
    frames=int(meta['num_frames']);hands=[];objects=[];valid=[]
    for frame in range(frames):
        with np.load(sequence/camera/f'labels_{frame:06d}.npz') as data:
            hands.append(np.asarray(data['pose_m']).reshape(51))
            poses=np.asarray(data['pose_y']).reshape(-1,3,4)
            good=np.linalg.det(poses[:,:3,:3])>.5
            quaternion=np.tile([0.,0.,0.,1.],(len(poses),1))
            quaternion[good]=Rotation.from_matrix(poses[good,:3,:3]).as_quat()
            objects.append(np.c_[quaternion,poses[:,:,3]])
            valid.append(bool(good[index] and np.abs(hands[-1]).sum()>1e-5))
    raw=output/'mano_raw';raw.mkdir(parents=True,exist_ok=True)
    shape=read_yaml(dataset/f'calibration/mano_{meta["mano_calib"][0]}/mano.yml')['betas']
    calibration=read_yaml(dataset/f'calibration/extrinsics_{meta["extrinsics"]}/extrinsics.yml')['extrinsics']
    extrinsic=np.eye(4);extrinsic[:3]=np.asarray(calibration[camera]).reshape(3,4)
    for name,value in (('hand_pose_51',hands),('hand_shape_10',shape),
                       ('object_pose_camera_7',objects),('extrinsics_4x4',extrinsic),('frame_valid',valid)):
        np.save(raw/f'{name}.npy',np.asarray(value))
    plane=table_plane(dataset,sequence,camera,label,objects[0],index)
    metadata=dict(source='dexycb_mano_raw',dataset='DexYCB',
        object=label,object_index=index,hand_side='right',camera=dict(serial=camera),
        source_sequence=str(sequence),source_frame_count=frames)
    if plane is not None:
        metadata['world_up']=(extrinsic[:3,:3].T@plane['normal_camera']).tolist()
        metadata['table_plane']=plane
    (raw/'meta.json').write_text(json.dumps(metadata,indent=2)+'\n')
    tree=ET.parse(scene_template);root=tree.getroot()
    collision=next(g for g in root.findall("worldbody/body[@name='obj']/geom") if g.get('contype','1')!='0')
    mesh_name=collision.get('mesh');mesh=root.find(f"asset/mesh[@name='{mesh_name}']")
    if mesh is None:raise ValueError('DexForge template requires a mesh object collider')
    if label.partition('_')[2] not in mesh_name:
        raise ValueError(f'physics template does not match grasped object {label}')
    for item in root.findall('asset/*'):
        if item.get('file'):item.set('file',str((Path(scene_template).resolve().parent/item.get('file')).resolve()))
    mesh.set('file',str(dataset/f'models/{label}/textured_simple.obj'))
    for texture in root.findall('asset/texture'):
        if texture.get('file'):texture.set('file',str(dataset/f'models/{label}/texture_map.png'))
    scene=output/'scene/leaphand'/f'{label.partition("_")[2]}_mesh.xml'
    scene.parent.mkdir(parents=True,exist_ok=True);tree.write(scene,encoding='utf-8',xml_declaration=True)
    (output/'data_provenance.json').write_text(json.dumps(dict(dataset=str(dataset),
        sequence=str(sequence),camera=camera,object=label,object_index=index,
        physics_template=str(Path(scene_template).resolve())),indent=2)+'\n')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--subject',default='20200709-subject-01');p.add_argument('--sequence',default='20200709_142802')
    p.add_argument('--camera');p.add_argument('--sync',action='store_true')
    p.add_argument('--scene-template',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.sync:sequence,camera=sync_sequence(args.dataset,args.subject,args.sequence,args.camera)
    else:sequence,camera=args.dataset/args.subject/args.sequence,args.camera
    print(prepare_sequence(args.dataset,sequence,camera,args.scene_template,args.output))


if __name__=='__main__':main()
