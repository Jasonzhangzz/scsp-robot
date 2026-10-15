"""Render recorded MuJoCo rollout poses without rerunning optimization."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import imageio.v2 as imageio
import mujoco
import numpy as np


def render(folder,mode='full',output=None,goals=False,demo_overlay=True,demo_alpha=.3):
    folder=Path(folder);output=Path(output or folder/mode/'rollout.mp4')
    output.parent.mkdir(parents=True,exist_ok=True)
    model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
    qadr=model.jnt_qposadr[model.actuator_trnid[:,0]]
    joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'obj_joint')
    object_q=int(model.jnt_qposadr[joint])
    palm=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'right_palm_link')
    with np.load(folder/mode/'rollout.npz') as saved:
        positions=saved['hand_q'];objects=saved['object_pose'];frames=saved['reference_frame']
        demo_frames=saved['demo_frame'] if 'demo_frame' in saved else frames
    targets=np.load(folder/'geometric_pose_targets.npz')['targets'] if goals else None
    dt=json.loads((folder/'config.json').read_text())['command_dt']
    overlay=None
    if demo_overlay and (folder/'demonstration_visual.npz').exists():
        from examples.retarget.visualization import DemonstrationOverlay
        with np.load(folder/'demonstration_visual.npz') as visual:
            overlay=DemonstrationOverlay(folder/'scene.xml',visual['vertices'],visual['faces'],visual['object_pose'],demo_alpha)
    display_model=overlay.model if overlay else model
    renderer=mujoco.Renderer(display_model,height=min(720,model.vis.global_.offheight),width=min(960,model.vis.global_.offwidth))
    camera=mujoco.MjvCamera();camera.distance=.75;camera.azimuth=135;camera.elevation=-25
    try:
        with imageio.get_writer(output,fps=1/dt) as writer:
            for q,pose,frame,demo_frame in zip(positions,objects,frames,demo_frames):
                data.qpos[:]=model.qpos0;data.qpos[qadr]=q;data.qpos[object_q:object_q+7]=pose
                mujoco.mj_forward(model,data)
                camera.lookat[:]=(data.xpos[palm]+pose[:3])/2
                if overlay:overlay.set_camera(camera)
                if overlay:
                    if overlay.update(data,demo_frame):overlay.upload(renderer)
                renderer.update_scene(overlay.data if overlay else data,camera)
                if targets is not None:
                    for point in targets[frame]:
                        scene=renderer.scene
                        if scene.ngeom>=scene.maxgeom:break
                        mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.full(3,.002),point,np.eye(3).ravel(),np.array([.1,1.,.2,.6]))
                        scene.ngeom+=1
                writer.append_data(renderer.render())
    finally:renderer.close()
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path);parser.add_argument('--mode',choices=['ik-only','full'],default='full')
    parser.add_argument('--output',type=Path);parser.add_argument('--goals',action='store_true')
    parser.add_argument('--demo-overlay',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--demo-alpha',type=float,default=.3)
    args=parser.parse_args();print(render(args.folder,args.mode,args.output,args.goals,args.demo_overlay,args.demo_alpha))
