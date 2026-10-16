"""Visualize saved/live FA + contact-search rollouts; no optimization in replay."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

if '--video' in sys.argv or '--check-image' in sys.argv:os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class ContactViewer:
    def __init__(self, experiment, *, passive=True, control_path=None, demo_overlay=True, demo_alpha=.3):
        self.root=Path(experiment)
        self.model=mujoco.MjModel.from_xml_path(str(self.root/'sequence/scene/wuji_right/sugar_box_mesh.xml'))
        self.data=mujoco.MjData(self.model);self.paused=False;self.control_path=control_path
        self.xray=False;self.previous_xray=False
        self.reference=np.load(self.root/'contactaware/object_pose_7.npy')[:,[4,5,6,3,0,1,2]]
        self.overlay=None
        if demo_overlay:
            from examples.retarget.visualization import DemonstrationOverlay
            path=self.root/'demonstration_visual.npz'
            if not path.exists():
                raise FileNotFoundError(f'MANO animation missing: {path}. Prepare the demonstration_visual.npz asset or use --no-demo-overlay.')
            with np.load(path) as visual:
                if len(visual['vertices'])!=len(self.reference) or not np.allclose(visual['object_pose'],self.reference,atol=1e-6,rtol=0.):
                    raise ValueError('MANO overlay does not match this experiment reference')
                self.demo_times=visual['times'].copy() if 'times' in visual else np.arange(len(self.reference))/30
                if self.demo_times.shape!=(len(self.reference),) or not np.all(np.isfinite(self.demo_times)) or np.any(np.diff(self.demo_times)<=0):
                    raise ValueError('MANO overlay times must be strictly increasing')
                self.overlay=DemonstrationOverlay(self.root/'sequence/scene/wuji_right/sugar_box_mesh.xml',
                    visual['vertices'],visual['faces'],visual['object_pose'],demo_alpha)
        self.display_model=self.overlay.model if self.overlay else self.model
        self.display_data=self.overlay.data if self.overlay else self.data
        self.alpha=self.display_model.geom_rgba[:,3].copy()
        self.obj=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'obj')
        g=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'obj_collision')
        mesh=self.model.geom_dataid[g];start=self.model.mesh_vertadr[mesh];count=self.model.mesh_vertnum[mesh]
        vertices=self.model.mesh_vert[start:start+count]@Rotation.from_quat(self.model.geom_quat[g],scalar_first=True).as_matrix().T+self.model.geom_pos[g]
        self.half=(vertices.max(0)-vertices.min(0))/2;self.center=(vertices.max(0)+vertices.min(0))/2
        self.camera=mujoco.MjvCamera();self.camera.lookat[:]=self.reference[:,:3].mean(0)
        self.camera.distance=.6;self.camera.azimuth=45;self.camera.elevation=-25
        self.viewer=None
        if passive:
            from mujoco import viewer as mj_viewer
            self.viewer=mj_viewer.launch_passive(self.display_model,self.display_data,key_callback=self.key)
            with self.viewer.lock():
                self.viewer.cam.lookat[:]=self.camera.lookat
                self.viewer.cam.distance=self.camera.distance;self.viewer.cam.azimuth=45;self.viewer.cam.elevation=-25
            print('Orange translucent hand: MANO motion | cyan object: demonstration | blue: MANO contacts | green: optimized targets | grey: samples | red: observed contacts | purple: ideal planned forces. SPACE: pause; X: transparent physical surfaces.',flush=True)

    def key(self, key):
        if key==32:
            self.paused=not self.paused
            if self.control_path is not None:
                temporary=self.control_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(paused=self.paused)));temporary.replace(self.control_path)
        if key==88:self.xray=not self.xray

    def sync(self):
        if self.previous_xray!=self.xray:
            self.display_model.geom_rgba[:,3]=self.alpha
            moving=(self.display_model.geom_bodyid>0)&(self.display_model.geom_bodyid<self.model.nbody)
            if self.xray:self.display_model.geom_rgba[moving,3]*=.25
            self.previous_xray=self.xray
        if self.viewer is not None:self.viewer.sync()

    @staticmethod
    def point(scene, position, color, radius=.002):
        if scene.ngeom>=scene.maxgeom:return
        geom=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,radius),position,np.eye(3).ravel(),np.asarray(color))
        scene.ngeom+=1

    @staticmethod
    def arrow(scene, start, end, color, width=.001):
        if scene.ngeom>=scene.maxgeom or np.linalg.norm(end-start)<1e-8:return
        geom=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.asarray(color))
        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,width,start,end)
        scene.ngeom+=1

    def decorate(self, scene, seconds, snapshot):
        from examples.retarget.matrix_report import reference
        desired=reference(self.reference,np.array([seconds]))[0]
        goal_rotation=Rotation.from_quat(desired[3:],scalar_first=True).as_matrix()
        if self.overlay is None and scene.ngeom<scene.maxgeom:
            g=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_BOX,self.half,
                desired[:3]+goal_rotation@self.center,goal_rotation.ravel(),np.array([.1,.9,.8,.18]))
            scene.ngeom+=1
        if snapshot is None:return
        r=self.data.xmat[self.obj].reshape(3,3);p=self.data.xpos[self.obj]
        def world(points):return np.asarray(points)@r.T+p
        near=min(2,len(snapshot['mask'])-1);active=snapshot['mask'][near]>0
        for a,b in zip(world(snapshot['template'][near,active]),world(snapshot['z'][near,active])):
            self.point(scene,a,[.1,.35,1.,1.]);self.point(scene,b,[.1,1.,.15,1.])
        if 'search_candidates' in snapshot:
            for candidate in snapshot['search_candidates']:
                for point in world(candidate[0,active]):self.point(scene,point,[.7,.7,.7,.35],.0008)
            targets=world(snapshot['search_points'][0,active]);forces=snapshot['search_forces'][0,active]@r.T
            # Incoming force arrows remain outside the opaque object; their
            # direction is still the force applied to the surface point.
            for point,force in zip(targets,forces):self.arrow(scene,point-.008*force,point,[.85,.2,.9,.85])
            for point in world(snapshot['observed_points'][snapshot['observed_mask']>0]):self.point(scene,point,[1.,.15,.1,1.],.0017)

    def update(self, qpos, qvel, seconds, snapshot=None):
        self.data.qpos[:]=qpos;self.data.qvel[:]=qvel;self.data.time=seconds
        mujoco.mj_forward(self.model,self.data)
        def update_demo():
            if self.overlay is None:return False
            frame=np.interp(seconds,self.demo_times,np.arange(len(self.demo_times)))
            return self.overlay.update(self.data,frame)
        if self.viewer is not None:
            if not self.viewer.is_running():raise KeyboardInterrupt('Viewer closed')
            with self.viewer.lock():
                changed=update_demo()
                self.viewer.user_scn.ngeom=0;self.decorate(self.viewer.user_scn,seconds,snapshot)
            if changed:self.viewer.update_mesh(self.overlay.mesh)
            self.sync()
        else:
            update_demo();self.sync()

    def upload(self, renderer):
        if self.overlay is not None:self.overlay.upload(renderer)

    def close(self):
        if self.viewer is not None:self.viewer.close()


def load_case(case):
    case=Path(case);full=(case/'rollout.npz').exists()
    trajectory=dict(np.load(case/('rollout.npz' if full else 'partial.npz')))
    name='contact_admm.npz' if full else 'partial_contact_admm.npz'
    snapshots={}
    if (case/name).exists():
        with np.load(case/name) as saved:
            indices=saved['window_indices'] if 'window_indices' in saved else np.arange(len(saved['mask']))
            for row,window in enumerate(indices):
                snapshots[int(window)]={k:saved[k][row] for k in saved.files if k!='window_indices'}
    return trajectory,snapshots


def follow_case(experiment, case, *, demo_overlay=True, demo_alpha=.3):
    path=case/'live_state.npz';print(f'Waiting for live state: {path}',flush=True)
    while not path.exists():
        if (case/'viewer_finished.json').exists():raise RuntimeError('Solver ended before publishing a state')
        time.sleep(.1)
    visual=ContactViewer(experiment,control_path=case/'viewer_control.json',demo_overlay=demo_overlay,demo_alpha=demo_alpha)
    stamp=None
    try:
        while visual.viewer.is_running():
            modified=path.stat().st_mtime_ns
            if modified!=stamp:
                with np.load(path) as state:
                    snapshot={k[len('snapshot_'):]:state[k] for k in state.files if k.startswith('snapshot_')}
                    visual.update(state['qpos'],state['qvel'],float(state['seconds']),snapshot or None)
                stamp=modified
            else:visual.sync()
            time.sleep(.02)
    finally:visual.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,default=Path('outputs/fa_admm_solver_20261016'))
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--speed',type=float,default=1.)
    p.add_argument('--video',type=Path);p.add_argument('--check-image',type=Path)
    p.add_argument('--loop',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--follow',action='store_true',help='Follow atomic live states from the solver process')
    p.add_argument('--xray',action='store_true',help='Start with transparent hand/object surfaces; X toggles this')
    p.add_argument('--demo-overlay',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--demo-alpha',type=float,default=.3)
    a=p.parse_args()
    if a.speed<=0:raise ValueError('Positive playback speed required')
    if a.follow:
        follow_case(a.experiment,a.case,demo_overlay=a.demo_overlay,demo_alpha=a.demo_alpha);return
    saved,snapshots=load_case(a.case)
    visual=ContactViewer(a.experiment,passive=not(a.video or a.check_image),demo_overlay=a.demo_overlay,demo_alpha=a.demo_alpha)
    visual.xray=a.xray
    try:
        if a.video or a.check_image:
            import imageio.v2 as imageio
            with mujoco.Renderer(visual.display_model,480,640) as renderer:
                writer=imageio.get_writer(a.video,fps=100/3) if a.video else None
                try:
                    frames=[len(saved['qpos_traj'])-1] if a.check_image else range(0,len(saved['qpos_traj']),3)
                    for i in frames:
                        snap=snapshots.get(max((i-1)//3,0))
                        visual.update(saved['qpos_traj'][i],saved['qvel_traj'][i],i*.01,snap)
                        visual.upload(renderer)
                        renderer.update_scene(visual.display_data,camera=visual.camera)
                        visual.decorate(renderer.scene,i*.01,snap)
                        frame=renderer.render()
                        if writer:writer.append_data(frame)
                        else:imageio.imwrite(a.check_image,frame)
                finally:
                    if writer:writer.close()
        else:
            while visual.viewer.is_running():
                for i,(q,v) in enumerate(zip(saved['qpos_traj'],saved['qvel_traj'])):
                    while visual.paused and visual.viewer.is_running():visual.sync();time.sleep(.02)
                    if not visual.viewer.is_running():return
                    visual.update(q,v,i*.01,snapshots.get(max((i-1)//3,0)));time.sleep(.01/a.speed)
                if not a.loop:break
    except KeyboardInterrupt:pass
    finally:visual.close()


if __name__=='__main__':main()
