"""Hold each saved control forecast fixed and replay it in MuJoCo mesh physics."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import mujoco
from examples.retarget.fa_mpc_diagnostics import mujoco_contacts,force_summary


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    model=mujoco.MjModel.from_xml_path('outputs/fa_admm_solver_20261016/sequence/scene/wuji_right/sugar_box_mesh.xml')
    model.opt.timestep=.002;data=mujoco.MjData(model);rollout=np.load(a.run/'rollout.npz')
    body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'obj_joint');body=int(model.jnt_bodyid[body]);oq=26
    trace=[json.loads(line) for line in (a.run/'diagnostics/trace.jsonl').read_text().splitlines()]
    predictions={(r['window'],r['stage']):r for r in trace if r['kind']=='prediction'}
    control=rollout['ctrl_traj'];result=[];last=0;history_error=0.
    deduplicated=copy.copy(model);seen={};duplicates=[]
    for geom in range(model.ngeom):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):continue
        key=(int(model.geom_bodyid[geom]),int(model.geom_type[geom]),int(model.geom_dataid[geom]),
            tuple(model.geom_pos[geom]),tuple(model.geom_quat[geom]),tuple(model.geom_size[geom]))
        if key in seen:
            duplicates.append([seen[key],geom]);deduplicated.geom_contype[geom]=0;deduplicated.geom_conaffinity[geom]=0
        else:seen[key]=geom
    def advance(d,q,v,u,physical_model=model):
        d.qpos[:]=q;d.qvel[:]=v;d.ctrl[:]=u;d.qfrc_applied[:]=0.
        snapshots=[]
        for sub in range(5):
            pose=d.qpos[oq:oq+7].copy();mujoco.mj_step(physical_model,d)
            cs=mujoco_contacts(physical_model,d,body,pose);snapshots.append(dict(substep=sub,contacts=cs,force=force_summary(cs,model.body_mass[body],model.opt.gravity)))
        mujoco.mj_forward(physical_model,d)
        return d.qpos.copy(),d.qvel.copy(),snapshots
    for window in (80,100,110,120,125,130):
        dense=window*3
        if (window,0) not in predictions:continue
        for i in range(last,dense):
            q,v,_=advance(data,rollout['qpos_traj'][i],rollout['qvel_traj'][i],control[i])
            history_error=max(history_error,float(np.max(np.abs(q-rollout['qpos_traj'][i+1]))))
        last=dense;clone=copy.copy(data)
        q=rollout['qpos_traj'][dense].copy();v=rollout['qvel_traj'][dense].copy()
        snapshots=[];saved=np.load(a.run/f'diagnostics/window_{window:04d}.npz')
        for k,u in enumerate(saved['controls']):
            q,v,forces=advance(clone,q,v,u);pred=predictions[window,k]
            snapshots.append(dict(stage=k,time=dense*.01+(k+1)*.01,qpos=q.tolist(),qvel=v.tolist(),
                position_error_mm=float(1000*np.linalg.norm(q[oq:oq+3]-pred['next_qpos'][oq:oq+3])),
                force_average_N=np.mean([r['force']['hand_vertical_N'] for r in forces]).item(),substeps=forces))
        alternate=copy.copy(data);aq=rollout['qpos_traj'][dense].copy();av=rollout['qvel_traj'][dense].copy()
        for u in saved['controls']:aq,av,_=advance(alternate,aq,av,u,deduplicated)
        result.append(dict(window=window,time=dense*.01,predicted_terminal_z_mm=1000*predictions[window,29]['next_qpos'][28],
            mesh_terminal_z_mm=1000*q[28],terminal_position_error_mm=snapshots[-1]['position_error_mm'],steps=snapshots))
        result[-1].update(deduplicated_mesh_terminal_z_mm=1000*aq[28],
            deduplicated_terminal_position_error_mm=float(1000*np.linalg.norm(aq[oq:oq+3]-predictions[window,29]['next_qpos'][oq:oq+3])))
    output=dict(replayed_closed_loop_history_max_qpos_error=history_error,duplicate_mesh_geom_pairs=duplicates,forecasts=result)
    (a.run/'same_control_mesh_replay.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({**output,'forecasts':[{k:v for k,v in r.items() if k!='steps'} for r in result]},indent=2))


if __name__=='__main__':main()
