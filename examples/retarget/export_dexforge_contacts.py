#!/usr/bin/env python3
"""Replay one DexForge result to export its native contacts, without optimization.

Run in DexForge's pi0 environment. The original solver and assets are imported
from --dexforge-root; this script does not modify either CA or FA.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def rotation(q):
    result=np.empty(9)
    mujoco.mju_quat2Mat(result,np.asarray(q,dtype=float))
    return result.reshape(3,3)


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def contact_record(model,body,point,frame,force,gap,pose):
    r=rotation(pose[3:])
    return dict(body_name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(body)),
        point_object=(point-pose[:3])@r,normal_object=frame[0]@r,
        tangents_object=frame[1:]@r,force_world=force,
        normal_force=float(frame[0]@force),gap=float(gap))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dexforge-root',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--mujoco-dt',type=float,help='Defaults to the FA executor timestep for a matched replay.')
    parser.add_argument('--mujoco-exclusions-from',type=Path,
        help='Use the current controller scene structural collision exclusions in the matched mesh replay.')
    args=parser.parse_args();root=args.dexforge_root.resolve();result=args.result.resolve()
    sys.path.insert(0,str(root/'forceaware'))
    from forceaware.config import load_retarget_config
    from forceaware.runtime import setup_runtime
    saved=np.load(result/'forceaware/rollout.npz')
    metadata=json.loads((result/'forceaware/config.json').read_text())
    cfg=load_retarget_config(result/'sequence',result/'forceaware_config.yaml',
        hand=metadata['sequence']['hand'],contactaware_dir=result/'contactaware')
    setup_runtime(cfg)
    import comfree_warp
    from forceaware.workflow import _load_runtime_models
    from forceaware.targets import build_index
    from forceaware.executor import build_action_executor
    models=_load_runtime_models(cfg,comfree_warp);model=models.executor.cpu_model
    index=build_index(model);executor=build_action_executor(models.executor,index,cfg)
    params=executor.compiled.base.contact_rows.parameters
    pairs=params.contact_geom.numpy();bodies=model.geom_bodyid[pairs]
    row_ids=params.row_contact.numpy()[:params.contact_row_count]
    edges=params.row_edge.numpy()[:params.contact_row_count]
    mu=params.contact_friction.numpy()
    scale=float(params.friction_scale.numpy()[0])
    obj=index.object_body;objq=index.object_qpos
    object_pairs=np.flatnonzero(np.any(bodies==obj,axis=1))
    guide=np.load(result/'contactaware/contact_guidance.npz')
    def guidance(frame,pose,positions,matrices):
        f=int(np.clip(round(float(frame)),0,len(guide['contact_mask'])-1));r=rotation(pose[3:]);items=[]
        for k in np.flatnonzero(guide['contact_mask'][f]>0):
            body=int(guide['hand_query_body_ids'][f,k])
            point=positions[body]+matrices[body]@guide['hand_query_local_pos'][f,k]
            items.append(dict(body_name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body),
                point_object=guide['contact_pos_obj'][f,k],hand_point_object=(point-pose[:3])@r))
        return items
    maximum=0.;force_error=0.;qpos=saved['qpos_traj'][0].copy();qvel=saved['qvel_traj'][0].copy()
    with (result/'forceaware/contact_diagnostics.jsonl').open('w') as stream:
        for step,control in enumerate(saved['ctrl_traj']):
            pose=qpos[objq:objq+7].copy()
            execution=executor.execute_action_sequence(qpos,qvel,actions=control[None])
            workspace=executor.workspaces[(executor.substeps-1)%len(executor.workspaces)]
            collision=workspace.base.collision.contacts
            distances=collision.distance.numpy()[0];active=collision.active.numpy()[0]
            positions=collision.position.numpy()[0];frames=collision.frame.numpy()[0]
            coefficients=workspace.contact.output.force.numpy()[0][:params.contact_row_count]
            contacts=[];floor=np.zeros(3)
            for c in object_pairs:
                if not active[c]:continue
                sign=-1. if bodies[c,0]==obj else 1.
                body=int(bodies[c,1] if bodies[c,0]==obj else bodies[c,0])
                force=np.zeros(3)
                for row in np.flatnonzero(row_ids==c):
                    axis=int(edges[row])//2;side=1. if int(edges[row])%2==0 else -1.
                    direction=frames[c,0].copy()
                    if axis<2:direction+=side*mu[c,axis]*scale*frames[c,axis+1]
                    force+=sign*coefficients[row]*direction
                if body==0:floor+=force;continue
                if distances[c]>.0001 and np.linalg.norm(force)<.001:continue
                frame=frames[c].copy();frame[0]*=sign
                contacts.append(contact_record(model,body,positions[c],frame,force,distances[c],pose))
            frame=float(saved['qpos_frames'][step])
            item=dict(step=step,time=step*cfg.simulator.time_grid.exec_dt,reference_frame=frame,
                pose=pose,actual_contacts=contacts,floor_force_world=floor,
                guidance=guidance(frame,pose,workspace.base.pose.body_position.numpy()[0],
                    workspace.base.pose.body_matrix.numpy()[0]))
            objdof=int(model.jnt_dofadr[model.body_jntadr[obj]])
            total=floor+np.asarray([c['force_world'] for c in contacts]).reshape(-1,3).sum(0)
            constraint=workspace.contact.output.constraint_force.numpy()[0,objdof:objdof+3]
            force_error=max(force_error,float(np.max(np.abs(total-constraint))))
            stream.write(json.dumps(item,default=serial)+'\n')
            qpos,qvel=execution.next_qpos,execution.next_qvel
            maximum=max(maximum,float(np.max(np.abs(qpos-saved['qpos_traj'][step+1]))))
    if maximum>1e-4:raise RuntimeError(f'ComFree control replay diverged: max qpos error {maximum}')
    if force_error>1e-3:raise RuntimeError(f'Contact force reconstruction differs from generalized force: {force_error}')
    audit=dict(replay_max_qpos_error=maximum,object_mass=float(model.body_mass[obj]),
        object_com=model.body_ipos[obj],object_inertia=model.body_inertia[obj],gravity=model.opt.gravity,
        execution_dt=cfg.simulator.time_grid.exec_dt,
        contact_friction=sorted(set(float(x) for x in mu[:,0] if x>0)),
        hand_kp=model.actuator_gainprm[index.hand_ctrl,0],
        hand_force_ranges=model.actuator_forcerange[index.hand_ctrl],
        scene=str(cfg.sequence.xml),force_source='ComFree constraint-row forces',
        force_reconstruction_max_error_N=force_error,
        contact_time='pre-step',input=str(result/'forceaware/rollout.npz'))
    (result/'forceaware/contact_audit.json').write_text(json.dumps(audit,default=serial,indent=2)+'\n')
    # Also inspect the independent native MuJoCo control replay, in its own scene.
    mesh=cfg.sequence.xml.with_name(cfg.sequence.xml.stem.rsplit('_',1)[0]+'_mesh.xml')
    if args.mujoco_exclusions_from is not None:
        scene=ET.parse(mesh).getroot();contact=scene.find('contact')
        if contact is None:contact=ET.SubElement(scene,'contact')
        reference=ET.parse(args.mujoco_exclusions_from).getroot()
        for exclude in reference.findall('contact/exclude'):
            ET.SubElement(contact,'exclude',**exclude.attrib)
        for element in scene.iter():
            if element.get('file'):element.set('file',str((mesh.parent/element.get('file')).resolve()))
        native=mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding='unicode'))
    else:native=mujoco.MjModel.from_xml_path(str(mesh))
    data=mujoco.MjData(native)
    native_dt=args.mujoco_dt or cfg.simulator.time_grid.exec_dt
    control_dt=cfg.simulator.time_grid.mpc_dt
    substeps=round(control_dt/native_dt)
    if substeps<1 or not np.isclose(substeps*native_dt,control_dt):raise ValueError('MuJoCo dt must divide FA control sample dt')
    native.opt.timestep=native_dt
    data.qpos[:]=saved['qpos_traj'][0];data.qvel[:]=saved['qvel_traj'][0]
    data.ctrl[index.hand_ctrl]=saved['ctrl_traj'][0]
    mujoco.mj_forward(native,data)
    with (result/'forceaware/mujoco_contact_diagnostics.jsonl').open('w') as stream:
        for step,control in enumerate(saved['ctrl_traj']):
            pose=data.qpos[objq:objq+7].copy();contacts=[];floor=np.zeros(3)
            for i,c in enumerate(data.contact):
                pair=native.geom_bodyid[[c.geom1,c.geom2]]
                if obj not in pair:continue
                sign=-1. if pair[0]==obj else 1.;body=int(pair[1] if pair[0]==obj else pair[0])
                local=np.zeros(6);mujoco.mj_contactForce(native,data,i,local)
                force=sign*c.frame.reshape(3,3).T@local[:3]
                if body==0:floor+=force;continue
                if c.dist>.0001 and np.linalg.norm(force)<.001:continue
                frame=c.frame.reshape(3,3).copy();frame[0]*=sign
                contacts.append(contact_record(native,body,c.pos,frame,force,c.dist,pose))
            frame=float(saved['qpos_frames'][step])
            item=dict(step=step,time=step*control_dt,reference_frame=frame,pose=pose,
                actual_contacts=contacts,floor_force_world=floor,
                guidance=guidance(frame,pose,data.xpos,data.xmat.reshape(-1,3,3)))
            stream.write(json.dumps(item,default=serial)+'\n')
            data.ctrl[index.hand_ctrl]=control
            for _ in range(substeps):mujoco.mj_step(native,data)
            mujoco.mj_forward(native,data)
    audit['object_mass']=float(native.body_mass[obj]);audit['scene']=str(mesh)
    audit['object_inertia']=native.body_inertia[obj];audit['execution_dt']=native_dt
    audit['structural_exclusions_reference']=str(args.mujoco_exclusions_from) if args.mujoco_exclusions_from else None
    audit['contact_friction']=sorted(set(float(c) for c in native.geom_friction[:,0]))
    audit['force_source']='MuJoCo mj_contactForce';audit.pop('replay_max_qpos_error')
    (result/'forceaware/mujoco_contact_audit.json').write_text(json.dumps(audit,default=serial,indent=2)+'\n')
    # A geometry-only query at IDENTICAL FA hand and object poses separates
    # collision representation differences from accumulated replay divergence.
    ghost=mujoco.MjData(native)
    with (result/'forceaware/same_state_mesh_geometry.jsonl').open('w') as stream:
        for step,q in enumerate(saved['qpos_traj'][:-1]):
            ghost.qpos[:]=q;mujoco.mj_forward(native,ghost);pose=q[objq:objq+7];r=rotation(pose[3:]);items=[]
            for c in ghost.contact:
                pair=native.geom_bodyid[[c.geom1,c.geom2]]
                if obj not in pair or c.dist>.0001:continue
                sign=-1. if pair[0]==obj else 1.;body=int(pair[1] if pair[0]==obj else pair[0])
                if body==0:continue
                items.append(dict(body_name=mujoco.mj_id2name(native,mujoco.mjtObj.mjOBJ_BODY,body),
                    point_object=(c.pos-pose[:3])@r,normal_object=sign*c.frame[:3]@r,gap=float(c.dist)))
            stream.write(json.dumps(dict(step=step,reference_frame=float(saved['qpos_frames'][step]),
                contacts=items,scope='geometry query only; no dynamic execution or force claim'),default=serial)+'\n')
    print(json.dumps(dict(exported=str(result),max_qpos_replay_error=maximum)),flush=True)


if __name__=='__main__':main()
