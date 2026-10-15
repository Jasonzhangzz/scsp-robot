#!/usr/bin/env python3
"""Continue a saved physical grasp with contact-point MPC, without Lambda forces.

Replay controls to a specified cycle, keep that frame's semantic reference, and
request a small vertical lift. This isolates grip acquisition from trajectory
timing. It is a diagnostic continuation, not a full retargeting benchmark.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np
from models.wuji_hand import ObjectSurface,WujiHand,MaterialPoint
from planning.wuji_retarget_mpc import WujiOnlineMPC
from examples.retarget.test_wrist_transport import certificate
from examples.retarget.verify_force_closure import patch_arrays,gravity_capacity,longest_run
from examples.retarget.diagnose_lift import gravity_wrench
from examples.retarget.retarget import write_json


def run(directory,cycle=37,steps=50,height=.01,output=None,native_force_feedback=False,contact_penetration_limit=None,regrasp=True):
    directory=Path(directory);xml=directory/'scene.xml';tree=ET.parse(xml).getroot()
    config=json.loads((directory/'config.json').read_text())
    collider=next(g for g in tree.findall("worldbody/body[@name='obj']/geom") if g.get('contype','1')!='0')
    mesh=tree.find(f"asset/mesh[@name='{collider.get('mesh')}']")
    surface=ObjectSurface((xml.parent/mesh.get('file')).resolve(),scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' '),convex=True)
    hand=WujiHand(xml,surface);semantic=np.load(directory/'semantic_contacts.npz');q=np.load(directory/'wuji_q.npy')
    inherited=config.get('contact_penetration_limit')
    hand.contact_penetration_limit=contact_penetration_limit if contact_penetration_limit is not None else (.003 if inherited is None else inherited)
    saved=np.load(directory/'full/rollout.npz');dt=config['command_dt']
    hand.set_state(q[0],semantic['object_pose'][0],hand.data);hand.servo=q[0].copy()
    for target in saved['controls'][:cycle+1]:hand.execute(target,dt)
    frame=int(saved['reference_frame'][cycle]);active=semantic['mask'][frame].astype(float)
    materials=[MaterialPoint(int(b),l,n,str(f),str(r)) for b,l,n,f,r in zip(semantic['body'],semantic['local'],semantic['local_normal'],semantic['finger'],semantic['region'])]
    qref=q[frame];anchors=semantic['anchors'][frame];inward=semantic['normals'][frame]
    rates=json.loads((directory/'full/hand_rate_limits.json').read_text())['rate_limits']
    mpc=WujiOnlineMPC(hand,len(materials),dt=dt,horizon=4,rates=rates,native_force_feedback=native_force_feedback,regrasp=regrasp)
    initial=hand.object_pose();goal=initial.copy();goal[2]+=height
    folder=Path(output) if output is not None else directory/'full/load_support_continuation'
    folder.mkdir(parents=True,exist_ok=True)
    records=[];controls=[];poses=[];qs=[]
    with (folder/'diagnostics.jsonl').open('w') as log:
        for step in range(steps):
            target,report=mpc.plan(goal,qref,materials,anchors,anchors,inward,active,lambda_valid=False)
            hand.execute(target,dt);cert,force,floor,actual=certificate(hand)
            p,n,t,caps=patch_arrays(actual,.01)
            capacity=gravity_capacity(p,n,t,hand.friction,caps,gravity_wrench(hand,hand.object_pose()))
            row=dict(step=step,object_pose=hand.object_pose(),force=force,floor_force=floor,
                gravity_capacity_ratio=capacity,certificate=cert,mpc=report,
                actual_contacts=actual,finger_contact_gaps=hand.finger_contact_gaps())
            from examples.retarget.retarget import jsonable
            row=jsonable(row);records.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            controls.append(target);poses.append(hand.object_pose());qs.append(hand.data.qpos[hand.qadr].copy())
            if step%5==0:print(f'step={step} closure={cert["force_closure"]} load={cert["load_feasible"]} capacity={capacity:.3f} lift_mm={1000*(hand.object_pose()[2]-initial[2]):.3f} source={report["selected_control"]}',flush=True)
    np.savez_compressed(folder/'rollout.npz',controls=controls,object_pose=poses,hand_q=qs)
    summary=dict(start_cycle=cycle,reference_frame=frame,steps=steps,physics_unchanged=True,external_wrench_target=False,
        online_regrasp=regrasp,regrasp_attempts=max((x['mpc'].get('regrasp',{}).get('attempts',0) for x in records),default=0),
        active_contact_collision_limit_m=hand.contact_penetration_limit,
        lift_reference_released=any(x['mpc'].get('transport_active',False) for x in records),
        max_lift_m=float(max(x['object_pose'][2]-initial[2] for x in records)),
        max_gravity_capacity_ratio=max(x['gravity_capacity_ratio'] for x in records),
        longest_closure_s=dt*longest_run([x['certificate']['force_closure'] for x in records]),
        longest_load_support_s=dt*longest_run([x['certificate']['load_feasible'] for x in records]))
    write_json(folder/'summary.json',summary);print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--cycle',type=int,default=37);parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--height',type=float,default=.01);parser.add_argument('--output',type=Path)
    parser.add_argument('--native-force-feedback',action='store_true')
    parser.add_argument('--contact-penetration-limit',type=float)
    parser.add_argument('--regrasp',action=argparse.BooleanOptionalAction,default=True);args=parser.parse_args()
    run(args.directory,args.cycle,args.steps,args.height,args.output,args.native_force_feedback,args.contact_penetration_limit,args.regrasp)
