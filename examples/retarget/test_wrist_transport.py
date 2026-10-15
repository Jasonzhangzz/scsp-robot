#!/usr/bin/env python3
"""Counterfactual physical test: retain finger servo targets and move the wrist.

Starts each trial from the same actually observed force-closed simulation state.
No Lambda forces, object pose commands, MPC gates or changed physics are used.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from models.wuji_hand import ObjectSurface,WujiHand
from planning.grasp_verification import verify_grasp
from examples.retarget.verify_force_closure import contacts,patch_arrays
from examples.retarget.diagnose_lift import gravity_wrench
from examples.retarget.retarget import write_json


def certificate(hand):
    actual,floor=contacts(hand,hand.data)
    p,n,t,caps=patch_arrays(actual,.01)
    result=verify_grasp(p,n,hand.friction,edges=4,tangent_directions=t,
        external_wrench=gravity_wrench(hand,hand.object_pose()),normal_force_limits=caps)
    if not len(p):result['load_feasible']=False
    force=np.sum([c['force_world'] for c in actual],axis=0) if actual else np.zeros(3)
    return result,force,floor,actual


def run(directory,height=.04):
    directory=Path(directory)
    audit_folder=directory/'full/force_closure_audit'
    audit=json.loads((audit_folder/'certificates.json').read_text())
    candidates=[r for r in audit['physics_steps'] if r['native_certificate']['force_closure']]
    robust=[r for r in candidates if r['robust_certificate']['force_closure']]
    if not candidates:raise ValueError('No actually observed native force-closed state is available')
    chosen=max(robust or candidates,key=lambda r:r['gravity_capacity_ratio'])
    replay=np.load(audit_folder/'replayed_prefix.npz');commands=replay['controls']
    semantic=np.load(directory/'semantic_contacts.npz');geometric=np.load(directory/'wuji_q.npy')
    config=json.loads((directory/'config.json').read_text())
    xml=directory/'scene.xml';tree=ET.parse(xml).getroot()
    collider=next(g for g in tree.findall("worldbody/body[@name='obj']/geom") if g.get('contype','1')!='0')
    mesh=tree.find(f"asset/mesh[@name='{collider.get('mesh')}']")
    surface=ObjectSurface((xml.parent/mesh.get('file')).resolve(),scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' '),convex=True)
    hand=WujiHand(xml,surface)
    hand.set_state(geometric[0],semantic['object_pose'][0],hand.data)
    hand.servo=geometric[0].copy();hand.data.ctrl[:]=hand.servo
    dt=config['command_dt'];substeps=round(dt/hand.model.opt.timestep)
    for cycle in range(chosen['cycle']+1):
        hand.servo=commands[cycle].copy();hand.data.ctrl[:]=hand.servo
        for substep in range(substeps):
            hand.data.qfrc_applied[:]=0;mujoco.mj_step(hand.model,hand.data)
            if substep==substeps-1:mujoco.mj_forward(hand.model,hand.data)
            if cycle==chosen['cycle'] and substep==chosen['substep']:break
    mujoco.mj_forward(hand.model,hand.data)
    np.testing.assert_allclose(hand.object_pose(),chosen['object_pose'],atol=1e-6,rtol=0)
    initial_certificate,_,_,_=certificate(hand)
    if not initial_certificate['force_closure']:raise RuntimeError('Replayed start is not force closed')
    specification=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(hand.model,specification));mujoco.mj_getState(hand.model,hand.data,state,specification)
    initial_servo=hand.servo.copy();initial_q=hand.data.qpos[hand.qadr].copy();initial_pose=hand.object_pose()
    z=semantic['object_pose'][:,2];peak=int(np.argmax(z))
    rising=np.flatnonzero(z[:peak+1]>z[0]+.005)
    if not len(rising):raise ValueError('Demonstration has no lift segment')
    start=max(0,int(rising[0])-1);times=semantic['times'][start:peak+1]-semantic['times'][start]
    path=geometric[start:peak+1,:6]-geometric[start,:6]
    output=directory/'full/wrist_only_test';output.mkdir(parents=True,exist_ok=True)
    write_json(output/'start.json',dict(cycle=chosen['cycle'],substep=chosen['substep'],
        selected_robust=bool(robust),initial_certificate=initial_certificate,initial_servo=initial_servo,
        initial_hand_q=initial_q,initial_object_pose=initial_pose,demo_start_frame=start,demo_peak_frame=peak))
    summaries={}
    modes=['hold','vertical_slow','vertical_immediate','demo','demo_immediate','demo_slow']
    if chosen['reference_frame']<peak:modes.append('demo_remaining_immediate')
    for mode in modes:
        mujoco.mj_setState(hand.model,hand.data,state,specification);mujoco.mj_forward(hand.model,hand.data)
        hand.servo=initial_servo.copy()
        scale=3. if mode=='demo_slow' else 1.
        motion_times,motion_path=times,path
        if mode=='demo_remaining_immediate':
            frame=chosen['reference_frame']
            motion_times=semantic['times'][frame:peak+1]-semantic['times'][frame]
            motion_path=geometric[frame:peak+1,:6]-geometric[frame,:6]
        duration=max(float(motion_times[-1])*scale,.2) if mode.startswith('demo') else 1.
        settle,hold=(0. if mode.endswith('immediate') else .3),.4
        records=[];qhistory=[];poses=[];controls=[];frames=[]
        for step in range(int(np.ceil((settle+duration+hold)/dt))):
            elapsed=(step+1)*dt;motion=max(0.,elapsed-settle)
            target=initial_servo.copy();delta=np.zeros(6)
            if mode.startswith('vertical'):
                s=np.clip(motion/duration,0,1);delta[2]=height*(10*s**3-15*s**4+6*s**5)
            elif mode.startswith('demo'):
                delta=np.array([np.interp(motion/scale,motion_times,motion_path[:,j]) for j in range(6)])
            target[:6]+=delta
            target=np.clip(target,hand.lower,hand.upper)
            hand.execute(target,dt)
            cert,force,floor,actual=certificate(hand)
            q=hand.data.qpos[hand.qadr].copy();pose=hand.object_pose()
            collision=hand.collision_report(hand.data,{c['body'] for c in actual})
            records.append(dict(time=elapsed,wrist_target_delta=delta,wrist_actual_delta=q[:6]-initial_q[:6],
                object_delta=pose[:3]-initial_pose[:3],force_closure=cert['force_closure'],
                load_feasible=cert['load_feasible'],hand_force_world=force,floor_force_world=floor,
                finger_drift_rad=float(np.sqrt(np.mean((q[6:]-initial_q[6:])**2))),
                finger_tracking_error_rad=float(np.sqrt(np.mean((q[6:]-initial_servo[6:])**2))),
                normal_force_sum_N=float(sum(c['normal_force'] for c in actual)),
                contact_count=len(actual),penetration=collision['max_penetration']))
            qhistory.append(q);poses.append(pose);controls.append(target)
            frames.append(chosen['reference_frame'])
        tail=records[-max(1,round(hold/dt)):]
        summary=dict(initial_native_force_closed=True,mode=mode,cycles=len(records),
            wrist_command_lift_m=float(records[-1]['wrist_target_delta'][2]),
            wrist_actual_lift_m=float(records[-1]['wrist_actual_delta'][2]),
            final_object_lift_m=float(records[-1]['object_delta'][2]),
            max_object_lift_m=float(max(r['object_delta'][2] for r in records)),
            min_object_lift_in_final_hold_m=float(min(r['object_delta'][2] for r in tail)),
            force_closed_cycles=sum(r['force_closure'] for r in records),
            closed_and_load_feasible_cycles=sum(r['force_closure'] and bool(r['load_feasible']) for r in records),
            still_force_closed_before_motion=(records[max(0,round(settle/dt)-1)]['force_closure'] if settle else True),
            final_hand_upward_force_N=float(np.mean([r['hand_force_world'][2] for r in tail])),
            final_floor_upward_force_N=float(np.mean([r['floor_force_world'][2] for r in tail])),
            max_finger_tracking_error_rad=float(max(r['finger_tracking_error_rad'] for r in records)),
            max_penetration_m=float(max(r['penetration'] for r in records)),
            max_finger_drift_rad=float(max(r['finger_drift_rad'] for r in records)))
        destination=output/mode;destination.mkdir(exist_ok=True)
        np.testing.assert_allclose(np.asarray(controls)[:,6:],
            np.broadcast_to(initial_servo[6:],(len(controls),hand.nu-6)),atol=1e-12,rtol=0)
        write_json(destination/'metrics.json',summary);write_json(destination/'diagnostics.json',records)
        np.savez_compressed(destination/'rollout.npz',hand_q=qhistory,controls=controls,object_pose=poses,reference_frame=frames)
        summaries[mode]=summary
    write_json(output/'summary.json',summaries)
    return summaries


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--height',type=float,default=.04)
    args=parser.parse_args()
    if not np.isfinite(args.height) or args.height<=0:parser.error('height must be positive')
    print(json.dumps(run(args.directory,args.height),indent=2))
