#!/usr/bin/env python3
"""Compare actual contacts on a shared DexYCB sequence in object coordinates.

All methods use the same hull, wrench origin/radius and GWS calculation. Native
friction and a common friction value are both reported. Never extrapolate a
stalled controller to demonstration frames it did not reach.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from scipy.spatial import ConvexHull,QhullError

from models.wuji_hand import ObjectSurface,WujiHand,rotation
from planning.grasp_verification import contact_wrench_rays
from examples.retarget.retarget import write_json


def gws_margin(points,normals,tangents,friction,center,radius):
    """Signed minimum hull-facet margin; positive means six-D force closure.

    Forces have unit Euclidean norm and moments are divided by the common
    object radius. Duplicating contact rays does not change this metric.
    """
    if len(points)<3:return 0.
    normals=np.asarray(normals,float).copy();tangents=np.asarray(tangents,float).copy()
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    # Warp saves float32 frames. Remove rounding drift while retaining each
    # native pyramid's orientation; do not invent a different tangent gauge.
    if np.max(np.abs(np.einsum('nki,nli->nkl',np.concatenate((normals[:,None],tangents),axis=1),
        np.concatenate((normals[:,None],tangents),axis=1))-np.eye(3)))>1e-4:
        raise ValueError('Exported contact frames are not approximately orthonormal')
    tangents[:,0]-=np.einsum('ni,ni->n',tangents[:,0],normals)[:,None]*normals
    tangents[:,0]/=np.linalg.norm(tangents[:,0],axis=1)[:,None]
    tangents[:,1]-=np.einsum('ni,ni->n',tangents[:,1],normals)[:,None]*normals
    tangents[:,1]-=np.einsum('ni,ni->n',tangents[:,1],tangents[:,0])[:,None]*tangents[:,0]
    tangents[:,1]/=np.linalg.norm(tangents[:,1],axis=1)[:,None]
    rays=contact_wrench_rays(np.asarray(points)-center,normals,friction,edges=4,
        tangent_directions=tangents)
    rays[3:]/=radius
    rays/=np.linalg.norm(rays[:3],axis=0)
    if np.linalg.matrix_rank(rays,tol=1e-8)<6:return 0.
    try:return float(np.min(-ConvexHull(rays.T).equations[:,-1]))
    except QhullError:return 0.


def nearest_region_errors(contacts,targets):
    errors=[];unmatched=0
    for c in contacts:
        candidates=[t['point_object'] for t in targets if t['body_name']==c['body_name']]
        if candidates:errors.append(float(np.min(np.linalg.norm(np.asarray(candidates)-c['point_object'],axis=1))))
        else:unmatched+=1
    missing=sum(not any(c['body_name']==t['body_name'] for c in contacts) for t in targets)
    return errors,unmatched,missing


def sample_metrics(row,targets,friction,center,radius,weight,common_mu):
    contacts=[c for c in row['actual_contacts'] if c['gap']<=.0001 or c['normal_force']>.001]
    points=np.asarray([c['point_object'] for c in contacts]).reshape(-1,3)
    normals=np.asarray([c['normal_object'] for c in contacts]).reshape(-1,3)
    tangents=np.asarray([c['tangents_object'] for c in contacts]).reshape(-1,2,3)
    loaded=np.array([c['normal_force']>.01 for c in contacts],bool)
    epsilon=gws_margin(points,normals,tangents,common_mu,center,radius)
    native=gws_margin(points,normals,tangents,friction,center,radius)
    loaded_epsilon=gws_margin(points[loaded],normals[loaded],tangents[loaded],common_mu,center,radius)
    error,unmatched,missing=nearest_region_errors(contacts,targets)
    force=np.asarray([c['force_world'] for c in contacts]).reshape(-1,3).sum(0)
    clearance=float(row['clearance'])
    return dict(step=row['step'],time=row['time'],frame=row['reference_frame'],
        contacts=len(contacts),loaded_contacts=int(loaded.sum()),epsilon_common=epsilon,
        epsilon_native=native,loaded_epsilon_common=loaded_epsilon,
        force_closed=epsilon>1e-7,loaded_force_closed=loaded_epsilon>1e-7,
        region_error_mm=float(np.mean(error)*1000) if error else None,
        unmatched_contacts=unmatched,missing_target_regions=missing,targets=len(targets),
        hand_vertical_N=float(force[2]),weight_N=weight,
        floor_vertical_N=float(row['floor_force_world'][2]),clearance_mm=1000*clearance,
        z_span_mm=1000*float(np.ptp(points[:,2])) if len(points) else 0.)


def read_jsonl(path):
    text=Path(path).read_text()
    # A running controller may be appending its last JSON record.
    return [json.loads(line) for line in text[:text.rfind('\n')+1].splitlines()]


def ours_records(root,surface,allow_partial=False):
    hand=WujiHand(root/'scene.xml',surface);semantic=np.load(root/'semantic_contacts.npz')
    geometric=np.load(root/'wuji_q.npy')
    logs=read_jsonl(root/'full/diagnostics.jsonl');cfg=json.loads((root/'config.json').read_text())
    if (root/'full/rollout.npz').exists():controls=np.load(root/'full/rollout.npz')['controls']
    elif allow_partial:
        from examples.retarget.verify_force_closure import recover_commands
        controls=recover_commands(logs,geometric[0],hand.lower,hand.upper)
    else:raise ValueError('Current-method run is incomplete; use --allow-partial for an explicit prefix audit')
    hand.set_state(geometric[0],semantic['object_pose'][0],hand.data);hand.servo=geometric[0].copy()
    rows=[];error=0.
    for target,log in zip(controls,logs):
        hand.execute(target,cfg['command_dt']);pose=hand.object_pose();r=rotation(pose)
        actual=[];floor=np.zeros(3)
        for i,c in enumerate(hand.data.contact):
            pair=hand.model.geom_bodyid[[c.geom1,c.geom2]]
            if hand.obj not in pair:continue
            sign=-1. if pair[0]==hand.obj else 1.;body=int(pair[1] if pair[0]==hand.obj else pair[0])
            local=np.zeros(6);mujoco.mj_contactForce(hand.model,hand.data,i,local)
            force=sign*c.frame.reshape(3,3).T@local[:3]
            if body==0:floor+=force;continue
            if body not in hand.hand_bodies or (c.dist>.0001 and np.linalg.norm(force)<.001):continue
            normal=sign*c.frame[:3]
            actual.append(dict(body_name=mujoco.mj_id2name(hand.model,mujoco.mjtObj.mjOBJ_BODY,body),
                point_object=(c.pos-pose[:3])@r,normal_object=normal@r,
                tangents_object=c.frame.reshape(3,3)[1:]@r,force_world=force,
                normal_force=float(normal@force),gap=float(c.dist)))
        error=max(error,float(np.max(np.abs(pose-np.asarray(log['object_pose'])))))
        rows.append(dict(step=log['step'],time=(log['step']+1)*cfg['command_dt'],
            reference_frame=int(semantic['frame_ids'][log['reference_frame']]),pose=pose,
            actual_contacts=actual,floor_force_world=floor,
            clearance=float((surface.mesh.vertices@r.T+pose[:3])[:,2].min())))
    if error>1e-6:raise RuntimeError(f'Current-method command replay mismatch: {error}')
    return rows,hand,semantic,error


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ours',type=Path,required=True);p.add_argument('--dexforge',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--common-friction',type=float,default=.5)
    p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    tree=ET.parse(args.ours/'scene.xml').getroot()
    obj=tree.find("worldbody/body[@name='obj']")
    collider=next(g for g in obj.findall('geom') if g.get('contype','1')!='0')
    mesh=tree.find(f"asset/mesh[@name='{collider.get('mesh')}']")
    surface=ObjectSurface(mesh.get('file'),scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' '),convex=True)
    ours,hand,sem,replay_error=ours_records(args.ours,surface,args.allow_partial)
    center=hand.model.body_ipos[hand.obj];radius=float(np.linalg.norm(surface.mesh.vertices-center,axis=1).max())
    mano={int(frame):[dict(body_name=mujoco.mj_id2name(hand.model,mujoco.mjtObj.mjOBJ_BODY,int(sem['body'][k])),
        point_object=sem['anchors'][i,k]) for k in np.flatnonzero(sem['mask'][i])] for i,frame in enumerate(sem['frame_ids'])}
    trajectories={'ours':ours};audits={'ours':dict(object_mass=float(hand.model.body_mass[hand.obj]),
        hand_object_friction=hand.friction,hand_kp=hand.model.actuator_gainprm[:,0],
        hand_force_ranges=hand.model.actuator_forcerange)}
    for name,prefix in [('DexForge',''),('DexForge_MuJoCo','mujoco_')]:
        path=args.dexforge/'forceaware'/f'{prefix}contact_diagnostics.jsonl'
        rows=read_jsonl(path)
        for row in rows:
            pose=np.asarray(row['pose']);row['clearance']=float((surface.mesh.vertices@rotation(pose).T+pose[:3])[:,2].min())
        trajectories[name]=rows
        audits[name]=json.loads((args.dexforge/'forceaware'/f'{prefix}contact_audit.json').read_text())
    metrics={};contact_table=[];target_table=[]
    for name,rows in trajectories.items():
        mu=audits[name].get('hand_object_friction',max(audits[name].get('contact_friction',[.5])))
        weight=audits[name]['object_mass']*9.81;metrics[name]=[]
        for row in rows:
            frame=int(np.clip(round(row['reference_frame']),min(mano),max(mano)))
            values=sample_metrics(row,mano.get(frame,[]),mu,center,radius,weight,args.common_friction)
            metrics[name].append(values)
            for c in row['actual_contacts']:
                contact_table.append(dict(method=name,step=row['step'],time=row['time'],frame=row['reference_frame'],
                    body=c['body_name'],x=c['point_object'][0],y=c['point_object'][1],z=c['point_object'][2],
                    normal_force=c['normal_force'],gap=c['gap']))
            for target in row.get('guidance',[]):
                error,_,_=nearest_region_errors([target],mano.get(frame,[]))
                target_table.append(dict(method=name,frame=row['reference_frame'],body=target['body_name'],
                    x=target['point_object'][0],y=target['point_object'][1],z=target['point_object'][2],
                    displacement_to_mano_mm=1000*error[0] if error else None))
    for frame,targets in mano.items():
        for t in targets:target_table.append(dict(method='MANO',frame=frame,body=t['body_name'],
            x=t['point_object'][0],y=t['point_object'][1],z=t['point_object'][2],displacement_to_mano_mm=0.))
    def save_csv(name,rows):
        if not rows:return
        with (args.output/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save_csv('samples.csv',[dict(method=name,**r) for name,rows in metrics.items() for r in rows])
    save_csv('contact_points.csv',contact_table);save_csv('target_points.csv',target_table)
    snapshots=[]
    for name,rows in trajectories.items():
        for frame in (24,28,31):
            row=min(rows,key=lambda r:(abs(r['reference_frame']-frame),r['time']))
            if abs(row['reference_frame']-frame)>.2:continue
            for c in row['actual_contacts']:
                snapshots.append(dict(method=name,requested_frame=frame,frame=row['reference_frame'],
                    time=row['time'],body=c['body_name'],x_mm=1000*c['point_object'][0],
                    y_mm=1000*c['point_object'][1],z_mm=1000*c['point_object'][2],
                    nx=c['normal_object'][0],ny=c['normal_object'][1],nz=c['normal_object'][2],
                    normal_N=c['normal_force'],vertical_N=c['force_world'][2]))
    save_csv('aligned_contact_snapshots.csv',snapshots)
    geometry_path=args.dexforge/'forceaware/same_state_mesh_geometry.jsonl'
    geometry=[]
    if geometry_path.exists():
        for native,query in zip(trajectories['DexForge'],read_jsonl(geometry_path)):
            errors=[];angles=[];missing=0
            for c in native['actual_contacts']:
                candidates=[p for p in query['contacts'] if p['body_name']==c['body_name']]
                if not candidates:missing+=1;continue
                closest=min(candidates,key=lambda p:np.linalg.norm(np.asarray(p['point_object'])-c['point_object']))
                errors.append(np.linalg.norm(np.asarray(closest['point_object'])-c['point_object']))
                angles.append(np.degrees(np.arccos(np.clip(np.dot(c['normal_object'],closest['normal_object']),-1,1))))
            geometry.append(dict(frame=native['reference_frame'],comfree_contacts=len(native['actual_contacts']),
                mujoco_contacts=len(query['contacts']),missing_comfree_contact_bodies=missing,
                same_body_position_error_mm=1000*float(np.mean(errors)) if errors else None,
                same_body_normal_error_deg=float(np.mean(angles)) if angles else None))
        save_csv('same_state_geometry.csv',geometry)
    summary={}
    common=set.intersection(*[set(int(round(r['frame'])) for r in rows) for rows in metrics.values()])
    for name,rows in metrics.items():
        contact=[r for r in rows if r['contacts']]
        aligned=[r for r in rows if int(round(r['frame'])) in common and r['targets']]
        # Use the last visit to each frame explicitly; do not fabricate later frames.
        last={int(round(r['frame'])):r for r in aligned}
        errors=[r['region_error_mm'] for r in last.values() if r['region_error_mm'] is not None]
        summary[name]=dict(samples=len(rows),duration_s=rows[-1]['time'],
            maximum_demo_frame=max(r['frame'] for r in rows),physical_contact_samples=len(contact),
            force_closure_fraction_common=float(np.mean([r['force_closed'] for r in rows])),
            loaded_force_closure_fraction_common=float(np.mean([r['loaded_force_closed'] for r in rows])),
            maximum_epsilon_common=max(r['epsilon_common'] for r in rows),
            max_clearance_mm=max(r['clearance_mm'] for r in rows),
            maximum_hand_vertical_N=max(r['hand_vertical_N'] for r in rows),
            aligned_frames=len(last),aligned_last_region_error_mm=float(np.mean(errors)) if errors else None,
            aligned_last_missing_target_body_fraction=float(sum(r['missing_target_regions'] for r in last.values())/sum(r['targets'] for r in last.values())) if last else None,
            aligned_last_force_closure_fraction=float(np.mean([r['force_closed'] for r in last.values()])) if last else None)
    write_json(args.output/'summary.json',dict(sequence='20200709_142802',summary=summary,
        audits=audits,common_demo_frames=sorted(common),object_radius_m=radius,
        common_friction=args.common_friction,ours_replay_max_pose_error=replay_error,
        alignment='object coordinates; original input frame IDs; no extrapolation beyond reached frames',
        epsilon='minimum signed facet distance of normalized primitive-wrench convex hull',
        contact_tolerance_m=.0001,loaded_normal_threshold_N=.01,
        ours_run_complete=(args.ours/'full/metrics.json').exists(),
        same_state_geometry_summary=dict(samples=len(geometry),
            mean_position_error_mm=float(np.mean([r['same_body_position_error_mm'] for r in geometry if r['same_body_position_error_mm'] is not None])) if geometry else None,
            mean_normal_error_deg=float(np.mean([r['same_body_normal_error_deg'] for r in geometry if r['same_body_normal_error_deg'] is not None])) if geometry else None)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'ours':'#2468b4','DexForge':'#db6b22','DexForge_MuJoCo':'#299458'}
    fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for name,rows in metrics.items():
        x=np.array([r['frame'] for r in rows]);time=np.array([r['time'] for r in rows])
        for ax,key,title in [(axes[0,0],'contacts','Physical contact count'),
            (axes[0,1],'epsilon_common',f'GWS signed margin (common mu={args.common_friction})'),
            (axes[1,0],'region_error_mm','Nearest MANO anchor error, same body (mm)'),
            (axes[1,1],'hand_vertical_N','Hand vertical force (N)')]:
            frames=sorted(set(int(round(r['frame'])) for r in rows));values=[];lower=[];upper=[]
            for frame in frames:
                samples=[r[key] for r in rows if int(round(r['frame']))==frame and r[key] is not None]
                values.append(np.median(samples) if samples else np.nan)
                lower.append(np.quantile(samples,.1) if samples else np.nan)
                upper.append(np.quantile(samples,.9) if samples else np.nan)
            ax.plot(frames,values,color=colors[name],label=name,alpha=.8)
            ax.fill_between(frames,lower,upper,color=colors[name],alpha=.12)
            ax.set_title(title);ax.set_xlabel('Demonstration frame (median, 10-90% band)')
        axes[2,0].plot(time,[r['clearance_mm'] for r in rows],label=name,color=colors[name]);axes[2,0].set_title('Lowest object vertex clearance (mm)');axes[2,0].set_xlabel('Execution time (s)')
        axes[2,1].plot(time,x,label=name,color=colors[name]);axes[2,1].set_title('Reached demonstration frame');axes[2,1].set_xlabel('Execution time (s)')
    for ax in axes.flat:ax.grid(alpha=.2)
    axes[0,0].legend();fig.savefig(args.output/'trajectory_metrics.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharex=True,constrained_layout=True)
    for axis,coordinate in zip(axes,('x','y','z')):
        targets=[r for r in target_table if r['method']=='MANO']
        axis.scatter([r['frame'] for r in targets],[1000*r[coordinate] for r in targets],s=5,c='black',alpha=.2,label='MANO')
        for name in trajectories:
            actual=[r for r in contact_table if r['method']==name]
            axis.scatter([r['frame'] for r in actual],[1000*r[coordinate] for r in actual],s=5,c=colors[name],alpha=.35,label=name)
        axis.set_title(f'Object-local contact {coordinate} (mm)');axis.set_xlabel('Demonstration frame');axis.grid(alpha=.2)
    axes[0].legend(markerscale=2);fig.savefig(args.output/'contact_positions.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
