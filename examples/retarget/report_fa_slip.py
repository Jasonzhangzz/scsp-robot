"""Compare accepted forecasts to measured contacts, without rerunning optimization."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(path):
    rows=[json.loads(s) for s in (path/'diagnostics/trace.jsonl').read_text().splitlines()]
    predicted={(r['window'],r['stage']):r for r in rows if r['kind']=='prediction'}
    execution=[r for r in rows if r['kind']=='execution'];plans=[r for r in rows if r['kind']=='plan']
    import mujoco
    model=mujoco.MjModel.from_xml_path('outputs/fa_admm_solver_20261016/sequence/scene/wuji_right/sugar_box_mesh.xml')
    data=mujoco.MjData(model);oq=26;mass=.2;gravity=9.81
    pairs=[];contacts=[];targets=[];horizon=[];force_balance=[];ideal=[];source_time=None
    for p in plans:
        window=p['window'];forecast=[predicted[window,k] for k in range(30)]
        saved=np.load(path/f'diagnostics/window_{window:04d}.npz');reference=saved['parameters'][0,27:30]
        if p.get('template_update') and not p['template_update'].get('held_template',False):source_time=p['time']
        if p.get('ideal_force_plan') is not None:
            for k,(forces,points) in enumerate(zip(p['ideal_force_plan'],p['ideal_point_plan'])):
                for slot,(force,point) in enumerate(zip(forces,points)):
                    if np.linalg.norm(force)<1e-8:continue
                    body=int(saved['query_body'][min(6*k+5,29),slot])
                    ideal.append(dict(window=window,time=p['time'],source_plan_time=source_time,coarse_stage=k,slot=slot,body=body,
                        body_name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body),
                        **dict(zip(('fx_N','fy_N','fz_N'),force)),force_norm_N=np.linalg.norm(force),
                        **dict(zip(('object_x_mm','object_y_mm','object_z_mm'),np.array(point)*1000))))
        fz=np.array([r['force']['hand_vertical_N'] for r in forecast])
        z=np.array([r['qpos'][oq+2] for r in forecast]+[forecast[-1]['next_qpos'][oq+2]])
        horizon.append(dict(window=window,time=p['time'],initial_z_mm=1000*z[0],terminal_z_mm=1000*z[-1],
            terminal_height_error_mm=1000*(z[-1]-reference[2]),minimum_z_mm=1000*z.min(),
            hand_vertical_mean_N=float(fz.mean()),hand_vertical_min_N=float(fz.min()),
            hand_vertical_first30ms_N=float(fz[:3].mean()),accepted=p['accepted_steps'],**p['cost_components']))
        for k,r in enumerate(forecast[:-1]):
            dv=np.array(forecast[k+1]['qvel'][26:29])-r['qvel'][26:29]
            force_balance.append(np.linalg.norm(mass*dv/.01-np.array(r['force']['total_force'])-np.array([0,0,-mass*gravity])))
    for r in execution:
        p=predicted[r['window'],r['stage']];average=r['force_average'];actual=r['contacts']
        a=np.array(r['next_qpos'][oq:oq+3]);b=np.array(p['next_qpos'][oq:oq+3])
        distances=[]
        for contact in p['contacts']:
            if contact['body']==0 or contact['normal_force']<.01:continue
            same=[c for c in actual if c['body']==contact['body'] and c['normal_force']>.01]
            if same:distances.append(min(np.linalg.norm(np.array(contact['point_object'])-c['point_object'])*1000 for c in same))
        pairs.append(dict(time=r['time'],window=r['window'],stage=r['stage'],
            prediction_position_error_mm=float(1000*np.linalg.norm(a-b)),prediction_z_error_mm=float(1000*(a[2]-b[2])),
            predicted_hand_vertical_N=p['force']['hand_vertical_N'],actual_hand_vertical_N=average['hand_vertical_N'],
            predicted_normal_N=p['force']['total_normal_N'],actual_normal_N=average['total_normal_N'],
            predicted_loaded=p['force']['loaded_contacts'],actual_loaded=average['loaded_contacts'],
            predicted_cost=p['cost']['total'],measured_state_FA_cost=r['cost']['total'],
            contact_same_body_distance_mean_mm=float(np.mean(distances)) if distances else None))
        saved=np.load(path/f'diagnostics/window_{r["window"]:04d}.npz');stage=r['stage']
        data.qpos[:]=r['qpos'];mujoco.mj_kinematics(model,data)
        R=Rotation.from_quat(np.array(r['qpos'])[oq+3:oq+7],scalar_first=True).as_matrix()
        origin=np.array(r['qpos'])[oq:oq+3]
        for slot in np.flatnonzero(saved['mask'][stage]>0):
            body=int(saved['query_body'][stage,slot]);target=saved['template'][stage,slot]
            query=data.xpos[body]+data.xmat[body].reshape(3,3)@saved['query_local'][stage,slot]
            query_local=(query-origin)@R
            measured=[c for c in actual if c['body']==body and c['normal_force']>.01]
            predicted_contacts=[c for c in p['contacts'] if c['body']==body and c['normal_force']>.01]
            nearest=lambda cs: min((np.linalg.norm(np.array(c['point_object'])-target)*1000 for c in cs),default=None)
            targets.append(dict(time=r['time'],window=r['window'],stage=stage,slot=int(slot),body=body,
                body_name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body),
                template_x_mm=1000*target[0],template_y_mm=1000*target[1],template_z_mm=1000*target[2],
                query_x_mm=1000*query_local[0],query_y_mm=1000*query_local[1],query_z_mm=1000*query_local[2],
                query_template_distance_mm=1000*np.linalg.norm(query_local-target),
                original_template_drift_mm=1000*np.linalg.norm(target-saved['original_template'][stage,slot]),
                predicted_contact_template_distance_mm=nearest(predicted_contacts),actual_contact_template_distance_mm=nearest(measured),
                predicted_body_vertical_N=sum(c['force'][2] for c in predicted_contacts),actual_body_vertical_N=sum(c['force'][2] for c in measured)))
    for r in rows:
        if r['kind'] not in ('prediction','execution'):continue
        snapshots=r.get('substeps') or [dict(time=r['time'],contacts=r['contacts'])]
        for sub in snapshots:
            for c in sub['contacts']:
                contacts.append(dict(kind=r['kind'],window=r['window'],stage=r['stage'],time=sub['time'],
                    contact_id=c['contact_id'],geom1=c['geom1'],geom2=c['geom2'],body=c['body'],body_name=c['body_name'],
                    **dict(zip(('x_mm','y_mm','z_mm'),np.array(c['point'])*1000)),
                    **dict(zip(('object_x_mm','object_y_mm','object_z_mm'),np.array(c['point_object'])*1000)),
                    **dict(zip(('fx_N','fy_N','fz_N'),c['force'])),normal_N=c['normal_force'],tangent_N=c['tangent_force'],
                    friction_utilization=c['friction_utilization'],gap_mm=1000*c['gap']))
    for name,values in [('comparison',pairs),('contacts',contacts),('targets',targets),('horizons',horizon),('ideal_forces',ideal)]:write_csv(path/(name+'.csv'),values)
    baseline_name='comfree_retained_rti3' if 'comfree' in path.name else 'mujoco_retained_rti3'
    baseline=np.load(Path('outputs/fa_direct_mpc_20261016')/baseline_name/'rollout.npz');rollout=np.load(path/'rollout.npz')
    differences={k:float(np.max(np.abs(rollout[k]-baseline[k]))) for k in ('qpos_traj','qvel_traj','ctrl_traj')}
    error=np.array([r['prediction_position_error_mm'] for r in pairs]);balance=np.array(force_balance)
    result=dict(instrumentation_vs_baseline_max_abs=differences,prediction_executed_steps_error_mm=dict(mean=float(error.mean()),max=float(error.max())),
        native_linear_force_balance_residual_N=dict(median=float(np.median(balance)),p95=float(np.quantile(balance,.95)),max=float(balance.max())),
        selected_horizons=[min(horizon,key=lambda r:abs(r['time']-t)) for t in (2.4,3.,3.6,3.9,4.2,4.5,4.8)],
        window_count=len(plans),actual_steps=len(execution),raw_contact_rows=len(contacts))
    (path/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result,pairs,execution,plans


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(14,13),sharex='col')
    force_fig,force_axes=plt.subplots(5,2,figsize=(13,11),sharex=True)
    point_fig=plt.figure(figsize=(13,6))
    summaries={}
    for col,name in enumerate(('comfree','mujoco')):
        path=args.root/name
        if not (path/'rollout.npz').exists():continue
        result,pairs,execution,plans=analyze(path);summaries[name]=result
        t=[r['time'] for r in pairs]
        axes[0,col].plot(t,[r['qpos'][28]*1000 for r in execution],label='executed height')
        for plan in plans[::5]:
            saved=np.load(path/f'diagnostics/window_{plan["window"]:04d}.npz')
            axes[0,col].plot(plan['time']+np.arange(31)*.01,saved['states'][:,28]*10,alpha=.35,color='tab:orange')
        axes[0,col].set_title(name);axes[0,col].set_ylabel('Object height (mm)');axes[0,col].legend()
        axes[1,col].plot(t,[r['predicted_hand_vertical_N'] for r in pairs],label='predicted hand Fz')
        axes[1,col].plot(t,[r['actual_hand_vertical_N'] for r in pairs],label='executed hand Fz',alpha=.8)
        axes[1,col].axhline(1.962,color='k',ls='--',label='weight 1.962 N');axes[1,col].set_ylabel('Force (N)');axes[1,col].legend()
        for field in ('object_position','object_rotation','physical_contact','marked_query','hand_base','control_acceleration','state_acceleration'):
            axes[2,col].plot([r['time'] for r in plans],[r['cost_components'][field] for r in plans],label=field)
        axes[2,col].set_ylabel('Horizon FA cost (scaled 1e-6)');axes[2,col].legend(fontsize=7,ncol=2)
        axes[3,col].plot(t,[r['prediction_position_error_mm'] for r in pairs],label='same-control 10–30 ms pose error')
        axes[3,col].set_ylabel('Prediction error (mm)');axes[3,col].set_xlabel('Time (s)');axes[3,col].legend()
        for ax in axes[:,col]:ax.grid(alpha=.25);ax.set_xlim(2.4,5.1)
        trace=[json.loads(line) for line in (path/'diagnostics/trace.jsonl').read_text().splitlines()]
        pred={(r['window'],r['stage']):r for r in trace if r['kind']=='prediction'}
        for finger in range(1,6):
            def fz(cs):return sum(c['force'][2] for c in cs if c['body_name'].startswith(f'right_finger{finger}_'))
            predicted=[fz(pred[r['window'],r['stage']]['contacts']) for r in execution]
            measured=[np.mean([fz(s['contacts']) for s in (r.get('substeps') or [r])]) for r in execution]
            ax=force_axes[finger-1,col];ax.plot(t,predicted,label='forecast');ax.plot(t,measured,label='execution',alpha=.8)
            ax.set_ylabel(f'Finger {finger} Fz (N)');ax.grid(alpha=.25);ax.axhline(0,color='k',lw=.5);ax.set_xlim(2.4,5.1)
            if finger==1:ax.set_title(name);ax.legend()
            if finger==5:ax.set_xlabel('Time (s)')
        current=min(execution,key=lambda r:abs(r['time']-3.6));p0=pred[current['window'],current['stage']]
        snap=np.load(path/f'diagnostics/window_{current["window"]:04d}.npz');mask=snap['mask'][0]>0
        ax=point_fig.add_subplot(1,2,col+1,projection='3d')
        template=snap['template'][0,mask]*1000
        ax.scatter(*template.T,marker='^',s=40,color='tab:orange',label='ADMM target template')
        for contacts_,label,color in [(p0['contacts'],'predicted physical contacts','tab:blue'),(current['contacts'],'executed physical contacts','tab:green')]:
            cs=[c for c in contacts_ if c['body'] and c['normal_force']>.01]
            if not cs:continue
            points=np.array([c['point_object'] for c in cs])*1000
            ax.scatter(*points.T,s=35,color=color,label=label)
            forces=np.array([c['force_object'] for c in cs])*5
            ax.quiver(*points.T,*forces.T,color=color,arrow_length_ratio=.2)
        ax.set_title(name+' at 3.60 s (force arrows: 5 mm/N)')
        ax.set_xlabel('Object x (mm)');ax.set_ylabel('Object y (mm)');ax.set_zlabel('Object z (mm)');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(args.root/'slip_comparison.png',dpi=160)
    force_fig.tight_layout();force_fig.savefig(args.root/'finger_forces.png',dpi=160)
    point_fig.tight_layout();point_fig.savefig(args.root/'contacts_3p6.png',dpi=160)
    ablation={};comparison_fig,comparison_axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    original=np.load('outputs/fa_admm_solver_20261016/contactaware/contact_guidance.npz')['contact_pos_obj'][-1]
    for name in ('comfree','mujoco','comfree_fixed_template','comfree_deeper_search','comfree_freeze_hold'):
        path=args.root/name
        if not (path/'rollout.npz').exists():continue
        saved=np.load(path/'rollout.npz');q=saved['qpos_traj'][1:];ref=saved['reference_qpos'];t=np.arange(1,len(q)+1)*.01
        height=1000*(q[:,28]-ref[:,28]);position=1000*np.linalg.norm(q[:,26:29]-ref[:,26:29],axis=1)
        windows=[json.loads(line) for line in (path/'windows.jsonl').read_text().splitlines()]
        first=lambda mm:next((float(x) for x,h in zip(t,height) if h < -mm),None)
        ablation[name]=dict(mean_object_error_mm=float(position.mean()),final_height_error_mm=float(height[-1]),
            first_height_deficit_20mm_s=first(20),first_height_deficit_80mm_s=first(80),
            stalled_windows=sum(r['accepted_SQP_steps']==0 for r in windows),windows=len(windows))
        comparison_axes[0].plot(t,height,label=name)
        template_path=path/'templates.npz'
        if template_path.exists():
            template=np.load(template_path);active=template['mask'][:,0]>0
            drift=np.linalg.norm(template['template'][:,0]-original,axis=-1)*1000
            times=template['window']*.03;hold=times>=2.4-1e-8
            maximum=np.max(np.where(active,drift,0),axis=1)
            if hold.any():
                comparison_axes[1].step(np.r_[times[hold],5.4],np.r_[maximum[hold],maximum[hold][-1]],where='post',label=name)
                ablation[name]['maximum_template_drift_in_hold_mm']=float(maximum[hold].max())
            ablation[name]['template_commits']=int(template['updated'].sum())
    comparison_axes[0].set_ylabel('Height minus MANO target (mm)');comparison_axes[0].axhline(-80,color='k',ls='--',alpha=.4)
    comparison_axes[1].set_ylabel('Max target-template drift (mm)');comparison_axes[1].set_xlabel('Time (s)')
    for ax in comparison_axes:ax.grid(alpha=.25);ax.legend(fontsize=8);ax.set_xlim(2.4,5.4)
    comparison_fig.tight_layout();comparison_fig.savefig(args.root/'ablation.png',dpi=160)
    (args.root/'ablation.json').write_text(json.dumps(ablation,indent=2)+'\n')
    (args.root/'diagnosis.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
