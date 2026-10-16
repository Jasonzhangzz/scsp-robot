"""Replay measured states/controls once to recover per-contact execution forces."""
import csv
import json
from pathlib import Path
import re
import numpy as np


def main():
    from examples.retarget.matrix_comfree import ComFreeEngine
    from examples.retarget.fa_mpc_diagnostics import native_contacts,force_summary
    e=ComFreeEngine(Path('outputs/fa_admm_solver_20261016'),Path('thirdparty/retarget_assets/DexForge'))
    root=Path('outputs/fa_matched_ablation_20261016');results={};contact_rows=[];drifts={}
    import mujoco
    mesh_model=mujoco.MjModel.from_xml_path('outputs/fa_admm_solver_20261016/sequence/scene/wuji_right/sugar_box_mesh.xml')
    metadata={field:bool(np.array_equal(getattr(mesh_model,field),getattr(e.model,field)))
        for field in ('actuator_ctrlrange','actuator_ctrllimited')}
    assert all(metadata.values()),metadata
    native_clamp=not bool(e.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
    assert native_clamp==(not bool(mesh_model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL))
    (root/'actuator_metadata_validation.json').write_text(json.dumps(dict(mesh_native_equal=metadata,
        native_control_clamping_enabled=native_clamp),indent=2)+'\n')
    cases=list((root/'comfree').glob('*'))+list(root.glob('direct_*'))
    mass=float(e.model.body_mass[e.index.object_body])
    from examples.retarget.matrix_report import reference
    poses=np.load('outputs/fa_admm_solver_20261016/contactaware/object_pose_7.npy')[:,[4,5,6,3,0,1,2]]
    for case in cases:
        if not (case/'rollout.npz').exists() or (case/'INVALIDATED.json').exists():continue
        saved=np.load(case/'rollout.npz');q=saved['qpos_traj'];v=saved['qvel_traj'];u=saved['ctrl_traj'];samples=[]
        if (case/'templates.npz').exists():
            from forceaware.targets import make_window
            templates=np.load(case/'templates.npz');rows=[]
            for index,window in enumerate(templates['window']):
                target=make_window(e.sequence,executed_steps=int(window)*3,horizon=5,time_grid=e.cfg.simulator.time_grid,
                    start_frame=0.,end_frame=len(e.sequence.hand)-1.)
                delta=1000*np.linalg.norm(templates['template'][index]-target.contact_position_object,axis=-1)[templates['mask'][index]>0]
                rows.append(dict(time_s=float(window*.03),updated=bool(templates['updated'][index]),
                    max_shift_mm=float(np.max(delta,initial=0.)),mean_shift_mm=float(delta.mean()) if delta.size else 0.))
            drifts[case.name]=rows
        sample_times={2.4,3.6,4.8,5.3}
        # Preserve common snapshots and inspect the approach to a measured drop.
        times=np.arange(len(q))*.01;goals=reference(poses,times)
        lost=np.flatnonzero((goals[:,2]-q[:,-5]>.08)&(goals[:,2]-goals[0,2]>.05))
        if len(lost):
            sample_times.update(round(max(times[lost[0]]-offset,0.),2) for offset in (.3,.03,0.))
        for time in sorted(sample_times):
            dense=round(time/.01)
            if dense>=len(u):continue
            next_q,next_v,buffer=e.step(q[dense],v[dense],u[dense],dt=.01)
            error=float(np.max(abs(next_q[0]-q[dense+1])));assert error<2e-5,(case,time,error)
            contacts=native_contacts(e,buffer,q[dense]);summary=force_summary(contacts,mass,e.model.opt.gravity)
            fingers={}
            for c in contacts:
                match=re.search(r'finger(\d+)',c['body_name'])
                if match is None:continue
                key=match.group(1);finger=fingers.setdefault(key,dict(force_N=[0.,0.,0.],normal_N=0.,loaded_contacts=0))
                finger['force_N']=(np.asarray(finger['force_N'])+c['force']).tolist()
                finger['normal_N']+=c['normal_force'];finger['loaded_contacts']+=int(c['normal_force']>.01)
            samples.append(dict(time=time,**summary,object_velocity_mm_s=(v[dense,e.ov:e.ov+3]*1000).tolist(),
                object_angular_velocity_norm=float(np.linalg.norm(v[dense,e.ov+3:e.ov+6])),
                replay_next_qpos_max_error=error,fingers=fingers,contacts=contacts))
            for c in contacts:
                contact_rows.append(dict(case=case.name,time=time,body=c['body'],body_name=c['body_name'],contact_id=c['contact_id'],
                    geom1=c['geom1'],geom2=c['geom2'],**dict(zip(('object_x_mm','object_y_mm','object_z_mm'),np.array(c['point_object'])*1000)),
                    **dict(zip(('fx_N','fy_N','fz_N'),c['force'])),normal_N=c['normal_force'],tangent_N=c['tangent_force'],gap_mm=c['gap']*1000))
        results[case.name]=samples
    (root/'contact_comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    (root/'template_drift_from_MANO.json').write_text(json.dumps(drifts,indent=2)+'\n')
    if contact_rows:
        with (root/'contact_forces.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(contact_rows[0]));w.writeheader();w.writerows(contact_rows)
    print(json.dumps({k:[dict(time=r['time'],Fz=r['hand_vertical_N'],normal=r['total_normal_N'],loaded=r['loaded_contacts']) for r in v] for k,v in results.items()},indent=2))


if __name__=='__main__':main()
