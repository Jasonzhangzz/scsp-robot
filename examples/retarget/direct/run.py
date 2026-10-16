"""Closed-loop MANO/CA tracking experiment for the direct ComFree OCP."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np


def metrics(q,reference,engine):
    o=engine.oq;hand=engine.index.hand_qpos
    p=1000*np.linalg.norm(q[:,o:o+3]-reference[:,o:o+3],axis=1)
    a=q[:,o+3:o+7];b=reference[:,o+3:o+7]
    dot=np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
    angle=np.rad2deg(2*np.arccos(np.clip(np.abs(dot),0,1)))
    h=q[:,hand]-reference[:,hand]
    return dict(object_position_mean_mm=float(p.mean()),object_position_rmse_mm=float(np.sqrt(np.mean(p*p))),
        object_position_max_mm=float(p.max()),object_rotation_mean_deg=float(angle.mean()),
        object_rotation_max_deg=float(angle.max()),object_final_position_mm=float(p[-1]),object_final_rotation_deg=float(angle[-1]),
        hand_base_position_mean_mm=float(1000*np.linalg.norm(h[:,:3],axis=1).mean()),
        hand_base_rotation_coordinate_rmse_rad=float(np.sqrt(np.mean(h[:,3:6]**2))),
        hand_finger_joint_rmse_rad=float(np.sqrt(np.mean(h[:,6:]**2))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,default=Path('outputs/fa_admm_solver_20261016'))
    p.add_argument('--forge',type=Path,default=Path('thirdparty/retarget_assets/DexForge'))
    p.add_argument('--output',type=Path,default=Path('outputs/fa_direct_mpc_20261016'))
    p.add_argument('--executor',choices=['mujoco','comfree'],default='mujoco')
    p.add_argument('--duration',type=float,default=5.4)
    p.add_argument('--rti-steps',type=int,default=3)
    p.add_argument('--line-search-steps',type=int,default=4,help='Nonlinear backtracking trials; experimental value eight reaches 1/128')
    p.add_argument('--post-step-tracking',action='store_true',help='Match original FA: track the post-step qpos while geometry uses the pre-step state')
    p.add_argument('--relative-warm',action='store_true',help='Shift reference-relative corrections instead of absolute controls')
    p.add_argument('--lm',type=float,default=.01,help='Gauss-Newton LM damping; exposed for one-factor diagnosis')
    p.add_argument('--mode',choices=['ocp','reference'],default='ocp')
    p.add_argument('--contact-admm',action='store_true')
    p.add_argument('--template-period',type=int,default=5,help='Template update period in 30 ms MPC windows')
    p.add_argument('--template-weight',type=float,default=5.,help='Soft displacement cost per squared mm from held template')
    p.add_argument('--consistent-template-commit',action='store_true',help='Score material-point candidates exactly as they will be committed')
    p.add_argument('--freeze-templates-after',type=float,default=None,help='Diagnostic ablation: hold the last committed templates after this simulation time')
    p.add_argument('--stop-contact-frame-gradients',action='store_true',help='Ablation: use the upstream FA stopped normal/frame VJP')
    p.add_argument('--diagnostics',action='store_true',help='Record accepted MPC forecasts, FA cost components and physical contact forces')
    p.add_argument('--trace-start',type=float,default=0.)
    p.add_argument('--trace-end',type=float,default=5.4)
    args=p.parse_args()
    if (args.output/'rollout.npz').exists() or (args.output/'windows.jsonl').exists():
        raise FileExistsError(f'Use a fresh output directory: {args.output}')
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
    from examples.retarget.common.engine import ComFreeEngine
    from examples.retarget.direct.solver import DirectComFreeMPC
    import mujoco
    engine=ComFreeEngine(args.experiment,args.forge)
    from forceaware.targets import make_window,initial_state
    q,v=initial_state(engine.model,engine.index,engine.sequence,start_frame=0.,ref_dt=1/30,hold=False)
    v=np.zeros_like(v);previous=q[engine.index.hand_qpos].copy()
    controller=DirectComFreeMPC(engine,rti_steps=args.rti_steps,full_contact_jacobian=not args.stop_contact_frame_gradients,
        line_search_steps=args.line_search_steps,post_step_tracking=args.post_step_tracking,levenberg_marquardt=args.lm,
        relative_warm=args.relative_warm) if args.mode=='ocp' else None
    model=mujoco.MjModel.from_xml_path(str(args.experiment/'sequence/scene/wuji_right/sugar_box_mesh.xml'))
    model.opt.timestep=.002;data=mujoco.MjData(model)
    if controller is not None and args.contact_admm:
        from examples.retarget.direct.templates import ContactTemplates
        controller.templates=ContactTemplates(controller,model,args.output,period=args.template_period,anchor_weight=args.template_weight,
            commit_consistent=args.consistent_template_commit)
    tracer=None
    if args.diagnostics:
        if controller is None:raise ValueError('Prediction diagnostics require OCP mode')
        from examples.retarget.direct.diagnostics import MPCTrace,mujoco_contacts,native_contacts
        controller.save_prediction=True
        tracer=MPCTrace(controller,args.output/'diagnostics',args.executor,start=args.trace_start,end=args.trace_end)
    qs=[q.copy()];vs=[v.copy()];controls=[];references=[];windows=[]
    started=perf_counter();dense=0;total=int(round(args.duration/.01))
    while dense<total:
        target=make_window(engine.sequence,executed_steps=dense,horizon=5,time_grid=engine.cfg.simulator.time_grid,
            start_frame=0.,end_frame=len(engine.sequence.hand)-1.)
        count=min(3,total-dense)
        if controller is not None and controller.templates is not None and args.freeze_templates_after is not None:
            controller.templates.frozen=dense*.01>args.freeze_templates_after+1e-9
        if controller is not None:commands,diag=controller.solve(q,v,target,previous,execution_steps=count)
        else:commands=target.dense_actions;diag=dict(wall_s=0.,statuses=[],mode='reference')
        if tracer is not None:tracer.begin(dense)
        windows.append(dict(dense=dense,**diag))
        with (args.output/'windows.jsonl').open('a') as f:f.write(json.dumps(windows[-1])+'\n')
        for stage,servo in enumerate(commands[:count]):
            before_q=q.copy();before_v=v.copy();contact_steps=[];contacts=[]
            if args.executor=='mujoco':
                data.qpos[:]=q;data.qvel[:]=v;data.ctrl[engine.index.hand_ctrl]=servo
                data.qfrc_applied[:]=0.
                for sub in range(5):
                    pose=data.qpos[engine.oq:engine.oq+7].copy()
                    mujoco.mj_step(model,data)
                    if tracer is not None and tracer.active:
                        cs=mujoco_contacts(model,data,engine.index.object_body,pose)
                        contact_steps.append(dict(time=dense*.01+sub*.002,contacts=cs))
                if contact_steps:contacts=contact_steps[0]['contacts']
                mujoco.mj_forward(model,data);q=data.qpos.copy();v=data.qvel.copy()
            else:
                q1,v1,buffer=engine.step(q,v,servo,dt=.01)
                if tracer is not None and tracer.active:contacts=native_contacts(engine,buffer,q)
                q=q1[0];v=v1[0]
            if not np.isfinite(q).all() or not np.isfinite(v).all():raise FloatingPointError('Nonfinite executed state')
            if tracer is not None:tracer.execution(dense,stage,before_q,before_v,servo,q,v,contacts,substep=contact_steps)
            controls.append(servo.copy());qs.append(q.copy());vs.append(v.copy());dense+=1
            # Dense FA targets are look-ahead. Evaluation uses the actual
            # physical time, interpolated on the original demonstration clock.
            from examples.retarget.common.trajectory import reference as pose_reference
            r=q.copy();time=dense*.01;frame=min(time*30,len(engine.sequence.hand)-1)
            i=int(np.floor(frame));j=min(i+1,len(engine.sequence.hand)-1);alpha=frame-i
            r[engine.index.hand_qpos]=(1-alpha)*engine.sequence.hand[i]+alpha*engine.sequence.hand[j]
            pose=engine.sequence.object_pose_xyzw[:,[4,5,6,3,0,1,2]]
            r[engine.oq:engine.oq+7]=pose_reference(pose,np.array([time]))[0]
            references.append(r)
        previous=commands[count-1].copy()
        if len(windows)%10==0 or dense==total:
            print(json.dumps(dict(window=len(windows),dense=dense,status=diag['statuses'],wall_s=diag['wall_s'],
                **metrics(np.asarray(qs[1:]),np.asarray(references),engine))),flush=True)
        if len(windows)%10==0:
            np.savez_compressed(args.output/'partial.npz',qpos_traj=qs,qvel_traj=vs,ctrl_traj=controls,reference_qpos=references)
    np.savez_compressed(args.output/'rollout.npz',qpos_traj=qs,qvel_traj=vs,ctrl_traj=controls,reference_qpos=references,
        mpc_step_sec=np.array([w['wall_s'] for w in windows]))
    if tracer is not None:tracer.close()
    warm=np.array([w['wall_s'] for w in windows[1:]])
    report=dict(arguments=vars(args),complete=dense==total,simulated_seconds=dense*.01,wall_seconds=perf_counter()-started,
        tracking=metrics(np.asarray(qs[1:]),np.asarray(references),engine),
        failed_windows=sum(any(s!=0 for s in w['statuses']) for w in windows),
        unconverged_QP_windows=sum(any(s!=0 for s in w.get('qp_statuses',[])) for w in windows),windows=len(windows),
        warm_median_s=float(np.median(warm)) if len(warm) else None,
        warm_p95_s=float(np.quantile(warm,.95)) if len(warm) else None,
        audit=dict(direct_multiple_shooting=args.mode=='ocp',native_ComFree_dynamics=True,finite_difference_jacobians=False,
            native_coupled_corrections=int(engine.compiled.contact.iterations),
            prediction_dt=.01,prediction_duration=.3,execution_interval=.03,controls='30 dense servo targets',
            contact_templates='low-rate anchored ADMM' if args.contact_admm else 'fixed MANO/CA',
            actual_physics_contacts='native querypoint/SDF narrowphase; never template or contact-search candidates',
            cost='FA geometry + tracking + physical smoothing',
            contact_Hessian='individual residual Gauss-Newton',initial_velocity='zero',initial_pose_refinement=False))
    if controller is not None and controller.templates is not None:
        report['template_updates']=controller.templates.updates
        snapshots=controller.templates.snapshots
        if snapshots:
            np.savez_compressed(args.output/'templates.npz',**{k:np.stack([r[k] for r in snapshots]) for k in snapshots[0]})
    (args.output/'report.json').write_text(json.dumps(report,default=str,indent=2)+'\n')
    print(json.dumps(report,default=str,indent=2),flush=True)


if __name__=='__main__':main()
