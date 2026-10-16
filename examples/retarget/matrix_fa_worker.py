"""Original FA optimizer, with explicitly selected closed-loop executor."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

import mujoco
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--executor',choices=['mujoco','comfree'],required=True)
    parser.add_argument('--forge',type=Path,required=True);parser.add_argument('--duration',type=float,default=5.4)
    parser.add_argument('--variant',choices=['baseline','admm','lbfgs','admm-force'],default='baseline')
    parser.add_argument('--case-name',default='dexforge')
    parser.add_argument('--output-base',type=Path,help='Write new cases outside the source experiment')
    parser.add_argument('--ablation',choices=['none','absolute-warm','full-contact-jacobian','dense-smoothing','control-bounds','slow-templates','dense-controls','tracking-time'],default='none')
    parser.add_argument('--stop-after-drop',action='store_true',help='End a failed ablation once height is 80 mm below the lifted target')
    parser.add_argument('--rho',type=float,default=1.)
    parser.add_argument('--admm-outer',type=int,default=4)
    parser.add_argument('--iterations',type=int)
    parser.add_argument('--multistarts',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--contact-samples',type=int,default=7)
    parser.add_argument('--contact-radius',type=float,default=.005)
    parser.add_argument('--contact-force-cap',type=float,default=40.)
    parser.add_argument('--contact-force-weight',type=float,default=.01)
    parser.add_argument('--contact-force-iterations',type=int,default=2)
    parser.add_argument('--contact-qp-iterations',type=int,default=64)
    parser.add_argument('--contact-solver',choices=['shooting','batch-lm'],default='shooting')
    parser.add_argument('--hand-solver',choices=['lbfgs','slsqp','acados'],default='lbfgs',
        help='acados solves condensed local control QPs; --evaluation-backend selects the nonlinear oracle')
    parser.add_argument('--hand-rti-iterations',type=int,default=1,help='Maximum acados QP updates per hand block')
    parser.add_argument('--hand-single-block',action='store_true',help='Fixed-contact ablation: keep L-BFGS curvature history across the full hand budget')
    parser.add_argument('--evaluation-backend',choices=['native','compiled'],default='native',
        help='compiled uses generated ComFree equations and analytic FA control gradients')
    parser.add_argument('--compiled-cache',default='')
    parser.add_argument('--compiled-threads',type=int,default=8)
    parser.add_argument('--memoize-evaluations',action=argparse.BooleanOptionalAction,default=False,
        help='Reuse previously evaluated per-candidate values during backtracking')
    parser.add_argument('--viewer',action='store_true')
    args=parser.parse_args()
    if args.evaluation_backend=='compiled' and args.variant=='baseline':
        parser.error('--evaluation-backend compiled requires --variant lbfgs, admm or admm-force')
    p=args.experiment.resolve();forge=args.forge.resolve()
    sys.path[:0]=[str(forge/'forceaware'),str(forge)]
    from forceaware.config import load_retarget_config,config_dict
    from forceaware.runtime import setup_runtime
    from forceaware.time_grid import SolveRequest
    cfg=load_retarget_config(p/'sequence',p/'forceaware.yaml',hand='wuji_right',contactaware_dir=p/'contactaware')
    if args.iterations is not None:cfg=replace(cfg,optimizer=replace(cfg.optimizer,n_iter=args.iterations))
    if args.multistarts is not None:cfg=replace(cfg,optimizer=replace(cfg.optimizer,multistart_count=args.multistarts))
    case=(args.output_base.resolve() if args.output_base is not None else p)/args.executor/args.case_name;case.mkdir(parents=True,exist_ok=True)
    print(f'[case] {case}',flush=True)
    prefix=dict(np.load(case/'partial.npz')) if args.resume else None
    if args.resume and args.variant=='admm-force':
        raise ValueError('Force-contact resume needs contact/dual/force memory; start a fresh case')
    if not args.resume and any((case/n).exists() for n in ('rollout.npz','partial.npz','admm_windows.jsonl','live_state.npz')):
        raise FileExistsError(f'{case} contains a previous run; use a new --case-name')
    resumed_windows=len(prefix['mpc_execution_steps']) if prefix is not None else 0
    if prefix is not None:
        from examples.retarget.fa_admm_resume import install
        install_prefix=install
        path=case/'admm_windows.jsonl';lines=[]
        if path.exists():
            for line in path.read_text().splitlines():
                try:record=json.loads(line)
                except json.JSONDecodeError:break
                if record['window']<resumed_windows:lines.append(json.dumps(record))
            path.write_text('\n'.join(lines)+'\n')
    grid=replace(cfg.simulator.time_grid,exec_dt=.002 if args.executor=='mujoco' else .01)
    cfg=replace(cfg,sequence=replace(cfg.sequence,output_dir_name=str(case)),
        simulator=replace(cfg.simulator,time_grid=grid,exec_backend=args.executor))
    if args.ablation=='dense-controls':
        cfg=replace(cfg,optimizer=replace(cfg.optimizer,horizon=30),
            simulator=replace(cfg.simulator,time_grid=replace(grid,knot_dt=.01,knot_substeps=1)))
    setup_runtime(cfg)
    from forceaware import workflow as workflow
    from forceaware.executor import ActionExecution,ExecutorRuntime
    from forceaware.servo_prior import ServoPriorOptimizer
    optimizer_base=ServoPriorOptimizer
    if args.variant!='baseline':
        from examples.retarget.fa_contact_admm import ADMMSettings,optimizer_class
        native=mujoco.MjModel.from_xml_path(str(p/'sequence/scene/wuji_right/sugar_box_mesh.xml'))
        settings=ADMMSettings(rho=args.rho,outer=args.admm_outer,optimize_contacts=args.variant in ('admm','admm-force'),
            contact_solver=args.contact_solver,hand_solver=args.hand_solver,hand_rti_iterations=args.hand_rti_iterations,
            fixed_hand_single_block=args.hand_single_block,evaluation_backend=args.evaluation_backend,
            compiled_cache=args.compiled_cache,compiled_threads=args.compiled_threads,memoize_evaluations=args.memoize_evaluations)
        if args.variant=='admm-force':
            from examples.retarget.comfree_contact_search import write_object_scene
            object_scene=write_object_scene(cfg.sequence.xml,case/'object_contact_scene.xml')
            settings=replace(settings,contact_mode='comfree-force',object_scene=str(object_scene),
                contact_samples=args.contact_samples,contact_radius=args.contact_radius,
                contact_force_cap=args.contact_force_cap,contact_force_weight=args.contact_force_weight,
                contact_force_iterations=args.contact_force_iterations,contact_qp_iterations=args.contact_qp_iterations)
        optimizer_base=optimizer_class(native,settings)
    snapshots=[];snapshot_indices=[];window_diagnostics=[];cursor={'window':resumed_windows};shared_sequence={}
    if prefix is not None and (case/'partial_contact_admm.npz').exists():
        old=np.load(case/'partial_contact_admm.npz')
        for row,index in enumerate(old['window_indices']):
            if index<resumed_windows:
                snapshots.append({k:old[k][row] for k in old.files if k!='window_indices'})
                snapshot_indices.append(int(index))
    class FixedInitialOptimizer(optimizer_base):
        def solve(self,*a,**kw):
            solve_started=time.perf_counter()
            cursor['optimizer']=self
            kw['allow_init']=False
            if args.ablation!='none' and not getattr(self,'_ablation_installed',False):
                from examples.retarget.fa_matched_ablation import install
                install(self,args.ablation,native,case,sequence=shared_sequence.get('value'))
                self._ablation_installed=True
                if getattr(self,'_ablation_noop',False):
                    (case/'NO_OP.json').write_text(json.dumps(dict(factor=args.ablation,
                        reason='The SDF baseline already differentiates contact normals/frames; this switch changes no computation.'),indent=2)+'\n')
                    raise SystemExit(0)
            plan=super().solve(*a,**kw)
            if args.variant!='baseline':
                from forceaware.mpc_loss import MpcObjective
                selected=plan.diag['selected']
                snap=self.objective.contact_memory if args.variant=='admm-force' else self.objective.snapshot(selected)
                if args.ablation=='slow-templates':snap['original_template']=a[2].contact_position_object.copy()
                snapshots.append(snap);snapshot_indices.append(cursor['window'])
                # Final ranking uses the pure FA objective at the variable
                # surface contact points, without any augmented penalty.
                if args.variant=='admm-force':fa=self.last_solver_diag['FA_loss']
                else:
                    MpcObjective.evaluate(self.objective,self.control,allow_init=False)
                    fa=float(self.loss.total.numpy()[selected])
                    if args.ablation=='tracking-time':fa+=float(self._ablation_state['constant'].numpy()[selected])*1e6
                mask=snap['mask']>0
                primal=np.linalg.norm(snap['c_before_surface_polish']-snap['z'],axis=-1)[mask]*1000
                shift=np.linalg.norm(snap['z']-snap['template'],axis=-1)[mask]*1000
                record=dict(window=cursor['window'],FA_loss=fa,selection_merit=plan.loss,
                    primal_mean_mm=float(primal.mean()) if len(primal) else 0.,
                    primal_max_mm=float(primal.max()) if len(primal) else 0.,
                    object_contact_shift_mean_mm=float(shift.mean()) if len(shift) else 0.,
                    object_contact_shift_max_mm=float(shift.max()) if len(shift) else 0.,
                    seconds=time.perf_counter()-solve_started,solver_wall_seconds=plan.seconds,solver=self.last_solver_diag)
                if args.variant=='admm-force':
                    record['object_contact_cost']=float(self.objective.contact_search.last['object_cost'][selected])
                    record['final_ranking']='FA_plus_object_contact_cost'
                window_diagnostics.append(record)
                with (case/'admm_windows.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            cursor['window']+=1
            return replace(plan,seconds=time.perf_counter()-solve_started)
    workflow.WindowOptimizer=FixedInitialOptimizer
    class EarlyDrop(Exception):pass
    early_stop={};run_steps=workflow._run_steps
    def checked_steps(ctx):
        try:run_steps(ctx)
        except EarlyDrop:pass
    workflow._run_steps=checked_steps
    if args.ablation=='dense-controls':
        def shift_dense(raw,dimension,config,*,execution_steps):
            from examples.retarget.fa_matched_ablation import shift_dense_matched
            return shift_dense_matched(raw,dimension,execution_steps=execution_steps,warm_scale=config.optimizer.warm_shift_scale)
        workflow._shift_warm=shift_dense
    if prefix is not None:install_prefix(workflow,prefix,case)
    initial_run=workflow._initial_run
    def initial(*a,**kw):
        shared_sequence['value']=kw['sequence']
        result=initial_run(*a,**kw)
        result=replace(result,qvel=np.zeros_like(result.qvel))
        if visualizer is not None:visualizer.update(result.qpos,result.qvel,0.)
        return result
    workflow._initial_run=initial
    visualizer=None
    if args.viewer:
        from examples.retarget.fa_viewer_bridge import ViewerProcess
        visualizer=ViewerProcess(p,case)
    workflow.make_solve_request=lambda config:SolveRequest.create(config.simulator.time_grid,duration_seconds=args.duration)
    # Initial q/v remain common across the eight arms. CA, contact guidance,
    # FA dynamics/cost, multistart and optimizer iterations remain unchanged.
    class MujocoExecutor:
        def __init__(self,runtime,index,cfg):
            self.model=runtime.cpu_model;self.data=mujoco.MjData(self.model);self.index=index;self.elapsed=0.
            self.substeps=round(cfg.simulator.time_grid.mpc_dt/self.model.opt.timestep)
        def execute_action_sequence(self,qpos,qvel,*,actions):
            d=self.data;d.qpos[:]=qpos;d.qvel[:]=qvel;qs=[];vs=[]
            for action in actions:
                d.ctrl[self.index.hand_ctrl]=action;d.qfrc_applied[:]=0.
                for _ in range(self.substeps):mujoco.mj_step(self.model,d)
                mujoco.mj_forward(self.model,d);qs.append(d.qpos.copy());vs.append(d.qvel.copy())
            self.elapsed+=len(actions)*self.substeps*self.model.opt.timestep
            return ActionExecution(qs[-1],vs[-1],np.array(qs),np.array(vs))
    if args.executor=='mujoco':
        def load_executor(config,comfree_warp,**kw):
            cpu=mujoco.MjModel.from_xml_path(str(p/'sequence/scene/wuji_right/sugar_box_mesh.xml'))
            cpu.opt.timestep=.002
            return ExecutorRuntime(cpu,None)
        workflow._load_executor_runtime=load_executor
        workflow.build_action_executor=lambda runtime,index,cfg:MujocoExecutor(runtime,index,cfg)
    # Preserve the original numerical workflow, save arrays without rendering
    # or an unrelated open-loop replay after completion.
    def save(config,recorder,*,compiled,cpu_model,index,sequence,**kw):
        from forceaware.output import _object_qpos
        arrays=recorder.arrays();arrays['object_reference_qpos']=_object_qpos(sequence.object_pose_xyzw,
            np.r_[arrays['qpos_frames'][0],arrays['target_frames']])
        np.savez_compressed(case/'rollout.npz',**arrays)
        if snapshots:
            np.savez_compressed(case/'contact_admm.npz',window_indices=np.array(snapshot_indices),
                **{k:np.stack([s[k] for s in snapshots]) for k in snapshots[0]})
        if args.ablation=='slow-templates':
            optimizer_for_save=cursor.get('optimizer')
            template_snapshots=optimizer_for_save._ablation_state['templates'].snapshots if optimizer_for_save is not None else []
            if template_snapshots:np.savez_compressed(case/'templates.npz',**{k:np.stack([s[k] for s in template_snapshots]) for k in template_snapshots[0]})
        (case/'config.json').write_text(json.dumps(config_dict(config),default=str,indent=2))
        (case/'audit.json').write_text(json.dumps(dict(executor=args.executor,closed_loop=True,
            initial_pose_refinement=False,initial_velocity='zero',duration=args.duration,
            original_FA_optimizer=True,rendering=False),indent=2))
        if early_stop:(case/'early_stop.json').write_text(json.dumps(early_stop,indent=2)+'\n')
        if args.variant!='baseline':
            audit=json.loads((case/'audit.json').read_text())
            audit.update(original_FA_optimizer=False,base_optimizer={
                'lbfgs':'independent batched L-BFGS hand block',
                'slsqp':'independent batched SciPy SLSQP hand block',
                'acados':'condensed control SQP/RTI with acados HPIPM and native ComFree oracle'}[args.hand_solver],
                variant=args.variant,ADMM=vars(settings),force_QP=False,template_query_targets_optimized=args.variant in ('admm','admm-force'),
                contact_mask_unchanged=True,force_gates=False,FA_cost_formulas_and_weights_unchanged=True,
                MANO_role='initialization_only' if args.variant in ('admm','admm-force') else 'fixed_template',
                final_ranking='FA_plus_object_contact_cost' if args.variant=='admm-force' else 'pure_FA')
            if args.variant=='admm-force':
                audit.update(force_model='native hard ComFree object/environment with ideal friction-cone contact forces',
                    static_force_balance_QP=False,force_shooting_projected_QP=args.contact_solver=='shooting',
                    contact_solver=args.contact_solver,
                    force_jacobian='analytic_native_ComFree_adjoint' if args.contact_solver=='batch-lm' else 'central_differences',
                    global_objective='F_FA(u,c) + object_weight * G_ComFree(z)',hand_FA_cost_unchanged=True,
                    physical_executor_receives='hand servo controls only',contact_memory_across_windows=True,
                    object_cost='trajectory pose + force effort',contact_search='MANO + incumbent + consensus + measured + local samples')
            audit.update(resumed=prefix is not None,auxiliary_snapshot_windows=len(snapshots),
                         auxiliary_prefix_missing=bool(resumed_windows and (not snapshot_indices or snapshot_indices[0]>0)))
            (case/'audit.json').write_text(json.dumps(audit,indent=2))
        if args.ablation!='none':
            audit=json.loads((case/'audit.json').read_text());audit['matched_ablation']=args.ablation
            audit['FA_cost_formulas_and_weights_unchanged']=args.ablation not in ('dense-smoothing','tracking-time')
            if args.ablation=='slow-templates':audit.update(template_query_targets_optimized=True,MANO_role='initialization_and_slow_template_updates')
            (case/'audit.json').write_text(json.dumps(audit,indent=2))
        return case
    workflow.save_outputs=save
    print_step=workflow._print_step
    def progress(window,total,plan,*,recorder):
        print_step(window,total,plan,recorder=recorder)
        if visualizer is not None:
            visualizer.update(recorder.qpos[-1],recorder.qvel[-1],(window+1)*.03,
                snapshots[-1] if snapshots else None)
        if window%10==0:
            temporary=case/'partial_tmp.npz'
            np.savez_compressed(temporary,**recorder.arrays());temporary.replace(case/'partial.npz')
            if snapshots:
                temporary=case/'partial_contact_tmp.npz'
                np.savez_compressed(temporary,window_indices=np.array(snapshot_indices),
                    **{k:np.stack([s[k] for s in snapshots]) for k in snapshots[0]})
                temporary.replace(case/'partial_contact_admm.npz')
        if args.stop_after_drop:
            elapsed=(len(recorder.qpos)-1)*.01
            poses=shared_sequence['value'].object_pose_xyzw
            frame=min(elapsed*30,len(poses)-1);i=int(np.floor(frame));j=min(i+1,len(poses)-1)
            z=float((1-frame+i)*poses[i,6]+(frame-i)*poses[j,6]);actual=float(recorder.qpos[-1][-5])
            if z-poses[0,6]>.05 and z-actual>.08:
                early_stop.update(reason='height_deficit_over_80mm',time=elapsed,planned_duration=args.duration,reference_height=z,actual_height=actual)
                raise EarlyDrop()
    workflow._print_step=progress
    started=time.time()
    try:workflow.rolling_mpc_retarget(cfg,config_path=p/'forceaware.yaml')
    finally:
        if visualizer is not None:visualizer.close()
    (case/'timing.json').write_text(json.dumps(dict(wall_seconds=time.time()-started)))


if __name__=='__main__':main()
