#!/usr/bin/env python3
"""Seven objects x three seeds x four models; sync completes before async."""
import contextlib
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from examples.mpc.fingertips.test.test_scm_mujoco import build_parser,_run_rollout
from planning.point_force_benchmark import (MODELS,OBJECTS,TrialRows,trial_summary,
    write_json,write_rows,json_value)
from planning.rollout_timing import distribution
import numpy as np

SOURCES = ['models/corrected_scm.py','planning/point_force_gpu.py',
    'planning/point_force_rollout_gpu.py','planning/point_force_collision_gpu.py',
    'planning/point_force_prediction.py',
    'examples/mpc/fingertips/test/audit_contact_benchmark.py',
    'examples/mpc/fingertips/test/run_contact_benchmark_background.py',
    'planning/point_force_benchmark.py','planning/object_pose_cost.py',
    'examples/mpc/fingertips/test/test_scm_mujoco.py',
    'examples/mpc/fingertips/test/benchmark_contact_models.py',
    'examples/mpc/fingertips/test/params.py',
    'examples/mpc/franka/scm/contact_dynamics.py',
    'examples/mpc/franka/scm/lcp_backend.py',
    'examples/mpc/franka/scm/scm_backend.py',
    'examples/mpc/franka/scm/runtime_env.py','planning/rollout_timing.py']


def run_trial(args,model,obj,seed,output,async_mode=False,viewer=False,replay=None):
    cfg = copy.deepcopy(args)
    cfg.obj,cfg.contact_model,cfg.async_planner,cfg.viewer = obj,model,async_mode,viewer
    cfg.replay_schedule = replay
    rows,error = TrialRows(),None
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix('.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            rows = _run_rollout(cfg,model,obj,seed=seed,steps=args.steps)
        except Exception as exc:
            rows = getattr(exc,'rows',rows)
            error = f'{type(exc).__name__}: {exc}'
            traceback.print_exc()
        summary = trial_summary(rows,model,obj,seed,async_mode,error)
        print('TRIAL_RESULT '+json.dumps(json_value(summary)),flush=True)
    write_rows(output,rows,dict(trial_summary=summary,
        formal_trial=output.parent.name in ('sync','async')))
    return rows,summary


def preflight(args):
    results,backends = [],{}
    for model in args.models:
        for obj in args.objects:
            for seed in range(args.trial_start,args.trial_start+args.trials):
                cfg = copy.copy(args);cfg.steps=1
                rows,summary = run_trial(cfg,model,obj,seed,
                    args.output_dir/'preflight'/f'{model}_{obj}_{seed:03d}.json')
                if summary['error']:
                    raise RuntimeError(f'Preflight failed: {summary}')
                results.append(dict(model=model,object=obj,seed=seed,
                    validation=rows.initialization['validation']))
                backend = summary['actual_backend']
                previous = backends.get(model)
                backends[model] = 'cpu-adamw' if 'cpu-adamw' in (previous,backend) else backend
                print(f'preflight {model} {obj} seed={seed}: {backend}',flush=True)
    write_json(args.output_dir/'backend_validation.json',dict(locked_backends=backends,results=results))
    return backends


def compare_trials(reference,candidate):
    keys = ('object_pose','velocity_body','best_contact_point','optimal_force_contact','applied_force_body')
    maximum = {}
    first = None
    for i,(a,b) in enumerate(zip(reference,candidate)):
        for key in keys:
            delta = float(np.max(np.abs(np.asarray(a[key])-b[key])))
            maximum[key] = max(maximum.get(key,0.),delta)
            if delta>1e-7 and first is None:
                first = dict(step=i+1,field=key,max_absolute_error=delta)
        for key in ('selected_idx','execution_substeps','execution_completed','completed_execution_steps','mujoco_unstable'):
            if a.get(key)!=b.get(key) and first is None:
                first = dict(step=i+1,field=key)
    same_success = bool(reference and candidate and reference[-1]['success']==candidate[-1]['success'])
    return dict(passed=first is None and len(reference)==len(candidate) and same_success,
        reference_steps=len(reference),viewer_steps=len(candidate),same_success=same_success,
        maximum_absolute_errors=maximum,first_difference=first)


def viewer_validation(args):
    results = []
    for async_mode in (False,True):
        for model in args.models:
            directory = args.output_dir/'viewer_consistency'/('async' if async_mode else 'sync')/model
            reference,summary = run_trial(args,model,'foam_brick',0,directory/'headless.json',async_mode=async_mode)
            reference_error = summary['error']
            if reference_error and not reference:
                raise RuntimeError(f'Viewer reference failed: {summary}')
            replay = copy.deepcopy(reference.planning_rounds) if async_mode else None
            viewed,summary = run_trial(args,model,'foam_brick',0,directory/'viewer.json',
                async_mode=async_mode,viewer=True,replay=replay)
            result = compare_trials(reference,viewed)
            result.update(model=model,rollout='async' if async_mode else 'sync',
                          viewer_status=viewed.initialization.get('viewer_status'),error=summary['error'],
                          reference_error=reference_error,
                          asynchronous_publish_schedule_replayed=async_mode)
            # A failed physical trial is still a complete parity experiment:
            # reproduce both its trajectory AND its terminal failure. Never
            # require the benchmark models to succeed to enter the denominator.
            if summary['error'] != reference_error or not result['viewer_status']:
                result['passed'] = False
            if async_mode:
                live,live_summary = run_trial(args,model,'foam_brick',0,directory/'viewer_realtime.json',
                    async_mode=True,viewer=True)
                result['realtime_viewer'] = live_summary
                result['realtime_headless'] = trial_summary(reference,model,'foam_brick',0,True)
                result['realtime_schedule_may_differ'] = True
                result['realtime_outcomes_differ'] = (live_summary['success']!=result['realtime_headless']['success'] or
                    live_summary['steps']!=result['realtime_headless']['steps'] or live_summary['error']!=reference_error)
                if not live.initialization.get('viewer_status'):
                    result['passed'] = False
            results.append(result)
            write_json(directory/'comparison.json',result)
            write_json(args.output_dir/'viewer_consistency'/'summary.json',dict(passed=all(r['passed'] for r in results),results=results))
            print(f'viewer parity {model} {result["rollout"]}: {result["passed"]}',flush=True)
            if not result['passed']:
                raise AssertionError(f'Visualizer consistency failed: {result}')
    return results


def summarize(out):
    trials,groups = [],{}
    for path in sorted(out.glob('*/trial_*.json')):
        if path.parent.name not in ('sync','async'):
            continue
        payload = json.loads(path.read_text())
        trial = payload['metadata']['trial_summary']
        trials.append(trial)
        for key in ((trial['rollout'],trial['model'],trial['object']),
                    (trial['rollout'],trial['model'],'ALL')):
            group = groups.setdefault(key,dict(trials=[],times=[]))
            group['trials'].append(trial)
            group['times'].extend(r['planning_time_ms'] for r in payload['planning_rounds'] if r.get('planning_time_ms') is not None)
    statistics = []
    for (rollout,model,obj),group in sorted(groups.items()):
        ts = group['trials'];successful = [t for t in ts if t['success']]
        statistics.append(dict(rollout=rollout,model=model,object=obj,
            successes=len(successful),total_trials=len(ts),success_rate=len(successful)/len(ts),
            completion_steps_success_only=distribution([t['completion_steps'] for t in successful]),
            executed_steps_all_trials=distribution([t['steps'] for t in ts]),
            planning_time_ms_all_rounds=distribution(group['times']),
            planning_rounds=len(group['times']),failures=[dict(seed=t['seed'],object=t['object'],error=t['error'],
                reason=t.get('failure_reason')) for t in ts if not t['success']]))
    protocol=json.loads((out/'protocol.json').read_text())
    expected_trials=2*len(protocol['models'])*len(protocol['objects'])*len(protocol['seeds'])
    report = dict(completed_trials=len(trials),expected_trials=expected_trials,
        completed_sync=sum(t['rollout']=='sync' for t in trials),
        completed_async=sum(t['rollout']=='async' for t in trials),statistics=statistics,trials=trials,
        definitions=dict(steps='completed policy cycles; attempted_steps includes a terminal partial cycle; completion_steps null for failures',
            planning='full actual planning calls; reused commands have no timing sample; completed unused async calls included',
            initialization='scene construction, JIT/capture warmup and equation validation excluded',
            failure_denominator='all attempted formal trials, including solver/physics exceptions'))
    write_json(out/'summary.json',report)
    import csv
    fields = ['rollout','model','object','seed','success','steps','attempted_steps','completion_steps','planning_rounds',
              'planning_mean_ms','planning_p95_ms','final_pos','final_quat','actual_backend','failure_reason','async_deadline_misses','error']
    with (out/'summary.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for trial in trials:
            row = {k:trial.get(k) for k in fields}
            row.update(planning_mean_ms=trial['planning_time_ms']['mean'],planning_p95_ms=trial['planning_time_ms']['p95'])
            writer.writerow(row)
    lines = ['rollout model object success completion_steps_mean planning_mean_ms planning_p95_ms']
    for item in statistics:
        lines.append(f'{item["rollout"]} {item["model"]} {item["object"]} '
            f'{item["successes"]}/{item["total_trials"]} '
            f'{item["completion_steps_success_only"]["mean"]} '
            f'{item["planning_time_ms_all_rounds"]["mean"]} {item["planning_time_ms_all_rounds"]["p95"]}')
    (out/'summary.log').write_text('\n'.join(lines)+'\n')
    return report


def main():
    p = build_parser()
    p.add_argument('--objects',default=','.join(OBJECTS))
    p.add_argument('--models',default=','.join(MODELS))
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--phase',choices=('all','preflight','viewer','sync','async','summary'),default='all')
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    args.objects,args.models = args.objects.split(','),args.models.split(',')
    if (not args.objects or not set(args.objects)<=set(OBJECTS) or not set(args.models)<=set(MODELS)
        or len(set(args.objects))!=len(args.objects) or len(set(args.models))!=len(args.models) or args.trials<1):
        p.error('Choose unique known objects/models and trials >= 1')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    args.output_dir = (args.output_dir or ROOT/'log'/f'contact_models_{stamp}').resolve()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.phase == 'summary':
        summarize(args.output_dir);return
    if args.phase in ('all','preflight'):
        args.locked_backends = preflight(args)
    else:
        args.locked_backends = json.loads((args.output_dir/'backend_validation.json').read_text())['locked_backends']
    sources = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    import mujoco,warp,torch
    protocol = dict(objects=args.objects,models=args.models,seeds=list(range(args.trial_start,args.trial_start+args.trials)),
        rollout_order=['sync','async'],max_steps=args.steps,policy_dt=args.policy_dt,
        physics_substeps=args.force_physics_substeps,force_duration=args.policy_dt,
        args={k:v for k,v in vars(args).items() if k not in ('phase','resume','output_dir')},
        backends=args.locked_backends,source_sha256=sources,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        python=sys.executable,mujoco=mujoco.__version__,warp=warp.__version__,torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0),
        scm='scm_corrected: dynamic diagonal impulse plus one global alpha per prediction substep; r=1e-8; no soft response',
        contact_sdf='existing untrained dynamic extension Q=M/h^2; sigma=120',
        prediction=f'{args.prediction_substeps} substeps over {args.policy_dt}s; candidate-dependent geometry, inertia and dynamics; float64 GPU geometry and rigid-body dynamics, independent float64 contact solves; CPU MuJoCo trajectory oracle',
        collision='MuJoCo 3.6 plane-mesh support/neighbor algorithm shared by all models; vendored Warp four-contact collision replaced',
        force_constraint=dict(norm_N=args.max_contact_force,
            force_only_COM_delta_v_m_s=args.max_linear_force_delta_v,
            force_only_angular_delta_v_rad_s=args.max_angular_force_delta_v,
            definition='norm(h*F/m) and norm(h*I_COM^-1*((p-COM) cross F)); constraints parameterized inside every candidate, no executor clipping'),
        asynchronous='measured-state latest-result publication; bounded hold with age/contact/total-impulse checks; adaptive search budget',
        async_result_wait_ms=800*args.policy_dt if args.async_result_wait_ms is None else args.async_result_wait_ms,
        async_wait_semantics='wall-clock bounded wait before physics; fast results apply to the same measured simulation state',
        target=dict(frame=args.target_frame,height_mode=args.target_height_mode,requested_height=args.target_height,
                    clearance=args.target_clearance),
        solver_budget=dict(cem_samples_per_point=args.population_size,cem_generations=args.population_iterations,
            cem_elite_fraction=args.cem_elite_fraction,candidate_points=args.sample_num,
            async_budget_ms=800*args.policy_dt if args.async_planning_budget_ms is None else args.async_planning_budget_ms,
            prediction_substeps=args.prediction_substeps,lcp_max_iterations=args.lcp_max_iterations,
            lcp_residual_tolerance=args.lcp_tolerance,qp_newton_max_iterations=100,
            qp_line_search_max_iterations=40,qp_stationarity_tolerance=args.lcp_tolerance),
        execution=f'continuous physical body-fixed point force, {args.force_physics_substeps} substeps, no pose/velocity overwrites',
        success='position<0.02m, quaternion metric<0.015, 10 unforced intervals and settled')
    from importlib.metadata import version
    protocol['dependency_versions']={name:version(name) for name in
        ('numpy','scipy','trimesh','mujoco','warp-lang','torch','casadi')}
    protocol['nvidia_driver']=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version',
        '--format=csv,noheader'],text=True).strip()
    path = args.output_dir/'protocol.json'
    existing_formal = any(args.output_dir.glob('sync/trial_*.json')) or any(args.output_dir.glob('async/trial_*.json'))
    if path.exists():
        previous = json.loads(path.read_text())
        changed = previous['source_sha256']!=sources or previous['args']!=json_value(protocol['args'])
        if changed and (existing_formal or args.phase not in ('all','preflight')):
            raise RuntimeError('Source or experimental protocol changed; use a fresh output directory')
    if not path.exists() or args.phase in ('all','preflight'):
        write_json(path,protocol)
        for name in SOURCES:
            destination = args.output_dir/'source'/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes((ROOT/name).read_bytes())
    if args.phase == 'preflight':
        return
    if args.phase in ('all','viewer'):
        viewer_validation(args)
    if args.phase == 'viewer':
        return
    verification = json.loads((args.output_dir/'viewer_consistency'/'summary.json').read_text())
    if not verification.get('passed'):
        raise RuntimeError('Formal trials require passing visualizer parity checks')
    for rollout in ('sync','async'):
        if args.phase not in ('all',rollout):
            continue
        if rollout == 'async':
            expected = len(args.models)*len(args.objects)*args.trials
            if len(list((args.output_dir/'sync').glob('trial_*.json')))!=expected:
                raise RuntimeError('Complete all synchronous trials before asynchronous trials')
        for model in args.models:
            for obj in args.objects:
                for seed in range(args.trial_start,args.trial_start+args.trials):
                    output = args.output_dir/rollout/f'trial_{model}_{obj}_{seed:03d}.json'
                    if output.exists():
                        if args.resume:
                            continue
                        raise FileExistsError(str(output))
                    _,summary = run_trial(args,model,obj,seed,output,async_mode=rollout=='async')
                    print(json.dumps(json_value(summary)),flush=True)
                    summarize(args.output_dir)
    summarize(args.output_dir)


if __name__ == '__main__':
    main()
