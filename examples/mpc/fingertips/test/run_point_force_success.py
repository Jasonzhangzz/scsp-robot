#!/usr/bin/env python3
"""Run 7 x 10 paired synchronous trials for model and solver comparisons."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['SCSP_PLANNER_ONLY']='1'
os.environ['MUJOCO_HEADLESS']='1'
import argparse
import contextlib
import copy
import csv
from datetime import datetime
import hashlib
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
from examples.mpc.fingertips.test.test_scm_mujoco import build_parser as legacy_parser
from planning.point_force_benchmark import (OBJECTS,TrialRows,scene,surface_points,run_rollout,
    trial_summary,write_json,write_rows,json_value)
from planning.point_force_success import SuccessPlanner
from planning.rollout_timing import distribution
PAPER_SCM_LAW = 'spectral_residual_one_step'

CONFIGS={'scm_gpu_lm':('scm_corrected','gpu-lm'),'comfree_gpu_lm':('comfree','gpu-lm'),
         'contactsdf_gpu_lm':('contactsdf','gpu-lm'),'contactsdf_lm':('contactsdf','lm'),
         'qp_adamw':('qp','adamw'),'comfree_lm':('comfree','lm'),'scm_lm':('scm_corrected','lm'),
         'scm_cmaes':('scm_corrected','cma-es'),'scm_cem':('scm_corrected','cem'),
         'scm_ipopt':('scm_corrected','ipopt'),'qp_lm':('qp','lm')}
DEFAULT_CONFIGS=('qp_adamw','comfree_gpu_lm','contactsdf_gpu_lm','scm_gpu_lm','scm_cmaes','scm_cem','scm_ipopt')
SOURCES=['models/corrected_scm.py','models/comfree_contact.py','planning/point_force_success.py',
    'planning/point_force_lm_gpu.py','planning/point_force_benchmark.py','planning/point_force_gpu.py','planning/point_force_prediction.py',
    'planning/object_pose_cost.py','planning/rollout_timing.py','examples/mpc/fingertips/test/params.py',
    'examples/mpc/fingertips/test/run_point_force_success.py',
    'examples/mpc/franka/scm/contact_dynamics.py','examples/mpc/franka/scm/residual_scm.py']


def parser():
    p=legacy_parser()
    p.description=__doc__
    for names,kw in [
        (['--objects'],dict(default=','.join(OBJECTS))),
        (['--configs'],dict(default=','.join(DEFAULT_CONFIGS))),
        (['--trials'],dict(type=int,default=10)),
        (['--jobs'],dict(type=int,default=3)),
        (['--output-dir'],dict(type=Path,default=ROOT/'output/benchmarks/point_force_success_v2_20261022')),
        (['--adamw-iterations'],dict(type=int,default=20)),
        (['--policy-dt'],dict(type=float,default=.02)),
        (['--prediction-substeps'],dict(type=int,default=1)),
        (['--arm-friction'],dict(type=float,default=1.2)),
        (['--linear-velocity-coef'],dict(type=float,default=0.)),
        (['--angular-velocity-coef'],dict(type=float,default=0.)),
        (['--pose-velocity-scale'],dict(type=float,default=1.)),
        # Per-interval object delta-v a single contact command may deliver.
        # With a 10 g object, 3 N over 20 ms is 6 m/s: an unbounded one-step
        # dead-beat impulse launches the object off the table regardless of
        # the contact model, so the feasible set must bound the impulse.
        (['--max-linear-force-delta-v'],dict(type=float,default=.3)),
        (['--max-angular-force-delta-v'],dict(type=float,default=3.)),
        (['--async-result-wait-ms'],dict(type=float,default=None)),
        (['--async-max-command-age'],dict(type=float,default=.1)),
        (['--async-planning-budget-ms'],dict(type=float,default=None)),
        (['--target-frame'],dict(default='asset')),
        (['--target-height-mode'],dict(default='supported')),
        (['--target-clearance'],dict(type=float,default=.0005)),
        (['--target-height'],dict(type=float,default=.03)),
        (['--comfree-stiffness'],dict(type=float,default=.1)),
        (['--comfree-damping'],dict(type=float,default=.001)),
        (['--softplus-temperature'],dict(type=float,default=1e-4)),
        (['--lcp-tolerance'],dict(type=float,default=1e-8)),
        (['--lcp-max-iterations'],dict(type=int,default=1000)),
        (['--contactsdf-sigma'],dict(type=float,default=120.)),
        (['--device'],dict(default='cuda:0')),
        (['--resume'],dict(action='store_true')),
        (['--summarize-only'],dict(action='store_true')),
        (['--smoke'],dict(action='store_true'))]:
        if not any(name in p._option_string_actions for name in names):p.add_argument(*names,**kw)
    p.set_defaults(solver='lm',sample_num=70,steps=5000,max_contact_force=3.,
        force_reg_coef=.01,friction_reg_coef=.03,pos_coef=500.,ori_coef=20.,
        random_init_tilt=True,init_tilt_deg=75.,init_tilt_min_deg=35.,two_stage_object_cost=True,
        contact_noslip_iterations=0)
    return p


def worker(task):
    args,config,obj,seed=task
    name,solver=CONFIGS[config]
    cfg=copy.deepcopy(args);cfg.obj=obj;cfg.solver=solver;cfg.viewer=False;cfg.async_planner=False
    destination=args.output_dir/config/f'{obj}_{seed:03d}.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    rows=TrialRows();error=None
    with destination.with_suffix('.log').open('w',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        try:
            rows=run_rollout(cfg,name,obj,seed=seed,steps=args.steps,planner_class=SuccessPlanner)
            expected=json.loads((args.output_dir/'initial_states.json').read_text())[f'{obj}:{seed}']
            np.testing.assert_array_equal(rows.initialization['initial_pose'],expected['pose'])
            np.testing.assert_array_equal(rows.initialization['target_pose'],expected['target'])
            np.testing.assert_array_equal(rows.initialization['sample_points'],expected['points'])
            for row in rows:
                info=row['execution_info']
                assert info['pose_overwrites']==0 and not info['force_was_clipped']
                if row['command_valid_for_step']:
                    assert info['active_substeps']==args.force_physics_substeps
                    np.testing.assert_allclose(info['actual_applied_impulse_body'][:3],
                        np.asarray(row['applied_force_body'])*args.policy_dt,atol=1e-12)
        except Exception as exc:
            rows=getattr(exc,'rows',rows);error=f'{type(exc).__name__}: {exc}'
            traceback.print_exc()
        summary=trial_summary(rows,name,obj,seed,False,error)
        if error:summary['success']=False
        summary.update(configuration=config,solver=solver,
                       law_version=PAPER_SCM_LAW if name=='scm_corrected' else name)
        print('TRIAL_RESULT '+json.dumps(json_value(summary)),flush=True)
    archive=destination.with_suffix('.json.gz');temporary=archive.with_suffix('.gz.tmp')
    with gzip.open(temporary,'wt',compresslevel=4) as stream:
        json.dump(json_value(dict(metadata=dict(trial_summary=summary),rows=list(rows),
            planning_rounds=rows.planning_rounds,initialization=rows.initialization)),stream,allow_nan=False)
    temporary.replace(archive)
    write_json(destination.with_suffix('.summary.json'),summary)
    return summary


def summarize(args):
    trials=[]
    for config in args.configs:
        for path in sorted((args.output_dir/config).glob('*.summary.json')):
            trials.append(json.loads(path.read_text()))
    groups=[]
    for config in args.configs:
        for obj in [*args.objects,'ALL']:
            subset=[t for t in trials if t['configuration']==config and (obj=='ALL' or t['object']==obj)]
            if not subset:continue
            good=[t for t in subset if t['success']]
            groups.append(dict(configuration=config,object=obj,successes=len(good),total=len(subset),
                success_rate=len(good)/len(subset),steps_success_only=distribution([t['steps'] for t in good]),
                failures=[dict(object=t['object'],seed=t['seed'],reason=t['failure_reason'],error=t['error']) for t in subset if not t['success']]))
    expected=len(args.configs)*len(args.objects)*args.trials
    report=dict(completed=len(trials),expected=expected,complete=len(trials)==expected,groups=groups,trials=trials)
    write_json(args.output_dir/'summary.json',report)
    with (args.output_dir/'success_rates.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['configuration','object','successes','total','success_rate'])
        writer.writeheader();writer.writerows({k:g[k] for k in writer.fieldnames} for g in groups)
    lines=['| Object | '+' | '.join(args.configs)+' |','|---|'+'---:|'*len(args.configs)]
    for obj in [*args.objects,'ALL']:
        cells=[]
        for config in args.configs:
            g=next((g for g in groups if g['configuration']==config and g['object']==obj),None)
            cells.append(f'{g["successes"]}/{g["total"]} ({100*g["success_rate"]:.1f}%)' if g else 'pending')
        lines.append('| '+obj+' | '+' | '.join(cells)+' |')
    (args.output_dir/'success_rates.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    p=parser();args=p.parse_args()
    args.objects=args.objects.split(',');args.configs=args.configs.split(',')
    if not set(args.objects)<=set(OBJECTS) or not set(args.configs)<=set(CONFIGS):p.error('Unknown object/configuration')
    if args.prediction_substeps!=1:p.error('This experiment uses one frozen-contact prediction step')
    if args.trials<1 or args.jobs<1:p.error('trials/jobs must be positive')
    if args.smoke:args.trials=1;args.steps=3;args.objects=['foam_brick']
    args.output_dir=args.output_dir.resolve();args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:summarize(args);return
    protocol=dict(model_law=PAPER_SCM_LAW,temperature=args.softplus_temperature,configs={c:CONFIGS[c] for c in args.configs},
        objects=args.objects,seeds=list(range(args.trials)),expected_trials=len(args.configs)*len(args.objects)*args.trials,
        synchronous=True,policy_dt=args.policy_dt,physics_substeps=args.force_physics_substeps,max_steps=args.steps,
        execution='body-fixed point force held on every physics substep; zero extra torque; no collider or pose overwrite',
        force_feasible_set=dict(max_contact_force=args.max_contact_force,arm_friction=args.arm_friction,
            max_linear_force_delta_v=args.max_linear_force_delta_v,max_angular_force_delta_v=args.max_angular_force_delta_v,
            note='identical cone and per-interval delta-v caps on CPU (force_commands) and GPU (bounded_force) paths'),
        prediction='one frozen-contact 20 ms full-dynamic step; common measured mass, contacts, friction and objective',
        success='position<0.02m, quaternion metric<0.015, 10 unforced settled intervals',
        comfree=dict(commit='ba8b9969f5f7859d51b1f738f458f8d2f49be080',stiffness=args.comfree_stiffness,damping=args.comfree_damping,
                     implementation='upstream efc row mass/forward law ported to float64 on common MuJoCo contacts'),
        contactsdf=dict(commit='3a21edf8e51788f72a7d77259af4dd584d5ec43f',sigma=args.contactsdf_sigma,
            implementation='dynamic LSE contact-law extension Q=M/h^2 on common MuJoCo contacts; not the official full robot controller'),
        solvers='gpu-lm: float64 CUDA batched LM, all residuals/Jacobians/3x3 systems/accept-reject on GPU, central difference 1e-4, fixed iteration budget, graph capture; historical lm: batched CPU LM with central finite-difference residual Jacobians; QP outer Torch AdamW with central finite-difference gradients and CUDA native-objective inner QP; NumPy full-covariance CMA-ES with cumulative paths and bound repair; CUDA CEM; separable batched CasADi IPOPT with exact AD',
        args=vars(args),source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES},
        mujoco=mujoco.__version__,python=sys.executable,started_at=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat())
    path=args.output_dir/'protocol.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if not args.resume:raise RuntimeError('Output exists; use --resume with the identical protocol')
        current=json_value(protocol)
        for item in ('source_sha256','configs','objects','seeds','max_steps','policy_dt','physics_substeps'):
            if previous[item]!=current[item]:raise RuntimeError(f'Cannot mix cohorts: {item} changed')
    else:
        write_json(path,protocol)
        for name in SOURCES:
            target=args.output_dir/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
        initial={}
        for obj in args.objects:
            for seed in range(args.trials):
                m,d,target,_=scene(args,obj,seed);points,frames=surface_points(m,d,args.sample_num,seed)
                initial[f'{obj}:{seed}']=dict(pose=d.qpos.copy(),velocity=d.qvel.copy(),target=target,points=points,frames=frames)
        write_json(args.output_dir/'initial_states.json',initial)
    tasks=[(args,c,o,s) for s in range(args.trials) for o in args.objects for c in args.configs
           if not (args.output_dir/c/f'{o}_{s:03d}.summary.json').exists()]
    def status(state,**kw):
        report=summarize(args)
        write_json(args.output_dir/'STATUS.json',dict(status=state,pid=os.getpid(),completed=report['completed'],
            expected=report['expected'],updated_at=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),**kw))
    status('running')
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker,t) for t in tasks]
        for future in as_completed(futures):
            result=future.result()
            print(f'{result["configuration"]} {result["object"]} seed={result["seed"]} '
                  f'success={result["success"]} steps={result["steps"]} error={result["error"]}',flush=True)
            status('running',last_trial=result)
    report=summarize(args)
    status('completed',complete=report['complete'])
    print((args.output_dir/'success_rates.md').read_text(),flush=True)

if __name__=='__main__':main()
