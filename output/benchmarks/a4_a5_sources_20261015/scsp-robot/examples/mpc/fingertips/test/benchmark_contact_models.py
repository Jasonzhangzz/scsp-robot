"""Matched SCM / native MuJoCo-QP / coupled LCP point-force trials.

Each model selects contact points AND forces using its own predictions. All
three use the same real MuJoCo executor, initial seeds, objectives and stopping
criteria. Initialization and writing result files are outside execution timing.
"""
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from examples.mpc.fingertips.test.test_scm_mujoco import build_parser, _run_rollout, _write_rows
from planning.rollout_timing import TIMING_DEFINITIONS, trial_timing, aggregate_trials


def summarize(out):
    trials, groups, starts = [], {}, {}
    for path in sorted(out.glob('trial_*.json')):
        data = json.loads(path.read_text())
        meta, rows = data['metadata'], data['rows']
        trial = meta['trial_summary']
        trials.append(trial)
        key = (trial['model'], trial['object'])
        group = groups.setdefault(key, dict(trials=[], rows=[]))
        group['trials'].append(trial); group['rows'].extend(rows)
        if rows:
            pose = rows[0]['initial_object_pose']
            start_key = trial['object'], trial['seed']
            if start_key in starts:
                np.testing.assert_allclose(pose, starts[start_key], atol=1e-12)
            starts[start_key] = pose
    by_object = {model:{} for model in dict.fromkeys(t['model'] for t in trials)}
    for (model,obj), group in groups.items():
        by_object[model][obj] = aggregate_trials(**group)
    overall = {}
    for model in by_object:
        ts, rs = [], []
        for (m,_), group in groups.items():
            if m == model:
                ts.extend(group['trials']); rs.extend(group['rows'])
        overall[model] = aggregate_trials(ts, rs)
    report = dict(timing_definitions=TIMING_DEFINITIONS, models=overall,
                  by_object=by_object, trials=trials, matched_initial_states=True,
                  notes=[
                      'Execution time is success-only wall time, including planning and physics; failed/capped trials are not successful completion times.',
                      'Planning time includes all candidate solves, prediction and selection per round. Aggregate planning samples include failed trials.',
                      'SCM retains the reference Isaac quasi-dynamic model and ACADOS force optimizer.',
                      'QP is native MuJoCo Newton with pyramidal friction, evaluated in a separate model/data at measured state.',
                      'LCP is the coupled physical-mass, regularized nonnegative friction-ray LCP, solved with scaled QuantEcon Lemke and certified on the unscaled problem.',
                      'QP/LCP use finite-difference local pose linearization and SLSQP force solves, then rank exact model predictions. Timing compares model+solver implementations, not model equations alone.',
                      'QP/LCP physical predictors average the 0.5ms pulse impulse over one 20ms Euler policy step. All executions use 40 physical substeps.',
                      'Identical seeds, point samples, force bounds, objective stages, ideal-point executor, pulse duration and success/settling thresholds are used.',
                      'Every attempted trial is included in the success denominator; solver exceptions have null completion time.'
                  ])
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    fields=['model','object','seed','success','steps','execution_time_s','elapsed_wall_time_s',
            'simulation_time_s','planning_mean_ms','planning_p95_ms','final_pos','final_quat','error']
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for t in trials:
            row={k:t.get(k) for k in fields}
            row['planning_mean_ms']=t['planning_time_ms']['mean']
            row['planning_p95_ms']=t['planning_time_ms']['p95']
            writer.writerow(row)
    lines=['| Model | Success | Execution time (s), successful trials | Planning time (ms), all rounds |',
           '| --- | --- | --- | --- |']
    for model,s in overall.items():
        e,p=s['execution_time_s_success_only'],s['planning_time_ms_all_rounds']
        fmt=lambda a:'—' if a['n']==0 else '%.3f ± %.3f (n=%d)'%(a['mean'],a['std'],a['n'])
        lines.append('| %s | %d/%d | %s | %s |'%(model,s['successes'],s['total_trials'],fmt(e),fmt(p)))
    lines += ['', 'Population standard deviation; failures are excluded only from successful-completion time.',
              'Wall-clock execution includes planning; simulation-time values are saved separately.',
              'QP/LCP force optimization differs from ACADOS, so planning time measures each full implementation.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    return report


def main():
    parser=build_parser()
    parser.add_argument('--objects', default='foam_brick,elephant,stanford_bunny2')
    parser.add_argument('--models', default='scm,qp,lcp')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--trial-start', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--analyze-only', action='store_true')
    args=parser.parse_args()
    models=[s.strip() for s in args.models.split(',')]
    if args.trials<1 or args.trial_start<0 or any(m not in ('scm','qp','lcp') for m in models):
        parser.error('trials >= 1, trial-start >= 0, models must be scm,qp,lcp')
    objects=[s.strip() for s in args.objects.split(',') if s.strip()]
    if not objects or len(models)!=len(set(models)):
        parser.error('objects must be nonempty and models unique')
    stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S')
    out=(args.output_dir or ROOT/'output/diagnostics'/('contact_models_'+stamp)).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if args.analyze_only:
        summarize(out);return
    protocol={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
              if k not in ('output_dir','resume','analyze_only','models','objects','trials','trial_start')}
    # Run sequentially so methods do not compete with each other for CPU time.
    for obj in objects:
        for seed in range(args.trial_start,args.trial_start+args.trials):
            for model in models:
                path=out/('trial_%s_%s_%03d.json'%(model,obj,seed))
                if path.exists():
                    old=json.loads(path.read_text())['metadata']
                    if args.resume and old.get('protocol')==protocol:
                        continue
                    raise FileExistsError('Result exists or protocol changed: %s'%path)
                run=copy.deepcopy(args);run.skip_model_comparison=True
                mode='surrogate' if model=='scm' else model
                print('benchmark model=%s object=%s seed=%d'%(model,obj,seed),flush=True)
                error=None
                try:
                    rows=_run_rollout(run,mode,obj,seed=seed,mass=None,steps=args.steps)
                except (RuntimeError,ValueError,FloatingPointError) as exc:
                    rows=getattr(exc,'rows',[]);error='%s: %s'%(type(exc).__name__,exc)
                    print(error,flush=True)
                last=rows[-1] if rows else {}
                trial=dict(model=model,object=obj,seed=seed,steps=len(rows),
                           success=bool(last.get('success',False)),settled=bool(last.get('settled',False)),
                           final_pos=last.get('goal_pos'),final_quat=last.get('goal_quat'),
                           error=error,**trial_timing(rows))
                _write_rows(path,rows,dict(protocol=protocol,trial_summary=trial,
                            timing_definitions=TIMING_DEFINITIONS))
                summarize(out)
    summarize(out)


if __name__=='__main__':
    main()
