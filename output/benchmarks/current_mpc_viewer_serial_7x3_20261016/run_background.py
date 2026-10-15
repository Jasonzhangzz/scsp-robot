"""Serial visible-window evaluation: 42 asynchronous trials, matched source and seeds."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import statistics
import subprocess
import threading
import traceback

OUT = Path(__file__).resolve().parent
MANIFEST = OUT/'manifest.json'


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def metric(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values)>1 else None)


def summarize(rows):
    successful = [r for r in rows if r['success']]
    count = sum(r['planning_wall_count'] for r in rows)
    total = sum(r['planning_wall_total_s'] for r in rows)
    squares = sum(r['planning_wall_sum_squares_s2'] for r in rows)
    variance = max(0., (squares-total*total/count)/(count-1)) if count>1 else None
    return dict(trials=len(rows),successes=len(successful),success_rate=len(successful)/len(rows),
        execution_time_s=metric([r['completion_time_s'] for r in successful]),
        completion_wall_time_s=metric([r['completion_wall_time_s'] for r in successful]),
        planning_time_s=total/count if count else None,
        planning_time_std_s=variance**.5 if variance is not None else None,
        planning_cycles=count,
        final_pos_error_m=metric([r['final_pos_err'] for r in rows]),
        final_quat_error=metric([r['final_quat_err'] for r in rows]),
        stable_flip_successes=sum(bool(r['stable_flip_success']) for r in rows),
        physics_policy_ticks=sum(r['physics_policy_ticks'] for r in rows),
        mpc_failures=sum(r['mpc_failures'] for r in rows),
        lambda_failures=sum(r.get('lambda_failures',0) for r in rows),
        mpc_fallbacks=sum(r['mpc_fallbacks'] for r in rows),
        lambda_fallbacks=sum(r.get('lambda_fallbacks',0) for r in rows),
        planner_solver_backends=sorted({r['planner_solver_backend'] for r in rows}),
        successful_cases=[dict(object=r['object'],seed=r['seed']) for r in successful])


def main():
    manifest=json.loads(MANIFEST.read_text())
    assert manifest['status']=='prepared', 'Refusing to mix runs'
    source=Path(manifest['source_root'])
    python='/home/zz/miniconda3/envs/acc/bin/python'
    env=os.environ.copy()
    env['PATH']=str(Path(python).parent)+os.pathsep+env.get('PATH','')
    env['LD_LIBRARY_PATH']='/home/zz/miniconda3/envs/acc/lib:'+env.get('LD_LIBRARY_PATH','')
    env['DISPLAY']=manifest['display']
    env['XAUTHORITY']=manifest['xauthority']
    env['XDG_RUNTIME_DIR']='/run/user/1000'
    slots=queue.Queue()
    slots.put(sorted(os.sched_getaffinity(0)))
    manifest.update(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
                    jobs=1,cpu_pinning=False,allowed_cpus=sorted(os.sched_getaffinity(0)),
                    numerical_thread_environment={k:env.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    write_json(MANIFEST,manifest)
    complete={}
    lock=threading.Lock()
    print('Visible-window MPC: 42 matched trials; one case at a time; async planner; default pacing; no CPU pinning.',flush=True)

    def worker(case):
        obj,task=case
        directory=OUT/task/obj
        directory.mkdir(parents=True)
        path=directory/'trials.jsonl'
        command=[python,'-u',str(source/manifest['entry']),
            '--'+task,'--obj',obj,'--viewer','--async-planner',
            '--physics-backend','isaac','--trial-start','0','--trial_num','3',
            '--max_rollout_length','5000',
            '--ranking-metrics-jsonl',str(path),'--rollout-summary-json',str(directory/'rollout_summary.json')]
        cpus=slots.get()
        try:
            write_json(directory/'command.json',dict(argv=command,allowed_cpus=cpus,cpu_pinning=False))
            print('START %s/%s cpus=%s'%(task,obj,cpus),flush=True)
            with (directory/'run.log').open('w') as log:
                subprocess.run(command,
                    cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            assert sorted(r['seed'] for r in rows)==[0,1,2]
            for row in rows:
                assert row['object']==obj and not row['aborted']
                assert row['async_planner'] and row['headless_realtime'] and not row['headless']
                assert row['physics_backend']=='isaac'
                assert row['target_type']==('rotation' if task=='rotation' else 'ground-rotation')
                assert row['cpo_variant']=='full'
                assert row['steps']==row['planning_wall_count'] and row['steps']<=5000
                if row['success']: assert row['completion_time_s'] is not None
                row['task']=task
            with lock:
                complete[task+'/'+obj]=summarize(rows)
                write_json(OUT/'progress.json',dict(status='running',
                    completed_trials=sum(r['trials'] for r in complete.values()),cells=complete))
            print('DONE %s/%s: %d/3 successes'%(task,obj,sum(r['success'] for r in rows)),flush=True)
            return rows
        finally:
            slots.put(cpus)

    objects=['stanford_bunny2']+[o for o in manifest['objects'] if o!='stanford_bunny2']
    cases=[(obj,task) for obj in objects for task in ('flipping','rotation')]
    groups=[worker(case) for case in cases]
    rows=[row for group in groups for row in group]
    assert len(rows)==42 and len({(r['task'],r['object'],r['seed']) for r in rows})==42
    for file,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((source/file).read_bytes()).hexdigest()==digest,file
    results={}
    for task in manifest['tasks']:
        subset=[r for r in rows if r['task']==task]
        assert len(subset)==21
        results[task]=dict(overall=summarize(subset),objects={obj:summarize(
            [r for r in subset if r['object']==obj]) for obj in manifest['objects']})
    write_json(OUT/'trials.json',rows)
    write_json(OUT/'statistics.json',results)
    fields=['task','object','seed','success','steps','completion_time_s','completion_wall_time_s',
            'planning_wall_mean_ms','planning_wall_count','planning_wall_total_s',
            'final_pos_err','final_quat_err','stable_flip_success','async_planner','headless_realtime',
            'physics_policy_ticks','sim_end_time_s','planner_solver_backend','mpc_failures','mpc_fallbacks']
    with (OUT/'trials.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    def fmt(x,digits=3): return '--' if x is None else format(x,'.%df'%digits)
    report=['# Serial visible-window test_mpc_isaac.py benchmark','',
        'Seven objects, seeds 0/1/2 per object and task. 42 trials; real visible-window Isaac execution and asynchronous inference. One case runs at a time with no CPU pinning. Controller defaults, 5000-planning-cycle limit, source snapshot and seeds match the previous benchmark. Exact commands are retained.','',
        '| Task | Successes | Success rate | Execution (s) | Planning (s) | Position error (m) | Quaternion error |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for task in manifest['tasks']:
        r=results[task]['overall']
        report.append('| %s | %d/21 | %.2f%% | %s | %s | %s | %s |'%(task,r['successes'],
            100*r['success_rate'],fmt(r['execution_time_s']['mean'],2),fmt(r['planning_time_s']),
            fmt(r['final_pos_error_m']['mean']),fmt(r['final_quat_error']['mean'])))
    report+=['','Execution time uses simulator timestamps and only successful trials. Planning time pools all planning callbacks, including failed trials, excluding startup and compilation. Final errors include all 21 trials per task. JSON includes sample standard deviations, per-object summaries, stable flips, solver failures and fallback counts. Wall timing reflects serial visible-window execution without CPU pinning.','',
             'The paper table and PDF are not modified by this benchmark.']
    baseline=json.loads((Path(manifest['baseline_directory'])/'statistics.json').read_text())
    baseline_rows=json.loads((Path(manifest['baseline_directory'])/'trials.json').read_text())
    indexed={(r['task'],r['object'],r['seed']):r for r in baseline_rows}
    comparison={}
    report+=['','## Comparison with previous parallel headless batch','',
        '| Task | Previous successes | Viewer successes | Previous execution (s) | Viewer execution (s) |',
        '|---|---:|---:|---:|---:|']
    for task in manifest['tasks']:
        current=results[task]['overall'];previous=baseline[task]['overall']
        pairs=[dict(object=r['object'],seed=r['seed'],before_success=indexed[(task,r['object'],r['seed'])]['success'],
            viewer_success=r['success']) for r in rows if r['task']==task]
        comparison[task]=dict(previous=previous,viewer=current,paired_outcomes=pairs)
        report.append('| %s | %d/21 | %d/21 | %s | %s |'%(task,previous['successes'],current['successes'],
            fmt(previous['execution_time_s']['mean'],2),fmt(current['execution_time_s']['mean'],2)))
    report+=["","Execution means use each batch's successful trials; they are not paired time differences."]
    write_json(OUT/'comparison.json',comparison)
    (OUT/'report.md').write_text('\n'.join(report)+'\n')
    manifest.update(status='complete',completed_trials=42,completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(MANIFEST,manifest)
    write_json(OUT/'progress.json',dict(status='complete',completed_trials=42,cells=complete))
    print(json.dumps({t:results[t]['overall'] for t in results},indent=2),flush=True)
    print('COMPLETE: statistics.json, trials.json, trials.csv and report.md written.',flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        manifest=json.loads(MANIFEST.read_text())
        manifest.update(status='failed',error=str(error))
        write_json(MANIFEST,manifest)
        traceback.print_exc()
        raise
