"""Isolated SCM solver trials with persistent logs and per-trial results.

Run one controller per --solver. Children share the controller's process group
so a detached batch can be stopped with kill -- -<controller-pid>.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
OBJECTS = ('foam_brick', 'mug', 'rubber_duck', 'elephant', 'piggy_bank',
           'stanford_bunny2', 'teapot')


def timestamp():
    return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(timespec='seconds')


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def run_worker(args):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    rows, error, config = [], None, {}
    started = timestamp()
    try:
        from examples.mpc.fingertips.test.test_scm_mujoco import (
            build_parser, _run_rollout, _write_rows,
        )
        from planning.rollout_timing import trial_timing, TIMING_DEFINITIONS
        cfg = build_parser().parse_args([
            '--solver', args.solver, '--obj', args.worker_object,
            '--steps', str(args.steps), '--population-size', str(args.population_size),
            '--population-iterations', str(args.population_iterations),
            '--solver-seed', str(args.solver_seed),
            '--cem-elite-fraction', str(args.cem_elite_fraction),
        ])
        config = vars(cfg).copy()
        rows = _run_rollout(cfg, 'surrogate', args.worker_object,
                            seed=args.worker_seed, mass=.01, steps=args.steps)
    except Exception as exc:
        rows = getattr(exc, 'rows', rows)
        error = '%s: %s' % (type(exc).__name__, exc)
        traceback.print_exc()
    last = rows[-1] if rows else {}
    # Import/setup failures still produce an explicit result and let the
    # controller continue with the remaining scheduled trials.
    timing = dict(execution_time_s=None, elapsed_wall_time_s=None,
                  simulation_time_s=None, planning_time_ms=dict(mean=None, n=0))
    if rows:
        timing = trial_timing(rows)
    result = dict(solver=args.solver, object=args.worker_object, seed=args.worker_seed,
                  steps=len(rows), success=bool(last.get('success', False)),
                  settled=bool(last.get('settled', False)),
                  final_pos=last.get('goal_pos'), final_quat=last.get('goal_quat'),
                  solver_backends=sorted({r.get('solver_backend', 'unknown') for r in rows}),
                  fallback_rounds=sum(bool(r.get('solver_fallback')) for r in rows),
                  unstable_rounds=sum(bool(r.get('mujoco_unstable')) for r in rows),
                  error=error, started_at=started, completed_at=timestamp(), **timing)
    if not result['success'] and error is None:
        result['error'] = 'Goal/settling criteria not reached within the step limit'
    metadata = dict(trial_summary=result, args=config, mass=.01,
                    default_random_tilt=True, execution='wrench')
    if '_write_rows' in locals():
        metadata['timing_definitions'] = TIMING_DEFINITIONS
        _write_rows(args.worker_output, rows, metadata)
    else:
        write_json(args.worker_output, dict(metadata=metadata, rows=[]))
    print('TRIAL_RESULT ' + json.dumps(result, allow_nan=False), flush=True)
    return 0 if result['success'] else 1


def run_controller(args):
    objects = [x.strip() for x in args.objects.split(',') if x.strip()]
    seeds = [int(x) for x in args.seeds.split(',')]
    if len(objects) != len(set(objects)) or not objects or any(x not in OBJECTS for x in objects):
        raise ValueError('objects must be unique members of the seven fingertip objects')
    if len(seeds) != len(set(seeds)) or not seeds or min(seeds) < 0:
        raise ValueError('seeds must be unique nonnegative integers')
    if args.jobs < 1 or args.steps < 1:
        raise ValueError('jobs and steps must be positive')
    for obj in objects:
        for path in (ROOT/'envs/xmls'/('env_fingertips_'+obj+'.xml'),
                     ROOT/'envs/assets/objects'/(obj+'.stl')):
            if not path.is_file():
                raise FileNotFoundError(path)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logs = out/'trial_logs'
    logs.mkdir(exist_ok=True)
    tasks = [(obj, seed) for seed in seeds for obj in objects]
    protocol = dict(solver=args.solver, objects=objects, seeds=seeds,
                    total_trials=len(tasks), steps=args.steps, jobs=args.jobs,
                    mass=.01, execution='wrench', default_random_tilt=True,
                    population_size=args.population_size,
                    population_iterations=args.population_iterations,
                    solver_seed=args.solver_seed, cem_elite_fraction=args.cem_elite_fraction)
    write_json(out/'protocol.json', protocol)
    print('BATCH_START ' + json.dumps(dict(protocol, pid=os.getpid(), started_at=timestamp())), flush=True)
    completed = []

    def run_task(obj, seed):
        path = out/('trial_%s_%03d.json' % (obj, seed))
        log = logs/('%s_%03d.log' % (obj, seed))
        if path.exists():
            raise FileExistsError('Fresh batch result already exists: %s' % path)
        command = [sys.executable, '-u', str(Path(__file__).resolve()),
                   '--worker', '--solver', args.solver, '--worker-object', obj,
                   '--worker-seed', str(seed), '--worker-output', str(path),
                   '--steps', str(args.steps), '--population-size', str(args.population_size),
                   '--population-iterations', str(args.population_iterations),
                   '--solver-seed', str(args.solver_seed),
                   '--cem-elite-fraction', str(args.cem_elite_fraction)]
        print('TRIAL_START solver=%s object=%s seed=%d log=%s time=%s' %
              (args.solver, obj, seed, log, timestamp()), flush=True)
        with log.open('w') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            code = child.wait()
        if path.exists():
            data = json.loads(path.read_text())
            trial = data['metadata']['trial_summary']
            del data
        else:
            trial = dict(solver=args.solver, object=obj, seed=seed, success=False,
                         settled=False, steps=0, final_pos=None, final_quat=None,
                         error='Worker exited without a result JSON; see trial log',
                         completed_at=timestamp(), planning_time_ms=dict(mean=None, n=0),
                         execution_time_s=None, elapsed_wall_time_s=None,
                         simulation_time_s=None)
            write_json(path, dict(metadata=dict(trial_summary=trial, protocol=protocol), rows=[]))
        return dict(trial, worker_exit_code=code, result_file=str(path), log_file=str(log))

    def publish(status):
        report = dict(protocol=protocol, status=status, updated_at=timestamp(),
                      completed=len(completed), successes=sum(t['success'] for t in completed),
                      remaining=len(tasks)-len(completed), trials=completed)
        write_json(out/'summary.json', report)
        fields = ['solver', 'object', 'seed', 'success', 'settled', 'steps', 'final_pos',
                  'final_quat', 'execution_time_s', 'elapsed_wall_time_s', 'simulation_time_s',
                  'planning_mean_ms', 'error', 'worker_exit_code', 'result_file', 'log_file']
        tmp = out/'summary.csv.tmp'
        with tmp.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for trial in completed:
                row = {k: trial.get(k) for k in fields}
                row['planning_mean_ms'] = trial['planning_time_ms']['mean']
                writer.writerow(row)
        tmp.replace(out/'summary.csv')

    publish('running')
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_task, obj, seed): (obj, seed) for obj, seed in tasks}
        for future in as_completed(futures):
            obj, seed = futures[future]
            try:
                trial = future.result()
            except Exception as exc:
                trial = dict(solver=args.solver, object=obj, seed=seed, success=False,
                             settled=False, steps=0, final_pos=None, final_quat=None,
                             planning_time_ms=dict(mean=None, n=0),
                             error='%s: %s' % (type(exc).__name__, exc))
                traceback.print_exc()
            completed.append(trial)
            publish('running')
            print('RESULT ' + json.dumps(trial, allow_nan=False), flush=True)
            print('BATCH_PROGRESS solver=%s completed=%d/%d successes=%d' %
                  (args.solver, len(completed), len(tasks), sum(t['success'] for t in completed)), flush=True)
    publish('finished')
    print('BATCH_COMPLETE solver=%s completed=%d/%d successes=%d time=%s' %
          (args.solver, len(completed), len(tasks), sum(t['success'] for t in completed), timestamp()), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', choices=('acados', 'ipopt', 'cma-es', 'cem'), required=True)
    parser.add_argument('--objects', default=','.join(OBJECTS))
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--population-size', type=int, default=24)
    parser.add_argument('--population-iterations', type=int, default=12)
    parser.add_argument('--solver-seed', type=int, default=0)
    parser.add_argument('--cem-elite-fraction', type=float, default=.25)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker-object', choices=OBJECTS, help=argparse.SUPPRESS)
    parser.add_argument('--worker-seed', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.worker_object is None or args.worker_seed is None or args.worker_output is None:
            parser.error('Worker requires an object, seed and output path')
        return run_worker(args)
    if args.output_dir is None:
        parser.error('--output-dir is required')
    return run_controller(args)


if __name__ == '__main__':
    raise SystemExit(main())
