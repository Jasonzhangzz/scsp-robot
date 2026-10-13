"""Run 35 A3 rotation trials sequentially and persist all requested metrics."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/zz/scsp-robot')
OUTPUT = Path(__file__).resolve().parent
RESULTS = OUTPUT / 'results'
OBJECTS = ['foam_brick', 'stanford_bunny2', 'elephant', 'piggy_bank',
           'teapot', 'mug', 'rubber_duck']
SEEDS = list(range(5))
PYTHON = '/home/zz/miniconda3/envs/acc/bin/python'


def save_json(path, value):
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def moments(values):
    values = [float(v) for v in values if v is not None]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values) > 1 else None)


def aggregate(rows, expected):
    good = [r for r in rows if r['success'] and not r['aborted']]
    valid = len(rows) == expected and not any(r['aborted'] for r in rows)
    cycles = sum(r['planning_wall_count'] for r in rows)
    seconds = sum(r['planning_wall_total_s'] for r in rows)
    squares = sum(r['planning_wall_sum_squares_s2'] for r in rows)
    variance = max(0., (squares - seconds * seconds / cycles) / (cycles - 1)) if cycles > 1 else None
    iterations = sum(r['iteration_wall_count'] for r in rows)
    return dict(expected_trials=expected, completed_trials=len(rows), complete=valid,
        successes=len(good), success_rate=len(good) / expected if valid else None,
        completion_sim_s=moments([r['completion_time_s'] for r in good]),
        completion_wall_s=moments([r['completion_wall_time_s'] for r in good]),
        execution_wall_all_trials_s=moments([r['trial_wall_time_s'] for r in rows]),
        planning_cycle_ms=dict(n=cycles, mean=seconds * 1000 / cycles if cycles else None,
            std=variance ** .5 * 1000 if variance is not None else None),
        iteration_wall_mean_ms=(sum(r['iteration_wall_total_s'] for r in rows) * 1000 / iterations
                                if iterations else None),
        solve_cycle_mean_ms=(sum(r['solve_dt_mean_s'] * r['steps'] for r in rows) * 1000
                             / sum(r['steps'] for r in rows) if sum(r['steps'] for r in rows) else None),
        final_pos_err_m=moments([r['final_pos_err'] for r in rows]),
        final_quat_err=moments([r['final_quat_err'] for r in rows]),
        mpc_failures=sum(r['mpc_failures'] for r in rows),
        lambda_failures=sum(r['lambda_failures'] for r in rows),
        mpc_fallbacks=sum(r['mpc_fallbacks'] for r in rows),
        lambda_fallbacks=sum(r['lambda_fallbacks'] for r in rows))


def main():
    RESULTS.mkdir(exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MAX_JOBS='2')
    env.pop('SCSP_PLANNER_ONLY', None)
    env['PATH'] = str(Path(PYTHON).parent) + os.pathsep + env.get('PATH', '')
    env['ACADOS_SOURCE_DIR'] = '/home/zz/zz_ws/acados'
    env['LD_LIBRARY_PATH'] = '/home/zz/miniconda3/envs/acc/lib:/home/zz/zz_ws/acados/lib:' + env.get('LD_LIBRARY_PATH', '')
    commands = {}
    for obj in OBJECTS:
        command = [PYTHON, '-u', str(ROOT / 'examples/mpc/franka/ik2/run_ablation.py'),
            '--a3', '--rotation', '--obj', obj, '--headless', '--async-planner',
            '--headless-realtime', '--num-envs', '1', '--solver-backend', 'acados',
            '--trial-start', '0', '--trial_num', '5', '--max_rollout_length', '5000',
            '--sample_num', '70', '--compute-threads', '1',
            '--ranking-metrics-jsonl', str(RESULTS / (obj + '.jsonl')),
            '--rollout-summary-json', str(RESULTS / (obj + '_rollout_summary.json'))]
        if obj in ('mug', 'rubber_duck'):
            command += ['--collision-manifest', str(OUTPUT / (obj + '_collision_manifest.json'))]
        commands[obj] = command
        save_json(RESULTS / (obj + '_command.json'), command)
    files = ['examples/mpc/franka/ik2/run_ablation.py',
             'examples/mpc/franka/ik2/params_ablation.py',
             'examples/mpc/franka/ik2/test_mpc_isaac_ablation.py',
             'examples/mpc/franka/ik2/contact_policy_ablation.py',
             'planning/mpc_explicit_ablation.py', 'utils/metrics.py']
    manifest = dict(status='running', pid=os.getpid(),
        started_utc=datetime.now(timezone.utc).isoformat(),
        python=PYTHON, objects=OBJECTS, seeds=SEEDS, expected_trials=35,
        max_rollout_length=5000, physics='Isaac Gym CPU PhysX',
        scheduling='async planner, realtime physics, one object process at a time',
        commands=commands,
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        metric_definitions=dict(
            success='position <= 0.02 m, quaternion error <= 0.015, pose stage',
            quaternion_error='1 - dot(q, target_q)**2 (dimensionless)',
            completion='successful trials only; simulator and wall time reported separately',
            execution_wall='all trials; excludes initialization and final reporting',
            planning='CSO + approach + MPC callback wall time; excludes physics/IPC/rendering; iteration weighted',
            final_errors='all completed trials, including failures'))
    save_json(RESULTS / 'manifest.json', manifest)
    (RESULTS / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff', '--', *files], cwd=ROOT))
    summaries, all_rows, failures = {}, [], {}
    started = time.perf_counter()
    print('A3 rotation benchmark: 7 objects x 5 seeds; 5000-step limit.', flush=True)
    for obj, command in commands.items():
        print('\nStarting ' + obj + ': ' + shlex.join(command), flush=True)
        wall_start = time.perf_counter()
        try:
            with (RESULTS / (obj + '.log')).open('w') as log:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError('Child exited with code %d; see %s.log' % (completed.returncode, obj))
            rows = [json.loads(line) for line in (RESULTS / (obj + '.jsonl')).read_text().splitlines() if line.strip()]
            if sorted(r['seed'] for r in rows) != SEEDS or any(r['aborted'] for r in rows):
                raise RuntimeError('Missing, duplicate or aborted trial records')
            if any(r['target_type'] != 'rotation' or r['ablation_mode'] != 'a3' for r in rows):
                raise RuntimeError('Unexpected task or ablation mode')
            if any(r['planning_wall_count'] != r['steps'] for r in rows):
                raise RuntimeError('Planning cycle count does not match executed steps')
            summary = aggregate(rows, len(SEEDS))
            summary['successful_seeds'] = [r['seed'] for r in rows if r['success']]
            summary['process_wall_s'] = time.perf_counter() - wall_start
            summaries[obj] = summary
            all_rows.extend(rows)
            print(obj + ' metrics: ' + json.dumps(summary, allow_nan=False), flush=True)
        except Exception:
            failures[obj] = traceback.format_exc()
            print(failures[obj], flush=True)
        save_json(RESULTS / 'progress.json', dict(objects=summaries, failures=failures))
    overall = aggregate(all_rows, len(OBJECTS) * len(SEEDS))
    result = dict(overall=overall, objects=summaries, errors=failures,
                  total_process_wall_s=time.perf_counter() - started,
                  metric_definitions=manifest['metric_definitions'])
    save_json(RESULTS / 'summary.json', result)
    if all_rows:
        fields = list(dict.fromkeys(k for row in all_rows for k in row))
        with (RESULTS / 'trials.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
    manifest.update(status='complete' if overall['complete'] and not failures else 'failed',
                    finished_utc=datetime.now(timezone.utc).isoformat())
    save_json(RESULTS / 'manifest.json', manifest)
    print('\nFINAL RESULTS:\n' + json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0 if manifest['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
