"""STOCS-3D rotation cohort: seven objects, seeds 0-4, then Isaac execution."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'output/benchmarks/stocs3d_rotation_7x5_20261021'
PYTHON = '/home/zz/miniconda3/envs/acc/bin/python'
OBJECTS = ('foam_brick', 'stanford_bunny2', 'elephant', 'piggy_bank', 'teapot', 'mug', 'rubber_duck')
SEEDS = (0, 1, 2, 3, 4)


def _env():
    env = os.environ.copy()
    env.pop('SCSP_PLANNER_ONLY', None)
    env['PYTHONUNBUFFERED'] = '1'
    env['PATH'] = '/home/zz/miniconda3/envs/acc/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = '/home/zz/miniconda3/envs/acc/lib:' + env.get('LD_LIBRARY_PATH', '')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    return env


def cases():
    rows = []
    for obj in OBJECTS:
        for seed in SEEDS:
            rows.append(dict(case_id='%s_seed%d' % (obj, seed), object=obj, seed=seed))
    return rows


def _folder(case):
    return OUT / 'cases' / case['case_id']


def solve_case(case):
    folder = _folder(case)
    path = folder / ('trial_%03d.json' % case['seed'])
    if path.is_file() and 'planning_time_s' in json.loads(path.read_text()):
        return dict(case_id=case['case_id'], status='reused')
    folder.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, '-u', str(ROOT / 'examples/mpc/franka/stocs3d/run.py'),
               '--stocs3d', '--stocs-solve-only', '--obj', case['object'], '--rotation',
               '--headless', '--trial-start', str(case['seed']), '--trial_num', '1',
               '--max_rollout_length', '2500', '--compute-threads', '1',
               '--rollout-summary-json', str(folder / 'rollout_summary.json')]
    log = folder / 'solve.log'
    with log.open('w') as stream:
        process = subprocess.run(command, cwd=ROOT, env=_env(), stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode != 0 or not path.is_file():
        raise RuntimeError('%s solve failed; see %s' % (case['case_id'], log))
    return dict(case_id=case['case_id'], status='solved')


def execute_case(case):
    folder = _folder(case)
    summary_path = folder / 'execution_summary.json'
    if summary_path.is_file():
        row = json.loads(summary_path.read_text())['trials'][0]
        if not row.get('aborted'):
            return row
    command = [PYTHON, '-u', str(ROOT / 'examples/mpc/franka/ik2/test_mpc_isaac.py'),
               '--stocs-trajectory', str(folder), '--obj', case['object'], '--rotation',
               '--headless', '--no-headless-realtime', '--sync-planner',
               '--physics-backend', 'isaac', '--num-envs', '1',
               '--trial-start', str(case['seed']), '--trial_num', '1',
               '--max_rollout_length', '2500', '--compute-threads', '1',
               '--rollout-summary-json', str(summary_path),
               '--trial-output', str(folder / 'trials')]
    log = folder / 'execute.log'
    with log.open('w') as stream:
        process = subprocess.run(command, cwd=ROOT, env=_env(), stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode != 0 or not summary_path.is_file():
        raise RuntimeError('%s execution failed; see %s' % (case['case_id'], log))
    return json.loads(summary_path.read_text())['trials'][0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = cases()
    (OUT / 'manifest.json').write_text(json.dumps(dict(
        task='rotation', objects=OBJECTS, seeds=SEEDS, trials=len(jobs),
        max_rollout_length=2500), indent=2) + '\n')
    print('[stocs3d] solving %d rotation trajectories' % len(jobs), flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(solve_case, case): case for case in jobs}
        for future in as_completed(futures):
            case = futures[future]
            try:
                print('[solved]', case['case_id'], future.result()['status'], flush=True)
            except Exception as error:
                print('[solve-failed]', case['case_id'], error, flush=True)
    ready = [case for case in jobs if (_folder(case) / ('trial_%03d.json' % case['seed'])).is_file()]
    print('[stocs3d] executing %d trajectories' % len(ready), flush=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(execute_case, case): case for case in ready}
        for future in as_completed(futures):
            case = futures[future]
            try:
                row = future.result()
                print('[executed]', case['case_id'], 'success', row['success'], flush=True)
            except Exception as error:
                print('[execute-failed]', case['case_id'], error, flush=True)


if __name__ == '__main__':
    main()
