"""Run the fixed Table IV cohort through STOCS-3D and summarize its column.

The 15 cases are the outcome-stratified baseline sample already used for
ContactSDF: seven objects, 12 cached successes and 3 cached failures.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / 'output/benchmarks/contactsdf_official_single_tip_15_20261020/manifest.json'
OUT = ROOT / 'output/benchmarks/stocs3d_table4_15_20261020'
PYTHON = sys.executable


def _env():
    env = os.environ.copy()
    env.pop('SCSP_PLANNER_ONLY', None)
    env['PYTHONUNBUFFERED'] = '1'
    env['PATH'] = '/home/zz/miniconda3/envs/acc/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = '/home/zz/miniconda3/envs/acc/lib:' + env.get('LD_LIBRARY_PATH', '')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    return env


def _case_dir(case):
    return OUT / 'cases' / case['case_id']


def solve_case(case):
    folder = _case_dir(case)
    path = folder / ('trial_%03d.json' % case['seed'])
    if path.is_file():
        plan = json.loads(path.read_text())
        if 'planning_time_s' in plan and 'init_xyz' in plan:
            return dict(case_id=case['case_id'], planning_time_s=plan['planning_time_s'],
                        solved=plan['solved'], iterations=plan['iterations'], wall_s=0.0, reused=True)
    folder.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, '-u', str(ROOT / 'examples/mpc/franka/stocs3d/run.py'),
               '--stocs3d', '--stocs-solve-only', '--obj', case['object'], '--flipping',
               '--headless', '--trial-start', str(case['seed']), '--trial_num', '1',
               '--max_rollout_length', '2500', '--compute-threads', '1',
               '--rollout-summary-json', str(folder / 'rollout_summary.json')]
    log = folder / 'solve.log'
    started = time.time()
    with log.open('w') as stream:
        process = subprocess.run(command, cwd=ROOT, env=_env(), stdout=stream, stderr=subprocess.STDOUT)
    path = folder / ('trial_%03d.json' % case['seed'])
    if process.returncode != 0 or not path.is_file():
        raise RuntimeError('%s solve failed; see %s' % (case['case_id'], log))
    plan = json.loads(path.read_text())
    init_diff = max(abs(a - b) for a, b in zip(plan['init_xyz'] + plan['init_quat_wxyz'],
                                                case['init_xyz'] + case['init_quat_wxyz']))
    goal_diff = max(abs(a - b) for a, b in zip(plan['goal_xyz'] + plan['goal_quat_wxyz'],
                                                case['goal_xyz'] + case['goal_quat_wxyz']))
    if init_diff > 1e-6 or goal_diff > 1e-9:
        raise RuntimeError('%s pose does not match the cached baseline' % case['case_id'])
    return dict(case_id=case['case_id'], planning_time_s=plan['planning_time_s'],
                solved=plan['solved'], iterations=plan['iterations'], wall_s=time.time() - started)


def execute_case(case):
    folder = _case_dir(case)
    command = [PYTHON, '-u', str(ROOT / 'examples/mpc/franka/ik2/test_mpc_isaac.py'),
               '--stocs-trajectory', str(folder), '--obj', case['object'], '--flipping',
               '--headless', '--no-headless-realtime', '--sync-planner',
               '--physics-backend', 'isaac', '--num-envs', '1',
               '--trial-start', str(case['seed']), '--trial_num', '1',
               '--max_rollout_length', '2500', '--compute-threads', '1',
               '--rollout-summary-json', str(folder / 'execution_summary.json'),
               '--trial-output', str(folder / 'trials')]
    log = folder / 'execute.log'
    with log.open('w') as stream:
        process = subprocess.run(command, cwd=ROOT, env=_env(), stdout=stream, stderr=subprocess.STDOUT)
    summary_path = folder / 'execution_summary.json'
    if process.returncode != 0 or not summary_path.is_file():
        raise RuntimeError('%s execution failed; see %s' % (case['case_id'], log))
    summary = json.loads(summary_path.read_text())
    row = summary['trials'][0]
    if row['trial'] != case['seed'] or row['object'] != case['object'] or row['aborted']:
        raise RuntimeError('%s execution record does not match the cached case' % case['case_id'])
    return row


def _mean_std(values):
    if not values:
        return None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return dict(n=len(values), mean=mean, sample_std=std)


def summarize(cases, rows):
    by_id = {row['trial_key']: row for row in rows}
    ordered = [by_id[case['case_id']] for case in cases]
    successes = [row for row in ordered if row['success']]
    result = dict(
        task='flipping',
        trials=len(ordered),
        successes=len(successes),
        success_rate=len(successes) / len(ordered),
        execution_time_s=_mean_std([row['execution_time_s'] for row in successes]),
        planning_time_s=_mean_std([row['planning_time_s'] for row in ordered]),
        position_error_m=_mean_std([row['final_pos_err'] for row in ordered]),
        quaternion_error=_mean_std([row['final_quat_err'] for row in ordered]),
        definitions=dict(
            success_rate='Successes within 50 s (2500 rollout steps of 0.02 s).',
            execution_time_s='Simulated completion time, successful trials only.',
            planning_time_s='Offline STOCS contact selection and SNOPT CITO time, one sample per trial.',
            position_error_m='Terminal position error over all trials, including failures.',
            quaternion_error='Terminal quaternion error over all trials, including failures.',
            std='Sample standard deviation.',
        ),
        cases=ordered,
    )
    (OUT / 'table4_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    manifest = json.loads(MANIFEST.read_text())
    cases = manifest['cases']
    assert len(cases) == 15 and sum(case['baseline_success'] for case in cases) == 12
    assert len({case['object'] for case in cases}) == 7
    OUT.mkdir(parents=True, exist_ok=True)
    print('[stocs3d] solving 15 trajectories', flush=True)
    solved = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(solve_case, case): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                solved.append(future.result())
                print('[solved] %s' % solved[-1]['case_id'], flush=True)
            except Exception as error:
                print('[solve-failed] %s %s' % (case['case_id'], error), flush=True)
                solved.append(dict(case_id=case['case_id'], error=str(error)))
    print('[stocs3d] executing in Isaac', flush=True)
    rows = []
    runnable = [case for case in cases if (_case_dir(case) / ('trial_%03d.json' % case['seed'])).is_file()]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(execute_case, case): case for case in runnable}
        for future in as_completed(futures):
            case = futures[future]
            try:
                row = future.result()
            except Exception as error:
                print('[execute-failed] %s %s' % (case['case_id'], error), flush=True)
                continue
            row['trial_key'] = case['case_id']
            row['baseline_success'] = case['baseline_success']
            rows.append(row)
            print('[executed] %s success=%s' % (case['case_id'], row['success']), flush=True)
    if len(rows) != len(cases):
        raise SystemExit('Completed %d of %d executions' % (len(rows), len(cases)))
    result = summarize(cases, rows)
    print(json.dumps({key: result[key] for key in (
        'success_rate', 'execution_time_s', 'planning_time_s', 'position_error_m', 'quaternion_error')},
        indent=2), flush=True)


if __name__ == '__main__':
    main()
