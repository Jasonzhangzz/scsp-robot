"""Run the prespecified 7-object x 3-seed A7 cohort for both paper tasks.

Outputs retain every trial, commands, source hashes, logs and exact statistics.
Execution time is success-only simulated time; planning time pools every cycle;
final pose errors include every completed trial, including failures.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
OBJECTS = ('foam_brick', 'stanford_bunny2', 'elephant', 'piggy_bank',
           'teapot', 'mug', 'rubber_duck')
SEEDS = (0, 1, 2)
TASKS = ('rotation', 'flipping')


def metric(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values) > 1 else None)


def summarize(rows):
    successes = [row for row in rows if row['success']]
    cycles = sum(row['planning_wall_count'] for row in rows)
    return dict(trials=len(rows), successes=len(successes), success_rate=len(successes)/len(rows),
        execution_time_s=metric([row['completion_time_s'] if row['completion_time_s'] is not None
            else row['completion_steps']*row['policy_interval_s'] for row in successes]),
        planning_time_s=sum(row['planning_wall_total_s'] for row in rows)/cycles,
        planning_cycles=cycles, final_pos_error_m=metric([row['final_pos_err'] for row in rows]),
        final_quat_error=metric([row['final_quat_err'] for row in rows]),
        mppi_solves=sum(row['a7_mppi']['solves'] for row in rows),
        successful_cases=[dict(object=row['object'], seed=row['seed']) for row in successes])


def load_cell(path, obj, task):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert sorted(row['seed'] for row in rows) == list(SEEDS), path
    for row in rows:
        assert row['object'] == obj and not row['aborted'], path
        assert row['ablation_mode'] in ('mppi', 'a7'), path
        assert row['solver_backend'] == row['planner_solver_backend'] == 'mppi', path
        assert row['target_type'] == ('rotation' if task == 'rotation' else 'ground-rotation'), path
        assert row['steps'] == row['planning_wall_count'] == row['a7_mppi']['solves'], path
        assert 0 <= row['steps'] <= 2500, path
        assert row['lambda_solves'] == row['mpc_solves'] == row['mpc_failures'] == 0, path
        assert not row['async_planner'] and row['physics_backend'] == 'isaac', path
        row['task'] = task
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=7)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # One representative CPU from each physical core prevents sibling contention.
    available = os.sched_getaffinity(0)
    cpus, seen = [], set()
    for cpu in sorted(available):
        topology = Path('/sys/devices/system/cpu/cpu%d/topology' % cpu)
        key = ((topology/'physical_package_id').read_text().strip(),
               (topology/'core_id').read_text().strip())
        if key not in seen:
            seen.add(key)
            cpus.append(cpu)
    jobs = min(args.jobs, len(OBJECTS), len(cpus))
    if jobs < 1:
        raise ValueError('--jobs must be positive')
    commands = {}
    env = os.environ.copy()
    env['PATH'] = str(Path(sys.executable).parent)+os.pathsep+env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = str(Path(sys.executable).parents[1]/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH', '')
    for task in TASKS:
        (out/task).mkdir()
        for obj in OBJECTS:
            cell = out/task/obj
            cell.mkdir()
            commands[task+'/'+obj] = [sys.executable, '-u', str(ROOT/'examples/mpc/franka/experiments/ik2/run_ablation.py'),
                '--a7', '--task', task, '--obj', obj, '--headless', '--sync-planner',
                '--no-headless-realtime', '--physics-backend', 'isaac', '--solver-backend', 'ipopt',
                '--trial-start', '0', '--trial_num', '3', '--max_rollout_length', '2500',
                '--sample_num', '70', '--compute-threads', '1',
                '--a7-mppi-samples', '128', '--a7-mppi-horizon', '10', '--a7-mppi-iterations', '1',
                '--a7-mppi-temperature', '.005', '--a7-mppi-noise', '.003',
                '--ranking-metrics-jsonl', str(cell/'trials.jsonl'),
                '--rollout-summary-json', str(cell/'rollout_summary.json')]
            (cell/'command.json').write_text(json.dumps(commands[task+'/'+obj], indent=2)+'\n')
    source_files = [Path(__file__).resolve().relative_to(ROOT),
        Path('planning/franka_mppi_ablation.py'), Path('planning/mpc_explicit_ablation.py'),
        Path('planning/object_pose_cost.py'), Path('models/cartesian_rigid_contact.py'),
        Path('models/contact_geometry.py'), Path('models/soft_scm_contact.py')]
    source_files += [Path('examples/mpc/franka/experiments/ik2')/name for name in (
        'run_ablation.py', 'params_ablation.py', 'test_mpc_isaac_ablation.py',
        'run_isaac_ablation.py', 'contact_policy_ablation.py')]
    source_files += [Path('examples/mpc/franka/runtime')/name for name in ('contact.py', 'simulator.py')]
    hashes = {}
    for relative in source_files:
        data = (ROOT/relative).read_bytes()
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
        destination = out/'source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    manifest = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
        objects=OBJECTS, seeds=SEEDS, tasks=TASKS, expected_trials=42, max_rollout_length=2500,
        selection='Fixed seed IDs 0,1,2 per object and task, selected before observing outcomes',
        scheduling='Synchronous 20 ms policy intervals; up to %d object workers on distinct physical cores' % jobs,
        worker_cpus=cpus[:jobs], commands=commands, source_sha256=hashes,
        definitions=dict(execution_time='Successful trials: simulated seconds to task success',
            planning_time='Sum of all planning callback seconds / sum of all control cycles',
            errors='Mean final errors over all completed trials, including failures'))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    started = time.perf_counter()

    def worker(index, obj):
        rows = []
        cpu = cpus[index % jobs]
        for task in TASKS:
            cell = out/task/obj
            with (cell/'run.log').open('w') as log:
                result = subprocess.run(['taskset', '-c', str(cpu), *commands[task+'/'+obj]],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('%s/%s failed: see %s' % (task, obj, cell/'run.log'))
            records = load_cell(cell/'trials.jsonl', obj, task)
            rows.extend(records)
            print('[a7] %s/%s: %d/3 successes, elapsed %.1fs' % (
                task, obj, sum(row['success'] for row in records), time.perf_counter()-started), flush=True)
        return rows

    all_rows = []
    try:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(worker, index, obj) for index, obj in enumerate(OBJECTS)]
            for future in as_completed(futures):
                all_rows.extend(future.result())
        for path, digest in hashes.items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
        statistics_by_task = {}
        for task in TASKS:
            rows = [row for row in all_rows if row['task'] == task]
            assert len(rows) == 21
            statistics_by_task[task] = dict(overall=summarize(rows), objects={
                obj: summarize([row for row in rows if row['object'] == obj]) for obj in OBJECTS})
        (out/'statistics.json').write_text(json.dumps(statistics_by_task, indent=2, allow_nan=False)+'\n')
        (out/'trials.json').write_text(json.dumps(sorted(all_rows,
            key=lambda row: (row['task'], row['object'], row['seed'])), indent=2, allow_nan=False)+'\n')
        with (out/'trials.csv').open('w') as stream:
            columns = ['task', 'object', 'seed', 'success', 'steps', 'completion_time_s',
                'planning_wall_total_s', 'planning_wall_count', 'planning_wall_mean_ms',
                'final_pos_err', 'final_quat_err', 'solver_backend']
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(sorted(all_rows, key=lambda row: (row['task'], row['object'], row['seed'])))
        manifest.update(status='complete', completed_trials=len(all_rows), wall_time_s=time.perf_counter()-started)
        print(json.dumps({task: result['overall'] for task, result in statistics_by_task.items()}, indent=2), flush=True)
    except BaseException as error:
        manifest.update(status='failed', error=str(error))
        raise
    finally:
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
