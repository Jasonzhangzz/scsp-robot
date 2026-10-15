"""Prespecified A4/A5 evaluation: seven objects, three seeds, two tasks.

Uses real Isaac execution, retains unsuccessful trials, and records all source
and command provenance. --resume skips only complete, validated trial cells.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from examples.mpc.franka.ik2.evaluate_a7 import OBJECTS, SEEDS, TASKS, metric

VARIANTS = ('a4', 'a5')


def load_cell(file, variant, task, obj):
    rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
    if sorted(row['seed'] for row in rows) != list(SEEDS):
        raise ValueError('Missing or duplicated trial seeds: '+str(file))
    for row in rows:
        assert row['object'] == obj and not row['aborted'], file
        assert row['ablation_mode'] == variant and row['cpo_variant'] == 'full', file
        assert row['physics_backend'] == 'isaac' and row['async_planner'], file
        assert row['headless_realtime'], file
        assert row['target_type'] == ('rotation' if task == 'rotation' else 'ground-rotation'), file
        assert row['steps'] == row['planning_wall_count'] and row['steps'] <= 2500, file
        assert row['a7_mppi'] is None and row['planner_solver_backend'] == 'acados', file
        if row['success']:
            assert row['completion_time_s'] is not None, file
        backend = row['cso_backend']
        assert backend['fallback_to_scm'] == 0 and backend['solves'] > 0, file
        if variant == 'a4':
            assert backend['model'] == 'mujoco-native-pyramidal-qp' and backend['solver'] == 'slsqp', file
            assert backend['observations'] == row['steps'] and backend['maxiter'] == 12, file
        else:
            assert backend['model'] == 'soft-scm' and backend['solver'] == 'cma-es', file
            assert backend['population_size'] == 24 and backend['generations'] == 12, file
        row.update(task=task, variant=variant)
    return rows


def summarize(rows):
    success = [row for row in rows if row['success']]
    cycles = sum(row['planning_wall_count'] for row in rows)
    return dict(trials=len(rows), successes=len(success), success_rate=len(success)/len(rows),
        execution_time_s=metric([row['completion_time_s'] for row in success]),
        async_planner=all(row['async_planner'] for row in rows),
        headless_realtime=all(row['headless_realtime'] for row in rows),
        physics_policy_ticks=sum(row['physics_policy_ticks'] for row in rows),
        planning_time_s=sum(row['planning_wall_total_s'] for row in rows)/cycles,
        planning_cycles=cycles, final_pos_error_m=metric([row['final_pos_err'] for row in rows]),
        final_quat_error=metric([row['final_quat_err'] for row in rows]),
        stable_flip_successes=sum(row['stable_flip_success'] for row in rows),
        cso_solves=sum(row['cso_backend']['solves'] for row in rows),
        cso_evaluations=sum(row['cso_backend']['evaluations'] for row in rows),
        cso_failed_or_nonconverged=sum(row['cso_backend'].get('nonconverged_local_solves',
            row['cso_backend'].get('failures', 0)) for row in rows),
        mpc_failures=sum(row['mpc_failures'] for row in rows),
        mpc_fallbacks=sum(row['mpc_fallbacks'] for row in rows),
        successful_cases=[dict(object=row['object'], seed=row['seed']) for row in success])


def collect(out):
    results, all_rows = {}, []
    for variant in VARIANTS:
        results[variant] = {}
        for task in TASKS:
            rows, objects = [], {}
            for obj in OBJECTS:
                cell = load_cell(out/variant/task/obj/'trials.jsonl', variant, task, obj)
                rows.extend(cell)
                objects[obj] = summarize(cell)
            assert len(rows) == 21
            all_rows.extend(rows)
            results[variant][task] = dict(overall=summarize(rows), objects=objects)
    assert len({(r['variant'], r['task'], r['object'], r['seed']) for r in all_rows}) == 84
    (out/'statistics.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    (out/'trials.json').write_text(json.dumps(sorted(all_rows,
        key=lambda r: (r['variant'], r['task'], r['object'], r['seed'])), indent=2, allow_nan=False)+'\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--jobs', default=8, type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    cpus, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        path = Path('/sys/devices/system/cpu/cpu%d/topology' % cpu)
        key = ((path/'physical_package_id').read_text(), (path/'core_id').read_text())
        if key not in seen:
            seen.add(key)
            cpus.append(cpu)
    # Give the physics and planner processes different physical cores to run
    # on concurrently; a single pinned core would slow down realtime physics.
    jobs = min(args.jobs, len(cpus)//2)
    if jobs < 1:
        raise ValueError('--jobs must be positive')
    files = [Path(__file__).resolve().relative_to(ROOT)]
    files += [Path('examples/mpc/franka/ik2')/name for name in (
        'params_ablation.py', 'contact_policy_ablation.py', 'run_ablation.py',
        'test_mpc_isaac_ablation.py', 'test_mppi_isaac.py', 'run_isaac_ablation.py',
        'headless_rollout_ablation.py', 'scene_friction.py', 'contact_frames.py')]
    files += [Path('planning')/name for name in ('franka_cso_ablation.py',
        'point_contact_baselines.py', 'contact_force_sampling.py', 'mlqp_point.py',
        'mpc_explicit_ablation.py', 'object_pose_cost.py')]
    files += [Path('models')/name for name in ('soft_scm_contact.py', 'cartesian_rigid_contact.py', 'contact_geometry.py')]
    hashes = {str(file):hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in files}
    manifest_path = out/'manifest.json'
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        assert manifest['source_sha256'] == hashes, 'Source changed; use a new cohort directory'
    else:
        for file in files:
            destination = out/'source'/file
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT/file).read_bytes())
        manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), objects=OBJECTS,
            source_root=str(ROOT),
            seeds=SEEDS, tasks=TASKS, variants=VARIANTS, expected_trials=84, max_rollout_length=2500,
            selection='Seed IDs 0,1,2 per object and task, fixed before observing outcomes',
            source_sha256=hashes, worker_cpus=[cpus[2*i:2*i+2] for i in range(jobs)], jobs=jobs,
            physics='Isaac Gym CPU PhysX', scheduling='Asynchronous planner with realtime 20 ms physics policy intervals; each worker uses two exclusive physical cores',
            definitions=dict(execution='Mean simulated seconds on successful trials',
                planning='All planning callback seconds divided by all control cycles',
                error='Mean final error over all trials, including failures'),
            packages={name:version(name) for name in ('numpy','casadi','mujoco','cma','coacd','torch')})
    manifest.update(status='running')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    env = os.environ.copy()
    env['PATH'] = str(Path(sys.executable).parent)+os.pathsep+env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = str(Path(sys.executable).parents[1]/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH', '')
    slots = queue.Queue()
    for i in range(jobs):
        slots.put(','.join(map(str, cpus[2*i:2*i+2])))
    started = time.perf_counter()
    def worker(variant, task, obj):
        cell = out/variant/task/obj
        cell.mkdir(parents=True, exist_ok=True)
        trials = cell/'trials.jsonl'
        if args.resume and trials.exists():
            try:
                records = load_cell(trials, variant, task, obj)
                print('[%s] retained %s/%s: %d/3 successes' % (variant, task, obj, sum(r['success'] for r in records)), flush=True)
                return
            except (AssertionError, ValueError):
                raise RuntimeError('Partial or invalid trial cell must be audited before a retry: '+str(cell))
        command = [sys.executable, '-u', str(Path(__file__).resolve().with_name('run_ablation.py')),
            '--'+variant, '--task', task, '--obj', obj, '--headless', '--async-planner',
            '--headless-realtime', '--physics-backend', 'isaac', '--solver-backend', 'acados',
            '--trial-start', '0', '--trial_num', '3', '--max_rollout_length', '2500',
            '--sample_num', '70', '--compute-threads', '1', '--a4-qp-maxiter', '12',
            '--a5-population-size', '24', '--a5-population-iterations', '12',
            '--ranking-metrics-jsonl', str(trials), '--rollout-summary-json', str(cell/'rollout_summary.json')]
        (cell/'command.json').write_text(json.dumps(command, indent=2)+'\n')
        cpu = slots.get()
        try:
            with (cell/'run.log').open('w') as log:
                result = subprocess.run(['taskset', '-c', str(cpu), *command],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('Trial cell failed: '+str(cell/'run.log'))
            records = load_cell(trials, variant, task, obj)
            print('[%s] %s/%s: %d/3 successes; %.1fs elapsed' % (
                variant, task, obj, sum(r['success'] for r in records), time.perf_counter()-started), flush=True)
        finally:
            slots.put(cpu)
    try:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(worker, variant, task, obj)
                for task in TASKS for obj in OBJECTS for variant in VARIANTS]
            for future in as_completed(futures):
                future.result()
        for file, digest in hashes.items():
            assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == digest, file
        results = collect(out)
        manifest.update(status='complete', completed_trials=84,
            completed_utc=datetime.now(timezone.utc).isoformat(), wall_time_s=time.perf_counter()-started)
        print(json.dumps({v:{t:results[v][t]['overall'] for t in TASKS} for v in VARIANTS}, indent=2), flush=True)
    except BaseException as error:
        manifest.update(status='failed', error=str(error))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
