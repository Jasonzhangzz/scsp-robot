"""Parallel, reproducible Isaac MPC trials and iteration-weighted timing reports.

Run with the same Python environment as test_mpc_isaac.py. This runner changes
no controller settings during a benchmark. Every object/seed receives the same
iteration budget; completion means the entry point's full pose success flag.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import platform
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[5]
ENTRY = ROOT / 'examples/mpc/franka/ik2/test_mpc_isaac.py'


def supported_objects():
    tree = ast.parse((ENTRY.parent / 'params.py').read_text())
    function = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef)
                    and n.name == '_source_standing_quat_wxyz')
    for node in ast.walk(function):
        if isinstance(node, ast.Compare) and any(isinstance(op, ast.NotIn) for op in node.ops):
            names = ast.literal_eval(node.comparators[0])
            if isinstance(names, tuple) and all(isinstance(n, str) for n in names):
                return list(names)
    raise RuntimeError('Could not discover the audited object list')


def read_rows(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return rows


def mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def aggregate(rows, expected):
    good = [r for r in rows if r['success'] and not r['aborted']]
    iterations = sum(r['iteration_wall_count'] for r in rows)
    iteration_seconds = sum(r['iteration_wall_total_s'] for r in rows)
    return dict(
        expected_trials=expected, completed_trials=len(rows),
        successes=len(good), success_rate=len(good) / expected if len(rows) == expected else None,
        observed_success_rate=len(good) / len(rows) if rows else None,
        stable_flip_successes=sum(bool(r['stable_flip_success']) for r in rows),
        aborted_trials=sum(bool(r['aborted']) for r in rows),
        successful_seeds=[r['seed'] for r in good],
        completion_sim_time_mean_s=mean([r['completion_time_s'] for r in good]),
        completion_wall_time_mean_s=mean([r['completion_wall_time_s'] for r in good]),
        iteration_count=iterations,
        iteration_wall_total_s=iteration_seconds,
        iteration_wall_mean_s=iteration_seconds / iterations if iterations else None,
        solve_wall_mean_s=(sum(r['solve_dt_mean_s'] * r['iteration_wall_count'] for r in rows)
                           / iterations if iterations else None),
        mpc_failures=sum(r['mpc_failures'] for r in rows),
        lambda_failures=sum(r['lambda_failures'] for r in rows),
    )


def save_reports(output, objects, seeds, commands, process_times, workers=1):
    rows = [row for obj in objects for row in read_rows(output / (obj + '.jsonl'))]
    per_object = {obj: aggregate([r for r in rows if r['object'] == obj], len(seeds))
                  for obj in objects}
    total = aggregate(rows, len(objects) * len(seeds))
    total.pop('successful_seeds')
    result = dict(overall=total, objects=per_object, process_wall_time_s=process_times,
                  success_definition='position error <= .02 m, quaternion error <= .015, pose stage',
                  iteration_timing='observation processing through policy execution and next observation; excludes initialization',
                  successful_trials=[dict(object=r['object'], seed=r['seed'],
                                           numpy_initialization_seed=100+r['seed'],
                                           completion_sim_time_s=r['completion_time_s'],
                                           completion_wall_time_s=r['completion_wall_time_s'])
                                     for r in rows if r['success'] and not r['aborted']])
    result.update(backend='isaac-physx', parallel_simulations=workers,
                  completed=len(rows) == len(objects) * len(seeds) and not total['aborted_trials'])
    write_json(output / 'summary.json', result)
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (output / 'trials.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with (output / 'objects.csv').open('w', newline='') as handle:
        fields = ['object'] + list(next(iter(per_object.values())))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(object=obj, **values) for obj, values in per_object.items())
    replay = ['#!/usr/bin/env bash', 'set -euo pipefail',
              'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=2',
              'export PATH=' + shlex.quote(str(Path(sys.executable).parent)) + ':"$PATH"',
              'cd ' + shlex.quote(str(ROOT))]
    for row in rows:
        if row['success'] and not row['aborted']:
            command = commands[row['object']][:]
            command[command.index('--trial-start') + 1] = str(row['seed'])
            command[command.index('--trial_num') + 1] = '1'
            command[command.index('--ranking-metrics-jsonl') + 1] = str(
                output / ('replay_%s_seed%d.jsonl' % (row['object'], row['seed'])))
            replay.extend(['# %s seed %d' % (row['object'], row['seed']), shlex.join(command)])
    path = output / 'reproduce_successes.sh'
    path.write_text('\n'.join(replay) + '\n')
    path.chmod(0o755)
    seconds = lambda value: 'pending' if value is None else '%.6f' % value
    report = ['backend: isaac-physx; parallel workers: %d' % workers,
              'completed: %d/%d; successes: %d; total success rate: %s' % (
                  total['completed_trials'], total['expected_trials'], total['successes'],
                  'pending' if total['success_rate'] is None else '%.2f%%' % (100 * total['success_rate'])),
              'Completion times average successful trials; iteration times include all finished trials.',
              'Iteration wall time excludes initialization/compilation and includes concurrent resource contention.',
              'object  completed  successes  completion_sim_s  completion_wall_s  iteration_wall_s  successful_seeds']
    for obj, values in per_object.items():
        report.append('%s  %d/%d  %d/%d  %s  %s  %s  %s' % (
            obj, values['completed_trials'], len(seeds), values['successes'], len(seeds),
            seconds(values['completion_sim_time_mean_s']), seconds(values['completion_wall_time_mean_s']),
            seconds(values['iteration_wall_mean_s']), values['successful_seeds']))
    report.append('ALL  iteration_wall_mean_s: %s; completion_sim_mean_s: %s; completion_wall_mean_s: %s' % (
        seconds(total['iteration_wall_mean_s']), seconds(total['completion_sim_time_mean_s']),
        seconds(total['completion_wall_time_mean_s'])))
    (output / 'summary.txt').write_text('\n'.join(report) + '\n')
    return result


def snapshot(output, commands, workers, seeds, max_iterations):
    git = lambda *args: subprocess.check_output(['git', *args], cwd=str(ROOT))
    (output / 'workspace.patch').write_bytes(git('diff', '--binary'))
    files = git('ls-files', '-co', '--exclude-standard', '--', '*.py', '*.xml', '*.urdf', '*.yaml').decode().splitlines()
    files = sorted(set(f for f in files if not f.startswith('outputs/')))
    hashes = {}
    with tarfile.open(str(output / 'source_snapshot.tar.gz'), 'w:gz') as archive:
        for name in files:
            path = ROOT / name
            if path.is_file():
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                archive.add(str(path), arcname=name)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), python=sys.executable,
                    python_version=sys.version, git_head=git('rev-parse', 'HEAD').decode().strip(),
                    git_status=git('status', '--short').decode(), commands=commands,
                    source_sha256=hashes, parallel_simulations=workers, backend='isaac-physx', runner_pid=os.getpid(),
                    trial_seeds=seeds, max_iterations=max_iterations,
                    cpu_threads_per_worker=1, status='running',
                    trial_seed_convention='trial-start is the trial seed; initial NumPy seed is 100 + trial seed',
                    completed=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'packages.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
    hardware = dict(platform=platform.platform(), cpu_count=os.cpu_count())
    try:
        hardware['gpu'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        hardware['gpu'] = None
    write_json(output / 'hardware.json', hardware)
    return manifest


def write_json(path, value):
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--trial-start', type=int, default=0)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--max-iterations', type=int, default=5000)
    parser.add_argument('--workers', type=int, default=1,
                        help='Independent object workers, sharing the GPU; wall times include contention')
    parser.add_argument('--objects', nargs='+', choices=supported_objects(),
                        help='Default: every supported object')
    parser.add_argument('--collision-geometry', choices=('auto', 'hull', 'decomposed'), default='auto')
    parser.add_argument('--contact-sampling', choices=('auto', 'surface', 'vertices'), default='auto')
    args = parser.parse_args()
    if min(args.workers, args.trials, args.max_iterations) < 1 or args.trial_start < 0:
        parser.error('workers/trials/max-iterations must be positive; trial-start must be nonnegative')
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError('Output directory already exists; refusing to mix benchmark runs')
    output.mkdir(parents=True)
    objects = args.objects or supported_objects()
    if len(set(objects)) != len(objects):
        parser.error('Duplicate objects')
    seeds = list(range(args.trial_start, args.trial_start + args.trials))
    common = ['--headless', '--sync-planner', '--no-adaptive', '--solver-backend', 'acados',
              '--sample_num', '70', '--max_rollout_length', str(args.max_iterations),
              '--collision-geometry', args.collision_geometry, '--contact-sampling', args.contact_sampling,
              '--target-type', 'ground-rotation']
    commands = {obj: [sys.executable, '-u', str(ENTRY), '--obj', obj,
                       '--trial-start', str(args.trial_start), '--trial_num', str(args.trials),
                       *common, '--ranking-metrics-jsonl', str(output / (obj + '.jsonl'))]
                for obj in objects}
    manifest = snapshot(output, commands, args.workers, seeds, args.max_iterations)
    manifest['timing_note'] = ('Per-trial latency includes GPU/CPU contention with up to %d workers; '
                               'iteration mean is weighted across all trials, including failures. '
                               'Completion times are means over successful trials only. '
                               'Initialization and compilation are excluded from iteration timing.' % args.workers)
    process_times, active, errors = {}, {}, {}
    pending = list(objects)
    env = dict(os.environ, MAX_JOBS='2', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    started = time.perf_counter()
    heartbeat = started

    def interrupted(signum, _frame):
        raise KeyboardInterrupt('signal %d' % signum)

    signal.signal(signal.SIGTERM, interrupted)
    try:
        while pending or active:
            while pending and len(active) < args.workers:
                obj = pending.pop(0)
                log = (output / (obj + '.log')).open('w')
                try:
                    process = subprocess.Popen(commands[obj], cwd=str(ROOT), env=env,
                                               stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                active[obj] = dict(process=process, log=log, start=time.perf_counter(), reported=0)
                print('START %s seeds=%s pid=%d' % (obj, seeds, process.pid), flush=True)
            for obj, job in list(active.items()):
                rows = read_rows(output / (obj + '.jsonl'))
                for row in rows[job['reported']:]:
                    print('TRIAL %s seed=%d success=%d sim_s=%s iteration_wall_ms=%.3f' % (
                        obj, row['seed'], row['success'], row['completion_time_s'],
                        1000 * row['iteration_wall_mean_s']), flush=True)
                job['reported'] = len(rows)
                status = job['process'].poll()
                if status is None:
                    continue
                process_times[obj] = time.perf_counter() - job['start']
                job['log'].close()
                del active[obj]
                if status != 0 or len(rows) != args.trials or any(r['aborted'] for r in rows):
                    errors[obj] = dict(exit_code=status, completed_trials=len(rows),
                                       message='Incomplete execution; inspect object log')
                elif [r['seed'] for r in rows] != seeds or any(r['object'] != obj for r in rows):
                    errors[obj] = dict(message='Object/seed identity mismatch')
                print('DONE %s exit=%d trials=%d process_wall_s=%.1f' % (
                    obj, status, len(rows), process_times[obj]), flush=True)
            save_reports(output, objects, seeds, commands, process_times, args.workers)
            elapsed = time.perf_counter() - started
            manifest.update(active_pids={obj: job['process'].pid for obj, job in active.items()},
                            pending_objects=pending[:], errors=errors, elapsed_wall_time_s=elapsed)
            write_json(output / 'manifest.json', manifest)
            if time.perf_counter() - heartbeat >= 60:
                print('RUNNING active=%s pending=%s elapsed_s=%.1f' % (
                    list(active), pending, elapsed), flush=True)
                heartbeat = time.perf_counter()
            if pending or active:
                time.sleep(2)
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                        error=repr(exc))
        raise
    finally:
        for obj, job in active.items():
            process = job['process']
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            job['log'].close()
            process_times[obj] = time.perf_counter() - job['start']
        result = save_reports(output, objects, seeds, commands, process_times, args.workers)
        done = not pending and not active and not errors and result['completed']
        manifest.update(completed=done, active_pids={}, errors=errors,
                        status='completed' if done else manifest.get('status') if manifest.get('status') != 'running' else 'incomplete',
                        finished_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_wall_time_s=time.perf_counter() - started)
        write_json(output / 'manifest.json', manifest)
        result.update(completed=done, errors=errors, parallel_simulations=args.workers,
                      timing_note=manifest['timing_note'], total_job_wall_time_s=manifest['elapsed_wall_time_s'])
        write_json(output / 'summary.json', result)
        print(json.dumps(result['overall']), flush=True)
    if not manifest['completed']:
        raise RuntimeError('Benchmark incomplete: inspect manifest.json and object logs')


if __name__ == '__main__':
    main()
