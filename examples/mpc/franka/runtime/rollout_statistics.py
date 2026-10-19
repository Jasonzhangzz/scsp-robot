"""Aggregate complete paired-seed rollout records, including timing moments."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import shlex

OBJECTS = ('foam_brick', 'stanford_bunny2', 'elephant', 'piggy_bank',
           'teapot', 'mug', 'rubber_duck')


def mean_std(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values) > 1 else None)


def aggregate(rows, expected_seeds):
    if sorted(row['seed'] for row in rows) != list(expected_seeds):
        raise ValueError('Missing or duplicate seeds')
    if any(row.get('aborted') for row in rows):
        raise ValueError('Aborted rollout cannot be reported as a completed experiment')
    if any(row['planning_wall_count'] != row['steps'] for row in rows):
        raise ValueError('Planning cycles must match executed policy steps')
    successful = [row for row in rows if row['success']]
    count = sum(row['planning_wall_count'] for row in rows)
    total = sum(row['planning_wall_total_s'] for row in rows)
    squares = sum(row['planning_wall_sum_squares_s2'] for row in rows)
    mean = total/count if count else None
    variance = max(0., (squares-total*total/count)/(count-1)) if count > 1 else None
    return dict(
        trials=len(rows), successes=len(successful), success_rate=len(successful)/len(rows),
        success_seeds=sorted(row['seed'] for row in successful),
        stable_flip_successes=sum(bool(row.get('stable_flip_success')) for row in rows),
        stable_flip_rate=(sum(bool(row.get('stable_flip_success')) for row in rows)/len(rows)
                          if all('stable_flip_success' in row for row in rows) else None),
        stable_flip_seeds=sorted(row['seed'] for row in rows if row.get('stable_flip_success')),
        first_stable_flip_s=mean_std([row['first_stable_flip_time_s'] for row in rows
                                     if row.get('stable_flip_success') and row.get('first_stable_flip_time_s') is not None]),
        completion_sim_s=mean_std([(row['completion_time_s'] if row.get('completion_time_s') is not None else row['completion_steps']*row['policy_interval_s']) for row in successful]),
        completion_wall_s=mean_std([row['completion_wall_time_s'] for row in successful]),
        planning_cycle_ms=dict(n=count, mean=mean*1000 if mean is not None else None,
                               std=variance**.5*1000 if variance is not None else None),
        planning_trial_mean_ms=mean_std([row['planning_wall_mean_ms'] for row in rows
                                        if row['planning_wall_mean_ms'] is not None]),
        mpc_failures=sum(row.get('mpc_failures', 0) for row in rows),
        lambda_failures=sum(row.get('lambda_failures', 0) for row in rows),
        mpc_fallbacks=sum(row.get('mpc_fallbacks', 0) for row in rows),
        lambda_fallbacks=sum(row.get('lambda_fallbacks', 0) for row in rows))


def formatted(values):
    if values['mean'] is None:
        return '—'
    return '%.2f ± %s' % (values['mean'], '%.2f' % values['std'] if values['std'] is not None else '—')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--objects', nargs='+', choices=OBJECTS, default=OBJECTS)
    args = parser.parse_args()
    summaries, all_rows, cases, flip_cases, backends = {}, [], [], [], set()
    for obj in args.objects:
        rows = [json.loads(line) for line in (args.directory/(obj+'.jsonl')).read_text().splitlines()]
        backend = rows[0].get('physics_backend', 'mjwarp')
        if any(row.get('physics_backend', 'mjwarp') != backend for row in rows):
            raise ValueError('Different physics backends mixed for '+obj)
        backends.add(backend)
        scene = rows[0].get('mjwarp_scene', 'legacy')
        if any(row.get('mjwarp_scene', 'legacy') != scene for row in rows):
            raise ValueError('Different physical scenes mixed for '+obj)
        if len({(row.get('physics_scene_sha256'), row.get('joint_velocity_limiter')) for row in rows}) != 1:
            raise ValueError('Different scene configurations mixed for '+obj)
        summaries[obj] = aggregate(rows, range(args.trials))
        summaries[obj]['mjwarp_scene'] = scene
        summaries[obj]['physics_backend'] = backend
        summaries[obj]['async_planner'] = bool(rows[0].get('async_planner', False))
        if len({bool(row.get('async_planner', False)) for row in rows}) != 1:
            raise ValueError('Different scheduling modes mixed for '+obj)
        command_path = args.directory/(obj+'_command.json')
        source_command = json.loads(command_path.read_text()) if command_path.exists() else []
        def capacity(flag, old_default):
            return int(source_command[source_command.index(flag)+1]) if flag in source_command else old_default
        nconmax = rows[0].get('mjwarp_nconmax', capacity('--mjwarp-nconmax', 128))
        njmax = rows[0].get('mjwarp_njmax', capacity('--mjwarp-njmax', 512))
        if backend == 'mjwarp':
            summaries[obj]['mjwarp_capacities'] = dict(nconmax=nconmax, njmax=njmax)
        all_rows.extend(rows)
        # Preserve the original world index and complete initial seed set.
        # Changing nworld changes the GPU numerical path even for one seed.
        command = ('python examples/mpc/franka/ik2/test_mpc_isaac.py '
                   '--physics-backend %s --headless %s --num-envs %s --obj %s '
                   '--trial-start 0 --trial_num %s --max_rollout_length 5000 '
                   '--solver-backend acados' % (backend, '--async-planner' if rows[0].get('async_planner') else '--sync-planner', rows[0].get('evaluation_workers', rows[0]['num_envs']), obj, args.trials))
        if backend == 'mjwarp':
            command += ' --mjwarp-scene %s --mjwarp-nconmax %s --mjwarp-njmax %s' % (scene, nconmax, njmax)
        else:
            command += ' --headless-realtime' if rows[0].get('headless_realtime', False) else ' --no-headless-realtime'
        if source_command:
            # Preserve the actual interpreter and geometry flags, but do not
            # append a replay to the original result or diagnostic files.
            replay, i = [], 0
            while i < len(source_command):
                if source_command[i] in ('--ranking-metrics-jsonl', '--rollout-summary-json', '--rollout-trace-jsonl'):
                    i += 2
                else:
                    replay.append(source_command[i])
                    i += 1
            if not any(flag in replay for flag in ('--async-planner', '--sync-planner')):
                replay.append('--async-planner' if rows[0].get('async_planner') else '--sync-planner')
            if backend == 'isaac' and not any(flag in replay for flag in ('--headless-realtime', '--no-headless-realtime')):
                replay.append('--headless-realtime' if rows[0].get('headless_realtime') else '--no-headless-realtime')
            command = shlex.join(replay)
        for row in rows:
            if row.get('stable_flip_success'):
                flip_cases.append(dict(object=obj, seed=row['seed'], numpy_seed=row['numpy_seed'],
                    first_stable_flip_time_s=row['first_stable_flip_time_s'],
                    final_task_success=bool(row['success']), command=command))
            if row['success']:
                cases.append(dict(object=obj, seed=row['seed'], numpy_seed=row['numpy_seed'],
                                  world_index=row.get('parallel_worker', row['seed']), batch_seeds=list(range(args.trials)),
                                  steps=row['completion_steps'], sim_time_s=(row['completion_time_s'] if row.get('completion_time_s') is not None else row['completion_steps']*.02),
                                  wall_time_s=row['completion_wall_time_s'], command=command))
    if len(backends) != 1:
        raise ValueError('Different physics backends mixed across the experiment')
    result = dict(physics_backend=next(iter(backends)), headless=True, trials_per_object=args.trials,
                  max_rollout_length=5000, policy_interval_s=.02, sample_std_ddof=1,
                  planning_definition='wall time of CSO + approach + MPC callback; excludes IPC, physics and rendering',
                  objects=summaries)
    (args.directory/'statistics.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.directory/'successful_cases.json').write_text(json.dumps(cases, indent=2)+'\n')
    (args.directory/'stable_flip_cases.json').write_text(json.dumps(flip_cases, indent=2)+'\n')
    with (args.directory/'trial_statistics.csv').open('w') as output:
        columns = ['object', 'seed', 'numpy_seed', 'success', 'stable_flip_success', 'first_stable_flip_time_s', 'steps', 'completion_time_s',
                   'completion_wall_time_s', 'planning_wall_count', 'planning_wall_mean_ms',
                   'planning_wall_std_ms', 'mpc_failures', 'lambda_failures']
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda row: (row['object'], row['seed'])))
    table = '| 物体 | 曾稳定翻起 | 最终任务成功 | 成功仿真耗时 (s) | 成功墙钟耗时 (s) | Planning/周期 (ms) | 最终成功 seeds |\n'
    table += '|---|---:|---:|---:|---:|---:|---|\n'
    for obj, result in summaries.items():
        table += '| %s | %s/%s | %s/%s (%.0f%%) | %s | %s | %s | %s |\n' % (
            obj, result['stable_flip_successes'], result['trials'], result['successes'], result['trials'], 100*result['success_rate'],
            formatted(result['completion_sim_s']), formatted(result['completion_wall_s']),
            formatted(result['planning_cycle_ms']), ', '.join(map(str,result['success_seeds'])) or '—')
    (args.directory/'table.md').write_text(table)
    print(table)


if __name__ == '__main__':
    main()
