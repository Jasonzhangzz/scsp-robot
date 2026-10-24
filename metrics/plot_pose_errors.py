#!/usr/bin/env python3
"""Plot position and orientation error versus physical execution time.

The bundled data layout mirrors the original repository:
  data/trajectories/videos_<method>/<object>/<trial>/traj.json
  data/trials/<object>/trails.json

Position error is Euclidean distance in trajectory units. Orientation error is
shortest quaternion angle in degrees.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

try:
    from trajectory_io import METHOD_TO_OUTPUT_MODE, cached_rows, execution_time_s, successful
except ImportError:  # pragma: no cover - allows importing as metrics.plot_pose_errors
    from metrics.trajectory_io import METHOD_TO_OUTPUT_MODE, cached_rows, execution_time_s, successful


def quat_angle_deg(q, goal):
    q = [float(v) for v in q]
    goal = [float(v) for v in goal]
    qn = math.sqrt(sum(v * v for v in q))
    gn = math.sqrt(sum(v * v for v in goal))
    if qn == 0.0 or gn == 0.0:
        raise ValueError('Quaternion norm is zero')
    dot = abs(sum(a * b for a, b in zip(q, goal)) / (qn * gn))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def pose_errors(traj_path: Path, goal_xyz, goal_quat, sample_rows=None):
    data = json.loads(traj_path.read_text(encoding='utf-8'))
    rows = []
    cached_times = execution_time_s(sample_rows) if sample_rows is not None else None
    time_offset = None
    for i, pose in enumerate(data.get('object_pose_wxyz', [])):
        xyz = [float(v) for v in pose['xyz']]
        pos = math.sqrt(sum((a - b) ** 2 for a, b in zip(xyz, goal_xyz)))
        ori = quat_angle_deg(pose['quat_wxyz'], goal_quat)
        if cached_times is not None and i < len(cached_times):
            time_s = float(cached_times[i])
        else:
            raw_time = pose.get('sim_time_s')
            if raw_time is None:
                raw_time = pose.get('execution_time_s')
            if raw_time is None:
                raw_time = pose.get('step', i) * 0.02
            time_s = float(raw_time)
            if time_offset is None:
                time_offset = time_s
            time_s -= time_offset
        rows.append((time_s, pos, ori))
    if not rows:
        raise ValueError(f'No object_pose_wxyz in {traj_path}')
    return rows


def load_trial(trials_root: Path, object_name: str, trial_id: int):
    path = trials_root / object_name / 'trails.json'
    entries = json.loads(path.read_text(encoding='utf-8'))
    for entry in entries:
        if int(entry['trial']) == trial_id:
            return entry
    raise KeyError(f'trial {trial_id} missing from {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--trajectory-root', type=Path, default=here/'data/trajectories')
    parser.add_argument('--trials-root', type=Path, default=here/'data/trials')
    parser.add_argument('--object-name', default='elephant')
    parser.add_argument('--trial-id', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, default=here/'data/pose_errors')
    parser.add_argument(
        '--execution-time-root', type=Path,
        default=here.parent/'output/trials/trajectories',
        help='Real rollout cache root. CSV rows use sim_time_s as execution time.',
    )
    parser.add_argument('--success-only', action='store_true',
                        help='Use only trajectories whose final cache row is successful.')
    parser.add_argument('--include-failed', action='store_true',
                        help='Plot cached samples even when the rollout did not succeed.')
    parser.add_argument('--real-only', action='store_true',
                        help='Skip archived JSON curves when no real output cache exists.')
    parser.add_argument('--save-stem', default=None)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    method_dirs = {
        'SCSP*': 'videos_panda', 'DyWA': 'videos_dywa', 'Sampling': 'videos_sampling',
        'ContactSDF': None, 'STOCS': None, 'A-MPC': 'videos_ampc',
        'CF-MPC': 'videos_cmpc', 'w/o RS': 'videos_wors',
    }
    trial = load_trial(args.trials_root, args.object_name, args.trial_id)
    curves = {}
    for method, method_dir in method_dirs.items():
        cached = cached_rows(args.execution_time_root, method, args.object_name, args.trial_id)
        if cached is not None:
            _, sample_rows = cached
            if args.success_only and not args.include_failed and not successful(sample_rows):
                continue
            values = []
            times = execution_time_s(sample_rows)
            for i, row in enumerate(sample_rows):
                pos = row.get('position_error')
                if pos is None:
                    break
                if 'quat_wxyz' in row:
                    ori = quat_angle_deg(row['quat_wxyz'], trial['goal_quat_wxyz'])
                elif all(key in row for key in ('qw', 'qx', 'qy', 'qz')):
                    ori = quat_angle_deg(
                        [row['qw'], row['qx'], row['qy'], row['qz']],
                        trial['goal_quat_wxyz'],
                    )
                else:
                    ori = row.get('orientation_error')
                if ori is None:
                    break
                values.append((float(times[i]), float(pos), float(ori)))
            if values:
                curves[method] = values
                continue
        if method_dir is None:
            continue
        if args.real_only and method in METHOD_TO_OUTPUT_MODE:
            continue
        path = args.trajectory_root / method_dir / args.object_name / f'{args.trial_id:03d}' / 'traj.json'
        if path.exists():
            raw = json.loads(path.read_text(encoding='utf-8'))
            if args.success_only and not args.include_failed and not raw.get('success', False):
                continue
            curves[method] = pose_errors(path, trial['goal_xyz'], trial['goal_quat_wxyz'])

    # I-MPC has no time-series file in the source data; reproduce its fixed initial-pose curve.
    if curves:
        max_len = max(len(v) for v in curves.values())
        init_xyz, init_quat = trial['init_xyz'], trial['init_quat_wxyz']
        init_pos = math.sqrt(sum((a - b) ** 2 for a, b in zip(init_xyz, trial['goal_xyz'])))
        init_ori = quat_angle_deg(init_quat, trial['goal_quat_wxyz'])
        reference_times = next(iter(curves.values()))
        curves['I-MPC'] = [
            (reference_times[i][0] if i < len(reference_times) else i * 0.02,
             init_pos, init_ori)
            for i in range(max_len)
        ]

    if not curves:
        raise FileNotFoundError('No trajectory files found for the requested case')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.save_stem or f'{args.object_name}_{args.trial_id:03d}_pose_errors'
    csv_path = args.output_dir / f'{stem}.csv'
    methods = list(curves)
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['execution_time_s'] + [x for m in methods for x in (f'{m}_pos_error', f'{m}_ori_error_deg')])
        for i in range(max(len(v) for v in curves.values())):
            row = [f'{max((curves[m][i][0] for m in methods if i < len(curves[m])), default=0.0):.8f}']
            for method in methods:
                values = curves[method]
                if i < len(values): row.extend([f'{values[i][1]:.8f}', f'{values[i][2]:.8f}'])
                else: row.extend(['', ''])
            writer.writerow(row)

    fig, (ax_pos, ax_ori) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for method, values in curves.items():
        x = [v[0] for v in values]
        ax_pos.plot(x, [v[1] for v in values], linewidth=1.6, label=method)
        ax_ori.plot(x, [v[2] for v in values], linewidth=1.6, label=method)
    ax_pos.set_ylabel('Pos Error (m)', fontsize=18)
    ax_ori.set_ylabel('Orientation error (deg)', fontsize=18)
    ax_ori.set_xlabel('Execution Time (s)', fontsize=18)
    ax_pos.tick_params(axis='both', labelsize=14)
    ax_ori.tick_params(axis='both', labelsize=14)
    ax_pos.grid(True, linestyle='--', alpha=0.4)
    ax_ori.grid(True, linestyle='--', alpha=0.4)
    ax_pos.legend(ncol=3, fontsize=8)
    fig.suptitle(f'{args.object_name} trial {args.trial_id}: pose errors')
    fig.tight_layout()
    fig.savefig(args.output_dir / f'{stem}.svg', format='svg')
    if args.show: plt.show()
    plt.close(fig)
    print(csv_path)


if __name__ == '__main__':
    main()
