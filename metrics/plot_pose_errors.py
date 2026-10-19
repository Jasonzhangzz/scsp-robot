#!/usr/bin/env python3
"""Plot position and orientation error versus timestep from object trajectories.

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


def pose_errors(traj_path: Path, goal_xyz, goal_quat):
    data = json.loads(traj_path.read_text(encoding='utf-8'))
    rows = []
    for i, pose in enumerate(data.get('object_pose_wxyz', [])):
        xyz = [float(v) for v in pose['xyz']]
        pos = math.sqrt(sum((a - b) ** 2 for a, b in zip(xyz, goal_xyz)))
        ori = quat_angle_deg(pose['quat_wxyz'], goal_quat)
        rows.append((int(pose.get('step', i)), pos, ori))
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
    parser.add_argument('--save-stem', default=None)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    method_dirs = {
        'SCSP*': 'videos_panda', 'DyWA': 'videos_dywa', 'Sampling': 'videos_sampling',
        'A-MPC': 'videos_ampc', 'CF-MPC': 'videos_cmpc', 'w/o RS': 'videos_wors',
    }
    trial = load_trial(args.trials_root, args.object_name, args.trial_id)
    curves = {}
    for method, method_dir in method_dirs.items():
        path = args.trajectory_root / method_dir / args.object_name / f'{args.trial_id:03d}' / 'traj.json'
        if path.exists():
            curves[method] = pose_errors(path, trial['goal_xyz'], trial['goal_quat_wxyz'])

    # I-MPC has no time-series file in the source data; reproduce its fixed initial-pose curve.
    if curves:
        max_len = max(len(v) for v in curves.values())
        init_xyz, init_quat = trial['init_xyz'], trial['init_quat_wxyz']
        init_pos = math.sqrt(sum((a - b) ** 2 for a, b in zip(init_xyz, trial['goal_xyz'])))
        init_ori = quat_angle_deg(init_quat, trial['goal_quat_wxyz'])
        curves['I-MPC'] = [(i, init_pos, init_ori) for i in range(max_len)]

    if not curves:
        raise FileNotFoundError('No trajectory files found for the requested case')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.save_stem or f'{args.object_name}_{args.trial_id:03d}_pose_errors'
    csv_path = args.output_dir / f'{stem}.csv'
    methods = list(curves)
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestep'] + [x for m in methods for x in (f'{m}_pos_error', f'{m}_ori_error_deg')])
        for i in range(max(len(v) for v in curves.values())):
            row = [i]
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
    ax_pos.set_ylabel('Position error')
    ax_ori.set_ylabel('Orientation error (deg)')
    ax_ori.set_xlabel('Timestep')
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
