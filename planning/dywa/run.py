"""Evaluate official DyWA on a single object or a reproducible object cohort."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OBJECTS = ('foam_brick', 'stanford_bunny2', 'elephant', 'piggy_bank',
           'teapot', 'mug', 'rubber_duck')
FIELDS = ('object', 'trial', 'status', 'success', 'steps', 'completion_steps',
          'final_pos_err', 'final_quat_err', 'min_pos_err', 'min_quat_err',
          'policy_interval_s', 'choose_dt_mean', 'choose_dt_max',
          'policy_unchanged', 'files_unchanged', 'output_dir', 'error')


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    from planning.dywa.runtime import (ensure_acc_runtime, build_experiment,
                                       execute_trial, DEFAULT_CKPT_DIR)
    ensure_acc_runtime()
    os.environ.setdefault('SCSP_PLANNER_ONLY', '1')
    from examples.mpc.franka.runtime.mpc import _add_rollout_policy_args

    parser = _add_rollout_policy_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--dywa', action='store_true', help='Use official DyWA with the authorized history-buffer fix.')
    parser.add_argument('--dywa-objects', nargs='+', choices=OBJECTS,
                        help='Object cohort; defaults to --obj.')
    parser.add_argument('--dywa-ckpt-dir', default=str(DEFAULT_CKPT_DIR))
    parser.add_argument('--dywa-output', default=str(ROOT / 'outputs/dywa_isaac'))
    parser.add_argument('--dywa-csv', default='', help='CSV path; default <dywa-output>/results.csv.')
    parser.add_argument('--dywa-dry-run', action='store_true', help='Write inputs without executing policy.')
    parser.set_defaults(trial_num=3, headless=False)
    args = parser.parse_args()
    if args.trial_start < 0 or args.trial_num <= 0 or args.max_rollout_length <= 0:
        parser.error('trial-start must be nonnegative; trial_num and max_rollout_length must be positive')
    if args.physics_backend != 'isaac':
        parser.error('DyWA uses its official Isaac Gym environment')
    if not args.sim_device.startswith('cuda'):
        parser.error('The official DyWA environment and renderer require a compatible CUDA device')
    if args.dywa_objects and len(set(args.dywa_objects)) != len(args.dywa_objects):
        parser.error('--dywa-objects must not contain duplicate objects')
    output = Path(args.dywa_output).expanduser().resolve()
    if output.exists() and any(output.glob('*/trial_*/spec.json')):
        parser.error('Output already contains trial specs; select a new --dywa-output directory')
    csv_path = Path(args.dywa_csv).expanduser() if args.dywa_csv else output / 'results.csv'
    rows = []
    for obj in args.dywa_objects or [args.obj]:
        args.obj = obj
        for trial in range(args.trial_start, args.trial_start + args.trial_num):
            trial_dir = output / obj / ('trial_%03d' % trial)
            trial_dir.mkdir(parents=True, exist_ok=True)
            row = dict(object=obj, trial=trial, status='error', output_dir=str(trial_dir))
            try:
                spec = build_experiment(args, trial)
                spec['output_dir'] = str(trial_dir)
                (trial_dir / 'spec.json').write_text(json.dumps(spec, indent=2))
                if args.dywa_dry_run:
                    row['status'] = 'dry_run'
                else:
                    row.update(execute_trial(spec))
                    row['status'] = 'completed'
            except Exception as exc:
                row['error'] = '%s: %s' % (type(exc).__name__, exc)
                preflight = trial_dir / 'preflight.json'
                if preflight.exists():
                    failure = json.loads(preflight.read_text()).get('device_error')
                    if failure:
                        row['error'] = failure.splitlines()[0]
                (trial_dir / 'error.txt').write_text(row['error'] + '\n')
            rows.append(row)
            save_csv(csv_path, rows)
            print('dywa_case: ' + json.dumps(row), flush=True)
    completed = [row for row in rows if row['status'] == 'completed']
    summary = dict(requested=len(rows), completed=len(completed),
                   errors=sum(row['status'] == 'error' for row in rows),
                   dry_run=bool(args.dywa_dry_run), csv=str(csv_path),
                   success_rate=(sum(row['success'] for row in completed) / len(completed)
                                 if len(completed) == len(rows) else None))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return 1 if summary['errors'] else 0


if __name__ == '__main__':
    sys.exit(main())
