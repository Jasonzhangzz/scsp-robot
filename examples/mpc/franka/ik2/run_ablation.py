"""Run the paper's isolated Franka A1/A2/A3 or ranking sensitivity grid.

All simulation/planner arguments from test_mpc_isaac.py are forwarded.
A1: direct nearest contact, no RS or Lambda feedback to MPC.
A2: direct globally best finite Lambda contact, no RS.
A3: baseline target selection, direct contact tracking + object collision cost.
Sensitivity uses the full, unchanged RS policy and matched trial seeds.

Examples:
  python examples/mpc/franka/ik2/run_ablation.py --a1 --task flipping --headless
  python examples/mpc/franka/ik2/run_ablation.py --a3 --trial_num 5 --headless
  python examples/mpc/franka/ik2/run_ablation.py --sensitive --dry-run
  python examples/mpc/franka/ik2/run_ablation.py --sensitive --trial_num 100 \
      --sensitivity-axis model_cost_error_threshold=3,6,12 \
      --sensitivity-axis verify_beta=0.09,0.18,0.36 --headless
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

VARIANTS = {
    "full": ("cso", "full"),
    "cso-pose-only": ("pose", "full"),
    "cso-random": ("random", "full"),
    "cso-nearest": ("nearest", "full"),
    "cpo-no-object-center": ("cso", "no_object_center"),
    "cpo-direct": ("cso", "direct"),
}

TASK_TARGET = {
    "rotation": "rotation",
    "flipping": "ground-rotation",
}

_DESCRIPTIONS = {
    "full": "CSO minimizes ell_cso; CPO keeps lift/place and object-center attraction.",
    "cso-pose-only": "CSO minimizes ell_pose; CPO terminal objective stays ell_cso.",
    "cso-random": "CSO prior is uniform on the feasible set; candidates stay available to RS.",
    "cso-nearest": "CSO prior is the nearest feasible sample; candidates stay available to RS.",
    "cpo-no-object-center": "CPO sets w_o=0 and keeps reference tracking plus lift/place.",
    "cpo-direct": "CPO drops lift/place potentials and solves reference, object-center, effort, and the terminal objective.",
}


def ablation_forward_argv(variant, task, rest):
    """Arguments for ``run.py`` that implement one table cell."""
    if variant not in VARIANTS:
        raise ValueError("unknown variant %r" % (variant,))
    if task not in TASK_TARGET:
        raise ValueError("unknown task %r" % (task,))
    guidance, cpo = VARIANTS[variant]
    flags = ["--planner", "mpc", "--target-type", TASK_TARGET[task]]
    if task == "rotation":
        flags.extend(["--no_random_init_tilt"])
    flags.extend(list(rest))
    # Variant flags come last so they win over forwarded overrides.
    flags.extend([
        "--planner", "mpc",
        "--cso-guidance", guidance,
        "--cpo-variant", cpo,
        "--ablation-variant", variant,
    ])
    return flags



# Only parameters used by the live ranking/confidence policy are accepted.
# value_rho is a diagnostic EMA, and the historical T1/T2/T3 window controls
# do not gate update_verify in the current implementation: do not sweep them.
SENSITIVITY_AXES = {
    'model_cost_error_threshold': (float, (3.0, 6.0, 12.0)),
    'verify_beta': (float, (0.09, 0.18, 0.36)),
    'contact_dwell_gamma': (float, (0.5, 0.7, 0.9)),
    'model_cost_error_min_steps': (int, (1, 3, 5)),
    'contact_dwell_steps': (int, (2, 4, 8)),
    'contact_switch_radius': (float, (0.02, 0.03, 0.05)),
    'contact_switch_margin_ratio': (float, (0.04, 0.08, 0.16)),
    'contact_switch_margin_abs': (float, (0.0005, 0.001, 0.002)),
    'contact_switch_confirm_steps': (int, (1, 5, 10)),
}
DEFAULT_GRID_AXES = ('model_cost_error_threshold', 'verify_beta', 'contact_dwell_gamma')


def sensitivity_grid(specs=()):
    axes = {}
    if not specs:
        axes = {key: list(SENSITIVITY_AXES[key][1]) for key in DEFAULT_GRID_AXES}
    for spec in specs:
        key, sep, values = spec.partition('=')
        key = key.lstrip('-').replace('-', '_')
        if not sep or key not in SENSITIVITY_AXES or key in axes:
            raise ValueError('Expected a unique PARAM=v1,v2 axis; supported: ' + ', '.join(SENSITIVITY_AXES))
        cast = SENSITIVITY_AXES[key][0]
        parsed = list(dict.fromkeys(cast(value) for value in values.split(',')))
        for value in parsed:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(key + ' values must be finite and positive')
            if key in ('verify_beta', 'contact_dwell_gamma', 'contact_switch_margin_ratio') and value > 1:
                raise ValueError(key + ' must be in (0, 1]')
        axes[key] = parsed
    return [dict(zip(axes, values)) for values in itertools.product(*axes.values())]


def experiment_argv(mode, task, rest):
    if mode not in ('full', 'a1', 'a2', 'a3'):
        raise ValueError('Unknown ablation: ' + mode)
    if '--floating-fingertip' in rest or '--sampling-c3' in rest:
        raise ValueError('These experiments require the Franka explicit MPC runtime')
    flags = []
    if task:
        flags += ['--target-type', TASK_TARGET[task]]
        if task == 'rotation':
            flags += ['--no_random_init_tilt']
    flags += list(rest)
    # Fix the controller/model family after user overrides; only the named
    # experiment varies from the test_mpc_isaac runtime.
    flags += ['--cso-guidance', 'cso', '--cpo-variant', 'full',
              '--ablation-mode', mode, '--ablation-variant', mode]
    return flags


def _child_command(flags):
    script = Path(__file__).resolve().with_name('test_mpc_isaac_ablation.py')
    return [sys.executable, '-u', str(script), *flags]


def _mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def summarize_grid_cell(rows):
    if not rows or any(row.get('aborted') for row in rows):
        raise ValueError('Missing or aborted trial data cannot be aggregated')
    if len({row['trial'] for row in rows}) != len(rows):
        raise ValueError('Duplicate trial IDs')
    successful = [row for row in rows if row['success']]
    return dict(trials=len(rows), successes=len(successful),
        success_rate=len(successful)/len(rows),
        completion_time_mean_s=_mean(successful, 'completion_time_s'),
        completion_steps_mean=_mean(successful, 'completion_steps'),
        solve_dt_mean_ms=_mean(rows, 'solve_dt_mean_ms'),
        planning_wall_mean_ms=_mean(rows, 'planning_wall_mean_ms'),
        final_pos_err_mean=_mean(rows, 'final_pos_err'),
        final_quat_err_mean=_mean(rows, 'final_quat_err'))


def run_sensitivity(args, rest):
    grid = sensitivity_grid(args.sensitivity_axis)
    output = Path(args.sensitivity_output).resolve()
    commands, cells = [], []
    for index, parameters in enumerate(grid):
        cell = output / ('cell_%03d' % index)
        flags = experiment_argv('full', args.task, rest)
        flags += ['--headless']
        for key, value in parameters.items():
            flags += ['--' + key, str(value)]
        flags += ['--ablation-variant', 'sensitive_%03d' % index,
                  '--ranking-metrics-jsonl', str(cell/'trials.jsonl'),
                  '--rollout-summary-json', str(cell/'rollout_summary.json')]
        commands.append(dict(index=index, parameters=parameters, command=_child_command(flags)))
    if args.dry_run:
        print(json.dumps(dict(output=str(output), cells=commands), indent=2))
        return 0
    # Reusing JSONL files would mix trial cohorts and corrupt paired statistics.
    if output.exists() and any(output.iterdir()):
        raise ValueError('Sensitivity output must be empty: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    (output/'grid.json').write_text(json.dumps(commands, indent=2)+'\n')
    expected_trials = None
    for entry in commands:
        cell = output / ('cell_%03d' % entry['index'])
        cell.mkdir()
        (cell/'parameters.json').write_text(json.dumps(entry, indent=2)+'\n')
        print('[sensitive] %d/%d %s' % (entry['index']+1, len(commands), entry['parameters']), flush=True)
        with (cell/'run.log').open('w') as log:
            subprocess.run(entry['command'], stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(line) for line in (cell/'trials.jsonl').read_text().splitlines() if line.strip()]
        trial_ids = sorted(row['trial'] for row in rows)
        if expected_trials is not None and trial_ids != expected_trials:
            raise ValueError('Sensitivity cells must have identical trial IDs/seeds')
        expected_trials = trial_ids
        record = dict(index=entry['index'], parameters=entry['parameters'],
                      **summarize_grid_cell(rows))
        cells.append(record)
        (output/'sensitivity_summary.json').write_text(json.dumps(cells, indent=2, allow_nan=False)+'\n')
    print('[sensitive] results: ' + str(output/'sensitivity_summary.json'), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('a1', 'a2', 'a3', 'sensitive'):
        modes.add_argument('--'+mode, action='store_true')
    # Preserve the previous component experiments for existing callers.
    modes.add_argument('--variant', choices=tuple(VARIANTS), help='Legacy component experiment')
    parser.add_argument('--task', choices=tuple(TASK_TARGET), default=None,
                        help='Optional; otherwise use the same default target as test_mpc_isaac.py')
    parser.add_argument('--sensitivity-axis', action='append', default=[], metavar='PARAM=v1,v2')
    parser.add_argument('--sensitivity-output', default='output/ablation_ranking_sensitivity')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without importing simulators or solving')
    args, rest = parser.parse_known_args(argv)
    try:
        if args.sensitive:
            return run_sensitivity(args, rest)
        if args.sensitivity_axis:
            raise ValueError('--sensitivity-axis requires --sensitive')
        if args.variant:
            flags = ablation_forward_argv(args.variant, args.task or 'flipping', rest)
            # The isolated dispatcher already inserts --planner mpc.
            flags = flags[2:-8] + flags[-6:]
        else:
            mode = next(mode for mode in ('a1', 'a2', 'a3') if getattr(args, mode))
            flags = experiment_argv(mode, args.task, rest)
        command = _child_command(flags)
        if not args.dry_run:
            print('[ablation] %s; isolated test_mpc_isaac runtime' % (args.variant or mode), flush=True)
        if args.dry_run:
            print(json.dumps(command, indent=2))
            return 0
        return subprocess.call(command)
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
