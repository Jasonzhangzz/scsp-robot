"""Completion and per-planning-round wall-clock statistics."""
import numpy as np


TIMING_DEFINITIONS = {
    'execution_time_s': 'Wall time from first planning round to successful settling, including planning and physics, excluding scene initialization; null for failures.',
    'planning_time_ms': 'Wall time of each complete contact-point and force planning round, excluding physical execution.',
    'simulation_time_s': 'Elapsed MuJoCo simulation time; separate from wall-clock execution_time_s.',
}


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return dict(n=0, mean=None, std=None, median=None, p95=None, min=None, max=None)
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('Timing samples must be finite and nonnegative')
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std()),
                median=float(np.median(a)), p95=float(np.percentile(a,95)),
                min=float(a.min()), max=float(a.max()))


def trial_timing(rows):
    last = rows[-1] if rows else {}
    success = bool(last.get('success', False))
    return dict(execution_time_s=last.get('elapsed_wall_time_s') if success else None,
                elapsed_wall_time_s=last.get('elapsed_wall_time_s'),
                simulation_time_s=last.get('simulation_time_s'),
                planning_time_ms=distribution([r['planning_time_ms'] for r in rows]),
                planning_rounds=len(rows))


def aggregate_trials(trials, rows):
    successful = [t for t in trials if t['success']]
    return dict(successes=len(successful), total_trials=len(trials),
                success_rate=len(successful)/len(trials) if trials else None,
                execution_time_s_success_only=distribution([t['execution_time_s'] for t in successful]),
                simulation_time_s_success_only=distribution([t['simulation_time_s'] for t in successful]),
                planning_time_ms_all_rounds=distribution([r['planning_time_ms'] for r in rows]),
                planning_time_ms_trial_means=distribution([t['planning_time_ms']['mean'] for t in trials
                                                          if t['planning_time_ms']['n']]),
                failed_trials=len(trials)-len(successful))
