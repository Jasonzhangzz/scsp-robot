"""Decision sampling and CDR for experiment.tex (control-step terminal reward).

Replay the complete unmodified prefix to reconstruct simulator, controller,
solver warm starts and RNG history. Reject unmatched prefixes rather than
silently treating a different state as a counterfactual.
"""
from __future__ import annotations

import copy
import numpy as np

from utils.ranking_metrics import discounted_terminal_reward, _mean_std


def _local_to_world(point, curr_q):
    if point is None:
        return None
    q = np.asarray(curr_q, dtype=float).reshape(-1)
    if q.size < 7 or not np.isfinite(q[:7]).all():
        return None
    w, x, y, z = q[3:7]
    n = float(np.linalg.norm([w, x, y, z]))
    if n < 1e-12:
        return None
    w, x, y, z = np.asarray([w, x, y, z]) / n
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    return (q[:3] + rotation @ np.asarray(point, dtype=float)).tolist()


def contact_decision_context(opt, curr_q, near_idx, guide_idx, *, near_arm,
                             observed, accept, hold, switch_observed, stage):
    """Describe the binary commitment before CPO, independent of index jitter."""
    best_idx = getattr(opt, 'last_global_idx', None)
    points = np.asarray(opt.sample_point)

    def point(idx):
        return None if idx is None else points[int(idx)].tolist()

    def distinct(a, b):
        if a is None or b is None:
            return False
        geodesic = getattr(opt, 'sample_geodesic', None)
        distance = (np.linalg.norm(points[int(a)] - points[int(b)]) if geodesic is None
                    else geodesic[int(a), int(b)])
        return bool(distance > .03)

    relocating = bool((not accept or switch_observed) and not hold)
    decision = 'switch' if relocating and distinct(near_idx, guide_idx) else 'stay'
    switch_idx = guide_idx if decision == 'switch' else best_idx
    candidate_ids = np.asarray(getattr(opt, 'last_candidate_ids', []), dtype=int)
    costs = np.asarray(getattr(opt, 'last_candidate_costs', []), dtype=float)
    hits = np.flatnonzero(candidate_ids == switch_idx) if switch_idx is not None else []
    feasible = bool(len(hits) and np.isfinite(costs[int(hits[0])]))
    eligible = bool((near_arm or observed) and (decision == 'switch' or accept or hold)
                    and distinct(near_idx, switch_idx) and feasible)
    best_world = _local_to_world(point(guide_idx), curr_q)
    near_world = _local_to_world(point(near_idx), curr_q)
    return dict(decision=decision, eligible=eligible, near_idx=near_idx,
                optimal_idx=best_idx, guide_idx=guide_idx, switch_idx=switch_idx,
                best_contact_idx=guide_idx, best_contact_local=point(guide_idx),
                near_local=point(near_idx), optimal_local=point(best_idx),
                switch_local=point(switch_idx),
                best_contact_world=best_world,
                best_contact_point_world=best_world,
                p_arm_world=near_world,
                near_world=near_world,
                optimal_world=_local_to_world(point(best_idx), curr_q),
                switch_world=_local_to_world(point(switch_idx), curr_q),
                stage=stage,
                policy_q=np.asarray(curr_q).tolist())


def assert_matched(actual, expected, *, atol, path='prefix'):
    """Compare nested physical observations/actions, allowing only given atol."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise RuntimeError(f'{path}: replay fields differ')
        for key in expected:
            assert_matched(actual[key], expected[key], atol=atol, path=f'{path}.{key}')
    elif isinstance(expected, (list, tuple, np.ndarray)):
        try:
            a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        except (TypeError, ValueError):
            if not isinstance(actual, (list, tuple, np.ndarray)) or len(actual) != len(expected):
                raise RuntimeError(f'{path}: replay sequence shape differs')
            for index, (a, b) in enumerate(zip(actual, expected)):
                assert_matched(a, b, atol=atol, path=f'{path}[{index}]')
        else:
            if a.shape != b.shape or not np.allclose(a, b, atol=atol, rtol=0, equal_nan=True):
                raise RuntimeError(f'{path}: replay state/action mismatch (atol={atol:g})')
    elif isinstance(expected, (float, np.floating)):
        if not np.isclose(actual, expected, atol=atol, rtol=0, equal_nan=True):
            raise RuntimeError(f'{path}: replay scalar mismatch')
    elif actual != expected:
        raise RuntimeError(f'{path}: replay decision/value mismatch')


class DecisionSampler:
    """Record switching onsets and best-change + accepted-near timestamps.

    Near sample jitter does not create an event. Consecutive commands pursuing
    the same destination are one switching event. Initial ranking is not a
    best-change event. Travel without a nearby/observed patch is not eligible.
    """
    def __init__(self, *, best_change_distance=.01, target=None, expected=None,
                 forced=None, match_atol=1e-7):
        if (not np.isfinite(best_change_distance) or best_change_distance <= 0
                or not np.isfinite(match_atol) or match_atol < 0):
            raise ValueError('best-change-distance must be positive; match-atol nonnegative')
        self.distance = best_change_distance
        self.target, self.expected, self.forced = target, expected, forced
        self.atol = match_atol
        self.step, self.sim_time = 0, 0.
        self.events, self.decisions = [], []
        self.previous_best = self.previous_switch = None
        self.switch_count = self.evaluations = 0
        self.applied = False

    def __call__(self, context):
        context = copy.deepcopy(context)
        context.update(step=self.step, sim_time=self.sim_time)
        self.decisions.append(context)
        self.evaluations += 1
        best = context['optimal_local']
        changed = (best is not None and self.previous_best is not None
                   and np.linalg.norm(np.asarray(best)-self.previous_best) > self.distance)
        switching = context['eligible'] and context['decision'] == 'switch'
        switch = context['switch_local'] if switching else None
        onset = switching and (self.previous_switch is None or
            np.linalg.norm(np.asarray(switch)-self.previous_switch) > self.distance)
        if onset:
            self.switch_count += 1
        if context['eligible'] and (onset or (changed and context['decision'] == 'stay')):
            context['event_kind'] = 'switch' if onset else 'best_changed_stay'
            self.events.append(context)
        self.previous_best = None if best is None else np.asarray(best)
        self.previous_switch = None if switch is None else np.asarray(switch)
        if self.target is not None and self.step <= self.target:
            if self.expected is None or self.evaluations > len(self.expected):
                raise RuntimeError('Replay has an unexpected ranking evaluation')
            assert_matched(context, self.expected[self.evaluations-1], atol=self.atol,
                           path=f'decision[{self.step}]')
        if self.step == self.target:
            if self.applied or not context['eligible']:
                raise RuntimeError('Counterfactual target is repeated or ineligible')
            self.applied = True
            return self.forced
        return None


def select_events(events, *, max_per_kind=10, seed=0):
    """Uniform, reproducible subsampling within each event stratum; 0 keeps all."""
    if max_per_kind < 0:
        raise ValueError('max-per-kind must be nonnegative')
    rng = np.random.default_rng(seed)
    selected = []
    for kind in ('switch', 'best_changed_stay'):
        group = [e for e in events if e['event_kind'] == kind]
        if max_per_kind and len(group) > max_per_kind:
            indices = sorted(rng.choice(len(group), max_per_kind, replace=False))
            group = [group[i] for i in indices]
        selected.extend(group)
    return sorted(selected, key=lambda e: e['step'])


def continuation_outcome(summary, start_step, *, gamma, policy_dt=.02):
    if summary.get('aborted'):
        raise RuntimeError('Aborted rollout is not a task failure')
    remaining = int(summary['steps']) - int(start_step)
    if remaining < 0:
        raise RuntimeError('Rollout terminated before the sampled decision')
    success = bool(summary['success'])
    return dict(success=success, steps=remaining,
                completion_steps=remaining if success else None,
                completion_time_s=remaining*policy_dt if success else None,
                reward=discounted_terminal_reward(success, remaining, gamma=gamma))


def evaluate_state(event, rollouts, *, gamma):
    """Estimate each Q by averaging rewards BEFORE taking max and regret."""
    if not 0 < gamma < 1:
        raise ValueError('CDR gamma must be in (0, 1)')
    q = {}
    for branch in ('switch', 'stay'):
        outcomes = rollouts[branch]
        if not outcomes:
            raise ValueError('Both decisions require at least one rollout')
        rewards = [discounted_terminal_reward(o['success'], o['steps'], gamma=gamma)
                   for o in outcomes]
        q[branch] = float(np.mean(rewards))
    rs = event['decision']
    opposite = 'stay' if rs == 'switch' else 'switch'
    advantage = q[rs] - q[opposite]
    return dict(**event, rollouts=rollouts, q=q, q_rs=q[rs],
                value_original=q[rs], value_perturbed=q[opposite],
                value_delta=q[opposite] - q[rs],
                advantage_rs=advantage, advantage_switch=q['switch']-q['stay'],
                cdr=max(q.values())-q[rs])


def summarize_cdr(states, trials, *, gamma, policy_dt=.02):
    from utils.switching_frequency import statistics as switching_statistics
    physical = switching_statistics(trials)
    successful = [t for t in trials if t['success'] and not t.get('aborted')]
    steps = sum(t['steps'] for t in successful)
    switches = sum(t['rs_switch_events'] for t in successful)
    durations = [float(t.get('completion_time_s') or t['steps']*policy_dt)
                 for t in successful]
    total_duration = sum(durations)
    best_events = [t.get('best_contact_point_world_change_events',
                         t.get('guided_contact_switch_events')) for t in successful]
    best_events = [int(v) for v in best_events if v is not None]
    original_values = [s['value_original'] for s in states if 'value_original' in s]
    perturbed_values = [s['value_perturbed'] for s in states if 'value_perturbed' in s]
    return dict(gamma=gamma, policy_interval_s=policy_dt, n_states=len(states),
        source_trials=len(trials), successful_source_trials=len(successful),
        cdr=_mean_std([s['cdr'] for s in states]),
        advantage_rs=_mean_std([s['advantage_rs'] for s in states]),
        by_event={kind: dict(cdr=_mean_std([s['cdr'] for s in states if s['event_kind']==kind]),
            advantage_rs=_mean_std([s['advantage_rs'] for s in states if s['event_kind']==kind]))
            for kind in ('switch', 'best_changed_stay')},
        switching_frequency_hz=physical['switching_frequency_hz'],
        switch_fraction_per_ranking=physical['switch_fraction_per_ranking'],
        switches_per_successful_trial=physical['switches_per_rollout'],
        physical_switching_measured_trials=physical['measured_rollouts'],
        ranking_strategy_switching_frequency_hz=(
            switches/total_duration if total_duration else None),
        best_contact_point_world_changes=sum(best_events) if best_events else None,
        best_contact_point_world_change_frequency_hz=(
            sum(best_events)/total_duration if best_events and total_duration else None),
        value_original=_mean_std(original_values),
        value_perturbed=_mean_std(perturbed_values),
        value_delta=_mean_std([b-a for a, b in zip(original_values, perturbed_values)]),
        rs_decision_trigger_frequency_hz=switches/(steps*policy_dt) if steps else None,
        rs_decision_trigger_fraction_per_ranking=(switches/sum(t['rs_evaluations'] for t in successful)
            if sum(t['rs_evaluations'] for t in successful) else None))
