"""Binary decision regret, event sampling, and matched-prefix orchestration."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from utils.ranking_cdr import (
    DecisionSampler, assert_matched, contact_decision_context, continuation_outcome,
    evaluate_state, select_events, summarize_cdr,
    select_decision_points, source_decision_statistics, table_vi_statistics,
)
from examples.mpc.franka.experiments.ik2.test_rs import MatchedRolloutBus, run


def context(best=.1, decision='stay', eligible=True):
    return dict(decision=decision, eligible=eligible, near_idx=0, optimal_idx=1,
                guide_idx=1 if decision == 'switch' else 0, switch_idx=1,
                near_local=[0, 0, 0], optimal_local=[best, 0, 0],
                switch_local=[best, 0, 0], stage='pose', policy_q=[0]*10)


def feed(sampler, step, **kw):
    sampler.step, sampler.sim_time = step, .02*step
    sampler.in_physical_contact = True
    return sampler(context(**kw))


def test_event_sampling_deduplicates_switch_and_requires_best_change_and_near():
    s = DecisionSampler()
    feed(s, 0)  # Initial ranking is not a change.
    feed(s, 1, best=.105)  # mesh jitter
    feed(s, 2, best=.2)  # accepted nearby after a genuine best change
    feed(s, 3, best=.3, eligible=False)  # traveling, no binary contact decision
    feed(s, 4, best=.3, decision='switch')
    feed(s, 5, best=.3, decision='switch')  # same destination command
    feed(s, 6, best=.302, decision='switch')
    feed(s, 7, best=.4, decision='switch')
    assert [(e['step'], e['event_kind']) for e in s.events] == [
        (2, 'best_changed_stay'), (4, 'switch'), (7, 'switch')]
    assert s.switch_count == 2
    assert s.evaluations == 8
    assert len(select_events(s.events, max_per_kind=1)) == 2
    assert select_events(s.events, max_per_kind=1, seed=8) == select_events(s.events, max_per_kind=1, seed=8)
    assert select_events(s.events, max_per_kind=0) == s.events


def test_context_uses_committed_target_not_raw_best_or_sample_index_change():
    opt = SimpleNamespace(sample_point=np.array([[0,0,0], [.01,0,0], [.2,0,0]]),
                          last_global_idx=2, last_candidate_ids=np.array([0,1,2]),
                          last_candidate_costs=np.array([1., .1, 0.]))
    kw = dict(near_arm=True, observed=False, switch_observed=False, stage='pose')
    # Raw optimum changes, but the nearby contact is deliberately retained.
    c = contact_decision_context(opt, [0]*10, 0, 0, accept=True, hold=True, **kw)
    assert c['decision'] == 'stay' and c['eligible'] and c['switch_idx'] == 2
    # Reject a nearby-index change as a switching action.
    c = contact_decision_context(opt, [0]*10, 0, 1, accept=False, hold=False, **kw)
    assert c['decision'] == 'stay' and not c['eligible']
    c = contact_decision_context(opt, [0]*10, 0, 2, accept=False, hold=False, **kw)
    assert c['decision'] == 'switch' and c['eligible']
    opt.last_candidate_costs[2] = np.inf
    assert not contact_decision_context(opt, [0]*10, 0, 2, accept=False, hold=False, **kw)['eligible']


def test_q_average_before_max_and_cdr_and_signed_advantage():
    e = dict(context(decision='stay'), event_kind='best_changed_stay')
    rollouts = {'stay': [dict(success=True, steps=2), dict(success=False, steps=3)],
                'switch': [dict(success=True, steps=3), dict(success=True, steps=3)]}
    result = evaluate_state(e, rollouts, gamma=.9)
    assert result['q']['stay'] == pytest.approx(.9**2/2)
    assert result['q']['switch'] == pytest.approx(.9**3)
    assert result['cdr'] == pytest.approx(.9**3 - .9**2/2)
    assert result['advantage_rs'] == pytest.approx(-result['cdr'])
    assert result['advantage_switch'] == pytest.approx(result['cdr'])
    both_fail = {b: [dict(success=False, steps=100)] for b in rollouts}
    assert evaluate_state(e, both_fail, gamma=.9)['cdr'] == 0
    better = dict(e, decision='switch')
    assert evaluate_state(better, rollouts, gamma=.9)['cdr'] == 0


def test_remaining_control_steps_and_aborted_are_not_failure():
    o = continuation_outcome(dict(success=True, steps=25), 20, gamma=.9)
    assert o['steps'] == 5 and o['reward'] == pytest.approx(.9**5)
    assert o['completion_time_s'] == .1
    assert continuation_outcome(dict(success=False, steps=25), 20, gamma=.9)['reward'] == 0
    with pytest.raises(RuntimeError, match='Aborted'):
        continuation_outcome(dict(success=False, steps=25, aborted=True), 20, gamma=.9)
    with pytest.raises(RuntimeError, match='before'):
        continuation_outcome(dict(success=True, steps=19), 20, gamma=.9)
    with pytest.raises(ValueError):
        evaluate_state(context(), {'stay': [], 'switch': []}, gamma=.9)


def test_one_time_override_and_prefix_decision_rejection():
    original = DecisionSampler()
    for i, best in enumerate((.1, .2, .2)):
        feed(original, i, best=best)
    replay = DecisionSampler(target=1, expected=original.decisions, forced='switch')
    assert feed(replay, 0) is None
    assert feed(replay, 1, best=.2) == 'switch'
    assert replay.applied
    assert feed(replay, 2, best=.2) is None
    mismatched = DecisionSampler(target=1, expected=original.decisions, forced='switch')
    with pytest.raises(RuntimeError, match='mismatch'):
        feed(mismatched, 0, best=.4)
    with pytest.raises(RuntimeError):
        assert_matched({'q': [1.]}, {'q': [1.01]}, atol=1e-7)
    # Saved prefixes are lists of dictionaries containing NumPy arrays.
    assert_matched([{'q': np.array([1., 2.])}], [{'q': [1., 2.]}], atol=1e-7)
    with pytest.raises(RuntimeError, match='mismatch'):
        assert_matched([{'q': np.array([1., 2.])}], [{'q': [1., 3.]}], atol=1e-7)


class FakePhysics:
    def __init__(self, changed=False):
        self.i, self.changed = 0, changed
        self.commands = []

    def wait_obs(self, timeout=180.):
        return dict(seq=self.i, sim_time=self.i*.02, policy_q=[float(self.changed)]+[0.]*9,
                    object_linear_velocity=np.zeros(3), contact_distance=float('inf'))

    def publish_cmd(self, command):
        self.commands.append(command)
        self.i += 1

    def close(self):
        pass


def test_bus_checks_observations_and_predecision_actions_but_allows_branch_action():
    s = DecisionSampler()
    original = MatchedRolloutBus(FakePhysics(), s)
    original.wait_obs()
    original.publish_cmd({'action': [0,0,0], 'submitted_at': 1.})
    original.wait_obs()
    replay = MatchedRolloutBus(FakePhysics(), DecisionSampler(target=1), reference=original)
    replay.wait_obs()
    replay.publish_cmd({'action': [0,0,0], 'submitted_at': 999.})
    replay.wait_obs()
    replay.publish_cmd({'action': [1,0,0]})  # intervention at target is permitted
    bad = MatchedRolloutBus(FakePhysics(changed=True), DecisionSampler(target=1), reference=original)
    with pytest.raises(RuntimeError, match='mismatch'):
        bad.wait_obs()
    bad_action = MatchedRolloutBus(FakePhysics(), DecisionSampler(target=1), reference=original)
    bad_action.wait_obs()
    with pytest.raises(RuntimeError, match='mismatch'):
        bad_action.publish_cmd({'action': [1,0,0]})


def args(tmp_path):
    return SimpleNamespace(ranking_metric_gamma=.9, trial_start=0, trial_num=2,
        rs_samples_per_trajectory=1, rs_successful_trajectories=1,
        rs_sampling='decisions', rs_event_stage='all',
        rs_repeats=2, rs_max_per_kind=0, rs_best_change_distance=.01,
        rs_match_atol=1e-7, num_envs=1, async_planner=False, floating_fingertip=False,
        sampling_c3=False, online_scm=False, scm_record_data=False, cso_guidance='cso',
        cpo_variant='full', rs_output=str(tmp_path), rs_sample_seed=0,
        max_rollout_length=10)


def test_orchestration_samples_only_successful_sources_and_restores_normal_policy(tmp_path):
    calls = []
    def runner(args, trial, sampler, reference=None):
        calls.append((trial, sampler.target, sampler.forced, reference is not None))
        for i, best in enumerate((.1, .2, .2)):
            feed(sampler, i, best=best)
        success = trial == 0 and sampler.forced != 'switch'
        return SimpleNamespace(summaries=[dict(success=success, steps=3 if success else 10)],
                               observations=[], commands=[])
    result = run(args(tmp_path), trial_runner=runner)
    state = json.loads((tmp_path/'states.jsonl').read_text())
    assert state['step'] == 2 and state['remaining_budget'] == 8
    assert state['q']['stay'] == pytest.approx(.9)
    assert state['q']['switch'] == 0 and state['cdr'] == 0
    assert state['advantage_rs'] == pytest.approx(.9)
    assert all(len(x) == 2 for x in state['rollouts'].values())
    # Uniform decision sampling can include stable near selections, too.
    assert calls == [(0,None,None,False), (0,2,'switch',True),
                     (0,2,None,True), (0,2,'switch',True)]
    assert result['successful_source_trials'] == 1
    assert result['n_states'] == 1
    assert json.loads((tmp_path/'summary.json').read_text()) == result


def test_empty_measurements_remain_pending_and_frequency_uses_sources_not_samples():
    summary = summarize_cdr([], [], gamma=.9)
    assert summary['cdr']['mean'] is None and summary['switching_frequency_hz'] is None
    trials = [dict(success=True, steps=100, rs_switch_events=20, mpc_contact_switches=2, rs_evaluations=100, policy_interval_s=.02),
              dict(success=False, steps=100, rs_switch_events=20, rs_evaluations=100)]
    summary = summarize_cdr([], trials, gamma=.9)
    assert summary['switching_frequency_hz'] == 1.
    assert summary['switch_fraction_per_ranking'] == .02


def test_resume_reuses_pairs_after_verifying_configuration_and_source(tmp_path):
    calls = []
    def runner(args, trial, sampler, reference=None):
        calls.append((trial, sampler.forced))
        for i, best in enumerate((.1, .2, .2)):
            feed(sampler, i, best=best)
        success = trial == 0 and sampler.forced != 'switch'
        return SimpleNamespace(summaries=[dict(success=success, steps=3 if success else 10)],
                               observations=[], commands=[])
    values = args(tmp_path)
    first = run(values, trial_runner=runner)
    original_states = (tmp_path/'states.jsonl').read_text()
    calls.clear()
    values.rs_resume = True
    assert run(values, trial_runner=runner) == first
    assert calls == [(0, None)]
    assert (tmp_path/'states.jsonl').read_text() == original_states
    values.ranking_metric_gamma = .99
    with pytest.raises(ValueError, match='configuration differs'):
        run(values, trial_runner=runner)
    assert (tmp_path/'states.jsonl').read_text() == original_states


def test_uniform_fifty_points_include_stable_decisions_and_require_full_sample():
    sampler = DecisionSampler()
    for step in range(120):
        feed(sampler, step, eligible=step >= 10)
    assert sampler.events == []
    points = select_decision_points(sampler.decisions, count=50, seed=7)
    assert len(points) == len({p['step'] for p in points}) == 50
    assert points == select_decision_points(sampler.decisions, count=50, seed=7)
    assert min(p['step'] for p in points) >= 10
    with pytest.raises(ValueError, match='found 110'):
        select_decision_points(sampler.decisions, count=111)


def test_contact_sampling_excludes_detached_nearby_states_and_transit():
    decisions = [dict(context(), step=i, in_physical_contact=i >= 20,
                      in_transit=20 <= i < 40) for i in range(60)]
    points = select_decision_points(decisions, count=5, seed=3)
    assert len(points) == 5
    assert all(p['step'] >= 40 for p in points)
    with pytest.raises(ValueError, match='found 20'):
        select_decision_points(decisions, count=21)


def test_seven_table_metrics_use_sources_and_keep_zero_advantage():
    decisions = [dict(context(decision=branch), step=i,
                      optimal_local=[best, 0, 0], optimal_world=[i, 0, 0])
                 for i, (branch, best) in enumerate([
                     ('stay', .1), ('stay', .1), ('switch', .2), ('stay', .2)])]
    source = dict(success=True, steps=4, policy_interval_s=.02, mpc_contact_switches=2)
    source.update(source_decision_statistics(decisions, source))
    excluded = dict(source, rs_source_selected=False, mpc_contact_switches=100)
    states = [dict(advantage_rs=a) for a in [0, .2, -.1, .3]]
    result = table_vi_statistics(states, [source, excluded])
    assert result['contact_switch_count'] == 2
    assert result['p_near'] == .75 and result['p_optimal'] == .25
    assert result['f_near_hz'] == 37.5
    assert result['f_p_star_hz'] == 12.5  # Ignore rigid world-frame motion.
    assert result['p_advantage_positive'] == .5
    assert result['advantage']['mean'] == pytest.approx(.1)
    assert result['advantage']['std'] == pytest.approx(np.std([0,.2,-.1,.3], ddof=1))


def test_failed_sources_are_skipped_before_selecting_one_success(tmp_path):
    seen = []
    def runner(args, trial, sampler, reference=None):
        seen.append((trial, sampler.target))
        for step in range(55):
            feed(sampler, step)
        return SimpleNamespace(summaries=[dict(success=trial == 1, steps=60)],
                               observations=[], commands=[])
    values = args(tmp_path)
    values.rs_samples_per_trajectory = 50
    values.rs_repeats = 1
    result = run(values, trial_runner=runner)
    assert seen[:2] == [(0, None), (1, None)]
    assert len(seen) == 52
    assert result['n_states'] == 50
    assert result['table_vi']['source_trajectories'] == 1
