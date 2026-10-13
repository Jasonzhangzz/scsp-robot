"""Offline count/time denominators, source deduplication and pending coverage."""
import json
import subprocess
import sys

import pytest

from utils.switching_frequency import load_cache, read_records, summarize


def row(seed, switches, steps, success=True, **kw):
    return dict(object='foam_brick', seed=seed, success=success, steps=steps,
                mpc_contact_switches=switches, rs_evaluations=steps,
                policy_interval_s=.02, target_type='ground-rotation', **kw)


def test_frequency_uses_original_rollouts_and_pooled_simulated_time():
    # Includes the zero-switch trajectory; failure/aborted trials do not bias
    # the default successful-only result. Rate means and pooled rates differ.
    rows = [row(0, 0, 100), row(1, 10, 50), row(2, 99, 100, False),
            row(3, 99, 100, aborted=True)]
    stats = summarize(rows)['overall']
    assert stats['measured_rollouts'] == 2
    assert stats['total_switches'] == 10
    assert stats['switches_per_rollout']['mean'] == 5
    assert stats['switches_per_rollout']['std'] == pytest.approx(50**.5)
    assert stats['switching_frequency_hz'] == pytest.approx(10/3)
    assert stats['per_rollout_frequency_hz']['mean'] == 5
    assert stats['switch_fraction_per_ranking'] == pytest.approx(10/150)
    assert summarize(rows, successful_only=False)['overall']['total_switches'] == 109
    assert 'flipping' in summarize(rows)['by_task']


def test_missing_counts_are_pending_not_zero_and_objects_show_coverage():
    r = row(0, 0, 10)
    del r['mpc_contact_switches']
    cases = [dict(object='foam_brick', seed=0), dict(object='mug', seed=1)]
    report = summarize([r], expected_cases=cases)
    assert report['overall']['total_switches'] is None
    assert report['overall']['switches_per_rollout']['mean'] is None
    assert report['overall']['missing_switch_count_rollouts'] == 1
    assert report['overall']['unavailable_rollouts'] == 1
    assert report['by_object']['mug']['measured_rollouts'] == 0
    assert report['by_object']['mug']['unavailable_rollouts'] == 1


def test_async_uses_cached_simulation_duration_not_number_of_planner_cycles():
    r = row(0, 10, 50, async_planner=True)
    r['completion_time_s'] = 10.
    assert summarize([r])['overall']['switching_frequency_hz'] == 1.
    del r['completion_time_s']
    stats = summarize([r])['overall']
    assert stats['switches_per_rollout']['mean'] == 10
    assert stats['switching_frequency_hz'] is None


def test_manifest_excludes_archives_and_cdr_samples(tmp_path):
    (tmp_path/'cases.json').write_text(json.dumps([dict(object='foam_brick',seed=0)]))
    case = tmp_path/'foam_brick_seed0'
    case.mkdir()
    (case/'trials.jsonl').write_text(json.dumps(row(0,2,10))+'\n')
    (case/'states.jsonl').write_text('not a source rollout\n')
    archived = tmp_path/'archive'/'foam_brick_seed0'
    archived.mkdir(parents=True)
    (archived/'trials.jsonl').write_text(json.dumps(row(0,200,1000))+'\n')
    records, cases, files = load_cache(tmp_path)
    assert len(records) == 1 and records[0]['mpc_contact_switches'] == 2
    assert len(cases) == 1 and len(files) == 1
    with pytest.raises(ValueError, match='Duplicate'):
        summarize(records+records)


def test_concurrent_partial_tail_is_ignored_but_corrupt_complete_line_is_error(tmp_path):
    file = tmp_path/'trials.jsonl'
    file.write_text(json.dumps(row(0,1,10))+'\n{"unfinished":')
    assert len(read_records(file)) == 1
    file.write_text('{"bad":\n')
    with pytest.raises(json.JSONDecodeError):
        read_records(file)


def test_cli_emits_json_and_csv_without_simulator(tmp_path):
    (tmp_path/'trials.jsonl').write_text(json.dumps(row(0,2,20))+'\n')
    subprocess.run([sys.executable,'-m','utils.switching_frequency',str(tmp_path)],
                   check=True,capture_output=True,text=True)
    report = json.loads((tmp_path/'switching_frequency.json').read_text())
    assert report['overall']['switches_per_rollout']['mean'] == 2
    assert report['overall']['switching_frequency_hz'] == 5
    assert (tmp_path/'switching_frequency.csv').read_text().count('\n') == 4


@pytest.mark.parametrize('count', [-1, 1.5, float('nan'), '2'])
def test_invalid_cached_counts_are_rejected(count):
    with pytest.raises(ValueError):
        summarize([row(0,count,100)])


def test_rs_trigger_counts_are_never_substituted_for_physical_switches():
    r = row(0,2,100)
    del r['mpc_contact_switches']
    r['rs_switch_events'] = 101
    report = summarize([r])
    assert report['overall']['measured_rollouts'] == 0
    assert report['overall']['switching_frequency_hz'] is None
    assert report['overall']['total_switches'] is None


def test_offline_execution_trace_derives_completed_switch_count(tmp_path):
    r = row(0,2,8)
    del r['mpc_contact_switches']
    r['trial'] = 0
    (tmp_path/'trials.jsonl').write_text(json.dumps(r)+'\n')
    specs = [([0,0,0],[0,0,.01])]*3 + [([0,0,0],[.1,0,.01])] + \
        [(None,[.1,0,.01])]*2 + [([.1,0,0],[.1,0,.01])]*3
    trace = []
    for i,(point,target) in enumerate(specs):
        trace.append(dict(step=i,sim_time=.02*i,in_contact=point is not None,
            contact_point_local=point,tip_local=[.1,0,.01] if i>=6 else [0,0,.01],
            executed_target_local=target))
    file = tmp_path/'execution_contact_trace_trial0.jsonl'
    file.write_text(''.join(json.dumps(x)+'\n' for x in trace))
    rows,_,_ = load_cache(tmp_path)
    assert rows[0]['mpc_contact_switches'] == 1
    assert summarize(rows)['overall']['switches_per_rollout']['mean'] == 1
    file.write_text(''.join(json.dumps(x)+'\n' for x in trace[:-1]))
    rows,_,_ = load_cache(tmp_path)
    assert rows[0].get('mpc_contact_switches') is None
    assert summarize(rows)['overall']['switching_frequency_hz'] is None
