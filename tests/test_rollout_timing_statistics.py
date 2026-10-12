import pytest

from examples.mpc.franka.ik2.summarize_rollouts import aggregate


def record(seed, times, success):
    return dict(seed=seed, success=success, steps=len(times), planning_wall_count=len(times),
                planning_wall_total_s=sum(times), planning_wall_sum_squares_s2=sum(t*t for t in times),
                planning_wall_mean_ms=sum(times)/len(times)*1000,
                completion_steps=len(times) if success else None, policy_interval_s=.02,
                completion_wall_time_s=1. if success else None)


def test_pooling_uses_cycles_and_success_timing_excludes_failures():
    rows = [record(0, [.001, .003], True), record(1, [.005], False)]
    result = aggregate(rows, range(2))
    assert result['planning_cycle_ms']['n'] == 3
    assert result['planning_cycle_ms']['mean'] == pytest.approx(3.)
    assert result['planning_cycle_ms']['std'] == pytest.approx(2.)
    assert result['completion_sim_s'] == dict(n=1, mean=.04, std=None)
    assert result['success_seeds'] == [0]
    assert result['success_rate'] == .5


def test_no_success_is_missing_time_not_zero_and_incomplete_data_fails():
    rows = [record(0, [.001, .003], False)]
    assert aggregate(rows, range(1))['completion_sim_s'] == dict(n=0, mean=None, std=None)
    for bad, expected in [(rows, range(2)), (rows+rows, range(2))]:
        with pytest.raises(ValueError):
            aggregate(bad, expected)
    rows[0]['aborted'] = True
    with pytest.raises(ValueError, match='Aborted'):
        aggregate(rows, range(1))


def test_async_completion_time_uses_observed_simulation_clock():
    row = record(0, [.001, .003], True)
    row['completion_time_s'] = .18  # Async skipped observations, two planner calls.
    row['async_planner'] = True
    assert aggregate([row], range(1))['completion_sim_s'] == dict(n=1, mean=.18, std=None)


def test_stable_flip_is_reported_separately_from_final_pose_success():
    rows = [record(0, [.001], False), record(1, [.003], True)]
    for row in rows:
        row.update(stable_flip_success=True, first_stable_flip_time_s=2.)
    result = aggregate(rows, range(2))
    assert result['stable_flip_rate'] == 1.
    assert result['stable_flip_seeds'] == [0, 1]
    assert result['success_rate'] == .5
    assert result['success_seeds'] == [1]
