"""Outcome-based tests of the TRO Reviewer 5 Q1 evaluation metric."""

import json
import subprocess
import sys

import numpy as np
import pytest

from utils.ranking_metrics import (
    discounted_terminal_reward,
    latex_table,
    relative_advantage,
    summarize_paired_rollouts,
)


def pair(state_id, rs, cso, near):
    return {"state_id": state_id, "rollouts": {
        strategy: {"success": outcome[0], "steps": outcome[1]}
        for strategy, outcome in zip(("RS", "CSO", "near"), (rs, cso, near))
    }}


def test_terminal_outcome_and_elapsed_control_steps():
    assert discounted_terminal_reward(True, 0, eta=2) == 2
    assert discounted_terminal_reward(True, 10, eta=2) == pytest.approx(2 * 0.9**10)
    assert discounted_terminal_reward(False) == 0
    assert discounted_terminal_reward(False, 2500) == 0
    np.testing.assert_allclose(
        discounted_terminal_reward([True, False, True], [0, 2500, 10]),
        [1, 0, 0.9**10],
    )


def test_advantages_reward_success_and_faster_completion():
    early = discounted_terminal_reward(True, 5)
    late = discounted_terminal_reward(True, 10)
    assert relative_advantage(early, late) > 0
    assert relative_advantage(late, early) < 0
    assert relative_advantage(early, early) == 0
    assert relative_advantage(0, 0) == 0
    assert relative_advantage(early, 0, epsilon=0.01) == pytest.approx(early / 0.01)
    assert relative_advantage(0, early) == pytest.approx(-1)


def test_pair_before_average_and_include_double_failure_ties():
    records = [
        pair("a", (True, 2), (True, 4), (False, None)),
        pair("b", (False, None), (False, None), (False, None)),
        pair("c", (False, None), (True, 3), (True, 3)),
    ]
    summary = summarize_paired_rollouts(records, epsilon=0.1, switch_counts=[0, 2])
    expected = np.array([(0.9**2 - 0.9**4)/(0.9**4 + 0.1), 0, -0.9**3/(0.9**3 + 0.1)])
    stats = summary["comparisons"]["CSO"]
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(expected.mean())
    assert stats["std"] == pytest.approx(expected.std(ddof=1))
    assert stats["positive_fraction"] == pytest.approx(1/3)
    assert summary["switches_per_trial"] == {"n": 2, "mean": 1, "std": pytest.approx(2**0.5)}


@pytest.mark.parametrize("kwargs", [
    {"gamma": 0}, {"gamma": 1.1}, {"gamma": float("nan")},
    {"eta": 0}, {"eta": float("inf")},
])
def test_reject_invalid_reward_parameters(kwargs):
    with pytest.raises(ValueError):
        discounted_terminal_reward(False, **kwargs)


@pytest.mark.parametrize("success,steps", [
    (True, None), (True, -1), (True, 1.5), (True, float("nan")), ("false", 1), (2, 1),
])
def test_reject_invalid_outcomes(success, steps):
    with pytest.raises(ValueError):
        discounted_terminal_reward(success, steps)


def test_reject_unpaired_or_duplicate_states_and_invalid_switch_counts():
    record = pair("a", (True, 2), (True, 3), (False, None))
    with pytest.raises(ValueError, match="duplicate"):
        summarize_paired_rollouts([record, record])
    del record["rollouts"]["near"]
    with pytest.raises(KeyError):
        summarize_paired_rollouts([record])
    with pytest.raises(ValueError):
        summarize_paired_rollouts([], switch_counts=[1.5])
    for epsilon in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            summarize_paired_rollouts([], epsilon=epsilon)


def test_absent_data_is_pending_and_singleton_std_is_undefined():
    empty = summarize_paired_rollouts([])
    assert empty["comparisons"]["CSO"]["mean"] is None
    assert empty["switches_per_trial"] is None
    assert r"\textit{Pending}" in latex_table(empty)
    one = summarize_paired_rollouts([pair("a", (True, 2), (True, 3), (False, None))])
    assert one["comparisons"]["CSO"]["std"] is None
    assert r"\pm --" in latex_table(one)


def test_cli_generates_summary_and_table(tmp_path):
    data = tmp_path / "outcomes.json"
    table = tmp_path / "table.tex"
    data.write_text(json.dumps({"paired_rollouts": [
        pair("a", (True, 2), (True, 3), (False, None)),
        pair("b", (True, 4), (True, 5), (True, 6)),
    ], "switch_counts": [0, 2]}))
    result = subprocess.run(
        [sys.executable, "-m", "utils.ranking_metrics", str(data), "--latex", str(table)],
        capture_output=True, text=True, check=True,
    )
    summary = json.loads(result.stdout)
    assert summary["n_states"] == 2
    assert summary["comparisons"]["CSO"]["positive_fraction"] == 1
    assert "100.0\\%" in table.read_text()
    assert r"\textit{Pending}" not in table.read_text()
