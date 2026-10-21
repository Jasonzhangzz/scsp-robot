"""Mean scaling preserves the observed CV without inventing 200 observations."""
import copy
import json
from pathlib import Path
import statistics

import pytest

from tro_revision import scale_table_iv_execution as scaling


def test_estimated_sd_matches_rescaling_the_successful_observations():
    rows = [dict(success=True, execution_time_s=v) for v in (1., 2., 9.)]
    rows.append(dict(success=False, execution_time_s=10000.))
    original = copy.deepcopy(rows)
    value = scaling.estimate(rows, 20.)
    scaled = [r['execution_time_s']*value['scale_factor'] for r in rows if r['success']]
    assert statistics.mean(scaled) == pytest.approx(20.)
    assert statistics.stdev(scaled) == pytest.approx(value['estimated_std_s'])
    assert value['estimated_std_s']/20 == pytest.approx(value['coefficient_of_variation'])
    assert value['source_successes'] == 3
    assert rows == original
    with pytest.raises(ValueError):
        scaling.estimate([dict(success=True, execution_time_s=0.)]*2, 20.)


def test_cached_proxy_update_preserves_means_and_measured_cells():
    paper = scaling.PAPER/'experiment.tex'
    text = paper.read_text()
    source = json.loads(scaling.SOURCE.read_text())
    updated, estimates = scaling.update(text, source)
    assert scaling.update(updated, source)[0] == updated
    before = scaling.parse_paper(text)
    after = scaling.parse_paper(updated)
    for task, methods in before.items():
        for method, value in methods.items():
            assert after[task][method]['retained_mean_s'] == value['retained_mean_s']
            if method not in scaling.PROXY_METHODS:
                assert after[task][method]['original_latex'] == value['original_latex']
        for estimate in estimates[task].values():
            assert estimate['target_total_trials'] == 200
            assert estimate['status'] == 'mean_scaled_proxy_estimate_not_observed_200_trial_sd'
    sample = estimates['Rotation']['Sampling']
    assert sample['source_successes'] == 20
    assert sample['estimated_std_s'] == pytest.approx(43.9107, abs=.001)
    assert r'^{\dagger}' in after['Rotation']['Sampling']['original_latex']
