import numpy as np
import json
from pathlib import Path

from planning.rollout_timing import distribution, trial_timing, aggregate_trials


def test_failed_trial_is_censored_not_a_successful_execution_time():
    failed=[dict(success=False,elapsed_wall_time_s=4.,simulation_time_s=2.,planning_time_ms=6.)]
    successful=[dict(success=True,elapsed_wall_time_s=3.,simulation_time_s=1.,planning_time_ms=2.)]
    f,s=trial_timing(failed),trial_timing(successful)
    assert f['execution_time_s'] is None
    assert f['elapsed_wall_time_s']==4.
    stats=aggregate_trials([dict(f,success=False),dict(s,success=True)],failed+successful)
    assert stats['successes']==1 and stats['total_trials']==2
    assert stats['execution_time_s_success_only']['mean']==3.
    assert stats['planning_time_ms_all_rounds']['mean']==4.
    assert stats['planning_time_ms_all_rounds']['n']==2


def test_empty_timings_have_no_invented_completion_time():
    stats=distribution([])
    assert stats['n']==0 and stats['mean'] is None
    assert trial_timing([])['execution_time_s'] is None
    np.testing.assert_allclose(distribution([1.,3.])['std'],1.)


def test_lcp_certifies_nearly_redundant_physical_contacts():
    from planning.point_contact_baselines import solve_physical_lcp
    p=json.loads((Path(__file__).parent/'fixtures/lcp_nearly_redundant_contacts.json').read_text())
    J,inv,free,gaps=[np.asarray(p[k]) for k in ('J','inverse','free','gaps')]
    v,residual=solve_physical_lcp(free,J,inv,gaps,p['h'])
    assert residual < 1e-8
    assert np.isfinite(v).all()
    # The rigid contact velocity inequalities must hold on the original scale.
    assert np.min(J@v+gaps/p['h']) > -1e-8
