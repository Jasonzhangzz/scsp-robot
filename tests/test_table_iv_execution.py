"""Execution SD must be reproducible, success-only, and labeled when a proxy."""
import copy
import math

import pytest

from tro_revision import recover_table_iv_execution as execution


def test_execution_std_uses_successful_samples_and_ddof_one():
    rows=[dict(trajectory_id=str(i),success=success,execution_time_s=value)
          for i,(success,value) in enumerate(((True,1.),(True,3.),(False,100.)))]
    result=execution.summary(rows)
    assert result['mean_s']==2
    assert result['sample_std_s']==pytest.approx(math.sqrt(2))
    assert result['successes']==2 and result['trials']==3
    assert execution.summary(rows[:1])['sample_std_s'] is None
    assert execution.summary(rows[-1:])['sample_std_s'] is None


def test_proxy_groups_are_disjoint_and_do_not_select_by_time():
    pool=[dict(trajectory_id='observed/%d'%i,success=i<100,execution_time_s=float(i+1))
          for i in range(400)]
    targets={method:dict(target_success_rate=rate)
             for method,rate in (('DyWA',.86),('Sampling',.56),('CF-MPC',.01))}
    original=copy.deepcopy(pool)
    groups=execution.proxy_groups(pool,'Rotation',targets)
    assert pool==original
    ids=[r['trajectory_id'] for g in groups.values() for r in g['selected_trajectories']]
    assert len(ids)==len(set(ids))
    assert groups['DyWA']['successes']==30 and groups['DyWA']['trials']==35
    assert groups['Sampling']['successes']==20 and groups['Sampling']['trials']==35
    assert groups['CF-MPC']['successes']==2 and groups['CF-MPC']['trials']==200
    scaled=[dict(r,execution_time_s=r['execution_time_s']*2) for r in pool]
    other=execution.proxy_groups(scaled,'Rotation',targets)
    for method in groups:
        a,b=groups[method],other[method]
        assert [r['trajectory_id'] for r in a['selected_trajectories']]==[
            r['trajectory_id'] for r in b['selected_trajectories']]
        assert b['sample_std_s']==pytest.approx(2*a['sample_std_s'])
        assert a['status']=='proxy_dispersion_not_method_measurements'


def test_proxy_sd_is_not_formatted_as_a_measured_mean_sd_pair():
    result=execution.format_cell(r'\textbf{15.16}',15.16,11.82,proxy=True)
    assert '15.16' in result and '[11.82]' in result and r'\dagger' in result
    assert r'\pm' not in result
    assert execution.format_cell('--',None,None)=='--'
    assert execution.format_cell('35.00',35.,47.78)==r'$35.00\pm47.78$'
