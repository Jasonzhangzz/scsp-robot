import json
from types import SimpleNamespace

import numpy as np
import pytest

from utils.ranking_cdr import DecisionSampler
from utils.rollout_cache import best_point_changes, save_source_cache, load_source_cache


def test_best_changes_separate_raw_cso_and_guided_point_from_rs_triggers():
    rows=[dict(step=i,optimal_local=p,best_contact_local=[0,0,0])
          for i,p in enumerate(([0,0,0],[.1,0,0],[.1,0,0],[0,0,0]))]
    result=best_point_changes(rows,dict(steps=4,completion_time_s=2,rs_switch_events=99))
    assert result['best_contact_switch_events']==2
    assert result['best_contact_switch_frequency_hz']==1
    assert result['guided_contact_switch_events']==0
    assert best_point_changes(rows[:-1],dict(steps=4))['best_contact_switch_events'] is None


def test_best_changes_prefer_world_coordinates_for_rotating_object():
    rows = [
        dict(step=0, optimal_local=[1, 0, 0], best_contact_local=[1, 0, 0],
             optimal_world=[0, 1, 0], best_contact_world=[0, 1, 0]),
        dict(step=1, optimal_local=[1, 0, 0], best_contact_local=[1, 0, 0],
             optimal_world=[0, 1, 0], best_contact_world=[0, 0, 1]),
    ]
    result = best_point_changes(rows, dict(steps=2, completion_time_s=.2))
    assert result['best_contact_switch_events'] == 0
    assert result['guided_contact_switch_events'] == 1
    assert result['guided_contact_switch_frequency_hz'] == 5


def test_complete_source_cache_roundtrip_and_configuration_guard(tmp_path):
    sampler=DecisionSampler()
    sampler.decisions=[dict(step=i,optimal_local=[.1*i,0,0],best_contact_local=[0,0,0]) for i in range(3)]
    sampler.evaluations=3
    observations=[dict(seq=i,sim_time=.02*i,policy_q=np.r_[np.zeros(3),[1,0,0,0],[0,0,.1]],
                       if_contact=False,actual_contact=None,contact_distance=np.inf) for i in range(4)]
    commands=[dict(seq=i,press=np.array([.1,0,0])) for i in range(3)]
    summary=dict(trial=1,seed=1,object='foam_brick',steps=3,success=True,policy_interval_s=.02)
    bus=SimpleNamespace(summaries=[summary],observations=observations,commands=commands)
    args=SimpleNamespace(obj='foam_brick',target_type='ground-rotation',async_planner=False,
                         solver_backend='acados',sample_num=70,rollout_cache_dir=str(tmp_path))
    path=save_source_cache(tmp_path,bus,sampler,args)
    loaded,restored=load_source_cache(path,args,1)
    assert len(loaded.observations)==4 and len(loaded.commands)==3
    np.testing.assert_array_equal(loaded.observations[-1]['policy_q'],observations[-1]['policy_q'])
    assert loaded.summaries[0]['best_contact_switch_events']==2
    assert loaded.summaries[0]['mpc_contact_switches']==0
    assert restored.evaluations==3 and restored.decisions==sampler.decisions
    args.sample_num=10
    with pytest.raises(ValueError,match='sample_num'):
        load_source_cache(path,args,1)
    with pytest.raises(ValueError,match='seed'):
        load_source_cache(path,args,2)
