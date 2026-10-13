"""Physical relocations versus planning jitter, sliding and failed travel."""
import numpy as np
import pytest

from utils.executed_contact_switches import completed_contact_switches, execution_trace


A, B = [0.,0.,0.], [.1,0.,0.]


def trace(spec):
    rows = []
    for i, (point, target) in enumerate(spec):
        tip = np.array(point if point is not None else [0.,0.,.1]) + [0,0,.01]
        rows.append(dict(step=i,sim_time=.02*i,in_contact=point is not None,
            contact_point_local=point,tip_local=tip.tolist(),
            executed_target_local=None if target is None else (np.array(target)+[0,0,.01]).tolist(),
            best_contact_idx=i%2,selected_idx=i%3))
    return rows


def test_only_completed_mpc_relocation_counts_not_initial_contact_or_repeated_targets():
    rows = trace([(A,A)]*3+[(A,B)]+[(None,B)]*2+[(B,B)]*8)
    result = completed_contact_switches(rows)
    assert result['mpc_contact_switches'] == 1
    assert result['events'][0]['departure_step'] == 4
    assert result['events'][0]['arrival_step'] == 6
    assert result['events'][0]['confirmed_step'] == 8
    assert result['events'][0]['from_local'] == A
    assert result['events'][0]['to_local'] == B


def test_optimizer_jitter_and_target_only_changes_do_not_count():
    # Best/selected IDs change every frame and the command repeatedly changes
    # destination, but the finger never leaves its physically occupied patch.
    rows = trace([(A, A if i%2 else B) for i in range(100)])
    assert completed_contact_switches(rows)['mpc_contact_switches'] == 0


def test_sliding_chatter_same_patch_recontact_and_failed_travel_do_not_count():
    sliding = trace([([.005*i,0,0],[.005*i,0,0]) for i in range(40)])
    assert completed_contact_switches(sliding)['mpc_contact_switches'] == 0
    same_patch = trace([(A,A)]*3+[(A,B)]+[(None,B)]*3+[(A,A)]*5)
    assert completed_contact_switches(same_patch)['mpc_contact_switches'] == 0
    chatter = trace([(A,A)]*3+[(A,B),(None,B)]+[(B,B)]*5)
    assert completed_contact_switches(chatter)['mpc_contact_switches'] == 0
    failed = trace([(A,A)]*3+[(A,B)]+[(None,B)]*8)
    assert completed_contact_switches(failed)['mpc_contact_switches'] == 0
    unconfirmed = trace([(A,A)]*3+[(A,B)]+[(None,B)]*2+[(B,B)]*2)
    assert completed_contact_switches(unconfirmed)['mpc_contact_switches'] == 0


def test_passive_contact_change_without_executed_far_target_is_not_mpc_switch():
    rows = trace([(A,None)]*3+[(None,None)]*3+[(B,None)]*5)
    assert completed_contact_switches(rows)['mpc_contact_switches'] == 0


def test_missing_contact_geometry_or_incomplete_trace_is_not_measured_zero():
    with pytest.raises(ValueError,match='Missing'):
        completed_contact_switches([])
    rows = trace([(A,A)]*4)
    rows[3]['contact_point_local'] = None
    with pytest.raises(ValueError,match='geometry'):
        completed_contact_switches(rows)
    rows = trace([(A,A)]*4)
    rows[3]['step'] = 5
    with pytest.raises(ValueError,match='every control'):
        completed_contact_switches(rows)


def test_execution_cache_uses_previous_executed_command_and_object_local_geometry():
    q0 = np.r_[[1.,2.,3.],[1.,0.,0.,0.],[1.,2.,3.01]]
    q1 = np.r_[[5.,6.,7.],[2**-.5,0.,0.,2**-.5],[5.,6.1,7.01]]
    observations = [dict(seq=0,sim_time=0.,policy_q=q0,if_contact=True,
        actual_contact=dict(point_world=[1.,2.,3.],dist=0.)),
        dict(seq=1,sim_time=.02,policy_q=q1,if_contact=True,
        actual_contact=dict(point_world=[5.,6.1,7.],dist=0.))]
    commands = [dict(seq=0,press=[1.1,2.,3.01]),dict(seq=1,press=[99.,99.,99.])]
    rows = execution_trace(observations,commands)
    assert rows[0]['executed_target_local'] is None
    np.testing.assert_allclose(rows[1]['contact_point_local'],B,atol=1e-12)
    np.testing.assert_allclose(rows[1]['executed_target_local'],[.1,0,.01],atol=1e-12)
    assert rows[1]['executed_command_step'] == 0
