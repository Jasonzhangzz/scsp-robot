import numpy as np
import json
import argparse
from pathlib import Path
import pytest

from planning.execution_constraints import floor_command_bounds, table_edge_command_bounds


def test_user_test_default_and_baseline_override(monkeypatch):
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    from examples.mpc.franka.runtime.mpc import _add_rollout_policy_args
    parser = _add_rollout_policy_args(argparse.ArgumentParser())
    assert parser.parse_args(['--obj', 'stanford_bunny2']).pose_executor_constraints
    assert parser.parse_args(['--obj', 'stanford_bunny2']).flip_executor_constraints
    assert not parser.parse_args(['--obj', 'stanford_bunny2', '--no-pose-executor-constraints']).pose_executor_constraints
    assert not parser.parse_args(['--obj', 'stanford_bunny2', '--no-flip-executor-constraints']).flip_executor_constraints


def test_first_command_obeys_executor_endpoint_floor_with_scalar_box():
    bounds, info = floor_command_bounds(.36115571856, .362, -.005, .005, max_norm=.005)
    assert info == dict(applied=True, rebased=False, emergency=False)
    assert abs(bounds[0][2] - .00084428144) < 1e-10
    assert np.linalg.norm(np.clip(np.zeros(3), *bounds)) <= .005


def test_floor_does_not_modify_a_feasible_slew_window():
    original = (np.array([-.001, -.002, -.001]), np.array([.002, .001, .002]))
    bounds, info = floor_command_bounds(.38, .362, -.005, .005, original, max_norm=.005)
    assert not info['rebased']
    for actual, expected in zip(bounds, original):
        np.testing.assert_array_equal(actual, expected)


def test_stale_slew_box_is_rebased_to_a_feasible_floor_command():
    previous = np.array([.003, .003, -.002])
    slew = .0025 / np.sqrt(3)
    original = (previous - slew, previous + slew)
    bounds, info = floor_command_bounds(.361, .362, -.005, .005, original,
                                      previous, slew, .005)
    assert info['rebased'] and not info['emergency']
    assert np.all(bounds[0] <= bounds[1])
    assert bounds[0][2] >= .001 - 1e-12
    assert np.linalg.norm(np.clip(np.zeros(3), *bounds)) <= .005


def test_force_ball_and_floor_have_feasible_slew_intersection():
    previous = np.array([.0035, .0035, 0.])
    slew = .0001
    bounds, info = floor_command_bounds(.358, .362, -.005, .005,
                                      (previous-slew, previous+slew), previous, slew, .005)
    assert info['rebased']
    assert np.linalg.norm(np.clip(np.zeros(3), *bounds)) <= .005 + 1e-12
    assert bounds[0][2] >= .004 - 1e-12


def test_emergency_braking_does_not_make_the_bounded_qp_infeasible():
    original = (np.full(3, -.001), np.full(3, .001))
    bounds, info = floor_command_bounds(.35, .362, -.005, .005, original, max_norm=.005)
    assert info['emergency'] and not info['applied']
    assert bounds is original


def test_table_edge_rejects_only_the_outward_component():
    for axis, sign in [(0, -1), (0, 1), (1, -1), (1, 1)]:
        normal = np.zeros(2); normal[axis] = sign
        lower, upper, applied = table_edge_command_bounds(-.005, .005, .01, normal, .06, True)
        assert applied
        assert (upper[axis] if sign > 0 else lower[axis]) == 0.
        assert lower[1-axis] == -.005 and upper[1-axis] == .005
        assert lower[2] == -.005 and upper[2] == .005


def test_table_edge_bounds_are_inactive_when_detached_or_far():
    for active, gap in [(False, .01), (True, .1)]:
        lower, upper, applied = table_edge_command_bounds(-.005, .005, gap, [-1., 0.], .06, active)
        assert not applied
        np.testing.assert_array_equal(lower, np.full(3, -.005))
        np.testing.assert_array_equal(upper, np.full(3, .005))


def test_edge_and_floor_allow_reentry_without_an_infeasible_slew_box():
    lower, upper, _ = table_edge_command_bounds(-.005, .005, .01, [-1., 0.], .06, True)
    previous = np.array([-.004, .001, -.001])
    slew = .0025 / np.sqrt(3)
    bounds, info = floor_command_bounds(.3615, .362, lower, upper,
                                      (previous-slew, previous+slew), previous, slew, .005)
    assert info['rebased']
    assert bounds[0][0] >= 0 and bounds[0][2] >= .0005 - 1e-12
    assert np.all(bounds[0] <= bounds[1])
    assert np.linalg.norm(np.clip(np.zeros(3), *bounds)) <= .005


def test_emergency_floor_recovery_still_respects_active_edge_guard():
    lower, upper, _ = table_edge_command_bounds(-.005, .005, .01, [-1., 0.], .06, True)
    previous = np.array([-.004, .001, 0.])
    slew = .0025 / np.sqrt(3)
    bounds, info = floor_command_bounds(.35, .362, lower, upper,
                                      (previous-slew, previous+slew), previous, slew, .005)
    assert info['emergency'] and not info['applied']
    assert bounds[0][0] >= 0.
    assert np.all(bounds[0] <= bounds[1])
    assert np.linalg.norm(np.clip(np.zeros(3), *bounds)) <= .005


def test_flip_execution_constraints_can_be_disabled_independently(monkeypatch):
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    from planning.mpc_explicit import _build_mpc_planner_runtime, handle_mpc_request
    from examples.mpc.franka.runtime import mpc as policy
    record = json.loads((Path(__file__).parent/'fixtures/bunny_pose_contact.json').read_text())
    record['args']['pose_executor_constraints'] = True
    record['args']['flip_executor_constraints'] = False
    args, param, mpc, trackers = _build_mpc_planner_runtime(record)
    assert trackers['object_cost_stages'].stage == 'flip'
    def unexpected(*_):
        raise AssertionError('explicitly disabled flip constraints entered')
    monkeypatch.setattr(policy, '_table_edge_gap_and_normal', unexpected)
    result = handle_mpc_request(args, param, mpc, trackers, dict(record['msg'], dwell=None))
    assert not result['mpc_executor_edge']
    assert not result['mpc_executor_floor']['applied']


@pytest.mark.parametrize('stage,option', [('pose', 'pose_executor_constraints'),
                                         ('flip', 'flip_executor_constraints')])
def test_captured_contact_respects_executor_bounds_only_when_stage_enabled(monkeypatch, stage, option):
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    from planning.mpc_explicit import _build_mpc_planner_runtime, handle_mpc_request
    record = json.loads((Path(__file__).parent/'fixtures/bunny_pose_contact.json').read_text())
    for enabled in (False, True):
        init = dict(record, args=dict(record['args'], **{option: enabled}))
        args, param, mpc, trackers = _build_mpc_planner_runtime(init)
        trackers['object_cost_stages'].stage = stage
        msg = dict(record['msg'], dwell=None)
        q = np.array(msg['policy_q']); q[0] -= .1; q[7] -= .1
        msg['policy_q'] = q
        contact = dict(msg['actual_contact'])
        point = np.array(contact['point_world']); point[0] -= .1
        contact['point_world'] = point; msg['actual_contact'] = contact
        result = handle_mpc_request(args, param, mpc, trackers, msg)
        assert result['mpc_executor_edge'] is enabled
        if enabled:
            assert result['action'][0] >= -1e-10
            assert mpc.first_action_bounds[0][0] >= 0.
            assert result['action'][2] >= param.mpc_q_lb_[9] - q[9] - 1e-9
        assert result['solver_backend'] == 'acados'
        assert result['mpc_failures'] == 0
