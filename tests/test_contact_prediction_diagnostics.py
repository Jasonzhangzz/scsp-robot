"""World-frame prediction diagnostics must not influence the planned action."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from planning.contact_prediction_diagnostics import (
    local_prediction_to_world, pose_prediction_diagnostics,
    contact_prediction_diagnostics,
)
from planning.object_pose_cost import cost_config


def test_local_contact_prediction_composes_rotation_and_translation():
    rotation = Rotation.from_euler('xyz', [.3, -.4, 1.1])
    change = Rotation.from_euler('xyz', [-.1, .2, .3])
    current = np.r_[[.4, .1, .38], rotation.as_quat()[[3, 0, 1, 2]]]
    local = np.r_[[.001, -.002, .003], change.as_quat()[[3, 0, 1, 2]]]
    predicted = local_prediction_to_world(current, local)
    np.testing.assert_allclose(predicted[:3], current[:3] + rotation.apply(local[:3]))
    np.testing.assert_allclose(predicted[3:], (rotation * change).as_quat()[[3, 0, 1, 2]])
    assert local_prediction_to_world(current, None) is None


def test_total_gain_does_not_imply_position_gain():
    quaternion = lambda yaw: Rotation.from_euler('z', yaw).as_quat()[[3, 0, 1, 2]]
    current, predicted = [np.r_[[x, 0, 0], quaternion(yaw)]
                          for x, yaw in ((.10, .5), (.101, .3))]
    target = np.r_[np.zeros(3), 1., 0., 0., 0.]
    config = np.r_[cost_config(500, 20, 20), 0., 1-np.cos(np.radians(10)), 2000.]
    result = pose_prediction_diagnostics(current, predicted, target, config)
    assert result['object_cost_reduction'] > 0
    assert result['position_reduction_mm'] < 0
    np.testing.assert_allclose(result['object_cost_reduction'], sum(
        result[name + '_cost_reduction'] for name in ('position', 'quaternion', 'upright')))


def test_detached_patch_is_not_reported_as_measured_prediction():
    pose = np.r_[np.zeros(3), 1., 0., 0., 0.]
    param = SimpleNamespace(target_p_=np.zeros(3), target_q_=pose[3:],
                            object_cost_config_=cost_config(500, 20),
                            lambda_optimizer=SimpleNamespace(last_executed_x_plus=pose))
    result = contact_prediction_diagnostics(param, SimpleNamespace(model=SimpleNamespace()), pose,
        dict(cached_x_plus=pose, value_info=dict(using_observed_contact=False)), {})
    assert result['lambda_best'] is not None
    assert result['lambda_measured'] is None
    assert result['mpc_executed_first_step'] is None


def test_native_captured_pose_diagnostics_preserve_command(monkeypatch):
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    from planning.mpc_explicit import _build_mpc_planner_runtime, handle_mpc_request
    record = json.loads((Path(__file__).parent/'fixtures/bunny_pose_contact.json').read_text())
    actions = []
    for tracing in (False, True):
        init = dict(record, args=dict(record['args'], rollout_trace_jsonl='diagnostic' if tracing else ''))
        args, param, mpc, trackers = _build_mpc_planner_runtime(init)
        trackers['object_cost_stages'].stage = 'pose'
        result = handle_mpc_request(args, param, mpc, trackers, dict(record['msg'], dwell=None))
        actions.append(result['action'])
        if tracing:
            diagnostics = result['policy']['contact_prediction_diagnostics']
            assert diagnostics['lambda_best']['position_reduction_mm'] > 0
            assert diagnostics['lambda_measured'] is not None
            assert diagnostics['mpc_executed_first_step'] is not None
            assert diagnostics['mpc_tip_displacement_response'] < .05
            json.dumps(diagnostics, allow_nan=False)
    np.testing.assert_array_equal(*actions)


def test_rejected_measured_patch_does_not_force_unverified_withdrawal(monkeypatch):
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    import planning.mpc_explicit as planning
    record = json.loads((Path(__file__).parent/'fixtures/bunny_pose_contact.json').read_text())
    record['args']['rollout_trace_jsonl'] = ''
    args, param, mpc, trackers = planning._build_mpc_planner_runtime(record)
    trackers['object_cost_stages'].stage = 'pose'
    original = planning._call_rollout_contact_via

    def rejected_patch(*arguments, **keywords):
        policy = original(*arguments, **keywords)
        info = policy['value_info']
        assert info['using_observed_contact']
        info['accept_p_arm'] = False
        info['switch_observed_contact'] = False
        param.lambda_optimizer.contact_switch_confidence = .01
        policy['best_contact_track_world'] = np.asarray(record['msg']['policy_q'])[7:10] + [.1, 0., 0.]
        return policy

    monkeypatch.setattr(planning, '_call_rollout_contact_via', rejected_patch)
    result = planning.handle_mpc_request(args, param, mpc, trackers, dict(record['msg'], dwell=None))
    # A scalar confidence/point-distance heuristic is insufficient evidence
    # to drop unilateral support. Retain the original MPC contact branch.
    assert not result['mpc_skipped']
    assert result['policy']['execution_mode'] == 'contact'
