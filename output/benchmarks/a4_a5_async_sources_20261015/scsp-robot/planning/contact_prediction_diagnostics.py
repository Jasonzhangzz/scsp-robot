"""Compare contact predictions in the measured world frame.

Lambda predicts a hypothetical, already reached patch. MPC predicts the
command at the measured contact manifold. Keep those meanings explicit.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from planning.object_pose_cost import object_cost_np, rotate_axis_np, param_cost_config


def local_prediction_to_world(current_pose, local_prediction):
    if local_prediction is None:
        return None
    current = np.asarray(current_pose, dtype=float).reshape(-1)[:7]
    local = np.asarray(local_prediction, dtype=float).reshape(-1)[:7]
    if local.size != 7 or not np.isfinite(local).all() or np.linalg.norm(local[3:]) < 1e-12:
        return None
    rotation = Rotation.from_quat(current[[4, 5, 6, 3]])
    orientation = rotation * Rotation.from_quat(local[[4, 5, 6, 3]])
    return np.r_[current[:3] + rotation.apply(local[:3]),
                 orientation.as_quat()[[3, 0, 1, 2]]]


def pose_prediction_diagnostics(current_pose, predicted_pose, target, config):
    if predicted_pose is None:
        return None
    current, predicted, target = [np.asarray(x, dtype=float).reshape(-1)[:7]
                                  for x in (current_pose, predicted_pose, target)]
    if predicted.size != 7 or not np.isfinite(predicted).all():
        return None
    position_error = float(np.linalg.norm(predicted[:3] - target[:3]))
    q, qd = predicted[3:] / np.linalg.norm(predicted[3:]), target[3:] / np.linalg.norm(target[3:])
    # Evaluate each term using the same shared region/barrier implementation.
    terms = {}
    for name, keep in [('position', (0,)), ('quaternion', (1,)), ('upright', (2, 11))]:
        component = np.asarray(config, dtype=float).copy()
        for index in (0, 1, 2, 11):
            if index < len(component) and index not in keep:
                component[index] = 0.
        terms[name + '_cost_reduction'] = (
            object_cost_np(current, target, component) - object_cost_np(predicted, target, component))
    return dict(pose_world=predicted.tolist(), position_error_m=position_error,
                position_reduction_mm=1000 * (np.linalg.norm(current[:3] - target[:3]) - position_error),
                quaternion_error=float(1 - np.clip(np.dot(q, qd), -1., 1.)**2),
                tilt_deg=float(np.degrees(np.arccos(np.clip(
                    np.dot(rotate_axis_np(q, np.asarray(config)[3:6]), np.asarray(config)[6:9]), -1., 1.)))),
                object_cost_reduction=object_cost_np(current, target, config) - object_cost_np(predicted, target, config),
                **terms)


def contact_prediction_diagnostics(param, mpc, current_pose, policy, solution):
    """Diagnostics only: never feed hypothetical gains into execution feedback."""
    opt = param.lambda_optimizer
    target = np.r_[param.target_p_, param.target_q_]
    config = param_cost_config(param)
    compare = lambda predicted: pose_prediction_diagnostics(current_pose, predicted, target, config)
    terminal = solution.get('rollout_q')
    terminal = None if terminal is None else np.asarray(terminal)[-1, :7]
    return dict(
        best_contact_idx=policy.get('best_contact_idx'),
        measured_contact_idx=(policy.get('value_info') or {}).get('actual_contact_idx'),
        execution_mode=policy.get('execution_mode'),
        contact_relocation_requested=bool((policy.get('value_info') or {}).get('contact_relocation_requested', False)),
        lambda_best_assumes_contact_reached=True,
        lambda_best=compare(local_prediction_to_world(current_pose, policy.get('cached_x_plus'))),
        lambda_measured=(compare(local_prediction_to_world(current_pose, getattr(opt, 'last_executed_x_plus', None)))
                         if (policy.get('value_info') or {}).get('using_observed_contact') else None),
        mpc_executed_first_step=compare(policy.get('execution_object_pose_world')),
        mpc_terminal=compare(terminal),
        mpc_tip_displacement_response=getattr(mpc.model, 'response', None),
    )
