"""Counterfactual single-step model/plant comparisons from identical states."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
os.environ['MUJOCO_HEADLESS'] = '1'
from examples.mpc.fingertips.test import test_scm_mujoco as m
from planning.direct_contact_force import step_point_force
from planning.scm_batched_lm import GPUExactBatchLM


def local_pose(before, after):
    r0 = Rotation.from_quat(before[[4, 5, 6, 3]])
    r1 = Rotation.from_quat(after[[4, 5, 6, 3]])
    return r0.inv().apply(after[:3]-before[:3]), r0.inv()*r1


def audit(obj, seed, replay=None):
    args = m.build_parser().parse_args(['--obj', obj])
    args.rollout, args.solver = True, 'torch-gn'
    param = m.ExplicitMPCParams(args, rand_seed=seed, target_type='ground-rotation', model='explicit')
    param.trial_seed_ = seed
    env = m.MjSimulator(param)
    m._set_physical_mass(param, env, .01)
    opt = m.configure_direct_force_optimizer(param, env, args)
    if replay is not None:
        env.data_.qpos[:7] = replay['object_pose']
        r = m._object_frame(env.data_.qpos)
        env.data_.qvel[:6] = np.r_[r@np.asarray(replay['velocity_body'])[:3], replay['velocity_body'][3:]]
        env.data_.time = replay['simulation_time_s']
        mujoco.mj_forward(env.model_, env.data_)
    lm = GPUExactBatchLM(opt, iterations=20)
    lm.install()
    lm.validate = True
    solved = m._solve_step(param, args, env, m.Contact(param), 'surrogate')
    initial = copy.copy(env.data_)
    q0 = env.get_state().copy()
    pred = solved['response']['selected']
    p_pred = np.asarray(pred['x_plus'])[:3]
    r_pred = Rotation.from_quat(np.asarray(pred['x_plus'])[[4, 5, 6, 3]])
    force = solved['force_local']
    reaction = pred['contact_force_body']
    frame = np.column_stack((solved['n_arm'], solved['t1'], solved['t2']))
    baseline = opt.predict_contact_response(np.zeros(3), solved['p_local'], frame,
        solved['gravity'], velocity=solved['prediction_velocity'])
    baseline_p = np.asarray(baseline['x_plus'])[:3]
    baseline_r = Rotation.from_quat(np.asarray(baseline['x_plus'])[[4, 5, 6, 3]])
    cases = []
    for name, duration, f in [('pulse', .0005, force), ('continuous_actuator', .02, force),
                               ('continuous_reaction', .02, reaction), ('unforced', 0., force)]:
        env.data_ = copy.copy(initial)
        active = int(round(duration/env.model_.opt.timestep))
        count = [0]
        def command():
            count[0] += 1
            return f if count[0] <= active else np.zeros(3)
        info = step_point_force(env, solved['p_local'], np.zeros(3), force_at_substep=command)
        after = env.get_state()
        p_actual, r_actual = local_pose(q0, after)
        r0 = m._object_frame(q0)
        r1 = m._object_frame(after)
        velocity = np.r_[r0.T@env.data_.qvel[:3], r0.T@r1@env.data_.qvel[3:6]]
        cases.append(dict(execution=name, force_duration_s=duration,
            measured_translation_m=p_actual.tolist(), measured_rotation_deg=np.degrees(r_actual.as_rotvec()).tolist(),
            position_prediction_error_m=float(np.linalg.norm(p_actual-p_pred)),
            rotation_prediction_error_deg=float(np.degrees((r_pred.inv()*r_actual).magnitude())),
            measured_velocity_initial_body=velocity.tolist(),
            linear_velocity_error_m_s=float(np.linalg.norm(velocity[:3]-pred['v_plus'][:3])),
            angular_velocity_error_rad_s=float(np.linalg.norm(velocity[3:]-pred['v_plus'][3:])),
            impulse_body=info['actual_applied_impulse_body'].tolist()))
    unforced_p = np.asarray(cases[-1]['measured_translation_m'])
    unforced_r = Rotation.from_rotvec(np.radians(cases[-1]['measured_rotation_deg']))
    predicted_increment = baseline_r.inv()*r_pred
    for case in cases:
        measured_increment = unforced_r.inv()*Rotation.from_rotvec(np.radians(case['measured_rotation_deg']))
        case['force_effect_translation_error_m'] = float(np.linalg.norm(
            np.asarray(case['measured_translation_m'])-unforced_p-(p_pred-baseline_p)))
        case['force_effect_rotation_error_deg'] = float(np.degrees(
            (predicted_increment.inv()*measured_increment).magnitude()))
    return dict(object=obj, seed=seed, replay_step=None if replay is None else replay['step'], initial_pose=q0[:7].tolist(),
        initial_velocity=initial.qvel[:6].tolist(),
        selected_point=solved['p_local'].tolist(), actuator_force_body_N=force.tolist(),
        reaction_force_body_N=reaction.tolist(),
        prediction_translation_m=p_pred.tolist(), prediction_rotation_deg=np.degrees(r_pred.as_rotvec()).tolist(),
        prediction_velocity=pred['v_plus'].tolist(), validation=lm.validations, cases=cases)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--objects', default=','.join(m.DEFAULT_OBJECTS))
    parser.add_argument('--replay', type=Path)
    args = parser.parse_args()
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for obj in args.objects.split(','):
        result = audit(obj, 0)
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2)+'\n')
        print(obj, [(r['execution'], round(r['position_prediction_error_m']*1000, 3),
                     round(r['rotation_prediction_error_deg'], 3)) for r in result['cases']], flush=True)
    if args.replay:
        trial = json.loads(args.replay.read_text())
        for index in [49, 199, 999, 2999]:
            state = trial['rows'][index]
            results.append(audit(state['object'], state['seed'], replay=state))
            args.output.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
