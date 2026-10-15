#!/usr/bin/env python3
"""Verify actual MuJoCo contacts at every physics step, including interrupted runs.

Only saved position commands are replayed. No optimizer, ideal force, object
pose replay, or controller changes enter this audit. Diagnostics-only runs can
recover their commands from the logged servo increments.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import linprog

from models.wuji_hand import ObjectSurface, WujiHand, rotation
from planning.grasp_verification import verify_grasp, contact_wrench_rays
from examples.retarget.diagnose_lift import gravity_wrench
from examples.retarget.retarget import write_json


def longest_run(values):
    maximum = current = 0
    for value in values:
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def recover_commands(records, initial, lower, upper):
    """The logged executed_action is target minus the preceding servo target."""
    servo = np.asarray(initial, float).copy()
    commands = []
    for step, record in enumerate(records):
        if record['step'] != step:
            raise ValueError('A contiguous diagnostic prefix starting at step zero is required')
        action = np.asarray(record['mpc']['executed_action'], float)
        if action.shape != servo.shape or not np.isfinite(action).all():
            raise ValueError('Invalid or missing executed position command')
        servo = np.clip(servo + action, lower, upper)
        commands.append(servo.copy())
    return np.asarray(commands)


def gravity_capacity(points, normals, tangents, friction, caps, external):
    """Largest multiple of gravity supportable within measured normal budgets."""
    if not len(points):
        return 0.
    rays = contact_wrench_rays(points, normals, friction, edges=4, tangent_directions=tangents)
    radius = max(float(np.linalg.norm(points, axis=1).max()), .001)
    scale = np.r_[np.ones(3), np.full(3, 1/radius)]
    equality = scale[:, None] * np.c_[rays, external]
    normal_map = np.c_[np.kron(np.eye(len(points)), np.ones((1, 4))), np.zeros(len(points))]
    result = linprog(np.r_[np.zeros(rays.shape[1]), -1.], A_eq=equality, b_eq=np.zeros(6),
        A_ub=normal_map, b_ub=caps, bounds=(0., None), method='highs')
    if not result.success:
        raise RuntimeError(f'Gravity capacity LP failed: {result.message}')
    return float(result.x[-1])


def contacts(hand, data):
    pose = data.qpos[hand.oq:hand.oq+7]
    r = rotation(pose)
    actual = []
    floor_force = np.zeros(3)
    for index, c in enumerate(data.contact):
        bodies = hand.model.geom_bodyid[[c.geom1, c.geom2]]
        if hand.obj not in bodies:
            continue
        sign = -1. if bodies[0] == hand.obj else 1.
        local = np.zeros(6)
        mujoco.mj_contactForce(hand.model, data, index, local)
        force = sign * (c.frame.reshape(3, 3).T @ local[:3])
        other = int(bodies[1] if bodies[0] == hand.obj else bodies[0])
        if other == 0:
            floor_force += force
        if other not in hand.hand_bodies or np.linalg.norm(force) <= .001:
            continue
        if c.dim != 3 or not np.allclose(c.friction[:2], hand.friction, atol=1e-10):
            raise ValueError('This native-pyramid audit requires uniform isotropic condim=3 hand contacts')
        normal = sign * c.frame[:3]
        fn = max(float(normal @ force), 0.)
        tangential = c.frame.reshape(3, 3)[1:] @ force
        cone_violation = max(float(np.sum(np.abs(tangential))/hand.friction-fn), 0.) if hand.friction else float(np.linalg.norm(tangential))
        actual.append(dict(point=(c.pos-pose[:3]) @ r, normal=normal @ r,
            tangents=c.frame.reshape(3, 3)[1:] @ r, force_world=force,
            normal_force=fn, native_cone_force_violation_N=cone_violation, body=other,
            body_name=mujoco.mj_id2name(hand.model, mujoco.mjtObj.mjOBJ_BODY, other)))
    return actual, floor_force


def patch_arrays(actual, minimum):
    eligible = [c for c in actual if c['normal_force'] >= minimum]
    return (np.array([c['point'] for c in eligible]).reshape(-1, 3),
        np.array([c['normal'] for c in eligible]).reshape(-1, 3),
        np.array([c['tangents'] for c in eligible]).reshape(-1, 2, 3),
        np.array([c['normal_force'] for c in eligible]))


def actuator_support_witness(hand, snapshot, minimum_normal, *, measured_budget):
    """Static point-contact motor model; omits joint-stop and self-contact support.

    Infeasibility in this approximation does not prove that the real grip cannot
    lift: joint limits, self contacts and changing patches can transmit loads.
    """
    if (not np.allclose(hand.model.actuator_gear[:, 0], 1.) or not np.allclose(hand.model.actuator_gear[:, 1:], 0.) or
            np.any(hand.model.actuator_gaintype != 0) or np.any(hand.model.actuator_biastype != 1)):
        return dict(supported=False, reason='requires unit-gear position actuators with fixed gain and affine bias')
    actual = [c for c in snapshot['contacts'] if c['normal_force'] >= minimum_normal]
    if not actual:
        return dict(supported=True, feasible=False)
    points, normals, tangents, caps = patch_arrays(actual, minimum_normal)
    rays = contact_wrench_rays(points, normals, hand.friction, edges=4, tangent_directions=tangents)
    pose = np.asarray(snapshot['object_pose']);r = rotation(pose)
    data = mujoco.MjData(hand.model)
    hand.set_state(np.asarray(snapshot['hand_q']), pose, data)
    required_bias = data.qfrc_bias[hand.dadr]-data.qfrc_passive[hand.dadr]
    gain = hand.model.actuator_gainprm[:, 0]
    constant = hand.model.actuator_biasprm[:, 0]+hand.model.actuator_biasprm[:, 1]*data.actuator_length
    limited = hand.model.actuator_forcelimited.astype(bool)
    force_lower = np.where(limited, hand.model.actuator_forcerange[:, 0], -np.inf)
    force_upper = np.where(limited, hand.model.actuator_forcerange[:, 1], np.inf)
    # Include both actual force clamps and the executor's servo target bounds.
    endpoints = np.array([gain*hand.lower+constant, gain*hand.upper+constant])
    torque_lower = np.clip(endpoints.min(axis=0), force_lower, force_upper)
    torque_upper = np.clip(endpoints.max(axis=0), force_lower, force_upper)
    torque_map = np.zeros((hand.nu, rays.shape[1]))
    for i, contact in enumerate(actual):
        jp,jr = np.zeros((3,hand.model.nv)),np.zeros((3,hand.model.nv))
        mujoco.mj_jac(hand.model, data, jp, jr, pose[:3]+r@np.asarray(contact['point']), int(contact['body']))
        torque_map[:, 4*i:4*i+4] = jp[:,hand.dadr].T @ r @ rays[:3,4*i:4*i+4]
    inequality = np.r_[torque_map, -torque_map]
    bounds = np.r_[torque_upper-required_bias, required_bias-torque_lower]
    finite = np.isfinite(bounds)
    inequality, bounds = inequality[finite], bounds[finite]
    if measured_budget:
        inequality = np.r_[inequality, np.kron(np.eye(len(points)), np.ones((1,4)))]
        bounds = np.r_[bounds, caps]
    radius = max(float(np.linalg.norm(points,axis=1).max()), .001)
    scaling = np.r_[np.ones(3),np.full(3,1/radius)]
    external = gravity_wrench(hand,pose)
    result = linprog(np.ones(rays.shape[1]), A_eq=scaling[:,None]*rays, b_eq=-scaling*external,
        A_ub=inequality,b_ub=bounds,bounds=(0.,None),method='highs')
    report = dict(supported=True, feasible=bool(result.success), measured_normal_budget=measured_budget)
    if result.success:
        forces = np.array([rays[:3,4*i:4*i+4]@result.x[4*i:4*i+4] for i in range(len(points))])
        report.update(required_normal_forces=result.x.reshape(-1,4).sum(axis=1),
            measured_normal_forces=caps, bodies=[c['body_name'] for c in actual],
            required_joint_torque=required_bias+torque_map@result.x,
            max_contact_force_N=float(np.linalg.norm(forces,axis=1).max()),
            object_wrench_residual=float(np.linalg.norm(rays@result.x+external)))
    return report


def audit(directory, minimum_normal=.01, robust_normal=.05, friction_margin=.8, stable_cycles=6):
    directory = Path(directory)
    config = json.loads((directory/'config.json').read_text())
    lines = (directory/'full/diagnostics.jsonl').read_text().splitlines()
    records = []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1:
                raise
    if not records:
        raise ValueError('No recorded control cycles')
    semantic = np.load(directory/'semantic_contacts.npz')
    geometric = np.load(directory/'wuji_q.npy')
    xml = directory/'scene.xml'
    tree = ET.parse(xml).getroot()
    collider = next(g for g in tree.findall("worldbody/body[@name='obj']/geom") if g.get('contype', '1') != '0')
    mesh = tree.find(f"asset/mesh[@name='{collider.get('mesh')}']")
    surface = ObjectSurface((xml.parent/mesh.get('file')).resolve(),
        scale=np.fromstring(mesh.get('scale', '1 1 1'), sep=' '), convex=True)
    hand = WujiHand(xml, surface)
    if hand.model.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL:
        raise ValueError('The executor must use the pyramidal cone for this native audit')
    hand.set_state(geometric[0], semantic['object_pose'][0], hand.data)
    hand.servo = geometric[0].copy()
    hand.data.ctrl[:] = hand.servo
    commands = recover_commands(records, geometric[0], hand.lower, hand.upper)
    source = 'diagnostic_servo_increments'
    saved_path = directory/'full/rollout.npz'
    if saved_path.exists():
        with np.load(saved_path) as saved:
            if len(saved['controls']) != len(records):
                raise ValueError('Saved rollout and diagnostics cover different cycles')
            np.testing.assert_allclose(commands, saved['controls'], atol=1e-12, rtol=0)
            commands = saved['controls'].copy()
            source = 'saved_controls'
    dt = float(hand.model.opt.timestep)
    substeps = round(config['command_dt']/dt)
    if abs(substeps*dt-config['command_dt']) > 1e-10:
        raise ValueError('Command dt must be an integer number of physics steps')
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(hand.model, specification))
    probe = mujoco.MjData(hand.model)
    rows, qhistory, poses = [], [], []
    max_error = 0.
    for cycle, (command, log) in enumerate(zip(commands, records)):
        hand.servo = np.clip(command, hand.lower, hand.upper)
        hand.data.ctrl[:] = hand.servo
        for substep in range(substeps):
            hand.data.qfrc_applied[:] = 0.
            mujoco.mj_step(hand.model, hand.data)
            if substep == substeps-1:
                mujoco.mj_forward(hand.model, hand.data)
                data = hand.data
            else:
                # Evaluate post-step contacts without changing warm starts or
                # adding an intermediate forward call to the real replay.
                mujoco.mj_getState(hand.model, hand.data, state, specification)
                mujoco.mj_setState(hand.model, probe, state, specification)
                mujoco.mj_forward(hand.model, probe)
                data = probe
            pose = data.qpos[hand.oq:hand.oq+7].copy()
            actual, floor_force = contacts(hand, data)
            points, normals, tangents, caps = patch_arrays(actual, minimum_normal)
            external = gravity_wrench(hand, pose)
            native = verify_grasp(points, normals, hand.friction, edges=4, tangent_directions=tangents,
                external_wrench=external, normal_force_limits=caps)
            if not len(points):
                native['load_feasible'] = False
            circle = verify_grasp(points, normals, hand.friction)
            rp, rn, rt, _ = patch_arrays(actual, robust_normal)
            robust = verify_grasp(rp, rn, hand.friction*friction_margin, edges=4, tangent_directions=rt)
            capacity = gravity_capacity(points, normals, tangents, hand.friction, caps, external)
            force = np.sum([c['force_world'] for c in actual], axis=0) if actual else np.zeros(3)
            vertices = surface.mesh.vertices @ rotation(pose).T + pose[:3]
            local_force=rotation(pose).T@force
            moment=sum((np.cross(c['point'],rotation(pose).T@c['force_world']) for c in actual),start=np.zeros(3))
            measured_residual=np.r_[local_force,moment]+external
            residual_scale=np.r_[np.ones(3),np.full(3,1/max(np.linalg.norm(surface.mesh.vertices,axis=1).max(),.001))]
            relative_residual=float(np.linalg.norm(residual_scale*measured_residual)/max(np.linalg.norm(external[:3]),1e-9))
            rows.append(dict(cycle=cycle, substep=substep, time=(cycle*substeps+substep+1)*dt,
                reference_frame=log['reference_frame'], native_certificate=native, circular_certificate=circle,
                robust_certificate=robust, gravity_capacity_ratio=capacity, contacts=actual,
                eligible_contact_count=len(points), hand_force_world=force, floor_force_world=floor_force,
                measured_gravity_wrench_residual=measured_residual,
                relative_measured_gravity_residual=relative_residual,
                object_pose=pose, hand_q=data.qpos[hand.qadr].copy(),
                lowest_object_vertex_z=float(vertices[:, 2].min())))
        error = np.linalg.norm(hand.object_pose()[:3]-log['object_pose'][:3])
        max_error = max(max_error, float(error))
        qhistory.append(hand.data.qpos[hand.qadr].copy())
        poses.append(hand.object_pose())
    if max_error > 1e-6:
        raise RuntimeError(f'Physical replay mismatch {max_error:g} m; no certificate accepted')
    native_values = [x['native_certificate']['force_closure'] for x in rows]
    robust_values = [x['robust_certificate']['force_closure'] for x in rows]
    loaded = [x['native_certificate']['force_closure'] and x['native_certificate']['load_feasible'] for x in rows]
    floor=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    floor_z=float(hand.model.geom_pos[floor,2]) if floor>=0 else 0.
    weight=hand.model.body_mass[hand.obj]*np.linalg.norm(hand.model.opt.gravity)
    carrying=[bool(x['native_certificate']['load_feasible'] and
        x['lowest_object_vertex_z']>floor_z+.002 and abs(x['floor_force_world'][2])<.05*weight and
        x['relative_measured_gravity_residual']<.2) for x in rows]
    summary = dict(recorded_cycles=len(records), physics_steps=len(rows), physics_dt=dt,
        command_dt=config['command_dt'], command_source=source,
        complete_controller_run=(directory/'full/metrics.json').exists() and saved_path.exists(),
        commanded_hand_frames=len({r['reference_frame'] for r in records}), total_hand_frames=len(geometric),
        minimum_normal_N=minimum_normal, robust_minimum_normal_N=robust_normal,
        friction=hand.friction, robust_friction=hand.friction*friction_margin,
        native_force_closed_steps=sum(native_values), circular_force_closed_steps=sum(x['circular_certificate']['force_closure'] for x in rows),
        longest_native_closed_s=longest_run(native_values)*dt,
        robust_force_closed_steps=sum(robust_values), longest_robust_closed_s=longest_run(robust_values)*dt,
        native_load_feasible_steps=sum(bool(x['native_certificate']['load_feasible']) for x in rows),
        closed_and_load_feasible_steps=sum(loaded), longest_closed_and_loaded_s=longest_run(loaded)*dt,
        actual_hand_carry_steps=sum(carrying),longest_actual_hand_carry_s=longest_run(carrying)*dt,
        closed_hand_carry_steps=sum(a and b for a,b in zip(carrying,native_values)),
        longest_closed_hand_carry_s=longest_run([a and b for a,b in zip(carrying,native_values)])*dt,
        actual_carry_criteria=dict(clearance_m=.002,maximum_floor_weight_fraction=.05,maximum_relative_wrench_residual=.2),
        stable_window_s=stable_cycles*config['command_dt'],
        stable_native_closure=longest_run(native_values)>=stable_cycles*substeps,
        stable_robust_closure=longest_run(robust_values)>=stable_cycles*substeps,
        stable_closed_and_loaded=longest_run(loaded)>=stable_cycles*substeps,
        max_gravity_capacity_ratio=max(x['gravity_capacity_ratio'] for x in rows),
        max_hand_upward_force_N=max(x['hand_force_world'][2] for x in rows),
        gravity_N=float(hand.model.body_mass[hand.obj]*np.linalg.norm(hand.model.opt.gravity)),
        max_lift_m=max(x['object_pose'][2] for x in rows)-float(semantic['object_pose'][0, 2]),
        max_replay_position_error_m=max_error,
        max_measured_native_cone_violation_N=max((c['native_cone_force_violation_N'] for row in rows for c in row['contacts']), default=0.),
        mpc_preload_cycles=sum(r['mpc'].get('made_contact_count', 0)>0 for r in records),
        mpc_grasp_response_cycles=sum(r['mpc'].get('grasp_response', False) for r in records))
    closed_rows = [row for row in rows if row['native_certificate']['force_closure']]
    witnesses = []
    if closed_rows:
        best = max(closed_rows,key=lambda row:row['gravity_capacity_ratio'])
        bp,bn,bt,_=patch_arrays(best['contacts'],minimum_normal)
        external=gravity_wrench(hand,np.asarray(best['object_pose']))
        configured_caps=np.full(len(bp),config['force_limit']/np.sqrt(1+hand.friction**2))
        witnesses.append(dict(cycle=best['cycle'],substep=best['substep'],time=best['time'],
            gravity_capacity_ratio=best['gravity_capacity_ratio'],
            point_force_limit_gravity_capacity_ratio=gravity_capacity(bp,bn,bt,hand.friction,configured_caps,external),
            point_force_limit_certificate=verify_grasp(bp,bn,hand.friction,edges=4,tangent_directions=bt,
                external_wrench=external,normal_force_limits=configured_caps),
            motor_and_measured_budget=actuator_support_witness(hand,best,minimum_normal,measured_budget=True),
            motor_with_future_preload=actuator_support_witness(hand,best,minimum_normal,measured_budget=False)))
    output = directory/'full/force_closure_audit'
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'certificates.json', dict(summary=summary, representative_static_witnesses=witnesses, physics_steps=rows))
    np.savez_compressed(output/'replayed_prefix.npz', hand_q=qhistory, controls=commands,
        object_pose=poses, reference_frame=[r['reference_frame'] for r in records])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--minimum-normal', type=float, default=.01)
    parser.add_argument('--robust-normal', type=float, default=.05)
    parser.add_argument('--friction-margin', type=float, default=.8)
    parser.add_argument('--stable-cycles', type=int, default=6)
    args = parser.parse_args()
    if (not np.isfinite([args.minimum_normal, args.robust_normal, args.friction_margin]).all() or
            args.minimum_normal<=0 or args.robust_normal<args.minimum_normal or
            not 0<args.friction_margin<=1 or args.stable_cycles<1):
        parser.error('Invalid contact threshold, friction margin or stability window')
    print(json.dumps(audit(args.directory, args.minimum_normal, args.robust_normal,
        args.friction_margin, args.stable_cycles), indent=2))


if __name__ == '__main__':
    main()
