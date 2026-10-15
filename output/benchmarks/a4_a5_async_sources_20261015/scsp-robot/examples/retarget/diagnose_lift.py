#!/usr/bin/env python3
"""Audit a saved retarget rollout by replaying its actual position commands.

No optimizer is rerun and no contact force or object trajectory is injected.
Closure is a geometry certificate; bounded gravity support is checked separately.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from collections import Counter
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from models.wuji_hand import WujiHand, ObjectSurface, MaterialPoint, rotation
from planning.grasp_verification import verify_grasp
from examples.retarget.contact_geometry import contact_geometry
from examples.retarget.retarget import write_json


def material(value):
    return MaterialPoint(int(value['body']), np.asarray(value['local']),
                         np.asarray(value['normal']), value['finger'], value['region'])


def gravity_wrench(hand, pose):
    r = rotation(pose)
    force = hand.model.body_mass[hand.obj] * hand.model.opt.gravity
    arm = r @ hand.model.body_ipos[hand.obj]
    return np.r_[r.T @ force, r.T @ np.cross(arm, force)]


def loaded_certificate(points, normals, friction, external, **limits):
    result = verify_grasp(points, normals, friction, external_wrench=external, **limits)
    if not len(points):
        result['load_feasible'] = False
    return result


def diagnose(directory, mode='full'):
    directory = Path(directory)
    config = json.loads((directory/'config.json').read_text())
    saved = np.load(directory/mode/'rollout.npz')
    semantic = np.load(directory/'semantic_contacts.npz')
    geometric = np.load(directory/'wuji_q.npy')
    records = [json.loads(line) for line in (directory/mode/'diagnostics.jsonl').read_text().splitlines()]
    plans = json.loads((directory/mode/'contact_plans.json').read_text())
    if len(records) != len(plans) or len(records) != len(saved['controls']):
        raise ValueError('rollout, contact plans and diagnostics must cover the same cycles')
    xml = directory/'scene.xml'
    tree = ET.parse(xml).getroot()
    collider = next(g for g in tree.findall("worldbody/body[@name='obj']/geom") if g.get('contype', '1') != '0')
    mesh = tree.find(f"asset/mesh[@name='{collider.get('mesh')}']")
    surface = ObjectSurface((xml.parent/mesh.get('file')).resolve(),
        scale=np.fromstring(mesh.get('scale', '1 1 1'), sep=' '), convex=True)
    hand = WujiHand(xml, surface)
    hand.set_state(geometric[0], semantic['object_pose'][0], hand.data)
    hand.servo = geometric[0].copy()
    hand.data.ctrl[:] = hand.servo
    materials = [MaterialPoint(int(b), p, n, str(f), str(r)) for b, p, n, f, r in zip(
        semantic['body'], semantic['local'], semantic['local_normal'], semantic['finger'], semantic['region'])]
    rows = []
    for step, (command, log, plan) in enumerate(zip(saved['controls'], records, plans)):
        reference = int(log['reference_frame'])
        active = semantic['mask'][reference].astype(float)
        ids = np.flatnonzero(active)
        if mode == 'full' and plan['valid']:
            for slot, value in zip(plan['slots'], plan.get('ik', {}).get('materials', [])):
                materials[slot] = material(value)
        # Match the MPC's actual semantic mask, including its infeasible fallback.
        live_controller = 'lambda_valid' in log['mpc']
        effective = active if plan['valid'] or live_controller else np.zeros_like(active)
        _, jac, _, _, _, stick, _ = contact_geometry(hand, materials, effective, len(materials)+9)
        measured, _ = hand.observed_contacts(materials, effective)
        normal_rows = jac.reshape(-1, 4, 6+hand.nu)[:len(materials), :, :3].mean(axis=1)
        made = (stick > .5) & (np.einsum('ij,ij->i', normal_rows, measured) > .01)
        if not live_controller:made &= effective > 0  # Historical controller's gating.
        hand.execute(command, config['command_dt'])
        pose = hand.object_pose()
        r = rotation(pose)
        external = gravity_wrench(hand, pose)
        _, actual = hand.observed_contacts(materials, active)
        points = np.array([(c['point']-pose[:3]) @ r for c in actual]).reshape(-1, 3)
        normals = np.array([c['normal'] @ r for c in actual]).reshape(-1, 3)
        caps = np.maximum([c['normal'] @ c['force'] for c in actual], 0.)
        actual_cert = loaded_certificate(points, normals, hand.friction, external, normal_force_limits=caps)
        actual_unbounded = loaded_certificate(points, normals, hand.friction, external)
        original_cert = loaded_certificate(semantic['anchors'][reference, ids],
            semantic['normals'][reference, ids], hand.friction, external, force_limit=config['force_limit'])
        selected_cert = None
        ideal_force = np.zeros(3)
        ideal_wrench = np.zeros(6)
        if mode == 'full' and plan['valid'] and len(ids):
            selected_cert = loaded_certificate(plan['points'], plan['normals'], hand.friction, external,
                force_limit=config['force_limit'])
            forces = np.asarray(plan['force_vectors_local'])
            ideal_force = forces.sum(axis=0) @ r.T
            ideal_wrench = np.r_[forces.sum(axis=0), np.cross(plan['points'], forces).sum(axis=0)]
        floor_force = np.zeros(3)
        for index, contact in enumerate(hand.data.contact):
            bodies = hand.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if hand.obj not in bodies or 0 not in bodies:
                continue
            local = np.zeros(6)
            mujoco.mj_contactForce(hand.model, hand.data, index, local)
            floor_force += contact.frame.reshape(3, 3).T @ local[:3] * (-1 if bodies[0] == hand.obj else 1)
        actual_force = np.sum([c['force'] for c in actual], axis=0) if actual else np.zeros(3)
        rejected = plan.get('rejected_candidates', [])
        collision_free_pose_rejections = [x for x in rejected if not x.get('collisions') and
            x.get('max_contact_error', float('inf')) <= config['contact_ik_tolerance'] and
            x.get('pose_neighborhood_valid') is False]
        rows.append(dict(step=step, reference_frame=reference, raw_frame=int(semantic['frame_ids'][reference]),
            active_count=len(ids), lambda_valid=plan['valid'], lambda_source=plan['source'],
            original_certificate=original_cert, selected_certificate=selected_cert,
            actual_certificate=actual_cert, actual_unbounded_load_feasible=actual_unbounded['load_feasible'],
            actual_contacts=actual, actual_distinct_bodies=len({c['body'] for c in actual}),
            mpc_made_slots=int(made.sum()), mpc_grasp_response=log['mpc'].get('grasp_response', False),
            actual_force_world=actual_force, floor_force_world=floor_force, ideal_force_world=ideal_force,
            ideal_gravity_wrench_residual=float(np.linalg.norm(ideal_wrench+external)) if selected_cert else None,
            gravity_norm=float(np.linalg.norm(external[:3])),
            pose_only_rejected_candidates=len(collision_free_pose_rejections),
            original_rejection=next((x for x in rejected if x['source']=='mano_original'), None),
            object_z=float(pose[2]), demo_object_z=float(semantic['object_pose'][reference, 2]),
            object_goal_z=float(semantic['object_pose'][log['object_reference_frame'], 2]),
            replay_position_error=float(np.linalg.norm(pose[:3]-saved['object_pose'][step, :3])),
            replay_joint_error=float(np.max(np.abs(hand.data.qpos[hand.qadr]-saved['hand_q'][step])))))
    contact_rows = [x for x in rows if x['active_count']]
    stats = dict(mode=mode, cycles=len(rows), contact_cycles=len(contact_rows),
        valid_contact_plan_cycles=sum(x['lambda_valid'] for x in contact_rows),
        original_force_closed_cycles=sum(x['original_certificate']['force_closure'] for x in contact_rows),
        original_bounded_gravity_support_cycles=sum(bool(x['original_certificate']['load_feasible']) for x in contact_rows),
        selected_force_closed_cycles=sum(bool(x['selected_certificate'] and x['selected_certificate']['force_closure']) for x in contact_rows),
        selected_bounded_gravity_support_cycles=sum(bool(x['selected_certificate'] and x['selected_certificate']['load_feasible']) for x in contact_rows),
        actual_force_closed_cycles=sum(x['actual_certificate']['force_closure'] for x in rows),
        actual_bounded_gravity_support_cycles=sum(bool(x['actual_certificate']['load_feasible']) for x in rows),
        actual_unbounded_gravity_support_cycles=sum(bool(x['actual_unbounded_load_feasible']) for x in rows),
        max_actual_distinct_bodies=max(x['actual_distinct_bodies'] for x in rows),
        max_mpc_made_slots=max(x['mpc_made_slots'] for x in rows),
        grasp_response_cycles=sum(x['mpc_grasp_response'] for x in rows),
        max_hand_upward_force_N=max(x['actual_force_world'][2] for x in rows),
        gravity_N=rows[0]['gravity_norm'],
        object_lift_m=max(x['object_z'] for x in rows)-float(semantic['object_pose'][0, 2]),
        demo_lift_m=float(np.max(semantic['object_pose'][:, 2])-semantic['object_pose'][0, 2]),
        failed_cycles_with_pose_only_rejection=sum(x['pose_only_rejected_candidates']>0 for x in rows),
        plan_sources=Counter(x['lambda_source'] for x in rows),
        max_replay_position_error_m=max(x['replay_position_error'] for x in rows),
        max_replay_joint_error_rad=max(x['replay_joint_error'] for x in rows))
    geometry_reports = json.loads((directory/'ik_diagnostics.json').read_text())
    contact_geometry_reports = [report for report, mask in zip(geometry_reports, semantic['mask']) if np.any(mask)]
    stats['geometric_frames_with_contacts'] = len(contact_geometry_reports)
    stats['geometric_frames_satisfying_semantic_contacts'] = sum(
        bool(report['semantic_contact_valid']) for report in contact_geometry_reports)
    stats['geometric_contact_error_median_m'] = float(np.median([
        report['max_contact_error'] for report in contact_geometry_reports])) if contact_geometry_reports else None
    if stats['max_replay_position_error_m'] > 1e-6 or stats['max_replay_joint_error_rad'] > 1e-6:
        raise RuntimeError('physical replay differs from saved states; check assets and MuJoCo version')
    return dict(summary=stats, cycles=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--mode', choices=['full', 'ik-only'], default='full')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = diagnose(args.directory, args.mode)
    output = args.output or args.directory/args.mode/'lift_diagnosis.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
