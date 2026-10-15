"""Completed MPC contact relocations, computed offline from execution traces.

A switch requires an executed far contact target, physical loss of contact,
then stable physical contact on a distinct object-local patch at that target.
Best/selected sample IDs and ranking decisions never enter this detector.
"""
from __future__ import annotations

import numpy as np


def rotation_matrix(q):
    q = np.asarray(q, dtype=float).reshape(4)
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or norm < 1e-12:
        raise ValueError('Invalid object quaternion')
    w, x, y, z = q/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def execution_trace(observations, commands):
    """Extract measured contacts and actually issued targets from saved prefix data."""
    obs_by_step = {int(o['seq']): o for o in observations}
    command_by_step = {int(c['seq']): c for c in commands}
    rows = []
    for obs in observations:
        step = int(obs['seq'])
        q = np.asarray(obs['policy_q'], dtype=float)
        rot = rotation_matrix(q[3:7])
        contact = obs.get('actual_contact')
        gap = float(contact.get('dist', obs.get('contact_distance', np.inf))) if contact else np.inf
        physical = bool(obs.get('if_contact', False) or (contact is not None and gap <= 0))
        point = (rot.T @ (np.asarray(contact['point_world'])-q[:3])).tolist() if contact else None
        target = None
        previous_command = command_by_step.get(step-1)
        previous_obs = obs_by_step.get(step-1)
        if previous_command is not None and previous_command.get('press') is not None:
            previous_q = np.asarray(previous_obs['policy_q'], dtype=float)
            target = (rotation_matrix(previous_q[3:7]).T @
                      (np.asarray(previous_command['press'])-previous_q[:3])).tolist()
        rows.append(dict(step=step, sim_time=float(obs['sim_time']), in_contact=physical,
            contact_point_local=point, tip_local=(rot.T @ (q[7:10]-q[:3])).tolist(),
            executed_target_local=target, executed_command_step=step-1 if previous_command else None))
    return rows


def completed_contact_switches(rows, *, patch_distance=.03, arrival_distance=.02,
                               loss_steps=2, confirm_steps=3):
    """Ignore initial contact, index jitter, sliding, failed travel and same-patch recontact.

    Defaults require two consecutive detached control observations and three
    stable new-contact observations. These are explicit sampled-data detector
    parameters. No frequency is inferred from RS trigger counts.
    """
    if patch_distance <= 0 or arrival_distance <= 0 or loss_steps < 1 or confirm_steps < 1:
        raise ValueError('Detector distances and confirmation lengths must be positive')
    if not rows:
        raise ValueError('Missing execution contact trace')
    anchor = anchor_tip = initial = candidate = None
    initial_count = candidate_count = gap_count = 0
    intent = False
    departure_step = None
    events = []
    previous_step = None
    for row in rows:
        step = int(row['step'])
        if previous_step is not None and step != previous_step+1:
            raise ValueError('Execution trace must contain every control observation')
        previous_step = step
        if 'in_contact' not in row or 'executed_target_local' not in row:
            raise ValueError('Trace lacks physical contact or executed MPC target fields')
        contact = bool(row['in_contact'])
        point = row.get('contact_point_local')
        if contact and point is None:
            raise ValueError('Physical contact lacks measured patch geometry')
        point = None if point is None else np.asarray(point, dtype=float).reshape(3)
        tip = np.asarray(row['tip_local'], dtype=float).reshape(3)
        target = row['executed_target_local']
        target = None if target is None else np.asarray(target, dtype=float).reshape(3)
        if not np.isfinite(tip).all() or (point is not None and not np.isfinite(point).all()) or (
                target is not None and not np.isfinite(target).all()):
            raise ValueError('Nonfinite contact trace geometry')
        if anchor is None:
            if not contact:
                initial, initial_count = None, 0
                continue
            if initial is None or np.linalg.norm(point-initial) > patch_distance:
                initial, initial_count = point.copy(), 1
            else:
                initial_count += 1
            if initial_count >= confirm_steps:
                anchor, anchor_tip = point.copy(), tip.copy()
            continue
        reference_tip = tip if contact and gap_count == 0 else anchor_tip
        if target is not None:
            if np.linalg.norm(target-reference_tip) > patch_distance:
                intent = True
            elif gap_count and np.linalg.norm(target-anchor_tip) <= arrival_distance:
                intent = False
        if not contact:
            if gap_count == 0:
                departure_step = step
            gap_count += 1
            candidate, candidate_count = None, 0
            continue
        distinct = np.linalg.norm(point-anchor) > patch_distance
        arrived = target is not None and np.linalg.norm(tip-target) <= arrival_distance
        if intent and gap_count >= loss_steps and distinct:
            if arrived:
                if candidate is None or np.linalg.norm(point-candidate) > patch_distance:
                    candidate, candidate_count = point.copy(), 1
                    arrival_step = step
                else:
                    candidate_count += 1
                if candidate_count >= confirm_steps:
                    events.append(dict(departure_step=departure_step, arrival_step=arrival_step,
                        confirmed_step=step, sim_time=row['sim_time'],
                        from_local=anchor.tolist(), to_local=point.tolist(),
                        executed_target_local=target.tolist()))
                    anchor, anchor_tip = point.copy(), tip.copy()
                    gap_count, candidate_count, intent = 0, 0, False
                    candidate = None
            else:
                candidate, candidate_count = None, 0
        else:
            # Supported sliding updates the incumbent; it is not relocation.
            anchor, anchor_tip = point.copy(), tip.copy()
            gap_count, candidate_count, intent = 0, 0, False
            candidate = None
    return dict(mpc_contact_switches=len(events), events=events,
        definition='executed far MPC target + contact loss + stable distinct measured contact at target',
        parameters=dict(patch_distance_m=patch_distance, arrival_distance_m=arrival_distance,
                        loss_control_steps=loss_steps, confirm_control_steps=confirm_steps))
