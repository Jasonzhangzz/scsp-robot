"""Measured-state, one-finger-at-a-time contact recovery for online Wuji MPC."""
from dataclasses import dataclass

import numpy as np

from models.wuji_hand import rotation
from planning.grasp_verification import verify_grasp


@dataclass(frozen=True)
class RegraspSettings:
    failure_cycles: int = 15
    lost_cycles: int = 3
    phase_cycles: int = 40
    clearance: float = .003


class OnlineRegrasp:
    """Change contact intent, never contacts, forces or simulated positions.

    Actual unilateral contact remains in MPC throughout release. Transitions
    require measured separation/arrival; elapsed time alone never fakes touch.
    """
    def __init__(self, settings=None):
        self.settings = settings or RegraspSettings()
        self.mode = 'track'
        self.stalled = 0
        self.age = 0
        self.lost = {}
        self.seen = set()
        self.overrides = {}
        self.last_finger = None
        self.attempts = 0
        self.finger_attempts = {}
        self.settled = 0
        self.last_goal_error = None
        self.best_gap = np.inf
        self.move = None

    def update(self, hand, materials, anchors, normals, active, actual, certificate, goal, original):
        s = self.settings
        pose = hand.object_pose();r = rotation(pose)
        anchors = np.asarray(anchors, float).copy()
        normals = np.asarray(normals, float).copy()
        active = np.asarray(active, bool)
        materials = list(materials)
        if self.move is not None and active[self.move['slot']]:
            materials[self.move['slot']] = self.move['material']
        self.overrides = {k:v for k,v in self.overrides.items() if active[k]}
        for k, (point, normal, _) in self.overrides.items():
            anchors[k] = original[k]
            normals[k] = hand.surface.query(np.asarray(original[k:k+1]))[2][0]
            materials[k] = self.overrides[k][2]
        measured = [c for c in actual if c['normal']@c['force'] > .01]
        fingers = sorted({m.finger for k,m in enumerate(materials) if active[k] and m.finger != 'palm'})
        touching = {materials[c['slot']].finger for c in measured if c['slot'] >= 0}
        if hasattr(hand,'finger_contact_gaps'):
            physical_touching={finger for finger,gap in hand.finger_contact_gaps().items() if gap<=.0001}
        else:
            physical_touching={materials[c['slot']].finger for c in actual
                if c['slot']>=0 and c.get('gap',0.)<=.0001}
        self.seen = (self.seen|touching)&set(fingers)
        for f in fingers:self.lost[f] = 0 if f in touching else self.lost.get(f, 0)+1
        points = hand.kinematics(materials)[0]
        local = (points-pose[:3])@r
        _, gaps, _ = hand.surface.query(local) if len(materials) else (None, np.empty(0), None)
        gap = float(np.min(np.abs(gaps[active]))) if np.any(active) else np.inf
        error = float(np.linalg.norm(pose[:3]-np.asarray(goal)[:3]))
        progressing = self.last_goal_error is not None and self.last_goal_error-error > .0002
        approaching = gap < self.best_gap-.0002
        good = bool(certificate.get('load_feasible'))
        self.stalled = 0 if good or progressing or approaching or not fingers else self.stalled+1
        self.best_gap = gap if approaching else min(self.best_gap, gap)
        self.last_goal_error = error
        event = None
        if self.move is not None and not active[self.move['slot']]:
            self.move = None;self.mode = 'track';event = 'reference_released'
        if self.mode == 'track' and fingers:
            lost = [f for f in fingers if f in self.seen and self.lost[f] >= s.lost_cycles]
            if lost or self.stalled >= s.failure_cycles:
                forces = {f:sum(max(c['normal']@c['force'], 0.) for c in measured
                    if c['slot'] >= 0 and materials[c['slot']].finger == f) for f in fingers}
                # On an unheld object, repeatedly repairing only the weakest
                # ring/pinky patches can starve the thumb whose contact geometry
                # prevents load balance. Explore every requested finger before
                # retrying one. A supporting grasp still prioritizes lost pads.
                choices = (lost or fingers) if good else fingers
                finger = min(choices, key=lambda f:(self.finger_attempts.get(f,0),
                    f == self.last_finger, forces[f]))
                remaining = [c for c in measured if c['slot'] < 0 or materials[c['slot']].finger != finger]
                # Releasing a supporting finger in mid-air needs the remaining
                # measured contacts to carry the load. On the table recovery
                # can proceed even when the current grasp cannot close.
                vertices = hand.surface.mesh.vertices@r.T+pose[:3]
                if vertices[:, 2].min() > .002:
                    inertia, _, gravity = hand.dynamics()
                    cert = verify_grasp(np.array([(c['point']-pose[:3])@r for c in remaining]).reshape(-1,3),
                        np.array([c['normal']@r for c in remaining]).reshape(-1,3), hand.friction,
                        edges=4, tangent_directions=np.array([c['tangents']@r for c in remaining]).reshape(-1,2,3),
                        external_wrench=np.r_[r.T@gravity[:3],gravity[3:]],
                        normal_force_limits=np.array([c['normal']@c['force'] for c in remaining]))
                    if not cert['load_feasible']:
                        return anchors, normals, dict(phase='waiting_support', reason='remaining_contacts_cannot_carry',
                            finger=finger, attempts=self.attempts, moving_slots=[]), {}
                slots = [k for k,m in enumerate(materials) if active[k] and m.finger == finger]
                k = min(slots, key=lambda j:np.linalg.norm(local[j]-anchors[j]))
                # Recovery always aims back at the demonstration. Nearby
                # projections are independently evaluated by the online MPC;
                # they never replace MANO with a randomly selected new target.
                target = np.asarray(original[k], float).copy()
                normal = hand.surface.query(target[None])[2][0]
                closest, _, current_normal = hand.surface.query(local[k:k+1])
                finger_slots=[j for j,m in enumerate(materials) if m.finger==finger]
                self.move = dict(slot=k, finger=finger, slots=finger_slots, material=materials[k],
                    target=target, normal=normal, retreat=closest[0]-s.clearance*current_normal[0],
                    reason='contact_lost' if lost else 'grasp_stalled')
                self.mode = 'release';self.age = 0;self.settled = 0;self.attempts += 1
                self.finger_attempts[finger] = self.finger_attempts.get(finger,0)+1
                event = 'begin_release'
        motion = {}
        if self.move is not None:
            move = self.move;k = move['slot'];self.age += 1
            move['target'] = np.asarray(original[k], float).copy()
            move['normal'] = hand.surface.query(move['target'][None])[2][0]
            if self.mode == 'release' and move['finger'] not in physical_touching and gaps[k] >= .6*s.clearance:
                self.mode = 'approach';self.age = 0;event = 'measured_separation'
            # MANO correspondence is soft. After true separation, nearby
            # recontact may be accepted by MPC without first passing a hard
            # distance threshold around the MANO anchor.
            # A proximal link touching does not establish the distal material
            # being recovered. Accept nearby patches on the requested body,
            # without imposing a distance threshold around the MANO anchor.
            arrived = any(c['body']==move['material'].body for c in measured)
            if self.mode in ('approach','settle'):
                self.mode = 'settle' if arrived else 'approach'
                self.settled = self.settled+1 if arrived else 0
                if self.settled >= 3:
                    self.overrides[k] = (move['target'].copy(), move['normal'].copy(), move['material'])
                    self.last_finger = move['finger'];self.mode = 'track';self.move = None
                    self.stalled = 0;self.best_gap = np.inf;event = 'measured_recontact'
            if self.move is not None and self.age >= s.phase_cycles:
                self.last_finger = move['finger'];self.move = None;self.mode = 'track'
                self.stalled = 0;event = 'phase_timeout'
            elif self.move is not None:
                anchors[k] = move['retreat'] if self.mode == 'release' else move['target']
                normals[k] = move['normal']
                motion[k] = move['material']
        for k, (_, _, material) in self.overrides.items():motion[k] = material
        moving = self.move['slots'] if self.move is not None else []
        report = dict(phase=self.mode, event=event, attempts=self.attempts, moving_slots=moving,
            finger=None if self.move is None else self.move['finger'],
            slot=None if self.move is None else self.move['slot'],
            reason=None if self.move is None else self.move['reason'],
            target_source='MANO_object_anchor',nearby_contact_source='current_hand_nearest_projection',
            target=None if self.move is None else self.move['target'].tolist(),
            mano_anchor_error_m=None if self.move is None else float(np.linalg.norm(local[self.move['slot']]-self.move['target'])),
            clearance_m=s.clearance,
            persistent_slots=sorted(self.overrides))
        return anchors, normals, report, motion
