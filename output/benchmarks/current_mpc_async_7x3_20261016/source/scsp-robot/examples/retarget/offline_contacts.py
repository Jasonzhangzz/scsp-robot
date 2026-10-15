"""Optional demonstration-only Lambda/IK preparation, outside the control loop."""
import hashlib
import json
from pathlib import Path

import numpy as np

from models.wuji_hand import MaterialPoint, rotation
from planning.wuji_retarget import LocalLambdaPolicy, execution_contact_subsets
from examples.retarget.contact_geometry import support_parameters


def plan_fingerprint(hand, demo, contacts, geometric, args):
    digest = hashlib.sha256(b'wuji_offline_contacts_v3_nominal_surface_ik')
    for value in (demo.poses, demo.times, contacts.mask, contacts.anchors, contacts.normals,
                  geometric, [m.body for m in contacts.materials], [m.local for m in contacts.materials],
                  [m.normal for m in contacts.materials], hand.model.mesh_vert, hand.model.geom_friction,
                  hand.model.actuator_gainprm, hand.model.actuator_biasprm, hand.model.actuator_forcerange,
                  hand.model.actuator_ctrlrange, hand.model.body_mass,hand.model.body_inertia,
                  hand.model.body_ipos,hand.model.opt.gravity):
        digest.update(np.asarray(value).tobytes())
    digest.update(repr((hand.friction, args.force_limit, args.search_radii, args.fps_samples,
        args.candidate_capacity, args.lambda_iterations, args.contact_ik_iterations,
        args.contact_ik_tolerance, args.hand_patch_radius, args.lookahead, args.command_dt,
        hand.model.body_mass[hand.obj], hand.model.body_inertia[hand.obj],
        args.contact_penetration_limit, args.contact_target_depth,args.seed,hand.floor_friction)).encode())
    return digest.hexdigest()


def restore_plan(value):
    value = dict(value)
    value['slots'] = tuple(value['slots'])
    for key in ('points', 'normals', 'force_vectors_local'):
        value[key] = np.asarray(value[key], float).reshape(-1, 3)
    if 'ik' in value:
        value['ik'] = dict(value['ik'])
        if 'q' in value['ik']:value['ik']['q'] = np.asarray(value['ik']['q'], float)
        value['ik']['materials'] = [MaterialPoint(int(m['body']), np.asarray(m['local']),
            np.asarray(m['normal']), m['finger'], m['region'],
            None if m.get('source_local') is None else np.asarray(m['source_local']),
            m.get('contact_patch', False)) for m in value['ik'].get('materials', [])]
    return value


def execution_reference(plans,reference,target_reference):
    """Use a reachable upcoming grasp during an infeasible contact transition.

    Hand-shape playback still visits its current frame. Contact intent can
    look ahead as far as the already selected object reference; it must not
    chase an unrelated later release or replace a feasible current plan.
    """
    if plans[reference]['valid']:return reference
    for frame in range(reference+1,min(target_reference,len(plans)-1)+1):
        if plans[frame]['valid'] and plans[frame]['slots']:return frame
    return reference


def prepare_contacts(hand, demo, contacts, geometric, args, shape_materials, shape_targets):
    from examples.retarget.retarget import write_json
    fingerprint = plan_fingerprint(hand, demo, contacts, geometric, args)
    cache = Path(args.contact_plan_cache) if args.contact_plan_cache else args.output/'offline_contact_plans.json'
    if cache.exists():
        stored = json.loads(cache.read_text())
        if stored.get('fingerprint') != fingerprint:
            if args.contact_plan_cache:
                raise ValueError(f'offline contact cache does not match this scene/trajectory/settings: {cache}')
            print('[offline Lambda] preparation changed; rebuilding local cache',flush=True)
        else:
            if len(stored['plans']) != len(demo.poses):raise ValueError('incomplete contact cache')
            print(f'[offline Lambda] loaded {len(stored["plans"])} frames from {cache}', flush=True)
            return [restore_plan(p) for p in stored['plans']]
    policy = LocalLambdaPolicy(hand.surface, device=args.lambda_device, seed=args.seed,
        iterations=args.lambda_iterations, friction=hand.friction, force_limit=args.force_limit,
        radii=tuple(args.search_radii), samples=args.fps_samples, capacity=args.candidate_capacity,
        require_closure_descent=False)
    plans = [];seed = geometric[0].copy()
    for frame, pose in enumerate(demo.poses):
        hand.set_state(geometric[frame], pose, hand.data)
        hand.servo = geometric[frame].copy()
        target = demo.poses[min(np.searchsorted(demo.times, demo.times[frame]+args.lookahead), len(demo.poses)-1)]
        inertia, _, external = hand.dynamics()
        points = hand.kinematics(contacts.materials)[0]
        projection, gaps, projection_normals = hand.surface.query((points-pose[:3])@rotation(pose)) if len(points) else (np.empty((0,3)),np.empty(0),np.empty((0,3)))
        slots = tuple(np.flatnonzero(contacts.mask[frame]))
        def solve(fit):
            def verify(candidate, normals):
                nonlocal seed
                # Nominal surface reachability is independent of servo
                # compression. The online virtual target adds closure depth.
                world = pose[:3]+candidate@rotation(pose).T
                best = None
                for guess in (seed.copy(), geometric[frame].copy()):
                    data = hand.set_state(guess, pose)
                    materials = [hand.nearest_material(contacts.materials[k], p, data,
                        radius=args.hand_patch_radius) for k,p in zip(fit, world)]
                    q, report = hand.solve_ik(guess, pose, materials, world,
                        target_normals=normals@rotation(pose).T, iterations=args.contact_ik_iterations,
                        tolerance=args.contact_ik_tolerance, shape_materials=shape_materials,
                        shape_targets=shape_targets[frame], surface_priors=[contacts.materials[k] for k in fit],
                        surface_radius=args.hand_patch_radius)
                    report['q'] = q.copy()
                    if best is None or (report['valid'], -report['max_contact_error']) > (best['valid'], -best['max_contact_error']):best = report
                    if report['valid']:break
                if not best.get('collisions'):seed = best['q'].copy()
                return best
            ids = list(fit)
            return policy.solve(contacts.anchors[frame, ids], contacts.normals[frame, ids], fit,
                current=pose, target=target, inertia=inertia, velocity=np.zeros(6), external=external,
                dt=args.command_dt, verify=verify, projected_points=projection[ids],
                projected_normals=projection_normals[ids], **support_parameters(hand))
        plan = solve(slots)
        if slots and not plan['valid']:
            options = [solve(subset) for subset in execution_contact_subsets(contacts,frame,gaps,np.zeros(len(gaps)))]
            feasible = [p for p in options if p['valid']]
            if feasible:
                plan = min(feasible,key=lambda p:p.get('pose_cost',float('inf')))
                plan['semantic_relaxation'] = 'full_joint_contact_ik_infeasible'
        plan['prepared_offline'] = True
        plans.append(plan)
        print(f'[offline Lambda] {frame+1}/{len(demo.poses)} valid={plan["valid"]} source={plan["source"]}',flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    write_json(cache,dict(fingerprint=fingerprint,plans=plans,
        scope='demonstration_contact_and_ik_seeds_only; actual contacts and regrasp stay online'))
    return plans
