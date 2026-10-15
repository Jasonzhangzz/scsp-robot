#!/usr/bin/env python3
"""MANO -> Wuji semantic IK, local Lambda and closed-loop contact MPC."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import argparse
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import is_dataclass,asdict

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from models.wuji_hand import WujiHand,ObjectSurface,MaterialPoint,build_scene,rotation,normalize
from planning.wuji_retarget import load_demonstration,extract_contacts,LocalLambdaPolicy,ReferenceProgress,HandReferenceProgress,WUJI_FINGERS,stabilize_demonstration
from planning.grasp_verification import verify_grasp
from examples.retarget.contact_geometry import support_parameters


CACHE=ROOT/'thirdparty/retarget_assets'
SEQUENCE='dexycb_004_sugar_box_20200709_142802'


def jsonable(value):
    if isinstance(value,np.ndarray):return jsonable(value.tolist())
    if isinstance(value,np.generic):return jsonable(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,Path):return str(value)
    if is_dataclass(value):return jsonable(asdict(value))
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items() if k!='rank'}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    return value


def write_json(path,value):
    Path(path).write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n')


def physics_audit(hand,hand_xml,object_scene):
    """Record compiled physics and check inherited DexForge hand parameters."""
    source=mujoco.MjModel.from_xml_path(str(hand_xml))
    fields=('timestep','gravity','integrator','solver','cone','iterations',
        'ls_iterations','tolerance','ls_tolerance','noslip_iterations',
        'noslip_tolerance','impratio','disableflags','enableflags')
    source_options={name:getattr(source.opt,name) for name in fields}
    options={name:getattr(hand.model.opt,name) for name in fields}
    differences=[name for name in fields if not np.array_equal(source_options[name],options[name])]
    actuator_fields=('actuator_gainprm','actuator_biasprm','actuator_ctrlrange',
        'actuator_forcerange','actuator_ctrllimited','actuator_forcelimited')
    if any(not np.allclose(getattr(source,name),getattr(hand.model,name)) for name in actuator_fields):
        raise ValueError('generated scene changed DexForge actuators')
    geom_fields=('geom_contype','geom_conaffinity','geom_condim','geom_friction',
        'geom_solref','geom_solimp','geom_priority')
    for i in range(source.ngeom):
        name=mujoco.mj_id2name(source,mujoco.mjtObj.mjOBJ_GEOM,i)
        j=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_GEOM,name)
        if j<0 or any(not np.allclose(getattr(source,k)[i],getattr(hand.model,k)[j]) for k in geom_fields):
            raise ValueError(f'generated scene changed DexForge geom {name}')
    return dict(hand_asset=hand_xml,object_scene=object_scene,options=options,mujoco_version=mujoco.__version__,
        source_options=source_options,option_overrides=differences,
        hand_geoms_and_actuators_match=True,gravity_compensation=False,
        structural_mount_exclusions=sorted(hand.mount_exclusions),
        object_mass=float(hand.model.body_mass[hand.obj]),object_inertia=hand.model.body_inertia[hand.obj],
        hand_object_friction=hand.friction,object_floor_friction=hand.floor_friction,
        object_force_injection=False)


def shape_references(hand,demo):
    """All phalanges follow MANO directions with the Wuji bone lengths.

    Absolute human fingertip locations overconstrain a different hand's bone
    lengths. Palm/wrist motion stays in world coordinates; finger joint targets
    are built from the measured human directions and neutral robot geometry.
    """
    materials=[];targets=[]
    neutral=hand.set_state(np.clip(np.zeros(hand.nu),hand.lower,hand.upper),demo.poses[0])
    palm_rotation=neutral.xmat[hand.palm].reshape(3,3)
    longitudinal=normalize(demo.joints[:,9]-demo.joints[:,0])
    transverse=demo.joints[:,5]-demo.joints[:,17]
    transverse=normalize(transverse-longitudinal*np.einsum('ij,ij->i',transverse,longitudinal)[:,None])
    rotations=np.stack((normalize(np.cross(transverse,longitudinal)),transverse,longitudinal),axis=2)
    for name,index in [('thumb',1),('index',5),('middle',9),('ring',13),('pinky',17)]:
        site=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_SITE,f'right_{name}_tip')
        if site<0:raise ValueError(f'missing {name} tip site')
        joints=[]
        for link,region in ((1,'proximal'),(3,'middle'),(4,'distal')):
            body=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,f'right_finger{WUJI_FINGERS[name]}_link{link}')
            joints.append(MaterialPoint(body,np.zeros(3),np.array([1.,0,0]),name,region))
        joints.append(MaterialPoint(int(hand.model.site_bodyid[site]),hand.model.site_pos[site].copy(),np.array([1.,0,0]),name,'distal'))
        positions=hand.kinematics(joints,neutral)[0]
        offset=palm_rotation.T@(positions[0]-neutral.xpos[hand.palm])
        goal=demo.joints[:,0]+np.einsum('bij,j->bi',rotations,offset)
        materials.append(joints[0]);targets.append(goal.copy())
        lengths=np.linalg.norm(np.diff(positions,axis=0),axis=1)
        for segment in range(3):
            direction=normalize(demo.joints[:,index+segment+1]-demo.joints[:,index+segment])
            goal=goal+lengths[segment]*direction
            materials.append(joints[segment+1]);targets.append(goal.copy())
    materials.append(MaterialPoint(hand.palm,np.zeros(3),np.array([1.,0,0]),'palm','palm'))
    targets.append(demo.joints[:,0].copy())
    return materials,np.stack(targets,axis=1)


def initialize_hand(hand,materials,targets,pose):
    q=np.clip(.25*(hand.lower+hand.upper),hand.lower,hand.upper);q[:6]=0
    data=hand.set_state(q,pose)
    source,_,_=hand.kinematics(materials,data)
    sc,tc=source.mean(0),targets.mean(0)
    u,_,vt=np.linalg.svd((targets-tc).T@(source-sc))
    r=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
    # MuJoCo's serial rx/ry/rz wrist uses intrinsic XYZ coordinates.
    q[3:6]=Rotation.from_matrix(r).as_euler('XYZ')
    data=hand.set_state(q,pose);p,_,_=hand.kinematics(materials,data)
    q[:3]+=np.mean(targets-p,axis=0)
    return np.clip(q,hand.lower+1e-6,hand.upper-1e-6)


def geometric_ik(hand,demo,contacts,args):
    shape,targets=shape_references(hand,demo)
    q=initialize_hand(hand,shape,targets[0],demo.poses[0]);states=[];reports=[]
    for i,pose in enumerate(demo.poses):
        slots=np.flatnonzero(contacts.mask[i]);materials=[contacts.materials[k] for k in slots]
        # Geometry tracks the MANO hand material points before Lambda moves anchors.
        q,report=hand.solve_ik(q,pose,materials,contacts.hand_targets[i,slots],
            target_normals=contacts.hand_normals[i,slots],shape_materials=shape,
            shape_targets=targets[i],iterations=args.ik_iterations,tolerance=args.ik_tolerance,
            proximity=.035 if i==0 else .25,material_weight=.2,shape_weight=1.)
        # A hand pose remains a valid geometric target when a particular
        # human surface contact cannot be matched by this morphology. Lambda
        # evaluates that contact reachability separately, with its tighter IK.
        report['semantic_contact_valid']=report['valid']
        # Shape and semantic reproduction are least-squares objectives. Their
        # residuals never reject a physically valid geometric posture.
        report['geometric_pose_valid']=bool(np.isfinite(q).all() and not report.get('collisions'))
        report['valid']=report['geometric_pose_valid']
        states.append(q.copy());reports.append(report)
        if i%20==0:print(f'[IK] {i+1}/{len(demo.poses)} contacts={len(slots)} valid={report["valid"]} error={report["max_contact_error"]:.4f}',flush=True)
    return np.array(states),reports


def metrics(hand,records,demo,contacts):
    positions=[];angles=[];demo_positions=[];demo_angles=[];hand_errors=[];object_errors=[];penetration=[];force_errors=[];slip=[];closures=[];margins=[];latencies=[]
    for item in records:
        pose=np.array(item['object_pose']);reference=demo.poses[item.get('object_reference_frame',item['reference_frame'])]
        positions.append(float(np.linalg.norm(pose[:3]-reference[:3])))
        angles.append(float(2*np.arccos(np.clip(abs(pose[3:]@reference[3:]),0,1))))
        timed=demo.poses[item['reference_frame']]
        demo_positions.append(float(np.linalg.norm(pose[:3]-timed[:3])))
        demo_angles.append(float(2*np.arccos(np.clip(abs(pose[3:]@timed[3:]),0,1))))
        hand_errors.extend(item['hand_contact_error']);object_errors.extend(item['object_anchor_error'])
        penetration.append(item['penetration']);force_errors.extend(item['force_error']);slip.extend(item['slip'])
        closures.append(item['actual_force_closure']['force_closure'])
        margins.append(item['actual_force_closure']['positive_span_margin']);latencies.append(item['cycle_time'])
    summary=lambda values:dict(mean=float(np.mean(values)) if len(values) else None,max=float(np.max(values)) if len(values) else None)
    return dict(steps=len(records),position_error_m=summary(positions),rotation_error_rad=summary(angles),
        hand_contact_error_m=summary(hand_errors),object_anchor_error_m=summary(object_errors),penetration_m=summary(penetration),
        demonstration_position_error_m=summary(demo_positions),demonstration_rotation_error_rad=summary(demo_angles),
        force_error_N=summary(force_errors),slip_m=summary(slip),actual_force_closure_fraction=float(np.mean(closures)) if closures else 0.,
        actual_closure_margin=summary(margins),
        contact_retention=float(np.mean([r['contact_fraction'] for r in records if r['active_contact_count']])) if any(r['active_contact_count'] for r in records) else None,
        solve_latency_seconds=summary(latencies),reference_completion=float(records[-1]['reference_frame']/max(len(demo.poses)-1,1)) if records else 0.,
        lambda_failures=sum(not r['lambda_valid'] for r in records),
        mpc_solver_failures=sum(any(reason in r['mpc']['status'] for reason in
            ('acados status','Maximum_Iterations_Exceeded','Infeasible_Problem_Detected',
             'Restoration_Failed','Invalid_Number_Detected','Error_In_Step_Computation')) for r in records),
        collision_protected_holds=sum(r['mpc'].get('selected_control')=='unsafe_hold' for r in records),
        object_reference_completion=float(records[-1]['matched_frame']/max(len(demo.poses)-1,1)) if records else 0.,
        ideal_torque_limit_violation_fraction=float(np.mean([r['ideal_torque_limit_violation'] for r in records if r.get('ideal_torque_limit_violation') is not None])) if any(r.get('ideal_torque_limit_violation') is not None for r in records) else None,
        hand_frames_tracked=len(set(r['reference_frame'] for r in records)),
        hand_tracking_error_rad=summary([r['finger_tracking_error'] for r in records if 'finger_tracking_error' in r]),
        wrist_tracking_error_m=summary([r['wrist_tracking_error'] for r in records if 'wrist_tracking_error' in r]),
        ideal_force_applied_to_object=False)


def run_rollout(hand,demo,contacts,geometric,args,mode,output):
    from planning.wuji_retarget_mpc import WujiOnlineMPC
    from examples.retarget.contact_geometry import contact_geometry
    output.mkdir(parents=True,exist_ok=True)
    hand.set_state(geometric[0],demo.poses[0],hand.data)
    hand.servo=geometric[0].copy();hand.data.ctrl[:]=hand.servo
    # Start from the same measured q/v in both methods, without teleporting later.
    cursor=ReferenceProgress(demo.poses,demo.times,args.reference_window,args.lookahead)
    hand_cursor=HandReferenceProgress(demo.times)
    policy=LocalLambdaPolicy(hand.surface,device=args.lambda_device,seed=args.seed,iterations=args.lambda_iterations,
        friction=hand.friction,force_limit=args.force_limit,radii=tuple(args.search_radii),samples=args.fps_samples,
        capacity=args.candidate_capacity,require_closure_descent=True)
    base_rates=np.r_[np.full(3,.15),np.full(3,.8),np.full(hand.nu-6,1.5)]
    peak=np.max(np.abs(np.diff(geometric,axis=0)/np.diff(demo.times)[:,None]),axis=0) if len(geometric)>1 else np.zeros(hand.nu)
    rates=np.maximum(base_rates,args.hand_rate_scale*peak)
    command_limits=args.command_dt*rates
    write_json(output/'hand_rate_limits.json',dict(rate_limits=rates,demonstration_peak=peak,scale=args.hand_rate_scale))
    mpc=WujiOnlineMPC(hand,len(contacts.episodes),args.command_dt,args.horizon,args.mpc_solver,
        rates=rates,costs=args.mpc_costs,force_limit=args.force_limit,
        native_force_feedback=getattr(args,'native_force_feedback',False)) if mode=='full' else None
    if mpc is not None:
        mpc.param_.mpc_failure_snapshot_=str(output/'acados_failure.npz')
        write_json(output/'mpc_costs.json',dict(weights=mpc.weights,force_residual_scale=mpc.force_scale))
        write_json(output/'planning_model.json',dict(version='native_contact_pose_mpc_v13',
            friction_cone='pyramidal',cone_edges=4,lambda_explicit_tangents=True,
            made_contact_frames='dominant_actual_MuJoCo_patch',
            future_contact_frames='normal_completion_with_observed_axis_transport',
            runtime_contact_friction=True,transport_requires_current_load_certificate=False,
            lambda_requires_closure_and_pose_descent=True,mpc_candidate_validation='native_MuJoCo_horizon',
            grasp_response_requires_force_closure=False,grasp_response_minimum_made_slots=2,
            external_wrench_target=False,lambda_force_tracking=False,external_pressure_override=False,
            internal_pressure_feedback='same_affine_command_map_in_prediction_and_execution',
            internal_preload='measured_pressure_candidates_and_stage_terminal_soft_cost',
            internal_load_objective='gravity_force_and_moment_balance',
            force_feedback=True,candidate_ranking_physics='native_MuJoCo_all_stages',
            native_force_sensitivity='bounded_Gauss_Newton_seed' if getattr(args,'native_force_feedback',False) else 'disabled',
            internal_preload_allocation='measured_patch_static_support_with_actuator_and_slot_limits',
            preload_candidates='signed_physical_patch_motion_and_bounded_equilibrium_servo_native_cost_selected',
            predicted_preload_growth='SCM_normal_response_bounded_by_force_limit',
            dynamics='reduced_grouped_SCM_and_force_limited_response',
            execution='unchanged_DexForge_MuJoCo_position_servos'))
    runtime_materials=contacts.materials.copy()
    records=[];plans=[];qhistory=[];controls=[];pose_history=[];phase='approach';last_qref=geometric[0].copy();last_slots=()
    stable=0;failure_streak=0;previous_actual={};renderer=None;writer=None;viewer=None;finished_dwell=0
    readiness_materials,_=shape_references(hand,demo)
    if not args.headless:
        from mujoco import viewer as mj_viewer
        viewer=mj_viewer.launch_passive(hand.model,hand.data)
    if args.video:
        import imageio.v2 as imageio
        renderer=mujoco.Renderer(hand.model,height=480,width=640)
        writer=imageio.get_writer(output/'rollout.mp4',fps=1/args.command_dt)
    log=(output/'diagnostics.jsonl').open('w')
    try:
        for step in range(args.max_steps):
            started=time.perf_counter();pose=hand.object_pose();q=hand.data.qpos[hand.qadr].copy()
            # The hand is a continuous demonstration trajectory. Contact or
            # object-pose stalls must not turn it into a static grasp target.
            reference=hand_cursor.update(args.command_dt,ready=step>0)
            index,target_index=cursor.update(pose,args.command_dt)
            target_shape=hand.kinematics(readiness_materials,hand.set_state(geometric[reference],pose))[0]
            slots=tuple(np.flatnonzero(contacts.mask[reference]));active=contacts.mask[reference].astype(float)
            goal=demo.poses[target_index];materials=runtime_materials if mode=='full' else contacts.materials
            points,normals,jp=hand.kinematics(materials)
            local=(points-pose[:3])@rotation(pose)
            projected,_,projection_normals=hand.surface.query(local) if materials else (np.empty((0,3)),None,np.empty((0,3)))
            measured,actual=hand.observed_contacts(materials,active)
            actual_points=np.full((len(materials),3),np.nan);actual_normals=np.full_like(actual_points,np.nan)
            actual_tangents=np.full((len(materials),2,3),np.nan);dominant_force=np.zeros(len(materials))
            for c in actual:
                if c['slot']>=0 and c['normal']@c['force']>dominant_force[c['slot']]:
                    dominant_force[c['slot']]=c['normal']@c['force']
                    actual_points[c['slot']]=(c['point']-pose[:3])@rotation(pose)
                    actual_normals[c['slot']]=c['normal']@rotation(pose)
                    actual_tangents[c['slot']]=c['tangents']@rotation(pose)
            anchors=contacts.anchors[reference].copy();inward=contacts.normals[reference].copy();desired=np.zeros_like(anchors)
            inertia,_,external=hand.dynamics()
            def verify(candidate,candidate_normals):
                nonlocal last_qref
                world=pose[:3]+candidate@rotation(pose).T
                seeds=[geometric[reference].copy(),last_qref.copy(),hand.data.qpos[hand.qadr].copy()]
                best=None
                for seed in seeds:
                    for _ in range(2):
                        data=hand.set_state(seed,pose)
                        robot_materials=[hand.nearest_material(contacts.materials[k],world[j],data,radius=args.hand_patch_radius) for j,k in enumerate(slots)]
                        seed,report=hand.solve_ik(seed,pose,robot_materials,world,
                            target_normals=candidate_normals@rotation(pose).T,iterations=args.contact_ik_iterations,tolerance=args.contact_ik_tolerance,
                            shape_materials=readiness_materials,shape_targets=target_shape,
                            additional_contact_bodies={contact['body'] for contact in actual},
                            surface_priors=[contacts.materials[k] for k in slots],surface_radius=args.hand_patch_radius)
                        robot_materials=report['materials']
                        report['q']=seed.copy()
                        report['material_offset']=float(sum(np.linalg.norm(m.local-contacts.materials[k].local) for k,m in zip(slots,robot_materials)))
                        if best is None or (not best['valid'] and report['valid']) or (best['valid']==report['valid'] and report['max_contact_error']<best['max_contact_error']):best=report
                        if report['valid']:break
                    if best['valid']:break
                report=best
                return report
            plan=dict(valid=True,source='ik_only',slots=slots,force_vectors_local=np.zeros((len(slots),3)),points=anchors[list(slots)],normals=inward[list(slots)],solve_time=0.)
            if mode=='full':
                candidate_pose=hand.set_state(geometric[reference],pose)
                geometric_points=hand.kinematics([materials[k] for k in slots],candidate_pose)[0]
                geometric_local=(geometric_points-pose[:3])@rotation(pose)
                geometric_projected,_,geometric_normals=hand.surface.query(geometric_local) if slots else (np.empty((0,3)),None,np.empty((0,3)))
                # Both current contacts and nearest projections are refreshed each
                # cycle; original-only feasibility remains the first solve.
                plan=policy.solve(anchors[list(slots)],inward[list(slots)],slots,current=pose,target=goal,
                    inertia=inertia,velocity=hand.data.qvel[hand.ov:hand.ov+6],external=external,dt=args.command_dt,
                    verify=verify,actual_points=actual_points[list(slots)],actual_normals=actual_normals[list(slots)],
                    actual_tangents=actual_tangents[list(slots)],
                    projected_points=projected[list(slots)],projected_normals=projection_normals[list(slots)],
                    seed_sets=[(geometric_projected,geometric_normals)],**support_parameters(hand))
                if plan['valid']:
                    failure_streak=0;last_qref=plan.get('ik',{}).get('q',geometric[reference]).copy()
                    for k,material in zip(slots,plan.get('ik',{}).get('materials',[])):runtime_materials[k]=material
                    materials=runtime_materials
                    anchors[list(slots)]=plan['points'];inward[list(slots)]=plan['normals'];desired[list(slots)]=plan['force_vectors_local']
                else:failure_streak+=1
            else:last_qref=geometric[reference].copy()
            plans.append(jsonable(plan))
            torque_violation=None;ideal_torque=[]
            if mode=='full' and slots and plan['valid']:
                witness=hand.set_state(last_qref,pose)
                _,_,witness_jac=hand.kinematics([materials[k] for k in slots],witness)
                world_force=desired[list(slots)]@rotation(pose).T
                ideal_torque=witness.qfrc_bias[hand.dadr]+np.einsum('ijk,ij->k',witness_jac,world_force)
                limited=hand.model.actuator_forcelimited.astype(bool)
                bounds=hand.model.actuator_forcerange
                torque_violation=bool(np.any(ideal_torque[limited]<bounds[limited,0]-1e-5) or np.any(ideal_torque[limited]>bounds[limited,1]+1e-5))
            near=bool(np.linalg.norm(q[:3]-geometric[reference,:3])<.02 and np.max(np.abs(q[6:]-geometric[reference,6:]))<.2)
            fraction=float(np.mean(np.linalg.norm(measured[list(slots)],axis=1)>.01)) if slots else 1.
            if phase=='approach' and near:phase='contact' if slots else 'track'
            if phase=='contact':
                stable=stable+1 if (fraction>=.5 or len(actual)>0) and plan['valid'] else 0
                if stable>=3:phase='track'
            if phase=='track' and slots!=last_slots and slots:
                phase='contact';stable=0
            if not slots and phase=='contact':phase='track'
            last_slots=slots
            if mode=='ik-only':
                # Baseline follows the same clock/progress, but has no contact planner.
                phase='track';target=np.clip(hand.servo_reference(geometric[reference]),hand.servo-command_limits,hand.servo+command_limits);mpc_report=dict(status='IK_SERVO',backend='position')
            elif not plan['valid']:
                target,mpc_report=mpc.plan(goal,geometric[reference],materials,anchors,contacts.anchors[reference],inward,active,
                    original_materials=contacts.materials,lambda_valid=False)
                mpc_report['status']='GEOMETRIC_TRACK_LAMBDA_INFEASIBLE; '+mpc_report['status']
            else:
                target,mpc_report=mpc.plan(goal,geometric[reference],materials,anchors,contacts.anchors[reference],inward,active,original_materials=contacts.materials,contact_qref=last_qref,approach=phase=='approach')
                if ('acados status' in mpc_report['status'] or mpc_report['selected_control']=='unsafe_hold') and not (output/'mpc_failure.npz').exists():
                    np.savez_compressed(output/'mpc_failure.npz',step=step,**mpc.last_problem)
            hand.execute(target,args.command_dt)
            pose_after=hand.object_pose();p_after,_,_=hand.kinematics(materials);force_after,actual_after=hand.observed_contacts(materials,active)
            actual_local=np.array([(c['point']-pose_after[:3])@rotation(pose_after) for c in actual_after]).reshape(-1,3)
            actual_inward=np.array([c['normal']@rotation(pose_after) for c in actual_after]).reshape(-1,3)
            from examples.retarget.contact_geometry import native_grasp_certificate
            certificate=native_grasp_certificate(hand,materials,active)
            # Measure original hand material and object-material errors independently.
            semantic_points=hand.kinematics(contacts.materials)[0]
            hand_error=np.linalg.norm(semantic_points[list(slots)]-(pose_after[:3]+contacts.anchors[reference,list(slots)]@rotation(pose_after).T),axis=1)
            object_error=[];slip=[]
            for c in actual_after:
                k=c['slot']
                if k in slots:
                    a=(c['point']-pose_after[:3])@rotation(pose_after)
                    object_error.append(float(np.linalg.norm(a-contacts.anchors[reference,k])))
                    if k in previous_actual:slip.append(float(np.linalg.norm(a-previous_actual[k])))
                    previous_actual[k]=a
            collision=hand.collision_report(hand.data,{materials[k].body for k in slots})
            record=dict(step=step,phase=phase,reference_frame=reference,matched_frame=index,object_reference_frame=target_index,object_pose=pose_after,
                geometric_q_reference=geometric[reference],contact_q_reference=last_qref,
                ideal_joint_torque=ideal_torque,ideal_torque_limit_violation=torque_violation,
                finger_tracking_error=float(np.sqrt(np.mean((hand.data.qpos[hand.qadr][6:]-geometric[reference,6:])**2))),
                wrist_tracking_error=float(np.linalg.norm(hand.data.qpos[hand.qadr][:3]-geometric[reference,:3])),
                contact_handoff='measured_physical_contact' if actual else 'pending',
                lambda_valid=plan['valid'],lambda_source=plan['source'],lambda_time=plan['solve_time'],mpc=mpc_report,
                hand_contact_error=hand_error,object_anchor_error=object_error,force_error=np.linalg.norm(force_after[list(slots)]-desired[list(slots)]@rotation(pose_after).T,axis=1),
                contact_fraction=fraction,active_contact_count=len(slots),actual_contact_count=len(actual_after),penetration=collision['max_penetration'],actual_force_closure=certificate,
                slip=slip,cycle_time=time.perf_counter()-started,actual_forces=force_after,nearest_projection=projected)
            if mode=='ik-only':record['force_error']=[]  # No Lambda force witness in this baseline.
            records.append(jsonable(record));log.write(json.dumps(jsonable(record),allow_nan=False)+'\n');log.flush()
            qhistory.append(hand.data.qpos[hand.qadr].copy());controls.append(target.copy());pose_history.append(pose_after)
            if viewer:viewer.sync()
            if writer:
                camera=mujoco.MjvCamera();camera.lookat[:]=demo.poses[reference,:3];camera.distance=.6;camera.azimuth=135;camera.elevation=-25
                renderer.update_scene(hand.data,camera);writer.append_data(renderer.render())
            if step%10==0:print(f'[{mode}] step={step} phase={phase} ref={reference} contacts={fraction:.2f} lambda={plan["source"]} status={mpc_report["status"]}',flush=True)
            from planning.wuji_retarget import tracking_task_complete
            finished_dwell=finished_dwell+1 if tracking_task_complete(reference,len(demo.poses),pose_after,demo.poses[-1]) else 0
            if finished_dwell>=5:break
            if args.stop_on_contact_failure and failure_streak>=args.max_infeasible_steps:break
    finally:
        log.close()
        if viewer:viewer.close()
        if writer:writer.close()
        if renderer:renderer.close()
    np.savez_compressed(output/'rollout.npz',hand_q=qhistory,controls=controls,object_pose=pose_history,
        reference_frame=[r['reference_frame'] for r in records],matched_frame=[r['matched_frame'] for r in records])
    write_json(output/'contact_plans.json',plans)
    report=metrics(hand,records,demo,contacts)
    report['termination_reason']='task_completed' if finished_dwell>=5 else ('repeated_lambda_infeasibility' if args.stop_on_contact_failure and failure_streak>=args.max_infeasible_steps else 'step_budget')
    report['object_task_completed']=finished_dwell>=5
    report['all_hand_frames_commanded']=len(set(r['reference_frame'] for r in records))==len(demo.poses)
    write_json(output/'metrics.json',report)
    return report


def mpc_limits(hand,args):return args.command_dt*np.r_[np.full(3,.15),np.full(3,.8),np.full(hand.nu-6,1.5)]


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path);p.add_argument('--asset-cache',type=Path,default=CACHE)
    p.add_argument('--dexforge-root',type=Path);p.add_argument('--sequence',type=Path);p.add_argument('--hand-assets',type=Path)
    p.add_argument('--mano-models',type=Path,default=Path('/home/zz/GraspSONIC/thirdparty/DexForge/assets/mano'))
    p.add_argument('--sync-assets',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'outputs/wuji_retarget')
    p.add_argument('--mode',choices=['ik-only','full'],default='full');p.add_argument('--compare',action='store_true')
    p.add_argument('--headless',action='store_true');p.add_argument('--video',action='store_true')
    p.add_argument('--mpc-solver',choices=['acados','ipopt'],default='acados');p.add_argument('--horizon',type=int,default=4)
    p.add_argument('--native-force-feedback',action='store_true',help='Verification: add a native finite-difference Gauss-Newton candidate; increases computation.')
    p.add_argument('--command-dt',type=float,default=.03);p.add_argument('--sim-dt',type=float,default=.01)
    p.add_argument('--fps',type=float,default=30.);p.add_argument('--subsample',type=int,default=1);p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--max-steps',type=int,default=600);p.add_argument('--max-infeasible-steps',type=int,default=20)
    p.add_argument('--stop-on-contact-failure',action='store_true',help='Stop after repeated infeasible contacts; otherwise continue bounded geometric hand tracking.')
    p.add_argument('--ik-iterations',type=int,default=60);p.add_argument('--contact-ik-iterations',type=int,default=50)
    p.add_argument('--hand-patch-radius',type=float,default=.03)
    p.add_argument('--hand-rate-scale',type=float,default=1.25,help='Minimum control rates are scaled peak geometric IK trajectory velocities.')
    p.add_argument('--contact-ik-tolerance',type=float,default=.002)
    p.add_argument('--ik-tolerance',type=float,default=.012);p.add_argument('--lambda-device',default='cpu')
    p.add_argument('--mpc-costs',type=json.loads,default={},help='MPC cost weights as JSON; also accepts a mapping in YAML config.')
    p.add_argument('--lambda-iterations',type=int,default=20);p.add_argument('--force-limit',type=float,default=10.)
    p.add_argument('--search-radii',type=float,nargs='+',default=[.005,.01]);p.add_argument('--fps-samples',type=int,default=8)
    p.add_argument('--candidate-capacity',type=int,default=8);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--reference-window',type=float,default=.5);p.add_argument('--lookahead',type=float,default=.06)
    p.add_argument('--prepare-only',action='store_true',help='Save semantic contacts and geometric IK without simulating.')
    p.add_argument('--include-mount-collisions',action='store_true',help='Diagnostic: retain upstream convex-hull mount intersections instead of structural exclusions.')
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.config:
        config=yaml.safe_load(args.config.read_text()) or {}
        unknown=set(config)-set(vars(args))
        if unknown:p.error(f'unknown config keys: {sorted(unknown)}')
        # Explicit CLI options override YAML; argparse defaults do not.
        explicit={token.split('=')[0] for token in (sys.argv[1:] if argv is None else argv) if token.startswith('--')}
        for key,value in config.items():
            if '--'+key.replace('_','-') not in explicit:setattr(args,key,value)
    from planning.wuji_retarget_mpc import WujiRetargetCosts
    try:WujiRetargetCosts(**args.mpc_costs)
    except (TypeError,ValueError) as exc:p.error(f'invalid mpc-costs: {exc}')
    for key in ('command_dt','sim_dt','fps','ik_tolerance','contact_ik_tolerance','force_limit','reference_window','hand_rate_scale'):
        if not np.isfinite(getattr(args,key)) or getattr(args,key)<=0:p.error(f'{key} must be positive and finite')
    if abs(round(args.command_dt/args.sim_dt)*args.sim_dt-args.command_dt)>1e-10:p.error('command-dt must be a multiple of sim-dt')
    if args.max_frames<0 or args.max_infeasible_steps<1 or args.lambda_iterations<1:p.error('invalid frame or iteration count')
    if not np.isfinite(args.hand_patch_radius) or args.hand_patch_radius<0:p.error('hand-patch-radius must be finite and nonnegative')
    if not np.isfinite(args.lookahead) or args.lookahead<0:p.error('lookahead must be finite and nonnegative')
    if min(args.horizon,args.subsample,args.max_steps,args.ik_iterations,args.contact_ik_iterations,args.fps_samples,args.candidate_capacity)<1:p.error('counts must be positive')
    if args.fps_samples>8 or args.candidate_capacity>64:p.error('local search supports at most 8 FPS points and 64 combinations per round')
    if any(not np.isfinite(r) or r<=0 for r in args.search_radii) or args.search_radii!=sorted(args.search_radii):p.error('search radii must be increasing positive values')
    args.asset_cache=Path(args.asset_cache);root=Path(args.dexforge_root or args.asset_cache/'DexForge')
    if args.sync_assets:
        from examples.retarget.sync_assets import sync_assets
        root=sync_assets(args.asset_cache)
    args.sequence=Path(args.sequence or root/'data/dexycb'/SEQUENCE)
    args.hand_assets=Path(args.hand_assets or root/'assets/wuji');args.mano_models=Path(args.mano_models);args.output=Path(args.output)
    args.output.mkdir(parents=True,exist_ok=True);write_json(args.output/'config.json',vars(args))
    demo=load_demonstration(args.sequence,root,args.mano_models,args.fps,args.subsample,args.max_frames)
    meta=json.loads((args.sequence/'mano_raw/meta.json').read_text())
    label=meta['object'].partition('_')[2] if meta['object'].partition('_')[0].isdigit() else meta['object']
    scene=args.sequence/'scene/leaphand'/f'{label}_mesh.xml';tree=ET.parse(scene).getroot()
    collider=next(g for g in tree.findall("worldbody/body[@name='obj']/geom") if g.get('contype','1')!='0')
    mesh=tree.find(f"asset/mesh[@name='{collider.get('mesh')}']")
    mesh_path=(scene.parent/mesh.get('file')).resolve()
    surface=ObjectSurface(mesh_path,args.seed,scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' '),convex=True)
    inertial=tree.find("worldbody/body[@name='obj']/inertial")
    center_mass=np.fromstring(inertial.get('pos','0 0 0'),sep=' ') if inertial is not None else surface.mesh.center_mass
    ground=stabilize_demonstration(demo,surface,center_mass)
    write_json(args.output/'ground_stabilization.json',ground)
    xml=build_scene(args.hand_assets/'right_mesh.xml',scene,args.output/'scene.xml',demo.poses[0],args.sim_dt,exclude_mounts=not args.include_mount_collisions)
    hand=WujiHand(xml,surface)
    write_json(args.output/'physics_parameters.json',physics_audit(hand,args.hand_assets/'right_mesh.xml',scene))
    print(f'[prepare] frames={len(demo.poses)} hand_dof={hand.nu}',flush=True)
    contacts=extract_contacts(demo,hand,args.hand_assets)
    print(f'[prepare] semantic episodes={len(contacts.episodes)} regions={sorted(set((e.finger,e.region) for e in contacts.episodes))}',flush=True)
    np.savez_compressed(args.output/'semantic_contacts.npz',mask=contacts.mask,anchors=contacts.anchors,normals=contacts.normals,
        hand_targets=contacts.hand_targets,hand_normals=contacts.hand_normals,
        body=[e.material.body for e in contacts.episodes],local=[e.material.local for e in contacts.episodes],
        source_local=[e.material.source_local for e in contacts.episodes],
        local_normal=[e.material.normal for e in contacts.episodes],finger=[e.finger for e in contacts.episodes],region=[e.region for e in contacts.episodes],
        mano_vertex=[e.vertex for e in contacts.episodes],start=[e.start for e in contacts.episodes],end=[e.end for e in contacts.episodes],
        object_pose=demo.poses,frame_ids=demo.frame_ids,times=demo.times)
    geometric,reports=geometric_ik(hand,demo,contacts,args)
    shape,targets=shape_references(hand,demo)
    np.savez_compressed(args.output/'geometric_pose_targets.npz',targets=targets,mano_joints=demo.joints,
        body=[m.body for m in shape],local=[m.local for m in shape],times=demo.times,frame_ids=demo.frame_ids)
    np.save(args.output/'wuji_q.npy',geometric);write_json(args.output/'ik_diagnostics.json',reports)
    if args.prepare_only:return args.output
    modes=['ik-only','full'] if args.compare else [args.mode]
    results={mode:run_rollout(hand,demo,contacts,geometric,args,mode,args.output/mode) for mode in modes}
    write_json(args.output/'comparison.json',results)
    print(f'[done] {args.output}',flush=True)
    return args.output


if __name__=='__main__':main()
