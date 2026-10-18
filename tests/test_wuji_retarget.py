"""Semantic identity, original-first selection, and real MuJoCo/Warp contracts."""
from pathlib import Path
from types import SimpleNamespace
import json

import mujoco
import numpy as np
import pytest
import trimesh

from models.wuji_hand import ObjectSurface,MaterialPoint,WujiHand,build_scene
from planning.wuji_retarget import (classify_vertices,xyzw_xyz_to_pose,ReferenceProgress,HandReferenceProgress,fps,LocalLambdaPolicy)


POSE=np.array([0.,0.,.2,1.,0.,0.,0.])


def test_source_pose_conversion_normalizes_without_mutating():
    source=np.array([0.,0.,0.,2.,1.,2.,3.]);result=xyzw_xyz_to_pose(source)
    np.testing.assert_array_equal(source,[0,0,0,2,1,2,3])
    np.testing.assert_allclose(result,[1,2,3,1,0,0,0])
    with pytest.raises(ValueError):xyzw_xyz_to_pose(np.zeros(7))


def test_geometric_closure_uses_gap_and_retains_finger_only_rate_bounds():
    from planning.wuji_retarget_mpc import geometric_closure_step
    jac=np.zeros((2,8));jac[0,6]=.04;jac[1,7]=-.03
    limits=np.full(8,.015)
    closing=geometric_closure_step(jac,np.array([0.,-.005]),.004,limits)
    assert closing[6]>0 and closing[7]==0
    np.testing.assert_array_equal(closing[:6],0.)
    assert np.all(np.abs(closing)<=limits)
    # Once geometric compression is sufficient, no further closing is added.
    np.testing.assert_array_equal(geometric_closure_step(jac,np.array([-.005,-.005]),.004,limits),0.)


def test_hand_clock_visits_every_frame_even_when_object_matching_stalls():
    times=np.arange(6)*.03;poses=np.tile(POSE,(6,1));poses[:,0]+=np.arange(6)*.02
    hand=HandReferenceProgress(times);obj=ReferenceProgress(poses,times,lookahead=0.)
    seen=[hand.index]
    for _ in range(5):
        frame=hand.update(.2)
        seen.append(frame)
        matched,_=obj.update(POSE,.03,minimum_index=frame)
        assert matched>=frame
    assert seen==list(range(6))
    before=hand.clock;hand.update(.2,ready=False)
    assert hand.clock==before


def test_end_of_hand_motion_does_not_report_unlifted_object_as_complete():
    from planning.wuji_retarget import tracking_task_complete
    goal=POSE.copy();goal[2]+=.17
    assert not tracking_task_complete(68,69,POSE,goal)
    assert not tracking_task_complete(67,69,goal,goal)
    assert tracking_task_complete(68,69,goal,goal)
    flipped=goal.copy();flipped[3:]*=-1
    assert tracking_task_complete(68,69,flipped,goal)


def test_hand_playback_retimes_measured_lag_without_rejecting_or_skipping_frames():
    from planning.wuji_retarget import hand_reference_rate
    reference=np.zeros(26);measured=reference.copy()
    assert hand_reference_rate(measured,reference)==1.
    measured[:3]=[.2,0,0];measured[6:]=.3
    rate=hand_reference_rate(measured,reference)
    assert 0<rate<.1
    progress=HandReferenceProgress(np.arange(4)*.03)
    seen=[0]
    for _ in range(60):
        index=progress.update(.03*rate)
        if index!=seen[-1]:seen.append(index)
    assert seen==[0,1,2,3]


def test_semantic_relaxation_proposes_regions_without_swapping_fingers_or_palm():
    from planning.wuji_retarget import execution_contact_subsets
    episodes=[SimpleNamespace(finger=f,region=r) for f,r in
        [('thumb','distal'),('thumb','middle'),('index','distal'),('index','proximal'),('palm','palm')]]
    contacts=SimpleNamespace(episodes=episodes,mask=np.ones((1,5),bool))
    proposals=execution_contact_subsets(contacts,0,np.array([.01,.001,.02,.002,.003]),np.array([0.,2.,0.,3.,0.]))
    assert proposals[0]==(0,2,4) and (1,3,4) in proposals
    for proposal in proposals:
        assert [episodes[k].finger for k in proposal]==['thumb','index','palm']
    assert len(proposals)<=3 and len(set(proposals))==len(proposals)


def test_wuji_commands_are_not_limited_by_demo_or_contact_speed(physical_hand):
    from planning.wuji_retarget_mpc import WujiMPCParams
    first=WujiMPCParams(physical_hand,3,rates=np.ones(physical_hand.nu))
    second=WujiMPCParams(physical_hand,3,rates=np.full(physical_hand.nu,2.))
    np.testing.assert_array_equal(first.mpc_u_ub_,physical_hand.upper-physical_hand.lower)
    np.testing.assert_array_equal(first.mpc_u_ub_,second.mpc_u_ub_)
    assert first.tag==second.tag


def test_partial_grasp_preserves_support_from_floor():
    import casadi as cs
    from models.allegro_grasp_response import force_limited_grasp_cs,grasp_operators
    # The pads cannot support gravity on their own. A stationary object has
    # already received floor support in SCM and must remain stationary after
    # replacing only the pad response.
    jac=np.zeros((8,8))
    for i,sign in enumerate([1.,-1.]):
        normal=np.array([sign,0,0]);t1=np.array([0,1.,0]);t2=np.array([0,0,1.])
        for k,d in enumerate([normal+t1,normal+t2,normal-t1,normal-t2]):
            jac[4*i+k,:3]=d;jac[4*i+k,6+i]=-sign
    vm,rj,gi,gm,internal=grasp_operators(jac,np.ones(2),np.eye(6),2)
    kwargs=dict(h=.03,friction=1.,velocity_map=cs.DM(vm),robot_jac=cs.DM(rj),object_mass=cs.DM.eye(6),
        grasp_inverse=cs.DM(gi),grasp_mobility=cs.DM(gm),internal_map=cs.DM(internal),
        reference_force=cs.DM([.1,0,0,-.1,0,0]),normal_limits=cs.DM([.1,.1]),other_wrench=cs.DM.zeros(6))
    velocity,_=force_limited_grasp_cs(cs.DM.zeros(8),kwargs['reference_force'],cs.DM.zeros(6),
        cs.DM.ones(2),cs.DM.eye(6),cs.DM([0,0,-1.,0,0,0]),cs.DM(jac),cs.DM.zeros(8),
        retain_other_contacts=True,**kwargs)
    np.testing.assert_allclose(np.asarray(velocity).ravel(),0.,atol=1e-10)


def test_floor_support_spans_the_face_and_releases_when_lifted(physical_hand):
    from planning.wuji_contact_geometry import contact_geometry
    hand=physical_hand;q=np.clip(np.zeros(hand.nu),hand.lower,hand.upper);q[:3]=[1.,1.,1.]
    pose=POSE.copy();pose[2]=-hand.surface.mesh.vertices[:,2].min()
    hand.set_state(q,pose,hand.data)
    phi,jac,*_=contact_geometry(hand,[],np.empty(0),12)
    rows=jac[:16].reshape(4,4,6+hand.nu).mean(axis=1)
    np.testing.assert_allclose(rows[:,:3],np.tile([0.,0.,1.],(4,1)))
    assert np.linalg.matrix_rank(rows[:,3:5]-rows[0,3:5])==2
    assert np.max(phi[:16])<.00501
    pose[2]+=.1;hand.set_state(q,pose,hand.data)
    lifted=contact_geometry(hand,[],np.empty(0),12)[0]
    assert np.min(lifted[:16])>=.1-1e-10


def test_semantic_finger_order_is_anatomical_not_mano_or_wuji_column_order():
    finger,region=classify_vertices(np.eye(16))
    assert list(finger[[0,1,4,7,10,13]])==['palm','index','middle','pinky','ring','thumb']
    assert list(region[1:4])==['proximal','middle','distal']


def test_reference_clock_advances_static_pose_without_jumping_or_reversing():
    poses=np.tile(POSE,(40,1));times=np.arange(40)*.03
    progress=ReferenceProgress(poses,times)
    i,t=progress.update(POSE,.09)
    assert i==3 and t>=5
    assert progress.update(POSE,1.,ready=False)==(3,t)
    old=progress.clock
    progress.update(POSE,.03)
    assert progress.index>=3 and progress.clock==pytest.approx(old+.03)


def test_reference_matches_only_forward_local_window_and_quaternion_sign():
    poses=np.tile(POSE,(50,1));poses[:,0]=np.arange(50)*.01
    progress=ReferenceProgress(poses,np.arange(50)*.03,window=.12)
    current=poses[40].copy();current[3:]*=-1
    i,_=progress.update(current,.03)
    assert i<=4
    i2,_=progress.update(poses[0],.03)
    assert i2>=i


def test_fps_is_bounded_unique_and_keeps_nearest_seed():
    points=np.c_[np.linspace(0,.005,30),np.zeros((30,2))]
    ids=fps(points,8,points[0])
    assert len(ids)==8 and len(set(ids))==8 and ids[0]==0
    assert fps(np.empty((0,3)),8,np.zeros(3)).size==0


class Buffer:
    def __init__(self,value):self.value=np.asarray(value)
    def numpy(self):return self.value


class FakeSolver:
    def __init__(self,cost):self.cost=cost;self.initials=[]
    def solve_device(self,points,normals,**kwargs):
        self.points=np.array(points);self.normals=np.array(normals);self.initials.append(kwargs['initial_forces'])
        return SimpleNamespace(valid=Buffer(np.ones(len(points))),pose_costs=Buffer([self.cost(p) for p in points]))
    def gather_selected(self,row,batch):
        n=len(self.points[row]);force=np.tile([.1,0,0],(n,1))
        return dict(points=self.points[row].copy(),normals=self.normals[row].copy(),control_input=force,
            force_vectors_local=.1*self.normals[row],x_plus_opt=POSE.copy(),cost=self.cost(self.points[row]),
            pose_reduction=-self.cost(self.points[row]))


@pytest.fixture
def policy():
    surface=ObjectSurface(trimesh.creation.box(extents=[.08,.08,.08]),samples=1000)
    p=LocalLambdaPolicy(surface,radii=(.005,.01),samples=8,capacity=64)
    fake=FakeSolver(lambda points:float(np.sum(points[:,1]**2)))
    p.solver=lambda n:fake
    p.fake=fake
    return p


def invoke(policy,verify,slots=(0,),anchors=None,**kwargs):
    if anchors is None:anchors=np.array([[.04,0.,0.]])
    return policy.solve(anchors,np.tile([-1.,0,0],(len(slots),1)),slots,
        current=POSE,target=POSE,inertia=np.eye(6),velocity=np.zeros(6),external=np.zeros(6),dt=.03,verify=verify,**kwargs)


def test_original_contacts_are_not_quantized_and_valid_pose_does_not_trigger_fps(policy,monkeypatch):
    import planning.wuji_retarget as module
    monkeypatch.setattr(module,'fps',lambda *a:pytest.fail('FPS must not run for a feasible original'))
    original=np.array([[.04,.0001234,.0004321]])
    result=invoke(policy,lambda p,n:dict(valid=True),anchors=original)
    np.testing.assert_array_equal(result['points'],original)
    assert result['source']=='mano_original' and not result['force_closure_certificate']['force_closure']
    # It is feasible even with a negative reduction or an imperfect pose cost.
    assert result['pose_reduction']<0


def test_exploration_rejects_unreachable_rows_and_selects_minimum_pose_cost(policy):
    seen=[]
    def check(p,n):
        seen.append(p.copy());return dict(valid=bool(abs(p[0,1])>.00001))
    result=invoke(policy,check)
    assert result['valid'] and result['source']=='local_fps'
    assert result['candidate_count']<=64
    assert np.linalg.norm(result['points'][0]-[.04,0,0])<=result['exploration_radius']+1e-12
    feasible=[p for p in seen[1:] if abs(p[0,1])>.00001]
    assert result['cost']==pytest.approx(min(policy.fake.cost(p) for p in feasible))


def test_contact_identity_controls_force_warm_start(policy):
    invoke(policy,lambda p,n:dict(valid=True),slots=(10,))
    invoke(policy,lambda p,n:dict(valid=True),slots=(10,))
    assert policy.fake.initials[-1] is not None
    invoke(policy,lambda p,n:dict(valid=True),slots=(11,))
    assert policy.fake.initials[-1] is None


def test_empty_contact_phase_has_no_force_solve(policy):
    result=invoke(policy,lambda p,n:pytest.fail('no IK for empty force phase'),slots=(),anchors=np.empty((0,3)))
    assert result['source']=='no_contact' and result['force_vectors_local'].shape==(0,3)
    assert not policy.fake.initials


def test_infeasible_incumbent_is_reverified_not_silently_retained(policy):
    invoke(policy,lambda p,n:dict(valid=True))
    result=invoke(policy,lambda p,n:dict(valid=False))
    assert not result['valid'] and result['source']=='infeasible'


def test_contact_plan_requires_native_closure_and_pose_descent_before_joint_ik(policy,monkeypatch):
    policy.require_closure_descent=True;policy.radii=()
    single=invoke(policy,lambda *a:pytest.fail('Nonclosed geometry must not run joint IK'))
    assert not single['valid'] and single['rejected_candidates'][0]['reason']=='not_force_closed'
    points=np.array([[.04,0,0],[-.04,.03,0],[-.04,-.03,0]])
    normals=np.array([[-1.,0,0],[1.,0,0],[1.,0,0]])
    original=policy.fake.gather_selected
    reduction=[-.1];stationary=[False]
    def gather(*a,**kw):
        result=original(*a,**kw);result['pose_reduction']=reduction[0]
        if stationary[0]:result.update(current_pose_cost=0.,pose_cost=0.)
        return result
    monkeypatch.setattr(policy.fake,'gather_selected',gather)
    def solve():
        return policy.solve(points,normals,(1,2,3),current=POSE,target=POSE,
            inertia=np.eye(6),velocity=np.zeros(6),external=np.zeros(6),dt=.03,
            verify=lambda *a:dict(valid=True))
    rejected=solve()
    assert not rejected['valid'] and rejected['rejected_candidates'][0]['reason']=='no_pose_cost_descent'
    reduction[0]=.1
    accepted=solve()
    assert accepted['valid'] and accepted['force_closure_certificate']['force_closure']
    assert accepted['pose_cost_decreased'] and accepted['source']=='mano_original'
    reduction[0]=0.;stationary[0]=True
    held=solve()
    assert held['valid'] and held['target_already_reached'] and not held['pose_cost_decreased']


def test_mpc_candidate_ranking_does_not_depend_on_force_certificates():
    from planning.wuji_retarget_mpc import select_contact_candidate
    hold=dict(source='hold',score=20.,certificate=dict(force_closure=False))
    cheap=dict(source='near_projection',score=1.,certificate=dict(force_closure=False,load_feasible=False),
        object_pose_reduction=-.1,object_pose_cost=1.)
    expensive=dict(source='certified',score=10.,certificate=dict(force_closure=True,load_feasible=True),
        object_pose_reduction=.1,object_pose_cost=.5)
    for acquiring in (False,True):
        chosen,status=select_contact_candidate([hold,cheap,expensive],20.,acquiring=acquiring)
        assert chosen is cheap and status=='native_cost_descent'
        cheap['certificate']=dict(force_closure=True,load_feasible=True)
        expensive['certificate']=dict(force_closure=False,load_feasible=False)
        assert select_contact_candidate([hold,cheap,expensive],20.,acquiring=acquiring)[0] is cheap
    worse=dict(source='worse',score=21.)
    assert select_contact_candidate([hold,worse],20.)[0] is hold


def test_real_lambda_original_force_obeys_coulomb_cone_and_limit():
    surface=ObjectSurface(trimesh.creation.box(extents=[.08,.08,.08]),samples=30)
    policy=LocalLambdaPolicy(surface,device='cpu',iterations=3,force_limit=2.)
    result=invoke(policy,lambda p,n:dict(valid=True))
    force=result['control_input']
    assert force.shape==(1,3) and force[0,0]>=0
    assert np.linalg.norm(force[0,1:])<=.5*force[0,0]+1e-6
    assert np.linalg.norm(force[0])<=2.+1e-6


@pytest.fixture(scope='module')
def physical_hand(tmp_path_factory):
    root=Path(__file__).resolve().parents[1]/'thirdparty/retarget_assets/DexForge'
    if not (root/'assets/wuji/right_mesh.xml').is_file():pytest.skip('run retarget.py --sync-assets for asset integration tests')
    seq=root/'data/dexycb/dexycb_004_sugar_box_20200709_142802'
    surface=ObjectSurface(seq/'model/model.obj',samples=100)
    pose=POSE.copy()
    xml=build_scene(root/'assets/wuji/right_mesh.xml',seq/'scene/leaphand/sugar_box_mesh.xml',
        tmp_path_factory.mktemp('wuji')/'scene.xml',pose)
    return WujiHand(xml,surface)


def test_dexforge_physics_reference_preserves_mesh_and_copies_object_and_floor(physical_hand,tmp_path):
    root=Path(__file__).resolve().parents[1]/'thirdparty/retarget_assets/DexForge'
    reference=tmp_path/'sdf.xml'
    reference.write_text('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="1 1 .05" friction=".9 .003 .0002" condim="6"/>
      <body name="obj"><joint type="free" armature=".0001" frictionloss="0"/>
      <inertial mass=".2" pos="0 0 0" diaginertia=".001 .002 .003"/>
      <geom type="sdfmesh" file="unused.obj" friction="1 .005 .0001" priority="1" condim="3"/>
      </body></worldbody></mujoco>''')
    xml=build_scene(root/'assets/wuji/right_mesh.xml',
        root/'data/dexycb/dexycb_004_sugar_box_20200709_142802/scene/leaphand/sugar_box_mesh.xml',
        tmp_path/'matched.xml',POSE,physics_reference=reference)
    matched=WujiHand(xml,physical_hand.surface)
    assert matched.model.body_mass[matched.obj]==pytest.approx(.2)
    np.testing.assert_allclose(matched.model.body_inertia[matched.obj],[.001,.002,.003])
    assert matched.friction==pytest.approx(1.)
    floor=mujoco.mj_name2id(matched.model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    np.testing.assert_allclose(matched.model.geom_friction[floor],[.9,.003,.0002])
    assert matched.model.geom_condim[floor]==6
    np.testing.assert_array_equal(matched.model.mesh_vert,physical_hand.model.mesh_vert)
    np.testing.assert_array_equal(matched.model.actuator_gainprm,physical_hand.model.actuator_gainprm)


def test_material_jacobian_matches_fk_finite_difference(physical_hand):
    hand=physical_hand
    body=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,'right_finger2_link4')
    point=MaterialPoint(body,np.array([-.004,0,.02]),np.array([-1.,0,0]),'index','distal')
    q=np.zeros(hand.nu);q[6:]=np.clip(.1,hand.lower[6:]+.001,hand.upper[6:]-.001)
    data=hand.set_state(q,POSE)
    positions,_,jac=hand.kinematics([point],data)
    step=np.zeros(hand.nu);step[12]=1e-6
    shifted=hand.kinematics([point],hand.set_state(q+step,POSE))[0]
    np.testing.assert_allclose((shifted-positions)/1e-6,jac[:,:,12],atol=1e-6)


def test_kinematic_collision_evaluation_matches_full_mujoco_forward(physical_hand):
    hand=physical_hand;q=np.clip(np.zeros(hand.nu),hand.lower,hand.upper)
    full=hand.set_state(q,POSE)
    positions=full.xpos.copy();rotations=full.xmat.copy()
    contacts=[(c.geom1,c.geom2,float(c.dist)) for c in full.contact]
    fast=hand.set_state(q,POSE,kinematic=True)
    np.testing.assert_array_equal(fast.xpos,positions)
    np.testing.assert_array_equal(fast.xmat,rotations)
    assert [(c.geom1,c.geom2,float(c.dist)) for c in fast.contact]==contacts


def test_real_mpc_state_force_and_inactive_contact_contract(physical_hand):
    from planning.wuji_retarget_mpc import WujiOnlineMPC
    hand=physical_hand;absent=POSE.copy();absent[0]+=2.
    hand.set_state(np.zeros(hand.nu),absent,hand.data);hand.servo=np.zeros(hand.nu)
    body=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,'right_finger2_link4')
    material=MaterialPoint(body,np.array([-.004,0,.02]),np.array([-1.,0,0]),'index','distal')
    mpc=WujiOnlineMPC(hand,1,dt=.03,horizon=1,solver='ipopt')
    target,report=mpc.plan(absent,hand.servo,[material],np.zeros((1,3)),np.zeros((1,3)),np.array([[-1.,0,0]]),np.zeros(1),approach=True)
    assert target.shape==(26,) and np.isfinite(target).all()
    np.testing.assert_allclose(report['predicted_force'],0.,atol=1e-12)
    # A requested but not yet made contact must not masquerade as loaded.
    _,requested=mpc.plan(absent,hand.servo,[material],np.zeros((1,3)),np.zeros((1,3)),
        np.array([[-1.,0,0]]),np.ones(1),lambda_valid=False)
    assert requested['made_contact_count']==0 and requested['force_source']=='contact_dynamics'
    before=hand.data.xfrc_applied[hand.obj].copy();hand.execute(target,.03)
    np.testing.assert_array_equal(hand.data.xfrc_applied[hand.obj],before)
    assert hand.data.qfrc_applied[hand.ov:hand.ov+6].sum()==0


def test_ground_stabilization_preserves_object_local_hand_geometry():
    from planning.wuji_retarget import stabilize_demonstration
    from models.wuji_hand import rotation
    surface=ObjectSurface(trimesh.creation.box(extents=[.05,.08,.18]),samples=30)
    from scipy.spatial.transform import Rotation
    q=Rotation.from_euler('xyz',[.1,.03,.4]).as_quat()[[3,0,1,2]]
    pose=np.r_[[.2,.1,.12],q]
    vertices=np.array([[[.22,.1,.2],[.25,.2,.1]]])
    demo=SimpleNamespace(poses=pose[None].copy(),vertices=vertices.copy(),joints=vertices.copy())
    before=(vertices[0]-pose[:3])@rotation(pose)
    report=stabilize_demonstration(demo,surface)
    np.testing.assert_allclose((demo.vertices[0]-demo.poses[0,:3])@rotation(demo.poses[0]),before,atol=1e-12)
    z=(surface.mesh.vertices@rotation(demo.poses[0]).T+demo.poses[0,:3])[:,2]
    assert z.min()==pytest.approx(0,abs=1e-12) and report['rotation_angle_rad']<.2


def test_contact_material_maps_palmar_surface_and_episodes_are_stable(physical_hand):
    from planning.wuji_retarget import load_demonstration,extract_contacts
    root=Path(__file__).resolve().parents[1]/'thirdparty/retarget_assets/DexForge'
    mano=Path('/home/zz/GraspSONIC/thirdparty/DexForge/assets/mano')
    if not (mano/'MANO_RIGHT.pkl').is_file():pytest.skip('local MANO model unavailable')
    demo=load_demonstration(root/'data/dexycb/dexycb_004_sugar_box_20200709_142802',root,mano,subsample=4)
    contacts=extract_contacts(demo,physical_hand,root/'assets/wuji')
    assert contacts.mask.shape==(len(demo.poses),len(contacts.episodes))
    assert contacts.mask.any()
    for k,episode in enumerate(contacts.episodes):
        assert np.flatnonzero(contacts.mask[:,k]).tolist()==list(range(episode.start,episode.end))
        assert episode.material.finger==episode.finger and episode.material.region==episode.region
        # Proximal links rotate in a different frame from middle/distal pads.
        # Finger identity and the physical surface are invariant to that frame.
        assert np.linalg.norm(episode.material.normal)==pytest.approx(1.)


def test_near_static_reconstruction_noise_does_not_stall_progress():
    poses=np.tile(POSE,(30,1));poses[:,0]+=np.arange(30)*.00001
    progress=ReferenceProgress(poses,np.arange(30)*.03)
    i,_=progress.update(POSE,.3)
    assert i==10


def test_shape_is_a_running_soft_preference_without_terminal_servo_depth(physical_hand):
    from planning.wuji_retarget_mpc import WujiOnlineMPC
    hand=physical_hand;hand.set_state(np.zeros(hand.nu),POSE,hand.data);hand.servo=np.zeros(hand.nu)
    mpc=WujiOnlineMPC(hand,0,dt=.03,horizon=1,solver='ipopt')
    mpc.plan(POSE,hand.servo,[],np.empty((0,3)),np.empty((0,3)),np.empty((0,3)),np.empty(0),approach=True)
    # Isolate posture from collision penalties, which should depend on q.
    cp=mpc.current_cp.copy();cp[-8*(1+hand.nu):]=np.c_[np.ones(8),np.zeros((8,hand.nu))].ravel()
    state=mpc.state();baseline=float(mpc.final_cost_fn(state,cp))
    running=float(mpc.path_cost_fn(state,np.zeros(hand.nu),cp))
    state[-hand.nu+1]+=.02
    assert float(mpc.final_cost_fn(state,cp))==pytest.approx(baseline)
    assert float(mpc.path_cost_fn(state,np.zeros(hand.nu),cp))==pytest.approx(running)
    state[8]+=.20
    assert float(mpc.path_cost_fn(state,np.zeros(hand.nu),cp))>running
    assert float(mpc.final_cost_fn(state,cp))==pytest.approx(baseline)


def test_posture_tracking_remains_active_when_all_semantic_contacts_are_made(physical_hand):
    from planning.wuji_retarget_mpc import WujiOnlineMPC
    hand=physical_hand;hand.set_state(np.zeros(hand.nu),POSE,hand.data);hand.servo=np.zeros(hand.nu)
    body=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,'right_finger2_link4')
    material=MaterialPoint(body,np.array([.004,0,.02]),np.array([1.,0,0]),'index','distal')
    mpc=WujiOnlineMPC(hand,1,horizon=1,solver='ipopt')
    mpc.plan(POSE,hand.servo,[material],np.zeros((1,3)),np.zeros((1,3)),
        np.array([[-1.,0,0]]),np.zeros(1),approach=True)
    cp=mpc.current_cp.copy()
    state=mpc.state();state[-hand.nu:]=cp[7+hand.nu:7+2*hand.nu]
    state[7+6]+=.1
    # Attraction is just before the semantic mask and force linearization.
    cp[mpc.attraction_index]=0.
    loaded=np.asarray(mpc.residual_fn(state,np.zeros(hand.nu),cp)).ravel()[mpc.posture_residual_slice]
    cp[mpc.attraction_index]=1.
    unloaded=np.asarray(mpc.residual_fn(state,np.zeros(hand.nu),cp)).ravel()[mpc.posture_residual_slice]
    assert loaded[6]>0
    assert loaded[6]==pytest.approx(unloaded[6]*np.sqrt(mpc.weights.posture_contact_scale))


def test_fa_normalized_object_cost_and_terminal_task_objective(physical_hand):
    from planning.wuji_retarget_mpc import WujiOnlineMPC,WujiRetargetCosts
    with pytest.raises(ValueError):WujiRetargetCosts(posture_contact_scale=0.)
    hand=physical_hand;hand.set_state(np.zeros(hand.nu),POSE,hand.data);hand.servo=np.zeros(hand.nu)
    mpc=WujiOnlineMPC(hand,0,horizon=1,solver='ipopt')
    empty=np.empty((0,3))
    mpc.plan(POSE,hand.servo,[],empty,empty,empty,np.empty(0))
    state=mpc.state();cp=mpc.current_cp
    baseline=float(mpc.path_cost_fn(state,np.zeros(hand.nu),cp))
    for axis,weight in [(0,1.),(1,1.),(2,1.)]:
        shifted=state.copy();shifted[axis]+=.002
        assert float(mpc.path_cost_fn(shifted,np.zeros(hand.nu),cp))-baseline==pytest.approx(weight*.002**2)
        assert float(mpc.final_cost_fn(shifted,cp))-float(mpc.final_cost_fn(state,cp))==pytest.approx(2*weight*.002**2)
    outside=state.copy();outside[0]+=.02
    assert float(mpc.path_cost_fn(outside,np.zeros(hand.nu),cp))-baseline==pytest.approx(.02**2)


def test_measured_contact_survives_lambda_failure_and_reference_release(physical_hand,monkeypatch):
    from planning.wuji_contact_geometry import contact_geometry
    from planning import grasp_force_allocation
    from planning.wuji_retarget_mpc import WujiOnlineMPC
    hand=physical_hand;hand.set_state(np.zeros(hand.nu),POSE,hand.data);hand.servo=np.zeros(hand.nu)
    body=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,'right_finger2_link4')
    m=MaterialPoint(body,np.array([.004,0,.02]),np.array([1.,0,0]),'index','distal')
    point=hand.kinematics([m])[0][0]
    force=np.array([2.,0,0])
    actual=[dict(slot=0,body=body,point=point,normal=np.array([1.,0,0]),force=force,gap=-.0001)]
    monkeypatch.setattr(hand,'observed_contacts',lambda *args,**kwargs:(force[None].copy(),actual))
    monkeypatch.setattr(grasp_force_allocation,'normal_pressure_command',lambda *a,**kw:pytest.fail('MPC must choose its own normal motion'))
    phi,jac,_,_,_,stick,_=contact_geometry(hand,[m],np.zeros(1),10)
    assert stick[0]==1 and np.linalg.norm(jac[:4])>0 and phi[0]<0
    mpc=WujiOnlineMPC(hand,1,horizon=1,solver='ipopt')
    monkeypatch.setattr(mpc,'plan_once',lambda **kwargs:dict(action=np.zeros(hand.nu),u_traj=np.zeros((1,hand.nu)),solve_status='test'))
    _,report=mpc.plan(POSE,hand.servo,[m],np.zeros((1,3)),np.zeros((1,3)),
        np.array([[1.,0,0]]),np.zeros(1),lambda_valid=False)
    assert report['made_contact_count']==1
    assert report['force_source']=='contact_dynamics' and not report['external_wrench_target']
    assert not report['lambda_valid']
    anchor=np.asarray(report['execution_anchors']).copy()
    # Apparent contact-patch sliding does not reset a made material anchor.
    actual[0]['point']=point+np.array([0,.005,0])
    _,next_report=mpc.plan(POSE,hand.servo,[m],np.zeros((1,3)),np.zeros((1,3)),
        np.array([[1.,0,0]]),np.zeros(1),lambda_valid=False)
    np.testing.assert_array_equal(next_report['execution_anchors'],anchor)
    # An offline/online Lambda proposal may name another material on the same
    # link. It must not replace the made patch's cost geometry or reset its
    # sticking anchor; original semantic correspondence remains separate.
    _,lambda_report=mpc.plan(POSE,hand.servo,[m],np.array([[.02,.01,0]]),np.zeros((1,3)),
        np.array([[1.,0,0]]),np.ones(1),lambda_valid=True)
    np.testing.assert_array_equal(lambda_report['execution_anchors'],anchor)
    assert lambda_report['planned_material_contact_mask']==[1.]
    np.testing.assert_allclose(lambda_report['execution_materials'][0]['local'],
        lambda_report['physics_materials'][0]['local'])
    _,scm_rows,*_=contact_geometry(hand,[m],np.zeros(1),10)
    np.testing.assert_allclose(mpc.physics.projection,np.eye(hand.nu),atol=1e-12)
    np.testing.assert_allclose(mpc.physics.pressure_bias[:6],0.,atol=1e-12)


def test_distance_grip_tracks_lift_without_force_or_load_permission(physical_hand,monkeypatch):
    from planning import grasp_force_allocation,wuji_load_support
    from planning.wuji_retarget_mpc import WujiOnlineMPC
    hand=physical_hand;hand.set_state(np.zeros(hand.nu),POSE,hand.data);hand.servo=np.zeros(hand.nu)
    bodies=[mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,f'right_finger{i}_link4') for i in (2,3)]
    materials=[MaterialPoint(b,np.zeros(3),np.array([1.,0,0]),name,'distal') for b,name in zip(bodies,['index','middle'])]
    normals=np.array([[-1.,0,0],[1.,0,0]])
    points=POSE[:3]+np.array([[.025,0,0],[-.025,0,0]])
    measured=.05*normals
    actual=[dict(slot=i,body=bodies[i],point=points[i],normal=normals[i],force=measured[i],gap=-.0001) for i in range(2)]
    monkeypatch.setattr(hand,'observed_contacts',lambda *args,**kwargs:(measured.copy(),actual))
    monkeypatch.setattr(grasp_force_allocation,'allocate_contact_wrench',lambda *a,**kw:pytest.fail('No wrench allocator in distance grip'))
    monkeypatch.setattr(wuji_load_support,'static_preload_seed',lambda *a,**kw:pytest.fail('No load-feasibility judgement in distance grip'))
    mpc=WujiOnlineMPC(hand,2,horizon=1,solver='ipopt')
    monkeypatch.setattr(mpc,'plan_once',lambda **kwargs:dict(action=np.zeros(hand.nu),u_traj=np.zeros((1,hand.nu)),solve_status='test'))
    goal=POSE.copy();goal[2]+=.02
    _,report=mpc.plan(goal,hand.servo,materials,np.zeros((2,3)),np.zeros((2,3)),normals,
        np.ones(2),lambda_valid=False)
    assert report['made_contact_count']==2 and report['grasp_response']
    assert not report['grasp_certificate']['load_feasible']
    assert report['transport_active'] and not report['force_certificates_affect_control']
    np.testing.assert_array_equal(report['executed_object_goal'],goal)
    assert report['closure_depth_m']==pytest.approx(.004)
    assert mpc.param_.contact_force_limit==100.
    assert not report['internal_pressure_control']
    assert report['geometric_gap_closure']
    np.testing.assert_allclose(mpc.physics.projection,np.eye(hand.nu),atol=1e-12)
    assert np.isfinite(mpc.physics.pressure_bias).all()
    assert report['candidate_cost_physics']=='native_MuJoCo'
    # Measured patches still calibrate physical force prediction; they do not
    # turn into force targets or transport permission.
    adopted=hand.kinematics(mpc.physical_materials)[0]
    np.testing.assert_allclose(adopted,points+.00005*normals,atol=1e-12)
    offset=mpc.force_offset+mpc.param_.n_qpos_+hand.nu
    np.testing.assert_allclose(mpc.current_cp[offset:offset+6],measured.reshape(-1))
    low=mpc.current_cp.copy();low[offset+6:-8*(1+hand.nu)]=0.
    high=low.copy();high[offset:offset+6]=(6.*normals).reshape(-1)
    assert float(mpc.final_cost_fn(mpc.state(),high))==pytest.approx(float(mpc.final_cost_fn(mpc.state(),low)))
    # A collider already on the surface need not penetrate the object. Moving
    # its virtual servo target inward must improve the closing objective
    # while the measured hand and object states remain fixed.
    state=mpc.state();state[-hand.nu:]=state[7:7+hand.nu]
    cp=mpc.current_cp.copy()
    rows=[]
    for patch in mpc.physical_materials:
        _,_,jac=hand.kinematics([patch]);rows.append(jac[0])
    normal_jac=np.einsum('ij,ijk->ik',normals,np.asarray(rows))[:,6:]
    step=np.linalg.lstsq(normal_jac,np.full(2,mpc.closure_depth),rcond=None)[0]
    closed=state.copy();closed[-hand.nu+6:]+=step
    assert float(mpc.final_cost_fn(closed,cp))<float(mpc.final_cost_fn(state,cp))
    np.testing.assert_array_equal(closed[:-hand.nu],state[:-hand.nu])
    limits=np.asarray(report['effective_command_limits'])
    np.testing.assert_allclose(limits,hand.upper-hand.lower)
    assert not report['rate_limits_enabled']
    assert any(c['source']=='closure_seed' for c in report['candidate_scores'])
    # Torque saturation can leave a large target/measured angle difference.
    # Closing distance must use FK at the target, not extrapolate a measured-q
    # Jacobian across that difference (which can reverse open vs closed).
    before_servo=hand.servo.copy()
    hand.servo[10:14]=[.4,.1,.7,.3]
    hand.servo[14:18]=[.6,0.,.5,.4]
    _,lagged=mpc.plan(goal,before_servo,materials,np.zeros((2,3)),np.zeros((2,3)),normals,
        np.ones(2),lambda_valid=False)
    adopted=[MaterialPoint(m['body'],np.array(m['local']),np.array(m['normal']),materials[k].finger,'distal')
        for k,m in enumerate(lagged['execution_materials'])]
    sp=hand.kinematics(adopted,hand.set_state(hand.servo,POSE,kinematic=True))[0]
    ap,_,aj=hand.kinematics(adopted)
    wrong=ap+(aj.reshape(-1,hand.nu)@(hand.servo-hand.data.qpos[hand.qadr])).reshape(-1,3)
    assert np.max(np.linalg.norm(wrong-sp,axis=1))>.001
    values=np.asarray(mpc.residual_fn(mpc.state(),np.zeros(hand.nu),mpc.current_cp)).ravel()
    world=POSE[:3]+np.asarray(lagged['execution_anchors'])
    for k,sl in enumerate(mpc.closure_residual_slices):
        expected=np.sqrt(mpc.weights.closure)*min(normals[k]@(sp[k]-world[k])-mpc.closure_depth,0.)
        np.testing.assert_allclose(values[sl],[expected],atol=1e-10)
    # Compression is a runtime reference, independent of collision tolerance;
    # tightening it must not require rebuilding the MPC dynamics/cost graph.
    deeper=mpc.current_cp.copy();deeper[mpc.closure_depth_index]=.012
    deep_values=np.asarray(mpc.residual_fn(mpc.state(),np.zeros(hand.nu),deeper)).ravel()
    for k,sl in enumerate(mpc.closure_residual_slices):
        expected=np.sqrt(mpc.weights.closure)*min(normals[k]@(sp[k]-world[k])-.012,0.)
        np.testing.assert_allclose(deep_values[sl],[expected],atol=1e-10)
    hand.servo=before_servo
    # Holding preserves the actual servo target exactly.
    import planning.wuji_retarget_mpc as module
    monkeypatch.setattr(module,'select_contact_candidate',lambda candidates,cost,**kwargs:
        (next(c for c in candidates if c['source']=='hold'),'hold_no_cost_improvement'))
    target,_=mpc.plan(goal,hand.servo,materials,np.zeros((2,3)),np.zeros((2,3)),normals,
        np.ones(2),lambda_valid=False)
    np.testing.assert_allclose(target,np.clip(hand.servo,hand.lower,hand.upper),atol=1e-12)
    # A changed object goal follows immediately, even after repeated weak touch.
    goal[2]+=.01
    _,changed=mpc.plan(goal,hand.servo,materials,np.zeros((2,3)),np.zeros((2,3)),normals,
        np.ones(2),lambda_valid=False)
    np.testing.assert_array_equal(changed['executed_object_goal'],goal)
    # Native refinement also works with force costs disabled: it refines the
    # same pose/servo-distance objective, rather than requiring load feedback.
    mpc.native_force_feedback=True
    _,refined=mpc.plan(goal,hand.servo,materials,np.zeros((2,3)),np.zeros((2,3)),normals,
        np.ones(2),lambda_valid=False)
    assert mpc.weights.load_balance==0 and refined['native_force_sensitivity_seed']
    mpc.native_force_feedback=False
    _,released=mpc.plan(goal,hand.servo,materials,np.zeros((2,3)),np.zeros((2,3)),normals,
        np.zeros(2),lambda_valid=False)
    assert not released['transport_active'] and released['grasp_response']


def test_online_recovery_unloads_without_erasing_actual_contact_dynamics(physical_hand,monkeypatch):
    from planning.wuji_retarget_mpc import WujiOnlineMPC
    from planning.wuji_regrasp import RegraspSettings
    hand=physical_hand;pose=POSE.copy();pose[2]=-hand.surface.mesh.vertices[:,2].min()+.0001
    hand.set_state(np.zeros(hand.nu),pose,hand.data);hand.servo=np.zeros(hand.nu)
    bodies=[mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,f'right_finger{i}_link4') for i in (1,2)]
    normals=np.array([[-1.,0,0],[1.,0,0]])
    points=pose[:3]+np.array([[.025,0,0],[-.025,0,0]])
    materials=[MaterialPoint(b,np.zeros(3),normals[k],name,'distal') for k,(b,name) in enumerate(zip(bodies,['thumb','index']))]
    forces=np.array([.02,.05])[:,None]*normals
    actual=[dict(slot=k,body=b,point=points[k],normal=normals[k],force=forces[k],gap=-.0001,
        tangents=np.array([[0,1.,0],[0,0,1.]])) for k,b in enumerate(bodies)]
    monkeypatch.setattr(hand,'observed_contacts',lambda *a,**kw:(forces.copy(),actual))
    mpc=WujiOnlineMPC(hand,2,horizon=1,solver='ipopt',regrasp=True)
    mpc.regrasp.settings=RegraspSettings(failure_cycles=1)
    monkeypatch.setattr(mpc,'plan_once',lambda **kw:dict(action=np.zeros(hand.nu),u_traj=np.zeros((1,hand.nu)),solve_status='test'))
    original=(points-pose[:3]).copy();original[0,1]+=.01
    for _ in range(2):
        _,report=mpc.plan(pose,hand.servo,materials,points-pose[:3],original,normals,np.ones(2),lambda_valid=False)
    assert report['regrasp']['phase']=='release'
    assert report['regrasp']['target_source']=='MANO_object_anchor'
    assert report['made_contact_count']==2 and report['physical_material_count']==2
    assert report['release_mask']==[1.,0.] and not report['external_wrench_target']
    assert report['load_objective_scale']==pytest.approx(.1)
    assert np.linalg.norm(mpc.last_problem['jacobian'][:4])>0
    np.testing.assert_allclose(report['regrasp']['target'],original[0])
    assert report['internal_pressure_slots']==[]  # Released finger is never re-pressurized.
    cp=mpc.current_cp.copy();start=mpc.force_offset+mpc.param_.n_qpos_+hand.nu
    cp[start+6:-8*(1+hand.nu)]=0.
    loaded=cp.copy();loaded[start:start+3]=normals[0]
    unloaded=cp.copy();unloaded[start:start+3]=0.
    assert float(mpc.final_cost_fn(mpc.state(),unloaded))==pytest.approx(float(mpc.final_cost_fn(mpc.state(),loaded)))
    opening=cp.copy()
    # Place the virtual target inside the surface to model a compressed grip.
    offset=mpc.servo_kinematics_offset+hand.nu
    opening[offset:offset+3]+=.02*normals[0]
    released=np.asarray(mpc.residual_fn(mpc.state(),np.zeros(hand.nu),opening)).ravel()
    gripping=opening.copy();gripping[mpc.release_mask_index]=0.
    gripping=np.asarray(mpc.residual_fn(mpc.state(),np.zeros(hand.nu),gripping)).ravel()
    # A geometric release must not also demand inward spring compression.
    sl=mpc.closure_residual_slices[0]
    assert released[sl][0]>0 and gripping[sl][0]==pytest.approx(0.)


def test_native_pyramid_certificate_uses_actual_tangent_axes():
    from planning.grasp_verification import verify_grasp
    points=np.zeros((1,3));normals=np.array([[0.,0,1.]])
    external=np.array([-.6,-.6,-1.,0,0,0])
    aligned=np.array([[[1.,0,0],[0,1.,0]]])
    diagonal=np.array([[[1.,1.,0],[-1.,1.,0]]])/np.sqrt(2.)
    a=verify_grasp(points,normals,1.,edges=4,tangent_directions=aligned,
        external_wrench=external,normal_force_limits=[1.])
    b=verify_grasp(points,normals,1.,edges=4,tangent_directions=diagonal,
        external_wrench=external,normal_force_limits=[1.])
    assert not a['load_feasible'] and b['load_feasible']
    with pytest.raises(ValueError):verify_grasp(points,normals,1.,tangent_directions=np.ones((1,2,3)))


def test_contact_prediction_uses_observed_native_frame_not_an_arbitrary_rotation(physical_hand,monkeypatch):
    from planning.wuji_contact_geometry import contact_geometry
    hand=physical_hand;hand.set_state(np.zeros(hand.nu),POSE,hand.data)
    body=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,'right_finger2_link4')
    material=MaterialPoint(body,np.zeros(3),np.array([1.,0,0]),'index','distal')
    tangents=np.array([[0.,1.,1.],[0.,-1.,1.]])/np.sqrt(2.)
    actual=[dict(slot=0,body=body,point=POSE[:3]+[.03,0,0],normal=np.array([1.,0,0]),
        force=np.array([1.,0,0]),tangents=tangents,gap=-.0001)]
    monkeypatch.setattr(hand,'observed_contacts',lambda *a,**kwargs:(np.array([[1.,0,0]]),actual))
    _,rows,*_=contact_geometry(hand,[material],[1.],10)
    predicted=np.array([rows[0,:3]-rows[2,:3],rows[1,:3]-rows[3,:3]])/(2*hand.friction)
    np.testing.assert_allclose(predicted,tangents,atol=1e-12)


def test_runtime_native_scm_impulses_have_nonnegative_mujoco_ray_representation():
    import casadi as cs
    from scipy.optimize import linprog
    from models.soft_scm_contact import grouped_contact_velocity_cs,soft_scm_coefficients
    mu=.5;rows=np.array([[mu,0,1.],[0,mu,1.],[-mu,0,1.],[0,-mu,1.]])
    runtime_mu=cs.SX.sym('runtime_mu',1);free=cs.SX.sym('free',3);a,b=soft_scm_coefficients(.03)
    _,impulses=grouped_contact_velocity_cs(free,cs.DM(rows),cs.DM.zeros(4),cs.DM.eye(3),
        .03,a,b,1e-8,return_impulses=True,friction=runtime_mu,friction_cone='pyramidal')
    fn,ft=impulses[0];evaluate=cs.Function('native_impulses',[free,runtime_mu],[cs.vertcat(ft,fn)])
    for velocity in ([10,10,-1],[-10,10,-1],[.1,.2,-1],[10,10,1]):
        force=np.asarray(evaluate(velocity,mu)).ravel()
        represented=linprog(np.ones(4),A_eq=rows.T,b_eq=force,bounds=(0,None),method='highs')
        assert represented.success


def test_anatomical_surface_coordinates_are_invariant_to_global_axis_orientation():
    from planning.wuji_retarget import anatomical_coordinates
    from scipy.spatial.transform import Rotation
    points=np.array([[0.,0.,0.],[.02,.003,.04],[-.015,-.004,.06]])
    axis=np.array([0.,0.,1.]);palmar=np.array([1.,0.,0.]);thumb=np.array([0.,1.,0.])
    reference=anatomical_coordinates(points,axis,palmar,thumb)
    rotation=Rotation.from_euler('xyz',[.3,1.1,.7]).as_matrix()
    transformed=anatomical_coordinates(points@rotation.T,rotation@axis,rotation@palmar,rotation@thumb)
    np.testing.assert_allclose(transformed,reference,atol=1e-12)


def test_predictor_contains_joint_stops_under_gravity(physical_hand):
    from planning.wuji_retarget_mpc import WujiMPCParams,WujiContactModel
    hand=physical_hand;p=WujiMPCParams(hand,0,horizon=1,solver='ipopt');physics=WujiContactModel(p,0)
    state=np.zeros(p.n_qpos_);state[3]=1.;state[7:7+hand.nu]=hand.lower
    state[-hand.nu:]=hand.lower
    physics.robot_external=-np.ones(hand.nu)
    nxt=np.asarray(physics.prediction_fn(state,np.zeros(hand.nu),np.ones(4*p.max_ncon_),
        np.zeros((4*p.max_ncon_,p.n_qvel_)),physics.runtime_parameters(1.))).ravel()
    assert np.all(nxt[7:7+hand.nu]>=hand.lower-1e-10)
    assert np.all(nxt[7:7+hand.nu]<=hand.upper+1e-10)


def test_offline_contact_seed_tracks_measured_object_without_teleporting(physical_hand):
    from models.wuji_hand import rotation
    from scipy.spatial.transform import Rotation
    hand=physical_hand;q=np.clip(np.zeros(hand.nu),hand.lower,hand.upper)
    q[:3]=[.1,.2,.3];q[3:6]=[2.,-1.4,-.3]
    source=POSE.copy();target=POSE.copy();target[:3]+=[.02,-.01,.025]
    target[3:]=Rotation.from_rotvec([.01,-.04,.12]).as_quat()[[3,0,1,2]]
    body=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,'right_finger2_link4')
    materials=[MaterialPoint(body,np.array([.004,0,.02]),np.array([1.,0,0]),'index','distal')]
    before=hand.data.qpos.copy();original=q.copy()
    points,normals,_=hand.kinematics(materials,hand.set_state(q,source))
    seed=hand.transport_configuration(q,source,target)
    moved,moved_normals,_=hand.kinematics(materials,hand.set_state(seed,target))
    r=rotation(target)@rotation(source).T
    np.testing.assert_allclose(moved,target[:3]+(points-source[:3])@r.T,atol=1e-12)
    np.testing.assert_allclose(moved_normals,normals@r.T,atol=1e-12)
    np.testing.assert_array_equal(seed[6:],q[6:])
    np.testing.assert_array_equal(q,original)
    np.testing.assert_array_equal(hand.data.qpos,before)


def test_physical_control_preview_matches_execution_without_mutating_measurement(physical_hand):
    hand=physical_hand;q=np.clip(np.zeros(hand.nu),hand.lower,hand.upper)
    hand.set_state(q,POSE,hand.data);hand.servo=q.copy();hand.data.ctrl[:]=q
    position=hand.data.qpos.copy();velocity=hand.data.qvel.copy();control=hand.data.ctrl.copy()
    target=q.copy();target[0]+=.001
    preview=hand.preview_control(target,.03)
    predicted_q=preview.qpos.copy();predicted_v=preview.qvel.copy()
    np.testing.assert_array_equal(hand.data.qpos,position)
    np.testing.assert_array_equal(hand.data.qvel,velocity)
    np.testing.assert_array_equal(hand.data.ctrl,control)
    np.testing.assert_array_equal(hand.servo,q)
    hand.execute(target,.03)
    np.testing.assert_allclose(hand.data.qpos,predicted_q,atol=1e-10)
    np.testing.assert_allclose(hand.data.qvel,predicted_v,atol=1e-10)


def test_native_horizon_preview_matches_multiple_position_servo_commands(physical_hand):
    hand=physical_hand;q=np.clip(np.zeros(hand.nu),hand.lower,hand.upper)
    hand.set_state(q,POSE,hand.data);hand.servo=q.copy();hand.data.ctrl[:]=q
    initial=hand.data.qpos.copy()
    targets=[q.copy(),q.copy()];targets[0][0]+=.001;targets[1][0]+=.002
    observed=[]
    preview=hand.preview_sequence(targets,.03,observer=lambda data:observed.append((data.qpos.copy(),data.ctrl.copy())))
    predicted_q,predicted_v=preview.qpos.copy(),preview.qvel.copy()
    np.testing.assert_array_equal(hand.data.qpos,initial)
    assert len(observed)==len(targets)
    for target,(observed_q,observed_control) in zip(targets,observed):
        np.testing.assert_array_equal(observed_control,target)
        hand.execute(target,.03)
        np.testing.assert_allclose(hand.data.qpos,observed_q,atol=1e-10)
    np.testing.assert_allclose(hand.data.qpos,predicted_q,atol=1e-10)
    np.testing.assert_allclose(hand.data.qvel,predicted_v,atol=1e-10)


def test_same_link_face_patches_keep_distinct_normals_despite_semantic_release(physical_hand,monkeypatch):
    hand=physical_hand
    body=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,'right_finger4_link4')
    geom=int(np.flatnonzero(hand.model.geom_bodyid==body)[0])
    obj_geom=int(np.flatnonzero(hand.model.geom_bodyid==hand.obj)[0])
    normals=np.array([[1.,0,0],[0,1.,0]])
    body_rotation=hand.data.xmat[body].reshape(3,3)
    materials=[MaterialPoint(body,np.zeros(3),body_rotation.T@n,'ring','distal',contact_patch=True) for n in normals]
    patches=[SimpleNamespace(geom1=geom,geom2=obj_geom,pos=np.zeros(3),dist=-.0001,
        frame=np.array([[1.,0,0],[0,1.,0],[0,0,1.]]).reshape(-1)),
        SimpleNamespace(geom1=geom,geom2=obj_geom,pos=np.zeros(3),dist=-.0001,
        frame=np.array([[0.,1.,0],[1.,0,0],[0,0,-1.]]).reshape(-1))]
    data=SimpleNamespace(contact=patches,xmat=hand.data.xmat.copy())
    monkeypatch.setattr(hand,'kinematics',lambda *a,**kw:(np.zeros((2,3)),normals,np.zeros((2,3,hand.nu))))
    def measured(model,data,index,value):value[0]=1.
    monkeypatch.setattr(mujoco,'mj_contactForce',measured)
    forces,actual=hand.observed_contacts(materials,np.array([1.,0.]),data=data)
    assert [c['slot'] for c in actual]==[0,1]
    np.testing.assert_allclose(forces,normals)


def test_collision_ik_reduces_reported_mesh_penetration(physical_hand):
    hand=physical_hand;q=np.clip(np.zeros(hand.nu),hand.lower,hand.upper)
    before=hand.collision_report(hand.set_state(q,POSE))['max_penetration']
    _,report=hand.solve_ik(q,POSE,[],np.empty((0,3)),iterations=60)
    assert before>.0015
    assert report['max_penetration']<before/2
    assert report['valid']


def test_upstream_active_contact_allowance_does_not_relax_noncontact_collisions(physical_hand):
    hand=physical_hand
    body=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,'right_finger1_link4')
    hand_geom=int(np.flatnonzero(hand.model.geom_bodyid==body)[0])
    object_geom=int(np.flatnonzero(hand.model.geom_bodyid==hand.obj)[0])
    contact=SimpleNamespace(geom1=hand_geom,geom2=object_geom,dist=-.004)
    data=SimpleNamespace(contact=[contact])
    assert not hand.collision_report(data,{body})['valid']
    hand.contact_penetration_limit=.008
    assert hand.collision_report(data,{body})['valid']
    assert not hand.collision_report(data,set())['valid']
    contact.dist=-.009
    assert not hand.collision_report(data,{body})['valid']


def test_semantic_material_is_on_physical_collider_not_internal_visual_query(physical_hand):
    hand=physical_hand
    body=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,'right_finger2_link2')
    original=np.array([0.,.006,.03])
    material=hand.material_on_collider(body,original,np.array([1.,0.,0.]),'index','proximal')
    np.testing.assert_array_equal(material.source_local,original)
    assert material.local[0]>=original[0]
    assert material.local[1:]==pytest.approx(original[1:])
    # At least one containing collider has its outward supporting face here.
    gaps=[np.min(bounds-normals@material.local) for normals,bounds in hand._body_hulls[body]]
    assert any(abs(gap)<1e-6 for gap in gaps)


def test_robot_predictor_respects_scene_actuator_force_caps(physical_hand):
    from planning.wuji_retarget_mpc import WujiMPCParams,WujiContactModel
    hand=physical_hand;p=WujiMPCParams(hand,0,horizon=1,solver='ipopt');physics=WujiContactModel(p,0)
    physics.robot_mass=np.ones(hand.nu);physics.external=np.zeros(6)
    state=np.zeros(p.n_qpos_);state[3]=1.;state[7:7+hand.nu]=hand.lower
    state[-hand.nu:]=hand.upper
    next_state=np.asarray(physics.prediction_fn(state,np.zeros(hand.nu),np.ones(4*p.max_ncon_),
        np.zeros((4*p.max_ncon_,p.n_qvel_)),physics.runtime_parameters(1.))).reshape(-1)
    hand_velocity=next_state[7+hand.nu+6:7+hand.nu+p.n_qvel_]
    limited=hand.model.actuator_forcelimited.astype(bool)
    assert np.all(np.abs(hand_velocity[limited])<=p.h_*np.maximum(np.abs(p.actuator_lower[limited]),np.abs(p.actuator_upper[limited]))+1e-8)


def test_robot_only_contact_accepts_explicit_friction_without_moving_object():
    import casadi as cs
    from models.soft_scm_contact import grouped_contact_velocity_cs,soft_scm_coefficients
    mu=.5;normal=np.r_[np.zeros(6),0.,0.,1.];t1=np.r_[np.zeros(6),1.,0.,0.];t2=np.r_[np.zeros(6),0.,1.,0.]
    rows=np.array([normal+mu*t1,normal+mu*t2,normal-mu*t1,normal-mu*t2])
    free=cs.DM(np.r_[np.zeros(6),1.,0.,-1.]);a,b=soft_scm_coefficients(.03)
    supported,impulses=grouped_contact_velocity_cs(free,cs.DM(rows),cs.DM.zeros(4),cs.DM.eye(9),.03,a,b,1e-8,
        friction=mu,return_impulses=True)
    result=np.asarray(supported).reshape(-1)
    np.testing.assert_allclose(result[:6],0,atol=1e-12)
    assert 0<result[6]<1 and result[8]>-1
    fn,ft=impulses[0]
    assert np.linalg.norm(np.asarray(ft))<=mu*float(fn)+1e-10


def test_jointly_compatible_projection_set_is_a_protected_local_candidate(policy):
    original=np.array([[.04,0.,0.],[.04,.015,0.]])
    seed=original+np.array([0.,0.,.001]);normals=np.tile([-1.,0.,0.],(2,1))
    result=invoke(policy,lambda points,n:dict(valid=bool(np.allclose(points,seed,atol=1e-12,rtol=0))),
        slots=(10,20),anchors=original,seed_sets=[(seed,normals)])
    assert result['valid'] and result['source']=='local_fps'
    np.testing.assert_allclose(result['points'],seed)


def test_hand_patch_refinement_keeps_body_region_and_bounded_original_prior(physical_hand):
    hand=physical_hand
    body=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,'right_finger2_link4')
    original=hand.material_on_collider(body,np.array([.006,0.,.02]),np.array([1.,0.,0.]),'index','distal')
    data=hand.set_state(np.zeros(hand.nu),POSE)
    r=data.xmat[body].reshape(3,3)
    target=data.xpos[body]+r@(original.local+np.array([0.,0.,.003]))
    refined=hand.nearest_material(original,target,data,radius=.01)
    assert (refined.body,refined.finger,refined.region)==(original.body,original.finger,original.region)
    assert np.linalg.norm(refined.local-original.local)<=.01+1e-12
    assert refined.normal@original.normal>=.2
    frozen=hand.nearest_material(original,target,data,radius=0.)
    np.testing.assert_array_equal(frozen.local,original.local)


def test_contact_ik_updates_material_coordinates_inside_each_joint_iteration(physical_hand):
    hand=physical_hand;body=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,'right_finger2_link4')
    original=hand.material_on_collider(body,np.array([.006,0.,.02]),np.array([1.,0.,0.]),'index','distal')
    q=np.clip(np.zeros(hand.nu),hand.lower,hand.upper)
    distant=POSE.copy();distant[:3]=[2.,2.,2.];data=hand.set_state(q,distant)
    target=data.xpos[body]+data.xmat[body].reshape(3,3)@(original.local+np.array([0.,0.,.003]))
    _,report=hand.solve_ik(q,distant,[original],target[None],iterations=20,tolerance=.001,
        surface_priors=[original],surface_radius=.01)
    assert report['valid'] and report['max_contact_error']<.001
    selected=report['materials'][0]
    assert (selected.body,selected.finger,selected.region)==(original.body,original.finger,original.region)
    assert np.linalg.norm(selected.local-original.local)<=.01+1e-12


def test_wuji_grasp_response_predicts_bounded_physical_preload_growth(physical_hand):
    from planning.wuji_retarget_mpc import WujiMPCParams,WujiContactModel
    from models.allegro_grasp_response import grasp_operators
    hand=physical_hand;p=WujiMPCParams(hand,1,horizon=1,solver='ipopt')
    physics=WujiContactModel(p,1)
    state=np.zeros(p.n_qpos_);state[3]=1.
    phi=np.ones(4*p.max_ncon_);phi[:4]=0.
    jac=np.zeros((4*p.max_ncon_,p.n_qvel_))
    normal=np.zeros(p.n_qvel_);normal[0]=1.;normal[6]=-1.
    tangent1=np.zeros(p.n_qvel_);tangent1[1]=1.;tangent1[7]=-1.
    tangent2=np.zeros(p.n_qvel_);tangent2[2]=1.;tangent2[8]=-1.
    jac[:4]=[normal+p.friction*tangent1,normal+p.friction*tangent2,
             normal-p.friction*tangent1,normal-p.friction*tangent2]
    physics.object_mass=np.eye(6);physics.external=np.array([0.,0.,-5.,0.,0.,0.])
    (physics.grasp_map,physics.grasp_robot,physics.grasp_inverse,
     physics.grasp_mobility,physics.grasp_internal)=grasp_operators(jac,np.ones(hand.nu),np.eye(6),1)
    physics.grasp_reference=np.array([.3,0.,0.]);physics.grasp_limits=np.array([.3]);physics.grasp_active=True
    physics.pressure_bias[0]=.001
    mp=physics.runtime_parameters(1.)
    assert len(mp)==physics.model_parameter_dim
    command=np.full(hand.nu,10.)
    force=np.asarray(physics.contact_force_fn(state,command,phi,jac,mp)).reshape(3)
    physics.grasp_active=False
    raw=np.asarray(physics.contact_force_fn(state,command,phi,jac,physics.runtime_parameters(1.))).reshape(3)
    physics.grasp_active=True
    assert .3<force[0]<=max(.3,raw[0])+1e-10
    assert np.sum(np.abs(force[1:]))<=p.friction*force[0]+1e-10
    assert np.linalg.norm(force)<=p.contact_force_limit+1e-10
    prediction=np.asarray(physics.prediction_fn(state,command,phi,jac,mp)).reshape(-1)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction[-hand.nu:],np.clip(p.mpc_u_ub_,hand.lower,hand.upper))
    # A large desired/reference force cannot attach a separated pad.
    phi[:4]=.1;physics.grasp_reference[:]=100.
    separated=np.asarray(physics.contact_force_fn(state,command,phi,jac,physics.runtime_parameters(1.))).ravel()
    np.testing.assert_allclose(separated,0.,atol=1e-12)
