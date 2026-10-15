"""Semantic identity, original-first selection, and real MuJoCo/Warp contracts."""
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import trimesh

from models.wuji_hand import ObjectSurface,MaterialPoint,WujiHand,build_scene
from planning.wuji_retarget import (classify_vertices,xyzw_xyz_to_pose,ReferenceProgress,HandReferenceProgress,fps,LocalLambdaPolicy)


POSE=np.array([0.,0.,.2,1.,0.,0.,0.])


def test_internal_preload_budget_scales_with_weight_and_native_cone():
    from planning.wuji_retarget_mpc import internal_preload_floor
    weight=.514*9.81
    floor=internal_preload_floor(weight,.5,5,10.)
    assert floor==pytest.approx(1.3*np.sqrt(2)*weight/(.5*5))
    assert .5/np.sqrt(2)*5*floor>=weight
    assert internal_preload_floor(weight,.5,4,10.)>floor
    assert internal_preload_floor(2*weight,.5,5,10.)>floor
    assert internal_preload_floor(weight,.5,0,10.)==pytest.approx(.6)
    assert internal_preload_floor(100.,.5,2,10.)<=10/np.sqrt(1+.5**2)


def test_source_pose_conversion_normalizes_without_mutating():
    source=np.array([0.,0.,0.,2.,1.,2.,3.]);result=xyzw_xyz_to_pose(source)
    np.testing.assert_array_equal(source,[0,0,0,2,1,2,3])
    np.testing.assert_allclose(result,[1,2,3,1,0,0,0])
    with pytest.raises(ValueError):xyzw_xyz_to_pose(np.zeros(7))


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


def test_wuji_solver_cache_key_distinguishes_control_rate_bounds(physical_hand):
    from planning.wuji_retarget_mpc import WujiMPCParams
    first=WujiMPCParams(physical_hand,3,rates=np.ones(physical_hand.nu))
    second=WujiMPCParams(physical_hand,3,rates=np.full(physical_hand.nu,2.))
    assert first.tag!=second.tag


def test_geometric_ik_solves_each_contact_and_noncontact_frame(monkeypatch):
    from examples.retarget import retarget
    calls=[]
    class Hand:
        def solve_ik(self,q,pose,materials,targets,**kwargs):
            calls.append((q.copy(),len(materials),kwargs['shape_targets'].copy()))
            return q+1,dict(valid=True,max_contact_error=0.)
    demo=SimpleNamespace(poses=np.tile(POSE,(4,1)))
    contacts=SimpleNamespace(mask=np.array([[False],[True],[False],[True]]),materials=['pad'],
        hand_targets=np.zeros((4,1,3)),hand_normals=np.zeros((4,1,3)))
    shape=np.arange(12).reshape(4,1,3)
    monkeypatch.setattr(retarget,'shape_references',lambda hand,demo:(['shape'],shape))
    monkeypatch.setattr(retarget,'initialize_hand',lambda *args:np.zeros(26))
    q,_=retarget.geometric_ik(Hand(),demo,contacts,SimpleNamespace(ik_iterations=5,ik_tolerance=.01))
    assert [call[1] for call in calls]==[0,1,0,1]
    for frame,call in enumerate(calls):
        np.testing.assert_array_equal(call[0],np.full(26,frame))
        np.testing.assert_array_equal(call[2],shape[frame])
    np.testing.assert_array_equal(q[:,0],[1,2,3,4])


def test_geometric_hand_pose_is_distinct_from_contact_feasibility(monkeypatch):
    from examples.retarget import retarget
    class Hand:
        def solve_ik(self,q,*args,**kwargs):
            return q,dict(valid=False,collisions=[],max_contact_error=.016,max_shape_error=.20)
    demo=SimpleNamespace(poses=POSE[None])
    contacts=SimpleNamespace(mask=np.ones((1,1),bool),materials=['pad'],
        hand_targets=np.zeros((1,1,3)),hand_normals=np.zeros((1,1,3)))
    monkeypatch.setattr(retarget,'shape_references',lambda *args:(['shape'],np.zeros((1,1,3))))
    monkeypatch.setattr(retarget,'initialize_hand',lambda *args:np.zeros(26))
    _,reports=retarget.geometric_ik(Hand(),demo,contacts,SimpleNamespace(ik_iterations=5,ik_tolerance=.012))
    assert reports[0]['geometric_pose_valid']
    assert not reports[0]['semantic_contact_valid']


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


def test_mpc_selects_closed_descending_plan_and_labels_contact_acquisition():
    from planning.wuji_retarget_mpc import select_contact_candidate
    def candidate(source,cost,closed,reduction):
        return dict(source=source,score=cost,certificate=dict(force_closure=closed),
            object_pose_reduction=reduction,object_pose_cost=1.)
    hold=candidate('hold',20.,False,0.)
    nonclosed=candidate('near_projection',1.,False,.2)
    increasing=candidate('bad_closed',2.,True,-.1)
    closed=candidate('good_closed',10.,True,.1)
    chosen,status=select_contact_candidate([hold,nonclosed,increasing,closed],20.)
    assert chosen is closed and status=='closed_pose_descent'
    chosen,status=select_contact_candidate([hold,nonclosed],20.)
    assert chosen is nonclosed and status=='contact_acquisition'
    chosen,status=select_contact_candidate([hold,candidate('worse',21.,True,.2)],20.)
    assert chosen is hold and status=='hold_pending_closure_or_descent'


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


def test_trigrasp_object_cost_scaling_and_terminal_task_objective(physical_hand):
    from planning.wuji_retarget_mpc import WujiOnlineMPC,WujiRetargetCosts
    with pytest.raises(ValueError):WujiRetargetCosts(posture_contact_scale=0.)
    hand=physical_hand;hand.set_state(np.zeros(hand.nu),POSE,hand.data);hand.servo=np.zeros(hand.nu)
    mpc=WujiOnlineMPC(hand,0,horizon=1,solver='ipopt')
    empty=np.empty((0,3))
    mpc.plan(POSE,hand.servo,[],empty,empty,empty,np.empty(0))
    state=mpc.state();cp=mpc.current_cp
    baseline=float(mpc.path_cost_fn(state,np.zeros(hand.nu),cp))
    for axis,weight in [(0,300.),(1,300.),(2,120.)]:
        shifted=state.copy();shifted[axis]+=.002
        assert float(mpc.path_cost_fn(shifted,np.zeros(hand.nu),cp))-baseline==pytest.approx(weight*.002**2)
        assert float(mpc.final_cost_fn(shifted,cp))-float(mpc.final_cost_fn(state,cp))==pytest.approx(4*weight*.002**2)
    outside=state.copy();outside[0]+=.02
    assert float(mpc.path_cost_fn(outside,np.zeros(hand.nu),cp))-baseline==pytest.approx(300*.02**2+2e6*.015**2)


def test_measured_contact_survives_lambda_failure_and_reference_release(physical_hand,monkeypatch):
    from examples.retarget.contact_geometry import contact_geometry
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
    _,scm_rows,*_=contact_geometry(hand,[m],np.zeros(1),10)
    np.testing.assert_allclose(mpc.physics.projection,np.eye(hand.nu),atol=1e-12)
    np.testing.assert_allclose(mpc.physics.pressure_bias,0.,atol=1e-12)


def test_mpc_uses_contacts_and_pose_without_external_force_allocation(physical_hand,monkeypatch):
    from planning import grasp_force_allocation
    from planning.wuji_retarget_mpc import WujiOnlineMPC
    hand=physical_hand;hand.set_state(np.zeros(hand.nu),POSE,hand.data);hand.servo=np.zeros(hand.nu)
    bodies=[mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,f'right_finger{i}_link4') for i in (2,3)]
    materials=[MaterialPoint(b,np.zeros(3),np.array([1.,0,0]),name,'distal') for b,name in zip(bodies,['index','middle'])]
    normals=np.array([[-1.,0,0],[1.,0,0]])
    points=POSE[:3]+np.array([[.025,0,0],[-.025,0,0]])
    measured=.05*normals
    actual=[dict(slot=i,body=bodies[i],point=points[i],normal=normals[i],force=measured[i],gap=-.0001) for i in range(2)]
    monkeypatch.setattr(hand,'observed_contacts',lambda *args,**kwargs:(measured.copy(),actual))
    monkeypatch.setattr(grasp_force_allocation,'allocate_contact_wrench',lambda *a,**kw:pytest.fail('No external wrench allocator in Wuji MPC'))
    mpc=WujiOnlineMPC(hand,2,horizon=1,solver='ipopt')
    monkeypatch.setattr(mpc,'plan_once',lambda **kwargs:dict(action=np.zeros(hand.nu),u_traj=np.zeros((1,hand.nu)),solve_status='test'))
    _,report=mpc.plan(POSE,hand.servo,materials,np.zeros((2,3)),np.zeros((2,3)),normals,
        np.ones(2),lambda_valid=False)
    assert report['made_contact_count']==2 and report['force_source']=='contact_dynamics'
    assert not report['external_wrench_target']
    assert not report['grasp_certificate']['force_closure']
    assert not report['grasp_certificate']['load_feasible']
    assert report['grasp_response'] and not report['transport_active']
    # Pausing the lift reference must not disable partial-grasp physics or
    # the internal force-building loop (the old circular gate did both).
    np.testing.assert_allclose(report['executed_object_goal'],POSE)
    assert report['friction_cone']=='pyramidal'
    assert report['internal_preload_floor_N']==pytest.approx(.6)
    assert report['internal_preload_budget_N']>5.
    assert report['opposing_coverage']==pytest.approx(1.)
    assert mpc.current_cp[mpc.preload_floor_index]==pytest.approx(report['internal_preload_floor_N'])
    assert report['candidate_cost_physics']=='native_MuJoCo' and report['force_feedback']
    assert report['physical_material_count']==2
    # Same-link contact must adopt the actual hand-side surface, rather than
    # claiming the requested (potentially distant) semantic material touched.
    adopted=hand.kinematics(mpc.physical_materials)[0]
    np.testing.assert_allclose(adopted,points+.00005*normals,atol=1e-12)
    assert not np.allclose(adopted,hand.kinematics(materials)[0])
    force_start=mpc.force_offset+mpc.param_.n_qpos_+hand.nu
    np.testing.assert_allclose(mpc.current_cp[force_start:force_start+6],measured.reshape(-1))
    # Terminal pressure cannot be postponed forever by a short RTI horizon.
    low=mpc.current_cp.copy();low[force_start+6:-8*(1+hand.nu)]=0.
    high=low.copy();high[force_start:force_start+6]=(6.*normals).reshape(-1)
    assert float(mpc.final_cost_fn(mpc.state(),high))<float(mpc.final_cost_fn(mpc.state(),low))
    # Internal equilibrium is generated from scene gravity, without a Lambda
    # wrench input. Keep every other residual fixed while supplying net weight.
    up=low.copy();balanced=measured.copy()
    balanced[:,2]+=-hand.model.body_mass[hand.obj]*hand.model.opt.gravity[2]/2
    up[force_start:force_start+6]=balanced.reshape(-1)
    # This only probes the objective: force feasibility remains the dynamics'
    # job, and this synthetic force split is not a physical grasp certificate.
    assert float(mpc.final_cost_fn(mpc.state(),up))<float(mpc.final_cost_fn(mpc.state(),low))
    # A weak, partial grasp must not cancel the common wrist motion while
    # requesting more preload. The bounded dynamics still determines support.
    # Internal pressure feedback can compensate wrist-induced normal motion
    # with fingers, but must never cancel the commanded wrist motion itself.
    np.testing.assert_allclose(mpc.physics.projection[:6,:6],np.eye(6),atol=1e-12)
    np.testing.assert_allclose(mpc.physics.projection[:6,6:],0.,atol=1e-12)
    np.testing.assert_allclose(mpc.physics.pressure_bias[:6],0.,atol=1e-12)
    assert report['internal_pressure_control']
    transport_probe=mpc.current_cp.copy();transport_probe[2]+=.02
    residual=np.asarray(mpc.residual_fn(mpc.state(),np.zeros(hand.nu),transport_probe)).ravel()
    assert any(np.linalg.norm(residual[s])>1e-6 for s in mpc.transport_residual_slices)
    # Recentring only force output linearization must not move wrench origins.
    cp=mpc.current_cp.copy();offset=mpc.force_offset
    cp[offset+mpc.param_.n_qpos_+hand.nu:offset+mpc.param_.n_qpos_+hand.nu+6]=[0,0,10,0,0,0]
    cp[offset+mpc.param_.n_qpos_+hand.nu+6:-8*(1+hand.nu)]=0.
    state=mpc.state();cost=float(mpc.path_cost_fn(state,np.zeros(hand.nu),cp))
    cp[offset+1]+=.1
    assert float(mpc.path_cost_fn(state,np.zeros(hand.nu),cp))==pytest.approx(cost)
    # Reference release disables pressure regulation, but actual unilateral
    # contact physics is retained until real separation.
    _,released=mpc.plan(POSE,hand.servo,materials,np.zeros((2,3)),np.zeros((2,3)),normals,
        np.zeros(2),lambda_valid=False)
    assert not released['internal_pressure_control'] and released['grasp_response']
    np.testing.assert_allclose(mpc.physics.projection,np.eye(hand.nu),atol=1e-12)


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
    assert float(mpc.final_cost_fn(mpc.state(),unloaded))<float(mpc.final_cost_fn(mpc.state(),loaded))


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
    from examples.retarget.contact_geometry import contact_geometry
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


def test_interrupted_rollout_command_recovery_preserves_clipping_and_order():
    from examples.retarget.verify_force_closure import recover_commands,longest_run,gravity_capacity
    records=[dict(step=i,mpc=dict(executed_action=delta)) for i,delta in enumerate([[.1,-.2],[.1,0.]])]
    np.testing.assert_allclose(recover_commands(records,[0,0],[-1,-1],[.15,1]),[[.1,-.2],[.15,-.2]])
    assert longest_run([False,True,True,False,True])==2
    records[0]['step']=1
    with pytest.raises(ValueError):recover_commands(records,[0,0],[-1,-1],[1,1])
    capacity=gravity_capacity(np.zeros((1,3)),np.array([[0.,0,1.]]),
        np.array([[[1.,0,0],[0,1.,0]]]),.5,np.array([2.]),np.array([0,0,-1.,0,0,0]))
    assert capacity==pytest.approx(2.)


def test_static_native_support_witness_respects_motor_and_servo_force_bounds():
    from examples.retarget.verify_force_closure import actuator_support_witness
    def evaluate(cap,measured):
        model=mujoco.MjModel.from_xml_string(f'''<mujoco><compiler autolimits="true"/>
          <worldbody><body name="right" pos=".05 0 .05"><joint name="r" type="slide" axis="1 0 0" range="-.2 .2"/><geom type="sphere" size=".025"/></body>
          <body name="left" pos="-.05 0 .05"><joint name="l" type="slide" axis="1 0 0" range="-.2 .2"/><geom type="sphere" size=".025"/></body>
          <body name="obj" pos="0 0 .05"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia=".001 .001 .001"/><geom type="box" size=".025 .025 .025"/></body></worldbody>
          <actuator><position joint="r" kp="100" ctrlrange="-.2 .2" forcerange="-{cap} {cap}"/>
          <position joint="l" kp="100" ctrlrange="-.2 .2" forcerange="-{cap} {cap}"/></actuator></mujoco>''')
        hand=SimpleNamespace(model=model,nu=2,obj=3,qadr=np.array([0,1]),dadr=np.array([0,1]),lower=np.full(2,-.2),upper=np.full(2,.2),friction=1.)
        def set_state(q,pose,data):
            data.qpos[:2]=q;data.qpos[2:]=pose;data.qvel[:]=0;mujoco.mj_forward(model,data)
        hand.set_state=set_state
        snapshot=dict(hand_q=np.zeros(2),object_pose=[0,0,.05,1,0,0,0],contacts=[
            dict(point=[.025,0,0],normal=[-1,0,0],tangents=[[0,1,0],[0,0,1]],normal_force=1.,body=1,body_name='right'),
            dict(point=[-.025,0,0],normal=[1,0,0],tangents=[[0,1,0],[0,0,1]],normal_force=1.,body=2,body_name='left')])
        return actuator_support_witness(hand,snapshot,.01,measured_budget=measured)
    assert not evaluate(2.,False)['feasible']
    assert not evaluate(10.,True)['feasible']
    future=evaluate(10.,False)
    assert future['feasible']
    np.testing.assert_allclose(future['required_normal_forces'],[4.905,4.905],atol=1e-7)
    np.testing.assert_allclose(future['required_joint_torque'],[-4.905,4.905],atol=1e-7)


def test_anatomical_surface_coordinates_are_invariant_to_global_axis_orientation():
    from planning.wuji_retarget import anatomical_coordinates
    from scipy.spatial.transform import Rotation
    points=np.array([[0.,0.,0.],[.02,.003,.04],[-.015,-.004,.06]])
    axis=np.array([0.,0.,1.]);palmar=np.array([1.,0.,0.]);thumb=np.array([0.,1.,0.])
    reference=anatomical_coordinates(points,axis,palmar,thumb)
    rotation=Rotation.from_euler('xyz',[.3,1.1,.7]).as_matrix()
    transformed=anatomical_coordinates(points@rotation.T,rotation@axis,rotation@palmar,rotation@thumb)
    np.testing.assert_allclose(transformed,reference,atol=1e-12)


def test_scene_preserves_dexforge_options_contact_masks_and_servos(physical_hand):
    from examples.retarget.retarget import physics_audit
    root=Path(__file__).resolve().parents[1]/'thirdparty/retarget_assets/DexForge'
    report=physics_audit(physical_hand,root/'assets/wuji/right_mesh.xml',
        root/'data/dexycb/dexycb_004_sugar_box_20200709_142802/scene/leaphand/sugar_box_mesh.xml')
    assert report['option_overrides']==[]
    assert report['options']['timestep']==pytest.approx(.01)
    assert report['options']['iterations']==10
    assert report['object_mass']==pytest.approx(.514)
    assert report['hand_object_friction']==pytest.approx(.5)
    assert report['object_floor_friction']==pytest.approx(1.)


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


def test_dexycb_import_uses_grasp_identity_calibration_and_valid_masks(tmp_path):
    from examples.retarget.dexycb_data import prepare_sequence,read_yaml
    import yaml,json
    dataset=tmp_path/'data';sequence=dataset/'subject'/'sequence';camera='123'
    (sequence/camera).mkdir(parents=True)
    metadata=dict(mano_sides=['right'],serials=[camera],ycb_ids=[1,3],ycb_grasp_ind=1,
        num_frames=2,extrinsics='test',mano_calib=['test'])
    (sequence/'meta.yml').write_text(yaml.safe_dump(metadata))
    shape=dataset/'calibration/mano_test';shape.mkdir(parents=True)
    (shape/'mano.yml').write_text(yaml.safe_dump(dict(betas=np.arange(10).astype(float).tolist())))
    ext=dataset/'calibration/extrinsics_test';ext.mkdir(parents=True)
    transform=np.c_[np.eye(3),[.1,.2,.3]]
    (ext/'extrinsics.yml').write_text(yaml.dump(dict(extrinsics={camera:tuple(transform.ravel().tolist())})))
    assert read_yaml(ext/'extrinsics.yml')['extrinsics'][camera]==transform.ravel().tolist()
    for frame in range(2):
        poses=np.tile(np.c_[np.eye(3),np.zeros(3)],(2,1,1))
        poses[1,:,3]=[.2,.3,.4]
        if frame==0:poses[1]=0.
        np.savez(sequence/camera/f'labels_{frame:06d}.npz',pose_m=np.ones((1,51)),pose_y=poses)
    template=tmp_path/'scene.xml'
    template.write_text('<mujoco><asset><mesh name="sugar_box_mesh" file="mesh.obj"/></asset><worldbody><body name="obj"><joint name="obj_joint" type="free"/><inertial mass="0.514" pos="0 0 0" diaginertia=".001 .001 .001"/><geom mesh="sugar_box_mesh" type="mesh"/></body></worldbody></mujoco>')
    output=prepare_sequence(dataset,sequence,camera,template,tmp_path/'prepared');raw=output/'mano_raw'
    meta=json.loads((raw/'meta.json').read_text())
    assert meta['object_index']==1 and meta['object']=='004_sugar_box'
    np.testing.assert_array_equal(np.load(raw/'frame_valid.npy'),[False,True])
    np.testing.assert_array_equal(np.load(raw/'extrinsics_4x4.npy')[:3],transform)
    poses=np.load(raw/'object_pose_camera_7.npy')
    np.testing.assert_allclose(poses[1,1],[0,0,0,1,.2,.3,.4])


def test_table_plane_uses_object_support_to_reject_larger_background_wall():
    from examples.retarget.dexycb_data import fit_table_plane
    rng=np.random.default_rng(7)
    floor=np.c_[rng.uniform(-.5,.5,(1000,2)),1+rng.normal(0,.0005,1000)]
    wall=np.c_[np.ones(3000),rng.uniform(-.5,.5,3000),rng.uniform(.3,1.3,3000)]
    vertices=np.array([[x,y,z] for x in (-.04,.04) for y in (-.05,.05) for z in (.85,.999)])
    result=fit_table_plane(np.vstack((floor,wall)),vertices)
    np.testing.assert_allclose(result['normal_camera'],[0,0,-1],atol=.001)
    assert result['offset_camera']==pytest.approx(-1.,abs=.001)
    assert result['inliers']>=1000 and result['rms_m']<.001


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
