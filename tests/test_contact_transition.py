"""Relocation must price the unsupported travel, using the shared objective."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from planning.contact_transition import ContactTransitionPredictor
from planning.franka_contact_motion import FrankaApproachVia
from planning.mlqp_point import LambdaContactControlOptimizer
from planning.object_pose_cost import cost_config, object_cost_np, target_up_axis


@pytest.fixture
def support_case(tmp_path):
    import trimesh
    mesh = trimesh.creation.box(extents=[.08,.16,.08])
    path = tmp_path/'support.stl'; mesh.export(str(path))
    opt = LambdaContactControlOptimizer(str(path), solver='ipopt', sample_num=8,
        max_contacts=4, time_step=.02, contact_dynamics='quasi-dynamic',
        grouped_contact_response=True, robot_stiffness=600., max_contact_force=3.)
    predictor = ContactTransitionPredictor(opt, mesh.vertices)
    angle = np.pi/6
    pose = np.r_[[0.,0.,.04*(np.cos(angle)+np.sin(angle))],
                 [np.cos(angle/2),0.,np.sin(angle/2),0.]]
    target = np.r_[[0.,0.,.04], [2**-.5,0.,2**-.5,0.]]
    config = cost_config(500,0,20,target_up_axis(target[3:]))
    rotation = Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
    opt.sample_point = np.array([[-.04,-.08,.04],[-.04,.08,.04]])
    opt.normal = np.array([[1.,0.,0.]]*2)
    opt.t1, opt.t2 = np.array([[0.,1.,0.]]*2), np.array([[0.,0.,1.]]*2)
    opt.last_candidate_ids = np.array([0,1])
    opt.last_candidate_forces = [np.array([1.,0.,0.]), np.array([2.4,0.,0.])]
    point = opt.sample_point[0]
    contact = dict(point_world=rotation@point+pose[:3], normal_world=-rotation@opt.normal[0])
    state = np.r_[pose, rotation@(point-.01*opt.normal[0])+pose[:3]]
    common = dict(pose=pose,target=target,config=config,gravity_world=np.array([0.,0.,-.098]),
                  floor_z=0.,mu=.5)
    return opt, predictor, state, contact, common


def test_instant_winner_loses_when_withdrawal_loses_flip_progress(support_case):
    opt, predictor, state, contact, common = support_case
    instant = [predictor.rollout(point=opt.sample_point[i],frame=np.eye(3),
        force=opt.last_candidate_forces[i],delay_steps=0,steps=1,**common) for i in (0,1)]
    assert instant[1]['cost'] < instant[0]['cost']  # teleporting favors the other point
    result = predictor.evaluate(state, common['target'], common['config'], contact,
        opt.last_candidate_forces[0], [1], FrankaApproachVia(), .01,0.,0.,.5,common['gravity_world'])
    assert result['travel_steps'] > 1
    assert result['steps'] == predictor.max_steps
    assert result['steps'] < result['travel_steps']  # cannot reach it in this horizon
    assert result['stay_cost'] < result['switch_cost']
    assert not result['switch']


def test_zero_pressure_retains_support_but_withdrawal_removes_it(support_case):
    opt, predictor, state, contact, common = support_case
    held = predictor.rollout(point=opt.sample_point[0],frame=np.eye(3),
        force=np.zeros(3),delay_steps=0,steps=30,**common)
    withdrawn = predictor.rollout(point=opt.sample_point[0],frame=np.eye(3),
        force=np.zeros(3),delay_steps=30,steps=30,**common)
    assert held['cost'] < withdrawn['cost']
    assert object_cost_np(held['pose'],common['target'],common['config']) < object_cost_np(
        withdrawn['pose'],common['target'],common['config'])


def test_batched_support_matches_contact_frame_geometry(support_case):
    from examples.mpc.franka.runtime.contact import mesh_table_contact_points, support_patch_jacobians
    opt,predictor,state,contact,common = support_case
    rot = Rotation.from_quat(state[[4,5,6,3]]).as_matrix()
    jac,phi = predictor._support_batch(state[None,:3],rot[None,:,:],0.,.5)
    points = mesh_table_contact_points(state[:3],rot,predictor.vertices,max_points=4)
    # Tangent signs can change; each normal and tangent disk must agree.
    for i,point in enumerate(points):
        _, reference = support_patch_jacobians(state[:3],rot,point,6,.5)
        block = jac[0,4*i:4*i+4]
        np.testing.assert_allclose(block.mean(axis=0), reference.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(block.T@block, reference.T@reference, atol=1e-12)
        np.testing.assert_allclose(phi[0,4*i:4*i+4], point[2], atol=1e-12)


def test_cost_comparison_can_still_choose_a_better_relocation(support_case, monkeypatch):
    opt,predictor,state,contact,common = support_case
    def forecasts(**kwargs):
        # First 3 are stay controls, remaining 3 are relocation controls.
        return [dict(cost=c) for c in (12.,11.,10.,9.,8.,7.)]
    monkeypatch.setattr(predictor,'_rollout_batch',forecasts)
    result = predictor.evaluate(state,common['target'],common['config'],contact,
        opt.last_candidate_forces[0],[1],FrankaApproachVia(),.01,0.,0.,.5,common['gravity_world'])
    assert result['switch'] and result['best_idx'] == 1
    assert result['stay_cost'] == 10. and result['switch_cost'] == 7.


def test_approach_inset_does_not_change_the_measured_contact_radius(support_case):
    opt,predictor,state,contact,common = support_case
    results = [predictor.evaluate(state,common['target'],common['config'],contact,
        opt.last_candidate_forces[0],[],FrankaApproachVia(),.01,margin,0.,.5,
        common['gravity_world']) for margin in (0.,-.0005)]
    # The margin moves a destination; it must not invent a positive gap at
    # an already measured physical sphere contact.
    assert results[0]['stay_cost'] == pytest.approx(results[1]['stay_cost'],abs=1e-12)


@pytest.mark.parametrize('stage',['flip','pose'])
@pytest.mark.parametrize('switch',[False,True])
def test_transition_decision_controls_contact_and_cancels_old_via(stage,switch):
    from examples.mpc.fingertips.test.test_0902 import (
        compute_rollout_contact_via, ContactValueTracker, ModelCostConfidence,
    )
    points = np.array([[.02,0.,.02],[-.02,0.,.02]])
    normals = np.array([[-1.,0.,0.],[1.,0.,0.]])
    current = np.array([0.,0.,0.,1.,0.,0.,0.])
    target = np.r_[[.08,0.,0.],current[3:]]
    predictions = [np.r_[[-.001,0.,0.],current[3:]],np.r_[[.01,0.,0.],current[3:]]]
    forces = [np.array([.1,0.,0.])]*2
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.__dict__.update(sample_point=points,normal=normals,top_k=2,
        sample_geodesic=np.array([[0.,.04],[.04,0.]]),last_global_idx=1,
        last_selected_idx=1,last_executed_idx=0,contact_switch_radius=.03,
        contact_switch_confidence=1.,mu_arm_obj=.5,pos_coef=500.,ori_coef=20.,
        _object_cost_config=cost_config(500,20,20),
        update_Jacobian=lambda *a,**kw:None,
        get_availble_point_idx=lambda *a,**kw:np.array([0,1]),
        filter_rankable_indices=lambda ids:ids)
    def rank(*a,**kw):
        costs = opt._rescore_candidate_costs([0,1],[3.,3.],predictions,target,current)
        opt._store_candidate_buffers([0,1],costs,predictions,forces)
        opt.last_best_x_plus,opt.last_best_force,opt.last_best_cost = predictions[1],forces[1],costs[1]
        return points[1],normals[1],costs[1],costs[0],1
    def solve(*a,**kw):
        return points[0],-normals[0],predictions[0],4.,dict(control_input=forces[0],solver_failed=False)
    opt.choose_contact_points, opt.resolve_executed_contact, opt.optimize_control_input = rank,solve,solve
    origin = np.array([.4,.1,.4])
    contact = dict(point_world=origin+points[0],normal_world=-normals[0])
    predictor = SimpleNamespace(evaluate=lambda *a,**kw:dict(switch=switch,best_idx=1,
        stay_cost=10.,switch_cost=9. if switch else 12.,steps=30,travel_steps=25))
    param = SimpleNamespace(lambda_optimizer=opt,target_p_=origin+target[:3],target_q_=target[3:],
        object_cost_stage_=stage,object_cost_config_=cost_config(500,20,20),
        lambda_actual_contact_=contact,contact_transition_predictor_=predictor)
    args = SimpleNamespace(ground_height_threshold=.012,ideal_contact_surface_margin=-.0005)
    tip = origin+points[0]-.0095*normals[0]
    via = FrankaApproachVia()
    via.goal,via.center,via.height,via.phase,via.blocked = origin+[-.1,0.,.05],origin.copy(),.6,'cross',True
    tracker = ContactValueTracker()
    tracker.contact_active = tracker._holding_p_arm = True
    result = compute_rollout_contact_via(param,args,np.r_[origin,current[3:],tip],np.eye(3),
        np.zeros(6),np.zeros((40,6)),.01,tracker,ModelCostConfidence(),via,
        True,1,floor_ground=.362,floor_z=.35)
    assert result['value_info']['switch_observed_contact'] == switch
    assert result['value_info']['accept_p_arm'] == (not switch)
    assert result['best_contact_idx'] == (1 if switch else 0)
    assert result['value_info']['contact_transition_steps'] == 30
    if not switch:
        assert not result['value_info']['path_blocked']
        assert via.phase == 'drop'
        assert np.linalg.norm(via.goal - tip) < .01
        np.testing.assert_allclose(result['p_arm_surface_world'],contact['point_world'])
