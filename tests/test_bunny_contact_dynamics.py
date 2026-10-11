from types import SimpleNamespace

import casadi as cs
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from models.cartesian_rigid_contact import CartesianRigidContactModel
from planning.measured_contact import measured_patch_index
from planning.object_pose_cost import ObjectCostStages, object_cost_np, object_cost_cs, object_cost_torch


def test_sparse_sample_boundary_does_not_move_measured_contact_target():
    optimizer = SimpleNamespace()
    p = np.array([.003759, .006621, .046966])
    assert measured_patch_index(optimizer, p, [1,0,0], 29) == 29
    # Bunny's measured patch advances 1.96 mm while its nearest ranking
    # sample changes to one 56.6 mm away along the mesh.
    pnext = np.array([.005522, .007437, .046734])
    assert measured_patch_index(optimizer, pnext, [1,.03,0], 8) == 29
    # Fixed anchor prevents a sequence of small moves preserving identity
    # indefinitely across the whole object.
    assert measured_patch_index(optimizer, p+[.04,0,0], [1,0,0], 8) == 8
    assert measured_patch_index(optimizer, p+[.04,0,0], [0,0,1], 4) == 4


def rigid_model():
    return CartesianRigidContactModel(SimpleNamespace(
        h_=.02, max_ncon_=1, sim_obj_mass_=.01,
        sim_obj_inertia_diag_=np.array([1e-5,2e-5,3e-5]),
        gravity_=np.array([0.,0.,-9.8,0.,0.,0.]),
        soft_scm_regularization=1e-8, soft_scm_damping_ratio=1.,
        Q=np.diag([50.]*3+[.05]*3+[600.]*3), obj_mass_=.01,soft_scm_response_time=.05,
        robot_stiff_=600*np.eye(3)))


def test_velocity_survives_horizon_and_world_to_body_conversion():
    model = rigid_model()
    rotation = Rotation.from_euler('xyz', [20,35,60], degrees=True)
    pose = np.r_[[0,0,.5], rotation.as_quat()[[3,0,1,2]], [.1,0,.6]]
    omega = np.array([1.,-.5,.3])
    state = model.observation_state(pose, [.1,0,0], omega, [0,0,0])
    ref = state.copy()
    for _ in range(5):
        state = np.array(model.prediction_fn(state, np.zeros(3), np.ones(4), np.zeros((4,9)), 7.)).ravel()
    assert state[0] == pytest.approx(ref[0]+.01)
    assert state[2] == pytest.approx(ref[2]-9.8*.02**2*15)
    np.testing.assert_allclose(Rotation.from_quat(state[[4,5,6,3]]).apply(state[13:16]), omega, atol=1e-10)
    assert Rotation.from_quat(state[[4,5,6,3]]).inv().__mul__(rotation).magnitude() > .1


def test_cartesian_increment_integrates_osc_pd_wrench_instead_of_teleporting():
    model = rigid_model()
    pose = np.r_[[0,0,.5], [1.,0.,0.,0.], [.1,0,.6]]
    state = model.observation_state(pose)
    output = np.array(model.prediction_fn(state, [.005,0,0], np.ones(4), np.zeros((4,9)), 7.)).ravel()
    decay = np.exp(-40./2.5*.02)
    expected = .005*600/40*(.02-2.5/40*(1-decay))
    assert output[7]-state[7] == pytest.approx(expected)
    assert 0 < output[7]-state[7] < .001
    assert output[16] > (output[7]-state[7])/.02


def test_flip_region_and_pose_upright_cost_agree_in_all_backends():
    import torch
    stage = ObjectCostStages(SimpleNamespace(obj='stanford_bunny2', pos_coef=500., ori_coef=20., two_stage_object_cost=True))
    target = np.r_[np.zeros(3), [1.,0.,0.,0.]]
    pose = target.copy(); pose[:3] = [.07,0,.2]
    assert object_cost_np(pose, target, stage.config()) == pytest.approx(0)
    pose[0] = .1
    assert object_cost_np(pose, target, stage.config()) == pytest.approx(.2)
    stage.stage = 'pose'
    pose[:3] = 0
    pose[3:] = Rotation.from_euler('x', 30, degrees=True).as_quat()[[3,0,1,2]]
    config = stage.config()
    x, xd, c = cs.SX.sym('x',7), cs.SX.sym('xd',7), cs.SX.sym('c',12)
    fn = cs.Function('region_cost', [x,xd,c], [object_cost_cs(x,xd,c)])
    expected = object_cost_np(pose,target,config)
    assert expected > 25.
    assert float(fn(pose,target,config)) == pytest.approx(expected, abs=1e-8)
    assert float(object_cost_torch(torch.tensor(pose[:3]),torch.tensor(pose[3:]),torch.tensor(target),config)) == pytest.approx(expected)


def bunny_replay():
    from pathlib import Path
    records = np.load(Path(__file__).parent/'fixtures/bunny_contact_transition.npz')
    dimensions = np.diff(records['bounds'],axis=0)[0]
    inertia = .01/12*np.array([dimensions[1]**2+dimensions[2]**2,
        dimensions[0]**2+dimensions[2]**2,dimensions[0]**2+dimensions[1]**2])
    param = rigid_model().param_
    param.max_ncon_, param.sim_obj_inertia_diag_ = 10, inertia
    model = CartesianRigidContactModel(param)
    x, ref, u = cs.SX.sym('state',19), cs.SX.sym('ref',19), cs.SX.sym('u',3)
    phi, jac = cs.SX.sym('phi',40), cs.SX.sym('jac',40,9)
    from models.contact_geometry import contact_geometry_cs
    gaps, rows = contact_geometry_cs(x[:10],ref[:10],phi,jac)
    step = cs.Function('replay',[x,u,phi,jac,ref],[model.prediction_fn(x,u,gaps,rows,7.)])
    return records,model,step


def tilt(state):
    return np.degrees(np.arccos(np.clip(Rotation.from_quat(
        np.asarray(state)[[4,5,6,3]]).apply([0,0,1])[2],-1,1)))


def test_recorded_bunny_tip_response_and_release_fall():
    # Real baseline Isaac observations, seed 0, 20 ms. These points cover
    # loaded motion and the 44-degree flip that fell after contact was lost.
    records,model,step = bunny_replay()
    errors,old_errors = [],[]
    for i in range(4):
        state = model.observation_state(records['q'][i],records['linear'][i],
            records['angular'][i],records['tip_velocity'][i])
        predicted = np.asarray(step(state,records['actions'][i,0],
            records['phi'][i],records['jac'][i],state)).ravel()
        errors.append(np.linalg.norm(predicted[7:10]-records['observed'][i,0,7:10]))
        old_errors.append(np.linalg.norm(records['legacy_next'][i,7:10]-records['observed'][i,0,7:10]))
    assert np.mean(errors) < .0002
    assert np.mean(errors) < np.mean(old_errors)/10
    i = 4
    state = model.observation_state(records['q'][i],records['linear'][i],
        records['angular'][i],records['tip_velocity'][i])
    reference = state.copy()
    for action in records['actions'][i]:
        state = np.asarray(step(state,action,records['phi'][i],records['jac'][i],reference)).ravel()
    assert abs(tilt(state)-tilt(records['observed'][i,-1])) < 3.
    assert tilt(state) > tilt(reference)+10.


def test_rigid_contact_torch_matches_casadi_and_has_finite_control_gradient():
    import torch
    records,model,_ = bunny_replay()
    for i in (0,4):
        state = model.observation_state(records['q'][i],records['linear'][i],
            records['angular'][i],records['tip_velocity'][i])
        x = torch.tensor(state,dtype=torch.float64)
        u = torch.tensor(records['actions'][i,0],dtype=torch.float64,requires_grad=True)
        phi,jac = records['phi'][i],records['jac'][i]
        actual = model.prediction_torch(x,u,torch.tensor(phi),torch.tensor(jac))
        expected = np.asarray(model.prediction_fn(state,u.detach().numpy(),phi,jac,7.)).ravel()
        np.testing.assert_allclose(actual.detach(),expected,atol=1e-9)
        actual[:10].sum().backward()
        assert torch.isfinite(u.grad).all()


def test_future_table_support_follows_collision_hull_after_roll():
    from models.contact_geometry import refresh_table_support
    vertices = np.array(np.meshgrid(*([[-.04,.04]]*3))).T.reshape(-1,3)
    rotation = Rotation.from_euler('y',30,degrees=True)
    state = np.r_[[0,0,.04*(np.cos(np.pi/6)+np.sin(np.pi/6))],
        rotation.as_quat()[[3,0,1,2]],[.1,0,.1]]
    phi,jac = np.ones(12),np.zeros((12,9))
    jac[:4,6] = 1.
    refreshed,rows = refresh_table_support(vertices,state,phi,jac,0.,.5)
    np.testing.assert_array_equal(rows[:4],jac[:4])
    np.testing.assert_array_equal(refreshed[:4],phi[:4])
    assert refreshed[4] == pytest.approx(0.,abs=1e-12)
    # The two low corners form the support edge. Padding stays inactive.
    assert np.count_nonzero(np.linalg.norm(rows.reshape(3,4,9),axis=(1,2))) == 3
    assert rows[4,3:6] @ np.array([0,1,0]) != pytest.approx(0.)


def test_flip_region_preserves_other_objects_and_honors_explicit_override():
    args = SimpleNamespace(obj='elephant', pos_coef=500., ori_coef=20.,
                           two_stage_object_cost=True, flip_position_radius=None)
    assert ObjectCostStages(args).config()[9] == 0.
    args.obj = 'stanford_bunny2'
    assert ObjectCostStages(args).config()[9] == .08
    args.flip_position_radius = 0.
    assert ObjectCostStages(args).config()[9] == 0.
    args.obj, args.flip_position_radius = 'elephant', .04
    assert ObjectCostStages(args).config()[9] == .04
