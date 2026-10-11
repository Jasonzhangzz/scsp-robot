import casadi as cs
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from models.contact_geometry import contact_gaps_cs, contact_gaps_torch


def geometry_function():
    state,ref = cs.SX.sym('state',10),cs.SX.sym('reference',10)
    phi,jac = cs.SX.sym('phi',8),cs.SX.sym('jac',8,9)
    gaps = contact_gaps_cs(state,ref,phi,jac)
    return cs.Function('gaps_test',[state,ref,phi,jac],[gaps,cs.jacobian(gaps,state)])


def test_withdrawal_opens_only_normal_gap_and_padding_is_preserved():
    fn = geometry_function()
    reference = np.r_[np.zeros(3),[1.,0.,0.,0.],[.05,0.,.04]]
    jac = np.zeros((8,9))
    pyramid = np.array([[-1.,.5,0.],[-1.,0.,.5],[-1.,-.5,0.],[-1.,0.,-.5]])
    jac[:4,:3],jac[:4,6:] = pyramid,-pyramid
    phi = np.r_[np.zeros(4),np.ones(4)]
    state = reference.copy(); state[7] += .02; state[8] += .1
    actual,derivative = fn(state,reference,phi,jac)
    np.testing.assert_allclose(np.array(actual).ravel(),[.02]*4+[1.]*4)
    assert np.isfinite(derivative).all()
    # No tangential motion may create a normal contact bias.
    state[7] = reference[7]
    np.testing.assert_allclose(fn(state,reference,phi,jac)[0],phi[:,None])


@pytest.mark.parametrize('sign',[-1.,1.])
def test_body_rotation_and_torch_derivatives_agree_at_arbitrary_world_frame(sign):
    torch = pytest.importorskip('torch')
    fn = geometry_function()
    base = Rotation.from_euler('xyz',[35,-20,80],degrees=True)
    relative = Rotation.from_euler('y',5,degrees=True)
    reference = np.r_[[.4,.1,.4],base.as_quat()[[3,0,1,2]],[.43,.11,.45]]
    state = reference.copy();state[3:7] = sign*(base*relative).as_quat()[[3,0,1,2]]
    jac = np.zeros((8,9));jac[:4,4] = -.04  # normal gap lever arm in BODY axes
    phi = np.r_[np.zeros(4),np.ones(4)]
    actual,derivative = fn(state,reference,phi,jac)
    np.testing.assert_allclose(np.array(actual).ravel()[:4],-.08*np.sin(np.deg2rad(2.5)),atol=1e-12)
    x = torch.tensor(state,dtype=torch.float64,requires_grad=True)
    args = [torch.tensor(v,dtype=torch.float64) for v in (reference,phi,jac)]
    value = contact_gaps_torch(x,*args)
    np.testing.assert_allclose(value.detach().numpy(),np.array(actual).ravel(),atol=1e-12)
    value.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(),np.array(derivative).sum(axis=0),atol=1e-12)
    assert torch.isfinite(x.grad).all()


def test_rti_publishes_executed_prediction_without_a_nonlinear_gate():
    from types import SimpleNamespace
    from planning.mpc_explicit import MPCExplicit
    current = np.r_[np.zeros(3),[1.,0.,0.,0.],[.05,0.,.04]]
    command = np.array([.004,0.,0.])
    reachable = current.copy();reachable[7:] += command
    ghost = reachable.copy();ghost[0] += .002
    class RTI:
        calls = 0
        def set(self,*a): pass
        def constraints_set(self,*a): pass
        def solve(self):
            self.calls += 1
            return 0  # QP reports success even on the wrong contact branch
        def get(self,k,field):
            if field == 'u': return command
            return current if k == 0 else (ghost if self.calls == 1 else reachable)
        def get_stats(self,*a): return 1
    solver = MPCExplicit.__new__(MPCExplicit)
    solver.param_ = SimpleNamespace(mpc_horizon_=1,n_cmd_=3,n_qpos_=10,
        robot_stiff_=600*np.eye(3),model_params=1.,runtime_object_cost_=True)
    solver.model = SimpleNamespace(runtime_parameters=lambda sigma:sigma)
    solver.acados_solver_ = RTI()
    solver.acados_solve_count = solver.acados_failure_count = 0
    solver.contact_force_limit,solver.update_contact_gaps = 3.,True
    solver.stage_prediction_fn = lambda *a:reachable
    solver.path_cost_fn,solver.final_cost_fn = lambda *a:0.,lambda *a:0.
    solver.stage_cost_on_next = False
    solver.nlp_lam_g0_ = cs.DM.zeros(11)
    result = solver._plan_once_acados(current,np.zeros(4),np.zeros((4,9)),np.zeros(7),
        None,-.005*np.ones(3),.005*np.ones(3))
    assert result is not None and solver.acados_solver_.calls == 1
    assert solver.last_acados_rti_iterations == 1
    assert solver.acados_failure_count == 0
    np.testing.assert_allclose(result['rollout_q'][0],reachable)


@pytest.mark.parametrize('sign', [-1., 1.])
def test_transported_contact_geometry_matches_rigid_patches_and_torch(sign):
    from models.contact_geometry import contact_geometry_cs, contact_geometry_torch
    from examples.mpc.franka.ik2.contact_frames import (
        support_patch_jacobians, contact_jacobian_angular_body_frame,
    )
    torch = pytest.importorskip('torch')
    initial = Rotation.from_euler('xyz', [15, 30, 70], degrees=True)
    predicted = initial * Rotation.from_euler('y', 25, degrees=True)
    origin, point = np.array([.4, .1, .45]), np.array([.04, -.03, -.05])
    R, S = initial.as_matrix(), predicted.as_matrix()
    reference = np.r_[origin, initial.as_quat()[[3, 0, 1, 2]], origin+R@np.array([-.05,0.,.02])]
    state = reference.copy()
    state[:3] += [.01, -.02, .03]
    state[3:7] = sign*predicted.as_quat()[[3, 0, 1, 2]]
    state[7:10] = state[:3]+S@np.array([-.05,0.,.02])
    jac = np.zeros((12, 9))
    world, _ = support_patch_jacobians(origin, R, origin+R@point, 9, .5)
    jac[:4] = contact_jacobian_angular_body_frame(world, R)
    # A body-fixed fingertip patch, with inward +X and a nonzero moment arm.
    p = np.array([-.04,0.,.02])
    skew = np.array([[0.,-p[2],p[1]], [p[2],0.,-p[0]], [-p[1],p[0],0.]])
    pyramid = np.array([[1.,.5,0.], [1.,0.,.5], [1.,-.5,0.], [1.,0.,-.5]])
    jac[4:8] = np.c_[pyramid@R.T, -pyramid@skew, -pyramid@R.T]
    phi = np.r_[[.001]*4, [0.]*4, [1.]*4]
    x, ref = cs.SX.sym('x',10), cs.SX.sym('ref',10)
    gaps, rows = contact_geometry_cs(x,ref,cs.DM(phi),cs.DM(jac))
    fn = cs.Function('transport_test',[x,ref],[gaps,rows,cs.jacobian(gaps,x)])
    measured = fn(reference,reference)
    np.testing.assert_allclose(measured[0],phi[:,None],atol=1e-12)
    np.testing.assert_allclose(measured[1],jac,atol=1e-12)
    actual, changed, derivative = fn(state,reference)
    expected_world, _ = support_patch_jacobians(state[:3],S,state[:3]+S@point,9,.5)
    np.testing.assert_allclose(np.array(changed)[:4],contact_jacobian_angular_body_frame(expected_world,S),atol=1e-12)
    np.testing.assert_allclose(np.array(actual).ravel()[:4],.001+(state[:3]+S@point-origin-R@point)[2],atol=1e-12)
    np.testing.assert_allclose(np.array(actual).ravel()[4:], [0.]*4+[1.]*4, atol=1e-12)
    np.testing.assert_allclose(np.array(changed)[4:8,:3],pyramid@S.T,atol=1e-12)
    np.testing.assert_allclose(np.array(changed)[8:],0.,atol=1e-12)
    tip_withdrawn = state.copy();tip_withdrawn[7:10] -= .006*S[:,0]
    np.testing.assert_allclose(np.array(fn(tip_withdrawn,reference)[0]).ravel()[4:8],.006,atol=1e-12)
    tx = torch.tensor(state,dtype=torch.float64,requires_grad=True)
    tg, tj = contact_geometry_torch(tx,torch.tensor(reference),torch.tensor(phi),torch.tensor(jac))
    np.testing.assert_allclose(tg.detach(),np.array(actual).ravel(),atol=1e-12)
    np.testing.assert_allclose(tj.detach(),changed,atol=1e-12)
    tg.sum().backward()
    np.testing.assert_allclose(tx.grad,np.array(derivative).sum(axis=0),atol=1e-12)
