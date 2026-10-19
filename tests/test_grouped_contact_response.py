"""Four pyramid rows describe one physical contact, including padding."""
import numpy as np
import casadi as cs
import pytest
from models.soft_scm_contact import (grouped_contact_velocity_cs,
    grouped_contact_velocity_numpy, grouped_contact_velocity_torch,
    soft_scm_coefficients)


def table(mu):
    n, t1, t2 = np.eye(3)[2], np.eye(3)[0], np.eye(3)[1]
    j = np.zeros((40, 6))
    j[:4, :3] = np.stack([n+mu*t1, n+mu*t2, n-mu*t1, n-mu*t2])
    return j


@pytest.mark.parametrize('mu', [0., .5, .9])
def test_one_contact_does_not_duplicate_normal_support(mu):
    h = .02; a, b = soft_scm_coefficients(h)
    v = np.array([0., 0., -.2, 0., 0., 0.])
    qi = np.diag([.02]*3 + [20.]*3)
    actual = grouped_contact_velocity_numpy(v, table(mu), np.zeros(40), qi, h, a, b, 1e-8)
    assert actual[2] == pytest.approx(-.2*(1-a), abs=1e-7)
    assert -.2 <= actual[2] <= 0.  # gravity cannot create an upward bounce
    np.testing.assert_allclose(actual[[0,1,3,4,5]], 0., atol=1e-12)


def test_positive_gap_cannot_support_until_it_closes_within_the_step():
    import torch
    h=.02; a,b=soft_scm_coefficients(h)
    j,qi = table(.5),np.eye(6)
    speed = np.array([0.,0.,-.2,0.,0.,0.])  # closes only 4 mm in this step
    phi = np.ones(40);phi[:4] = .005
    numpy_v = grouped_contact_velocity_numpy(speed,j,phi,qi,h,a,b,1e-8)
    np.testing.assert_allclose(numpy_v,speed,atol=1e-12)
    value = grouped_contact_velocity_cs(cs.DM(speed),cs.DM(j),cs.DM(phi),cs.DM(qi),h,a,b,1e-8)
    np.testing.assert_allclose(np.array(value).ravel(),speed,atol=1e-12)
    value = grouped_contact_velocity_torch(*[torch.tensor(v,dtype=torch.float64)
        for v in (speed,j,phi,qi)],h,a,b,1e-8)
    np.testing.assert_allclose(value.numpy(),speed,atol=1e-12)
    phi[:4] = .003  # impact is reachable, so support is now possible
    assert grouped_contact_velocity_numpy(speed,j,phi,qi,h,a,b,1e-8)[2] > speed[2]


@pytest.mark.parametrize('speed', [[.3,.4,-.2,0,0,0], [0,0,0,0,0,0]])
def test_grouped_backends_and_batch_agree_and_have_finite_derivatives(speed):
    import torch
    j, qi = table(.5), np.diag([.02]*3+[20.]*3)
    h = .02; a,b = soft_scm_coefficients(h)
    phi = np.ones(40); phi[:4] = -.001
    expected = grouped_contact_velocity_numpy(np.array(speed), j, phi, qi, h, a,b,1e-8)
    x = cs.SX.sym('speed',6)
    result = grouped_contact_velocity_cs(x,cs.DM(j),cs.DM(phi),cs.DM(qi),h,a,b,1e-8)
    fn = cs.Function('group_contact_test',[x],[result,cs.jacobian(result,x)])
    value, derivative = fn(speed)
    np.testing.assert_allclose(np.array(value).reshape(-1),expected,atol=1e-10)
    assert np.isfinite(derivative).all()
    v = torch.tensor(speed,dtype=torch.float64,requires_grad=True)
    args = [torch.tensor(arg,dtype=torch.float64) for arg in [j,phi,qi]]
    value = grouped_contact_velocity_torch(v,*args,h,a,b,1e-8)
    np.testing.assert_allclose(value.detach().numpy(),expected,atol=1e-10)
    value.sum().backward(); assert torch.isfinite(v.grad).all()
    batch = grouped_contact_velocity_torch(v.detach().repeat(3,1),*args,h,a,b,1e-8)
    np.testing.assert_allclose(batch.numpy(),np.tile(expected,(3,1)),atol=1e-10)


def test_tangential_response_is_limited_by_single_normal_impulse():
    h=.02; a,b=soft_scm_coefficients(h)
    speed=np.array([10.,10.,-.2,0,0,0])
    qi=np.eye(6)
    actual=grouped_contact_velocity_numpy(speed,table(.5),np.zeros(40),qi,h,a,b,1e-8)
    impulse=actual-speed
    assert np.linalg.norm(impulse[:2]) <= .5*impulse[2]+1e-9
    speed[2]=.2
    np.testing.assert_allclose(grouped_contact_velocity_numpy(speed,table(.5),np.zeros(40),qi,h,a,b,1e-8),speed)


def test_downward_press_couples_to_off_center_table_reaction():
    from examples.mpc.franka.runtime.contact import support_patch_jacobians
    h = .02; a, b = soft_scm_coefficients(h)
    qi = np.diag([.02]*3 + [20.]*3 + [1/600.]*3)
    support, _ = support_patch_jacobians(np.zeros(3), np.eye(3), [.02, 0, -.04], 9, 0.)
    tip = np.zeros((4, 9))
    tip[:, 2], tip[:, 4], tip[:, 8] = -1., -.02, 1.
    jac = np.vstack([support, tip])
    def predict(force, contacts):
        free = np.r_[0, 0, -.098, 0, 0, 0, 0, 0, -force/(h*600)]
        return grouped_contact_velocity_numpy(free, contacts, np.zeros(len(contacts)), qi, h, a, b, 1e-8)
    weak, strong = predict(.8, jac), predict(3., jac)
    assert abs(strong[4]) > 2 * abs(weak[4])
    assert abs(strong[4]) > 2 * abs(predict(3., tip)[4])
    np.testing.assert_allclose(strong, predict(3., np.vstack([tip, support])), atol=1e-8)
    # Coupled derivatives and batching must preserve the same reaction.
    x = cs.SX.sym('driven', 9)
    output = grouped_contact_velocity_cs(x, cs.DM(jac), cs.DM.zeros(8), cs.DM(qi), h, a, b, 1e-8)
    fn = cs.Function('coupled_press', [x], [output, cs.jacobian(output, x)])
    free = np.r_[0, 0, -.098, 0, 0, 0, 0, 0, -3/(h*600)]
    value, derivative = fn(free)
    np.testing.assert_allclose(np.array(value).ravel(), strong, atol=1e-9)
    assert np.isfinite(derivative).all()
    assert abs(float(derivative[4, 8])) > 1.  # table-supported rotation responds to the tip
    np.testing.assert_allclose(grouped_contact_velocity_numpy(np.tile(free, (2, 1)), jac,
        np.zeros(8), qi, h, a, b, 1e-8), np.tile(strong, (2, 1)), atol=1e-9)
