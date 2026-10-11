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
