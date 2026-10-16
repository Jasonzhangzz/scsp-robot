"""Physical diagonal response, frozen-geometry derivatives and backend parity."""
import casadi as cs
import numpy as np
import pytest

from models.diagonal_contact import (diagonal_contact_velocity_numpy,
    diagonal_contact_velocity_cs, diagonal_contact_velocity_torch)


def rays(n, t1, t2, mu=.5):
    return np.stack((n+mu*t1, n+mu*t2, n-mu*t1, n-mu*t2))


def geometry():
    j = []
    for x, y in [(-.03, -.02), (.03, -.02), (0., .03)]:
        point = np.array([x, y, -.04])
        basis = np.eye(3)[[2, 0, 1]]
        spatial = np.c_[basis, np.cross(point, basis)]
        j.extend(rays(*spatial))
    return np.array(j)


def test_single_contact_impulse_is_mass_times_closing_velocity_and_no_history():
    j = rays(*np.eye(3)[[2, 0, 1]], mu=0.)
    free = np.array([0., 0., -.3])
    inverse = np.eye(3)/.2
    first, p = diagonal_contact_velocity_numpy(free, j, np.zeros(4), inverse, .002, return_impulses=True)
    np.testing.assert_allclose(first, 0., atol=1e-9)
    assert p[0][0] == pytest.approx(.2*.3, abs=1e-9)
    duplicate = diagonal_contact_velocity_numpy(free, np.tile(j, (2, 1)), np.zeros(8), inverse, .002)
    np.testing.assert_allclose(duplicate, first, atol=1e-9)
    # No state or parameter learned by an intervening query.
    diagonal_contact_velocity_numpy(-free, j, np.zeros(4), inverse, .002)
    np.testing.assert_array_equal(first, diagonal_contact_velocity_numpy(free,j,np.zeros(4),inverse,.002))


def test_positive_gap_cannot_support_an_unreachable_contact():
    j = rays(*np.eye(3)[[2, 0, 1]])
    free = np.array([.1, .2, -.1])
    result, impulses = diagonal_contact_velocity_numpy(free,j,np.full(4,.01),np.eye(3),.002,return_impulses=True)
    np.testing.assert_array_equal(result, free)
    np.testing.assert_array_equal(np.r_[impulses[0][0],impulses[0][1]], 0.)


@pytest.mark.parametrize('passes', [1, 4])
def test_numpy_casadi_torch_values_and_jacobians_match(passes):
    torch = pytest.importorskip('torch')
    j = geometry()
    # Full, coupled generalized mass, not a diagonal-inertia-only shortcut.
    inverse = np.diag([5., 5., 5., 3000., 2500., 2000.])
    inverse[0, 4] = inverse[4, 0] = .4
    gap = np.full(len(j), -.0001)
    velocity = np.array([.23, -.12, -.24, .1, -.2, .15])
    v = cs.SX.sym('v', 6)
    out = diagonal_contact_velocity_cs(v, cs.DM(j), cs.DM(gap), cs.DM(inverse), .002, passes=passes)
    fn = cs.Function('diagonal_prior', [v], [out, cs.jacobian(out, v)])
    expected, jac = fn(velocity)
    result, impulses = diagonal_contact_velocity_numpy(velocity,j,gap,inverse,.002,passes=passes,return_impulses=True)
    np.testing.assert_allclose(result, np.array(expected).ravel(), atol=1e-13)
    for normal, tangent in impulses:
        assert normal >= 0
        assert np.max(np.abs(tangent)) <= .5*normal+1e-14
    to = lambda x: torch.as_tensor(x, dtype=torch.float64)
    vt = to(velocity).requires_grad_(True)
    forward = lambda x: diagonal_contact_velocity_torch(x,to(j),to(gap),to(inverse),.002,passes=passes)
    np.testing.assert_allclose(forward(vt).detach(), result, atol=1e-13)
    np.testing.assert_allclose(torch.autograd.functional.jacobian(forward,vt).detach(),np.array(jac),atol=1e-12)
    epsilon = 1e-6
    finite = np.column_stack([(np.array(fn(velocity+epsilon*e)[0])-np.array(fn(velocity-epsilon*e)[0])).ravel()/(2*epsilon) for e in np.eye(6)])
    np.testing.assert_allclose(finite,np.array(jac),atol=1e-8)
    batch = np.stack([velocity, .9*velocity, 1.1*velocity])
    np.testing.assert_allclose(diagonal_contact_velocity_numpy(batch,j,gap,inverse,.002,passes=passes),
        forward(to(batch)).detach(),atol=1e-13)


def test_frozen_geometry_velocity_map_has_zero_hessian_inside_each_branch():
    v = cs.SX.sym('v', 6)
    j = cs.DM(geometry())
    result = diagonal_contact_velocity_cs(v,j,cs.DM.zeros(12),cs.DM.eye(6),.002)
    hessians = cs.vertcat(*[cs.reshape(cs.hessian(result[i],v)[0],36,1) for i in range(6)])
    np.testing.assert_array_equal(np.array(cs.Function('hessian',[v],[hessians])([.2,.1,-.3,.1,.2,.1])),0.)


def test_single_pass_cannot_increase_kinetic_energy_over_free_prediction():
    rng = np.random.default_rng(2701)
    j = geometry()
    for _ in range(50):
        factor = rng.normal(size=(6, 6))
        mass = factor.T@factor+np.diag([.1, .1, .1, .01, .01, .01])
        inverse = np.linalg.inv(mass)
        free = rng.normal(size=6)
        gap = np.repeat(rng.uniform(-.005, .005, 3), 4)
        velocity = diagonal_contact_velocity_numpy(free,j,gap,inverse,.002)
        assert velocity@mass@velocity <= free@mass@free+1e-11
