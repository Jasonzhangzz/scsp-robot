"""Analytic mechanics and invariance checks for the unsmoothed Coulomb NCP."""
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from models.ncp_contact import NCPSettings, natural_map, solve_ncp


def solve(free, mu=.5, gap=0., J=None, inverse=None, settings=NCPSettings()):
    free=np.asarray(free,float)
    J=np.eye(len(free)).reshape(-1,3,len(free)) if J is None else J
    gap=np.broadcast_to(gap,(len(J),))
    inverse=np.eye(len(free)) if inverse is None else inverse
    v,p,d=solve_ncp(free,J,gap,inverse,.02,mu,settings)
    return v,p,d


@pytest.mark.parametrize('tangent', [[.1,.2],[3.,4.],[-4.,3.]])
def test_single_contact_exact_disk_and_no_dilatancy(tangent):
    v,p,d=solve([-1.,*tangent])
    assert d['converged'], d
    np.testing.assert_allclose(p[0,0],1.,atol=1e-9)
    expected=-np.asarray(tangent)*min(1.,.5/np.linalg.norm(tangent))
    np.testing.assert_allclose(p[0,1:],expected,atol=1e-9)
    assert abs(v[0])<1e-9  # associated cone-QP would cause positive normal velocity
    assert d['friction_dissipation_defect_Ns_m_s']<1e-9


def test_signed_gap_separation_and_penetration():
    for gap,free,expected in [(.04,-1.,0.),(-.002,0.,.1),(0.,1.,0.)]:
        v,p,d=solve([free,0.,0.],gap=gap)
        assert d['converged']
        np.testing.assert_allclose(p[0,0],expected,atol=1e-9)


def test_full_off_diagonal_mass_coupling():
    inverse=np.array([[2.,.7,.2],[.7,1.5,.4],[.2,.4,1.]])
    free=np.array([-.5,1.2,-.3])
    v,p,d=solve(free,inverse=inverse)
    assert d['converged'],d
    np.testing.assert_allclose(v,free+inverse@p.ravel(),atol=1e-12)
    assert abs(v[0])<1e-8
    assert np.linalg.norm(p[0,1:])==pytest.approx(.5*p[0,0],abs=1e-8)
    assert p[0,1]*v[2]-p[0,2]*v[1]==pytest.approx(0.,abs=1e-8)
    diagonal_result=solve(free,inverse=np.diag(np.diag(inverse)))[0]
    assert np.linalg.norm(v-diagonal_result)>.01


def test_tangent_basis_rotation_invariance():
    angle=.63
    R=np.eye(3);R[1:,1:]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    free=np.array([-.7,1.,.8])
    original=solve(free)
    transformed=solve(free,J=R[None])
    assert original[2]['converged'] and transformed[2]['converged']
    np.testing.assert_allclose(original[0],transformed[0],atol=1e-8)
    np.testing.assert_allclose(original[1][0],R.T@transformed[1][0],atol=1e-8)


def test_redundant_contacts_preserve_total_impulse_and_velocity():
    J=np.repeat(np.eye(3)[None],4,axis=0)
    v,p,d=solve([-1.,3.,4.],J=J)
    assert d['converged'],d
    np.testing.assert_allclose(p.sum(0),[1.,-.3,-.4],atol=1e-8)
    np.testing.assert_allclose(v,[0.,2.7,3.6],atol=1e-8)
    assert d['n_contacts']==4


def test_distinct_contacts_retain_cross_contact_delassus_blocks():
    root3=np.sqrt(3.)
    J=np.array([np.eye(3),[[.5,root3/2,0.],[-root3/2,.5,0.],[0.,0.,1.]]])
    v,p,d=solve([-1.,-1./root3,0.],mu=0.,J=J)
    assert d['converged'],d
    np.testing.assert_allclose(v,0.,atol=1e-8)
    np.testing.assert_allclose(p[:,0],[2./3,2./3],atol=1e-8)


def test_analytic_generalized_jacobian_away_from_kinks():
    rng=np.random.default_rng(12)
    for _ in range(10):
        A=rng.normal(size=(6,6));W=A@A.T+.2*np.eye(6)
        b=rng.normal(size=6);p=rng.normal(size=6);p[::3]=[.3,1.1]
        mu=np.array([.5,.7]);rho=np.array([.2,.4])
        _,jac=natural_map(p,W,b,mu,rho,jacobian=True)
        fd=approx_derivative(lambda x:natural_map(x,W,b,mu,rho),p)
        np.testing.assert_allclose(jac,fd,atol=1e-7,rtol=1e-6)


def test_nonconvergence_is_reported_even_with_finite_iterate():
    _,p,d=solve([-.5,1.2,-.3],inverse=np.array([[2.,.7,.2],[.7,1.5,.4],[.2,.4,1.]]),
                settings=NCPSettings(max_evaluations=1,max_projection_iterations=0))
    assert np.isfinite(p).all()
    assert not d['converged'] and d['residual_m_s']>1e-7


def test_free_flight_and_frictionless_limit():
    free=np.array([-.3,2.,-1.])
    v,p,d=solve(free,J=np.empty((0,3,3)),gap=np.empty(0))
    np.testing.assert_array_equal(v,free)
    assert d['converged'] and p.shape==(0,3)
    v,p,d=solve(free,mu=0.)
    assert d['converged']
    np.testing.assert_allclose(v,[0.,2.,-1.],atol=1e-9)


def test_invalid_problem_rejected():
    with pytest.raises(ValueError): solve([0.,0.,0.],inverse=-np.eye(3))
    with pytest.raises(ValueError): solve([0.,0.,0.],mu=-.5)
