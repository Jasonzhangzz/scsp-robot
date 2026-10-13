"""Franka CSO forces must be representable by the MPC contact force rays."""
import casadi as cs
import numpy as np
import pytest
from scipy.optimize import linprog

from planning.mlqp_point import LambdaContactControlOptimizer


def optimizer(cone):
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.friction_cone = cone
    opt._friction_force_cap_factor = 1.0 if cone == 'pyramid' else 2.0
    opt.mu_arm_obj = 1.2
    return opt


@pytest.mark.parametrize('tangents', [(0., 0.), (1.2, 0.), (-1.2, 0.),
                                    (.6, .6), (-.6, .6), (.9, .9), (-.9, -.9)])
def test_native_cone_matches_nonnegative_mpc_force_ray_combination(tangents):
    opt = optimizer('pyramid')
    force = np.r_[1., tangents]
    feasible = bool(np.all(np.asarray(opt._friction_residuals_cs(cs.DM(force), 1.2)) <= 1e-10))
    # Columns are the actual four MPC force rays in the contact frame.
    rays = np.array([[1., 1., 1., 1.], [1.2, 0., -1.2, 0.], [0., 1.2, 0., -1.2]])
    represented = linprog(np.zeros(4), A_eq=rays, b_eq=force,
                         bounds=(0, None), method='highs').success
    assert feasible is represented


def test_diagonal_box_corner_is_rejected_by_franka_cone():
    force = cs.DM([1., 1.2, 1.2])
    assert np.all(np.asarray(optimizer('box')._friction_residuals_cs(force, 1.2)) <= 0)
    assert np.any(np.asarray(optimizer('pyramid')._friction_residuals_cs(force, 1.2)) > 0)


@pytest.mark.parametrize('cone', ['box', 'pyramid'])
def test_rti_projection_respects_cone_and_total_force_cap(cone):
    opt = optimizer(cone)
    rng = np.random.default_rng(42)
    for proposal in rng.normal(size=(100, 3))*10:
        force = opt._project_contact_force(proposal, 3.)
        assert force[0] >= 0
        assert np.linalg.norm(force) <= 3.+1e-10
        assert np.all(np.asarray(opt._friction_residuals_cs(cs.DM(force), 1.2)) <= 1e-10)
    # Already feasible inputs retain both direction and magnitude.
    force = np.array([1., .3, -.4])
    np.testing.assert_allclose(opt._project_contact_force(force, 3.), force)


def test_torch_single_and_batch_controls_share_native_cone_and_finite_gradients():
    torch = pytest.importorskip('torch')
    opt = optimizer('pyramid')
    raw = torch.tensor([[0., 0.], [3., -3.], [.2, .4]], requires_grad=True)
    tangent = opt._bounded_tangents_torch(raw)
    assert torch.all(tangent.abs().sum(dim=1) <= 1.+1e-7)
    torch.testing.assert_close(opt._bounded_tangents_torch(raw[1]), tangent[1])
    tangent.square().sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_runtime_force_cap_retains_pyramid_bound():
    opt = optimizer('pyramid')
    opt.set_contact_force_limit(3.)
    assert opt.max_normal_force == pytest.approx(3./np.sqrt(1.+1.2**2))
