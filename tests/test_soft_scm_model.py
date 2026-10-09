"""Planning contact laws: impulse units, effective mass and derivatives."""
from types import SimpleNamespace

import casadi as cs
import numpy as np
import pytest
import torch

from models.explicit_model import ExplicitModel
from models.soft_scm_contact import soft_scm_coefficients
from planning.mlqp_point import LambdaContactControlOptimizer


def _case(nv=9):
    rng = np.random.default_rng(62)
    # A dense SPD inverse tests cross-coordinate terms, not only diagonal M.
    factor = rng.normal(size=(nv, nv))
    inverse = factor @ factor.T + np.eye(nv)
    jac = rng.normal(size=(8, nv)) * .2
    jac[4:7, 6:] = 0  # environment rows have no moving robot block
    jac[-1] = 0  # deliberately negative gap in an inactive padding row
    phi = np.array([-.002, -.003, .01, -.004, -.001, .02, -.003, -1.0])
    incoming = rng.normal(size=nv)*.04
    return inverse, jac, phi, incoming


def _reference(smooth, jac, phi, inverse, h, tau, zeta, r):
    # Solve the independent implicit spring-damper row equations directly.
    mobility = np.einsum('ij,jk,ik->i', jac, inverse, jac)
    effective_mass = 1/(mobility + r)
    stiffness = effective_mass/tau**2
    damping = 2*zeta*effective_mass/tau
    impulse = np.maximum(-h*(stiffness*(phi+h*(jac@smooth)) + damping*(jac@smooth)) /
                         (1+h*damping/effective_mass+h*h*stiffness/effective_mass), 0)
    impulse[~np.any(jac != 0, axis=1)] = 0
    return smooth + inverse @ jac.T @ impulse


def _param(h, mass, mode='soft-scm'):
    return SimpleNamespace(n_qpos_=10, n_qvel_=9, n_robot_qpos_=3, n_cmd_=3, max_ncon_=2,
        Q=mass, obj_mass_=.2, gravity_=np.array([0, 0, -9.81, 0, 0, 0]),
        robot_stiff_=np.eye(3)*100, h_=h, contact_model=mode,
        soft_scm_response_time=.05, soft_scm_damping_ratio=1., soft_scm_regularization=1e-8)


@pytest.mark.parametrize('h', [.002, .02, .05])
@pytest.mark.parametrize('tau,zeta', [(.05, 1.), (.008, .25)])
def test_lambda_soft_response_matches_implicit_rows_and_backend_gradients(h, tau, zeta):
    inverse, jac, phi, smooth = _case(6)
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.contact_model, opt.h, opt.max_contacts = 'soft-scm', h, 2
    opt.Q_inv, opt.env_phi = inverse, phi
    opt.soft_scm_regularization = 1e-8
    opt.soft_scm_a, opt.soft_scm_b = soft_scm_coefficients(h, tau, zeta)
    weights = opt.compute_env_diag_inverse(jac)
    expected = _reference(smooth, jac, phi, inverse, h, tau, zeta, 1e-8)
    np.testing.assert_allclose(opt._contact_velocity_numpy(smooth, jac, weights, phi), expected, atol=1e-12)
    assert np.all(weights[-1] == 0)

    v = cs.SX.sym('v', 6)
    out = opt._contact_velocity_cs(v, cs.DM(jac), cs.DM(weights), cs.DM(phi))
    fn = cs.Function('response', [v], [out, cs.jacobian(out, v)])
    result, derivative = fn(smooth)
    np.testing.assert_allclose(result.full().ravel(), expected, atol=1e-12)
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    torch_fn = lambda v: opt._contact_velocity_torch(v, t(jac), t(weights), t(phi), t(inverse))
    np.testing.assert_allclose(torch_fn(t(smooth)).numpy(), expected, atol=1e-12)
    np.testing.assert_allclose(torch.autograd.functional.jacobian(torch_fn, t(smooth)).numpy(),
                               derivative.full(), atol=1e-12)
    batch = np.stack([smooth, smooth+.03, smooth-.07])
    expected_batch = np.stack([_reference(v, jac, phi, inverse, h, tau, zeta, 1e-8) for v in batch])
    np.testing.assert_allclose(torch_fn(t(batch)).numpy(), expected_batch, atol=1e-12)
    # Compare to central differences away from contact activation boundaries.
    eps = 1e-6
    finite = np.column_stack([(fn(smooth+eps*e)[0].full().ravel()-fn(smooth-eps*e)[0].full().ravel())/(2*eps)
                              for e in np.eye(6)])
    np.testing.assert_allclose(derivative.full(), finite, atol=1e-9)


@pytest.mark.parametrize('h', [.02, .05])
@pytest.mark.parametrize('mode', ['soft-scm', 'comfree'])
def test_explicit_step_and_torch_mpc_use_same_contact_law_and_command_gradient(h, mode):
    inverse, jac, phi, incoming = _case()
    param = _param(h, np.linalg.inv(inverse), mode)
    model = ExplicitModel(param)
    q = np.r_[0., 0., .1, 1., 0., 0., 0., .1, 0., .1]
    cmd = np.array([.002, -.001, .003])
    force = np.r_[param.obj_mass_*param.gravity_, param.robot_stiff_@cmd]
    smooth = incoming + h*inverse@force
    position, velocity = model.step_state_fn(q, incoming, cmd, phi, jac, .2)
    if mode == 'soft-scm':
        expected = _reference(smooth, jac, phi, inverse, h, .05, 1., 1e-8)
        np.testing.assert_allclose(velocity.full().ravel(), expected, atol=1e-12)
        # Legacy sigma must not silently rescale the new contact coefficients.
        np.testing.assert_array_equal(model.step_state_fn(q, incoming, cmd, phi, jac, .7)[1].full(), velocity.full())
    np.testing.assert_allclose(position.full().ravel(), np.asarray(model.cs_qposInteg_(q, velocity)).ravel(), atol=1e-12)
    assert np.linalg.norm(position.full().ravel()[3:7]) == pytest.approx(1.)
    np.testing.assert_allclose(model.step(q, cmd, phi, jac, .2).full(),
        model.step_state_fn(q, np.zeros(9), cmd, phi, jac, .2)[0].full(), atol=1e-12)

    t = lambda a: torch.tensor(a, dtype=torch.float64)
    def torch_fn(u):
        f = torch.cat((t(param.obj_mass_*param.gravity_), t(param.robot_stiff_)@u))
        return model.contact_velocity_torch(t(incoming)+h*t(inverse)@f, t(jac), t(phi), t(inverse), .2)
    np.testing.assert_allclose(torch_fn(t(cmd)).detach().numpy(), velocity.full().ravel(), atol=1e-12)
    u = cs.SX.sym('u', 3)
    v = model.step_state_fn(cs.DM(q), cs.DM(incoming), u, cs.DM(phi), cs.DM(jac), .2)[1]
    grad = cs.Function('command_gradient', [u], [cs.jacobian(v, u)])
    np.testing.assert_allclose(torch.autograd.functional.jacobian(torch_fn, t(cmd)).numpy(), grad(cmd).full(), atol=1e-12)
    no_contact = model.step_state_fn(q, incoming, cmd, phi, np.zeros_like(jac), .2)[1]
    np.testing.assert_allclose(no_contact.full().ravel(), smooth, atol=1e-12)


def test_soft_scm_approaches_original_scm_with_fast_response():
    a, b = soft_scm_coefficients(.02, 1e-9, 1.)
    assert a == pytest.approx(1.)
    assert b == pytest.approx(1., abs=2e-7)


@pytest.mark.parametrize('h', [.02, .05])
@pytest.mark.parametrize('mode', ['soft-scm', 'comfree'])
def test_quasi_dynamic_displacement_command_keeps_main_gain(h, mode):
    stiffness = np.diag([100., 150., 200.])
    metric = np.diag([50., 50., 50., .05, .05, .05, 100., 150., 200.])
    param = _param(h, metric, mode)
    param.contact_dynamics = 'quasi-dynamic'
    param.robot_stiff_ = stiffness
    param.gravity_ = np.zeros(6)
    model = ExplicitModel(param)
    q = np.r_[0., 0., .1, 1., 0., 0., 0., .1, 0., .1]
    cmd = np.array([.005, -.003, .002])
    jac, phi = np.zeros((8, 9)), np.zeros(8)
    next_q = model.step(q, cmd, phi, jac, .2).full().ravel()
    np.testing.assert_allclose(next_q[7:] - q[7:], cmd, atol=1e-15)
    # Checks the displacement Jacobian which controls approach optimization.
    u = cs.SX.sym('u', 3)
    expression = model.step_once_fn(q, u, phi, jac, .2)
    derivative = cs.Function('displacement_gain', [u], [cs.jacobian(expression[7:], u)])
    np.testing.assert_allclose(derivative(cmd).full(), np.eye(3), atol=1e-15)
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    def torch_step(u):
        force = torch.cat((torch.zeros(6, dtype=torch.float64), t(stiffness)@u))
        smooth = model.force_velocity_scale * t(np.linalg.inv(metric)) @ force
        return h * model.contact_velocity_torch(smooth, t(jac), t(phi), t(np.linalg.inv(metric)), .2)[6:]
    np.testing.assert_allclose(torch_step(t(cmd)).numpy(), cmd, atol=1e-15)
    np.testing.assert_allclose(torch.autograd.functional.jacobian(torch_step, t(cmd)).numpy(), np.eye(3), atol=1e-15)


@pytest.mark.parametrize('name,value', [('h_', 0), ('soft_scm_response_time', -1),
    ('soft_scm_damping_ratio', 0), ('soft_scm_regularization', float('nan'))])
def test_invalid_soft_parameters_are_rejected(name, value):
    param = _param(.02, np.eye(9))
    setattr(param, name, value)
    with pytest.raises(ValueError, match='soft-SCM'):
        ExplicitModel(param)


def test_ik2_parameter_factory_passes_contact_model_to_ranking(monkeypatch):
    from examples.mpc.franka.ik2 import params
    captured = {}
    def capture(**kwargs):
        captured.update(kwargs)
        return captured
    monkeypatch.setattr(params, 'LambdaContactControlOptimizer', capture)
    param = SimpleNamespace(mesh_path_='unused.stl', mu_object_=.5, model_params=.2,
        contact_model='soft-scm', soft_scm_response_time=.07,
        soft_scm_damping_ratio=.5, soft_scm_regularization=2e-8)
    args = SimpleNamespace(sample_num=8, pos_coef=1, ori_coef=.01, solver_backend='ipopt')
    params.build_lambda_optimizer(param, args)
    assert captured['contact_model'] == 'soft-scm'
    assert captured['soft_scm_response_time'] == .07
    assert captured['soft_scm_damping_ratio'] == .5
    assert captured['soft_scm_regularization'] == 2e-8
    assert captured['time_step'] == .05


@pytest.mark.parametrize('mode', ['soft-scm', 'comfree'])
def test_ik2_real_parameters_keep_displacement_units_and_shared_contact_settings(mode, monkeypatch):
    from examples.mpc.franka.ik2 import params
    monkeypatch.setattr(params, 'LambdaContactControlOptimizer', lambda **kwargs: SimpleNamespace(**kwargs))
    args = SimpleNamespace(contact_cost_param=1., attract_coef=.5, reject_coef=.001,
        contact_coef=.7, reject_dis=.02, obj='foam_brick', collision_hull=False,
        model_param=7., sample_num=8, pos_coef=500., ori_coef=20., solver_backend='ipopt',
        contact_model=mode, soft_scm_response_time=.07, soft_scm_damping_ratio=.5)
    param = params.ExplicitMPCParams(args)
    model = ExplicitModel(param)
    assert model.contact_dynamics == 'quasi-dynamic'
    assert param.lambda_optimizer.contact_model == model.contact_model == mode
    assert param.lambda_optimizer.soft_scm_response_time == model.soft_scm_response_time == .07
    assert param.lambda_optimizer.soft_scm_damping_ratio == model.soft_scm_damping_ratio == .5
    q = np.r_[param.init_obj_qpos_, param.init_fingertip_pos_]
    cmd = np.array([.005, -.001, .002])
    next_q = model.step(q, cmd, np.ones(40), np.zeros((40, 9)), param.model_params).full().ravel()
    np.testing.assert_allclose(next_q[7:] - q[7:], cmd, atol=1e-14)
