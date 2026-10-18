"""ComFree conformance, backend parity and differentiable contact regression."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import casadi as cs
import numpy as np
import pytest
import torch
import warp as wp

from models.comfree_contact import row_acceleration, reduced_row_masses
from models.explicit_model import ExplicitModel
from planning.mlqp_point import LambdaContactControlOptimizer


@pytest.fixture(scope="module")
def upstream(tmp_path_factory):
    # Execute the unmodified upstream kernel without importing its MuJoCo-3.6
    # package initializer in this MuJoCo-3.11 environment.
    path = Path(__file__).resolve().parents[1] / "thirdparty/comfree_warp/comfree_warp/comfree_core/_src/forward.py"
    source = path.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                and n.name == "_compute_qfrc_constraint")
    module_path = tmp_path_factory.mktemp("asu_kernel") / "upstream_comfree_kernel.py"
    module_path.write_text('import warp as wp\nwp.set_module_options({"enable_backward": False})\n@wp.kernel\n' + ast.get_source_segment(source, node))
    spec = importlib.util.spec_from_file_location("upstream_comfree_kernel", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._compute_qfrc_constraint


def test_row_forces_and_generalized_wrench_match_original_upstream_kernel(upstream):
    from models.explicit_model_warp import _compute_qfrc_constraint
    rng = np.random.default_rng(17)
    J = rng.normal(size=(2, 8, 6)).astype(np.float32)
    velocity = rng.normal(size=(2, 6)).astype(np.float32)
    gaps = rng.uniform(-.006, .002, size=(2, 8)).astype(np.float32)
    masses = rng.uniform(.03, .4, size=(2, 8)).astype(np.float32)
    dt, stiffness, damping = np.array([.01, .004]), np.array([.2, .37]), np.array([.001, .002])
    def array(v):
        return wp.array(v, dtype=float, device="cpu")
    inputs = [array(dt), array(stiffness), array(damping), array(J), array(gaps), array(masses),
              array(velocity), 6, wp.array([8, 5], dtype=int, device="cpu")]
    results = []
    for kernel in (upstream, _compute_qfrc_constraint):
        force = wp.zeros((2, 8), dtype=float, device="cpu")
        wrench = wp.zeros((2, 6), dtype=float, device="cpu")
        wp.launch(kernel, dim=(2, 8), inputs=inputs, outputs=[force, wrench], device="cpu")
        results.append((force.numpy(), wrench.numpy()))
    np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-7)
    np.testing.assert_allclose(results[0][1], results[1][1], atol=3e-7)
    speed = np.einsum('wrd,wd->wr', J, velocity)
    expected = np.maximum(masses*row_acceleration(speed, gaps, dt[:, None], stiffness[:, None], damping[:, None]), 0)
    expected[1, 5:] = 0
    np.testing.assert_allclose(results[0][0], expected, atol=1e-7)


@pytest.mark.parametrize("dt", [.004, .03])
def test_lambda_numpy_casadi_torch_same_predicted_velocity_and_derivative(dt):
    optimizer = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    optimizer.h, optimizer.K_contact, optimizer.comfree_damping, optimizer.max_contacts = dt, .2, .001, 2
    optimizer.Q_inv = np.diag([5, 5, 5, 200, 180, 170])
    J = np.zeros((8, 6)); J[:4, 2] = 1
    J[:4, :2] = [[.7, 0], [-.7, 0], [0, .7], [0, -.7]]
    J[:4, 3:] = [[.01, -.02, 0]]*4
    optimizer.env_phi = np.r_[np.full(4, -.003), np.zeros(4)]
    masses = optimizer.compute_env_diag_inverse(J)
    smooth = np.array([.08, -.03, -.15, .2, -.1, .1])
    expected = optimizer._comfree_velocity_numpy(smooth, J, masses, optimizer.env_phi)
    symbol = cs.SX.sym('v', 6)
    expression = optimizer._comfree_velocity_cs(symbol, cs.DM(J), cs.DM(masses), cs.DM(optimizer.env_phi))
    fn = cs.Function('response', [symbol], [expression, cs.jacobian(expression, symbol)])
    actual, derivative = fn(smooth)
    np.testing.assert_allclose(actual.full().reshape(6), expected, atol=1e-12)
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    value = t(smooth).requires_grad_()
    result = optimizer._comfree_velocity_torch(value, t(J), t(masses), t(optimizer.env_phi), t(optimizer.Q_inv))
    np.testing.assert_allclose(result.detach().numpy(), expected, atol=1e-12)
    gradient = torch.autograd.functional.jacobian(
        lambda v: optimizer._comfree_velocity_torch(v, t(J), t(masses), t(optimizer.env_phi), t(optimizer.Q_inv)), value)
    np.testing.assert_allclose(gradient.numpy(), derivative.full(), atol=1e-12)
    # The row response uses the incoming velocity, rather than its increment.
    assert np.linalg.norm(expected-smooth) > .001
    assert np.all(masses[4:] == 0)


@pytest.mark.parametrize("model", ["comfree", "scm-corrected"])
def test_lambda_direct_models_apply_selected_wrench_without_arm_activation(model):
    """Lambda direct-force models bypass arm gating; MPC owns that response."""
    optimizer = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    optimizer.h = .01
    optimizer.Q_inv = np.diag([2., 3., 4., 5., 6., 7.])
    optimizer.max_contacts = 2
    optimizer.contact_model = model
    optimizer.K_contact = .2
    optimizer.comfree_damping = .001
    optimizer.soft_scm_regularization = 1e-8
    optimizer.robot_stiffness = 600.
    optimizer.gravity_velocity_scale = 1. / optimizer.h
    optimizer.wrench_velocity_scale = 1. / optimizer.h
    optimizer.base_mu_arm_obj = optimizer.mu_arm_obj = .8

    velocity = np.array([.02, -.03, .04, .01, -.02, .03])
    tau = np.array([.1, -.2, .3, .04, -.05, .06])
    force = np.array([.7, -.2, .4])
    point = np.array([.03, -.02, .01])
    frame = np.eye(3)
    env_j = np.zeros((8, 6))
    phi = np.array([-.02, -.01, 0., .01, .02, -.03, .01, 0.])
    inverse = optimizer.Q_inv
    skew = np.array([[0., -point[2], point[1]],
                     [point[2], 0., -point[0]],
                     [-point[1], point[0], 0.]])
    relative = np.c_[np.eye(3), -skew]
    expected_object = (velocity + optimizer.gravity_velocity_scale * inverse @ tau
                       + optimizer.wrench_velocity_scale * inverse @ relative.T @ force)

    numpy_result = optimizer._arm_velocity_numpy(
        velocity, tau, force, point, frame, env_j, phi)
    np.testing.assert_allclose(numpy_result, expected_object, atol=1e-12)

    cs_result_a = optimizer._arm_velocity_cs(
        cs.DM(velocity), cs.DM(tau), cs.DM(force), cs.DM(point), cs.DM(frame),
        cs.DM(env_j), cs.DM(phi), cs.DM(inverse), cs.DM.zeros(6, 1),
        arm_gap=-.2)
    cs_result_b = optimizer._arm_velocity_cs(
        cs.DM(velocity), cs.DM(tau), cs.DM(force), cs.DM(point), cs.DM(frame),
        cs.DM(env_j), cs.DM(phi), cs.DM(inverse), cs.DM.zeros(6, 1),
        arm_gap=.2)
    np.testing.assert_allclose(np.asarray(cs_result_a).reshape(6), expected_object, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cs_result_a), np.asarray(cs_result_b), atol=1e-12)

    tensor = lambda value: torch.tensor(value, dtype=torch.float64)
    torch_result = optimizer._arm_velocity_torch(
        tensor(velocity), tensor(tau), tensor(force), tensor(point), tensor(frame),
        tensor(env_j), tensor(phi), tensor(inverse))
    np.testing.assert_allclose(torch_result.numpy(), expected_object, atol=1e-12)


def test_explicit_model_is_physical_comfree_euler_and_normalizes_quaternion():
    mass = np.diag([.2, .2, .2, .003, .004, .005, 1, 1, 1])
    param = SimpleNamespace(n_qpos_=10, n_qvel_=9, n_robot_qpos_=3, n_cmd_=3, max_ncon_=1,
                            Q=mass, obj_mass_=.2, gravity_=np.array([0, 0, -9.81, 0, 0, 0]),
                            robot_stiff_=np.eye(3)*100, h_=.01, contact_model='comfree')
    model = ExplicitModel(param)
    q = np.array([0, 0, .1, 1, 0, 0, 0, .1, 0, .1], dtype=float)
    J = np.zeros((4, 9)); J[:2, 2] = 1; J[0, 4] = -.01; J[1, 4] = .01
    gaps = np.array([-.004, -.004, 0, 0]); cmd = np.array([.01, -.02, 0])
    smooth = param.h_*np.linalg.solve(mass, np.r_[param.obj_mass_*param.gravity_, param.robot_stiff_@cmd])
    masses = reduced_row_masses(J, np.linalg.inv(mass), gaps)
    force = np.maximum(masses*row_acceleration(J@smooth, gaps, param.h_), 0)
    velocity = smooth + param.h_*np.linalg.solve(mass, J.T@force)
    expected = np.asarray(model.cs_qposInteg_(q, velocity)).reshape(-1)
    actual = np.asarray(model.step(q, cmd, gaps, J, .2)).reshape(-1)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert np.linalg.norm(actual[3:7]) == pytest.approx(1)
    incoming = np.array([.02, -.04, -.1, .1, -.2, .3, .01, .02, .03])
    smooth_velocity = incoming + smooth
    reaction = np.maximum(masses*row_acceleration(J@smooth_velocity, gaps, param.h_), 0)
    expected_velocity = smooth_velocity + param.h_*np.linalg.solve(mass, J.T@reaction)
    next_position, next_velocity = model.step_state_fn(q, incoming, cmd, gaps, J, .2)
    np.testing.assert_allclose(np.asarray(next_velocity).reshape(-1), expected_velocity, atol=1e-12)
    np.testing.assert_allclose(np.asarray(next_position).reshape(-1), np.asarray(model.cs_qposInteg_(q, expected_velocity)).reshape(-1), atol=1e-12)
    no_contact = np.asarray(model.step(q, cmd, np.zeros(4), np.zeros((4, 9)), .2)).reshape(-1)
    assert no_contact[2] == pytest.approx(q[2] - .01**2*9.81)


def test_environment_gap_padding_is_explicit_and_validated():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.max_contacts = 2; opt.env_phi = np.zeros(8); opt.J_tilde = np.zeros((8, 6))
    opt.update_Jacobian(np.ones((4, 9)), phi=[-.002])
    np.testing.assert_allclose(opt.env_phi, [-.002]*4+[0]*4)
    opt.update_Jacobian(np.ones((4, 6)))
    assert np.all(opt.env_phi == 0)
    with pytest.raises(ValueError, match='phi'):
        opt.update_Jacobian(np.ones((4, 6)), phi=[np.nan])


def test_native_rows_use_asu_mass_margin_and_predicted_velocity():
    from models.dexforge_fast_step import ensure_comfree_warp
    ensure_comfree_warp()
    from models.comfree_native import _kernels
    from comfree_warp.native_adjoint.explicit_damp import (
        ExplicitDampParameters, CoefficientInputs, CoefficientOutputs, CoefficientReductions, vec5)
    from comfree_warp.comfree_core._src.types import L0_IMPEDANCE
    from models.comfree_contact import row_mass
    Parameters, _, kernel = _kernels(.2, .001)
    def array(value, dtype=float, grad=False):
        return wp.array(value, dtype=dtype, device='cpu', requires_grad=grad)
    native = ExplicitDampParameters()
    native.contact_row_count = 4
    native.contact_geom = array([[0, 1]], wp.vec2i)
    native.contact_solimp = array([[.9, .95, .0005, .5, 2]], vec5)
    native.row_contact = array([0, 0, 0, 0, -1], int)
    native.geom_bodyid = array([0, 1], int)
    native.body_invweight = array([[[0, 0], [5, 100]]], wp.vec2)
    native.jnt_solimp = array([[[.9, .95, .0005, .5, 2]]], vec5)
    native.jnt_dofadr = array([0], int)
    native.dof_invweight = array([[7]], float)
    native.limit_joint = array([0], int)
    for name, variable in native._cls.vars.items():
        if isinstance(variable.type, wp.array) and getattr(native, name) is None:
            setattr(native, name, wp.zeros((1,)*variable.type.ndim, dtype=variable.type.dtype, device="cpu"))
    params = Parameters(); params.native = native
    params.timestep = array([.01]); params.margin = array([.002, .001])
    params.row_count = params.row_capacity = 5; params.dof_count = 6
    position = np.array([[-.004, -.004, -.004, -.004, -.005]], dtype=np.float32)
    current = np.array([[-.1, .04, -.2, .08, -.3]], dtype=np.float32)
    acceleration = np.array([[-9, -4, 2, 3, -8]], dtype=np.float32)
    reductions = CoefficientReductions()
    reductions.current_velocity = array(current, grad=True)
    reductions.smooth_acceleration = array(acceleration, grad=True)
    inputs = CoefficientInputs()
    inputs.position = array(position, grad=True); inputs.active = array([[1]*5], int)
    inputs.reductions = reductions
    outputs = CoefficientOutputs()
    outputs.free_force = wp.zeros((1, 5), dtype=float, device='cpu', requires_grad=True)
    outputs.velocity_weight = wp.zeros((1, 5), dtype=float, device='cpu', requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(kernel, dim=(1, 5), inputs=[params, inputs], outputs=[outputs], device='cpu')
    distance = position + [[.002]*4+[.001]]
    mass = row_mass([[5]*4+[7]], position)
    expected = mass*row_acceleration(current+.01*acceleration, distance, .01)/L0_IMPEDANCE
    np.testing.assert_allclose(outputs.free_force.numpy(), expected, rtol=3e-6, atol=2e-6)
    assert np.all(outputs.velocity_weight.numpy() == 0)  # no iterative feedback
    tape.backward(grads={outputs.free_force: array(np.ones((1, 5)))})
    np.testing.assert_allclose(reductions.current_velocity.grad.numpy(), -mass*(.001/.01+.2)/L0_IMPEDANCE,
                               rtol=3e-6, atol=2e-6)
    np.testing.assert_allclose(reductions.smooth_acceleration.grad.numpy(),
                               reductions.current_velocity.grad.numpy()*.01, rtol=3e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for MuJoCo-Warp collision')
def test_system_warp_active_pyramid_has_upstream_row_masses():
    import mujoco
    from models.explicit_model_warp import ExplicitModelWarp
    from models.comfree_contact import row_mass
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".01"/>
      <default><geom friction=".7 .005 .0001"/></default><worldbody>
      <geom type="plane" size="1 1 .1"/><body pos="0 0 .039"><freejoint/>
      <geom type="sphere" size=".04" mass=".2"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model); data.qvel[2] = -.1; mujoco.mj_forward(model, data)
    step = ExplicitModelWarp(model, data, nconmax=8, njmax=16, device='cuda:0')
    step.forward()
    state = step.data_wp
    count = int(state.nefc.numpy()[0]); assert count == 4
    gaps = state.efc.pos.numpy()[0, :count]
    masses = state.efc.efc_mass.numpy()[0, :count]
    np.testing.assert_allclose(masses, row_mass(1/.2, gaps), rtol=3e-6)
    # All friction edges share the same body mass; no friction multiplier.
    np.testing.assert_allclose(masses, np.full(4, masses[0]), rtol=1e-7)
    speed = state.efc.J.numpy()[0, :count, :model.nv] @ state.qvel_smooth_pred.numpy()[0]
    expected_force = np.maximum(masses*row_acceleration(speed, gaps, .01), 0)
    np.testing.assert_allclose(state.efc.force.numpy()[0, :count], expected_force, rtol=3e-6, atol=1e-7)
