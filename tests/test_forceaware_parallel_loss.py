"""Parallel Panda cost and servo integration agree with the native formulas."""

import numpy as np
import pytest

wp = pytest.importorskip("warp")
from planning.dexforge_forceaware import _classes

try:
    api = _classes()
except ImportError as error:
    pytest.skip(f"DexForge runtime unavailable: {error}", allow_module_level=True)
mpc_loss = api.mpc_loss
from comfree_warp.native_adjoint.dynamics import DynamicsInput, DynamicsModel
from comfree_warp.native_adjoint.dynamics import vec10


@wp.kernel
def serial_cost(job: mpc_loss.LossJob, loss: wp.array(dtype=float)):
    world = wp.tid()
    loss[world] = 3.0 * job.state.body_position[world, 0][0] + mpc_loss._finger_joint_collision_loss(job, world)


@wp.kernel
def base_cost(job: mpc_loss.LossJob, loss: wp.array(dtype=float)):
    world = wp.tid()
    loss[world] = 3.0 * job.state.body_position[world, 0][0]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_parallel_625_pair_loss_and_vjp_match_original(device):
    wp.init()
    if not wp.is_device_available(device):
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(31)
    pairs = np.array([(i, j) for i in range(25) for j in range(25, 50)], np.int32)
    job = mpc_loss.LossJob()
    job.model.finger_joint_body = wp.array(np.arange(50) % 10, dtype=int, device=device)
    local = rng.normal(0, .012, (50, 3)).astype(np.float32)
    job.model.finger_joint_start = wp.array(local, dtype=wp.vec3, device=device)
    job.model.finger_joint_end = wp.array(local, dtype=wp.vec3, device=device)
    job.model.finger_joint_radius = wp.array(rng.uniform(.005, .035, 50), device=device, dtype=float)
    job.model.finger_joint_pair_left = wp.array(pairs[:, 0], dtype=int, device=device)
    job.model.finger_joint_pair_right = wp.array(pairs[:, 1], dtype=int, device=device)
    job.model.finger_joint_pair_count = len(pairs)
    job.weights.finger_joint_collision = 1.7
    job.weights.finger_joint_surface_margin_mm = 4.0
    position = rng.normal(0, .04, (2, 10, 3)).astype(np.float32)
    matrix = np.tile(np.eye(3, dtype=np.float32), (2, 10, 1, 1))
    job.state.body_position = wp.array(position, dtype=wp.vec3, device=device, requires_grad=True)
    job.state.body_matrix = wp.array(matrix, dtype=wp.mat33, device=device, requires_grad=True)
    for struct in (job.model, job.state, job.target, job.auxiliary):
        for name, var in struct._cls.vars.items():
            if hasattr(var.type, "dtype") and hasattr(var.type, "ndim") and getattr(struct, name) is None:
                setattr(struct, name, wp.zeros((0,) * var.type.ndim, dtype=var.type.dtype, device=device))
    results = []
    for parallel in (False, True):
        job.state.body_position.grad.zero_()
        job.state.body_matrix.grad.zero_()
        loss = wp.zeros(2, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(base_cost if parallel else serial_cost, dim=2, inputs=[job], outputs=[loss], device=device)
            if parallel:
                wp.launch(api.panda_pair_loss, dim=(2, len(pairs)), inputs=[job], outputs=[loss], device=device)
        tape.backward(grads={loss: wp.ones(2, device=device)})
        results.append((loss.numpy(), job.state.body_position.grad.numpy(), job.state.body_matrix.grad.numpy()))
    for expected, actual in zip(*results):
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=.05)


def test_implicit_servo_derivative_respects_torque_and_control_clipping(monkeypatch):
    model = DynamicsModel()
    model.damper_enabled = model.actuation_enabled = model.clamp_control = 1
    model.actuator_count = 1
    model.dof_damping = wp.array([[.3]], dtype=float, device="cpu")
    model.actuator_dofid = wp.array([0], dtype=int, device="cpu")
    model.actuator_qposadr = wp.array([0], dtype=int, device="cpu")
    model.actuator_gainprm = wp.array([[[10., 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=vec10, device="cpu")
    model.actuator_biasprm = wp.array([[[0., -20., -5., 0, 0, 0, 0, 0, 0, 0]]], dtype=vec10, device="cpu")
    model.actuator_gear = wp.array([[[2., 0, 0, 0, 0, 0]]], dtype=wp.spatial_vector, device="cpu")
    model.actuator_ctrllimited = wp.array([True], dtype=bool, device="cpu")
    model.actuator_forcelimited = wp.array([True], dtype=bool, device="cpu")
    model.actuator_ctrlrange = wp.array([[[-1., 1.]]], dtype=wp.vec2, device="cpu")
    model.actuator_forcerange = wp.array([[[-2., 2.]]], dtype=wp.vec2, device="cpu")
    inputs = DynamicsInput()
    inputs.qpos = wp.array([[.01], [.01], [.01]], dtype=float, device="cpu")
    inputs.qvel = wp.array([[.02], [.02], [.02]], dtype=float, device="cpu")
    inputs.ctrl = wp.array([[.04], [1.], [-1.]], dtype=float, device="cpu")
    damping = wp.empty((3, 1), device="cpu")
    wp.launch(api.servo_implicit_damping, dim=(3, 1), inputs=[model, inputs], outputs=[damping], device="cpu")
    np.testing.assert_allclose(damping.numpy()[:, 0], [20.3, .3, .3], rtol=1e-6)

    # Both optimizers have their own forward path. Check that each updates the
    # integration derivative before executing the native step.
    from types import SimpleNamespace
    from comfree_warp.native_adjoint import fast_step, observation_adjoint
    monkeypatch.setattr(api.optimizer, "_build_initial", lambda *a, **k: None)
    monkeypatch.setattr(api.optimizer, "observation_forward", lambda *a, **k: None)
    monkeypatch.setattr(observation_adjoint, "forward", lambda *a, **k: None)
    calls = []
    def step(*args):
        np.testing.assert_allclose(damping.numpy()[:, 0], [20.3, .3, .3], rtol=1e-6)
        calls.append(True)
    monkeypatch.setattr(api.optimizer, "step_forward", step)
    monkeypatch.setattr(fast_step, "step", step)
    for cls in (api.PandaWindowOptimizer, api.PandaMppiWindowOptimizer):
        optimizer = cls.__new__(cls)
        optimizer.settings = SimpleNamespace(horizon=0)
        optimizer.compiled = SimpleNamespace(base=SimpleNamespace(dynamics=SimpleNamespace(model=model)))
        optimizer.servo_damping, optimizer.device, optimizer.control = damping, "cpu", None
        optimizer.dense_knot_map = ()
        optimizer.steps = (SimpleNamespace(inputs=SimpleNamespace(dynamics=inputs),
                                            workspace=SimpleNamespace(forward=None)),)
        optimizer.terminal_observation = SimpleNamespace(qpos=None, workspace=None)
        damping.zero_()
        optimizer._forward(allow_init=False)
    assert len(calls) == 2
