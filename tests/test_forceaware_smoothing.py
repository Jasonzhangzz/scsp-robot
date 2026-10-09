"""Numerical checks of the native FA regularizer on fourteen Panda angles."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from tests import test_dexforge_retarget as target_fixtures


def _regularizer():
    pytest.importorskip("warp")
    from planning.dexforge_forceaware import _classes

    try:
        api = _classes()
    except ImportError as error:
        pytest.skip(f"DexForge runtime unavailable: {error}")
    from forceaware.targets import ModelIndex

    # Put the free object first: hand qpos addresses are 7..20, but its velocity
    # addresses are 6..19. A qpos/qvel indexing mixup changes the state cost.
    joints = "".join(
        f'<body name="arm{i}"><joint name="joint{i}" axis="0 1 0"/>'
        '<geom type="sphere" size="0.01" mass="0.1"/></body>' for i in range(14))
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="obj"><freejoint/>'
        '<geom type="sphere" size="0.02" mass="0.2"/></body>'
        f'{joints}</worldbody></mujoco>')
    index = ModelIndex(np.arange(7, 21, dtype=np.int32), np.arange(14, dtype=np.int32), 0, 1)
    fa, _, _ = target_fixtures.TestPandaForceAwareTargets()._adapter(horizon=3, substeps=2)
    api.wp.init()
    reg = api.PandaRegularizer(model, index, fa.cfg, worlds=1, device="cpu")
    wp = api.wp
    control = SimpleNamespace(
        raw=wp.zeros((1, 56), dtype=float, device="cpu"),
        gradient=wp.zeros((1, 56), dtype=float, device="cpu"),
        previous_action=wp.zeros((1, 14), dtype=float, device="cpu"),
        actions=tuple(wp.array(np.full((1, 14), value, np.float32), device="cpu")
                      for value in (0.01, 0.02, 0.03)))
    steps = tuple(SimpleNamespace(
        recorded=SimpleNamespace(result=SimpleNamespace(
            qvel=wp.zeros((1, model.nv), dtype=float, device="cpu"),
            qacc=wp.zeros((1, model.nv), dtype=float, device="cpu"))),
        cotangent=SimpleNamespace(qvel=wp.zeros((1, model.nv), dtype=float, device="cpu"),
                                  qacc=wp.zeros((1, model.nv), dtype=float, device="cpu")))
        for _ in range(6))
    reg.prepare_window(np.zeros(14))
    return reg, control, steps, api


def test_panda_regularizer_uses_uniform_angular_scales_and_dof_mapping():
    reg, _, _, api = _regularizer()
    cfg, model = reg.cfg, reg.model
    expected = {
        "control_velocity_scale": cfg.smoothing.control_velocity_finger_scale,
        "control_velocity_weight": cfg.smoothing.control_velocity_weight,
        "control_acceleration_scale": cfg.smoothing.control_acceleration_finger_scale,
        "state_velocity_scale": api.mpc_loss.HAND_STATE_FINGER_VELOCITY_SCALE_RAD_S,
        "state_acceleration_scale": api.mpc_loss.HAND_STATE_FINGER_ACCELERATION_SCALE_RAD_S2,
        "state_component_ratio": 1.0,
    }
    for name, value in expected.items():
        np.testing.assert_array_equal(getattr(model, name).numpy(), np.full(14, value))
    np.testing.assert_array_equal(model.hand_qvel.numpy(), np.arange(6, 20))


def test_native_control_velocity_acceleration_cost_and_gradient():
    reg, control, steps, api = _regularizer()
    cfg = reg.cfg
    dt = cfg.simulator.time_grid.knot_dt
    assert dt == pytest.approx(0.02)
    reg.evaluate(control, steps, allow_init=False)
    velocity = 14 * 3 * cfg.smoothing.control_velocity_weight * (
        0.01 / dt / cfg.smoothing.control_velocity_finger_scale) ** 2
    # The ramp starts from rest. Its first acceleration is nonzero; subsequent
    # equal increments have zero acceleration.
    acceleration = 14 * cfg.smoothing.control_acceleration_weight * (
        0.01 / dt**2 / cfg.smoothing.control_acceleration_finger_scale) ** 2
    np.testing.assert_allclose(reg.runtime.components.numpy()[0], [0, velocity, acceleration, 0],
                               rtol=1.e-6, atol=1.e-7)
    reg.add_regularization_gradient(control, allow_init=False)
    from forceaware.numerics import LOSS_GRADIENT_SCALE

    analytic = control.gradient.numpy()[0] / float(LOSS_GRADIENT_SCALE)
    epsilon = 2.e-4
    for knot, joint in ((0, 0), (1, 7), (2, 13)):
        action = control.actions[knot]
        values = action.numpy()
        values[0, joint] += epsilon
        action.assign(values)
        positive = float(reg.evaluate(control, steps, allow_init=False).numpy()[0])
        values[0, joint] -= 2 * epsilon
        action.assign(values)
        negative = float(reg.evaluate(control, steps, allow_init=False).numpy()[0])
        values[0, joint] += epsilon
        action.assign(values)
        numerical = (positive - negative) / (2 * epsilon)
        expected = analytic[14 * (knot + 1) + joint]
        assert numerical == pytest.approx(expected, abs=0.005, rel=0.01)
    np.testing.assert_allclose(analytic[14::14], [0.5, 0.0, 50.0], atol=1.e-5)


def test_native_state_cost_and_cotangent_ignore_object_dofs():
    reg, control, steps, api = _regularizer()
    for step in steps:
        # Large object velocity/acceleration must not count as hand motion.
        velocity = np.full((1, 20), 0.4, np.float32)
        acceleration = np.full((1, 20), 10.0, np.float32)
        velocity[:, :6], acceleration[:, :6] = 100.0, 1000.0
        step.recorded.result.qvel.assign(velocity)
        step.recorded.result.qacc.assign(acceleration)
    reg.evaluate(control, steps, allow_init=False)
    weight = reg.cfg.smoothing.hand_state_weight
    velocity_scale = api.mpc_loss.HAND_STATE_FINGER_VELOCITY_SCALE_RAD_S
    acceleration_scale = api.mpc_loss.HAND_STATE_FINGER_ACCELERATION_SCALE_RAD_S2
    ratio = api.mpc_loss.HAND_ACCELERATION_RATIO
    expected = 6 * 14 * weight * ((0.4 / velocity_scale) ** 2
                                  + ratio * (10 / acceleration_scale) ** 2)
    assert reg.runtime.components.numpy()[0, 3] == pytest.approx(expected, rel=1.e-6)
    reg.add_state_cotangent(steps)
    from forceaware.numerics import LOSS_GRADIENT_SCALE

    for step in steps:
        velocity_gradient = step.cotangent.qvel.numpy()[0] / float(LOSS_GRADIENT_SCALE)
        acceleration_gradient = step.cotangent.qacc.numpy()[0] / float(LOSS_GRADIENT_SCALE)
        np.testing.assert_array_equal(velocity_gradient[:6], np.zeros(6))
        np.testing.assert_array_equal(acceleration_gradient[:6], np.zeros(6))
        np.testing.assert_allclose(velocity_gradient[6:], 2 * weight * 0.4 / velocity_scale**2)
        np.testing.assert_allclose(acceleration_gradient[6:], 2 * weight * ratio * 10 / acceleration_scale**2)


def test_native_history_preserves_constant_command_velocity_across_windows():
    reg, control, _, _ = _regularizer()
    reg.commit_window(np.zeros(14))
    reg.prepare_window(np.full(14, 0.01))
    control.previous_action.fill_(0.01)
    for action, value in zip(control.actions, (0.02, 0.03, 0.04)):
        action.fill_(value)
    reg.evaluate(control, [], allow_init=False)
    assert reg.runtime.components.numpy()[0, 1] == pytest.approx(10.5)
    assert reg.runtime.components.numpy()[0, 2] == pytest.approx(0.0, abs=1.e-10)
    np.testing.assert_array_equal(reg._previous_previous.numpy(), np.zeros((1, 14)))
    reg.commit_window(np.full(14, 0.01))
    reg.prepare_window(np.full(14, 0.02))
    np.testing.assert_allclose(reg._previous_previous.numpy(), 0.01)
    # Losing history turns an already moving command into a false restart.
    reg.prepare_window(np.full(14, 0.01), previous_previous_action=np.full(14, 0.01))
    reg.evaluate(control, [], allow_init=False)
    assert reg.runtime.components.numpy()[0, 2] == pytest.approx(0.035, rel=1.e-6)


def test_adam_retargets_from_both_shifted_trajectory_and_fresh_ca_with_same_bounds():
    from planning.dexforge_forceaware import _classes

    api = _classes()
    solver = object.__new__(api.PandaWindowOptimizer)
    solver.index = SimpleNamespace(action_dim=14)
    solver.settings = SimpleNamespace(worlds=2, init_noise=0., control_noise=0., initial_raw_limit=.5)
    # After CA changes its joint reference to .5, the shifted old trajectory
    # still commands .3. Keep it and try motion toward the new reference.
    warm = np.r_[np.zeros(14), np.full(28, -.2)].astype(np.float32)
    seeds = solver._multistart(warm, seed=0)
    wp = api.wp
    raw = wp.array(seeds, dtype=float, device='cpu')
    target = wp.array(np.full(14, .5, np.float32), device='cpu')
    lower = wp.array(np.full((2, 14), .15, np.float32), device='cpu')
    upper = wp.array(np.full((2, 14), .35, np.float32), device='cpu')
    for knot in range(2):
        wp.launch(api.project_offsets, dim=(2, 14),
                  inputs=[raw, target, lower, upper, knot, 14], device='cpu')
    commands = raw.numpy()[:, 14:].reshape(2, 2, 14) + .5
    np.testing.assert_allclose(commands[0], .3, atol=1e-7)
    np.testing.assert_allclose(commands[1], .35, atol=1e-7)
    np.testing.assert_array_equal(raw.numpy()[:, :14], 0.)
    assert np.all(commands >= .15) and np.all(commands <= .35)
