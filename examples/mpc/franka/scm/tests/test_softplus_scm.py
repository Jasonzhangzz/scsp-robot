"""Softplus activation, autodiff, and removal of the old corrected soft law."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_env import activate_dependencies
activate_dependencies()

import mujoco
import numpy as np
import pytest
import warp as wp
from scm_backend import ScmEngine, softplus_impulse, diagonal_contact, correct_amplitude
from benchmark_utils import Backend
from compare_prediction import make_model, fresh_data


@wp.kernel
def activate(values: wp.array(dtype=float), temperature: float, result: wp.array(dtype=float)):
    i = wp.tid()
    result[i] = softplus_impulse(values[i], temperature)


@pytest.mark.parametrize("temperature", [1e-5, 1e-4, 1e-3])
def test_softplus_is_finite_has_correct_gradient_and_uniform_hard_limit(temperature):
    wp.init()
    values = np.array([-1000., -1., -10*temperature, -temperature, 0.,
                       temperature, 10*temperature, 1., 1000.], dtype=np.float32)
    x = wp.array(values, dtype=wp.float32, device="cuda:0", requires_grad=True)
    y = wp.zeros(len(values), dtype=wp.float32, device="cuda:0", requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(activate, dim=len(values), inputs=[x, temperature], outputs=[y], device="cuda:0")
    tape.backward(grads={y: wp.ones_like(y)})
    actual = y.numpy()
    expected = temperature*np.logaddexp(0., values.astype(float)/temperature)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-10)
    expected_gradient = np.exp(-np.logaddexp(0., -values.astype(float)/temperature))
    np.testing.assert_allclose(x.grad.numpy(), expected_gradient, rtol=2e-5, atol=2e-7)
    assert np.isfinite(actual).all()
    assert actual[4] == pytest.approx(temperature*np.log(2), rel=2e-6)
    assert np.max(actual-np.maximum(values, 0)) <= temperature*np.log(2)+2e-10


def test_corrected_soft_scm_is_independent_of_legacy_spring_damper_parameters():
    model = make_model("box", .5, .002, .15)
    initial = fresh_data(model)
    initial.qpos[2] = .025-.00005
    initial.qvel[:] = [.2, .1, -.1, 0., 0., .1]
    mujoco.mj_forward(model, initial)
    predictions = []
    for tau, zeta in [(.008, .25), (.2, 4.)]:
        backend = Backend("scm_soft_corrected", model, initial, scm_response_time=tau,
                          scm_damping_ratio=zeta, nconmax=128, njmax=2048)
        data = fresh_data(model, initial.qpos, initial.qvel)
        backend.reset(data)
        backend.step(data)
        predictions.append(data.qvel.copy())
        assert backend.api.response_time == 0
    np.testing.assert_array_equal(*predictions)
    with pytest.raises(ValueError, match="removed"):
        ScmEngine(model, initial, response_time=.008, amplitude_correction=True)


@pytest.mark.parametrize("temperature", [-1., float("nan"), float("inf")])
def test_invalid_softplus_temperature_is_rejected(temperature):
    model = make_model("box", .5, .002, .15)
    with pytest.raises(ValueError, match="temperature"):
        ScmEngine(model, fresh_data(model), softplus_temperature=temperature)


@pytest.mark.parametrize("use_diagonal", [True, False])
@pytest.mark.parametrize("increment", [True, False])
def test_nonnegative_distance_in_both_stages_with_coupling(use_diagonal, increment):
    """Negative and zero gaps must give identical impulses at fixed geometry.

    Multiple worlds, mixed gaps, and full inertia exercise both kernel stages;
    using signed gap in only the amplitude numerator fails this regression.
    """
    wp.init()
    inverse = np.array([[2., .4], [.4, 1.]])
    if use_diagonal:
        inverse = np.diag(np.diag(inverse))
    J = np.array([[1., 0.], [1., 0.], [.5, 1.]])
    gaps = np.array([[-.001, -.002, .003], [0., 0., .003],
                     [-.01, -.02, -.03], [0., 0., 0.]])
    h, r, temperature = .01, .02, 1e-4
    incoming = np.array([.1, -.2])
    free = np.array([-1., -.2])
    W = J @ inverse @ J.T + r * np.eye(3)
    projected = free - incoming if increment else free
    g = np.tile(J @ projected, (4, 1)) + np.maximum(gaps, 0.) / h
    initial = temperature * np.logaddexp(0., -g / (np.diag(W) * temperature))
    A = np.einsum('bi,ij,bj->b', initial, W, initial)
    alpha_expected = np.clip(-np.sum(g * initial, axis=1) / A, 0., 1.)
    assert np.all((alpha_expected > 0) & (alpha_expected < 1))
    with wp.ScopedDevice("cuda:0"):
        force = wp.zeros((4, 3), dtype=wp.float32)
        reaction = wp.zeros((4, 2), dtype=wp.float32)
        alpha = wp.zeros(4, dtype=wp.float32)
        common = [2, wp.array([h], dtype=wp.float32), wp.array([3]*4, dtype=wp.int32),
                  wp.array(np.tile(J, (4, 1, 1)), dtype=wp.float32),
                  wp.array(gaps, dtype=wp.float32),
                  wp.array(np.tile(free, (4, 1)), dtype=wp.float32),
                  wp.array(np.tile(incoming, (4, 1)), dtype=wp.float32),
                  wp.array(np.diag(inverse), dtype=wp.float32),
                  wp.array(inverse, dtype=wp.float32), use_diagonal, increment, r]
        wp.launch(diagonal_contact, dim=(4, 3),
                  inputs=common + [0., 1., temperature, True], outputs=[force, reaction])
        np.testing.assert_allclose(h * force.numpy(), initial, rtol=3e-6, atol=1e-9)
        wp.launch(correct_amplitude, dim=4, inputs=common + [True],
                  outputs=[force, reaction, alpha])
        impulses = h * force.numpy()
        np.testing.assert_allclose(alpha.numpy(), alpha_expected, rtol=3e-6)
        np.testing.assert_allclose(impulses, alpha_expected[:, None]*initial, rtol=3e-6, atol=1e-9)
        np.testing.assert_array_equal(impulses[0], impulses[1])
        np.testing.assert_array_equal(impulses[2], impulses[3])
        np.testing.assert_allclose(h * reaction.numpy(), impulses @ J, rtol=3e-6)
    before = .5*A + np.sum(g*initial, axis=1)
    after = .5*np.einsum('bi,ij,bj->b', impulses, W, impulses) + np.sum(g*impulses, axis=1)
    assert np.all(after <= np.minimum(before, 0.) + 1e-8)


def test_penetrating_separating_point_has_no_depth_recovery_impulse():
    """A 1 mm penetration cannot create an upward recovery velocity."""
    model = make_model("sphere", .5, .002, .15)
    initial = fresh_data(model)
    initial.qpos[2] = .04 - .001
    initial.qvel[2] = .1
    mujoco.mj_forward(model, initial)
    backend = Backend("scm_soft_corrected", model, initial, nconmax=128, njmax=2048)
    assert backend.api.nonnegative_gap
    data = fresh_data(model, initial.qpos, initial.qvel)
    backend.step(data)
    assert int(backend.wdata.nefc.numpy()[0]) > 0
    np.testing.assert_allclose(data.qvel, backend.wdata.qvel_smooth_pred.numpy()[0], atol=1e-9)
