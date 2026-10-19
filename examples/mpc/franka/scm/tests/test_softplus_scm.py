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
from scm_backend import ScmEngine, softplus_impulse
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
