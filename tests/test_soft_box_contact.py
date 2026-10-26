"""Smooth Coulomb-box values and AD derivatives at activation boundaries."""
import casadi as cs
import numpy as np
import pytest

from models.soft_box_contact import (
    soft_box_contact_velocity_cs,
    soft_box_contact_velocity_numpy,
)


def _sigmoid(value):
    return np.exp(-np.logaddexp(0., -value))


def _single_contact(symbol, temperature):
    velocity = symbol.sym('velocity', 3)
    _, impulses = soft_box_contact_velocity_cs(
        velocity, [cs.DM.eye(3)], cs.DM.zeros(1), cs.DM.eye(3), .02, [.5],
        temperature=temperature, regularization=0., sweeps=1)
    return velocity, impulses[0]


@pytest.mark.parametrize('symbol', [cs.SX, cs.MX])
@pytest.mark.parametrize('temperature', [1e-4, .005])
def test_normal_curvature_at_zero_and_extreme_inputs(symbol, temperature):
    velocity, impulse = _single_contact(symbol, temperature)
    fn = cs.Function('normal_derivatives', [velocity], [
        impulse[0], cs.jacobian(impulse[0], velocity),
        cs.hessian(impulse[0], velocity)[0]])
    for normal_speed in [-1., -temperature, 0., temperature, 1.]:
        value, jacobian, hessian = fn([normal_speed, 0., 0.])
        probability = _sigmoid(-normal_speed / temperature)
        assert float(value) == pytest.approx(
            temperature * np.logaddexp(0., -normal_speed / temperature), abs=1e-12)
        assert float(jacobian[0]) == pytest.approx(-probability, abs=1e-12)
        assert float(hessian[0, 0]) == pytest.approx(
            probability * _sigmoid(normal_speed / temperature) / temperature,
            abs=1e-10, rel=1e-10)


@pytest.mark.parametrize('symbol', [cs.SX, cs.MX])
def test_tangential_derivatives_at_both_soft_clip_boundaries(symbol):
    temperature = .005
    velocity, impulse = _single_contact(symbol, temperature)
    fn = cs.Function('tangent_derivatives', [velocity], [
        impulse, cs.jacobian(impulse, velocity),
        cs.hessian(impulse[1], velocity)[0]])
    normal_speed = -.02
    cap = .5 * temperature * np.logaddexp(0., -normal_speed / temperature)
    for trial in [-cap, 0., cap]:
        projected, jacobian, hessian = fn([normal_speed, -trial, 0.])
        upper = _sigmoid((trial + cap) / temperature)
        lower = _sigmoid((trial - cap) / temperature)
        expected_gradient = -(upper - lower)
        expected_curvature = (
            upper * _sigmoid(-(trial + cap) / temperature)
            - lower * _sigmoid(-(trial - cap) / temperature)) / temperature
        assert float(jacobian[1, 1]) == pytest.approx(expected_gradient, abs=1e-12)
        assert float(hessian[1, 1]) == pytest.approx(expected_curvature, abs=1e-10)
        assert abs(float(projected[1])) <= cap + 1e-12


@pytest.mark.parametrize('symbol', [cs.SX, cs.MX])
def test_three_sweeps_preserve_numpy_values_and_force_bounds(symbol):
    rows = np.array([np.eye(3), [[.6, .8, 0.], [-.8, .6, 0.], [0., 0., 1.]]])
    inverse = np.array([[2., .2, 0.], [.2, 1., .1], [0., .1, .8]])
    gaps, friction, free = np.array([0., .001]), np.array([.5, .7]), np.array([-.02, .01, -.03])
    velocity = symbol.sym('free', 3)
    result, impulses = soft_box_contact_velocity_cs(
        velocity, [cs.DM(row) for row in rows], cs.DM(gaps), cs.DM(inverse), .02,
        friction, sweeps=3)
    fn = cs.Function('coupled_contacts', [velocity], [result, cs.horzcat(*impulses)])
    actual_velocity, actual_impulses = fn(free)
    expected_velocity, expected_impulses = soft_box_contact_velocity_numpy(
        free, rows, gaps, inverse, .02, friction, sweeps=3, return_impulses=True)
    actual_impulses = actual_impulses.full().T
    np.testing.assert_allclose(actual_velocity.full().ravel(), expected_velocity, atol=1e-12)
    np.testing.assert_allclose(actual_impulses, expected_impulses, atol=1e-12)
    assert np.all(actual_impulses[:, 0] >= 0.)
    assert np.all(np.abs(actual_impulses[:, 1:]) <=
                  friction[:, None] * actual_impulses[:, :1] + 1e-12)
