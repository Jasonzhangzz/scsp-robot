"""Upstream ContactSDF dynamics and checkpoint configuration."""
import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from examples.mpc.franka.scm.vendor.contactsdf.ball_lse_contact_detection import Contact
from examples.mpc.franka.runtime.params import add_lambda_predictor_args
from models.contactsdf import (
    CUBE_THETA, FOAM_THETA, STICK_THETA, ContactSDFModel, apply_contactsdf_args,
    contactsdf_ipopt_options, load_lse_dynamic, theta_for_object,
)


def contactsdf_factor_rows(normal, relative):
    detector = object.__new__(Contact)
    detector.param_ = SimpleNamespace(con_vec_=np.ones(3))
    frame = detector.calcContactFrame(np.asarray(normal, float))
    rows = np.c_[frame, -frame[:, -2:]].T @ relative
    return np.repeat(rows[0][None, :], 4, axis=0), rows[1:]


def _param():
    return SimpleNamespace(
        n_qpos_=10, n_qvel_=9, n_cmd_=3, max_ncon_=2, h_=0.1,
        gravity_=np.array([0., 0., -9.8, 0., 0., 0.]),
        contactsdf_theta_=CUBE_THETA, model_params=1.)


def test_factor_rows_match_upstream_layout():
    normal = np.array([0., 0., 1.])
    relative = np.eye(3, 9)
    jac_n, jac_f = contactsdf_factor_rows(normal, relative)
    tangent = np.cross(normal, [1., 1., 1.])
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    bitangent /= np.linalg.norm(bitangent)
    frame = np.stack((normal, tangent, bitangent), axis=1)
    rows = np.concatenate((frame, -frame[:, 1:]), axis=1).T @ relative
    np.testing.assert_allclose(jac_n, np.repeat(rows[0][None, :], 4, axis=0))
    np.testing.assert_allclose(jac_f, rows[1:])


def test_step_is_the_upstream_function_and_does_not_normalize_quaternions():
    param = _param()
    model = ContactSDFModel(param)
    direct = load_lse_dynamic()(SimpleNamespace(
        n_qvel_=9, n_qpos_=10, n_cmd_=3, max_ncon_=2, h_=0.1,
        gravity_=param.gravity_))
    q = np.array([0., 0., 0.03, 1., 0., 0., 0., 0.05, 0., 0.])
    cmd = np.array([0.01, -0.01, 0.002])
    phi = np.ones(8)
    relative = np.zeros((3, 9))
    relative[:, :3] = np.eye(3)
    relative[:, 6:] = -np.eye(3)
    one_n, one_f = contactsdf_factor_rows([0., 0., 1.], relative)
    jac_n, jac_f = np.zeros((8, 9)), np.zeros((8, 9))
    jac_n[:4], jac_f[:4] = one_n, one_f
    # One real contact and upstream's phi=1 padding.
    phi[:4] = -0.001
    wrapped = np.array(model.step(q, cmd, phi, jac_n, jac_f)).ravel()
    raw = np.array(direct.dyn_once_fn(q, cmd, phi, jac_n, jac_f, CUBE_THETA)).ravel()
    np.testing.assert_allclose(wrapped, raw, atol=0.)
    velocity = np.zeros(9)
    velocity[3] = 1.
    integrated = np.array(model.dynamics.cs_qposInteg_(q, velocity)).ravel()
    assert abs(np.linalg.norm(integrated[3:7]) - 1.) > 1e-6


def test_published_theta_and_ipopt_options():
    foam, source = theta_for_object('foam_brick')
    np.testing.assert_allclose(foam, FOAM_THETA)
    assert source == 'foambrick'
    np.testing.assert_allclose(theta_for_object('stick')[0], STICK_THETA)
    np.testing.assert_allclose(theta_for_object('bunny')[0], CUBE_THETA)
    parser = add_lambda_predictor_args(argparse.ArgumentParser())
    args = parser.parse_args(['--contactsdf'])
    args.obj = 'foam_brick'
    apply_contactsdf_args(args, argv=['--contactsdf'])
    assert args.contact_model == 'contactsdf'
    assert args.solver_backend == 'ipopt'
    assert args.contactsdf_theta_source == 'foambrick'
    np.testing.assert_allclose(args.contactsdf_theta, FOAM_THETA)
    assert contactsdf_ipopt_options(100) == {
        'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0, 'ipopt.max_iter': 100}
    mixed = parser.parse_args(['--contactsdf', '--contact-model', 'comfree'])
    with pytest.raises(ValueError, match='cannot be combined'):
        apply_contactsdf_args(mixed, argv=['--contactsdf'])


