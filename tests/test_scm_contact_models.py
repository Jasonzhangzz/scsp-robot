import numpy as np

from planning.scm_contact_models import (
    LCPContactModel,
    appendix_accuracy,
    contact_jacobian,
    motion_accuracy,
    pyramid_matrix,
    surrogate_response,
)


def test_contact_jacobian_body_frame_convention():
    jac = contact_jacobian([1.0, 2.0, 3.0])
    np.testing.assert_allclose(jac[:, :3], np.eye(3))
    np.testing.assert_allclose(jac[:, 3:], [[0, 3, -2], [-3, 0, 1], [2, -1, 0]])


def test_diagonal_delassus_scm_matches_lcp():
    J = np.zeros((4, 6))
    J[:, :4] = np.eye(4)
    q_inv = np.eye(6)
    b = np.array([-1.0, -0.25, 0.5, -0.75, 0.0, 0.0])
    scm = surrogate_response(q_inv, J, b, regularization=1e-8)
    lcp = LCPContactModel(regularization=1e-8)
    lam, v_plus = lcp.respond(scm["v_free"], J, q_inv)
    np.testing.assert_allclose(scm["lambda_env"], lam, atol=1e-7)
    np.testing.assert_allclose(scm["v_plus"], v_plus, atol=1e-7)
    assert lcp.last["converged"]
    assert lcp.last["residual"] < 1e-7


def test_coupling_scale_changes_dense_lcp_response():
    J = np.zeros((4, 6))
    J[0, 0] = 1.0
    J[1, 0] = 1.0
    J[1, 1] = 1.0
    J[2, 2] = 1.0
    J[3, 3] = 1.0
    q_inv = np.eye(6)
    b = np.array([-1.0, -0.5, 0.0, 0.0, 0.0, 0.0])
    scm = surrogate_response(q_inv, J, b, regularization=1e-6)
    lcp = LCPContactModel(regularization=1e-6)
    _, v0 = lcp.respond(scm["v_free"], J, q_inv, coupling_scale=0.0)
    _, v1 = lcp.respond(scm["v_free"], J, q_inv, coupling_scale=1.0)
    assert np.linalg.norm(v1 - v0) > 1e-6


def test_motion_accuracy_handles_zero_velocity():
    stats = motion_accuracy(np.eye(6), np.zeros(6), np.zeros(6))
    assert stats["direction_error"] is None
    assert stats["cos_theta"] is None
    assert stats["magnitude_error"] == 0.0


def test_appendix_metrics_are_finite_for_active_contact():
    J = np.zeros((4, 6))
    J[:, :4] = np.eye(4)
    q_inv = np.eye(6)
    b = np.array([-1.0, -0.25, 0.5, -0.75, 0.0, 0.0])
    scm = surrogate_response(q_inv, J, b, regularization=1e-8)
    stats = appendix_accuracy(
        np.eye(6), q_inv, J, scm["v_plus"], scm["lambda_env"],
        scm["v_plus"], 0.02, lam_ref=scm["lambda_env"])
    for key in ("magnitude_error", "lambda_gap_D", "gamma_env", "coupling_norm"):
        assert np.isfinite(stats[key])


def test_pyramid_matrix_reconstructs_axis_force():
    A = pyramid_matrix(0.5)
    lam = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(A @ lam, [1.0, 0.5, 0.0])
