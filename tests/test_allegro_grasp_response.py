from types import SimpleNamespace

import numpy as np

from examples.mpc.franka.trigrasp.trigrasp import AllegroOnlineContactModel
from models.allegro_grasp_response import grasp_operators
from models.support_geometry import support_jacobians


def _loaded_model():
    h, nq, n = .025, 22, 3
    mass = np.diag([.03]*3+[.0001]*3)
    param = SimpleNamespace(h_=h, n_cmd_=nq, n_qpos_=79, n_qvel_=28, num_contacts=n,
        max_ncon_=n, obj_inertia_=mass/h**2, obj_mass_=.03, mu_object_=.8,
        gravity_=np.array([0., 0., -9.81, 0., 0., 0.]),
        robot_stiff_=np.eye(nq)*1000., robot_damping_=np.full(nq, 40.))
    model = AllegroOnlineContactModel(param)
    points = np.array([[.04, 0., 0.], [-.04, .03, 0.], [-.04, -.03, 0.]])
    normals = np.array([[-1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
    blocks = []
    for point, normal in zip(points, normals):
        block, _ = support_jacobians(np.zeros(3), np.eye(3), point, 28, .8, normal)
        block[:, 6:12] = -block[:, :6]
        blocks.append(block)
    jac = np.vstack(blocks)
    state = np.zeros(79); state[3] = 1.
    # Position spring balances the object's weight on the robot as well.
    state[-nq+2] = .03*9.81/1000
    model.reference = state[:29].copy()
    (model.grasp_velocity_map, model.grasp_robot_jac, model.grasp_inverse,
     model.grasp_mobility, model.grasp_internal) = grasp_operators(jac, model.robot_inverse, mass, n)
    forces = np.array([[-.6, 0., .14715], [.3, 0., .073575], [.3, 0., .073575]])
    model.grasp_reference_force = forces.reshape(-1)
    model.grasp_normal_limits = np.array([.6, .3, .3])
    model.grasp_active = True
    return model, state, np.full(12, -.001), jac, points, normals


def test_verified_preloaded_grasp_is_stationary_under_gravity():
    model, state, phi, jac, _, _ = _loaded_model()
    result = np.asarray(model.prediction_fn(state, np.zeros(22), phi, jac, model.runtime_parameters(1))).ravel()
    np.testing.assert_allclose(result[:3], state[:3], atol=1e-11)
    np.testing.assert_allclose(result[29:57], 0., atol=1e-10)
    model.grasp_active = False
    unsupported = np.asarray(model.prediction_fn(state, np.zeros(22), phi, jac, model.runtime_parameters(1))).ravel()
    assert unsupported[2] < state[2]-.001


def test_upward_wrench_obeys_newton_balance_and_measured_cones():
    model, state, phi, jac, points, normals = _loaded_model()
    u = np.zeros(22); u[2] = .001
    mp = model.runtime_parameters(1)
    next_state = np.asarray(model.prediction_fn(state, u, phi, jac, mp)).ravel()
    force = np.asarray(model.contact_force_fn(state, u, phi, jac, mp)).reshape(3, 3)
    assert next_state[2] > state[2]
    wrench = np.r_[force.sum(0), np.cross(points, force).sum(0)]
    expected = .025*model.object_inverse@(model.external_wrench+wrench)
    np.testing.assert_allclose(next_state[29:35], expected, atol=1e-10)
    normal = np.sum(force*normals, axis=1)
    assert np.all(normal >= 0.)
    assert np.all(normal <= model.grasp_normal_limits+1e-10)
    assert np.all(np.linalg.norm(force-normal[:, None]*normals, axis=1) <= .8*normal+1e-10)


def test_predicted_separation_cannot_keep_a_rigid_attachment_force():
    model, state, phi, jac, _, _ = _loaded_model()
    separated = state.copy(); separated[7] += .005
    force = np.asarray(model.contact_force_fn(separated, np.zeros(22), phi, jac, model.runtime_parameters(1))).reshape(3, 3)
    np.testing.assert_array_equal(force[0], 0.)


def test_saturated_base_does_not_cancel_unrelated_finger_motion():
    model, state, phi, jac, _, _ = _loaded_model()
    model.command_limits = np.r_[np.full(6, .001), np.full(16, .0125)]
    action = np.zeros(22)
    action[2], action[6] = .001, .005
    np.testing.assert_allclose(model.effective_command(action), action)
    predicted = np.asarray(model.prediction_fn(state, action, phi, jac, model.runtime_parameters(1))).ravel()
    np.testing.assert_allclose(predicted[-22:]-state[-22:], action, atol=1e-12)
