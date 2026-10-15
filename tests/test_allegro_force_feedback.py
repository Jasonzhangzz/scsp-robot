from types import SimpleNamespace

import numpy as np

from planning.allegro_mpc_costs import calibrated_pad_forces, contact_cost_parameters_at_state, measured_pad_targets


def test_output_correction_matches_measurement_and_tracks_increment():
    baseline = np.array([[.1, 0., .02]])
    measured = np.array([[.3, 0., .1]])
    normal = np.array([[1., 0., 0.]])
    np.testing.assert_allclose(calibrated_pad_forces(baseline, baseline, measured, [3.],
        normal, [-.001], 1.), measured)
    raw = np.array([[.14, 0., .04]])
    np.testing.assert_allclose(calibrated_pad_forces(raw, baseline, measured, [3.],
        normal, [-.001], 1.), [[.42, 0., .16]])


def test_separation_removes_measured_preload_and_cone_is_respected():
    normal = np.array([[1., 0., 0.]])
    zero = np.zeros((1, 3))
    measured = np.array([[.3, .2, .2]])
    np.testing.assert_array_equal(calibrated_pad_forces(zero, zero, measured, [3.],
        normal, [.005], .5), zero)
    force = calibrated_pad_forces(zero, zero, measured, [3.], normal, [-.001], .5)
    assert np.linalg.norm(force[0, 1:]) <= .5*force[0, 0]+1e-12


def test_backtracking_uses_exact_force_after_a_nonlinear_mode_change():
    physics = SimpleNamespace(contact_force_fn=lambda x,u,*rest: np.array([u[0]**2, 0., 0.]))
    # Unrelated parameters, seed state/action, value, Jacobian and sentinel.
    cp = np.array([9., 0., 0., 0., 0., 0., 0., 0., 0., 7.])
    actual = contact_cost_parameters_at_state(cp, [.2], [.5], physics, None, None, None, 1)
    np.testing.assert_allclose(actual[1:6], [.2, .5, .25, 0., 0.])
    np.testing.assert_array_equal(actual[[0, -1]], cp[[0, -1]])
    np.testing.assert_array_equal(cp[1:6], 0.)


def test_midpoint_pad_target_controls_preload_without_pinning_initial_patch():
    initial = np.array([[.1, .2, .3], [.4, .5, .6]])
    points = np.array([[.0005, 0., 0.], [.4, .5, .6]])
    normals = np.array([[1., 0., 0.], [1., 0., 0.]])
    targets = measured_pad_targets(initial, points, normals, [-.001, .01], [.5, 0.],
        np.array([[.25, 5., 0.], [.2, 0., 0.]]), radius=.012, max_depth=.005, minimum_normal=.2)
    np.testing.assert_allclose(targets[0], [-.0115, 0., 0.], atol=1e-12)
    np.testing.assert_array_equal(targets[1], initial[1])


def test_made_pad_cost_detects_slip_without_imposing_ik_indentation():
    from examples.mpc.franka.trigrasp.trigrasp import AllegroOnlineContactModel
    from models.support_geometry import support_jacobians
    from planning.allegro_mpc_costs import build_allegro_contact_costs
    p = SimpleNamespace(h_=.025, n_cmd_=22, n_qpos_=79, n_qvel_=28, num_contacts=3,
        max_ncon_=3, obj_inertia_=np.eye(6), obj_mass_=.03, mu_object_=.7,
        gravity_=np.array([0., 0., -9.81, 0., 0., 0.]), robot_stiff_=np.eye(22)*1000,
        robot_damping_=np.full(22, 40.), force_tracking_weight=0., pressure_sync_weight=0.,
        pressure_control=True, tip_weight=1., lateral_weight=0., object_weight=0.,
        quat_weight=0., horizontal_recovery_weight=0., action_weights=np.zeros(22),
        posture_weights=np.zeros(22), terminal_weight=1.)
    p.online_model = AllegroOnlineContactModel(p)
    cost, _ = build_allegro_contact_costs(p)
    pose = np.array([0., 0., 0., 1., 0., 0., 0.])
    tip_jac = np.tile(np.c_[np.eye(3), np.zeros((3, 19))], (3, 1))
    block, _ = support_jacobians(np.zeros(3), np.eye(3), np.zeros(3), 28, .7, [1., 0., 0.])
    cp = np.r_[pose, np.zeros(22), 0., np.zeros(4), np.zeros(22), np.zeros(9),
        tip_jac.reshape(-1, order='F'), np.zeros(9), 0., np.zeros(9), np.full(12, -.001),
        np.tile(block, (3, 1)).reshape(-1, order='F'), p.online_model.runtime_parameters(1.)]
    state = np.zeros(79); state[3] = 1.
    normal = state.copy(); normal[7] += .005
    slip = state.copy(); slip[8] += .005
    assert float(cost(normal, np.zeros(22), cp)) < 1e-12
    assert float(cost(slip, np.zeros(22), cp)) > 7e-5
    # A missing pad must still approach in the normal direction.
    cp[p.cost_param_head_size+9:p.cost_param_head_size+21] = .005
    assert float(cost(normal, np.zeros(22), cp)) > 7e-5
