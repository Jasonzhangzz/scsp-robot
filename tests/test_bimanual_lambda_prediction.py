import numpy as np

from planning.bimanual_lambda_prediction import BimanualLambdaPrediction


def predictor(goal=None):
    solver = BimanualLambdaPrediction(horizon=8)
    pose = np.array([0., 0., .3, 1., 0., 0., 0.])
    goal = pose+[0., 0., .05, 0., 0., 0., 0.] if goal is None else goal
    solver.set_observation(pose, goal, np.zeros(6), np.diag([.01]*3+[1e-5]*3),
        [0., 0., -.0981, 0., 0., 0.], friction=.7, support_friction=.7)
    return solver


NORMALS = np.array([[0., 1., 0.], [0., -1., 0.]])
MATCHED = np.array([[0., -.03, 0.], [0., .03, 0.]])
OFFSET = np.array([[-.025, -.03, 0.], [.025, .03, 0.]])


def test_offset_pair_is_penalized_by_predicted_rotation_without_geometry_rules():
    solver = predictor()
    displaced, matched = solver.evaluate(OFFSET, NORMALS), solver.evaluate(MATCHED, NORMALS)
    assert np.isfinite(displaced['cost']) and np.isfinite(matched['cost'])
    assert displaced['next_orientation_cost'] > matched['next_orientation_cost']+1e-10
    assert abs(displaced['rollout'][-1, 6]) > 1e-4
    assert displaced['cost'] > matched['cost']


def test_optimizer_reduces_actual_coupled_prediction_cost():
    solver = predictor()
    initial, optimized = solver.evaluate(MATCHED, NORMALS), solver.evaluate(MATCHED, NORMALS, optimize=True)
    assert optimized['cost'] <= initial['cost']
    assert np.max(np.abs(optimized['command_delta_world'])) <= solver.bound+1e-12
    assert not np.allclose(optimized['command_delta_world'], initial['command_delta_world'])


def test_candidate_neutral_at_com_is_scored_not_rejected_by_a_height_rule():
    solver = predictor()
    scores = solver.rank(np.array([MATCHED, OFFSET]), np.array([NORMALS]*2), [True, True], refinements=2)
    assert np.isfinite(scores).all()
    assert scores[0] < scores[1]
    assert solver.results[0]['rollout'].shape == (8, 25)


def test_forecast_force_matches_mpc_contact_model_and_has_finite_response():
    solver = predictor()
    geometry = solver._geometry(MATCHED, NORMALS)
    result = solver.evaluate(MATCHED, NORMALS)
    state, phi, jac, parameters, inward = geometry
    u = solver.pressure/solver.stiffness*inward.ravel()
    expected = np.asarray(solver.model.prediction_fn(state, u, phi, jac, parameters)).ravel()
    force = np.asarray(solver.model.contact_force_fn(state, u, phi, jac, parameters)).ravel().reshape(2, 3)
    np.testing.assert_allclose(result['x_plus'], expected[:7])
    np.testing.assert_allclose(result['force_vectors_local'], force@solver.rotation)
    assert np.max(np.linalg.norm(force, axis=1)) < solver.pressure  # No instantaneous force realization.


def test_rotation_objective_depends_on_requested_pose():
    solver = predictor()
    first = solver.evaluate(OFFSET, NORMALS)
    solver.target = first['x_plus'].copy()
    same_motion = solver.evaluate(OFFSET, NORMALS)
    assert same_motion['next_orientation_cost'] < 1e-10
    assert same_motion['next_position_cost'] < 1e-10
    np.testing.assert_allclose(first['x_plus'], same_motion['x_plus'])


def test_nearby_prediction_uses_actual_separation_and_pad_velocity():
    solver = predictor()
    touch = solver.evaluate(MATCHED, NORMALS)
    solver.tips_world = MATCHED+solver.pose[:3]-.025*NORMALS
    solver.tip_velocity[:] = [0., .02, 0., 0., -.01, 0.]
    approaching = solver.evaluate(MATCHED, NORMALS)
    geometry = solver._geometry(MATCHED, NORMALS)
    np.testing.assert_allclose(geometry[1][:8], .015)
    np.testing.assert_array_equal(approaching['prediction_initial_state'][19:], solver.tip_velocity)
    np.testing.assert_allclose(approaching['force_vectors_local'], 0., atol=1e-10)
    assert np.linalg.norm(touch['force_vectors_local']) > 0


def test_cached_analytic_jacobian_matches_finite_difference():
    solver = predictor()
    geometry = solver._geometry(OFFSET, NORMALS)
    u = .002*geometry[-1].ravel()
    analytic = np.asarray(solver.residual_jac_fn(u, *geometry[:4], solver.target, solver.pressure,
                                                (geometry[-1]@solver.rotation).T))
    step = 1e-7
    numeric = np.column_stack([(solver._rollout(u+step*np.eye(6)[i], geometry)-
                              solver._rollout(u-step*np.eye(6)[i], geometry))/(2*step) for i in range(6)])
    np.testing.assert_allclose(analytic, numeric, atol=1e-5, rtol=1e-4)


def test_pressure_reference_is_a_load_floor_not_osc_stiffness_times_depth():
    solver = predictor()
    pressure = solver.pressure
    solver.set_observation(solver.pose, solver.target, solver.velocity, solver.inertia, solver.external,
        friction=.7, support_friction=.7, squeeze_depth=.02)
    assert solver.pressure == pressure
    assert pressure < .1
    result = solver.evaluate(MATCHED, NORMALS)
    assert result['cost_components']['pressure'] < .2*result['cost']


def test_observed_preload_balances_normals_and_sticks_against_gravity():
    solver = BimanualLambdaPrediction(tangent_impedance=1/(1+.1/(.9*10)))
    base = predictor()
    solver.set_observation(base.pose, base.target, base.velocity, base.inertia, base.external,
        friction=.7, support_friction=.7, measured_normal=[.7, .7])
    state, phi, jac, parameters, _ = solver._geometry(MATCHED, NORMALS)
    forces = np.asarray(solver.model.contact_force_fn(state, np.zeros(6), phi, jac, parameters)).reshape(2, 3)
    np.testing.assert_allclose(np.sum(forces*NORMALS, axis=1), [.7, .7], atol=1e-6)
    np.testing.assert_allclose(forces.sum(0), -base.external[:3], atol=.002)
    predicted = np.asarray(solver.model.prediction_fn(state, np.zeros(6), phi, jac, parameters)).ravel()
    assert abs(predicted[15]) < .002
    # A saved preload does not supply pressure or friction after separation.
    separated = state.copy()
    separated[7:13] -= (.02*NORMALS).ravel()
    free_forces = np.asarray(solver.model.contact_force_fn(separated, np.zeros(6), phi, jac, parameters))
    np.testing.assert_allclose(free_forces, 0., atol=1e-12)


def test_torsional_contact_counters_axial_rotation_within_shared_load_budget():
    solver = BimanualLambdaPrediction(tangent_impedance=.989, spin_friction=.08, roll_friction=.01)
    base = predictor()
    velocity = np.array([0., 0., 0., 0., 2., 0.])
    solver.set_observation(base.pose, base.target, velocity, base.inertia, base.external,
        friction=.7, support_friction=.7, measured_normal=[.7, .7])
    state, phi, jac, parameters, _ = solver._geometry(MATCHED, NORMALS)
    command = np.zeros(6)
    next_state = np.asarray(solver.model.prediction_fn(state, command, phi, jac, parameters)).ravel()
    force = np.asarray(solver.model.contact_force_fn(state, command, phi, jac, parameters)).reshape(2, 3)
    moment = np.asarray(solver.model.contact_moment_fn(state, command, phi, jac, parameters)).reshape(2, 3)
    assert abs(next_state[17]) < .01  # spin about the unactuated point-contact axis
    fn = np.sum(force*NORMALS, axis=1)
    for side in range(2):
        ft = force[side]-fn[side]*NORMALS[side]
        spin = abs(moment[side]@NORMALS[side])/.08
        roll = (abs(moment[side, 0])+abs(moment[side, 2]))/.01
        assert (abs(ft[0])+abs(ft[2]))/.7+spin+roll <= fn[side]+1e-8
    total = base.external+np.r_[force.sum(0), np.cross(MATCHED, force).sum(0)+moment.sum(0)]
    expected_velocity = velocity+solver.param.h_*np.linalg.solve(base.inertia, total)
    np.testing.assert_allclose(next_state[13:19], expected_velocity, atol=1e-7)
    separated = state.copy()
    separated[7:13] -= (.02*NORMALS).ravel()
    np.testing.assert_allclose(solver.model.contact_moment_fn(separated, command, phi, jac, parameters), 0., atol=1e-12)


def test_coupled_preload_observation_matches_asymmetric_real_pressures():
    solver = BimanualLambdaPrediction(tangent_impedance=.989, spin_friction=.08, roll_friction=.01)
    base = predictor()
    solver.set_observation(base.pose, base.target, base.velocity, base.inertia, base.external,
        friction=.7, support_friction=.7, measured_normal=[.7, .4])
    state, phi, jac, parameters, _ = solver._geometry(OFFSET, NORMALS)
    force = np.asarray(solver.model.contact_force_fn(state, np.zeros(6), phi, jac, parameters)).reshape(2, 3)
    np.testing.assert_allclose(np.sum(force*NORMALS, axis=1), [.7, .4], atol=1e-5)


def test_rolling_friction_remains_when_sliding_friction_is_zero():
    solver = BimanualLambdaPrediction(tangent_impedance=.989, spin_friction=.08, roll_friction=.01)
    base = predictor()
    velocity = np.array([0., 0., 0., 1., 0., 0.])
    solver.set_observation(base.pose, base.target, velocity, base.inertia, base.external,
        friction=0., support_friction=0., measured_normal=[.7, .7])
    state, phi, jac, parameters, _ = solver._geometry(MATCHED, NORMALS)
    next_state = np.asarray(solver.model.prediction_fn(state, np.zeros(6), phi, jac, parameters)).ravel()
    assert abs(next_state[16]) < .01
    force = np.asarray(solver.model.contact_force_fn(state, np.zeros(6), phi, jac, parameters)).reshape(2, 3)
    np.testing.assert_allclose(force[:, 2], 0., atol=1e-12)
