"""Solver selection, force feasibility, reproducibility and SCM graph parity."""
from types import SimpleNamespace

import casadi as cs
import numpy as np
import pytest

from planning.contact_force_sampling import ForcePopulationSearch, feasible_forces
from planning.mlqp_point import LambdaContactControlOptimizer
from planning.object_pose_cost import object_cost_np
from planning.runtime_compat import resolve_solver_backend


@pytest.mark.parametrize('pyramid', [False, True])
@pytest.mark.parametrize('mu', [0., .3, 1.2, 3.])
def test_population_mapping_obeys_native_force_constraints(pyramid, mu):
    limit = 3.
    cap = limit / np.sqrt(1 + (1 if pyramid else 2) * mu**2)
    samples = np.random.default_rng(2).uniform(-1., 2., (1000, 3))
    force = feasible_forces(samples, cap, mu, pyramid)
    assert np.all(force[:, 0] >= 1e-6)
    assert np.all(force[:, 0] <= cap)
    tangents = (np.abs(force[:, 1:]).sum(axis=1) if pyramid
                else np.abs(force[:, 1:]).max(axis=1))
    assert np.all(tangents <= mu * force[:, 0] + 1e-12)
    assert np.all(np.linalg.norm(force, axis=1) <= limit + 1e-12)


@pytest.mark.parametrize('backend', ['cma-es', 'cem'])
def test_population_search_improves_known_objective_and_repeats(backend, monkeypatch):
    f = cs.SX.sym('force', 3)
    optimum = np.array([.7, .15, -.1])
    fn = cs.Function('quadratic_force', [f], [cs.vertcat(f, 1., 0., 0., 0.),
                     cs.sumsqr(f - optimum)], ['lam_arm'], ['x_plus_opt', 'cost'])
    # Real evaluator accepts extra model parameters; the simple quadratic
    # exposes those names too so this exercises the same batched call path.
    theta, limit = cs.SX.sym('theta', 6), cs.SX.sym('limit')
    fn = cs.Function('quadratic_parameters', [f, theta, limit], fn(f),
                     ['lam_arm', 'scm_theta', 'force_limit'], ['x_plus_opt', 'cost'])
    opt = SimpleNamespace(force_objective_fn=fn, contact_model='soft-scm',
                          base_mu_arm_obj=1., friction_cone='pyramid',
                          _friction_force_cap_factor=1.)
    kwargs = dict(scm_theta=np.zeros(6), force_limit=3.)
    search = ForcePopulationSearch(backend, 24, 25, 10)
    calls = []
    if backend == 'cem':
        import torch
        torch_state = torch.random.get_rng_state().clone()
        ask = search.cem_ask
        def library_ask(*args, **kwargs):
            calls.append(True)
            return ask(*args, **kwargs)
        monkeypatch.setattr(search, 'cem_ask', library_ask)
    else:
        constructor = search.cma.CMAEvolutionStrategy
        def library_cma(*args, **kwargs):
            calls.append(True)
            return constructor(*args, **kwargs)
        monkeypatch.setattr(search.cma, 'CMAEvolutionStrategy', library_cma)
    state = np.random.get_state()
    first = search.solve(opt, kwargs)
    assert calls, 'The optimizer must call the installed library implementation'
    second = ForcePopulationSearch(backend, 24, 25, 10).solve(opt, kwargs)
    np.testing.assert_array_equal(first['lam_arm_opt'], second['lam_arm_opt'])
    np.testing.assert_allclose(first['lam_arm_opt'], optimum, atol=.04)
    assert first['cost'] < .002
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    if backend == 'cem':
        assert torch.equal(torch_state, torch.random.get_rng_state())


@pytest.mark.parametrize('backend', ['ipopt', 'cma-es', 'cem'])
def test_scm_backend_runs_in_real_mujoco_without_acados(backend, monkeypatch):
    from examples.mpc.fingertips.test.test_scm_mujoco import build_parser, _run_rollout
    monkeypatch.setenv('MUJOCO_HEADLESS', '1')
    monkeypatch.setattr(LambdaContactControlOptimizer, '_build_acados_contact_solver',
                        lambda *a: pytest.fail('Explicit alternative must not build acados'))
    args = build_parser().parse_args(['--solver', backend, '--population-size', '8',
                                     '--population-iterations', '2'])
    assert resolve_solver_backend(args.solver) == backend
    rows = _run_rollout(args, 'surrogate', 'foam_brick', seed=0, steps=2)
    assert len(rows) == 2
    assert not np.allclose(rows[-1]['object_pose'], rows[0]['initial_object_pose'])
    for row in rows:
        assert row['solver_requested'] == row['solver_backend'] == backend
        assert not row['solver_fallback'] and not row['mujoco_unstable']
        assert row['optimizer_model'] == 'coupled-quasi-dynamic-soft-scm'
        force = np.asarray(row['optimal_force_contact'])
        assert abs(force[1]) + abs(force[2]) <= 1.2 * force[0] + 1e-7
        assert np.linalg.norm(force) <= 3.000001


def test_population_evaluator_matches_shared_prediction_and_ipopt_objective():
    from examples.mpc.fingertips.test.test_scm_mujoco import build_parser
    from examples.mpc.franka.ik2.params import ExplicitMPCParams, build_lambda_optimizer
    args = build_parser().parse_args(['--solver', 'cem'])
    source = ExplicitMPCParams(args, rand_seed=0, build_optimizer=False)
    opt = build_lambda_optimizer(source, args)
    # Include an active support patch, nonzero gap and online calibration.
    jac = np.zeros_like(opt.J_tilde)
    n, t1, t2 = np.eye(3)[2], np.eye(3)[0], np.eye(3)[1]
    jac[:4, :3] = [n+.7*t1, n+.7*t2, n-.7*t1, n-.7*t2]
    gaps = np.ones(len(jac)); gaps[:4] = -.0001
    opt.set_scm_parameters(np.array([.1, -.05, .02, -.02, .03, .1]))
    opt.update_Jacobian(jac, phi=gaps)
    point = opt.sample_point[0]
    frame = np.column_stack((opt.normal[0], opt.t1[0], opt.t2[0]))
    force, velocity = np.array([.6, .05, -.02]), np.array([.01, 0., 0., 0., .1, 0.])
    gravity = np.array([0., 0., -.098, 0., 0., 0.])
    target = np.array([.02, 0., 0., 1., 0., 0., 0.])
    kwargs = dict(x_d=target, current_x=[0., 0., 0., 1., 0., 0., 0.], v_last=velocity,
                  J_tilde=opt.J_tilde, D_inv=opt.compute_env_diag_inverse(opt.J_tilde),
                  tau_o_np=gravity, n_arm=frame[:, 0], t1=frame[:, 1], t2=frame[:, 2],
                  p_arm=point, curr_ori_coef=opt.ori_coef, env_phi=opt.env_phi,
                  scm_theta=opt.scm_theta, object_config=opt.object_cost_config(), force_limit=3.)
    evaluation = opt.force_objective_fn(lam_arm=force, **kwargs)
    prediction = opt.predict_contact_response(force, point, frame, gravity, velocity=velocity)
    np.testing.assert_allclose(np.array(evaluation['x_plus_opt']).ravel(), prediction['x_plus'], atol=1e-12)
    expected = (object_cost_np(prediction['x_plus'], target, opt.object_cost_config()) +
                opt.friction_reg_coef * force[1:] @ force[1:] + opt.force_reg_coef * force @ force)
    assert float(evaluation['cost']) == pytest.approx(expected, abs=1e-12)
    # Rebuild the existing IPOPT problem and compare its unexpanded graph.
    opt._precompile_optimization_function()
    unexpanded = opt.force_objective_fn(lam_arm=force, **kwargs)
    np.testing.assert_allclose(evaluation['x_plus_opt'], unexpanded['x_plus_opt'], atol=1e-12)
    assert float(evaluation['cost']) == pytest.approx(float(unexpanded['cost']), abs=1e-12)


@pytest.mark.parametrize('size,iterations,fraction', [(3, 2, .25), (8, 0, .25), (8, 2, 0.)])
def test_invalid_population_budget_is_rejected(size, iterations, fraction):
    with pytest.raises(ValueError):
        ForcePopulationSearch('cem', size, iterations, elite_fraction=fraction)
