"""Native regressions for startup work and an observed contact-switch QP failure."""
import argparse
import ctypes
import json
from pathlib import Path

import casadi as cs
import numpy as np
import pytest


def args():
    from examples.mpc.franka.ik2.test_mpc_isaac import _add_rollout_policy_args
    return _add_rollout_policy_args(argparse.ArgumentParser()).parse_args(
        ['--obj', 'foam_brick', '--sample_num', '70', '--solver-backend', 'acados'])


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    from planning.acados_env import ensure_acados_env
    ensure_acados_env()
    pytest.importorskip('acados_template')
    from planning.mpc_explicit import _build_mpc_planner_runtime
    a = args()
    return _build_mpc_planner_runtime(dict(args=vars(a), trial_count=0))


def test_observed_min_step_state_recovers_without_increasing_force_limit(runtime):
    _, param, mpc, _ = runtime
    record = json.loads((Path(__file__).parent/'fixtures/mpc_switch_floor_recovery.json').read_text())
    state, phi, jac, cp, lb, ub = [np.array(record[k]) for k in
                                 ('state', 'gaps', 'jacobian', 'cost_parameters', 'u_lower', 'u_upper')]
    warm = {k:np.array(v) for k, v in record['warm'].items()}
    param.object_cost_config_ = np.array(record['object_config'])
    mpc.first_action_bounds = tuple(np.array(record['first_action_bounds']))
    assert mpc.acados_solver_ is not None, str(mpc._acados_init_error)
    assert mpc.recoverable_floor
    for guess in (warm, None, warm):
        result = mpc._plan_once_acados(state, phi, jac, cp, guess, lb, ub)
        assert result is not None, mpc.last_acados_error
        assert result['solver_backend'] == 'acados'
        force = np.linalg.norm(result['u_traj'] @ param.robot_stiff_.T, axis=1)
        assert np.max(force) <= param.contact_command_force_limit_ + 1e-7
        np.testing.assert_array_less(mpc.first_action_bounds[0]-1e-6, result['action'])
        np.testing.assert_array_less(result['action'], mpc.first_action_bounds[1]+1e-6)
        # Braking reduces the predicted dip from ~3 mm (no floor constraint)
        # to <1 mm while retaining the original bounded-wrench dynamics.
        assert np.min(result['rollout_q'][:, 9]) > param.mpc_q_lb_[9]-.001
        assert result['action'][2] > .004
    assert mpc.acados_failure_count == mpc.acados_fallback_count == 0
    assert mpc.ipopt_solver is None
    assert param.lambda_optimizer.optimization_fn is None


def test_acados_startup_reuses_local_param_and_does_not_build_ipopt(monkeypatch):
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    from planning.acados_env import ensure_acados_env
    ensure_acados_env()
    pytest.importorskip('acados_template')
    from examples.mpc.franka.ik2.params import ExplicitMPCParams
    from examples.mpc.franka.ik2.test_mpc_isaac import adapt_param_for_cartesian_solver, _configure_rollout_param
    from planning.mpc_explicit import _build_mpc_planner_runtime, planner_init_payload
    monkeypatch.setattr(cs, 'nlpsol', lambda *a, **kw: pytest.fail('eager IPOPT construction in acados startup'))
    a = args()
    param = _configure_rollout_param(adapt_param_for_cartesian_solver(ExplicitMPCParams(a, rand_seed=0), a), a)
    opt = param.lambda_optimizer
    _, plan_param, mpc, _ = _build_mpc_planner_runtime(planner_init_payload(a, param, 0), param)
    assert plan_param is param and plan_param.lambda_optimizer is opt
    assert mpc.acados_solver_ is not None
    assert mpc.ipopt_solver is None and opt.optimization_fn is None
    # Simulation-side parameters must not build a CSO runtime at all.
    physics_param = ExplicitMPCParams(a, rand_seed=0, build_optimizer=False)
    adapt_param_for_cartesian_solver(physics_param, a, build_optimizer=False)
    _configure_rollout_param(physics_param, a)
    assert physics_param.lambda_optimizer is None


def test_contact_ranking_resets_each_candidate_and_seeds_its_dynamics(runtime, monkeypatch):
    _, param, _, _ = runtime
    opt = param.lambda_optimizer
    solver, original_reset = opt.acados_solver, opt.acados_solver.reset
    resets = []
    def reset(*a, **kw):
        resets.append(1)
        return original_reset(*a, **kw)
    monkeypatch.setattr(solver, 'reset', reset)
    current = np.r_[np.zeros(3), [1., 0., 0., 0.]]
    target = current.copy(); target[0] = .03
    opt.update_Jacobian(np.zeros((4*opt.max_contacts, 6)))
    common = dict(current_x=current, x_d=target, v_last=np.zeros(6), J_tilde=opt.J_tilde,
                  D_inv=opt.compute_env_diag_inverse(opt.J_tilde), tau_o_np=np.zeros(6),
                  n_arm=[1., 0., 0.], t1=[0., 1., 0.], t2=[0., 0., 1.], curr_ori_coef=1.)
    for point in ([0., 0., .03], [0., .03, 0.], [0., 0., .03]):
        result = opt._solve_optimization(p_arm=point, **common)
        assert result is not None and opt.last_solver_status == 'acados'
        assert np.linalg.norm(result['lam_arm_opt']) <= opt.max_contact_force + 1e-7
    assert len(resets) == 3
    assert opt.acados_failure_count == opt.acados_fallback_count == 0


def test_compact_codegen_preserves_predictions_and_analytic_jacobian(runtime):
    _, _, mpc, _ = runtime
    ocp = getattr(mpc.acados_solver_, 'ocp', None)
    if ocp is None:
        ocp = mpc.acados_solver_.acados_ocp
    model = ocp.model
    actual = cs.Function('compact_check', [model.x, model.u, model.p],
                         [model.disc_dyn_expr, model.disc_dyn_custom_jac_ux_expr])
    x, u = cs.SX.sym('test_x', 19), cs.SX.sym('test_u', 3)
    p = cs.SX.sym('test_p', model.p.numel())
    nphi, njac = 4*mpc.param_.max_ncon_, 4*mpc.param_.max_ncon_*mpc.param_.n_qvel_
    ncost = mpc.path_cost_fn.size1_in(2)
    phi, jac = p[:nphi], cs.reshape(p[nphi:nphi+njac], nphi, mpc.param_.n_qvel_)
    cp_end = nphi+njac+ncost
    prediction = mpc.stage_prediction_fn(x, u, phi, jac,
        p[cp_end:cp_end+mpc.model.model_parameter_dim], p[-19:])
    reference = cs.Function('expanded_check', [x, u, p],
        [prediction, cs.jacobian(prediction, cs.vertcat(u, x))])
    record = json.loads((Path(__file__).parent/'fixtures/mpc_switch_floor_recovery.json').read_text())
    state = np.array(record['state'])
    parameters = np.r_[record['gaps'], np.array(record['jacobian']).reshape(-1, order='F'),
        record['cost_parameters'], np.atleast_1d(mpc.model.runtime_parameters(mpc.param_.model_params)), state]
    rng = np.random.default_rng(42)
    for _ in range(5):
        measured = state + rng.normal(0, 1e-4, state.size)
        command = rng.uniform(-.003, .003, 3)
        for a, b in zip(actual(measured, command, parameters), reference(measured, command, parameters)):
            np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-10)
    # Protect the cold-start improvement: the old single Jacobian was 15 MB.
    directory = Path(ocp.code_gen_options.code_export_directory)
    assert max(p.stat().st_size for p in directory.rglob('*dyn_disc_phi_fun_jac.c')) < 3*1024**2


def test_native_buffered_stdout_cannot_leak_after_solve_context(capfd):
    from planning.acados_env import quiet_acados_stderr
    libc = ctypes.CDLL(None)
    with quiet_acados_stderr():
        libc.puts(b'hidden native QP log')
    libc.puts(b'visible after solve')
    libc.fflush(None)
    out, err = capfd.readouterr()
    assert 'hidden native' not in out+err
    assert 'visible after solve' in out


def test_native_warm_cache_load_never_generates_or_compiles(runtime, monkeypatch):
    from acados_template import AcadosOcpSolver
    from planning.acados_env import build_cached_solver
    _, _, mpc, _ = runtime
    ocp = getattr(mpc.acados_solver_, 'ocp', None)
    if ocp is None:
        ocp = mpc.acados_solver_.acados_ocp
    if not hasattr(ocp, 'code_gen_options'):
        pytest.skip('modern acados cache schema required')
    code = ocp.code_gen_options
    # Deliberately fail the upstream raw serializer hash check: the local
    # canonical comparison must still reuse this equivalent native model.
    monkeypatch.setattr(AcadosOcpSolver, 'is_code_reuse_possible', staticmethod(lambda *a, **kw: False))
    def forbidden(*a, **kw):
        pytest.fail('generating/compiling a valid warm cache')
    monkeypatch.setattr(AcadosOcpSolver, 'generate', staticmethod(forbidden))
    monkeypatch.setattr('subprocess.run', forbidden)
    directory = Path(code.code_export_directory)
    solver = build_cached_solver(ocp, code.json_file, directory/('libacados_ocp_solver_'+ocp.model.name+'.so'))
    assert solver is not None
