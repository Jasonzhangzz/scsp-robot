"""Do not solve a contact horizon when its action is replaced by approach."""
import sys
import ast
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock

import numpy as np
import pytest

from planning.mpc_explicit import handle_mpc_request


@pytest.mark.parametrize('runtime', [True, False])
def test_margin_patch_survives_isaac_observation_transport(runtime):
    source = Path(__file__).resolve().parents[1] / 'examples/mpc/franka/ik2/test_mppi_isaac.py'
    tree = ast.parse(source.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('_plan_payload', '_plan_from_obs')]
    scope = {'np': np}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(source), 'exec'), scope)
    q = np.r_[np.zeros(3), [1., 0., 0., 0.], np.ones(3)]
    measured = dict(dist=.0005, point_world=np.zeros(3), normal_world=np.array([1.,0.,0.]))
    jac = np.zeros((4, 9))
    contact = SimpleNamespace(detect_once=lambda env: (.0005*np.ones(4), jac, [], jac, False),
                              get_actual_fingertip_contact=lambda: measured)
    env = SimpleNamespace(param_=SimpleNamespace(runtime_object_cost_=runtime),
                          get_policy_state=lambda: q, get_state=lambda: q)
    observation, _, physical_contact, gap = scope['_plan_payload'](env, contact, .362)
    request, _ = scope['_plan_from_obs'](observation)
    assert not physical_contact and not request['if_contact']
    assert gap == .0005
    assert (request['actual_contact'] is not None) == runtime
    np.testing.assert_array_equal(request['phi_vec'], .0005*np.ones(4))


@pytest.mark.parametrize('contact,flags,near,skip,staged,gap', [
    (True, {'switch_observed_contact': True}, True, True, False, None),
    (True, {'path_blocked': True}, True, False, False, None),
    (False, {}, False, True, False, None),
    (True, {}, True, False, False, None),
    (False, {}, True, False, False, None),
    (False, {}, True, True, True, None),
    (False, {}, True, False, True, .0005),  # Measured margin patch can close.
    (False, {}, True, True, True, .02),    # A distant patch needs approach.
    (False, {'path_blocked': True}, True, False, True, .0005),
])
def test_only_solve_when_mpc_action_will_be_executed(monkeypatch, contact, flags, near, skip, staged, gap):
    rollout = ModuleType('examples.mpc.fingertips.test.test_0902')
    warp = ModuleType('planning.MPPIWarp')
    q = np.r_[[0., 0., .4, 1., 0., 0., 0.], [.1, 0., .4]]
    tip = q[7:10]
    policy = dict(arrived_hold=False, arrived_dest_idx=None, verify_cost=0. if flags.get('path_blocked') else .7,
        value_info=dict(flags, verify=.7, contact_active=True), mpc_virtual_point=tip+np.array([.004,0.,0.]),
        mpc_contact_point=tip, p_arm_world=tip if near else tip+.1,
        x_plus_opt=np.ones(7))
    rollout.compute_rollout_contact_via = lambda *a, **kw: policy
    rollout._verify_is_chatter = lambda *a: False
    warp._apply_dwell_payload = lambda *a: None
    warp._opt_snapshot = lambda *a: {}
    warp._pickle_safe = lambda value: value
    warp.clamp_via_to_tip = lambda tip, via, lead: via
    monkeypatch.setitem(sys.modules, rollout.__name__, rollout)
    monkeypatch.setitem(sys.modules, warp.__name__, warp)
    warm = {'x0': np.ones(3)}
    tracker = dict(value_tracker=None, model_cost_conf=SimpleNamespace(tightness=lambda:0., accum=0.),
        approach_via=None, arrived_hold=False, arrived_dest_idx=None, last_verify_cost=0., sol_guess=warm)
    param = SimpleNamespace(lambda_optimizer=SimpleNamespace(m=.01), gravity_=np.array([0,0,-9.8]),
        max_ncon_=1, n_qvel_=9, table_height=.35, mpc_u_ub_=.005*np.ones(3),
        target_p_=np.zeros(3), target_q_=np.array([1.,0,0,0]), runtime_object_cost_=False)
    mpc = SimpleNamespace(plan_once=Mock(return_value=dict(action=np.array([.001,0.,0.]),
        sol_guess=warm, cost_opt=1., solver_backend='acados')))
    if staged:
        tracker['object_cost_stages'] = SimpleNamespace(enabled=True, stage='flip',
            update=lambda msg: False, config=lambda *a: np.ones(9),
            diagnostics=lambda: {}, progress_scale=lambda:20.)
        param.lambda_optimizer.set_object_cost_config = lambda value: None
        param.lambda_optimizer.set_contact_force_limit = lambda value: None
    msg = dict(policy_q=q, phi_vec=np.ones(4), jac_mat=np.zeros((4,9)),
        jac_mat_env=np.zeros((4,9)), table_ground=.362, if_contact=contact)
    if gap is not None:
        msg['actual_contact'] = dict(dist=gap, point_world=tip, normal_world=np.array([1.,0.,0.]))
        msg['phi_vec'][:] = gap
    result = handle_mpc_request(SimpleNamespace(), param, mpc, tracker, msg)
    # Request handling must not turn a smooth critic into a binary contact gate.
    assert result['verify_cost'] == .7
    assert result['policy']['value_info']['contact_active']
    assert (param.lambda_actual_contact_ is not None) == (staged and gap is not None and not gap > .005)
    if not skip:
        np.testing.assert_array_equal(mpc.plan_once.call_args[0][3], msg['phi_vec'])
        assert mpc.plan_once.call_args[1]['verify_cost_param'] == .7
    assert result['mpc_skipped'] == skip
    assert mpc.plan_once.call_count == (0 if skip else 1)
    np.testing.assert_allclose(result['action'], [.004 if skip else .001, 0., 0.], atol=1e-8)
    assert result['policy']['execution_mode'] == ('approach' if skip else 'contact')
    assert result['total_dt'] >= result['rank_dt'] + result['plan_dt']
    if skip:
        assert tracker['sol_guess'] is None
        assert result['sol_guess'] is None
        assert result['policy']['x_plus_opt'] is None


@pytest.mark.parametrize('converges', [True, False])
def test_rti_projects_force_without_refinement_or_fallback(converges):
    import casadi as cs
    from planning.mpc_explicit import MPCExplicit
    class Solver:
        iteration = 0
        def set(self, *args): pass
        def constraints_set(self, *args): pass
        def solve(self):
            self.iteration += 1
            return 0
        def get(self, stage, field):
            if field == 'x': return np.zeros(3)
            excess = max(0., (7-self.iteration)*1e-5) if converges else 1e-5
            return np.array([(3.+excess)/600., 0., 0.])
        def get_stats(self, field): return 1
    mpc = MPCExplicit.__new__(MPCExplicit)
    mpc.param_ = SimpleNamespace(mpc_horizon_=2, n_cmd_=3, n_qpos_=3,
        robot_stiff_=600*np.eye(3), model_params=1., runtime_object_cost_=False)
    mpc.model = SimpleNamespace(runtime_parameters=lambda p: p)
    mpc.acados_solver_ = Solver()
    mpc.acados_solve_count = mpc.acados_failure_count = 0
    mpc.contact_force_limit, mpc.update_contact_gaps = 3., False
    mpc.nlp_lam_g0_ = cs.DM.zeros(6)
    result = mpc._plan_once_acados(np.zeros(3), np.ones(4), np.zeros((4,3)),
        np.zeros(3), None, -.005*np.ones(3), .005*np.ones(3))
    assert result['solver_backend'] == 'acados'
    assert mpc.acados_solver_.iteration == 1
    assert mpc.acados_failure_count == 0
    forces = result['u_traj'] @ mpc.param_.robot_stiff_.T
    assert np.max(np.linalg.norm(forces, axis=1)) <= 3.+1e-10
