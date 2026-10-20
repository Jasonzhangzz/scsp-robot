"""Upstream algorithm parity and absence of the local policy stack."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import argparse
import hashlib
import json

import casadi as cs
import numpy as np
import pytest

from planning.contactsdf_franka import upstream_parameters, mesh_pointcloud, FrankaContactSDF
from examples.mpc.franka.scm.vendor.contactsdf import rotations, pointcloudMap
from examples.mpc.franka.scm.vendor.contactsdf.ball_lse_contact_detection import Contact
from examples.mpc.franka.scm.vendor.contactsdf.single_ball_cost import Cost

VENDOR = Path(__file__).resolve().parents[1]/'examples/mpc/franka/scm/vendor/contactsdf'


def original_module(relative, detector=False):
    """Execute a preserved upstream file with only import paths redirected."""
    code = (VENDOR/'upstream'/relative).read_text()
    code = code.replace('from utils import rotations', '').replace('from utils import pointcloudMap', '')
    if detector:
        code = code.replace('from simulator.fts_simulator import MjSimulator', 'MjSimulator = object')
    module = ModuleType('reference')
    module.rotations, module.pointcloudMap = rotations, pointcloudMap
    exec(compile(code, relative, 'exec'), module.__dict__)
    return module


def test_original_sources_and_live_solver_are_preserved():
    provenance = json.loads((VENDOR/'PROVENANCE.json').read_text())
    for name, digest in provenance['original_sha256'].items():
        assert hashlib.sha256((VENDOR/'upstream'/name).read_bytes()).hexdigest() == digest
    original = (VENDOR/'upstream/planning/ball_mpc_lse.py').read_text()
    live = (VENDOR/'ball_mpc_lse.py').read_text()
    assert live == original.replace('from dynamics.lse_dynamics import LseDynamic',
                                     'from .lse_dynamics import LseDynamic')
    assert (VENDOR/'lse_dynamics.py').read_bytes() == (VENDOR/'upstream/dynamics/lse_dynamics.py').read_bytes()


@pytest.mark.parametrize('family,file', [
    ('cube', 'cost/ball_cube/cost_three_ball_cube.py'),
    ('foambrick', 'cost/ball_foambrick/cost_three_ball_foambrick.py'),
    ('stick', 'cost/ball_stick/cost_three_ball_stick.py')])
def test_generalized_cost_matches_original_three_tip_values_and_derivatives(family, file):
    param = upstream_parameters(family, fingertip_count=3)
    actual = Cost(param).init_cost_fn()
    expected = original_module(file).Cost(param).init_cost_fn()
    x, u, target = cs.SX.sym('x', 16), cs.SX.sym('u', 9), cs.SX.sym('target', 7)
    a, b = actual[0](x, u, target), expected[0](x, u, target)
    af, bf = actual[1](x, target), expected[1](x, target)
    compare = cs.Function('cost_parity', [x, u, target],
        [a, b, af, bf, cs.jacobian(a, x), cs.jacobian(b, x),
         cs.jacobian(af, x), cs.jacobian(bf, x)])
    rng = np.random.default_rng(5)
    for _ in range(8):
        state = rng.normal(0, .1, 16); state[3:7] /= np.linalg.norm(state[3:7])
        goal = rng.normal(0, .1, 7); goal[3:7] /= np.linalg.norm(goal[3:7])
        values = compare(state, rng.uniform(-.01, .01, 9), goal)
        for i in (0, 2, 4, 6):
            np.testing.assert_allclose(values[i], values[i+1], atol=1e-10, rtol=1e-12)


def test_single_tip_cost_retains_closure_instead_of_adding_local_targets():
    param = upstream_parameters('cube')
    path, terminal = param.planning_cost_.init_cost_fn()
    pose = np.r_[np.zeros(3), 1., 0., 0., 0., [.05, 0., 0.]]
    target = np.r_[[.1, 0., 0.], 1., 0., 0., 0.]
    expected = .05**2 + .1*(.05/(.05**2+1e-4))**2
    assert float(path(pose, np.zeros(3), target)) == pytest.approx(expected)
    assert float(terminal(pose, target)) == pytest.approx(1000*10*.1**2)
    assert path.n_in() == 3 and path.size1_in(2) == 7


def test_detector_matches_original_with_three_tips_and_uses_half_sdf(tmp_path):
    import trimesh
    mesh = tmp_path/'cube.stl'; trimesh.creation.box(extents=[.056]*3).export(mesh)
    param = upstream_parameters('cube', fingertip_count=3)
    param.pointcloud_path_ = mesh_pointcloud(mesh, tmp_path/'points')
    param.gd_points_ = np.array([[0., 0., 0.], [.02, 0., 0.], [-.02, 0., 0.]])
    original = original_module('contact/ball_lse_contact_detection.py', detector=True).Contact(param)
    adapted = Contact(param)
    q = np.r_[[0., 0., .028], 1., 0., 0., 0., [.032, 0., .028], [0., .032, .028], [.1, .1, .1]]
    for a, b in zip(adapted.detectOnce(q), original.detectOnce(q)):
        np.testing.assert_array_equal(a, b)
    single = upstream_parameters('cube')
    single.pointcloud_path_, single.gd_points_ = param.pointcloud_path_, param.gd_points_
    phi, n, f = Contact(single).detectOnce(q[:10])
    distance = float(adapted.contact_lse_fn_(q[7:10]-q[:3]))
    assert phi[0] == pytest.approx(.5*distance)
    assert n.shape == f.shape == (16, 9)


def test_single_tip_controller_only_detects_and_calls_official_plan_once(monkeypatch, tmp_path):
    import planning.contactsdf_franka as module
    calls = []
    class Detector:
        def __init__(self, p): self.p = p
        def detectOnce(self, q):
            calls.append(('detect', q.copy()))
            return np.ones(16), np.zeros((16, 9)), np.zeros((16, 9))
    class Planner:
        def __init__(self, p):
            assert p.h_ == .1 and p.mpc_horizon_ == 4 and p.mpc_u_ub_ == .01
        def planOnce(self, *args):
            calls.append(('plan', args))
            return dict(action=np.array([.001, 0., 0.]), sol_guess=dict(x0=[1]))
    monkeypatch.setattr(module, 'Contact', Detector)
    monkeypatch.setattr(module, 'MPC_IPOPT', Planner)
    monkeypatch.setattr(module, 'mesh_pointcloud', lambda *a: 'input.txt')
    scene = SimpleNamespace(contactsdf_theta_source_='cube', contactsdf_theta_=np.ones(8),
        mesh_path_='mesh.stl', init_obj_qpos_=np.r_[[.6, .2, .4], 1., 0., 0., 0.],
        table_height=.35, target_p_=np.array([.5, .1, .4]), target_q_=np.array([1., 0., 0., 0.]))
    model = FrankaContactSDF(scene, tmp_path)
    state = np.r_[scene.init_obj_qpos_, [.65, .2, .4]]
    model.plan(state); model.plan(state)
    assert [c[0] for c in calls] == ['detect', 'plan', 'detect', 'plan']
    assert calls[1][1][-1] is None
    assert calls[3][1][-1] == dict(x0=[1])
    np.testing.assert_allclose(calls[0][1][:3], [0., 0., .05])
    assert not hasattr(model, 'lambda_optimizer')


def test_standalone_does_not_build_lambda_and_rejects_strategy_flags(monkeypatch):
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    from examples.mpc.franka.runtime.mpc import _add_rollout_policy_args
    from examples.mpc.franka.runtime.params import ExplicitMPCParams
    from examples.mpc.franka.contactsdf_run import validate_options
    import examples.mpc.franka.runtime.params as params
    def forbidden(*args, **kwargs): raise AssertionError('Lambda was constructed')
    monkeypatch.setattr(params, 'build_lambda_optimizer', forbidden)
    parser = _add_rollout_policy_args(argparse.ArgumentParser())
    args = parser.parse_args(['--contactsdf', '--obj', 'elephant', '--headless', '--sync-planner'])
    validate_options(args, ['--contactsdf'])
    scene = ExplicitMPCParams(args)
    assert scene.lambda_optimizer is None
    for flag in ['--verify_beta', '--ema', '--two-stage-object-cost', '--cso-guidance', '--mpc-contact-multistart']:
        with pytest.raises(ValueError, match='strategy'):
            validate_options(args, ['--contactsdf', flag])


@pytest.mark.parametrize('flags', [
    ['--contactsdf'], ['--contact-model=contactsdf'], ['--contact-model', 'contactsdf']])
def test_public_dispatch_replaces_process_before_local_rollout(monkeypatch, flags):
    from models.contactsdf import dispatch_standalone
    import os
    import sys
    class Dispatched(Exception): pass
    def execv(executable, argv):
        assert executable == sys.executable
        assert argv[2] == '/repository/examples/mpc/franka/contactsdf_run.py'
        assert argv[3:] == flags
        raise Dispatched
    monkeypatch.setattr(os, 'execv', execv)
    with pytest.raises(Dispatched):
        dispatch_standalone('/repository', flags)


def test_local_mpc_cannot_construct_a_hybrid_contactsdf_controller():
    from planning.mpc_explicit import MPCExplicit
    with pytest.raises(ValueError, match='official MPC'):
        MPCExplicit(SimpleNamespace(contact_model='contactsdf'))


def test_cached_task_restores_interface_state_and_rejects_wrong_case(tmp_path):
    from examples.mpc.franka.contactsdf_run import load_cached_task, apply_cached_task
    task = dict(object='foam_brick', seed=2, target_type='rotation',
        goal_xyz=[.4, .1, .38], goal_quat_wxyz=[1., 0., 0., 0.],
        init_policy_state=[.3, 0., .4, 1., 0., 0., 0., .2, 0., .4],
        init_joint_position=[0., -.7, 0., -2.5, 0., 1.5, .8],
        collision_surface='/original/surface.stl', collision_parts=['/original/part_000.stl'])
    path = tmp_path/'task.json'
    path.write_text(json.dumps(task))
    args = SimpleNamespace(obj='foam_brick', target_type='rotation', contactsdf_task_json=str(path))
    scene = SimpleNamespace(contact_model='contactsdf', lambda_optimizer=None)
    apply_cached_task(scene, load_cached_task(args, 2))
    np.testing.assert_array_equal(scene.target_p_, task['goal_xyz'])
    np.testing.assert_array_equal(scene.init_obj_qpos_, task['init_policy_state'][:7])
    np.testing.assert_array_equal(scene.init_robot_qpos_, task['init_joint_position'])
    assert scene.mesh_path_ == '/original/surface.stl'
    assert scene.collision_mesh_paths_ == ['/original/part_000.stl']
    assert scene.lambda_optimizer is None
    with pytest.raises(ValueError, match='differs'):
        load_cached_task(args, 3)
