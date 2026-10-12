"""Evaluation accounting and fixed-step semantics, without Isaac/CUDA imports."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.mpc.franka.ik2.headless_rollout import trial_ranges, report, PipeBus


@pytest.mark.parametrize('flags, entry', [
    (['--headless'], 'headless_rollout.py'),
    (['--headless', '--async-planner'], 'headless_rollout.py'),
    (['--headless', '--sync-planner'], 'headless_rollout.py'),
    (['--physics-backend', 'mjwarp', '--headless'], 'headless_rollout.py'),
    (['--headless', '--viewer'], 'run.py'),
    (['--viewer', '--headless'], 'headless_rollout.py'),
])
def test_headless_keeps_developer_async_default_and_sync_is_explicit(monkeypatch, flags, entry):
    import argparse
    import os
    import sys
    source = Path('examples/mpc/franka/ik2/test_mpc_isaac.py').resolve()
    tree = ast.parse(source.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    monkeypatch.setattr(sys, 'argv', [str(source), *flags])
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    launches = []
    monkeypatch.setattr(os, 'execv', lambda exe, argv: launches.append(argv))
    scope = dict(argparse=argparse, os=os, sys=sys, __file__=str(source))
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), 'exec'), scope)
    scope['main']()
    assert Path(launches[0][2]).name == entry
    assert all(flag in launches[0] for flag in flags)


@pytest.mark.parametrize('lanes, async_planner', [(1, True), (5, True), (5, False)])
def test_parallel_isaac_dispatch_preserves_requested_cadence(lanes, async_planner):
    source = Path('examples/mpc/franka/ik2/headless_rollout.py')
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_isaac_headless')
    calls = []
    scope = dict(run_isaac_parallel=lambda args: calls.append(args))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), scope)
    args = SimpleNamespace(num_envs=lanes, async_planner=async_planner)
    scope['run_isaac_headless'](args)
    assert calls == [args]
    assert args.async_planner == async_planner


def test_uneven_batches_cover_every_trial_once():
    ranges = trial_ranges(5, 7, 3)
    assert [list(r) for r in ranges] == [[5, 6, 7], [8, 9], [10, 11]]
    assert [list(r) for r in trial_ranges(0, 2, 8)] == [[0], [1]]


@pytest.mark.parametrize('values', [(-1, 2, 1), (0, 0, 1), (0, 2, 0)])
def test_invalid_evaluation_sizes_fail(values):
    with pytest.raises(ValueError):
        trial_ranges(*values)


def test_aggregate_requires_completed_unique_trials(tmp_path):
    args = SimpleNamespace(trial_start=3, trial_num=2, physics_backend='mjwarp',
                           solver_backend='acados', num_envs=2,
                           rollout_summary_json=str(tmp_path/'summary.json'))
    records = [dict(trial=3, success=1), dict(trial=4, success=0)]
    result = report(args, records, 0.)
    assert result['success_rate'] == .5
    assert json.loads((tmp_path/'summary.json').read_text())['trials'] == 2
    for bad in [records[:1], [records[0], records[0]]]:
        with pytest.raises(RuntimeError):
            report(args, bad, 0.)
    records[0]['aborted'] = True
    with pytest.raises(RuntimeError, match='Aborted'):
        report(args, records, 0.)


def test_pipe_bus_preserves_every_action_and_summary():
    import multiprocessing as mp
    parent, child = mp.Pipe()
    try:
        bus = PipeBus(child)
        parent.send({'trial': 2, 'seq': 0})
        assert bus.wait_obs(timeout=1.) == {'trial': 2, 'seq': 0}
        bus.publish_cmd({'action': [1, 2, 3]})
        bus.record_summary({'trial': 2, 'success': 1})
        assert parent.recv() == ('command', {'action': [1, 2, 3]})
        assert parent.recv() == ('summary', {'trial': 2, 'success': 1})
    finally:
        parent.close()
        child.close()


def test_headless_synchronous_bus_has_no_render_or_realtime_wait(monkeypatch):
    # Load only this dependency-free class, with a fake physics executor.
    # The test verifies action admission and exactly ten 2 ms frames.
    tree = ast.parse(Path('examples/mpc/franka/ik2/run.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SynchronousMPCBus')
    calls = []
    env = SimpleNamespace(sim_dt_=.002, break_out_signal_=False,
                          step_control_frame=lambda **kw: calls.append(kw),
                          sync_realtime=lambda: pytest.fail('Realtime wait in headless evaluation'))
    scope = dict(POLICY_INTERVAL=.02, viewer_draw_stride=lambda *a: 8,
                 viewer_closed=lambda e: False, apply_incoming=lambda *a: None)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<synchronous-bus>', 'exec'), scope)
    bus = scope['SynchronousMPCBus'](SimpleNamespace(trial_start=0, trial_num=1, headless=True))
    bus.env = env
    bus.publish_cmd({'action': [0, 0, 0]})
    assert calls == [dict(draw=False, sync_realtime=False)]*10
    assert bus.seq == 1


def test_real_mjwarp_fixed_step_and_isolated_reset(monkeypatch):
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    pytest.importorskip('mujoco_warp')
    wp = pytest.importorskip('warp')
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip('CUDA device required')
    import argparse
    import numpy as np
    import time
    from examples.mpc.franka.ik2.test_mpc_isaac import _add_rollout_policy_args
    from examples.mpc.franka.ik2.mjwarp_rollout import MJWarpBatch
    args = _add_rollout_policy_args(argparse.ArgumentParser()).parse_args(
        ['--physics-backend', 'mjwarp', '--obj', 'foam_brick', '--sample_num', '8', '--headless'])
    batch = MJWarpBatch(args, [3, 4])
    initial = [batch.observation(i, i+3) for i in range(2)]
    monkeypatch.setattr(time, 'sleep', lambda *a: pytest.fail('sleep in headless physics'))
    for i, action in enumerate(([.005, 0, 0], [0, -.005, 0])):
        batch.command(i, dict(seq=0, via=initial[i]['policy_q'][7:]+action, action=action))
    batch.step([0, 1])
    for i in range(2):
        obs = batch.observation(i, i+3)
        assert obs['seq'] == 1
        assert obs['sim_time'] == pytest.approx(.02, abs=1e-7)
        assert np.isfinite(obs['policy_q']).all()
    untouched = {name: getattr(batch.wdata, name).numpy()[1].copy()
                 for name in ('qpos', 'qvel', 'qacc_warmstart', 'time')}
    batch.reset(0, 9)
    assert batch.observation(0, 9)['sim_time'] == 0.
    assert batch.observation(0, 9)['seq'] == 0
    for name, value in untouched.items():
        np.testing.assert_array_equal(getattr(batch.wdata, name).numpy()[1], value)
    # Finished worlds stay frozen while the remaining world advances.
    frozen = batch.observation(0, 9)['policy_q'].copy()
    batch.command(1, dict(seq=1, via=batch.observation(1, 4)['policy_q'][7:], action=[0, 0, 0]))
    batch.step([1])
    np.testing.assert_array_equal(batch.observation(0, 9)['policy_q'], frozen)
    assert batch.observation(0, 9)['sim_time'] == 0.
    assert batch.observation(1, 4)['sim_time'] == pytest.approx(.04, abs=1e-7)
    # The URDF speed limits prevent an initially overspeed arm from being
    # passed to the next OSC frame. Finished worlds remain untouched.
    velocities = batch.wdata.qvel.numpy()
    velocities[1, :7] = 30.
    batch.wdata.qvel.assign(velocities)
    batch.envs[1].data.qvel[:7] = 30.
    batch.step([1])
    assert batch.envs[1].joint_velocity_projections > 0
    assert np.isfinite(batch.envs[1].data.qpos).all()
    assert np.all(np.abs(batch.envs[1].data.qvel[:7]) <= batch.velocity_limits + 1e-6)
    np.testing.assert_array_equal(batch.observation(0, 9)['policy_q'], frozen)


def test_mjwarp_scene_matches_exported_inertias_and_franka_kinematics(monkeypatch):
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    mj = pytest.importorskip('mujoco')
    pytest.importorskip('mujoco_warp')
    import argparse
    import numpy as np
    from examples.mpc.franka.ik2.test_mpc_isaac import _add_rollout_policy_args, _franka_fk_T_np
    from examples.mpc.franka.ik2.mjwarp_rollout import build_param, scene_xml, isaac_robot_dynamics
    args = _add_rollout_policy_args(argparse.ArgumentParser()).parse_args(
        ['--physics-backend', 'mjwarp', '--obj', 'foam_brick', '--sample_num', '8', '--headless'])
    model = mj.MjModel.from_xml_string(scene_xml(build_param(args, 0), args))
    for body in isaac_robot_dynamics()['bodies']:
        i = model.body(body['name'].replace('panda_link', 'link')).id
        assert model.body_mass[i] == pytest.approx(body['mass'])
        np.testing.assert_allclose(model.body_ipos[i], body['com_pos'], atol=1e-12)
        rotation = np.zeros(9)
        mj.mju_quat2Mat(rotation, model.body_iquat[i])
        rotation = rotation.reshape(3, 3)
        inertia = rotation @ np.diag(model.body_inertia[i]) @ rotation.T
        # Exported PhysX tensors are float32; MuJoCo's inertia eigensolver also
        # has a finite tolerance when reconstructing small off-diagonal terms.
        np.testing.assert_allclose(inertia, body['inertia'], atol=1e-7)
    expected_pairs = {frozenset(('object_part_0', 'table')): (.7, 0.),
                      frozenset(('object_part_0', 'fingertip')): (1.2, .001),
                      frozenset(('fingertip', 'table')): (1., .001)}
    for i in range(model.npair):
        pair = frozenset((model.geom(model.pair_geom1[i]).name, model.geom(model.pair_geom2[i]).name))
        friction, margin = expected_pairs[pair]
        np.testing.assert_allclose(model.pair_friction[i, :2], [friction, friction])
        assert model.pair_margin[i] == pytest.approx(margin)
    data = mj.MjData(model)
    rng = np.random.default_rng(7)
    for q in rng.uniform(model.jnt_range[:7, 0], model.jnt_range[:7, 1], (20, 7)):
        data.qpos[:7] = q
        mj.mj_forward(model, data)
        expected = _franka_fk_T_np(q)
        tip = model.body('fingertip').id
        np.testing.assert_allclose(data.xpos[tip], expected[:3, 3], atol=2e-7)
        np.testing.assert_allclose(data.xmat[tip].reshape(3, 3), expected[:3, :3], atol=2e-7)
