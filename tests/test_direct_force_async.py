"""A pending solve must not block physics or observe mutable plant buffers."""
import threading
from types import SimpleNamespace

import mujoco
import numpy as np

from planning.direct_force_async import AsyncForcePlanner


class Stages:
    def config(self):
        return {}

    def progress_scale(self):
        return 1.


def test_pending_planner_uses_isolated_snapshot_and_retains_command(monkeypatch):
    from planning import object_pose_cost
    monkeypatch.setattr(object_pose_cost, 'object_cost_np', lambda *args: 0.)
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <body><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
        </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    opt = SimpleNamespace(last_selected_idx=0, last_solver_status='test-lm',
                          note_contact_progress=lambda *args, **kwargs: None)
    param = SimpleNamespace(lambda_optimizer=opt, force_cost_stages_=Stages(),
                            target_p_=np.zeros(3), target_q_=np.array([1., 0., 0., 0.]))
    env = SimpleNamespace(model_=model, data_=data, param_=param)
    args = SimpleNamespace(contact_dwell_gamma=.85, contact_dwell_steps=6)
    entered, release = threading.Event(), threading.Event()
    observations = []

    def solve(p, args, observation, contact, **kw):
        assert observation.data_ is not env.data_
        if observations:
            entered.set()
            assert release.wait(timeout=5)
        observations.append(observation.data_.qpos.copy())
        return {'p_local': np.array([.1, 0., 0.])}

    planner = AsyncForcePlanner(env, param, args, solve, lambda p: None, 'surrogate', 1., 1.)
    try:
        command, updated = planner.poll(env)
        assert not updated
        assert entered.wait(timeout=5)
        env.data_.qpos[0] = .7
        # Polling while inference is pending returns immediately, retaining
        # the last command; advancing physics cannot mutate its observation.
        assert planner.poll(env) == (command, False)
        release.set()
        planner.future.result(timeout=5)
        assert observations[-1][0] == 0.
        assert env.data_.qpos[0] == .7
        new_command, updated = planner.poll(env)
        assert updated and new_command is not command
        np.testing.assert_array_equal(new_command['p_local'], [.1, 0., 0.])
    finally:
        release.set()
        planner.close()


def test_contact_settings_and_isaac_angular_decay(monkeypatch):
    from examples.mpc.fingertips.test.test_0902 import build_parser
    from examples.mpc.fingertips.test.params import ExplicitMPCParams
    from envs.fingertips_env import MjSimulator
    from planning.direct_contact_force import configure_direct_force_optimizer

    monkeypatch.setenv('MUJOCO_HEADLESS', '1')
    args = build_parser().parse_args([])
    args.rollout, args.solver = True, 'ipopt'
    param = ExplicitMPCParams(args, rand_seed=0)
    env = MjSimulator(param)
    configure_direct_force_optimizer(param, env, args)
    assert env.model_.opt.impratio == 10.
    assert env.model_.opt.noslip_iterations == 3
    # An unforced principal-axis rotation must decay with Isaac's 0.5/s
    # rate, independent of the object's very small rotational inertia.
    env.data_.qpos[2] = 2.
    env.data_.qvel[:] = 0.
    env.data_.qvel[5] = 1.
    env.model_.opt.gravity[:] = 0.
    for _ in range(2000):
        mujoco.mj_step(env.model_, env.data_)
    np.testing.assert_allclose(env.data_.qvel[5], np.exp(-.5), atol=1e-4)
