"""Exercise OSC contact transitions without loading Isaac's native runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np


def _osc_contact_methods():
    source = Path(__file__).resolve().parents[1] / "examples/mpc/franka/ik2/test_mpc_isaac.py"
    tree = ast.parse(source.read_text())
    simulator = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef)
                     and node.name == "IsaacFrankaOSCSimulator")
    names = {"_simulate_once", "_fingertip_contact_info", "_start_via_action", "set_via_action"}
    methods = [node for node in simulator.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {}
    constants = {"POLICY_INTERVAL", "AIR_VIA_STEP", "ISAAC_ACTION_SLEW", "ISAAC_TABLE_CLEARANCE"}
    declarations = [node for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id in constants
                            for target in node.targets)]
    exec(compile(ast.Module(body=declarations, type_ignores=[]), str(source), "exec"), namespace)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("draw", [False, True])
def test_osc_observes_contact_entry_and_exit_between_physics_frames(draw):
    methods = _osc_contact_methods()
    physical_contact = [False]
    reads = []

    def measure_contact():
        reads.append(physical_contact[0])
        return physical_contact[0], None, 0.0

    def simulate(_sim):
        physical_contact[0] = not physical_contact[0]

    env = SimpleNamespace(
        gym=SimpleNamespace(simulate=simulate, fetch_results=lambda *_: None),
        sim=object(), viewer_=None, svg_screenshot_recorder_=None,
        fingertip_object_contact=measure_contact,
    )
    contact = methods["_fingertip_contact_info"]
    advance = methods["_simulate_once"]
    assert contact(env)[0] is False
    assert contact(env)[0] is False
    assert reads == [False]  # Reuse the measurement within one physics frame.
    advance(env, draw=draw)
    assert contact(env)[0] is True
    advance(env, draw=draw)
    assert contact(env)[0] is False
    assert reads == [False, True, False]


@pytest.mark.parametrize("clearance,expected_z", [(None, -0.002), (0.012, 0.002)])
def test_default_contact_press_keeps_downward_pd_command(clearance, expected_z):
    from examples.mpc.franka.ik2.contact_frames import clip_via_target, slew_mpc_action, mpc_ball_trajectory
    from examples.mpc.franka.ik2.isaac_bus import PolicyActionSchedule

    methods = _osc_contact_methods()
    methods.update(np=np, _clip_via_target=clip_via_target,
                   _slew_mpc_action=slew_mpc_action, _mpc_ball_trajectory=mpc_ball_trajectory)
    tip = np.array([0.4, 0.1, 0.36])  # 10 mm sphere resting on a 0.35 m table.
    param = SimpleNamespace(table_height=0.35, mpc_u_ub_=0.005)
    if clearance is not None:
        param.table_clearance_ = clearance
    env = SimpleNamespace(
        param_=param, get_end_effector_pos=lambda: (tip, np.eye(3)),
        _fingertip_in_physical_contact=lambda: True, R_d_hold=np.eye(3),
        _policy_schedule=PolicyActionSchedule(),
    )
    action = np.array([0.0, 0.0, -0.002])
    methods["set_via_action"](env, tip+action, action=action, policy_dt=0.02)
    pending = env._policy_schedule.take_pending(origin=tip)
    methods["_start_via_action"](env, tip+action, action=pending["action"], policy_dt=0.02)
    assert env._mpc_action[2] == pytest.approx(expected_z, abs=1e-8)


@pytest.mark.parametrize("draw_stride", [1, 8, 17])
def test_async_observations_share_pose_contact_frame_and_physics_cadence(draw_stride):
    source = Path(__file__).resolve().parents[1] / "examples/mpc/franka/ik2/run.py"
    tree = ast.parse(source.read_text())
    loop = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "isaac_loop")
    env = SimpleNamespace(
        sim_dt_=0.002, frame=0, break_out_signal_=False, closed=False,
        param_=SimpleNamespace(),
    )

    def step(**_kwargs):
        env.frame += 1
        env.break_out_signal_ = env.frame >= 25

    env.step_control_frame = step
    env.hold_current_pose = lambda: None
    env.sync_realtime = lambda: None
    env.close = lambda: setattr(env, "closed", True)
    published = []
    bus = SimpleNamespace(publish_obs=published.append, take_cmd=lambda: None)
    scope = {
        "POLICY_INTERVAL": 0.02,
        "open_env": lambda *_: (env, object(), 0.362),
        "refresh_contact": lambda *_: {"contact_frame": env.frame},
        "pack_state": lambda _env, seq, trial, contact: {
            "seq": seq, "trial": trial, "pose_frame": env.frame, **contact},
        "viewer_closed": lambda *_: False,
        "viewer_draw_stride": lambda *_: draw_stride,
        "apply_incoming": lambda *_: "ok",
    }
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(source), "exec"), scope)
    scope["isaac_loop"](bus, SimpleNamespace(trial_num=1), "mpc")
    observations = [obs for obs in published if "pose_frame" in obs]
    assert [obs["pose_frame"] for obs in observations] == [0, 10, 20]
    assert all(obs["pose_frame"] == obs["contact_frame"] for obs in observations)
    assert env.closed


def test_sync_bus_executes_one_interval_and_advances_trials_without_a_peer():
    source = Path(__file__).resolve().parents[1] / "examples/mpc/franka/ik2/run.py"
    tree = ast.parse(source.read_text())
    bus_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == "SynchronousMPCBus")
    environments = []

    def open_env(_args, _trial):
        env = SimpleNamespace(sim_dt_=0.002, frame=0, closed=False)
        env.hold_current_pose = lambda: None
        env.sync_realtime = lambda: None
        env.close = lambda: setattr(env, "closed", True)
        env.step_control_frame = lambda **_: setattr(env, "frame", env.frame+1)
        environments.append(env)
        return env, object(), 0.35

    scope = {
        "POLICY_INTERVAL": 0.02, "open_env": open_env,
        "refresh_contact": lambda env, *_: {"contact_frame": env.frame},
        "pack_state": lambda env, seq, trial, contact: {
            "seq": seq, "trial": trial, "pose_frame": env.frame, **contact},
        "apply_incoming": lambda *_: "ok", "viewer_closed": lambda *_: False,
        "viewer_draw_stride": lambda *_: 8,
    }
    exec(compile(ast.Module(body=[bus_class], type_ignores=[]), str(source), "exec"), scope)
    bus = scope["SynchronousMPCBus"](SimpleNamespace(trial_start=4, trial_num=2))
    assert bus.wait_obs()["pose_frame"] == 0
    for seq in [1, 2]:
        bus.publish_cmd({"kind": "via"})
        obs = bus.wait_obs()
        assert obs["seq"] == seq
        assert obs["pose_frame"] == obs["contact_frame"] == 10*seq
        assert bus.wait_obs() == obs
    bus.publish_cmd({"cmd": "end_trial"})
    assert environments[0].closed
    assert bus.wait_obs()["trial"] == 5
    bus.publish_cmd({"cmd": "end_trial"})
    assert bus.wait_obs()["cmd"] == "stop"
    assert all(env.closed for env in environments)


def test_episode_end_is_delivered_after_an_occupied_action_queue():
    import queue
    import threading
    from examples.mpc.franka.ik2.isaac_bus import IsaacBus

    attempted = threading.Event()

    class CommandQueue(queue.Queue):
        def put(self, item, block=True, timeout=None):
            if block:
                attempted.set()
            return super().put(item, block, timeout)

    commands = CommandQueue(maxsize=1)
    bus = IsaacBus(commands, queue.Queue())
    bus.publish_cmd({"kind": "via"})
    received = []

    def consume():
        if attempted.wait(timeout=1.0):
            received.append(commands.get(timeout=1.0))
            received.append(commands.get(timeout=1.0))

    peer = threading.Thread(target=consume, daemon=True)
    peer.start()
    try:
        assert bus.publish_cmd({"cmd": "end_trial"})
    finally:
        peer.join(timeout=2.0)
    assert received == [{"kind": "via"}, {"cmd": "end_trial"}]
