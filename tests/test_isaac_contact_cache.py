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


def test_osc_restores_object_pose_and_velocity_after_tensor_warmup():
    root = Path(__file__).resolve().parents[1] / 'examples/mpc/franka/ik2'
    base_source = root / 'test_mppi_isaac.py'
    base_tree = ast.parse(base_source.read_text())
    base = next(n for n in base_tree.body if isinstance(n, ast.ClassDef)
                and n.name == 'IsaacFrankaSimulator')
    helper = next(n for n in base.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_reset_object_pose')
    scope = {'np': np, 'gymapi': SimpleNamespace(STATE_ALL=0)}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(base_source), 'exec'), scope)
    states = dict(pose=dict(p=[None], r=[None]),
                  vel=dict(linear=[None], angular=[None]))
    expected = np.array([.4, .1, .39, .5, -.5, .5, -.5])

    class Base:
        _reset_object_pose = scope['_reset_object_pose']

        def reset_mj_env(self):
            # Tensor warmup advances an uncontrolled object.
            states['pose']['p'][0] = (.1, .1, .1)
            states['pose']['r'][0] = (0., 0., 0., 1.)
            states['vel']['linear'][0] = (1., 2., 3.)
            states['vel']['angular'][0] = (4., 5., 6.)

    source = root / 'test_mpc_isaac.py'
    tree = ast.parse(source.read_text())
    osc = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'IsaacFrankaOSCSimulator')
    reset = next(n for n in osc.body if isinstance(n, ast.FunctionDef) and n.name == 'reset_mj_env')
    osc.body = [reset]
    scope['IsaacFrankaSimulator'] = Base
    exec(compile(ast.Module(body=[osc], type_ignores=[]), str(source), 'exec'), scope)
    env = scope['IsaacFrankaOSCSimulator']()
    env.param_ = SimpleNamespace(init_obj_qpos_=expected)
    env.gym = SimpleNamespace(get_actor_rigid_body_states=lambda *_: states,
                             set_actor_rigid_body_states=lambda *_: None)
    env.env, env.obj_actor = object(), object()
    env._place_fingertip_for_rollout = lambda: None
    observed = []
    env._sync_desired_pose_with_current = lambda: observed.append(states['pose']['p'][0])
    env.get_current_joint_position = lambda: np.zeros(7)
    env._clear_mpc_action = lambda: None
    env.reset_mj_env()
    np.testing.assert_allclose(observed[0], expected[:3])
    np.testing.assert_allclose(states['pose']['r'][0], expected[[4,5,6,3]])
    np.testing.assert_allclose(states['vel']['linear'][0], 0.)
    np.testing.assert_allclose(states['vel']['angular'][0], 0.)


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
@pytest.mark.parametrize("startup_waits", [0, 5])
def test_async_observations_share_pose_contact_frame_and_physics_cadence(draw_stride, startup_waits):
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
    rendered_frames = []
    env.render_current_state = lambda: rendered_frames.append(env.frame)
    env.close = lambda: setattr(env, "closed", True)
    published = []
    commands = iter([None] * startup_waits + [{"kind": "via"}])
    bus = SimpleNamespace(publish_obs=published.append, take_cmd=lambda: next(commands, None))
    scope = {
        "POLICY_INTERVAL": 0.02,
        "open_env": lambda *_: (env, object(), 0.362),
        "refresh_contact": lambda *_: {"contact_frame": env.frame},
        "pack_state": lambda _env, seq, trial, contact: {
            "seq": seq, "trial": trial, "pose_frame": env.frame, **contact},
        "viewer_closed": lambda *_: False,
        "viewer_draw_stride": lambda *_: draw_stride,
        "apply_incoming": lambda *_: "ok",
        "time": SimpleNamespace(sleep=lambda _: None),
    }
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(source), "exec"), scope)
    scope["isaac_loop"](bus, SimpleNamespace(trial_num=1), "mpc")
    observations = [obs for obs in published if "pose_frame" in obs]
    assert [obs["pose_frame"] for obs in observations] == [0, 10, 20]
    assert all(obs["pose_frame"] == obs["contact_frame"] for obs in observations)
    assert rendered_frames == [0] * startup_waits
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


@pytest.mark.parametrize('gap,physical', [(.0005,False),(-.0005,True)])
@pytest.mark.parametrize('reverse', [False, True])
def test_contact_observation_uses_surface_gap_and_inward_object_jacobian(gap, physical, reverse):
    from scipy.spatial.transform import Rotation
    from examples.mpc.franka.ik2 import physx_contact as pc
    from examples.mpc.franka.ik2 import contact_frames as cf
    source = Path(__file__).resolve().parents[1]/'examples/mpc/franka/ik2/test_mpc_isaac.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ContactIsaacCartesian')
    ns = dict(np=np, Rotation=Rotation, **{k:getattr(pc,k) for k in
        ['end_effector_position','fingertip_radius','fingertip_body_index','contact_overlap','contact_impulse','physx_signed_gap','DEFAULT_CONTACT_OFFSET']})
    ns.update(_tangent_basis_from_normal=cf.tangent_basis_from_normal,
        _contact_jacobian=cf._contact_jacobian_np, _planar_table_jacobians=cf.planar_table_jacobians,
        contact_jacobian_angular_body_frame=cf.contact_jacobian_angular_body_frame)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),ns)
    p=SimpleNamespace(n_qvel_=9,max_ncon_=10,mu_object_=.5,table_height=.35,
        object_vertices_=np.array([[0,0,-.04],[0,0,.04]]))
    obj=np.array([.4,0,.39]); surface=obj+np.array([.04,0,0]); tip=surface+np.array([.01+gap,0,0])
    row=np.array((max(0.,-gap),0.),dtype=[('initial_overlap',float),('lambda',float)])
    normal=np.array([-1.,0,0]) if reverse else np.array([1.,0,0])
    skew=lambda x: np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    env=SimpleNamespace(param_=p, obj_body_idx=1,table_body_idx=2,tip_body_idx=3,env=object(),
        get_state=lambda:np.r_[obj,[1,0,0,0],tip], get_end_effector_pos=lambda:(tip,None),
        gym=SimpleNamespace(get_env_rigid_contacts=lambda _: [row]),
        _contact_bodies=lambda _: (3,1) if reverse else (1,3),
        _contact_sep_normal=lambda _: (gap,not physical,normal),
        _contact_world_pos=lambda *_:surface, _skew=skew)
    detector=ns['ContactIsaacCartesian'](p)
    phi,j,_,env_j,has_contact=detector.detect_once(env)
    assert has_contact is physical
    assert phi[0] == pytest.approx(gap,abs=1e-7)
    # Closing from the right (negative tip X) has a negative gap rate.
    assert np.mean(j[:4],axis=0)[6] == pytest.approx(1.)
    np.testing.assert_allclose(detector.get_actual_fingertip_contact()['normal_world'],[1,0,0])
    assert detector.env_phi[0] == pytest.approx(0.,abs=1e-7)
    assert phi[4] == pytest.approx(detector.env_phi[0])
    assert np.mean(env_j[:4],axis=0)[2] == pytest.approx(1.)
