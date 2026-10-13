"""Free-body lambda coordinates and actual FA control execution in MuJoCo."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

try:
    from examples.mpc.franka.bigrasp.bigrasp import BimanualPandaGrasper, build_argparser, resolve_control_timing
except ImportError as error:
    pytest.skip(f"BigRasp runtime dependencies unavailable: {error}", allow_module_level=True)

from models.comfree_gs_torch import GaussianCloud
from planning.lambda_contact_warp import LambdaContactWarp


@pytest.mark.parametrize("mode", ["--mppi"])
def test_ca_fa_defaults_use_stable_physics_without_shortening_control_knots(mode):
    parser = build_argparser()
    args = resolve_control_timing(parser.parse_args([mode]))
    assert args.mujoco_dt == .01
    assert args.adam_knot_substeps == 6
    assert args.mujoco_dt * args.adam_knot_substeps == pytest.approx(.06)
    override = resolve_control_timing(parser.parse_args([mode, "--mujoco-dt", ".002"]))
    assert override.adam_knot_substeps == 30
    explicit = resolve_control_timing(parser.parse_args([mode, "--mujoco-dt", ".002", "--adam-knot-substeps", "7"]))
    assert explicit.adam_knot_substeps == 7


def test_online_mpc_uses_short_horizon_without_changing_explicit_overrides():
    parser = build_argparser()
    online = resolve_control_timing(parser.parse_args(["--mpc"]))
    assert online.planner_horizon == 8
    assert online.planner_dt == .01
    assert online.online_approach_speed == .25
    assert resolve_control_timing(parser.parse_args(["--mppi"])).planner_horizon == 20
    assert resolve_control_timing(parser.parse_args(["--mpc", "--planner-horizon", "20"])).planner_horizon == 20


def test_fa_adam_cli_default_matches_dexforge_and_remains_overridable():
    from planning.dexforge_imports import ensure_dexforge

    ensure_dexforge()
    from forceaware.config import load_config

    parser = build_argparser()
    args = parser.parse_args(["--mppi"])
    assert args.adam_lr == pytest.approx(load_config()["optimizer"]["lr_max"])
    assert args.adam_lr == pytest.approx(0.0015)
    custom = parser.parse_args(["--mppi", "--adam-lr", "0.004"])
    assert custom.adam_lr == pytest.approx(0.004)


def test_online_contact_confirmation_rejects_wrong_patch_and_resets_on_pair_switch():
    grasper = BimanualPandaGrasper.__new__(BimanualPandaGrasper)
    points = np.array([[-.05, 0., 0.], [.05, 0., 0.]])
    grasper.args = SimpleNamespace(min_normal_force=.1, online_contact_tolerance=.005,
                                  online_contact_confirm_steps=2)
    grasper._online_contact_stable_steps = 0
    grasper._online_contact_ids = None
    grasper.optimizer = SimpleNamespace(last_grasp_result={"contact_indices": np.array([1, 2])})
    resets = []
    grasper.planner = SimpleNamespace(reset=lambda: resets.append(True))
    contacts = {"left": [dict(normal_force=1., local_pos=points[0])],
                "right": [dict(normal_force=1., local_pos=points[1] + [0., .02, 0.])]}
    grasper.extract_object_contacts = lambda: contacts
    mask, established = grasper._online_selected_contacts(points)
    np.testing.assert_array_equal(mask, [True, False])
    assert not established
    contacts["right"][0]["local_pos"] = points[1]
    assert not grasper._online_selected_contacts(points)[1]
    assert grasper._online_selected_contacts(points)[1]
    grasper.optimizer.last_grasp_result["contact_indices"] = np.array([1, 3])
    assert not grasper._online_selected_contacts(points)[1]
    assert len(resets) == 2


def test_physical_grasp_latch_survives_off_anchor_contacts_and_short_force_dips():
    grasper = BimanualPandaGrasper.__new__(BimanualPandaGrasper)
    points = np.array([[-.05, 0., 0.], [.05, 0., 0.]])
    grasper.args = SimpleNamespace(min_normal_force=None, obj_mass=.01, online_contact_tolerance=.015,
        online_contact_confirm_steps=3, online_grasp_release_steps=3)
    contacts = {"left": [dict(normal_force=.19, local_pos=points[0])],
                "right": [dict(normal_force=.19, local_pos=points[1] + [0., .0267, 0.])]}
    grasper.extract_object_contacts = lambda: contacts
    np.testing.assert_array_equal(grasper._online_measured_contact_mask(points), [True, False])
    assert not grasper._online_grasp_observation()[2]
    assert not grasper._online_grasp_observation()[2]
    assert grasper._online_grasp_observation()[2]
    contacts["right"][0]["normal_force"] = .02
    assert grasper._online_grasp_observation()[2]  # below entry, above exit threshold
    contacts["right"] = []
    assert grasper._online_grasp_observation()[2]
    assert grasper._online_grasp_observation()[2]
    assert not grasper._online_grasp_observation()[2]


def test_online_observation_keeps_body_angular_velocity_and_both_pad_velocities():
    grasper, _ = _grasper()
    grasper.planner_mode = "mpc_ee"
    grasper.data.qvel[:6] = np.arange(6) * .1
    grasper.get_planner_state_tips = lambda: np.r_[np.zeros(3), 1., np.zeros(9)]
    jac = np.zeros((6, 7))
    jac[:3, :3] = np.eye(3)
    grasper.get_arm_jacobian = lambda arm: jac
    grasper.get_current_joint_velocity = lambda arm: np.r_[[1., 2., 3.] if arm is grasper.left_arm else [-1., -2., -3.], np.zeros(4)]
    state = grasper.get_planner_state()
    assert state.shape == (25,)
    np.testing.assert_array_equal(state[13:19], grasper.data.qvel[:6])
    np.testing.assert_array_equal(state[19:25], [1., 2., 3., -1., -2., -3.])


@pytest.mark.parametrize("mode", ["mpc_ee", "mppi_ee"])
def test_online_contact_switching_is_frozen_by_default_and_joint_verify_is_explicit(mode):
    parser = build_argparser()
    assert parser.parse_args([]).freeze_contact_pair
    assert parser.parse_args(["--mpc"]).freeze_contact_pair
    assert not parser.parse_args(["--mpc", "--allow-contact-switch"]).freeze_contact_pair
    grasper, rotation = _grasper()
    grasper.planner_mode = mode
    grasper.args = SimpleNamespace(planner_attract_offset=.012, freeze_contact_pair=True)
    grasper.plan_params = SimpleNamespace(h_=.01)
    grasper.planner = SimpleNamespace(knot_substeps=6)
    grasper.contact_points_local = np.zeros((2, 3))
    grasper.object_target_pos = np.array([0., 0., 1.1])
    grasper.object_target_quat = np.array([1., 0., 0., 0.])
    grasper.get_tip_pose = lambda arm: (np.zeros(3), np.eye(3))
    grasper._online_measured_contact_mask = lambda points: np.array([False, False])
    calls = []

    def solve(target, current, **kwargs):
        calls.append(kwargs)
        return (np.zeros((2, 3)), np.tile([1., 0., 0.], (2, 1)), current, 0., {"source": "fixed_pair"})

    grasper.optimizer = SimpleNamespace(update_contacts=solve, last_grasp_result={})
    grasper._build_contact_targets_from_local_contacts = lambda *args, **kwargs: {}
    pose = grasper.data.qpos[grasper.obj_qpos_adr:grasper.obj_qpos_adr + 7].copy()
    grasper._update_warp_lambda_contacts(pose[:3], pose[3:], rotation)
    assert calls[-1]["freeze_contact_pair"] and calls[-1]["joint_verify"]
    assert not calls[-1].get("lock_contact_pair", False)
    grasper.args.freeze_contact_pair = False
    grasper._update_warp_lambda_contacts(pose[:3], pose[3:], rotation)
    assert not calls[-1]["freeze_contact_pair"] and calls[-1]["joint_verify"]


def test_online_wrist_alignment_does_not_accelerate_the_position_task():
    from tests.test_dexforge_retarget import _fast_step, _contact_inputs

    step = _fast_step()
    grasper = BimanualPandaGrasper.__new__(BimanualPandaGrasper)
    grasper.model, grasper.data = step.cpu_model, mujoco.MjData(step.cpu_model)
    qpos, _, _, _ = _contact_inputs(step)
    grasper.data.qpos[:] = qpos
    mujoco.mj_forward(grasper.model, grasper.data)
    ids = step._ids
    joints = [mujoco.mj_name2id(grasper.model, mujoco.mjtObj.mjOBJ_JOINT, f"left_joint{i}") for i in range(1, 8)]
    arm = SimpleNamespace(qpos_adr=ids["left_qpos"], dof_adr=grasper.model.jnt_dofadr[joints],
        home_q=qpos[ids["left_qpos"]].copy(), nullspace_stiffness=10., torque_limits=np.full(7, 1e6))
    site = mujoco.mj_name2id(grasper.model, mujoco.mjtObj.mjOBJ_SITE, "left_tip_center")
    jp, jr = np.zeros((3, grasper.model.nv)), np.zeros((3, grasper.model.nv))
    mujoco.mj_jacSite(grasper.model, grasper.data, jp, jr, site)
    jac = np.vstack((jp[:, arm.dof_adr], jr[:, arm.dof_adr]))
    grasper.get_arm_jacobian = lambda arm: jac
    grasper.get_tip_pose = lambda arm: (grasper.data.site_xpos[site], grasper.data.site_xmat[site].reshape(3, 3))
    grasper.args = SimpleNamespace(online_orientation_stiffness=15.)
    torque = grasper._compute_arm_osc_torque(arm, np.zeros(3), np.zeros(3),
                                            approach_direction=np.array([1., 0., 0.]))
    mass = _full_mass(grasper.model, grasper.data)[np.ix_(arm.dof_adr, arm.dof_adr)]
    assert np.linalg.norm(torque) > 1e-4
    np.testing.assert_allclose(jac[:3] @ np.linalg.solve(mass, torque), 0., atol=1e-9)


def _grasper():
    # The free-body origin deliberately differs from its COM, as it does for
    # BigRasp's mesh objects. Independent scalar joints exercise 14-D controls.
    arms = "".join(
        f'<body name="arm{i}" pos="{i * 0.1} 0 1">'
        f'<joint name="joint{i}" axis="0 1 0" damping="1"/>'
        '<geom type="sphere" size="0.02" pos="0.04 0 0" mass="1" contype="0" conaffinity="0"/>'
        '</body>' for i in range(14)
    )
    actuators = "".join(
        f'<position name="act{i}" joint="joint{i}" kp="5" ctrllimited="true" ctrlrange="-1 1"/>'
        for i in range(14)
    )
    xml = (
        '<mujoco><compiler angle="radian"/><option timestep="0.001" gravity="0 0 -9.81"/>'
        '<worldbody><body name="obj" pos="0 0 1">'
        '<freejoint name="obj_freejoint"/>'
        '<inertial pos="0.03 -0.02 0.01" mass="0.2" diaginertia="0.001 0.002 0.003"/>'
        '<geom type="sphere" size="0.04" contype="0" conaffinity="0"/>'
        f'</body>{arms}</worldbody><actuator>{actuators}</actuator></mujoco>'
    )
    grasper = BimanualPandaGrasper.__new__(BimanualPandaGrasper)
    grasper.model = mujoco.MjModel.from_xml_string(xml)
    grasper.data = mujoco.MjData(grasper.model)
    grasper.obj_body_id = mujoco.mj_name2id(grasper.model, mujoco.mjtObj.mjOBJ_BODY, "obj")
    joint = mujoco.mj_name2id(grasper.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")
    grasper.obj_qpos_adr = int(grasper.model.jnt_qposadr[joint])
    grasper.obj_dof_adr = int(grasper.model.jnt_dofadr[joint])
    grasper.left_arm = SimpleNamespace(actuator_ids=np.arange(7))
    grasper.right_arm = SimpleNamespace(actuator_ids=np.arange(7, 14))
    rotation = Rotation.from_euler("xyz", [20.0, 30.0, -40.0], degrees=True)
    start = grasper.obj_qpos_adr
    grasper.data.qpos[start + 3:start + 7] = rotation.as_quat()[[3, 0, 1, 2]]
    mujoco.mj_forward(grasper.model, grasper.data)
    return grasper, rotation.as_matrix()


def _full_mass(model, data):
    result = np.empty((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, result)
    except TypeError:
        mujoco.mj_fullM(model, result, data.qM)
    return result


def test_online_osc_receives_free_space_reference_and_disables_it_on_contact():
    grasper, _ = _grasper()
    grasper.left_arm.torque_limits = grasper.right_arm.torque_limits = np.full(7, 100.)
    grasper.args = SimpleNamespace(online_force_limit=12., online_approach_speed=.25,
        online_task_stiffness=600., online_task_damping=40., online_unilateral_force_limit=.05)
    grasper.normals_local = np.array([[1., 0., 0.], [-1., 0., 0.]])
    grasper._planner_tracking_steps = lambda: 20  # 20 ms with this fixture's 1 ms physics
    grasper.get_arm_jacobian = lambda arm: np.zeros((6, 7))
    grasper.get_current_joint_velocity = lambda arm: np.zeros(7)
    grasper._compute_preferred_tip_approach = lambda normal: normal
    calls = []
    grasper._compute_arm_osc_torque = lambda arm, increment, velocity, **kwargs: (calls.append(kwargs) or np.zeros(7))
    gap, rows = np.ones(24), np.zeros((24, 12))
    gap[:4] = 0.  # left touching, right free
    rows[:4, 0], rows[4:8, 0] = 1., -1.
    grasper._online_mpc_contacts = lambda: (gap, rows)
    grasper.step_online_mpc_action([.006, 0., 0., -.006, 0., 0.])
    np.testing.assert_array_equal(calls[0]["v_ref"], np.zeros(3))
    assert calls[0]["force_limit"] == .05
    np.testing.assert_allclose(calls[1]["v_ref"], [-.25, 0., 0.])
    grasper._online_grasp_active = True
    calls.clear()
    grasper.step_online_mpc_action([.006, 0., 0., -.006, 0., 0.])
    for call in calls:
        np.testing.assert_array_equal(call["v_ref"], np.zeros(3))
        assert call["force_limit"] == 12.


def test_offcenter_object_mass_and_gravity_wrench_match_freejoint_coordinates():
    grasper, rotation = _grasper()
    mass, external_world = grasper._lambda_object_dynamics(rotation)
    start = grasper.obj_dof_adr
    expected_mass = _full_mass(grasper.model, grasper.data)[start:start + 6, start:start + 6]
    np.testing.assert_allclose(mass, expected_mass, atol=1.0e-14)
    assert np.linalg.norm(mass[:3, 3:]) > 1.0e-4
    force = grasper.model.body_mass[grasper.obj_body_id] * grasper.model.opt.gravity
    com_world = rotation @ grasper.model.body_ipos[grasper.obj_body_id]
    np.testing.assert_allclose(external_world[:3], force, atol=1.0e-12)
    np.testing.assert_allclose(external_world[3:], np.cross(com_world, force), atol=1.0e-12)


def test_lambda_external_wrench_reproduces_mujoco_acceleration_with_body_velocity():
    grasper, rotation = _grasper()
    start = grasper.obj_dof_adr
    velocity = np.array([0.2, -0.1, 0.3, 1.1, -0.7, 0.5])
    grasper.data.qvel[start:start + 6] = velocity
    grasper.data.qfrc_applied[start:start + 6] = [2.0, -1.0, 0.5, 0.01, 0.02, -0.03]
    mujoco.mj_forward(grasper.model, grasper.data)
    mass, external_world = grasper._lambda_object_dynamics(rotation)
    external_generalized = external_world.copy()
    external_generalized[3:] = rotation.T @ external_world[3:]
    acceleration = np.linalg.solve(mass, external_generalized)
    np.testing.assert_allclose(acceleration, grasper.data.qacc[start:start + 6], atol=1.0e-11)
    optimizer = LambdaContactWarp(
        cloud=GaussianCloud([[0.0, 0.0, 0.0]], radius=0.04),
        obj_mass=0.2, max_contact_force=1.0e-12, device="cpu",
    )
    points = np.array([[0.04, 0, 0], [-0.04, 0, 0]])
    optimizer.last_grasp_result = {
        "contact_points": points,
        "contact_normals": np.array([[-1.0, 0, 0], [1.0, 0, 0]]),
    }
    qstart = grasper.obj_qpos_adr
    pose = grasper.data.qpos[qstart:qstart + 7].copy()
    dt = 0.003
    _, _, predicted, _, info = optimizer.optimize_control_input(
        pose, pose, tau_o=external_world, p_arm=points, v_obj=velocity,
        r_obj_to_world=rotation, object_inertia=mass, prediction_dt=dt,
    )
    np.testing.assert_allclose(info["velocity_next"], velocity + dt * acceleration, atol=2.0e-10)
    np.testing.assert_allclose(predicted[:3], pose[:3] + dt * info["velocity_next"][:3], atol=1.0e-12)


def test_fa_executes_each_absolute_control_at_one_physics_step():
    grasper, _ = _grasper()
    grasper.data.qpos[7:] = np.linspace(-0.3, 0.3, 14)
    grasper.data.qvel[6:] = np.linspace(-0.1, 0.1, 14)
    mujoco.mj_forward(grasper.model, grasper.data)
    expected = mujoco.MjData(grasper.model)
    expected.qpos[:] = grasper.data.qpos
    expected.qvel[:] = grasper.data.qvel
    controls = np.vstack((np.linspace(-0.2, 0.2, 14),
                          np.linspace(0.1, -0.1, 14),
                          np.linspace(-1.5, 1.5, 14)))
    for control in controls:
        expected.ctrl[:] = np.clip(control, -1.0, 1.0)
        mujoco.mj_step(grasper.model, expected)
    mujoco.mj_forward(grasper.model, expected)
    grasper.step_joint_controls(controls)
    assert grasper.data.time == pytest.approx(3 * grasper.model.opt.timestep)
    np.testing.assert_array_equal(grasper.data.ctrl, np.clip(controls[-1], -1.0, 1.0))
    np.testing.assert_allclose(grasper.data.qpos, expected.qpos, atol=1.0e-12)
    np.testing.assert_allclose(grasper.data.qvel, expected.qvel, atol=1.0e-12)


def test_live_lambda_uses_measured_dynamics_at_fa_interval_and_preserves_arm_assignment():
    grasper, rotation = _grasper()
    grasper.args = SimpleNamespace(planner_attract_offset=.012)
    grasper.planner = SimpleNamespace(knot_substeps=4)
    grasper.object_target_pos = np.array([.2, -.1, 1.1])
    grasper.object_target_quat = np.array([1., 0., 0., 0.])
    left_tip, right_tip = np.array([.25, 0., 1.]), np.array([-.25, 0., 1.])
    grasper.get_tip_pose = lambda arm: ((left_tip if arm is grasper.left_arm else right_tip).copy(), np.eye(3))
    # These deliberate left/right assignments differ from sorting object X.
    points = np.array([[.08, .01, 0.], [-.07, -.02, 0.]])
    normals = np.array([[-1., 0., 0.], [1., 0., 0.]])
    coefficients = np.array([[2., .1, .2], [3., .3, .4]])
    vectors = np.array([[-2., .2, -.1], [3., -.4, .3]])
    calls = []

    def solve(mode, target, current, **kwargs):
        calls.append((mode, target.copy(), current.copy(),
                      {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in kwargs.items()}))
        predicted = current.copy()
        predicted[2] += .02
        grasper.optimizer.last_grasp_result = dict(contact_indices=np.array([17, 9]),
            witness_contact_forces_local=coefficients.copy(), witness_force_vectors_local=vectors.copy(),
            total_cost=.001)
        return points.copy(), normals.copy(), predicted, .001, {"source": mode}

    grasper.optimizer = SimpleNamespace(last_grasp_result=None,
        initialize_offline=lambda *args, **kwargs: solve("offline_global", *args, **kwargs),
        update_contacts=lambda *args, **kwargs: solve("online_cones", *args, **kwargs))
    grasper._ordered_contact_data = lambda *args, **kwargs: pytest.fail("Live lambda assignment must be preserved")
    qstart, dstart = grasper.obj_qpos_adr, grasper.obj_dof_adr
    current = grasper.data.qpos[qstart:qstart + 7].copy()
    first = grasper._update_warp_lambda_contacts(current[:3], current[3:], rotation, initial=True)
    grasper.data.qvel[dstart:dstart + 6] = [.1, -.2, .3, .4, -.5, .6]
    grasper.data.qpos[qstart:qstart + 3] += [.01, -.02, .03]
    mujoco.mj_forward(grasper.model, grasper.data)
    current = grasper.data.qpos[qstart:qstart + 7].copy()
    targets, predicted, cost, info = grasper._update_warp_lambda_contacts(current[:3], current[3:], rotation)
    assert [call[0] for call in calls] == ["offline_global", "online_cones"]
    np.testing.assert_array_equal(calls[1][1], np.r_[grasper.object_target_pos, grasper.object_target_quat])
    np.testing.assert_array_equal(calls[1][2], current)
    np.testing.assert_array_equal(calls[1][3]["tips_world"], [left_tip, right_tip])
    np.testing.assert_array_equal(calls[1][3]["v_obj"], grasper.data.qvel[dstart:dstart + 6])
    assert calls[1][3]["prediction_dt"] == pytest.approx(4 * grasper.model.opt.timestep)
    mass, external = grasper._lambda_object_dynamics(rotation)
    np.testing.assert_array_equal(calls[1][3]["object_inertia"], mass)
    np.testing.assert_array_equal(calls[1][3]["tau_o"], external)
    np.testing.assert_array_equal(targets["contact_points_local"], points)
    np.testing.assert_array_equal(targets["normals_local"], normals)
    np.testing.assert_array_equal(targets["contact_indices"], [17, 9])
    np.testing.assert_array_equal(targets["witness_contact_forces_local"], coefficients)
    np.testing.assert_array_equal(targets["witness_force_vectors_local"], vectors)
    np.testing.assert_allclose(targets["contact_points_world"], points @ rotation.T + current[:3])
    np.testing.assert_allclose(targets["virtual_points_world"],
                               targets["contact_points_world"] - .012 * normals @ rotation.T)
    assert info["source"] == "online_cones" and cost == pytest.approx(.001)
    assert predicted[2] == pytest.approx(current[2] + .02)
    assert first[3]["source"] == "offline_global"
