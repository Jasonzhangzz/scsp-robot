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


@pytest.mark.parametrize("mode", ["--mpc", "--mppi"])
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


def test_fa_adam_cli_default_matches_dexforge_and_remains_overridable():
    from planning.dexforge_imports import ensure_dexforge

    ensure_dexforge()
    from forceaware.config import load_config

    parser = build_argparser()
    args = parser.parse_args(["--mpc"])
    assert args.adam_lr == pytest.approx(load_config()["optimizer"]["lr_max"])
    assert args.adam_lr == pytest.approx(0.0015)
    custom = parser.parse_args(["--mpc", "--adam-lr", "0.004"])
    assert custom.adam_lr == pytest.approx(0.004)


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
