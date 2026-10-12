"""Ramp collision/planner regressions without the Isaac native runtime."""
import argparse
import ast
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace

import casadi as cs
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from examples.mpc.franka.ik2 import contact_frames as cf, physx_contact as pc
from examples.mpc.franka.test_tilted_push.params import ExplicitMPCParams
from examples.mpc.franka.test_tilted_push.ramp_support import (
    prepare_object_geometry, configure_support_objective, fit_mesh_center_on_ramp,
    repo_asset_path,
)
from models.contact_geometry import (
    contact_geometry_cs, contact_geometry_torch, refresh_table_support,
)
from models.explicit_model import ExplicitModel
from models.support_geometry import mesh_support_points, support_jacobians
from planning.mpc_explicit import MPCExplicit
from planning.object_pose_cost import object_cost_np
from utils import rotations

ROOT = Path(__file__).resolve().parents[1]


def extract(relative, names, scope):
    source = ROOT / relative
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)


@pytest.fixture
def ramp_scope():
    scope = dict(np=np, os=os, re=re, Rotation=Rotation, rotations=rotations,
                 mesh_support_points=mesh_support_points, support_jacobians=support_jacobians,
                 fit_mesh_center_on_ramp=fit_mesh_center_on_ramp, repo_asset_path=repo_asset_path)
    scope.update({k: getattr(pc, k) for k in [
        'end_effector_position', 'fingertip_radius', 'fingertip_body_index',
        'contact_overlap', 'contact_impulse', 'physx_signed_gap', 'DEFAULT_CONTACT_OFFSET']})
    scope.update(_tangent_basis_from_normal=cf.tangent_basis_from_normal,
        _contact_jacobian=cf._contact_jacobian_np,
        contact_jacobian_angular_body_frame=cf.contact_jacobian_angular_body_frame)
    extract('examples/mpc/franka/ik2/test_mpc_isaac.py', {'ContactIsaacCartesian'}, scope)
    source = ROOT / 'examples/mpc/franka/test_tilted_push/test_tiltedpush.py'
    tree = ast.parse(source.read_text())
    constants = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets)]
    # Constants imported from ik2 are used only by unrelated declarations.
    scope.update(CONTACT_NEAR_PRESS=.012, ROLLOUT_FINGERTIP_FRICTION=1.5)
    exec(compile(ast.Module(body=constants, type_ignores=[]), str(source), 'exec'), scope)
    names = {'_normalize_quaternion_wxyz_np', '_quat_xyzw_to_wxyz_np',
        '_quat_wxyz_to_xyzw_np', '_compose_quat_wxyz', '_quat_from_local_z_rotation_wxyz',
        '_quat_from_world_z_rotation_wxyz', '_compute_tilted_ramp_config',
        '_support_plane', '_support_signed_distance', '_clip_tilted_ramp_local_xy',
        '_place_pose_on_tilted_ramp_local', '_project_world_offset_to_tilted_ramp_local_xy',
        '_build_goal_pose_on_tilted_ramp', '_apply_tilted_ramp_scene_to_param',
        '_apply_dywa_physics_to_param', '_allocate_trial_record_dir', 'ContactIsaacRamp'}
    extract('examples/mpc/franka/test_tilted_push/test_tiltedpush.py', names, scope)
    extract('examples/mpc/franka/ik2/params.py', {'_box_inertia_diag'}, scope)
    return scope


@pytest.fixture
def configured(ramp_scope, tmp_path):
    scope = {}
    extract('examples/mpc/fingertips/test/test_0902.py', {'add_rollout_via_args'}, scope)
    parser = argparse.ArgumentParser()
    scope['add_rollout_via_args'](parser)
    args = parser.parse_args([])
    args.obj, args.seed, args.solver = 'foam_brick', 15, 'ipopt'
    param = ExplicitMPCParams(args)
    cube = tmp_path / 'cube.stl'
    trimesh.creation.box(extents=[.08, .08, .08]).export(cube)
    param.source_mesh_path_ = str(cube)
    param.obj_mesh_scale_ = np.ones(3)
    prepare_object_geometry(param)
    ramp_scope['_apply_dywa_physics_to_param'](param)
    ramp_scope['_apply_tilted_ramp_scene_to_param'](param, True, True,
        [-.18, 0.], 0., [0., -.1, .1], True)
    configure_support_objective(param, args)
    param.robot_stiff_ = 600*np.eye(3)
    param.Q[6:, 6:] = param.robot_stiff_
    param.planner_solver_ = 'ipopt'
    param.contact_force_limit_ = 10.
    param.contact_command_force_limit_ = 3.
    return param, args


def cube_on_ramp(param, *, tilt=0., offset=(0., 0.)):
    vertices = np.array(np.meshgrid(*([[-.04, .04]]*3))).T.reshape(-1, 3)
    hull = trimesh.Trimesh(vertices=vertices).convex_hull
    param.support_vertices_, param.support_edges_ = hull.vertices, hull.edges_unique
    param.object_vertices_ = hull.vertices
    basis = np.vstack([param.support_surface_tangents_, param.support_surface_normal_])
    ramp_rotation = Rotation.from_matrix(basis.T)
    rotation = ramp_rotation * Rotation.from_euler('x', tilt, degrees=True)
    normal = param.support_surface_normal_
    height = -np.min((hull.vertices @ rotation.as_matrix().T) @ normal)
    position = (param.support_surface_point_ + np.asarray(offset) @ param.support_surface_tangents_
                + height*normal)
    pose = np.r_[position, rotation.as_quat()[[3, 0, 1, 2]]]
    return np.r_[pose, position + [-.1, 0., .1]]


def detect(scope, param, state):
    simulator = SimpleNamespace(param_=param, table_body_idx=2, obj_body_idx=1,
        tip_body_idx=3, env=None, get_state=lambda: state,
        get_end_effector_pos=lambda: (state[7:10], None), _skew=cf._skew,
        gym=SimpleNamespace(get_env_rigid_contacts=lambda _: []))
    detector = scope['ContactIsaacRamp'](param)
    result = detector.detect_once(simulator)
    assert simulator.table_body_idx == 2
    assert param.table_height == .35
    return detector, result


def test_lambda_and_mpc_use_same_contact_interval_force_units_and_material(configured):
    from models.soft_scm_contact import SOFT_SCM_RESPONSE_TIME, SOFT_SCM_DAMPING_RATIO, SOFT_SCM_REGULARIZATION
    from planning.contact_limits import contact_force_limit, contact_command_force_limit
    from planning.runtime_compat import resolve_solver_backend
    param, args = configured
    scope = dict(np=np, LambdaContactControlOptimizer=lambda **kw: SimpleNamespace(**kw),
        SOFT_SCM_RESPONSE_TIME=SOFT_SCM_RESPONSE_TIME, SOFT_SCM_DAMPING_RATIO=SOFT_SCM_DAMPING_RATIO,
        SOFT_SCM_REGULARIZATION=SOFT_SCM_REGULARIZATION, resolve_solver_backend=resolve_solver_backend,
        contact_force_limit=contact_force_limit, contact_command_force_limit=contact_command_force_limit)
    extract('examples/mpc/franka/ik2/params.py', {'build_lambda_optimizer'}, scope)
    assert param.lambda_optimizer is None
    opt = scope['build_lambda_optimizer'](param, args)
    assert opt.time_step == param.h_ == .02
    assert opt.wrench_is_force and opt.contact_dynamics == param.contact_dynamics == 'quasi-dynamic'
    assert opt.grouped_contact_response and opt.robot_stiffness == param.robot_stiff_[0, 0] == 600.
    assert opt.arm_friction == param.contact_friction_ == 1.2
    assert param.support_friction_ == .7
    model = ExplicitModel(param)
    state = param.init_obj_qpos_.tolist() + [.3, 0., .6]
    output = np.asarray(model.prediction_fn(state, [.005, 0., 0.], np.ones(40), np.zeros((40, 9)), 7.)).ravel()
    np.testing.assert_allclose(output[7:10]-state[7:10], [.005, 0., 0.], atol=1e-12)


@pytest.mark.parametrize('tilt,points', [(0., 4), (30., 2)])
def test_mesh_support_gap_and_moment_arms_agree_in_both_frames(configured, ramp_scope, tilt, points):
    param, _ = configured
    state = cube_on_ramp(param, tilt=tilt)
    detector, (phi, jac, patches, env_j, _) = detect(ramp_scope, param, state)
    assert len(patches) == points
    np.testing.assert_allclose(phi[:4*points], 0., atol=5e-8)
    np.testing.assert_allclose(detector.env_phi[:4*points], phi[:4*points], atol=1e-12)
    rotation = Rotation.from_quat(state[[4, 5, 6, 3]]).as_matrix()
    np.testing.assert_allclose(jac[:4*points, :3] @ rotation, env_j[:4*points, :3], atol=1e-7)
    np.testing.assert_allclose(jac[:4*points, 3:6], env_j[:4*points, 3:6], atol=1e-7)
    assert np.linalg.norm(jac[:4*points, 3:6]) > .01
    np.testing.assert_allclose(param.gravity_[:3], [0., 0., -9.81])
    wrench = cf.gravity_wrench_object_frame(param.gravity_[:3], .01, rotation, param.support_surface_normal_)
    np.testing.assert_allclose(rotation @ wrench[:3], [0., 0., -.0981], atol=1e-12)


def test_rim_clips_partial_support_and_removes_support_after_departure(configured, ramp_scope):
    param, _ = configured
    state = cube_on_ramp(param, offset=(.32, 0.))
    _, (_, _, patches, _, _) = detect(ramp_scope, param, state)
    rotation = Rotation.from_quat(state[[4, 5, 6, 3]]).as_matrix()
    world = np.asarray(patches) @ rotation.T + state[:3]
    uv = (world-param.support_surface_point_) @ param.support_surface_tangents_.T
    assert len(patches) >= 2
    assert np.max(uv[:, 0]) == pytest.approx(.30)
    state = cube_on_ramp(param, offset=(.35, 0.))
    detector, (phi, jac, patches, env_j, _) = detect(ramp_scope, param, state)
    assert not len(patches)
    np.testing.assert_array_equal(jac, 0.)
    np.testing.assert_array_equal(env_j, 0.)
    np.testing.assert_array_equal(phi, 1.)
    np.testing.assert_array_equal(detector.env_phi, 1.)


def test_horizon_support_remains_tilted_and_refreshes_real_pivot(configured):
    param, _ = configured
    state = cube_on_ramp(param, tilt=30.)
    phi, jac = refresh_table_support(param.support_vertices_, state,
        np.ones(40), np.zeros((40, 9)), .35, param.support_friction_,
        support_point=param.support_surface_point_, support_normal=param.support_surface_normal_,
        support_tangents=param.support_surface_tangents_, support_half_extents=param.support_surface_half_extents_,
        support_edges=param.support_edges_)
    np.testing.assert_allclose(jac[:4, :3].mean(axis=0), param.support_surface_normal_, atol=1e-12)
    assert phi[0] == pytest.approx(0., abs=5e-8)
    assert np.linalg.norm(jac[:4, 3:6]) > .01


def test_predicted_departure_disables_support_in_casadi_and_torch(configured, ramp_scope):
    torch = pytest.importorskip('torch')
    param, _ = configured
    ref = cube_on_ramp(param, offset=(.24, 0.))
    _, (phi, jac, _, _, _) = detect(ramp_scope, param, ref)
    bounds = dict(support_point=param.support_surface_point_,
        support_tangents=param.support_surface_tangents_, support_half_extents=param.support_surface_half_extents_)
    x = cs.SX.sym('x', 10)
    outputs = contact_geometry_cs(x, cs.DM(ref), cs.DM(phi), cs.DM(jac), **bounds)
    fn = cs.Function('finite_support', [x], list(outputs))
    for distance in (0., .12):
        state = ref.copy()
        state[:3] += distance*param.support_surface_tangents_[0]
        cg, cj = [np.asarray(v) for v in fn(state)]
        tx = torch.tensor(state, dtype=torch.float64, requires_grad=True)
        tg, tj = contact_geometry_torch(tx, *[torch.tensor(v, dtype=torch.float64)
            for v in (ref, phi, jac)], **bounds)
        np.testing.assert_allclose(tg.detach(), cg.ravel(), atol=1e-8)
        np.testing.assert_allclose(tj.detach(), cj, atol=1e-8)
        (tg.sum()+tj.sum()).backward()
        assert torch.isfinite(tx.grad).all()
        if distance:
            np.testing.assert_array_equal(cj, 0.)


class InspectMPC(MPCExplicit):
    def init_MPC(self):
        pass


def test_runtime_model_carries_velocity_and_prices_tipping_and_rim(configured, ramp_scope):
    param, _ = configured
    state = cube_on_ramp(param)
    mpc = InspectMPC(param)
    assert mpc.velocity_state and mpc.update_contact_gaps
    assert mpc.contact_force_limit == 3.
    x = mpc.observation_state(state, dict(object_linear_velocity=[0., .1, 0.]))
    assert x.shape == (19,) and x[11] == pytest.approx(.1)
    predicted = np.asarray(mpc.stage_prediction_fn(x, np.zeros(3), np.ones(40),
        np.zeros((40, 9)), 7., x)).ravel()
    assert predicted[12] == pytest.approx(-.02*9.81)
    assert predicted[2] < state[2]
    cost = mpc._cost_params(target_p=state[:3], target_q=state[3:7],
        virtual_point=state[7:10], contact_point=state[7:10], contact_reference=state[:7])
    original = float(mpc.final_cost_fn(state, cost))
    tilted = state.copy()
    rotation = Rotation.from_quat(state[[4, 5, 6, 3]]) * Rotation.from_euler('x', 35., degrees=True)
    tilted[3:7] = rotation.as_quat()[[3, 0, 1, 2]]
    assert float(mpc.final_cost_fn(tilted, cost)) > original + 10.
    # Changing goal yaw in position-only mode leaves the complete cost unchanged.
    other = cost.copy(); other[3:7] = [1., 0., 0., 0.]
    assert float(mpc.final_cost_fn(tilted, other)) == pytest.approx(float(mpc.final_cost_fn(tilted, cost)))
    rim = state.copy(); rim[:3] += .28*param.support_surface_tangents_[0]
    rim_cost = mpc._cost_params(target_p=rim[:3], target_q=rim[3:7],
        virtual_point=rim[7:10], contact_point=rim[7:10], contact_reference=rim[:7])
    assert float(mpc.final_cost_fn(rim, rim_cost)) > original + 10.
    assert object_cost_np(tilted, state, param.object_cost_config_) > 1.


def test_scaled_geometry_matches_collision_mesh_and_physical_inertia(configured, ramp_scope):
    param, _ = configured
    param.obj_mesh_scale_ = np.array([2., .5, 1.])
    prepare_object_geometry(param)
    ramp_scope['_apply_dywa_physics_to_param'](param)
    actual = trimesh.load_mesh(param.mesh_path_)
    np.testing.assert_allclose(actual.bounds, [param.object_aabb_lo, param.object_aabb_hi], atol=1e-8)
    np.testing.assert_allclose(param.obj_mesh_scale_, 1.)
    np.testing.assert_allclose(np.ptp(param.support_vertices_, axis=0), actual.extents, atol=1e-8)
    expected = param.sim_obj_mass_/12 * np.array([
        actual.extents[1]**2+actual.extents[2]**2,
        actual.extents[0]**2+actual.extents[2]**2,
        actual.extents[0]**2+actual.extents[1]**2])
    np.testing.assert_allclose(param.sim_obj_inertia_diag_, expected, rtol=1e-6)


@pytest.mark.parametrize('launch_directory', ['paper', 'external'])
def test_repo_assets_load_from_other_launch_directories(configured, tmp_path, monkeypatch, launch_directory):
    _, args = configured
    args = SimpleNamespace(**vars(args))
    args.obj = 'elephant'
    monkeypatch.chdir(ROOT / 'tro_revision' if launch_directory == 'paper' else tmp_path)
    param = ExplicitMPCParams(args)
    source = ROOT / 'envs/assets/objects/elephant.stl'
    assert Path(param.source_mesh_path_) == source
    assert Path(param.model_path_) == ROOT / 'envs/xmls/env_fingertips_elephant.xml'
    np.testing.assert_allclose(param.object_aabb_lo, trimesh.load_mesh(str(source)).bounds[0])
    # Also cover callers that supply an explicit repository-relative source.
    param.source_mesh_path_ = 'envs/assets/objects/elephant.stl'
    param.obj_mesh_scale_ = np.ones(3)
    prepare_object_geometry(param)
    assert Path(param.source_mesh_path_) == source
    assert Path(param.mesh_path_).is_file()
    assert Path(param.visual_mesh_path_).is_file()


def test_default_recording_paths_are_writable_from_paper_directory(ramp_scope, monkeypatch):
    monkeypatch.chdir(ROOT / 'tro_revision')
    records = Path(ramp_scope['DEFAULT_RECORD_OUTPUT_DIR'])
    screenshots = Path(ramp_scope['DEFAULT_SVG_SCREENSHOT_DIR'])
    assert records == ROOT / 'outputs/videos_tilted_pushing'
    assert screenshots == records / 'screenshots'
    screenshots.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='_test_recording_', dir=str(records)) as base:
        allocate = ramp_scope['_allocate_trial_record_dir']
        first = Path(allocate(base, 'elephant'))
        second = Path(allocate(base, 'elephant'))
        assert first.is_dir() and second.is_dir()
        assert first.name == '000' and second.name == '001'
        (first / 'traj.json').write_text('{}\n')
        assert allocate('', 'elephant') is None


def test_default_init_and_goal_fit_the_whole_rotated_mesh(configured):
    param, _ = configured
    for pose in [param.init_obj_qpos_, np.r_[param.target_p_, param.init_obj_qpos_[3:7]]]:
        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        world = param.support_vertices_ @ rotation.T + pose[:3]
        footprint = (world-param.support_surface_point_) @ param.support_surface_tangents_.T
        assert np.all(np.abs(footprint) <= param.support_surface_half_extents_-.04+5e-8)
        assert np.min((world-param.support_surface_point_) @ param.support_surface_normal_) >= -5e-8


def test_default_elephant_has_a_quarter_metre_uphill_task(ramp_scope):
    scope = {}
    extract('examples/mpc/fingertips/test/test_0902.py', {'add_rollout_via_args'}, scope)
    parser = argparse.ArgumentParser()
    scope['add_rollout_via_args'](parser)
    args = parser.parse_args([])
    args.obj, args.seed = 'elephant', 15
    for trial in range(5):
        param = ExplicitMPCParams(args, rand_seed=trial)
        param.obj_mesh_scale_ = np.ones(3)
        prepare_object_geometry(param)
        ramp_scope['_apply_dywa_physics_to_param'](param)
        ramp_scope['_apply_tilted_ramp_scene_to_param'](param, True, True,
            ramp_scope['DEFAULT_TARGET_RAMP_LOCAL_XY'], 0.,
            ramp_scope['DEFAULT_TARGET_WORLD_OFFSET_FROM_INIT'], True)
        tangent_delta = (param.target_p_ - param.init_obj_qpos_[:3]) @ param.support_surface_tangents_.T
        assert np.linalg.norm(tangent_delta) == pytest.approx(.25, abs=.015)
        assert tangent_delta[1] < -.23
        assert np.linalg.norm(param.target_p_-param.init_obj_qpos_[:3]) > .23


def test_pose_terminal_and_control_weights_match_mppi_reference(configured):
    from planning.object_pose_cost import cost_config, extended_cost_config
    reference = dict(np=np)
    extract('planning/MPPIWarp.py', {'object_pose_cost', 'terminal_object_pose_cost'}, reference)
    param, _ = configured
    param.ori_coef = 20.
    param.object_cost_config_ = extended_cost_config(cost_config(500., 20.))
    mpc = InspectMPC(param)
    position = param.support_surface_point_ + .08*param.support_surface_normal_
    quat = np.array([.8, .2, -.1, .55]); quat /= np.linalg.norm(quat)
    target_p = position + [.01, -.05, .02]
    target_q = np.array([0., 2**-.5, 0., 2**-.5])
    state = np.r_[position, quat, position + [-.1, 0., .05]]
    cost = mpc._cost_params(target_p=target_p, target_q=target_q,
        virtual_point=state[7:10], contact_point=state[7:10], contact_reference=state[:7])
    expected = reference['terminal_object_pose_cost'](position, quat, target_p, target_q, 500., 20.)
    assert float(mpc.final_cost_fn(state, cost)) == pytest.approx(expected, abs=1e-9)
    command = np.array([.001, -.002, .003])
    effort_delta = float(mpc.path_cost_fn(state, command, cost)-mpc.path_cost_fn(state, np.zeros(3), cost))
    assert effort_delta == pytest.approx(50.*np.dot(command, command), abs=1e-12)


@pytest.mark.parametrize('physical', [False, True])
def test_near_patch_damps_tracking_error_until_actual_contact(ramp_scope, physical):
    scope = dict(ramp_scope)
    scope.update(POLICY_INTERVAL=.02, ISAAC_TASK_KP=600., ISAAC_TASK_KD=40.,
        FREE_SPACE_FORCE_LIMIT=12., CONTACT_FORCE_LIMIT=10.,
        _remaining_along_action=cf.remaining_along_action,
        _project_along_action=cf.project_along_action, _isaac_task_force=cf.isaac_task_force,
        _near_press_force_mode=cf.near_press_force_mode,
        _osc_action_task_force=cf.osc_action_task_force, _press_normal_outward=cf.press_normal_outward,
        _franka_nullspace_posture_torque=lambda *a: np.zeros(7))
    tree = ast.parse((ROOT/'examples/mpc/franka/test_tilted_push/test_tiltedpush.py').read_text())
    simulator = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'IsaacFrankaOSCSimulator')
    method = next(n for n in simulator.body if isinstance(n, ast.FunctionDef) and n.name == '_compute_osc_torques')
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<ramp_osc>', 'exec'), scope)
    tip, action = np.array([.4, 0., .5]), np.array([.005, 0., 0.])
    env = SimpleNamespace(param_=SimpleNamespace(contact_force_limit_=10.),
        _refresh_osc_tensors=lambda: None, get_current_joint_position=lambda: np.zeros(7),
        get_current_joint_velocity=lambda: np.r_[.1, np.zeros(6)],
        get_end_effector_pos=lambda: (tip, np.eye(3)),
        _get_position_jacobian=lambda q: np.c_[np.eye(3), np.zeros((3, 4))],
        position_d=tip+action, _mpc_action=action, _mpc_p0=tip, _mpc_T=.02,
        _mpc_press=tip+[.008, 0., 0.], _path_blocked=False,
        _fingertip_contact_info=lambda: (physical, [-1., 0., 0.]),
        q_d_nullspace=np.zeros(7), nullspace_stiffness=0.,
        activate_tool_compensation=False, torque_limits=100*np.ones(7))
    scope['_compute_osc_torques'](env)
    # Before touch: K*u - D*(v-u/h) = 9 N, rather than the old -1 N brake.
    assert env._last_osc_force[0] == pytest.approx(-1. if physical else 9.)


def test_support_friction_opposes_downhill_gravity(configured, ramp_scope):
    param, _ = configured
    q = cube_on_ramp(param)
    _, (phi, jac, _, _, _) = detect(ramp_scope, param, q)
    mpc = InspectMPC(param)
    state = mpc.observation_state(q)
    speeds = []
    for mu in (0., param.support_friction_):
        gaps, rows = refresh_table_support(param.support_vertices_, state, phi, jac,
            .35, mu, **mpc.support_plane)
        prediction = np.asarray(mpc.stage_prediction_fn(state, np.zeros(3), gaps, rows, 7., state)).ravel()
        speeds.append(prediction[10:13] @ param.support_surface_tangents_[1])
    assert speeds[0] == pytest.approx(.02*9.81*np.sin(np.radians(15)), rel=1e-6)
    assert abs(speeds[1]) < .2*speeds[0]


def test_withdrawing_finger_loses_flip_progress_on_the_ramp(configured, ramp_scope):
    param, _ = configured
    basis = np.vstack([param.support_surface_tangents_, param.support_surface_normal_])
    rotation = basis.T @ Rotation.from_euler('y', 30, degrees=True).as_matrix()
    q = cube_on_ramp(param)
    q[3:7] = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    height = -np.min((param.support_vertices_ @ rotation.T) @ param.support_surface_normal_)
    q[:3] = param.support_surface_point_ + height*param.support_surface_normal_
    point = np.array([-.04, 0., .04])
    q[7:10] = q[:3] + rotation @ (point + [-.01, 0., 0.])
    _, (phi, jac, _, _, _) = detect(ramp_scope, param, q)
    relative = np.c_[np.eye(3), -cf._skew(rotation @ point), -np.eye(3)]
    n, t1, t2 = cf.tangent_basis_from_normal(rotation[:, 0])
    jac[16:20] = cf._contact_jacobian_np(n, t1, t2, relative, param.contact_friction_)
    jac[16:20, 3:6] = jac[16:20, 3:6] @ rotation
    phi[16:20] = 0.
    model = ExplicitModel(param)
    x, u = cs.SX.sym('q', 10), cs.SX.sym('u', 3)
    gaps, rows = contact_geometry_cs(x, cs.DM(q), cs.DM(phi), cs.DM(jac))
    step = cs.Function('ramp_flip_step', [x, u], [model.prediction_fn(x, u, gaps, rows, 7.)])

    def pitch_after(command):
        state = q.copy()
        for _ in range(5):
            state = np.asarray(step(state, command)).ravel()
        orientation = Rotation.from_quat(state[[4, 5, 6, 3]]).as_matrix()
        return Rotation.from_matrix(basis @ orientation).as_euler('xyz', degrees=True)[1]

    assert pitch_after(rotation @ np.array([1/600., 0., 0.])) > 30.
    assert pitch_after(-.004*rotation[:, 0]) < 30.
