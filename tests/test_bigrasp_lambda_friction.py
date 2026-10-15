"""Lambda force bounds and materials must match bigrasp's grouped Coulomb SCM."""
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples.mpc.franka.bigrasp.bigrasp import (
    BimanualPandaGrasper, PANDA_XML_PATH, build_argparser,
    lambda_friction_options, mujoco_pair_sliding_friction,
)
from planning.lambda_batched_lm import BatchedLambdaLM


@pytest.mark.parametrize('priorities,expected', [('0 0', 1.4), ('1 0', .7), ('0 1', 1.4)])
def test_lambda_receives_effective_mujoco_pair_friction(priorities, expected):
    p, q = priorities.split()
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
        <geom name="pad" type="sphere" size=".01" friction=".7 .01 .001" priority="{p}"/>
        <geom name="obj" type="sphere" size=".01" friction="1.4 .01 .001" priority="{q}"/>
        </worldbody></mujoco>''')
    mu = mujoco_pair_sliding_friction(model, 0, 1)
    assert mu == pytest.approx(expected)
    args = build_argparser().parse_args([])
    options = lambda_friction_options(args, mu)
    assert options == dict(arm_friction=expected, friction_cone='coulomb', force_reg_coef=.01, friction_reg_coef=.03)
    args = build_argparser().parse_args(['--optimizer-arm-friction', '.4', '--friction_reg_coef', '0'])
    assert lambda_friction_options(args, mu)['arm_friction'] == .4
    assert lambda_friction_options(args, mu)['friction_reg_coef'] == 0.


def test_explicit_contact_pair_overrides_geom_materials():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <geom name="a" type="sphere" size=".01" friction="5 .01 .001"/>
        <geom name="b" type="sphere" size=".01" friction="5 .01 .001"/>
        </worldbody><contact><pair geom1="a" geom2="b" friction=".3 .3 .01 .001 .001"/></contact></mujoco>''')
    assert mujoco_pair_sliding_friction(model, 0, 1) == pytest.approx(.3)


@pytest.mark.parametrize('impratio,noslip,cone,dim',
    [(10., 3, 'elliptic', 3), (10., 10, 'pyramidal', 6), (2., 0, 'elliptic', 4)])
def test_scene_materials_are_applied_before_building_lambda_and_mpc(tmp_path, monkeypatch, impratio, noslip, cone, dim):
    from types import SimpleNamespace
    import planning.bimanual_online_mpc as mpc
    monkeypatch.setattr(mpc, 'BimanualOnlineMPC', lambda p: SimpleNamespace(planner_solver_='acados', horizon=2))
    (tmp_path/'assets').symlink_to(PANDA_XML_PATH.parent/'assets', target_is_directory=True)
    args = build_argparser().parse_args(['--mpc', '--no-visualize', '--mppi-device', 'cpu',
        '--scene-output', str(tmp_path/'scene.xml'), '--sample-num', '24',
        '--arm-friction', '.7', '--object-friction', '1.4',
        '--contact-impratio', str(impratio), '--contact-noslip-iterations', str(noslip),
        '--contact-friction-cone', cone, '--object-contact-dim', str(dim)])
    demo = BimanualPandaGrasper(args)
    try:
        assert demo.optimizer.mu_arm_obj == pytest.approx(1.4)
        assert demo.plan_params.arm_friction == pytest.approx(1.4)
        assert demo.optimizer.lm.friction_cone == 'coulomb'
        assert demo.model.opt.cone == (mujoco.mjtCone.mjCONE_ELLIPTIC if cone == 'elliptic' else mujoco.mjtCone.mjCONE_PYRAMIDAL)
        assert demo.model.opt.impratio == pytest.approx(impratio)
        assert demo.model.opt.noslip_iterations == noslip
        assert demo.model.geom_condim[demo.obj_geom_id] == dim
        for arm in (demo.left_arm, demo.right_arm):
            assert demo.model.geom_friction[arm.tip_geom_id, 0] == pytest.approx(.7)
    finally:
        demo.close()


def test_default_execution_preserves_previous_torsional_and_rolling_contacts():
    args = build_argparser().parse_args(['--mpc'])
    assert args.contact_impratio == 10.
    assert args.contact_noslip_iterations == 10
    assert args.contact_friction_cone == 'pyramidal'
    assert args.object_contact_dim == 6


def test_global_contacts_validate_real_arm_geometry_without_changing_execution(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import planning.bimanual_online_mpc as mpc
    monkeypatch.setattr(mpc, 'BimanualOnlineMPC', lambda p: SimpleNamespace(planner_solver_='acados', horizon=8))
    (tmp_path/'assets').symlink_to(PANDA_XML_PATH.parent/'assets', target_is_directory=True)
    args = build_argparser().parse_args(['--mpc', '--no-visualize', '--mppi-device', 'cpu',
        '--scene-output', str(tmp_path/'scene.xml')])
    demo = BimanualPandaGrasper(args)
    try:
        before = [demo.data.qpos.copy(), demo.data.qvel.copy(), demo.data.ctrl.copy(), demo.data.time]
        pos, quat, rotation = demo.get_object_pose()
        targets, _, _, info = demo._update_warp_lambda_contacts(pos, quat, rotation, initial=True)
        selection = info['contact_selection']
        assert selection['approach_and_lift_collision_free']
        assert info['selection_backend'] == 'coupled_soft_scm'
        assert np.isfinite(selection['cost'])
        assert selection['rollout'].shape == (8, 25)
        ids = info['contact_indices']
        np.testing.assert_array_equal(targets['contact_points_local'], demo.optimizer.sample_point[ids])
        np.testing.assert_array_equal(targets['normals_local'], demo.optimizer.normal[ids])
        prediction = demo._lambda_contact_prediction
        geometry = prediction._geometry(targets['contact_points_local'], targets['normals_local'])
        expected = prediction._rollout(selection['command_delta_world'].ravel(), geometry, details=True)
        np.testing.assert_allclose(info['x_plus_opt'], expected['x_plus'])
        np.testing.assert_allclose(info['force_vectors_local'], expected['force_vectors_local'])
        # The same antipodal pair, one metre above the body, is outside the
        # robot workspace and must not pass on force geometry alone.
        invalid = demo._prospective_pair_feasible(targets['contact_points_local']+[0., 0., 1.],
            targets['normals_local'], pos, rotation)
        assert not invalid['valid'] and 'IK' in invalid['reason']
        for actual, original in zip((demo.data.qpos, demo.data.qvel, demo.data.ctrl, demo.data.time), before):
            np.testing.assert_array_equal(actual, original)
    finally:
        demo.close()


@pytest.mark.parametrize('mu', [0., .7, 1.2, 5.])
def test_batched_lambda_forces_obey_coulomb_disk_under_object_rotation(mu):
    rotation = Rotation.from_euler('xyz', [.3, -.4, .7]).as_matrix()
    quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    points = np.array([[[.04, 0., 0.], [-.04, .01, 0.]]])
    normals = np.array([[[-1., 0., 0.], [1., 0., 0.]]])
    solver = BatchedLambdaLM(num_contacts=2, device='cpu', iterations=3,
        mu=mu, max_contact_force=3., friction_cone='coulomb')
    result = solver.solve(points, normals, current_x=np.r_[np.zeros(3), quat],
        x_d=np.r_[[.03, -.02, .08], quat], object_inertia=np.diag([.2]*3+[.004]*3),
        prediction_dt=.05, initial_forces=np.array([[[1., 10., -7.], [2., -4., 8.]]]))
    forces = result['control_input']
    assert np.all(np.linalg.norm(forces[:, 1:], axis=1) <= mu*forces[:, 0]+2e-6)
    assert np.all(np.linalg.norm(forces, axis=1) <= 3.+2e-6)
    for normal, force in zip(normals[0], result['force_vectors_local']):
        world_n, world_force = rotation@normal, rotation@force
        normal_force = world_n@world_force
        tangent_force = world_force-normal_force*world_n
        assert normal_force >= -2e-6
        assert np.linalg.norm(tangent_force) <= mu*normal_force+2e-6


def test_small_tangential_penalty_keeps_useful_lifting_force():
    points = np.array([[[.04, 0., 0.], [-.04, 0., 0.]]])
    normals = np.array([[[-1., 0., 0.], [1., 0., 0.]]])
    inputs = dict(current_x=[0., 0., 0., 1., 0., 0., 0.], x_d=[0., 0., .1, 1., 0., 0., 0.],
        object_inertia=np.diag([.1]*3+[.004]*3), prediction_dt=.1)
    def solve(penalty):
        return BatchedLambdaLM(device='cpu', iterations=20, friction_cone='coulomb',
            force_reg_coef=.01, friction_reg_coef=penalty).solve(points, normals, **inputs)
    small, excessive = solve(.03), solve(1.)
    assert small['contact_force_world'][2] > 2*excessive['contact_force_world'][2]
    assert small['x_plus_opt'][2] > excessive['x_plus_opt'][2]
