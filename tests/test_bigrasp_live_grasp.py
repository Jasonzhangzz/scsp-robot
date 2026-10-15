"""Live grasp certificates must use real patches, including their torque arms."""
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from examples.mpc.franka.bigrasp.bigrasp import BimanualPandaGrasper, build_argparser
from planning.bimanual_live_grasp import contact_certificate, choose_nearby_pair, live_force_target
from planning.lambda_batched_lm import BatchedLambdaLM


def test_native_fa_scene_accepts_elliptic_execution_and_uses_effective_pair_friction(tmp_path):
    from models.dexforge_planner_scene import write_dexforge_planner_scene
    viewer = tmp_path/'viewer.xml'
    viewer.write_text('''<mujoco><option cone="elliptic" noslip_iterations="10" integrator="implicitfast"/>
      <worldbody><body name="obj"><freejoint/><geom name="obj" type="box" size=".03 .03 .03" mass=".01"/></body>
      <body name="left_attachment"><geom type="sphere" size=".01"/></body>
      <body name="right_attachment"><geom type="sphere" size=".01"/></body></worldbody></mujoco>''')
    original = viewer.read_bytes()
    cloud = SimpleNamespace(centers=np.zeros((1, 3)), radii=np.array([.004]))
    planner = write_dexforge_planner_scene(viewer, cloud, pair_sliding_friction=1.4)
    root = ET.parse(planner).getroot()
    assert root.find('option').get('cone') == 'pyramidal'
    assert root.find('option').get('noslip_iterations') == '0'
    for name in ('obj_gs', 'left_tip_gs', 'right_tip_gs'):
        geom = next(g for g in root.iter('geom') if g.get('name') == name)
        assert float(geom.get('friction').split()[0]) == pytest.approx(1.4)
    assert viewer.read_bytes() == original


POINTS = np.array([[-.05, 0., 0.], [.05, 0., 0.]])
NORMALS = np.array([[1., 0., 0.], [-1., 0., 0.]])
EXTERNAL = np.array([0., 0., -.2943, 0., 0., 0.])


def test_antipodal_two_point_grasp_is_not_full_six_dimensional_force_closure():
    cert = contact_certificate(POINTS, NORMALS, .7, EXTERNAL, normal_limits=[1., 1.])
    assert cert['wrench_rank'] == 5
    assert not cert['full_wrench_force_closure'] and not cert['force_closure']
    assert cert['force_closure_in_contact_span'] and cert['load_feasible']
    unsupported = EXTERNAL.copy()
    unsupported[3] = .01  # Pure torque along the contact line is unavailable.
    assert not contact_certificate(POINTS, NORMALS, .7, unsupported)['load_feasible']
    task = contact_certificate(POINTS, NORMALS, .7, unsupported)
    assert task['task_load_feasible']
    np.testing.assert_allclose(task['uncontrolled_wrench'], [0., 0., 0., -.01, 0., 0.])


def test_condim6_torsion_closes_axis_that_point_force_contacts_cannot_control():
    external = EXTERNAL.copy()
    external[3] = -.01
    point = contact_certificate(POINTS, NORMALS, .7, external, normal_limits=[1., 1.])
    assert not point['load_feasible']
    full = contact_certificate(POINTS, NORMALS, .7, external, normal_limits=[1., 1.],
                               rotational_friction=(.02, .01))
    assert full['full_wrench_force_closure'] and full['wrench_rank'] == 6 and full['load_feasible']
    np.testing.assert_allclose(full['uncontrolled_wrench'], 0.)
    weak = contact_certificate(POINTS, NORMALS, .7, external, normal_limits=[.01, .01],
                               rotational_friction=(.02, .01))
    assert not weak['load_feasible']
    allocation = live_force_target(POINTS, NORMALS, .7, -external, external,
        minimum_normal=.2, rotational_friction=(.02, .01))
    assert allocation['feasible'] and allocation['task_wrench_scale'] == 1.
    force, moment = allocation['forces'], allocation['moments']
    np.testing.assert_allclose(np.r_[force.sum(0), np.cross(POINTS, force).sum(0)+moment.sum(0)], -external, atol=1e-8)
    np.testing.assert_allclose(allocation['uncontrolled_wrench'], 0.)


def test_real_noncollinear_manifold_patches_can_certify_full_wrench_closure():
    points = np.array([[-.05, -.02, 0.], [-.05, .02, 0.], [.05, 0., 0.]])
    cert = contact_certificate(points, NORMALS[[0, 0, 1]], .7, EXTERNAL,
                               normal_limits=[1., 1., 1.])
    assert cert['wrench_rank'] == 6 and cert['full_wrench_force_closure']
    assert cert['load_feasible']


def test_nearby_replacement_preserves_real_contact_and_ignores_remote_anchor():
    stale = POINTS.copy()
    stale[0, 1] = .03
    real = [[dict(point=POINTS[0], normal=NORMALS[0], normal_force=1.)], []]
    candidates = [[], [(POINTS[1], NORMALS[0]), (POINTS[1], NORMALS[1])]]
    points, normals, replaced, cert = choose_nearby_pair(stale, NORMALS, real, candidates, .7, EXTERNAL)
    np.testing.assert_array_equal(replaced, [1])
    np.testing.assert_allclose(points, POINTS)
    np.testing.assert_allclose(normals, NORMALS)
    assert cert['force_closure_in_contact_span'] and cert['load_feasible']
    # The same prospective certificate does not supply forces before touch.
    now = contact_certificate(POINTS[:1], NORMALS[:1], .7, EXTERNAL, normal_limits=[1.])
    assert not now['force_closure_in_contact_span']


def test_unreachable_or_unselected_finger_never_contributes_to_certificate():
    real = [[dict(point=POINTS[0], normal=NORMALS[0], normal_force=1.)], []]
    _, _, replaced, cert = choose_nearby_pair(POINTS, NORMALS, real, [[], []], .7, EXTERNAL)
    assert len(replaced) == 0 and cert is None


def test_future_preload_is_not_clamped_to_weak_measured_pressure():
    allocation = live_force_target(POINTS, NORMALS, .7, -EXTERNAL, EXTERNAL, minimum_normal=.2)
    assert allocation['feasible']
    force = allocation['forces']
    np.testing.assert_allclose(np.r_[force.sum(0), np.cross(POINTS, force).sum(0)], -EXTERNAL, atol=1e-7)
    assert np.min(allocation['normal_forces']) >= .2
    actual = contact_certificate(POINTS, NORMALS, .7, EXTERNAL, normal_limits=[.01, .01])
    assert not actual['load_feasible']


def test_uncontrolled_task_torque_is_explicit_and_preserves_lifting_force():
    requested = -EXTERNAL.copy()
    requested[3] = .01
    allocation = live_force_target(POINTS, NORMALS, .7, requested, EXTERNAL)
    assert allocation['feasible'] and allocation['task_wrench_scale'] == 1.
    np.testing.assert_allclose(allocation['uncontrolled_wrench'], [0., 0., 0., .01, 0., 0.])
    np.testing.assert_allclose(allocation['requested_wrench'], -EXTERNAL)


@pytest.mark.parametrize('freeze,reject_replan,infeasible',
    [(False, False, False), (True, False, False), (False, True, False), (False, False, True)])
def test_real_mujoco_patches_replace_offline_points_and_reallocate_load(freeze, reject_replan, infeasible):
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep=".002" cone="elliptic" impratio="10" noslip_iterations="3"/>
      <default><geom friction=".7 .01 .001" condim="3" solref=".02 1"/></default>
      <worldbody><body name="obj" pos="0 0 .3"><freejoint/>
        <geom name="obj" type="box" size=".05 .02 .02" mass=".03"/></body>
        <body name="left_pad" pos="-.058 0 .3"><geom name="left_tip" type="sphere" size=".01"/>
          <site name="left_site"/></body>
        <body name="right_pad" pos=".058 0 .3"><geom name="right_tip" type="sphere" size=".01"/>
          <site name="right_site"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    for _ in range(30):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    demo = BimanualPandaGrasper.__new__(BimanualPandaGrasper)
    demo.model, demo.data = model, data
    demo.obj_body_id, demo.obj_geom_id = model.body('obj').id, model.geom('obj').id
    demo.obj_qpos_adr, demo.obj_dof_adr = 0, 0
    demo.args = build_argparser().parse_args(['--mpc', '--obj-mass', '.03'])
    demo.args.freeze_contact_pair = freeze
    demo.planner_mode = 'mpc_ee'
    demo.plan_params = SimpleNamespace(h_=.01)
    demo.planner = SimpleNamespace(reset=lambda: None if reject_replan or infeasible else
                                  pytest.fail('Loaded contacts must not trigger a speculative switch'))
    demo.support_top = 0.
    for name in ('left', 'right'):
        setattr(demo, name+'_arm', SimpleNamespace(tip_geom_id=model.geom(name+'_tip').id,
            tip_site_id=model.site(name+'_site').id,
            body_ids_by_name={name+'_pad': model.body(name+'_pad').id}))
    demo.get_tip_pos = lambda arm: data.site_xpos[arm.tip_site_id].copy()
    demo.object_target_pos = data.qpos[:3]+[0., 0., .06]
    demo.object_target_quat = data.qpos[3:7].copy()
    demo.contact_points_local = POINTS+[0., .03, 0.]
    demo.normals_local = NORMALS.copy()
    demo.fingertip_object_friction = .7
    demo.optimizer = SimpleNamespace(sample_point=POINTS, normal=NORMALS,
        surface=SimpleNamespace(object_radius=.05, body_sides=False,
                                mesh=SimpleNamespace(vertices=np.array([[-.05, -.02, -.02], [.05, .02, .02]]))),
        lm=BatchedLambdaLM(num_contacts=2, device='cpu', iterations=3, mu=.7,
                          friction_cone='coulomb', max_contact_force=12.))
    before = data.qpos.copy()
    pos, quat, rotation = demo.get_object_pose()
    targets, predicted, _, info = demo._update_warp_lambda_contacts(pos, quat, rotation)
    np.testing.assert_array_equal(data.qpos, before)
    actual = demo.extract_object_contacts()
    for arm, name in enumerate(('left', 'right')):
        assert actual[name]
        patch = max(actual[name], key=lambda c: c['normal_force'])
        np.testing.assert_allclose(targets['contact_points_local'][arm], patch['local_pos'])
        np.testing.assert_allclose(targets['normals_local'][arm], patch['normal_local'])
        assert patch['normal_world'] @ (pos-demo.get_tip_pos(getattr(demo, name+'_arm'))) > 0
    assert info['live_grasp']['force_allocation_feasible']
    assert info['live_grasp']['actual_certificate']['load_feasible']
    assert info['live_grasp']['local_contact_updates'] == []
    assert predicted[2] > pos[2]
    assert np.all(info['contact_forces_local'][:, 0] > 0)
    if infeasible:
        # Both pads touch, but their same-facing normals cannot form a positive
        # wrench span, even with unlimited measured pressure.
        demo.extract_object_contacts = lambda: {
            'left': [dict(local_pos=POINTS[0], normal_local=NORMALS[0], normal_force=1., dist=0.)],
            'right': [dict(local_pos=POINTS[1], normal_local=NORMALS[0], normal_force=1., dist=0.)]}
        demo._contact_search_worker = SimpleNamespace(poll=lambda: None, ready=False, pending=True)
        for _ in range(demo.args.contact_replan_lost_steps):
            _, _, _, info = demo._update_warp_lambda_contacts(pos, quat, rotation)
        live = info['live_grasp']
        assert live['geometry_infeasible'] and live['recovery_stage'] == 'release'
        assert demo._online_recovery_mask.all()
        np.testing.assert_allclose(info['force_vectors_local'], 0.)
        centers, mask = demo._online_contact_approach_centers(pos, rotation)
        assert mask.all() and centers[0, 0] < demo.get_tip_pos(demo.left_arm)[0]
        assert not demo._online_grasp_observation()[2]
    if reject_replan:
        demo.contact_points_local = targets['contact_points_local'].copy()
        demo.normals_local = targets['normals_local'].copy()
        # Simulate a lost-contact observation with no reachable local patch.
        # A rejected global Lambda search must keep these last live anchors.
        demo.extract_object_contacts = lambda: dict(left=[], right=[])
        demo.optimizer.sample_point = POINTS+[0., 1., 0.]
        demo.args.online_nearby_contact_radius = 1e-6
        demo._nearby_contact_reachable = lambda arm, point: False
        demo._online_approach_origin = pos-[.1, 0., 0.]
        demo._online_lost_contact_steps = 3
        attempts = []
        def reject(*args, **kwargs):
            attempts.append(True)
            raise FloatingPointError('No admissible low-curvature lambda contact pair')
        demo.optimizer.initialize_offline = reject
        demo._lambda_selection_options = lambda *args: {}
        demo._contact_search_snapshot = lambda *args: lambda: reject()
        retry, _, _, info = demo._update_warp_lambda_contacts(pos, quat, rotation)
        from concurrent.futures import wait
        wait([demo._contact_search_worker.future], timeout=1.)
        retry, _, _, info = demo._update_warp_lambda_contacts(pos, quat, rotation)
        assert info['live_grasp']['global_contact_replan_error']
        assert not info['live_grasp']['global_contact_replan']
        assert not info['live_grasp']['actual_certificate']['task_load_feasible']
        np.testing.assert_array_equal(retry['contact_points_local'], targets['contact_points_local'])
        np.testing.assert_array_equal(data.qpos, before)
        demo._update_warp_lambda_contacts(pos, quat, rotation)
        assert len(attempts) == 1  # Failure retries respect wall time, not fast simulation time.
        assert info['live_grasp']['recovery_stage'] == 'release'
        assert not info['live_grasp']['force_allocation_feasible']
        demo._contact_search_worker.close()
