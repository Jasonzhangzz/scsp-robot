"""Physical checks for the pose-aware Isaac LCS (no Isaac import required)."""

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from .rigid import RigidScene, pack_state, orientation_error, stage_costs, QUAT, quaternion_angle_hessian
from .rigid_sampling import RigidSamplingController


@pytest.fixture
def scene(tmp_path):
    mesh = trimesh.creation.box(extents=[.08, .06, .10])
    path = tmp_path / 'box.obj'
    mesh.export(path)
    inertia = np.diag(.2 / 12 * np.array([.06**2+.10**2, .08**2+.10**2, .08**2+.06**2]))
    return RigidScene([path], .2, inertia, .35)


def test_table_balances_gravity_without_tip_force(scene):
    x = pack_state([-.08, 0., .43], [0., 0., .40], [1., 0., 0., 0.])
    model = scene.linearize(x)
    nxt, force = model.simulate(x, np.zeros(3))
    assert np.allclose(nxt, x, atol=1e-7)
    assert np.all(force >= -1e-8)
    assert np.min(model.gap(x, np.zeros(3), force)) >= -1e-7
    assert np.max(np.abs(force * model.gap(x, np.zeros(3), force))) < 1e-7


def test_finger_force_has_equal_opposite_force_and_real_moment_arm(scene):
    x = pack_state([-.05, 0., .43], [0., 0., .40], [1., 0., 0., 0.])
    model = scene.linearize(x)
    # Sum the four friction rays: tangents cancel, leaving a pure normal.
    acceleration = model.D[10:, :4].sum(axis=1) / model.dt
    assert np.allclose(scene.tip_mass * acceleration[:3] + scene.mass * acceleration[6:9], 0., atol=1e-9)
    lever = model.contacts[0].point - x[7:10]
    object_force = scene.mass * acceleration[6:9]
    assert np.allclose(scene.inertia @ acceleration[3:6], np.cross(lever, object_force), atol=1e-9)
    assert np.linalg.norm(acceleration[3:6]) > 1.


def test_orientation_cost_uses_target_and_is_quaternion_sign_invariant(scene):
    q = Rotation.from_euler('y', 65, degrees=True).as_quat()[[3, 0, 1, 2]]
    Q = stage_costs(5, 24, q)[0][0][QUAT, QUAT]
    assert abs(q @ Q @ q) < 1e-8
    assert abs((-q) @ Q @ (-q)) < 1e-8
    assert np.array([1., 0., 0., 0.]) @ Q @ np.array([1., 0., 0., 0.]) > 100.
    assert orientation_error(q, -q) < 1e-7


def test_native_quaternion_hessian_matches_angle_loss_curvature():
    q = np.array([.8, .3, -.2, .4]); q /= np.linalg.norm(q)
    r = np.array([.9, -.1, .2, .3]); r /= np.linalg.norm(r)
    def loss(value):
        return (2*np.arccos(np.clip((value @ r)/np.linalg.norm(value), -1., 1.)))**2
    eps, eye = 2e-5, np.eye(4)
    finite = np.empty((4, 4))
    for i in range(4):
        for j in range(4):
            finite[i, j] = (loss(q+eps*(eye[i]+eye[j]))-loss(q+eps*(eye[i]-eye[j]))
                            -loss(q+eps*(-eye[i]+eye[j]))+loss(q-eps*(eye[i]+eye[j])))/(4*eps**2)
    assert np.allclose(quaternion_angle_hessian(q, r), finite, atol=2e-5)


def test_anitescu_gap_has_no_artificial_tangential_position_bias(scene):
    x = pack_state([-.06, 0., .43], [0., 0., .40], [1., 0., 0., 0.])
    model = scene.linearize(x)
    # At the linearization state, free dynamics contact velocity plus phi/dt.
    free = model.A @ x + model.d
    expected = np.repeat([c.gap / model.dt for c in model.contacts], 4) + model.Jc @ free[10:]
    assert np.allclose(model.E @ x + model.c, expected, atol=1e-8)
    assert np.allclose(model.F, model.F.T)
    assert np.linalg.eigvalsh(model.F).min() > -1e-8


def test_rigid_controller_responds_to_orientation_goal(scene):
    pytest.importorskip('pyscipopt')
    x = pack_state([-.05, 0., .43], [0., 0., .40], [1., 0., 0., 0.])
    controller = RigidSamplingController(scene, n_samples=0, projection_time_limit=.5)
    try:
        upright = controller.plan(x, x[7:10], [1., 0., 0., 0.])
        goal = Rotation.from_euler('y', 50, degrees=True).as_quat()[[3, 0, 1, 2]]
        tilted = controller.plan(x, x[7:10], goal)
    finally:
        controller.close()
    assert np.linalg.norm(upright['ee_command'] - tilted['ee_command']) > 1e-5
    assert tilted['plan']['x'].shape == (6, 19)
    assert np.allclose(np.linalg.norm(tilted['plan']['x'][:, QUAT], axis=1), 1.)
    assert orientation_error(tilted['plan']['x'][-1, QUAT], goal) < orientation_error(x[QUAT], goal)
    assert np.min(tilted['plan']['nominal_x'][1:, 2]) >= scene.table_height+scene.tip_radius+.002-1e-7
    # Feedforward and nominal EE knots must come from the same dynamics QP,
    # rather than mixing its knots with the projected visualization inputs.
    assert np.allclose(tilted['force_command'], tilted['plan']['u'][0])


def test_optimization_contact_reduction_keeps_richer_physical_rollout(scene):
    x = pack_state([-.05, 0., .43], [0., 0., .40], [1., 0., 0., 0.])
    planning = scene.linearize(x, for_planning=True)
    physical = scene.linearize(x)
    assert planning.m == 16
    assert physical.m == 24
    assert sum(c.kind == 'finger' for c in planning.contacts) == 1
    assert sum(c.kind == 'table' for c in planning.contacts) == 3
    assert all(c.kind != 'tip_table' for c in planning.contacts)
    assert any(c.kind == 'tip_table' for c in physical.contacts)


def test_isaac_adapter_preserves_pose_velocity_and_contact_tracking_contract(scene):
    from types import SimpleNamespace
    from .isaac import SamplingC3Controller

    calls = []
    def plan(state, target_p, target_q):
        calls.append((state, target_p, target_q))
        return dict(ee_command=state[:3]+[.002, 0., 0.], is_c3=True, mode='c3',
                    position_error=.1, orientation_error=.5, sample_costs=[1.])
    adapter = SamplingC3Controller()
    adapter.controller = SimpleNamespace(plan=plan, scene=scene)
    q = np.array([1., 0., 0., 0.])
    payload = dict(policy_q=np.r_[[0., 0., .4], q, [-.08, 0., .43]],
                   object_angular_velocity=[.1, .2, .3], object_linear_velocity=[.4, .5, .6])
    result = adapter.step(payload, [.1, 0., .4], q)
    assert np.allclose(calls[0][0][3:7], q)
    assert np.allclose(calls[0][0][13:16], [.1, .2, .3])
    assert np.allclose(calls[0][2], q)
    assert not result['policy']['release_pressure']
    # Press is a surface witness, separated from the next free-space via.
    assert np.linalg.norm(result['execution_press']-result['policy']['mpc_virtual_point']) > .01
