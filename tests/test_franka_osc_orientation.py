"""Tool-attitude control must not inherit a preferred world-X direction."""
import os

os.environ['SCSP_PLANNER_ONLY'] = '1'
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples.mpc.franka.runtime import mpc as entry
from tests.test_franka_contact_motion import osc_stub
from examples.mpc.franka.runtime.contact import slew_tool_rotation, downward_tool_frame


@pytest.mark.parametrize('q', [
    entry.FRANKA_QD_NULLSPACE,
    np.array([.2, -.5, .3, -2., -.4, 1.7, .9]),
    np.array([-.3, -.8, -.2, -1.7, .5, 2., -.7]),
])
def test_tool_pose_jacobian_matches_world_frame_finite_difference(q):
    jac = entry._franka_pose_jacobian_np(q)
    eps = 1e-6
    for index in range(7):
        step = np.eye(7)[index]*eps
        plus, minus = entry._franka_fk_T_np(q+step), entry._franka_fk_T_np(q-step)
        np.testing.assert_allclose(jac[:3, index],
            (plus[:3, 3]-minus[:3, 3])/(2*eps), atol=1e-8)
        np.testing.assert_allclose(jac[3:, index],
            Rotation.from_matrix(plus[:3, :3]@minus[:3, :3].T).as_rotvec()/(2*eps),
            atol=1e-8)


def orientation_stub(monkeypatch, pitch=0., angular_velocity=0.):
    sim = osc_stub()
    sim.osc_task_kp = np.r_[[12000.]*3, [10.]*3]
    sim._tool_rotation_world = np.array([0., .1, 0.])
    sim._flip_tool_rotation = sim._tool_rotation_world.copy()
    down = Rotation.from_rotvec([np.pi, 0., 0.]).as_matrix()
    sim.orientation_d = down
    sim.get_end_effector_pos = lambda: (
        np.array([.4, 0., .4]), Rotation.from_rotvec([0., .1+pitch, 0.]).as_matrix()@down)
    sim.get_current_joint_velocity = lambda: np.array([0., 0., 0., 0., angular_velocity, 0., 0.])
    jac = np.zeros((6, 7))
    jac[:6, :6] = np.eye(6)
    monkeypatch.setattr(entry, '_franka_pose_jacobian_np', lambda q: jac)
    return sim


def test_opposite_pitch_errors_get_equal_opposite_restoring_torques(monkeypatch):
    positive = orientation_stub(monkeypatch, pitch=.1)
    tau_positive = positive._compute_osc_torques()
    negative = orientation_stub(monkeypatch, pitch=-.1)
    tau_negative = negative._compute_osc_torques()
    assert tau_positive[4] < 0. < tau_negative[4]
    np.testing.assert_allclose(tau_positive[3:], -tau_negative[3:], atol=1e-7)
    np.testing.assert_allclose(tau_positive[:3], tau_negative[:3], atol=1e-7)


def test_angular_damping_opposes_wrist_velocity_without_changing_translation(monkeypatch):
    sim = orientation_stub(monkeypatch, angular_velocity=.1)
    tau = sim._compute_osc_torques()
    assert tau[4] < 0.
    np.testing.assert_allclose(tau[:3], sim._last_osc_force, atol=1e-7)


def test_posture_does_not_fight_the_cartesian_attitude_task(monkeypatch):
    sim = orientation_stub(monkeypatch)
    sim.nullspace_stiffness = 10.
    sim.q_d_nullspace[4] = .5
    tau = sim._compute_osc_torques()
    assert tau[4] == pytest.approx(0., abs=1e-7)


def test_zero_orientation_gain_preserves_free_wrist_control(monkeypatch):
    sim = orientation_stub(monkeypatch, pitch=.1)
    sim.osc_task_kp[3:] = 0.
    tau = sim._compute_osc_torques()
    np.testing.assert_allclose(tau[3:], 0., atol=1e-7)


def test_pose_stage_keeps_existing_executor_and_flip_restores_attitude(monkeypatch):
    sim = orientation_stub(monkeypatch, pitch=.1)
    sim._object_cost_stage = 'pose'
    np.testing.assert_allclose(sim._compute_osc_torques()[3:], 0., atol=1e-7)
    sim._object_cost_stage = 'flip'
    assert sim._compute_osc_torques()[4] < 0.


def test_stage_metadata_changes_only_when_the_queued_action_is_admitted():
    sim = osc_stub()
    from examples.mpc.franka.runtime.bus import PolicyActionSchedule
    sim._policy_schedule = PolicyActionSchedule()
    sim._object_cost_stage = 'flip'
    sim.set_via_action([.405, 0., .4], action=[.005, 0., 0.], object_cost_stage='pose')
    assert sim._object_cost_stage == 'flip'
    assert sim._pending_via_meta['object_cost_stage'] == 'pose'
    meta = sim._pending_via_meta
    sim._start_via_action(meta['via'], action=[.005, 0., 0.],
                          press=meta['press'], object_cost_stage=meta['object_cost_stage'])
    assert sim._object_cost_stage == 'pose'


def test_flip_curl_mirrors_with_contact_side_and_has_no_fixed_world_x_bias():
    left = slew_tool_rotation(np.zeros(3), [0., -.2, 0.], 1.)
    right = slew_tool_rotation(np.zeros(3), [0., .2, 0.], 1.)
    assert left[1] < 0. < right[1]
    np.testing.assert_allclose(left, -right)
    yaw = Rotation.from_rotvec([0., 0., .7]).as_matrix()
    rotated = slew_tool_rotation(np.zeros(3), yaw@np.array([0., -.2, 0.]), 1.)
    np.testing.assert_allclose(rotated, yaw@left, atol=1e-10)


def test_flip_curl_is_angle_and_rate_limited_and_releases_smoothly():
    ref = np.zeros(3)
    for _ in range(1000):
        nxt = slew_tool_rotation(ref, [0., -.35, 0.], .002, max_angle=np.deg2rad(20.))
        assert np.linalg.norm(nxt-ref) <= .002+1e-12
        assert np.linalg.norm(nxt) <= np.deg2rad(20.)+1e-12
        ref = nxt
    released = slew_tool_rotation(ref, np.zeros(3), .002)
    assert np.linalg.norm(released) < np.linalg.norm(ref)
    assert np.linalg.norm(released-ref) <= .002+1e-12


def test_no_planned_preparation_leaves_wrist_free(monkeypatch):
    sim = orientation_stub(monkeypatch, pitch=.1)
    sim._tool_rotation_world = np.zeros(3)
    np.testing.assert_allclose(sim._compute_osc_torques()[3:], 0., atol=1e-7)


@pytest.mark.parametrize('tilt', [0., .4, -.3, np.pi])
def test_preparation_removes_placement_tilt_and_rotates_with_world_yaw(tilt):
    placed = Rotation.from_euler('xyz', [np.pi, tilt, .2]).as_matrix()
    down = downward_tool_frame(placed)
    np.testing.assert_allclose(down[:, 2], [0., 0., -1.], atol=1e-9)
    yaw = Rotation.from_rotvec([0., 0., .8]).as_matrix()
    np.testing.assert_allclose(downward_tool_frame(yaw@placed), yaw@down, atol=1e-9)
    np.testing.assert_allclose(down.T@down, np.eye(3), atol=1e-9)


@pytest.mark.parametrize('prepared', [False, True])
@pytest.mark.parametrize('phase', ['prepare', 'hook'])
def test_tilt_tracking_cannot_zero_the_mpc_translation(prepared, phase):
    sim = osc_stub()
    sim.osc_task_kp = np.r_[[12000.]*3, [10.]*3]
    sim._fingertip_in_physical_contact = lambda: False
    down = Rotation.from_rotvec([np.pi, 0., 0.]).as_matrix()
    offset = np.array([0., .2, 0.])
    sim.R_d_hold = down
    current = Rotation.from_rotvec(offset if prepared else -offset).as_matrix()@down
    sim.get_end_effector_pos = lambda: (np.array([.4, 0., .4]), current)
    sim._last_mpc_action = None
    sim._start_via_action([.405, 0., .4], action=[.005, 0., 0.],
        tool_rotation_world=offset, approach_phase=phase, object_cost_stage='flip')
    assert np.linalg.norm(sim._mpc_action) == pytest.approx(.005, abs=1e-8)


def test_spherical_tool_twist_does_not_block_or_drive_the_tilt_task(monkeypatch):
    sim = orientation_stub(monkeypatch)
    down = Rotation.from_rotvec([np.pi, 0., 0.]).as_matrix()
    target = Rotation.from_rotvec(sim._tool_rotation_world).as_matrix()@down
    twisted = target@Rotation.from_rotvec([0., 0., .7]).as_matrix()
    sim.get_end_effector_pos = lambda: (np.array([.4, 0., .4]), twisted)
    np.testing.assert_allclose(sim._compute_osc_torques()[3:], 0., atol=1e-7)
    sim._fingertip_in_physical_contact = lambda: False
    sim.R_d_hold = down
    sim._last_mpc_action = None
    sim._start_via_action([.405, 0., .4], action=[.005, 0., 0.],
        tool_rotation_world=sim._tool_rotation_world, approach_phase='prepare')
    assert sim._mpc_action[0] == pytest.approx(.005)


def test_unvalidated_force_entry_is_opt_in():
    import argparse
    parser=entry._add_rollout_policy_args(argparse.ArgumentParser())
    assert not parser.parse_args([]).force_aligned_approach
    assert parser.parse_args(['--force-aligned-approach', 'true']).force_aligned_approach


@pytest.mark.parametrize('angle',[-.02,.02])
def test_mpc_rotation_increment_drives_wrist_without_changing_xyz(monkeypatch,angle):
    sim=orientation_stub(monkeypatch)
    sim._object_cost_stage='flip'
    sim._tool_rotation_world=np.zeros(3)
    sim._tool_rotation_action=np.array([0.,angle,0.])
    sim._tool_rotation_target=Rotation.from_rotvec(sim._tool_rotation_action).as_matrix()
    tau=sim._compute_osc_torques()
    assert tau[4]==pytest.approx(10.*angle)
    np.testing.assert_allclose(tau[:3],sim._last_osc_force,atol=1e-7)


def test_mpc_rotation_is_admitted_with_xyz_at_policy_boundary():
    from examples.mpc.franka.runtime.bus import PolicyActionSchedule
    sim=osc_stub();sim._policy_schedule=PolicyActionSchedule()
    angular=np.array([.01,-.02,.005]);xyz=np.array([.005,0.,0.])
    sim.set_via_action([.405,0.,.4],action=xyz,tool_rotation_action=angular)
    assert getattr(sim,'_tool_rotation_action',None) is None
    meta=sim._pending_via_meta
    sim._start_via_action(meta['via'],action=xyz,tool_rotation_action=meta['tool_rotation_action'])
    np.testing.assert_allclose(sim._tool_rotation_action,angular)
    np.testing.assert_allclose(sim._mpc_action,xyz)
    np.testing.assert_allclose(sim._tool_rotation_target,Rotation.from_rotvec(angular).as_matrix())


@pytest.mark.parametrize('obj,backend,explicit,expected',[
    ('elephant','isaac',None,.01),('stanford_bunny2','isaac',None,.03),
    ('stanford_bunny2','isaac',0.,0.),('stanford_bunny2','mjwarp',None,10.),
    ('elephant','mjwarp',None,10.),('elephant','isaac',10.,10.),
    ('elephant','isaac',0.,0.),('elephant','mjwarp',0.,0.),
])
def test_passive_wrist_default_and_explicit_posture_override(obj,backend,explicit,expected):
    import argparse
    from examples.mpc.franka.runtime.contact import resolve_nullspace_stiffness
    args=entry._add_rollout_policy_args(argparse.ArgumentParser()).parse_args(
        ['--obj',obj,'--physics-backend',backend]+
        ([] if explicit is None else ['--nullspace-stiffness',str(explicit)]))
    assert args.tool_orientation_control=='free'
    assert resolve_nullspace_stiffness(args)==expected


def test_inertia_scaled_posture_damping_is_dissipative_and_task_free():
    from examples.mpc.franka.runtime.contact import franka_nullspace_posture_torque
    rng=np.random.default_rng(42)
    jac=rng.normal(size=(3,7));a=rng.normal(size=(7,7));mass=a@a.T+.01*np.eye(7)
    for _ in range(20):
        velocity=rng.normal(size=7)
        torque=franka_nullspace_posture_torque(jac,np.zeros(7),velocity,np.zeros(7),.03,mass_matrix=mass)
        assert velocity@torque<=1e-7
        np.testing.assert_allclose(jac@torque,0.,atol=2e-6)


def test_lighter_wrist_gets_less_damping_and_zero_regularization_releases_it():
    from examples.mpc.franka.runtime.contact import franka_nullspace_posture_torque
    jac=np.c_[np.eye(3),np.zeros((3,4))];velocity=np.r_[np.zeros(3),np.ones(4)]
    mass=np.diag([1.,1.,1.,1.,.01,.01,.001])
    torque=franka_nullspace_posture_torque(jac,np.zeros(7),velocity,np.zeros(7),.03,mass_matrix=mass)
    assert abs(torque[-1])<abs(torque[4])<abs(torque[3])
    # Scaling inertia by 4 scales damping by 2, with no restoring error.
    doubled=franka_nullspace_posture_torque(jac,np.zeros(7),velocity,np.zeros(7),.03,mass_matrix=4*mass)
    np.testing.assert_allclose(doubled,2*torque,atol=1e-7)
    np.testing.assert_allclose(franka_nullspace_posture_torque(jac,np.ones(7),velocity,np.zeros(7),0.,mass_matrix=mass),0.)


def test_inertia_damping_does_not_brake_a_pure_position_task_velocity():
    from examples.mpc.franka.runtime.contact import franka_nullspace_posture_torque
    rng=np.random.default_rng(12);jac=rng.normal(size=(3,7));a=rng.normal(size=(7,7))
    mass=a@a.T+.001*np.eye(7);velocity=jac.T@np.array([1.,-.5,.7])
    torque=franka_nullspace_posture_torque(jac,np.zeros(7),velocity,np.zeros(7),.03,mass_matrix=mass)
    np.testing.assert_allclose(torque,0.,atol=1e-6)


def test_isaac_executor_uses_measured_inertia_for_weak_posture_damping(monkeypatch):
    sim=orientation_stub(monkeypatch,angular_velocity=.1)
    sim._tool_rotation_world=np.zeros(3);sim.nullspace_stiffness=.03
    sim._get_arm_mass_matrix=lambda:np.diag([1.,1.,1.,.02,.01,.01,.001])
    before=sim._compute_osc_torques()
    sim.param_.nullspace_inertia_damping_=True
    after=sim._compute_osc_torques()
    assert abs(after[4])<abs(before[4])
    assert after[4]<0.
    np.testing.assert_allclose(after[:3],before[:3],atol=1e-7)


@pytest.mark.parametrize('backend,explicit,expected', [
    ('isaac', None, 'legacy'),
    ('isaac', 'legacy', 'legacy'),
    ('isaac', 'projected-inertia', 'projected-inertia'),
    ('mjwarp', None, 'legacy'),
])
def test_bunny_damping_choice_reaches_the_executor(monkeypatch, backend, explicit, expected):
    import argparse
    from types import SimpleNamespace
    args = entry._add_rollout_policy_args(argparse.ArgumentParser()).parse_args(
        ['--obj', 'stanford_bunny2', '--physics-backend', backend]
        + ([] if explicit is None else ['--nullspace-damping-model', explicit]))
    param = entry._configure_rollout_param(
        SimpleNamespace(attract_coef=.5, contact_coef=.7, lambda_optimizer=None), args)
    sim = orientation_stub(monkeypatch, angular_velocity=.1)
    sim.param_ = param
    sim._tool_rotation_world = np.zeros(3)
    sim.nullspace_stiffness = .03
    sim.q_d_nullspace = sim.get_current_joint_position().copy()
    sim._get_arm_mass_matrix = lambda: np.diag([1., 1., 1., .02, .01, .01, .001])
    torque = sim._compute_osc_torques()
    damping = 2.*np.sqrt(.03*(.01 if expected == 'projected-inertia' else 1.))
    assert torque[4] == pytest.approx(-damping*.1)
