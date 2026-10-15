"""Virtual work and gap transport for a multi-DOF grasp/contact horizon."""
import casadi as cs
import numpy as np
from scipy.spatial.transform import Rotation

from models.contact_geometry import joint_contact_geometry_cs


def _pose(rotation, joints):
    return np.r_[[.4, -.1, .3], rotation.as_quat()[[3, 0, 1, 2]], joints]


def _block(rotation, lever, directions, point_jac):
    return np.c_[directions, np.cross(lever, directions@rotation), -directions@point_jac]


def test_joint_patch_rotation_preserves_world_virtual_work():
    rng = np.random.default_rng(42)
    nq = 22
    observed = Rotation.from_rotvec([.2, -.3, .1])
    predicted = Rotation.from_rotvec([-.4, .2, .3])
    reference = _pose(observed, rng.normal(size=nq))
    state = _pose(predicted, reference[7:]+rng.normal(size=nq)*.01)
    state[:3] += [.003, -.002, .005]
    frame = Rotation.from_rotvec([.1, .3, -.2]).as_matrix()
    n, t1, t2 = frame.T
    directions = np.array([n+.7*t1, n+.7*t2, n-.7*t1, n-.7*t2])
    lever = np.array([.02, -.03, .01])
    point_jac = rng.normal(size=(3, nq))*.03
    block = _block(observed.as_matrix(), lever, directions, point_jac)
    x = cs.SX.sym('x', 7+nq)
    gap, rows = joint_contact_geometry_cs(x, cs.DM(reference), cs.DM(np.full(4, -.001)), cs.DM(block))
    fn = cs.Function('geometry', [x], [gap, rows])
    actual_gap, actual_rows = map(np.asarray, fn(state))
    transported = directions@observed.as_matrix()@predicted.as_matrix().T
    expected = _block(predicted.as_matrix(), lever, transported, point_jac)
    np.testing.assert_allclose(actual_rows, expected, atol=1e-12)
    point_delta = state[:3]-reference[:3]+(predicted.as_matrix()-observed.as_matrix())@lever
    normal = transported.mean(0)
    expected_gap = -.001+normal@(point_delta-point_jac@(state[7:]-reference[7:]))
    np.testing.assert_allclose(actual_gap.ravel(), expected_gap, atol=1e-12)
    measured_gap, measured_rows = map(np.asarray, fn(reference))
    np.testing.assert_allclose(measured_rows, block, atol=1e-12)
    np.testing.assert_allclose(measured_gap.ravel(), -.001, atol=1e-12)
    # Quaternion sign cannot alter contact gaps or force/wrench mapping.
    state[3:7] *= -1
    flipped_gap, flipped_rows = map(np.asarray, fn(state))
    np.testing.assert_allclose(flipped_gap, actual_gap, atol=1e-12)
    np.testing.assert_allclose(flipped_rows, actual_rows, atol=1e-12)


def test_support_and_robot_only_rows_are_not_rotated_with_the_object():
    nq = 5
    reference = _pose(Rotation.identity(), np.zeros(nq))
    state = _pose(Rotation.from_rotvec([0., .3, .2]), np.ones(nq)*.02)
    state[2] += .01
    directions = np.array([[.4, 0, 1], [0, .4, 1], [-.4, 0, 1], [0, -.4, 1]])
    lever = np.array([.03, .01, -.02])
    support = _block(np.eye(3), lever, directions, np.zeros((3, nq)))
    robot_only = np.zeros_like(support)
    robot_only[:, 6:] = [.1, -.2, .3, -.4, .5]
    blocks = np.r_[support, robot_only, np.zeros_like(support)]
    x = cs.SX.sym('x', 7+nq)
    gap, rows = joint_contact_geometry_cs(x, cs.DM(reference), cs.DM(np.r_[np.zeros(8), np.ones(4)]), cs.DM(blocks))
    actual_gap, actual_rows = map(np.asarray, cs.Function('geometry_other', [x], [gap, rows])(state))
    np.testing.assert_allclose(actual_rows[:4, :3], directions, atol=1e-12)
    rotation = Rotation.from_quat(state[[4, 5, 6, 3]]).as_matrix()
    expected_gap = state[2]-reference[2]+((rotation-np.eye(3))@lever)[2]
    np.testing.assert_allclose(actual_gap[:4].ravel(), expected_gap, atol=1e-12)
    np.testing.assert_allclose(actual_rows[4:8], robot_only, atol=1e-12)
    np.testing.assert_allclose(actual_gap[4:8].ravel(), robot_only[0, 6:]@state[7:], atol=1e-12)
    np.testing.assert_array_equal(actual_rows[8:], 0.)
    np.testing.assert_array_equal(actual_gap[8:], 1.)
