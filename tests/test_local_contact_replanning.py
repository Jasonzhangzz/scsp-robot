import numpy as np

from planning.local_contact_replanning import choose_local_contact_updates


def _grasp(touching, reachable, caps=None, nearest_normals=None):
    points = np.array([[-.04, -.02, 0], [-.04, .02, 0], [.04, 0, -.02], [.04, 0, .02]])
    normals = np.array([[1., 0, 0], [1., 0, 0], [-1., 0, 0], [-1., 0, 0]])
    old = points + [0., .01, 0.]
    return old, points, choose_local_contact_updates(old, normals, points,
        normals if nearest_normals is None else nearest_normals, touching, reachable,
        [.6]*4 if caps is None else caps, .7, [0., 0., -.3, 0., 0., 0.])


def test_nearby_finger_can_close_grasp_without_waiting_for_fourth_finger():
    old, actual, (points, _, chosen, certificate) = _grasp([1, 1, 0, 0], [0, 0, 1, 0])
    np.testing.assert_array_equal(chosen, [2])
    np.testing.assert_array_equal(points[:3], actual[:3])
    np.testing.assert_array_equal(points[3], old[3])
    assert certificate['force_closure'] and certificate['load_feasible']


def test_distant_planned_finger_is_not_a_phantom_support():
    # Two contacts on one side fail even though the four planned points close.
    _, _, (_, _, chosen, certificate) = _grasp([1, 0, 0, 0], [0, 1, 0, 0])
    assert len(chosen) == 0 and certificate is None


def test_closure_without_available_load_capacity_is_rejected():
    _, _, (_, _, chosen, certificate) = _grasp([1, 1, 0, 0], [0, 0, 1, 0], caps=[.001]*4)
    assert len(chosen) == 0 and certificate is None


def test_two_nearby_fingers_may_be_needed_together():
    _, _, (_, _, chosen, certificate) = _grasp([1, 0, 0, 0], [0, 1, 1, 0])
    np.testing.assert_array_equal(chosen, [1, 2])
    assert certificate['force_closure']
