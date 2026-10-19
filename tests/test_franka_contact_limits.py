from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from planning.contact_limits import contact_force_limit, contact_command_force_limit
from examples.mpc.franka.runtime.contact import mesh_table_contact_points, support_patch_jacobians


@pytest.mark.parametrize('cap,step,expected', [(10., .005, 3.), (.8, .005, .8), (10., .02, 10.)])
def test_prediction_force_is_reachable_by_executed_increment(cap, step, expected):
    args = SimpleNamespace(max_contact_force=cap, mpc_step_limit=step)
    assert contact_force_limit(args) == cap
    assert contact_command_force_limit(args, 600.) == pytest.approx(expected)


@pytest.mark.parametrize('cap', [0., -.8, np.nan, np.inf])
def test_invalid_force_cap_is_rejected(cap):
    with pytest.raises(ValueError):
        contact_force_limit(SimpleNamespace(max_contact_force=cap))


def test_table_support_preserves_real_mesh_pivot_and_body_frame():
    rotation = Rotation.from_euler('z', 35, degrees=True).as_matrix()
    origin = np.array([.4, .1, .4])
    # All low points lie to one side of the COM; projecting the COM onto the
    # table would lose the lever arm, and a high ear is not a support contact.
    vertices = np.array([[.02, -.02, -.05], [.02, .02, -.05], [.04, -.02, -.05],
                         [.04, .02, -.05], [0., 0., .08]])
    points = mesh_table_contact_points(origin, rotation, vertices)
    assert len(points) == 4
    np.testing.assert_allclose(points[:, 2], .35)
    assert np.all(np.linalg.norm(points[:, :2] - origin[:2], axis=1) > .02)
    world, body = support_patch_jacobians(origin, rotation, points[0], 9, .5)
    n = world.mean(axis=0)
    assert np.linalg.norm(n[3:6]) > .02
    np.testing.assert_allclose(n[:3], [0., 0., 1.])
    np.testing.assert_allclose(world[:, 6:], 0.)
    transform = np.zeros((6, 6)); transform[:3, :3] = transform[3:, 3:] = rotation
    np.testing.assert_allclose(body, world[:, :6] @ transform)
