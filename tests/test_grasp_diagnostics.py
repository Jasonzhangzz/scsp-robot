"""Independent wrench balance and lift-capacity diagnostics."""
import mujoco
import numpy as np
import pytest

from planning.grasp_diagnostics import object_contact_wrenches, bounded_vertical_acceleration


def test_contact_wrench_matches_mujoco_generalized_reaction():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option cone="elliptic"/>
        <worldbody><geom name="floor" type="plane" size="1 1 .1"/>
        <body name="object" pos="0 0 .019"><freejoint/><geom type="sphere" size=".02" mass=".1"/></body>
        </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = model.body('object').id
    totals, contacts = object_contact_wrenches(model, data, body, [])
    assert len(contacts) == 1
    assert totals['support'][2] > 0
    np.testing.assert_allclose(totals['all'][:3], data.qfrc_constraint[:3], atol=1e-10)
    np.testing.assert_allclose(totals['all'][3:], data.xmat[body].reshape(3, 3)@data.qfrc_constraint[3:], atol=1e-10)
    np.testing.assert_array_equal(totals['pads'], 0)


@pytest.mark.parametrize('normal,expected', [(.01, False), (.5, True)])
def test_geometric_closure_does_not_imply_available_lift_force(normal, expected):
    points = np.array([[.04, .02, 0], [.04, -.02, 0], [-.04, .02, 0], [-.04, -.02, 0]])
    normals = np.array([[-1., 0, 0]]*2+[[1., 0, 0]]*2)
    mass = np.diag([.1]*3+[.001]*3)
    report = bounded_vertical_acceleration(points, normals, .7, mass,
        [0, 0, -.981, 0, 0, 0], [normal]*4)
    assert report['force_closure']
    assert report['feasible']
    assert (report['max_upward_acceleration'] > 0) is expected


def test_no_hand_contact_has_only_free_fall_capacity():
    report = bounded_vertical_acceleration([], [], .7, np.diag([.1]*3+[.001]*3),
        [0, 0, -.981, 0, 0, 0], [])
    assert report['feasible']
    assert not report['force_closure']
    assert report['max_upward_acceleration'] == pytest.approx(-9.81)
