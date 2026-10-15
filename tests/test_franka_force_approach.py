import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from planning.franka_contact_motion import ForceAlignedApproachVia


def route(yaw=0., height=.4):
    r = Rotation.from_rotvec([0., 0., yaw]).as_matrix()
    via = ForceAlignedApproachVia(floor_z=.362)
    normal, force = r@np.array([-1., 0., 0.]), r@np.array([1., 0., .7])
    press, obj = r@np.array([.4, 0., height]), r@np.array([.5, 0., height])
    via.set_force_context(force, normal, active=True, touching=False)
    return via, r, force, normal, press, obj


@pytest.mark.parametrize('yaw', [0., np.pi/2, np.pi, -np.pi/2])
def test_precontact_arc_has_lambda_force_tangent_and_prepares_outward(yaw):
    via, r, force, normal, press, obj = route(yaw)
    tip = press+.05*normal
    phases, rotations = [], []
    for _ in range(150):
        _, nxt, phase = via.update(tip, obj, press-.01*normal, press,
                                  .45, .06, False, press)
        assert np.linalg.norm(nxt-tip) <= .005+1e-10
        assert nxt[2] >= .362-1e-10
        phases.append(phase)
        rotations.append(via.tool_rotation_world.copy())
        tip = nxt
        if via.progress == 1. and np.linalg.norm(tip-press) < .001:
            break
    assert 'prepare' in phases and 'hook' in phases
    tangent = via.arc[-1]-via.arc[-2]
    np.testing.assert_allclose(tangent/np.linalg.norm(tangent), force/np.linalg.norm(force))
    # All controls stay on the outside of the destination plane.
    assert np.all((via.arc-press)@normal >= -1e-10)
    assert via.arc[0, 2] < press[2]
    axis = np.cross([0., 0., 1.], normal)
    assert np.dot(rotations[0], axis) < 0.  # tilt outward before touching
    assert np.dot(rotations[-1], axis) > 0.  # curl during upward entry


def test_arc_and_attitude_are_equivariant_under_world_yaw():
    base, _, _, n, c, obj = route()
    other, r, _, rn, rc, robj = route(.8)
    a, b = c+.05*n, rc+.05*rn
    for _ in range(100):
        _, a, _ = base.update(a, obj, c-.01*n, c, .45, .06, False, c)
        _, b, _ = other.update(b, robj, rc-.01*rn, rc, .45, .06, False, rc)
        np.testing.assert_allclose(b, r@a, atol=1e-9)
        np.testing.assert_allclose(other.tool_rotation_world, r@base.tool_rotation_world, atol=1e-9)


def test_measured_touch_hands_off_without_restarting_the_approach():
    via, _, force, n, c, obj = route()
    via.update(c+.05*n, obj, c-.01*n, c, .45, .06, False, c)
    assert via.force_approach_active
    via.set_force_context(force, n, active=True, touching=True)
    use, _, phase = via.update(c, obj, c-.01*n, c, .45, .06, True, c)
    assert not use and not via.blocked and not via.force_approach_active
    assert phase == 'contact'


@pytest.mark.parametrize('active,force', [(False, [1.,0.,.7]), (True, [1.,0.,0.]), (True, [0.,0.,0.])])
def test_pose_or_nonlifting_force_does_not_create_an_upward_hook(active, force):
    via, _, _, n, c, obj = route()
    via.set_force_context(force, n, active=active, touching=False)
    via.update(c+.05*n, obj, c-.01*n, c, .45, .06, False, c)
    assert not via.force_approach_active
    np.testing.assert_allclose(via.tool_rotation_world, 0.)


def test_floor_clearance_prevents_an_infeasible_undercut():
    via, _, _, n, c, obj = route(height=.3625)
    via.update(c+.05*n, obj, c-.01*n, c, .45, .06, False, c)
    assert not via.force_approach_active


def test_rank_changes_do_not_turn_the_wrist_away_from_the_latched_arc():
    via, _, force, n, c, obj = route()
    tip = c+.05*n
    via.update(tip, obj, c-.01*n, c, .45, .06, False, c)
    controls = via.arc.copy()
    rotation = via.tool_rotation_world.copy()
    via.set_force_context([0., 1., .7], [0., -1., 0.], active=True, touching=False)
    via.update(tip, obj, c-.01*n, c, .45, .06, False, c)
    np.testing.assert_allclose(via.arc, controls)
    np.testing.assert_allclose(via.tool_rotation_world, rotation)
    curl = via.arc_curl_rotation.copy()
    via.set_force_context(-force, -n, active=True, touching=True)
    via.update(c, obj, c-.01*n, c, .45, .06, True, c)
    first = via.tool_rotation_world.copy()
    assert np.linalg.norm(first-rotation) < np.linalg.norm(curl-rotation)
    via.set_force_context([0., 1., .7], [0., -1., 0.], active=True, touching=True)
    via.update(c, obj, c-.01*n, c, .45, .06, True, c)
    np.testing.assert_allclose(via.tool_rotation_world, .8*rotation+.2*curl)
    for _ in range(10):
        via.update(c, obj, c-.01*n, c, .45, .06, True, c)
    np.testing.assert_allclose(via.tool_rotation_world, 0.)


def test_curl_rotation_produces_surface_velocity_along_the_force_tangent():
    via, _, _, n, c, obj = route()
    f = np.array([1., .4, .7])
    via.set_force_context(f, n, active=True, touching=False)
    via.update(c+.05*n, obj, c-.01*n, c, .45, .06, False, c)
    tangent = f-np.dot(f,n)*n
    velocity = np.cross(via.arc_curl_rotation, -n)
    np.testing.assert_allclose(velocity/np.linalg.norm(velocity),
                               tangent/np.linalg.norm(tangent))


def test_shallow_upward_force_gets_a_smaller_undercut_and_preparation():
    weak, _, _, n, c, obj = route()
    strong, _, _, _, _, _ = route()
    weak.set_force_context([1., 0., .08], n, active=True, touching=False)
    strong.set_force_context([1., 0., .7], n, active=True, touching=False)
    for via in [weak, strong]:
        via.update(c+.05*n, obj, c-.01*n, c, .45, .06, False, c)
        assert via.force_approach_active
    assert c[2]-weak.arc[0,2] < c[2]-strong.arc[0,2]
    assert np.linalg.norm(weak.arc_prepare_rotation) < np.linalg.norm(strong.arc_prepare_rotation)


def test_touch_during_the_hook_finishes_its_upward_segment_before_mpc_handoff():
    via, _, force, n, c, obj = route()
    tip=c+.05*n
    for _ in range(100):
        use, tip, phase = via.update(tip, obj, c-.01*n, c, .45, .06, False, c)
        if phase == 'hook' and via.progress >= .5:
            break
    controls=via.arc.copy()
    contact_height=tip[2]
    via.set_force_context([0., 1., .7], [0., -1., 0.], active=True, touching=True)
    use, tip, phase = via.update(tip, obj, c-.01*n, c, .45, .06, True, c)
    assert use and via.force_approach_active and phase == 'hook'
    np.testing.assert_allclose(via.arc, controls)
    for _ in range(100):
        use, tip, phase = via.update(tip, obj, c-.01*n, c, .45, .06, True, c)
        if phase == 'contact': break
    assert phase == 'contact' and not use and not via.force_approach_active
    assert tip[2] > contact_height+.001
