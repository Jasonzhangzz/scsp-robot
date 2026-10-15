import numpy as np

from planning.grasp_force_allocation import allocate_contact_wrench


def test_live_allocation_retains_preload_and_required_tangential_lift():
    points = np.array([[.04, 0, 0], [-.04, .03, 0], [-.04, -.03, 0]])
    normals = np.array([[-1., 0, 0], [1., 0, 0], [1., 0, 0]])
    wrench = np.array([0., 0, .5, .002, -.001, 0.])
    result = allocate_contact_wrench(points, normals, wrench, 1., minimum_normal=.2,
        force_limit=2., friction_margin=.8)
    assert result['feasible']
    forces = result['forces']
    np.testing.assert_allclose(np.r_[forces.sum(0), np.cross(points, forces).sum(0)], wrench, atol=1e-7)
    normal = np.sum(normals*forces, axis=1)
    tangent = forces-normal[:, None]*normals
    assert np.all(normal >= .2-1e-7)
    assert np.all(np.linalg.norm(tangent, axis=1) <= .8*normal+1e-7)
    assert np.all(np.linalg.norm(forces, axis=1) <= 2.+1e-7)
    assert forces[:, 2].sum() == .5


def test_insufficient_friction_cannot_be_reported_as_a_lifting_wrench():
    result = allocate_contact_wrench([[.04, 0, 0], [-.04, 0, 0]],
        [[-1, 0, 0], [1, 0, 0]], [0, 0, .5, 0, 0, 0], 0.,
        minimum_normal=.2, force_limit=2.)
    assert not result['feasible']
    assert result['forces'] is None


def test_refinement_or_fallback_respects_a_small_force_limit():
    result = allocate_contact_wrench([[.04, 0, 0], [-.04, .03, 0], [-.04, -.03, 0]],
        [[-1, 0, 0], [1, 0, 0], [1, 0, 0]], [0, 0, .15, 0, 0, 0], .9,
        minimum_normal=.03, force_limit=.15)
    assert result['feasible']
    assert np.max(np.linalg.norm(result['forces'], axis=1)) <= .15+1e-7


def test_available_force_and_requested_preload_are_distinct():
    points = [[.04, 0, 0], [-.04, .03, 0], [-.04, -.03, 0]]
    normals = [[-1, 0, 0], [1, 0, 0], [1, 0, 0]]
    wrench = [0, 0, .8, 0, 0, 0]
    available = allocate_contact_wrench(points, normals, wrench, .7,
        minimum_normal=0., normal_limits=[.05]*3, force_limit=2.)
    assert not available['feasible']
    requested = allocate_contact_wrench(points, normals, wrench, .7,
        minimum_normal=.2, force_limit=2.)
    assert requested['feasible']
    assert np.max(requested['normal_forces']) > .6
    np.testing.assert_allclose(requested['forces'].sum(0), [0, 0, .8], atol=1e-7)


def test_held_object_motion_is_not_cancelled_as_stationary_object_opening():
    from planning.grasp_force_allocation import normal_pressure_command
    jac = np.tile(np.c_[np.eye(3), np.zeros((3, 19))], (3, 1))
    for i in range(3):
        jac[3*i:3*i+3, 6+3*i:9+3*i] = np.eye(3)
    normals = [[1., 0, 0], [-1., 0, 0], [-1., 0, 0]]
    action = np.zeros(22); action[0] = .001
    held, bias = normal_pressure_command(jac, normals, [.6]*3, [.6]*3, .0003, .001,
        grasp_established=True)
    np.testing.assert_allclose(held@action+bias, action, atol=1e-12)
    static, _ = normal_pressure_command(jac, normals, [.6]*3, [.6]*3, .0003, .001)
    np.testing.assert_allclose((jac@(static@action)).reshape(3, 3)[:, 0], 0., atol=1e-8)


def test_native_refinement_retains_diamond_and_actual_tangent_orientation():
    points=np.zeros((1,3));normal=[[0.,0.,1.]]
    basis=np.array([[[1.,0,0],[0,1.,0]]])
    # This diagonal force is inside the circular cone, outside the native
    # diamond when Fn is capped at one. Refinement must not escape the LP cone.
    for cone,expected in [('coulomb',True),('pyramidal',False)]:
        result=allocate_contact_wrench(points,normal,[.6,.6,1,0,0,0],1.,
            force_limit=3.,normal_limits=[1.],friction_margin=1.,
            friction_cone=cone,tangent_directions=basis)
        assert result['feasible']==expected
    rotated=np.array([[[1.,1.,0],[-1.,1.,0]]])/np.sqrt(2.)
    result=allocate_contact_wrench(points,normal,[.6,.6,1,0,0,0],1.,
        force_limit=3.,normal_limits=[1.],friction_margin=1.,
        friction_cone='pyramidal',tangent_directions=rotated)
    assert result['feasible']
    force=result['forces'][0]
    assert np.abs(rotated[0]@force).sum()<=force[2]+1e-7
