import numpy as np

from planning.grasp_verification import verify_grasp


def opposing_triangle():
    return (np.array([[.04, 0., 0.], [-.04, .03, 0.], [-.04, -.03, 0.]]),
            np.array([[-1., 0., 0.], [1., 0., 0.], [1., 0., 0.]]))


def test_three_frictional_contacts_can_close_all_six_wrench_directions():
    points, normals = opposing_triangle()
    result = verify_grasp(points, normals, .8,
        external_wrench=[0., 0., -.3, 0., 0., 0.], force_limit=2.)
    assert result["wrench_rank"] == 6 and result["force_closure"]
    assert result["positive_span_margin"] > 0
    assert result["load_feasible"] and result["load_residual"] < 1e-10


def test_two_point_contacts_cannot_resist_rotation_about_their_connecting_line():
    points, normals = opposing_triangle()
    result = verify_grasp(points[:2], normals[:2], 2.5)
    assert result["wrench_rank"] == 5 and not result["force_closure"]


def test_gravity_support_alone_does_not_certify_force_closure():
    points, _ = opposing_triangle()
    result = verify_grasp(points, np.tile([0., 0., 1.], (3, 1)), .8,
        external_wrench=[0., 0., -.3, 0., 0., 0.], force_limit=2.)
    assert result["load_feasible"]
    assert not result["force_closure"]


def test_closure_does_not_imply_sufficient_bounded_load_capacity():
    points, normals = opposing_triangle()
    result = verify_grasp(points, normals, .8,
        external_wrench=[0., 0., -100., 0., 0., 0.], force_limit=.1)
    assert result["force_closure"] and not result["load_feasible"]


def test_collinear_contacts_and_empty_contacts_are_not_force_closure():
    points, normals = opposing_triangle()
    points[:, 1] = 0.
    assert not verify_grasp(points, normals, 2.5)["force_closure"]
    assert not verify_grasp(np.empty((0, 3)), np.empty((0, 3)), 2.5)["force_closure"]
