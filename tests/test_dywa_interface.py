import numpy as np

from planning.dywa_interface import (
    look_at_rotation_and_translation,
    place_pair_on_table,
    viewer_camera_for_scene,
)


def test_place_pair_keeps_direction_and_fits_the_table():
    init_xy = np.array([0.36, 0.04])
    target_xy = np.array([0.40, 0.10])
    placed_init, placed_target = place_pair_on_table(init_xy, target_xy)

    source_delta = target_xy - init_xy
    placed_delta = placed_target - placed_init
    assert np.dot(source_delta, placed_delta) > 0
    assert np.allclose(placed_init, -placed_target)
    assert np.max(np.abs(placed_init)) <= 0.35 + 1e-8
    assert np.max(np.abs(placed_target)) <= 0.35 + 1e-8


def test_place_pair_scales_a_translation_that_would_leave_the_table():
    placed_init, placed_target = place_pair_on_table(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]), half_extents=np.array([0.12, 0.35])
    )
    assert abs(float(placed_target[0] - placed_init[0]) - 0.24) < 1e-8
    assert abs(float(placed_target[1])) < 1e-8


def test_viewer_camera_matches_the_mpc_offset_from_the_robot_base():
    origin, target = viewer_camera_for_scene(
        table_pos=(0.0, 0.0, 0.2),
        table_dims=(0.4, 0.5, 0.4),
        keepout_radius=0.3,
        base_height="table",
    )
    base = np.array([-0.5, 0.0, 0.4])
    assert np.allclose(origin - base, (0.7, 0.0, 0.63))
    assert np.allclose(target - base, (0.1, 0.0, 0.32))


def test_policy_look_at_matches_pytorch3d_for_the_dywa_camera():
    rotation, translation = look_at_rotation_and_translation(
        eye=(0.458, -0.40, 0.740),
        at=(0.0, 0.0, 0.55),
        up=(0.0, 0.0, 1.0),
    )
    expected = np.array([
        [-0.65780580, -0.22462934, -0.71891111],
        [-0.75318766, 0.19618283, 0.62786996],
        [0.00000000, 0.95449150, -0.29823822],
    ])
    assert np.allclose(rotation[0], expected, atol=1e-6)
    assert np.allclose(translation[0], (0.0, -0.524970353, 0.801105559), atol=1e-6)
