import numpy as np
import pytest

from planning.dywa_interface import (
    _iterative_closest_point,
    _make_evaluation_env,
    _validate_required_config,
    look_at_rotation_and_translation,
    map_poses_to_table,
    place_pair_on_table,
    runtime_environment,
    validate_reset_state,
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


def test_table_mapping_preserves_both_clearances_and_the_relative_task():
    initial = np.array([0.36, 0.04, 0.392, 1.0, 0.0, 0.0, 0.0])
    target = np.array([0.40, 0.10, 0.373])
    mapped_initial, mapped_target = map_poses_to_table(initial, target, 0.35)

    assert mapped_initial[2] - 0.4 == pytest.approx(initial[2] - 0.35)
    assert mapped_target[2] - 0.4 == pytest.approx(target[2] - 0.35)
    assert np.allclose(mapped_target - mapped_initial[:3], target - initial[:3])
    assert np.array_equal(mapped_initial[3:], initial[3:])
    assert np.allclose(initial[:3], [0.36, 0.04, 0.392])


def test_table_mapping_uses_the_configured_table_center_and_height():
    initial = [0.36, 0.04, 0.392, 1.0, 0.0, 0.0, 0.0]
    target = [0.40, 0.10, 0.373]
    mapped_initial, mapped_target = map_poses_to_table(
        initial, target, 0.35, table_pos=(0.1, -0.2, 0.3), table_dims=(0.4, 0.5, 0.4)
    )
    assert np.allclose((mapped_initial[:2] + mapped_target[:2]) / 2, (0.1, -0.2))
    assert mapped_initial[2] == pytest.approx(0.542)
    assert mapped_target[2] == pytest.approx(0.523)


def test_table_mapping_rejects_a_task_that_would_be_silently_scaled():
    with pytest.raises(ValueError, match="translation does not fit"):
        map_poses_to_table([0, 0, 0.4, 1, 0, 0, 0], [0, 0.8, 0.4], 0.35)


def test_icp_recovers_rotation_and_translation_instead_of_inverse_rotation():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(42)
    source = 0.03 * torch.randn(2, 100, 3, dtype=torch.float64, generator=generator)
    angle = np.deg2rad(5)
    rotation = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ], dtype=source.dtype).expand(2, 3, 3)
    translation = torch.tensor([[0.001, -0.001, 0.002], [-0.002, 0.001, 0.001]], dtype=source.dtype)
    target = source @ rotation + translation[:, None]
    solution = _iterative_closest_point(source, target, max_iterations=50)

    assert solution.converged
    assert torch.allclose(solution.Xt, target, atol=1e-9)
    assert torch.allclose(solution.RTs.R, rotation, atol=1e-9)
    assert torch.allclose(solution.RTs.T, translation, atol=1e-9)
    assert solution.rmse.shape == (2,)
    assert torch.all(solution.rmse < 1e-9)


def test_reset_check_rejects_unmapped_height_and_quaternion_order():
    spec = {
        "init_obj_qpos": [0, 0, 0.442, 1, 0, 0, 0],
        "target_p": [0.04, 0.06, 0.423],
        "target_q": [1, 0, 0, 0],
    }
    settled_initial = np.array([0, 0, 0.440, 0, 0, 0, 1])
    goal = np.array([0.04, 0.06, 0.423, 0, 0, 0, 1])
    validate_reset_state(settled_initial, goal, spec)
    old_goal = goal.copy()
    old_goal[2] -= 0.05
    with pytest.raises(RuntimeError, match="goal pose"):
        validate_reset_state(settled_initial, old_goal, spec)
    with pytest.raises(RuntimeError, match="initial pose"):
        validate_reset_state(spec["init_obj_qpos"], goal, spec)


def test_runtime_environment_preserves_paths_and_does_not_reexec_forever(tmp_path):
    python = tmp_path / "acc/bin/python"
    original = {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/opt/cuda/lib64"}
    updated = runtime_environment(python, original)
    assert updated["PATH"].split(":") == [str(python.parent), "/usr/bin"]
    assert updated["LD_LIBRARY_PATH"].split(":") == [str(tmp_path / "acc/lib"), "/opt/cuda/lib64"]
    assert runtime_environment(python, updated) == updated
    assert original == {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/opt/cuda/lib64"}


def test_unmodified_dywa_config_is_rejected_with_missing_hooks():
    from types import SimpleNamespace

    cfg = SimpleNamespace(env=SimpleNamespace(
        single_object_scene=SimpleNamespace(mesh=SimpleNamespace(density=300)),
        task=SimpleNamespace(), franka=SimpleNamespace(),
    ))
    with pytest.raises(RuntimeError, match="use_custom_end_effector"):
        _validate_required_config(cfg)


def test_missing_source_and_weights_are_reported_before_starting_child(tmp_path, monkeypatch):
    import planning.dywa_interface as interface

    monkeypatch.setattr(interface, "DYWA_SRC", tmp_path / "src")
    monkeypatch.setattr(interface, "DYWA_TRAIN", tmp_path / "train")
    monkeypatch.setattr(interface, "DYWA_CFG", tmp_path / "cfg")
    def unexpected_launch(*args, **kwargs):
        pytest.fail("child process should not start with missing runtime files")
    monkeypatch.setattr(interface.subprocess, "run", unexpected_launch)
    spec = {key: str(tmp_path / key) for key in (
        "mesh_path", "student_ckpt", "teacher_ckpt", "env_ckpt", "icp_ckpt"
    )}
    with pytest.raises(FileNotFoundError) as error:
        interface.execute_trial(spec)
    assert "DyWA environment source" in str(error.value)
    assert "student weights" in str(error.value)
    assert "environment statistics" in str(error.value)


def test_evaluation_reset_never_steps_random_actions():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    obs = {"partial_cloud": torch.zeros(1, 512, 3)}
    student_resets = []
    state_resets = []
    class Backend:
        num_env = 1
        device = "cpu"
        def reset(self):
            return obs
        def step(self, *args):
            pytest.fail("evaluation reset must not execute any actions")

    class TrainingWrapper:
        def __init__(self, cfg, env, student):
            self.env = env
            self.student_module = student
            self.num_env = env.num_env
            self.device = env.device
        def reset(self):
            pytest.fail("training mixed_reset must not be used in evaluation")

    student = SimpleNamespace(reset=student_resets.append, reset_state=state_resets.append)
    env = _make_evaluation_env(TrainingWrapper, None, Backend(), student)
    assert env.reset() is obs
    assert env.last_obs is obs
    assert student_resets == [obs]
    assert len(state_resets) == 1
    assert torch.equal(state_resets[0], torch.tensor([False]))
