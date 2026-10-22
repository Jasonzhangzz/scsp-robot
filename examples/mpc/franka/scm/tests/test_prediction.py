"""Numerical contracts of the prediction benchmark; run with pi0 Python."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_env import activate_dependencies
activate_dependencies()

import numpy as np
import mujoco
import pytest
from fidelity_metrics import Moments, motion_metrics, contact_observations
from compare_prediction import (make_model, fresh_data, reference_mode, quaternion_angle,
                                backend_for, replay_states, MESH_SHAPES, mesh_world_vertices,
                                sample_states, argument_parser)


def options():
    return SimpleNamespace(device="cuda:0", contact_stiffness=.1, contact_damping=.001,
                           nconmax=128, njmax=2048, scm_regularization=1e-8, scm_motion="dynamic")


def test_contact_weighted_statistics_not_frame_weighted():
    stats = Moments()
    stats.add([1])
    stats.add([])
    stats.add([3, 5, 7])
    report = stats.report()
    assert report["count"] == 4
    assert report["mean"] == 4
    assert report["std"] == pytest.approx(np.sqrt(5))


def test_zero_direction_and_diagonal_mass_equivalence():
    assert motion_metrics(np.zeros(6), np.zeros(6), np.eye(6))["angle_deg"] is None
    a, b = np.arange(1, 7.), np.arange(6, 0, -1.)
    mass = np.arange(1, 7.)
    assert motion_metrics(a, b, mass) == motion_metrics(a, b, np.diag(mass))
    assert motion_metrics(a, -a, mass)["angle_deg"] == pytest.approx(180)


def test_mode_labels_use_reference_load_and_velocity():
    native = {"force": np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])}
    obs = {"relative_velocity": np.zeros((2, 6))}
    kwargs = dict(force_threshold=1e-4, slip_threshold=.001, normal_threshold=.01)
    assert reference_mode(native, obs, **kwargs) == "sticking"
    obs["relative_velocity"][0, 1] = .2
    assert reference_mode(native, obs, **kwargs) == "sliding"
    obs["relative_velocity"][0, 0] = .2
    assert reference_mode(native, obs, **kwargs) == "transition"
    native["force"][:] = 0
    assert reference_mode(native, obs, **kwargs) == "separation"


def test_quaternion_sign_does_not_change_orientation_error():
    q = np.array([1., 0, 0, 0])
    assert quaternion_angle(q, -q) == 0
    assert quaternion_angle(q, np.array([0., 1, 0, 0])) == pytest.approx(np.pi)


@pytest.mark.parametrize("shape", MESH_SHAPES)
def test_repository_mesh_starts_above_floor_and_saved_model_reloads(shape, tmp_path):
    model = make_model(shape, .5, .002, .15)
    data = fresh_data(model)
    assert model.nq == 7 and model.nv == 6
    assert model.body_mass[1] == pytest.approx(.15)
    assert np.all(model.body_inertia[1] > 0)
    assert mesh_world_vertices(model, data)[:, 2].min() == pytest.approx(.005)
    path = tmp_path / "model.xml"
    mujoco.mj_saveLastXML(str(path), model)
    restored = mujoco.MjModel.from_xml_path(str(path))
    np.testing.assert_allclose(restored.qpos0, model.qpos0, atol=1e-7)
    np.testing.assert_allclose(restored.body_inertia, model.body_inertia, rtol=1e-5)


def test_bunny_alias_and_explicit_mode_inputs():
    args = argument_parser().parse_args(["--shapes", "stanford_bunn2", "mug"])
    assert args.shapes == ["stanford_bunny2", "mug"]
    model = make_model("box", .5, .002, .15)
    settled = fresh_data(model)
    # A tilted free body exercises local-to-world angular velocity conversion.
    settled.qpos[3:7] = [np.cos(.3), np.sin(.3), 0, 0]
    rng = np.random.default_rng(42)
    for mode in ("separation", "sliding", "sticking"):
        samples = list(sample_states(model, settled, mode, 2, rng, .5, .15))
        for qpos, qvel, wrench, condition in samples:
            if mode == "sliding":
                assert .2 <= np.linalg.norm(qvel[:2]) <= 1.5
                rotation = np.zeros(9)
                mujoco.mju_quat2Mat(rotation, qpos[3:7])
                world_angular = rotation.reshape(3, 3) @ qvel[3:]
                np.testing.assert_allclose(world_angular[:2], 0, atol=1e-15)
            elif mode == "sticking":
                np.testing.assert_array_equal(qvel, np.zeros(6))
                assert 0 < np.linalg.norm(wrench[:2]) < .2 * .5 * .15 * 9.81
        if mode == "separation":
            assert samples[0][0][2] > settled.qpos[2] + .02
            assert samples[1][1][2] >= .2


def test_native_friction_power_frame_and_sign():
    model = make_model("sphere", .5, .002, .15)
    data = fresh_data(model)
    # Normal +z, tangents +x,+y. Force on object opposes +x slip.
    native = {"geom": np.array([[0, 1]]), "position": np.array([[0., 0, 0]]),
              "frame": np.array([[[0., 0, 1], [1, 0, 0], [0, 1, 0]]]),
              "gap": np.array([-.001]), "friction": np.array([[.5, .5, .01, .001, .001]]),
              "force": np.array([[2., -1, 0, 0, 0, 0]])}
    obs = contact_observations(model, data, np.array([1., 0, 0, 0, 0, 0]), native)
    assert obs["power"][0] == -1
    assert obs["depth"][0] == .001
    assert obs["utilization"][0, 0] == 1


@pytest.mark.parametrize("motion", ["dynamic", "increment"])
@pytest.mark.parametrize("engine", ["scm_corrected", "scm_soft", "scm_soft_corrected"])
@pytest.mark.parametrize("shape", ["sphere", "box"])
def test_gpu_scm_matches_numpy_diagonal_formula_and_reset(motion, engine, shape):
    import warp as wp
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA required by pinned upstream")
    model = make_model(shape, .5, .002, .15)
    state = fresh_data(model)
    state.qpos[2] = (.04 if shape == "sphere" else .025) - .00005
    state.qvel[:] = [-.3, .1, -.02, 0, 0, .1]
    mujoco.mj_forward(model, state)
    args = options()
    args.scm_motion = motion
    backend = backend_for(engine, model, state, args)
    query = fresh_data(model, state.qpos, state.qvel)
    backend.reset(query)
    backend.step(query)
    count = int(backend.wdata.nefc.numpy()[0])
    assert count == (4 if shape == "sphere" else 16)
    J = backend.wdata.efc.J.numpy()[0, :count, :model.nv].astype(float)
    gap = backend.wdata.efc.efc_dist.numpy()[0, :count].astype(float)
    vf = backend.wdata.qvel_smooth_pred.numpy()[0, :model.nv].astype(float)
    mass = np.zeros((6, 6))
    mujoco.mj_fullM(model, mass, state.qM)
    inverse = np.linalg.inv(mass)
    diagonal = np.einsum("ij,jk,ik->i", J, inverse, J)
    # The GPU receives float32 initial velocity. Subtract that exact incoming
    # value: float64-vs-float32 cancellation can fabricate tiny angular impulses.
    incoming = np.asarray(state.qvel, dtype=np.float32).astype(float)
    projected = vf if motion == "dynamic" else vf - incoming
    distance = np.maximum(gap, 0.) if engine == "scm_soft_corrected" else gap
    g = J @ projected + distance / .002
    if engine == "scm_soft":
        x = .002 / .05
        denominator = 1 + 2 * x + x * x
        a, b = (2 * x + x * x) / denominator, x * x / denominator
        g = a * (J @ projected) + b * gap / .002
    p = np.maximum(-g / (diagonal + 1e-8), 0)
    if engine == "scm_soft_corrected":
        temperature = 1e-4
        p = temperature * np.logaddexp(0., -g / ((diagonal + 1e-8) * temperature))
        assert backend.api.response_time == 0.
        assert backend.api.softplus_temperature == temperature
    if engine in ("scm_corrected", "scm_soft_corrected"):
        W = J @ inverse @ J.T + 1e-8 * np.eye(count)
        denominator = p @ W @ p
        alpha = float(np.clip(-(g @ p) / denominator, 0, 1)) if denominator > 0 else 1.
        assert backend.last_amplitude_scale == pytest.approx(alpha, rel=3e-6, abs=1e-7)
        original_objective = .5 * p @ W @ p + g @ p
        p *= alpha
        assert .5 * p @ W @ p + g @ p <= original_objective + 1e-10
    np.testing.assert_allclose(backend.wdata.efc.force.numpy()[0, :count] * .002, p, rtol=2e-6, atol=1e-9)
    np.testing.assert_allclose(query.qvel, vf + inverse @ J.T @ p, rtol=2e-6, atol=1e-7)
    assert backend.last_contacts["force"][:, 0].sum() == pytest.approx(p.sum() / .002, rel=3e-6)
    repeated = fresh_data(model, state.qpos, state.qvel)
    backend.reset(repeated)
    backend.step(repeated)
    np.testing.assert_array_equal(query.qpos, repeated.qpos)
    np.testing.assert_array_equal(query.qvel, repeated.qvel)


def test_gpu_three_engines_agree_without_contact():
    import warp as wp
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    model = make_model("box", .5, .002, .15)
    initial = fresh_data(model)
    initial.qpos[2] = .5
    initial.qvel[:] = [.4, -.2, .1, 0, 0, 0]
    mujoco.mj_forward(model, initial)
    predictions = []
    for engine in ("mjwarp", "comfree", "scm_corrected", "scm_soft", "scm_soft_corrected"):
        backend = backend_for(engine, model, initial, options())
        data = fresh_data(model, initial.qpos, initial.qvel)
        backend.reset(data)
        backend.step(data)
        assert len(backend.last_contacts["gap"]) == 0
        if engine in ("scm_corrected", "scm_soft_corrected"):
            assert backend.last_amplitude_scale == 1
        predictions.append(data)
    for data in predictions[1:]:
        np.testing.assert_allclose(data.qpos, predictions[0].qpos, rtol=0, atol=1e-12)
        np.testing.assert_allclose(data.qvel, predictions[0].qvel, atol=1e-8)


def test_popup_default_and_headless_switch(monkeypatch, tmp_path):
    import compare_prediction as experiment
    from benchmark_utils import Backend
    from mujoco import viewer
    launches = []

    class Window:
        cam = SimpleNamespace(lookat=np.zeros(3), distance=1.)
        texts = []
        def __enter__(self):
            return self
        def __exit__(self, *unused):
            pass
        def is_running(self):
            return True
        def sync(self):
            pass
        def set_texts(self, texts):
            self.texts.append(texts)

    def launch(model, data):
        launches.append(True)
        return Window()

    # GUI branching test only: use fast CPU physics, not a simulated GPU claim.
    monkeypatch.setattr(viewer, "launch_passive", launch)
    monkeypatch.setattr(experiment, "backend_for", lambda e, m, d, a: Backend("mujoco", m, d, collect_contacts=True))
    for headless in (False, True):
        args = experiment.argument_parser().parse_args(["--headless"] if headless else [])
        assert args.headless == headless
        args.samples_per_mode = 1
        args.modes = ["sticking"]
        args.engines = ["mjwarp"]
        args.no_plots = True
        args.preview_seconds = .006
        args.display_pause = 0
        experiment.run_configuration("box", .5, .002, args, np.random.default_rng(42),
                                     tmp_path / ("headless" if headless else "default"))
    assert len(launches) == 1
    assert any("preview" in t[2] for t in Window.texts)
    assert any("requested: sticking" in t[2] for t in Window.texts)
    with np.load(tmp_path / "default" / "states.npz") as shown, np.load(tmp_path / "headless" / "states.npz") as hidden:
        for key in ("qpos_initial", "qvel_initial", "wrench_world_N_Nm", "qpos_predicted", "qvel_predicted"):
            np.testing.assert_array_equal(shown[key], hidden[key])


def test_replay_uses_reference_initial_state_and_wrench(tmp_path):
    qpos = np.arange(14.).reshape(2, 7)
    qvel = np.arange(12.).reshape(2, 6)
    wrench = -qvel
    np.savez(tmp_path / "states.npz", engine=["scm", "mjwarp"],
             requested_mode=["sliding", "sliding"], sample_id=[0, 0],
             qpos_initial=qpos, qvel_initial=qvel, wrench_world_N_Nm=wrench,
             qpos_predicted=qpos + 100)
    recorded = list(replay_states(tmp_path, "sliding", 1))
    np.testing.assert_array_equal(recorded[0][0], qpos[1])
    np.testing.assert_array_equal(recorded[0][1], qvel[1])
    np.testing.assert_array_equal(recorded[0][2], wrench[1])
    with pytest.raises(ValueError, match="only 1"):
        list(replay_states(tmp_path, "sliding", 2))


def test_amplitude_kernel_uses_full_mass_coupling():
    import warp as wp
    from scm_backend import correct_amplitude
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    inverse = np.array([[2., .4], [.4, 1.]])
    J = np.array([[1., 0.], [1., 0.], [.5, 1.]])
    h, r = .01, .02
    vf = np.array([-1., -.2])
    gap = np.array([-.001, -.001, -.002])
    W = J @ inverse @ J.T + r * np.eye(3)
    g = J @ vf + gap / h
    p = np.maximum(-g / np.diag(W), 0)
    expected_alpha = np.clip(-(g @ p) / (p @ W @ p), 0, 1)
    assert 0 < expected_alpha < 1
    with wp.ScopedDevice("cuda:0"):
        force = wp.array((p / h)[None], dtype=wp.float32)
        reaction = wp.array((J.T @ p / h)[None], dtype=wp.float32)
        alpha = wp.zeros(1, dtype=wp.float32)
        wp.launch(correct_amplitude, dim=1, inputs=[
            2, wp.array([h], dtype=wp.float32), wp.array([3], dtype=wp.int32),
            wp.array(J[None], dtype=wp.float32), wp.array(gap[None], dtype=wp.float32),
            wp.array(vf[None], dtype=wp.float32), wp.zeros((1, 2), dtype=wp.float32),
            wp.array(np.diag(inverse), dtype=wp.float32), wp.array(inverse, dtype=wp.float32),
            False, False, r, False], outputs=[force, reaction, alpha])
        np.testing.assert_allclose(alpha.numpy(), expected_alpha, rtol=3e-6)
        np.testing.assert_allclose(force.numpy()[0] * h, expected_alpha * p, rtol=3e-6)
        np.testing.assert_allclose(reaction.numpy()[0] * h, expected_alpha * J.T @ p, rtol=3e-6)


@pytest.mark.parametrize("tau,zeta", [(.05, 1.), (.008, .25)])
def test_soft_scm_fixed_geometry_force_jacobian_matches_gpu_differences(tau, zeta):
    import warp as wp
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    model = make_model("sphere", .5, .002, .15)
    initial = fresh_data(model)
    initial.qpos[2] = .03995
    initial.qvel[:] = [-.3, .1, -.02, 0, 0, .1]
    mujoco.mj_forward(model, initial)
    settings = options()
    settings.scm_response_time, settings.scm_damping_ratio = tau, zeta
    backend = backend_for("scm_soft", model, initial, settings)
    baseline = fresh_data(model, initial.qpos, initial.qvel)
    backend.reset(baseline)
    backend.step(baseline)
    count = int(backend.wdata.nefc.numpy()[0])
    J = backend.wdata.efc.J.numpy()[0, :count, :6].astype(float)
    active = backend.wdata.efc.force.numpy()[0, :count] > 0
    mass = np.zeros((6, 6))
    mujoco.mj_fullM(model, mass, initial.qM)
    inverse = np.linalg.inv(mass)
    D = np.einsum("ij,jk,ik->i", J, inverse, J) + 1e-8
    x = .002 / tau
    a = (2 * zeta * x + x * x) / (1 + 2 * zeta * x + x * x)
    derivative = (np.eye(6) - a * inverse @ J.T @ np.diag(active / D) @ J) @ (.002 * inverse)
    for axis in range(3):
        results = []
        for sign in (-1, 1):
            wrench = np.zeros(6)
            wrench[axis] = sign * .01
            data = fresh_data(model, initial.qpos, initial.qvel, wrench)
            backend.reset(data)
            backend.step(data)
            np.testing.assert_array_equal(backend.wdata.efc.force.numpy()[0, :count] > 0, active)
            results.append(data.qvel.copy())
        measured = (results[1] - results[0]) / .02
        np.testing.assert_allclose(measured, derivative[:, axis], rtol=.01, atol=2e-5)
