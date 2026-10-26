"""Native PhysX bridge invariants; small actual CPU-SDK checks when available."""
import json
import os
from pathlib import Path
import subprocess
import sys
import ast
from types import SimpleNamespace

import numpy as np
import pytest

from examples.mpc.franka.scm.runtime_env import activate_dependencies
activate_dependencies()
import mujoco

from examples.mpc.fingertips.test.collision_prediction_isaacgym_native import (
    IsaacGymNative, contact_frame, export_scene_spec, normalize_manifold, quaternion_matrix)


def contact(body=(0, 1), point=(.1, .2, 0.), normal=(0., 0., 1.), gap=-.001):
    return dict(body0=body[0], body1=body[1], point_world=point, normal_world=normal,
                gap=gap, friction=.5, include_margin=.002, force_N=123., raw={"lambda": 123.})


def test_canonical_manifold_ignores_forces_and_orders_contacts():
    a, b = contact(), contact((1, 2), (.2, 0., .03), (1., 0., 0.), .0005)
    reverse_b = dict(b, body0=2, body1=1, normal_world=(-1., 0., 0.), force_N=999.)
    first = normalize_manifold([b, a])
    second = normalize_manifold([a, reverse_b])
    assert set(first) == {"gap", "position", "frame", "body", "friction", "include_margin"}
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    np.testing.assert_array_equal(first["body"], [[0, 1], [1, 2]])
    np.testing.assert_array_equal(first["include_margin"], [.002, .002])
    assert first["gap"][1] > 0  # Speculative native candidates remain available.
    for frame in first["frame"]:
        np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-15)
        assert np.linalg.det(frame) == pytest.approx(1.)


def test_empty_manifold_has_complete_shapes():
    result = normalize_manifold([])
    assert result["frame"].shape == (0, 3, 3)
    assert result["position"].shape == (0, 3)
    assert result["body"].shape == (0, 2)
    assert result["friction"].shape == (0, 5)
    with pytest.raises(ValueError):
        contact_frame([0, 0, 0])
    with pytest.raises(ValueError):
        normalize_manifold([contact(gap=np.nan)])


def model_file(tmp_path, kind="sphere", inertial=""):
    geometry = {"sphere": 'type="sphere" size=".03"',
                "cylinder": 'type="cylinder" size=".025 .03"'}[kind]
    source = tmp_path / "model.xml"
    source.write_text(f'''<mujoco><option timestep=".002" gravity="0 0 -9.81"/>
      <worldbody><geom type="plane" size="2 2 .1" friction=".5 0 0"/>
       <body name="object"><freejoint/>{inertial}<geom {geometry} mass=".01" friction=".5 0 0"/></body>
       <body name="ball"><freejoint/><geom type="sphere" size=".01" mass=".01" friction=".5 0 0"/></body>
      </worldbody></mujoco>''')
    return source


def test_cylinder_export_is_fixed_96_slices_and_preserves_inertia(tmp_path):
    source = model_file(tmp_path, "cylinder")
    model = mujoco.MjModel.from_xml_path(str(source))
    spec = export_scene_spec(source, tmp_path / "export")
    assert spec["cylinder_slices"] == 96
    assert spec["cylinder_max_radial_deficit_m"] == pytest.approx(.025*(1-np.cos(np.pi/96)))
    part = tmp_path / "export/object_part_0000.obj"
    vertices = np.array([[float(x) for x in line.split()[1:]] for line in part.read_text().splitlines()
                         if line.startswith("v ")])
    assert vertices.shape == (192, 3)
    np.testing.assert_allclose(np.linalg.norm(vertices[:, :2], axis=1), .025)
    np.testing.assert_allclose(spec["bodies"][0]["inertia"], np.diag(model.body_inertia[1]))
    assert spec["bodies"][0]["expected_shapes"] == 1
    assert spec["velocity_iterations"] == 1
    assert spec["max_depenetration_velocity"] == 10.
    assert spec["physx_parameters"]["max_gpu_contact_pairs"] == 8388608
    assert spec["physx_parameters"]["default_buffer_size_multiplier"] == 16.
    assert json.loads((tmp_path / "export/scene_spec.json").read_text())["source_model_sha256"]
    with pytest.raises(ValueError):
        export_scene_spec(source, tmp_path / "bad", cylinder_slices=32)


def test_off_center_body_exports_explicit_full_inertia_without_extra_collision(tmp_path):
    source = model_file(tmp_path, inertial='''<inertial pos=".001 .002 -.003"
        quat=".9238795325 0 0 .3826834324" mass=".01" diaginertia=".000004 .000005 .000006"/>''')
    spec = export_scene_spec(source, tmp_path / "export")
    body = spec["bodies"][0]
    np.testing.assert_allclose(body["ipos"], [.001, .002, -.003])
    assert abs(body["inertia"][0][1]) > 1e-8
    urdf = Path(body["urdf"]).read_text()
    assert urdf.count("<collision ") == 1
    assert 'ixy="-' in urdf


def test_native_cpu_roundtrip_freeflight_impulse_and_independent_endpoint(tmp_path):
    interpreter = Path.home()/"miniconda3/envs/acc/bin/python"
    if not interpreter.is_file():
        pytest.skip("Legacy Isaac Gym Python not installed")
    source = model_file(tmp_path, inertial='''<inertial pos=".001 .002 -.003"
        quat=".9238795325 0 0 .3826834324" mass=".01" diaginertia=".000004 .000005 .000006"/>''')
    spec = export_scene_spec(source, tmp_path / "export")
    q = np.array([[0, 0, .2, 1, 0, 0, 0, .2, 0, .2, 1, 0, 0, 0],
                  [0, 0, .02998, 1, 0, 0, 0, 0, 0, .06996, 1, 0, 0, 0]], float)
    v, force = np.zeros((2, 12)), np.zeros((2, 3, 6))
    force[:, 2, 0] = .5
    with IsaacGymNative(spec, batch_size=2, use_gpu=False, log_path=tmp_path/"sdk.log") as native:
        fingerprint = native.metadata["runtime_fingerprint"]
        assert fingerprint["numpy_version"]
        assert fingerprint["python_executable"].endswith("python")
        assert len(fingerprint["native_libraries"][0]["sha256"]) == 64
        geometry = native.geometry(q, v)
        assert not geometry["contacts"][0]
        assert {tuple((x["body0"], x["body1"])) for x in geometry["contacts"][1]} == {(0, 1), (1, 2)}
        for row in geometry["contacts"][1]:
            assert "force_N" not in row and "lambda" not in row["raw"]
            assert row["friction"] == .5
            assert row["include_margin"] == pytest.approx(.002)
            assert row["normal_world"][2] > .999
            assert row["gap"] == pytest.approx(-2e-5, abs=2e-7)
        assert geometry["native_warnings"] == [] or all("CUDA device" in line for line in geometry["native_warnings"])
        config = geometry["physics_configuration"]
        assert config["physx"]["num_velocity_iterations"] == 1
        assert config["enable_gyroscopic_forces"]
        assert config["environment_origins"] == "zero"
        first = native.reference(q, v, force)
        second = native.reference(q, v, force)
    assert np.asarray(first["trajectory_qpos"]).shape == (2, 11, 14)
    assert first["elapsed_seconds"] == pytest.approx(.02, abs=1e-8)
    assert first["qvel"][0][6] == pytest.approx(1., abs=2e-6)  # .5N*.02s/.01kg
    assert first["qvel"][0][8] == pytest.approx(-.1962, abs=2e-6)
    np.testing.assert_allclose(first["qpos"], second["qpos"], atol=1e-12)
    np.testing.assert_allclose(first["qvel"], second["qvel"], atol=1e-12)
    for rows in first["snapshots"]:
        assert len(rows) == 11
        assert rows[-1]["time"] == .02
        assert rows[-1]["endpoint_diagnostic"]
        assert rows[-1]["force_interval_s"] == [.02, .022]
    assert first["diagnostic"]["endpoint_integrated_state_discarded"]
    assert first["asset_validation"][0]["mass"] == pytest.approx(.01)


def test_native_capacity_diagnostics_preserve_the_complete_original_message(tmp_path):
    worker_directory = Path(__file__).resolve().parents[1]/"examples/mpc/fingertips/test"
    messages = [
        "/source/gpubroadphase/src/PxgCudaBroadPhaseSap.cpp (579) : invalid parameter : "
        "the application need to increase the PxgDynamicsMemoryConfig::foundLostPairsCapacity "
        "parameter to 1607039, otherwise the simulation will miss interactions",
        "Invalid Parameter: foundLostAggregatePairsCapacity too small; simulation missed interactions",
        "GPU dynamics capacity exhausted",
    ]
    # Execute the real, stdlib-only capture function in isolation. Loading the
    # entire legacy SDK is unnecessary for testing native file descriptors and
    # can trigger its uninitialized-plugin shutdown bug without a live scene.
    script = ("import sys, json, ast, contextlib, ctypes, os, tempfile, re\n"
              "from pathlib import Path\n"
              f"source = Path({str(worker_directory/'collision_prediction_isaacgym_worker.py')!r})\n"
              "tree = ast.parse(source.read_text())\n"
              "function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='capture_native_log')\n"
              "exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'))\n"
              "with capture_native_log() as captured:\n"
              f" for line in {messages!r}: print(line, file=sys.stderr)\n"
              "print(json.dumps(captured))\n")
    result = subprocess.run([sys.executable, "-c", script], env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                            capture_output=True, text=True, check=True)
    captured = json.loads(result.stdout)
    assert captured["warnings"] == messages
    for line in messages:
        assert line in result.stderr


@pytest.mark.parametrize("use_gpu", [False, True], ids=["cpu", "gpu"])
def test_high_spin_frozen_queries_compare_native_world_velocities(tmp_path, use_gpu):
    if use_gpu and os.environ.get("SCSP_ISAACGYM_GPU_TESTS") != "1":
        pytest.skip("Set SCSP_ISAACGYM_GPU_TESTS=1 for the actual GPU-SDK check")
    if not (Path.home()/"miniconda3/envs/acc/bin/python").is_file():
        pytest.skip("Legacy Isaac Gym Python not installed")
    # Exact original frozen inputs sphere_s42_000058, 000098 and 000114.
    # Their 54--72 rad/s ball spin is obtained during source settling. Rotating
    # its float32 world velocity back to body axes perturbs a near-zero body
    # component by 3.5--5.3e-6 rad/s, although the native upload is correct.
    q = np.array([
        [.06428704011471242,-.007540864113815132,.029932059347629547,
         .43337680896533526,.46621407860873787,.7432766825334132,.20583709121982657,
         -.3045026086887119,.03571986653301547,.009645726531744003,
         .4639311961484527,.47580973509533653,-.5388105418760817,-.5177414922541971],
        [-.03794283347566523,.05244072411906236,.029932059347629544,
         -.014187766453928718,-.4930536168801908,.7257505940557216,.4795655053836385,
         .17971910828577284,-.2483924106737843,.009645726531744002,
         .660595415344489,.6846625834035861,.21897226065284783,.21656893772464017],
        [-.0026281108936153865,-.023697948700847966,.029931679368019104,
         .567941957400562,.19949576391464688,-.3246372228765878,.7295575276184308,
         .012082757505369076,.10896846693435834,.009645220823585987,
         -.29279232331818306,-.6133662454645382,.685090356511069,-.26211787287892996]])
    v = np.array([
        [.13653845129344433,-.016016055165271936,2.3133236481953645e-7,
         3.8694140911102295,2.4646401405334473,-8.1197104009334e-5,
         -.7044954360770097,.08264150928702334,-8.015719288323453e-8,
         72.20983123779297,-.5238755941390991,-8.226203499361873e-5],
        [-.08058623713115082,.11137827925458013,2.3133236481953642e-7,
         3.8694140911102295,2.4646401405334473,-8.1197104009334e-5,
         .415796892291031,-.5746793739744745,-8.015719288323452e-8,
         72.20983123779297,-.5238755941390991,-8.226203499361873e-5],
        [-.010609970155041172,-.0956781312098594,1.3730168575420976e-5,
         -1.126521110534668,-3.008493185043335,-2.022426633629948e-5,
         .05863136263457006,.528757535115836,.007431109435856342,
         -.0033480797428637743,54.472293853759766,-.001156885176897049]])
    force = np.zeros((3, 3, 6))
    force[:, 2, 2] = [-.5, -.5, -.2]
    spec = export_scene_spec(model_file(tmp_path), tmp_path/"export")
    with IsaacGymNative(spec, batch_size=3, use_gpu=use_gpu, log_path=tmp_path/"sdk.log") as native:
        first = native.reference(q, v, force)
        second = native.reference(q, v, force)
    readback = first["initial_readback"]
    assert readback["velocity_guard_frame"] == "native world COM linear and world angular velocity"
    assert readback["max_position_roundtrip_m"] < 1e-7
    assert readback["max_body_velocity_roundtrip_component"] > 3e-6
    assert readback["max_world_angular_velocity_roundtrip_rad_s"] < 1e-5
    assert readback["max_world_linear_velocity_roundtrip_m_s"] < 1e-7
    assert readback["velocity_guard_atol"] == 3e-6
    assert readback["velocity_guard_rtol"] == 2e-5
    assert first["elapsed_seconds"] == pytest.approx(.02, abs=1e-8)
    np.testing.assert_array_equal(first["qpos"], second["qpos"])
    np.testing.assert_array_equal(first["qvel"], second["qvel"])
    if use_gpu:
        assert first["native_warnings"] == []
        assert first["physics_device_log"] == ["Physics Device: cuda:0", "Physics Device: cuda:0"]


def test_native_world_velocity_guard_rejects_actual_upload_corruption():
    source = Path(__file__).resolve().parents[1]/"examples/mpc/fingertips/test/collision_prediction_isaacgym_worker.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NativeScene")
    reset = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "reset")
    scope = dict(np=np, quaternion_matrix=quaternion_matrix, gymapi=SimpleNamespace(STATE_ALL=0))
    exec(compile(ast.Module(body=[reset],type_ignores=[]),str(source),"exec"),scope)
    state_type = np.dtype([("pose", [("p", "f4", (3,)), ("r", "f4", (4,))]),
                          ("vel", [("linear", "f4", (3,)), ("angular", "f4", (3,))])])
    q = np.array([[0,0,.1,1,0,0,0,.2,0,.1,1,0,0,0]], float)
    v = np.zeros((1,12))
    # Keep the reported body-frame state pristine but corrupt the actual native
    # world angular velocity; the upload guard must inspect native quantities.
    links = [{1:(q[0,:3], q[0,3:7],np.zeros(3),np.array([.1,0,0])),
              2:(q[0,7:10],q[0,10:14],np.zeros(3),np.zeros(3))}]
    fake = SimpleNamespace(nworld=1,spec={"bodies":[{"role_id":1,"ipos":[0,0,0]},
        {"role_id":2,"ipos":[0,0,0]}]},environments=[0],actors=[[0,1]],
        gym=SimpleNamespace(get_actor_rigid_body_states=lambda *_:np.zeros(1,dtype=state_type),
                            set_actor_rigid_body_states=lambda *_:True),
        read=lambda:(q.copy(),v.copy(),links))
    with pytest.raises(RuntimeError,match="world COM/angular velocity"):
        scope["reset"](fake,q,v)
