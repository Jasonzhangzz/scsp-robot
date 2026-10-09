"""A freely supported elephant must settle without any robot contact."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from examples.mpc.franka.bigrasp.bigrasp import (
    OBJECT_ASSET_DIR, PANDA_HOME_Q, PANDA_XML_PATH, build_bimanual_scene_xml,
    load_mesh_bounds,
)


@pytest.fixture
def resting_scene(tmp_path):
    (tmp_path / "assets").symlink_to(PANDA_XML_PATH.parent / "assets", target_is_directory=True)
    mesh = OBJECT_ASSET_DIR / "elephant.stl"
    pose = np.array([.58, 0., .32 - load_mesh_bounds(mesh, np.ones(3))[0, 2]])
    path = build_bimanual_scene_xml(
        mesh, np.ones(3), .01, 5., pose, [1., 0., 0., 0.], .58, .75,
        pedestal_pos=(.58, 0., .26), scene_output_path=tmp_path / "scene.xml",
        position_actuators=True, support_noslip_iterations=10,
    )
    return path, pose


def _initial_data(model, pose):
    data = mujoco.MjData(model)
    q = int(model.jnt_qposadr[model.joint("obj_freejoint").id])
    data.qpos[q:q + 7] = np.r_[pose, 1., 0., 0., 0.]
    for side in ("left", "right"):
        for index, value in enumerate(PANDA_HOME_Q, 1):
            data.qpos[model.jnt_qposadr[model.joint(f"{side}_joint{index}").id]] = value
            data.ctrl[model.actuator(f"{side}_actuator{index}").id] = value
    mujoco.mj_forward(model, data)
    return data, q


def test_uncontrolled_elephant_settles_and_remains_free_to_move(resting_scene):
    path, pose = resting_scene
    model = mujoco.MjModel.from_xml_path(str(path))
    data, q = _initial_data(model, pose)
    obj = model.body("obj").id
    velocity = int(model.jnt_dofadr[model.joint("obj_freejoint").id])
    samples = []
    for step in range(600):  # Six seconds with the original 10 ms servo step.
        mujoco.mj_step(model, data)
        for contact in data.contact:
            bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
            if obj in bodies:
                other = bodies[1] if bodies[0] == obj else bodies[0]
                assert not model.body(int(other)).name.startswith(("left_", "right_"))
        if step >= 300:
            samples.append(np.r_[data.qpos[q:q + 7], data.qvel[velocity:velocity + 6]])
    samples = np.array(samples)
    assert np.max(np.ptp(samples[:, :3], axis=0)) < 1.e-5
    assert np.ptp(samples[:, 2]) < 1.e-7
    assert np.max(np.ptp(samples[:, 3:7], axis=0)) < 1.e-5
    assert np.max(np.abs(samples[:, 7:10])) < 1.e-5
    assert np.max(np.abs(samples[:, 10:])) < 1.e-4
    np.testing.assert_array_equal(data.xfrc_applied, 0.)
    np.testing.assert_array_equal(data.qfrc_applied, 0.)
    assert data.ncon > 0

    # No freezing, velocity reset or artificial pose constraint: a real force
    # must lift the same free body, and removing it must restore free fall.
    z = data.qpos[q + 2]
    data.xfrc_applied[obj, 2] = .3
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert data.qpos[q + 2] > z + .02
    data.xfrc_applied[:] = 0.
    mujoco.mj_forward(model, data)
    assert data.qacc[velocity + 2] == pytest.approx(-9.81, abs=1.e-6)


def test_support_pieces_preserve_inertia_and_only_replace_table_contacts(resting_scene):
    path, _ = resting_scene
    model = mujoco.MjModel.from_xml_path(str(path))
    root = ET.parse(path).getroot()
    body = root.find(".//body[@name='obj']")
    support = [g for g in body.findall("geom") if g.get("name", "").startswith("object_support_")]
    assert len(support) == 4
    for geom in support:
        assert geom.get("mass") == "0"
        body.remove(geom)
    baseline_path = path.with_name("baseline.xml")
    ET.ElementTree(root).write(baseline_path)
    original = mujoco.MjModel.from_xml_path(str(baseline_path))
    for field in ("body_mass", "body_ipos", "body_iquat", "body_inertia"):
        np.testing.assert_allclose(getattr(model, field)[model.body("obj").id],
                                   getattr(original, field)[original.body("obj").id], atol=1.e-14)
    def collision(a, b):
        a, b = model.geom(a).id, model.geom(b).id
        return bool((model.geom_contype[a] & model.geom_conaffinity[b])
                    | (model.geom_contype[b] & model.geom_conaffinity[a]))
    assert not collision("obj", "pedestal_geom")
    assert collision("obj", "floor")
    assert collision("obj", "left_fingertip")
    assert collision("obj", "right_fingertip")
    assert collision("left_fingertip", "pedestal_geom")
    for geom in support:
        assert collision(geom.get("name"), "pedestal_geom")
        assert not collision(geom.get("name"), "left_fingertip")
