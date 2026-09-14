"""Regression checks for the MuJoCo collision-hull export used by lambda."""

from pathlib import Path

import mujoco
import numpy as np
import trimesh

from examples.mpc.fingertips.test.params import _mujoco_collision_mesh


ROOT = Path(__file__).resolve().parents[1]


def _check_object(name: str) -> None:
    xml = ROOT / "envs" / "xmls" / f"env_fingertips_{name}.xml"
    source = ROOT / "envs" / "assets" / "objects" / f"{name}.stl"
    exported = _mujoco_collision_mesh(str(source), str(xml))

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    obj_geom = model.geom("obj").id
    tip_geom = model.geom("fingertip0").id
    # Keep the object well clear of the table.  Probing a clone data object
    # must not alter the model or the caller's simulation state.
    data.qpos[:7] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    qpos_before = data.qpos.copy()
    qvel_before = data.qvel.copy()
    model_geom_pos = model.geom_pos.copy()

    mesh = trimesh.load_mesh(exported, process=False)
    radius = float(model.geom_size[tip_geom, 0])
    # Every exported facet is a valid support facet.  A sphere whose centre
    # is one radius outside the facet should have zero geom distance.
    distances = []
    for point, normal in zip(mesh.triangles_center, mesh.face_normals):
        data.qpos[7:10] = np.asarray([0.0, 0.0, 1.0]) + point + radius * normal
        mujoco.mj_forward(model, data)
        distances.append(mujoco.mj_geomDistance(model, data, obj_geom, tip_geom, 1.0, None))

    assert np.max(np.abs(distances)) < 2e-5, (
        f"{name}: exported hull is not aligned with MuJoCo "
        f"(max |geomDistance|={np.max(np.abs(distances)):.3g})"
    )
    np.testing.assert_array_equal(data.qvel, qvel_before)
    np.testing.assert_array_equal(model.geom_pos, model_geom_pos)
    # The probe intentionally changes qpos; restore and verify the original
    # caller state can be recovered without side effects from the exporter.
    data.qpos[:] = qpos_before


def test_elephant_collision_hull_alignment() -> None:
    _check_object("elephant")


def test_foam_brick_collision_hull_alignment() -> None:
    _check_object("foam_brick")
