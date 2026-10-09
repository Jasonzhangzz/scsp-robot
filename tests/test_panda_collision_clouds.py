"""Planner wrist geometry covers the physical offset meshes without a giant capsule."""

import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from models.dexforge_planner_scene import write_dexforge_planner_scene
from models.comfree_gs_torch import GaussianCloud


def test_generated_wrist_and_flange_clouds_cover_offset_collision_mesh_vertices(tmp_path):
    mesh = trimesh.creation.box(extents=[.18, .08, .06])
    mesh.apply_translation([.04, .015, .005])
    mesh.export(tmp_path / "wrist.obj")
    xml = '<mujoco><asset><mesh name="wrist" file="wrist.obj"/></asset><worldbody>'
    for side in ("left", "right"):
        xml += (f'<body name="{side}_link6"><geom type="mesh" mesh="wrist"/>'
                f'<body name="{side}_link7" pos=".088 0 0">'
                '<geom type="mesh" mesh="wrist"/></body></body>')
    xml += '<body name="obj"><freejoint/><geom name="obj" type="sphere" size=".04" mass=".1"/></body></worldbody></mujoco>'
    viewer = tmp_path / "viewer.xml"
    viewer.write_text(xml)
    planner = write_dexforge_planner_scene(viewer, GaussianCloud([[0., 0., 0.]], radius=.004))
    tree = ET.parse(planner)
    for side in ("left", "right"):
        for part, link in (("wrist", 6), ("flange", 7)):
            cloud = np.load(tmp_path / f"_generated_{side}_{part}_gs.npz")
            centers, radii = cloud["local_pos"], cloud["radius"]
            gap = np.linalg.norm(mesh.vertices[:, None] - centers, axis=2) - radii
            assert np.max(np.min(gap, axis=1)) <= -.0019
            assert radii.max() < .07  # No origin-centred 10 cm enclosing ball.
            assert np.linalg.norm(centers.mean(axis=0)) > .02
            node = tree.find(f".//body[@name='{side}_link{link}']/geom[@name='{side}_{part}_gs']")
            assert node is not None and node.get("pos") == "0 0 0"


def test_object_contact_mode_preserves_object_contacts_and_all_query_geometry(tmp_path):
    viewer = tmp_path / "viewer.xml"
    viewer.write_text(
        '<mujoco><option noslip_iterations="10"/><worldbody><geom name="floor" type="plane" size="1 1 .1"/>'
        '<body name="left_attachment"><geom name="left_pad" type="sphere" size=".01"/></body>'
        '<body name="right_attachment"><geom name="right_pad" type="sphere" size=".01"/></body>'
        '<body name="obj"><freejoint/><geom name="obj" type="sphere" size=".04" mass=".1"/>'
        '</body></worldbody></mujoco>')
    cloud = GaussianCloud([[0., 0., 0.]], radius=.004)
    full = ET.parse(write_dexforge_planner_scene(viewer, cloud, tmp_path / "full.xml"))
    lean = ET.parse(write_dexforge_planner_scene(viewer, cloud, tmp_path / "lean.xml", mutual_arm_contacts=False))
    assert full.find("option").get("noslip_iterations") == "0"
    assert lean.find("option").get("noslip_iterations") == "0"
    assert ET.parse(viewer).find("option").get("noslip_iterations") == "10"
    full_geoms = {g.get("name"): g for g in full.iter("geom")}
    geoms = {g.get("name"): g for g in lean.iter("geom")}
    def enabled(a, b, source=geoms):
        a, b = source[a], source[b]
        return bool((int(a.get("contype")) & int(b.get("conaffinity")))
                    | (int(b.get("contype")) & int(a.get("conaffinity"))))
    for side in ("left", "right"):
        assert enabled(f"{side}_tip_gs", "obj_gs")
        assert not enabled(f"{side}_tip_gs", "floor")
        for key in ("file", "pos", "size"):
            assert geoms[f"{side}_tip_gs"].get(key) == full_geoms[f"{side}_tip_gs"].get(key)
    assert enabled("obj_gs", "floor")
    assert not enabled("left_tip_gs", "right_tip_gs")
    assert enabled("left_tip_gs", "right_tip_gs", full_geoms)
