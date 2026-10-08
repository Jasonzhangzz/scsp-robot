"""Write a DexForge-loadable GS planner XML from the BigRasp viewer scene."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


PLANNER_SCENE_NAME = "_generated_bigrasp_planner.xml"
OBJECT_GS_NAME = "_generated_object_gs.npz"


def write_sphere_npz(path, centers, radii):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    radii = np.asarray(radii, dtype=np.float32).reshape(-1)
    if radii.shape[0] == 1:
        radii = np.full((centers.shape[0],), float(radii[0]), dtype=np.float32)
    np.savez(path, local_pos=centers, radius=np.maximum(radii, 1.0e-5))
    return path


def write_dexforge_planner_scene(
    viewer_xml,
    cloud,
    output_xml=None,
    *,
    tip_radius=0.01,
    forearm_radius=0.045,
    wrist_radius=0.035,
):
    """Copy the viewer scene and attach object / arm Gaussian query clouds."""
    viewer_xml = Path(viewer_xml)
    xml_dir = viewer_xml.parent
    output_xml = Path(output_xml) if output_xml is not None else xml_dir / PLANNER_SCENE_NAME
    tree = ET.parse(viewer_xml)
    root = tree.getroot()

    object_gs = write_sphere_npz(xml_dir / OBJECT_GS_NAME, cloud.centers, cloud.radii)
    _configure_planner_option(root)
    _add_object_gs(root, object_gs.name)
    _write_arm_query_assets(xml_dir, tip_radius, forearm_radius, wrist_radius)
    _add_arm_query_geoms(root)
    _apply_mutual_collision_groups(root)

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=False)
    return output_xml


def _configure_planner_option(root):
    """DexForge ``compile_fast_step`` only differentiates Euler integration."""
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.set("integrator", "Euler")
    # Planner GS has no object-vs-table support. Keep gravity off so a 0.3s
    # horizon does not drop the free object and drag contact targets to the floor.
    option.set("gravity", "0 0 0")
    obj_body = _find_body(root, "obj")
    if obj_body is not None:
        obj_body.set("gravcomp", "1")


def _add_object_gs(root, filename):
    obj_body = _find_body(root, "obj")
    if obj_body is None:
        raise RuntimeError("planner scene is missing body 'obj'")
    for geom in obj_body.findall("geom"):
        if geom.get("name") == "obj":
            geom.set("contype", "1")
            geom.set("conaffinity", "6")
    existing = [geom for geom in obj_body.findall("geom") if geom.get("name") == "obj_gs"]
    if existing:
        return
    ET.SubElement(
        obj_body,
        "geom",
        {
            "name": "obj_gs",
            "type": "gs",
            "file": filename,
            "size": "0.05",
            "contype": "1",
            "conaffinity": "6",
            "condim": "3",
            "friction": "1.0 0.005 0.0001",
            "priority": "1",
            "group": "3",
        },
    )


def _write_arm_query_assets(xml_dir, tip_radius, forearm_radius, wrist_radius):
    specs = {
        "_generated_left_tip_gs.npz": (np.zeros((1, 3), np.float32), tip_radius),
        "_generated_right_tip_gs.npz": (np.zeros((1, 3), np.float32), tip_radius),
        "_generated_left_forearm_gs.npz": (np.zeros((1, 3), np.float32), forearm_radius),
        "_generated_right_forearm_gs.npz": (np.zeros((1, 3), np.float32), forearm_radius),
        "_generated_left_wrist_gs.npz": (np.zeros((1, 3), np.float32), wrist_radius),
        "_generated_right_wrist_gs.npz": (np.zeros((1, 3), np.float32), wrist_radius),
    }
    for name, (centers, radius) in specs.items():
        write_sphere_npz(xml_dir / name, centers, radius)


def _add_arm_query_geoms(root):
    attachments = (
        ("left_attachment", "_generated_left_tip_gs.npz", "left_tip_gs", "2", "5", "0 0 0.06"),
        ("right_attachment", "_generated_right_tip_gs.npz", "right_tip_gs", "4", "3", "0 0 0.06"),
        ("left_link4", "_generated_left_forearm_gs.npz", "left_forearm_gs", "2", "5", "0 0 0"),
        ("right_link4", "_generated_right_forearm_gs.npz", "right_forearm_gs", "4", "3", "0 0 0"),
        ("left_link6", "_generated_left_wrist_gs.npz", "left_wrist_gs", "2", "5", "0 0 0"),
        ("right_link6", "_generated_right_wrist_gs.npz", "right_wrist_gs", "4", "3", "0 0 0"),
    )
    for body_name, filename, geom_name, contype, conaffinity, pos in attachments:
        body = _find_body(root, body_name)
        if body is None:
            continue
        if any(geom.get("name") == geom_name for geom in body.findall("geom")):
            continue
        ET.SubElement(
            body,
            "geom",
            {
                "name": geom_name,
                "type": "gs",
                "file": filename,
                "size": "0.01",
                "pos": pos,
                "contype": contype,
                "conaffinity": conaffinity,
                "condim": "3",
                "group": "3",
                "rgba": "0 0 0 0",
            },
        )


def _apply_mutual_collision_groups(root):
    for body in root.iter("body"):
        name = body.get("name", "")
        if name.startswith("left_"):
            contype, conaffinity = "2", "5"
        elif name.startswith("right_"):
            contype, conaffinity = "4", "3"
        else:
            continue
        for geom in body.findall("geom"):
            geom_class = geom.get("class", "")
            if "visual" in geom_class:
                continue
            if geom.get("contype") == "0" and geom.get("conaffinity", "0") == "0":
                continue
            geom.set("contype", contype)
            geom.set("conaffinity", conaffinity)


def _find_body(root, name):
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    return None
