"""Content-addressed CoACD assets and matched two-free-body collision scenes."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
MESH_OBJECTS = ("elephant", "stanford_bunny2", "teapot", "foam_brick",
                "rubber_duck", "mug", "piggy_bank")
OBJECTS = ("sphere", "cube", "cylinder") + MESH_OBJECTS
DEFAULT_CACHE = ROOT / "envs/assets/objects/_generated_collision/prediction_coacd"
COACD_PARAMETERS = dict(threshold=.001, real_metric=True, seed=42,
                        max_convex_hull=-1, preprocess_mode="auto",
                        preprocess_resolution=50, resolution=2000, mcts_nodes=20,
                        mcts_iterations=150, mcts_max_depth=3, pca=False,
                        merge=True, decimate=False, max_ch_vertex=256,
                        extrude=False, extrude_margin=.01, apx_mode="ch")


def numbers(value):
    return " ".join(format(float(x), ".17g") for x in np.asarray(value).reshape(-1))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(quaternion):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, np.asarray(quaternion, dtype=float))
    return result.reshape(3, 3)


def source_description(name):
    import trimesh
    scene = ROOT / "envs/xmls" / f"env_fingertips_{name}.xml"
    tree = ET.parse(scene)
    geom = tree.find(".//geom[@name='obj']")
    mesh = tree.find(f".//mesh[@name='{geom.get('mesh')}']")
    source = (scene.parent / mesh.get("file")).resolve()
    vertices = trimesh.load(source, force="mesh").vertices
    scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
    quat = np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ")
    pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
    body_vertices = (vertices * scale) @ rotation(quat).T + pos
    model = mujoco.MjModel.from_xml_path(str(scene))
    body = model.body("obj").id
    return dict(name=name, source=str(source), source_sha256=digest(source),
                scene_sha256=digest(scene), scale=scale.tolist(),
                geom_quat=quat.tolist(), geom_pos=pos.tolist(),
                mass=float(model.body_mass[body]), ipos=model.body_ipos[body].tolist(),
                iquat=model.body_iquat[body].tolist(), inertia=model.body_inertia[body].tolist(),
                bounds=[body_vertices.min(0).tolist(), body_vertices.max(0).tolist()])


def prepare_asset(name, cache=DEFAULT_CACHE, threshold=.001):
    """Never substitute a single hull when decomposition fails. Cache is immutable."""
    import coacd
    import trimesh
    info = source_description(name)
    parameters = dict(COACD_PARAMETERS, threshold=threshold)
    info.update(coacd_version=importlib.metadata.version("coacd"), parameters=parameters)
    key = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
    destination = Path(cache) / name / key
    manifest = destination / "asset.json"
    if manifest.is_file():
        saved = json.loads(manifest.read_text())
        for part, checksum in saved["part_sha256"].items():
            if digest(destination / part) != checksum:
                raise ValueError(f"Corrupt CoACD cache: {destination / part}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"CoACD {name}: threshold={threshold:g} m -> {destination}", flush=True)
    source = trimesh.load(info["source"], force="mesh")
    vertices = np.asarray(source.vertices) * info["scale"]
    parts = coacd.run_coacd(coacd.Mesh(vertices, np.asarray(source.faces)), **parameters)
    if not parts:
        raise RuntimeError(f"CoACD returned no parts for {name}")
    temporary = Path(tempfile.mkdtemp(prefix=".prepare-", dir=destination.parent))
    try:
        checksums = {}
        for index, (vertices, faces) in enumerate(parts):
            vertices = np.asarray(vertices)
            if not np.isfinite(vertices).all() or len(vertices) < 4:
                raise ValueError(f"Invalid convex component {index} for {name}")
            # Rebuild EACH returned component's hull to repair CoACD face
            # winding / nearly coplanar triangles. Never hull the whole object.
            part = trimesh.convex.convex_hull(vertices)
            if not part.is_convex or abs(part.volume) <= 1e-18:
                raise ValueError(f"Degenerate convex component {index} for {name}: volume={part.volume}")
            path = temporary / f"part_{index:04d}.obj"
            part.export(path)
            checksums[path.name] = digest(path)
        info.update(cache_key=key, part_sha256=checksums, part_count=len(parts),
                    source_watertight=bool(source.is_watertight))
        (temporary / "asset.json").write_text(json.dumps(info, indent=2) + "\n")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def scene_xml(name, asset_dir=None, *, timestep=.002, friction=.5,
              primitive_mass=.01, ball_mass=.01, ball_radius=.01):
    """Explicit source inertial prevents overlapping convex pieces adding mass."""
    root = ET.Element("mujoco", model=f"collision_prediction_{name}")
    ET.SubElement(root, "compiler", angle="radian")
    option = ET.SubElement(root, "option", timestep=str(timestep), integrator="Euler",
                          cone="pyramidal", iterations="100", tolerance="1e-10", gravity="0 0 -9.81",
                          ccd_iterations="100")
    ET.SubElement(option, "flag", warmstart="disable", sensor="disable")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", condim="3", friction=f"{friction} 0 0", margin="0", gap="0")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 2", dir="0 0 -1")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 .1",
                  contype="2", conaffinity="5", rgba=".25 .3 .4 1")
    body = ET.SubElement(world, "body", name="object", pos="0 0 .1")
    ET.SubElement(body, "freejoint", name="object_free")
    common = dict(contype="1", conaffinity="6", rgba=".9 .55 .15 1")
    if name in MESH_OBJECTS:
        if asset_dir is None:
            raise ValueError(f"Prepared CoACD assets required for {name}")
        asset_dir = Path(asset_dir).resolve()
        info = json.loads((asset_dir / "asset.json").read_text())
        assets = ET.SubElement(root, "asset")
        # The packaged visual mesh is in source coordinates; only convex parts
        # have scale baked in. MuJoCo handles each mesh's centering internally.
        ET.SubElement(assets, "mesh", name="visual_mesh", file=info["source"], scale=numbers(info["scale"]))
        transform = dict(pos=numbers(info["geom_pos"]), quat=numbers(info["geom_quat"]))
        ET.SubElement(body, "inertial", mass=str(info["mass"]), pos=numbers(info["ipos"]),
                      quat=numbers(info["iquat"]), diaginertia=numbers(info["inertia"]))
        ET.SubElement(body, "geom", name="object_visual", type="mesh", mesh="visual_mesh",
                      contype="0", conaffinity="0", group="2", mass="0", rgba=".9 .55 .15 1", **transform)
        for index, filename in enumerate(sorted(info["part_sha256"])):
            ET.SubElement(assets, "mesh", name=f"part_{index}", file=str(asset_dir / filename))
            ET.SubElement(body, "geom", name=f"object_part_{index}", type="mesh", mesh=f"part_{index}",
                          mass="0", group="3", **transform, **dict(common, rgba=".9 .55 .15 0"))
        bounds = np.asarray(info["bounds"])
    else:
        kind, size, extent = {"sphere": ("sphere", ".03", [.03]*3),
                              "cube": ("box", ".03 .03 .03", [.03]*3),
                              "cylinder": ("cylinder", ".025 .03", [.025, .025, .03])}[name]
        ET.SubElement(body, "geom", name="object_geom", type=kind, size=size,
                      mass=str(primitive_mass), **common)
        bounds = np.array([-np.asarray(extent), extent])
        info = dict(name=name, mass=primitive_mass, bounds=bounds.tolist(), collision="analytic_primitive")
    ball = ET.SubElement(world, "body", name="ball", pos=".3 0 .1")
    ET.SubElement(ball, "freejoint", name="ball_free")
    ET.SubElement(ball, "geom", name="ball_geom", type="sphere", size=str(ball_radius),
                  mass=str(ball_mass), contype="4", conaffinity="3", rgba=".2 .6 1 1")
    info = dict(info, length_scale_m=float(np.linalg.norm(bounds[1] - bounds[0])),
                ball_radius=ball_radius, ball_mass=ball_mass, friction=friction)
    return ET.tostring(root, encoding="unicode"), info


def package_scene(xml, destination):
    """Make model.xml self-contained with hashed visual/convex assets."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tree = ET.fromstring(xml)
    for mesh in tree.findall("./asset/mesh"):
        source = Path(mesh.get("file"))
        target = Path("assets") / (digest(source)[:16] + source.suffix)
        (destination / "assets").mkdir(exist_ok=True)
        shutil.copy2(source, destination / target)
        mesh.set("file", str(target))
    path = destination / "model.xml"
    path.write_text(ET.tostring(tree, encoding="unicode") + "\n")
    return path


def object_vertices(model):
    """Collision vertices in body coordinates (analytic extents are exact)."""
    vertices = []
    for geom in np.flatnonzero((model.geom_bodyid == model.body("object").id) & (model.geom_contype != 0)):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            points = model.mesh_vert[start:start+count]
        else:
            # Used only for conservative support bounds on primitives; exact
            # sphere/cylinder support is handled by support_height below.
            size = model.geom_size[geom]
            points = np.array(np.meshgrid(*[[-s, s] for s in size])).reshape(3, -1).T
        vertices.extend(points @ rotation(model.geom_quat[geom]).T + model.geom_pos[geom])
    return np.asarray(vertices)


def support_height(name, quat, vertices):
    R = rotation(quat)
    if name == "sphere":
        return .03
    if name == "cylinder":
        z = np.clip(abs(R[2, 2]), 0., 1.)
        return .03*z + .025*np.sqrt(max(0., 1-z*z))
    return -float(np.min(vertices @ R[2]))
