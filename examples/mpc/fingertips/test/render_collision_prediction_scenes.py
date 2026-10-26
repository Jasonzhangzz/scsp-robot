#!/usr/bin/env python3
"""Render two saved collision queries with MuJoCo's native offscreen renderer.

No simulation or generative image model is used. Rendering changes only the
camera, lighting and visual appearance; meshes, dimensions and saved poses are kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from examples.mpc.franka.scm.runtime_env import activate_dependencies
activate_dependencies()

import mujoco
import numpy as np
from PIL import Image

from examples.mpc.fingertips.test.collision_prediction_assets import object_vertices


SCENES = (
    dict(object="cube", sample_id="cube_s42_000005", frame=4,
         filename="collision_scene_cube", azimuth_offset=35., elevation=-24.),
    dict(object="stanford_bunny2", sample_id="stanford_bunny2_s42_000005", frame=3,
         filename="collision_scene_bunny", azimuth_offset=35., elevation=-24.),
)


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(run_dir, output_dir, scene, width=2400, height=1800):
    folder = run_dir / scene["object"]
    model_path = folder / "model.xml"
    query_path = folder / "queries" / (scene["sample_id"] + ".npz")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    with np.load(query_path, allow_pickle=False) as archive:
        index = scene["frame"]
        data.qpos[:] = archive["reference_qpos"][index]
        data.qvel[:] = archive["reference_qvel"][index]
        data.xfrc_applied[:] = archive["wrench"]
        data.time = float(archive["reference_times"][index])
        metadata = json.loads(str(archive["metadata_json"]))
    mujoco.mj_forward(model, data)

    # Frame the actual two-body silhouette, including the ball's true radius.
    obj, ball = model.body("object").id, model.body("ball").id
    points = object_vertices(model) @ data.xmat[obj].reshape(3, 3).T + data.xpos[obj]
    radius = model.geom_size[model.geom("ball_geom").id, 0]
    corners = np.array(np.meshgrid(*[[-radius, radius]]*3)).reshape(3, -1).T
    points = np.concatenate([points, corners+data.xpos[ball]])
    bearing = np.degrees(np.arctan2(*(data.xpos[ball, :2]-data.xpos[obj, :2])[::-1]))
    azimuth = bearing + scene["azimuth_offset"]
    az, el = np.radians([azimuth, scene["elevation"]])
    outward = np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), -np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.])
    up = np.cross(outward, right)
    center = (points.min(0)+points.max(0))/2
    projected = (points-center) @ np.stack([right, up]).T
    middle = (projected.min(0)+projected.max(0))/2
    center += middle[0]*right + middle[1]*up
    span = max(np.ptp(projected[:, 1]), np.ptp(projected[:, 0])*height/width)*1.45

    # All of the following edits are visualization parameters in memory only.
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    model.vis.global_.orthographic, model.vis.global_.fovy = 1, span
    model.vis.quality.offsamples, model.vis.quality.shadowsize = 4, 4096
    model.vis.quality.numslices, model.vis.quality.numstacks = 64, 64
    model.vis.rgba.haze[:] = [1., 1., 1., 1.]
    model.vis.headlight.ambient[:] = [.38, .38, .38]
    model.vis.headlight.diffuse[:] = [.45, .45, .45]
    model.vis.headlight.specular[:] = [.12, .12, .12]
    model.geom_rgba[model.geom("floor").id] = [.94, .945, .955, 1.]
    light_offset = np.r_[outward[:2]*.06, .35]
    model.light_pos[0] = center + light_offset
    model.light_dir[0] = -light_offset
    model.light_diffuse[0] = [.25, .25, .25]
    model.light_ambient[0] = [.04, .04, .04]
    model.stat.center[:] = center
    model.stat.extent = max(float(np.linalg.norm(np.ptp(points, axis=0))), .1)
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.orthographic = 1
    camera.lookat[:] = center
    camera.distance = 3*model.stat.extent
    # MuJoCo azimuth describes the viewing direction, opposite the radial
    # direction from the target to the camera used in the framing calculation.
    camera.azimuth, camera.elevation = azimuth+180., scene["elevation"]
    option = mujoco.MjvOption()
    # Only original display meshes are shown; invisible convex collision geoms
    # remain part of the model, with all their physical parameters intact.
    option.geomgroup[3] = 0
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
    option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0
    option.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = 0
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
            geom.specular, geom.shininess, geom.reflectance = .1, .2, 0.
        rgb = renderer.render().copy()
    path = output_dir / (scene["filename"] + ".png")
    Image.fromarray(rgb).save(path, dpi=(300, 300))

    # The PDF is a lossless wrapper around the native raster render.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(width/300, height/300), dpi=300)
    axes = figure.add_axes([0, 0, 1, 1])
    axes.imshow(rgb)
    axes.set_axis_off()
    figure.savefig(path.with_suffix(".pdf"), dpi=300, pad_inches=0)
    plt.close(figure)
    return dict(**scene, png=str(path), pdf=str(path.with_suffix(".pdf")),
                source_model=str(model_path), source_query=str(query_path),
                model_sha256=checksum(model_path), query_sha256=checksum(query_path),
                time_s=data.time, measured_window_mode=metadata["mode"], force_N=metadata["force_N"],
                qpos=data.qpos.tolist(), camera=dict(azimuth=float(camera.azimuth), elevation=scene["elevation"],
                    lookat=center.tolist(), orthographic_height_m=float(span)),
                size_px=[width, height], dpi=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/collision_prediction_final_20261026")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tro_revision/figures/collision_prediction_scenes")
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=1800)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = [render(args.run_dir.resolve(), args.output_dir.resolve(), scene, args.width, args.height)
              for scene in SCENES]
    (args.output_dir / "scene_renders.json").write_text(json.dumps(dict(
        renderer="MuJoCo native offscreen EGL", mujoco_version=mujoco.__version__,
        changes="camera, lighting, material appearance and neutral floor color only; saved state and geometry unchanged", images=result), indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
