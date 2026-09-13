import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if p.name == "scsp-robot"), None) or next(p for p in Path(__file__).resolve().parents if (p / "planning" / "acados_env.py").is_file())
DEFAULT_INPUT_DIR = REPO_ROOT / "outputs" / "videos_panda"

SVG_SCREENSHOT_CAMERA_POSITION = np.array([0.7, 0.0, 0.63], dtype=np.float32)
SVG_SCREENSHOT_CAMERA_TARGET = np.array([0.1, 0.0, 0.32], dtype=np.float32)
POSE_AXIS_LENGTH = 0.08
POSE_AXIS_RADIUS = 0.0015
POSE_AXIS_CENTER_RADIUS = 0.004
gymapi = None


def _require_isaacgym():
    global gymapi
    if gymapi is not None:
        return gymapi
    local_isaacgym_python = REPO_ROOT / "DyWA" / "isaacgym" / "python"
    if local_isaacgym_python.exists():
        sys.path.insert(0, str(local_isaacgym_python))
    try:
        from isaacgym import gymapi as imported_gymapi
    except ImportError as exc:
        raise ImportError(
            "Isaac Gym is required to render goal-pose PNGs. "
            "Run this script from the environment where isaacgym is installed."
        ) from exc
    gymapi = imported_gymapi
    return gymapi


def _normalize_quat_wxyz(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat_wxyz))
    if norm < 1e-12:
        raise ValueError(f"invalid zero-length quaternion: {quat_wxyz}")
    return (quat_wxyz / norm).astype(np.float32)


def _quat_wxyz_to_xyzw(quat_wxyz):
    quat_wxyz = _normalize_quat_wxyz(quat_wxyz)
    return np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
        dtype=np.float32,
    )


def _quat_wxyz_to_matrix(quat_wxyz):
    w, x, y, z = _normalize_quat_wxyz(quat_wxyz).astype(np.float64)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _reshape_isaacgym_color_image(color_image, width, height):
    color_image = np.asarray(color_image, dtype=np.uint8)
    if color_image.ndim == 3 and color_image.shape in ((height, width, 4), (height, width, 3)):
        return color_image
    if color_image.ndim == 2 and color_image.shape == (height, width * 4):
        return color_image.reshape(height, width, 4)
    if color_image.size == height * width * 4:
        return color_image.reshape(height, width, 4)
    if color_image.size == height * width * 3:
        return color_image.reshape(height, width, 3)
    raise ValueError(f"unsupported Isaac Gym color image shape: {color_image.shape}")


def _write_png(color_frame, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_frame = np.asarray(color_frame[:, :, :3], dtype=np.uint8)
    try:
        from PIL import Image

        Image.fromarray(rgb_frame).save(output_path)
        return
    except ImportError:
        pass

    try:
        import imageio.v2 as imageio

        imageio.imwrite(output_path, rgb_frame)
        return
    except ImportError:
        pass

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.imsave(output_path, rgb_frame)


def _camera_view_angles(camera_position, camera_target):
    camera_position = np.asarray(camera_position, dtype=np.float64).reshape(3)
    camera_target = np.asarray(camera_target, dtype=np.float64).reshape(3)
    offset = camera_position - camera_target
    distance = float(np.linalg.norm(offset))
    xy_distance = float(np.linalg.norm(offset[:2]))
    if distance < 1e-9:
        raise ValueError("camera position and target must be different")
    elev = math.degrees(math.atan2(offset[2], max(xy_distance, 1e-12)))
    azim = math.degrees(math.atan2(offset[1], offset[0]))
    return elev, azim


def _parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("1", "true", "t", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def _render_goal_pose_matplotlib(goal_xyz, goal_quat_wxyz, output_path, width, height, background=True):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dpi = 100.0
    fig = plt.figure(figsize=(float(width) / dpi, float(height) / dpi), dpi=dpi)
    try:
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    except AttributeError:
        ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    if background:
        table_z = 0.35
        table_corners = np.array(
            [
                [0.2, -1.0, table_z],
                [2.2, -1.0, table_z],
                [2.2, 1.0, table_z],
                [0.2, 1.0, table_z],
            ],
            dtype=np.float32,
        )
        table = Poly3DCollection(
            [table_corners],
            facecolors=(0.58, 0.54, 0.49, 1.0),
            edgecolors="none",
            zorder=0,
        )
        ax.add_collection3d(table)

    goal_xyz = np.asarray(goal_xyz, dtype=np.float32).reshape(3)
    rot = _quat_wxyz_to_matrix(goal_quat_wxyz)
    colors = ["#ff3333", "#33ff33", "#3373ff"]
    for axis_idx, color in enumerate(colors):
        direction = rot[:, axis_idx] * POSE_AXIS_LENGTH
        ax.quiver(
            goal_xyz[0],
            goal_xyz[1],
            goal_xyz[2],
            direction[0],
            direction[1],
            direction[2],
            color=color,
            linewidth=4.0,
            arrow_length_ratio=0.18,
            normalize=False,
            zorder=10,
        )
        axis_end = goal_xyz + direction
        ax.plot(
            [goal_xyz[0], axis_end[0]],
            [goal_xyz[1], axis_end[1]],
            [goal_xyz[2], axis_end[2]],
            color=color,
            linewidth=5.0,
            zorder=11,
        )
    ax.scatter(
        [goal_xyz[0]],
        [goal_xyz[1]],
        [goal_xyz[2]],
        c=[(0.55, 0.55, 0.55)],
        s=60,
        depthshade=False,
        zorder=12,
    )

    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(-0.35, 0.35)
    ax.set_zlim(0.30, 0.60)
    ax.set_box_aspect((0.8, 0.7, 0.3))
    elev, azim = _camera_view_angles(SVG_SCREENSHOT_CAMERA_POSITION, SVG_SCREENSHOT_CAMERA_TARGET)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.grid(False)
    alpha = 1.0 if background else 0.0
    ax.set_facecolor((1.0, 1.0, 1.0, alpha))
    fig.patch.set_facecolor((1.0, 1.0, 1.0, alpha))

    try:
        fig.savefig(output_path, dpi=dpi, pad_inches=0.0, transparent=not background)
    finally:
        plt.close(fig)


def _prepare_pose_axes_urdf_asset(repo_root):
    asset_root = repo_root / "envs" / "assets" / "objects" / "_isaac_tmp"
    asset_root.mkdir(parents=True, exist_ok=True)
    pose_axes_urdf_rel = "goal_pose_axes_visual.urdf"
    pose_axes_urdf_abs = asset_root / pose_axes_urdf_rel

    pose_axes_urdf = f"""<?xml version="1.0"?>
<robot name="goal_pose_axes_visual">
  <link name="base">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="{POSE_AXIS_CENTER_RADIUS}"/>
      </geometry>
      <material name="pose_axis_center">
        <color rgba="0.55 0.55 0.55 1.0"/>
      </material>
    </visual>
    <visual>
      <origin xyz="{0.5 * POSE_AXIS_LENGTH} 0 0" rpy="0 1.57079632679 0"/>
      <geometry>
        <cylinder radius="{POSE_AXIS_RADIUS}" length="{POSE_AXIS_LENGTH}"/>
      </geometry>
      <material name="pose_axis_x">
        <color rgba="1.0 0.2 0.2 1.0"/>
      </material>
    </visual>
    <visual>
      <origin xyz="0 {0.5 * POSE_AXIS_LENGTH} 0" rpy="-1.57079632679 0 0"/>
      <geometry>
        <cylinder radius="{POSE_AXIS_RADIUS}" length="{POSE_AXIS_LENGTH}"/>
      </geometry>
      <material name="pose_axis_y">
        <color rgba="0.2 1.0 0.2 1.0"/>
      </material>
    </visual>
    <visual>
      <origin xyz="0 0 {0.5 * POSE_AXIS_LENGTH}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{POSE_AXIS_RADIUS}" length="{POSE_AXIS_LENGTH}"/>
      </geometry>
      <material name="pose_axis_z">
        <color rgba="0.2 0.45 1.0 1.0"/>
      </material>
    </visual>
  </link>
</robot>
"""
    pose_axes_urdf_abs.write_text(pose_axes_urdf, encoding="ascii")
    return pose_axes_urdf_rel, str(asset_root)


def _create_sim(gym, sim_device, graphics_device_id, width, height, background=True):
    _require_isaacgym()
    sim_params = gymapi.SimParams()
    sim_dt = 0.0125
    sim_substeps = 1
    compute_id = int(str(sim_device).split(":")[-1]) if ":" in str(sim_device) else 0

    sim_params.dt = sim_dt
    sim_params.substeps = sim_substeps
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    # Match test_scsp.py: it uses CPU-style get/set actor state APIs.
    sim_params.use_gpu_pipeline = False
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.bounce_threshold_velocity = 2.0 * 9.81 * sim_dt / max(sim_substeps, 1)
    sim_params.physx.max_depenetration_velocity = 10.0
    sim_params.physx.use_gpu = compute_id >= 0

    sim = gym.create_sim(compute_id, int(graphics_device_id), gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("failed to create Isaac Gym sim")

    if background:
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        gym.add_ground(sim, plane)

    env = gym.create_env(
        sim,
        gymapi.Vec3(-2.0, -2.0, 0.0),
        gymapi.Vec3(2.0, 2.0, 2.0),
        1,
    )

    if background:
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = gym.create_box(sim, 2.0, 2.0, 0.35, table_opts)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(1.2, 0.0, 0.175)
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        table_actor = gym.create_actor(env, table_asset, table_pose, "table", 0, 0)
        for body_idx in range(gym.get_actor_rigid_body_count(env, table_actor)):
            gym.set_rigid_body_color(
                env,
                table_actor,
                body_idx,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.45, 0.47, 0.50),
            )

    axes_urdf_rel, axes_asset_root = _prepare_pose_axes_urdf_asset(REPO_ROOT)
    axes_opts = gymapi.AssetOptions()
    axes_opts.fix_base_link = True
    axes_opts.disable_gravity = True
    axes_asset = gym.load_asset(sim, axes_asset_root, axes_urdf_rel, axes_opts)

    camera_props = gymapi.CameraProperties()
    camera_props.width = int(width)
    camera_props.height = int(height)
    if hasattr(camera_props, "supersampling_horizontal"):
        camera_props.supersampling_horizontal = 2
    if hasattr(camera_props, "supersampling_vertical"):
        camera_props.supersampling_vertical = 2
    camera_handle = gym.create_camera_sensor(env, camera_props)
    if camera_handle is None:
        raise RuntimeError("failed to create Isaac Gym camera sensor")
    gym.set_camera_location(
        camera_handle,
        env,
        gymapi.Vec3(*[float(v) for v in SVG_SCREENSHOT_CAMERA_POSITION]),
        gymapi.Vec3(*[float(v) for v in SVG_SCREENSHOT_CAMERA_TARGET]),
    )

    gym.prepare_sim(sim)
    return sim, env, axes_asset, camera_handle


def _set_actor_pose(gym, env, actor, goal_xyz, goal_quat_wxyz):
    quat_xyzw = _quat_wxyz_to_xyzw(goal_quat_wxyz)
    state = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
    state["pose"]["p"][0] = (float(goal_xyz[0]), float(goal_xyz[1]), float(goal_xyz[2]))
    state["pose"]["r"][0] = (
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
        float(quat_xyzw[3]),
    )
    state["vel"]["linear"][0] = (0.0, 0.0, 0.0)
    state["vel"]["angular"][0] = (0.0, 0.0, 0.0)
    gym.set_actor_rigid_body_states(env, actor, state, gymapi.STATE_ALL)


def _capture_png(gym, sim, env, camera_handle, output_path, width, height):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    color_image = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
    color_frame = _reshape_isaacgym_color_image(color_image, int(width), int(height))
    _write_png(color_frame, output_path)


def _load_trials(trails_path):
    with open(trails_path, "r", encoding="utf-8") as f:
        trials = json.load(f)
    if not isinstance(trials, list):
        raise ValueError(f"{trails_path} must contain a JSON list")
    return trials


def _iter_goal_pose_outputs(input_dir, output_name):
    for object_dir in sorted(path for path in Path(input_dir).iterdir() if path.is_dir()):
        trails_path = object_dir / "trails.json"
        if not trails_path.exists():
            continue
        trials = _load_trials(trails_path)
        for fallback_idx, trial in enumerate(trials):
            trial_id = int(trial.get("trial", fallback_idx))
            goal_xyz = np.asarray(trial["goal_xyz"], dtype=np.float32).reshape(3)
            goal_quat_wxyz = np.asarray(trial["goal_quat_wxyz"], dtype=np.float32).reshape(4)
            output_path = object_dir / f"{trial_id:03d}" / output_name
            yield goal_xyz, goal_quat_wxyz, output_path


def generate_goal_pose_images_matplotlib(args):
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")

    generated = 0
    for goal_xyz, goal_quat_wxyz, output_path in _iter_goal_pose_outputs(input_dir, args.output_name):
        _render_goal_pose_matplotlib(
            goal_xyz,
            goal_quat_wxyz,
            output_path,
            args.width,
            args.height,
            background=args.background,
        )
        generated += 1
        print(f"saved {output_path}")
    print(f"generated {generated} goal-pose PNG images")


def generate_goal_pose_images_isaac(args):
    _require_isaacgym()
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")

    gym = gymapi.acquire_gym()
    sim = None
    camera_handle = None
    try:
        sim, env, axes_asset, camera_handle = _create_sim(
            gym,
            sim_device=args.sim_device,
            graphics_device_id=args.graphics_device_id,
            width=args.width,
            height=args.height,
            background=args.background,
        )

        axes_pose = gymapi.Transform()
        axes_pose.p = gymapi.Vec3(0.0, 0.0, -10.0)
        axes_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        axes_actor = gym.create_actor(env, axes_asset, axes_pose, "goal_pose_axes", 0, 0)

        generated = 0
        for goal_xyz, goal_quat_wxyz, output_path in _iter_goal_pose_outputs(input_dir, args.output_name):
            _set_actor_pose(gym, env, axes_actor, goal_xyz, goal_quat_wxyz)
            _capture_png(gym, sim, env, camera_handle, output_path, args.width, args.height)
            generated += 1
            print(f"saved {output_path}")

        print(f"generated {generated} goal-pose PNG images")
    finally:
        if sim is not None:
            if camera_handle is not None:
                try:
                    gym.destroy_camera_sensor(sim, env, camera_handle)
                except Exception:
                    pass
            gym.destroy_sim(sim)


def generate_goal_pose_images(args):
    if not args.background:
        generate_goal_pose_images_matplotlib(args)
    elif args.renderer == "matplotlib":
        generate_goal_pose_images_matplotlib(args)
    else:
        generate_goal_pose_images_isaac(args)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render table-only Isaac Gym PNGs with a red/green/blue goal-pose axis "
            "for each trial listed in outputs/videos_panda/*/trails.json."
        )
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing object subfolders.")
    parser.add_argument("--output-name", default="goal_pose.png", help="PNG filename saved inside each trial folder.")
    parser.add_argument("--width", type=int, default=1280, help="Camera image width.")
    parser.add_argument("--height", type=int, default=960, help="Camera image height.")
    parser.add_argument(
        "--background",
        type=_parse_bool_arg,
        default=True,
        help=(
            "Whether to render the floor/table/background. "
            "Use --background False to save transparent RGBA PNGs with only the three pose axes."
        ),
    )
    parser.add_argument(
        "--renderer",
        choices=("matplotlib", "isaac"),
        default="isaac",
        help="Renderer to use. isaac matches test_scsp.py; matplotlib is only a portable fallback.",
    )
    parser.add_argument("--sim-device", default="cuda:0", help="Isaac Gym simulation device, e.g. cuda:0 or cpu.")
    parser.add_argument("--graphics-device-id", type=int, default=0, help="Isaac Gym graphics device id.")
    return parser.parse_args()


if __name__ == "__main__":
    generate_goal_pose_images(parse_args())
