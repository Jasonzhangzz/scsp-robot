import math
import os
import re
import shutil
import subprocess
import time

from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt

try:
    import mujoco
except ImportError:  # pragma: no cover - depends on local simulator install
    mujoco = None

gymapi = None


@dataclass(frozen=True)
class FreeCameraConfig:
    lookat: np.ndarray
    distance: float
    azimuth_deg: float
    elevation_deg: float


@dataclass(frozen=True)
class IsaacGymCameraPose:
    position: np.ndarray
    target: np.ndarray


def _require_mujoco():
    if mujoco is None:
        raise ImportError(
            "MuJoCo is required for the MuJoCo screenshot helpers in planning/screenshot.py"
        )


def _require_isaacgym():
    global gymapi
    if gymapi is not None:
        return gymapi
    try:
        from isaacgym import gymapi as imported_gymapi
    except ImportError as exc:  # pragma: no cover - depends on local simulator install
        raise ImportError(
            "Isaac Gym is required for the Isaac Gym screenshot helpers in planning/screenshot.py"
        ) from exc
    gymapi = imported_gymapi
    return gymapi


def _as_vec3(name, value):
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a 3D vector, got shape {array.shape}")
    return array


def build_free_camera_config(lookat, distance, azimuth_deg, elevation_deg):
    return FreeCameraConfig(
        lookat=_as_vec3("lookat", lookat),
        distance=float(distance),
        azimuth_deg=float(azimuth_deg),
        elevation_deg=float(elevation_deg),
    )


def build_isaacgym_camera_pose(camera_position, camera_target):
    return IsaacGymCameraPose(
        position=_as_vec3("camera_position", camera_position),
        target=_as_vec3("camera_target", camera_target),
    )


def build_free_camera_config_from_position(camera_position, lookat, pitch_deg=None):
    lookat = _as_vec3("lookat", lookat)
    camera_position = _as_vec3("camera_position", camera_position)

    offset = camera_position - lookat
    distance = float(np.linalg.norm(offset))
    if distance < 1e-9:
        raise ValueError("camera_position must be different from lookat")

    xy_distance = float(np.linalg.norm(offset[:2]))
    azimuth_deg = math.degrees(math.atan2(-offset[1], -offset[0]))
    elevation_from_position = math.degrees(
        math.atan2(-offset[2], max(xy_distance, 1e-12))
    )
    elevation_deg = float(pitch_deg) if pitch_deg is not None else float(
        elevation_from_position
    )
    if pitch_deg is not None:
        elevation_rad = math.radians(elevation_deg)
        azimuth_rad = math.radians(azimuth_deg)
        lookat = camera_position + distance * np.array(
            [
                math.cos(elevation_rad) * math.cos(azimuth_rad),
                math.cos(elevation_rad) * math.sin(azimuth_rad),
                math.sin(elevation_rad),
            ],
            dtype=np.float64,
        )

    return FreeCameraConfig(
        lookat=lookat,
        distance=distance,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
    )


def _build_mjv_camera(camera_config):
    _require_mujoco()
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray(camera_config.lookat, dtype=np.float64)
    camera.distance = float(camera_config.distance)
    camera.azimuth = float(camera_config.azimuth_deg)
    camera.elevation = float(camera_config.elevation_deg)
    return camera


def get_model_offscreen_buffer_size(model):
    offscreen_width = int(getattr(model.vis.global_, "offwidth", 640))
    offscreen_height = int(getattr(model.vis.global_, "offheight", 480))
    return max(offscreen_width, 1), max(offscreen_height, 1)


def clamp_render_size(model, requested_width, requested_height):
    requested_width = int(requested_width)
    requested_height = int(requested_height)
    offscreen_width, offscreen_height = get_model_offscreen_buffer_size(model)

    width = min(requested_width, offscreen_width)
    height = min(requested_height, offscreen_height)
    return width, height, offscreen_width, offscreen_height


def save_rgb_frame_to_svg(rgb_frame, output_path):
    rgb_frame = np.asarray(rgb_frame, dtype=np.uint8)
    if rgb_frame.ndim != 3 or rgb_frame.shape[2] not in (3, 4):
        raise ValueError(
            f"rgb_frame must have shape (H, W, 3) or (H, W, 4), got {rgb_frame.shape}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = rgb_frame.shape[:2]
    dpi = 100.0
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(rgb_frame)
    ax.set_axis_off()

    try:
        fig.savefig(output_path, format="svg", dpi=dpi, pad_inches=0.0)
    finally:
        plt.close(fig)


def _infer_next_frame_index(output_dir, filename_prefix):
    output_dir = Path(output_dir)
    pattern = re.compile(rf"^{re.escape(filename_prefix)}_(\d+)_")
    max_frame_index = -1
    for existing_path in output_dir.glob(f"{filename_prefix}_*.svg"):
        match = pattern.match(existing_path.stem)
        if match is None:
            continue
        max_frame_index = max(max_frame_index, int(match.group(1)))
    return max_frame_index + 1


def _make_unique_output_path(output_path):
    output_path = Path(output_path)
    if not output_path.exists():
        return output_path

    suffix_index = 1
    while True:
        candidate = output_path.with_name(
            f"{output_path.stem}_{suffix_index:03d}{output_path.suffix}"
        )
        if not candidate.exists():
            return candidate
        suffix_index += 1


class PeriodicSVGScreenshotRecorder:
    """
    MuJoCo's renderer produces raster frames. Saving them as SVG writes an SVG
    container that embeds the rendered image, which keeps the workflow in .svg
    files even though the scene itself is rasterized.
    """

    def __init__(
        self,
        model,
        output_dir,
        camera_config,
        interval_seconds=2.0,
        width=1280,
        height=960,
        filename_prefix="frame",
        capture_on_start=True,
    ):
        _require_mujoco()
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera = _build_mjv_camera(camera_config)
        self.interval_seconds = float(interval_seconds)
        self.renderer = None
        (
            self.width,
            self.height,
            offscreen_width,
            offscreen_height,
        ) = clamp_render_size(self.model, width, height)
        self.filename_prefix = str(filename_prefix)
        self.frame_index = _infer_next_frame_index(
            self.output_dir,
            self.filename_prefix,
        )
        self.next_capture_time = 0.0 if capture_on_start else self.interval_seconds
        if self.width != int(width) or self.height != int(height):
            print(
                "[PeriodicSVGScreenshotRecorder] Requested render size "
                f"{int(width)}x{int(height)} exceeds MuJoCo offscreen buffer "
                f"{offscreen_width}x{offscreen_height}. "
                f"Clamping to {self.width}x{self.height}."
            )
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

    def _render_rgb_frame(self, data):
        mujoco.mj_forward(self.model, data)
        self.renderer.update_scene(data, camera=self.camera)
        return self.renderer.render()

    def capture(self, data, sim_time=None, label=None):
        if sim_time is None:
            sim_time = float(data.time)
        rgb_frame = self._render_rgb_frame(data)
        base_name = f"{self.filename_prefix}_{self.frame_index:04d}_t{sim_time:07.2f}"
        if label:
            base_name = f"{base_name}_{label}"
        output_path = self.output_dir / f"{base_name}.svg"
        output_path = _make_unique_output_path(output_path)
        save_rgb_frame_to_svg(rgb_frame, output_path)
        self.frame_index += 1
        return output_path

    def capture_if_due(self, data):
        if self.interval_seconds <= 0.0:
            return None

        sim_time = float(data.time)
        if sim_time + 1e-9 < self.next_capture_time:
            return None

        while self.next_capture_time <= sim_time + 1e-9:
            self.next_capture_time += self.interval_seconds
        return self.capture(data, sim_time=sim_time)

    def close(self):
        if hasattr(self.renderer, "close"):
            self.renderer.close()


class PeriodicMuJoCoMP4Recorder:
    """
    Streams MuJoCo offscreen-rendered RGB frames into ffmpeg and writes an .mp4 file.
    """

    def __init__(
        self,
        model,
        output_path,
        camera_config,
        fps=20.0,
        width=1280,
        height=960,
        capture_on_start=True,
        codec="libx264",
        pixel_format="yuv420p",
        encoder_preset="slow",
        encoder_crf=15,
        wall_clock_timing=True,
    ):
        _require_mujoco()
        self.model = model
        self.output_path = _make_unique_output_path(Path(output_path))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.camera = _build_mjv_camera(camera_config)
        self.fps = float(fps)
        if self.fps <= 0.0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        self.interval_seconds = 1.0 / self.fps
        self.codec = str(codec)
        self.pixel_format = str(pixel_format)
        self.encoder_preset = str(encoder_preset)
        self.encoder_crf = int(encoder_crf)
        self.wall_clock_timing = bool(wall_clock_timing)
        self.next_capture_time = 0.0 if capture_on_start else self.interval_seconds
        self._start_wall_time = time.perf_counter()
        self.next_capture_wall_time = (
            self._start_wall_time
            if capture_on_start
            else self._start_wall_time + self.interval_seconds
        )
        self.frames_written = 0
        self._closed = False
        self._ffmpeg_process = None
        (
            self.width,
            self.height,
            offscreen_width,
            offscreen_height,
        ) = clamp_render_size(self.model, width, height)
        if self.width != int(width) or self.height != int(height):
            print(
                "[PeriodicMuJoCoMP4Recorder] Requested render size "
                f"{int(width)}x{int(height)} exceeds MuJoCo offscreen buffer "
                f"{offscreen_width}x{offscreen_height}. "
                f"Clamping to {self.width}x{self.height}."
            )
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError(
                "ffmpeg is required for MuJoCo MP4 recording in planning/screenshot.py. "
                "Install ffmpeg or disable video recording."
            )
        self._ffmpeg_path = ffmpeg_path
        self._start_ffmpeg_process()

    def _start_ffmpeg_process(self):
        command = [
            self._ffmpeg_path,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            self.codec,
            "-preset",
            self.encoder_preset,
            "-crf",
            str(self.encoder_crf),
            "-pix_fmt",
            self.pixel_format,
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]
        self._ffmpeg_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def _render_rgb_frame(self, data):
        mujoco.mj_forward(self.model, data)
        self.renderer.update_scene(data, camera=self.camera)
        return self.renderer.render()

    def _write_rgb_frame(self, rgb_frame, repeat=1):
        if self._ffmpeg_process is None or self._ffmpeg_process.stdin is None:
            raise RuntimeError("MP4 recorder ffmpeg process is not available")
        rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
        payload = rgb_frame.tobytes()
        try:
            for _ in range(max(int(repeat), 1)):
                self._ffmpeg_process.stdin.write(payload)
                self.frames_written += 1
        except BrokenPipeError as exc:
            raise RuntimeError(
                "ffmpeg terminated unexpectedly while writing MuJoCo MP4 frames"
            ) from exc

    def capture(self, data, sim_time=None, repeat=1):
        if self._closed:
            return None
        if sim_time is None:
            sim_time = float(data.time)
        rgb_frame = self._render_rgb_frame(data)
        self._write_rgb_frame(rgb_frame[:, :, :3], repeat=repeat)
        return sim_time

    def capture_if_due(self, data):
        if self._closed:
            return None

        if self.wall_clock_timing:
            now = time.perf_counter()
            if now + 1e-9 < self.next_capture_wall_time:
                return None
            repeat = 0
            while self.next_capture_wall_time <= now + 1e-9:
                self.next_capture_wall_time += self.interval_seconds
                repeat += 1
            return self.capture(data, repeat=repeat)

        sim_time = float(data.time)
        if sim_time + 1e-9 < self.next_capture_time:
            return None
        repeat = 0
        while self.next_capture_time <= sim_time + 1e-9:
            self.next_capture_time += self.interval_seconds
            repeat += 1
        return self.capture(data, sim_time=sim_time, repeat=repeat)

    def close(self):
        if self._closed:
            return self.output_path if self.frames_written > 0 else None

        self._closed = True
        return_code = 0
        stderr_text = ""
        try:
            if self._ffmpeg_process is not None:
                if self._ffmpeg_process.stdin is not None:
                    self._ffmpeg_process.stdin.close()
                stderr_bytes = (
                    self._ffmpeg_process.stderr.read()
                    if self._ffmpeg_process.stderr is not None
                    else b""
                )
                return_code = self._ffmpeg_process.wait()
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                self._ffmpeg_process = None
        finally:
            if hasattr(self.renderer, "close"):
                self.renderer.close()

        if return_code != 0:
            raise RuntimeError(
                "ffmpeg failed while finalizing MuJoCo MP4 recording"
                + (f": {stderr_text}" if stderr_text else ".")
            )

        if self.frames_written <= 0:
            try:
                self.output_path.unlink()
            except FileNotFoundError:
                pass
            return None

        return self.output_path


def create_mujoco_mp4_recorder(
    model,
    output_path,
    camera_config,
    fps=20.0,
    width=1280,
    height=960,
    capture_on_start=True,
    codec="libx264",
    pixel_format="yuv420p",
    encoder_preset="slow",
    encoder_crf=15,
    wall_clock_timing=True,
):
    return PeriodicMuJoCoMP4Recorder(
        model=model,
        output_path=output_path,
        camera_config=camera_config,
        fps=fps,
        width=width,
        height=height,
        capture_on_start=capture_on_start,
        codec=codec,
        pixel_format=pixel_format,
        encoder_preset=encoder_preset,
        encoder_crf=encoder_crf,
        wall_clock_timing=wall_clock_timing,
    )


def _reshape_isaacgym_color_image(color_image, width, height):
    color_image = np.asarray(color_image, dtype=np.uint8)

    if color_image.ndim == 3 and color_image.shape == (height, width, 4):
        return color_image
    if color_image.ndim == 3 and color_image.shape == (height, width, 3):
        return color_image
    if color_image.ndim == 2 and color_image.shape == (height, width * 4):
        return color_image.reshape(height, width, 4)
    if color_image.size == height * width * 4:
        return color_image.reshape(height, width, 4)
    if color_image.size == height * width * 3:
        return color_image.reshape(height, width, 3)

    raise ValueError(
        "Unsupported Isaac Gym IMAGE_COLOR shape: "
        f"{color_image.shape}, expected {(height, width, 4)} or packed equivalent."
    )


class PeriodicIsaacGymSVGScreenshotRecorder:
    """
    Isaac Gym camera sensors render raster RGBA frames. Saving them as SVG writes
    an SVG container that embeds the rendered image, mirroring the MuJoCo helper.
    """

    def __init__(
        self,
        gym,
        sim,
        env,
        output_dir,
        camera_pose,
        interval_seconds=1.0,
        width=1280,
        height=960,
        filename_prefix="frame",
        capture_on_start=True,
    ):
        _require_isaacgym()
        self.gym = gym
        self.sim = sim
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = float(interval_seconds)
        self.width = int(width)
        self.height = int(height)
        self.filename_prefix = str(filename_prefix)
        self.frame_index = _infer_next_frame_index(self.output_dir, self.filename_prefix)
        self.next_capture_time = 0.0 if capture_on_start else self.interval_seconds
        self.camera_pose = build_isaacgym_camera_pose(
            camera_pose.position,
            camera_pose.target,
        )

        cam_props = gymapi.CameraProperties()
        cam_props.width = self.width
        cam_props.height = self.height
        self.camera_handle = self.gym.create_camera_sensor(self.env, cam_props)
        if self.camera_handle is None:
            raise RuntimeError("Failed to create Isaac Gym camera sensor for SVG screenshots")
        self._apply_camera_pose(self.camera_pose)

    def _apply_camera_pose(self, camera_pose):
        cam_pos = gymapi.Vec3(
            float(camera_pose.position[0]),
            float(camera_pose.position[1]),
            float(camera_pose.position[2]),
        )
        cam_target = gymapi.Vec3(
            float(camera_pose.target[0]),
            float(camera_pose.target[1]),
            float(camera_pose.target[2]),
        )
        self.gym.set_camera_location(self.camera_handle, self.env, cam_pos, cam_target)

    def set_camera_pose(self, camera_position, camera_target):
        self.camera_pose = build_isaacgym_camera_pose(camera_position, camera_target)
        self._apply_camera_pose(self.camera_pose)

    def _render_color_frame(self, step_graphics=True):
        if step_graphics:
            self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        color_image = self.gym.get_camera_image(
            self.sim,
            self.env,
            self.camera_handle,
            gymapi.IMAGE_COLOR,
        )
        return _reshape_isaacgym_color_image(color_image, self.width, self.height)

    def capture(self, sim_time=None, label=None, step_graphics=True):
        if sim_time is None:
            sim_time = float(self.gym.get_sim_time(self.sim))
        color_frame = self._render_color_frame(step_graphics=step_graphics)
        base_name = f"{self.filename_prefix}_{self.frame_index:04d}_t{sim_time:07.2f}"
        if label:
            base_name = f"{base_name}_{label}"
        output_path = self.output_dir / f"{base_name}.svg"
        output_path = _make_unique_output_path(output_path)
        save_rgb_frame_to_svg(color_frame, output_path)
        self.frame_index += 1
        return output_path

    def capture_if_due(self, sim_time=None, step_graphics=True):
        if self.interval_seconds <= 0.0:
            return None

        if sim_time is None:
            sim_time = float(self.gym.get_sim_time(self.sim))

        if sim_time + 1e-9 < self.next_capture_time:
            return None

        while self.next_capture_time <= sim_time + 1e-9:
            self.next_capture_time += self.interval_seconds
        return self.capture(sim_time=sim_time, step_graphics=step_graphics)

    def close(self):
        if self.camera_handle is not None:
            self.gym.destroy_camera_sensor(self.sim, self.env, self.camera_handle)
            self.camera_handle = None


class PeriodicIsaacGymMP4Recorder:
    """
    Streams Isaac Gym camera frames into ffmpeg and writes an .mp4 file directly.
    """

    def __init__(
        self,
        gym,
        sim,
        env,
        output_path,
        camera_pose,
        fps=20.0,
        width=1280,
        height=960,
        capture_on_start=True,
        codec="libx264",
        pixel_format="yuv420p",
        encoder_preset="slow",
        encoder_crf=15,
        wall_clock_timing=True,
        supersampling=2,
    ):
        _require_isaacgym()
        self.gym = gym
        self.sim = sim
        self.env = env
        self.output_path = _make_unique_output_path(Path(output_path))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        if self.fps <= 0.0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        self.interval_seconds = 1.0 / self.fps
        self.width = int(width)
        self.height = int(height)
        self.codec = str(codec)
        self.pixel_format = str(pixel_format)
        self.encoder_preset = str(encoder_preset)
        self.encoder_crf = int(encoder_crf)
        self.wall_clock_timing = bool(wall_clock_timing)
        self.supersampling = max(int(supersampling), 1)
        self.next_capture_time = 0.0 if capture_on_start else self.interval_seconds
        self._start_wall_time = time.perf_counter()
        self.next_capture_wall_time = self._start_wall_time if capture_on_start else self._start_wall_time + self.interval_seconds
        self.camera_pose = build_isaacgym_camera_pose(
            camera_pose.position,
            camera_pose.target,
        )
        self.frames_written = 0
        self._closed = False
        self._ffmpeg_process = None

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError(
                "ffmpeg is required for Isaac Gym MP4 recording in planning/screenshot.py. "
                "Install ffmpeg or disable video recording."
            )
        self._ffmpeg_path = ffmpeg_path

        cam_props = gymapi.CameraProperties()
        cam_props.width = self.width
        cam_props.height = self.height
        if hasattr(cam_props, "supersampling_horizontal"):
            cam_props.supersampling_horizontal = self.supersampling
        if hasattr(cam_props, "supersampling_vertical"):
            cam_props.supersampling_vertical = self.supersampling
        self.camera_handle = self.gym.create_camera_sensor(self.env, cam_props)
        if self.camera_handle is None:
            raise RuntimeError("Failed to create Isaac Gym camera sensor for MP4 recording")
        self._apply_camera_pose(self.camera_pose)
        self._start_ffmpeg_process()

    def _start_ffmpeg_process(self):
        command = [
            self._ffmpeg_path,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            self.codec,
            "-preset",
            self.encoder_preset,
            "-crf",
            str(self.encoder_crf),
            "-pix_fmt",
            self.pixel_format,
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]
        self._ffmpeg_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def _apply_camera_pose(self, camera_pose):
        cam_pos = gymapi.Vec3(
            float(camera_pose.position[0]),
            float(camera_pose.position[1]),
            float(camera_pose.position[2]),
        )
        cam_target = gymapi.Vec3(
            float(camera_pose.target[0]),
            float(camera_pose.target[1]),
            float(camera_pose.target[2]),
        )
        self.gym.set_camera_location(self.camera_handle, self.env, cam_pos, cam_target)

    def set_camera_pose(self, camera_position, camera_target):
        self.camera_pose = build_isaacgym_camera_pose(camera_position, camera_target)
        self._apply_camera_pose(self.camera_pose)

    def _render_color_frame(self, step_graphics=True):
        if step_graphics:
            self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        color_image = self.gym.get_camera_image(
            self.sim,
            self.env,
            self.camera_handle,
            gymapi.IMAGE_COLOR,
        )
        return _reshape_isaacgym_color_image(color_image, self.width, self.height)

    def _write_rgb_frame(self, rgb_frame, repeat=1):
        if self._ffmpeg_process is None or self._ffmpeg_process.stdin is None:
            raise RuntimeError("MP4 recorder ffmpeg process is not available")
        rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
        payload = rgb_frame.tobytes()
        try:
            for _ in range(max(int(repeat), 1)):
                self._ffmpeg_process.stdin.write(payload)
                self.frames_written += 1
        except BrokenPipeError as exc:
            raise RuntimeError("ffmpeg terminated unexpectedly while writing MP4 frames") from exc

    def capture(self, sim_time=None, step_graphics=True, repeat=1):
        if self._closed:
            return None
        if sim_time is None:
            sim_time = float(self.gym.get_sim_time(self.sim))
        color_frame = self._render_color_frame(step_graphics=step_graphics)
        rgb_frame = np.ascontiguousarray(color_frame[:, :, :3], dtype=np.uint8)
        self._write_rgb_frame(rgb_frame, repeat=repeat)
        return sim_time

    def capture_if_due(self, sim_time=None, step_graphics=True):
        if self._closed:
            return None
        if self.wall_clock_timing:
            now = time.perf_counter()
            if now + 1e-9 < self.next_capture_wall_time:
                return None
            repeat = 0
            while self.next_capture_wall_time <= now + 1e-9:
                self.next_capture_wall_time += self.interval_seconds
                repeat += 1
            return self.capture(sim_time=sim_time, step_graphics=step_graphics, repeat=repeat)
        if sim_time is None:
            sim_time = float(self.gym.get_sim_time(self.sim))
        if sim_time + 1e-9 < self.next_capture_time:
            return None
        while self.next_capture_time <= sim_time + 1e-9:
            self.next_capture_time += self.interval_seconds
        return self.capture(sim_time=sim_time, step_graphics=step_graphics)

    def close(self):
        if self._closed:
            return self.output_path if self.frames_written > 0 else None

        self._closed = True
        return_code = 0
        stderr_text = ""
        try:
            if self._ffmpeg_process is not None:
                if self._ffmpeg_process.stdin is not None:
                    self._ffmpeg_process.stdin.close()
                stderr_bytes = self._ffmpeg_process.stderr.read() if self._ffmpeg_process.stderr is not None else b""
                return_code = self._ffmpeg_process.wait()
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                self._ffmpeg_process = None
        finally:
            if self.camera_handle is not None:
                self.gym.destroy_camera_sensor(self.sim, self.env, self.camera_handle)
                self.camera_handle = None

        if return_code != 0:
            raise RuntimeError(
                "ffmpeg failed while finalizing MP4 recording"
                + (f": {stderr_text}" if stderr_text else ".")
            )

        if self.frames_written <= 0:
            try:
                self.output_path.unlink()
            except FileNotFoundError:
                pass
            return None

        return self.output_path


def create_isaacgym_svg_screenshot_recorder(
    gym,
    sim,
    env,
    output_dir,
    camera_position,
    camera_target,
    interval_seconds=1.0,
    width=1280,
    height=960,
    filename_prefix="frame",
    capture_on_start=True,
):
    camera_pose = build_isaacgym_camera_pose(camera_position, camera_target)
    return PeriodicIsaacGymSVGScreenshotRecorder(
        gym=gym,
        sim=sim,
        env=env,
        output_dir=output_dir,
        camera_pose=camera_pose,
        interval_seconds=interval_seconds,
        width=width,
        height=height,
        filename_prefix=filename_prefix,
        capture_on_start=capture_on_start,
    )


def create_isaacgym_mp4_recorder(
    gym,
    sim,
    env,
    output_path,
    camera_position,
    camera_target,
    fps=20.0,
    width=1280,
    height=960,
    capture_on_start=True,
    codec="libx264",
    pixel_format="yuv420p",
    encoder_preset="slow",
    encoder_crf=15,
    wall_clock_timing=True,
    supersampling=2,
):
    camera_pose = build_isaacgym_camera_pose(camera_position, camera_target)
    return PeriodicIsaacGymMP4Recorder(
        gym=gym,
        sim=sim,
        env=env,
        output_path=output_path,
        camera_pose=camera_pose,
        fps=fps,
        width=width,
        height=height,
        capture_on_start=capture_on_start,
        codec=codec,
        pixel_format=pixel_format,
        encoder_preset=encoder_preset,
        encoder_crf=encoder_crf,
        wall_clock_timing=wall_clock_timing,
        supersampling=supersampling,
    )
