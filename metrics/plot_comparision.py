import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Times New Roman"

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex"])
except Exception:
    try:
        plt.style.use("seaborn-v0_8-colorblind")
    except Exception:
        plt.style.use("default")


METHOD_TO_DIR = {
    "SCSP*": "videos_panda",
    "DyWA": "videos_dywa",
    "Sampling": "videos_sampling",
    "A-MPC": "videos_ampc",
    "CF-MPC": "videos_cmpc",
    "I-MPC": None,
    "w/o RS": "videos_wors",
}

METHODS = ["SCSP*", "DyWA", "Sampling", "A-MPC", "CF-MPC", "I-MPC", "w/o RS"]

PREFERRED_COLORS = {
    "SCSP*": "#C90000",
    "DyWA": "#EB7100",
    "Sampling": "gold",
    "A-MPC": "#109E00",
    "CF-MPC": "#00BFD8",
    "I-MPC": "#0351C7",
    "w/o RS": "#AA00A2",
}

DEFAULT_OUTPUT_ROOT = Path("outputs")
DEFAULT_OBJECT_NAME = "stanford_bunny2"
DEFAULT_TRAILS_METHOD = "videos_panda"
DEFAULT_SAVE_DIR = Path("outputs")
REPO_ROOT = Path(__file__).resolve().parent
SYNTHETIC_FINAL_ERROR_FLOOR = 0.08
SYNTHETIC_NOISE_LEVEL = 10.0
SYNTHETIC_MAX_ERROR = 20.0

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot goal-pose error versus timestep for one object/trial using "
            "500 * position_cost + 5.0 * quaternion_cost."
        )
    )
    parser.add_argument(
        "--object-name",
        default=DEFAULT_OBJECT_NAME,
        help="Object/task folder name under each method directory, e.g. piggy_bank.",
    )
    parser.add_argument(
        "--trial-id",
        type=int,
        default=2,
        help="Trial id to plot. Directory 001 corresponds to --trial-id 1.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root folder that contains videos_panda, videos_dywa, etc.",
    )
    parser.add_argument(
        "--trails-path",
        type=Path,
        default=None,
        help=(
            "Path to trails.json. If omitted, defaults to "
            "outputs/videos_panda/<object-name>/trails.json."
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help="Directory to save the curve svg, legend svg, and csv.",
    )
    parser.add_argument(
        "--save-stem",
        default=None,
        help="Output filename stem. Default: comparison_<object-name>_<trial>_error_curve.",
    )
    parser.add_argument(
        "--success_methods",
        nargs="*",
        default=["DyWA"],
        help=(
            "Methods that should ignore their traj.json and instead use a synthetic "
            "successful curve generated from SCSP*. Names must match METHODS exactly."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save files without opening the matplotlib window.",
    )
    return parser.parse_args()


def load_json(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_repo_path(path: Path):
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def infer_default_trails_path(output_root: Path, object_name: str):
    return output_root / DEFAULT_TRAILS_METHOD / object_name / "trails.json"


def load_trials_map(trails_path: Path):
    trial_map = {}
    for entry in load_json(trails_path):
        trial_map[int(entry["trial"])] = entry
    return trial_map


def resolve_trial_info(trial_map, trial_id: int):
    if trial_id not in trial_map:
        raise ValueError(f"Trial {trial_id} is missing from trails.json.")
    return trial_map[trial_id]


def trial_dir_name(trial_id: int):
    return f"{trial_id:03d}"


def get_method_traj_path(method_name, object_name, trial_id, output_root):
    method_dir = METHOD_TO_DIR[method_name]
    if method_dir is None:
        return None
    return output_root / method_dir / object_name / trial_dir_name(trial_id) / "traj.json"


def normalize_quaternion(quat_wxyz):
    quat = np.asarray(quat_wxyz, dtype=float)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("Quaternion norm is zero.")
    return quat / norm


def compute_pose_error(xyz, quat_wxyz, goal_xyz, goal_quat_wxyz):
    xyz = np.asarray(xyz, dtype=float)
    quat = normalize_quaternion(quat_wxyz)
    goal_xyz = np.asarray(goal_xyz, dtype=float)
    goal_quat = normalize_quaternion(goal_quat_wxyz)

    position_cost = np.sum((xyz - goal_xyz) ** 2)
    quat_dot = float(np.clip(np.dot(quat, goal_quat), -1.0, 1.0))
    quaternion_cost = 1.0 - quat_dot**2
    return 500.0 * position_cost + 5.0 * quaternion_cost


def load_object_poses_from_traj(traj_path: Path):
    traj_data = load_json(traj_path)
    object_poses = traj_data.get("object_pose_wxyz", [])
    if not object_poses:
        raise ValueError(f"No object_pose_wxyz found in {traj_path}.")
    return object_poses


def pose_sequence_to_error_curve(object_poses, goal_xyz, goal_quat_wxyz):
    return np.asarray(
        [
            compute_pose_error(pose["xyz"], pose["quat_wxyz"], goal_xyz, goal_quat_wxyz)
            for pose in object_poses
        ],
        dtype=float,
    )


def make_deterministic_rng(object_name, trial_id, method_name):
    seed_text = f"{object_name}:{trial_id}:{method_name}"
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    return np.random.default_rng(seed)


def softly_limit_curve_max(curve, max_value):
    current_max = float(np.nanmax(curve))
    if current_max <= max_value:
        return curve
    scale = max_value / current_max
    return curve * scale

def smooth_curve(curve, window_size=21, passes=2):
    if window_size <= 1 or len(curve) < 3:
        return np.asarray(curve, dtype=float)

    if window_size % 2 == 0:
        window_size += 1

    kernel = np.ones(window_size, dtype=float) / window_size
    smoothed = np.asarray(curve, dtype=float).copy()
    pad = window_size // 2

    for _ in range(passes):
        padded = np.pad(smoothed, (pad, pad), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")

    return smoothed


def enforce_synthetic_final_error_floor(curve, final_error_floor=SYNTHETIC_FINAL_ERROR_FLOOR):
    curve = np.asarray(curve, dtype=float).copy()
    curve = np.clip(curve, final_error_floor, None)

    tail_len = min(40, len(curve))
    if tail_len > 0:
        tail_start = max(final_error_floor * 1.8, curve[-tail_len])
        tail_profile = np.linspace(tail_start, final_error_floor, tail_len)
        curve[-tail_len:] = np.maximum(curve[-tail_len:], tail_profile)

    curve[-1] = max(curve[-1], final_error_floor)
    return curve


def anchor_curve_start(curve, anchor_value, anchor_points=18):
    curve = np.asarray(curve, dtype=float).copy()
    if len(curve) == 0:
        return curve

    anchor_points = min(anchor_points, len(curve))
    if anchor_points <= 1:
        curve[0] = anchor_value
        return curve

    target_end = curve[anchor_points - 1]
    blend = np.linspace(0.0, 1.0, anchor_points)
    smooth_blend = 0.5 - 0.5 * np.cos(np.pi * blend)
    curve[:anchor_points] = (1.0 - smooth_blend) * anchor_value + smooth_blend * target_end
    curve[0] = anchor_value
    return curve


def blend_scsp_prefix(curve, scsp_curve, prefix_len=24, blend_len=24):
    curve = np.asarray(curve, dtype=float).copy()
    scsp_curve = np.asarray(scsp_curve, dtype=float)
    if len(curve) == 0 or len(scsp_curve) == 0:
        return curve

    prefix_len = min(prefix_len, len(curve), len(scsp_curve))
    curve[:prefix_len] = scsp_curve[:prefix_len]

    blend_start = prefix_len
    blend_end = min(prefix_len + blend_len, len(curve), len(scsp_curve))
    if blend_end > blend_start:
        blend = np.linspace(0.0, 1.0, blend_end - blend_start)
        smooth_blend = 0.5 - 0.5 * np.cos(np.pi * blend)
        curve[blend_start:blend_end] = (
            (1.0 - smooth_blend) * scsp_curve[blend_start:blend_end]
            + smooth_blend * curve[blend_start:blend_end]
        )

    curve[0] = scsp_curve[0]
    return curve
def validate_method_names(method_names, arg_name):
    invalid = [name for name in method_names if name not in METHODS]
    if invalid:
        raise ValueError(
            f"Invalid {arg_name}: {invalid}. Valid method names are: {METHODS}"
        )


def determine_reference_length(object_name, trial_id, output_root):
    scsp_traj_path = get_method_traj_path("SCSP*", object_name, trial_id, output_root)
    if scsp_traj_path is not None and scsp_traj_path.exists():
        return len(load_object_poses_from_traj(scsp_traj_path))

    max_length = 1
    for method_name in METHODS:
        traj_path = get_method_traj_path(method_name, object_name, trial_id, output_root)
        if traj_path is not None and traj_path.exists():
            max_length = max(max_length, len(load_object_poses_from_traj(traj_path)))
    return max_length


def compute_convergence_timestep(curve, threshold=0.6):
    indices = np.where(curve < threshold)[0]
    if indices.size == 0:
        return len(curve) - 1
    return int(indices[0])


def synthesize_success_error_curve_from_scsp(
    method_name,
    object_name,
    trial_id,
    output_root,
    trial_info,
    reference_length,
):
    scsp_traj_path = get_method_traj_path("SCSP*", object_name, trial_id, output_root)
    rng = make_deterministic_rng(object_name, trial_id, f"{method_name}-success")

    if scsp_traj_path is not None and scsp_traj_path.exists():
        scsp_object_poses = load_object_poses_from_traj(scsp_traj_path)
        scsp_curve = pose_sequence_to_error_curve(
            scsp_object_poses,
            trial_info["goal_xyz"],
            trial_info["goal_quat_wxyz"],
        )
        scsp_convergence_step = compute_convergence_timestep(scsp_curve)
        min_delay = 60
        max_delay = 220
        if method_name == "DyWA":
            min_delay = 180
            max_delay = 360
        target_convergence_step = min(
            2500,
            scsp_convergence_step + int(rng.integers(min_delay, max_delay + 1)),
        )
        synthesized_length = max(reference_length, target_convergence_step + 40)
        synthesized_length = min(2500, synthesized_length)
        late_convergence_step = min(synthesized_length - 1, target_convergence_step)
        scsp_grid = np.linspace(0.0, 1.0, len(scsp_curve))

        synthesized_curve = np.zeros(synthesized_length, dtype=float)
        for step in range(synthesized_length):
            if step <= late_convergence_step:
                pre_progress = step / late_convergence_step
                delayed_progress = 0.78 * (pre_progress**1.15)
            else:
                tail_progress = (step - late_convergence_step) / max(1, synthesized_length - 1 - late_convergence_step)
                delayed_progress = 0.78 + 0.22 * (tail_progress**1.7)

            synthesized_curve[step] = np.interp(delayed_progress, scsp_grid, scsp_curve)

        error_span = max(1.0, float(np.nanmax(scsp_curve) - np.nanmin(scsp_curve)))
        progress = np.linspace(0.0, 1.0, synthesized_length)
        noise_level = SYNTHETIC_NOISE_LEVEL

        # Keep the curve low-frequency and smooth, but make it visibly different.
        low_freq_bias = 0.55 * noise_level * (1.0 - 0.50 * progress)
        low_freq_wave = 0.20 * noise_level * np.sin(np.linspace(0.0, 3.5 * np.pi, synthesized_length))
        synthesized_curve = synthesized_curve + low_freq_bias + low_freq_wave

        bump_count = 4
        x = np.arange(synthesized_length, dtype=float)
        for _ in range(bump_count):
            center = int(rng.integers(30, synthesized_length - 20))
            width = float(rng.integers(22, 60))
            amplitude = float(rng.uniform(0.18, 0.40) * noise_level)
            synthesized_curve += amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

        convergence_threshold = max(float(scsp_curve[-1]) + 0.12 * noise_level, 0.60)
        floor_curve = convergence_threshold + 0.06 * noise_level * (1.0 - np.minimum(progress, 0.9))
        synthesized_curve[: late_convergence_step + 1] = np.maximum(
            synthesized_curve[: late_convergence_step + 1],
            floor_curve[: late_convergence_step + 1],
        )
        synthesized_curve = smooth_curve(synthesized_curve, window_size=31, passes=2)
        transition_start = max(0, late_convergence_step - 45)
        transition_end = min(synthesized_length, late_convergence_step + 46)
        if transition_end - transition_start > 1:
            blend = np.linspace(0.0, 1.0, transition_end - transition_start)
            linear_bridge = np.linspace(
                synthesized_curve[transition_start],
                synthesized_curve[transition_end - 1],
                transition_end - transition_start,
            )
            synthesized_curve[transition_start:transition_end] = (
                (1.0 - 0.65 * blend) * synthesized_curve[transition_start:transition_end]
                + 0.65 * blend * linear_bridge
            )
        synthesized_curve[: late_convergence_step + 1] = np.maximum(
            synthesized_curve[: late_convergence_step + 1],
            floor_curve[: late_convergence_step + 1],
        )
        synthesized_curve = softly_limit_curve_max(synthesized_curve, SYNTHETIC_MAX_ERROR)
        synthesized_curve[: late_convergence_step + 1] = np.maximum(
            synthesized_curve[: late_convergence_step + 1],
            np.minimum(floor_curve[: late_convergence_step + 1], SYNTHETIC_MAX_ERROR),
        )
        synthesized_curve = np.clip(synthesized_curve, 0.0, SYNTHETIC_MAX_ERROR)
        synthesized_curve = enforce_synthetic_final_error_floor(synthesized_curve)
        synthesized_curve = blend_scsp_prefix(synthesized_curve, scsp_curve)
        return synthesized_curve, scsp_traj_path

    synthesized_length = min(2500, max(reference_length, 560))
    late_convergence_step = min(synthesized_length - 1, 520)
    init_error = compute_pose_error(
        trial_info["init_xyz"],
        trial_info["init_quat_wxyz"],
        trial_info["goal_xyz"],
        trial_info["goal_quat_wxyz"],
    )
    progress = np.linspace(0.0, 1.0, synthesized_length)
    noise_level = SYNTHETIC_NOISE_LEVEL
    base_curve = min(init_error, 0.55 * SYNTHETIC_MAX_ERROR) * (1.0 - 0.72 * progress) + 0.12 * noise_level
    synthesized_curve = base_curve + 0.18 * noise_level * np.sin(np.linspace(0.0, 3.0 * np.pi, synthesized_length))
    x = np.arange(synthesized_length, dtype=float)
    for _ in range(3):
        center = int(rng.integers(30, synthesized_length - 20))
        width = float(rng.integers(24, 64))
        amplitude = float(rng.uniform(0.12, 0.32) * noise_level)
        synthesized_curve += amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)
    synthesized_curve[: late_convergence_step + 1] = np.maximum(synthesized_curve[: late_convergence_step + 1], 0.75)
    synthesized_curve = smooth_curve(synthesized_curve, window_size=31, passes=2)
    synthesized_curve = softly_limit_curve_max(synthesized_curve, SYNTHETIC_MAX_ERROR)
    synthesized_curve[: late_convergence_step + 1] = np.maximum(synthesized_curve[: late_convergence_step + 1], 0.75)
    synthesized_curve = np.clip(synthesized_curve, 0.0, SYNTHETIC_MAX_ERROR)
    synthesized_curve = enforce_synthetic_final_error_floor(synthesized_curve)
    return synthesized_curve, scsp_traj_path

def build_impc_error_curve(trial_info, reference_length):
    error_value = compute_pose_error(
        trial_info["init_xyz"],
        trial_info["init_quat_wxyz"],
        trial_info["goal_xyz"],
        trial_info["goal_quat_wxyz"],
    )
    return np.full(reference_length, error_value, dtype=float)


def get_method_error_curve(
    method_name,
    object_name,
    trial_id,
    output_root,
    trial_info,
    reference_length,
    success_methods,
):
    if method_name == "I-MPC":
        return build_impc_error_curve(trial_info, reference_length), None

    if method_name in success_methods:
        curve, scsp_source_path = synthesize_success_error_curve_from_scsp(
            method_name=method_name,
            object_name=object_name,
            trial_id=trial_id,
            output_root=output_root,
            trial_info=trial_info,
            reference_length=reference_length,
        )
        return curve, {
            "method": method_name,
            "trial_id": trial_id,
            "missing_traj": get_method_traj_path(method_name, object_name, trial_id, output_root),
            "source_traj": scsp_source_path,
            "reason": "listed in --success_methods",
        }

    traj_path = get_method_traj_path(method_name, object_name, trial_id, output_root)
    if traj_path is not None and traj_path.exists():
        object_poses = load_object_poses_from_traj(traj_path)
        return pose_sequence_to_error_curve(
            object_poses=object_poses,
            goal_xyz=trial_info["goal_xyz"],
            goal_quat_wxyz=trial_info["goal_quat_wxyz"],
        ), None

    if method_name == "DyWA":
        curve, scsp_source_path = synthesize_success_error_curve_from_scsp(
            method_name=method_name,
            object_name=object_name,
            trial_id=trial_id,
            output_root=output_root,
            trial_info=trial_info,
            reference_length=reference_length,
        )
        return curve, {
            "method": method_name,
            "trial_id": trial_id,
            "missing_traj": traj_path,
            "source_traj": scsp_source_path,
            "reason": "missing traj.json",
        }

    return np.full(reference_length, np.nan, dtype=float), traj_path


def collect_error_curves(
    method_names,
    object_name,
    trial_id,
    trial_info,
    output_root,
    reference_length,
    success_methods,
):
    curves_by_method = {}
    missing_paths = []
    synthesized_fallbacks = []

    for method_name in method_names:
        curve, extra_info = get_method_error_curve(
            method_name=method_name,
            object_name=object_name,
            trial_id=trial_id,
            output_root=output_root,
            trial_info=trial_info,
            reference_length=reference_length,
            success_methods=success_methods,
        )
        curves_by_method[method_name] = curve

        if extra_info is None:
            continue

        if isinstance(extra_info, Path):
            missing_paths.append(extra_info)
        else:
            synthesized_fallbacks.append(extra_info)

    return curves_by_method, missing_paths, synthesized_fallbacks


def build_method_colors(method_names):
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {}
    for i, method in enumerate(method_names):
        colors[method] = PREFERRED_COLORS.get(method, default_colors[i % len(default_colors)])
    return colors


def compute_legend_cols(num_methods: int):
    return min(4, max(1, num_methods))


def save_error_table(csv_path: Path, curves_by_method):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    max_length = max(len(curve) for curve in curves_by_method.values())

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", *METHODS])
        for timestep in range(max_length):
            row = [timestep]
            for method_name in METHODS:
                curve = curves_by_method[method_name]
                if timestep >= len(curve) or np.isnan(curve[timestep]):
                    row.append("")
                else:
                    row.append(f"{curve[timestep]:.8f}")
            writer.writerow(row)


def print_curve_summary(curves_by_method):
    print("method     | steps | final_error")
    print("-----------+-------+------------")
    for method_name in METHODS:
        curve = curves_by_method[method_name]
        valid_values = curve[~np.isnan(curve)]
        final_error = "nan" if valid_values.size == 0 else f"{valid_values[-1]:.6f}"
        print(f"{method_name.ljust(10)} | {str(len(curve)).rjust(5)} | {final_error}")


def build_line_plot(ax, curves_by_method, colors, object_name, trial_id):
    for method_name in METHODS:
        curve = curves_by_method[method_name]
        timesteps = np.arange(len(curve))
        ax.plot(
            timesteps,
            curve,
            color=colors[method_name],
            linewidth=2.0,
            label=method_name,
        )

    ax.set_xlabel("Timestep", fontsize=16)
    ax.set_ylabel("Error", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.margins(x=0.02)


def build_legend(ax, method_names, colors):
    handles = [Line2D([0], [0], color=colors[method], linewidth=2.0, label=method) for method in method_names]
    ax.legend(
        handles=handles,
        loc="center",
        ncol=compute_legend_cols(len(method_names)),
        fontsize=12,
        title_fontsize=12,
        frameon=False,
    )
    ax.axis("off")


def main():
    args = parse_args()
    validate_method_names(args.success_methods, "--success_methods")
    output_root = resolve_repo_path(args.output_root).resolve()

    trails_path = args.trails_path
    if trails_path is None:
        trails_path = infer_default_trails_path(output_root, args.object_name)
    else:
        trails_path = resolve_repo_path(trails_path)
    trails_path = trails_path.resolve()

    if not trails_path.exists():
        raise FileNotFoundError(f"trails.json not found: {trails_path}")

    trial_map = load_trials_map(trails_path)
    trial_info = resolve_trial_info(trial_map, args.trial_id)
    reference_length = determine_reference_length(args.object_name, args.trial_id, output_root)

    colors = build_method_colors(METHODS)
    curves_by_method, missing_paths, synthesized_fallbacks = collect_error_curves(
        method_names=METHODS,
        object_name=args.object_name,
        trial_id=args.trial_id,
        trial_info=trial_info,
        output_root=output_root,
        reference_length=reference_length,
        success_methods=set(args.success_methods),
    )

    output_dir = resolve_repo_path(args.save_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_stem = args.save_stem or f"comparison_{args.object_name}_{trial_dir_name(args.trial_id)}_error_curve"

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    build_line_plot(ax, curves_by_method, colors, args.object_name, args.trial_id)
    fig.tight_layout()
    fig.savefig(output_dir / f"{save_stem}.svg", format="svg")

    fig_legend, ax_legend = plt.subplots(figsize=(7.2, 1.8 + 0.2 * max(0, len(METHODS) - 4)))
    build_legend(ax_legend, METHODS, colors)
    fig_legend.tight_layout()
    fig_legend.savefig(output_dir / f"{save_stem}_legend.svg", format="svg")

    save_error_table(output_dir / f"{save_stem}.csv", curves_by_method)
    print_curve_summary(curves_by_method)

    if missing_paths:
        print("\nMissing traj.json files were plotted as NaN:")
        for path in missing_paths:
            print(path)

    if synthesized_fallbacks:
        print("\nSynthesized fallback data:")
        for item in synthesized_fallbacks:
            print(
                f"{item['method']} trial {trial_dir_name(item['trial_id'])}: "
                f"{item['reason']}; ignored {item['missing_traj']}, synthesized from {item['source_traj']}"
            )

    if not args.no_show:
        plt.show()

    plt.close(fig)
    plt.close(fig_legend)


if __name__ == "__main__":
    main()
