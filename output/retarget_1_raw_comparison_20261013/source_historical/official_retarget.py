"""Compatibility archive adapter for raw EMF SDK finger retargeting.

Recorded world skeletons supply wrist/visualization metadata only; finger
commands come exclusively from the pinned SDK offline pipeline. Legacy YAML
utility functions remain for geometry tooling, outside the retarget path."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Mapping, Sequence
import os

import numpy as np
import yaml

from wuji_retarget.hand_retarget import (
    DEFAULT_IK_ANCHOR,
    FINGER_SKELETON_CHAINS,
    ScaledTargets,
    SkeletonSequence,
    _matrices_to_quat_xyzw,
    _matrices_to_quat_xyzw_allow_nan,
    hand_to_robot_rotations_for_urdf,
    load_world_skeletons,
    normalize_ik_anchor,
)
from wuji_retarget.wuji_hand_names import (
    HAND1,
    HAND2,
    actuated_joint_names,
    detect_generation_from_names,
    detect_generation_from_urdf,
    resolve_hand_urdfs_for_composite,
)
from wuji2_retarget.skeleton_mapping import (
    build_hand2_joint_mapping,
    wrist_from_four_finger_mcp,
)

_REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "assets").is_dir() and (parent / "thirdparty").is_dir()
)
_OFFICIAL_REPO = _REPO_ROOT / "thirdparty" / "wuji-retargeting"
GEORT_WUJI_RETARGETING_20260526 = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/geort/wuji-retargeting-2026.05.26"
)
_GEORT_20260526_PACKAGE_NAME = "wuji_retargeting_20260526"
_GEORT_20260526_PACKAGE = None
_GEORT_20260526_IMPORT_LOCK = None
_OFFICIAL_YAML = {
    HAND1: {
        "left": (
            _OFFICIAL_REPO / "example/config/adaptive_analytical_wuji_glove_left.yaml"
        ),
        "right": (
            _OFFICIAL_REPO / "example/config/adaptive_analytical_wuji_glove_right.yaml"
        ),
    },
    HAND2: {
        "left": (
            _OFFICIAL_REPO
            / "example/config/adaptive_analytical_wuji_glove_wuji_hand_2_left.yaml"
        ),
        "right": (
            _OFFICIAL_REPO
            / "example/config/adaptive_analytical_wuji_glove_wuji_hand_2_right.yaml"
        ),
    },
}
DEFAULT_HAND_URDFS = {
    "left": _REPO_ROOT / "assets/wuji_hand2/urdf/left.urdf",
    "right": _REPO_ROOT / "assets/wuji_hand2/urdf/right.urdf",
}


def official_yaml_path(
    side: str,
    generation: str | None = None,
    *,
    urdf_path: str | Path | None = None,
) -> Path:
    """Official AdaptiveOptimizerAnalytical YAML for one glove → hand pair.

    Hand 1 (``ex001_wuji``) uses ``adaptive_analytical_wuji_glove_{side}.yaml``.
    Hand 2 (``ex001_wuji2``) uses the ``*_wuji_hand_2_*`` YAML, including its
    ``link_naming`` and ``mediapipe_rotation``.
    """

    if generation is None:
        generation = (
            detect_generation_from_urdf(urdf_path) if urdf_path is not None else HAND2
        )
    try:
        path = _OFFICIAL_YAML[generation][side]
    except KeyError as exc:
        raise ValueError(
            f"side must be left or right and generation hand1/hand2, "
            f"got side={side!r} generation={generation!r}"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(
            f"Official wuji-retargeting config missing: {path}"
        )
    return path


def load_official_retargeter(
    side: str,
    *,
    urdf_path: str | Path | None = None,
    generation: str | None = None,
    glove_urdf: str | Path | None = None,
    sdk_user: str | None = None,
):
    """Build a ``Retargeter`` from the official glove YAML for that hand.

    ``urdf_path`` overrides the YAML model so solved q uses the same joint
    names and limits as the composite replay URDF.  Relative paths in the
    YAML are resolved against the YAML file, as upstream does.

    ``urdf_path`` swaps the Pinocchio model (FK / limits / joint names),
    matching official ``optimizer.urdf_path``.  Recorded keypoints keep
    their size.  FullHandVec keeps YAML ``segment_scaling`` only.
    CUDA and NLopt share both steps.
    """

    from calibrated.official_hand_ik import (
        ensure_official_retargeting_importable,
    )

    ensure_official_retargeting_importable()
    from wuji_retargeting import Retargeter

    if generation is None and urdf_path is not None:
        generation = detect_generation_from_urdf(urdf_path)
    yaml_path = official_yaml_path(side, generation, urdf_path=urdf_path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["__yaml_dir"] = str(yaml_path.parent)
    opt_cfg = config.setdefault("optimizer", {})
    if urdf_path is not None:
        opt_cfg["urdf_path"] = str(Path(urdf_path).resolve())
    return Retargeter.from_config(config, hand_side=side)


def _geort_20260526_import_lock():
    global _GEORT_20260526_IMPORT_LOCK
    import threading

    if _GEORT_20260526_IMPORT_LOCK is None:
        _GEORT_20260526_IMPORT_LOCK = threading.Lock()
    return _GEORT_20260526_IMPORT_LOCK


def _load_module(fullname: str, filepath: Path, *, search_locations: list[str] | None = None):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        fullname,
        filepath,
        submodule_search_locations=search_locations,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {fullname} from {filepath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
    return module


def _import_geort_20260526_package():
    """Load 2026.05.26 under ``wuji_retargeting_20260526``.

    In-process batch workers already import the vendored ``wuji_retargeting``.
    Purging that name races across threads.  A private package name keeps
    relative imports (``from ..robot`` / ``from .base``) inside this tree.
    """

    global _GEORT_20260526_PACKAGE
    import sys
    import types

    with _geort_20260526_import_lock():
        if _GEORT_20260526_PACKAGE is not None:
            return _GEORT_20260526_PACKAGE
        existing = sys.modules.get(_GEORT_20260526_PACKAGE_NAME)
        if existing is not None and getattr(existing, "Retargeter", None) is not None:
            _GEORT_20260526_PACKAGE = existing
            return existing

        from calibrated.official_hand_ik import install_nlopt_scipy_stub

        install_nlopt_scipy_stub()
        if not GEORT_WUJI_RETARGETING_20260526.is_dir():
            raise FileNotFoundError(
                f"2026.05.26 wuji-retargeting missing: {GEORT_WUJI_RETARGETING_20260526}"
            )

        name = _GEORT_20260526_PACKAGE_NAME
        pkg_dir = GEORT_WUJI_RETARGETING_20260526 / "wuji_retargeting"
        opt_dir = pkg_dir / "opt"
        pkg = types.ModuleType(name)
        pkg.__file__ = str(pkg_dir / "__init__.py")
        pkg.__path__ = [str(pkg_dir)]
        pkg.__package__ = name
        sys.modules[name] = pkg

        robot = _load_module(f"{name}.robot", pkg_dir / "robot.py")
        mediapipe = _load_module(f"{name}.mediapipe", pkg_dir / "mediapipe.py")
        pkg.robot = robot
        pkg.mediapipe = mediapipe

        opt_pkg = types.ModuleType(f"{name}.opt")
        opt_pkg.__file__ = str(opt_dir / "__init__.py")
        opt_pkg.__path__ = [str(opt_dir)]
        opt_pkg.__package__ = f"{name}.opt"
        sys.modules[f"{name}.opt"] = opt_pkg

        base = _load_module(f"{name}.opt.base", opt_dir / "base.py")
        adaptive = _load_module(
            f"{name}.opt.adaptive_analytical",
            opt_dir / "adaptive_analytical.py",
        )
        opt_pkg.base = base
        opt_pkg.adaptive_analytical = adaptive
        opt_pkg.BaseOptimizer = base.BaseOptimizer
        opt_pkg.LPFilter = base.LPFilter
        opt_pkg.AdaptiveOptimizerAnalytical = adaptive.AdaptiveOptimizerAnalytical
        pkg.opt = opt_pkg

        retarget = _load_module(f"{name}.retarget", pkg_dir / "retarget.py")
        pkg.retarget = retarget
        pkg.Retargeter = retarget.Retargeter
        pkg.BaseOptimizer = base.BaseOptimizer
        pkg.LPFilter = base.LPFilter
        pkg.apply_mediapipe_transformations = mediapipe.apply_mediapipe_transformations
        _GEORT_20260526_PACKAGE = pkg
        return pkg


def load_normal_retargeter(
    side: str,
    *,
    urdf_path: str | Path | None = None,
    generation: str | None = None,
    glove_urdf: str | Path | None = None,
    sdk_user: str | None = None,
):
    """Official 2026.05.26 AdaptiveOptimizerAnalytical (NLopt ``LD_SLSQP``).

    Hand 1 builds ``Retargeter`` from
    ``geort/wuji-retargeting-2026.05.26`` and keeps that package loaded so
    ``from_config`` cannot pick up a later ``AdaptiveOptimizerAnalytical``.
    That tree has no ``link_naming``, so Hand 2 keeps the same SLSQP through
    the later official package and its Hand 2 YAML.
    """

    del glove_urdf, sdk_user
    if generation is None and urdf_path is not None:
        generation = detect_generation_from_urdf(urdf_path)
    if generation is None:
        generation = HAND2
    if generation == HAND2:
        print(
            "[official-hand] normal SLSQP: Hand 2 uses later official "
            "AdaptiveOptimizerAnalytical (2026.05.26 has no link_naming)",
            flush=True,
        )
        return load_official_retargeter(
            side,
            urdf_path=urdf_path,
            generation=generation,
        )
    package = _import_geort_20260526_package()
    yaml_path = (
        GEORT_WUJI_RETARGETING_20260526
        / f"example/config/adaptive_analytical_wuji_glove_{side}.yaml"
    )
    if not yaml_path.is_file():
        raise FileNotFoundError(f"2026.05.26 glove YAML missing: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["__yaml_dir"] = str(yaml_path.parent)
    opt_cfg = config.setdefault("optimizer", {})
    if urdf_path is not None:
        opt_cfg["urdf_path"] = str(Path(urdf_path).resolve())
    print(
        f"[official-hand] normal SLSQP from {GEORT_WUJI_RETARGETING_20260526} "
        f"({side} {generation})",
        flush=True,
    )
    return package.Retargeter.from_config(config, hand_side=side)


def remap_official_q(
    qpos: np.ndarray,
    source_names: Sequence[str] | list[str],
    side: str,
    generation: str | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Reorder official Pinocchio q into stored hinge order."""

    if generation is None:
        generation = detect_generation_from_names(source_names)
    target_names = actuated_joint_names(side, generation)
    source = list(source_names)
    missing = [name for name in target_names if name not in source]
    if missing:
        raise KeyError(
            f"Official retarget q is missing {side} joints {missing}; "
            f"got {source}"
        )
    order = [source.index(name) for name in target_names]
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.shape[-1] != len(source):
        raise ValueError(
            f"Official q last dim {qpos.shape[-1]} != {len(source)} joint names"
        )
    return qpos[..., order], target_names


def construct_raw_glove_targets(
    positions_world: np.ndarray,
    rotations_world: np.ndarray,
    mapping,
    side: str,
    *,
    hand_to_robot_rotation: np.ndarray,
) -> ScaledTargets:
    """Archive raw glove joints. Official IK keeps recorded skeleton size."""

    from adapters import wuji_glove_utils as wg

    positions_world = np.asarray(positions_world, dtype=np.float64)
    rotations_world = np.asarray(rotations_world, dtype=np.float64)
    if positions_world.ndim != 3 or positions_world.shape[1:] != (21, 3):
        raise ValueError(
            f"positions_world must have shape [T,21,3], got {positions_world.shape}"
        )
    if rotations_world.shape != (*positions_world.shape[:2], 3, 3):
        raise ValueError(
            f"rotations_world has incompatible shape {rotations_world.shape}"
        )
    A = np.asarray(hand_to_robot_rotation, dtype=np.float64)
    if A.shape != (3, 3) or not np.allclose(A.T @ A, np.eye(3), atol=1e-6):
        raise ValueError("hand_to_robot_rotation must be a 3x3 rotation matrix")
    index = {name: i for i, name in enumerate(wg.JOINT_ORDER)}
    wrist_p = positions_world[:, index["wrist"]]
    wrist_R_glove = rotations_world[:, index["wrist"]]
    wrist_R_robot = wrist_R_glove @ A.T
    frame_count = int(positions_world.shape[0])
    count = len(mapping)
    local_positions = np.empty((frame_count, count, 3), dtype=np.float64)
    local_rotations = np.empty((frame_count, count, 3, 3), dtype=np.float64)
    human_lengths = np.empty((frame_count, count), dtype=np.float64)
    target_world_positions = np.empty_like(local_positions)
    raw_local_by_name: dict[str, np.ndarray] = {"wrist": np.zeros((frame_count, 3))}
    for column, item in enumerate(mapping):
        child_i = index[item.skeleton_joint]
        child_delta_world = positions_world[:, child_i] - wrist_p
        child_glove_local = np.einsum("tji,tj->ti", wrist_R_glove, child_delta_world)
        child_robot_local = np.einsum("ij,tj->ti", A, child_glove_local)
        raw_local_by_name[item.skeleton_joint] = child_robot_local
        local_positions[:, column] = child_robot_local
        parent_raw = raw_local_by_name[item.skeleton_parent]
        segment = child_robot_local - parent_raw
        human_lengths[:, column] = np.linalg.norm(segment, axis=-1)
        joint_R_glove_local = np.einsum(
            "tji,tjk->tik", wrist_R_glove, rotations_world[:, child_i]
        )
        local_rotations[:, column] = np.einsum("ij,tjk->tik", A, joint_R_glove_local)
        target_world_positions[:, column] = positions_world[:, child_i]
    target_world_rotations = np.einsum(
        "tij,tnjk->tnik", wrist_R_robot, local_rotations
    )
    return ScaledTargets(
        local_positions=local_positions,
        local_rotations=local_rotations,
        human_lengths=human_lengths,
        target_local_positions=local_positions,
        target_local_rotations=local_rotations,
        target_world_positions=target_world_positions,
        target_world_rotations=target_world_rotations,
        wrist_world_positions=wrist_p,
        wrist_world_rotations=wrist_R_robot,
    )


def _contact_mask(
    requested: np.ndarray | None,
    frame_count: int,
    frame_indices: np.ndarray,
    side: str,
) -> np.ndarray:
    if requested is None:
        return np.zeros((frame_count, 5), dtype=bool)
    mask = np.asarray(requested, dtype=bool)
    if (
        mask.ndim == 2
        and mask.shape[1] == 5
        and mask.shape[0] != frame_count
        and len(mask) > int(frame_indices[-1])
    ):
        mask = mask[frame_indices]
    if mask.shape != (frame_count, 5):
        raise ValueError(
            f"{side} contact tip mask must have shape {(frame_count, 5)}"
        )
    return mask


def retarget_episode_official(
    episode: str | Path,
    *,
    composite_urdf: str | Path,
    hand_urdfs: Mapping[str, str | Path] = DEFAULT_HAND_URDFS,
    glove_urdfs: Mapping[str, str | Path] | None = None,
    glove_sdk_user: str | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    min_confidence: float = 0.0,
    contact_tip_masks: Mapping[str, np.ndarray] | None = None,
    fallback_tip_masks: Mapping[str, np.ndarray] | None = None,
    contact_tip_position_cost: float = 2000.0,
    contact_tip_orientation_cost: float = 2.0,
    anchor_skeleton_joint: str = DEFAULT_IK_ANCHOR,
    apply_filter: bool = True,
    verbose: bool = False,
    sequence: SkeletonSequence | None = None,
    model_config: str | Path | None = None,
    generation: str | None = None,
    **_ignored,
) -> dict[str, np.ndarray]:
    """Retarget raw local EMF through the SDK; retain recorded world wrist targets."""

    from wuji_retarget.sdk_retarget import use_emf_input
    use_emf_input(episode)
    if generation is None:
        generation = detect_generation_from_urdf(composite_urdf)
    generation = str(generation).lower()
    if generation not in {HAND1, HAND2}:
        raise ValueError(f"generation must be hand1 or hand2, got {generation!r}")
    anchor_skeleton_joint = normalize_ik_anchor(
        anchor_skeleton_joint, field="anchor_skeleton_joint"
    )
    hand_urdfs = resolve_hand_urdfs_for_composite(composite_urdf, hand_urdfs)
    if model_config is None:
        model_config = _ignored.get("model_config")

    if sequence is None:
        sequence = load_world_skeletons(
            episode,
            stride=stride,
            max_frames=max_frames,
            min_confidence=min_confidence,
        )
    from wuji_retarget.sdk_retarget import sdk_metadata, emf_commands_for_frames
    if verbose:
        from wuji_retarget.sdk_retarget import sdk_hand_model_name

        print(
            f"[official-retarget] EMF -> {sdk_hand_model_name(generation)} "
            f"({generation})",
            flush=True,
        )
    sdk_emf_commands = emf_commands_for_frames(episode, sequence.frame_indices,
        model_config=model_config, generation=generation)
    kept = np.asarray(sdk_emf_commands["frame_indices"], dtype=np.int64)
    if not np.array_equal(kept, np.asarray(sequence.frame_indices, dtype=np.int64)):
        locator = {
            int(index): row
            for row, index in enumerate(np.asarray(sequence.frame_indices))
        }
        rows = np.asarray([locator[int(index)] for index in kept], dtype=np.int64)
        dropped = sdk_emf_commands.get("dropped_source_frame_indices", [])
        if verbose:
            preview = np.asarray(dropped).reshape(-1)[:20].tolist()
            print(
                f"[official-retarget] dropping {len(dropped)} camera frames "
                f"without exact EMF match: {preview}",
                flush=True,
            )
        sequence = sequence.take(rows)
        if contact_tip_masks is not None:
            contact_tip_masks = {
                side: np.asarray(contact_tip_masks[side])[rows]
                for side in contact_tip_masks
            }
        if fallback_tip_masks is not None:
            fallback_tip_masks = {
                side: np.asarray(fallback_tip_masks[side])[rows]
                for side in fallback_tip_masks
            }
    frame_count = len(sequence.frame_indices)
    output = sdk_metadata(generation)
    output.update(frame_indices=sequence.frame_indices, timestamps=sequence.timestamps)
    output.update(sdk_emf_commands)
    output["frame_indices"] = sequence.frame_indices
    output["timestamps"] = sequence.timestamps
    from adapters import wuji_glove_utils as wg

    hand_to_robot_rotations = hand_to_robot_rotations_for_urdf(
        sequence.hand_to_robot_rotations
        or {side: wg.get_position_transform(side) for side in ("left", "right")},
        composite_urdf,
    )

    from calibrated.official_hand_lm_fused import (
        maybe_resize_glove_skeleton,
        resolve_glove_urdf,
    )

    prepared: dict[str, dict] = {}
    for side in ("left", "right"):
        requested_contact_mask = _contact_mask(
            None if contact_tip_masks is None else contact_tip_masks[side],
            frame_count,
            sequence.frame_indices,
            side,
        )
        valid_recorded_tip_pose = (
            np.all(np.isfinite(sequence.tip_positions[side]), axis=-1)
            & np.all(np.isfinite(sequence.tip_rotations[side]), axis=(-2, -1))
        )
        side_contact_mask = requested_contact_mask & valid_recorded_tip_pose
        side_fallback_mask = (
            np.zeros_like(side_contact_mask)
            if fallback_tip_masks is None
            else np.asarray(fallback_tip_masks[side], dtype=bool)
        )
        if side_fallback_mask.shape != side_contact_mask.shape:
            raise ValueError(f"{side} fallback tip mask must have shape [T,5]")
        side_fallback_mask = side_fallback_mask & ~requested_contact_mask

        if generation == HAND2:
            mapping = build_hand2_joint_mapping(
                composite_urdf,
                side,
                default_hand_urdfs=hand_urdfs,
            )
        else:
            from wuji_retarget.hand_retarget import (
                build_joint_mapping as build_hand1_mapping,
            )

            mapping = build_hand1_mapping(composite_urdf, side)
        raw_keypoints = np.asarray(sequence.positions[side], dtype=np.float64)
        glove_override = None if glove_urdfs is None else glove_urdfs.get(side)
        glove_path = resolve_glove_urdf(
            side,
            glove_override,
            sdk_user=glove_sdk_user,
        )
        keypoints = maybe_resize_glove_skeleton(
            raw_keypoints, side, glove_override
        )
        targets = construct_raw_glove_targets(
            sequence.positions[side],
            sequence.rotations[side],
            mapping,
            side,
            hand_to_robot_rotation=hand_to_robot_rotations[side],
        )
        prepared[side] = {
            "requested_contact_mask": requested_contact_mask,
            "side_contact_mask": side_contact_mask,
            "side_fallback_mask": side_fallback_mask,
            "mapping": mapping,
            "targets": targets,
            "keypoints": keypoints,
            "source_keypoints": raw_keypoints,
            "glove_path": glove_path,
        }

    for side in ("left", "right"):
        requested_contact_mask = prepared[side]["requested_contact_mask"]
        side_contact_mask = prepared[side]["side_contact_mask"]
        side_fallback_mask = prepared[side]["side_fallback_mask"]
        targets = prepared[side]["targets"]
        mapping = prepared[side]["mapping"]
        hand_q = sdk_emf_commands[f"{side}_hand_q"]
        hand_q_names = actuated_joint_names(side, generation)
        valid_recorded_tip_pose = (np.all(np.isfinite(sequence.tip_positions[side]), axis=-1)
            & np.all(np.isfinite(sequence.tip_rotations[side]), axis=(-2, -1)))
        nan_tips = np.full((frame_count, 5), np.nan, dtype=np.float64)
        prefix = f"{side}_"
        output[prefix + "hand_q"] = hand_q
        output[prefix + "hand_q_names"] = np.asarray(hand_q_names)
        wrist_positions = np.asarray(targets.wrist_world_positions, dtype=np.float64)
        wrist_rotations = np.asarray(targets.wrist_world_rotations, dtype=np.float64)
        output[prefix + "wrist_position"] = wrist_positions
        output[prefix + "wrist_quaternion_xyzw"] = _matrices_to_quat_xyzw(
            wrist_rotations
        )
        output[prefix + "task_rms"] = np.full(frame_count, np.nan)
        output[prefix + "joint_position_rms"] = np.full(frame_count, np.nan)
        output[prefix + "ik_iterations"] = np.zeros(frame_count, dtype=np.int64)
        output[prefix + "success"] = np.ones(frame_count, dtype=bool)
        output[prefix + "contact_tip_position_error_m"] = nan_tips.copy()
        output[prefix + "contact_tip_orientation_error_rad"] = nan_tips.copy()
        output[prefix + "mink_alignment_tip_position_error_m"] = nan_tips.copy()
        output[prefix + "mink_alignment_tip_orientation_error_rad"] = nan_tips.copy()
        output[prefix + "target_wrist_position"] = wrist_positions
        output[prefix + "target_wrist_quaternion_xyzw"] = _matrices_to_quat_xyzw(
            wrist_rotations
        )
        scaled_skeleton_positions = np.concatenate(
            (
                targets.wrist_world_positions[:, None, :],
                targets.target_world_positions,
            ),
            axis=1,
        )
        output[prefix + "source_skeleton_positions_world"] = sequence.positions[side]
        output[prefix + "resized_skeleton_positions_world"] = prepared[side]["keypoints"]
        output[prefix + "source_skeleton_quaternions_xyzw"] = _matrices_to_quat_xyzw(
            sequence.rotations[side]
        )
        output[prefix + "scaled_skeleton_positions_world"] = scaled_skeleton_positions
        output[prefix + "scaled_skeleton_source_quaternions_xyzw"] = (
            _matrices_to_quat_xyzw(sequence.rotations[side])
        )
        output[prefix + "scaled_skeleton_anchor_joint"] = np.asarray(
            anchor_skeleton_joint
        )
        output[prefix + "target_joint_positions_world"] = targets.target_world_positions
        output[prefix + "target_joint_quaternions_xyzw"] = _matrices_to_quat_xyzw(
            targets.target_world_rotations
        )
        output[prefix + "raw_joint_positions_wrist"] = targets.local_positions
        output[prefix + "raw_joint_quaternions_wrist_xyzw"] = _matrices_to_quat_xyzw(
            targets.local_rotations
        )
        output[prefix + "target_joint_positions_wrist"] = targets.target_local_positions
        output[prefix + "target_joint_quaternions_wrist_xyzw"] = _matrices_to_quat_xyzw(
            targets.target_local_rotations
        )
        output[prefix + "human_link_lengths"] = targets.human_lengths
        output[prefix + "recorded_tip_positions_world"] = sequence.tip_positions[side]
        output[prefix + "recorded_tip_quaternions_xyzw"] = (
            _matrices_to_quat_xyzw_allow_nan(sequence.tip_rotations[side])
        )
        output[prefix + "detected_contact_tip_mask"] = requested_contact_mask
        output[prefix + "contact_tip_mask"] = side_contact_mask
        output[prefix + "fallback_tip_mask"] = side_fallback_mask
        output[prefix + "mink_alignment_tip_mask"] = (
            side_contact_mask | side_fallback_mask
        )
        output[prefix + "mink_alignment_tip_positions_world"] = sequence.tip_positions[
            side
        ]
        output[prefix + "mink_alignment_tip_quaternions_xyzw"] = (
            _matrices_to_quat_xyzw_allow_nan(sequence.tip_rotations[side])
        )
        output[prefix + "contact_tip_pose_valid"] = valid_recorded_tip_pose
        output[prefix + "contact_tip_names"] = np.asarray(
            [finger for finger, _ in FINGER_SKELETON_CHAINS]
        )
        output[prefix + "contact_tip_position_cost"] = np.asarray(
            contact_tip_position_cost, dtype=np.float64
        )
        output[prefix + "contact_tip_orientation_cost"] = np.asarray(
            contact_tip_orientation_cost, dtype=np.float64
        )
        robot_lengths = np.asarray(
            [item.robot_length for item in mapping], dtype=np.float64
        )
        output[prefix + "robot_link_lengths"] = robot_lengths
        output[prefix + "median_scale_factors"] = robot_lengths / np.median(
            targets.human_lengths, axis=0
        )
        output[prefix + "skeleton_joint_names"] = np.asarray(
            [item.skeleton_joint for item in mapping]
        )
        output[prefix + "robot_joint_names"] = np.asarray(
            [item.robot_joint for item in mapping]
        )
        output[prefix + "robot_frame_names"] = np.asarray(
            [item.robot_child_link for item in mapping]
        )
        output[prefix + "hand_to_robot_rotation"] = np.asarray(
            hand_to_robot_rotations[side], dtype=np.float64
        )

    output["scaled_skeleton_cache_version"] = np.asarray(2, dtype=np.int64)
    cuda_state = {}
    mcp_cuda = None
    if anchor_skeleton_joint == "mcp":
        try:
            if mcp_cuda is None:
                from calibrated.mcp_wrist_lm_fused import solve_mcp_wrist_from_keypoints

                mcp_cuda = solve_mcp_wrist_from_keypoints(
                    {side: prepared[side]["keypoints"] for side in ("left", "right")},
                    hand_urdfs,
                    hand_to_robot_rotations=hand_to_robot_rotations,
                    wrist_rotations={
                        side: np.asarray(
                            sequence.rotations[side][:, 0], dtype=np.float64
                        )
                        for side in ("left", "right")
                    },
                )
            wrist_p, wrist_R, _resid = mcp_cuda
            p_np = wrist_p.detach().cpu().numpy().astype(np.float64)
            r_np = wrist_R.detach().cpu().numpy().astype(np.float64)
            for index, side in enumerate(("left", "right")):
                prefix = f"{side}_"
                output[prefix + "wrist_position"] = p_np[:, index]
                output[prefix + "wrist_quaternion_xyzw"] = _matrices_to_quat_xyzw(
                    r_np[:, index]
                )
                output[prefix + "target_wrist_position"] = output[
                    prefix + "wrist_position"
                ]
                output[prefix + "target_wrist_quaternion_xyzw"] = output[
                    prefix + "wrist_quaternion_xyzw"
                ]
            cuda_state["wrist_p"] = wrist_p
            cuda_state["wrist_R"] = wrist_R
            output["cuda_mcp_wrist"] = np.asarray(True, dtype=bool)
        except Exception as exc:
            if verbose:
                print(
                    f"[official-retarget] MCP wrist fallback to host pose: {exc}",
                    flush=True,
                )
            for side in ("left", "right"):
                prefix = f"{side}_"
                wrist_rotations, wrist_positions = wrist_from_four_finger_mcp(
                    hand_urdfs[side],
                    side,
                    output[prefix + "hand_q"],
                    prepared[side]["keypoints"],
                    hand_to_robot_rotation=hand_to_robot_rotations[side],
                    wrist_rotations=np.asarray(
                        sequence.rotations[side][:, 0], dtype=np.float64
                    ),
                )
                output[prefix + "wrist_position"] = wrist_positions
                output[prefix + "wrist_quaternion_xyzw"] = _matrices_to_quat_xyzw(
                    wrist_rotations
                )
                output[prefix + "target_wrist_position"] = wrist_positions
                output[prefix + "target_wrist_quaternion_xyzw"] = _matrices_to_quat_xyzw(
                    wrist_rotations
                )
    if cuda_state:
        output["_cuda"] = cuda_state
    return output


__all__ = [
    "DEFAULT_HAND_URDFS",
    "construct_raw_glove_targets",
    "load_normal_retargeter",
    "load_official_retargeter",
    "official_yaml_path",
    "remap_official_q",
    "retarget_episode_official",
]
