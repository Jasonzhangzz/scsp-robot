"""Raw local EMF is the only input to offline finger retargeting.

Each side owns a pipeline and session for the native episode stream.
No hardware is connected and no additional finger optimizer runs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

SDK_VERSION = "2026.8.31"
OFFLINE_H5_NAME = "wuji_glove_offline.h5"


def _episode_roots(episode) -> list[Path]:
    """Directories that may contain recorded EMF, without following JSON-cache hops."""

    source = Path(episode).expanduser()
    roots: list[Path] = []

    def add(item: Path) -> None:
        item = Path(item)
        if item not in roots:
            roots.append(item)

    if source.is_dir():
        add(source)
    add(source.parent)
    absolute = Path(os.path.abspath(source))
    if absolute.is_dir():
        add(absolute)
    add(absolute.parent)
    return roots


def resolve_offline_h5(episode) -> Path:
    """Locate ``wuji_glove_offline.h5`` next to the episode, not a JSON cache."""

    tried: list[Path] = []
    for root in _episode_roots(episode):
        path = root / OFFLINE_H5_NAME
        tried.append(path)
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Raw EMF is required: " + ", ".join(str(path) for path in tried)
    )


def use_sdk_for_hand1(generation: str) -> bool:
    """Compatibility name: both supported hand generations use the SDK only."""
    if generation not in {"hand1", "hand2"}:
        raise ValueError(f"Unsupported hand generation: {generation}")
    return True


def sdk_hand_model_name(generation: str) -> str:
    """Official wuji-sdk retarget model for one hand generation."""

    use_sdk_for_hand1(generation)
    return "WujiHand2" if generation == "hand2" else "WujiHand"



def preserve_sdk_q(replay) -> bool:
    return bool(np.asarray(replay.get("hand_sdk_preserve_q", False)).item())


def sdk_metadata(generation="hand1") -> dict[str, np.ndarray]:
    use_sdk_for_hand1(generation)
    return {
        "hand_retarget_backend": np.asarray("wuji_sdk_offline_pipeline"),
        "hand_sdk_version": np.asarray(SDK_VERSION),
        "hand_sdk_model": np.asarray(sdk_hand_model_name(generation)),
        "hand_sdk_preserve_q": np.asarray(True),
        "hand_sdk_step_policy": np.asarray("one_step_per_native_emf_frame"),
        "hand_sdk_controller_filter_applied": np.asarray(False),
    }



def retarget_keypoints(*args, **kwargs):
    """Removed skeleton-input API: use compute_episode with recorded local EMF."""
    raise ValueError("Skeleton-input retargeting was removed; provide an episode with wuji_glove_offline.h5")



def use_emf_input(episode) -> bool:
    """Require original local EMF; never fall back to recorded skeletons."""
    resolve_offline_h5(episode)
    return True



def emf_commands_for_frames(episode, frame_indices, model_config=None, generation="hand1"):
    from wuji_retarget.emf_offline import compute_episode
    import json
    _, camera, info = compute_episode(episode, model_config, generation=generation)
    lookup = {int(value): i for i, value in enumerate(camera["frame_indices"])}
    kept = [int(i) for i in frame_indices if int(i) in lookup]
    missing = [int(i) for i in frame_indices if int(i) not in lookup]
    if not kept:
        raise ValueError(
            f"No exact EMF match for any requested source frames {missing[:10]}"
        )
    rows = np.asarray([lookup[int(i)] for i in kept], dtype=np.int64)
    output = {
        "hand_sdk_input": np.asarray("local_emf_offline_pipeline"),
        "hand_sdk_step_policy": camera["hand_sdk_step_policy"],
        "hand_sdk_emf_metadata_json": np.asarray(json.dumps(info)),
        "frame_indices": np.asarray(kept, dtype=np.int64),
        "dropped_source_frame_indices": np.asarray(missing, dtype=np.int64),
    }
    for side in ("left", "right"):
        output[f"{side}_hand_q"] = camera[f"{side}_hand_q"][rows]
        output[f"{side}_sdk_emf_source_timestamp_ns"] = camera[f"{side}_source_timestamp_ns"][rows]
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Raw EMF -> Wuji SDK offline pipeline -> joint angles")
    parser.add_argument("--episode", "--test_data", "--tast_data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-frames", "--max_frames", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--input", choices=["emf"], default="emf")
    parser.add_argument("--generation", choices=["hand1", "hand2"], default="hand1")
    parser.add_argument("--model-config")
    args = parser.parse_args(argv)
    if args.stride < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("stride and max-frames must be positive")
    from wuji_retarget.emf_offline import compute_episode
    _, output, _ = compute_episode(args.episode, args.model_config, generation=args.generation)
    count = len(output["frame_indices"])
    rows = np.flatnonzero(output["frame_indices"] % args.stride == 0)[:args.max_frames]
    for key, value in list(output.items()):
        if value.ndim and len(value) == count and not key.endswith("names"):
            output[key] = value[rows]
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as f:
        np.savez_compressed(f, **output)
    print(f"Saved {len(rows)} EMF-derived SDK frames to {destination}")



if __name__ == "__main__":
    main()
