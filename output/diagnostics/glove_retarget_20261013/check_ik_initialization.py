"""Run in ego_fast_retarget's pi0 environment. Diagnostic only; writes no data.

python check_ik_initialization.py --episode EPISODE --models hand_models/cqq_v2
Synthetic FK samples test initialization sensitivity, not ground-truth accuracy.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from wuji_sdk import (WujiGlove, FrameHeader, EmfPoseArray, EmfPose, Pose,
                      Quaternion, set_log_level)


def fk_seed(path, bend=0.0):
    transforms = {"wrist": np.eye(4)}
    joints = list(ET.parse(path).getroot().findall("joint"))
    while joints:
        progressed = False
        for joint in joints[:]:
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            matrix = np.eye(4)
            origin = joint.find("origin")
            if origin is not None:
                matrix[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                matrix[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
            name = joint.get("name")
            if joint.get("type") == "revolute" and not name.startswith("thumb"):
                if "flex" in name or name.endswith(("_pip", "_dip")):
                    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                    matrix[:3, :3] @= Rotation.from_rotvec(axis * bend).as_matrix()
            transforms[joint.find("child").get("link")] = transforms[parent] @ matrix
            joints.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("Disconnected or cyclic URDF")
    inverse_tx = np.linalg.inv(transforms["emf_tx_base"])
    coils = [inverse_tx @ transforms[f + "_rx_coil"]
             for f in ("thumb", "index", "middle", "ring", "pinky")]
    return (np.array([t[:3, 3] for t in coils]),
            Rotation.from_matrix(np.array([t[:3, :3] for t in coils])).as_quat())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    set_log_level("error")
    with h5py.File(args.episode / "wuji_glove_offline.h5", "r", locking=False) as h:
        metadata = json.loads(h.attrs["metadata_json"])
        for side in ("left", "right"):
            group = h[side]
            positions = group["emf_poses/position"][:]
            quaternions = group["emf_poses/orientation_xyzw"][:]
            confidence = group["emf_poses/confidence"][:]
            timestamps = group["source_timestamp_ns"][:] // 1000
            model = (args.models / f"{side}_hand.urdf").resolve()
            for mode, bend in (("cold", None), ("neutral", 0.0), ("bent_seed", 0.3)):
                pipeline = WujiGlove.offline_pipeline(
                    sn=metadata["devices"][side]["sn"], hand_side=side,
                    urdf_path=str(model))
                resource = pipeline.hand_skeleton()
                def frame(p, q, c, seq, stamp):
                    return EmfPoseArray(
                        header=FrameHeader(seq=seq, timestamp_us=int(stamp),
                                           frame_id=side[0] + "_hand_emf_tx"),
                        poses=[EmfPose(pose=Pose(position=p[j].tolist(),
                                   orientation=Quaternion(*q[j].tolist())),
                                   confidence=float(c[j])) for j in range(5)])
                if bend is not None:
                    p, q = fk_seed(model, bend)
                    for i in range(3):
                        resource.compute(frame(p, q, np.ones(5), i,
                                               timestamps[0] - (3-i) * 16666))
                distances = []
                for i, (p, q, c) in enumerate(zip(positions, quaternions, confidence)):
                    skeleton = resource.compute(frame(p, q, c, i+3, timestamps[i]))
                    joints = {j.name: np.array(j.pose.position) for j in skeleton.joints}
                    distances.append(np.linalg.norm(joints["thumb_tip"] -
                                                    joints["index_finger_tip"]) * 1000)
                print(side, mode, "pinch_min_median_max_mm=",
                      np.quantile(distances, [0, .5, 1]).round(3).tolist(), flush=True)


if __name__ == "__main__":
    main()
