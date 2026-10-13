"""Recorded local EMF -> SDK offline_pipeline -> Wuji Hand 1 commands.

Reads rd001.wuji_offline_h5. Native left/right streams retain independent
source timestamps. Camera outputs select exact recorded EMF payloads, never
interpolate q or re-run the session at camera rate. No hardware is connected.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET

import h5py
import ijson
import numpy as np
from wuji_retarget.sdk_retarget import SDK_VERSION, sdk_metadata
from wuji_retarget.wuji_hand_names import actuated_joint_names

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'models/wuji_v3_20260920/config.json'


def configured_urdf_path(side, path=None):
    """Resolve ``{side}_urdf`` from the pinned model JSON, or None if builtin."""

    side = str(side).lower()
    if side not in ('left', 'right'):
        raise ValueError(f'side must be left or right, got {side!r}')
    config_path = Path(path or os.environ.get('GEORT_WUJI_MODEL_CONFIG', DEFAULT_CONFIG)).resolve()
    payload = json.loads(config_path.read_text())
    value = payload.get(f'{side}_urdf')
    if not value:
        return None
    urdf = Path(value)
    urdf = urdf.resolve() if urdf.is_absolute() else (config_path.parent / urdf).resolve()
    if not urdf.is_file():
        raise FileNotFoundError(f'{side} URDF from {config_path} is not a file: {urdf}')
    return urdf


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model_config(path=None):
    path = Path(path or os.environ.get('GEORT_WUJI_MODEL_CONFIG', DEFAULT_CONFIG)).resolve()
    cfg = json.loads(path.read_text())
    if cfg['sdk_version'] != SDK_VERSION or version('wuji-sdk') != SDK_VERSION:
        raise RuntimeError(f'Requires wuji-sdk=={SDK_VERSION}')
    for side in ('left', 'right'):
        value = cfg.get(f'{side}_urdf')
        if value:
            urdf = (path.parent / value).resolve()
            root = ET.parse(urdf).getroot()
            links = {e.attrib['name'] for e in root.findall('link')}
            if 'wrist' not in links or 'thumb_tip' not in links:
                raise ValueError(f'{urdf} is not a glove human-hand URDF')
            digest = sha256(urdf)
            expected = cfg.get(f'{side}_sha256')
            if expected and digest != expected:
                raise ValueError(f'{side} URDF checksum mismatch')
            cfg[f'{side}_urdf'] = str(urdf)
            cfg[f'{side}_sha256'] = digest
        elif cfg.get(f'{side}_model_source') != 'builtin_default':
            raise ValueError(f'{side}: explicit URDF or builtin_default policy required')
    cfg['config_path'] = str(path)
    return cfg


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def fingerprint(p, q, c):
    # Payloads preserve float64 numeric values in both HDF5 and episode JSON.
    return np.concatenate((np.asarray(p).ravel(), np.asarray(q).ravel(), np.asarray(c).ravel())).astype('<f8').tobytes()


def json_emf_payload(message):
    poses = message.get('poses', [])
    if len(poses) != 5:
        raise ValueError('Expected five local EMF poses')
    def vec(v, fields):
        return [float(v[k]) for k in fields] if isinstance(v, dict) else list(v)
    p = [vec(x['pose']['position'], 'xyz') for x in poses]
    q = [vec(x['pose']['orientation'], 'xyzw') for x in poses]
    c = [float(x['confidence']) for x in poses]
    return p, q, c


def validate_stream(g, side):
    p = g['emf_poses/position'][:]
    q = g['emf_poses/orientation_xyzw'][:]
    c = g['emf_poses/confidence'][:]
    ts = g['source_timestamp_ns'][:]
    seq = g['source_seq'][:]
    action_ts = g['action_timestamp_ns'][:]
    frame_id = [_text(x) for x in g['source_frame_id'][:]]
    n = len(ts)
    if not n or p.shape != (n,5,3) or q.shape != (n,5,4) or c.shape != (n,5):
        raise ValueError(f'{side}: invalid EMF shapes')
    if any(len(x) != n for x in (seq, action_ts, frame_id)):
        raise ValueError(f'{side}: unaligned EMF metadata')
    if not all(np.isfinite(x).all() for x in (p,q,c)) or np.any(np.linalg.norm(q, axis=-1) < 1e-8):
        raise ValueError(f'{side}: nonfinite EMF or zero quaternion')
    if np.any(np.diff(ts) <= 0) or np.any(ts < 0):
        raise ValueError(f'{side}: source timestamps must be strictly increasing')
    expected = f'{side[0]}_hand_emf_tx'
    if any(x != expected for x in frame_id):
        raise ValueError(f'{side}: expected local {expected} frame, never world/related')
    return p,q,c,ts,seq,action_ts,frame_id


def solve_native(ep, config, generation="hand1"):
    from wuji_sdk import (WujiGlove, RetargetSession, HandModel, Handedness,
                          FrameHeader, EmfPose, EmfPoseArray, Pose, Quaternion, set_log_level)
    set_log_level('error')
    arrays = sdk_metadata(generation)
    arrays['hand_sdk_input'] = np.asarray('local_emf_offline_pipeline')
    arrays['hand_sdk_step_policy'] = np.asarray('one_step_per_native_emf_frame')
    arrays['hand_sdk_preserve_q'] = np.asarray(True)
    info = {'sdk_version': SDK_VERSION, 'models': config, 'sides': {}}
    lookups = {}
    with h5py.File(ep/'wuji_glove_offline.h5', 'r', locking=False) as h:
        if _text(h.attrs.get('schema', '')) != 'rd001.wuji_offline_h5':
            raise ValueError('Unsupported HDF5 schema')
        metadata = json.loads(_text(h.attrs['metadata_json']))
        info['recorded_sdk_version'] = metadata.get('sdk_version')
        for side in ('left','right'):
            p,q,c,ts,seq,ats,frame_ids = validate_stream(h[side], side)
            sn = metadata['devices'][side]['sn']
            urdf = config.get(f'{side}_urdf')
            pipeline = WujiGlove.offline_pipeline(sn=sn, hand_side=side, urdf_path=urdf)
            if urdf:
                if pipeline.urdf_source != 'override' or Path(pipeline.urdf_source_path).resolve() != Path(urdf):
                    raise RuntimeError(f'{side}: SDK did not load the pinned URDF')
            elif pipeline.urdf_source != 'builtin_default':
                raise RuntimeError(f'{side}: expected builtin_default, got {pipeline.urdf_source}')
            model = HandModel.WujiHand if generation == "hand1" else HandModel.WujiHand2
            if side == "left":
                print(
                    f"[emf-offline] SDK RetargetSession model={model} "
                    f"generation={generation}",
                    flush=True,
                )
            session = RetargetSession.for_hand(
                model,
                side=Handedness.Left if side=='left' else Handedness.Right,
            )
            n = len(ts)
            skeletons = np.empty((n,21,3),np.float32)
            rotations = np.empty((n,21,4),np.float64)
            confidence = np.empty((n,21),np.float32)
            commands = np.empty((n,20),np.float32)
            lookup = {}
            names = None
            for i in range(n):
                frame = EmfPoseArray(header=FrameHeader(seq=int(seq[i]),timestamp_us=int(ts[i]//1000),frame_id=frame_ids[i]),
                    poses=[EmfPose(pose=Pose(position=p[i,j].tolist(),orientation=Quaternion(*q[i,j].tolist())),
                                   confidence=float(c[i,j])) for j in range(5)])
                skeleton = pipeline.hand_skeleton().compute(frame)
                current_names = tuple(j.name for j in skeleton.joints)
                if names is None:
                    from adapters.wuji_glove_utils import JOINT_ORDER
                    names = tuple(JOINT_ORDER)
                if len(current_names) != 21 or set(current_names) != set(names):
                    raise RuntimeError('Unexpected SDK skeleton joint names')
                joints = {j.name:j for j in skeleton.joints}
                skeletons[i] = [joints[k].pose.position for k in names]
                rotations[i] = [[getattr(joints[k].pose.orientation,a) for a in 'xyzw'] for k in names]
                confidence[i] = [joints[k].confidence for k in names]
                commands[i] = session.step(skeletons[i])
                lookup.setdefault(fingerprint(p[i],q[i],c[i]),[]).append(i)
            if not all(np.isfinite(x).all() for x in (skeletons, rotations, confidence, commands)):
                raise RuntimeError(f'{side}: SDK produced nonfinite values')
            arrays.update({f'{side}_hand_q':commands, f'{side}_skeleton_position':skeletons,
                f'{side}_skeleton_orientation_xyzw':rotations, f'{side}_skeleton_confidence':confidence,
                f'{side}_source_timestamp_ns':ts, f'{side}_source_seq':seq,
                f'{side}_action_timestamp_ns':ats, f'{side}_hand_q_names':np.asarray(actuated_joint_names(side,generation)),
                f'{side}_skeleton_joint_names':np.asarray(names)})
            info['sides'][side] = {'frames':n, 'sn':sn,'urdf_source':pipeline.urdf_source,
                'urdf_source_path':pipeline.urdf_source_path,'urdf_sha256':config.get(f'{side}_sha256'),
                'timestamp_start_ns':int(ts[0]),'timestamp_end_ns':int(ts[-1])}
            lookups[side] = lookup
    return arrays, info, lookups


_METADATA_JSON_NAMES = {
    "instruction.json",
    "instruction_meta.json",
    "diversity.json",
    "calibration.json",
    "replay_metadata.json",
    "diagnostics.json",
    "wuji_glove_offline_manifest.json",
}


def episode_json_path(episode):
    """Locate the camera-timeline JSON even when the directory was renamed."""

    source = Path(episode)
    if source.is_file():
        return source
    named = source / f"{source.name}.json"
    if named.is_file():
        return named
    matches = [
        path
        for path in sorted(source.glob("*.json"))
        if path.name not in _METADATA_JSON_NAMES
        and not path.name.endswith(".meta.json")
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot locate episode JSON for EMF alignment in {source}")


def align_camera(ep, native, lookups, episode_json=None):
    rows, timestamps, matches = [],[],{'left':[],'right':[]}
    missing = {'left':[], 'right':[]}
    with Path(episode_json or episode_json_path(ep)).open('rb') as f:
        for frame_index, frame in enumerate(ijson.items(f,'data.item')):
            rows.append(frame_index)
            timestamp = int(frame['timestamp'])
            timestamps.append(timestamp)
            for side in ('left','right'):
                msg = frame.get(f'{side}_glove_emf_poses')
                candidates = []
                if isinstance(msg,dict):
                    candidates = lookups[side].get(fingerprint(*json_emf_payload(msg)),[])
                if candidates:
                    # Identical payloads can repeat. Resolve by the recorded action clock.
                    times = native[f'{side}_action_timestamp_ns']
                    index = min(candidates,key=lambda i:abs(int(times[i])-timestamp))
                else:
                    index = -1
                    missing[side].append(frame_index)
                matches[side].append(index)
    valid = np.logical_and(np.asarray(matches['left'])>=0,np.asarray(matches['right'])>=0)
    if not valid.any():
        raise RuntimeError('No camera frame has exact left/right EMF payload matches')
    output = {k: v for k, v in native.items() if k.startswith("hand_")}
    output.update(hand_sdk_input=np.asarray('local_emf_offline_pipeline'),
        hand_sdk_step_policy=np.asarray('native_emf_then_exact_camera_payload_selection'),
        hand_sdk_preserve_q=np.asarray(True), frame_indices=np.asarray(rows,dtype=np.int64)[valid],
        timestamps=np.asarray(timestamps,dtype=np.int64)[valid]/1e9,
        camera_timestamp_ns=np.asarray(timestamps,dtype=np.int64)[valid])
    for side in ('left','right'):
        indices=np.asarray(matches[side],dtype=np.int64)[valid]
        output[f'{side}_native_frame_indices']=indices
        for suffix in ('hand_q','skeleton_position','skeleton_orientation_xyzw','skeleton_confidence','source_timestamp_ns'):
            output[f'{side}_{suffix}']=native[f'{side}_{suffix}'][indices]
        output[f'{side}_hand_q_names']=native[f'{side}_hand_q_names']
    output['wuji_hand_q']=np.concatenate([output['right_hand_q'],output['left_hand_q']],axis=1)
    output['wuji_hand_q_names']=np.concatenate([output['right_hand_q_names'],output['left_hand_q_names']])
    return output, {'camera_frames':len(rows),'paired_frames':int(valid.sum()),
                    'dropped_frame_indices':np.asarray(rows)[~valid].tolist(),'unmatched_by_side':missing}


def compute_episode(episode, config=None, *, generation="hand1"):
    from wuji_retarget.sdk_retarget import resolve_offline_h5, use_emf_input
    h5 = resolve_offline_h5(episode)
    ep = h5.parent
    json_path = episode_json_path(episode)
    if not json_path.is_file():
        json_path = episode_json_path(ep)
    use_emf_input(ep)
    sdk_metadata(generation)
    cfg=load_model_config(config)
    native, info, lookup=solve_native(ep,cfg,generation)
    camera,alignment=align_camera(ep,native,lookup,json_path)
    info['alignment']=alignment
    camera['hand_sdk_emf_metadata_json']=np.asarray(json.dumps(info))
    return native,camera,info


def process_episode(episode, output, config=None):
    started=time.monotonic()
    native,camera,info=compute_episode(episode,config)
    destination=Path(output)/Path(episode).name
    destination.mkdir(parents=True,exist_ok=True)
    for name,data in [('emf_native.npz',native),('hand_sdk.npz',camera)]:
        temp=destination/(name+'.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,**data)
        temp.replace(destination/name)
    info.update(episode=str(episode),output=str(destination),seconds=time.monotonic()-started,status='complete')
    manifest=destination/'manifest.json'
    temp=manifest.with_suffix('.tmp')
    temp.write_text(json.dumps(info,indent=2));temp.replace(manifest)
    return info


def _worker(args):
    try:return process_episode(*args)
    except Exception:
        return {'episode':str(args[0]),'status':'failed','error':traceback.format_exc()}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--episode');source.add_argument('--collection')
    p.add_argument('--output',required=True);p.add_argument('--model-config',default=str(DEFAULT_CONFIG))
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args(argv)
    if args.workers<1:p.error('--workers must be positive')
    cfg=load_model_config(args.model_config)
    episodes=[Path(args.episode)] if args.episode else sorted(p for p in Path(args.collection).iterdir() if p.is_dir())
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    tasks=[(str(ep),str(out),args.model_config) for ep in episodes]
    report={'collection':args.collection,'model_config':cfg,'episodes':[],'total':len(tasks)}
    summary=out/'summary.json'
    def record(result):
        report['episodes'].append(result)
        report['completed']=sum(x['status']=='complete' for x in report['episodes'])
        report['failed']=sum(x['status']=='failed' for x in report['episodes'])
        temp=summary.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(summary)
        print(f"[{len(report['episodes'])}/{len(tasks)}] {Path(result['episode']).name} {result['status']}",flush=True)
        if result['status']=='failed':print(result['error'],flush=True)
    if args.workers==1:
        for task in tasks:record(_worker(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(_worker,t) for t in tasks]):record(future.result())
    if report['failed']:raise SystemExit(1)


if __name__=='__main__':main()
