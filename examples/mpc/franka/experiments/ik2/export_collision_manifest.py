"""Export shared convex geometry in a Python environment containing CoACD.

The Isaac Gym environment can read this manifest without installing CoACD.
Source and asset hashes ensure that both environments use identical meshes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))


def main():
    os.environ['SCSP_PLANNER_ONLY'] = '1'
    from examples.mpc.franka.runtime.mpc import _add_rollout_policy_args
    from examples.mpc.franka.runtime.params import ExplicitMPCParams
    parser = _add_rollout_policy_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    param = ExplicitMPCParams(args, rand_seed=0, target_type=args.target_type, build_optimizer=False)
    if not param.collision_mesh_paths_:
        parser.error('Choose a decomposed object, or --collision-geometry decomposed')
    files = [param.mesh_path_, *param.collision_mesh_paths_]
    payload = dict(surface=param.mesh_path_, parts=param.collision_mesh_paths_,
        source_sha256=hashlib.sha256(Path(param.visual_mesh_path_).read_bytes()).hexdigest(),
        asset_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in files},
        decomposition=('mug-wall-sectors-v2' if args.obj == 'mug' else 'coacd-v1'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+'\n')
    print('Saved %s convex parts to %s' % (len(param.collision_mesh_paths_), args.output))


if __name__ == '__main__':
    main()
