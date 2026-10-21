"""Download pinned official assets and checkpoints; never change policy code."""
import argparse
import json
from pathlib import Path
import subprocess
import urllib.request

from planning.dywa.audit import sha256
from planning.dywa.runtime import DYWA_ROOT, DEFAULT_CKPT_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DYWA_ROOT)
    parser.add_argument('--ckpt-dir', type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument('--build-kinematics', action='store_true',
                        help='Build original CUDA kinematics extensions with the active PyTorch/CUDA.')
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('upstream').joinpath('PROVENANCE.json').read_text())
    if not args.root.exists():
        subprocess.run(['git', 'clone', manifest['repository'], str(args.root)], check=True)
        subprocess.run(['git', '-C', str(args.root), 'checkout', manifest['commit']], check=True)
    actual = subprocess.check_output(['git', '-C', str(args.root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != manifest['commit']:
        raise RuntimeError('Asset repository is not at the pinned upstream revision')
    weights = json.loads(Path(__file__).with_name('WEIGHTS.json').read_text())
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    for name, info in weights.items():
        path = args.ckpt_dir / name
        if path.exists() and sha256(path) == info['sha256']:
            print('Verified', path, flush=True)
            continue
        partial = path.with_suffix('.partial')
        print('Downloading', info['url'], flush=True)
        urllib.request.urlretrieve(info['url'], partial)
        if sha256(partial) != info['sha256']:
            raise RuntimeError('Downloaded checkpoint hash mismatch: ' + name)
        partial.replace(path)
    if args.build_kinematics:
        from torch.utils.cpp_extension import load
        source = Path(__file__).parent / 'upstream/dywa/c_src'
        for name in ('franka', 'ur5'):
            build = args.root / 'native/cxx'
            build.mkdir(parents=True, exist_ok=True)
            load(name + '_kin_cuda',
                 sources=[str((source / (name + '_kin_cuda.cpp')).resolve()),
                          str((source / (name + '_kin_cuda_kernel.cu')).resolve())],
                 extra_include_paths=['/usr/include/eigen3'],
                 build_directory=str(build.resolve()), verbose=True)
    print('Official assets and weights are ready. Native Python/CUDA dependencies are separate.')


if __name__ == '__main__':
    main()
