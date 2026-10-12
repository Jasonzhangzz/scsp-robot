"""Restore the saved source; refuse to overwrite changes made after the experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--restore', action='store_true', help='Perform rollback; otherwise only list files.')
args = parser.parse_args()
folder = Path(__file__).resolve().parent
root = folder.parents[2]
manifest = json.loads((folder/'baseline/manifest.json').read_text())
for relative, expected in manifest['experiment_files'].items():
    path = root/relative
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit('Refusing rollback: source changed after the experiment: '+relative)
for relative, expected in manifest['files'].items():
    if hashlib.sha256((folder/'baseline'/relative).read_bytes()).hexdigest() != expected:
        raise SystemExit('Refusing rollback: baseline backup changed: '+relative)
print('Baseline commit:', manifest['commit'])
for relative in manifest['files']:
    print('Restore:', relative)
    if args.restore:
        shutil.copy2(folder/'baseline'/relative, root/relative)
for relative in manifest['added_files']:
    print('Remove:', relative)
    if args.restore:
        (root/relative).unlink()
print('Rollback complete.' if args.restore else 'Dry run only; add --restore to roll back.')
