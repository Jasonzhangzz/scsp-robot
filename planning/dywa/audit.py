"""Read-only source/checkpoint and in-memory policy integrity checks."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_upstream(root=None):
    root = Path(__file__).parent / 'upstream' if root is None else Path(root)
    manifest = json.loads((root / 'PROVENANCE.json').read_text())
    patch_path = root / 'PATCHES.json'
    patches = json.loads(patch_path.read_text()) if patch_path.exists() else {}
    if set(patches) - set(manifest['files']):
        raise RuntimeError('Authorized patch names a file outside upstream provenance')
    for name, expected in manifest['files'].items():
        if name in patches:
            patch = patches[name]
            contents = (root / name).read_bytes()
            anchor = patch['insert_after'].encode()
            inserted = anchor + patch['inserted_line'].encode()
            restored = contents.replace(inserted, anchor, 1)
            if (patch['original_sha256'] != expected or contents.count(inserted) != 1
                    or hashlib.sha256(restored).hexdigest() != expected
                    or hashlib.sha256(contents).hexdigest() != patch['patched_sha256']):
                raise RuntimeError('DyWA source differs beyond the authorized patch: ' + name)
        elif sha256(root / name) != expected:
            raise RuntimeError('Official DyWA source differs: ' + name)
    return dict(repository=manifest['repository'], commit=manifest['commit'],
                verified_files=len(manifest['files']),
                unmodified_files=len(manifest['files']) - len(patches),
                authorized_patches=patches)


def fingerprint_files(spec):
    expected = json.loads(Path(__file__).with_name('WEIGHTS.json').read_text())
    result = {key: sha256(spec[key]) for key in
              ('student_ckpt', 'teacher_ckpt', 'env_ckpt', 'icp_ckpt')}
    for key, digest in result.items():
        name = Path(spec[key]).name
        if name not in expected or digest != expected[name]['sha256']:
            raise RuntimeError('Checkpoint differs from the official published weights: ' + name)
    return result


def policy_digest(student):
    """Include registered buffers as well as every parameter; exclude recurrent history."""
    digest = hashlib.sha256()
    for name, tensor in sorted(student.state_dict().items()):
        if name.endswith(".history"):
            continue
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_student(student, path):
    """Strictly load every trained tensor, initialize only batch-dependent memory.

    The upstream loader intentionally drops history before strict loading, but
    history is a persistent buffer. Supply the evaluation batch's reset memory
    explicitly so strict=True never ignores missing/unexpected policy weights.
    """
    import torch
    checkpoint = torch.load(str(path), map_location='cpu')['self']
    current = student.state_dict()
    transient = {'history_tokenizer.history', 'aggregator.aggregator.memory'}
    state = dict(checkpoint)
    for key in transient:
        if key in current:
            state[key] = current[key].detach().clone()
        else:
            state.pop(key, None)
    student.load_state_dict(state, strict=True)
    for key, tensor in student.state_dict().items():
        if key not in transient and not torch.equal(tensor.detach().cpu(), checkpoint[key]):
            raise RuntimeError('Loaded DyWA tensor differs from checkpoint: ' + key)
