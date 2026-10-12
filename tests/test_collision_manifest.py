import hashlib
import json
import sys

import pytest

from models.object_collision_mesh import decomposed_collision_mesh


def test_manifest_reuses_exact_assets_without_coacd_and_rejects_changes(tmp_path, monkeypatch):
    source, surface, part = [tmp_path/name for name in ('source.stl', 'surface.stl', 'part.stl')]
    for path in (source, surface, part):
        path.write_bytes(path.name.encode())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(source_sha256=digest(source), surface=str(surface),
        parts=[str(part)], asset_sha256={str(p):digest(p) for p in (surface,part)})))
    monkeypatch.setitem(sys.modules, 'coacd', None)
    assert decomposed_collision_mesh(str(source), str(manifest)) == (str(surface), [str(part)])
    source.write_bytes(b'changed source')
    with pytest.raises(ValueError, match='canonical object mesh'):
        decomposed_collision_mesh(str(source), str(manifest))
    source.write_bytes(source.name.encode())
    part.write_bytes(b'changed collision mesh')
    with pytest.raises(ValueError, match='asset changed'):
        decomposed_collision_mesh(str(source), str(manifest))
