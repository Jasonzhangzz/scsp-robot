"""Locate DexForge's in-memory CA/FA APIs without invoking their dataset CLIs."""

import os
import sys
from pathlib import Path


def ensure_dexforge():
    candidates = (
        os.environ.get("DEXFORGE_ROOT"),
        Path(__file__).resolve().parents[2] / "GraspSONIC/thirdparty/DexForge",
    )
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser().resolve()
        packages = (root / "contactaware", root / "forceaware")
        if not all((p / p.name / "__init__.py").is_file() for p in packages):
            continue
        comfree = Path(os.environ.get(
            "COMFREE_WARP_ROOT", str(root / "forceaware/third_party/comfree_warp")
        )).expanduser().resolve()
        for package in (*packages, comfree):
            if str(package) not in sys.path:
                sys.path.insert(0, str(package))
        os.environ.setdefault("COMFREE_WARP_ROOT", str(comfree))
        return root
    raise ImportError("DexForge CA/FA not found; set DEXFORGE_ROOT to the DexForge checkout")
