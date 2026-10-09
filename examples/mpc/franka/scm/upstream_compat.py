"""Minimal compile fix for the pinned upstream sparse Newton implementation.

Source: asu-iris/comfree_warp, ba8b9969f5f7859d51b1f738f458f8d2f49be080,
comfree_warp/mujoco_warp/_src/solver.py (Apache-2.0).
The diagonal-block branch calls tile_assign on a tile that was never defined.
Allocating it before the branch preserves the original Hessian calculation.
Only the benchmark's local dependency directory may be patched.
"""

from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import sys


MARKER = "      # SCM benchmark compatibility: initialize the diagonal-block tile."
ORIGINAL_SHA256 = "1b2a3fd547abb4bdc7c4735c65fb0f04a0d92a6c90df1ef06602a1c8384985e6"
PATCHED_SHA256 = "2fd358212b1c59aeb3b843291483daeecfd46196b49f5918ebaccb73963af798"
ANCHOR = """      J_ki = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(k, offset_i), bounds_check=False)

      if offset_i != offset_j:"""
REPLACEMENT = """      J_ki = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(k, offset_i), bounds_check=False)
""" + MARKER + """
      J_kj = wp.tile_zeros(shape=(TILE_SIZE, TILE_SIZE), dtype=wp.float32)

      if offset_i != offset_j:"""


def patch_sparse_newton():
    spec = importlib.util.find_spec("comfree_warp")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install requirements.txt into this directory's .deps using pi0 Python first.")
    source = Path(spec.origin).parent / "mujoco_warp" / "_src" / "solver.py"
    text = source.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest == PATCHED_SHA256:
        return
    if digest != ORIGINAL_SHA256 or text.count(ANCHOR) != 1:
        raise RuntimeError("Sparse Newton compatibility fix expects the pinned upstream revision in requirements.txt.")
    dependencies = Path(__file__).resolve().parent / ".deps"
    if not source.resolve().is_relative_to(dependencies):
        raise RuntimeError("Only the benchmark package in .deps may be patched; pi0's shared packages are left intact.")
    if "comfree_warp.mujoco_warp._src.solver" in sys.modules:
        raise RuntimeError("Run this benchmark in a fresh Python process so the compatibility fix is applied before import.")
    # Keep the exact upstream file so the local change is inspectable/reversible.
    backup = source.with_suffix(".py.upstream")
    if not backup.exists():
        backup.write_text(text)
    source.write_text(text.replace(ANCHOR, REPLACEMENT))
    importlib.invalidate_caches()
    print(f"Applied pinned upstream sparse Newton compile fix: {source}", flush=True)


if __name__ == "__main__":
    from runtime_env import activate_dependencies
    activate_dependencies()
    patch_sparse_newton()
