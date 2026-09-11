#!/usr/bin/env python3
"""Smoke-test that the clean scsp-robot migration is import-complete."""

from __future__ import annotations

import ast
import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = Path("/home/lab423/scsp/Franka-contact-face-detection-manipulation-main")
LOCAL_ROOTS = ("examples", "planning", "envs", "contact", "models", "utils", "mujoco_mpc")
HEAVY_OPTIONAL = {
    "isaacgym", "torch", "jax", "mujoco", "casadi", "curobo", "trimesh",
    "brax", "scienceplots", "art", "emoji", "yaml", "matplotlib",
}


def repo_py_files() -> list[Path]:
    skip = {"tests/test_smoke.py"}
    out = []
    for p in ROOT.rglob("*.py"):
        rel = str(p.relative_to(ROOT))
        if rel in skip or "__pycache__" in rel:
            continue
        out.append(p)
    return sorted(out)


def find_entries() -> list[Path]:
    entries = []
    for root in (ROOT / "examples/mpc/fingertips", ROOT / "examples/mpc/franka"):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            has_main_fn = any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body)
            has_guard = False
            for n in tree.body:
                if not isinstance(n, ast.If):
                    continue
                t = n.test
                if (
                    isinstance(t, ast.Compare)
                    and isinstance(t.left, ast.Name)
                    and t.left.id == "__name__"
                    and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in t.comparators)
                ):
                    has_guard = True
            if has_main_fn or has_guard:
                entries.append(path)
    return entries


def resolve_local(module: str, names: list[str], from_file: Path, root: Path = ROOT) -> list[Path]:
    found = []
    if module and "." not in module:
        sibling = from_file.parent / f"{module}.py"
        if sibling.is_file():
            found.append(sibling)
        src_sibling = SRC / from_file.parent.relative_to(ROOT) / f"{module}.py" if from_file.is_relative_to(ROOT) else None
        if root is SRC and src_sibling is not None and src_sibling.is_file():
            found.append(src_sibling)
    if not module:
        return found
    pkg_root = module.split(".")[0]
    if pkg_root not in LOCAL_ROOTS:
        return found
    parts = module.split(".")
    search_roots = [root]
    if pkg_root == "mujoco_mpc":
        search_roots.append(root / "mujoco_mpc" / "python")
    for search in search_roots:
        base = search.joinpath(*parts)
        if base.with_suffix(".py").is_file():
            found.append(base.with_suffix(".py"))
        if (base / "__init__.py").is_file():
            found.append(base / "__init__.py")
        for name in names:
            if name == "*":
                continue
            sub = base / f"{name}.py"
            if sub.is_file():
                found.append(sub)
    return found


def check_local_imports(path: Path) -> list[str]:
    missing = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            items = [(a.name, []) for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            items = [(n.module or "", [a.name for a in n.names])]
        else:
            continue
        for module, names in items:
            root = module.split(".")[0] if module else ""
            if root in HEAVY_OPTIONAL or root in sys.builtin_module_names:
                continue
            if root and root not in LOCAL_ROOTS:
                continue
            resolved = resolve_local(module, names, path, ROOT)
            if module and root in LOCAL_ROOTS and not resolved:
                # Only fail when the source tree actually has this module.
                # Pre-existing broken imports in the original repo are not
                # treated as migration gaps.
                src_has = False
                if SRC.is_dir():
                    src_file = SRC / path.relative_to(ROOT)
                    src_has = bool(resolve_local(module, names, src_file if src_file.exists() else path, SRC))
                    extra = SRC / "Complementarity-Free-Dexterous-Manipulation" / "envs" / f"{module.split('.')[-1]}.py"
                    if extra.is_file() and module.startswith("envs."):
                        src_has = True
                if src_has:
                    missing.append(f"{path.relative_to(ROOT)} -> {module}")
    return missing


def check_default_assets() -> list[str]:
    missing = []
    required = [
        "utils/rotations.py",
        "utils/metrics.py",
        "planning/mpc_implicit.py",
        "envs/xmls/env_fingertips_foam_brick.xml",
        "envs/assets/objects/foam_brick.stl",
        "envs/assets/textures/iris_block.png",
        "envs/robots/assets/urdf/franka_description/robots/franka_panda_gripper.urdf",
        "envs/robots/assets/urdf/franka_description/robots/franka_panda.urdf",
        "envs/robots/assets/urdf/franka_description/meshes/visual/link0.dae",
        "envs/xmls/panda_nohand.xml",
        "envs/xmls/assets/link0.stl",
        "IsaacGymEnvs/assets/urdf/sektion_cabinet_model/urdf/sektion_cabinet_2.urdf",
        "IsaacGymEnvs/assets/urdf/sektion_cabinet_model/meshes/sektion.obj",
        "mujoco_mpc/python/mujoco_mpc/demos/predictive_sampling/predictive_sampling.py",
        "envs/assets/objects/elephant.stl",
        "envs/assets/objects/teapot.stl",
        "examples/mpc/fingertips/test/test_0902.py",
        "examples/mpc/franka/ik2/test_mppi_isaac.py",
        "thirdparty/spider/spider/assets/robots/allegro/right.xml",
        "thirdparty/spider/spider/assets/robots/allegro/assets/base_link.stl",
    ]
    for rel in required:
        if not (ROOT / rel).is_file():
            missing.append(rel)
    return missing


def main() -> int:
    sys.path.insert(0, str(ROOT))
    errors = []

    entries = find_entries()
    if not entries:
        errors.append("no executable main entry points found")
    print(f"[smoke] entry points: {len(entries)}")
    for p in entries:
        print(f"  - {p.relative_to(ROOT)}")

    py_files = repo_py_files()
    print(f"[smoke] compiling {len(py_files)} python files")
    for p in py_files:
        try:
            py_compile.compile(str(p), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"compile failed: {p.relative_to(ROOT)}: {exc}")

    print("[smoke] resolving local imports")
    for p in py_files:
        errors.extend(check_local_imports(p))

    print("[smoke] checking required assets")
    errors.extend(f"missing asset: {m}" for m in check_default_assets())

    print("[smoke] importing lightweight modules")
    try:
        from utils import rotations, metrics  # noqa: F401
    except Exception as exc:
        errors.append(f"import utils failed: {exc}")

    try:
        import models.explicit_model  # noqa: F401
    except ModuleNotFoundError as exc:
        print(f"[smoke] skip models.explicit_model (optional dep missing): {exc}")
    except Exception as exc:
        errors.append(f"import models.explicit_model failed: {exc}")

    # Import params modules that do not construct a simulator at import time.
    try:
        import examples.mpc.franka.ik2.params  # noqa: F401
        print("[smoke] imported examples.mpc.franka.ik2.params")
    except ModuleNotFoundError as exc:
        print(f"[smoke] skip ik2.params (optional dep missing): {exc}")
    except Exception as exc:
        errors.append(f"import ik2.params failed: {exc}")

    if errors:
        print("\n[smoke] FAILED")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\n[smoke] PASSED")
    print(f"  entries={len(entries)} python_files={len(py_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
