#!/usr/bin/env python3
"""Migrate executable MPC mains and their used deps into a clean scsp-robot tree."""

from __future__ import annotations

import ast
import json
import os
import py_compile
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET


SRC = Path("/home/lab423/scsp/Franka-contact-face-detection-manipulation-main").resolve()
DST = Path("/home/lab423/scsp/scsp-robot").resolve()
ENTRY_DIRS = [
    SRC / "examples/mpc/fingertips",
    SRC / "examples/mpc/franka",
]

LOCAL_ROOTS = ("examples", "planning", "envs", "contact", "models", "utils", "mujoco_mpc")
ASSET_SEARCH_ROOTS = (
    SRC,
    SRC / "envs/robots/assets/urdf",
    SRC / "envs/xmls",
    SRC / "envs/assets/objects",
    SRC / "IsaacGymEnvs/assets",
    SRC / "IsaacGymEnvs/assets/urdf",
)
ASSET_SUFFIXES = {
    ".py", ".xml", ".urdf", ".stl", ".obj", ".ply", ".off", ".mjcf",
    ".png", ".jpg", ".jpeg", ".mtl", ".yml", ".yaml", ".dae", ".stl",
}
SKIP_NAME_PARTS = {
    "__pycache__", ".git", "outputs", "figs", ".ipynb_checkpoints",
}
SKIP_SUFFIXES = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".svg", ".pptx", ".zip",
    ".pyc", ".pyo", ".so", ".o", ".backup",
}
SKIP_ASSET_PREFIXES = (
    "planning/dial_mpc/models/",
    "planning/dial_mpc/examples/",
    "outputs/",
    "examples/mpc/franka/ik2/figs/",
)
GENERATED_NAME_MARKERS = ("_generated_", "_isaac_tmp", "c_generated_code")
OLD_ROOT_STR = str(SRC)
STDLIB_HINTS = set(sys.builtin_module_names) | {
    "argparse", "os", "sys", "time", "json", "re", "math", "copy", "logging",
    "dataclasses", "pathlib", "typing", "collections", "concurrent", "functools",
    "itertools", "tempfile", "hashlib", "warnings", "signal", "shutil", "subprocess",
    "xml", "ast", "bisect", "importlib", "ctypes", "abc", "enum", "traceback",
    "multiprocessing", "threading", "queue", "glob", "fnmatch", "pprint",
    "datetime", "textwrap", "inspect", "pkgutil", "types", "io", "struct",
}


def is_skipped_path(rel: str) -> bool:
    parts = Path(rel).parts
    if any(p in SKIP_NAME_PARTS for p in parts):
        return True
    if any(rel.startswith(p) for p in SKIP_ASSET_PREFIXES):
        return True
    if any(marker in rel for marker in GENERATED_NAME_MARKERS):
        return True
    suffix = Path(rel).suffix.lower()
    if suffix in SKIP_SUFFIXES:
        return True
    name = Path(rel).name
    if name.endswith(".ipopt-backup.py") or name.endswith(".backup-0902"):
        return True
    return False


def find_entry_points() -> list[Path]:
    entries = []
    for root in ENTRY_DIRS:
        for path in sorted(root.rglob("*.py")):
            rel = str(path.relative_to(SRC))
            if is_skipped_path(rel):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
            except SyntaxError:
                continue
            has_main_fn = any(
                isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body
            )
            has_main_guard = False
            for n in tree.body:
                if not isinstance(n, ast.If):
                    continue
                test = n.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                    and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators)
                ):
                    has_main_guard = True
            if has_main_fn or has_main_guard:
                entries.append(path)
    return entries


def parse_imports(path: Path) -> list[tuple[str, list[str]]]:
    """Return [(module, [names])] for import statements."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError:
        return []
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for alias in n.names:
                out.append((alias.name, []))
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            names = [a.name for a in n.names]
            out.append((mod, names))
    return out


def candidate_files_for_module(module: str, names: list[str], from_file: Path) -> list[Path]:
    cands = []
    if not module and names:
        # relative-looking same-dir imports already handled separately
        return cands
    root = module.split(".")[0] if module else ""
    if root and root not in LOCAL_ROOTS:
        # same-directory sibling import: from trigrasp_casadi_param import ...
        if module and "." not in module:
            sibling = from_file.parent / f"{module}.py"
            if sibling.is_file():
                cands.append(sibling)
            sibling_pkg = from_file.parent / module / "__init__.py"
            if sibling_pkg.is_file():
                cands.append(sibling_pkg)
        return cands

    parts = module.split(".") if module else []
    search_roots = [SRC]
    if parts and parts[0] == "mujoco_mpc":
        search_roots.append(SRC / "mujoco_mpc" / "python")
    for root_dir in search_roots:
        base = root_dir.joinpath(*parts)
        if base.with_suffix(".py").is_file():
            cands.append(base.with_suffix(".py"))
        if (base / "__init__.py").is_file():
            cands.append(base / "__init__.py")
        for name in names:
            if name == "*":
                continue
            sub_py = base / f"{name}.py"
            sub_init = base / name / "__init__.py"
            if sub_py.is_file():
                cands.append(sub_py)
            if sub_init.is_file():
                cands.append(sub_init)
    return cands


def _const_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        chunks = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                chunks.append(v.value)
            else:
                return None
        return "".join(chunks)
    return None


def extract_joined_paths(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError:
        return []
    out: list[str] = []

    def join_args(args: list[ast.AST]) -> str | None:
        parts = []
        for arg in args:
            s = _const_str(arg)
            if s is None:
                return None
            parts.append(s)
        return str(Path(*parts)) if parts else None

    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            func = n.func
            is_join = (
                (isinstance(func, ast.Attribute) and func.attr == "join")
                or (isinstance(func, ast.Name) and func.id == "join")
            )
            if is_join and n.args:
                joined = join_args(n.args)
                if joined:
                    out.append(joined)
                # also keep join of the tail args (repo-relative fragments)
                if len(n.args) >= 2:
                    tail = join_args(n.args[1:])
                    if tail:
                        out.append(tail)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
            left = _const_str(n.left)
            right = _const_str(n.right)
            if left and right:
                out.append(str(Path(left) / right))
            # flatten nested Path / "a" / "b"
            parts = []
            cur = n
            ok = True
            while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Div):
                s = _const_str(cur.right)
                if s is None:
                    ok = False
                    break
                parts.append(s)
                cur = cur.left
            head = _const_str(cur)
            if ok and head:
                parts.append(head)
                out.append(str(Path(*reversed(parts))))
    return out


def extract_string_literals(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError:
        return []
    out = []
    for n in ast.walk(tree):
        s = _const_str(n) if isinstance(n, (ast.Constant, ast.JoinedStr)) else None
        if s:
            out.append(s)
    out.extend(extract_joined_paths(path))
    return out


def extract_obj_defaults(path: Path) -> set[str]:
    objs = set()
    text = path.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(
        r"add_argument\(\s*['\"]--obj['\"][^)]*?default\s*=\s*['\"]([^'\"]+)['\"]",
        text,
        flags=re.S,
    ):
        objs.add(m.group(1))
    return objs


def looks_like_repo_relpath(value: str) -> bool:
    if not value or len(value) > 400:
        return False
    if "\n" in value:
        return False
    cleaned = value.replace("\\", "/").lstrip("./")
    if cleaned.startswith(OLD_ROOT_STR):
        return True
    if any(cleaned.startswith(p + "/") or cleaned == p for p in LOCAL_ROOTS + ("IsaacGymEnvs",)):
        suffix = Path(cleaned).suffix.lower()
        return (not suffix) or suffix in ASSET_SUFFIXES or cleaned.endswith("/")
    suffix = Path(cleaned).suffix.lower()
    keywords = (
        "franka_description", "sektion_cabinet", "panda_nohand", "panda.xml",
        "nv_humanoid", "urdf/", "xmls/", "assets/objects",
    )
    if any(k in cleaned for k in keywords) and suffix in ASSET_SUFFIXES:
        return True
    return suffix in ASSET_SUFFIXES and ("/" in cleaned or suffix in {".xml", ".urdf", ".stl"})


def resolve_literal_path(value: str, from_file: Path) -> Path | None:
    value = value.strip()
    if value.startswith(OLD_ROOT_STR):
        p = Path(value)
        return p if p.exists() else None
    rel = value.lstrip("./")
    candidates = [SRC / rel, (from_file.parent / value)]
    for root in ASSET_SEARCH_ROOTS:
        candidates.append(root / rel)
        candidates.append(root / Path(rel).name)
    # package-style fragments
    if "franka_description" in rel:
        candidates.append(SRC / "envs/robots/assets/urdf" / rel)
    if "sektion_cabinet" in rel:
        candidates.append(SRC / "envs/robots/assets/urdf" / rel)
        candidates.append(SRC / "IsaacGymEnvs/assets" / rel)
        candidates.append(SRC / "IsaacGymEnvs/assets/urdf" / Path(rel).name)
    for c in candidates:
        c = c.resolve() if c.exists() else c
        if c.is_file() and str(c).startswith(str(SRC)):
            return c
    return None


def xml_referenced_files(path: Path) -> list[Path]:
    refs = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return refs
    meshdirs = [path.parent]
    for m in re.finditer(r'meshdir\s*=\s*["\']([^"\']+)["\']', text):
        meshdirs.append((path.parent / m.group(1)).resolve())
    for m in re.finditer(r'texturedir\s*=\s*["\']([^"\']+)["\']', text):
        meshdirs.append((path.parent / m.group(1)).resolve())
    attr_re = re.compile(
        r'''(?:file|filename|href)=["']([^"']+)["']''',
        re.I,
    )
    for m in attr_re.finditer(text):
        raw = m.group(1)
        pkg = None
        if raw.startswith("package://"):
            pkg = raw.split("package://", 1)[1]
            raw = pkg
        if raw.startswith(OLD_ROOT_STR):
            p = Path(raw)
            if p.exists():
                refs.append(p)
            continue
        # skip thirdparty absolute paths outside SRC
        if raw.startswith("/"):
            p = Path(raw)
            if p.exists() and str(p).startswith(str(SRC)):
                refs.append(p)
            continue
        candidates = [(path.parent / raw).resolve()]
        for d in meshdirs:
            candidates.append((d / raw).resolve())
        if pkg:
            pkg_name = pkg.split("/")[0]
            pkg_rest = "/".join(pkg.split("/")[1:])
            candidates.append((SRC / "envs/robots/assets/urdf" / pkg).resolve())
            candidates.append((SRC / "IsaacGymEnvs/assets/urdf" / pkg).resolve())
            candidates.append((path.parent.parent / pkg).resolve())
            if pkg_rest:
                candidates.append((path.parents[1] / pkg_rest).resolve())
                candidates.append((SRC / "IsaacGymEnvs/assets/urdf" / pkg_name / pkg_rest).resolve())
                candidates.append((SRC / "envs/robots/assets/urdf" / pkg_name / pkg_rest).resolve())
        for c in candidates:
            if c.exists() and c.is_file() and str(c).startswith(str(SRC)):
                refs.append(c)
    return refs


def add_object_assets(obj: str, files: set[Path]) -> None:
    for xml_name in (
        f"envs/xmls/env_fingertips_{obj}.xml",
        f"envs/xmls/env_{obj}.xml",
        f"envs/xmls/{obj}.xml",
    ):
        p = SRC / xml_name
        if p.is_file():
            files.add(p)
    special = {
        "football": "envs/xmls/env_football.xml",
        "sphere": "envs/xmls/env_football.xml",
        "microwave": "envs/xmls/env_microwave.xml",
        "study_table_drawer": "envs/xmls/env_study_table_drawer.xml",
    }
    if obj in special:
        p = SRC / special[obj]
        if p.is_file():
            files.add(p)
    for ext in (".stl", ".obj", ".ply", ".off"):
        p = SRC / "envs/assets/objects" / f"{obj}{ext}"
        if p.is_file():
            files.add(p)


def collect_closure(entries: list[Path]) -> tuple[set[Path], dict]:
    files: set[Path] = set(entries)
    pending = list(entries)
    seen = set(entries)
    import_edges = defaultdict(list)
    objs: set[str] = set()

    while pending:
        path = pending.pop()
        if path.suffix == ".py":
            for module, names in parse_imports(path):
                root = module.split(".")[0] if module else ""
                if root in STDLIB_HINTS:
                    continue
                for cand in candidate_files_for_module(module, names, path):
                    import_edges[str(path.relative_to(SRC))].append(str(cand.relative_to(SRC)))
                    if cand not in seen:
                        seen.add(cand)
                        files.add(cand)
                        pending.append(cand)
            objs |= extract_obj_defaults(path)
            for lit in extract_string_literals(path):
                if not looks_like_repo_relpath(lit):
                    continue
                resolved = resolve_literal_path(lit, path)
                if resolved is None or not resolved.is_file():
                    continue
                rel = str(resolved.relative_to(SRC))
                if is_skipped_path(rel):
                    continue
                if resolved not in seen:
                    seen.add(resolved)
                    files.add(resolved)
                    pending.append(resolved)
        if path.suffix.lower() in {".xml", ".urdf", ".mjcf"}:
            rel = str(path.relative_to(SRC))
            if rel.startswith("planning/dial_mpc/"):
                continue
            for ref in xml_referenced_files(path):
                try:
                    rrel = str(ref.relative_to(SRC))
                except ValueError:
                    continue
                if is_skipped_path(rrel):
                    continue
                if ref not in seen:
                    seen.add(ref)
                    files.add(ref)
                    pending.append(ref)

    for obj in sorted(objs):
        before = set(files)
        add_object_assets(obj, files)
        for p in files - before:
            if p not in seen:
                seen.add(p)
                pending.append(p)
    # second pass for newly added xml/object files
    while pending:
        path = pending.pop()
        if path.suffix.lower() in {".xml", ".urdf", ".mjcf"}:
            rel = str(path.relative_to(SRC))
            if rel.startswith("planning/dial_mpc/"):
                continue
            for ref in xml_referenced_files(path):
                try:
                    rrel = str(ref.relative_to(SRC))
                except ValueError:
                    continue
                if is_skipped_path(rrel):
                    continue
                if ref not in seen:
                    seen.add(ref)
                    files.add(ref)
                    pending.append(ref)

    meta = {
        "entry_points": [str(p.relative_to(SRC)) for p in entries],
        "object_defaults": sorted(objs),
        "import_edges": {k: sorted(set(v)) for k, v in import_edges.items()},
    }
    return files, meta


def ensure_init_files(files: set[Path]) -> None:
    packages = set()
    for f in list(files):
        rel = f.relative_to(SRC)
        if rel.parts[0] not in LOCAL_ROOTS:
            continue
        parent = f.parent
        while parent != SRC and parent.is_dir():
            packages.add(parent)
            parent = parent.parent
    for pkg in packages:
        init = pkg / "__init__.py"
        if init.is_file():
            files.add(init)
        else:
            # create empty init in destination later
            files.add(init)


def rewrite_text(text: str) -> str:
    return text.replace(OLD_ROOT_STR, str(DST))


def copy_files(files: set[Path]) -> list[str]:
    copied = []
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    for src in sorted(files, key=lambda p: str(p)):
        rel = src.relative_to(SRC) if src.exists() or True else None
        try:
            rel = src.relative_to(SRC)
        except ValueError:
            continue
        dst = DST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            if src.name == "__init__.py":
                dst.write_text("", encoding="utf-8")
                copied.append(str(rel))
            continue
        if src.suffix.lower() in {".py", ".xml", ".urdf", ".mjcf", ".yml", ".yaml", ".md", ".txt", ".mtl"}:
            text = src.read_text(encoding="utf-8", errors="replace")
            dst.write_text(rewrite_text(text), encoding="utf-8")
        else:
            shutil.copy2(src, dst)
        copied.append(str(rel))
    return copied


def write_support_files(entries: list[Path], copied: list[str], meta: dict) -> None:
    (DST / "examples" / "__init__.py").write_text("", encoding="utf-8")
    (DST / "examples" / "mpc" / "__init__.py").write_text("", encoding="utf-8")
    for pkg in ("planning", "envs", "contact", "models", "utils"):
        init = DST / pkg / "__init__.py"
        init.parent.mkdir(parents=True, exist_ok=True)
        if not init.exists():
            init.write_text("", encoding="utf-8")

    reqs = """# Runtime extras used by the migrated MPC examples.
# Install only what a given entry point needs.
numpy
scipy
casadi
trimesh
tqdm
matplotlib
jax
torch
mujoco
"""
    (DST / "requirements.txt").write_text(reqs, encoding="utf-8")

    entry_lines = "\n".join(f"- `{p}`" for p in meta["entry_points"])
    obj_lines = ", ".join(meta["object_defaults"])
    readme = f"""# scsp-robot

Clean subset of the Franka contact / face-detection / manipulation MPC examples.

This tree keeps only:

1. Executable `main` entry points under `examples/mpc/fingertips` and `examples/mpc/franka`
2. Local Python modules those entries import
3. Mesh / XML / URDF / texture assets those modules actually reference

Videos, generated figures, unused object meshes, and unused third-party trees were not copied.

## Entry points

{entry_lines}

Default `--obj` names found on those entries: {obj_lines}

## Run

From this directory:

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
export ACADOS_SOURCE_DIR="/home/lab423/scsp/thirdparty/acados"
export LD_LIBRARY_PATH="/home/lab423/scsp/thirdparty/acados/lib:${{LD_LIBRARY_PATH}}"
python examples/mpc/fingertips/test/test_0902.py --headless --trial_num 1
python examples/mpc/franka/ik2/test_mppi_isaac.py --sim-device cuda:0 --mppi-device cuda:0
```

Isaac Gym, MuJoCo, acados, cuRobo, and `/home/lab423/scsp/thirdparty/spider` stay as external dependencies.

## Smoke test

```bash
python tests/test_smoke.py
```
"""
    (DST / "README.md").write_text(readme, encoding="utf-8")

    (DST / "MIGRATION_MANIFEST.json").write_text(
        json.dumps(
            {
                "source": str(SRC),
                "destination": str(DST),
                "entry_points": meta["entry_points"],
                "object_defaults": meta["object_defaults"],
                "copied_files": copied,
                "copied_count": len(copied),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    tests = DST / "tests"
    tests.mkdir(exist_ok=True)
    (tests / "__init__.py").write_text("", encoding="utf-8")
    smoke = r'''#!/usr/bin/env python3
"""Smoke-test that the clean scsp-robot migration is import-complete."""

from __future__ import annotations

import ast
import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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


def resolve_local(module: str, names: list[str], from_file: Path) -> list[Path]:
    found = []
    if module and "." not in module:
        sibling = from_file.parent / f"{module}.py"
        if sibling.is_file():
            found.append(sibling)
    if not module:
        return found
    root = module.split(".")[0]
    if root not in LOCAL_ROOTS:
        return found
    parts = module.split(".")
    base = ROOT.joinpath(*parts)
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
            resolved = resolve_local(module, names, path)
            if module and root in LOCAL_ROOTS and not resolved:
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
'''
    (tests / "test_smoke.py").write_text(smoke, encoding="utf-8")


def main() -> int:
    print(f"source: {SRC}")
    print(f"dest:   {DST}")
    entries = find_entry_points()
    print(f"entry points: {len(entries)}")
    for e in entries:
        print(f"  {e.relative_to(SRC)}")
    files, meta = collect_closure(entries)
    ensure_init_files(files)
    print(f"closure files: {len(files)}")
    copied = copy_files(files)
    write_support_files(entries, copied, meta)
    print(f"copied: {len(copied)} -> {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
