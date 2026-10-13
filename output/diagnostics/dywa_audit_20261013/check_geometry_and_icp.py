"""Independent numerical checks; no policy weights or simulator are loaded."""
import ast
import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = json.loads((HERE / "spec.json").read_text())
vertices = np.asarray(trimesh.load(spec["mesh_path"], process=False).vertices)
geometry = {}
for name, position, quaternion in (
    ("initial", spec["init_obj_qpos"][:3], spec["init_obj_qpos"][3:]),
    ("goal", spec["target_p"], spec["target_q"]),
):
    quaternion = np.asarray(quaternion)
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
    bottom = float((rotation.apply(vertices) + position)[:, 2].min())
    geometry[name] = {
        "mesh_bottom_z": bottom,
        "upstream_tabletop_z": 0.4,
        "gap_to_upstream_table": bottom - 0.4,
    }

# Execute the actual nested function from the current repository, independently
# of _install_pytorch3d_ops(), which imports the missing pytorch3d package.
module = ast.parse((ROOT / "planning/dywa_interface.py").read_text())
function = next(
    node for node in ast.walk(module)
    if isinstance(node, ast.FunctionDef) and node.name == "iterative_closest_point"
)
namespace = {
    "torch": torch,
    "SimilarityTransform": namedtuple("SimilarityTransform", "R T s"),
    "ICPSolution": namedtuple("ICPSolution", "converged rmse Xt RTs t_history"),
}
exec(compile(ast.Module(body=[function], type_ignores=[]), "dywa_icp", "exec"), namespace)
torch.manual_seed(42)
source = 0.03 * torch.randn(1, 100, 3, dtype=torch.float64)
angle = np.deg2rad(5)
rotation = torch.tensor([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 1],
], dtype=torch.float64)
target = source @ rotation
actual = namespace["iterative_closest_point"](source, target, max_iterations=1)
centered_source = source - source.mean(1, keepdim=True)
centered_target = target - target.mean(1, keepdim=True)
left, _, right = torch.linalg.svd(centered_source.transpose(1, 2) @ centered_target)
correct = left @ right
icp = {
    "known_rotation_degrees": 5,
    "actual_rotation_equals_inverse": bool(torch.allclose(actual.RTs.R[0], rotation.T, atol=1e-8)),
    "actual_one_step_corresponding_point_rmse_m": float(torch.sqrt(((actual.Xt - target) ** 2).mean())),
    "correct_one_step_corresponding_point_rmse_m": float(torch.sqrt(((source @ correct - target) ** 2).mean())),
}
assert icp["actual_rotation_equals_inverse"]
assert icp["correct_one_step_corresponding_point_rmse_m"] < 1e-10
result = {"geometry": geometry, "icp": icp}
(HERE / "numerical_checks.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
