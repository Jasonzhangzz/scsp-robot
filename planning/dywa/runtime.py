"""DyWA student policy as an algorithm for the Isaac Franka experiment.

The policy owns its camera and low-level controller, so it runs inside DyWA's
simulator. The object mesh, init/goal orientation, mass, friction, and success
metric come from the same parameter builder as
``examples/mpc/franka/ik2/test_mpc_isaac.py``. World positions are shifted onto
DyWA's table while the relative translation and both quaternions are kept.
Actions retain the pretrained policy's 100 ms hold time.
"""

from __future__ import annotations

import json
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DYWA_ROOT = Path(os.environ.get("DYWA_ROOT", REPO_ROOT / "thirdparty" / "DyWA")).expanduser().resolve()
UPSTREAM = Path(__file__).resolve().parent / "upstream"
DYWA_SRC = UPSTREAM / "dywa" / "src"
DYWA_TRAIN = UPSTREAM / "dywa" / "exp" / "train"
DYWA_CFG = DYWA_SRC / "data" / "cfg"
DEFAULT_CKPT_DIR = DYWA_ROOT / "ckpts"


def resolve_acc_python():
    """Prefer an explicit interpreter, then the active/local acc environment."""
    configured = os.environ.get("DYWA_PYTHON")
    if configured:
        python = Path(configured).expanduser()
        if not python.is_file():
            raise FileNotFoundError(f"DYWA_PYTHON does not exist: {python}")
        return python
    if Path(sys.prefix).name == "acc":
        return Path(sys.executable)
    for root in ("miniconda3", "anaconda3", "mambaforge", "miniforge3"):
        python = Path.home() / root / "envs" / "acc" / "bin" / "python"
        if python.is_file():
            return python
    return Path(sys.executable)


ACC_PYTHON = resolve_acc_python()


def runtime_environment(python, environ=None):
    """Set loader paths before starting Python/Isaac Gym, preserving user paths."""
    env = dict(os.environ if environ is None else environ)
    python = Path(python).resolve()
    for key, directory in (("PATH", python.parent), ("LD_LIBRARY_PATH", python.parent.parent / "lib")):
        parts = [part for part in env.get(key, "").split(os.pathsep) if part]
        path = str(directory)
        if path not in parts:
            env[key] = os.pathsep.join([path, *parts])
    env.setdefault("PYTORCH_JIT", "0")
    return env


def ensure_acc_runtime():
    """Re-exec once if needed; changing LD_LIBRARY_PATH after startup is too late."""
    env = runtime_environment(ACC_PYTHON)
    if Path(sys.executable).resolve() != ACC_PYTHON.resolve() or any(
        env.get(key) != os.environ.get(key) for key in ("PATH", "LD_LIBRARY_PATH")
    ):
        os.execve(str(ACC_PYTHON), [str(ACC_PYTHON), *sys.argv], env)

# Pin the same table in both pose mapping and Hydra configuration.
DYWA_TABLE_POS = (0.0, 0.0, 0.2)
DYWA_TABLE_DIMS = (0.4, 0.5, 0.4)
_TABLE_HALF_EXTENTS = np.array([0.12, 0.17], dtype=np.float64)
POLICY_INTERVAL = 0.1
SIM_DT = 0.0125
ACTION_PERIOD = 8
# test_mpc_isaac.SVG_SCREENSHOT_CAMERA_POSITION / TARGET, relative to a Franka
# base at the origin. DyWA puts that base on the -X table edge.
_MPC_CAMERA_FROM_BASE = np.array([0.7, 0.0, 0.63], dtype=np.float64)
_MPC_CAMERA_TARGET_FROM_BASE = np.array([0.1, 0.0, 0.32], dtype=np.float64)


def place_pair_on_table(init_xy, target_xy, half_extents=_TABLE_HALF_EXTENTS):
    """Move a world-frame pair onto the table without changing its direction."""
    init_xy = np.asarray(init_xy, dtype=np.float64).reshape(2)
    target_xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
    half_extents = np.asarray(half_extents, dtype=np.float64).reshape(2)
    delta = target_xy - init_xy
    init_local = -0.5 * delta
    target_local = 0.5 * delta
    scale = 1.0
    for axis in range(2):
        need = max(abs(float(init_local[axis])), abs(float(target_local[axis])))
        limit = float(half_extents[axis])
        if need > limit and need > 1e-9:
            scale = min(scale, limit / need)
    return init_local * scale, target_local * scale


def map_poses_to_table(init_qpos, target_p, source_table_height,
                       table_pos=DYWA_TABLE_POS, table_dims=DYWA_TABLE_DIMS,
                       object_radius=0.08):
    """Preserve clearance and orientation when moving a pair between tables."""
    initial = np.asarray(init_qpos, dtype=np.float64).reshape(7).copy()
    target = np.asarray(target_p, dtype=np.float64).reshape(3).copy()
    table_pos = np.asarray(table_pos, dtype=np.float64).reshape(3)
    table_dims = np.asarray(table_dims, dtype=np.float64).reshape(3)
    limits = 0.5 * table_dims[:2] - float(object_radius)
    if np.any(limits <= 0):
        raise ValueError("DyWA object radius does not fit on the table")
    initial_xy, target_xy = place_pair_on_table(initial[:2], target[:2], limits)
    # Shrinking a task silently changes the benchmark. Reject pairs that do
    # not fit with the object's bounding radius instead.
    if not np.allclose(target_xy - initial_xy, target[:2] - initial[:2], atol=1e-9, rtol=0):
        raise ValueError("DyWA initial/goal translation does not fit on the table")
    initial[:2] = initial_xy + table_pos[:2]
    target[:2] = target_xy + table_pos[:2]
    height_offset = table_pos[2] + 0.5 * table_dims[2] - float(source_table_height)
    initial[2] += height_offset
    target[2] += height_offset
    return initial, target


def viewer_camera_for_scene(table_pos, table_dims, keepout_radius, base_height):
    """Place the MPC viewer on DyWA's Franka base."""
    table_pos = np.asarray(table_pos, dtype=np.float64).reshape(3)
    table_dims = np.asarray(table_dims, dtype=np.float64).reshape(3)
    if base_height == "table":
        base_z = float(table_pos[2] + 0.5 * table_dims[2])
    elif base_height == "ground":
        base_z = 0.0
    else:
        raise KeyError(f"Unknown base_height = {base_height}")
    base = np.array([
        float(table_pos[0] - 0.5 * table_dims[0] - float(keepout_radius)),
        float(table_pos[1]),
        base_z,
    ], dtype=np.float64)
    return base + _MPC_CAMERA_FROM_BASE, base + _MPC_CAMERA_TARGET_FROM_BASE


def look_at_rotation_and_translation(eye, at, up):
    """Match pytorch3d.renderer.cameras.look_at_view_transform for an explicit eye.

    DyWA unprojects depth with this rotation. The cross-product order and the
    final transpose are part of that convention.
    """
    def as_rows(value):
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, 3)
        return array.reshape(-1, 3)

    eye = as_rows(eye)
    at = as_rows(at)
    up = as_rows(up)
    count = max(eye.shape[0], at.shape[0], up.shape[0])
    eye = np.broadcast_to(eye, (count, 3)).copy()
    at = np.broadcast_to(at, (count, 3)).copy()
    up = np.broadcast_to(up, (count, 3)).copy()

    def normalize(vector):
        length = np.linalg.norm(vector, axis=-1, keepdims=True)
        return vector / np.maximum(length, 1e-5)

    z_axis = normalize(at - eye)
    x_axis = normalize(np.cross(up, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    degenerate = np.all(np.abs(x_axis) <= 5e-3, axis=-1, keepdims=True)
    if np.any(degenerate):
        x_axis = np.where(degenerate, normalize(np.cross(y_axis, z_axis)), x_axis)
    rotation = np.swapaxes(np.stack([x_axis, y_axis, z_axis], axis=1), 1, 2)
    translation = -np.einsum("nji,nj->ni", rotation, eye)
    return rotation, translation


def _as_list(values):
    return [float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


def validate_runtime_files(spec):
    """Report all absent source/weights before importing GPU extensions."""
    files = {
        "DyWA environment source": DYWA_SRC / "env/env/base.py",
        "DyWA student source": DYWA_TRAIN / "distill.py",
        "DyWA evaluation source": DYWA_TRAIN / "test_rma.py",
        "DyWA Hydra config": DYWA_CFG / "show.yaml",
    }
    for label, key in (("object mesh", "mesh_path"), ("student weights", "student_ckpt"),
                       ("teacher weights", "teacher_ckpt"), ("environment statistics", "env_ckpt"),
                       ("ICP encoder weights", "icp_ckpt")):
        files[label] = Path(spec[key])
    if "asset_root" in spec:
        files["official robot asset"] = Path(spec["asset_root"]) / "franka_description/robots/franka_panda.urdf"
    missing = [f"  {label}: {path}" for label, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "DyWA runtime files are missing:\n" + "\n".join(missing)
            + "\nRestore the official DyWA assets and weights."
            + " Use DYWA_ROOT and --dywa-ckpt-dir for alternate locations."
        )


def validate_runtime_dependencies():
    from importlib.util import find_spec

    missing = [name for name in ("isaacgym", "torch", "tensorboard", "hydra",
                                 "omegaconf", "pytorch3d", "einops") if find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"DyWA dependencies are missing from {sys.executable}: " + ", ".join(missing)
            + ". Install the original native DyWA dependencies; no numerical fallbacks are used."
        )


def _validate_required_config(cfg):
    """Legacy custom-source diagnostic; not used by the official runtime."""
    required = (
        "env.single_object_scene.mesh.mass",
        "env.single_object_scene.init_obj_qpos_",
        "env.task.target_p_",
        "env.task.target_q_",
        "env.franka.use_custom_end_effector",
    )
    missing = []
    for path in required:
        value = cfg
        for key in path.split("."):
            if not hasattr(value, key):
                missing.append(path)
                break
            value = getattr(value, key)
    if missing:
        raise RuntimeError(
            "This DyWA source lacks the SCSP integration hooks: " + ", ".join(missing)
            + ". Restore the custom DyWA version; unmodified upstream is not compatible."
        )


def build_experiment(args, trial_count):
    """Sample the same trial as the MPC Isaac entry and pack a DyWA spec."""
    os.environ.setdefault("SCSP_PLANNER_ONLY", "1")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from examples.mpc.franka.runtime.params import ExplicitMPCParams
    from examples.mpc.franka.runtime.mpc import _apply_dywa_physics_to_param

    param = ExplicitMPCParams(
        args,
        rand_seed=int(trial_count),
        target_type=getattr(args, "target_type", "ground-rotation"),
        mpc_model="explicit",
        build_optimizer=False,
    )
    param = _apply_dywa_physics_to_param(param)
    init_qpos = np.asarray(param.init_obj_qpos_, dtype=np.float64).reshape(7)
    target_p = np.asarray(param.target_p_, dtype=np.float64).reshape(3)
    target_q = np.asarray(param.target_q_, dtype=np.float64).reshape(4)
    mapped_init, mapped_target = map_poses_to_table(
        init_qpos, target_p, param.table_height,
        object_radius=param.object_circumradius,
    )
    ckpt_dir = Path(getattr(args, "dywa_ckpt_dir", DEFAULT_CKPT_DIR)).expanduser()
    mesh_path = (REPO_ROOT / param.mesh_path_).resolve()
    if not mesh_path.is_file():
        mesh_path = Path(param.mesh_path_).expanduser().resolve()
    # The student outputs a Cartesian delta of at most 6 cm. Training holds
    # that delta for dt * action_period = 0.0125 * 8 = 0.1 s so the joint PD
    # can reach it. A 20 ms hold leaves the tool short of every command.
    action_window_s = POLICY_INTERVAL
    return {
        "trial": int(trial_count),
        "object": str(getattr(args, "obj", "foam_brick")),
        "mesh_path": str(mesh_path),
        "collision_mesh_paths": [str(Path(p).resolve()) for p in param.collision_mesh_paths_],
        "init_obj_qpos": _as_list(mapped_init),
        "target_p": _as_list(mapped_target),
        "target_q": _as_list(target_q),
        "source_init_obj_qpos": _as_list(init_qpos),
        "source_target_p": _as_list(target_p),
        "source_target_q": _as_list(target_q),
        "source_table_height": float(param.table_height),
        "table_pos": _as_list(DYWA_TABLE_POS),
        "table_dims": _as_list(DYWA_TABLE_DIMS),
        "mass": float(param.sim_obj_mass_),
        "table_friction": float(param.table_friction_),
        "object_friction": float(param.object_friction_),
        "dt": 0.0125,
        "substeps": 1,
        "action_period": 8,
        "policy_interval_s": float(action_window_s),
        "max_steps": max(1, int(args.max_rollout_length)),
        "success_pos": 0.02,
        "success_quat": 0.015,
        "metric_gamma": float(getattr(args, "ranking_metric_gamma", 0.9)),
        "metric_eta": float(getattr(args, "ranking_metric_eta", 1.0)),
        "headless": bool(getattr(args, "headless", False)),
        "device": str(getattr(args, "sim_device", "cuda:0")),
        "graphics_device_id": int(getattr(args, "graphics_device_id", 0)),
        "seed": int(100 + trial_count),
        "asset_root": str(DYWA_ROOT / "dywa/src/data/assets"),
        "ckpt_dir": str(ckpt_dir),
        "student_ckpt": str(ckpt_dir / "last.ckpt"),
        "teacher_ckpt": str(ckpt_dir / "teacher-last.ckpt"),
        "env_ckpt": str(ckpt_dir / "env-last.ckpt"),
        "icp_ckpt": str(ckpt_dir / "512-32-balanced-SAM-wd-5e-05-920.ckpt"),
        "output_dir": str(REPO_ROOT / "outputs" / "dywa_isaac" / str(args.obj) / f"trial_{int(trial_count):03d}"),
        "use_custom_fingertip": True,
    }


def _hydra_overrides(spec):
    device = spec["device"]
    viewer = "false" if spec["headless"] else "true"
    return [
        "+platform=debug",
        "+env=abs_goal_1view",
        "+run=teacher_base",
        "+student=dywa/base",
        "++name=dywa",
        f"++path.root={spec['output_dir']}",
        "++env.num_env=1",
        f"++global_device={device}",
        "++student.norm=ln",
        f"++load_ckpt={spec['teacher_ckpt']}",
        f"++load_environment={spec['env_ckpt']}",
        f"+load_student={spec['student_ckpt']}",
        "++plot_pc=false",
        "++use_nvdr_record_episode=false",
        "++use_nvdr_record_viewer=false",
        "++add_teacher_state=true",
        "++student.decoder.film_mlp=true",
        # Upstream's 'student' branch expects actor_net, which StudentAgentRMA
        # does not expose. Its original anneal branch with alpha fixed to one
        # selects the student output on every step, including the first.
        "++dagger_train_env.agent_type=anneal",
        "++dagger_train_env.alpha0=1.0",
        "++dagger_train_env.alpha1=1.0",
        "++dagger_train_env.deterministic_action=false",
        "++dagger_train_env.anneal_step=1",
        "++log_categorical_results=false",
        f"++test_step={int(spec['max_steps'])}",
        f"++env.use_viewer={viewer}",
        f"++env.seed={int(spec['seed'])}",
        "++env.single_object_scene.base_set=[mesh]",
        "++env.single_object_scene.table_pos=" + json.dumps(spec["table_pos"]),
        "++env.single_object_scene.table_dims=" + json.dumps(spec["table_dims"]),
        f"++env.single_object_scene.mesh.filename={spec['mesh_path']}",
        f"++env.single_object_scene.target_mass={spec['mass']}",
        "++env.single_object_scene.mesh.num_poses=8",
        "++env.single_object_scene.mesh.num_points=512",
        f"++env.single_object_scene.mesh.cache_dir={spec['output_dir']}/pose_cache",
        "++env.single_object_scene.filter_file=null",
        "++env.single_object_scene.filter_complex=false",
        "++env.single_object_scene.num_object_types=1",
        "++env.single_object_scene.use_wall=false",
        "++env.single_object_scene.use_dr=false",
        "++env.single_object_scene.use_dr_on_setup=false",
        "++env.single_object_scene.use_scale_dr=false",
        "++env.single_object_scene.mode=valid",
        f"++env.single_object_scene.table_friction={spec['table_friction']}",
        f"++env.single_object_scene.object_friction={spec['object_friction']}",
        f"++icp_obs.icp.ckpt={spec['icp_ckpt']}",
        "++use_wandb=false",
    ]


def _iterative_closest_point(*args, **kwargs):
    """Compatibility diagnostic; the runtime uses native PyTorch3D exclusively."""
    from planning.dywa.legacy_ops import _iterative_closest_point as diagnostic
    return diagnostic(*args, **kwargs)


def _activate_dywa_import_path():
    import types
    # Isaac Gym's unchanged torch_utils signature references the removed alias.
    # np.float was exactly Python float; this restores import compatibility.
    if "float" not in np.__dict__:
        np.float = float
    for path in (str(DYWA_ROOT / "native"), str(DYWA_TRAIN), str(DYWA_SRC)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    roots = (DYWA_SRC.resolve(), DYWA_TRAIN.resolve())
    conflicting = {"models", "envs", "env", "data", "util", "train", "cxx"}
    for name, module in list(sys.modules.items()):
        if name.split(".", 1)[0] not in conflicting:
            continue
        file_name = getattr(module, "__file__", None)
        if not file_name:
            del sys.modules[name]
            continue
        resolved = Path(file_name).resolve()
        if any(_is_relative_to(resolved, root) for root in roots):
            continue
        del sys.modules[name]
    # Upstream exp/train/envs is a namespace directory without __init__.py.
    # Bind it explicitly so the repository's regular envs package cannot win.
    package = types.ModuleType("envs")
    package.__path__ = [str(DYWA_TRAIN / "envs")]
    package.__package__ = "envs"
    sys.modules["envs"] = package


def _is_relative_to(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _xyzw_to_wxyz(quat_xyzw):
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def _object_and_goal(env):
    root = env.tensors["root"]
    actor = int(env.scene.cur_ids.reshape(-1)[0].item())
    pose = root[actor, :7].detach().cpu().numpy()
    goal = env.task.goal.reshape(-1, env.task.goal.shape[-1])[0, :7].detach().cpu().numpy()
    return pose, goal


def validate_reset_state(pose_xyzw, goal_xyzw, spec):
    """Check that the custom reset hooks actually applied the mapped WXYZ poses."""
    def check(label, actual, position, quaternion, tolerance):
        actual = np.asarray(actual, dtype=np.float64).reshape(7)
        expected_q = np.asarray(quaternion, dtype=np.float64).reshape(4)[[1, 2, 3, 0]]
        pos_error = float(np.linalg.norm(actual[:3] - position))
        norm_product = np.linalg.norm(actual[3:]) * np.linalg.norm(expected_q)
        alignment = abs(float(np.dot(actual[3:], expected_q))) / max(float(norm_product), 1e-12)
        if not np.all(np.isfinite(actual)) or pos_error > tolerance or alignment < 0.995:
            raise RuntimeError(
                f"DyWA {label} was not applied correctly after reset: "
                f"position error={pos_error:.6f} m, quaternion alignment={alignment:.6f}. "
                "Check the custom pose hooks, WXYZ-to-XYZW conversion and table height."
            )

    initial = np.asarray(spec["init_obj_qpos"], dtype=np.float64)
    # Allow the 2 mm initial clearance to settle, but catch the old 5 cm error.
    check("initial pose", pose_xyzw, initial[:3], initial[3:], 0.01)
    check("goal pose", goal_xyzw, np.asarray(spec["target_p"]), spec["target_q"], 1e-5)


def _compose_config(spec):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.types import ConvertMode
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(DYWA_CFG), version_base=None):
        cli_cfg = compose(config_name="show", overrides=_hydra_overrides(spec))
    from test_rma import Config

    default_config = Config()
    default_cfg = OmegaConf.structured(default_config)
    cli_cfg = instantiate(cli_cfg, _convert_=ConvertMode.PARTIAL)
    cfg = OmegaConf.to_object(OmegaConf.structured(OmegaConf.merge(default_cfg, cli_cfg)))
    gpu = int(str(spec["device"]).split(":")[-1])
    cfg.env.use_viewer = not bool(spec["headless"])
    scene = cfg.env.single_object_scene
    camera_origin, camera_target = viewer_camera_for_scene(
        scene.table_pos,
        scene.table_dims,
        cfg.env.franka.keepout_radius,
        cfg.env.franka.base_height,
    )
    cfg.env.viewer_camera_origin = tuple(float(v) for v in camera_origin)
    cfg.env.viewer_camera_target = tuple(float(v) for v in camera_target)
    cfg.env.graphics_device_id = int(spec.get("graphics_device_id", gpu))
    cfg.env.compute_device_id = gpu
    cfg.sync_frame_time = False
    return cfg


def _make_evaluation_env(wrapper_type, cfg, env, student):
    """Reuse student stepping while avoiding DAgger's random mixed-reset rollout."""
    class EvaluationEnv(wrapper_type):
        def reset(self):
            import torch

            obs = self.env.reset()
            self.student_module.reset(obs)
            self.student_module.reset_state(
                torch.zeros(self.num_env, dtype=torch.bool, device=self.device)
            )
            self.last_obs = obs
            return obs

    return EvaluationEnv(cfg, env, student)


def run_spec(spec):
    """Load the student and roll the matched trial. Isaac Gym must already be imported."""
    validate_runtime_files(spec)
    validate_runtime_dependencies()
    import torch as th
    from torch.utils.tensorboard import SummaryWriter

    os.environ.setdefault("PYTORCH_JIT", "0")
    _activate_dywa_import_path()
    from env.env.base import EnvBase
    from test_rma import update_net_cfg
    from train_ppo_arm import AddTensorboardWriter, load_agent, load_env, setup as setup_logging
    from train_rma import DAggerTrainerEnv, setup_rma_env_v2
    from distill import StudentAgentRMA
    from dataclasses import replace

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from utils import metrics
    from planning.dywa.audit import fingerprint_files, policy_digest, verify_upstream, load_student
    from utils.ranking_metrics import discounted_terminal_reward

    from planning.dywa.inputs import install_task_inputs, configure_inputs
    install_task_inputs(spec)
    cfg = configure_inputs(_compose_config(spec), spec)
    audit = {"upstream": verify_upstream(), "files_before": fingerprint_files(spec)}
    audit["control"] = dict(dt=cfg.env.dt, action_period=cfg.env.action_period,
                            mode=cfg.env.franka.ctrl_mode,
                            max_pos=cfg.env.franka.max_pos, max_ori=cfg.env.franka.max_ori)
    if not np.isclose(cfg.env.dt * cfg.env.action_period, spec["policy_interval_s"]):
        raise RuntimeError("Native DyWA action timing differs from the experiment specification")
    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    path = setup_logging(cfg)
    writer = SummaryWriter(path.tb_train)
    cfg, env = load_env(cfg, path, freeze_env=True, check_viewer=False)
    env.unwrap(target=AddTensorboardWriter).set_writer(writer)
    cfg = replace(cfg, net=update_net_cfg(cfg.net, env, cfg.state_net_blocklist))
    teacher_agent = load_agent(cfg, env, None, None)
    teacher_agent.eval()
    student = StudentAgentRMA(
        replace(cfg.student, batch_size=env.num_env),
        writer=writer,
        device=env.device,
    ).to(env.device)
    load_student(student, spec["student_ckpt"])
    student.eval()
    audit["policy_before"] = policy_digest(student)
    env = setup_rma_env_v2(
        cfg,
        env,
        teacher_agent,
        state_size=128,
        is_student=False,
        dagger=True,
    )
    env.load(Path(spec["env_ckpt"]), strict=True)
    env = _make_evaluation_env(DAggerTrainerEnv, cfg.dagger_train_env, env, student)

    success_pos = float(spec["success_pos"])
    success_quat = float(spec["success_quat"])
    max_steps = int(spec["max_steps"])
    min_pos = float("inf")
    min_quat = float("inf")
    pos_err = quat_err = None
    trial_success = False
    steps = 0
    choose_times = []
    base = env.unwrap(target=EnvBase)
    trace = Path(spec["output_dir"], "trajectory.csv").open("w", newline="")
    trace_writer = csv.writer(trace)
    trace_writer.writerow(["step", "sim_time_s", "x", "y", "z", "qx", "qy", "qz", "qw",
                           "pos_error_m", "quat_error", "done"])
    try:
        from env.util import set_seed
        set_seed(int(spec["seed"]))
        env.reset()
        initial_pose, actual_goal = _object_and_goal(base)
        validate_reset_state(initial_pose, actual_goal, spec)
        for _step in range(max_steps):
            started = time.perf_counter()
            with th.no_grad():
                _obs, _rew, done, info = env.step()
            choose_times.append(time.perf_counter() - started)
            pose, goal = _object_and_goal(base)
            pos_err = float(metrics.comp_pos_error(pose[:3], goal[:3]))
            quat_err = float(metrics.comp_quat_error(
                _xyzw_to_wxyz(pose[3:7]), _xyzw_to_wxyz(goal[3:7])))
            min_pos = min(min_pos, pos_err)
            min_quat = min(min_quat, quat_err)
            steps = _step + 1
            trace_writer.writerow([steps, steps * spec["policy_interval_s"], *pose.tolist(),
                                   pos_err, quat_err, bool(done.reshape(-1)[0].item())])
            trace.flush()
            internal_success = bool(info.get("success", th.zeros(1))[0].item()) if info else False
            print(
                "dywa_step:", steps,
                "pos_err:", round(pos_err, 5),
                "quat_err:", round(quat_err, 5),
                "internal_success:", int(internal_success),
                flush=True,
            )
            if pos_err <= success_pos and quat_err <= success_quat:
                trial_success = True
                break
            if done is not None and bool(done.reshape(-1)[0].item()) and not trial_success:
                break
    finally:
        trace.close()
        close = getattr(env, "close", None)
        if close is not None:
            close()
        writer.close()

    audit["policy_after"] = policy_digest(student)
    audit["files_after"] = fingerprint_files(spec)
    audit["policy_unchanged"] = audit["policy_before"] == audit["policy_after"]
    audit["files_unchanged"] = audit["files_before"] == audit["files_after"]
    Path(spec["output_dir"], "audit.json").write_text(json.dumps(audit, indent=2))
    if not audit["policy_unchanged"] or not audit["files_unchanged"]:
        raise RuntimeError("DyWA source or weights changed during evaluation")
    choose = np.asarray(choose_times, dtype=np.float64) if choose_times else np.array([0.0])
    summary = {
        "trial": int(spec["trial"]),
        "policy_unchanged": audit["policy_unchanged"],
        "files_unchanged": audit["files_unchanged"],
        "mode": "dywa",
        "object": spec["object"],
        "success": int(trial_success),
        "steps": int(steps),
        "completion_steps": int(steps) if trial_success else None,
        "terminal_reward": discounted_terminal_reward(
            trial_success, steps if trial_success else None,
            gamma=float(spec["metric_gamma"]), eta=float(spec["metric_eta"]),
        ),
        "metric_gamma": float(spec["metric_gamma"]),
        "metric_eta": float(spec["metric_eta"]),
        "policy_interval_s": float(spec["policy_interval_s"]),
        "final_pos_err": None if pos_err is None else round(float(pos_err), 5),
        "final_quat_err": None if quat_err is None else round(float(quat_err), 5),
        "min_pos_err": None if not np.isfinite(min_pos) else round(float(min_pos), 5),
        "min_quat_err": None if not np.isfinite(min_quat) else round(float(min_quat), 5),
        "choose_dt_mean": round(float(np.mean(choose)), 5),
        "choose_dt_max": round(float(np.max(choose)), 5),
        "mass": float(spec["mass"]),
        "table_friction": float(spec["table_friction"]),
        "object_friction": float(spec["object_friction"]),
        "dt": float(spec["dt"]),
        "action_period": int(spec["action_period"]),
    }
    result_path = spec.get("result_json")
    if result_path:
        Path(result_path).write_text(json.dumps(summary), encoding="utf-8")
    print("trial_summary: " + repr(summary), flush=True)
    return summary


def execute_trial(spec):
    """Run one trial in a fresh acc interpreter so Isaac Gym loads before torch."""
    validate_runtime_files(spec)
    python = ACC_PYTHON if ACC_PYTHON.is_file() else Path(sys.executable)
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "spec.json"
    result_path = output_dir / "summary.json"
    payload = dict(spec)
    payload["result_json"] = str(result_path)
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    if result_path.exists():
        result_path.unlink()
    child_env = runtime_environment(python)
    with (output_dir / "runtime.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [str(python), "-u", str(Path(__file__).resolve()), "--spec", str(spec_path)],
            cwd=str(REPO_ROOT), env=child_env, check=False,
            stdout=log, stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        lines = (output_dir / "runtime.log").read_text(errors="replace").splitlines()
        error = next((line for line in reversed(lines) if line.startswith(
            ("RuntimeError:", "ImportError:", "ModuleNotFoundError:", "ValueError:", "NameError:"))),
            next((line for line in reversed(lines) if line.strip()), ""))
        raise RuntimeError(
            f"DyWA trial {spec['trial']} exited with status {completed.returncode}. "
            f"{error} See {output_dir / 'runtime.log'}."
        )
    if not result_path.is_file():
        raise RuntimeError(f"DyWA trial {spec['trial']} did not write {result_path}.")
    return json.loads(result_path.read_text(encoding="utf-8"))


def main():
    import argparse
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    parser = argparse.ArgumentParser(description="Run one DyWA spec inside Isaac Gym.")
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    validate_runtime_files(spec)
    ensure_acc_runtime()
    from isaacgym import gymapi, gymtorch  # noqa: F401
    import torch
    from planning.dywa.audit import fingerprint_files, verify_upstream

    preflight = dict(upstream=verify_upstream(), checkpoints=fingerprint_files(spec),
                     python=sys.version, torch=torch.__version__, device=spec["device"])
    try:
        # Fail before creating a simulator if this PyTorch build cannot execute
        # native kernels on the selected GPU. No CPU or numerical substitute.
        torch.empty(8, device=spec["device"]).uniform_()
        if str(spec["device"]).startswith("cuda"):
            torch.cuda.synchronize()
            preflight["gpu"] = torch.cuda.get_device_name()
            preflight["capability"] = torch.cuda.get_device_capability()
        preflight["device_ready"] = True
    except RuntimeError as exc:
        preflight["device_ready"] = False
        preflight["device_error"] = str(exc)
        raise
    finally:
        Path(spec["output_dir"], "preflight.json").write_text(json.dumps(preflight, indent=2))
    validate_runtime_dependencies()

    run_spec(spec)


if __name__ == "__main__":
    main()
