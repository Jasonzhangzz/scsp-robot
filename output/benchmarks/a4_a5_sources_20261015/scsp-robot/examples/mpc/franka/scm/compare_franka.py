"""Compare ComFree and MuJoCo Warp on the upstream Panda grasp task.

The IK targets, phase schedule, and perturbations are adapted from
https://github.com/asu-iris/comfree_warp/blob/ba8b9969f5f7859d51b1f738f458f8d2f49be080/test_local/test_franka_grasp.py
by ASU IRIS. Local changes add matched backend runs, headless execution,
trajectory/contact logging, and a viewer enabled by default. This is a migration of
the public grasp example, not a reproduction of the paper's fidelity sweeps.
"""

from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .runtime_env import activate_dependencies
else:
    from runtime_env import activate_dependencies

activate_dependencies()

import mujoco
import numpy as np

from benchmark_utils import (
    Backend,
    add_common_arguments,
    contact_metrics,
    engines_for,
    run_metadata,
    save_results,
)


MODEL_PATH = Path(__file__).resolve().parent / "assets" / "franka_cube_grasp" / "scene.xml"
ARM_DOF = 7
OPEN_GRIPPER = 0.04
CLOSED_GRIPPER = 0.014
LIFTED_THRESHOLD = 0.18
PEDESTAL_TOP = 0.12


@dataclass(frozen=True)
class Phase:
    name: str
    steps: int
    arm_target: np.ndarray
    finger_target: float
    force_scale: float = 0.0


def load_keyframe_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if model.nkey:
        key = model.key(0)
        data.qpos[:] = key.qpos
        data.qvel[:] = 0.0
        if model.nu:
            data.ctrl[:] = key.ctrl
    mujoco.mj_forward(model, data)


def solve_position_ik(
    model: mujoco.MjModel,
    seed_qpos: np.ndarray,
    target_pos: np.ndarray,
    cube_qpos: np.ndarray,
    *,
    site_name: str = "gripper",
    max_steps: int = 250,
    step_scale: float = 0.7,
    damping: float = 1e-4,
    tol: float = 1e-5,
) -> np.ndarray:
    """Upstream damped least-squares position IK for the seven arm joints."""
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Model has no site named {site_name!r}.")
    lower = model.jnt_range[:ARM_DOF, 0]
    upper = model.jnt_range[:ARM_DOF, 1]
    jacp = np.zeros((3, model.nv))
    qpos = seed_qpos.astype(np.float64).copy()

    for _ in range(max_steps):
        data.qpos[:ARM_DOF] = qpos
        data.qpos[ARM_DOF:ARM_DOF + 2] = OPEN_GRIPPER
        data.qpos[ARM_DOF + 2:] = cube_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        err = target_pos - data.site_xpos[site_id]
        if np.linalg.norm(err) < tol:
            break
        jacp.fill(0.0)
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        arm_jac = jacp[:, :ARM_DOF]
        lhs = arm_jac @ arm_jac.T + damping * np.eye(3)
        dq = arm_jac.T @ np.linalg.solve(lhs, err)
        qpos = np.clip(qpos + step_scale * dq, lower, upper)
    return qpos


def build_phase_targets(model: mujoco.MjModel, data: mujoco.MjData) -> list[Phase]:
    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    cube_qpos_adr = model.jnt_qposadr[cube_joint_id]
    cube_qpos = data.qpos[cube_qpos_adr:cube_qpos_adr + 7].copy()
    center = cube_qpos[:3]
    seed = data.ctrl[:ARM_DOF].copy()
    approach_q = solve_position_ik(model, seed, center + np.array([0.0, 0.0, 0.14]), cube_qpos)
    pregrasp_q = solve_position_ik(model, approach_q, center + np.array([0.0, 0.0, 0.07]), cube_qpos)
    grasp_q = solve_position_ik(model, pregrasp_q, center + np.array([0.0, 0.0, 0.03]), cube_qpos)
    lift_q = solve_position_ik(model, grasp_q, center + np.array([0.0, 0.0, 0.18]), cube_qpos)
    return [
        Phase("approach", 320, approach_q, OPEN_GRIPPER),
        Phase("pregrasp", 260, pregrasp_q, OPEN_GRIPPER),
        Phase("descend", 360, grasp_q, OPEN_GRIPPER),
        Phase("settle", 220, grasp_q, OPEN_GRIPPER),
        Phase("squeeze", 460, grasp_q, CLOSED_GRIPPER),
        Phase("lift", 420, lift_q, CLOSED_GRIPPER),
        Phase("hold", 560, lift_q, CLOSED_GRIPPER),
        Phase("perturb", 720, lift_q, CLOSED_GRIPPER, force_scale=1.5),
    ]


def build_inputs(
    model: mujoco.MjModel, initial_data: mujoco.MjData, phases: list[Phase], steps_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Build one deterministic input sequence shared by every backend."""
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    controls, forces, phase_indices, schedule = [], [], [], []
    previous_arm = initial_data.ctrl[:ARM_DOF].copy()
    previous_finger = float(initial_data.ctrl[ARM_DOF])
    for phase_index, phase in enumerate(phases):
        steps = max(1, int(round(phase.steps * steps_scale)))
        schedule.append({"name": phase.name, "steps": steps})
        for local_step in range(steps):
            alpha = (local_step + 1) / steps
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            ctrl = initial_data.ctrl.copy()
            ctrl[:ARM_DOF] = (1.0 - alpha) * previous_arm + alpha * phase.arm_target
            ctrl[ARM_DOF] = (1.0 - alpha) * previous_finger + alpha * phase.finger_target
            force = np.zeros_like(initial_data.xfrc_applied)
            if phase.force_scale:
                elapsed = local_step * model.opt.timestep
                force[hand_id, :3] = [
                    phase.force_scale * math.sin(14.0 * elapsed),
                    0.75 * phase.force_scale * math.cos(9.0 * elapsed),
                    0.0,
                ]
            controls.append(ctrl)
            forces.append(force)
            phase_indices.append(phase_index)
        previous_arm = phase.arm_target
        previous_finger = phase.finger_target
    return np.asarray(controls), np.asarray(forces), np.asarray(phase_indices), schedule


def run_trial(
    args: argparse.Namespace,
    engine: str,
    model: mujoco.MjModel,
    controls: np.ndarray,
    forces: np.ndarray,
    phase_indices: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    data = mujoco.MjData(model)
    load_keyframe_state(model, data)
    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    cube_address = int(model.jnt_qposadr[cube_joint_id])
    backend = Backend(
        engine, model, data, device=args.device,
        stiffness=args.contact_stiffness, damping=args.contact_damping,
        nconmax=args.nconmax, njmax=args.njmax,
    )
    records = {name: [] for name in (
        "time", "qpos", "qvel", "ctrl", "cube_position", "ncontact",
        "max_penetration_m", "phase_index",
    )}

    def record(phase_index: int) -> None:
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                and np.isfinite(data.ctrl).all() and np.isfinite(data.time)):
            raise RuntimeError(f"{engine}: nonfinite state at time {data.time}.")
        geometry = contact_metrics(model, data)
        records["time"].append(float(data.time))
        records["qpos"].append(data.qpos.copy())
        records["qvel"].append(data.qvel.copy())
        records["ctrl"].append(data.ctrl.copy())
        records["cube_position"].append(data.qpos[cube_address:cube_address + 3].copy())
        records["ncontact"].append(int(geometry["ncontact"]))
        records["max_penetration_m"].append(float(geometry["max_penetration_m"]))
        records["phase_index"].append(phase_index)

    record(-1)
    requested_steps = len(controls) if args.max_steps is None else min(args.max_steps, len(controls))
    viewer_context = nullcontext(None)
    if not args.headless:
        from mujoco import viewer as mujoco_viewer
        viewer_context = mujoco_viewer.launch_passive(model, data)
    with viewer_context as viewer:
        for step_index in range(requested_steps):
            if viewer is not None and not viewer.is_running():
                break
            data.ctrl[:] = controls[step_index]
            data.xfrc_applied[:] = forces[step_index]
            started = time.perf_counter()
            backend.step(data)
            record(int(phase_indices[step_index]))
            if viewer is not None:
                viewer.sync()
                delay = float(model.opt.timestep) - (time.perf_counter() - started)
                if delay > 0.0:
                    time.sleep(delay)

    trajectory = {name: np.asarray(values) for name, values in records.items()}
    heights = trajectory["cube_position"][:, 2]
    steps = len(heights) - 1
    truncated = steps < len(controls)
    success = None if truncated else bool(
        heights[-1] > LIFTED_THRESHOLD and heights.min() > PEDESTAL_TOP - 0.005
    )
    summary = {
        "engine": engine, "steps": steps, "planned_steps": len(controls),
        "truncated": truncated, "success": success,
        "final_cube_height_m": float(heights[-1]),
        "min_cube_height_m": float(heights.min()),
        "max_cube_height_m": float(heights.max()),
        "max_penetration_m": float(trajectory["max_penetration_m"].max()),
        "mean_ncontact": float(trajectory["ncontact"].mean()),
    }
    return trajectory, summary


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--steps-scale", type=positive_float, default=1.0,
                        help="Scale each original phase length; shorter runs change task timing.")
    parser.add_argument("--max-steps", type=positive_int,
                        help="Truncate for a smoke check; a truncated run has no success verdict.")
    parser.add_argument("--require-success", action="store_true",
                        help="Exit unsuccessfully if a complete grasp trial fails.")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    initial_data = mujoco.MjData(model)
    load_keyframe_state(model, initial_data)
    controls, forces, phase_indices, schedule = build_inputs(
        model, initial_data, build_phase_targets(model, initial_data), args.steps_scale,
    )
    trajectories, summaries = {}, {}
    for engine in engines_for(args.engine):
        print(f"Running {engine}: dt={model.opt.timestep:g}, planned steps={len(controls)}", flush=True)
        trajectory, summary = run_trial(args, engine, model, controls, forces, phase_indices)
        trajectories[engine] = trajectory
        summaries[engine] = summary
        verdict = "unassessed (truncated)" if summary["truncated"] else str(summary["success"])
        print(f"{engine}: steps={summary['steps']}, final cube height="
              f"{summary['final_cube_height_m']:.4f} m, success={verdict}", flush=True)
    metadata = run_metadata(args, MODEL_PATH, model)
    metadata.update({
        "benchmark": "upstream_franka_cube_grasp",
        "phase_schedule": schedule,
        "phase_index_initial": -1,
        "success_definition": "complete trial; final cube center z > 0.18 m and min z > 0.115 m",
        "upstream_source": "https://github.com/asu-iris/comfree_warp/blob/"
                           "ba8b9969f5f7859d51b1f738f458f8d2f49be080/test_local/test_franka_grasp.py",
    })
    save_results(args.output_dir, trajectories, summaries, metadata)
    if args.require_success and any(summary["success"] is False for summary in summaries.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
