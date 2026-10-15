#!/usr/bin/env python3
"""Compare ComFree, MJWarp and diagonal SCM on rigid-body physics scenes.

The scene assets and backend selection are adapted from asu-iris/comfree_warp
at ba8b9969f5f7859d51b1f738f458f8d2f49be080, particularly
test_local/test_viewer.py.  Logging and quantitative metrics are local additions;
this is not the unpublished implementation of the paper's experiments.

Each backend starts from the same state and runs an independent trajectory.
Trajectory divergence is therefore different from shared-state one-step error.
Use compare_prediction.py for errors by separation/sliding/sticking input mode.
SCM uses the explicitly selected dynamic/increment and point/6D extension.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from contextlib import nullcontext
import math
from pathlib import Path
import time

if __package__:
    from .runtime_env import activate_dependencies
else:
    from runtime_env import activate_dependencies

activate_dependencies()

import mujoco
import numpy as np

if __package__:
    from .fidelity_metrics import Moments, contact_observations
    from .compare_prediction import CONTACT_FIELDS, write_contacts
else:
    from fidelity_metrics import Moments, contact_observations
    from compare_prediction import CONTACT_FIELDS, write_contacts

if __package__:
    from .benchmark_utils import (
        Backend,
        add_common_arguments,
        contact_metrics,
        engines_for,
        run_metadata,
        save_results,
    )
else:
    from benchmark_utils import (
        Backend,
        add_common_arguments,
        contact_metrics,
        engines_for,
        run_metadata,
        save_results,
    )


SCENES = ("primitives", "ball_rotation", "cylinder_rolling", "sliding")
ASSET_DIRECTORY = Path(__file__).resolve().parent / "assets" / "physics"


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def positive_float(value: str) -> float:
    parsed = finite_float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def initial_state(
    model: mujoco.MjModel, scene: str, initial_speed: float
) -> mujoco.MjData:
    """Preserve the upstream poses; initialize the friction scenes explicitly."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if scene == "ball_rotation":
        joint = model.joint("ball_joint")
        data.qvel[int(joint.dofadr[0])] = initial_speed
    elif scene == "cylinder_rolling":
        joint = model.joint("cylinder_free")
        dof_address = int(joint.dofadr[0])
        body_id = int(joint.bodyid[0])
        geom_id = int(model.body_geomadr[body_id])
        radius = float(model.geom_size[geom_id, 0])
        data.qvel[dof_address] = initial_speed
        # For rolling in world +x, omega_world = (0, speed / radius, 0).
        # MuJoCo stores free-joint angular qvel in the local body frame.  The
        # upstream cylinder is rotated +90 degrees about x, so this maps to
        # local -z, rather than the fifth qvel entry (local y).
        rotation = data.xmat[body_id].reshape(3, 3)
        angular_velocity_world = np.array([0.0, initial_speed / radius, 0.0])
        data.qvel[dof_address + 3 : dof_address + 6] = (
            rotation.T @ angular_velocity_world
        )
    elif scene == "sliding":
        data.qvel[:] = [2, 0, 0, .1, .1, .1]
    mujoco.mj_forward(model, data)
    return data


def sample_state(
    model: mujoco.MjModel, data: mujoco.MjData, body_ids: np.ndarray
) -> dict[str, np.ndarray | float | int]:
    metrics = contact_metrics(model, data)
    linear_speed = np.empty(len(body_ids))
    angular_speed = np.empty(len(body_ids))
    horizontal_speed = np.empty(len(body_ids))
    spatial_velocity = np.empty(6)
    for index, body_id in enumerate(body_ids):
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), spatial_velocity, 0
        )
        angular_speed[index] = np.linalg.norm(spatial_velocity[:3])
        linear_speed[index] = np.linalg.norm(spatial_velocity[3:])
        horizontal_speed[index] = np.linalg.norm(spatial_velocity[3:5])
    return {
        "time": float(data.time),
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "ncontact": int(metrics["ncontact"]),
        "max_penetration_m": float(metrics["max_penetration_m"]),
        "kinetic_energy_j": float(data.energy[1]),
        "body_position_m": data.xpos[body_ids].copy(),
        "body_quaternion": data.xquat[body_ids].copy(),
        "body_linear_speed_m_s": linear_speed,
        "body_angular_speed_rad_s": angular_speed,
        "body_horizontal_speed_m_s": horizontal_speed,
    }


def summarize(trajectory: dict[str, np.ndarray], expected_steps: int) -> dict:
    finite = all(np.isfinite(value).all() for value in trajectory.values())
    penetration = trajectory["max_penetration_m"]
    kinetic_energy = trajectory["kinetic_energy_j"]
    generalized_velocity_norm = np.linalg.norm(trajectory["qvel"], axis=1)
    initial_energy = float(kinetic_energy[0])
    final_energy = float(kinetic_energy[-1])
    contact_frames = trajectory["ncontact"] > 0
    return {
        "completed_steps": len(trajectory["time"]) - 1,
        "requested_steps": expected_steps,
        "finite": bool(finite),
        "max_penetration_m": float(np.max(penetration)),
        "mean_step_max_penetration_m": float(np.mean(penetration)),
        "mean_contact_step_max_penetration_m": (
            float(np.mean(penetration[contact_frames]))
            if np.any(contact_frames)
            else None
        ),
        "max_ncontact": int(np.max(trajectory["ncontact"])),
        "contact_frames": int(np.count_nonzero(contact_frames)),
        # Generalized velocity combines translational and angular components;
        # body speed metrics below retain their physical units separately.
        "final_generalized_velocity_norm": float(generalized_velocity_norm[-1]),
        "max_generalized_velocity_norm": float(np.max(generalized_velocity_norm)),
        "final_max_body_linear_speed_m_s": float(
            np.max(trajectory["body_linear_speed_m_s"][-1], initial=0.0)
        ),
        "max_body_linear_speed_m_s": float(
            np.max(trajectory["body_linear_speed_m_s"], initial=0.0)
        ),
        "final_max_body_angular_speed_rad_s": float(
            np.max(trajectory["body_angular_speed_rad_s"][-1], initial=0.0)
        ),
        "max_body_angular_speed_rad_s": float(
            np.max(trajectory["body_angular_speed_rad_s"], initial=0.0)
        ),
        "initial_kinetic_energy_j": initial_energy,
        "final_kinetic_energy_j": final_energy,
        "max_kinetic_energy_j": float(np.max(kinetic_energy)),
        "kinetic_energy_decay_ratio": (
            final_energy / initial_energy if initial_energy > 1.0e-12 else None
        ),
    }


def simulate(
    engine: str,
    model: mujoco.MjModel,
    initial_data: mujoco.MjData,
    body_ids: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict]:
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, initial_data)
    backend = Backend(
        engine,
        model,
        data,
        device=args.device,
        stiffness=args.contact_stiffness,
        damping=args.contact_damping,
        nconmax=args.nconmax,
        njmax=args.njmax,
        collect_contacts=True,
        scm_regularization=args.scm_regularization,
        scm_motion=args.scm_motion,
        scm_friction=args.scm_friction,
        scm_response_time=args.scm_response_time,
        scm_damping_ratio=args.scm_damping_ratio,
    )
    samples = [sample_state(model, data, body_ids)]
    samples[0].update(solve_ncontact=0, step_seconds=0, friction_power_W=0,
                      friction_work_J=0, max_slip_speed_m_s=0,
                      max_torsion_rate_rad_s=0, max_rolling_rate_rad_s=0)
    moments = Moments()
    pose = mujoco.MjData(model)
    contact_file = gzip.open(args.output_dir / f"{engine}_contacts.csv.gz", "wt", newline="")
    contact_writer = csv.writer(contact_file)
    contact_writer.writerow(CONTACT_FIELDS)
    if not args.headless:
        from mujoco import viewer as mujoco_viewer

        viewer_context = mujoco_viewer.launch_passive(model, data)
    else:
        viewer_context = nullcontext(None)
    failure = None
    failure_time = None
    with contact_file, viewer_context as viewer:
        wall_start = time.perf_counter()
        initial_time = float(data.time)
        for step in range(args.steps):
            if viewer is not None and not viewer.is_running():
                break
            try:
                mujoco.mj_copyData(pose, model, data)
                backend.step(data)
            except RuntimeError as exc:
                # Preserve valid samples and report a failed stability trial.
                # Capacity/dependency failures still propagate as invalid runs.
                failure = str(exc)
                failure_time = float(data.time) if np.isfinite(data.time) else None
                print(f"{engine}: {failure}; stopping this trajectory")
                break
            sample = sample_state(model, data, body_ids)
            native = backend.last_contacts
            obs = contact_observations(model, pose, data.qvel, native)
            moments.add(obs["depth"])
            write_contacts(contact_writer, step, engine, args.scene, "rollout", "rollout", native, obs)
            rel = obs["relative_velocity"]
            sample.update(solve_ncontact=len(native["gap"]), step_seconds=backend.last_step_seconds,
                          friction_power_W=float(obs["power"].sum()),
                          friction_work_J=float(obs["power"].sum()) * model.opt.timestep,
                          max_slip_speed_m_s=float(np.linalg.norm(rel[:, 1:3], axis=1).max(initial=0)),
                          max_torsion_rate_rad_s=float(np.abs(rel[:, 3]).max(initial=0)),
                          max_rolling_rate_rad_s=float(np.linalg.norm(rel[:, 4:6], axis=1).max(initial=0)))
            if not all(np.isfinite(value).all() for value in sample.values()):
                failure = "non-finite diagnostic sample"
                failure_time = float(data.time)
                print(f"{engine}: {failure}; stopping this trajectory")
                break
            samples.append(sample)
            if viewer is not None:
                viewer.sync()
                remaining = data.time - initial_time - (
                    time.perf_counter() - wall_start
                )
                if remaining > 0:
                    time.sleep(remaining)
    trajectory = {
        key: np.asarray([sample[key] for sample in samples]) for key in samples[0]
    }
    summary = summarize(trajectory, args.steps)
    summary.update(penetration_all_contacts_mm=moments.report(1000),
                   friction_work_J=float(trajectory["friction_work_J"].sum()),
                   full_step_seconds_mean=float(np.mean(trajectory["step_seconds"][1:])) if len(samples) > 1 else None,
                   status="completed" if len(samples) - 1 == args.steps else "stopped")
    if failure is not None:
        summary.update(finite=False, failure=failure, failure_time_s=failure_time,
                       status="invalid" if "overflow" in failure else "failed")
    return trajectory, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, include_scm=True)
    parser.add_argument("--scene", choices=SCENES, default="primitives")
    parser.add_argument("--steps", type=positive_integer, default=1000)
    parser.add_argument("--timestep", type=positive_float, default=None)
    parser.add_argument("--ccd-iterations", type=positive_integer, default=100,
                        help="Convex collision iteration limit shared by all backends (upstream XML default: 35)")
    parser.add_argument(
        "--initial-speed",
        type=finite_float,
        default=1.0,
        help="ball angular speed in rad/s or cylinder rolling speed in m/s; ignored for primitives",
    )
    parser.add_argument("--friction-coefficient", type=positive_float, default=None,
                        help="Override torsional (ball) or rolling (cylinder) coefficient on all geoms")
    parser.add_argument("--scm-regularization", type=positive_float, default=1e-8)
    parser.add_argument("--scm-motion", choices=("dynamic", "increment"), default="dynamic")
    parser.add_argument("--scm-response-time", type=positive_float, default=.05)
    parser.add_argument("--scm-damping-ratio", type=positive_float, default=1.)
    parser.add_argument("--scm-friction", choices=("point", "matched"), default="point",
                        help="point: original translational contacts; matched: additional 6D friction extension")
    args = parser.parse_args()

    model_path = ASSET_DIRECTORY / f"{args.scene}.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.ccd_iterations = args.ccd_iterations
    if args.timestep is not None:
        model.opt.timestep = args.timestep
    if args.friction_coefficient is not None:
        if args.scene not in ("ball_rotation", "cylinder_rolling"):
            parser.error("--friction-coefficient applies only to ball_rotation/cylinder_rolling")
        model.geom_friction[:, 1 if args.scene == "ball_rotation" else 2] = args.friction_coefficient
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    initial_data = initial_state(model, args.scene, args.initial_speed)
    body_ids = np.flatnonzero(model.body_dofnum > 0)
    metadata = run_metadata(args, model_path, model)
    metadata.update(
        {
            "experiment": "upstream_primitive_scene_trajectories",
            "dynamic_body_ids": body_ids.tolist(),
            "dynamic_body_names": [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
                for body_id in body_ids
            ],
            "initial_qpos": initial_data.qpos.tolist(),
            "initial_qvel": initial_data.qvel.tolist(),
            "metrics_note": (
                "CPU MuJoCo recomputes geometry and kinetic energy at each backend's "
                "state; penetration is max(0, -contact.dist). Independent trajectory "
                "differences are not shared-state one-step prediction errors."
            ),
            "native_contact_statistics": "solve-time GPU geometry/forces, depth=max(0,-gap); mean/std over all individual detected contacts",
            "friction_power": "native contact wrench dotted with post-impulse relative velocity at solve pose; negative dissipates",
            "timing": "one world, synchronized full step, excludes input upload/readback/CPU diagnostics/rendering/initial compilation",
            "scm": "diagonal Delassus + rI, no additional damping, common dual-cone rows; dynamic/increment and point/matched friction explicit",
            "scm_corrected": "same SCM followed by alpha=clip(-g.T*p/(p.T*(J*Minv*J.T+rI)*p),0,1); one scalar per world; free motion unchanged",
            "scm_soft": {"response_time_s": args.scm_response_time, "damping_ratio": args.scm_damping_ratio,
                         "law": "x=h/tau; a=(2*zeta*x+x*x)/(1+2*zeta*x+x*x); b=x*x/(1+2*zeta*x+x*x); p=max(-D^-1*(a*J*v_projected+b*gap/h),0)"},
            "primary_reporting": "paper_metrics.json: all-contact penetration mean±std table, Fig. 5/6 body trajectories, Fig. 7 contact-count/time scatter",
            "supplementary_diagnostics": "max penetration, energy, trajectory RMSE and friction power/work remain additional diagnostics",
        }
    )
    trajectories = {}
    summaries = {}
    for engine in engines_for(args.engine, include_scm=True):
        print(f"Running {engine}: scene={args.scene}, steps={args.steps}", flush=True)
        trajectories[engine], summaries[engine] = simulate(
            engine, model, initial_data, body_ids, args
        )
        result = summaries[engine]
        print(
            f"{engine}: penetration={result['penetration_all_contacts_mm']['mean']} ± "
            f"{result['penetration_all_contacts_mm']['std']} mm (all contacts), "
            f"final kinetic energy={result['final_kinetic_energy_j']:.6g} J, "
            f"finite={result['finite']}",
            flush=True,
        )
    save_results(args.output_dir, trajectories, summaries, metadata)
    if __package__:
        from .paper_reporting import write_physics_report
    else:
        from paper_reporting import write_physics_report
    write_physics_report(args.output_dir, trajectories, summaries, metadata)


if __name__ == "__main__":
    main()
