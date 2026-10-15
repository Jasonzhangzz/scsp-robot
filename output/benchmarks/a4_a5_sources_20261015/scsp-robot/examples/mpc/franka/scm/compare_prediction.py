#!/usr/bin/env python3
"""Shared-state one-step prediction errors by REFERENCE rigid-contact mode.

This is a new experiment, not an upstream or Section IV prediction benchmark.
MJWarp is a numerical reference, not measured physical ground truth. Every
engine receives identical qpos, qvel and world-frame external wrench; solver
warm starts are reset for every query. No predicted trajectory feeds later
samples. The SCM variant uses diagonal Delassus, R=rI and no extra damping
on the common polyhedral point-contact rows, with explicit dynamic/increment
choices. See README.md for the distinction from the quasi-dynamic CSO model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from xml.sax.saxutils import quoteattr

if __package__:
    from .runtime_env import activate_dependencies
else:
    from runtime_env import activate_dependencies
activate_dependencies()

import mujoco
import numpy as np

if __package__:
    from .benchmark_utils import Backend, UPSTREAM_COMMIT, positive_int, nonnegative_float
    from .fidelity_metrics import Moments, contact_observations, motion_metrics
else:
    from benchmark_utils import Backend, UPSTREAM_COMMIT, positive_int, nonnegative_float
    from fidelity_metrics import Moments, contact_observations, motion_metrics

MODES = ("separation", "sliding", "sticking")
SHAPES = {"box": (".04 .03 .025", .025), "sphere": (".04", .04),
          "cylinder": (".035 .04", .04), "capsule": (".025 .03", .055),
          "ellipsoid": (".04 .03 .025", .025)}
REPOSITORY = Path(__file__).resolve().parents[4]
MESH_SHAPES = ("elephant", "stanford_bunny2", "teapot", "foam_brick", "rubber_duck", "mug")


def canonical_shape(shape):
    return "stanford_bunny2" if shape == "stanford_bunn2" else shape


def mesh_asset(shape):
    """Use repository STL scale and the existing manipulation scene's orientation."""
    shape = canonical_shape(shape)
    path = REPOSITORY / "envs" / "assets" / "objects" / f"{shape}.stl"
    if not path.is_file():
        raise FileNotFoundError(f"Object mesh not found: {path}")
    scene = REPOSITORY / "envs" / "xmls" / f"env_fingertips_{shape}.xml"
    geom = ET.parse(scene).find(".//geom[@name='obj']")
    if geom is None:
        raise ValueError(f"No object geom in {scene}")
    return path, geom.get("quat", "1 0 0 0")


def mesh_world_vertices(model, data):
    geom = model.geom("object").id
    mesh = int(model.geom_dataid[geom])
    adr, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    return (model.mesh_vert[adr:adr + count] @ data.geom_xmat[geom].reshape(3, 3).T
            + data.geom_xpos[geom])


def positive_float(value):
    value = nonnegative_float(value)
    if value == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def make_model(shape, friction, timestep, mass):
    shape = canonical_shape(shape)
    if shape in MESH_SHAPES:
        path, quat = mesh_asset(shape)
        asset = f'<asset><mesh name="object_mesh" file={quoteattr(str(path))}/></asset>'
        geometry = f'type="mesh" mesh="object_mesh" quat={quoteattr(quat)}'
        height = 0.
    else:
        size, height = SHAPES[shape]
        asset, geometry = "", f'type="{shape}" size="{size}"'
    xml = f'''<mujoco model="prediction_{shape}">
      <option timestep="{timestep}" integrator="Euler" cone="pyramidal" iterations="100" tolerance="1e-10"/>
      <default><joint damping="0"/><geom condim="3" friction="{friction} 0 0"/></default>
      <visual><global azimuth="135" elevation="-25"/><headlight ambient=".4 .4 .4"/></visual>
      {asset}
      <worldbody><light pos="0 0 3"/>
        <geom name="floor" type="plane" size="2 2 .1" rgba=".25 .3 .4 1"/>
        <body name="object" pos="0 0 {height + .005}"><freejoint name="object_free"/>
          <geom name="object" {geometry} mass="{mass}" rgba=".9 .55 .15 1"/>
        </body>
      </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    model.opt.ccd_iterations = 100
    if shape in MESH_SHAPES:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Compiled mesh vertices include MuJoCo's centering/principal-axis transform.
        # Place the lowest world vertex 5 mm above the plane, irrespective of STL origin.
        shift = .005 - float(mesh_world_vertices(model, data)[:, 2].min())
        model.qpos0[2] += shift
        model.body_pos[1, 2] += shift
        mujoco.mj_setConst(model, data)
    return model


def fresh_data(model, qpos=None, qvel=None, wrench=None):
    data = mujoco.MjData(model)
    if qpos is not None:
        data.qpos[:] = qpos
    if qvel is not None:
        data.qvel[:] = qvel
    if wrench is not None:
        data.xfrc_applied[1] = wrench
    mujoco.mj_forward(model, data)
    data.qacc_warmstart[:] = 0
    return data


def backend_for(engine, model, initial, args):
    return Backend(engine, model, initial, device=args.device,
                   stiffness=args.contact_stiffness, damping=args.contact_damping,
                   nconmax=args.nconmax, njmax=args.njmax, collect_contacts=True,
                   scm_regularization=args.scm_regularization, scm_motion=args.scm_motion,
                   scm_response_time=getattr(args, "scm_response_time", .05),
                   scm_damping_ratio=getattr(args, "scm_damping_ratio", 1.),
                   lcp_regularization=getattr(args, "lcp_regularization", 1e-8),
                   lcp_tolerance=getattr(args, "lcp_tolerance", 1e-8),
                   lcp_max_iterations=getattr(args, "lcp_max_iterations", 2000),
                   lcp_solver=getattr(args, "lcp_solver", "lemke"),
                   scm_friction="point")


def reference_mode(native, observations, *, force_threshold, slip_threshold, normal_threshold):
    """Force/velocity based contact mode, with explicit transition bin.

    The three-way label is not inferred from the requested sampling category
    or from ComFree/SCM facet activation. Soft-contact impacts with substantial
    loaded normal speed are retained as 'transition', not silently relabelled.
    """
    loaded = native["force"][:, 0] > force_threshold
    if not np.any(loaded):
        return "separation"
    velocity = observations["relative_velocity"][loaded]
    if np.max(np.abs(velocity[:, 0])) > normal_threshold:
        return "transition"
    slip = np.linalg.norm(velocity[:, 1:3], axis=1)
    return "sliding" if np.max(slip) > slip_threshold else "sticking"


def quaternion_angle(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(2 * np.arccos(np.clip(abs(np.dot(a, b)), 0, 1)))


def world_velocity(initial, predicted):
    # MuJoCo free-joint angular qvel is local; compare all angular velocities
    # in the SAME pre-step world frame, rather than mixing post-step frames.
    return np.r_[predicted.qvel[:3], initial.xmat[1].reshape(3, 3) @ predicted.qvel[3:]]


def prediction_metrics(initial, reference, prediction, mass, free_velocity, args):
    ref, pred = reference.qvel.copy(), prediction.qvel.copy()
    row = {"linear_velocity_error_m_s": float(np.linalg.norm(pred[:3] - ref[:3])),
           "angular_velocity_error_rad_s": float(np.linalg.norm(pred[3:] - ref[3:])),
           "position_error_m": float(np.linalg.norm(prediction.qpos[:3] - reference.qpos[:3])),
           "orientation_error_rad": quaternion_angle(prediction.qpos[3:7], reference.qpos[3:7]),
           "horizontal_speed_m_s": float(np.linalg.norm(pred[:2])),
           "height_m": float(prediction.xpos[1, 2]),
           "kinetic_energy_j": float(prediction.energy[1]),
           "linear_speed_m_s": float(np.linalg.norm(pred[:3])),
           "angular_speed_rad_s": float(np.linalg.norm(pred[3:]))}
    for prefix, r, p in (("velocity", ref, pred),
                         ("increment", ref - initial.qvel, pred - initial.qvel),
                         ("contact_correction", ref - free_velocity, pred - free_velocity)):
        stats = motion_metrics(r, p, mass, tolerance=args.direction_threshold)
        row.update({f"{prefix}_{key}": value for key, value in stats.items()})
    row["translation_increment_m"] = float(np.linalg.norm(prediction.qpos[:3] - initial.qpos[:3]))
    row["rotation_increment_rad"] = quaternion_angle(initial.qpos[3:7], prediction.qpos[3:7])
    return row


def sample_states(model, settled, mode, number, rng, friction, mass):
    """Controlled categories; measured MJWarp outcome is the analysis label."""
    for index in range(number):
        qpos, qvel, wrench = settled.qpos.copy(), np.zeros(6), np.zeros(6)
        yaw = rng.uniform(-np.pi, np.pi)
        # Rotate the settled pose around WORLD z (floor symmetry).
        qyaw = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        orientation = np.zeros(4)
        mujoco.mju_mulQuat(orientation, qyaw, qpos[3:7])
        qpos[3:7] = orientation
        angle = rng.uniform(-np.pi, np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        if mode == "separation":
            if index % 2 == 0:
                qpos[2] += rng.uniform(.02, .15)
                qvel[:3] = rng.uniform(-.5, .5, 3)
                condition = "airborne"
            else:
                qvel[2] = rng.uniform(.2, .8)
                condition = "lift_off"
            qvel[3:] = rng.uniform(-.2, .2, 3)
        elif mode == "sliding":
            qvel[:2] = direction * rng.uniform(.2, 1.5)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, qpos[3:7])
            # Free-joint angular velocity is local. Rotate about WORLD z so a
            # tilted mesh's supported contacts do not acquire artificial normal motion.
            qvel[3:] = rotation.reshape(3, 3).T @ np.array([0., 0., rng.uniform(-.2, .2)])
            wrench[:2] = direction * rng.uniform(-.3, .3) * friction * mass * 9.81
            condition = "tangential_motion"
        else:
            # Below the static friction bound; reference decides whether this
            # actually sticks under its soft contact and finite step size.
            wrench[:2] = direction * rng.uniform(0, .2) * friction * mass * 9.81
            condition = "subthreshold_force"
        yield qpos, qvel, wrench, condition


def display_state(window, model, display, data, title, wrench):
    mujoco.mj_copyData(display, model, data)
    window.set_texts((None, None, title,
                     f"speed: {np.linalg.norm(data.qvel[:3]):.3f} m/s\n"
                     f"external force: {np.linalg.norm(wrench[:3]):.3f} N"))
    window.sync()


def preview_mode(window, model, display, backend, qpos, qvel, wrench,
                 requested, initial_mode, args):
    """Independent MJWarp animation; never contributes samples to the benchmark."""
    duration = args.preview_seconds
    if duration <= 0:
        return
    data = fresh_data(model, qpos, qvel, wrench)
    backend.reset(data)
    steps = max(1, int(np.ceil(duration / model.opt.timestep)))
    render_stride = max(1, int(round(1 / (60 * model.opt.timestep))))
    started = time.monotonic()
    before = mujoco.MjData(model)
    display_state(window, model, display, data,
                  f"MJWarp preview | {requested}\ninitial reference: {initial_mode}\nnot included in statistics", wrench)
    for step in range(steps):
        if not window.is_running():
            break
        mujoco.mj_copyData(before, model, data)
        try:
            backend.step(data)
        except (RuntimeError, FloatingPointError) as exc:
            print(f"{requested} visualization preview stopped: {exc}", flush=True)
            break
        if (step + 1) % render_stride == 0 or step + 1 == steps:
            obs = contact_observations(model, before, data.qvel, backend.last_contacts)
            mode = reference_mode(backend.last_contacts, obs, force_threshold=args.force_threshold,
                                  slip_threshold=args.slip_threshold, normal_threshold=args.normal_threshold)
            display_state(window, model, display, data,
                          f"MJWarp preview | {requested}\ninitial reference: {initial_mode}; current: {mode}\nnot included in statistics", wrench)
            remaining = (step + 1) * model.opt.timestep - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)


def replay_states(source, mode, number):
    """Replay recorded inputs, never a backend's predicted state."""
    with np.load(source / "states.npz", allow_pickle=False) as states:
        mask = (states["engine"] == "mjwarp") & (states["requested_mode"] == mode)
        indices = np.flatnonzero(mask)
        if len(indices) < number:
            raise ValueError(f"{source}: only {len(indices)} recorded {mode} inputs; requested {number}")
        ids = states["sample_id"][indices]
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"{source}: duplicate reference sample IDs")
        for i in indices[:number]:
            yield (states["qpos_initial"][i].copy(), states["qvel_initial"][i].copy(),
                   states["wrench_world_N_Nm"][i].copy(), "replayed_input")


CONTACT_FIELDS = ["sample_id", "engine", "requested_mode", "reference_mode", "predicted_mode",
                  "contact_index", "geom1", "geom2", "dim", "gap_m", "depth_m"] + [
    f"{name}_{i}" for name, width in (("position_m", 3), ("frame", 9), ("friction", 5),
                                    ("force_N_Nm", 6), ("relative_velocity", 6), ("utilization", 3))
    for i in range(width)] + ["friction_power_W"]


def write_contacts(writer, sample_id, engine, requested, mode, predicted_mode, native, obs):
    for i in range(len(native["gap"])):
        writer.writerow([sample_id, engine, requested, mode, predicted_mode, i,
                         *native["geom"][i], native["dim"][i], native["gap"][i], obs["depth"][i],
                         *native["position"][i], *native["frame"][i].reshape(-1),
                         *native["friction"][i], *native["force"][i],
                         *obs["relative_velocity"][i], *obs["utilization"][i], obs["power"][i]])


def finite_stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "rmse": None, "p95": None, "max": None}
    return {"count": len(values), "mean": float(values.mean()), "std": float(values.std()),
            "rmse": float(np.sqrt(np.mean(values ** 2))), "p95": float(np.quantile(values, .95)),
            "max": float(values.max())}


def aggregate(rows, contact_moments, post_moments):
    report = {}
    for engine in sorted({r["engine"] for r in rows}):
        report[engine] = {}
        for mode in (*MODES, "transition", "invalid_reference"):
            selected = [r for r in rows if r["engine"] == engine and r["reference_mode"] == mode]
            if not selected:
                continue
            valid = [r for r in selected if r["status"] == "ok"]
            metrics = {key: finite_stats([r.get(key) for r in valid]) for key in (
                "linear_velocity_error_m_s", "angular_velocity_error_rad_s", "position_error_m",
                "orientation_error_rad", "velocity_weighted_error", "increment_weighted_error",
                "velocity_relative_error", "increment_relative_error", "contact_correction_relative_error",
                "increment_angle_deg", "contact_correction_angle_deg", "step_seconds", "scm_amplitude_scale",
                "max_slip_speed_m_s", "friction_power_W", "max_friction_utilization",
                "poststate_max_penetration_m", "lcp_residual", "lcp_iterations")}
            for prefix in ("velocity", "increment", "contact_correction"):
                defined = [r[prefix + "_same_direction"] for r in valid
                           if r.get(prefix + "_same_direction") is not None]
                metrics[prefix + "_direction"] = {
                    "defined_count": len(defined), "undefined_count": len(valid) - len(defined),
                    "same_direction_rate": float(np.mean(defined)) if defined else None}
            report[engine][mode] = {"attempted": len(selected), "valid": len(valid),
                                    "failed": len(selected) - len(valid), "metrics": metrics,
                                    "outcome_agreement_with_reference_rate": float(np.mean([
                                        r["predicted_mode"] == r["reference_outcome_mode"] for r in valid])) if valid else None,
                                    "predicted_modes": dict(Counter(r.get("predicted_mode", "failed") for r in selected)),
                                    "solve_pose_penetration_mm": contact_moments.get((engine, mode), Moments()).report(1000),
                                    "poststate_penetration_mm": post_moments.get((engine, mode), Moments()).report(1000)}
    return report


def plot_results(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    engines = sorted({r["engine"] for r in rows if r["engine"] != "mjwarp"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, key, label in zip(axes, ("linear_velocity_error_m_s", "angular_velocity_error_rad_s", "increment_angle_deg"),
                                ("Linear velocity error [m/s]", "Angular velocity error [rad/s]", "Increment angle [deg]")):
        for j, engine in enumerate(engines):
            means = []
            deviations = []
            for mode in MODES:
                values = [r.get(key) for r in rows if r["engine"] == engine and r["reference_mode"] == mode and r["status"] == "ok"]
                stats = finite_stats(values)
                means.append(stats["mean"] if stats["count"] else np.nan)
                deviations.append(stats["std"] if stats["count"] else np.nan)
            valid = np.isfinite(means)
            if np.any(valid):
                axis.bar((np.arange(3) + (j - (len(engines) - 1) / 2) * .25)[valid], np.asarray(means)[valid],
                         .25, yerr=np.asarray(deviations)[valid], capsize=3, label=engine)
        axis.set_xticks(np.arange(3), MODES)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=.25)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend()
    fig.suptitle("New contact-mode prediction experiment: mean ± std (reference: MJWarp)")
    fig.tight_layout()
    fig.savefig(path / "prediction_errors.png", dpi=180)
    fig.savefig(path / "prediction_errors.pdf")
    plt.close(fig)


def run_configuration(shape, friction, timestep, args, rng, output):
    shape = canonical_shape(shape)
    output.mkdir(parents=True, exist_ok=False)
    model = make_model(shape, friction, timestep, args.mass)
    asset_info = None
    if shape in MESH_SHAPES:
        path, quat = mesh_asset(shape)
        asset_info = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "scale": [1, 1, 1], "geom_quaternion_wxyz": [float(v) for v in quat.split()],
                      "collision_geometry": "single MuJoCo convex mesh hull"}
    mujoco.mj_saveLastXML(str(output / "model.xml"), model)
    initial = fresh_data(model)
    engines = tuple(dict.fromkeys(("mjwarp", *args.engines)))
    backends = {e: backend_for(e, model, initial, args) for e in engines}
    settled = fresh_data(model)
    replay_source = args.replay_dir / output.name if args.replay_dir else None
    if replay_source:
        source_metadata = json.loads((replay_source / "metrics.json").read_text())["metadata"]
        for key, value in (("shape", shape), ("friction", friction), ("mass_kg", args.mass), ("timestep_s", timestep)):
            if source_metadata[key] != value:
                raise ValueError(f"Replay {key} mismatch: {source_metadata[key]} != {value}")
        if asset_info and source_metadata.get("mesh_asset") != asset_info:
            raise ValueError("Replay mesh asset differs from the recorded geometry/orientation")
        with np.load(replay_source / "settled_state.npz", allow_pickle=False) as state:
            settled = fresh_data(model, state["qpos"], state["qvel"])
        settle_steps_used = source_metadata.get("settle_steps_used", source_metadata["settle_steps"])
    else:
        backends["mjwarp"].reset(settled)
        for _ in range(args.settle_steps):
            backends["mjwarp"].step(settled)
        settle_steps_used = args.settle_steps
        if shape in MESH_SHAPES and np.max(np.abs(settled.qvel)) > .02:
            print(f"{shape}: extending reference settling beyond {args.settle_steps} steps", flush=True)
            for _ in range(4 * args.settle_steps):
                backends["mjwarp"].step(settled)
                settle_steps_used += 1
                if np.max(np.abs(settled.qvel)) <= .02:
                    break
    settled.time = 0
    if np.max(np.abs(settled.qvel)) > .02:
        raise RuntimeError(f"Reference settling failed for {shape}: max |qvel|={np.max(np.abs(settled.qvel))}. Increase --settle-steps.")
    np.savez_compressed(output / "settled_state.npz", qpos=settled.qpos, qvel=settled.qvel)
    display = fresh_data(model)
    if args.headless:
        viewer_context = nullcontext(None)
    else:
        from mujoco import viewer
        viewer_context = viewer.launch_passive(model, display)
    rows, saved, moments, post_moments = [], [], {}, {}
    metadata = {"shape": shape, "friction": friction, "mass_kg": args.mass, "timestep_s": timestep,
                "nworld": 1, "warmstarts": "reset for every prediction", "reference": "vendored MJWarp",
                "sampling_seed": args.seed, "settle_steps": args.settle_steps,
                "settle_steps_used": settle_steps_used,
                "mesh_asset": asset_info,
                "visualization": {"preview_seconds": args.preview_seconds,
                                  "preview_engine": "mjwarp", "preview_in_statistics": False},
                "replay_source": str(replay_source.resolve()) if replay_source else None,
                "mode_definition": "input point velocities and reference solve-time load; post-impulse modes recorded separately",
                "thresholds": {"normal_force_N": args.force_threshold, "slip_m_s": args.slip_threshold,
                               "loaded_normal_speed_m_s": args.normal_threshold, "weighted_direction_norm": args.direction_threshold},
                "scm": {"regularization": args.scm_regularization, "motion": args.scm_motion,
                        "damping": 0, "contact_dimensions": 3, "basis": "four dual-cone pyramid faces per contact",
                        "law": "p=max(-(J*v_projected+gap/h)/(diag(J*Minv*J.T)+r),0)",
                        "amplitude_correction": "scm_corrected only: alpha=clip(-g.T*p/(p.T*(J*Minv*J.T+rI)*p),0,1); one scalar per world; p=0 => alpha=1"}}
    metadata["scm_soft"] = {"response_time_s": args.scm_response_time, "damping_ratio": args.scm_damping_ratio,
                            "law": "x=h/tau; a=(2*zeta*x+x*x)/(1+2*zeta*x+x*x); b=x*x/(1+2*zeta*x+x*x); p=max(-(a*J*v_projected+b*gap/h)/D,0)",
                            "interpretation": "per-row implicit spring-damper, effective mass=1/D; full contact coupling still approximated"}
    stopped = False
    metadata["lcp"] = {"regularization": args.lcp_regularization,
                       "tolerance": args.lcp_tolerance, "max_iterations": args.lcp_max_iterations,
                       "law": "0 <= p perp (J*Minv*J.T+rI)*p+J*v_free+gap/h >= 0",
                       "solver": args.lcp_solver,
                       "library": "QuantEcon 0.11.2" if args.lcp_solver == "lemke" else "independent feasible active-set",
                       "implementation": "float64 CPU LCP; common GPU collision and integration"}
    with gzip.open(output / "contacts.csv.gz", "wt", newline="") as contact_file, \
            gzip.open(output / "poststate_contacts.csv.gz", "wt", newline="") as post_file, viewer_context as window:
        contact_writer = csv.writer(contact_file)
        contact_writer.writerow(CONTACT_FIELDS)
        post_writer = csv.writer(post_file)
        post_writer.writerow(["sample_id", "engine", "reference_mode", "contact_index", "geom1", "geom2", "gap_m", "depth_m"])
        if window is not None:
            window.cam.lookat[:] = settled.xpos[1]
            window.cam.distance = max(.5, 4 * model.geom_rbound[model.geom("object").id])
        sample_id = 0
        for requested in args.modes:
            mode_start = len(rows)
            print(f"{shape}: starting {requested} ({args.samples_per_mode} independent one-step inputs)", flush=True)
            samples = (replay_states(replay_source, requested, args.samples_per_mode) if replay_source else
                       sample_states(model, settled, requested, args.samples_per_mode, rng, friction, args.mass))
            for sample_index, (qpos, qvel, wrench, condition) in enumerate(samples):
                if window is not None and not window.is_running():
                    stopped = True
                    break
                start = fresh_data(model, qpos, qvel, wrench)
                M = np.zeros((6, 6))
                mujoco.mj_fullM(model, M, start.qM)
                free_velocity = start.qvel + timestep * start.qacc_smooth
                predictions = {}
                for engine, backend in backends.items():
                    data = fresh_data(model, qpos, qvel, wrench)
                    backend.reset(data)
                    try:
                        backend.step(data)
                        native = backend.last_contacts
                        obs = contact_observations(model, start, data.qvel, native)
                        if not (np.isfinite(native["force"]).all() and np.isfinite(obs["power"]).all()):
                            raise FloatingPointError("nonfinite native contact diagnostic")
                        mode = reference_mode(native, obs, force_threshold=args.force_threshold,
                                              slip_threshold=args.slip_threshold, normal_threshold=args.normal_threshold)
                        predictions[engine] = (data, native, obs, mode, None, backend.last_step_seconds)
                    except (RuntimeError, FloatingPointError) as exc:
                        predictions[engine] = (None, None, None, "failed", str(exc), None)
                reference = predictions["mjwarp"][0]
                if reference is not None:
                    ref_native = predictions["mjwarp"][1]
                    input_obs = contact_observations(model, start, start.qvel, ref_native)
                    mode = reference_mode(ref_native, input_obs, force_threshold=args.force_threshold,
                                          slip_threshold=args.slip_threshold, normal_threshold=args.normal_threshold)
                else:
                    mode = "invalid_reference"
                if window is not None and sample_index == 0 and reference is not None:
                    preview_mode(window, model, display, backends["mjwarp"], qpos, qvel, wrench,
                                 requested, mode, args)
                for engine, (data, native, obs, predicted_mode, failure, elapsed) in predictions.items():
                    row = {"sample_id": sample_id, "shape": shape, "friction": friction, "timestep_s": timestep,
                           "requested_mode": requested, "condition": condition, "reference_mode": mode,
                           "reference_outcome_mode": predictions["mjwarp"][3],
                           "engine": engine, "predicted_mode": predicted_mode, "status": "ok", "failure": failure,
                           "step_seconds": elapsed}
                    if engine == "scm_corrected" and data is not None:
                        row["scm_amplitude_scale"] = backends[engine].last_amplitude_scale
                    if engine == "lcp" and data is not None:
                        row.update(lcp_residual=backends[engine].api.last_lcp["residual"],
                                   lcp_iterations=backends[engine].api.last_lcp["iterations"])
                    if reference is None or data is None:
                        row["status"] = "invalid_reference" if reference is None else "failed"
                    else:
                        row.update(prediction_metrics(start, reference, data, M, free_velocity, args))
                        loaded = native["force"][:, 0] > args.force_threshold
                        row.update(ncontact=len(native["gap"]), loaded_contacts=int(loaded.sum()),
                                   max_penetration_m=float(obs["depth"].max(initial=0)),
                                   max_slip_speed_m_s=float(np.linalg.norm(obs["relative_velocity"][loaded, 1:3], axis=1).max(initial=0)),
                                   friction_power_W=float(obs["power"].sum()),
                                   friction_work_J=float(obs["power"].sum() * timestep),
                                   max_friction_utilization=float(obs["utilization"][loaded, 0].max(initial=0)))
                    # Preserve native contacts even when another engine/reference
                    # fails; errors remain undefined instead of treating it as zero.
                    if native is not None:
                        moments.setdefault((engine, mode), Moments()).add(obs["depth"])
                        write_contacts(contact_writer, sample_id, engine, requested, mode, predicted_mode, native, obs)
                    if data is not None:
                        gaps = np.asarray(data.contact.dist[:data.ncon])
                        depths = np.maximum(-gaps, 0)
                        post_moments.setdefault((engine, mode), Moments()).add(depths)
                        row.update(poststate_ncontact=data.ncon, poststate_max_penetration_m=float(depths.max(initial=0)))
                        for c in range(data.ncon):
                            post_writer.writerow([sample_id, engine, mode, c, *data.contact.geom[c], gaps[c], depths[c]])
                    rows.append(row)
                    saved.append({"sample_id": sample_id, "engine": engine, "reference_mode": mode,
                                  "requested_mode": requested, "qpos_initial": qpos.copy(), "qvel_initial": qvel.copy(),
                                  "wrench_world_N_Nm": wrench.copy(), "free_velocity": free_velocity.copy(),
                                  "qpos_predicted": data.qpos.copy() if data else np.full(7, np.nan),
                                  "qvel_predicted": data.qvel.copy() if data else np.full(6, np.nan),
                                  "world_velocity_predicted": world_velocity(start, data) if data else np.full(6, np.nan),
                                  "status": row["status"]})
                    if window is not None and data is not None:
                        display_state(window, model, display, data,
                                      f"One-step | {shape} | {engine}\nrequested: {requested}; reference: {mode}\noutcome: {predicted_mode}; sample: {sample_index + 1}/{args.samples_per_mode}", wrench)
                        time.sleep(args.display_pause)
                sample_id += 1
            labels = Counter(r["reference_mode"] for r in rows[mode_start:] if r["engine"] == "mjwarp")
            print(f"{shape} mu={friction:g} h={timestep:g}: requested {requested}, measured {dict(labels)}", flush=True)
            if stopped:
                break
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if saved:
        np.savez_compressed(output / "states.npz", **{key: np.asarray([s[key] for s in saved]) for key in saved[0]})
    report = {"metadata": metadata, "stopped_by_viewer": stopped, "requested_shared_states": len(args.modes) * args.samples_per_mode,
              "completed_shared_states": len(saved) // len(engines),
              "requested_vs_reference_modes": dict(Counter(f"{r['requested_mode']}->{r['reference_mode']}" for r in rows if r["engine"] == "mjwarp")),
              "by_reference_mode": aggregate(rows, moments, post_moments)}
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if __package__:
        from .paper_reporting import write_prediction_report
    else:
        from paper_reporting import write_prediction_report
    write_prediction_report({output.name: report}, output)
    if not args.no_plots and rows:
        plot_results(rows, output)
    return report


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Disable the visualization popup (enabled by default)")
    parser.add_argument("--preview-seconds", type=nonnegative_float, default=.6,
                        help="Independent MJWarp animation before each mode; excluded from statistics. 0 disables it")
    parser.add_argument("--display-pause", type=nonnegative_float, default=.02,
                        help="Wall-clock hold per displayed one-step prediction [s]")
    parser.add_argument("--engines", nargs="+", choices=("mjwarp", "comfree", "scm", "scm_corrected", "scm_soft", "lcp", "mujoco"), default=["mjwarp", "comfree", "scm", "scm_corrected"])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--shapes", nargs="+", type=canonical_shape, choices=(*SHAPES, *MESH_SHAPES),
                        default=["box", "sphere"], help="Primitives or repository STL objects; stanford_bunn2 aliases stanford_bunny2")
    parser.add_argument("--samples-per-mode", type=positive_int, default=100, help="Shared states per shape/friction/timestep and requested mode")
    parser.add_argument("--frictions", nargs="+", type=positive_float, default=[.5])
    parser.add_argument("--timesteps", nargs="+", type=positive_float, default=[.002])
    parser.add_argument("--mass", type=positive_float, default=.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--settle-steps", type=positive_int, default=500,
                        help="Initial reference settling steps; mesh objects extend up to 5x if still moving")
    parser.add_argument("--contact-stiffness", type=nonnegative_float, default=.1)
    parser.add_argument("--contact-damping", type=nonnegative_float, default=.001)
    parser.add_argument("--scm-regularization", type=positive_float, default=1e-8)
    parser.add_argument("--scm-motion", choices=("dynamic", "increment"), default="dynamic")
    parser.add_argument("--scm-response-time", type=positive_float, default=.05, help="Soft SCM per-row relaxation time [s]")
    parser.add_argument("--scm-damping-ratio", type=positive_float, default=1., help="Soft SCM per-row damping ratio")
    parser.add_argument("--lcp-regularization", type=positive_float, default=1e-8)
    parser.add_argument("--lcp-tolerance", type=positive_float, default=1e-8)
    parser.add_argument("--lcp-max-iterations", type=positive_int, default=2000)
    parser.add_argument("--lcp-solver", choices=("lemke", "active-set"), default="lemke",
                        help="Dedicated QuantEcon Lemke solver (default), or independent QP active-set for validation")
    parser.add_argument("--slip-threshold", type=positive_float, default=.001, help="Loaded contact sticking tolerance [m/s]")
    parser.add_argument("--normal-threshold", type=positive_float, default=.01, help="Loaded |normal speed| above this is a transition [m/s]")
    parser.add_argument("--force-threshold", type=positive_float, default=1e-4, help="Contact load threshold [N]")
    parser.add_argument("--direction-threshold", type=positive_float, default=1e-5, help="Minimum inertia-weighted motion norm to define angle")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nconmax", type=positive_int, default=128)
    parser.add_argument("--njmax", type=positive_int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "prediction")
    parser.add_argument("--replay-dir", type=Path, help="Reuse exact recorded initial states/wrenches from a previous prediction run")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Reject duplicate inputs; config folder identifiers must remain unique.
    if any(len(v) != len(set(v)) for v in (args.shapes, args.frictions, args.timesteps, args.modes)):
        parser.error("Do not repeat shapes, frictions, timesteps or modes")
    rng = np.random.default_rng(args.seed)
    reports = {}
    for shape in args.shapes:
        for friction in args.frictions:
            for timestep in args.timesteps:
                key = f"{shape}_mu{friction}_h{timestep}"
                reports[key] = run_configuration(shape, friction, timestep, args, rng, args.output_dir / key)
    import sys
    import warp
    arguments = {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items() if not key.startswith("_")}
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "experiment": "shared_state_single_step_contact_mode_prediction_error",
        "arguments": arguments, "python": sys.executable, "mujoco": mujoco.__version__, "warp": warp.__version__,
        "mode_definition": "input contact-point velocities at solve pose, with MJWarp solve-time normal load; outcome modes use post-impulse velocity separately",
        "upstream_commit": UPSTREAM_COMMIT, "reference": "vendored MJWarp, not physical ground truth",
        "paper_section_iv": {"A": "all-contact penetration population mean/std",
                             "B": "torsional angular-speed and rolling COM-speed decay curves; separate compare_physics scenes",
                             "C": "horizontal speed, height and kinetic energy diagnostics",
                             "D": "full-step synchronized wall time, excluding readback/rendering, single world here",
                             "E": "parallel throughput is not measured by this single-step experiment"},
        "configurations": list(reports)}, indent=2, allow_nan=False) + "\n")
    summary = []
    for configuration, report in reports.items():
        for engine, modes in report["by_reference_mode"].items():
            for mode, group in modes.items():
                row = {"configuration": configuration, "engine": engine, "reference_mode": mode,
                       "attempted": group["attempted"], "valid": group["valid"], "failed": group["failed"]}
                for metric, statistics in group["metrics"].items():
                    row.update({f"{metric}_{stat}": value for stat, value in statistics.items()})
                for key in ("solve_pose_penetration_mm", "poststate_penetration_mm"):
                    row.update({f"{key}_{stat}": value for stat, value in group[key].items()})
                summary.append(row)
    if summary:
        columns = list(dict.fromkeys(k for row in summary for k in row))
        with (args.output_dir / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(summary)
    if __package__:
        from .paper_reporting import write_prediction_report
    else:
        from paper_reporting import write_prediction_report
    write_prediction_report(reports, args.output_dir)
    print(f"Saved {len(reports)} configurations to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
