#!/usr/bin/env python3
"""Shared-state prediction benchmark with MuJoCo ground truth.

This is a new experiment, not an upstream or Section IV prediction benchmark.
MuJoCo is the reference backend. Every engine receives identical qpos, qvel and world-frame external wrench; solver
warm starts are reset for every query. No predicted trajectory feeds later
samples. ``scm`` is the paper SCM of tro_revision/method.tex (the law used by
planning/point_force_gpu.py in the MPC): n_d=4 facet rows per MuJoCo point
contact, softplus_T diagonal-Delassus estimate with d=max(phi,0), then one
residual step of length 1/lambda_max(W) on the coupled Delassus operator,
evaluated on CPU in float64 with MuJoCo collision and smooth dynamics.
``scm_corrected`` keeps the older hard-max diagonal impulse with the global
amplitude clip; scm_residual/scm_preconditioned are the Warp GPU variants.
``isaacgym`` rebuilds the same body (MuJoCo mass/inertia/compiled geometry) in
PhysX and advances one dt=h step. See README.md.
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
    from .runtime_env import activate_dependencies, preload_isaacgym
else:
    from runtime_env import activate_dependencies, preload_isaacgym
activate_dependencies()
# Isaac Gym must be imported before MuJoCo (USD symbol clash), see runtime_env.
preload_isaacgym()

import mujoco
import numpy as np

if __package__:
    from .benchmark_utils import Backend, UPSTREAM_COMMIT, positive_int, nonnegative_float
    from .fidelity_metrics import Moments, contact_observations, motion_metrics
else:
    from benchmark_utils import Backend, UPSTREAM_COMMIT, positive_int, nonnegative_float
    from fidelity_metrics import Moments, contact_observations, motion_metrics

CONTACT_MODES = ("transition", "sliding", "sticking")
ROLLOUT_MODES = ("rotation", "flipping")
# Keep the public name used by older callers while making transition the
# reported contact class. ``separation`` remains accepted as a legacy request.
MODES = CONTACT_MODES + ("separation",)
SHAPES = {"box": (".04 .03 .025", .025), "sphere": (".04", .04),
          "cylinder": (".035 .04", .04), "capsule": (".025 .03", .055),
          "ellipsoid": (".04 .03 .025", .025)}
REPOSITORY = Path(__file__).resolve().parents[4]
MESH_SHAPES = ("elephant", "stanford_bunny2", "teapot", "foam_brick", "rubber_duck", "mug", "piggy_bank")


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


def nonnegative_int(value):
    try:
        value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


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


def backend_for(engine, model, initial, args, friction=.5):
    return Backend(engine, model, initial, device=args.device,
                   stiffness=args.contact_stiffness, damping=args.contact_damping,
                   friction=friction,
                   nconmax=args.nconmax, njmax=args.njmax, collect_contacts=True,
                   scm_regularization=args.scm_regularization, scm_motion=args.scm_motion,
                   scm_response_time=getattr(args, "scm_response_time", .05),
                   scm_damping_ratio=getattr(args, "scm_damping_ratio", 1.),
                   scm_softplus_temperature=getattr(args, "scm_softplus_temperature", 1e-4),
                   scm_correction_steps=getattr(args, "scm_correction_steps", 1),
                   scm_sweeps=getattr(args, "scm_sweeps", 3),
                   scm_soft_contact=getattr(args, "scm_soft_contact", "none"),
                   lcp_regularization=getattr(args, "lcp_regularization", 1e-8),
                   lcp_tolerance=getattr(args, "lcp_tolerance", 1e-8),
                   lcp_max_iterations=getattr(args, "lcp_max_iterations", 2000),
                   lcp_solver=getattr(args, "lcp_solver", "lemke"),
                   contactsdf_sigma=getattr(args, "contactsdf_sigma", 120.),
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


def full_mass(model, data):
    """MuJoCo 3.6 calls the packed buffer qM; newer releases call it M."""
    matrix = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, matrix)
    except TypeError:
        mujoco.mj_fullM(model, matrix, getattr(data, "qM", getattr(data, "M", None)))
    return matrix


def prediction_metrics(initial, reference, prediction, mass, free_velocity, args):
    # Free-joint angular qvel is body-local. The historical benchmark and
    # the rotation/flipping rollout logs compare angular velocity in the
    # common pre-step world frame; comparing raw local qvel exaggerates errors
    # after the body orientation changes.
    ref = world_velocity(initial, reference)
    pred = world_velocity(initial, prediction)
    initial_world = np.r_[initial.qvel[:3], initial.xmat[1].reshape(3, 3) @ initial.qvel[3:]]
    free_world = np.r_[free_velocity[:3], initial.xmat[1].reshape(3, 3) @ free_velocity[3:]]
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
                         ("increment", ref - initial_world, pred - initial_world),
                         ("contact_correction", ref - free_world, pred - free_world)):
        stats = motion_metrics(r, p, mass, tolerance=args.direction_threshold)
        row.update({f"{prefix}_{key}": value for key, value in stats.items()})
    row["translation_increment_m"] = float(np.linalg.norm(prediction.qpos[:3] - initial.qpos[:3]))
    row["rotation_increment_rad"] = quaternion_angle(initial.qpos[3:7], prediction.qpos[3:7])
    return row


def sample_states(model, settled, mode, number, rng, friction, mass):
    """Controlled categories; measured MuJoCo outcome is the analysis label."""
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
        if mode == "transition":
            # Match the earlier velocity-error benchmark: supported pose with
            # an approaching normal speed and bounded tangential/rotational
            # motion. This guarantees that the transition bin contains actual
            # impact/new-contact states instead of airborne free-flight rows.
            qvel[2] = -rng.uniform(.2, 1.2)
            qvel[:2] = direction * rng.uniform(0., .4)
            qvel[3:] = rng.uniform(-1., 1., 3)
            condition = "contact_transition"
        elif mode == "separation":
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
        reference = json.loads((source / "metrics.json").read_text()).get("metadata", {}).get("reference", "mujoco")
        mask = (states["engine"] == reference) & (states["requested_mode"] == mode)
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
        for mode in (*CONTACT_MODES, "invalid_reference"):
            selected = [r for r in rows if r["engine"] == engine and r["reference_mode"] == mode]
            if not selected:
                continue
            valid = [r for r in selected if r["status"] == "ok"]
            metrics = {key: finite_stats([r.get(key) for r in valid]) for key in (
                "linear_velocity_error_m_s", "angular_velocity_error_rad_s", "position_error_m",
                "orientation_error_rad", "velocity_weighted_error", "increment_weighted_error",
                "velocity_relative_error", "increment_relative_error", "contact_correction_relative_error",
                "increment_angle_deg", "contact_correction_angle_deg", "step_seconds", "scm_amplitude_scale",
                "scm_lipschitz", "scm_step_size", "scm_power_iterations",
                "max_slip_speed_m_s", "friction_power_W", "max_friction_utilization",
                "poststate_max_penetration_m", "lcp_residual", "lcp_iterations")}
            for prefix in ("velocity", "increment", "contact_correction"):
                defined = [r[prefix + "_same_direction"] for r in valid
                           if r.get(prefix + "_same_direction") is not None]
                metrics[prefix + "_direction"] = {
                    "defined_count": len(defined), "undefined_count": len(valid) - len(defined),
                    "same_direction_rate": float(np.mean(defined)) if defined else None}
            by_contacts = {}
            for contact_group in ("1", "2", "3plus"):
                subset = [r for r in valid if r.get("contact_count_group") == contact_group]
                if not subset:
                    continue
                by_contacts[contact_group] = {
                    "attempted": len(subset), "valid": len(subset),
                    "metrics": {key: finite_stats([r.get(key) for r in subset]) for key in (
                        "linear_velocity_error_m_s", "angular_velocity_error_rad_s")},
                }
            report[engine][mode] = {"attempted": len(selected), "valid": len(valid),
                                    "failed": len(selected) - len(valid), "metrics": metrics,
                                    "by_contact_count": by_contacts,
                                    "outcome_agreement_with_reference_rate": float(np.mean([
                                        r["predicted_mode"] == r["reference_outcome_mode"] for r in valid])) if valid else None,
                                    "predicted_modes": dict(Counter(r.get("predicted_mode", "failed") for r in selected)),
                                    "solve_pose_penetration_mm": contact_moments.get((engine, mode), Moments()).report(1000),
                                    "poststate_penetration_mm": post_moments.get((engine, mode), Moments()).report(1000)}
    return report


def aggregate_rollouts(rows, reference_engine):
    """Aggregate per-step rotation/flipping errors for Table IV rollouts."""
    result = {}
    for mode in ROLLOUT_MODES:
        # One-step (teacher-forced) rows only; ``*_openloop`` rows are the
        # divergence of independently integrated trajectories.
        selected = [r for r in rows if r.get("experiment") == mode and
                    r.get("engine") != reference_engine and r.get("status") == "ok"
                    and not str(r.get("engine", "")).endswith("_openloop")
                    and "linear_velocity_error_m_s" in r]
        if not selected:
            continue
        result[mode] = {
            "attempted_steps": len(selected),
            "successful_trials": int(sum(bool(r.get("final_success", False)) for r in rows
                                          if r.get("experiment") == mode and
                                          r.get("engine") == reference_engine)),
            "metrics": {key: finite_stats([r.get(key) for r in selected]) for key in (
                "linear_velocity_error_m_s", "angular_velocity_error_rad_s")},
        }
    return result


def _format_cell(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v) and abs(float(v)) < 1e6]
    if not values:
        return "--"
    values = np.asarray(values, dtype=float)
    return f"${values.mean():.2f}\\pm{values.std():.2f}$"


SCM_ENGINES = ("scm_box", "scm", "scm_corrected")


def scm_engine_name(engines):
    """The SCM variant that Table IV compares (first present in priority order)."""
    engines = tuple(engines)
    return next((name for name in SCM_ENGINES if name in engines), None)


def write_table_iv(output_dir, args):
    """Materialize the requested Table IV cells from saved prediction rows."""
    mode_order = ("transition", "sliding", "sticking")
    count_order = ("1", "2", "3plus")
    scm_name = scm_engine_name(args.engines) or "scm"
    comparisons = (("SCM--MuJoCo", scm_name), ("SCM--IsaacGym", "scm_vs_isaacgym"))
    raw = []
    for path in sorted(Path(output_dir).glob("*/predictions.csv")):
        with path.open(newline="") as stream:
            raw.extend(csv.DictReader(stream))
    for row in raw:
        for key in ("linear_velocity_error_m_s", "angular_velocity_error_rad_s"):
            try: row[key] = float(row[key])
            except (TypeError, ValueError): row[key] = None
    rows, matrix = [], []
    for comparison, engine in comparisons:
        for metric, key in (("Linear", "linear_velocity_error_m_s"),
                            ("Angular", "angular_velocity_error_rad_s")):
            cells = []
            for mode in mode_order:
                for count in count_order:
                    values = [r[key] for r in raw if r.get("engine") == engine and
                              r.get("reference_mode") == mode and
                              r.get("contact_count_group") == count and r[key] is not None]
                    cells.append(_format_cell(values))
                    rows.append({"comparison": comparison, "metric": metric,
                                 "category": mode, "contact_count": count,
                                 "value": cells[-1], "n": len(values)})
            for mode in ROLLOUT_MODES:
                values = [r[key] for r in raw if r.get("engine") == engine and
                          r.get("experiment") == mode and r[key] is not None]
                # Rollout rows are stored in rollouts.jsonl, merged below.
                cells.append(_format_cell(values))
                rows.append({"comparison": comparison, "metric": metric,
                             "category": mode, "contact_count": "all",
                             "value": cells[-1], "n": len(values)})
            matrix.append((comparison, metric, cells))
    rollout_raw = []
    for path in sorted(Path(output_dir).glob("*/rollouts.jsonl")):
        for line in path.read_text().splitlines():
            try: rollout_raw.append(json.loads(line))
            except json.JSONDecodeError: pass
    for comparison, engine in comparisons:
        for metric, key in (("Linear", "linear_velocity_error_m_s"),
                            ("Angular", "angular_velocity_error_rad_s")):
            entry = next((item for item in matrix if item[:2] == (comparison, metric)), None)
            if entry is None: continue
            cells = entry[2]
            for idx, mode in enumerate(ROLLOUT_MODES):
                values = [r.get(key) for r in rollout_raw if r.get("engine") == engine and
                          r.get("experiment") == mode]
                cells[9 + idx] = _format_cell(values)
                for row in rows:
                    if (row["comparison"], row["metric"], row["category"]) == (comparison, metric, mode):
                        row["value"], row["n"] = cells[9 + idx], len(values)
    csv_path = Path(output_dir) / "table_iv.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["comparison", "metric", "category", "contact_count", "value", "n"])
        writer.writeheader(); writer.writerows(rows)
    md = ["| Comparison | Metric | Contact transition (1/2/≥3) | Sliding (1/2/≥3) | Sticking (1/2/≥3) | Rotation | Flipping |",
          "|---|---|---|---|---|---|---|"]
    for comparison, metric, cells in matrix:
        md.append("| %s | %s | %s | %s | %s | %s | %s |" %
                  (comparison, metric, " / ".join(cells[0:3]), " / ".join(cells[3:6]),
                   " / ".join(cells[6:9]), cells[9], cells[10]))
    (Path(output_dir) / "table_iv.md").write_text("\n".join(md) + "\n")
    if args.update_table_iv:
        paper = REPOSITORY / "tro_revision" / "experiment.tex"
        if paper.exists():
            text = paper.read_text()
            begin = text.find(r"\label{tab:scm_velocity_error}")
            if begin >= 0:
                end = text.find(r"\end{table*}", begin)
                block = text[begin:end]
                body_start = block.find(r"\midrule")
                prefix = block if body_start < 0 else block[:body_start]
                body = "" if body_start < 0 else block[body_start:]
                for _, _, cells in matrix:
                    for value in cells:
                        body = body.replace(r"\placeholder", value, 1)
                block = prefix + body
                text = text[:begin] + block + text[end:]
                paper.write_text(text)
    return {"csv": str(csv_path), "markdown": str(Path(output_dir) / "table_iv.md"),
            "rows": rows, "matrix": matrix}


def plot_results(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    engines = sorted({r["engine"] for r in rows if r["engine"] != "mujoco"})
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
    fig.suptitle("Contact-mode prediction error: mean ± std (reference: MuJoCo)")
    fig.tight_layout()
    fig.savefig(path / "prediction_errors.png", dpi=180)
    fig.savefig(path / "prediction_errors.pdf")
    plt.close(fig)


def contact_count_group(count):
    """Paper grouping for the number of active reference contacts."""
    count = int(count)
    return "1" if count == 1 else "2" if count == 2 else "3plus" if count >= 3 else "0"


def _quat_error_axis(current, target):
    """Return a world-frame small-angle vector for wxyz MuJoCo quaternions."""
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    inv = np.zeros(4)
    mujoco.mju_negQuat(inv, current)
    rel = np.zeros(4)
    mujoco.mju_mulQuat(rel, target, inv)
    sign = 1. if rel[0] >= 0 else -1.
    return 2. * sign * rel[1:4]


def _rollout_target(initial_qpos, mode):
    target = np.asarray(initial_qpos, dtype=float).copy()
    # Match the direct-force experiments' fixed ground target and tolerances;
    # rotation changes the yaw on the support plane, flipping changes the
    # upright axis while leaving the target position unchanged.
    if mode == "rotation":
        target[:2] += np.array([0., .1])
        delta = np.array([np.cos(.6 / 2), 0., 0., np.sin(.6 / 2)])
    else:
        delta = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0., 0.])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, delta, target[3:7])
    target[3:7] = out
    return target


def _optimal_wrench(model, data, target, mode, mass):
    """Simple deterministic optimal-point wrench used by both references.

    The point is the support extreme in the desired motion direction.  The
    force/torque is a bounded PD wrench, equivalent to the direct optimal
    point-force setup while remaining independent of a particular optimizer.
    """
    pos_err = target[:3] - data.qpos[:3]
    vel = data.qvel[:3]
    if mode == "rotation":
        direction = np.array([0., 1., 0.])
        force = np.clip(35. * pos_err + 2. * (direction * .15 - vel), -.5, .5)
        torque = 1.5 * _quat_error_axis(data.qpos[3:7], target[3:7]) - .04 * data.qvel[3:]
    else:
        force = np.clip(35. * pos_err - 2. * vel, -.5, .5)
        torque = 2.5 * _quat_error_axis(data.qpos[3:7], target[3:7]) - .08 * data.qvel[3:]
    # Apply at the lowest support vertex, which creates the desired moment.
    try:
        points = mesh_world_vertices(model, data)
        point = points[np.argmin(points[:, 2])]
        com = data.xpos[1]
        torque = torque + np.cross(point - com, force)
    except Exception:
        pass
    torque = np.clip(torque, -.08, .08)
    return np.r_[force, torque]


def _rollout_success(qpos, target):
    return bool(np.linalg.norm(qpos[:3] - target[:3]) < .02 and
                quaternion_angle(qpos[3:7], target[3:7]) < .015)


def run_rollouts(model, settled, backends, args, shape, friction, rng):
    """Run rotation/flipping windows and record every next-velocity error."""
    if args.rollout_samples <= 0:
        return []
    rows = []
    reference_engine = args.reference_engine
    for mode in ROLLOUT_MODES:
        for trial in range(args.rollout_samples):
            initial = fresh_data(model, settled.qpos, np.zeros(6))
            # Keep independent random yaw while retaining native settling
            # height, as in ExplicitMPCParams/test_mpc_isaac.
            yaw = float(rng.uniform(-np.pi, np.pi))
            qyaw = np.array([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)])
            q = np.zeros(4); mujoco.mju_mulQuat(q, qyaw, initial.qpos[3:7]); initial.qpos[3:7] = q
            mujoco.mj_forward(model, initial)
            target = _rollout_target(initial.qpos, mode)
            datas = {e: fresh_data(model, initial.qpos, initial.qvel) for e in backends}
            for e, backend in backends.items():
                backend.reset(datas[e])
            success = {e: False for e in backends}
            for step in range(args.rollout_steps):
                reference_start = fresh_data(model, datas[reference_engine].qpos,
                                             datas[reference_engine].qvel)
                wrench = _optimal_wrench(model, reference_start, target, mode, args.mass)
                reference_data = None
                reference_native = None
                reference_obs = None
                for engine, backend in backends.items():
                    start = fresh_data(model, datas[engine].qpos, datas[engine].qvel, wrench)
                    backend.reset(start)
                    try:
                        backend.step(start)
                        if (not np.isfinite(start.qpos).all() or not np.isfinite(start.qvel).all()
                                or np.max(np.abs(start.qvel)) > 1e3):
                            raise FloatingPointError("rollout became nonfinite or unstable")
                        datas[engine] = start
                        success[engine] |= _rollout_success(start.qpos, target)
                        if engine == reference_engine:
                            reference_data = start
                            reference_native = backend.last_contacts
                            reference_obs = contact_observations(model, reference_start,
                                                                 start.qvel, reference_native)
                    except (RuntimeError, FloatingPointError) as exc:
                        rows.append({"experiment": mode, "trial": trial, "step": step,
                                     "engine": engine, "status": "failed", "failure": str(exc)})
                if reference_data is None:
                    continue
                count = len(reference_native["gap"]) if reference_native is not None else 0
                group = contact_count_group(count)
                success[reference_engine] |= _rollout_success(reference_data.qpos, target)
                M = full_mass(model, reference_start)[:6, :6]
                free = reference_start.qvel + model.opt.timestep * reference_start.qacc_smooth
                # One-step predictions from the SAME reference state (teacher
                # forcing): every engine is reset to the reference state and
                # advanced once, exactly as in the contact-mode measurements.
                # These rows carry the plain engine name and feed Table IV.
                # The divergence of the independently integrated trajectories
                # is kept under ``<engine>_openloop`` for reference only.
                probes = {}
                for engine, backend in backends.items():
                    if engine == reference_engine:
                        continue
                    probe = fresh_data(model, reference_start.qpos, reference_start.qvel, wrench)
                    backend.reset(probe)
                    try:
                        backend.step(probe)
                        if not (np.isfinite(probe.qpos).all() and np.isfinite(probe.qvel).all()):
                            raise FloatingPointError("one-step probe became nonfinite")
                        probes[engine] = probe
                    except (RuntimeError, FloatingPointError) as exc:
                        rows.append({"experiment": mode, "trial": trial, "step": step,
                                     "engine": engine, "status": "failed", "failure": str(exc)})
                    # Restore the engine's own open-loop state (PhysX keeps state).
                    if datas.get(engine) is not None:
                        backend.reset(datas[engine])
                base = {"experiment": mode, "trial": trial, "step": step, "status": "ok",
                        "contact_count": count, "contact_count_group": group,
                        "success": bool(success[reference_engine])}
                for engine, probe in probes.items():
                    row = dict(base, engine=engine)
                    row.update(prediction_metrics(reference_start, reference_data, probe, M, free, args))
                    rows.append(row)
                for engine, data in datas.items():
                    if engine == reference_engine or data is None:
                        continue
                    row = dict(base, engine=f"{engine}_openloop")
                    row.update(prediction_metrics(reference_start, reference_data, data, M, free, args))
                    rows.append(row)
                rollout_scm = scm_engine_name(datas)
                if "isaacgym" in datas and rollout_scm is not None:
                    if "isaacgym" in probes and rollout_scm in probes:
                        pair = dict(base, engine="scm_vs_isaacgym")
                        pair.update(prediction_metrics(reference_start, probes["isaacgym"],
                                                       probes[rollout_scm], M, free, args))
                        rows.append(pair)
                    pair = dict(base, engine="scm_vs_isaacgym_openloop")
                    pair.update(prediction_metrics(reference_start, datas["isaacgym"],
                                                   datas[rollout_scm], M, free, args))
                    rows.append(pair)
            for engine in backends:
                rows.append({"experiment": mode, "trial": trial, "step": args.rollout_steps,
                             "engine": engine, "status": "ok", "success": bool(success[engine]),
                             "final_success": bool(success[engine])})
    return rows


def close_backends(backends):
    for backend in backends.values():
        api = getattr(backend, "api", None)
        if api is not None and hasattr(api, "close"):
            api.close()


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
    engines = tuple(dict.fromkeys((args.reference_engine, *args.engines)))
    backends = {e: backend_for(e, model, initial, args, friction) for e in engines}
    try:
        return _run_configuration(shape, friction, timestep, args, rng, output,
                                  model, asset_info, engines, backends)
    finally:
        # PhysX allows one Foundation per process: a sim left open by a failed
        # configuration (for example reference settling) would make the next
        # Isaac Gym backend abort the whole batch.
        close_backends(backends)


def _run_configuration(shape, friction, timestep, args, rng, output, model, asset_info, engines, backends):
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
        backends[args.reference_engine].reset(settled)
        for _ in range(args.settle_steps):
            backends[args.reference_engine].step(settled)
        settle_steps_used = args.settle_steps
        if shape in MESH_SHAPES and np.max(np.abs(settled.qvel)) > .02:
            print(f"{shape}: extending reference settling beyond {args.settle_steps} steps", flush=True)
            for _ in range(4 * args.settle_steps):
                backends[args.reference_engine].step(settled)
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
                "nworld": 1, "warmstarts": "reset for every prediction", "reference": args.reference_engine,
                "backends": {name: {"isaacgym_fallback": bool(getattr(backend, "isaacgym_fallback", False)),
                                     "isaacgym_geometry_fallback": bool(getattr(backend, "isaacgym_geometry_fallback", False)),
                                     "isaacgym_error": getattr(backend, "isaacgym_error", None),
                                     "isaacgym_collision": getattr(backend, "isaacgym_collision", None),
                                     "isaacgym_worker_python": getattr(backend, "isaacgym_worker", None)}
                             for name, backend in backends.items()},
                "sampling_seed": args.seed, "settle_steps": args.settle_steps,
                "settle_steps_used": settle_steps_used,
                "mesh_asset": asset_info,
                "visualization": {"preview_seconds": args.preview_seconds,
                                  "preview_engine": args.reference_engine, "preview_in_statistics": False},
                "replay_source": str(replay_source.resolve()) if replay_source else None,
                "mode_definition": "input point velocities and reference solve-time load; post-impulse modes recorded separately",
                "thresholds": {"normal_force_N": args.force_threshold, "slip_m_s": args.slip_threshold,
                               "loaded_normal_speed_m_s": args.normal_threshold, "weighted_direction_norm": args.direction_threshold},
                "scm": {"source": "tro_revision/method.tex eq. scm_single_step_projection; "
                                  "same law as planning/point_force_gpu.py model 0 (GPU CEM of test_mpc_isaac.py)",
                        "regularization": args.scm_regularization, "motion": "dynamic",
                        "contact_dimensions": 3, "basis": "n_d=4 facet rows n+-mu*t per contact",
                        "softplus_temperature_N_s": args.scm_softplus_temperature,
                        "law": "g=J*v_free+max(gap,0)/h; D=diag(J*Minv*J.T)+eps; beta0=softplus_T(-g/D); "
                               "beta=softplus_T(beta0-(W*beta0+g)/L); L=lambda_max(W); v+=v_free+Minv*J.T*beta",
                        "mass": "full MuJoCo generalized mass at the query state; W applied matrix-free",
                        "collision": "MuJoCo CPU active point contacts at the query pose"},
                "scm_box": {"source": "models/soft_box_contact.py",
                            "rows": "(n, t1, t2) per contact; Coulomb box |lambda_t| <= mu lambda_n",
                            "law": "lambda_n=softplus_T(lambda_n-(n.v+max(gap,0)/h+eps*lambda_n)/D_n); "
                                   "lambda_t=clip_T(lambda_t-(t.v+eps*lambda_t)/D_t, mu*lambda_n); "
                                   "clip_T(x,c)=x-softplus_T(x-c)+softplus_T(-x-c); v+=Minv*J_i.T*(new-old)",
                            "sweeps": getattr(args, "scm_sweeps", 3),
                            "soft_contact": getattr(args, "scm_soft_contact", "none"),
                            "order": "fixed MuJoCo contact order, unrolled (explicit finite composition)",
                            "softplus_temperature_N_s": args.scm_softplus_temperature,
                            "regularization": args.scm_regularization},
                "scm_corrected": {"law": "p=max(-(J*v_projected+gap/h)/(diag(J*Minv*J.T)+r),0)",
                                  "amplitude_correction": "alpha=clip(-g.T*p/(p.T*(J*Minv*J.T+rI)*p),0,1); one scalar per world; p=0 => alpha=1"}}
    metadata["scm_soft"] = {"response_time_s": args.scm_response_time, "damping_ratio": args.scm_damping_ratio,
                            "law": "x=h/tau; a=(2*zeta*x+x*x)/(1+2*zeta*x+x*x); b=x*x/(1+2*zeta*x+x*x); p=max(-(a*J*v_projected+b*gap/h)/D,0)",
                            "interpretation": "per-row implicit spring-damper, effective mass=1/D; full contact coupling still approximated"}
    metadata["scm_soft_corrected"] = {
        "law_version": "softplus_nonnegative_gap_v2",
        "law": "d=max(gap,0); g=J*v_projected+d/h; p0=T*log(1+exp((-g/D)/T)); p=alpha*p0",
        "gap_policy": "max(signed_gap,0) in both initial impulse and amplitude correction",
        "temperature_N_s": getattr(args, "scm_softplus_temperature", 1e-4),
        "response_time_s": 0.,
        "alpha": "clip(-g.T*p0 / (||J.T*p0||_Minv^2+r*||p0||^2),0,1)",
        "interpretation": "SCM with LogSumExp smoothed positive-part impulses and the same global line search",
        "limitation": "smooths row activation only; global clipping and collision remain piecewise smooth"}
    metadata["scm_residual"] = {
        "law_version": "spectral_residual_v1",
        "law": "beta0=softplus_T(-g/D); r=W@beta0+g; beta=softplus_T(beta0-r/L); L=lambda_max(W)",
        "W": "J @ Minv @ (J.T @ beta) + r * beta; L by power iteration, once per contact geometry",
        "gap_policy": "max(signed_gap,0)",
        "temperature_N_s": getattr(args, "scm_softplus_temperature", 1e-4),
        "correction_steps": getattr(args, "scm_correction_steps", 1),
        "interpretation": "one Euclidean projected-gradient step on the coupled dual QP; no shared amplitude clip"}
    metadata["scm_preconditioned"] = {
        "law_version": "jacobi_gershgorin_residual_v1",
        "law": "r=W@beta0+g; beta=softplus_T(beta0-omega*D^{-1}@r); omega=1/L_N",
        "L_N": "max_i sum_j |W_ij|/sqrt(D_ii D_jj), Gershgorin bound of the D-normalized Delassus",
        "gap_policy": "max(signed_gap,0)",
        "temperature_N_s": getattr(args, "scm_softplus_temperature", 1e-4),
        "correction_steps": getattr(args, "scm_correction_steps", 1),
        "interpretation": "one Jacobi-preconditioned projected-gradient step; D and L_N fixed for the step"}
    metadata["dart"] = {
        "source": "DART v6.16.0 ContactConstraint/PgsBoxedLcpSolver equations, deterministic NumPy port",
        "collision": "MuJoCo CPU active contacts (touching or penetrating only)",
        "contact_dimensions": 3, "cfm": 1e-5, "erp": .01, "max_erv_m_s": .001,
        "friction": "two tangent box bounds; no torsion/rolling; zero restitution",
        "integration": "MuJoCo smooth dynamics and Euler integration; no DART collision detector",
        "distinction": "lcp engine remains the coupled associated friction-pyramid QP, not DART boxed friction"}
    metadata["contactsdf"] = {
        "source": "asu-iris/ContactSDF 3a21edf8e51788f72a7d77259af4dd584d5ec43f dynamics/lse_dynamics.py",
        "sigma": getattr(args, "contactsdf_sigma", 120.),
        "model": "dynamic extension of upstream LSE projection; Q=M/h^2; b=M*v/h+f_smooth",
        "collision": "MuJoCo CPU active point contacts; not upstream C-SDF geometry",
        "mass": "full current generalized mass including COM offset and angular coupling",
        "limitation": "untrained physical-mass dynamic extension; not the pretrained upstream quasi-dynamic baseline"}
    metadata['diagonal_contact'] = {
        'law': 'one ordered normal/tangent pass, p_i=project(-J_i*v_current/(diag(J*Minv*J.T)_i+r))',
        'collision': 'MuJoCo CPU; full current generalized mass',
        'prior': 'physical normal/tangent diagonal Delassus; no learned metric or time-history update',
        'passes': 1, 'regularization': args.scm_regularization,
        'friction': 'two independent tangent box bounds; not a Coulomb disk or associated friction-pyramid QP',
        'gap': 'max(phi,0)/h; no negative-gap restoration gain',
        'LM': 'frozen geometry: finite piecewise-affine velocity map in candidate force',
        'limitations': 'contact-order dependent; only one pass, no convergence guarantee; no penetration restoration'}
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
                M = full_mass(model, start)[:6, :6]
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
                reference = predictions[args.reference_engine][0]
                if reference is not None:
                    ref_native = predictions[args.reference_engine][1]
                    input_obs = contact_observations(model, start, start.qvel, ref_native)
                    mode = reference_mode(ref_native, input_obs, force_threshold=args.force_threshold,
                                          slip_threshold=args.slip_threshold, normal_threshold=args.normal_threshold)
                    if mode == "separation":
                        mode = "transition"
                else:
                    mode = "invalid_reference"
                reference_contact_count = len(ref_native["gap"]) if reference is not None else 0
                if window is not None and sample_index == 0 and reference is not None:
                    preview_mode(window, model, display, backends[args.reference_engine], qpos, qvel, wrench,
                                 requested, mode, args)
                # Keep a paired SCM-vs-IsaacGym row as well as the common
                # MuJoCo-reference rows.  This makes the two reference cells
                # in Table IV directly comparable without rerunning samples.
                scm_engine = scm_engine_name(e for e in predictions if predictions[e][0] is not None)
                if scm_engine and "isaacgym" in predictions:
                    scm_data = predictions[scm_engine][0]
                    isaac_data, isaac_native, _, isaac_mode, isaac_failure, _ = predictions["isaacgym"]
                    if scm_data is not None and isaac_data is not None:
                        # Same input-state mode and reference contact-count
                        # grouping as the MuJoCo rows, so the two Table IV
                        # comparisons share identical sample bins.
                        pair = {"sample_id": sample_id, "shape": shape, "friction": friction,
                                "timestep_s": timestep, "requested_mode": requested,
                                "condition": condition, "reference_mode": mode,
                                "reference_outcome_mode": isaac_mode, "engine": "scm_vs_isaacgym",
                                "predicted_mode": predictions[scm_engine][3], "status": "ok",
                                "failure": isaac_failure, "step_seconds": predictions[scm_engine][5],
                                "isaacgym_fallback": bool(getattr(backends["isaacgym"], "isaacgym_fallback", False))}
                        pair.update(prediction_metrics(start, isaac_data, scm_data, M,
                                                       free_velocity, args))
                        pair["ncontact"] = len(isaac_native["gap"])
                        pair["reference_ncontact"] = reference_contact_count
                        pair["contact_count_group"] = contact_count_group(reference_contact_count)
                        rows.append(pair)
                for engine, (data, native, obs, predicted_mode, failure, elapsed) in predictions.items():
                    row = {"sample_id": sample_id, "shape": shape, "friction": friction, "timestep_s": timestep,
                           "requested_mode": requested, "condition": condition, "reference_mode": mode,
                           "reference_outcome_mode": predictions[args.reference_engine][3],
                           "engine": engine, "predicted_mode": predicted_mode, "status": "ok", "failure": failure,
                           "step_seconds": elapsed}
                    if engine in ("scm_corrected", "scm_soft_corrected") and data is not None:
                        row["scm_amplitude_scale"] = backends[engine].last_amplitude_scale
                    if engine == "scm" and data is not None:
                        lipschitz = backends[engine].api.last_lipschitz
                        row["scm_lipschitz"] = None if lipschitz is None else float(lipschitz)
                        row["scm_step_size"] = None if not lipschitz else 1. / float(lipschitz)
                        initial = backends[engine].api.last_initial_impulse
                        row["scm_initial_impulse_N_s"] = None if initial is None else float(np.sum(initial))
                        row["scm_impulse_N_s"] = float(np.sum(backends[engine].api.last_problem["impulse"]))
                    if engine in ("scm_residual", "scm_preconditioned") and data is not None:
                        lipschitz, step_size = backends[engine].last_scm_lipschitz, backends[engine].last_scm_step_size
                        row["scm_lipschitz"] = None if lipschitz is None else float(lipschitz)
                        row["scm_step_size"] = None if step_size is None else float(step_size)
                        row["scm_power_iterations"] = int(backends[engine].last_scm_power_iterations)
                        row["scm_spectral_source"] = backends[engine].last_scm_spectral_source
                    if engine == "lcp" and data is not None:
                        row.update(lcp_residual=backends[engine].api.last_lcp["residual"],
                                   lcp_iterations=backends[engine].api.last_lcp["iterations"])
                    if engine == "dart" and data is not None:
                        row.update(dart_converged=backends[engine].api.last_lcp["converged"],
                                   dart_iterations=backends[engine].api.last_lcp["iterations"])
                    if reference is None or data is None:
                        row["status"] = "invalid_reference" if reference is None else "failed"
                    else:
                        row.update(prediction_metrics(start, reference, data, M, free_velocity, args))
                        loaded = native["force"][:, 0] > args.force_threshold
                        row.update(ncontact=len(native["gap"]),
                                   reference_ncontact=reference_contact_count,
                                   contact_count_group=contact_count_group(reference_contact_count),
                                   loaded_contacts=int(loaded.sum()),
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
            labels = Counter(r["reference_mode"] for r in rows[mode_start:] if r["engine"] == args.reference_engine)
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
    rollout_rows = run_rollouts(model, settled, backends, args, shape, friction, rng)
    if rollout_rows:
        with (output / "rollouts.jsonl").open("w") as stream:
            for row in rollout_rows:
                stream.write(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                         for k, v in row.items()}, allow_nan=False) + "\n")
    report = {"metadata": metadata, "stopped_by_viewer": stopped, "requested_shared_states": len(args.modes) * args.samples_per_mode,
              "completed_shared_states": len(saved) // len(engines),
              "requested_vs_reference_modes": dict(Counter(f"{r['requested_mode']}->{r['reference_mode']}" for r in rows if r["engine"] == args.reference_engine)),
              "by_reference_mode": aggregate(rows, moments, post_moments),
              "rollouts": aggregate_rollouts(rollout_rows, args.reference_engine)}
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
                        help="Independent reference-backend animation before each mode; excluded from statistics. 0 disables it")
    parser.add_argument("--display-pause", type=nonnegative_float, default=.02,
                        help="Wall-clock hold per displayed one-step prediction [s]")
    parser.add_argument("--reference-engine", choices=("mujoco", "isaacgym"), default="mujoco",
                        help="Ground-truth backend. MuJoCo is the default and is always included.")
    parser.add_argument("--engines", nargs="+", choices=(
        "mjwarp", "comfree", "scm_corrected", "scm_soft", "scm_soft_corrected",
        "scm_residual", "scm_preconditioned", "lcp", "dart", "contactsdf",
        "diagonal_contact", "scm", "scm_box", "mujoco", "isaacgym"), default=["scm", "isaacgym"])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--shapes", nargs="+", type=canonical_shape, choices=(*SHAPES, *MESH_SHAPES),
                        default=["box", "sphere"], help="Primitives or repository STL objects; stanford_bunn2 aliases stanford_bunny2")
    parser.add_argument("--samples-per-mode", type=positive_int, default=100, help="Shared states per shape/friction/timestep and requested mode")
    parser.add_argument("--frictions", nargs="+", type=positive_float, default=[.5])
    parser.add_argument("--timesteps", nargs="+", type=positive_float, default=[.002],
                        help="Integration steps; TABLE IV uses h=0.002 s.")
    parser.add_argument("--mass", type=positive_float, default=.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--settle-steps", type=positive_int, default=500,
                        help="Initial reference settling steps; mesh objects extend up to 5x if still moving")
    parser.add_argument("--contact-stiffness", type=nonnegative_float, default=.1)
    parser.add_argument("--contact-damping", type=nonnegative_float, default=.001)
    parser.add_argument("--scm-regularization", type=positive_float, default=1e-8)
    parser.add_argument("--scm-motion", choices=("dynamic", "increment"), default="dynamic")
    parser.add_argument("--scm-response-time", type=positive_float, default=.008,
                        help="Soft SCM relaxation time [s]; default calibrated for the 2 ms physics benchmark")
    parser.add_argument("--scm-damping-ratio", type=positive_float, default=.25,
                        help="Soft SCM damping ratio; benchmark default .25 (MPC model defaults are separate)")
    parser.add_argument("--scm-softplus-temperature", type=positive_float, default=1e-3,
                        help="Softplus temperature T in N s for the paper SCM (`scm`, method.tex T=1e-3) "
                             "and the softplus SCM variants")
    parser.add_argument("--scm-sweeps", type=positive_int, default=3,
                        help="Unrolled Gauss-Seidel sweeps of the smooth 3-row Coulomb-box SCM (`scm_box`)")
    parser.add_argument("--scm-soft-contact", choices=("none", "mujoco"), default="none",
                        help="`scm_box` only: 'mujoco' aligns the row targets/compliance with "
                             "MuJoCo solref/solimp (aref = -b J v - k pos, regularizer R)")
    parser.add_argument("--scm-correction-steps", type=nonnegative_int, default=1,
                        help="Residual applications per prediction for scm_residual and scm_preconditioned; the published model uses 1")
    parser.add_argument("--contactsdf-sigma", type=positive_float, default=120.,
                        help="LSE sharpness for the dynamic ContactSDF extension (Q=M/h^2)")
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
    parser.add_argument("--rollout-samples", type=nonnegative_int, default=0,
                        help="Rotation/flipping trials per shape. 0 disables rollout windows.")
    parser.add_argument("--rollout-steps", type=positive_int, default=250,
                        help="Maximum steps in each rotation/flipping prediction rollout.")
    parser.add_argument("--update-table-iv", action="store_true",
                        help="Write table_iv.csv/.md and replace the velocity table placeholders in experiment.tex")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Record a failed shape/configuration and continue the batch")
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if args.update_table_iv and any(abs(float(t) - .002) > 1e-12 for t in args.timesteps):
        parser.error("TABLE IV requires h=0.002 s")
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
                try:
                    reports[key] = run_configuration(shape, friction, timestep, args, rng, args.output_dir / key)
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    print(f"{key}: skipped after {type(exc).__name__}: {exc}", flush=True)
                    reports[key] = {"metadata": {"shape": shape, "friction": friction,
                                                  "timestep_s": timestep, "reference": args.reference_engine},
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "by_reference_mode": {}, "rollouts": {}}
    import sys
    try:
        import warp
        warp_version = warp.__version__
    except ImportError:
        warp_version = None
    arguments = {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items() if not key.startswith("_")}
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "experiment": "mujoco_ground_truth_contact_prediction_error",
        "arguments": arguments, "python": sys.executable, "mujoco": mujoco.__version__, "warp": warp_version,
        "mode_definition": "input contact-point velocities at solve pose, with MuJoCo solve-time normal load; outcome modes use post-impulse velocity separately",
        "upstream_commit": UPSTREAM_COMMIT, "reference": args.reference_engine,
        "paper_section_iv": {"A": "all-contact penetration population mean/std",
                             "B": "torsional angular-speed and rolling COM-speed decay curves; separate compare_physics scenes",
                             "C": "horizontal speed, height and kinetic energy diagnostics",
                             "D": "full-step synchronized wall time, excluding readback/rendering, single world here",
                             "E": "parallel throughput is not measured by this single-step experiment"},
        "backend_status": {key: report.get("metadata", {}).get("backends", {})
                           for key, report in reports.items()},
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
                for count, subgroup in group.get("by_contact_count", {}).items():
                    count_row = {"configuration": configuration, "engine": engine,
                                 "reference_mode": mode, "contact_count": count,
                                 "attempted": subgroup["attempted"], "valid": subgroup["valid"],
                                 "failed": 0}
                    for metric, statistics in subgroup["metrics"].items():
                        count_row.update({f"{metric}_{stat}": value
                                          for stat, value in statistics.items()})
                    summary.append(count_row)
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
    table_iv = write_table_iv(args.output_dir, args)
    (args.output_dir / "table_iv.json").write_text(json.dumps(table_iv, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(reports)} configurations to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
