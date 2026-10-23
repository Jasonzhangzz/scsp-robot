"""Shared, single-world backends and output for the migrated ComFree benchmarks.

MJWarp means the baseline vendored by ComFree at the pinned upstream revision,
so the two GPU engines use the same collision/integration implementation.
Trajectory logging and paired metrics are additions made in this project.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import time
from pathlib import Path
import sys

if __package__:
    from .runtime_env import activate_dependencies
else:
    from runtime_env import activate_dependencies

activate_dependencies()

import mujoco
import numpy as np


UPSTREAM_COMMIT = "ba8b9969f5f7859d51b1f738f458f8d2f49be080"


class Backend:
    """Keep a GPU trajectory on device; copy back only state for diagnostics."""

    def __init__(self, engine, model, initial_data, *, device="cuda:0",
                 stiffness=0.2, damping=0.001, nconmax=2048, njmax=16384,
                 scm_regularization=1e-8, scm_motion="dynamic", scm_friction="point",
                 scm_response_time=.05, scm_damping_ratio=1.0,
                 scm_softplus_temperature=1e-4, scm_correction_steps=1,
                 lcp_regularization=1e-8, lcp_tolerance=1e-8, lcp_max_iterations=2000,
                 lcp_solver="lemke", lcp_law="convex", lcp_gap_policy="signed",
                 contactsdf_sigma=120.,
                 collect_contacts=False):
        self.engine, self.model = engine, model
        self.collect_contacts = collect_contacts
        if engine == "mujoco":
            return
        if engine in ("dart", "contactsdf", "diagonal_contact"):
            if __package__:
                from .contact_dynamics import CpuContactEngine
            else:
                from contact_dynamics import CpuContactEngine
            self.api = CpuContactEngine(model, engine, sigma=contactsdf_sigma,
                                       regularization=scm_regularization)
            self.cpu_contact = True
            return
        if model.nv > 60 and engine == "mjwarp":
            if __package__:
                from .upstream_compat import patch_sparse_newton
            else:
                from upstream_compat import patch_sparse_newton
            patch_sparse_newton()
        try:
            import warp as wp
            import comfree_warp
            from comfree_warp import mujoco_warp
        except ImportError as exc:
            raise RuntimeError("Install this directory's requirements.txt into .deps using pi0 Python first.") from exc
        if engine not in ("comfree", "mjwarp", "scm_corrected", "scm_soft", "scm_soft_corrected",
                          "scm_residual", "scm_preconditioned", "lcp"):
            raise ValueError(f"Unknown engine: {engine}")
        if mujoco.__version__ != "3.6.0":
            raise RuntimeError(f"Pinned ComFree requires MuJoCo 3.6.0; found {mujoco.__version__}. Reinstall requirements.txt into .deps.")
        wp.init()
        self.wp, self.device = wp, wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("The pinned upstream GPU kernels require a CUDA device (for example cuda:0).")
        self.nconmax, self.njmax = nconmax, njmax
        with wp.ScopedDevice(self.device):
            self.api = comfree_warp if engine == "comfree" else mujoco_warp
            if engine in ("scm_corrected", "scm_soft", "scm_soft_corrected"):
                if engine == "scm_soft_corrected" and (
                        not np.isfinite(scm_softplus_temperature) or scm_softplus_temperature <= 0):
                    raise ValueError("scm_soft_corrected requires positive finite scm_softplus_temperature (N s)")
                if __package__:
                    from .scm_backend import ScmEngine
                else:
                    from scm_backend import ScmEngine
                self.api = ScmEngine(model, initial_data, regularization=scm_regularization,
                                     motion=scm_motion, friction=scm_friction,
                                     amplitude_correction=engine in ("scm_corrected", "scm_soft_corrected"),
                                     response_time=scm_response_time if engine == "scm_soft" else 0.,
                                     softplus_temperature=scm_softplus_temperature if engine == "scm_soft_corrected" else 0.,
                                     damping_ratio=scm_damping_ratio)
            if engine in ("scm_residual", "scm_preconditioned"):
                if not np.isfinite(scm_softplus_temperature) or scm_softplus_temperature <= 0:
                    raise ValueError(f"{engine} requires positive finite scm_softplus_temperature (N s)")
                if scm_correction_steps < 0:
                    raise ValueError("scm_correction_steps must be nonnegative")
                if __package__:
                    from .residual_scm import ResidualScmEngine
                else:
                    from residual_scm import ResidualScmEngine
                self.api = ResidualScmEngine(
                    model, initial_data, regularization=scm_regularization, motion=scm_motion,
                    friction=scm_friction, softplus_temperature=scm_softplus_temperature,
                    correction="spectral" if engine == "scm_residual" else "preconditioned",
                    correction_steps=scm_correction_steps)
            if engine == "lcp":
                if __package__:
                    from .lcp_backend import LcpEngine
                else:
                    from lcp_backend import LcpEngine
                self.api = LcpEngine(model, initial_data, regularization=lcp_regularization,
                                     tolerance=lcp_tolerance, max_iterations=lcp_max_iterations,
                                     solver=lcp_solver, law=lcp_law, gap_policy=lcp_gap_policy)
            kwargs = dict(comfree_stiffness=stiffness, comfree_damping=damping) if engine == "comfree" else {}
            self.wmodel = self.api.put_model(model, **kwargs)
            # The pinned upstream sensor module has an undefined xmat in its
            # UNKNOWN-object branch. Warp 1.16 compiles even zero-size sensor
            # launches. These scenes have no sensors, so skip that empty stage
            # for both engines without changing their dynamics/contact code.
            if model.nsensor == 0:
                self.wmodel.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
                # GPU energy diagnostics compile the same faulty sensor module.
                # Logged energies are recomputed on CPU at the resulting state.
                self.wmodel.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_ENERGY)
            self.wdata = self.api.put_data(model, initial_data, nworld=1, nconmax=nconmax, njmax=njmax)
            # Compile on disposable data; the measured run starts at the exact initial state.
            self.api.step(self.wmodel, self.wdata)
            self.api.step(self.wmodel, self.wdata)
            wp.synchronize_device(self.device)
            self.wdata = self.api.put_data(model, initial_data, nworld=1, nconmax=nconmax, njmax=njmax)

    def step(self, data):
        if getattr(self, "cpu_contact", False):
            started = time.perf_counter()
            self.api.step(data)
            self.last_step_seconds = time.perf_counter() - started
        elif self.engine == "mujoco":
            started = time.perf_counter()
            mujoco.mj_step(self.model, data)
            self.last_step_seconds = time.perf_counter() - started
        else:
            wp = self.wp
            with wp.ScopedDevice(self.device):
                for name in ("ctrl", "qfrc_applied", "xfrc_applied"):
                    value = np.asarray(getattr(data, name), dtype=np.float32)[None, ...]
                    if value.size:
                        target = getattr(self.wdata, name)
                        wp.copy(target, wp.array(value, dtype=target.dtype, device=self.device))
                if self.engine in ("scm_corrected", "scm_soft", "scm_soft_corrected",
                                   "scm_residual", "scm_preconditioned", "lcp"):
                    self.api.update_mass(data)
                wp.synchronize_device(self.device)
                started = time.perf_counter()
                self.api.step(self.wmodel, self.wdata)
                wp.synchronize_device(self.device)
                self.last_step_seconds = time.perf_counter() - started
                if self.engine in ("scm_corrected", "scm_soft_corrected"):
                    self.last_amplitude_scale = float(self.api.amplitude.numpy()[0])
                if self.engine in ("scm_residual", "scm_preconditioned"):
                    self.last_scm_lipschitz = self.api.last_lipschitz
                    self.last_scm_step_size = self.api.last_step_size
                    self.last_scm_power_iterations = self.api.last_power_iterations
                    self.last_scm_spectral_source = self.api.last_spectral_source
                ncontact = int(self.wdata.nacon.numpy()[0])
                nconstraint = int(self.wdata.nefc.numpy()[0])
                ncollision = int(self.wdata.ncollision.numpy()[0])
                if (ncontact > self.wdata.naconmax or ncollision > self.wdata.naconmax
                        or nconstraint > self.wdata.njmax):
                    raise RuntimeError(f"{self.engine} buffer overflow: {ncollision} collision pairs, "
                                       f"{ncontact} contacts, {nconstraint} constraints. "
                                       "Increase --nconmax/--njmax; this trial is invalid.")
                data.qpos[:] = self.wdata.qpos.numpy()[0]
                data.qvel[:] = self.wdata.qvel.numpy()[0]
                if self.model.na:
                    data.act[:] = self.wdata.act.numpy()[0]
                data.time = float(self.wdata.time.numpy()[0])
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.time)):
            raise RuntimeError(f"{self.engine} produced a nonfinite state at t={data.time}.")
        if self.collect_contacts:
            self.last_contacts = self.solver_contacts(data)
        # Use the same CPU collision diagnostics at the resulting state for all engines.
        # This does not feed positions or MuJoCo constraint forces back to the GPU.
        mujoco.mj_forward(self.model, data)

    def reset(self, data):
        """Reset shared-state probes, including all solver warm-start state."""
        if self.engine != "mujoco" and not getattr(self, "cpu_contact", False):
            with self.wp.ScopedDevice(self.device):
                self.wdata = self.api.put_data(self.model, data, nworld=1,
                                               nconmax=self.nconmax, njmax=self.njmax)

    def solver_contacts(self, data):
        """Native solve-time geometry and forces, before CPU diagnostic forward."""
        if getattr(self, "cpu_contact", False):
            return self.api.last_contacts
        if self.engine == "mujoco":
            count = data.ncon
            force = np.zeros((count, 6))
            for i in range(count):
                mujoco.mj_contactForce(self.model, data, i, force[i])
            return {"gap": np.asarray(data.contact.dist[:count]).copy(),
                    "position": np.asarray(data.contact.pos[:count]).copy(),
                    "frame": np.asarray(data.contact.frame[:count]).reshape(-1, 3, 3).copy(),
                    "geom": np.asarray(data.contact.geom[:count]).copy(),
                    "friction": np.asarray(data.contact.friction[:count]).copy(),
                    "dim": np.asarray(data.contact.dim[:count]).copy(), "force": force}
        count = int(self.wdata.nacon.numpy()[0])
        contact = self.wdata.contact
        dimension = contact.dim.numpy()[:count]
        force = np.zeros((count, 6))
        coefficients = self.wdata.efc.force.numpy()[0]
        addresses = contact.efc_address.numpy()[:count]
        friction = contact.friction.numpy()[:count]
        for i, dim in enumerate(dimension):
            adr = int(addresses[i, 0])
            if adr < 0:
                continue
            if self.model.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL:
                raise ValueError("Native force decoding currently requires the common pyramidal cone.")
            if dim == 1:
                force[i, 0] = coefficients[adr]
            else:
                pyramid = coefficients[adr:adr + 2 * (int(dim) - 1)].reshape(-1, 2)
                force[i, 0] = pyramid.sum()
                force[i, 1:int(dim)] = (pyramid[:, 0] - pyramid[:, 1]) * friction[i, :int(dim) - 1]
        return {"gap": contact.dist.numpy()[:count], "position": contact.pos.numpy()[:count],
                "frame": contact.frame.numpy()[:count], "geom": contact.geom.numpy()[:count],
                "friction": friction, "dim": dimension, "force": force}


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative_float(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def add_common_arguments(parser, *, include_scm=False):
    parser.add_argument("--headless", action="store_true", help="Disable the visualization window (shown by default)")
    choices = ("both", "all", "mujoco", "mjwarp", "comfree") + (("three", "four", "scm_corrected", "scm_soft") if include_scm else ())
    parser.add_argument("--engine", choices=choices, default="three" if include_scm else "both",
                        help="both=ComFree+MJWarp; three adds SCM; four adds amplitude-corrected SCM; five adds soft SCM (physics only); all also includes MuJoCo CPU")
    parser.add_argument("--device", default="cuda:0", help="CUDA device used by both Warp engines")
    parser.add_argument("--contact-stiffness", type=nonnegative_float, default=0.2)
    parser.add_argument("--contact-damping", type=nonnegative_float, default=0.001)
    parser.add_argument("--nconmax", type=positive_int, default=2048)
    parser.add_argument("--njmax", type=positive_int, default=16384)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")


def engines_for(choice, *, include_scm=False):
    if choice == "scm":
        raise ValueError("Uncorrected SCM was removed; use scm_corrected")
    return {"both": ("mjwarp", "comfree"), "three": ("mjwarp", "comfree", "scm_corrected"),
            "four": ("mjwarp", "comfree", "scm_corrected", "scm_soft"),
            "all": ("mujoco", "mjwarp", "comfree") + (("scm_corrected", "scm_soft") if include_scm else ())}.get(choice, (choice,))


def contact_metrics(model, data):
    """Geometric depth at the current state, recomputed by CPU mj_forward."""
    distances = np.asarray(data.contact.dist[:data.ncon])
    return {"ncontact": int(data.ncon),
            "max_penetration_m": float(max(0.0, -distances.min())) if distances.size else 0.0}


def run_metadata(args, model_path, model):
    versions = {"mujoco": mujoco.__version__, "numpy": np.__version__}
    for package in ("warp-lang", "comfree_warp"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"arguments": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
            "model_path": str(Path(model_path).resolve()), "timestep": float(model.opt.timestep),
            "nq": int(model.nq), "nv": int(model.nv), "nworld": 1,
            "python_executable": sys.executable, "python_environment": sys.prefix,
            "jacobian_format": mujoco.mjtJacobian(model.opt.jacobian).name,
            "ccd_iterations": int(model.opt.ccd_iterations),
            "upstream_commit": UPSTREAM_COMMIT, "versions": versions,
            "mjwarp_source": "comfree_warp.mujoco_warp (upstream vendored baseline)",
            "empty_sensor_stage_skipped": bool(model.nsensor == 0),
            "sparse_newton_compile_fix": bool(model.nv > 60 and "mjwarp" in engines_for(args.engine)),
            "penetration_definition": "max(0, -contact.dist) at each post-step state, CPU collision diagnostics",
            "comparison_definition": "independent rollouts from identical initial state and prescribed inputs"}


def _write_csv(path, arrays):
    columns = []
    labels = []
    for name, value in arrays.items():
        value = np.asarray(value)
        if value.ndim == 1:
            labels.append(name)
            columns.append(value[:, None])
        else:
            flat = value.reshape(value.shape[0], -1)
            labels.extend(f"{name}_{i}" for i in range(flat.shape[1]))
            columns.append(flat)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(labels)
        writer.writerows(np.concatenate(columns, axis=1))


def save_results(output_dir, trajectories, summaries, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for engine, arrays in trajectories.items():
        np.savez_compressed(output_dir / f"{engine}.npz", **arrays)
        _write_csv(output_dir / f"{engine}.csv", arrays)
    comparisons = {}
    if "mjwarp" in trajectories:
        reference = trajectories["mjwarp"]
        for engine, trajectory in trajectories.items():
            if engine == "mjwarp":
                continue
            count = min(len(reference["time"]), len(trajectory["time"]))
            if not np.allclose(reference["time"][:count], trajectory["time"][:count], rtol=1e-4, atol=1e-6):
                raise ValueError("Trajectories have different time grids; paired metrics would be invalid.")
            diff = trajectory["qvel"][:count] - reference["qvel"][:count]
            comparison = {"reference": "mjwarp", "compared_samples": count,
                          "qvel_rmse_per_dof": np.sqrt(np.mean(diff ** 2, axis=0)).tolist(),
                          "qvel_max_abs_difference_per_dof": np.max(np.abs(diff), axis=0).tolist()}
            if "cube_position" in trajectory:
                distances = np.linalg.norm(trajectory["cube_position"][:count] - reference["cube_position"][:count], axis=1)
                comparison.update(cube_position_rmse_m=float(np.sqrt(np.mean(distances ** 2))),
                                  cube_position_max_difference_m=float(distances.max()))
            if "body_position_m" in trajectory:
                distances = np.linalg.norm(trajectory["body_position_m"][:count] - reference["body_position_m"][:count], axis=-1)
                comparison.update(body_position_rmse_m=float(np.sqrt(np.mean(distances ** 2))),
                                  max_body_position_difference_m=float(distances.max()),
                                  final_max_body_position_difference_m=float(distances[-1].max()))
            comparison["max_penetration_difference_m"] = float(np.max(np.abs(
                trajectory["max_penetration_m"][:count] - reference["max_penetration_m"][:count])))
            comparisons[f"{engine}_vs_mjwarp"] = comparison
    report = {"metadata": metadata, "engines": summaries, "comparisons": comparisons}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    concise = {pair: {key: value for key, value in metrics.items() if not key.endswith("_per_dof")}
               for pair, metrics in comparisons.items()}
    print(json.dumps({"engines": summaries, "comparisons": concise}, indent=2, allow_nan=False))
    print(f"Saved results to {output_dir.resolve()}")
