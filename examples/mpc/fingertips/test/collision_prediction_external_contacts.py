"""SCM and pinned ComFree predictions from externally supplied PhysX contacts.

This module never runs MuJoCo/MJWarp collision detection. MuJoCo supplies only
smooth rigid-body dynamics, Jacobians and the existing Euler integration law;
the pinned ComFree ``run_collision_detection=False`` interface accepts the
same external manifolds on CUDA. It does not modify PhysX's internal solver.

Canonical body IDs in the portable manifold are floor=0, object=1, ball=2.
``frame`` has rows (normal, tangent1, tangent2), with the normal pointing from
``body[:, 0]`` to ``body[:, 1]``. Positions are in the query's world frame and
gaps remain physical, signed surface separations (negative for overlap).
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from examples.mpc.franka.scm.runtime_env import activate_dependencies
activate_dependencies()

import mujoco
import numpy as np

from examples.mpc.franka.scm.contact_dynamics import full_mass
from examples.mpc.fingertips.test.collision_prediction_core import Settings, native_diagnostics
from models.soft_box_contact import soft_box_contact_velocity_numpy


def smooth_data(model, sample=None):
    """Populate only smooth dynamics; deliberately omit mj_fwdPosition/collision.

    The benchmark consists of two unactuated free bodies, with no tendons,
    flexes, equalities or other articulated mechanisms. The explicit stages
    below reproduce its mass matrix and qacc_smooth without any contact solve.
    """
    data = mujoco.MjData(model)
    if sample is not None:
        data.qpos[:] = sample["qpos"]
        data.qvel[:] = sample["qvel"]
        data.xfrc_applied[:] = sample["wrench"]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_crb(model, data)
    mujoco.mj_makeM(model, data)
    mujoco.mj_factorM(model, data)
    mujoco.mj_fwdVelocity(model, data)
    mujoco.mj_fwdActuation(model, data)
    mujoco.mj_fwdAcceleration(model, data)
    data.qacc_warmstart[:] = 0.
    return data


class ExternalContactPredictors:
    """Independent 20-ms predictions sharing only an initial PhysX manifold.

    CUDA initialization is lazy, so constructing this class and using SCM does
    not require a GPU. Each call allocates/reset its state and never consumes
    a native PhysX contact force or a future reference manifold.
    """

    def __init__(self, model_path, settings=Settings()):
        self.settings = settings
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        model = self.model
        if ((model.nq, model.nv, model.nbody) != (14, 12, 3) or model.nu or model.neq
                or model.ntendon or model.nflex
                or model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER):
            raise ValueError("External contacts require the unactuated two-free-body Euler scene")
        self.body_ids = np.array([0, model.body("object").id, model.body("ball").id], dtype=int)
        if not np.array_equal(self.body_ids, [0, 1, 2]):
            raise ValueError("The model's canonical body order must be floor/object/ball")
        if (not np.isfinite(settings.horizon) or settings.horizon <= 0
                or settings.nconmax < 1 or settings.njmax < 4):
            raise ValueError("Positive prediction horizon and contact capacities are required")
        model.opt.timestep = settings.horizon
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_ENERGY)
        self.scm_model = copy.copy(model)
        self.scm_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.representative_geoms = np.array([
            self._representative_geom(body) for body in self.body_ids], dtype=int)
        self.comfree_model = None
        self.warmed_sizes = set()
        self.last_scm_problem = None
        self.last_comfree_timing = None

    def _representative_geom(self, body):
        ids = np.flatnonzero((self.model.geom_bodyid == body)
                            & ((self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0)))
        if not len(ids):
            raise ValueError(f"Body {body} has no collision geom")
        # PhysX reports rigid bodies, not the original MuJoCo shape index. All
        # compound parts are on the same body; its inertial/Jacobian mapping is
        # therefore exact. Do not present this representative as a native ID.
        return int(ids[0])

    def _sample(self, sample):
        values = {}
        for key, shape in (("qpos", (14,)), ("qvel", (12,)), ("wrench", (3, 6))):
            value = np.asarray(sample[key], dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{key} must be finite with shape {shape}")
            values[key] = value.copy()
        for start in (3, 10):
            if not np.isclose(np.linalg.norm(values["qpos"][start:start+4]), 1., rtol=0., atol=1e-5):
                raise ValueError("Initial quaternions must be normalized")
        return values

    def _manifold(self, manifold):
        gap = np.asarray(manifold["gap"], dtype=float).reshape(-1)
        count = len(gap)
        result = {"gap": gap.copy()}
        for key, shape, dtype in (("position", (count, 3), float),
                                  ("frame", (count, 3, 3), float),
                                  ("body", (count, 2), int),
                                  ("friction", (count, 5), float)):
            raw = np.asarray(manifold[key])
            if raw.size != np.prod(shape) or not np.isfinite(raw).all():
                raise ValueError(f"Manifold {key} must be finite with shape {shape}")
            value = np.asarray(raw, dtype=dtype).reshape(shape)
            if key == "body" and not np.array_equal(value, raw.reshape(shape)):
                raise ValueError("Manifold body IDs must be integers")
            result[key] = value.copy()
        margin = np.asarray(manifold.get("include_margin", np.zeros(count)), dtype=float).reshape(-1)
        if margin.shape != (count,) or not np.isfinite(np.r_[gap, margin]).all() or np.any(margin < 0):
            raise ValueError("Finite signed gaps and nonnegative include_margin are required")
        result["include_margin"] = margin.copy()
        bodies, frame, friction = result["body"], result["frame"], result["friction"]
        if count > self.settings.nconmax or 4*count > self.settings.njmax:
            raise ValueError("External manifold exceeds nconmax/njmax; no contacts were truncated")
        if (np.any(bodies < 0) or np.any(bodies > 2) or np.any(bodies[:, 0] == bodies[:, 1])):
            raise ValueError("Canonical body IDs must identify distinct floor/object/ball bodies")
        if np.any(friction < 0) or not np.allclose(friction[:, 0], friction[:, 1], rtol=0, atol=1e-12):
            raise ValueError("Point-contact laws require equal nonnegative tangent friction")
        if count and (not np.allclose(frame @ frame.transpose(0, 2, 1), np.eye(3), atol=2e-5, rtol=0)
                      or not np.allclose(np.linalg.det(frame), 1., atol=2e-5, rtol=0)):
            raise ValueError("Contact frames must be right-handed and orthonormal")
        return result

    def _jacobians(self, data, manifold):
        count = len(manifold["gap"])
        jac = np.zeros((count, 3, self.scm_model.nv))
        for index, (position, bodies, frame) in enumerate(zip(
                manifold["position"], manifold["body"], manifold["frame"])):
            point = []
            for body in bodies:
                linear = np.zeros((3, self.scm_model.nv))
                mujoco.mj_jac(self.scm_model, data, linear, None, position, int(self.body_ids[body]))
                point.append(linear)
            jac[index] = frame @ (point[1] - point[0])
        return jac

    @staticmethod
    def _failed(exc, seconds=0.):
        return dict(status="failed", failure_reason=f"{type(exc).__name__}: {exc}",
                    qpos=None, qvel=None, time=None, seconds=float(seconds))

    def scm(self, sample, manifold):
        began = time.perf_counter()
        try:
            sample, manifold = self._sample(sample), self._manifold(manifold)
            data = smooth_data(self.scm_model, sample)
            mass = full_mass(self.scm_model, data)
            inverse = np.linalg.inv(mass)
            free = data.qvel + self.settings.horizon*data.qacc_smooth
            jac = self._jacobians(data, manifold)
            velocity, impulses = soft_box_contact_velocity_numpy(
                free, jac, manifold["gap"], inverse, self.settings.horizon,
                manifold["friction"][:, 0], temperature=self.settings.scm_temperature,
                regularization=self.settings.scm_regularization, sweeps=self.settings.scm_sweeps,
                return_impulses=True)
            self.last_scm_problem = dict(mass=mass, free_velocity=free, J=jac, impulse=impulses)
            data.qvel[:] = velocity
            mujoco.mj_integratePos(self.scm_model, data.qpos, data.qvel, self.settings.horizon)
            if not np.isfinite(np.r_[data.qpos, data.qvel]).all():
                raise RuntimeError("SCM produced a nonfinite prediction")
            return dict(status="ok", failure_reason="", qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                        time=float(self.settings.horizon), seconds=time.perf_counter()-began,
                        execution="cpu", actual_nworld=1)
        except Exception as exc:
            self.last_scm_problem = None
            return self._failed(exc, time.perf_counter()-began)

    def _ensure_comfree(self):
        if self.comfree_model is not None:
            return
        if mujoco.__version__ != "3.6.0":
            raise RuntimeError("External ComFree requires the benchmark's pinned MuJoCo 3.6.0")
        import warp as wp
        import comfree_warp
        wp.init()
        self.wp, self.api = wp, comfree_warp
        self.device = wp.get_device(self.settings.device)
        if not self.device.is_cuda:
            raise ValueError("Pinned ComFree requires a CUDA device")
        with wp.ScopedDevice(self.device):
            model = self.api.put_model(self.model, comfree_stiffness=self.settings.contact_stiffness,
                                      comfree_damping=self.settings.contact_damping)
            # Set this before the first forward/step, including warm-up.
            model.opt.run_collision_detection = False
            self.comfree_model = model

    def _contact_parameters(self, geoms):
        """MuJoCo's existing pair impedance mix; geometry comes only from PhysX."""
        count = len(geoms)
        solref, solimp = np.zeros((count, 2)), np.zeros((count, 5))
        for index, (first, second) in enumerate(geoms):
            priorities = self.model.geom_priority[[first, second]]
            if priorities[0] != priorities[1]:
                chosen = first if priorities[0] > priorities[1] else second
                solref[index], solimp[index] = self.model.geom_solref[chosen], self.model.geom_solimp[chosen]
            else:
                mix = self.model.geom_solmix[[first, second]]
                weight = .5 if mix.sum() < mujoco.mjMINVAL else mix[0]/mix.sum()
                first_ref, second_ref = self.model.geom_solref[first], self.model.geom_solref[second]
                solref[index] = (weight*first_ref + (1-weight)*second_ref
                                 if min(first_ref[0], second_ref[0]) > 0
                                 else np.minimum(first_ref, second_ref))
                solimp[index] = weight*self.model.geom_solimp[first] + (1-weight)*self.model.geom_solimp[second]
        return solref, solimp

    def _new_comfree_data(self, samples, manifolds):
        wp = self.wp
        data = self.api.put_data(self.model, smooth_data(self.scm_model), nworld=len(samples),
                                 nconmax=self.settings.nconmax, njmax=self.settings.njmax)

        def upload(target, values):
            wp.copy(target, wp.array(values, dtype=target.dtype, device=self.device))

        for key, sample_key in (("qpos", "qpos"), ("qvel", "qvel"), ("xfrc_applied", "wrench")):
            upload(getattr(data, key), np.asarray([sample[sample_key] for sample in samples], dtype=np.float32))
        data.time.zero_()
        data.qacc_warmstart.zero_()
        data.ncollision.zero_()
        data.nacon.zero_()
        data.nefc.zero_()
        data.efc.force.zero_()
        data.contact.efc_address.fill_(-1)
        data.contact.type.zero_()
        count = sum(len(manifold["gap"]) for manifold in manifolds)
        if count > data.naconmax:
            raise ValueError("External contacts overflow the allocated batch contact buffer")
        if count:
            bodies = np.concatenate([item["body"] for item in manifolds])
            geoms = self.representative_geoms[bodies]
            solref, solimp = self._contact_parameters(geoms)
            gap = np.concatenate([item["gap"] for item in manifolds])
            margin = np.concatenate([item["include_margin"] for item in manifolds])
            # Native PhysX records are already accepted collision candidates.
            # ComFree activates dist < includemargin strictly, so preserve the
            # native margin and add only a float32-safe activation epsilon for
            # touching/candidate points on its exact boundary. Dist is untouched.
            margin = np.maximum(margin, gap + np.maximum(1e-8, np.abs(gap)*5e-7))
            values = dict(dist=gap, pos=np.concatenate([item["position"] for item in manifolds]),
                          frame=np.concatenate([item["frame"] for item in manifolds]),
                          friction=np.concatenate([item["friction"] for item in manifolds]),
                          includemargin=margin, geom=geoms, solref=solref, solimp=solimp,
                          solreffriction=np.zeros((count, 2)), dim=np.full(count, 3),
                          worldid=np.repeat(np.arange(len(samples)), [len(item["gap"]) for item in manifolds]),
                          type=np.ones(count, dtype=int), geomcollisionid=np.arange(count))
            for key, value in values.items():
                target = getattr(data.contact, key)
                padded = np.zeros((data.naconmax,) + np.asarray(value).shape[1:], dtype=np.asarray(value).dtype)
                padded[:count] = value
                upload(target, padded)
        upload(data.nacon, np.array([count], dtype=np.int32))
        # Collision detection is skipped; this diagnostic counter denotes the
        # externally provided point records rather than native broadphase pairs.
        upload(data.ncollision, np.array([count], dtype=np.int32))
        return data

    def comfree_batch(self, samples, manifolds):
        if len(samples) != len(manifolds):
            raise ValueError("Samples and manifolds must have the same batch length")
        if not samples:
            self.last_comfree_timing = dict(nworld=0, physics_seconds=0., end_to_end_seconds=0.)
            return []
        began = time.perf_counter()
        outputs, prepared_samples, prepared_manifolds, indices = [None]*len(samples), [], [], []
        for index, (sample, manifold) in enumerate(zip(samples, manifolds)):
            try:
                prepared_samples.append(self._sample(sample))
                try:
                    prepared_manifolds.append(self._manifold(manifold))
                except Exception:
                    prepared_samples.pop()
                    raise
                indices.append(index)
            except Exception as exc:
                outputs[index] = self._failed(exc)
        nworld = len(indices)
        physics = None
        try:
            if nworld:
                self._ensure_comfree()
                with self.wp.ScopedDevice(self.device):
                    if nworld not in self.warmed_sizes:
                        warm = self._new_comfree_data(prepared_samples, prepared_manifolds)
                        self.api.step(self.comfree_model, warm)
                        self.wp.synchronize_device(self.device)
                        self.warmed_sizes.add(nworld)
                    data = self._new_comfree_data(prepared_samples, prepared_manifolds)
                    self.wp.synchronize_device(self.device)
                    with native_diagnostics() as warnings:
                        physics_began = time.perf_counter()
                        self.api.step(self.comfree_model, data)
                        self.wp.synchronize_device(self.device)
                        physics = time.perf_counter()-physics_began
                    if warnings:
                        raise RuntimeError("ComFree native warning: " + "; ".join(warnings))
                    if np.any(data.nefc.numpy() > data.njmax):
                        raise RuntimeError("ComFree constraint buffer overflow")
                    qpos, qvel, times = data.qpos.numpy(), data.qvel.numpy(), data.time.numpy()
                    for world, index in enumerate(indices):
                        if not np.isfinite(np.r_[qpos[world], qvel[world], times[world]]).all():
                            outputs[index] = self._failed(RuntimeError("Nonfinite ComFree prediction"))
                        elif abs(times[world]-self.settings.horizon) > 1e-7:
                            outputs[index] = self._failed(RuntimeError("ComFree prediction horizon mismatch"))
                        else:
                            outputs[index] = dict(status="ok", failure_reason="", qpos=qpos[world].astype(float),
                                                  qvel=qvel[world].astype(float), time=float(times[world]),
                                                  seconds=physics/nworld, batch_seconds=physics,
                                                  actual_nworld=nworld,
                                                  execution="gpu_batch" if nworld > 1 else "gpu_single")
        except Exception as exc:
            for index in indices:
                outputs[index] = self._failed(exc, time.perf_counter()-began)
        self.last_comfree_timing = dict(nworld=nworld, requested_nworld=len(samples), physics_seconds=physics,
                                       end_to_end_seconds=time.perf_counter()-began,
                                       n_failed=sum(item["status"] != "ok" for item in outputs))
        return outputs
