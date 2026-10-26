"""Native MJWarp multi-world rollout, including world-local contact decoding.

One compiled object model is shared across worlds. Different objects are
batched separately. No Python-thread or process parallelism is used.
"""
from __future__ import annotations

import time

import numpy as np

from examples.mpc.fingertips.test.collision_prediction_core import (
    make_data, native_diagnostics, native_reference_contacts, reference_snapshot)


class BatchedMjwarpReference:
    def __init__(self, experiment):
        self.experiment = experiment
        self.model = experiment.models["mjwarp"]
        self.backend = experiment.backends["mjwarp"]
        self.api, self.wp = self.backend.api, self.backend.wp
        self.settings = experiment.settings
        self.warmed_sizes = set()
        self.batches = []

    def new_data(self, samples):
        data = self.api.put_data(self.model, make_data(self.model), nworld=len(samples),
                                 nconmax=self.settings.nconmax, njmax=self.settings.njmax)
        # Use the same input construction as Experiment.reference/reset before
        # float32 upload. No state or force from another world is reused.
        poses = [make_data(self.model, sample["qpos"], sample["qvel"], sample["wrench"])
                 for sample in samples]
        for key in ("qpos", "qvel", "xfrc_applied"):
            target = getattr(data, key)
            values = np.asarray([getattr(pose, key) for pose in poses], dtype=np.float32)
            self.wp.copy(target, self.wp.array(values, dtype=target.dtype, device=self.backend.device))
        return data

    def native_contacts(self, data):
        """Contact arrays are GLOBAL; efc addresses and forces are WORLD LOCAL."""
        return native_reference_contacts(data)

    def rollout(self, samples):
        nworld = len(samples)
        with self.wp.ScopedDevice(self.backend.device):
            if nworld not in self.warmed_sizes:
                # Compile and exercise multi-world solver branches on disposable
                # data. Warm-up never contributes a measured state or warning.
                with native_diagnostics():
                    warm = self.new_data(samples)
                    self.api.step(self.backend.wmodel, warm)
                    self.api.forward(self.backend.wmodel, warm)
                    self.wp.synchronize_device(self.backend.device)
                self.warmed_sizes.add(nworld)
            started = time.perf_counter()
            data = self.new_data(samples)
            poses = [make_data(self.model, sample["qpos"], sample["qvel"], sample["wrench"])
                     for sample in samples]
            qpos = [np.array([pose.qpos for pose in poses])]
            qvel = [np.array([pose.qvel for pose in poses])]
            snapshots = [[] for _ in samples]
            physics_seconds = 0.
            with native_diagnostics() as warnings:
                for step in range(self.settings.reference_steps + 1):
                    self.wp.synchronize_device(self.backend.device)
                    physics_start = time.perf_counter()
                    if step < self.settings.reference_steps:
                        self.api.step(self.backend.wmodel, data)
                    else:
                        self.api.forward(self.backend.wmodel, data)
                    self.wp.synchronize_device(self.backend.device)
                    physics_seconds += time.perf_counter() - physics_start
                    native, worlds, addresses = self.native_contacts(data)
                    for world, sample in enumerate(samples):
                        pose = make_data(self.model, qpos[-1][world], qvel[-1][world], sample["wrench"])
                        keep = worlds == world
                        local = {key: value[keep] for key, value in native.items()}
                        snapshots[world].append(reference_snapshot(
                            self.model, pose, local, addresses[keep], step*self.settings.reference_dt))
                    if step < self.settings.reference_steps:
                        qpos.append(data.qpos.numpy().astype(float))
                        qvel.append(data.qvel.numpy().astype(float))
            if warnings:
                raise RuntimeError("MJWarp batch native collision warning: " + "; ".join(warnings))
            qpos, qvel = np.asarray(qpos), np.asarray(qvel)
            times = data.time.numpy()
            if (not np.isfinite(qpos).all() or not np.isfinite(qvel).all()
                    or not np.allclose(times, self.settings.horizon, rtol=0., atol=1e-7)):
                raise RuntimeError("MJWarp batch produced nonfinite states or incorrect horizon")
            results = []
            for world, sample in enumerate(samples):
                final = make_data(self.model, qpos[-1, world], qvel[-1, world], sample["wrench"])
                final.time = float(times[world])
                results.append(dict(reference=(final, snapshots[world], qpos[:, world], qvel[:, world]),
                                    error=None, execution="gpu_batch", fallback_reason="",
                                    batch_sample_ids=[item["_sample_id"] for item in samples]
                                    if all("_sample_id" in item for item in samples) else []))
            return results, physics_seconds, time.perf_counter()-started

    def evaluate(self, samples):
        if not samples:
            return []
        began = time.perf_counter()
        try:
            results, physics, elapsed = self.rollout(samples)
            self.batches.append(dict(nworld=len(samples), execution="gpu_batch", physics_seconds=physics,
                                     end_to_end_seconds=elapsed, serial_retries=0, fallback_reason=""))
            return results
        except Exception as exc:
            # Upstream GJK/EPA printf does not identify the offending world.
            # Split into smaller independent batches before falling back to
            # single-query execution. Never invalidate healthy worlds because
            # another world emitted an un-attributed warning.
            reason = f"{type(exc).__name__}: {exc}"
            if len(samples) > 2:
                record = dict(nworld=len(samples), execution="split_retry", physics_seconds=None,
                              end_to_end_seconds=None, serial_retries=0, fallback_reason=reason)
                self.batches.append(record)
                middle = len(samples)//2
                results = self.evaluate(samples[:middle]) + self.evaluate(samples[middle:])
                record.update(end_to_end_seconds=time.perf_counter()-began,
                              serial_retries=sum(item["execution"] == "serial_retry" for item in results))
                return results
            results = []
            for sample in samples:
                reference = None
                try:
                    with native_diagnostics() as warnings:
                        reference = self.experiment.reference(sample)
                    if warnings:
                        raise RuntimeError("MJWarp native collision warning: " + "; ".join(warnings))
                    error = None
                except Exception as serial_error:
                    error = f"{type(serial_error).__name__}: {serial_error}"
                results.append(dict(reference=reference, error=error, execution="serial_retry", fallback_reason=reason))
            self.batches.append(dict(nworld=len(samples), execution="serial_retry", physics_seconds=None,
                                     end_to_end_seconds=time.perf_counter()-began, serial_retries=len(samples),
                                     fallback_reason=reason))
            return results
