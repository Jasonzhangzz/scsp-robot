"""Physics-in-the-loop MPPI for bimanual EE delta-pose control.

The optimizer deliberately depends on a small rollout interface rather than
importing MuJoCo-Warp at module import time.  This keeps the acc/Python 3.8
tests usable while allowing a Spider/MJWarp backend to provide the same
``rollout`` method in the dedicated Python 3.12 environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
import time
from typing import Any, Callable, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_pair import ContactPair


ACTION_DIM = 12


def shared_approach_translation(current, targets, steps, translation_limit):
    """Return one world-frame translation delta for each arm.

    Both arms move the same fraction of their remaining position error.  The
    fraction is ``1/steps`` unless that would push the farther arm past
    ``translation_limit``, in which case the farther arm sets the fraction.
    Repeating the delta therefore finishes both approaches on the same step
    instead of letting the nearer arm arrive first.
    """
    current = np.asarray(current, dtype=np.float64).reshape(2, 3)
    targets = np.asarray(targets, dtype=np.float64).reshape(2, 3)
    error = targets - current
    farther = float(np.max(np.linalg.norm(error, axis=1)))
    if farther <= 1.0e-9:
        return np.zeros((2, 3), dtype=np.float64)
    step_count = max(int(steps), 1)
    step_fraction = min(1.0 / float(step_count), abs(float(translation_limit)) / farther)
    return error * step_fraction


def mjwarp_available() -> bool:
    """Return whether the optional Spider/MuJoCo-Warp stack is importable."""
    try:
        import mujoco_warp  # noqa: F401
        import warp  # noqa: F401
    except Exception:
        return False
    return True


def _as_pose6(position, quaternion) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64).reshape(3)
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    quaternion = quaternion / max(float(np.linalg.norm(quaternion)), 1.0e-12)
    rotvec = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_rotvec()
    return np.concatenate((position, rotvec))


def _quat_from_wxyz(quaternion) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return quaternion / max(float(np.linalg.norm(quaternion)), 1.0e-12)


def _lift_progress_signal(
    object_z,
    start_z,
    lift_scale,
    contact_mask,
    normal_force,
    force_target,
    weight,
):
    """Return a force-gated upward progress reward for one physical step.

    The gate is intentionally conjunctive: both MuJoCo contact bits must be
    true and both measured normal forces are compared with lambda's desired
    magnitudes.  This makes unilateral object motion contribute zero reward,
    while the height term remains continuous once a bilateral grasp exists.
    """
    mask = np.asarray(contact_mask, dtype=bool).reshape(2)
    force = np.asarray(normal_force, dtype=np.float64).reshape(2)
    target = np.maximum(np.asarray(force_target, dtype=np.float64).reshape(2), 1.0e-4)
    force_gate = np.clip(force / target, 0.0, 1.0)
    bilateral_gate = float(np.prod(force_gate)) * float(np.all(mask))
    scale = max(float(lift_scale), 1.0e-4)
    height_gain = max(float(object_z) - float(start_z), 0.0)
    normalized_lift = min(height_gain / scale, 1.5)
    reward = max(float(weight), 0.0) * bilateral_gate * normalized_lift
    return reward, bilateral_gate, normalized_lift


@dataclass
class PhysicalRollout:
    """State and contact traces returned by one physical candidate."""

    object_pose_se3: np.ndarray
    ee_pose_se3: np.ndarray
    contact_mask: np.ndarray
    normal_force: np.ndarray
    object_xy_drift: np.ndarray
    stage_cost: np.ndarray
    terminal_cost: float
    total_cost: float
    state_trace: Any = None

    def __post_init__(self):
        self.object_pose_se3 = np.asarray(self.object_pose_se3, dtype=np.float64).reshape(-1, 6)
        self.ee_pose_se3 = np.asarray(self.ee_pose_se3, dtype=np.float64).reshape(-1, 12)
        self.contact_mask = np.asarray(self.contact_mask, dtype=bool).reshape(-1, 2)
        self.normal_force = np.asarray(self.normal_force, dtype=np.float64).reshape(-1, 2)
        self.object_xy_drift = np.asarray(self.object_xy_drift, dtype=np.float64).reshape(-1, 2)
        self.stage_cost = np.asarray(self.stage_cost, dtype=np.float64).reshape(-1)
        self.terminal_cost = float(self.terminal_cost)
        self.total_cost = float(self.total_cost)


class CallablePhysicalBackend:
    """Adapter for tests and external Spider environments.

    ``rollout_fn`` receives ``(controls, contact_pair, target_pos,
    target_quat)`` and may return either a :class:`PhysicalRollout` or a
    dictionary with the same fields.
    """

    def __init__(self, rollout_fn: Callable[..., Any]):
        self.rollout_fn = rollout_fn

    def rollout(self, controls, contact_pair, target_object_pos, target_object_quat, **kwargs):
        parameters = inspect.signature(self.rollout_fn).parameters
        if not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        value = self.rollout_fn(
            np.asarray(controls, dtype=np.float64),
            contact_pair,
            np.asarray(target_object_pos, dtype=np.float64),
            np.asarray(target_object_quat, dtype=np.float64),
            **kwargs,
        )
        if isinstance(value, PhysicalRollout):
            return value
        return PhysicalRollout(**value)


class MJWarpBimanualRolloutBackend(CallablePhysicalBackend):
    """Lazy Spider/MJWarp bridge.

    A Spider environment can expose ``rollout_ee_delta_pose`` directly, or a
    caller can provide an equivalent callback.  No MJWarp module is imported
    until a rollout is requested, so the existing acc environment remains
    importable on Python 3.8.
    """

    def __init__(self, env=None, rollout_fn=None):
        if rollout_fn is None:
            if env is None or not hasattr(env, "rollout_ee_delta_pose"):
                raise ValueError(
                    "MJWarpBimanualRolloutBackend requires rollout_fn or an "
                    "environment exposing rollout_ee_delta_pose"
                )
            rollout_fn = env.rollout_ee_delta_pose
        super().__init__(rollout_fn)
        self.env = env

    def rollout_batch(self, controls_batch, contact_pair, target_object_pos, target_object_quat, **kwargs):
        """Use an environment-provided vectorized rollout when available."""
        if self.env is not None and hasattr(self.env, "rollout_ee_delta_pose_batch"):
            values = self.env.rollout_ee_delta_pose_batch(
                np.asarray(controls_batch, dtype=np.float64),
                contact_pair,
                np.asarray(target_object_pos, dtype=np.float64),
                np.asarray(target_object_quat, dtype=np.float64),
                **kwargs,
            )
            return [value if isinstance(value, PhysicalRollout) else PhysicalRollout(**value) for value in values]
        return [
            self.rollout(controls, contact_pair, target_object_pos, target_object_quat, **kwargs)
            for controls in np.asarray(controls_batch)
        ]


class MujocoBimanualRolloutBackend:
    """Sequential MuJoCo fallback with the same contract as MJWarp.

    The implementation is intentionally conservative: each sampled candidate
    starts from an identical snapshot, executes the real Cartesian impedance
    controller, and reads object pose/contact forces from MuJoCo.  It is slower
    than the vectorized MJWarp backend but is useful for validating the
    physical objective in the existing ``acc`` environment.
    """

    def __init__(
        self,
        env,
        *,
        ee_target_offset: float = 0.0,
        physics_steps: Optional[int] = None,
        lift_progress_weight: float = 30.0,
        lift_terminal_weight: float = 60.0,
        lift_target_floor: float = 0.01,
    ):
        self.env = env
        self.ee_target_offset = float(ee_target_offset)
        self.physics_steps = physics_steps
        # A squared target-pose residual is nearly constant before the object
        # has moved.  These terms provide a dense signal once the real rollout
        # has bilateral contact, without rewarding a free or unilateral push.
        self.lift_progress_weight = max(float(lift_progress_weight), 0.0)
        self.lift_terminal_weight = max(float(lift_terminal_weight), 0.0)
        self.lift_target_floor = max(float(lift_target_floor), 1.0e-4)

    def _snapshot(self):
        data = self.env.data
        fields = (
            "qpos", "qvel", "qacc", "act", "ctrl", "time", "qacc_warmstart",
            "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat",
        )
        arrays = {name: np.asarray(getattr(data, name)).copy() for name in fields if hasattr(data, name)}
        arms = {}
        for label, arm in (("left", self.env.left_arm), ("right", self.env.right_arm)):
            arms[label] = {
                name: np.asarray(getattr(arm, name)).copy()
                for name in ("position_d", "orientation_d", "p_d", "R_d")
                if hasattr(arm, name)
            }
        return arrays, arms

    def _restore(self, snapshot):
        arrays, arms = snapshot
        for name, value in arrays.items():
            target = getattr(self.env.data, name)
            if np.asarray(target).ndim == 0:
                setattr(self.env.data, name, float(np.asarray(value).reshape(-1)[0]))
            else:
                target[...] = value
        for label, arm in (("left", self.env.left_arm), ("right", self.env.right_arm)):
            for name, value in arms[label].items():
                setattr(arm, name, value.copy())
        self.env.apply_object_wrench_world()
        import mujoco  # lazy: importing the planner must not require MuJoCo

        mujoco.mj_forward(self.env.model, self.env.data)

    def initial_control_guess(
        self,
        contact_pair,
        horizon: int,
        translation_limit: float,
        rotation_limit: float,
    ):
        """Build a deterministic approach seed for the first MPPI solve.

        A zero-mean distribution cannot discover a 15--20 cm approach with a
        5 mm noise scale.  The seed is used as a nominal first candidate and
        can also be blended into the warm start from the measured pose.
        """
        del rotation_limit
        object_pos, object_quat, _ = self.env.get_object_pose()
        projected = contact_pair.project_world(object_pos, object_quat)
        targets = projected["contact_points_world"] - projected["normals_world"] * self.ee_target_offset
        current = np.stack(
            (
                np.asarray(self.env.get_tip_pose(self.env.left_arm)[0], dtype=np.float64),
                np.asarray(self.env.get_tip_pose(self.env.right_arm)[0], dtype=np.float64),
            ),
            axis=0,
        )
        horizon = max(int(horizon), 1)
        # The online loop executes only the first action before replanning.
        # Spreading the whole approach over the MPPI horizon would therefore
        # make the real controller move by only 1/H of the error per cycle.
        # Use a short nominal approach rate, shared by both arms so the nearer
        # fingertip does not reach the surface while the other is still out.
        # The same rate is repeated across the horizon to represent that
        # receding feedback rather than stopping after a single prefix.
        approach_horizon = max(1, min(horizon, 4))
        delta = shared_approach_translation(
            current,
            targets,
            approach_horizon,
            translation_limit,
        )
        seed = np.zeros((horizon, ACTION_DIM), dtype=np.float64)
        seed[:, 0:3] = delta[0]
        seed[:, 6:9] = delta[1]
        return seed

    def rollout(self, controls, contact_pair, target_object_pos, target_object_quat, **kwargs):
        controls = np.asarray(controls, dtype=np.float64).reshape(-1, ACTION_DIM)
        hold_mask = np.asarray(kwargs.get("hold_mask", (False, False)), dtype=bool).reshape(2)
        initial = self._snapshot()
        viewer = getattr(self.env, "viewer", None)
        original_command_steps = getattr(getattr(self.env, "args", None), "mj_steps_per_command", None)
        self.env.viewer = None
        if self.physics_steps is not None and hasattr(getattr(self.env, "args", None), "mj_steps_per_command"):
            self.env.args.mj_steps_per_command = max(int(self.physics_steps), 1)
        start_pos, start_quat, _ = self.env.get_object_pose()
        target_pos = np.asarray(target_object_pos, dtype=np.float64).reshape(3)
        lift_target = max(float(target_pos[2] - start_pos[2]), 0.0)
        lift_scale = max(lift_target, self.lift_target_floor)
        target_quat = _quat_from_wxyz(target_object_quat)
        target_rot = Rotation.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]]).as_matrix()
        ee_target_rot = None
        if kwargs.get("ee_target_quat") is not None:
            ee_target_quat = np.asarray(kwargs["ee_target_quat"], dtype=np.float64).reshape(2, 4)
            ee_target_rot = np.stack(
                [
                    Rotation.from_quat(
                        [quat[1], quat[2], quat[3], quat[0]]
                    ).as_matrix()
                    for quat in ee_target_quat
                ],
                axis=0,
            )
        object_trace = []
        ee_trace = []
        contact_trace = []
        force_trace = []
        xy_trace = []
        stage_cost = []
        state_trace = []
        try:
            self._restore(initial)
            for action in controls:
                action = np.asarray(action, dtype=np.float64).copy()
                if hold_mask[0]:
                    action[:6] = 0.0
                if hold_mask[1]:
                    action[6:12] = 0.0
                self.env.step_ee_pose_delta(
                    action,
                    hold_mask=(False, False),
                    orientation_reference=kwargs.get("ee_target_quat"),
                )
                obj_pos, obj_quat, obj_rot = self.env.get_object_pose()
                left_pos, left_rot = self.env.get_tip_pose(self.env.left_arm)
                right_pos, right_rot = self.env.get_tip_pose(self.env.right_arm)
                contacts = self.env.extract_object_contacts()
                mask = np.array([bool(contacts.get("left")), bool(contacts.get("right"))], dtype=bool)
                forces = np.array(
                    [
                        max((float(item.get("normal_force", 0.0)) for item in contacts.get("left", [])), default=0.0),
                        max((float(item.get("normal_force", 0.0)) for item in contacts.get("right", [])), default=0.0),
                    ],
                    dtype=np.float64,
                )
                projected = contact_pair.project_world(obj_pos, obj_quat)
                target_points = projected["contact_points_world"] - projected["normals_world"] * self.ee_target_offset
                ee_pos = np.stack((left_pos, right_pos), axis=0)
                delta = ee_pos - target_points
                normal_gap = np.sum(delta * projected["normals_world"], axis=1)
                tangent_err = delta - normal_gap[:, None] * projected["normals_world"]
                ee_forward = np.stack((left_rot[:, 2], right_rot[:, 2]), axis=0)
                outward = -projected["normals_world"]
                # Keep the per-arm contact approach error separate from the
                # object orientation error below.  Reusing the same variable
                # here silently discarded the EE orientation term and made
                # MPPI prefer positional approaches that can hit the object
                # with the wrong face.
                ee_orientation_err = 1.0 - np.sum(ee_forward * outward, axis=1) ** 2
                if ee_target_rot is not None:
                    # The IK orientation is a late-contact refinement.  Keep
                    # it as a soft term so a large initial rotation error
                    # cannot overwhelm the translational approach signal.
                    orientation_gate = np.exp(-np.sum(delta ** 2, axis=1) / (0.05 ** 2))
                    ee_orientation_err = ee_orientation_err + 0.1 * np.asarray(
                        [
                            Rotation.from_matrix(ee_target_rot[index].T @ arm_rot).magnitude() ** 2
                            for index, arm_rot in enumerate((left_rot, right_rot))
                        ],
                        dtype=np.float64,
                    ) * orientation_gate
                force_target = np.maximum(np.linalg.norm(projected["desired_force_world"], axis=1), 1.0e-4)
                witness_target = np.maximum(np.linalg.norm(projected["witness_force_world"], axis=1), 1.0e-4)
                force_deficit = np.maximum(force_target - forces, 0.0) / force_target
                witness_deficit = np.maximum(witness_target - forces, 0.0) / witness_target
                # Keep a finite unilateral penalty so the optimizer can still
                # traverse the short physical interval before both arms touch.
                unilateral_penalty = 80.0 if bool(mask[0] ^ mask[1]) else 0.0
                missing_penalty = 25.0 * float(np.count_nonzero(~mask))
                bilateral_bonus = 0.0
                target_err = np.linalg.norm(obj_pos - target_pos)
                object_orientation_err = Rotation.from_matrix(target_rot.T @ obj_rot).magnitude()
                lateral = np.linalg.norm(obj_pos[:2] - start_pos[:2])
                # A free object can have tiny numerical XY motion during a
                # rollout.  The dangerous motion is the one produced while
                # exactly one arm is in contact, so scale that component by
                # the 1 cm re-plan threshold instead of suppressing all
                # no-contact candidates.
                if bool(mask[0] ^ mask[1]):
                    lateral_cost = 80.0 * (lateral / 0.01) ** 2
                else:
                    lateral_cost = 140.0 * lateral ** 2
                # Smoothly prefer a rollout in which both fingertips are
                # close to their own contact targets at the same time.  This
                # keeps a one-sided contact from becoming cheaper than
                # bringing the second arm in, while remaining a continuous
                # distance term rather than a contact-stage rule.
                proximity = np.exp(-np.sum(delta ** 2, axis=1) / (0.025 ** 2))
                synchronized_approach_reward = 220.0 * float(np.prod(proximity))
                # Use physical contact feedback for the lift signal.  A
                # fingertip that is merely close to the object cannot earn
                # progress, and one-sided contact cannot drag the object into
                # a lower-cost lifted state.
                lift_progress_reward, _, _ = _lift_progress_signal(
                    obj_pos[2],
                    start_pos[2],
                    lift_scale,
                    mask,
                    forces,
                    force_target,
                    self.lift_progress_weight,
                )
                stage = (
                    3000.0 * float(np.sum(tangent_err ** 2))
                    + 1000.0 * float(np.sum(normal_gap ** 2))
                    + 35.0 * float(np.sum(ee_orientation_err))
                    + 12.0 * float(np.sum(force_deficit ** 2))
                    + 8.0 * float(np.sum(witness_deficit ** 2))
                    + unilateral_penalty
                    + missing_penalty
                    + bilateral_bonus
                    + 60.0 * target_err ** 2
                    + 15.0 * object_orientation_err ** 2
                    + lateral_cost
                    + 0.5 * float(np.sum(action ** 2))
                    - synchronized_approach_reward
                    - lift_progress_reward
                )
                object_trace.append(_as_pose6(obj_pos, obj_quat))
                ee_trace.append(
                    np.concatenate((_as_pose6(left_pos, _mat_to_wxyz(left_rot)), _as_pose6(right_pos, _mat_to_wxyz(right_rot))))
                )
                contact_trace.append(mask)
                force_trace.append(forces)
                xy_trace.append(obj_pos[:2] - start_pos[:2])
                stage_cost.append(stage)
                state_trace.append(
                    np.concatenate((np.asarray(self.env.data.qpos).copy(), np.asarray(self.env.data.qvel).copy()))
                )
            final_pos, final_quat, _ = self.env.get_object_pose()
            final_contacts = contact_trace[-1] if contact_trace else np.zeros(2, dtype=bool)
            final_force = force_trace[-1] if force_trace else np.zeros(2, dtype=np.float64)
            target_err = np.linalg.norm(final_pos - target_pos)
            final_rot = self.env.get_object_pose()[2]
            final_orientation_err = Rotation.from_matrix(target_rot.T @ final_rot).magnitude()
            terminal = 300.0 * float(target_err ** 2)
            terminal += 40.0 * float(final_orientation_err ** 2)
            terminal_lateral = np.linalg.norm(final_pos[:2] - start_pos[:2])
            if bool(final_contacts[0] ^ final_contacts[1]):
                terminal += 120.0 * float((terminal_lateral / 0.01) ** 2)
            else:
                terminal += 120.0 * float(terminal_lateral ** 2)
            terminal += 200.0 if not bool(np.all(final_contacts)) else 0.0
            terminal += 20.0 * float(np.sum(np.maximum(0.0, 1.0e-4 - final_force) ** 2))
            # Penalise the remaining lift deficit only after the rollout has
            # real bilateral contact.  Approach candidates keep the explicit
            # missing-contact penalty but are not forced to lift a free body.
            if bool(np.all(final_contacts)) and lift_target > 0.0:
                final_projected = contact_pair.project_world(final_pos, final_quat)
                final_force_target = np.maximum(
                    np.linalg.norm(final_projected["desired_force_world"], axis=1),
                    1.0e-4,
                )
                final_force_gate = np.clip(final_force / final_force_target, 0.0, 1.0)
                final_bilateral_gate = float(np.prod(final_force_gate))
                final_gain = max(float(final_pos[2] - start_pos[2]), 0.0)
                lift_deficit = max(1.0 - min(final_gain / lift_scale, 1.0), 0.0)
                terminal += self.lift_terminal_weight * final_bilateral_gate * lift_deficit ** 2
            return PhysicalRollout(
                object_pose_se3=np.asarray(object_trace),
                ee_pose_se3=np.asarray(ee_trace),
                contact_mask=np.asarray(contact_trace),
                normal_force=np.asarray(force_trace),
                object_xy_drift=np.asarray(xy_trace),
                stage_cost=np.asarray(stage_cost),
                terminal_cost=terminal,
                total_cost=float(np.sum(stage_cost) + terminal),
                state_trace=np.asarray(state_trace),
            )
        finally:
            self._restore(initial)
            self.env.viewer = viewer
            if original_command_steps is not None:
                self.env.args.mj_steps_per_command = original_command_steps


def _mat_to_wxyz(rotation_matrix):
    quat_xyzw = Rotation.from_matrix(np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


class PhysicalBimanualMPPI:
    """Spider-style sampling MPPI over physical bimanual rollouts."""

    def __init__(
        self,
        backend,
        *,
        horizon: int = 20,
        samples: int = 64,
        iterations: int = 2,
        init_iterations: Optional[int] = None,
        temperature: float = 1.0,
        noise_sigma: float = 0.01,
        noise_decay: float = 0.85,
        elite_frac: float = 0.1,
        translation_limit: float = 0.05,
        rotation_limit: float = 0.12,
        knot_count: int = 5,
        dt: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.backend = backend
        self.horizon = max(int(horizon), 1)
        self.samples = max(int(samples), 2)
        self.iterations = max(int(iterations), 1)
        self.init_iterations = max(int(self.iterations if init_iterations is None else init_iterations), 1)
        self.temperature = max(float(temperature), 1.0e-6)
        self.noise_sigma = max(float(noise_sigma), 1.0e-7)
        self.noise_decay = float(np.clip(noise_decay, 0.05, 1.0))
        self.elite_frac = float(np.clip(elite_frac, 0.01, 1.0))
        self.translation_limit = abs(float(translation_limit))
        self.rotation_limit = abs(float(rotation_limit))
        self.knot_count = max(min(int(knot_count), self.horizon), 1)
        self.dt = max(float(dt), 1.0e-6)
        self.rng = np.random.default_rng(seed)
        self.u_mean = np.zeros((self.horizon, ACTION_DIM), dtype=np.float64)
        self._has_warm_start = False
        self.last_result = None

    def reset(self):
        self.u_mean.fill(0.0)
        self._has_warm_start = False
        self.last_result = None

    def _clip(self, controls):
        controls = np.asarray(controls, dtype=np.float64).copy()
        controls[..., 0:3] = np.clip(controls[..., 0:3], -self.translation_limit, self.translation_limit)
        controls[..., 6:9] = np.clip(controls[..., 6:9], -self.translation_limit, self.translation_limit)
        controls[..., 3:6] = np.clip(controls[..., 3:6], -self.rotation_limit, self.rotation_limit)
        controls[..., 9:12] = np.clip(controls[..., 9:12], -self.rotation_limit, self.rotation_limit)
        return controls

    @staticmethod
    def _apply_hold_mask(controls, hold_mask):
        hold_mask = np.asarray(hold_mask, dtype=bool).reshape(2)
        controls = np.asarray(controls, dtype=np.float64).copy()
        if hold_mask[0]:
            controls[..., :6] = 0.0
        if hold_mask[1]:
            controls[..., 6:12] = 0.0
        return controls

    def _sample(self, sigma):
        knots = self.rng.normal(0.0, sigma, size=(self.samples, self.knot_count, ACTION_DIM))
        knot_x = np.linspace(0.0, 1.0, self.knot_count)
        step_x = np.linspace(0.0, 1.0, self.horizon)
        noise = np.empty((self.samples, self.horizon, ACTION_DIM), dtype=np.float64)
        for sample in range(self.samples):
            for dim in range(ACTION_DIM):
                noise[sample, :, dim] = np.interp(step_x, knot_x, knots[sample, :, dim])
        candidates = self._clip(self.u_mean[None, :, :] + noise)
        # Keep the current nominal sequence in the population.  A finite
        # sample set can otherwise perturb every candidate away from the
        # deterministic approach seed (or a good warm start), making the
        # optimizer prefer a cheaper but still non-contact rollout.
        candidates[0] = self._clip(self.u_mean)
        return candidates

    def plan_once(
        self,
        contact_pair,
        target_object_pos,
        target_object_quat,
        *,
        state=None,
        trajectory_path=None,
        hold_mask=(False, False),
        **backend_kwargs,
    ):
        del state  # The backend owns the complete physical state snapshot.
        approach_seed_weight = float(
            np.clip(backend_kwargs.pop("approach_seed_weight", 0.0), 0.0, 1.0)
        )
        solve_t0 = time.perf_counter()
        if not isinstance(contact_pair, ContactPair):
            contact_pair = ContactPair.from_lambda_targets(contact_pair)
        had_warm_start = self._has_warm_start
        best = None
        controls_best = None
        if hasattr(self.backend, "initial_control_guess"):
            approach_seed = self._clip(
                self.backend.initial_control_guess(
                    contact_pair,
                    horizon=self.horizon,
                    translation_limit=self.translation_limit,
                    rotation_limit=self.rotation_limit,
                )
            )
            if not had_warm_start:
                self.u_mean = approach_seed
            elif approach_seed_weight > 0.0:
                self.u_mean = self._clip(
                    (1.0 - approach_seed_weight) * self.u_mean
                    + approach_seed_weight * approach_seed
                )
        self.u_mean = self._apply_hold_mask(self.u_mean, hold_mask)
        iteration_count = self.iterations if had_warm_start else self.init_iterations
        for iteration in range(iteration_count):
            candidates = self._sample(self.noise_sigma * (self.noise_decay ** iteration))
            candidates = self._apply_hold_mask(candidates, hold_mask)
            if hasattr(self.backend, "rollout_batch"):
                rollouts = self.backend.rollout_batch(
                    candidates,
                    contact_pair,
                    target_object_pos,
                    target_object_quat,
                    hold_mask=hold_mask,
                    **backend_kwargs,
                )
            else:
                rollouts = [
                    self.backend.rollout(
                        candidates[i],
                        contact_pair,
                        target_object_pos,
                        target_object_quat,
                        hold_mask=hold_mask,
                        **backend_kwargs,
                    )
                    for i in range(self.samples)
                ]
            costs = np.nan_to_num(
                np.asarray([item.total_cost for item in rollouts], dtype=np.float64),
                nan=1.0e12,
                posinf=1.0e12,
                neginf=-1.0e12,
            )
            elite_count = max(1, min(self.samples, int(round(self.samples * self.elite_frac))))
            elite = np.argsort(costs)[:elite_count]
            weights = np.exp(-(costs[elite] - costs[elite[0]]) / self.temperature)
            weights /= max(float(np.sum(weights)), 1.0e-12)
            self.u_mean = self._clip(np.sum(candidates[elite] * weights[:, None, None], axis=0))
            best_idx = int(np.argmin(costs))
            best = rollouts[best_idx]
            controls_best = candidates[best_idx].copy()

        action = self._apply_hold_mask(self._clip(self.u_mean[0]), hold_mask)
        shifted = np.zeros_like(self.u_mean)
        if self.horizon > 1:
            shifted[:-1] = self.u_mean[1:]
        self.u_mean = shifted
        self._has_warm_start = True
        result = {
            "action": action.copy(),
            "ee_delta_pose": controls_best.copy(),
            "object_pose_se3": best.object_pose_se3.copy(),
            "ee_pose_se3": best.ee_pose_se3.copy(),
            "contact_mask": best.contact_mask.copy(),
            "normal_force": best.normal_force.copy(),
            "object_xy_drift": best.object_xy_drift.copy(),
            "stage_cost": best.stage_cost.copy(),
            "time": np.arange(self.horizon, dtype=np.float64) * self.dt,
            "terminal_cost": float(best.terminal_cost),
            "cost": float(best.total_cost),
            "cost_opt": float(best.total_cost),
            "rollout": best.object_pose_se3.copy(),
            "rollout_q": best.state_trace,
            "solver_backend": "physical_mppi",
            "solve_status": "success",
            "solve_time": float(time.perf_counter() - solve_t0),
            "warm_start": bool(had_warm_start),
        }
        if trajectory_path is not None:
            self.save_trajectory(trajectory_path, result)
        self.last_result = result
        return result

    @staticmethod
    def save_trajectory(path, result):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ee_delta_pose": np.asarray(result["ee_delta_pose"], dtype=np.float64),
            "object_pose_se3": np.asarray(result["object_pose_se3"], dtype=np.float64),
            "ee_pose_se3": np.asarray(result["ee_pose_se3"], dtype=np.float64),
            "time": np.asarray(result.get("time", np.arange(len(result["ee_delta_pose"]))), dtype=np.float64),
            "contact_mask": np.asarray(result["contact_mask"], dtype=np.int8),
            "normal_force": np.asarray(result["normal_force"], dtype=np.float64),
            "object_xy_drift": np.asarray(result["object_xy_drift"], dtype=np.float64),
            "stage_cost": np.asarray(result["stage_cost"], dtype=np.float64),
            "terminal_cost": float(result["terminal_cost"]),
            "total_cost": float(result["cost"]),
        }
        if result.get("rollout_q") is not None:
            payload["state_trace"] = np.asarray(result["rollout_q"], dtype=np.float64)
        np.savez_compressed(path, **payload)


# Explicit alias for downstream scripts.
SpiderBimanualMPPI = PhysicalBimanualMPPI
