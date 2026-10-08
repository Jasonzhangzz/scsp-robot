"""Warp explicit contact model extracted from comfree-warp.

The CasADi :class:`models.explicit_model.ExplicitModel` takes precomputed
``(phi, J, sigma)``.  This Warp model runs MuJoCo-Warp kinematics, collision,
and constraint assembly, then replaces ``solver.solve`` with the closed-form
unilateral spring-damper from ``comfree_core``:

    efc_vel = J @ (qvel + qacc_smooth * dt)
    efc_penetration = efc_vel * dt + efc_pos
    f = max(efc_D * (-b * efc_vel - k * efc_penetration), 0)
    qfrc_constraint += J^T f

``efc.pos`` / ``efc.D`` are the standard mjwarp equivalents of comfree's
``efc_dist`` / ``efc_mass``.  Collision and integration stay on system
``mujoco_warp``; only the constraint force is complementarity-free.

Contact kernels are adapted from ``thirdparty/comfree_warp/comfree_warp/comfree_core``
(noncommercial academic research license).
"""

from __future__ import annotations

import numpy as np

try:
    import warp as wp

    wp.set_module_options({"enable_backward": False})
except ImportError:  # pragma: no cover - torch Adam path does not need Warp
    wp = None

try:
    import mujoco_warp as mjwarp
    from mujoco_warp._src.warp_util import event_scope
except ImportError:  # pragma: no cover - public API fallback
    mjwarp = None

    def event_scope(fn):
        return fn


@wp.kernel
def _advance_vel(
    opt_timestep: wp.array(dtype=float),
    qvel_in: wp.array2d(dtype=float),
    qacc_smooth_in: wp.array2d(dtype=float),
    qvel_out: wp.array2d(dtype=float),
    qfrc_constraint: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    qvel_out[worldid, dofid] = qvel_in[worldid, dofid] + qacc_smooth_in[worldid, dofid] * timestep
    qfrc_constraint[worldid, dofid] = 0.0


@wp.kernel
def _compute_qfrc_constraint(
    opt_timestep: wp.array(dtype=float),
    comfree_stiffness: wp.array(dtype=float),
    comfree_damping: wp.array(dtype=float),
    J: wp.array3d(dtype=float),
    efc_dist: wp.array2d(dtype=float),
    efc_mass: wp.array2d(dtype=float),
    qvel_smooth_pred: wp.array2d(dtype=float),
    nv: int,
    nefc: wp.array(dtype=int),
    efc_force: wp.array2d(dtype=float),
    qfrc_constraint: wp.array2d(dtype=float),
):
    worldid, efcid = wp.tid()
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    if efcid >= nefc[worldid]:
        return

    efc_vel = float(0.0)
    for i in range(nv):
        efc_vel += J[worldid, efcid, i] * qvel_smooth_pred[worldid, i]

    stiffness = comfree_stiffness[worldid % comfree_stiffness.shape[0]] / timestep
    damping = comfree_damping[worldid % comfree_damping.shape[0]] / timestep
    efc_penetration = efc_vel * timestep + efc_dist[worldid, efcid]
    efc_acc = -damping * efc_vel - stiffness * efc_penetration
    efc_frc = wp.max(efc_mass[worldid, efcid] * efc_acc, 0.0)

    efc_force[worldid, efcid] = efc_frc
    for i in range(nv):
        wp.atomic_add(qfrc_constraint, worldid, i, J[worldid, efcid, i] * efc_frc)


@wp.kernel
def _compute_qfrc_total(
    qfrc_smooth: wp.array2d(dtype=float),
    qfrc_constraint: wp.array2d(dtype=float),
    qfrc_total: wp.array2d(dtype=float),
):
    worldid, dofid = wp.tid()
    qfrc_total[worldid, dofid] = qfrc_smooth[worldid, dofid] + qfrc_constraint[worldid, dofid]


def attach_comfree_fields(model, data, comfree_stiffness=0.2, comfree_damping=0.001):
    """Attach the extra Model/Data arrays used by the closed-form contact law."""
    device = model.opt.timestep.device
    if not hasattr(model, "comfree_stiffness"):
        model.comfree_stiffness = wp.array(
            np.atleast_1d(comfree_stiffness), dtype=wp.float32, device=device
        )
    if not hasattr(model, "comfree_damping"):
        model.comfree_damping = wp.array(
            np.atleast_1d(comfree_damping), dtype=wp.float32, device=device
        )
    if not hasattr(data, "qvel_smooth_pred"):
        data.qvel_smooth_pred = wp.zeros(data.qvel.shape, dtype=float, device=device)
    if not hasattr(data, "qfrc_total"):
        data.qfrc_total = wp.zeros(data.qvel.shape, dtype=float, device=device)
    return model, data


@event_scope
def compute_qfrc_total(model, data):
    """Closed-form constraint forces, then ``qfrc_total = qfrc_smooth + qfrc_constraint``."""
    wp.launch(
        _advance_vel,
        dim=(data.nworld, model.nv),
        inputs=[model.opt.timestep, data.qvel, data.qacc_smooth],
        outputs=[data.qvel_smooth_pred, data.qfrc_constraint],
    )
    wp.launch(
        _compute_qfrc_constraint,
        dim=(data.nworld, data.efc.J.shape[1]),
        inputs=[
            model.opt.timestep,
            model.comfree_stiffness,
            model.comfree_damping,
            data.efc.J,
            data.efc.pos,
            data.efc.D,
            data.qvel_smooth_pred,
            model.nv,
            data.nefc,
        ],
        outputs=[data.efc.force, data.qfrc_constraint],
    )
    wp.launch(
        _compute_qfrc_total,
        dim=(data.nworld, model.nv),
        inputs=[data.qfrc_smooth, data.qfrc_constraint],
        outputs=[data.qfrc_total],
    )


@event_scope
def forward_explicit(model, data):
    """Forward dynamics with closed-form contact instead of ``solver.solve``."""
    energy = model.opt.enableflags & mjwarp.EnableBit.ENERGY

    mjwarp.fwd_position(model, data, factorize=False)
    data.sensordata.zero_()
    mjwarp.sensor_pos(model, data)
    if energy:
        if getattr(model, "sensor_e_potential", 0) == 0:
            mjwarp.energy_pos(model, data)
    else:
        data.energy.zero_()

    mjwarp.fwd_velocity(model, data)
    mjwarp.sensor_vel(model, data)
    if energy and getattr(model, "sensor_e_kinetic", 0) == 0:
        mjwarp.energy_vel(model, data)

    if not (model.opt.disableflags & mjwarp.DisableBit.ACTUATION):
        callback = getattr(model, "callback", None)
        if callback is not None and callback.control:
            callback.control(model, data)
    mjwarp.fwd_actuation(model, data)
    mjwarp.fwd_acceleration(model, data, factorize=True)

    if data.njmax == 0 or model.nv == 0:
        wp.copy(data.qacc, data.qacc_smooth)
        wp.copy(data.qfrc_total, data.qfrc_smooth)
    else:
        compute_qfrc_total(model, data)
        mjwarp.solve_m(model, data, data.qacc, data.qfrc_total)

    mjwarp.sensor_acc(model, data)


@event_scope
def step_explicit(model, data):
    """Advance every world one timestep with closed-form contact."""
    forward_explicit(model, data)
    integrator = model.opt.integrator
    if integrator == mjwarp.IntegratorType.EULER:
        wp.copy(data.efc.Ma, data.qfrc_total)
        mjwarp.euler(model, data)
        return
    if integrator in (mjwarp.IntegratorType.IMPLICITFAST, mjwarp.IntegratorType.IMPLICIT):
        wp.copy(data.efc.Ma, data.qfrc_total)
        mjwarp.implicit(model, data)
        return
    raise NotImplementedError(f"integrator {integrator} is not supported by ExplicitModelWarp")


class ExplicitModelWarp:
    """Batched MuJoCo-Warp stepper that uses closed-form contact for rollouts."""

    def __init__(
        self,
        mj_model,
        mj_data,
        *,
        nworld: int = 1,
        nconmax: int = 48,
        njmax: int = 80,
        comfree_stiffness: float = 0.2,
        comfree_damping: float = 0.001,
        device: str = "cuda:0",
    ):
        self.mjwarp = mjwarp
        self.wp = wp
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.nworld = max(int(nworld), 1)
        self.nconmax = int(nconmax)
        self.njmax = int(njmax)
        self.comfree_stiffness = float(comfree_stiffness)
        self.comfree_damping = float(comfree_damping)
        self.device = str(device)
        wp.init()
        wp.set_device(self.device)
        self.model_wp = mjwarp.put_model(mj_model)
        self.data_wp = self.make_data()

    def make_data(self):
        data = self.mjwarp.put_data(
            self.mj_model,
            self.mj_data,
            nworld=self.nworld,
            nconmax=self.nconmax,
            njmax=self.njmax,
        )
        attach_comfree_fields(
            self.model_wp,
            data,
            comfree_stiffness=self.comfree_stiffness,
            comfree_damping=self.comfree_damping,
        )
        return data

    def forward(self, data=None) -> None:
        forward_explicit(self.model_wp, self.data_wp if data is None else data)

    def step(self, data=None) -> None:
        step_explicit(self.model_wp, self.data_wp if data is None else data)

    def broadcast_state(self, qpos, qvel, ctrl, data=None) -> None:
        """Copy a single-world host state into every Warp world and run forward."""
        data = self.data_wp if data is None else data
        wp = self.wp
        qpos_t = self._expand_state(qpos)
        qvel_t = self._expand_state(qvel)
        ctrl_t = self._expand_state(ctrl)
        wp.copy(data.qpos, wp.from_torch(qpos_t))
        wp.copy(data.qvel, wp.from_torch(qvel_t))
        wp.copy(data.ctrl, wp.from_torch(ctrl_t))
        self.forward(data)

    def copy_state(self, src, dst) -> None:
        wp = self.wp
        for name in ("qpos", "qvel", "act", "act_dot", "qacc_warmstart", "ctrl"):
            if hasattr(src, name) and hasattr(dst, name):
                wp.copy(getattr(dst, name), getattr(src, name))

    def capture_step_graph(self, data=None):
        data = self.data_wp if data is None else data
        wp = self.wp

        def _once():
            self.step(data)

        with wp.ScopedDevice(self.device):
            _once()
            _once()
            wp.synchronize()
            try:
                with wp.ScopedCapture() as capture:
                    _once()
                wp.synchronize()
                return capture.graph
            except Exception:
                return None

    def _expand_state(self, value):
        import torch

        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, -1).expand(self.nworld, -1).contiguous()
        return tensor



# Torch GS / Adam dynamics live in models/comfree_gs_torch.py so --mpc
# never imports Warp CUDA kernels next to the MuJoCo viewer.
