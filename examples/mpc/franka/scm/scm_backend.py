"""SCM diagonal-Delassus contact law in the common ComFree/MJWarp pipeline.

The benchmark uses impulse units:
  v_free = v + h M^-1 f_smooth,
  p = max(-(J v_free + phi/h) / (diag(J M^-1 J.T) + r), 0),
  v_next = v_free + M^-1 J.T p.
The original and amplitude-corrected SCM add no contact damping. Soft SCM
uses a finite response time and damping ratio in the same diagonal row law.
Coefficients are frozen for ONE step
and recomputed at the next state. This is the full-dynamic extension of the
SCM contact approximation in method.tex, not the CSO optimizer itself.
The optional increment mode projects only h*a_smooth, like the current CSO
increment formulation. Point-contact SCM has condim <= 3; matched 6D friction
is an explicitly selected extension, not a claimed feature of original SCM.
The optional amplitude correction minimizes the matched coupled quadratic
along the segment [0, p], without forming/inverting the coupled Delassus.
"""

import numpy as np
import mujoco
import warp as wp
import comfree_warp as cf
from comfree_warp.comfree_core._src import constraint
from comfree_warp.comfree_core._src.forward import _advance_vel, _compute_qfrc_total
from comfree_warp.mujoco_warp._src import collision_driver, smooth
from comfree_warp.mujoco_warp._src.forward import euler, fwd_acceleration, fwd_actuation, fwd_velocity


@wp.kernel
def diagonal_contact(
    nv: int, timestep: wp.array(dtype=float), nefc: wp.array(dtype=int),
    J: wp.array3d(dtype=float), gap: wp.array2d(dtype=float),
    vfree: wp.array2d(dtype=float), v: wp.array2d(dtype=float),
    inverse_diagonal: wp.array(dtype=float), inverse_dense: wp.array2d(dtype=float),
    use_diagonal: bool, increment: bool, regularization: float,
    response_time: float, damping_ratio: float,
    force: wp.array2d(dtype=float), generalized_force: wp.array2d(dtype=float),
):
    world, row = wp.tid()
    if row >= nefc[world]:
        return
    h = timestep[world % timestep.shape[0]]
    velocity = float(0.0)
    mobility = float(0.0)
    for i in range(nv):
        predicted = vfree[world, i]
        if increment:
            predicted = predicted - v[world, i]
        velocity += J[world, row, i] * predicted
        if use_diagonal:
            mobility += J[world, row, i] * J[world, row, i] * inverse_diagonal[i]
        else:
            product = float(0.0)
            for j in range(nv):
                product += inverse_dense[i, j] * J[world, row, j]
            mobility += J[world, row, i] * product
    velocity_gain = float(1.0)
    gap_gain = float(1.0)
    if response_time > 0.0:
        x = h / response_time
        denominator = 1.0 + 2.0 * damping_ratio * x + x * x
        velocity_gain = (2.0 * damping_ratio * x + x * x) / denominator
        gap_gain = x * x / denominator
    impulse = wp.max(-(velocity_gain * velocity + gap_gain * gap[world, row] / h) / (mobility + regularization), 0.0)
    reaction = impulse / h
    force[world, row] = reaction
    for i in range(nv):
        wp.atomic_add(generalized_force, world, i, J[world, row, i] * reaction)


@wp.kernel
def correct_amplitude(
    nv: int, timestep: wp.array(dtype=float), nefc: wp.array(dtype=int),
    J: wp.array3d(dtype=float), gap: wp.array2d(dtype=float),
    vfree: wp.array2d(dtype=float), v: wp.array2d(dtype=float),
    inverse_diagonal: wp.array(dtype=float), inverse_dense: wp.array2d(dtype=float),
    use_diagonal: bool, increment: bool, regularization: float,
    response_time: float, damping_ratio: float,
    force: wp.array2d(dtype=float), generalized_force: wp.array2d(dtype=float),
    amplitude: wp.array(dtype=float),
):
    # One scalar per world, not per contact. force is p/h; all quantities
    # below are converted to impulse units before the scalar line search.
    world = wp.tid()
    h = timestep[world % timestep.shape[0]]
    a = float(1.0)
    b = float(1.0)
    if response_time > 0.0:
        x = h / response_time
        divisor = 1.0 + 2.0 * damping_ratio * x + x*x
        a = (2.0 * damping_ratio * x + x*x) / divisor
        b = x*x / divisor
    numerator = float(0.0)
    denominator = float(0.0)
    for row in range(nefc[world]):
        impulse = h * force[world, row]
        velocity = float(0.0)
        for i in range(nv):
            predicted = vfree[world, i]
            if increment:
                predicted -= v[world, i]
            velocity += J[world, row, i] * predicted
        numerator += (a * velocity + b * gap[world, row] / h) * impulse
        denominator += regularization * impulse * impulse
    for i in range(nv):
        q_i = h * generalized_force[world, i]
        if use_diagonal:
            denominator += q_i * q_i * inverse_diagonal[i]
        else:
            product = float(0.0)
            for j in range(nv):
                product += inverse_dense[i, j] * h * generalized_force[world, j]
            denominator += q_i * product
    alpha = float(1.0)
    if denominator > 0.0:
        alpha = wp.clamp(-numerator / denominator, 0.0, 1.0)
    amplitude[world] = alpha
    for row in range(nefc[world]):
        force[world, row] *= alpha
    for i in range(nv):
        generalized_force[world, i] *= alpha


class ScmEngine:
    """Independent bodies for the fidelity suite; exact current mass mobility."""

    def __init__(self, model, initial_data, *, regularization=1e-8, motion="dynamic", friction="point", amplitude_correction=True,
                 response_time=0.0, damping_ratio=1.0):
        if regularization <= 0:
            raise ValueError("SCM r must be positive (R=rI, in inverse-mass units).")
        if type(self) is ScmEngine and not amplitude_correction and response_time == 0.:
            raise ValueError("Standalone uncorrected SCM was removed; enable amplitude correction")
        if model.neq or model.nu or model.nsensor:
            raise ValueError("SCM benchmark currently supports unactuated, sensorless rigid bodies without equality constraints.")
        if model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
            raise ValueError("SCM benchmark uses the common Euler integrator.")
        self.model, self.regularization = model, regularization
        self.motion, self.friction = motion, friction
        self.amplitude_correction = amplitude_correction
        if response_time < 0 or damping_ratio <= 0:
            raise ValueError("Use positive damping ratio and nonnegative response time.")
        self.response_time, self.damping_ratio = response_time, damping_ratio
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, M, initial_data.qM)
        self.use_diagonal = np.allclose(M, np.diag(np.diag(M)), rtol=0, atol=1e-12)
        # For independent free bodies with COM at the body origin and one-body
        # hinge tests, generalized M is constant in MuJoCo's local angular qvel.
        joint_bodies = model.jnt_bodyid
        independent = np.all(model.body_parentid[joint_bodies] == 0)
        centered = np.allclose(model.body_ipos[joint_bodies], 0, atol=1e-12)
        simple = np.all(np.isin(model.jnt_type, [mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE]))
        self.constant_diagonal = bool(self.use_diagonal and independent and centered and simple)
        if model.nv > 60 and not self.constant_diagonal:
            raise ValueError("Large SCM scenes require independent centered bodies with diagonal generalized mass.")
        self.inverse_diagonal = wp.array(1 / np.diag(M), dtype=wp.float32)
        self.inverse_dense = wp.array(np.linalg.inv(M) if not self.use_diagonal else np.zeros((1, 1)), dtype=wp.float32)

    def update_mass(self, data):
        if self.constant_diagonal:
            return
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, data.qM)
        # General small scenes use the full inverse; only the CONTACT Delassus
        # is diagonalized. Off-diagonal inertia is not dropped.
        self.use_diagonal = False
        self.inverse_dense = wp.array(np.linalg.inv(M), dtype=wp.float32)

    def put_model(self, model, **kwargs):
        result = cf.put_model(model, comfree_stiffness=0, comfree_damping=0)
        if self.friction == "point":
            wp.copy(result.geom_condim, wp.array(np.minimum(model.geom_condim, 3), dtype=wp.int32))
            if model.npair:
                wp.copy(result.pair_dim, wp.array(np.minimum(model.pair_dim, 3), dtype=wp.int32))
        return result

    def put_data(self, model, data, **kwargs):
        result = cf.put_data(model, data, **kwargs)
        if self.amplitude_correction:
            self.amplitude = wp.ones(result.nworld, dtype=wp.float32)
        return result

    def prepare_step(self, m, d):
        # Reuse the upstream smooth dynamics, collision and dual-cone rows;
        # replace only the contact reaction. No MuJoCo solver force is reused.
        smooth.kinematics(m, d)
        smooth.com_pos(m, d)
        smooth.camlight(m, d)
        smooth.flex(m, d)
        smooth.tendon(m, d)
        smooth.crb(m, d)
        smooth.tendon_armature(m, d)
        smooth.factor_m(m, d)
        collision_driver.collision(m, d)
        constraint.make_constraint(m, d)
        smooth.transmission(m, d)
        fwd_velocity(m, d)
        fwd_actuation(m, d)
        fwd_acceleration(m, d, factorize=True)
        wp.launch(_advance_vel, dim=(d.nworld, m.nv),
                  inputs=[m.opt.timestep, d.qvel, d.qacc_smooth],
                  outputs=[d.qvel_smooth_pred, d.qfrc_constraint])

    def step(self, m, d):
        self.prepare_step(m, d)
        wp.launch(diagonal_contact, dim=(d.nworld, d.efc.J.shape[1]),
                  inputs=[m.nv, m.opt.timestep, d.nefc, d.efc.J, d.efc.efc_dist,
                          d.qvel_smooth_pred, d.qvel, self.inverse_diagonal, self.inverse_dense,
                          self.use_diagonal, self.motion == "increment", self.regularization,
                          self.response_time, self.damping_ratio],
                  outputs=[d.efc.force, d.qfrc_constraint])
        if self.amplitude_correction:
            wp.launch(correct_amplitude, dim=d.nworld,
                      inputs=[m.nv, m.opt.timestep, d.nefc, d.efc.J, d.efc.efc_dist,
                              d.qvel_smooth_pred, d.qvel, self.inverse_diagonal, self.inverse_dense,
                              self.use_diagonal, self.motion == "increment", self.regularization,
                              self.response_time, self.damping_ratio],
                      outputs=[d.efc.force, d.qfrc_constraint, self.amplitude])
        self.finish_step(m, d)

    def finish_step(self, m, d):
        wp.launch(_compute_qfrc_total, dim=(d.nworld, m.nv),
                  inputs=[d.qfrc_smooth, d.qfrc_constraint], outputs=[d.qfrc_total])
        smooth.solve_m(m, d, d.qacc, d.qfrc_total)
        wp.copy(d.efc.Ma, d.qfrc_total)
        euler(m, d)
