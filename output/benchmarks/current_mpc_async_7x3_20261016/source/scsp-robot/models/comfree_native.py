# Contact equations adapted from ASU IRIS comfree_core.
# Noncommercial academic research license: thirdparty/comfree_warp/comfree_warp/comfree_core/LICENSE
"""Differentiable upstream ComFree law on DexForge's native GS geometry.

Use the native diagonal projection and its exact gathered adjoint. Its legacy
L0 scale is cancelled in coefficient preparation, in both forward and VJP.
No Newton contact iterations or softplus contact forces are used.
"""
from dataclasses import replace
from functools import cache


@cache
def _kernels(stiffness, damping):
    import warp as wp
    from comfree_warp.native_adjoint.explicit_damp import (
        ExplicitDampParameters, ReductionInputs, CoefficientReductions,
        CoefficientInputs, CoefficientOutputs)
    from comfree_warp.comfree_core._src.types import L0_IMPEDANCE

    @wp.struct
    class Parameters:
        native: ExplicitDampParameters
        margin: wp.array(dtype=float)
        timestep: wp.array(dtype=float)
        row_count: int
        row_capacity: int
        dof_count: int

    # Preserve the tiled row contraction and its deterministic native VJP.
    def make_reduce(dofs):
        @wp.kernel(module="unique")
        def tiled(params: Parameters, inputs: ReductionInputs, outputs: CoefficientReductions):
            world, row = wp.tid()
            j = wp.tile_load(inputs.jacobian[world, row], shape=dofs, bounds_check=True)
            v = wp.tile_load(inputs.qvel[world], shape=dofs, bounds_check=True)
            a = wp.tile_load(inputs.qacc_smooth[world], shape=dofs, bounds_check=True)
            wp.tile_store(outputs.current_velocity[world], wp.tile_sum(j*v), offset=row, bounds_check=True)
            wp.tile_store(outputs.smooth_acceleration[world], wp.tile_sum(j*a), offset=row, bounds_check=True)
        return tiled

    @wp.kernel(module="unique")
    def coefficients(params: Parameters, inputs: CoefficientInputs, outputs: CoefficientOutputs):
        world, row = wp.tid()
        outputs.velocity_weight[world, row] = 0.0
        if row >= params.row_count or inputs.active[world, row] == 0:
            outputs.free_force[world, row] = 0.0
            return
        native = params.native
        position = inputs.position[world, row]
        distance = position
        invweight = float(0.0)
        solimp = native.jnt_solimp[0, 0]
        if row < native.contact_row_count:
            contact = native.row_contact[row]
            solimp = native.contact_solimp[contact]
            geom = native.contact_geom[contact]
            body0 = native.geom_bodyid[geom[0]]
            body1 = native.geom_bodyid[geom[1]]
            model = world % native.body_invweight.shape[0]
            invweight = native.body_invweight[model, body0][0] + native.body_invweight[model, body1][0]
            distance += params.margin[contact]
        else:
            joint = native.limit_joint[row - native.contact_row_count]
            solimp = native.jnt_solimp[world % native.jnt_solimp.shape[0], joint]
            invweight = native.dof_invweight[world % native.dof_invweight.shape[0], native.jnt_dofadr[joint]]
            # Limit rows contain distance - joint margin too.
            distance += params.margin[native.contact_geom.shape[0] + row - native.contact_row_count]
        minimum = wp.clamp(solimp[0], 0.0001, 0.9999)
        maximum = wp.clamp(solimp[1], 0.0001, 0.9999)
        middle = wp.clamp(solimp[3], 0.0001, 0.9999)
        power = wp.max(solimp[4], 1.0)
        x = wp.abs(position) / 0.01
        curve = wp.where(x < middle, wp.pow(x, power)/wp.pow(middle, power-1.0),
                         1.0-wp.pow(1.0-x, power)/wp.pow(1.0-middle, power-1.0))
        imp = wp.clamp(minimum + curve*(maximum-minimum), minimum, maximum)
        imp = wp.where(x > 1.0, maximum, imp)
        mass = 1.0 / wp.max(invweight*(1.0-imp)/imp, 1e-15)
        dt = params.timestep[world % params.timestep.shape[0]]
        velocity = inputs.reductions.current_velocity[world, row] + dt*inputs.reductions.smooth_acceleration[world, row]
        acceleration = -(damping/dt)*velocity - (stiffness/dt)*(dt*velocity + distance)
        outputs.free_force[world, row] = mass*acceleration / wp.static(L0_IMPEDANCE)

    return Parameters, make_reduce, coefficients


def use_comfree_model(compiled, cpu_model, *, stiffness=0.2, damping=0.001):
    """Return a compiled step using the official law and native exact VJP."""
    import numpy as np
    import warp as wp
    if not np.isfinite(stiffness) or stiffness <= 0 or not np.isfinite(damping) or damping < 0:
        raise ValueError("ComFree stiffness must be positive and damping nonnegative")
    if compiled.sparse_contact:
        raise NotImplementedError("ComFree native adapter currently requires dense constraint rows")
    Parameters, make_reduce, kernel = _kernels(float(stiffness), float(damping))
    solver = compiled.base.contact_solver
    source = solver.parameters
    params = Parameters()
    params.native = source
    params.timestep = source.timestep
    params.row_count = source.row_count
    params.row_capacity = source.row_capacity
    params.dof_count = source.dof_count
    margins = compiled.base.contact_rows.parameters.contact_includemargin.numpy()
    limits = source.limit_joint.numpy()
    params.margin = wp.array(np.concatenate((margins, cpu_model.jnt_margin[limits])), dtype=float, device=compiled.base.device)
    solver = replace(solver, parameters=params, reduction_kernel=make_reduce(source.dof_count), coefficient_kernel=kernel)
    base = replace(compiled.base, contact_solver=solver)
    compiled.contact.iterations = 0
    return replace(compiled, base=base)
