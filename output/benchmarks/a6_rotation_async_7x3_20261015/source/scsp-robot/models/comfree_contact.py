# Contact equations adapted from ASU IRIS comfree_core.
# Noncommercial academic research license: thirdparty/comfree_warp/comfree_warp/comfree_core/LICENSE
"""ASU IRIS ComFree row law (upstream ba8b9969).

Geometry supplies signed distances, Jacobians and constraint row masses.
The row force is evaluated at the *smooth predicted* velocity; rows do not
iteratively feed their response into one another. See upstream forward.py.
"""
import numpy as np

COMFREE_STIFFNESS = 0.2
COMFREE_DAMPING = 0.001
COMFREE_IMPEDANCE_MIN = 0.9
COMFREE_IMPEDANCE_MAX = 0.95


def row_acceleration(velocity, distance, dt, stiffness=COMFREE_STIFFNESS,
                     damping=COMFREE_DAMPING):
    return -(damping / dt) * velocity - (stiffness / dt) * (dt * velocity + distance)


def impedance(distance, minimum=0.9, maximum=0.95):
    """Default solimp, including upstream's fixed 1 cm impedance width."""
    x = np.minimum(np.abs(distance) / 0.01, 1.0)
    curve = np.where(x < 0.5, 2 * x*x, 1 - 2*(1-x)**2)
    return minimum + (maximum - minimum) * curve


def row_mass(invweight, distance, minimum=0.9, maximum=0.95):
    imp = impedance(distance, minimum, maximum)
    return 1.0 / np.maximum(np.asarray(invweight) * (1-imp) / imp, 1e-15)


def reduced_row_masses(jacobian, inverse_mass, distance=0.0):
    """Reduced rigid-body contacts against a fixed environment.

    The upstream pyramidal model uses body_invweight0's translational average,
    shared by all four friction edges (without MuJoCo's friction/impratio
    multiplier). A reduced model supplies that average from its mass matrix.
    Zero padded rows have zero mass and therefore exert exactly zero force.
    """
    jacobian = np.asarray(jacobian)
    weight = np.trace(np.asarray(inverse_mass)[:3, :3]) / 3
    return np.where(np.any(jacobian != 0, axis=-1), row_mass(weight, distance), 0.0)
