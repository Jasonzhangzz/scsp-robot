"""Shared Cartesian actuator wrench limit for ranking, MPC, and OSC."""
import math


DEFAULT_CONTACT_FORCE_LIMIT = 10.0


def contact_force_limit(args):
    limit = float(getattr(args, 'max_contact_force', DEFAULT_CONTACT_FORCE_LIMIT))
    if not math.isfinite(limit) or limit <= .002:
        raise ValueError('max_contact_force must be finite and greater than .002 N')
    return limit


def contact_command_force_limit(args, stiffness):
    """Reachable K*u wrench within the executed Cartesian norm bound.

    OSC may use the full force cap for damping; the predictive actuator
    command must also respect the 5 mm (configurable) execution increment.
    """
    step = float(getattr(args, 'mpc_step_limit', .005))
    stiffness = float(stiffness)
    if not math.isfinite(step) or step <= 0 or not math.isfinite(stiffness) or stiffness <= 0:
        raise ValueError('Cartesian step and stiffness must be positive and finite')
    return min(contact_force_limit(args), stiffness * step)
