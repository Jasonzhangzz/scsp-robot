"""Material and contact-pair friction for the Franka rollout scenes."""
import math

ROLLOUT_TABLE_FRICTION = 0.5
ROLLOUT_OBJECT_FRICTION = 0.9
ROLLOUT_FINGERTIP_FRICTION = 1.5
# PhysX averages the two materials. Raise fingertip/object grip to 1.6,
# keeping table/object friction at 0.7 so pose translation has the same drag.
ISAAC_TABLE_FRICTION = 0.2
ISAAC_OBJECT_FRICTION = 1.2
ISAAC_FINGERTIP_FRICTION = 2.0
DEFAULT_TANGENTIAL_FORCE_PENALTY = 0.03


def configure_scene_friction(param, combine_mode=None, *, physics_backend=None):
    """Match PhysX's average, or the legacy MuJoCo scene's maximum rule."""
    mode = combine_mode or getattr(param, 'friction_combine_mode_', 'average')
    if mode not in ('average', 'max'):
        raise ValueError('friction combine mode must be average or max')
    defaults = ((ISAAC_TABLE_FRICTION, ISAAC_OBJECT_FRICTION, ISAAC_FINGERTIP_FRICTION)
                if physics_backend == 'isaac' else
                (ROLLOUT_TABLE_FRICTION, ROLLOUT_OBJECT_FRICTION, ROLLOUT_FINGERTIP_FRICTION))
    values = []
    for name, default in (
        ('table_friction_', defaults[0]),
        ('object_friction_', defaults[1]),
        ('fingertip_friction_', defaults[2]),
    ):
        value = float(getattr(param, name, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError(name + ' must be finite and nonnegative')
        setattr(param, name, value)
        values.append(value)
    table, obj, fingertip = values
    combine = (lambda a, b: .5 * (a + b)) if mode == 'average' else max
    param.friction_combine_mode_ = mode
    param.mu_table_object_ = combine(table, obj)
    param.mu_fingertip_object_ = combine(fingertip, obj)
    # Existing support/transition helpers consume this environment coefficient.
    param.mu_object_ = param.mu_table_object_
    return param
