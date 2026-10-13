"""Material and contact-pair friction for the Franka rollout scenes."""
import math

ROLLOUT_TABLE_FRICTION = 0.5
ROLLOUT_OBJECT_FRICTION = 0.9
ROLLOUT_FINGERTIP_FRICTION = 1.5
DEFAULT_TANGENTIAL_FORCE_PENALTY = 0.03


def configure_scene_friction(param, combine_mode=None):
    """Match PhysX's average, or the legacy MuJoCo scene's maximum rule."""
    mode = combine_mode or getattr(param, 'friction_combine_mode_', 'average')
    if mode not in ('average', 'max'):
        raise ValueError('friction combine mode must be average or max')
    values = []
    for name, default in (
        ('table_friction_', ROLLOUT_TABLE_FRICTION),
        ('object_friction_', ROLLOUT_OBJECT_FRICTION),
        ('fingertip_friction_', ROLLOUT_FINGERTIP_FRICTION),
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
