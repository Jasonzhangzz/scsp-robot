"""Small public adapter surface for the optional Spider/MJWarp environment.

The Spider checkout is intentionally not vendored into scsp-robot.  A Python
3.12 runner creates its MJWP environment and passes it here; the environment
only needs to implement ``rollout_ee_delta_pose`` with the callback contract
documented by :class:`MJWarpBimanualRolloutBackend`.
"""

from .physical_bimanual_mppi import (
    MJWarpBimanualRolloutBackend,
    PhysicalBimanualMPPI,
    PhysicalRollout,
)


def build_spider_mjwp_planner(env, **planner_kwargs):
    """Construct the physical planner around a Spider MJWP environment."""
    return PhysicalBimanualMPPI(
        MJWarpBimanualRolloutBackend(env=env),
        **planner_kwargs,
    )

__all__ = [
    "MJWarpBimanualRolloutBackend",
    "PhysicalBimanualMPPI",
    "PhysicalRollout",
    "build_spider_mjwp_planner",
]
