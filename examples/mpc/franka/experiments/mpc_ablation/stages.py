"""A8 retains the baseline pose cost while removing stage transitions."""
from planning.object_pose_cost import ObjectCostStages


class PoseOnlyStages(ObjectCostStages):
    def __init__(self, args, body_axis):
        super().__init__(args, body_axis)
        self.stage = 'pose'

    def update(self, obs):
        # Reuse the baseline's stable-flip measurement, without permitting its
        # flip/pose transition or fallback. config() still includes all the
        # normal second-stage quaternion/upright/barrier terms.
        enabled = self.enabled
        self.enabled = False
        try:
            super().update(obs)
        finally:
            self.enabled = enabled
        return False


def make_object_cost_stages(args, body_axis):
    cls = PoseOnlyStages if args.ablation_mode == 'a8' else ObjectCostStages
    return cls(args, body_axis)
