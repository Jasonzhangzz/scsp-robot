import numpy as np

from planning.mpc_explicit2 import MPCExplicit as _BaseExplicitMPC


def _clip_xy(vec, max_norm):
    norm = np.linalg.norm(vec)
    if norm <= max_norm or norm < 1e-9:
        return vec
    return vec * (max_norm / norm)


class HumanoidFootActionLimiter:
    def __init__(
        self,
        max_step_x=0.06,
        max_step_y=0.05,
        max_step_z_up=0.05,
        max_step_z_down=0.03,
        max_step_xy_norm=0.065,
        max_foot_height=0.18,
        min_foot_height=0.015,
        max_foot_spread_y=0.32,
    ):
        self.max_step_x = float(max_step_x)
        self.max_step_y = float(max_step_y)
        self.max_step_z_up = float(max_step_z_up)
        self.max_step_z_down = float(max_step_z_down)
        self.max_step_xy_norm = float(max_step_xy_norm)
        self.max_foot_height = float(max_foot_height)
        self.min_foot_height = float(min_foot_height)
        self.max_foot_spread_y = float(max_foot_spread_y)

    def clip_delta(self, action):
        action = np.asarray(action, dtype=np.float64).copy()
        for start in (0, 3):
            delta = action[start:start + 3]
            delta[0] = np.clip(delta[0], -self.max_step_x, self.max_step_x)
            delta[1] = np.clip(delta[1], -self.max_step_y, self.max_step_y)
            delta[2] = np.clip(delta[2], -self.max_step_z_down, self.max_step_z_up)
            delta[:2] = _clip_xy(delta[:2], self.max_step_xy_norm)
            action[start:start + 3] = delta
        return action

    def clip_world_targets(self, left_target, right_target, pelvis_pos):
        left_target = np.asarray(left_target, dtype=np.float64).copy()
        right_target = np.asarray(right_target, dtype=np.float64).copy()
        pelvis_pos = np.asarray(pelvis_pos, dtype=np.float64)

        left_target[2] = np.clip(left_target[2], self.min_foot_height, self.max_foot_height)
        right_target[2] = np.clip(right_target[2], self.min_foot_height, self.max_foot_height)

        left_rel_y = np.clip(left_target[1] - pelvis_pos[1], 0.02, self.max_foot_spread_y)
        right_rel_y = np.clip(right_target[1] - pelvis_pos[1], -self.max_foot_spread_y, -0.02)
        left_target[1] = pelvis_pos[1] + left_rel_y
        right_target[1] = pelvis_pos[1] + right_rel_y
        return left_target, right_target


class MPCExplicitFootBall(_BaseExplicitMPC):
    def __init__(self, param, limiter=None):
        super().__init__(param)
        self.limiter = limiter or HumanoidFootActionLimiter()

    def plan_once(
        self,
        target_p,
        target_q,
        curr_x,
        phi_vec,
        jac_mat,
        verify_cost_param,
        virtual_point,
        contact_point,
        curr_ori_coef,
        sol_guess=None,
    ):
        sol = super().plan_once(
            target_p,
            target_q,
            curr_x,
            phi_vec,
            jac_mat,
            verify_cost_param,
            virtual_point,
            contact_point,
            curr_ori_coef,
            sol_guess=sol_guess,
        )
        raw_action = np.asarray(sol["action"], dtype=np.float64)
        sol["action_raw"] = raw_action.copy()
        sol["action"] = self.limiter.clip_delta(raw_action)
        return sol


def compute_limited_foot_yaws(ball_pos, pelvis_yaw, left_pos, right_pos, yaw_limit=0.35):
    ball_pos = np.asarray(ball_pos, dtype=np.float64)
    left_pos = np.asarray(left_pos, dtype=np.float64)
    right_pos = np.asarray(right_pos, dtype=np.float64)

    def _single(foot_pos):
        direction = ball_pos[:2] - foot_pos[:2]
        if np.linalg.norm(direction) < 1e-6:
            return pelvis_yaw
        yaw = np.arctan2(direction[1], direction[0])
        delta = np.arctan2(np.sin(yaw - pelvis_yaw), np.cos(yaw - pelvis_yaw))
        delta = np.clip(delta, -yaw_limit, yaw_limit)
        return pelvis_yaw + delta

    return _single(left_pos), _single(right_pos)
