import numpy as np
import torch
import torch.nn.functional as F


def _sample_ctrls_impl(planner, ctrls, sample_params=None):
    sample_params = sample_params or {}
    global_noise_scale = float(sample_params.get("global_noise_scale", 1.0))
    noise = (
        torch.randn(
            planner.num_samples_,
            planner.horizon_,
            planner.n_cmd_,
            device=planner.device_,
            dtype=planner.dtype_,
        )
        * planner.noise_scale_[None]
        * global_noise_scale
    )
    ctrls_samples = ctrls[None] + noise
    return torch.clamp(ctrls_samples, planner.u_lb_[None, None], planner.u_ub_[None, None])


if hasattr(torch, "compile"):
    _sample_ctrls_compiled = torch.compile(_sample_ctrls_impl)
else:
    _sample_ctrls_compiled = _sample_ctrls_impl


def _compute_weights_impl(costs, num_samples, temperature, elite_frac):
    rewards = -costs
    nan_mask = torch.isnan(rewards) | torch.isinf(rewards)
    rew_min = (
        rewards[~nan_mask].min()
        if (~nan_mask).any()
        else torch.tensor(-1e6, device=rewards.device, dtype=rewards.dtype)
    )
    rewards = torch.where(nan_mask, rew_min, rewards)

    top_k = max(1, int(elite_frac * num_samples))
    top_indices = torch.topk(rewards, k=top_k, largest=True).indices
    weights = torch.zeros_like(rewards)
    top_rewards = rewards[top_indices]
    top_rewards = (top_rewards - top_rewards.mean()) / (top_rewards.std() + 1e-2)
    weights[top_indices] = F.softmax(top_rewards / temperature, dim=0)
    return weights


if hasattr(torch, "compile"):
    _compute_weights_compiled = torch.compile(_compute_weights_impl)
else:
    _compute_weights_compiled = _compute_weights_impl


class MPPICabinet:
    def __init__(self, param):
        self.param_ = param
        self.horizon_ = int(self.param_.mpc_horizon_)
        self.n_qpos_ = int(self.param_.n_qpos_)
        self.n_qvel_ = int(self.param_.n_qvel_)
        self.n_cmd_ = int(self.param_.n_cmd_)
        self.max_ncon_ = int(self.param_.max_ncon_)
        self.n_robot_qpos_ = int(self.param_.n_robot_qpos_)

        self.num_samples_ = int(getattr(self.param_, "mppi_samples_", 512))
        self.num_iters_ = int(getattr(self.param_, "mppi_iterations_", 4))
        self.num_init_iters_ = int(getattr(self.param_, "mppi_init_iterations_", 8))
        self.temperature_ = float(getattr(self.param_, "mppi_lambda_", 1.0))
        self.noise_sigma_ = float(getattr(self.param_, "mppi_noise_sigma_", 0.01))
        self.noise_decay_ = float(getattr(self.param_, "mppi_noise_decay_", 0.85))
        self.elite_frac_ = float(getattr(self.param_, "mppi_elite_frac_", 0.1))
        self.use_torch_compile_ = bool(getattr(self.param_, "mppi_use_torch_compile_", False))

        requested_device = getattr(self.param_, "mppi_device_", None)
        if requested_device is None:
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_ = torch.device(requested_device)
        self.dtype_ = torch.float32

        self.u_lb_ = self._expand_cmd_bound(self.param_.mpc_u_lb_)
        self.u_ub_ = self._expand_cmd_bound(self.param_.mpc_u_ub_)
        horizon_scale = self.noise_decay_ ** torch.arange(
            self.horizon_, device=self.device_, dtype=self.dtype_
        )
        self.noise_scale_ = torch.full(
            (self.horizon_, self.n_cmd_),
            self.noise_sigma_,
            device=self.device_,
            dtype=self.dtype_,
        )
        self.noise_scale_ = self.noise_scale_ * horizon_scale[:, None]
        self.h_ = torch.tensor(float(self.param_.h_), device=self.device_, dtype=self.dtype_)
        self.obj_mass_ = torch.tensor(float(self.param_.obj_mass_), device=self.device_, dtype=self.dtype_)
        self.model_params_ = torch.tensor(float(self.param_.model_params), device=self.device_, dtype=self.dtype_)
        self.gravity_ = self._tensor(self.param_.gravity_)
        self.robot_stiff_ = self._tensor(self.param_.robot_stiff_)
        self.Q_inv_ = self._tensor(np.linalg.inv(self.param_.Q))

        self.contact_cost_param_ = float(self.param_.contact_cost_param)
        self.attract_coef_ = float(self.param_.attract_coef)
        self.reject_coef_ = float(self.param_.reject_coef)
        self.contact_coef_ = float(self.param_.contact_coef)
        self.reject_dis_ = float(self.param_.reject_dis)
        self.epsilon_ = 1e-3

        self.final_cost_mode_ = str(getattr(self.param_, "final_cost_mode_", "drawer_open"))
        self.drawer_open_axis_ = int(getattr(self.param_, "drawer_open_axis_", 0))
        self.drawer_open_weight_ = float(getattr(self.param_, "drawer_open_weight_", 500.0))
        self.drawer_lateral_weight_ = float(getattr(self.param_, "drawer_lateral_weight_", 25.0))
        self.drawer_quat_weight_ = float(getattr(self.param_, "drawer_quat_weight_", 0.0))
        self.w_ref_q_ = float(getattr(self.param_, "mppi_w_ref_q_", 30.0))
        self.w_ref_u_ = float(getattr(self.param_, "mppi_w_ref_u_", 3.0))
        self.w_joint_limit_ = float(getattr(self.param_, "mppi_w_joint_limit_", 5.0))

        self.q_lb_ = self._tensor(np.asarray(self.param_.mpc_q_lb_[-self.n_robot_qpos_ :], dtype=np.float32))
        self.q_ub_ = self._tensor(np.asarray(self.param_.mpc_q_ub_[-self.n_robot_qpos_ :], dtype=np.float32))
        self.u_traj_ = torch.zeros((self.horizon_, self.n_cmd_), device=self.device_, dtype=self.dtype_)
        self._ref_q_traj = None
        self._ref_u_traj = None
        self._init_fk_constants()

    def _tensor(self, value):
        return torch.as_tensor(value, device=self.device_, dtype=self.dtype_)

    def _expand_cmd_bound(self, bound):
        arr = np.broadcast_to(np.asarray(bound, dtype=np.float32), (self.n_cmd_,)).copy()
        return torch.as_tensor(arr, device=self.device_, dtype=self.dtype_)

    def _init_fk_constants(self):
        dh = np.asarray(
            [
                (0.0, 0.0, 0.333),
                (0.0, -np.pi / 2, 0.0),
                (0.0, np.pi / 2, 0.316),
                (0.0825, np.pi / 2, 0.0),
                (-0.0825, -np.pi / 2, 0.384),
                (0.0, np.pi / 2, 0.0),
                (0.088, np.pi / 2, 0.0),
            ],
            dtype=np.float32,
        )
        self._fk_a = self._tensor(dh[:, 0])
        self._fk_d = self._tensor(dh[:, 2])
        alpha = self._tensor(dh[:, 1])
        self._fk_alpha_cos = torch.cos(alpha)
        self._fk_alpha_sin = torch.sin(alpha)
        self._fk_eye4 = torch.eye(4, device=self.device_, dtype=self.dtype_)

        attach_quat = self._tensor([0.3826834, 0.0, 0.0, 0.9238795])
        w, x, y, z = attach_quat
        attach_rot = torch.stack(
            [
                torch.stack([w * w + x * x - y * y - z * z, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)]),
                torch.stack([2.0 * (x * y + w * z), w * w - x * x + y * y - z * z, 2.0 * (y * z - w * x)]),
                torch.stack([2.0 * (x * z - w * y), 2.0 * (y * z + w * x), w * w - x * x - y * y + z * z]),
            ]
        )
        self._fk_attach = self._fk_eye4.clone()
        self._fk_attach[:3, :3] = attach_rot
        self._fk_attach[:3, 3] = self._tensor([0.0, 0.0, 0.107])

        self._fk_tip = self._fk_eye4.clone()
        self._fk_tip[2, 3] = 0.06
        self._fk_tool = self._fk_attach @ self._fk_tip

    @staticmethod
    def _shift_traj(ctrls):
        shifted = torch.zeros_like(ctrls)
        shifted[:-1] = ctrls[1:]
        shifted[-1] = ctrls[-1]
        return shifted

    def _sample_ctrls(self, ctrls, sample_params=None):
        if self.use_torch_compile_:
            return _sample_ctrls_compiled(self, ctrls, sample_params)
        return _sample_ctrls_impl(self, ctrls, sample_params)

    def _compute_weights(self, costs):
        if self.use_torch_compile_:
            return _compute_weights_compiled(
                costs, self.num_samples_, self.temperature_, self.elite_frac_
            )
        return _compute_weights_impl(
            costs, self.num_samples_, self.temperature_, self.elite_frac_
        )

    def _quat_body_mat(self, quat):
        return torch.stack(
            [
                torch.stack([-quat[..., 1], quat[..., 0], quat[..., 3], -quat[..., 2]], dim=-1),
                torch.stack([-quat[..., 2], -quat[..., 3], quat[..., 0], quat[..., 1]], dim=-1),
                torch.stack([-quat[..., 3], quat[..., 2], -quat[..., 1], quat[..., 0]], dim=-1),
            ],
            dim=-2,
        )

    def _franka_fk_pos(self, q):
        batch = q.shape[0]
        theta = q[:, : self.n_robot_qpos_]
        ct = torch.cos(theta)
        st = torch.sin(theta)

        Ti = torch.zeros(
            (batch, self.n_robot_qpos_, 4, 4),
            device=q.device,
            dtype=q.dtype,
        )
        Ti[:, :, 0, 0] = ct
        Ti[:, :, 0, 1] = -st
        Ti[:, :, 0, 3] = self._fk_a.unsqueeze(0)
        Ti[:, :, 1, 0] = st * self._fk_alpha_cos.unsqueeze(0)
        Ti[:, :, 1, 1] = ct * self._fk_alpha_cos.unsqueeze(0)
        Ti[:, :, 1, 2] = -self._fk_alpha_sin.unsqueeze(0)
        Ti[:, :, 1, 3] = -self._fk_d.unsqueeze(0) * self._fk_alpha_sin.unsqueeze(0)
        Ti[:, :, 2, 0] = st * self._fk_alpha_sin.unsqueeze(0)
        Ti[:, :, 2, 1] = ct * self._fk_alpha_sin.unsqueeze(0)
        Ti[:, :, 2, 2] = self._fk_alpha_cos.unsqueeze(0)
        Ti[:, :, 2, 3] = self._fk_d.unsqueeze(0) * self._fk_alpha_cos.unsqueeze(0)
        Ti[:, :, 3, 3] = 1.0

        T01 = torch.bmm(Ti[:, 0], Ti[:, 1])
        T23 = torch.bmm(Ti[:, 2], Ti[:, 3])
        T45 = torch.bmm(Ti[:, 4], Ti[:, 5])
        T0123 = torch.bmm(T01, T23)
        T012345 = torch.bmm(T0123, T45)
        T = torch.bmm(T012345, Ti[:, 6])
        T_world = torch.bmm(T, self._fk_tool.unsqueeze(0).expand(batch, -1, -1))
        return T_world[:, :3, 3]

    def _step_batch(self, qpos, cmd, phi_vec, jac_mat):
        batch = qpos.shape[0]
        b_o = (self.obj_mass_ * self.gravity_).expand(batch, -1)
        b_r = cmd @ self.robot_stiff_.T
        b = torch.cat([b_o, b_r], dim=-1)

        q_inv_b = b @ self.Q_inv_.T
        v_non_contact = q_inv_b / self.h_
        contact_term = q_inv_b @ jac_mat.T + phi_vec[None]
        contact_force = torch.clamp(-self.model_params_ * contact_term, min=0.0)
        v_contact = (contact_force @ jac_mat) @ self.Q_inv_.T / self.h_
        qvel = v_non_contact + v_contact

        next_obj_pos = qpos[:, 0:3] + self.h_ * qvel[:, 0:3]
        quat = qpos[:, 3:7]
        quat_update = torch.bmm(self._quat_body_mat(quat).transpose(1, 2), qvel[:, 3:6].unsqueeze(-1)).squeeze(-1)
        next_obj_quat = quat + 0.5 * self.h_ * quat_update
        next_robot_qpos = qpos[:, -self.n_robot_qpos_ :] + self.h_ * qvel[:, -self.n_robot_qpos_ :]
        return torch.cat([next_obj_pos, next_obj_quat, next_robot_qpos], dim=-1)

    def _log_barrier(self, point, virtual_point):
        diff = point - virtual_point[None]
        return torch.log(torch.sum(diff * diff, dim=-1) + self.epsilon_)

    def _path_cost(self, x, u, verify_cost, virtual_point, contact_point, ref_q=None, ref_u=None):
        q_robot = x[:, -self.n_robot_qpos_ :]
        ee_pos = self._franka_fk_pos(q_robot)
        contact_cost = torch.sum((x[:, 0:3] - ee_pos) ** 2, dim=-1)
        virtual_point_cost = self._log_barrier(ee_pos, virtual_point)
        diff = ee_pos - contact_point[None]
        dist2 = torch.sum(diff * diff, dim=-1)
        contact_point_cost = torch.expm1(10.0 * dist2)

        reject_distance = torch.sum((x[:, 0:2] - ee_pos[:, 0:2]) ** 2, dim=-1) + self.epsilon_
        obstacle_cost = torch.where(
            reject_distance < self.reject_dis_,
            1.0 / reject_distance,
            torch.zeros_like(reject_distance),
        )
        attract_cost = self.attract_coef_ * virtual_point_cost + self.reject_coef_ * obstacle_cost
        # base_cost = (1.0 - verify_cost) * attract_cost + self.contact_coef_ * verify_cost * (
        #     self.contact_cost_param_ * contact_cost + (1.0 - self.contact_cost_param_) * contact_point_cost
        # )
        base_cost = (1.0 - verify_cost) * attract_cost + self.contact_coef_ * verify_cost * (
            self.contact_cost_param_ * contact_cost + (1.0 - self.contact_cost_param_) * contact_point_cost
        )
        joint_limit_cost = torch.sum(torch.clamp(self.q_lb_[None] - q_robot, min=0.0) ** 2, dim=-1)
        joint_limit_cost = joint_limit_cost + torch.sum(torch.clamp(q_robot - self.q_ub_[None], min=0.0) ** 2, dim=-1)
        control_cost = 50.0 * torch.sum(u * u, dim=-1)

        ref_q_cost = torch.zeros_like(control_cost)
        ref_u_cost = torch.zeros_like(control_cost)
        if ref_q is not None:
            ref_q_cost = self.w_ref_q_ * torch.sum((q_robot - ref_q[None]) ** 2, dim=-1)
        if ref_u is not None:
            ref_u_cost = self.w_ref_u_ * torch.sum((u - ref_u[None]) ** 2, dim=-1)

        return 500.0 * base_cost + control_cost + ref_q_cost + ref_u_cost + self.w_joint_limit_ * joint_limit_cost
        # return contact_point_cost

    def _final_cost(self, x, target_p, target_q):
        if self.final_cost_mode_ == "drawer_open":
            axis = self.drawer_open_axis_
            axis_cost = (x[:, axis] - target_p[None, axis]) ** 2
            pos_vec = x[:, 0:3] - target_p[None]
            lateral_cost = torch.sum(pos_vec ** 2, dim=-1) - axis_cost
            quaternion_cost = 1.0 - torch.sum(x[:, 3:7] * target_q[None], dim=-1) ** 2
            return 10.0 * (
                self.drawer_open_weight_ * axis_cost
                + self.drawer_lateral_weight_ * lateral_cost
                + self.drawer_quat_weight_ * quaternion_cost
            )
        position_cost = torch.sum((x[:, 0:3] - target_p[None]) ** 2, dim=-1)
        quaternion_cost = 1.0 - torch.sum(x[:, 3:7] * target_q[None], dim=-1) ** 2
        return 10.0 * (500.0 * position_cost + 5.0 * quaternion_cost * 4.0)

    def _extract_warm_start(self, sol_guess):
        if isinstance(sol_guess, dict):
            u_traj = sol_guess.get("u_traj")
            if u_traj is not None:
                u_traj = self._tensor(u_traj)
                if tuple(u_traj.shape) == (self.horizon_, self.n_cmd_):
                    return u_traj.clone()
        if self._ref_u_traj is not None:
            return self._ref_u_traj.clone()
        return self.u_traj_.clone()

    def plan_once(
        self,
        target_p,
        target_q,
        curr_x,
        phi_vec,
        jac_mat,
        verify_cost_param=0.0,
        virtual_point=None,
        contact_point=None,
        curr_ori_coef=None,
        sol_guess=None,
        ref_joint_traj=None,
        ref_ctrl_traj=None,
    ):
        del curr_ori_coef
        target_p_t = self._tensor(target_p).reshape(3)
        target_q_t = self._tensor(target_q).reshape(4)
        curr_x_t = self._tensor(curr_x).reshape(self.n_qpos_)
        phi_vec_t = self._tensor(phi_vec).reshape(-1)
        jac_mat_t = self._tensor(jac_mat).reshape(self.max_ncon_ * 4, self.n_qvel_)
        verify_cost_t = self._tensor(np.asarray(verify_cost_param).reshape(1))[0]
        virtual_point_t = self._tensor(virtual_point).reshape(3)
        contact_point_t = self._tensor(contact_point).reshape(3)
        self._ref_q_traj = None if ref_joint_traj is None else self._tensor(ref_joint_traj).reshape(self.horizon_, self.n_robot_qpos_)
        self._ref_u_traj = None if ref_ctrl_traj is None else self._tensor(ref_ctrl_traj).reshape(self.horizon_, self.n_cmd_)

        ctrls = self._extract_warm_start(sol_guess)
        best_ctrls = ctrls.clone()
        best_cost = torch.tensor(float("inf"), device=self.device_, dtype=self.dtype_)
        best_rollout_q = None

        n_iters = self.num_iters_ if sol_guess is not None else self.num_init_iters_
        for i in range(n_iters):
            ctrl_samples = self._sample_ctrls(
                ctrls, {"global_noise_scale": self.noise_decay_ ** i}
            )
            x = curr_x_t[None].repeat(self.num_samples_, 1)
            total_cost = torch.zeros(self.num_samples_, device=self.device_, dtype=self.dtype_)
            rollout_q = []

            for t in range(self.horizon_):
                ref_q_t = None if self._ref_q_traj is None else self._ref_q_traj[t]
                ref_u_t = None if self._ref_u_traj is None else self._ref_u_traj[t]
                u_t = ctrl_samples[:, t]
                total_cost = total_cost + self._path_cost(
                    x,
                    u_t,
                    verify_cost_t,
                    virtual_point_t,
                    contact_point_t,
                    ref_q=ref_q_t,
                    ref_u=ref_u_t,
                )
                x = self._step_batch(x, u_t, phi_vec_t, jac_mat_t)
                rollout_q.append(x)

            total_cost = total_cost + self._final_cost(x, target_p_t, target_q_t)
            rollout_q = torch.stack(rollout_q, dim=1)

            best_idx = torch.argmin(total_cost)
            if total_cost[best_idx] < best_cost:
                best_cost = total_cost[best_idx]
                best_ctrls = ctrl_samples[best_idx].clone()
                best_rollout_q = rollout_q[best_idx].clone()

            weights = self._compute_weights(total_cost)
            ctrls = (weights[:, None, None] * ctrl_samples).sum(dim=0)
            ctrls = torch.clamp(ctrls, self.u_lb_[None], self.u_ub_[None])

        self.u_traj_ = self._shift_traj(best_ctrls)
        result = {
            "action": best_ctrls[0].detach(),
            "sol_guess": {
                "u_traj": self.u_traj_.detach().clone(),
                "best_u_traj": best_ctrls.detach().clone(),
                "ref_u_traj": None if self._ref_u_traj is None else self._ref_u_traj.detach().clone(),
                "opt_cost": float(best_cost.detach().item()),
            },
            "cost_opt": torch.tensor([float(best_cost.detach().item())], device=self.device_, dtype=self.dtype_),
            "solve_status": "mppi_cabinet_torch",
        }
        if best_rollout_q is not None:
            result["rollout_q"] = best_rollout_q.detach().clone()
        return result
