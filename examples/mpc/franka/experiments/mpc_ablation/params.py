"""Baseline scene and dynamics parameters, with ablation-only cost changes."""
import copy

import casadi as cs
import numpy as np

from examples.mpc.franka.runtime.params import (
    ExplicitMPCParams as BaselineParams,
    build_lambda_optimizer as _baseline_optimizer,
    add_lambda_predictor_args,
    _box_inertia_diag,
)
from .config import configure_ablation


def build_lambda_optimizer(param, args):
    if param.ablation_mode == 'a6':
        return None
    if param.ablation_mode == 'a3':
        from .obb import cached_mesh_obb
        obb = cached_mesh_obb(param.mesh_path_, args.a3_obb_cache_dir)
        # Only Lambda receives the surrogate. In particular the simulator's
        # meshes, MPC support vertices, mass, inertia and target stay intact.
        surrogate = copy.copy(param)
        surrogate.mesh_path_ = str(obb.mesh_path)
        surrogate.collision_mesh_paths_ = []
        surrogate.collision_hull = False
        surrogate.contact_sampling_ = 'surface'
        opt = _baseline_optimizer(surrogate, args)
        opt.obb_geometry = obb
        opt.contact_geometry = 'obb'
        return opt
    if param.ablation_mode != 'a5':
        return _baseline_optimizer(param, args)
    # Only the Lambda solver changes; MPC keeps --solver-backend as requested.
    lambda_args = copy.copy(args)
    lambda_args.solver_backend = 'cma-es'
    lambda_args.population_size = args.a5_population_size
    lambda_args.population_iterations = args.a5_population_iterations
    return _baseline_optimizer(param, lambda_args)


class ExplicitMPCParams(BaselineParams):
    def __init__(self, args, rand_seed=1, target_type='ground-rotation',
                 mpc_model='explicit', build_optimizer=True):
        configure_ablation(args)
        self.ablation_mode = args.ablation_mode
        self.a6_center_weight = float(args.a6_center_weight)
        super().__init__(args, rand_seed, target_type, mpc_model, build_optimizer=False)
        # Geometry center is the actor-frame AABB midpoint, transported by the
        # predicted pose. In particular, do not assume mesh origin equals center.
        self.object_geometry_center_local_ = .5 * (
            np.asarray(self.object_aabb_lo) + np.asarray(self.object_aabb_hi))
        if self.ablation_mode in ('a1', 'a2'):
            self.cpo_direct_trajectory = True
        if build_optimizer:
            self.lambda_optimizer = build_lambda_optimizer(self, args)

    def init_cost_fns(self):
        if self.ablation_mode == 'a6':
            from planning.franka_adamw_ablation import build_a6_cost_fns
            return build_a6_cost_fns(self)
        path, final = super().init_cost_fns()
        if self.ablation_mode not in ('a1', 'a2'):
            return path, final
        x, u, p = path.sx_in(0), path.sx_in(1), path.sx_in(2)
        # Preserve the baseline parameter contract used by every MPC backend.
        verify_index = 7 + self.max_ncon_ * 4 * (1 + self.n_qvel_)
        if self.ablation_mode in ('a1', 'a2'):
            contact = p[verify_index + 4:verify_index + 7]
            pose_cost = (self.pos_coef * cs.sumsqr(x[:3] - p[:3])
                         + self.ori_coef * (1 - cs.dot(x[3:7], p[3:7])**2))
            if self.runtime_object_cost_:
                from models.contact_geometry import attached_point_cs
                from planning.object_pose_cost import object_cost_cs
                contact = attached_point_cs(x, p[-19:-12], contact)
                pose_cost = object_cost_cs(x[:7], p[:7], p[-12:])
            cost = self.contact_coef * cs.sumsqr(x[7:10] - contact) + 50 * cs.sumsqr(u)
            if self.runtime_object_cost_:
                cost += pose_cost
            return (cs.Function('path_cost_fn', [x, u, p], [cost]),
                    cs.Function('final_cost_fn', [x, p], [10 * pose_cost]))
