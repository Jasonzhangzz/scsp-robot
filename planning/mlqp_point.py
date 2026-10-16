import time
import numpy as np
import casadi as cs
try:
    import torch
except ImportError:  # keep the IPOPT backend usable in non-Torch installs
    torch = None
import os
import sys
import ctypes
import hashlib
from models.soft_scm_contact import (
    SOFT_SCM_RESPONSE_TIME, SOFT_SCM_DAMPING_RATIO, SOFT_SCM_REGULARIZATION,
    normalize_contact_model, soft_scm_coefficients, inverse_diagonal_numpy, inverse_diagonal_cs,
)
from scipy.sparse.csgraph import dijkstra
try:
    # The reference fingertip experiment samples mesh vertices.  Keep the
    # original projection helper here; ``project_point1`` uses face centers
    # and therefore changes the contact-point candidates.
    from project_point import ProjectionPoint
except:
    from planning.project_point import ProjectionPoint


def _set_acados_code_export_directory(ocp, directory):
    options = getattr(ocp, 'code_gen_opts', None)
    if options is not None:
        options.code_export_directory = directory
    elif hasattr(ocp, 'code_export_directory'):
        ocp.code_export_directory = directory
    else:
        raise AttributeError(
            'AcadosOcp exposes neither code_gen_opts nor code_export_directory')


def _acados_codegen_cache_is_current(json_file, shared_file):
    if not (os.path.isfile(json_file) and os.path.isfile(shared_file)):
        return False
    try:
        import json
        with open(json_file, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        return all(key in payload for key in ('acados_lib_path',
                                              'code_export_directory'))
    except (OSError, ValueError, TypeError):
        return False

from models.scm_parameters import (scaled_inverse_cs, scaled_inverse_numpy,
    friction_jacobian_cs, friction_jacobian_numpy, response_cs, validate_theta)

from planning.object_pose_cost import (cost_config, object_cost_cs, object_cost_np, object_cost_torch,
                                       extended_cost_config, OBJECT_COST_CONFIG_SIZE)

class LambdaContactControlOptimizer:
    friction_cone = 'box'
    _friction_force_cap_factor = 2.0

    def __init__(self, mesh_path, obj_mass=0.01, arm_friction=0.9, 
                 contact_stiffness=0.2, time_step=0.01, max_contacts=10, sample_num=70,
                 pos_coef=1, ori_coef=0.0005, friction_reg_coef=0.0,
                 force_reg_coef=0.0,
                 max_contact_force=10.0,
                 contact_switch_radius=0.03,
                 contact_switch_margin_ratio=0.2,
                 contact_switch_margin_abs=1e-3,
                 scale_factors=[1.0, 1.0, 1.0],
                 collision_hull=False,
                 normal_stability_cos=0.90,
                 solver='acados', torch_max_iter=100,
                 fingertip_clearance=0.011,
                 obj_inertia=None,
                 dynamics_scale=None,
                 wrench_is_force=False,
                 top_k=2, comfree_damping=0.001,
                 contact_model='soft-scm',
                 soft_scm_response_time=SOFT_SCM_RESPONSE_TIME,
                 soft_scm_damping_ratio=SOFT_SCM_DAMPING_RATIO,
                 soft_scm_regularization=SOFT_SCM_REGULARIZATION,
                 contact_dynamics=None, grouped_contact_response=False, robot_stiffness=None,
                 surface_sampling=False, collision_part_paths=None, fingertip_radius=.01,
                 shared_response=None, friction_cone='box',
                 population_size=24, population_iterations=12,
                 population_seed=0, cem_elite_fraction=0.25):
        self.shared_response = shared_response
        self.rank_object_velocity = np.zeros(6)
        self.rank_tip_velocity = np.zeros(3)
        # 系统参数
        self.m = obj_mass
        self.mu_arm_obj = arm_friction
        self.base_mu_arm_obj = float(arm_friction)
        if friction_cone not in ('box', 'pyramid'):
            raise ValueError('friction_cone must be box or pyramid')
        self.friction_cone = friction_cone
        # Four MPC force rays n +/- mu*t1, n +/- mu*t2 span an L1
        # pyramid. Independent tangent bounds instead admit its corners.
        self._friction_force_cap_factor = 1.0 if friction_cone == 'pyramid' else 2.0
        self.K_contact = float(contact_stiffness)
        self.comfree_damping = float(comfree_damping)
        if (not np.isfinite(self.K_contact) or not np.isfinite(self.comfree_damping)
                or self.K_contact <= 0 or self.comfree_damping < 0):
            raise ValueError("ComFree stiffness must be positive and damping nonnegative")
        self.h = time_step
        self.contact_model = normalize_contact_model(contact_model)
        self.soft_scm_response_time = float(soft_scm_response_time)
        self.soft_scm_damping_ratio = float(soft_scm_damping_ratio)
        self.soft_scm_regularization = float(soft_scm_regularization)
        if self.contact_model == 'diagonal-contact':
            self.soft_scm_a, self.soft_scm_b = 1., 0.
            if shared_response is not None:
                raise ValueError('Diagonal contact is a separate model; shared-response variants are unsupported')
        else:
            self.soft_scm_a, self.soft_scm_b = soft_scm_coefficients(
                self.h, self.soft_scm_response_time, self.soft_scm_damping_ratio,
                self.soft_scm_regularization)
        # Rollout receives a physical force from the fingertip PD loop.  The
        # legacy ideal-contact path historically supplied an impulse-like
        # lambda; retain that convention there for backwards compatibility.
        self.wrench_is_force = bool(wrench_is_force)
        if contact_dynamics not in (None, 'quasi-dynamic'):
            raise ValueError('lambda contact_dynamics must be None or quasi-dynamic')
        self.gravity_velocity_scale = 1. / self.h if contact_dynamics == 'quasi-dynamic' else self.h
        self.wrench_velocity_scale = (1. / self.h if contact_dynamics == 'quasi-dynamic'
                                      else (self.h if self.wrench_is_force else 1.))
        self.grouped_contact_response = bool(grouped_contact_response)
        self.robot_stiffness = None if robot_stiffness is None else float(robot_stiffness)
        if self.robot_stiffness is not None and (not np.isfinite(self.robot_stiffness)
                or self.robot_stiffness <= 0 or not self.grouped_contact_response
                or contact_dynamics != 'quasi-dynamic'):
            raise ValueError('robot stiffness requires grouped quasi-dynamic contact')
        self.max_contacts = max_contacts
        self.pp = ProjectionPoint(mesh_path, scale_factors,
                                  collision_hull=collision_hull,
                                  normal_stability_cos=normal_stability_cos,
                                  surface_sampling=surface_sampling,
                                  collision_part_paths=collision_part_paths)

        self.sample_num = sample_num
        self.top_k = max(1, int(top_k))
        if surface_sampling:
            self.sampling_frame = self.pp.sample_surface_with_normals(
                self.sample_num, fingertip_radius=float(fingertip_radius))
        else:
            self.sampling_frame = self.pp.sample_vertices_with_normals(num_samples=self.sample_num)
        self.sample_point = self.sampling_frame['points']
        self.normal = self.sampling_frame['normals']
        self.t1 = self.sampling_frame['tangent1']
        self.t2 = self.sampling_frame['tangent2']
        # These indices are graph anchors only. Surface positions and normals
        # must never be replaced by the nearest mesh vertex.
        self.sample_vertex_indices = np.asarray(
            self.pp.kdtree.query(self.sample_point)[1], dtype=np.int32)
        self.sample_geodesic = self._precompute_sample_geodesic()

        self.J_tilde = np.zeros([4 * self.max_contacts, 6])
        self.env_phi = np.zeros(4 * self.max_contacts)

        # 构建系统刚度矩阵Q
        self.obj_inertia = np.eye(6)
        if obj_inertia is None:
            # Keep a conditioned fallback for callers that do not provide a
            # physical model.  The MuJoCo fingertip benchmark passes its
            # measured generalized mass explicitly.
            self.obj_inertia[0:3, 0:3] = 50 * np.eye(3)
            self.obj_inertia[3:, 3:] = 0.05 * np.eye(3)
        else:
            candidate_inertia = np.asarray(obj_inertia, dtype=np.float64).reshape(6, 6)
            if (not np.isfinite(candidate_inertia).all() or
                    np.min(np.linalg.eigvalsh(0.5 * (candidate_inertia + candidate_inertia.T))) <= 0.0):
                raise ValueError('obj_inertia must be finite and positive definite')
            self.obj_inertia[:, :] = 0.5 * (candidate_inertia + candidate_inertia.T)
        Q = np.zeros((6,6))
        Q[:6, :6] = self.obj_inertia
        # Evaluate M^{-1} through a dimensionless coordinate transform.  This
        # is algebraically identical to inv(M), but avoids adding an absolute
        # 1e-8 regularizer that would overwhelm MuJoCo's ~1e-6 rotational
        # inertias.  If S maps normalized generalized velocities to physical
        # velocities, M_n=S^T M S and M^-1=S M_n^-1 S^T.
        if dynamics_scale is None:
            dynamics_scale = np.eye(6, dtype=np.float64)
        self.dynamics_scale = np.asarray(dynamics_scale, dtype=np.float64).reshape(6, 6)
        if (not np.isfinite(self.dynamics_scale).all() or
                np.min(np.abs(np.linalg.eigvals(self.dynamics_scale))) <= 0.0):
            raise ValueError('dynamics_scale must be finite and nonsingular')
        mass_scaled = self.dynamics_scale.T @ Q @ self.dynamics_scale
        self.scaled_mass_matrix = mass_scaled
        self.dynamics_condition_number = float(np.linalg.cond(mass_scaled))
        self.Q_inv = self.dynamics_scale @ np.linalg.inv(mass_scaled) @ self.dynamics_scale.T

        self.base_Q_inv = self.Q_inv.copy()
        self.scm_theta = np.zeros(6)
        self.pos_coef = pos_coef
        self.ori_coef = ori_coef
        # Penalize tangential (friction) force.  In the contact frame
        # lam_arm=[normal_force, tangent_1, tangent_2], so this is exactly
        # the squared deviation of the applied wrench from the object normal.
        self.friction_reg_coef = float(friction_reg_coef)
        self.force_reg_coef = float(force_reg_coef)
        self.max_contact_force = float(max_contact_force)
        self.contact_switch_radius = float(contact_switch_radius)
        self.contact_switch_margin_ratio = float(contact_switch_margin_ratio)
        self.contact_switch_margin_abs = float(contact_switch_margin_abs)
        # Distance from a surface point to the fingertip centre when
        # approaching along the outward normal.  This is used solely for the
        # ground reachability test; keeping it small preserves low points that
        # are useful while flipping an object.
        self.fingertip_clearance = max(float(fingertip_clearance), 0.0)
        if not np.isfinite(self.max_contact_force) or self.max_contact_force <= 0:
            raise ValueError('max_contact_force must be a positive finite value')
        # The public limit is the Euclidean wrench magnitude.  Together with
        # the friction cone this gives a conservative normal-force cap.
        self.max_normal_force = self.max_contact_force / np.sqrt(1.0 + self._friction_force_cap_factor * self.mu_arm_obj ** 2)
        if self.max_normal_force <= 0.001:
            raise ValueError('max_contact_force is too small for the minimum normal force')
        # Prefer acados; IPOPT remains the compiled fallback.
        # The Torch backend is selected explicitly by ``torch-lbfgs``.
        solver = str(solver).strip().lower()
        if solver == 'snopt':
            solver = 'acados'
        self.solver = solver
        from planning.contact_force_sampling import ForcePopulationSearch
        self.population_search = (ForcePopulationSearch(
            solver, population_size, population_iterations, population_seed,
            cem_elite_fraction) if solver in ('cma-es', 'cem') else None)
        self.torch_max_iter = int(torch_max_iter)
        self.last_solver_status = 'ipopt'
        # Diagnostics for rollout timing.  Acados normally solves each
        # candidate in a few milliseconds; a failed QP falls back to the
        # CasADi/IPOPT function and can take hundreds of milliseconds.  Keep
        # counters so callers can distinguish solver spikes from simulation
        # overhead without printing inside the hot loop.
        self.acados_solve_count = 0
        self.acados_failure_count = 0
        self.acados_fallback_count = 0
        self.acados_qp_failure_count = 0
        self.acados_failure_reasons = {}
        self.last_acados_failure_reason = None
        # A bad contact state can make every candidate's acados QP
        # infeasible.  Falling back to IPOPT for all samples then turns one
        # control cycle into a 0.6--1 s stall.  Keep a small per-cycle budget;
        # the remaining failed candidates are marked invalid and skipped.
        try:
            self.acados_max_fallbacks_per_cycle = max(
                0, int(os.environ.get('LAMBDA_ACADOS_MAX_FALLBACKS', '1')))
        except ValueError:
            self.acados_max_fallbacks_per_cycle = 1
        self._acados_fallbacks_this_cycle = 0
        self.point_idx = np.arange(self.sample_num)
        self.last_selected_local = None
        self.last_selected_idx = None
        self.last_anchor_sample_idx = None
        self.last_transition_cost = None
        self.last_selected_total_cost = None
        self.last_switch_required = False
        # A noisy contact objective can alternate between two distant mesh
        # samples when the fingertip is not yet in contact.  Requiring a
        # candidate to win on consecutive cycles prevents that one-cycle
        # chatter from moving the MPC target back and forth.  The incumbent
        # remains active while a switch is pending; a genuinely better patch
        # is still accepted quickly (two rollout periods).
        try:
            self.contact_switch_confirm_steps = max(
                1, int(os.environ.get('LAMBDA_CONTACT_SWITCH_CONFIRM_STEPS', '1')))
        except ValueError:
            self.contact_switch_confirm_steps = 1
        self._pending_selected_idx = None
        self._pending_selected_count = 0
        # Confidence in the current (nearest/anchored) surface patch.  A
        # value of one preserves the historical anti-switching behavior;
        # reducing it lets the raw lambda objective select a distant patch
        # when the anchored patch can no longer make progress.
        self.contact_switch_confidence = 1.0
        # Dwell / no-progress bookkeeping for the confidence schedule.
        # While a patch keeps lowering the task cost, confidence stays at 1
        # (lazy switching).  After a run of stagnant steps the confidence is
        # multiplied by gamma each cycle, which is the cheap on-policy
        # analogue of a discounted "time-on-this-action" penalty.
        self._dwell_idx = None
        self._dwell_steps = 0
        self._dwell_best_cost = None
        self._dwell_last_cost = None
        self._dwell_was_active = False
        self._dwell_blocked = False
        # Optional diagnostic mode: keep the incumbent sampled patch fixed
        # while its candidate remains finite.  This separates MPC tracking
        # from contact-point re-selection; normal rollouts leave it disabled.
        self.lock_contact_patch = False
        self.last_global_idx = None
        self.last_global_total_cost = None
        self.last_best_x_plus = None
        self.last_best_cost = None
        # Executed nearest-sample contact (p_arm).  This is a distinct
        # quantity from last_selected_idx (the globally ranked patch) and
        # must obey the same lock / confidence / block / curvature policy.
        self.last_executed_idx = None
        self.last_executed_x_plus = None
        self.last_executed_cost = None
        self.last_executed_force = None
        self.last_candidate_ids = None
        self.last_candidate_costs = None
        self.last_candidate_raw_costs = None
        self.last_candidate_deltas = None
        self.last_delta_lo = 0.0
        self.last_delta_hi = 0.0
        self.last_delta_center = 0.0
        self.last_delta_scale = 0.0
        self.last_pose_cost_now = 0.0
        self.last_cost_lo = 0.0
        self.last_cost_hi = 0.0
        self.last_best_delta = None
        self.last_candidate_pose_costs = None
        self.last_candidate_x_plus = None
        self.last_candidate_forces = None
        self.last_topk_ids = np.zeros(0, dtype=np.int32)
        self.rank_query_local = None
        self.last_topk_costs = np.zeros(0, dtype=np.float64)
        # Cycle-normalized ranking scores, keyed by sample id.  The NLP
        # still minimizes the physical pos/ori/force mix; verify and
        # best_contact compare the unit-range scores instead.
        self._rank_score_ids = None
        self._rank_score_values = None
        self.rank_score_ema_rate = 0.35
        self.last_candidate_delta_norms = None
        # Local mesh curvature is independent of sparse FPS sample spacing.
        self.region_max_point_curvature = 0.25
        self.region_max_mean_curvature = 0.10
        self.point_curvature = self._estimate_point_curvature()
        if self._drop_crease_samples():
            self.sample_vertex_indices = np.asarray(
                self.pp.kdtree.query(self.sample_point)[1], dtype=np.int32)
            self.sample_geodesic = self._precompute_sample_geodesic()
        # Temporary blacklist for patches that were reached without yielding
        # contact or pose progress.  This prevents immediate re-selection of
        # the same low-authority ear/foot neighborhood.
        self._blocked_contact_indices = {}
        self._contact_patch_failures = {}
        self.contact_patch_max_block_cycles = 320
        self.init_utils()
        self.optimization_fn = None
        self.force_objective_fn = None
        if self.solver == 'ipopt':
            self._precompile_optimization_function()
        elif self.population_search is not None:
            self._precompile_optimization_function(build_solver=False)
        self.acados_solver = None
        self.acados_error = None
        if self.solver not in ('ipopt', 'torch-lbfgs', 'torch-gn', 'cma-es', 'cem'):
            try:
                self.acados_solver = self._build_acados_contact_solver()
                self.solver = 'acados'
            except Exception as exc:
                self.acados_error = exc
                print(f'acados contact solver unavailable; using IPOPT fallback: {exc}')
                self.solver = 'ipopt'

    def object_cost_config(self):
        config = getattr(self, '_object_cost_config', None)
        return extended_cost_config(cost_config(self.pos_coef, self.ori_coef)) if config is None else config

    def set_object_cost_config(self, config):
        c = extended_cost_config(config)
        self._object_cost_config = np.r_[cost_config(*c[:3], body_axis=c[3:6], reference_up=c[6:9]), c[9:]]

    def _friction_residuals_cs(self, force, mu):
        """Nonpositive rows describe the configured tangential force cone."""
        if self.friction_cone == 'pyramid':
            return cs.vertcat(*[s1*force[1] + s2*force[2] - mu*force[0]
                               for s1, s2 in ((1, 1), (1, -1), (-1, 1), (-1, -1))])
        return cs.vertcat(force[1]-mu*force[0], -force[1]-mu*force[0],
                          force[2]-mu*force[0], -force[2]-mu*force[0])

    def _bounded_tangents_torch(self, raw):
        """Map single or batched proposals into the same cone as native solvers."""
        tangent = torch.tanh(raw)
        if self.friction_cone == 'pyramid':
            tangent = tangent / tangent.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        return tangent

    def _project_contact_force(self, force, force_limit):
        """Remove numerical constraint violations before reporting RTI predictions."""
        cap = float(force_limit) / np.sqrt(1 + self._friction_force_cap_factor*self.mu_arm_obj**2)
        fn = float(np.clip(force[0], 0.0, cap))
        ft = np.asarray(force[1:3], dtype=float).copy()
        bound = self.mu_arm_obj * fn
        if self.friction_cone == 'pyramid':
            size = float(np.sum(np.abs(ft)))
            if size > bound:
                ft *= bound / size
        else:
            ft = np.clip(ft, -bound, bound)
        return np.r_[fn, ft]

    def set_scm_parameters(self, theta):
        if self.contact_model != 'soft-scm':
            raise ValueError('Online calibration requires soft-SCM')
        self.scm_theta = validate_theta(theta)
        if hasattr(self, "base_mu_arm_obj"):
            self.mu_arm_obj = self.base_mu_arm_obj*np.exp(self.scm_theta[5])
            self.max_normal_force = self.max_contact_force/np.sqrt(1+self._friction_force_cap_factor*self.mu_arm_obj**2)
        self.Q_inv = scaled_inverse_numpy(self.base_Q_inv, self.scm_theta)
        self.soft_scm_a, self.soft_scm_b = soft_scm_coefficients(
            self.h, self.soft_scm_response_time*np.exp(self.scm_theta[4]),
            self.soft_scm_damping_ratio, self.soft_scm_regularization)

    def set_contact_force_limit(self, limit):
        """Runtime wrench cap in the configured units, without solver rebuild."""
        limit = float(limit)
        if not np.isfinite(limit) or limit <= .002:
            raise ValueError('contact force limit must be finite and > .002')
        self.max_contact_force = limit
        self.max_normal_force = limit / np.sqrt(1 + self._friction_force_cap_factor*self.mu_arm_obj**2)

    def update_Jacobian(self, J_tilde=None, phi=None):
        required_rows = 4 * self.max_contacts
        """更新环境接触雅可比矩阵"""
        if J_tilde is None:
            return self.J_tilde

        J_tilde = np.asarray(J_tilde)[:, :6]
        padded_J_tilde = np.zeros((required_rows, 6))
        valid_rows = min(J_tilde.shape[0], required_rows)
        padded_J_tilde[:valid_rows, :] = J_tilde[:valid_rows, :]
        self.J_tilde = padded_J_tilde
        self.env_phi[:] = 0.0
        if phi is not None:
            gaps = np.asarray(phi, dtype=float).reshape(-1)
            if gaps.size == (valid_rows + 3)//4:
                gaps = np.repeat(gaps, 4)
            if gaps.size < valid_rows or not np.isfinite(gaps).all():
                raise ValueError("phi must contain one gap per contact or per constraint row")
            self.env_phi[:valid_rows] = gaps[:valid_rows]
        return self.J_tilde

    def set_contact_switch_confidence(self, confidence):
        """Set the weight of surface-transition penalties in point selection.

        This is intentionally a small public hook used by rollout policies.
        The lambda objective itself is unchanged; only the hysteresis term
        that favors the current contact patch is scaled.
        """
        confidence = float(confidence)
        if not np.isfinite(confidence):
            raise ValueError('contact switch confidence must be finite')
        self.contact_switch_confidence = float(np.clip(confidence, 0.0, 1.0))
        return self.contact_switch_confidence

    def note_contact_progress(self, selected_idx, progress_cost, active=True,
                              gamma=0.85, min_dwell_steps=6, improve_eps=1e-3,
                              unlock_confidence=0.05, block_cycles=20,
                              dead_increment=False, merge_radius=None,
                              block_radius=None, time_decay=False,
                              recover=0.12):
        """Decay switch-confidence if one patch stops improving the task cost.

        ``progress_cost`` must decrease when the incumbent is useful (pose
        error, or the lambda objective).  ``active`` is false while the
        fingertip is still travelling: neither dwell time nor passive object
        motion earns evidence for that patch unless ``time_decay`` is set.
        With ``time_decay``, a long hover that never lands is discounted
        the same way as a stagnant contact (RL-style γ^t).  Neighbouring
        mesh samples share one fixed patch anchor, and repeated failed
        visits receive a longer, bounded cooldown.
        """
        if selected_idx is None:
            return self.contact_switch_confidence
        try:
            cost = float(progress_cost)
        except (TypeError, ValueError):
            return self.contact_switch_confidence
        if not np.isfinite(cost):
            return self.contact_switch_confidence

        idx = int(selected_idx)
        unlock = float(unlock_confidence)
        same_patch = self._dwell_idx is not None and idx == int(self._dwell_idx)
        if self._dwell_idx is not None and not same_patch:
            try:
                merge = float(self.contact_switch_radius if merge_radius is None
                              else merge_radius)
                same_patch = bool(
                    self.sample_geodesic[int(self._dwell_idx), idx] <= merge)
            except (AttributeError, IndexError, TypeError):
                pass
        # A cooldown that just expired permits a fresh attempt even if the
        # ranker had no alternative and kept returning the blocked sample.
        retry_expired = (getattr(self, '_dwell_blocked', False) and
                         idx not in self._blocked_contact_indices)
        already_blocked = (idx in getattr(self, '_blocked_contact_indices', {})
                           and not retry_expired)
        if already_blocked:
            # Still sitting on a blacklisted patch.  Do not restore
            # confidence, and keep the cooldown from expiring under the
            # fingertip.  Keep the original cluster id when jitter lands
            # on a blocked neighbour.
            if (self._dwell_idx is None or
                    int(self._dwell_idx) not in self._blocked_contact_indices):
                self._dwell_idx = idx
            self._dwell_blocked = True
            hold_radius = (float(block_radius) if block_radius is not None
                           else self.contact_switch_radius)
            self.block_contact_patch(
                idx, cycles=max(20, int(block_cycles)), radius=hold_radius)
            return self.contact_switch_confidence
        if not same_patch or retry_expired:
            carry = (bool(time_decay) and not retry_expired and
                     self._dwell_best_cost is not None and
                     cost >= float(self._dwell_best_cost) - float(improve_eps))
            self._dwell_idx = idx
            self._dwell_last_cost = cost
            self._dwell_was_active = bool(active) or bool(time_decay)
            self._dwell_blocked = False
            if carry:
                # Same failed episode, new sample.  Do not restore
                # confidence just because ranking hopped to a neighbour.
                self._dwell_steps = int(getattr(self, '_dwell_steps', 0)) + 1
                wait = max(1, int(min_dwell_steps))
                decay = float(np.clip(gamma, 0.0, 1.0))
                if self._dwell_steps >= wait:
                    self.set_contact_switch_confidence(
                        self.contact_switch_confidence * decay)
                return self.contact_switch_confidence
            self._dwell_steps = 0
            self._dwell_best_cost = cost
            self.set_contact_switch_confidence(1.0)
            return self.contact_switch_confidence

        last_cost = getattr(self, '_dwell_last_cost', self._dwell_best_cost)
        was_active = getattr(self, '_dwell_was_active', True)
        self._dwell_last_cost = cost
        traveling = (not bool(active)) and bool(time_decay)
        self._dwell_was_active = bool(active) or traveling
        if not active and not time_decay:
            return self.contact_switch_confidence
        if getattr(self, '_dwell_blocked', False):
            return self.contact_switch_confidence
        if traveling:
            dead_increment = False
        if not was_active and not traveling:
            # Rebase against the final travelling sample.  Otherwise gravity
            # or an earlier push during a lift resets accumulated failures on
            # the next approach, despite this patch doing no useful work.
            self._dwell_best_cost = last_cost
        best = self._dwell_best_cost
        improved = best is None or cost < float(best) - float(improve_eps)
        if improved:
            self._dwell_best_cost = cost
            # A dead on-patch visit is already a failed local contact.
            # Pose chatter (the object sliding a few millimetres while
            # rotation gets worse) must not wipe the failure streak.
            if not dead_increment:
                self._dwell_steps = 0
                gain = float(np.clip(recover, 0.0, 1.0))
                self.set_contact_switch_confidence(
                    self.contact_switch_confidence + gain)
                return self.contact_switch_confidence

        self._dwell_steps += 1
        # A contact that the fingertip already reached, but that cannot
        # produce a usable x_plus, is a dead local optimum.  Decay faster
        # and require fewer grace steps than a patch that is still moving
        # the object a little.
        wait = max(1, int(min_dwell_steps))
        decay = float(np.clip(gamma, 0.0, 1.0))
        if dead_increment:
            wait = max(1, wait // 2)
            decay = decay * decay
        if self._dwell_steps >= wait:
            self.set_contact_switch_confidence(
                self.contact_switch_confidence * decay)
            if self.contact_switch_confidence <= unlock:
                self.lock_contact_patch = False
                self._dwell_blocked = True
                if block_radius is None:
                    radius = 2.0 * self.contact_switch_radius
                else:
                    radius = float(block_radius)
                neighbours = np.flatnonzero(
                    self.sample_geodesic[int(self._dwell_idx)] <= radius)
                failures = getattr(self, '_contact_patch_failures', {})
                visits = 1 + max(
                    (failures.get(int(neighbour), 0) for neighbour in neighbours),
                    default=0)
                for neighbour in neighbours:
                    failures[int(neighbour)] = visits
                self._contact_patch_failures = failures
                max_cycles = max(1, int(getattr(
                    self, 'contact_patch_max_block_cycles', 320)))
                cooldown = min(
                    max_cycles,
                    max(40, int(block_cycles)) * 2 ** min(visits - 1, 8))
                # Count one failure per visit, not once per low-confidence
                # frame.  Keep the failed neighbourhood excluded long enough
                # to reach and evaluate a different patch before revisiting.
                self.block_contact_patch(
                    self._dwell_idx,
                    cycles=cooldown,
                    radius=radius)
                self.last_selected_idx = None
                self.last_selected_local = None
                self.last_executed_idx = None
                self.last_executed_x_plus = None
                self.last_executed_cost = None
                self.last_executed_force = None
                self._pending_selected_idx = None
                self._pending_selected_count = 0
        return self.contact_switch_confidence

    def block_contact_patch(self, sample_idx, cycles=20, radius=None):
        """Temporarily suppress a failed sample and nearby samples."""
        if sample_idx is None:
            return
        idx = int(sample_idx)
        if idx < 0 or idx >= len(self.sample_point):
            return
        if radius is None:
            radius = 1.5 * self.contact_switch_radius
        try:
            neighbours = np.flatnonzero(self.sample_geodesic[idx] <= float(radius))
        except Exception:
            neighbours = np.asarray([idx], dtype=np.int64)
        count = max(1, int(cycles))
        for neighbour in neighbours:
            key = int(neighbour)
            self._blocked_contact_indices[key] = max(
                count, int(self._blocked_contact_indices.get(key, 0)))

    def _estimate_point_curvature(self):
        """Use incident mesh faces, never other sparse ranking samples."""
        frame = self.sampling_frame
        if 'curvature' in frame:
            self.point_curvature_mean = np.asarray(frame['curvature_mean']).copy()
            return np.asarray(frame['curvature']).copy()
        ids = self.sample_vertex_indices
        self.point_curvature_mean = np.asarray(self.pp.vertex_curvature_mean)[ids].copy()
        return np.asarray(self.pp.vertex_curvature)[ids].copy()

    def _high_curvature_mask(self, ids=None):
        """True on corners / junctions, False on flat faces near a crease.

        ``point_curvature`` is the worst neighbour (0.5 at 90°).
        ``point_curvature_mean`` is the neighbourhood average.  A useful
        side face that sees one orthogonal neighbour has high max and
        low mean; a foot/leg junction or tip has both high.
        """
        n = int(len(np.asarray(self.point_curvature).reshape(-1)))
        if ids is None:
            ids = np.arange(n, dtype=np.int32)
        else:
            ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        if ids.size == 0 or n == 0:
            return np.zeros(ids.shape, dtype=bool)
        curv = np.asarray(self.point_curvature, dtype=np.float64).reshape(-1)
        high_max = curv[ids] > float(self.region_max_point_curvature)
        mean = getattr(self, 'point_curvature_mean', None)
        if mean is None:
            return high_max
        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        mean_thr = float(getattr(self, 'region_max_mean_curvature', 0.10))
        return high_max & (mean[ids] > mean_thr)

    def _destination_crease_mask(self, ids=None):
        """A genuine local mesh crease must not own ``best_contact``."""
        n = int(len(np.asarray(self.point_curvature).reshape(-1)))
        if ids is None:
            ids = np.arange(n, dtype=np.int32)
        else:
            ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        if ids.size == 0 or n == 0:
            return np.zeros(ids.shape, dtype=bool)
        curv = np.asarray(self.point_curvature, dtype=np.float64).reshape(-1)
        return curv[ids] > float(self.region_max_point_curvature)

    def _same_side_mask(self, ids, tip_local=None):
        """True when a sample sits on the fingertip's XY side of the COM.

        Object-frame samples have the COM at the origin.  ``dot(tip_xy,
        sample_xy) <= 0`` is the same test as ``_on_opposite_sides``.
        A near-COM sample (``||xy|| < 1.5 cm``) is treated as same-side
        so a belly patch can still win after a flip.
        """
        ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        if ids.size == 0:
            return np.zeros((0,), dtype=bool)
        tip = tip_local
        if tip is None:
            tip = getattr(self, 'rank_query_local', None)
        if tip is None:
            return np.ones(ids.shape, dtype=bool)
        tip = np.asarray(tip, dtype=np.float64).reshape(3)
        tip_h = tip[:2]
        if float(np.linalg.norm(tip_h)) < 1e-6:
            return np.ones(ids.shape, dtype=bool)
        pts = np.asarray(self.sample_point, dtype=np.float64)[ids]
        goal_h = pts[:, :2]
        gn = np.linalg.norm(goal_h, axis=1)
        return (gn < 0.015) | (goal_h @ tip_h > 0.0)

    def _stable_finite_pool(self, ids, finite_mask):
        """Finite candidates that are allowed to own ``best_contact``."""
        pool = np.asarray(finite_mask, dtype=bool).reshape(-1).copy()
        if getattr(self, 'point_curvature', None) is not None:
            pool &= ~self._destination_crease_mask(ids)
        return pool

    def _delta_near_tie(self, winner_delta, alt_delta, quality_frac=0.015):
        """True when ``alt`` is close enough that table-J noise can flip them."""
        if not np.isfinite(winner_delta) or not np.isfinite(alt_delta):
            return False
        # An absolute 1e-3 treats zero/negative progress as equivalent to a
        # useful small reduction. Keep the tolerance in active cost units.
        current_cost = abs(float(getattr(self, 'last_pose_cost_now', 0.0) or 0.0))
        if not np.isfinite(current_cost):
            current_cost = 0.0
        margin = max(float(quality_frac) * abs(float(winner_delta)),
                     1e-6 * current_cost, 1e-12)
        return float(alt_delta) >= float(winner_delta) - margin

    def _prefer_stable_ranking_local(self, ids, costs, finite_mask, local_idx):
        """Keep last_global off a crease; same-side only wins a near-tie.

        A hard same-side gate pinned ``best_contact`` to the occupied
        flank after first touch.  Opposite faces must still be allowed
        when they clearly reduce pose cost; the via then leaves and
        orbits instead of grazing.  Table-J 1e-2 ΔC ties stay on the
        fingertip's side so we do not hook-then-push.
        """
        ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        costs = np.asarray(costs, dtype=np.float64).reshape(-1)
        finite_mask = np.asarray(finite_mask, dtype=bool).reshape(-1)
        local_idx = int(local_idx)
        if local_idx < 0 or local_idx >= ids.size or not finite_mask[local_idx]:
            return local_idx
        stable = self._stable_finite_pool(ids, finite_mask)
        if not stable[local_idx] and np.any(stable):
            local_idx = int(np.flatnonzero(stable)[int(np.argmin(costs[stable]))])
        same = self._same_side_mask(ids)
        if same[local_idx] or not np.any(stable & same):
            return local_idx
        alt = int(np.flatnonzero(stable & same)[int(np.argmin(costs[stable & same]))])
        winner_d = self.pose_delta_for_sample(int(ids[local_idx]))
        alt_d = self.pose_delta_for_sample(int(ids[alt]))
        if winner_d is None or alt_d is None:
            if float(costs[alt]) <= float(costs[local_idx]) + 0.05:
                return alt
            return local_idx
        if self._delta_near_tie(winner_d, alt_d):
            return alt
        return local_idx

    def _drop_crease_samples(self):
        """Remove true corners / junctions from the contact set.

        Worst-neighbour-only dropping also deleted the flat side faces
        next to a crease.  Those faces are the ones whose x_plus can
        still improve pose; without them ranking falls back to belly
        or back patches that look obviously useless in the viewer.
        """
        curv = np.asarray(self.point_curvature, dtype=np.float64).reshape(-1)
        if curv.size == 0:
            return False
        keep = ~self._high_curvature_mask()
        n_keep = int(np.count_nonzero(keep))
        n_all = int(curv.size)
        min_keep = min(8, max(1, n_all // 2))
        if n_keep == n_all or n_keep < min_keep:
            return False
        self.sample_point = np.asarray(self.sample_point)[keep]
        self.normal = np.asarray(self.normal)[keep]
        self.t1 = np.asarray(self.t1)[keep]
        self.t2 = np.asarray(self.t2)[keep]
        self.sample_num = n_keep
        self.point_curvature = curv[keep]
        mean = getattr(self, 'point_curvature_mean', None)
        if mean is not None:
            self.point_curvature_mean = np.asarray(mean, dtype=np.float64)[keep]
        self.sampling_frame = {
            'points': self.sample_point,
            'normals': self.normal,
            'tangent1': self.t1,
            'tangent2': self.t2,
        }
        return True

    def _filter_contact_policy_indices(self, candidate_idx,
                                       drop_blocked=True,
                                       drop_high_curvature=True):
        """Apply best-contact policy gates to a sample index set.

        Ranking already uses the floor-clear sample pool.  The executed
        nearest point (p_arm) must additionally drop blacklisted patches
        and high-curvature / unstable tips so it cannot snap to a trunk
        vertex that choose_contact_points would refuse to keep.
        """
        ids = np.asarray(candidate_idx, dtype=np.int32).reshape(-1)
        if ids.size == 0:
            return ids
        keep = np.ones(ids.size, dtype=bool)
        if drop_blocked and self._blocked_contact_indices:
            blocked = np.asarray(
                [int(idx) in self._blocked_contact_indices for idx in ids],
                dtype=bool)
            if np.any(keep & ~blocked):
                keep &= ~blocked
        if drop_high_curvature:
            if getattr(self, 'point_curvature', None) is not None:
                high_curv = self._high_curvature_mask(ids)
                if np.any(keep & ~high_curv):
                    keep &= ~high_curv
            stability = (None if getattr(self.pp, 'surface_sampling', False)
                         else getattr(self.pp, 'vertex_normal_stability', None))
            if stability is not None and self.sample_vertex_indices is not None:
                vertex_idx = self.sample_vertex_indices[ids]
                valid = (vertex_idx >= 0) & (vertex_idx < len(stability))
                unstable = np.zeros(ids.size, dtype=bool)
                if np.any(valid):
                    unstable[valid] = (
                        stability[vertex_idx[valid]]
                        < float(getattr(self.pp, 'normal_stability_cos', 0.90)))
                if np.any(keep & ~unstable):
                    keep &= ~unstable
        filtered = ids[keep]
        return filtered if filtered.size else ids

    def filter_rankable_indices(self, candidate_idx):
        """Same curvature / stability gate used for executed p_arm.

        Ranking used to keep ear/trunk/foot-crease tips that execution then
        refused, so best_contact and the press target sat on different patches.
        """
        return self._filter_contact_policy_indices(
            candidate_idx, drop_blocked=False, drop_high_curvature=True)

    def select_executed_contact_idx(self, query_local, candidate_idx,
                                    sphere_radius=0.0):
        """Nearest sample under lock / confidence / block / curvature rules.

        This is the p_arm counterpart of ``_select_contact_candidate``.
        While confidence is high the incumbent executed patch is held
        (hard-lock, or a geodesic neighbourhood); after collapse the
        nearest unblocked, non-sharp sample is taken.
        """
        ids = self._filter_contact_policy_indices(candidate_idx)
        if ids.size == 0:
            ids = np.asarray(candidate_idx, dtype=np.int32).reshape(-1)
        if ids.size == 0:
            return 0
        query = np.asarray(query_local, dtype=np.float64).reshape(3)
        prev = self.last_executed_idx
        confidence = float(np.clip(
            getattr(self, 'contact_switch_confidence', 1.0), 0.0, 1.0))
        if (getattr(self, 'lock_contact_patch', False) and prev is not None
                and confidence >= (1.0 - 1e-9) and int(prev) in ids):
            return int(prev)
        if prev is not None and confidence >= (1.0 - 1e-9) and int(prev) in ids:
            try:
                geo = self.sample_geodesic[int(prev), ids]
                local = ids[geo <= float(self.contact_switch_radius)]
            except Exception:
                local = ids
            if local.size:
                ids = local
            else:
                return int(prev)
        pts = np.asarray(self.sample_point[ids], dtype=np.float64)
        # ``query_local`` is the fingertip centre.  When requested, compare it
        # with each candidate's sphere centre rather than the mesh surface;
        # this keeps the nearest-contact choice in the same geometry used by
        # the MPC contact target.
        radius = float(sphere_radius)
        if np.isfinite(radius) and radius > 0.0:
            pts = pts - radius * np.asarray(self.normal[ids], dtype=np.float64)
        return int(ids[int(np.argmin(np.linalg.norm(pts - query[None, :], axis=1)))])

    def resolve_executed_contact(self, query_local, candidate_idx,
                                 x_d, current_x, tau_o, v_last=None,
                                 sphere_radius=0.0):
        """Solve (or reuse) the lambda QP at the policy-filtered p_arm sample."""
        idx = self.select_executed_contact_idx(
            query_local, candidate_idx, sphere_radius=sphere_radius)
        self.last_executed_idx = int(idx)
        p_obj = np.asarray(self.sample_point[idx], dtype=np.float64)
        n_in = np.asarray(self.normal[idx], dtype=np.float64)
        ids = getattr(self, 'last_candidate_ids', None)
        if ids is not None:
            hits = np.flatnonzero(np.asarray(ids, dtype=np.int32) == int(idx))
            if hits.size:
                loc = int(hits[0])
                x_plus = None
                if self.last_candidate_x_plus is not None:
                    x_plus = np.asarray(self.last_candidate_x_plus[loc], dtype=np.float64).reshape(7)
                cost = float(self.last_candidate_costs[loc]) if self.last_candidate_costs is not None else float('inf')
                force = (np.asarray(self.last_candidate_forces[loc], dtype=np.float32)
                         if self.last_candidate_forces is not None
                         else np.zeros(3, dtype=np.float32))
                failed = (x_plus is None) or (not np.isfinite(cost))
                self.last_executed_x_plus = None if failed else x_plus
                self.last_executed_cost = cost
                self.last_executed_force = force
                return p_obj, -n_in, self.last_executed_x_plus, cost, {
                    'control_input': force,
                    'solver_failed': failed,
                    'solve_time': 0.0,
                    'resulting_pose': self.last_executed_x_plus,
                }
        return self.optimize_control_input(
            x_d, current_x, tau_o, p_arm=p_obj, v_last=v_last, sample_idx=idx)

    def compute_env_diag_inverse(self, J_tilde):
        """Frozen row weights in the historical stacked 4x4-block API.

        soft-SCM uses inverse diagonal Delassus; ComFree uses its prior row
        masses. Zero Jacobian padding exerts no reaction in either model.
        """
        if getattr(self, 'contact_model', 'comfree') in ('soft-scm', 'diagonal-contact'):
            j = friction_jacobian_numpy(J_tilde, self.scm_theta) if hasattr(self, "scm_theta") else J_tilde
            masses = inverse_diagonal_numpy(
                j, self.Q_inv, self.soft_scm_regularization)
        else:
            from models.comfree_contact import reduced_row_masses
            masses = reduced_row_masses(J_tilde, self.Q_inv, self.env_phi)
        return np.vstack([np.diag(masses[4*i:4*i+4]) for i in range(self.max_contacts)])

    def _arm_velocity_cs(self, velocity, tau, force, point, frame, env_j, phi, inverse, theta, arm_gap=0., full_velocity=False, tip_velocity=None, return_impulses=False):
        """Predict the driven fingertip contact using MPC's nine-DoF mobility.

        The decision is an actuator wrench, not a wrench applied directly to
        the object. It must pass through the unilateral/friction response.
        """
        from models.soft_scm_contact import grouped_contact_velocity_cs
        qi = type(inverse).zeros(9, 9)
        qi[:6, :6] = inverse
        qi[6:, 6:] = cs.DM.eye(3)/self.robot_stiffness
        if self.contact_model != 'diagonal-contact':
            env_j = friction_jacobian_cs(env_j, theta)
        relative = cs.horzcat(cs.DM.eye(3), -cs.skew(point), -cs.DM.eye(3))
        mu = (self.base_mu_arm_obj if self.contact_model == 'diagonal-contact'
              else self.base_mu_arm_obj*cs.exp(theta[5]))
        n, t1, t2 = frame[:, 0], frame[:, 1], frame[:, 2]
        pyramid = cs.vertcat((n+mu*t1).T, (n+mu*t2).T, (n-mu*t1).T, (n-mu*t2).T)
        j = cs.vertcat(cs.horzcat(env_j, cs.DM.zeros(env_j.size1(), 3)), pyramid@relative)
        if getattr(self, 'shared_response', None) is not None:
            if return_impulses:
                raise ValueError('Contact impulse export is not available for shared-response variants')
            from models.cartesian_contact_response import contact_response_cs
            settings = dict(self.shared_response)
            physical_inverse = cs.DM(np.diag(1 / np.asarray(settings.pop('mass'))))
            tip_velocity = cs.DM.zeros(3) if tip_velocity is None else tip_velocity
            result, _ = contact_response_cs(velocity, tip_velocity, force, tau,
                j, cs.vertcat(phi, cs.repmat(arm_gap, 4, 1)), physical_inverse, qi,
                **settings)
            return result if full_velocity else result[:6]
        free = cs.vertcat(velocity + self.gravity_velocity_scale*inverse@tau,
                          force/(self.h*self.robot_stiffness))
        if self.contact_model == 'diagonal-contact':
            from models.diagonal_contact import diagonal_contact_velocity_cs
            result = diagonal_contact_velocity_cs(free, j, cs.vertcat(phi, cs.repmat(arm_gap,4,1)),
                qi, self.h, self.soft_scm_regularization, return_impulses=return_impulses)
        else:
            a, b = response_cs(self.h, self.soft_scm_response_time, self.soft_scm_damping_ratio, theta)
            result = grouped_contact_velocity_cs(free, j, cs.vertcat(phi, cs.repmat(arm_gap,4,1)),
                qi, self.h, a, b, self.soft_scm_regularization, return_impulses=return_impulses)
        if return_impulses:
            velocity_next, impulses = result
            return (velocity_next if full_velocity else velocity_next[:6]), impulses
        return result if full_velocity else result[:6]

    def predict_contact_response(self, force_contact, point, frame, gravity,
                                 velocity=None, current_pose=None):
        """Evaluate the exact coupled SCM used by the contact NLP.

        In the quasi-dynamic model Q is a mobility/stiffness matrix: the
        constraint dual has units N/s, so h times that dual is the contact
        reaction in N. The optimization variable drives the modeled fingertip
        and is distinct from the predicted object contact reaction. The ideal
        direct-force executor uses that variable as a short point-force pulse.
        """
        if self.robot_stiffness is None or self.contact_model not in ('soft-scm', 'diagonal-contact'):
            raise ValueError('Contact reaction export requires coupled quasi-dynamic soft-SCM')
        force_contact = np.asarray(force_contact, dtype=float).reshape(3)
        point = np.asarray(point, dtype=float).reshape(3)
        frame = np.asarray(frame, dtype=float).reshape(3, 3)
        gravity = np.asarray(gravity, dtype=float).reshape(6)
        velocity = np.zeros(6) if velocity is None else np.asarray(velocity, dtype=float).reshape(6)
        current_pose = (np.array([0., 0., 0., 1., 0., 0., 0.]) if current_pose is None
                        else np.asarray(current_pose, dtype=float).reshape(7))
        fn = getattr(self, '_contact_response_export_fn', None)
        if fn is None:
            v, g, f, p = [cs.SX.sym(name, count) for name, count in
                           (('v', 6), ('g', 6), ('f', 3), ('p', 3))]
            r = cs.SX.sym('frame', 3, 3)
            j = cs.SX.sym('env_j', 4*self.max_contacts, 6)
            phi = cs.SX.sym('phi', 4*self.max_contacts)
            inverse = cs.SX.sym('inverse', 6, 6)
            theta = cs.SX.sym('theta', 6)
            full, impulses = self._arm_velocity_cs(
                v, g, r@f, p, r, j, phi, inverse, theta,
                full_velocity=True, return_impulses=True)
            n, t = impulses[-1]
            reaction = self.h * (r[:, 0]*n + r[:, 1:]@t)
            env_reactions = cs.vertcat(*[self.h*cs.vertcat(n, t).T
                                         for n, t in impulses[:-1]])
            fn = self._contact_response_export_fn = cs.Function(
                'lambda_contact_response', [v, g, f, p, r, j, phi, inverse, theta],
                [full, reaction, env_reactions])
        full, contact, reactions = fn(velocity, gravity, force_contact, point, frame,
                                     self.J_tilde, self.env_phi, self.Q_inv, self.scm_theta)
        full = np.asarray(full).reshape(9)
        contact = np.asarray(contact).reshape(3)
        return dict(v_plus=full[:6], full_velocity=full,
                    x_plus=np.asarray(self.cs_qposInteg_(current_pose, full[:6])).reshape(7),
                    contact_force_body=contact,
                    contact_force_contact=frame.T @ contact,
                    environment_reactions=np.asarray(reactions).reshape(-1, 3),
                    model='coupled-quasi-dynamic-'+self.contact_model)

    def _arm_velocity_numpy(self, velocity, tau, force, point, frame, env_j, phi):
        if getattr(self, 'shared_response', None) is not None:
            result = self._arm_velocity_cs(cs.DM(velocity), cs.DM(tau), cs.DM(force),
                cs.DM(point), cs.DM(frame), cs.DM(env_j), cs.DM(phi), cs.DM(self.Q_inv),
                cs.DM(self.scm_theta), tip_velocity=cs.DM(self.rank_tip_velocity))
            return np.asarray(result).reshape(6)
        from models.soft_scm_contact import grouped_contact_velocity_numpy
        qi = np.zeros((9, 9)); qi[:6, :6] = self.Q_inv
        qi[6:, 6:] = np.eye(3)/self.robot_stiffness
        free = np.r_[velocity + self.gravity_velocity_scale*self.Q_inv@tau,
                     force/(self.h*self.robot_stiffness)]
        x, y, z = point
        skew = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        relative = np.c_[np.eye(3), -skew, -np.eye(3)]
        n, t1, t2 = frame.T
        mu = self.mu_arm_obj
        pyramid = np.stack([n+mu*t1, n+mu*t2, n-mu*t1, n-mu*t2])
        j = np.r_[np.c_[friction_jacobian_numpy(env_j, self.scm_theta), np.zeros((env_j.shape[0], 3))], pyramid@relative]
        if self.contact_model == 'diagonal-contact':
            from models.diagonal_contact import diagonal_contact_velocity_numpy
            return diagonal_contact_velocity_numpy(free, j, np.r_[phi, np.zeros(4)], qi,
                self.h, self.soft_scm_regularization)[:6]
        return grouped_contact_velocity_numpy(free, j, np.r_[phi, np.zeros(4)], qi,
            self.h, self.soft_scm_a, self.soft_scm_b, self.soft_scm_regularization)[:6]

    def _arm_velocity_torch(self, velocity, tau, force, point, frame, env_j, phi, inverse):
        from models.soft_scm_contact import grouped_contact_velocity_torch
        qi = torch.block_diag(inverse, torch.eye(3, device=inverse.device, dtype=inverse.dtype)/self.robot_stiffness)
        free = torch.cat([velocity + self.gravity_velocity_scale*(inverse@tau),
                          force/(self.h*self.robot_stiffness)])
        zero = point[0]*0
        skew = torch.stack([torch.stack([zero,-point[2],point[1]]),
                            torch.stack([point[2],zero,-point[0]]),
                            torch.stack([-point[1],point[0],zero])])
        relative = torch.cat([torch.eye(3, device=point.device, dtype=point.dtype), -skew,
                              -torch.eye(3, device=point.device, dtype=point.dtype)], dim=1)
        n,t1,t2 = frame.T
        mu = self.mu_arm_obj
        pyramid = torch.stack([n+mu*t1,n+mu*t2,n-mu*t1,n-mu*t2])
        blocks = env_j.reshape(-1,4,6); normal = blocks.mean(dim=1,keepdim=True)
        env_j = (normal + float(np.exp(self.scm_theta[5]))*(blocks-normal)).reshape(env_j.shape)
        j = torch.cat([torch.cat([env_j, env_j.new_zeros((env_j.shape[0],3))],dim=1), pyramid@relative])
        if self.contact_model == 'diagonal-contact':
            from models.diagonal_contact import diagonal_contact_velocity_torch
            return diagonal_contact_velocity_torch(free,j,torch.cat([phi,phi.new_zeros(4)]),qi,
                self.h,self.soft_scm_regularization)[:6]
        if getattr(self, 'shared_response', None) is not None:
            from models.cartesian_contact_response import contact_response_torch
            settings = dict(self.shared_response)
            physical_inverse = torch.diag(1/torch.as_tensor(settings.pop('mass'), dtype=inverse.dtype, device=inverse.device))
            tip_velocity = torch.as_tensor(self.rank_tip_velocity, dtype=inverse.dtype, device=inverse.device)
            result, _ = contact_response_torch(velocity, tip_velocity, force, tau,
                j, torch.cat([phi, phi.new_zeros(4)]), physical_inverse, qi, **settings)
            return result[:6]
        return grouped_contact_velocity_torch(free,j,torch.cat([phi,phi.new_zeros(4)]),qi,
            self.h,self.soft_scm_a,self.soft_scm_b,self.soft_scm_regularization)[:6]

    def _contact_velocity_cs(self, smooth, jacobian, weights, phi, inverse=None, theta=None):
        if self.contact_model == 'diagonal-contact':
            from models.diagonal_contact import diagonal_contact_velocity_cs
            mobility = cs.DM(self.Q_inv) if inverse is None else inverse
            return diagonal_contact_velocity_cs(smooth, jacobian, phi, mobility,
                self.h, self.soft_scm_regularization)
        if self.contact_model == 'comfree':
            return self._comfree_velocity_cs(smooth, jacobian, weights, phi)
        inverse = cs.DM(self.Q_inv) if inverse is None else inverse
        a, b = self.soft_scm_a, self.soft_scm_b
        if theta is not None:
            jacobian = friction_jacobian_cs(jacobian, theta)
            a, b = response_cs(self.h, self.soft_scm_response_time, self.soft_scm_damping_ratio, theta)
            d = inverse_diagonal_cs(jacobian, inverse, self.soft_scm_regularization)
            weights = cs.vertcat(*[cs.diag(d[4*i:4*i+4]) for i in range(self.max_contacts)])
        if getattr(self, 'grouped_contact_response', False):
            from models.soft_scm_contact import grouped_contact_velocity_cs
            return grouped_contact_velocity_cs(smooth, jacobian, phi, inverse, self.h, a, b, self.soft_scm_regularization)
        velocity = smooth
        for i in range(self.max_contacts):
            sl = slice(4*i, 4*(i+1))
            J = jacobian[sl, :]
            residual = a * (J @ smooth) + b * phi[sl] / self.h
            impulse = cs.fmax(-(weights[sl, :] @ residual), 0)
            velocity = velocity + inverse @ J.T @ impulse
        return velocity

    def _contact_velocity_numpy(self, smooth, jacobian, weights, phi):
        if self.contact_model == 'diagonal-contact':
            from models.diagonal_contact import diagonal_contact_velocity_numpy
            return diagonal_contact_velocity_numpy(smooth, jacobian, phi, self.Q_inv,
                self.h, self.soft_scm_regularization)
        if self.contact_model == 'comfree':
            return self._comfree_velocity_numpy(smooth, jacobian, weights, phi)
        if hasattr(self, "scm_theta"):
            jacobian = friction_jacobian_numpy(jacobian, self.scm_theta)
        if getattr(self, 'grouped_contact_response', False):
            from models.soft_scm_contact import grouped_contact_velocity_numpy
            return grouped_contact_velocity_numpy(smooth, jacobian, phi, self.Q_inv, self.h, self.soft_scm_a, self.soft_scm_b, self.soft_scm_regularization)
        velocity = np.array(smooth, copy=True)
        for i in range(self.max_contacts):
            sl = slice(4*i, 4*(i+1))
            J = jacobian[sl]
            residual = self.soft_scm_a * (J @ smooth) + self.soft_scm_b * phi[sl] / self.h
            impulse = np.maximum(-(weights[sl] @ residual), 0)
            velocity += self.Q_inv @ J.T @ impulse
        return velocity

    def _contact_velocity_torch(self, smooth, jacobian, weights, phi, inverse):
        if self.contact_model == 'diagonal-contact':
            from models.diagonal_contact import diagonal_contact_velocity_torch
            return diagonal_contact_velocity_torch(smooth, jacobian, phi, inverse,
                self.h, self.soft_scm_regularization)
        if self.contact_model == 'comfree':
            return self._comfree_velocity_torch(smooth, jacobian, weights, phi, inverse)
        if hasattr(self, "scm_theta"):
            blocks = jacobian.reshape(-1, 4, jacobian.shape[1])
            normal = blocks.mean(dim=1, keepdim=True)
            jacobian = (normal + float(np.exp(self.scm_theta[5]))*(blocks-normal)).reshape(jacobian.shape)
        if getattr(self, 'grouped_contact_response', False):
            from models.soft_scm_contact import grouped_contact_velocity_torch
            return grouped_contact_velocity_torch(smooth, jacobian, phi, inverse, self.h, self.soft_scm_a, self.soft_scm_b, self.soft_scm_regularization)
        velocity = smooth
        for i in range(self.max_contacts):
            sl = slice(4*i, 4*(i+1))
            J = jacobian[sl]
            residual = self.soft_scm_a * (smooth @ J.T) + self.soft_scm_b * phi[sl] / self.h
            impulse = torch.clamp(-(residual @ weights[sl].T), min=0)
            velocity = velocity + impulse @ (J @ inverse.T)
        return velocity

    def _comfree_velocity_cs(self, smooth, jacobian, masses, phi):
        velocity = smooth
        for i in range(self.max_contacts):
            sl = slice(4*i, 4*(i+1))
            J = jacobian[sl, :]
            speed = J @ smooth
            acceleration = -(self.comfree_damping/self.h)*speed
            acceleration -= (self.K_contact/self.h)*(self.h*speed + phi[sl])
            force = cs.fmax(masses[sl, :] @ acceleration, 0)
            velocity = velocity + self.h*cs.DM(self.Q_inv) @ J.T @ force
        return velocity

    def _comfree_velocity_numpy(self, smooth, jacobian, masses, phi):
        velocity = np.array(smooth, copy=True)
        for i in range(self.max_contacts):
            sl = slice(4*i, 4*(i+1)); J = jacobian[sl]
            speed = J @ smooth
            acceleration = -(self.comfree_damping/self.h)*speed
            acceleration -= (self.K_contact/self.h)*(self.h*speed + phi[sl])
            force = np.maximum(masses[sl] @ acceleration, 0)
            velocity += self.h*self.Q_inv @ J.T @ force
        return velocity

    def _comfree_velocity_torch(self, smooth, jacobian, masses, phi, inverse):
        velocity = smooth
        for i in range(self.max_contacts):
            sl = slice(4*i, 4*(i+1)); J = jacobian[sl]
            speed = smooth @ J.T
            acceleration = -(self.comfree_damping/self.h)*speed
            acceleration = acceleration - (self.K_contact/self.h)*(self.h*speed + phi[sl])
            force = torch.clamp(acceleration @ masses[sl].T, min=0)
            velocity = velocity + self.h*(force @ (J @ inverse.T))
        return velocity

    def _precompile_optimization_function(self, build_solver=True):
        """预编译优化函数 - 同时优化接触力和接触点位置"""
        opti = cs.Opti()
        
        # 定义优化变量和参数
        x_d = opti.parameter(7)
        current_x = opti.parameter(7)
        v_last = opti.parameter(6)
        tip_velocity = opti.parameter(3) if getattr(self, 'shared_response', None) is not None else None
        J_tilde = opti.parameter(4 * self.max_contacts, 6)
        D_inv = opti.parameter(4 * self.max_contacts, 4)
        env_phi = opti.parameter(4 * self.max_contacts)
        scm_theta = opti.parameter(6)
        qi = scaled_inverse_cs(self.base_Q_inv, scm_theta) if self.contact_model == "soft-scm" else cs.DM(self.Q_inv)
        tau_o_np = opti.parameter(6)
        p_arm = opti.parameter(3)    # 接触点位置
        n_arm = opti.parameter(3)
        t1 = opti.parameter(3)
        t2 = opti.parameter(3)
        curr_ori_coef = opti.parameter(1)
        object_config = opti.parameter(OBJECT_COST_CONFIG_SIZE)
        force_limit = opti.parameter()
        R_contact = cs.horzcat(n_arm, t1, t2)

        regularization_weight = opti.parameter()
        opti.set_value(regularization_weight, 0.01)

        # 机械臂接触力作为优化变量
        lam_arm = opti.variable(3)  # fn, ft1, ft2
        
        # 初始猜测
        opti.set_initial(lam_arm, [0.01, 0, 0])
        
        # 计算世界坐标系下的接触雅可比
        J_arm_world = self.compute_contact_jacobian(p_arm)
        
        # 构建b向量
        # Gravity is supplied as a force, whereas the dual contact variable
        # is an impulse.  Convert gravity to an impulse before solving for
        # the generalized velocity increment.
        # In physical rollout mode the optimized contact wrench is a force,
        # so convert it to an impulse over this control interval.  The legacy
        # ideal-contact mode keeps its historical impulse convention.
        wrench_scale = self.wrench_velocity_scale
        b = self.gravity_velocity_scale * tau_o_np + wrench_scale * cs.transpose(J_arm_world) @ (R_contact @ lam_arm)
        
        if self.contact_model == "soft-scm":
            v_now = self._contact_velocity_cs(v_last + qi @ b, J_tilde, D_inv, env_phi, qi, scm_theta)
        else:
            v_now = self._contact_velocity_cs(v_last + qi @ b, J_tilde, D_inv, env_phi)

        if self.robot_stiffness is not None:
            v_now = self._arm_velocity_cs(v_last, tau_o_np, R_contact@lam_arm, p_arm, R_contact, J_tilde, env_phi, qi, scm_theta, tip_velocity=tip_velocity)

        # 计算预测位姿x+
        x_plus = self.cs_qposInteg_(current_x, v_now)
        
        # 目标函数
        friction_cost = cs.sumsqr(lam_arm[1:3])
        force_cost = cs.sumsqr(lam_arm)
        objective = (object_cost_cs(x_plus, x_d, object_config) +
                     self.friction_reg_coef * friction_cost +
                     self.force_reg_coef * force_cost)
        opti.minimize(objective)
        
        # 摩擦锥约束
        mu = self.base_mu_arm_obj*cs.exp(scm_theta[5]) if self.contact_model == "soft-scm" else self.mu_arm_obj
        cap = force_limit/cs.sqrt(1+self._friction_force_cap_factor*mu**2)
        opti.subject_to(self._friction_residuals_cs(lam_arm, mu) <= 0)
        # Allow the optimizer to disengage when the predicted pose is already
        # close to target.  A hard 1e-3N lower bound causes unavoidable drift
        # and overshoot in the near-target regime.
        opti.subject_to(lam_arm[0] >= 1e-6)
        opti.subject_to(cs.sumsqr(lam_arm) <= force_limit ** 2)
        opti.subject_to(lam_arm[0] <= cap)
        if self.robot_stiffness is not None:
            opti.set_initial(lam_arm, [.9 * self.max_normal_force, 0., 0.])

        # Sampling backends evaluate this exact IPOPT objective/prediction
        # graph without constructing or calling an NLP solver.
        inputs = [x_d, current_x, v_last, J_tilde, D_inv, tau_o_np, n_arm,
                  t1, t2, p_arm, curr_ori_coef, env_phi, scm_theta,
                  object_config, force_limit] + ([tip_velocity] if tip_velocity is not None else [])
        names = ['x_d', 'current_x', 'v_last', 'J_tilde', 'D_inv', 'tau_o_np',
                 'n_arm', 't1', 't2', 'p_arm', 'curr_ori_coef', 'env_phi',
                 'scm_theta', 'object_config', 'force_limit'] + (['tip_velocity'] if tip_velocity is not None else [])
        self.force_objective_fn = cs.Function(
            'contact_force_objective', [lam_arm] + inputs, [x_plus, objective],
            ['lam_arm'] + names, ['x_plus_opt', 'cost'])
        if not build_solver:
            # Scalar expansion substantially reduces the per-sample overhead
            # of the small coupled contact map without changing its algebra.
            self.force_objective_fn = self.force_objective_fn.expand()
            return

        # IPOPT is the compiled fallback for the acados contact NLP.
        p_opts = {"print_time": False, "jit": False}
        s_opts = {
            "max_iter": 200,
            "tol": 1e-6,
            "linear_solver": "mumps",
            "print_level": 0,
        }
        opti.solver('ipopt', p_opts, s_opts)

        # 构建优化函数
        self.optimization_fn = opti.to_function(
            'optimization_fn_joint',
            inputs,
            [lam_arm, x_plus, objective],
            names,
            ['lam_arm_opt', 'x_plus_opt', 'cost']
        )

    def _ipopt_optimization(self, **kwargs):
        """Build the fallback only when it is actually needed."""
        if self.optimization_fn is None:
            self._precompile_optimization_function()
        return self.optimization_fn(**kwargs)

    def _select_contact_candidate(self, visible_face_idx, costs, force_buffer=None,
                                  contact_anchor_local=None, contact_anchor_idx=None,
                                  force_required=False):
        ids = np.asarray(visible_face_idx, dtype=np.int32).reshape(-1)
        costs = np.asarray(costs, dtype=np.float64).reshape(-1)
        finite_mask = np.isfinite(costs)
        # When the fingertip is already in physical contact, a zero-wrench
        # candidate is a false local optimum: it minimizes the one-step pose
        # cost by doing nothing while leaving the object unchanged.  Prefer a
        # finite candidate with a positive normal force whenever one exists.
        if force_required and force_buffer is not None:
            try:
                force_norms = np.asarray(force_buffer, dtype=np.float64)[:, 0]
                force_mask = np.isfinite(force_norms) & (force_norms > 1e-3)
                blocked_mask = np.asarray([
                    int(idx) in self._blocked_contact_indices for idx in ids], dtype=bool)
                if np.any(finite_mask & force_mask & ~blocked_mask):
                    finite_mask &= force_mask
                elif np.any(finite_mask & ~blocked_mask):
                    # The only positive-force candidate may itself be in a
                    # failed-patch cooldown.  Relax force_required for this
                    # cycle so blacklist recovery can actually switch away.
                    finite_mask &= ~blocked_mask
                elif np.any(finite_mask & force_mask):
                    finite_mask &= force_mask
            except Exception:
                pass
        if self._blocked_contact_indices:
            blocked = np.asarray([
                int(idx) in self._blocked_contact_indices for idx in ids], dtype=bool)
            # Keep the optimizer defined if every visible candidate is in
            # cooldown; otherwise suppress the failed patch for this cycle.
            if np.any(finite_mask & ~blocked):
                finite_mask &= ~blocked
        if not np.any(finite_mask):
            fallback_idx = int(ids[0])
            self.last_best_idx = fallback_idx
            self.last_global_idx = fallback_idx
            self.last_global_total_cost = float('inf')
            self.last_best_force = np.zeros(3, dtype=np.float32)
            self.last_best_x_plus = None
            self.last_best_cost = float('inf')
            self.last_selected_local = np.asarray(self.sample_point[fallback_idx], dtype=np.float32)
            self.last_selected_idx = fallback_idx
            self.last_transition_cost = 0.0
            self.last_selected_total_cost = float('inf')
            self.last_switch_required = False
            return fallback_idx, 1.0, 1.0, 0

        finite_local_indices = np.flatnonzero(finite_mask)
        finite_costs = costs[finite_mask]

        anchor_sample_idx = self._resolve_anchor_sample_idx(contact_anchor_local, contact_anchor_idx)
        prev_sample_idx = self.last_selected_idx if self.last_selected_idx is not None else anchor_sample_idx
        guided_local = self._ablation_guidance_local(ids, finite_mask)
        if guided_local is not None:
            # Random / nearest replace the cost argmin. Candidate costs stay
            # stored so the ranking strategy can still score the feasible set.
            return self._finish_guided_candidate(
                ids, costs, force_buffer, guided_local, finite_costs, prev_sample_idx)
        transition_costs = np.zeros_like(costs, dtype=np.float64)
        if anchor_sample_idx is not None:
            anchor_geo = self.sample_geodesic[anchor_sample_idx, ids]
            transition_costs += anchor_geo / max(self.contact_switch_radius, 1e-6)
            self.last_anchor_sample_idx = int(anchor_sample_idx)
        else:
            anchor_geo = np.zeros_like(costs, dtype=np.float64)
            self.last_anchor_sample_idx = None

        if prev_sample_idx is not None:
            prev_geo = self.sample_geodesic[prev_sample_idx, ids]
            transition_costs += 0.5 * prev_geo / max(self.contact_switch_radius, 1e-6)
        else:
            prev_geo = np.zeros_like(costs, dtype=np.float64)

        if anchor_sample_idx is not None:
            anchor_normal = np.asarray(self.normal[anchor_sample_idx], dtype=np.float64).reshape(3)
            candidate_normal = np.asarray(self.normal[ids], dtype=np.float64)
            normal_cost = 1.0 - np.clip(candidate_normal @ anchor_normal, -1.0, 1.0)
            # Sharp normal changes usually indicate a vertex/edge projection
            # artifact rather than a useful new patch.  Penalize them enough
            # to keep neighbouring contacts continuous during sliding.
            transition_costs += 1.0 * normal_cost

        # Select using the physical lambda objective first.  Transition cost
        # is a policy term (lazy switching), not part of the physical score:
        # mixing it into the global argmin makes the local-vs-global test
        # circular and can permanently trap the optimizer on the anchor.
        transition_weight = float(np.clip(
            getattr(self, 'contact_switch_confidence', 1.0), 0.0, 1.0))
        total_costs = costs + transition_weight * transition_costs
        total_costs[~finite_mask] = np.inf
        # Keep indices in the original candidate array.  The argmin of the
        # filtered costs is not an index into ids/force_buffer when failed or
        # zero-force candidates have been removed.
        global_local = int(finite_local_indices[int(np.argmin(finite_costs))])
        global_local = self._prefer_stable_ranking_local(
            ids, costs, finite_mask, global_local)
        global_idx = int(ids[global_local])
        self.last_global_idx = global_idx
        self.last_global_total_cost = float(costs[global_local])
        chosen_local = global_local

        # Not sitting on a trusted patch: ranking *is* the cycle-normalized
        # lambda optimum (plateau-hold already applied).  A multi-cycle
        # debounce here would keep a nearby foot as ``best_contact`` while
        # the fingertip travels, which is the local-optimum trap.
        if anchor_sample_idx is None:
            self.last_switch_required = bool(
                prev_sample_idx is not None and int(prev_sample_idx) != global_idx)
            self._pending_selected_idx = None
            self._pending_selected_count = 0
            chosen_idx = global_idx
            self.last_best_idx = chosen_idx
            if force_buffer is None:
                self.last_best_force = None
            else:
                self.last_best_force = np.asarray(force_buffer[chosen_local], dtype=np.float32)
            self.last_selected_local = np.asarray(self.sample_point[chosen_idx], dtype=np.float32)
            self.last_selected_idx = chosen_idx
            self.last_transition_cost = float(np.asarray(transition_costs[chosen_local]).reshape(()))
            self.last_selected_total_cost = float(np.asarray(total_costs[chosen_local]).reshape(()))
            return chosen_idx, float(np.min(finite_costs)), float(np.max(finite_costs)), chosen_local

        locked_incumbent_local = None
        # A hard lock only makes sense while the dwell policy still trusts
        # this patch.  Once confidence has collapsed, keep the incumbent
        # visible but let the raw lambda optimum explore.
        if (getattr(self, 'lock_contact_patch', False) and prev_sample_idx is not None
                and transition_weight > 0.05):
            hits = np.flatnonzero(ids == int(prev_sample_idx))
            if hits.size and finite_mask[int(hits[0])]:
                locked_incumbent_local = int(hits[0])

        local_mask = finite_mask.copy()
        if anchor_sample_idx is not None:
            local_mask = local_mask & (anchor_geo <= self.contact_switch_radius)
        # A zero confidence is an explicit recovery command, not merely a
        # smaller hysteresis margin.  Bypass the normal acceptance margin so
        # the raw lambda optimum is selected even when the local candidate is
        # only moderately worse (the exact failure mode this override fixes).
        if locked_incumbent_local is not None:
            chosen_local = locked_incumbent_local
            self.last_switch_required = False
        elif transition_weight <= 1e-8:
            chosen_local = global_local
            self.last_switch_required = bool(anchor_sample_idx is not None and
                                             global_idx != anchor_sample_idx)
        elif np.any(local_mask):
            local_indices = np.flatnonzero(local_mask)
            local_local = int(local_indices[np.argmin(costs[local_indices])])
            local_total = float(costs[local_local])
            global_total = float(costs[global_local])
            # Keep the incumbent contact sample while it remains valid.  The
            # previous implementation re-minimized the whole local patch at
            # every cycle, which made the fingertip chase small cost/noise
            # variations and caused unstable contact during support motions.
            # Only leave the incumbent when the raw lambda objective shows a
            # clear improvement (or the incumbent is no longer finite).
            incumbent_local = None
            if prev_sample_idx is not None:
                hits = np.flatnonzero(ids == int(prev_sample_idx))
                if hits.size and finite_mask[int(hits[0])]:
                    incumbent_local = int(hits[0])
            # Prefer the cycle span over |best|.  Normalized scores put
            # the winner at 0, so a |best|-scaled margin collapses and
            # every neighbour looks like a switch.  The span is ~1 after
            # unit-ranging; on raw costs it is the inter-sample gap.
            cost_scale = max(
                float(np.max(finite_costs) - np.min(finite_costs)),
                abs(global_total), 1e-4)
            switch_margin = float(self.contact_switch_margin_abs +
                                  transition_weight * self.contact_switch_margin_ratio * cost_scale)
            if incumbent_local is not None:
                incumbent_total = float(costs[incumbent_local])
                if incumbent_total <= global_total + switch_margin:
                    chosen_local = incumbent_local
                    self.last_switch_required = False
                elif local_total <= global_total + switch_margin:
                    chosen_local = local_local
                    self.last_switch_required = False
                else:
                    chosen_local = global_local
                    self.last_switch_required = True
            elif local_total <= global_total + switch_margin:
                chosen_local = local_local
                self.last_switch_required = False
            else:
                chosen_local = global_local
                self.last_switch_required = True
        else:
            self.last_switch_required = bool(anchor_sample_idx is not None and global_idx != prev_sample_idx)

        chosen_idx = int(ids[chosen_local])
        # Debounce patch switches.  ``_select_contact_candidate`` is called
        # once per rollout step, and the physical candidate costs can vary as
        # the object rotates.  Without this small temporal filter a tie (or a
        # transient acados/IPOPT discrepancy) makes the selected surface point
        # jump every cycle.  Keep the incumbent until the same replacement
        # candidate is selected for the configured number of consecutive
        # cycles.  If the incumbent is no longer in the visible set, commit
        # immediately because there is no valid point to hold.
        incumbent_idx = self.last_selected_idx
        if (incumbent_idx is not None and chosen_idx != int(incumbent_idx)):
            incumbent_hits = np.flatnonzero(ids == int(incumbent_idx))
            if self._pending_selected_idx == chosen_idx:
                self._pending_selected_count += 1
            else:
                self._pending_selected_idx = chosen_idx
                self._pending_selected_count = 1
            # Exhausted confidence is an explicit explore command; do not
            # sit on the dead incumbent for another confirm window.
            if (transition_weight > 0.05 and incumbent_hits.size and
                    finite_mask[int(incumbent_hits[0])] and
                    self._pending_selected_count < self.contact_switch_confirm_steps):
                chosen_idx = int(incumbent_idx)
                chosen_local = int(incumbent_hits[0])
                self.last_switch_required = False
        else:
            self._pending_selected_idx = None
            self._pending_selected_count = 0
        if chosen_idx == self._pending_selected_idx and self._pending_selected_count >= self.contact_switch_confirm_steps:
            self._pending_selected_idx = None
            self._pending_selected_count = 0
        self.last_best_idx = chosen_idx
        if force_buffer is None:
            self.last_best_force = None
        else:
            self.last_best_force = np.asarray(force_buffer[chosen_local], dtype=np.float32)
        self.last_selected_local = np.asarray(self.sample_point[chosen_idx], dtype=np.float32)
        self.last_selected_idx = chosen_idx
        self.last_transition_cost = float(np.asarray(transition_costs[chosen_local]).reshape(()))
        self.last_selected_total_cost = float(np.asarray(total_costs[chosen_local]).reshape(()))
        return chosen_idx, float(np.min(finite_costs)), float(np.max(finite_costs)), chosen_local

    def _resolve_anchor_sample_idx(self, contact_anchor_local=None, contact_anchor_idx=None):
        if contact_anchor_idx is not None:
            anchor_vertex = int(contact_anchor_idx)
            match = np.flatnonzero(self.sample_vertex_indices == anchor_vertex)
            if match.size > 0:
                return int(match[0])
        if contact_anchor_local is not None:
            anchor = np.asarray(contact_anchor_local, dtype=np.float64).reshape(3)
            dist = np.linalg.norm(np.asarray(self.sample_point, dtype=np.float64) - anchor[None, :], axis=1)
            return int(np.argmin(dist))
        if self.last_selected_local is not None:
            anchor = np.asarray(self.last_selected_local, dtype=np.float64).reshape(3)
            dist = np.linalg.norm(np.asarray(self.sample_point, dtype=np.float64) - anchor[None, :], axis=1)
            return int(np.argmin(dist))
        return None

    def _precompute_sample_geodesic(self):
        if getattr(getattr(self, 'pp', None), 'graph', None) is None:
            return np.linalg.norm(self.sample_point[:, None, :] - self.sample_point[None, :, :], axis=2)

        n_samples = self.sample_vertex_indices.shape[0]
        geo = np.full((n_samples, n_samples), np.inf, dtype=np.float64)
        for i, vertex_idx in enumerate(self.sample_vertex_indices):
            dist = dijkstra(self.pp.graph, directed=False, indices=int(vertex_idx))
            geo[i, :] = np.asarray(dist[self.sample_vertex_indices], dtype=np.float64)
        direct = np.linalg.norm(self.sample_point[:, None, :] - self.sample_point[None, :, :], axis=2)
        offset = np.linalg.norm(self.sample_point-self.pp.vertices[self.sample_vertex_indices], axis=1)
        geo += offset[:, None] + offset[None, :]
        # Separate convex pieces may overlap without sharing sampled vertices.
        # Keep a finite, conservative travel penalty in that case: infinity
        # would prevent all contact switches across decomposition boundaries.
        geo[~np.isfinite(geo)] = 4.*direct[~np.isfinite(geo)]
        geo = np.maximum(geo, direct)
        np.fill_diagonal(geo, 0.0)
        return geo

    def _solve_optimization(self, **kwargs):
        if getattr(self, 'shared_response', None) is not None:
            kwargs.setdefault('tip_velocity', self.rank_tip_velocity)
        kwargs.setdefault('object_config', self.object_cost_config())
        kwargs.setdefault('force_limit', self.max_contact_force)
        kwargs.setdefault('env_phi', self.env_phi)
        kwargs.setdefault('scm_theta', getattr(self, 'scm_theta', np.zeros(6)))
        if self.solver in ('cma-es', 'cem'):
            try:
                result = self.population_search.solve(self, kwargs)
                self.last_solver_status = self.solver
                return result
            except (RuntimeError, ValueError, FloatingPointError):
                self.last_solver_status = self.solver + '-failed'
                return None
        if self.acados_solver is not None and self.solver not in ('ipopt', 'torch-lbfgs', 'torch-gn'):
            self.acados_solve_count += 1
            try:
                result = self._solve_optimization_acados(**kwargs)
                self.last_solver_status = 'acados'
                return result
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                self.acados_failure_count += 1
                self.last_acados_failure_reason = str(exc)
                self.acados_failure_reasons[str(exc)] = self.acados_failure_reasons.get(str(exc), 0) + 1
            if self._acados_fallbacks_this_cycle >= self.acados_max_fallbacks_per_cycle:
                self.last_solver_status = 'acados-failed-skip'
                return None
            try:
                self._acados_fallbacks_this_cycle += 1
                self.acados_fallback_count += 1
                result = self._ipopt_optimization(**kwargs)
                self.last_solver_status = 'ipopt-fallback'
                return result
            except RuntimeError:
                self.last_solver_status = 'failed'
                return None
        if self.solver in ('torch-lbfgs', 'torch-gn'):
            try:
                result = self._solve_optimization_torch(**kwargs)
                self.last_solver_status = 'torch-lbfgs'
                return result
            except (RuntimeError, ValueError, FloatingPointError):
                # Keep the original IPOPT problem as a per-contact fallback.
                # This is important during rollout when a degenerate Jacobian
                # can make the small Torch problem temporarily non-finite.
                try:
                    result = self._ipopt_optimization(**kwargs)
                    self.last_solver_status = 'ipopt-fallback'
                    return result
                except RuntimeError:
                    self.last_solver_status = 'failed'
                    return None
        try:
            result = self._ipopt_optimization(**kwargs)
            self.last_solver_status = 'ipopt'
            return result
        except RuntimeError:
            self.last_solver_status = 'failed'
            return None

    def _build_acados_contact_solver(self):
        from planning.acados_env import ensure_acados_env
        ensure_acados_env()
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, ACADOS_INFTY
        # Fast path: N=1 SQP_RTI with lambda as the control.  One QP per
        # candidate. Each patch gets its own primal/dual initial iterate.
        cs_p = cs.SX.sym('xd', 7); cs_v = cs.SX.sym('vlast', 6)
        cs_tip = cs.SX.sym('tip_velocity', 3) if getattr(self, 'shared_response', None) is not None else None
        cs_j = cs.SX.sym('Jenv', 4 * self.max_contacts, 6); cs_d = cs.SX.sym('Dinv', 4 * self.max_contacts, 4)
        cs_tau = cs.SX.sym('tau', 6); cs_n = cs.SX.sym('n', 3); cs_t1 = cs.SX.sym('t1', 3); cs_t2 = cs.SX.sym('t2', 3); cs_cp = cs.SX.sym('p', 3)
        cs_phi = cs.SX.sym('phi', 4*self.max_contacts)
        scm_theta = cs.SX.sym('scm_theta', 6)
        object_config = cs.SX.sym('object_config', OBJECT_COST_CONFIG_SIZE)
        force_limit = cs.SX.sym('force_limit')
        prm = cs.vertcat(cs_phi, cs_p, cs_v, cs.reshape(cs_j, -1, 1), cs.reshape(cs_d, -1, 1), cs_tau, cs_n, cs_t1, cs_t2, cs_cp, scm_theta, object_config,
                         *([cs_tip] if cs_tip is not None else []), force_limit)
        x = cs.SX.sym('x', 7); u = cs.SX.sym('u', 3)
        Jc = cs.SX.zeros(3, 6); Jc[:3, :3] = cs.SX.eye(3); Jc[0,4],Jc[0,5]=cs_cp[2],-cs_cp[1]; Jc[1,3],Jc[1,5]=-cs_cp[2],cs_cp[0]; Jc[2,3],Jc[2,4]=cs_cp[1],-cs_cp[0]
        wrench_scale = self.wrench_velocity_scale
        Rct = cs.horzcat(cs_n, cs_t1, cs_t2); b = self.gravity_velocity_scale*cs_tau + wrench_scale*cs.transpose(Jc) @ (Rct @ u); qinv = scaled_inverse_cs(self.base_Q_inv, scm_theta) if self.contact_model == "soft-scm" else cs.DM(self.Q_inv); qib=qinv@b
        vn = (self._contact_velocity_cs(cs_v+qib, cs_j, cs_d, cs_phi, qinv, scm_theta)
              if self.contact_model == "soft-scm" else self._contact_velocity_cs(cs_v+qib, cs_j, cs_d, cs_phi))
        if self.robot_stiffness is not None:
            vn = self._arm_velocity_cs(cs_v, cs_tau, Rct@u, cs_cp, Rct, cs_j, cs_phi, qinv, scm_theta, tip_velocity=cs_tip)
        quat=x[3:7]; H=cs.vertcat(cs.horzcat(-quat[1],quat[0],quat[3],-quat[2]),cs.horzcat(-quat[2],-quat[3],quat[0],quat[1]),cs.horzcat(-quat[3],quat[2],-quat[1],quat[0])).T
        qn=cs.vertcat(x[:3]+self.h*vn[:3], quat+0.5*self.h*H@vn[3:6]); qn=cs.vertcat(qn[:3],qn[3:7]/cs.norm_2(qn[3:7]))
        self._acados_candidate_fn = cs.Function('contact_candidate', [x, u, prm],
            [qn, object_cost_cs(qn, cs_p, object_config)
             + self.friction_reg_coef * cs.sumsqr(u[1:3]) + self.force_reg_coef * cs.sumsqr(u)])
        # Terminal cost is on the terminal state ``x`` (qn after the
        # discrete transition).  Putting qn itself here is rejected because
        # a terminal cost may not depend on ``u``.
        mass_key = hashlib.sha1(np.r_[np.asarray(self.Q_inv).reshape(-1), self.h, self.K_contact, self.comfree_damping, self.soft_scm_response_time, self.soft_scm_damping_ratio, self.soft_scm_regularization, float(self.wrench_is_force), self.gravity_velocity_scale, self.wrench_velocity_scale, float(self.grouped_contact_response), self.robot_stiffness or 0., self.pos_coef, self.ori_coef, self.friction_reg_coef, self.force_reg_coef].astype('<f8').tobytes() + cs.__version__.encode()).hexdigest()[:10]
        if getattr(self, 'shared_response', None) is not None:
            import json
            mass_key = hashlib.sha1((mass_key + json.dumps(self.shared_response, sort_keys=True)).encode()).hexdigest()[:10]
        if self.friction_cone == 'pyramid':
            mass_key = hashlib.sha1((mass_key + '_l1_pyramid').encode()).hexdigest()[:10]
        mu_key = int(round(float(self.mu_arm_obj) * 1000.0))
        contact_tag = self.contact_model.replace('-', '_')
        model=AcadosModel(); model.name=f'contact_lambda_{contact_tag}_v25_m{self.max_contacts}_f{int(self.max_contact_force*1000)}_u{mu_key}_q{mass_key}'; model.x=x; model.u=u; model.p=prm; model.disc_dyn_expr=qn
        model.cost_expr_ext_cost = self.friction_reg_coef * cs.sumsqr(u[1:3]) + self.force_reg_coef * cs.sumsqr(u)
        model.cost_expr_ext_cost_e = object_cost_cs(x, cs_p, object_config)
        ocp=AcadosOcp(); ocp.model=model; ocp.parameter_values=np.zeros(int(prm.size1())); ocp.parameter_values[-1]=self.max_contact_force; ocp.cost.cost_type='EXTERNAL'; ocp.cost.cost_type_e='EXTERNAL';
        mu = self.base_mu_arm_obj*cs.exp(scm_theta[5]) if self.contact_model == 'soft-scm' else float(self.mu_arm_obj)
        cap = force_limit/cs.sqrt(1+self._friction_force_cap_factor*mu**2)
        # The conservative normal cap plus friction pyramid already implies
        # ||u|| <= force_limit. Its redundant nonlinear row has a zero gradient
        # at disengagement and makes the QP unnecessarily degenerate.
        model.con_h_expr = cs.vertcat(u[0], u[0]-cap, self._friction_residuals_cs(u, mu))
        ocp.constraints.lh = np.array([0.0] + [-ACADOS_INFTY]*5)
        ocp.constraints.uh = np.array([ACADOS_INFTY] + [0.]*5)
        ocp.constraints.idxbx_0=np.arange(7); ocp.constraints.lbx_0=np.zeros(7); ocp.constraints.ubx_0=np.zeros(7)
        if self.robot_stiffness is not None:
            ocp.solver_options.cost_scaling = np.ones(2)
        ocp.solver_options.N_horizon=1; ocp.solver_options.tf=float(self.h); ocp.solver_options.qp_solver='PARTIAL_CONDENSING_HPIPM'; ocp.solver_options.hessian_approx='EXACT'; ocp.solver_options.integrator_type='DISCRETE'; ocp.solver_options.nlp_solver_type='SQP_RTI'; ocp.solver_options.regularize_method='PROJECT'; ocp.solver_options.print_level=0
        d='/tmp/'+model.name+'_codegen'; os.makedirs(d,exist_ok=True); _set_acados_code_export_directory(ocp, d); jf=os.path.join(d,model.name+'.json'); so=os.path.join(d,'libacados_ocp_solver_'+model.name+'.so')
        from planning.acados_env import build_cached_solver
        return build_cached_solver(ocp, jf, so)

    def _solve_optimization_acados(self, **kwargs):
        solver=self.acados_solver; xcur=np.asarray(kwargs['current_x'],float).reshape(7)
        p=np.concatenate([np.asarray(kwargs['env_phi']).reshape(-1), np.asarray(kwargs['x_d']).reshape(-1),
                          np.asarray(kwargs['v_last']).reshape(-1),
                          np.asarray(kwargs['J_tilde']).reshape(-1,order='F'),
                          np.asarray(kwargs['D_inv']).reshape(-1,order='F'),
                          np.asarray(kwargs['tau_o_np']).reshape(-1),
                          np.asarray(kwargs['n_arm']).reshape(-1),
                          np.asarray(kwargs['t1']).reshape(-1),
                          np.asarray(kwargs['t2']).reshape(-1),
                          np.asarray(kwargs['p_arm']).reshape(-1), np.asarray(kwargs.get('scm_theta', self.scm_theta)).reshape(-1),
                          extended_cost_config(kwargs.get('object_config', self.object_cost_config())),
                          *([np.asarray(kwargs.get('tip_velocity', self.rank_tip_velocity)).reshape(3)] if getattr(self, 'shared_response', None) is not None else []),
                          [kwargs.get('force_limit', self.max_contact_force)]])
        solver.reset()
        solver.set(0,'x',xcur); solver.set(0,'lbx',xcur); solver.set(0,'ubx',xcur)
        initial_normal = .01
        if self.robot_stiffness is not None:
            # A near-zero command can remain on the disengaged branch of the
            # coupled unilateral contact map. RTI then sees no pressure gain.
            # Seed inside the feasible force cone so it can optimize a press.
            initial_normal = .9 * float(kwargs.get('force_limit', self.max_contact_force)) / np.sqrt(1 + self._friction_force_cap_factor*self.mu_arm_obj**2)
        solver.set(0,'u',np.array([initial_normal,0,0])); solver.set(0,'p',p)
        seed_pose = np.asarray(self._acados_candidate_fn(xcur, np.array([initial_normal,0,0]), p)[0]).reshape(7)
        solver.set(1,'x',seed_pose); solver.set(1,'p',p)
        from planning.acados_env import quiet_acados_stderr
        with quiet_acados_stderr():
            status=int(solver.solve())
        if status != 0:
            self.acados_qp_failure_count += 1
            raise RuntimeError(f'acados status {status}')
        lam=np.asarray(solver.get(0,'u')).reshape(3)
        if not np.isfinite(lam).all():
            raise FloatingPointError('acados returned non-finite contact wrench')
        lam = self._project_contact_force(lam, kwargs.get('force_limit', self.max_contact_force))
        # Report the prediction of the feasible RTI control. Do not replace
        # it with a seed/zero-pressure action based on a second one-step cost
        # comparison: that override also changes the contact ranking.
        qnext, cost = self._acados_candidate_fn(xcur, lam, p)
        qnext, cost = np.asarray(qnext).reshape(7), float(cost)
        if not np.isfinite(qnext).all() or not np.isfinite(cost):
            raise FloatingPointError('non-finite contact prediction')
        return {'lam_arm_opt':lam,'x_plus_opt':qnext,'cost':cost}

    def _solve_optimization_torch(self, x_d, current_x, v_last, J_tilde, D_inv,
                                  tau_o_np, n_arm, t1, t2, p_arm, curr_ori_coef, env_phi=None, scm_theta=None, object_config=None, force_limit=None):
        """Solve the single-contact problem with differentiable Torch L-BFGS.

        This mirrors ``_precompile_optimization_function``: the optimized
        variable is the 3D contact wrench ``[fn, ft1, ft2]`` and the same
        projected environment response and quaternion integration are used.
        Friction and normal-force bounds are enforced by a smooth
        parameterization, avoiding the unconstrained/oversized fingertip
        forces that otherwise make the rollout jump.
        """
        if torch is None:
            raise RuntimeError('Torch is not installed')
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dt = torch.float32
        t = lambda x: torch.as_tensor(np.asarray(x), device=dev, dtype=dt)
        xd, xc, vl = t(x_d).reshape(-1), t(current_x).reshape(-1), t(v_last).reshape(-1)
        Jenv, D = t(J_tilde), t(D_inv)
        phi = t(self.env_phi if env_phi is None else env_phi)
        tau, n, tt1, tt2, p = map(t, (tau_o_np, n_arm, t1, t2, p_arm))
        qinv = t(self.Q_inv)

        # Contact Jacobian [I, -skew(p)] in the convention used above.
        Jc = torch.zeros((3, 6), device=dev, dtype=dt)
        Jc[:, :3] = torch.eye(3, device=dev, dtype=dt)
        Jc[0, 4], Jc[0, 5] = p[2], -p[1]
        Jc[1, 3], Jc[1, 5] = -p[2], p[0]
        Jc[2, 3], Jc[2, 4] = p[1], -p[0]
        Rcontact = torch.stack((n, tt1, tt2), dim=1)

        normal_cap = (self.max_contact_force if force_limit is None else float(force_limit)) / np.sqrt(1 + self._friction_force_cap_factor*self.mu_arm_obj**2)
        # Match the active-contact initial guess used by the native solvers.
        initial_normal = .9 * normal_cap if self.robot_stiffness is not None else .01
        fn0 = (initial_normal - 1e-6) / (normal_cap - 1e-6)
        fn0 = float(np.clip(fn0, 1e-4, 1.0 - 1e-4))
        z0 = torch.zeros(3, device=dev, dtype=dt)
        z0[0] = torch.log(torch.tensor(fn0 / (1.0 - fn0), device=dev, dtype=dt))
        z = torch.nn.Parameter(z0)
        opt = torch.optim.LBFGS(
            [z], max_iter=self.torch_max_iter, tolerance_grad=1e-5,
            tolerance_change=1e-9, line_search_fn='strong_wolfe')

        def evaluate():
            fn = 1e-6 + (normal_cap - 1e-6) * torch.sigmoid(z[0])
            lam = torch.cat((fn.reshape(1), self.mu_arm_obj * fn * self._bounded_tangents_torch(z[1:])))
            wrench_scale = self.wrench_velocity_scale
            b = self.gravity_velocity_scale * tau + wrench_scale * (Jc.T @ (Rcontact @ lam))
            qinv_b = qinv @ b
            vnow = self._contact_velocity_torch(vl+qinv_b, Jenv, D, phi, qinv)
            if self.robot_stiffness is not None:
                vnow = self._arm_velocity_torch(vl, tau, Rcontact@lam, p, Rcontact, Jenv, phi, qinv)
            quat = xc[3:7]
            Ht = torch.stack((
                torch.stack((-quat[1], quat[0], quat[3], -quat[2])),
                torch.stack((-quat[2], -quat[3], quat[0], quat[1])),
                torch.stack((-quat[3], quat[2], -quat[1], quat[0])),
            ), dim=0).T
            qnext = torch.cat((xc[:3] + self.h * vnow[:3],
                               quat + 0.5 * self.h * (Ht @ vnow[3:6])))
            qnext = torch.cat((qnext[:3], qnext[3:7] / torch.linalg.vector_norm(qnext[3:7]).clamp_min(1e-8)))
            friction_cost = torch.sum(lam[1:] * lam[1:])
            force_cost = torch.sum(lam * lam)
            cost = (object_cost_torch(qnext[:3], qnext[3:7], xd, self.object_cost_config() if object_config is None else object_config) + self.friction_reg_coef * friction_cost +
                    self.force_reg_coef * force_cost)
            return cost, lam, qnext

        def closure():
            opt.zero_grad()
            cost, _, _ = evaluate()
            if not torch.isfinite(cost):
                raise FloatingPointError('non-finite Torch contact objective')
            cost.backward()
            return cost

        opt.step(closure)
        with torch.no_grad():
            cost, lam, qnext = evaluate()
        if not (torch.isfinite(cost) and torch.isfinite(lam).all() and torch.isfinite(qnext).all()):
            raise FloatingPointError('non-finite Torch contact solution')
        return {'lam_arm_opt': lam.detach().cpu().numpy(),
                'x_plus_opt': qnext.detach().cpu().numpy(),
                'cost': float(cost.detach().cpu())}

    def _choose_contact_points_torch_batch(self, x_d, current_x, tau_o, visible_face_idx,
                                           v_last, D_inv):
        """Evaluate all visible contact candidates in one Torch L-BFGS solve.

        The old implementation launched one optimizer (and repeatedly copied
        six small arrays to Torch) per face.  A batched objective has identical
        independent rows, but lets BLAS/GPU evaluate all candidates together.
        """
        if torch is None:
            raise RuntimeError('Torch is not installed')
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dt = torch.float32
        t = lambda x: torch.as_tensor(np.asarray(x), device=dev, dtype=dt)
        xd, xc, vl = t(x_d).reshape(-1), t(current_x).reshape(-1), t(v_last).reshape(-1)
        Jenv, D = t(self.J_tilde), t(D_inv)
        phi = t(self.env_phi)
        ids = np.asarray(visible_face_idx, dtype=np.int32).reshape(-1)
        n = t(self.normal[ids]); tt1 = t(self.t1[ids]); tt2 = t(self.t2[ids]); p = t(self.sample_point[ids])
        tau, qinv = t(tau_o).reshape(-1), t(self.Q_inv)
        count = ids.size

        Jc = torch.zeros((count, 3, 6), device=dev, dtype=dt)
        Jc[:, :, :3] = torch.eye(3, device=dev, dtype=dt).expand(count, -1, -1)
        Jc[:, 0, 4], Jc[:, 0, 5] = p[:, 2], -p[:, 1]
        Jc[:, 1, 3], Jc[:, 1, 5] = -p[:, 2], p[:, 0]
        Jc[:, 2, 3], Jc[:, 2, 4] = p[:, 1], -p[:, 0]
        Rcontact = torch.stack((n, tt1, tt2), dim=2)

        initial_normal = .9 * self.max_normal_force if self.robot_stiffness is not None else .01
        fn0 = (initial_normal - 1e-6) / (self.max_normal_force - 1e-6)
        fn0 = float(np.clip(fn0, 1e-4, 1.0 - 1e-4))
        z = torch.nn.Parameter(torch.zeros((count, 3), device=dev, dtype=dt))
        z.data[:, 0] = torch.log(torch.tensor(fn0 / (1.0 - fn0), device=dev, dtype=dt))
        opt = torch.optim.LBFGS([z], max_iter=self.torch_max_iter,
                                tolerance_grad=1e-5, tolerance_change=1e-9,
                                line_search_fn='strong_wolfe')

        def evaluate():
            fn = 1e-6 + (self.max_normal_force - 1e-6) * torch.sigmoid(z[:, 0])
            lam = torch.cat((fn[:, None], self.mu_arm_obj * fn[:, None] * self._bounded_tangents_torch(z[:, 1:])), dim=1)
            wrench = torch.bmm(Rcontact, lam[:, :, None]).squeeze(-1)
            wrench_scale = self.wrench_velocity_scale
            b = self.gravity_velocity_scale * tau[None, :] + wrench_scale * torch.bmm(Jc.transpose(1, 2), wrench[:, :, None]).squeeze(-1)
            qinv_b = torch.matmul(b, qinv.T)
            vnow = self._contact_velocity_torch(qinv_b+vl[None, :], Jenv, D, phi, qinv)
            if self.robot_stiffness is not None:
                vnow = torch.stack([self._arm_velocity_torch(vl, tau, wrench[i], p[i], Rcontact[i], Jenv, phi, qinv) for i in range(count)])
            quat = xc[3:7]
            Ht = torch.stack((
                torch.stack((-quat[1], quat[0], quat[3], -quat[2])),
                torch.stack((-quat[2], -quat[3], quat[0], quat[1])),
                torch.stack((-quat[3], quat[2], -quat[1], quat[0])),
            ), dim=0).T
            qpos = xc[:3][None, :] + self.h * vnow[:, :3]
            qrot = quat[None, :] + 0.5 * self.h * torch.matmul(vnow[:, 3:6], Ht.T)
            qrot = qrot / torch.linalg.vector_norm(qrot, dim=1, keepdim=True).clamp_min(1e-8)
            friction_cost = torch.sum(lam[:, 1:] * lam[:, 1:], dim=1)
            force_cost = torch.sum(lam * lam, dim=1)
            costs = (object_cost_torch(qpos, qrot, xd, self.object_cost_config()) + self.friction_reg_coef * friction_cost +
                     self.force_reg_coef * force_cost)
            return costs, lam, torch.cat((qpos, qrot), dim=1)

        def closure():
            opt.zero_grad()
            costs, _, _ = evaluate()
            if not torch.isfinite(costs).all():
                raise FloatingPointError('non-finite batched Torch contact objective')
            costs.sum().backward()
            return costs.sum()

        opt.step(closure)
        with torch.no_grad():
            costs, lam, qnext = evaluate()
        if not (torch.isfinite(costs).all() and torch.isfinite(lam).all() and torch.isfinite(qnext).all()):
            raise FloatingPointError('non-finite batched Torch contact solution')
        self.last_solver_status = 'torch-lbfgs'
        return ids, costs.cpu().numpy(), lam.cpu().numpy(), qnext.cpu().numpy()

    def init_utils(self):
        # -------------------------------
        #    quaternion integration fn
        # -------------------------------
        quat = cs.SX.sym('quat', 4)
        H_q_body = cs.vertcat(cs.horzcat(-quat[1], quat[0], quat[3], -quat[2]),
                              cs.horzcat(-quat[2], -quat[3], quat[0], quat[1]),
                              cs.horzcat(-quat[3], quat[2], -quat[1], quat[0]))
        self.cs_qmat_body_fn_ = cs.Function('cs_qmat_body_fn', [quat], [H_q_body.T])

        # -------------------------------
        #    state integration fn
        # -------------------------------
        qvel = cs.SX.sym('qvel', 6)
        qpos = cs.SX.sym('qpos', 7)
        next_obj_pos = qpos[0:3] + self.h * qvel[0:3]
        next_obj_quat = (qpos[3:7] + 0.5 * self.h * self.cs_qmat_body_fn_(qpos[3:7]) @ qvel[3:6])
        next_obj_quat = next_obj_quat / cs.norm_2(next_obj_quat)
        next_qpos = cs.vertcat(next_obj_pos, next_obj_quat)
        self.cs_qposInteg_ = cs.Function('cs_qposInte', [qpos, qvel], [next_qpos])

    @staticmethod
    def compute_contact_jacobian(p):
        """优化后的接触雅可比计算 - MX 版本"""
        J_c = cs.MX.zeros(3, 6)  # 改为 MX 类型
        J_c[:3, :3] = cs.MX.eye(3)
        # 使用 CasADi 构建斜对称矩阵
        J_c[0, 4], J_c[0, 5] = p[2], -p[1]
        J_c[1, 3], J_c[1, 5] = -p[2], p[0]
        J_c[2, 3], J_c[2, 4] = p[1], -p[0]
        return J_c
    
    def optimize_control_input(self, x_d, current_x, tau_o, p_arm=None, v_last=None,
                               sample_idx=None, candidate_idx=None,
                               contact_point_local=None, contact_normal_inward=None):
        """优化控制输入 - 更新接口"""
        # This is a separate solve from contact-point ranking and gets its own
        # bounded fallback budget.
        self._acados_fallbacks_this_cycle = 0
        if p_arm is None:
            p_arm = np.array([-1, 0, 0])
        if v_last is None:
            v_last = self.rank_object_velocity.copy() if getattr(self, 'shared_response', None) is not None else np.zeros(6)

        ori_align_sq = cs.dot(current_x[3:7], x_d[3:7]) ** 2
        th = 0.85
        scale = 10
        curr_ori_coef = (1.0 + cs.tanh(scale * (ori_align_sq - th)))

        if sample_idx is None and candidate_idx is not None:
            sample_idx = self.select_executed_contact_idx(p_arm, candidate_idx)
        if sample_idx is not None:
            idx = int(sample_idx)
            p_obj_local = np.asarray(self.sample_point[idx], dtype=np.float64)
            n = np.asarray(self.normal[idx], dtype=np.float64)
            t1 = np.asarray(self.t1[idx], dtype=np.float64)
            t2 = np.asarray(self.t2[idx], dtype=np.float64)
            normal_obj_local = -n
            self.last_executed_idx = idx
        else:
            p_obj_local, n, t1, t2, _ = self.pp.project_surface(p_arm)
            normal_obj_local = -np.asarray(n)
        if (contact_point_local is None) != (contact_normal_inward is None):
            raise ValueError('observed contact needs both point and inward normal')
        if contact_point_local is not None:
            p_obj_local = np.asarray(contact_point_local, dtype=float).reshape(3)
            n = np.asarray(contact_normal_inward, dtype=float).reshape(3)
            if not np.isfinite(np.r_[p_obj_local, n]).all() or np.linalg.norm(n) < 1e-10:
                raise ValueError('invalid observed contact geometry')
            n = n / np.linalg.norm(n)
            ref = np.array([0.,0.,1.]) if abs(n[2]) < .9 else np.array([0.,1.,0.])
            t1 = np.cross(n, ref); t1 /= np.linalg.norm(t1)
            t2 = np.cross(n, t1)
            normal_obj_local = -n
        D_inv = self.compute_env_diag_inverse(self.J_tilde)

        start_time = time.time()
        sol = self._solve_optimization(
            x_d=x_d, 
            current_x=current_x, 
            v_last=v_last,
            J_tilde=self.J_tilde, 
            D_inv=D_inv,
            tau_o_np=tau_o,
            n_arm=n,
            t1=t1,
            t2=t2,
            p_arm=p_obj_local,
            curr_ori_coef=curr_ori_coef
        )
        if sol is None:
            lam_arm = np.zeros(3, dtype=np.float32)
            x_plus_opt = np.asarray(current_x, dtype=np.float32).copy()
            cost = float("inf")
        else:
            lam_arm = sol['lam_arm_opt']
            x_plus_opt = sol['x_plus_opt']
            cost = float(sol['cost'])

        info = {
            "solve_time": time.time() - start_time,
            "control_input": lam_arm,
            "resulting_pose": x_plus_opt,
            "solver_failed": sol is None,
        }
        if sample_idx is not None:
            self.last_executed_x_plus = None if sol is None else np.asarray(x_plus_opt, dtype=np.float64)
            self.last_executed_cost = cost
            self.last_executed_force = np.asarray(lam_arm, dtype=np.float32)
        
        # Preserve the repository's extended return API while using the
        # reference point-selection rule; callers that do not need the normal
        # can simply ignore the second value.
        return p_obj_local, normal_obj_local, x_plus_opt, cost, info

    def choose_contact_points(self, x_d, current_x, tau_o, visible_face_idx, v_last=None,
                              contact_anchor_local=None, contact_anchor_idx=None,
                              force_required=False, query_local=None):
        # Reset the IPOPT fallback budget for this contact-selection cycle.
        if query_local is not None:
            self.rank_query_local = np.asarray(query_local, dtype=np.float64).reshape(3)
        self._acados_fallbacks_this_cycle = 0
        if self._blocked_contact_indices:
            expired = []
            for key, remaining in self._blocked_contact_indices.items():
                remaining = int(remaining) - 1
                if remaining <= 0:
                    expired.append(key)
                else:
                    self._blocked_contact_indices[key] = remaining
            for key in expired:
                self._blocked_contact_indices.pop(key, None)
        if not len(visible_face_idx):
            # 确保返回不是None
            return self.sample_point[0], self.normal[0], 1, 1, 1
        if v_last is None:
            v_last = self.rank_object_velocity.copy() if getattr(self, 'shared_response', None) is not None else np.zeros(6)

        if self.solver in ('torch-lbfgs', 'torch-gn'):
            D_inv = self.compute_env_diag_inverse(self.J_tilde)
            try:
                ids, costs, force_buffer, x_plus_buffer = self._choose_contact_points_torch_batch(
                    x_d, current_x, tau_o, visible_face_idx, v_last, D_inv)
                raw_costs = np.asarray(costs, dtype=np.float64).reshape(-1)
                costs = self._rescore_candidate_costs(
                    ids, raw_costs, x_plus_buffer, x_d, current_x)
                self._store_candidate_buffers(
                    ids, costs, x_plus_buffer, force_buffer, raw_costs=raw_costs)
                selected_idx, min_error, max_error, selected_local = self._select_contact_candidate(
                    ids, costs, force_buffer=force_buffer,
                    contact_anchor_local=contact_anchor_local,
                    contact_anchor_idx=contact_anchor_idx,
                    force_required=force_required)
                self.last_best_idx = int(selected_idx)
                self.last_best_x_plus = np.asarray(x_plus_buffer[int(selected_local)], dtype=np.float64)
                self.last_best_cost = float(costs[int(selected_local)])
                self._record_global_delta()
                return self.sample_point[selected_idx], self.normal[selected_idx], min_error, max_error, 1
            except (RuntimeError, ValueError, FloatingPointError):
                # Fall through to the original per-candidate path.  Each row
                # then has its own IPOPT fallback through _solve_optimization.
                pass

        curr_ori_coef = 1
        n_vis = int(np.asarray(visible_face_idx).reshape(-1).shape[0])
        error_values = np.full(n_vis, np.inf, dtype=np.float64)
        x_plus_buffer = []
        force_buffer = []
        D_inv = self.compute_env_diag_inverse(self.J_tilde)
        for i, idx in enumerate(visible_face_idx):
            sol = self._solve_optimization(
                    x_d=x_d, 
                    current_x=current_x, 
                    v_last=v_last,
                    J_tilde=self.J_tilde, 
                    D_inv=D_inv,
                    tau_o_np=tau_o,
                    n_arm=self.normal[idx],
                    t1=self.t1[idx],
                    t2=self.t2[idx],
                    p_arm=self.sample_point[idx],
                    curr_ori_coef=curr_ori_coef
                )
            if sol is None:
                x_plus_buffer.append(np.asarray(current_x, dtype=np.float64).copy())
                force_buffer.append(np.zeros(3, dtype=np.float32))
                continue

            error_values[i] = float(sol['cost'])
            x_plus_buffer.append(np.asarray(sol['x_plus_opt'], dtype=np.float64).reshape(7))
            force_buffer.append(sol['lam_arm_opt'])
        raw_costs = np.asarray(error_values, dtype=np.float64).reshape(-1)
        error_values = self._rescore_candidate_costs(
            visible_face_idx, raw_costs, x_plus_buffer, x_d, current_x)
        self._store_candidate_buffers(
            visible_face_idx, error_values, x_plus_buffer, force_buffer,
            raw_costs=raw_costs)
        finite_mask = np.isfinite(error_values)
        if not np.any(finite_mask):
            fallback_idx = int(visible_face_idx[0])
            self.last_best_x_plus = None
            self.last_best_cost = float('inf')
            self.last_best_delta = None
            return self.sample_point[fallback_idx], self.normal[fallback_idx], 1, 1, 1

        # Skipped/failed acados candidates are represented by ``inf`` in
        # error_list.  Do not let those sentinels poison the finite cost span:
        # an infinite max makes the rollout accept every near-contact state
        # and disables the contact-quality hysteresis.
        finite_errors = error_values[finite_mask]
        min_error = float(np.min(finite_errors))
        max_error = float(np.max(finite_errors))
        min_idx, _, _, chosen_local = self._select_contact_candidate(
            visible_face_idx,
            error_values,
            force_buffer=force_buffer,
            contact_anchor_local=contact_anchor_local,
            contact_anchor_idx=contact_anchor_idx,
            force_required=force_required,
        )
        self.last_best_idx = int(min_idx)
        self.last_best_force = np.asarray(force_buffer[int(chosen_local)], dtype=np.float32)
        self.last_best_x_plus = np.asarray(x_plus_buffer[int(chosen_local)], dtype=np.float64).reshape(7)
        self.last_best_cost = float(error_values[int(chosen_local)])
        self._record_global_delta()
        return self.sample_point[min_idx], self.normal[min_idx], min_error, max_error, 1

    @staticmethod
    def _span_limits(values):
        """Tukey fence of finite values.  Returns (lo, hi) or (None, None)."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None, None
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if finite.size >= 4:
            q25, q75 = np.percentile(finite, [25.0, 75.0])
            iqr = float(q75 - q25)
            if iqr > 1e-12:
                hi = min(hi, float(q75) + 1.5 * iqr)
            else:
                ordered = np.sort(finite)
                if ordered[-1] > ordered[-2] + 1e-12:
                    hi = float(ordered[-2])
        return lo, max(hi, lo)

    @staticmethod
    def _unit_range_costs(values, hi_percentile=90.0):
        """Map this cycle's finite costs onto [0, 1].

        0 is the cheapest finite value.  The high end is a robust
        Tukey fence (or the raw max on tiny sets) so one unreachable
        sample cannot stretch the verify gate.  The map is monotone
        below that fence, so it does not by itself change the argmin.
        """
        _ = hi_percentile
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        out = np.full(values.shape, np.inf, dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            return out
        vals = values[finite]
        lo, hi = LambdaContactControlOptimizer._span_limits(vals)
        if lo is None:
            return out
        span = hi - lo
        if span <= 1e-12:
            out[finite] = np.where(vals <= lo + 1e-12, 0.0, 1.0)
            return out
        out[finite] = np.clip((vals - lo) / span, 0.0, 1.0)
        return out

    def _pose_cost(self, x, x_d):
        """Active object objective used for candidate reductions and ranking."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        xd = np.asarray(x_d, dtype=np.float64).reshape(-1)
        if x.size < 7 or xd.size < 7 or not np.isfinite(x[:7]).all():
            return float('inf')
        return object_cost_np(x, xd, self.object_cost_config())

    def _pose_cost_deltas(self, current_x, x_plus_buffer, x_d, raw_costs):
        """Cost reduction of applying each ``x_plus_opt`` to the current pose."""
        raw = np.asarray(raw_costs, dtype=np.float64).reshape(-1)
        n = int(raw.size)
        deltas = np.full(n, np.nan, dtype=np.float64)
        c_now = self._pose_cost(current_x, x_d)
        self.last_pose_cost_now = float(c_now) if np.isfinite(c_now) else 0.0
        if not np.isfinite(c_now) or x_plus_buffer is None:
            return deltas
        from planning.object_pose_cost import object_cost_batch_np
        count = min(n, len(x_plus_buffer))
        if count:
            costs = object_cost_batch_np(np.asarray(x_plus_buffer[:count]).reshape(count, 7),
                                        x_d, self.object_cost_config())
            valid = np.isfinite(raw[:count]) & np.isfinite(costs)
            deltas[:count] = np.where(valid, c_now-costs, np.nan)
        return deltas

    def _record_best_delta(self, local_idx):
        deltas = getattr(self, 'last_candidate_deltas', None)
        if deltas is None:
            self.last_best_delta = None
            return
        try:
            value = float(np.asarray(deltas, dtype=np.float64).reshape(-1)[int(local_idx)])
        except (TypeError, ValueError, IndexError):
            self.last_best_delta = None
            return
        self.last_best_delta = value if np.isfinite(value) else None

    def _record_global_delta(self):
        """Keep last_best_delta on the ranked yellow sample, not the hold."""
        gidx = getattr(self, 'last_global_idx', None)
        ids = getattr(self, 'last_candidate_ids', None)
        if gidx is None or ids is None:
            return
        hits = np.flatnonzero(np.asarray(ids, dtype=np.int32).reshape(-1) == int(gidx))
        if hits.size:
            self._record_best_delta(int(hits[0]))

    def pose_delta_for_sample(self, sample_idx):
        """Predicted pose-cost reduction ``C(now)-C(x_plus)`` of one sample."""
        if sample_idx is None:
            return None
        ids = getattr(self, 'last_candidate_ids', None)
        deltas = getattr(self, 'last_candidate_deltas', None)
        if ids is None or deltas is None:
            gidx = getattr(self, 'last_global_idx', None)
            if (gidx is not None and int(sample_idx) == int(gidx)
                    and getattr(self, 'last_best_delta', None) is not None):
                value = float(self.last_best_delta)
                return value if np.isfinite(value) else None
            return None
        hits = np.flatnonzero(
            np.asarray(ids, dtype=np.int32).reshape(-1) == int(sample_idx))
        if not hits.size:
            return None
        try:
            value = float(np.asarray(deltas, dtype=np.float64).reshape(-1)[int(hits[0])])
        except (TypeError, ValueError, IndexError):
            return None
        return value if np.isfinite(value) else None

    def has_improving_delta(self, eps=1e-9):
        """True if any solved sample's ``x_plus`` reduces pose cost."""
        deltas = getattr(self, 'last_candidate_deltas', None)
        if deltas is None:
            return False
        finite = np.asarray(deltas, dtype=np.float64).reshape(-1)
        finite = finite[np.isfinite(finite)]
        return bool(finite.size and float(np.max(finite)) > float(eps))

    def has_delta_span(self):
        """True when this cycle's x_plus reductions are not all identical."""
        try:
            return float(getattr(self, 'last_delta_scale', 0.0) or 0.0) > 1e-12
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _delta_norm_stats(values):
        """Median / robust scale, the usual neural-net input standardization.

        Scale is IQR/1.349 when it is informative (RobustScaler), otherwise
        the population std.  Identical values yield scale 0.
        """
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, 0.0
        center = float(np.median(finite))
        scale = 0.0
        if finite.size >= 4:
            q25, q75 = np.percentile(finite, [25.0, 75.0])
            iqr = float(q75 - q25)
            if iqr > 1e-12:
                scale = iqr / 1.349
        if scale <= 1e-12 and finite.size >= 2:
            scale = float(np.std(finite))
        return center, scale

    @staticmethod
    def _nn_normalize_deltas(values, center=None, scale=None):
        """``tanh((Δ - median) / scale)`` in (-1, 1), order-preserving."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        out = np.full(values.shape, np.nan, dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            return out
        if center is None or scale is None:
            center, scale = LambdaContactControlOptimizer._delta_norm_stats(
                values[finite])
        if float(scale) <= 1e-12:
            out[finite] = 0.0
            return out
        out[finite] = np.tanh((values[finite] - float(center)) / float(scale))
        return out

    def normalize_pose_cost(self, cost):
        """Same unit-range as ranking: 0 at this cycle's C(x_plus) min.

        The high end is the Tukey fence stored by ``_rescore_candidate_costs``.
        A real MuJoCo cost may fall outside that band; it is not clipped
        here so a graze worse than every predicted x_plus stays visible.
        """
        if cost is None:
            return None
        try:
            value = float(cost)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        lo = float(getattr(self, 'last_cost_lo', 0.0) or 0.0)
        hi = float(getattr(self, 'last_cost_hi', 0.0) or 0.0)
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        span = hi - lo
        if span <= 1e-12:
            return 0.0 if value <= lo + 1e-12 else 1.0
        return float((value - lo) / span)

    def normalize_cost_delta(self, delta):
        """Project a raw ΔC through the ranking ``C(x_plus)`` band.

        Ranking scores are unit-range predicted pose costs, which is an
        affine map of ``ΔC = C(now) - C``.  Predicted and actual
        reductions must use that same map:

            1 - unit_range(C_now - ΔC)

        so a no-change (airborne ΔC=0) sits at or below 0 whenever
        C(now) is above every predicted C(x_plus), the selected
        x_plus maps near +1, and a graze that raises pose cost goes
        negative.  Do not fall back to ``ΔC / C(now)``: that scale
        was left behind when ranking switched to unit-range C, and
        it made actual look ~20× smaller than the ranking scores.
        """
        if delta is None:
            return None
        try:
            value = float(delta)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        now = float(getattr(self, 'last_pose_cost_now', 0.0) or 0.0)
        if not np.isfinite(now):
            return None
        score = self.normalize_pose_cost(now - value)
        if score is None:
            return None
        return float(np.clip(1.0 - score, -1.0, 1.0))

    def _decompose_pose_residuals(self, x_d, x_plus_buffer, raw_costs):
        """Unweighted pose residuals of each solved candidate."""
        raw = np.asarray(raw_costs, dtype=np.float64).reshape(-1)
        n = int(raw.size)
        pos = np.full(n, np.inf, dtype=np.float64)
        ori = np.full(n, np.inf, dtype=np.float64)
        xd = np.asarray(x_d, dtype=np.float64).reshape(7)
        qd = xd[3:7]
        qd = qd / max(float(np.linalg.norm(qd)), 1e-9)
        if x_plus_buffer is None:
            return pos, ori
        n_buf = len(x_plus_buffer)
        for i in range(min(n, n_buf)):
            if not np.isfinite(raw[i]):
                continue
            x = np.asarray(x_plus_buffer[i], dtype=np.float64).reshape(-1)
            if x.size < 7 or not np.isfinite(x[:7]).all():
                continue
            dpos = x[:3] - xd[:3]
            pos[i] = float(np.dot(dpos, dpos))
            q = x[3:7]
            qn = float(np.linalg.norm(q))
            if qn <= 1e-12:
                continue
            q = q / qn
            ori[i] = float(1.0 - np.clip(np.dot(q, qd), -1.0, 1.0) ** 2)
        return pos, ori

    def _ema_rank_scores(self, ids, scores, rate=None):
        """Blend this cycle's scores with the previous cycle, by sample id."""
        ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if rate is None:
            rate = float(getattr(self, 'rank_score_ema_rate', 0.35))
        rate = float(np.clip(rate, 0.0, 1.0))
        out = scores.copy()
        prev_ids = getattr(self, '_rank_score_ids', None)
        prev_scores = getattr(self, '_rank_score_values', None)
        if prev_ids is not None and prev_scores is not None and rate < 1.0 - 1e-12:
            prev_map = {}
            for idx, score in zip(np.asarray(prev_ids).reshape(-1),
                                  np.asarray(prev_scores, dtype=np.float64).reshape(-1)):
                if np.isfinite(score):
                    prev_map[int(idx)] = float(score)
            for k, idx in enumerate(ids):
                old = prev_map.get(int(idx))
                if old is not None and np.isfinite(out[k]):
                    out[k] = (1.0 - rate) * old + rate * out[k]
        self._rank_score_ids = ids.copy()
        self._rank_score_values = out.copy()
        return out

    def _apply_plateau_hold(self, ids, scores):
        """Keep last_global when it still sits on this cycle's cost plateau.

        A 1e-4 raw-cost jitter used to flip best_contact across the mesh
        and drag the via with it.  A clearly better sample still wins
        immediately, which is the travel behaviour we want.
        """
        ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1).copy()
        prev = getattr(self, 'last_global_idx', None)
        finite = np.isfinite(scores)
        if prev is None or not np.any(finite):
            return scores
        hits = np.flatnonzero(ids == int(prev))
        if not hits.size or not finite[int(hits[0])]:
            return scores
        finite_scores = scores[finite]
        span = float(np.max(finite_scores) - np.min(finite_scores))
        confidence = float(np.clip(
            getattr(self, 'contact_switch_confidence', 1.0), 0.0, 1.0))
        hold = float(self.contact_switch_margin_abs +
                     confidence * self.contact_switch_margin_ratio *
                     max(span, 1e-4))
        prev_local = int(hits[0])
        best = float(np.min(finite_scores))
        if float(scores[prev_local]) <= best + hold:
            scores[prev_local] = best - 1e-9
        return scores

    def _rescore_candidate_costs(self, ids, raw_costs, x_plus_buffer, x_d,
                                 current_x=None):
        """Rank by unit-range predicted pose cost ``C(x_plus)``.

        Argmin of ``C(x_plus)`` is the same as argmax of
        ``ΔC = C(now) - C(x_plus)``.  These scores are the physical
        ranking used for ``best_contact`` / ``last_global_idx``.
        Cycle-to-cycle EMA and plateau-hold used to rewrite the
        argmin so the yellow marker lagged the true reduction.
        """
        if current_x is None:
            current_x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        deltas = self._pose_cost_deltas(current_x, x_plus_buffer, x_d, raw_costs)
        self.last_candidate_deltas = np.asarray(deltas, dtype=np.float64).reshape(-1)
        finite = self.last_candidate_deltas[np.isfinite(self.last_candidate_deltas)]
        if finite.size:
            self.last_delta_lo = float(np.min(finite))
            self.last_delta_hi = float(np.max(finite))
        else:
            self.last_delta_lo = 0.0
            self.last_delta_hi = 0.0
        center, scale = self._delta_norm_stats(finite)
        self.last_delta_center = float(center)
        self.last_delta_scale = float(scale)
        self.last_candidate_delta_norms = self._nn_normalize_deltas(
            self.last_candidate_deltas, center, scale)
        c_plus = np.full(self.last_candidate_deltas.shape, np.nan, dtype=np.float64)
        now = float(getattr(self, 'last_pose_cost_now', 0.0) or 0.0)
        valid_delta = np.isfinite(self.last_candidate_deltas)
        if np.isfinite(now):
            c_plus[valid_delta] = now - self.last_candidate_deltas[valid_delta]
        self.last_candidate_pose_costs = c_plus
        finite_c = c_plus[np.isfinite(c_plus)]
        if finite_c.size:
            lo, hi = self._span_limits(finite_c)
            self.last_cost_lo = 0.0 if lo is None else float(lo)
            self.last_cost_hi = 0.0 if hi is None else float(hi)
        else:
            self.last_cost_lo = 0.0
            self.last_cost_hi = 0.0
        scores = self._unit_range_costs(c_plus)
        # A non-improving x_plus cannot be best_contact once any
        # sample reduces pose cost.
        improving = valid_delta & (self.last_candidate_deltas > 1e-9)
        if np.any(improving):
            scores[~improving] = np.inf
        # Do not EMA or plateau-hold these scores.  Unit-range already
        # preserves argmin C(x_plus) = argmax ΔC; hold/EMA then freeze
        # last_global on a near-best incumbent so the yellow marker is
        # no longer the pose-improving optimum (best_delta < delta_hi).
        # Lazy switch stays in _select_contact_candidate.
        if np.any(np.isfinite(scores)):
            self._record_best_delta(int(np.nanargmin(scores)))
        else:
            self.last_best_delta = None
        return scores

    def _store_candidate_buffers(self, ids, costs, x_plus_buffer, force_buffer,
                                 raw_costs=None):
        self.last_candidate_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        self.last_candidate_costs = np.asarray(costs, dtype=np.float64).reshape(-1)
        if raw_costs is None:
            raw_costs = getattr(self, 'last_candidate_raw_costs', None)
        if raw_costs is not None:
            self.last_candidate_raw_costs = np.asarray(
                raw_costs, dtype=np.float64).reshape(-1)
        self.last_candidate_x_plus = [
            np.asarray(x, dtype=np.float64).reshape(7) for x in x_plus_buffer]
        self.last_candidate_forces = [
            np.asarray(f, dtype=np.float32).reshape(-1) for f in force_buffer]
        finite = np.isfinite(self.last_candidate_costs)
        order = np.argsort(np.where(finite, self.last_candidate_costs, np.inf))
        k = min(int(getattr(self, 'top_k', 2)), int(np.count_nonzero(finite)))
        self.last_topk_ids = self.last_candidate_ids[order[:k]] if k else np.zeros(0, dtype=np.int32)
        self.last_topk_costs = self.last_candidate_costs[order[:k]] if k else np.zeros(0, dtype=np.float64)

    def _ablation_guidance_local(self, ids, finite_mask):
        """Index into ``ids`` for random or nearest guidance, else None."""
        mode = str(getattr(self, 'cso_guidance', 'cso'))
        if mode not in ('random', 'nearest'):
            return None
        local = np.flatnonzero(np.asarray(finite_mask, dtype=bool).reshape(-1))
        if local.size == 0:
            return None
        if mode == 'nearest':
            query = getattr(self, 'rank_query_local', None)
            if query is None:
                return None
            chosen_ids = np.asarray(ids, dtype=np.int32).reshape(-1)[local]
            points = np.asarray(self.sample_point, dtype=np.float64)[chosen_ids]
            dist = np.linalg.norm(
                points - np.asarray(query, dtype=np.float64).reshape(1, 3), axis=1)
            return int(local[int(np.argmin(dist))])
        rng = getattr(self, '_cso_rng', None)
        if rng is None:
            rng = np.random.default_rng(0)
            self._cso_rng = rng
        return int(local[int(rng.integers(int(local.size)))])

    def _finish_guided_candidate(self, ids, costs, force_buffer, chosen_local,
                                 finite_costs, prev_sample_idx):
        ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        costs = np.asarray(costs, dtype=np.float64).reshape(-1)
        chosen_local = int(chosen_local)
        chosen_idx = int(ids[chosen_local])
        self.last_global_idx = chosen_idx
        self.last_global_total_cost = float(costs[chosen_local])
        self.last_switch_required = bool(
            prev_sample_idx is not None and int(prev_sample_idx) != chosen_idx)
        self._pending_selected_idx = None
        self._pending_selected_count = 0
        self.last_best_idx = chosen_idx
        if force_buffer is None:
            self.last_best_force = None
        else:
            self.last_best_force = np.asarray(force_buffer[chosen_local], dtype=np.float32)
        self.last_selected_local = np.asarray(self.sample_point[chosen_idx], dtype=np.float32)
        self.last_selected_idx = chosen_idx
        self.last_transition_cost = 0.0
        self.last_selected_total_cost = float(costs[chosen_local])
        finite_costs = np.asarray(finite_costs, dtype=np.float64).reshape(-1)
        return chosen_idx, float(np.min(finite_costs)), float(np.max(finite_costs)), chosen_local

    def choose_nearby_topk_idx(self, query_local, k=None, quality_frac=0.015):
        """Prefer a nearer runner-up only if it is almost as improving.

        Unit-range scores collapse the top-k onto ``~0``, so a nearby
        foot/crease used to beat the true λ optimum by proximity.  Rank
        runner-ups by pose-cost reduction ``ΔC`` instead, and never
        replace a stable destination with a crease tip.
        """
        if str(getattr(self, 'cso_guidance', 'cso')) in ('random', 'nearest'):
            idx = getattr(self, 'last_selected_idx', None)
            if idx is None:
                idx = getattr(self, 'last_global_idx', None)
            return None if idx is None else int(idx)
        ids = np.asarray(getattr(self, 'last_topk_ids', []), dtype=np.int32).reshape(-1)
        if ids.size == 0:
            return None
        k_keep = int(self.top_k if k is None else k)
        take = min(max(1, k_keep), int(ids.size))
        ids = ids[:take]
        deltas = np.full(take, np.nan, dtype=np.float64)
        for i, idx in enumerate(ids):
            value = self.pose_delta_for_sample(int(idx))
            if value is not None:
                deltas[i] = float(value)
        crease = (
            self._destination_crease_mask(ids)
            if getattr(self, 'point_curvature', None) is not None
            else np.zeros(take, dtype=bool)
        )
        same = self._same_side_mask(ids, query_local)
        # A crease never owns the yellow marker.  An opposite face only
        # yields when a same-side member is a ΔC near-tie (table-J).
        if crease[0] and np.any(~crease):
            stable = np.flatnonzero(~crease)
            if not np.any(np.isfinite(deltas[stable])):
                return int(ids[int(stable[0])])
            return int(ids[int(stable[int(np.nanargmax(deltas[stable]))])])
        same_stable = same & ~crease
        if (not same[0]) and np.any(same_stable):
            alt = np.flatnonzero(same_stable)
            pick = int(alt[int(np.nanargmax(deltas[alt]))]) if np.any(
                np.isfinite(deltas[alt])) else int(alt[0])
            if self._delta_near_tie(deltas[0], deltas[pick], quality_frac):
                return int(ids[pick])
            return int(ids[0])
        if not np.isfinite(deltas[0]):
            gidx = getattr(self, 'last_global_idx', None)
            return int(gidx) if gidx is not None else int(ids[0])
        best = float(deltas[0])
        ok = np.zeros(take, dtype=bool)
        ok[0] = True
        if take > 1:
            ok[1:] = np.array([
                self._delta_near_tie(best, delta, quality_frac)
                for delta in deltas[1:]]) & ~crease[1:]
        query = np.asarray(query_local, dtype=np.float64).reshape(3)
        pts = np.asarray(self.sample_point[ids], dtype=np.float64)
        dist = np.linalg.norm(pts - query[None, :], axis=1)
        dist = np.where(ok, dist, np.inf)
        if not np.any(np.isfinite(dist)):
            gidx = getattr(self, 'last_global_idx', None)
            return int(gidx) if gidx is not None else int(ids[0])
        return int(ids[int(np.argmin(dist))])
    
    def get_availble_point_idx(self, pos, R, target_pos, threshold=0.025,
                               viewpoint_local=None, viewpoint_cos=-0.50,
                               heading_filter=True, floor_z=0.0,
                               support_point=None, support_normal=None):
        """Return sampled contacts whose fingertip target clears the support.

        ``self.normal`` points into the object.  The fingertip centre is
        therefore approached as ``surface - clearance * normal``.  Checking
        that target, instead of only the surface height, removes underside
        points that would put the fingertip sphere below the table while
        allowing the same mesh points again after an object is flipped.

        ``support_normal`` / ``support_point`` replace the world-up table
        when the object sits on a ramp.  With the default ``+Z`` plane
        this is the original ``floor_z`` test.
        """
        centers_world = (R @ self.sample_point.T).T + pos
        # Filter by the height of the *reachable fingertip centre*, rather
        # than allowing an arbitrary band below the floor.  ``normal`` is
        # stored inward, so subtracting it moves from the surface outwards,
        # matching the target construction in test_0902.py.  A downward
        # facing underside therefore gets rejected near the floor, while an
        # upward/side-facing point at a similarly low height remains usable
        # during a flip.
        floor_z = float(floor_z)
        if support_normal is None:
            support_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            support_point = np.array([0.0, 0.0, floor_z], dtype=np.float64)
        else:
            support_normal = np.asarray(support_normal, dtype=np.float64).reshape(3)
            nrm = float(np.linalg.norm(support_normal))
            support_normal = (
                support_normal / nrm
                if nrm > 1e-8
                else np.array([0.0, 0.0, 1.0], dtype=np.float64)
            )
            support_point = (
                np.array([0.0, 0.0, floor_z], dtype=np.float64)
                if support_point is None
                else np.asarray(support_point, dtype=np.float64).reshape(3)
            )
        # Even with a zero CLI threshold, the sphere centre must remain at
        # least one clearance above the plane.  The default threshold adds a
        # small extra safety margin without imposing a large global height
        # cutoff on flip contacts.
        floor_margin = max(float(threshold), self.fingertip_clearance)
        inward_world = (R @ self.normal.T).T
        outward_world = -inward_world
        signed = (centers_world - support_point[None, :]) @ support_normal
        inward_along = inward_world @ support_normal
        fingertip_center_signed = signed - self.fingertip_clearance * inward_along
        # The sphere occupies ``clearance`` below its centre even when the
        # contact normal is sideways.  A downward-facing sole that has
        # rotated just enough to clear the old fingertip-z test still cannot
        # be pressed from above and must not win ranking.
        sphere_low_signed = fingertip_center_signed - self.fingertip_clearance
        underside = (
            ((outward_world @ support_normal) < -0.35)
            & (signed < 0.045)
        )
        common_mask = (
            (signed > 0.0)
            & (fingertip_center_signed > floor_margin)
            & (sphere_low_signed > 0.0)
            & ~underside
        )
        # A brick sitting on a ramp has a large +normal face.  Those top
        # samples score as a cheap hold once gravity is cancelled, so
        # ranking parks p_arm / via in the air and the arm climbs to them.
        if abs(float(support_normal[2])) < 0.999:
            common_mask = common_mask & ((outward_world @ support_normal) < 0.45)

        # A global lambda optimum may lie on the far side of a concave
        # silhouette (the elephant's ear/foot are typical samples).  The
        # straight segment from the current fingertip to such a patch passes
        # through the object, so MPC repeatedly collides and escapes without
        # making pose progress.  When a viewpoint is supplied, retain the
        # visible hemisphere with a soft cosine gate.  Keep the gate soft so
        # silhouette points remain eligible; the fallback below still keeps
        # the optimizer defined if every point is rejected.
        if viewpoint_local is not None:
            view = np.asarray(viewpoint_local, dtype=np.float64).reshape(3)
            rel = view[None, :] - self.sample_point
            rel_norm = np.linalg.norm(rel, axis=1)
            outward = -np.asarray(self.normal, dtype=np.float64)
            view_cos = np.sum(outward * rel, axis=1) / np.maximum(rel_norm, 1e-9)
            common_mask = common_mask & (view_cos >= float(viewpoint_cos))

        direction = target_pos - pos
        dis = np.linalg.norm(direction[:2])
        
        if heading_filter and dis > 5e-2:
            direction /= np.linalg.norm(direction)
            vertex_normals_world = (R @ self.normal.T).T
            face_dot_products = vertex_normals_world @ direction
            # Do not discard vertical underside normals.  They have almost
            # zero dot product with the horizontal target direction but can
            # generate the required lifting torque.  Exclude only normals
            # strongly opposing the desired planar motion.
            common_mask = common_mask & (face_dot_products > -0.2)

        available_idx = np.where(common_mask)[0]
        if available_idx.size:
            return available_idx

        # Keep the optimizer well-defined if a transient pose leaves every
        # sampled point below the gate.  Prefer the greatest clearance that
        # is still not a top face on a ramp; otherwise the fallback undoes
        # the filter and parks p_arm in the air again.
        above_floor = np.flatnonzero(signed > 0.0)
        if abs(float(support_normal[2])) < 0.999:
            side_ok = above_floor[
                (outward_world[above_floor] @ support_normal) < 0.45
            ] if above_floor.size else above_floor
            if side_ok.size:
                above_floor = side_ok
        if above_floor.size:
            best_idx = int(above_floor[np.argmax(fingertip_center_signed[above_floor])])
        else:
            # This can only happen after severe simulation penetration; keep
            # a deterministic least-penetrating point for recovery.
            best_idx = int(np.argmax(fingertip_center_signed))
        return np.asarray([best_idx], dtype=np.int64)
    
# 使用示例
if __name__ == "__main__":
    # 创建优化器实例 (10cm x 10cm x 10cm 的方块)
    optimizer = LambdaContactControlOptimizer(
        box_size=(0.1, 0.1, 0.1),
        obj_mass=0.01,
        arm_stiffness=200,
        ground_friction=0.9,
        arm_friction=0.9,
        contact_stiffness=10,
        time_step=0.05,
        max_contacts=5
    )
    
    # 更新接触点 (底面和机械臂接触点)
    new_contact_points = [
        {'type': 'ground', 'position': np.array([0.05, 0.05, -0.05]), 'face': 'bottom'},
        {'type': 'ground', 'position': np.array([-0.05, 0.05, -0.05]), 'face': 'bottom'},
        {'type': 'ground', 'position': np.array([0.05, -0.05, -0.05]), 'face': 'bottom'},
        {'type': 'ground', 'position': np.array([-0.05, -0.05, -0.05]), 'face': 'bottom'},
    ]
    optimizer.update_contact_points(new_contact_points)
    
    # 设置目标位姿和当前位姿
    target_pose = np.array([0.1, -0., 0.0, 0.0, 0.0, 0.5])  # [x,y,z, rx,ry,rz]
    current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    tau_o = np.array([0.0, 0.0, -0.01 * 9.81, 0.0, 0.0, 0.0])  # 重力
    p_arm = np.array([0.0, -0.05, -0.0])  # 左侧
    n_arm = np.array([-0, 1, 0])  # 法线方向指向右侧

    lam_arm, x_plus_opt, info = optimizer.optimize_control_input(
        target_pose, current_pose, tau_o, n_arm=n_arm, p_arm=p_arm
    )
    
    # 转换结果为NumPy数组
    lam_arm_np = np.array(cs.evalf(lam_arm)).flatten()
    x_plus_opt_np = np.array(cs.evalf(x_plus_opt)).flatten()
    
    # 打印结果
    print("\n优化结果:")
    print(f"求解时间: {info['solve_time']:.6f}s")
    # print(f"位置误差: {info['position_error']:.6f}")
    # print(f"姿态误差: {info['orientation_error']:.6f}")
    print(f"最优控制输入: {lam_arm_np}")
    print(f"预测位姿: {x_plus_opt_np}")
    print(f"目标位姿: {target_pose}")
