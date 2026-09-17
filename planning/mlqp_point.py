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
from scipy.sparse.csgraph import dijkstra
try:
    # The reference fingertip experiment samples mesh vertices.  Keep the
    # original projection helper here; ``project_point1`` uses face centers
    # and therefore changes the contact-point candidates.
    from project_point import ProjectionPoint
except:
    from planning.project_point import ProjectionPoint

class LambdaContactControlOptimizer:
    # Smooth positive part used by all backends.  The contact projection is
    # physically unilateral, but a hard max has an undefined derivative at
    # zero and makes the acados SQP model disagree with its post-processing.
    _positive_part_eps = 1e-12

    def __init__(self, mesh_path, obj_mass=0.01, arm_friction=0.9, 
                 contact_stiffness=12.5, time_step=0.01, max_contacts=10, sample_num=70,
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
                 wrench_is_force=False,
                 top_k=2):
        # 系统参数
        self.m = obj_mass
        self.mu_arm_obj = arm_friction
        self.K_contact = contact_stiffness
        self.h = time_step
        # Rollout receives a physical force from the fingertip PD loop.  The
        # legacy ideal-contact path historically supplied an impulse-like
        # lambda; retain that convention there for backwards compatibility.
        self.wrench_is_force = bool(wrench_is_force)
        self.max_contacts = max_contacts
        self.pp = ProjectionPoint(mesh_path, scale_factors,
                                  collision_hull=collision_hull,
                                  normal_stability_cos=normal_stability_cos)

        self.sample_num = sample_num
        self.top_k = max(1, int(top_k))
        # Match the reference optimizer: uniformly/farthest sampled vertices
        # with their vertex normals, rather than face-center candidates.
        self.sampling_frame = self.pp.sample_vertices_with_normals(num_samples=self.sample_num)
        self.sample_point = self.sampling_frame['points']
        self.normal = self.sampling_frame['normals']
        self.t1 = self.sampling_frame['tangent1']
        self.t2 = self.sampling_frame['tangent2']
        self.sample_vertex_indices = np.asarray([
            int(self.pp.project_point_to_mesh(point)[0]) for point in self.sample_point
        ], dtype=np.int32)
        self.sample_geodesic = self._precompute_sample_geodesic()

        self.J_tilde = np.zeros([4 * self.max_contacts, 6])

        # 构建系统刚度矩阵Q
        self.obj_inertia = np.eye(6)
        if obj_inertia is None:
            # Preserve the historical effective inertia for ranking.
            # Using MuJoCo's raw free-body mass here (especially the 1e-6
            # rotational block) makes the reduced one-step model explode;
            # execution still uses MuJoCo.
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
        self.Q_inv = np.linalg.inv(Q + 1e-8 * np.eye(Q.shape[0]))

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
        self.max_normal_force = self.max_contact_force / np.sqrt(1.0 + 2.0 * self.mu_arm_obj ** 2)
        if self.max_normal_force <= 0.001:
            raise ValueError('max_contact_force is too small for the minimum normal force')
        # Prefer acados; IPOPT remains the compiled fallback.
        # The Torch backend is selected explicitly by ``torch-lbfgs``.
        solver = str(solver).strip().lower()
        if solver == 'snopt':
            solver = 'acados'
        self.solver = solver
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
        self.last_topk_costs = np.zeros(0, dtype=np.float64)
        # Cycle-normalized ranking scores, keyed by sample id.  The NLP
        # still minimizes the physical pos/ori/force mix; verify and
        # best_contact compare the unit-range scores instead.
        self._rank_score_ids = None
        self._rank_score_values = None
        self.rank_score_ema_rate = 0.35
        self.last_candidate_delta_norms = None
        # Sample-neighborhood curvature.  Worst-neighbour disagreement
        # flags a dihedral; mean disagreement flags a true corner/tip.
        # Ranking drops only the conjunction so a flat side face that
        # merely sees one crease neighbour stays eligible.  Using the
        # max alone deleted ~half the elephant samples and left only
        # belly/back flats whose x_plus cannot improve pose.
        self.curvature_neighbor_k = 8
        self.region_max_point_curvature = 0.25
        self.region_max_mean_curvature = 0.10
        self.point_curvature = self._estimate_point_curvature()
        if self._drop_crease_samples():
            self.sample_vertex_indices = np.asarray([
                int(self.pp.project_point_to_mesh(point)[0])
                for point in self.sample_point
            ], dtype=np.int32)
            self.sample_geodesic = self._precompute_sample_geodesic()
        # Temporary blacklist for patches that were reached without yielding
        # contact or pose progress.  This prevents immediate re-selection of
        # the same low-authority ear/foot neighborhood.
        self._blocked_contact_indices = {}
        self._contact_patch_failures = {}
        self.contact_patch_max_block_cycles = 320
        self.init_utils()
        self._precompile_optimization_function()
        self.acados_solver = None
        self.acados_error = None
        if self.solver not in ('ipopt', 'torch-lbfgs', 'torch-gn'):
            try:
                self.acados_solver = self._build_acados_contact_solver()
                self.solver = 'acados'
            except Exception as exc:
                self.acados_error = exc
                print(f'acados contact solver unavailable; using IPOPT fallback: {exc}')
                self.solver = 'ipopt'

    def update_Jacobian(self, J_tilde=None):
        required_rows = 4 * self.max_contacts
        """更新环境接触雅可比矩阵"""
        if J_tilde is None:
            return self.J_tilde

        J_tilde = np.asarray(J_tilde)[:, :6]
        padded_J_tilde = np.zeros((required_rows, 6))
        valid_rows = min(J_tilde.shape[0], required_rows)
        padded_J_tilde[:valid_rows, :] = J_tilde[:valid_rows, :]
        self.J_tilde = padded_J_tilde
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
                              block_radius=None, time_decay=False):
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
                self.set_contact_switch_confidence(1.0)
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
        """Worst-neighbour normal change; high at creases and sharp tips.

        ``0.5 * (1 - n·n_k)`` is 0 on a flat patch and 0.5 when a neighbour
        is orthogonal.  Using the mean of those terms lets a foot/leg
        junction hide among same-face samples; the max does not.
        """
        n_samples = int(len(self.sample_point))
        if n_samples <= 1:
            return np.zeros((n_samples,), dtype=np.float64)
        from scipy.spatial import cKDTree
        query_k = min(n_samples, self.curvature_neighbor_k + 1)
        _, neighbor_idx = cKDTree(self.sample_point).query(self.sample_point, k=query_k)
        neighbor_idx = np.asarray(neighbor_idx, dtype=int)
        if neighbor_idx.ndim == 1:
            neighbor_idx = neighbor_idx.reshape(-1, 1)
        if neighbor_idx.shape[1] <= 1:
            return np.zeros((n_samples,), dtype=np.float64)
        neighbor_idx = neighbor_idx[:, 1:]
        neighbor_normals = self.normal[neighbor_idx]
        ref_normals = self.normal[:, None, :]
        normal_dot = np.clip(np.sum(ref_normals * neighbor_normals, axis=2), -1.0, 1.0)
        disagreement = 0.5 * (1.0 - normal_dot)
        self.point_curvature_mean = np.asarray(
            np.mean(disagreement, axis=1), dtype=np.float64)
        return np.asarray(np.max(disagreement, axis=1), dtype=np.float64)

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
            stability = getattr(self.pp, 'vertex_normal_stability', None)
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
        """
        Return the reference environment stiffness matrix in block form.

        The historical name is kept because the acados/Torch adapters use it
        as their parameter hook; each 4-row contact block is K * I.
        """
        J_tilde = np.asarray(J_tilde)
        # The reference uses K = contact_stiffness * h times one identity
        # over all environment-contact rows.  Keep the block-shaped return
        # value required by the acados/Torch adapters, but encode that exact
        # matrix so their dynamics are numerically identical.
        del J_tilde
        block = (self.K_contact * self.h) * np.eye(4)
        return np.tile(block, (self.max_contacts, 1))

    def _precompile_optimization_function(self):
        """预编译优化函数 - 同时优化接触力和接触点位置"""
        opti = cs.Opti()
        
        # 定义优化变量和参数
        x_d = opti.parameter(7)
        current_x = opti.parameter(7)
        v_last = opti.parameter(6)
        J_tilde = opti.parameter(4 * self.max_contacts, 6)
        D_inv = opti.parameter(4 * self.max_contacts, 4)
        tau_o_np = opti.parameter(6)
        p_arm = opti.parameter(3)    # 接触点位置
        n_arm = opti.parameter(3)
        t1 = opti.parameter(3)
        t2 = opti.parameter(3)
        curr_ori_coef = opti.parameter(1)
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
        wrench_scale = self.h if self.wrench_is_force else 1.0
        b = self.h * tau_o_np + wrench_scale * cs.transpose(J_arm_world) @ (R_contact @ lam_arm)
        
        # Environment response, matching the reference K J Q^{-1} b model.
        Q_inv_mx = cs.MX(self.Q_inv)
        Q_inv_b = Q_inv_mx @ b
        # Q^{-1} b is the generalized velocity increment (the dual contact
        # solve is formulated in impulse/velocity-increment coordinates).
        # Integrate it over this optimizer period exactly once below.
        v_plus = Q_inv_b

        for i in range(self.max_contacts):
            row_start = 4 * i
            row_end = row_start + 4
            J_tilde_i = J_tilde[row_start:row_end, :]
            D_inv_i = D_inv[row_start:row_end, :]
            J_tilde_i_Q_inv_b = J_tilde_i @ Q_inv_b
            contact_force_i = -D_inv_i @ J_tilde_i_Q_inv_b
            contact_force_i = 0.5 * (contact_force_i + cs.sqrt(
                contact_force_i * contact_force_i + self._positive_part_eps))
            v_plus += Q_inv_mx @ J_tilde_i.T @ contact_force_i

        v_now = v_last + v_plus

        # 计算预测位姿x+
        x_plus = self.cs_qposInteg_(current_x, v_now)
        
        # 目标函数
        position_error = x_plus[:3] - x_d[:3]
        orientation_error = 1 - cs.dot(x_plus[3:7], x_d[3:7]) ** 2
        friction_cost = cs.sumsqr(lam_arm[1:3])
        force_cost = cs.sumsqr(lam_arm)
        objective = (self.pos_coef * cs.sumsqr(position_error) +
                     self.ori_coef * orientation_error +
                     self.friction_reg_coef * friction_cost +
                     self.force_reg_coef * force_cost)
        opti.minimize(objective)
        
        # 摩擦锥约束
        mu = self.mu_arm_obj
        opti.subject_to(lam_arm[1] <= mu * lam_arm[0])
        opti.subject_to(lam_arm[1] >= -mu * lam_arm[0])
        opti.subject_to(lam_arm[2] <= mu * lam_arm[0])
        opti.subject_to(lam_arm[2] >= -mu * lam_arm[0])
        # Allow the optimizer to disengage when the predicted pose is already
        # close to target.  A hard 1e-3N lower bound causes unavoidable drift
        # and overshoot in the near-target regime.
        opti.subject_to(lam_arm[0] >= 1e-6)
        opti.subject_to(cs.sumsqr(lam_arm) <= self.max_contact_force ** 2)
        opti.subject_to(lam_arm[0] <= self.max_normal_force)

        
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
            [x_d, current_x, v_last, J_tilde, D_inv, tau_o_np, n_arm, t1, t2, p_arm, curr_ori_coef],
            [lam_arm, x_plus, objective],
            ['x_d', 'current_x', 'v_last', 'J_tilde', 'D_inv', 'tau_o_np', 'n_arm', 't1', 't2', 'p_arm', 'curr_ori_coef'],
            ['lam_arm_opt', 'x_plus_opt', 'cost']
        )

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
        geo[np.isnan(geo)] = np.inf
        np.fill_diagonal(geo, 0.0)
        return geo

    def _solve_optimization(self, **kwargs):
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
                result = self.optimization_fn(**kwargs)
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
                    result = self.optimization_fn(**kwargs)
                    self.last_solver_status = 'ipopt-fallback'
                    return result
                except RuntimeError:
                    self.last_solver_status = 'failed'
                    return None
        try:
            result = self.optimization_fn(**kwargs)
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
        # candidate; do not reset the solver between patches.
        cs_p = cs.SX.sym('xd', 7); cs_v = cs.SX.sym('vlast', 6)
        cs_j = cs.SX.sym('Jenv', 4 * self.max_contacts, 6); cs_d = cs.SX.sym('Dinv', 4 * self.max_contacts, 4)
        cs_tau = cs.SX.sym('tau', 6); cs_n = cs.SX.sym('n', 3); cs_t1 = cs.SX.sym('t1', 3); cs_t2 = cs.SX.sym('t2', 3); cs_cp = cs.SX.sym('p', 3)
        prm = cs.vertcat(cs_p, cs_v, cs.reshape(cs_j, -1, 1), cs.reshape(cs_d, -1, 1), cs_tau, cs_n, cs_t1, cs_t2, cs_cp)
        x = cs.SX.sym('x', 7); u = cs.SX.sym('u', 3)
        Jc = cs.SX.zeros(3, 6); Jc[:3, :3] = cs.SX.eye(3); Jc[0,4],Jc[0,5]=cs_cp[2],-cs_cp[1]; Jc[1,3],Jc[1,5]=-cs_cp[2],cs_cp[0]; Jc[2,3],Jc[2,4]=cs_cp[1],-cs_cp[0]
        wrench_scale = self.h if self.wrench_is_force else 1.0
        Rct = cs.horzcat(cs_n, cs_t1, cs_t2); b = self.h*cs_tau + wrench_scale*cs.transpose(Jc) @ (Rct @ u); qinv = cs.DM(self.Q_inv); qib=qinv@b; vp=qib
        for i in range(self.max_contacts):
            sl=slice(4*i,4*(i+1)); Ji=cs_j[sl,:]; Di=cs_d[sl,:]; fi=cs.fmax(-Di@(Ji@qib),0); vp += qinv@cs.transpose(Ji)@fi
        vn=cs_v+vp; quat=x[3:7]; H=cs.vertcat(cs.horzcat(-quat[1],quat[0],quat[3],-quat[2]),cs.horzcat(-quat[2],-quat[3],quat[0],quat[1]),cs.horzcat(-quat[3],quat[2],-quat[1],quat[0])).T
        qn=cs.vertcat(x[:3]+self.h*vn[:3], quat+0.5*self.h*H@vn[3:6]); qn=cs.vertcat(qn[:3],qn[3:7]/cs.norm_2(qn[3:7]))
        # Terminal cost is on the terminal state ``x`` (qn after the
        # discrete transition).  Putting qn itself here is rejected because
        # a terminal cost may not depend on ``u``.
        terminal_pos_cost = self.pos_coef * cs.sumsqr(x[:3] - cs_p[:3])
        terminal_ori_cost = self.ori_coef * (1-cs.dot(x[3:7],cs_p[3:7])**2)
        model=AcadosModel(); model.name=f'contact_lambda_acados_v10_dv_m{self.max_contacts}_f{int(self.max_contact_force*1000)}'; model.x=x; model.u=u; model.p=prm; model.disc_dyn_expr=qn
        model.cost_expr_ext_cost = self.friction_reg_coef * cs.sumsqr(u[1:3])
        model.cost_expr_ext_cost_e = terminal_pos_cost + terminal_ori_cost
        ocp=AcadosOcp(); ocp.model=model; ocp.parameter_values=np.zeros(int(prm.size1())); ocp.cost.cost_type='EXTERNAL'; ocp.cost.cost_type_e='EXTERNAL';
        mu = float(self.mu_arm_obj)
        model.con_h_expr = cs.vertcat(u[0], u[1]-mu*u[0], -u[1]-mu*u[0], u[2]-mu*u[0], -u[2]-mu*u[0], cs.sumsqr(u))
        ocp.constraints.lh = np.array([0.0, -ACADOS_INFTY, -ACADOS_INFTY, -ACADOS_INFTY, -ACADOS_INFTY, 0.0])
        ocp.constraints.uh = np.array([self.max_normal_force, 0., 0., 0., 0., self.max_contact_force ** 2])
        ocp.constraints.idxbx_0=np.arange(7); ocp.constraints.lbx_0=np.zeros(7); ocp.constraints.ubx_0=np.zeros(7)
        ocp.solver_options.N_horizon=1; ocp.solver_options.tf=float(self.h); ocp.solver_options.qp_solver='PARTIAL_CONDENSING_HPIPM'; ocp.solver_options.hessian_approx='EXACT'; ocp.solver_options.integrator_type='DISCRETE'; ocp.solver_options.nlp_solver_type='SQP_RTI'; ocp.solver_options.regularize_method='PROJECT'; ocp.solver_options.print_level=0
        d='/tmp/'+model.name+'_codegen'; os.makedirs(d,exist_ok=True); ocp.code_gen_opts.code_export_directory=d; jf=os.path.join(d,model.name+'.json'); so=os.path.join(d,'libacados_ocp_solver_'+model.name+'.so')
        if os.path.isfile(jf) and os.path.isfile(so):
            return AcadosOcpSolver(ocp,json_file=jf,generate=False,build=False,check_reuse_possible=False,verbose=False)
        return AcadosOcpSolver(ocp,json_file=jf,generate=True,build=True,check_reuse_possible=True,verbose=False)

    def _solve_optimization_acados(self, **kwargs):
        solver=self.acados_solver; xcur=np.asarray(kwargs['current_x'],float).reshape(7)
        p=np.concatenate([np.asarray(kwargs['x_d']).reshape(-1),
                          np.asarray(kwargs['v_last']).reshape(-1),
                          np.asarray(kwargs['J_tilde']).reshape(-1,order='F'),
                          np.asarray(kwargs['D_inv']).reshape(-1,order='F'),
                          np.asarray(kwargs['tau_o_np']).reshape(-1),
                          np.asarray(kwargs['n_arm']).reshape(-1),
                          np.asarray(kwargs['t1']).reshape(-1),
                          np.asarray(kwargs['t2']).reshape(-1),
                          np.asarray(kwargs['p_arm']).reshape(-1)])
        solver.set(0,'x',xcur); solver.set(0,'lbx',xcur); solver.set(0,'ubx',xcur)
        solver.set(0,'u',np.array([.01,0,0])); solver.set(0,'p',p)
        solver.set(1,'x',xcur); solver.set(1,'p',p)
        status=int(solver.solve())
        if status != 0:
            self.acados_qp_failure_count += 1
            raise RuntimeError(f'acados status {status}')
        lam=np.asarray(solver.get(0,'u')).reshape(3)
        if not np.isfinite(lam).all():
            raise FloatingPointError('acados returned non-finite contact wrench')
        fn = float(np.clip(lam[0], 0.0, self.max_normal_force))
        ft = np.clip(lam[1:3], -self.mu_arm_obj * fn, self.mu_arm_obj * fn)
        lam = np.asarray([fn, ft[0], ft[1]], dtype=float)
        cp=np.asarray(kwargs['p_arm'], dtype=float).reshape(3)
        Jc=np.zeros((3, 6), dtype=float); Jc[:, :3] = np.eye(3)
        Jc[0, 4], Jc[0, 5] = cp[2], -cp[1]
        Jc[1, 3], Jc[1, 5] = -cp[2], cp[0]
        Jc[2, 3], Jc[2, 4] = cp[1], -cp[0]
        Rcontact=np.column_stack((kwargs['n_arm'], kwargs['t1'], kwargs['t2']))
        wrench_scale = self.h if self.wrench_is_force else 1.0
        b=self.h*np.asarray(kwargs['tau_o_np']).reshape(6)+wrench_scale*Jc.T@(Rcontact@lam)
        qib=self.Q_inv@b; vplus=qib
        Jenv=np.asarray(kwargs['J_tilde']); D=np.asarray(kwargs['D_inv'])
        for i in range(self.max_contacts):
            sl=slice(4*i,4*(i+1)); fi=np.maximum(-(D[sl]@(Jenv[sl]@qib)),0.0)
            vplus += self.Q_inv@Jenv[sl].T@fi
        vnow=np.asarray(kwargs['v_last']).reshape(6)+vplus
        qnext=np.asarray(self.cs_qposInteg_(xcur,vnow)).reshape(7)
        if (not np.isfinite(lam).all() or lam[0] < -1e-5
                or lam[0] > self.max_normal_force + 1e-3
                or np.linalg.norm(lam) > self.max_contact_force + 1e-3):
            raise RuntimeError(f'invalid contact wrench returned: {lam}')
        pos_err=qnext[:3]-np.asarray(kwargs['x_d'])[:3]
        cost=float(self.pos_coef*np.dot(pos_err,pos_err)+self.ori_coef*(1-np.dot(qnext[3:7],np.asarray(kwargs['x_d'])[3:7])**2)+self.friction_reg_coef*np.dot(lam[1:3],lam[1:3])+self.force_reg_coef*np.dot(lam,lam))
        return {'lam_arm_opt':lam,'x_plus_opt':qnext,'cost':cost}

    def _solve_optimization_torch(self, x_d, current_x, v_last, J_tilde, D_inv,
                                  tau_o_np, n_arm, t1, t2, p_arm, curr_ori_coef):
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
        tau, n, tt1, tt2, p = map(t, (tau_o_np, n_arm, t1, t2, p_arm))
        qinv = t(self.Q_inv)

        # Contact Jacobian [I, -skew(p)] in the convention used above.
        Jc = torch.zeros((3, 6), device=dev, dtype=dt)
        Jc[:, :3] = torch.eye(3, device=dev, dtype=dt)
        Jc[0, 4], Jc[0, 5] = p[2], -p[1]
        Jc[1, 3], Jc[1, 5] = -p[2], p[0]
        Jc[2, 3], Jc[2, 4] = p[1], -p[0]
        Rcontact = torch.stack((n, tt1, tt2), dim=1)

        # Initialize near the IPOPT initial guess fn=.01, ft=0.
        fn0 = (0.01 - 1e-6) / (self.max_normal_force - 1e-6)
        fn0 = float(np.clip(fn0, 1e-4, 1.0 - 1e-4))
        z0 = torch.zeros(3, device=dev, dtype=dt)
        z0[0] = torch.log(torch.tensor(fn0 / (1.0 - fn0), device=dev, dtype=dt))
        z = torch.nn.Parameter(z0)
        opt = torch.optim.LBFGS(
            [z], max_iter=self.torch_max_iter, tolerance_grad=1e-5,
            tolerance_change=1e-9, line_search_fn='strong_wolfe')

        def evaluate():
            fn = 0.001 + (self.max_normal_force - 0.001) * torch.sigmoid(z[0])
            lam = torch.cat((fn.reshape(1), self.mu_arm_obj * fn * torch.tanh(z[1:])))
            wrench_scale = self.h if self.wrench_is_force else 1.0
            b = self.h * tau + wrench_scale * (Jc.T @ (Rcontact @ lam))
            qinv_b = qinv @ b
            vplus = qinv_b
            for i in range(self.max_contacts):
                sl = slice(4 * i, 4 * (i + 1))
                gap_impulse = -(D[sl] @ (Jenv[sl] @ qinv_b))
                fi = 0.5 * (gap_impulse + torch.sqrt(
                    gap_impulse * gap_impulse + self._positive_part_eps))
                vplus = vplus + qinv @ Jenv[sl].T @ fi
            vnow = vl + vplus
            quat = xc[3:7]
            Ht = torch.stack((
                torch.stack((-quat[1], quat[0], quat[3], -quat[2])),
                torch.stack((-quat[2], -quat[3], quat[0], quat[1])),
                torch.stack((-quat[3], quat[2], -quat[1], quat[0])),
            ), dim=0).T
            qnext = torch.cat((xc[:3] + self.h * vnow[:3],
                               quat + 0.5 * self.h * (Ht @ vnow[3:6])))
            qnext = torch.cat((qnext[:3], qnext[3:7] / torch.linalg.vector_norm(qnext[3:7]).clamp_min(1e-8)))
            pos_err = qnext[:3] - xd[:3]
            ori_err = 1.0 - torch.dot(qnext[3:7], xd[3:7]) ** 2
            friction_cost = torch.sum(lam[1:] * lam[1:])
            force_cost = torch.sum(lam * lam)
            cost = (self.pos_coef * torch.sum(pos_err * pos_err) +
                    self.ori_coef * ori_err + self.friction_reg_coef * friction_cost +
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

        fn0 = (0.01 - 0.001) / (self.max_normal_force - 0.001)
        fn0 = float(np.clip(fn0, 1e-4, 1.0 - 1e-4))
        z = torch.nn.Parameter(torch.zeros((count, 3), device=dev, dtype=dt))
        z.data[:, 0] = torch.log(torch.tensor(fn0 / (1.0 - fn0), device=dev, dtype=dt))
        opt = torch.optim.LBFGS([z], max_iter=self.torch_max_iter,
                                tolerance_grad=1e-5, tolerance_change=1e-9,
                                line_search_fn='strong_wolfe')

        def evaluate():
            fn = 1e-6 + (self.max_normal_force - 1e-6) * torch.sigmoid(z[:, 0])
            lam = torch.cat((fn[:, None], self.mu_arm_obj * fn[:, None] * torch.tanh(z[:, 1:])), dim=1)
            wrench = torch.bmm(Rcontact, lam[:, :, None]).squeeze(-1)
            wrench_scale = self.h if self.wrench_is_force else 1.0
            b = self.h * tau[None, :] + wrench_scale * torch.bmm(Jc.transpose(1, 2), wrench[:, :, None]).squeeze(-1)
            qinv_b = torch.matmul(b, qinv.T)
            vplus = qinv_b
            for i in range(self.max_contacts):
                sl = slice(4 * i, 4 * (i + 1))
                env_proj = torch.matmul(qinv_b, Jenv[sl].T)
                gap_impulse = -torch.matmul(env_proj, D[sl].T)
                fi = 0.5 * (gap_impulse + torch.sqrt(
                    gap_impulse * gap_impulse + self._positive_part_eps))
                vplus = vplus + torch.matmul(fi, Jenv[sl] @ qinv.T)
            vnow = vplus + vl[None, :]
            quat = xc[3:7]
            Ht = torch.stack((
                torch.stack((-quat[1], quat[0], quat[3], -quat[2])),
                torch.stack((-quat[2], -quat[3], quat[0], quat[1])),
                torch.stack((-quat[3], quat[2], -quat[1], quat[0])),
            ), dim=0).T
            qpos = xc[:3][None, :] + self.h * vnow[:, :3]
            qrot = quat[None, :] + 0.5 * self.h * torch.matmul(vnow[:, 3:6], Ht.T)
            qrot = qrot / torch.linalg.vector_norm(qrot, dim=1, keepdim=True).clamp_min(1e-8)
            pos_err = qpos - xd[:3][None, :]
            ori_err = 1.0 - torch.sum(qrot * xd[3:7][None, :], dim=1) ** 2
            friction_cost = torch.sum(lam[:, 1:] * lam[:, 1:], dim=1)
            force_cost = torch.sum(lam * lam, dim=1)
            costs = (self.pos_coef * torch.sum(pos_err * pos_err, dim=1) +
                     self.ori_coef * ori_err + self.friction_reg_coef * friction_cost +
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
                               sample_idx=None, candidate_idx=None):
        """优化控制输入 - 更新接口"""
        # This is a separate solve from contact-point ranking and gets its own
        # bounded fallback budget.
        self._acados_fallbacks_this_cycle = 0
        if p_arm is None:
            p_arm = np.array([-1, 0, 0])
        if v_last is None:
            v_last = np.zeros(6)

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
            closest_idx, n, t1, t2  = self.pp.project_point_to_mesh(p_arm)
            p_obj_local = self.pp.scaled_mesh.vertices[closest_idx]
            normal_obj_local = -np.asarray(n)
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
                              force_required=False):
        # Reset the IPOPT fallback budget for this contact-selection cycle.
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
            v_last = np.zeros(6)

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
        """Lambda pose term: ``pos_coef||Δp||² + ori_coef·ori``."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        xd = np.asarray(x_d, dtype=np.float64).reshape(-1)
        if x.size < 7 or xd.size < 7 or not np.isfinite(x[:7]).all():
            return float('inf')
        dpos = x[:3] - xd[:3]
        qn = float(np.linalg.norm(x[3:7]))
        qdn = float(np.linalg.norm(xd[3:7]))
        if qn <= 1e-12 or qdn <= 1e-12:
            return float('inf')
        q = x[3:7] / qn
        qd = xd[3:7] / qdn
        ori = 1.0 - float(np.clip(np.dot(q, qd), -1.0, 1.0)) ** 2
        return (float(self.pos_coef) * float(np.dot(dpos, dpos))
                + float(self.ori_coef) * ori)

    def _pose_cost_deltas(self, current_x, x_plus_buffer, x_d, raw_costs):
        """Cost reduction of applying each ``x_plus_opt`` to the current pose."""
        raw = np.asarray(raw_costs, dtype=np.float64).reshape(-1)
        n = int(raw.size)
        deltas = np.full(n, np.nan, dtype=np.float64)
        c_now = self._pose_cost(current_x, x_d)
        self.last_pose_cost_now = float(c_now) if np.isfinite(c_now) else 0.0
        if not np.isfinite(c_now) or x_plus_buffer is None:
            return deltas
        n_buf = len(x_plus_buffer)
        for i in range(min(n, n_buf)):
            if not np.isfinite(raw[i]):
                continue
            c_plus = self._pose_cost(x_plus_buffer[i], x_d)
            if not np.isfinite(c_plus):
                continue
            deltas[i] = c_now - c_plus
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

    def choose_nearby_topk_idx(self, query_local, k=None, quality_frac=0.015):
        """Prefer a nearer runner-up only if its cost is almost the best."""
        ids = np.asarray(getattr(self, 'last_topk_ids', []), dtype=np.int32).reshape(-1)
        costs = np.asarray(getattr(self, 'last_topk_costs', []), dtype=np.float64).reshape(-1)
        if ids.size == 0:
            return None
        k_keep = int(self.top_k if k is None else k)
        take = min(max(1, k_keep), int(ids.size))
        ids = ids[:take]
        costs = costs[:take]
        best = float(costs[0])
        ok = np.zeros(take, dtype=bool)
        ok[0] = True
        if take > 1:
            # Runner-ups must sit on the best-cost plateau, not merely
            # inside a wide top-k spread (that pulled in weak nearby faces).
            # After cycle-normalization the winner is ~0, so scale the
            # margin by the remaining top-k span rather than |best|.
            span = float(np.max(costs) - np.min(costs))
            margin = max(float(quality_frac) * max(abs(best), span, 1e-3), 1e-3)
            ok[1:] = costs[1:] <= best + margin
        query = np.asarray(query_local, dtype=np.float64).reshape(3)
        pts = np.asarray(self.sample_point[ids], dtype=np.float64)
        dist = np.linalg.norm(pts - query[None, :], axis=1)
        dist = np.where(ok, dist, np.inf)
        return int(ids[int(np.argmin(dist))])
    
    def get_availble_point_idx(self, pos, R, target_pos, threshold=0.025,
                               viewpoint_local=None, viewpoint_cos=-0.50,
                               heading_filter=True, floor_z=0.0):
        """Return sampled contacts whose fingertip target clears the floor.

        ``self.normal`` points into the object.  The fingertip centre is
        therefore approached as ``surface - clearance * normal``.  Checking
        that target, instead of only the surface height, removes underside
        points that would put the fingertip sphere below the table while
        allowing the same mesh points again after an object is flipped.
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
        # Even with a zero CLI threshold, the sphere centre must remain at
        # least one clearance above the plane.  The default threshold adds a
        # small extra safety margin without imposing a large global height
        # cutoff on flip contacts.
        floor_margin = max(float(threshold), self.fingertip_clearance)
        inward_world = (R @ self.normal.T).T
        outward_world = -inward_world
        fingertip_center_z = centers_world[:, 2] - self.fingertip_clearance * inward_world[:, 2]
        # The sphere occupies ``clearance`` below its centre even when the
        # contact normal is sideways.  A downward-facing sole that has
        # rotated just enough to clear the old fingertip-z test still cannot
        # be pressed from above and must not win ranking.
        sphere_low_z = fingertip_center_z - self.fingertip_clearance
        underside = (
            (outward_world[:, 2] < -0.35)
            & (centers_world[:, 2] < floor_z + 0.045)
        )
        common_mask = (
            (centers_world[:, 2] > floor_z)
            & (fingertip_center_z > floor_z + floor_margin)
            & (sphere_low_z > floor_z)
            & ~underside
        )

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
        # sampled point below the gate.  Select the point whose predicted
        # fingertip centre has the greatest clearance; this avoids falling
        # back to an arbitrary (often underside) sample.
        above_floor = np.flatnonzero(centers_world[:, 2] > floor_z)
        if above_floor.size:
            best_idx = int(above_floor[np.argmax(fingertip_center_z[above_floor])])
        else:
            # This can only happen after severe simulation penetration; keep
            # a deterministic least-penetrating point for recovery.
            best_idx = int(np.argmax(fingertip_center_z))
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
