"""SCM ranking and support-plane filter on top of the pushed Lambda optimizer.

Isaac keeps constructing ``LambdaContactControlOptimizer``.  Fingertip
rollout and tilted push construct this subclass so crease / same-side
ranking, dwell recovery, and ramp support filtering stay out of that path.
"""
import numpy as np
from planning.mlqp_point import LambdaContactControlOptimizer


def promote(optimizer):
    """Attach SCM ranking to an optimizer built by ``build_lambda_optimizer``."""
    optimizer.__class__ = LambdaContactControlOptimizerSCM
    if not hasattr(optimizer, "rank_query_local"):
        optimizer.rank_query_local = None
    return optimizer


class LambdaContactControlOptimizerSCM(LambdaContactControlOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank_query_local = None

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

    def _destination_crease_mask(self, ids=None):
        """True on a sharp tip/foot that must not own ``best_contact``.

        Ranking keeps high-max / low-mean faces in the pool so a useful
        flank next to a crease can still be pressed.  The yellow
        destination cannot sit on the crease itself: table-J at the COM
        overstates the lever arm of a foot tip, and nearby-topk then
        sticks ``best_contact`` there for the whole trial.
        """
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
            return True
        margin = max(float(quality_frac) * max(abs(float(winner_delta)), 1e-3), 1e-3)
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
        from planning.acados_env import quiet_acados_stderr
        with quiet_acados_stderr():
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

    def choose_contact_points(self, x_d, current_x, tau_o, visible_face_idx, v_last=None,
                              contact_anchor_local=None, contact_anchor_idx=None,
                              force_required=False, query_local=None):
        if query_local is not None:
            self.rank_query_local = np.asarray(query_local, dtype=np.float64).reshape(3)
        return super().choose_contact_points(
            x_d, current_x, tau_o, visible_face_idx, v_last=v_last,
            contact_anchor_local=contact_anchor_local,
            contact_anchor_idx=contact_anchor_idx,
            force_required=force_required,
        )

    def choose_nearby_topk_idx(self, query_local, k=None, quality_frac=0.015):
        """Prefer a nearer runner-up only if it is almost as improving.

        Unit-range scores collapse the top-k onto ``~0``, so a nearby
        foot/crease used to beat the true λ optimum by proximity.  Rank
        runner-ups by pose-cost reduction ``ΔC`` instead, and never
        replace a stable destination with a crease tip.
        """
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
            margin = max(float(quality_frac) * max(abs(best), 1e-3), 1e-3)
            ok[1:] = np.isfinite(deltas[1:]) & (deltas[1:] >= best - margin) & ~crease[1:]
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

