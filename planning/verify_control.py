"""Audit the continuous EMA weight actually supplied to contact MPC."""
import math


VERIFY_CONTROL_VERSION = 'ema_mpc_v1'


class VerifyControlStats:
    def __init__(self):
        self.count = 0
        self.minimum = self.maximum = None
        self.total = 0.0
        self.max_error = 0.0
        self.acceptance_steps = self.accepted_steps = self.acceptance_violations = 0
        self.acceptance_threshold = None

    def observe(self, result):
        info = (result.get('policy') or {}).get('value_info') or {}
        if info.get('acceptance_gate_applied'):
            threshold = info['acceptance_threshold']
            if self.acceptance_threshold is not None and self.acceptance_threshold != threshold:
                raise ValueError('Acceptance threshold changed within a trial')
            self.acceptance_threshold = threshold
            self.acceptance_steps += 1
            accepted = bool(info['accept_p_arm'])
            self.accepted_steps += int(accepted)
            self.acceptance_violations += int((threshold == 0 and not accepted)
                                             or (threshold == 1 and accepted))
        applied = result.get('verify_cost_applied')
        if applied is None:
            return  # Geometric approach did not solve a contact MPC problem.
        applied, score = float(applied), float(result['verify_cost'])
        if not math.isfinite(applied) or not math.isfinite(score):
            raise ValueError('Non-finite verify weight in MPC')
        self.count += 1
        self.total += applied
        self.minimum = applied if self.minimum is None else min(self.minimum, applied)
        self.maximum = applied if self.maximum is None else max(self.maximum, applied)
        self.max_error = max(self.max_error, abs(applied - score))

    def summary(self):
        return dict(acceptance_gate_version='quality_gate_v1' if self.acceptance_steps else None,
                    acceptance_gate_steps=self.acceptance_steps,
                    acceptance_p_arm_steps=self.accepted_steps,
                    acceptance_endpoint_violations=self.acceptance_violations,
                    acceptance_threshold=self.acceptance_threshold,
                    verify_control_version=VERIFY_CONTROL_VERSION,
                    verify_mpc_steps=self.count,
                    verify_mpc_weight_min=self.minimum,
                    verify_mpc_weight_max=self.maximum,
                    verify_mpc_weight_mean=self.total / self.count if self.count else None,
                    verify_mpc_max_abs_error=self.max_error)


def acceptance_decision(threshold, tightness, *, same_patch=False, near_arm=False,
                        is_best_sample=False, holding=False):
    """Quality gate, with the baseline .20 hysteresis width at interior points.

    .55 means tightness < .45 to enter, < .65 to retain. The explicit
    endpoints disable local rejection (0) or always choose the ranked guide (1).
    """
    threshold, tightness = float(threshold), float(tightness)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('verify quality threshold must be finite and in [0, 1]')
    if not math.isfinite(tightness) or not 0 <= tightness <= 1:
        raise ValueError('tightness must be finite and in [0, 1]')
    if threshold == 0:
        return True
    if threshold == 1:
        return False
    # Compare tightness to avoid cancellation in 1-tightness at the boundary.
    limit = min(1., round(1.-threshold + (.20 if holding else 0.), 12))
    return bool(near_arm and (is_best_sample or (same_patch and tightness < limit)))
