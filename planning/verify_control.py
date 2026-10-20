"""Audit the continuous EMA weight actually supplied to contact MPC."""
import math


VERIFY_CONTROL_VERSION = 'ema_mpc_v1'


class VerifyControlStats:
    def __init__(self):
        self.count = 0
        self.minimum = self.maximum = None
        self.total = 0.0
        self.max_error = 0.0

    def observe(self, result):
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
        return dict(verify_control_version=VERIFY_CONTROL_VERSION,
                    verify_mpc_steps=self.count,
                    verify_mpc_weight_min=self.minimum,
                    verify_mpc_weight_max=self.maximum,
                    verify_mpc_weight_mean=self.total / self.count if self.count else None,
                    verify_mpc_max_abs_error=self.max_error)
