"""Official three-ball objective, with sums generalized to the fingertip count.

The upstream originals and their SHA256 values are retained under upstream/.
No Franka verify, candidate selection or stage-dependent terms are added.
"""
import casadi as cs
from . import rotations


class Cost:
    def __init__(self, param):
        self.param_ = param

    def init_cost_fn(self):
        p = self.param_
        x = cs.SX.sym('x', p.n_qpos_)
        u = cs.SX.sym('u', p.n_cmd_)
        target = cs.SX.sym('target', 7)
        tips = [x[i:i+3] for i in range(7, p.n_qpos_, 3)]
        rotation = rotations.cs_quat2rot_fn(x[3:7])
        offsets = [rotation.T@(tip-x[:3]) for tip in tips]
        contact = sum(cs.sumsqr(tip-x[:3]) for tip in tips)
        closure = cs.sumsqr(sum(v/(cs.sumsqr(v)+1e-4) for v in offsets))
        path = contact + .1*closure + cs.sumsqr(u)
        terminal = p.terminal_scale_*(p.p_weighting_*cs.sumsqr(x[:3]-target[:3])
                                      + p.q_weighting_*(1-cs.dot(x[3:7], target[3:7])**2))
        return (cs.Function('path_cost_fn', [x, u, target], [path]),
                cs.Function('final_cost_fn', [x, target], [terminal]))
