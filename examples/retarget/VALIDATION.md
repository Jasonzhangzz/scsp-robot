# Historical validation with the hard shape gate

The 25 mm shape gate below has since been removed. See
[soft-cost validation](SOFT_COST_VALIDATION.md) for the current implementation.

Original GraspSonic DexYCB labels, subject `20200709-subject-01`, camera
`836212060125`. The valid hand annotations are decoded and solved in order.
MANO directions use Wuji bone lengths and palm layout; hand tracking errors
below compare measured robot joints/wrist against that geometric IK reference.
The object column compares physical object motion with the full demonstration
at the commanded hand frame, rather than only the nearest local Lambda goal.

Test configuration: local Python 3.11 / MuJoCo 3.11, acados SQP_RTI, four-step
horizon, 30 ms commands, source 10 ms physics, eight local combinations, and
`--contact-ik-iterations 20 --max-steps 120`. The configurable default contact
IK budget is 50 iterations. Related unit/integration suite: **91 passed**.

| Sequence | Valid geometric IK | Hand frames commanded | Mean finger RMS (rad) | Mean wrist error (mm) | Mean full-demo object error (mm) | Lambda failures | MPC solver failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20200709_142517 | 74/74 | 74 | 0.0291 | 20.9 | 78.2 | 58 | 0 |
| 20200709_142802 | 69/69 | 69 | 0.0431 | 20.3 | 100.8 | 59 | 0 |

All hand frames are commanded in both physical rollouts; posture objectives
remain active during contact. **Full physical object-trajectory reproduction
is not passed.** Multiple contact proposals cannot simultaneously satisfy the
contact tolerance and geometric-pose neighborhood. Infeasible contact plans
fall back to bounded geometric tracking, which preserves the hand motion but
cannot reproduce the demonstrated object lift. Ideal torque diagnostics also
remain available because an object-only force witness does not certify motor
feasibility. No superiority or real-time claim follows from these runs.

The subsequent [lift failure audit](LIFT_DIAGNOSIS.md) reproduces both rollouts
from their saved control commands and separates original-anchor closure,
actual-contact load capacity, IK rejection, and MPC preload activation.

Source MuJoCo options, inertia, friction, actuator gains and force limits are
preserved and audited in each `physics_parameters.json`. Structural convex
mesh mounting overlaps are excluded and listed explicitly; use
`--include-mount-collisions` to retain the unmodified source pairings.

Measured trajectories and plots: `outputs/wuji_retarget_final_review/summary.json`.
Per-sequence physics, contact plans, actual force/ideal force comparisons and
recorded rollouts are in `outputs/wuji_retarget_delivery_142517` and
`outputs/wuji_retarget_delivery_142802`. `render_rollout.py` replays these recorded
states; its green markers are geometric joint targets, not physical contacts.
