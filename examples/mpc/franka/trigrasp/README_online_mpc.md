# Allegro IK pose MPC

`trigrasp.py` uses `AllegroOnlineMPC` with its own objective in
`planning/allegro_mpc_costs.py`. The bimanual MPC and bigrasp cost definitions
are unchanged. Only the numerical MPC solver infrastructure is shared.

Lambda selects object contact patches, and coupled hand IK converts those
patches into a 22-dimensional target: palm translation, three palm hinges,
and 16 finger joints. The online MPC tracks this target:

```text
stage_cost = ||q_hand - q_IK||²_W_hand
           + w_xy ||object_xy - goal_xy||²
           + w_z  (object_z - goal_z)²
           + w_R/8 ||R_goal.T R_object - I||²_F
           + w_horizontal ||horizontal corridor violation||²
           + ||action||²_W_action
```

The terminal cost uses the same state terms, scaled by
`--mpc-terminal-weight`. There is no fingertip-to-contact-point distance
objective or fingertip-distance synchronization objective. Contact gaps and
Jacobians remain inputs to the contact dynamics.

Default IK tracking weights are 1000 for palm translation, 500 for palm
hinges, and 5000 for finger joints. Adjust them with
`--online-base-position-weight`, `--online-palm-orientation-weight`, and
`--online-posture-weight`. The old `--online-tip-weight` and
`--online-synchronization-weight` options are accepted for compatibility
and do not affect this objective.

By default `--online-guidance mpc` uses MPC throughout the approach.
`--online-guidance ik-mpc` retains the optional initial IK servo approach.
Another bounded startup IK places every pad outside the same frozen
patches, using `--online-pregrasp-clearance` (15 mm). MPC first reaches this
pregrasp pose, then tracks the closing pose; subtracting a constant from
finger angles does not guarantee clearance for the thumb and finger bank.
During approach the object XY target and its soft corridor stay anchored
to the initial position; the Z target holds the observed object height,
and the IK hand target stays fixed in world coordinates. An ungrasped
object's drift or rotation must not keep moving the hand's approach target.
Verified contact advances an independent lift trajectory. An unsafe
contact check pauses this trajectory without resetting the hand downwards.
The IK palm reference translates by its accumulated lift offset. Equivalent Euler
representations are selected near measured palm hinges to avoid a full
rotation at the ±pi boundary.

IK checks hand self-collisions as well as fingertip residuals. A candidate
with more than 0.1 mm of hand self-penetration is rejected even when all
tips reach their targets. Measured hand self-contacts enter MPC as hand-only
normal constraints, without applying a fictitious wrench to the object.

The online state has 79 entries: object and hand pose (29), velocity (28),
and the position-servo target (22). Each action increments the persistent
servo target. A zero action holds that target instead of resetting it to
the measured joint positions and discarding contact preload.

Within 2 cm of pad contact, MPC applies slower bounds: 0.04 m/s for palm
translation, 0.2 rad/s for palm hinges and 0.5 rad/s for finger joints.
Use `--online-contact-base-speed`, `--online-contact-rotation-speed` and
`--online-contact-joint-speed` to change them. The planned pad indentation
is 5 mm (`--squeeze-depth`) and is frozen together with the contact set.
RTI actions and bounded IK correction candidates are compared using the
same nonlinear contact prediction and full MPC objective.

For headless diagnostics:

```bash
python examples/mpc/franka/trigrasp/trigrasp.py --headless --no-video \
  --screenshot-interval 0 --scene-output /tmp/trigrasp_scene.xml \
  --diagnostics-output /tmp/trigrasp_ik_mpc.json
```

The JSON includes `mpc_objective`, `hand_ik_target`, `hand_q`, joint tracking
errors and measured contact verification, including per-finger geometric
gap, height error and normal force, plus hand self-collisions.
Passing objective tests does not
establish complete physical grasp/lift success; inspect the rollout's
`success` field separately.
