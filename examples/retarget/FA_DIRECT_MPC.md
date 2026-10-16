# Direct ComFree MPC experiment

This is an experimental multiple-shooting OCP, separate from the earlier
`fa_hand_acados.py` condensed control QP. It is not the default controller.

## Formulation

- Each OCP node contains native qpos/qvel and two servo-target history states:
  33 + 32 + 26 + 26 = 117 states, with 26 servo-target controls.
- Thirty 10 ms dynamics stages retain the original 300 ms prediction duration.
  The original FA control parameterization has five interpolated knots; this
  experiment has thirty dense controls. Its discrete control smoothing grid
  therefore also differs, although the physical scales and weights are reused.
- `disc_dyn_expr` calls the actual nonlinear native ComFree stage through a
  CasADi external-function ABI. On a cache miss it evaluates native physics at
  the supplied x/u; it is not an affine model supplied as dynamics parameters.
  The wrapper still invokes Python/Warp, rather than compiling the full native
  simulator into C. This overhead is included in wall time.
- Native reverse-mode basis batches supply the dynamics Jacobian, including
  state-dependent contact normals/frames on the current hard-contact branch.
  FA residual Jacobians use native observation/geometry adjoints and the same
  stopped auxiliary sigmoid weights as the original FA objective. There are no
  finite differences in the optimizer or the validation program.
- Individual FA geometry residuals, symbolic tracking residuals, and physical
  control/state smoothing enter Gauss–Newton. An instance-local code generator
  omits unused exact residual-Hessian functions: the installed acados template
  otherwise attempts second derivatives even in Gauss–Newton mode.
- Each control cycle performs the requested number of SQP/RTI updates. A local
  control trust box limits each proposal to 5 mm / 0.05 rad / 0.25 rad for base
  position / base rotation / finger coordinates. Each proposal is checked on a
  dynamically feasible nonlinear rollout, with up to four backtracking steps.
  Cost and rollout checks are included in full-cycle timing.

Prediction/execution contact-force traces can be recorded with `--diagnostics`
and restricted using `--trace-start` / `--trace-end`. They record accepted
nonlinear forecasts, per-contact forces and positions, cost components, and
MuJoCo's individual execution substeps. `--line-search-steps 8` and
`--freeze-templates-after 2.4` are diagnostic ablations; default backtracking
remains four trials. See the [slip investigation](../../outputs/fa_slip_diagnosis_20261016/report.md)
for the verified baseline comparisons, failed fixes and template-update issue.

The dynamics retain the FA runtime's native contact implementation and coupled
corrections. This does not substitute the reduced constant-mass closed formula
in `models/explicit_model.py`. Thus this experiment does not measure the speed
of that reduced formula compiled directly into an acados C solver.

## Physical contacts and slowly updated targets

The physical contact locations, distances and frames come from the native
querypoint/SDF narrowphase at each predicted state. Neither a MANO contact
template nor an ADMM candidate point/force is supplied as a physical contact or
an applied force to this dynamics function. The querypoint/SDF collision proxy
is the existing FA geometry, not MuJoCo's mesh collision geometry.

The MANO/CA hand trajectory remains a posture reference. Its object-side contact
targets initialize a separate, slower template planner:

1. Refresh at most once every five 30 ms control windows, i.e. every 150 ms.
2. Run two point/force ADMM updates, with analytic ComFree batch LM for forces.
3. Add `contact_template_weight * sum(mask * ||1000*(z-template)||^2)` to
   candidate ranking; the default weight is 5. The anchor is the currently held
   template, not the original MANO template after every refresh.
4. Commit the selected combination only after a positive coordinate merit
   improvement and a material point displacement. Equal scores retain the
   incumbent by deterministic candidate ordering. Hold committed object-local
   points between refreshes, matched by hand-body/material-query identity.

The positive displacement penalty discourages template drift and switching;
it does not prove a unique global minimizer of the nonconvex contact problem.
Only real servo controls are executed. Ideal forces from the object-side
planner are diagnostics and are never injected into MPC physics or execution.

The matched investigation found a scoring/commit mismatch in the legacy
template planner: search scores a time-varying point sequence, but committing
the earliest proposal per material query can change that sequence by over
10 mm. The experimental `--consistent-template-commit` switch expands each
candidate into the material-point map that will actually be held, then scores
that exact sequence. Its incumbent is the actual held template. Every update
asserts that the scored and committed active points coincide. The default is
kept unchanged for controlled ablation. On this trajectory, with post-step
tracking, the switch reduced mean position error from 16.08 to 11.78 mm and
avoided the 5.31 s drop through the 5.4 s endpoint; substantial pose drift
remains. This is not evidence of indefinitely stable holding.

`--relative-warm` shifts the previous reference-relative correction and adds
the new dense reference, rather than carrying absolute controls forward. In
the post-step/fixed-template Direct test, this reduced mean position error
from 11.06 to 6.89 mm and mean rotation error from 7.81 to 5.11 degrees. The
default branch's first 0.6 s remains bitwise identical. Increasing only that
test's RTI cap from three to eight made the result worse (15.81 mm, a drop at
5.18 s), despite successful QPs. Thus the earlier fragmented-L-BFGS budget
failure is not a sufficient explanation of Direct's failure. Constant LM
changes alone also failed: both 1e-4 and 1 were worse than the default 0.01.

## Validation and evidence

`validate_fa_direct_mpc.py` checks native forward prediction, external Jacobian
transport, a separately seeded native VJP, FA residual value/gradient agreement,
and full tracking/geometry cost agreement. It also verifies bitwise unchanged
dynamics after changing contact targets, with a corresponding cost change.
For a completed rollout it checks accepted nonlinear cost descent and the
low-frequency template-update schedule.

A CUDA graph must retain all allocation owners. The implementation keeps its
`StepInput`, applied-force/clock/softness arrays, outputs, cotangents and every
workspace alive. The validation intentionally interleaves native executor and
Jacobian calls, which exposed a missing allocation owner during development.
Those earlier trajectories have `INVALIDATED.json` markers and must not be
used for tracking or timing conclusions.

The current experiment report, raw arrays, templates, validation results and
comparison plot live under `outputs/fa_direct_mpc_20261016/`. Historical FA
tracking is evaluated on the same demonstration clock, but it is not a matched
solver-budget comparison because the control parameterization differs.

The matched investigation under `outputs/fa_matched_ablation_20261016/` found
an additional temporal difference: upstream FA tracks the **post-step qpos**,
while its geometry comes from the **pre-step body/contact observation**. The
original direct-OCP experiment tracked the pre-step qpos as well. The explicit
`--post-step-tracking` ablation restores FA's tracking timing, with an inline
analytic expression and a separate generated-solver cache. Do not infer full
trajectory-objective equivalence from the earlier single-state residual tests.
Also, this SDF scene already differentiates contact normals in upstream FA;
the full-contact-gradient switch is a no-op here. The upstream stop policy
only applies to hard Gaussian/SPHERE_TARGET contacts.

The [matched ablation report](../../outputs/fa_matched_ablation_20261016/report.md)
separates complete closed loops, common-state optimization probes, analytic
gradient checks and isolated timing. Its baseline reproduces the successful
FA qpos, qvel and executed controls bit for bit. Reducing the old wrapper to
three iterations distributes them across blocks as `[1, 1, 1, 0, 0]`, resetting
L-BFGS history on every update. A separate `[3, 0, 0, 0, 0]` experiment preserves
history within those three updates; these are materially different budgets.
This finding concerns that wrapper and must not be generalized to all
three-step SQP/Gauss–Newton methods. Likewise, the temporal mismatch is a real
objective difference, but the matched FA timing ablation alone did not cause
a drop on this trajectory.

## Run

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
conda run --no-capture-output -n pi0 python -m examples.retarget.run_fa_contact_mpc \
  --solver direct-mpc --executor mujoco --duration 5.4 --rti-steps 3 \
  --template-period 5 --template-weight 5 \
  --output outputs/fa_direct_mpc_new_case

# The standalone entry point permits fixed templates by omitting --contact-admm.
conda run --no-capture-output -n pi0 python -m examples.retarget.validate_fa_direct_mpc \
  --rollout-dir outputs/fa_direct_mpc_new_case
```

Use a fresh output directory for each run. All 10 ms executed states, commands,
common-clock references, complete MPC timing, QP/NLP statuses and template
refresh snapshots are saved; initialization/JIT are excluded from warm timing.
