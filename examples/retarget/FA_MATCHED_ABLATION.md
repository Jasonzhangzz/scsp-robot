# Successful-FA matched ablation

The study uses `outputs/fa_admm_solver_20261016/comfree/lbfgs_fixed` as its
successful reference. Its unchanged worker reproduces all qpos, qvel and
executed controls bit for bit. The measured trajectory is 5.4 s, including
the hold after the MANO demonstration ends. Physics advances on the fixed
simulation clock, independently of optimizer wall time.

Results and evidence:

- [Report](../../outputs/fa_matched_ablation_20261016/report.md),
  [machine-readable results](../../outputs/fa_matched_ablation_20261016/results.json),
  [closed-loop curves](../../outputs/fa_matched_ablation_20261016/closed_loop_ablation.png).
- `protocol.md` defines factors and excluded experiments. Each worker saves
  `config.json`, `audit.json`, `admm_windows.jsonl`, `rollout.npz`, and timing.
  Failed runs may stop after the first 80 mm height deficit; compare common
  intervals rather than means over different durations.
- `embedding_validation.json` checks five-knot interpolation embedded in the
  thirty-control space, retaining the original endpoint smoothing functional.
  Its warm check verifies `P * shift_coarse * endpoint_select`; new interior
  directions restart from that endpoint embedding on each control window.
- `tracking_time_validation.json` checks the literal pre/post-step objective
  difference. `objective_discretization_audit.json` decomposes that difference
  on identical successful controls with bitwise identical predicted states.
- `full_horizon_gradient_validation.json` compares two independent analytic
  adjoints for the full tracking/geometry objective. Different smoothing
  definitions are disabled on both sides for that check.
- `timing_serial/probes/*` contains isolated common-state wall measurements.
  Paused/concurrent closed-loop timings must not be used for speed rankings.
- `condensed_qp_validation.json` checks the 130-dimensional acados local QP
  against analytic unconstrained solutions and projected box KKT conditions.
- `contact_comparison.json` and `contact_forces.csv` reconstruct physical
  per-contact forces; the next state is checked against the saved trajectory.
- `template_commit_discrepancy.json` audits the scored versus committed point
  sequence. `audit_fa_template_commit.py` asserts agreement for the corrected
  expansion. The legacy default's unchanged 0.6 s trajectory is checked in
  `smoke/default_template_unchanged/unchanged_default_validation.json`.

Use the `pi0` environment, with one BLAS/OpenMP thread. Output directories
must be fresh. All optimizer and validation derivatives are analytic.

```bash
PY=/home/zz/miniconda3/envs/pi0/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# One-factor closed loops. Only run one GPU worker at a time for timing.
$PY -m examples.retarget.run_fa_matched_sweep \
  --output outputs/fa_matched_new --mode closed-loop \
  --factors baseline budget3 single-start slsqp acados tracking-time \
    dense-smoothing control-bounds absolute-warm full-contact-jacobian \
    slow-templates dense-controls

# Same three updates, continuous history rather than three one-step blocks.
$PY -m examples.retarget.matrix_fa_worker \
  --experiment outputs/fa_admm_solver_20261016 \
  --forge thirdparty/retarget_assets/DexForge --executor comfree \
  --variant lbfgs --iterations 3 --multistarts 8 --duration 5.4 \
  --hand-single-block --case-name budget3-contiguous \
  --output-base outputs/fa_matched_new

# Common successful-state optimizations, not closed-loop success tests.
$PY -m examples.retarget.run_fa_matched_sweep \
  --output outputs/fa_matched_new --mode probes \
  --factors baseline budget3 budget3-contiguous single-start slsqp acados \
    tracking-time dense-smoothing control-bounds absolute-warm \
    full-contact-jacobian slow-templates dense-controls

$PY -m examples.retarget.report_fa_matched_ablation outputs/fa_matched_new

# Combined Direct corrections, retaining low-frequency contact ADMM.
# The measured study had 6.59 mm / 4.95 deg mean error over 5.4 s;
# this is still less accurate than the successful FA baseline.
$PY -m examples.retarget.run_fa_direct_mpc --executor comfree \
  --post-step-tracking --relative-warm --contact-admm \
  --consistent-template-commit --rti-steps 3 --duration 5.4 \
  --output outputs/fa_direct_corrected_new
```

`acados` in this matched sweep changes the original 130-dimensional control
optimizer to a BFGS/HPIPM control QP while retaining the native FA/Warp oracle.
It is distinct from the thirty-stage direct multiple-shooting OCP and from
fully compiling reduced ComFree dynamics into C. The two acados experiments
answer different questions.

The earlier Direct experiment changed several settings together:

| Setting | Successful FA | Earlier Direct OCP | Matched test |
|---|---|---|---|
| Control variables | Five interpolated 26-dimensional knots | Thirty independent controls | `dense-controls` retains old smoothing |
| Initial guesses | Eight candidates, including velocity-prior family | One shifted sequence | `single-start` |
| Update budget | Forty updates, five blocks | Up to three RTI proposals | `budget3`, plus continuous-history control |
| Control smoothing | Five endpoints, 60 ms intervals | Thirty samples, 10 ms intervals | `dense-smoothing` retains old variables |
| Cross-window warm start | Shift reference-relative correction | Shift absolute controls | `absolute-warm` |
| Control bounds | Clip inside native physics | Also constrain optimizer controls | `control-bounds` |
| Contact target | Fixed MANO/CA template | Low-frequency anchored point search | `slow-templates`, plus Direct fixed-template inverse test |
| Pose tracking time | Post-step | Pre-step | `tracking-time`, plus Direct `--post-step-tracking` |
| Curvature model | L-BFGS | Residual Gauss–Newton | **Not separately isolated**; matched control-QP acados uses BFGS |
| Nonlinear backtracking | Up to eight trials | Up to four trials | Earlier Direct inverse used eight; see report |
| SDF normal/frame gradient | Enabled | Enabled | Audited no-op |

Do not interpret the three-update L-BFGS result as an isolated test of three
RTI steps. Do not infer that the BFGS control-QP result validates the Direct
Gauss–Newton formulation. These comparisons establish which changes can
produce failure alone and which differences still involve interactions.

The contact-gradient factor audits the compiled collision policy first. In
this SDF scene the original FA already propagates the normal/frame gradient,
so that factor records a no-op and does not run a redundant full trajectory.
