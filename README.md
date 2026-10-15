# scsp-robot

Clean subset of the Franka contact / face-detection / manipulation MPC examples.

This tree keeps only:

1. Executable `main` entry points under `examples/mpc/fingertips` and `examples/mpc/franka`
2. Local Python modules those entries import
3. Mesh / XML / URDF / texture assets those modules actually reference

Videos, generated figures, unused object meshes, and unused third-party trees were not copied.

Current tree: 38 entry points, ~317 files, ~91MB. `python tests/test_smoke.py` is the migration completeness check.

## Entry points

- `examples/mpc/fingertips/football/param_detect.py`
- `examples/mpc/fingertips/football/test_humanoid.py`
- `examples/mpc/fingertips/football copy/param_detect.py`
- `examples/mpc/fingertips/test/param_detect.py`
- `examples/mpc/fingertips/test/test_0902.py`
- `examples/mpc/franka/bigrasp/bigrasp.py`
- `examples/mpc/franka/bigrasp/bigrasp_show copy.py`
- `examples/mpc/franka/bigrasp/bigrasp_show.py`
- `examples/mpc/franka/bigrasp/bigrasp_track.py`
- `examples/mpc/franka/football/param_detect.py`
- `examples/mpc/franka/ik2/param_detect.py`
- `examples/mpc/franka/ik2/param_detect_isaac.py`
- `examples/mpc/franka/ik2/test_benchmark_point.py`
- `examples/mpc/franka/ik2/test_mpc_isaac copy.py`
- `examples/mpc/franka/ik2/test_mpc_isaac.py`
- `examples/mpc/franka/ik2/test_mpc_isaac1.py`
- `examples/mpc/franka/ik2/test_mppi_cabinet.py`
- `examples/mpc/franka/ik2/test_mppi_isaac copy.py`
- `examples/mpc/franka/ik2/test_mppi_isaac.py`
- `examples/mpc/franka/ik2/test_mppi_isaac_for_cimpc.py`
- `examples/mpc/franka/mppi/test.py`
- `examples/mpc/franka/mppi2/test_mppi.py`
- `examples/mpc/franka/test_benchmark/ampc/test_ampc.py`
- `examples/mpc/franka/test_benchmark/ampc/test_ampc_ik.py`
- `examples/mpc/franka/test_benchmark/cfmpc/test_cfmpc.py`
- `examples/mpc/franka/test_benchmark/goal_pose.py`
- `examples/mpc/franka/test_benchmark/impc/test_impc.py`
- `examples/mpc/franka/test_benchmark/sampling/test_sampling.py`
- `examples/mpc/franka/test_benchmark/test_scsp.py`
- `examples/mpc/franka/test_benchmark/test_wors.py`
- `examples/mpc/franka/test_benchmark/woRS/test_wors.py`
- `examples/mpc/franka/test_flip_rot/test copy.py`
- `examples/mpc/franka/test_flip_rot/test.py`
- `examples/mpc/franka/test_tilted_push/test_tiltedpush.py`
- `examples/mpc/franka/trigrasp/trigrasp.py`
- `examples/mpc/franka/trigrasp/trigrasp_casadi.py`
- `examples/mpc/franka/trigrasp/trigrasp_show.py`
- `examples/mpc/franka/trigrasp/trigrasp_show_new.py`

Default `--obj` names found on those entries: elephant, foam_brick, football, mug, piggy_bank, rubber_duck, stanford_bunny2, teapot

## Run

Entry points walk up to the `scsp-robot` directory and put it on `sys.path`. You can run them from any cwd without `PYTHONPATH` or `ACADOS_SOURCE_DIR`:

```bash
python examples/mpc/fingertips/test/test_0902.py --headless --trial_num 1
python examples/mpc/franka/ik2/test_mppi_isaac.py --sim-device cuda:0 --mppi-device cuda:0
```

acados is located in-process (`planning/acados_env.py`): `$ACADOS_SOURCE_DIR` if already set, then the local installation `/home/lab423/push_ws/acados`, `../thirdparty/acados`, and `./thirdparty/acados`.

Generated MPC solvers are cached in `~/.cache/scsp-robot/acados` (or under
`$XDG_CACHE_HOME`). Set `SCSP_ACADOS_CACHE_DIR` to override that directory.
A changed cost, dynamics or horizon triggers a new build; subsequent starts
with the same configuration load the existing library. Compilation uses one
compiler process to bound peak memory.

The online Allegro `trigrasp.py` planner shares analytic directional derivative
code across cost Jacobian columns. After loading a compiled solver it removes
generated C/header/object files, retaining the shared library, JSON model and
build log so later starts can still reuse the cache.

Its force tracking residual uses a measured-state linearization refreshed each
control cycle; the dynamics and force diagnostics retain the nonlinear SCM map.
The pad wrench residual is normalized by object weight; necessary tangential
forces and internal preload need not match an arbitrary per-finger force split.
The live allocator imposes the requested wrench as an equality, requests a
balanced preload within its physical force budget, and compensates non-pad hand
contacts. Measured pressure separately certifies the force available now; it
does not freeze the future pressure reference at an already weakening value.
`--online-min-normal-force` defaults to 0.6 N. Table support
is excluded from the hand's load budget. The object goal stays at the requested lift height during
approach and contact recovery. Physical contact count, force closure and bounded
load support enable contact transport toward that goal; startup IK patch errors
are diagnostic only. `target_z` and `lift_gate` in the runtime log show the active
goal and any unmet lift checks. Three sufficiently preloaded physical contacts
can enable lifting if their own bounded certificate passes; use
`--lift-contact-fingers 4` to require all four.

Each MPC cycle also queries the nearest object surface for missing fingers.
Locally reachable replacements must form force closure and support the load
together with the existing physical contacts. Unselected or distant planned
fingers do not contribute to this certificate. Logs show `[local-contact]`,
and `--online-local-contact-distance` sets the search distance (default 2 cm).
A prospective contact becomes a pad target first; it cannot supply a measured
lifting force before contact. Existing pad regions stay fixed in object
coordinates against slip, with indentation regulated by measured pressure.
After loss, a certified local repair gets up to 0.5 simulated seconds before
global recovery opens the pads.

Video recording is off by default. Pass `--save` to `trigrasp.py` to record
an MP4; `--video-output-path` and `--video-output-dir` select its location.
On grasp loss, object displacement/rotation or an unconfirmed contact attempt,
the online controller searches live contact patches, checks a bounded coupled
IK seed, resets the old MPC trajectory/servo pressure, and uses MPC to open the
pads before approaching again. The recovery region follows the measured object;
the airborne object goal is retained. Logs show `[regrasp]`, `regrasp_open` and
`regrasp`; `--regrasp-*` options configure displacement, cooldown and timeout.

BigRasp's online Lambda uses the MuJoCo fingertip/object pair friction by
default, including material priorities. Its Lambda, grouped SCM and online
MuJoCo scene use circular Coulomb friction (MuJoCo `cone="elliptic"`).
Like `ik2/test_mpc_isaac.py`, it uses a small tangential-force penalty
of `0.03` and total-force penalty of `0.01`. These are configurable with
`--lambda-friction-reg-coef` and `--lambda-force-reg-coef`;
`--optimizer-arm-friction` is an explicit experimental override.

BigRasp's `--mpc` and `--mppi` use Lambda to seed the initial approach. Each
subsequent cycle retains measured loaded patches and searches reachable
nearby replacements for missing contacts (default search radius 2 cm).
`--freeze-contact-pair` disables prospective replacements while still updating
actual geometry and forces. Confirmed loss/drift can request a new global
approach guess while retaining loaded arms. The measured state is never reset.
Live force allocation uses the actual lever arms, mass and external wrench,
with bounded circular friction and a positive pressure reserve. Current
measured pressure certifies available load independently of future force targets.
Online MPC prices the resulting force and torque instead of individual
tangential force tracking; its object pose cost remains active. RTI actions
are checked against the nonlinear contact rollout before execution.
Diagnostics under `live_grasp` distinguish full six-dimensional force closure
from positive closure in the available contact span and bounded task support.
Two pure point contacts have wrench rank five: unavailable axial torque is
reported explicitly as `uncontrolled_task_wrench`, never presented as full
force closure. Additional real manifold patches can certify rank six.
The native FA backend accepts four-ray pyramids only, so its generated planner
XML uses an inscribed pyramid with the measured fingertip/object friction.
The execution scene keeps elliptic friction. This adapter also avoids the
old Gaussian-query material overriding the real pair coefficient with 1.0.

Allegro SCM response parameters are exposed as `--soft-scm-response-time`
(`--force-response-time`), `--soft-scm-damping-ratio` and
`--soft-scm-regularization`. Response time defaults to the scene's MuJoCo solref
time scale; damping and regularization use the shared SCM defaults. The online
scene uses elliptic friction and `--contact-impratio 10`, hardening tangential
contact without changing the friction coefficient. The online scene adds three
friction-only NoSlip sweeps to suppress MuJoCo's regularized
steady creep even inside the Coulomb cone. `--contact-noslip-iterations 0`
disables this refinement for contact-model comparisons. The main Newton solver
keeps tolerance 1e-8. These settings do not increase the friction coefficient.
NoSlip is enabled after startup settling so that this execution refinement
does not change the initial object pose used by Lambda and IK. Online IK
retains the 2 mm mean / 3 mm maximum tip-error limits and 0.1 mm self-contact
penetration limit. Its candidate selection prefers a feasible iterate over
a lower-score infeasible iterate; failure logs show the exact blocking limits
and whether the startup search budget expired.
Measured force-closed
contacts enable a force-limited response that couples object inertia and the
finite hand servo, with unilateral forces removed after predicted separation.
This does not
replace its joint-servo dynamics with Franka's Cartesian contact predictor.
`planning/grasp_diagnostics.py` separates measured pad/other-hand/support
wrenches and checks bounded upward wrench capacity independently of the MPC.

See `THIRD_PARTY.md` for what was vendored vs left as an install. Allegro hand assets now live in `thirdparty/spider/`. Isaac Gym, acados, and cuRobo stay as external installs.

### BigRasp DexForge CA/FA

`bigrasp.py --mpc` and `--mppi` update the lambda contact pair on every control
cycle using Warp batched LM. `--mpc` solves FA with Warp Adam;
`--mppi` solves FA with MPPI. The bounded contact-force solve predicts
`x_plus_opt` using the measured object pose, velocity, mass and inertia. That
prediction and the two object-local contacts form a trajectory with one frame.
Before measured bilateral contact, CA/FA track the measured object pose: the
ideal-force lambda prediction cannot lift a target that the pads have not yet
reached. After bilateral contact, translation and rotation of the reference are
limited by `--retarget-object-speed` (0.05 m/s) and
`--retarget-object-angular-speed` (0.5 rad/s). ContactAware's anchor/normal residuals and constrained QP produce a 14-joint
Panda reference. ForceAware receives this reference through its original
`SequenceData` and `make_window` interfaces, and optimizes controls using its
original objective and the ASU IRIS ComFree contact model described below.
CA is a contact-aware two-arm IK solve with 14 joint angles, fixed object pose,
surface anchors, pad normals, joint limits and collision constraints. The single
reference frame is held across the FA horizon; the horizon remains configurable.

Use a Python environment with the dependencies in `requirements-spider-mjwp.txt`
and the mesh-loading dependencies for BigRasp. Set `DEXFORGE_ROOT` if DexForge
is not at `../GraspSONIC/thirdparty/DexForge`:

```bash
DEXFORGE_ROOT=/home/zz/GraspSONIC/thirdparty/DexForge \
python examples/mpc/franka/bigrasp/bigrasp.py --mpc --no-visualize --ik-backend mujoco
```

`--retarget-ca-iters` controls the CA QP iteration limit; the existing
`--adam-horizon`, `--adam-iters`, `--adam-lr` and `--adam-knot-substeps` configure
FA. `--adam-multistart-count` (2) retains the shifted warm trajectory and also
optimizes a fresh CA reference. Both starts use the same measured state, executed
command boundary, bounds and FA objective; the optimizer selects the best valid
result. This helps prevent a changed contact pair trapping Adam in its previous
local solution. The executor applies FA's absolute joint targets at each physics step.
Returned telemetry includes the raw `lambda_x_plus_opt`, the actual reference
`x_plus_opt`, the `approach`/`manipulate` phase, CA contact error and the one-frame
hand/object reference under `retargeting`.

`--mppi` uses that same CA reference, object/contact losses and physical rollout.
It samples additive angular control offsets around the warm start, updates the
mean using exponentially weighted trajectory costs, and evaluates the updated
mean before executing its controls. Configure FA sampling with `--mppi-samples`,
`--mppi-iterations`, `--mppi-init-iterations`, `--mppi-lambda`,
`--mppi-noise-sigma` (radians), `--mppi-noise-decay` and `--mppi-seed`.
For FA, MPPI weights use the loss multiplied by DexForge's gradient
normalization (`1e-6`); `--mppi-lambda` is expressed in those normalized units.
Reported costs retain the original FA units. This prevents the default
temperature from collapsing the weights to a single sample on large mm-based
contact costs.
`--adam-horizon` and `--adam-knot-substeps` set the FA time grid for both solvers.
The FA sample count honors `--mppi-samples` independently of the contact search.

Lambda preprocesses the original mesh once using area sampling, topology-aware
surface-normal variation and fingertip-sized patch area. This excludes sharp
tips, thin ridges and patches too small for the pad. For elephant,
`--lambda-contact-region auto` additionally keeps the broad transverse sides of
the central body. The longitudinal axis is inferred from the original mesh;
`--lambda-body-core-fraction` (0.5) excludes the head and tail regions. Surface
normals must face laterally, so a rounded head or tail root cannot pass merely
because it has low curvature. `--lambda-contact-region all` selects the generic
patch policy; `body-sides` explicitly enables the body restriction. It evaluates
all admissible pairs in the finite candidate pool once to select the offline reference, then
searches each arm's tip-to-geometric-centre cone at every control cycle. Pair
filters enforce pad/support clearance, facing normals and friction-compatible
antipodality. Online candidates and the offline reference are evaluated with the
same current pose, velocity, inertia and external wrench. Quality is measured as
`C(current, target) - C(predicted, target)`, excluding force regularization,
following the Isaac cost-reduction criterion. Within the quality band, the live
selector prefers smaller summed fingertip travel; a small switching margin
retains an eligible incumbent. Empty cones or insufficient quality explicitly
fall back to the offline pair.

`--lambda-lm-iters` (20), `--lambda-cone-half-angle` (45 degrees),
`--lambda-max-per-arm` (32), `--lambda-quality-fraction` (0.015) and
`--lambda-switch-margin` (0.002 m) configure this policy. The quality margin is
relative to the best reduction with a tiny current-cost floor, avoiding the
Isaac script's absolute tolerance overwhelming BigRasp's smaller cost units.
`--sample-num` sets the low-curvature candidate count and the existing curvature
threshold options control patch acceptance. Offline mesh preprocessing runs on
the CPU; live filtering, dynamics, LM and pair selection run on Warp. Only
simulator input/output and selected-result transfers cross the device boundary.
Telemetry `lambda_prediction` includes pair IDs, patch curvature, quality
reference/margin, selection source and offline/online solve counts.

CA and FA share mesh-derived sphere clusters for the forearms, wrists and
flanges, including their body-local offsets. This prevents references that are
reachable by the pads but blocked by physical wrist-to-wrist collision. CA also
accounts for the object GS surface inflation when setting pad clearance, so the
physical pad can reach the original-mesh anchor.

The Panda adapter uses angular scales for all 14 joints and maps guidance only
to fingertip-object collision outputs. The active original FA terms include
hand/object reference tracking, marked-query tracking, marked-contact depth
robustness, penetration and inter-arm collision. Both solvers share these terms
and the original smoothing kernels. They penalize control-knot velocity and
acceleration, plus simulated joint velocity and acceleration; there is no jerk
term. Panda uses uniform radian scales and weights rather than DexForge's
virtual-wrist/finger groups, so the numeric objective is an arm adaptation.

The default Adam learning rate now matches DexForge (`0.0015`, decaying to
`1e-5`). Replanning preserves the last executed setpoint as the interpolation
boundary, and rebases shifted absolute warm-start knots against each newly
solved CA reference. `planner_diagnostics.cost_components` reports the original
trajectory cost, each smoothing component, and separate hand/object reference,
marked-query, contact-depth, penetration and inter-arm terms for the selected
physical plan. `--print-mppi-timing` prints these terms and MPPI mean/selected
costs and effective sample count. The first 20 CA/FA cycles always log the
measured tip heights, phase and contact-query error; `ee` in these branches is
the FA query-to-lambda-anchor distance, not the legacy stand-off waypoint error.

With the default FA weights, each reference stage contributes
`1e6 * (0.03^2 * ||q-q_CA||^2 + 100*(||p-p_ref||^2 + (0.3*angle)^2))`;
the terminal reference is multiplied by two. Marked-query attraction contributes
`sigmoid((distance_mm-10)/1) * ||query-anchor||_mm^2`. Contact-depth robustness
has weight 40 and target gap -4 mm. Penetration weights are 15/75 and inter-arm
collision weight is 100 with a 2 mm surface margin. Depth robustness alone is
flat outside the collision activation range; marked-query tracking supplies
the long-range attraction. Control velocity/acceleration and simulated joint
velocity/acceleration supply smoothing, rather than a contact-force target.

The native Euler integration includes Panda's affine servo velocity derivative
in its implicit damping matrix, including gear scaling and zero derivative for
torque-saturated actuators. The original actuator force law, force limits and
execution `implicitfast` integrator remain intact. Both retargeting branches
default to 10 ms physics and 60 ms control knots (`H=5`, `K=6`: 30 dense steps).
`--mujoco-dt` and `--adam-knot-substeps` override these values; setting only the
physics timestep adapts the default substep count to preserve 60 ms knots.

The 625 inter-arm sphere-pair penalties use parallel Warp threads with the
original FA cost and VJP. Default CA/FA rollouts simulate arm/object contact
and object/floor support in one collision batch. Cross-arm avoidance remains
in CA constraints and the complete FA pair cost; physical execution retains
mutual arm collisions. `--retarget-full-contact-dynamics` additionally simulates
arm/arm and arm/floor GS contact reactions (nine batches, substantially slower).
On an RTX 5060 Ti, warmed default Adam (40 iterations, two starts) measured
0.89–0.94 s for FA and 0.94–1.27 s including CA, versus 37–45 s with the former 300-step
serial-pair implementation. Full contact dynamics measured 1.39 s for FA.
First-use JIT compilation and graph construction are additional startup work;
these timings do not establish a 60 ms real-time controller.
`retargeting.ca_time`, `retargeting.fa_time` and FA diagnostics expose stage
timings, dense steps, collision batches and pair count. Linear interpolation
and soft smoothing costs do not guarantee velocity/acceleration limits or
continuously differentiable commands.

FA MPPI captures both its sampling iterations and its final mean/fallback
evaluation in reusable CUDA graphs. The final mean and retained sampled
trajectories compete under the same complete FA cost, including smoothing.
A finite mean is not automatically preferred: nonlinear contact dynamics can
make an average lose the contact achieved by a lower-cost sampled trajectory.
Zero inter-arm penalties skip atomic
updates; all positive penalties retain the same loss and gradient. CA evaluates
its sphere constraints in arrays and uses only MuJoCo kinematics/Jacobians.
For equal-radius object GS it uses exact nearest-centre queries (with stable
ties); unequal radii retain the full nearest-gap search. Its QP omits only rows
proved redundant throughout the complete trust/joint-limit box, while nonlinear
line search still checks all collision geometry.
On the same GPU, default warmed MPPI (256 samples, four iterations) measured
64–72 ms for FA and 78–149 ms including CA, versus 111–117 ms and 160–402 ms
before these changes. The first solve also compiles/captures graphs and uses
eight sampling iterations. These measurements do not guarantee a 60 ms
control deadline or completion of the lift task.


### ASU IRIS ComFree model

Lambda LM, FA Adam and FA MPPI now use the contact law from
[`asu-iris/comfree_warp`](https://github.com/asu-iris/comfree_warp/tree/ba8b9969f5f7859d51b1f738f458f8d2f49be080).
For each active collision/limit row:

```text
v_smooth = v + h M^-1 f_smooth
u = J v_smooth
f_row = max(D * (-(damping/h)*u - (stiffness/h)*(h*u + distance)), 0)
v_next = v + h M^-1 (f_smooth + J^T f_row)
```

The native adapter retains differentiable GS collision, native rigid-body
mass/bias/actuation, implicit servo damping, and CUDA graphs. It replaces
DexForge's iterative contact response with the upstream explicit row law and
its gathered VJP. Impedance uses the upstream fixed 1 cm width; pyramidal row
masses use the bodies' translational inverse weights without MuJoCo's
friction/impratio multiplier. Signed distances include the collision/limit
margin, matching upstream `efc_dist`. Softplus is no longer used for physical
contact force. The upstream package disables backward kernels and expects
MuJoCo 3.6; this adapter works with the existing MuJoCo installation. It does
not install a second conflicting `comfree_warp` package or modify DexForge.

BigRasp uses `--comfree-stiffness 0.2 --comfree-damping 0.001` for all three
planners. These are dimensionless upstream parameters, with the division by
physics timestep shown above. `--contact-stiffness` remains the legacy
ideal-contact compliance setting. Lambda still optimizes bounded arm wrenches;
these are applied forces in the smooth rigid-body prediction. Its reduced
support contact uses the same ComFree law as a frictionless plane row. Native
FA rollouts resolve the arm/object and floor friction pyramids from GS geometry.
The current planner scene retains a legacy approximation: gravity is disabled
and its native support plane is the floor at z=0, whereas the default execution
pedestal top is at z=0.32 m. The raised box pedestal is not a native GS support
contact. Consequently, using the same ComFree row law does not yet make the
predicted object motion match the execution support dynamics; object reference
penalties can dominate contact attraction. Approach/contact checks do not
establish sustained contact or successful lifting under this approximation.

Execution-side tabletop support uses four body-local convex mesh pieces
(XY quadrants, with a whole-hull fallback for sparse meshes). They are
massless collision geoms and collide only with the finite pedestal; arms and
floor still contact the original object mesh. This prevents the single convex
mesh manifold from repeatedly switching between elephant feet. CA/FA execution
also enables three MuJoCo NoSlip iterations with `impratio=10` to suppress
soft-friction drift. `--contact-noslip-iterations` and `--contact-impratio`
override these execution settings for comparisons. They apply to all contacts
and do not change the material friction coefficients.
The Warp planner explicitly clears this execution-only option, which its loader
does not support, and retains the ComFree law. This is a support geometry
approximation, not a full convex decomposition of the original nonconvex mesh.
The six-second, no-arm-contact regression checks stationary pose/velocity,
unchanged mass and inertia, and free-body response to an applied force and
gravity; it does not freeze the object or reset its velocity during execution.

`planning/mlqp_point.py` uses the same ComFree prediction in IPOPT, acados,
Torch single-contact and Torch batched contact solves, including the incoming
velocity. Its `contact_stiffness` default is now 0.2, and `comfree_damping` is
0.001. Pass actual environment gaps with
`update_Jacobian(J_env, phi=phi_env)` (one gap per contact or per row).
Omitting gaps preserves the compatibility assumption that the supplied active
rows lie on the surface. `compute_env_diag_inverse` retains its old block API
but now returns row masses. Provide physical `obj_inertia` for physical
predictions; the historical conditioned fallback is still available.

`models/explicit_model.py` supplies CasADi ComFree dynamics and normalized
quaternion integration. `Q`, or optional `comfree_mass_matrix`, is interpreted
as a generalized mass matrix; the `sigma` input is now dimensionless stiffness.
`step_state_fn(q, v, cmd, phi, J, sigma)` returns `(q_next, v_next)`.
The existing five-input `step_once_fn` remains the zero-initial-velocity
compatibility interface; existing position-only MPC consumers retain that
assumption. The Cartesian Isaac wrapper and its Torch fallback share this law.
Generated acados model names include the new dynamics to prevent stale code
from being reused. The system MuJoCo-Warp rollout also uses upstream contact
row masses and impedance width.

Validation includes the unmodified upstream force kernel, CasADi/Torch/NumPy
velocity and derivative parity, and a native active-contact finite-difference
check. With the new law, warmed default MPPI FA measured 51–54 ms and default
Adam control cycles approximately 0.8–0.9 s on the RTX 5060 Ti. Headless elephant
runs reached three consecutive cycles of bilateral physical contact (MPPI seed
0: 14 cycles; Adam seed 7: 38 cycles). These runs verify approach/contact,
not completion of the lift.

## Smoke test

```bash
python tests/test_smoke.py
```


cd /home/zz/scsp-robot/tro_revision
pdflatex -interaction=nonstopmode -file-line-error root.tex
bibtex root
pdflatex -interaction=nonstopmode -file-line-error root.tex
pdflatex -interaction=nonstopmode -file-line-error root.tex
