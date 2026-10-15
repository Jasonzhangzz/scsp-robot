# MANO → Wuji semantic retarget

`retarget.py` runs semantic geometric IK, persistent MANO contact attractions, and
closed-loop material-point MPC in MuJoCo. The initial example is DexForge's
prepared right-hand DexYCB Sugar Box sequence. The wrist has six actuated
coordinates and the hand has twenty actuated joints. Only the hand's position
servos act on the object; Lambda forces are planning witnesses and diagnostics.
They are not inputs to the online MPC.

The live viewer now overlays a **translucent orange MANO mesh** and a
**translucent cyan demonstration object** on the physical Wuji scene. Ghosts
are compiled into a separate display model with collision disabled; they never
enter the execution model, contact solver, MPC previews or force certificates.
The solid Wuji and textured object interact through the original MuJoCo contacts.

```bash
cd /home/zz/scsp-robot
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/zz/miniconda3/envs/pi0/bin/python examples/retarget/retarget.py \
  --sequence thirdparty/retarget_assets/dexycb_prepared/20200709_142802 \
  --lambda-mode off --max-steps 600 \
  --output outputs/wuji_live_mano
```

Lambda is disabled by default. Every demonstrated contact remains a soft MPC
target even when its exact location is unreachable; no force/IK solve removes
the target or changes its contact mask. MPC evaluates nearby projection controls
online. A failed nearby contact or recovery continues to aim at the MANO anchor.
The geometric IK trajectory is a posture prior, not a contact acceptance test.

Legacy `--lambda-mode offline` and `online` remain available for comparison.
With offline mode, plans are
saved to `offline_contact_plans.json`; `--contact-plan-cache PATH` reuses them
only when the scene, trajectory and preparation settings match. They are nominal
contact/IK seeds, not certificates of the current physical grasp. `--lambda-mode
online` keeps the previous measured-state Lambda replanning option. Neither mode
supplies Lambda forces or a wrench target to MPC.

MPC reads the measured hand/object state each cycle, predicts contact dynamics,
checks native MuJoCo horizon costs and collisions, and executes one position
command. It retains MANO contact regions and closes 4 mm inward along their
normals. Finger kp is scaled by 50, damping by sqrt(50), and both finger torque
clamps by 10. Lambda and MPC share a 100 N contact force cap. These defaults are
configurable with `--closure-depth`, `--finger-stiffness-scale`,
`--finger-force-scale` and `--force-limit` (also in `default.yaml`).

Force-closure and load certificates are logged for diagnosis; they do not gate
transport or rank control candidates. Static load allocation and force-error
pressure feedback are disabled. Recovery is off by default; `--regrasp` enables
one-finger recovery based on separation and MANO correspondence, without force
judgements. Position commands alone drive the object.

Ghosts follow the tracked reference by default, pausing during recovery.
`--demo-playback time` instead plays the demonstration at simulation time.
`--demo-alpha 0.3` adjusts transparency; `--no-demo-overlay` hides ghosts and
`--regrasp` enables optional recovery. `--headless --video` records the
same overlay. The controller is online feedback, but current solve times are
slower than a 30 Hz wall-clock control loop.
Implementation details and validation scope: [online execution](ONLINE_EXECUTION.md).

Current cost design and checks: [soft-cost validation](SOFT_COST_VALIDATION.md).
Current contact-only MPC architecture (no external wrench target):
[contact/pose MPC validation](CONTACT_POSE_MPC_VALIDATION.md).
[Load-support diagnosis against trigrasp](LOAD_SUPPORT_DIAGNOSIS.md) explains
the measured preload, material-point and actuator-allocation failures.
Current distance-control changes and validation: [grip control fix](GRIP_CONTROL_FIX.md).
`--physics-reference PATH_TO_DEXFORGE_SCENE.xml` copies its object inertia,
contact coefficients and floor while preserving native MuJoCo mesh geometry.
The generated scene and `physics_parameters.json` record the exact parameters.
Use `--finger-stiffness-scale 1 --finger-force-scale 1 --sim-dt .01` to compare
with the unscaled Wuji servos and executor step of the bundled DexForge case.
`export_dexforge_contacts.py` exports native ComFree forces and its MuJoCo replay;
`compare_contact_trajectories.py` compares object-local points and recomputes GWS
margin with the same force normalization, object radius and friction for all runs.
DexForge contact, force-closure and control differences, including the limits of
the Sharpa comparison: [DexForge comparison](DEXFORGE_COMPARISON.md).
Earlier native closure and wrist tests: [force-closure validation](FORCE_CLOSURE_VALIDATION.md).

Use a Python 3.11 environment with NumPy, SciPy, MuJoCo, CasADi, Warp, trimesh,
rtree, PyYAML and pytest. Loading legacy MANO pickles also requires chumpy.
The tested local interpreter is `/home/zz/miniconda3/envs/pi0/bin/python`.
The shared acados locator finds the installed tree automatically. IPOPT is
available with `--mpc-solver ipopt`; acados errors are reported without silently
switching solver backends.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/zz/miniconda3/envs/pi0/bin/python examples/retarget/retarget.py \
  --sync-assets --headless --compare --config examples/retarget/default.yaml \
  --output outputs/wuji_retarget
```

The optional `--sync-assets` copies the authorized SSH example, Wuji assets,
query points, required right-hand meshes and current DexForge MANO decoder into
`thirdparty/retarget_assets`. Subsequent runs omit this flag. The input sequence
and upstream repository are not modified. MANO model files are not copied; use
`--mano-models /path/to/mano` to select models already available to you.
Use `--asset-cache`, `--dexforge-root`, `--hand-assets` and `--sequence` to
configure local dependencies. V1 accepts prepared DexYCB right-hand inputs.

The original GraspSonic dataset can be tested directly. This downloads labels,
calibration, the initial depth frame and the grasped object mesh, then runs
both modes and writes trajectory plots plus aggregate metrics:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/zz/miniconda3/envs/pi0/bin/python examples/retarget/benchmark.py \
  --sync --sequences 20200709_142802 20200709_142553 \
  --output outputs/wuji_dexycb_benchmark
```

`dexycb_data.py` also imports one sequence independently. The grasped object
comes from `ycb_grasp_ind`, rather than whichever object moves most. Calibration
and MANO shape are loaded from that subject's metadata. The initial depth plane
supporting the object defines world up; background walls are excluded by the
object support test. Both hand and object receive the same rigid transform.
Other object types require a matching `--scene-template` from DexForge, so mass,
inertia and contact coefficients are never guessed from a mesh.

```bash
# Prepare geometric IK only (full mode also prepares offline Lambda).
/home/zz/miniconda3/envs/pi0/bin/python examples/retarget/retarget.py \
  --headless --mode ik-only --prepare-only --output outputs/wuji_geometry

# Short deterministic CPU validation, including both physical rollouts.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/zz/miniconda3/envs/pi0/bin/python examples/retarget/retarget.py \
  --headless --compare --subsample 4 --horizon 2 --mpc-solver ipopt \
  --max-steps 100 --output outputs/wuji_comparison

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/zz/miniconda3/envs/pi0/bin/python -m pytest tests/test_wuji_retarget.py -q
```

`--mode ik-only` runs the geometric position-servo baseline. `--compare` runs
both modes from identical initial states and scene parameters. Drop `--headless`
for an interactive viewer, or add `--video` for `rollout.mp4` (requires imageio
and ffmpeg; use `MUJOCO_GL=egl` for headless rendering).

## Contact and selection contracts

- Internal poses are `[x, y, z, qw, qx, qy, qz]`, lengths are meters, and angles
  are radians. Source `[qx, qy, qz, qw, x, y, z]` poses are explicitly converted.
  One common rigid correction places the initial object on its nearest stable
  supporting face while preserving yaw and hand/object relative geometry. The
  correction is saved in `ground_stabilization.json`.
- MANO skinning weights identify palm, each finger, and proximal/middle/distal
  regions (distal includes fingertip). Surface-facing tests, distance hysteresis
  and stable material vertices produce contact episodes. Each episode maps only
  to the same Wuji finger and anatomical region. Multiple episodes/regions of a
  finger are supported. The original episode material point stays fixed as a semantic prior. Contact IK
  can select a physical patch within 30 mm on the same body/region; both original
  and execution material points remain in the MPC objective and diagnostics.
  Palm and finger descriptors use longitudinal, palmar and thumb-side axes,
  rather than treating MANO's global axes as Wuji's body axes. Palmar normal
  coordinates remain continuous, including lateral contacts.
- IK jointly solves all 26 coordinates with analytic point/normal Jacobians,
  joint bounds and mesh collision penalties. Diagnostics explicitly mark
  configurations that fail reachability or collision validation. Geometric pose
  tolerance is 12 mm; the separate contact IK tolerance defaults to 2 mm.
  Every valid decoded input frame is solved, including frames with no contact.
  The 21 joint landmarks preserve MANO phalange directions using Wuji bone
  lengths and palm layout. Hand-shape fitting is primary; semantic surface
  positions remain soft preferences for a different hand morphology. Reported
  mesh collision gradients are checked numerically before IK uses their sign.
  `geometric_pose_valid` checks finite joints and collisions; shape and semantic
  errors are diagnostics and soft least-squares objectives, never shape-error
  feasibility thresholds. `semantic_contact_valid` reports the stricter physical
  contact witness separately. A failed contact fit does not discard a hand pose.
- The hand clock visits every valid frame in order, without dropping frames
  because object matching or contact establishment stalls. MPC always retains
  a nonzero running measured-joint objective for that frame's geometric IK.
  Contact IK is an additional seed/constraint and never replaces the geometric
  target. Position commands include bounded gravity feedforward; no gravity
  compensation force is injected. Command rates cover 1.25 times the geometric
  trajectory's peak per-coordinate velocity, with configured minimum rates.
  The original hand material position remains a soft prior; its search radius
  is independent of the strict 5/10 mm object-anchor FPS radius. Existing real
  hand/object contact bodies remain permitted during joint contact IK.
  Closest surface material coordinates are eliminated inside each contact-IK
  iteration; normal distance Jacobians allow tangential pad correspondence to
  change without imposing an extra rigid point constraint. Posture attraction
  weakens near contact but stays nonzero. There is no 25 mm shape threshold or
  equality requiring exact pose/semantic reproduction. Terminal costs emphasize
  object motion and contact tasks; servo depth can build behind the surface.
- Lambda uses the prepared MANO anchors on the executed object collision surface
  exactly. MuJoCo mesh collision uses convex hulls; the source visual mesh is
  retained for rendering. Hand queries inside visual concavities are projected
  onto the actual collider surface, preserving finger/region. A numerically/physically
  valid force solve and collision-checked joint IK remain required.
  Force closure and pose-descent certificates do not reject candidates in
  retargeting; hand shape remains a soft objective.
- Only failure of that original proposal enables FPS exploration: 5 mm, then
  10 mm, up to eight dense-surface samples per anchor and 64 bounded combinations
  per round. The default online budget is eight combinations; use
  `--candidate-capacity 64` for the larger search. Measured contact regions and current nearest projections are
  eligible local candidates. Complete nearest-projection sets are retained
  explicitly, and single-finger changes share the budget with joint combinations.
  Explored feasible proposals are ordered by
  optimized object pose cost, then semantic offset and switching.
- Poses equivalent within 5 mm / 5 degrees use elapsed demonstration time to
  disambiguate forward local object matching. Both object indices remain
  monotonic and are matched to measured object state independently of the
  per-frame hand clock. Hand progress never forces an unachievable jump in
  the nearest-pose Lambda goal.
- Actual contacts are measured independently of Lambda's ideal force witness.
  Inactive semantic slots have zero contact rows; no-contact phases skip Lambda.
  Plane support and existing physical contacts are included separately. The
  hand-object friction (0.5 for Sugar Box) is shared across all three paths;
  Wuji Lambda, grouped SCM and grasp response use the
  same four-ray native pyramidal bound `abs(Ft1)+abs(Ft2) <= mu*Fn`.
  Made contacts retain MuJoCo tangent axes; future frames are predictions
  refreshed from measured patches. The default shared Lambda/SCM APIs retain
  their existing cone choices for other callers.
  mass/inertia, servo parameters, torque caps and external gravity come from
  the generated physical scene. Robot-only friction rows use the explicit
  scene coefficient rather than inferring it from absent object columns.
  Table friction follows MuJoCo's source priority/mixing rule (1.0 in this
  scene), and MPC uses that separate coefficient. Lambda plane support remains
  a unilateral normal-response approximation without table sliding/rolling.
- MuJoCo contact geometry, mass, inertia and friction are inherited from the
  source scene. Finger servo gains and both torque clamps use the explicit
  scales above; wrist settings and joint/control position ranges stay intact.
  `physics_parameters.json` verifies the declared tuning and inherited geometry.
  Structural mount hull exclusions are retained; `--include-mount-collisions`
  restores the original pairings for diagnostics.
- MPC retains object/contact pose, nonzero MANO posture, action regularization
  and terminal task costs. Contact alignment includes the configured inward
  closure distance. Default force-preload and gravity-wrench cost weights are
  zero; no online static support allocator or pressure-force bias runs.
- The object reference follows demonstration progress without load permission.
  Native MuJoCo costs and collision checks select candidates, including bounded
  IK, material, normal-distance closure and transport seeds. Closure/load
  certificates cannot alter their ranking. At least two measured slots enable
  partial-grasp dynamics; this does not gate the object goal. Measured material
  anchors and original MANO semantic priors are tracked separately.
- Failure at both search radii rechecks the incumbent against the current state.
  If it is also infeasible, bounded geometric hand tracking continues while
  that contact force witness is omitted. Use `--stop-on-contact-failure` with
  `--max-infeasible-steps` to terminate repeated failures explicitly.

## Outputs

`scene.xml`, `config.json`, `semantic_contacts.npz`, `wuji_q.npy` and
`ik_diagnostics.json` describe preparation. Each mode writes `rollout.npz`,
`contact_plans.json`, `diagnostics.jsonl`, `metrics.json` and optional video.
The first solver failure or collision-protected hold additionally saves
`mpc_failure.npz` with measured state, contact geometry and model/cost parameters.
`full/planning_model.json` records friction-frame and transport-gate semantics.
`comparison.json` reports both modes: pose errors, hand-material and object-anchor
errors, retention/slip, penetration, ideal-versus-actual force errors, physical
closure fraction, progress, failures and cycle latency. A finite rollout or
successful numerical solve does not imply that the full demonstration completed;
check `reference_completion`, failures and collision diagnostics. Performance
claims against DexForge CA/FA are outside this first implementation.
`hand_trajectory_completed` and `all_hand_frames_commanded` describe command
coverage, not physical object success. Per-cycle geometric/contact references,
finger joint RMS and wrist position error measure actual hand tracking.
`benchmark.py --results <folders...>` plots measured wrist and object trajectories
against their demonstration references without rerunning simulation.
`render_rollout.py <folder> --mode full --goals` renders recorded physical poses
with the geometric joint targets overlaid, without rerunning optimization.
`geometric_pose_targets.npz` includes the 21 scaled joint targets and original
MANO joints. Metrics distinguish local object-goal error from error against the
full demonstration at the commanded hand frame. Static ideal joint torques
are checked against the source actuator limits and reported separately: an
object-only Lambda force witness does not certify actuator feasibility.

## Current validation limits

The archived first Sugar Box comparison in `outputs/wuji_retarget_validation_final`
ran 300 control cycles per mode with acados SQP_RTI and a four-step horizon.
Both modes reached reference frame 31 and exhausted the step budget. Full
retarget mean hand/object contact errors were 11.1/7.85 mm versus 18.3/20.5 mm
for IK replay. Full mean rotation error was worse (0.166 versus 0.100 rad).
These are errors against the selected local reference, not a completed
trajectory benchmark. Full mean cycle latency was 222 ms; 30 Hz execution has
not been demonstrated. The full run encountered acados QP failures and loaded
contact collision holds. A subsequent 180-cycle diagnostic run reached the
same reference without QP failure and still hit the collision guard.

That experiment predates continuous per-frame hand tracking and the source
physics audit. It is retained as an earlier failure record. Results from original
DexYCB labels and the current per-frame tracker are documented in
[VALIDATION.md](VALIDATION.md), with final artifacts in
`outputs/wuji_retarget_delivery_*` and `outputs/wuji_retarget_final_review`.
Command coverage and physical object accuracy must be assessed separately.
No claim of outperforming DexForge is supported by these measurements.
