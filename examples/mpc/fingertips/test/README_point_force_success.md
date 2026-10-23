# Paired continuous point-force success benchmark

Run `run_point_force_success.py` in the pi0 environment. Default cohort: seven
objects, seeds 0–9, seven configurations, 490 trials. The same object/seed has
bit-identical initial pose, target and sampled candidate contacts in all configurations. `initial_states.json` saves these inputs before formal trials.

```bash
/home/zz/miniconda3/envs/pi0/bin/python examples/mpc/fingertips/test/run_point_force_success.py \
  --jobs 1 --output-dir output/benchmarks/point_force_success_gpu_20261022
```

Current defaults are QP+AdamW, ComFree+GPU batch LM, ContactSDF+GPU batch LM,
SCM+GPU batch LM, SCM+CMA-ES, SCM+CEM and SCM+IPOPT. Historical `lm` still
selects NumPy CPU LM; `gpu-lm` selects the new float64 Warp/CUDA implementation
in `planning/point_force_lm_gpu.py`, not the legacy GPUExactBatchLM model.
Residuals (22 components), seven-point finite differences, independent damped
3x3 systems and accept/reject updates all run on CUDA. Three starting forces
(warm, zero, full normal) are compared before 20 fixed LM iterations; no CPU
prediction or optimization runs inside the captured iteration graph. Damping
starts at 1e-3, changes by 0.4/4 on accept/reject, and the finite-difference
step is 1e-4. The fixed budget avoids the historical CPU stopping check, which
compared a proposal to the previous planning call's warm start.

The archived first cohort used QP+LM, superseded by the separate QP+AdamW
cohort. AdamW is the outer force optimizer (Torch, 20 iterations, lr=0.025,
weight_decay=0, central differences with step 1e-5); native QP response is
still evaluated internally. ContactSDF here is the dynamic extension of the
upstream LSE contact projection (commit 3a21edf8, Q=M/h^2, sigma=120), with the
same measured contact geometry as other models. It is not the official full
ContactSDF robot controller or its learned quasi-static parameters. SCM is
`softplus_nonnegative_gap_v2` (T=1e-4 N s, r=1e-8): both impulse activation and
the global line correction use d=max(phi,0). This replaces the old hard,
signed-gap `models.corrected_scm` law. Standalone fidelity backends retain their
explicitly named historical baselines for comparison with archived data.

One frozen-contact full-dynamic prediction step spans 20 ms. Native MuJoCo
physics independently executes the selected body-fixed force over all 40
substeps (0.5 ms each). The point and force direction follow the object's body
frame. There is no robot collider, state teleportation, pulse duty scaling or
execution force clipping. All solvers share the 3 N force bound, the friction
pyramid and the per-interval object delta-v caps (`--max-linear-force-delta-v`,
default 0.3 m/s; `--max-angular-force-delta-v`, default 3 rad/s). The caps are
part of the planner's feasible set (CPU `force_commands`, GPU `bounded_force`,
IPOPT CasADi expression), never an executor clip. Without them a 3 N command on
a 10 g object is a 6 m/s one-step dead-beat impulse: cohorts run that way
(e.g. `point_force_success_20261022`) failed mostly by launching the object off
the table (`physics_instability`/`native_reset`), not by a contact-model error.
Native QP uses its actual soft constraint objective; noslip postprocessing
is disabled so it does not silently add a second solver after QP. ComFree uses
the pinned upstream row mass and force equations (ba8b9969), ported to float64
on the common MuJoCo geometry. This is a model-law comparison with common
geometry, not a comparison of entire native simulator stacks.

CPU batch LM uses 20 damped least-squares iterations with central finite differences.
CMA-ES uses batched full 3x3 covariance matrices, cumulative step-size and
covariance paths, positive log weights and clipping of out-of-box samples.
CEM uses CUDA and per-contact distributions; CMA-ES/CEM use 24 samples and
12 generations. IPOPT solves the separable candidate problems in one sparse
NLP, with exact CasADi derivatives and at most 12 iterations. Solver status
and feasible incumbents are logged; a fixed iteration budget is not a claim
of global optimality or convergence. Initial force guesses are warm-started
where supported. Timings from concurrent trials measure contention and must
not be used as isolated inference-time comparisons. CUDA CEM and CPU batch LM
also use different hardware paths; these measurements do not establish that
CEM is faster than GPU batch LM. `verify_cem_timing.py` separately measures
the full CUDA CEM planning call with device synchronization before and after.

Starts use yaw randomization and tilt 35–75 degrees. The task first flips the
object upright and then reaches its target pose. Maximum length is 5000 policy
cycles (100 simulated seconds). Success requires position error <2 cm,
quaternion metric <0.015, and ten consecutive unforced intervals satisfying
the pose tolerances and terminal speed limits (0.05 m/s and 0.5 rad/s).
Exceptions and timeouts stay in the failure denominator.

Each trajectory is a `.json.gz` containing every physical/control step,
planning rounds, solver metadata and initialization checks. A separate
`.summary.json` permits incremental aggregation without reading full traces.
`STATUS.json`, `summary.json`, `success_rates.csv` and `success_rates.md` update
as trials finish. Source snapshots and SHA256 hashes are frozen in `protocol.json`.
`--resume` requires matching source hashes and cohort settings. Historical pulse
results are not reused. A three-cycle smoke run is available with `--smoke`;
its success rates are not task-performance evidence.

Validation: 22 tests in `tests/test_point_force_success.py` cover the coupled
line-search inequality, nonnegative-gap invariance, NumPy/CasADi/Torch parity,
loaded-state QP parity, least-squares objective equivalence CUDA LM/CPU LM step and warm-start parity for SCM, ComFree and ContactSDF,
and continuous physical point forces. GPU LM tests reject CPU/CEM fallback. A separate 42-case short
preflight (all objects/configurations, seed 0) completed without exceptions.
Two pre-existing tests in `test_comfree_model.py` instantiate a legacy lambda
optimizer without `scm_theta` and fail before reaching the new benchmark law;
the other seven tests in that file pass. This benchmark does not use that
legacy optimizer's force-prediction graph.

For only the new GPU LM cohort (three models, 210 trials):

```bash
/home/zz/miniconda3/envs/pi0/bin/python examples/mpc/fingertips/test/run_point_force_success.py \
  --configs scm_gpu_lm,comfree_gpu_lm,contactsdf_gpu_lm --jobs 1 \
  --output-dir output/benchmarks/point_force_gpu_lm_20261022
```

CEM maintains one diagonal Gaussian for each candidate contact, initialized at
(0.1,0.5,0.5) with standard deviation 0.3 in normalized force coordinates.
Each generation samples 24 candidates/contact and clips coordinates to [0,1].
The best six replace the distribution mean/variance without smoothing;
variance has a 1e-8 floor (standard deviation 1e-4). Each generation includes
zero force and, after generation zero, the incumbent. The first generation
also seeds normal magnitudes 1e-4, 1e-3, 1e-2, 0.1 and 1. Force coordinates
map to the same bounded friction pyramid used by all solvers. The best action
over all points and 12 generations is returned. Collision detection and mass
preparation happen once per measured observation, not per population sample.
CUDA synchronization and result readback precede the end of the planning timer.
