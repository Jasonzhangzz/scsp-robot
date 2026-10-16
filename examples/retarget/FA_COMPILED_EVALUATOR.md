# FA equation and gradient backend

`--evaluation-backend compiled` replaces the hand optimizer's numerical oracle.
It retains the FA control knots, multistart family, warm start, L-BFGS/SLSQP/
condensed-control-SQP solver, budgets and line search. The executor is unchanged.

The generated C equations include articulated mass, bias and actuator forces,
joint limits, friction rows, the configured four coupled ComFree corrections,
implicit Euler damping and quaternion integration. A C++ mesh BVH searches the
actual collision geometry at each predicted state. Contact templates only enter
the objective. Forward-selected mesh features are saved for the analytic stage
adjoint, matching the native FA tape's discrete-selection convention. A C
rollout assembles the scalar cost and control gradient; no Python simulation
callback or finite-difference Jacobian is used in that rollout.

The objective retains FA's post-step tracking, pre-step contact geometry,
terminal weighting, detached marked weights, capsule self-collision and both
control/state smoothing grids. SPD systems use Cholesky and an analytic VJP.
Subtree accumulation changes the implementation of rigid-body calculations,
without freezing the mass matrix or contact geometry.

Current scope: the 26-actuator Wuji model, one SDF mesh, `contact_topk=1`, and a
world-fixed support plane. Unsupported geometry configurations raise an error.
The generated equations use FP64; native Warp uses FP32. At a mesh feature
boundary they can choose different nearest faces/edges and therefore different
normal derivatives, even when forward positions agree closely. This backend
requires closed-loop validation and is not a promise of equivalent success
rates or real-time performance.

Use the same optimization settings when comparing backends:

```bash
python -m examples.retarget.run_fa_contact_mpc \
  --variant lbfgs --executor comfree --multistarts 8 --iterations 40 \
  --evaluation-backend compiled \
  --compiled-cache outputs/fa_compiled_evaluator_20261016/cache \
  --case-name compiled_fa
```

`--evaluation-backend native` selects the original Warp evaluator. The optional
`--memoize-evaluations` applies to either backend: it reuses per-candidate cost
values when backtracking assembles a batch entirely from previously evaluated
FP32 raw controls. Gradients still use the selected analytic evaluator. The
cache is reset for each window and invalidated when ADMM changes contact targets.
Native state buffers are refreshed before contact updates or plan export if a
cache hit made them stale. `--no-memoize-evaluations` disables this optimization.

Initial C/C++ compilation is cached and excluded from solver-window wall time.
`FA_COMPILED_OPT=0/1/2/3` selects the C compiler optimization level. Compiler
range-propagation passes are disabled for the large generated conditional graph.
`FA_COMPILED_SCALAR_MATH=0` and `FA_COMPILED_RECURSIVE_SUMS=0` retain the matrix
and flat-summation variants for implementation comparisons. These are numerical
implementation choices, not alternate contact models.

Validation and timing:

```bash
python -m examples.retarget.validate_fa_compiled --windows 0 80 120 --solve
python -m pytest -q tests/test_fa_compiled_spd.py \
  tests/test_fa_value_cache.py tests/test_fa_hand_slsqp.py
```

Numerical comparisons, analytic-gradient diagnostics and measured performance
are saved under `outputs/fa_compiled_evaluator_20261016/`. See its `report.md`
for implementation details, the complete native trajectory, and the partial compiled trajectory. Measurements
from debug builds or concurrent workloads must not be presented as isolated
warm performance.
