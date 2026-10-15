# GPU batch LM successful-task spot check

Exact nine-DoF grouped quasi-dynamic soft SCM, original object objective and force constraints; float64 GPU projected LM, 20 iterations; finite differences 1e-4. All nonlinear candidate predictions, residual Jacobians and 3x3 LM updates run in Warp CUDA kernels, candidates in parallel. CUDA Graphs reuse the fixed iteration schedule. Host computes frozen contact geometry coefficients and uploads inputs; it does not run candidate nonlinear predictions or LM solves. No acados/IPOPT/CPU fallback.

RTX 5060 Ti. Compilation, graph creation and initial validation excluded from timers. Planning is the entire original _solve_step call, including contact detection, geometry prep, transfers, GPU solve, synchronization and host ranking. Execution is wall time until successful settling, including planning and physics; excludes scene initialization. GPU completion is synchronized before returning candidate results.

Samples selected beforehand from successful acados cache: foam brick, elephant, bunny, seed 0; all three GPU LM rollouts succeeded. This is a selected small sample, not a full-benchmark success-rate claim.

| Object | Steps | Execution (s) | Planning mean ± population std (ms) |
|---|---:|---:|---:|
| foam_brick | 344 | 3.70 | 9.11 ± 0.48 |
| elephant | 202 | 2.21 | 9.12 ± 0.47 |
| stanford_bunny2 | 198 | 2.23 | 9.43 ± 0.51 |

Execution: 2.71 ± 0.70 s, n=3 successful trials.
Planning: 9.20 ± 0.51 ms, n=744 pooled successful-trial rounds.

CasADi reference check: 12 states, 604 candidate predictions, both flip and pose stages. Maximum pose error 1.21e-17; maximum objective error 5.21e-13. All trace forces finite and within original bounds, all recorded backends GPU LM without fallback.

The preceding CPU LM timing is superseded for the requested GPU experiment. CPU and GPU trajectories may diverge under floating-point differences and discrete contact selection; retain each full trace rather than treating CPU timings as matched-rollout speedups. Paper tables unchanged.
