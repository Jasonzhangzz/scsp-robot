# Exact SCM batch LM rollout spot check

CPU float64, compiled CasADi residual/prediction, mapped evaluation and vectorized 3x3 projected LM solves. 20 iterations, finite differences 1e-4, adaptive damping, no scalar/acados/IPOPT fallback. Preserves SCM, stage objective, candidate selection, force constraints, initialization and physical MuJoCo execution. Builds are excluded from execution timing. This is not the previous different-model GPU Warp implementation.

Three initial states selected before running LM from successful acados cached trials (seed 0): foam brick, elephant, Stanford bunny. All three LM rollouts succeeded. The selection-conditioned result is not a full-benchmark success-rate estimate.

| Object | Steps | Execution (s) | Planning mean ± population std (ms) |
|---|---:|---:|---:|
| foam_brick | 474 | 24.12 | 49.24 ± 1.41 |
| elephant | 202 | 11.31 | 54.24 ± 2.07 |
| stanford_bunny2 | 198 | 10.73 | 52.43 ± 1.62 |

Execution over 3 successful tasks: 15.39 ± 6.18 s.
Planning pooled over 874 rounds: 51.12 ± 2.69 ms.

Every planning round checks residual SSE against the original scalar objective; all traces have the LM backend without fallback, finite forces within the 10 N norm bound. Paper tables remain unchanged.
