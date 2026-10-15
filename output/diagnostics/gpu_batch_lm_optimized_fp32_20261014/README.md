# Optimized GPU batch LM

Four requested changes: analytic forward Jacobian of the exact nine-DoF SCM/residual; per-candidate projected-gradient and accepted-small-step stopping; one fused CUDA kernel per planning solve; cached residual/Jacobian/Gram for rejected proposals and removal of zero padding contact blocks. The eight SCM contact sweeps remain unchanged.

FP64 analytic Jacobians matched CasADi to about 3e-17, but FP64 fused rollout timings were 7.55–11.97 ms, so structural changes alone did not improve throughput. The selected fast implementation uses FP32 GPU arithmetic, with independent FP64 CasADi checks. Maximum 20 LM iterations, projected-gradient tolerance 1e-6*(1+force norm), accepted-step stop only when relative cost decrease <1e-7 and relative step <1e-5. No solver fallback.

All host geometry, transfers, synchronization, result scoring and detection remain included in planning time. CUDA events isolate the fused GPU solve. Builds and validation are excluded from rollout timers.

| Task (seed 0) | Success | Steps | Execution (s) | Planning mean ± population std (ms) |
|---|---|---:|---:|---:|
| foam_brick | True | 664 | 2.91 | 2.60 ± 0.17 |
| elephant | True | 162 | 0.73 | 2.59 ± 0.22 |
| stanford_bunny2 | True | 128 | 0.52 | 2.23 ± 0.18 |

Planning, pooled 954 rounds: 2.55 ± 0.22 ms.
GPU-only fused solve, 120 state-repeat measurements: 0.746 ± 0.196 ms.
Steps over 3 selected successful tasks: 318.00 ± 245.05.

Validation: 12 initial/intermediate states, 620 candidate predictions, both flip and pose objectives. Max pose error vs FP64 CasADi 1.30e-7; max objective error 1.87e-6; max analytic Jacobian error 8.75e-8. All rollout force bounds, finite predictions, settled success and GPU-only backend checks pass. Nine same-state comparisons against previous FP64 finite-difference LM: maximum best-objective relative increase 9.73e-7. 471 sampled candidates averaged 9.29 iterations and 13.68 model evaluations; 61.8% stopped before budget exhaustion.

Limits: these are three selected tasks, not full-benchmark success estimates. Changing derivatives, stopping and precision changes the selected contact trajectories. In particular foam brick required 664 steps versus 344 in the previous GPU run; elephant improved from 202 to 162 and bunny from 198 to 128. Do not claim task-step performance is unchanged. Speedup is a combined result of structural changes and precision, not solely analytic Jacobians or solely early stopping.

Run: `/home/zz/miniconda3/envs/pi0/bin/python output/diagnostics/gpu_batch_lm_optimized_fp32_20261014/run.py`
Checks: `validate.py`, `compare_quality.py`; CUDA timing: `audit_gpu_timing.py`.
