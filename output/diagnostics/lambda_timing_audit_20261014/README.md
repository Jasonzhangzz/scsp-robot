# Lambda timing audit

Successful cached trials: foam_brick and elephant; first successful seed per method. Two pre-planning states per trial (25% and 75% of rollout), one warmup and three measured repeats per state. Frozen solver-benchmark source; one BLAS thread; sequential methods. No paper data changed.

Replay reconstructs object pose and body velocity from the previous cached row, keeps the recorded objective stage, and resets contact dwell/cooldown. It does not restore full simulator/controller history; this is a timing spot check, not exact trajectory reproduction. Cache timing includes all of _solve_step; lambda timing isolates choose_contact_points; candidate timing covers the full _solve_optimization call, not an internal IPOPT/SQP iteration.

| Method | Cache round ms | Replay round ms | Lambda selection ms | Candidate solves/round | Candidate solve ms |
|---|---:|---:|---:|---:|---:|
| ipopt | 479.43 | 407.51 | 406.75 | 51.25 | 7.95 |
| acados | 22.56 | 13.84 | 13.13 | 51.25 | 0.24 |
| qp | 124.92 | 94.20 | 93.36 | 51.25 | 1.81 |
| lcp | 98.18 | 87.52 | 86.62 | 51.00 | 1.68 |
