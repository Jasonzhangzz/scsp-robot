# Current test_mpc_isaac.py benchmark

Seven objects, seeds 0/1/2 per object and task. 42 trials; asynchronous inference with realtime Isaac physics. Controller defaults are retained, including the 5000-planning-cycle limit. Frozen sources and exact commands are retained.

| Task | Successes | Success rate | Execution (s) | Planning (s) | Position error (m) | Quaternion error |
|---|---:|---:|---:|---:|---:|---:|
| rotation | 21/21 | 100.00% | 15.79 | 0.035 | 0.018 | 0.004 |
| flipping | 14/21 | 66.67% | 31.69 | 0.039 | 0.046 | 0.266 |

Execution time uses simulator timestamps and only successful trials. Planning time pools all planning callbacks, including failed trials, excluding startup and compilation. Final errors include all 21 trials per task. JSON includes sample standard deviations, per-object summaries, stable flips, solver failures and fallback counts. Wall timing reflects the recorded concurrent CPU allocation.

The paper table and PDF are not modified by this benchmark.
