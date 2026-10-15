# CUDA event timing confirmation

Six states from the three successful GPU-LM rollouts (25% and 75% of each rollout); 20 measured repeats plus one discarded warmup per state. Same float64 SCM graph, 20 projected LM iterations, precompiled and captured graphs. Input upload is synchronized before recording the start CUDA event; the end event is recorded after graph completion. CUDA event time measures actual GPU graph execution, not host launch latency. Profiling instrumentation may add small overhead to the full-step measurement.

| Component | Mean (ms) |
|---|---:|
| Full _solve_step | 9.095 |
| GPU LM graph, CUDA events | 7.356 |
| Frozen geometry and uploads | 0.342 |
| Readback and adapter bookkeeping | 0.135 |
| Host contact ranking | 0.584 |
| Other planning work | 0.660 |

| LM iterations | Mean GPU graph time (ms), across three states |
|---|---:|
| 1 | 0.347 |
| 5 | 1.775 |
| 10 | 3.646 |
| 20 | 7.335 |

The 9.20 ms table value is a complete planning cycle with 20 LM iterations, not a single LM iteration and not GPU-only time. The measured CUDA graph averages 7.36 ms. No evidence of a ms/s error or timing only asynchronous launch. Each LM iteration uses five exact model/residual evaluations (base plus three finite differences plus proposal); each model prediction runs eight grouped contact sweeps. Earlier ~0.7 ms microbenchmark used a simpler, different contact model. Lower iteration-count runs measure speed only: their rollout success and optimization quality have not been validated, so they do not replace the paper values.
