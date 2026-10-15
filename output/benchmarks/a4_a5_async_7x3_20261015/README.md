# A4/A5 asynchronous Franka ablations

The cohort was fixed before observing outcomes: A4 and A5; rotation and flipping; foam_brick, stanford_bunny2, elephant, piggy_bank, teapot, mug, rubber_duck; seeds 0, 1, 2 for every object/task/variant (84 trials). Each trial allows at most 2500 planner cycles with 70 sampled contacts. Unsuccessful trials remain in the cohort.

Both variants run independent Isaac Gym physics and planner processes (`--async-planner --headless-realtime`). Physics continues during inference at the nominal 20 ms policy interval. Each worker receives two exclusive physical CPU cores. The 2500-cycle limit counts completed planner callbacks; with asynchronous inference it is not a 50-second simulated-time limit. Per-trial records retain physics ticks, simulated end time, observation gaps, and inference wall time.

A4 refreshes an object-only MuJoCo scene from each measured Isaac observation and evaluates contact forces with MuJoCo's native pyramidal contact QP. The inner force optimization follows the existing `PointContactBaseline` implementation used by `test_scm_mujoco.py`, with SLSQP, 12 maximum iterations, and native QP scoring of the feasible incumbent. It does not substitute SCM predictions when a local solve stops early.

A5 retains the existing soft-SCM objective and solves it with the repository's pycma CMA-ES adapter: 24 population members, at most 12 generations per contact, plus the shared feasible anchor evaluations. CasADi JIT compiles the same objective graph. Neither variant changes the ranking strategy, CPO policy, execution scene, target, or success criteria.

## Statistics

- Success rate: complete task successes divided by all 21 trials per variant/task; a stable flip alone does not count as final task success.
- Execution time: mean simulated completion time over successful trials; unavailable when no trial succeeds. For asynchronous execution, this uses observed simulator timestamps rather than multiplying planner cycles by 20 ms.
- Planning time: sum of complete planning-callback wall time divided by all planner cycles, including failed trials. Startup, compilation, IPC/physics waiting, and reporting are excluded.
- Position and quaternion errors: means of terminal errors over all 21 trials, including failures.

`manifest.json` records the frozen source root, hashes, package versions, CPU allocation, and batch status. Each object directory contains the exact command, complete run log, per-trial JSONL, and rollout summary. `statistics.json` and `trials.json` are generated only after all 84 records pass cohort and backend validation.

The earlier synchronous batch at `../a4_a5_7x3_20261015` is superseded and excluded from paper statistics. Five-cycle smoke tests under `/tmp/scsp_a{4,5}_async_smoke*` are also excluded.

## Background continuation

`batch.log` records completed cells, final statistics, and paper compilation status. `finalization.json` records the continuation state. `background_finalize.py` waits for the original running batch, then uses `fill_paper_table.py` to update only A4/A5 table cells and their protocol description, rebuilds `tro_revision/root.pdf`, and removes the root compilation intermediates. Failures retain raw data and diagnostics; a compilation failure restores the preceding PDF.
