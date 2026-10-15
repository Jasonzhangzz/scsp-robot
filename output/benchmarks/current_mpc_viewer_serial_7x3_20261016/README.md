# Serial visible-window benchmark

42 trials: seven objects x seeds 0,1,2 x rotation/flipping. The controller source and seeds match the previous `current_mpc_async_7x3_20261016` benchmark. Real Isaac Gym viewer windows are displayed on desktop `:1`; only one object/task case runs at a time, with an asynchronous planner. No CPU pinning, no concurrent benchmark workers, and no explicit numerical-thread override. The code defaults to one numerical thread when no environment override is present. Default realtime pacing and the 5000-planning-cycle budget are retained. Bunny Flipping runs first, followed by bunny Rotation and the other object/task pairs.

The windows advance automatically through trials and objects. Closing a window aborts that trial and marks this cohort incomplete rather than counting it as a task failure.

`batch.log` records case progress and final summaries. `TASK/OBJECT/run.log` contains full controller logs; `command.json` records exact commands. On completion, `statistics.json`, `trials.json`, `trials.csv`, `report.md`, and `comparison.json` contain complete metrics and the comparison with the previous headless batch. No paper table is overwritten. Execution time is simulated completion time over successful trials; planning time pools callback wall time including failed trials; final errors include all trials.
