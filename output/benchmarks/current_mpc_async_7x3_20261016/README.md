# Current MPC: 7 objects x 3 trials x 2 tasks

Frozen source: `source/scsp-robot`, entry `examples/mpc/franka/ik2/test_mpc_isaac.py`. Seeds 0,1,2 for each object and task; 42 trials total. Current controller defaults retained, with a 5000-planner-cycle limit, asynchronous planner, realtime Isaac execution, and one numerical compute thread. Two exclusive physical cores per worker, at most seven workers. Task/object pairs are interleaved so both tasks start immediately.

`batch.log`: background progress and final statistics. `TASK/OBJECT/run.log`: full controller output. `TASK/OBJECT/command.json`: exact arguments and CPU allocation. `manifest.json`: source hashes, protocol, and final status. `progress.json`: completed trial cells only; active incomplete cells remain in individual logs.

On completion, `statistics.json` includes overall and per-object metrics, `trials.json` and `trials.csv` contain all 42 raw outcomes, and `report.md` presents the main metrics. Successful-trial execution time uses simulator timestamps; planning time pools wall time of all callback cycles, including failures, excluding startup/compilation; terminal errors include all trials. Sample standard deviations and solver failure/fallback counts are retained. A stable flip alone is not final task success. The paper table is not modified.
