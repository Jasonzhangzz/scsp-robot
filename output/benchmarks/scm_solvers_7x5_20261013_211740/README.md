# SCM solver batch: 7 objects × 5 trials × 4 methods

Started: 2026-10-13T21:19:35+08:00

Each method schedules 35 trials, seeds 0–4; execution order 1,2,3,4,0.
Each trial uses the default random tilt, wrench execution, mass 0.01 kg and at most 5000 steps.
Two trials per method execute concurrently, with one BLAS/OpenMP thread per process.
Method logs contain RESULT lines; per-trial logs contain step progress.
JSON/CSV traces and live summary.json/summary.csv are stored in each method directory.
Success, step-limit and exception results all count as completed trials; failed trials do not stop the batch.
Wall times include concurrent workloads and should not be interpreted as isolated performance benchmarks.
Python sources, meshes and XMLs are frozen under source/scsp-robot; source_sha256.json records hashes.

| Method | Controller PID / process group | Log |
| --- | --- | --- |
| acados | 3073422 | acados.log |
| ipopt | 3073423 | ipopt.log |
| cma-es | 3073424 | cma-es.log |
| cem | 3073425 | cem.log |

To stop only one method and its worker trials, use `kill -- -PID` with its controller PID from the table.
Do not stop unrelated benchmark processes.

## First complete result observed for each method

All four first trials were foam_brick, seed 1, and succeeded with settling. This does not mean the full batch is complete.

| Method | Steps | Controller PID | First result |
| --- | --- | --- | --- |
| acados | 577 | 3073422 | /home/zz/scsp-robot/output/benchmarks/scm_solvers_7x5_20261013_211740/acados/trial_foam_brick_001.json |
| ipopt | 890 | 3073423 | /home/zz/scsp-robot/output/benchmarks/scm_solvers_7x5_20261013_211740/ipopt/trial_foam_brick_001.json |
| cma-es | 751 | 3073424 | /home/zz/scsp-robot/output/benchmarks/scm_solvers_7x5_20261013_211740/cma-es/trial_foam_brick_001.json |
| cem | 717 | 3073425 | /home/zz/scsp-robot/output/benchmarks/scm_solvers_7x5_20261013_211740/cem/trial_foam_brick_001.json |

Monitoring has ended; the four detached controllers and their remaining worker trials are left running.
