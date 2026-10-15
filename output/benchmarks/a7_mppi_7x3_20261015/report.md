# A7: prespecified 7-object x 3-seed cohort

Both tasks use seed IDs 0, 1, 2 for each object, with a 2500-step limit. All completed trials are included. Execution time is simulated time on successful trials; planning time pools all planning callbacks and cycles; final pose errors include failures. MPPI uses 128 sampled trajectories, horizon 10, one update per cycle, temperature 0.005, and noise standard deviation 0.003 m. The model, weights and settings were fixed before the cohort. CPU workers run on distinct physical cores.

| Task | Object | Success | Execution (s) | Planning (s/cycle) | Position error (m) | Quaternion error |
|---|---|---:|---:|---:|---:|---:|
| rotation | foam_brick | 0/3 | -- | 0.012845 | 0.373857 | 0.345527 |
| rotation | stanford_bunny2 | 0/3 | -- | 0.012687 | 0.361147 | 0.350607 |
| rotation | elephant | 0/3 | -- | 0.012780 | 0.429030 | 0.438413 |
| rotation | piggy_bank | 0/3 | -- | 0.013113 | 0.232803 | 0.653667 |
| rotation | teapot | 1/3 | 0.7400 | 0.012393 | 0.247537 | 0.008573 |
| rotation | mug | 0/3 | -- | 0.012907 | 0.249880 | 0.017530 |
| rotation | rubber_duck | 1/3 | 7.0800 | 0.013236 | 0.145583 | 0.629587 |
| rotation | ALL | 2/21 | 3.9100 | 0.012858 | 0.291405 | 0.349129 |
| flipping | foam_brick | 0/3 | -- | 0.013054 | 0.431870 | 0.999147 |
| flipping | stanford_bunny2 | 0/3 | -- | 0.012952 | 0.378020 | 0.677733 |
| flipping | elephant | 0/3 | -- | 0.013000 | 0.390090 | 0.705977 |
| flipping | piggy_bank | 0/3 | -- | 0.013214 | 0.351333 | 0.851370 |
| flipping | teapot | 0/3 | -- | 0.012825 | 0.418120 | 0.677593 |
| flipping | mug | 0/3 | -- | 0.012366 | 0.237160 | 0.869767 |
| flipping | rubber_duck | 0/3 | -- | 0.012611 | 0.223043 | 0.815517 |
| flipping | ALL | 0/21 | -- | 0.012860 | 0.347091 | 0.799586 |

## Separate failure diagnosis (excluded from table statistics)

A repeat of teapot/flipping/seed 0 reproduced the formal final errors exactly. It reached stable upright orientation at 15.38 simulated seconds, but ended with 0.41633 m position error and 0.03528 quaternion error. All 2500 planning cycles used MPPI; no solver failures or requested-versus-executed command disagreements were observed. Among 2026 near-contact pairs with nontrivial predicted and observed position changes, 419 had opposite signs. In 316 cycles the model predicted position improvement while the measured position worsened (both exceeding 0.01 mm). This identifies contact-prediction error as a failure mechanism; it does not establish that correcting the model alone would make the task succeed. The diagnostic replay was not added to the 42-trial cohort.
