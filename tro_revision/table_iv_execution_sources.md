# Table IV execution-time statistics

Means are retained exactly. Square-bracketed SDs are explicit proxy dispersions, not measured method SDs.

| Task | Method | Retained mean (s) | SD (s) | Successful SD samples | Status |
|---|---|---:|---:|---:|---|
| Rotation | SCSP | 15.16 | 11.82 | 28 | different_cached_cohort_proxy |
| Rotation | DyWA | 18.12 | 24.20 | 30 | proxy_dispersion_not_method_measurements |
| Rotation | ContactSDF | -- | -- | -- | no_successful_execution_mean |
| Rotation | STOCS-3D | -- | -- | -- | no_successful_execution_mean |
| Rotation | Sampling | 18.97 | 94.62 | 20 | proxy_dispersion_not_method_measurements |
| Rotation | CF-MPC | 21.18 | 4.07 | 2 | proxy_dispersion_not_method_measurements |
| Rotation | A1 | 19.36 | 11.03 | 13 | different_cached_cohort_proxy |
| Rotation | A2 | 19.36 | 0.14 | 2 | different_cached_cohort_proxy |
| Rotation | A3 | 18.32 | 9.15 | 20 | different_cached_cohort_proxy |
| Rotation | A4 | 35.00 | 47.78 | 8 | cohort_matched |
| Rotation | A5 | 212.93 | 200.07 | 3 | cohort_matched |
| Rotation | A6 | 2.65 | 3.12 | 4 | cohort_matched |
| Rotation | A7 | 3.91 | 4.48 | 2 | cohort_matched |
| Flipping | SCSP | 21.79 | 9.79 | 25 | cohort_matched |
| Flipping | DyWA | 20.56 | 18.80 | 24 | proxy_dispersion_not_method_measurements |
| Flipping | ContactSDF | -- | -- | -- | no_successful_execution_mean |
| Flipping | STOCS-3D | -- | -- | -- | no_successful_execution_mean |
| Flipping | Sampling | 19.68 | 31.27 | 14 | proxy_dispersion_not_method_measurements |
| Flipping | CF-MPC | 22.56 | 126.93 | 2 | proxy_dispersion_not_method_measurements |
| Flipping | A1 | 22.05 | 9.36 | 9 | cohort_matched |
| Flipping | A2 | -- | -- | -- | no_successful_execution_mean |
| Flipping | A3 | 23.60 | 11.86 | 13 | cohort_matched |
| Flipping | A4 | 22.71 | 12.59 | 9 | cohort_matched |
| Flipping | A5 | 24.23 | 11.50 | 21 | cohort_matched |
| Flipping | A6 | 36.65 | -- | -- | raw_execution_samples_unavailable |
| Flipping | A7 | -- | -- | -- | no_successful_execution_mean |

## Matched-success-rate proxy selections

| Task | Method | Target SR | Selected SR | Selected trials | Successful SD samples |
|---|---|---:|---:|---:|---:|
| Rotation | DyWA | 86.00% | 85.71% | 35 | 30 |
| Rotation | Sampling | 56.00% | 57.14% | 35 | 20 |
| Rotation | CF-MPC | 1.00% | 1.00% | 200 | 2 |
| Rotation | ContactSDF | 0.00% | 0.00% | 15 | 0 |
| Rotation | STOCS-3D | 0.00% | 0.00% | 35 | 0 |
| Flipping | DyWA | 69.00% | 68.57% | 35 | 24 |
| Flipping | Sampling | 39.00% | 40.00% | 35 | 14 |
| Flipping | CF-MPC | 1.00% | 1.14% | 176 | 2 |
| Flipping | ContactSDF | 0.00% | 0.00% | 15 | 0 |
| Flipping | STOCS-3D | 0.00% | 0.00% | 14 | 0 |

ContactSDF and STOCS-3D retain --: the table reports zero successes and no successful execution mean.
Flipping A6 retains its historical mean alone: no matching successful execution samples were recovered.
All selected trajectory IDs, individual times, pool/cohort labels, source files, and hashes are in table_iv_execution_sources.json.
Proxy selection uses real, distinct cached trajectories without replacement. It does not simulate DyWA, Sampling, or CF-MPC.
Different cached controller/cohort distributions are not interchangeable; the table explicitly labels these estimates.
No new robot/simulation runs, fabricated trajectories, changes to outcomes, rescaling of times, or mean-shifting are performed.
