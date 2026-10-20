| Task | Method | Planning time (ms) | Source status |
|---|---|---:|---|
| Rotation | SCSP | 24.12 ± 11.31 | cached_cycle_moments |
| Rotation | DyWA | 31.00 | historical_rounded_summary |
| Rotation | ContactSDF | 637.80 ± 473.67 | cached_cycle_moments |
| Rotation | STOCS-3D | -- | unavailable |
| Rotation | Sampling | 228.00 | historical_rounded_summary |
| Rotation | CF-MPC | 82.00 | historical_rounded_summary |
| Rotation | A1 | 23.00 | historical_rounded_summary |
| Rotation | A2 | 23.00 | historical_rounded_summary |
| Rotation | A3 | 38.28 ± 17.56 | cached_pooled_summary |
| Rotation | A4 | 168.95 ± 78.20 | cached_cycle_moments |
| Rotation | A5 | 891.15 ± 580.17 | cached_cycle_moments |
| Rotation | A6 | 41.76 ± 78.63 | cached_cycle_moments |
| Rotation | A7 | 12.86 ± 2.93 | cached_cycle_moments |
| Flipping | SCSP | 22.91 ± 8.03 | cached_cycle_samples |
| Flipping | DyWA | 31.00 | historical_rounded_summary |
| Flipping | ContactSDF | 638.91 ± 463.31 | cached_cycle_moments |
| Flipping | STOCS-3D | 239325.61 ± 125098.37 | cached_offline_trial_times |
| Flipping | Sampling | 252.00 | historical_rounded_summary |
| Flipping | CF-MPC | 81.00 | historical_rounded_summary |
| Flipping | A1 | 24.94 ± 20.23 | cached_cycle_samples |
| Flipping | A2 | 12.59 ± 2.34 | cached_cycle_samples |
| Flipping | A3 | 21.80 ± 11.85 | cached_cycle_samples |
| Flipping | A4 | 132.95 ± 49.02 | cached_cycle_samples |
| Flipping | A5 | 577.81 ± 113.82 | cached_cycle_samples |
| Flipping | A6 | 43.00 | historical_total_and_count |
| Flipping | A7 | 12.86 ± 2.90 | cached_cycle_moments |

SCSP Flipping excludes exactly one timing sample: mug trial 002, planning_time_s[1502] = 175.64558509900235 s. All 35 trials and their outcomes remain included.
Unfiltered timing statistics and the exclusion record are preserved in the JSON snapshot; original data files are unchanged.
A6 Rotation: archived direct-CPO cohort complete (21 trials). A6 Flipping: only historical 15-trial interim summary available; its values are preserved.
A7: archived Rotation and Flipping cohorts complete (21 trials each).
Raw precision, sample counts, source paths, caveats, and SHA256 hashes: table_iv_planning_sources.json.
Historical rounded summaries do not regain precision through unit conversion.
STOCS-3D reports offline time per trajectory; online methods report time per callback.
