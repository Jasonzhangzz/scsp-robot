# Onset load audit

Table IV labels are unchanged. Force and impulse statistics use only the ten reference solves over 0–20 ms. The 20–22 ms PhysX endpoint diagnostic and the MJWarp endpoint forward diagnostic are excluded from these load statistics.

Normal force is summed across geometric contact points of each interaction pair. Peak force is the maximum of these sums over the ten 2 ms substeps. Reported impulse is the cumulative scalar normal impulse, not the magnitude of the net vector impulse. These are sampled/substep-average loads, not continuous-time impact peaks.

| Reference | Onset interaction | All onset labels | Onset within actual ten solves | Peak force range (N) | Normal impulse range (mN s) |
|---|---|---:|---:|---:|---:|
| MJWarp | object–ball | 2000 | 1996 | 0.0421298–1.31591 | 0.152287–18.775 |
| IsaacGym | object–ball | 14 | 13 | 0.00147451–0.138988 | 0.00844832–1.57909 |
| IsaacGym | object–ground | 407 | 404 | 0.0135715–1.81771 | 0.0500747–5.27669 |

Counts are interaction-specific and may overlap. MJWarp onset labels comprise 2,000 object–ball onsets. PhysX onset labels comprise 407 object–ground and 14 object–ball onsets, including two queries with both. Loads for the four endpoint-diagnostic-only object–ball onsets in MJWarp, three endpoint-diagnostic-only object–ground onsets and one endpoint-diagnostic-only object–ball onset in PhysX are not used for the ranges above.

The imposed sphere-center force spans 0–0.5 N in each onset group. This imposed force is distinct from collision contact force. Its external impulse spans 0–10 mN s over 20 ms.

Impulse quartiles in onset_load_strata.csv are descriptive and rank-based, computed separately within each reference and interaction pair. Object, inertia, impact time and contact location are not controlled within these strata, so they cannot establish a causal force–error relationship. The 13 actual PhysX object–ball onsets are insufficient for a well-populated impact-strength study.
