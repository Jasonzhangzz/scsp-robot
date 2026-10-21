# Figure and Table Audit Report

Scope: Table IV, execution-time rows and their sample/source notes only.
Inputs checked: experiment.tex, cached per-trial records, table_iv_execution_sources.json.
Build/source status: generated from source records; whole-manuscript claims are outside scope.
Summary: 1 existing blocking provenance issue, 2 recommended limitations, 1 author-verification item.

## Inventory
| ID | Path/location | Caption/title | First callout | Source/script |
|---|---|---|---|---|
| tab:all_data | experiment.tex; Rotation/Flipping execution rows | Comparison and ablation results | Simulation-results paragraph | recover_table_iv_execution.py |

## Blocking Issues
| Location | Figure/table | Issue | Evidence | Fix |
|---|---|---|---|---|
| SCSP success-rate cells (pre-existing, unchanged) | Table IV | Their reported SRs are not the rates of the recovered timing cohorts | Rotation table 0.95 vs cache 28/35; Flipping table 0.81 vs cache 25/35 | Author must document the distinct SR cohort/projection or use its matching raw data; execution-only edits do not resolve it |

## Recommended Fixes
| Location | Figure/table | Issue | Fix |
|---|---|---|---|
| Proxy SD cells | Table IV | Outcome-matching does not establish a baseline method timing distribution; mixed cache protocols differ | Keep square-bracket/dagger disclosure; replace with method-specific successful records when available |
| CF-MPC proxy cells | Table IV | Only two successful proxy trajectories contribute to each SD | Treat as a weak descriptive estimate, not a precise uncertainty claim |

## Minor / Production Issues
No additional typography changes outside the two execution rows and their notes.

## Author Verification Needed
| Figure/table | Why verification is needed |
|---|---|
| Table IV retained historical means and Flipping A6 | Matching historical raw samples were not recovered for all cells; A6 SD remains unavailable |

## Readiness Checklist
| Dimension | PASS/FAIL/PARTIAL/NA | Notes |
|---|---|---|
| Execution SD arithmetic | PASS | Success-only sample SD, ddof=1 |
| Mean preservation | PASS | All original numerical execution means retained |
| Proxy source disclosure | PASS | Bracket/dagger notation, captions/notes, exact IDs and hashes |
| Distinct proxy identities | PASS | No selected trajectory reused across the five comparison groups within a task |
| Raw method-specific evidence | PARTIAL | Available cells are measured; others explicitly proxy/undefined |
| Whole-paper evidence readiness | PARTIAL | Existing SR provenance and historical means require author verification |
| Full manuscript cross-reference audit | NA | Only the requested table was scoped |
