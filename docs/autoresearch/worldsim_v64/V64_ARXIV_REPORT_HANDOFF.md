# WorldSim V6.4 ArXiv Report Handoff

- Date: `2026-08-27`
- Branch: `research/worldsim-v6.4-native-uq`
- Research state: `v64_research_complete_report_ready`
- Scientific execution added by this handoff: none
- Failure-ledger delta: none
- Remote lifecycle after successful push: shutdown requested

This document is the compact authoring handoff. It does not replace the chronological ledgers or the detailed evidence index. Use
`ARXIV_EVIDENCE_INDEX.md` to navigate all stages and use this file to draft claims, tables, figures, limitations, and reproducibility
statements without reinterpreting the experiment history.

## 1. One-paragraph result

WorldSim V6.4 converts native occupancy-model uncertainty into an independently calibrated conditional state compiler. A full-native
MLP recovered a 40% selective policy after a PCA-16 risk head failed case-level calibration, and a frozen stratum-conditional map raised
independent exact-once mean coverage from approximately 40.0% to 47.5% with zero case failures. After target-free state baking and sparse
Gaussian adaptation, a route-aware M1 compiler preserved total coverage and reduced route-hidden-FREE conflict under a common
fixed-opportunity denominator on an untouched 96-case cohort. The reduction did not transfer to a calibrated collision critic:
uncertainty-verified augmentation missed unsafe actions, failed one independent threshold recovery, and exhibited cross-cohort unsafe
ranking degradation. The result supports conditional state compilation and bounded empirical route-local risk reduction, not physical
collision avoidance, closed-loop planning, a population bound, or a safety guarantee.

## 2. Copy-ready claim table

| Claim | Evidence | Allowed wording | Forbidden promotion |
|---|---|---|---|
| Native density contains relative uncertainty signal | P4N U2-U0 pooled AUROC gain `+0.08305`; scene support `2/2` | U2 improved pooled ranking over U0 on the frozen fresh evaluation | calibrated uncertainty or strong within-scene separation |
| Supervised compressed risk ranks but is not selective authority | P5 AUROC `0.65812`, FPR95 `0.86774`; P6 no positive feasible coverage | PCA-16 head improved ranking but failed the case-risk contract | deployable risk score |
| Full-native selective compiler confirms | P6R exact: coverage `0.399940`, failures `1/96` | 40% selective policy passed one independent exact-once cohort | population risk bound |
| Conditional coverage improves | P4C exact C0/M0 coverage `0.399944/0.474961`; M0 failures `0/96` | frozen stratum conditioning increased empirical coverage by `0.075016` | universal stratum calibration |
| Route-aware fixed-opportunity result confirms | P10R4 fixed CVaR `0.020726 -> 0.010821`; pooled density `0.004945 -> 0.002001`; coverage delta `0` | M1 reduced exact empirical route-local conflict on the untouched fixed-opportunity cohort | selected-denominator, collision, planning, or safety improvement |
| Collision-critic route fails | P11 recall `0.01087` verified; P11R recall `0.62044`, policy false-safe `2`; P11D AUROC `0.71165 -> 0.56274` | tested uncertainty-filtered augmentation did not yield a calibrated collision authority | all uncertainty-aware critics or planners fail |

## 3. Validation audit

The following checks were performed read-only immediately before this handoff. They validate evidence availability and documentation
consistency; they are not new scientific tests.

| Validation axis | Checked evidence | Result |
|---|---|---|
| Branch state | current branch and upstream ref | v64 branch clean and synchronized before handoff edits |
| Source ancestry | v6.3 surface-tail source branch is an ancestor | passed |
| Terminal machine state | `AUTORESEARCH_STATE.current.json` | terminal, no active task/hypothesis, no unlocked stage |
| Selective confirmation | P6R exact run directory, `summary.json`, `status.json`, `CASE_METRICS.jsonl` | present and JSON-readable |
| Conditional confirmation | P4C exact run directory and the same core artifacts | present and JSON-readable |
| Route selected-denominator evidence | P10R2 exact artifacts | present and JSON-readable |
| Route fixed-opportunity evidence | P10R4 exact artifacts | present and JSON-readable |
| Critic primary/recovery/diagnosis | P11, P11R, and P11D summaries/status plus retained action rows/models where applicable | present and JSON-readable |
| Mandatory ledgers | `RESEARCH_STATUS.md`, `EXPERIMENTS.md`, `RESEARCH_FAILURES.md` | terminal milestone represented in all three |
| Integrity policy | repository and run inspection | no artifact hash, checksum, or fingerprint introduced |
| Test policy | documentation-only handoff | no smoke or regression matrix executed |

The non-login shell must activate `motionproj` before Python reads, as already recorded by `V64-F03`; the final audit followed that
contract. No run was rewritten and no result was recomputed.

## 4. Canonical paper tables

### Table A: uncertainty and calibration progression

| Stage | Representation/policy | Primary result | Verdict |
|---|---|---|---|
| P4N | native boundary-global GMM U2 | AUROC `0.51855`, gain over U0 `0.08305` | relative-only, weak absolute |
| P5 | PCA-16 logistic U3 | AUROC `0.65812`, FPR95 `0.86774` | ranking-only |
| P6 | PCA-16 case policy | `41/192` failures at 5% coverage | rejected |
| P6R | full-273 MLP, global 40% | exact coverage `0.399940`, `1/96` failure | supported |
| P4C | stratum-conditional M0 | exact coverage `0.474961`, `0/96` failures | supported |

### Table B: denominator-sensitive route evaluation

| Evaluation | M0 | M1 | Relative statement |
|---|---:|---:|---|
| P10R2 selected-denominator worst-10 CVaR | 0.039181 | 0.040313 | M1 relative improvement unsupported |
| P10R4 fixed-opportunity worst-10 CVaR | 0.020726 | 0.010821 | M1 lower |
| P10R4 fixed-opportunity pooled density | 0.004945 | 0.002001 | M1 lower |
| P10R4 total coverage | 0.474970 | 0.474970 | preserved |

### Table C: terminal critic result

| Stage | Verified unsafe recall | Policy false-safe | Progress / stuck | Interpretation |
|---|---:|---:|---:|---|
| P11 frozen 0.5 threshold | 0.01087 | 12 | 1.0 / 0 | narrow policy gate pass, authority rejected |
| P11R independent threshold | 0.62044 | 2 | 0.87240 / 0.11458 | recovery rejected |
| P11D calibration -> evaluation | AUROC `0.71165 -> 0.56274` | n/a | n/a | prior and ranking shift |

## 5. Recommended figures

1. **System pipeline:** native sidecar/evidence -> U2/U3 -> full-native selective MLP -> C0/M0 -> target-free state bake -> Gaussian
   adapter -> route-aware M1 -> bounded critic audit.
2. **Calibration progression:** U2/U3 ranking versus P6 failure, followed by P6R and P4C coverage/failure points. Do not draw a smooth
   learning curve across these different stages.
3. **Denominator distinction:** show identical eligible opportunities, M0/M1 selected subsets, and conflicts; place P10R2 and P10R4
   metrics side by side to prevent denominator conflation.
4. **Critic transfer failure:** calibration/evaluation unsafe prior plus AP/AUROC and unsafe-score quantiles for naive and verified arms.

Suggested plots must be generated from retained canonical rows, not transcribed from rounded values in this handoff.

## 6. Failure appendix organization

| Appendix group | IDs | Treatment |
|---|---|---|
| Scientific negatives/limitations | `V64-F10`, `V64-F11`, `V64-F15`, `V64-F21`, `V64-F28` | include in method/limitation discussion |
| Denominator ambiguity and recovery | `V64-F25` | explain selected versus fixed opportunity; do not rewrite P10R2 |
| I/O, data, runtime, and operations recovery | `V64-F01`--`F09`, `F12`--`F14`, `F16`--`F20`, `F22`--`F27` | reproducibility appendix; not algorithm-negative counts |

`RESEARCH_FAILURES.md` remains authoritative for exact status and wording. This grouping is only an authoring aid.

## 7. Reproducibility statement

All V6.4 stages fit one RTX 3090. Native workers peaked at `4.1314 GiB`; the critic runs used less than `0.074 GiB` CUDA allocation.
Shared-storage I/O, not GPU capacity, was the dominant systems constraint. Restricted-shard extraction, per-scene staging, canonical
processed reuse, and ready-first CPU/GPU queues reduced avoidable GPU waits; the last two untouched-test scenes waited only
`0.0646/0.0625 s` after becoming native-ready. Multi-GPU execution was not required.

## 8. Authoring locks

- Keep P10R2 selected-denominator and P10R4 fixed-opportunity estimands separate.
- Report the P10R4 half-tie probability as descriptive only; no significance gate was frozen.
- Preserve the P11 narrow formal gate pass and the full-denominator authority rejection together.
- Do not relabel resolved engineering failures as scientific negative trials.
- Do not infer physical collision, planning, closed-loop, population, deployment, or safety claims.
- Any new analysis that selects a threshold, metric, policy, action lattice, or model requires a new version and unread denominator.

## 9. Source-of-truth order

1. Canonical run `summary.json`, `status.json`, per-case/action rows, and `resolved.yaml`.
2. `docs/EXPERIMENTS.md` for chronological exact results.
3. `docs/RESEARCH_FAILURES.md` for failure state and non-repetition locks.
4. `docs/RESEARCH_STATUS.md` and `AUTORESEARCH_STATE.current.json` for terminal project state.
5. `ARXIV_EVIDENCE_INDEX.md` and this handoff for report navigation.

If rounded values differ, use the canonical run summary and retain the claim boundary above.
