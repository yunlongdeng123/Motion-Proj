# WorldSim V6.5 ArXiv Evidence Index

- Working title: **From Voxel Confidence to Visited-State Reliability**
- Documentation freeze: `2026-08-28`
- Branch: `research/worldsim-v6.5-task-conditioned-authority`
- Terminal state: `v65_research_complete_arxiv_report_ready`
- Scientific outcome: given-trajectory visited-state ranking/calibration supported; direct fixed-action authority rejected
- New experiment in this report handoff: none

This is the report-writing entry point for V6.5. Exact experiment rows remain in `docs/EXPERIMENTS.md`; exact failure wording remains in
`docs/RESEARCH_FAILURES.md`. The technical report is `V65_ARXIV_TECHNICAL_REPORT.md`.

## 1. Executive result

The original voxel-level question did not survive task-conditioned transfer. Map context entered the model but did not improve
authority, Actor formulations failed at trajectory aggregation, and smooth-tail pooling optimized any-error detection at the expense
of expected visited-error cost. The successful object asks whether the world states visited by a given Ego trajectory over two seconds
are reliable.

Frozen Qmean transferred on a fresh cohort, its frozen monotone calibrator transferred on a second cohort, and direct action ranking
transferred on a third. The sole combined confirmation retained ranking and calibration but missed the preregistered action-selection
benefit. Therefore the report claims a reliability evaluator and expected-error calibrator, not an action authority compiler.

## 2. Canonical evidence chain

| stage | canonical evidence | frozen result | report use |
| --- | --- | --- | --- |
| P1R3 | `docs/autoresearch/worldsim_v65/P1R3_MAP_CONTEXT_RESULT.md` | map-conditioned voxel arm rejected | motivates prediction-object change |
| P1R4 | `run://worldsim_v65/WS-V65-P1R4-TRAJECTORY-VISITED-STATE-01/20260827T121500Z__visited-state-s0-r1` | train-only Qmean positive | freezes visited-state object |
| P1R5 | `run://worldsim_v65/WS-V65-P1R5-ACTOR-FALSE-SAFE-01/20260827T123100Z__actor-false-safe-s0-r1` | Actor companion rejected | excludes Actor-state claim |
| P1R6 | `run://worldsim_v65/WS-V65-P1R6-SMOOTH-TAIL-VISITED-STATE-01/20260827T124500Z__smooth-tail-s0-r1` | smooth tail rejected | retains deterministic mean |
| P1R7 | `run://worldsim_v65/WS-V65-P1R7-MONOTONE-VISITED-STATE-CALIBRATION-01/20260827T125000Z__monotone-calibration-s0-r1` | train-only slope/bias frozen | calibrator source |
| P2V inputs | preparation/native/evidence paths in `P2V_INPUT_PIPELINE_RESULT.md` | 6 scenes / 72 units | first fresh denominator |
| P2V formal | `run://worldsim_v65/WS-V65-P2V-VISITED-STATE-TRANSFER-01/20260827T142000Z__fresh-visited-transfer-s0-r2` | ranking and selected cost supported | given-trajectory claim |
| P3C inputs | preparation/native/evidence paths in `P3C_INPUT_PIPELINE_RESULT.md` | 6 scenes / 72 units | independent calibration denominator |
| P3C formal | `run://worldsim_v65/WS-V65-P3C-MONOTONE-CALIBRATION-TRANSFER-01/20260827T155000Z__calibration-transfer-s0-r2` | frozen calibration supported | expected-error claim |
| P10V inputs | preparation/native/evidence paths in `P10V_INPUT_PIPELINE_RESULT.md` | 6 scenes / 72 units | action-ranking denominator |
| P10V formal | `run://worldsim_v65/WS-V65-P10V-ACTION-VISITED-STATE-TRANSFER-01/20260828T003000Z__action-transfer-s0-r1` | fixed-action ranking supported | action-level reliability claim |
| P10X inputs | preparation/native/evidence paths in `P10X_INPUT_PIPELINE_RESULT.md` | 6 scenes / 72 units | one-shot denominator |
| P10X formal | `run://worldsim_v65/WS-V65-P10X-COMBINED-CONFIRMATION-01/20260828T013000Z__combined-confirmation-s0-r1` | 5/6 gates, terminal reject | authority boundary |

## 3. Paper result table

| stage | eligible denominator | ranking | calibration/decision result | verdict |
| --- | ---: | ---: | --- | --- |
| P2V | 63/72 trajectories | Spearman `0.633963`; AUROC `0.994152` | selected-40% cost `-49.25%` | supported |
| P3C | 60/72 trajectories | Spearman `0.715491` unchanged | MSE `-92.80%`; calibration error `-88.31%` | supported |
| P10V | 813/864 actions | Spearman `0.740235`; pairwise `0.732534` | selected-25% cost `-33.26%` | supported |
| P10X route | 60/72 routes | Spearman `0.609813`; AUROC `0.988868` | MSE `-81.39%`; calibration error `-87.19%` | component pass |
| P10X action | 739/864 actions | Spearman `0.772946`; pairwise `0.655686` | selected-25% cost only `-16.38%` | combined reject |

## 4. Prediction-object and method terminology

| term | meaning | allowed interpretation |
| --- | --- | --- |
| `q0(x)` | frozen V6.4 pointwise native uncertainty | input score, not calibrated probability |
| `V(tau)` | world-state samples inside the future 2s/1.5m trajectory footprint | task-conditioned support |
| `Qmean(tau)` | mean q0 over `V(tau)` | deterministic reliability ranking statistic |
| hidden-FREE rate | fraction of visited samples carrying the frozen error label | continuous empirical target |
| unsafe trajectory/action | at least one hidden-FREE visited sample | ranking label, not physical collision |
| frozen monotone map | fixed slope/bias applied to Qmean logit | expected-error calibrator |
| fixed action lattice | 4 progress ratios x 3 lateral offsets | bounded candidate set, not a planner |

## 5. Failure and recovery map

| group | IDs | report treatment |
| --- | --- | --- |
| prediction-object/method negatives | `V65-F03/F04/F06/F07/F09/F10/F11/F12/F19` | main method and limitations |
| target/capability/config corrections | `V65-F01/F02/F05/F08` | reproducibility appendix |
| streamed-pipeline/runtime entries | `V65-F13/F14/F16/F18` | systems appendix; no scientific read |
| narrow formal-entry recoveries | `V65-F15/F17` | disclose partial/no exposure and unchanged contract |

`RESEARCH_FAILURES.md` is authoritative. This grouping is an authoring aid and must not be used to reduce the recorded failure count.

## 6. Exposure and denominator notes

- P2V r1 loaded one unit before a shape error but emitted no aggregate metric, gate, verdict, or compact cache; r2 changed only tensor
  view semantics and is the canonical read.
- P3C r1 failed before input loading because its run-relative model locator was wrong; r2 changed only the locator to the same frozen
  q0 artifact used by P2V.
- P10X was the only combined confirmation read. Its failure is terminal; no scene replacement or second cohort is allowed.
- Footprint exclusions are part of frozen estimands. `scene-1046` and `scene-0718` remain in their cohorts even when their route units
  are below the minimum footprint.

## 7. Supported claims

1. Frozen Qmean ranked hidden-FREE rate over states visited by a given trajectory on fresh cohorts.
2. A frozen strictly monotone map improved expected visited-error calibration independently while preserving ranking.
3. Qmean ranked reliability across a bounded fixed action lattice on a fresh cohort.
4. The combined confirmation retained ranking/calibration but did not meet direct-selection benefit.
5. One RTX 3090 was sufficient; archive I/O was the dominant systems bottleneck.

## 8. Unsupported claims

- Actor-state reliability;
- collision probability, avoidance, or physical safety;
- planner, policy, closed-loop, deployment, or control authority;
- conformal or population guarantees;
- exact-test performance;
- successful TAC-Compiler action authority;
- general failure of local calibration, decision-focused learning, or learned critics.

## 9. Retained artifacts

| artifact family | retained location |
| --- | --- |
| canonical summaries/status/configs | `/root/autodl-tmp/runs/worldsim_v65/<task>/<run>/` |
| P10V action rows | P10V canonical run `ACTION_ROWS.jsonl`, 813 rows |
| compact formal caches | `/root/autodl-tmp/cache/worldsim_v65/` |
| chronological experiment ledger | `docs/EXPERIMENTS.md` |
| failure ledger | `docs/RESEARCH_FAILURES.md` |
| terminal state | `docs/autoresearch/worldsim_v65/AUTORESEARCH_STATE.current.json` |
| scene/exposure ledgers | `USED_SCENE_LEDGER_V65.json`, `SELECTION_EXPOSURE_LEDGER.json` |
| result summaries | `docs/autoresearch/worldsim_v65/*_RESULT.md` |

Repository documents use stable `run://` identities and do not duplicate multi-gigabyte native arrays.

## 10. Reproducibility and resource inventory

- GPU: one RTX 3090, 24GB; multi-GPU not required.
- Maximum native worker peak: `4.1314GiB`.
- P10V/P10X evaluator peak allocation: about `0.0392GiB`.
- P10X preparation: 10,718 members, targeted shards 3/7/8, no full fallback.
- P10X canonical native/evidence: 72 targets / 72 units, 48 evidence units reused from the overlapped partial run.
- V6.5 run tree at handoff: approximately 16GB.
- Free `/root/autodl-tmp` space at handoff: approximately 95GB.

## 11. Report-handoff validation

The following read-only checks were completed before documentation edits:

| validation | result |
| --- | --- |
| branch and upstream state | clean and synchronized |
| active V6.5 processes | none |
| P2V/P3C/P10V/P10X summary/status | present and JSON-readable |
| P10V action rows | present, 813 rows |
| terminal state | no active task/hypothesis; P10X read recorded |
| mandatory ledgers | terminal P10X milestone present |
| extra scientific tests | none |
| hashes/checksums/fingerprints | none added |

The first shell audit attempted an unqualified `python` in a non-login shell and encountered the already-known missing PATH entry. The
same read-only JSON check was immediately run with `/root/autodl-tmp/envs/motionproj/bin/python` and passed. No run or document was
created by the failed invocation.

## 12. Source-of-truth order

1. canonical run artifacts;
2. `EXPERIMENTS.md`;
3. `RESEARCH_FAILURES.md`;
4. `RESEARCH_STATUS.md` and terminal JSON state;
5. this index and `V65_ARXIV_TECHNICAL_REPORT.md`.

Rounded-value disagreements must be resolved from canonical summaries without recomputation or post-hoc threshold changes.
