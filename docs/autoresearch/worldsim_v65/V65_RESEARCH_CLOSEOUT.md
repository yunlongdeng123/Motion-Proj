# WorldSim V6.5 research closeout

Date: 2026-08-28  
Branch: `research/worldsim-v6.5-task-conditioned-authority`  
Terminal status: visited-state reliability supported; direct action authority rejected

Formal report entry points:

- [`V65_ARXIV_TECHNICAL_REPORT.md`](V65_ARXIV_TECHNICAL_REPORT.md)
- [`ARXIV_EVIDENCE_INDEX.md`](ARXIV_EVIDENCE_INDEX.md)

## Research question and final answer

The initial question asked whether the current voxel is correct. Train-only and fresh experiments showed that voxel-, map-,
Actor-, admission-, and smooth-tail formulations did not provide a transferable authority signal. The successful prediction
object was changed to:

> If Ego executes trajectory `tau`, are the world states visited over the next `H=2s` reliable?

For this object, frozen Qmean transferred as a reliability ranking signal and a two-parameter monotone map transferred as an
expected visited-error calibrator. However, using the same score directly to select the lowest-quarter fixed actions did not
deliver the preregistered decision benefit on the one-shot combined confirmation. V6.5 therefore establishes an evaluator
interface, not a planner or action-authority compiler.

## Canonical evidence chain

| stage | independent status | canonical result |
| --- | --- | --- |
| P1R4 | train-only supported | trajectory-visited Qmean object identified |
| P2V | fresh supported | Spearman `0.63396`, unsafe AUROC `0.99415`, selected cost `-49.25%` |
| P1R7 | train-only supported | monotone slope/bias frozen before independent calibration |
| P3C | fresh calibration supported | MSE `-92.80%`, 5-bin error `-88.31%`, ranking unchanged |
| P10V | fresh fixed-action supported | Spearman `0.74023`, AUROC `0.85878`, pairwise `0.73253`, selected cost `-33.26%` |
| P10X | one-shot combined rejected | 5/6 gates; calibration strong, selected cost only `-16.38%` vs required `-25%` |

Canonical run references:

- P2V: `run://worldsim_v65/WS-V65-P2V-VISITED-STATE-TRANSFER-01/20260827T142000Z__fresh-visited-transfer-s0-r2`
- P3C: `run://worldsim_v65/WS-V65-P3C-MONOTONE-CALIBRATION-TRANSFER-01/20260827T155000Z__calibration-transfer-s0-r2`
- P10V: `run://worldsim_v65/WS-V65-P10V-ACTION-VISITED-STATE-TRANSFER-01/20260828T003000Z__action-transfer-s0-r1`
- P10X: `run://worldsim_v65/WS-V65-P10X-COMBINED-CONFIRMATION-01/20260828T013000Z__combined-confirmation-s0-r1`

## ArXiv-ready claims

1. A task-conditioned visited-state query transferred where local voxel, map-context, Actor false-safe, learned-admission,
   and smooth-tail formulations failed under their frozen protocols.
2. Frozen Qmean ranked hidden-FREE rate over the states visited by a given trajectory on multiple disjoint fresh cohorts.
3. A frozen two-parameter monotone map substantially improved expected visited-error MSE/calibration on two independent
   fresh reads without changing rankings or selected sets.
4. Fixed-action reliability ranking transferred, but ranking and calibration were insufficient to guarantee the required
   downstream action-selection benefit on the one-shot confirmation.
5. The empirical boundary is consistent with task-relevant failure detection and decision-calibration literature: uncertainty
   must be judged through downstream cost, and correcting miscalibration need not remove grouping-loss regret.

## Claims that must not be made

- no collision probability or collision avoidance guarantee;
- no planner, policy, closed-loop, control, or safety authority;
- no Actor-state reliability claim after the R5 false-safe rejection;
- no conformal coverage, population guarantee, or exact-test claim;
- no claim that P10X passed because 5/6 component gates passed;
- no second-confirmation, replacement-scene, relaxed-threshold, or post-read critic result.

## Primary negative evidence

| failure family | outcome | interpretation |
| --- | --- | --- |
| R3 map/context | rejected | more local scene context did not rescue voxel-level transfer |
| R5 Actor false-safe | rejected | the available evidence did not support Actor-state reliability |
| R6 smooth tail | rejected | tail smoothing did not create transferable visited-state authority |
| V64-F28 critic precedent | rejected | learned collision critic was not reopened in V6.5 |
| V65-F19 combined authority | rejected | strong calibration/ranking left insufficient direct selection benefit |

Operational failures `V65-F13`–`V65-F18` remain in `RESEARCH_FAILURES.md` with exposure audits; none should be represented
as scientific negatives. P2V r2 and P3C r2 are valid narrow recoveries because their failed entries disclosed no scientific
metric and changed only execution/locator defects.

## Suggested paper tables and figures

- formulation funnel: voxel/map/Actor/admission/smooth-tail versus visited-state query outcomes;
- cohort-disjoint result table for P2V, P3C, P10V, and P10X;
- raw versus frozen-monotone calibration plot using P3C and P10X equal-count bins;
- action reliability scatter/rank plot from P10V `ACTION_ROWS.jsonl` and P10X compact cache;
- scene-level selection delta plot showing P10X `5/0/1` heterogeneity;
- system diagram separating reliability evaluation from downstream action authority.

## Literature anchors

- [Task-Relevant Failure Detection for Trajectory Predictors in Autonomous Vehicles](https://proceedings.mlr.press/v205/farid23a.html)
- [Decision from Suboptimal Classifiers: Excess Risk Pre- and Post-Calibration](https://proceedings.mlr.press/v258/perez-lebel25a.html)
- [On the role of model uncertainties in Bayesian optimisation](https://proceedings.mlr.press/v216/foldager23a.html)
- [Local calibration: metrics and recalibration](https://proceedings.mlr.press/v180/luo22a.html)

## Terminal rule

V6.5 is closed on the P10X one-shot verdict. Any local/multicalibration, decision-focused regret learner, Actor-state method,
or new action head must start as a new version with a new method, protocol, branch, and untouched evaluation cohort. It cannot
be presented as a recovery of the V6.5 combined candidate.

The final report handoff performed only read-only artifact availability/JSON checks and added no new scientific experiment. Remote
shutdown is requested only after the report commit has been pushed successfully.
