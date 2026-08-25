# WorldSim V6.3 ArXiv Evidence Index

- Working title: **SurfNCC: Native-Feature Surface Tail-Risk Control for Verifiable Driving World Compilation**
- Documentation freeze: `2026-08-26`
- Branch: `research/worldsim-v6.3-surface-tail`
- Terminal state: `v63_surface_architecture_family_closed_negative_p7_locked`
- Scientific outcome: mixed mechanism evidence, terminal negative at P6 Stop 2
- New experiment in this closeout: none

This file is the report-writing entry point for V6.3. It indexes the frozen run evidence and records which statements are supported,
rejected, or unread. It does not replace `RESEARCH_STATUS.md`, `EXPERIMENTS.md`, or the single canonical failure ledger
`RESEARCH_FAILURES.md`.

## 1. Executive result

V6.3 established that native IR-WM features, deterministic candidate-surface compilation, exact hard projection, and bounded SurfNCC
training are executable on one RTX 3090. It also isolated a weighted-objective positive-authority collapse and showed that proxy
primal-dual constraints can recover a checkpoint that satisfies the frozen training-side retention and coverage gates.

That recovery did **not** produce a P6 method candidate. In the matched two-scene P6 evaluation, the independently trained B3
Surface-Mean arm preserved zero hard violations and non-degenerate OCC output, but worsened common hidden-FREE surface tail risk in both
scenes and emitted less than half the Native B2 OCC area. H-P6-001 was therefore rejected with `0/2` supporting scenes, Stop 2 closed the
surface architecture family, and B4/B5/M0 plus P7--P11 remained unexecuted or locked.

The defensible paper conclusion is therefore:

> Exact local evidence constraints and training-side coverage constraints are necessary but not sufficient for useful hidden-surface
> physical-state authority. In this frozen two-scene development evaluation, the tested Surface-Mean representation did not dominate a
> native-feature pointwise baseline on either surface tail risk or accepted area.

This is a development-set mechanism result, not a calibrated safety guarantee or an end-to-end world-simulation performance claim.

## 2. Checkpoint and candidate terminology

| Artifact | Correct name | Why it cannot be promoted further |
|---|---|---|
| P5 epoch 3 | best training-objective checkpoint | hard violations were zero, but retention=`0`, OCC coverage=`0.037198`, and source-valid UNKNOWN=`0.861807`; it was never a SurfNCC candidate |
| P5R epoch 6 | promotable P5R training candidate | exact training-side gates passed and P6 was unlocked, but no P6 comparative claim followed from this alone |
| P6 B3 epoch 1 | feasible B3 training candidate | it passed the internal B3 checkpoint contract, then failed the common P6 stage comparison in both scenes |
| P6 terminal | no V6.3 stage candidate | B3 was rejected; B4/B5/M0 were not executed; no input was frozen for P7 |

In particular, neither a low scalar training objective nor `candidate_promotable=true` in a training summary means “best SurfNCC
candidate” at the version level. Promotion is lexicographic and stage-local: hard/retention/coverage gates precede comparative tail-risk
and accepted-area evidence.

## 3. Canonical evidence chain

| Stage | Canonical evidence | Frozen result | Report use |
|---|---|---|---|
| P0/P1 | `SCOPE_FREEZE.md`, `P1_NOVELTY_PROTOCOL_FREEZE.md` | branch, method boundary, cohorts, metrics, gates, and stop rules frozen before quality read | protocol and novelty boundary |
| P2 | `run://worldsim_v63/WS-V63-P2-NATIVE-SIDECAR-01/20260824T145110Z__native-dl-s1-r1` | 76/76 native logits/BEV sidecars complete and finite; no prototype bridge | native-interface capability |
| P2D | `run://worldsim_v63/WS-V63-P2D-NATIVE-POINTWISE-DIAGNOSTIC-01/20260824T145924Z__native-pointwise-s0-r1` | Native B2 remained `4/28 ACCEPT, 4/4 false-safe`, hard violations `0` | native features alone did not rescue the pointwise route |
| P3 | `run://worldsim_v63/WS-V63-P3-SURFACE-CORPUS-01/20260824T154059Z__surface-dl-s20260824-r1` | 72/72 units; 86,360 surfaces, 111,282 patches, 11,583,001 points; schema and negative contracts passed | surface-data construction capability |
| P4 | `run://worldsim_v63/WS-V63-P4-CAPACITY-01/20260825T051200Z__capacity-h002-s0-r3` | complete-proposal path passed all frozen capacity gates in 11.863 s at 0.256589 GiB peak | bounded execution and gradient-path capability |
| P5 | `run://worldsim_v63/WS-V63-P5-SURFNCC-TRAIN-01/20260825T051530Z__surfncc-train-s0-r1` | finite 7-epoch training and hard0, but positive-authority collapse prevented candidate promotion | weighted-sum failure evidence |
| P5D | `run://worldsim_v63/WS-V63-P5D-AUTHORITY-COLLAPSE-DIAGNOSTIC-01/20260825T084844Z__authority-diagnostic-s0-r2` | raw network emitted no safe-OCC OCC; authority veto count was zero; weighted tail gradient dominated retention and opposed it | mechanism diagnosis |
| P5R | `run://worldsim_v63/WS-V63-P5R-CONSTRAINED-SURFNCC-TRAIN-01/20260825T091631Z__constrained-train-s0-r1` | epoch 6 passed exact training gates under proxy primal-dual optimization | constrained-optimization recovery, training scope only |
| P6 B0/B1/B2 | `run://worldsim_v63/WS-V63-P6-DEVELOPMENT-AB-01/20260825T151200Z__baselines-s0-r1` | Native B2 comparator frozen on 24 units / 2 scenes | matched reference |
| P6 B3 train | `run://worldsim_v63/WS-V63-P6-DEVELOPMENT-AB-01/20260825T152000Z__b3-mean-s0-r1` | epoch 1 was the only best feasible B3 training checkpoint | checkpoint selection, not stage verdict |
| P6 B3 eval | `run://worldsim_v63/WS-V63-P6-DEVELOPMENT-AB-01/20260826T014500Z__b3-eval-s0-r1` | stage failed; supporting scenes `0/2` | terminal comparative result |

Run summaries, manifests, per-unit rows, checkpoints, and diagnostic plots remain under
`/root/autodl-tmp/runs/worldsim_v63/`. The repository records stable `run://` identities rather than duplicating large artifacts.

## 4. Mechanism evidence: collapse and constrained recovery

| Checkpoint/run | Hidden-FREE tail | Safe-OCC retention | OCC coverage | UNKNOWN | Interpretation |
|---|---:|---:|---:|---:|---|
| P5 epoch 3 | 0.014507 | 0.000000 | 0.037198 | 0.861807 | scalar objective minimum obtained by rejecting useful positive authority |
| P5R epoch 6 | 0.464393 | 0.721226 | 0.114148 | 0.313899 | frozen training-side candidate after constrained recovery |
| P6 B3 epoch 1 | 0.608174 | 0.636863 | 0.285326 | 0.550411 | feasible internal B3 checkpoint; comparative verdict still pending at this point |

P5D used all 48 train units and 79 frozen gradient batches without optimizer updates. Safe-OCC points had raw/projected/post-authority
state counts `[FREE,OCCUPIED,UNKNOWN]=[153,0,62301]`, and the authority veto count was zero. Raw safe-OCC versus hidden-FREE
`P(OCC)` retained binned AUC `0.722684`, but absolute OCC probability had collapsed. Weighted tail-to-retention full-model gradient norm
ratio was `5.531`, while their mean cosine was `-0.411568`. This supports weighted-objective optimization collapse as the primary root,
with weak authority-supervision alignment as a secondary limitation; it rejects hard projection or authority veto as the primary cause.

P5R shows that the unchanged representation can regain the frozen training-side rate constraints under proxy primal-dual optimization.
It does not show calibration, generalization, or P6 dominance.

## 5. Terminal P6 comparison

### Pooled descriptive metrics

| Metric | Native B2 | B3 Surface-Mean | B3 result |
|---|---:|---:|---:|
| Common hidden-FREE surface CVaR | 0.491496 | 0.608174 | 23.739% worse |
| Proposal false-safe surrogate | 0.396840 | 0.515384 | 29.872% worse |
| Emitted OCC points | 2,298,450 | 1,047,186 | area ratio 0.455605 |
| Safe-OCC retention | 0.851056 | 0.636863 | B3 passes absolute retention gate |
| Source-valid UNKNOWN | 0.266284 | 0.554227 | B3 passes pooled UNKNOWN gate only |
| Hard violations | 0 | 0 | hard projection preserved |
| Accepted cases | 24/24 | 24/24 | no all-UNKNOWN case-collapse loophole |

### Frozen scene-level gate

| Scene | B2 tail | B3 tail | Relative improvement | B3/B2 area | B3 retention | B3 source-valid UNKNOWN | Support |
|---|---:|---:|---:|---:|---:|---:|---|
| scene-0450 | 0.497850 | 0.596685 | -19.852% | 0.406270 | 0.601623 | 0.651678 | no |
| scene-1089 | 0.465122 | 0.655861 | -41.008% | 0.499323 | 0.704815 | 0.445030 | no |

Both scenes passed hard, retention, case, actor, and static checks. Both failed the tail-improvement and accepted-area checks;
scene-0450 additionally failed the UNKNOWN check. The negative result is therefore not explained by a hard-solver regression or a
complete all-UNKNOWN policy. The common representation produced worse tail risk while retaining only `40.6%--49.9%` of B2 area.

## 6. Validation and provenance inventory

| Validation axis | Frozen evidence | Status |
|---|---|---|
| Source/protocol | P0/P1 scope, config, preregistrations, sequential stop rules | complete |
| Native-interface completeness | P2 formal summaries for all 76 authorized targets | passed |
| Surface corpus structure | P3 summary plus 72 immutable unit artifacts and 8/8 negative contracts | passed |
| Complete-proposal execution | P4 finite loss/gradients, exact hard projection, proposal token, reload/repeat | passed |
| Training capability | P5, P5R, and B3 summaries report finite training, optimizer steps, wall time, peak GPU, and hard violations | passed |
| Mechanism diagnosis | P5D distributions, decision-stage counts, gradient norms/cosines, and six-panel plots | complete on train partition |
| P6 denominator | baseline and B3 summaries both report 24 selection units and 2 scenes; per-unit rows are retained | matched |
| P6 stage decision | summary `stage_gate` records every scene check and `0/2` support | rejected |
| Unread partitions | legacy28 in P6, calibration, confirmation, and exact-once test | intentionally unread/locked |
| Current documentation closeout | canonical summaries reconciled with state, experiment, failure, and closeout ledgers; JSON/JSONL syntax checked | complete |

No new smoke or regression matrix was added for this documentation closeout. No hash, checksum, or fingerprint artifact was added.

## 7. Failure record for report appendices

The canonical detailed entries are `V63-F01`--`V63-F24` in `docs/RESEARCH_FAILURES.md`. For report organization:

| Group | IDs | Frozen status and use |
|---|---|---|
| Active scientific negative evidence | `V63-F02`, `V63-F24` | native pointwise false-safe persisted; Surface-Mean then failed both P6 scenes and closed the route |
| Recovered scientific failure | `V63-F19` | P5 positive-authority collapse was diagnosed and training-side feasibility recovered by P5R; the later P6 rejection still stands |
| Surface/schema/protocol recoveries | `V63-F03`--`F07`, `F10`--`F16` | resolved or resolved before formal execution; preserve as reproducibility and denominator-integrity lessons, not method rejections |
| Runtime/numerics/operations recoveries | `V63-F01`, `F08`, `F09`, `F17`, `F18`, `F20`--`F23` | resolved; failed entrances and diagnostic artifacts never replace canonical scientific runs |

No failure ID was added during the final documentation audit. B4, B5, M0, and P7--P11 are **not executed/locked**, not rejected
experiments.

## 8. Claims that are and are not supported

Supported:

- the native IR-WM sidecar and deterministic surface corpus are technically executable on the frozen development denominator;
- exact hard projection consistently enforces observed FREE/OCC, contradiction, and lifecycle constraints in the reported learned runs;
- the original weighted objective exhibited positive-authority collapse, with train-only diagnostics supporting gradient conflict as the
  primary mechanism;
- proxy primal-dual optimization restored the frozen training-side retention/coverage feasibility without changing the hard solver;
- B3 Surface-Mean failed to improve Native B2 in either P6 selection scene under the frozen common metric and anti-triviality gates.

Not supported:

- a version-level “best SurfNCC candidate,” calibrated OCC authority, risk bound, or real-world safety guarantee;
- superiority of Surface-Max, Surface-CVaR, or M0 authority, because B4/B5/M0 were never executed;
- any legacy28, calibration, confirmation, exact-once test, multi-actor downstream, or deployment claim for V6.3;
- a general theorem that surface models or CVaR cannot work; only the frozen Surface-Mean architecture and protocol were rejected.

## 9. Suggested technical-report structure

1. **Motivation:** V6.1 oracle authority versus learned false-safe; V6.2 hard0 but `4/4` false-safe.
2. **Problem and protocol:** observed evidence, hidden-FREE surface risk, positive OCC authority, lexicographic promotion, and locked splits.
3. **System capability:** P2 native interface, P3 surface compiler, P4 complete-proposal capacity.
4. **Mechanism study:** P5 collapse, P5D distributions/gradients, P5R constrained recovery.
5. **Matched evaluation:** B0/B1/B2 and B3; report pooled metrics descriptively and use the two scene-level gates for the verdict.
6. **Negative evidence:** distinguish training feasibility, stage promotion, active scientific failures, resolved engineering failures, and
   unexecuted hypotheses.
7. **Limitations and future work:** two development scenes, fixed IR-WM/proposal generator, no calibration or held-out confirmation;
   future work requires a fresh version with explicit feature-level uncertainty and scene/stratum-conditional coverage.

The exact P6 result tables and Stop-2 interpretation are also frozen in `P6_SURFACE_FAMILY_CLOSEOUT.md`.
