# WorldSim V6.4 ArXiv Evidence Index

- Working title: **Uncertainty-Native Conditional World Compilation with Fixed-Opportunity Route-Risk Evaluation**
- Documentation freeze: `2026-08-27`
- Branch: `research/worldsim-v6.4-native-uq`
- Terminal state: `v64_research_complete_report_ready`
- Scientific outcome: supported conditional compiler and fixed-denominator route result; terminal negative collision critic
- New experiment in this closeout: none

This is the report-writing entry point for V6.4. It indexes frozen canonical evidence and separates supported claims, negative evidence,
and unread/unexecuted scope. Exact experiment rows remain in `docs/EXPERIMENTS.md`; the single failure fact source remains
`docs/RESEARCH_FAILURES.md`.

## 1. Executive result

V6.4 moved from weak native uncertainty ranking to an independently calibrated conditional state compiler. Native voxel density U2
improved pooled ranking relative to U0 but had weak within-scene separation. A supervised PCA-16 head improved AUROC but retained high
FPR and failed the first case-calibration route. A frozen full-native MLP then supported 40% selective coverage and exact-once
confirmation, after which a stratum-conditional map raised independent mean coverage to about 47.5% with zero case failures.

The compiler was materialized without target-model access, converted into sparse Gaussian state, and consumed by a fixed logged-route
operator. The first selected-denominator tail audit rejected current M0. A route-aware M1 policy passed an absolute fresh confirmation,
but did not improve the original selected-denominator CVaR. A preregistered fixed-opportunity denominator resolved this ambiguity:
on the untouched 96-case test, M1 preserved total coverage and reduced both worst-10 CVaR and pooled route-conflict density relative to
M0, with no case worse.

That result did not transfer to collision prediction. The P11 linear critics missed nearly all unsafe actions at the frozen operating
point. Independent long-tail threshold calibration still failed unsafe recall and the all-stop anti-trivial comparison. Cross-cohort
diagnostics showed that unsafe ranking degraded in addition to a prior shift. The defensible conclusion is therefore:

> Native uncertainty can support conditional state selection and improve exact empirical route-local conflict under a common
> opportunity denominator, but the tested uncertainty-filtered augmentation does not yield a calibrated collision authority.

This is a bounded cohort result, not a physical collision, planning, closed-loop, population-bound, or safety claim.

## 2. Method and artifact terminology

| Name | Role | Frozen interpretation |
|---|---|---|
| U0 | native baseline uncertainty | comparator only |
| U2 | native feature-density uncertainty | relative ranking signal; weak absolute separation |
| U3 | PCA-16 supervised risk head | ranking-only result; high FPR and failed case calibration |
| C0 | global 40% selective compiler | independently calibrated baseline |
| M0 | stratum-conditional compiler | confirmed coverage uplift; current selected-denominator route tail rejected |
| M1 | route-aware reallocation of M0 | fixed total coverage; supported on untouched fixed-opportunity route metric |
| P11 critic | bounded 13-action actor-envelope classifier | formal narrow policy gate pass, rejected as collision authority |

`supported` is stage-local. It never promotes a representation, compiler, or critic to a deployment or safety guarantee.

## 3. Canonical evidence chain

| Stage | Canonical evidence | Frozen result | Report use |
|---|---|---|---|
| P0/P1 | `P0_SCOPE.md`, `P1_CORE_UQ_FREEZE.md` | source branch, cohorts, UQ family, metrics, gates, and stop rules frozen | protocol and version boundary |
| P2/P2E | `run://worldsim_v64/WS-V64-P2-FRESH-NATIVE-SIDECAR-01/20260826T082600Z__fresh-native-s0-r3`; `run://worldsim_v64/WS-V64-P2E-FRESH-EVIDENCE-01/20260826T084000Z__fresh-evidence-s0-r1` | 6 scenes / 72 native targets and 72 evidence units complete | fresh native interface and denominator |
| P4N | `run://worldsim_v64/WS-V64-P4N-FRESH-NATIVE-VOXEL-UQ-01/20260826T091500Z__fresh-native-voxel-uq-s0-r2` | U2-U0 pooled AUROC gain `0.08305`; weak absolute separation | unsupervised UQ mechanism |
| P5 | `run://worldsim_v64/WS-V64-P5-SUPERVISED-RISK-01/20260826T093000Z__supervised-risk-s0-r1` | pooled AUROC `0.65812`, FPR95 `0.86774` | ranking-only supervised result |
| P6 | `run://worldsim_v64/WS-V64-P6-CALIBRATION-01/20260826T131000Z__case-calibration-s0-r1` | rejected: no positive coverage satisfying the frozen case-risk contract | compressed-representation calibration negative |
| P6R train/calibration | `run://worldsim_v64/WS-V64-P6R-SELECTIVE-MLP-01/20260826T134500Z__selective-mlp-s0-r1`; `run://worldsim_v64/WS-V64-P6R-CALIBRATION-01/20260826T141500Z__case-calibration-s0-r1` | full-native MLP; 40% policy with zero independent calibration failures | selective compiler recovery |
| P6R exact | `run://worldsim_v64/WS-V64-P6R-EXACT-ONCE-CONFIRMATION-01/20260826T153500Z__exact-once-confirmation-s0-r1` | mean coverage `0.39994`; `1/96` case failure | independent selective confirmation |
| P4C exact | `run://worldsim_v64/WS-V64-P4C-CONDITIONAL-EXACT-ONCE-CONFIRMATION-01/20260826T173000Z__exact-once-confirmation-s0-r1` | C0/M0 coverage `0.39994/0.47496`; M0 failures `0/96` | conditional coverage result |
| P10M/P10G | `run://worldsim_v64/WS-V64-P10M-CONDITIONAL-STATE-BAKE-01/20260826T180000Z__conditional-state-bake-s0-r1`; `run://worldsim_v64/WS-V64-P10G-GAUSSIAN-STATE-ADAPTER-01/20260826T181500Z__gaussian-state-adapter-s0-r1` | 96 target-free packages; 534,581 M0 Gaussians | compiler materialization capability |
| P10R/P10C | `run://worldsim_v64/WS-V64-P10R-GAUSSIAN-ROUTE-CONSUMER-01/20260826T183000Z__gaussian-route-consumer-s0-r1`; `run://worldsim_v64/WS-V64-P10C-ROUTE-CONFLICT-AUDIT-01/20260826T184500Z__route-conflict-audit-s0-r1` | bounded route exposure and low pooled route conflict | downstream interface capability |
| P10T | `run://worldsim_v64/WS-V64-P10T-ROUTE-TAIL-AUDIT-01/20260826T190000Z__route-tail-audit-s0-r1` | M0 tail `0.05171 > 0.05`, rejected | current-M0 negative result |
| P10R2 | `run://worldsim_v64/WS-V64-P10R2-EXACT-ONCE-CONFIRMATION-01/20260826T203000Z__exact-once-confirmation-s3-r1` | M1 absolute tail passed; selected-denominator relative delta `+0.00113` | absolute M1 confirmation, no relative claim |
| P10R3 | `run://worldsim_v64/WS-V64-P10R3-FIXED-DENOMINATOR-AUDIT-01/20260827T013000Z__fixed-denominator-audit-s0-r1` | fixed-denominator direction consistent on consumed cohorts | post-hoc diagnosis only |
| P10R4 | `run://worldsim_v64/WS-V64-P10R4-FIXED-DENOMINATOR-EXACT-ONCE-01/20260827T025000Z__exact-once-fixed-denominator-s4-r1` | all three preregistered fixed-opportunity gates passed | strongest relative route result |
| P11 | `run://worldsim_v64/WS-V64-P11-BOUNDED-COLLISION-CRITIC-01/20260827T033000Z__bounded-collision-critic-s0-r1` | narrow policy gates pass; unsafe recall collapse | collision-authority negative |
| P11R | `run://worldsim_v64/WS-V64-P11R-CALIBRATED-COLLISION-CRITIC-01/20260827T034500Z__calibrated-collision-critic-s0-r1` | independent threshold recovery rejected | terminal P11 negative |
| P11D | `run://worldsim_v64/WS-V64-P11D-COLLISION-CRITIC-SHIFT-DIAGNOSTIC-01/20260827T040000Z__collision-critic-shift-s0-r1` | unsafe prior and ranking both shift | failure mechanism characterization |

All large summaries, rows, native sidecars, and evidence units remain under `/root/autodl-tmp/runs/worldsim_v64/`. Repository documents
use stable `run://` identities and do not duplicate large artifacts.

## 4. Calibration and conditional-compiler evidence

| Stage | Coverage | Failures | Key result |
|---|---:|---:|---|
| P6 PCA-16 case calibration | no positive feasible coverage | `41/192` even at 5% | rejected |
| P6R independent calibration | nominal `0.40` | `0` | simultaneous upper bound `0.04865` |
| P6R exact confirmation | `0.399940` | `1/96` | four-stratum exact-once pass |
| P4C C0 exact confirmation | `0.399944` | `0/96` | global baseline |
| P4C M0 exact confirmation | `0.474961` | `0/96` | coverage uplift `+0.075016` |

The P6-to-P6R recovery changed the representation from the frozen PCA-16 head to the full 273-dimensional native feature vector and
used one fixed MLP architecture. It was not a threshold sweep. The P4C conditional map was frozen from calibration before its new
confirmation cohort was read.

## 5. Route-aware result and denominator distinction

| Cohort/metric | M0 | M1 | M1-M0 | Interpretation |
|---|---:|---:|---:|---|
| P10R2 fresh selected-denominator worst-10 CVaR | 0.039181 | 0.040313 | +0.001132 | M1 absolute pass; relative improvement unsupported |
| P10R4 untouched fixed-denominator worst-10 CVaR | 0.020726 | 0.010821 | -0.009905 | relative pass |
| P10R4 untouched pooled fixed density | 0.004945 | 0.002001 | -0.002943 | relative pass |
| P10R4 mean total coverage | 0.474970 | 0.474970 | 0 | preserved |

P10R2 divides conflicts by each arm's selected route voxels, while P10R4 divides by the same route-eligible opportunity count for both
arms. M1 deliberately reallocates selections away from the route, so these are different estimands. P10R4 supports the relative result
only for the preregistered fixed-opportunity estimand. It does not erase P10T's current-M0 rejection or turn P10R2 into a relative pass.

P10R4 paired lower/equal/higher counts were `18/78/0`; the half-tie probability `0.59375` is descriptive because no bootstrap or
significance gate was preregistered.

## 6. Terminal collision-critic evidence

| Metric | Real-only | Real + naive | Real + UNC verified |
|---|---:|---:|---:|
| P11 training positives | 3 | 191 | 96 |
| P11 unsafe-action recall | 0.02174 | 0 | 0.01087 |
| P11 selected-policy false-safe | 13 | 12 | 12 |
| P11 mean progress | 1.0 | 1.0 | 1.0 |
| P11R evaluation unsafe recall | 1.0 | 0.61314 | 0.62044 |
| P11R selected-policy false-safe | 0 | 3 | 2 |
| P11R mean progress / stuck | 0 / 1.0 | 1.0 / 0 | 0.87240 / 0.11458 |

The Real-only P11R operating point eliminates false-safe by rejecting every action and falling back to stop in all 96 cases. The
verified arm preserves progress but misses about 38% of unsafe actions. Its calibration-to-evaluation AP changed
`0.24710 -> 0.13740`, AUROC `0.71165 -> 0.56274`, and unsafe prior `0.07051 -> 0.10978`. These observations reject another
threshold-only recovery without implying that all collision critics or uncertainty-aware planning must fail.

## 7. Failure and recovery map

The detailed canonical entries are `V64-F01`--`V64-F28` in `docs/RESEARCH_FAILURES.md`.

| Group | Principal IDs | Frozen use |
|---|---|---|
| Scientific limitations/negative evidence | `V64-F10`, `V64-F11`, `V64-F15`, `V64-F21`, `V64-F28` | weak absolute UQ, high-FPR ranking, failed PCA calibration, current-M0 tail rejection, terminal collision-critic rejection |
| Recovered evaluation ambiguity | `V64-F25` | fixed-opportunity denominator preregistered and supported once on untouched test; selected-denominator result remains unchanged |
| Data/runtime/I-O recoveries | `V64-F01`--`F09`, `F12`--`F14`, `F16`--`F20`, `F22`--`F27` | preserve as reproducibility lessons; they are not method-negative counts |

Some earlier limitations remain part of the historical evidence even when a later architecture or estimand succeeds. The detailed
ledger is authoritative for each ID's exact terminal status; this index does not reclassify failure records.

## 8. Prior-art migration boundary

The implementation used prior work as a design basis rather than as external experimental evidence:

- OCCUQ native voxel feature-density uncertainty: <https://github.com/ika-rwth-aachen/OCCUQ>
- Waymax fixed counterfactual rollout and metric structure: <https://github.com/waymo-research/waymax>
- nuPlan collision/progress/stuck metric taxonomy: <https://github.com/motional/nuplan-devkit/blob/master/docs/metrics_description.md>
- class-balanced loss: <https://github.com/richardaecn/class-balanced-loss>
- logit adjustment for long tails: <https://arxiv.org/abs/2007.07314>
- Recovery RL safety-critic separation: <https://github.com/abalakrishna123/recovery-rl>

No external result is counted as a V6.4 result. These sources explain why the recoveries used native voxel density, fixed opportunity
denominators, bounded action metrics, and one independent long-tail calibration instead of a large backbone or RL sweep.

## 9. Reproducibility and resource inventory

| Axis | Frozen evidence | Status |
|---|---|---|
| Branch/protocol | P0/P1 and per-stage freeze documents | complete |
| Independent cohorts | development, calibration, confirmations, and seed-4 untouched test recorded before reads | complete |
| Native execution | repeated 8-scene / 96-target aggregates; maximum worker CUDA `4.1314 GiB` | single RTX 3090 sufficient |
| I/O scheduling | restricted shard catalogs, scene-local staging, canonical reuse, ready-first GPU feed | recovered; final GPU wait near zero for ready scenes |
| Exact evaluations | P6R, P4C, P10R2, P10R4 each have one canonical exact-once run | complete |
| Terminal negative | P11, one P11R recovery, and rows-only P11D diagnosis | complete |
| Integrity mechanism | semantic run IDs and immutable run leaves | no hash/checksum/fingerprint added |
| Test scope | targeted execution checks already required by stages | no broad smoke or regression matrix added |

## 10. Supported and unsupported report claims

Supported:

- relative native uncertainty ranking and the need for full-native supervised representation;
- independent case-level selective calibration and stratum-conditional coverage uplift;
- target-free state baking and sparse Gaussian/route consumption capability;
- exact empirical fixed-opportunity M1 route-risk reduction on one untouched 96-case cohort;
- failure of the tested uncertainty-verified collision-critic route, including cross-cohort ranking degradation.

Unsupported:

- calibrated probability semantics for U2/U3 outside their frozen stages;
- population bounds, statistical significance, physical collision, comfort, planning, closed-loop, deployment, or safety claims;
- an M1 relative improvement under the P10R2 selected-denominator metric;
- superiority of uncertainty-verified augmentation over naive augmentation in P11;
- claims about large NWM or RL models, because they were deliberately not trained.

## 11. Suggested technical-report structure

1. Motivation and inherited V6.3 failure: native features need uncertainty and conditional coverage, not another pointwise/surface head.
2. Protocol: split roles, exact-once reads, case-level risk, stratum map, and version-local stop rules.
3. Native uncertainty: U2 relative signal, U3 ranking, and the failed PCA case calibration.
4. Selective recovery: full-native MLP, independent 40% calibration, and exact confirmation.
5. Conditional compiler: C0-to-M0 coverage uplift and target-free Gaussian materialization.
6. Route evaluation: current-M0 failure, M1 selected-denominator ambiguity, and fixed-opportunity untouched confirmation.
7. Collision-critic negative: P11 narrow gate, P11R independent recovery failure, and P11D shift diagnosis.
8. Systems lessons: single-GPU resource envelope and I/O producer/consumer scheduling.
9. Limitations: empirical cohorts only, fixed proxies, no large NWM/RL, no closed-loop or safety claim.

The concise terminal decision is frozen in `V64_RESEARCH_FAMILY_CLOSEOUT.md`.
