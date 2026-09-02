# HARP-3D contribution, evidence, and claim map

## Central claim

World validity is not world difficulty. HARP-3D repairs evidence-inconsistent physical geometry while preserving legitimate
hazardous Actors, and keeps physical repairability separate from task-conditioned trajectory authority.

The paper-level causal chain is intentionally narrow:

```text
physical artifact repair
  -> hazard and Actor preservation
  -> selective reliability with explicit abstention
  -> non-causal, fixed-lattice interface utility
```

## Current contribution map

| Contribution | Paper location | Code-facing interface | Canonical evidence | Supported claim | Prohibited extension |
| --- | --- | --- | --- | --- | --- |
| C1. Actor-preserving physical 3D compilation | Figure 1 metadata-locked four-quadrant teaser; Method 4.1; Experiments “Zero-shot Actor-local surface evidence”, “Four-action physical compilation”, “Frozen camera evidence”, and “Visibility-conditioned physical evidence”; supplement A--B/F | `av2_canonical_surface.py`, `av2_four_action_compiler.py`, `av2_p3_hard_evidence.py`, `av2_p3b_camera_evidence.py`, `visibility_certificate.py`, `visible_failure_attribution.py`, `analyze_worldsim_v7_p15_fresh_hazard_action.py`, `run_worldsim_v7_p11_provenance_authority_audit.py` | P3 hard evidence, P3-B camera evidence, consumed and fresh P3-C bidirectional ray certificates, P3-D/P15 action provenance attribution, and P11 provenance-authority audit | Matched-ray free space is respected by construction; target-only aggregate depth/ray/Chamfer and visibility F-score improve on consumed and independently fresh AV2; fresh action attribution separates COMPLETE-driven early returns from KEEP-dominated surface contradictions; Actor identity, trajectory, extent, and hazard are retained | No completeness proof for unseen surfaces; no probabilistic sensor-noise certificate; no per-Actor repair guarantee; nearest-output provenance is not an action ablation; no post-hoc completion carving or no-COMPLETE authority gate; observed-ray provenance is not future-view consistency; Gaussian opacity is not collision occupancy; UNKNOWN is not FREE |
| C2. Validity--hazard factorization and zero-shot selective repair | Method 4.2; Experiments “nuScenes-to-AV2 selective factorization”, “Fresh exact-once selector boundary”, and “Physical authority is not a visibility certificate”; supplement C | `selective_validity_hazard.py`, `sparsity_consistent_selector.py`, `run_worldsim_v7_p10_frozen_physical_authority_audit.py`, `run_worldsim_v7_p12_nuscenes_visibility_authority.py`, `run_worldsim_v7_p13_defer_to_query_composite.py`, `analyze_worldsim_v7_p14_hazard_stratified_defer.py` | P4 primary selector; P8-A fresh nuScenes rejection; P6-C fresh AV2 support; P10--P14 authority/system audits; V7-F19/V7-F20/V7-F22/V7-F23 | nuScenes-only factorization transfers empirically to AV2; a visibility head transfers target-specific ranking; defer-to-query and exact hazard-stratified accounting separate conditional risk, population utility, and physical-group burden while retaining every Actor | No cross-domain conformal/exchangeability guarantee; no AV2 adaptation or post-hoc tuning; P4 does not confidence-separate visibility or reduce hazard-stratum risk; low conditional risk need not improve the fallback world; an AV2 frontier point is not universal dominance |
| C3. Explainable reliability and explicit safety boundaries | Method 4.3--4.4; Experiments “Interpretable safety envelope”, “Frozen continuous and joint reliability”, “Physical repair is not trajectory authority”; C3 reliability table; supplement D--E | `actor_reliability.py`, `boundary_cost_density.py`, `runtime_surface.py`, `analyze_worldsim_v7_p7_interpretable_safety_envelope.py`, `run_worldsim_v7_p7b_geometry_cost_sensitivity.py`, `run_worldsim_v7_p7c_validity_interval_certificate.py` | Frozen V6.7 P182/P183 marginal density, P199/P201 joint-horizon confirmation, P346 authority boundary; P7/P7-B/P7-C explanation; P5/P5-B exact-identity interface | Frozen marginal and joint reliability objects improve proper-score/calibration metrics on distinct fresh scene-level cohorts; sensor-opportunity dependence, analytic clearance sensitivity, and decision intervals remain Actor-specific; physical repairability, geometry sensitivity, and motion uncertainty are distinct | Inherited C3 evidence is not a V7 repair effect; P346 reuses P201 and fails source held-out-H risk, so it is not a cross-horizon or formal calibration guarantee; Integrated Gradients is sensitivity, not causality; P5 retained rows are descriptive, not causal |
| C4. Minimal composed-authority interface | Experiments “Minimal composed-authority utility”; P9 table | `run_worldsim_v7_p9_composed_authority_fixed_lattice.py` | P9 retained-source 2x2 query/HARP-3D x none/P346 audit | Physical surface choice and frozen task authority compose without changing Actor, hazard, or action denominators; authorized retained sets have lower conditional cost/risk | No executed planner, policy learning, closed loop, collision intervention, or causal benefit from physical repair |

### P20/P21 hard-evidence addendum

P20 corrects the later source audit from target-nearest proximity to literal minimum-positive-depth first return, using
`true_first_return_attribution.py`; P21 derives the exact deletion-monotonicity theorem and descriptive safety--surface frontier using
`analyze_worldsim_v7_p21_monotone_safety_boundary.py`. The supported claim is narrow: deletion cannot increase the literal early
predicate for fixed rays. It does not guarantee hit retention, Chamfer, collision freedom, road safety, or fresh transfer. The prior
AV2 P3/P15 metrics retain ownership as frozen target-nearest diagnostics and are not retroactively relabeled first-return evidence.
Supplement Figure~1 is rendered by `plot_worldsim_v7_p21_safety_boundary.py` from the frozen P20/P21 summaries. Its proxy bars
use the already reported P20 legacy rates; it performs no fit, policy selection, or new data read.

## Canonical evidence registry

| Stage | Canonical run | Paper ownership | Failure/boundary ownership |
| --- | --- | --- | --- |
| P3 | `run://worldsim_v7/WS-V7-P3-AV2-HARD-EVIDENCE-01/20260902T143000Z__ray-certified-s0-r3` | 30-log aggregate hard physical evidence | Aggregate evidence is not a per-Actor certificate |
| P3-B | `run://worldsim_v7/WS-V7-P3B-AV2-CAMERA-EVIDENCE-01/20260902T150000Z__camera-evidence-s0-r1` | 30 panels/30 videos and frozen qualitative protocol | V7-F09: 17.19% per-Actor Chamfer worsening remains exposed |
| P3-C | `run://worldsim_v7/WS-V7-P3C-AV2-VISIBILITY-CERTIFICATE-DEV-01/20260902T223000Z__visibility-audit-s0-r1` | Bidirectional target-ray/surface evidence and pooled visibility improvement | V7-F18: only 64.04% of Actors add no visible violation; consumed cohort is descriptive |
| P3-C fresh | `run://worldsim_v7/WS-V7-P3C-AV2-VISIBILITY-CERTIFICATE-FRESH-01/20260902T231500Z__fresh-visibility-s0-r1` | Exact-once 20-log independent visibility confirmation | Aggregate direction replicates; nonnew-visible remains 62.72% and Chamfer-worsened is 19.50%, independently retaining V7-F18 |
| P3-D | `run://worldsim_v7/WS-V7-P3D-AV2-VISIBLE-FAILURE-ATTRIBUTION-01/20260902T224500Z__visible-failure-attribution-s0-r1` | KEEP/PROJECT/COMPLETE attribution of all target rays | Completion yields 14.96 new hits per new early ray and only 13.49% of output contradictions; carving recovery is closed |
| P4 | `run://worldsim_v7/WS-V7-P4-NUSCENES-SELECTIVE-FACTORIZE-01/20260902T161000Z__selective-factor-s70401-r2` | Primary nuScenes-only factorized selector and consumed-AV2 empirical transfer | Coverage/calibration shift is not cross-domain validity |
| P6-C fit + fresh AV2 | `run://worldsim_v7/WS-V7-P6C-SPARSITY-CONSISTENT-SELECTOR-01/20260902T173000Z__sparsity-consistent-s70602-r1` | Source-only sparsity-consistency fit and exact-once 20-log fresh AV2 read | V7-F12 rejects ratio-only recovery; V7-F15 rejects fresh-nuScenes ranking; V7-F19 records fresh-domain reversal and P4 retention |
| P7 | `run://worldsim_v7/WS-V7-P7-INTERPRETABLE-SAFETY-ENVELOPE-01/20260902T163000Z__safety-envelope-s0-r1` | Risk--coverage, score shift, and IG sensor-opportunity explanation | V7-F11: zero leakage does not imply sensor-invariant validity |
| P5 | `run://worldsim_v7/WS-V7-P5-PHYSICAL-RELIABILITY-ALIGNMENT-AUDIT-01/20260902T180000Z__physical-reliability-alignment-s0-r1` | Exact identity join and direct-fit admissibility decision | V7-F13: only two train scenes/five Actors, so joint fitting is rejected |
| P5-B | `run://worldsim_v7/WS-V7-P5B-FROZEN-PHYSICAL-RELIABILITY-INTERFACE-01/20260902T183000Z__frozen-physical-reliability-interface-s0-r1` | Frozen test-Actor descriptive interface | Zero selected false-safe events is descriptive, not a confidence bound |
| P7-B | `run://worldsim_v7/WS-V7-P7B-GEOMETRY-COST-SENSITIVITY-01/20260902T191500Z__geometry-cost-sensitivity-s0-r2` | FP64 deterministic geometry-to-cost bound | V7-F14 records the rejected FP32 tolerance path |
| P8-A | `run://worldsim_v7/WS-V7-P8A-FRESH-NUSCENES-EXACT-ONCE-01/20260902T200000Z__fresh-nuscenes-final-s0-r1` | Fresh nuScenes exact-once candidate rejection; P4 retention | V7-F15: P6-C loses 3.50 repair-AUROC points and has worse AURC |
| P9 | `run://worldsim_v7/WS-V7-P9-COMPOSED-AUTHORITY-FIXED-LATTICE-01/20260902T213000Z__composed-authority-s0-r1` | Minimal retained-source composed-authority result | V7-F16 corrects held-out horizon prose to executable 2.5 s; result is not closed-loop |
| P7-C | `run://worldsim_v7/WS-V7-P7C-VALIDITY-INTERVAL-CERTIFICATE-01/20260902T220000Z__validity-interval-s0-r1` | Actor-level decision intervals and feature-group explanation | V7-F17: 47 stable AV2 false repairs prove stability is not correctness |
| P10 | `run://worldsim_v7/WS-V7-P10-FROZEN-PHYSICAL-AUTHORITY-AUDIT-01/20260903T010000Z__physical-authority-s0-r2` | Exact frozen join of fresh-AV2 P4 authority to bidirectional visibility evidence | V7-F20: selected visible risk 36.39% vs. 37.28% always-repair, but 40.40% upper bound gives no confidence separation |
| P11 | `run://worldsim_v7/WS-V7-P11-PROVENANCE-AUTHORITY-AUDIT-01/20260903T011500Z__provenance-authority-s0-r2` | Deterministic no-COMPLETE provenance witness and P4 conjunction | V7-F21: dual risk improves to 20.16%, but Chamfer-worsening is 43.55% and hazard coverage 3.52%; observed provenance is not joint authority |
| P12 | `run://worldsim_v7/WS-V7-P12-NUSCENES-VISIBILITY-AUTHORITY-01/20260903T004500Z__visibility-authority-s71201-r1` | Frozen source-only visibility head and consumed-AV2 transfer audit | V7-F22: target-specific ranking transfers, but dual coverage 7.46%, Chamfer-worsening 35.90%, and hazard coverage 3.52% reject the head |
| P13 | `run://worldsim_v7/WS-V7-P13-DEFER-TO-QUERY-COMPOSITE-01/20260903T023000Z__defer-to-query-s0-r1` | Frozen accept-or-defer composite-world frontier | V7-F23: provenance defer is dominated by query-only; conditional selected risk is not fallback-system utility |
| P14 | `run://worldsim_v7/WS-V7-P14-HAZARD-STRATIFIED-DEFER-01/20260903T031500Z__hazard-stratified-s0-r1` | Exact hazard/clear decomposition of defer-to-query utility and introduced failures | Hazard-state preservation is not hazard-stratum visible-risk reduction; descriptive consumed-AV2 boundary, no new failure ID |
| P15 | `run://worldsim_v7/WS-V7-P15-FRESH-HAZARD-ACTION-AUDIT-01/20260903T044500Z__fresh-hazard-action-audit-s0-r1` | Fresh hazard/clear target-ray and compiler-action mechanism attribution | COMPLETE dominates new early returns; KEEP dominates surface contradictions; selector does not suppress hazardous completion mechanism; no causal ablation claim |
| P20 | `run://worldsim_v7/WS-V7-P20-TRUE-FIRST-RETURN-AUDIT-01/20260903T124500Z__true-first-return-audit-r1` | Literal source first-return correction across frozen P17/P17R/P19 | Baseline hazard exposure is 8.742%; all deletion policies reduce early events but fail Chamfer; consumed diagnostic only |
| P21 | `run://worldsim_v7/WS-V7-P21-MONOTONE-SAFETY-BOUNDARY-01/20260903T134500Z__monotone-safety-boundary-r1` | Set-inclusion theorem and events-per-hit/mm frontier | Exact early monotonicity; empirical ratios are not formal safety or transfer bounds |
| V6.7 P182/P183 | `run://worldsim_v67/WS-V67-P182-LOG-COST-MIXTURE-DENSITY-01/20260830T150500Z__log-cost-mixture-density-s0-r1`; `run://worldsim_v67/WS-V67-P183-LOG-COST-DENSITY-CONFIRMATION-01/20260830T152500Z__log-cost-density-confirmation-s0-r1` | Inherited marginal log-cost density and disjoint 10-log fresh confirmation | Fresh Brier/calibration gains are scene-level empirical support, not formal probability calibration or a physical-repair effect |
| V6.7 P199/P201 | `run://worldsim_v67/WS-V67-P199-JOINT-HORIZON-RELIABILITY-COPULA-01/20260830T181000Z__joint-horizon-reliability-copula-s0-r2`; `run://worldsim_v67/WS-V67-P201-JOINT-HORIZON-COPULA-CONFIRMATION-01/20260830T184500Z__joint-horizon-copula-confirmation-s0-r2` | Inherited joint-horizon dependence and disjoint 10-log fresh confirmation | Joint-event gains are scene-level only; no session/population, collision, or safety guarantee |
| V6.7 P346 | `run://worldsim_v67/WS-V67-P346-ISOTONIC-MULTICALIBRATED-VISITED-RELIABILITY-01/20260901T134500Z__isotonic-multicalibrated-visited-reliability-s0-r2` | Frozen P9 task authority and q90 development boundary | Reused P201 q90 support coexists with source held-out-H risk 26.71%; not formal multicalibration or cross-horizon stability |

## Result ownership and replacement rules

- `paper/results/results_macros.tex` is the only numeric interface used by manuscript prose. Canonical summaries own macro values;
  values must not be copied from console output or development logs.
- P3/P3-B/P3-C/P3-D own physical aggregate, qualitative, visibility, and provenance evidence. P4 owns the primary selector. P7/P7-B/P7-C own descriptive
  explanation and boundary audits. P5/P5-B/P9 own retained-source interface evidence.
- P8-A owns only the fresh nuScenes exact-once comparison and the decision to reject P6-C as the primary selector.
- The completed P6-C fresh AV2 phase adds an external row and cross-fresh boundary, but it does not alter P4/P8-A models, thresholds,
  cohorts, gates, verdicts, or existing numbers.
- P10 owns the frozen authority--visibility alignment only. It cannot change P4/P6-C scores or thresholds; its negative visible-risk
  verdict coexists with P4's positive Chamfer-worsening reduction.
- P11 closes no-COMPLETE/provenance gating on the consumed cohort. Its narrow measurement witness remains valid, but it cannot be
  used to delete completion, claim future-view completeness, or filter hazardous Actors.
- P12 owns the one frozen source-only visibility-head attempt. Its AV2 risk improvement cannot authorize another seed, architecture,
  feature, source-coverage, or target-threshold sweep on the consumed cohort.
- P13 owns only the frozen composition of existing policies with query fallback. It cannot promote P6-C/P12, select a new utility
  weight, or reinterpret zero introduced failure as absence of pre-existing query contradiction.
- P14 owns only the exact hazard/clear accounting of P13 policies. It cannot calibrate strata, change a selector, claim fairness,
  or reinterpret the deterministic hazard proxy as collision, planning, causal, or real-road safety evidence.
- P15 owns only nearest-output action provenance on the already consumed fresh cohort. It cannot be cited as a blind confirmation,
  action ablation, or proof that zero emitted PROJECT provenance means zero PROJECT effect.
- P20/P21 own the literal source first-return correction and deletion theorem. They do not replace P3/P15 frozen AV2 metrics,
  authorize a failed policy, or establish target transfer, hit/Chamfer monotonicity, collision freedom, or closed-loop safety.
- A negative exact-once result remains in the denominator and paper. It is never replaced by another scene/log, seed, threshold,
  feature group, or checkpoint.

## Final claim audit checklist

- Every paper number resolves to a macro and a canonical summary.
- Every qualitative case resolves through `PROJECT_PAGE_ASSET_INDEX.md` to the frozen P3-B run.
- Figure 1 uses the first non-hazardous and first hazardous frozen main cases by metadata only; each valid/artifact pair
  retains identical Actor, trajectory, extent, camera, and hazard metadata, with no visual-quality selection or new render.
- Every “certificate” names its deterministic object: matched ray, analytic inequality, or network feature box.
- Population false-repair rate is not described as conditional selected risk.
- Empirical zero-shot transfer is not described as conformal, exchangeable, calibrated across domains, or formally safe.
- Retained-source P5/P9 evidence is not described as causal planning, collision, policy, or closed-loop improvement.
- Fresh AV2 language is now bound to the canonical P6-C `summary.json`; the one-shot evaluator exited normally with status `done`.
- Repair-or-abstain is always scoped to frozen Chamfer non-worsening; it is never called a bidirectional visibility certificate.
- Inherited V6.7 P183/P201 rows are labeled scene-level fresh evidence, while P346 is labeled reused development; none is
  described as a V7 repair effect, AV2 transfer, cross-horizon stability, or formal calibration guarantee.
- Target-nearest proximity and literal first-return are named separately; only P20/P21 own the latter operator and theorem.
