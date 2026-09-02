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
| C1. Actor-preserving physical 3D compilation | Method 4.1; Experiments “Zero-shot Actor-local surface evidence”, “Four-action physical compilation”, “Frozen camera evidence”, and “Visibility-conditioned physical evidence”; supplement A--B/F | `av2_canonical_surface.py`, `av2_four_action_compiler.py`, `av2_p3_hard_evidence.py`, `av2_p3b_camera_evidence.py`, `visibility_certificate.py`, `visible_failure_attribution.py` | P3 hard evidence, P3-B camera evidence, consumed and fresh P3-C bidirectional ray certificates, and P3-D all-Actor provenance attribution | Matched-ray free space is respected by construction; target-only aggregate depth/ray/Chamfer and visibility F-score improve on consumed and independently fresh AV2; completion's hit-gain/early-tail is quantified; Actor identity, trajectory, extent, and hazard are retained | No completeness proof for unseen surfaces; no probabilistic sensor-noise certificate; no per-Actor repair guarantee; no post-hoc completion carving; Gaussian opacity is not collision occupancy; UNKNOWN is not FREE |
| C2. Validity--hazard factorization and zero-shot selective repair | Method 4.2; Experiments “nuScenes-to-AV2 selective factorization”, “Fresh exact-once selector boundary”; supplement C | `selective_validity_hazard.py`, `sparsity_consistent_selector.py` | P4 primary factorized selector; P8-A fresh nuScenes rejection; P6-C fresh AV2 external support; V7-F19 cross-fresh reversal | nuScenes-only factorization has zero structural cross-input leakage and empirically transfers to AV2 with repair-or-abstain; P6-C improves fresh-AV2 AUROC/Chamfer but loses fresh-nuScenes ranking, so P4 remains primary | No cross-domain conformal/exchangeability guarantee; no AV2 adaptation; no post-hoc threshold/radius/feature tuning; an external gate pass is not universal dominance |
| C3. Explainable reliability and explicit safety boundaries | Method 4.3--4.4; Experiments “Interpretable safety envelope”, “Physical repair is not trajectory authority”; supplement D--E | `actor_reliability.py`, `boundary_cost_density.py`, `runtime_surface.py`, `analyze_worldsim_v7_p7_interpretable_safety_envelope.py`, `run_worldsim_v7_p7b_geometry_cost_sensitivity.py`, `run_worldsim_v7_p7c_validity_interval_certificate.py` | P7 score/IG audit; P7-B FP64 geometry-to-cost sensitivity; P7-C actor-level interval audit; P5/P5-B exact-identity interface | Sensor-opportunity dependence, analytic clearance sensitivity, and frozen-network decision intervals are measurable and Actor-specific; physical repairability, geometry sensitivity, and motion uncertainty are distinct | Integrated Gradients is sensitivity, not causality; standardized feature boxes are not calibrated sensor noise; a robust network decision is not a correct repair or road-safety certificate; P5 retained rows are descriptive, not causal |
| C4. Minimal composed-authority interface | Experiments “Minimal composed-authority utility”; P9 table | `run_worldsim_v7_p9_composed_authority_fixed_lattice.py` | P9 retained-source 2x2 query/HARP-3D x none/P346 audit | Physical surface choice and frozen task authority compose without changing Actor, hazard, or action denominators; authorized retained sets have lower conditional cost/risk | No executed planner, policy learning, closed loop, collision intervention, or causal benefit from physical repair |

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

## Result ownership and replacement rules

- `paper/results/results_macros.tex` is the only numeric interface used by manuscript prose. Canonical summaries own macro values;
  values must not be copied from console output or development logs.
- P3/P3-B/P3-C/P3-D own physical aggregate, qualitative, visibility, and provenance evidence. P4 owns the primary selector. P7/P7-B/P7-C own descriptive
  explanation and boundary audits. P5/P5-B/P9 own retained-source interface evidence.
- P8-A owns only the fresh nuScenes exact-once comparison and the decision to reject P6-C as the primary selector.
- The completed P6-C fresh AV2 phase adds an external row and cross-fresh boundary, but it does not alter P4/P8-A models, thresholds,
  cohorts, gates, verdicts, or existing numbers.
- A negative exact-once result remains in the denominator and paper. It is never replaced by another scene/log, seed, threshold,
  feature group, or checkpoint.

## Final claim audit checklist

- Every paper number resolves to a macro and a canonical summary.
- Every qualitative case resolves through `PROJECT_PAGE_ASSET_INDEX.md` to the frozen P3-B run.
- Every “certificate” names its deterministic object: matched ray, analytic inequality, or network feature box.
- Population false-repair rate is not described as conditional selected risk.
- Empirical zero-shot transfer is not described as conformal, exchangeable, calibrated across domains, or formally safe.
- Retained-source P5/P9 evidence is not described as causal planning, collision, policy, or closed-loop improvement.
- Fresh AV2 language is now bound to the canonical P6-C `summary.json`; the one-shot evaluator exited normally with status `done`.
