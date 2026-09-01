# WorldSim V7 P7 interpretable safety envelope freeze

Date: 2026-09-02  
Task: `WS-V7-P7-INTERPRETABLE-SAFETY-ENVELOPE-01`  
Hypothesis: `WS-V7-H-P7-001`  
Status: frozen descriptive analysis before execution

## Motivation

P4 supports a fixed nuScenes-to-AV2 repair-or-abstain operating point, but its coverage changes
from 8.93% on nuScenes calibration to 75.55% on AV2. A single threshold table is insufficient for
reviewers to understand the safety/geometry trade-off or which physical evidence drives repair.

P5 C3 integration is temporarily deferred: only a small subset of P4 physical Actors aligns with
the frozen V6.7 trajectory-density rows, so immediate light retraining would risk a scene/Actor
shortcut rather than demonstrate that physical repair improves task reliability.

## Frozen analysis

- Source is exactly P4 r2 model, standardizers, threshold, and retained calibration/test/AV2 rows.
- No training, refit, recalibration, threshold selection, AV2 adaptation, or new data IO.
- Report the full fixed 0--100% coverage grid for shared and factorized heads on nuScenes test and AV2.
- At every point report population false repair, conditional selected failure, hazard coverage,
  selective Chamfer, and selected geometric gain.
- Mark the already frozen P4 thresholds; curves are descriptive and cannot select a new threshold.
- Decompose the frozen identities
  `population false repair = coverage × conditional selected failure` and
  `selective-query mean = coverage × selected conditional delta`.
- Quantify calibration-to-test score shift with Wasserstein and KS statistics as diagnostics only.
- Explain the factorized validity head with 64-step Integrated Gradients relative to the
  nuScenes-train standardized mean. Report nuScenes test and AV2 separately.

## Interpretation boundary

Integrated Gradients follows the ICML 2017 implementation-invariant path attribution, but remains
model sensitivity relative to a chosen baseline, not a causal statement about sensors or physical
worlds. Structural zero leakage is stronger: the two factorized heads have disjoint computational
graphs, hence repair-score derivatives with respect to hazard inputs and hazard-score derivatives
with respect to validity inputs are exactly zero.

Risk--coverage follows the selective-prediction object; distribution-shift work is used only to
motivate why the AV2 curve must be empirical. This package adds no formal external guarantee,
scientific gates, hash/checksum/fingerprint, smoke matrix, or repeated P4 quality read.
