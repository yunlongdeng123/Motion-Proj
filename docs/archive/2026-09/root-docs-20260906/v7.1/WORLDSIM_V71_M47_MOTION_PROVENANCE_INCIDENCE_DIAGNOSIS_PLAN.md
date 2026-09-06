# WorldSim V7.1 M47 — Motion / provenance / incidence diagnosis

## Question

M45 and M46 improve hit and clear-ray early-return behavior but regress hazard early return. M47 asks whether that split is explained by rigid actor motion, supervision provenance, or oriented-kernel incidence rather than by insufficient local surface optimization.

## Frozen protocol

- Reuse the exposed 66-actor holdout exactly; no training, model selection, threshold selection, or decision gate.
- Freeze M39 categorical anchor/child authority and M46 CDF-supervised normal/thickness support.
- Join the already compiled M31 physical trajectory and KEEP/PROJECT contradiction attribution by `(scene_name, track_id)`.
- Report M46-minus-M39 early/hit changes for moving/quasi-static and their hazard/clear intersections; fixed incidence bins use `|ray·normal| < 0.35`, `[0.35, 0.70)`, and `>= 0.70`.
- Measure child-normal incidence at the GT target using normalized categorical component responsibility. This is diagnosis only: incidence, motion, provenance, and hazard are never model inputs and cannot delete or reweight a prediction.
- Report actor-level Pearson associations with displacement, prior projected-anchor contradiction fraction, anchor early rate, incidence, and child responsibility. These are descriptive associations, not causal claims.

## Decision after diagnosis

- If degradation concentrates in moving actors, stop asking local support to absorb pose and introduce an explicit rigid dynamic layer with trajectory supervision.
- If it concentrates in projected-anchor provenance, correct the producer supervision before changing geometry.
- If it concentrates at grazing incidence, make visibility/incidence part of the supervised return measure, not a post-hoc filter.
- If none dominates, treat the failure as a joint identifiability problem and move to a higher-level latent completion/scene-flow factorization.

M43 remains the only frozen external candidate and its partial AV2 quality is not read.
