# P17 Ray-Set Completion Freeze

Date: 2026-09-02

## Motivation and migration

`V7-F24` shows that independently correct completion candidates are not compositionally correct under first-occupied rendering.
P17 migrates the accumulated-transmittance interface used by SelfOcc (CVPR 2024), the official OccFlowNet differentiable rendering
pipeline (WACV 2025), and OpenOcc/OccNet ray evaluation: a ray terminates at point `i` with weight
`T_i alpha_i`, where `T_i=product_{j<i}(1-alpha_j)`.

## Frozen model and source protocol

Reuse P16's 11 pre-target, dimensionless Actor-local candidate features but discard its three-state point labels and weighted
cross-entropy. One `64-64-1` ReLU MLP with seed `71701` outputs candidate occupancy logits. For every held-out nuScenes target ray,
the frozen KEEP/PROJECT core provides the terminal fallback and all completion candidates before it compete jointly in depth order.
Forward selection is hard at sigmoid `.5`; gradients use the straight-through sigmoid. The only loss is mean-per-Actor Smooth-L1
between rendered first-return depth and held-out LiDAR depth.

Fit uses the same nuScenes train+calibration roles, 160 epochs, Actor batch 8, at most 1024 deterministically spaced influential rays
per Actor, AdamW `.001/.0001`. nuScenes test remains disjoint. There is no architecture/seed/loss/ray-budget/threshold sweep.

## Decision and external boundary

Source must strictly lower hazardous new-early rate and make population Chamfer no worse than frozen always-COMPLETE. Failure
closes P17 before external data. Passing freezes the checkpoint and authorizes exactly one read of the already metadata-frozen 10-log
AV2 cohort after download completion, with the same two Pareto directions. KEEP/PROJECT, geometry tolerances, Actor state, cohort,
and failed-case denominator remain unchanged. The result is empirical ray/geometry transfer, not calibrated occupancy or safety proof.
