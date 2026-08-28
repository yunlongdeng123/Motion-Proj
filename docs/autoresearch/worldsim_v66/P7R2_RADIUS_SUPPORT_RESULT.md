# P7R2 one-native-voxel support recovery result

Canonical run:
`run://worldsim_v66/WS-V66-P7R2-RADIUS-SUPPORTED-ACTOR-REPAIR-01/20260828T094232Z__radius-surface-repair-s0-r1`.

P7R2 was the sole pre-frozen recovery of the exact-hit P7R repair. It retained the same P7 L0 action set, target-only
evaluation and nine gates, and changed only same-Actor hit support from exact evidence-voxel identity to a fixed
`0.512m` nearest-hit radius (one native voxel side). No radius, budget or threshold sweep was performed.

| Metric | P7R exact hit | P7R2 radius support | P7R2 gate |
|---|---:|---:|---|
| conflict reduction | 0.847660 | 0.417872 | FAIL (`>=0.50`) |
| overall boundary retention | 0.383588 | 0.617684 | PASS (`>=0.40`) |
| clean boundary retention | 0.395715 | 0.619549 | PASS (`>=0.40`) |
| Actor/shell/track/trajectory retention | 1 | 1 | PASS |
| removed Actors | 0 | 0 | PASS |
| hazard proxy shift | 0 | 0 | PASS |
| scene yield | 1 | 1 | PASS |

P7R2 repaired 14,565 of 23,580 boundary points and retained 13,881 of 22,405 clean points, but retained 684 of 1,175
conflict points. Therefore 8/9 gates passed and the verdict is
`rejected_conflict_reduction_after_single_recovery`.

This closes the sensor-supported surface-repair family as a terminal negative. Intermediate radii, gate relaxation,
new action budgets and a completion-model recovery are prohibited. The supported P7 result remains fixed-budget
triage only; physical repair, RL-ready distribution and P9 claims remain locked.
