# P20 True First-Return Audit Freeze

Date: 2026-09-02

## Audit defect

The current visible-failure attribution finds the Euclidean-nearest compiled point for each held-out target point and then tests that
point's projected depth/lateral distance. This is a useful proximity proxy but not the physical first return of an occupied surface:
a nearer positive-depth point on the same ray can be ignored when it is farther in Euclidean distance from the target.

The CVPR 2024 evidential occupancy evaluation explicitly marches each raw LiDAR ray and compares the first occupied intersection to
the measured depth. P20 adopts that operator directly: among compiled points with positive projected depth and lateral distance within
the already-frozen tolerance, take minimum depth; classify early/hit with the already-frozen depth tolerance. Query and compiled
surfaces use the identical operator.

## Frozen audit

Recompile only the consumed nuScenes test role once and evaluate the already-frozen P17, P17R, and P19 policies. Architectures,
checkpoints, `.5` threshold, P19 one-slot action, KEEP/PROJECT points, Chamfer implementation, tolerances, and source scenes do not
change. Report true first-return new-early/new-hit by hazard stratum alongside unchanged Chamfer. No policy is trained or selected
during the audit.

The role is diagnostic, not new confirmation, and fresh AV2 remains unread. Each variant is marked Pareto-valid only if hazardous
true-first-return new-early strictly decreases and population Chamfer is no worse than always-COMPLETE. No tolerance/ray/operator
sweep is allowed; an all-fail outcome registers `V7-F29`.

Reference: [Accurate Training Data for Occupancy Map Prediction in Automated Driving Using Evidence Theory, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Kalble_Accurate_Training_Data_for_Occupancy_Map_Prediction_in_Automated_Driving_Using_Evidence_Theory_CVPR_2024_paper.html).
