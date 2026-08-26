# P10C Route-Local Conflict Severity Audit Freeze

Date: 2026-08-26

## Saturation and migration

P10R's binary route interception is 96/96 for both C0 and M0. It cannot express whether the 375 additional route-support cells are useful occupancy or hidden FREE. The metric must not be retuned after this read.

The recovery follows cell-level occupancy evaluation in the Waymo Occupancy Flow metrics, continuous planner-location occupancy queries in Implicit Occupancy Flow Fields, and continuous soft collision potentials in occupancy-grid planning:

- https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/occupancy_flow_metrics.proto
- https://openaccess.thecvf.com/content/CVPR2023/html/Agro_Implicit_Occupancy_Flow_Fields_for_Perception_and_Prediction_in_Self-Driving_CVPR_2023_paper.html
- https://openaccess.thecvf.com/content/CVPR2023W/E2EAD/papers/Kedia_Integrated_Perception_and_Planning_for_Autonomous_Vehicle_Navigation_An_Optimization-Based_CVPRW_2023_paper.pdf

V64-F20 is `recovery_frozen_pre_target_audit`.

## Frozen task

Task: `WS-V64-P10C-ROUTE-CONFLICT-AUDIT-01`

Run: `20260826T184500Z__route-conflict-audit-s0-r1`

Hypothesis: `WS-V64-H-P10C-001`

Use the unchanged P10M C0/M0 emitted states and the unchanged P10R two-second, 20-frame, 1.5 metre route corridor. Read the already locked target evidence once and label route-emitted native voxels as hidden FREE or not. Do not refit the model, change policy, change route parameters, or select cases.

## Gates and claim boundary

Only two gates are used: M0 emits a positive number of additional route-local voxels, and pooled M0 route-local hidden-FREE conflict is at most 0.05. Per-case threshold failures are descriptive, not a third gate.

This is a route-local target conflict severity audit. Target evidence is read; physical collision ground truth and a planner are not. No collision, counterfactual planning, closed-loop, or safety claim is authorized. No hash, checksum, fingerprint, smoke matrix, regression matrix, or parameter sweep is added.
