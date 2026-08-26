# P10R3 Fixed Route-Denominator Audit Freeze

Date: 2026-08-27  
Task: `WS-V64-P10R3-FIXED-DENOMINATOR-AUDIT-01`  
Hypothesis: `WS-V64-H-P10R3-001`  
Run ID: `20260827T013000Z__fixed-denominator-audit-s0-r1`

P10R2 fresh confirmation passed the absolute M1 CVaR gate but did not confirm relative improvement over M0. M1 emitted fewer
route voxels, so conflict rate among only selected voxels has a smaller and arm-dependent denominator. This audit does not
change or rescue the frozen policy.

Waymo Occupancy Flow evaluates occupancy on a fixed ego-centric grid using cell-level AUC and Soft-IoU, and Implicit Occupancy
Flow supports fixed spatial queries. The migrated diagnostic therefore uses, per case and arm:

`route hidden-FREE conflict count / route-eligible voxel count`.

The route-eligible count is fixed between M0 and M1 within each case. The same empirical worst10/96 mean is computed separately
for the consumed calibration cohort and fresh confirmation cohort. Aggregate conflict counts, selected counts, pooled fixed-
denominator density, and M1-minus-M0 CVaR are reported.

This is explicitly a post-hoc, rows-only exploratory diagnostic. It rereads no target, model, or evidence, changes no policy,
and has no confirmatory gate. Directional consistency is descriptive and cannot close V64-F25 or unlock P11. No denominator,
tail, route, scene, or threshold sweep is allowed.
