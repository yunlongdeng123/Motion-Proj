# WorldSim V7.1 M50 — frame-balanced first-return supervision

Date: 2026-09-05

## Motivation

M8 makes target-to-surface coverage frame-balanced, but its literal first-return/free-before-hit loss samples pooled target
rays. Dense target frames can therefore dominate physical supervision even though canonical geometry is expected to remain
valid under every read-only rigid Actor pose. M47 showed that a moving/static label does not explain the residual, and M48
closed learned visibility. The remaining narrow question is whether equal target-frame weighting improves temporal physical
closure without adding another state to geometry.

Primary references support explicit separation rather than a shared shortcut: DynamicVGGT predicts motion under scene-flow
supervision, while DeGO separates rigid and non-rigid deformation. We do not import their visual foundation teachers or add
non-rigid capacity because the current vehicle corpus has annotated rigid poses but no credible deformation target.

- DynamicVGGT, CVPR 2026: <https://openaccess.thecvf.com/content/CVPR2026/html/He_DynamicVGGT_Learning_Dynamic_Point_Maps_for_4D_Scene_Reconstruction_in_CVPR_2026_paper.html>
- DeGO, CVPR 2026: <https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Deformable_Gaussian_Occupancy_Decoupling_Rigid_and_Nonrigid_Motion_with_Factorized_CVPR_2026_paper.html>

## Frozen intervention

- initialize exactly from frozen M8;
- keep the four-child canonical geometry, immutable anchors, set/plane/scale and frame-coverage losses;
- group GT target rays by canonicalized sensor origin and average literal first-return plus free-before-hit loss equally over
  target frames;
- normalize by the frozen M8 frame-balanced reference, then use the existing two-objective PCGrad update;
- geometry input remains build-only and excludes time, velocity, trajectory, hazard, category, image, semantics, and visibility;
- annotated rigid pose remains a read-only canonical-to-world transform;
- four epochs, one seed, one run; no weight/seed/margin sweep.

## Frozen read

Primary metric: Actor-mean worst-frame early-return rate versus frozen M8. The candidate is supported only if this improves
and one aggregate Pareto guard also holds: hazardous early rate regresses by at most 0.1 percentage point, all-ray hit recall
by at most 0.5 point, and Actor-mean Chamfer by at most 0.5 mm. These are two decisions, not independent opportunities for
post-result selection.

The run is development-exposed and cannot replace M39 or alter the frozen M43 external candidate. It never reads partial AV2
quality. If rejected, frame-balanced physical fine-tuning closes without changing ray groups, loss weights, seed, or epochs.
