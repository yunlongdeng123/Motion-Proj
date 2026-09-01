# P6-C Source Sparsity-Consistent Selector Freeze

Date: 2026-09-02

## Motivation

P6-B ratio normalization obtains exact opportunity invariance but fails the nuScenes source non-inferiority gate (`.60728` vs
the frozen `.62908` floor). `V7-F12` shows that deleting evidence amount is too lossy. The fresh AV2 cohort has not been compiled
or scored, so a new source-only hypothesis can still be frozen without target feedback.

Single Domain Generalization for LiDAR Semantic Segmentation (CVPR 2023) treats unseen sensor configurations by randomly
subsampling source scans and enforcing sparsity-invariant feature consistency. An Empirical Study of the Generalization Ability of
LiDAR 3D Object Detectors to Unseen Domains (CVPR 2024) independently finds source-domain augmentation useful for lower-resolution
sensors. V7 migrates only their common source-only principle to the low-capacity Actor validity head.

References:

- https://openaccess.thecvf.com/content/CVPR2023/html/Kim_Single_Domain_Generalization_for_LiDAR_Semantic_Segmentation_CVPR_2023_paper.html
- https://openaccess.thecvf.com/content/CVPR2024/html/Eskandar_An_Empirical_Study_of_the_Generalization_Ability_of_Lidar_3D_CVPR_2024_paper.html

## Frozen candidate

`WS-V7-H-P6-002` retains the 13 P4 validity inputs, including interpretable evidence amount. For every nuScenes training Actor it
creates fixed `.5x` and `.75x` opportunity views by jointly scaling observation-frame count, canonical surfel count, temporal
support, and view support. Physical residuals, query-frame density, fractions, ratios, labels, and hazard inputs are unchanged.

One hidden-32 validity MLP (`seed=70602`, 80 epochs) minimizes supervised BCE on original and both augmented views plus weight-1
probability consistency between each pair. The standardizer is fit on the combined source training views. Calibration uses only
original nuScenes calibration rows at alpha `.05`. No feature/factor/weight/model/seed sweep is allowed.

## Two-stage boundary

Fit must pass both before any fresh AV2 compilation:

1. nuScenes-test repair AUROC is within `.02` of frozen P4 factorized;
2. mean absolute repair-score shift under `.5x/.75x` interventions is at most `.70` times the frozen P4 shift on identical rows.

Only then may the same run read the frozen 20-log fresh AV2 cohort once. External gates remain coverage at least 10%, population
false repair below always repair, selective Chamfer no worse than clean query, and calibration-to-fresh score Wasserstein below
frozen P4 on the same rows. Failure at fit closes P6-C without consuming fresh quality. Failure at external closes the candidate
without target threshold tuning.
