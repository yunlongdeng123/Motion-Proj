# P12 nuScenes-Only Visibility Authority Freeze

Date: 2026-09-02

## Question

Can a low-capacity head trained on the actual nuScenes held-out-ray visibility target separate fresh-AV2 visible failures,
where the Chamfer-trained P4 score and deterministic no-COMPLETE witness fail?

## Fixed mechanism

- Recompute visibility labels only on the already frozen P4 nuScenes train/calibration/test scene split, using the unchanged
  four-action compiler and P3-C 0.20 m bidirectional ray partition.
- Train one 13-input, hidden-16 MLP on the 11 nuScenes train scenes only. Inputs are the existing interpretable P4 validity
  features; label is `nonnew_visible_violation`. Seed 71201, 80 epochs, and all optimizer settings are fixed once.
- Set the visibility threshold by the top 25% of scores on the disjoint nuScenes calibration scenes. The target coverage is
  fixed before labels are generated and does not optimize an AV2 metric.
- Report the visibility head alone and `P4 AND visibility-head` on nuScenes test, then on the already consumed 20-log AV2
  development join. P4 scores and decisions remain frozen.
- Seven gates are fixed before source labels or aggregate results: source-test safe-visible AUROC >= .55; source selected risk
  below source always; external dual risk and its 95% upper below P4 risk; external dual Chamfer tail no worse than P4;
  external dual coverage >= 10%; external hazard coverage >= 50%.
- No architecture, seed, feature, epoch, coverage, loss, gate, or threshold sweep. No AV2 fit or recalibration.

This is the only learned recovery after V7-F20/V7-F21. The AV2 cohort is consumed development evidence; a positive result
cannot enter the primary claim without a newly frozen untouched cohort. A failure closes the visibility-head family.
