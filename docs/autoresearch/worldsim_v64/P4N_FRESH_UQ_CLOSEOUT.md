# P4N fresh native-voxel UQ closeout

- Task: `WS-V64-P4N-FRESH-NATIVE-VOXEL-UQ-01`
- Hypothesis: `WS-V64-H-P4N-001`
- Canonical run: `run://worldsim_v64/WS-V64-P4N-FRESH-NATIVE-VOXEL-UQ-01/20260826T091500Z__fresh-native-voxel-uq-s0-r2`
- Verdict: `supported_relative_only_weak_absolute`

## Frozen denominator and execution

The run used the preregistered unique native occupied-boundary voxels: the six-neighbor boundary of native argmax OCC union method-observed OCC, restricted to method UNKNOWN, non-contradictory, target-ROI-valid voxels. Four fit scenes supplied 200,000 points; the two untouched evaluation scenes supplied 333,009 points, including 27,495 hidden-FREE positives (prevalence `0.082565`). The boundary-global diagonal GMM-4 used the frozen 17-D logits plus 256-D BEV, StandardScaler, PCA-16, and seed 0.

The CPU-only run completed in `22.3767 s`, with peak RSS `1.0705 GiB`; no multi-GPU resource was required.

## Result

| Scope | points | prevalence | best U0 AUROC | U2 AUROC | best U0 AUPRC | U2 AUPRC | U2 FPR@95TPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| pooled | 333,009 | 0.082565 | 0.435498 | 0.518545 | 0.070965 | 0.085650 | 0.964709 |
| scene-0359 | 173,908 | 0.107879 | 0.430968 | 0.498387 | 0.092352 | 0.104552 | 0.965465 |
| scene-0998 | 159,101 | 0.054896 | 0.482815 | 0.498295 | 0.052462 | 0.056673 | 0.960623 |

The frozen relative gates pass: pooled AUROC gain is `+0.083047 >= 0.02`, and U2 AUROC exceeds the best U0 in both scenes (`2/2`). At 50% pooled coverage, U2 risk is `0.076917`, below prevalence `0.082565` and the best U0 risk `0.096827`.

## Claim boundary

This is only relative fresh mechanism support. Both within-scene U2 AUROCs are slightly below 0.5, FPR@95TPR remains about 0.96, scene-0359 AUPRC is below its prevalence, and scene-0998 50%-coverage risk is worse than its prevalence. The pooled AUROC can therefore be partly driven by scene-level prevalence/score shifts rather than usable within-scene ranking. The run does not support authority, calibration, conditional coverage, or a safety claim.

No GMM component, PCA dimension, seed, denominator, or gate sweep is allowed on these evaluation scenes. `V64-F10` records the active algorithm/evaluation limitation. Following the supervised/hybrid uncertainty direction used by OCCUQ, ReliOcc, and EvOcc, the only unlocked recovery is a separately preregistered fit-only hidden-FREE risk head, evaluated once on the identical two-scene denominator.

