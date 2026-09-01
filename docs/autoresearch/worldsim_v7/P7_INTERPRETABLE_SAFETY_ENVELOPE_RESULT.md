# P7 Interpretable Safety Envelope Result

Date: 2026-09-02

## Verdict

- canonical: `run://worldsim_v7/WS-V7-P7-INTERPRETABLE-SAFETY-ENVELOPE-01/20260902T163000Z__safety-envelope-s0-r1`
- conclusion: `frozen_descriptive_interpretable_safety_envelope`
- source: frozen P4 r2 model, standardizer, thresholds, and retained Actor rows
- interventions: no training, refit, recalibration, AV2 adaptation, or threshold selection

P7 preserves the P4 7/7 gate verdict as an empirical nuScenes-trained AV2 zero-shot operating point, but rejects the stronger
interpretation that its validity score is domain invariant. The factorized computation graph still gives exact zero validity--hazard
cross-input derivative. That structural statement does not imply that the validity inputs themselves are stable across sensors.

## Frozen operating point

At the unchanged factorized threshold `0.999436`, coverage is `8.93%` on nuScenes calibration, `3.95%` on nuScenes test, and
`75.55%` on AV2. AV2 population false repair remains `8.99%`, conditional selected failure `11.90%`, hazard coverage `92.17%`,
and selective Chamfer `.182073m`. Risk and geometry factorization identities close to numerical precision (`<=1.5e-8`).

## Score shift

The factorized repair score mean/median/95th percentile is `.7614/.9453/.9997` on nuScenes calibration,
`.8283/.9792/.9992` on nuScenes test, and `.9783/.999992/1.000000` on AV2. Relative to calibration, nuScenes-test
Wasserstein/KS distance is `.06794/.16949`; AV2 is `.21702/.70415`. The KS p-value is descriptive only and is not used as a
gate. The AV2 distribution is nearly saturated, which explains the coverage jump under the unchanged threshold.

## Integrated Gradients

Integrated Gradients uses 64 steps and the nuScenes-train standardized mean as baseline. Mean absolute completeness residual is
`.00211` on nuScenes test and `.01286` on AV2. The leading normalized absolute attributions are:

| Feature | nuScenes test | AV2 zero-shot |
|---|---:|---:|
| observation frame count | 18.25% | **49.53%** |
| canonical surfels | **20.91%** | 14.06% |
| completion candidate fraction | 11.62% | 5.61% |
| sensor range | 5.26% | 7.38% |
| unknown query fraction | 7.31% | 4.33% |

IG is model sensitivity, not causal evidence. Nevertheless, the threefold rise in observation-window reliance together with score
saturation is a concrete sensor-opportunity shortcut diagnosis (`V7-F11`), not merely an unspecified calibration gap.

## Scientific boundary and recovery

The existing 30 AV2 logs are consumed. They may continue to support the frozen P3 physical evidence and descriptive P4/P7
zero-shot result, but cannot validate a post-hoc normalized selector. A recovery candidate must replace raw opportunity counts with
dimensionless density/support ratios or otherwise enforce sparsity invariance using nuScenes-only development. Its external claim
must be read once on a metadata-frozen, previously unused AV2 cohort or Waymo; no threshold may be selected on the consumed logs.

## Resources and retained artifacts

- wall: `1.564s`
- device: RTX 3090 / CUDA
- peak GPU: `.0173GiB`
- peak RSS: `.815GiB`
- run size: `348KiB`
- retained: 21-point coverage curves, operating-point decomposition, score-shift statistics, Integrated Gradients JSON/figure,
  summary, and status

