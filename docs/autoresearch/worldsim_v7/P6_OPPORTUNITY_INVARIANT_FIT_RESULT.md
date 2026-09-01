# P6 Opportunity-Invariant Fit Result

Date: 2026-09-02

## Verdict

- canonical: `run://worldsim_v7/WS-V7-P6-OPPORTUNITY-INVARIANT-SELECTOR-01/20260902T170000Z__opportunity-invariant-s70601-r1`
- fit status: `fit_rejected_external_not_read`
- external AV2 quality read: false
- failed gate: nuScenes-test repair AUROC non-inferiority

The ratio-only representation is exactly invariant to the frozen `.5x/2x` opportunity intervention, but loses too much source-domain
repairability ordering. It is rejected before any fresh AV2 Actor row is compiled or scored.

## nuScenes-only result

The candidate trains on 29 Actors and calibrates on 56 Actors. Its calibration threshold is `.998570`, adjusted population risk
`.03509`, and coverage `14.29%`. On 228 nuScenes test Actors it obtains repair AUROC/AUPRC/Brier=
`.60728/.74010/.23857`, versus frozen P4 repair AUROC `.64908`. The preregistered floor is `.62908`, so the gate fails.

At the frozen threshold, nuScenes-test coverage is `13.16%`, population false repair `4.82%`, conditional selected failure
`36.67%`, and selective Chamfer `.23155m` versus clean query `.25130m`. The opportunity-feature shift is exactly `0`.

## Interpretation

Dividing all surfel and support counts by observation count removes both the identified shortcut and legitimate information about
evidence amount. This is the expected cost of a non-invertible invariant representation: invariance alone need not preserve the
label relationship. The result closes the ratio-only normalization candidate; it does not invalidate the P3 physical compiler or
the descriptive P4/P7 results.

## Literature-guided next hypothesis

CVPR 2023 Single Domain Generalization for LiDAR Semantic Segmentation addresses sensor-pattern shift with source-domain random
subsampling plus sparsity-invariant feature consistency, rather than deleting density information. CVPR 2024's empirical study of
LiDAR detector generalization likewise finds source-domain low-resolution augmentation useful. V7 therefore freezes a separate
source-only sparsity-consistency candidate: retain the interpretable raw features, expose fixed opportunity-subsampled views during
nuScenes training, and penalize score disagreement. The fresh 20-log AV2 cohort remains untouched and can serve its one external read.
