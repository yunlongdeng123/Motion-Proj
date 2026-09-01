# WorldSim V7 P4 selective factorization result

Date: 2026-09-02  
Task: `WS-V7-P4-NUSCENES-SELECTIVE-FACTORIZE-01`  
Canonical: `run://worldsim_v7/WS-V7-P4-NUSCENES-SELECTIVE-FACTORIZE-01/20260902T161000Z__selective-factor-s70401-r2`  
Verdict: `supported_nuscenes_trained_av2_zero_shot_selective_factorization`

## Outcome

The formal r2 run completed all 63 frozen nuScenes scenes and all 30 frozen AV2 logs. All seven
pre-registered gates passed. The usable Actor corpus contains 29 train, 56 calibration, and 228
nuScenes test Actors, plus 634 AV2 zero-shot Actors. The small train count is retained as a claim
limit rather than repaired by relaxing geometric admissibility.

## nuScenes in-domain result

| model | repair AUROC/AUPRC | hazard AUROC/AUPRC | coverage | population false repair | selective failure | selective Chamfer |
|---|---:|---:|---:|---:|---:|---:|
| shared input | .6426/.7617 | .9166/.8484 | 11.84% | 3.51% | 29.63% | .2309 m |
| factorized | .6491/.7846 | .9811/.9672 | 3.95% | .44% | 11.11% | .2456 m |

The factorized model is non-inferior on repairability and improves hazard AUROC by 6.46 points.
Its two paired input-swap shifts are exactly zero by structure; the shared model shifts repair
scores by .2014 under hazard swaps and hazard scores by .2886 under validity swaps. Mean clean-query,
always-repair, and factorized-selective Chamfer are `.2513/.2047/.2456 m` respectively.

The nuScenes calibration threshold is `.999436`; calibration coverage is 8.93%. The adjusted
population false-repair loss is `.0351 <= .05`. This controls selected failures as a fraction of
the whole population, not the conditional failure rate among selected Actors.

## AV2 zero-shot result

The frozen factorized threshold selects 479/634 Actors (75.55%). It reduces population false repair
from the always-repair failure rate `105/634=16.56%` to `57/634=8.99%`; conditional failure among
selected Actors is 11.90%. Mean Chamfer is `.2577 m` for clean query, `.1770 m` for always repair,
and `.1821 m` for selective repair. Thus selection improves the safe fallback by 29.35%, while
costing 5.03 mm relative to the mean-optimal but less selective always-repair policy.

Zero-shot repairability AUROC/AUPRC are `.6996/.9032`; hazard AUROC/AUPRC are `.9805/.9693`.
Hazard Actor coverage is 92.17%, and selected hazardous-Actor failure is 2.36%, so the selector does
not obtain its result by deleting hazardous Actors. Actor/hazard state retention and both
factorized paired input-swap shifts are exactly one/zero respectively.

## Safety and transfer boundary

Coverage changes from 8.93% on nuScenes calibration to 75.55% on AV2 under the identical threshold.
The empirical AV2 result is favorable, but that score-distribution shift is direct evidence that
nuScenes exchangeability cannot be assumed across datasets. No conformal, conditional, collision,
planning, closed-loop, or road-safety guarantee transfers to AV2. The supported claim is narrower:
a nuScenes-only structurally factorized selector reduces zero-shot AV2 false repair versus always
repair, improves clean-query geometry, retains hazards, and exposes abstention explicitly.

## Resources and artifacts

- wall: `434.29 s`; peak GPU: `.0772 GiB`; peak RSS: `1.463 GiB`.
- run size: `2.7 MiB`; `/root/autodl-tmp` free: about `128 GiB`.
- retained: role Actor JSONLs, AV2 Actor rows, all scores, model/standardizers/thresholds,
  resolved config, summary, and status.
- no hash/checksum/fingerprint, second seed, threshold sweep, AV2 fit, or repeated formal read.
- CVPR draft: P4 method/result/safety boundary and a four-row cross-domain table; 5 pages,
  932,940 bytes after TeX Live compile and page-level visual inspection.

## Pre-quality failure

r1 stopped before point-cloud or method-quality read because raw nuScenes `sample.json` does not
contain the devkit-generated `anns` reverse-index shortcut. r2 used the official
`sample_annotation.sample_token` relationship with no scientific-contract change (`V7-F10`).
