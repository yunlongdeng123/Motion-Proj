# P15 Fresh Hazard-by-Action Attribution Freeze

## Question

Which frozen 3D compiler action explains the extra ray-termination burden on hazardous Actors, and does P4/P6-C selection actually remove that mechanism or merely select a different mixture of Actors?

## Literature migration

- Savinov et al. (CVPR 2016) treat visibility as a viewing-ray consistency constraint rather than an unordered point metric.
- EvOcc (CVPR 2025) evaluates occupancy by casting each LiDAR ray to the first predicted occupied surface and comparing termination depth/semantics.
- Ren et al. (ICCV 2021) distinguish earlier versus later occupancy consequences under occlusion. P15 migrates only the asymmetric early-termination report, not their planning or safety claim.

## Frozen execution

1. Reuse the existing P3-D CUDA attribution runner on the already consumed 20-log fresh AV2 cohort with the unchanged P2 compiler, observed-hit PROJECT output, `0.20 m` lateral tolerance, and `0.20 m` depth tolerance. Raw run ID is fixed to `20260903T043000Z__fresh-hazard-action-raw-s0-r1`.
2. Exact-join its Actor rows to the frozen P6-C `FRESH_AV2_SCORES.jsonl` and report `always`, P4 selected/abstained, and P6-C selected/abstained scopes.
3. Within hazardous and clear strata, report target-ray denominators, new-early and new-hit rates, resolved query early rate, surface contradictions, and KEEP/PROJECT/COMPLETE shares and hit-to-early ratios.

The raw stage is one bounded single-3090 run; the join is CPU-only. There is no training, fit, calibration, threshold change, policy search, new cohort, or deletion.

## Provenance caveat

The compiler concatenates KEEP before PROJECT and voxel-deduplicates the combined output. An observed-hit PROJECT that shares a voxel with KEEP is therefore attributed to KEEP. `PROJECT=0` in the emitted provenance is a deterministic deduplication consequence, not causal proof that PROJECT creates zero harm.

## Claim boundary

This cohort was already consumed by P3-C/P6-C and provides a mechanism audit, not another independent confirmation. Nearest-output attribution is descriptive, not a counterfactual action intervention. Hazard is the existing deterministic proxy, not collision ground truth, planner behavior, or a real-road safety label. No pass/fail gate is added; a new failure is registered only for a real contradiction with an existing claim. Next available ID remains `V7-F24`.
