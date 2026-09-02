# P16 Evidential Completion Responsibility Freeze

Date: 2026-09-02

## Diagnosed mechanism

P15 shows that COMPLETE explains 94.70% of new early target-ray returns and 99.94% of new target hits. Hazardous Actors have
1.951x the clear-Actor new-early rate, while P4 and P6-C leave the hazardous rate at 1.0055x and 1.0010x of always repair.
Actor-level selection therefore cannot repair the candidate-level completion mechanism.

## Literature migration

P16 adopts only the common physical interface of OccupancyM3D (CVPR 2024), evidential occupancy mapping (CVPR 2024), EvOcc
(CVPR 2025), and object-centric temporal occupancy completion (NeurIPS 2024): every completion candidate has an explicit
FREE/OCCUPIED/UNKNOWN target-ray state, and geometry is judged by the first occupied return against held-out LiDAR. It does not
import a camera backbone, a scene-scale occupancy grid, or an uncertainty/safety guarantee.

## Frozen research direction

Train one small candidate-level network on nuScenes train+calibration only, retaining the previous nuScenes test role as disjoint
source evaluation. Inputs must be available before the held-out target read and describe
object-local position, query-hole geometry, reflection support, temporal support, and view diversity. Labels come from disjoint
held-out nuScenes LiDAR rays: OCCUPIED for target-supported and non-early candidates, FREE for contradicted early-return candidates,
and UNKNOWN otherwise. The action rule has no target-tuned threshold: only argmax OCCUPIED emits COMPLETE; FREE/UNKNOWN retain
UNKNOWN. KEEP and PROJECT remain exactly frozen.

The network is fixed before source execution: 11 dimensionless/source-observable features; a `64-64-3` ReLU MLP; seed `71601`;
120 epochs; batch 512; AdamW learning rate `.001`, weight decay `.0001`; and inverse-square-root source-frequency cross-entropy
weights normalized to mean one. This is a single candidate, not a seed/architecture/loss/feature sweep.

The model family, features, source split, loss, seed, and epoch count must be committed before the new AV2 cohort is evaluated.
Existing 30-log and 20-log AV2 cohorts are unavailable for fit, feature selection, thresholds, or candidate recovery. This is an
empirical source-to-target test, not calibrated probability, causal action ablation, or road-safety certification.

## Fresh external cohort and resources

`configs/worldsim_v7/av2_evicomp_fresh_cohort_v1.json` contains 10 metadata-only logs: sort the 150 official val UUIDs, remove
the previous 50 consumed logs, and select positions 0,10,...,90 from the 100-log complement. Download is one log at a time through
the official public S3 bucket, stops before free space falls below 75 GiB, and never launches a second downloader. The observed
historical footprint implies roughly 11 GiB for this cohort, within the current 108 GiB free space. A single RTX 3090 is sufficient.

## Decision surface

The primary comparison is the unchanged always-COMPLETE compiler versus P16 on the same fresh candidate set. Report composite
Chamfer gain, target-hit gain, new-early rate, first-return depth error, hazard/clear decomposition, COMPLETE coverage, and explicit
UNKNOWN mass. Support requires lower hazardous new-early rate without worse population composite Chamfer than the frozen
always-COMPLETE baseline; all
other outcomes narrow the mechanism claim. No threshold, tolerance, feature, or policy sweep is allowed after any fresh AV2 read.
