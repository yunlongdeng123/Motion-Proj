# P6 Opportunity-Invariant Recovery Freeze

Date: 2026-09-02

## Motivation

P7 exposed `V7-F11`: the factorized P4 validity score is structurally independent of hazard inputs, yet its AV2 sensitivity is
dominated by raw observation-frame count (49.53%). nuScenes-calibration to AV2 score Wasserstein distance is `.2170`, and the AV2
median saturates at `.999992`. This recovery addresses only that diagnosed layer and does not reopen the physical compiler.

## Frozen hypothesis

`WS-V7-H-P6-001`: replacing observation-opportunity counts with dimensionless per-opportunity surface/support ratios will reduce
sensor-domain score shift while retaining nuScenes repairability ordering and the P4 empirical repair-or-abstain benefit.

The candidate removes raw `observation_frame_count`; replaces canonical surfel count with surfels per observation; and divides
temporal/view support counts by observation count. Query-frame density, physical residuals, fractions, ratios, and metric range are
retained. It is one fixed 32-unit MLP with seed `70601`; there is no feature/model/seed sweep. Hazard scoring and the exact disjoint
validity--hazard interface stay frozen from P4.

## Data boundary

- train/model design/standardization: P4 nuScenes train only
- repair threshold: P4 nuScenes calibration role only, alpha `.05`
- nuScenes test: retained independent role
- consumed AV2 v1 30 logs: unavailable for feature, model, threshold, or gate selection
- external recovery read: 20 previously unused AV2 val logs in `av2_zero_shot_recovery_cohort_v1.json`

The fresh cohort is metadata-only: sort all 150 UUIDs, remove consumed indices `0,5,...,145`, and choose every sixth item of the
120-log complement. This gives original indices `1,8,16,...,143` and was frozen before any recovery score or quality read.

## One-read decision

The candidate must be within `.02` repair AUROC of P4 factorized on nuScenes test, preserve transformed-feature invariance under
fixed `.5x/2x` opportunity interventions, cover at least 10% of fresh AV2 Actors, reduce fresh population false repair relative to
always repair, keep fresh selective Chamfer no worse than clean query, and reduce calibration-to-fresh-AV2 score Wasserstein
distance relative to the frozen P4 head evaluated on the same fresh rows. These are the only gates.

Passing supports an opportunity-normalized empirical transfer result, not a formal external risk or road-safety guarantee. Failure
closes this single recovery without threshold tuning on the fresh cohort.
