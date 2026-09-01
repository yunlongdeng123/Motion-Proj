# P6-C Source Sparsity-Consistent Fit Result

Date: 2026-09-02

## Verdict

- canonical: `run://worldsim_v7/WS-V7-P6C-SPARSITY-CONSISTENT-SELECTOR-01/20260902T173000Z__sparsity-consistent-s70602-r1`
- fit status: `model_frozen_waiting_fresh_av2`
- fit verdict: passed, 2/2 gates
- fresh AV2 quality read: false

The source-only candidate preserves nuScenes repairability ordering within the frozen non-inferiority margin while reducing score
sensitivity to fixed opportunity subsampling by 89.33% relative to P4. The model and nuScenes-calibrated threshold are frozen.

## nuScenes result

On 228 nuScenes test Actors, repair AUROC/AUPRC/Brier is `.63239/.75872/.22203`. Frozen P4 AUROC is `.64908` and the
preregistered floor is `.62908`, so the candidate passes by `.00331`. Hazard AUROC remains the frozen factorized P4 value `.98114`.

The `.998561` nuScenes-calibrated threshold covers `19.64%` of calibration Actors with adjusted population risk `.03509`. On test,
coverage is `24.12%`, population false repair `6.14%`, conditional selected failure `25.45%`, and selective Chamfer `.22195m`
versus clean query `.25130m` and always repair `.20471m`.

## Opportunity intervention

Under the frozen `.5x/.75x` opportunity views, mean absolute repair-score shift is `.019526` for P6-C and `.183026` for the frozen
P4 head on identical nuScenes-test rows. The ratio is `.10668`, far below the frozen `.70` limit. This is source intervention
robustness, not an external-domain guarantee.

## Boundary

The fit used 29/56/228 nuScenes train/calibration/test Actors, one seed, one consistency weight, and no sweep. No consumed or fresh
AV2 Actor row was loaded. The next legal action is an exactly-once external phase after all 20 metadata-frozen fresh logs complete.
Passing that phase would support empirical opportunity-robust transfer only.

## Resources

- wall: `1.375s`
- peak GPU: `.01694GiB`
- peak RSS: `.9720GiB`
