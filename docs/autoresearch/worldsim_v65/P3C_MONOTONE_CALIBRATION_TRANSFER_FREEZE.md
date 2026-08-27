# P3C Independent Monotone Calibration Transfer Freeze

## Trigger and claim

P2V independently supported deterministic Qmean as a trajectory-level visited-world-state reliability ranking. R7 had already frozen a positive-slope two-parameter map that improved train-only expected-error calibration without changing ordering. P3C now asks whether that exact map transfers to a separate unused cohort.

This is one independent calibration-transfer read. It does not refit the map, reuse P2V quality, claim conformal coverage, or change selection ordering.

## Metadata-only cohort selection

The processed data root contains no direct-key scene that is both unconsumed and independent of prior V6.3/V6.4/V6.5 quality work. Therefore P3C uses unprocessed scenes.

Before any target, q0, native, or evidence quality read:

1. Exclude every scene already named in repository configs/docs/scripts or present as a processed directory.
2. Require a direct key in the frozen 700-scene IR-WM temporal metadata.
3. Use the public archive's 85-index capability bands. The retained member index contains 21,345 files from 12 scenes; every scene maps exclusively to its expected band, including five shard-1 scenes and one scene in each audited band 2/4/5/6/8/9/10.
4. Select bands 1, 5, and 10 for early/middle/late index coverage. Within each band's eligible scenes, take the 1/3 and 2/3 deterministic quantiles.

Frozen cohort:

| archive band | processed index | scene |
| ---: | ---: | --- |
| 1 | 29 | scene-0030 |
| 1 | 52 | scene-0055 |
| 5 | 367 | scene-0453 |
| 5 | 393 | scene-0501 |
| 10 | 786 | scene-1046 |
| 10 | 825 | scene-1085 |

Only shards 1/5/10 are scanned first. If any required member is absent, that is a pre-quality capability failure and the fallback is a full ten-shard scan with the same scenes—not replacement scenes.

## Frozen calibrator and evaluation

`calibrated_error = sigmoid(1.703977108001709 * logit(Qmean) - 0.4792216420173645)` with probability clip `1e-6`.

- no fit or parameter update on P3C;
- same future 20-frame / 2-second, 1.5 m trajectory footprint and minimum 16 visited samples as R4/P2V;
- same maximum 8,192 valid boundary samples per unit and seed 0;
- 12 target frames per scene, 72 source units;
- five equal-count calibration bins and matched 40% selected set.

All gates must pass:

- MSE reduction at least 50%;
- five-bin absolute calibration error reduction at least 30%;
- per-scene MSE improves in at least five scenes;
- Spearman and unsafe AUROC do not decrease beyond `1e-6`;
- selected-40% unit indices are exactly unchanged.

References: Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html); Zhang et al., [Instance-Wise Monotonic Calibration by Constrained Transformation](https://proceedings.mlr.press/v286/zhang25c.html); Ma and Blaschko, [Meta-Cal](https://proceedings.mlr.press/v139/ma21a.html).
