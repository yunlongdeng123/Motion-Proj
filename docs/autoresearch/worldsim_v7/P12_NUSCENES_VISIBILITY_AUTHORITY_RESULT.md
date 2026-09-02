# P12 nuScenes-Only Visibility Authority Result

Date: 2026-09-02

Canonical run:

```text
run://worldsim_v7/WS-V7-P12-NUSCENES-VISIBILITY-AUTHORITY-01/20260903T004500Z__visibility-authority-s71201-r1
```

## Result

The frozen source-only head uses 13 inputs, one 16-unit hidden layer, seed 71201, and 80 epochs. Its threshold is fixed by
the top 25% source-calibration score coverage. The source train/calibration/test sets contain 29/56/228 Actors; source-test
safe-visible AUROC is 0.753. It selects 108/228 Actors (47.37%) and reduces source-test visible-failure risk from 36.40% to
19.44% (one-sided 95% Wilson upper 26.42%).

On the already consumed 20-log/523-Actor AV2 development join, without target fitting or threshold selection, the visibility
head has safe-visible AUROC 0.625. It selects 43 Actors (8.22%) with 11.63% visible-failure risk, 22.02% Wilson upper,
37.21% Chamfer-worsening, and only five hazardous Actors. The final P4-and-visibility authority selects 39 Actors (7.46%)
with 10.26% visible-failure risk, 20.98% upper, 35.90% Chamfer-worsening, and 3.52% hazard coverage.

## Decision

Source ranking/risk and external visible-risk point/upper gates pass. Chamfer non-regression, minimum 10% external coverage,
and minimum 50% hazard coverage fail. The frozen verdict is `rejected_visibility_targeted_source_head`; register `V7-F22`
and close this visibility-head family rather than sweep architecture, seed, source coverage, features, or AV2 threshold.

The result shows genuine zero-shot transfer of a target-specific visibility ranking, but it also exposes the safety boundary:
low visible risk alone is not valid authority when geometry worsens and hazardous Actors are almost entirely rejected. Any
future positive confirmation would require a newly frozen untouched cohort, not reuse of these consumed AV2 rows.
