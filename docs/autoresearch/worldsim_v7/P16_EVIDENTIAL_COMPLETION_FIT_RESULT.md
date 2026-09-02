# P16 Evidential Completion Fit Result

Date: 2026-09-02

Canonical: `run://worldsim_v7/WS-V7-P16-EVIDENTIAL-COMPLETION-FIT-01/20260903T073500Z__completion-fit-s71601-r2`

## Outcome

The fixed source-only candidate is rejected before any fresh AV2 read. The 85 nuScenes fit Actors provide 1,770 candidates with
FREE/OCCUPIED/UNKNOWN counts `6/1739/25`. On 228 disjoint test Actors and 3,325 candidates, accuracy is 96.93% but macro F1 is
only `.3488`: FREE recall is zero and UNKNOWN F1 is `.0619`. High aggregate accuracy is therefore the occupied-class prior, not
evidence that the two safety-relevant minority states were learned.

## Physical result

The frozen always-COMPLETE source test has mean Chamfer `.1945868 m`, new-early rate `.96618%`, and hazardous new-early rate
`1.43622%`. P16 retains 97.65% of completion candidates yet worsens mean Chamfer to `.1950206 m`, total new-early to `.98886%`,
and hazardous new-early to `1.48110%`; new hits fall from 39,255 to 38,356. Both intended Pareto directions fail on source.

This is not merely class imbalance. Removing an individually uncertain candidate can expose a different, earlier point as the first
return on the same ray. Candidate correctness is therefore non-compositional under first-occupied rendering. Independent point
classification is the wrong prediction object for the mechanism identified by P15.

## Boundary and next step

Register `V7-F24`. Do not run the P16 external phase, do not tune class weights/features/thresholds, and do not consume the third
AV2 cohort. The already-running metadata-frozen download may finish for a future pre-frozen model, but its payload remains unread.
The only justified successor changes the prediction object to joint ray/set occupancy and trains through accumulated transmittance or
first-return depth, following differentiable occupancy rendering rather than another point classifier.
