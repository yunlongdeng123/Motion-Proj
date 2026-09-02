# P22 Consumed AV2 Literal First-Return Correction Result

## Status

- Canonical: `run://worldsim_v7/WS-V7-P22-AV2-TRUE-FIRST-RETURN-CORRECTION-01/20260903T160000Z__consumed-av2-true-first-return-s0-r1`
- Verdict: `descriptive_consumed_av2_literal_first_return_metric_correction`
- Cohort: 20 already consumed AV2 Sensor validation logs
- Actors / target rays: 523 / 1,435,391
- Fresh third AV2 cohort read: false

## Proxy-to-literal correction

| Stratum | Actors | Proxy new-early | Literal new-early | Proxy rate | Literal rate | Multiplier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 523 | 19,354 | 142,022 | 1.3483% | 9.8943% | 7.338x |
| hazardous | 142 | 10,854 | 79,074 | 1.9115% | 13.9261% | 7.285x |
| clear | 381 | 8,500 | 62,948 | 0.9797% | 7.2556% | 7.406x |

The source P20 correction was about sixfold; the same operator exposes a roughly 7.3-fold underestimate on AV2. This is
cross-sensor agreement of the measurement failure, not an independent model-transfer test.

## Literal action attribution

- All 142,022 newly early literal returns are attributed to COMPLETE.
- Literal new matched hits are 169,581: KEEP 3,658, PROJECT 0, COMPLETE 165,923.
- Literal new-hit/new-early ratio is 1.194, replacing the proxy ratio 14.51 for first-return interpretation.
- Output surface points are KEEP 146,014, PROJECT 0, COMPLETE 17,855.
- Surface contradictions remain a different reverse-direction channel: KEEP 487 and COMPLETE 22.
- Query/compiled early counts are 475,138/608,120; 9,040 query-early rays are resolved.
- Provenance reconstruction is aligned for every Actor.

The zero PROJECT count remains a KEEP-first voxel-deduplication artifact and is not evidence of causal zero effect.

## Claim correction

P3/P15 target-nearest rates remain valid only as proximity diagnostics. They must not be called first-return, ray-termination,
or visibility-safety evidence. Literal first return shows that completion trades 142,022 newly early rays for 169,581 newly
matched hits on this consumed cohort, not approximately fifteen hits per early event.

The result does not select a deletion policy, alter P4, or authorize the unread third AV2 cohort. It is not a fresh confirmation,
formal domain-generalization guarantee, collision metric, planning result, or road-safety certificate.

## Resources

The unique run completed all 20 logs in 46.32 s on one RTX 3090, with peak GPU allocation 0.1844 GiB and peak RSS
1.2255 GiB. No training, fitting, calibration, thresholding, policy change, failed-log deletion, or second run was used.
