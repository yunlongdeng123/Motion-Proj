# P23 Fresh AV2 Literal First-Return Confirmation Result

## Canonical run

`run://worldsim_v7/WS-V7-P23-FRESH-AV2-FIRST-RETURN-CONFIRMATION-01/20260903T183000Z__fresh-av2-literal-first-return-s0-r1`

Status is `done`; verdict is `supported_fresh_av2_literal_first_return_confirmation`. The exact-once read contains ten
metadata-frozen, previously quality-unread AV2 Sensor-val logs, 233 Actors (81 hazardous), and 650,145 target rays. No log was
replaced or deleted.

## Proxy versus literal first return

| Stratum | Actors | Rays | Proxy new-early | Literal new-early | Literal / proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 233 | 650,145 | 8,905 / 1.3697% | 70,220 / 10.8007% | 7.885x |
| hazardous | 81 | 326,432 | 5,837 / 1.7881% | 48,368 / 14.8172% | 8.286x |
| clear | 152 | 323,713 | 3,068 / 0.9478% | 21,852 / 6.7504% | 7.123x |

Both frozen confirmation gates pass: literal rate exceeds the target-nearest proxy for all Actors and for the hazardous stratum.
This independently confirms that the proxy severely understates literal first-return exposure on a disjoint AV2 cohort.

## Action provenance

All 70,220 literal new-early returns have COMPLETE provenance; KEEP/PROJECT contribute zero. Literal new hits total 86,883:
KEEP/PROJECT/COMPLETE contribute 1,392/0/85,491, so COMPLETE supplies 98.398%. The literal new-hit/new-early ratio is 1.237,
close to the consumed P22 value 1.194 and far below the old target-nearest efficiency narrative.

The compiled output has 54,692 KEEP and 8,567 COMPLETE points. Surface contradictions are 178 KEEP and 18 COMPLETE. All Actor
provenance reconstructions align exactly with the compiled surfaces.

## Claim boundary

P23 supports fresh transfer of a measurement failure and completion-action provenance, not fresh validation of a learned model or
policy. It does not select P16/P17/P17R/P19, tune a tolerance, prove collision freedom, execute a planner, or imply road safety.
No scientific failure ID is consumed; the next available ID remains `V7-F30`.

## Resources and completion

The single RTX 3090 run completed in 23.42 s with 0.1838 GiB peak GPU allocation and 1.2103 GiB peak RSS. The downloader recorded
`all_complete logs=10`, and downloader and P23 processes then exited normally. The 10-log download used no retry.
