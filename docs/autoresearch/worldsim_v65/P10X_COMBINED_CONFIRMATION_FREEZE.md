# P10X one-shot combined confirmation freeze

Date: 2026-08-28  
Task/Hypothesis: `WS-V65-P10X-COMBINED-CONFIRMATION-01 / WS-V65-H-P10X-001`

## Candidate and boundary

The candidate is final before this cohort is prepared:

1. raw Qmean over the world states visited by the nominal future Ego trajectory;
2. the exact R7 monotone map `sigmoid(1.7039771080 * logit(Qmean) - 0.4792216420)`, without refit;
3. the V6.4 fixed 12-non-stop-action lattice scored by mean Qmean over each trajectory footprint.

P10X is one combined exact empirical confirmation. It cannot support population guarantees, collision avoidance, planning or
policy improvement, closed-loop behavior, or safety. Failure closes the combined candidate; no second confirmation cohort is
allowed.

## Independent cohort and I/O

After excluding every repository-mentioned scene and processed directory, 568 unprocessed direct-key scenes remain. Use the
deterministic 1/3 and 2/3 eligible quantiles in previously unused archive bands 3/7/8:

| band | processed index | scene |
| ---: | ---: | --- |
| 3 | 194 | scene-0245 |
| 3 | 229 | scene-0287 |
| 7 | 538 | scene-0686 |
| 7 | 563 | scene-0718 |
| 8 | 634 | scene-0817 |
| 8 | 657 | scene-0868 |

Scan only shards 3/7/8 first. Missing members permit only a same-cohort ten-shard fallback, never scene replacement. The input
contract remains six scenes, 12 target frames each, maximum 8,192 boundary points per case, seed 0, future 2.0 seconds,
1.5-meter corridor, and minimum 16 visited points.

## One read and six gates

The nominal action is the frozen progress-1.0/lateral-0 action index 7. Report nominal route Spearman, raw/calibrated MSE and
five-bin calibration error. Report all P10V action metrics and scene/case support, but gate only the six core claims:

- nominal route Spearman `>=0.60`;
- frozen-map nominal route MSE reduction `>=50%`;
- pooled action Spearman `>=0.55`;
- unsafe-action AUROC `>=0.80`;
- within-case pairwise concordance `>=0.65` for target gaps `>=0.02`;
- lowest-Qmean 25% action cost reduction `>=25%`.

No model/head/critic/calibrator fit, lattice or threshold sweep, second confirmation, hash, checksum, or fingerprint is allowed.
