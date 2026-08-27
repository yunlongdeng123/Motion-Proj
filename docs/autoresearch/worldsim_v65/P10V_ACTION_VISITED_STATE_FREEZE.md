# P10V fixed-action visited-state transfer freeze

Date: 2026-08-28  
Task/Hypothesis: `WS-V65-P10V-ACTION-VISITED-STATE-TRANSFER-01 / WS-V65-H-P10V-001`

## Question and boundary

For each frozen Ego candidate trajectory `tau`, can deterministic Qmean rank the reliability of world states that would be
visited over the next two seconds? This is not another voxel-correctness question and does not train a collision critic.

The result may support action-conditioned visited-world-state representation only. It cannot establish collision avoidance,
planning quality, policy improvement, closed-loop behavior, or safety.

## Independent cohort

Before any P10V target or score read, all scenes mentioned in repository configs/docs/scripts and all processed directories were
excluded. Of 700 direct IR-WM temporal keys, 574 unprocessed and unmentioned scenes remained. To avoid P3C shards 1/5/10,
bands 2/6/9 were frozen; deterministic 1/3 and 2/3 eligible quantiles give:

| archive band | processed index | scene |
| ---: | ---: | --- |
| 2 | 119 | scene-0159 |
| 2 | 143 | scene-0184 |
| 6 | 459 | scene-0577 |
| 6 | 479 | scene-0599 |
| 9 | 722 | scene-0955 |
| 9 | 745 | scene-0983 |

Only shards 2/6/9 are scanned first. Missing members trigger the same-cohort ten-shard capability fallback; scenes are not
replaced. Six scenes times 12 frozen target frames produce 72 source cases.

## Fixed action and target contracts

P10V reuses the V6.4 P11 generator without fitting its failed critic: progress ratios `[0.25, 0.50, 0.75, 1.00]` crossed with
lateral offsets `[-1.5, 0, 1.5]` meters. The stop action is reported as a capability/control row but is not a reliability unit,
because it visits no future world-state footprint.

For each of the 12 non-stop trajectories, use the same 2.0-second, 1.5-meter corridor as P2V. Sample at most 8,192 valid
boundary points per case with seed 0. An action is evaluable only with at least 16 visited points. Prediction is the mean frozen
q0 probability over its visited points; target cost is the fraction of visited points labeled hidden-FREE.

No new critic, attention module, calibration refit, threshold, or parameter sweep is allowed.

## Frozen metrics and gates

- pooled Qmean/target Spearman `>=0.55`;
- unsafe-action AUROC `>=0.80`, where unsafe means at least one hidden-FREE visited point;
- pairwise concordance `>=0.65` for within-case action pairs separated by target cost at least `0.02`;
- within-case lowest-Qmean 25% action cost reduction versus all evaluable actions `>=25%`;
- at least five scenes have non-increasing selected mean cost;
- at least 48 cases have two or more evaluable actions.

All gates must pass. Failure closes action-level visited-state ranking without training a critic or changing the lattice.

VAD (ICCV 2023) and SparseDrive motivate sparse trajectory/scene interaction, but P10V keeps only the bounded fixed lattice and
direct representation test. It deliberately precedes any learned action head.
