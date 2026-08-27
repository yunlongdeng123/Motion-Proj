# P10V fixed-action visited-state transfer result

Date: 2026-08-28  
Hypothesis: `WS-V65-H-P10V-001`  
Verdict: `supported_fresh_fixed_action_visited_state_ranking`

## Canonical run

`run://worldsim_v65/WS-V65-P10V-ACTION-VISITED-STATE-TRANSFER-01/20260828T003000Z__action-transfer-s0-r1`

The fixed lattice contains 12 non-stop trajectories per case: four progress ratios crossed with three lateral offsets. Stop was
excluded by the preregistered prediction-object contract because it visits no future world-state footprint.

Of 864 source actions, 813 met the frozen minimum of 16 visited points and 51 were excluded. All 72 cases retained at least two
evaluable actions. The eligible actions contained 55,411 visited points, 6,826 hidden-FREE outcomes, and 659 unsafe actions.

## Result

| metric | value | frozen gate |
| --- | ---: | ---: |
| pooled Qmean/target Spearman | 0.740235 | >=0.55 |
| unsafe AUROC / AUPRC | 0.858779 / 0.945415 | AUROC >=0.80 |
| pairwise count / concordance | 2,834 / 0.732534 | >=0.65 |
| evaluable cases | 72 | >=48 |
| all / lowest-Qmean-25% cost | 0.109772 / 0.0732644 | reduction >=25% |
| relative selected-cost reduction | 33.26% | pass |
| scene lower/equal/higher | 6/0/0 | non-increasing >=5 |

All six gates passed. The run wrote 813 per-action rows for later subgroup and paper analysis; no action head or critic was
trained, and the compact cache was newly materialized rather than reused.

## Interpretation and boundary

The positive result supports the revised prediction object: given an Ego trajectory, frozen Qmean ranks the reliability of the
world states that trajectory would visit over the next two seconds, including within a bounded candidate action set. It does not
support collision prediction, planner or policy improvement, closed-loop behavior, or safety.

No lattice, threshold, head, calibrator, or footprint sweep follows on this consumed cohort. The only allowed next stage is a
single combined empirical confirmation on a new cohort.

Resources: one RTX 3090, wall `8.459s`, peak allocated GPU memory `0.03917GiB`, peak RSS `1.001GiB`.
