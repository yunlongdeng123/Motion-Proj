# P3C frozen monotone calibration transfer result

Date: 2026-08-28  
Hypothesis: `WS-V65-H-P3C-001`  
Verdict: `supported_independent_monotone_visited_state_calibration_transfer`

## Canonical run

`run://worldsim_v65/WS-V65-P3C-MONOTONE-CALIBRATION-TRANSFER-01/20260827T155000Z__calibration-transfer-s0-r2`

The exact R7 map was applied without refitting:

`sigmoid(1.7039771080 * logit(Qmean) - 0.4792216420)`

Of 72 source units, 60 met the frozen minimum of 16 visited samples. These units contained 6,675 visited points,
708 hidden-FREE outcomes, and 48 unsafe trajectories. All 12 units from `scene-1046` were below the footprint
minimum, leaving five evaluable scenes; this exclusion followed the frozen rule and was not a post-read scene choice.

## Result

| metric | raw Qmean | frozen monotone | change |
| --- | ---: | ---: | ---: |
| MSE | 0.0287445 | 0.00207044 | -92.80% |
| MAE | 0.162039 | 0.0379486 | -76.58% |
| 5-bin absolute calibration error | 0.162039 | 0.0189368 | -88.31% |
| Spearman | 0.715491 | 0.715491 | 0 |
| unsafe AUROC | 0.982639 | 0.982639 | 0 |
| unsafe AUPRC | 0.995763 | 0.995763 | 0 |

Scene MSE lower/equal/higher was `5/0/0`. The lowest-risk 40% contained the exact same 24 units before and after
calibration, with mean actual cost `0.0298324`. All six preregistered gates passed.

## Recovery disclosure

Formal r1 failed before any unit, target, score, metric, or cache read because its config mixed a nonexistent V6.4 run
with the wrong model-relative directory. `V65-F17` records this. R2 changed only the locator to the exact frozen q0
artifact already used by P2V; the model and every scientific contract remained unchanged.

## Claim boundary

This supports independent transfer of a monotone expected visited-error calibration map. It is not a conformal
guarantee and does not establish admission, planning, collision avoidance, or safety. No sweep or refit follows.

Resources: one RTX 3090, wall `7.923s`, peak allocated GPU memory `0.02359GiB`, peak RSS `1.195GiB`.
