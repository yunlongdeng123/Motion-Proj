# P10X one-shot combined confirmation result

Date: 2026-08-28  
Verdict: `rejected_one_shot_combined_visited_state_confirmation`

Canonical run:

`run://worldsim_v65/WS-V65-P10X-COMBINED-CONFIRMATION-01/20260828T013000Z__combined-confirmation-s0-r1`

## Frozen inputs and exposure

The run used the sole frozen six-scene confirmation cohort, the frozen V6.4 q0 model, the no-refit R7 monotone map, and
the fixed 12-action non-stop lattice. The compact cache did not exist and was materialized in this run. No head, critic,
calibrator, action lattice, threshold, or scene was changed after reading quality. This is the only P10X confirmation read;
a second cohort is not allowed.

## Nominal-route reliability

Of 72 source cases, 60 nominal routes met the frozen 16-point footprint minimum; all 12 nominal routes in `scene-0718`
were below that minimum. Raw Qmean retained useful ordering (`Spearman=0.609813`) and unsafe ranking
(`AUROC/AUPRC=0.988868/0.997419`). The frozen monotone map reduced MSE from `0.0318414` to `0.00592580`
(`-81.39%`) and 5-bin absolute calibration error from `0.159217` to `0.0203975` (`-87.19%`) while preserving ranking.
Scene MSE lower/equal/higher was `4/0/1` across five evaluable scenes.

## Fixed-action reliability and decision boundary

The run materialized 864 source actions. The frozen footprint rule excluded 125, leaving 739 eligible actions with
80,282 visited points, 10,818 hidden-FREE outcomes, and 577 unsafe actions.

| metric | value | frozen gate |
| --- | ---: | --- |
| pooled Qmean-target Spearman | 0.772946 | pass (`>=0.55`) |
| unsafe AUROC / AUPRC | 0.972730 / 0.991627 | pass (`AUROC>=0.80`) |
| pairwise pairs / concordance | 2,216 / 0.655686 | pass (`>=0.65`) |
| evaluable cases | 65 | descriptive |
| all / selected-25% actual cost | 0.120215 / 0.100520 | `-16.38%`, **fail** (`>=25%`) |
| action-selection scene lower/equal/higher | 5/0/1 | descriptive |

Five of six core gates passed. The selected-quarter cost reduction missed its preregistered magnitude gate, so the combined
candidate is rejected even though reliability ranking and calibration transferred strongly. `scene-0817` was the one scene
whose selected cost increased (`0.118135` to `0.122824`); this row is retained rather than gated away.

## Interpretation and stop rule

The result supports the narrower empirical interface: given an Ego trajectory `tau`, frozen Qmean ranks whether the world
states visited over the next two seconds are reliable, and the frozen monotone map improves expected-error calibration.
It does **not** support compiling that score directly into action selection with the required decision benefit. This separation
matches recent decision-calibration work that distinguishes miscalibration regret from grouping-loss regret: calibration alone
need not improve downstream decisions.

P10X is terminal. No second confirmation, scene replacement, threshold relaxation, lattice sweep, or critic recovery is
authorized. No collision, planner, policy, closed-loop, population, or safety claim is made.

Resources: wall `8.218s`, peak allocated GPU `0.03917GiB`, peak RSS `1.052GiB`; one RTX 3090 was sufficient.
