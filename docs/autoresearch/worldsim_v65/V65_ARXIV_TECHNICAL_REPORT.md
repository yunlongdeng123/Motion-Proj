# From Voxel Confidence to Visited-State Reliability

## Task-Conditioned Uncertainty Evaluation for Driving World Models

- Version: WorldSim V6.5
- Documentation freeze: 2026-08-28
- Branch: `research/worldsim-v6.5-task-conditioned-authority`
- Research state: `v65_research_complete_arxiv_report_ready`
- Scientific execution added by this report update: none
- Terminal result: visited-state reliability evaluation supported; direct fixed-action authority rejected

## Abstract

Uncertainty scores in driving world models are usually attached to local voxels or state elements, although downstream decisions
depend only on the states that an Ego trajectory will visit. WorldSim V6.5 studies whether a frozen native uncertainty score can be
compiled into a task-conditioned reliability interface. Local voxel, map-context, Actor, learned-admission, and smooth-tail variants
failed their frozen transfer contracts. We therefore change the prediction object from “is this voxel correct?” to “if Ego executes
trajectory `tau`, what fraction of world states visited over the next two seconds will be unreliable?” A deterministic mean of the
frozen pointwise score transfers to a fresh six-scene cohort with Spearman `0.63396` and unsafe AUROC `0.99415`; selecting its lowest
40% reduces realized visited-state error by `49.25%`. A two-parameter monotone map, frozen on train-only data, reduces independent
fresh-cohort MSE by `92.80%` and 5-bin calibration error by `88.31%` without changing ranking. Reliability ranking also transfers to
a fixed 12-action lattice. However, the sole combined confirmation rejects direct action authority: five of six gates pass, but
lowest-quarter action selection reduces realized cost by only `16.38%`, below the preregistered `25%`. The defensible result is a
task-conditioned reliability evaluator and expected-error calibrator, not a planner, collision critic, policy, or safety authority.

## 1. Problem formulation

Let `q0(x)` be the frozen V6.4 pointwise uncertainty score for a world-state sample `x`. For an Ego candidate trajectory `tau`, define
the visited footprint over horizon `H=2s` with corridor radius `r=1.5m`:

```text
V(tau) = {x : distance(x_xy, tau_0:H) <= r}.
```

The realized target is the hidden-FREE fraction within that footprint:

```text
y(tau) = sum[x in V(tau)] 1[hidden-FREE(x)] / |V(tau)|.
```

The deterministic reliability statistic is:

```text
Qmean(tau) = mean[x in V(tau)] q0(x).
```

A trajectory is called unsafe for ranking evaluation when at least one visited sample is hidden-FREE. This binary label is only an
evaluation view of the continuous target; it is not a physical collision label. Trajectories with fewer than 16 visited samples are
excluded by a frozen footprint rule.

The frozen expected-error calibrator is a strictly monotone two-parameter map:

```text
Qcal(tau) = sigmoid(1.703977108 * logit(clip(Qmean(tau))) - 0.479221642).
```

Strict monotonicity preserves rankings and selected sets. It can improve expected-cost calibration but cannot by itself improve
within-case action ordering.

## 2. Research progression

V6.5 used progressive preregistration and changed prediction objects only after a frozen negative result. It did not sweep multiple
architectures or thresholds on the same consumed cohort.

| stage | prediction object or mechanism | result | consequence |
| --- | --- | --- | --- |
| P1/P2 | trajectory-conditioned residual over local state | fresh ranking invariant | close original residual family |
| P2R | hard Actor collision outcome | zero positive support | move to continuous Actor-time cost |
| P2C | continuous Actor-time forecast | time-aware arm worse than snapshot | close Actor-token family |
| P4T | learned admission/coverage | one held-out risk violation | no differentiable admission compiler |
| P1R3 | map-conditioned voxel reliability | map readable, no authority gain | reject voxel-level object |
| P1R4 | trajectory-visited world-state rate | strong train-only result | freeze visited-state object |
| P1R5 | Actor false-safe companion | trajectory extrema break transfer | retain world-state object only |
| P1R6 | smooth-tail aggregation | any-error AUROC up, expected cost worse | retain deterministic mean |
| P1R7 | monotone expected-cost calibration | strong train-only calibration | freeze slope/bias |
| P2V | fresh visited-state transfer | supported | independent ranking evidence |
| P3C | independent frozen calibration | supported | independent calibration evidence |
| P10V | fresh fixed-action ranking | supported | one combined confirmation allowed |
| P10X | one-shot combined confirmation | rejected, 5/6 gates | close direct action authority |

The central scientific pivot is P1R3 to P1R4: semantic context could enter the computation, but it did not improve the per-voxel
authority decision. The prediction object was therefore changed to the reliability of the states actually visited by a trajectory.

## 3. Data roles and exposure control

The canonical positive and terminal cohorts are disjoint:

| role | scenes | use |
| --- | --- | --- |
| visited-state fresh selection | `0001/0219/0402/0594/0822/1110` | P2V ranking and selection |
| independent calibration | `0030/0055/0453/0501/1046/1085` | P3C frozen-map transfer |
| fixed-action fresh selection | `0159/0184/0577/0599/0955/0983` | P10V action ranking |
| one-shot confirmation | `0245/0287/0686/0718/0817/0868` | P10X final AND verdict |

P10X allowed one quality read and no second confirmation. The action lattice and all six core gates were frozen before its native and
evidence inputs were scored. No exact-test cohort, closed-loop rollout, or population-level evaluation was executed.

## 4. Main results

### 4.1 Fresh given-trajectory reliability

P2V used 72 source units. Nine failed the frozen 16-point footprint minimum, leaving 63 eligible trajectories with 8,862 visited
samples, 1,055 hidden-FREE outcomes, and 57 unsafe trajectories.

| metric | value |
| --- | ---: |
| Qmean-target Spearman | `0.633963` |
| unsafe AUROC / AUPRC | `0.994152 / 0.999390` |
| all / selected-40% realized cost | `0.102965 / 0.0522594` |
| relative selected-cost reduction | `49.25%` |
| scene lower/equal/higher | `5/1/0` |

This supports a bounded empirical claim: frozen Qmean ranks the hidden-FREE rate over future world states visited by a given
trajectory on this fresh cohort.

### 4.2 Independent expected-error calibration

P3C evaluated the frozen two-parameter map on a new six-scene cohort. Sixty of 72 source units were eligible. `scene-1046` had no unit
meeting the frozen footprint minimum, so scene-level MSE was evaluated on the other five scenes without replacing the scene.

| metric | raw Qmean | frozen monotone map | change |
| --- | ---: | ---: | ---: |
| MSE | `0.0287445` | `0.00207044` | `-92.80%` |
| 5-bin calibration error | `0.162039` | `0.0189368` | `-88.31%` |
| Spearman | `0.715491` | `0.715491` | unchanged |
| unsafe AUROC / AUPRC | `0.982639 / 0.995763` | same | unchanged |
| scene MSE lower/equal/higher | n/a | `5/0/0` | all evaluable scenes improve |

The result is expected-error calibration, not conformal coverage or a risk guarantee.

### 4.3 Fixed-action reliability

P10V reused a fixed non-stop action generator: four progress ratios `[0.25, 0.50, 0.75, 1.00]` crossed with three lateral offsets
`[-1.5, 0, 1.5]m`. Stop was excluded because it does not visit a future footprint. Of 864 source actions, 813 were eligible.

| metric | value |
| --- | ---: |
| pooled Qmean-target Spearman | `0.740235` |
| unsafe AUROC / AUPRC | `0.858779 / 0.945415` |
| pairwise concordance, 2,834 pairs | `0.732534` |
| all / selected-25% realized cost | `0.109772 / 0.0732644` |
| relative selected-cost reduction | `33.26%` |
| scene lower/equal/higher | `6/0/0` |

This is fixed-lattice representation ranking. No critic or collision label was trained.

### 4.4 One-shot combined confirmation

P10X materialized 864 source actions on its sole confirmation cohort. The frozen footprint rule excluded 125, leaving 739 actions,
80,282 visited samples, 10,818 hidden-FREE outcomes, and 577 unsafe actions.

| frozen gate | observed | threshold | result |
| --- | ---: | ---: | --- |
| nominal-route Spearman | `0.609813` | `>=0.60` | pass |
| frozen-map MSE reduction | `81.39%` | `>=50%` | pass |
| action Spearman | `0.772946` | `>=0.55` | pass |
| action unsafe AUROC | `0.972730` | `>=0.80` | pass |
| pairwise concordance | `0.655686` | `>=0.65` | pass |
| selected-25% cost reduction | `16.38%` | `>=25%` | **fail** |

The action-selection scene split was `5/0/1`; `scene-0817` worsened from `0.118135` to `0.122824`. Under the preregistered AND rule,
five passing gates cannot override the failed decision-benefit gate. The terminal verdict is
`rejected_one_shot_combined_visited_state_confirmation`.

## 5. Calibration is not decision authority

P10X exposes the main boundary of V6.5. The frozen map reduced route MSE from `0.0318414` to `0.00592580` and 5-bin calibration error
from `0.159217` to `0.0203975`, while action ranking remained strong. Nevertheless, direct lowest-score action selection delivered
less benefit than required.

This is consistent with decision-calibration literature that separates regret caused by global miscalibration from regret caused by
grouping loss or unresolved conditional structure. Recalibration can improve probability estimates without supplying all information
needed for optimal downstream decisions. V6.5 does not retrofit a local calibrator, decision-focused head, or new critic after seeing
this result; any such method requires a new version and untouched evaluation cohort.

## 6. Negative results and failure taxonomy

Scientific negatives are part of the result, not discarded trials:

- task-risk residual ranking did not transfer (`V65-F03`);
- hard Actor collision supervision had zero positive support (`V65-F04`);
- Actor-time and Actor false-safe formulations did not transfer (`V65-F06`, `V65-F11`);
- learned admission violated held-out risk (`V65-F07`);
- map semantics did not rescue voxel-level authority (`V65-F09`);
- a learned calibration head damaged selection ranking (`V65-F10`);
- smooth-tail aggregation improved any-error AUROC but worsened expected cost (`V65-F12`);
- the one-shot direct-action candidate missed its decision-benefit gate (`V65-F19`).

Input, runtime, and artifact-entry failures `V65-F01/F02/F05/F08/F13-F18` are documented separately with exposure audits. They are
not counted as method negatives. P2V r2 and P3C r2 are valid narrow recoveries because the failed entries disclosed no aggregate
scientific metric and changed only tensor shape or artifact location.

## 7. System and resource result

All V6.5 work fit one RTX 3090. Native workers peaked at `4.1314GiB`; P10V/P10X formal evaluators allocated only about `0.0392GiB`.
Archive I/O, not GPU capacity, was the dominant systems bottleneck. The final pipeline overlapped three archive scans, two scene-ready
preprocess workers, two native workers, and a four-worker evidence job. P10X preparation found 10,718 members on targeted shards
3/7/8 without full-shard fallback; 48 evidence units were produced while later scenes were still being preprocessed/inferred.

Canonical V6.5 run artifacts occupy approximately 16GB. At report freeze, `/root/autodl-tmp` retained about 95GB free space.
Multi-GPU execution was not required.

## 8. Limitations and claim boundary

The report supports only empirical reliability ranking and expected-error calibration for the frozen model, footprints, action set,
and nuScenes cohorts. It does not establish:

- Actor-state reliability;
- physical collision probability or collision avoidance;
- planner, policy, control, closed-loop, or deployment authority;
- conformal coverage or a population guarantee;
- exact-test performance or a safety guarantee;
- superiority of all task-conditioned uncertainty methods over voxel uncertainty;
- failure of all learned critics, local calibration, or decision-focused learning.

## 9. Reproducibility and evidence audit

Immediately before the report handoff, the P2V, P3C, P10V, and P10X canonical `summary.json` and `status.json` files were checked for
existence and JSON readability using the project environment. P10V retained 813 `ACTION_ROWS.jsonl` records. Mandatory ledgers were
terminal and the branch was synchronized before documentation edits. This audit did not recompute metrics, execute smoke/regression
tests, or add hashes, checksums, or fingerprints.

Source-of-truth order:

1. canonical run summaries, statuses, resolved configs, and retained per-action rows;
2. `docs/EXPERIMENTS.md` for chronological experiment results;
3. `docs/RESEARCH_FAILURES.md` for failure wording and non-repetition locks;
4. `docs/RESEARCH_STATUS.md` and `AUTORESEARCH_STATE.current.json` for terminal state;
5. `ARXIV_EVIDENCE_INDEX.md`, `V65_RESEARCH_CLOSEOUT.md`, and this report for authoring navigation.

## 10. Conclusion

WorldSim V6.5 finds that local uncertainty becomes substantially more useful when queried through the future states a candidate Ego
trajectory will visit. The resulting deterministic mean score supports fresh reliability ranking, and a frozen monotone map supports
independent expected-error calibration. Yet these properties do not automatically confer action-selection authority. The terminal
one-shot confirmation preserves that distinction: task-conditioned reliability evaluation transfers, direct fixed-action authority
does not meet its required decision-benefit contract.

## References

- Farid et al., [Task-Relevant Failure Detection for Trajectory Predictors in Autonomous Vehicles](https://proceedings.mlr.press/v205/farid23a.html), CoRL 2022.
- Rhinehart et al., [PRECOG: PREdiction Conditioned on Goals in Visual Multi-Agent Settings](https://openaccess.thecvf.com/content_ICCV_2019/html/Rhinehart_PRECOG_PREdiction_Conditioned_on_Goals_in_Visual_Multi-Agent_Settings_ICCV_2019_paper.html), ICCV 2019.
- Jiang et al., [VAD: Vectorized Scene Representation for Efficient Autonomous Driving](https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_VAD_Vectorized_Scene_Representation_for_Efficient_Autonomous_Driving_ICCV_2023_paper.html), ICCV 2023.
- Perez-Lebel et al., [Decision from Suboptimal Classifiers: Excess Risk Pre- and Post-Calibration](https://proceedings.mlr.press/v258/perez-lebel25a.html), AISTATS 2025.
- Foldager et al., [On the role of model uncertainties in Bayesian optimisation](https://proceedings.mlr.press/v216/foldager23a.html), UAI 2023.
- Luo et al., [Local calibration: metrics and recalibration](https://proceedings.mlr.press/v180/luo22a.html), UAI 2022.
