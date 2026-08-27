# WorldSim V6.5 Failure Analysis

Terminal synthesis (2026-08-28): V6.5 recorded `V65-F01` through `V65-F19`. The scientific chain rejects local voxel/map,
Actor, learned-admission, smooth-tail, and direct-action-authority formulations, while retaining the changed prediction object of
given-trajectory visited-state reliability ranking and expected-error calibration. `V65-F13--F18` are engineering/entry failures
with explicit exposure audits, not method-negative trials. `V65-F19` is the terminal one-shot scientific rejection. The authoritative
chronology remains `docs/RESEARCH_FAILURES.md`; this file explains mechanisms and paper treatment.

## V65-F01 — static-label target semantics mismatch

P1 T0 的 trajectory condition 对 unit 内 shuffle 有可测响应，却无法改善冻结 q0，说明问题不是条件通路完全失效，
而是监督目标要求 task query 去解释 task-agnostic static hidden-FREE。

迁移依据：WoTE（ICCV 2025）用候选轨迹预测未来结果并评价轨迹；UniAD（CVPR 2023）与 VAD（ICCV 2023）
都把 planning query 与未来 occupancy / actor / map 表示交互。项目内对应迁移是保留 q0 的物理语义，另建 task
risk，不把两者混成一个需要同时提升 global voxel AUROC 的分数。

- WoTE：https://openaccess.thecvf.com/content/ICCV2025/papers/Li_End-to-End_Driving_with_Online_Trajectory_Evaluation_via_BEV_World_Model_ICCV_2025_paper.pdf
- UniAD：https://github.com/OpenDriveLab/UniAD
- VAD：https://github.com/hustvl/VAD

若 P1R 仍无 fixed-opportunity 增量，则关闭纯 trajectory task-risk family，转向 Actor×time/query outcome，而不再
修改同一 residual。

## V65-F02 — frozen temporal-info capability mismatch before quality read

首版 fresh cohort 依据 scene description、既有 processed availability 与旧 config exposure 冻结，但冻结 IR-WM
worker 会直接索引 `nuscenes_temporal_infos_train.pkl["infos"][scene]`。该 pickle 只有 700 个 scene keys；
`0520/0781/0800/0106` 缺失，因此失败发生于官方模型输入构造前，而非模型表现阶段。

官方 BEVFormer 的标准方案是执行 nuScenes `tools/create_data.py` 生成 temporal train/val infos：
https://github.com/fundamentalvision/BEVFormer/blob/master/docs/prepare_dataset.md 。本项目当前采用更小的迁移边界：
不重建 infos，不改变 CAN bus/schema/checkpoint contract，只把 cohort 换成冻结 pickle 已支持且旧 configs 未使用的
`0996/0443/0002/0043/0023/0072`。该替换只读 capability metadata，发生在任何 P2 quality read 前；因此不形成
selection bias，也不改变 P2 gate。后续 cohort freeze 必须在 metadata selection 时同时审计 backend key availability。

## V65-F03 — fresh task-risk ranking invariance

Fresh P2 在全部 6 scenes 上 lower/equal/higher=`0/6/0`；q0 与 task arm 都选择到 18 个 route conflicts，说明
P1R 的小幅 legacy gain 没有迁移到固定 40% ranking boundary。monotone semantics 正常、scene regression 为零，
所以这不是数值异常；task arm 还在 non-route 多发射 4 个 conflicts。trajectory-only residual family 由此关闭。

相关顶会工作给出的结构性替代不是扩大同一 residual：PRECOG（ICCV 2019）做 goal-conditioned multi-agent
forecasting；M2I（CVPR 2022）显式建 influencer/reactor conditional prediction；GameFormer（ICCV 2023）联合
ego plan 与多 actor response；VAD（ICCV 2023）使用 vectorized agent motion 作为 instance-level planning
constraint；Implicit Occupancy Flow（CVPR 2023）让 planner 查询连续时空 occupancy/flow。项目迁移边界因此是
actor-time/action-outcome 的新 train-only hypothesis；已消费 P2 scenes 不得用于新模型选择。

- PRECOG：https://openaccess.thecvf.com/content_ICCV_2019/html/Rhinehart_PRECOG_PREdiction_Conditioned_on_Goals_in_Visual_Multi-Agent_Settings_ICCV_2019_paper.html

## V65-F10 — calibration improvement is not trajectory selection improvement

The R4 neural head cut MSE by 87.35% but reduced Spearman by 0.11636 and increased selected realized cost by 51.35%.
This is a concrete instance of optimizing average calibration while damaging the order used by a downstream selector.
The within-scene shuffle response was positive, so the failure is not a disconnected input; the learned remapping simply
did not preserve task-relevant tails.

The direct Qagg arm is the important positive control: Spearman 0.75149, unsafe AUROC 0.97826, and 62.98% realized-cost
reduction at 40% coverage. Following the task-relevant-failure principle, the project retains this trajectory-level
reduction and removes the unnecessary learned head. Any later learning must operate on a genuinely new Actor false-safe
target or fresh action-level transfer, not tune R4 after its read.

## V65-F11 — Actor extrema break per-token forecast transfer

P2C's strong per-actor snapshot correlation did not survive maximum-cost reduction to the trajectory level. The A0
Spearman dropped to 0.6261, and the weaker A1 forecast could not serve as a disagreement monitor: positive disagreement
was essentially random for false-safe gaps and selected a worse subset. This is consistent with extreme-value
aggregation amplifying the worst per-token forecast error.

Changing max to a smooth maximum after seeing the result would change the prediction object and aggregation contract.
The project therefore closes this Actor companion and continues only the already-supported world-state Qagg fresh
transfer. The 9/24 positive gaps show that the negative result is scientifically identifiable rather than label-empty.
- M2I：https://openaccess.thecvf.com/content/CVPR2022/html/Sun_M2I_From_Factored_Marginal_Trajectory_Prediction_to_Interactive_Prediction_CVPR_2022_paper.html
- GameFormer：https://openaccess.thecvf.com/content/ICCV2023/html/Huang_GameFormer_Game-theoretic_Modeling_and_Learning_of_Transformer-based_Interactive_Prediction_and_ICCV_2023_paper.html
- VAD：https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_VAD_Vectorized_Scene_Representation_for_Efficient_Autonomous_Driving_ICCV_2023_paper.html
- Implicit Occupancy Flow：https://openaccess.thecvf.com/content/CVPR2023/html/Agro_Implicit_Occupancy_Flow_Fields_for_Perception_and_Prediction_in_Self-Driving_CVPR_2023_paper.html

## V65-F04 — zero support for hard collision outcome

P2R 在 476 train 与 302 eval Actor tokens 上都没有任何 1.5m collision positive；模型排序指标因此不可定义。
这不是扩大 corridor、换 scene 或反复采样的理由。连续 cost 迁移依据：CVPR 2021 joint road-dynamics/cost-map
直接学习可供 planner 积分的时空 cost；Waymo Occupancy Flow 同时表达 occupancy 与 motion；DTPP/DiffStack 将
prediction 与 differentiable cost/planning objective 联合。P2C 固定 `exp(-distance/6m)`，其余 split/features/
capacity 不变。

- Joint dynamics and cost map：https://openaccess.thecvf.com/content/CVPR2021/html/Amirloo_Self-Supervised_Simultaneous_Multi-Step_Prediction_of_Road_Dynamics_and_Cost_Map_CVPR_2021_paper.html
- Occupancy Flow：https://waymo.com/research/occupancy-flow-fields-for-motion-forecasting-in-autonomous-driving/
- DTPP：https://github.com/MCZhi/DTPP
- DiffStack：https://proceedings.mlr.press/v205/karkus23a.html

## V65-F12 — any-error tail separation is not expected visited-cost authority

The fixed smooth-tail statistic raised unsafe AUROC from `0.978261` to `1.000000`, but reduced continuous-target Spearman from
`0.751487` to `0.708230` and worsened selected-40% realized cost by `27.27%`. An upper tail emphasizes whether any high-risk sample is
present, while the frozen target is the expected hidden-FREE fraction over all visited samples. No temperature or coverage sweep was
performed after this read. The deterministic mean remained the supported prediction statistic.

## V65-F13--F18 — engineering entries and exposure boundaries

These failures are required for reproducibility but are not method negatives:

| IDs | failure class | scientific exposure | recovery |
| --- | --- | --- | --- |
| `F13` | scene-ready launcher omitted task parent creation | none | create the exact parent before launch |
| `F14` | launcher bypassed base+overlay config composition | none | call the existing validated wrapper |
| `F15` | evaluator assumed `[B,1]` rather than frozen `[B]` logits | one unit loaded; no metric/cache/verdict | replace dimension-specific squeeze with scalar reshape |
| `F16` | evidence CLI omitted required processed root | none | pass the frozen standard processed root |
| `F17` | formal config used a nonexistent run-relative q0 locator | no input unit or metric | point to the same frozen artifact already used by P2V |
| `F18` | direct feeder entry omitted repository root from module path | none | process-local `PYTHONPATH=.` |

P2V r2 and P3C r2 did not change the model, scene, target, sampler, calibrator, seed, or gates. The exact exposure audits remain in
`RESEARCH_FAILURES.md` and must be disclosed in a reproducibility appendix without counting them as scientific ablations.

## V65-F19 — calibration and ranking do not guarantee direct action benefit

The sole combined confirmation passed route ranking, frozen-map MSE reduction, action ranking, unsafe AUROC, and pairwise concordance.
It failed the sixth AND gate: lowest-Qmean-quarter action selection reduced realized cost from `0.120215` to `0.100520`, a `16.38%`
reduction below the frozen `25%` requirement. Five scenes improved and `scene-0817` worsened. `scene-0718` had no eligible nominal
route but did improve under action selection, so the terminal failure cannot be removed by blaming or deleting one sparse scene.

This is consistent with decision-risk decompositions in which global recalibration removes miscalibration regret but leaves grouping
loss from unresolved conditional structure. It does not license a post-read local calibrator or decision-focused head. V6.5 closes
direct action authority and retains only given-trajectory visited-state ranking and expected-error calibration.

## Terminal failure taxonomy

| category | IDs | paper treatment |
| --- | --- | --- |
| scientific prediction-object/method negatives | `F03/F04/F06/F07/F09/F10/F11/F12/F19` | main text and limitations |
| target/capability/config corrections | `F01/F02/F05/F08` | reproducibility appendix |
| streamed-pipeline/runtime entries | `F13/F14/F16/F18` | systems appendix |
| narrow formal-entry recoveries | `F15/F17` | disclose exposure and unchanged contract |

The next available failure identifier remains `V65-F20`; the documentation/report handoff introduces no new scientific failure.

## V65-F09 — map semantics cannot rescue the per-voxel prediction object

R3 consumed official v1.3 map semantics and produced a positive within-unit shuffle response, yet AUROC fell by
`0.000496`, AUPRC fell by `0.002280`, pooled fixed-route risk was unchanged, and scene support was `1/14/1`.
This separates “the network can read the map” from “the map changes reliable authority decisions.” The frozen q0 already
contains strong local physical-boundary information, while the voxel-level target still weights errors independently of
whether the Ego trajectory will visit them.

Task-Relevant Failure Detection (CoRL 2022) propagates prediction errors to planning cost and detects harmful rather than
generic failures. PRECOG (ICCV 2019) conditions other-agent futures on the controlled agent's goal. The project migration
therefore changes the supervised object to a `(scene, unit, trajectory)` future visited-state outcome, using q0 and context
as inputs and the realized route-corridor conflict as target. It does not enlarge or rerun R3.

- paper：https://proceedings.mlr.press/v205/farid23a.html
- official code：https://github.com/NVlabs/pred-fail-detector
- PRECOG：https://openaccess.thecvf.com/content_ICCV_2019/html/Rhinehart_PRECOG_PREdiction_Conditioned_on_Goals_in_Visual_Multi-Agent_Settings_ICCV_2019_paper.html

## V65-F07 — learned coverage is not risk-controlled admission

G0 learned useful average capacity but over-predicted one case by about 0.011 coverage and crossed the frozen 0.05
hidden-FREE boundary. It also moved fixed-route risk in the wrong direction. Conformal Risk Control chooses a monotone risk
parameter on a separate calibration set; applying it after seeing this held-out miss would be a new calibration experiment,
not a correction of the same result. SOFT top-k only makes allocation differentiable and does not create missing risk signal;
the plan therefore correctly keeps it conditional on P4 success. GroupDRO is likewise not justified because pooled G0 itself
failed before any isolated worst-group failure.

- Conformal Risk Control：https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html
- SOFT top-k：https://papers.neurips.cc/paper_files/paper/2020/hash/ec24a54d62ce57ba93a531b460fa8d18-Abstract.html
- GroupDRO official code：https://github.com/kohpangwei/group_DRO

## V65-F08 — map data presence did not imply devkit schema compatibility

The project meta root had only legacy raster PNGs. The public volume's v1.2 expansion JSON was structurally complete but the
installed devkit intentionally rejects versions below 1.3. Official nuScenes documentation states that v1.3 adds lidar
basemap support and removes a broken lane. Downgrading code or bypassing that check would create an untracked map contract.
The recovery uses the official co-located v1.3 archive in a separate root; the eight requested semantic layers rasterized
successfully before R3 was preregistered.

- nuScenes devkit/map versions：https://github.com/nutonomy/nuscenes-devkit

## V65-F05 — shared materializer leaked a binary-only config dependency

P2C first entry failed before evidence I/O or training because the shared P2R materializer indexed
`route_corridor_radius_m` even though the continuous target never consumes the binary label. Hydra config composition
documents the relevant software boundary: common configuration and task-specific groups are composed, rather than making
every variant provide unrelated fields. The project migration was one optional read with a neutral unused fallback; the
failed run is preserved and the scientific contract was not altered.

- Hydra composition：https://hydra.cc/docs/tutorials/basic/your_first_app/composition/

## V65-F06 — temporal sensitivity is not incremental Actor-time authority

P2C removed zero-support ambiguity but A1 still lost to A0: Spearman gain `-0.014889`, MSE reduction `-34.59%`, and
both evaluation scenes had higher matched-coverage selected cost. The `+0.098817` real-minus-shuffled Spearman confirms
that history/time entered the computation, so enlarging the same model is not a justified fix. DTPP and DiffStack instead
couple prediction/cost with candidate policy or downstream planning objectives; UniAD makes agent-goal interaction
planning-oriented. In this project those ideas would require a new action/candidate supervision contract, not a rescue of
the closed Actor-token family. P3 therefore remains locked.

- UniAD：https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper
- DTPP：https://github.com/MCZhi/DTPP
- DiffStack：https://proceedings.mlr.press/v205/karkus23a.html
