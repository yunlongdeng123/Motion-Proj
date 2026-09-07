# WorldSim V7.2 EAS-VGGT 执行结果与下一步

更新：2026-09-08；实现阶段 `E1–E5 executed`。本页只总结最新 EAS-VGGT 主线；旧 AdaPoinTr D1 负结果仍保留但不再代表当前方向。

## 结论

V7.2 已从“接入外部模型补全全局点云”改成以 EAS 为研究对象，并真实完成四个核心机制：VGGT late-fusion 连续证据、blocking/detection 分类回波测度、物理/外观所有权桥、SE(3) 刚体轨迹组合。当前最可靠的正结果是 **route-select 上的测量证据校准改善**；几何没有改善，Pi3X 没有复现，冻结 source cohort 因没有可观测 Actor 候选而不可计算。因此不能声称通用 EAS 插件、独立泛化或顶会强结果已经成立。

| 维度 | Canonical 结果 | 判定 |
|---|---|---|
| 开发集 VGGT 连续证据 | late EAS Brier `.18755` vs no-visual `.19186`；NLL `.79123` vs `.79168`；early fusion `.27455`；scalar `.41679` | late fusion 有小幅正增量；结构选择有证据 |
| Pi3X 迁移 | visual Brier `.24066` vs no-visual `.19264` | 负结果；不支持跨基座通用 |
| Route 证据 | fixed reliability `alpha=.35`：Brier `.23434` vs `.23753`，NLL `.94380` vs `.95972`；raw visual Brier `.24477` | route-select 支持；2/3 logs Brier 改善 |
| 物理表面 | route F-score 均 `.66756`，CD 约 `.21031m`；dev displacement RMSE 基本不变 | 没有几何贡献 |
| 有序回波 | 16,384 controlled rays：block/detect NLL `.86215` vs single hazard `1.05022`；Brier `.000005` vs `.11969`；质量分裂误差 0 | 机制支持；真实全场景回波待验证 |
| 物理/外观 | 309 parents→3,090 visual primitives；6/6 held-out views 提升；pooled PSNR `17.186→17.730dB`；物理/轨迹无 RGB 更新 | 所有权和单向对应支持；距 StreetGS `7.62dB` |
| SE(3) | 11 Actors/40 poses；commutation `6.82e-13m`；frame recovery `1.48e-13m`；pairwise `2.82e-13m`；最大位移 `5.24m` | 表示层等变组合支持 |
| MapAnything | 原生 RGB+intrinsics+metric pose；median surface residual `.799m`，0.2m hit `4.76%` | 第三几何来源已接通；只作诊断 |
| Source | 677/677 LiDAR；80 Actors；12 VGGT windows；8 pose matches；0 observed candidates | evidence metrics 未定义；`inconclusive_insufficient_support` |

## Canonical runs

- E1 VGGT/Pi3X：`run://worldsim_v72/WS-V72-E1-VGGT-EVIDENCE-IO-01/20260907T144500Z__e1-vggt-pi3x-train-observation-s7201-r4`。
- E1 MapAnything：`run://worldsim_v72/WS-V72-E1-VGGT-EVIDENCE-IO-01/20260907T235000Z__e1-mapanything-route-s7103-r3`。
- E2 late fusion：`run://worldsim_v72/WS-V72-E2-LEARNED-VISUAL-EVIDENCE-01/20260907T183000Z__e2-canonical-late-fusion-s7202-r1`。
- E2 ordered return：`run://worldsim_v72/WS-V72-E2-ORDERED-RETURN-MEASURE-01/20260907T195500Z__e2-ordered-return-s7203-r2`。
- E3 rendering bridge：`run://worldsim_v72/WS-V72-E3-DECOUPLED-APPEARANCE-01/20260907T213500Z__e3-render-bridge-s7302-r2`。
- E4 SE(3)：`run://worldsim_v72/WS-V72-E4-SE3-RIGID-TRAJECTORY-01/20260907T191500Z__e4-se3-rigid-trajectory-s7401-r4`。
- E5 route：`run://worldsim_v72/WS-V72-E5-FROZEN-CONFIRMATION-01/20260907T223000Z__e5-route-calibrated-s7501-r2`。
- Source support audit：`run://worldsim_v72/WS-V72-E5-FROZEN-SUPPORT-AUDIT-01/20260908T002500Z__e5-source-support-audit-s7502-r4`。

## 对负结果的迁移

Raw visual fusion 在 route 上伤害 Brier 后，参考 CVPR 2025 MoME 的独立专家/质量路由和 CVPR 2021 domain-drift calibration，迁移为不改变 simplex 的 visual-expert reliability discount。冻结 source 出现零支持后，参考 missing-modality robustness 的结论，将其处理为显式 support failure；没有换场景、调模型或把 `NaN` 当成分数。

- MoME：https://openaccess.thecvf.com/content/CVPR2025/html/Park_Resilient_Sensor_Fusion_Under_Adverse_Sensor_Failures_via_Multi-Modal_Expert_CVPR_2025_paper.html
- Domain-drift calibration：https://openaccess.thecvf.com/content/CVPR2021/html/Tomani_Post-Hoc_Uncertainty_Calibration_for_Domain_Drift_Scenarios_CVPR_2021_paper.html
- Missing modality：https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Are_Multimodal_Transformers_Robust_to_Missing_Modality_CVPR_2022_paper.html
- MapAnything：https://github.com/facebookresearch/map-anything

## 下一阶段研究门槛

本轮 V7.2 工程和机制原型完成；投稿级科学问题仍有四项明确缺口：

1. 在任何 quality read 前冻结 target-free camera/Actor support eligibility，并在全新独立 cohort 确认。
2. 让视觉证据改善定位后的物理 surface/event，而不只改善 F/O/U proper score。
3. 在真实有效 firing 上联合评估 return/no-return、背景/多 Actor 遮挡和距离分布。
4. 让第二基座复现，或将论文明确收窄为 VGGT-specific EAS。

`paper_v72/` 已按这些已成立结果与边界重写并编译；它是下一阶段的可审稿研究草稿，不把未完成缺口包装成结论。
