> 历史归档（2026-09-10）：以下为原版本事实与指令快照；当前执行以 docs/RESEARCH_STATUS.md 为准。

# Motion-Proj 文档导航

更新：2026-09-07。当前主线已进入 WorldSim V7.3 几何解码适配；V7.2 EAS-VGGT E1–E5 作为已完成证据基线保留。历史文档中的 `running/pending` 不覆盖以下入口。

## 当前研究

- [V7.3 科研计划](WORLDSIM_V7_3_RESEARCH_PLAN.md)：动态刚体 Actor 表面重建、原生几何解码有限适配、局部空间查询、表面/自由空间/首事件一致监督与研究决策规则。
- [状态](../../../RESEARCH_STATUS.md)、[失败账本](../../../RESEARCH_FAILURES.md)、[实验台账](../../../EXPERIMENTS.md)：V7.3 每个里程碑同步更新；当前 M0 完成、M1 原生几何适配基线启动。

- [EAS-VGGT 执行结果](WORLDSIM_V7_2_EAS_VGGT_RESULTS.md)：最短的结论、数值表、canonical run、可写/禁写主张和下一阶段门槛。
- [EAS-VGGT Recovery Plan revision 3](WORLDSIM_V7_2_EAS_VGGT_RECOVERY_PLAN.md)：研究问题、V7/V7.1 正结果继承、四组实验及执行状态。
- [基础模型适配调研](WORLDSIM_V7_2_FOUNDATION_ADAPTATION_RESEARCH.md)：VGGT、Pi3X、MapAnything、DynamicVGGT、强适配/神经 LiDAR 对标。
- [E1 基座和 beam 报告](WORLDSIM_V7_2_E1_BACKBONE_AND_BEAM_REPORT.md)：VGGT/Pi3X 初始诊断和数据接口。
- V7.2 最新失败为 `V71-F68`（route raw visual 负结果与校准恢复）和 `V71-F69`（source insufficient support）；V7.3 下一可用编号为 `V71-F70`。

## 论文与历史

- [V7.2 EAS-VGGT 论文草稿](../../../../paper_v72/README.md)：主文、补充材料和 arXiv PDF 已按最新证据重写。
- [V7.1 论文入口](../../../paper/README.md)、[中文 main](../../../paper/main_zh.md)、[V7.1 综合](WORLDSIM_V71_RESEARCH_SYNTHESIS.md)。
- 旧 task-first/AdaPoinTr D1、LiDAR4D capability 和外部补全路线保留为历史证据；它们不替代 EAS-VGGT 方向。
- [环境](../../../ENVIRONMENT.md)、[机器迁移](../../../MACHINE_MIGRATION.md)、[第三方依赖](../../../THIRD_PARTY.md)、[资产保留](../../../ARTIFACT_RETENTION.md)。

当前准确边界：连续证据、有序回波、物理/外观与 SE(3) 机制成立；几何提升、第二基座、独立 source、真实完整 no-return 和顶会强实证尚未成立。
