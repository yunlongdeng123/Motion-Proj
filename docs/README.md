# Motion-Proj 文档导航

更新：2026-09-07；事实基线 `debe8697`。当前方向为 **EAS-VGGT**，计划 E0 已交付，E1–E5 待实现。状态与执行范围以 [RESEARCH_STATUS.md](RESEARCH_STATUS.md) 文首为准。

## 当前研究

- [EAS-VGGT Recovery Plan](WORLDSIM_V7_2_EAS_VGGT_RECOVERY_PLAN.md)：V7/V7.1 正结果继承、物理/外观解耦、连续证据与分类回波、SE(3) 轨迹、VGGT 接入与实验矩阵。
- [当前状态](RESEARCH_STATUS.md)、[统一失败账本](RESEARCH_FAILURES.md)、[实验事实源](EXPERIMENTS.md)。最新纠偏 `V71-F65`；新训练和独立测试尚未执行。
- 原 task-first 总计划、执行计划、GPU handoff 和上一版 research-first recovery 均已标记历史；其 A/B、R1–R7 队列不再调度。

## 已有论文与证据

- [V7.1 论文阅读入口](paper/README.md)、[中文 main](paper/main_zh.md)、[英文稿说明](../paper/README.md)、[写作更正](../paper/REVISION_NOTES.md)。
- [V7.1 研究综合](WORLDSIM_V71_RESEARCH_SYNTHESIS.md)：历史正/负结果；新的继承范围以 EAS-VGGT 计划为准。
- [旧 V7.2 负向技术报告](../paper_v72/README.md)：保留 baseline/A1/capability 结果，不作为新研究的完成证明。

## 环境与历史

- [环境说明](ENVIRONMENT.md)、[机器迁移](MACHINE_MIGRATION.md)、[第三方依赖](THIRD_PARTY.md)、[资产保留合同](ARTIFACT_RETENTION.md)。
- [归档总索引](archive/README.md)、[旧路径查询](archive/2026-09/root-docs-20260906/README.md)、[研究凭证](autoresearch/README.md)、[运行清单](run_manifests/README.md)。

历史记录中“当前”“下一步”只属于当时快照。现有 checkpoint、数据、run、PDF 和已暴露的数据身份保留；本轮同步不产生新的方法质量结论。
