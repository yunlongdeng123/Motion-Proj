# Motion-Proj

当前研究主线为 **EAS-VGGT**：继承 V7/V7.1 的 canonical 表面、连续证据与分类回波、物理/外观解耦和 SE(3) 刚体组合，学习 VGGT 视觉证据带来的增量。
更新于 2026-09-07，事实基线 `debe8697`；沿用工作分支 `research/worldsim-v7.2-task-first-completion-lidar`，分支旧名称不代表当前方向。新 recovery plan 已完成，E1–E5 实现/实验仍为 `pending`。

## 当前入口

- [EAS-VGGT Recovery Plan](docs/WORLDSIM_V7_2_EAS_VGGT_RECOVERY_PLAN.md)：问题定义、正结果继承表、三机制设计、基线/消融、I/O 与阶段任务。
- [唯一当前状态](docs/RESEARCH_STATUS.md)、[统一失败账本](docs/RESEARCH_FAILURES.md)、[实验事实源](docs/EXPERIMENTS.md)。
- [文档导航](docs/README.md)：环境、复现和历史材料。

原 task-first occupancy/neural-LiDAR 计划与 NKSR/LiDAR-RT recovery 队列已被替代；AdaPoinTr/LiDAR4D 保留为历史比较/能力证据，不再决定主线。`V71-F63` 的 A1 负结果保留，`V71-F64/F65` 分别纠正过早收口与方向漂移。下一步是 E1 的 RGB/投影/特征缓存接入。

## 既有论文与资产

- [V7.1 EAS 英文 main](paper/main.pdf) / [本地 supplement](paper/supplement.pdf)：7 页主稿与 32 页历史档案，含可继承的正结果及边界；不是新 EAS-VGGT 的实验结果。
- [中文精翻](docs/paper/main_zh.md)、[英文源码/构建说明](paper/README.md)、[写作更正](paper/REVISION_NOTES.md)。
- [旧 V7.2 技术报告说明](paper_v72/README.md)：保留 A1 和外部基线事实，整体完成解释已更正。

`motion_proj/`、`scripts/`、`configs/`、`tests/` 存放代码与配置；`docs/autoresearch/`、`docs/run_manifests/` 保留运行凭证；`docs/archive/` 只作历史追溯。当前计划交付不等于项目全部完成，不触发关机。
