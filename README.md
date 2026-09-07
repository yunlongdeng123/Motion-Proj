# Motion-Proj

当前主线为 **EAS-VGGT**：将视觉几何先验与连续观测证据学习为传感器可查询、可随 SE(3) 轨迹编辑且与外观保持对应的动态场景表示。
2026-09-07 已融合用户补充调研，计划 revision=`2`，事实基线 `35ca52da`；E0 文档完成，E1–E5 实验均为 pending。资源按研究需要安排，不以单卡、2GB 或极小模型限定贡献。

## 当前入口

- [EAS-VGGT Recovery Plan](docs/WORLDSIM_V7_2_EAS_VGGT_RECOVERY_PLAN.md)：主任务、V7/V7.1 正结果继承、方法增量、四组核心实验、数据与贡献边界。
- [补充调研核对与迁移](docs/WORLDSIM_V7_2_FOUNDATION_ADAPTATION_RESEARCH.md)：TALO、VGGT-Segmentor、Marigold-DC、CAPA 等官方来源及接入决策。
- [唯一当前状态](docs/RESEARCH_STATUS.md)、[统一失败账本](docs/RESEARCH_FAILURES.md)、[实验事实源](docs/EXPERIMENTS.md)、[文档导航](docs/README.md)。

下一步 E1 同时处理有效基座、RGB/beam 观测合同和两基座瓶颈诊断。必需证据包括同测量预算的校正/融合/深度适配/scalar/LiDAR-only 基线、返回存在性与遮挡、物理—外观对应，以及独立场景和第三几何来源迁移。M39 条件 median、PSNR 不变或坐标恒等式各自不能代替完整方法证据；当前风险见 `V71-F66`。

旧 task-first A/B、NKSR/LiDAR-RT recovery 队列已被替代。外部方法按新协议作比较，不能替换 EAS 的方法主线。沿用分支 `research/worldsim-v7.2-task-first-completion-lidar` 以保持提交连续性。

## 既有证据与目录

- [V7.1 EAS main](paper/main.pdf) / [本地 supplement](paper/supplement.pdf)、[中文稿](docs/paper/main_zh.md)、[源码说明](paper/README.md)、[写作更正](paper/REVISION_NOTES.md)。
- [旧 V7.2 技术报告](paper_v72/README.md)：A1、简单基线与 LiDAR4D capability 的历史结果；不是新 EAS-VGGT 实验。

`motion_proj/`、`scripts/`、`configs/`、`tests/` 为实现与配置；`docs/autoresearch/`、`docs/run_manifests/` 为凭证；`docs/archive/` 为历史材料。既有 checkpoint、数据曝光身份与 PDF 保留，计划完成不触发关机。
