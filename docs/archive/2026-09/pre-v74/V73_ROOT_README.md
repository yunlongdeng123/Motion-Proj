> 历史归档（2026-09-10）：以下为原版本事实与指令快照；当前执行以 docs/RESEARCH_STATUS.md 为准。

# Motion-Proj

当前主线为 **EAS-VGGT**：将视觉几何先验与连续观测证据学习为传感器可查询、可随 SE(3) 轨迹编辑且与外观保持对应的动态场景表示。
2026-09-07 已融合用户补充调研，计划 revision=`2`；E1 正在推进，E2–E5 pending。VGGT-1B/Pi3X 官方权重、真实三相机诊断、原生 beam/outcome 合同与质量盲 Waymo context 划分已经落地。资源按研究需要安排，不以单卡、2GB 或极小模型限定贡献。

## 当前入口

- [EAS-VGGT Recovery Plan](WORLDSIM_V7_2_EAS_VGGT_RECOVERY_PLAN.md)：主任务、V7/V7.1 正结果继承、方法增量、四组核心实验、数据与贡献边界。
- [补充调研核对与迁移](WORLDSIM_V7_2_FOUNDATION_ADAPTATION_RESEARCH.md)：TALO、VGGT-Segmentor、Marigold-DC、CAPA 等官方来源及接入决策。
- [E1 双基座与 beam 报告](WORLDSIM_V7_2_E1_BACKBONE_AND_BEAM_REPORT.md)：正式 run、基座瓶颈、RGB/feature cache、Waymo return contract 与冻结 split。
- [唯一当前状态](../../../RESEARCH_STATUS.md)、[统一失败账本](../../../RESEARCH_FAILURES.md)、[实验事实源](../../../EXPERIMENTS.md)、[文档导航](../../../README.md)。

下一步补齐授权后的 Waymo development payload 审计，同时推进 E2 的同测量校正/融合/深度适配/scalar/LiDAR-only 基线与 EAS 机制。后续仍需返回存在性与遮挡、物理—外观对应、独立场景和第三几何来源迁移。M39 条件 median、PSNR 不变或坐标恒等式各自不能代替完整方法证据；当前主张风险见 `V71-F66`，已修复工程问题见 `V71-F67`。

旧 task-first A/B、NKSR/LiDAR-RT recovery 队列已被替代。外部方法按新协议作比较，不能替换 EAS 的方法主线。沿用分支 `research/worldsim-v7.2-task-first-completion-lidar` 以保持提交连续性。

## 既有证据与目录

- [V7.1 EAS main](../../../../paper/main.pdf) / [本地 supplement](../../../../paper/supplement.pdf)、[中文稿](../../../paper/main_zh.md)、[源码说明](../../../../paper/README.md)、[写作更正](../../../../paper/REVISION_NOTES.md)。
- [旧 V7.2 技术报告](../../../../paper_v72/README.md)：A1、简单基线与 LiDAR4D capability 的历史结果；不是新 EAS-VGGT 实验。

`motion_proj/`、`scripts/`、`configs/`、`tests/` 为实现与配置；`docs/autoresearch/`、`docs/run_manifests/` 为凭证；`docs/archive/` 为历史材料。既有 checkpoint、数据曝光身份与 PDF 保留，计划完成不触发关机。
