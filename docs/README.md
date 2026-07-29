# Motion-Proj 文档导航

更新时间：2026-07-29

动态驾驶重建路线已按预注册 novelty gate 走到研究终态：

> AD-GS 六场景复现完成 → DGGT upstream 对照闭合 → M6 身份负结果跨六场景重复 → M7 novelty rejected → M8/M9 未授权。

## 当前必须先读

1. [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)：唯一当前状态与授权入口；
2. [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)：失败、边界和防重复规则；
3. [`EXPERIMENTS.md`](EXPERIMENTS.md)：当前路线实验注册表；
4. [`DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md`](DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md)：完整方案与最终里程碑日志；
5. [`DR_M5_DGGT_REPORT.md`](DR_M5_DGGT_REPORT.md)：DGGT blocked 证据与 common-observation 边界；
6. [`DR_M7_NOVELTY_AUDIT.md`](DR_M7_NOVELTY_AUDIT.md)：唯一候选与官方 novelty matrix；
7. [`human-review/dynamic-reconstruction-results-v1/README.md`](human-review/dynamic-reconstruction-results-v1/README.md)：M9 未触发的机器终止包；
8. [`20260729T150248+0800_summary.md`](20260729T150248+0800_summary.md)：本轮结束总结。

## 当前路线

- M4：AD-GS 官方 nuScenes 六 scenes exact reproduction `done`，三项带宽通过；
- M5：DGGT inference-only `blocked`，正式证据
  `/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/`；
  common-observation 未触发；
- M6：`persistent_object_identity_unavailable` 在 6/6 scenes 重复，0/12 object slots；
- M7：候选 A novelty `rejected`；
- M8/M9：`rejected / not authorized`，没有方法数字、盲审 clips 或人工 verdict；
- 当前禁止：继续本轮方法实现、事后注册 endpoint、把适配工程或 0 coverage 重命名为创新。

## 长期维护文档

- [`ENVIRONMENT.md`](ENVIRONMENT.md)：当前 Motion-Proj 环境；
- [`THIRD_PARTY.md`](THIRD_PARTY.md)：第三方代码与许可证；
- [`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md)：产物保留/清理规则；
- [`MACHINE_MIGRATION.md`](MACHINE_MIGRATION.md)：迁移说明；
- [`run_manifests/`](run_manifests/)：轻量 run manifests。

AD-GS/DGGT 环境与精确版本已经登记在 [`ENVIRONMENT.md`](ENVIRONMENT.md)；新路线必须使用新的任务 ID、
独立 novelty delta 与前瞻 endpoint，不能覆盖本轮 rejected/ABSTAIN 证据。

## 历史路线

- [2026-07 总归档索引](archive/2026-07/README.md)
- [cut-in 路线最终封存](archive/2026-07/cutin-mining-closed/README.md)
- [OccGS V7 可行性](archive/2026-07/v7-feasibility/)
- [OccGS V7.1 H1 拒绝](archive/2026-07/v7.1-h1-reject/)
- [event-first mini 拒绝](archive/2026-07/event-first-mini-reject/)
- [N1 运动学第三轮](archive/2026-07/n1-kinematic-third-reject/)
- [N1 接收车第四轮](archive/2026-07/n1-receiver-cutin-fourth-review/)

历史文档只解释当时发生了什么，不授予继续运行旧路线的权限。任何冲突以当前
[`RESEARCH_STATUS.md`](RESEARCH_STATUS.md) 为准。

## 文档规则

- 活跃根目录只保留状态、失败、实验、当前计划和长期维护文档；
- 路线结束后移入 `archive/YYYY-MM/<route>/`，不在根目录堆多轮报告；
- 失败教训永不因路线切换而删除；
- 工具生成的 `.codexbak.*` 不作为正式历史，正式快照必须有命名、索引和恢复路径；
- 每完成一个里程碑，立即更新 plan、status、experiments，再决定下一步；
- 人工 verdict 只能由用户或指定评审者填写。
