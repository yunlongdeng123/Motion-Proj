# Motion-Proj 文档导航

- 更新时间：2026-08-14
- 当前路线：`WorldSim V5 / StructDelta`，分支=`research/worldsim-v5-structdelta`
- 当前授权：`WS-V5-M2-GEOMETRY-FIRST-REPAIR-01=rejected`；r015 已正式确认没有 absolute geometry-safe candidate，M2 router/validation/neural surface 均未解锁。下一步只开放 `WS-V5-M3-CONSTRAINT-PROJECTED-TEMPORAL-01` 的 result-blind 协议与证据审查；fresh validation/test quality 和 KITTI 方法调参仍未授权
- V5 计划：[`WORLDSIM_V5_STRUCTDELTA_PLAN.md`](WORLDSIM_V5_STRUCTDELTA_PLAN.md)
- KITTI archive/adapter：[`KITTI_TRACKING_ARCHIVE_AUDIT_V5.md`](KITTI_TRACKING_ARCHIVE_AUDIT_V5.md)、[`KITTI_TRACKING_ADAPTER_SMOKE_V5.md`](KITTI_TRACKING_ADAPTER_SMOKE_V5.md)，状态=`done`；0001 缺失 LiDAR `177–180` 以 abstain 保留
- 最新关闭路线：`WorldSim V4 / EviDelta-GS`
- V4 终态：`M1 rejected / M2 done with geometry caveat / M3 confirmed`
- V4 当前执行授权：`none_v4_closed`
- V4 终局归档：[`archive/2026-08/worldsim-v4-final/`](archive/2026-08/worldsim-v4-final/README.md)
- 技术报告附录入口：[`archive/2026-08/worldsim-v4-final/TECHNICAL_REPORT_APPENDIX_INDEX.md`](archive/2026-08/worldsim-v4-final/TECHNICAL_REPORT_APPENDIX_INDEX.md)

## 恢复顺序

1. [`../AGENTS.md`](../AGENTS.md)：环境、研究连续性和 Git 约定；
2. [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)：唯一当前状态与执行授权入口；
3. [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)：跨路线失败、禁止重复项与复开条件；
4. [`EXPERIMENTS.md`](EXPERIMENTS.md)：canonical run、hash、指标和任务终态；
5. [`archive/2026-08/worldsim-v4-final/`](archive/2026-08/worldsim-v4-final/README.md)：V4 final snapshot、附录索引和轻量 canonical evidence；
6. [`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md)：V4 canonical run、冻结资产和数据保留边界；
7. [`ENVIRONMENT.md`](ENVIRONMENT.md)、[`THIRD_PARTY.md`](THIRD_PARTY.md)、[`MACHINE_MIGRATION.md`](MACHINE_MIGRATION.md)：环境、依赖和恢复事实。

归档中的“当前任务”“下一步”和 agent 提示词只解释历史，不构成新执行授权。新路线必须先在 `RESEARCH_STATUS.md` 登记分支、task、split 和门禁。

## V4 最终结论

| 模块 | 终态 | 结论边界 |
|---|---|---|
| M1 | `rejected` | scene-disjoint validation=`3 evaluable + 3 abstain`，directional support=`0/6` |
| M2 | `done` | selective routing 有效，但 hole geometry MAE 退化 `+3.3908096237 m` |
| M3 | `confirmed` | exact-once test=`12 evaluable + 6 abstain`，仅覆盖冻结 nuScenes 18 scenes、三前向相机、2–4 s clips、单 RTX 3090 |

V4 的 M2 selective 结论与 geometry caveat 必须成对引用；M3 时序正结果不得覆盖 M1 rejection 或 M2 geometry 失败。V4 已读取的 30 个 nuScenes scene 永久失去后续路线 confirmatory-test 身份。

## 当前事实源

- [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)：当前路线、授权、commit、证据路径和下一步；
- [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)：仍约束后续路线的负结论；
- [`EXPERIMENTS.md`](EXPERIMENTS.md)：完整实验台账；
- [`KITTI_TRACKING_ARCHIVE_AUDIT_V5.md`](KITTI_TRACKING_ARCHIVE_AUDIT_V5.md)：V5 KITTI 压缩包、frame gate、存储预算和 adapter 阻塞结论；
- [`KITTI_TRACKING_ARCHIVE_METADATA_V5.json`](KITTI_TRACKING_ARCHIVE_METADATA_V5.json)：逐 archive、split、sequence 和 class/track metadata；
- [`KITTI_TRACKING_ARCHIVES_V5.sha256`](KITTI_TRACKING_ARCHIVES_V5.sha256)：7 个原始 ZIP 的标准 SHA-256 清单；
- [`WS_V5_M1_FAILURE_FORENSICS.md`](WS_V5_M1_FAILURE_FORENSICS.md)：V4 M1 rejection 的 V5 retrospective 诊断与缺失证据边界；
- [`WS_V5_M2_GEOMETRY_FORENSICS.md`](WS_V5_M2_GEOMETRY_FORENSICS.md)：V4 M2 `+3.3908 m` 的 risk saturation、oracle regret 与 denominator 分解；
- [`WS_V5_M2_GEOMETRY_FIRST_DEVELOPMENT.md`](WS_V5_M2_GEOMETRY_FIRST_DEVELOPMENT.md)：V5 M2 r001–r015、逐 actor request 修正、G0–G5 失败链、cross-view coverage 与正式 rejection；
- [`archive/2026-08/worldsim-v5-m2/APPENDIX_INDEX.md`](archive/2026-08/worldsim-v5-m2/APPENDIX_INDEX.md)：M2 技术报告附录与机器元数据入口；
- [`../configs/worldsim_v5/p0_scope_v1.yaml`](../configs/worldsim_v5/p0_scope_v1.yaml)：V5 P0 科学范围、claim、门禁和授权；
- [`../configs/worldsim_v5/nuscenes_fresh_cohort_v1.yaml`](../configs/worldsim_v5/nuscenes_fresh_cohort_v1.yaml)：fresh 8/8/20 结果前 cohort 合同，当前尚未选择 scenes；
- [`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md)：驻留/非驻留边界；
- [`archive/2026-08/worldsim-v4-final/SHA256SUMS`](archive/2026-08/worldsim-v4-final/SHA256SUMS)：V4 终局包完整性。

## 根目录兼容文件

V4 计划、P0/D0/B0/KITTI 审计和更早 V3/V2 计划继续保留在 `docs/` 根目录，原因是冻结 config、run manifest、source snapshot 和历史链接引用原路径。它们是 compatibility copies，不是当前任务入口。

V5 计划 `WORLDSIM_V5_STRUCTDELTA_PLAN.md` 已进入 P0 scope freeze，但 P0 尚未收口；计划本身不授权完整 M1/M2/M3、fresh test 读取或 KITTI 调参。KITTI 7 个原始 ZIP 已完成 archive-level 审计，但真实 adapter 在 common-frame policy、calibration 与 OXTS 修复及 2-sequence smoke 完成前保持 `blocked`。

## 归档索引

- [WorldSim V4 终局归档](archive/2026-08/worldsim-v4-final/README.md)
- [WorldSim V3.2 终局归档](archive/2026-08/worldsim-v3.2/README.md)
- [WorldSim V3.1 终局归档](archive/2026-08/worldsim-v3.1/README.md)
- [2026-08 总归档](archive/2026-08/README.md)
- [动态重建 V1 终态](archive/2026-07/dynamic-reconstruction-v1/README.md)
- [2026-07 总归档](archive/2026-07/README.md)
- [`run_manifests/`](run_manifests/README.md)：早期轻量 manifests。

## 文档规则

- 根目录只维护当前事实源、长期环境/资产文档和 hash/link compatibility copies；
- 完成、阻塞或拒绝任务后同步更新 STATUS / EXPERIMENTS / FAILURES；
- task 状态只使用 `pending/running/blocked/done/rejected`；
- `blocked`、`rejected`、`abstain` 和 `done` 必须分开，且完整 denominator 不得删除；
- 正式 run 的 config、manifest、fingerprint、summary、source snapshot 和关键 SHA 必须可追溯；
- 人工 verdict 只能由用户或指定评审者填写。
