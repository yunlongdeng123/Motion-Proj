# Motion-Proj 文档导航

- 更新时间：2026-08-10
- 当前路线：WorldSim V3.2 语义资产修复
- 当前任务：`WS-V32-R0-INTEGRATION-01`（`done`）
- 最新完成：`WS-V32-R0-INTEGRATION-01`（canonical r4，8/8 gates）
- 当前执行授权：全部单卡 RTX 3090 可执行的 V3.2 工作已完成；S4 temporal 与 S5 仍受外部门禁阻塞，当前无下一执行项
- 当前计划：[`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_2.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_2.md)
- S0 审计：[`WS_V32_S0_SOTA_AUDIT.md`](WS_V32_S0_SOTA_AUDIT.md)
- V3.1 归档：[`archive/2026-08/worldsim-v3.1/`](archive/2026-08/worldsim-v3.1/README.md)

## 恢复顺序

1. [`../AGENTS.md`](../AGENTS.md)：仓库级环境、研究连续性和提交约定；
2. [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)：唯一当前状态与执行授权入口；
3. [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)：跨路线失败与禁止重复项；
4. [`EXPERIMENTS.md`](EXPERIMENTS.md)：V2/V3.1 冻结事实与 V3.2 当前注册表；
5. [`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_2.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_2.md)：当前执行合同；
6. [`WS_V32_S0_SOTA_AUDIT.md`](WS_V32_S0_SOTA_AUDIT.md)：第三方 source/license/weight/hardware 审计。

清空对话、换机或换实例后，不从归档计划、旧 terminal 或计划中的“下一步”恢复动作。只有
`RESEARCH_STATUS.md` 可以授权新的 task、run、训练或评测。

## 当前事实源

- [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)：最新状态、终局裁决、机器与工作树边界；
- [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)：仍约束后续路线的失败、负结果和复开条件；
- [`EXPERIMENTS.md`](EXPERIMENTS.md)：canonical run、hash、指标和任务终态；
- [`ENVIRONMENT.md`](ENVIRONMENT.md)：机器、环境、数据、权重和镜像策略；
- [`THIRD_PARTY.md`](THIRD_PARTY.md)：第三方代码、commit、license 与驻留状态；
- [`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md)：资产保留边界；
- [`MACHINE_MIGRATION.md`](MACHINE_MIGRATION.md)：换机与恢复流程。

## V3.2 当前文件

- [`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_2.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_2.md)：当前计划；
- [`WS_V32_S0_SOTA_AUDIT.md`](WS_V32_S0_SOTA_AUDIT.md)：S0 source/license/weight 审计；
- [`../configs/worldsim_v32/s0_sources_v1.yaml`](../configs/worldsim_v32/s0_sources_v1.yaml)：机器可读 source/weight manifest。
- [`../configs/worldsim_v32/s1_semantic_lift_v3.yaml`](../configs/worldsim_v32/s1_semantic_lift_v3.yaml)：S1 修正后的
  actor identity、双向 SAM2、QC 与 Gaussian lift 合同；v2 仅保留为 identity-invalid 负证据。
- [`../configs/worldsim_v32/s3_asset_harvester_v1.yaml`](../configs/worldsim_v32/s3_asset_harvester_v1.yaml)：S3 官方
  Asset Harvester source/weights、冻结视图、生成参数、actor-local adapter 与 StreetGS 回注合同。
- [`../configs/worldsim_v32/s2_3dgic_v1.yaml`](../configs/worldsim_v32/s2_3dgic_v1.yaml)：S2 的 train-only
  depth-guided cross-view、Telea unseen completion、`GENERATED_BACKGROUND` 与 held-out 门禁合同。
- [`../configs/worldsim_v32/s4_harmonizer_v1.yaml`](../configs/worldsim_v32/s4_harmonizer_v1.yaml)：S4 的
  G0/G1/G2 固定输入、Harmonizer non-temporal JIT runtime adaptation、语义保持门与只读 2D provenance 合同。
- [`../configs/worldsim_v32/r0_final_integration_v1.yaml`](../configs/worldsim_v32/r0_final_integration_v1.yaml)：R0 的
  输入资产、semantic extension、P2-style mixed storage、P3-style chunk package、三视角 replay 与资源门合同。
- [`../scripts/run_worldsim_v32_r0_integration.py`](../scripts/run_worldsim_v32_r0_integration.py)：canonical R0
  一次性 formal runner；生产链为 S1 extended semantics + S2 mixed scene + S3 generated-actor override + exact chunk。

## V3.1 冻结文件

- [`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_1.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_1.md)：已执行并收口的
  V3.1 权威计划；
- [`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3.md)：被 V3.1 取代的初始计划；
- [`archive/2026-08/worldsim-v3.1/`](archive/2026-08/worldsim-v3.1/README.md)：上述计划与 R0 收口时
  STATUS/EXPERIMENTS/FAILURES 快照的归档副本。

两份计划继续保留在 `docs/` 根目录，是因为 R0 protocol、snapshot hash 和历史链接引用原路径。它们是
hash-stable compatibility copies，不是当前计划，不得从其中未启动的 F1、P4、D3/D4、A3 formal/R2–R4 恢复执行。

## 历史路线

- V1：动态重建历史路线，已冻结；
- V2：M0–M4 `done`，M5 部分执行后冻结，M6–M8 不再授权；
- V3/V3.1：A0–A4、F0 与 R0 已闭环，终态 `none_plan_complete`；
- 三场景只支持模型消融和工程判断，不是新 benchmark；
- identity、actor binding、基础轨迹编辑和 scene graph 是继承基础设施，不是 V3.1 新贡献。

V2 原计划
[`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)
保留在原位，只用于解释历史 run，后续不再修改。

## 归档索引

- [WorldSim V3.1 终局归档](archive/2026-08/worldsim-v3.1/README.md)
- [2026-08 归档索引](archive/2026-08/README.md)
- [动态重建 V1 终态](archive/2026-07/dynamic-reconstruction-v1/README.md)
- [2026-07 总归档](archive/2026-07/README.md)
- [cut-in mining 关闭归档](archive/2026-07/cutin-mining-closed/README.md)
- [V2 启动前清理账本](archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md)
- [`run_manifests/`](run_manifests/README.md)：历史轻量 manifests，不是 V3.1 formal run。

归档中的“当前任务”“下一步”和命令只解释当时发生了什么，不构成执行授权。

## 文档规则

- 根目录只维护当前事实源、长期环境/资产文档和为 hash/链接兼容保留的冻结计划；
- 实际 run、文件 hash 和现场审计高于旧文档中的驻留描述；
- 完成、阻塞或拒绝任务后，同步更新 STATUS / EXPERIMENTS / FAILURES 与对应计划或归档；
- task 状态只使用 `pending/running/blocked/done/rejected`；
- 工程 `blocked`、研究 `rejected` 和任务 `done` 必须分开；
- 人工 verdict 只能由用户或指定评审者填写。
