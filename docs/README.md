# Motion-Proj 文档导航

- 更新时间：2026-08-11
- 当前路线：WorldSim V3.3 对象感知与可维护资产
- 最新有效完成任务：`WS-V33-S1-OBJECT-AWARE-GS-01`（canonical formal r9）
- 研究终态：`running`
- 当前执行授权：仅 `WS-V33-S2-ROADPATCH-INPAINT-01`
- 当前计划：[`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_3.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_3.md)
- P0 审计：[`WS_V33_P0_SOTA_AUDIT.md`](WS_V33_P0_SOTA_AUDIT.md)
- S1 对象场报告：[`WS_V33_S1_OBJECT_AWARE_GS.md`](WS_V33_S1_OBJECT_AWARE_GS.md)
- V3.2 终局归档：[`archive/2026-08/worldsim-v3.2/`](archive/2026-08/worldsim-v3.2/README.md)

## 恢复顺序

1. [`../AGENTS.md`](../AGENTS.md)：仓库级环境、研究连续性和提交约定；
2. [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)：唯一当前状态与执行授权入口；
3. [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)：跨路线失败、禁止重复项与复开条件；
4. [`EXPERIMENTS.md`](EXPERIMENTS.md)：canonical run、hash、指标和任务终态；
5. [`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_3.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_3.md)：当前路线与顺序门禁；
6. [`WS_V33_P0_SOTA_AUDIT.md`](WS_V33_P0_SOTA_AUDIT.md)：V3.3 source/license/weights/hardware 事实边界；
7. [`WS_V33_S1_OBJECT_AWARE_GS.md`](WS_V33_S1_OBJECT_AWARE_GS.md)：immutable base + instance-opacity 的实现、run 与指标。

清空对话、换机或换实例后，不从归档计划、旧 terminal 或计划中的“下一步”恢复动作。只有
`RESEARCH_STATUS.md` 可以授权新的 task、run、训练或评测；当前只授权 V3.3 S2。

## 当前事实源

- [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)：最新状态、终局裁决、机器与工作树边界；
- [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)：仍约束后续路线的失败、负结果和复开条件；
- [`EXPERIMENTS.md`](EXPERIMENTS.md)：V2、V3.1 与 V3.2 的完整实验台账；
- [`ENVIRONMENT.md`](ENVIRONMENT.md)：机器、环境、数据、权重和镜像策略；
- [`THIRD_PARTY.md`](THIRD_PARTY.md)：第三方代码、commit、license 与驻留状态；
- [`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md)：资产保留边界；
- [`MACHINE_MIGRATION.md`](MACHINE_MIGRATION.md)：换机与恢复流程。

## V3.3 当前文件

- [`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_3.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_3.md)：当前计划与任务注册表；
- [`WS_V33_P0_SOTA_AUDIT.md`](WS_V33_P0_SOTA_AUDIT.md)：P0 canonical SOTA/source 审计；
- [`WS_V33_S1_OBJECT_AWARE_GS.md`](WS_V33_S1_OBJECT_AWARE_GS.md)：S1 canonical 对象场报告；
- [`../configs/worldsim_v33/p0_sources_v1.yaml`](../configs/worldsim_v33/p0_sources_v1.yaml)：机器可检验事实配置；
- [`../configs/worldsim_v33/s1_instance_field_v1.yaml`](../configs/worldsim_v33/s1_instance_field_v1.yaml)：S1 冻结配置；
- [`../scripts/audit_worldsim_v33_sources.py`](../scripts/audit_worldsim_v33_sources.py)：只读审计 runner；
- [`../scripts/run_worldsim_v33_s1_instance_field.py`](../scripts/run_worldsim_v33_s1_instance_field.py)：S1 训练/评测 runner；
- `/root/autodl-tmp/runs/worldsim_v33/`：V3.3 唯一正式 run namespace。

## V3.2 冻结文件

- [`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_2.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_2.md)：已执行并收口的
  V3.2 计划兼容副本；
- [`WS_V32_S0_SOTA_AUDIT.md`](WS_V32_S0_SOTA_AUDIT.md)：S0 source/license/weight/hardware 审计兼容副本；
- [`archive/2026-08/worldsim-v3.2/`](archive/2026-08/worldsim-v3.2/README.md)：计划、审计、三份事实源快照与
  SHA-256 清单的权威归档入口；
- [`../configs/worldsim_v32/`](../configs/worldsim_v32/)：S0–S4 与 R0 冻结协议；
- [`../scripts/run_worldsim_v32_r0_integration.py`](../scripts/run_worldsim_v32_r0_integration.py)：canonical R0 runner。

V3.2 最终生产候选固定为：S1 extended semantic sidecars + S2 generated-background mixed scene + S3
generated-actor override + R0 exact chunk package。S4 non-temporal 仅为 excluded diagnostic；S4 temporal 与 S5
保持外部门禁 `blocked`。这些阻塞项不再构成 V3.2 的“下一步”，也不妨碍路线终态为
`none_plan_complete`。

## 历史路线

- V1：动态重建历史路线，已冻结；
- V2：M0–M4 `done`，M5 部分执行后冻结，M6–M8 不再授权；
- V3/V3.1：A0–A4、F0 与 R0 闭环，终态 `none_plan_complete`；
- V3.2：S0–S4 与 R0 完成，S5 外部门禁阻塞，终态 `none_plan_complete`。

V3/V3.1、V3.2 和 V2 的原计划继续保留在 `docs/` 根目录，原因是冻结 protocol、snapshot hash 和历史链接引用
原路径。它们是 hash/link compatibility copies，不是当前计划；不得从其中恢复未启动或外部阻塞的分支。

## 归档索引

- [WorldSim V3.2 终局归档](archive/2026-08/worldsim-v3.2/README.md)
- [WorldSim V3.1 终局归档](archive/2026-08/worldsim-v3.1/README.md)
- [2026-08 归档索引](archive/2026-08/README.md)
- [动态重建 V1 终态](archive/2026-07/dynamic-reconstruction-v1/README.md)
- [2026-07 总归档](archive/2026-07/README.md)
- [cut-in mining 关闭归档](archive/2026-07/cutin-mining-closed/README.md)
- [V2 启动前清理账本](archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md)
- [`run_manifests/`](run_manifests/README.md)：历史轻量 manifests。

## 文档规则

- 根目录只维护当前事实源、长期环境/资产文档和为 hash/链接兼容保留的冻结计划；
- 实际 run、文件 hash 和现场审计高于旧文档中的驻留描述；
- 完成、阻塞或拒绝任务后，同步更新 STATUS / EXPERIMENTS / FAILURES 与对应计划或归档；
- task 状态只使用 `pending/running/blocked/done/rejected`；
- 工程 `blocked`、研究 `rejected` 和任务 `done` 必须分开；
- 人工 verdict 只能由用户或指定评审者填写。
