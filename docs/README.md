# Motion-Proj 文档导航

- 更新时间：2026-08-05
- 当前路线：面向世界仿真的动态驾驶 3DGS 复现、模型增强与工程化 V3
- 当前任务：`WS-V3-A0-NATIVE-BASELINE-01`（`running`）
- 唯一当前计划：[`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3.md)

## 恢复顺序

1. [`../AGENTS.md`](../AGENTS.md)：仓库级环境、研究连续性和提交约定；
2. [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)：唯一当前状态与授权入口；
3. [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)：跨路线失败与禁止重复项；
4. [`EXPERIMENTS.md`](EXPERIMENTS.md)：V2 冻结证据和 V3 task 注册表；
5. [`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3.md)：当前执行合同。

清空对话或换实例后，不从 V2 计划、归档报告或旧 terminal 恢复“下一步”。当前唯一入口由
`RESEARCH_STATUS.md` 和 V3 计划共同定义。

## 路线边界

- V1：动态重建历史路线，已冻结；
- V2：M0–M4 `done`，M5 部分执行后冻结，M6–M8 不再授权；
- V3：当前路线，完成 A0–A4 WorldSim 模型链和消融；
- 三场景是模型消融场，不是新 benchmark；
- identity、actor binding、基础轨迹编辑和 scene graph 是继承基础设施，不是 V3 贡献。

V2 原计划
[`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)
保留在原位，只用于解释历史 run，后续不再修改。

## 当前维护文档

- [`ENVIRONMENT.md`](ENVIRONMENT.md)：机器、环境、数据、权重和镜像策略；
- [`THIRD_PARTY.md`](THIRD_PARTY.md)：第三方代码、commit、license 与驻留状态；
- [`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md)：资产保留边界；
- [`MACHINE_MIGRATION.md`](MACHINE_MIGRATION.md)：换机与清空 context 后恢复顺序；
- [`run_manifests/`](run_manifests/)：历史轻量 manifests，不是 V3 formal run。

## 历史归档

- [动态重建 V1 终态](archive/2026-07/dynamic-reconstruction-v1/README.md)
- [2026-07 总归档](archive/2026-07/README.md)
- [V2 启动前清理账本](archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md)
- [V1 人工审核材料](archive/2026-07/dynamic-reconstruction-v1/human-review/dynamic-reconstruction-results-v1/README.md)

归档中的“当前任务”“下一步”和命令只解释当时发生了什么，不构成执行授权。

## 文档规则

- 活跃根目录只保留当前状态、失败账本、实验注册表、当前计划和长期维护文档；
- 实际 run、文件 hash 和现场审计高于旧文档中的驻留描述；
- 完成、阻塞或拒绝一个 V3 task 后，同步更新 PLAN / STATUS / EXPERIMENTS / FAILURES；
- task 状态只使用 `pending/running/blocked/done/rejected`；
- 人工 verdict 只能由用户或指定评审者填写。
