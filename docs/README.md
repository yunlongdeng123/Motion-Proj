# Motion-Proj 文档导航

- 更新时间：2026-08-02
- 当前路线：动态驾驶场景可编辑重建与失败诊断 V2
- 当前状态：用户已授权，尚未执行研究里程碑；下一步只执行 M0
- 唯一当前计划：[`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)

## 开始执行前必读

1. [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)：唯一当前状态与授权入口；
2. [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)：跨路线失败与禁止重复项；
3. [`EXPERIMENTS.md`](EXPERIMENTS.md)：V2 里程碑注册表；
4. [`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)：当前执行合同；
5. [`../AGENTS.md`](../AGENTS.md)：仓库级环境、研究连续性与提交约定。

清空对话上下文后，不从归档文档恢复“下一步”。当前唯一入口是 V2 M0：
`DR-V2-M0-BOOTSTRAP-01`。

## 2026-08-02 现场结论

- V1 已结束且终态不变：AD-GS 六场景 exact reproduction `done`；DGGT inference 未运行；
  V1 的身份候选 novelty `rejected`。
- 六场景 AD-GS `model_60000` checkpoint、官方 render、指标和 processed 输入仍驻留，V2 不重复训练。
- DGGT 完整预下载候选已驻留：
  `/root/autodl-tmp/checkpoints/dggt_preload/model_latest_nuscenes.pt`，大小
  `5,411,266,466` bytes，SHA-256
  `fd15644b3a878849470cbf5f0f9eae39167cfec1b853092898ae754c4f3acde9`。
- DriveStudio 源码与环境存在，但 V2 的 `scene-0230/0242/0255` processed data 和 actor-aware
  checkpoint 尚不存在；M3 不能把历史 `003/004/005` 资产冒充 V2 baseline。
- V2 run namespace 尚不存在；本次只做文档归档、事实校准与存储清理，不算 M0 已执行。

## 当前维护文档

- [`ENVIRONMENT.md`](ENVIRONMENT.md)：实际机器、环境、数据、权重和镜像策略；
- [`THIRD_PARTY.md`](THIRD_PARTY.md)：V2 第三方代码、commit、license 与驻留状态；
- [`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md)：V2 保留边界；
- [`MACHINE_MIGRATION.md`](MACHINE_MIGRATION.md)：换机与清空 context 后恢复顺序；
- [`run_manifests/`](run_manifests/)：早期路线的轻量历史 manifests，不是 V2 当前 run。

## 历史归档

- [动态重建 V1 终态](archive/2026-07/dynamic-reconstruction-v1/README.md)
- [2026-07 总归档](archive/2026-07/README.md)
- [V2 启动前清理账本](archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md)
- [V1 人工审核材料](archive/2026-07/dynamic-reconstruction-v1/human-review/dynamic-reconstruction-results-v1/README.md)

归档中的“当前任务”“下一步”和命令只解释当时发生了什么，不构成执行授权。

## 文档规则

- 活跃根目录只保留当前状态、失败账本、实验注册表、当前计划和长期维护文档；
- 旧计划、结束总结、阶段报告和 Agent 心跳终态统一进入按年月/路线命名的归档目录；
- 实际 run、文件 hash 和现场审计高于旧文档中的驻留描述；
- 完成、阻塞或拒绝一个 V2 里程碑后，同步更新 PLAN / STATUS / EXPERIMENTS；
- 人工 verdict 只能由用户或指定评审者填写。
