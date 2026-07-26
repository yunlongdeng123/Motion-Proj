# Motion-Proj 文档导航

更新时间：2026-07-26

当前研究主线已经从 nuScenes cut-in 挖掘转为：

> 动态驾驶场景重建 → 对象级轨迹编辑压力测试 → 遮挡/去遮挡与感知一致性 → 基于稳定失败选择创新。

## 当前必须先读

1. [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)：唯一当前状态与授权入口；
2. [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)：失败、边界和防重复规则；
3. [`EXPERIMENTS.md`](EXPERIMENTS.md)：当前路线实验注册表；
4. [`DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md`](DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md)：下一阶段完整方案；
5. [`human-review/dynamic-reconstruction-plan-v1/README.md`](human-review/dynamic-reconstruction-plan-v1/README.md)：本轮人工审核包。

## 当前路线

- 主基线：AD-GS exact reproduction；
- 前馈对照：DGGT inference-only；
- 编辑参考：DrivingEditor；
- 可见性/补密方向必查：VAD-GS；
- 主数据：AD-GS 官方 nuScenes 六 scenes、固定 60 帧、三前向相机；
- 当前状态：计划完成，等待用户开放内存与 GPU；
- 当前禁止：训练、预处理、权重下载、cut-in 继续调参、baseline 前集成改进。

## 长期维护文档

- [`ENVIRONMENT.md`](ENVIRONMENT.md)：当前 Motion-Proj 环境；
- [`THIRD_PARTY.md`](THIRD_PARTY.md)：第三方代码与许可证；
- [`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md)：产物保留/清理规则；
- [`MACHINE_MIGRATION.md`](MACHINE_MIGRATION.md)：迁移说明；
- [`run_manifests/`](run_manifests/)：轻量 run manifests。

新路线的 AD-GS/DGGT 独立环境将在用户开放资源并通过 `DR-M2-ENV-ASSET-01` 后登记；不要提前把计划值写成已安装事实。

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
