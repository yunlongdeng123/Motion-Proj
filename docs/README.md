# Motion-Proj 文档入口

> **当前阶段**：OccGS-Resim V7 单卡 feasibility 已完成，核心方法验证尚未完成。
> **当前决策**：`modify_method_then_scale`；先补方法与证据，不直接扩场景。
> **唯一状态入口**：[`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)。

## 建议阅读顺序

1. [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)：当前进展、当前任务、执行边界与下一步顺序；
2. [`OCCGS_RESIM_AUTORESEARCH_PLAN_V7.md`](OCCGS_RESIM_AUTORESEARCH_PLAN_V7.md)：V7 feasibility 之后的研究计划与门禁；
3. [`EXPERIMENTS.md`](EXPERIMENTS.md)：仅保留 V7 的实验事实和证据路径；
4. [`OCCGS_FINAL_REPORT.md`](OCCGS_FINAL_REPORT.md)：V7 feasibility 轮次的收口审计；
5. [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)：当前必须处理的风险及仍有效的历史禁令。

归档文档中的“当前任务”“下一步”“允许执行”等措辞均已过期，不构成执行授权。

## 当前文档职责

| 文档 | 只负责什么 | 不负责什么 |
|---|---|---|
| `RESEARCH_STATUS.md` | 当前状态、当前任务、优先级、授权边界 | 详细实验日志 |
| `OCCGS_RESIM_AUTORESEARCH_PLAN_V7.md` | 工作假设、阶段门禁、停止条件 | 声称尚未获得的结果 |
| `EXPERIMENTS.md` | V7 已发生的实验事实与证据路径 | 解释历史路线、授权续跑 |
| `OCCGS_FINAL_REPORT.md` | 已结束 feasibility 轮次的结论边界 | 充当当前任务队列 |
| `RESEARCH_FAILURES.md` | 防重复约束、V7 未决风险、重开条件 | 原始 run 指标 |

运维资料仍单独保留：

- [`ENVIRONMENT.md`](ENVIRONMENT.md)：环境与路径；
- [`THIRD_PARTY.md`](THIRD_PARTY.md)：第三方依赖；
- [`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md)：非 Git 产物保留策略；
- [`MACHINE_MIGRATION.md`](MACHINE_MIGRATION.md)：换机接续。

## 归档

- [`archive/2026-07/README.md`](archive/2026-07/README.md)：V1–V7 历史资料分类索引；
- `archive/2026-07/v6/`：已拒绝的 ReSim C1 V6 计划与终报；
- `archive/2026-07/v7-feasibility/`：V7 原始长计划、阶段报告和整理前事实源快照；
- `run_manifests/`：早期轻量 run 证据，不等同于 V7 当前 run 的完整 provenance。

所有原始 run、checkpoint、指标与 review material 仍按各自路径保留；归档只改变文档信息架构，不改写实验事实。
