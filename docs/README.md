# Motion-Proj 文档入口

> **当前阶段**：V7.1 H1 已 reject；N1 mini 第一次 reject；full-domain 第二次人工审核以
> 2 TP / 35 FP 再次 `REJECTED`；kinematics-first 第三版已预注册、formal evaluation 待运行。
> **当前决策**：只推进 scene-disjoint `N1-EVENT-KINEMATIC-01` 并交付第三次人工审核材料；
> N2–N5 保持封闭。
> **唯一状态入口**：[`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)。

## 建议阅读顺序

1. [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)：当前结论、授权边界、阻塞和任务队列；
2. [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)：失败机制、已知卡点、禁止重复项和复开条件；
3. [`N1_KINEMATIC_PREREGISTRATION.md`](N1_KINEMATIC_PREREGISTRATION.md)：第三版数据隔离、
   2 Hz 运动学、branch-safe interaction 与机器/人工门槛；
4. [`N1_FULLDOMAIN_EVENT_POOL_REPORT.md`](N1_FULLDOMAIN_EVENT_POOL_REPORT.md)：第二版机器海选及其
   37 条人工复核入口（最终裁决以状态与实验账本为准）；
5. [`POST_OCCGS_RESEARCH_DIRECTIONS.md`](POST_OCCGS_RESEARCH_DIRECTIONS.md)：路线调研、替代方向与边界；
6. [`N0_ASSET_AND_EVENT_PREFLIGHT.md`](N0_ASSET_AND_EVENT_PREFLIGHT.md)：scene→map、actor continuity 与最小外部资产；
7. [`N1_MINI_EVENT_POOL_REPORT.md`](N1_MINI_EVENT_POOL_REPORT.md)：第一次 N1 provenance 与失败分解；
8. [`EXPERIMENTS.md`](EXPERIMENTS.md)：V7/V7.1 与 event-first 已发生实验的数值事实；
9. [`OCCGS_FINAL_REPORT.md`](OCCGS_FINAL_REPORT.md)：V7 feasibility 轮次的历史收口。

V7.1 执行计划已经完成并归档，不再是当前授权入口：
[`archive/2026-07/v7.1-h1-reject/OCCGS_RESIM_AUTORESEARCH_PLAN_V7.1_EXECUTED.md`](archive/2026-07/v7.1-h1-reject/OCCGS_RESIM_AUTORESEARCH_PLAN_V7.1_EXECUTED.md)。

## 当前文档职责

| 文档 | 负责 | 不负责 |
|---|---|---|
| `RESEARCH_STATUS.md` | 当前状态、任务、优先级、授权边界 | 详细实验流水 |
| `RESEARCH_FAILURES.md` | 失败账本、防重复约束、复开条件 | 隐藏或美化负结果 |
| `POST_OCCGS_RESEARCH_DIRECTIONS.md` | 调研、路线排序、预检闸门 | 自动授权数据/权重下载 |
| `N1_KINEMATIC_PREREGISTRATION.md` | 第三版冻结合同与停止条件 | 事后调门槛或代替人工 verdict |
| `N1_FULLDOMAIN_EVENT_POOL_REPORT.md` | 第二版机器 run provenance 与 audit pack | 把机器 `COMPLETE` 写成人工通过 |
| `N1_MINI_EVENT_POOL_REPORT.md` | N0/N1 正式结果、边界、下一方向 | 复开已拒绝 mini gate |
| `EXPERIMENTS.md` | 已发生实验事实、路径、hash | 为未来路线授权 |
| `OCCGS_FINAL_REPORT.md` | 已结束 feasibility 的历史边界 | 当前任务队列 |

## 运维资料

- [`ENVIRONMENT.md`](ENVIRONMENT.md)：环境与路径；
- [`THIRD_PARTY.md`](THIRD_PARTY.md)：第三方依赖；
- [`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md)：非 Git 产物保留策略；
- [`MACHINE_MIGRATION.md`](MACHINE_MIGRATION.md)：换机接续。

## 归档

- [`archive/2026-07/README.md`](archive/2026-07/README.md)：V1–V7.1 历史索引；
- `archive/2026-07/v7-feasibility/`：V7 feasibility 原计划、阶段报告与旧事实快照；
- `archive/2026-07/v7.1-h1-reject/`：V7.1 完整执行计划、H1 reject 快照、复盘与编辑备份；
- `run_manifests/`：早期轻量 run 证据，不等同于 V7.1 正式 run provenance。

归档文档里的“当前任务”“下一步”“approved”等措辞均已过期，不构成执行授权。原始 run、checkpoint、
指标和 review material 仍按原路径保留；归档只改变文档信息架构，不改写实验事实。
