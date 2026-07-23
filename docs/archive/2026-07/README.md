# 2026-07 研究文档归档

> **权威性**：历史证据，不是当前计划。
> **当前入口**：[`../../README.md`](../../README.md) → [`../../RESEARCH_STATUS.md`](../../RESEARCH_STATUS.md)。

本目录保存 Motion-Proj 从 explicit projection、physics preference、Route Pivot V5、ReSim C1 V6 到
OccGS-Resim V7 feasibility 的历史计划、报告、提示词与整理前事实源。

归档文件保留当时措辞，可能包含“当前任务”“下一步”“approved”“awaiting reviews”等过期状态。任何 agent 或
研究者都不得据此启动实验、恢复训练、请求人审或切换硬件。发生冲突时，以实际 run → 当前
`EXPERIMENTS.md` → `RESEARCH_STATUS.md` → `RESEARCH_FAILURES.md` → 本归档为准。

## 1. Explicit projection 与早期 Autoresearch

| 文件 | 历史用途 |
|---|---|
| [`MOTION_PROJ_CVPR_PLAN.md`](MOTION_PROJ_CVPR_PLAN.md) | 最初 dynamics projection distillation 方案 |
| [`CVPR2027_PLAN.md`](CVPR2027_PLAN.md) | P2 V1/V2 结构诊断与旧总计划 |
| [`AUTORESEARCH_EXPERIMENT_PLAN.md`](AUTORESEARCH_EXPERIMENT_PLAN.md) | C0/P0/P1/E0/F0/F1 低成本实验预注册 |
| [`AUTORESEARCH_LITERATURE_MATRIX.md`](AUTORESEARCH_LITERATURE_MATRIX.md) | 2024–2026 最近邻矩阵 |
| [`AUTORESEARCH_PHASE2_PREREGISTRATION.md`](AUTORESEARCH_PHASE2_PREREGISTRATION.md) | Phase 2 假设与门禁 |
| [`AUTORESEARCH_PHASE2_REPORT.md`](AUTORESEARCH_PHASE2_REPORT.md) | Phase 2 结果 |
| [`AUTORESEARCH_ROUTE_DECISION.md`](AUTORESEARCH_ROUTE_DECISION.md) | explicit projection 停止决策 |
| [`AUTORESEARCH_RETROSPECTIVE_2026-07.md`](AUTORESEARCH_RETROSPECTIVE_2026-07.md) | 早期复盘 |
| [`PROJECTION_INSPECTION.md`](PROJECTION_INSPECTION.md) | 旧 P1 人工检查协议 |
| [`AUTORESEARCH_PHASE2.prompt.md`](AUTORESEARCH_PHASE2.prompt.md) | 历史自主研究提示词 |

## 2. Physics preference / DrivePO

| 文件 | 历史用途 |
|---|---|
| [`PHYSICS_DPO_AUTORESEARCH_PLAN.md`](PHYSICS_DPO_AUTORESEARCH_PLAN.md) | Physics-DPO v1 |
| [`PHYSICS_DPO_AUTORESEARCH_PLAN_V2_AC_REVISED.md`](PHYSICS_DPO_AUTORESEARCH_PLAN_V2_AC_REVISED.md) | SAP-DPO / preference v2 |
| [`PHYSICS_DPO_AUTORESEARCH_PLAN_V3.md`](PHYSICS_DPO_AUTORESEARCH_PLAN_V3.md) | uncertainty-aware partial order v3 |
| [`PHYSICS_DPO_AUTORESEARCH_PLAN_V4_AC_REVISED.md`](PHYSICS_DPO_AUTORESEARCH_PLAN_V4_AC_REVISED.md) | UPO 与 earlier-fork fallback v4 |
| [`PHYSICS_DPO_PA0_REVIEW_SPLIT_SCHEMA.md`](PHYSICS_DPO_PA0_REVIEW_SPLIT_SCHEMA.md) | review、scene split 与 schema |
| [`PHYSICS_DPO_PA1_BRANCH_PROTOCOL.md`](PHYSICS_DPO_PA1_BRANCH_PROTOCOL.md) | common-prefix sibling protocol |
| [`PA1_BRANCH_HUMAN_REVIEW_PROMPT.md`](PA1_BRANCH_HUMAN_REVIEW_PROMPT.md) | 已完成结构盲审提示词 |
| [`PA2_PAIR_HUMAN_REVIEW_PROMPT.md`](PA2_PAIR_HUMAN_REVIEW_PROMPT.md) | 已完成 48-case 提示词 |
| [`PA2_CANDIDATE_FALLBACK_HUMAN_REVIEW_PROMPT.md`](PA2_CANDIDATE_FALLBACK_HUMAN_REVIEW_PROMPT.md) | 未启动 fallback 人审提示词 |

## 3. Route Pivot V5

| 文件 | 历史用途 |
|---|---|
| [`MOTION_ROUTE_PIVOT_AUTORESEARCH_PLAN_V5.md`](MOTION_ROUTE_PIVOT_AUTORESEARCH_PLAN_V5.md) | V5 Route A/B/C 计划；已关闭 |
| [`ROUTE_PIVOT_FINAL_REPORT.md`](ROUTE_PIVOT_FINAL_REPORT.md) | V5 终报 |
| [`ROUTE_PIVOT_LITERATURE_MATRIX.md`](ROUTE_PIVOT_LITERATURE_MATRIX.md) | 一手文献矩阵 |
| [`ROUTE_PIVOT_TEMPORAL_AUDIT.md`](ROUTE_PIVOT_TEMPORAL_AUDIT.md) | 真实时间与 SVD fps 审计 |
| [`ROUTE_PIVOT_REAL_MOTION_TARGET_AUDIT.md`](ROUTE_PIVOT_REAL_MOTION_TARGET_AUDIT.md) | ego–actor target 审计 |
| [`ROUTE_PIVOT_MOTION_FEATURE_AUDIT.md`](ROUTE_PIVOT_MOTION_FEATURE_AUDIT.md) | frozen feature scan |
| [`ROUTE_PIVOT_NATURAL_ROLLOUT_AUDIT.md`](ROUTE_PIVOT_NATURAL_ROLLOUT_AUDIT.md) | natural rollout ceiling |
| [`BACKBONE_MIGRATION_AUDIT.md`](BACKBONE_MIGRATION_AUDIT.md) | ReSim/VISTA/OpenDWM 等迁移审计 |

## 4. ReSim C1 V6

目录：[`v6/`](v6/)

| 文件 | 历史用途 |
|---|---|
| [`v6/MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md`](v6/MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md) | V6 单卡 action-response 计划；H1 已 rejected |
| [`v6/C1_V6_FINAL_REPORT.md`](v6/C1_V6_FINAL_REPORT.md) | C1B-00/01/02 收口；`RF-18` 证据 |

V6 的 C1P/C1S 未运行，禁止通过换 seed、降门槛或扩大同分布 scene 重开。

## 5. OccGS-Resim V7 feasibility

目录：[`v7-feasibility/`](v7-feasibility/)

该目录保存整理前原始长计划、阶段报告和 V1–V7 全量事实源快照。该阶段当时的准确结论是
“feasibility 完成，H1/H2/H3 未验证”；之后 V7.1 已正式运行并在 H1-11D rejected，见下一节。

| 文件 | 历史用途 |
|---|---|
| [`v7-feasibility/OCCGS_RESIM_AUTORESEARCH_PLAN_V7_EXECUTED.md`](v7-feasibility/OCCGS_RESIM_AUTORESEARCH_PLAN_V7_EXECUTED.md) | 1909 行原执行计划、append-only 进度与旧关机授权 |
| [`v7-feasibility/OCCGS_E0_ENV_MANIFEST.md`](v7-feasibility/OCCGS_E0_ENV_MANIFEST.md) | 环境清单 |
| [`v7-feasibility/OCCGS_THIRD_PARTY_AUDIT.md`](v7-feasibility/OCCGS_THIRD_PARTY_AUDIT.md) | 第三方与数据覆盖审计 |
| [`v7-feasibility/OCCGS_LICENSE_AND_DATA_POLICY.md`](v7-feasibility/OCCGS_LICENSE_AND_DATA_POLICY.md) | license / data policy |
| [`v7-feasibility/OCCGS_DATA_PREPARATION.md`](v7-feasibility/OCCGS_DATA_PREPARATION.md) | D0 数据准备 |
| [`v7-feasibility/OCCGS_RECONSTRUCTION_BASELINE.md`](v7-feasibility/OCCGS_RECONSTRUCTION_BASELINE.md) | B0 重建结果 |
| [`v7-feasibility/OCCGS_OCCUPANCY_STATE.md`](v7-feasibility/OCCGS_OCCUPANCY_STATE.md) | O0 occupancy artifact |
| [`v7-feasibility/OCCGS_COUNTERFACTUAL_PROTOCOL.md`](v7-feasibility/OCCGS_COUNTERFACTUAL_PROTOCOL.md) | S0/C0 历史协议与机器 screen |
| [`v7-feasibility/OCCGS_L0_COMPLETION.md`](v7-feasibility/OCCGS_L0_COMPLETION.md) | L0 Telea feasibility |
| [`v7-feasibility/OCCGS_FINAL_REPORT_ORIGINAL.md`](v7-feasibility/OCCGS_FINAL_REPORT_ORIGINAL.md) | 整理前 V7 终报原文 |
| [`v7-feasibility/ARTIFACT_RETENTION_20260719.md`](v7-feasibility/ARTIFACT_RETENTION_20260719.md) | 2026-07-19 历史清理清单、hash 与验证结果 |
| [`v7-feasibility/EXPERIMENTS_V1_V7_SNAPSHOT.md`](v7-feasibility/EXPERIMENTS_V1_V7_SNAPSHOT.md) | 整理前 V1–V7 全量实验事实源 |
| [`v7-feasibility/RESEARCH_STATUS_LEGACY_SNAPSHOT.md`](v7-feasibility/RESEARCH_STATUS_LEGACY_SNAPSHOT.md) | 整理前混合 V5/V6/V7 状态文件 |
| [`v7-feasibility/RESEARCH_FAILURES_RF01_RF18.md`](v7-feasibility/RESEARCH_FAILURES_RF01_RF18.md) | 完整旧 RF 原文与重开条件 |

## 6. OccGS-Resim V7.1 H1 reject

目录：[`v7.1-h1-reject/`](v7.1-h1-reject/)

| 文件 | 历史用途 |
|---|---|
| [`v7.1-h1-reject/OCCGS_RESIM_AUTORESEARCH_PLAN_V7_SUPERSEDED.md`](v7.1-h1-reject/OCCGS_RESIM_AUTORESEARCH_PLAN_V7_SUPERSEDED.md) | V7 feasibility 后过渡计划；已被 V7.1 取代 |
| [`v7.1-h1-reject/OCCGS_RESIM_AUTORESEARCH_PLAN_V7.1_EXECUTED.md`](v7.1-h1-reject/OCCGS_RESIM_AUTORESEARCH_PLAN_V7.1_EXECUTED.md) | V7.1 完整执行计划；H1 reject 后失效 |
| [`v7.1-h1-reject/V7_1_H1_REJECT_RETROSPECTIVE.md`](v7.1-h1-reject/V7_1_H1_REJECT_RETROSPECTIVE.md) | H1-CERT/H1-PROJ 收口复盘与失败分解 |
| [`v7.1-h1-reject/RESEARCH_STATUS_H1_REJECT_SNAPSHOT.md`](v7.1-h1-reject/RESEARCH_STATUS_H1_REJECT_SNAPSHOT.md) | 转入 Post-OccGS 前的状态快照 |
| [`v7.1-h1-reject/RESEARCH_FAILURES_H1_REJECT_SNAPSHOT.md`](v7.1-h1-reject/RESEARCH_FAILURES_H1_REJECT_SNAPSHOT.md) | 详细补账前的失败账本快照 |
| [`v7.1-h1-reject/EXPERIMENTS_H1_REJECT_SNAPSHOT.md`](v7.1-h1-reject/EXPERIMENTS_H1_REJECT_SNAPSHOT.md) | V7.1 收口时实验事实快照 |
| [`v7.1-h1-reject/README.md`](v7.1-h1-reject/README.md) | 目录说明、文件 hash 与备份边界 |

V7.1 工程 gate 11A/11B/11C 通过；冻结的 11D matched pilot 同时拒绝 H1-CERT 与 H1-PROJ，
未进入 H2/H3/scale。

## 7. Event-first N0/N1 mini reject

目录：[`event-first-mini-reject/`](event-first-mini-reject/)

该目录索引 Post-OccGS 路线的 N0 asset pass 与 N1 mini event-pool reject。N1 在冻结的 topology +
exact-target-token interaction 定义下得到 45 eligible actors、71 transitions、22 topology pass、
0 interaction positive 和 0 same-actor pair，因此 N2–N5 未触发。

配置逐字快照保存在不可变 run 的 `resolved.yaml`；归档 README 记录 commit、fingerprint、hash、路径和禁止复开项。

## 8. 如何复查历史

1. 先读当前 [`../../RESEARCH_FAILURES.md`](../../RESEARCH_FAILURES.md)，定位适用的 RF 或 V7 risk；
2. 再读对应归档计划/报告，确认当时阈值、证据边界与路径；
3. 回到 `/root/autodl-tmp/runs/` 核对原始 config、metrics、checkpoint、review 与终态标记；
4. 用 `git log --follow -- <path>` 追查文档演化。

本目录不接收新的当前计划。未来状态、实验与授权只写回活跃 `docs/` 层。
