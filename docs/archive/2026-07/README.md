# 2026-07 研究文档归档

> **归档日期**：2026-07-18（Route Pivot / V5 审计于 2026-07-20 迁入）
> **权威性**：历史证据，不是当前计划
> **当前入口**：[`../../RESEARCH_STATUS.md`](../../RESEARCH_STATUS.md)
> **已关闭的 C1 计划（仍在活跃 docs）**：[`../../MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md`](../../MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md)、[`../../C1_V6_FINAL_REPORT.md`](../../C1_V6_FINAL_REPORT.md)

本目录保存 Motion-Proj 从 explicit dynamics projection 转向 physics preference alignment，再到 SVD
common-prefix sibling、Route Pivot（V5）与 backbone 迁移审计期间的计划、预注册、报告、人工评审协议和
自主研究提示词。

这些文件按当时语境原样归档，因此可能仍包含“唯一前向计划”“当前任务”“下一步”“awaiting reviews”
等已经过期的措辞；正文中偶见历史路径 `docs/ROUTE_PIVOT_*.md`（现为本目录内同名文件）。任何 agent
或研究者都不得据此启动实验、恢复训练、请求人审或配置双卡。

发生冲突时，按以下顺序处理：正式 run → `docs/EXPERIMENTS.md` → `docs/RESEARCH_STATUS.md` →
`docs/RESEARCH_FAILURES.md` → 本归档。

## Explicit projection 与早期 Autoresearch

| 文件 | 历史用途 |
|---|---|
| [`MOTION_PROJ_CVPR_PLAN.md`](MOTION_PROJ_CVPR_PLAN.md) | 最初 dynamics projection distillation 论文/工程方案 |
| [`CVPR2027_PLAN.md`](CVPR2027_PLAN.md) | P2 V1/V2 结构诊断与旧里程碑总计划 |
| [`AUTORESEARCH_EXPERIMENT_PLAN.md`](AUTORESEARCH_EXPERIMENT_PLAN.md) | C0/P0/P1/E0/F0/F1 低成本实验预注册 |
| [`AUTORESEARCH_LITERATURE_MATRIX.md`](AUTORESEARCH_LITERATURE_MATRIX.md) | 2024–2026 最近邻方法矩阵 |
| [`AUTORESEARCH_PHASE2_PREREGISTRATION.md`](AUTORESEARCH_PHASE2_PREREGISTRATION.md) | Phase 2 冻结假设与门禁 |
| [`AUTORESEARCH_PHASE2_REPORT.md`](AUTORESEARCH_PHASE2_REPORT.md) | Phase 2 结果报告 |
| [`AUTORESEARCH_ROUTE_DECISION.md`](AUTORESEARCH_ROUTE_DECISION.md) | explicit projection 路线停止决策与 corrigendum |
| [`AUTORESEARCH_RETROSPECTIVE_2026-07.md`](AUTORESEARCH_RETROSPECTIVE_2026-07.md) | 2026-07-11 至 07-14 早期复盘 |
| [`PROJECTION_INSPECTION.md`](PROJECTION_INSPECTION.md) | 旧 P1 projection target 人工检查协议 |
| [`AUTORESEARCH_PHASE2.prompt.md`](AUTORESEARCH_PHASE2.prompt.md) | 当时用于续跑 Phase 2 的完整自主研究提示词 |

## Physics preference / DrivePO

| 文件 | 历史用途 |
|---|---|
| [`PHYSICS_DPO_AUTORESEARCH_PLAN.md`](PHYSICS_DPO_AUTORESEARCH_PLAN.md) | Physics-DPO v1 计划 |
| [`PHYSICS_DPO_AUTORESEARCH_PLAN_V2_AC_REVISED.md`](PHYSICS_DPO_AUTORESEARCH_PLAN_V2_AC_REVISED.md) | SAP-DPO / preference v2 AC 修订计划 |
| [`PHYSICS_DPO_AUTORESEARCH_PLAN_V3.md`](PHYSICS_DPO_AUTORESEARCH_PLAN_V3.md) | uncertainty-aware partial order v3 计划 |
| [`PHYSICS_DPO_AUTORESEARCH_PLAN_V4_AC_REVISED.md`](PHYSICS_DPO_AUTORESEARCH_PLAN_V4_AC_REVISED.md) | common-support UPO 与唯一 earlier-fork fallback v4 计划 |
| [`PHYSICS_DPO_PA0_REVIEW_SPLIT_SCHEMA.md`](PHYSICS_DPO_PA0_REVIEW_SPLIT_SCHEMA.md) | PA0 review、scene split 与 schema |
| [`PHYSICS_DPO_PA1_BRANCH_PROTOCOL.md`](PHYSICS_DPO_PA1_BRANCH_PROTOCOL.md) | PA1 common-prefix sibling pilot 协议 |
| [`PA1_BRANCH_HUMAN_REVIEW_PROMPT.md`](PA1_BRANCH_HUMAN_REVIEW_PROMPT.md) | 已完成人工结构盲审提示词 |
| [`PA2_PAIR_HUMAN_REVIEW_PROMPT.md`](PA2_PAIR_HUMAN_REVIEW_PROMPT.md) | 已完成 48-case preference 人审提示词 |
| [`PA2_CANDIDATE_FALLBACK_HUMAN_REVIEW_PROMPT.md`](PA2_CANDIDATE_FALLBACK_HUMAN_REVIEW_PROMPT.md) | fallback 预备提示词；机器门禁失败后未启动人审 |

## Route Pivot（V5）与 backbone 迁移

| 文件 | 历史用途 |
|---|---|
| [`MOTION_ROUTE_PIVOT_AUTORESEARCH_PLAN_V5.md`](MOTION_ROUTE_PIVOT_AUTORESEARCH_PLAN_V5.md) | V5 预注册计划（Route A/B/C）；`done`，不得恢复已拒绝任务 |
| [`ROUTE_PIVOT_FINAL_REPORT.md`](ROUTE_PIVOT_FINAL_REPORT.md) | V5 终报；最多 3 个后续实验建议，不含执行授权 |
| [`ROUTE_PIVOT_LITERATURE_MATRIX.md`](ROUTE_PIVOT_LITERATURE_MATRIX.md) | RP-LIT-01 一手文献矩阵 |
| [`ROUTE_PIVOT_TEMPORAL_AUDIT.md`](ROUTE_PIVOT_TEMPORAL_AUDIT.md) | RP-R1-02 真实时间与 SVD fps 审计 |
| [`ROUTE_PIVOT_REAL_MOTION_TARGET_AUDIT.md`](ROUTE_PIVOT_REAL_MOTION_TARGET_AUDIT.md) | RP-A0-03 真实 ego–actor target 审计 |
| [`ROUTE_PIVOT_MOTION_FEATURE_AUDIT.md`](ROUTE_PIVOT_MOTION_FEATURE_AUDIT.md) | RP-A1-SCAN-04A 冻结特征 scan（actor rejected） |
| [`ROUTE_PIVOT_NATURAL_ROLLOUT_AUDIT.md`](ROUTE_PIVOT_NATURAL_ROLLOUT_AUDIT.md) | RP-B0-05 natural-rollout ceiling（rejected） |
| [`BACKBONE_MIGRATION_AUDIT.md`](BACKBONE_MIGRATION_AUDIT.md) | RP-C0-07 只读迁移审计；选择 C1=ReSim feasibility |

V5 选择的 C1 由活跃层 [`MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md`](../../MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md)
执行并已在 H1（`RF-18`）关闭；不得从本目录 V5 终报“后续实验”列表直接启动下载/推理/训练。

## 如何复查历史

1. 先从 [`../../RESEARCH_FAILURES.md`](../../RESEARCH_FAILURES.md) 找到对应 `RF-*` 条目；
2. 再读取本目录中的原计划/报告，确认当时预注册阈值和结论边界；
3. 回到 `/root/autodl-tmp/runs/` 核对 manifest、resolved config、指标、人工 verdict 与终止标记；
4. 必要时使用 `git log --follow -- <path>` 追查文档演化。

本目录不接收新的当前计划。未来计划必须回到活跃 `docs/` 层，并由 `RESEARCH_STATUS.md` 明确授权。
