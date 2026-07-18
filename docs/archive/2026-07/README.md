# 2026-07 研究文档归档

> **归档日期**：2026-07-18
> **权威性**：历史证据，不是当前计划
> **当前入口**：[`../../RESEARCH_STATUS.md`](../../RESEARCH_STATUS.md)

本目录保存 Motion-Proj 从 explicit dynamics projection 转向 physics preference alignment，再到 SVD
common-prefix sibling 路线停止期间的计划、预注册、报告、人工评审协议和自主研究提示词。

这些文件按当时语境原样归档，因此可能仍包含“唯一前向计划”“当前任务”“下一步”“awaiting reviews”
等已经过期的措辞。任何 agent 或研究者都不得据此启动实验、恢复训练、请求人审或配置双卡。

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

## 如何复查历史

1. 先从 [`../../RESEARCH_FAILURES.md`](../../RESEARCH_FAILURES.md) 找到对应 `RF-*` 条目；
2. 再读取本目录中的原计划/报告，确认当时预注册阈值和结论边界；
3. 回到 `/root/autodl-tmp/runs/` 核对 manifest、resolved config、指标、人工 verdict 与终止标记；
4. 必要时使用 `git log --follow -- <path>` 追查文档演化。

本目录不接收新的当前计划。未来计划必须回到活跃 `docs/` 层，并由 `RESEARCH_STATUS.md` 明确授权。
