# 动态重建路线 V1 人工审核包

- 生成日期：2026-07-26
- 审核对象：旧路线封存、清理边界、下一阶段研究方案
- 当前实验状态：只完成文档与轻量资产审计；没有安装、下载、预处理、训练或推理
- 权威计划：[`../../DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md`](../../DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md)
- 人工结论：必须由用户或指定评审者填写，Codex 不代填

## 建议审核顺序

1. 读 [`../../RESEARCH_STATUS.md`](../../RESEARCH_STATUS.md)，确认当前裁决与下一门禁；
2. 读 [`../../RESEARCH_FAILURES.md`](../../RESEARCH_FAILURES.md) 的 `PIVOT-F01` 至 `PIVOT-F05`；
3. 读 [`../../archive/2026-07/cutin-mining-closed/CLEANUP_MANIFEST.md`](../../archive/2026-07/cutin-mining-closed/CLEANUP_MANIFEST.md)，确认删除边界；
4. 读 [`../../DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md`](../../DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md)；
5. 对照 [`EVIDENCE_MANIFEST.md`](EVIDENCE_MANIFEST.md) 与 [`REVIEW_CHECKLIST.md`](REVIEW_CHECKLIST.md)；
6. 将结论填写到 [`HUMAN_VERDICT_TEMPLATE.md`](HUMAN_VERDICT_TEMPLATE.md) 的副本中，不覆盖空白模板。

## 本轮需要人工决定的五件事

1. 是否同意 cut-in 路线维持 `rejected / frozen`，仅保留以后可选演示用途；
2. 是否认可已删除内容全部为可再生中间产物，保护项足够；
3. 是否批准“AD-GS exact → DGGT inference → 压力测试 → 再选创新”的顺序；
4. 是否认可 VAD-GS/Perception-aware/Real2Sim 等 2026 工作已经收紧 novelty 边界；
5. 是否按计划开放下一轮最低资源：≥32 GB RAM（推荐 64 GB）和 1×24 GB GPU。

## 审核结果含义

- `APPROVE`：允许在资源开放后启动 `DR-M2-ENV-ASSET-01`；
- `APPROVE_WITH_CHANGES`：先修改计划，M2 仍保持 `pending`；
- `REJECT`：当前方案不启动，写明替代方向；
- `UNCERTAIN`：列出需要补充的证据，M2 仍保持 `pending`。

无论结论如何，都不能通过本审核包恢复 cut-in threshold tuning，也不能越过 AD-GS exact reproduction 门禁。
