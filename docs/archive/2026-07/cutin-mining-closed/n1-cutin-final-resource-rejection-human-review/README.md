# N1 cut-in final：资源拒绝人工复核包

> **包类型**：终止证据复核，不是 prospective candidate 标注包。
> **待复核结论**：`REJECTED / stop_nuscenes_cutin_mining`。
> **N2 授权**：`false`。

本包让人工确认 final plan 是否依照冻结资源合同正确停止。因为 parent formal 在任何 evaluation scene 前就失败，
没有可供标注的 cut-in candidate；不要在本包中填写 `TRUE_POSITIVE` / `FALSE_POSITIVE`。

## 内容

- [`HUMAN_REVIEW_PROMPT.md`](HUMAN_REVIEW_PROMPT.md)：完整的复核目的、文件路径、步骤与判断边界；
- [`review_template.jsonl`](review_template.jsonl)：单条空白复核记录模板；
- [`review_working.jsonl`](review_working.jsonl)：供指定评审复制/填写的工作副本；
- 上游不可变 run：
  - parent formal：`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-01/v71_n1-event-cutin-final-01__receiver-cutin-final-v1__s0__20260726T121042935837Z__a6b12de0`；
  - 独立 rejection：`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-RESOURCE-AUDIT-01/v71_n1-event-cutin-final-resource-audit-01__resource-contract-reject-v1__s0__20260726T121624740059Z__025850f8`。

此复核不会触发 N2，也不会自动复开 cut-in mining。若评审请求重跑，必须由用户另行授权新的资源方案与新研究任务。
