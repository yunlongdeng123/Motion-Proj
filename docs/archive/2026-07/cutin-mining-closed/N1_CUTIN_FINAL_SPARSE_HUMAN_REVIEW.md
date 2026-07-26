# N1 final 稀疏终局人工审核入口

> **用途**：`sparse_terminal_diagnostic_only`
> **parent 终态**：`REJECTED / stop_nuscenes_cutin_mining_too_sparse`
> **人工条目**：1 primary + 3 diagnostic
> **N2 授权**：`false`

完整包：

```text
/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-SPARSE-AUDIT-01/
v71_n1-event-cutin-final-sparse-audit-01__human-review-v1__s0__20260726T145456566329Z__d3ceeef5
```

审核顺序：

1. 阅读 `audit/HUMAN_REVIEW_PROMPT.md`；
2. 只使用 `audit/index.html` 盲审；人工完成前不要读取 `audit/debug_index.html`；
3. 只编辑 `audit/review_working.jsonl` 的四个 component、overall、failure codes、reviewer 和 notes；
4. 不改 audit ID、tier、hash、顺序或证据引用；
5. 完成后运行：

```bash
cd /root/autodl-tmp/motion_proj_final_clean_beee1de
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motionproj
PYTHONPATH=. python scripts/validate_n1_cutin_review.py \
  --run-dir /root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-SPARSE-AUDIT-01/v71_n1-event-cutin-final-sparse-audit-01__human-review-v1__s0__20260726T145456566329Z__d3ceeef5 \
  --review-file /root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-SPARSE-AUDIT-01/v71_n1-event-cutin-final-sparse-audit-01__human-review-v1__s0__20260726T145456566329Z__d3ceeef5/audit/review_working.jsonl \
  --output /root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-SPARSE-AUDIT-01/v71_n1-event-cutin-final-sparse-audit-01__human-review-v1__s0__20260726T145456566329Z__d3ceeef5/human_review_validation.json
```

immutable set SHA256：
`949aed9405721643613a72f9947cbea1a47e94caec4f8f14bc5e1d491b41ec7a`。

人工结果不能改变 parent 的数量门失败。即使唯一 primary 为 TP，也只有 1 TP / 1 scene，低于冻结的 sparse
最低 3 TP / 3 scenes；不得据此恢复挖掘器、生成正式 seed pool、调规则或启动 N2。
