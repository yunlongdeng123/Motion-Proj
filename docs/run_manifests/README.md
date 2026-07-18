# Run 轻量归档

本目录从 `/root/autodl-tmp/runs/` 摘取部分早期正式 run 的 `resolved.yaml`、`manifest.json`、
`summary.json`、`machine_summary.json` 与人工评审说明，主要用于换机后核对 commit 和 fingerprint。

这里不是完整实验证据，也不是当前实验排程：`metrics.jsonl`、panel、checkpoint 和后续 preference run
仍以正式 run 目录及 [`../EXPERIMENTS.md`](../EXPERIMENTS.md) 为准。

| 目录 | 已确认范围 | 后续状态 |
|---|---|---|
| `autoresearch-c0-conditioning-s20260714-v2` | 官方 SVD conditioning parity 通过 | 可复用生成协议，不代表训练收益 |
| `autoresearch-p0-projector-s20260714-v1` | P-UNC point-space machine gate 通过 | 后续 PA0 人审已完成；不等于 RGB target 或 preference 合法 |
| `autoresearch-p1-target-s20260714-v2` | RGB/VAE target legality machine fail | explicit target 路线保持拒绝 |
| `autoresearch-e0-evaluator-s20260714-v2` | evaluator scope 实现缺陷 | 仅保留为历史证据 |
| `autoresearch-e0-evaluator-s20260714-v3` | CoTracker3 重跑与扰动排序 machine pass | 后续 PA0 人审已完成；不标定绝对物理量 |
| `p2-v2-gen04-track8-s20260713-59c3f05-net1` | 无 future-GT generated point-track gate 通过 | 只支持 object-track 基础设施 |
| `p2-v2-replay05-review20-s20260713-8d750f3` | object-only replay 20-case review 通过 | 后续 endpoint capacity/locality gate 失败 |

2026-07-15 之后的 Physics-DPO、UPO 与 candidate fallback 没有复制到本轻量目录；其正式证据位于：

```text
/root/autodl-tmp/runs/autoresearch-pa2-pair-expanded-s20260715-v1
/root/autodl-tmp/runs/autoresearch-pa2-upo-s20260716-v2
/root/autodl-tmp/runs/autoresearch-pa2-cand-fallback-s20260716-v1
```

当前状态见 [`../RESEARCH_STATUS.md`](../RESEARCH_STATUS.md)，跨路线的 research 负结论见
[`../RESEARCH_FAILURES.md`](../RESEARCH_FAILURES.md)。
