# N1 receiver-centric cut-in 第四轮 K4 calibration fixture

本 fixture 从以下不可变 parent run 提取，仅用于 final strict v2 的校准与回归：

```text
/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/
v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T173015103731Z__5b1634e3
```

- 已完成的人审文件 SHA256：`983e4b7a4160ff7aec127343b5ca3e1e9a1f07f06d799f4db9695fa241851321`
- parent event-pool canonical SHA256：`850434a349c65e2f8fc9ece98357e3a0a2f94afcd55d544e7648b47e44affe7f`
- parent immutable audit-set SHA256：`5379059a2554b808000eee1b88f416a0e2dfee87d531d26e2b8f645bf9c3da30`
- 人工标签：18 条、16 个 scene；逐条原始 evidence hash 见 `audit_manifest_minimal.json`。

本目录故意不复制 panel PNG、topdown PNG 或相机资产；回归只需要人工已填写的
`review_working.jsonl`、每条 JSON evidence 和冻结的 strict-v2 期望。它们不能进入 final
evaluation metric，也不能被用来声明 prospective precision。

`K4-010` 与 `K4-011` 是 release-blocking TP；`K4-009` 保留其人工 `TRUE_POSITIVE` 标签但
不是 blocking 样本。其余 15 条人审 FP 在 final strict verifier 中均不得 PASS。
