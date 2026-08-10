# WorldSim V3.1 归档

- 归档日期：2026-08-10
- 分支：`research/worldsim-v3`
- 收口提交：`d91e80e`（`docs(worldsim): close R0 integration`）
- 终态：`none_plan_complete`
- 最新任务：`WS-V3-R0-INTEGRATION-01`（`done`）

本目录冻结 V3/V3.1 的权威计划和三份研究事实源快照，只用于解释已经发生的实验、裁决与交付。归档中的
“下一步”、命令或未启动分支不构成当前执行授权。

## 归档内容

- [`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_1.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_1.md)：V3.1 已执行权威计划；
- [`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3.md)：被 V3.1 取代的初始 V3 计划；
- [`RESEARCH_STATUS_V3_1_SNAPSHOT.md`](RESEARCH_STATUS_V3_1_SNAPSHOT.md)：R0 收口时的状态快照；
- [`EXPERIMENTS_V3_1_SNAPSHOT.md`](EXPERIMENTS_V3_1_SNAPSHOT.md)：R0 收口时的实验台账快照；
- [`RESEARCH_FAILURES_V3_1_SNAPSHOT.md`](RESEARCH_FAILURES_V3_1_SNAPSHOT.md)：补记 V3.1 终局失败项之前的账本快照。

三份事实源快照保持收口时的原始字节，不重写其中的相对链接；快照内以 `archive/...` 或计划文件名开头的链接应按
原 `docs/` 根目录解释。文件完整性见 [`SHA256SUMS`](SHA256SUMS)。

## Canonical 收口证据

R0 canonical run：

```text
/root/autodl-tmp/runs/worldsim_v3/WS-V3-R0-INTEGRATION-01/
20260809T194625Z__r0-integration-s0-r1
```

- source：`64e3d15ca30de44088c2f6fbfb6da048a31a4acf`；
- terminal：`done`；
- summary SHA-256：`3ffe99ea25302a1bfd8a73329133ae052632f8cf32d8124bc7df4d35e85f15a7`；
- manifest SHA-256：`a9b052a636de3410700bca6899c6efda88248398b2befb96cd247ac16f3e1d90`；
- terminal SHA-256：`207758b92d750cd239fa998ed7572c5f404f8747fadb0a4b74a12295983015c6`；
- 验证范围：63/63 inputs、23/23 decisions、12/12 deliverables、26/26 manifest files；
- 最终链：
  `A1-C0-off__A2-D2-boundary-priority__A3-R0-off__A4-P2-mixed__A4-P3-exact-chunk`。

## 路径保留说明

仓库根 `docs/` 下的 V3/V3.1 原计划文件继续保留为 hash-stable compatibility copies，因为冻结的 R0 protocol
和历史链接仍引用原路径。它们不再是当前计划；本目录是 V3.1 的归档导航入口。不要移动、删改或从原计划中的
未启动任务恢复执行。

## 终局边界

- A1=`C0-off / done_off`，不表示 C0 在所有场景、所有指标都最好；
- A2 D2 是 boundary-priority tradeoff，D1 保留 fallback，不表示 D2 dominance；
- A3 R1 被资源门与 diagnostic tradeoff 拒绝，生产链使用 R0/D2 exact alias；
- A4 P1 被质量门拒绝；P2 只支持 mixed storage，P3 只支持 exact asset separation；
- F0 没有在本机执行 Instant NuRec inference；
- R0 证明证据链闭环，不证明完整 world model、跨场景泛化或闭环安全。

后续若创建新路线，先读取当前 [`../../../RESEARCH_STATUS.md`](../../../RESEARCH_STATUS.md) 和
[`../../../RESEARCH_FAILURES.md`](../../../RESEARCH_FAILURES.md)，使用新任务 ID、新协议和新 run，不续写本归档终态。
