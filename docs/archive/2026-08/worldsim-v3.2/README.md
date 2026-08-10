# WorldSim V3.2 归档

- 归档日期：2026-08-11
- 分支：`research/worldsim-v3.2-semantic-repair`
- 研究收口提交：`f44deca`（`research(worldsim): complete V3.2 integration and closeout`）
- 终态：`none_plan_complete`
- 最新任务：`WS-V32-R0-INTEGRATION-01`（`done`）

本目录冻结 V3.2 的权威计划、S0 审计和三份研究事实源快照，只用于解释已经发生的实验、裁决与交付。归档中的
“当前任务”“下一步”、命令或外部门禁项不构成当前执行授权。

## 归档内容

- [`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_2.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_2.md)：V3.2 已执行计划；
- [`WS_V32_S0_SOTA_AUDIT.md`](WS_V32_S0_SOTA_AUDIT.md)：S0 source/license/weight/hardware 审计；
- [`RESEARCH_STATUS_V3_2_SNAPSHOT.md`](RESEARCH_STATUS_V3_2_SNAPSHOT.md)：归档前状态快照；
- [`EXPERIMENTS_V3_2_SNAPSHOT.md`](EXPERIMENTS_V3_2_SNAPSHOT.md)：归档前实验台账快照；
- [`RESEARCH_FAILURES_V3_2_SNAPSHOT.md`](RESEARCH_FAILURES_V3_2_SNAPSHOT.md)：补记终局复开矩阵之前的失败账本快照。

三份事实源快照保持归档前的原始字节，不重写其中的相对链接；快照内链接应按原 `docs/` 根目录解释。文件完整性
见 [`SHA256SUMS`](SHA256SUMS)。

## Canonical 收口证据

R0 canonical run：

```text
/root/autodl-tmp/runs/worldsim_v32/WS-V32-R0-INTEGRATION-01/
20260810T134658Z__r0-final-integration-s0-r1
```

- source：`f44decacca812fe1b476253b6bdc8aac1869f873`；
- terminal：`done`；
- 8/8 frozen gates passed；
- summary SHA-256：`40624cbc79a004e9e07e57b00cebc535b900297a10f0d070fb4e9305a5f7937a`；
- manifest SHA-256：`358d9fc7fde6a535c2ffb0bb2ff34cf1f9df3c151066f3051e24859a5d73a27e`；
- terminal SHA-256：`d31a4f8e62f31dbbf6bbf2520243f5061c68e6682ea5011ef8c64a8dbb541617`；
- report SHA-256：`b5397f555270a901013f0a6ce82ba20c8a868d9e22039ba1d9cc2066adf20913`；
- 定向回归：`36 passed`；
- 最终链：S1 extended semantic sidecars + S2 generated-background mixed scene + S3 generated-actor override +
  R0 exact chunk package。

## 终局处置

- S1 canonical r6、S2 canonical r3、S3 canonical r3 与 R0 canonical r4 为 selected evidence；
- S4 non-temporal task 已执行完，但删除语义保持门失败，仅保留 optional diagnostic，不进入生产链；
- S4 temporal 受 gated Cosmos base 权重阻塞；S5 受许可证门阻塞；二者不构成未完成的 V3.2 执行项；
- R0 只证明当前 scene 的可追踪集成、三固定视角 storage/package 等价和冻结资源合同；不证明生成内容为 GT、
  streaming/load/render 收益、跨场景泛化或闭环安全；
- 后续若外部条件变化，必须建立新 task ID、冻结新 protocol 并创建新 run，不能续写本归档 terminal。

## 路径保留说明

仓库根 `docs/` 下的 V3.2 计划和 S0 审计继续保留为 hash/link compatibility copies，因为冻结 protocol 与历史链接
引用原路径。它们不再是当前计划；不要移动、删改或从计划中的 S4 temporal、S5 等分支恢复执行。

恢复研究时先读取当前 [`../../../RESEARCH_STATUS.md`](../../../RESEARCH_STATUS.md) 和
[`../../../RESEARCH_FAILURES.md`](../../../RESEARCH_FAILURES.md)。当前 `next_action=none_plan_complete`。
