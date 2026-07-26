# N1 receiver-centric 第四次人工审核交付归档

> **归档日期**：2026-07-26
> **权威性**：第四版 formal 与交付时点的历史索引，不是未来执行授权。
> **当前状态入口**：[`../../../RESEARCH_STATUS.md`](../../../RESEARCH_STATUS.md)。

## 1. 成功 parent run

- task：`N1-EVENT-CUTIN-01`；
- run：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T173015103731Z__5b1634e3/`；
- commit：`f13eb0f1e39b608de1c5e698cd678c2dfd8365a4`，`code_dirty=false`；
- config：
  `5b1634e3347c81ca8d6c7a1b6b3d5a737b092732a3ff9b5c79b093fccfd846c5`；
- data：
  `79651f9111e72a5510f5f5444a202cc0d20215ac3319a7f224fd0073202ad7e9`；
- event-pool：
  `850434a349c65e2f8fc9ece98357e3a0a2f94afcd55d544e7648b47e44affe7f`；
- formal：685 scenes；18 positives / 16 scenes / 6 negatives / 6 pairs；
- terminal：唯一 `AWAITING_HUMAN_REVIEW`；
- N2：`n2_authorized=false`。

## 2. 第四次审核包

审核根：

`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T173015103731Z__5b1634e3/audit/`

- population/items：18/18；
- immutable files：58；
- immutable-set SHA256：
  `5379059a2554b808000eee1b88f416a0e2dfee87d531d26e2b8f645bf9c3da30`；
- blank review SHA256：
  `15c4fc52489783383e843788c15e834125b957b232f1536f6269c3e5ba7198ae`；
- prompt SHA256：
  `b03267e8b63fa05a7c61c79ba55e0dcdcc518a97efebdba30e85ae92b54391bb`；
- verdict 状态：18 行全部空白，Codex 未代填。

活跃层快照：

- [`../../../N1_RECEIVER_CUTIN_EVENT_POOL_REPORT.md`](../../../N1_RECEIVER_CUTIN_EVENT_POOL_REPORT.md)；
- [`../../../N1_RECEIVER_CUTIN_HUMAN_REVIEW_PROMPT.md`](../../../N1_RECEIVER_CUTIN_HUMAN_REVIEW_PROMPT.md)；
- [`../../../N1_RECEIVER_CUTIN_PREREGISTRATION.md`](../../../N1_RECEIVER_CUTIN_PREREGISTRATION.md)。

这些链接在交付时指向逐字节同版材料；后续若文档演化，以 run 内 artifact 和 hash 为最终事实。

## 3. 保留的工程失败

### 配置契约缺项

- run：
  `.../v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T170948229629Z__46186120/`；
- commit：`f5c9bbe4c819abce42e1cca0b8800e16a77af680`；
- failure：`kinematics_control_config_contract_missing`；
- 研究输出：未写入；
- 终态：`FAILED`，旧 `RUNNING` 保留为 `RUNNING.invalidated`。

### cgroup page-cache SIGKILL

- run：
  `.../v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T171746938858Z__5b1634e3/`；
- commit：`8581d4dcd1bf9a4f92b426c601e1149c804afc5a`；
- failure：`external_sigkill_under_cgroup_memory_pressure`；
- 进度：96/685 scenes，研究输出未写入；
- 终态：`FAILED`，旧 `RUNNING` 保留为 `RUNNING.invalidated`。

两个目录均不能删除、续跑、覆盖或计入 research reject。完整根因和防重复规则见
[`../../../RESEARCH_FAILURES.md`](../../../RESEARCH_FAILURES.md) 的 `N1-F16`、`N1-F17`。

## 4. 交付时验真

- manifest 四个顶层 artifact hashes 与 artifact-set 复算一致；
- event-pool canonical hash、8,416 transition record hashes、6 negative hashes 复算一致；
- 58 个 audit immutable hashes 与 set hash 复算一致；
- calibration/evaluation scene 交集为 0；
- 18 个 panels 为 `1600×780`，18 个 topdown 均可解码；
- blank validator 在 `K4-001` 缺 component verdict 处按预期 fail closed；
- parent 只有 `AWAITING_HUMAN_REVIEW`，没有 `RUNNING/FAILED/REJECTED/COMPLETE` 冲突 marker；
- `n2_authorized=false`。

## 5. 历史边界

该快照只说明第四版 machine support 与审核材料合同完成。第四次人工真实性尚未判定，不得把
`machine_gate_passed=true` 或本归档本身写成 N1 human pass；也不得据此启动 N2。
