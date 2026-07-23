# Event-first N0/N1 mini reject 归档

> **归档日期**：2026-07-24
> **权威性**：已执行研究的历史索引，不是后续运行授权。
> **当前状态**：[`../../../RESEARCH_STATUS.md`](../../../RESEARCH_STATUS.md)。

## 1. 决策链

1. V7.1 H1 因 0 positive、certificate precision fail 与 D2 0 export 被拒；
2. Post-OccGS 路线预注册先补官方 vector map，再证明 natural interaction event 存在；
3. 用户提供 `nuScenes-map-expansion-v1.3.zip`；
4. `N0-ASSET-01` 通过资产、地图、scene→map 与 pose contract；
5. `N1-EVENT-01` 在冻结定义下得到 0 positive / 0 same-actor pair；
6. 按 stop rule 以 `reject_mini_event_pool` 结束，N2–N5 未触发。

## 2. 不可变 run

### N0

`/root/autodl-tmp/runs/event_first/N0-ASSET-01/v71_n0-asset-01__map-v1-3__s0__20260723T232427413355Z__e250cccd/`

- terminal：`COMPLETE`
- code：`fcb5a7392c0169934d5388b9efd12a637b236ff9`
- config：`e250cccdf415561e617600ae0e93b3e1f2b190aefd4d960f72023301d5b15696`
- data：`b03d6cbf6093869c92659626c7a1add182157a71ff9f89cf722e24fcef6ac56b`
- asset manifest：`48e8ace83e286b79b31d1adbfedc33bcacf2c43b3f19587dd9c2fc6fbd0c60a7`
- scene-map registry：`7c83e936e150ab5ed3ab21c57a55a3ee0143d64264fa0da17cdbd77e9d3560c1`

### N1

`/root/autodl-tmp/runs/event_first/N1-EVENT-01/v71_n1-event-01__mini-event-v1__s0__20260723T232920917536Z__cd56b326/`

- terminal：`REJECTED`
- code：`82117c7ec58db9bbe7e26d0f866442b620b617f6`
- config：`cd56b326cd38ecda3ab6dd36bb31a38ce03f14aacbef7e199ff8f073558f5cf3`
- data：`919b08593a5fdf13e668714865cc2f1d2129f5bf221a3d7c3ad54af80ccbc0a3`
- event pool：`6f39cc8b917c277adfc9a8b17130c4d5d1e762beb862cbaf51c92b61727dc792`

## 3. 文档快照入口

- 正式报告：[`../../../N1_MINI_EVENT_POOL_REPORT.md`](../../../N1_MINI_EVENT_POOL_REPORT.md)
- N0 资产记录：[`../../../N0_ASSET_AND_EVENT_PREFLIGHT.md`](../../../N0_ASSET_AND_EVENT_PREFLIGHT.md)
- 执行路线与后续方向：[`../../../POST_OCCGS_RESEARCH_DIRECTIONS.md`](../../../POST_OCCGS_RESEARCH_DIRECTIONS.md)
- 失败账本：[`../../../RESEARCH_FAILURES.md`](../../../RESEARCH_FAILURES.md)
- 数值事实源：[`../../../EXPERIMENTS.md`](../../../EXPERIMENTS.md)

冻结配置的逐字快照已经保存在各 run 的 `resolved.yaml`，不在 docs 中复制第二份，以免产生配置漂移。

## 4. 防误用

- 不能通过删除 rear requirement、扩大 60 m、删 scene 或后验拼接 token 复开 N1；
- 22 个 topology-pass cases 只能作为未来 corridor evaluator 的 calibration/audit pool；
- calibration verdict 不得回填本次 `REJECTED`；
- 后续 full nuScenes/nuPlan/Waymo 数据和人工审计均需要新的明确授权。
