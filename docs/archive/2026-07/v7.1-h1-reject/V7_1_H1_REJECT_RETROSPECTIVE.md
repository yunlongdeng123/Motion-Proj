# V7.1 H1 Reject Retrospective

> **收口日期**：2026-07-24
> **路线状态**：OccGS occupancy-certificate/trajectory-projection 方法 claim 已 reject。
> **保留资产**：object-centric GS、WorldState、typed label/depth、evaluator 与 run contract。
> **当前入口**：[`../../../RESEARCH_STATUS.md`](../../../RESEARCH_STATUS.md)。

## 1. 裁决

V7.1 按计划完成 11A 坐标/状态底座、11B 分层 occupancy/certificate calibration、11C 同步 renderer/typed
label 和 11D matched pilot。11A–11C 的 `COMPLETE` 只表示工程 gate 通过。11D 的正式结果同时触发：

- `H1-CERT = REJECTED`：precision `0.75 < 0.80`；
- `H1-PROJ = REJECTED`：30/30 reject，0 comparable export，usable yield 0；
- H2/H3/scale 未触发。

## 2. 冻结 run

| 字段 | 值 |
|---|---|
| run ID | `v71_v7-h1-11d__pilot-3-matched__s0__20260723T155755269940Z__cf8d5ebc` |
| 路径 | `/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11D/v71_v7-h1-11d__pilot-3-matched__s0__20260723T155755269940Z__cf8d5ebc/` |
| code | `304407b94350ddfd17a9d4f29e43b7d1b789a326` |
| config SHA256 | `cf8d5ebc1429e076fc5142aa6a759a18f54b7f3f937c8423d51505a094bc9fe3` |
| proposal SHA256 | `f8986915f8d2be0cddddfa6be86f4d2d1ece456c12bf9a962cafec78fd058cd7` |
| terminal | 唯一 `REJECTED` |

## 3. 数值

- 3 scenes × 2 actors × P1–P5 = 30 proposals，S1 未删除；
- C/D1 realized trajectory hash 30/30 相同；
- D1：TP=15、FP=5、FN=2（含 abstention）、FAIL=20、UNKNOWN=10、PASS=0；
- precision=0.75、recall=0.8824、abstention=0.3333、PASS coverage=0；
- C external hard violation=17/30；
- D2 accept/export=0/30，usable yield=0，external rate=undefined；
- scenario effect：0 positive、25 negative、5 source-positive/non-event、0 same-actor pair。

## 4. 失败机制

### 事件先验失败

P1–P5 是通用横移，不是由 lane topology、target-lane interaction 和 corridor crossing 构造。结果中没有任何
positive 或 same-actor pair，因此即便 certificate 完美，也没有 H3 所需的比较对象。

### 证据覆盖与精度同时失败

三场景 base UNKNOWN 约为 96–98%。D1 既有 10/30 abstention，又有 5 FP。5 FP 集中于 004 actor 8：
coarse certificate 报 5 static-overlap voxels，raw LiDAR 为 0 points。这提示 representation mismatch，
但 LiDAR 点缺失不等于自由空间真值，不能后验改标签。

### repair 没有可比较输出

D2 全拒绝，所以无法测量修复后的外部违例率。0 export 是吞吐失败，不是安全收益。H2 render audit 与 blind
pack 没有实例化，避免了在前置方法 gate 失败后继续消耗算力。

## 5. 唯一修复

首版 aggregate 将 rejection 计为零违例。计划允许的唯一修复把它登记为 `metric_aggregation_bug`，保留旧
aggregate，并让 denominator=0 时 fail closed。修复提交 `b82c540` 未改变任何方法输出或终局裁决。

## 6. 为什么不能继续调参

降低 coverage、把 UNKNOWN 并入 PASS、删除 S1/005/004 actor 8、换 actor/方向、改 P1–P5 或只报告
recall，都会使用 11D 结果反向改变冻结设计。它们不能提供独立复现，只会把 reject 变成后验选择。

## 7. 下一路线要求

下一路线必须先完成：

1. 官方 vector map 与 scene→map provenance；
2. natural/event-first pool，含冻结 positive/negative 和 same-actor pair；
3. motion-compensated raw sweeps + map 的独立 evaluator；
4. lane-reachable proposal；
5. 只有上述通过后才复用 GS renderer 和执行下游效用实验。

具体设计见活跃文档
[`../../../POST_OCCGS_RESEARCH_DIRECTIONS.md`](../../../POST_OCCGS_RESEARCH_DIRECTIONS.md)。
