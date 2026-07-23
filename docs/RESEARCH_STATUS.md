# Motion-Proj 当前研究状态

> **文档职责**：唯一当前状态、研究边界与下一阶段入口。
> **最后更新**：2026-07-24
> **当前阶段**：`POST_V7.1 / OCCGS_H1_REJECTED / N0_EXTERNAL_ASSET_GATE`
> **当前决策**：`archive_occgs_method_claim_and_pursue_event_first_route`
> **当前路线**：[`POST_OCCGS_RESEARCH_DIRECTIONS.md`](POST_OCCGS_RESEARCH_DIRECTIONS.md)
> **当前任务**：文档归档与只读 N0 预检已完成；等待最小 nuScenes map-expansion 下载授权，尚无新实验 run。
> **执行授权**：用户授权持续 Auto Research，直至 research reject、必须人工审核、缺少外部授权或硬阻塞。
> **授权边界**：不包含 push、双卡、全量数据或大型权重下载。当前文档调研不自动授权任何数据下载。

正式数值以 [`EXPERIMENTS.md`](EXPERIMENTS.md) 和实际 run 产物为准；为什么不能重复旧尝试见
[`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)；V7.1 完整计划、收口快照和编辑备份见
[`archive/2026-07/v7.1-h1-reject/`](archive/2026-07/v7.1-h1-reject/)。

## 1. 当前结论

V7/V7.1 已完成 object-centric GS、统一 `WorldState`、同步 typed label、外部 evaluator 和 fail-closed
run contract 的工程闭环。它没有证明 occupancy certificate/trajectory projection 的方法主张：

- 冻结的 30-proposal bank 中没有任何 `0→1` positive，也没有 same-actor positive pair；
- D1 precision 为 `0.75 < 0.80`，10/30 abstain，PASS coverage 为 0；
- D2 拒绝 30/30，0 个 comparable export，usable yield 为 0；
- 因此 H1-CERT 与 H1-PROJ 均按预注册 `REJECTED`，H2/H3/scale 未触发。

这不是“工程没跑通”，也不是“违例率降为 0”。准确结论是：在冻结对象、proposal、阈值和证据定义下，
方法没有产生可比较样本，且 certificate 精度未达标。

## 2. H1-11D 冻结事实

| 项目 | 冻结结果 |
|---|---|
| 正式 run | `v71_v7-h1-11d__pilot-3-matched__s0__20260723T155755269940Z__cf8d5ebc` |
| run 根 | `/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11D/v71_v7-h1-11d__pilot-3-matched__s0__20260723T155755269940Z__cf8d5ebc/` |
| 代码 | `304407b94350ddfd17a9d4f29e43b7d1b789a326` |
| 配置 SHA256 | `cf8d5ebc1429e076fc5142aa6a759a18f54b7f3f937c8423d51505a094bc9fe3` |
| proposal-bank SHA256 | `f8986915f8d2be0cddddfa6be86f4d2d1ece456c12bf9a962cafec78fd058cd7` |
| 样本 | 3 scenes × 2 actors × P1–P5 = 30；source-only eligibility |
| matched 性 | C/D1 realized trajectory hash 30/30 相同 |
| D1 | TP=15，FP=5，FN=2（含 abstention），FAIL=20，UNKNOWN=10，PASS=0 |
| D1 指标 | precision `0.75`，recall `0.8824`，abstention `0.3333`，PASS coverage `0` |
| C 外部 hard violation | 17/30；003=5/10，005=7/10，004=5/10 |
| D2 | accept/export `0/30`，usable yield `0`，external violation rate 不可定义 |
| scenario effect | positive=0，negative=25，source-positive/non-event=5，same-actor pair=0 |
| 终态 | 唯一 terminal marker `REJECTED` |

唯一允许的修复是 `metric_aggregation_bug`：首版错误地把 rejection 计作零违例；修复提交
`b82c540` 保留了修复前 aggregate，并在无 export 时 fail closed。该修复没有改变任何方法输出。

## 3. 失败分层

| 层 | 观察到的事实 | 裁决 | 对下一路线的约束 |
|---|---|---|---|
| 事件存在性 | 0 个 `0→1` positive、0 个同 actor 对 | proposal bank 不支持 H3 比较 | 先挖出真实事件和可配对候选，再做编辑或渲染 |
| 独立证据 | base UNKNOWN 约 96–98%；D1 10/30 abstain | coarse voxel certificate 覆盖不足且 precision fail | 用矢量地图与运动补偿 raw sweeps 建独立几何证据 |
| 修复吞吐 | D2 30/30 reject、0 export | H1-PROJ reject | 不得把拒绝当成零违规；必须先过 usable-yield gate |
| 下游效用 | 无 positive pair，H1 已拒绝 | H3 not triggered | 不得训练 detector/event task 或声称数据增益 |
| 渲染/补全 | H1 前置门禁失败 | H2 not triggered | GS 仅作为已验证 renderer 基础设施，不承担安全证明 |

完整的“观察—推断—未知—复开条件”见失败账本。

## 4. 本机预检与当前硬阻塞

2026-07-24 只读审计：

- nuScenes 根：`/root/autodl-tmp/data/nuscenes`；
- `maps/` 只有 4 个 raster PNG，没有 map-expansion vector JSON；
- `/root/autodl-tmp/data` 下没有 Waymo 或 nuPlan 数据；
- DriveStudio 只含 nuPlan/Waymo adapter 代码，不等于数据已存在；
- `/root/autodl-tmp` 约 128G，总已用 64G，可用约 65G。

进一步预检确认：processed `003/004` 对应 `boston-seaport`，`005` 对应 `singapore-queenstown`；冻结的
6 actors 虽然 track 均连续，但 003:38、003:35、005:23 的完整 track 位移均不足 1 m。详见
[`N0_ASSET_AND_EVENT_PREFLIGHT.md`](N0_ASSET_AND_EVENT_PREFLIGHT.md)。

因此目前可以继续做代码/标注审计和 mini 上的事件存在性 smoke；要运行基于 lane graph 的正式预检，至少需要
nuScenes 官方 map expansion 资产。Waymo/nuPlan/大权重属于新的下载与许可边界，不能静默启动。

## 5. 下一路线与闸门

首选路线是“event-first map-and-raw-evidence counterfactual pipeline”，不是重命名后的 OccGS H1：

| Gate | 目的 | 通过条件 | 失败动作 |
|---|---|---|---|
| `N0-ASSET` | 建立可审计地图/数据底座 | 官方 vector map 可加载；scene→map 映射与 hash 完整 | 缺授权则暂停并交付下载清单 |
| `N1-EVENT` | 证明比较对象存在 | 冻结事件定义后，独立池内有足够正例及 same-actor `0→1/0→0` 对 | 事件池不足则 reject mini 路线，不渲染 |
| `N2-EVIDENCE` | 建立独立合法性参照 | motion-compensated raw sweeps + map；报告 precision/recall/coverage/abstention | 覆盖或精度失败则停止，不调阈值救结果 |
| `N3-PROPOSAL` | 生成 lane-reachable 候选 | 固定预算下有正效应、可判定且不靠后验挑 actor | 失败则 reject proposal family |
| `N4-RENDER` | 复用 GS 生成同步可视产物 | 仅对通过 N1–N3 的冻结候选导出 | renderer 不用于证明合法性 |
| `N5-UTILITY` | 检验下游收益 | scene-disjoint、matched budget、≥3 seeds、任务指标 | 无效则 reject data-utility claim |

详细预注册建议、文献依据、替代路线和禁止项见
[`POST_OCCGS_RESEARCH_DIRECTIONS.md`](POST_OCCGS_RESEARCH_DIRECTIONS.md)。

## 6. 可复用与冻结边界

可以复用：

- `WorldState`、坐标合同、typed depth/label、run contract、artifact index；
- object-centric GS reconstruction 与 renderer；
- D1/D2 evaluator 的接口、三态 `PASS/FAIL/UNKNOWN` 和 fail-closed aggregation；
- 冻结 proposal-bank 作为负对照与回归 fixture。

不得复开：

- P1–P5 固定横移 proposal family 的 H1 claim；
- 通过降低 known-fraction、删 S1、删 004 actor 8、换 actor/方向或把 UNKNOWN 并入 PASS 来翻案；
- 把 0 export 写成 0 violation；
- 用 GS、学习 occupancy 或同一方法生成的标签充当独立安全真值；
- 在 N1 之前启动 H2/H3/scale。

## 7. 当前任务队列

1. 已完成文档归档、失败账本和下一路线预注册；
2. 已完成不下载新数据的 scene→map、selected-actor continuity 与最小资产预检；
3. 需要用户批准 nuScenes 官方 map expansion 资产后运行 N0/N1；
4. 未授权时停在外部资产 gate，不从 raster 猜 lane，也不静默下载其他数据。

## 8. 事实源优先级

发生冲突时按以下顺序处理：

1. 实际 run 产物、resolved config、原始指标、checkpoint 与 terminal marker；
2. [`EXPERIMENTS.md`](EXPERIMENTS.md)；
3. 本文件；
4. [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)；
5. 当前预注册路线；
6. `docs/archive/` 中的历史计划、报告和提示词。
