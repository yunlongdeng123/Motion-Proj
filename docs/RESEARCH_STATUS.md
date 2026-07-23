# Motion-Proj 当前研究状态

> **文档职责**：唯一当前状态、研究边界与下一阶段入口。
> **最后更新**：2026-07-24
> **当前阶段**：`POST_V7.1 / N0_ASSET_PASSED / N1_MINI_EVENT_POOL_REJECTED`
> **当前决策**：`stop_n2_n5_and_reject_mini_event_pool`
> **当前路线**：[`POST_OCCGS_RESEARCH_DIRECTIONS.md`](POST_OCCGS_RESEARCH_DIRECTIONS.md)
> **当前任务**：N0 已通过、N1 已按冻结 gate reject；更新事实源与归档，不启动 N2/N3/render/training。
> **执行授权**：用户授权持续 Auto Research，直至 research reject、必须人工审核、缺少外部授权或硬阻塞。
> **授权边界**：本轮已到 research reject。后续 trainval/Waymo/nuPlan 数据、人工 candidate audit、push、
> 双卡或大型权重均需新的明确授权。

正式数值以 [`EXPERIMENTS.md`](EXPERIMENTS.md) 和实际 run 产物为准；为什么不能重复旧尝试见
[`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)；V7.1 完整计划、收口快照和编辑备份见
[`archive/2026-07/v7.1-h1-reject/`](archive/2026-07/v7.1-h1-reject/)。

## 1. 当前结论

event-first 路线已得到第二个、层级更上游的正式负结论：

- 用户提供的 `nuScenes-map-expansion-v1.3.zip` 已通过压缩包完整性、文件 SHA、四图 version/layer、
  scene→map 和 raw→processed pose 合同；`N0-ASSET-01` 为唯一 `COMPLETE`；
- N1 在查看事件结果前冻结 actor eligibility、map matching、稳定 source/target、merge/lane-change topology、
  target-lane front/rear relation 和样本量 gate；
- 003/005/004 共 45 个 eligible actors、3,915 个 eligible poses，其中 3,678 个 map matched；
- 71 个稳定 token transition 中，39 个是 route continuation、19 个 merge、3 个 lane change、10 个
  unresolved；22 个通过 topology；
- 22/22 topology-pass transition 都未同时找到冻结范围内、精确 target-token 上的 front 和 rear actor；
- 因此 positive=0、negative pairing=0、same-actor pair=0、positive scenes=0，`N1-EVENT-01` 唯一终态
  为 `REJECTED / reject_mini_event_pool`。

准确边界是：**这三个 mini scenes 在冻结的 topology + exact-target-token interaction 定义下不能提供可比较
事件池**。它不证明三场景绝对没有人类可识别交互，也不证明 full nuScenes、nuPlan 或 Waymo 没有事件。
N1 的 exact-token relation 可能低估跨相邻 longitudinal token 的车辆关系，这一未知不得通过本 run 后验放宽。

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

## 4. N0/N1 冻结事实

| 项目 | 冻结结果 |
|---|---|
| N0 run | `v71_n0-asset-01__map-v1-3__s0__20260723T232427413355Z__e250cccd` |
| N0 code/config | commit `fcb5a73`；config `e250cccdf415561e617600ae0e93b3e1f2b190aefd4d960f72023301d5b15696` |
| archive SHA | `9dbc80a095b6b28d9b79fc9a43471a750dc92ca78c6d0db288fd92b34be5a144` |
| N0 hashes | asset `48e8ace8…c60a7`；scene-map registry `7c83e936…560c1` |
| N0 pose contract | 121 keyframes；max translation `0 m`；max rotation `1.01065e-7 rad` |
| N1 run | `v71_n1-event-01__mini-event-v1__s0__20260723T232920917536Z__cd56b326` |
| N1 code/config/data | commit `82117c7`；config `cd56b326…`；data `919b0859…` |
| N1 eligibility | 003=7、005=22、004=16，共 45 actors |
| N1 map match | 003 `464/522`；005 `1976/2067`；004 `1238/1326` |
| N1 transitions | route continuation=39、merge=19、lane change=3、unresolved=10 |
| N1 interaction | topology pass=22；interaction PASS=0/22 |
| N1 gate | positive=0、negative=0、pair=0、positive scenes=0；`REJECTED` |

完整报告见 [`N1_MINI_EVENT_POOL_REPORT.md`](N1_MINI_EVENT_POOL_REPORT.md)。

## 5. 下一路线与闸门

首选路线是“event-first map-and-raw-evidence counterfactual pipeline”，不是重命名后的 OccGS H1：

| Gate | 目的 | 通过条件 | 失败动作 |
|---|---|---|---|
| `N0-ASSET` | 建立可审计地图/数据底座 | 官方 vector map 可加载；scene→map 映射与 hash 完整 | **PASSED** |
| `N1-EVENT` | 证明比较对象存在 | ≥2 positive、≥2 negative、≥2 same-actor pairs、≥2 positive scenes | **REJECTED** |
| `N2-EVIDENCE` | 建立独立合法性参照 | N1 先通过 | **not triggered** |
| `N3-PROPOSAL` | 生成 lane-reachable 候选 | N1/N2 先通过 | **not triggered** |
| `N4-RENDER` | 复用 GS 生成同步可视产物 | N1–N3 先通过 | **not triggered** |
| `N5-UTILITY` | 检验下游收益 | scene-disjoint、matched budget、≥3 seeds | **not triggered** |

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
- 复开已拒绝的 mini N1 时降低 interaction gap、把单侧邻车算 positive、跨 token 后验拼接或删 scene；
- 在新的 event pool 通过前启动 N2/N3/render/training。

## 7. 当前任务队列

1. 完成 N0/N1 结果登记、失败分解和 `event-first-mini-reject` 归档；
2. 停止 mini 上的 N2–N5；不得用阈值修改翻案；
3. 若用户另行授权，优先请求 nuScenes `v1.0-trainval` annotations/metadata，先做 scene-disjoint
   corridor-relation calibration，再冻结大池评估；这比直接下载传感器全量或大模型更小、更同域；
4. 22 个 topology-pass candidate 可作为 calibration/audit 对象，但不能回流为本 run 的 positive。
   若采用人工审计，必须另交完整盲评提示词且 verdict 只能由用户/指定评审填写；
5. full nuScenes 仍不足时，才独立评估 nuPlan/Waymo/ScenarioNet 的许可、容量和最小 shard。

## 8. 事实源优先级

发生冲突时按以下顺序处理：

1. 实际 run 产物、resolved config、原始指标、checkpoint 与 terminal marker；
2. [`EXPERIMENTS.md`](EXPERIMENTS.md)；
3. 本文件；
4. [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)；
5. 当前预注册路线；
6. `docs/archive/` 中的历史计划、报告和提示词。
