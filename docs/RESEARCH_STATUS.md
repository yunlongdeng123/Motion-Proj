# Motion-Proj 当前研究状态

> **文档职责**：唯一当前状态、研究边界与下一阶段入口。
> **最后更新**：2026-07-25
> **当前阶段**：`POST_V7.1 / N1_THIRD_HUMAN_REJECTED / N1_RECEIVER_CUTIN_PREREGISTERED`
> **当前决策**：`continue_fourth_n1_receiver_cutin_no_n2`
> **当前路线**：[`POST_OCCGS_RESEARCH_DIRECTIONS.md`](POST_OCCGS_RESEARCH_DIRECTIONS.md)
> **当前任务**：第三次人审 12/12 均为 FP，独立 adjudication 已 `REJECTED`。第四版已冻结
> receiver-centric outside→inside + 独立 RECEIVER 定义；49 条历史人审 calibration replay
> 达到第三次 FP 12/12 拒绝、第二次 FP 35/35 拒绝、旧 TP 1/2 保留。正在完成未见 train evaluation、
> 第四次完整盲审包和文档归档；N2 继续封闭。
> **执行授权**：用户授权持续 Auto Research，直至 research reject、必须人工审核、缺少外部授权或硬阻塞。
> **授权边界**：用户明确授权持续迭代 N1 挖掘算法、参考本地 cut-in 规则包并交付第四次人工 audit pack；
> 明确禁止进入 N2。传感器 sweeps 下载/抽取、N2/N3、外部数据集下载、push、双卡或大型权重均未授权。

正式数值以 [`EXPERIMENTS.md`](EXPERIMENTS.md) 和实际 run 产物为准；为什么不能重复旧尝试见
[`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)；V7.1 完整计划、收口快照和编辑备份见
[`archive/2026-07/v7.1-h1-reject/`](archive/2026-07/v7.1-h1-reject/)。

## 1. 当前结论

第二次 full-domain N1 的 machine `COMPLETE` 已被后续人工证据推翻：

- parent 在 val 146 上给出 37 machine positives；
- completed review SHA256 为
  `ae71b31e02faf1d783c36748e629e85acf32a132f35ce2f98102a5f62201dd05`；
- 用户确认 2 `TRUE_POSITIVE` / 35 `FALSE_POSITIVE` / 0 `UNCERTAIN`，audited precision 仅
  `0.054054`；
- 独立 `N1-EVENT-FULL-AUDIT-01` 在 clean commit `1e2f5ea` 上以唯一 `REJECTED` 结束；
- 主要错误不是 graph gap，而是把“target 有多个 incoming”当成当前 subject 的 merge，忽略原始 2 Hz
  车辆运动学、subject identity 与 front/rear 的跨时刻持续性；
- 旧 17-scene N2 list 已清空并 fail closed；没有下载、抽取或启动任何 N2 数据/任务。

第三版 [`N1_KINEMATIC_PREREGISTRATION.md`](N1_KINEMATIC_PREREGISTRATION.md) 的 parent formal
run 已在 clean commit `aa162ef` 上完成：

- official train 694 scenes，8,631 transitions → 1,879 topology-pass → 244 physical-motion-pass →
  12 persistent-interaction candidates；
- 12 candidates 覆盖 9 scenes，但 same-actor negative/pair 只有 2/2，低于冻结的 4/4；
- machine gate 的 candidate-count/scene checks 通过，negative/pair checks 失败；
- 第三次 review SHA256 为
  `005cd74b874833808435fd2f47387d1d8e446cdea2d3a5cae6146e34bf331e96`；
- 12/12 已审，TP=0、FP=12、UNCERTAIN=0，subject maneuver `INVALID=12`；
- 主要 failure 为 `SUBJECT_NO_LATERAL_MANEUVER=12`、`ROUTE_CONTINUATION=11`，说明第三版仍把
  地图分支收敛当成车辆行为；
- parent 唯一终态保持历史 `AWAITING_HUMAN_REVIEW`，独立 adjudication run
  `v71_n1-event-kinematic-audit-01__human-audit-reject-v1__s0__20260725T155754010881Z__4c51f0d9`
  已在 commit `1fbbbc1` 上以唯一 `REJECTED` 结束；
- `n2_authorized=false`，没有启动任何 N2。

完整结果、pair 失败分解、审核入口和下一方向见
[`N1_KINEMATIC_EVENT_POOL_REPORT.md`](N1_KINEMATIC_EVENT_POOL_REPORT.md)。

第四版 [`N1_RECEIVER_CUTIN_PREREGISTRATION.md`](N1_RECEIVER_CUTIN_PREREGISTRATION.md)
已经在 formal evaluation 前冻结：

- 事件改为 receiver-centric：subject 的 2 Hz 车辆中心必须从接收车道外进入，post oriented box 稳定在
  车道内；进入前相对 receiver corridor 必须近似同向；
- parallel lane change 排除 source token；merge 只枚举不同于 source 的 direct incoming；
- RECEIVER 必须在 pre/post 为同一最近后车，bumper gap `[0.5,40] m`；
- 30-frame negative 不缩短、不与 physical event overlap，并同样要求持续 RECEIVER；
- 第二、三次共 49 条标签和 26 scenes 只作 calibration；正式 train 排除全部已审 scene；
- 冻结 replay：第三次 12/12 FP 拒绝、第二次 35/35 FP 拒绝、第二次 TP 保留 1/2；
- 轻量 map-expansion reader、单 location cache 与流式 metadata 已解除 2 GiB cgroup 的 `RC=137`
  工程卡点，reference discretization 回归通过。

更早的 mini 第一次 N1 负结论仍冻结，不因第三版 formal 而回写：

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
| N1-FULL run | `v71_n1-event-full-01__fulldomain-v1__s0__20260724T081214528945Z__f12c886c` |
| N1-FULL code/config | commit `2bac4c9`(dirty)；config `f12c886c…`；event-pool `9635b514…` |
| N1-FULL split | calibration=mini 003/004/005；evaluation=官方 val 146（排除全部 mini scene） |
| N1-FULL calibration | 逐位复现 mini：45 eligible、71 transition、22 topology-pass、0 positive |
| N1-FULL evaluation | 1,361 eligible、1,898 transition、396 topology-pass |
| N1-FULL machine gate | positive=37、negative=7、pair=7、positive scenes=17；父 run 的机器终态保持 `COMPLETE` |
| N1-FULL human audit | 37/37 已审；TP=2、FP=35、UNCERTAIN=0；precision=`0.054054`；独立裁决 run 唯一终态 `REJECTED` |
| N1-FULL audit provenance | review SHA256 `ae71b31e…dd05`；clean adjudication commit `1e2f5ea`；run `v71_n1-event-full-audit-01__human-audit-reject-v1__s0__20260725T083929632491Z__6507cbac` |
| N1-FULL 归因 | 37 positive 中 36 依赖 corridor 挽回的跨-token 邻车；核心错误是以 target 多 incoming 代替 subject 运动学与持续交互 |
| N1-KINEMATIC run | `v71_n1-event-kinematic-01__kinematic-v1__s0__20260725T092427030639Z__8c2247b6` |
| N1-KINEMATIC code/config/data | commit `aa162ef4…e49`；config `8c2247b6…b3b1b`；data `4f914ac8…efe3`；`code_dirty=false` |
| N1-KINEMATIC split | calibration=第二次 val 人审 17 scenes；evaluation=official train 694，排除全部 10 mini；scene-disjoint |
| N1-KINEMATIC calibration | 35/35 旧 FP 拒绝；2 个旧 TP 保留 1；只用于设计，不混入 formal metric |
| N1-KINEMATIC evaluation | 8,631 transition；1,879 topology；244 physical motion；12 interaction candidates / 9 scenes |
| N1-KINEMATIC pair support | negative=2、same-actor pair=2，均低于冻结阈值 4；`machine_gate_passed=false` |
| N1-KINEMATIC audit | review 12/12：TP=0、FP=12、UNCERTAIN=0；SHA `005cd74b…31e96`；独立 adjudication `REJECTED` |
| N1-CUTIN calibration | 49 条历史人审 / 26 scenes；第三次 FP 拒绝 12/12、第二次 FP 拒绝 35/35、旧 TP 保留 1/2；三项 gate 通过 |
| N1-CUTIN 定义 | 2 Hz center outside→post box inside；pre heading；独立 RECEIVER pre/post identity；gap `[0.5,40] m`；30-frame matched negative |
| N1-CUTIN 工程 | lightweight map reader 与官方 arcline reference 一致；单 location cache；流式 metadata；batch=32 |

完整报告见 [`N1_MINI_EVENT_POOL_REPORT.md`](N1_MINI_EVENT_POOL_REPORT.md)、
[`N1_FULLDOMAIN_EVENT_POOL_REPORT.md`](N1_FULLDOMAIN_EVENT_POOL_REPORT.md) 与
[`N1_KINEMATIC_EVENT_POOL_REPORT.md`](N1_KINEMATIC_EVENT_POOL_REPORT.md)。

## 5. 下一路线与闸门

首选路线是“event-first map-and-raw-evidence counterfactual pipeline”，不是重命名后的 OccGS H1：

| Gate | 目的 | 通过条件 | 失败动作 |
|---|---|---|---|
| `N0-ASSET` | 建立可审计地图/数据底座 | 官方 vector map 可加载；scene→map 映射与 hash 完整 | **PASSED** |
| `N1-EVENT` (mini) | 证明比较对象存在 | ≥2 positive、≥2 negative、≥2 same-actor pairs、≥2 positive scenes | **REJECTED**（mini pool 太小） |
| `N1-EVENT-FULL` (val 146) | 同上，full-domain + graph-corridor relation | 机器支持 + 人工真实性 | **HUMAN REJECTED**：机器 37 pos 中仅 2 TP / 35 FP；见 [`N1_FULLDOMAIN_EVENT_POOL_REPORT.md`](N1_FULLDOMAIN_EVENT_POOL_REPORT.md) |
| `N1-EVENT-KINEMATIC` (train, 第三版) | 以 subject 物理运动学与持续交互重建事件池 | 先过冻结机器支持门槛，再过第三次人工门槛 | **HUMAN REJECTED**：12/12 FP，且 negative/pair=2/2 |
| `N1-EVENT-CUTIN` (train, 第四版) | receiver-centric 车身进入 + 独立接收车 | machine：8 pos / 4 neg / 4 pair / 5 scenes；human：≥6 TP、precision≥0.8、Wilson≥0.5 | **IN PROGRESS / calibration passed**：49 条旧审标签只作 calibration；等待 formal train 与第四次人审 |
| `N2-EVIDENCE` | 建立独立合法性参照 | 新 N1 通过 + 用户显式裁决 + 传感器授权 | **locked / forbidden**；旧名单已失效且当前配置 fail closed |
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
- 用 10 Hz 插值点计算速度、加速度或 yaw rate，或把插值 cadence 当作独立物理观测；
- 再次用“target lane/lane_connector 有多个 incoming”直接判定 subject 正在 merge，或在分叉 corridor
  同时纳入多条支路制造虚假 front/rear；
- 在新的 event pool 通过前启动 N2/N3/render/training。

## 7. 当前任务队列

1. 已完成：N0/mini-N1 归档；第二版 full-domain 机器海选与 37/37 人工裁决分开固化，最终人工证据为
   2 TP / 35 FP，第二次 N1 正式 `REJECTED`；
2. 已完成：第三版 12/12 人审均为 FP；review hash、失败码与独立 `REJECTED` adjudication 已固化；
3. 已完成：第四版 receiver-centric 事件定义、历史 49 条 calibration replay、内存工程修复和人审 schema
   validator；阈值只由历史已审数据冻结；
4. **当前自动执行**：在 clean preregistration commit 上跑 official train scene-disjoint formal，
   生成 event pool、matched controls、完整 K4 panel/evidence/checklist/prompt/blank JSONL，并复算全部 hash；
5. **下一人工点**：只在完整第四次材料交付后由用户/指定评审填写 `review_working.jsonl`；agent 不代填；
6. N2 保持 fail closed；不得事后挑 scene、放宽阈值、复用 source rear、缩短 negative、允许 event overlap，
   或因第四版存在候选就自动启动 N2。

## 8. 事实源优先级

发生冲突时按以下顺序处理：

1. 实际 run 产物、resolved config、原始指标、checkpoint 与 terminal marker；
2. [`EXPERIMENTS.md`](EXPERIMENTS.md)；
3. 本文件；
4. [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)；
5. 当前预注册路线；
6. `docs/archive/` 中的历史计划、报告和提示词。
