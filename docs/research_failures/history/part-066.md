# 历史原始记录 066

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### N1-F11：完整审核包不等于每项都有前视相机可见证据

**观察**

- 正式包包含 12/12 panels、evidence、topdown、checklist、prompt 和逐文件 SHA256；
- 本机 full train 数据完整覆盖 CAM_FRONT，但其他五个相机目录只有 mini 规模，不能为这些 formal scenes
  提供稳定六相机视图；
- 首尾面板 QA 正常；部分 subject/front/rear 不在 CAM_FRONT 视野，但 2 Hz annotation topdown 仍存在；
- 40 个 immutable audit files 复算 0 hash mismatch；空白 review validator 按预期 fail closed。

**审核边界**

不在 CAM_FRONT 中的角色不能被猜测为 VALID。评审先使用 topdown、vector centerline 和跨时刻 identity；
若相机与 annotation 冲突或证据仍不足，必须判 `UNCERTAIN` 并记录
`INSUFFICIENT_VISUAL_EVIDENCE`。补六相机或 raw LiDAR 需要独立资产/用途授权，不能偷偷进入本轮或 N2。

### N1 第三版历史禁止重试矩阵

| 快捷做法 | 为什么无效 | 合法后续 |
|---|---|---|
| 把 parent `AWAITING_HUMAN_REVIEW` 写成 pass | negative/pair 失败且人工 12/12 FP | 独立 adjudication 已 `REJECTED` |
| 用人工结果覆盖 pair gate | authenticity 与 comparison support 是不同问题 | 两类 gate 均保留 |
| 缩短/重叠 negative window | 事后改变 matched-control 定义 | 新任务、新 split、新预注册 |
| 用其他 actor 补足 same-actor pair | 混入 actor/scene confound | 预注册 matched-other-actor 设计 |
| 把 63 个 physical lane changes 加入 positive | 它们没有通过冻结 interaction | 单独 subject-only diagnostic |
| 用单侧 front 或 rear 算 interaction | 改变“插入双侧 gap”的研究对象 | 另立事件 subtype |
| 没有相机框也猜 TRUE/FALSE | 把证据缺失转成标签 | `UNCERTAIN` |
| 审核后自动启动 N2 | 本 run 明确 `n2_authorized=false` | 新授权 + 新 gate |

## N1 full-domain 第二次 reject（2026-07-25）

### N1-F05：把 target 多 incoming 的地图类别误当成 subject 行为

**观察**

- 父机器 run `N1-EVENT-FULL-01` 在 val 146 上报 37 个 positive，其中 topology 为 35 merge + 2 lane change；
- 完成人审文件 SHA256 为
  `ae71b31e02faf1d783c36748e629e85acf32a132f35ce2f98102a5f62201dd05`；
- 用户确认的逐项结果为 `TRUE_POSITIVE=2`、`FALSE_POSITIVE=35`、`UNCERTAIN=0`，机器候选精度
  `2/37=0.054054`；
- 多数 reviewer notes 明确指出：subject 沿与 target 共线的主路 lane/connector 正常直行，真正汇入
  target 的是另一条 incoming branch；旧规则却只因 `target_incoming_count>=2` 就把 subject 标为 merge；
- 独立 audit adjudication：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-FULL-AUDIT-01/v71_n1-event-full-audit-01__human-audit-reject-v1__s0__20260725T083929632491Z__6507cbac/`，
  唯一终态 `REJECTED`。

**能下的结论**

graph-corridor 修复了邻车跨 token fragmentation，却没有证明 subject 本身执行了 lateral maneuver。
“target 有多个 incoming”是地图节点属性，不是 actor-specific merge 证据；第二次 N1 不能进入 N2。

**不能下的结论**

不能据此断言 full nuScenes 没有真实 lane change/merge，也不能把 2 个标注 TP 当成已验证的完整事件池。
旧 audit panel 没有把 subject/front/rear 的 3D identity 投影到图像，且 reviewer 字段包含多个来源；
用户已整体确认 reject，但单条 TP 仍只能作为第三版 calibration 标签，不得直接进入 formal evaluation。

### N1-F06：10 Hz 插值 cadence 被错误提升为物理证据

**观察**

nuScenes `sample` 是 2 Hz 标注关键帧。第二版把 2 Hz box 线性/SLERP 插值到 10 Hz 后，用连续 lane-token run
寻找 transition；该做法对齐了 DriveStudio cadence，却没有产生新的物理观测。第二次人审 notes 多次指出
短 token 切换、轨迹插值或 map assignment 假象。

**防重复**

- 第三版速度、加速度、yaw-rate、lane preference、front/rear persistence 只能用原始 2 Hz keyframe；
- 10 Hz 只用于 frame 对齐、可视化和复现旧 transition 候选，不得计算导数或宣称 0.1 s 观测；
- 至少 3 个 pre 和 3 个 post keyframes；不足时为 `UNKNOWN`，不得靠插值补齐。

### N1-F07：单时刻中心距不是持续物理交互

**观察**

第二版在单一 relation frame 上用中心线 `s` 与中心距 `[2,60] m` 选择 front/rear，没有扣除 box extent，
也不要求同一 front/rear identity 跨时刻持续。37 个 machine positive 中 36 个至少依赖一个跨-token 邻车，
因此 branch 选择错误会直接翻转结果。

**第三版解除条件**

1. target corridor 每个 graph edge 同时满足方向连续和 endpoint 连续，并只选单一最连续分支；
2. 使用 oriented box 在 lane tangent 上的投影半长，报告 bumper gap 与 center gap；
3. 至少 2/3 个连续 2 Hz keyframe 保持同一 front/rear identity、方向和次序；
4. 同时报告 longitudinal speed、closing speed、headway/TTC；它们是诊断，不得替代人审。

### N1-F08：第二版审核合同与 provenance 不足

**观察**

- 父机器 run 诚实记录 `code_dirty=true`；它可定位但不是 clean-commit formal baseline；
- 旧人审清单给了逐项 verdict 定义，却未预注册聚合阈值；
- 因此第二次 reject adjudication 没有查看结果后补造阈值，只登记用户明确决定；
- 旧 CAM_FRONT 清单没有身份 box overlay，容易把画面中“真正并道的另一辆车”认成 subject。

**复开条件**

第三版必须在 clean commit 上运行；正式 audit pack 同时提供盲序、subject/front/rear 颜色框、2 Hz 俯视轨迹、
逐项 component verdict、failure codes、完整提示词、immutable file hashes、预注册统计阈值和独立
adjudication 命令。Agent 不得填写人工 verdict。

### N1 full-domain 禁止重试矩阵

| 快捷做法 | 为什么无效 | 第三版允许替代 |
|---|---|---|
| 继续调 `graph_hops` 或 gap | 35/37 误报的主体事件本身不成立 | actor-specific 2 Hz kinematics 先行 |
| target 有多个 incoming 就叫 merge | 把地图节点类别当行为 | 比较 source 与主路 incoming 的 approach geometry |
| 用已审 val 37 条挑最终阈值并在同一 split 报结果 | calibration/evaluation 泄漏 | val 只 calibration；official train formal evaluation |
| 从 10 Hz 插值计算速度/横移 | 人造高频证据 | 原始 sample timestamp + 2 Hz boxes |
| 单帧 front/rear 中心距 | branch/identity 易跳变，忽略车长 | branch-safe corridor + temporal identity + bumper gap |
| 把 2 个旧 TP 直接当第三版正例 | panel identity 仍有未决风险 | 只作 calibration；第三版候选重新盲审 |
| 机器候选一出现就启动 N2 | 人工真实性与样本支持尚未通过 | `AWAITING_HUMAN_REVIEW`，`n2_authorized=false` |

## N1 mini event-pool reject（2026-07-24）

### N1-F01：interaction-support failure

**观察**

- N0 map-expansion、scene→map 与 pose contract 已通过，不再是资产缺失；
- 45 个 source-only eligible actors 产生 71 个 stable token transitions；
- topology taxonomy：39 route continuations、19 merges、3 lane changes、10 unresolved；
- 19 merges + 3 lane changes 共 22 个 topology-pass candidates；
- 22/22 的 exact-target-token front/rear relation 为 FAIL；
- 18 个没有 target-token 邻车，4 个只有 front、没有 rear；0 个同时满足 2–60 m front/rear；
- positive=0、negative pairing=0、same-actor pair=0、positive scenes=0，唯一终态 `REJECTED`。

**能下的结论**

冻结 mini split 不支持可比较 interaction event pool，N2–N5 不触发。地图缺失不是旧 H1 的唯一根因；
补地图后 mini interaction support 仍为零。

**不能下的结论**

不能写成“人类绝对看不到任何交互”或“full nuScenes 也没有事件”。exact target token 可能把同一
longitudinal corridor 上的 actor 分到相邻 lane/connector token；该表示风险尚未独立校准。

**复开条件**

mini run 不复开。新的路线必须：

1. 使用不同 run/task ID；
2. 以 22 topology-pass mini cases 仅作 calibration/audit，不作 formal evaluation；
3. 在 graph corridor 上定义 route-aligned curvilinear front/rear，而非后验放宽欧氏半径；
4. calibration 与 evaluation scenes 分离；
5. 优先在 full nuScenes trainval annotations/metadata 上冻结并评估。

### N1-F02：exact-token corridor fragmentation

**观察**

71 transitions 中 39 个只是 directed route continuation，说明官方 lane graph 将连续道路划分为多个
lane/lane_connector token。当前 interaction 只接受 relation frame 上与 subject 完全相同的 target token。

**推断**

该规则高精度但可能低 recall，尤其在 lane→connector→lane 或短 lane segment 附近。它是 0 interaction
PASS 的一个可能贡献因素，但不是已证实的唯一原因；mini 本身也可能确实缺少前后车。

**禁止快捷修补**

- 不把“相邻 token”全部并入；
- 不把只有 front 或只有 rear 改成 positive；
- 不把 82–89 m front 后验纳入 60 m；
- 不在同一 22 cases 上调 graph hops、gap 或 heading 直到出现 positive。

允许的修复是先定义有向 corridor、route-aligned `s` 和 branch disambiguation，再由独立 calibration
审计冻结；formal evaluation 必须 scene-disjoint。

