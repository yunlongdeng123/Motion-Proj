# Motion-Proj 当前研究风险与防重复账本

> **最后更新**：2026-07-24
> **当前范围**：V1–V7.1 防重复约束、H1 reject 根因分解与 Post-OccGS 新路线风险。
> **历史账本**：完整 `RF-01`–`RF-18` 原文见
> [`archive/2026-07/v7-feasibility/RESEARCH_FAILURES_RF01_RF18.md`](archive/2026-07/v7-feasibility/RESEARCH_FAILURES_RF01_RF18.md)。
> **事实源**：[`EXPERIMENTS.md`](EXPERIMENTS.md) 和实际 run 产物。

本文件保留仍约束后续路线的历史结论，并把 H1-11D 的失败严格分为“观察到的事实、合理推断、尚未知、
复开条件”。归档不会使旧失败失效；任何新计划复用旧机制时仍须满足原 RF 的重开条件。

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

### N1-F03：mini scale 与静止对象密度

**观察**

- 003/005/004 eligible actors 为 7/22/16；
- 因首尾位移不足 5 m 被拒的 actor 为 107/17/5；
- eligible pose map-match coverage 为 88.89% / 95.60% / 93.36%；
- 官方 full nuScenes 有 1,000 个约 20 秒 scenes，850 个为 train/val，而当前 formal pool 只有 3 scenes。

**结论**

mini 三场景对多 scene interaction event pool 的统计支持不足。下一步应扩数据底座，不应换 actor 或删场景。
优先同域 `v1.0-trainval` annotations/metadata，只有其 event gate 仍失败才评估 nuPlan/Waymo。

### N1-F04：negative=0 的语义

N1 只为已经有 positive 的 actor 构造 same-actor comparable negative。因此 `negative=0` 是
`positive actor set=∅` 的结构结果，不证明没有稳定非事件窗口。后续报告必须同时给出 positive actor 分母，
不得把 negative=0 解释为数据中全是事件或完全无普通驾驶。

### N1 禁止重试矩阵

| 快捷做法 | 为什么无效 | 允许替代 |
|---|---|---|
| 删除 rear requirement | 改变冻结 interaction claim | corridor calibration + scene-disjoint evaluation |
| 扩大 60 m 到覆盖 82–89 m | 看结果后调阈值 | 在新 calibration pool 依据任务时间窗冻结 |
| exact token 改成任意相邻 token | 可能跨 branch/对向车道误配 | directed corridor + route-aligned `s` |
| 从 22 cases 挑“看起来像”的 positive | 人工/后验标签泄漏 | 完整盲审协议；calibration 不进入 eval |
| 在 005 单 scene 继续 | 删除失败 scene、失去多 scene gate | full trainval scene-disjoint split |
| 直接启动 N2/N3/render | 没有 comparable event | 新 N1 先通过 |

## 0. H1 reject 执行摘要

### 0.1 为什么 reject

| ID | 层级 | 观察到的事实 | 能下的结论 | 不能下的结论 |
|---|---|---|---|---|
| `H1-F01` | 事件存在性 | 30 proposals：0 positive、25 negative、5 source-positive/non-event、0 same-actor pair | 冻结 proposal bank 不支持 H3 或配对因果比较 | “occupancy 一定无效”或“换几个 actor 就会成功” |
| `H1-F02` | certificate 精度 | D1 TP=15、FP=5、precision=0.75 < 0.80 | H1-CERT 按预注册 reject | 仅因 recall=0.8824 就称 certificate 通过 |
| `H1-F03` | certificate 覆盖 | D1 UNKNOWN=10/30、PASS=0、PASS coverage=0 | 当前证据无法给出足够确定的正判定 | 把 UNKNOWN 排除或并入 PASS 后重算 |
| `H1-F04` | repair 吞吐 | D2 reject=30/30、export=0、usable yield=0 | H1-PROJ 按预注册 reject；外部 rate 不可定义 | “导出集 0/0 违规，所以修复完美” |
| `H1-F05` | 数据效用 | 无 positive pair，H1 已拒绝 | H3 不触发 | 以 RGB 差分、accept rate 或 proxy 代替下游任务 |
| `H1-F06` | 高成本阶段 | H1 前置 gate 失败 | H2/render audit/blind pack 不实例化是正确停止 | “没跑 H2，所以 H1 结论不完整” |
| `H1-F07` | 统计实现 | 首版 aggregate 把 rejection 计成零违例 | 聚合 bug 已修复且不影响方法输出 | 用首版 aggregate 支持方法 claim |
| `H1-F08` | 资产/证据 | 本机 map 只有 raster PNG；base UNKNOWN 约 96–98% | lane/road support 与独立覆盖存在硬缺口 | 从 raster 或 learned occupancy 静默补成真值 |

### 0.2 冻结证据

- 正式 run：
  `/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11D/v71_v7-h1-11d__pilot-3-matched__s0__20260723T155755269940Z__cf8d5ebc/`；
- proposal-bank SHA256：
  `f8986915f8d2be0cddddfa6be86f4d2d1ece456c12bf9a962cafec78fd058cd7`；
- config SHA256：
  `cf8d5ebc1429e076fc5142aa6a759a18f54b7f3f937c8423d51505a094bc9fe3`；
- C/D1 realized trajectory 30/30 identity；
- C external hard violation 17/30：003=5/10、005=7/10、004=5/10；
- D1：15 TP、5 FP、2 FN（含 abstention）、20 FAIL、10 UNKNOWN、0 PASS；
- D2：0 accept/export、0 usable yield；
- 唯一 terminal marker：`REJECTED`。

### 0.3 逐失败点根因、卡点与复开要求

#### `H1-F01`：proposal-support failure

**观察**

- source-only eligibility 固定为 3 scenes × 2 actors；
- 每 actor 固定 P1–P5，共 30 个 proposal；
- scenario-effect 没有产生任何 `0→1` positive；
- 5 个 source-positive case 在 proposal 后成为 non-event，25 个为 negative；
- 没有 same-actor positive/negative pair。
- 后续只读 continuity 审计发现，冻结 actor 003:38、003:35、005:23 在完整连续 track 内的 world
  displacement 仅 0.88 / 0.29 / 0.76 m；原 source-only 排序偏好长、清晰 track，但没有事件相关性。

**推断**

固定横向位移只满足几何“移动过”，没有以 lane topology、corridor crossing、target-lane front/rear gap、
duration 或 interaction 定义事件。该设计与“cut-in/merge 正例”的目标错位。这是由结果和 schema 支持的
最强解释，但尚未通过 vector map 重标注，所以不能断言每个 case 的唯一失败原因。

**卡点**

- 本机缺 nuScenes map-expansion vector JSON；
- mini 三场景的真实事件上限未知；
- 没有先冻结 natural-event pool，就不能知道是 proposal family 失败还是场景本身无事件。

**复开条件**

不是改 P1–P5。必须新建 event-first 路线：先冻结 map/track 事件定义和 actor pool，证明存在预定数量的
positive/negative 与 same-actor pair，然后才允许生成候选。若 mini 事件池不足，应 reject mini pool 或
请求新数据授权。

#### `H1-F02`：certificate precision failure

**观察**

5 个 FP 全来自 scene 004 actor 8；certificate 报告 5 个 static-overlap voxels，而独立 raw LiDAR
检查为 0 points。D1 precision 为 0.75，低于冻结门槛 0.80。

**推断**

结果与 coarse voxel quantization、box-to-voxel 接触或证据层不一致相符；尚不能证明是哪一个机制，也不能
从“0 raw points”推出空间一定安全，因为 LiDAR 可能受遮挡和采样稀疏影响。

**卡点**

- `0.4m` 离散 grid 将连续几何压成二值接触；
- 单一 voxel overlap 缺少距离、置信度和观测支持；
- static/dynamic 分层仍可能受历史 sweep 和运动补偿影响；
- raw point absence 不是 free-space ground truth。

**复开条件**

在 scene-disjoint calibration pool 上比较 coarse voxel 与 motion-compensated raw sweeps 的连续
point-to-OBB/swept-volume distance；逐类报告量化、动态残影、遮挡、地图边界和标注误差。门槛必须在
冻结评估前预注册，不能用 actor 004:8 调到通过。

#### `H1-F03`：coverage/abstention failure

**观察**

三场景 base unknown 约为 97.10% / 96.04% / 97.57%；D1 10/30 UNKNOWN，PASS coverage 为 0。
两个 FN 位于 005 的 P3/P5，D1 known fraction 为 0，而 raw LiDAR 只有 3/2 points。

**推断**

当前 single/coarse observation 无法支持大部分 free/occupied 判定。两个 FN 说明“极少 raw points”
也不能自动解决判定；具体是遮挡、采样、时序或标注问题仍未知。

**卡点**

- raw LiDAR 稀疏；
- 缺 vector drivable/lane polygons；
- 多 sweep 若不做动态/ego motion compensation 会制造 ghost；
- learned completion 会提高表面 coverage，却失去独立真值身份。

**复开条件**

增加独立 evidence，而不是调低 known-fraction：官方 vector map、ego/dynamic compensated sweeps、
显式 truth tier 与 uncertainty。继续报告 PASS/FAIL coverage 和 abstention；任何 learned occupancy
只能是附加证据层，不能作为外部 evaluator。

#### `H1-F04`：repair all-reject failure

**观察**

D2 没有接受或导出任何 proposal；usable yield=0，外部 violation rate 无分母。

**推断**

当前 projection/repair 约束组合没有可用工作区，或者 proposal 全都离可行域过远。因为 0 export，无法
区分“repair 算法差”与“输入候选全不可修复”各自贡献。

**卡点**

- 没有成功样本用于 paired outcome；
- 先验 proposal 不由 lane-reachable set 生成；
- 二值 certificate 既可能过严又可能不准；
- H2/H3 都依赖 D2 产出，故被同时锁死。

**复开条件**

先通过 N1 证明事件存在，再以 lane graph/target state 生成 reachable proposal；冻结 minimum usable yield、
comparable export 数和外部 evaluator。若仍为 all-reject，直接 reject proposal/repair family。

#### `H1-F05`：metric aggregation bug

**观察**

首版 summary 把 rejection 计为零违例，使 0 export 看起来像 0% external violation。唯一允许的
`metric_aggregation_bug` 修复保留了旧 aggregate；修复后无 export 时 fail closed。修复提交为
`b82c540`，不改变 proposal、trajectory、certificate 或 D2 输出。

**防重复**

- 所有 rate 必须同时报告 numerator、denominator、rejected、unknown；
- denominator=0 时写 `undefined`，不能写 0；
- terminal decision 必须读取 comparable export 和 usable yield；
- 原始 aggregate 不覆盖，修复生成新版本并记录 migration。

#### `H1-F06`：地图资产与证据缺口

**观察**

`/root/autodl-tmp/data/nuscenes/maps/` 只有 4 个 PNG，没有 vector JSON；本机没有 Waymo/nuPlan 数据。
`/root/autodl-tmp` 约有 65G 可用空间。

**卡点**

官方 lane graph/drivable polygon 暂不可查询；不能可靠地定义 target lane、connectivity、off-road 或
corridor crossing。DriveStudio adapter 代码的存在不等于数据和许可就绪。

**复开条件**

先生成最小资产清单并取得下载授权；保存来源、许可、大小、SHA256 和 scene→map 映射。不得从 raster PNG
反推正式 lane graph，也不得静默下载全量 Waymo/nuPlan。

### 0.4 禁止重试矩阵

| 快捷做法 | 为什么无效 | 允许的替代 |
|---|---|---|
| 降 known-fraction / coverage | 把无证据改名为有证据 | 增加独立 map/raw evidence |
| UNKNOWN 并入 PASS/FAIL | 改变预注册语义和分母 | 继续三态并单列 coverage |
| 删除 S1、005 或 004 actor 8 | 后验删难例 | scene-disjoint 新 pool |
| 换 actor、方向、P1–P5 幅度 | 用结果挑 proposal | 先冻结 event definition 与 actor pool |
| 0 export 报 0 violation | denominator=0 | 报 undefined + yield=0 |
| multi-sweep 直接堆叠 | 动态物体会 ghost | ego/dynamic motion compensation |
| learned occupancy 当 GT | 方法与 evaluator 循环 | raw/map 独立 evaluator + calibration |
| GS floaters/画质当安全证据 | renderer 不是物理传感器 | GS 只在 N4 导出 |
| 先做 H2/H3/scale | 没有 comparable positive | N1–N3 先过门 |
| 重命名 OccGS 复开 | 没有解除原失败 | 新路线必须满足复开条件 |

### 0.5 可复用资产

失败不否定以下工程资产：

- coordinate contract、`WorldState`、typed label/depth；
- run contract、artifact index、terminal marker 和 fail-closed aggregate；
- object-centric GS reconstruction/renderer；
- D1/D2 接口和 `PASS/FAIL/UNKNOWN` schema；
- 冻结 H1 bank 作为负对照与回归 fixture。

复用这些资产不能继承 H1 claim；新路线必须有新 preregistration、独立 event pool 和 evaluator。

## 1. 仍直接约束 V7 的历史结论

| ID | 状态 | 对 V7 的约束 |
|---|---|---|
| `RF-05` | rejected | 合法轨迹/点或局部像素变化不等于 RGB、遮挡、source removal、depth、identity 与标签都合法 |
| `RF-06` | rejected | 局部 loss 或 mask 不保证参数/输出只在局部改变；必须测 outside、boundary、frame-0 与 held-out |
| `RF-08` | limitation | 可复现的机器 evaluator 不等于绝对物理真值，更不能替代人工 verdict |
| `RF-09` | rejected | same-scene、shared identity 或结构合法不等于人类能辨别方法收益 |
| `RF-16` | limitation | layout/trajectory controllability 不等于 action-disentangled actor physics 或数据效用 |
| `RF-18` | rejected | ReSim `exp0_no_carla` 的 E-vs-F action response 不足；V7 不得借归档重开 C1P/C1S |

其他 RF 仍完整有效，但当前 OccGS 计划不直接复用对应的 SVD projection/preference 配方。

## 2. V7 风险索引

| ID | 状态 | 风险 | 禁止的快捷修补 |
|---|---|---|---|
| `V7-RISK-01` | rejected_v71 | occupancy 已接入 11D，但 certificate precision 与 repair yield 均未过预注册 gate | 因为 occupancy 文件存在或 D2 无 export 就宣称 H1 通过 |
| `V7-RISK-02` | limitation | C0 24/24 是按效应 top-k 的机器筛选，不是用户人工评测 | 写成 human pass，或只报 top-k 隐藏 46/62 全分布 |
| `V7-RISK-03` | open_risk | L0 mask 来自 RGB 差分，outside=0 由 hard composition 构造保证 | 用 0 leakage 宣称 occupancy-guided completion 有质量收益 |
| `V7-RISK-04` | open_risk | U0 以极端 V4 为 naive 对照且没有下游任务 | 把 accept rate / RGB signal 写成优于 naive GS 或 mAP 收益 |
| `V7-RISK-05` | legacy_limitation | V7 既有 run 缺正式 manifest、resolved config 与终态标记；V7.1 新 run 已由 EV-10 fail closed | 事后猜 seed/fingerprint 或伪造 immutable provenance |
| `V7-RISK-06` | open_risk | 只覆盖 mini 三场景，S1 held-out 质量偏弱 | 先扩规模、只筛容易场景或把三场景外推为论文结论 |
| `V7-RISK-07` | interface_mitigated_v71 | 11C 已闭合 WorldState→renderer→typed-label 工程链；occupancy repair 的方法增益仍未验证 | 把 label-sync 工程通过写成 occupancy certificate/projection 通过 |
| `V7-RISK-08` | legacy_risk_mitigated_v71 | O0 坐标注释、metadata 与实际变换含义不一致；11A 已冻结显式 frame 合同 | 沿用含义不明的 `pose/T`，或在 round-trip 前计算 H1 指标 |
| `V7-RISK-09` | confirmed_mitigated_v71 | 旧 rotated-corner AABB 使 PILOT-3 动态体素量膨胀 1.72–2.83 倍；扁平语义不能诚实移除 actor | 把旧 O0 AABB 当正式安全几何，或移除 actor 后把体积恢复为 free |
| `V7-RISK-10` | confirmed_failure_v71 | 高 UNKNOWN 在 11D 导致 10/30 D1 abstain、D2 30/30 拒绝与 0 usable yield | 把 UNKNOWN 并入 PASS/FAIL，或降低观测门槛追求 yield |
| `V7-RISK-15` | architecture_mitigated_v71 | certificate detection 与 trajectory projection 若混组会混淆检测和修复收益 | D1 修改 C trajectory，或把 D1/D2 合成单一 validity 数字 |
| `V7-RISK-16` | confirmed_failure_v71 | 冻结 30-proposal bank 得到 0 个 0→1 positive 和 0 个 same-actor pair | 用位移幅度或 RGB 差分代替 scenario-effect gate，或事后换 actor |
| `V7-RISK-17` | confirmed_mitigated_v71 | 单一 `depth` 名称会混淆 expected、first-hit 与 LiDAR measured truth tier；11C 已强制分名和 sidecar | 把 expected depth 登记为 measured GT，或省略 validity/truth-tier |

## 3. 风险详情与解除条件

### V7-RISK-01：occupancy 尚未进入方法

**观察**

- `occupancy/build_scene_occupancy.py` 独立写出 per-frame grid；
- `resim/s0_trajectory_editor.py` 只检查横向运动学、yaw、actor/ego 距离和粗横向范围；
- `resim/c0_counterfactual_render.py` 改写 RigidNodes pose，但没有查询 occupancy；
- `resim/l0_local_completion.py` 用 V0/edited RGB 差分构 mask。

**边界**

O0 是有用的世界状态基础设施，但当前不能支持“occupancy 提高合法性”或“occupancy-guided completion”主张。

**解除条件**

按 `V7-H1-11` 建立统一 actor/state mapping，让 occupancy 进入 edit certificate、visibility 与标签重生，并对
matched kinematic-only/naive baselines 做非循环消融。只添加一次 occupancy lookup 或 post-hoc filter 不足以解除。

### V7-RISK-02：机器 top-k 不等于人工合法率

**观察**

- C0 全部可见 case 为 46/62 machine legal；
- 24/24 是按 mean edit effect 排序后的 top-24；
- 当前 `reviews/` 目录是机器面板与机器 JSON，没有用户填写的 verdict。

**边界**

可表述为“机器筛选 top-24 均满足当前规则”，不得表述为“24/24 人工合法”或用其估计全候选分布。

**解除条件**

先冻结 blind sample、逐项 rubric、失败优先级、JSONL schema 与聚合阈值，再由用户或指定评审者完成 verdict。
agent 不代填，也不以机器字段映射成人工答案。

### V7-RISK-03：L0 primary metric 目前是构造不变量

**观察**

hard composition 直接复制 mask 外的 edited GS，因此 outside-mask L1 必然为 0；当前 12 帧结果只验证实现遵守
公式。mask 由 RGB 差分阈值和膨胀获得，不包含 ray visibility、unknown/free 或 source footprint geometry。

**边界**

L0 只证明 local composition 工程可行。没有证据表明 Telea 改善视觉、时序、depth 或 identity。

**解除条件**

使用 geometry-derived disocclusion mask，并在有真值的 pseudo-hole 上比较 no completion、Telea 与局部生成；
primary 必须包含 inside quality、boundary、temporal、depth/instance，而不是继续调阈值追 outside=0。

### V7-RISK-04：U0 proxy 不识别数据效用

**观察**

`naive_V4` 是约 39–50 m 的强制横移负例；它被拒绝只能证明 validator 能识别一个极端错误。当前没有训练
detector、occupancy model 或 event classifier，JSON 明确记录 `u0_full_map_pass=false`。

**边界**

不能声称 OccGS 优于 matched naive GS、real-only 或提供下游增益。

**解除条件**

对相同 proposal、相同样本量和相同训练预算比较 R / R+naive / R+OccGS / R+OccGS+completion，并使用
scene-disjoint split、至少 3 seeds 和任务指标。三场景只可用于 pipeline smoke。

### V7-RISK-05：既有 run provenance 不完整

**观察**

`runs/occgs_resim/` 现有 B0/C0/L0/U0 目录未发现 `manifest.json`、`resolved.yaml` 或终态标记。B0 仍有
`config.yaml`、metrics、checkpoint；其他阶段有 JSON 报告，但不足以满足正式 run contract。

**边界**

现有数值可作为 retrospective evidence，不能声称是完整、不可变、可从 manifest 一键复现的正式 run。

**解除条件**

`V7-EV-10` 为既有证据生成显式缺失项索引；所有新 run 通过 fail-closed wrapper 产生完整协议。禁止事后补造
未知字段或覆盖旧目录。

**2026-07-23 缓解结果**

- `V7_EVIDENCE_INDEX.json` 已逐文件索引 B0/O0/S0/C0/L0/U0 的 1,610 个文件，并保留正式字段的
  `missing/unknown_not_inferred`；
- V7.1 run contract 对 run ID 复用、三层 hash、artifact bytes、summary、冲突终态标记和 optional
  `not_triggered` 分支 fail closed；
- 正式 smoke 在 commit `3590558` 上以唯一 `COMPLETE` 结束，25 项相关测试通过。

该缓解只约束 V7.1 新 run；V7 旧 run 的 provenance 缺口不可逆，仍保持 retrospective/legacy limitation。

### V7-RISK-06：场景覆盖与质量

**观察**

本机只有 mini 10 scenes 具备前向完整 sweep；feasibility 只使用 3 scenes。S1 test PSNR/SSIM 为 20.18/0.472，
明显弱于 S0/S2。

**边界**

当前结果不能外推到 trainval、长时、多相机、夜间或复杂交互；也不能只删掉 S1 后报告更好均值。

**解除条件**

H1 先在冻结三场景与 worst-case 上通过，再审计可获得的 scene-disjoint 数据。扩展必须保留困难场景分层、
真实/插值 provenance 与相同门禁。

### V7-RISK-07：标签链未闭环

**观察**

C0 已改写 RigidNodes pose 并输出 RGB/depth/rigid 分量，但尚未形成统一的 semantic、instance、2D/3D box、
occupancy 与 visibility regeneration 流水线。

**边界**

“label synchronization”当前只可称 proxy/interface 可行，不是完整传感器与标签一致性。

**解除条件**

同一 world-state record 驱动 renderer 与所有标签 writer，逐帧验证 pose、depth、mask、box 和 occupancy 共位；
对缺失/不可见标签 fail closed。

**2026-07-23 缓解结果**

- 11C 在 PILOT-3 的 V0/V1、三场景、三前向相机上生成 18 个样本和 432 个 typed sidecar；
- 独立审计验证 18/18 样本、6/6 WorldState hash、temporal identity、三相机覆盖、instance-depth z-order 与
  state-specific safety/observation/render-support 引用；
- expected、first-hit、LiDAR measured depth 分名，有限 semantic scope 和 visibility provenance 均写入 sidecar；
- S1 保留，正式 run 以唯一 `COMPLETE` 结束。

该结果只解除 renderer/label 工程接口风险；11D 之前仍不能声称 occupancy certificate 或 repair 有方法收益。

### V7-RISK-08：O0 坐标框架歧义已确认

**观察**

- `occupancy/build_scene_occupancy.py` 文件头将 grid 描述为首帧 ego-centric；
- `meta.json` 将同一产物描述为 per-frame ego-centric；
- 实际实现每帧读取 `lidar_pose/{t}.txt`，以其逆矩阵把 world box 变换到 grid，同时直接使用 sensor-local
  LiDAR 点。因此产物实际是 per-frame LiDAR-sensor grid，而不是首帧固定 grid，也不能在未审计 LiDAR-to-ego
  外参前简称 ego frame；
- DriveStudio 则以起始 `CAM_FRONT` 的 `camera_to_world` 逆矩阵定义 model frame。

**边界**

现有 O0 数值仍可作为 coarse retrospective evidence，但在显式记录 `T_grid_world`、`T_model_world`、
`T_world_camera` 并通过 world→model/grid→world round trip 前，不得用于 H1 合法性指标。

**解除条件**

`V7-H1-11A` 统一使用 `T_dst_src` 命名，修正新 schema/adapter 的 frame 声明，以 synthetic fixtures 和
PILOT-3 原始标定验证 translation、yaw、box corners、camera projection 及 checkpoint pose round trip。
旧 O0 文件不原地改写；正式 H1 evidence 产生新版本与新 fingerprint。

**2026-07-23 缓解结果**

- 11A 将 annotation/model/grid/camera/LiDAR frame 分别冻结为 world、start-CAM_FRONT、per-frame-LiDAR、
  `T_world_camera` 与 `T_world_lidar`；
- 三场景 1,679 个 actor poses 的 translation、rotation、box 和三前向相机投影 round-trip gate 通过；
- registry 跨独立进程重建 hash 完全一致，正式 run 以唯一 `COMPLETE` 结束。

旧 O0 metadata 不原地改写，故该风险仍是 retrospective artifact 的 legacy limitation；V7.1 后续模块必须引用
11A coordinate contract 和新 fingerprint。

### V7-RISK-09/10：AABB 膨胀与高 UNKNOWN 已确认

**观察**

- 在完全相同的 PILOT-3 raw annotation、grid 和 240 帧上，旧 rotated-corner AABB 相对 oriented-box
  center-inclusion 的动态体素量比分别为 003 `1.721×`、005 `2.249×`、004 `2.833×`；
- 分离 dynamic instance layers 后，base unknown 比例仍为 `97.10% / 96.04% / 97.57%`；
- source actor removal 后原体积恢复 UNKNOWN，不会恢复 FREE；edited layer 可独立 remove/insert，三场景未出现
  layer overlap；
- 缺少 nuScenes map-expansion polygons 时 road-support 与 off-road control 保持 UNKNOWN。

**边界**

11B 已消除 AABB 作为正式动态几何和扁平 layer 删除污染，但没有降低 observation sparsity。30 条可测真实
controls 的 retention 为 100%，collision/teleport 可检测负例为 2/2；然而加入 road-support 后 32 条完整
certificate 全为 UNKNOWN。这是诚实 abstention，不是 H1-CERT pass。

**后续约束**

D1 必须报告 precision、recall、abstention 和 PASS coverage；UNKNOWN 不进入 TP/FP/FN。只有独立观测或 map
证据能把 UNKNOWN 变为可判定状态，禁止通过调大 unknown threshold、把 box 当 background surface 或用 Gaussian
floaters 补 safety evidence。

### V7-RISK-15/16：certificate/projector 与 scenario effect 必须继续拆分

11B 已冻结 `scenario-effect-v1` 的纯 3D 0→1/0→0 gate、same-actor pair schema 和
`certificate-calibration-v1` 三态接口。11D 必须让 D1 逐字节复用 C trajectory，D2 才允许修改轨迹；位移 proposal
若未形成冻结的 corridor crossing、duration、gap 与 TTC/headway 条件，只能标为 non-event，不能靠命名成为
cut-in/merge positive。

### V7-RISK-17：typed depth 语义混淆已缓解

11C 把 depth 冻结为三个不同产品：diagnostic expected depth、T1 Gaussian first-hit depth、T0 LiDAR measured
depth；每个产品有独立 validity、definition、truth tier 与 artifact sidecar。独立审计确认三类各 18 个，且没有
expected-as-measured 混写。后续 export/evaluator 必须继续按产品名和 truth tier 消费，不能重新折叠成无类型
`depth`。

### V7-H1-11D：H1-CERT / H1-PROJ 预注册拒绝

**冻结事实**

- source-only eligibility 覆盖 3 scenes × 2 actors，P1–P5 共 30 proposals；S1 未删除；
- C/D1 realized trajectory hash 30/30 完全相同；
- D1：precision `0.75`、recall `0.8824`、abstention `0.3333`、PASS coverage `0`；
- C external hard violation `17/30`；
- D2：0/30 export、0 usable yield，external rate 不可定义；
- scenario-effect：0 positive、25 negative、5 source-positive/non-event，0 same-actor pair。

**裁决**

H1-CERT 因 precision 低于 `0.80` 拒绝；H1-PROJ 因拒绝全部 proposal、无 comparable export、usable yield
低于 `70%` 拒绝。按路线转向规则停止 OccGS 方法 claim，只保留 object-centric GS、WorldState、typed label、
certificate/evaluator 与 run-contract 基础设施。

**唯一修复与防重复**

首版聚合把 rejection 计成零违规，已作为 `metric_aggregation_bug` 唯一修复，旧 aggregate 保留。修复未改变
方法输出；第二版对无 export 的 rate fail closed。不得继续：

- 调低 known-evidence/coverage 门槛把 UNKNOWN 改成 PASS；
- 删除 005/S1 或 004 actor 8；
- 根据现有结果重选 actor、方向、proposal 或 event threshold；
- 用固定-pool `0/30 violation` 隐藏 D2 的 `30/30 reject`；
- 因 recall 达标而隐藏 precision fail，或把 UNKNOWN 排除后重算；
- 在当前配方上继续 H2/H3/scale。

## 4. 跨路线必须保留的原则

1. 先证明监督/比较对象存在，再训练或扩量。
2. occupancy、编辑、渲染和标签必须共享同一显式状态，不允许旁路文档绑定。
3. matched baseline 使用相同 proposal、scene、actor、幅度、seed 与预算。
4. top-k 只用于诊断，不替代全分布、coverage 与 worst-case。
5. machine pass 只解锁下一门禁，不自动成为 human verdict、论文 claim 或 scale 授权。
6. hard composition 的局部性与 completion 的质量是两个独立门禁。
7. 下游效用必须由任务指标证明，不能由约束 accept rate、RGB 差分或 PSNR 代替。
8. 工程失败与 research reject 分开登记；既有 provenance 缺失必须诚实标记。
9. 失败范围不能过度外推，但也不能通过改名、放宽阈值或只挑成功场景重复旧问题。

## 5. 新实验防重复检查表

- [ ] 是否明确引用了相关 `RF-*` 与 `V7-RISK-*`？
- [ ] occupancy 是否真正进入决策/状态链，而非只在磁盘上存在？
- [ ] baseline 是否 matched，而非故意构造的极端负例？
- [ ] primary endpoint 是否避免“方法规则自己定义方法成功”的循环论证？
- [ ] 是否同时报告全分布、coverage、per-scene 与 worst case？
- [ ] completion 是否测 inside quality/temporal/depth，而非只测 outside exact？
- [ ] human verdict 是否只由用户/指定评审者填写？
- [ ] run 是否有唯一 ID、resolved config、fingerprint、metrics、summary 与终态标记？
- [ ] 哪个单卡门禁失败时停止，什么条件才允许 scale？
