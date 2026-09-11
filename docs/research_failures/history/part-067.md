# 历史原始记录 067

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

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

