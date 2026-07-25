# Post-OccGS 下一研究方向：event-first、map-and-raw-evidence 反事实路线

> **调研日期**：2026-07-24；2026-07-25 按第二次 reject 与第三版 formal 结果更新
> **状态**：N0 pass；mini N1 reject；第二版 full-domain 人审 `REJECTED`；第三版 kinematics-first
> formal 为 `AWAITING_HUMAN_REVIEW`，但 machine support 已因 negative/pair 不足失败。
> **输入事实**：V7.1 H1-CERT/H1-PROJ 均 rejected；30 个 proposal 中 0 positive、D2 0 export。
> **首选方向**：先证明事件存在，再建立独立几何证据，最后才生成、渲染和检验效用。

## 0. 执行结果与路线更新

- `N0-ASSET-01`：`COMPLETE`。map-expansion v1.3、四图 layer、scene→map、pose contract 与 hash 通过；
- `N1-EVENT-01`（mini）：`REJECTED / reject_mini_event_pool`。45 eligible actors、71 stable transitions、
  22 topology-pass、0 interaction positive、0 same-actor pair；
- `N1-EVENT-FULL-01`（val 146）机器层：37 positive、7 negative/pair、17 scenes；后续 37/37 人审只有
  2 TP / 35 FP，独立 `N1-EVENT-FULL-AUDIT-01` 唯一 `REJECTED`。核心错误是把 target 多 incoming
  当作 subject merge，而非 graph gap 数值不足；
- `N1-EVENT-KINEMATIC-01`（official train 694）：原始 2 Hz subject kinematics + branch-safe temporal
  interaction 得到 12 candidates / 9 scenes，但 negative=2、same-actor pair=2，machine support 未通过；
  12/12 人审材料已就绪，唯一 `AWAITING_HUMAN_REVIEW`。详见
  [`N1_KINEMATIC_EVENT_POOL_REPORT.md`](N1_KINEMATIC_EVENT_POOL_REPORT.md)；
- N2–N5：`locked / forbidden`。旧 N2 名单已失效；第三次审核和用户独立裁决前不得下载、抽取或执行。

第三版说明“先运动学、后交互”能把旧 35/35 已知 FP 拒绝，并在未见标签的 train split 找到 12 个更窄
候选；它尚未证明候选真实，且没有满足 matched-control 支持。现在的首要问题已经从“地图能否找到候选”
转成两个分开的验证对象：**merge authenticity** 与 **long-horizon comparable control availability**。

## 1. 研究问题重新定义

旧路线把问题写成“occupancy 能否修复 object-centric GS actor edit”。H1-11D 表明更上游的问题尚未解决：

1. 被编辑的 actor/proposal 是否形成目标事件？
2. 是否有独立于方法自身的证据判定合法性？
3. 通过安全闸门后，是否仍有足够 usable yield？
4. 只有前三项成立，渲染质量和下游效用才有可比较对象。

因此新路线的核心对象不是 GS，也不是粗 voxel occupancy，而是：

> **给定真实 log、矢量 lane topology 与 raw sensor evidence，能否先冻结一组确实存在、可配对、可判定的
> 交通事件，再在固定预算下生成独立合法且可渲染的 counterfactual？**

这一定义保留了已有的渲染和标签工程资产，但切断了“renderer 自证安全”和“没有事件也继续训练”的循环。

## 2. 本机可行性审计

2026-07-24 的只读结果：

| 资产 | 本机状态 | 影响 |
|---|---|---|
| nuScenes trainval metadata/annotations | 已有，`v1.0-trainval` 约 2.5G；已完成 694-scene annotation-only formal | 可继续只读 N1 诊断 |
| nuScenes sensors | full CAM_FRONT/LIDAR_TOP samples 在盘；其余相机主要为 mini，sweeps 约 4.7G 且覆盖未审 | 当前未授权用于 N2；存在不等于可执行 |
| nuScenes raster maps | 4 个 PNG | 可显示，不足以可靠查询 lane connectivity/polygon |
| nuScenes map expansion vector JSON | v1.3 已安装并通过 N0 | lane graph、drivable polygon 可查询 |
| Waymo / nuPlan 数据 | 缺失 | 不能直接使用 Waymax/nuPlan 做正式 scenario mining |
| DriveStudio adapters | 有 nuPlan/Waymo adapter 代码 | 仅说明接口可能复用，不说明数据或许可已就绪 |
| 磁盘 | `/root/autodl-tmp` 可用约 63G | 小型 audit/diagnostic 可行；大型数据需独立预算与授权 |

只读 N0 进一步确认了 processed-scene 映射与冻结 actor 的轨迹上限：003/004 属于
`boston-seaport`，005 属于 `singapore-queenstown`；6 actors 中有 3 个完整 track 位移不足 1 m。
详见 [`N0_ASSET_AND_EVENT_PREFLIGHT.md`](N0_ASSET_AND_EVENT_PREFLIGHT.md)。

官方 nuScenes prediction API 提供 closest lane、incoming/outgoing lane 和 lane discretization，说明 lane graph
是现成且可审计的 proposal/event 基础，而不是必须重新从 raster 猜测
([nuScenes prediction tutorial](https://www.nuscenes.org/tutorials/prediction_tutorial.html))。

## 3. 路线排序

| 优先级 | 路线 | 解决的核心失败 | 本机代价 | 裁决 |
|---|---|---|---|---|
| A | event-first + vector map + motion-compensated raw sweeps | 真实性、pair、UNKNOWN、coarse voxel FP | map 已就绪；raw 阶段仍需授权 | **停在 N1 第三次人审；N2 locked** |
| B | Waymo/nuPlan/ScenarioNet 的长日志 scenario mining + closed-loop | nuScenes 20 秒内 same-actor control 不足 | 高；需数据、许可、存储 | 第三次人审后条件性后备 |
| C | uncertainty-calibrated learned occupancy | raw LiDAR 稀疏导致 coverage 低 | 高；需训练、校准集 | 仅探索，不作真值 |
| D | 直接用 diffusion/generative traffic 造事件 | proposal 多样性 | 高；评价更难 | 暂不启动 |

## 4. 首选路线 A：分阶段预注册

### N0-ASSET：最小资产与 provenance

目标：只补足 lane topology 所需的官方 vector map，不引入大数据集或模型权重。

必须冻结：

- 资产来源、许可、下载时间、文件清单与 SHA256；
- nuScenes version、scene→log→map-name 映射；
- 只读原始资产与派生缓存分离；
- coordinate contract 延续 V7.1 的 `T_dst_src` 命名和 round-trip fixture。

通过条件：

1. 四张地图均可由 `NuScenesMap` 加载；
2. lane/lane_connector、drivable_area、road_segment 查询可重复；
3. 003/004/005 的 scene→map 映射唯一；
4. 不通过 raster image processing 反推 lane graph。

停止条件：官方资产或许可无法取得、文件 hash 不稳定、scene→map 映射不唯一。

### N1-EVENT：先证明事件存在

目标：在任何新编辑、渲染或训练前建立冻结事件池。

建议从真实轨迹和 lane graph 挖掘：

- lane change / cut-in / merge；
- actor 的 source lane、target lane、lane boundary crossing time；
- target-lane front/rear actor、gap、headway、TTC；
- 事件前后持续时间、可见性、annotation 连续性；
- 事件级 `positive / negative / non-event / unknown`，unknown 不得后验并入 positive。

最小事件定义必须同时包含拓扑、运动和交互：

1. source/target lane 不同且图上可达；
2. actor center 或 oriented box 跨越冻结 corridor；
3. crossing 在冻结时窗内持续足够帧；
4. 对 cut-in/merge，目标 lane 上存在明确 front/rear 关系；
5. 不以位移幅度、RGB 差分或文件名代替事件。

建议 pilot gate：

- source-only eligibility，先冻结 actor pool；
- 每 scene 至少 2 个可用 actor，否则该 scene 不进入配对评估；
- 总池必须同时包含 positive 和 negative；
- 至少形成预先规定数量的 same-actor `0→1` 与 `0→0` 对；
- 若 mini 无法提供，则结论为 `reject_mini_event_pool`，转向路线 B，而不是后验挑 actor。

### N2-EVIDENCE：独立 raw/map evaluator

目标：解决 coarse `0.4m` voxel overlap 的 representation mismatch 与 96–98% UNKNOWN。

建议证据链：

- 对多 sweep LiDAR 做 ego-motion compensation；
- 用 annotation track 对动态 actor 做逐 sweep compensation 或显式排除；
- source actor points/box 与静态环境分层；
- 用 point-to-OBB、continuous signed distance 或 swept-volume distance，避免单个粗 voxel 接触直接等同碰撞；
- 用 map polygon/lane topology 判定 drivable/lane support；
- raw points、map、annotation 和 learned evidence 分开记录 truth tier；
- 继续三态 `PASS/FAIL/UNKNOWN`，但以独立观测增加 coverage，而不是放宽 unknown threshold。

校准池必须与评估池 scene-disjoint；冻结后报告：

- precision、recall、specificity；
- PASS/FAIL coverage 与 abstention；
- per-scene、per-actor、worst-case；
- 与单 sweep/coarse voxel 的 matched comparison；
- error taxonomy：量化接触、动态残影、遮挡、地图边界、标注误差。

注意：H1-11D 的 5 个 FP 全来自 004 actor 8，certificate 报 5 个 static overlap voxels，而 raw LiDAR
为 0 points。这与离散化或证据层不一致相符，但还不能证明唯一因果机制。N2 必须用冻结对照实验验证，
不得事后把“0 raw points”直接设成 PASS。

### N3-PROPOSAL：lane-reachable 候选而非固定横移

建议以 target state / reachable set 生成 proposal：

- target lane/corridor 来自 lane graph；
- 横向过程使用平滑曲率、速度、加速度和 yaw-rate 约束；
- interaction 约束来自 target-lane front/rear gaps；
- 每 actor 固定候选预算和 tie-break；
- 生成器不能看到评估池的外部标签；
- D1 检测复用未修改轨迹，D2 才允许 projection/repair。

N3 的首要 endpoint 不是 RGB 或 accept rate，而是：

1. scenario-effect positive rate；
2. same-actor pair availability；
3. independent-evidence coverage；
4. D2 usable yield；
5. comparable export 数量。

只要 usable yield 为 0，外部违例率仍为不可定义，不能写成 0。

### N4-RENDER 与 N5-UTILITY

只有 N1–N3 通过后，才允许：

- 复用 GS/WorldState/typed-label 导出 RGB、depth、semantic、instance、box；
- 独立 human review 仅评视觉/语义一致性，不替代安全 evaluator；
- scene-disjoint 的 `R / R+matched naive / R+new method`；
- 至少 3 seeds、相同样本量和训练预算；
- 以 detector/event/forecasting 任务指标检验效用。

GS 在此处是 renderer，不是安全证书。completion 的 outside-mask invariance 与 inside quality 继续分开。

## 5. 条件性路线 B：换事件数据底座

mini、val 人审和 official train kinematics-first 已依次表明：同域扩容能找到候选，但真实性和 same-actor
control 支持不能由 scene 数量自动解决。当前 full-domain 已执行，不能再把“扩到 full nuScenes”列为未做动作。

- nuScenes 官方数据包含 1,000 个约 20 秒 scenes；本项目已分别使用 val 146 和 train 694 做第二/三版
  evaluation。第三版 10/12 positive actors 找不到等价 negative，说明约 20 秒 scene 与更短 actor track
  对 same-actor non-overlap control 是结构性限制，而不是继续扩同一 split 就会消失；
- annotation+map-only 事件挖掘已完成，不需要、也没有因此自动取得 N2 camera/LiDAR sweeps 使用权。

- nuPlan 官方基准包含 1,282 小时、4 城市，并专门挖掘常见/稀有 scenario，支持 closed-loop interaction
  ([nuPlan paper](https://arxiv.org/abs/2403.04133))。
- Waymo 的 motion 方向已经把 Scenario Generation、Sim Agents、Interaction Prediction 作为独立任务，
  说明“事件生成—联合行为—交互评价”比单 actor 固定横移更接近当前研究对象
  ([Waymo challenges](https://waymo.com/open/challenges/))。
- Waymax 提供 overlap、offroad、wrong-way、route-following、kinematic infeasibility 和 log divergence，
  以及多智能体 closed-loop；但需要申请 Waymo Open Motion Dataset 访问并完成认证
  ([Waymax](https://github.com/waymo-research/waymax))。
- ScenarioNet 统一 Waymo、nuScenes、Lyft 和 nuPlan scenario，并能在仿真中 replay/interaction，可作为
  跨数据适配层候选 ([ScenarioNet](https://arxiv.org/abs/2306.12241))。

路线 B 的启动条件：用户明确批准数据/许可/存储预算，并先做最小 shard 资产审计。不要先下载全量数据。

## 6. 探索路线 C：不确定性感知 occupancy

学习 occupancy 可能提升 sparse raw observation 的覆盖：

- UnO 从 LiDAR 自监督学习连续 4D occupancy，并报告跨任务迁移和较高 recall
  ([UnO](https://arxiv.org/abs/2406.08691))；
- α-OCC 使用分层 conformal prediction，强调 prediction-set coverage 和 uncertainty
  ([α-OCC](https://arxiv.org/abs/2406.11021))。

但它们只能作为额外证据层，不能同时充当方法输入和外部真值。若探索，必须：

- scene-disjoint calibration；
- raw/map evaluator 仍独立；
- 报告 coverage guarantee、prediction-set size 与 failure modes；
- 不把 learned occupied/free 覆盖到观测 UNKNOWN 后称为 ground truth；
- 先做小规模 offline calibration，不直接接入 D2。

## 7. 明确不做的方向

- 不复跑 P1–P5 固定横移并改名字；
- 不降低 known-fraction、coverage 或 precision 门槛；
- 不删除 005/S1、004 actor 8 或后验换 actor；
- 不把 multi-sweep 简单堆叠当 motion compensation；
- 不让 learned occupancy、GS Gaussian 或 RGB difference 自己验证自己；
- 不在没有 positive/same-actor pair 时训练 H3；
- 不以 0 export 的集合计算 0 violation；
- 不先上大型 diffusion/generative model，再回头定义事件和 evaluator；
- 不把 renderer 画面质量当作物理合法性证据。

## 8. 推荐的最小下一步

### 当前唯一动作

1. 用户按
   [`N1_KINEMATIC_HUMAN_REVIEW_PROMPT.md`](N1_KINEMATIC_HUMAN_REVIEW_PROMPT.md)
   完成 12/12 第三次盲审；
2. validator 只校验/汇总，最终 verdict 由用户确认并另建 adjudication run；
3. N2/N3/render/training 继续封闭。machine pair gate 已失败，人工不能把它覆盖为 true。

### 审核后可突破方向

1. **若 merge authenticity 低**：第三版 reject；冻结 train，换独立 calibration/evaluation 或引入
   lane-polygon/人工事件标签，不在同 split 调 branch angle；
2. **若 authenticity 高但 pair 仍不足**：优先换更长 trajectory/log 的事件底座，或新预注册
   matched-other-actor control。nuPlan 官方 devkit提供 lane change、interaction 等 scenario 类型与筛选，
   其 metrics 也按 lane/lane_connector occupancy、expert route 与 projected boxes 限定 TTC；
3. **若 front/rear 是主误差**：研究 oriented-box swept ordering 和 lane-polygon occupancy；不得改成
   单侧邻车回写当前双侧 gap-insertion 定义；
4. **parallel lane-change 独立诊断**：对本 run 的 63 个 physical lane-change 单独审核主体动作，
   区分真实换道与 map-match 假象；它是新 diagnostic，不补进当前 12 条；
5. **视觉 evidence 升级**：若 CAM_FRONT + annotation topdown 产生大量 `UNCERTAIN`，再请求六相机或
   raw sensor 资产授权；该授权不等于 N2 pass。

路线选择依据不是“哪个最容易凑够 4 pairs”，而是第三次人审暴露的主失败层。任何新方向都要新 task ID、
scene-disjoint split、预注册阈值与 fail-closed terminal。

## 9. 来源范围与证据等级

本调研优先使用官方文档、官方 devkit 和原始论文：

- [nuScenes prediction tutorial](https://www.nuscenes.org/tutorials/prediction_tutorial.html)
- [nuPlan benchmark paper](https://arxiv.org/abs/2403.04133)
- [nuPlan devkit](https://github.com/motional/nuplan-devkit)
- [Waymo challenges](https://waymo.com/open/challenges/)
- [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset)
- [Waymax](https://github.com/waymo-research/waymax)
- [ScenarioNet](https://arxiv.org/abs/2306.12241)
- [UnO](https://arxiv.org/abs/2406.08691)
- [α-OCC](https://arxiv.org/abs/2406.11021)

文献支持的是组件和研究设计的可行性，不直接证明 Motion-Proj 新路线会通过。所有 claim 仍须按 N0–N5
在本项目冻结数据与 evaluator 上裁决。
