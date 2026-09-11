# 历史原始记录 075

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V6-F34：删除式可见性筛选不能补齐远路线新暴露表面，且 5m 深度代理失去米制有效性

H-R13-004 canonical rejected run `20260821T121956Z__worldspace-visibility-s20260821-r1` 使用冻结三相机
logged LiDAR 在每个目标视角执行 4 像素/0.30 相对深度筛选。它保持共同 lateral route `3.0m`、精确复跑、源不可变和
V6 false-safe `0.0`，但两帧 5m 几何 MRE 仍为 `9.0323/9.4071`，photo MAE 为 `0.1404/0.1495`，
两帧均失败；5m 仍保留约 `78%` 的旧点，说明仅删除矛盾点并未提供新暴露表面。

独立诊断还显示，5m 目标视角投影后的实测 LiDAR 中位 z 约 `13.17/13.34m`，而 StreetGS 目标 depth proxy 中位数仅
`1.30/1.22`，该 proxy 在此外推距离不能作为米制几何真值。不得扫描 visibility 阈值或放宽 photo/geometry gate 来追逐
这个失效代理。路线偏移结论固定为 H-R13-002 已验证的 lateral `3m`、forward `2m`；后续直接进入计划尚未覆盖的 actor
add/remove、trajectory modification 与 traffic-density typed edit 实验。

### V6-F35：跨 run 回放内容等价比较必须排除非语义 repeat 序号

H-R13-005 首个正式 run `20260821T122830Z__dynamic-edits-s20260821-r1` 的三个 V6 typed edit 均通过全部
编辑、依赖闭包、时序和精确复跑检查，但总 gate 因 `base_matches_frozen_r12_replay=false` 保持 `rejected`。逐字段定位确认
唯一差异是当前重新加载调用使用 `repeat_index=0`，而冻结文件 `DYNAMIC_REPLAY_REPEAT1.json` 记录 `repeat_index=1`；
`replay_content_sha256` 及 actor、trajectory、semantic、collision、sensor、event 全部内容一致。

修复只在跨 run 内容等价比较的两侧移除非语义 `repeat_index`，仍严格比较冻结内容 hash 和所有功能字段；不改三个 edit、
四方法臂、actor/时间戳分母、碰撞计算、false-safe、资源合同或任何阈值。repeat 序号继续保留在各自运行记录内，但不得被当作
compiled-world 内容漂移。

### V6-F36：浮点 renderer RGB 归一化必须容忍轻微大于 1 的辐射 overshoot

H-R13-006 首个正式 run `20260821T123716Z__actor-sensor-perception-s20260821-r1` 完成 16 次 DeepLab
推理且精确复跑，但 AD-GS 两个 case 的 target RGB effect 被错误压到约 `0.00035`，继而使感知变化近零。StreetGS 两例保持
约 `0.09` target effect。根因是 AD-GS 的归一化浮点 RGB 存在轻微 `>1` overshoot，初版主 runner 把它误判为 0--255
后再除 255；worker 同样没有把它放大到 uint8 动态范围。该 run 的 AD-GS 指标无效，不能作为方法 rejection。

修复把浮点 renderer 合同明确为最大值 `<=2.0` 时仍按归一化辐射值处理：主 runner 直接 clip 到 `[0,1]`，perception
worker 乘 255 后 round/clip 为 uint8；只有明显大于 2 的数组才按 0--255 输入。冻结 render 字节、模型、case、mask、
阈值、repeat、资源和 gate 全部不变。以后不同 frontend 的 RGB 必须由显式 range contract 归一化，禁止用严格 `max<=1`
启发式区分编码。

### V6-F37：局部 RGB 编辑不保证全帧感知输出局部，宽感受野必须进入 verifier 因子设计

H-R13-006 canonical rejected run `20260821T123835Z__actor-sensor-perception-s20260821-r1` 在修复 RGB
range 后确认 StreetGS/AD-GS 两帧共 4 个 actor-remove case 均具有强 sensor locality：target RGB MAE
`0.0872--0.0954`、outside RGB MAE `0.000175--0.000399`、locality enrichment `224--498x`，16 次
DeepLab 推理精确复跑且峰值仅 `688MiB`。但全帧 DeepLab 在 target 外仍改变 `6.34%--10.31%` 标签，4 个 case
只有 AD-GS frame57 达到 2x perception locality，故假设按预注册 gate 正式 `rejected`。

不得把 outside 2% 或 enrichment 2x 阈值调松，也不得因 RGB 局部就宣称 perception failure 已解决。H-R13-007 改用
factorized ROI 机制：固定 256px tile/128px candidate stride，只根据 logged dynamic opacity 选择最高 actor fraction target
和与其不重叠的最低 actor fraction static tile，对两者独立执行冻结模型；full-frame rejection 永久保留，不被 ROI 结果覆盖。

### V6-F38：remove-all 编辑不能为 factorized perception 提供无 actor 的静态 ROI 对照

H-R13-007 canonical rejected run `20260821T124443Z__factorized-perception-s20260821-r1` 按冻结
256px tile/128px stride，在每个 frontend/frame 选择最高 actor fraction target 与不重叠的最低 actor fraction static。
四个 target 的 actor fraction 为 `0.753--0.936`，但四个所谓 static 仍为 `0.099--0.373`，全部超过预注册
`0.01` 上限；static RGB MAE 也为 `0.0095--0.0267`，证明 remove-all 操作本身横跨全图，而不是 tile 选择偶然失败。

不得扫描 tile 大小/stride 或放宽 static denominator。H-R13-008 改变真正的因果变量：利用 StreetGS 冻结 checkpoint 的
per-Gaussian `point_ids` 与 `instances_fv`，只删除在两帧均可见且 Gaussian 数最多的单个 model actor；actor 选择不读取
RGB/semantic outcome。冻结 AD-GS 不保留可审计 per-actor ID，必须 ABSTAIN，不得伪造跨 frontend 单 actor 对齐。

### V6-F39：actor Gaussian 数量不等于下游感知敏感度，单启发式选择必须扩展为完整分母

H-R13-008 canonical rejected run `20260821T125151Z__single-actor-perception-s20260821-r1` 从两帧均可见的
12 个 StreetGS model actor 中，按最大 Gaussian 数且最小 index 的预注册规则选中 index `2`（`13,490` Gaussians）。
logged rerender 对冻结 R3 RGB 的 MAE 为精确 `0`，单 actor 删除的 effect pixel 为 `10,271/9,047`，target RGB
MAE 为 `0.0210/0.0286`，outside RGB MAE 仅约 `1e-6`；但 target DeepLab label change 只有
`0.00068/0.0`，两帧都未达到冻结 2% 感知效应门。

该负结论拒绝“Gaussian 最多 actor 最能触发 perception”的启发式，不得改选第二大 actor 当作同一假设 recovery，也不得放宽
2% threshold。H-R13-009 一次性评估冻结 metadata 定义的全部 12 个 eligible actor，以固定完整分母报告 ACCEPT/ABSTAIN
覆盖率；只有两帧都通过完全相同门槛的 actor 才可被 V6 接受，其余必须显式 abstain。

## 7. 历史新路线启动前附加检查

- [ ] 是否明确说明该步骤直接服务于重建、编辑或可信评测，而不是重新做事件挖掘？
- [ ] AD-GS exact reproduction 是否已经通过冻结门禁？
- [ ] 是否把 upstream 原始结果与 compatibility patch 结果分开？
- [ ] 是否对 VAD-GS 等已公开的 visibility/completion 工作做 novelty 边界核对？
- [ ] 反事实无真值指标是否有真实 held-out/pseudo-hole 证据，而不是自洽规则？
- [ ] 是否同时评估目标区变化、非目标区保持、几何/时序一致性和下游感知？
- [ ] 遇到内存/GPU不足时是否按 `N1-F24/PIVOT-F05` 停机并等待授权？

## 8. WorldSim 后续正式消融前检查

- [ ] 是否使用 V3 task ID、新 run 和冻结 config/source hash，而不是续写 V2 terminal？
- [ ] 是否保持 scene-0230/0242/0255、split、seed、相机、步数和 actor cohort 不变？
- [ ] 是否把原生 Affine/CamPose/LiDAR init 与新增实现分开？
- [ ] rolling-shutter 路径是否有真实 row timing；没有时是否显式 `not_supported`？
- [ ] actor-aware 变化是否只增加一个可归因因子，并保留 module-off 原生等价测试？
- [ ] 是否同时报告 actor/boundary 质量、GS 数、训练时间、VRAM 和 non-target 保持？
- [ ] local refinement 是否冻结 affected set 外参数，并只使用 Tier-A/多视图/LiDAR 可观测证据？
- [ ] expected/first-hit/measured depth 是否继续保持 typed separation？
- [ ] 工程 `blocked`、方法负结果 `rejected` 和任务完成 `done` 是否没有混写？
- [ ] 结论是否明确限制在三场景消融，不写成大规模泛化或闭环安全结论？
- [ ] 新路线是否只选择一个 primary hypothesis，并说明它具体解除 `V3-F18`–`V3-F25` 中哪一项？
- [ ] 是否在任何训练、推理或新结果读取前冻结 matched baseline、主端点、资源门、停止条件和确认场景？
- [ ] 是否避免把更小剪枝 fraction、提高旧资源 ceiling、全量读取 chunk 或继续调 R1 配方伪装成新研究？

### V6-F40：可见 actor cohort 分母不得替代 SceneIR 全量 actor 分母

H-R13-010 canonical run `20260821T130810Z__sceneir-sensor-binding-s20260821-r1` 正确完成同一 scene-0242 checkpoint 的 SceneIR 编译、model index `0` 到 `actor_0000` / `streetgs_actor_0000` / `12,390` primitives 的绑定、actor remove、两次 fresh replay，以及未受影响 actor state、trajectory、semantic label 和 collision pair 的精确保持；继承的 H-R13-009 V6 false-safe 仍为 `0`。但是 preregistration 把 H-R13-009 中“两帧均可见”的 `12` 个 actor cohort 错当成 checkpoint 转换后的 SceneIR 全量 actor 数，实际冻结 converter 输出为 `27` 个 actor；因此预注册的 `15→14`、`2940→2744`、`20580→17836` 与实际 `27→26`、`5292→5096`、`68796→63700` 不符，typed dependency-closure gate 按约定拒绝。

该 run 保持 `rejected`，不能用其余检查通过来覆盖错误分母。恢复假设 H-R13-011 只把全量分母来源改为编辑前冻结 checkpoint 的 deterministic SceneIR converter 输出，并预注册上述实际总量；不修改 checkpoint、actor mapping、edit、replay、quality threshold、继承 verdict、资源合同或 unsupported claim。以后必须把 visibility/evaluation cohort 与 compiled-world total denominator 分别命名和冻结。

### V6-F41：下游 regression consumer 必须冻结上游 gate 的完整 decision 值

H-PT1-001 首个 formal run `20260821T132127Z__regression-utility-s20260821-r1` 在读取 scene-0230 development contract 时 fail-closed。上游 H-R13-005 gate 的实际 decision 是 `accept_typed_dynamic_edit_dependency_closure`，初版 consumer 却硬编码了缩写 `accept_typed_dynamic_edits`，因此在读取 heldout replay、构造四类 mutation 或计算三臂 quality metric 前即抛出 `PT1RegressionError`；该 run 只有 `TERMINAL.json`，不得产生方法结论。

修复把 scene-0230 与 scene-0242 两个上游 gate 的完整 accepted decision 值写入冻结 config，consumer 只按 config 精确比较。不得改变四类 stale-factor mutation、三方法臂、false-safe/detection gate、source hash、资源合同或 unsupported claim；修复后以新 run 重试 H-PT1-001。跨阶段消费者以后不得从 task 名或人类缩写猜测 structured gate value。

### V6-F42：policy 输入不得直接包含 verifier 的 decision statistic

H-PT2-001 canonical run `20260821T133158Z__risk-policy-s20260821-r1` 的数值 gate 全部为真，V6 arm 在 scene-0255 heldout 上达到 balanced accuracy `1.0`、false-safe `0`、safe-route completion `1.0`，并把 naive stale-label arm 的 false-safe 从 `1.0` 降为 `0`。但是 Real-only arm 同样达到 balanced accuracy `1.0` 和 false-safe `0`，尽管其 98 条训练行的 positive fraction 为 `0`。原因是 policy 直接接收 signed AABB clearance，而 hazard label 正是 `clearance<=0`；这等于把 verifier 判定边界编码进输入，Real-only 只需把阈值放在最小 logged clearance 以下就会偶然分开固定 synthetic offsets。

因此该 run 只保留“naive stale labels 有害”的诊断，不晋级为 incremental post-training utility；初版 gate 缺少对 Real-only 的增益约束也是方法治理缺口。H-PT2-002 保持三场景、frame 分母、clone offsets、label、heldout 与任务指标不变，移除 signed clearance/AABB extent/hazard verdict，只向 policy 提供原始绝对 ego-relative forward/lateral position；固定 axis-aligned rectangle ERM 候选网格，并新增相对 Real-only 与 naive 两者 false-safe 至少降低 `0.50` 的 gate。以后任何 learned verifier/policy 实验必须审计 feature 是否直接重编码 label rule。

### V6-F43：声明值经坐标变换后必须做远小于物理阈值间隔的数值 canonicalization

H-PT2-002 canonical run `20260821T133626Z__risk-policy-s20260821-r1` 在移除 signed-clearance feature leakage 后，使 Real-only 与 naive arm 都成为 constant CONTINUE，heldout false-safe 均为 `1.0`；V6 raw-position rectangle policy 把 false-safe 降至 `0.1122449`，safe-route completion 保持 `1.0`，但 balanced accuracy `0.9438776` 与 false-safe 仍未达到冻结的 `0.95` / `0.05` 门，故方法正式 `rejected`。

逐行诊断确认 11 个漏检全部是配置声明的 `3.0m` clone：`inv(T_ego) @ (T_ego @ T_offset)` 后 raw forward feature 变成 `3.00000000000011–3.00000000000045`，严格 `<=3.0` 比较失败；没有其他 hazard 漏检，也没有 false brake。H-PT2-003 只在 policy raw feature 写入前按 9 位小数 canonicalize（`1e-9m`），label 仍按未取整 signed clearance 计算；该尺度高于浮点漂移、但比最小候选阈值间隔小至少九个数量级。不得改样本、offset、threshold grid、label 或质量门。以后由声明变换生成的控制量必须把表示 canonicalization 与方法容差分开冻结。

### V6-F44：单一 zero-lateral intervention 训练不能支持二维风险泛化

H-PT3-001 canonical run `20260821T134426Z__intervention-robustness-s20260821-r1` 冻结 H-PT2-003 的全部 policy 参数，在未参与 PT2 训练或选择的 scene-0048 上评估 forward `1.5/4.5/7.5m` × nonzero lateral `0.75/1.5/2.5m` 的 441 个 edit rows，加 49 个 clean rows。分母含 232 hazards / 258 safe；冻结 V6 policy 的 lateral threshold 为 `0.0m`，因此对 232 hazards 检出 `0`，false-safe `1.0`，与 Real-only/naive constant policies 完全相同。source immutability、repeat exact、safe-route completion 均通过，所以是方法性 `rejected`。

不得把 PT2 的跨场景同 intervention pass 写成二维 policy generalization，也不得只把冻结 lateral threshold 放宽。H-PT3-002 改变训练 evidence：在 scene-0230/0242 使用 forward `0/2/4/6/8m` × lateral `0/1/2/3m` 的完整 factorized typed-edit grid，V6 重算标签、naive 保留 stale labels；scene-0048 使用与训练离散的 half-offset grid，所有质量门不变。以后 policy coverage 必须按 intervention factor denominator 报告，不能只报 scene denominator。

### V6-F45：最近 actor 的位置不足以表达 box overlap，必须保留 factorized extent

H-PT3-002 canonical run `20260821T134843Z__factorized-policy-training-s20260821-r1` 在 scene-0230/0242 完成 1,960 条二维 typed clone train rows，并在 scene-0048 的 980 条离散 half-offset edits 上评估。V6 相对两基线把 false-safe 从 `1.0` 降至 `0.4505208`、safe-route completion 保持 `1.0`，但 balanced accuracy 仅 `0.7747396`，未过冻结门，正式 `rejected`。

诊断显示 position-only policy 为每条 episode 只保留 signed-clearance 最小 actor 的 `|x|/|y|`，却丢掉 projected box half-extent 与 yaw；远处 safe clone 会由更近但窄或旋转的真实 actor 取代特征，同一位置因此对应不同 overlap label。训练 lateral `2.0m` 恰在默认 box 边界，未 canonicalize 的 factor label 还出现 66/32 等浮点混合。H-PT3-003 保持数据、grid、三臂、heldout 和质量门，新增 raw projected half-extents，按 1e-9m canonicalize forward/lateral factor labels，分别训练两个 logistic overlap heads 后 AND；不得输入 signed clearance 或最终 hazard verdict。以后几何 policy 的 factor representation 必须保留决定接触边界的尺寸/朝向信息。

### V6-F46：单一 synthetic box size/yaw 无法识别可迁移的 extent 系数

H-PT3-003 canonical run `20260821T135421Z__factorized-policy-training-s20260821-r1` 使用 raw `|x|/|y|` 与 projected half-extents，分别训练 forward/lateral logistic overlap heads。V6 train overall balanced accuracy 为 `1.0`，forward/lateral factor train accuracy 为 `1.0/0.9986`，但在 disjoint scene-0048 grid 上 balanced accuracy `0.7963`、false-safe `0.22135`、safe-route completion `0.81395`，未过冻结门，正式 `rejected`；两基线 false-safe 均为 `1.0`。

这不是优化未收敛，而是 synthetic train 全部采用默认 `4.5x2.0m`、relative yaw `0°`，projected extent 几乎常数，position 与 extent 的相反系数无法由 synthetic positives/negatives 识别，只能依赖稀疏真实 box 分布，换场景即漂移。H-PT3-004 保留 factor-head 架构、loss、task gates 与场景，扩展训练 denominator 为三种 size × 四种 yaw × 原 position grid；heldout 使用完全离散的三种 size × 三种 yaw。不得输入 signed gap 或固定解析碰撞公式来伪装 learning gain。以后 intervention coverage 必须同时报告 position、size 与 orientation factors。

### V6-F47：factor head 的训练目标必须与冻结的 balanced 指标对齐

H-PT3-004 canonical run `20260821T135942Z__factorized-policy-training-s20260821-r1` 在 23,520 条 multi-size/multi-yaw train interventions 与 8,820 条完全离散 heldout interventions 上运行。V6 把 false-safe 从 Real-only 的 `0.97133`、naive 的 `1.0` 降至 `0.14821`，但 balanced accuracy `0.85613`、safe-route completion `0.86047`，仍未通过冻结的 `0.90/0.10/0.90` 门，因此正式 `rejected`。

训练的 lateral factor 正例比例为 `0.88174`，原始 unweighted BCE 按出现频率主导梯度，与 task 从一开始冻结的 balanced accuracy/false-safe/completion 三目标不一致；这会同时留下 `14.82%` false-safe 与 `13.95%` false brake。H-PT3-005 保持场景、position/size/yaw 分母、raw feature、标签、三臂、步数和所有 gate 不变，只让每个 factor head 的正负类别在 BCE 梯度中各占一半。不得通过调决策阈值、改 heldout 或放宽门来恢复。

### V6-F48：类别平衡 BCE 不是可分 factor boundary 的充分机制

H-PT3-005 canonical run `20260821T140408Z__factorized-policy-training-s20260821-r1` 只把两个 logistic factor heads 改为正负类别总权重各半，其余数据、特征、三臂和 gate 均不变。V6 heldout balanced accuracy 降为 `0.82294`，false-safe 升为 `0.24746`，completion 为 `0.89334`；naive false-safe 也由 `1.0` 变为 `0.67863`，导致相对 naive 的 reduction 只有 `0.43116`。所有冻结质量门仍未通过，正式 `rejected`。

两个 V6 head 的 train balanced accuracy 仍只有 `0.97503/0.96014`，说明仅重加权有限步 smooth BCE 并没有形成稳定分离 margin；它改变了错误权衡，却没有消除训练边界错误。H-PT3-006 保持相同 denominator、raw feature 和原 gate，改用 deterministic class-balanced linear max-margin heads，并保留两 head AND。不得再通过类别权重或阈值扫描恢复。

### V6-F49：可分训练集上的最大间隔仍受离散 intervention 支持分辨率约束

H-PT3-006 canonical run `20260821T140847Z__factorized-policy-training-s20260821-r1` 使用相同 raw features 与 multi-size/multi-yaw 分母，将两个 factor heads 换为 class-balanced linear max-margin。V6 的 forward、lateral 与 joint train balanced accuracy 均为 `1.0`；heldout false-safe 降到 `0.05226`，相对 Real-only/naive 的 reduction 为 `0.66264/0.62637`，三项均过门。但 safe-route completion 只有 `0.84790`，balanced accuracy `0.89782`，仍未通过冻结门，正式 `rejected`。

原 train position 只有 forward `0/2/4/6/8` × lateral `0/1/2/3`，即使训练完全可分，最大间隔也可落在相邻采样位置之间；heldout 恰使用离散半步位置，暴露 `15.21%` false brake。H-PT3-007 只加密 train position denominator，新增位置全部与 heldout position set 离散，保持 size/yaw、heldout、features、SVM 和 gate 不变。不得调 SVM threshold 或删除 near-boundary heldout rows。

### V6-F50：set aggregation 下不能只用 false-safe 单轴增益比较会过度刹车的基线

H-PT6-001 canonical run `20260821T142626Z__compositional-risk-s20260821-r1` 在 scene0450 的 784 个双 clone episodes 上，用同一 frozen per-actor policy 对 logged actors 与两 clone 做 Boolean OR。V6 balanced accuracy/false-safe/completion 为 `1.0/0/1.0`；Real-only 为 `0.5/1.0/1.0`；naive 为 `0.52211/0.08844/0.13265`。V6 的全部绝对质量门通过，但相对 naive 的 false-safe reduction 最大只能是 `0.08844`，无法达到从 single-clone task 继承的 `0.50`，所以 H-PT6-001 仍正式 `rejected`。

naive 并未获得可用策略，而是 set OR 后以 `86.73%` false-brake 换取较低 false-safe。H-PT6-002 不追认旧 run，保持 frozen policy、scene、episode、label 和绝对门不变，预注册 paired Pareto gate：V6 对每个 baseline 的 false-safe 不更差、completion 不更差，并且 balanced accuracy 至少高 `0.20`。以后 actor-set policy 不得只用某一风险轴的下降评价会全刹车的 baseline。

### V6-F51：multi-actor development 的完美结果未在 one-shot confirmation 保持 false-safe

H-PT7-001 canonical run `20260821T143359Z__compositional-risk-confirmation-s20260821-r1` 在 attempt 先于质量读取、policy 与 H-PT6-002 gate 均冻结、scene0862 与八个 clone case tuples 全新的合同下，评估 784 个双 clone episodes。V6 balanced accuracy `0.92333`、safe-route completion `1.0`，且仍 Pareto 支配两基线；但 600 个 hazard 中漏掉 92 个，false-safe `0.15333`，超过冻结的 `0.10` 绝对门，因此 one-shot confirmation 正式 `rejected` 并消费。

不得用 scene0862 的漏检分布为该 multi-actor candidate 调 policy、case、threshold 或 gate，也不得用其余 Pareto checks 通过覆盖 false-safe failure。该 family 关闭。后续 H-PT8-001 来自此前一直显式保留的 `ABSTAIN_NO_LONGITUDINAL_CONTROLLER`，使用预注册 kinematic scenario grid 开始独立的 closed-loop utility family，不读取 PT7 badcase 生成参数。

### V6-F52：closed-loop collision 分母必须区分 policy 可避免与动力学不可避免，并审计 Real-only 监督

H-PT8-001 canonical run `20260821T143902Z__closed-loop-utility-s20260821-r1` 在 360 个静止单 actor、五秒、jerk-limited 纵向 scenarios 上，将三个 frozen policy arms 作为相同三秒 preview controller 的碰撞信号。V6 collision rate `0.25`、safe completion `0.9375`、comfort `1.0`、balanced accuracy `0.84375`，未过冻结门，正式 `rejected`；Real-only 与 V6 完全相同，naive balanced accuracy `0.69554`。

70 个 V6 collisions 按 6/8/10m/s 为 `3/22/45`，按 15/20/25/30/35m 初距为 `41/22/7/0/0`，并随 actor size 增大，说明原 denominator 把 t=0 已无法在同一 decel/jerk contract 下停车的场景也算成 policy false-safe。同时，Real-only equality 提醒后续必须审计它继承的 factor-label supervision；不得通过调 preview horizon 掩盖 baseline equality。H-PT8-002 只加入相同 dynamics 的 t=0 full-brake oracle，把 hazards 分为 avoidable/unavoidable，保留全部计数并只在 avoidable stratum 评价 policy collision；其余均不变。

### V6-F53：真实 box factor supervision 已解释静止 AABB preview，不能人为削弱 Real-only 制造增益

H-PT8-002 canonical run `20260821T144333Z__closed-loop-utility-s20260821-r1` 用同一 jerk/decel contract 的 t=0 full-brake oracle 将 280 个 uncontrolled hazards 分为 210 个 avoidable 与 70 个 unavoidable。V6 在 avoidable stratum collision `0`、safe completion `0.9375`、balanced accuracy `0.96875`，说明 H-PT8-001 的绝对 collision failure 来自可行性分母；但 Real-only 的三项指标完全相同，增量门仍失败，H-PT8-002 正式 `rejected`。

Real-only factor heads 虽无最终 clean collision positives，但合法读取了 logged box 的 forward/lateral overlap factor labels，已经足以学习静止 AABB 边界。不得删除这些真实监督、重命名 baseline 或只比较更弱 naive 来制造 V6 closed-loop gain；该静止单 actor family 关闭。H-R14-001 转回 core compiler 的既有 H-C 缺口，测试邻帧 temporal evidence-conditioned proposal，不由 PT8 结果选择参数。

