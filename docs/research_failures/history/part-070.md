# 历史原始记录 070

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### N1-F23：共享 cgroup 的 start 合同是研究终止门，而非可绕过的工程告警

final formal 在 clean commit `7104f5c` 的 preflight 已将 runner 自身 RSS 降至 `20,705,280` bytes，仍记录
`cgroup_memory_current_bytes=1,523,929,088`，超过冻结上限 `1,350,000,000`。它在任何 evaluation scene 前
安全失败；独立裁决冻结证据并以 `REJECTED/stop_nuscenes_cutin_mining` 结束。development override 的 32/96
smoke、清页缓存或 K4 回归均不能替代正式 start 合同。禁止杀死 Cursor/Jupyter/TensorBoard 等用户服务、修改
正式阈值、截断正式 split（当时 expected 常量误写为 669，见 N1-F25）或把这次结果说成“nuScenes 没有
cut-in”；`n2_authorized=false` 保持不变。

**2026-07-26 用户复开授权**

上述 `REJECTED` 仍是 Resource Contract V1 下不可改写的历史裁决，但不再代表任务永久停滞。用户随后显式
扩大容器内存并授权继续本次 final：现场复核 `memory.max=128,849,018,880` bytes（120 GiB），原
`2,147,483,648` bytes（2 GiB）资源前提已改变。因此必须保留失败 parent 与独立拒绝裁决，同时使用新的
Resource Contract V2、全新 config fingerprint 和不可复用 run ID 恢复 scene-disjoint formal
（经 N1-F25 确认为 675 scenes）；不得覆盖或续写
V1 失败目录，也不得把 V2 成功倒写成 V1 当时没有失败。

### N1-F24：内存不足时停止并等待资源授权，不继续死磕

**新执行规则**

1. 任何正式或开发任务若触发启动/运行 stop 阈值、`RC=137`、SIGKILL，或观察到持续逼近 cgroup
   `memory.max`，立即停止启动新 batch，并尽最大可能写入结构化 `FAILED/failure.json`、最后完成 scene、
   process RSS、cgroup current、anon/file cache 与 `memory.events`；
2. 不通过反复重跑、杀死 Cursor/Jupyter/TensorBoard 等用户服务、缩短正式 split、降低证据质量、跳过
   audit、修改研究阈值或清理不属于本任务的缓存来争抢资源；
3. 把失败点、最低所需资源和恢复命令回报用户，然后等待用户开放资源；没有新的明确授权时不得自行恢复；
4. 用户开放资源后，先记录新的 `memory.max/current` 和授权时间，版本化 resource contract，使用新 run ID
   从冻结研究配置重新运行。资源合同变化只允许调整资源阈值，不允许调整 cut-in taxonomy、hard gate、
   calibration/evaluation split、抽样或人工聚合门槛；
5. 资源暂停与研究拒绝分开登记。未读取 prospective evaluation scene 的资源失败不能被写成方法精度失败，
   后续在新授权下完成的结果也不能删除或覆盖先前工程失败证据。

### N1-F25：final 的 669-scene 预期是 split 算术错误，不是 evaluation 集合定义

Resource Contract V2 的首次 clean formal run：
`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-01/v71_n1-event-cutin-final-01__receiver-cutin-final-v1__s0__20260726T142634031503Z__5c8c65d7`
在 K4 regression 通过后、任何 evaluation scene 或 candidate 读取前 fail closed，错误为
`final evaluation scene 数不匹配: 675 != 669`。

独立复算表明：nuScenes official `train` 为 700 scenes；冻结的 42 个 calibration scenes 中，25 个属于
`train`、17 个属于 `val`，且没有 split 外 scene。因此 scene-disjoint evaluation 的确定数量是
`700 - 25 = 675`。`_resolve_evaluation_scenes` 已正确执行
`set(train) - set(all_calibration_scenes)`；错误只在 YAML 的 `expected_scene_count` 常量把不属于 train 的
calibration scene 也错误计入了减法。

合法修复仅为把 Resource Contract V2 配置中的 assertion 从 669 改为 675，并生成新 config fingerprint、
新 clean commit 和新 run ID。不得借此增删 calibration scene、显式挑选 evaluation scene、查看 candidate 后
改 split，或修改 taxonomy、strict gate、K4、抽样与人工门槛。失败 run 必须保留为工程契约失败，不能统计成
research reject。

### N1-F26：strict v2 在 675 scenes 上只有 1 个 PASS，不能靠人审单例或放宽规则扩池

Resource Contract V2 的 clean formal run：
`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-01/v71_n1-event-cutin-final-01__receiver-cutin-final-v1__s0__20260726T142941598714Z__883fae9a`
在 commit `beee1de`、seed 0、config fingerprint
`883fae9a6514c0bff5bba8bcaf81a22c79e6d719586221596a7d4b5364c337da` 上完成 675/675 scenes。

结果为 `ABSTAIN=1,556`、`FAIL=200`、`PASS=1`，唯一 PASS 只覆盖 1 scene；冻结 machine-readiness 要求
至少 3 candidates / 3 scenes，故 parent 以
`REJECTED / stop_nuscenes_cutin_mining_too_sparse` 结束。K4、raw-only 和资源合同检查均通过，峰值 batch
process RSS 为 `337,154,048` bytes、cgroup current 为 `4,556,898,304` bytes；这次拒绝不再是资源失败。

独立稀疏终局人工包保留唯一 PASS 和 3 条 diagnostic，但人工结果不能改变数量门失败：即使唯一 PASS 被判为
TP，仍只有 1 TP / 1 scene，低于 sparse 的 3/3。禁止为了形成池而把 ABSTAIN 提升为 PASS、恢复
`receiver_branch_merge`、放宽 raw/parallel/receiver 时序门、事后改 scene 或把单例人工真实性外推为总体
precision。准确结论是“当前冻结 strict v2 的 prospective pool 过稀”，不是“nuScenes 没有 cut-in”。

<a id="detail-cross-route"></a>

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

## 6. 2026-07-26 路线转向新增失败与防重复项

### PIVOT-F01：nuScenes cut-in 没有可验证的召回率分母

**观察**

nuScenes 官方公开的是场景、样本、对象实例、类别、属性、3D 框、传感器与地图等结构；`scene.description`
是自由文本，不是事件级 cut-in 真值。官方没有发布 cut-in 场景占比、逐事件标签或可直接用于召回率计算的全集分母。
四轮挖掘和最终 675-scene prospective run 最多只能测量“冻结规则产出的候选质量”，不能测量“数据集中所有
cut-in 被召回了多少”。

**最终证据**

- `N1-F26`：strict v2 在 675 个 scene 上只有 `1 PASS / 1 scene`；
- 最终人工稀疏包即使把唯一 PASS 判为真阳性，也仍低于预注册的 `3 candidates / 3 scenes`；
- 该结果不能外推为“nuScenes 没有 cut-in”，也不能用事后放宽规则伪造召回。

**裁决**

`cut-in mining` 状态固定为 `rejected / frozen`。以后 cut-in 只允许作为已经具备重建与编辑能力后的可选演示，
不再承担数据集入口、方法定义、训练前置条件或论文成立条件。

**解除条件**

只有新的、独立的数据源提供事件级真值及明确分母，或新的任务本身不需要宣称事件召回率，才允许创建全新任务 ID
重新讨论；不得恢复当前 strict-v2 阈值调参。

### PIVOT-F02：贡献漂移——工程系统吞噬了重建与编辑研究

**观察**

过去路线的主要投入逐步变成事件挖掘、地图匹配、接收车身份、规则校准、候选审核与资源合同。它们改善了审计性，
却没有自然回答动态对象几何、连续运动表示、遮挡/去遮挡、反事实轨迹编辑或下游感知一致性。

**边界**

这不否定已形成的 WorldState、typed label、run contract、审计与人工审核基础设施；它只否定把“更好的 cut-in
挖掘器”作为 3DGS/4D 重建论文的核心贡献。

**后续约束**

新路线必须先复现公开强基线，再通过重建/编辑压力测试选择创新点。数据工程模块只能服务于冻结实验，不得重新成为
论文主任务。每个里程碑都必须说明它直接回答的重建或编辑问题。

### PIVOT-F03：未完成 exact reproduction 前禁止集成式“改进”

**观察**

`RF-05/06/08/09/16/18` 与 `V7-RISK-03/04/06/07/10/16/17` 共同表明：输入、状态、比较对象、覆盖率和真值定义
未冻结时，模块堆叠会把工程可运行误当成方法收益。AD-GS 的公开 nuScenes 协议提供了固定 scene、帧区间、预处理、
训练和评测入口，适合作为新的事实锚点。

**禁令**

在 AD-GS exact reproduction 门禁通过前，不得：

- 合并 Motion-Proj/StreetGS/OccGS 模块；
- 加 occupancy、物理约束、扩散补全、感知损失或轨迹编辑；
- 更换为自选事件场景、调低分辨率后对齐论文指标或只展示成功帧；
- 把兼容性补丁、预处理修复或运行成功表述为方法改进。

任何 unavoidable compatibility patch 必须独立提交、最小化、附 upstream diff 与消融；原始基线结果必须保留。

