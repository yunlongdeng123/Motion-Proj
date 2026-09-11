# 历史原始记录 068

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

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

