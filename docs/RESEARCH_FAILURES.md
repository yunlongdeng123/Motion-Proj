# Motion-Proj 当前研究风险与防重复账本

> **最后更新**：2026-07-23
> **当前范围**：OccGS-Resim V7 retrospective evidence 与 V7.1。
> **历史账本**：完整 `RF-01`–`RF-18` 原文见
> [`archive/2026-07/v7-feasibility/RESEARCH_FAILURES_RF01_RF18.md`](archive/2026-07/v7-feasibility/RESEARCH_FAILURES_RF01_RF18.md)。
> **事实源**：[`EXPERIMENTS.md`](EXPERIMENTS.md) 和实际 run 产物。

本文件只保留对 V7 下一步仍有直接约束的历史结论，以及 V7 feasibility 暴露但尚未形成 research reject 的风险。
归档不会使旧失败失效；任何新计划复用旧机制时仍须满足原 RF 的重开条件。

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
| `V7-RISK-01` | open_risk | O0 occupancy 与 S0/C0/L0 主链断开 | 因为 occupancy 文件存在就宣称 H1 通过 |
| `V7-RISK-02` | limitation | C0 24/24 是按效应 top-k 的机器筛选，不是用户人工评测 | 写成 human pass，或只报 top-k 隐藏 46/62 全分布 |
| `V7-RISK-03` | open_risk | L0 mask 来自 RGB 差分，outside=0 由 hard composition 构造保证 | 用 0 leakage 宣称 occupancy-guided completion 有质量收益 |
| `V7-RISK-04` | open_risk | U0 以极端 V4 为 naive 对照且没有下游任务 | 把 accept rate / RGB signal 写成优于 naive GS 或 mAP 收益 |
| `V7-RISK-05` | legacy_limitation | V7 既有 run 缺正式 manifest、resolved config 与终态标记；V7.1 新 run 已由 EV-10 fail closed | 事后猜 seed/fingerprint 或伪造 immutable provenance |
| `V7-RISK-06` | open_risk | 只覆盖 mini 三场景，S1 held-out 质量偏弱 | 先扩规模、只筛容易场景或把三场景外推为论文结论 |
| `V7-RISK-07` | interface_mitigated_v71 | 11C 已闭合 WorldState→renderer→typed-label 工程链；occupancy repair 的方法增益仍未验证 | 把 label-sync 工程通过写成 occupancy certificate/projection 通过 |
| `V7-RISK-08` | legacy_risk_mitigated_v71 | O0 坐标注释、metadata 与实际变换含义不一致；11A 已冻结显式 frame 合同 | 沿用含义不明的 `pose/T`，或在 round-trip 前计算 H1 指标 |
| `V7-RISK-09` | confirmed_mitigated_v71 | 旧 rotated-corner AABB 使 PILOT-3 动态体素量膨胀 1.72–2.83 倍；扁平语义不能诚实移除 actor | 把旧 O0 AABB 当正式安全几何，或移除 actor 后把体积恢复为 free |
| `V7-RISK-10` | confirmed_open_risk | 分层后 base unknown 仍约 96.0–97.6%，证书容易通过拒绝/abstain 获得表面 precision | 把 UNKNOWN 并入 PASS/FAIL，或降低观测门槛追求 yield |
| `V7-RISK-15` | architecture_mitigated_v71 | certificate detection 与 trajectory projection 若混组会混淆检测和修复收益 | D1 修改 C trajectory，或把 D1/D2 合成单一 validity 数字 |
| `V7-RISK-16` | open_risk | lateral displacement proposal 可能不形成真实 3D cut-in/merge label transition | 用位移幅度或 RGB 差分代替 scenario-effect gate |
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
