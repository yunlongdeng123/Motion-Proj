# OccGS-Resim V7.1 Auto Research Plan

> **归档状态**：`EXECUTED / H1-CERT_REJECTED / H1-PROJ_REJECTED`
> **归档日期**：2026-07-24
> **权威当前入口**：[`../../../RESEARCH_STATUS.md`](../../../RESEARCH_STATUS.md)
> **说明**：本文保留执行时的计划、门槛和授权措辞，仅用于复核。H1-11D 已在冻结 pilot 上触发停止；
> 文中的未执行 H2/H3/scale 不再构成当前任务或授权。

> - **工作名称**：Visibility-Certified Counterfactual Resimulation with Instance-Aware Gaussian Scene Graphs
> - **中文名称**：基于实例感知高斯场景图的可见性认证反事实重仿真
> - **计划版本**：V7.1
> - **最后更新**：2026-07-23
> - **文档状态**：`pending`；候选计划，仅定义研究与执行协议，不单独构成实验授权
> - **证据基线**：`9722fa2`（V7 feasibility 收口提交）
> - **当前决策**：`modify_method_then_scale`
> - **当前优先任务**：`V7-EV-10`，随后为 `V7-H1-11`
> - **首要硬件约束**：单张 RTX 4090 24 GB；正式单 run 峰值显存目标 `<22 GB`
> - **前序计划**：[`OCCGS_RESIM_AUTORESEARCH_PLAN_V7.md`](OCCGS_RESIM_AUTORESEARCH_PLAN_V7.md)
> - **历史执行记录**：[`archive/2026-07/v7-feasibility/OCCGS_RESIM_AUTORESEARCH_PLAN_V7_EXECUTED.md`](archive/2026-07/v7-feasibility/OCCGS_RESIM_AUTORESEARCH_PLAN_V7_EXECUTED.md)

当前状态、执行授权和下一任务只看 [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)。如果该文件尚未把“当前计划”
切换到 V7.1，则本文件是候选计划，Auto Research Agent 不得仅凭本文件启动长实验、下载大数据或改写当前研究状态。
数值事实以 [`EXPERIMENTS.md`](EXPERIMENTS.md) 和原始 run 产物为准；失败边界以
[`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md) 为准。

---

## 0. Executive decision

### 0.1 V7.1 的核心改线

V7.1 不再把“逐帧 2D diffusion 补洞”设为核心方法。主线调整为：

```text
日志观测 + 标定 + instance-aware Gaussian scene graph + 分层几何/观测/渲染证据
→ 唯一 WorldStateSequence
→ matched actor trajectory proposal
→ 3D scenario-effect gate
→ safety / free-unknown / visibility / kinematics certificate
→ 同一状态驱动 RGB、分类型 depth、vehicle instance、有限语义、box 与 evidence maps
→ 几何可见性生成 disocclusion 与 provenance
→ 真实跨相机/跨时间证据优先恢复
→ 必要时进行 3D-native Gaussian completion
→ 只有满足触发条件时，才使用 geometry-conditioned video prior 并蒸馏回 3D
→ 带 truth tier、unknown mask、world/render/artifact hashes 的训练样本
```

V7.1 的方法主张是：

> **显式 3D 世界状态决定几何、动作、遮挡和标签；每个输出像素说明其证据来源与不确定性；生成先验只处理无法由真实证据恢复的外观，并且不能绕过统一 3D 状态直接成为带硬标签的正式传感器真值。**

### 0.2 成功标准

成功不是“编辑后画面更好看”，而是同时满足：

1. actor edit 在运动学、连续碰撞、free/occupied/unknown、道路支撑、事件效果和可见性上可审计；
2. RGB、各类 depth、vehicle instance、有限语义、2D/3D box 和 evidence maps 共享
   `world_state_hash`，并分别绑定正确的 `render_request_hash` 与实际 `artifact_hash`；
3. source removal、new occlusion 和 disocclusion 由几何定义，而不是 RGB 差分猜测；
4. 每个像素和标签都有 support source、visibility event、uncertainty 和可用 truth tier；
5. 方法在 matched proposal 上比 kinematic-only / pairwise-only baseline 降低外部违例，同时不靠拒绝一切获胜；
6. 合成数据在 untouched real test scenes 上带来下游任务增益。

### 0.3 默认执行顺序

```text
V7-EV-10 证据与 run contract
→ V7-H1-11 WorldState / evidence / label 闭环
→ V7-H2-12A 几何 disocclusion 与 provenance audit
→ V7-H3-13A 下游数据管线 smoke
→ 视 H2 audit 的 residual unknown 决定：
   ├─ unknown 不具实质影响：跳过生成分支，直接进入正式效用验证
   └─ unknown 具有实质影响：
      V7-H2-12B 真实观测恢复
      → V7-H2-12C 3D-native completion
      → 仍不足时才允许 V7-H2-12D 生成先验蒸馏
→ controlled validation expansion
→ V7-H1/H2/H3 confirmatory
→ V7-SCALE-14
→ V7-PAPER-15 claim freeze
```

`V7-H2-12D` 不是默认任务，而是条件任务模板。若触发条件不成立：

- 不创建 H2-12D run；
- 不写 `REJECTED`、`FAILED`、`BLOCKED` 或任何 terminal marker；
- 只在 H2 父任务 `summary.json` 中记录：

```json
{
  "h2_generation_branch": "not_triggered",
  "trigger_reason": "materiality_not_met"
}
```

仓库计划状态仍只使用 `pending/running/blocked/done/rejected`。这里通过“不实例化任务”表达 not applicable，
避免把“无需 diffusion”错误汇总成“diffusion 假设失败”。

### 0.4 清空上下文后的启动合同

新的 Auto Research Agent 收到用户明确的“执行 V7.1 plan”指令后，必须把该指令视为启动授权请求，而不是依赖旧对话。
第一轮只做以下同步，不直接跳到 H1：

1. 完整读取 `AGENTS.md`、`RESEARCH_STATUS.md`、`RESEARCH_FAILURES.md`、`EXPERIMENTS.md` 和本文件；
2. 检查 Git status、相关 run manifest、terminal markers 与证据根；
3. 若 `RESEARCH_STATUS.md` 仍指向 V7，则根据用户当轮明确授权，将当前计划切换到本文件，记录日期、当前 commit/
   dirty fingerprint、文档路径和下一任务 `V7-EV-10`；
4. 将执行范围先锁定为 `V7-EV-10`；只有 EV-10 gate 完成并更新状态文件后，才解锁 11A；
5. 任何旧 run 只按 retrospective evidence 读取，不补造 provenance。

因此，即使对话上下文被清空，执行入口仍是“状态同步 → EV-10”，而不是从文档中任意挑选实验。

---

## 1. 当前事实、证据边界与技术债

### 1.1 已完成的 V7 feasibility

| 模块 | 冻结事实 | V7.1 可继承内容 | 不得误写成 |
|---|---|---|---|
| D0 数据 | mini 003/004/005；3 个前向相机；8 秒；10 Hz 处理 | 冻结 pilot scenes、标定、actor annotations | scene-disjoint 或规模结论 |
| B0 重建 | 3/3 StreetGS 完成；test PSNR 25.60 / 20.18 / 25.37 | object-centric GS checkpoint 与背景/actor 分解 | OccGS 方法优于其他重建器 |
| O0 occupancy | 200×200×16，0.4 m；unknown/free/static/dynamic；unknown 保留 | occupancy artifact 与初始接口 | occupancy 已参与编辑或已准确 |
| S0 editor | raised-cosine 横移；运动学和中心距离规则可运行 | proposal 形式和 V0/V1/V2/V3 历史对照 | occupancy-certified trajectory |
| C0 render | RigidNodes pose 可重写；RGB/depth/rigid component 可渲染 | 三维 actor transform 主链 | 完整多相机与同步标签闭环 |
| L0 completion | RGB-diff mask + Telea + hard composition；12 帧 outside L1=0 | hard-composition 实现不变量和 Telea 弱基线 | completion 质量或几何正确 |
| U0 proxy | accepted edit 有 RGB signal；极端 V4 被拒绝 | 数据出口 smoke 的部分输入 | matched downstream utility |

### 1.2 现有实现必须先承认的边界

| 文件 | 当前行为 | V7.1 必须修复或隔离的内容 |
|---|---|---|
| `occupancy/build_scene_occupancy.py` | LiDAR ray carving；动态 box 写入扁平 semantics | 坐标说明需统一；box 为旋转角点 AABB 粗填充；source actor 下的 base state 丢失；unknown 很高 |
| `resim/s0_trajectory_editor.py` | 运动学、横向范围、actor/ego 中心距离验证 | 没有 occupancy、swept OBB、road support、visibility；聚合文件曾被单次运行覆盖 |
| `resim/c0_counterfactual_render.py` | 从 edit JSON 改 `RigidNodes` pose；默认只取 camera 0 | state 未持久化；actor mapping 未版本化；没有同步 semantic/instance/box/occupancy；多相机未闭环 |
| `resim/l0_local_completion.py` | V0/edited RGB 差分、膨胀、Telea | mask 不是 disocclusion；逐帧图像结果没有三维解释；outside=0 是合成公式保证 |
| `resim/u0_utility_screen.py` | constraint / RGB / closing-rate proxy | 极端 V4 不是 matched naive baseline；没有真实下游模型 |

### 1.3 当前开放假设

| 假设 | 精确定义 | 当前状态 | 对应旧风险 |
|---|---|---|---|
| H1-CERT | 在 C 的同一 realized trajectory 上，occupancy/visibility certificate 能发现 pairwise-only 漏掉的违规 | open | `RF-05/08/16`，`V7-RISK-01/02/07` |
| H1-PROJ | occupancy-aware projection 能在保持 scenario effect 和 edit magnitude 时提高 usable yield | open | `RF-05/08/16`，`V7-RISK-01/07` |
| H2 | 几何 provenance 能可靠区分 known/unknown，并且真实证据优先的恢复能降低 unknown 而不破坏已知区域 | open | `RF-05/06`，`V7-RISK-03` |
| H3 | 带 certificate、truth tier 和同步标签的反事实样本能提高 untouched-real 长尾任务指标 | open | `RF-09/16`，`V7-RISK-04/06` |
| H2-GEN | residual unknown 在几何恢复后仍具实质影响，且 3D-distilled generation 能提供额外收益 | not evaluated；仅触发后实例化 | `RF-05/06`，新增生成一致性风险 |

### 1.4 新增、执行时必须登记的风险

以下是从现有代码直接暴露的 V7.1 风险。实现阶段若确认，应同步加入
[`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)，不得只留在 run 日志：

| 建议 ID | 风险 |
|---|---|
| `V7-RISK-08` | occupancy 注释、metadata 与实际变换可能存在坐标框架理解歧义；任何 H1 指标前必须做 round-trip audit |
| `V7-RISK-09` | 当前动态 box 使用旋转角点 AABB 填充，可能产生系统性假碰撞；扁平 semantics 无法诚实移除 source actor |
| `V7-RISK-10` | occupancy 全局 unknown 约 96%，固定拍脑袋阈值可能让方法通过拒绝一切获得表面 precision |
| `V7-RISK-11` | S1 reconstruction 质量较弱，可能使 reconstruction error 与 edit/recovery error 混杂 |
| `V7-RISK-12` | 生成外观与显式 depth/instance 不一致时，给生成像素附硬标签会制造伪真值 |
| `V7-RISK-13` | synthetic artifact 或 reconstruction identity 泄漏可能让下游模型学习域捷径而非 cut-in/merge 规律 |
| `V7-RISK-14` | 单一 state hash 会掩盖 renderer config 差异；非规范浮点/四元数序列化会使重复构建 hash 漂移 |
| `V7-RISK-15` | certificate 与 trajectory projection 混在同一 D 组，会把检测增益、修复增益、缩幅和拒绝混为一谈 |
| `V7-RISK-16` | 仅按 lateral displacement 生成 proposal 可能没有真实 cut-in/merge 事件或 label transition |
| `V7-RISK-17` | Gaussian expected depth、first-hit depth 和 LiDAR measured depth 淵称 metric depth 会制造伪 GT |
| `V7-RISK-18` | pseudo-hole 只代表可由 alternate view 观测的背景，不能给真正未观测 disocclusion 提供像素 GT |
| `V7-RISK-19` | 冻结旧 Gaussians 不阻止新增 Gaussians 改变 known-ray transmittance、depth order 或 held-out cameras |
| `V7-RISK-20` | 仅在自定义事件分类器上获益不足以支持通用数据效用，且需要标准任务确认 |

---

## 2. 研究定位、创新边界与非目标

### 2.1 准确定位

V7.1 是：

> **基于已记录真实场景、逐场景重建的 offline counterfactual neural resimulator。**

它不是：

- 跨场景 feed-forward 4D reconstruction；
- 能从 ego action 自由 rollout 的完整 world model；
- CARLA 的闭环动力学替代；
- 任意视角、任意车辆、任意天气的通用编辑器；
- 仅靠图像生成模型给出几何真值的数据生成器。

允许离线使用同一日志的其他相机和其他时间作为背景证据，但必须保存：

- `source_camera_id`；
- `source_timestamp`；
- `time_offset_ms`；
- `causal_or_noncausal`；
- 重投影和遮挡检查；
- target view 是否从候选源池中严格排除。

如果使用未来帧背景，该产物可称为 offline resimulation，不得写成 causal prediction 或 action-response world model。

### 2.2 预期论文贡献

按优先级排序：

1. **Unified Instance-Aware Gaussian World State**：安全几何、观测证据、render support、background
   Gaussians、vehicle actor nodes、trajectory、
   cameras、visibility 与标签由同一版本化状态驱动。
2. **Visibility and Validity Certificate**：对轨迹、帧和像素分别输出 kinematic、collision、free/unknown、
   road support、visibility、label synchronization certificate。
3. **Evidence-Aware Counterfactual Rendering**：区分观测、重投影、scene-state 推导、生成和 unknown，
   不把所有渲染像素当成同等级真值。
4. **Geometry-First Background Recovery**：先复用真实跨相机/跨时间观测，再做 3D-native completion；
   生成模型仅是触发式增强。
5. **Training-data validity study**：在 matched naive GS 和 real-only 对照下验证真正的下游收益。

### 2.3 明确非主张

V7.1 在相应 gate 通过前不得声称：

- occupancy improves legality；
- multi-view consistency；
- physically correct simulation；
- diffusion completion improves quality；
- synthetic data improves detection / prediction；
- human preference pass；
- scene-generalizable world model；
- real-time 或 closed-loop。

---

## 3. 不可违反的研究不变量

1. **唯一状态源**：editor、renderer、label writer、occupancy writer 不得各自读取并改写不同 pose JSON。
2. **三层哈希**：同一 sample 必须同时校验 `world_state_hash`、`render_request_hash` 和逐文件
   `artifact_hash`；只比较一个 state hash 不足以证明可比。
3. **unknown 不是 free**：未知空间不得因缺少 LiDAR hit 被自动标成可行。
4. **source actor 移除不等于 free**：移除 actor 后，底层未观测区域恢复为 unknown，而不是凭空 free。
5. **标签不来自 diffusion**：3D box、safety geometry、instance identity 和 measured depth 不能从生成像素反推。
6. **生成不绕过 3D**：正式数据中的生成内容必须蒸馏/优化回 3D state，再由统一 renderer 输出；否则只能作为
   `visualization_only`。
7. **matched proposal**：A/B/C/D1/D2 使用相同 scene、actor、requested trajectory、相机、时间、seed 和预算；
   D1 必须逐字节复用 C 的 realized trajectory。
8. **独立 evaluator**：方法用 voxel occupancy 时，主 evaluator 使用连续 OBB、原始 annotations、map 和
   held-out visibility checks，不得让方法规则自己定义成功。
9. **不按效果 top-k 报主结果**：top-k 只用于诊断，主结果报告全 proposal pool、coverage、per-scene 和 worst case。
10. **hard composition 与质量分开**：outside exact 是实现检查，不是 H2 结论。
11. **人工 verdict 归用户**：Agent 只能生成盲审包、完整提示词和空结果模板。
12. **失败可追溯**：任何 threshold、actor、scene、mask 或 baseline 变更都产生新 config fingerprint 和 run ID。
13. **事件先于像素**：cut-in/merge positive、negative 和 label transition 完全由 realized 3D state 定义，不读取
    RGB、重建质量或 downstream prediction。
14. **有限语义范围**：V7.1 只声明 `vehicle-instance + undifferentiated static-background`，不使用完整 panoptic
    或全类别 semantic claim。

---

## 4. V7.1 统一系统设计

### 4.1 总体架构

```text
                        ┌──────────────────────────────┐
nuScenes / annotations ─▶ safety geometry               │
LiDAR observations ────▶ free / unknown evidence       │
StreetGS + source views ▶ render support                ├─▶ WorldStateSequence
calibration / poses ───▶ camera + ego state            │        │
actor registry / edit ─▶ requested trajectory + event  │        │ world_state_hash
                        └──────────────────────────────┘        ▼
                                               B/C/D2 projector
                                               + D1 certificate
                                                           │
                                      ┌────────────────────┴───────────────────┐
                                      ▼                                        ▼
                              synchronized render                    validity/provenance
                    RGB / typed depth / alpha / instance       frame/pixel truth tiers
                    limited semantic / boxes / evidence               uncertainty
                           render_request_hash
                                      │                                        │
                                      └────────────────────┬───────────────────┘
                                                           ▼
                                      artifact_hash + training-data exporter
```

### 4.2 `WorldStateSequence`

`WorldStateSequence` 是一个 edit case 的不可变顶层记录，至少包含：

```text
WorldStateSequence
├── schema_version
├── sequence_id
├── parent_sequence_id
├── task_id / run_id / proposal_id
├── scene_id / split / data_fingerprint
├── coordinate_convention
├── timebase_hz / start_timestamp / end_timestamp
├── reconstruction
│   ├── checkpoint_sha256
│   ├── background_gaussian_version
│   └── third_party_fingerprint
├── actor_registry_ref / actor_registry_sha256
├── safety_geometry_ref / safety_geometry_sha256
├── observation_evidence_ref / observation_evidence_sha256
├── render_support_ref / render_support_sha256
├── edit_spec
│   ├── requested_trajectory
│   ├── realized_trajectory
│   ├── projection_delta
│   └── scenario_effect
├── frames[]
└── world_state_hash
```

路径只保存在 manifest 的 locator 字段中，hash payload 只保存内容 hash。renderer、分辨率和 rasterizer policy
不属于 WorldState，必须进入独立 RenderRequest。certificate verdict、projection algorithm/config、运行统计和
provenance report 是引用 `world_state_hash` 的 derived artifacts，不进入 WorldState hash payload；否则 D1 在同一
C trajectory 上增加 certificate 就会错误改变世界状态。

### 4.3 `WorldStateFrame`

每帧至少包含：

```text
WorldStateFrame
├── frame_index / timestamp
├── timestamp_provenance: observed | interpolated
├── ego_pose_world
├── camera_models[]
│   ├── camera_id
│   ├── intrinsics
│   ├── camera_to_ego
│   └── camera_to_world
├── actor_nodes[]
│   ├── true_instance_id
│   ├── rigid_model_index
│   ├── occupancy_instance_id
│   ├── class_name / dimensions
│   ├── canonical_gaussian_ref
│   ├── source_pose_world
│   ├── edited_pose_world
│   ├── velocity / acceleration / yaw_rate / jerk
│   ├── pose_provenance
│   └── visibility_state
├── safety_geometry_frame_ref
├── observation_evidence_frame_ref
├── render_support_frame_ref
└── scenario_effect_state
```

所有矩阵统一使用右手系、米、秒和明确的 `T_dst_src` 命名。禁止只写含义不明确的 `pose`、`transform`、`T`。

### 4.4 三层哈希与 canonical serialization

#### A. `world_state_hash`

只描述“世界是什么”，payload 必须包含：

- actor source/edited poses、dimensions、identity 和轨迹；
- ego pose 与 timebase；
- safety geometry、observation evidence、render support 的内容 hash；
- camera intrinsics/extrinsics 和坐标 convention；
- actor registry 内容 hash；
- source checkpoint SHA256；
- requested/realized trajectory 与 scenario effect；
- WorldState schema 和 canonicalization version。

计算：

```text
world_state_hash = SHA256(canonical_json(world_state_payload))
```

不得包含 camera output resolution、rasterizer、depth policy、写入路径或运行时间。

#### B. `render_request_hash`

描述“如何从该世界生成一个具体 artifact”，payload 必须包含：

- `world_state_hash`；
- frame/timestamp 与 camera ID；
- output resolution、crop/resize policy；
- rasterizer mode、precision 和 deterministic flags；
- DriveStudio commit、adapter version 与 renderer config 内容 hash；
- depth definitions；
- alpha/first-hit threshold；
- background/actor compositing policy；
- vehicle-instance 与 limited-semantic rendering policy；
- color space、quantization、image/array encoding。

计算：

```text
render_request_hash = SHA256(canonical_json(render_request_payload))
```

同一 WorldState 在不同 resolution、depth definition 或 renderer commit 下必须产生不同
`render_request_hash`。

#### C. `artifact_hash`

`artifact_hash` 是实际输出文件字节的 SHA256，不是 metadata hash：

```text
artifact_hash = SHA256(file_bytes)
```

每个文件的 sidecar/manifest 保存：

```json
{
  "relative_path": "artifacts/.../rgb.png",
  "world_state_hash": "...",
  "render_request_hash": "...",
  "artifact_hash": "...",
  "size_bytes": 0,
  "media_type": "image/png"
}
```

多文件 sample 另有 `artifact_set_hash`，它是按 `relative_path` 排序后的 artifact manifest 做 canonical
serialization 后的 SHA256；不能用目录 mtime 或文件枚举顺序计算。

#### D. 规范编码

V7.1 统一使用 UTF-8 canonical JSON，并冻结 `canonicalization_version`。规则如下：

- object keys 按 UTF-8 code-point 顺序排序，无多余空白；
- 文本先做 Unicode NFC；
- 有语义顺序的数组保持原顺序；actor、camera 等无序集合先按稳定 ID 排序；
- 整数使用无前导零十进制；
- 浮点先转 IEEE-754 binary64，`-0.0` 规范为 `+0.0`，再编码为带类型前缀的 JSON string
  `@f64:<固定16位小写big-endian-hex>`；
- NaN、`+Inf`、`-Inf` 一律拒绝；
- quaternion 固定 `wxyz`、归一化，并令第一个绝对值大于容差的分量为正；近零 quaternion 拒绝；
- matrix 固定 row-major；
- 不包含绝对路径、写入时间、临时目录、进程 ID 或随机枚举顺序；
- 大型 array 不内嵌，只写 shape、dtype、semantic version 和实际内容 SHA256。

同一输入跨进程重复构建三次，`world_state_hash` 和 `render_request_hash` 必须完全一致；改变任一被声明字段必须触发
对应 hash 改变。hash tests 必须覆盖浮点 `-0.0`、四元数 `q/-q`、key 顺序、NaN/Inf 和 renderer-config change。

### 4.5 版本化 actor registry

建立以下一一映射：

```text
nuScenes / instances_info true_id
↔ DriveStudio dataset instance column
↔ RigidNodes model_idx
↔ occupancy instance_id
↔ exported limited-semantic / vehicle-instance label id
```

registry 每行保存：

- 原始 ID 与类型；
- 映射来源和解析算法版本；
- actor 出现时间范围；
- canonical dimensions；
- RigidNodes 是否存在；
- source observation coverage；
- checkpoint hash；
- 映射验证结果。

以下任一情况必须 fail closed：

- 一对多或多对一；
- actor 在 annotations 中存在但 RigidNodes 缺失；
- model index 在 checkpoint reload 后变化；
- instance mask 的主要连通域与投影 actor 完全不重叠；
- registry hash 与 WorldState 中记录不一致。

### 4.6 三类证据分离，而不是一个 occupancy 包办全部职责

V7.1 明确维护三个用途和物理含义不同的产品：

```text
SafetyGeometryFrame
├── continuous actor / ego OBBs
├── swept volumes
├── HD-map drivable polygons（若可用）
└── static collision geometry with explicit source

ObservationEvidenceFrame
├── LiDAR observed mask
├── ray-free evidence
├── unknown
├── per-instance dynamic evidence layers
├── evidence count / age
└── sensor uncertainty

RenderSupportFrame
├── Gaussian primitives / checkpoint content hash
├── source-view support observations
├── supporting camera-time observations
├── reprojection residual
└── render uncertainty
```

`RenderSupportFrame` 只保存阈值无关的连续/raw support state；使用具体 alpha/first-hit threshold 得到的 hull、
surface 和 per-camera visibility map 属于 RenderRequest artifact，因此阈值进入 `render_request_hash`，不进入
`world_state_hash`。

职责固定为：

| 产品 | 可以决定 | 不能决定 |
|---|---|---|
| Safety geometry | collision、road support、swept clearance | 不用 Gaussian floaters 判断安全 |
| Observation evidence | free/unknown、abstention、sensor coverage | 不把“没看见”当 free，也不把 box 当背景几何 |
| Render support | visibility、provenance、disocclusion support | 不作为唯一轨迹通行判据 |

ObservationEvidence 的组合保持分层，不直接覆盖原始证据：

```text
unknown
→ ray-free
→ static occupied
→ dynamic occupied by instance
```

编辑 actor 时：

1. 从 observation evidence 的 `dynamic_instance_layers[source_actor]` 移除 source layer；
2. 不修改 base static/free/unknown evidence；
3. 在 edited pose 处重新体素化 canonical actor occupancy；
4. safety collision 仍由连续 OBB/map 独立检查；
5. render coverage 仍由 Gaussian/source-view support 独立检查；
6. 输出三类 evidence 的 source/edited 内容 hash。

动态 actor 初始实现使用真正的 oriented-box voxelization，不得继续使用旋转角点 AABB 作为正式 H1 方法。
当前 O0 AABB 版本只保留为 `observation_evidence_v1_coarse` baseline。Gaussian alpha hull 不得写回 safety
geometry；box occupancy 不得冒充真实 background surface。

### 4.7 Instance-Aware Gaussian scene graph

scene graph 至少含：

```text
SceneGraph
├── static_background_node
├── actor_nodes[instance_id]
│   ├── canonical Gaussians
│   ├── canonical actor geometry proxy
│   └── time-indexed SE(3)
├── ego_node
├── camera_nodes
└── safety / observation / render evidence refs
```

V7.1 正式名称使用 “Instance-Aware”，首轮只建模：

```text
vehicle instances + undifferentiated static background
```

`limited_semantic_mask` 只能包含 `unknown / static_background / vehicle` 及明确 ignore 值。road、vegetation、
pedestrian 等未建模类别不能从背景外观猜出。只有未来覆盖 static semantic classes 和多类动态实例后，才允许把
“Panoptic Gaussian Scene Graph”作为正式方法名。

### 4.8 同步输出、depth 与有限语义合同

同一 `WorldStateFrame` 必须生成：

- RGB；
- alpha；
- `depth_render_expected`；
- `depth_surface_first_hit`；
- `depth_lidar_measured` 与 sparse-valid mask；
- background-only / actor-only RGB、alpha 和明确命名的 first-hit depth；
- vehicle instance mask；
- `limited_semantic_mask`；
- raw projected 2D box；
- visible 2D box；
- 3D box；
- safety geometry、observation evidence 与 render support maps；
- per-actor visibility；
- support-source map；
- visibility-event map；
- uncertainty map；
- `world_state_hash`、`render_request_hash` 和各文件 `artifact_hash`。

三种 depth 语义不可互换：

| 名称 | 定义 | Truth tier / 用途 |
|---|---|---|
| `depth_render_expected` | Gaussian alpha-weighted expected depth | visualization / diagnostic；不是无误差 metric GT |
| `depth_surface_first_hit` | 累积 alpha 首次越过冻结阈值时的 surface depth | visibility、z-order、T1 state-derived；必须记录 threshold |
| `depth_lidar_measured` | 原始 LiDAR 投影到相机的稀疏测量 | T0 sparse truth；保存 valid mask、sensor timestamp 与 calibration |

actor/background composited depth 必须按上述定义加 layer suffix，例如
`depth_surface_first_hit_background`，不得再输出含义不明的 `depth.png` 并称 metric ground truth。

2D box 同时保存两个定义：

- `box2d_projected_raw`：3D box 八角点投影后的 unclipped/clipped box；
- `box2d_visible`：由可见 instance raster 得到，低于最小可见面积则标为不可见。

不能用二者之一冒充另一个。

### 4.9 Certificate 采用 `PASS / FAIL / UNKNOWN`

每帧和整条 trajectory 均输出三态 certificate。至少包含：

| Certificate | 核心检查 |
|---|---|
| `kinematic` | 速度、加速度、jerk、yaw rate 与相对原轨迹变化 |
| `swept_collision` | 连续时间 swept OBB 与 ego/其他 actor 的相交或安全间隔 |
| `static_safety_geometry` | actor OBB/swept volume 与独立 static collision geometry 的交叠 |
| `dynamic_safety_geometry` | 与其他 actor/ego continuous OBB 的交叠或 clearance |
| `unknown_intrusion` | actor swept volume 落入 LiDAR observation-evidence unknown 的比例 |
| `free_evidence` | actor footprint 的 ray-free sensor evidence；只作 evidence/abstention |
| `road_support` | HD map 优先；无 map 时保留 UNKNOWN，不用 Gaussian 或 box occupancy 伪造道路 |
| `visibility` | render support、多相机投影面积、visible fraction、first-hit ordering 和时间连续性 |
| `scenario_effect` | corridor crossing、gap/TTC/headway、label transition 与 event duration |
| `edit_adherence` | realized trajectory 对 requested trajectory 的保真度 |
| `label_sync` | typed depth/instance/box/evidence 的 world/render/artifact hashes 与几何共位 |

示例统计量：

$$
r_{\mathrm{safety}}
=
\frac{|V_{\mathrm{actor}}\cap G_{\mathrm{safety}}|}
{|V_{\mathrm{actor}}|},
\qquad
r_{\mathrm{unknown}}
=
\frac{|V_{\mathrm{actor}}\cap V_{\mathrm{unknown}}|}
{|V_{\mathrm{actor}}|}.
$$

阈值不从 edited test proposal 的结果调优。先在冻结 calibration pool 上使用：

- 未编辑真实 actor trajectories 作为 positive controls；
- 人工构造且机制明确的 collision/off-road/teleport negatives；
- 原始连续 box/map evaluator；

选择能保留至少 95% 可测真实 controls、同时识别预注册 negatives 的阈值，然后冻结 config fingerprint。
如果 observation evidence 稀疏导致真实 controls 大量 UNKNOWN，应先修 evidence，不得把 UNKNOWN 改成 FREE。

### 4.10 Certificate 与 trajectory projection 因果拆分

H1 不再用一个 D 组同时做检测和轨迹修改。正式分组为：

| 组别 | realized trajectory | 回答的问题 |
|---|---|---|
| A `raw_rigid` | requested trajectory 原样写入 | naive edit 有多少问题 |
| B `kinematic` | kinematic projection | 运动学修复带来什么 |
| C `pairwise` | B + continuous pairwise projection | 不使用 occupancy 时能做到什么 |
| D1 `occgs_certify_only` | **逐字节复用 C 的 realized trajectory**，禁止修改 | occupancy/visibility certificate 能否发现 C 漏掉的违规 |
| D2 `occgs_project` | 从同一 requested trajectory 做 occupancy-aware minimal projection | occupancy-aware repair 能否提高 yield 且保留事件效果 |

D1 必须满足：

- `realized_trajectory_hash_D1 == realized_trajectory_hash_C`；
- `world_state_hash_D1 == world_state_hash_C`；
- 不创建新的 candidate search；
- 不改变 pose、timing、magnitude 或 labels；
- 只输出 `PASS/FAIL/UNKNOWN` 和 certificate components。

D1 报告 `certificate precision / recall / abstention coverage`。D2 单独报告 `repair success / projection delta /
scenario-effect retention / final usable yield`。论文不得把二者合成一个“OccGS improves validity”数字。

允许在预注册范围内进行最小修复：

$$
\tau^\star
=
\arg\min_{\tau}
\Big[
\lambda_p D(\tau,\tau_{\mathrm{proposal}})
+\lambda_s E_{\mathrm{smooth}}(\tau)
\Big]
\quad
\text{s.t. D2 的 safety/evidence/visibility constraints。}
$$

首轮不需要复杂可微优化器，可使用确定性有限候选投影。C 与 D2 必须共享 candidate set、搜索预算和 tie-break；
唯一差异是 D2 增加冻结的 safety/evidence/visibility constraints。

一个 proposal 只有在以下条件同时满足时才计入 `usable_yield`：

- D2 certificate 为 PASS；
- realized peak edit 至少达到 requested peak 的 80%；
- 时间峰值偏移不超过预注册容差；
- requested positive/negative scenario effect 在 realized state 上仍成立；
- label_sync PASS；
- 没有把 UNKNOWN 偷记为 PASS。

这样 D2 不能通过把轨迹缩回 V0、把 positive 修成 negative 或拒绝全部困难样本获得虚假优势。

---

## 5. 像素 provenance、可见性事件与 truth tier

### 5.1 不使用一个含义混杂的 map

“像素来自哪里”和“编辑改变了什么可见性”是两个正交问题。V7.1 默认输出两张 map。

#### A. `support_source_map`

| Code | 名称 | 含义 |
|---:|---|---|
| 0 | `invalid_or_outside` | 无有效相机射线或输出范围外 |
| 1 | `observed_static` | target/source real observation 对该三维表面有直接支持 |
| 2 | `transformed_actor` | canonical actor Gaussians 经 edited SE(3) 渲染 |
| 3 | `reprojected_static` | 从其他相机/时间通过 depth 和 z-buffer 重投影 |
| 4 | `background_gs_supported` | background GS 有三维支持，但 target 像素未被直接观测 |
| 5 | `completed_3d` | 新增/优化后的 3D Gaussians，经多视角验证后渲染 |
| 6 | `generated_appearance` | 生成 prior 候选；尚未通过 3D 统一时只可 visualization |
| 7 | `unknown` | 无足够几何或外观证据 |
| 8 | `mixed_boundary` | alpha 混合、深度不确定或多来源边界 |

#### B. `visibility_event_map`

| Code | 名称 | 含义 |
|---:|---|---|
| 0 | `unchanged_visible` | source 和 edited state 的主要可见层一致 |
| 1 | `newly_occluded` | edited actor/geometry 新遮挡原可见背景或 actor |
| 2 | `disoccluded_known` | source actor 移走后显露，且存在可验证背景支持 |
| 3 | `disoccluded_unknown` | source actor 移走后显露，但无可验证背景支持 |
| 4 | `actor_relocated` | edited actor 的新可见 footprint |
| 5 | `occlusion_boundary` | z-order 或 alpha 边界不稳定 |
| 6 | `not_applicable` | 非编辑影响区域 |

如果最终论文只展示一张 provenance 图，可把两张 map 组合为 bitfield 或可视化颜色；原始产物仍必须分别保存。

### 5.2 Truth tier

每个 pixel/label 保存训练可用等级：

| Tier | 来源 | 允许用途 |
|---|---|---|
| `T0_MEASURED` | 原始传感器/annotation 直接证据，包括 sparse `depth_lidar_measured` | 可用于相应观测标签 |
| `T1_STATE_DERIVED` | 同一显式 WorldState 的 actor transform、box、safety geometry、first-hit surface、vehicle instance | 可用于声明范围内的几何/实例标签；不冒充 measured depth |
| `T2_GEOMETRY_RECOVERED` | 真实重投影或多视角一致的 3D completion | 通过专项 gate 后用于 masked geometry/RGB 训练 |
| `T3_APPEARANCE_ONLY` | 未统一到 3D 的生成或 2D completion | 只用于 visualization 或不附硬几何标签的 masked RGB 任务 |
| `IGNORE_UNKNOWN` | 无证据或证据冲突 | 所有监督 loss ignore |

任何 `T3_APPEARANCE_ONLY` 像素不得附带看似精确的 measured depth、free-space、negative detection 或 safety GT。
若生成结果已蒸馏到 3D 并通过 H2-12C/D 的一致性 gate，可升级到 `T2_GEOMETRY_RECOVERED`，但必须保留原始生成
provenance。

### 5.3 Uncertainty

至少保存：

- typed-depth uncertainty 与 depth-definition ID；
- observation-evidence count；
- number of supporting views；
- reprojection residual；
- GS alpha / transmittance；
- boundary flag；
- temporal consistency residual；
- completion method ID。

不要求首轮训练一个 learned uncertainty model；可先使用可解释、校准后的规则量。禁止把未经校准的 score 称为
probability。

---

## 6. Geometry-derived disocclusion 与分层恢复

### 6.1 几何 mask 定义

对每个相机和时间，使用 `depth_surface_first_hit` 定义分别渲染：

- background-only depth/alpha：$D_{\mathrm{bg}}, A_{\mathrm{bg}}$；
- source actor depth/alpha：$D_{\mathrm{src}}, A_{\mathrm{src}}$；
- edited actor depth/alpha：$D_{\mathrm{edit}}, A_{\mathrm{edit}}$。

用统一 z-buffer 容差 $\epsilon_z$ 定义可见 actor footprint：

$$
M_{\mathrm{src}}
=
[A_{\mathrm{src}}>\alpha_{\min}]
\land
[D_{\mathrm{src}}<D_{\mathrm{bg}}-\epsilon_z],
$$

$$
M_{\mathrm{edit}}
=
[A_{\mathrm{edit}}>\alpha_{\min}]
\land
[D_{\mathrm{edit}}<D_{\mathrm{bg}}-\epsilon_z].
$$

核心事件：

$$
M_{\mathrm{disocc}}
=
M_{\mathrm{src}}\land\neg M_{\mathrm{edit}},
\qquad
M_{\mathrm{newocc}}
=
M_{\mathrm{edit}}\land\neg M_{\mathrm{src}}.
$$

边界区域由 alpha、depth gradient 和 z-order margin 单独扩展为 `occlusion_boundary`，不得与稳定内部像素混报。
当前 `|I_{\mathrm{edited}}-I_{\mathrm{V0}}|` 只保留为诊断对照，不再定义主方法 mask。

### 6.2 两类 disocclusion stratum

Pseudo-hole 只能代表存在可验证 alternate-view evidence 的区域，不能替真正未观测表面制造 RGB ground truth。
H2 必须先把每个 disocclusion pixel 分为：

#### A. `alternate_view_observed`

定义：

- 在冻结 source pool 中至少有一个其他 camera/time 看到同一静态 surface；
- 通过 first-hit depth、z-buffer、dynamic mask 和 reprojection residual gate；
- target camera/time 从 source pool 严格排除；
- source observation 和 support count 可追溯。

可评价：

- PSNR / SSIM / LPIPS；
- mask precision / boundary；
- metric reprojection residual；
- typed depth 和 multi-view consistency。

Pseudo-hole benchmark 只从 A stratum 采样，且必须匹配真实 disocclusion 的 mask shape、depth、texture 和 boundary。

#### B. `no_qualifying_observation`

定义为：在冻结日志、相机、时间和 background-support corpus 内找不到通过 gate 的 alternate observation。它是相对
当前 observation corpus 的“truly unobserved”，不是对现实世界所有可能视角的绝对证明。

不得评价：

- 以不存在的 target RGB 计算 PSNR/SSIM/LPIPS；
- 把生成 patch 当 pixel GT；
- 用视觉 plausibility 代替 geometry truth。

只能评价：

- residual-unknown coverage；
- abstention 与 uncertainty；
- 3D/multi-view self-consistency；
- added-Gaussian ray influence；
- known-view transmittance/depth-order preservation；
- held-out-camera spill；
- 人工 plausibility（单独、非 GT、不得代填）。

所有 H2 aggregate 必须按 A/B 两个 strata 分开报告，禁止用 A 的 pseudo-hole 数字替 B 背书。

### 6.3 恢复级别

#### Level 0：不补，诚实输出 unknown

```text
disoccluded_unknown
→ RGB 可保留 background GS 原始渲染或显式占位
→ truth_tier = IGNORE_UNKNOWN
→ geometry / occupancy / detector negative 均不监督
```

Level 0 是正式 baseline，也是安全退路。

#### Level 1：真实观测恢复

按顺序搜索：

1. background-only Gaussian layer 的已支持表面；
2. 同 timestamp 其他 camera；
3. 其他 timestamp 的相同静态表面；
4. 多帧 LiDAR/depth 支持的 inverse view warp；
5. 多个候选的 z-buffer、photometric 和 temporal consensus。

约束：

- A-stratum/pseudo-hole 评估时 target frame/camera 必须从 source pool 排除；
- actor 和动态遮挡必须剔除；
- 记录 source view、time offset 和 support count；
- 候选冲突时输出 UNKNOWN，不做无依据平均；
- future observation 可用于 offline recovery，但必须显式标为 noncausal。

#### Level 2：3D-native Gaussian completion

仅对有可靠 geometry anchor、但 appearance/point support 稀疏的区域：

- 从真实重投影或 point/depth 初始化新 Gaussians；
- 冻结原有 observed Gaussians；
- 只优化新增 Gaussians或明确的 unknown subset；
- 使用多相机、多时间一致性和 alpha/depth regularization；
- 所有正式输出重新由 3D renderer 产生；
- 为新增 Gaussian 构建 3D authorized volume 和逐相机 authorized ray mask；
- 输出 `added_gaussian_ray_influence_mask`：新增 Gaussian 对 ray color/alpha/transmittance 产生超过冻结阈值影响的区域；
- 测量 `known_ray_transmittance_preservation`；
- 测量 `known_view_depth_order_preservation`；
- 测量 `held_out_camera_spill_ratio`；
- known region 的变化必须单独测量。

“旧 Gaussian 参数未更新”不是 locality 证据。新增 Gaussian 即使自身位于 unknown volume，也可能遮挡原有 surface、
改变 transmittance/depth，或投影到其他 cameras；只有上述 ray-level safeguards 通过才能进入正式输出。

#### Level 3：geometry-conditioned video prior，蒸馏回 3D

只在第 6.5 节触发条件全部成立后允许：

```text
geometry / depth / instance / box / map conditions
→ multi-frame or multi-view candidate generation
→ candidate consistency filtering
→ 仅优化 unknown-region Gaussians
→ multi-view held-out validation
→ unified 3D rerender
```

禁止：

- 通用单帧 inpainting 直接写成正式 simulator RGB；
- 每个 camera 独立采样后直接拼为多相机数据；
- 用生成像素反推 3D box/depth/occupancy；
- 让新增 Gaussians改变 observed region 来换取更好感知指标；
- 无法蒸馏时把 2D patch 标成 `T2_GEOMETRY_RECOVERED`。

### 6.4 H2 方法对照

| 组别 | 方法 | A：observed | B：no observation | 是否可作为正式带标签数据 |
|---|---|---|---|---|
| R0 | no completion + unknown/ignore | baseline | baseline | 是，按 mask 使用 |
| R1 | Telea + hard composition | 弱像素 baseline | 仅 plausibility | 默认 T3 |
| R2 | background-only GS | 可测 | 只测 support/一致性 | 按原 support tier |
| R3 | cross-camera/time real reprojection | 主方法 | 不适用，保持 UNKNOWN | 通过 A-stratum gate 后可为 T2 |
| R4 | visibility-aware 3D Gaussian completion | 可测 | 只测一致性与 spill | 通过分层 gate 后可为 T2 |
| R5 | geometry-conditioned generation → 3D distillation | 可测 | 只测一致性与 spill | 仅在触发且通过专项 gate 后可为 T2 |

### 6.5 生成分支触发条件

只有以下条件全部满足才允许 `V7-H2-12D`：

1. `V7-H1-11` 的 WorldState、label_sync 和 matched pilot 已通过；
2. `V7-H2-12A/B/C` 已完成，R2–R4 无法继续解释 residual unknown；
3. residual unknown 在至少 2 个场景、至少 20 个冻结 clips 中满足下列任一 materiality 条件：
   - median 占完整图像至少 0.5%；或
   - median 占 true disocclusion 至少 10%；或
   - 对预注册下游/人工 endpoint 产生稳定、可复现的显著损害；
4. 有能在单卡 `<22 GB` 运行的 multi-frame / multi-view 或 geometry-conditioned 模型；
5. 正式标签不依赖生成像素，且存在回灌 3D 的实现与 held-out multi-view evaluator；
6. 下载权重、存储和 license 已单独审计并获得当前状态文件授权。

若不满足，R5 不执行，也不创建 H2-12D run。H2 父 summary 写：

```json
{
  "h2_generation_branch": "not_triggered",
  "trigger_reason": "materiality_not_met"
}
```

R0–R4 足以支撑一篇 visibility-certified resimulation 工作。

---

## 7. 全局实验设计

### 7.1 数据 cohort 与结论等级

| Cohort | 范围 | 用途 | 允许结论 |
|---|---|---|---|
| `PILOT-3` | 冻结 mini 003/005/004 | schema、renderer、certificate、H1/H2 pipeline pilot | 三场景方法可行性，不外推 |
| `MINI-VALID` | mini 其余场景，经预注册 eligibility audit 后冻结 | 小规模 scene-disjoint validation | mini 范围的趋势与 worst-case |
| `TRAINVAL-CONFIRM` | nuScenes trainval 的 scene-disjoint split | H1/H2/H3 正式确认 | 仅覆盖冻结数据分布 |
| `EXTERNAL` | 新城市/天气/数据集 | 未来外部有效性 | 不属于 V7.1 最小成功条件 |

当前未授权下载/处理 trainval。`TRAINVAL-CONFIRM` 的准备必须由 `RESEARCH_STATUS.md` 明确解锁。

### 7.2 Eligibility audit

scene/actor 选择只能使用 source observation，不得查看编辑后效果。actor 至少满足：

- vehicle class；
- RigidNodes 和 registry 映射成功；
- 有足够 source track length；
- 至少一个前向相机中达到预注册 visible-area coverage；
- source pose/box/occupancy 坐标 round trip 通过；
- 不因 edited result 漂亮或容易而入选。

如果某 scene 不足 2 个 eligible actors，记录 `insufficient_actor_coverage`。Pilot 若任一冻结场景不足 2 actor，
`V7-H1-11` 状态为 `blocked`，不得悄悄删掉该场景或改成只挑成功 actor。

### 7.3 冻结 proposal bank

PILOT-3 每 scene 至少 2 actors，每 actor 使用 5 个 matched proposals，共至少 30 个 trajectory proposals。
为继承旧 V1/V2/V3 并避免极端 V4，以下只作为 **candidate seeds**：

| Proposal | peak lateral offset | timing |
|---|---:|---|
| P1 | 0.4 m | centered |
| P2 | 0.8 m | early |
| P3 | 0.8 m | late |
| P4 | 1.2 m | centered |
| P5 | 1.6 m | centered |

方向由 source actor 相对 ego corridor/可用 map 的确定性规则决定。若 map 不可用，使用已有 ego-frame side 规则，
并明确记录为 proxy。旧 V4 只用于 validator negative control，不进入 matched primary comparison。

位移模板本身不等于有效 scenario。所有 candidate 必须再经过第 7.4 节的纯 3D scenario-effect gate，最终
proposal bank 同时保存 candidate seed、effect label、pair ID 和 label-transition provenance。

所有 proposal 在运行 A/B/C/D1/D2 前一次性生成和 hash；后续任何改动都创建新 proposal-bank version。

### 7.4 纯 3D scenario-effect gate 与正负配对

event gate 只读取 source/realized 3D WorldState，不读取 RGB、LPIPS、renderer 质量或 downstream model。

`ego_corridor` 优先由 HD map lane corridor 与 ego swept footprint 定义；无 map 时使用 ego-frame swept corridor proxy，
并在 proposal 中标注 `corridor_source=proxy`。在 11B calibration 完成前，以下阈值必须写入
`scenario_effect_v1.yaml` 并冻结：

```text
event_min_consecutive_frames = 5        # 10 Hz 下至少 0.5 s
min_lateral_gap_change_m = 0.5
min_boundary_crossings = 1
ttc_valid_range_s = [1.0, 6.0]
time_headway_valid_range_s = [0.5, 4.0]
positive_label_transition = 0 → 1
negative_label_transition = 0 → 0
```

TTC 只在 longitudinal closing speed 为正且数值稳定时使用；否则使用 time-headway，并记录所用定义。阈值若需根据
真实 cut-in calibration controls 修改，只能在查看 PILOT edit outcome 前修改一次，随后产生新 config hash。

#### Positive cut-in/merge

必须同时满足：

1. source state 在冻结 horizon 内为 negative；
2. realized actor footprint 从 ego-corridor boundary 外跨入；
3. footprint 与 corridor 连续相交至少 `event_min_consecutive_frames`；
4. minimum lateral gap 至少减少 `min_lateral_gap_change_m`；
5. minimum TTC 或 time-headway 进入预注册区间；
6. event duration 达标；
7. realized label 从 0 变 1。

#### Matched negative

使用同一 `scene_id + source_actor_id + source checkpoint + camera/time window`，并满足：

1. 与 positive 来自同一 candidate family；
2. motion magnitude、visibility 和 render budget 按预注册 tolerance matching；若多候选满足，使用固定
   lexicographic tie-break，禁止人工挑选；
3. actor footprint 不满足 corridor 持续进入条件；
4. realized label 保持 0；
5. 不靠换 actor appearance、scene background 或 reconstruction identity 形成负例。

每个 pair 保存：

```text
counterfactual_pair_id
source_actor_id
positive_proposal_id / negative_proposal_id
requested_effect / realized_effect
scenario_effect_hash
```

H1 可以保留 `non_event` proposal 作为 validity stratum；H3 只能使用同一 source actor 的合格正负对。若某 actor
无法从纯 3D state 形成正负 pair，标记 `h3_pair_ineligible`，不得根据 RGB 效果换 actor。D2 修复后若 positive
变成 negative 或 event duration 不足，该 sample 不计 usable positive yield，且 label 必须按 realized state 重算。

### 7.5 H1 matched groups

| 组别 | 状态与约束 |
|---|---|
| A `raw_rigid` | 直接写入 requested SE(3)，不做合法性投影；作为 naive GS edit |
| B `kinematic` | A + 速度/加速度/jerk/yaw-rate 约束 |
| C `pairwise` | B + ego/actor continuous swept OBB 和安全间隔 |
| D1 `occgs_certify_only` | 在 C 的同一 realized trajectory 上只运行 safety/evidence/visibility certificate |
| D2 `occgs_project` | 从同一 requested trajectory 运行 occupancy-aware minimal projection |

A/B/C/D2 使用同一 requested proposal。C 与 D2 使用同一 projection candidate set、预算和 tie-break。D1
的 realized trajectory hash 必须与 C 完全相同。原始 requested trajectory 即使被某组拒绝也保留，可在 sandbox
中生成诊断 render，但不得进入该组正式 export。

### 7.6 时间与相机采样

- certificate 和 state/label generation：完整 80 个 10 Hz frame；
- 多相机一致性：3 个冻结前向 cameras；
- pilot 高成本 render：固定 12 个时间点，使用等距采样并强制包含 edit peak 与最大 occlusion-change frame；
- 选帧规则在查看 RGB 质量前由 proposal geometry 冻结；
- confirmatory render budget 在 pilot 后预注册，所有组一致；
- 不使用 mean RGB edit effect 排序的 top-k 作为主评估集。

### 7.7 Reconstruction quality 分层

S1 test PSNR 20.18，明显弱于 S0/S2。所有结果同时报告：

- reconstruction-quality stratum；
- scene-level metrics；
- worst scene；
- exclusion-free aggregate。

主结果不得删除 S1。若错误主要由基础重建导致，应把结论写成 reconstruction-bounded，而不是继续调 edit mask。

### 7.8 统计单元

- trajectory legality 的独立单元是 `scene × actor × proposal`，不是 frame；
- pixel/label metrics 先在 trajectory 内聚合，再做 scene-clustered bootstrap；
- 下游实验先在 seed 内得到 untouched-real test metric，再跨 seed 汇总；
- 禁止把同一 trajectory 的数千像素当成独立样本制造虚假显著性。

### 7.9 预注册与统计

每个 confirmatory run 前冻结：

- primary endpoint；
- comparison direction；
- practical effect threshold；
- exclusion criteria；
- proposal bank；
- scenario-effect thresholds 与 counterfactual pair manifest；
- scene split；
- seeds；
- bootstrap unit；
- missing/UNKNOWN policy。

默认报告：

- paired effect；
- 95% scene-clustered bootstrap CI；
- per-scene effect；
- worst case；
- raw counts；
- coverage/yield；
- 不只报告 p-value。

PILOT-3 只有 3 scenes，不以“统计显著”作为论文结论。其 gate 用于判断是否值得进入 scene-disjoint validation。
正式显著性只在 `MINI-VALID` 或 `TRAINVAL-CONFIRM` 的预注册规模上解释。

---

## 8. 指标与 Gate 定义

### 8.1 H1 primary endpoints

#### A. D1 certificate precision / recall / abstention

D1 在 C 的固定 realized trajectory 上比较独立 external reference：

- precision：D1 FAIL 中 external violation 的比例；
- recall：C 漏过但 external evaluator 判 violation 的 case 中，D1 标 FAIL 的比例；
- abstention：D1 UNKNOWN 占固定 C pool 的比例；
- PASS coverage：D1 PASS 占固定 C pool 的比例。

UNKNOWN 不计 true positive，也不偷偷并入 FAIL。若 C pool 中没有 external-positive case，recall 不可定义，
H1-CERT 标 `blocked`（insufficient positive support），不能写 100% recall。

#### B. D2 external hard-violation rate

外部 evaluator 不读取方法 certificate 的最终 verdict，独立使用：

- continuous OBB / swept volume；
- 原始 ego/other actor annotations；
- map polygon（若可用）；
- raw calibration；
- independent depth/visibility checks；
- 人工盲审仅作为独立补充。

每条 export trajectory 若出现 collision、严重 off-road、不可解释 teleport、depth-order contradiction 或 label hash
冲突，则记 external hard violation。

#### C. D2 repair effect 与 usable yield

$$
\mathrm{usable\ yield}
=
\frac{\#\{\text{PASS, adherence pass, scenario-effect pass, label\_sync pass}\}}
{\#\{\text{fixed matched proposals}\}}.
$$

同时报告：

- C→D2 violation transition matrix；
- repair success rate；
- projection delta；
- magnitude/timing retention；
- positive/negative scenario-effect retention；
- reject 与 UNKNOWN 原因。

#### H1 pilot gate

进入 H2/H3 pipeline 的最低条件：

1. registry 覆盖每个 PILOT-3 scene 至少 2 actors；
2. D1 与 C 的 `realized_trajectory_hash` 100% 相同；
3. D1 在冻结 calibration/measurable pool 上 precision 和 recall 均至少 0.80；若 matched C pool 无 external
   positives，则 H1-CERT 为 `blocked` 而不是 pass；
4. D2 相对 C 的 aggregate external hard-violation rate 严格更低；
5. 至少 2/3 scenes 的 D2 violation rate 不高于 C；
6. 任一 scene 的 D2 violation rate 恶化不得超过 10 个百分点；
7. D2 usable yield 至少 70%，且不是通过退化到 V0 或破坏 scenario effect；
8. D1/D2 的 UNKNOWN 单独报告，不能并入 PASS/FAIL；
9. label_sync 和三层 hash 关键不变量 100% 通过；
10. S1 必须单独报告，无“方向可解释”人工豁免。

这十条由聚合脚本直接给出 boolean gate；不得在结果出来后增加“可解释”例外。

#### H1 confirmatory gate

H1 只有在 scene-disjoint validation 中同时满足以下条件才标为 `supported`：

- D1 certificate precision/recall 达到预注册 practical threshold，abstention 完整报告；
- D2 对 C 的 paired hard-violation risk difference 的 95% CI 上界 `<0`；
- 预注册 practical reduction 达标；
- usable yield `≥70%`，且相对 C 的下降不超过预注册 non-inferiority margin；
- magnitude/timing、scenario effect、visibility 和 label-sync 不劣；
- 没有单一 scene 完全失效且被 aggregate 隐藏。

H1-CERT 与 H1-PROJ 分别给 verdict。D1 只提高 precision 但 recall/coverage 崩溃，或 D2 只靠 shrink/reject
降低 violation，均不能写成 H1 supported。

### 8.2 Label synchronization 指标

| 指标 | 定义或检查 |
|---|---|
| world-state agreement | 同一 sample 的 typed depth/instance/box/evidence 引用相同 `world_state_hash` |
| render-request agreement | camera/resolution/renderer/depth policy 完全相同才允许比较 `render_request_hash` |
| artifact integrity | 每个实际文件 SHA256 等于 sidecar `artifact_hash`，bundle 等于 `artifact_set_hash` |
| pose round-trip | world→model→world translation / rotation residual |
| instance-depth order | instance 可见像素与 actor/background `depth_surface_first_hit` z-order 一致 |
| visible box containment | instance visible mask 位于 emitted visible box 内 |
| raw box projection | 3D box 角点独立投影与 writer 输出一致 |
| evidence/box alignment | edited instance observation layer 与 edited 3D box 共位；不把它当背景几何 |
| temporal identity | instance ID 在全序列和跨 camera 不改变 |
| visibility continuity | 无几何原因的 single-frame identity/visibility 跳变率 |

实现不变量要求 100% 通过；数值容差写入 config 并由 synthetic unit fixtures 校准。

### 8.3 Provenance / mask 指标

只在 `alternate_view_observed` stratum 使用可测 pseudo-disocclusion：

1. 从已知静态区域采样与真实 actor footprint 形状、面积、深度和边界复杂度匹配的 mask；
2. target view 从 source pool 排除；
3. 运行 provenance classifier 和 recovery；
4. 使用被隐藏的真实 target 作为 ground truth。

指标：

- A/B stratum counts 与 classification provenance；
- known/unknown precision、recall、F1；
- false-known rate（高优先级）；
- mask IoU、boundary F-score；
- support-source calibration；
- per-source coverage；
- disocclusion area distribution；
- full-image 与 edit-ROI residual unknown。

H2-12A 最低 gate：

- A/B stratum 必须分开，B 不得出现伪造 PSNR/LPIPS；
- false-known rate 不高于预注册 2% 上限；
- known/unknown precision 的 95% CI 下界达到预注册安全阈值；
- geometry mask 相对 RGB-diff mask 在 pseudo-hole IoU/boundary 上有一致提升；
- source/new actor footprint 与 disocclusion 不再混为同一 mask。

### 8.4 Recovery 指标

Primary：

- A stratum：pseudo-hole LPIPS 与 multi-view reprojection error；
- B stratum：residual unknown coverage、uncertainty、3D/multi-view self-consistency；
- 两个 strata：known-region preservation 与 failure rate。

Secondary / safeguards：

- PSNR、SSIM；
- boundary seam gradient error；
- temporal warping error / flicker；
- typed-depth ordering violation；
- instance contamination；
- known-region L1/LPIPS；
- added-Gaussian ray influence area；
- known-ray transmittance change；
- known-view first-hit depth-order change；
- held-out-camera spill ratio；
- source-view count 和 failure rate；
- runtime、peak VRAM、disk。

H2-12B/C gate：

1. A stratum 中 R3/R4 相对 R2 在 pseudo-hole LPIPS 和 reprojection 上有 paired 改善；
2. B stratum 不计算像素 GT，residual unknown/uncertainty/self-consistency 完整报告；
3. residual unknown 至少降低预注册幅度；
4. false-known、typed depth、instance 和 known-region safeguard 不劣；
5. added-Gaussian influence、transmittance、depth-order 和 spill 全部通过冻结阈值；
6. 最终输出来自统一 3D render；
7. 三个场景均报告，S1 不被排除。

H2-12D 除上述条件外，还要求：

- 相对最佳非生成 R4 有增量收益；
- held-out cameras/time 的一致性改善；
- generated candidates 无法通过 3D distillation 的比例完整报告；
- 直接 2D 输出与 3D-rerender 输出分开，正式结果只用后者。

### 8.5 H3 下游指标

H3 分为 pilot task 和 paper confirmatory standard task，不能只依赖一个为本数据定制的分类器。

#### Task E：pilot / targeted event task

> **camera-only cut-in / merge risk prediction**：给定冻结时长的前向相机历史 clip，预测目标 actor 是否在未来
> 冻结 horizon 内进入 ego corridor，并输出风险分数。

选择原因：

- 与当前 actor trajectory edit 直接对应；
- 比完整 camera-3D detector 更适合单卡先验证；
- 可以在 untouched real scenes 上评估；
- 标签可从真实 annotation 或 WorldState 独立生成。

Task E primary：

- untouched-real test PR-AUC。

Task E secondary：

- macro-F1；
- recall at fixed FPR；
- calibration ECE/Brier；
- per-distance、per-occlusion、per-scene、night/weather strata。

#### Task S：paper confirmatory standard task

默认使用：

> **nuScenes camera-only 3D detection**，采用已公开、非本项目定制且能在当前单卡预算内训练的标准 baseline，
> 报告官方 NDS、mAP、vehicle-class AP 和距离/遮挡 strata。

具体 baseline、commit、官方 evaluator 和 config 必须在 H3-13B preflight 中审计并冻结。若该任务在当前环境确实
不可执行，不能由 Agent 临时换成另一个自定义分类器；必须把 H3-13B 标 `blocked`，更新计划后才能改用 BEV occupancy
或标准 trajectory-prediction benchmark。

synthetic 样本只对 vehicle class 提供正/负监督；pedestrian、vegetation 等未建模类别必须设 ignore，不能因
`limited_semantic_mask=static_background` 被错误当成 detector background negative。

Task S 的作用不是要求所有 synthetic 数据都提升 overall NDS，而是防止只在专门为当前 edit 构造的 Task E 上获得
不可外推的收益。至少报告：

- overall NDS / mAP；
- vehicle AP；
- cut-in/merge-related、near-range、occluded vehicle strata；
- real-test non-inferiority safeguards。

#### 冻结配置

- Task E 的 input clip、帧率、分辨率、camera set、horizon 和 corridor；
- Task S baseline/commit、data pipeline 和 official evaluator；
- positive/negative pair manifest；
- scene/actor balanced sampler；
- model architecture 与 pretrained-state policy；
- optimizer、steps、batch、augmentation；
- 3 个以上固定 seeds。

正式组别：

```text
R          real only
R+N        real + matched A/raw-rigid GS edits
R+O        real + D2/occgs_project certified V7.1 edits
R+O+Rcv    real + V7.1 + validated geometry-first recovery
R+O+Gen    可选；只有 H2-12D 通过才加入
```

所有增广组：

- synthetic 样本数相同；
- 每个 source actor 的 positive/negative 数量相同；
- scene 和 actor sampling 权重相同；
- positive/negative pair 必须位于同一 split，不能拆开；
- synthetic event label 只从 realized WorldState 计算；
- real 样本、训练 steps、optimizer 和 seed 相同；
- 只在 train scenes 生成 synthetic；
- val/test 完全真实且 scene-disjoint；
- sample identity、background 和 reconstruction checkpoint 不跨 split 泄漏。

#### 强制反捷径 controls

1. **Synthetic artifact classifier**：预测 real vs synthetic，报告 domain separability；同时验证 artifact score 与
   event label 在 actor/scene-balanced sampling 后不相关。
2. **Background-only classifier**：移除/遮蔽目标 actor，只用 background 预测正负 label。同一 actor 正负 pair 的
   background 应匹配；若仍显著高于 permutation baseline，说明 scene/render leakage。
3. **Label-shuffle control**：在 train 内按 actor/scene block 打乱 label，性能必须回落到由 permutation distribution
   定义的 chance interval。
4. **Pair-ID control**：禁止 pair ID、proposal ID、method ID、文件名、目录结构或 metadata 进入模型输入。
5. **Actor/scene balance audit**：每个 group 输出 class × actor × scene 计数及最大 imbalance。

domain classifier AUC 高本身说明 domain gap，但不自动证明主任务无效；真正的 hard gate 是 artifact/background
features 不得预测 event label，label-shuffle 不得保持主任务性能。chance interval 使用至少 1,000 次
actor/scene-blocked label permutations 预先生成；background-only、artifact-to-event 和 label-shuffle 的 primary
metric 必须不高于 permutation distribution 的 95th percentile。balance audit 要求每个 source actor 的正负计数
完全相等，group 间 scene/actor count 差为 0。

#### H3 confirmatory gate

1. Task E：`R+O` 的 real-test PR-AUC 同时优于 R 与 R+N；
2. Task E：paired seed/scene CI 下界超过 0，并达到预注册 practical margin；
3. Task S：完成 official real-test/val evaluation；vehicle/cut-in-related standard endpoint 相对 R 和 R+N 有预注册
   增益，且 overall NDS/mAP 不低于 non-inferiority margin；
4. 两任务增益不只来自单一 scene、actor 或 distance stratum；
5. calibration 和关键安全 stratum 不出现不可接受退化；
6. background-only 和 label-shuffle controls 位于冻结 chance interval；
7. artifact-label correlation、pair metadata leakage 和 actor/scene imbalance 均通过 audit。

若只通过 Task E，允许结论仅为“targeted cut-in risk utility”，不能进入宽泛 C4 数据效用 claim。Paper-level H3
需要 Task S 也达到预注册 gate。`R+O+Rcv` 是增量贡献；`R+O+Gen` 只有 H2-12D 实际触发并通过时才创建。

---

## 9. 任务分解与逐 Gate 交付

### 9.1 `V7-EV-10` — 证据索引与正式 run contract

**状态**：`pending`
**GPU**：不需要
**目标**：诚实索引旧证据，并让所有 V7.1 新 run fail closed。

#### 工作包

1. 构建 `runs/occgs_resim/V7_EVIDENCE_INDEX.json`；
2. 对 B0/O0/S0/C0/L0/U0 逐项记录现存文件、hash 和缺失字段；
3. 旧 run 统一标记 `evidence_mode=retrospective`；
4. 复用并扩展 `motion_proj.runtime.experiment`、`fingerprint`、`atomic`、`stage`；
5. 新增 V7.1 run wrapper、唯一 run ID、resolved config、三层 hash 和 terminal marker 检查；
6. 修复 S0 聚合不得覆盖已有 scene summary；
7. 增加 conflicting terminal markers、run ID reuse、missing world/render/artifact hash、missing summary 的 tests；
8. 实现 optional branch “trigger 不成立则不创建 run”的父 summary schema 与聚合测试。

#### 禁止

- 事后猜 seed；
- 把 `9722fa2` 伪装成每个旧 run 开始时 commit；
- 给旧目录补伪造 `COMPLETE`；
- 修改旧 metrics；
- 覆盖旧 run ID。

#### Gate

- 既有证据可从 index 定位；
- 缺失项显式为 `missing`；
- 新 run 缺任何必需 artifact 时不能 COMPLETE；
- 只有一个 terminal marker；
- 未触发 H2-12D 时没有 run/terminal marker，且父 summary 为 `not_triggered`；
- `pytest` 目标测试通过；
- 生成 run-contract smoke artifact。

#### 交付

```text
runs/occgs_resim/V7_EVIDENCE_INDEX.json
configs/resim/v71/run_contract.yaml
motion_proj/runtime/（最小必要扩展）
tests/test_v71_run_contract.py
docs/EXPERIMENTS.md（仅登记已验证的新事实）
docs/RESEARCH_STATUS.md（仅在获得执行授权且 gate 完成后更新）
```

---

### 9.2 `V7-H1-11A` — 坐标、schema 与 actor registry

**状态**：`pending`
**目标**：在任何 occupancy claim 前建立可信状态底座。

#### 工作包

- 实现 `WorldStateSequence/Frame` schema；
- 明确 world/ego/lidar/camera/model/grid 的 `T_dst_src`；
- 对现有 occupancy metadata、代码注释和实际变换做 audit；
- 建立 versioned actor registry；
- synthetic fixture 验证 translation、yaw、box corners、camera projection 和 model pose round trip；
- 实现第 4.4 节 canonical encoder；
- 同一输入跨进程重复构建 `world_state_hash` 必须一致；
- renderer policy 变化只改变 `render_request_hash`，不得改变 `world_state_hash`；
- actual output 逐文件计算 `artifact_hash`；
- 插值帧保留 provenance。

#### Gate

- PILOT-3 每 scene 至少 2 actors 一一映射；
- 所有 coordinate round-trip tests 通过；
- actor registry reload 后稳定；
- `q/-q`、key order、`-0.0` canonicalization tests 通过，NaN/Inf fail closed；
- 任意映射冲突 fail closed；
- 如 `V7-RISK-08` 确认，先更新风险账本再进入 11B。

---

### 9.3 `V7-H1-11B` — 分层证据、scenario effect 与 certificate calibration

**状态**：`pending`
**目标**：冻结 safety geometry、observation evidence、render support、scenario effect 与 certificate 的独立职责。

#### 工作包

- 实现 OBB voxelization；
- 建立 continuous OBB/map safety geometry；
- 分离 LiDAR base free/static evidence 与 per-instance dynamic observation layers；
- 建立 Gaussian/source-view render-support product；
- source actor removal 和 edited actor insertion；
- 输出 evidence count、unknown、observation age；
- 加入 continuous swept OBB evaluator；
- 有 map 时接入 drivable-area；无 map 时显式 UNKNOWN；
- 构建 real-positive / controlled-negative calibration pool；
- 实现第 7.4 节纯 3D scenario-effect evaluator；
- 冻结 certificate thresholds、scenario-effect thresholds 和 pair schema。

#### 必做消融

```text
observation_evidence_v1_coarse_AABB
vs.
observation_evidence_v2_layered_OBB
```

#### Gate

- 移除 actor 后不会把其原体积自动改为 free；
- edited actor layer 可逆且不污染其他 instance；
- Gaussian floaters 不进入 safety collision；
- box occupancy 不冒充 background surface；
- real source trajectories 的 certificate retention 达到预注册目标；
- controlled negatives 的检测能力达到预注册目标；
- scenario-effect gate 在冻结 synthetic fixtures 和 real controls 上机器可判定；
- unknown 不被自动转为 PASS；
- safety/evidence/render-support overlays 分开生成，Agent 不代填人工 verdict。

---

### 9.4 `V7-H1-11C` — 同步 renderer 与 label regeneration

**状态**：`pending`
**目标**：同一 WorldState 驱动全部传感器和标签。

#### 工作包

- renderer 读取 WorldState，而不是独立 edit JSON；
- 扩展到 3 个冻结前向 cameras；
- 输出 background-only / actor-only layers；
- 分别生成 `depth_render_expected`、`depth_surface_first_hit` 和 sparse `depth_lidar_measured`；
- 生成 vehicle instance、`limited_semantic_mask`、raw/visible 2D box、3D box 和三类 evidence maps；
- 每个 artifact sidecar 嵌入 `world_state_hash`、`render_request_hash`、`artifact_hash`；
- 实现 label synchronization audit；
- 严格区分不可见、截断、UNKNOWN 和缺失。

#### Gate

- label-sync implementation invariants 100% 通过；
- 同 actor 在跨相机和全时间保持 identity；
- source/edited state 可重复渲染；
- expected/first-hit/measured depth 不混名，expected depth 不登记为 T0；
- limited semantic scope 只含 unknown/static_background/vehicle/ignore；
- background-only 与 actor-only 合成关系在容差内成立；
- no-edit V0 与原 renderer 的回归差异在预注册容差内；
- S1 同样通过接口 gate，不能因视觉质量差而跳过。

---

### 9.5 `V7-H1-11D` — Matched pilot ablation

**状态**：`pending`
**目标**：分别检验 certificate detection 和 occupancy-aware trajectory repair，不混合二者。

#### 固定范围

- PILOT-3；
- 至少 2 actors/scene；
- 5 proposals/actor；
- 纯 3D scenario-effect tags 与同 actor 正负 pair；
- A/B/C/D1/D2 五组；
- D1 逐字节复用 C realized trajectory；
- 完整 state/certificate；
- 固定 12-frame × 3-camera render audit；
- 所有 proposal、不是 effect top-k。

#### 输出

- proposal-level verdict 和 realized trajectory；
- D1 certificate precision/recall/abstention；
- D2 repair transition、projection delta 与 scenario-effect retention；
- external evaluator metrics；
- usable yield / UNKNOWN / adherence；
- label sync；
- scene/worst-case；
- blind review pack 和完整评测提示词草稿；
- compute profile。

#### Gate

按第 8.1 节 H1 pilot gate。若失败：

- 记录是 occupancy fidelity、threshold、mapping、reconstruction 还是核心假设失败；
- 只允许一次由明确失败机制驱动的修复；
- 不得换成极端 V4、删 S1、降 coverage 或只报 top-k。

---

### 9.6 `V7-H2-12A` — Visibility / provenance audit

**状态**：`pending`
**前置**：11C 完成；11D 至少通过 state/label 工程 gate
**目标**：先回答“真正需要恢复的 unknown 到底有多少”。

#### 工作包

- 渲染 source/edited actor 和 background layers；
- 生成 `M_src`、`M_edit`、`M_disocc`、`M_newocc`、boundary；
- 生成 support-source、visibility-event、truth-tier maps；
- 将 disocclusion 分为 A `alternate_view_observed` 与 B `no_qualifying_observation`；
- 只在 A 构建形状/深度匹配的 pseudo-holes；
- 比较 geometry mask 与 RGB-diff mask；
- 统计 background GS 已覆盖、其他相机/时间可恢复、最终 unknown 的比例。

#### Gate

按第 8.3 节。必须产出：

```text
full image unknown
edit ROI unknown
true disocclusion unknown
A/B disocclusion stratum
per scene / camera / time / typed-depth stratum
```

若 residual unknown 不满足第 6.5 节 materiality，不创建 H2-12D run；父 summary 写
`h2_generation_branch=not_triggered`。

---

### 9.7 `V7-H2-12B` — 真实跨相机/跨时间恢复

**状态**：`pending`
**目标**：最大化利用真实观测，不引入 hallucination。

#### 工作包

- source view retrieval；
- first-hit / measured-depth-based warp；
- z-buffer and dynamic-mask rejection；
- multi-source consensus；
- support count / time offset / causal flag；
- target-view exclusion test；
- A/B stratum routing；
- R0/R1/R2/R3 对照。

#### Gate

- A-stratum pseudo-hole primary endpoint 改善；
- B stratum 保持 UNKNOWN，不计算伪 PSNR/LPIPS；
- false-known 不劣；
- multi-view/depth/instance safeguards 通过；
- 每个 recovered pixel 可回溯 source；
- 无支持则 UNKNOWN。

---

### 9.8 `V7-H2-12C` — Visibility-aware 3D Gaussian completion

**状态**：`pending`
**触发**：12B 后仍有 material residual unknown，且存在可靠 geometry anchor
**目标**：只对 unknown subset 增加三维表达能力。

#### 工作包

- 新 Gaussian 初始化；
- observed Gaussians freeze；
- unknown-only optimization mask；
- 3D authorized volume 与 camera-ray authorization；
- multi-view/time loss；
- depth/alpha regularization；
- added-Gaussian ray influence mask；
- known-ray transmittance preservation；
- known-view first-hit depth-order preservation；
- held-out-camera spill ratio；
- 用 no-op / authorized-volume synthetic controls 冻结 influence、transmittance、depth-order 和 spill thresholds，
  然后再打开正式 12C targets；
- held-out view evaluation；
- R4 对照与 regression tests。

#### Gate

- 最终统一 3D rerender；
- held-out multi-view improvement；
- 参数 freeze、ray influence、transmittance、depth order 和 spill 全部通过；
- residual unknown 下降；
- 无 identity/depth-order 退化；
- peak VRAM `<22 GB`。

---

### 9.9 `V7-H2-12D` — Optional generative prior distillation

**条件任务模板**：触发前不创建任务记录、run 或 terminal marker
**触发**：第 6.5 节全部满足
**目标**：验证生成 prior 是否在最佳几何恢复之上提供必要增益。

只有第 6.5 节全部满足时，才在 `RESEARCH_STATUS.md` 中实例化 `V7-H2-12D`，初始计划状态为 `pending`。
若 trigger 不成立，只更新 H2 父 summary 为 `not_triggered`；这不是 `REJECTED`。只有实际执行了 12D 且预注册
research gate 失败，才使用 `REJECTED`。

#### 工作包

- 先做 model/license/VRAM audit；
- geometry-conditioned multi-frame candidate generation；
- multi-view candidate consistency filter；
- 只蒸馏到 unknown Gaussians；
- final 3D rerender；
- direct-2D 与 distilled-3D 分开报告。

#### Gate

按第 8.4 节 H2-12D gate。若不能稳定蒸馏回 3D：

- 结果只可 `visualization_only`；
- H2-GEN 标为 rejected；
- 不继续调通用 2D inpainting。

---

### 9.10 `V7-H3-13A` — Downstream pipeline smoke

**状态**：`pending`
**前置**：11C 完成；正式使用 synthetic 前 11D 通过
**目标**：只验证数据 schema、truth-tier masking、train/eval 入口和 leakage tests。

#### 范围

- PILOT-3；
- 小模型、单 seed；
- R / R+N / R+O；
- 同 source actor positive/negative pairs；
- scene/actor balanced sampler；
- synthetic-artifact、background-only 和 label-shuffle controls；
- 只产生 engineering conclusion；
- 不写下游收益。

#### Gate

- real/synthetic label schema 一致；
- IGNORE_UNKNOWN 正确进入 loss mask；
- 未建模非 vehicle 类在 synthetic detection loss 中为 ignore，不得当 background negative；
- split 和 actor identity 无泄漏；
- pair ID 不进入模型输入且 pair 不跨 split；
- realized-state label transition 校验；
- PILOT-3 每 scene 至少有 1 个同 actor 合格正负 pair；不足则 `blocked`，不得用换 scene 或看 RGB 后选 actor 修补；
- background-only / label-shuffle smoke 回落到 chance；
- sample count 与训练 budget matched；
- untouched-real evaluation 可重复；
- training smoke 无 NaN/OOM。

---

### 9.11 `V7-H3-13B` — Formal downstream utility

**状态**：`blocked`
**触发**：13A 通过，scene-disjoint 数据和执行授权到位
**目标**：同时在 targeted event task 与标准 camera 3D detection 上检验 real-test 数据效用。

#### 范围

- scene-disjoint train/val/test；
- R / R+N / R+O / R+O+Rcv；
- 至少 3 seeds；
- Task E cut-in/merge risk prediction；
- Task S nuScenes camera-only 3D detection 标准 baseline 与官方指标；
- 同 actor 正负对、scene/actor balance；
- synthetic-artifact、background-only、label-shuffle controls；
- 等 synthetic 数量和训练预算；
- untouched real test；
- optional R+O+Gen 只有 12D 通过才加入。

#### Gate

按第 8.5 节。若 `R+O` 不优于 R+N：

- H3 rejected；
- 不以更漂亮的 RGB、accept rate 或 constraint score 替代；
- 保留 WorldState/provenance 作为基础设施结果；
- 不进入大规模生成。

若只有 Task E 通过而 Task S 未通过，只能形成 targeted utility 结论，不能形成 paper-level 通用 synthetic-data claim。

---

### 9.12 `V7-SCALE-14` — 扩规模与多卡

**状态**：`blocked`

只有以下全部满足才解锁：

1. H1 confirmatory supported；
2. H3 Task E 与 Task S 达到第 8.5 节 paper-level gate；
3. H2 路线已明确选择 no-fill / geometry recovery / 3D completion 中的正式版本；
4. 单卡瓶颈是吞吐，不是 OOM、低 candidate yield 或核心方法失败；
5. trainval/external 数据 license、磁盘和 preprocessing 已审计；
6. scene-disjoint split 和 run contract 可复用。

双卡优先用于独立 scene、baseline 或 seed 并行，不默认 DDP。多卡不能用来掩盖单卡超 24 GB 或方法不稳定。

---

### 9.13 `V7-PAPER-15` — Claim freeze 与复现包

**状态**：`blocked`

只有完成相应 confirmatory gates 后：

- 冻结 method config；
- 冻结主表与图的数据源；
- 从 raw JSON/JSONL 自动重建所有数字；
- 写 limitations；
- 生成 minimal reproduction commands；
- 对 citation、license、data split、human review provenance 做最终审计；
- 禁止在写作阶段用未登记 trial 补主结果。

---

## 10. 建议代码与配置结构

核心逻辑进入可测试 package，顶层 `resim/` 只保留 CLI：

```text
motion_proj/resim/
├── schema.py
├── canonical_hash.py
├── coordinates.py
├── actor_registry.py
├── world_state.py
├── safety_geometry.py
├── observation_evidence.py
├── render_support.py
├── scenario_effect.py
├── trajectory_projection.py
├── certificates.py
├── render_request.py
├── drivestudio_adapter.py
├── label_regeneration.py
├── provenance.py
├── recovery.py
├── completion_3d.py
├── ray_influence.py
├── downstream_export.py
├── downstream_controls.py
└── metrics.py

resim/
├── v71_build_registry.py
├── v71_build_world_state.py
├── v71_build_scenario_pairs.py
├── v71_calibrate_certificates.py
├── v71_run_certify_only.py
├── v71_run_projection_ablation.py
├── v71_render_and_label.py
├── v71_audit_provenance.py
├── v71_recover_background.py
├── v71_export_downstream.py
└── v71_aggregate.py

configs/resim/v71/
├── run_contract.yaml
├── pilot3_split.yaml
├── proposal_bank_v1.yaml
├── scenario_effect_v1.yaml
├── world_state_v1.yaml
├── canonical_hash_v1.yaml
├── safety_geometry_v1.yaml
├── observation_evidence_v2.yaml
├── render_support_v1.yaml
├── certificate_calibration.yaml
├── h1_matched_pilot.yaml
├── h2_provenance.yaml
├── h2_recovery.yaml
├── h3_cut_in_smoke.yaml
├── h3_camera3d_confirm.yaml
└── h3_shortcut_controls.yaml
```

第三方 DriveStudio 不直接大改；通过 adapter 隔离 checkpoint、dataset 和 renderer 差异。

### 10.1 最小测试矩阵

```text
tests/test_v71_run_contract.py
tests/test_v71_world_state_schema.py
tests/test_v71_canonical_hash.py
tests/test_v71_render_request_hash.py
tests/test_v71_artifact_hash.py
tests/test_v71_coordinate_roundtrip.py
tests/test_v71_actor_registry.py
tests/test_v71_layered_occupancy.py
tests/test_v71_actor_remove_insert.py
tests/test_v71_swept_obb.py
tests/test_v71_certificate_tristate.py
tests/test_v71_certify_only_trajectory_identity.py
tests/test_v71_scenario_effect.py
tests/test_v71_counterfactual_pairing.py
tests/test_v71_typed_depth.py
tests/test_v71_limited_semantic_scope.py
tests/test_v71_multicamera_labels.py
tests/test_v71_visibility_masks.py
tests/test_v71_provenance_truth_tier.py
tests/test_v71_disocclusion_strata.py
tests/test_v71_target_view_exclusion.py
tests/test_v71_known_region_freeze.py
tests/test_v71_added_gaussian_ray_influence.py
tests/test_v71_split_leakage.py
tests/test_v71_shortcut_controls.py
tests/test_v71_optional_branch_not_triggered.py
tests/test_v71_matched_budget.py
```

关键 synthetic fixtures：

- 空场景 + 单 box；
- 旋转 box 与 AABB 差异；
- source actor 移除后 base unknown 恢复；
- 两 actor 连续 crossing 的 swept collision；
- actor 在相机边界截断；
- actor 与背景 depth 前后关系切换；
- expected/first-hit/measured depth 区分；
- 两相机同 actor identity；
- alternate-view-observed pseudo-hole；
- no-qualifying-observation disocclusion；
- 新 Gaussian 遮挡 known ray / held-out-camera spill；
- conflicting source views；
- `q/-q`、`-0.0`、key-order canonical hash；
- render config 改变但 world state 不变；
- artifact byte corruption；
- C/D1 trajectory byte mismatch；
- positive→negative scenario-effect loss；
- same-actor positive/negative balance；
- background-only/label-shuffle leakage；
- optional H2-12D trigger-not-met；
- terminal marker conflict。

---

## 11. Run、artifact 与 provenance 合同

### 11.1 目录

```text
runs/occgs_resim/v71/<task_id>/<run_id>/
├── manifest.json
├── resolved.yaml
├── fingerprints/
│   ├── code.json
│   ├── environment.json
│   ├── data.json
│   ├── third_party.json
│   └── checkpoints.json
├── proposal_bank.json
├── actor_registry.json
├── world_state_manifest.json
├── render_request_manifest.json
├── artifact_manifest.json
├── metrics.jsonl
├── summary.json
├── artifacts/
├── logs/
└── exactly one of COMPLETE / FAILED / REJECTED / BLOCKED
```

### 11.2 Run ID

建议格式：

```text
v71_<task>__<split>__s<seed>__<UTC timestamp>__<config_hash8>
```

run ID 永不复用。重试使用新 run ID，并在 manifest 记录 `parent_run_id` 和 `retry_reason`。

### 11.3 Manifest 必需字段

- task ID；
- run ID / parent run ID；
- command；
- plan version；
- code commit、dirty flags、tracked/untracked diff hash；
- config fingerprint；
- data/split/proposal fingerprints；
- third-party commit；
- checkpoint hashes；
- scene/actor/camera/time；
- seed；
- environment / CUDA / GPU；
- WorldState schema version；
- canonicalization version；
- world state / render request / artifact-set hashes；
- safety-geometry / observation-evidence / render-support versions；
- certificate/scenario-effect/provenance versions；
- causal/noncausal recovery policy；
- started/ended time；
- exit reason；
- terminal status。

### 11.4 Terminal marker

只能有一个：

- `COMPLETE`：计划内工作成功结束，必需 artifacts 全部存在；
- `FAILED`：工程运行失败；
- `REJECTED`：运行完成，但预注册 research gate 未通过；
- `BLOCKED`：缺数据、映射、授权或外部条件，无法继续。

`COMPLETE` 不等于 hypothesis supported；research verdict 单独写入 `summary.json`。

terminal marker 只适用于已经实例化并实际创建目录的 run：

- `REJECTED` 只表示实验实际执行完成但预注册 research gate 失败；
- optional branch trigger 不成立时不创建 run，不写 marker；
- 父任务只记录 `h2_generation_branch=not_triggered` 和 `trigger_reason`；
- 聚合器不得把 `not_triggered` 计入 rejected/failed/blocked denominator。

### 11.5 自动聚合

所有主表必须从 `summary.json` / `metrics.jsonl` 重新生成。聚合脚本必须：

- 验证 config/proposal/split fingerprints；
- 验证 canonicalization version、三层 hashes 和 artifact bytes；
- 拒绝混合不同 metric schema；
- 拒绝 conflicting marker；
- 报告 missing runs；
- 将 optional-not-triggered 与 missing-required-run 严格区分；
- 报告全 pool 与 exclusion；
- 输出 machine-readable JSON 和 Markdown table；
- 不覆盖 raw run。

---

## 12. Auto Research Agent 执行协议

### 12.1 每次开始

依次：

1. 读取 `docs/RESEARCH_STATUS.md`；
2. 读取 `docs/RESEARCH_FAILURES.md`；
3. 读取 `docs/EXPERIMENTS.md`；
4. 检查 Git status；
5. 检查目标 run 的 manifest/terminal marker；
6. 确认当前任务在状态文件中被授权；
7. 运行最小 preflight。

### 12.2 单次研究循环

```text
读取授权
→ 冻结问题、对照、endpoint、预算
→ 实现最小变更与单元测试
→ smoke
→ 检查 artifact contract
→ pilot
→ 自动聚合
→ 根据预注册 gate 标 COMPLETE / REJECTED / BLOCKED
→ 更新事实源与状态
→ 决定是否进入下一任务
```

### 12.3 Agent 可以自动做

- 只读审计；
- 小范围代码实现；
- unit/smoke tests；
- 已授权任务内的单卡短 run；
- 自动聚合和可视化材料生成；
- 失败诊断；
- 按证据更新当前文档；
- 构建空的人审模板和完整提示词。

### 12.4 Agent 不得自动做

- 越过 `RESEARCH_STATUS.md` 启动下一 gate；
- 下载全量 trainval、Waymo/PandaSet 或大型生成权重；
- 启动双卡/长 sweep；
- 改写旧 run；
- 代填人工 verdict；
- 以目视印象修改 primary endpoint；
- 为了过 gate 删除 S1、困难 scene 或 UNKNOWN；
- push 远端；
- 把工程 COMPLETE 写成 hypothesis supported。

### 12.5 调参纪律

- metric/schema validity 未通过前，不做超参 sweep；
- 每个失败机制最多一次有理论依据的修复；
- 第二次仍失败则登记 blocked/rejected，并请求新的研究决策；
- threshold 只能在 calibration pool 上调；
- confirmatory endpoint 打开后不得反向改 proposal、split 或 primary metric；
- exploratory 结果单独目录，不能混入 confirmatory aggregate。

---

## 13. 人工评测协议

人工评测不是 H1 工程 gate 的替代，但可作为 external validity 的补充。

### 13.1 样本

- 从全 matched pool 分层随机抽样，不按 edit effect 或机器分数 top-k；
- 覆盖 A/C/D1/D2、3 scenes、5 proposal types、positive/negative/non-event、3 cameras、low/high
  reconstruction strata；
- 方法名、certificate、文件路径和自动分数对评审隐藏；
- 同一 case 的视频、depth、instance overlay、box overlay 分开提供，避免单张 RGB 掩盖标签错误。

### 13.2 Verdict 优先级

1. identity / source-removal failure；
2. collision / road / motion impossibility；
3. depth / occlusion order；
4. instance / box / typed-depth / evidence misalignment；
5. temporal or cross-camera inconsistency；
6. appearance artifact；
7. pass。

高优先级失败出现时，不能因“整体看起来不错”给 pass。

### 13.3 Agent 交付义务

请求用户开始评测前，必须同时在对话和仓库中给出完整提示词，包含：

- 目的与非目标；
- blind protocol；
- 禁止读取的信息；
- 素材范围；
- 每类 verdict 的定义、优先级和边界例；
- JSONL schema；
- 聚合阈值；
- 完成后的精确命令；
- verdict 对下一阶段的影响。

Agent 只生成空 JSONL 模板。用户或指定评审者填写后，才可用于 gate。

---

## 14. 单卡资源、存储与运行预算

### 14.1 已知资源事实

当前服务器审计：

- 3-camera、8-second B0 run 目录约 1.1–1.3 GiB/scene；
- 现有 C0 约 109 MiB；
- L0 约 8.5 MiB；
- O0 occupancy 约 7.1 MiB；
- `/root/autodl-tmp` 当前可用空间约 66 GiB；
- V7 文档记录 B0 训练峰值约 5 GB，未发现 OOM。

这些只用于初始预算，不代表 V7.1 新模块实际成本。

### 14.2 Phase budget

| Phase | GPU | 数据范围 | 硬限制 |
|---|---|---|---|
| EV-10 | CPU | 旧 evidence | 不改旧产物 |
| H1-11A/B | CPU 为主 | PILOT-3 | 全量 schema/tests 后再渲染 |
| H1-11C/D | 单 4090 | 30+ proposals，固定 render subset | peak `<22 GB` |
| H2-12A/B | 单 4090 | A/B strata + PILOT-3 | 只在 A 做 pseudo-hole；不下载大模型 |
| H2-12C | 单 4090 | unknown-only Gaussians | ray influence/transmittance/spill；peak `<22 GB` |
| H2-12D | 条件任务模板 | material residual only | 未触发则不创建 run；触发后先做权重/license/VRAM preflight |
| H3-13A | 单 4090 | smoke、单 seed | 只验证管线 |
| H3-13B | 单 4090 | scene-disjoint、≥3 seeds | 组间预算严格 matched |

### 14.3 磁盘规则

- 长 run 前至少保留 30 GiB；
- render 默认存压缩 PNG/NPZ 和必要 layers，不无界保存 debug tensor；
- checkpoint 采用明确 retention policy；
- raw evidence 和正式 summary 不自动删除；
- 不为了腾空间删除唯一证据；
- 大数据/权重下载前给出预计大小与清理方案，并等待授权。

---

## 15. 风险—设计—停止规则映射

| 风险 | V7.1 设计 | 失败时动作 |
|---|---|---|
| `RF-05` 合法点/轨迹不等于完整输出合法 | WorldState + synchronized render/labels + external evaluator | label/occlusion 不闭环则 H1 rejected |
| `RF-06` mask 不保证局部性 | geometry masks、ray influence、transmittance/depth-order、held-out views | known rays 退化则停止 H2 |
| `RF-08` machine evaluator 不是绝对真值 | continuous evaluator + blind human supplement | 不把 machine pass 写 human pass |
| `RF-09` same-scene 不等于可感知/可泛化 | scene-disjoint untouched-real H3 | mini-only 只写 pilot |
| `RF-16` controllability 不等于 physics/utility | H1 validity 与 H3 utility 分开 | 不用 adherence 代替 utility |
| `RF-18` action response 不足 | 明确 offline resimulator 定位 | 不重开闭环 world-model claim |
| `V7-RISK-01` occupancy 旁路 | safety/observation/render evidence 分别进入 certificate/state | 只加 post-hoc lookup 不算完成 |
| `V7-RISK-02` top-k machine screen | 全 proposal pool + blind stratified sample | 禁止只报 top-k |
| `V7-RISK-03` RGB-diff/Telea | geometry disocclusion + pseudo-hole | outside=0 不作为 pass |
| `V7-RISK-04` utility proxy | real-test task、matched synthetic groups | proxy 不过渡为论文结论 |
| `V7-RISK-05` provenance 缺失 | EV-10 fail-closed contract | 不补造旧字段 |
| `V7-RISK-06` 3 scenes/S1 弱 | quality strata、worst-case、scene expansion | 不删 S1 |
| `V7-RISK-07` 标签未闭环 | same-state label regeneration | 缺一类关键 label 即 fail closed |
| `V7-RISK-08` 坐标歧义 | explicit transforms + round-trip fixtures | audit 未过禁止 H1 |
| `V7-RISK-09` AABB/扁平 occupancy | continuous safety OBB + layered observation evidence | 假碰撞主导则重建证据 |
| `V7-RISK-10` unknown 过高 | real-control calibration + tri-state | yield 坍缩则 blocked，不改 unknown=free |
| `V7-RISK-11` reconstruction 混杂 | V0 regression、quality stratum | 先修/界定 reconstruction ceiling |
| `V7-RISK-12` 生成标签错配 | T3 appearance-only + 3D distillation gate | 不能回灌 3D 则 visualization only |
| `V7-RISK-13` synthetic shortcut | same-actor pairs + artifact/background/shuffle controls | 捷径可解释增益则 H3 rejected |
| `V7-RISK-14` hash 语义/稳定性不足 | world/render/artifact 三层 hash + canonical encoder | 任一 hash 不稳定则禁止正式 run |
| `V7-RISK-15` certificate/projector 混杂 | D1 固定 C trajectory；D2 单独 repair | 两类结果不合并 |
| `V7-RISK-16` proposal 无事件效果 | 纯 3D corridor/gap/TTC/label-transition gate | 无正负 pair 的 actor 不进入 H3 |
| `V7-RISK-17` depth 语义混淆 | expected/first-hit/LiDAR measured 分名与 truth tier | expected depth 不登记为 measured GT |
| `V7-RISK-18` pseudo-hole 外推 | A/B disocclusion strata | B 不计算伪 pixel GT |
| `V7-RISK-19` 新 Gaussian ray spill | influence/transmittance/depth-order/held-out spill | 任一 safeguard fail 则停止 H2-C |
| `V7-RISK-20` 自定义 H3 task | Task E + 标准 camera 3D detection Task S | 只过 E 仅允许 targeted claim |

---

## 16. 停止、转向与扩展规则

### 16.1 立即停止当前 phase

- WorldState 映射无法稳定覆盖 PILOT-3 每 scene 至少 2 actors；
- coordinate round trip 或三层 hash/canonicalization 不稳定；
- occupancy 只能通过 UNKNOWN→FREE 才有可用 yield；
- D1 修改了 C trajectory，或 D2 通过缩回 V0、破坏 scenario effect、拒绝大多数 proposal 获胜；
- renderer 和 label writer 无法由同一 state 驱动；
- geometry mask 对 pseudo-hole 不优于 RGB-diff；
- 在 B stratum 虚构 RGB GT；
- recovery 改善 RGB 但破坏 typed depth/instance/known-ray transmittance/depth order；
- 新 Gaussian held-out-camera spill 超阈值；
- 生成结果无法回灌 3D；
- 下游 gain 仅在 synthetic/seen scenes 上出现；
- background-only 或 label-shuffle control 显著高于 chance；
- H3 只用自定义 Task E 却试图形成通用数据效用 claim；
- 任何主结果依赖删 S1、top-k 或 future actor truth。

### 16.2 允许一次方法修复

只有明确定位到以下工程机制时允许一次修复：

- 坐标或 ID mapping bug；
- AABB voxelization 假冲突；
- renderer adapter 未同步；
- source-view leakage；
- threshold calibration 实现错误；
- metric aggregation bug。

修复必须新 run ID、新 fingerprint、保留旧失败证据。若同一机制再次失败，进入研究决策，不继续无界调参。

### 16.3 路线转向

| 结果 | 决策 |
|---|---|
| H1-CERT fail | 不声称 occupancy certificate 有检测增益；D2 结果不能替代 |
| H1-CERT pass、H1-PROJ fail | 只保留 certificate claim，不声称 repair/yield 增益 |
| H1-CERT/H1-PROJ fail | 停止 OccGS 方法 claim；保留 object-centric GS infrastructure |
| H1 pass、H2 unknown 很小 | 不实例化 H2-12D；父 summary 记 `not_triggered` |
| H1 pass、H2 geometry recovery pass | 主打 geometry-first recovery；不需要 diffusion |
| H1 pass、H2 residual material、H2-GEN pass | 加入 3D-distilled generative prior |
| H1 pass、仅 H3 Task E pass | 只声称 targeted cut-in utility |
| H1 pass、H3 Task S fail | 不形成通用 synthetic-data claim |
| H1 + H3 Task E/S pass | 解锁 scale 和 paper claim |

---

## 17. Claim ladder 与论文产物

### 17.1 Claim ladder

| Level | 所需证据 | 允许表述 |
|---|---|---|
| C0 | 现有 V7 feasibility | 单卡 object-centric GS edit 工程闭环可运行 |
| C1 | 11A–11C | 三层 hash 下的 Instance-Aware WorldState 可同步生成有限范围传感器与标签 |
| C2a | H1-CERT confirmatory | certificate 在固定 C trajectories 上发现额外违规 |
| C2b | H1-PROJ confirmatory | occupancy-aware repair 在保留事件效果时提高 usable yield |
| C3 | H2-12A/B/C | A/B provenance 可审计，geometry-first recovery 降低 unknown 且不污染 known rays |
| C4a | H3 Task E | certified edits 改善 targeted untouched-real cut-in task |
| C4b | H3 Task S | certified edits 在标准 camera 3D detection endpoint 上也有可复现收益 |
| C5 | H2-12D | 3D-distilled generative prior 在最佳几何方法上有额外收益 |

不能跳级。论文标题和摘要只使用已经达到的最高层。

### 17.2 推荐标题

主标题：

> **Visibility-Certified Counterfactual Resimulation with Instance-Aware Gaussian Scene Graphs**

若 H3 强、方法增量较保守：

> **Are Edited Neural Driving Simulators Valid Training Data?**

不推荐：

> Occupancy-Guided Diffusion Completion for 3DGS Driving Editing

因为 diffusion 不是 V7.1 必要贡献，且直接 2D completion 与统一几何/标签目标冲突。

### 17.3 预期主图表

- Figure 1：WorldState → certificate → synchronized render/labels → provenance；
- Figure 2：source actor、edited actor、new occlusion、known/unknown disocclusion；
- Figure 3：geometry-first recovery ladder；
- Table 1a：D1 certificate-only precision/recall/abstention；
- Table 1b：A/B/C/D2 projection、event retention 与 usable yield；
- Table 2：R0–R4/R5 的 A/B-stratified recovery、ray influence 与 multi-view；
- Table 3a：Task E 的 R/R+N/R+O/R+O+Rcv；
- Table 3b：Task S 标准 camera 3D detection；
- Table 4：coverage、UNKNOWN、failure、VRAM、runtime；
- Appendix：schema、run contract、threshold calibration、per-scene worst cases、human protocol。

所有图表必须由 frozen aggregate 自动生成。

---

## 18. Definition of Done

V7.1 只有在以下项目逐项完成后才算研究轮次完成：

### Evidence

- [ ] EV-10 retrospective index 完成；
- [ ] 新 run contract 和 tests 通过；
- [ ] 所有正式 run 有唯一 terminal marker；
- [ ] optional branch 未触发时不创建 run，父 summary 正确记录 `not_triggered`；
- [ ] 旧 evidence 未被改写。

### World state

- [ ] 坐标 convention 和 round-trip 通过；
- [ ] actor registry 一一映射；
- [ ] safety geometry、observation evidence、render support 职责分离；
- [ ] actor observation layer 可诚实 remove/insert；
- [ ] world/render/artifact 三层 hash 与 canonicalization 稳定；
- [ ] expected/first-hit/measured depth 与 limited semantic scope 固定；
- [ ] 3 cameras × full sequence 标签同步。

### H1

- [ ] ≥30 matched PILOT-3 proposals；
- [ ] scenario-effect gate 与 same-actor positive/negative pairs；
- [ ] A/B/C/D1/D2 全 pool aggregate；
- [ ] C/D1 trajectory hashes 100% 相同；
- [ ] D1 certificate 与 D2 repair 分别给 verdict；
- [ ] external evaluator 独立；
- [ ] usable yield、UNKNOWN、adherence、worst-scene 完整；
- [ ] pilot gate 有明确 verdict；
- [ ] confirmatory 前预注册 split 和 primary endpoint。

### H2

- [ ] geometry-derived masks；
- [ ] support-source、visibility-event、truth-tier maps；
- [ ] A/B disocclusion strata；
- [ ] 只在 A 执行 pseudo-hole benchmark；
- [ ] residual unknown materiality audit；
- [ ] R0–R4 完成或有明确不执行理由；
- [ ] added-Gaussian ray influence/transmittance/depth-order/spill safeguards；
- [ ] R5 只有触发时实例化；未触发不记 REJECTED；
- [ ] 正式输出统一 3D rerender。

### H3

- [ ] truth-tier-aware exporter；
- [ ] split/identity leakage tests；
- [ ] pipeline smoke；
- [ ] scene-disjoint real-only test；
- [ ] same-actor positive/negative 与 scene/actor balanced sampling；
- [ ] artifact/background-only/label-shuffle controls；
- [ ] R/R+N/R+O 等量、多 seed；
- [ ] Task E targeted event endpoint；
- [ ] Task S 标准 camera 3D detection endpoint；
- [ ] primary/secondary/strata/shortcut diagnostics；
- [ ] 正式 verdict。

### Research record

- [ ] `EXPERIMENTS.md` 登记数值与证据；
- [ ] `RESEARCH_STATUS.md` 登记当前状态、commit、证据和下一步；
- [ ] 新失败或风险写入 `RESEARCH_FAILURES.md`；
- [ ] 人工结果由用户/指定评审者填写；
- [ ] claim ladder 未越级；
- [ ] scale 决策有明确依据。

---

## 19. 最终决策图

```text
V7-EV-10
  │
  ▼
V7-H1-11A/B/C：WorldState + 3 evidence products + typed depth/limited labels
  │
  ├─ mapping / coordinate / label chain fail
  │      └─ BLOCKED/REJECTED，停止方法实验
  │
  ▼
V7-H1-11D：A/B/C/D1/D2 matched pilot
  │
  ├─ D1 不能在固定 C trajectory 上可靠检测
  │      └─ H1-CERT REJECTED/BLOCKED
  │
  ├─ D2 只靠拒绝/缩幅/破坏事件获胜
  │      └─ H1-PROJ REJECTED
  │
  ├─ D2 无 matched repair/yield 增益
  │      └─ 只保留通过的 certificate claim
  │
  ▼
V7-H2-12A：disocclusion + provenance materiality audit
  │
  ├─ residual unknown 不具实质影响
  │      └─ 不创建 12D run；父 summary = not_triggered
  │
  └─ residual unknown 具实质影响
         ▼
      H2-12B real reprojection
         │
         ├─ 足够 ─▶ 正式化 geometry-first recovery
         └─ 不足且有 geometry anchor
                ▼
             H2-12C 3D completion
                │
                ├─ 足够 ─▶ 正式化 3D recovery
                └─ 仍不足且满足全部触发条件
                       ▼
                    H2-12D generation → 3D distillation

H1 工程链完成后并行：
V7-H3-13A pipeline smoke
  │
  ▼
scene-disjoint data authorization
  │
  ▼
H1/H2/H3 confirmatory
  │
  ├─ 只过 Task E ─▶ targeted utility claim
  ├─ Task S fail ─▶ 不形成通用数据效用 claim
  └─ H1 + Task E/S pass
          ▼
      V7-SCALE-14
          ▼
      V7-PAPER-15
```

当前最高信息增益仍然是：

```text
V7-EV-10
→ V7-H1-11A/B/C/D1/D2
→ V7-H2-12A
```

不是下载 diffusion 权重，不是扩大 scene 数，也不是切换双卡。

---

## 20. 方法设计参考

以下工作用于限定设计空间，不表示 V7.1 必须复现全部方法：

1. [Street Gaussians](https://github.com/zju3dv/street_gaussians)：background + object-centric actor Gaussians。
2. [DriveStudio](https://github.com/ziyc/drivestudio)：驾驶 3DGS 数据与模型框架。
3. [FlexDrive](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_FlexDrive_Toward_Trajectory_Flexibility_in_Driving_Scene_Gaussian_Splatting_Reconstruction_CVPR_2025_paper.html)：几何优先的 out-of-path supervision。
4. [VAD-GS](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_VAD-GS_Visibility-Aware_Densification_for_3D_Gaussian_Splatting_in_Dynamic_Urban_CVPR_2026_paper.pdf)：visibility-aware densification。
5. [StreetCrafter](https://zju3dv.github.io/street_crafter/)：geometry-conditioned video generation 与 3DGS 优化。
6. [ReconDreamer](https://openaccess.thecvf.com/content/CVPR2025/html/Ni_ReconDreamer_Crafting_World_Models_for_Driving_Scene_Reconstruction_via_Online_CVPR_2025_paper.html)：生成先验回灌 reconstruction。
7. [SceneCrafter](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_SceneCrafter_Controllable_Multi-View_Driving_Scene_Editing_CVPR_2025_paper.html)：跨相机编辑与 empty-street prior。
8. [HorizonForge](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_HorizonForge_Driving_Scene_Editing_with_Any_Trajectories_and_Any_Vehicles_CVPR_2026_paper.html)：显式 3D edit 后的视频 renderer。
9. [GaussianEditor](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_GaussianEditor_Swift_and_Controllable_3D_Editing_with_Gaussian_Splatting_CVPR_2024_paper.html)：3D semantic tracing 与受控 Gaussian editing。
10. [Instruct-NeRF2NeRF](https://openaccess.thecvf.com/content/ICCV2023/html/Haque_Instruct-NeRF2NeRF_Editing_3D_Scenes_with_Instructions_ICCV_2023_paper.html)：render–edit–optimize，而非直接把单张 2D edit 当最终 3D 结果。

V7.1 从这些工作的共同经验中只采用一条原则：

> **先建立可解释、可渲染、可重生标签的统一三维状态；真实观测优先；生成先验最后使用，并接受三维一致性审计。**
