# WorldSim V6.6 — HARP-Compiler：物理伪影抑制与危险 Actor 保真编译器

> 英文工作名：**HARP-Compiler — Hallucination-Aware, Risk-Preserving World Compilation for RL-Ready Driving Simulation**
>
> 中文工作名：**HARP-Compiler：面向 RL 就绪驾驶仿真的幻觉感知、风险保真世界编译器**
>
> 状态：`active_p2_development_certificate`
>
> 上游冻结分支：`research/worldsim-v6.5-task-conditioned-authority`
>
> 上游终态提交：`288fa9f`
>
> 建议新分支：`research/worldsim-v6.6-harp-compiler`
>
> 默认资源：现有 AutoDL RTX 3090；物理状态和 Actor 实验默认单卡，只有视频修复或正式 RL 吞吐出现证据后才考虑多卡。

> 实际分支：`research/worldsim-v6.6-harp-compiler`；`main` 已于 2026-08-28 快进合入 V6.5 终态
> `288fa9f`，V6.6 再从该 `main` 建立。

> 执行裁剪：按用户授权跳过冗长 P0 校验和回归矩阵。第一轮直接执行 P1-D development mechanism atlas，
> 复用 V6.5 P10V 已消费的六场景 Actor evidence/native sidecar，只用于方法开发；fresh selection/confirmation
> 权限不会由该结果替代。V6.6 不新增 hash/checksum/fingerprint。

> P1-D 终态（2026-08-28）：409 actor-unit / 2,045 paired clusters / 8,180 rows；冻结 q0 的 paired
> artifact/hazard AUROC均为0.50，五类可观测 factor certificate recall均为1.0，clean-hazard false artifact=0，
> hazard-pair delta=0。该满分来自确定性注入的development construction，只授权P2接口固化与repair capability，
> 不授权自然artifact/fresh泛化claim。

---

# 0. 执行摘要

V6.6 不再把“危险”当成“无效”，也不再使用单一风险分数决定世界中哪些 Actor 应被保留。

V6.6 的核心问题是：

> **如何只移除或修复违反传感器证据、时空连续性、几何约束和动力学约束的伪影，同时完整保留真实、可追踪、物理可行但具有危险性的 Actor 与交互？**

北极星不是构造一个更安全、更容易的世界，而是构造一个：

```text
artifact-clean
+
hazard-preserving
+
actor-identity-preserving
+
reactive
+
deterministic-runtime
```

的 RL 训练世界。

核心二维分解：

|  | 低危险 | 高危险 |
|---|---|---|
| **物理合法** | 保留 | **必须保留并优先用于 RL** |
| **物理伪影** | 修复/删除/ABSTAIN | 修复/删除；不得作为“高价值危险样本” |

因此：

```text
Validity / Legitimacy
!=
Hazard / Difficulty
```

V6.6 只允许“合法性”决定修复、删除或拒绝；“危险度”只能决定采样、课程和评测权重。

---

# 1. V6.4–V6.5 终局复盘

## 1.1 必须先纠正一个结论：仓库没有真正完成正式 RL 实验

V6.4 P11 实际训练的是：

```text
10 维手工特征
→ 单层 Linear collision critic
```

并不是大型 NWM，也不是闭环 RL policy。

V6.5 也明确终止在：

```text
visited-state reliability evaluator
```

而不是 planner、policy 或 closed-loop RL。

所以现有证据应表述为：

> **下游 collision-critic proxy 没有从 verified generated data 获得有效增量。**

不能直接写成：

> V6.4/V6.5 已证明 RL policy 训练失败。

V6.6 必须先做 WorldSim 数据质量因果审计，再执行真正 matched RL 实验。

---

## 1.2 P11 的真正问题不是“明确删除了危险 Actor”，而是“验证分数与危险性近乎正交”

P11 的代码语义：

```text
actual_unsafe
=
候选 Ego corridor
与 TARGET_EVIDENCE 的 actor_swept_envelope 是否重叠
```

但 `verified_generated` 的排序分数是：

```text
m1_selected_route_mean_risk
=
V6.4 hidden-FREE q0 在路线附近的均值
```

二者分别回答：

```text
Actor 是否与 Ego action 冲突？
```

和：

```text
路线附近的 Occupancy 状态是否像 hidden-FREE 幻觉？
```

它们不是同一个变量。

P11 训练集统计：

```text
Real-only:
384 rows / 3 positives

Real + naive generated:
1152 rows / 191 positives

Real + verified generated:
768 rows / 96 positives
```

扣除 Real-only 后：

```text
naive generated:
188 / 768 = 24.48% unsafe

verified generated:
93 / 384 = 24.22% unsafe
```

所以当前证据不支持：

> q0 系统性把合法危险 Actor 筛掉了。

更精确的结论是：

1. q0 对 Actor 危险性基本无判别力；
2. verified arm 无差别砍掉约一半 generated rows；
3. 危险正例绝对数量从 188 减到 93，但危险比例几乎不变；
4. 世界状态验证没有改善 Actor hazard distribution，只降低了样本量和多样性；
5. critic 仍需从十个粗粒度标量中学习 Actor collision，表示明显不足。

这与用户判断并不冲突，而是把根因说得更准确：

> **当前系统没有显式区分“伪影”和“合法危险 Actor”，因此只能做危险性无感的粗暴删样本。**

---

## 1.3 V6.4 bake 丢失了已经存在的 Actor 结构

V6.4 `conditional_state_bake.py` 输出：

```text
native_indices
centers_m
risk_score
c0_state
m0_state
```

没有保留：

```text
actor_id
actor class
trajectory
lifecycle
track support
provenance
static/dynamic role
```

而 q0 学习目标只是：

```text
hidden-FREE conflict
```

所以 V6.4 物理包实际上是：

> **Actor-blind 的 voxel risk package。**

它无法表达：

- 这是一个真实加塞车的车体；
- 这是该车周围少量错误 Gaussian；
- 这是完全没有 Actor 支持的 ghost；
- 这是视觉重影，但物理 Actor 轨迹仍合法；
- 这是 Actor 本身合法，但轨迹违反动力学；
- 这是合法碰撞，不是“穿模伪影”。

---

## 1.4 项目其实已经具备 Actor-aware 基础资产，只是 V6.4/V6.5 没有把它们接回来

可直接复用的仓库资产包括：

### `worldsim_v62/evidence.py`

已经保存：

```text
actor_hit_indices / actor_hit_ids
actor_current_envelope_indices / ids
actor_swept_envelope_indices / ids
actor_envelope_indices / ids
```

而且静态点与 Actor 点已经分离。

### `worldsim_v62/query_dataset.py`

已经可以把 query 映射到：

```text
actor_id
actor_current_support
actor_swept_support
```

### `worldsim_v6/sceneir.py`

已有 Actor-centric SceneIR：

```text
actor ID
actor frame
actor trajectory
visibility
actor-owned chunks
provenance/support
```

并明确反对把 validity 压成单一标量，已有：

```text
q_photo
q_geometry
q_semantic
q_dynamics
```

### `worldsim_v6/r13_dynamic_edits.py`

已有：

```text
actor_remove
actor_trajectory_translation
actor_add_clone
collision label recomputation
unaffected actor/collision exactness audit
```

### 既有 cut-in / receiver / kinematic 审计

已有对：

```text
Actor identity continuity
receiver continuity
bumper gap
closing speed
TTC
path-clear
subject/receiver branch consistency
```

的较完整工具。

因此 V6.6 不需要从零写一个新 Actor 系统。正确做法是：

> **把 V6 早期 Actor/SceneIR/provenance 结构重新接到 V6.4 的 q0 与 V6.5 的 task query 上。**

---

# 2. 前沿研究动向与 V6.6 的迁移边界

## 2.1 Layered world 已成为 simulation-ready reconstruction 的主流接口

Instant NuRec、UniSim、RealEngine、DecoupleGS 等工作都把驾驶世界明确拆成：

```text
static background
+
individually editable dynamic actors
```

Instant NuRec 甚至直接输出 dynamic 3DGS layer 和 Actor trajectory。

V6.6 迁移：

> static artifact 与 Actor artifact 必须分别处理；Actor 不允许被压回无身份 voxel 集合。

不迁移：

> 不在 P0 直接更换 Instant NuRec 或其他 reconstruction backbone。

---

## 2.2 危险 Actor 是模拟器要保留和生成的目标，不是清洗对象

ReSim 和 ReinDriveGen 都把以下内容作为有价值的 simulator capability：

```text
hazardous non-expert actions
sudden cut-in
near collision
front collision
vehicle drifting
pedestrian jaywalking
cyclist crossing
```

它们的目标是提高这些 OOD 危险编辑的生成质量和可控性，而不是将其过滤掉。

V6.6 迁移：

> hazard score 用于保留、过采样和 curriculum；不得进入 artifact rejection gate。

---

## 2.3 “视觉好看”不能替代物理完整性

CrashTwin 指出：

```text
高感知质量
可能掩盖严重的物理违规
```

其评价维度包括：

```text
spatio-temporal consistency
momentum / kinetic-energy behavior
world-dynamics integrity
identity/appearance stability
```

V6.6 迁移：

> P2 先建显式物理证书和 violation benchmark，再训练任何 learned fixer。

不迁移：

> 不把 CrashTwin 的视频恢复 pipeline 直接搬进当前 3D/SceneIR 工程。

---

## 2.4 修复伪影应保持驾驶结构，而不是重写场景

OmniDreams 展示了 reconstruction fixer：

```text
remove blur / ghosting / missing regions / spurious geometry
while preserving scene layout, viewpoint, and driving-relevant structures
```

V6.6 迁移：

> appearance fixer 必须以已经冻结的 Actor ID、轨迹、mask 和物理 SceneIR 为条件；只能修像素，不得新增/删除 Actor 或改变轨迹。

但它是后期可选阶段，不是 V6.6 第一实验。

---

## 2.5 Instance identity 是动态世界保真的基本单位

ConsisDrive、InstaDrive 和 VISA 的共同趋势是：

```text
instance identity
track
trajectory mask
track-to-voxel grounding
instance-level audit
```

比匿名 voxel 或全局 caption alignment 更适合动态驾驶世界。

V6.6 迁移：

> artifact / legitimacy 预测以 Actor track 为一级单位，以 Actor-owned primitive 为二级单位。

---

## 2.6 RL-ready WorldSim 还必须具备 reactive agents

ReactSim-Bench 明确指出：

```text
log replay realism
!=
reactive capability
```

当 Ego 偏离日志后，周围 Actor 应能够合理跟车、避让、制动或保持规则，而不是继续机械播放日志轨迹。

V6.6 迁移：

> 物理伪影清理通过后，再单独验证 Actor 对 off-log Ego action 的反应。

不迁移：

> 不在 V6.6 第一轮直接训练大型 diffusion behavior model。

---

# 3. V6.6 北极星与数学接口

## 3.1 新世界状态

定义：

\[
\mathcal W_t^{66}
=
\left(
\mathcal G_t^{app},
\mathcal O_t^{static},
\mathcal A_t,
\mathcal E_t^{artifact},
\mathcal P_t,
\mathbf q_t
\right)
\]

其中：

- \(\mathcal G_t^{app}\)：外观层；
- \(\mathcal O_t^{static}\)：静态物理状态；
- \(\mathcal A_t=\{A_i\}\)：有身份的动态 Actor 集；
- \(\mathcal E_t^{artifact}\)：物理/时间/渲染伪影场；
- \(\mathcal P_t\)：来源和观测支持；
- \(\mathbf q_t\)：分解后的 validity，而不是单标量。

Actor：

\[
A_i=
\left(
ID_i,
C_i,
G_i,
X_{i,0:H},
V_{i,0:H},
L_i,
S_i,
p_i^{legit},
h_i^{hazard}
\right)
\]

分别表示：

- 身份；
- 类别；
- 几何壳/owned primitives；
- 轨迹；
- 速度/动力学；
- lifecycle；
- sensor/provenance support；
- Actor 合法性；
- 危险度。

---

## 3.2 Validity 与 Hazard 两条正交轴

### 合法性/伪影轴

\[
p_i^{artifact}
=
P(A_i\text{ or its primitives violate evidence/physics})
\]

分解为：

\[
\mathbf e_i=
(
 e_i^{sensor},
 e_i^{free},
 e_i^{ghost},
 e_i^{flicker},
 e_i^{teleport},
 e_i^{kinematic},
 e_i^{identity},
 e_i^{penetration},
 e_i^{render}
)
\]

### 危险轴

\[
h_i^{hazard}
=
H(
TTC,
cut\text{-}in,
closing\ speed,
collision,
near\ miss,
VRU\ crossing,
interaction
)
\]

重要约束：

\[
\frac{\partial p^{artifact}}{\partial h^{hazard}}
\approx 0
\]

含义：

> 把同一辆合法车从正常跟车改成合法加塞，不应该让 artifact score 上升。

反过来：

> 给同一场景注入 teleport/duplicate/flicker ghost，应提高 artifact score，但不改变原本设计的 hazard label。

该公式是语义合同，不预先限定用什么网络实现。

---

## 3.3 Compiler 决策

每个 Actor 或 primitive 输出：

```text
KEEP
REPAIR
ABSTAIN_LOCAL_GEOMETRY
DROP_ARTIFACT_PRIMITIVE
DROP_ARTIFACT_ACTOR
```

硬规则：

1. 已有真实 track/provenance/lifecycle 支持的 Actor，默认保留 Actor existence；
2. Actor 局部 geometry 不可靠时，优先修复或局部 UNKNOWN，不删除 Actor；
3. cut-in、hard brake、collision、negative TTC、near miss 本身都不是 artifact；
4. 只有缺乏 Actor support 且违反证据/时空/几何/动力学时，才允许删除 Actor；
5. appearance artifact 不得修改 physical state；
6. hazard score 只能控制采样权重，不能控制 DROP。

---

# 4. V6.6 研究问题

1. 当前 q0 对“伪影”和“合法危险 Actor”到底有多大混淆？
2. 不训练网络，仅用 provenance、track、kinematic、geometry 证书能否可靠区分二者？
3. learned model 是否能在不增加危险 Actor 误删的条件下，提高 artifact detection？
4. 对伪影，REPAIR 是否优于 DROP 和 all-UNKNOWN？
5. 伪影清理后，hazard distribution 是否保持，而不是变成 easier world？
6. 当 Ego 偏离日志时，周围 Actor 是否具备合理 reactive behavior？
7. 在 matched hazard distribution 下，HARP world 是否真正改善 RL，而不是只改善感知指标？

---

# 5. 数据协议：Factorial Validity × Hazard Benchmark

## 5.1 四象限配对样本

每个 base case 构造最多四种配对版本：

```text
V0-H0：合法 + 普通
V0-H1：合法 + 危险
V1-H0：伪影 + 普通
V1-H1：伪影 + 危险
```

其中：

- `V0`：物理合法；
- `V1`：注入已知 artifact；
- `H0`：低危险；
- `H1`：合法危险交互。

同一 base case 的四个版本必须属于同一统计 cluster，不能当作四个独立 scene 样本。

---

## 5.2 合法危险编辑

优先复用已有 Actor editing / cut-in 工具，构造：

- 合法加塞；
- 合流；
- 前车急刹；
- 侧向侵入；
- VRU crossing；
- cyclist crossing；
- narrow-gap passing；
- near collision；
- physically valid collision。

每种编辑必须先过：

```text
track continuity
map compliance
velocity/acceleration/jerk
identity/lifecycle
actor geometry
unaffected actors exactness
```

没有通过物理证书的“危险编辑”不能作为合法危险样本。

---

## 5.3 Artifact 注入族

只允许预注册的、可审计的 deterministic corruption：

### Geometry

- observed-FREE ghost；
- actor duplicate shell；
- floating actor/static blob；
- scale/shape jump；
- unsupported occupied island；
- interpenetration without corresponding contact response。

### Temporal

- one-frame pop/flicker；
- teleport；
- track ID switch；
- lifecycle break；
- duplicate track。

### Kinematics

- velocity discontinuity；
- acceleration/jerk discontinuity；
- impossible heading jump；
- collision without momentum/contact response。

### Appearance-only

- render ghost；
- blur；
- missing region；
- identity appearance drift。

Appearance-only 注入不能改变 physical state label。

---

## 5.4 Fresh cohort

推荐正式 cohort：

| Role | fresh scenes | base units |
|---|---:|---:|
| D-Train | 12 | 144 |
| S-Separation Selection | 6 | 72 |
| R-Repair Selection | 6 | 72 |
| C-Calibration | 6 | 72 |
| H-RL Confirmation | 6 | 72 |
| T-Exact Test | 6 | 72 |
| **总计** | **42** | **504** |

训练/选择阶段可以生成四象限配对变体；正式统计单位仍是 base case / scene cluster。

V6.1–V6.5 所有 quality-read scenes：

```text
Tier L — mechanism / regression only
```

不得进入 V6.6 selection/calibration/confirmation/test。

---

# 6. Baseline Staircase

```text
B0  V6.4 q0 hidden-FREE score
B1  V6.4 q0 + actor support lookup（不改变分数）
B2  Deterministic factorized certificate
A0  Low-capacity actor artifact head
A1  Actor-set / temporal interaction head（条件解锁）
R0  DROP artifact baseline
R1  LOCAL UNKNOWN / ABSTAIN
R2  Physical repair
R3  Appearance fixer（条件解锁）
H0  Hazard-preserving HARP bake
X0  Reactive actor replay（条件解锁）
RL0 Real-only
RL1 Real + naive WorldSim
RL2 Real + q0-filtered WorldSim
RL3 Real + HARP WorldSim
RL4 Real + HARP-reactive WorldSim（条件解锁）
```

每一步只回答一个问题。

---

# 7. P0 — Inheritance、代码审计和协议冻结

Task：

```text
WS-V66-P0-HARP-SCOPE-01
```

必须完成：

1. 从 V6.5 terminal commit 创建新分支；
2. 冻结 V6.4/V6.5 terminal claim；
3. 不重开 Surface、CPSC、GMM UQ、trajectory residual、Actor-time cost、learned admission、direct action authority；
4. 建立 `V66_ACTOR_ASSET_AUDIT.md`；
5. 建立 `V66_SOURCE_LICENSE_MATRIX.md/json`；
6. 将旧 Actor/SceneIR/provenance/r13 edits 接口列为复用资产；
7. 建立 `V66_VALIDITY_HAZARD_TAXONOMY.yaml`；
8. 冻结 fresh cohort 和 exposure ledger；
9. 冻结 artifact injection family 和 hazard edit family；
10. 冻结所有 gate 和 stop rule。

P0 不做 GPU 正式实验。

---

# 8. P1 — Separation Atlas：先证明问题真实存在

Task：

```text
WS-V66-P1-VALIDITY-HAZARD-SEPARATION-ATLAS-01
```

P1 不训练复杂网络，只评估：

- q0；
- entropy/margin；
- Actor support count；
- provenance；
- track length；
- lifecycle；
- velocity/acceleration/jerk；
- shape consistency；
- free-space conflict；
- identity continuity。

必须输出二维矩阵：

```text
artifact score × hazard score
```

主要问题：

1. q0 对四象限是否可分？
2. q0 是否对危险编辑近似不变？
3. 哪些 artifact family q0 能发现？
4. 哪些合法危险 Actor 会被 q0 高分误伤？
5. P11 的 verification score 与 actor hazard 的 Spearman/AUROC 到底多低？

P1 是诊断，不宣称方法成功。

### P1 进入 P2 的最低条件

- 每个 artifact/hazard stratum 样本充足；
- Actor ID 和 paired lineage 可审计；
- 四象限没有 role leakage；
- 不把 corruption metadata 暴露给 inference baseline；
- 至少存在一种 q0 未解决但 deterministic feature 有 signal 的 artifact family。

如果不存在任何可辨识 artifact signal：

```text
关闭 V6.6 learned branch
回到数据/真值设计
```

---

# 9. P2 — Deterministic Actor Legitimacy / Physics Certificate

Task：

```text
WS-V66-P2-FACTOR-CERTIFICATE-01
```

先不训练网络。

每个 Actor 生成：

\[
\mathbf C_i=
(
C_i^{provenance},
C_i^{track},
C_i^{lifecycle},
C_i^{sensor},
C_i^{geometry},
C_i^{kinematic},
C_i^{map},
C_i^{contact},
C_i^{render}
)
\]

每个 factor 输出：

```text
PASS / FAIL / UNKNOWN
```

Actor 总状态：

```text
LEGITIMATE
ARTIFACT
UNKNOWN
```

但不允许简单多数票；必须保留 reason codes。

### 证书原则

- negative TTC 不是 FAIL；
- cut-in 不是 FAIL；
- collision 不是 FAIL；
- 大加速度只有在 class/scene/interaction 不支持时才是 FAIL；
- observed raw hits、稳定 track 和 lifecycle 是 Actor existence 的强保护证据；
- appearance drift 只能让 `C_render` FAIL，不能直接让 Actor existence FAIL；
- 物理壳不确定时，Actor existence 保留，owned geometry 可 UNKNOWN。

### P2 Gate

相对 B0 q0：

```text
pooled artifact recall >= 0.70
每个 sample-sufficient artifact family recall >= 0.55
legitimate hazardous Actor retention >= 0.97
每个 hazard stratum retention >= 0.90
clean-hazard false artifact <= 0.05
Actor ID/lifecycle retention = 1.0
hazard edit 前后 artifact score delta <= 0.03
hard observed evidence violations = 0
```

若 deterministic certificate 无法通过：

> 不训练 learned model；先修 taxonomy/label，而不是堆网络。

---

# 10. P3 — Learned Artifact / Legitimacy Model

Task：

```text
WS-V66-P3-LEARNED-ACTOR-ARTIFACT-01
```

只在 P2 有效后解锁。

## 10.1 A0：低容量 Actor Summary Head

输入只包含方法时刻可见信息：

```text
actor class
box/shape statistics
track length
sensor hit support
current/swept support
position / velocity / acceleration / jerk
heading continuity
shape consistency
free-space conflict
local q0 distribution
provenance type
certificate factors
```

输出：

```text
p_artifact
p_legitimate
factor reason heads
```

第一版：

```text
small MLP / monotone residual
```

不使用 scene ID，不使用 hazard label，不使用大 Transformer。

## 10.2 Pair-invariance training

对同一 Actor：

```text
benign trajectory
vs
physically valid cut-in trajectory
```

约束 legitimacy 表示保持一致。

对同一 base case：

```text
clean
vs
ghost/flicker/teleport corruption
```

约束 artifact 表示分离。

可以使用：

```text
supervised contrastive / pairwise margin
+
factor classification
```

但 P1 只冻结一种正式 loss。

## 10.3 A1 条件解锁

只有 A0 出现：

- train 信号明确；
- selection scene gap；
- 错误集中在多 Actor interaction / temporal identity；

才允许：

```text
Deep Sets / Set Transformer
或
低容量 temporal actor graph
```

二者只能选一个正式 arm。

### P3 Gate，相对 P2 deterministic

```text
artifact AUPRC >= +0.05 absolute
artifact AUROC >= +0.03 absolute
>=5/6 selection scenes support
任何 artifact family 不恶化 >5%
clean-hazard false artifact <=3%
legitimate hazardous Actor retention >=97%
hazard-pair artifact-score delta <=0.03
Actor ID/lifecycle exact
```

不超过 deterministic baseline：

```text
关闭 learned artifact family
保留 deterministic HARP
```

不做 seed/width/depth sweep。

---

# 11. P4 — Repair-first Compiler：优先修复，不优先删除

Task：

```text
WS-V66-P4-ARTIFACT-REPAIR-01
```

三臂 matched：

```text
R0 DROP
R1 LOCAL UNKNOWN / ABSTAIN
R2 REPAIR
```

## 11.1 物理状态修复优先级

### Actor shell repair

- 用稳定 track 与 canonical actor box 恢复局部 owned geometry；
- 删除无 support 的 duplicate/ghost primitive；
- 不删除 Actor ID；
- 轨迹保持冻结。

### Temporal repair

- 用相邻合法帧插值填补 one-frame pop；
- 修复 ID switch/lifecycle break；
- 不平滑真实急刹、加塞或碰撞。

### Static repair

- observed-FREE ghost 删除；
- unsupported island → UNKNOWN；
- static/Actor ownership 冲突重分配；
- provenance 不足处 abstain。

### Kinematic repair

只有 corruption oracle 明确时允许恢复轨迹；自然场景若无法确认：

```text
UNKNOWN_DYNAMICS
```

而不是自动“平滑成正常驾驶”。

### P4 Gate

```text
artifact violation count -50% relative
clean hazardous Actor retention >=98%
Actor track/trajectory/ID exact for retained actors
hazard event count/stratum shift <=2%
TTC/closing-speed distribution matched
non-artifact geometry regression <=2%
hard observed evidence violations=0
```

如果 DROP 通过而 REPAIR 不通过：

- 可以保留 DROP 作为 baseline；
- 不能称 hazard-preserving compiler 成功。

---

# 12. P5 — Appearance Fixer（条件解锁）

Task：

```text
WS-V66-P5-APPEARANCE-FIXER-01
```

只有 P4 physical state 通过，且剩余主要错误是：

```text
blur
render ghost
missing region
appearance identity drift
```

才解锁。

输入条件：

```text
rendered frame
frozen camera pose
frozen physical depth/geometry
actor masks
actor IDs
actor trajectories
scene layout
```

输出只允许修改 RGB/appearance。

禁止：

- 新增 Actor；
- 删除 Actor；
- 改 trajectory；
- 改 collision state；
- 改 physical occupancy；
- 用 prompt 把危险行为重写成安全行为。

先做轻量 paired fixer capability；只有有 signal 才考虑 OmniDreams/ReconDreamer-style diffusion fixer。

### P5 Gate

```text
render artifact rate -40%
actor count/ID/trajectory exact
layout/viewpoint exact
physical state exact
hazard event exact
perception downstream non-regression
```

---

# 13. P6 — HARP SceneIR Bake

Task：

```text
WS-V66-P6-HARP-BAKE-01
```

输出 package：

```text
STATIC_STATE.npz
ACTORS.jsonl
ACTOR_PRIMITIVES.npz
ARTIFACT_FACTORS.jsonl
REPAIR_LOG.jsonl
HAZARD_ATTRIBUTES.jsonl
PROVENANCE.jsonl
RUNTIME_MANIFEST.json
```

每个 Actor 至少包含：

```text
actor_id
class
track/lifecycle
trajectory
owned primitive indices
sensor/provenance support
legitimacy state
artifact factor vector
hazard attributes
repair action
```

Runtime 必须：

- 不加载 learned artifact model；
- 不加载 hidden target；
- 不读取 hazard label 决定 Actor existence；
- 显式、可查询、确定性；
- 每次 replay 返回同一 state；
- Actor 和 static layers 分离；
- physical state 与 appearance state 分离。

---

# 14. P7 — Hazard-Preserving World Distribution

Task：

```text
WS-V66-P7-HAZARD-PRESERVING-DISTRIBUTION-01
```

生成世界的采样权重：

\[
w(c)
\propto
I[valid(c)]
\cdot
(1+\lambda h(c))
\cdot
D(c)
\]

其中：

- `valid`：物理合法；
- `h`：危险度；
- `D`：场景多样性；
- hazard 越高不能被自动降权。

matched arms：

```text
N0 naive replay
Q0 q0-filtered replay
D0 deterministic HARP
L0 learned HARP（若P3通过）
O0 oracle injected-clean benchmark
```

必须在**同样 hazard actor / TTC / cut-in / VRU 分布**下比较 artifact rate。

### P7 Gate

```text
artifact rate -50% vs naive
hazardous actor retention >=97%
hazard event count shift <=2%
TTC/near-miss/cut-in distribution matched
actor identity switch -50%
teleport/flicker/float/interpenetration -50%
world yield >0
no all-UNKNOWN / easy-world collapse
```

只有 P7 通过，才允许称：

```text
RL-ready physical replay candidate
```

---

# 15. P8 — Reactive Actor Capability（条件解锁但原则上必要）

Task：

```text
WS-V66-P8-REACTIVE-ACTOR-01
```

原因：

> artifact-clean 的 log replay 仍然不是闭环 WorldSim。

Ego 一旦偏离日志，其他车若继续机械回放，会产生新的非物理碰撞或穿越，这些同样会污染 RL。

增量阶梯：

```text
X0 logged non-reactive replay
X1 deterministic kinematic response
X2 learned reactive response（条件解锁）
```

X1 先覆盖：

- lead/follow braking；
- lane keeping；
- collision avoidance；
- yield / maintain；
- bounded acceleration/jerk；
- map compliance。

X2 只有 X1 出现明显行为 ceiling 时才执行。

采用 ReactSim-style protocol：

```text
AV action 独立输入
surrounding agents 单独模拟
AV 轨迹偏离日志
```

### P8 Gate

```text
collision response validity
map compliance
kinematic feasibility
actor identity/lifecycle exact
hazard interaction retained
response latency plausible
>=5/6 scenes support
```

如果只能做 non-reactive replay：

> 允许用于有限 counterfactual perception/log replay，不允许宣称闭环 RL simulator。

---

# 16. P9 — 真正的 RL 实验

Task：

```text
WS-V66-P9-MATCHED-RL-01
```

这是 V6.4/V6.5 没有做过的正式实验。

## 16.1 先做 artifact exploitation probe

在训练 policy 前，固定一个 planner/heuristic，评估它是否会：

- 钻 ghost free-space；
- 对 duplicate Actor 过度制动；
- 利用 teleport/flicker；
- 对 render-only ghost 产生物理反应；
- 因合法危险 Actor 被删而获得虚假高 reward。

输出：

```text
Artifact Exploitation Score
```

## 16.2 Matched RL Arms

同一 RL 算法、网络、预算、seed contract：

```text
RL0 Real-only
RL1 Real + naive replay
RL2 Real + q0-filtered replay
RL3 Real + deterministic HARP
RL4 Real + learned HARP（若P3通过）
RL5 Real + HARP reactive（若P8通过）
```

禁止不同 arm 使用不同：

- policy architecture；
- reward；
- optimizer budget；
- hazard curriculum；
- scene denominator。

## 16.3 训练 distribution 必须 matched

所有 WorldSim arms：

- 相同 base logs；
- 相同 Actor IDs；
- 相同 hazard edits；
- 相同 TTC/cut-in/collision strata；
- 只改变 artifact treatment / reactivity。

否则 RL 增益无法归因。

## 16.4 指标

Primary：

- collision / near-miss；
- unsafe-action recall；
- safe-action precision；
- route progress；
- stuck；
- comfort；
- closed-loop completion；
- off-log Actor response；
- artifact exploitation。

分离报告：

```text
Hazard Competence
Artifact Robustness
Reactive Generalization
```

### P9 Gate

HARP 必须同时：

```text
优于 naive replay
优于 q0-filtered replay
hazard competence 不降低
artifact exploitation 明显降低
progress/stuck 过门
independent confirmation 支持
```

如果 HARP simulator 质量通过但 RL 无增量：

> 如实报告 simulator improvement 没有转化为 RL gain；不修改 artifact gate 救 RL。

---

# 17. Confirmation / Exact Test

## P10 — One-shot Confirmation

```text
WS-V66-P10-CONFIRMATION-01
```

冻结：

- compiler；
- repair policy；
- hazard sampling；
- reactive model；
- RL policy；
- thresholds；
- cohort。

一次读取。

## P11 — Exact-once Test

```text
WS-V66-P11-EXACT-ONCE-TEST-01
```

只有 P10 通过后执行。

测试必须同时包含：

- natural reconstruction artifacts；
- deterministic injected artifacts；
- clean hazardous actors；
- artifact + hazardous paired cases；
- off-log Ego actions；
- reactive Actor pressure。

---

# 18. 关键 Gate 总表

| Phase | 生死门 |
|---|---|
| P1 | 四象限数据和 paired lineage 可审计；存在可辨识 artifact signal |
| P2 | artifact recall≥0.70；hazard Actor retention≥0.97；false artifact≤0.05 |
| P3 | 相对 deterministic AUPRC +0.05，且危险 Actor retention 不退化 |
| P4 | artifact -50%，hazard retention≥0.98，hazard distribution shift≤2% |
| P5 | RGB artifact -40%，physical/Actor state exact |
| P7 | artifact -50%，hazard actor/event 保真，no easier-world collapse |
| P8 | off-log Actor 反应在 collision/map/kinematic 上过门 |
| P9 | RL 同时改善 artifact robustness 与 hazard competence |
| P10/P11 | one-shot / exact-once；失败不重跑 |

---

# 19. Stop Rules

## Stop 1 — Taxonomy 不可辨识

如果 deterministic injection 与自然样本无法形成可信 validity label：

```text
停止 learned branch
先修数据定义
```

## Stop 2 — Deterministic certificate 误删危险 Actor

如果：

```text
hazard retention <0.97
或 clean-hazard false artifact >0.05
```

关闭当前 certificate family，不训练网络掩盖问题。

## Stop 3 — Learned model 不超过 deterministic

关闭 learned family，保留 deterministic HARP。

## Stop 4 — Repair 变成 easier-world

如果 hazard distribution、Actor count、TTC 或危险事件被明显削弱：

关闭 repair candidate。

## Stop 5 — Appearance fixer 改物理状态

任何 Actor/trajectory/occupancy 变化：

```text
rejected
```

## Stop 6 — Reactive model 不物理

不进入 closed-loop RL claim；保留有限 replay 用途。

## Stop 7 — RL 无增量

不做 reward/threshold/scene sweep；保留 WorldSim 质量结论，RL claim 关闭。

## Stop 8 — Confirmation/Test 失败

不换场景，不降 gate，不重跑。

---

# 20. 禁止事项

- 用 hazard score 删除 Actor；
- 把 cut-in/collision/near-miss 当 artifact；
- Actor geometry 不可靠时直接删 Actor existence；
- 把 q0 继续作为唯一 validity；
- 把 q0-filtered world 称为“安全世界”；
- 用 P11 linear critic 结果冒充 RL；
- 先上 diffusion fixer，再修物理状态；
- 在 selection 上扫 artifact threshold；
- seed/model-size/temporal-window sweep；
- 回开 V6.3 Surface、V6.4 UQ、V6.5 trajectory residual；
- 用 pooled artifact 指标掩盖 clean-hazard Actor 误删；
- 用 all-UNKNOWN 或删除所有危险 Actor 获得低 collision；
- 在不同 RL arms 改 hazard distribution；
- 只在训练 simulator 内评测 RL policy。

---

# 21. 代码结构建议

```text
docs/WORLDSIM_V6_6_HARP_COMPILER_PLAN.md

docs/autoresearch/worldsim_v66/
  AUTORESEARCH_STATE.current.json
  HYPOTHESES.jsonl
  REFLECTIONS.jsonl
  USED_SCENE_LEDGER_V66.json
  SELECTION_EXPOSURE_LEDGER.json
  V66_SOURCE_LICENSE_MATRIX.md
  V66_ACTOR_ASSET_AUDIT.md
  V66_VALIDITY_HAZARD_TAXONOMY.md

configs/worldsim_v66/
  p0_scope_v1.yaml
  p1_factorial_atlas_v1.yaml
  p2_factor_certificate_v1.yaml
  p3_actor_artifact_v1.yaml
  p4_repair_v1.yaml
  p5_appearance_fixer_v1.yaml
  p6_harp_bake_v1.yaml
  p7_hazard_distribution_v1.yaml
  p8_reactive_actor_v1.yaml
  p9_matched_rl_v1.yaml
  p10_confirmation_v1.yaml
  p11_test_v1.yaml

motion_proj/worldsim_v66/
  taxonomy.py
  actor_state.py
  actor_grounding.py
  artifact_injection.py
  hazard_edits.py
  physics_certificates.py
  artifact_model.py
  paired_invariance.py
  physical_repair.py
  appearance_repair.py
  harp_bake.py
  hazard_sampler.py
  reactive_actor.py
  artifact_exploitation.py
  rl_protocol.py

scripts/
  run_worldsim_v66_p1_atlas.py
  run_worldsim_v66_p2_certificate.py
  run_worldsim_v66_p3_artifact_model.py
  run_worldsim_v66_p4_repair.py
  run_worldsim_v66_p6_bake.py
  run_worldsim_v66_p7_distribution.py
  run_worldsim_v66_p8_reactive.py
  run_worldsim_v66_p9_rl.py
  run_worldsim_v66_p10_confirmation.py
  run_worldsim_v66_p11_test.py
```

优先复用：

```text
motion_proj/worldsim_v62/evidence.py
motion_proj/worldsim_v62/query_dataset.py
motion_proj/worldsim_v6/sceneir.py
motion_proj/worldsim_v6/sceneir_adapters.py
motion_proj/worldsim_v6/r5_provenance.py
motion_proj/worldsim_v6/r13_dynamic_edits.py
现有 cut-in / receiver / kinematic audit
V6.4 q0 model
V6.5 visited-state evaluator（仅作 diagnostic）
```

---

# 22. 资源与流水线

## 22.1 单卡策略

P0–P4：

```text
1× RTX 3090 足够
```

Actor summary/certificate 主要是 CPU + 小 GPU。

P5 appearance fixer：

- 先单卡 capability；
- 若单卡可放但吞吐慢，可 2–4 卡 scene parallel；
- 不因显存降低分辨率、ROI、Actor 数或时间窗。

P8 reactive / P9 RL：

- 先单卡 small-scale capability；
- 只有正式 rollout 吞吐成为瓶颈才加卡；
- 资源 blocked 与科学 rejected 分开。

## 22.2 I/O

必须继承 V6.5 已验证的：

```text
archive shard targeted scan
scene-ready preprocess
2× CPU producer
2× native GPU worker
partial evidence reuse
raw cleanup after canonical publish
```

不再回到整批 I/O barrier。

---

# 23. 第一轮执行顺序

```text
P0  inheritance + Actor asset audit + taxonomy freeze
 ↓
P1  factorial validity×hazard atlas
 ↓
P2  deterministic factor certificate
 ↓
     FAIL → 修 taxonomy/data，不训练网络
     PASS
 ↓
P3  low-capacity learned artifact model
 ↓
     无增量 → 保留 deterministic HARP
     有增量
 ↓
P4  DROP vs ABSTAIN vs REPAIR
 ↓
P6  actor-preserving HARP bake
 ↓
P7  matched hazard-distribution audit
 ↓
     FAIL → 不进入 RL
     PASS
 ↓
P8  reactive actor capability
 ↓
P9  actual matched RL experiment
 ↓
P10 one-shot confirmation
 ↓
P11 exact-once test
```

P5 appearance fixer 是旁支：只有 physical repair 通过且仍有 RGB artifact 时插入 P4→P6 之间。

---

# 24. V6.6 预期论文贡献

如果全部阶段成立，可能形成：

1. **Validity–Hazard Factorization**
   - 将物理合法性与任务危险度显式解耦。

2. **Actor-Preserving Artifact Compiler**
   - 修复 Actor-owned geometry/appearance，而不删除合法危险 Actor。

3. **Factorial WorldSim Benchmark**
   - 同一 base case 的合法/伪影 × 普通/危险配对评价。

4. **Hazard-Preserving World Distribution**
   - 保持 TTC、cut-in、near-miss、collision 等危险分布，只降低 ghost/teleport/flicker/float 等伪影。

5. **Reactive RL-Ready Simulation**
   - 分离 Ego 控制与周围 Actor 响应，真正执行 matched RL。

如果 learned model 不超过 deterministic：

- 不主张 learned artifact detector；
- factorized certificate、repair compiler 和 matched RL protocol 仍可形成技术报告。

如果 RL 无增量：

- 不主张 RL contribution；
- 仍可主张 simulator artifact reduction 与 hazard-preserving fidelity。

---

# 25. 一句话主张

中文：

> **我们不把危险世界清洗成安全世界，而是将物理伪影与合法危险交互显式解耦，只修复违反证据和动力学的 hallucination，同时保留并强化可追踪、可反应的危险 Actor，构建真正可用于强化学习的驾驶世界。**

英文：

> **We do not sanitize dangerous worlds into easy ones; we factor physical artifacts from legitimate hazards, repair only evidence- or physics-violating hallucinations, and preserve track-consistent, reactive safety-critical actors for RL-ready driving simulation.**
