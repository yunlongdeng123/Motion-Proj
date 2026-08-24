# WorldSim V6.2 — Constraint-Aware Physical State Completion

> 中文名：**约束感知的物理状态补全**
>
> 方法工作名：**CPSC（Constraint-Aware Physical State Completion）**
>
> 推荐论文题目：
>
> **CPSC: Hard-Evidence-Constrained Physical State Completion for Verifiable Driving World Compilation**
>
> 中文表述：
>
> **CPSC：面向可验证自动驾驶世界编译的硬证据约束物理状态补全**
>
> 研究版本：`WorldSim V6.2`
>
> 默认资源：单卡 RTX 3090 24GB
>
> 起点：V6.1 最小实验负结果之后的新研究路线
>
> V6.1 终态必须保持不可变：
>
> ```text
> v61_minimum_experiment_closed_negative
> ```
>
> V6.2 不是 V6.1 的结果后修补，也不是继续遍历新的 Occupancy 后端；它提出一个新的、可训练的方法问题：
>
> > **能否把真实 FREE/OCC 观测作为不可违反的硬约束，把 learned Occupancy 作为可被推翻的软先验，并通过保守的 UNKNOWN 与集合式预测，将感知型 Occupancy 转换为适合 LogSim、WorldSim 和 NWM 强化学习的物理状态？**
>
> 执行模式：Autoresearch；普通科研分支、工程恢复、候选淘汰与反思无需人工逐项审批。
>
> 数据纪律：development → calibration → freeze → one-shot confirmation → freeze → exact-once test。
>
> 核心安全纪律：**方法侧证据与独立评测证据严格分离；任何 hidden evaluator 不得进入方法输入、候选选择或阈值拟合。**

---

# 0. 结论先行

V6.1 已经得到三条足够清晰的研究事实：

```text
Oracle Occupancy
→ 10/28 ACCEPT
→ 0 false-safe
→ 物理状态补全具有真实上界

Hunyuan3D 四个固定 Actor 提案臂
→ 0/6
→ 三维生成表面仍侵入 observed FREE

GaussianWorld / IR-WM
→ 都能达到 10/28 的表面支持
→ 但 accepted cases 全部 false-safe
→ learned argmax Occupancy 不能直接充当安全权威
```

因此 V6.2 的问题不再是：

```text
再换一个 Occupancy 模型
```

也不是：

```text
预测 Occupancy
→ 看到 observed FREE 冲突
→ 事后全部 veto
```

而是：

> **把物理约束放进模型前向与训练目标，使模型从一开始就学习“哪些 prior 可以补全，哪些必须回退到 UNKNOWN”。**

V6.2 推荐的核心结构：

```text
冻结的 learned Occupancy prior
IR-WM 为主，GaussianWorld 为跨后端测试
                  │
                  ↓
        Sparse Prior Adapter
                  │
真实传感器硬证据 ─┼─→ Constraint-Aware Query Decoder
FREE / OCC / conflict
                  │
                  ↓
     Differentiable Feasibility Projection
                  │
                  ↓
       Evidential FREE / OCC / UNKNOWN
                  │
                  ↓
      Grouped Conformal Safety Set
                  │
                  ↓
   singleton OCC / singleton FREE / UNKNOWN
                  │
                  ↓
        SceneIR Physical State
                  │
       ┌──────────┼──────────┐
       ↓          ↓          ↓
    LogSim     WorldSim    NWM RL
```

V6.2 的最小成功标准仍绑定 V6.1 的 28-case 机制基准：

```text
>=5/28 ACCEPT
false-safe = 0
保留 V6 R10 的 3 个安全 case
至少新增 1 个 Actor case
至少新增 1 个 static/disocclusion case
accepted mask-area yield >=12%
```

但必须明确：

> V6.1 的 28-case 已经被读取，只能作为 legacy mechanism benchmark，不能承担 V6.2 的最终泛化主张。  
> V6.2 论文候选还必须通过全新的 calibration、one-shot confirmation 和 exact-once test。

---

# 1. V6.2 与 V6.1 的逻辑连续性

## 1.1 V6.1 回答了什么

V6.1 回答：

> **现成 predicted Occupancy 是否可以直接成为安全 authority？**

结论：

```text
No
```

原因不是吞吐或资源，而是：

```text
predicted OCC 与 proposal 对得上
        ↓
method-side surface coverage 高
        ↓
隐藏 observed FREE 显示 proposal 实际占据空域
        ↓
false-safe
```

## 1.2 V6.2 回答什么

V6.2 回答：

> **能否把 learned prior 约束成一个对真实观测守信、对未观测空间保守、能够输出 UNKNOWN 的 Physical State Completion？**

V6.2 的变化不是“增加一个 verifier”，而是：

| 层级 | V6.1 | V6.2 |
|---|---|---|
| Learned Occupancy | argmax surface | soft prior distribution / feature |
| Observed FREE | hidden evaluator 或独立 gate | 方法内硬约束的一部分 |
| Observed OCC | support evidence | 方法内硬正约束 |
| Contradiction | false-safe 后拒绝 | 前向时强制 UNKNOWN |
| Unknown region | prior argmax 填满 | selective completion |
| Calibration | 无训练、无校准 | 独立 calibration split |
| 输出 | dense class | FREE / OCC / UNKNOWN prediction set |
| 训练 | frozen backend inference | lightweight constrained completion head |
| 下游 | 未解锁 ME4 | 通过 fresh gates 后解锁 |

---

# 2. 调研结论与创新边界

## 2.1 可靠性与不确定性并不是新问题

### ReliOcc

已有工作已经证明，视觉 Occupancy 的精度提高不等于可靠性提高；可以通过体素不确定性学习、相对不确定性混合和离线校准改善可靠性。

V6.2 不能把：

```text
增加 uncertainty head
```

单独写成贡献。

### OCCUQ

已有工作使用单次前向的认知不确定性与偶然不确定性估计，提高传感器故障和分布外条件下的置信度校准。

V6.2 不能把：

```text
最大软概率 / 熵 / 不确定性阈值
```

当成新的安全方法。

### α-OCC

已有工作已经把分层保形预测（Hierarchical Conformal Prediction）用于 Occupancy，生成具有覆盖保证的预测集合，并处理类别不平衡。

V6.2 的保形预测必须服务于：

```text
case-level false-safe-controlled world acceptance
```

而不是重复体素语义集合预测。

---

## 2.2 FREE / OCC / UNKNOWN 与矛盾证据也已有成熟思想

### EvOcc

证据理论（Evidence Theory）已经用于：

- 未观测空间；
- 矛盾测量；
- FREE / OCC / UNKNOWN；
- 更可靠的 Occupancy 训练数据与模型损失。

所以 V6.2 不能把：

```text
三态 Occupancy
```

本身写成创新。

V6.2 的差异在于：

> **将三态证据场嵌入 Generative World Compiler，并作为 proposal bake、碰撞状态和下游任务资格的权威接口。**

---

## 2.3 连续四维射线查询已有重要先例

### QueryOcc

QueryOcc 直接在连续四维时空中采样查询点：

- 激光雷达 hit 之前的射线上采样 FREE query；
- hit 后缓冲区采样 OCC query；
- 使用相邻时刻的独立查询监督；
- 避免先把所有点聚合成固定体素真值。

V6.2 应迁移其**查询式监督思想**，但不完整复现其多卡训练骨干。

V6.2 的区别：

```text
QueryOcc:
学习通用自监督 Occupancy

CPSC:
把 frozen learned prior 约束成 task-verifiable physical state
```

---

## 2.4 语义、几何、不确定性指导稀疏补全已有相关工作

SUG-Occ 已经使用：

- 语义和不确定性抑制自由空间错误投影；
- 无符号距离编码；
- 稀疏粗到细补全。

OccAny 已经提供跨相机配置、跨域、无精确标定的通用 Occupancy 能力。

GaussianFlowOcc、GaussianOcc 等已经研究：

- Gaussian 表示 Occupancy；
- 时序 flow；
- 弱监督 Occupancy；
- Gaussian rendering supervision。

所以 V6.2 不能把：

```text
稀疏 Occupancy
Gaussian Occupancy
temporal Occupancy
```

单独写成贡献。

---

## 2.5 硬约束投影有通用方法学基础

可微凸优化层（Differentiable Convex Optimization Layer）、可微投影（Differentiable Projection）以及硬约束神经网络（Hard-Constrained Neural Network）已经证明：

> 神经网络可以先提出 unconstrained prediction，再通过可微投影层保证输入相关的硬约束在前向中成立。

V6.2 不需要把大规模三维网格直接交给通用凸优化器。

更适合单卡的实现是：

```text
稀疏 active query set
+
closed-form hard mask projection
+
3–5 步局部可微 proximal / primal-dual update
```

---

# 3. V6.2 推荐方法：CPSC

# CPSC
## Constraint-Aware Physical State Completion

CPSC 包含五个核心组件：

```text
H0  Tri-State Hard Evidence Field
H1  Frozen Soft Prior Adapter
H2  Constraint-Aware Sparse Query Decoder
H3  Differentiable Feasibility Projection
H4  Set-Valued Safety Calibration
```

以及一个关键训练机制：

```text
Counterfactual Evidence Dropout
```

---

# 4. 问题形式化

对时刻 \(t\)、空间查询点 \(x\)，物理状态为：

\[
y(x,t)\in
\{
\mathrm{FREE},
\mathrm{OCCUPIED},
\mathrm{UNKNOWN}
\}.
\]

输入分为两类。

## 4.1 硬证据

\[
E_t=
\left(
E_t^F,
E_t^O,
E_t^C,
E_t^A
\right)
\]

其中：

- \(E_t^F\)：真实射线证明的 FREE；
- \(E_t^O\)：激光雷达 hit / 可靠表面证明的 OCCUPIED；
- \(E_t^C\)：冲突区域；
- \(E_t^A\)：Actor identity、OBB、trajectory、lifecycle 等结构约束。

## 4.2 软先验

\[
P_\theta^0(y\mid x,t,I_{1:T})
\]

来自冻结的 Occupancy backend：

```text
主 prior：
IR-WM

跨后端复验：
GaussianWorld

可选后续：
OccAny
```

软先验可以提供：

- occupancy logits；
- semantic logits；
- BEV / voxel feature；
- temporal feature；
- flow；
- model uncertainty。

软先验没有权利覆盖硬证据。

## 4.3 CPSC 输出

\[
Q_\psi(y\mid x,t,E_t,P_\theta^0)
\]

并进一步生成集合式输出：

\[
\Gamma(x,t)
\subseteq
\{
F,O,U
\}.
\]

World Compiler 的物理状态为：

\[
\hat y(x,t)=
\begin{cases}
F,&\Gamma(x,t)=\{F\}\\
O,&\Gamma(x,t)=\{O\}\\
U,&\text{otherwise}
\end{cases}
\]

不允许从：

```text
{O,U}
```

强行取 argmax OCC。

---

# 5. H0 — 三态硬证据场

## 5.1 方法输入与评测证据继续分离

继承 V6.1：

```text
O_method
O_eval
```

但 V6.2 新增训练期层级：

```text
E_input
E_dropout_target
E_validation
E_eval
```

### `E_input`

模型前向可见的真实证据。

### `E_dropout_target`

从 `E_input` 内部预先随机留出的 ray/block，仅在训练损失中使用。

### `E_validation`

development / calibration 的独立目标。

### `E_eval`

one-shot confirmation 与 exact-once test 的隐藏独立证据。

任何 source hash 重叠都 fail-closed。

---

## 5.2 射线硬约束

对真实激光雷达射线：

\[
r(s)=o+s\,d
\]

hit 深度为 \(d_h\)。

冻结规则：

\[
r(s),\quad 0<s<d_h-\delta_f
\Rightarrow
FREE
\]

\[
d_h-\delta_o
\le s\le
d_h+\delta_o
\Rightarrow
OCCUPIED
\]

\[
s>d_h+\delta_o
\Rightarrow
UNKNOWN
\]

除非有其他视角或时刻证据。

禁止：

```text
hit 后空间直接填成 FREE
```

---

## 5.3 矛盾证据

若同一体素/查询同时收到可靠 FREE 与 OCC：

```text
不是投票
不是看谁权重大
不是用模型 prior 决胜
```

而是：

\[
E^C(x,t)=1
\Rightarrow
Q(x,t)=UNKNOWN
\]

并保存：

```text
contradiction provenance
source rays
time difference
actor/static identity
```

---

## 5.4 Actor 层

Actor 物理状态：

\[
O_{a,t}
=
T_a(t)O_a^{canonical}
\]

硬约束：

- Actor identity 不变；
- lifecycle 外必须 UNKNOWN / absent；
- native Actor OBB 只定义 envelope/identity，不自动把整个 box 填为 OCC；
- source removal 后恢复 UNKNOWN，不恢复 FREE；
- static 和 Actor 层不能扁平覆盖。

---

# 6. H1 — 冻结软先验适配器

## 6.1 首选 IR-WM

原因：

- V6.1 中 FREE conflict 明显低于 GaussianWorld；
- 3090 能力合同已经闭合；
- 有 current occupancy、temporal history 与 vision features；
- 不需要再做后端遍历。

## 6.2 冻结内容

第一阶段冻结：

- backbone；
- temporal encoder；
- world model；
- Occupancy logits；
- source weights。

只训练：

```text
prior adapter
query decoder
evidential head
projection-compatible residual
```

## 6.3 Sidecar 训练

为了降低单卡成本：

```text
IR-WM forward
→ 持久化 sparse prior sidecar
→ 释放 IR-WM 进程
→ 训练 CPSC head
```

sidecar 至少包含：

```text
prior logits
selected latent features
coordinate metadata
frame transforms
source hashes
backend identity
```

训练时不让 IR-WM 常驻显存。

---

# 7. H2 — 约束感知稀疏查询解码器

## 7.1 为什么不用全局 dense 3D UNet

V6.1 target grid：

```text
300 × 300 × 40
```

约 360 万体素。

完整 dense training：

- 显存高；
- 大量 EASY FREE；
- 不直接服务 proposal；
- 单卡不划算。

V6.2 使用 active query set：

\[
\mathcal A=
\mathcal A_{prior}
\cup
\mathcal A_{ray}
\cup
\mathcal A_{proposal}
\cup
\mathcal A_{actor}
\cup
\mathcal A_{boundary}.
\]

只在：

- prior occupied 附近；
- 真实 ray；
- proposal volume；
- Actor swept volume；
- FREE/OCC 边界；
- contradiction 周围；

采样查询。

active set 外：

```text
UNKNOWN
```

---

## 7.2 查询特征

每个查询 \(q=(x,y,z,t)\) 输入：

### Prior features

- IR-WM logits；
- semantic logits；
- prior entropy；
- selected latent feature；
- temporal prediction residual。

### Hard-evidence features

- 是否 observed FREE；
- 是否 observed OCC；
- 是否 contradiction；
- 距离最近 FREE ray；
- 距离最近 OCC hit；
- ray position / hit ordering；
- view count；
- sweep count；
- temporal support。

### Structural features

- world / lidar / actor canonical coordinates；
- static / actor layer；
- actor identity；
- lifecycle；
- distance to Actor OBB surface；
- ground-relative height；
- proposal type。

---

## 7.3 网络结构

第一版使用轻量查询网络：

```text
Feature normalization
→ 4-layer MLP
→ hidden width 256
→ residual blocks ×2
→ three-state evidential head
→ constraint trust head
```

条件允许时增加：

```text
local sparse neighborhood encoder
```

但不在第一轮同时引入完整 SparseConv + Transformer。

---

# 8. H3 — 可微可行性投影

## 8.1 投影集合

定义约束集合：

\[
\mathcal C(E)=
\left\{
Q:
\begin{array}{l}
Q_v=(1,0,0),\ v\in E^F\\
Q_v=(0,1,0),\ v\in E^O\\
Q_v=(0,0,1),\ v\in E^C\\
Q_O(r(s))=0,\ s<d_h-\delta_f\\
Q_O=0,\ \text{outside lifecycle/envelope}\\
Q\in\Delta^3
\end{array}
\right\}.
\]

其中 \(\Delta^3\) 为三态概率单纯形。

## 8.2 初始输出

\[
Q^0=
\operatorname{softmax}
\left(
Z_{\theta}^{prior}
+
R_\psi
\right).
\]

## 8.3 迭代投影

使用固定 3–5 步：

\[
\tilde Q^{k+1}
=
Q^k
-
\eta\nabla_Q
\mathcal E_\psi
\left(
Q^k
\right)
\]

\[
Q^{k+1}
=
\Pi_{\mathcal C(E)}
\left(
\tilde Q^{k+1}
\right).
\]

软能量：

\[
\mathcal E=
\lambda_p D_{\mathrm{KL}}(Q\|P^0)
+
\lambda_t L_{\mathrm{temporal}}
+
\lambda_s L_{\mathrm{surface}}
+
\lambda_a L_{\mathrm{actor}}.
\]

关键纪律：

> 硬证据由 \(\Pi_{\mathcal C(E)}\) 精确保证，不通过“大权重 penalty”近似保证。

## 8.4 单卡实现

第一版不依赖通用大规模凸求解器。

实现：

- masked closed-form projection；
- sparse gather/scatter；
- ray segment projection；
- actor lifecycle projection；
- small local proximal update；
- fixed iteration count；
- synthetic exact oracle。

cvxpylayers 只用于小型 operator parity fixture，不进入百万查询正式路径。

---

# 9. Counterfactual Evidence Dropout

这是 V6.2 最关键的新训练机制之一。

V6.1 的 hidden FREE conflict 说明：

> 方法只尊重“当前看见的 FREE”还不够；必须学习在未观测区域中何时应该保持 UNKNOWN。

## 9.1 训练时证据留出

从 `E_input` 内预先冻结：

```text
ray-level dropout
block-level dropout
time-level dropout
camera-sector dropout
```

模型只看剩余证据。

被留出的真实射线作为：

```text
pseudo-hidden FREE/OCC target
```

## 9.2 目标

当 soft prior 在被留出的 FREE 空间预测 OCC 时：

```text
模型应提高 UNKNOWN
而不是继续输出 OCC
```

当 soft prior 在被留出的 OCC 表面预测 OCC 且邻近观测支持一致时：

```text
模型应保留 OCC yield
```

## 9.3 不能做的事

- 不从真实 confirmation/test 构造 dropout；
- 不根据 V6.1 10 个 false-safe case 手工画黑名单；
- 不按 scene/frame/case ID 学习；
- 不把 O_eval 直接输入网络；
- 不以 evaluator conflict ratio 作为 test-time feature。

---

# 10. Evidential 三态头

## 10.1 输出

网络输出非负证据：

\[
e_F,e_O,e_U\ge0
\]

Dirichlet 参数：

\[
\alpha_k=e_k+1
\]

总证据：

\[
S=\sum_k\alpha_k
\]

概率：

\[
p_k=\frac{\alpha_k}{S}
\]

总不确定性：

\[
u=\frac{K}{S},\quad K=3.
\]

## 10.2 语义

- prior 强但硬证据少：提高 UNKNOWN；
- hard FREE：投影为 FREE；
- hard OCC：投影为 OCC；
- contradiction：投影为 UNKNOWN；
- backend disagreement：作为不确定性输入，不直接投票。

## 10.3 Loss

### 查询分类损失

\[
L_{\mathrm{query}}
\]

作用于 training target queries。

### Evidential loss

\[
L_{\mathrm{evid}}
\]

降低错误高证据。

### Hidden-FREE risk loss

\[
L_{\mathrm{hidden-free}}
\]

直接惩罚在 evidence dropout FREE query 上输出 OCC。

### Safe-OCC retention loss

\[
L_{\mathrm{safe-occ}}
\]

防止模型通过“全部 UNKNOWN”取得零 false-safe。

### Temporal consistency

\[
L_{\mathrm{temp}}
\]

只比较 motion-compensated、identity-consistent queries。

### Prior preservation

\[
L_{\mathrm{prior}}
\]

仅在有足够支持且无硬冲突的区域保留 prior。

总损失：

\[
L=
L_{\mathrm{query}}
+
\lambda_eL_{\mathrm{evid}}
+
\lambda_fL_{\mathrm{hidden-free}}
+
\lambda_oL_{\mathrm{safe-occ}}
+
\lambda_tL_{\mathrm{temp}}
+
\lambda_pL_{\mathrm{prior}}.
\]

最终 world decision 不把这些项重新加成一个置信度。

---

# 11. H4 — 集合式安全校准

## 11.1 为什么不用 max-softmax threshold

V6.1 已经说明：

```text
argmax Occupancy
```

无法识别 false-safe。

OCCUQ、ReliOcc 等研究也说明：

```text
最大概率 / 熵
```

通常不足以覆盖传感器故障和分布外风险。

## 11.2 Grouped Conformal Calibration

V6.2 采用分组保形校准：

```text
calibration unit = case / target frame
```

不是把数百万相关体素当独立样本。

每个 case 的非一致性分数包含：

- 最大 FREE-as-OCC risk；
- occupied support miss；
- UNKNOWN undercoverage；
- proposal-level closure；
- Actor case 的 swept-volume risk。

校准生成：

\[
\Gamma(x,t)
\]

并在 proposal 级别输出：

```text
ACCEPT
ABSTAIN
REJECT
```

## 11.3 保证边界

只能声明：

> 在 calibration / test exchangeability 假设下，prediction set 具有有限样本的边际覆盖性质。

不能声明：

- 现实道路绝对安全；
- 任意分布漂移保证；
- zero empirical false-safe 等于概率为零。

---

# 12. Physical State Compiler 输出

CPSC 输出不是普通 Occupancy benchmark tensor。

## 12.1 Voxel / query sidecar

每个有效 query：

```text
position
time
state_set
p_free
p_occupied
p_unknown
evidence_mass
hard_evidence_type
prior_backend
actor_identity
provenance
```

## 12.2 Collision asset

只有 singleton OCC 且通过 proposal gate 的区域才能生成：

- sparse occupancy；
- SDF；
- canonical Actor collision body；
- static collision chunk。

## 12.3 UNKNOWN

UNKNOWN：

- 不生成 collision surface；
- 不写为 background；
- 不参与安全 reward；
- 触发 abstain / fallback；
- 可以保留 appearance Gaussian，但必须标记 physical-invalid。

---

# 13. 数据与 split

## 13.1 Tier L — Legacy Mechanism Benchmark

```text
V6.1 28-case
```

用途：

- 重放 V6.1 failure；
- operator development；
- matched mechanism comparison；
- 检查是否达到 5/28。

限制：

```text
已经读取
不能写成新泛化验证
```

## 13.2 Tier D — 新 Development / Training

从官方 nuScenes train split 中结果前选择：

```text
6 scenes
每场 >=10 target frames
>=60 target units
```

选择只使用 metadata：

- sensor completeness；
- day/night；
- actor density；
- motion level；
- map location；
- camera/LiDAR availability。

不得使用 Occupancy quality 或 proposal outcome 选 scene。

## 13.3 Tier C — Calibration

```text
2–4 scenes
>=40 case-level calibration units
```

独立于训练。

用于：

- grouped conformal quantile；
- fixed acceptance policy；
- no gradient；
- no architecture selection。

## 13.4 Tier H — One-shot Confirmation

```text
3 scenes
每场 >=6 target units
```

候选冻结后一次读取。

失败即消耗。

## 13.5 Tier T — Exact-once Test

```text
4–6 scenes
```

在 confirmation PASS 后解锁。

每个 scene：

```text
exclusive-create attempt before any content read
```

---

# 14. 数据构造

## 14.1 Paired evidence

每个 target：

```text
method sweeps
target sweeps
```

按 source timestamp / sweep identity 分离。

## 14.2 Motion compensation

静态：

```text
ego motion compensation
```

动态：

```text
actor identity + pose compensation
```

禁止直接堆叠 multi-sweep 生成 ghost。

## 14.3 Query types

每个 target 按冻结比例抽样：

```text
hard FREE ray query
hard OCC hit-band query
unknown behind-hit query
prior OCC proposal query
proposal boundary query
actor canonical query
temporal correspondence query
contradiction query
```

## 14.4 建议查询预算

单 target：

```text
100k–300k sparse queries
```

而不是 QueryOcc 完整多卡设置。

正式值由无质量资源 smoke 冻结。

---

# 15. Baselines

## B0 — IR-WM Argmax

V6.1 canonical baseline：

```text
10/28 ACCEPT
10/10 false-safe
```

只重放 frozen artifact，不重复模型 inference。

## B1 — Hard Veto / Clip

```text
argmax prior
+
visible O_method hard projection
```

作用：

> 检查“仅做显式 hard clamp”是否足够。

预期：

- 当前可见冲突消失；
- hidden conflict 仍可能存在；
- yield 可能塌缩。

这是必要控制，但不能作为主方法。

## B2 — Max-softmax / Entropy

常规不确定性 threshold baseline。

threshold 只在 Tier C 校准。

## B3 — Evidential Head without Projection

检验：

```text
uncertainty learning
```

是否能替代硬约束。

## B4 — Projection without Evidence Dropout

检验：

```text
hard constraints
```

能否处理 hidden FREE。

## B5 — Projection + Evidence Dropout, no Conformal

检验 training mechanism。

## M0 — Full CPSC

```text
soft prior
+
hard projection
+
counterfactual evidence dropout
+
evidential UNKNOWN
+
grouped conformal set
```

## M1 — Cross-backend Transfer

在 CPSC frozen 后：

```text
IR-WM → GaussianWorld prior
```

只训练轻量 adapter，或先 zero-shot adapter。

用途：

> 验证方法是否只是 IR-WM 特定补丁。

---

# 16. 核心消融

每次只去掉一个机制：

| Arm | Hard Projection | Evidence Dropout | Evidential UNKNOWN | Temporal | Conformal |
|---|---:|---:|---:|---:|---:|
| B0 | 否 | 否 | 否 | 原生 | 否 |
| B1 | 是 | 否 | 否 | 原生 | 否 |
| B3 | 否 | 是 | 是 | 是 | 否 |
| B4 | 是 | 否 | 是 | 是 | 否 |
| B5 | 是 | 是 | 是 | 是 | 否 |
| M0 | 是 | 是 | 是 | 是 | 是 |

禁止第一轮加入：

- 新 Occupancy backbone；
- diffusion completion；
-大语言模型（LLM）；
- radar；
- map neural prior；
-完整 SparseConv transformer；
-大规模 LoRA。

---

# 17. 指标

## 17.1 世界编译主指标

### Safe Valid Yield

\[
\mathrm{SVY}
=
\frac{
\#(\mathrm{ACCEPT}\land\mathrm{Safe})
}{
\#\mathrm{All}
}
\]

### False-safe

\[
\mathrm{FS}
=
\frac{
\#(\mathrm{ACCEPT}\land\mathrm{Unsafe})
}{
\#\mathrm{ACCEPT}
}
\]

同时报告：

- numerator；
- denominator；
- Wilson / Clopper–Pearson 区间；
- per-scene；
- worst case。

## 17.2 Coverage

- ACCEPT；
- ABSTAIN；
- REJECT；
- accepted mask-area；
- accepted volume；
- Actor / static / disocclusion；
- near / mid / far；
- moving / stationary。

## 17.3 物理状态

- FREE conflict；
- OCC support；
- UNKNOWN volume；
- contradiction；
- RayIoU；
- occupancy IoU；
- calibration；
- Brier；
- NLL；
- ECE；
- prediction set size。

## 17.4 下游

- LogSim replay identity；
- collision event consistency；
- WorldSim safe route length；
- Actor insertion yield；
- multi-Actor false-safe；
- RL collision critic recall；
- planner collision；
- route progress；
- stuck；
- comfort。

---

# 18. 最小实验 ME-CPSC

Task：

```text
WS-V62-ME-CPSC-LEGACY28-01
```

## 18.1 目的

回答：

> CPSC 是否能在同一 V6.1 28-case 上恢复 oracle 上界的一部分，并避免 GaussianWorld/IR-WM 的 hidden FREE false-safe？

## 18.2 输入

- frozen V6.1 IR-WM prior sidecar；
- frozen ME0 O_method；
- frozen ME1 O_eval；
- frozen R10 comparator；
- 新 CPSC model；
- 不重新运行 IR-WM。

## 18.3 Gate

```text
ACCEPT >=5/28
false-safe = 0
R10 3/3 retained
至少 1 Actor 新增
至少 1 static/disocclusion 新增
accepted mask-area >=12%
mean accepted FREE conflict <=0.05
worst accepted FREE conflict <=0.05
```

## 18.4 Anti-trivial gate

```text
UNKNOWN / ABSTAIN 不能超过预注册上限
safe OCC retention 必须达到 oracle accepted surface 的 >=50%
```

否则：

```text
all-UNKNOWN safety hack
```

## 18.5 解释边界

PASS：

```text
legacy mechanism breakthrough
```

不是：

```text
scene-disjoint generalization confirmed
```

---

# 19. 正式阶段计划

# P0 — V6.1 Freeze / V6.2 Scope

Task：

```text
WS-V62-P0-SCOPE-FREEZE-01
```

动作：

1. 不改 V6.1 closeout；
2. 从 V6.1 final commit 新建 V6.2 branch；
3. 更新状态文档；
4. 在 failure ledger 新增 `V62-F01`；
5. 冻结新 plan/config/hash；
6. 确认 confirmation/test 仍 unread；
7. 冻结路径 resolver；
8. 记录 GPU/disk/cgroup。

推荐分支：

```text
research/worldsim-v6.2-cpsc
```

---

# P1 — Literature / Novelty Gate

Task：

```text
WS-V62-P1-NOVELTY-AUDIT-01
```

核对：

- ReliOcc；
- OCCUQ；
- α-OCC；
- EvOcc；
- QueryOcc；
- SUG-Occ；
- OccAny；
- GaussianFlowOcc；
- Differentiable Projection；
- HardNet；
- Physics-Constrained Flow Matching。

必须回答：

```text
是否已有工作同时做到：
hard observed FREE/OCC
+
learned Occupancy prior
+
selective UNKNOWN
+
proposal bake
+
world-simulation false-safe evaluation
```

若发现直接重合：

```text
先改 contribution，再编码
```

---

# P2 — Evidence Query Dataset

Task：

```text
WS-V62-P2-EVIDENCE-QUERY-DATASET-01
```

输出：

```text
QUERY_MANIFEST.jsonl
METHOD_EVIDENCE.jsonl
DROPOUT_TARGETS.jsonl
TARGET_EVIDENCE.jsonl
SPLIT_MANIFEST.json
```

无训练。

Gate：

- method/target disjoint；
- coordinate exact；
- query class denominator；
- actor/static split；
- dropout deterministic；
- repeated build byte-exact；
- no validation/test read。

---

# P3 — Hard Projection Operator

Task：

```text
WS-V62-P3-FEASIBILITY-PROJECTION-01
```

先 synthetic，再真实 fixture。

必须验证：

```text
hard FREE exact
hard OCC exact
contradiction → UNKNOWN exact
ray-before-hit OCC = 0
lifecycle outside OCC = 0
probability simplex exact
gradient finite
fresh process deterministic
```

资源：

```text
CPU/small GPU
```

失败不能读取质量。

---

# P4 — Prior Sidecar Materialization

Task：

```text
WS-V62-P4-IRWM-PRIOR-SIDECAR-01
```

- 复用 V6.1 IR-WM environment / weights；
- 新 development scenes；
- batch1；
- scene workers 串行或最多 2 个；
- sidecar 内容寻址；
- 模型前后 hash；
- 不训练 IR-WM；
- 不读 target evidence。

---

# P5 — CPSC-Lite Training

Task：

```text
WS-V62-P5-CPSC-LITE-TRAIN-01
```

## 训练内容

```text
query decoder
evidential head
projection residual
```

IR-WM frozen。

## 资源目标

```text
peak VRAM <=18 GiB
single run <=12h
disk incremental <=20 GiB
```

## 最小配置

- FP16 / BF16；
- batch by query count；
- gradient accumulation；
- 3 projection iterations；
- early stop only on development objective；
- seed 0/1/2 smoke 后固定 seeds。

---

# P6 — Legacy 28 Minimum Experiment

Task：

```text
WS-V62-P6-LEGACY28-ME-01
```

执行 B0–B5/M0 matched comparison。

PASS 后继续 fresh calibration/confirmation。

FAIL：

```text
关闭当前 CPSC-Lite family
```

允许一次机制级 recovery：

- projection architecture；
- evidence dropout；
- set-valued head；

三者只能选择一个。

禁止：

- threshold/grid/window sweep；
- 换 backend；
- 加模型堆料；
-删 case。

---

# P7 — Fresh Development Evaluation

Task：

```text
WS-V62-P7-FRESH-DEVELOPMENT-01
```

目标：

- scene-disjoint；
- 选择单一 candidate；
- 固定 conformal score；
- 固定 all metrics；
- 不读取 confirmation。

晋级：

```text
>=2 independent scenes positive
false-safe = 0
safe valid yield > IR-WM/B1
anti-trivial coverage pass
calibration improves
```

---

# P8 — Calibration

Task：

```text
WS-V62-P8-GROUPED-CONFORMAL-CALIBRATION-01
```

只读取 Tier C。

输出：

```text
CALIBRATION_STATE.npz
NONCONFORMITY.jsonl
PREDICTION_SET_POLICY.json
```

候选、网络、feature、loss 全冻结。

---

# P9 — One-shot Confirmation

Task：

```text
WS-V62-P9-ONE-SHOT-CONFIRMATION-01
```

attempt 在任何 quality read 前创建。

Gate：

```text
false-safe = 0
>=2/3 scenes 有非零 ACCEPT
scene-balanced SVY > matched prior
prediction-set coverage 达标
无 catastrophic geometry regression
```

失败：

```text
candidate rejected
confirmation consumed
不得 refit
```

---

# P10 — Exact-once Test

Task：

```text
WS-V62-P10-EXACT-ONCE-TEST-01
```

输出：

- full denominator；
- per-scene；
- worst case；
- confidence interval；
- no rerun；
- paper table。

---

# 20. ME4 — 多 Actor 物理压力测试

仅在 P9 PASS 后解锁。

Task：

```text
WS-V62-ME4-MULTIACTOR-PHYSICAL-STATE-01
```

## Proposal 数量

```text
>=3 scenes
>=150 proposals
2 / 3 / 4 Actor
```

## Arms

```text
AABB broad phase
OBB
IR-WM argmax
CPSC Occupancy
CPSC swept occupancy
```

## Gate

```text
accepted >=60 proposals
empirical false-safe = 0
95% Clopper–Pearson upper bound <5%
accepted yield 比 conservative AABB/OBB 提高 >=10%
no lifecycle phantom collision
no actor-static penetration
```

AABB 只作 broad phase，不作最终 authority。

---

# 21. 三下游任务

# 21.1 GS + LogSim

目标：

```text
case replay / regression interception
```

CPSC 作用：

- 原始 collision state；
- FREE/OCC consistency；
- sensor / label / collision 共享状态；
- 检查重建是否改变关键事件。

Gate：

```text
same log
same physical state hash
same actor state
same collision events
same labels
deterministic sensor replay
```

---

# 21.2 GS + WorldSim

目标：

- 新路线；
-新 Actor；
-轨迹改变；
-多 Actor；
-场景扩展。

主指标：

```text
Safe Valid Yield
False-safe
Verified route length
Verified world area
Actor insertion yield
UNKNOWN
```

V6.2 不以 PSNR 为主终点。

---

# 21.3 GS + NWM 强化学习

仅在 ME4 PASS 后解锁。

## RL-0 Collision Critic

三臂：

```text
Real-only
Real + naive generated
Real + CPSC verified
```

相同：

- NWM；
- planner；
- seed；
- rollout；
- action；
- budget。

Primary：

```text
collision false-safe
unsafe-action recall
safe-action precision
calibration
```

Secondary：

```text
progress
stuck
comfort
reward
```

## RL-1

若 RL-0 PASS，再执行轻量：

- preference optimization；
- GRPO；
- small correction policy；
- model-based rollout update。

禁止：

```text
全刹车降低碰撞
→ 宣称安全增益
```

---

# 22. 单卡 3090 策略

## 22.1 不重训 Occupancy backbone

第一版：

```text
frozen IR-WM
+
offline feature/logit sidecar
+
lightweight CPSC
```

## 22.2 GPU 串行

- IR-WM extraction；
- CPSC training；
- renderer；
- verifier；

分进程串行释放。

## 22.3 资源 stop

- 同机制第二次 OOM；
- peak >22 GiB；
- disk <安全下限；
- cgroup 持续压力；
- CUDA health anomaly。

## 22.4 自动 recovery

允许一次：

- batch size；
- query chunk；
- gradient accumulation；
- CPU memmap；
- process separation。

不允许：

- 降正式分辨率后称 faithful；
- 改 query distribution 获得漂亮结果；
- 删除远距 / Actor queries。

---

# 23. Autoresearch Loop

每轮：

```text
OBSERVE
→ DIAGNOSE
→ HYPOTHESIS
→ NOVELTY / FAILURE GATE
→ PREREGISTER
→ EXECUTE
→ AUDIT
→ REFLECT
→ PROMOTE / REJECT / PIVOT
```

## 23.1 Hypothesis schema

```text
hypothesis_id
problem
mechanism
hard constraints
soft priors
minimum experiment
expected result
falsification
resource
failure refs
```

## 23.2 Reflection

保存：

```text
docs/autoresearch/worldsim_v62/REFLECTIONS.jsonl
```

真正可复用失败继续写：

```text
docs/RESEARCH_FAILURES.md
```

不创建第二本 failure ledger。

## 23.3 自进化允许

- 调整 query decoder；
- 改 projection mechanism；
- 改 dropout design；
- 改 uncertainty head；
- 换训练 objective；
- 增加单因素消融；
- 在冻结前换 candidate。

## 23.4 禁止

- 结果后改当前 primary gate；
- 读 confirmation 后 refit；
- 用 O_eval 做 test-time feature；
- 换 backend 搜到通过；
- 覆盖 failed run；
- 把 all-UNKNOWN 写成安全成功。

---

# 24. Failure / Stop Rules

## Stop 1 — Hard Projection Control

若 B1 已能得到：

```text
>=5/28
0 false-safe
```

则 learned CPSC 的贡献空间不足。

转向：

```text
projection-only compiler
+
fresh generalization
```

诚实调整方法定位。

## Stop 2 — All UNKNOWN

若：

```text
false-safe = 0
但 ACCEPT <5/28
或 safe OCC retention <50%
```

拒绝。

## Stop 3 — Hidden FREE 不下降

若 full CPSC 相对 IR-WM：

```text
accepted hidden-FREE conflict reduction <80%
```

关闭当前 head。

## Stop 4 — Fresh Confirmation 失败

不复开同 candidate。

## Stop 5 — Cross-backend 不迁移

若 IR-WM PASS、GaussianWorld prior 完全失败：

- 保留 IR-WM-specific result；
- 不宣称 backend-agnostic；
- 论文标题去掉 general；
- 不继续遍历第三 backend。

## Stop 6 — RL 无增量

若 Real-only 与 CPSC 一样好：

```text
不削弱 Real-only
不制造 naive 弱基线
```

RL 作为负结果关闭。

---

# 25. 论文主张边界

## 可以主张

若通过：

1. learned Occupancy prior 不能直接成为物理权威；
2. 硬传感器约束 + learned prior + selective UNKNOWN 可以提高安全 valid yield；
3. CPSC 前向严格保持 observed FREE/OCC；
4. grouped conformal set 改善 case-level false-safe calibration；
5. compiled physical state 可用于 LogSim、WorldSim 和多 Actor collision；
6. 单卡可训练。

## 不可主张

- 现实自动驾驶绝对安全；
- conformal 对任意分布保证；
- 所有 Occupancy model 都失败；
- zero empirical false-safe 等于零风险；
- 只在 legacy 28-case PASS 就证明泛化；
- Occupancy IoU 提升自动等于 RL 增益。

---

# 26. ArXiv 技术报告结构

## 1. Introduction

- V6.1 empirical contradiction；
- Oracle 10/28 vs learned 10/10 false-safe；
- safety authority gap。

## 2. Related Work

- uncertainty-aware Occupancy；
- evidential mapping；
- continuous query supervision；
- constrained neural prediction；
- world generation / world compilation。

## 3. Problem

- hard evidence；
- soft prior；
- FREE/OCC/UNKNOWN；
- proposal safety；
- three downstream tasks。

## 4. Method

- sparse queries；
- evidence dropout；
- evidential head；
- differentiable projection；
- grouped conformal sets；
- SceneIR Physical State。

## 5. Experiments

- legacy 28；
- fresh dev/calibration；
- confirmation/test；
- multi-Actor；
- three downstream tasks；
- resource。

## 6. Failure Analysis

- all-UNKNOWN；
- prior preservation；
- sensor sparsity；
- distribution shift；
- calibration limits。

## 7. Limitations

- LiDAR coverage；
- map availability；
- open-world Actor；
- exchangeability；
- finite scene count；
- no real-road safety claim。

---

# 27. 推荐主表

## Table 1 — Legacy 28

| Method | ACCEPT | False-safe | SVY | Mask Yield | UNKNOWN |
|---|---:|---:|---:|---:|---:|
| R10 | | | | | |
| IR-WM | | | | | |
| Hard Projection | | | | | |
| Evidential only | | | | | |
| CPSC | | | | | |

## Table 2 — Fresh Generalization

- scene-balanced；
- worst case；
- calibration；
- prediction set size；
- false-safe interval。

## Table 3 — Ablation

- projection；
- dropout；
- evidential；
- temporal；
- conformal。

## Table 4 — Multi-Actor

- AABB；
- OBB；
- argmax Occ；
- CPSC；
- swept CPSC。

## Table 5 — Downstream

- LogSim；
- WorldSim；
- NWM critic / RL。

## Table 6 — Resources

- prior extraction；
- training；
- inference；
- memory；
- storage；
- collision latency。

---

# 28. 首轮执行顺序

```text
P0  V6.1 freeze / V6.2 branch
↓
P1  novelty audit
↓
P2  new evidence-query dataset
↓
P3  exact hard projection
↓
P4  frozen IR-WM prior sidecars
↓
P5  CPSC-Lite training
↓
P6  legacy28 minimum experiment
↓
    FAIL → one mechanism-level recovery or close
    PASS
↓
P7  fresh development candidate selection
↓
P8  grouped conformal calibration
↓
P9  one-shot confirmation
↓
P10 exact-once test
↓
ME4 multi-Actor
↓
LogSim / WorldSim
↓
NWM collision critic
↓
optional RL
```

---

# 29. 第一轮可执行任务表

| Task | 目的 | GPU | 成功条件 |
|---|---|---:|---|
| `WS-V62-P0-SCOPE-FREEZE-01` | 新路线冻结 | 否 | source/data/test lock |
| `WS-V62-P1-NOVELTY-AUDIT-01` | 创新边界 | 否 | 无 direct overlap |
| `WS-V62-P2-EVIDENCE-QUERY-DATASET-01` | 训练数据 | 低 | disjoint + exact |
| `WS-V62-P3-FEASIBILITY-PROJECTION-01` | 硬约束算子 | 低 | exact + differentiable |
| `WS-V62-P4-IRWM-PRIOR-SIDECAR-01` | prior 提取 | 中 | deterministic sidecar |
| `WS-V62-P5-CPSC-LITE-TRAIN-01` | 方法训练 | 高 | finite + <=18GiB |
| `WS-V62-P6-LEGACY28-ME-01` | 最小实验 | 中 | 5/28 + 0 FS |
| `WS-V62-P7-FRESH-DEVELOPMENT-01` | 新场景开发 | 中 | 2+ scenes |
| `WS-V62-P8-GROUPED-CONFORMAL-CALIBRATION-01` | 安全校准 | 低 | frozen policy |
| `WS-V62-P9-ONE-SHOT-CONFIRMATION-01` | 一次确认 | 中 | prereg gate |
| `WS-V62-P10-EXACT-ONCE-TEST-01` | 正式测试 | 中 | exact-once |

---

# 30. 最终方法故事

V6.1 的失败不是：

```text
Occupancy 无用
```

而是：

```text
感知型 Occupancy 的 dense argmax
不能直接升级成仿真的物理真值
```

V6.2 的核心方法主张应为：

> **CPSC treats real sensor evidence as immutable physical constraints, learned occupancy as a defeasible prior, and unsupported completion as UNKNOWN. It learns a constraint-aware posterior through sparse 4D query supervision and counterfactual evidence dropout, then compiles only calibrated singleton physical states into deterministic driving worlds.**

中文：

> **CPSC 将真实传感器证据视为不可违反的物理约束，将学习式 Occupancy 视为可被推翻的软先验，并把缺乏支持的补全保留为 UNKNOWN。方法通过稀疏四维查询监督和反事实证据留出学习约束感知后验，只将经过校准的单值物理状态编译进确定性驾驶世界。**

这条路线相对 V6.1 的真实增量是：

```text
post-hoc verification
        ↓
method-internal constraint-aware completion
```

相对普通 Occupancy prediction 的真实增量是：

```text
accuracy-oriented dense prediction
        ↓
false-safe-controlled physical state compilation
```

相对纯 safety filter 的真实增量是：

```text
reject unsafe output
        ↓
learn when to preserve prior, when to complete, and when to abstain
```

---

# 31. 参考研究

建议在正式 related-work 审计中至少纳入：

1. ReliOcc：Occupancy 可靠性与不确定性学习；
2. OCCUQ：高效认知/偶然不确定性；
3. α-OCC：分层保形预测；
4. EvOcc：证据理论 FREE/OCC/UNKNOWN；
5. QueryOcc：连续四维查询式监督；
6. SUG-Occ：语义/不确定性引导稀疏补全；
7. OccAny：跨域与非固定相机 Occupancy；
8. GaussianFlowOcc / GaussianOcc：Gaussian Occupancy 与时序监督；
9. Differentiable Convex Optimization Layers；
10. Differentiable Projection；
11. Hard-Constrained Neural Networks；
12. Physics-Constrained Flow Matching。

---

# 32. 对 Autoresearch Agent 的执行指令

你接手的是 V6.2 CPSC 路线。

你的目标不是继续寻找第三个 predicted Occupancy backend，而是：

> **训练一个轻量、约束感知、能够把 hard sensor evidence 和 soft learned prior 编译成 selective Physical State 的方法。**

执行要求：

1. V6.1 closeout 保持 immutable；
2. 新 branch、新 task、新 run；
3. 禁止硬编码机器路径；
4. 先 operator，再数据，再训练；
5. IR-WM frozen；
6. O_eval 永远不进入 method；
7. hard FREE/OCC 在每次 forward 后 exact；
8. contradiction 必须 UNKNOWN；
9. all-UNKNOWN 必须被 anti-trivial gate 拒绝；
10. legacy28 只作 mechanism benchmark；
11. fresh calibration/confirmation/test 决定正式 claim；
12. 一个 experiment 只改变一个 primary mechanism；
13. 工程失败最小恢复，旧 run 保留；
14. research reject 自动反思并切下一合法 hypothesis；
15. 不通过换 backend、降阈值、调网格或删 case 追求通过；
16. ME4 与 RL 只有在 fresh confirmation PASS 后解锁；
17. 每轮更新唯一 failure ledger；
18. 达到外部许可、资源或真实人工判断前，无需逐项请求审批。

---

# 33. 一句话北极星

> **从“预测一个看起来完整的 Occupancy”，升级为“编译一个服从真实物理证据、允许 UNKNOWN、能够安全进入闭环世界的 Physical State”。**
