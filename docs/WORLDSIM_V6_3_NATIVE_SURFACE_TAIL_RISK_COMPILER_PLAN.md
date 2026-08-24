# WorldSim V6.3 — 面向曲面尾部风险的原生约束编译器

> 英文工作名：**SurfNCC — Native-Feature Surface-Tail Risk Constraint Compiler**
>
> 推荐技术报告题目：
>
> **SurfNCC: Native-Feature Surface Tail-Risk Control for Verifiable Driving World Compilation**
>
> 中文题目：
>
> **SurfNCC：面向可验证驾驶世界编译的原生特征曲面尾部风险控制**
>
> 状态：`pending`
>
> 上游终态：
>
> - V6.2 分支：`research/worldsim-v6.2-cpsc`
> - V6.2 冻结 HEAD：`bcd4143e840bc33622621d5e9a4ba992d2f203bb`
> - V6.2 状态：`v62_cpsc_lite_family_closed_negative`
> - V6.2 失败证据：`V62-F06 active; recovery exhausted`
> - P7/P8/confirmation/test：未解锁、未读取
>
> 默认资源：**单卡 RTX 3090 24GB**
>
> 资源升级授权：
>
> > 默认先在单卡 RTX 3090 上执行。若正式最小实验在保持冻结科学协议的前提下，经过一次有界的显存/吞吐工程恢复后仍因硬件资源失败，Codex 必须保存完整 `blocked_resource` 证据、停止相关 GPU 分支，并向用户申请明确的多卡或大显存资源。不得偷偷降低分辨率、曲面采样数、场景分母、原生特征维度或时序窗口后继续声称完成。
>
> 核心任务：
>
> > **将 V6.2 的逐查询平均风险分类器，升级为使用原生 Occupancy 特征、对完整候选曲面进行联合建模、直接优化曲面尾部 False-safe 风险，并通过独立 case-level 校准输出 OCC / UNKNOWN 的原生约束世界编译器。**

---

# 0. 结论先行

V6.3 不重新打开 V6.2 的 CPSC-Lite family。

V6.2 已经正式证伪：

```text
argmax / prototype Occupancy prior
+
逐查询 MLP
+
逐查询 hard projection
+
平均 hidden-FREE loss
+
evidential UNKNOWN
```

不能建立隐藏曲面的安全权威。

最终证据：

```text
4 / 28 ACCEPT
4 / 4 false-safe
UNKNOWN 63.85%
0 hard-evidence violations
```

这说明：

```text
局部硬约束正确
≠
完整曲面安全

平均 query 风险下降
≠
proposal-level false-safe 受控

UNKNOWN 增加或减少
≠
剩余 ACCEPT 获得物理权威
```

V6.3 必须同时改变四件事：

1. **原生接口**：所有方法输入使用真实 per-voxel logits / latent feature，禁止 prototype bridge；
2. **结构单位**：从独立 query 升级为有 identity、邻接、法向、射线和时序支持的完整候选曲面；
3. **风险目标**：从 query mean 升级为 patch / surface / proposal 的尾部风险；
4. **决策资格**：从裸 argmax/evidential threshold 升级为独立 calibration cohort 上冻结的 case-level 风险控制策略。

核心范式：

```text
Frozen IR-WM native logits / latent
               │
Hard FREE / OCC / contradiction / lifecycle
               │
Candidate surface with native topology
               ↓
Native Surface Encoder
               ↓
Point state + hidden-FREE risk + support authority
               ↓
Exact hard projection
               ↓
Patch CVaR / worst-component / proposal tail risk
               ↓
Independent case-level calibration
               ↓
{OCC} / UNKNOWN / REJECT
               ↓
SceneIR Physical State
```

---

# 1. V6.2 必须继承的事实

## 1.1 保留的正事实

### Oracle Physical State 上界存在

V6.1 Oracle Occupancy：

```text
10 / 28 ACCEPT
0 false-safe
```

因此：

> 物理状态补全本身不是死路。

### Learned prior 有局部修正信号

V6.2 P5 在 scene-disjoint development 上，相对 projection-only：

```text
hidden-FREE false-OCC:
45.37% → 38.46%

safe-OCC retention:
90.07% → 90.11%

target accuracy:
35.68% → 48.38%
```

因此：

> 原生 IR-WM feature 可以作为软先验，不能因为 legacy 失败而丢弃。

### Hard projection 实现正确

P6/P6R：

```text
hard violations = 0
safe-OCC retention = 1.0
```

因此：

> V6.3 不重新发明 FREE/OCC one-hot projection；它将作为不可变底层算子继承。

---

## 1.2 必须关闭的旧解释

V6.3 不得把以下做法包装成新方法：

```text
再换一个 Occupancy backend
再调 confidence / entropy threshold
再改 voxel grid / history window
再做 prototype bridge
再增加 query MLP 宽度
再做 evidence dropout
再降低 UNKNOWN threshold
再用 O_eval 事后 veto
再用平均 hidden-FREE loss
再做第二次 CPSC-Lite recovery
```

---

## 1.3 V6.2 真正暴露的四个根因

### 根因 A：原生 feature 在 legacy 接口丢失

V6.2 legacy 只有：

```text
argmax class_label
```

P6/P6R 只能使用 class prototype 替代：

```text
17D logits
256D BEV latent
per-cell uncertainty
```

V6.3 禁止任何这种桥接。

### 根因 B：风险单位错误

V6.2 优化：

```text
mean query hidden-FREE risk
```

最终要求：

```text
完整 proposal surface 没有局部高风险穿模
```

平均风险会掩盖局部高风险尾部。

### 根因 C：结构表示错误

V6.2 逐 query 独立分类，没有显式建模：

- connected surface；
- geodesic neighborhood；
- ray ordering；
- surface normal；
- local closure；
- patch continuity；
- Actor canonical surface；
- proposal-level interaction。

### 根因 D：没有独立的最终资格校准

V6.2 evidential UNKNOWN 只是网络输出，不是：

```text
经过独立 calibration 冻结的安全风险边界
```

---

# 2. V6.3 的核心科学问题

## 2.1 主问题

> **在保留原生 Occupancy logits/features 与精确硬约束的前提下，面向完整候选曲面优化高分位隐藏 FREE 风险，能否在 0 empirical false-safe 下，获得非平凡的安全有效产率？**

---

## 2.2 次问题

1. Prototype bridge 是否是 V6.2 失败的主因，还是仅为加重因素？
2. Surface joint modeling 是否优于 native pointwise CPSC？
3. CVaR / worst-component risk 是否优于 mean risk？
4. 独立 calibration 能否校准“不确定性排序”，而不是只增加 UNKNOWN？
5. Surface-tail risk 是否能跨：
   - route-support；
   - static/disocclusion；
   - Actor；
   - 多 Actor interaction；
   - 不同 scene；
   - 不同前端或 Occupancy prior？
6. 最终改善是否能转化为：
   - GS + LogSim；
   - GS + WorldSim；
   - GS + NWM collision critic？

---

# 3. 推荐方法：SurfNCC

## 3.1 表示单位

对一个候选 proposal \(p\)，定义候选闭合或半闭合曲面：

\[
S_p=\{x_i\}_{i=1}^{N_p}
\]

每个 surface point / surface voxel \(x_i\) 必须具有：

```text
surface_id
patch_id
proposal_id
case_id
scene_id
frame_id
surface coordinate
surface normal
surface type
native prior cell identity
ray bundle identity
time support
Actor identity / canonical identity
lifecycle
hard evidence state
target label（仅 train/selection/calibration/eval 合法阶段）
```

曲面类型：

```text
route_support
static_disocclusion
actor_surface
actor_swept_surface
ground_contact
static_collision_boundary
```

---

## 3.2 “曲面结构”不等于旧 M1 Graph

V5.1 禁止的 Graph 是：

```text
稀疏语义 observation
→ 用空间邻接传播
→ 制造不存在的 ownership evidence
```

V6.3 的 surface topology 是：

```text
proposal 已经声明一个物理曲面
→ 用确定性几何邻接联合计算该曲面的风险
```

它不能：

- 把 FREE 传播成 OCC；
- 从无观测处制造标签；
- 更改 proposal 几何；
- 替代真实证据；
- 在未知空间自动生长曲面。

它只用于：

- 局部结构编码；
- 尾部风险聚合；
- connected-component 风险；
- patch-level abstention。

---

# 4. 原生输入合同

## 4.1 Occupancy prior

第一主后端继续固定为 IR-WM。

每个 native prior cell 至少保存：

```text
17-class logits
256D BEV latent
source-valid
native grid coordinate
native class argmax
entropy
top-1 / top-2 margin
temporal frame identity
source grid identity
```

若官方 forward 可合法提取：

```text
flow / residual feature
3D query feature
temporal token
```

可以在 P1 capability audit 后增加，但必须：

- 在任何质量结果前冻结；
- 对全部 train/calibration/confirmation/test 一致；
- 不为 legacy case 特制。

---

## 4.2 硬证据

继续使用：

```text
observed FREE
observed OCCUPIED
contradiction
outside lifecycle
Actor current/swept support
```

硬优先级：

```text
contradiction
→ UNKNOWN

observed FREE
→ FREE

observed OCCUPIED
→ OCCUPIED

outside lifecycle
→ UNKNOWN

other
→ learned completion
```

---

## 4.3 禁止 prototype bridge

任何正式 V6.3 arm 中不得存在：

```text
class → mean logits
class → mean BEV latent
argmax label → fabricated confidence
```

如果一个资产没有原生 sidecar：

```text
重新运行冻结 IR-WM
```

若协议不允许重跑：

```text
该资产只能用于历史 comparator
不能进入 V6.3 primary denominator
```

---

# 5. 曲面构造

## 5.1 proposal 输入

V6.3 不训练 proposal generator。

proposal source 在质量结果前冻结：

- V6 R9/R10 同刻跨前端 proposal；
- V6.1 / V6.2 的 route/static/Actor proposal scaffold；
- deterministic Actor translation / insertion；
- deterministic pseudo-hole；
- frozen generator proposal，仅作附加 stratum。

V6.3 研究的是：

```text
proposal 是否有资格进入 Physical State
```

不是：

```text
proposal 怎么生成得更漂亮
```

---

## 5.2 静态曲面

对 static / route / disocclusion proposal：

1. 在 proposal occupancy volume 上取 6-connected boundary；
2. 保存每个 boundary voxel 的原始 source identity；
3. 通过 deterministic connected components 分区；
4. 组件过大时使用固定 geodesic farthest-point seeds 切分 patch；
5. 不改变原始 occupied volume；
6. patch 分割只用于编码与风险聚合。

---

## 5.3 Actor 曲面

Actor 使用：

```text
canonical OBB / mesh / occupancy / SDF
+
T_world_actor(t)
+
lifecycle
```

表面在 canonical frame 建立：

\[
S_a^{canonical}
\]

运行时：

\[
S_{a,t}=T_a(t)S_a^{canonical}
\]

必须保存：

- canonical point identity；
- triangle/voxel identity；
- normal；
- lifecycle；
- swept surface；
- source actor ID。

---

## 5.4 Patch 合同

每个 patch：

```text
minimum points: 64
target points: 512
maximum points before chunking: 2048
```

数字仅作为默认起点。

P1 必须在不读取质量结果的情况下，根据：

- native voxel resolution；
- surface component size；
- 3090 memory probe；

冻结最终值。

不得根据 false-safe 结果调整 patch size。

---

# 6. SurfNCC 网络

## 6.1 Point encoder

每个点的输入：

\[
f_i=
[
l_i^{native},
z_i^{BEV},
u_i^{prior},
e_i^{hard},
d_i^{F},
d_i^{O},
x_i^{local},
n_i,
r_i,
t_i,
a_i
]
\]

包括：

- native logits；
- native BEV latent；
- entropy / margin；
- hard evidence one-hot；
- 到最近 observed FREE/OCC 的有符号距离；
- patch-local coordinate；
- normal；
- ray direction / hit ordering；
- temporal support；
- Actor current/swept/canonical features。

---

## 6.2 Surface encoder

默认结构：

```text
point MLP
→ local surface neighborhood aggregation
→ patch token
→ proposal token
→ point + patch + proposal heads
```

推荐第一版：

- Point MLP：2 layers；
- deterministic radius / surface-neighbor aggregation：2 blocks；
- patch self-attention：2–4 layers；
- proposal token：1 token；
- hidden dimension：256；
- mixed precision。

允许实现：

- Point Transformer；
- Set Transformer；
- sparse surface attention；
- deterministic mesh-neighborhood attention。

禁止：

- 从非 proposal 空间创建新 node；
- learned graph expansion；
- KNN label propagation；
- unrestricted dense voxel attention。

---

## 6.3 输出头

### 三态状态头

\[
P_i(F),P_i(O),P_i(U)
\]

### Hidden-FREE 风险头

\[
q_i^{HF}
=
P(
x_i \in FREE_{\mathrm{hidden}}
\mid
\text{method-visible evidence}
)
\]

### Occupied-authority 头

\[
q_i^{AUTH}
=
P(
x_i \text{具有足够正向 OCC 权威}
)
\]

### Patch 风险头

\[
q_j^{patch}
\]

### Proposal 风险头

\[
q_p^{proposal}
\]

Proposal head 不能单独决定 ACCEPT；它必须和可解释的 surface risk 一致。

---

# 7. 硬约束投影

沿用 V6.2 的 exact closed-form projection：

\[
\Pi_{\mathcal C(E)}
\]

要求：

```text
hard violations = 0
```

V6.3 新增 surface-level hard rejects：

```text
observed FREE connected conflict
lifecycle 外 Actor surface
Actor canonical identity mismatch
invalid surface component
non-finite geometry
source/eval identity overlap
```

注意：

> surface-level hard reject 只使用方法允许看到的证据。隐藏 O_eval 不能进入方法决策。

---

# 8. 曲面尾部风险

## 8.1 为什么不用 mean

V6.2 的错误是：

```text
整个曲面大部分区域风险低
+
局部区域严重侵入 FREE
→ mean 看起来可接受
```

WorldSim 的碰撞安全由局部最坏区域决定。

---

## 8.2 Point risk

当模型准备把点写为 OCC 时：

\[
\rho_i
=
P_i(O)\cdot q_i^{HF}
\]

也可以使用冻结的 monotone score：

\[
\rho_i
=
q_i^{HF}
+
\lambda_U P_i(U)
-
\lambda_A q_i^{AUTH}
\]

最终形式必须在 P1 prereg 时冻结。

---

## 8.3 Patch CVaR

对 patch \(S_j\)，固定 \(\alpha\)：

\[
R_j^{tail}
=
\operatorname{CVaR}_{\alpha}
\left(
\{\rho_i:i\in S_j\}
\right)
\]

默认研究起点：

```text
alpha = 0.90
```

即聚合最坏 10% 曲面点。

不允许在 legacy28/confirmation/test 上 sweep：

```text
0.80 / 0.90 / 0.95 / 0.99
```

若需要比较，只能作为在 P1 预注册的 development ablation。

---

## 8.4 Proposal risk

\[
R_p
=
\max_j R_j^{tail}
\]

同时报告：

```text
mean patch risk
max patch risk
top-component risk
largest connected conflict
```

Primary 使用：

```text
max patch CVaR
```

不能用全部点平均值替代。

---

## 8.5 Safe-OCC retention tail

对 target OCC surface：

\[
R_p^{miss}
=
\max_j
\operatorname{CVaR}_{\alpha}
\left(
1-P_i(O)
\right)
\]

避免通过：

```text
全部 UNKNOWN
```

获得低 false-safe。

---

# 9. 训练监督：曲面级反事实证据留出

V6.2 的独立 query dropout 被证伪。

V6.3 使用结构化 dropout：

## 9.1 Ray-bundle dropout

连续移除：

- 同一相机 sector；
- 邻近 LiDAR rays；
- 相邻 azimuth/elevation bundle。

## 9.2 Spatial-block dropout

移除连续三维 block，而不是独立 query。

## 9.3 Temporal-window dropout

移除连续 sweep/time interval。

## 9.4 Surface-patch dropout

移除完整 proposal patch 的方法证据，但保留 target label 作为监督。

## 9.5 Actor-support dropout

对 Actor：

- 移除一部分观测点；
- 保留 canonical/lifecycle；
- 测试模型是否错误扩张 Actor surface。

所有 dropout mask 必须：

- 由 train-only RNG 生成；
- 在读取 selection quality 前冻结；
- 不与 target/O_eval 决策共享；
- 保存到 run artifact。

---

# 10. 训练目标

总目标：

\[
L=
L_{state}
+
\lambda_{tail}L_{tail}
+
\lambda_{ret}L_{retention}
+
\lambda_{rank}L_{rank}
+
\lambda_{cons}L_{surface}
+
\lambda_{auth}L_{authority}
\]

---

## 10.1 State loss

仅对合法 supervised point：

\[
L_{state}
=
CE(P_i,y_i)
\]

hard conflict target 不进入普通 CE。

---

## 10.2 Hidden-FREE surface tail loss

对每个训练 proposal：

\[
L_{tail}
=
\frac{1}{|\mathcal P|}
\sum_p
\operatorname{CVaR}_{\alpha}
\left(
\{P_i(O):x_i\in S_p\cap F_{heldout}\}
\right)
\]

无 heldout FREE 的 surface 不贡献该项。

---

## 10.3 Safe-OCC retention

\[
L_{retention}
=
\frac{1}{|\mathcal P_O|}
\sum_p
\operatorname{CVaR}_{\alpha}
\left(
\{1-P_i(O):x_i\in S_p\cap O_{heldout}\}
\right)
\]

---

## 10.4 Proposal ranking

构造 matched safe/unsafe proposal：

\[
L_{rank}
=
\max(0,m+R_{safe}-R_{unsafe})
\]

目标：

```text
unsafe surface risk
>
safe surface risk
```

而不是只提高分类准确率。

---

## 10.5 Surface consistency

只在 proposal 曲面内部使用：

- normal consistency；
- local probability total variation；
- ray ordering；
- canonical Actor consistency。

不得把该项用于无 proposal 区域空间传播。

---

## 10.6 Positive authority

模型不能只因为“没发现 FREE 冲突”就写 OCC。

OCC 权威至少要求以下之一：

```text
observed OCC support
multi-time native prior consistency
Actor canonical support
surface closure support
independent modality support
```

定义：

\[
L_{authority}
\]

监督 `q_i^{AUTH}`。

推理时：

```text
low authority
→ UNKNOWN
```

---

# 11. 防止 all-UNKNOWN

正式报告同时包含：

```text
safe-OCC retention
UNKNOWN fraction
accepted surface area
accepted proposal count
Actor/static strata coverage
```

最低门：

```text
safe-OCC retention >= 0.60
source-valid UNKNOWN <= 0.60
accepted surface area >= matched baseline
```

最终数值在 P1 结合 denominator 冻结，但不能低于 V6.2 的反作弊原则。

---

# 12. 独立 case-level 风险校准

## 12.1 为什么不能 voxel-level calibration

同一 surface 内的 voxel 高度相关。

把几百万 voxel 当独立 calibration sample 会虚假放大样本量。

校准单位固定为：

```text
proposal / target unit / case
```

---

## 12.2 Calibration score

对 case \(c\)：

\[
s_c=
\max_{p\in c}R_p
\]

或：

\[
s_c=
\max_{p\in c}
\left[
R_p^{tail}
+
\beta R_p^{miss}
\right]
\]

score 在 calibration quality read 前冻结。

---

## 12.3 Risk loss

定义：

\[
\ell_{\lambda}(c)
=
\mathbf 1[
\text{ACCEPT}_{\lambda}(c)
\land
\text{FREEConflict}(c)>\tau_F
]
\]

也可同时报告连续损失：

\[
\ell_{\lambda}^{cont}(c)
=
\text{ACCEPT}_{\lambda}(c)
\cdot
\min
\left(
1,
\frac{\text{FREEConflict}(c)}{\tau_F}
\right)
\]

---

## 12.4 校准方法

推荐：

```text
Learn-Then-Test / Conformal Risk Control
```

候选 threshold grid 必须在 calibration read 前冻结。

选择：

> 满足预注册风险上界的最大 coverage threshold。

必须报告：

- calibration n；
- target risk \(\epsilon\)；
- confidence \(1-\delta\)；
- empirical risk；
- finite-sample upper bound；
- accepted coverage；
- group distribution。

---

## 12.5 Group calibration

strata：

```text
route_support
static_disocclusion
actor
multi_actor
```

只有 group calibration sample 数达到冻结下限时，才使用 group-specific threshold。

否则：

```text
fallback to global threshold
```

不得用小样本 group 阈值宣称条件安全保证。

---

## 12.6 Set-valued Physical State

输出：

```text
{OCCUPIED}
{FREE}
{OCCUPIED, UNKNOWN}
{FREE, UNKNOWN}
{UNKNOWN}
```

只有：

```text
{OCCUPIED}
```

允许写入新的 collision state。

其他包含 OCC 的非 singleton 集合：

```text
UNKNOWN / abstain
```

---

# 13. 数据纪律

## 13.1 Tier L：Legacy 机制集

```text
V6 legacy28
scene-0048 / scene-0242
28 cases
```

用途：

- retrospective mechanism audit；
- 与 V6.1/V6.2 同 denominator 对比；
- 验证 prototype bridge 是否为主要根因。

禁止：

- 训练；
- threshold fitting；
- calibration；
- candidate selection；
- 论文 fresh generalization claim。

---

## 13.2 Tier D：训练与 selection

优先复用 V6.2 已冻结、具有原生 sidecar 的六场：

```text
scene-0071
scene-0317
scene-0450
scene-0862
scene-1012
scene-1089
```

保持：

```text
4 train scenes
2 selection scenes
```

可以增加新 train scene，但必须：

- metadata-only selection；
- 不读取 quality；
- 与 C/H/T scene-disjoint；
- 先冻结 cohort。

---

## 13.3 Tier C：独立 calibration

要求：

```text
>= 6 fresh scenes
>= 72 target units
scene-disjoint from D/L/H/T
```

选择只使用：

- location；
- time-of-day；
- weather；
- actor count metadata；
- sensor completeness；
- processed frame count。

不得使用：

- Occupancy quality；
- proposal yield；
- false-safe；
- IR-WM score。

---

## 13.4 Tier H：one-shot confirmation

要求：

```text
>= 3 fresh scenes
>= 36 target units
```

候选、模型、threshold、score、strata policy 全部冻结后才可读取。

---

## 13.5 Tier T：exact-once test

要求：

```text
>= 4 fresh scenes
>= 48 target units
```

每个 test attempt：

- exclusive-create；
- 失败也消费；
- 不允许重跑同一 candidate；
- 不允许 test 后继续调该 family。

---

# 14. Method / Eval 证据隔离

每个 target 必须构建：

```text
E_method
E_dropout_train
E_calibration_target
E_eval_hidden
```

角色互斥：

- Train：method + train dropout labels；
- Selection：selection labels；
- Calibration：calibration labels，仅校准 threshold；
- Confirmation/Test：method decisions 先固化，再读取 eval hidden truth。

必须保存：

```text
role overlap audit
ray identity audit
point payload identity audit
scene/frame/camera/sweep identity
```

按本轮最新用户指令，V6.3 新产物不得新增 hash、checksum 或 fingerprint；身份只使用语义路径、task/run ID、
冻结配置版本与普通 Git 历史。

---

# 15. 正式 baseline matrix

| Arm | 原生 features | Surface encoder | Tail risk | Hard projection | Independent calibration |
|---|---:|---:|---:|---:|---:|
| B0 IR-WM argmax | 是 | 否 | 否 | 否 | 否 |
| B1 hard projection | 是 | 否 | 否 | 否/规则 | 否 |
| B2 Native CPSC-Lite | 是 | 否 | query mean | 是 | 否 |
| B3 Surface-Mean | 是 | 是 | surface mean | 是 | 否 |
| B4 Surface-Max | 是 | 是 | max | 是 | 否 |
| B5 Surface-CVaR | 是 | 是 | CVaR | 是 | 否 |
| M0 SurfNCC | 是 | 是 | CVaR + authority | 是 | 否 |
| M1 SurfNCC-Cal | 是 | 是 | CVaR + authority | 是 | **是** |

B2 必须重新用 native sidecar 执行，不能复用 prototype P6/P6R output。

---

# 16. P0 — V6.2 收口继承与 Git 前置

Task：

```text
WS-V63-P0-SCOPE-GIT-01
```

## 16.1 确认上游

必须确认：

```text
research/worldsim-v6.2-cpsc
HEAD = bcd4143...
worktree clean
remote synchronized
V6.2 status closed_negative
P7/P8 not unlocked
```

## 16.2 文档

更新：

- `docs/RESEARCH_STATUS.md`
- `docs/RESEARCH_FAILURES.md`
- `docs/EXPERIMENTS.md`

新增 V6.3 scope，不修改 V6.2 closeout 结论。

## 16.3 分支

若 V6.2 尚未合并 main：

```text
创建 integration branch
合入 V6.2
运行定向测试
普通 push main
```

随后从最新 main 创建：

```text
research/worldsim-v6.3-surface-tail
```

若同名存在，递增 `-rNN`。

禁止 force push。

---

# 17. P1 — 文献、novelty、协议冻结

Task：

```text
WS-V63-P1-SCOPE-NOVELTY-01
```

必须审计：

- RELIOcc；
- OCCUQ；
- α-OCC；
- QueryOcc；
- EvOcc；
- Conformal Risk Control；
- Non-exchangeable CRC；
- structured / segmentation conformal risk；
- CVaR；
- sparse/continuous occupancy；
- surface reconstruction with FREE-space constraints。

## Novelty 边界

以下单项不能独立作为贡献：

```text
使用原生 logits
使用 Point Transformer
使用 CVaR
使用 conformal calibration
使用 hard projection
使用 UNKNOWN
```

SurfNCC 可主张的组合贡献：

> **将原生 Occupancy 特征、物理曲面联合解码、proposal-surface CVaR 尾部风险、正向 OCC authority 与 case-level 风险控制统一为可固化的驾驶世界 Physical State Compiler。**

P1 必须在任何新质量结果前冻结：

- surface definition；
- patch construction；
- feature schema；
- CVaR alpha；
- loss weights；
- score；
- calibration procedure；
- cohort；
- gates；
- resource contract。

---

# 18. P2 — Native sidecar 全链

Task：

```text
WS-V63-P2-NATIVE-SIDECAR-01
```

## 18.1 开发侧

复用或重建 V6.2 P4：

```text
17D logits
256D BEV
query→cell mapping
source-valid
```

要求 72/72 existing D units 全部可读取。

## 18.2 Legacy28 原生 sidecar

V6.3 是新 scope，可以合法重跑冻结 IR-WM：

```text
scene-0048
scene-0242
target frames
```

但：

- 不读取 legacy O_eval 调整；
- 不训练；
- 不改 checkpoint；
- 不用 legacy quality 选模型。

## 18.3 Fresh C/H/T

按阶段生成。

Confirmation/Test sidecar 可以提前生成方法输入，但：

```text
target/eval evidence 必须保持 sealed
```

## 18.4 P2 Gate

- 原生 logits/features 100%；
- 无 prototype；
- source-valid mapping；
- hard evidence alignment；
- native grid round-trip；
- no target leakage；
- fresh process reload；
- resource audit。

---

# 19. P2D — Native CPSC-Lite retrospective diagnostic

Task：

```text
WS-V63-P2D-NATIVE-POINTWISE-DIAGNOSTIC-01
```

目的：

> 分离 prototype bridge 与 pointwise architecture 两个根因。

使用：

- V6.2 P5 frozen best；
- legacy28 新 native sidecar；
- 未改的 P6 gate；
- 不训练、不调 threshold。

结果解释：

### 若 native B2 仍 false-safe

说明：

```text
pointwise / risk objective 是主要根因
```

继续 SurfNCC。

### 若 native B2 达到原 gate

说明：

```text
prototype bridge 是主要根因
```

仍可继续 SurfNCC，但必须以 Native CPSC-Lite 作为强 baseline，且 SurfNCC 必须在 fresh cohort 显著优于它；不能仅凭复杂度成为主方法。

该诊断不能替代 fresh confirmation。

---

# 20. P3 — Surface corpus

Task：

```text
WS-V63-P3-SURFACE-CORPUS-01
```

每个 surface 保存：

```text
SURFACE_REGISTRY.jsonl
SURFACE_POINTS.npz / sharded arrays
PATCH_REGISTRY.jsonl
NATIVE_FEATURE_INDEX.jsonl
EVIDENCE_ROLE_INDEX.jsonl
PROPOSAL_REGISTRY.jsonl
```

必须统计：

- surface count；
- patch count；
- point count；
- component sizes；
- route/static/Actor strata；
- target FREE/OCC/UNKNOWN；
- hidden-FREE rate；
- authority source；
- actor/lifecycle coverage；
- surface normal validity。

## P3 negative tests

- empty surface；
- disconnected fragment；
- source/eval overlap；
- Actor lifecycle outside；
- inconsistent normal；
- invalid native mapping；
- prototype/fabricated feature；
- duplicate surface ID。

---

# 21. P4 — Capacity / Resource Probe

Task：

```text
WS-V63-P4-CAPACITY-01
```

只做：

- 1 train scene；
- 1 selection scene；
- 少量 optimizer steps；
- 不做 quality conclusion。

验证：

- loader；
- native feature gather；
- surface batching；
- hard projection；
- CVaR gradient；
- proposal token；
- checkpoint；
- determinism；
- peak memory；
- wall throughput。

---

# 22. 资源合同与多卡申请

## 22.1 默认单卡策略

允许的有界工程优化：

1. 冻结 IR-WM，提前物化 native sidecar；
2. FP16/BF16；
3. surface microbatch；
4. gradient accumulation；
5. activation checkpointing；
6. patch chunking；
7. CPU-side surface preprocessing；
8. 单次运行串行处理 strata；
9. memory-mapped sidecar。

这些不能改变科学 denominator。

---

## 22.2 禁止为了适配 3090 改什么

禁止：

- 降低 voxel resolution；
- 缩小 ROI；
- 减少 scene；
- 删除大 surface；
- 降低 target points；
- 缩短 temporal window；
- 丢掉 256D native latent；
- 改成 prototype；
- 只选容易 case；
- 修改 CVaR alpha；
- 修改 final gate。

---

## 22.3 Resource failure 定义

满足任一条件：

```text
同一 frozen minimum config 两次 OOM
合法 memory recovery 后仍无法前向/反向
单个 formal stage 预计 wall >24h
native sidecar / surface corpus 无法在磁盘安全上限内完成
必须 joint-finetune backbone 才能验证机制，但单卡无法容纳
```

则：

```text
status = blocked_resource
```

Codex 必须：

1. 保留 failed run；
2. 更新 failure ledger；
3. 释放 GPU；
4. 保持工作树 clean；
5. 停止该 GPU 分支；
6. 向用户申请资源。

---

## 22.4 资源申请格式

Codex 向用户说明：

```text
blocked stage
minimum faithful config
observed peak / OOM
已尝试的唯一有界恢复
为什么不能降低协议
推荐资源
```

推荐按瓶颈申请：

### 训练显存不足

```text
2× RTX 3090/4090 24GB
或
1× A100/H100 80GB
```

### 需要 end-to-end native backbone finetune

```text
至少 2× A100 40GB
优先 1–2× A100/H100 80GB
```

### 数据并行吞吐不足

```text
2–4× 24GB GPU
```

在用户提供资源前，不把 `blocked_resource` 写成算法 rejected。

---

# 23. P5 — SurfNCC 训练

Task：

```text
WS-V63-P5-SURFNCC-TRAIN-01
```

默认：

```text
frozen IR-WM
train SurfNCC only
```

正式配置建议起点：

```text
hidden dim: 256
surface blocks: 2
patch attention layers: 2
proposal layers: 1
CVaR alpha: 0.90
precision: fp16
optimizer: AdamW
one primary seed
```

精确数值在 P1 冻结。

## Selection objective

不能再用普通 weighted total loss。

主 selection 目标：

```text
surface hidden-FREE CVaR
+
proposal false-safe surrogate
+
safe-OCC retention constraint
+
coverage constraint
```

推荐 lexicographic：

1. hard violations = 0；
2. surface tail risk 最低；
3. safe-OCC retention 过门；
4. acceptance coverage 过门；
5. target accuracy 只作次级。

---

# 24. P6 — Fresh development matched AB

Task：

```text
WS-V63-P6-DEVELOPMENT-AB-01
```

依次运行：

```text
B0
B1
B2
B3
B4
B5
M0
```

禁止一开始直接完整融合。

## P6 主要问题

### Q1

Native B2 是否解除 prototype failure？

### Q2

Surface encoder 是否优于 native pointwise？

### Q3

CVaR 是否优于 mean/max？

### Q4

Authority head 是否减少“无冲突即 OCC”的误授权？

## P6 晋级门

必须同时满足：

- 至少 2 个 selection scenes 支持；
- hard violation = 0；
- proposal false-safe surrogate 相对 Native B2 降低；
- surface FREE-conflict CVaR 相对 Native B2 降低至少预注册幅度；
- safe-OCC retention 不低于门；
- accepted surface yield 不低于门；
- Actor/static 至少一个 stratum 各有非退化 coverage。

只冻结一个 M0 candidate。

---

# 25. P7 — Independent Calibration

Task：

```text
WS-V63-P7-CALIBRATION-01
```

输入：

- frozen M0；
- frozen score；
- frozen threshold grid；
- Tier C。

输出：

```text
CALIBRATION_POLICY.json
CALIBRATION_CASES.jsonl
RISK_CURVE.jsonl
GROUP_POLICY.json
FINITE_SAMPLE_REPORT.json
```

## P7 Gate

- calibration unit 数满足下限；
- target risk upper bound 通过；
- global policy 可用；
- group policy 仅在 sample sufficient 时启用；
- accepted coverage 非零；
- safe-OCC retention 过门；
- no threshold fit on legacy/H/T。

---

# 26. P8 — Legacy28 retrospective

Task：

```text
WS-V63-P8-LEGACY28-RETROSPECTIVE-01
```

candidate 与 calibration policy 已冻结后执行。

Primary historical gate：

```text
ACCEPT >= 5/28
false-safe = 0
R10 3/3 retained
new Actor >=1
new static/disocclusion >=1
accepted mask-area >=12%
worst accepted FREE conflict <=0.05
safe-OCC retention >=0.60
source-valid UNKNOWN <=0.60
```

若失败：

- 记录机制负结果；
- 不用 legacy 重新校准；
- fresh confirmation 是否解锁取决于 P6/P7 的预注册逻辑，不得事后修改。

建议强门：

> P8 false-safe 非零时，不解锁 P9。

---

# 27. P9 — One-shot Confirmation

Task：

```text
WS-V63-P9-CONFIRMATION-01
```

执行前冻结：

- source commit；
- model checkpoint；
- native sidecar schema；
- surface schema；
- risk score；
- calibration policy；
- thresholds；
- exact scene/case order；
- verdict rules。

通过条件：

```text
empirical false-safe = 0
risk bound / confidence report valid
accepted coverage >= frozen minimum
safe-OCC retention >= frozen minimum
>=2 independent scenes support
no catastrophic Actor/static regression
```

失败：

```text
candidate rejected
```

不得用 H 调参。

---

# 28. P10 — Exact-once Test

Task：

```text
WS-V63-P10-EXACT-ONCE-TEST-01
```

必须：

- exclusive attempt；
- attempt before quality read；
- failure consumes attempt；
- all cases remain denominator；
- undefined/blocked/UNKNOWN 全报告；
- no candidate change。

通过后才可称：

```text
SurfNCC confirmed
```

---

# 29. 核心评价指标

## 29.1 Proposal-level

```text
ACCEPT / ABSTAIN / REJECT
false-safe count
false-safe rate
safe valid yield
accepted mask-area
accepted surface-area
```

## 29.2 Surface-tail

```text
mean point risk
patch CVaR
max patch CVaR
worst connected component
95th/99th percentile conflict
surface FREE conflict
```

## 29.3 Retention

```text
safe-OCC retention
observed OCC retention
R10 retention
Actor/static coverage
```

## 29.4 Calibration

```text
empirical risk
finite-sample upper bound
coverage
risk-coverage curve
group coverage
prediction-set size
```

## 29.5 Anti-trivial

```text
UNKNOWN fraction
all-UNKNOWN rate
empty surface rate
accepted area
Actor gain
static gain
```

## 29.6 Runtime

```text
native sidecar time
surface compile time
risk inference time
GPU peak
package size
collision query latency
```

---

# 30. 多 Actor压力测试

只有 P10 通过后解锁。

Task：

```text
WS-V63-P11-MULTIACTOR-STRESS-01
```

至少：

```text
3 scenes
2 / 3 / 4 Actor
>=150 proposals
```

比较：

```text
AABB
OBB
native CPSC pointwise
SurfNCC surface risk
SurfNCC + swept collision
```

主指标：

```text
0 empirical false-safe
Clopper-Pearson upper bound
accepted yield
Actor-static penetration
Actor-Actor overlap
lifecycle phantom
```

---

# 31. 三个下游任务

## 31.1 GS + LogSim

SurfNCC 用于：

- 原 case 物理状态资格；
- collision state；
- Actor lifecycle；
- 原始事件不被生成内容篡改；
- exact replay。

验收：

```text
same log
→ same physical state
→ same collision labels
→ same sensor replay
```

---

## 31.2 GS + WorldSim

SurfNCC 用于：

- 新路线；
- disocclusion；
- Actor 插入；
- 多 Actor 编辑；
- route-support；
- collision volume。

主指标：

```text
safe valid yield
false-safe
verified world area
verified route length
Actor insertion yield
```

---

## 31.3 GS + NWM RL

P10/P11 通过后，只先解锁 collision critic：

```text
Real-only
Real + naive generated
Real + SurfNCC verified
```

主指标：

- collision false-safe；
- unsafe-action recall；
- safe-action precision；
- risk calibration。

Reward/progress/stuck/comfort 为次指标。

若 Real-only 与 SurfNCC 一样好：

```text
诚实报告无增量
```

不得削弱 baseline。

---

# 32. Auto Research Loop

每轮：

```text
OBSERVE
→ DIAGNOSE
→ HYPOTHESIS
→ FAILURE/NOVELTY GATE
→ PREREGISTER
→ EXECUTE
→ AUDIT
→ REFLECT
→ PROMOTE / REJECT
```

## 允许自主调整

- surface encoder 实现；
- batching；
- resource routing；
- development-only diagnostic；
- 新 hypothesis；
- 代码修复；
- 文档；
- failure ledger。

## 不允许自主调整

结果可见后不能改当前实验：

- CVaR alpha；
- cohort；
- primary score；
- threshold grid；
- acceptance gate；
- confirmation/test；
- comparator；
- denominator。

需要改变时：

```text
关闭当前 hypothesis
注册新 hypothesis
```

---

# 33. 允许的算法 recovery

## Recovery A：Native interface

仅当：

- feature mapping 错误；
- sidecar schema 缺字段；
-坐标不一致；

允许一次新 run 修复。

## Recovery B：Resource

按第 22 节执行；必要时向用户申请多卡。

## Recovery C：Surface model capacity

仅当训练/selection 同时显示：

```text
underfit
+
tail risk 未下降
+
训练 loss 高
+
没有 overfit
```

才允许一次预注册 capacity recovery。

不得：

- 根据 legacy/H/T 结果加宽模型；
- seed sweep；
- CVaR alpha sweep；
- threshold sweep。

## Calibration failure

P7 失败不得在同一 calibration set 重拟 score family。

必须关闭当前 score，重新定义新 family 并重新建立独立 calibration。

---

# 34. Stop Rules

## Stop 1：Native pointwise 已经足够

若 Native B2 在 fresh + legacy 达到所有主门：

- SurfNCC 仍必须证明 fresh significant gain；
- 若无增量，停止复杂方法，论文转为 native-interface correction。

## Stop 2：Surface encoder 无增量

若 B3 不优于 Native B2：

```text
关闭 surface architecture family
```

不继续 CVaR。

## Stop 3：CVaR 无增量

若 B5 不优于 B3/B4：

```text
关闭 tail-risk objective
```

不包装成 full method。

## Stop 4：校准只能 all-UNKNOWN

若 P7 通过风险门只能靠：

```text
coverage 接近 0
```

则 rejected。

## Stop 5：Legacy false-safe 非零

不解锁 confirmation。

## Stop 6：Confirmation false-safe 非零

family closed。

## Stop 7：资源不足

按资源合同停止并向用户申请多卡，不写 algorithm rejected。

---

# 35. ArXiv 技术报告结构

## 题目

**SurfNCC: Native-Feature Surface Tail-Risk Control for Verifiable Driving World Compilation**

## 1. Motivation

- Oracle Physical State work；
- learned argmax false-safe；
- V6.2 0 hard violation 仍 4/4 false-safe；
- query mean 不能控制 surface tail。

## 2. Problem

定义：

- candidate surface；
- method evidence；
- hidden evaluation；
- surface false-safe；
- safe valid yield。

## 3. Method

- native prior interface；
- hard projection；
- surface encoder；
- CVaR；
- authority；
- calibration；
- SceneIR bake。

## 4. Experiments

- native vs prototype；
- pointwise vs surface；
- mean/max/CVaR；
- calibration；
- legacy28；
- fresh confirmation/test；
- multi-Actor；
- downstream。

## 5. Negative Evidence

保留：

- CPSC-Lite；
- evidence dropout；
- all-UNKNOWN；
- calibration sample limits；
- resource limits。

## 6. Limitations

- finite calibration；
- exchangeability；
- IR-WM dependency；
- proposal generator fixed；
- not absolute real-world safety guarantee。

---

# 36. 计划交付文件

建议在 V6.3 branch 创建：

```text
docs/WORLDSIM_V6_3_NATIVE_SURFACE_TAIL_RISK_COMPILER_PLAN.md

docs/autoresearch/worldsim_v63/
  AUTORESEARCH_STATE.current.json
  HYPOTHESES.jsonl
  REFLECTIONS.jsonl
  P0_SCOPE.md
  P1_NOVELTY.md
  ...

configs/worldsim_v63/
  p0_scope_v1.yaml
  p1_method_contract_v1.yaml
  p2_native_sidecars_v1.yaml
  p3_surface_corpus_v1.yaml
  p4_capacity_v1.yaml
  p5_surfncc_train_v1.yaml
  p7_calibration_v1.yaml
  p8_legacy28_v1.yaml
  p9_confirmation_v1.yaml
  p10_test_v1.yaml

motion_proj/worldsim_v63/
  native_features.py
  surface_builder.py
  surface_dataset.py
  surface_encoder.py
  tail_risk.py
  hard_projection.py
  authority.py
  calibration.py
  compiler.py

scripts/worldsim_v63/
  run_*.py

tests/worldsim_v63/
```

禁止新建第二套 failure ledger。

---

# 37. 第一轮执行顺序

```text
P0  V6.2 inheritance + branch
↓
P1  novelty / method / cohorts / gates / resource freeze
↓
P2  native sidecars
↓
P2D native pointwise diagnostic
↓
P3  surface corpus
↓
P4  capacity/resource probe
↓
P5  SurfNCC train
↓
P6  fresh matched AB
↓
P7  independent calibration
↓
P8  legacy28 retrospective
↓
P9  one-shot confirmation
↓
P10 exact-once test
↓
P11 multi-Actor stress
↓
LogSim / WorldSim / NWM critic
```

---

# 38. 对 Codex Agent 的最终指令

你接手的是 WorldSim V6.3。

你的北极星不是：

```text
让平均 Occupancy 指标更高
让 UNKNOWN 更低
让某个 legacy aggregate 看起来更好
```

而是：

> **使用原生 Occupancy feature 和真实硬证据，对候选物理曲面的最坏局部风险进行联合建模，在独立校准下只将安全资格充分的曲面写入 SceneIR Physical State。**

执行要求：

1. 先读当前状态、V62-F01/F05/F06/F07 和 P6R closeout；
2. 保持 V6.2 closed negative；
3. 不使用 prototype bridge；
4. 不使用 legacy O_eval 调参；
5. 不把 voxel 当独立 calibration sample；
6. 不用 mean query risk 替代 surface tail；
7. 不用高 UNKNOWN 冒充安全；
8. 不用无冲突冒充 OCC authority；
9. 每个正式实验先 commit/preregister；
10. 失败即写 failure ledger 并反思；
11. 普通工程失败自主恢复；
12. 默认单卡 3090；
13. 若冻结最小实验发生真实资源阻塞，停止 GPU 工作、释放资源、保存证据并向用户申请多卡；
14. 未经用户同意不得擅自改变硬件预算后继续；
15. confirmation/test 不得调参；
16. 达到 stop rule 时立即关闭 family，不继续 sweep。

---

# 39. 一句话研究主张

英文：

> **We compile physical surfaces—not independent voxels—by combining native occupancy features, exact sensor constraints, tail-risk optimization, and case-level calibration, admitting a surface into the driving world only when its worst local hidden-free risk is controlled.**

中文：

> **我们不再独立判定每个体素，而是联合编译完整物理曲面：融合原生 Occupancy 特征与精确传感器约束，直接优化隐藏 FREE 的曲面尾部风险，并经 case-level 独立校准后，才允许曲面进入驾驶世界。**

---

# 40. 主要参考

- RELIOcc：`https://www.ijcai.org/proceedings/2025/0220.pdf`
- OCCUQ：`https://github.com/ika-rwth-aachen/OCCUQ`
- α-OCC：`https://arxiv.org/abs/2406.11021`
- QueryOcc：`https://arxiv.org/abs/2511.17221`
- EvOcc：`https://openaccess.thecvf.com/content/CVPR2025/html/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.html`
- Conformal Risk Control：`https://openreview.net/forum?id=33XGfHLtZg`
- Non-Exchangeable Conformal Risk Control：`https://openreview.net/forum?id=j511LaqEeP`
- Conformal Semantic Segmentation：`https://arxiv.org/abs/2405.05145`
- Controlling False Positives in Segmentation：`https://arxiv.org/abs/2511.15406`
- CVaR：Rockafellar & Uryasev, *Optimization of Conditional Value-at-Risk*。
