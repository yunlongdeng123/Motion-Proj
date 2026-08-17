# WorldSim V5.1 M1-Only Research Plan
## Observability-Aware Bayesian Gaussian Ownership for Driving World Simulation

## 2026-08-18 machine amendment：H evaluation-only two-phase contract

本 amendment 只把既有 Stage B 指标与 split 原则机器化，不改变方法路线或 validation/test exact-once 约束。

- Phase A（r013）只读 H evaluation frames=`2/42/82/122/162`、cameras=`0/1/2`，用 r010 frozen PCA
  transform official DINOv2 features；PCA fit、membership proxy、renderer、uplift feature 与 quality read 全为 false。
- Phase B 必须在 Phase A result freeze 后另行 clean preregister，且首次读取 r012 B0/B1 质量前固定：
  same-Gaussian cross-view cosine repeatability、same-actor cosine、actor-background cosine/margin、heldout
  reprojection cosine、Background/Rigid coverage、pair sampling 与 abstention。
- membership 只能声明为 `model_membership_proxy_not_ground_truth`，仅供 evaluation，不得作为方法、PCA 或 uplift 输入。
  每 actor 至少 32 个 covered Gaussian，最多 4,096 pairs，seed=`20260814`；无 eligible actor 的 scene abstain。
- H pass gate 固定为：至少 2/3 scenes evaluable；至少 2 scenes 的 B1 actor-background margin `>0`；scene-balanced
  B1 margin mean `>0`；mean Rigid coverage `>=0.60`；heldout reprojection cosine 的 scene-balanced
  `B1-B0 >= -0.01`。未过 H gate 不得读取 S/C，过门后也只能按既有 S→C development 顺序推进。
- final heldout remainder=`4`、validation/test、KITTI、M2/M3 保持锁定；`V51-F15` 在 Phase B proxy/evaluation
  证据审计完成前保持 active。

> **目标**：只推进 M1，把已经在 V5 中显示正信号的 Bayesian Unary 升级成一个 **可跨场景稳定工作、可严格消融、具有顶会论文方法贡献潜力** 的 Gaussian ownership / semantic sidecar 方法。
> **M2 / M3**：全部冻结为 `pending`，V5.1 期间不推进、不调参、不占用研究预算。

---

# 0. 一页执行结论

V5 已经给出三个关键事实：

1. **Bayesian / reliability-aware Unary 不是死路。**
   - B1/B3 在 scene-0471 上相对 B0 同时改善：
     - IoU
     - Boundary F1
     - Brier
     - ECE
     - NLL
     - FP semantic mass
   - 因此 V5.1 必须保留 Bayesian Unary 作为起点。

2. **V5 Graph 的失败不能简单解释为“Graph 没用”。**
   - scene-0471：G3 有正收益；
   - scene-1087：G3 为负；
   - scene-0379：混合；
   - 最终 replication 只有 `3/6` scene×unary unit 为正。
   - 失败场景同时具有非常稀疏的 SAM 可观测性，例如 scene-1087 只有 `2/30` view 有目标。
   - 当前首要假设变成：

\[
\text{Sparse / missing observations}
\rightarrow
\text{weak unary support}
\rightarrow
\text{graph 被迫做 semantic completion}
\rightarrow
\text{scene-dependent propagation failure}
\]

3. **V5.1 不一次性实现全部 16 个 idea。**
   - 按最小改动到大改动逐层升级；
   - 外部论文方法必须先做 **faithful port / 原始机制迁移**；
   - 原始迁移不 work：直接舍弃；
   - 原始迁移 work：才允许添加我们的创新；
   - 每一级都必须 matched ablation；
   - 不允许把多个 idea 一次堆在一起后无法归因。

V5.1 主路线：

```text
V5 Bayesian Unary
        ↓
Stage A
修 observation / visibility / unknown
        ↓
Stage B
原样迁移 Learning-Free 2D→3D Feature Uplift
        ↓
Stage C
原样迁移 Semantic-Gated Graph
        ↓
Stage D
如果 raw-Gaussian graph 不稳定：
提升 Graph Node → Super-Primitive / Anchor
        ↓
Stage E
如果固定 graph propagation 仍不稳：
替换为 Identity Embedding / Progressive Grouping
        ↓
Stage F
如果 Graph 整体不成立：
保留 Bayesian Unary，
切换到 Kernelized Bayesian / Graph-Free Conservative Propagation
        ↓
Fresh Dev Confirmation
        ↓
8-scene Validation
        ↓
Freeze
        ↓
20-scene Exact-Once Test
        ↓
KITTI Frozen Cross-Dataset
```

最终 Method 必须收敛到 **2–3 个核心模块**，不能把 16 个 idea 全部写成贡献。

---

# 1. 当前仓库与 V5 事实

## 1.1 项目

```text
/root/autodl-tmp/motion_proj
```

V5 最终分支：

```text
research/worldsim-v5-structdelta
```

V5 最终已知 HEAD：

```text
f7566beb4d37115700a1d702f524d99cbab24b4e
```

Codex 接手时必须以远端 canonical Git / run / summary / manifest 为准。

如果现场已经存在更晚的 V5.1 提交：

```text
禁止退回旧 HEAD
```

---

## 1.2 V5 M1 当前正式结论

必须保留以下事实，不允许重写历史：

```text
M1 overall:
rejected

Bayesian Unary:
有方向性正信号

V5 physical graph:
scene-disjoint replication failed

M1B semantic split:
not unlocked / rejected
原因：
boundary ambiguity 不是主要误差来源
```

关键 V5 结果：

```text
scene-0471:
B1/B3 unary positive
G3 positive but small

scene-1087:
SAM usable target views = 2/30
evaluation usable views 极少
G3 negative

scene-0379:
SAM usable target views = 6/30
unary strong positive but FN tradeoff
G3 mixed
```

V5 graph frozen gate：

```text
positive scene×unary units:
3 / 6

required:
>= 4 / 6
```

因此 V5 Graph rejection 是：

> **均值略正，但复制稳定性不足。**

不是：

> 所有 Graph idea 都已被证伪。

---

# 2. V5.1 Scope Freeze

Task：

```text
WS-V51-P0-M1-SCOPE-FREEZE-01
```

V5.1 只允许修改 / 新增：

```text
M1 semantic observation
M1 Bayesian unary
M1 semantic feature uplift
M1 graph / grouping / propagation
M1 semantic sidecar
M1 evaluator
M1 M1-only paper artifacts
```

明确禁止：

```text
M2 repair
M2 geometry
M2 router
M3 temporal delta
M3 constraint projection
new diffusion model
new driving reconstruction backbone
StreetGS base retraining parameter search
AD-GS modification
fresh validation tuning
fresh test tuning
```

M2/M3 registry：

```text
WS-V51-M2 = pending
WS-V51-M3 = pending
```

## 2.1 Failure ledger 绑定

本计划启动前必须渐进式复核统一账本 `docs/RESEARCH_FAILURES.md`，不得创建 V5.1 独立 failure 文档。

- scope / data / protocol：`V5-F09`、`V5-F11`–`V5-F14`、`V5-F18`；
- Bayesian unary / evaluation：`V5-F20`–`V5-F26`；
- replication / infrastructure / graph boundary：`V5-F29`–`V5-F33`。

每个 config/run metadata 必须写 `failure_ledger_refs`；每次收口写 `failure_ledger_delta`，无新增失败时显式写
`none`。出现 blocked/rejected、分母错误、工程恢复、算法假设推翻或旧风险解除时，在同一逻辑提交中窄改统一账本。

---

# 3. V5.1 科研问题

核心问题不再是：

> “Graph 能不能让 Gaussian ownership 更平滑？”

而是：

> **在自动驾驶场景中，2D SAM 对动态 actor 的观测高度稀疏、遮挡且跨视角不一致时，如何从可靠的 Bayesian semantic observations 构造一个不会产生语义泄漏的 3D Gaussian ownership field？**

拆成四个子问题：

### Q1 Observation

```text
什么情况下一个 view 对 Gaussian 的语义观测是真的“有效观测”？
```

### Q2 Missingness

```text
没有观测
是否被错误当成 background / negative evidence？
```

### Q3 Structure

```text
需要传播时，
传播应该发生在 raw Gaussian、Anchor、Super-Primitive
还是 identity embedding 上？
```

### Q4 Propagation

```text
应该做：
global diffusion
progressive growing
graph cut
kernelized Bayesian inference
还是根本不传播？
```

---

# 4. V5.1 核心科研假设

## H1：Missing observation ≠ negative semantic evidence

建立：

\[
v_{iv}\in\{0,1\}
\]

表示 Gaussian \(i\) 在 view \(v\) 是否有资格产生语义观测。

语义 ownership：

\[
z_i\in\{0,1\}
\]

必须建模：

\[
P(m_{iv}\mid z_i,v_{iv}=1)
\]

而不是：

\[
P(m_{iv}\mid z_i)
\]

当：

\[
v_{iv}=0
\]

该 view 对 posterior：

```text
不做正更新
不做负更新
```

---

## H2：UNKNOWN 是真实状态，不应被强迫成 foreground/background

对于：

- 可见度不足；
- SAM response ambiguous；
- evidence count 太低；
- cross-view disagreement 太大；

节点应保持：

```text
UNKNOWN
```

而不是强行：

```text
actor / background
```

核心目标：

\[
\text{precision before coverage}
\]

尤其在自动驾驶 asset editing 中：

> 错删背景比暂时不知道一个 Gaussian 属于谁更危险。

---

## H3：Graph 应该 regularize evidence，不应该 invent evidence

如果一个区域：

```text
effective semantic observations ≈ 0
```

Graph 不允许长距离补全 actor ownership。

需要显式：

\[
g_i^{prop}
=
g_i
\cdot
\mathbb{1}
[
n_i^{eff}\ge n_{min}
]
\]

或者：

```text
UNKNOWN remains UNKNOWN
```

---

## H4：Graph edge 必须包含 semantic evidence

V5 G3：

```text
Mahalanobis
+
normal
+
boundary barrier
```

主要表示：

> 几何上是否相邻。

但并不能充分表示：

> 是否属于同一对象。

V5.1 必须至少比较：

\[
W_{ij}^{geo}
\]

与：

\[
W_{ij}^{geo}
\cdot
W_{ij}^{semantic}
\]

---

## H5：如果 raw Gaussian 是错误 graph node，则应更换 node，而不是继续调 edge

如果 semantic observation 在单 Gaussian 上过稀：

```text
raw Gaussian
```

可能不是合理的推理单元。

候选：

```text
Super-Primitive
Anchor
Feature Component
Instance Embedding
```

---

# 5. 数据协议：不要重新浪费 V5 的 clean validation / test

V5 的 fresh cohort：

```text
development = 8
validation  = 8
test        = 20
```

V5 没有读取 validation / test quality。

因此 V5.1 可以继续使用：

```text
8 validation
20 test
```

作为 clean confirmatory data。

禁止重新选 validation/test。

---

# 6. Development 重新分层

V5 development 8 scenes：

```text
0471
1087
0379
0998
0359
0875
0535
0436
```

V5.1 不把 8 scene 全部拿来反复调。

---

## 6.1 H：Historical Diagnostic

已经被 V5 深度观察：

```text
H1 scene-0471
H2 scene-1087
H3 scene-0379
```

允许：

- debug
- threshold study
- mechanism diagnostic
- feature visualization
- ablation screening
- profiler
- failure analysis

不允许用这 3 scene 单独宣称 generalization。

---

## 6.2 S：Development Screening

按原 V5 cohort 顺序固定：

```text
S1 scene-0998
S2 scene-0359
```

用途：

```text
candidate screening
```

规则：

- 某 candidate 在 H 上选定后才能进入 S；
- S 上不得重新调 candidate parameter；
- S 结果不允许用于重新回头调整该 candidate；
- 若失败，arm 直接 rejected。

---

## 6.3 C：Development Confirmation

剩余：

```text
C1 scene-0875
C2 scene-0535
C3 scene-0436
```

用途：

```text
development confirmation
```

只有最终不超过：

```text
2 个 candidate families
```

允许读取 C quality。

C 读取前必须冻结：

```text
candidate family
method config
threshold
feature backbone
graph parameters
propagation steps
abstain policy
```

---

# 7. V5.1 Evaluation Split 内部纪律

每 scene 保持 V5 的：

```text
evidence views
development evaluation views
heldout remainder
```

额外约束：

### H scenes

允许读取：

```text
evidence
development evaluation
```

### S scenes

运行结束后一次性读取 development evaluation。

### C scenes

最终候选 freeze 后一次性读取。

### Validation

所有 V5.1 方法 freeze 后一次性读取。

### Test

validation PASS 后：

```text
V51_TEST_FREEZE.json
```

生成并提交以后 exact-once 读取。

---

# 8. Baseline Freeze

必须永远保留：

## U0：V5 B0

```text
hard vote unary
```

## U1：V5 B1

```text
reliability-weighted hard observation
```

## U2：V5 B3

```text
reliability-weighted SAM soft-probability unary
```

## G0

```text
best frozen Bayesian unary
without graph
```

## G-V5

```text
V5 frozen G3
Mahalanobis + normal + boundary barrier
```

所有新方法必须和：

```text
U1/U2
G0
G-V5
```

matched comparison。

---

# 9. V5.1 总执行原则：Small → Faithful Port → Innovation → Replacement

每个外部 idea 使用：

```text
PAPER-0:
faithful mechanism port

PAPER-1:
only if PAPER-0 works
add V5.1 innovation
```

禁止：

```text
PAPER-0 还没跑
就直接做“改进版”
```

否则无法判断：

> 是原论文机制有效，还是我们自己的修改有效。

---

# 10. Stage A：只修 Bayesian Unary，不碰 Graph

Task：

```text
WS-V51-M1-A-UNARY-OBSERVABILITY-01
```

这是最高优先级。

---

# 10.1 A0：V5 Bayesian Unary Exact Replay

必须 byte / metric exact replay：

```text
scene-0471
scene-1087
scene-0379
```

确保：

```text
V5.1 baseline
==
V5 canonical unary
```

否则禁止后续实验。

---

# 10.2 A1：Visibility-Masked Bayesian Update

对应用户 Idea 1。

核心：

\[
v_{iv}
=
\mathbb 1[
\text{Gaussian contribution}_{iv}
>
\tau_{vis}
]
\]

仅：

\[
v_{iv}=1
\]

的 observation 进入：

\[
\alpha_i,\beta_i
\]

如果：

\[
v_{iv}=0
\]

则：

\[
\Delta\alpha_i
=
\Delta\beta_i
=
0
\]

### 原样迁移原则

先只做：

```text
visibility / semantic decoupling
```

禁止同时加：

- DINO
- graph
- anchor
- temporal
- LiDAR kernel

### Ablation

```text
U2 B3
vs
A1 B3 + visibility mask
```

---

# 10.3 A2：Semantic UNKNOWN / ABSTAIN

对应 Idea 2。

状态：

```text
ACTOR
BACKGROUND
UNKNOWN
```

定义 UNKNOWN 条件：

- effective observation count too low
- posterior entropy high
- SAM cross-view disagreement high

第一版只能使用：

```text
frozen threshold
```

来自 H scene training/evidence statistics。

禁止在 evaluation quality 上调。

### 输出

```text
posterior_actor
posterior_background
unknown_probability
effective_observation_count
```

### 评估

除了传统 IoU / BF1：

- coverage
- error@coverage
- selective semantic risk
- UNKNOWN precision
- UNKNOWN recall on errors

目标：

> UNKNOWN 应优先吸收错误，而不是随机降低 coverage。

---

# 10.4 A3：Correlation-Aware Effective Count

V5 暴露过一个重要机制风险：

> 多个 observation 可能高度相关，但 Beta posterior concentration 仍像独立观测一样增长。

V5.1 显式比较：

### A3-0

普通 fractional count：

\[
\alpha
=
\alpha_0
+
\sum_v r_{iv}p_{iv}
\]

### A3-1

effective count：

\[
n_i^{eff}
=
\frac{
(\sum_v r_{iv})^2
}{
\sum_v r_{iv}^2+\epsilon
}
\]

再让 posterior concentration 受：

\[
n_i^{eff}
\]

限制。

目的：

```text
10 个高度相关 view
≠
10 个独立证据
```

---

# 10.5 A4：CIF-Style Random Variable Decoupling

参考：

```text
Consistent Instance Field
CVPR 2026
```

Faithful mechanism port：

分开：

\[
P(o_i=1)
\]

occupancy / existence，

\[
P(z_i=a\mid o_i=1)
\]

conditional identity，

以及：

\[
P(v_{iv}=1)
\]

visibility。

V5.1 中第一版不引入 CIF 的完整 deformable field。

只迁移：

> **visibility / occupancy / identity probabilistic decoupling**

避免把：

```text
not visible
```

误解释成：

```text
not actor
```

---

# 10.6 Stage A Ablation Matrix

| Arm | Bayesian | Visibility | UNKNOWN | Effective Count | Conditional Identity |
|---|---:|---:|---:|---:|---:|
| U1 | B1 | old | no | no | no |
| U2 | B3 | old | no | no | no |
| A1 | B3 | yes | no | no | no |
| A2 | B3 | yes | yes | no | no |
| A3 | B3 | yes | yes | yes | no |
| A4 | decoupled | yes | yes | yes | yes |

---

# 10.7 Stage A Gate

只用 H：

候选必须：

```text
>= 2/3 scenes Boundary F1 positive vs U2

mean Boundary F1 > U2

mean IoU >= U2

mean FN semantic mass delta <= +0.02

Brier or ECE improves

UNKNOWN-enabled arm:
coverage >= 60%
and
abstained subset error > accepted subset error
```

最多保留：

```text
2 个 Stage-A candidate
```

进入 S。

---

# 10.8 Stage A Screening Gate

S1/S2：

```text
2 / 2 scenes
Boundary F1 non-negative

至少 1 / 2 clearly positive

mean Boundary F1 > U2

FN delta <= +0.02

calibration 不关键退化
```

若所有 Stage A arm 都失败：

> 保留 U1/U2，进入 Stage B，不再继续复杂化 Bayesian family。

---

# 11. Stage B：Learning-Free Renderer Back-Projection

Task：

```text
WS-V51-M1-B-LUDVIG-UPLIFT-01
```

对应 Idea 15。

参考：

```text
LUDVIG
ICCV 2025
```

第一阶段只迁移：

> 2D feature → Gaussian feature 的 learning-free uplift operator

不立即做 graph diffusion。

---

# 11.1 B0：V5 Current Intersection Lift

保留现有：

```text
Gaussian-pixel intersection
→ per-view aggregation
```

作为 baseline。

---

# 11.2 B1：LUDVIG-Style Rendering Transpose

标准 renderer：

\[
I
=
W G
\]

使用归一化转置：

\[
F_G
=
D^{-1}
W^\top
F_{2D}
\]

其中：

- \(W\)：alpha compositing contribution
- \(F_{2D}\)：DINOv2 / SAM encoder dense feature
- \(D\)：normalization

必须：

```text
learning-free
base Gaussian immutable
feature sidecar only
```

---

# 11.3 Feature Backbone

Faithful port 第一版：

```text
DINOv2
```

原因：

LUDVIG 使用 DINOv2 pairwise feature similarity。

禁止第一版同时比较：

```text
DINOv2
SigLIP2
CLIP
SAM encoder
```

先验证论文机制。

如果 DINOv2 port work：

第二阶段才比较：

```text
DINOv2
vs
SigLIP2
```

作为 driving-specific backbone ablation。

---

# 11.4 B1 Metrics

除了 ownership：

- feature repeatability across views
- same-actor cosine similarity
- actor-background feature margin
- feature coverage
- uplift wall time
- sidecar bytes
- peak VRAM

---

# 11.5 B Gate

如果 DINO uplift 无法使：

```text
same-actor similarity
>
actor-background similarity
```

形成稳定 separation：

```text
reject feature-graph path
```

不要继续 Graph。

如果成立：

解锁 Stage C。

---

# 12. Stage C：Faithful Semantic-Gated Graph Port

Task：

```text
WS-V51-M1-C-SEMANTIC-GRAPH-01
```

第一优先参考：

```text
LUDVIG
```

第二参考：

```text
SAM-guided Graph Cut
```

---

# 12.1 C0：LUDVIG Faithful Port

先尽量保持其机制：

```text
3D KNN
+
DINOv2 pairwise similarity
+
target-conditioned relevance
+
graph diffusion
```

边：

\[
W_{ij}
=
\mathbb 1[j\in KNN(i)]
\cdot
S_{DINO}(i,j)
\cdot
G_{target}(i,j)
\]

第一版禁止加入：

- V5 Mahalanobis
- normal
- depth barrier
- LiDAR kernel
- motion edge
- UNKNOWN innovation

原因：

> 先判断 LUDVIG 原始机制迁移到 driving Gaussian ownership 是否成立。

---

# 12.2 C0 Seeds

Graph seed：

```text
best Stage-A Bayesian unary
```

但必须额外跑：

```text
U2 + C0
```

和：

```text
best Stage-A + C0
```

以分离：

```text
graph gain
```

与：

```text
unary gain
```

---

# 12.3 C0 Gate

H：

```text
>= 2/3 scenes positive
mean Boundary F1 positive
mean FN delta <= +0.01
graph leakage lower than V5 G3
```

S：

```text
2/2 non-negative
```

若 C0 失败：

```text
LUDVIG graph port rejected
```

直接进入 Stage D。

禁止：

```text
继续调 graph lambda 直到变正
```

---

# 12.4 C1：Bayesian Observability-Gated Diffusion

只有 C0 work 才允许。

这是 V5.1 的第一个原创候选。

节点 gate：

\[
q_i
=
f(
p_i,
u_i,
n_i^{eff},
visibility_i
)
\]

传播：

\[
h_i^{t+1}
=
q_i h_i^0
+
(1-q_i)
\sum_j
\tilde W_{ij}h_j^t
\]

高 confidence：

```text
anchor to unary
```

低 confidence：

```text
允许有限传播
```

UNKNOWN / no-observation：

```text
禁止无限远传播
```

---

# 12.5 C2：SAM Multi-View Co-occurrence Edge

对应 Idea 9。

只有 C0 已经 work 后测试。

定义：

\[
S_{SAM}(i,j)
=
\frac{
\sum_v
r_v
\mathbb 1[
i,j
\text{ co-occur in same SAM mask}
]
}{
\sum_v r_v+\epsilon
}
\]

创新边：

\[
W_{ij}
=
W_{ij}^{LUDVIG}
\cdot
S_{SAM}(i,j)
\]

测试目的：

> SAM co-occurrence 是否能进一步阻断 vehicle-road leakage。

---

# 12.6 C3：Kinematic Edge

对应 Idea 11。

只在：

```text
C0/C1 already works
```

后做。

运动 affinity：

\[
S_{motion}(i,j)
=
\exp(
-\|
\Delta x_i-\Delta x_j
\|/\tau_m
)
\]

限制：

- 只用模型已有 motion / RigidNode state；
- 禁止使用 evaluation GT track 作为方法输入；
- 不跨明显 spatial/semantic barrier 直接连远距离点。

---

# 12.7 C Ablation

| Arm | Semantic Edge | Bayesian Gate | SAM Co-occurrence | Motion |
|---|---:|---:|---:|---:|
| C0 | DINO | no | no | no |
| C1 | DINO | yes | no | no |
| C2 | DINO | yes | yes | no |
| C3 | DINO | yes | yes | yes |

最终最多选择：

```text
一个 C-family candidate
```

---

# 13. Stage D：Progressive Propagation

Task：

```text
WS-V51-M1-D-PROGRESSIVE-01
```

如果：

```text
global diffusion
```

仍 scene-dependent，

先不换 node。

先换 propagation strategy。

参考：

```text
SAI3D
CVPR 2024
```

---

# 13.1 D0：SAI3D-Style Progressive Growing

Faithful mechanism port：

```text
high-confidence seed
→ strict merge
→ medium-confidence expansion
→ stop
```

不做一次性全图平滑。

Level 0：

\[
p_i\ge \tau_{high}
\]

只允许和：

```text
very high semantic affinity
+
geometry adjacency
```

邻居合并。

Level 1：

降低 affinity threshold。

Level 2：

如果 confidence 不够：

```text
UNKNOWN
```

而不是继续长。

---

# 13.2 D1：Sparsity-Adaptive Stop

对应 Idea 13。

只有 D0 work 后。

定义 scene/node observability：

\[
s_i
=
\frac{
n_i^{effective}
}{
N_{views}
}
\]

稀疏区：

```text
传播 hop 更少
```

高观测区：

```text
允许更多 progressive merge
```

原则：

> observation 越差，越保守。

---

# 13.3 D Gate

D0 如果不优于 C0/G0：

```text
progressive path rejected
```

禁止继续 D1。

---

# 14. Stage E：Graph Node Elevation

Task：

```text
WS-V51-M1-E-NODE-ELEVATION-01
```

只有：

```text
Bayesian unary work
+
raw-Gaussian propagation 不稳定
```

时启动。

这是第一次较大结构替换。

---

# 14.1 E0：Simple Voxel Super-Primitive

对应 Idea 5。

先做最简单的：

```text
multi-resolution voxel
```

同 voxel 内 Gaussian 聚成 node。

不做 learned anchor。

Node feature：

- mean Bayesian posterior
- effective count
- DINO feature
- position
- covariance summary
- motion summary
- visibility support

目的：

> 单 node 聚合更多 observation，降低 raw Gaussian 的 evidence sparsity。

---

# 14.2 E0 对照

必须比较：

```text
same graph edge / same propagation

raw Gaussian node
vs
voxel super-primitive
```

这样才能判断：

> 问题到底是不是 node granularity。

---

# 14.3 E1：PanoGS-Style Super-Primitive

参考：

```text
PanoGS
CVPR 2025
```

Faithful mechanism：

```text
geometry + visual feature
→ group Gaussians
→ super-primitives
→ SAM-guided affinity
→ graph clustering
```

V5.1 adaptation：

- base 3DGS immutable；
- super-primitives 保存为 sidecar；
- 不改 radiance / geometry checkpoint。

第一版禁止额外加入 Bayesian innovation。

先跑：

```text
PanoGS-like grouping alone
```

---

# 14.4 E2：AG²aussian-Style Anchor Graph Sidecar

参考：

```text
AG²aussian
ICCV 2025
```

由于 V5/V5.1 有 immutable-base 约束，

不完全重训其 anchor-based reconstruction。

做 closest faithful port：

```text
frozen Gaussian base
→ offline anchor assignment
→ anchor semantic features
→ anchor graph propagation
```

必须明确论文里写：

```text
inspired by / adapted from
```

不能声称完全复现 AG²aussian。

---

# 14.5 E3：Feature-Driven Component Anchor

对应 Idea 7。

只有 E0/E1/E2 至少一个有效以后才尝试。

构造：

```text
spatially connected
+
DINO/SigLIP similar
+
motion compatible
```

component。

目标：

> component 对应车身 / 车窗 / 轮胎 / road patch 等更稳定局部结构。

---

# 14.6 E Gate

Node elevation 必须证明：

```text
effective observations per node ↑
semantic purity ↑
cross-boundary leakage ↓
Boundary F1 ↑
```

而不是只：

```text
graph node 数量减少
```

---

# 15. Stage F：Gaussian Grouping Identity Embedding

Task：

```text
WS-V51-M1-F-IDENTITY-EMBEDDING-01
```

这是比 graph sidecar 更大的方法替换。

只有 A–E 没有达到稳定门，或想建立强 baseline 时做。

参考：

```text
Gaussian Grouping
ECCV 2024
```

---

# 15.1 F0：Faithful Identity Encoding Port

保持：

```text
Gaussian geometry frozen
Gaussian appearance frozen
```

每 Gaussian 只增加：

\[
e_i\in\mathbb R^d
\]

通过 differentiable rendering：

```text
SAM mask
→ 2D identity supervision
```

加入论文原始的：

```text
3D KNN spatial consistency regularization
```

第一版禁止：

- Bayesian initialization
- DINO graph
- anchor
- UNKNOWN innovation

先判断：

> Gaussian Grouping 的原机制，在 dynamic driving actor 上能否 work。

执行状态（2026-08-18）：F0 source/16D frozen-base adapter 与 official DEVA/SAM assets 已通过前置冻结；ResNet18/50
transitive assets 已固定（`V51-F53 resolved`）。allocator-only 与 batch=`64→32` 均无法让 default grid64 one-view SAM
在 24GB 上完成（`V51-F52/V51-F55/V51-F57`），最终累计规模而非 batch staging 是主因。当前只允许按 DEVA 官方资源建议
令 v6/r032 `SAM_NUM_POINTS_PER_SIDE=32`（1024 prompts），batch32、图像大小、阈值与其余方法语义保持；该适配不冒充
default-grid parity。resource/schema PASS 后仍须同-grid batch parity 与 3-view association+repeatability；在此之前
materialization、F0 training、F1/F2 均未授权。

---

# 15.2 F1：Bayesian-Calibrated Identity Seed

只有 F0 work 后。

创新：

```text
Bayesian Unary
→ initialize / weight identity supervision
```

高 confidence Gaussian：

```text
strong supervision
```

低 confidence：

```text
weak supervision
```

UNKNOWN：

```text
no forced class loss
```

---

# 15.3 F2：Observability-Weighted KNN Consistency

只有 F1 work 后。

KNN consistency 权重：

\[
\lambda_{ij}
=
S_{semantic}(i,j)
\cdot
q_i q_j
\]

避免：

```text
uncertain node
污染
confident node
```

---

# 16. Stage G：Trace3D-Style Ambiguity

Task：

```text
WS-V51-M1-G-AMBIGUITY-01
```

V5 旧的 3px boundary forensic 已经说明：

```text
2D boundary band
不是主要误差来源
```

因此不能直接复活旧 semantic split。

V5.1 只允许重新定义：

> **multi-view instance disagreement ambiguity**

参考：

```text
Trace3D
ICCV 2025
```

---

# 16.1 G0：Disagreement Matrix Diagnostic

每 Gaussian / Anchor：

\[
A_i
=
\{
m_{iv}
\}_v
\]

计算：

- cross-view vote entropy
- identity disagreement
- feature disagreement
- visibility-conditioned disagreement

先回答：

> 错误 Gaussian 是否显著富集在高 multi-view disagreement 区？

---

# 16.2 G Gate

只有：

```text
error enrichment >= 2x
```

且：

```text
>= 50% semantic error mass
```

落在高 disagreement node，

才解锁 split。

---

# 16.3 G1：Reversible Semantic Child Delta

如果解锁：

```text
base Gaussian immutable
```

sidecar：

```text
erase semantic identity of parent
+
child semantic nodes
```

不修改原 checkpoint。

必须 exact rollback。

---

# 17. Stage H：Graph-Free Fallback

Task：

```text
WS-V51-M1-H-GRAPHFREE-01
```

如果：

- Bayesian Unary 仍有效；
- Graph / Anchor / Identity propagation 都无法稳定泛化；

不允许继续无限调 Graph。

进入 Graph-Free 路线。

---

# 17.1 H0：BKI-Style Spatial Kernel Bayesian Update

参考：

```text
Bayesian Spatial Kernel Smoothing
```

核心原样迁移：

不是：

```text
Bayes posterior
→ graph smoothing
```

而是：

```text
spatial / semantic kernel
直接进入 Bayesian evidence update
```

例如：

\[
\alpha_i
=
\alpha_0
+
\sum_j
k(i,j)
r_jp_j
\]

\[
\beta_i
=
\beta_0
+
\sum_j
k(i,j)
r_j(1-p_j)
\]

第一版 kernel：

```text
spatial only
```

作为 faithful BKI mechanism。

---

# 17.2 H1：Semantic Kernel

只有 H0 work 后：

\[
k(i,j)
=
k_{spatial}
\cdot
k_{DINO}
\]

---

# 17.3 H2：LiDAR Barrier Kernel

对应 Idea 4。

只有 H1 work 后：

\[
k(i,j)
=
k_{spatial}
k_{semantic}
k_{lidar}
\]

LiDAR 不直接充当：

```text
actor probability
```

只影响：

```text
physical continuity kernel
```

例如：

- depth discontinuity
- surface separation
- LiDAR-supported foreground/background boundary

---

# 17.4 H3：No-Propagation Strong Unary

最终必须保留一个纯 unary fallback：

```text
best Bayesian unary
+
visibility decoupling
+
UNKNOWN
+
renderer-transpose uplift
```

完全不传播 ownership。

如果它在 C / validation 上最好：

> 接受“Graph 不需要”这一科研结论。

V5.1 不强迫 Graph 成为主方法。

---

# 18. Temporal Pre-Association：条件支线

对应 Idea 14。

Task：

```text
WS-V51-M1-TEMP-ID-ASSOCIATION-01
```

只在分析证明：

```text
单帧 sparse
+
跨时域可观测
```

明显存在时启动。

第一版只用：

```text
base model actor / RigidNode temporal identity
```

禁止使用：

```text
nuScenes GT instance token
```

作为方法输入。

GT track ID 只能 evaluator 使用。

聚合：

\[
\alpha_{i,t}
=
\rho\alpha_{i,t-1}
+
\alpha_{i,t}^{obs}
\]

但必须做到：

```text
visibility-aware
identity-safe
```

如果 actor association 不可靠：

```text
ABSTAIN
```

---

# 19. Semantic Sidecar

对应 Idea 16。

这不是主要 paper novelty，但必须实现。

输出：

```text
semantic_sidecar/
├── manifest.json
├── gaussian_or_anchor_index.bin
├── posterior_actor.bin
├── posterior_background.bin
├── unknown.bin
├── uncertainty.bin
├── effective_count.bin
├── semantic_feature.bin          # optional
├── topology.bin                  # optional
└── provenance.json
```

要求：

```text
base checkpoint SHA unchanged
sidecar content-addressed
exact reload
exact deterministic inference
```

---

# 20. 独立评价体系升级

V5.1 如果目标是顶会，不能只用和方法高度耦合的 SAM 指标。

必须保留三层评价。

---

## 20.1 Layer A：V5 Frozen Proxy

用于 exact continuity：

```text
V5 existing ownership proxy
V5 SAM-based 2D evaluator
```

作用：

```text
和历史 V5 可比
```

不能作为唯一 paper evidence。

---

## 20.2 Layer B：nuScenes Geometry-Aware Evaluation

利用 evaluation-only：

- nuScenes 3D actor boxes
- LiDAR returns
- camera calibration

构造：

### LiDAR actor support recall

actor 3D box 内 LiDAR point：

```text
nearest / rendered Gaussian ownership
```

应为 actor。

### Background leakage

明显在 actor box 外且有 LiDAR support 的 static region：

```text
不应被 ownership propagation 吞入 actor
```

方法运行时禁止使用该 evaluation GT。

---

## 20.3 Layer C：Independent Boundary Audit

优先检查本地是否已有：

```text
nuScenes panoptic / lidarseg
```

如果本地存在：

```text
evaluation-only
```

禁止拿来训练/调图。

如果不存在：

不主动下载。

最终 candidate 可建立：

```text
30–50 heldout frame
manual 2D actor boundary audit
```

要求：

- blind to method
- fixed before final test aggregation
- 不用于调参

用于验证：

```text
SAM teacher improvement
是否等价于真实 boundary improvement
```

---

# 21. 核心 Metrics

## 21.1 Ownership

- IoU
- Boundary F1
- Precision
- Recall
- FP semantic mass
- FN semantic mass

## 21.2 Calibration

- Brier
- ECE
- NLL
- reliability diagram

## 21.3 Observability

新增：

- valid SAM view ratio
- effective observation count
- UNKNOWN ratio
- posterior entropy
- observability-conditioned IoU
- observability-conditioned Boundary F1

分桶：

```text
sparse:
< 10%

medium:
10–30%

dense:
> 30%
```

必须单独报告。

这是 V5.1 的关键 paper figure。

---

## 21.4 Graph / Grouping

- graph leakage
- actor-background edge ratio
- component purity
- component fragmentation
- connected-component count
- cross-boundary propagation
- propagation distance
- semantic feature margin

---

## 21.5 Safety / Selective

- coverage
- error@coverage
- selective risk
- UNKNOWN error concentration
- hallucinated actor expansion rate

重点：

> 观测越稀疏时，方法是否自动更保守。

---

## 21.6 Engineering

- M1 wall time
- feature extraction wall
- graph build wall
- graph inference wall
- peak VRAM
- peak RAM
- sidecar bytes
- cold load
- semantic query latency

---

# 22. Main Ablation Tree

V5.1 不做一张 20 列的大表。

按 family 分表。

---

## Table A：Unary

```text
U0 B0
U1 B1
U2 B3
A1 Visibility-Masked
A2 + UNKNOWN
A3 + Effective Count
A4 CIF-style Decoupling
```

---

## Table B：2D→3D Uplift

```text
current intersection
vs
LUDVIG renderer-transpose
```

---

## Table C：Raw Gaussian Propagation

```text
G0 no graph
V5 G3
C0 LUDVIG faithful
C1 + Bayesian gate
C2 + SAM co-occurrence
C3 + motion
D0 progressive growing
```

---

## Table D：Node

```text
raw Gaussian
voxel super-primitive
PanoGS-like super-primitive
AG2 anchor-sidecar
feature component
```

---

## Table E：Representation

```text
scalar posterior
vs
Gaussian Grouping identity embedding
```

---

## Table F：Graph-Free

```text
best unary
BKI spatial
BKI semantic
BKI semantic + LiDAR barrier
```

---

# 23. Candidate Promotion Rules

每个 family：

```text
H diagnostic
→ S screening
→ C confirmation
```

不能跳级。

---

## 23.1 H Gate

单 candidate：

```text
>=2/3 scenes positive BF1
mean BF1 positive
IoU non-negative
FN delta <= +0.02
```

---

## 23.2 S Gate

```text
2/2 BF1 non-negative
>=1 clearly positive
calibration non-degraded
```

失败：

```text
candidate rejected
```

---

## 23.3 C Gate

最终最多 2 families。

3 scenes：

```text
>= 2/3 positive

scene-balanced mean
ΔBoundary-F1 >= +0.015

ΔIoU >= 0

ΔFN semantic mass <= +0.01

Brier or ECE improves

另一 calibration metric
no critical degradation
```

只有 C PASS：

```text
candidate can enter final method freeze
```

---

# 24. Final Development Selection

最终最多选：

```text
1 primary candidate
+
1 strong external baseline
```

Primary 必须是：

```text
Bayesian Unary
+
最多 1 个 structural propagation mechanism
+
最多 1 个 safety / observability mechanism
```

例如可能的成功形态：

### Candidate P1

```text
Visibility-Decoupled Bayesian Unary
+
LUDVIG Semantic Graph
+
UNKNOWN Gate
```

### Candidate P2

```text
Bayesian Unary
+
PanoGS-like Super-Primitive
+
SAM Co-occurrence Graph
```

### Candidate P3

```text
Bayesian Unary
+
BKI Semantic Kernel
+
UNKNOWN
```

禁止最终方法：

```text
A1+A2+A3+A4+B1+C1+C2+C3+D1+E3+...
```

---

# 25. Innovation Rule

顶会创新只能建立在已 work 的 external port 上。

---

## Case 1：LUDVIG faithful port work

允许创新：

> **Observability-Calibrated Semantic Diffusion**

核心：

\[
\text{Bayesian uncertainty}
\rightarrow
\text{adaptive graph diffusion strength}
\]

paper story：

> Foundation-model semantic affinity tells **where** propagation is semantically plausible; Bayesian observability tells **whether** propagation is justified.

---

## Case 2：PanoGS / Anchor work

允许创新：

> **Bayesian Super-Primitive Sidecar for Dynamic Driving Gaussians**

核心：

```text
raw Gaussian sparse observations
→ anchor aggregation
→ calibrated ownership
→ safe semantic propagation
```

---

## Case 3：BKI work / Graph fails

允许创新：

> **Observability-Aware Kernelized Bayesian Gaussian Ownership**

核心：

```text
spatial correlation
直接进入 posterior
而不是后验 graph smoothing
```

---

## Case 4：Gaussian Grouping works best

允许创新：

> **Calibrated Identity Embedding for Dynamic Driving Gaussians**

Bayesian unary：

```text
不是最终 scalar output
```

而是：

```text
confidence-weighted supervision / identity seed
```

---

# 26. Fresh Validation Gate

V5 8 validation scenes 仍未读取。

正式 validation 前：

创建：

```text
V51_M1_VALIDATION_FREEZE.json
```

包含：

- source commit
- base checkpoint SHA
- unary config SHA
- feature backbone
- feature checkpoint SHA
- graph/node strategy
- thresholds
- propagation steps
- UNKNOWN policy
- selected candidate
- baseline configs
- evaluator SHA
- scene list
- metrics schema

提交：

```text
research(m1): freeze v5.1 validation candidate
```

然后：

```text
8 validation scenes
只跑一次 candidate selection confirmation
```

---

# 27. Validation Success Gate

Primary candidate relative to V5 best Bayesian unary：

```text
directional support >= 6 / 8
```

scene-balanced：

```text
mean ΔBoundary-F1 >= +0.03
```

同时：

```text
mean ΔIoU > 0

FN semantic mass
<= +0.01

ECE / Brier
至少一个改善

另一个
no critical degradation
```

Observability：

```text
sparse bucket
不得出现明显 hallucination expansion
```

Selective：

```text
UNKNOWN / abstained subset error
>
accepted subset error
```

Engineering：

```text
单 RTX3090
无 OOM
```

失败：

```text
V5.1 M1 rejected
```

禁止：

- validation 调 threshold
- 换 feature backbone
- 改 node size
- 改 K
- 改 graph hops
- 只挑 positive scenes

---

# 28. Test Freeze

Validation PASS 后生成：

```text
V51_TEST_FREEZE.json
```

必须绑定：

- candidate
- 20 test scenes
- source SHA
- all configs
- all model weights
- all sidecars
- metric schema
- exact denominator
- test attempt ledger path

然后：

```text
freeze-only commit
```

---

# 29. 20-Scene Exact-Once Test

V5 fresh 20-scene test：

```text
只执行一次
```

规则继承 V4：

- attempt marker
- exact-once ledger
- SHA preflight
- resource preflight
- fail-closed
- blocked / abstain 留 denominator
- no rerun for quality

主报告：

- Boundary F1
- IoU
- Brier
- ECE
- NLL
- FP/FN semantic mass
- coverage
- observability buckets
- graph leakage
- wall / VRAM / bytes

---

# 30. KITTI Frozen Cross-Dataset

只有 nuScenes test 完成后。

现有 KITTI adapter smoke 已有基础。

V5.1：

```text
不调任何 M1 参数
```

尤其禁止：

- visibility threshold
- UNKNOWN threshold
- feature similarity threshold
- KNN K
- graph hops
- anchor voxel size
- kernel bandwidth

若数据域差异导致：

```text
method degradation
```

如实报告 cross-domain negative。

---

# 31. Paper 最低成功条件

V5.1 的目标不是“某一个 scene BF1 涨了”。

顶会 candidate 至少：

```text
8 validation
+
20 exact-once test
+
KITTI frozen cross-domain
```

方法：

```text
Bayesian Unary 保留明确收益

结构模块相对 Bayesian-only
在 scene-disjoint 数据上稳定增益
```

统计：

```text
scene-level
95% bootstrap CI
paired test
```

必须有：

```text
observability-stratified analysis
```

这是 V5.1 相对一般 3DGS segmentation paper 的重要 driving-specific evidence。

---

# 32. Paper 可能的核心 Story

最终根据结果选择，不提前写死。

---

## Story A：Graph 成功

标题方向：

> **ObsBayes-GS: Observability-Calibrated Semantic Propagation for Dynamic Driving Gaussians**

核心洞察：

> 2D foundation-model supervision 在 driving scenes 中不是简单 noisy，而是 **structurally missing**。现有 3D propagation 把 missingness 当成可被平滑的问题，会产生 scene-dependent hallucination。V5.1 通过 calibrated Bayesian observation 与 semantic-gated propagation 解耦“是否有证据”和“是否应该传播”。

---

## Story B：Anchor 成功

> **AnchorBayes-GS: Calibrated Super-Primitive Ownership for Dynamic Driving Gaussians**

洞察：

> Raw Gaussian 粒度下 evidence support 太稀疏；将语义推理提升到稳定 super-primitives 后，可显著提高 observation density 与 cross-view identity consistency。

---

## Story C：Graph-Free 成功

> **KernelBayes-GS: Observability-Aware Spatial Bayesian Ownership for Driving Gaussians**

洞察：

> 显式 iterative graph diffusion 并非必要；把局部相关性直接作为 Bayesian kernel likelihood，可在保留 uncertainty 的同时避免长距离 semantic hallucination。

---

# 33. Related Work / Paper Migration Matrix

| Work | Venue | V5.1 原样迁移内容 | 只有 work 后允许的创新 |
|---|---|---|---|
| Consistent Instance Field | CVPR 2026 | visibility / occupancy / identity decoupling | driving observability calibration |
| LUDVIG | ICCV 2025 | renderer-transpose uplift + DINO semantic graph | Bayesian observability gate |
| PanoGS | CVPR 2025 | super-primitives + SAM-guided affinity | calibrated dynamic-driving supergraph |
| Gaussian Grouping | ECCV 2024 | identity encoding + SAM supervision + KNN consistency | Bayesian-weighted identity learning |
| AG²aussian | ICCV 2025 | anchor graph concept | immutable semantic anchor sidecar |
| Trace3D | ICCV 2025 | multi-view disagreement diagnostic | reversible semantic child-delta |
| SAI3D | CVPR 2024 | progressive region growing | sparsity-adaptive stop |
| SAM-guided Graph Cut | ECCV 2024 | multi-view SAM edge affinity | Bayesian / motion gating |
| Bayesian Spatial Kernel Smoothing | RA-L | spatial kernel inside Bayesian inference | semantic/LiDAR kernel for Gaussian |
| AD-GS | ICCV 2025 | KNN local physical consistency reference | motion affinity only if graph base works |

---

# 34. Official Source List

Codex literature audit 优先使用官方 paper / project / repository。

### LUDVIG

https://openaccess.thecvf.com/content/ICCV2025/html/Marrie_LUDVIG_Learning-Free_Uplifting_of_2D_Visual_Features_to_Gaussian_Splatting_ICCV_2025_paper.html

### PanoGS

https://openaccess.thecvf.com/content/CVPR2025/html/Zhai_PanoGS_Gaussian-based_Panoptic_Segmentation_for_3D_Open_Vocabulary_Scene_Understanding_CVPR_2025_paper.html

### Consistent Instance Field

https://openaccess.thecvf.com/content/CVPR2026/html/Wu_Consistent_Instance_Field_for_Dynamic_Scene_Understanding_CVPR_2026_paper.html

### AG²aussian

https://openaccess.thecvf.com/content/ICCV2025/html/Wang_AG2aussian_Anchor-Graph_Structured_Gaussian_Splatting_for_Instance-Level_3D_Scene_Understanding_ICCV_2025_paper.html

### Trace3D

https://openaccess.thecvf.com/content/ICCV2025/html/Shen_Trace3D_Consistent_Segmentation_Lifting_via_Gaussian_Instance_Tracing_ICCV_2025_paper.html

### SAI3D

https://openaccess.thecvf.com/content/CVPR2024/html/Yin_SAI3D_Segment_Any_Instance_in_3D_Scenes_CVPR_2024_paper.html

### Gaussian Grouping

https://eccv.ecva.net/virtual/2024/poster/1309

### SAM-guided Graph Cut

https://eccv.ecva.net/virtual/2024/poster/2271

### Bayesian Spatial Kernel Smoothing

https://arxiv.org/abs/1909.04631

---

# 35. Engineering Layout

建议新增：

```text
configs/worldsim_v51/
├── p0_m1_scope_v1.yaml
├── development_roles_v1.yaml
├── m1_baselines_v1.yaml
├── m1_unary_visibility_v1.yaml
├── m1_unary_unknown_v1.yaml
├── m1_unary_effective_count_v1.yaml
├── m1_cif_decoupled_v1.yaml
├── m1_ludvig_uplift_v1.yaml
├── m1_ludvig_graph_v1.yaml
├── m1_progressive_v1.yaml
├── m1_superprimitive_v1.yaml
├── m1_anchor_graph_v1.yaml
├── m1_identity_embedding_v1.yaml
├── m1_trace_disagreement_v1.yaml
├── m1_bki_v1.yaml
├── m1_validation_freeze.yaml
└── metrics_v1.yaml
```

代码：

```text
motion_proj/worldsim_v51/
├── __init__.py
├── evidence/
│   ├── visibility.py
│   ├── bayesian_unary.py
│   ├── effective_count.py
│   ├── abstention.py
│   └── cif_state.py
├── features/
│   ├── renderer_transpose.py
│   ├── dinov2_uplift.py
│   └── feature_metrics.py
├── graph/
│   ├── knn.py
│   ├── ludvig_affinity.py
│   ├── sam_cooccurrence.py
│   ├── motion_affinity.py
│   ├── diffusion.py
│   └── progressive.py
├── nodes/
│   ├── voxel_anchor.py
│   ├── superprimitive.py
│   ├── anchor_graph.py
│   └── feature_component.py
├── identity/
│   ├── identity_encoding.py
│   ├── identity_renderer.py
│   └── spatial_consistency.py
├── kernel/
│   ├── bki.py
│   ├── semantic_kernel.py
│   └── lidar_barrier.py
├── sidecar/
│   ├── schema.py
│   ├── writer.py
│   └── loader.py
└── eval/
    ├── ownership_metrics.py
    ├── observability_metrics.py
    ├── graph_metrics.py
    ├── lidar_box_eval.py
    ├── selective_metrics.py
    └── statistics.py
```

scripts：

```text
scripts/
├── audit_worldsim_v51_start.py
├── replay_worldsim_v51_v5_unary.py
├── run_worldsim_v51_unary_ablation.py
├── build_worldsim_v51_dino_sidecar.py
├── run_worldsim_v51_ludvig_uplift.py
├── run_worldsim_v51_graph_ablation.py
├── run_worldsim_v51_progressive.py
├── build_worldsim_v51_superprimitives.py
├── build_worldsim_v51_anchor_graph.py
├── run_worldsim_v51_identity_embedding.py
├── run_worldsim_v51_disagreement_forensic.py
├── run_worldsim_v51_bki.py
├── aggregate_worldsim_v51_development.py
├── freeze_worldsim_v51_validation.py
├── run_worldsim_v51_validation.py
├── build_worldsim_v51_test_freeze.py
├── run_worldsim_v51_test_exact_once.py
└── aggregate_worldsim_v51_test.py
```

---

# 36. Tests

最低：

```text
tests/
├── test_worldsim_v51_visibility.py
├── test_worldsim_v51_missing_not_negative.py
├── test_worldsim_v51_unknown.py
├── test_worldsim_v51_effective_count.py
├── test_worldsim_v51_cif_state.py
├── test_worldsim_v51_renderer_transpose.py
├── test_worldsim_v51_feature_uplift.py
├── test_worldsim_v51_ludvig_affinity.py
├── test_worldsim_v51_graph_no_cross_unknown.py
├── test_worldsim_v51_sam_cooccurrence.py
├── test_worldsim_v51_progressive_stop.py
├── test_worldsim_v51_superprimitive.py
├── test_worldsim_v51_anchor_graph.py
├── test_worldsim_v51_identity_embedding.py
├── test_worldsim_v51_disagreement.py
├── test_worldsim_v51_bki.py
├── test_worldsim_v51_sidecar_exact.py
├── test_worldsim_v51_dev_roles.py
├── test_worldsim_v51_validation_freeze.py
└── test_worldsim_v51_test_exact_once.py
```

---

# 37. Content-Address / Provenance

所有正式 run 保存：

```text
source_commit
worktree_clean
config_sha
base_checkpoint_sha
SAM_sidecar_sha
feature_model_sha
feature_sidecar_sha
node_sidecar_sha
graph_sidecar_sha
method_output_sha
metrics_sha
```

任何 input drift：

```text
blocked
```

禁止自动重建旧 run。

---

# 38. Resource Discipline

RTX 3090 24GB。

Feature uplift / graph：

优先：

```text
CPU sparse graph
+
GPU feature extraction
```

禁止：

```text
一次把 N×N affinity 放 GPU
```

KNN / graph 必须 sparse。

如果 raw Gaussian graph：

```text
edge count
>
memory gate
```

直接使用 chunk / CSR。

不因为 OOM：

```text
减少 scene
减少 camera
降低正式 resolution
```

来偷过门。

---

# 39. Task Registry

| Task ID | Initial | 内容 |
|---|---:|---|
| `WS-V51-P0-M1-SCOPE-FREEZE-01` | pending | V5.1 M1-only scope |
| `WS-V51-D0-DEV-ROLE-FREEZE-01` | pending | H/S/C dev 分层 |
| `WS-V51-M1-A-UNARY-OBSERVABILITY-01` | pending | visibility / unknown / effective count / CIF |
| `WS-V51-M1-B-LUDVIG-UPLIFT-01` | pending | learning-free feature uplift |
| `WS-V51-M1-C-SEMANTIC-GRAPH-01` | conditional | LUDVIG semantic graph |
| `WS-V51-M1-D-PROGRESSIVE-01` | conditional | SAI3D progressive propagation |
| `WS-V51-M1-E-NODE-ELEVATION-01` | conditional | super-primitive / anchor |
| `WS-V51-M1-F-IDENTITY-EMBEDDING-01` | conditional | Gaussian Grouping port |
| `WS-V51-M1-G-AMBIGUITY-01` | conditional | Trace3D disagreement |
| `WS-V51-M1-H-GRAPHFREE-01` | conditional | BKI / graph-free |
| `WS-V51-M1-TEMP-ID-ASSOCIATION-01` | conditional | temporal pre-association |
| `WS-V51-M1-VALIDATION-01` | pending | 8-scene validation |
| `WS-V51-M1-TEST-01` | pending | 20-scene exact-once test |
| `WS-V51-M1-KITTI-01` | pending | frozen cross-dataset |
| `WS-V51-M1-PAPER-01` | pending | M1-only paper package |
| `WS-V51-M2` | pending | explicitly not authorized |
| `WS-V51-M3` | pending | explicitly not authorized |

---

# 40. 推荐执行优先级

严格：

```text
P0
↓
A unary
↓
B uplift
↓
C raw semantic graph
```

如果 C work：

```text
C innovation
→ C confirmation
```

如果 C 不 work：

```text
D progressive
```

如果 D 不 work：

```text
E node elevation
```

如果 E 不 work：

```text
F identity embedding
```

如果 ambiguity evidence strong：

```text
G Trace3D split
```

如果 Graph family整体不稳：

```text
H BKI / graph-free
```

禁止并行开发：

```text
C+D+E+F+G+H
```

避免算力和研究结论失控。

---

# 41. V5.1 早停原则

## Unary

如果 A1–A4 没有任何 arm 在 S 上超过 U2：

```text
冻结 U2
```

不再继续 Bayesian family。

---

## LUDVIG Uplift

如果 DINO uplift feature：

```text
same actor / background margin
没有稳定 separation
```

不跑 semantic graph。

---

## Raw Graph

C0 faithful port 失败：

```text
reject LUDVIG raw graph
```

不能靠 C1/C2/C3 救。

---

## Progressive

D0 原始 progressive 失败：

```text
不做 D1
```

---

## Node Elevation

E0 simple anchor 如果 observation density 不提升：

```text
停止 E1/E2
```

---

## Gaussian Grouping

F0 faithful port 失败：

```text
停止 F1/F2
```

---

## Trace3D

disagreement 不富集 semantic error：

```text
split locked
```

---

## BKI

H0 spatial kernel 本身失败：

```text
停止 H1/H2
```

---

# 42. Paper Novelty 防撞原则

V5.1 最终论文不能写：

```text
我们用了 DINO + KNN + SAM
```

也不能写：

```text
我们把 LUDVIG 搬到 driving
```

顶会 novelty 必须来自实验确认后的一个明确 insight。

推荐优先寻找：

### Insight 1

> **Observability is the missing variable in 2D-to-3D semantic lifting for driving Gaussians.**

### Insight 2

> **Semantic propagation must be confidence-bounded; graph smoothing should not hallucinate ownership where observations are structurally missing.**

### Insight 3

> **Raw Gaussian granularity causes evidence fragmentation; semantic ownership is more stable on calibrated super-primitives.**

### Insight 4

> **Spatial correlation is better introduced inside Bayesian evidence accumulation than through unconstrained posterior diffusion.**

只能选择实验真正支持的那个。

---

# 43. Paper 主图建议

## Figure 1

V5 failure mechanism：

```text
Dense scene:
SAM evidence → unary → graph works

Sparse scene:
SAM missing → weak unary → graph hallucination
```

---

## Figure 2

最终方法架构。

---

## Figure 3

Observability bucket：

```text
SAM valid-view ratio
vs
Boundary F1
```

比较：

```text
B3
V5 G3
V5.1
```

---

## Figure 4

Risk / coverage：

```text
semantic coverage
vs
ownership error
```

---

## Figure 5

Graph leakage visualization。

---

## Figure 6

Raw Gaussian vs Anchor / Kernel / final representation。

---

## Figure 7

nuScenes 20-test scene qualitative。

---

## Figure 8

KITTI zero-tuning qualitative。

---

# 44. Paper 主表最低要求

### Table 1：nuScenes Ownership

- IoU
- Boundary F1
- FP
- FN
- Brier
- ECE

### Table 2：Observability Robustness

- sparse
- medium
- dense

### Table 3：Propagation Safety

- leakage
- coverage
- hallucination
- UNKNOWN selective error

### Table 4：Ablation

Bayes / feature / node / propagation。

### Table 5：KITTI

frozen cross-dataset。

### Table 6：Engineering

- wall
- VRAM
- sidecar MB
- load
- query latency

---

# 45. V5.1 第一轮 Codex Agent 指令

以下可以直接交给 Coding Agent：

```text
你现在接手 Motion-Proj / WorldSim V5.1。

项目目录：
/root/autodl-tmp/motion_proj

目标：
只改进 M1 Gaussian ownership。
目标不是“多实现几个模块”，而是找到一个 scene-disjoint 稳定、可严谨消融、达到顶会 paper 水平的 M1 方法。

M2/M3 明确 pending。
未经用户后续授权禁止执行 M2/M3。

====================
一、先核对现场
====================

不要直接修改代码。

先核对：

git branch
git HEAD
git status

当前 V5 canonical branch：
research/worldsim-v5-structdelta

已知历史 HEAD：
f7566beb4d37115700a1d702f524d99cbab24b4e

如果远端 canonical 已更新，
以最新 canonical Git / summary / manifest 为准，
不得退回旧步骤。

读取：

AGENTS.md
README.md
docs/RESEARCH_STATUS.md
docs/RESEARCH_FAILURES.md
docs/EXPERIMENTS.md
V5 plan
V5 M1 unary docs
V5 graph replication docs
V5 boundary forensic docs

核对 canonical：
r037
r038
r042/r043
r045/r046
以及相关 summary/status/manifest/SHA。

====================
二、创建 V5.1
====================

基于最终 V5 clean branch 创建：

research/worldsim-v5.1-m1

新增：

docs/WORLDSIM_V5_1_M1_PLAN.md
configs/worldsim_v51/
motion_proj/worldsim_v51/

V5 canonical artifacts 全部只读。

====================
三、先冻结 development roles
====================

不要重新挑 scene。

Historical diagnostic：
0471
1087
0379

Screen：
0998
0359

Development confirmation：
0875
0535
0436

V5 8 validation
保持完全 unread。

V5 20 test
保持完全 unread。

输出：

configs/worldsim_v51/development_roles_v1.yaml

并绑定原 V5 cohort SHA。

====================
四、第一轮只执行 Stage A
====================

当前只授权：

WS-V51-P0-M1-SCOPE-FREEZE-01
WS-V51-D0-DEV-ROLE-FREEZE-01
WS-V51-M1-A-UNARY-OBSERVABILITY-01

不要提前实现：
LUDVIG graph
PanoGS
AG2aussian
Gaussian Grouping
Trace3D
BKI
M2
M3

A0：
exact replay V5 B0/B1/B3。

必须证明 r037/r042/r043 相关 baseline 可重放。

A1：
Visibility-Masked Bayesian Update。

原则：
不可见不是 negative evidence。

A2：
UNKNOWN / ABSTAIN state。

A3：
correlation-aware effective observation count。

A4：
CIF-style visibility / occupancy / conditional identity decoupling。

每次只改一个机制。

====================
五、Stage A 实验
====================

先 H 三场：

0471
1087
0379

每 arm 输出：

IoU
Boundary F1
FP semantic mass
FN semantic mass
Brier
ECE
NLL

valid observation ratio
effective observation count
UNKNOWN ratio
coverage
selective error

所有 blocked/abstain 保留 denominator。

Stage A H gate：

>=2/3 scene BF1 positive vs B3
mean BF1 positive
IoU non-negative
FN delta <= +0.02
Brier/ECE 至少一项改善

最多 2 candidate 进入 S。

然后：

0998
0359

S 上不调参数。

若 S 失败：
该 arm rejected。

====================
六、第一轮禁止
====================

禁止：

validation quality
test quality
KITTI method tuning
重新搜索 SAM threshold
更换 base reconstruction
Graph lambda 搜索
增加 DINO/SigLIP
super-primitive
anchor
identity embedding
semantic split

Stage A 收口前不碰这些。

====================
七、第一轮输出
====================

docs/WS_V51_M1_UNARY_OBSERVABILITY.md

configs/worldsim_v51/
p0_m1_scope_v1.yaml
development_roles_v1.yaml
m1_unary_baselines_v1.yaml
m1_unary_visibility_v1.yaml
m1_unary_unknown_v1.yaml
m1_unary_effective_count_v1.yaml
m1_cif_decoupled_v1.yaml

machine readable：

artifacts/worldsim_v51/m1_stage_a_results.json

同步：

docs/RESEARCH_STATUS.md
docs/EXPERIMENTS.md
docs/RESEARCH_FAILURES.md

====================
八、Stage A 后如何解锁
====================

如果 Stage A 有 candidate 通过 S：

下一步只允许：

WS-V51-M1-B-LUDVIG-UPLIFT-01

先原样迁移 LUDVIG 的 learning-free renderer-transpose feature uplift。

仍不允许 Graph。

只有 uplift feature 本身证明：
same-actor similarity
稳定高于 actor-background similarity

才解锁：

WS-V51-M1-C-SEMANTIC-GRAPH-01

C0 必须先做 LUDVIG faithful graph port。

如果 C0 不 work：
直接 reject，
不得先加 Bayesian gate/SAM edge/motion edge救它。

如果 C0 work：
才允许 C1/C2/C3 innovation。

严格执行 docs/WORLDSIM_V5_1_M1_PLAN.md。
```

---

# 46. V5.1 最后一条原则

> **先证明“别人的机制在我们的 driving Gaussian 问题上确实有效”，再谈创新。**

执行优先级永远：

```text
已验证 work 的 Bayesian Unary
>
修正 missing observation
>
原论文 faithful port
>
matched ablation
>
小创新
>
node replacement
>
representation replacement
>
Graph-free alternative
```

而不是：

```text
一次堆 16 个 idea
→ 指标涨
→ 不知道为什么
```

最终 V5.1 是否能发顶会，不取决于模块数量，而取决于能否给出一个被 8-val + 20-test + frozen KITTI 支持的明确研究结论：

> **为什么 driving Gaussian ownership 在 sparse multi-view supervision 下会失败，以及我们的方法为什么能稳定解决这个失败机制。**
