# WorldSim V5：Structured & Physics-Constrained Gaussian Editing Plan

## 面向可编辑驾驶 World Simulation 的结构化归属、几何可行修复与物理约束时序 Delta

- **项目根目录**：`/root/autodl-tmp/motion_proj`
- **V4 主计划**：`docs/WORLDSIM_V4_EVIDELTA_GS_PLAN.md`
- **V5 建议分支**：`research/worldsim-v5-structdelta`
- **V5 建议 run namespace**：`/root/autodl-tmp/runs/worldsim_v5/`
- **默认硬件**：单卡 NVIDIA RTX 3090 24GB
- **主数据集**：nuScenes
- **跨数据集确认**：KITTI，参数完全冻结后执行
- **V5 working title**：

> **StructDelta-GS: Structured, Geometry-Safe and Physics-Constrained Gaussian Editing for Driving World Simulation**

---

## 当前执行快照（2026-08-14）

- 当前事实源：[`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)，本节只提供计划导航。
- `WS-V5-P0-SCOPE-FREEZE-01=done`；formal closeout=`20260814T091100Z__p0-scope-closeout-s0-r001`，summary SHA=`ca4248cff7085d8d5a57c842827b1a549b6d1d82fa95c2f81a2976c1192f5d38`。
- `WS-V5-D0-NUSCENES-FRESH-COHORT-01=done`；8/8/20 与 V4 30-scene exclusion 已按 metadata-only 协议冻结，cohort SHA=`553373159023218b44615be27aeeb5533a6c585be276e06425235fe09b6b48b1`，fresh quality 未读。
- `WS-V5-M1-D0-BAYES-FORENSICS-01=done`；formal evidence 与缺失字段采集合约见 [`WS_V5_M1_FAILURE_FORENSICS.md`](WS_V5_M1_FAILURE_FORENSICS.md)。
- `WS-V5-M1-STRUCTURED-OWNERSHIP-01=running`；当前执行 evidence schema / effective-count unary instrumentation 与 result-blind raw preparation。DriveStudio 精确输入为六相机+`LIDAR_TOP` 完整时间链，metadata plan=`14,220 files / 0 present`；8 个 development scene 的 selective extraction 与 preprocess 尚未完成，graph 仍 disabled。
- `WS-V5-M2-D0-GEOMETRY-FORENSICS-01=done`；154-request 机器重算与 staged-geometry 合约见 [`WS_V5_M2_GEOMETRY_FORENSICS.md`](WS_V5_M2_GEOMETRY_FORENSICS.md)。
- `WS-V5-D1-KITTI-ARCHIVE-AUDIT-01=done`，但 `WS-V5-D1-KITTI-ADAPTER-01=blocked`；见 [`KITTI_TRACKING_ARCHIVE_AUDIT_V5.md`](KITTI_TRACKING_ARCHIVE_AUDIT_V5.md)。
- 当前只开放冻结 development scenes 上的 M1 structured ownership 与 M2 geometry-first/evidence-schema 工作；validation parameter search、完整 M3、新模型大训练、fresh test quality 与 KITTI 方法调参继续禁止。

---

# 0. V5 一页执行结论

V5 不推翻 V4，而是针对 V4 已经暴露出的三个机制问题继续推进。

V4 已得到的正式结论：

```text
M1:
Bayesian / calibrated Gaussian evidence field
→ scene-disjoint validation rejected

M2:
Risk-aware repair router
→ selective risk / abstention 有效
→ RGB / hole appearance 有收益
→ 但 hole geometry MAE 存在 +3.3908096237 m 严重退化

M3:
SE(3) cubic B-spline temporal delta
→ validation PASS
→ 18-scene frozen test PASS
→ warp L1 / temporal LPIPS 获得稳定提升
```

因此 V5 不重新做 V4，而是改成：

```text
Immutable Gaussian Base
        ↓
M1. Structured Gaussian Ownership Field
    Reliability-aware Bayesian Semantic Evidence
        +
    Physical Gaussian Topology Graph
        +
    Boundary / Geometry Consistency
        ↓
M2. Geometry-Feasible Repair Routing
    Failure Forensics
        →
    Geometry-First Repair
        →
    Feasibility-First Selective Routing
        ↓
Reversible Gaussian Delta
        ↓
M3. Constraint-Projected Temporal Delta
    Desired Counterfactual Motion
        →
    Kinematic / Contact / Smoothness Projection
        →
    Continuous-Time SE(3) Delta
```

V5 核心思想：

> **不再让一个 scalar probability / scalar risk score 独自承担整个 3D 物理世界。将 observation、topology、geometry feasibility、uncertainty 和 temporal constraints 分层建模。**

---

# 1. V5 硬约束

1. 默认单卡 RTX 3090 24GB。
2. 单卡完整闭环前禁止通过多卡改变算法、分辨率、模型容量。
3. 多卡如后续启用，只允许 scene-level parallel。
4. nuScenes 是主数据集。
5. KITTI 不阻塞 nuScenes 主线。
6. KITTI 不允许针对 V5 重新调参。
7. V4 已经读取过的 18-scene test 永久失去 V5 confirmatory-test 身份。
8. V4 test scene 只允许用于：
   - failure analysis
   - regression
   - mechanism diagnosis
   - visualization
9. V5 必须重新冻结 fresh nuScenes development / validation / test。
10. validation / test 禁止重新搜索参数。
11. failed / blocked / abstain 必须保留 denominator 与 provenance。
12. checkpoint / config / manifest / run / asset 必须内容寻址并记录 SHA。
13. 正式图像指标至少保留：
   - PSNR
   - SSIM
   - LPIPS
14. V5 继续保留：
   - temporal metrics
   - geometry metrics
   - engineering metrics
   - downstream metrics
   - scene-level statistics
15. V5 不允许用“数学包装”替代真实机制：
   - 所有公式必须对应代码
   - 所有公式必须有可消融实现
   - 所有物理约束必须有对应 metric
16. 不为追求复杂度主动接大 diffusion / foundation model。
17. M1 / M2 / M3 每一步必须先做机制诊断，再决定是否增加复杂度。

---

# 2. V4 作为 V5 的固定起点

## 2.1 M1：保留问题，不保留具体 Beta-only 建模

V4 M1 的失败不解释为：

```text
SAM supervision 无效
```

而解释为：

```text
“独立 Gaussian + 单一 Beta posterior +
SAM / Depth / LiDAR 全部压进同一概率更新”
这一建模假设泛化不足
```

V5 继续保留：

- SAM mask / pseudo-label supervision
- multi-view observation
- uncertainty
- calibration
- depth
- LiDAR

但改变它们在模型中的角色。

---

## 2.2 M2：保留 selective routing，但必须解决 geometry failure

V4 已证明：

- abstain group error 高于 accepted group
- selective routing 有意义
- hole appearance 有明显收益
- global RGB 基本保持

但 V5 必须正面解决：

```text
hole geometry MAE degradation
= +3.3908096237 m
```

V5 不允许继续用：

```text
“appearance 提升即可通过”
```

Geometry 变为硬成功门。

---

## 2.3 M3：保留连续时间 Delta，但降级 B-spline 的 novelty 地位

V4 已证明：

```text
SE(3) B-spline temporal representation
确实带来跨 scene 的 temporal improvement
```

因此 B-spline 不删除。

但 V5 不再声称：

```text
B-spline 本身是主要创新
```

V5 M3 的核心改为：

> **如何把任意 counterfactual edit trajectory 投影成物理可行、时序连续、可精确 rollback 的 Gaussian Delta。**

---

# 3. V5 三个核心科研假设

## H1：Gaussian ownership 不能被建模为彼此独立的 Bernoulli variable

V4 近似：

\[
p_i
=
P(z_i = \mathrm{actor}
\mid
\mathrm{SAM},
\mathrm{Depth},
\mathrm{LiDAR},
\ldots)
\]

但真实 Gaussian ownership 有明显空间结构：

\[
P(z_i \mid z_j, \mathcal G)
\neq
P(z_i)
\]

因此：

```text
Bayes
负责 observation reliability

Graph / topology
负责 3D structural prior
```

V5 M1 的核心是：

> **Reliability-aware Bayesian unary + Gaussian physical topology graph。**

---

## H2：Repair routing 不应让不同物理风险无条件相互补偿

V4：

\[
R(a)
=
\lambda_p R_p
+
\lambda_g R_g
+
\lambda_t R_t
+
\lambda_u U
+
\lambda_c C
\]

V5 假设：

> Geometry / temporal feasibility 应首先作为 constraint，而不是仅作为 weighted penalty。

即：

```text
physical feasibility
        ↓
candidate safe set
        ↓
risk / quality selection
        ↓
ABSTAIN when infeasible
```

---

## H3：Temporal editing 的创新应位于 constraint projection，而非 spline 类型

V5 不再研究：

```text
linear vs cubic vs another spline
```

作为主创新。

而研究：

\[
\hat T(t)
\rightarrow
\Pi_{\mathcal C}
\rightarrow
T^*(t)
\]

其中：

- \(\hat T(t)\)：用户期望的 counterfactual trajectory
- \(\mathcal C\)：物理 / 几何 / temporal constraints
- \(T^*(t)\)：满足约束的连续时间可执行编辑轨迹

---

# 4. M1：Structured Gaussian Ownership Field

Task：

```text
WS-V5-M1-STRUCTURED-OWNERSHIP-01
```

---

# 4.1 M1 总体结构

V4：

```text
SAM
Depth
LiDAR
visibility
confidence
        ↓
all compressed into scalar evidence weight
        ↓
Beta posterior
```

V5：

```text
SAM
 ↓
semantic observation likelihood

Depth
 ↓
occlusion / depth boundary / cross-view geometry

LiDAR
 ↓
3D measurement support / surface anchor

Gaussian position + covariance
 ↓
physical topology

Motion
 ↓
rigid / actor consistency

Multi-view visibility
 ↓
observation reliability
```

最后：

```text
Bayesian unary evidence
        +
Gaussian physical topology
        +
boundary barriers
        ↓
structured ownership posterior
```

---

# 4.2 Reliability-Aware Bayesian Semantic Unary

Gaussian \(i\)：

\[
z_i \in \{0,1\}
\]

表示 actor ownership。

视角 \(v\) 的 SAM observation：

\[
m_{iv}
\]

V5 不再假设所有 observation 同质且独立。

定义 observation reliability：

\[
r_{iv}
=
f(
c^{sam}_{iv},
v^{vis}_{iv},
d^{boundary}_{iv},
c^{depth}_{iv},
\theta_{view}
)
\]

其中：

- \(c^{sam}\)：SAM confidence
- \(v^{vis}\)：render visibility
- \(d^{boundary}\)：到 mask boundary 距离
- \(c^{depth}\)：depth consistency
- \(\theta_{view}\)：view angle / projection confidence

Bayesian unary：

\[
\tilde p_i
=
P(
z_i=1
\mid
\{m_{iv},r_{iv}\}_{v}
)
\]

同时输出：

\[
u_i^{semantic}
\]

---

# 4.3 Bayesian 方案候选

Development only 比较：

```text
B0
V4 Beta-Bernoulli

B1
Reliability-weighted Beta

B2
Hierarchical Beta

B3
Effective-count Bayesian fusion
```

禁止一次接过多复杂 Bayesian family。

选择标准：

- calibration
- Boundary F1
- FN/FP semantic mass
- scene-disjoint directional support
- implementation stability

---

# 4.4 Gaussian Physical Topology Graph

构建：

\[
\mathcal G = (V,E)
\]

其中：

- node = Gaussian
- edge = potential physical / semantic neighborhood relation

KNN 仅用于建立 sparse candidate graph：

```text
position KNN
→ candidate edges
```

最终 edge weight 不等于简单 Euclidean KNN。

---

# 4.5 Gaussian-Native Spatial Affinity

使用 Gaussian covariance：

\[
D_M(i,j)
=
(\mu_i-\mu_j)^T
(\Sigma_i+\Sigma_j)^{-1}
(\mu_i-\mu_j)
\]

候选进一步研究：

- Mahalanobis distance
- Bhattacharyya distance between Gaussians
- covariance principal-axis similarity
- surface normal similarity

核心原则：

> 3DGS topology 应尽量使用 Gaussian 自己的 center / covariance / orientation，而不是只看 XYZ 欧氏距离。

---

# 4.6 Surface / Motion Affinity

定义：

\[
w_{ij}
=
\exp(
-\lambda_s D_{spatial}
-\lambda_n D_{normal}
-\lambda_m D_{motion}
)
\cdot B_{ij}
\]

其中：

### Spatial

\[
D_{spatial}
\]

来自 Gaussian-native geometry。

### Surface normal

\[
D_{normal}
=
1-
|\mathbf n_i^T\mathbf n_j|
\]

### Motion consistency

动态对象：

\[
D_{motion}
=
\|\mathbf v_i-\mathbf v_j\|
\]

或 canonical motion consistency。

---

# 4.7 Depth Boundary Barrier

禁止 topology propagation 跨越真实 geometry discontinuity。

定义：

\[
B_{ij}
\in [0,1]
\]

如果 Gaussian \(i,j\) 在多视角中长期落在明显 depth boundary 两侧：

\[
B_{ij}
\rightarrow 0
\]

避免：

```text
vehicle semantic
→ diffusion
→ road surface
```

---

# 4.8 LiDAR 的重新定位

V5 中 LiDAR 不直接扮演：

```text
ownership probability
```

LiDAR 主要表示：

```text
real 3D surface support
```

用于：

- geometry anchor
- foreground/background separation
- surface normal estimation
- topology affinity
- depth discontinuity
- repair geometry support

---

# 4.9 Graph Posterior / Anisotropic Graph Diffusion

Bayesian unary：

\[
\tilde p_i
\]

Graph inference：

\[
p^*
=
\arg\min_p
\sum_i
c_i(p_i-\tilde p_i)^2
+
\lambda_G
\sum_{(i,j)\in E}
w_{ij}(p_i-p_j)^2
\]

矩阵形式：

\[
(C+\lambda_G L)p
=
C\tilde p
\]

其中：

- \(C\)：semantic confidence matrix
- \(L\)：weighted Gaussian graph Laplacian

解释：

```text
Bayesian unary：
每个 Gaussian 自己的多视角语义证据

Topology graph：
周围 3D 物理结构的约束
```

---

# 4.10 Structured Evidence State

V5 不再只有：

```text
posterior
uncertainty
```

正式保存：

```text
semantic_probability
semantic_uncertainty

topology_agreement
topology_disagreement

boundary_ambiguity

depth_support
lidar_support

visibility_support
multi_view_disagreement

motion_consistency
```

定义：

\[
E_i
=
(
p_i,
u_i^{sem},
u_i^{topo},
u_i^{geom},
u_i^{view},
q_i^{lidar},
q_i^{motion}
)
\]

M1 不提前压成 scalar。

---

# 4.11 Boundary Gaussian Conditional Branch

检测：

```text
high semantic entropy
+
high topology disagreement
+
rendered boundary proximity
```

标记：

```text
AMBIGUOUS_BOUNDARY_GAUSSIAN
```

如果 M1 graph 已明显解决 Boundary F1：

```text
stop
```

否则条件启动：

```text
WS-V5-M1B-REVERSIBLE-SEMANTIC-SPLIT-01
```

---

# 4.12 Reversible Semantic Split Delta（条件任务）

保持 base immutable：

```text
base Gaussian
不修改
```

对于 ambiguous Gaussian：

```text
erase selector
+
semantic child Gaussians
```

形成：

\[
\Delta_i^{split}
=
(
erase(g_i),
g_i^{fg},
g_i^{bg}
)
\]

目标：

- split foreground/background ownership
- preserve rollback
- improve boundary
- avoid destructive base modification

此任务仅在 M1 graph 无法达到成功门时解锁。

---

# 4.13 M1 Ablation

正式主消融：

| Arm | 方法 |
|---|---|
| M1-A0 | V3.3 O1 |
| M1-A1 | V4 Beta |
| M1-A2 | Reliability-aware Bayesian unary |
| M1-A3 | A2 + Euclidean KNN graph |
| M1-A4 | A2 + Gaussian physical graph |
| M1-FULL | A4 + depth/boundary barrier |
| M1-A5 | FULL + reversible semantic split，仅条件启动 |

---

# 4.14 M1 Metrics

必须：

### Semantic

- IoU
- Boundary F1
- FP semantic mass
- FN semantic mass

### Calibration

- ECE
- Brier Score
- NLL

### Topology

- component purity
- graph leakage
- neighbor consistency
- cross-boundary leakage
- topology disagreement

### Boundary

- ambiguous Gaussian ratio
- boundary false propagation
- boundary fragmentation

### Preservation

- base RGB SHA exact
- PSNR / SSIM / LPIPS

---

# 4.15 M1 Validation Gate

Fresh validation 8 scenes：

```text
directional support >= 6 / 8

mean ΔBoundary-F1 >= +0.04

FN semantic mass <= +0.01
relative to matched frozen baseline

ECE / Brier:
至少一项改善
另一项不得关键退化

graph cross-boundary leakage
不得显著升高

base RGB
exact
```

若失败：

1. 先判断 Bayesian unary 失败还是 topology 失败；
2. 不允许直接堆 Transformer；
3. 只有 boundary ambiguity 明确为主要残差时，才解锁 semantic split。

---

# 5. M2：Geometry-Feasible Repair Routing

Task：

```text
WS-V5-M2-GEOMETRY-FEASIBLE-ROUTER-01
```

M2 不直接修改 V4 router。

第一阶段必须先做：

```text
WS-V5-M2-D0-GEOMETRY-FORENSICS-01
```

---

# 5.1 M2-D0：+3.3908 m Geometry Forensics

对所有 V4 可用 request 做 retrospective mechanism diagnosis。

这些数据只用于：

- failure analysis
- debug
- mechanism identification

不能作为 V5 confirmatory test。

---

# 5.2 Geometry Failure Pipeline

对每个 request 逐层检查：

```text
A.
background / hole geometry reference
是否可靠？

B.
candidate raw geometry
是否已经错误？

C.
raw geometry
→ Gaussian asset
是否引入误差？

D.
Gaussian rendering / alpha compositing
是否进一步恶化 geometry？

E.
候选里是否存在好 geometry candidate，
但 router 选择了坏 candidate？
```

---

# 5.3 Geometry Oracle

定义：

\[
a_{oracle}^{geom}
=
\arg\min_a
E_{geom}(a)
\]

比较：

\[
a_{router}
\quad
vs
\quad
a_{oracle}^{geom}
\]

形成四类：

### Case 1：Routing Failure

```text
存在 geometry-good candidate
+
router 选择 geometry-bad candidate
```

### Case 2：Candidate Failure

```text
所有 candidate geometry 都差
```

### Case 3：Gaussianization Failure

```text
raw 3D geometry 正常
+
Gaussian representation / rendering 后变差
```

### Case 4：Metric / Reference Failure

```text
geometry reference 不可靠
```

任何 M2 改法必须建立在这个分类结果上。

---

# 5.4 V4 Risk Saturation Audit

优先检查当前类似：

\[
R_g
=
clip(
E_g / s_g,
0,
1
)
\]

的风险映射。

如果：

```text
0.6 m
2.0 m
5.5 m
```

大量都变成：

```text
geometry_risk = 1
```

则 V4 geometry risk 已失去排序能力。

必须绘制：

```text
raw geometry MAE
vs
normalized geometry risk
vs
router choice
```

并报告：

- saturation ratio
- unique risk values
- rank correlation
- bad-tail distinguishability

---

# 5.5 Geometry Risk Mapping Ablation

Development only：

### R0

V4 clipping：

\[
r(x)=\min(x/s,1)
\]

### R1

log：

\[
r(x)=\log(1+x/s)
\]

### R2

Huber / pseudo-Huber

保留大误差排序能力。

### R3

robust quantile mapping

仅根据 development distribution 拟合。

### R4

rank-preserving calibrated risk

核心标准：

> 风险表示必须保留 geometry tail 的排序能力。

---

# 5.6 Geometry-First Repair

如果 forensic 显示 candidate builder 本身是主要问题：

V5 新增：

```text
Geometry-First Candidate
```

流程：

```text
edited hole
        ↓
surrounding static Gaussian
+
LiDAR
+
multi-view depth
        ↓
local surface estimation
        ↓
3D geometry scaffold
        ↓
Gaussian geometry
        ↓
appearance assignment
        ↓
cross-view / TELEA / generated texture
```

原则：

> Geometry 决定 Gaussian 在哪里；appearance 只决定 Gaussian 长什么样。

---

# 5.7 Local Surface Models

按最小复杂度逐级尝试：

### G0

Robust road plane

### G1

Piecewise plane

### G2

Moving Least Squares

### G3

Local quadratic surface

不允许一开始上神经 surface model。

选择依据：

- hole geometry MAE
- cross-view consistency
- static preservation
- compute cost

---

# 5.8 Geometry Support Confidence

每个 candidate 输出：

```text
geometry_support
lidar_support
multi_view_depth_support
surface_fit_residual
extrapolation_distance
occlusion_uncertainty
```

M2 不再只得到：

```text
geometry_risk scalar
```

---

# 5.9 Feasibility-First Router

V5 主方法不再优先使用无条件 weighted sum。

先定义 safe candidate set：

\[
\mathcal A_{safe}
=
\{
a:
E_g(a)\le\tau_g,
E_t(a)\le\tau_t,
U_g(a)\le\tau_u
\}
\]

然后：

\[
a^*
=
\arg\min_{a\in\mathcal A_{safe}}
[
R_{photo}(a)
+
\lambda_uU(a)
+
\lambda_cC(a)
]
\]

如果：

\[
\mathcal A_{safe}
=
\varnothing
\]

则：

\[
a^*
=
ABSTAIN
\]

---

# 5.10 Candidate Interface

建议：

```python
RepairCandidate(
    method,
    gaussians,

    photo_error,
    photo_uncertainty,

    geometry_error,
    geometry_uncertainty,
    geometry_support,

    temporal_error,
    temporal_uncertainty,

    compute_cost,
    provenance,
)
```

禁止重新把所有字段压成不可解释单一 score。

---

# 5.11 M2 Ablation

| Arm | 方法 |
|---|---|
| M2-A0 | TELEA |
| M2-A1 | V4 Router |
| M2-A2 | V4 Router + non-saturating geometry risk |
| M2-A3 | Feasibility-first Router |
| M2-A4 | Geometry-first candidate only |
| M2-FULL | Geometry-first candidate + Feasibility-first Router |

RoadPatch：

- 保持原安全 gate；
- 不为提高 coverage 放宽冻结安全阈值。

---

# 5.12 M2 Metrics

### RGB

Global：

- PSNR
- SSIM
- LPIPS

Hole：

- PSNR
- SSIM
- LPIPS

### Geometry

- hole depth MAE
- median depth error
- relative depth error
- P90 / P95 depth error
- Chamfer
- static LiDAR MAE

### Selective

- coverage
- accepted error
- abstain error
- selective risk curve
- risk-coverage curve
- valid edit yield

### Router

- oracle agreement
- geometry oracle regret
- photo oracle regret
- candidate availability
- risk saturation ratio

---

# 5.13 M2 Validation Gate

Fresh validation：

```text
Global RGB:
PSNR / SSIM / LPIPS
matched non-inferior

Hole appearance:
relative to best matched baseline
至少一个主要视觉端点改善

Geometry:
mean hole geometry MAE
不得退化

P95 geometry error
不得关键退化

Selective:
abstain group error
>
accepted group error

Coverage:
目标 >= 60%
```

强目标：

```text
hole geometry MAE
relative improvement >= 10%
```

如果：

```text
RGB improved
+
geometry degraded
```

则：

```text
M2 rejected
```

V5 不再允许通过。

---

# 6. M3：Constraint-Projected Temporal Delta

Task：

```text
WS-V5-M3-CONSTRAINT-PROJECTED-TEMPORAL-01
```

---

# 6.1 M3 定位变化

V4：

```text
continuous SE(3) B-spline
= temporal contribution
```

V5：

```text
B-spline
= trajectory parameterization

constraint projection
= temporal contribution
```

---

# 6.2 Desired Counterfactual Motion

用户 / editor 给出：

\[
\hat T(t)
\]

表示目标 edit trajectory。

例如：

- lateral translation
- acceleration
- braking
- stop
- restart
- lane change
- insert trajectory

V5 不直接执行。

---

# 6.3 Constraint Projection Operator

定义：

\[
T^*
=
\Pi_{\mathcal C}(\hat T)
\]

通过：

\[
T^*
=
\arg\min_T
D(T,\hat T)
+
\lambda_jJ_{jerk}
+
\lambda_cJ_{contact}
+
\lambda_hJ_{heading}
+
\lambda_oJ_{collision}
\]

subject to physical constraints。

---

# 6.4 Kinematic Constraints

### Speed

\[
v(t)
\le
v_{max}
\]

### Acceleration

\[
|a(t)|
\le
a_{max}
\]

### Yaw rate

\[
|\dot \psi(t)|
\le
\omega_{max}
\]

### Lateral acceleration

\[
a_{lat}
=
v^2\kappa
\]

约束：

\[
|a_{lat}|
\le
a_{lat,max}
\]

---

# 6.5 Heading-Velocity Consistency

避免：

```text
vehicle heading
与
velocity direction
严重不一致
```

定义：

\[
J_{heading}
=
\sum_t
d(
\mathbf h(t),
\mathbf v(t)
)
\]

---

# 6.6 Road Contact Constraint

车辆 footprint / bottom plane：

\[
d(
actor,
road
)
\approx 0
\]

指标：

- vertical contact error
- floating rate
- road penetration rate

---

# 6.7 Collision / Penetration

检测：

- actor-actor overlap
- actor-static obstacle overlap
- road / curb penetration

定义：

\[
J_{collision}
\]

不追求完整 closed-loop dynamics。

只保证：

> counterfactual edit 不产生明显几何不可行轨迹。

---

# 6.8 Temporal Smoothness

继续保留：

### Velocity

\[
\dot \xi(t)
\]

### Acceleration

\[
\ddot \xi(t)
\]

### Jerk

\[
\dddot \xi(t)
\]

重点加入：

\[
J_{jerk}
=
\int
\|
\dddot\xi(t)
\|^2
dt
\]

相比仅二阶 smoothness，更直接约束编辑轨迹的突变。

---

# 6.9 B-Spline Parameterization

仍允许：

\[
\xi(t)
=
\sum_k
B_{k,3}(t)c_k
\]

\[
T(t)
=
T_0
\exp(
\hat \xi(t)
)
\]

但 Method 中明确：

> cubic B-spline 是连续时间优化的数值参数化，不作为 V5 主要 novelty。

---

# 6.10 Optional Topology-Constrained Local Residual

只有 global actor motion 无法处理：

- local contact
- occlusion boundary
- local alignment

时才启动。

定义：

\[
T_i(t)
=
T_{actor}(t)
\exp(
\delta\xi_i(t)
)
\]

要求：

\[
\|\delta\xi_i\|
\ll
\|\xi_{actor}\|
\]

并用 M1 topology graph 约束：

\[
J_{graph-residual}
=
\sum_{(i,j)\in E}
w_{ij}
\|
\delta\xi_i-
\delta\xi_j
\|^2
\]

禁止发展成大型 deformation network。

---

# 6.11 M3 Ablation

| Arm | 方法 |
|---|---|
| T0 | frame-independent |
| T1 | linear interpolation |
| T2 | V4 frozen SE(3) B-spline |
| T3 | minimum-jerk continuous trajectory |
| T4 | T3 + road/contact constraints |
| T5 | T4 + vehicle kinematics |
| M3-FULL | T5 + optional topology-constrained local residual |

V5 的关键 comparator：

```text
V4 frozen B-spline
```

而不是只和 frame-independent 比。

---

# 6.12 M3 Metrics

### Visual Temporal

- temporal LPIPS
- flow warp L1
- flow warp LPIPS
- flicker
- mask IoU jitter
- Boundary F1 jitter
- centroid jitter

### Trajectory

- trajectory RMSE
- target translation error
- target yaw error

### Physics / Geometry

- velocity violation
- acceleration violation
- jerk
- lateral acceleration violation
- heading-velocity mismatch
- contact error
- floating rate
- penetration rate
- collision rate

### Semantic

- deleted semantic reintroduction
- identity switch

### Reversibility

- rollback render SHA exact

---

# 6.13 M3 Validation Gate

Comparator：

```text
V4 frozen SE(3) B-spline
```

成功条件二选一：

### Gate A

```text
warp / temporal LPIPS
relative improvement >= 10%
```

### Gate B

```text
visual temporal metrics
不关键退化

且

physical violation
relative reduction >= 50%
```

同时：

```text
operation success
不退化

deleted semantic reintroduction
= 0

rollback
= 100% exact
```

---

# 7. 数据策略

## 7.1 V4 Scene 的角色

V4 所有已读 scene：

```text
historical diagnostic only
```

允许：

- failure diagnosis
- mechanism study
- unit / regression
- retrospective plots
- qualitative forensic

禁止：

- V5 confirmatory claim
- V5 final test statistics

---

# 7.2 Fresh nuScenes Cohort

建议重新冻结：

```text
development: 8 scenes
validation:  8 scenes
test:       20 scenes

total:      36 fresh scenes
```

要求：

```text
与 V4 30 scenes
scene-disjoint
```

---

# 7.3 Scene Stratification

只根据结果前可观测 metadata：

- day / night
- clear / rain
- straight / curve
- intersection
- high actor density
- heavy occlusion
- static / moving actor
- near / mid range
- LiDAR support strong / weak
- SAM boundary complexity
- road geometry
- actor class

禁止基于模型 quality 选 scene。

---

# 7.4 Fresh Test Protocol

V5 test 前必须生成：

```text
V5_TEST_FREEZE.json
```

包含：

- source commit
- config SHA
- dataset split SHA
- M1 method
- M2 method
- M3 method
- all thresholds
- graph parameters
- Bayesian parameters
- geometry constraints
- temporal constraints
- baseline list
- metrics schema
- asset SHA

然后：

```text
freeze-only commit
```

再执行：

```text
20-scene exact-once test
```

---

# 7.5 KITTI Cross-Dataset

nuScenes V5 冻结后执行。

流程：

```text
2-sequence adapter smoke
        ↓
10-sequence frozen formal
```

KITTI 禁止重新调：

- Bayesian reliability
- topology graph K
- graph lambda
- boundary barrier
- geometry risk mapping
- geometry threshold
- feasibility threshold
- temporal smoothness
- vehicle kinematic limits
- contact threshold

失败必须区分：

```text
blocked_dataset_adapter
```

与：

```text
cross-domain method failure
```

---

# 8. Baseline 体系

## Tier A：必须执行

### B0

V3.3 frozen

### B1

V4 frozen final

### B2

StreetGS / native reconstruction

### B3

AD-GS

重点用于：

- ownership/motion related comparison
- temporal modeling boundary
- reconstruction quality

---

## Tier B：能力边界比较

根据可执行性：

- SplatAD
- IDSplat
- Gaussian Grouping
- LUDVIG
- AG²aussian / graph-based Gaussian grouping
- relevant object-aware Gaussian baselines

如果数据 adapter / code 不可运行：

```text
paper-only comparison
```

禁止伪造数值。

---

# 9. V5 统计协议

统计单位：

```text
scene
```

不是 pixel。

所有主指标报告：

- mean
- median
- std
- IQR
- 95% scene-bootstrap CI

Pairwise：

- paired bootstrap
- permutation test
- Wilcoxon signed-rank

随机模块：

```text
final candidate
至少 3 seeds
```

确定性模块：

```text
byte-exact replay
```

---

# 10. Engineering Metrics

继续保留 V4 工程优势。

每 scene：

```text
T_prepare
T_train
T_semantic
T_graph
T_repair
T_compile
T_render
T_eval
T_total
```

GPU：

- peak VRAM
- peak torch allocated
- peak torch reserved
- peak RAM
- OOM count
- oom_kill count

Asset：

- base bytes
- semantic state bytes
- graph bytes
- delta bytes
- actor bytes
- package bytes

Runtime：

- cold load
- warm load
- compose wall
- P50 frame time
- P95 frame time
- FPS

Production：

- pipeline success
- valid edit yield
- accepted clips / GPU-hour
- scenes / GPU-day
- GPU-hours / accepted clip
- retry amplification
- resume efficiency

---

# 11. Downstream Utility

冻结公开 detector / tracker / BEV model。

四臂：

```text
real
base render
edited render
rollback
```

报告：

- detection mAP
- recall
- prediction consistency
- tracking consistency
- real-to-sim gap

编辑专项：

### Remove

目标 actor detection 应消失。

### Insert

目标 actor 应在预期位置被检测。

### Lateral

预测位置应随编辑轨迹移动。

### Rollback

应恢复 base prediction。

---

# 12. V5 主表设计

## Table 1：Reconstruction / Preservation

- PSNR
- SSIM
- LPIPS
- Global / Static / Actor / Boundary

## Table 2：Structured Ownership

- IoU
- Boundary F1
- FP/FN semantic mass
- ECE
- Brier
- graph leakage
- component purity

## Table 3：Repair

- Hole PSNR
- Hole LPIPS
- Geometry MAE
- P95 geometry error
- coverage
- valid yield

## Table 4：Selective Risk

- accepted error
- abstain error
- risk-coverage
- oracle regret

## Table 5：Temporal

- tLPIPS
- warp
- flicker
- trajectory RMSE
- identity switch

## Table 6：Physical Validity

- acceleration violation
- jerk
- lateral acceleration
- contact error
- collision / penetration

## Table 7：KITTI Cross-Dataset

frozen V5 only。

## Table 8：Engineering

production metrics。

## Table 9：Main Ablation

M1 / M2 / M3。

---

# 13. V5 Figure 设计

## Figure 1

V4 failures → V5 structured solution：

```text
scalar evidence
→ structured field

weighted risk
→ feasibility-first routing

B-spline motion
→ constraint-projected delta
```

## Figure 2

M1：

```text
SAM views
↓
Bayesian unary
↓
Gaussian topology graph
↓
boundary-aware posterior
```

## Figure 3

Gaussian physical graph：

- covariance
- KNN candidates
- depth barrier
- LiDAR anchor
- motion edge

## Figure 4

M2 geometry forensic：

```text
candidate
→ raw geometry
→ Gaussianization
→ render geometry
→ router
```

## Figure 5

Feasibility-first router。

## Figure 6

M3 desired trajectory → projected feasible trajectory。

## Figure 7

Risk-coverage / geometry-tail plot。

## Figure 8

nuScenes qualitative。

## Figure 9

KITTI frozen qualitative。

## Figure 10

quality / geometry / wall / VRAM Pareto。

---

# 14. V5 Method 章节建议

```text
3. Method

3.1 Reversible Gaussian World Asset

3.2 Reliability-Aware Semantic Evidence

3.3 Physical Gaussian Topology Graph

3.4 Structured Ownership Inference

3.5 Geometry-First Repair Candidates

3.6 Geometry-Feasible Selective Routing

3.7 Constraint-Projected Temporal Delta

3.8 Atomic Composition and Rollback
```

---

# 15. V5 Contribution 设计

最终最多写 4 个。

## Contribution 1：Structured Gaussian Ownership

> 将 noisy multi-view SAM supervision 建模为 reliability-aware Bayesian unary，并通过 Gaussian-native physical topology graph、depth boundary 与 LiDAR surface support 做结构化 3D ownership inference。

---

## Contribution 2：Geometry-Feasible Selective Repair

> 将 repair 从无条件 weighted risk minimization 改成 geometry / temporal feasibility 约束下的 selective source routing，在无可信修复时显式 abstain。

---

## Contribution 3：Constraint-Projected Temporal Delta

> 将任意 counterfactual actor trajectory 投影到满足车辆运动学、路面接触、平滑与碰撞约束的连续时间 SE(3) Delta；B-spline 只作为 trajectory parameterization。

---

## Contribution 4：Transactional Reversible Gaussian World Editing

> Immutable base、content-addressed delta、atomic composition、exact rollback、provenance 与 fail-closed execution，共同形成可维护驾驶 WorldSim asset。

---

# 16. V5 Critical Path

严格顺序：

```text
P0
V5 scientific scope freeze
        ↓
D0
fresh 36-scene nuScenes cohort
        ↓
M1-D0
V4 Bayesian failure diagnosis
        ↓
M1.1
Reliability-aware Bayesian unary
        ↓
M1.2
Gaussian physical topology
        ↓
M1.3
Graph posterior
        ↓
M1 validation
        ↓
PASS → freeze M1
        ↓
M2-D0
+3.3908m geometry forensic
        ↓
identify:
router / candidate / gaussianization / metric
        ↓
M2.1
Geometry-first candidate
        ↓
M2.2
Feasibility-first router
        ↓
M2 validation
        ↓
PASS → freeze M2
        ↓
M3.1
constraint projection
        ↓
M3.2
kinematic/contact constraints
        ↓
M3.3
optional topology residual
        ↓
M3 validation
        ↓
freeze all methods
        ↓
V5_TEST_FREEZE.json
        ↓
20-scene exact-once test
        ↓
KITTI frozen cross-domain
        ↓
engineering
        ↓
downstream
        ↓
statistics
        ↓
paper / release
```

---

# 17. Task Registry

| Task ID | 初始状态 | 内容 |
|---|---|---|
| `WS-V5-P0-SCOPE-FREEZE-01` | pending | 冻结 V5 科学问题、协议、claim |
| `WS-V5-D0-NUSCENES-FRESH-COHORT-01` | pending | 8 dev + 8 val + 20 fresh test |
| `WS-V5-M1-D0-BAYES-FORENSICS-01` | pending | 复盘 V4 M1 失败机制 |
| `WS-V5-M1-STRUCTURED-OWNERSHIP-01` | pending | Bayesian unary + physical graph |
| `WS-V5-M1B-REVERSIBLE-SEMANTIC-SPLIT-01` | pending | 条件任务；仅在 M1 boundary gate 需要时解锁 semantic split |
| `WS-V5-M2-D0-GEOMETRY-FORENSICS-01` | pending | +3.3908m 根因诊断 |
| `WS-V5-M2-GEOMETRY-FIRST-REPAIR-01` | pending | 3D scaffold repair |
| `WS-V5-M2-GEOMETRY-FEASIBLE-ROUTER-01` | pending | feasibility-first routing |
| `WS-V5-M3-CONSTRAINT-PROJECTED-TEMPORAL-01` | pending | physics-constrained temporal delta |
| `WS-V5-E0-NUSCENES-TEST-01` | pending | fresh 20-scene exact-once test |
| `WS-V5-D1-KITTI-ADAPTER-01` | pending | KITTI frozen adapter |
| `WS-V5-E1-KITTI-CROSSDATA-01` | pending | KITTI formal confirmation |
| `WS-V5-E2-ENGINEERING-BENCH-01` | pending | production benchmark |
| `WS-V5-E3-DOWNSTREAM-UTILITY-01` | pending | detector/tracker/BEV utility |
| `WS-V5-R0-RELEASE-01` | pending | release |
| `WS-V5-W0-PAPER-01` | pending | paper |

---

# 18. V5 Early Stop Rules

## M1

如果 Bayesian unary 自身没有优于 V4 Beta：

```text
先停止 graph 大扩展
```

检查：

- observation reliability
- calibration
- pseudo-label quality

如果 unary 有效、graph 失败：

检查：

- KNN
- covariance affinity
- boundary leakage

禁止直接加 transformer。

---

## M2

如果 forensic 显示：

```text
reference / metric 自身有问题
```

先修 evaluator。

如果：

```text
candidate 全部 geometry bad
```

先修 candidate。

如果：

```text
有 geometry-good candidate
但 router 选错
```

才改 router。

禁止跳过 causal diagnosis。

---

## M3

如果 constraint projection：

```text
physical metrics 不改善
```

不再继续加 local residual。

如果 physical validity 改善但 visual temporal 关键退化：

```text
M3 rejected / simplify
```

---

# 19. V5 Paper Claim Boundary

允许：

> 在 fresh scene-disjoint nuScenes 与 frozen KITTI cross-domain 协议上，V5 通过结构化 Gaussian ownership、geometry-feasible selective repair 与 constraint-projected temporal delta，提高可编辑驾驶 Gaussian world asset 的语义边界、几何可靠性、时序一致性与可逆维护能力。

禁止：

- full world model
- closed-loop safety SOTA
- physics simulator
- guaranteed collision-free planning
- arbitrary city / weather universal generalization
- generated content = ground truth
- KITTI 调参后称 cross-domain
- 用 V4 已打开 test 当 V5 final confirmation

---

# 20. V5 第一阶段 Coding Agent 提示词

```text
接手 Motion-Proj / WorldSim V5。

项目目录：
/root/autodl-tmp/motion_proj

V4 已正式完成：
M1 rejected；
M2 validation/test 证明 selective routing 有效，但 hole geometry MAE 有 +3.3908096237m 严重 caveat；
M3 SE(3) B-spline temporal delta 已通过 frozen validation 与 18-scene exact-once test。

不要重新执行 V4。
不要修改 V4 canonical artifact。

当前只执行：

1.
WS-V5-P0-SCOPE-FREEZE-01

2.
WS-V5-M1-D0-BAYES-FORENSICS-01

3.
WS-V5-M2-D0-GEOMETRY-FORENSICS-01

暂时禁止直接实现完整 M1/M2/M3。

A. 先审计：
- git HEAD / status
- V4 final manifests
- r335 / r336
- M1 canonical val artifacts
- M2 canonical dev/val/test artifacts
- M3 canonical artifacts
- current V4 paper / failure docs

B. M1 forensic：
目标不是证明 SAM supervision 错。
目标是判断 V4 Beta evidence 为什么无法 scene-disjoint 泛化。

逐项检查：
- per-view SAM observation
- visibility
- depth consistency
- LiDAR support
- boundary distance
- effective evidence count
- posterior saturation
- view disagreement
- calibration
- neighborhood inconsistency

生成：
- per-Gaussian diagnostic table
- per-scene failure categories
- boundary vs interior statistics
- posterior vs local topology disagreement

C. M2 forensic：
专项审查 +3.3908096237m hole geometry degradation。

逐 request 生成：
- candidate raw geometry
- candidate rendered geometry
- geometry reference
- photo risk
- raw geometry error
- normalized geometry risk
- temporal risk
- uncertainty
- selected candidate
- geometry oracle candidate

必须检查：
是否存在 risk clipping / saturation，
使 2m 和 5m geometry error 映射成相同 risk。

分类：
1. router failure
2. candidate failure
3. Gaussianization / rendering failure
4. metric/reference failure

D. P0 同时冻结：
V5 fresh nuScenes protocol：
8 dev
8 val
20 test
全部与 V4 30 scenes scene-disjoint。

V4 test scenes 只允许 diagnostic，
不得成为 V5 confirmatory test。

E. 输出：
docs/WORLDSIM_V5_STRUCTDELTA_PLAN.md
docs/WS_V5_M1_FAILURE_FORENSICS.md
docs/WS_V5_M2_GEOMETRY_FORENSICS.md
configs/worldsim_v5/p0_scope_v1.yaml
configs/worldsim_v5/nuscenes_fresh_cohort_v1.yaml

F. P0 / forensic 完成以前禁止：
- 新模型大训练
- validation 调参
- fresh test quality
- KITTI 方法调参
- diffusion / transformer 扩展
```

---

# 21. V5 最终原则

V4：

```text
single Bayesian probability
解决 ownership

weighted scalar risk
解决 repair

B-spline
解决 temporal
```

V5：

```text
Bayesian Evidence
+
Physical Topology
→ ownership
```

```text
Geometry Feasibility
→ Risk Selection
→ repair
```

```text
Desired Motion
→ Constraint Projection
→ Reversible Continuous Delta
→ temporal editing
```

最终研究原则：

> **Observation 不等于 structure；uncertainty 不等于 geometry；photometric quality 不等于 physical correctness；temporal smoothness 不等于 feasible motion。**

V5 的目标不是继续堆模块，而是把 V4 已经验证出的三条有效研究线升级为一个更结构化、更物理可信、更容易 defend 的顶会方法。
