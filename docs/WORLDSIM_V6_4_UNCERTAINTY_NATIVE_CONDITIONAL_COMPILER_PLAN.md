# WorldSim V6.4 — 原生不确定性与条件风险物理状态编译器

> 英文工作名：**UNC-Compiler — Uncertainty-Native Conditional Physical-State Compiler**
>
> 推荐技术报告题目：
>
> **UNC-Compiler: Native Uncertainty and Conditional Risk Control for Verifiable Driving World Compilation**
>
> 中文题目：
>
> **UNC-Compiler：面向可验证驾驶世界编译的原生不确定性与条件风险控制**
>
> 状态：`pending`
>
> 默认执行环境：**现有 AutoDL RTX 3090 服务器**
>
> GPU 合同：**默认单卡；允许按阶段使用 2–4 张 RTX 3090，不限制必须单卡**
>
> 上游冻结终态：
>
> - 分支：`research/worldsim-v6.3-surface-tail`
> - V6.3 最终状态：`v63_surface_architecture_family_closed_negative_p7_locked`
> - V6.3 最终文档提交：运行时以远端分支 HEAD 为准；用户提供的最新证据为 `c192955`
> - V6.3 关键失败：`V63-F24`
> - V6.3 B4/B5/M0、P7–P11：未执行、未读取
>
> 核心任务：
>
> > **保留已经证明更强的 Native Pointwise B2，不再增加 Surface 编码器；在 IR-WM 原生 logits / BEV features 上显式学习 aleatoric / epistemic uncertainty，以 scene / stratum 条件风险约束决定 OCC authority，并通过独立 case-level 校准输出可固化的 FREE / OCCUPIED / UNKNOWN 物理状态。**
>
> 可选扩展：
>
> > 只有当冻结 IR-WM 的原生特征已经证明具有不确定性信号、但 scene-disjoint 条件可靠性仍受表示上限限制时，才解锁一次 **targeted LoRA / PEFT**。不直接全量解冻 IR-WM。

---

# 0. V6.4 北极星

V6.4 不再研究：

```text
更多 Surface 邻接
更大的 Surface Transformer
Surface-Max / Surface-CVaR 补跑
B3 换 seed
降低 area / UNKNOWN gate
用 pooled 指标覆盖 scene failure
```

V6.3 已经证明：

```text
Native Pointwise B2
>
当前 Surface-Mean B3
```

正式 P6 两个 scene 均不支持 B3：

```text
scene-0450:
tail 相对 B2 恶化 19.85%
写入面积仅为 B2 的 40.63%
UNKNOWN 失门

scene-1089:
tail 相对 B2 恶化 41.01%
写入面积仅为 B2 的 49.93%
```

pooled：

```text
B2 tail = 0.491496
B3 tail = 0.608174

B3 写入面积 / B2 = 45.56%
```

因此 V6.4 的问题定义正式改为：

> **隐藏空间 False-safe 的核心瓶颈，不是缺少曲面聚合，而是当前 Occupancy 表示无法可靠表达“模型什么时候不知道”，并且 pooled 约束不能保证不同 scene / failure stratum 的条件可靠性。**

核心路线：

```text
IR-WM Native Logits / BEV Latent
                  │
        ┌─────────┴─────────┐
        │                   │
  Occupancy State       Native Uncertainty
                      ┌──────┴──────┐
                  Aleatoric      Epistemic
                      └──────┬──────┘
                             │
              Observed FREE / OCC / Conflict
                             │
             Scene / Stratum Conditional Constraints
                             │
                  FREE / OCCUPIED / UNKNOWN
                             │
               Independent Case-Level Calibration
                             │
                   SceneIR Physical State
                             │
        ┌────────────────────┼────────────────────┐
        ↓                    ↓                    ↓
      LogSim              WorldSim            NWM / RL
```

---

# 1. V6.3 必须继承的科研事实

## 1.1 强基线：Native Pointwise B2

冻结 P6 B2：

```text
hard violations = 0
pooled common hidden-FREE tail = 0.491496
safe-OCC retention = 0.851056
source-valid UNKNOWN = 0.266284
OCC area coverage = 0.626256
emitted OCC points = 2,298,450
```

V6.4 的任何 learned method 必须和 B2 matched comparison。

禁止将：

```text
IR-WM argmax
```

作为唯一强基线。

---

## 1.2 P5D / P5R 的保留结论

V6.3 P5D 证明：

```text
tail weighted gradient
≈ retention weighted gradient 的 5.53 倍

tail / retention gradient cosine
≈ -0.412
```

原 weighted objective 导致：

```text
safe-OCC retention = 0
```

P5R 通过 proxy primal-dual 恢复了训练侧可行性：

```text
hard = 0
retention = 0.721226
OCC coverage = 0.114148
non-UNKNOWN = 0.686101
tail = 0.464393
```

因此 V6.4 继续使用：

```text
显式可行性约束
+
lexicographic candidate selection
```

禁止回到普通 weighted sum / argmin loss。

---

## 1.3 V6.3 已经证伪的具体方法家族

已关闭：

```text
当前 Surface-Mean architecture
+
当前 surface corpus / encoder
+
当前 P6 matched protocol
```

没有被证明失败、但按 Stop 2 未执行：

```text
Surface-Max
Surface-CVaR
M0 Authority
P7 calibration
P8 legacy
P9 confirmation
P10 test
```

V6.4 不补跑这些 arm。

---

## 1.4 V6.4 需要回答的新问题

1. Native BEV feature 是否包含可泛化的 aleatoric / epistemic uncertainty？
2. Feature-level uncertainty 是否明显优于：
   - max-softmax probability；
   - entropy；
   - logit margin；
   - Native B2 的 UNKNOWN probability？
3. scene / stratum conditional constraints 是否能避免 pooled gate 通过、逐 scene 失败？
4. 不确定性输出经过独立 case-level calibration 后，能否在不退化为 all-UNKNOWN 的情况下控制 False-safe？
5. Frozen IR-WM 是否已经足够？
6. 若 frozen feature 出现 representation ceiling，targeted LoRA 是否能减少 scene-disjoint gap？
7. 改善是否能转化到：
   - GS + LogSim；
   - GS + WorldSim；
   - GS + NWM collision critic？

---

# 2. 前沿方法调研结论

调研日期：**2026-08-26**

以下只使用论文官方页面、会议页面和官方代码仓库作为执行依据。

---

## 2.1 OCCUQ — V6.4 第一 faithful migration

论文：

```text
OCCUQ: Exploring Efficient Uncertainty Quantification for 3D Occupancy Prediction
ICRA 2025
```

官方代码：

```text
https://github.com/ika-rwth-aachen/OCCUQ
```

许可证：

```text
Apache-2.0
```

核心机制：

```text
Dense Occupancy Head
├── Aleatoric UQ Module
└── Feature-Level Gaussian Mixture Model
       ↓
   Epistemic Uncertainty
```

官方公开流程包括：

- 单卡拟合 Gaussian Mixture Model（GMM）；
- 单卡推理；
- corruption / missing-camera 下的 uncertainty 评测；
- feature-level epistemic uncertainty；
- voxel-level aleatoric uncertainty。

V6.4 迁移方式：

```text
IR-WM native occupancy feature
+
IR-WM native BEV latent
        ↓
OCCUQ-style aleatoric head
+
class / geometry conditioned GMM
```

不 faithful 的部分必须显式标记：

```text
OCCUQ 原始 backbone = SurroundOCC
V6.4 backbone = IR-WM
```

因此 V6.4 主张：

> faithful uncertainty mechanism migration，而不是 faithful full-system reproduction。

---

## 2.2 ReliOcc — Hybrid uncertainty 与离线校准参考

论文：

```text
Reliable and Calibrated Semantic Occupancy Prediction by Hybrid Uncertainty Learning
IJCAI 2025
```

官方会议页面：

```text
https://www.ijcai.org/proceedings/2025/220
```

核心：

- individual voxel uncertainty；
- sampling noise；
- relative-voxel uncertainty；
- mix-up learning；
- uncertainty-aware offline calibration；
- sensor failure / OOD robustness。

代码状态：

```text
截至本计划冻结时，未确认有可直接依赖的官方完整实现。
```

V6.4 使用方式：

- 作为 UQ fusion 与 corruption protocol 参考；
- 不在 P3 第一轮一次性复现全部 ReliOcc；
- 只有 OCCUQ faithful arm 有效后，才允许单独注册一个 ReliOcc-inspired hybrid arm；
- 不允许把论文描述直接包装成“官方代码迁移”。

---

## 2.3 EvOcc — FREE / OCC / unobserved / contradiction target

论文：

```text
EvOcc: Accurate Semantic Occupancy for Automated Driving Using Evidence Theory
CVPR 2025
```

官方论文：

```text
https://openaccess.thecvf.com/content/CVPR2025/html/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.html
```

核心：

- 显式区分未观测区域；
- 显式处理 measurement contradiction；
- evidence theory semantic occupancy；
- ray-based evaluation。

官方代码状态：

```text
计划冻结时，公开仓库不具备可直接依赖的完整正式实现。
```

V6.4 使用方式：

- 迁移 evidential target definition；
- 沿用项目已有 FREE / OCCUPIED / contradiction / UNKNOWN；
- 将 contradiction 与 unobserved 作为 UQ head 的显式监督；
- 不声称 faithful EvOcc code port。

---

## 2.4 α-OCC — 分层保形预测

论文：

```text
α-OCC: Uncertainty-Aware Camera-based 3D Semantic Occupancy Prediction
```

官方页面：

```text
https://arxiv.org/abs/2406.11021
```

核心：

- Depth uncertainty propagation；
- geometry / semantic hierarchical conformal prediction；
- 处理 OCC 高度类别不平衡；
- 输出 prediction set，而非单一 argmax。

V6.4 使用方式：

```text
第一层：
FREE / OCCUPIED / UNKNOWN 几何集合

第二层：
仅对 singleton OCCUPIED 的语义类别做集合校准
```

当前 World Compiler 的第一主目标是几何 authority，因此：

```text
P6 先完成几何级集合输出
语义级 HCP 为后续消融
```

---

## 2.5 Conformal Risk Control

核心参考：

```text
Conformal Risk Control
Non-Exchangeable Conformal Risk Control
```

官方页面：

```text
https://openreview.net/forum?id=33XGfHLtZg
https://openreview.net/forum?id=j511LaqEeP
```

V6.4 使用边界：

- calibration 单位必须是 case / target unit；
- 不把百万个相关 voxel 当成百万个独立样本；
- 只对冻结、单调、有界 loss 做 risk calibration；
- 报告有限样本上界；
- 不将其表述为现实道路绝对安全保证；
- 非交换版本只在明确存在时间/场景加权合同后使用。

---

## 2.6 End-to-end Conditional Robust Optimization

论文：

```text
End-to-end Conditional Robust Optimization
UAI 2024
```

官方页面：

```text
https://proceedings.mlr.press/v244/chenreddy24a.html
```

迁移价值：

- 不只优化 pooled risk；
- 同时考虑 decision risk 与 conditional coverage quality；
- 适合解释 V6.3 的：
  - pooled feasibility；
  - scene-0450 / scene-1089 双双失败。

必须保持边界：

> 该工作不等于为 V6.4 提供形式化 scene-conditional coverage 保证。V6.4 只主张 empirical conditional robustness，除非后续理论条件真正满足。

---

## 2.7 IR-WM — 继续作为第一 foundation

论文 / 官方代码：

```text
Vision-Centric 4D Occupancy Forecasting and Planning via Implicit Residual World Models
https://arxiv.org/abs/2510.16729
https://github.com/APRIL-ZJU/IR-WM
```

关键结构：

```text
Scene Encoder
+
Previous BEV Temporal Prior
+
Residual World Evolution
+
Feature Alignment
+
Occupancy Head
+
Planning Head
```

V6.4 继续使用现有冻结 IR-WM checkpoint 与 sidecar。

原因：

- 已有完整本地环境、checkpoint 与 native feature pipeline；
- B2 已证明 frozen features 有强信号；
- 直接换 backbone 会同时改变表示、数据、训练和评价，破坏归因。

---

## 2.8 DriveDiTFit — PEFT / LoRA 的驾驶领域先例

论文 / 官方代码：

```text
DriveDiTFit: Fine-tuning Diffusion Transformers for Autonomous Driving Data Generation
https://arxiv.org/abs/2407.15661
https://github.com/TtuHamg/DriveDiTFit
```

它证明：

- 参数高效微调（PEFT）可用于驾驶 domain adaptation；
- 少量可训练参数可以优于直接 full fine-tuning；
- target module 与任务 discrepancy 比“LoRA”名义本身更重要。

它不是 Occupancy World Model 方法。

V6.4 只将其作为：

```text
targeted PEFT 的领域先例
```

而不是直接迁移方法。

---

## 2.9 Emerging references

可登记但不作为 P0–P4 依赖：

- OccOoD：feature-level Occupancy OOD detection；
- SUG-Occ：semantic / uncertainty guided sparse projection；
- M²-Occ：missing-camera feature reconstruction；
- Query2Uncertainty：query density 与分布移位不确定性。

所有 emerging method：

- 必须在使用前重新检查正式发表状态；
- 必须检查官方代码与许可证；
- 不得仅因发布时间更新就替换 OCCUQ / EvOcc / ReliOcc 主基线。

---

# 3. V6.4 方法：UNC-Compiler

## 3.1 世界状态

继续使用：

\[
\mathcal W_t =
(
\mathcal G_t^{app},
\mathcal O_t^{phys},
\mathcal X_t^{dyn},
\mathcal P_t,
\mathbf q_t
)
\]

V6.4 只改变：

\[
\mathcal O_t^{phys}
\]

的 learned authority。

---

## 3.2 输入

每个 native query / voxel：

\[
f_v =
[
l_v^{IRWM},
z_v^{BEV},
e_v^{hard},
d_v^F,
d_v^O,
m_v^{source},
s_v^{stratum},
c_v^{scene}
]
\]

其中：

- \(l_v^{IRWM}\)：17 类 native occupancy logits；
- \(z_v^{BEV}\)：256 维 native BEV latent；
- \(e_v^{hard}\)：FREE / OCCUPIED / contradiction / lifecycle；
- \(d_v^F,d_v^O\)：到 observed FREE / OCC 的距离；
- \(m_v^{source}\)：source-valid / temporal support；
- \(s_v^{stratum}\)：route/static/actor/multi-actor；
- \(c_v^{scene}\)：仅用于条件约束分组，不作为可记忆 scene ID embedding。

禁止：

```text
class prototype
argmax → fabricated logits
surface graph / surface transformer
scene ID lookup embedding
O_eval feature
```

---

## 3.3 State Head

保留 Native B2 的 pointwise architecture 与 hard projection。

输出：

\[
P_v(F),P_v(O),P_v(U)
\]

硬投影：

```text
Observed FREE
→ FREE

Observed OCC
→ OCCUPIED

Contradiction
→ UNKNOWN

Outside Actor lifecycle
→ UNKNOWN
```

要求：

```text
hard violations = 0
```

---

## 3.4 Aleatoric Uncertainty Head

参考 OCCUQ：

```text
native logits / latent
→ aleatoric head
→ u_v^ale
```

推荐第一版：

- 两层 MLP；
- hidden 128；
- 输出 positive scale / evidence；
- 使用 softplus；
- 对 observed contradiction / label ambiguity 进行监督；
- 对同一 source 的 sensor corruption / evidence dropout 建模。

可选实现：

### A. Heteroscedastic Logit Scale

\[
\tilde l_v =
l_v / \sigma_v
\]

### B. Dirichlet Evidence

\[
\alpha_{v,k}=e_{v,k}+1
\]

P1 必须只冻结一个正式方案。

另一个方案仅可作为预注册消融。

---

## 3.5 Epistemic Uncertainty

第一正式方案使用 OCCUQ-style GMM：

```text
IR-WM BEV / occupancy feature
→ feature projection
→ class / geometry conditioned GMM
→ negative log-likelihood
→ u_v^epi
```

### GMM 输入

不直接用 256D 全量 feature 拟合 full-covariance。

推荐：

```text
256D BEV
→ frozen PCA / train-only linear projection
→ 16D or 32D
→ diagonal GMM
```

维度和 component 数必须在任何 selection quality 前，通过 train-only BIC / capability protocol冻结。

正式 P3 不允许：

```text
在 selection 上调 component 数
```

### 条件化层级

第一层：

```text
FREE / OCCUPIED / UNKNOWN geometry
```

第二层可选：

```text
semantic class
```

如果 rare class sample 不足：

```text
fallback to geometry-level GMM
```

---

## 3.6 Hybrid Uncertainty

\[
u_v^{hyb}
=
h_\phi
(
u_v^{ale},
u_v^{epi},
H(P_v),
margin(P_v),
m_v^{source}
)
\]

第一版 `h_\phi` 使用单调、低容量融合：

- non-negative linear；
- isotonic / monotone MLP；
- 不使用大 Transformer。

目的：

```text
避免再引入一个能够记忆 scene 的复杂后处理模型
```

---

## 3.7 Evidential Target

每个 query target：

```text
Observed FREE
Observed OCCUPIED
Contradiction
Unobserved / hidden
```

其中：

- contradiction 不使用多数票变成 FREE/OCC；
- unobserved 不自动作为 FREE；
- hidden O_eval 只可作为 train dropout / selection / calibration / confirmation 对应角色使用；
- method 时刻不可见的 target 不能泄漏到 inference feature。

---

## 3.8 Conditional Risk / Coverage

V6.3 pooled 约束：

```text
global retention
global coverage
global UNKNOWN
```

V6.4 改成两层。

### Global Constraints

\[
g^{global}_{ret}
=
\rho_{global}-Retention
\]

\[
g^{global}_{unk}
=
UNKNOWN-\tau_{global}
\]

\[
g^{global}_{risk}
=
Risk-\epsilon_{global}
\]

### Stratum Constraints

对：

```text
route_support
static_disocclusion
actor
multi_actor（后续）
```

分别定义：

\[
g^{stratum}_{k}
\]

### Scene-Robust Objective

训练期不能记忆固定 scene ID。

使用：

```text
每个 batch / epoch 的 scene group loss
→ worst-scene / robust aggregation
```

推荐第一版：

\[
L_{robust}
=
\operatorname{LogSumExp}_\tau
\left(
L_{scene_1},...,L_{scene_n}
\right)
\]

或：

```text
per-scene multipliers
```

P1 只冻结一个。

---

## 3.9 Conditional Primal-Dual

继续使用 V6.3 已验证的 proxy primal-dual。

可行性顺序：

1. hard violation = 0；
2. global constraints pass；
3. 每个 sample-sufficient stratum constraint pass；
4. scene-robust risk；
5. pooled risk；
6. accuracy。

候选必须：

```text
global feasible
+
至少 3/4 selection scenes feasible
+
每个正式 stratum 非退化
```

不得：

```text
pooled pass
但 0/4 scene support
```

---

## 3.10 Set-Valued Physical State

经过独立 calibration：

```text
{FREE}
{OCCUPIED}
{UNKNOWN}
{FREE, UNKNOWN}
{OCCUPIED, UNKNOWN}
{FREE, OCCUPIED, UNKNOWN}
```

写入规则：

```text
{OCCUPIED}
→ 允许写入新的 collision state

{FREE}
→ 允许写入 certified FREE

其他
→ UNKNOWN / ABSTAIN
```

---

# 4. V6.4 Baseline Matrix

| Arm | Native State | Aleatoric | Epistemic GMM | Conditional Constraints | Calibration | LoRA |
|---|---:|---:|---:|---:|---:|---:|
| B0 IR-WM argmax | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| B1 Hard Projection | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| B2 Native Pointwise | ✓ | ✗ | ✗ | global | ✗ | ✗ |
| U0 Entropy/Margin | ✓ | post-hoc | ✗ | ✗ | ✗ | ✗ |
| U1 Aleatoric | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| U2 Epistemic GMM | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| U3 Hybrid UQ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| C0 Global UNC | ✓ | ✓ | ✓ | global | ✗ | ✗ |
| M0 UNC-Compiler | ✓ | ✓ | ✓ | **scene + stratum** | ✗ | ✗ |
| M1 UNC-Compiler-Cal | ✓ | ✓ | ✓ | scene + stratum | **case-level** | ✗ |
| L0 UNC-Compiler-LoRA | ✓ | ✓ | ✓ | scene + stratum | case-level | **targeted** |

---

# 5. 数据纪律

## 5.1 旧数据处理

V6.1–V6.3 已经读取 quality 的 scene / target：

```text
Tier L — Legacy / mechanism only
```

用途：

- 机制解释；
- 历史回归；
- frozen retrospective；
- arXiv failure analysis。

禁止：

- V6.4 train；
- model selection；
- UQ threshold；
- GMM component 选择；
- calibration；
- LoRA selection。

必须由脚本从：

- `RESEARCH_STATUS.md`
- `EXPERIMENTS.md`
- autoresearch states
- run manifests

自动构建：

```text
USED_SCENE_LEDGER.json
```

---

## 5.2 Fresh Cohort

推荐正式 cohort：

### Tier D-Train

```text
12 fresh scenes
12 target units / scene
共 144 units
```

### Tier D-Selection

```text
4 fresh scenes
12 target units / scene
共 48 units
```

### Tier C-Calibration

```text
6 fresh scenes
12 target units / scene
共 72 case-level units
```

### Tier H-Confirmation

```text
3 fresh scenes
12 target units / scene
共 36 units
```

### Tier T-Test

```text
4 fresh scenes
12 target units / scene
共 48 units
```

总计：

```text
29 fresh scenes
348 target units
```

如果本地可用 scene 不足：

- 不得事后减少 denominator；
- 在 P1 quality read 前冻结可用最大 cohort；
- Calibration 不低于 72 case units；
- Confirmation 不低于 3 scenes；
- Test 不低于 4 scenes。

---

## 5.3 Metadata-Only Scene Selection

可使用：

- location；
- time-of-day；
- weather；
- actor count metadata；
- sensor completeness；
- frame count；
- motion level metadata；
- map availability。

禁止使用：

- Occupancy accuracy；
- hidden-FREE risk；
- UQ score；
- proposal acceptance；
- B2/B3 result；
- rendered quality。

---

## 5.4 Stratum

每个 cohort 至少覆盖：

```text
route_support
static_disocclusion
actor
```

Multi-actor 在 P10 后解锁。

每个正式 stratum 在 D-Selection 至少：

```text
12 target units
```

否则该 stratum 只能报告，不允许训练独立 threshold / multiplier。

---

# 6. Evidence Role 隔离

每个 target：

```text
E_method
E_train_dropout
E_selection_target
E_calibration_target
E_confirmation_target
E_test_target
```

要求：

- role identity 完全可审计；
- method 不得读取未来 target；
- GMM 只拟合 train features；
- calibration 只选择风险 policy，不训练模型；
- confirmation / test 先生成 decision，再读取 target；
- exact-once attempt 在 quality read 前创建。

---

# 7. P0 — Repo / 文档 / 分支前置

Task：

```text
WS-V64-P0-SCOPE-GIT-01
```

## 7.1 运行时确认

```bash
git fetch --all --prune
git status --short
git rev-parse HEAD
git rev-parse origin/research/worldsim-v6.3-surface-tail
```

确认：

```text
V6.3 terminal
V63-F24 route-closed
P7–P11 unread
worktree clean
```

## 7.2 V6.3 合并

若 V6.3 尚未进入 main：

1. 从最新 main 创建 integration branch；
2. merge V6.3；
3. 运行定向文档/配置/核心测试；
4. 普通 push main；
5. 不 force。

## 7.3 新分支

```text
research/worldsim-v6.4-native-uq
```

存在时：

```text
research/worldsim-v6.4-native-uq-rNN
```

## 7.4 状态文档

更新：

- `docs/RESEARCH_STATUS.md`
- `docs/RESEARCH_FAILURES.md`
- `docs/EXPERIMENTS.md`

新增：

```text
docs/WORLDSIM_V6_4_UNCERTAINTY_NATIVE_CONDITIONAL_COMPILER_PLAN.md
docs/autoresearch/worldsim_v64/
```

不修改 V6.3 closeout 结论。

---

# 8. P1 — Novelty / License / Protocol Freeze

Task：

```text
WS-V64-P1-NOVELTY-PROTOCOL-01
```

## 8.1 Source Matrix

输出：

```text
V64_SOURCE_MATRIX.md
V64_LICENSE_MATRIX.json
```

至少包括：

- OCCUQ；
- EvOcc；
- ReliOcc；
- α-OCC；
- Conformal Risk Control；
- Conditional Robust Optimization；
- IR-WM；
- DriveDiTFit；
- OccOoD（reference only）。

## 8.2 License Gate

- OCCUQ Apache-2.0：允许参考/迁移；
- IR-WM：按官方仓库许可证审计；
- 代码未公开的论文：只做 paper mechanism；
- 无明确许可证仓库：不得 vendor / copy；
- checkpoint 使用范围单独记录。

## 8.3 方法冻结

在任何 V6.4 quality read 前冻结：

- UQ architecture；
- GMM feature；
- GMM dimension；
- component selection protocol；
- hybrid fusion；
- scene/stratum constraints；
- candidate selection；
- cohort；
- calibration loss；
- gates；
- LoRA unlock rule；
- resource contract。

---

# 9. P2 — Fresh Cohort 与 Native Sidecar

Task：

```text
WS-V64-P2-FRESH-NATIVE-SIDECAR-01
```

## 9.1 复用代码

优先复用：

```text
motion_proj/worldsim_v62/irwm_sidecar.py
motion_proj/worldsim_v63/native_features.py
motion_proj/worldsim_v63/native_pointwise.py
motion_proj/worldsim_v62/projection.py
motion_proj/worldsim_v62/query_dataset.py
```

## 9.2 Sidecar 内容

每个 target：

```text
17D occupancy logits
256D BEV latent
native class
entropy
margin
source-valid
native grid coordinate
hard FREE/OCC/contradiction
temporal support
stratum
scene role
```

## 9.3 多 GPU

Sidecar extraction：

```text
1 GPU worker / scene
```

允许：

- 1×3090：串行；
- 2–4×3090：scene-level 并行。

禁止同一 scene 多 worker 重复写同一 output。

## 9.4 Gate

- 全部 cohort sidecar 完整；
- finite；
- no prototype；
- no role leakage；
- native mapping round-trip；
- hard evidence alignment；
- fresh process reload；
- per-scene denominator 完整；
- source checkpoint 不变。

---

# 10. P3 — Faithful Native UQ

Task：

```text
WS-V64-P3-NATIVE-UQ-01
```

这是第一个正式方法阶段。

---

## 10.1 U0 Baseline

比较：

```text
max probability
entropy
top1-top2 margin
Native B2 UNKNOWN probability
```

所有 baseline 不训练。

---

## 10.2 U1 Aleatoric

只训练 aleatoric head。

输入：

```text
native logits + native feature
```

target：

- observed contradiction；
- sensor corruption；
- structured evidence dropout；
- label ambiguity。

不训练 state head。

---

## 10.3 U2 Epistemic GMM

只使用 D-Train。

步骤：

1. train-only feature projection；
2. geometry-level GMM；
3. semantic GMM（sample sufficient 时）；
4. fit；
5. frozen inference。

不使用 D-Selection 选择 GMM component。

Component selection：

```text
仅在 D-Train 内部 nested split / BIC
```

---

## 10.4 U3 Hybrid

训练 monotone fusion：

```text
u_ale
u_epi
entropy
margin
source support
```

模型容量必须低。

---

## 10.5 Corruption / OOD Protocol

在 D-Selection 之前冻结：

- missing camera；
- fog / photometric degradation；
- temporal dropout；
- calibration perturbation；
- unseen scene / weather；
- actor-density shift。

不改变 method target truth。

---

## 10.6 P3 Metrics

### Hidden-FREE

- AUROC；
- AUPRC；
- FPR@95TPR；
- risk-coverage；
- expected calibration error；
- Brier。

### OOD / corruption

- mAUROC；
- corruption severity response；
- missing-camera response。

### Non-redundancy

- aleatoric / epistemic Spearman；
- per-corruption ranking；
- conditional histograms。

---

## 10.7 P3 Gate

U3 必须同时满足：

```text
Hidden-FREE AUROC:
>= best U0 + 0.05 absolute

Hidden-FREE AUPRC:
>= best U0 + 0.03 absolute

scene support:
>= 3/4 D-Selection scenes

corruption:
至少 2 类 corruption 显著优于 entropy

hard violations:
0
```

若 U1/U2 之一无信号：

- 允许 U3 使用有效分支；
- 但论文 claim 对应降级；
- 不允许伪造 hybrid 贡献。

若 U3 不优于简单 baseline：

```text
关闭 frozen-UQ family
```

不得继续 P4。

---

# 11. P4 — Conditional UNC-Compiler

Task：

```text
WS-V64-P4-CONDITIONAL-COMPILER-01
```

P3 通过后解锁。

---

## 11.1 Arms

### C0

```text
Native B2
+
U3
+
global primal-dual constraints
```

### M0

```text
Native B2
+
U3
+
global constraints
+
scene-robust objective
+
stratum conditional constraints
```

架构、数据、seed、budget 相同。

只改变 conditional optimization。

---

## 11.2 Constraints

Global：

```text
retention >= 0.70
emitted OCC coverage >= 0.20
non-UNKNOWN >= 0.50
hard = 0
```

Per-scene：

```text
retention >= 0.60
emitted area >= 0.60 × B2 scene area
UNKNOWN <= 0.60
```

Per-stratum：

```text
retention >= 0.60
nonzero emission
risk <= frozen target
```

精确门在 P1 根据 D cohort 结构冻结，不得低于上述下界。

---

## 11.3 Risk

Primary：

```text
case-level hidden-FREE conflict surrogate
```

Secondary：

- pooled voxel risk；
- accuracy；
- mIoU；
- semantic metrics。

candidate selection：

1. hard=0；
2. global gates；
3. scene gates；
4. stratum gates；
5. worst-scene risk；
6. pooled risk；
7. accuracy。

---

## 11.4 P4 Gate

M0 相对 B2：

```text
worst-scene risk 降低 >=10%
support >=3/4 scenes
任何 scene 风险不得恶化 >5%
pooled emitted area >=0.80 × B2
每 scene emitted area >=0.60 × B2
retention >=0.70
UNKNOWN <=0.50 pooled
UNKNOWN <=0.60 per scene
actor/static 均非零
hard=0
```

并且：

```text
M0 至少在 2/4 scenes 优于 C0
```

若 C0 有效、M0 无增量：

- 关闭 conditional method；
- 保留 global UNC baseline；
- 不强行进入 LoRA。

---

# 12. P5 — Targeted LoRA / PEFT（条件解锁）

Task：

```text
WS-V64-P5-IRWM-LORA-01
```

不是默认必跑。

---

## 12.1 解锁条件

同时满足：

1. P3 UQ 有效；
2. P4 train/global 指标有效；
3. P4 主要失败来自 scene-disjoint gap；
4. frozen native feature 的 UQ separation 在 train 明显高于 selection；
5. 不是 hard constraint / data leakage / calibration 问题。

---

## 12.2 Target Modules

先做源码级 module audit。

默认冻结：

```text
image backbone
early scene / BEV encoder
```

LoRA：

```text
autoregressive residual world predictor
temporal attention Q/K/V/O
feature-alignment attention / projection
world decoder attention
```

完全可训练：

```text
occupancy decoder final block
aleatoric head
hybrid UQ head
conditional authority head
```

禁止：

```text
planning head
full visual backbone
全部 linear 层
```

---

## 12.3 LoRA Contract

第一正式配置：

```text
rank = 16
alpha = 32
dropout = 0
seed = 0
```

不 sweep rank。

必须保存：

```text
selected module full paths
module kind
trainable parameter count
non-LoRA trainable whitelist
adapter-only checkpoint
base checkpoint identity
```

复用项目已有 LoRA fail-closed 经验：

```text
未匹配 module
→ fail

出现非白名单 trainable tensor
→ fail

adapter module 与 manifest 不一致
→ fail
```

---

## 12.4 Single-GPU Memory Probe

默认先在 1×3090：

```text
forward
backward
optimizer step
```

记录：

- model memory；
- activation peak；
- optimizer peak；
- iteration time；
- cgroup memory；
- disk。

允许一次有界恢复：

- FP16/BF16；
- activation checkpointing；
- microbatch=1；
- gradient accumulation；
- frozen-prefix no-grad；
- tensor lifetime 优化。

禁止：

- 降分辨率；
- 缩 ROI；
- 减时序；
- 丢 native feature；
- 减 scene denominator。

---

## 12.5 多卡策略

### 单 rank 能放入 24GB，仅吞吐慢

```text
2–4×3090 DDP
```

### 单 rank 不能放入 24GB

一次有界恢复后仍 OOM：

```text
2–4×3090 FSDP / ZeRO-3 capability probe
```

如果仍无法 faithful：

```text
blocked_resource
释放 GPU
保存 run
向用户申请其他资源
```

---

## 12.6 LoRA Gate

L0 相对 M0 frozen：

```text
worst-scene risk 再降低 >=5%
支持 >=3/4 scenes
scene generalization gap 缩小 >=20% relative
anti-trivial gates 全过
hard=0
原 IR-WM forecasting 不发生灾难性回退
```

原能力 preservation：

```text
occupancy forecasting metric 回退 <=5%
planning head 仅作 frozen diagnostic
```

若 LoRA 只改善 train、不改善 selection：

```text
关闭 V6.4 PEFT family
```

不做 rank / seed / scope sweep。

---

# 13. P6 — 独立 Case-Level Calibration

Task：

```text
WS-V64-P6-CALIBRATION-01
```

使用 Tier C：

```text
6 scenes
72 case units
```

模型、score、fusion、group policy 已冻结。

---

## 13.1 Calibration Unit

```text
case / target unit
```

不是：

```text
voxel
point
ray
patch
```

---

## 13.2 Calibration Loss

定义：

\[
\ell_\lambda(c)
=
\mathbf 1[
ACCEPT_\lambda(c)
\land
FREEConflict(c)>\tau_F
]
\]

Primary：

```text
False-safe case loss
```

Secondary：

- continuous conflict；
- retention；
- coverage；
- prediction-set size。

---

## 13.3 Hierarchical Set

Level 1：

```text
FREE / OCCUPIED / UNKNOWN
```

Level 2：

```text
semantic class（可选）
```

第一报告以 Level 1 为主。

---

## 13.4 Target Risk

```text
epsilon = 0.05
confidence = 0.95
```

输出：

- empirical risk；
- finite-sample upper bound；
- selected policy；
- coverage；
- prediction-set size；
- scene / stratum distribution。

如果 exchangeability 明显不成立：

- 只允许预注册 non-exchangeable weighted CRC；
- 权重来源必须 metadata-only；
- 不得按质量结果拟权重。

---

## 13.5 Calibration Gate

```text
risk upper bound <=0.05
coverage >0
safe-OCC retention >=0.60
UNKNOWN <=0.60
>=4/6 calibration scenes 有 emission
actor/static 均非零
```

如果只能 all-UNKNOWN 通过：

```text
rejected
```

---

# 14. P7 — Legacy Retrospective

Task：

```text
WS-V64-P7-LEGACY28-RETROSPECTIVE-01
```

只在 candidate + calibration policy 冻结后读取。

目的：

- 和 V6.1–V6.3 同 denominator 比较；
- 不用于调参；
- 不作为 fresh generalization claim。

Gate：

```text
ACCEPT >=5/28
false-safe = 0
R10 3/3 retained
至少新增 1 actor
至少新增 1 static/disocclusion
accepted mask area >=12%
worst accepted FREE conflict <=0.05
retention >=0.60
UNKNOWN <=0.60
```

失败：

```text
不回调 calibration
不改 model
不读 H/T
family closed
```

---

# 15. P8 — One-Shot Confirmation

Task：

```text
WS-V64-P8-CONFIRMATION-01
```

Tier H：

```text
3 scenes
36 units
```

Attempt：

- quality read 前 exclusive-create；
- 任何失败消费 attempt；
- 不重跑同一 candidate。

通过：

```text
empirical false-safe = 0
risk report valid
>=2/3 scenes 有合法 coverage
retention / UNKNOWN 过门
actor/static 非退化
```

---

# 16. P9 — Exact-Once Test

Task：

```text
WS-V64-P9-EXACT-ONCE-TEST-01
```

Tier T：

```text
4 scenes
48 units
```

要求：

- exact candidate；
- exact calibration policy；
- attempt before quality；
- 全 denominator；
- failed/UNKNOWN 全报告；
- 不更改 candidate。

只有通过 P9 才能称：

```text
UNC-Compiler confirmed
```

---

# 17. P10 — GS + LogSim / WorldSim

P9 通过后解锁。

---

## 17.1 GS + LogSim

目标：

- case 精确复现；
- collision/event 一致；
- physical state 不篡改原因果；
- deterministic replay。

指标：

```text
state exactness
collision label exactness
sensor replay
perception replay
case interception rate
false-safe
```

---

## 17.2 GS + WorldSim

目标：

- route deviation；
- actor insert/remove；
- trajectory edit；
- disocclusion；
- multi-actor composition。

指标：

```text
safe valid yield
false-safe
verified world area
verified route length
actor insertion yield
abstention
```

---

# 18. P11 — GS + NWM Collision Critic / RL

只有 P9 + P10 通过才解锁。

第一阶段不训练大型 NWM。

三臂：

```text
Real-only
Real + naive generated
Real + UNC-Compiler verified
```

主指标：

- collision false-safe；
- unsafe-action recall；
- safe-action precision；
- critic calibration。

次指标：

- route progress；
- stuck；
- comfort；
- total reward。

防作弊：

```text
全刹车
→ completion/stuck gate 拒绝
```

若 Real-only 与 V6.4 一样好：

```text
如实报告无增量
```

---

# 19. 资源合同

## 19.1 默认

```text
现有 RTX 3090 服务器
1 GPU 开发
1–4 GPU 正式
```

不迁移机器作为前置。

---

## 19.2 Stage 资源建议

| Stage | 默认 |
|---|---|
| P0–P1 | CPU |
| P2 Sidecar | 1–4×3090，scene parallel |
| P3 UQ | 1×3090；GMM 可 CPU/GPU |
| P4 Compiler | 1×3090，必要时 2×DDP |
| P5 LoRA | 1×3090 probe；2–4×3090 formal |
| P6–P9 Eval | 1–4×3090 scene parallel |
| P10 | 1–2×3090 |
| P11 | 视策略规模申请 |

---

## 19.3 Resource Stop

以下才算资源阻塞：

- faithful minimum 两次 OOM；
- 有界恢复后仍 OOM；
- FSDP/ZeRO capability 失败；
- 单 stage wall >24h 且加卡不可得；
- disk 安全线不足；
- NCCL/P2P 不可用且必须多卡。

Codex 必须：

1. 保留 failed run；
2. 写 `blocked_resource`；
3. 更新 failure ledger；
4. 释放 GPU；
5. 停止 GPU 分支；
6. 向用户说明所需卡数。

不得把资源 blocked 写成算法 rejected。

---

# 20. Auto Research Loop

每个 hypothesis：

```text
OBSERVE
→ DIAGNOSE
→ LITERATURE / FAILURE GATE
→ PREREGISTER
→ IMPLEMENT
→ FORMAL RUN
→ AUDIT
→ REFLECT
→ PROMOTE / REJECT
```

---

## 20.1 允许自主执行

- 普通工程修复；
- Git branch / commit / push；
- scene-level 多卡调度；
- UQ capability probe；
- 单个正式 hypothesis；
- 文档更新；
- failure ledger；
- 新 hypothesis；
- 资源 probe；
- blocked 后继续不依赖分支。

---

## 20.2 不允许

- seed sweep；
- rank sweep；
- GMM component selection on validation；
- threshold sweep；
- 偷读 calibration/H/T；
- pooled pass 覆盖 scene failure；
- 回开 surface family；
- 降 gate；
- 删除失败 scene；
- 用 O_eval 训练 inference score；
- full IR-WM unfreeze；
- 多个模块一次性同时改变。

---

# 21. Stop Rules

## Stop 1 — UQ 无信号

若 U3 不优于 entropy / margin：

```text
关闭 frozen-UQ family
```

不进入 compiler。

## Stop 2 — Conditional 无增量

若 M0 不优于 C0，或仍 0/4 scene support：

```text
关闭 conditional method
```

## Stop 3 — Frozen representation ceiling

只有 P3 有 UQ 信号、P4 train 有效但 scene selection gap 明确，才进入 LoRA。

## Stop 4 — LoRA 无泛化

LoRA 只改善 train、不改善 selection：

```text
关闭 PEFT family
```

不换 rank / seed。

## Stop 5 — Calibration all-UNKNOWN

```text
rejected
```

## Stop 6 — Legacy false-safe

不解锁 confirmation。

## Stop 7 — Confirmation false-safe

family closed。

## Stop 8 — Test failure

candidate rejected；不重跑。

---

# 22. 建议代码结构

```text
docs/WORLDSIM_V6_4_UNCERTAINTY_NATIVE_CONDITIONAL_COMPILER_PLAN.md

docs/autoresearch/worldsim_v64/
  AUTORESEARCH_STATE.current.json
  HYPOTHESES.jsonl
  REFLECTIONS.jsonl
  P0_SCOPE.md
  P1_NOVELTY_LICENSE.md
  P2_COHORT.md
  ...

configs/worldsim_v64/
  p0_scope_v1.yaml
  p1_protocol_v1.yaml
  p2_native_sidecars_v1.yaml
  p3_native_uq_v1.yaml
  p4_conditional_compiler_v1.yaml
  p5_irwm_lora_v1.yaml
  p6_calibration_v1.yaml
  p7_legacy28_v1.yaml
  p8_confirmation_v1.yaml
  p9_test_v1.yaml

motion_proj/worldsim_v64/
  cohort.py
  evidence_roles.py
  aleatoric.py
  epistemic_gmm.py
  hybrid_uq.py
  conditional_constraints.py
  uncertainty_compiler.py
  calibration.py
  irwm_lora.py

scripts/
  run_worldsim_v64_p0_scope.py
  run_worldsim_v64_p2_sidecars.py
  run_worldsim_v64_p3_native_uq.py
  run_worldsim_v64_p4_conditional_compiler.py
  run_worldsim_v64_p5_irwm_lora.py
  run_worldsim_v64_p6_calibration.py
  run_worldsim_v64_p7_legacy.py
  run_worldsim_v64_p8_confirmation.py
  run_worldsim_v64_p9_test.py

tests/worldsim_v64/
```

禁止创建第二套 failure ledger。

---

# 23. Run Contract

唯一 run ID：

```text
<UTC>__<task-slug>__s<seed>__r<nnn>
```

最小：

```text
resolved.yaml
status.json
events.jsonl
metrics.jsonl
summary.json
stdout.log
stderr.log
resource.json
```

UQ：

```text
UQ_CASES.jsonl
UQ_SCENE_METRICS.jsonl
CORRUPTION_METRICS.jsonl
GMM_MODEL/
```

Compiler：

```text
CONDITIONAL_METRICS.jsonl
SCENE_GATES.jsonl
STRATUM_GATES.jsonl
CANDIDATE.json
```

Calibration：

```text
CALIBRATION_CASES.jsonl
RISK_CURVE.jsonl
CALIBRATION_POLICY.json
FINITE_SAMPLE_REPORT.json
```

Formal run immutable。

---

# 24. Git 纪律

- 正式实验前 prereg commit；
- source commit 写入 run；
- result closeout 单独 commit；
- rejected 也提交；
- 定期普通 push；
- 不 force；
- 不提交 dataset / checkpoint / run output；
- 不覆盖历史 plan；
- 路径运行时解析，不在新 config 写死 AutoDL 绝对路径。

本机 capability：

```text
.local/worldsim_v64/capabilities.local.yaml
```

加入 `.gitignore`。

---

# 25. 第一轮执行顺序

```text
P0  V6.3 inheritance + main + V6.4 branch
↓
P1  novelty / license / method / cohort / gates freeze
↓
P2  fresh cohort + native sidecars
↓
P3  U0/U1/U2/U3 native uncertainty
↓
     FAIL → close frozen-UQ family
     PASS
↓
P4  C0 global vs M0 conditional compiler
↓
     representation ceiling?
     ├── NO → P6 calibration
     └── YES → P5 targeted LoRA
↓
P6  independent case-level calibration
↓
P7  legacy28 retrospective
↓
P8  one-shot confirmation
↓
P9  exact-once test
↓
P10 LogSim / WorldSim
↓
P11 optional NWM collision critic / RL
```

---

# 26. 对 Codex Agent 的最终执行指令

你现在是 WorldSim V6.4 Autonomous Research Agent。

目标不是：

```text
提高 pooled accuracy
降低平均 entropy
让更多 voxel 输出 OCC
```

目标是：

> **在原生 IR-WM 特征上学习能够区分数据噪声与分布移位的 uncertainty，并通过 scene / stratum 条件风险约束与独立 case-level calibration，只将具有可靠物理 authority 的 OCC 状态写入 SceneIR。**

执行纪律：

1. 完整读取 V6.3 docs 与 `V63-F24`；
2. 保持 V6.3 terminal，不回开 Surface；
3. Native B2 是强基线；
4. OCCUQ 先 faithful mechanism migration；
5. EvOcc / ReliOcc / α-OCC 没有完整官方代码时只迁移论文机制；
6. 先 UQ，再 conditional compiler，再决定 LoRA；
7. 不把所有模块一次堆入；
8. 默认现有 3090；
9. 模型能单卡放下时，多卡只解决吞吐；
10. LoRA 只在 frozen representation ceiling 被证据支持后解锁；
11. 不 full unfreeze IR-WM；
12. 不用 pooled gate 掩盖 scene failure；
13. calibration 以 case 为单位；
14. 不用 all-UNKNOWN 制造安全；
15. confirmation/test exact-once；
16. 每个 failure 更新统一 ledger；
17. 普通工程失败自主恢复；
18. 真实资源阻塞时停止 GPU 分支并向用户申请卡；
19. 达到 stop rule 即关闭，不做隐性 sweep；
20. 只有 P9 通过后才能称 V6.4 confirmed。

---

# 27. 一句话研究主张

英文：

> **We turn native occupancy features into task-eligible physical state by disentangling aleatoric and epistemic uncertainty, enforcing scene- and failure-stratum-conditional risk constraints, and calibrating case-level set-valued decisions before any occupied state is baked into a driving world.**

中文：

> **我们在原生 Occupancy 特征中分离数据不确定性与认知不确定性，以 scene / failure stratum 条件风险约束训练物理状态 authority，并在写入驾驶世界前对 case-level 集合式决策进行独立校准。**

---

# 28. 预期论文贡献

若全部阶段通过：

1. **Native Hybrid Uncertainty for Physical-State Compilation**
   - 原生 Occupancy feature 上的 aleatoric + epistemic UQ。
2. **Scene / Stratum Conditional Authority**
   - 从 pooled feasibility 升级为条件可靠性。
3. **Case-Level Calibrated World Admission**
   - 不是 voxel-level 虚假样本量。
4. **Targeted Foundation Adaptation**
   - 仅在 frozen representation ceiling 时使用 LoRA。
5. **Three-Task World Compiler**
   - LogSim / WorldSim / NWM collision critic。

如果 LoRA 未执行：

- 不主张 PEFT contribution；
- 前三项仍可独立形成技术报告。

---

# 29. 主要参考

- OCCUQ：`https://github.com/ika-rwth-aachen/OCCUQ`
- EvOcc：`https://openaccess.thecvf.com/content/CVPR2025/html/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.html`
- ReliOcc：`https://www.ijcai.org/proceedings/2025/220`
- α-OCC：`https://arxiv.org/abs/2406.11021`
- Conformal Risk Control：`https://openreview.net/forum?id=33XGfHLtZg`
- Non-Exchangeable Conformal Risk Control：`https://openreview.net/forum?id=j511LaqEeP`
- Conditional Robust Optimization：`https://proceedings.mlr.press/v244/chenreddy24a.html`
- IR-WM：`https://github.com/APRIL-ZJU/IR-WM`
- DriveDiTFit：`https://github.com/TtuHamg/DriveDiTFit`
