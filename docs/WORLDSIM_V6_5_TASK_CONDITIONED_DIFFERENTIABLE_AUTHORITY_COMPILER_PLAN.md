# WorldSim V6.5 — 任务条件可微物理权限编译器计划

> 英文工作名：**TAC-Compiler — Task-Conditioned Differentiable Authority Compiler**  
> 推荐技术报告题目：**TAC-Compiler: Learning Task-Conditioned Physical Authority for Verifiable Driving World Compilation**  
> 中文题目：**TAC-Compiler：面向可验证驾驶世界编译的任务条件物理权限学习**  
> 状态：`active`（2026-08-27 从 `add2f3f` 启动；按用户指令压缩 P0、直接进入 research）  
> 上游冻结分支：`research/worldsim-v6.4-native-uq`  
> 上游终态提交：以远端 `add2f3f` 为准  
> 默认资源：现有 AutoDL，单张 RTX 3090 优先；多卡不是前置条件

---

# 0. 版本定位

V6.5 不是重新做一套 Occupancy，也不是把 V6.4 的 MLP 换成更大的 Transformer。

V6.4 已经建立三条事实：

1. `q(x)` 式原生特征风险分数可以支持非平凡的选择性物理状态写入；
2. 按场景分层和路线条件改变写入策略，可以在固定机会分母下进一步降低路线局部冲突；
3. 世界状态可靠性不会自动转化为动作级碰撞可靠性，V6.4 的线性 collision critic 出现跨 cohort 排序退化。

V6.4 最强实现仍含明显硬编码：

```text
273D native feature
→ pointwise MLP risk q(x)
→ stratum → {0.40, 0.50} coverage lookup
→ fixed 2 s / 1.5 m route corridor
→ route coverage cap = 0.40
→ risk sort + non-route budget reallocation
→ deterministic bake
```

V6.5 的核心问题是：

> **能否把 V6.4 的“分层查表、路线硬掩码、固定覆盖上限和排序搬运”逐步替换为任务、轨迹、Actor 与上下文条件下的可微物理权限模型，同时保持独立校准与确定性 Runtime？**

V6.5 不预先押注一个最终数学模型。它只冻结：

- 问题分解；
- 数据角色；
- 对照基线；
- 每阶段最多允许测试的架构数量；
- 阶段解锁条件；
- Stop Rule；
- 最终端到端融合的资格。

具体网络采用**渐进式预注册**：前一阶段给出证据后，才冻结下一阶段的单个正式实现。

---

# 1. 顶会方法调研后的迁移结论

## 1.1 SafeDrive — CVPR 2026 Highlight

可迁移机制：

- 为每条候选轨迹构造 trajectory-conditioned sparse world；
- 显式建立 Ego–Actor pair-wise、time-wise 风险；
- 先感知预训练，再冻结感知训练规划/安全头，最后端到端微调。

V6.5 使用：

- 轨迹条件化世界权限；
- 稀疏 Actor token；
- Actor × time 风险分解；
- 分阶段训练。

V6.5 不直接复制：

- 概率连乘；
- weighted log-sum 总评分；
- 从第一天就训练完整 E2E planner。

原因：V6.3 已经证明普通加权目标可能产生 authority collapse，聚合形式必须由增量实验决定。

## 1.2 WoTE — ICCV 2025

可迁移机制：

- 用轨迹编码形成 state–action pair；
- 轨迹 query 与 BEV state 做 cross-attention；
- 用 action-conditioned future state 辅助轨迹评价。

V6.5 使用：

- 先验证 trajectory token 是否给 `q(x)` 增量；
- 若有信号，再验证小型局部 future/context 模块。

V6.5 不直接复制完整 recurrent BEV world model。只有静态任务条件化出现明确上限后，才允许研究短时 future latent。

## 1.3 World4Drive — ICCV 2025

可迁移机制：

- intention-aware latent world；
- 由 driving intention 条件化多模态轨迹生成和选择。

V6.5 使用：

- 将 route / trajectory / task intent 视为世界有效性查询的一部分；
- 不再把世界状态是否可靠定义成与下游任务无关的单标量。

V6.5 不替换 IR-WM foundation，也不第一轮训练新的视觉世界模型。

## 1.4 UniAD / VAD / SparseDrive

共同启示：

- 任务之间存在层级依赖，先稳定上游再联合优化；
- Actor、地图、轨迹适合使用稀疏或向量化 token，而不是把所有信息重新栅格化；
- 规划约束与碰撞重排可做成网络可学习、仍保持显式接口的模块。

V6.5 使用：

- 原生 voxel/BEV 表示保留；
- trajectory / actor / map 使用稀疏 token；
- 先模块化验证，最后才联合反传。

## 1.5 DiffStack

可迁移机制：

- 保持模块化、可解释的栈；
- 通过可微规划/选择模块把下游梯度传回上游。

V6.5 使用：

> 最终目标不是“把所有东西塞进一个黑盒”，而是构建**可微但仍可审计的编译栈**。

## 1.6 SelectiveNet、SOFT Top-k、GroupDRO、DSelect-k

可迁移机制：

- SelectiveNet：将拒绝/覆盖头从后处理阈值升级为联合学习模块；
- SOFT Top-k：为固定预算选择提供可微近似；
- GroupDRO：优化最差分组，而非只优化 pooled 风险；
- DSelect-k：在确有多模态上下文时使用可微稀疏专家路由。

V6.5 使用顺序：

```text
learned threshold / gate
→ constrained soft admission
→ 只有发现 hard top-k 梯度瓶颈，才用 SOFT Top-k
→ 只有连续上下文仍存在清晰多峰，才允许小型 MoE
```

禁止一开始直接堆：

```text
cross-attention + MoE + SOFT Top-k + GroupDRO + joint IR-WM
```

## 1.7 Conformal Risk Control

继续只用于独立 calibration / admission policy 冻结。

- calibration unit 仍是 case 或 scene block；
- 不把 voxel 当独立样本；
- calibration 数据不参与模型训练；
- 不因模型可微而把最终校准集并入端到端训练。

---

# 2. V6.5 数学表示：冻结“语义接口”，不冻结最终参数化

## 2.1 Task Contract

令一个编译任务查询为：

\[
\kappa_t =
(\tau^{ego}_{0:H},\; \mathcal A_{0:H},\; \mathcal M_t,\; h,\; m_{task})
\]

其中：

- \(\tau^{ego}_{0:H}\)：Ego 候选路线或轨迹；
- \(\mathcal A_{0:H}\)：Actor 状态、轨迹或 swept-envelope token 集合；
- \(\mathcal M_t\)：静态地图、可行驶区域或场景上下文；
- \(h\)：任务时间范围；
- \(m_{task}\)：任务类型，例如 general bake、route validity、actor insertion、action evaluation。

这里的 \(m_{task}\) 不是 scene ID，也不是人工 stratum lookup。

## 2.2 Candidate Unit

对候选物理状态单元：

\[
u=(x,y,z,t)
\]

保留 V6.4 原生输入：

\[
f_u = [l_u^{IRWM}, z_u^{BEV}, e_u^{hard}, s_u^{support}]
\]

新增条件表示：

\[
z_u = E_v(f_u),\quad
z_\tau = E_\tau(\tau),\quad
Z_A = E_A(\mathcal A),\quad
z_M=E_M(\mathcal M)
\]

## 2.3 输出接口

V6.5 的网络至少输出三类量：

\[
\mathcal V_\theta(u\mid\kappa)=
(r_u^{phys},\; r_u^{task},\; v_u^{task})
\]

以及一个软权限：

\[
a_u = A_\phi(u\mid\kappa)\in[0,1]
\]

语义为：

- \(r_u^{phys}\)：与具体任务无关的基础物理冲突风险；
- \(r_u^{task}\)：在给定轨迹/Actor/任务条件下的增量风险；
- \(v_u^{task}\)：该单元对当前任务的价值或相关性；
- \(a_u\)：compile-time 的软写入权限。

最终 Runtime 不保存模糊网络决策。独立校准后固化：

```text
FREE / OCCUPIED / UNKNOWN
+ provenance
+ task validity mask / contract
```

## 2.4 优化包络

V6.5 只冻结如下约束语义：

\[
\max_{a}\sum_u a_u v_u^{task}
\]

满足：

\[
R_{phys}(a)\le \epsilon_{phys},
\quad
R_{task}(a,\kappa)\le \epsilon_{task},
\quad
C_{min}\le \sum_u a_u\le C_{max},
\quad
HardViolation=0
\]

但不在 P0 就冻结求解器。

候选实现必须按顺序验证：

1. 连续阈值/门控；
2. primal-dual soft admission；
3. 可微 Top-k / 最优传输预算；
4. 稀疏专家路由。

前一个实现没有证明瓶颈时，后一个不得解锁。

---

# 3. 建议网络架构

```text
              Frozen IR-WM Native State
        17D logits + 256D BEV + evidence/support
                         │
                   Voxel Encoder
                         │ z_u
       ┌─────────────────┼──────────────────┐
       │                 │                  │
Trajectory Encoder   Actor Set Encoder   Map/Context Encoder
       │ z_τ             │ Z_A              │ z_M
       └─────────────────┴──────────────────┘
                         │
          Task-Conditioned Fusion Module
        FiLM/residual → cross-attention（条件解锁）
                         │
          ┌──────────────┴──────────────┐
          │                             │
   Physical/Task Risk Head        Task Utility Head
          │                             │
          └──────────────┬──────────────┘
                         │
        Differentiable Admission / Budget Module
     gate → constrained gate → soft top-k（条件解锁）
                         │
                  Soft Physical State
                         │
           Independent Case-Level Calibration
                         │
        Deterministic SceneIR Bake + Provenance
```

## 3.1 第一正式容量原则

- Voxel trunk 优先复用 V6.4 `273→128→64` MLP 权重；
- trajectory / actor token 默认维度不超过 64；
- 首个正式 attention 不超过 2 层；
- Actor token 数通过 metadata/capability 冻结，不根据质量挑选；
- 不使用 scene ID embedding；
- 不使用大 Transformer 作为第一 arm。

具体维度只在 train-only memory/capability probe 后冻结，不做 selection sweep。

---

# 4. 研究组织：渐进式预注册，而不是一次性锁死最终模型

## 4.1 Master Preregistration 冻结内容

在任何 V6.5 quality read 前冻结：

- 数据角色和 scene ledger；
- V6.4 immutable baselines；
- 六个核心问题；
- 每个问题最多允许的正式 arm 数；
- primary metrics；
- selection exposure budget；
- confirmation/test exact-once；
- Stop Rules；
- 资源合同。

## 4.2 Stage Preregistration

每个阶段只在上一阶段结论出来后，使用：

- train-only diagnostics；
- 源码/显存 capability；
- 上一阶段失败结构；

冻结下一阶段的具体网络。

不允许：

- 在 selection 结果后改同一 arm；
- 把“new hypothesis”作为无限消费同一 selection 的借口；
- 同一问题测试超过 2 个正式架构；
- 失败后换 seed、hidden size 或注意力层数救结果。

---

# 5. 数据纪律

## 5.1 V6.4 数据角色

V6.4 所有读取过质量的 scene：

```text
Tier L — legacy / mechanism / regression only
```

允许：

- 代码回归；
- V6.4 baseline replay；
- train-only warm-start；
- 失败可视化；
- 技术报告回溯。

禁止：

- V6.5 architecture selection；
- threshold / gate / expert selection；
- independent calibration；
- confirmation/test。

## 5.2 推荐 Fresh Cohort

```text
D-Train                 16 scenes × 12 = 192 units
D-Selection-Representation 6 × 12 = 72 units
D-Selection-Admission      6 × 12 = 72 units
C-Calibration            8 × 12 = 96 units
H-Confirmation           8 × 12 = 96 units
T-Exact-Test             8 × 12 = 96 units
------------------------------------------------
Total                   52 fresh scenes / 624 units
```

如果 untouched scene 不足：

- 在任何 quality read 前冻结最大可用 cohort；
- Calibration、Confirmation、Test 均不得低于 8 scenes / 96 units；
- 不得事后减 denominator；
- 不得用 V6.4 holdout 补齐。

## 5.3 Metadata-Only Selection

允许：

- weather、time-of-day；
- actor count / motion metadata；
- map availability；
- route curvature；
- frame count；
- sensor completeness；
- location；
- road topology metadata。

禁止：

- V6.4/V6.5 risk score；
- hidden-FREE conflict；
- action labels；
- model quality；
- emitted coverage；
- rendered quality。

## 5.4 分组只用于训练和审计

night/rain/construction/vulnerable 等组仍保留用于：

- GroupDRO；
- worst-group gate；
- 报告支持度。

但不允许作为 inference lookup key。网络必须从连续可观测 context 学习。

---

# 6. Baseline Staircase

```text
B0  V6.4 full-native pointwise MLP q(x)
B1  V6.4 C0 global 40%
B2  V6.4 M0 stratum lookup
B3  V6.4 M1 hard route cap + budget reallocation

T0  Continuous trajectory-feature residual probe
T1  Learned trajectory-conditioned risk
A0  Actor-pooled context
A1  Pair-wise / time-wise Actor interaction risk

G0  Learned context threshold
G1  Per-unit differentiable admission gate
G2  Group-robust constrained admission

S0  Hard rank selection on learned gate
S1  Differentiable budget allocator（条件解锁）
E0  Sparse expert routing（条件解锁）

J0  Best modular compiler
J1  Joint risk + admission training
J2  Targeted IR-WM PEFT（条件解锁）
```

每一层只回答一个问题。

---

# 7. P0 — V6.4 继承与协议冻结

Task：

```text
WS-V65-P0-INHERITANCE-PROTOCOL-01
```

工作：

1. 从 `research/worldsim-v6.4-native-uq` 最新终态创建：
   `research/worldsim-v6.5-task-conditioned-authority`；
2. 保持 V6.4 terminal，不改任何 canonical run；
3. 建立 `USED_SCENE_LEDGER_V65.json`；
4. 建立 `SELECTION_EXPOSURE_LEDGER.json`；
5. 完成 source/license/claim matrix；
6. 冻结 fresh cohort、trajectory generation contract、action/evidence roles；
7. 复用 V6.4 risk checkpoint 和 sidecar pipeline；
8. 不在 P0 训练新模型。

P0 输出：

```text
docs/WORLDSIM_V6_5_TASK_CONDITIONED_DIFFERENTIABLE_AUTHORITY_COMPILER_PLAN.md
docs/autoresearch/worldsim_v65/P0_PROTOCOL.md
configs/worldsim_v65/p0_protocol_v1.yaml
```

---

# 8. P1 — Signal Atlas：先判断“缺什么条件信息”

Task：

```text
WS-V65-P1-CONDITION-SIGNAL-ATLAS-01
```

这是避免“拿锤子找钉子”的关键阶段。

只在 D-Train 内部做 nested train-only probe，不读取正式 selection。

## 8.1 Probe Families

### R0：V6.4 baseline

```text
q0(x) = V6.4 full-native MLP
```

### R1：连续 trajectory geometry

候选特征：

- 到候选轨迹的连续距离；
- 轨迹切向/相对朝向；
- 沿轨迹弧长位置；
- 最近相交时间；
- task horizon；
- route relevance soft weight。

禁止使用硬 1.5 m corridor 标签作为 inference feature。

### R2：Actor proximity / time

- Ego/trajectory 与 Actor 的相对位置、速度、朝向；
- time-to-closest-approach；
- Actor swept-envelope 支持；
- temporal support / lifecycle。

### R3：Map / context

- drivable support；
- route curvature；
- static boundary support；
- sensor support；
- continuous environment context。

## 8.2 需要回答

1. trajectory 条件是否在 within-scene 指标上补充 `q0(x)`？
2. actor/time 条件是否只在 actor-dense case 有效？
3. 人工 stratum label 在加入连续 context 后是否仍提供额外信息？
4. 风险信号主要发生在 voxel、actor-time，还是 case level？

## 8.3 P1 输出

```text
V65_SIGNAL_ATLAS.json
V65_FEATURE_INCREMENT_TABLE.md
V65_INTERACTION_TARGET_AUDIT.md
```

P1 不宣称方法成功，只决定 P2 的最小正式架构。

---

# 9. P2 — Trajectory-Conditioned Validity

Task：

```text
WS-V65-P2-TRAJECTORY-CONDITIONED-RISK-01
```

## 9.1 T0：低容量条件残差

先冻结 V6.4 voxel trunk，只学习：

```text
trajectory encoder
+
FiLM / gated residual
+
Δ risk head
```

目标：验证“连续 trajectory condition”本身，而不是注意力容量。

## 9.2 T1：Trajectory Cross-Attention（条件解锁）

只有 T0 有正信号但出现局部表示上限时：

```text
trajectory tokens query local native BEV/voxel tokens
```

第一正式版本：

- 小型 1–2 层；
- 不做 recurrent future world；
- 不引入 Actor；
- 不改 admission。

## 9.3 P2 Primary Gate

相对 B0，在 D-Selection-Representation：

```text
matched total coverage 下：
fixed-opportunity route risk 至少降低 10%
scene support >= 5/6
任何 scene 不得恶化 >5%
hard violations = 0
```

辅助报告：

- within-scene AUROC/AUPRC；
- FPR@95TPR；
- risk-coverage；
- route / non-route 分离；
- trajectory perturbation severity response。

若 T0 无信号：

```text
关闭 trajectory-conditioned family
不执行 T1 / Actor / E2E
```

若 T0 有效且 T1 无增量：保留 T0，关闭 attention expansion。

---

# 10. P3 — Actor / Time Interaction Validity

Task：

```text
WS-V65-P3-ACTOR_TIME_INTERACTION-01
```

P2 通过后解锁。

## 10.1 A0：Actor Set Pooling

```text
trajectory-conditioned voxel
+
附近 Actor token set pooling
```

回答：Actor 信息本身是否增加风险可分性。

## 10.2 A1：Pair-wise / Time-wise Risk（条件解锁）

只有 A0 在 actor-dense case 有信号但整体仍有上限时：

```text
voxel / trajectory token
↔ Actor_i
↔ future timestep_h
→ r(i,h,u | trajectory)
```

聚合形式不在 P0 冻结。

P1 train-only diagnostics 后，仅从以下选择一个正式聚合：

- attention pooling；
- smooth max；
- noisy-OR 类单调聚合。

禁止直接照搬全量概率连乘。

## 10.3 P3 Gate

相对 P2 best：

```text
actor-dense fixed-opportunity risk 降低 >=10%
action/trajectory AUPRC 增量 >=0.03
scene support >=5/6
static case 风险不得恶化 >5%
hard=0
```

若只在 Actor stratum 有效：允许形成条件模块，不要求强行覆盖 static。

---

# 11. P4 — Learned Conditional Admission

Task：

```text
WS-V65-P4-DIFFERENTIABLE-ADMISSION-01
```

目标：替换 V6.4 的 stratum coverage lookup。

## 11.1 G0：Learned Context Threshold

固定 best risk model，学习：

```text
context → coverage / threshold
```

它必须只依赖可观测连续 context，不读取人工 stratum ID。

## 11.2 G1：Per-Unit Admission Gate

参考 integrated reject 思路：

```text
risk representation + task representation
→ a_u ∈ [0,1]
```

训练时使用显式 coverage / risk 约束，不使用普通 weighted sum 随意平衡。

## 11.3 G2：Group-Robust Admission

只有 G1 pooled 有效但某组持续失守时：

- group label 只进入 worst-group loss / multiplier；
- inference 不读取 group ID；
- 保持 per-group emission 非零。

## 11.4 P4 Gate

相对 V6.4 B3/M1，在 D-Selection-Admission：

满足以下二选一增量：

```text
A. 同风险下 total coverage 提高 >=5% absolute
或
B. 同 coverage 下 fixed-opportunity worst-tail risk 降低 >=10%
```

同时：

```text
scene support >=5/6
non-route pooled risk 不得恶化 >5%
任何正式 group 不得零 emission
hard=0
不能 all-UNKNOWN
```

若 learned admission 不优于硬编码 M1：

```text
关闭 differentiable admission family
保留 V6.4 M1
```

不因为“更优雅”而推广。

---

# 12. P5 — Differentiable Budget Allocation（条件解锁）

Task：

```text
WS-V65-P5-SOFT-BUDGET-ALLOCATION-01
```

只有以下证据成立才解锁：

1. G1/G2 已有效；
2. 梯度或边界诊断表明 hard sort / top-k 阻止 gate 学到更好的预算分配；
3. 不是 risk model、数据、calibration 或 metric denominator 问题。

## 12.1 S0：Expected-Coverage Gate

先用连续 gate 和期望 coverage constraint，不做 Top-k。

## 12.2 S1：SOFT Top-k / OT

只有 S0 出现清晰预算不匹配时，迁移可微 Top-k。

必须保持：

- 总写入预算可查询；
- route/non-route 不允许风险搬家；
- bake 时转换成 deterministic exact budget。

## 12.3 E0：小型稀疏专家（极晚期）

只有连续 context gate 在 train/selection 呈现稳定多峰，且单模型产生明显条件冲突时，才允许：

```text
2–4 个小 expert
+
连续可微 sparse router
```

禁止直接引入大规模 MoE 或 scene expert。

## 12.4 P5 Gate

相对 G1/G2 best：

```text
同风险 coverage +3% absolute
或
同 coverage fixed-risk -5%
scene support >=5/6
non-route risk 不恶化
hard=0
```

无增量则关闭 allocator expansion。

---

# 13. P6 — Modular-to-End-to-End Fusion

Task：

```text
WS-V65-P6-MODULAR_JOINT_FUSION-01
```

借鉴顶会阶段式训练，但保持 V6.5 自己的归因合同。

## Phase 1：Representation

```text
freeze IR-WM
freeze V6.4 base risk trunk initially
train trajectory / actor / context residual
```

## Phase 2：Admission

```text
freeze risk representation
train admission / budget module
```

## Phase 3：Modular Joint

```text
unfreeze task-conditioned risk + admission
IR-WM 仍冻结
```

同时记录：

- risk / coverage / utility 梯度范数；
- gradient cosine；
- authority collapse；
- group multiplier；
- emission distribution。

如果出现稳定梯度冲突，只允许一次预注册的 bounded recovery：

```text
PCGrad 或 CAGrad 二选一
```

选择依据只能来自 Phase 3 train-only gradient audit，不能在 selection 上比较多个优化器。

## Phase 4：End-to-End Claim Gate

J1 相对 best modular J0：

```text
同 coverage fixed-risk 再降低 >=5%
或
同风险 coverage 再提高 >=3%
scene support >=5/6
V6.4 global physical risk 回退 <=5%
hard=0
```

若不通过：

> 保留模块化 TAC-Compiler；“端到端融合无增量”是合法结论。

不得为追求 E2E 名义全量解冻 IR-WM。

---

# 14. P7 — Targeted IR-WM PEFT（条件解锁）

Task：

```text
WS-V65-P7-IRWM_TARGETED_PEFT-01
```

不是默认阶段。

同时满足才解锁：

1. task-conditioned model 在 D-Train 有强 signal；
2. modular/joint admission 有效；
3. D-Selection 出现稳定 representation gap；
4. 排除 selector、risk transfer、label、calibration 问题；
5. native feature separation 在 train 明显高于 selection。

优先模块：

- temporal/world blocks；
- residual world predictor；
- feature alignment；
- occupancy decoder final block。

冻结：

- visual backbone；
- early BEV encoder；
- planning head。

第一正式配置继续：

```text
rank=16
alpha=32
dropout=0
seed=0
```

不 sweep。

---

# 15. P8 — Independent Case-Level Calibration

Task：

```text
WS-V65-P8-INDEPENDENT_CALIBRATION-01
```

使用 C-Calibration：8 scenes / 96 cases。

在质量读取前，冻结统计合同：

- case / scene-block 是 calibration unit；
- risk loss；
- task contract；
- deterministic bake policy；
- coverage policy；
- 是否满足 exchangeability；
- finite-sample 报告方式。

不得：

- 把 voxel 当 96×百万样本；
- 用 calibration 继续训练 gate；
- 同时调 task encoder、coverage 和阈值；
- 用 all-UNKNOWN 过门。

建议经验门：

```text
mean coverage >=0.45
case false-safe <=2/96
>=6/8 scenes 有合法 emission
route / non-route 均非零
actor / static 均非零
hard=0
```

如果计划声称高概率风险上界，必须先证明有效独立样本量和统计假设；否则仅报告 exact empirical cohort。

---

# 16. P9 — One-Shot Confirmation

Task：

```text
WS-V65-P9-EXACT_ONCE_CONFIRMATION-01
```

使用 H-Confirmation：8 scenes / 96 cases。

确认：

1. 总物理 authority；
2. route-conditioned authority；
3. actor-dense / static；
4. non-route risk transfer；
5. deterministic bake；
6. task query perturbation consistency。

通过门：

```text
coverage / risk 过 P8 冻结门
fixed-opportunity risk 相对 V6.4 M1 有增量
scene support >=6/8
任何 scene 不发生灾难性回退
hard=0
```

失败即关闭当前 TAC family，不创建新的 confirmation cohort 救结果。

---

# 17. P10 — Action-Level Transfer：先评价表示，再训练 Critic

Task：

```text
WS-V65-P10-ACTION_CONDITIONED_TRANSFER-01
```

这是针对 V6.4 P11 失败的修正。

## 17.1 第一阶段：无新 Critic

先直接检验 task-conditioned validity 对固定 trajectory/action set 的排序：

- action AUROC / AUPRC；
- unsafe ranking；
- pairwise Actor-time recall；
- cross-cohort ranking gap；
- progress/stuck。

若 validity field 本身不具备 action ranking，禁止再训练 critic。

## 17.2 第二阶段：结构化 Action Risk Head（条件解锁）

仅当第一阶段有 action signal：

```text
trajectory token
+
Actor × time interaction token
+
compiled physical state
→ action risk
```

不得再回到 V6.4 的 10 个手工特征线性逻辑回归。

训练必须：

- action labels 只来自 D-Train；
- class-balanced sampler/loss 只在训练内使用；
- action generator 在 quality read 前冻结；
- operating point 使用独立 calibration；
- progress/stuck 防全刹车。

## 17.3 Action Gate

相对 V6.4 critic：

```text
action AUPRC +0.05 absolute
unsafe recall >=0.75
mean progress >=0.80
stuck <=0.20
calibration→evaluation AUROC drop <=0.08
policy false-safe 过冻结门
```

失败：

```text
关闭 action authority
不调 threshold
不直接进入 RL
```

---

# 18. P11 — Exact-Once Test 与 SceneIR Bake

Task：

```text
WS-V65-P11-EXACT_ONCE_TEST-01
```

使用 T-Exact-Test：8 scenes / 96 cases。

只允许冻结 candidate：

```text
exact risk model
exact task/actor/context module
exact admission/allocator
exact calibrator
exact deterministic bake
```

Primary：

- matched total coverage；
- fixed-opportunity physical/task risk；
- worst-scene / worst-group；
- paired lower/equal/higher；
- route/non-route risk transfer；
- deterministic package exactness。

Action head 如果 P10 通过，再作为独立 secondary claim。

只有 P11 通过才能称：

```text
TAC-Compiler confirmed
```

---

# 19. Stop Rules

## Stop 1 — Task condition 无增量

T0 不优于 V6.4 `q(x)`：

```text
关闭 task-conditioned representation family
```

不建 Transformer。

## Stop 2 — Attention 无增量

T0 有效、T1 无增量：保留低容量残差，关闭 attention expansion。

## Stop 3 — Actor interaction 无增量

A0/A1 不优于 trajectory-only：关闭 Actor family，不影响 trajectory result。

## Stop 4 — Learned admission 不优于硬 M1

关闭 differentiable admission，不因形式优雅继续。

## Stop 5 — Risk 搬家

route 风险下降但 non-route / global 风险显著上升：candidate reject。

## Stop 6 — Soft allocator 无增量

不继续 MoE / OT 变体。

## Stop 7 — Joint training 无增量或 collapse

保留 modular compiler，不全量解冻 foundation。

## Stop 8 — PEFT 只改善 train

关闭 PEFT family，不做 rank/seed sweep。

## Stop 9 — Calibration 只能 all-UNKNOWN

reject。

## Stop 10 — Confirmation/Test 失败

family closed，不换新 cohort 继续救。

## Stop 11 — Action ranking 失败

关闭 action authority，不进入 NWM/RL。

---

# 20. 资源合同

## 20.1 GPU

```text
P0–P1             CPU / 单卡 capability
P2–P6             1×3090
P7 PEFT            1×3090 probe；必要时 2–4×3090
P8–P11             1×3090；scene parallel 仅解决吞吐
```

首轮网络规模不应接近 24 GiB。

## 20.2 存储

V6.4 已证明主要瓶颈是 tar / processed / sidecar I/O，不是 GPU。

V6.5 启动前建议可用空间：

```text
>=80 GiB
```

若不足：

- restricted-shard extraction；
- raw scene 完成后立即删除可重建临时数据；
- canonical processed reuse；
- scene-ready CPU→GPU pipeline；
- compact train feature cache；
- 正式 sidecar/run 不覆盖。

不得通过降低 ROI、分辨率、时序或 native feature 解决磁盘。

## 20.3 资源阻塞

只有以下情况写 `blocked_resource`：

- faithful minimum 两次 OOM；
- 有界恢复后仍 OOM；
- 正式 cohort 无法落盘且无法安全流式处理；
- 单卡吞吐不可完成且多卡不可得；
- 磁盘安全线不足且无可删除的可重建缓存。

资源 blocked 不等于算法 rejected。

---

# 21. 建议代码结构

```text
docs/WORLDSIM_V6_5_TASK_CONDITIONED_DIFFERENTIABLE_AUTHORITY_COMPILER_PLAN.md

docs/autoresearch/worldsim_v65/
  AUTORESEARCH_STATE.current.json
  HYPOTHESES.jsonl
  REFLECTIONS.jsonl
  SELECTION_EXPOSURE_LEDGER.json
  SOURCE_MATRIX.md
  FAILURE_ANALYSIS.md

configs/worldsim_v65/
  p0_protocol_v1.yaml
  p1_signal_atlas_v1.yaml
  p2_trajectory_condition_v1.yaml
  p3_actor_time_v1.yaml
  p4_admission_v1.yaml
  p5_soft_budget_v1.yaml
  p6_joint_fusion_v1.yaml
  p7_irwm_peft_v1.yaml
  p8_calibration_v1.yaml
  p9_confirmation_v1.yaml
  p10_action_transfer_v1.yaml
  p11_test_v1.yaml

motion_proj/worldsim_v65/
  task_contract.py
  trajectory_encoder.py
  actor_set_encoder.py
  context_encoder.py
  conditional_validity.py
  interaction_risk.py
  differentiable_admission.py
  soft_budget_allocator.py
  modular_compiler.py
  joint_compiler.py
  gradient_audit.py
  calibration.py
  deterministic_bake.py
  action_risk.py

scripts/
  run_worldsim_v65_p1_signal_atlas.py
  run_worldsim_v65_p2_trajectory.py
  run_worldsim_v65_p3_actor_time.py
  run_worldsim_v65_p4_admission.py
  run_worldsim_v65_p5_soft_budget.py
  run_worldsim_v65_p6_joint.py
  run_worldsim_v65_p8_calibration.py
  run_worldsim_v65_p9_confirmation.py
  run_worldsim_v65_p10_action.py
  run_worldsim_v65_p11_test.py
```

复用：

- V6.4 `selective_mlp.py` 的 trunk/checkpoint；
- native sidecar/evidence mapping；
- fixed-route denominator evaluator；
- state bake / Gaussian adapter；
- scene-ready producer-consumer；
- run contract 和统一 failure ledger。

不得复制第二套 failure ledger。

---

# 22. 第一轮执行顺序

```text
P0  inheritance + source/protocol/data freeze
↓
P1  train-only signal atlas
↓
P2  trajectory condition
    FAIL → close task-conditioned family
↓
P3  actor/time interaction（条件解锁）
↓
P4  learned admission
    FAIL → retain V6.4 M1, close learned admission
↓
P5  soft budget（只有 hard selector ceiling 才解锁）
↓
P6  modular → joint fusion
    FAIL → keep modular candidate
↓
P7  targeted PEFT（只有 representation ceiling 才解锁）
↓
P8  independent calibration
↓
P9  exact-once confirmation
↓
P10 action transfer（先 score，再 critic）
↓
P11 exact-once test + deterministic bake
```

---

# 23. 预期论文贡献边界

如果全部通过，可以主张：

1. **Task-Conditioned Physical Validity Field**  
   从 `q(x)` 升级为 `q(x | trajectory, actor, context, task)`。
2. **Differentiable Conditional Admission**  
   将人工 stratum map、route cap 和 budget reallocation 替换为受风险/覆盖约束的可学习模块。
3. **Sparse Actor-Time Risk Decomposition**  
   为 Actor、时间和任务提供可解释风险归因。
4. **Staged-to-Joint Verifiable Compiler**  
   先模块化证伪，再联合反传，同时保留独立校准和确定性 Runtime。
5. **Fixed-Opportunity Evaluation**  
   防止 selective policy 通过改变分母制造假改进。

如果 action head 失败：

- 前四项仍可成立；
- 不主张 collision/planning contribution。

如果 joint training 失败：

- 保留 modular differentiable compiler；
- 不主张 end-to-end contribution。

如果 learned admission 不优于 V6.4 M1：

- V6.5 主 hypothesis rejected；
- 不靠更大模型或更多 expert 救结果。

---

# 24. 一句话研究主张

> **V6.5 不再用人工场景标签和固定路线阈值决定哪些 Occupancy 状态有资格进入驾驶世界，而是逐步学习一个由轨迹、Actor、时间与任务条件化的物理有效性和写入权限模型；只有每个增量模块被独立证明有效后，才进行端到端联合优化，并最终通过独立校准固化为确定性 SceneIR。**
