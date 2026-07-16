# Motion-Proj / DrivePO 自动研究计划 v4（AC 修订版）

> **工作方法名**：**DrivePO — Common-Support Partial-Order Alignment for Driving Video Diffusion**
> **文档状态**：前向唯一计划；取代 `docs/PHYSICS_DPO_AUTORESEARCH_PLAN_V3.md` 的未来排程，不修改历史实验事实。
> **最后更新**：2026-07-16
> **当前代码基线**：用户提供的 `Motion-Proj-main-v3`；最近文档提交为 `b5d642b3a54d3c629f37a500eaf3f73e7540dc69`，执行时必须重新核对远端 `HEAD`。
> **当前硬件**：单张 RTX 4090 24 GB；PA4 单卡筛选通过前禁止切换双卡。
> **投稿目标**：CVPR 2027。
> **状态词**：`pending / running / awaiting_reviews / blocked / done / rejected`。
> **当前唯一允许执行的任务**：`PA2-UPO-03B`，只在已有 sibling RGB 上重建可信偏好 oracle；不生成新候选、不训练、不切双卡。

---

# 0. Area Chair 决策

## 0.1 当前路线的准确状态

当前项目已经完成：

- 官方 SVD generation parity；
- 14-frame、25-step 的 common-prefix sibling 生成；
- 120 个 condition 的 sibling RGB 资产；
- 旧 P-UNC scorer 的 53 个 machine global pairs；
- 48 条两阶段人工 review。

人工结果已明确否决旧标签配方：

| 项目 | 结果 |
|---|---:|
| 总人工 review | 48/48 |
| P1 common-prefix review | 24 |
| P1 human decisive | **0/24** |
| P1 human tie | **22/24** |
| P1 uncertain | 1/24 |
| P1 both-invalid | 1/24 |
| P1 machine label | 13 `a_wins` + 11 `b_wins` |
| 总体 machine-human decisive agreement | 4/8 |
| Wilson 95% lower bound | 0.2152 |
| scorer-chosen catastrophic failure | 2 |

因此：

> 当前 53 个 global pair 和既有 local labels 禁止进入任何 DPO、AWR、SFT 或其他训练。

保留的是：

- 结构对齐 sibling RGB；
- scene split；
- generation provenance；
- review negative evidence；
- P-UNC 与 CoTracker 基础设施。

被否决的是：

```text
候选各自选 query
→ 各自跟踪与平滑
→ 比较不同 support 上的平均 projection energy
→ 最大 pair_confidence 选边
→ 强制 a_wins / b_wins
```

## 0.2 当前创新假设

DrivePO 不再把创新放在“physics score + DPO”。

真正待验证的闭环是：

\[
\boxed{
\text{common-support sibling evidence}
\rightarrow
\text{selective strict/tie/incomparable relation}
\rightarrow
\text{dynamic-track-tube preference alignment}
}
\]

其中：

1. **共同证据**：两条 sibling 必须在相同 first-frame query ID 和共同可见 support 上比较；
2. **选择性偏序**：允许 strict、tie 和 incomparable，不强制每条 pair 有 winner；
3. **动态轨迹局部化**：只在可回溯的 dynamic-track/time tube 上更新；
4. **防投机**：少动、track dropout、相机运动差异和画质失败不能产生 strict winner；
5. **训练稳定性**：tie-aware objective、winner safeguard、complement anchor 和 real-video anchor。

## 0.3 AC 总门槛

在实现 DPO 前，必须先回答：

> 新 oracle 能否在不读取 rollout 结果的情况下，消除旧 P1 human ties 上的 false-strict，并在未参与阈值校准的 sibling 上找到人工可确认的 strict preference？

如果不能，项目应停止 SVD sibling preference 路线，而不是继续“改 scorer—扩数据—再训练”。

---

# 1. 最新仓库事实

## 1.1 已有代码

当前与 preference 路线直接相关的实现包括：

```text
motion_proj/preference/pair_scoring.py
motion_proj/preference/review.py
motion_proj/diagnostics/physics_dpo_pair.py
motion_proj/diagnostics/physics_dpo_pair_merge.py
motion_proj/diagnostics/physics_dpo_branch.py
motion_proj/diagnostics/physics_dpo_horizon.py
motion_proj/data/physics_dpo_schema.py

tests/test_physics_dpo_pair.py
tests/test_physics_dpo_branch.py
tests/test_physics_dpo_horizon.py
tests/test_physics_dpo_schema.py
tests/test_physics_dpo_pa0_review.py
```

当前尚不存在：

```text
paired common-support tracker
candidate-set partial-order graph
ROPE / selective calibration
dynamic-track tube schema
DPO / tie-DPO / SDPO loss
preference trainer
DrivePO capacity run
```

不得把计划公式写成“已实现方法”。

## 1.2 旧 scorer 的代码语义

`pair_scoring.py` 当前：

- 每个 candidate 独立生成 track set；
- 分别计算 `projection_energy`；
- 使用 projection margin + non-inferiority 判 winner；
- `pair_confidence` 由 track confidence、relative margin、saturation 相乘；
- 每 condition 在两条 antithetic edge 中按最大 confidence 选一条。

该 confidence 不是 preference correctness probability，已有 24 条 P1 人审直接证伪其解释。

## 1.3 RAFT provider 的正式风险

`RAFTChainGeneratedTrackProvider` 当前会：

1. 从每个 candidate 自己的第一个 flow 中选择 query；
2. 使用 `background / dynamic_residual / foreground_candidate` 三个启发式 stratum；
3. 当某个 stratum 点不足时，从所有 `valid` 点 fallback；
4. 在输出前对坐标应用三帧中值平滑；
5. 把启发式点包装为 16×16 `Track` tube。

因此，新 oracle 不得直接复用当前 `track()` 的最终输出作为正式 paired measurement：

- 三个 stratum 不等于真实 object/actor instance；
- fallback 会污染 stratum 语义；
- 预平滑会改变待测 acceleration/curvature；
- candidate-wise query selection 破坏 paired comparison。

v4 统一使用：

```text
background anchor queries
dynamic-track candidate queries
```

禁止把它们写成“actor GT”或“实例轨迹”。

## 1.4 历史资产冻结

以下文件和 run 只读：

```text
/root/autodl-tmp/runs/autoresearch-pa2-pair-expanded-s20260715-v1
/root/autodl-tmp/runs/autoresearch-pa1-branch-s20260715-v5
/root/autodl-tmp/runs/autoresearch-pa0-review-s20260715-v1
/root/autodl-tmp/runs/autoresearch-e0-evaluator-s20260714-v3
```

旧：

```text
preferences.jsonl
segments.jsonl
candidate_scores.jsonl
```

不得原地升级、覆盖或转成训练数据。

---

# 2. 研究问题与准确 claim

## 2.1 核心研究问题

> 对同一驾驶场景的模型 sibling futures，能否在不使用 future GT 的前提下，在共同轨迹证据上构造可 abstain 的短时运动偏序，并将可信 strict/tie 关系稳定地注入视频扩散模型，使独立 evaluator 和人工判断中的 dynamic-track motion consistency 改善，同时不牺牲运动量、track survival、相机—背景一致性、身份和画质？

## 2.2 准确术语

论文和代码必须使用：

```text
camera-compensated image-plane motion
dynamic-track residual
background motion field
common-support acceleration/curvature outlier
dynamic-track tube
short-horizon motion consistency
```

不得使用：

```text
真实世界加速度
牛顿力学真值
actor ground truth
ego motion ground truth
物理定律已满足
```

## 2.3 三个可证伪假设

### H1：共同 support 能消除旧 scorer 的伪方向

同一 first-frame query set、共同 visibility 和相同 denominator 后，旧 22 条 human ties 中绝大多数不再被高置信判为 strict。

### H2：可选择偏序比 forced binary label 更可信

strict / tie / incomparable oracle 在 prospective blind review 中具有足够的 strict precision，并能识别 tie 与 invalid，而不是仅降低数据量。

### H3：tube-local tie-aware alignment 比数据筛选基线更有效

在同一 candidate pool、更新预算和 LoRA 下，完整方法必须超过：

- chosen-only SFT；
- scalar AWR；
- clip-level DPO；
- temporal DenseDPO-style baseline。

---

# 3. 与最近邻工作的边界

| 工作 | 已覆盖内容 | DrivePO 不可声称 | 必须保留的差异 |
|---|---|---|---|
| Diffusion-DPO，CVPR 2024 | diffusion DPO 基础目标 | DPO loss 本身 | driving common-support relation |
| VideoDPO，CVPR 2025 | 自动多指标评分、pair reweighting | 自动评分与大 margin 筛选 | 不用 scalar total score，允许 abstain |
| DenseDPO，NeurIPS 2025 | 结构对齐 pair、segment preference、低运动偏置 | sibling/segment 本身 | common query support、measurement-calibrated relation |
| LocalDPO，CVPR 2026 | 局部 corruption pair、region-aware DPO | region mask loss | 非人工 corruption；对应 dynamic tracks；tie/incomparable graph |
| Diffusion-SDPO，2025 | winner-preserving safeguard | safeguard 原创性 | 作为训练稳定组件 |
| Tie-aware DPO，NeurIPS 2025 | Rao-Kupper/Davidson tie likelihood | tie loss 原创性 | driving human-tie calibration 与动态 tube |
| Uncertainty-Penalized DPO，2024 | 不确定 pair 降权 | uncertainty regularization | paired support、selective false-strict control |
| SHIFT，2026 | motion reward、AWR、SFT hybrid | reward/AWR 本身 | AWR 是强 baseline；主方法为 selective partial order |
| VideoGPA，2026 | 自动几何 preference | 自动 geometry DPO | 无 3D foundation model；动态轨迹 common support |

如果完整方法不能显著超过 DenseDPO-style temporal baseline 与 LocalDPO-style region ablation，则当前创新不足以支持 CVPR 主张。

---

# 4. Reviewer-first 威胁模型

| 质疑 | 必须回答的证据 | 失败动作 |
|---|---|---|
| “只是 DenseDPO + driving metric” | unpaired→common-support 的人工 strict precision提升；tie/incomparable 消融 | 停止方法包装 |
| “stratum 不是 actor” | 准确称 dynamic-track；不使用实例语义 claim | 删除 actor claim |
| “P-UNC 仍是自定义 smoothness” | human calibration、CoTracker evaluation、generic smoothness baseline | 只在 P-UNC 改善则 reject |
| “22 ties 被拿来调到不报 strict” | calibration/holdout 分离 + prospective review | holdout false-strict 失败即 reject |
| “模型只是少动或丢 track” | freeze/slow/dropout stress；motion/survival 非劣 | 任一 shortcut 获胜即 reject |
| “相机运动解释了 actor 改善” | background equivalence gate；按 motion strata 分析 | camera field 不等价则 incomparable |
| “tube DPO 仍会全局泄漏” | one-pair capacity 的 tube/outside/frame-0 gate | 一次 fallback 后失败即 reject |
| “DPO 只是恶化 loser” | winner absolute error + SDPO safeguard | winner 退化则 reject |
| “数据筛选已经足够” | chosen-SFT、AWR 与 DPO 同数据对照 | DPO 不超过它们则改题或停止 |

---

# 5. 方法：DrivePO

# 5.1 Common-prefix sibling candidates

现有 candidate 生成保持冻结：

\[
X^i =
G_{\theta_0}^{t_f\rightarrow 0}
\left(
z_{t_f}^{\mathrm{prefix}}+\delta_i,c
\right),
\quad i=1,\dots,4.
\]

当前主资产：

```text
120 conditions
4 common-prefix siblings / condition
14 frames
25 denoising steps
fork fraction = 0.6
strength rho = 0.04
```

本阶段不重生成 candidate。

每 condition 最多有 6 条 sibling edge，但 relation construction 和训练采样都以 condition 为基本单位。

# 5.2 Formal query protocol

## 5.2.1 Query 只从共同 first frame 选择一次

因为 sibling first frame 已通过 exact/structure gate，定义：

\[
Q_c = Q_c^{bg}\cup Q_c^{dyn}.
\]

query set 从 Base guard 或所有 sibling 共用的 first frame 只计算一次，并保存：

```text
query_id
xy
stratum
selection score
query_set_hash
```

所有 sibling 使用完全相同的 query IDs 和起点。

## 5.2.2 禁止 stratum fallback

正式 oracle 中：

- `background` query 必须满足低 residual、高 flow confidence 和足够纹理；
- `dynamic` query 必须满足高 background-residual、足够 confidence 和空间间隔；
- 点不足时该 stratum 标为 invalid；
- 禁止从全部 valid 点补齐预算；
- query 不得同时属于两个 stratum；
- 不构造伪 object instance。

建议初始预算：

```text
background queries: up to 24
dynamic queries: up to 24
minimum background queries: 12
minimum dynamic queries: 8
minimum spatial clusters: 4
```

阈值必须在正式 run 前写入 YAML；不得为提高 strict yield 后改。

## 5.2.3 Raw 与 smoothed tracks 分离

新增 pair-mode provider，必须同时输出：

```text
raw_points
raw_visibility
raw_confidence
forward_backward_error
optional_smoothed_points
```

正式 preference measurement 使用 raw/common-support tracks 与 robust statistics。

三帧中值平滑：

- 只能作为 secondary diagnostic；
- 不得在计算 acceleration、curvature 和 bootstrap uncertainty 前原地修改 raw points；
- 不得同时用于产生 correction 和证明 correction 有效。

# 5.3 Paired common support

对 candidate \(i,j\)、query \(q\) 和 frame \(t\)：

\[
\Omega_{ij}
=
\{(q,t):
v^i_{q,t}=v^j_{q,t}=1,
c^i_{q,t}\ge\tau_c,
c^j_{q,t}\ge\tau_c,
\mathrm{FB}^i_{q,t}\le\tau_{fb},
\mathrm{FB}^j_{q,t}\le\tau_{fb}
\}.
\]

对 4-frame、stride-2 窗口：

\[
\mathcal I_s=\{s,s+1,s+2,s+3\},
\quad
s\in\{0,2,4,6,8,10\}.
\]

每条 edge/window 必须保存：

```text
common query count
common query-frame count
background/dynamic count
effective visible length
spatial cluster count
coverage by frame
missingness by candidate
```

初始比较门槛：

```text
common dynamic queries >= 8
common background queries >= 12
common dynamic query-frame observations >= 24
spatial clusters >= 4
```

任一不足：

```text
relation = incomparable_support
```

track dropout 不能通过缩小 denominator 获益；survival/missingness 作为独立 non-inferiority constraint。

# 5.4 Background nuisance compensation

## 5.4.1 当前阶段只使用 robust affine

仓库已有 `fit_affine_background_flow`。v4 不并行搜索 translation、homography、mesh 等模型。

每个 candidate、每个相邻 frame 只在共同 background anchors 上拟合 robust affine field：

\[
g_t^i(x).
\]

动态 residual：

\[
r_{q,t}^i =
p_{q,t+1}^i-p_{q,t}^i-g_t^i(p_{q,t}^i).
\]

要求：

- 只使用 background stratum；
- RANSAC/robust fit seed 固定；
- 保存 inlier ratio 和 residual；
- background fit 失败则 candidate invalid；
- 不读取 future ego pose。

## 5.4.2 Pair camera comparability

在共同 background support 上定义：

\[
D_g(i,j)=
\operatorname{median}_{q,t}
\left\|
g_t^i(p_{q,t})-g_t^j(p_{q,t})
\right\|_2.
\]

若 \(D_g\) 超过 measurement-equivalence band：

```text
relation = incomparable_camera
```

不判断哪条 candidate 的 camera motion 更“正确”。

# 5.5 Motion evidence vector

对 dynamic common support，计算 paired component：

\[
\boldsymbol\phi_{i,s}=
[
E_{\mathrm{punc}},
E_{\mathrm{acc}},
E_{\mathrm{curve}},
E_{\mathrm{coh}},
E_{\mathrm{surv}}
]_{i,s}.
\]

## 5.5.1 \(E_{\mathrm{punc}}\)

P-UNC 只在完全相同的 common raw tracks 上运行：

\[
E_{\mathrm{punc}}
=
\frac{
\sum w_{q,t}
\|p_{q,t}-\Pi_{\mathrm{UNC}}(p_q)_t\|_2^2
}{
\sum w_{q,t}+\varepsilon
}.
\]

只计入 correction SNR 不低于冻结门槛的点。

## 5.5.2 \(E_{\mathrm{acc}}\)

基于 background-compensated residual velocity：

\[
a_{q,t}=r_{q,t+1}-r_{q,t}.
\]

报告 robust acceleration magnitude 与 outlier fraction，不将绝对像素值解释为真实物理量。

## 5.5.3 \(E_{\mathrm{curve}}\)

使用相邻 residual velocity 的方向变化：

\[
\theta_{q,t}=
\angle(r_{q,t},r_{q,t+1}).
\]

只在速度高于 noise floor 时计算，避免静止点方向不稳定。

## 5.5.4 \(E_{\mathrm{coh}}\)

对 first-frame 邻近 dynamic queries，计算局部 residual motion 的 robust dispersion，用于发现：

- 融化；
- 局部撕裂；
- 同一纹理区域方向分裂。

不得声称这些 query 属于同一 object instance。

## 5.5.5 \(E_{\mathrm{surv}}\)

包含：

- common query early death；
- visibility asymmetry；
- sudden confidence collapse；
- track length 不足。

survival 不与其他 component 加权抵消。

# 5.6 Activity 与 quality 约束

单独保存 activity vector：

\[
\mathbf A_{i,s}=
[
\text{active fraction},
\text{residual speed},
\text{net residual displacement}
].
\]

候选不能通过以下方式获胜：

- residual speed 大幅下降；
- net displacement 大幅下降；
- active queries 消失；
- 画面冻结；
- track survival 下降。

质量只作 hard comparability gate，不进入 physics scalar reward。

初始 quality gate：

```text
finite and decodable
first frame valid
no black-frame / severe saturation
no catastrophic temporal feature jump
no catastrophic identity/geometry flag
pairwise quality difference within equivalence band
```

质量阈值由：

- real nuScenes RGB；
- Frozen Base；
- 已有 catastrophic human cases；

预注册并冻结。不得根据 machine winner 或训练 rollout 调节。

# 5.7 Paired uncertainty

对每个 component 的 paired difference：

\[
d_{q,s}^{(m)}
=
\phi_{j,q,s}^{(m)}-\phi_{i,q,s}^{(m)}.
\]

采用：

```text
spatial-cluster paired bootstrap
× temporal block bootstrap
```

要求：

- 相同 query ID paired resampling；
- window 内 frame 不独立抽样；
- bootstrap seed 固定；
- 保存 point estimate、CI、ESS、cluster count；
- 同 condition 的 6 条 edge 使用 Holm correction；
- 禁止先选最大 margin edge再计算普通 CI。

若 strict edge 构成 cycle：

```text
整个 condition graph = invalid_cycle
```

不得静默删除最弱 edge。只有无 cycle 图才允许做 transitive reduction。

# 5.8 Calibration：measurement ROPE 与 human selection threshold 分离

## 5.8.1 Measurement ROPE

每个 component 的 measurement-equivalence band 只能来自：

- identical rerun；
- benign photometric perturbation；
- codec/resize perturbation；
- tracker/query jitter；
- real/Base repeatability。

它描述 measurement noise，不读取 human preference。

## 5.8.2 22 个 human ties 的冻结拆分

现有 22 个 P1 ties 必须按 scene 哈希固定分为：

```text
12 calibration ties
10 retrospective holdout ties
```

要求：

- scene 不重叠；
- seed、split 和 hash 落盘；
- 不得重新抽到得到更好结果；
- 1 uncertain 和 1 both-invalid 只作 invalidity audit。

human ties 不用于逐 component 调多个阈值。样本太小，禁止拟合复杂校准器或 reward model。

只允许校准一个全局 strict-evidence threshold：

\[
T_{ij,s}
=
\max_m
\frac{
\text{simultaneous lower bound}^{(m)}
}{
\text{measurement ROPE}^{(m)}+\varepsilon
},
\]

同时必须满足全部 non-inferiority constraints。

使用 calibration ties 选择保守阈值，使 false-strict 风险不超过预注册水平；具体采用 split-conformal quantile，并保存 calibration coverage。

## 5.8.3 Retrospective holdout 门槛

在 10 个从未用于阈值选择的 human ties 上：

```text
false strict <= 1/10
high-confidence false strict = 0
invalid/tie mapping 可解释
```

该结果只证明旧 false-strict 风险降低，不证明 strict direction 正确。

# 5.9 Strict / tie / incomparable relation

只有先通过：

```text
same condition and prefix
same first-frame query IDs
support pass
background model pass
camera comparability pass
quality comparability pass
activity/survival non-inferiority
```

才进入 relation decision。

## Strict：\(i\succ_s j\)

满足：

1. 至少一个 primary component 的 simultaneous CI 越过 measurement ROPE 和 calibrated strict threshold；
2. 其他 primary component 无显著反向退化；
3. activity、survival、coverage、quality 非劣；
4. relation 对 bootstrap seed 与轻微 tracker perturbation 稳定；
5. condition graph 无 cycle。

## Tie：\(i\sim_s j\)

满足：

- support、camera、quality 均可比较；
- 所有 primary component CI 位于 measurement ROPE；
- activity/survival 差异也在 equivalence band。

## Incomparable：\(i\parallel_s j\)

包括：

```text
support insufficient
camera mismatch
quality mismatch
component conflict
uncertainty too wide
cycle
invalid track
```

incomparable 不进入 preference loss。

# 5.10 Prospective human validation

校准与 holdout 通过后，将冻结 oracle 应用于未参与 review 的 sibling conditions。

## 5.10.1 最低 yield

必须按 **condition 数** 统计，不按 edge 数统计。

准备 prospective review 的最低条件：

```text
至少 16 个 predicted-strict conditions
至少 8 个 predicted-tie conditions
至少 8 个 predicted-incomparable conditions
```

若 strict conditions 少于 16，不进行训练，也不降低阈值，进入一次候选 fallback。

## 5.10.2 Review 样本

固定 32 cases：

```text
16 strict
8 tie
8 incomparable
```

blind Stage A 后再看 diagnostics。

严格 precision 定义：

```text
所有 predicted strict 都进入分母
human tie / uncertain / invalid 均视为 false strict
方向相反视为 false strict
```

门槛：

```text
strict direction correct >=13/16
strict precision Wilson 95% lower >0.50
predicted tie 中 human tie >=6/8
low-motion chosen = 0
catastrophic chosen = 0
至少 25% 双人重叠 review
strict/tie 粗粒度 reviewer agreement >=75%
```

只有 prospective review 通过，oracle 才可用于训练数据。

---

# 6. 唯一候选 fallback

如果冻结 oracle 通过 tie holdout，但 strict condition 少于 16，只允许一次：

```text
8 new conditions
fork fraction: 0.4
rho: 0.04
其他 generation settings 不变
```

必须重新通过：

- exact Base guard；
- same-scene structure review 至少 7/8；
- first-frame/query protocol；
- quality gate；
- oracle prospective strict precision。

如果 earlier-fork：

- 仍主要为 human tie；
- 或造成 identity/layout mismatch；
- 或 strict 主要来自画质失败；

则：

```text
SVD common-prefix sibling route = rejected
```

不得继续搜索 fork/rho、增加 candidate 数或切双卡。

若 pilot 通过，可 append-only 生成新 conditions，直至获得：

```text
至少 24 strict condition graphs
至少 24 tie condition graphs
```

用于 PA3 capacity。正式 PA4 screening 前需要扩大到至少 96 strict conditions。

---

# 7. Dynamic-track tube construction

对 common query/window 构建每个 candidate 自己的 soft tube：

\[
M_{e,s}^{u}(t,h,w)
=
\sum_{q\in\Omega_{e,s}}
\bar w_{q,t}^{u}
K((h,w)-p_{q,t}^{u}),
\quad u\in\{i,j\}.
\]

要求：

- 只使用 dynamic common queries；
- frame 0 mask 强制为 0；
- 每个 common query 在 winner/loser 两侧贡献相同总质量；
- mask mass 按 common query 数和 window 长度归一化；
- tube radius 只允许一次离散 capacity 对照；
- 保存 boundary band；
- boundary 不进入 strict loss或 complement anchor；
- 没有 common query 时 fail closed；
- tube 能回溯到 query ID 和 relation component。

统一名称：

```text
dynamic-track tube
```

除非未来接入并验证 instance detector/segmenter，不得写 `actor tube`。

---

# 8. Tie-aware safeguarded diffusion alignment

只有 oracle prospective review 通过后才实现本节。

# 8.1 Pair diffusion construction

对 strict/tie edge 的两条真实 candidate latent：

\[
z_\tau^u
=
\alpha_\tau x_0^u+\sigma_\tau\epsilon,
\quad u\in\{a,b\}.
\]

要求：

- pair 内共享 \(\tau,\epsilon\)；
- official SVD conditioning；
- 使用现有已测试的 `model_output_from_x0`；
- reference 通过关闭 LoRA 顺序 no-grad forward；
- policy/reference 状态必须 exact restore。

# 8.2 Tube-local error

\[
\ell_\theta^u(e)
=
\frac{
\sum M_{e,s}^{u}
\|v_\theta(z_\tau^u,\tau,c)-v^{*,u}\|_2^2
}{
\sum M_{e,s}^{u}+\varepsilon
}.
\]

reference-relative margin：

\[
d_\theta(e)=
[\ell_\theta^l-\ell_{\mathrm{ref}}^l]
-
[\ell_\theta^w-\ell_{\mathrm{ref}}^w].
\]

# 8.3 Strict loss

\[
\mathcal L_{\mathrm{strict}}
=
-
\frac{
\sum_{e\in\mathcal E_\succ}q_e
\log\sigma(\beta d_\theta(e))
}{
\sum q_e+\varepsilon
}.
\]

其中：

- \(q_e\in[0,1]\) 只能降权不确定样本；
- condition-balanced sampling；
- transitive duplicate edge 不重复加权；
- 每 condition 每 update 最多贡献固定数量 edge。

# 8.4 Tie loss

主 tie objective 使用 Davidson tie-aware DPO，直接根据论文/官方实现适配 diffusion margin \(d_\theta\)。

要求：

- clear-preference 与 tie 的概率归一；
- zero adapter 时 tie margin 为 0；
- tie edge 推动 \(|d_\theta|\) 靠近 0；
- Davidson tie parameter 在训练前固定；
- 不根据 rollout 指标调节。

以下只作为 baseline：

```text
Huber zero-margin tie loss
discard ties
incorrectly force random winner
```

tie-aware DPO 是借鉴组件，不作为原创贡献。

# 8.5 Winner safeguard

strict edges 使用 Diffusion-SDPO 风格 winner-preserving loser scaling。

每 step 记录：

```text
winner policy absolute error
winner reference error
loser policy absolute error
loser reference error
raw margin
safe margin
loser scale
```

仅 margin 上升不构成通过；winner absolute tube error 不得系统增加。

# 8.6 Complement 与 real anchor

\[
\mathcal L
=
\mathcal L_{\mathrm{strict}}
+
\lambda_{\mathrm{tie}}\mathcal L_{\mathrm{tie}}
+
\lambda_{\mathrm{comp}}\mathcal L_{\mathrm{comp}}
+
\lambda_{\mathrm{real}}\mathcal L_{\mathrm{real}}.
\]

`comp`：

- 在每侧 tube complement 上模仿 frozen reference；
- boundary band 不参与；
- frame 0 全部进入 exact preservation check；
- policy/reference 使用相同 noisy input。

`real`：

- 真实训练视频 denoising SFT；
- 与 preference branch 共享 timestep；
- shape 允许时共享 noise；
- 权重用 gradient-RMS 预校准，不按 rollout 指标搜索。

# 8.7 Timestep 诊断

在 PA3 必须报告：

```text
per-timestep DPO logit
gradient RMS
winner/loser error
sigmoid saturation
```

如果早期高噪 timestep 出现严重方差或 off-policy instability，只允许一个预注册 fallback：

```text
timestep clipping/masking baseline
```

不直接引入完整 importance-sampling 新主线。

---

# 9. Oracle stress suite

训练前必须在不改模型的情况下通过：

## 9.1 Freeze attack

将后续帧重复或显著减速。

期望：

```text
不能成为 strict winner
activity/survival gate 拒绝
```

## 9.2 Time-slow attack

时间插值或重复 frame 使 acceleration 看似下降。

期望：

```text
motion exposure non-inferiority 拒绝
```

## 9.3 Track-dropout attack

遮挡或删除难跟踪区域，使 projection energy denominator 变小。

期望：

```text
common support / survival penalty 拒绝
```

## 9.4 Camera perturbation

对全局 affine camera field 做轻微变化，dynamic content 保持近似。

期望：

```text
camera mismatch => incomparable
不产生 dynamic strict winner
```

## 9.5 Quality attack

引入 blur、flicker、geometry deformation、identity corruption。

期望：

```text
quality invalid
不能作为 winner 或 loser 进入 physics DPO
```

## 9.6 Common transform invariance

对 pair 两侧施加相同 benign transform。

期望：

```text
relation direction稳定
tie/strict 不因公共 nuisance 翻转
```

任一 shortcut attack 可获得 high-confidence strict winner：

```text
PA2-UPO-03B = rejected
```

---

# 10. 里程碑

| ID | 当前状态 | 任务 | 通过条件 | 失败动作 |
|---|---|---|---|---|
| PA2-UPO-03B | pending | common-support oracle，复用现有 RGB | tie holdout + stress 通过 | reject oracle |
| PA2-PROSPECT-03C | blocked | 32-case prospective review | strict/tie precision 门槛 | reject oracle |
| PA2-CAND-03D | blocked | 唯一 earlier-fork fallback | 结构+strict precision通过 | reject SVD sibling |
| PA3-KERNEL-04 | blocked | strict/tie/tube/SDPO 代数与容量 | 1/8/24 condition 依次通过 | 只修实现或 reject |
| PA4-SCREEN-05 | blocked | 单卡方法筛选 | full 超强基线 | 不切双卡 |
| PA5-SCALE-06 | blocked | 双卡扩数据 | 300–800 validated relations | 保留单卡结果 |
| PA6-FORMAL-07 | blocked | 两训练 seed | 主指标同向 | reject |
| PA7-EVAL-08 | blocked | 128/256+ clips 评估 | 统计+人工通过 | 归档负结果 |
| PA8-PAPER-09 | blocked | 主表、消融、第二 backbone | 贡献可答辩 | 不投稿当前方法 |

---

# 11. PA2-UPO-03B：当前唯一任务

## 11.1 代码新增

新增，不修改旧 scorer 语义：

```text
motion_proj/preference/paired_tracks.py
motion_proj/preference/common_support.py
motion_proj/preference/residual_motion.py
motion_proj/preference/selective_order.py
motion_proj/preference/calibration.py

motion_proj/diagnostics/physics_preference_reaudit.py

configs/diagnostics/physics_preference_reaudit.yaml
```

旧：

```text
pair_scoring.py
preferences.jsonl
segments.jsonl
```

保持只读 baseline。

## 11.2 必须新增测试

```text
tests/test_paired_query_protocol.py
tests/test_no_stratum_fallback.py
tests/test_raw_track_preservation.py
tests/test_common_support.py
tests/test_background_residualization.py
tests/test_camera_incomparability.py
tests/test_measurement_rope.py
tests/test_calibration_split.py
tests/test_selective_partial_order.py
tests/test_partial_order_cycles.py
tests/test_shortcut_attacks.py
tests/test_reaudit_fail_closed.py
```

测试至少覆盖：

```text
same first-frame queries across siblings
query IDs unique across strata
insufficient stratum => invalid, no fallback
raw points unchanged by diagnostic smoothing
dropout cannot improve denominator
common transform preserves direction
cycle invalidates condition
calibration/holdout scene disjoint
old artifacts never overwritten
future-GT fails closed
```

## 11.3 正式 run

```text
run_id:
autoresearch-pa2-upo-s20260716-v1
```

只读输入：

```text
120-condition sibling RGB
old 24 P1 review mappings
PA0 scorer/evaluator fingerprints
scene split
```

输出：

```text
manifest.json
resolved.yaml
query_sets.jsonl
paired_tracks.jsonl
common_support.jsonl
background_fields.jsonl
component_differences.jsonl
bootstrap_intervals.jsonl
calibration_split.json
calibration_summary.json
holdout_summary.json
graphs.jsonl
stress_summary.json
summary.json
COMPLETE / REJECTED
```

禁止：

- candidate generation；
- VAE cache 改写；
- model training；
- review verdict 自动转 label；
- threshold performance search；
- 双卡。

## 11.4 PA2-UPO 通过门槛

全部满足：

1. calibration/holdout scene disjoint；
2. 至少 8 个 calibration ties 可比较；
3. 10 个 holdout ties 中 false strict ≤1；
4. high-confidence false strict = 0；
5. uncertain/both-invalid 不被 high-confidence strict；
6. 6 类 shortcut stress 全通过；
7. graph cycle rate低于预注册上限且 cycle condition 全 invalid；
8. threshold 对 bootstrap seed/benign perturbation 稳定；
9. 在未 review conditions 上至少有 16 strict conditions，才进入 prospective review；
10. 不通过降低门槛凑 strict yield。

如果 1–8 通过但 strict yield不足：

```text
status = blocked_candidate_yield
```

只允许 PA2-CAND 唯一 fallback。

---

# 12. PA3-KERNEL-04：代数与容量

只有 prospective oracle 人审通过且至少获得：

```text
24 strict conditions
24 tie conditions
```

才实现。

## 12.1 顺序

```text
synthetic tensor
→ 1 strict + 1 tie
→ 8 condition graphs
→ 16 train / 8 held-out condition graphs
```

## 12.2 单元测试

```text
zero adapter strict loss = log(2)
winner/loser swap flips logit
Davidson tie probability normalized
tie pushes margin to zero
incomparable produces no loss
shared sigma/noise deterministic
reference no grad
adapter state restored
tube equal mass
frame0 mask zero
empty support fail closed
cycle graph rejected
future-GT fail closed
SDPO safeguard bound
complement excludes boundary
condition-balanced sampler
```

## 12.3 Capacity gate

- finite；
- peak VRAM ≤22 GB；
- strict train margin 上升；
- tie margin 保持近零；
- winner absolute error不系统增加；
- outside-tube prediction drift ≤2%；
- frame-0 drift数值零；
- held-out strict direction ≥60%；
- dynamic exposure不下降；
- 不靠增加 steps 绕过失败。

一次允许的 locality fallback：

```text
提高 complement anchor，但保持预注册 gradient-RMS 比率
```

仍失败即停止 tube alignment。

---

# 13. PA4 单卡筛选

数据与更新预算完全匹配。

| ID | 方法 | 目的 |
|---|---|---|
| E0 | Frozen Base | 基准 |
| E1 | real-only SFT | 普通 fine-tuning |
| E2 | chosen-only SFT | 数据筛选基线 |
| E3 | scalar AWR | SHIFT-style baseline |
| E4 | independent-seed clip DPO | vanilla video DPO |
| E5 | common-prefix clip DPO | 结构 pair 基线 |
| E6 | temporal DenseDPO-style strict-only | segment credit baseline |
| E7 | tube strict-only + complement | localization ablation |
| E8 | tube strict + Davidson ties + safeguard + real anchor | DrivePO full |
| E9 | shuffled relation | 小预算诊断 |

正式筛选：

```text
1 training seed
100–500 updates
16 screen-eval clips
2 matched generation seeds
25 denoising steps
```

切双卡条件：

1. E8 independent CoTracker primary win rate ≥60%；
2. E8 不差于 E2/E3/E6/E7；
3. dynamic degree、residual speed/displacement、survival 无 >5% 退化；
4. background field与视觉非劣；
5. winner error稳定；
6. 人工盲审无 shortcut；
7. 单卡 profile 表明瓶颈是吞吐/实验并行，而非方法正确性。

未满足：

```text
不切双卡
不扩大关系数据
```

---

# 14. 最终评价

## 14.1 Primary endpoints

独立 CoTracker、固定 first-frame query protocol：

1. **common-support dynamic-track residual acceleration/curvature outlier rate**，越低越好；
2. **common-support survival / active coverage**，非劣。

## 14.2 Anti-shortcut

必须报告：

```text
VBench dynamic degree
active dynamic-query fraction
residual speed
net residual displacement
track survival
background motion field
freeze/slow/dropout attack win rate
```

## 14.3 Visual

```text
subject consistency
background consistency
temporal flicker
DINO identity distance to matched Base
LPIPS to matched Base
blind human quality preference
FVD only at >=256 clips
```

## 14.4 统计

```text
edge/window/query 不独立
generation seed 不独立
condition 内聚合
scene 内聚合
scene-level paired hierarchical bootstrap
10,000 samples
两个 primary endpoint Holm correction
invalid 不填 0
报告实际 n 与 coverage
```

## 14.5 正式 promotion

全部满足：

- oracle prospective review通过；
- primary dynamics CI改善；
- survival/coverage非劣；
- dynamic degree与运动暴露无 >5% 退化；
- 两 training seeds同向；
- 至少两 generation seeds同向；
- full超过 chosen-SFT、AWR、DenseDPO-style、tube strict-only；
- CoTracker与人工同向；
- P-UNC不是唯一改善指标；
- 结果不由单一 scene/branch/outlier主导。

---

# 15. 单卡与双卡策略

## 单卡阶段

以下全部单卡：

```text
PA2-UPO
prospective review material
PA3 kernel/capacity
PA4 screening
```

每阶段记录：

```text
seconds/video
seconds/score
seconds/update
peak VRAM
disk bytes/video
disk bytes/graph
```

## 双卡触发

只有 PA4 通过后，由用户停机切双卡。

双卡优先用于：

```text
GPU0 / GPU1 candidate or evaluator sharding
两个 training seed 并行
方法与 baseline 并行
```

普通 DDP 不增加单样本 24GB 显存；不用于解决 OOM。

---

# 16. 工程与 Git 要求

开始执行前：

```bash
git status --short
git rev-parse HEAD
git log -8 --oneline

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/motionproj
export PYTHONPATH=.

PYTHONPATH=. pytest -q
```

注意：最近一次 v3 提交只改文档、未运行代码测试，因此 PA2-UPO 开发前必须记录一次完整 baseline test。

每个逻辑任务独立 commit：

```text
docs(preference): 固化 DrivePO v4 门禁与结论边界
feat(preference): 增加 sibling 共同 query 跟踪
feat(preference): 构建 selective partial-order oracle
test(preference): 覆盖校准与 shortcut stress
research(preference): 完成 PA2-UPO retrospective audit
```

提交前：

```bash
PYTHONPATH=. pytest -q
git diff --cached --check
git diff --cached
```

不自动 push。

---

# 17. 明确停止条件

## Oracle

- old tie holdout false strict >1；
- high-confidence false strict >0；
- query fallback仍存在；
- dropout/slow/freeze 可获胜；
- camera mismatch被误判 dynamic winner；
- strict direction对 seed/perturbation不稳定；
- 只能靠降低阈值获得 yield。

## Candidate

- earlier-fork 仍主要 human tie；
- earlier-fork 导致 scene/identity mismatch；
- strict yield <10%；
- strict主要来自质量差异。

## Kernel

- tie objective不稳定；
- winner error上升；
- outside drift >2%；
- frame0不保持；
- held-out direction <60%；
- one-step无完整 rollout迁移。

## Screening

- 不超过 chosen-SFT/AWR/DenseDPO-style；
- 只在训练 scorer改善；
- 独立 evaluator或人工反向；
- motion collapse；
- 两 seeds不一致。

---

# 18. 可接受的论文贡献

只有全部实验成立后才允许写：

1. 一种用于 driving sibling futures 的 **common-support selective partial-order oracle**，显式控制 track support mismatch、measurement uncertainty 与 human ties；
2. 一种无需 future GT 的 **camera-nuisance-compensated dynamic-track preference representation**；
3. 一种将 strict/tie relations 映射到 **dynamic-track tubes** 的 tie-aware、winner-preserving diffusion alignment；
4. 一套以 false-strict calibration、shortcut attacks、独立 tracker 和 scene-level statistics 为核心的 driving video preference protocol。

不得写：

```text
首次在驾驶视频做 DPO
actor ground-truth alignment
真实物理定律优化
DenseDPO for driving
LocalDPO with object mask
自动 scorer 取代人工
```

---

# 19. 当前唯一下一步

现在只执行：

```text
PA2-UPO-03B
```

具体顺序：

```text
读取事实源与完整 baseline tests
→ 冻结 22 ties 的 12/10 scene-level split
→ 新增 no-fallback common first-frame query protocol
→ 输出 raw paired tracks
→ common support
→ robust affine background compensation
→ paired component intervals
→ measurement ROPE
→ split-conformal strict threshold
→ old tie holdout audit
→ shortcut stress
→ apply to unreviewed sibling conditions
→ 决定 prospective review / candidate fallback / reject
```

在完成前禁止：

```text
DPO/AWR trainer
LoRA训练
candidate扩量
双卡切换
旧53 pairs训练
自动填写人工review
```

---

# 20. 外部研究依据

- [Diffusion Model Alignment Using Direct Preference Optimization, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Optimization_CVPR_2024_paper.html)
- [VideoDPO: Omni-Preference Alignment for Video Diffusion Generation, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_VideoDPO_Omni-Preference_Alignment_for_Video_Diffusion_Generation_CVPR_2025_paper.html)
- [DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/fa9755043814e7f08d859a286bb83c35-Abstract-Conference.html)
- [Mind the Generative Details: Direct Localized Detail Preference Optimization for Video Diffusion Models, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_Mind_the_Generative_Details_Direct_Localized_Detail_Preference_Optimization_for_CVPR_2026_paper.html)
- [Diffusion-SDPO: Safeguarded Direct Preference Optimization for Diffusion Models](https://arxiv.org/abs/2511.03317)
- [On Extending Direct Preference Optimization to Accommodate Ties](https://arxiv.org/abs/2409.17431)
- [Uncertainty-Penalized Direct Preference Optimization](https://arxiv.org/abs/2410.20187)
- [SHIFT: Motion Alignment in Video Diffusion Models with Adversarial Hybrid Fine-Tuning](https://arxiv.org/abs/2603.17426)
- [VideoGPA: Distilling Geometry Priors for 3D-Consistent Video Generation](https://arxiv.org/abs/2601.23286)

实现必须以论文原文和官方代码为准；通用组件必须准确归因，不得包装成 DrivePO 原创。
