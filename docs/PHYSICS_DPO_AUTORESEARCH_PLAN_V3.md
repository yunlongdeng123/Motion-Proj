# Motion-Proj Physics Preference Alignment 自动研究计划 v3

> **工作方法名**：**DrivePO — Uncertainty-Calibrated Sibling Partial Orders for Dense Safeguarded Driving Video Alignment**。
>
> **文档定位**：本文件取代 `PHYSICS_DPO_AUTORESEARCH_PLAN_V2_AC_REVISED.md` 的未来排程；v2、`CVPR2027_PLAN.md`、`EXPERIMENTS.md` 和所有历史 run 继续作为事实源，不重写既有结论。
>
> **决策日期**：2026-07-15。
>
> **计划基线**：clean commit `eddb9e92904bd2f89a98fbd51a1a7f1db8336e4a`。
>
> **正式人工标注**：`/root/autodl-tmp/runs/autoresearch-pa2-pair-expanded-s20260715-v1/reviews.jsonl`，48/48 已完成。
>
> **当前状态**：PA2 machine gate 通过，但 v2 的 P-UNC 单点估计标签器未通过人工可信性门槛；现有 53 个 global pair 和 local labels **不得直接训练**。结构对齐 sibling 资产保留，下一步只做共同轨迹支持、风险校准和偏序重建。
>
> **状态词**：`pending / running / blocked / done / rejected`。
>
> **硬件边界**：PA2–PA4 继续使用单张 RTX 4090 24 GB；PA4 单卡筛选通过前不切双卡。
>
> **投稿目标**：CVPR 2027。论文只主张真实 RGB 驾驶视频中的短时、camera-compensated image-plane motion consistency，不把图像平面量写成真实世界牛顿动力学。

---

# 0. 执行摘要与研究决策

## 0.1 48 条人审改变了什么

正式 review 的聚合结果是：

| 项目 | 结果 |
|---|---:|
| 完成数 | 48/48 |
| Stage-B verdict | 23 tie / 13 B better / 7 A better / 3 uncertain / 2 both invalid |
| machine 与 human 同时 decisive | 8 |
| agreement | 4/8 = 50% |
| 95% Wilson lower bound | 0.2152 |
| scorer-chosen low-motion collapse | 0 |
| scorer-chosen catastrophic quality failure | 2 |

最关键的分 constructor 结果：

| Constructor | 人工结果 | 机器结果 | 结论 |
|---|---|---|---|
| P0 independent | 12/12 人工 decisive，主要由明显画质/相机运动失败区分 | 4 decisive、4 abstain、4 invalid | 可作为 vanilla baseline，不能作为结构对齐主数据 |
| P1 common-prefix | **22 tie、1 uncertain、1 both invalid，0 decisive** | **13 a_wins、11 b_wins** | 结构对齐成立，但当前 physics label 全部缺少人工可辨识性 |
| P2 Base re-noise | 8 decisive、1 tie、2 uncertain、1 both invalid | 5 decisive、4 abstain、3 invalid | 人工多在比较伪影/形变，不是干净的物理偏好 |

run 的 `summary.json` 按 v2 预注册聚合器原样记录为 `needs_more_reviews`，因为 decisive agreement 分母不足 24；v3 不覆盖或改写该终态。但研究决策不能把它解释成“只差更多 review”。当前 P1 的问题不是分母不足，而是：

> 自动 scorer 在人工认为不可区分的 sibling 上产生了高置信方向，继续 DPO 会把测量噪声和 support mismatch 放大成训练信号。

v3 作出以下决定：

1. `PA2-PAIR-03` 的**候选生成与结构资产保留**；
2. v2 的“独立候选打分 → projection energy margin → global/local winner”标签配方标为 `rejected`；
3. 冻结现有 48 条人审为 scorer calibration/diagnostic 数据，不进入训练；
4. 先在已有 sibling RGB 上重建共同轨迹支持的 uncertainty-aware partial order；
5. 只有新的 P1-only 人工校准通过，才解锁 DPO/AWR/kernel；
6. 现阶段不生成更多大规模候选、不训练、不切双卡。

## 0.2 v3 的核心取舍

不采用以下“大步子”：

- 不训练大型 reward model 或 VLM judge；
- 不引入 3D foundation model 作为训练标签器；
- 不做完整 sampling-chain backprop、PPO 或 GRPO；
- 不立即迁移第二 backbone；
- 不恢复已失败的 RGB projection target；
- 不同时搜索 fork、strength、LoRA rank、loss weight 和 timestep。

只增加三个相互闭环、可在现有资产上验证的模块：

1. **Common-Support Sibling Graph**：同一去噪前缀的 sibling set，使用相同 first-frame query ID 和共同可见 support 比较，不再比较两个各自采样的轨迹集合；
2. **Uncertainty-Calibrated Residual Partial Order**：将相机背景运动视为 nuisance，比较 actor residual 的区间估计；输出 strict / tie / incomparable，而不是强制二元 winner；
3. **Tube-Localized Safeguarded Alignment**：只在可信 query-time tube 上做 dense preference update，同时保留 tie、winner、背景和真实视频锚点。

这三个模块分别回答：

```text
可信：是否有共同证据支持方向，而非单个高分？
可局部化：是哪条轨迹、哪个时间窗、哪个空间 tube 出错？
不可投机：是否能靠少动、丢 track、改相机运动或毁掉画质获胜？
```

---

# 1. 已完成里程碑（简表）

以下只保留仍影响 v3 决策的里程碑；旧 endpoint projection、V1/V2 调参和已关闭工程 bug 的详细过程不再重复。

| ID | 状态 | 日期 / commit | 结论与证据 |
|---|---|---|---|
| P2-V2-PILOT-03 | rejected | 2026-07-13 / `9dd4c88` | endpoint correction 单 pair 可学，但共享 temporal LoRA 无法同时满足 locality/preserve；证据见 `/root/autodl-tmp/runs/p2-v2-pilot/` |
| PA0-REVIEW-00 | done | 2026-07-15 / `16b6975` | P-UNC 与 CoTracker3 既有人审完成；证据 `/root/autodl-tmp/runs/autoresearch-pa0-review-s20260715-v1` |
| PA0-SCENE-SPLIT-01 | done | 2026-07-15 / `5713267` | scene-level split 无泄漏；fingerprint `e525edf33bcfec169c0077d2eb2e528d953dbc9930e771c803c889a32983c73a` |
| PA1-HORIZON-01 | done | 2026-07-15 / `57987b0` | 冻结 14 frames、25 denoising steps；claim 仍限 short horizon |
| PA1-BRANCH-02 | done | 2026-07-15 / `0e11515`、`002b616` | common-prefix sibling machine 3/4 condition、6/8 group；人工结构盲审 8/8 `same_scene` |
| PA2-PAIR-03 machine | done | 2026-07-15 / `eddb9e9` | 120 conditions，53 valid global pairs，52 个含 non-tie local segment；三 constructor schema 完整 |
| PA2-PAIR-03 human | rejected | 2026-07-15 / 正式 48 条人审 | run 原状态保留为 `needs_more_reviews`；v3 将旧标签配方判为 rejected。P1 为 0/24 human-decisive；总 agreement 4/8，Wilson 0.2152，2 个 scorer-chosen catastrophic failure |
| PA2-DIAG-03A | done | 2026-07-15 / 本 v3 | 确认 support mismatch、未校准 confidence 和弱 quality guard 是首要根因；下一步 `PA2-UPO-03B` |

历史负结论继续有效：

- SVD 不接收 future ego control，正式偏好不得读取 future GT ego/track；
- self-estimated static RGB projection V1 已被人工拒绝；
- P-UNC 可作为待校准的相对 motion diagnostic，不能作为自证正确的最终 evaluator；
- CoTracker3 继续只作独立 evaluator，不反向参与主标签器拟合。

---

# 2. v2 标签器的可复现实质问题

当前 `motion_proj/preference/pair_scoring.py` 的设计并非线性加权 reward，但仍有四个会造成伪高置信的缺口。

## 2.1 候选之间没有共同测量 support

当前每条视频独立选择 query、独立存活、独立产生 projection points，再比较两个均值：

\[
\bar E_A=\frac{\sum_{q\in Q_A}e_q}{|Q_A|},\qquad
\bar E_B=\frac{\sum_{q\in Q_B}e_q}{|Q_B|}.
\]

但通常 (Q_A\neq Q_B)。reviewed P1 的 24 对中：

- projection-point ratio 中位数为 `1.213`；
- 7/24 大于 `1.5`；
- 3/24 大于 `2.0`；
- 最大为 `18.7`；
- 最大 mismatch 的 case 被人工判为 `both_invalid`，机器 pair confidence 仍为 `0.9869`。

这不是可靠的 paired comparison。低 energy 可能只表示更难的轨迹已经丢失。

## 2.2 `scorer_confidence` 不是偏好方向置信度

现有 pair confidence 由 track confidence、relative energy margin 和 saturation 相乘。reviewed P1 中：

- pair confidence 中位数为 `0.9251`；
- 14/24 不低于 `0.9`；
- 但人工 strict preference 为 0/24。

因此该数值只能描述内部测量条件，不能解释为“机器 winner 正确的概率”。

## 2.3 quality guard 只覆盖 finite / saturation

正式人审的主要失败原因是：

```text
blur_or_artifact
geometry_deformation
temporal_jitter
camera_motion_inconsistent
identity_switch
```

而当前自动 quality component 主要记录 finite、min/max 和 saturation fraction。它不能拦截融化、重影、曝光闪烁、透视拉伸或 identity failure，导致 2 个机器 chosen 为 catastrophic failure。

## 2.4 max-confidence 选择放大 winner's curse

当前每 condition 在两个 antithetic pair 中选择最大 `pair_confidence`。局部窗口又将 relative margin 直接映射到 `[0,1]`，可轻易饱和。多重比较、窗口相关性和选择后的不确定性均未计入。

v3 必须以 condition-level sibling graph 做 simultaneous uncertainty control，不能把每条 edge 当成独立样本。

---

# 3. 论文问题与准确 claim

## 3.1 核心问题

> 对同一驾驶场景的模型 sibling futures，能否在不使用 future GT 的前提下，构造具有共同轨迹证据、可选择性 abstain、能定位到 actor-time tube 且不能通过少动/丢 track 获利的局部物理偏序；并将这些不完整偏序稳定地注入视频扩散模型，在独立 tracker 和人工评测上改善 actor motion consistency，同时保持 camera/background motion、运动量、身份和画质？

## 3.2 三个操作性定义

### 可信（trustworthy）

一条 strict edge 必须同时满足：

- same-condition、same-prefix、same first-frame query IDs；
- 只在 pair 的共同可见 support 上计算；
- paired block bootstrap 的方向置信区间通过；
- condition 内多边比较经过 simultaneous correction；
- margin 超过由 human-tie cases 校准的 perceptual ROPE；
- 在独立 P1-only calibration set 上达到预注册 selective precision；
- scorer 与 evaluator provider 分离。

### 可局部化（localizable）

每个关系必须回溯到：

```text
condition
candidate pair
query/track ID
actor/background stratum
4-frame window
spatial tube mask
violation component
uncertainty interval
```

只保存 clip-level winner 不合格。

### 不可投机（anti-shortcut）

这里不声称形式化证明“任何攻击都不可能”，而采用可证伪的 operational contract：

- freeze / frame-repeat / time-slow candidate 不能成为 strict winner；
- 更低的 active-track coverage 或 survival 不能降低 penalty；
- motion exposure、net displacement 和 active residual speed 必须非劣；
- camera/background field 差异超界时关系为 incomparable；
- catastrophic quality 只会变成 invalid，不参与 physics ranking；
- 训练后 dynamic degree、active motion 和 track survival 必须非劣。

## 3.3 不声称

- 不声称恢复真实 3D acceleration、force 或 collision dynamics；
- 不声称单目图像能唯一分解 ego 与 object motion；
- 不声称自动 scorer 取代人工；
- 不声称 generic tie-DPO、uncertainty DPO、region DPO 或 SDPO 是本项目原创；
- 不声称解决长时世界模型或闭环规划；
- 未完成第二 backbone 前不声称架构通用性。

---

# 4. Reviewer-first 威胁模型

| 预期质疑 | v3 回答 | 必须给出的证据 | 失败即止损 |
|---|---|---|---|
| “只是 DenseDPO + driving score” | 训练单位是 human-risk-calibrated sibling partial-order graph，不是强制 segment winner；物理差值只在 common track support 上定义 | scalar / unpaired / temporal-only / full graph ablation | full 不超过 DenseDPO-style baseline |
| “只是 LocalDPO 的 track mask 版本” | LocalDPO 使用 real-positive/local-corruption 和已知 corruption mask；本项目使用 self-rollout siblings、共同 actor correspondence、strict/tie/incomparable 关系 | same data 上 frame-window mask 与 actor-tube mask 对照 | tube localization 无额外收益或无更低 outside drift |
| “P-UNC 不是真物理” | 准确命名为 camera-compensated image-plane residual consistency；projection energy 仅为多个区间分量之一 | real-video sanity、track-space stress、human calibration、CoTracker/GeCo secondary | 只有 P-UNC 自评改善 |
| “你们自己的 48 条人审已经否定 scorer” | 将该负结果作为方法动机；旧标签器明确 rejected，新方法必须首先消除 24 个 P1 false strict edges | old-review retrospective calibration + new P1-only prospective review | old ties 仍被高置信判 strict |
| “模型只是少动” | activity-matched comparability、motion floor、freeze stress 和主动轨迹 support 共同限制 | dynamic degree / residual speed / displacement / survival 非劣；freeze attack 0 win | 任一训练 seed 出现系统性低运动收益 |
| “camera motion 与 actor motion 混在一起” | 先拟合 candidate-specific robust background field，再比较 common-query actor residual；background field 不可比则 abstain | ego-dominant / actor-dominant / mixed strata；common camera transform invariance | actor 指标改善只来自 camera/background 变化 |
| “偏序置信区间是挑阈值” | tie ROPE 来自冻结 human-tie calibration；bootstrap seed、cluster、simultaneous correction 全预注册 | calibration curve、coverage-risk curve、固定阈值外测 | 需按 rollout 指标反复改阈值 |
| “DPO 通过恶化 loser 获益” | 使用 winner-preserving safeguard，并单独报告 winner absolute error | vanilla / fixed-scale / adaptive safeguard | margin 上升但 winner error 系统增加 |
| “局部 loss 仍会全局泄漏” | actor tube loss + outside-tube reference/real anchor；capacity gate 先看 prediction drift，再看 rollout | inside gain / boundary / outside / frame-0 drift | 一次预注册 anchor-gradient fallback 后仍泄漏 |
| “单一 SVD 太弱” | 先在单卡闭合方法正确性；正式主结果后才做最小第二-backbone transfer | frozen oracle + 32-condition transfer pilot | 主方法未成立前不扩大工程范围 |

---

# 5. 与最近邻工作的边界

| 工作 | 已覆盖内容 | v3 不可重复声称 | DrivePO 必须保留的差异 |
|---|---|---|---|
| Diffusion-DPO，CVPR 2024 | diffusion ELBO 近似的离线 DPO | diffusion DPO loss | driving sibling graph、common support、局部残差偏序 |
| VideoDPO，CVPR 2025 | 自动多维评分、pair reweighting | 自动打分和大 margin pair | 不用 scalar total score；选择性偏序与硬约束 |
| DenseDPO，NeurIPS 2025 | 结构相似 pair、segment labels、低运动偏置 | structural pair、temporal dense label | self-rollout sibling set；common-query actor residual；tie/incomparable；track tube |
| LocalDPO，CVPR 2026 | local corruption pair、spatiotemporal region-aware DPO | 局部 mask DPO | 非合成 corruption；物理 correspondence support；风险校准偏序 |
| VideoGPA，2026 | 几何 foundation model 产生 dense DPO signal | 自动 geometry preference | 无 3D foundation model；动态 actor residual 与低运动防投机 |
| SHIFT，2026 | pixel-motion reward、AWR、hybrid SFT、reward-hacking 缓解 | acceleration/flux reward + AWR | AWR 为强 baseline；主方法是区间偏序而非 scalar advantage |
| Diffusion-SDPO，2025 | winner-preserving loser gradient scale | safeguard 本身 | 作为稳定组件；与 track-tube strict/tie graph 联合 |
| Tie-aware DPO，2024 | Rao-Kupper / Davidson tie likelihood | “DPO 支持 tie” | tie 来源于 human-calibrated driving sibling equivalence class |
| Uncertainty-Penalized DPO，2024 | 对不确定 pair 降权 | uncertainty penalty | common-support paired uncertainty、selective abstention、condition-level simultaneous control |
| GeCo，2025/2026 | motion+depth 的静态场景几何 inconsistency map | dense geometry map | 只作独立 secondary；主标签关注驾驶 actor residual |
| ADGVE，2025 | driving video failure taxonomy 与综合 evaluator | driving scalar evaluator | 不将语义、画质和物理压成一个训练分数 |

论文的新意不来自其中任一通用组件，而来自以下闭环：

\[
\text{controlled sibling set}
+\text{paired common support}
+\text{ego-residual interval order}
+\text{track-tube strict/tie alignment}.
\]

---

# 6. 方法：DrivePO

## 6.1 Common-prefix sibling set

对 condition $c$，保留 exact Base guard：

\[
X^0=G_{\theta_0}(\xi,c).
\]

在冻结 fork 边界共享前缀 $z_{t_f}^{\mathrm{prefix}}$，生成 $K=4$ 个 sibling：

\[
X^i=G_{\theta_0}^{t_f\rightarrow0}
\left(z_{t_f}^{\mathrm{prefix}}+\delta_i,c\right),
\qquad i=1,\ldots,K.
\]

第一版不增加 candidate 数，先复用现有 120 conditions × 4 siblings。每个 condition 枚举最多 $\binom{4}{2}=6$ 条边，而不是只比较两个固定 antithetic pair。

约束：

- Base exact rerun；
- first frame exact；
- same scheduler / condition / prefix / decode；
- future-frame distance 位于 rerun noise floor 与 independent-seed 上界之间；
- structure mismatch 直接使整条 edge invalid；
- P0 independent 和 P2 re-noise 仅作 baseline，不混入主训练 graph。

不得按 physics score 先选择“最有利方向”再估计普通置信区间。所有 edge 的 uncertainty 必须在 condition family 内联合计算。

## 6.2 Paired common-query tracking

从共同 first frame 或 Base guard 一次性冻结 query set：

\[
Q_c=Q_c^{\mathrm{bg}}\cup Q_c^{\mathrm{dyn}}\cup Q_c^{\mathrm{fg}}.
\]

所有 sibling 使用相同 query ID、坐标和 stratum。对候选 $i,j$ 和窗口 $s$，共同 support 定义为：

\[
\Omega_{ij,s}=\{(q,t):
v^i_{q,t}=v^j_{q,t}=1,
\operatorname{FB}^i_{q,t}\le\tau_{fb},
\operatorname{FB}^j_{q,t}\le\tau_{fb},
t\in s\}.
\]

要求：

- projection / curvature / survival 都在同一 $\Omega_{ij,s}$ 上成对计算；
- 任一侧 track 丢失不记为 0，也不能因缩小 denominator 获益；
- 保存 common support count、effective sample size、spatial cluster count 和每帧 coverage；
- support 不足时输出 `incomparable_support`；
- 独立 CoTracker evaluator 使用同一 first-frame query protocol，但不读取 RAFT/P-UNC label。

这是 v3 相对 v2 最先必须关闭的测量问题。

## 6.3 Driving-specific ego–actor residualization

单目驾驶视频中的点运动同时包含 camera/background motion、actor motion、遮挡和 tracker noise。对每个候选独立从 (Q^{\mathrm{bg}}) 拟合 robust background field (g_t^i)，仅把它当 nuisance estimate：

\[
r^i_{q,t}=p^i_{q,t}-g_t^i(p_{q,0}).
\]

不得读取 dataset future ego pose。背景模型只允许预注册的 affine/homography family 和 robust fitting；model family 选择不能根据 pair winner 调节。

一条 edge 可比较前先要求：

\[
D_g(i,j)\le\epsilon_g,
\]

即 sibling 的 camera/background field 位于等价区间。超过时不判断哪条“更物理”，而标为 `incomparable_camera`。

在共同 actor support 上计算以下向量，不做任意加权和：

\[
\boldsymbol\phi_{i,s}=
\left[
E_{\mathrm{curv}},
E_{\mathrm{acc}},
E_{\mathrm{coh}},
E_{\mathrm{surv}}
\right]_{i,s}.
\]

其中：

- (E_{\mathrm{curv}})：residual velocity direction/curvature 的 robust outlier；
- (E_{\mathrm{acc}})：camera-compensated image-plane acceleration outlier；
- (E_{\mathrm{coh}})：first-frame 邻域 query 的 residual velocity coherence，用于发现局部融化/形变；
- (E_{\mathrm{surv}})：共同 query 的 early death、occlusion inconsistency 和 identity-risk penalty；
- jerk 只作 secondary diagnostic，14 帧下不作为单独 winner 依据。

运动暴露向量单独保存：

\[
\mathbf a_{i,s}=
[\text{active fraction},\text{residual speed},\text{net residual displacement}].
\]

低 violation 不能抵消 motion exposure 下降。

### 6.3.1 Quality 只作硬否决，不参与 physics winner

v2 人审表明 finite/saturation 不足以拦截融化、重影、曝光闪烁和透视拉伸。v3 不把这些质量量再加进 scalar reward，而定义独立的 `quality_valid`：

```text
Base guard 与 candidate 均 finite
saturation / black-frame fraction 合法
background-compensated luminance jump 合法
background-compensated temporal feature jump 合法
common-query coverage / survival 合法
无 condition-level catastrophic flag
```

其中 temporal feature 只使用冻结的轻量 DINO/LPIPS diagnostic；阈值由真实 nuScenes clips 与 Frozen Base guard 的分布预注册，再用已有人审 catastrophic cases 做一次 fail-closed 检查，不根据 physics winner 或 tuned rollout 调节。

关系规则：

- 任一侧 `quality_valid=false`：edge 为 `incomparable_quality`；
- 两侧均 valid、但 quality difference 超过等价带：edge 仍为 incomparable；
- quality 永远不能单独把一侧升级为 physics winner；
- P0/P2 的大伪影偏好不能补入 P1 主训练 graph。

## 6.4 Paired uncertainty interval

对共同 query 的 component difference：

\[
d^{(m)}_{q,s}=\phi^{(m)}_{j,q,s}-\phi^{(m)}_{i,q,s}.
\]

使用 spatial-cluster × temporal-block paired bootstrap，避免把相邻 query/frame 当独立样本。每条 edge 保存：

```text
point estimate
bootstrap lower / upper bound
effective query count
spatial cluster count
block length
bootstrap seed
failure / invalid fraction
```

同一 condition 的最多 6 条 edge 使用 max-statistic 或 Holm correction 控制多重比较。禁止先选最大 margin 再报告未经选择修正的 CI。

## 6.5 Human-tie-calibrated ROPE

现有 22 个 P1 human ties 不丢弃。使用新 common-support scorer 重算后，将它们作为冻结 calibration set，估计“人眼不可稳定区分”的 practical-equivalence region：

\[
\operatorname{ROPE}_m=[-\delta_m^{\mathrm{tie}},\delta_m^{\mathrm{tie}}].
\]

要求：

- calibration condition 不进入训练；
- threshold 只根据 human verdict 与新 scorer difference 冻结；
- 不根据 tuned-model rollout 指标调整；
- 报告 calibration coverage-risk curve；
- old P1 的 1 uncertain、1 both-invalid 只能用于 invalid/abstain 检查，不用于 tie quantile。

## 6.6 Strict / tie / incomparable partial order

对窗口 (s)，候选 (i,j) 首先必须满足 comparability：

```text
same condition / prefix / first-frame queries
both quality-valid
common support sufficient
camera/background field equivalent
motion exposure non-inferior
no catastrophic failure flag
```

然后输出三种关系。

### Strict edge：\(i\succ_s j\)

只有当：

1. 至少一个 primary component 的 simultaneous lower bound 超过对应 tie ROPE；
2. 其余 primary component 没有超过 non-inferiority 上界的反向退化；
3. winner 的 active fraction、residual speed、displacement、survival 和 visual guard 均非劣；
4. calibrated selective risk 通过；
5. edge 不造成 condition graph 的显著 cycle。

### Tie / equivalence：\(i\sim_s j\)

共同 support 充分，且所有可用 primary component 的 CI 位于 ROPE 内；两侧均有效。tie 不是“没有标签”，而是禁止模型任意拉开 margin 的约束。

### Incomparable：\(i\parallel_s j\)

包括：

- component conflict；
- support 不足；
- camera field 不可比；
- quality invalid；
- uncertainty interval 过宽；
- 关系会形成无法由不确定性解释的 cycle。

incomparable 不进入 preference loss。

每个 condition 先把 tie nodes 合并为 equivalence classes，再对 strict edges 做 cycle audit 和 transitive reduction。训练按 condition 均匀采样，不能因 6 条 edge 让某 condition 获得 6 倍权重。

## 6.7 Track-time tube localization

对 strict/tie edge 的共同 query 和 4-frame window，分别在两条视频中构建语义对应而非像素相同的 tube：

\[
M^u_{e,s}(t,h,w)
=\sum_{q\in\Omega_{e,s}}
w_{q,t}^u K\!\left((h,w)-p_{q,t}^u\right),
\quad u\in\{i,j\}.
\]

要求：

- frame 0 mask 强制为 0；
- visibility / uncertainty 决定 soft weight；
- tube 半径固定并只作一次离散 capacity 对照；
- 保存 boundary band 和 complement；
- 没有 common query 的窗口不能产生 mask；
- preference component 必须能回溯到 tube 内 query ID。

## 6.8 新 schema

每个 condition graph 至少保存：

```text
condition_id
scene_id
candidate_ids
base_guard_id
prefix_hash
first_frame_query_set_hash
query_strata
pair_edges
common_support_by_edge_window
background_field_by_candidate
background_equivalence_by_edge
component_point_estimates
component_confidence_intervals
motion_exposure_vectors
quality_guards
relation = strict / tie / incomparable
relation_reason
simultaneous_correction
calibration_version
calibration_split
track_tube_paths
scorer_fingerprint
uses_future_gt = false
```

旧 `preferences.jsonl` 和 `segments.jsonl` 保留只读，不原地升级 schema。

---

# 7. Dense safeguarded diffusion alignment

## 7.1 Tube-local denoising error

对 relation edge (e=(w,l,s))，winner/loser 使用相同 diffusion timestep 和 noise。调用既有已测试的 `model_output_from_x0`，不手写新参数化公式。

\[
\ell_\theta^u(e)=
\frac{
\sum M^u_{e,s}\,
\|v_\theta(z_\tau^u,\tau,c)-v^{*,u}\|_2^2
}{
\sum M^u_{e,s}+\varepsilon
},
\quad u\in\{w,l\}.
\]

reference-relative margin：

\[
d_\theta(e)=
[\ell_\theta^l-\ell_{\mathrm{ref}}^l]
-[\ell_\theta^w-\ell_{\mathrm{ref}}^w].
\]

## 7.2 Strict-edge loss

\[
\mathcal L_{\mathrm{strict}}
=-
\frac{
\sum_{e\in\mathcal E_\succ}q_e
\log\sigma(\beta d_\theta(e))
}{
\sum q_e+\varepsilon
}.
\]

(q_e) 是 capped selective confidence，只能降低不确定 edge 的作用，不能把少数大 margin outlier 放大到超过 1。

## 7.3 Tie constraint

对 (e\in\mathcal E_\sim)，第一版使用简单的 robust zero-margin constraint：

\[
\mathcal L_{\mathrm{tie}}
=
\frac{1}{|\mathcal E_\sim|}
\sum_e
\operatorname{Huber}
\left(
\frac{d_\theta(e)}{\tau_{\mathrm{tie}}}
\right).
\]

Davidson / Rao-Kupper tie likelihood 作为对照，不作为原创点。tie loss 的作用是防止训练把 v2 中大量人眼等价 sibling 任意分开。

## 7.4 Winner-preserving safeguard

strict edge 使用 Diffusion-SDPO 的 output-space adaptive loser scaling。每 step 必须记录：

```text
winner policy absolute error
winner reference error
loser policy absolute error
loser reference error
raw / safeguarded margin
safe loser scale
```

晋级要求不是“margin 上升”，而是 winner tube error 不系统增加。

该 safeguard 是借鉴组件，不作为论文独立贡献。

## 7.5 Complement 与 real anchor

总损失：

\[
\mathcal L=
\mathcal L_{\mathrm{strict}}
+\lambda_{\mathrm{tie}}\mathcal L_{\mathrm{tie}}
+\lambda_{\mathrm{comp}}\mathcal L_{\mathrm{comp}}
+\lambda_{\mathrm{real}}\mathcal L_{\mathrm{real}}.
\]

其中：

- `comp` 在 tube complement 上保持 reference prediction；
- `real` 使用真实 nuScenes 视频的普通 denoising SFT；
- preference / comp / real 共享 timestep，shape 允许时共享 noise；
- 两个 anchor weight 只按 gradient RMS 区间校准，不按 rollout metric 搜索；
- frame 0 单独报告并要求数值零漂移；
- 先固定 temporal-only rank-16 LoRA，不改容量。

预注册 gradient ratio：

\[
0.5\le
\frac{\|\lambda_{\mathrm{anchor}}g_{\mathrm{anchor}}\|}
{\|g_{\mathrm{preference}}\|+\varepsilon}
\le2.
\]

如果一次正式 capacity test 仍出现 outside drift >2%，只允许一个 fallback：将 preference gradient 投影到不一阶增加 anchor loss 的半空间。不得同时调大 anchor、降 LR、加 rank 和延长步数。

## 7.6 Sampling 与 exposure

训练采样顺序：

```text
uniform condition
→ strict / tie relation type
→ graph edge
→ local window / tube
→ fixed noise-bank draw
```

必须记录：

```text
condition exposure
strict-edge exposure
tie-edge exposure
query/tube exposure
pair p95 exposure
```

同一 condition 的组合边不能被视为独立样本或独立统计单位。

---

# 8. 如何证明解决 driving motion entanglement / 低运动偏置

两者至少有一个形成强主结论，另一个必须是非劣和 stress evidence。优先主张 **driving-specific ego–actor motion entanglement**，低运动偏置作为强安全结论；原因是 DenseDPO 已直接讨论低运动偏置，而 ego–actor residual + common support 更具领域差异。

## 8.1 Oracle stress suite（训练前）

只在 track space 或只读 RGB diagnostic 上构造，不作为训练 pair：

| Stress | 应有关系 |
|---|---|
| 对 A/B 同时施加相同 global camera transform | strict 方向保持或 abstain；actor residual difference 近似不变 |
| 只对一侧 actor residual 注入局部 jitter | 干净侧 strict win，且定位到对应 tube/window |
| time freeze / repeated frames | frozen 侧绝不能因 acceleration 变小获胜 |
| uniform slow-down | 若 motion exposure 不匹配，应 incomparable，而非 slow side win |
| track dropout / early death | dropout 侧 survival penalty，不允许 denominator 变小获益 |
| identity swap / local deformation | coherence/survival component 响应；单纯 acceleration 不足的 ablation 失败 |
| background-only perturbation | actor component不应伪改善；camera comparability 应拦截 |

通过标准：

- freeze / slow / dropout attack 的错误 winner 数为 0；
- actor jitter localization precision 不低于 90%（合成 query-time support）；
- common camera transform 后 strict direction flip rate 不高于 5%；
- component ablation 能显示 acceleration-only 无法覆盖 deformation/identity stress。

## 8.2 Model-level entanglement evidence

使用 Frozen Base 的独立 CoTracker 结果预先把 condition 分为：

```text
ego/background-dominant
actor-residual-dominant
mixed
low-observability
```

分层只由 Base 冻结，不根据 tuned model 重分桶。完整方法必须表现为：

- actor-dominant / mixed 中 actor residual curvature/acceleration 改善；
- ego/background-dominant 中 background field residual 非劣；
- common support、survival、active fraction 不下降；
- outside actor tube 的视觉/运动指标非劣；
- generic acceleration reward baseline 更容易混淆 global camera motion。

## 8.3 Low-motion anti-shortcut evidence

必须同时报告：

- VBench dynamic degree；
- common-support active fraction；
- residual speed；
- net residual displacement；
- track survival / length；
- near-static output rate；
- motion histogram 与 worst 10%。

任何 physics metric 改善若伴随以下任一项，均不晋级：

```text
dynamic degree 下降 >5%
active fraction 下降 >5 percentage points
residual displacement 下降 >5%
near-static collapse 显著增加
human 认为 chosen 通过少动作弊
```

---

# 9. 前向里程碑与门禁

| ID | 状态 | 任务 | 晋级门槛 | 失败动作 |
|---|---|---|---|---|
| PA2-DIAG-03A | done | 48-review 根因分析 | 明确旧标签器与保留资产 | 见本 v3 |
| PA2-UPO-03B | pending | common-query scorer、paired uncertainty、partial-order graph | old P1 ties 不再产生系统性 false strict；stress suite 通过 | 只修测量，不训练 |
| PA2-CAND-03C | blocked | 仅在 robust strict edge 不足时做一次 candidate 可辨识性 pilot | 保持结构、产生可人工辨认的 P1 strict edges | reject SVD sibling 路线 |
| PA2-CALIB-03D | blocked | P1-only prospective human calibration | strict precision、tie precision、Wilson、anti-collapse 全通过 | 标签器 rejected，不进入 PA3 |
| PA3-KERNEL-04 | blocked | strict/tie tube loss、SDPO、anchor 代数与 1/8/32 capacity | algebra、winner、tie、outside、held-out 全通过 | 只允许一次 anchor-gradient fallback |
| PA4-SCREEN-05 | blocked | 单卡强 baseline 筛选 | full 超过 chosen-SFT/AWR/DenseDPO-style，独立 evaluator 同向 | 不切双卡 |
| PA5-SCALE-06 | blocked | 扩 sibling graph 与双卡并行 | 足量 calibrated graph，不降选择阈值 | 保留单卡结果 |
| PA6-FORMAL-07 | blocked | 两 training seeds 正式对照 | 主指标、人工、视觉、运动量同向 | 方法 rejected |
| PA7-EVAL-08 | blocked | 128/256+/full-val 评估与 second-backbone 决策 | 统计和 transfer gate 通过 | 限定 claim 或不投稿当前方法 |
| PA8-PAPER-09 | blocked | 主表、消融、failure analysis | reviewer threat table 均有证据 | 不包装负结果为正结果 |

## 9.1 PA2-UPO-03B：当前唯一可执行阶段

先复用已有 RGB，不重生成 candidate：

1. 实现 pair-mode tracker，query 从共同 first frame/Base guard 冻结；
2. 对正式 review 中 24 个 P1 pair 重打 common-support score；
3. 用 22 个 human ties 冻结 ROPE；
4. 对 120 conditions × 4 siblings 枚举最多 6 edges；
5. 运行 paired spatial/temporal bootstrap、simultaneous correction、cycle audit；
6. 运行 freeze/slow/dropout/camera/actor-jitter stress suite；
7. 输出 graph yield、strict/tie/incomparable 比例和 P1 retrospective calibration。

预注册 gate：

- 22 个 human-tie P1 中，false strict 不超过 2；
- 1 uncertain 和 1 both-invalid 不得产生 strict edge；
- common support 与原 candidate-specific support 的差异完整记录；
- support mismatch 不再通过不同 denominator 产生 winner；
- stress suite 达到第 8.1 节门槛；
- 在 120 conditions 上至少产生 16 条可供 prospective review 的 strict edge，否则进入 `PA2-CAND-03C`，不得训练。

## 9.2 PA2-CAND-03C：只允许一个小范围 fallback

如果可靠 strict edge 不足，先判断是 scorer 选择性过强，还是当前 sibling 差异确实不可辨认。只允许：

```text
8 preference_dev conditions
K = 4
rho = 0.04 不变
fork fraction: 0.6 → 0.4
25 steps / 14 frames
```

理由：更早 fork 是一次针对 motion identifiability 的离散干预；不同时增加 strength 或 K。

门槛：

- exact Base、first frame、finite 全通过；
- structure blind review 至少 7/8 `same_scene`；
- 至少 8 条 robust strict edge 可供盲审；
- catastrophic condition 为 0；
- 若结构失配，禁止继续加大 rho 或引入 feature intervention，SVD sibling 主路线 `rejected`。

## 9.3 PA2-CALIB-03D：P1-only prospective review

不得再把 P0/P2 的明显伪影偏好混入主标签可信性分母。固定 32 条 preference-dev P1 cases：

```text
16 predicted strict
16 predicted tie
按 uncertainty / motion / scene strata 分层
calibration conditions 不进入训练
```

与 v2 不同，strict precision 的分母是**所有 predicted strict**；人工 tie、uncertain、both-invalid 都不能被排除后再提高 agreement。

通过要求：

- 16 个 predicted strict 中至少 13 个方向与 human decisive verdict 一致；
- strict precision 的 95% Wilson lower bound >0.50；
- 16 个 predicted tie 中至少 12 个为 human tie；
- scorer-chosen low-motion collapse = 0；
- scorer-chosen catastrophic failure = 0；
- Stage A→B 改判全部有具体原因；
- 至少 25% cases 由第二评审者重叠标注，strict/tie 粗粒度一致率至少 75%；
- 完整提示词、盲法、模板和 aggregation 代码按 `AGENTS.md` 另行交付；Codex 不代填。

只有全部通过，才 materialize 训练 graph 并解锁 `PA3-KERNEL-04`。

## 9.4 PA3-KERNEL-04：代数与容量

顺序：

```text
synthetic tensor unit tests
→ 1 strict + 1 tie edge
→ 8 condition graphs
→ 32 train / 8 held-out condition graphs
```

必须测试：

```text
zero adapter => strict loss = log(2)
swap strict winner/loser => logit sign flips
tie at zero adapter => zero margin / finite gradient
incomparable => no loss
shared sigma/noise deterministic
reference no grad and adapter state exact restore
tube frame mapping and frame-0 zero
empty common support fail closed
condition mismatch / stale calibration / future-GT fail closed
winner safeguard bound
outside complement mask
strict/tie sampling condition-balanced
```

capacity gate：

- finite，peak VRAM ≤22 GB；
- strict train margin/accuracy 上升；
- tie margin 保持在冻结 ROPE；
- winner absolute tube error 不系统增加；
- outside-tube prediction drift ≤2%；
- frame-0 drift 为数值零；
- held-out strict direction win rate ≥60%；
- dynamic exposure 不下降；
- 一次预注册 fallback 后仍失败则停止，不靠更多 step 解决。

## 9.5 PA4-SCREEN-05：单卡方法筛选

第一轮只比较必要强基线，避免矩阵爆炸：

| ID | 方法 | 目的 |
|---|---|---|
| E0 | Frozen Base | 基准 |
| E1 | real-only SFT | 检查普通 fine-tuning 的 motion collapse |
| E2 | chosen-only SFT | 检查收益是否只是 winner imitation |
| E3 | scalar acceleration/flux AWR + same SFT anchor | SHIFT-style 强 baseline |
| E4 | independent-seed clip VideoDPO | vanilla video preference baseline |
| E5 | common-prefix temporal DenseDPO-style | 结构 pair + segment credit baseline |
| E6 | common-support strict-only tube DPO | 消融 partial order tie 与完整 safeguard |
| E7 | DrivePO full | proposed |

固定：

```text
同一 scene split
同一有效 candidate pool
同一 LoRA / updates / timestep budget
1 training seed
16 screen-eval clips × 2 matched generation seeds
25 denoising steps
```

只有 E7 同时满足以下条件才切双卡：

1. independent CoTracker actor-residual primary 的 direction win rate ≥60%；
2. 不差于 E2/E3/E5/E6；
3. survival/common support/background field 非劣；
4. dynamic degree/residual displacement 无 >5% 退化；
5. winner error、outside tube、visual quality 均稳定；
6. 小规模人工盲审无 scorer hacking；
7. graph yield、condition exposure 和 provenance 完整。

---

# 10. 最终评价与统计

## 10.1 Primary endpoints

预注册：

1. **Independent CoTracker common-support actor-residual curvature/acceleration outlier rate**，越低越好；
2. **Independent CoTracker common-support survival / active coverage**，非劣且优先越高越好。

Driving-specific co-primary analysis：

- ego/background field residual；
- actor-dominant、ego-dominant、mixed strata 的 interaction；
- actor tube 内改善与 complement 非劣同时成立。

P-UNC projection energy 只作 training-aligned diagnostic，不进入唯一主结论。

## 10.2 Secondary evaluators

- CoTracker3：主独立轨迹 evaluator；
- GeCo motion/structure map：静态/背景几何 secondary，不作为动态 actor 真值；
- VBench：dynamic degree、motion smoothness、subject/background consistency、flicker；
- DINO/LPIPS：matched Base identity/complement drift；
- blind human：motion plausibility、identity、quality、motion amount；
- FVD：至少 256 clips，优先完整 val；不用于 16/32-clip 筛选。

## 10.3 统计单位

```text
generation seed 不是独立样本
edge/window/query 不是独立样本
先在 condition 内聚合 edge/window/seed
再在 scene 内聚合 condition
scene-level paired hierarchical bootstrap
10,000 bootstrap samples
两个 primary endpoint 做 Holm correction
invalid 不填 0，报告实际 n 和 coverage
```

报告：mean、median、paired delta、win rate、95% CI、worst 10%、stratum interaction、coverage 和 calibration risk。

## 10.4 正式 promotion

全部满足才允许论文主张：

- 新 P1-only preference oracle 通过 prospective human calibration；
- actor-residual primary 的 paired 95% CI 上界小于 0；
- survival/common support/background residual 非劣；
- dynamic degree、active residual speed/displacement 无 >5% 退化；
- 两 training seeds、至少两 generation seeds 同向；
- E7 超过 chosen-SFT、AWR、DenseDPO-style 和 strict-only tube DPO；
- 独立 evaluator 与 blind human 同向；
- 收益不由单一 scene、branch direction、motion stratum 或少数 outlier 主导；
- 第二 backbone 至少完成 frozen-oracle + 32-condition 最小 transfer，或论文明确限定 SVD 并降低通用性 claim。

---

# 11. 工程边界与建议结构

建议新增而不改写旧 schema：

```text
motion_proj/preference/
  paired_tracks.py
  residual_motion.py
  partial_order.py
  calibration.py
  track_tubes.py
  graph_dataset.py
  drivepo_loss.py

motion_proj/diagnostics/
  physics_preference_reaudit.py
  physics_preference_stress.py
  physics_preference_calibrate.py
  drivepo_capacity.py

tests/
  test_paired_common_support.py
  test_residual_camera_invariance.py
  test_partial_order_relations.py
  test_partial_order_cycles.py
  test_tie_rope_calibration.py
  test_track_tube_mapping.py
  test_drivepo_loss.py
  test_drivepo_tie_loss.py
  test_drivepo_safeguard.py
  test_drivepo_dataset_fail_closed.py
```

正式 run 必须保存：

```text
resolved.yaml
manifest.json
source run hashes
scene split / scorer / calibration fingerprints
query-set hash
bootstrap seeds
graph.jsonl
metrics.jsonl
summary.json
COMPLETE / FAILED / REJECTED
```

禁止：

- 覆盖现有 expanded run；
- 将 review verdict 自动转写为新 label；
- 在 dirty worktree 构建正式 calibration/graph；
- 用 P0/P2 明显画质差异补足 P1 strict 数量；
- 用更多 candidate、更多 GPU 或更多 update 绕过人工可辨识性失败；
- 自动 push。

---

# 12. 明确停止条件

## Preference oracle

出现任一项即停止：

- common-support 后 old P1 ties 仍大量被判 strict；
- strict direction 对 bootstrap seed / tracker微扰不稳定；
- candidate support mismatch 仍决定 winner；
- prospective strict precision 或 Wilson gate 失败；
- 只有降低 threshold 才能凑够 graph；
- freeze/slow/dropout attack 可获胜。

## Candidate construction

- fork 0.4 的一次 pilot 仍全部人工 tie；
- earlier fork 造成 structure/identity mismatch；
- strict edge yield <10% 且只能靠增加 artifact 获得可辨识性。

结论：SVD common-prefix sibling 不适合承载该物理偏好，不进入 feature intervention 自动扩张。

## Alignment kernel

- zero-adapter / tie / swap / reference 测试失败；
- winner error 与 margin 同时上升且 safeguard 无效；
- outside drift 在一次 fallback 后仍 >2%；
- held-out graph direction <60%；
- one-step 改善无法进入 25-step rollout。

## Screening / formal

- full 不超过 chosen-SFT、AWR 或 DenseDPO-style；
- 只在 P-UNC 上改善；
- independent CoTracker 或 human 反向；
- actor 改善由 camera/background 变化解释；
- dynamic-degree / active-motion collapse；
- 两 training seeds 不一致；
- 128-clip 结果无法在 256+/full val 复现。

---

# 13. 论文叙事与可接受贡献

## 13.1 一句话问题

现有视频偏好方法通常把不同随机未来压成 scalar winner，或在局部 segment 上使用未校准标签；在驾驶视频中，camera motion、actor motion、track visibility 和画质失败相互纠缠，使低运动或丢 track 成为可利用 shortcut。

## 13.2 一句话方法

DrivePO 从同一去噪前缀构造 sibling set，在共同 first-frame query support 上分解 ego/background nuisance 与 actor residual，用 human-tie-calibrated uncertainty intervals 建立 strict/tie/incomparable 局部偏序，再只在对应 actor track tubes 上进行 winner-preserving、tie-aware 的 dense diffusion alignment。

## 13.3 只有结果成立后才允许的贡献表述

1. 一个面向 driving self-rollouts 的 **common-support uncertainty-calibrated sibling partial order**，显式拒绝 support mismatch 与不可辨认 pair；
2. 一个无需 future GT 的 **ego–actor residual preference representation**，将 camera nuisance、actor motion、visibility 和 activity constraints 分开；
3. 一个将 strict/tie graph 映射到 **track-time tubes** 的 dense safeguarded diffusion objective；
4. 一套以正式负人审、freeze/slow/dropout stress、独立 tracker 和 scene-level statistics 为核心的 anti-shortcut evaluation protocol。

不得把贡献写成：

```text
首次对驾驶视频使用 DPO
acceleration reward + DPO
DenseDPO for driving
LocalDPO with track masks
SDPO for video
自动 scorer 超越人工
```

---

# 14. 当前唯一下一步

当前只执行 `PA2-UPO-03B`：

```text
不生成新 candidate
不训练
不切双卡
不修改旧 run

已有 120-condition sibling RGB
→ 冻结共同 first-frame queries
→ pair-mode RAFT tracking
→ common-support ego–actor residual components
→ paired block bootstrap + simultaneous correction
→ 用 22 个 P1 human ties 冻结 ROPE
→ 重建 strict / tie / incomparable graph
→ stress suite
→ 决定是否有资格请求新的 P1-only prospective review
```

只有当该阶段证明旧 24 个 P1 false strict 基本消失，并在现有数据中产生至少 16 条新的 robust strict candidates，才准备新的完整人工评测提示词。否则进入一次 8-condition earlier-fork pilot；仍失败则停止 SVD sibling 路线。

---

# 15. 外部研究依据

- [Diffusion Model Alignment Using Direct Preference Optimization, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Optimization_CVPR_2024_paper.html)
- [VideoDPO: Omni-Preference Alignment for Video Diffusion Generation, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_VideoDPO_Omni-Preference_Alignment_for_Video_Diffusion_Generation_CVPR_2025_paper.html)
- [DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/fa9755043814e7f08d859a286bb83c35-Abstract-Conference.html)
- [Mind the Generative Details: Direct Localized Detail Preference Optimization for Video Diffusion Models, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Mind_the_Generative_Details_Direct_Localized_Detail_Preference_Optimization_for_CVPR_2026_paper.pdf)
- [VideoGPA: Distilling Geometry Priors for 3D-Consistent Video Generation](https://arxiv.org/abs/2601.23286)
- [SHIFT: Motion Alignment in Video Diffusion Models with Adversarial Hybrid Fine-Tuning](https://arxiv.org/abs/2603.17426)
- [Diffusion-SDPO: Safeguarded Direct Preference Optimization for Diffusion Models](https://arxiv.org/abs/2511.03317)
- [On Extending Direct Preference Optimization to Accommodate Ties](https://arxiv.org/abs/2409.17431)
- [Uncertainty-Penalized Direct Preference Optimization](https://arxiv.org/abs/2410.20187)
- [GeCo: Evaluating Geometric Consistency for Video Generation via Motion and Structure](https://arxiv.org/abs/2512.22274)
- [Are AI-Generated Driving Videos Ready for Autonomous Driving? A Diagnostic Evaluation Framework](https://arxiv.org/abs/2512.06376)
- [MAD: Motion Appearance Decoupling for Efficient Driving World Models](https://arxiv.org/abs/2601.09452)

实现时优先参考论文原文和官方代码，不从二手博客复制 loss；所有借鉴组件在论文中准确归因。
