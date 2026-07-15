# Motion-Proj Physics Preference Alignment 自动研究计划 v2

> **文档定位**：本文件是 Motion-Proj 从 explicit endpoint projection 转向 physics preference alignment 后的唯一前向研发计划。
> **决策日期**：2026-07-14
> **计划基线**：`16b6975`，执行前必须重新核对当前 `HEAD`、分支和 worktree。
> **当前硬件**：单张 RTX 4090 24 GB；仅在单卡筛选门槛通过后，由用户执行一次停机并切换为双 RTX 4090。
> **当前状态**：`PA0-REVIEW-00 done`、`PA0-SCENE-SPLIT-01 done`、`PA1-HORIZON-01 done`。PA1 v3 以 clean `57987b0` 完成 2 condition × {8,14} 的 exact Base guard profile；14 帧通过预注册资源门槛，已冻结为后续 preference 数据帧数，claim scope 仍为 **short-horizon dynamics alignment**。`PA1-BRANCH-02` v1 的手动 scheduler continuation 被 exact Base safety gate 拦下；v2 是 schema RMS 尾差实现失败；v3 完成 20 candidate + CoTracker，但 scene-0126 的全部 sibling future distance 低于下界、scene-1102 Base track coverage 低于硬门槛，故 machine gate 为 2/4、4/8。唯一允许的 v4 是按离散阶梯将 strength 从 `rho=0.01` 推进到 `0.02`，不改 fork/condition/family/gate，仍不授权训练或切换双卡。
> **状态词**：`pending / running / awaiting_reviews / blocked / done / rejected`。
> **取代范围**：取代旧计划中未来关于 reward/DPO/AWR 的禁止性排程，不修改 V1/V2、F0、F1、P1 等历史负结论。
> **目标投稿**：CVPR/ICCV 级视觉顶会；方法必须落在真实 RGB 驾驶视频生成、时序运动和物理一致性上。

---

# 0. Area Chair 审查结论

## 0.1 方向判断

从当前证据看，转向 preference alignment 是合理的：

```text
已失败路线：
Base rollout
→ 连续轨迹投影
→ RGB crop/resize/paste
→ VAE/hybrid latent
→ endpoint regression

新研究问题：
真实生成候选视频
→ 无 future-GT 的物理偏好判定
→ 离线 preference optimization
→ 独立完整 rollout 评估
```

新方向绕开了 P1 已证伪的 counterfactual RGB/VAE target，不要求构造不存在的 \(X^\dagger\)。chosen 和 rejected 必须都是模型真实采样并解码得到的 RGB 视频：

\[
X^w \succ X^l.
\]

但“用物理分数自动打 pair，再套 Diffusion-DPO”本身不足以构成 CVPR/ICCV 方法贡献。VideoDPO 已覆盖多维自动评分和 pair 重加权；DenseDPO 已覆盖结构对齐 pair 与 segment-level preference；VideoGPA 已覆盖自动几何偏好；SHIFT 已覆盖运动 reward 与 AWR；Diffusion-SDPO 已覆盖 winner-preserving preference update。

因此，本项目只有在以下四项形成完整闭环时才有顶会价值：

1. **结构对齐候选（Structure-Aligned Candidates）**：降低候选之间的构图、身份和纹理混杂；
2. **约束优先的物理偏好（Constraint-Aware Physics Preference）**：不用任意加权总分把“少动”包装成“更物理”；
3. **局部时间信用分配（Transition/Segment Credit Assignment）**：只优化真正出现运动差异的时段；
4. **winner-preserving 更新（Safeguarded Preference Update）**：避免通过恶化 loser 同时破坏 winner。

暂定方法名：

> **SAP-DPO: Structure-Aligned Physics Preference Optimization for Driving Video World Models**

名称仅作工作标识。只有核心消融成立后才允许作为论文正式方法名。

## 0.2 论文级最低创新边界

论文不能仅声称：

- 首次把 DPO 用于驾驶视频；
- 用 acceleration/jerk 作为 reward；
- 用同 seed 生成 chosen/rejected；
- 用 CoTracker3 评价视频；
- Physics-DPO 优于 Base。

论文至少需要证明：

\[
\text{SAP-DPO}
>
\text{Vanilla Video DPO}
\]

并进一步证明收益来自：

\[
\text{uncertainty-aware physics preference}
+
\text{temporal credit assignment}
+
\text{anti-collapse constraints},
\]

而不是来自：

- chosen-only SFT；
- 更强的数据过滤；
- 更少运动；
- 更大视觉偏离；
- 某一种固定 branch recipe；
- 训练 scorer 的 metric hacking。

---

# 1. 历史结论边界与可复用资产

## 1.1 已经否定的内容

以下路线继续保持 `rejected`：

```text
synthetic corruption
→ low-noise absolute x0 / residual-v endpoint target
→ shared temporal LoRA
→ full-image anchor / outside preserve
```

当前不能重试：

- V1 Optuna；
- t10-800；
- 当前 32-clip synthetic cache 长训；
- P1 crop/paste renderer；
- masked hybrid latent target；
- 当前 endpoint 的 learning-rate、preserve-weight、rank 或 step sweep；
- 用 future ego/box/track 约束未接收这些条件的 SVD；
- 用训练 RAFT 同时造 label 和做唯一正式评价。

## 1.2 当前可复用资产

| 资产 | 当前可信范围 | 在新路线中的用途 |
|---|---|---|
| `svd_official_v1` | 与官方 Diffusers SVD pipeline 在 matched input 下 exact parity | 所有候选生成、训练 conditioning 和评估生成的唯一协议 |
| P-UNC | point-track space 机器门槛通过，仍需完成既有 12-panel 人审 | 训练 scorer 的 uncertainty、eligibility 和 projection-distance 诊断 |
| CoTracker3 E0 v3 | rerun 与扰动排序机器稳定，仍需完成既有 12-panel 人审 | 独立 hold-out evaluator |
| temporal-only rank-16 LoRA | 单卡可训练，adapter provenance 已有 | 第一版 policy 参数化 |
| raw-\(v\) 与 \(x_0\) 变换 | 与 SVD scheduler 已验证 | Diffusion-DPO loss |
| run/cache 基础设施 | manifest、fingerprint、JSONL、atomic terminal state | preference 数据与训练可追溯 |
| 历史 Base replay | provenance 可追溯 | scorer/evaluator sanity；不得作为 projected target |

历史复盘文件是本计划的事实边界，不应被新方向重写。当前旧路线的精确负结论与保留资产见既有实验复盘。

---

# 2. 研究问题、假设与非目标

## 2.1 核心研究问题

> 在不使用 future-GT、外部动作条件、counterfactual renderer、可训练 reward model 或完整 sampling-chain backprop 的前提下，能否从 SVD 自身 rollout 中构造结构对齐、置信度可解释的驾驶动力学偏好，并通过离线 preference optimization 改善完整 rollout 的短时物理一致性，同时不牺牲运动量、轨迹存活、视觉质量和身份一致性？

## 2.2 核心假设

### H1：结构对齐 pair 降低偏好混杂

共享 condition、initial latent 和去噪前缀的 sibling candidates，相比 independent-seed pair，具有更高的：

- first-frame consistency；
- scene-layout similarity；
- subject identity consistency；
- track correspondence；
- temporal segment alignment。

### H2：projection distance 可以作为相对物理偏好，而非像素 target

P-UNC 不再渲染 RGB target，而只输出：

- uncertainty-normalized correction energy；
- high-SNR violation coverage；
- motion invariant violations；
- temporal violation location。

在 anti-collapse 与质量非劣约束下，较低的高置信 projection distance 可以形成可信的相对偏好。

### H3：segment-level preference 优于 clip-level scalar preference

一个候选可能仅在部分时间段更优。将整条视频压成一个 winner 会错误惩罚另一条视频中本来正常的时段。对 4-frame 局部窗口做 preference，可减少时间信用分配错误。

### H4：标准 DPO 可能通过恶化 loser 扩大 margin

仅观察 DPO margin 不足以证明 winner 变好。winner-preserving safeguard 与 SFT anchor 应能降低视觉退化和训练不稳定。

## 2.3 明确非目标

本阶段不做：

- 训练大型 reward model；
- 在线 PPO/GRPO；
- 完整 25-step chain 反传；
- 5B 级 backbone；
- RGB/flow 联合扩散；
- GT future imitation；
- 大型视频编辑器生成 chosen；
- 闭环驾驶规划；
- 长期世界模型 rollout；
- 把 image-plane acceleration 称为真实世界物理加速度。

当前指标应准确命名为：

```text
camera-compensated image-plane acceleration
camera-compensated image-plane jerk
uncertainty-normalized projection energy
```

---

# 3. 与最近邻工作的关系

| 工作 | 已覆盖内容 | 本项目不能重复声称 | 本项目必须保留的差异 |
|---|---|---|---|
| Diffusion-DPO，CVPR 2024 | 用扩散 ELBO 近似实现离线 DPO | DPO loss 本身 | 驾驶动力学 pair、时段信用、约束偏好 |
| VideoDPO，CVPR 2025 | 多维视频评分、自动 pair、score reweighting | 多指标加权评分和 top-vs-bottom pair | uncertainty-aware 物理 partial order、结构对齐 sibling |
| DenseDPO，NeurIPS 2025 Spotlight | 结构相似视频、segment-level preference、低运动偏置分析 | “结构 pair”和“segment DPO”本身 | self-rollout 驾驶 dynamics projection energy 与 anti-collapse constraints |
| Diffusion-SDPO，2025 | loser gradient safeguard，winner first-order preservation | winner safeguard 本身 | 作为稳定组件，不作为原创贡献 |
| VideoGPA，2026 | 几何 foundation model 自动构造 DPO pair | 自动几何 preference | 动态目标轨迹、不确定度、无 3D foundation model |
| SHIFT，2026 | motion reward、AWR、noise alignment、reward hacking | AWR 和 noise alignment | 必须作为强 baseline，并证明 DPO 的额外价值 |
| SIFT，2026 | self-generated hard-case replay | self-rollout/hard case 本身 | preference construction 与可追溯 pairwise causal evidence |

## 3.1 顶会接受门槛

如果最终结果只是：

```text
P-UNC score 排序
→ Vanilla Diffusion-DPO
→ P-UNC score 上升
```

则该项目应判为 `rejected`，因为它是明显的 self-evaluation loop。

如果结果只比 Base 好、但不超过：

- chosen-only SFT；
- offline AWR；
- vanilla clip-level DPO；
- independent-seed DPO；

也不具备足够的方法证据。

---

# 4. 数据划分与泄漏控制

## 4.1 先冻结 scene-level split

在生成任何 preference candidate 前，按 nuScenes scene 划分：

```text
preference_train_scenes
preference_dev_scenes
screen_eval_scenes
formal_test_scenes
```

要求：

- 同一 scene 的相邻 clip 不能跨 split；
- candidate generation threshold 只能在 train/dev 冻结；
- formal test 不参与 reward weight、margin、fork step 或 branch strength 选择；
- Base、AWR、SFT、DPO 使用完全相同的 eval scene 清单。

建议：

- preference train：官方 train scenes；
- preference dev：从 train scenes 中独立留出；
- screen/formal test：官方 val scenes，固定后不修改。

## 4.2 Condition 是统计和数据采样基本单位

同一 condition 可以有多条 candidate 和多个 segment label，但：

- 训练采样必须先均匀采 condition，再采 pair/segment；
- 不允许 candidate 多的 condition 获得更高权重；
- bootstrap 以 scene/condition 为单位；
- generation seed 不是独立统计样本。

## 4.3 数据 schema

每条 candidate 保存：

```text
candidate_id
condition_id
scene_id
split
conditioning_frame
generation_protocol
base_model_fingerprint
initial_latent_hash
prefix_latent_hash
fork_step
branch_family
branch_direction
branch_strength
generation_seed
scheduler_fingerprint
fps
motion_bucket_id
guidance_schedule
num_frames
rgb_video_path
vae_latent_path
track_diagnostics_path
quality_diagnostics_path
uses_future_gt=false
git_commit
config_fingerprint
```

每条 preference 保存：

```text
pair_id
condition_id
candidate_a
candidate_b
global_label
segment_labels
segment_confidences
physics_components
quality_components
feasibility_flags
preference_margin
abstain_reason
human_review_id
scorer_fingerprint
```

禁止保存：

- P1 projected RGB；
- hybrid latent；
- future-GT；
- 无 provenance 的 mp4；
- 被覆盖的同名 run。

---

# 5. 方法定义：SAP-DPO

# 5.1 H0：时间长度与可观测性门槛

当前项目常用 8 帧，但 acceleration/jerk 对短序列和 tracking noise 高度敏感。PA1 前先 profile：

```text
8 frames
14 frames
```

每档只用 2 个 condition，记录：

- generation peak VRAM；
- seconds/video；
- score seconds/video；
- valid track length；
- acceleration/jerk repeatability；
- candidate storage。

选择规则：

- 若 14 帧单卡峰值显存不超过 22 GB，且吞吐不超过 8 帧的 2.2 倍，后续 preference 数据优先使用 14 帧；
- 否则保持 8 帧，但论文只能声称 **short-horizon dynamics alignment**；
- 8 帧设置中 jerk 只能作为 secondary diagnostic，不可作为唯一 primary reward。

不允许为了使用更长视频提前切双卡；普通 DDP 不增加单卡样本显存。

# 5.2 结构对齐候选生成

## 5.2.1 Candidate anchor

每个 condition 首先用 `svd_official_v1` 生成 exact Base guard：

\[
X^{\mathrm{base}}=G_{\theta_0}(\xi,c).
\]

Base guard：

- 用于 motion/quality non-inferiority；
- 不自动进入 chosen/rejected；
- 必须保存完整 denoising trace fingerprint；
- rerun 必须 exact。

## 5.2.2 主候选：common-prefix sibling branches

从同一个 initial latent 开始，运行到 fork timestep \(t_f\)，得到共享前缀 latent：

\[
z_{t_f}^{\mathrm{prefix}}.
\]

构造四个等范数、零均值、成对反向的 perturbation：

\[
\delta_1=-\delta_2,\qquad
\delta_3=-\delta_4,
\]

\[
\operatorname{RMS}(\delta_j)
=
\rho\,\sigma_{t_f}.
\]

候选：

\[
z_{t_f}^{(j)}
=
z_{t_f}^{\mathrm{prefix}}+\delta_j,
\qquad
X^{(j)}=\operatorname{Decode}
\left(
G_{\theta_0}^{t_f\rightarrow0}(z_{t_f}^{(j)},c)
\right).
\]

约束：

- 四个 perturbation equal norm；
- direction 由固定 seed 构造；
- 不对某 branch 使用更高 CFG 或额外后处理；
- 所有 sibling 使用相同 scheduler、condition 和 decode；
- perturbation 仅影响 fork 后路径；
- branch ID 不携带预设“好/坏”含义。

Base guard 之外，每 condition 默认 4 个 sibling candidates，共 5 条视频。

## 5.2.3 对照候选：Base re-noise structural pair

作为 pair-construction baseline，可将 Base latent forward-noise 到固定中噪尺度，再以多组 noise 重新 denoise：

\[
z_\tau^{(j)}
=
\alpha_\tau x_0^{\mathrm{base}}
+
\sigma_\tau\epsilon_j.
\]

该 family 只用于 PA1 小规模比较，不与 common-prefix family 同时扩规模。选择一个 family 后冻结。

## 5.2.4 PA1 仅允许的离散校准

只测试：

```text
fork fraction: {0.4, 0.6, 0.8}
strength level: {small, medium, large}
```

不是 \(3\times3\) 全网格。使用逐级最小强度策略：

1. 固定 fork 0.6，从 small 开始；
2. 若候选不可区分，再试 medium；
3. 若结构失配，再改 fork；
4. 最多 4 conditions；
5. 找到首个满足门槛的最小配置后冻结。

门槛包括：

- exact Base guard；
- candidate finite；
- first frame 一致；
- candidate 非重复；
- track correspondence coverage；
- subject/background consistency；
- future-frame perceptual distance位于“deterministic rerun noise floor”和“independent-seed distance”之间；
- branch winner 分布不过度偏置；
- 人工看起来是同一场景的不同未来，而非不同构图。

若 common-prefix 和 re-noise 均失败，PA1 `rejected`。禁止退回完全 independent-seed pair 作为主方法；independent seed 仅作为 vanilla DPO baseline。

# 5.3 物理偏好：可行性优先，不使用任意总分决定 winner

原计划中的线性总分：

\[
R(X)=\sum_i w_iS_i
\]

只能作为 baseline，不能作为主标签器。任意权重会导致：

- acceleration 与 dynamic degree 互相抵消；
- 低运动视频通过低 jerk 获胜；
- 高画质掩盖轨迹失效；
- 全局 normalizer 产生 condition bias。

主标签器采用两阶段决策。

## 5.3.1 Stage A：硬可行性约束

定义：

\[
\mathcal F(X)\in\{0,1\}.
\]

候选必须同时满足：

- `uses_future_gt=false`；
- finite；
- valid track coverage；
- minimum median track length；
- visibility/survival；
- no catastrophic decode；
- no first-frame failure；
- dynamic degree 位于相对 Base guard 的预注册区间；
- net displacement 不发生系统性 collapse；
- quality guard；
- scorer confidence；
- P-UNC human review 已通过。

不满足者不能成为 winner。若两个 candidate 都不可行，该 condition abstain。

## 5.3.2 Stage B：uncertainty-aware physics violation vector

对可行候选计算：

\[
\mathbf E(X)=
\left[
E_{\mathrm{proj}},
E_{\mathrm{acc}},
E_{\mathrm{jerk}},
E_{\mathrm{survival}},
E_{\mathrm{background}}
\right].
\]

其中主量为 P-UNC projection distance：

\[
E_{\mathrm{proj}}
=
\frac{
\sum_{i,t}
w_{i,t}
\left\|
p_{i,t}-
\Pi_{\mathrm{UNC}}(p_i)_t
\right\|_2^2
}{
\sum_{i,t}w_{i,t}+\varepsilon
}.
\]

只统计：

\[
\frac{
\|p-\Pi_{\mathrm{UNC}}(p)\|
}{
\sigma_{\mathrm{track}}+\varepsilon
}
\ge \tau_{\mathrm{SNR}}.
\]

其他量：

- \(E_{\mathrm{acc}}\)：camera-compensated acceleration outlier rate；
- \(E_{\mathrm{jerk}}\)：camera-compensated jerk outlier rate，仅 secondary；
- \(E_{\mathrm{survival}}\)：失活、身份切换和过短 track penalty；
- \(E_{\mathrm{background}}\)：背景残差与前景泄漏诊断。

所有分量按同 condition sibling candidates 做 rank/robust normalization，不跨 condition 直接比较绝对像素量。

## 5.3.3 Pareto + non-inferiority preference

候选 \(A\) 优于 \(B\) 仅当：

1. \(\mathcal F(A)=\mathcal F(B)=1\)；
2. \(A\) 在 primary physics energy 上有超过置信 margin 的改善；
3. \(A\) 在 survival、dynamic degree、net displacement、quality 上不劣于 \(B\) 超过容差；
4. 至少一个 physics component 显著改善；
5. 没有 component 出现 hard conflict；
6. scorer confidence 足够。

形式化：

\[
A\succ B
\]

当：

\[
E_{\mathrm{proj}}(A)
+
m_{\mathrm{proj}}
<
E_{\mathrm{proj}}(B),
\]

并且：

\[
Q_k(A)\ge Q_k(B)-\epsilon_k
\]

对所有 anti-collapse/quality component \(Q_k\) 成立。

若分量冲突，label 为 `abstain`。禁止强行用加权和解决冲突。

## 5.3.4 Pair confidence

pair 权重：

\[
q_{\mathrm{pair}}
=
q_{\mathrm{track}}
q_{\mathrm{margin}}
q_{\mathrm{quality}},
\qquad
q_{\mathrm{pair}}\in[0,1].
\]

所有分量必须 clip。禁止使用逆频率重权把罕见 outlier 放大为极高权重。VideoDPO 风格 histogram reweighting只作为消融。

# 5.4 局部时间偏好

对于 \(T\) 帧视频，使用长度 4 的重叠窗口：

\[
\mathcal I_s=
\{s,s+1,s+2,s+3\},
\qquad
s=0,\dots,T-4.
\]

4 帧窗口可观测：

- velocity；
- acceleration；
- jerk 的局部近似；
- survival；
- turn continuity。

每个窗口独立输出：

```text
local winner
local loser
tie / abstain
confidence
violation decomposition
```

一对视频允许出现：

```text
segment 0: A better
segment 1: tie
segment 2: B better
```

不能为了 clip-level label 把这些信息压成单一 winner。

Clip-level DPO 只作为 baseline：

- 仅使用局部标签方向一致的 pair；
- 或使用全局 Pareto dominance pair。

拟议主方法使用非 tie segment。

# 5.5 Diffusion-DPO 基线

对 segment 的 local winner \(x_0^w\) 和 loser \(x_0^l\)，共享 diffusion noise level 和噪声：

\[
z_\tau^u
=
\alpha_\tau x_0^u+\sigma_\tau\epsilon,
\qquad
u\in\{w,l\}.
\]

\(v^*\) 必须调用已经测试的：

```python
model_output_from_x0(z_tau, sigma_tau, x0)
```

不在新代码中手写另一套参数化公式。

定义 per-frame residual：

\[
e_\theta^u[k]
=
\operatorname{mean}_{c,h,w}
\left\|
v_\theta(z_\tau^u,\tau,c)[k]
-
v^{*,u}[k]
\right\|_2^2.
\]

segment denoising error：

\[
\ell_\theta^u(s)
=
\sum_{k\in\mathcal I_s}
a_{s,k}e_\theta^u[k].
\]

冻结 reference：

\[
\Delta_\theta^u(s)
=
\ell_\theta^u(s)-\ell_{\mathrm{ref}}^u(s).
\]

DPO logit：

\[
d_\theta(s)
=
\Delta_\theta^l(s)-\Delta_\theta^w(s).
\]

Clip-level baseline：

\[
\mathcal L_{\mathrm{clip}}
=
-\log\sigma\left(\beta d_\theta\right).
\]

Dense physics DPO：

\[
\mathcal L_{\mathrm{dense}}
=
-
\frac{
\sum_s q_s
\log\sigma\left(\beta d_\theta(s)\right)
}{
\sum_s q_s+\varepsilon
}.
\]

要求：

- pair 内共享 \(\tau,\epsilon\)；
- segment 之间可共享同一次 forward；
- tie/abstain segment 不进入 loss；
- candidate A/B 的 frame alignment 必须通过 PA1；
- loss 先按 latent element 归一化，再乘 \(\beta\)；
- 不照搬 Diffusion-DPO 图像实验中的数值 \(\beta\)。

## 5.5.1 \(\beta\) 的规模校准

只允许三档离散 scale calibration，不用 rollout metric 选择：

1. 在 8-pair capacity 数据上施加固定、极小 LoRA perturbation；
2. 统计 raw DPO logit 的 median absolute value 和 sigmoid saturation；
3. 选择使大部分 \(|\beta d|\) 位于非饱和区间的中间档；
4. 候选仅为 \(\{\beta_0/4,\beta_0,4\beta_0\}\)；
5. 选择后在 PA4 前冻结。

这属于数值尺度校准，不是性能超参搜索。

# 5.6 Winner-preserving safeguard

标准 Diffusion-DPO 可能通过增大 loser error 来扩大 margin，同时 winner error 也上升。必须逐 step 记录：

```text
winner policy error
winner reference error
loser policy error
loser reference error
DPO margin
```

拟议正式方法采用 Diffusion-SDPO 风格 safeguard。令 winner 与 loser 分支梯度为 \(g_w,g_l\)，更新方向近似：

\[
-g_w+\gamma g_l.
\]

为了使 winner loss 的一阶变化非正，若：

\[
\langle g_w,g_l\rangle>0,
\]

则限制：

\[
\gamma
\le
\kappa
\frac{
\|g_w\|_2^2
}{
\langle g_w,g_l\rangle+\varepsilon
},
\qquad
0<\kappa\le1.
\]

实际实现优先使用官方 Diffusion-SDPO 的 output-space approximation，并通过 parity test 验证。该 safeguard 是借鉴组件，不得作为论文原创点。

必须对比：

- vanilla Diffusion-DPO；
- fixed loser scale；
- adaptive safeguard。

Promotion 要求 winner absolute error 不发生系统性上升。

# 5.7 Real-video SFT anchor 与 noise alignment

Capacity gate 可先运行 DPO-only，正式 screening 的默认方法必须包含固定 real-video SFT anchor：

\[
\mathcal L
=
\mathcal L_{\mathrm{dense}}
+
\lambda_{\mathrm{real}}
\mathcal L_{\mathrm{real}}.
\]

要求：

- DPO 与 SFT branch 共享 diffusion timestep；
- shape 允许时共享 noise tensor；
- \(\lambda_{\mathrm{real}}\) 用 gradient RMS 比率校准，不根据 rollout 指标调节；
- 初始目标是：

\[
0.5
\le
\frac{
\|\lambda_{\mathrm{real}}g_{\mathrm{real}}\|
}{
\|g_{\mathrm{DPO}}\|+\varepsilon
}
\le
2.
\]

必须报告 no-anchor ablation。SHIFT 已表明 noise alignment 主要用于稳定外观—运动折中和缓解 reward hacking。

# 5.8 Offline AWR/RWR baseline

AWR 必须精确定义，不可仅写名称。

对同 condition candidate reward \(r_i\)：

\[
A_i
=
r_i-
\operatorname{median}_{j\in c}(r_j).
\]

使用 clipped linear 或 exponential weight：

\[
w_i
=
\operatorname{clip}
\left(
\exp(A_i/\tau_A),
w_{\min},w_{\max}
\right).
\]

训练：

\[
\mathcal L_{\mathrm{AWR}}
=
\frac{
\sum_i w_i\mathcal L_{\mathrm{denoise}}(x_i)
}{
\sum_iw_i+\varepsilon
}
+
\lambda_{\mathrm{real}}\mathcal L_{\mathrm{real}}.
\]

AWR 与 DPO：

- 使用同一 candidate pool；
- 相同 LoRA；
- 相同更新数；
- 相同 SFT anchor；
- 相同单卡/双卡预算。

---

# 6. 人工校准与 evaluator 独立性

# 6.1 PA0：先完成已有 review

必须先完成：

- P-UNC 12-panel；
- CoTracker3 E0 v3 12-panel。

P-UNC 通过要求：

```text
至少 11/12 decisive verdict 合理
无系统性静止化
无 frame-0 / visibility / support 违规
```

CoTracker3 通过要求：

```text
至少 10/12 decisive overlay 可信
identity switch / occlusion failure 可识别
无有效 track 返回 invalid 而非 0
```

人工结果不得由 Codex 自动填写。

# 6.2 Pair review 采用两阶段盲法

PA2 随机抽取至少 30 个 pair，分 strata 和 margin 分桶。

### 阶段 A：视频盲审

不展示自动 winner，不展示 physics score，只回答：

```text
A better / B better / tie / both invalid
motion plausibility
visual quality
motion amount
identity consistency
failure reason
```

### 阶段 B：诊断 adjudication

再展示：

- track overlay；
- local segment labels；
- coverage/confidence；
- physics decomposition。

记录 reviewer 是否改变 verdict。

通过要求：

- decisive scorer-human agreement \(\ge75\%\)；
- 95% Wilson lower bound高于 50%；
- chosen low-motion collapse 为 0；
- chosen catastrophic quality failure 为 0；
- video-only 与 overlay verdict 的系统性偏差必须解释。

如果 30 个 pair 数量不足以形成 decisive verdict，扩大人工 review，而不是放宽阈值。

# 6.3 训练 scorer 与 evaluator

| 角色 | Provider | 可读取内容 |
|---|---|---|
| pair label scorer | RAFT-chain + P-UNC | generated RGB、自己的 tracks、uncertainty |
| independent evaluator | CoTracker3 E0 v3 | generated RGB、自己的 fixed query grid |
| public secondary metrics | VBench 子指标 | generated RGB |
| human evaluator | blind videos，后续 overlay | 不读取自动 winner |

正式评估禁止用 P-UNC 作为唯一证据。

---

# 7. 里程碑与自动决策表

| ID | 状态 | 任务 | 晋级门槛 | 失败动作 |
|---|---|---|---|---|
| PA0-REVIEW-00 | done | 完成 P-UNC/E0 人审 | 两者均通过 | `blocked`，不生成 preference 数据 |
| PA0-SCENE-SPLIT-01 | done | materialize 唯一 scene-level split | source fingerprint、scene/clip 不泄漏与 `COMPLETE` 均通过 | `blocked`，不进入 PA1 |
| PA1-HORIZON-01 | done | 8/14 帧 profile | v3 通过：冻结 14 帧与 short-horizon claim scope | 仅保留 8 帧短时 claim |
| PA1-BRANCH-02 | running（v4 medium-strength calibration） | 结构对齐候选 pilot | family、fork、strength 冻结 | `rejected`，不退回 independent seed 主线 |
| PA2-PAIR-03 | pending | 32 condition pair legality | ≥24 有效 condition，30-pair 人审通过 | `rejected` |
| PA3-KERNEL-04 | pending | DPO/AWR/SDPO 代数与容量 | 1/8/32 pair 依次通过 | `blocked`，只修代数/实现 |
| PA4-SCREEN-05 | pending | 单卡方法筛选 | proposed 超过强基线，无 collapse | 终止，不切双卡 |
| PA5-SCALE-06 | pending | 双卡 candidate scale-out | 300–800 高质量 pairs | `blocked`，保留单卡证据 |
| PA6-FORMAL-07 | pending | 两 training seeds 正式对照 | 主指标与人工同向 | `rejected` |
| PA7-EVAL-08 | pending | 128/256+ clip 评估 | 统计门槛通过 | 归档负结果 |
| PA8-PAPER-09 | pending | 主表、消融、failure analysis | 可答辩贡献成立 | 不投稿当前方法 |

---

# 8. 各阶段详细执行

## PA1-HORIZON-01

只运行：

```text
2 conditions × {8,14} frames
Base guard only
```

输出：

- peak VRAM；
- generation time；
- score time；
- track coverage；
- metric repeatability；
- storage/video。

不训练。

## PA1-BRANCH-02

只运行：

```text
4 conditions
Base guard + 4 sibling candidates
最多两个 candidate families
25 denoising steps
```

输出：

```text
candidate_manifest.jsonl
per-step latent hashes
pairwise distances
track correspondence
branch balance
mp4 panels
profile.json
```

必须做 8–12 个候选组人工检查。

### 已登记的 PA1-BRANCH executions

在 PA1-HORIZON v3 冻结 `num_frames=14` 后，formal pilot 只允许使用：

```text
common-prefix family
fork fraction = 0.6（25 steps 中 shared prefix 为 15 transitions）
strength = small（rho = 0.01 × sigma_fork）
4 preference_dev conditions × (1 exact Base guard + 4 siblings)
```

两组 antithetic direction 使用固定 seed、零均值、理论等范数的 `+/-` 扰动。一个
independent-seed rollout 只作为 future-distance 上界 diagnostic，明确不写入
`candidate_manifest.jsonl`、不参加 pair 或训练。v1 config 为
`configs/diagnostics/physics_dpo_branch.yaml`（fingerprint `a4fdfbbb6d44`）。它的手动
scheduler continuation 没有 exact 重构 official Base trace，故在第一个 condition 的安全门禁
处失败；没有写出候选、评分、pair、训练或 method decision。

v2 config 为 `configs/diagnostics/physics_dpo_branch_v2.yaml`：只将实现替换为 official
`callback_on_step_end` 的 fixed-boundary injection，并保存/比对 pre-injection latent；研究参数、
data、family、fork、strength、阈值和 review protocol 不变。完整语义见
`docs/PHYSICS_DPO_PA1_BRANCH_PROTOCOL.md`。v2 仍须先通过 exact Base rerun 和 shared-prefix trace
核验，才允许落盘任何 sibling。v2 的四个 condition generation 均通过该核验，但 schema 将
permutation 的 float32 RMS 归约尾差误判为四条 sibling 不等范数；因此没有评分、pair、panel 或
训练输出。

v3 config 为 `configs/diagnostics/physics_dpo_branch_v3.yaml`：只让 schema 使用与 generator 已有
理论等范数检查相同的 `1e-7` 相对容差；不改变任何研究参数、生成轨迹、候选 family、fork、strength、
条件或 gate。v3 不得复用 v2 candidate artifact，必须重新生成并重新验证。

v3 完成 4 condition × 5 candidate、20 条独立 CoTracker score；所有 Base rerun、callback shared-prefix、
first-frame、track correspondence 和 antithetic distance symmetry 均通过。scene-0126 的四条 sibling
future-VAE ratio 为 `0.0376–0.0411`，均低于 `0.05` 下界；scene-1102 Base track coverage 为 `0.3055`
（门槛 `0.50`），median track length 亦不足。其他两 condition 的四个 antithetic groups 均通过，
无 structure mismatch，machine summary 为 `2/4 condition`、`4/8 groups`，因此不得进入 human review、
PA2、DPO/AWR 或训练。

v4 config 为 `configs/diagnostics/physics_dpo_branch_v4.yaml`。预注册离散强度阶梯为
`small=0.01 -> medium=0.02 -> large=0.04`；由于 scene-0126 的**全部** sibling 不可区分，v4 仅执行
下一档 `medium=0.02`，同一四 condition、same seed rules、fork `0.6`、family、阈值和 review protocol
保持不变。即使 scene-1102 继续不合格，v4 仍只有在其余三 condition、六个 groups 达标时才可 machine-pass。

若任一 condition 的全部 sibling 与 Base 不可区分，下一次只能升至 medium；若发生 structure mismatch，
下一次只能将 fork 调到 0.8；不得并行网格搜索，也不得跳入 re-noise、DPO/AWR、训练或双卡。

## PA2-PAIR-03

固定：

```text
32 train conditions
每 condition 1 Base guard + 4 sibling candidates
最多一个 preference pair / condition
```

目标：

- 至少 24 个可用 global pair；
- 至少 24 个 condition 有非 tie local segment；
- 记录 abstain rate；
- 不允许通过降低 margin 强行凑 pair。

必须比较 pair constructor：

```text
P0 independent-seed vanilla
P1 common-prefix sibling
P2 Base re-noise structural
```

PA2 只比较 pair 合法性，不训练。

## PA3-KERNEL-04

### 必须单元测试

```text
zero adapter => d=0 and loss=log(2)
swap winner/loser => logit sign flips
same sigma/noise => deterministic trace
independent noise increases estimator variance
reference has no grad
LoRA state restored after reference forward
candidate fingerprints fail closed
future-GT fails closed
tie segment excluded
segment labels map to correct frames
all-tie pair returns invalid
pair condition mismatch fails
NaN/Inf fails
winner safeguard bound test
official SDPO parity on synthetic tensor
SFT/DPO noise alignment test
```

### 容量顺序

```text
1 pair
→ 8 pairs
→ 32 train / 8 held-out conditions
```

1-pair 只证明可优化，不是方法成立。

通过要求：

- finite；
- peak VRAM \(\le22\) GB；
- train preference accuracy 上升；
- held-out positive margin \(\ge60\%\)；
- winner absolute error 不系统增加；
- reference exact；
- no adapter-state leak；
- real anchor gradient scale 合法；
- 无明显 dynamic-degree collapse。

若标准 DPO margin 上升但 winner error 也上升，必须进入 safeguard 对照，不能直接晋级。

## PA4-SCREEN-05：单卡筛选

数据规模：

```text
128 train conditions
1 Base guard + 4 candidates / condition
目标 100–250 global pairs
同时保留 local segment labels
```

训练：

```text
100–500 updates
1 training seed
16 held-out val clips
2 matched generation seeds
25 denoising steps
```

### 共同预算方法

| ID | 方法 | 目的 |
|---|---|---|
| E0 | Frozen Base | 基准 |
| E1 | real-only SFT | 检查普通微调的 motion collapse |
| E2 | chosen-only SFT | 检查收益是否只来自 winner imitation |
| E3 | offline AWR/RWR | SHIFT 类 reward-weighted baseline |
| E4 | independent-seed clip DPO | Vanilla video DPO |
| E5 | structure-aligned clip DPO | 只验证 pair construction |
| E6 | dense Physics-DPO | segment credit |
| E7 | dense Physics-DPO + safeguard + SFT anchor | 拟议完整方法 |
| E8 | shuffled-label DPO | 诊断 control，只跑小预算 |

单卡切双卡门槛：

1. E7 在独立 CoTracker 主指标上方向性胜率 \(\ge60\%\)；
2. E7 不差于 E2/E3/E5/E6；
3. E7 的 winner absolute error 稳定；
4. dynamic degree、survival、quality 无系统退化；
5. 人工盲审没有 reward hacking；
6. branch family 不垄断收益；
7. pair p95 exposure \(\le8\)；
8. 全部 provenance 完整；
9. 单卡 profile 显示扩大数据而非单样本显存是主要瓶颈。

未满足时不切双卡。

## PA5-SCALE-06：双卡扩规模

仅由用户执行停机并配置双 4090。

规模：

```text
256–512 conditions
1 Base guard + 4 candidates
1,280–2,560 total videos
300–800 high-quality global pairs
更多 local segment labels
```

双卡优先作为独立 worker：

```text
GPU 0: even condition generation/scoring
GPU 1: odd condition generation/scoring
```

每个 shard：

- 独立 run ID；
- 不重叠 condition；
- 独立 seed range；
- 独立 manifest；
- merge 时校验 fingerprint 和 duplicate。

不要默认 DDP。

## PA6-FORMAL-07

正式方法：

```text
Base
real-only SFT
chosen-only SFT
AWR
structure-aligned clip DPO
dense Physics-DPO
dense Physics-DPO + safeguard + SFT
```

至少两个 training seeds。

优先并行：

```text
GPU 0: method/seed A
GPU 1: method/seed B
```

只有当单卡每 run 500 updates 超过预注册夜间窗口，并且两个独立 seed 已验证一致后，才考虑 DDP。

## PA7-EVAL-08

层级：

```text
32 clips × 2 seeds：快速错误检查
128 clips × 2–3 seeds：晋级
至少 256 clips，优先完整 732 val：论文主结果与 FVD
```

FVD 不得在 16/32 clip 上使用。

---

# 9. Baseline 与消融矩阵

## 9.1 Pair construction

- independent seeds；
- common-prefix siblings；
- Base re-noise；
- no structural alignment。

## 9.2 Preference rule

- weighted scalar score；
- top-vs-bottom；
- Pareto/non-inferiority；
- no anti-collapse guard；
- no uncertainty normalization；
- shuffled labels。

## 9.3 Credit assignment

- clip-level；
- two-half segment；
- overlapping 4-frame windows；
- uniform segment weights；
- confidence-weighted segments。

## 9.4 Optimization

- chosen-only SFT；
- AWR；
- vanilla Diffusion-DPO；
- DPO + fixed loser scale；
- DPO + adaptive safeguard；
- no SFT anchor；
- independent SFT noise；
- noise-aligned SFT。

## 9.5 Scorer/evaluator

- RAFT scorer / CoTracker evaluator；
- RAFT scorer / RAFT evaluator，作为 metric hacking control；
- generic motion smoothness score；
- P-UNC projection energy。

核心结果必须证明：

\[
\text{P-UNC physics preference}
>
\text{generic smoothness preference}
\]

并且：

\[
\text{dense safeguarded DPO}
>
\text{clip-level DPO}.
\]

---

# 10. 最终评估协议

## 10.1 Primary metrics

预注册两个 primary endpoint：

1. **CoTracker camera-compensated acceleration outlier rate**，越低越好；
2. **CoTracker track survival / valid coverage**，非劣且优先越高越好。

Secondary：

- image-plane jerk outlier；
- track length；
- identity continuity；
- background residual；
- P-UNC projection energy；
- motion direction/turn preservation。

## 10.2 Anti-collapse metrics

必须报告：

- VBench dynamic degree；
- net displacement；
- mean velocity；
- track survival；
- active-track fraction；
- candidate motion histogram。

任何 dynamics 改善若伴随系统性 dynamic-degree 降低，均视为失败。

## 10.3 Visual metrics

- VBench motion smoothness；
- subject consistency；
- background consistency；
- temporal flicker；
- DINO/Base identity distance；
- outside/overall LPIPS to matched Base；
- blind human visual quality preference；
- FVD，仅大规模。

GT future LPIPS 只能作次要诊断，因为 I2V 未来并非唯一。

## 10.4 统计

统计单位为 scene/condition：

1. 同一 clip 内先聚合 generation seeds；
2. scene 内聚合 clip；
3. paired hierarchical bootstrap；
4. 10,000 bootstrap samples；
5. 报告 mean、median、win rate、95% CI、worst 10%、coverage；
6. 两个 primary 指标使用 Holm correction；
7. 无有效 track 的 clip 为 `invalid`，不填 0；
8. 报告每个指标实际 \(n\)。

## 10.5 正式 promotion 标准

全部满足才允许论文主张：

1. acceleration outlier 的 paired 95% CI 上界小于 0；
2. survival/coverage 不劣于 Base 超过预注册 margin；
3. dynamic degree 退化不超过 5%，且无 near-static collapse；
4. visual quality 无显著恶化；
5. CoTracker 与盲法人工 verdict 同向；
6. 两个 training seeds 同向；
7. 至少两个 generation seeds 同向；
8. 完整方法超过 AWR、chosen-only SFT 和 structure-aligned clip DPO；
9. training P-UNC 不是唯一改善指标；
10. 收益不由单一 branch、scene strata 或少数 outlier 主导。

---

# 11. 资源与单卡/双卡策略

## 11.1 单卡阶段

PA0–PA4 均在单张 4090 完成。

每个阶段必须测量：

```text
seconds/video
seconds/score
seconds/update
peak_vram
bytes/video
bytes/pair
```

预算：

\[
H_1
=
\frac{
N_vt_{\mathrm{gen}}
+
N_vt_{\mathrm{score}}
+
N_ut_{\mathrm{update}}
}{
3600
}.
\]

AutoDL 价格在实际切卡当天填写，不在计划中虚构。

## 11.2 双卡触发

只有 PA4 通过后，用户执行一次停机切双卡。切换后持续完成 PA5–PA7，不频繁切回。

双 4090 不等于 48 GB 单样本显存。默认策略：

- 两个 generation worker；
- 两个 scoring worker；
- 两个独立 training seed；
- 两个 baseline 并行。

DDP 仅用于吞吐，不用于解决 OOM。

## 11.3 磁盘

新单卡实例于 2026-07-15 实测可用磁盘约 74 GB（以每次 run 前 `df` 为准）。正式 scale-out 前必须根据 PA1 实测：

\[
S_{\mathrm{total}}
=
N_v(S_{\mathrm{mp4}}+S_{\mathrm{latent}}+S_{\mathrm{diag}})
+
S_{\mathrm{checkpoints}}.
\]

要求保留至少 30 GB 安全空间。超出预算时：

- 保留 mp4、VAE latent、manifest；
- 删除可重建的中间 tensor；
- 不删除正式 run summary 和 review；
- 不降低视频质量到影响 scorer。

---

# 12. 代码结构与测试

建议新增：

```text
motion_proj/preference/
  candidates.py
  branch_runner.py
  scorer.py
  constraints.py
  segments.py
  pairs.py
  dataset.py
  dpo_loss.py
  sdpo_safeguard.py
  awr_loss.py
  trainer.py

motion_proj/diagnostics/
  physics_dpo_horizon.py
  physics_dpo_branch.py
  physics_pair_validity.py
  physics_dpo_capacity.py

motion_proj/eval/
  physics_preference_eval.py

configs/preference/
  horizon.yaml
  branch.yaml
  scorer.yaml
  pair.yaml
  dpo.yaml
  awr.yaml
  screen.yaml
  formal.yaml
```

必须新增测试：

```text
tests/test_preference_scene_split.py
tests/test_common_prefix_branch.py
tests/test_antithetic_perturbation.py
tests/test_candidate_provenance.py
tests/test_physics_feasibility.py
tests/test_physics_pareto_preference.py
tests/test_segment_labels.py
tests/test_dpo_loss.py
tests/test_dense_dpo_loss.py
tests/test_sdpo_safeguard.py
tests/test_awr_loss.py
tests/test_reference_adapter_state.py
tests/test_reference_cache_equivalence.py
tests/test_noise_alignment.py
tests/test_preference_dataset_fail_closed.py
tests/test_hierarchical_bootstrap.py
```

## 12.1 Reference 计算策略

单卡 baseline：

```text
policy forward with LoRA
disable LoRA
reference no-grad forward
restore LoRA
```

必须验证 adapter state 恢复。

若 reference forward 成为主要吞吐瓶颈，可建立固定 noise bank reference cache：

```text
K = 4 or 8 noise/timestep draws per pair
```

启用前必须证明 cached 和 online reference 在相同 bank 上 exact。固定 bank 只用于 PA3/PA4；正式训练需报告是否存在 bank overfit。

## 12.2 每次提交前

```bash
git status --short
PYTHONPATH=. pytest -q
git diff --cached --check
git diff --cached
```

禁止：

- 修改 `sys.path` 掩盖 import；
- 在 dirty worktree 构建正式 candidate；
- 覆盖旧 run；
- 自动 push；
- 提交模型权重、视频或第三方 checkpoint。

---

# 13. 停止条件

出现任一项即停止对应路线：

## Pair construction

- sibling candidates 仍发生大构图/身份变化；
- candidate 差异低于 tracker/scorer noise；
- branch ID 系统性决定 winner；
- 结构对齐不优于 independent-seed；
- 有效 pair rate低于 25%。

## Scorer

- P-UNC/E0 人审失败；
- scorer-human agreement低于 75%；
- Wilson lower bound不高于 50%；
- winner 频繁低运动；
- preference components 系统冲突；
- 只有加权 scalar 能产生足够 pair。

## DPO kernel

- zero-adapter 不等于 \(\log2\)；
- reference 有梯度；
- pair noise 未共享；
- winner error 与 margin 同时上升且 safeguard 无效；
- 1/8-pair capacity 不成立；
- held-out preference accuracy不超过 60%。

## Screening

- proposed 不超过 AWR/chosen-only SFT；
- 只在 P-UNC 上改善；
- independent CoTracker 不支持；
- dynamic-degree collapse；
- 视觉质量明显退化；
- 人工偏好不支持；
- 需要切双卡才能完成方法正确性验证。

## Formal

- 两训练 seed 方向不一致；
- bootstrap CI 不通过；
- 结果由单一 strata/branch 主导；
- 128 clip 改善无法在 256/full val 复现；
- 方法贡献不能从 DenseDPO/VideoDPO/VideoGPA/SHIFT 中区分。

---

# 14. 推荐执行时间表

| 时间 | 任务 |
|---|---|
| 7 月 14–15 日 | PA0 人审、scene split、文档与 schema |
| 7 月 15–17 日 | PA1 horizon/branch pilot |
| 7 月 17–20 日 | PA2 32-condition pair 构建与 30-pair 人审 |
| 7 月 20–23 日 | PA3 DPO/AWR/SDPO kernel 与容量 |
| 7 月 23–28 日 | PA4 单卡 screening |
| PA4 通过后 | 用户停机，切双 4090 |
| 双卡第 1 周 | PA5 candidate/pair scale-out |
| 双卡第 2 周 | PA6 两 seed 正式训练与消融 |
| 双卡第 3 周 | PA7 128/256+ clip 评估与人工盲审 |
| 通过后 | PA8 论文主表、failure analysis、第二 backbone 决策 |

时间窗口不允许覆盖 gate。

---

# 15. 论文叙事草案

## 15.1 问题

现有 video preference alignment 使用独立生成 pair 或全局视频分数，容易将：

- 视觉伪影；
- 运动量；
- 轨迹物理；
- 身份变化；

混为同一个 preference，导致低运动偏置和错误时间信用。

## 15.2 方法

SAP-DPO：

1. 从相同 SVD 去噪前缀构造 sibling video candidates；
2. 以 uncertainty-aware dynamics projection distance 形成可解释 partial order；
3. 用 hard motion/quality non-inferiority 防止静止化；
4. 产生局部 4-frame physics preference；
5. 用 dense Diffusion-DPO 进行 temporal credit assignment；
6. 用 winner-preserving safeguard 与 noise-aligned SFT anchor 稳定训练。

## 15.3 可接受的贡献表述

只有实验成立后才允许：

- 一个无 future-GT 的驾驶视频物理偏好构造框架；
- 一个结构对齐、局部时间标注的 driving video preference dataset；
- 一个防低运动塌缩的 dense physics preference objective；
- 在公开驾驶视频上的单卡可运行、双卡可复现结果。

不得声称：

- 学到了牛顿物理；
- 解决了长期世界模型；
- image-plane jerk 等于真实 jerk；
- 自动 scorer 取代人工评价；
- 适用于任意视频模型而未做第二 backbone。

---

# 16. Reviewer 2 最可能的攻击点

1. **“这只是 DenseDPO + driving score。”**
   必须通过 uncertainty-aware projection partial order、anti-collapse constraints 和对应消融回答。

2. **“physics scorer 不是真物理，只是 tracker smoothness。”**
   必须有人审、independent CoTracker、synthetic sanity、motion floor 和 failure cases。

3. **“模型只是少动了。”**
   dynamic degree、net displacement、survival 和 human motion comparison 必须非劣。

4. **“pair construction 本身带来收益，与 DPO 无关。”**
   必须有 chosen-only SFT、AWR、clip-DPO、dense-DPO 同数据对照。

5. **“训练 scorer 与 evaluator 同源。”**
   RAFT/P-UNC 造 label，CoTracker + human 做正式评估。

6. **“8 帧不足以谈物理。”**
   优先 14 帧；若只能 8 帧，只声称 short-horizon。

7. **“标准 DPO 通过破坏 loser 获益。”**
   报告 winner absolute loss，并使用 safeguard。

8. **“只在一个 SVD backbone 上成立。”**
   第一阶段可以只用 SVD，但正式投稿前需评估第二 backbone 的最低迁移成本；未迁移时明确局限。

---

# 17. 外部研究依据

- [R1] Wallace et al., **Diffusion Model Alignment Using Direct Preference Optimization**, CVPR 2024.
- [R2] Liu et al., **VideoDPO: Omni-Preference Alignment for Video Diffusion Generation**, CVPR 2025.
- [R3] Wu et al., **DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models**, NeurIPS 2025 Spotlight.
- [R4] Fu et al., **Diffusion-SDPO: Safeguarded Direct Preference Optimization for Diffusion Models**, 2025.
- [R5] Du et al., **VideoGPA: Distilling Geometry Priors for 3D-Consistent Video Generation**, 2026.
- [R6] Ye et al., **SHIFT: Motion Alignment in Video Diffusion Models with Adversarial Hybrid Fine-Tuning**, 2026.
- [R7] **SIFT: Self-Imagination Fine-Tuning for Physically Plausible Motion in Video Diffusion Models**, 2026.
- [R8] VBench official benchmark and implementation.

实现时优先参考官方代码：

```text
SalesforceAIResearch/DiffusionDPO
CIntellifusion/VideoDPO
DenseDPO official project/code if released
AIDC-AI or ATH-MaaS/Diffusion-SDPO official implementation
VBench official repository
facebookresearch/co-tracker
```

不得从非官方博客复制关键 loss。

---

# 18. 给 Coding Agent 的唯一执行顺序

```text
PA0 人审
→ scene-level split 与 schema
→ PA1-HORIZON
→ PA1-BRANCH
→ PA2 pair legality + 30-pair blind review
→ PA3 DPO/AWR/SDPO algebra
→ 1/8/32-pair capacity
→ PA4 单卡 baseline screening
→ [仅通过后，用户停机切双 4090]
→ PA5 candidate scale-out
→ PA6 两 seed 正式训练与消融
→ PA7 独立评估
→ PA8 promote / reject
```

任何 gate 未通过时停在该处。不得用更多数据、更多更新或更多 GPU 绕过失败。

---

# 19. 当前唯一下一步

PA0 人审、唯一 scene split 与 PA1 horizon profile 已完成：

```text
autoresearch-pa0-scene-split-s20260715-v1
split fingerprint e525edf33bcfec169c0077d2eb2e528d953dbc9930e771c803c889a32983c73a
→ autoresearch-pa1-horizon-s20260715-v3
→ PA1-BRANCH-02
```

`autoresearch-pa1-horizon-s20260715-v1` 与 v2 已保留为 trace-hash 实现失败证据，均未形成有效 candidate/score/horizon 结论。v3 用 clean `57987b0` 完成：两种帧数均为 exact Base rerun，CoTracker aggregate repeatability 均为 0 相对差；14 帧 peak 为 `5.8049 GB`（阈值 `22 GB`），相对 8 帧 generation slowdown 为 `1.5799×`（阈值 `2.2×`）。因此冻结 `num_frames=14`，但仅允许 **short-horizon dynamics alignment** claim；该决定仅来自预注册资源规则。

`autoresearch-pa1-branch-s20260715-v1` 已作为 manual-continuation exact guard 的失败证据保留；v2 已作为 schema 数值容差失败证据保留；v3 已完整完成但 machine-blocked，三者均不进入 pair 或训练。当前唯一可执行的 GPU 工作是：

```text
autoresearch-pa1-branch-s20260715-v4
4 conditions × (1 exact Base guard + 4 common-prefix sibling candidates)
14 frames、25 denoising steps、fork 0.6、medium rho 0.02
```

在 PA1-BRANCH-02 family/fork/strength 冻结前禁止：

- DPO/AWR 实现；
- 训练；
- 切双卡；
- 扩 cache。

PA0 已汇报并保留以下证据：

```text
P-UNC human review
CoTracker human review
clean `57987b0` worktree
materialized scene split、schema 与 PA1 profile
next gate：PA1-BRANCH-02；冻结 candidate family/fork/strength 后才可进入 PA2-PAIR-03
```
