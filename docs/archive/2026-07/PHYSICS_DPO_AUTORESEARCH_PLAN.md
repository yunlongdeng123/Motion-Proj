# Motion-Proj Physics-DPO 自动研究计划

> 决策日期：2026-07-14
>
> 计划状态：`running`（只读核查已完成；任何生成、标注或训练任务尚未启动）
>
> 决策基线：`16b6975`（`main`，worktree clean）
>
> 当前硬件：单张 RTX 4090 24 GB；GPU 空闲；`/root/autodl-tmp` 实测可用约 164 GB
> 取代范围：本文件取代旧计划中**未来**的「停止 reward/DPO/GRPO 主线」限制；不修改或弱化 P1、F0、F1、V1/V2 的历史失败结论。

## 0. 这次转向的精确定义

当前被否定的是：

```text
Base rollout
→ 连续轨迹 correction
→ RGB crop / resize / paste
→ VAE 或 hybrid latent target
→ shared temporal-LoRA endpoint regression
```

它的问题是 counterfactual target 不合法，而不是“生成视频之间不能作偏好比较”。本计划新建一个独立研究问题：

```text
同一条件、同一初始噪声、同一去噪前缀的真实 SVD candidate videos
→ 无 future-GT 的 physics preference scorer
→ 置信度、运动量与画质共同过滤
→ 离线 Diffusion-DPO / AWR baseline
→ Base-matched 完整 rollout 的独立评估
```

chosen/rejected 都必须是模型实际解码得到的 RGB 视频：

\[
X^+ \succ X^-.
\]

本路线不再构造、编码或回归不存在的 \(X^\dagger\)。P-UNC 只用于无梯度的轨迹诊断/偏好打分，不再把其轨迹修正渲染成 RGB target。

### 不变的禁止事项

- 不使用 future ego、future box、future track、source-future metadata 或人工 GT 未来构造 label/reward。
- 不重用 P1 的 crop/paste、full/hybrid VAE target、V1 synthetic cache 或 V5 projected latent 作为训练样本。
- 不通过完整 25-step sampling chain 反传；初版只做离线 forward-process Diffusion-DPO。
- 不训练独立 reward model，不把 CoTracker3、RAFT 或 P-UNC 置于反向传播图中。
- 不在未过门禁时扩大数据、延长训练、扫描超参或切双卡。
- 不把训练 label scorer 的结果当成独立测试指标。

## 1. 当前证据与可复用资产

| 项目 | 已核实状态 | 对 Physics-DPO 的含义 |
|---|---|---|
| C0 official SVD protocol | `autoresearch-c0-conditioning-s20260714-v2` completed；官方 pipeline、wrapper、candidate condition 与 rerun 均 exact；旧 one-step conditioning 与官方分支不等价 | 新生成和新 DPO condition 必须使用版本化 `svd_official_v1`，不能调用 legacy `build_conditioning()` 作为 rollout 对齐证据 |
| P1 target legality | `autoresearch-p1-target-s20260714-v2` machine hard fail；存在零 RGB realization、LPIPS 0.06805、source duplication、无 depth-order overlap | 直接放弃 endpoint/RGB target；真实 candidate pair 绕开该问题 |
| P-UNC projector | machine eligible；无 future-GT、SNR 与运动不变量检查已实现，但 12-panel 人审仍是 `awaiting_reviews` | 可改造成 frozen physics diagnostic/reward，不可当合法 renderer |
| E0 CoTracker3 | v3 machine pass、独立 tracker provenance/重跑/扰动排序通过；12-panel 人审仍是 `awaiting_reviews` | 作为独立 hold-out evaluator 的候选；通过人审前不能用于晋级结论 |
| SVD/LoRA | `SVDBackbone` 有 v-pred API、关闭 LoRA 的 frozen reference、temporal-only rank-16 LoRA、adapter provenance | 单卡可实现 sequential policy/reference forward；不需要两份模型同时占显存 |
| 现有训练 | 正式 `Trainer` 没有 DPO；V2 loss 只在 capacity pilot 中使用 | 必须从 loss 单测与 1-pair capacity 开始，不能把旧 trainer 当作已验证 DPO |
| 运行基础设施 | manifest、resolved config、fingerprint、metrics JSONL、checkpoint、atomic terminal state 已有 | 每个新 gate 使用新的 run ID；不得覆盖历史 run/cache |

旧路线的失败证据仍有效：V1 的 LPIPS 改善与 dynamics 恶化并存；C/D/E 与 F0 证明当前 shared temporal LoRA endpoint 没有 locality 可行点；它们不等价于对全局偏好对齐的否定。

## 2. 可答辩的问题与停止边界

### 2.1 研究问题

在不使用 future-GT、外部 motion condition、counterfactual renderer、reward-model 训练或 sampling-chain backprop 的前提下，是否能用**结构对齐的真实 SVD rollout pair**和可解释的驾驶动力学偏好，使 LoRA 微调后的完整 rollout 在不发生低运动/画质塌缩时提升独立轨迹指标？

### 2.2 不是论文结论的内容

- “physics reward 较高”本身不是物理真实性。
- 仅 CoTracker3 或 P-UNC 变好，不构成独立收益。
- 只在 8/32 个 clip 上的 DPO margin 增长，不构成 rollout 改善。
- 仅比较完全独立 seed 的视频，不构成有效 preference pair。
- 只证明 Physics-DPO 有效、而没有与随机 pair、real-only SFT、AWR 对照，不构成机制证据。

### 2.3 新路线的停止线

以下任一项触发时，不切双卡、不扩容 pair、不启动长训：

1. P-UNC 或 CoTracker3 的人工校验未通过；
2. 结构对齐 pair 的人工偏好与 scorer 一致性未达门槛；
3. winner 经常以低 dynamic degree、低 track survival 或明显画质损坏取胜；
4. Diffusion-DPO 在 1/8/32 pair 的代数、reference、容量门禁失败；
5. 单卡小规模 rollout 的独立 evaluator 与人工复核不支持同方向改善；
6. 最终正式对照未超过 AWR / real-only SFT，或只在训练 scorer 上有效。

触发后将该子路线标记 `rejected` 或 `blocked`，保存证据；不得以更多 steps、更多 pairs 或切双卡掩盖失败。

## 3. 方法预注册：结构对齐的离线 Physics-DPO

### 3.1 Candidate 生成：共享来源、后段分叉

每个 condition 的候选必须共享：

- 同一个 conditioning frame；
- 同一个 official condition-noise 和 initial video latent；
- 同一 Base checkpoint、25-step scheduler、motion bucket、fps；
- 同一去噪前缀；
- 相同输出分辨率、8-frame 语义与 decode 路径。

candidate runner 以 C0 已验证的 `svd_official_v1` 为唯一基础。在固定第 15/25 个去噪步后才允许分叉；默认每个 condition 产生四个真实视频：

```text
B0: 精确 Base continuation
B1: 小幅 tail-CFG branch
B2: 反向小幅 tail-CFG branch
B3: 预注册、归一化且有上界的 late-latent stochastic branch
```

PA1 仅允许在三档有界 branch strength 中选择一个，使候选既非逐像素重复、也不产生大范围构图跃迁。选择规则是：取满足下列门槛的最小 strength；三档都不满足则该 branch family `rejected`，而不是继续连续搜索。

- B0 在固定 C0 seed 下逐项复现 official trace；
- 所有 candidate finite、可解码，且 conditioning frame 一致；
- 每 condition 至少有两个非重复 candidate；
- pairwise future-frame LPIPS 的中位数处于预注册的 `[0.01, 0.25]`；
- branch id 不能在后续 winner 中系统性垄断（任一 id 的 winner 占比不得超过 40%）。

branch 不是 target 编辑：它只是从共同 latent prefix 继续采样，输出始终是真实 SVD RGB 视频。每条视频记录 `condition_id`、initial-latent hash、prefix-latent hash、fork step、branch recipe、seed、Base/代码/config fingerprint 与完整 generation settings。

### 3.2 Physics preference vector：防止“少动即高分”

pair label 的训练 scorer 为 frozen P-UNC/RAFT diagnostics；其 output 不进入梯度图。对 candidate \(X\)，先以有效 coverage、置信度、visibility、track length 过滤，再构造归一化 reward vector：

\[
R(X)=
w_a S_{\mathrm{acc}}+
w_j S_{\mathrm{jerk}}+
w_s S_{\mathrm{survival}}+
w_m S_{\mathrm{dynamic}}+
w_q S_{\mathrm{quality}}+
w_c S_{\mathrm{confidence}}.
\]

其中：

- \(S_{\mathrm{acc}}\)、\(S_{\mathrm{jerk}}\)：仅在高置信、camera-compensated 的 generated point tracks 上比较；使用 robust rank/clip normalization，不能由空轨迹填零。
- \(S_{\mathrm{survival}}\)：有效轨迹存活、可见性与身份连续性。
- \(S_{\mathrm{dynamic}}\)：相对同 condition 的 B0 保留 net displacement、mean velocity、direction、turn sign 与 dynamic degree 的区间分数；低运动不能靠低 acceleration 获胜。
- \(S_{\mathrm{quality}}\)：future-frame temporal flicker、明显 decode/appearance 损坏和与 B0 的视觉偏离联合 guard；GT future LPIPS 只作次要诊断，不能决定 winner。
- \(S_{\mathrm{confidence}}\)：可评分 track 覆盖、P-UNC SNR 和 component validity；无有效 component 为 `invalid`，不是 0 分。

在 PA2 前必须把权重、normalizer、有效 coverage、margin 和每一项 hard filter 写进版本化 YAML；不会以 rollout 结果倒调。初版采用 component-wise rank，不训练标量 reward model。

candidate 只有同时满足下列条件才可进入 pair pool：

```text
uses_future_gt = false
candidate/track/score 全部 finite
有效轨迹、可见性和 coverage 通过下限
dynamic-degree / net-displacement / direction / turn 处于 B0 容许区间
没有 quality hard failure
不属于 catastrophic Base generation
```

同 condition 最多保留一个 chosen/rejected pair。label 仅在 \(R(X^+)-R(X^-)\ge\delta\) 且所有 anti-collapse guard 成立时写为 \(X^+\succ X^-\)。模糊 pair、quality 更差的 “winner”、或只因 motion 变小而胜出的 pair 一律丢弃。

### 3.3 独立性与人工校准

- **训练 label scorer：**P-UNC/RAFT diagnostics；只读、无 future-GT、与候选生成器分离。
- **独立正式 evaluator：**官方 CoTracker3 E0 v3 的 evaluator-only grid/affine-camera protocol；不读取 P-UNC query、track、projector 或 pair label。
- **人工审查：**盲法查看同 condition 的两条真实视频、track overlay 和基础质量信息；先做不暴露自动 winner 的 verdict，再对照 scorer。
- **次级稳健性检查：**用分离的 RAFT aggregation 报告，但不把它伪装成 CoTracker3 独立结论。

P-UNC 的 12-panel review 和 E0 v3 的 12-panel review 都必须完成。P-UNC 采用既有 `>=87.5%` 有效 verdict 门槛（即至少 11/12）；E0 新增 `>=10/12` decisive human alignment 门槛。任一失败时，当前 scorer/evaluator 不可静默替换，Physics-DPO `blocked`。

### 3.4 Offline Diffusion-DPO 目标

对真实 pair \((x^+,x^-)\)，用官方对齐 conditioning \(c_{\mathrm{off}}\)，并在 pair 内共享 \(\sigma,\epsilon\)：

\[
z^\pm=x^\pm+\sigma\epsilon,
\qquad
\ell_\theta(x)=w(\sigma)
\left\|v_\theta(z,\sigma,c_{\mathrm{off}})-v^*(z,\sigma,x)\right\|_2^2,
\]

其中 \(v^*\) 由 `model_output_from_x0` 从真实 candidate latent 得到。冻结 reference 以关闭 LoRA 的同一 SVD 依次 no-grad 计算，避免在单张 24 GB 卡上同时常驻两份模型。DPO logit 为：

\[
d_\theta=
[\ell_\theta(x^-)-\ell_{\rm ref}(x^-)]
-[\ell_\theta(x^+)-\ell_{\rm ref}(x^+)],
\qquad
\mathcal L_{\rm DPO}=-\log\sigma(\beta d_\theta).
\]

初版仅用该目标与固定 temporal-only rank-16 LoRA；通过 PA3 后才加入小权重、固定配置的 real-video SFT anchor：

\[
\mathcal L=\mathcal L_{\rm DPO}+\lambda_{\rm real}\mathcal L_{\rm real}.
\]

这里的 SFT 是全局视觉保留正则，不再声称能解决旧 endpoint 的空间 locality。reference KL proxy、chosen/rejected margin、real loss、dynamic degree 与 adapter norm 必须逐 step 记录。

## 4. 里程碑与自动决策表

状态词严格使用 `pending / running / blocked / done / rejected`。每项任务均保存新的 run ID、resolved config、manifest、数据 fingerprint、JSONL 指标、summary 与终态；不得覆盖任何历史 autoresearch run。

| ID | 状态 | 单卡工作与硬门禁 | 通过后 | 失败时 |
|---|---|---|---|---|
| PA0-EVAL-00 | pending | 完成 P-UNC 与 E0 v3 各 12 个既有 panel 的盲法人工 review；核对 manifest、weight hash、no-future-GT | 解锁 label scorer 与独立 evaluator | `blocked`；不生成 DPO 数据，不替代 provider |
| PA1-BRANCH-01 | pending | 为 `svd_official_v1` 实现可追溯 common-prefix branch runner；4 condition × 4 candidate 的 parity/diversity/profile | 冻结 branch recipe 和成本模型 | `rejected`；不退回 independent seed pairs |
| PA2-PAIR-02 | pending | 32 train conditions × 4 candidates = 128 视频；pair filter、score decomposition、20 条盲法 pair review | 解锁 DPO kernel 与小数据训练 | `rejected`；不扩大 candidate 规模 |
| PA3-DPO-03 | pending | DPO 单测；1-pair、8-pair overfit；32-train/8-held-out pair 的 single-card capacity | 解锁 128-condition 单卡 screening | `blocked`；只修正代数/condition/provenance bug |
| PA4-SCREEN-04 | pending | 128 conditions × 4 candidates；目标 100–250 pairs；100–500 update、一训练 seed、16 held-out clips × 2 gen seeds | 允许一次性切双卡 | `rejected`；不切卡、不长训 |
| PA5-SCALE-05 | pending | 双卡后 256–512 conditions × 4 candidates；300–800 filtered pairs；双 worker 生成+打分 | 解锁正式比较 | `blocked`；保留单卡结果和 profile |
| PA6-FORMAL-06 | pending | Physics-DPO、AWR、real-only SFT 的两个 training seed；每 run 500–1,000 update | 解锁 128-clip/论文级评估 | `rejected`；不以 DDP 或更多 step 覆盖 |
| PA7-PROMOTE-07 | pending | 32-clip 快速、128-clip晋级、2–3 generation seeds、独立 CoTracker3 与盲法人工评估 | 写入主表或进入论文阶段 | `rejected`；归档为新路线负结论 |

### PA0-EVAL-00：零训练前置门槛

这一步不占 GPU。人工 review 不是可以自动填补的字段：若需要人来判定，则 run 保持 `awaiting_reviews`。完成后写新 review aggregation run，保留原始 template 与逐条 verdict。只有 scorer/evaluator 均通过时，才允许 PA1 使用 GPU。

### PA1-BRANCH-01：先证明 candidate generator 合法且可控

实现前先抽取 C0 parity trace 的 unrolled sampler；B0 continuation 必须在固定 seed 上与 official 25-step trace 一致。PA1 只可做：

```text
4 conditions × 4 candidates
25 denoising steps
无 adapter、无 cache write、无训练
```

产物包括 `candidate_manifest.jsonl`、per-step/prefix latent hash、branch settings、mp4、VAE latents、throughput profile 和 branch parity report。该 profile 写出每视频 generation/score 秒数、峰值显存、磁盘占用，并据此计算后续预算；不得在未测吞吐时虚报 GPU-hour 或费用。

### PA2-PAIR-02：先审 pair，不训练

固定 32 个 train conditions，按 condition 留出法保留独立 validation conditions。每 condition 至多一个 pair，目标至少 24 个有效 condition/pair；不足 24 或 P-UNC 质量/动态 guard 不通过则不训练。对最多 20 条随机、分 strata 的有效 pair 做盲法人工审核：

- decisive pair 的 scorer–human preference agreement `>=70%`；
- chosen 的 low-dynamic / track-survival / visual-quality failure 计数为 0；
- branch id 与 winner 的列联检查没有单一 recipe 垄断；
- 训练 condition 与评估 condition 不重叠；
- 所有 label 有 reward decomposition、margin、filter reason、review provenance。

PA2 完成的是“偏好 label 是否可信”，不报告任何模型改善。

### PA3-DPO-03：代数与最小容量门槛

新增 DPO 实现前必须先有以下 tests：

```text
zero adapter => d_theta = 0 且 loss = log(2)
swap chosen/rejected => logit 符号翻转
共享 sigma/noise 的 pair trace 可复现
reference 无梯度、LoRA 开关状态可恢复
official conditioning 与 C0 conditional branch 对齐
candidate VAE latent / frame count / condition_id / fingerprint fail-closed
NaN/Inf、空 pair、future-GT、重复 condition fail-closed
```

容量顺序固定为 `1 pair → 8 pair → 32 train / 8 held-out pair`。只有在前一层通过后才运行后一层；不做 LR、beta、LoRA rank 的网格搜索。通过标准：finite、显存不超过 24 GB、训练 chosen–rejected margin 单调上升、reference drift 受控、held-out pair 的正 margin 比例至少 60%、real-loss/adapter-norm 不发生灾难性上升。若失败，仅定位 DPO algebra、official conditioning、noise sharing 或数据 provenance；不直接长训。

### PA4-SCREEN-04：当前单卡上的方法价值判断

保持单卡，将 candidate pool 扩到 `128 × 4 = 512` 条真实视频，筛选 100–250 pair。固定一个 training seed、100–500 update，使用 16 个未见 validation clips、每 clip 两个 matched generation seed、25 steps。与以下共同预算的基线比较：

```text
E0  Frozen Base
E1  real-only SFT LoRA
E2  offline reward-weighted regression / AWR
E3  Physics-DPO
E4  shuffled-label DPO（诊断 control，不进主表）
```

PA4 切双卡的条件必须同时成立：

- CoTracker3（独立）与 P-UNC（训练 scorer）在有效 clip 上不相反；
- 每个有效 primary dynamics component 的方向性胜率至少 60%，且 dynamic degree、track survival、visual guard 没有系统恶化；
- 16-clip 结果只称 screening，不称统计显著，但盲法人工 pairwise 不显示明显 motion/quality 投机；
- replay pair p95 exposure `<=8`，无 NaN/OOM，所有 provenance 完整；
- E3 至少不差于 E1/E2；若只优于 Base，不晋级。

## 5. 单卡到双卡的成本控制

### 5.1 现在保持单卡

PA0–PA4 全部在当前单卡完成。单卡足以完成 evaluator/reward 单测、128 candidate、DPO loss、1/8/32 pair capacity、小数据训练和 16-clip screening。当前没有任何理由因吞吐而提前停机切卡。

每个阶段首先记录真实 `seconds/video`、`seconds/score`、峰值显存和 bytes/video；预算公式为：

\[
H_{1}=\frac{N_{\rm video}t_{\rm gen}+N_{\rm video}t_{\rm score}+N_{\rm update}t_{\rm update}}{3600},
\qquad
\mathrm{cost}=H\times\mathrm{AutoDL\ hourly\ price}.
\]

价格由切卡当天的 AutoDL 控制台填写，不能用旧价格估算。未通过 PA4 前，candidate 数与 update 上限就是成本硬上限。

### 5.2 只在 PA4 通过后一次切双卡

切换需要用户停机、重新开双 4090，因此这是**操作员动作**，不是自动任务。切换后持续完成 PA5–PA7，不在主实验中频繁换回单卡。

双卡不会自动形成一张 48 GB 显存卡；普通 DDP 仍是每 GPU 一份 24 GB 模型。初版优先使用两张卡的独立吞吐：

```text
GPU 0: conditions / videos / evaluator shard 的偶数部分
GPU 1: conditions / videos / evaluator shard 的奇数部分
```

正式训练优先并行研究证据，而不是立即 DDP：

```text
GPU 0: Physics-DPO seed 0        GPU 1: Physics-DPO seed 1
随后：AWR 两 seed；随后：real-only SFT 两 seed
```

仅当已完成两个独立 seed，且单卡 500 update 超过可接受夜间窗口、或 dataset 超过约 500 pair 导致 global batch 成为唯一瓶颈时，才考虑两卡 DDP。DDP 仅为吞吐，不作为解决 OOM 或方法正确性的手段。

### 5.3 PA5–PA7 的双卡规模与评估

- `256–512 conditions × 4 candidates = 1,024–2,048` 视频；筛到 `300–800` high-quality pairs。
- 生成、score、review export 使用两个独立 worker；每条 worker 有独立 run shard、seed range 与 manifest，最终由 immutable merge manifest 聚合。
- 每个方法至少两个 training seed；每 run 固定 500–1,000 update，而不是用训练长度掩盖失败。
- 先 32 clips × 2 generation seeds 快筛，再 128 clips × 2–3 seeds 晋级；Base 与所有 adapter 使用相同 condition、initial noise、sampler、25 steps 与 seed。
- 统计以 condition/clip 为单位：先在同一 clip 内聚合 generation seeds，再做 paired hierarchical bootstrap；不把 seeds 当独立样本。

## 6. 最终评估与论文门槛

主报告必须同时包含：

| 维度 | 必报内容 |
|---|---|
| 动力学 | CoTracker3 camera-compensated acceleration/jerk、track survival、track length、valid coverage；独立于训练 P-UNC |
| anti-collapse | dynamic degree、net displacement、mean velocity、direction、turn preservation；无效 component 显式 `invalid` |
| 视觉 | temporal flicker、appearance/identity consistency、Base proximity、人工盲法质量 preference；GT LPIPS 仅次级 |
| 对照 | Base、real-only SFT、AWR、Physics-DPO、shuffled-label diagnostic |
| 统计 | paired delta、mean、median、win rate、worst 10%、bootstrap 95% CI、coverage 与每个 component 的样本量 |
| 可追溯性 | Base/model/data/pair/config/reward/evaluator fingerprints；train/generation seeds；review file；完整 run terminal state |

PA7 promotion 同时要求：

1. Physics-DPO 在至少一个独立 primary dynamics metric 的 paired bootstrap 95% CI 中改善；
2. 第二个有效 dynamics metric 的均值退化不超过 5%；
3. dynamic degree、track survival、valid coverage 与视觉质量相对 Base 的退化均不超过预注册阈值；
4. CoTracker3 与盲法人工 review 方向一致，P-UNC improvement 不能是唯一证据；
5. 两个 training seed、两个以上 generation seed 同向；
6. E3 超过 real-only SFT 与 AWR，而非只超过 Base；
7. 不存在某 branch recipe、某单一 condition strata 或低运动样本独占收益。

若这些条件未同时满足，结论应是“当前 structured Physics-DPO 未通过”，而不是将 DPO margin、训练 reward 或单一 tracker 解释为主结果。

## 7. 计划的代码与产物边界

本文件只预注册工作，不在本次文档变更中实现训练。实现时新增路径应与旧 endpoint 路径隔离，建议：

```text
configs/preference/physics_dpo_{branch,pair,dpo,eval}.yaml
motion_proj/preference/{candidates,scorer,pairs,dpo_loss,dataset}.py
motion_proj/diagnostics/physics_dpo_{branch,pair_validity,capacity}.py
motion_proj/eval/physics_preference_eval.py
tests/test_{physics_dpo_branch,pair,dpo_loss,official_conditioning}.py
/root/autodl-tmp/cache/physics_dpo/<dataset-id>/
/root/autodl-tmp/runs/physics-dpo/<run-id>/
```

dataset 中保存真实 candidate RGB（可压缩 mp4）、VAE latent、condition reference、branch/provenance JSON 和 pair metadata；不保存 P1 style projected target。视频、cache、模型权重和第三方 checkpoint 不进入 Git。每次 milestone 完成后更新 `docs/EXPERIMENTS.md`（仅事实）以及本计划的状态、日期、commit、证据路径和下一步。

## 8. 给下一位执行者的唯一顺序

```text
PA0 完成 P-UNC / E0 人审
  → PA1 official common-prefix candidate parity + profile
  → PA2 reward/pair legality + human agreement
  → PA3 DPO algebra + 1/8/32-pair capacity
  → PA4 单卡 128-condition screening
  → [仅通过后，用户停机切双 4090]
  → PA5 pair scale-out
  → PA6 两 seed 的 DPO / AWR / SFT 对照
  → PA7 独立评估、统计与 promote/reject
```

任何箭头的前置门槛未通过时，停止在该处。项目要求保持单卡可运行；正式研究证据不必执着于只用单卡完成。

## 9. 文档关系

- `docs/AUTORESEARCH_ROUTE_DECISION.md`、`docs/AUTORESEARCH_RETROSPECTIVE_2026-07.md` 保留旧 explicit projection 路线的失败事实与边界。
- `docs/CVPR2027_PLAN.md` 保留旧 P2-V2 里程碑及其 `blocked` 证据；不应再排程其中的 endpoint/rollout 后续任务。
- 本文件是用户在 2026-07-14 授权的新、独立的后续研究计划；它不把旧路径的负结果改写成正结果。
