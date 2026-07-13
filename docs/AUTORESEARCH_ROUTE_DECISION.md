# Motion-Proj Autoresearch Route Decision

审计日期：2026-07-13

审计基线：`bbb77a4`；诊断最终代码：`72ac28c`

硬件：单张 NVIDIA RTX 4090 24 GB

审计范围：仓库代码、配置、122 条 V5 replay、已有 runs/logs、F0/F1 新诊断、2024–2026 一手文献。

## 1. Executive conclusion

### 主结论：D. 当前证据不足，先完成指定诊断实验

这不是保守的“再看看”，而是两个预注册 hard gate 同时否决了立即选择 A/B/C：

1. **F0 endpoint locality gate 失败。**固定 replay pair、固定 noise、200 updates、真实外部 `lambda_preserve = {0, 0.25, 1, 4, 16}` 下，达到 `correction <= 20% initial` 的 11 个 checkpoint 中，没有一个同时满足 outside raw-v drift / Base RMS `<= 2%` 与 frame-0 raw-v max drift `<= 1e-6`。最佳可拟合点仍有 7.52% outside drift 和 0.2031 frame-0 max drift。
2. **F1 feature-resolution gate 失败。**最细的 stride-8 SVD 层中，projector correction 的中位数仅 0.0616 feature cell，94.97% 小于 0.5 cell；其余层为 98.20%–100%。没有任何 resolution-eligible feature layer，因此按规则不得运行 F2/F3。

因此：

- 不继续当前 temporal-LoRA endpoint projection 主线；
- 不把 projected feature relation 当作已经成立的答案；
- 先修复并复核 projector target 的语义、幅度与 evaluator 因果有效性；
- 只有 target 在像素与 feature 网格上都可辨、held-out one-step 可学、并有独立 rollout evaluator 后，才重新比较机制。

### Fallback：C. short-chain / rollout-level training

仅当以下前置条件全部通过、但 one-step 改善仍无法传到 25-step rollout 时，fallback 为 **2–4 step truncated rollout + explicit track dynamics loss**。它不是现在的下一步；当前最早失败点在 multi-pair one-step capacity/locality，短链不能修复一个不可辨或语义错误的 target。

## 2. Verified repository facts

## 2.1 当前训练目标：文档计划与正式 trainer 并不相同

### 正式 `Trainer` 仍是 V1

对 projection/replay/full 分支，`motion_proj/train/trainer.py` 实际执行：

\[
\sigma_p\sim q_{\mathrm{tube}},\quad
\epsilon_p\sim\mathcal N(0,I),\quad
z_p=y+\sigma_p\epsilon_p,
\]

\[
\mathcal L_{\mathrm{proj}}
=
\frac{\sum M\odot\lVert \hat x_{0,\theta}(z_p,\sigma_p,c)-\operatorname{sg}[x^\dagger]\rVert_2^2}
{\sum M},
\]

并通过 `bound_gate(sigma, y, x_dagger, B)` 对 sample 做 0/1 gate。可选 real 分支独立采样：

\[
\sigma_r\sim q_{\mathrm{train}},\quad
\epsilon_r\sim\mathcal N(0,I),\quad
z_r=x_{\mathrm{real}}+\sigma_r\epsilon_r,
\]

\[
\mathcal L_{\mathrm{real}}
=w_{\mathrm{EDM}}(\sigma_r)
\lVert \hat x_{0,\theta}(z_r,\sigma_r,c)-x_{\mathrm{real}}\rVert_2^2.
\]

同一 projection noisy latent 上还有 Base anchor：

\[
\mathcal L_{\mathrm{anchor}}
=\lVert \hat x_{0,\theta}(z_p,\sigma_p,c)
-\operatorname{sg}[\hat x_{0,\theta_0}(z_p,\sigma_p,c)]\rVert_2^2.
\]

因此正式总损失是：

\[
\boxed{
\mathcal L_{\mathrm{formal}}
=\lambda_{\mathrm{proj}}\mathcal L_{\mathrm{proj}}
+\mathbf 1_{\mathrm{real}}\mathcal L_{\mathrm{real}}
+\beta_{\mathrm{anchor}}\mathcal L_{\mathrm{anchor}}
}
\]

`flow` 是另一独立 experiment type，使用 `L_real + lambda_flow L_flow`，不是 V2 total 的一部分。正式 Trainer 的 projection 和 real **不共享** sigma/noise，尽管 `real_loss()` API 允许外部注入共享值。

### V2 只存在于 capacity pilot

`motion_proj/losses/v2.py` 定义了 V2 loss，但正式 Trainer 未调用；`motion_proj/train/pilot.py` 的 C/D/E capacity test 才调用它。pilot 输入 clean endpoint 是 frozen Base latent `x^b`，固定：

\[
z_\sigma=x^b+\sigma\epsilon.
\]

C 使用 absolute direct-v target `v(z,x^dagger)`；D 使用 teacher-relative residual-v；E 在 D 上加 continuous trust region。D/E 的 target 为：

\[
\Delta=\operatorname{sg}[x^\dagger-x^b],\qquad
r=\operatorname{RMS}_{M}(\Delta),
\]

\[
\eta_{\mathrm{eff}}
=\min\left(\eta,\frac{B(\sigma+\varepsilon)}{r+\varepsilon}\right),
\]

\[
\boxed{
v^{\mathrm{tar}}
=\operatorname{sg}\left[
v_{\theta_0}(z_\sigma,\sigma,c)
-\eta_{\mathrm{eff}}
\frac{\sqrt{1+\sigma^2}}{\sigma}
M\odot\Delta
\right].
}
\]

pilot 的实际总损失是：

\[
\boxed{
\mathcal L_{\mathrm{pilot}}
=\mathcal L_{\mathrm{corr}}
+\mathcal L_{\mathrm{pres}}.
}
\]

它没有接入 `L_real`，即使继承配置中 `use_real_loss: true`。计划文档中的

\[
\lambda_{\mathrm{corr}}L_{\mathrm{corr}}
+\lambda_{\mathrm{pres}}L_{\mathrm{pres}}
+\lambda_{\mathrm{real}}L_{\mathrm{real}}
\]

尚不是可正式训练的实现。这是 plan/code 的实质分离，故本轮停止正式方法实现。

### mask、stop-gradient 与 frame 0

- RGB object mask 为 `[T,1,H,W]`，通过 `/8` average pooling 变成 latent mask；不是 nearest/binary resize。
- V5 cache 的 `static_mask` 全零，当前 capacity 是 object-only。
- RGB、latent、mask 的 frame 0 在 cache 输出中显式冻结：122/122 样本 RGB/latent frame-0 diff 为 0，mask frame 0 为 0。
- correction 中 `x_dagger`、`delta`、teacher 和 target 都 stop-gradient；projector、auditor、cache 路径默认 no-grad。
- preserve 作用于 `1 - dilate(max(static, object))`，即 dilation 后 mask 外；mask 内和一圈 boundary 不 preserve。
- static/object correction loss API 分开计算，但当前 replay static 为空；正式 V1 只用 union mask。
- frame 0 没有独立参数 gate。它仅因 target mask 为零而没有 correction loss；共享 LoRA 仍能改变 frame 0，F0 已证实。

### 已确认的 preserve-weight 实现语义错误

`_weighted_huber(error, mask, scalar_weight)` 在 numerator 和 denominator 同时乘 scalar，因此任何正 scalar 都抵消。旧 pilot 的 `preserve_weight` 实际只有 0 与非 0 两种语义；单 pair 下 `_masked_mse` 的 constant sigma weight 也同样抵消，故 B 与 A 等价。

运行时复核：`weight={0.25,1,4,16}` 的 normalized loss 都是 `0.81124997`，只有 0 返回 0。F0 没有沿用这个错误，而是显式构造

\[
L=L_{corr}+\lambda_{pres}L_{pres}^{normalized}.
\]

修正后仍触发 locality failure，所以该 bug 是真实实现问题，但**不是 endpoint 失败的充分解释**。

## 2.2 当前 SVD 参数化

实际 scheduler 为 Diffusers `EulerDiscreteScheduler`，`prediction_type=v_prediction`、continuous timestep、Karras sigmas；25-step 表的非零 sigma 约从 700 到 0.002。代码执行：

\[
c_{in}=\frac{1}{\sqrt{1+\sigma^2}},\qquad
c_{noise}=\tfrac14\log\sigma,
\]

\[
v_\theta=mathrm{UNet}([c_{in}z;\,x_{img}],c_{noise},c),
\]

\[
\boxed{
\hat x_0=\frac{z}{1+\sigma^2}
-v_\theta\frac{\sigma}{\sqrt{1+\sigma^2}}.
}
\]

反变换为：

\[
\boxed{
v(z,x_0)=\left(\frac{z}{1+\sigma^2}-x_0\right)
\frac{\sqrt{1+\sigma^2}}{\sigma}.
}
\]

因此 teacher-relative target 的符号与系数和 direct-v 代数一致。新增的 scheduler oracle test 对 25 个 timestep 比较代码与 Diffusers：`c_noise` 与 input scale 最大误差为 0，x0 最大误差 `4.768e-7`。target roundtrip 在 F0 为 `4.59e-6` absolute / `1.40e-6` relative。未发现重复 preconditioning 或 scaling error。

风险边界：

- `model_output_from_x0` 含 `sqrt(1+sigma^2)/sigma`，极低 sigma 会放大 absolute correction target 与梯度；
- backbone 用 `sigma_floor=1e-3`，但 target 的 trust cap 使用调用者原始 sigma；若传入 0 或 `<1e-3`，两条路径会不一致；
- 现有 pilot sigma `>=0.01`，scheduler 最低非零约 0.002，未触发该不一致；
- E 的 trust-region 把 residual-v correction 连续限制在 B 量级，因此 E 不存在 D 那样的无界低 sigma target。

另有 rollout-transfer conditioning 差异：代码传 `fps=7`，官方 SVD pipeline 会传 `fps-1`；added-time-id 声称 `noise_aug_strength=0.02`，但 condition frame 未做对应 noise augmentation；生成用 CFG 1→3，而 one-step pilot 只跑 conditional raw output。这些不影响 raw-v 代数与 F0 locality 结论，但会混淆未来 one-step→rollout 归因，必须先做 parity test。

## 2.3 LoRA 与可训练参数

运行时 model manifest 而非配置猜测给出：

- selected modules：128；adapter/trainable tensors：256；trainable parameters：3,319,808；
- down / mid / up 分别 48 / 8 / 72 个 module；spatial module 为 0；
- 每个命中路径为
  `*.temporal_transformer_blocks.0.{attn1,attn2}.{to_q,to_k,to_v,to_out.0}`；
- 完整 128 行实际列表固化于 F0/F1 run 的 `selected_modules.txt`。

实际块集合是：

- `down_blocks.{0,1,2}.attentions.{0,1}`；
- `mid_block.attentions.0`；
- `up_blocks.{1,2,3}.attentions.{0,1,2}`；
- 上述每块的 attn1/attn2 × q/k/v/out 共 8 个 projection。

SVD temporal transformer 把 `[B*T,C,H,W]` 重排为 `[B*H*W,T,C]`，所以 attention 只在**同一空间索引**跨时间通信；但同一 LoRA 权重在所有空间位置和所有 frame 共享。它不具备 mask-conditioned spatial gate，也不具备 `t=0` 参数隔离，因此 frame 0 和 mask 外都可能被更新。模型包含 temporal/spatiotemporal convolution 等模块，但它们均被冻结；没有其他可训练 module。

## 2.4 Replay 数据语义

对 V5 object-only cache
`/root/autodl-tmp/cache/p2-v2/p2-v2-replay05-candidate64x2-s20260713-a41dfa4-objectonly`
的 122/122 kept samples 做 tensor + metadata audit：

- `source=replay_v2`、`parent_kind=base`、adapter count 0、25-step Base；不是 t10 或其他 adapter；
- future ego tensor 与 future track tensor 均为 0；provider diagnostics `uses_future_gt=false`；未发现训练 replay 的 future-GT leakage；
- `background`、`dynamic_residual`、`foreground_candidate` 是 query strata，但全部被转换为类别 `generated_point/<stratum>` 的固定 16×16 point boxes，并统一经过 object projector；
- fallback query sampling 可能使 strata 之间重复点；它们不是 object instance，当前所谓 “object component” 实际是 **point-track tube component**；
- all 122 object energy before→after 改善，object mask 非空，static mask 全空；latent object coverage mean/median 为 6.692%/6.634%（文档的 2.79% 是 RGB coverage 口径）。

完整 target 路径：

\[
X^b
\to\mathrm{RAFT\ flow}
\to\mathrm{affine\ background}
\to\mathrm{stratified\ point\ tracks}
\to\mathrm{3-frame\ median}
\to\mathrm{2nd-difference\ smoothing}
\to\mathrm{16\times16\ crop/resize/paste}
\to X^\dagger
\to E_{VAE}(X^\dagger).
\]

RGB compositor 复制源 patch 到目标位置而不清除源位置，无 occlusion/depth reasoning；包含 bilinear crop/resize、边界 clipping 与 feathering，因此会产生 duplication/ghosting/edge effects。VAE 是非局部的：在 122 samples 上，仅 RGB 局部修改导致 mask 外 latent RMS 均值 0.04918（mask 内 0.26717），平均 52.95% mask 外 latent 元素变化大于 `1e-6`。随后 target 又用 `M * (E(X^dagger)-E(Xb))` 截断，得到的其实是 hybrid latent，不一定对应任何可解码 RGB 视频。

## 2.5 Projector 是否把“合理”退化成“少运动”

当前 smoothing energy 只有 data fidelity 与二阶差分，没有 net displacement、mean velocity、direction、turn direction、visibility 或 support 的显式约束。F1 对 8 clips、351 tracks 的复核是：

| Statistic | Result | Interpretation |
|---|---:|---|
| correction px median / p90 | 0.519 / 2.709 | target 大多非常小 |
| net-displacement ratio median | 0.990 | 通常没有把净位移压成 0 |
| direction cosine median | 0.999 | 通常保持方向 |
| turn-direction preserved | 82.62% | 仍有约 17% turn sign 不保持 |
| acceleration RMS ratio median | 0.136 | 中位轨迹去掉约 86% acceleration |
| jerk RMS ratio median | 0.056 | 中位轨迹去掉约 94% jerk |
| visibility expansion | 5.52% | projector/render visibility 语义不完全一致 |
| frame-0 correction median / max | 0.499 / 10.165 px | track target 内部改了 p0，最终 RGB/latent 又靠 mask 强制冻回 p0 |

分 strata 的 net-displacement median 分别为 background 0.975、dynamic_residual 1.000、foreground_candidate 0.995；所以“完全静止塌缩”未被观察到。但它几乎消除高阶动力学、修正幅度又通常亚像素，不能自动等价为“更真实的驾驶动力学”。下一版投影能量至少应为：

\[
\mathcal E(\tau)=
\lambda_d\sum_t w_t\lVert p_t-p_t^b\rVert_1
+\lambda_a\sum_t\lVert\Delta^2p_t\rVert_1
+\lambda_j\sum_t\lVert\Delta^3p_t\rVert_1,
\]

并在优化中硬/软约束：

\[
\lVert p_T-p_0\rVert\approx\lVert p_T^b-p_0^b\rVert,
\quad
\frac1{T-1}\sum_t\Delta p_t
\approx
\frac1{T-1}\sum_t\Delta p_t^b,
\]

以及 direction、turn sign、original visibility、support 和 `p_0=p_0^b`。`background` 应作为 preservation/negative relation，`dynamic_residual` 作为主要 dynamics supervision，`foreground_candidate` 作为高优先级 supervision；不得再统一称为 object。

## 3. 当前失败的准确位置

| Failure class | Verdict | Evidence |
|---|---|---|
| target construction failure | **未排除，已有风险证据** | RGB copy/paste 无 occlusion、VAE 非局部 hybrid latent、所有 strata 同投影、projector 高阶抑制极强且修正多为亚像素 |
| optimization failure | **已证实** | 固定 4-train-pair capacity：C/D/E 200 update 的 target reduction 仅 0.95%/1.28%/23.45%，均未到 80% |
| parameterization failure | **当前排除** | scheduler oracle、raw-v↔x0 roundtrip、25-step algebra tests 均通过；无 double preconditioning |
| locality failure | **已证实** | F0 corrected-weight Pareto 中 0/11 feasible checkpoints；shared temporal LoRA 改变 mask 外与 frame 0 |
| rollout transfer failure | **V2 未测试，不能声称** | V2 capacity gate 失败后按规则未跑 rollout；V1 16×100/4×300 的 rollout 失败不能直接归因给 V2 transfer |
| metric failure | **legacy eval 已证实无效；新独立 eval 缺失** | legacy `generate_eval.py` 复用 source future-ego metadata，object boxes 为空；CoTracker3 存在但未接正式独立 evaluator |
| data distribution failure | **未证实 GT 泄漏；存在覆盖/语义风险** | 122 replay 无 future GT，但都是 Base point tubes、static 为空、strata 混用、coverage 小、只来自固定 Base distribution |

按问题逐项回答：

1. **单步 target 是否可拟合？** 单 pair 可以。F0 中 `lambda=0.25/1` 的 correction 分别降至初值 3.21%/4.79%；历史单-pair E 也可到约 4.8%。
2. **held-out one-step 是否改善？** 没有可信结果。capacity pilot 虽选择 4 held-out pairs，但实现只聚合 train batches；不能把“选了 held-out id”当验证。
3. **mask 外是否漂移？** 是。F0 在 correction 达标时 outside raw-v drift/Base RMS 至少 6%–17% 量级，远高于 2% gate。
4. **frame 0 是否漂移？** 是。F0 最优 correction 点 frame-0 raw-v max drift 0.2031–0.4629；mask frame0=0 不能冻结共享参数行为。
5. **完整 25-step rollout 是否改善？** V2 未运行；不得回答“是”。V1 的 16 个 100-step trial dynamics 全差于 Base，4 个 300-step continuation 继续恶化。
6. **独立 evaluator 是否改善？** 未运行；不得用 training auditor 自评。
7. **哪一步先失败？** single pair 可记忆；首先在 fixed multi-pair one-step capacity 与 locality 失败。one-step→rollout transfer 尚未获得被隔离的实验机会。

历史 V1 仅作为外部一致性证据：Base `static=8.2095, accel=4.3953, LPIPS=0.5088`；代表性 t10-100 为 `9.861, 5.714, 0.4453`，t10-300 为 `12.2478, 6.5092, 0.4502`。即视觉相似性改善与 dynamics 恶化稳定共存，但不能用这组结果证明 V2 residual-v 的 rollout transfer failure。

## 4. F0 / F1 最重要的诊断证据

### F0：Endpoint failure decomposition

预注册阈值在运行前写入 `configs/diagnostics/autoresearch_f0_endpoint.yaml`：correction fraction `<=0.20`、outside raw-v drift ratio `<=0.02`、frame-0 raw-v max drift `<=1e-6`。固定 sample 34、sigma 0.05、noise seed 20260714、LR `2e-4`、每个 lambda 最多 200 updates。

| lambda preserve | final correction fraction | final outside drift ratio | final frame-0 max drift | Gate |
|---:|---:|---:|---:|---|
| 0 | 0.0361 | 0.1758 | 0.4629 | fail |
| 0.25 | 0.0321 | 0.1050 | 0.3066 | fail |
| 1 | 0.0479 | 0.0845 | 0.2324 | fail |
| 4 | 0.1478 | 0.1055 | 0.2656 | fail |
| 16 | 0.2763 | 0.0781 | 0.1953 | correction fail |

`lambda=4` 的 best-correction checkpoint 在 step 150：fraction 0.1161、outside 0.0752、frame0 0.2031。梯度均 finite/nonzero；不同阶段 correction/preserve cosine 会从轻微负冲突变为正，说明不是简单“两个 loss 总在反向”。问题是共享参数没有局部可行子空间，而不是继续提高 preserve weight 就能解决。

Artifacts：`/root/autodl-tmp/runs/autoresearch-f0-endpoint-s20260713-6845411/`。

### F1：Feature discriminability audit

冻结 SVD，无训练；8 fixed cache pairs × 3 sigmas，7 个真实中间 hook，5847 valid point-time observations/layer。

| Layer | Grid | Stride | median correction (cell) | p90 (cell) | fraction < 0.5 cell | observed argmax PCK@1cell |
|---|---:|---:|---:|---:|---:|---:|
| down_s8 | 32×56 | 8 | 0.0616 | 0.3063 | 94.97% | 0.3198 |
| down_s16 | 16×28 | 16 | 0.0308 | 0.1531 | 98.20% | 0.5449 |
| down_s32 | 8×14 | 32 | 0.0154 | 0.0766 | 99.90% | 0.6899 |
| mid_s64 | 4×7 | 64 | 0.0077 | 0.0383 | 100% | 0.9786 |
| up_s32 | 8×14 | 32 | 0.0154 | 0.0766 | 99.90% | 0.6566 |
| up_s16 | 16×28 | 16 | 0.0308 | 0.1531 | 98.20% | 0.4621 |
| up_s8 | 32×56 | 8 | 0.0616 | 0.3063 | 94.97% | 0.1184 |

`mid_s64` 的高 PCK 是 64-pixel cell threshold 过宽造成的 tracking diagnostic，不能被解释为推荐层。observed 与 projected PCK/heatmap 几乎相同，说明 target difference 小于 grid 分辨率。决策为 `feature_resolution_failure`，`recommended_layer=null`。

Artifacts：`/root/autodl-tmp/runs/autoresearch-f1-features-s20260713-72ac28c/`。两个失败尝试也保留 manifest/log，分别定位 deterministic CUDA median 与 cuBLAS env 初始化顺序；成功 run 在修复后完成。

## 5. Route A–E 固定权重评分

评分在本次决策中使用用户指定固定权重，1=差、5=强。加权分不是自动晋级器；任何 hard gate 失败都覆盖总分。机制晋级还要求总分 `>=3.5`、无 hard failure、held-out 与 independent rollout 均有正证据。当前没有路线满足。

| Route | 解释当前失败 20% | 复用 15% | 单卡可行 15% | 创新区分 20% | rollout 概率 20% | 可靠评估 10% | Weighted / 5 | Evidence verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A endpoint projection | 2 | 5 | 5 | 3 | 1 | 4 | **3.10** | 复用/成本最高，但 F0 locality hard-fail，V1 rollout 全负；需 spatial gate/branch 后已不再是当前方法 |
| B projected track relation | 4 | 4 | 3 | 2 | 2 | 4 | **3.05** | 能针对 locality，但 F1 resolution hard-fail，且 Track4Gen/SARA/PhysAlign 碰撞高 |
| C generic motion feature | 2 | 2 | 2 | 1 | 2 | 3 | **1.90** | MoAlign/VideoREPA/Geometry Forcing/PhysAlign 已覆盖；会丢失 explicit projector |
| D short-chain rollout | 2 | 3 | 2 | 2 | 4 | 4 | **2.75** | 若唯一剩余问题是 transfer 会有价值；当前失败更早，且 SIFT/ShortFT 已占邻域、24 GB 尚未验证 |
| E reward/preference | 2 | 2 | 1 | 1 | 3 | 2 | **1.85** | SHIFT/DenseDPO/VideoGPA/Flash-GRPO 碰撞且 rollout/reward hacking 成本不符约束 |

### Route A 判定

尚未排除的实现问题包括 preserve scalar cancellation、formal V2 未接 Trainer、conditioning parity、legacy evaluator。但 F0 已在不依赖 preserve bug 的 objective 上失败，raw-v algebra 也通过，所以不能再用“可能还有 bug”无限延长 endpoint 路线。纯调 loss 权重无法产生 frame/spatially-local parameter subspace；若必须增加 spatial gate 或独立 residual branch，方法已转成新架构，还需面对 Track4Gen。

### Route B 判定

候选层只能来自实际 F1，而不是照搬 Track4Gen。当前没有推荐层：stride-8 仍有 95% correction 小于 0.5 cell。teacher-relative relation、acceleration/jerk relation、refiner 或 zero-conv 都不能恢复 target 本身缺失的空间可辨性。继续实现 F2/F3违反预注册 stop condition。

### Route C 判定

optical-flow encoder、VideoMAE/V-JEPA 或 geometry foundation feature alignment 都适合作 baseline；若作为主方法，会分别退化为 MoAlign/VideoREPA 或 Geometry Forcing/PhysAlign 的邻近复现，并抹去 Motion-Proj 的 explicit dynamics projector。

### Route D 判定

2–4 denoising steps 能改善 credit assignment，理论上最直接接触 rollout distribution；但当前 C/D/E 在 multi-pair single-step 就学不动，F0 也无局部解，短链只会放大显存和梯度问题。ShortFT 依赖 trajectory-preserving shortcut model，不能拿来证明朴素截断；SIFT 已用 3-step self-imagination motion supervision。它只保留为 conditional fallback。

### Route E 判定

标量 motion reward 会丢掉 projector 的可解释结构，且存在“减小 motion 就改善 acceleration/jerk”的 reward hacking。AWR/DPO/GRPO 都需要大量生成候选或 rollout。与 SHIFT、DenseDPO、VideoGPA、Flash-GRPO 的区分度和单卡成本均不够。

## 6. 最近邻论文撞车分析

完整矩阵见 `docs/AUTORESEARCH_LITERATURE_MATRIX.md`。四个 AC-level 结论：

1. **Track4Gen 风险高。**候选 F3 的 identity refiner + zero-conv + temporal training 基本是 Track4Gen 架构；“换成 projected tracks”只有在 B5>B4>B3>B0 的完整 rollout 因果链成立时才是贡献。
2. **Generic relation alignment 已拥挤。**VideoREPA→MoAlign→SARA 已覆盖 generic TRD、motion bottleneck、pair routing；PhysAlign 又加入 V-JEPA2 全时空 Gram 与 explicit 3D geometry。
3. **Geometry teacher 不是空白。**Geometry Forcing 已把 frozen VGGT feature 蒸馏到生成器，PhysAlign 进一步覆盖 geometry+kinematics relation。
4. **Reward/short-chain 也不是空白。**SHIFT、DenseDPO、VideoGPA、Flash-GRPO 覆盖多种 post-training；SIFT 用 3-step self-imagination + motion classifier，ShortFT 用 shortcut chain 完整反传。

Motion-Proj 尚可保留的唯一清晰边界是：公开 driving data、无 future GT、无新增 motion condition、frozen Base rollout、自生成 point-track dynamics projection、无完整 chain backprop、单卡、并以独立 tracker 在完整 rollout 上证明高阶动力学改善而不是少运动。

## 7. 最终主路线与 fallback

### 主路线：Projector Resolution & Causal Transfer Gate（诊断路线，选择 D）

本阶段不新增生成器训练机制。晋级逻辑定义为：

\[
\mathrm{Promote}(r)=
I_{\mathrm{noGT}}
I_{\mathrm{projector\ validity}}
I_{\mathrm{target\ resolvable}}
I_{\mathrm{locality}}
I_{\mathrm{heldout\ one-step}}
I_{\mathrm{independent\ rollout}}.
\]

任一 indicator 为 0 即不选择 A/B/C。当前状态为：noGT=1；projector validity=0/unknown；target resolvable=0；endpoint locality=0；held-out=unknown；independent rollout=unknown。

可训练模块：无。

stop-gradient：Base、auditor、projector、VAE、feature hooks、independent tracker 全部 no-grad。

方法贡献：不是论文方法，而是防止在错误 target 上投入 5 个月的 decision gate。

### Fallback：2–4 step Projected-Dynamics Truncated Rollout（选择 C）

仅当 target/resolution/locality/held-out 均通过、但 25-step transfer 失败时执行：冻结 Base teacher，LoRA 或独立 future-only residual branch 可训练，展开 2→4 steps、gradient checkpointing、共享 sigma/noise 做 Base prior anchor，在独立 decoded rollout tracks 上计算 position/acceleration/jerk 与 preservation。先做单 batch memory/throughput smoke；若 24 GB OOM 或每 update 超过可接受预算，直接停止。

## 8. 明确停止做什么

- 停止用当前 temporal-only LoRA 继续 endpoint LR/preserve-weight sweep；
- 停止把更大的 preserve weight 当 locality 修复；
- 停止 F2/F3，直到 revised target 通过 `>=0.5 feature cell` 的预注册分辨率门槛；
- 停止 300/800-step、Optuna、正式 cache 扩容和多个方向并行实现；
- 停止把 `background`/`dynamic_residual`/`foreground_candidate` 统称 object；
- 停止用 training auditor 或 legacy evaluator 宣称 rollout dynamics 改善；
- 停止 generic flow/VFM/geometry alignment 作为主贡献；它们只保留为 baseline；
- 停止 reward/DPO/GRPO 主线；
- 不删除 endpoint 代码，不重构正式 Trainer，不 push。

## 9. 未来两周最小执行顺序

### Week 1：先修 target 与 evaluator，不训练生成器

1. **P0 projector validity v2（CPU/只读）。**按 strata 分开，硬设 `p0=p0b`，加入 displacement/mean-velocity/direction/turn/visibility/support 约束；报告每条 track 的 correction、accel/jerk reduction 与 static-collapse flags。先在现有 8 clips 重算，不写正式 cache。
2. **P1 target render/encode locality。**比较 RGB copy/paste、target mask、完整 `E(Xdagger)` 与 masked hybrid latent；量化 source duplication、occlusion collision、边界、mask 外 VAE energy。若无法把 target 解释为合法视频，停止核心方向。
3. **E0 independent evaluator validity。**把 CoTracker3 只作为 evaluator，固定 Base 8–16 clips 做 seed repeatability、track survival、identity、acceleration/jerk；删除/隔离 future source metadata，验证 evaluator 不读训练 auditor outputs。
4. **C0 SVD conditioning parity。**用单 clip 对齐 official pipeline 的 fps id、noise augmentation 与 CFG，确认 Base 25-step baseline 可复现；不训练。

### Week 2：只按 gate 逐级执行

5. **重跑 F1。**仅 revised projector 通过 P0/P1 后；若最细合适 feature 仍有 >80% correction `<0.5 cell`，永久停止 direct feature-cell relation 路线。
6. **O1 held-out one-step。**8–24 pairs、fixed noise bank、最多 200 updates；真正分别报告 train/held-out，不许只评 train。
7. **条件 F2。**只有 F1 通过才跑 frozen-SVD feature head capacity，4–8 pairs；不 feedback UNet。
8. **条件 F3。**只有 F2 与 O1 都通过才跑 future-only refiner/zero-conv，最多 16 train/8 val、50–100 updates，并以 independent 25-step rollout 晋级。

详细预注册见 `docs/AUTORESEARCH_EXPERIMENT_PLAN.md`。

## 10. Reviewer 2 最可能的 5 个攻击点

1. **“Projector 只是把轨迹抹平。”**没有基于场景/车辆状态的动力学定律；acceleration/jerk 下降可以由少运动获得。必须报告 displacement、velocity、direction、dynamic degree 与 survival 的联合 Pareto。
2. **“Projected feature method 只是 Track4Gen + smoother。”**如果没有 observed vs generic smooth vs dynamics projection 的 B3/B4/B5 完整 rollout ordering，创新 claim 不成立。
3. **“评价与训练同源、存在 circularity。”**同一 RAFT/generated-track auditor 既造 target 又打分会放大自身偏差；必须使用冻结的独立 tracker/evaluator，并人工 spot-check occlusion/identity。
4. **“没有证明 one-step target 能改变完整 generative dynamics。”**capacity/self-fit 不是生成贡献；必须有 25-step、多个 seed/clip、Base-matched independent metrics 和视觉质量联合置信区间。
5. **“局部 RGB paste 不是合法 counterfactual。”**复制、遮挡、VAE 非局部和 hybrid latent 可能制造不可实现 target；必须证明 target consistency，或转为不依赖 RGB rendering 的显式轨迹机制。

## 11. 审计可复现性

新增诊断：

- `motion_proj/diagnostics/endpoint_locality.py`
- `motion_proj/diagnostics/feature_discriminability.py`
- `configs/diagnostics/autoresearch_f0_endpoint.yaml`
- `configs/diagnostics/autoresearch_f1_features.yaml`
- `tests/test_endpoint_locality.py`
- `tests/test_feature_discriminability.py`
- `tests/test_svd_parameterization.py` 的 Diffusers scheduler oracle

成功 runs：

- `/root/autodl-tmp/runs/autoresearch-f0-endpoint-s20260713-6845411`
- `/root/autodl-tmp/runs/autoresearch-f1-features-s20260713-72ac28c`

F1 failed attempts（保留用于 provenance，不作为数据）：

- `/root/autodl-tmp/runs/autoresearch-f1-features-s20260713-9200467`
- `/root/autodl-tmp/runs/autoresearch-f1-features-s20260713-171ec2a`

测试：targeted baseline 43 passed；F0 后 full suite 130 passed；F1 后 full suite 133 passed。三个诊断图已人工查看，数据与 summary 一致。
