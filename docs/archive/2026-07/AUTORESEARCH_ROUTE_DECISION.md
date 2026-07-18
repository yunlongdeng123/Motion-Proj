# Motion-Proj Autoresearch Route Decision

审计日期：2026-07-14

审计基线：`bbb77a4`；Phase 2 E0 最终诊断代码：`016f752`（文档提交前）。

硬件：单张 NVIDIA RTX 4090 24 GB

审计范围：仓库代码、配置、122 条 V5 replay、已有 runs/logs、F0/F1 新诊断、2024–2026 一手文献。

## 1. Executive conclusion

### 主结论：E. 停止当前 explicit dynamics projection 核心方向

Phase 2 已完成 C0、P0、P1、E0 的可执行门禁。C0 通过，P0 仅机器合格且仍待人工复核；但 P1 已在
7 个可构造 target 上机器失败：整数 crop/paste 可将连续 correction 量化成零 RGB 改变，hybrid/full
VAE 的最大 LPIPS 为 0.06805（高于 0.05），并出现 source-retention 与无 depth-order overlap。E0 已用
官方 CoTracker3 完成机器稳定性 gate，但 12 个 overlay 仍待人工复核，且没有任何训练方案可供 rollout
比较。P1 的 target legality hard failure 已足以否决当前 endpoint/feature/short-chain 的共同监督源；因此
不能把 E0 的 evaluator-only 机器通过写成生成质量或 rollout-performance improvement。

因此停止当前的“Base rollout → RAFT-chain projector → RGB crop/resize/paste → hybrid latent/feature
distillation”核心机制；不以扩大 mask、继续扫 loss、更多数据、更多训练步数或大型外部视频编辑模型掩盖
target 定义失败。

### Fallback：D. 继续诊断，尚不足以选择新机制

该 fallback 不是恢复训练。只有现有官方独立 evaluator 的人工一致性完成、并以不依赖大型视频编辑模型的
新 counterfactual construction 在相同 legality gate 上通过后，才允许建立新的只读诊断计划；届时它是
新的研究问题，而非对当前 explicit projection 路线的补丁。

### 1.1 结论边界修订（2026-07-14；A1 corrigendum）

本节不修改 F0/F1 的原始 run、数值、阈值或失败结论；它只收紧这些证据可以支持的外推范围。此前版本中将两个诊断结果表述为 endpoint 或连续 feature relation 的普遍性失败，超过了其固定输入、固定参数化和当前 projector 的证据边界。

**F0 实际证明：**在固定 replay pair、固定 `sigma=0.05`、固定 noise、当前 teacher-relative residual-`v` target、共享 temporal LoRA 与当前 preserve 定义下，即使修正正 preserve 权重的归一化语义，仍未找到同时满足预注册 correction、mask 外 locality 与 frame-0 locality 门槛的点。因此禁止继续扫描 preserve weight、提高学习率、延长该单 pair shared-LoRA endpoint pilot，或把旧 preserve bug 解释为唯一失败原因。

**F0 未证明：**所有 endpoint projection 都失败；所有 sigma、pair 或 mask policy 都无可行点；具有 hard-gated residual branch 或独立局部参数子空间的机制必然失败；raw-`v` one-step gate 与 decoded RGB 或完整 rollout locality 完全等价。

**F1 实际证明：**当前 projector 的 correction 很小；在当前 8 对 replay 与 frozen raw SVD feature probe 中，绝大多数 correction 小于现有 feature stride 的半个 cell，且该 raw feature 没有提供明显的现成 projected relation signal。这是高风险信号，足以阻止在旧 target 上直接启动 F2/F3。

**F1 未证明：**continuous feature target 必然不可学习；bilinear sampling、Gaussian relation target、soft-argmax、correlation distribution 或 sub-cell interpolation 必然没有信号。`<0.5 feature cell` 在后续 F1-R 中只保留为描述性统计，不能单独作为 hard fail。

本轮新的严格依赖图为：

```text
A1 conclusion corrigendum
        │
        ├── C0 official-SVD conditioning parity
        ├── P0 projector physical validity ──pass──> P1 RGB/VAE target validity
        └── E0 independent rollout evaluator validity

C0 + P0 + P1 + E0 all pass (including required human gates)
        │
        └── F1-R revised feature signal audit (read-only)
```

在 C0/P0/P1/E0 未全部通过前，不训练 feature head、zero-conv/refiner 或 short-chain，不构建新大规模 cache，也不运行 Optuna、300/800-step 或正式 Trainer V2 集成。所有新 gate 必须保存独立 config、manifest、metrics、summary 和终态文件；需要人工判断时只能标记为 `awaiting_reviews`，不得自行晋级。

### 1.2 C0 官方 SVD parity 更新（2026-07-14）

正式 run `autoresearch-c0-conditioning-s20260714-v2` 在 clean commit `b36e042`、冻结
SVD-XT、固定 train condition index 0、25 inference steps、seed `2026071401` 下完成。官方
Diffusers 0.31.0 pipeline、实际 `SVDBackbone.generate()` 与版本化 `svd_official_v1` 候选的
added-time IDs、condition-noise、initial latent、每步 scaled input、unconditional/conditional raw
output、CFG output、scheduler output、final latent 与 decoded RGB 均一致；逐项最大差为 0，重复运行也
逐位一致。证据：`/root/autodl-tmp/runs/autoresearch-c0-conditioning-s20260714-v2/`。

但 C0 同时确认旧单步 `build_conditioning()` 与 rollout conditional branch 不等价：official 的
fps time ID 为 6，legacy 为 7，且 condition noise、image embedding 与 image latent 都不同。因此
`svd_official_v1` 可作为未来显式协议通过 C0；既有 V5 Base rollout 的生成 provenance 不被否定，
但其 stored legacy context 不得用于新的 one-step-to-rollout transfer claim，也不得被静默改写。

### 1.3 P0 point-track tube 物理有效性更新（2026-07-14；machine pass，awaiting reviews）

正式 run `autoresearch-p0-projector-s20260714-v1` 在 clean commit `dfef913` 上只读重建同一
8 个 frozen-Base replay 的 RAFT-chain 点轨迹（351 tracks），没有生成新视频、加载 adapter、使用
future GT 或改写 cache。它比较 P-ID、当前 P-CUR、受约束 P-CON 和 uncertainty-gated P-UNC；
`background` 仅作 preservation/negative relation，所有正修正仅限 `dynamic_residual` 与
`foreground_candidate` 的 point-track tube component。

P-UNC 是唯一 machine-eligible 候选：frame-0 最大修正为 0，visibility/time-index/support
violation 均为 0，net-displacement median/p10 均为 1，direction median 为 1，turn preservation
为 95.40%，dynamic-degree median ratio 为 0.862；290 个实际 primary correction 均高于预注册
tracker-uncertainty SNR 门槛。合成集上它保留 clean motion，5/5 high-SNR single-frame outlier
改善，并且不放大 sub-uncertainty jitter。P-CON 保留硬不变量，但 turn preservation 88.79%、
dynamic-degree median ratio 0.736，不能晋级；P-CUR 更明确失败（frame-0 max 10.165 px、127 个
visibility expansion、turn 83.05%、dynamic-degree median 0.112）。

因此 P1 可按 P-UNC 的 machine eligibility 启动，但 P0 仍为 `awaiting_reviews`：run 已导出
12 个分 strata panel 与 `reviews.template.jsonl`，在完成至少 12 个有效人工 verdict 前，禁止
F1-R 和最终路线晋级。证据：`/root/autodl-tmp/runs/autoresearch-p0-projector-s20260714-v1/`。

### 1.4 P1 RGB/VAE counterfactual target 更新（2026-07-14；machine fail）

P1 以 P0 machine-eligible P-UNC 为唯一输入，在 7 个具有 primary point-track tube component 的
冻结 Base clips 上临时构造 RGB target；P0 index 114 仅含 background preservation、没有正向
counterfactual target，故不被伪装成 P1 failure sample。诊断比较 full latent、当前 masked hybrid 和
固定一格 dilated hybrid，未加载 adapter、未使用 future GT、未写 cache。P1 v1 保留了一个 scope bug：
occlusion proxy 把未移动的密集 query overlap 一并计数；v2 在 clean commit `960c4c2` 上只审计至少
一端发生实际 integer crop/paste move 的关系，仍得到同一机器失败结论。

硬不变量本身通过：所有 7/7 constructed RGB/latent frame-0 error `<=1e-6`，hybrid mask 外 latent
RMS/Base RMS 最大 0.00871。但 endpoint target 无法通过合法性 gate：(1) index 34 的 P-UNC correction
在整数 crop/paste 后完全没有 RGB 改变；(2) decode(hybrid) 对 projected RGB 的 LPIPS 最大 0.06805，
高于 0.05，且 full-VAE reconstruction 同样为 0.06805，表明限制已在 RGB/VAE target 而非仅 hybrid
mask；(3) 出现 1 个 source-retention duplication proxy；(4) 即使只计实际移动组件，仍有 588 个
无 depth/occlusion ordering 的 move-overlap relationship。当前 compositor 不能将 P0 的有效连续轨迹
稳定映射为合法、可解码的 RGB/latent counterfactual。

因此 endpoint route A 和 F1-R 均被 P1 阻断；不得靠扩大 mask、忽略 source 或引入大型视频编辑模型
掩盖该结论。P1 的 panel/review template 保留为诊断证据，但 machine gate 已失败，不等待人审来反转它。
证据：`/root/autodl-tmp/runs/autoresearch-p1-target-s20260714-v2/`（v1 也保留）。

### 1.5 E0 独立 rollout evaluator 更新（2026-07-14；machine pass，awaiting reviews）

E0 使用官方 CoTracker3 offline provider：query 固定为 evaluator 自己的 first-frame grid；输入只允许 frozen
`base_rgb` 与 evaluator weights；明确禁止读取 cache generated tracks、P0/P1 output、future GT 或
source-future metadata。实现采用 evaluator-only robust affine background motion，所有速度、加速度与 jerk
只命名为 `camera-compensated image-plane` 指标；无有效轨迹会返回 `invalid`，不会写成 0。

历史 v1 `autoresearch-e0-evaluator-s20260714-v1` 保留 checkpoint 缺失的 blocked evidence。用户上传官方
`scaled_offline.pth` 后，v2 固定实际 SHA256
`2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`，但审计发现原
survival-threshold self-correlation 可复用基线值而伪造 1.0，故 v2 只保留为诊断 scope-bug evidence。clean
commit `016f752` 将协议升为 `autoresearch-e0-independent-cotracker3-v2`：保存每种扰动的 aggregate，并以
Base 对 photometric/codec/resize 三种扰动的跨 clip Spearman（四项 aggregate metric）执行原有 `>=0.80`
门槛；缺失值 fail-closed。

正式 v3 `autoresearch-e0-evaluator-s20260714-v3` 在 clean commit `016f752`、同一 8 个 frozen Base
index、cache fingerprint `e2e3a3b35f6d…` 与 seed `20260713` 下完成。repository commit
`82e02e8029753ad4ef13cf06be7f4fc5facdda4d`、权重 SHA256 的实际值和预期值精确一致；`uses_future_gt=false`、
`fallback_used=false`。8/8 clip 有有效 track，identical rerun 的坐标 max error、visibility mismatch 与 aggregate
relative delta 均为 0；synthetic acceleration/jerk ordering、occlusion down-weighting 与 visibility-threshold
rank 均通过。跨扰动 rank 为 photometric/codec 全部 1.0，resize 的 acceleration 为 0.97619、其余为 1.0，
因此 machine gate 通过。resize 的绝对 aggregate relative delta 中位数为 10.03%、最大 31.84%，故它只支持
**排序稳定**，不把该 evaluator 的绝对 jerk 尺度写成经过物理校准的量。

v3 已生成并成功解码 8 个真实 + 4 个 synthetic、每个 8 帧的 overlay；`reviews.template.jsonl` 仍是 0/12
human verdict，最终状态严格为 `awaiting_reviews`，不能自行升格为 E0 pass，也没有运行任何生成模型的
comparative rollout evaluation。证据：v1
`/root/autodl-tmp/runs/autoresearch-e0-evaluator-s20260714-v1/`、v2
`/root/autodl-tmp/runs/autoresearch-e0-evaluator-s20260714-v2/`、v3
`/root/autodl-tmp/runs/autoresearch-e0-evaluator-s20260714-v3/`。

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
| metric failure | **legacy eval 已证实无效；独立 evaluator 机器有效但人审待完成** | legacy `generate_eval.py` 复用 source future-ego metadata，object boxes 为空；E0 v3 CoTracker3 的 provenance、repeatability、扰动排序和 synthetic gate 通过，但 12-panel human alignment 仍为 0/12 |
| data distribution failure | **未证实 GT 泄漏；存在覆盖/语义风险** | 122 replay 无 future GT，但都是 Base point tubes、static 为空、strata 混用、coverage 小、只来自固定 Base distribution |

按问题逐项回答：

1. **单步 target 是否可拟合？** 单 pair 可以。F0 中 `lambda=0.25/1` 的 correction 分别降至初值 3.21%/4.79%；历史单-pair E 也可到约 4.8%。
2. **held-out one-step 是否改善？** 没有可信结果。capacity pilot 虽选择 4 held-out pairs，但实现只聚合 train batches；不能把“选了 held-out id”当验证。
3. **mask 外是否漂移？** 是。F0 在 correction 达标时 outside raw-v drift/Base RMS 至少 6%–17% 量级，远高于 2% gate。
4. **frame 0 是否漂移？** 是。F0 最优 correction 点 frame-0 raw-v max drift 0.2031–0.4629；mask frame0=0 不能冻结共享参数行为。
5. **完整 25-step rollout 是否改善？** V2 未运行；不得回答“是”。V1 的 16 个 100-step trial dynamics 全差于 Base，4 个 300-step continuation 继续恶化。
6. **独立 evaluator 是否改善？** 没有训练模型的 comparative rollout；E0 v3 只验证 Base 上的 evaluator 机器稳定性，仍不得用 training auditor 自评或宣称改善。
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

## 5. Phase 2 Route A–E 最终比较

任何 hard gate 都覆盖复用性或成本上的优势。下表使用 Phase 2 定义的路线标签；它取代本文件早期将
short-chain/reward 单列的预诊断评分。

| Route | Evidence for | Evidence against | Decision |
|---|---|---|---|
| A. Endpoint projection distillation | C0 已证明 `svd_official_v1` rollout 可逐项复现；P0 P-UNC 机器不变量合格。 | F0 locality hard fail；P1 的 RGB/VAE target legality hard fail。 | rejected |
| B. Projected track / feature relation distillation | P0 有 290 个高 SNR 的 primary point corrections；F1 不能仅凭 sub-cell 数值永久否定连续 relation。 | B 与 A 共享 P1 的非法 target；P0/E0 人审未完成；F1-R 的 P1 前置不成立，且 Track4Gen 邻域拥挤。 | rejected |
| C. Short-chain / truncated rollout training | 仅在 one-step target 合法、局部性可控而 25-step transfer 失败时才有因果意义。 | 当前失败发生在 target 合法性之前；E0 仅完成 evaluator 机器有效性，short chain 不能把不可观察的 counterfactual 变合法。 | rejected |
| D. Continue diagnostics | 仍可独立验证新 evaluator 或全新的 target construction。 | 不是当前方法的训练晋级；没有新的合法 target/evaluator 前不能产生性能 claim。 | sole fallback, no run scheduled |
| E. Stop current explicit dynamics projection core | P1 直接满足“RGB/latent counterfactual 不合法”的停止条件；E0 尚待人工一致性且不存在训练方案比较。 | 不等于宣称所有 endpoint 或连续 feature mechanism 在任何 target 下都失败。 | **selected** |

## 6. 最近邻论文撞车分析

完整矩阵见 `docs/AUTORESEARCH_LITERATURE_MATRIX.md`。四个 AC-level 结论：

1. **Track4Gen 风险高。**候选 F3 的 identity refiner + zero-conv + temporal training 基本是 Track4Gen 架构；“换成 projected tracks”只有在 B5>B4>B3>B0 的完整 rollout 因果链成立时才是贡献。
2. **Generic relation alignment 已拥挤。**VideoREPA→MoAlign→SARA 已覆盖 generic TRD、motion bottleneck、pair routing；PhysAlign 又加入 V-JEPA2 全时空 Gram 与 explicit 3D geometry。
3. **Geometry teacher 不是空白。**Geometry Forcing 已把 frozen VGGT feature 蒸馏到生成器，PhysAlign 进一步覆盖 geometry+kinematics relation。
4. **Reward/short-chain 也不是空白。**SHIFT、DenseDPO、VideoGPA、Flash-GRPO 覆盖多种 post-training；SIFT 用 3-step self-imagination + motion classifier，ShortFT 用 shortcut chain 完整反传。

Motion-Proj 原本仅可能保留的边界是：公开 driving data、无 future GT、无新增 motion condition、frozen
Base rollout、自生成 point-track dynamics projection、无完整 chain backprop、单卡，并以独立 tracker 在完整
rollout 上证明高阶动力学改善而不是少运动。P1 已否决其 counterfactual supervision 源；E0 虽已完成
evaluator-only 机器稳定性，仍待人工一致性且没有可比较的训练 rollout，因此当前不能把这条边界写成方法贡献。

## 7. Phase 2 最终主路线与 fallback

### 主路线：停止当前 explicit dynamics projection（选择 E）

`Promote(r)` 的必要条件仍为 no-GT、projector physical validity、legal counterfactual target、locality、
held-out one-step 与 independent rollout。当前分别为 `1 / awaiting review / 0 / 0 / unknown /
evaluator machine pass awaiting review（无 comparative rollout）`；
其中 target 的 `0` 是 P1 机器 hard failure，不可由未完成的 P0 人审或更多训练覆盖。可训练模块为**无**；
Base、auditor、projector、VAE、feature hooks 和独立 tracker 均保持 no-grad。

### Fallback：新问题的只读诊断（选择 D）

最多重新打开两个前提：现有官方独立 tracker 的 12-panel human alignment 通过；以及一个不依赖大模型、可
明确 source removal/depth-order 的 counterfactual renderer 能在 P1-style gate 中通过。两者都不是当前项目
已有方法的小修；在它们之前不运行 short-chain、F1-R、F2、F3、O1 或任何生成器训练。

## 8. 明确停止做什么

- 停止用当前 temporal-only LoRA 继续 endpoint LR/preserve-weight sweep；
- 停止把更大的 preserve weight 当 locality 修复；
- 停止 F1-R、F2、F3、O1 与 short-chain；它们共享 P1 已失败的 target，且 E0 尚无人工完成的 evaluator 与任何训练 rollout 比较；
- 停止 300/800-step、Optuna、正式 cache 扩容和多个方向并行实现；
- 停止把 `background`/`dynamic_residual`/`foreground_candidate` 统称 object；
- 停止用 training auditor 或 legacy evaluator 宣称 rollout dynamics 改善；
- 停止用任一非官方或与 RAFT 同源的替代 tracker 填补 E0；
- 停止 generic flow/VFM/geometry alignment 作为主贡献；它们只保留为 baseline；
- 停止 reward/DPO/GRPO 主线；
- 不删除 endpoint 代码，不重构正式 Trainer，不 push。

## 9. 重新打开条件（未排程）

1. 完成 E0 v3 的 12-panel human review；它只能确认 evaluator alignment，不能反转 P1 failure 或创造 rollout improvement。
2. 若提出新的 counterfactual renderer，先在相同 7 个 frozen Base clips 上完成 source-removal、depth/occlusion-order、decoded trajectory realization 与 VAE round-trip 的只读 P1-style legality test；不训练生成器。
3. 仅在 1、2 都通过且 P0 人审完成后，重新预注册一个全新的机制选择问题。当前不会自动排程 F1-R、O1、F2、F3 或 short-chain。

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
- `motion_proj/diagnostics/svd_conditioning_parity.py`
- `motion_proj/diagnostics/projector_validity.py`
- `motion_proj/diagnostics/target_validity.py`
- `motion_proj/diagnostics/evaluator_validity.py`
- `motion_proj/eval/independent_tracks.py`
- `configs/diagnostics/autoresearch_f0_endpoint.yaml`
- `configs/diagnostics/autoresearch_f1_features.yaml`
- `configs/diagnostics/autoresearch_{c0_conditioning,p0_projector,p1_target}.yaml`
- `configs/diagnostics/autoresearch_e0_evaluator.yaml`、`autoresearch_e0_evaluator_v2.yaml`、`autoresearch_e0_evaluator_v3.yaml`
- `tests/test_endpoint_locality.py`
- `tests/test_feature_discriminability.py`
- `tests/test_svd_parameterization.py` 的 Diffusers scheduler oracle
- `tests/test_svd_conditioning_parity.py`、`tests/test_projector_validity.py`、`tests/test_target_validity.py`、`tests/test_independent_evaluator.py`

成功 runs：

- `/root/autodl-tmp/runs/autoresearch-f0-endpoint-s20260713-6845411`
- `/root/autodl-tmp/runs/autoresearch-f1-features-s20260713-72ac28c`
- `/root/autodl-tmp/runs/autoresearch-c0-conditioning-s20260714-v2`
- `/root/autodl-tmp/runs/autoresearch-p0-projector-s20260714-v1`
- `/root/autodl-tmp/runs/autoresearch-p1-target-s20260714-v2`
- `/root/autodl-tmp/runs/autoresearch-e0-evaluator-s20260714-v1`（checkpoint-missing evidence）
- `/root/autodl-tmp/runs/autoresearch-e0-evaluator-s20260714-v2`（perturbation self-correlation scope-bug evidence）
- `/root/autodl-tmp/runs/autoresearch-e0-evaluator-s20260714-v3`（machine pass / awaiting reviews）

F1 failed attempts（保留用于 provenance，不作为数据）：

- `/root/autodl-tmp/runs/autoresearch-f1-features-s20260713-9200467`
- `/root/autodl-tmp/runs/autoresearch-f1-features-s20260713-171ec2a`

测试：E0 v3 code commit `016f752` 前 full suite 为 `151 passed, 2 warnings`；官方 CoTracker3 权重已按
SHA256 固定，v3 生成 12 个可解码 overlay。P0/E0 panel 人审仍未完成，不能伪造人工一致性。完整 Phase 2
汇总见 `docs/AUTORESEARCH_PHASE2_REPORT.md`。
