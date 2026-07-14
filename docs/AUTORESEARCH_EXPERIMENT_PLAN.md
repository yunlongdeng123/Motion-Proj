# Motion-Proj Autoresearch：低成本实验预注册

更新日期：2026-07-14。Phase 2 的最终决策是 **E：停止当前 explicit dynamics projection 核心方向**，唯一 fallback 为 **D：仅在新 target 与独立 evaluator 均可验证时继续只读诊断**。C0 已通过；P0 为 machine pass / awaiting human review；P1 target legality 已 machine fail；E0 v3 为 machine pass / awaiting human review。因此 F1-R、F2、F3、O1、R0 与任何生成器训练均不排程。A1 仍有效：F0/F1 不可被外推为所有 endpoint 或连续 feature relation 的普遍失败；本次停止源于 P1 的合法 counterfactual hard failure，而非单独依赖 feature cell 统计。所有预算仍限制为单张 RTX 4090 24 GB，禁止 Optuna、300/800-step、数百 GB cache 和自动 push。

## 0. Gate graph

```text
A0 algebra/provenance ──pass──> F0 endpoint locality ──fail──> stop current endpoint
                                  |
P0/P1 projector validity ──pass──> F1 feature resolution ──fail──> stop F2/F3
                                  |
                                  pass
                                   v
                         F2 frozen feature head
                                   |
                                  pass
                                   v
                         O1 held-out one-step
                                   |
                                  pass
                                   v
                         F3 future-only feedback
                                   |
                   one-step pass / rollout fail only
                                   v
                         R0 short-chain smoke
```

已发生的分支：A0 pass；F0 fail；旧 projector 的 F1 fail；C0 pass；P0 machine pass / awaiting review；P1 fail；E0 v3 machine pass / awaiting review。没有自动“下一步”：任何重新打开都必须是新 run ID、只读且先完成新的预注册。

### A1 — F0/F1 结论边界修订（completed；不涉及训练）

F0 保留其固定 single-pair/shared-temporal-LoRA/raw-`v` locality hard fail：不得继续扫 preserve weight、学习率或该 pilot 的更新步数。它不证明所有 endpoint projection、所有 sigma/pair/mask policy，或具备独立局部参数子空间的机制都无解。

F1 保留其当前 projector 下 frozen raw-feature probe 的高风险发现：绝大多数 correction 小于半个 feature cell，且没有现成 projected relation signal。它不证明 sub-cell continuous relation 不可学习；F1-R 将以 target separation、relation-gradient SNR 和 synthetic sub-cell calibration 为主判断，`<0.5 cell` 仅为描述性统计。

本轮必须执行的 gate graph：

```text
A1
 ├── C0 official-SVD conditioning parity
 ├── P0 projector physical validity ──pass──> P1 RGB/VAE target validity
 └── E0 independent evaluator validity

C0 + P0 + P1 + E0 all pass (including human gates)
 └── F1-R revised feature signal audit (read-only)
```

在上述四项完成前，F2/F3/R0 继续 blocked。A1 的验证只包括文档差异、Markdown 渲染和 Git whitespace 检查；不涉及模型训练或生成。

## A0 — Repository algebra, parameter and provenance audit（completed）

| Field | Pre-registration |
|---|---|
| ID | A0 |
| Research question | 当前 raw-v/x0 代数、LoRA 选择、replay parent 与 GT 隔离是否可信？ |
| Hypothesis | raw-v 代数正确；replay 来自 frozen Base 且无 future GT；locality 风险来自共享 temporal LoRA 而非 spatial LoRA 泄漏。 |
| Code changes | 只新增/扩展 scheduler-oracle unit test；不改训练实现。 |
| Fixed variables | Diffusers SVD scheduler config、25 timesteps、现有 V5 cache 122 items、现有 model config。 |
| Independent variable | scheduler timestep；cache item；module path。 |
| Metrics | c_noise/input-scale/x0 oracle error；roundtrip error；parent_kind/adapter/future GT flags；trainable tensor/module/parameter count。 |
| Expected failure | sigma floor 与原始 sigma 在极低噪声不一致；文档与 formal Trainer 分叉。 |
| Promotion threshold | oracle max error `<1e-5`；122/122 no-GT/Base parent；0 spatial trainable modules；manifest count 自洽。 |
| Stop condition | 任一 GT leakage、raw-v error `>=1e-5`、找不到真实 hook/adapter path。 |
| Artifacts | `tests/test_svd_parameterization.py`；F0/F1 `selected_modules.txt`；本文与 route-decision 文档。 |
| Estimated GPU budget | `<0.1 GPU-hour`；主要为 CPU/tests。 |

结果：pass。25-step oracle 的 c_noise/scale error 为 0，x0 max error `4.768e-7`；122/122 Base/no-GT；128 temporal modules、3,319,808 trainable params、0 spatial modules。发现 formal V2 未接 Trainer、conditioning parity 风险与 positive scalar weight cancellation，但没有 parameterization failure。

## F0 — Endpoint failure decomposition（completed；hard fail）

| Field | Pre-registration |
|---|---|
| ID | F0 |
| Research question | temporal-only LoRA 是否存在“correction 显著下降，同时 outside/frame0 近零”的参数区域？ |
| Hypothesis | 若 endpoint 机制只缺正确 preserve trade-off，至少一个 lambda 会在 correction `<=20% initial` 时满足 locality。 |
| Code changes | 新增 `endpoint_locality.py`、独立外部 lambda objective、fixed noise bank、gradient norm/cosine 与 Pareto artifact；不改 endpoint trainer。 |
| Fixed variables | cache index 34；sample id 固定；sigma 0.05；noise seed 20260714；LR `2e-4`；rank-16 temporal LoRA；200 updates；同一 target/teacher。 |
| Independent variable | `lambda_preserve ∈ {0,0.25,1,4,16}`。 |
| Metrics | correction loss/fraction；outside raw-v 和 x0 drift/Base RMS；frame-0 max drift；LoRA correction/preserve/total grad norm；gradient cosine；direction cosine。 |
| Expected failure | positive scalar 在旧 normalized loss 中抵消；修复后共享 LoRA 仍造成 outside/frame0 leakage。 |
| Promotion threshold | 任一 checkpoint 同时：correction fraction `<=0.20`、outside raw-v ratio `<=0.02`、frame0 raw-v max `<=1e-6`；grad finite/nonzero。 |
| Stop condition | 200 updates 后 0 feasible points；NaN/OOM；目标代数 roundtrip 失败。 |
| Artifacts | `/root/autodl-tmp/runs/autoresearch-f0-endpoint-s20260713-6845411/{manifest.json,metrics.jsonl,summary.json,figures,noise_bank.pt,resolved.yaml,COMPLETE}` |
| Estimated GPU budget | 实际 741.8 s，约 `0.21 GPU-hour`。 |

结果：hard fail。11 个 correction-qualified checkpoints，0 feasible；详见 route decision。

## F1 — Feature discriminability audit（completed；hard fail）

| Field | Pre-registration |
|---|---|
| ID | F1 |
| Research question | frozen Base SVD 的哪些实际 feature 层能分辨 observed track 与 dynamics-projected track？ |
| Hypothesis | 至少一个 stride-8/16 层的 majority correction `>=0.5 cell`，且 observed tracking PCK 显著高于 chance。 |
| Code changes | 新增只读 hooks、fixed-noise feature extraction、correlation/soft-argmax probe、projector invariant audit；不训练 SVD/head。 |
| Fixed variables | 8 fixed cache indices；sigma `{0.05,0.2,1}`；同一 noise seed；Base adapter disabled；7 actual hook paths。 |
| Independent variable | feature layer/stride、sigma、query stratum。 |
| Metrics | observed/projected correlation；argmax/soft-argmax PCK/error；heatmap TV；feature stride；correction cells；dynamic-vs-background AUC；projector displacement/velocity/direction/turn/visibility/support。 |
| Expected failure | temporal-attention/up-block features弱跨帧 correspondence；projector correction 亚 feature-cell。 |
| Promotion threshold | resolution：`fraction(correction<0.5 cell) <=0.50`；tracking：observed PCK@1cell `>=0.50` 且 projected/observed target 可区分；至少一个 fully eligible layer。 |
| Stop condition | 无 resolution-eligible layer；找不到可靠 hook；GT leakage；OOM。 |
| Artifacts | success：`/root/autodl-tmp/runs/autoresearch-f1-features-s20260713-72ac28c/`；两个 failed provenance runs：`...-9200467`、`...-171ec2a`。 |
| Estimated GPU budget | success 实际 51.6 s；含失败重试仍 `<0.1 GPU-hour`。 |

结果：hard fail。最细 stride-8 层仍有 94.97% correction `<0.5 cell`；无推荐层。F2/F3 立即 blocked。

## P0 — Dynamics projector validity v2（machine pass；awaiting human review）

| Field | Pre-registration |
|---|---|
| ID | P0 |
| Research question | 当前 projector 是否在保持一阶运动与可见性的同时，只修正不合理的高阶动力学？ |
| Hypothesis | 按 strata 分离并加一阶/可见性约束后，可以避免 frame0/turn/visibility 语义错误；如果 correction 仍普遍亚像素，则显式 projector 的训练信号本身不足。 |
| Code changes | 仅新增 experimental projector config/diagnostic；硬设 `p0=p0b`；记录约束残差；不替换正式 projector/cache schema。 |
| Fixed variables | F1 的同 8 Base clips/tracks；同 provider/query；不重新采样 Base；同 support masks。 |
| Independent variable | P-ID/P-CUR/P-CON/P-UNC 四个预注册候选；P-CON/P-UNC 使用一组固定 constrained 参数，不做 lambda grid。 |
| Metrics | correction px/cell；net displacement ratio；mean velocity error；direction cosine；turn sign；visibility expansion；support；accel/jerk ratio；dynamic-degree ratio；per-stratum statistics。 |
| Expected failure | 高阶正则把轨迹变成近常速/静止；background 被当 positive motion；p0 与最终 frame0 freeze 冲突；correction 仍低于 feature resolution。 |
| Promotion threshold | `p0` max error `<=1e-6 px`；visibility expansion/support violation 0；net displacement median `[0.9,1.1]` 且 p10 `>=0.7`；direction median `>=0.98`；turn preservation `>=0.95`；dynamic-degree median ratio `[0.8,1.2]`；primary correction 高于预注册 uncertainty SNR；12-panel human valid `>=87.5%`。feature cell 只留给条件性 F1-R。 |
| Stop condition | 任何 GT 使用；通过降低 displacement/dynamic degree 才降低 accel/jerk；全部配置仍 >80% correction `<0.5 stride-8 cell`；需要写正式 cache 或 >8 clips 才能判断。 |
| Artifacts | `runs/autoresearch-p0-projector-*/{manifest.json,resolved.yaml,metrics.jsonl,summary.json,machine_summary.json,track_rows.csv,synthetic_rows.csv,panels,reviews.template.jsonl}`。 |
| Estimated GPU budget | CPU 为主；若复用 RAFT features `<0.2 GPU-hour`。 |

建议能量与硬约束：

\[
\mathcal E(\tau)=\lambda_d\sum_t w_t\lVert p_t-p_t^b\rVert_1
+\lambda_a\sum_t\lVert\Delta^2p_t\rVert_1
+\lambda_j\sum_t\lVert\Delta^3p_t\rVert_1,
\]

同时约束 `p0=p0b`、endpoint displacement、mean velocity、direction/turn sign、original visibility 与 support。background 只做 preservation/negative relation；dynamic_residual 与 foreground_candidate 分开加权。

结果（2026-07-14）：`autoresearch-p0-projector-s20260714-v1` 在 clean commit `dfef913` 和固定 8 个
Base replay indices `8/34/37/60/69/81/99/114` 上完成。P-UNC 是唯一 machine-eligible candidate：
351 tracks 中 primary correction 290 个，均通过 SNR gate；frame-0/visibility/time-index/support
violation 均为 0，net-displacement median/p10 均为 1，direction median 1，turn preservation 95.40%，
dynamic-degree median ratio 0.862。P-CON 的 turn 88.79% 与 dynamic-degree 0.736 未达门槛；P-CUR
出现 frame-0 10.165 px、127 个 visibility expansion 且 dynamic-degree 0.112。synthetic calibration
覆盖匀速、加速、刹车、转弯、并线、jitter、outlier、遮挡恢复；P-UNC 保留 clean motion、改善所有
high-SNR outlier、拒绝且不放大 sub-uncertainty jitter。因 12 个 panel 尚无人工 verdict，最终状态为
`awaiting_reviews`；P1 可以以 machine eligibility 运行，F1-R 仍 blocked。

## P1 — RGB projector / VAE target locality（machine fail；no generator training）

| Field | Pre-registration |
|---|---|
| ID | P1 |
| Research question | `Xb→Xdagger→E(Xdagger)→MΔ` 是否对应合法、局部、可解码的 counterfactual target？ |
| Hypothesis | 当前 crop/resize/paste + source retention + VAE 会产生明显非局部 hybrid latent；若无法定义一致 target，应停止 endpoint 核心方向。 |
| Code changes | 新增只读 target-consistency diagnostic 与 panels；不改 warper/cache schema。 |
| Fixed variables | P0 machine-eligible 的 7 个有 primary component clips（index 114 只有 background preservation，排除）；same P-UNC tracks/masks；deterministic VAE mode；same Base RGB。 |
| Independent variable | full `E(Xdagger)`、masked delta hybrid、固定 radius-1 dilated hybrid、decode-reencode；现有 crop/resize/paste compositor 只作被审计对象。 |
| Metrics | mask in/out latent RMS；outside changed fraction；decode-hybrid vs `Xdagger` LPIPS/PSNR；source duplication IoU；occlusion collision；edge energy；human 8-case validity。 |
| Expected failure | hybrid latent 不可对应 target RGB；复制/ghosting 与遮挡错误被 dynamics loss 学习。 |
| Promotion threshold | 所有可构造 P0-primary target 的 frame0 exact；outside latent RMS/Base RMS `<=0.02` 或有可证明的 full-latent target treatment；decode-hybrid 与 target LPIPS `<=0.05`；decisive human validity `>=87.5%`；无 systematic duplication/occlusion failure。 |
| Stop condition | target 只能靠扩大 mask/提高 preserve 掩盖；validity `<75%`；需要引入 depth/large external model 才可定义 target。 |
| Artifacts | `runs/autoresearch-p1-target-*/{manifest.json,resolved.yaml,metrics.jsonl,summary.json,machine_summary.json,target_rows.csv,source_duplication_rows.csv,occlusion_overlap_rows.csv,panels,reviews.template.jsonl}`。 |
| Estimated GPU budget | `<0.5 GPU-hour`（VAE encode/decode + panels）。 |

结果（2026-07-14）：v1 run 保留了一个未移动 query overlap 的 occlusion-proxy scope bug；v2
`autoresearch-p1-target-s20260714-v2` 在 clean commit `960c4c2` 上限定为实际 integer-paste move
后复跑，machine fail 不变。7/7 constructed target 的 frame0 RGB/latent exact，hybrid mask 外 latent
ratio 最大 0.00871；但 index 34 的 P-UNC correction 量化后没有产生任何 RGB target change，hybrid
LPIPS 最大 0.06805（full VAE reconstruction 同值，超过 0.05），出现 1 个 source-retention duplication
proxy，并有 588 个无 depth order 的 moved-component overlap。当前 crop/resize/paste + masked hybrid
不能提供合法 counterfactual；endpoint A 和 F1-R blocked，不通过扩 mask 或大型修复模型补救。

## E0 — Independent rollout evaluator validity（machine pass；awaiting human review）

| Field | Pre-registration |
|---|---|
| ID | E0 |
| Research question | 不复用 training auditor、future metadata 或 projector output 时，能否稳定测量 Base rollout 的 point dynamics？ |
| Hypothesis | CoTracker3 可作为冻结独立 evaluator，但必须先证明 seed/re-run 稳定、occlusion 与 identity 指标可信。 |
| Code changes | 新增 evaluator wrapper/diagnostic；明确禁止读 cache track、future ego/box；不改训练 auditor。 |
| Fixed variables | 8–16 existing Base clips；25 steps；fixed seeds；same decoded mp4；同一 tracker checkpoint。 |
| Independent variable | duplicate evaluation run、seed、tracker confidence threshold。 |
| Metrics | evaluator-only first-frame grid 的 track survival、camera-compensated image-plane velocity/acceleration/jerk、identical-video coordinate/visibility/aggregate rerun、photometric/codec/resize perturbation、synthetic ordering、12-panel human overlay。 |
| Expected failure | CoTracker3 对低纹理驾驶背景或遮挡不稳；metric variance 大；legacy source metadata 泄漏。 |
| Promotion threshold | identical-video rerun aggregate relative delta `<=2%`；survival-threshold 与 Base-vs-perturbation 跨 clip rank correlation `>=0.8`；synthetic acceleration/jerk ordering 正确；12-case decisive valid `>=87.5%`；输入 provenance 只含 generated RGB + first-frame grid + evaluator weights。 |
| Stop condition | evaluator 读取 training tracks/future GT；官方 provider/checkpoint 不可验证；repeatability 不过；acceleration/jerk 主要由 tracker jitter 决定。 |
| Artifacts | `runs/autoresearch-e0-evaluator-*/{manifest.json,resolved.yaml,metrics.jsonl,machine_summary.json,summary.json,track_overlay,reviews.template.jsonl,COMPLETE|awaiting_reviews}`。 |
| Estimated GPU budget | `0.5–1.5 GPU-hours`，不生成新 rollout 时更低。 |

结果（2026-07-14）：v1 `autoresearch-e0-evaluator-s20260714-v1` 保留 checkpoint 缺失的 blocked evidence。
用户上传官方 `scaled_offline.pth` 后，v2 在 clean commit `d846c9c` 固定实际 SHA256
`2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`，但审计发现旧的
survival-threshold self-correlation 会重用基线值，不能替代预注册的 perturbation-stability 检验；v2 因而只作
scope-bug evidence。clean commit `016f752` 将 E0 协议升为 v2，保存每个扰动 aggregate，并要求四项
aggregate metric 的 Base-vs-photometric/codec/resize 跨 clip Spearman 全部 `>=0.8`、缺失值 fail-closed。

正式 v3 `autoresearch-e0-evaluator-s20260714-v3` 在相同 8 个 frozen Base index、seed `20260713` 与 cache
fingerprint `e2e3a3b35f6d…` 上通过全部机器检查：8/8 valid；identical rerun 坐标 max/visibility mismatch/
aggregate delta 均为 0；synthetic acceleration/jerk ordering 与 occlusion down-weighting 通过；photometric 与
codec 的四项 rank 均为 1.0，resize 的 acceleration rank 为 0.97619、其余为 1.0。resize 的绝对 aggregate
relative delta 中位数 10.03%、最大 31.84%，故仅将其解释为排序稳定，不能当作绝对物理 jerk 的标定。v3 输出
8 个真实 + 4 个 synthetic overlay，均为 8 帧且可解码；`reviews.template.jsonl` 尚无人工 verdict，状态严格为
`awaiting_reviews`，没有任何模型 rollout-improvement claim，也不授权另换 provider。

## C0 — Official-SVD conditioning parity（completed；legacy one-step context 不兼容）

| Field | Pre-registration |
|---|---|
| ID | C0 |
| Research question | fps id、condition noise augmentation 与 CFG 差异是否造成 one-step/rollout distribution mismatch？ |
| Hypothesis | raw-v algebra正确，但 conditioning implementation 与 official pipeline 不完全一致；修正 parity 后 Base rollout 可精确复现。 |
| Code changes | 新增 parity test/diagnostic；第一轮不改正式 generation defaults。 |
| Fixed variables | one clip、one seed、official diffusers SVD pipeline、same scheduler/latents/25 steps。 |
| Independent variable | fps 7 vs official adjusted id；noise augmentation off/on；CFG raw conditional vs pipeline combination。 |
| Metrics | image embeds/image latent/added-time-id diff；per-step raw-v/x0 diff；final latent/RGB LPIPS；first-frame diff。 |
| Expected failure | current `fps=7` vs official 6；declared 0.02 noise aug 未作用于 condition frame；CFG makes pilot teacher not equal rollout denoiser。 |
| Promotion threshold | selected parity configuration per-step raw output max error `<1e-4`、final latent RMS `<1e-4` under matched inputs；all differences attributable to named CFG branch。 |
| Stop condition | 无法复现 official Base；需要修改 scheduler/preconditioning algebra；发现 replay parent 不是 claimed Base。 |
| Artifacts | `runs/autoresearch-c0-conditioning-*/{manifest.json,metrics.jsonl,summary.json}` + unit tests。 |
| Estimated GPU budget | `<0.5 GPU-hour`。 |

结果（2026-07-14）：`autoresearch-c0-conditioning-s20260714-v2` 在 clean commit `b36e042` 上完成，
Diffusers `0.31.0`、固定 train condition index 0、25 steps、seed `2026071401`。official pipeline、
实际 backbone wrapper 与 `svd_official_v1` candidate 的 added IDs、condition noise、initial latent、
per-step input/raw unconditional/raw conditional/CFG/scheduler output、final latent 和 decoded RGB 均为
0 差异，rerun 也 exact，因此版本化 generation protocol 通过 C0。旧 `build_conditioning()` 的
conditional-branch 对比未通过：time ID 为 `7` 而 official 为 `6`，image embedding、latent 和 noise
语义也不同。该结果不重写旧 V5 cache；其 Base generation provenance 可保留，但不允许它支持新的
one-step-to-rollout transfer claim。证据目录含 `tensor_diffs.json`、`summary.json`、`metrics.jsonl`、
`manifest.json` 和 `COMPLETE`；首个 v1 run 因诊断 device 类型错误失败并保留，未复用其 run ID。

## F1-R — Revised projector feature-resolution re-audit（not run）

未运行。必要条件为 C0 pass、P0 human pass、P1 pass、E0 pass；实际状态为 `pass / awaiting_reviews /
fail / machine pass awaiting_reviews`。P1 的 target legality hard failure 单独即可阻断 F1-R，E0 人审待完成
也禁止将 evaluator 升格为完整 rollout 解释；
不得借 F1-R 的 sub-cell calibration 把非法 RGB/VAE target 重新表述为 feature 问题。

| Field | Pre-registration |
|---|---|
| ID | F1-R |
| Research question | 通过 P0/P1 的 revised projected tracks 是否在实际 SVD feature grid 上可辨？ |
| Hypothesis | 若 explicit dynamics correction 是有效训练信号，至少一层应有 substantial displacement，而不是只靠扩大 heatmap。 |
| Code changes | 复用 F1，只新增 revised-track loader/config；不训练。 |
| Fixed variables | 同 F1 8 indices、sigmas/noise/hooks；只替换 projected track target。 |
| Independent variable | old vs revised projector。 |
| Metrics | F1 全部指标；特别是 correction-cell distribution 与 observed/projected heatmap TV。 |
| Expected failure | target 仍 sub-cell；通过人为放大位移才可辨，破坏一阶运动。 |
| Promotion threshold | 至少一层 `fraction<0.5cell <=0.50`；observed/projected heatmap TV `>=0.10`；observed track PCK 不低于同层预注册下限。 |
| Stop condition | >80% sub-cell；需要虚构/放大 correction；P0 invariants 退化。 |
| Artifacts | `runs/autoresearch-f1r-features-*/`，schema 与 F1 相同。 |
| Estimated GPU budget | `<0.1 GPU-hour`。 |

## F2 — Frozen-SVD feature-head capacity（blocked by P1/E0；not scheduled）

| Field | Pre-registration |
|---|---|
| ID | F2 |
| Research question | 在不回注 UNet 的情况下，temporary refiner/correlation head 能否学习 projected relation，同时保持 background relation？ |
| Hypothesis | 若 frozen features 含可辨 motion correspondence，head 能在 4–8 pairs 上显著拟合 projected target 且不 collapse。 |
| Code changes | 临时 feature projection/refiner + correlation head；不得 feedback UNet；不改 cache schema。 |
| Fixed variables | F1-R 推荐的唯一层；frozen SVD；4–8 pairs；fixed sigma/noise；最多 200 updates。 |
| Independent variable | head/refiner width（只允许 identity/no-refiner 两项）；observed vs projected target。 |
| Metrics | train/held-out projected-track error；observed-vs-projected distinguishability；background KL/drift；heatmap entropy；motion collapse；locality vs RGB endpoint。 |
| Expected failure | head 记忆 pair；softmax collapse；background relations 被拖动；target sub-cell。 |
| Promotion threshold | train error reduction `>=50%`；held-out `>=25%`；observed/projected classifier AUC `>=0.8`；background relation drift `<=2%`；entropy/dynamic degree 不低于 Base 95%。 |
| Stop condition | P1/E0 或 F1-R 未通过；只 train 改善；background/collapse gate 失败；>200 updates。 |
| Artifacts | `runs/autoresearch-f2-head-*/{manifest.json,metrics.jsonl,summary.json,figures,head.safetensors}`。 |
| Estimated GPU budget | `0.5–1.5 GPU-hours`。当前预算为 0，因为 blocked。 |

## O1 — True held-out one-step capacity/locality（blocked；not scheduled）

| Field | Pre-registration |
|---|---|
| ID | O1 |
| Research question | 通过 target/locality gates 的候选机制能否在未训练 replay pairs 上改善 one-step target？ |
| Hypothesis | 合法机制应在 train 与 held-out 都改善，而不是单 pair memorization。 |
| Code changes | 修复 capacity evaluator，明确 separate train/held-out batches；共享固定 noise bank；不改 formal Trainer。 |
| Fixed variables | 8 train + 8 held-out（最多 24 total）；fixed sigma/noise；同 Base/target；最多 200 updates。 |
| Independent variable | candidate mechanism（一次只测一个）；no LR sweep。 |
| Metrics | train/held-out target error；outside/frame0 drift；gradient norm/cosine；direction；Base prior loss；per-pair worst case。 |
| Expected failure | train fit、held-out 不动；共享参数 locality；少数 pair 均值掩盖 worst case。 |
| Promotion threshold | train reduction `>=80%`；held-out reduction `>=30%` 且至少 75% held-out pairs 同方向；outside `<=2%`；frame0 numerical `<=1e-6`；no Base-prior collapse。 |
| Stop condition | 任一 locality gate 失败；held-out 0/negative；需要提高 LR 或 >200 updates。 |
| Artifacts | `runs/autoresearch-o1-heldout-*/{manifest.json,metrics.jsonl,summary.json,figures,noise_bank.pt}`。 |
| Estimated GPU budget | `1–3 GPU-hours`。 |

## F3 — Future-frame-only feedback feasibility（blocked by P1/E0/F1-R）

| Field | Pre-registration |
|---|---|
| ID | F3 |
| Research question | identity refiner + zero-conv、且 residual 对 t=0 硬置零，能否得到比 endpoint 更好的 correction/preservation Pareto 和独立 rollout dynamics？ |
| Hypothesis | 独立 spatial feedback branch 可消除 temporal-LoRA 的 frame0/outside leakage；若完整 rollout 不改善，则 feature self-metric 不足。 |
| Code changes | temporary future-only refiner/zero-conv；identity/zero init；原 UNet/LoRA 初始冻结；不改正式 Trainer/cache schema。 |
| Fixed variables | 最多 16 train/8 val；50–100 updates；F1-R layer；same Base teacher、sigma/noise；25-step fixed evaluation seeds。 |
| Independent variable | B5 head-only vs B6 future-only feedback；唯一 feedback strength。 |
| Metrics | projected relation/dynamics error；outside feature/raw-v/RGB drift；frame0 bitwise/numeric diff；independent tracker accel/jerk/survival/identity/dynamic degree；LPIPS/visual panels。 |
| Expected failure | tracking head 自指标改善但 rollout 不变；zero-conv 学 appearance shortcut；Track4Gen-equivalent result；frame0 path 仍从 shared downstream blocks 泄漏。 |
| Promotion threshold | val relation error improvement `>=30%`；frame0 residual exactly 0 且 output max drift `<=1e-6`；outside drift `<=2%`；8 val clips 中至少 6 个 independent accel/jerk 同向改善，aggregate `>=10%`；survival/dynamic degree/LPIPS 各不恶化超过 5%。 |
| Stop condition | F2/O1 未通过；只 head metric 提升；任一 preservation gate 失败；100 updates；显存 >22 GB。 |
| Artifacts | `runs/autoresearch-f3-feedback-*/{manifest.json,metrics.jsonl,summary.json,figures,panels,adapter.safetensors}`。 |
| Estimated GPU budget | `3–8 GPU-hours`。当前预算为 0，因为 blocked。 |

候选反馈严格定义为：

\[
\widetilde F_t=F_t+\mathbf 1[t>0] Z_\psi(R_\phi(F)_t),
\]

其中 `R_phi` identity init、`Z_psi` zero init，且 `t=0` residual 在 tensor 上硬清零，而不是只靠 loss。

## R0 — Short-chain memory and credit-assignment smoke（blocked fallback；not scheduled）

| Field | Pre-registration |
|---|---|
| ID | R0 |
| Research question | 当 one-step 全部 gate 通过但 25-step transfer 失败时，2–4 step truncated rollout 是否能在 24 GB 上提供可用梯度？ |
| Hypothesis | 2 steps + LoRA + checkpointing 可能可行；4 steps 可能超出吞吐/显存。 |
| Code changes | isolated diagnostic unroll；不接正式 Trainer、不保存正式 adapter；明确不是 ShortFT shortcut model。 |
| Fixed variables | one batch/one pair；same seed/conditioning；LoRA rank；bf16；gradient checkpointing；最多各 5 updates。 |
| Independent variable | unroll length 1/2/4；checkpointing on/off 仅 smoke。 |
| Metrics | peak allocated/reserved GB；seconds/update；gradient finite/norm；loss sensitivity；one-step vs truncated gradient cosine。 |
| Expected failure | OOM、吞吐不可接受、gradient explosion、truncation bias。 |
| Promotion threshold | peak allocated `<=22 GB`；`<=180 s/update`；5/5 finite；2-step gradient direction稳定；4-step非必需。 |
| Stop condition | 前置 one-step→rollout transfer gap 未隔离；OOM 两次；需要 CPU offload/多卡；任何正式长训练。 |
| Artifacts | `runs/autoresearch-r0-shortchain-*/{manifest.json,metrics.jsonl,summary.json}`。 |
| Estimated GPU budget | `<0.5 GPU-hour` smoke。 |

## Conditional feature-route mathematical protocol（只定义，不晋级）

该协议只有 F1-R/F2/O1 通过后才可使用；当前不是推荐方法。

### Base rollout 与生成轨迹

\[
X^b=G_{\theta_0}(\xi,c),
\]

\[
\mathcal T^b=\{p^b_{i,t},w_{i,t},o_{i,t},s_i\},
\quad
s_i\in\{\text{background},\text{dynamic_residual},\text{foreground_candidate}\}.
\]

### Dynamics projection

\[
\mathcal T^\dagger=\Pi(\mathcal T^b),
\]

其中 `Pi` 必须满足 P0 的一阶运动、visibility/support 与 frame0 约束。

### Feature 与 correlation distribution

\[
F^\ell_\theta(z_\sigma)\in
\mathbb R^{T\times H_\ell\times W_\ell\times D},
\]

\[
s_{i,t}(u)=\frac{\cos(q_i,F_t(u))}{\tau},
\qquad
\pi_{i,t}(u)=\operatorname{softmax}_{u}s_{i,t}(u).
\]

### Teacher-relative projected target

\[
\pi^{\mathrm{tar}}_{i,t}
=(1-\eta)\operatorname{sg}[\pi^0_{i,t}]
+\eta H^\dagger_{i,t}.
\]

`Hdagger` 是在可分辨 feature grid 上、以 `pdagger` 为中心且归一化的 heatmap；禁止靠扩大 `eta` 或 heatmap 半径掩盖 sub-cell target。

### Relation 与 dynamics loss

\[
\mathcal L_{rel}=\sum_{i,t}w_{i,t}
D_{KL}(\pi^{tar}_{i,t}\Vert\pi^\theta_{i,t}),
\]

\[
\hat p_{i,t}=\sum_u u\,\pi_{i,t}(u),
\]

\[
\mathcal L_{dyn}=\lambda_p\lVert\hat p-p^\dagger\rVert_1
+\lambda_a\lVert\Delta^2\hat p-\Delta^2p^\dagger\rVert_1
+\lambda_j\lVert\Delta^3\hat p-\Delta^3p^\dagger\rVert_1.
\]

### Preservation

- frame0：`1[t>0]` 在 residual tensor 上硬 gate，并做 exact numeric assertion；
- background：对 `s_i=background` 使用 `KL(sg[pi0] || pi_theta)`，不是 projected positive target；
- Base denoising prior：同一 `z_sigma,c` 上蒸馏 frozen Base raw output；
- trainable：F2 仅 temporary refiner/head；F3 初始仅 refiner/zero-conv；原 UNet/LoRA 冻结；
- stop-gradient：Base rollout、tracks、Pi、Base features/relations、auditor、independent evaluator 全冻结；
- real SFT：F2 不需要；F3 若加入，必须与 feature/Base-prior branch **共享同一 sigma/noise draw 或严格 noise-aligned sampling**，并单独报告有/无 real anchor；
- total（conditional）：

\[
L=\lambda_{rel}L_{rel}+\lambda_{dyn}L_{dyn}
+\lambda_{bg}L_{bg}+\lambda_{base}L_{base}
+\lambda_{real}L_{real}.
\]

## B0–B7 conditional baseline ladder

只有 F3 获准后才运行，且先用 16 train/8 val、50–100 update screen；不得并行启动全部正式实验。

| ID | Method | Only changed factor | Required evidence |
|---|---|---|---|
| B0 | Base | frozen SVD | 所有 rollout metrics 与 bootstrap CI 的参考 |
| B1 | real-only SFT | standard real denoising only | 控制“多训练几步”效应；noise schedule 与其他行匹配 |
| B2 | current endpoint projection | V2 endpoint target | 复现实测 leakage/rollout failure，不使用旧失效 weight semantics |
| B3 | observed-track feature | `H=H(observed track)` | 控制 generic tracking supervision |
| B4 | generic smoothed-track feature | median/standard smooth，无 dynamics constraints | 控制“任何 smoothing 都有效” |
| B5 | dynamics-projected track feature | `H=H(Pi(Tb))` | 主 projector contribution；必须优于 B4/B3/B0 |
| B6 | B5 + future-only feedback | identity refiner + zero-conv + t0 hard gate | 架构贡献；必须优于 B5 且 preserve 更好 |
| B7 | generic flow feature alignment | frozen flow/VFM relation | 控制 MoAlign-like generic feature teacher |

统一研究问题：`dynamics-projected > generic smoothed > observed > no track` 是否在完整 rollout 上成立？

统一固定变量：同 Base checkpoint、clip/seed、conditioning、steps、noise bank、update budget、optimizer、evaluator；每行只改变表中 factor。

统一指标：independent acceleration/jerk、track survival、identity consistency、dynamic degree、tracker consistency、LPIPS/visual quality、frame0/outside drift，按 clip/seed bootstrap CI。

预期失败：B3/B4/B5 indistinguishable、B6 只改善自身 head、B7 复制 MoAlign、motion 通过静止化改善。

promotion threshold：B5 对 B4、B3、B0 至少一个预注册 independent dynamics metric 相对改善 `>=10%` 且 95% paired bootstrap CI 不跨 0；survival/dynamic degree/LPIPS 各不恶化 >5%；B6 还需 frame0/outside gates。

stop condition：ordering 不成立、只有 training auditor 改善、任何静止化/identity collapse、screen 后无效；不进入正式长训练。

artifacts：每行独立 immutable run manifest/metrics/summary/figures/panels，另写 paired-comparison summary。

estimated GPU budget：screen 每行约 3–8 GPU-hours；**当前全部预算为 0，因 F1 blocked**。

## Global stop rules

- 发现 future GT 或 parent adapter 泄漏：立即停止并废弃相关 cache/run；
- raw-v/oracle 不一致：先修代数，不做任何训练；
- target invalid 或 correction 仍低于 feature resolution：停止对应 endpoint/feature 路线；
- locality、held-out、independent rollout 任一不过：不得以更高 LR、更多 steps 或更强 preserve 掩盖；
- 需要 >24 pairs、>200 updates、>16 rollout clips、>22 GB peak 或多卡才能回答当前诊断：先汇报，不扩资源；
- 与 Track4Gen/MoAlign/Geometry Forcing/SHIFT/SIFT/PhysAlign 的差异无法通过 baseline 明确：停止作为核心方向；
- worktree 出现无法归因的用户修改：不覆盖，先汇报。
