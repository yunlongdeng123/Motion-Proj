# Motion-Proj Autoresearch Phase 2 预注册

日期：2026-07-14

状态：`running`

本文件冻结第二轮诊断的假设、依赖、资源边界和判定规则。它不覆盖 F0/F1 的原始 evidence；其结论边界由 `docs/AUTORESEARCH_ROUTE_DECISION.md` 的 A1 corrigendum 定义。

## 1. 共同约束

- 只使用已有的冻结 SVD Base rollout、现有 V5 candidate 或最多 8 个既有 decoded Base clips；不新建大规模 cache，不修改旧 cache。
- 禁止 future GT ego pose、future box、future track、adapter parent 或训练 auditor 输出进入任一 target/evaluator。
- 不训练生成器、feature head、zero-conv/refiner 或 short-chain；不运行 Optuna、300/800-step 或正式 Trainer V2 集成。
- 每个正式诊断使用唯一 run ID，保存 resolved config、manifest、seed、model/cache fingerprint、dirty 状态、`metrics.jsonl`、`summary.json` 和 `COMPLETE`/`FAILED`/`awaiting_reviews`。
- 单卡 RTX 4090，单个任务峰值显存不超过 22 GB；不并行运行多个大模型。
- 所有有人工判断的任务只在 review 聚合完成后才能写 `pass`；未收到结果时状态必须是 `awaiting_reviews`。

## 2. Gate graph

```text
A1 conclusion-boundary corrigendum
        │
        ├── C0 official SVD conditioning parity
        ├── P0 dynamics-projector physical validity ──machine pass──> P1 target validity
        └── E0 independent rollout evaluator validity

C0 + P0 + P1 + E0 pass, including human gates
        │
        └── F1-R revised feature-signal audit (read-only only)
```

P0 fail 会阻断 P1/F1-R，但不会阻断 C0/E0。C0 或 E0 fail 不得用于新的生成协议或 rollout 性能结论。F1-R 未通过时只做路线决策，不训练 F2/F3。

## 3. C0 — Official SVD conditioning parity

问题：项目 Base generation 是否逐项等价于本地安装 Diffusers 的 `StableVideoDiffusionPipeline`。

固定输入：同一权重、condition image、generator seed、initial latent、25 inference steps、fps、motion bucket、noise augmentation、guidance schedule、dtype 与 device。诊断必须逐项记录 official、legacy 和 candidate protocol 的 added-time IDs、condition-noise tensor、image latent、initial video latent、scheduler timesteps/input scale、conditional/unconditional raw output、CFG output、scheduler output、final latent 与 decoded RGB。

通过门槛：matched protocol 的 added-time IDs、condition-noise、initial latent 与 scheduler timestep 精确相同；per-step raw output max error `<1e-4`；final latent RMS `<1e-4`；两个同输入 rerun 完全可复现。任何不一致必须定位第一处差异并以显式 `generation.protocol` 版本化，旧 V5 cache 不得静默改写。

执行记录（2026-07-14）：`autoresearch-c0-conditioning-s20260714-v2` 通过 `svd_official_v1` 的
generation parity 与 exact rerun，但 legacy one-step conditioning 的 fps/noise/CFG branch 不等价。
因此 C0 generation protocol 通过，旧 V5 stored context 保持 legacy，不可支持新的 transfer claim。

## 4. P0 — Dynamics projector physical validity

候选上限为四个：P-ID（identity）、P-CUR（当前 smoother）、P-CON（constrained robust smoother）、P-UNC（uncertainty-gated constrained smoother）。`background` 仅作 preservation/negative relation；`dynamic_residual` 是主要 correction candidate；`foreground_candidate` 仍须通过 visibility、support、confidence 和 uncertainty gate；不再把三者统称 object instance，统一使用“point-track tube component”。

所有候选的硬不变量：`p_dagger[:, 0] == p_base[:, 0]`（最大误差 `<=1e-6 px`）、不扩张 visibility、不在 absent frame 生点、不越过 original support、不改变有效时间索引、不使用 future GT。synthetic calibration 必须覆盖匀速、合理匀加速、刹车、平滑转弯、并线、tracker jitter、单帧 outlier 和遮挡恢复。

machine-pass 门槛：synthetic clean motion 不被系统性修改，noisy/outlier trajectory 向 clean 改善；visibility expansion 与 support violation 均为 0；net-displacement median 位于 `[0.9,1.1]` 且 p10 `>=0.7`；direction median `>=0.98`；turn preservation `>=0.95`；dynamic-degree median ratio 位于 `[0.8,1.2]`；主要 correction 的 uncertainty-normalized SNR 达到 config 中预注册的阈值。所有生成 correction 若低于 tracker uncertainty，判为 `fail`，不得人为放大。

输出至少 12 个分 strata panel 与 `reviews.template.jsonl`。人工 review 未完成时，P0 只能是 `awaiting_reviews`；可在 machine pass 后启动 P1，但不得启动 F1-R 或作最终路线晋级。

## 5. P1 — RGB / VAE counterfactual target validity

仅在 P0 至少产生一个 machine-eligible projector 后运行，最多 8 个样本；P0 的人工 gate 仍必须在 F1-R 或最终路线晋级前通过。比较 `z_full=E(X_dagger)`、当前 masked hybrid latent、一个预注册 dilated-hybrid latent，以及 `decode(hybrid) -> encode` 回环。分别记录 frame-0 exactness、mask 内/外 RGB 与 latent RMS、full/hybrid distance、decode-reencode error、decoded trajectory realization、source duplication、ghosting、occlusion violation、texture stretching、identity 与 direction。

通过门槛：8/8 frame-0 RGB/latent exact；outside latent RMS/Base RMS `<=0.02`，或有可验证的 full-latent treatment；decode-hybrid 与 projected target LPIPS `<=0.05`；无系统性的 duplication/occlusion failure；decisive human validity `>=87.5%`。任一系统性 hybrid invalidity 或需要大型视频编辑模型才可修复时，endpoint route 为 `rejected`。

## 6. E0 — Independent rollout evaluator validity

优先使用官方 CoTracker3，作为 optional、冻结且与 RAFT-chain 机制独立的 evaluator。输入 provenance 只允许 generated RGB、first-frame/query sampling 和 evaluator 自身权重；不得读取 cache tracks、projector outputs、future GT 或 source future metadata。query 固定并按 background、dynamic-residual、foreground-candidate 分开报告；无有效 track 必须为 `invalid` 而非 0。

验证包含 identical-video rerun、极小 photometric/codec/resize perturbation、已知平移/加速/转弯/遮挡 synthetic sanity，以及至少 12 个 overlay review panel。通过门槛：rerun aggregate metric relative delta `<=2%`；threshold sweep rank correlation `>=0.8`；synthetic acceleration/jerk ordering 正确；low-texture/occlusion failure 被识别并降权；decisive human overlay validity `>=87.5%`。所有指标名称必须使用“camera-compensated image-plane acceleration/jerk”并记录背景运动模型。

## 7. F1-R — 条件性 revised feature signal audit

只有 C0、P0、P1、E0 全部 pass 才运行，且保持只读。除旧 F1 的 descriptive cell statistics 外，必须报告 target TV/JS separation、relation-gradient RMS 相对于 dtype/repeated-forward/tracker-uncertainty 的 SNR、0.05/0.10/0.25/0.50/1.00 cell synthetic sub-cell calibration 的单调性、soft-argmax 和实际 target distinguishability。仅当 actual target 不可区分、gradient 接近数值噪声、sub-cell calibration 失败、或 correction 低于 uncertainty 时停止 feature route；不得仅因 `<0.5 cell` 停止。

## 8. 预先声明的停止与报告规则

发现 protocol 无法重现、Base provenance/GT 泄漏、projector correction 低于 uncertainty、target 系统性不合法、evaluator 不稳定，或需要超过本轮资源边界时立即停止相关分支并报告。完成后必须新增 `AUTORESEARCH_PHASE2_REPORT.md`，给出唯一主路线、唯一 fallback、每个 gate 的证据路径与未完成的人审状态。
