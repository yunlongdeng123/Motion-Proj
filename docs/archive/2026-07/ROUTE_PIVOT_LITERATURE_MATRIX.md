# 路线切换一手文献矩阵

> **对应计划**：`MOTION_ROUTE_PIVOT_AUTORESEARCH_PLAN_V5.md` / `RP-LIT-01`
> **核查日期**：2026-07-18
> **范围**：只使用论文原文、会议页面、官方项目页和官方代码仓库。博客与二手解读不作为结论依据。
> **职责**：固定 Route A/B/C 的创新边界；不替代实验结果与 `EXPERIMENTS.md`。

## 1. 结论先行

文献核查收紧了可投稿主张：

1. “给视频扩散中间特征增加运动/几何对齐”本身已不新。Track4Gen、VideoREPA、MoAlign、
   Geometry Forcing 与 PhysAlign 已分别覆盖点轨迹、视频基础模型关系、flow-predictive motion subspace、
   VGGT 几何表示和 synthetic physics + 3D 对齐。
2. “自然 rollout + best-of-N physics reward”本身也不新。CVPR 2026 的 WMReward 已用 V-JEPA2
   surprise score 做 BoN 与 guided sampling；SHIFT 已覆盖自动 motion reward、rollout advantage、SFT anchor
   和 adversarial reward refresh。
3. Motion-Proj 仍有一个足够窄、可验证的缺口：在**真实驾驶视频**上，用标定与实例标注把相机自车运动和
   交通参与者 residual motion 分开；先证明冻结 SVD 表示中是否可辨识，再用局部、置信度感知且带
   anti-collapse safeguard 的训练侧辅助目标进入模型，推理时不增加 future trajectory condition。
4. Route B 只能先作为“当前 SVD support 内是否存在可选自然候选”的 ceiling diagnostic。若它通过，
   后续训练仍必须以 driving-specific、局部可解释、独立评估和低运动拒绝为核心；不能把 BoN、AWR、
   vanilla DPO 或单一 acceleration reward 当作论文贡献。
5. 单张 4090 不适合直接复现 MoAlign（两阶段各 4×H100）、Geometry Forcing（8×A100）、
   SHIFT（正式 SVD 对比为 64 GPUs）或 EOT-WM（64×A800）。当前 V5 的“小样本 legality → frozen probe →
   LoRA capacity”顺序是必要的可行性取舍，不应扩大成新 backbone 训练。

## 2. 一手工作对照

| Paper | Venue/year | Backbone | Training data | Motion/geometry teacher | Supervision layer | Trainable modules | Inference conditions | Compute | Direct overlap | Remaining novelty for Motion-Proj | Official code availability |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [Track4Gen](https://openaccess.thecvf.com/content/CVPR2025/papers/Jeong_Track4Gen_Teaching_Video_Diffusion_Models_to_Track_Points_Improves_Video_CVPR_2025_paper.pdf) | CVPR 2025 | SVD I2V UNet | 567 个 24-frame、320×576 video-trajectory pairs；由 flow 和 segmentation 生成轨迹 | dense optical-flow trajectories | 第 3 个 decoder block 的 upsampler；44×81 feature；cosine cost volume + soft-argmax + Huber correspondence | temporal transformer blocks、8 层 2D-conv refiner、zero-conv；保留 diffusion loss | 与 SVD 相同的首帧 I2V；无新增轨迹条件 | 20k steps，4×H100，总 batch 4；论文采样 fps=7 | SVD 中间特征、轨迹监督、training-only motion prior | 标定驱动的 ego/actor decomposition；真实 instance residual；不以通用 flow trajectory 代替驾驶运动；先做 identifiability gate | [项目页](https://hyeonho99.github.io/track4gen/)和论文公开；截至核查日项目页未给出官方代码链接 |
| [VideoREPA](https://arxiv.org/abs/2505.23656) | NeurIPS 2025 | CogVideoX T2V DiT | open-domain Koala-36M 子集（论文协议） | 自监督 video foundation model | diffusion token 的 spatial/temporal pairwise relation；TRD | diffusion backbone + projection（按论文微调协议） | 文本条件不变 | 大规模 DiT 微调；不适合当前单卡直接复现 | spatiotemporal relational alignment、physics framing | teacher 不是 driving geometry；未分 ego/actor；未给局部 GT confidence 与低运动拒绝 | [官方代码](https://github.com/aHapBean/VideoREPA)公开 |
| [MoAlign](https://arxiv.org/html/2510.19022v1) | ICLR 2026 | CogVideoX-2B | Open-Sora Plan 350K 子集 + Wan2.1-14B synthetic 16K | frozen VideoMAEv2，经 RAFT-flow regression 压缩到 64-D motion subspace | CogVideoX 第 18 个 MM-DiT block；跨帧 soft token-relation distillation，按 frame distance 加权 | Stage 1 的 3D-conv projection/flow decoder；Stage 2 的 4-layer MLP projector 与 CogVideoX fine-tuning | 文本条件不变；无额外 motion input | Stage 1：50k iter、4×H100 80GB、batch 128；Stage 2：4k iter、4×H100、batch 32 | motion-specific subspace、appearance/motion entanglement、soft relational alignment | 不能再声称“motion subspace”本身；缺口是**驾驶特有的 ego-induced 与 actor-residual 可分性**、标定真值、局部置信度和 scene-disjoint controls | [论文](https://openreview.net/forum?id=OR0ySm4l9h)公开；截至核查日未定位到官方实现 |
| [Geometry Forcing](https://arxiv.org/html/2507.07982v2) | ICLR 2026 | DFoT / Next-Frame Diffusion | RealEstate10K 16-frame；Minecraft 32-frame | frozen VGGT backbone features | intermediate diffusion features；frame/patch angular cosine alignment + normalized-input scale regression | video model + lightweight projector/scale head | 相机位姿条件或 action condition 保持存在；不因 alignment 增加新条件 | 2k–2.5k steps，8×A100 | intermediate geometry alignment、训练稳定性、显式 scale 信息 | 驾驶动态 actor 与相机几何的分层监督；稀疏 LiDAR primary truth；SVD 首帧 I2V 且无 future pose/action condition | [官方代码与 checkpoint](https://github.com/CIntellifusion/GeometryForcing)公开 |
| [SHIFT](https://arxiv.org/html/2603.17426) | ECCV 2026 | SVD-1.2B；Wan2.2-TI2V-5B | DAVIS2017 4K clips；WISA-80K | SEA-RAFT transport residual/confidence；CoTracker3 trajectory state/correlation；learned ViT discriminators | pixel-motion reward；forward-process advantage-weighted denoising loss + real-data SFT anchor | SVD temporal-attention rank-32 LoRA；Wan all-attention rank-32 LoRA；reward models交替更新 | 原 I2V/TI2V 条件不变 | SVD 正式比较使用 64 GPUs；SHIFT 约 2.46×SFT/epoch | natural rollout、自动 dense motion reward、AWR/SFT hybrid、reward hacking 与 dynamic-degree collapse | 不能复制 pixel flux + AWR；缺口是 driving-specific partial support、ego/actor localized uncertainty、独立 evaluator 和可证伪 anti-low-motion gate | [论文与项目页](https://xiye20.github.io/projects/SHIFT/)公开；截至核查日未定位到官方代码仓库 |
| [WMReward](https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Inference-time_Physics_Alignment_of_Video_Generative_Models_with_Latent_World_CVPR_2026_paper.html) | CVPR 2026 | MAGI-1、vLDM、Sora2 等 | 无生成器训练；在模型自然候选上 inference-time search/guidance | V-JEPA2 latent world model prediction surprise | 生成结果/预测 latent 的滑窗 reward；BoN、gradient guidance、guidance+BoN | 无生成器参数更新；V-JEPA2 frozen | 原 T2V/I2V/V2V 条件；增加 test-time compute | PhysicsIQ 通常 16 particles；成本随候选数增长 | B0 的 natural independent seeds、BoN ceiling、foundation-model physics reward | B0 只能是诊断；若继续，贡献必须是驾驶局部 motion decomposition、uncertainty-aware support 与 dense safeguarded training，而不是 BoN | [官方代码](https://github.com/facebookresearch/WMReward)公开 |
| [EOT-WM](https://ojs.aaai.org/index.php/AAAI/article/view/38403) | AAAI 2026 | CogVideoX-2B / TiDiT | nuScenes，25,109 train + 5,369 val videos，25 frames，10 Hz | GT ego 与 other-vehicle trajectories；标定投影 | BEV trajectory 投影为 trajectory video，经共享 STVAE 后与 video latent channel-concat | trajectory injection、3D conv、CogVideoX Expert DiT | **必须输入 future ego/other trajectories** 与首段视频/文本 | 768×1280，60 epochs，64×A800，总 batch 128 | driving-specific ego/other disentanglement、nuScenes、trajectory-video alignment | Motion-Proj 只在训练侧使用 future annotation，推理仍只有首帧；目标是内化而非 controllable trajectory condition | [AAAI 论文](https://ojs.aaai.org/index.php/AAAI/article/download/38403/42365)公开；截至核查日未定位到官方代码 |
| [OpenDWM](https://github.com/sensetime-fvg/opendwm) | 持续维护的官方开源系统（核查至 2026） | CTSD on SD2.1/3.0/3.5、CogVideoX VAE、LiDAR VQ/DiT 等 | nuScenes、Waymo、Argoverse、KITTI-360 等 | boxes、maps、layout、LiDAR 等显式 driving condition | 多视角/长视频生成管线与 evaluator | 依模型而定 | 文本、layout、box/map、multi-view 等显式条件 | 官方称短视频需约 32GB；6–40 帧长视频需 80GB | Route C 的可迁移 backbone、nuScenes pipeline、公开 FID/FVD config | 当前 24GB 单卡不适合长视频训练；只在 A/B 均拒绝后做只读迁移审计，不把工程迁移称创新 | 完整训练/推理/评估代码及多组 checkpoint，MIT |
| [DrivingGen](https://arxiv.org/html/2601.01528) | ICLR 2026 | benchmark；评测 14 个通用、physics 与 driving-specific 模型 | 多地域 open-domain 与 ego-conditioned driving 条件；统一 100-frame horizon | SLAM + depth 恢复 ego trajectory；MTR trajectory encoder；agent/quality models | evaluator only：FVD/FTD、视觉/轨迹质量、agent/trajectory consistency、ADE/DTW | 无 | 支持 open-domain 与 ego-trajectory-conditioned 评测 | 400×100-frame 全套评估约单卡 1–2 天 | driving-specific reviewer protocol；证明 FVD 不足；agent consistency/trajectory plausibility | D0 必须报告视觉质量和运动/agent 指标的 Pareto，而非单一 FVD；短 8-frame CAM_FRONT 结论必须声明外部效度限制 | [官方项目页](https://drivinggen-bench.github.io/)和论文公开；截至核查日未定位到完整官方代码链接 |
| [PhysAlign](https://arxiv.org/html/2603.13770) | arXiv 2026 | Wan2.2 I2V adapter/LoRA | Blender rigid-body synthetic，约 3K clips；带 RGB、metric depth 与 physics parameters | frozen V-JEPA2 + explicit synthetic 3D geometry | DiT spatiotemporal Gram relation + 3D geometry alignment | adapter/LoRA 与 alignment heads | I2V 条件不变 | 小数据 adapter 训练；论文未给出可直接等同本机的单卡预算 | synthetic physics、Gram relation、3D+motion unified latent | 不能把 Gram/3D alignment 当新；Motion-Proj 的边界是真实驾驶标定、ego/actor residual、观测不确定性与 driving evaluator | [项目页](https://physalign.github.io/PhysAlign)与公开 LoRA 权重可见；截至核查日未定位到完整官方训练代码 |
| [SARA](https://arxiv.org/abs/2605.07800) | arXiv 2026 | Wan2.2 continual training | open-domain video；entity mask supervision | frozen VFM + text-conditioned saliency | saliency-routed token-relation distillation | Stage-1 saliency aligner + video model alignment modules | 原文本条件 | 大模型 continual training；不属于当前单卡直接复现范围 | 选择性/局部 relational alignment，避免把预算分给无关 token pairs | “局部化”本身不够；本项目必须由标定 geometry、actor support 与 uncertainty 决定局部权重，而非 text saliency | 论文公开；截至核查日未定位到官方代码 |

## 3. Reviewer 预判与预注册回答

| 可能质疑 | 若不处理会怎样 | V5 中可接受的回答 | 必须保留的反证 |
|---|---|---|---|
| “这只是 driving 版 MoAlign/VideoREPA。” | 核心贡献被判 incremental | A0/A1 分别证明 ego flow 与 actor residual 的合法性和可辨识性；A1 必须含 single-frame、shuffle、absolute-vs-residual controls；A2 只对通过的局部层训练 | actor residual 不优于 absolute target、shuffle 不降、scene holdout 崩溃即拒绝 Route A |
| “你只是把 Track4Gen 的轨迹换成 nuScenes box center。” | 被视为数据替换 | actor target 是把同一 actor 在 camera motion 下的 static-world projection 与真实 projection 相减，并按真实 Δt 归一；background 与 actor 使用不同几何真值 | moving/stationary residual AUC、ego magnitude confound、visibility break 和 synthetic sign tests 任一失败均不得训练 |
| “局部化与 SARA/PhysAlign 的关系不清。” | 局部 loss 被判已有 | locality 来自可见 actor tube、LiDAR support、occlusion/visibility 和 target uncertainty；不使用文本 saliency，也不对全 volume Gram 做无差别匹配 | 必须报告 support coverage、困难样本保留率和 zero-support 行为 |
| “BoN 已被 WMReward 做过。” | Route B 无方法新意 | B0 只估计 SVD natural support ceiling；不作为最终方法主张。只有 driving-specific scorer 能在独立 evaluator 和人审上同时通过时，才进入后续 safeguarded training | 与 random、dynamic-degree-only、WM/foundation-style global scorer 对照；无显著增益则拒绝 B |
| “reward 只奖励少动/静止。” | 低运动偏置直接否定 physics improvement | 同时报 motion magnitude、actor displacement、track survival、quality 与 human motion correctness；低于 Base motion floor 的 candidate 不可成为 winner | static/near-static negative control；按 Base dynamic strata 分层；禁止事后删高运动失败样本 |
| “future GT 泄漏到自由生成评估。” | 结论无效 | future ego/boxes 只用于真实训练视频 target 与 frozen probe；generated rollout evaluator 只看生成视频及首帧可得信息 | `test_real_motion_no_eval_leakage.py`、manifest 字段审计、evaluator import boundary |
| “提升来自 fps micro-conditioning，不是方法。” | 训练效果混淆 | R1 在所有 Route A/B 正式实验前固定并版本化 generation fps；同 initial noise 做 2/4/7 对照 | 若 fps 改动显著改变 motion，后续所有 Base 和 method 必须使用同一已注册 fps |
| “单一 scorer 自证循环。” | reward hacking / evaluator leakage | training scorer、CoTracker evaluator、质量指标与 blind human review 分离；B0 scorer 不作为唯一 promotion criterion | scorer/evaluator 排名相关但不完全同构；disagreement cases 必须进入人审 |
| “8 帧、单相机不能支持 driving world model 结论。” | 过度外推 | 主张限定为 short-horizon CAM_FRONT SVD representation/capacity；DrivingGen 类长时、多代理指标列为后续外部验证 | `UR-06` 保持 unresolved；不得用本轮结果声称长时闭环可用 |

## 4. 对当前实现的直接约束

### 4.1 Route A

- A0 primary truth 只用真实 timestamp、calibration、ego pose、3D annotation 与稀疏 LiDAR；
  monocular depth 只能作为 proxy ablation。
- A1 先 probe 后训练。若冻结特征不能在 scene-disjoint holdout 上区分 actor residual，禁止用更大 head 或
  更长训练把不可辨识问题伪装成 capacity 问题。
- A2 不复制 VideoREPA/MoAlign 的全局 relation loss。第一版应只在 A1 通过的层和 support 上使用 masked
  local targets，并对 target uncertainty、visibility、zero-support、first-frame fidelity 和 motion floor 做 safeguard。
- “EgoActor-Align”只有 A2 rollout 对独立 evaluator 与人审同时成立后才可作为方法名。

### 4.2 Route B

- B0 只生成相同 condition 下的 independent natural seeds；不得恢复 denoising perturbation sibling。
- Best-of-N 必须与 random seed、dynamic-degree-only、quality-only 对照；必须报告 oracle ceiling、selector
  accuracy 与 candidate support/variance，而不是只报告 selected-vs-base。
- 即使 B0 通过，后续也不能直接采用 vanilla DPO。可行方向只能是：由局部 evidence 和 uncertainty
  形成 partial order；只在 common support 上建立自然 sibling pairs；dense loss 对局部片段生效，同时由
  first-frame、质量、motion floor、out-of-support abstention 和独立 evaluator safeguard。
- 这一路线仍需证明解决 driving-specific ego/actor motion entanglement 或低运动偏置之一，否则只算
  工程后训练，不足以支撑 CVPR 2027 主贡献。

### 4.3 Route C

- OpenDWM 的长视频官方显存需求高于当前 24GB；EOT-WM 的训练规模更远超单卡。
- 因此 Route C 仅在 A/B 均拒绝后做 checkpoint、condition schema、nuScenes support 和单卡 inference
  迁移审计，不在本轮启动 backbone 训练，也不申请双卡规避预注册边界。

## 5. 一句话创新边界

> 本项目不再声称发明 video motion alignment、relational alignment、best-of-N 或 reward-weighted
> fine-tuning；它要验证并最终解决的是：在真实驾驶视频中，如何把相机自车运动与交通参与者 residual
> motion 变成可信、可局部化、能拒绝不确定样本且不诱导低运动坍塌的训练信号，并在不增加 future
> inference condition 的前提下稳定进入视频扩散模型。
