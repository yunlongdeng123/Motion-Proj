# Motion-Proj Autoresearch：2024–2026 最近邻文献矩阵

更新日期：2026-07-13。检索只采用论文原文、官方会议页面、官方项目页和官方仓库。`Sampling-chain backprop` 指生成器参数是否通过多步反向采样链接收梯度；普通的单步 diffusion/flow-matching 训练记为“否”。

## 结论先行

候选空间已经相当拥挤：Track4Gen 覆盖了 SVD 中间特征相关性、可微 soft-argmax、identity refiner、zero-conv feedback 和 temporal-block 微调；VideoREPA、MoAlign、SARA、PhysAlign 分别覆盖 generic token-relation、motion-subspace relation、局部 pair routing、以及“几何 + 全时空 Gram relation”；Geometry Forcing 覆盖 geometry-foundation-feature distillation；SHIFT、DenseDPO、VideoGPA、Flash-GRPO 覆盖多种 reward/preference post-training。Motion-Proj 若继续，不能把“feature alignment”本身作为贡献，只能把**无未来 GT 的 Base rollout → 显式驾驶动力学投影 → 可验证的生成动力学改善**作为核心边界。

## 主矩阵

| Paper | Venue/year | Backbone | Supervision | Training level | Sampling-chain backprop | Motion representation | Appearance preservation | Locality mechanism | Compute | Similarity to Motion-Proj | Remaining novelty |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [Track4Gen](https://openaccess.thecvf.com/content/CVPR2025/html/Jeong_Track4Gen_Teaching_Video_Diffusion_Models_to_Track_Points_Improves_Video_CVPR_2025_paper.html) | CVPR 2025 | SVD | optical-flow trajectories；局部 cost volume、soft-argmax、Huber tracking loss | 第 3 个 decoder block 的 upsampler feature；训练 temporal transformer、8-layer 2D refiner、zero-conv | 否 | observed point tracks / feature correlation | 原 diffusion loss；refiner identity init；zero-conv 零初始化；反馈支路 `stopgrad` | 前景/背景均衡采样、半径 35 的局部 correlation window；不是严格的生成输出空间 gate | 567 video-track pairs，20k steps，4×H100 | 与候选 F2/F3 的 head、refiner、zero-conv、temporal-block 训练几乎同构 | 仅剩 Base-generated replay、dynamics-projected 而非 observed tracks、future-only 硬 gate、驾驶动力学独立 rollout 证据；必须用 B3/B4/B5 证明 projector 增益 |
| [MotionDirector](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7327_ECCV_2024_paper.php) | ECCV 2024 | AnimateDiff / text-to-video U-Net | standard denoising + appearance-debiased temporal loss | dual-path spatial LoRA 与 temporal LoRA | 否 | implicit temporal appearance/motion separation | spatial/temporal 两阶段与 appearance-debiased objective | 无局部空间监督；参数在空间共享 | 官方实现可在单张 A5000 约 14 GB 运行 | 证明 temporal LoRA 可低成本调 motion，也提示 appearance/motion 解耦必要 | Motion-Proj 的显式生成轨迹 projector 与驾驶 rollout 评估仍不同；单纯 temporal LoRA 不是创新 |
| [SHIFT](https://arxiv.org/abs/2603.17426) | arXiv 2026 | SVD、Wan | pixel-motion reward：instantaneous flow transport + long-term trajectory dynamics；generated AWR + real SFT | denoising output / ELBO proxy；SVD temporal-attention LoRA | 否；先采样 rollout，再用 forward-diffusion proxy | optical flow 与 long-horizon trajectories，标量 reward | real SFT anchor；SFT/AWR 严格 noise alignment；对抗更新 reward model | 无严格空间 gate；无 hard frame-0 preservation | 4k DAVIS clips；论文主实验为大规模多卡（含 64-GPU 设置） | 直接覆盖 motion-reward post-training；还独立报告 Track4Gen 可降低 FVD 却压低 motion score | Motion-Proj 只能保留显式、可解释、offline dynamics projector 与低成本无 reward-model 路径；reward 主路线创新弱且资源不符 |
| [MoAlign](https://iclr.cc/virtual/2026/poster/10009784) | ICLR 2026 | CogVideoX / Wan 系列 DiT | frozen VideoMAEv2 motion subspace；spatial relation + 时间加权 cross-frame relation | 中间 DiT hidden（论文默认中层）+ 小 projector；保留 diffusion loss | 否 | flow-supervised motion bottleneck + token relations | motion bottleneck 抑制 appearance；standard diffusion loss | 时间距离权重；没有显式 RGB mask、frame-0 hard gate | 两阶段、大模型多卡训练；不是单卡 4090 级验证 | 直接覆盖 generic motion-feature relation alignment | 只有“由 Base 生成轨迹、经显式驾驶 dynamics projector 修改后的局部 relation target”可能区分；泛化的 optical-flow/VideoMAE 对齐不能作主贡献 |
| [Geometry Forcing](https://geometryforcing.github.io/) | ICLR 2026 | video flow model | frozen VGGT feature；patch/frame-level angular alignment + scale prediction | 中间生成器 feature + lightweight projector | 否 | 3D geometry foundation representation | 原 flow-matching loss；辅助 projector | 无局部运动 gate；无 hard frame-0 preservation | 8×A100；RealEstate/Minecraft 数千步 | 覆盖“geometry feature distillation 改善 world model consistency” | Motion-Proj 必须坚持 explicit driving dynamics projection，而不能把 generic geometry teacher 当主方法 |
| [DenseDPO](https://papers.nips.cc/paper_files/paper/2025/hash/fa9755043814e7f08d859a286bb83c35-Abstract-Conference.html) | NeurIPS 2025 | video diffusion models | 同一 GT 的 corrupted-denoise 候选；segment-level dense preferences | forward-process Diffusion-DPO，时间片段粒度 | 否 | preference / dense temporal segments | pair 共享来源以对齐 coarse motion；reference model | 时间段局部，不是空间局部；无 hard frame-0 gate | 约 10k labels（论文称为 vanilla 的约 1/3）；仍需生成与打分成对视频 | 覆盖 dense preference route，并专门避免“人偏好低运动无伪影”的偏差 | offline physics projector 若只生成 scalar preference，创新会被压缩；保留连续、可解释 track correction 才有区分 |
| [VideoGPA](https://arxiv.org/abs/2601.23286) | ICML 2026（[官方仓库](https://github.com/Hongyang-Du/VideoGPA)） | video diffusion + LoRA | VGGT/Depth-Anything 3D reconstruction quality 构造 winner/loser；DPO | v-pred ELBO energy，pair 共享 noise/timestep | 否 | geometry-derived preference | frozen reference + LoRA；共享 noise/timestep | 无局部空间 gate；无 hard frame-0 gate | 约 2.5k pairs、10k steps、8×A100，论文附录为多日级 | 直接覆盖 geometry-derived self-supervised preference alignment | Motion-Proj 只有不降维成 scalar preference、保留 explicit dynamics projection 与单卡诊断才不同 |
| [ShortFT](https://openaccess.thecvf.com/content/ICCV2025/html/Guo_ShortFT_Diffusion_Model_Alignment_via_Shortcut-based_Fine-Tuning_ICCV_2025_paper.html) | ICCV 2025 | 主要验证于 text-to-image diffusion | 任意 differentiable reward | trajectory-preserving few-step shortcut model 上的完整短链 | 是，短链完整反传 | reward，不是显式 motion track | LoRA / progressive strategy；依赖 base/shortcut prior | 无运动空间 gate；无 frame-0 机制 | 比长链低，但必须先有 trajectory-preserving few-step model | 覆盖“用短链代替完整采样链”的一般机制 | projector-specific 2–4 step credit assignment 尚可区分，但不能把朴素截断称为 ShortFT，也不能忽略额外 shortcut model 成本 |
| [SIFT](https://arxiv.org/abs/2606.27741) | arXiv 2026 | Wan2.1-T2V-1.3B、CogVideoX | pure-noise self-imagination；R3D/SlowFast 四类 motion CE；real-video MSE；hard-case replay | 高噪声 3-step imagined output | 是，短链 | camera-only / object-only / both / static 的 coarse category | lightweight real MSE | 无 point-level locality、无 hard frame-0；分类器交替降低单 reward overfit | 10k prompts、1k updates、8×H100 | 直接证明 reconstruction shortcut 与 pixel bias，并占据 short-chain motion supervision 邻域 | Motion-Proj 可保留连续轨迹、加速度/jerk、驾驶视频和明确 projector；但 short-chain 不能再被称为未探索的新机制 |
| [SG-I2V](https://openreview.net/forum?id=uQjySppU9x) | ICLR 2025 | SVD I2V | inference-time feature matching along user box trajectories | middle-resolution spatial self-attention；优化 noisy latent | 不训练参数；每个选中 sampling step 内反传到 latent | box trajectory / self-attention correspondence | 把 optimized latent 的高频替换回原 latent | Gaussian box weighting；空间 attention 的 K/V 替换为 frame 0 并 stop-grad | 零训练但每样本多次 latent optimization | 直接研究 SVD feature：原 upsample/temporal-attention 跨帧 correspondence 弱；middle spatial attention 需显式 frame-0 对齐 | Motion-Proj 的 learned intrinsic dynamics prior 与无外部 motion condition 仍不同；也说明不能仅凭层名假设 SVD feature 可追踪 |
| [DrivePhysica](https://arxiv.org/abs/2412.08410) | arXiv 2024 | OpenSora ST-DiT + ControlNet | ego/world coordinate alignment、3D instance flow、3D box/map conditions | condition branches + full generator | 否 | explicit ego/world motion、3D flow、boxes | ControlNet-style branch；first-frame 有 0-timestep/no-noise 概率 | instance/box region guidance；不是无条件 hard frame-0 freeze | 20k + 2k + 100k iterations，8×A100 | 同为 driving physics，但依赖结构化未来 conditions、公司/nuScenes 条件和大训练 | Motion-Proj 的公开视频、无新增 motion condition、Base self-replay post-training 边界仍清楚 |
| [VideoREPA](https://arxiv.org/abs/2505.23656) | arXiv 2025 | CogVideoX-2B/5B | frozen VideoMAEv2；spatial 与 cross-frame token-relation L1 | DiT hidden（默认 depth 18）+ MLP projector | 否 | generic VFM pairwise relations | 原 diffusion loss；soft relation 避免 direct feature disruption | 全局 pair matrix；无 local mask / frame-0 gate | 32k/64k videos，2k–4k steps，8×A100 80GB | 覆盖 generic spatiotemporal relation distillation | projected track target 必须证明超越 VFM relation，并避免仅换 teacher/加 smoothing |
| [SARA](https://arxiv.org/abs/2605.07800) | arXiv 2026 | Wan2.2 high-noise DiT | V-JEPA TRD + text-conditioned saliency；SAM entity/background mask + InfoNCE | 中间 DiT hidden；Stage-1 aligner 冻结，Stage-2 训练 DiT/projector | 否 | saliency-routed token relations | diffusion loss；冻结 VFM、saliency aligner | fuzzy-OR pair routing 覆盖 FG-FG 与 FG-BG，显式降低 BG-BG 权重；仍无 hard frame-0 | 大规模 internal multi-subject corpus、两阶段、多卡 | 已覆盖“局部 relation supervision / background relation routing”一般设计 | Motion-Proj 只剩 dynamics-projected continuous track target、公开驾驶 replay、future-only exact gate 与动力学因果 baseline |
| [PhysAlign](https://arxiv.org/abs/2603.13770) | arXiv 2026 | Wan2.2-I2V-14B | V-JEPA2 全时空 Gram relation + synthetic rigid-body depth/3D loss | DiT block 16；LoRA + temporary depth heads | 否 | VFM kinematics + explicit synthetic 3D geometry | flow-matching loss；relation margin；推理丢弃辅助 heads | 全时空 relation，无 point tube / hard frame-0 | 3k synthetic clips，4×H100，约 24h | 高度覆盖“generic physics feature relation + explicit geometry”组合 | Motion-Proj 必须证明驾驶 projector 输出的是不同的、可测的动力学干预；简单加 acceleration/jerk relation 不足以形成边界 |
| [Flash-GRPO](https://openreview.net/forum?id=VHqDQ1BQVw) | ICML 2026 | 1.3B–14B video diffusion/flow models | single-step GRPO、iso-temporal grouping、temporal gradient rectification | 一个 policy denoising step，但仍依赖 rollout group/reward | 否完整链；单步 policy gradient | scalar reward / group advantage | KL/reference-policy 类约束取决于实现 | 无 point locality、无 frame-0 gate | 相对 full-trajectory 显著降低，但目标仍是大模型 RL rollout | 占据“低成本单步 reward alignment”邻域 | Motion-Proj 不应改写成 generic RL；offline deterministic projector 与无需多样本 group rollout 仍是区别 |

## 四项高风险撞车判断

### 1. “Projected track feature alignment = Track4Gen + trajectory smoothing？”

按当前候选架构，Reviewer 2 会回答“基本是”。Track4Gen 已包含 SVD 中间 feature、correlation distribution、soft-argmax、identity refiner、zero-conv feedback、temporal transformer 微调。只有同时满足以下条件才可能构成清楚边界：

- target 来自 frozen Base rollout，而不是 real-video observed flow；
- `Pi` 保留位移、平均速度、方向、转向、visibility/support，并只修改不合理的二/三阶动力学；
- `background` 是 preservation/negative relation，`dynamic_residual` 与 `foreground_candidate` 是不同权重的 positive supervision；
- frame 0 的 feedback 在数值上硬为零；
- B5 在完整 rollout 上严格优于 B4 generic smoothing、B3 observed track 和 B0 Base。

当前 F1 的最小步幅层 stride 为 8 px，而 94.97% projected corrections 小于 0.5 cell，因此尚不能声称上述 target 在特征网格上可辨别。

### 2. “Generic motion feature alignment 已被 MoAlign 覆盖？”

是。VideoREPA 已给出 generic spatio-temporal TRD，MoAlign 又引入 flow-supervised motion subspace 与 temporal weighting；SARA 继续加入 saliency pair routing。把 teacher 换成 optical-flow encoder、VideoMAE 或 V-JEPA，然后对齐 token/Gram relation，最多是强 baseline，不足以作 Motion-Proj 主方法。

### 3. “Geometry feature distillation 已被 Geometry Forcing 覆盖？”

是，而且 PhysAlign 已进一步把 V-JEPA2 relation 与显式 3D geometry loss 合并。Geometry foundation feature 应仅作为 B7/control baseline 或 evaluator，不能替代 dynamics projector 后再宣称原贡献成立。

### 4. “Reward-based motion post-training 已被 SHIFT / VideoGPA 覆盖？”

是。SHIFT 覆盖 flow/trajectory motion reward + AWR + real SFT；DenseDPO 覆盖 dense segment preference；VideoGPA 覆盖 geometry-derived DPO；Flash-GRPO 覆盖低预算单步 policy optimization。该路线还要求 rollout、候选采样与 reward robustness，不符合当前单卡和不做大规模 DPO 的约束。

## 对 Motion-Proj 可保留的创新边界

若后续证据最终支持继续，最窄但仍可答辩的 statement 是：

> 在不使用未来 GT、外部 motion condition、完整 sampling-chain 反传或大规模 preference rollout 的前提下，从 frozen driving-video generator 的 Base rollout 提取 point tracks，通过显式保持一阶运动与可见性的动力学投影只纠正高阶不合理性，并以严格局部、future-only 的机制把该干预蒸馏回生成器；贡献由 observed-track、generic smoothing、generic flow/VFM alignment 和 endpoint baselines 的完整 rollout 因果对比确认。

当前证据还不支持这句话：projector correction 在所有已测 SVD feature 网格上不可分辨，endpoint temporal LoRA 又无法满足局部性门槛。
