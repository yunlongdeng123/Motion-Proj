# Action/Trajectory-Conditioned Backbone 迁移审计

> **任务**：`RP-C0-07`
> **日期**：2026-07-18
> **审计类型**：只读；未下载新权重、未安装新环境、未启动推理或训练
> **本地基线**：`e0063ab83d141cfc90b944bca2ee78317706dc56`，clean worktree
> **正式证据**：`/root/autodl-tmp/runs/route-pivot-c0-backbone-audit-s20260718-v1/`
> **结论**：选择 `C1`，以公开的 **ReSim exp0_no_carla** 为第一可执行迁移候选；这只是迁移晋级，不是方法或训练晋级。

## 1. Executive decision

V5 的 Route A 与 Route B 均已按预注册门禁拒绝。C0 的问题不是“哪个 driving generator 画面最好”，而是：

> 哪个公开模型能把未来 ego 运动作为显式条件，同时让 other-agent motion 仍由模型预测，从而真正重开
> driving-specific ego–actor entanglement 的可辨识性，并允许后续构造不依赖 future-agent GT 的物理偏好？

在 2026-07-18 可核验的一手发布物中，只有 ReSim 同时具备：

- 公开的 action-conditioned checkpoint；
- 历史帧条件视频预测，而非纯 layout-to-video；
- 8 点未来 ego trajectory 条件；
- 输入 schema 不要求 future 3D boxes，周车运动仍需生成；
- 官方 nuScenes validation JSON，且其中首个 CAM_FRONT 文件与本机原始 nuScenes 完全匹配；
- 单进程推理入口、LoRA/trajectory encoder 与多卡 ZeRO-2 训练入口。

因此选择：

```text
C1 = migrate to an action-conditioned driving backbone
primary feasibility backbone = ReSim exp0_no_carla
fallback baseline = VISTA
layout-control baselines only = OpenDWM / MagicDrive-V2
```

该选择有四个硬边界：

1. 公开 checkpoint 是 `exp0_no_carla`，不包含论文中依赖 CARLA 的非专家/危险动作完整能力；
2. ReSim 仓库发布了采样代码，但在固定 commit 的树中未发现论文 FID/FVD、IDM、Video2Reward 的正式评价实现；
3. 官方入口是单 GPU，不等于已经证明能在 24 GB RTX 4090 上完成 512×896、13 latent frames 推理；
4. 当前磁盘仅余 42 GB，选择性下载约 34.4 GB 后无法保持项目规定的 30 GB 安全线，故本轮禁止下载。

## 2. 审计协议

### 2.1 硬门

候选必须按下列顺序审计，不能用视觉质量替代上游条件：

1. 官方源码与可定位的 pretrained checkpoint；
2. nuScenes 或可低成本映射的真实驾驶数据 schema；
3. inference-time ego action/trajectory，而不只是文本、fps 或 camera nuisance；
4. other-agent future 不由 GT box/track 直接规定；
5. 可从历史帧预测未来；
6. 单张 4090 推理证据或明确的最小 feasibility test；
7. 两张 4090 上 LoRA/adapter 的结构可行性；
8. 官方评价、许可证与迁移成本可核验。

### 2.2 固定源码快照

| 候选 | 官方源码 HEAD |
|---|---|
| ReSim | `bf13dff45975eabbabc4e7de778207d2bb785e9b` |
| VISTA | `cc9821b4253ca7987c32757613d2fc2448fa9f5d` |
| OpenDWM | `b0ecc3d4020612376ea5a87500f98bc76893428f` |
| MagicDrive-V2 | `4ed72c60e5e73e4fa6072a7321fcc2ed9668edee` |
| DriveDreamer | `da1ca92f831bc23d91b59ad418eb47b41cbb1fa9` |

VLA-World 的官方项目页在审计日没有源码、checkpoint 或数据下载入口，因此没有可固定的代码 commit。

## 3. 本机资源事实

| 项目 | 事实 |
|---|---|
| GPU | 1 × RTX 4090，24,564 MiB；审计时空闲 24,081 MiB |
| 磁盘 | `/root/autodl-tmp` 128 GB，总用量 87 GB，剩余 42 GB |
| 环境 | Python 3.10.20，PyTorch 2.4.1+cu121，CUDA runtime 12.1 |
| 数据 | nuScenes `samples/`、`sweeps/`、`maps/`、`v1.0-trainval/` 完整存在 |
| 新 backbone | 本机未发现 ReSim、VISTA、OpenDWM、MagicDrive-V2 或 DriveDreamer checkout/weights |
| 大文件动作 | 本轮未下载、未删除、未移动任何权重或数据 |

ReSim 官方 `nus_val_4k.json` 的首条图像路径：

```text
samples/CAM_FRONT/
n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151514512404.jpg
```

在 `/root/autodl-tmp/data/nuscenes/` 下实际存在。这证明原始 RGB/sweeps 可直接复用；不证明当前 SVD cache
可直接复用。

## 4. 候选总表

| 候选 | 官方权重 | 显式 ego action / trajectory | other-agent future | 历史帧预测 | 单 4090 | 两卡 adapter | 结论 |
|---|---|---|---|---|---|---|---|
| **ReSim exp0_no_carla** | 是，EMA 23.67 GB | 是，8×`[x,y,heading]` | 自由生成；schema 无 future boxes | 是，默认 3 context frames | 官方单进程；24 GB 未实测 | 结构支持，缩减 pilot 可行但未实测 | **C1 primary** |
| VISTA | 是，10.05 GB | 是，traj/cmd/steer/speed/goal | 自由生成 | 是，1 context frame | 官方建议至少 32 GB | 无现成双卡采样；需额外 offload/sharding | fallback |
| OpenDWM CTSD | 是 | camera transforms/pose 是轨迹代理，不是标准 action token | layout 版由 future boxes 规定 | 支持 reference frames | 官方短视频需 32 GB、长视频需 80 GB | 训练可分布式，2×24 GB 不保证单样本显存 | toolkit/layout baseline |
| MagicDrive-V2 | 是，EMA 8.15 GB | camera extrinsics/temporal control，不是 ego action 语义 | future boxes/maps 显式规定 | mask-based 条件生成 | 224×400 full 官方 21.93 GB，边缘可行 | sequence parallel 可用；stage-3 训练至少 4 GPU | geometry baseline |
| DriveDreamer | 未发现官方 checkpoint | action interaction 见论文/演示，发布代码主配置为 boxes/maps | future boxes/maps 规定 | 是 | 无公开权重，无法审计 | 无意义 | reject availability |
| VLA-World | 无公开代码/权重 | 是，action-derived trajectory | 论文声称自由预测 | next-frame imagination | 不可执行 | 不可执行 | watch only |

“自由生成”只表示没有把 future actor boxes 作为输入，不表示 actor physics 已正确。

## 5. Primary：ReSim exp0_no_carla

### 5.1 模型与条件

ReSim 论文采用 2B diffusion transformer、T5 与 3D causal VAE，并用两层 trajectory encoder 注入未来轨迹。
论文生成协议为 512×896、10 Hz、4 秒未来，条件包括 9 个历史帧与 4 秒、2 Hz 的 waypoint 序列。
[论文实现细节](https://proceedings.neurips.cc/paper_files/paper/2025/file/f502981cbe221d857ad409450a7917c3-Paper-Conference.pdf)

公开推理配置与当前 README 略有不同，但代码事实明确：

- `sampling_num_frames: 13`，允许 13/11/9 latent frames；
- `sampling_video_size: [512, 896]`；
- `apply_traj: true`；
- `fut_traj` encoder 的 `seq_len: 8`；
- diffusion backbone 为 30 层、hidden size 1920、30 heads；
- `inference_custom.sh` 固定 `WORLD_SIZE=1`；
- 采样后把 diffusion model 移到 CPU，再把 VAE 移到 GPU 串行 decode。

对应一手配置：

- [ReSim inference config](https://raw.githubusercontent.com/OpenDriveLab/ReSim/bf13dff45975eabbabc4e7de778207d2bb785e9b/sat/configs/infer_nus.yaml)
- [single-process launcher](https://raw.githubusercontent.com/OpenDriveLab/ReSim/bf13dff45975eabbabc4e7de778207d2bb785e9b/sat/inference_custom.sh)
- [sampling implementation](https://raw.githubusercontent.com/OpenDriveLab/ReSim/bf13dff45975eabbabc4e7de778207d2bb785e9b/sat/sample_video.py)

### 5.2 公开 checkpoint 的真实边界

官方 asset repo 提供：

```text
resim_ckpts/exp0_no_carla/30000-ema/mp_rank_00_model_states.pt  23,667,958,479 bytes
vae/3d-vae.pt                                                   1,176,149,148 bytes
resim_data_jsons/nus_val_4k.json                                  27,978,143 bytes
```

同时还有非 EMA checkpoint，不需要为推理重复下载。官方 README 要求 T5-v1_1-xxl，但 ReSim asset 文件列表
中没有 T5；需另取 CogVideoX-2b 的两片 text encoder，约 9.53 GB。选择性最小落盘约：

```text
23.67 + 1.18 + 9.53 + metadata/source ≈ 34.4 GB
```

来源：[ReSim Assets 文件清单](https://huggingface.co/api/models/OpenDriveLab-org/ReSim_Assets?blobs=true)。

公开的是论文 roadmap 中勾选的 **OpenDV + NAVSIM、without CARLA** checkpoint。未发布的 CARLA 版本才覆盖
完整危险/非专家动作分布。因此后续论文不得把当前公开 checkpoint 称为“已包含 ReSim 非专家能力”。
[官方发布说明](https://github.com/OpenDriveLab/ReSim/tree/bf13dff45975eabbabc4e7de778207d2bb785e9b)

### 5.3 nuScenes schema 与复用

官方 validation JSON 直接使用：

```json
{
  "img_seq": ["samples/CAM_FRONT/...", "sweeps/CAM_FRONT/..."],
  "traj_fut": [[x, y, heading], "... 8 points ..."],
  "cmd": "Moving_Forward",
  "token": 0
}
```

与本机兼容性：

| 资产 | 复用 |
|---|---|
| 原始 CAM_FRONT keyframes/sweeps | 直接复用 |
| nuScenes trainval metadata | 直接复用 |
| ReSim 官方 `nus_val_4k.json` | 小文件新增；路径可直接映射 |
| 当前 SVD latent/feature cache | 不复用 |
| A0 ego/actor geometry 与真实 delta-t 工具 | 可转为训练侧/审计侧 target，不进入 rollout evaluator |
| RAFT/P-UNC 与 CoTracker3 | 可继续保持 train/eval 隔离 |
| manifest、review、bootstrap、anti-collapse | 高比例复用 |

预计 raw-data 复用接近 100%，现有模型 cache 复用为 0%，评价/证据基础设施复用较高。迁移工作量为“中等”，
主要成本在独立环境、SAT checkpoint loader、显存验证与 ReSim 输出适配，而不是重做 nuScenes。

### 5.4 单卡与双卡判断

**单 4090 推理：`plausible but unverified`。** 官方确有单进程入口和 CPU/serial VAE decode，但论文只报告
单张 A100 生成 4 秒约 2 分钟，没有给 24 GB 峰值。故不能把“单 GPU”写成“4090 24 GB 已通过”。最小
smoke 必须先用 9 latent frames、1 case、fp16、串行 decode，实测峰值与输出完整性；OOM 后只允许预注册的
frame/resolution/offload 缩减，不能无界改模型。

**两张 4090 adapter：`reduced pilot feasible, scale unproven`。** 发布训练配置包含 rank-128 LoRA、trajectory
encoder、gradient checkpoint 与 ZeRO-2，多卡 launcher 存在；论文第二阶段也确实冻结 DiT、只训练 LoRA 与
trajectory encoder。但官方完整训练用了 40 张 A100，公开 config 的 `model_parallel_size=1` 会在每卡复制
主模型。两卡只适合 256×448、9 latent frames、micro-batch 1、小 rank、短 capacity pilot，不能据此承诺
512×896 正式规模。

### 5.5 官方评价与许可证

论文在 nuScenes action-free protocol 报告 FID 5.2、FVD 50.4，并用 inverse dynamics trajectory difference、
human trajectory-following 与 Video2Reward 做 action 评价；但固定源码树中未发现相应 eval/IDM/Video2Reward
实现。因此迁移后必须使用本项目独立 evaluator，并单独实现 action compliance audit，不能宣称复现官方指标。

代码为 Apache-2.0；checkpoint 继承仓库中的 CogVideoX model license，学术研究允许，商业使用另有登记和
限制。[model license](https://raw.githubusercontent.com/OpenDriveLab/ReSim/bf13dff45975eabbabc4e7de778207d2bb785e9b/MODEL_LICENSE)

## 6. 其他候选为什么不晋级

### 6.1 VISTA：动作条件正确，但资源与架构回退

VISTA 支持 trajectory、command、steering、speed 与 goal point，公开 10.05 GB checkpoint，模型共 2.5B 参数，
其中 UNet 1.6B；nuScenes action annotation 也公开。它是合法 fallback。
[官方仓库](https://github.com/OpenDriveLab/Vista/tree/cc9821b4253ca7987c32757613d2fc2448fa9f5d)

但官方采样文档建议至少 32 GB VRAM，且没有双卡采样方案；其架构仍基于 SVD。对当前项目而言，VISTA 能用
explicit action 缓解 ego ambiguity，却更接近已经暴露 actor entanglement 的 SVD family，因此不优先于 ReSim。
[sampling requirement](https://raw.githubusercontent.com/OpenDriveLab/Vista/cc9821b4253ca7987c32757613d2fc2448fa9f5d/docs/SAMPLING.md)

### 6.2 OpenDWM：良好 toolkit，不是干净的 action-conditioned actor testbed

OpenDWM 有 MIT 源码、nuScenes/Waymo/Argoverse 适配、多视角视频 checkpoint、camera transforms、ego pose、
3D boxes 与 HD map，并提供 FID/FVD 配置。其工程价值很高。
[官方 README](https://raw.githubusercontent.com/SenseTime-FVG/OpenDWM/b0ecc3d4020612376ea5a87500f98bc76893428f/README.md)

问题是 layout checkpoint 把 future actor boxes 直接渲染为条件；不用 boxes 又改变了 checkpoint 的训练分布。
它适合测试 layout adherence，不适合证明“模型学会了 other-agent physics”。官方还要求短视频至少 32 GB、
6–40 帧长视频 80 GB，因此单 4090 不成立。

### 6.3 MagicDrive-V2：单卡低分辨率可跑，但 actor 被条件规定

MagicDrive-V2 的官方表给出 224×400×6 views、17 帧/完整视频在 CPU offload 下分别约 17.91/21.93 GB，
是候选中单 4090 证据最清楚的模型。[显存表](https://raw.githubusercontent.com/flymin/MagicDrive-V2/4ed72c60e5e73e4fa6072a7321fcc2ed9668edee/doc/FAQ.md)

然而其输入包含跨时刻 camera、3D boxes、BEV map；周车轨迹主要由 future boxes 指定，且 stage-3 训练明确
至少需要 4 GPU。它可作为 geometry-control 画质/局部一致性 baseline，不能作为自由 actor motion 的主骨干。
源码为 AGPL-3.0，而 Hugging Face checkpoint card 标记 GPL-3.0，发布前还需单独处理许可证口径差异。

### 6.4 DriveDreamer 与 VLA-World：当前不可执行

DriveDreamer 发布研究代码但测试配置要求用户自己的 `weight_path`，未发现官方 checkpoint；full nuScenes
转换文档还明确可能耗时数天。VLA-World 的 action-derived trajectory 科学上高度相关，但项目页只有论文，
没有代码、权重、数据或模型许可证入口。因此二者只能进入 watch list。

## 7. C1 如何真正回应 reviewer，而不是换 backbone 重做 DPO

选择 ReSim 只解决“ego trajectory 可观测”这一可辨识性前提，不自动解决物理偏好。下一阶段的最小创新单元应是：

```text
same history + explicit ego trajectory
  ├─ action sibling：共同噪声、邻近可行 trajectory，用于因果 action-compliance audit
  └─ rollout sibling：同 history/trajectory、独立自然 seed，用于 preference support

common localized tracks / background support
→ ego-compensated actor residual + identity/survival/quality/motion-floor endpoints
→ scene-level bootstrap uncertainty + ROPE
→ strict / tie / incomparable partial order
→ dense common-support weights + outside/history/real-denoising anchors
→ short LoRA capacity test
```

它与 `acceleration reward + vanilla DPO` 的区别必须落实到四个不可删组件：

1. **Uncertainty-aware partial order**：无共同支持、CI 跨 ROPE 或指标冲突时 abstain；不强制二元 winner；
2. **Sibling design**：action sibling 只检验因果响应，rollout sibling 才进入偏好；不同难度 action 不直接全局比较；
3. **Dense safeguarded alignment**：只在共同可跟踪 support 上产生局部权重，support 外、历史帧和真实视频 denoising
   有显式 anchor；
4. **Entanglement/low-motion proof**：按固定 ego trajectory 做 actor residual，报告 stationary/moving、action shuffle、
   trajectory sensitivity 与 motion-floor；任何通过 slow/freeze/track loss 获胜的候选直接 invalid。

## 8. 预判 Reviewer 2

| 质疑 | 必须在下一阶段给出的证据 |
|---|---|
| “只是 ReSim + DenseDPO/UPO” | action sibling 与 rollout sibling 的角色分离；局部 partial-order loss、anchor 与 abstention 的独立消融 |
| “显式 trajectory 已经把答案喂给模型” | trajectory 只规定 ego；禁止 future actor boxes/tracks；评价 actor residual 与交互响应 |
| “scorer 自证” | RAFT/P-UNC 只做训练侧排序；CoTracker/第二独立几何 estimator + human review 做正式评价 |
| “偏好只是低运动/冻结” | action-conditioned motion floor、track survival、active coverage、identity 与 quality fail closed |
| “dense mask 只是 saliency” | localized gradient ratio、support-outside drift、track-stratum transfer 与 mask shuffle/null controls |
| “旧 SVD 负结果不能支持新方法” | 把旧结果只作为 failure diagnosis；在 ReSim Base 重新做 action-shuffle、stationary 与 actor-residual probe |
| “公开 checkpoint 不含论文关键非专家数据” | 明确命名 `exp0_no_carla`；不声称 full ReSim；先只做专家/邻近可行轨迹 |
| “小样本挑结果” | scene-disjoint preregistration、全部 invalid 计数、candidate yield 与 scene bootstrap，禁止事后筛 support |

## 9. 决策与停止条件

### 9.1 C1 晋级范围

C1 只解锁下一份计划中的 `ReSim single-card bootstrap feasibility`，不解锁：

- 大权重自动下载；
- 两卡训练；
- LoRA/DPO/AWR；
- full-CARLA/non-expert claim；
- future boxes 作为条件或评价真值。

### 9.2 下一阶段前置资源

在保持 30 GB 安全线时，最小 34.4 GB 下载要求至少约 65 GB 空闲；考虑源码、临时文件、输出与失败重试，
建议先把 `/root/autodl-tmp` 可用空间提高到 **70–80 GB**。单卡 smoke 通过前不需要配置第二张 GPU。

只有单卡 Base、action responsiveness、preference support 三个 gate 依次通过后，才需要为两卡 LoRA capacity
test 停机配置。若单卡 9-frame/offload 仍 OOM，或合法 rollout sibling yield 仍低于预注册门槛，则 C1 直接停止，
不以更多 GPU 掩盖候选/可辨识性失败。

## 10. 一手来源索引

- [ReSim paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/f502981cbe221d857ad409450a7917c3-Paper-Conference.pdf)
- [ReSim source at audited commit](https://github.com/OpenDriveLab/ReSim/tree/bf13dff45975eabbabc4e7de778207d2bb785e9b)
- [ReSim public assets](https://huggingface.co/OpenDriveLab-org/ReSim_Assets/tree/main)
- [VISTA source](https://github.com/OpenDriveLab/Vista/tree/cc9821b4253ca7987c32757613d2fc2448fa9f5d)
- [VISTA checkpoint](https://huggingface.co/OpenDriveLab/Vista)
- [OpenDWM source](https://github.com/SenseTime-FVG/OpenDWM/tree/b0ecc3d4020612376ea5a87500f98bc76893428f)
- [MagicDrive-V2 source](https://github.com/flymin/MagicDrive-V2/tree/4ed72c60e5e73e4fa6072a7321fcc2ed9668edee)
- [MagicDrive-V2 checkpoint](https://huggingface.co/flymin/MagicDriveDiT-stage3-40k-ft)
- [DriveDreamer source](https://github.com/JeffWang987/DriveDreamer/tree/da1ca92f831bc23d91b59ad418eb47b41cbb1fa9)
- [VLA-World project](https://vlaworld.github.io/)
