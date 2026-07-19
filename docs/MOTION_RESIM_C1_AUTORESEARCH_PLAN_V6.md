# Motion-Proj ReSim C1 Autoresearch 计划 v6（单卡 4090 证伪执行版）

> **工作标题**：Motion-Proj C1 — ReSim action feasibility → preference support → single-GPU pilot
> **目标投稿**：CVPR 2027（本 v6 只负责 idea screening，不把小样本结果包装成投稿结论）
> **创建 / 修订日期**：2026-07-19
> **计划基线**：`motion_proj` HEAD `43eda43878b5104cd043c4d8fee2ab177a356858`；ReSim HEAD `bf13dff45975eabbabc4e7de778207d2bb785e9b`
> **硬件边界**：仅一张 NVIDIA RTX 4090 24 GB；v6 不允许用第二张卡救场
> **计划状态**：`blocked`（C1A/`C1B-00` `done`；`C1B-01` proxy 校准 `blocked`，见 `RF-17`）
> **合法状态词**：`pending / running / blocked / done / rejected`
> **状态事实源**：[`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)、[`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)、[`EXPERIMENTS.md`](EXPERIMENTS.md)

---

# 0. 执行摘要

v6 只回答三个逐级问题：

1. **C1B — action feasibility**：公开的 ReSim `exp0_no_carla` 在训练支持内，是否会对专家 ego trajectory 产生可重复、可测量且不牺牲画质的响应？
2. **C1P — preference support**：在固定历史与固定专家轨迹下，只改变自然采样种子，是否能稳定得到足够多的、无退化捷径的 actor-motion 严格偏好对？
3. **C1S — single-GPU learning**：冻结 ReSim 全部原权重后，一个极小的新增 motion adapter 能否在单张 4090 上学习这些偏好，并在 scene-disjoint 留出样本上改善且不破坏 action、历史帧与非目标区域？

三问必须串行。前问失败，后问没有解释基础，立即停止。v6 的成功只表示“值得扩大样本和做正式消融”，不等于已形成 CVPR 主张。

## 0.1 本计划相对初稿的关键修正

- 主 action 对照从“乱序/物理非法轨迹”改为同一历史、同一 seed 下的 **expert trajectory vs action-free**。公开 checkpoint 只训练过专家轨迹和 action masking，不能把 OOD 乱序失败解释为物理能力。
- `exp0_no_carla` 的正确加载方式是 `args.load=<checkpoint 根目录>` 配合 `args.use_ema=true`；不得把 `30000-ema` 子目录直接当 `args.load`。
- `sampling_num_frames=9` 是 9 个 latent frames，对应 `4 × (9 - 1) + 1 = 33` 个 RGB frames：9 帧历史 + 24 帧未来，约 2.4 秒未来；它不是 9 个 RGB frames，也不是论文中的完整 4 秒未来。
- ReSim 的 `n_subset` 是 **shard 数**，不是“取前 N 条”。所有正式小样本运行必须生成精确的派生 JSON manifest，禁止用 `n_subset=1` 冒充单样本。
- 论文的 Trajectory Difference 依赖未公开的 IDM（XVO backbone + 额外 head）；本项目不得宣称复现该指标。先用真实 nuScenes 校准本地 ego-motion proxy，再辅以盲审；无法校准时结论为 `blocked/inconclusive`，不是 pass。
- 取消初稿的双卡 `C1-CAP`。v6 的容量与学习闭环必须在单卡 4090 上完成；若预注册的单卡内存方案仍 OOM，则拒绝本轮 single-GPU idea。
- 偏好门禁在生成候选前冻结预算、反退化阈值与扩样规则，避免看结果后换阈值或无限补 seed。
- 计划和任务状态只使用仓库规定的五个状态；等待人工复核写作 `blocked` 并记录阻塞原因。

## 0.2 授权与禁止

在当前计划已合入且执行代码干净的前提下，agent 可自动执行机器阶段，并在每个 gate 后更新三份事实文档。以下动作不在自动授权内：

- 删除或覆盖权重、数据集、环境、正式 run；空间不足时只能列出精确候选并向用户请求授权。
- 使用第二张 GPU、云 API、未记录的人工标签或外部闭源 evaluator。
- 在人工 gate 代替用户作出 human verdict。到达人工 gate 后必须生成完整 review packet 与可直接复制的 review prompt，将任务置为 `blocked`，等待用户回填。
- gate 失败后修改阈值、换主指标、追加 seed、扩大训练步数或把 held-out 变成训练集。
- 自动 push。代码或配置若为正式运行所必需，按 `AGENTS.md` 做小而自洽的 commit；未经用户要求不 push。

---

# 1. 已知事实、历史卡点与当前阻塞

## 1.1 V5 已经证伪的路线，不得在 v6 复活

| 证据 | 已知结论 | v6 约束 |
|---|---|---|
| A1：24 train / 8 val、21 个 layer/sigma 组合 | ego proxy 可改善 17.86%–25.01%，但 actor residual 低于零基线；stationary ratio 3.292–5.062 | ego camera-motion 改善不能冒充 actor physics |
| B0：16 条条件、最多 128 个 SVD 自然 rollout | 仅 1/16 条件有多样候选；独立 CoTracker 通过率 41.67% | 先做 preference support gate，未过不得训练 |
| 53 个机器偏好 / 48 个人工复核 | P1 的 22/24 为 tie；UPO 仅 2/96 strict | 禁止 forced binary；必须保留 strict/tie/incomparable |
| 既往单对容量实验 | 可以记忆，但会通过 temporal LoRA 泄漏到非目标区域 | 单对只作 capacity/locality diagnostic，不作科学结果 |
| RF-02 / RF-04 / RF-08 / RF-14 | 未条件化未来 GT 非法；低运动捷径；稳定 evaluator 不等于绝对物理；ego signal 不等于 actor residual | 所有门禁都显式检查信息合法性、运动量、独立 evaluator 与 actor/ego 分离 |
| RF-15 / RF-16 | 自然 seed 多样性不保证偏好支持；ReSim layout controllability 不等于 free actor physics | C1B 与 C1P 分开判；不能从 action pass 跳到 actor-physics pass |

更完整负结论以 [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md) 为准。若本计划与历史负结论冲突，先停止并修正文档，不得按较宽松版本运行。

## 1.2 2026-07-19 远端实况快照（环境就绪后修订）

远端根目录：

```text
/root/autodl-tmp/motion_proj
/root/autodl-tmp/third_party/ReSim                 # HEAD bf13dff45975eabbabc4e7de778207d2bb785e9b
/root/autodl-tmp/envs/resim                       # 独立推理/训练环境
/root/autodl-tmp/envs/motionproj                  # 保留：CoTracker/审计/本仓库测试
/root/autodl-tmp/runs
```

资产与环境盘点（修订后）：

| 项目 | 当前值 | 判断 |
|---|---:|---|
| GPU | RTX 4090，约 24 GB | 硬件符合 v6 |
| `/root/autodl-tmp` 可用空间 | 约 **42.7 GiB**（以实时 `df` 为准；清理/staging 前曾见约 34 GiB） | 高于 30 GiB 硬门槛；大批候选前仍建议 ≥40 GiB |
| `envs/resim` | 约 6.4G；Python 3.10；`torch 2.4.0+cu121`；`sat` editable；CUDA 可见 | **C1A-01 已就绪**（官方写 cu124，本机 cu121 wheel 可用） |
| `envs/motionproj` | 约 7.8G | **保留**，供独立 evaluator；禁止为腾盘删除 |
| asset root | `.../checkpoints/CogVideoX-2b-sat` 约 37G | 必需权重已齐 |
| `30000-ema` | 约 23.7G | 推理主权重；`args.load`→`resim_ckpts/exp0_no_carla` + `use_ema=true` |
| `30000` non-EMA | **已删除**（用户授权） | 推理不需要；需要时从 HF 重下 |
| `transformer.zip` / `vae.zip` | **已删除**（用户授权） | 官方 deprecated；解压件仍保留 |
| base `transformer/` | 约 3.5G | 保留至确认加载不依赖后再议 |
| VAE | `vae/3d-vae.pt` 约 1.1G | 已存在 |
| `nus_val_4k.json` | 约 28MB，4,519 clips | 已存在；须把 `meta.data_root` 改成本地路径 |
| T5 | `t5-v1_1-xxl/` 约 8.9G | **已就绪**；来自 `THUDM/CogVideoX-2b` 的 text_encoder+tokenizer 扁平合并 |
| nuScenes 本地 | `/root/autodl-tmp/data/nuscenes`；首条 CAM_FRONT 可解析 | 可用 |

**T5 说明（避免再踩坑）**：`OpenDriveLab-org/ReSim_Assets` **不含** T5；`--include "t5-v1_1-xxl/*"` 会 Fetching 0 files。正确来源是 `THUDM/CogVideoX-2b` 的 `text_encoder/*` + `tokenizer/*`，合并到：

```text
/root/autodl-tmp/third_party/ReSim/checkpoints/CogVideoX-2b-sat/t5-v1_1-xxl
```

并已用 `T5Tokenizer` + `T5EncoderModel` 做过加载烟雾测试（`LOAD_OK`）。

激活环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/resim
unset OMP_NUM_THREADS   # 或 export OMP_NUM_THREADS=1，避免 libgomp 警告
```

因此 **C1A 资产/环境前置已满足**；下一正式动作是 C1B smoke。所有写盘仍须遵守 §1.3。

## 1.3 磁盘硬约束与中间产物预算（128G 数据盘）

本机 **无法扩容** `/root/autodl-tmp`（128G）。推理/训练中间产物必须按预算写入，gate 结束后回收临时文件。

### 硬门槛

| 规则 | 阈值 | 说明 |
|---|---:|---|
| 运行中绝对下限 | **≥ 30 GiB** 可用 | 跌破则立即停止写盘；只列清理提案，等用户授权 |
| 正式候选批次启动前 | **建议 ≥ 40 GiB** | C1P 多 seed 视频极易吃光余量 |
| 单次 run 峰值预留 | `avail - peak ≥ 30 GiB` | 峰值估算写入 `RUN_PROVENANCE.md`；估不准则缩小 N/分辨率/存盘 |

### 允许落盘的位置

```text
正式证据（默认保留）:
  /root/autodl-tmp/runs/resim_c1_v6/<task_id>/<run_id>/

临时/可删（gate 结束后优先清）:
  ReSim/outputs/ 默认时间戳目录（复制进 run 后删除源）
  HF *.incomplete / 重复 staging
  未入选 candidate 视频、失败 smoke 重复 MP4、失败 trial ckpt
```

禁止把大临时文件写到系统盘 `/` 或未登记路径。

### 各阶段粗算预算（以实测 `du` 校准）

| 阶段 | 典型中间产物 | 粗算量级 | 回收策略 |
|---|---|---|---|
| C1B smoke | 少量 MP4 + log | 数百 MB–数 GB | 只保留正式 run 副本 |
| C1B action screen | 约 10×2 视频 | 数 GB | 面板/指标保留；多余视频可删 |
| C1P N=4→8 候选 | 最多约 16×8 视频 + tracks | **极易 20–40G+** | 先预注册预算；未入选必须可删；禁止并行第二批 |
| C1S 训练 | adapter ckpt、DeepSpeed/缓存 | 数 GB–十余 GB | 只保留通过 locality 的最终 adapter |
| Evaluator（motionproj） | CoTracker 缓存 | 可控 | 挂在对应 run 下 |

### 训练/生成时的强制检查

每次放大写盘前：

1. 记录 `df -B1 /root/autodl-tmp` 的 avail；
2. 估算本 run 峰值（视频 + ckpt + 余量）；
3. 若 `avail - peak < 30 GiB`：**不启动**，先清理或缩协议；
4. 禁止“先跑着看”；不能用第二张卡或静默删除正式 evidence 救磁盘。

受保护、默认不可删：正式 manifest/reviews/UPO 证据、R1/A0 待标材料、`30000-ema`、VAE、T5、`nus_val_4k.json`、本地 nuScenes、`envs/resim` 与 `envs/motionproj`。

---

# 2. 前沿方法与开源调研后的路线选择

以下调研以论文或官方仓库为准，日期截至 2026-07-19。

| 工作 | 可借鉴之处 | 不能直接解决本项目的原因 | v6 用法 |
|---|---|---|---|
| [ReSim, NeurIPS 2025 Spotlight](https://github.com/OpenDriveLab/ReSim) / [paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/f502981cbe221d857ad409450a7917c3-Paper-Conference.pdf) | 9 帧历史、future ego trajectory 条件、action masking；代码与 `exp0_no_carla` checkpoint 已公开 | 公开模型无 CARLA；论文 IDM 与 Video2Reward 未公开 | v6 唯一主 backbone；只在公开模型支持内作 action screen |
| [DriveLaW, CVPR 2026](https://github.com/xiaomi-research/drivelaw) / [video inference](https://github.com/xiaomi-research/drivelaw/blob/main/DriveLaW-Video/Infer/README.md) | 最新驾驶 world model 与大规模开源实现 | 当前视频推理接口是 condition video + text，没有显式 future ego trajectory，不能替代因果 action 对照 | 仅作扩展性与工程参考，不在 v6 换 backbone |
| [WorldLens, CVPR 2026 Oral](https://github.com/worldbench/WorldLens) | 将生成、重建、action-following、下游与人工偏好拆轴评测 | 不能提供本项目缺失的 actor GT；部分 Agent/26K 资源仍未完全释放 | 借鉴“分轴、不混成一个总分”的报告结构 |
| [PhyGenesis](https://arxiv.org/abs/2603.24506) | 面向 challenging trajectories 的多视角驾驶生成 | 条件中包含所有 agent future trajectories；这对本项目的 free actor prediction 是未来信息泄漏 | 作为边界对照，不作为 evaluator 或训练目标 |
| [RLGF, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/bb0f9af6a4881ccb6e14c11b8b4be710-Abstract-Conference.html) | 几何反馈、窗口化 latent 优化、点线面/occupancy 约束 | 几何一致性不等于 actor dynamics，且没有可直接复用的官方训练实现 | 只借鉴局部支持和几何 safeguard 思路 |
| [WorldModelBench, NeurIPS 2025 D&B](https://github.com/WorldModelBench-Team/WorldModelBench) | 多维 world-model benchmark 与开源评测 | 通用分数不能代替本项目的 action 因果性和独立 actor tracker | 可作后续 v7 外部验证，不作为 v6 主 gate |
| [WMReward, CVPR 2026](https://github.com/facebookresearch/WMReward) | 基于 V-JEPA2 的视频 reward 与 inference-time 多轨迹选择 | reward 仍可能偏好低运动/画质捷径；不能制造缺失的 preference support | 只有 C1P 通过后才允许作为附加 evaluator |
| [VideoDPO, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_VideoDPO_Omni-Preference_Alignment_for_Video_Diffusion_Generation_CVPR_2025_paper.html) / [VideoReward, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/76227feb18ea0ee40bd15cf02c33e18e-Abstract-Conference.html) | 多维偏好、偏好重加权、DPO/RWR 类训练 | 偏好优化不能修复“候选几乎相同”或“标签不可识别” | 只在 C1P 证明 strict pair 支持后借鉴优化形式 |
| [RLVR-World, NeurIPS 2025](https://github.com/thuml/RLVR-World) | verifiable reward 驱动的 world-model 强化学习 | 主要是序列/机器人 world model，不是 ReSim 扩散视频的轻量单卡 recipe | 不在 v6 引入 RL，避免额外不稳定性 |

路线决定：**保留 ReSim，先测其独有的显式 action 接口；评测采用本地校准 proxy + 独立 tracker + 盲审三角互证；训练只使用已经证明存在的自然 strict pairs。** v6 不进行 backbone tournament、reward-model 训练、PPO/GRPO 或多卡扩展。

# 3. 总体假设、因果边界与结论语言

## 3.1 预注册假设

- **H1（action feasibility）**：在同一 nuScenes 历史、同一文本 command、同一初始噪声下，给定数据内 expert ego trajectory 相比 action-free 会产生与请求方向一致的 future camera motion 响应，且不明显降低视觉质量。
- **H2（preference support）**：固定历史和 expert trajectory、只改变自然 seed 时，至少 12/16 个 scene-disjoint 条件能产生两个以上合格非重复候选，并形成足够的 strict actor-motion preference pairs。
- **H3（single-GPU learnability）**：冻结原 ReSim 全部参数后，预注册的小型新增 adapter 能在一张 4090 上改善 scene-disjoint held-out strict pairs，同时通过历史、画质、运动量、action 和非支持区 locality gate。

## 3.2 允许与禁止的主张

允许：

- “公开 ReSim checkpoint 对训练支持内 expert ego trajectories 有/没有可检测响应。”
- “当前自然 rollout 在给定协议下提供/不提供足够的严格偏好支持。”
- “单卡小适配器 pilot 在固定小样本上通过/未通过留出门禁。”

禁止：

- 把 action-free 当成“错误物理 GT”，或把 shuffled/OOD trajectory 失败当成物理错误。
- 把 camera/ego motion proxy 改善写成 actor dynamics 改善。
- 把 ReSim 论文的 IDM Trajectory Difference 名称用于本地 proxy。
- 对没有未来 actor GT 的 nuScenes 样本宣称绝对 physics accuracy。
- 把 8–16 个场景的 screening 结果写成 SOTA、泛化结论或 CVPR 主实验。

## 3.3 三类终止结果

- **工程失败**：资产损坏、依赖不一致、输出目录不确定、同 seed 不可复现或单卡 OOM。只有预注册工程修复可重跑，不能改科学 gate。
- **科学拒绝**：工程有效但 H1/H2/H3 的固定门禁未过，状态写 `rejected`，停止下游。
- **阻塞/不可判定**：例如 ego proxy 无法在真实数据上校准、人工 review 未回填或必要清理未获授权。状态写 `blocked`；不得把“测不准”写成模型 pass/fail。

---

# 4. 正式运行的通用契约

## 4.1 运行目录与证据

每次正式 run 使用：

```text
/root/autodl-tmp/runs/resim_c1_v6/<task_id>/<run_id>/
  RUN_PROVENANCE.md
  command.sh
  resolved_config.yaml
  env_freeze.txt
  git_state.txt
  asset_manifest.sha256
  data_manifest.json
  seeds.json
  stdout.log
  metrics.json
  outputs/
  review/                 # 仅人工阶段
```

`RUN_PROVENANCE.md` 至少记录：任务 ID、UTC/本地时间、repo HEAD 与 dirty diff hash、ReSim HEAD、GPU/driver/CUDA、Python/torch、完整命令、输入绝对路径及 sha256、样本 token→sample→scene 映射、seed、输出目录、峰值 VRAM、耗时、退出码和已知偏差。

ReSim `sat/sample_video.py` 默认写入 `ReSim/outputs/<config-stem>-<minute timestamp>`。正式 wrapper 必须在运行前后快照该目录，确认恰好新增一个输出目录，再复制/移动至 run 目录并记录原路径；若出现零个或多个候选目录，run 无效。禁止凭最新修改时间猜输出。

## 4.2 数据隔离与时间轴

- `nus_val_4k.json` 有 4,519 clips，clip 之间可能来自同一 scene。必须通过 nuScenes metadata 完成 `filename → sample_data → sample → scene` 映射后再做 scene-disjoint 划分。
- 正式小样本必须生成只含冻结条目的派生 JSON，并把 `meta.data_root` 改为本机真实 nuScenes 根路径。禁止使用远端不可达的 `/inspire/...`。
- 禁止把 `n_subset/ind_subset` 当样本筛选器。正式配置应指向精确 manifest，并禁用 shard 模糊选择。
- 原 JSON 的 `meta.fps_clip=12`、ReSim 模型配置的生成时间轴为 10 Hz、`traj_fut[:8]` 表示 2 Hz 的 8 个未来 waypoint。三者原样记录，任何重采样均需显式代码、单测和 manifest；不得悄悄互换。
- train、threshold-calibration、held-out 和 human-review 场景不得跨 split 重复。候选 seed 不能用于决定 scene split。

## 4.3 重复性与反结果导向

- 所有场景、seed、预算、阈值和扩样条件必须在查看对应候选指标前写入 `frozen_protocol.json` 并 sha256。
- 所有成对比较共享历史、文本、expert trajectory、采样步数和初始噪声；只改变该对照定义允许改变的变量。
- 主结果同时报告 per-scene 值、median、bootstrap CI 和 pass count，不只报告平均值。
- 一旦看过 held-out 指标或人工标签，该 split 永久冻结为 evaluation，不得回流训练。
- 每个 gate 只允许协议中明写的一次扩样/工程阶梯；否则需新版本计划。

---

# 5. Stage C1A — 资产、环境与执行前置

## C1A-00：资产与磁盘对账

**输入**：官方 ReSim README、当前 checkpoint/data 文件、当前磁盘快照。
**动作**：

1. 对 EMA、non-EMA、transformer、VAE、T5、`nus_val_4k.json` 和 nuScenes 数据根目录做存在性、大小、sha256/官方来源核验。
2. 打开 checkpoint 自带 `training_config.yaml`，确认 `latest=30000`、public run 为 `exp0_no_carla`、action masking 配置和模型尺寸。
3. 估算补齐资产、环境、一次 full-resolution smoke、一次 16 条候选批次所需空间。
4. 所有必需资产完成后，`/root/autodl-tmp` **硬性保留 ≥30 GiB**；建议启动正式候选生成前 ≥40 GiB。
5. 若不足，只输出清理提案。`transformer.zip`、`vae.zip` 是优先候选；non-EMA 与 base transformer 只有在验证 EMA 加载、恢复策略和来源后才能列为候选。未获授权不得删除。

**pass**：资产均可读、来源/指纹完整、nuScenes 本地路径有效、补齐后空间 ≥30 GiB。
**当前判断（2026-07-19 修订）**：`done`。EMA/VAE/T5/`nus_val_4k.json`/本地 nuScenes 可解析；zip 与 non-EMA 已按用户授权清理；可用约 42.7 GiB。进入 C1B 前仍须把 asset fingerprint 写入正式 run 目录，并遵守 §1.3。
**失败动作**：更新事实文档和精确清理提案，停止。

## C1A-01：独立 ReSim 环境

固定路径：

```text
/root/autodl-tmp/envs/resim
```

**已建立事实**：

| 项 | 值 |
|---|---|
| Python | 3.10.20 |
| torch / torchvision | `2.4.0+cu121`（官方文档写 cu124；本机 CUDA 可见且可用） |
| sat | vendored `SwissArmyTransformer` editable |
| 关键 imports | `torch`、`sat`、`diffusers`、`transformers`、`decord`、`cv2`、`einops`、`deepspeed` 已通过 |
| T5 路径 | `checkpoints/CogVideoX-2b-sat/t5-v1_1-xxl`（CogVideoX 合成，非 ReSim_Assets） |
| 体积 | 约 6.4G |
| 与 `motionproj` | **并存**；evaluator 继续用 `motionproj`，禁止混装污染 |

激活：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/resim
unset OMP_NUM_THREADS
```

**pass**：imports 全过、torch/CUDA/GPU 正常。
**当前判断**：`done`。
**fail**：工程失败；只修环境，不进入推理。


---

# 6. Stage C1B — ReSim action feasibility

## C1B-00：单样本 smoke、形状与确定性

### 6.1 固定加载语义

正式 resolved config 必须满足：

```yaml
args:
  load: /root/autodl-tmp/third_party/ReSim/checkpoints/CogVideoX-2b-sat/resim_ckpts/exp0_no_carla
  use_ema: true
  seed: <manifest_seed>
  sampling_num_frames: 9
  sampling_video_size: [512, 896]
  apply_traj: true
  save_gt: false
  concat_gt_for_demo: false
  save_recon: false
  model_parallel_size: 1
```

`latest=30000` 与 `use_ema=true` 应解析到 `30000-ema`。运行前打印并记录最终解析出的 checkpoint；解析不符即失败。单样本来自精确派生 JSON，不使用 `n_subset=1`。

9 latent frames 的 RGB 输出应为 33 帧，其中 9 帧历史、24 帧未来。若输出帧数、尺寸、轨迹长度或 history/future 边界不符，禁止继续。

派生配置还必须令 `model.network_config.params.num_frames=33`。该字段决定 Transformer 内部的
`compressed_num_frames`，必须与 `args.sampling_num_frames=9` 一致；它不同于下面保持 49 的 dataset/VAE 输入帧数。

ReSim 官方 `encode_first_stage` 的 dataset/VAE 输入契约仍为 49 个 source RGB frames；其 causal VAE 编码后，
采样只读取前三个 history latent。不得把 `sampling_num_frames=9` 误设成 33 帧 dataset 输入；正式输出仍必须是
9 latent / 33 RGB，且后续 evaluator 只把前 9 RGB 当 history、后 24 RGB 当生成 future。

主 smoke 不保存拼接 GT，减少显存/磁盘；source history 和 VAE round-trip baseline 单独提取。当前官方 sample code 已在 diffusion 后把 model 移至 CPU 再进行 VAE decode，不能把这一既有行为误报成新的优化。

### 6.2 单卡内存阶梯

只允许两级、严格顺序：

1. **B00-L0**：512×896、9 latent frames、官方采样步数、fp16、单样本。
2. **B00-L1**：若 L0 CUDA OOM，改为 256×448、仍为 9 latent frames。必须同步修改 `sampling_video_size`、dataset `video_size`、network `num_frames=33`、network `latent_height/latent_width = 32/56` 以及 position interpolation 的 height/width；运行 shape preflight 后再采样。

L1 仍 OOM，C1B 工程失败并停止。不得减未来帧、减少 diffusion steps、换 checkpoint 或加第二张卡来制造 pass。若 L0 过，则后续 action screen 使用 L0；只有 L0 OOM 时才统一使用 L1，不能按场景混分辨率。

### 6.3 确定性 gate

同一精确输入和 seed 连跑两次，在启用 PyTorch/CUDA 可用的 deterministic 设置后比较输出 sha256 与逐像素差异：

- 优先要求视频解码后的帧数组完全一致；
- 如底层算子仍非 bit-exact，必须在第三次重复前定位来源，并冻结一个数值重复性容差；容差只由重复 run 决定，不能看 action 结果后设定；
- 重复噪声大到足以覆盖 action effect 时，状态为 `blocked`，不得进入 C1B-02。

**pass**：checkpoint/shape 正确、单卡无 OOM、输出可播放、无 NaN/Inf、重复性噪声可控、峰值 VRAM 和时长已记录。
**fail**：工程失败，停止。

**执行结果（2026-07-19）**：`done`。v4 在 L0 CUDA OOM 后按唯一允许路径进入 L1；L1 两次完整采样均为
33×256×448，解码像素 SHA-256 相同、`max_abs=0`，峰值 VRAM 为 `23,617/23,651 MiB`。后续 C1B 统一使用 L1。

## C1B-01：场景冻结与 ego proxy 校准

### 6.4 场景集

从 `nus_val_4k` 预注册 10 个 scene-disjoint contexts：

- moving primary：8 个（4 forward、2 left、2 right）；
- stationary/near-stationary safeguard：2 个；
- 优先包含近距离可跟踪 actor、非纯空路、不过曝/夜间不可见、history 无严重损坏的场景；筛选只基于 source history、metadata 和 GT ego trajectory，不得查看生成 future。

每个 context 固定 token、scene、command、8 个 2 Hz expert waypoints、source frames、seed 和筛选理由。两个 stationary 场景只检查错误运动与画质，不计入 action 主 pass count。

### 6.5 本地 proxy 必须先在真实数据上校准

基于现有独立 tracking/robust background compensation 代码，从真实 nuScenes future frames 估计粗粒度 camera ego-motion。校准集与 10 个 screen contexts scene-disjoint。至少检查：

- left/right/forward class 的 balanced accuracy；
- 转向符号准确率；
- proxy displacement 与 GT ego displacement 的 Spearman 相关；
- 相对 constant/command-only baseline 的增益；
- stationary false-motion 分布。

在读取任何生成 future 前冻结 48 个 calibration scenes：stationary/forward/left/right 各 12 个，每类按稳定
scene hash 固定为 6 个 fit 与 6 个 held-out eval；同一 scene 只出现一次。真实 trajectory 以 2 Hz waypoint
线性插值到 2.4 秒 horizon。由于 ReSim source frame 的真实 timestamp 并非固定 10 Hz，校准必须读取完整
49-frame source sequence，从第 9 个 history frame 起按 `sample_data.timestamp` 最近邻、严格递增地重采样为
25 帧（0–2.4 秒、10 Hz）；禁止直接用 index 8–32 冒充 2.4 秒。proxy 与后续 C1B screen 均固定使用
C1B-00 选出的 256×448 分辨率。

在看生成结果前，将校准阈值与 95% null band 写入冻结协议。最低有效性要求：balanced accuracy ≥0.70、turn-sign accuracy ≥0.75、Spearman ρ ≥0.50；位移 MAE 必须同时优于 constant 与 command-only conditional-mean baseline。command 本身直接携带 left/right/forward 请求标签，因此 command-only 分类准确率只作 label-leakage sanity report，不能作为要求视频 proxy 超越的分类 gate。任一有效性要求不满足，机器 action 指标不可辨识，C1B 状态 `blocked`。不得用人工观感把未校准 proxy 变成 pass。

该 proxy 统一称为 **local ego-motion proxy**，禁止称作 ReSim IDM、Trajectory Difference 或 ADE。

## C1B-02：10-context 配对 action screen

### 6.6 对照臂

每个 moving context、同一 seed 生成：

- **E（expert-conditioned）**：`apply_traj=true`，使用该 clip 的 expert trajectory；
- **F（action-free）**：`apply_traj=false`，其余完全相同。

在最多 4 个 moving contexts 增加 **M（matched expert sensitivity）**：从同 command、同运动幅度区间、scene-disjoint 的另一条真实 expert trajectory 中选曲率不同者。M 只判断模型是否对支持内轨迹变化敏感，不进入 E-vs-F 主 pass，也不能被称为非专家/反事实物理 GT。

禁止把 random shuffle、倒序、跨 command 或 physically invalid trajectory 作为主对照。若 H1 已通过，可在单独标记的 exploratory run 中做 OOD stress test，但不得影响 v6 gate。

### 6.7 分轴指标

1. **历史保持**：source history 先经同一 VAE round-trip 得到可达基线；E/F 的 history fidelity 相对该基线报告，另报告 E-vs-F history 差。阈值来自 B00 重复 null，不再使用跨 backbone 的任意绝对 `PSNR ≥18 dB` 或 `LPIPS ≤0.15`。
2. **action response**：local ego-motion proxy 的 class、turn sign、displacement 与请求 trajectory 的一致性；报告 E 相对 F 的 paired improvement。
3. **非平凡分支效应**：future-only E/F 特征差必须显著大于 B00 同输入重复 null；history 差不得同步增大。
4. **视觉/运动 safeguard**：独立 CoTracker3、sharpness、flicker、survival、全局/actor motion ratio；所有阈值相对同场景 F 与 VAE/null 校准，不移植旧 SVD 的绝对阈值。
5. **actor 观察项**：只报告 track survival、速度/加速度统计与明显穿透/瞬移事件，不作为 H1 的 action pass，也不声称绝对 physics accuracy。

### 6.8 C1B 机器 gate

八个 moving contexts 同时满足：

- 至少 7/8 的 E 在预注册 action error 上优于 F，且 median paired improvement > 0；
- 至少 7/8 的 E/F future effect 超过重复 null，而 history effect 留在冻结 null band 内；
- E 不出现 F 没有的 catastrophic corruption，且 quality/motion safeguard 不被系统性破坏；
- 两个 stationary controls 均不超过真实 stationary false-motion 的冻结上界；
- M sensitivity 若执行，单独报告，不允许用 M 的好结果补 E-vs-F 的失败。

机器 gate 未过且 run 有效：H1 `rejected`，停止 C1P。

## C1B-03：action 盲审

机器 gate 通过后生成匿名 A/B review packet：8 个 moving contexts 的 E/F 随机左右、请求 trajectory 可视化、source history、未来视频、机器指标隐藏；另放 2 个重复/相同控制检查 reviewer 一致性。问题分开回答：

1. 哪个未来更符合请求的 ego trajectory：A / B / tie / unjudgeable？
2. 哪个视频更真实且无明显崩坏：A / B / tie / unjudgeable？
3. 是否存在 stationary shortcut、冻结、actor 瞬移/穿透或历史漂移？

agent 必须提供完整 review prompt、文件索引、随机化 key 的封存路径和回填格式，随后把任务设为 `blocked`。agent 不得自行填写 human verdict。

**human pass**：至少 6/8 action 项为 decisive，其中 E 获胜比例 ≥75%；质量项无 E 独有 catastrophic failure；两个相同控制不得产生相互矛盾的强偏好。
**fail**：H1 `rejected`，停止。
**pass 后动作**：才能进入 C1P。

---

# 7. Stage C1P — preference support，而不是先训练再找标签

## C1P-00：阈值、预算与 split 冻结

选择 16 个 scene-disjoint conditions，固定同一 source history、command 与 expert trajectory；与 C1B 的 10 个 screen contexts 及 proxy calibration scenes 不重 scene。按 command、actor 密度、ego motion 和可见性分层。C1S 的 train/held-out 将在这 16 个 conditions 内按 scene 再冻结划分。

在生成候选前完成：

1. 用 Base 的 exact-repeat、不同 seed baseline、VAE history round-trip 和真实 source future（只用于标定画质/运动分布，不作条件输入）冻结 duplicate、track survival、sharpness、flicker、global/actor motion、history drift 与 catastrophic 阈值。
2. 既往 B0 的 sharpness/flicker/motion/survival 阈值只作初始 sanity reference；因 backbone 已变更，不能不经校准直接搬用。
3. 固定候选预算为先 `N=4` seeds/condition；只有按下述唯一规则才扩到 `N=8`。硬上限为 128 个视频。
4. 冻结训练侧 scorer 与独立 evaluator：训练侧使用现有 RAFT/P-UNC 类局部物理分数；验证侧使用 CoTracker3/独立实现。二者不得共享 flow/cache/track 输出。
5. 冻结 strict/tie/incomparable 规则和所有 seed；生成后不得替换“难看”的 seed。

## C1P-01：自然候选池

对每个 condition 固定 expert trajectory，只改变预注册 natural seed。每个候选先过 eligibility：

- history 在 C1P-00 冻结 band 内；
- 无解码/黑屏/严重闪烁/冻结；
- actor 与全局运动量不落入 low-motion shortcut；
- 独立 tracker survival 达标；
- 两候选不是 feature/track duplicate。

一个 condition 有至少两个 eligible、non-duplicate candidates 才算 diverse。

**唯一扩样规则**：N=4 后若 diverse conditions <12/16，则所有未达标 condition 一次性补到 N=8；已经 diverse 的 condition 不补。N=8 后仍 <12/16，H2 `rejected`，停止。禁止 N=9、换 seed、改 duplicate 阈值或用 action siblings 填数。

## C1P-02：UPO 支持审计

仅在同 condition 的 natural rollout siblings 间建 pair。先应用 common-support mask，再进行 partial-order labeling：

- **strict**：两位候选均 eligible，目标 actor-motion 证据在训练 scorer 和独立 evaluator 上方向一致，差值超过冻结 null/margin，且 winner 没有以低运动、模糊、track loss 或画质退化取胜；
- **tie**：差异在 margin 内或多维证据等价；
- **incomparable**：维度冲突、support 不重合、任一候选不合格或证据不可判。

tie/incomparable 不得 forced binary，也不得当弱 strict。action E/F/M siblings 只用于 C1B，永不进入 preference training。

插入 exact duplicate、左右交换和冻结/低运动 synthetic controls，审计 false-strict 与方向一致性。

**机器 pass**：

- diverse conditions ≥12/16；
- strict pairs 覆盖 ≥10 个 conditions、≥10 个 scenes，而不是由少数场景的多个组合重复贡献；
- duplicate/tie control false-strict = 0；
- strict winner 的 catastrophic、low-motion shortcut 计数均为 0；
- 训练 scorer 与独立 evaluator 在 strict pairs 上方向一致率 ≥75%；
- 报告 strict/tie/incomparable 全分布和缺失支持机制，不只报告 strict 子集。

未过：H2 `rejected`，不进入任何训练。

## C1P-03：preference 盲审

从机器 strict 中按不同 scene/command 取 10 对；若机器 strict 覆盖超过 10 个 scene，则按预注册分层规则固定至最多 12 对。另加入 4 个 tie/duplicate controls；匿名左右，隐藏机器标签。人工分别判断：actor motion realism、整体质量、运动量捷径、历史漂移，以及 strict/tie/unjudgeable。

agent 到达此阶段时必须生成完整 prompt 与 packet，然后状态置 `blocked` 等用户回填。

**human pass**：

- 至少 10 对机器 strict 被人工确认同方向 strict，覆盖 ≥10 scenes；
- 对人工 decisive 的机器 strict，方向一致率 ≥75%；
- 低运动/冻结/模糊 winner = 0，catastrophic winner = 0；
- tie/duplicate controls 无 false-strict。

通过后，把人工确认的 strict pairs 按 scene 冻结为：6 train pairs + 至少 4 held-out pairs，两个 split 的 scene 完全不重合；若无法满足，H2 仍为 `rejected`。剩余 pair 不用于补看过的 held-out。

---

# 8. Stage C1S — 单卡 4090 学习闭环

## 8.1 参数边界

公开 checkpoint 自身已经包含 rank-128 LoRA 与 trajectory encoder。v6 禁止把这些原参数整体解冻后称作“轻量局部适配”。C1S-00 先做 module/parameter audit，输出完整参数表与 forward 位置。

唯一允许的训练方案：

- 冻结 ReSim backbone、已有 LoRA、trajectory encoder、VAE、T5 和所有原 checkpoint 参数；
- 在审计确认的最后两个 temporal/spatiotemporal mixing blocks 的 attention projection 上注入**新增、零初始化、rank-16 delta adapter**；
- 若无法在不改变 step-0 forward 的前提下注入并只训练新增参数，则 C1S `blocked`，不得退而解冻 rank-128 原 LoRA；
- step 0 的 Base 与 adapter-enabled 输出必须在数值重复容差内一致。

该小 scope 是单卡 screening 的工程折中，不宣称为最终最佳层选择。v6 不做 layer/rank sweep。

## C1S-00：一 optimizer step 与 OOM gate

固定配置：256×448、9 latent frames（33 RGB）、fp16、microbatch 1、gradient accumulation 4、activation checkpointing、AdamW、LR `1e-4`、weight decay `0.01`、grad clip `1.0`；只为新增 adapter 建 optimizer state。使用一对 train strict pair、冻结的 timestep/noise bank 和 common-support mask。

训练目标的结构固定为：

```text
L = L_pair_UPO
  + L_real_denoising_anchor
  + L_outside_support
  + L_history/action_preservation
```

各项先在 Base calibration batch 上按 detached robust scale 归一化，再以 1:1:1:1 合并；不得根据偏好结果调权。`L_pair_UPO` 只吃 strict direction，tie/incomparable 不产生方向梯度。禁止 full-chain video backprop、reward-model training、PPO/GRPO。

**pass**：一步 forward/backward/update 完成；峰值 VRAM <24,564 MiB；所有 loss/gradient/update finite；只有白名单新增参数变化；step-0 与保存/重载一致。
**fail**：若 OOM，可检查并修复未启用的预注册 checkpointing/冻结错误后重跑一次；仍 OOM 则 single-GPU hypothesis `rejected`。不得减少帧、改 rank/layer、换第二张卡。

## C1S-01：单对 capacity/locality diagnostic

只用一对 train strict pair，最多 200 optimizer updates。每 20 步在固定 noise/timestep bank 上评估：pair margin、real-denoising anchor、history、action proxy、support 内/外 drift 和独立 tracker。训练 seed、checkpoint interval 与 early-stop 规则在开始前冻结。

**capacity pass**：pair margin 方向正确并超过 C1P 冻结 margin，且在 ≤200 步内稳定出现；无 NaN/Inf。
**locality pass**：history/action 在 Base null band 内，outside-support drift 不超过 C1P-00 的 95% Base repeat band，无画质/低运动捷径。

任一未过：H3 `rejected`。通过也只表示“该参数化能学且暂未泄漏”，不能写成泛化结果。

## C1S-02：scene-disjoint 多对 pilot

从 C1P-03 冻结的 6 train strict pairs 训练，最多 400 optimizer updates；至少 4 个 held-out strict pairs 全程不可用于 early stopping、阈值或超参数选择。early stopping 只看 train objective 与预先独立的 anchor diagnostics，不看 held-out。

比较对象只有：

- frozen Base；
- C1S adapter 的预注册 final checkpoint（或仅依据 train/anchor 规则选出的 checkpoint）。

所有评估共享固定 seeds/noise、expert trajectory 和 exact manifest。报告：

- held-out strict win credit（win=1、tie=0.5、loss=0）；
- 独立 actor tracks 的运动连续性/加速度异常；
- history fidelity、action proxy、sharpness/flicker/survival、global/actor motion；
- support 内改善与 outside-support drift；
- train/held-out scene coverage 和所有失败个例。

**机器 pass**：至少 3/4 held-out pairs 对原 strict direction 有正向 win，mean win credit ≥0.75；没有 held-out catastrophic/low-motion winner；history/action/quality/outside-support 均通过冻结 non-regression gate。样本超过 4 对时，仍要求 ≥75% 正向且 bootstrap CI 原样报告。

## C1S-03：最终盲审

对所有 held-out 条件生成 Base vs adapter 匿名 A/B，加 duplicate controls；隐藏 checkpoint 与机器方向。问题分别覆盖 actor motion、整体真实感、action follow、低运动/冻结和历史漂移。

到达时 agent 必须生成完整 review prompt，状态置 `blocked`，等待用户。最终 human pass 要求：

- ≥75% decisive held-out comparisons 偏向 adapter；
- action-follow 与整体质量不劣于 Base；
- catastrophic、低运动捷径、历史漂移均为 0；
- duplicate controls 无 false-strict。

机器和人工都过，H3 才写 `done`；否则 `rejected`。

---

# 9. 任务 DAG 与停止规则

| Task ID | 任务 | 初始状态 | 依赖 | 通过后 |
|---|---|---|---|---|
| C1A-00 | 资产、数据根与磁盘对账 | `done` | — | C1A-01 |
| C1A-01 | 独立环境与 imports | `done` | C1A-00 | C1B-00 |
| C1B-00 | smoke、shape、VRAM、确定性 | `done` | C1A-01 | C1B-01 |
| C1B-01 | 10-context 冻结与 proxy calibration | `blocked` | C1B-00 | C1B-02（需新预注册 proxy 通过后） |
| C1B-02 | E vs F action screen | `pending` | C1B-01 | C1B-03 |
| C1B-03 | action 人工盲审 | `pending` | C1B-02 | C1P-00 |
| C1P-00 | 阈值、split、预算预注册 | `pending` | C1B-03 | C1P-01 |
| C1P-01 | N=4→条件式 N=8 候选池 | `pending` | C1P-00 | C1P-02 |
| C1P-02 | UPO 机器支持审计 | `pending` | C1P-01 | C1P-03 |
| C1P-03 | preference 人工盲审 | `pending` | C1P-02 | C1S-00 |
| C1S-00 | 新增 adapter 审计与一步训练 | `pending` | C1P-03 | C1S-01 |
| C1S-01 | 单对 capacity/locality | `pending` | C1S-00 | C1S-02 |
| C1S-02 | 多对 scene-disjoint pilot | `pending` | C1S-01 | C1S-03 |
| C1S-03 | 最终 held-out 人工盲审 | `pending` | C1S-02 | C1D-00 |
| C1D-00 | 汇总与 go/no-go | `pending` | C1S-03 或任一终止 gate | v7 或关闭路线 |

严格停止规则：

```text
C1A 工程不可用        → blocked / engineering report
C1B action 不可辨识    → blocked
C1B 有效但不过 gate    → H1 rejected，停止
C1P diversity/support 失败 → H2 rejected，停止
C1S 单卡 OOM/容量/局部性失败 → H3 rejected，停止
任一人工 review 未回填 → blocked
只有 H1 + H2 + H3 全过 → v6 done，允许起草 v7
```

---

# 10. C1D-00 交付物与 reviewer attack checklist

每个终止点都要更新：

- `docs/EXPERIMENTS.md`：命令、run path、原始数字、异常与 artifact；
- `docs/RESEARCH_STATUS.md`：当前 task、gate verdict、下一步；
- `docs/RESEARCH_FAILURES.md`：仅在形成可复用负结论时追加，不复制流水账；
- `runs/.../RUN_PROVENANCE.md` 与 checksum；
- 一份 `C1_V6_FINAL_REPORT.md`，明确是 engineering fail、blocked/inconclusive、H1/H2/H3 rejected 或 screening pass。

最终报告必须逐条回答 reviewer 可能的质疑：

1. 公开 checkpoint 没有 CARLA，证据是否严格限制在 expert/action-mask 支持内？
2. action response 是否只是 seed noise、history drift 或 camera motion proxy 偏差？
3. actor physics 没有未来 GT，为什么没有夸大成绝对准确率？
4. preference support 是否 MNAR，只覆盖低运动/易场景？strict/tie/incomparable 全分布是什么？
5. scorer 与 evaluator 是否共享 tracker/flow 导致自证？
6. winner 是否靠冻结、模糊、track loss 或低运动取胜？
7. adapter 是否泄漏到 history、action、背景和 common-support 之外？
8. single-pair capacity 是否被误写成 generalization？
9. held-out 是否 scene-disjoint，是否被用于 early stop 或阈值选择？
10. 单卡内存阶梯、失败 run、人工 ties 和所有排除样本是否完整披露？

若 v6 全过，v7 才讨论：扩大到至少 32+ scene-disjoint 场景、完整 4 秒 future（13 latent / 49 RGB）、更大人工池、正式 Base/SFT/DPO/UPO 消融、多视角/其他 backbone、统计功效和多卡扩展。上述内容不属于 v6 成功定义，也不能用来挽救 v6 失败。

---

# 11. 清空 context 后的启动清单

新 agent 必须按顺序执行，不能只读本文后直接跑训练：

1. 完整阅读仓库 `AGENTS.md`、`RESEARCH_STATUS.md`、`RESEARCH_FAILURES.md`、`EXPERIMENTS.md`、本文及本文引用的 V5/C0 报告。
2. 检查 `git status`、motion_proj/ReSim HEAD、GPU、磁盘、正在运行的下载/训练进程；不得覆盖用户未提交变更。
3. 重新核验第 1.2 节快照，因为资产和磁盘是时变状态；把差异写入 C1A-00 证据。
4. 先处理 C1A-00 的阻塞。若需删除，向用户给出精确清单并等待授权。
5. 每次开始 task 前将其状态改为 `running`；完成后只写 `done`、`blocked` 或 `rejected`，并同步三份事实源。
6. 任何正式 run 前冻结协议、commit 必要代码、建立 provenance；任何人工 gate 都生成完整 prompt 并停止等待。
7. 严格执行 gate。不得因为已经投入算力而放宽标准，不得启用第二张 GPU。

本计划的价值不在于保证正结果，而在于用一张 4090 以最短闭环区分：**ReSim action 接口不可用、偏好支持不存在、局部学习不可行，或 idea 确实值得扩大。**
