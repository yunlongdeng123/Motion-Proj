# Motion-Proj 路线切换 Autoresearch 计划 v5（AC 执行版）

> **工作标题**：Motion-Proj Route Pivot — Real-Motion Representation vs. Natural-Rollout Alignment
> **目标投稿**：CVPR 2027
> **文档职责**：当前唯一可执行研究计划；当前状态与授权同步到 `docs/RESEARCH_STATUS.md`
> **最后更新**：2026-07-18
> **计划基线**：`5e8e8d747aa42334204c5e58c49ba0ae96c74b55`
> **计划状态**：`running`（当前阶段 `RP-A1-SCAN-04A`）
> **状态词**：`pending / running / awaiting_reviews / blocked / done / rejected`
> **当前路线状态**：SVD 内部去噪扰动 sibling preference 路线已 `rejected`；禁止继续搜索 fork、\(\rho\)、candidate 数或旧 DPO 标签器。
> **当前硬件**：单张 RTX 4090 24 GB；本计划全部 autoresearch gate 默认单卡完成。
> **远端仓库预期路径**：`/root/autodl-tmp/motion_proj`，执行时必须自行确认。
> **核心执行要求**：连续完成多个独立研究 gate，不因一个中间失败或一次 smoke 通过而停下；除硬阻塞外不向用户请求确认。
> **当前主候选路线**：
>
> 1. **Route A：真实驾驶视频上的 Ego–Actor Motion Representation Alignment**；
> 2. **Route B：自然 rollout 的 best-of-N / reward-weighted self-training 可行性**。
>
> **候选路线 C**：只有 A、B 均失败时，做 action/trajectory-conditioned driving world model 的迁移审计，不在本轮训练新 backbone。

---

# 0. 给 Codex Agent 的执行契约

你现在接手 Motion-Proj 的路线切换 autoresearch。当前对话可能没有此前上下文，必须以：

```text
远端 Git 仓库
docs/RESEARCH_STATUS.md
docs/RESEARCH_FAILURES.md
docs/EXPERIMENTS.md
docs/archive/2026-07/PHYSICS_DPO_AUTORESEARCH_PLAN_V4_AC_REVISED.md
正式 run manifest / summary
当前代码
本文
```

为事实源。权威顺序固定为：正式 run 原始证据 → `EXPERIMENTS.md` → `RESEARCH_STATUS.md` →
`RESEARCH_FAILURES.md` → 本计划的执行协议 → `docs/archive/` 历史材料。归档文件不得恢复旧任务。

本计划负责“怎样执行”；`RESEARCH_STATUS.md` 负责“当前允许执行到哪里”。每个 gate 完成后必须同步两者，
并把正式结果追加到 `EXPERIMENTS.md`。若本计划正文中的未来动作与更新后的状态页冲突，以状态页为准。

## 0.1 不允许停在“分析建议”

本任务不是只写文献综述或方案建议。

在资源和安全边界允许时，必须连续完成：

```text
R0 仓库/事实/环境基线
R1 时间采样与 SVD conditioning 审计
A0 真实 ego–actor motion target 合法性
A1 冻结 SVD motion-feature probe
B0 natural rollout best-of-N 上限诊断
D0 路线决策
```

若 A1 通过，还必须继续完成：

```text
A2 小规模 auxiliary-alignment capacity test
```

不要在：

- 代码审计完成；
- 文献检索完成；
- 一个单元测试通过；
- 一次 smoke 通过；
- 一个 gate 失败；

之后停下。

某一条路线失败时，继续执行与它独立的其他路线。

## 0.2 过程更新

可以发送简短进度更新，但：

- 不询问“是否继续”；
- 不请求用户选择 A/B；
- 不因人工 review 尚未完成而停止机器侧独立任务；
- 不给未来工作时间承诺；
- 不等待用户回复后才继续。

只有以下硬阻塞可提前结束：

1. 远端 worktree 存在无法归因的用户改动；
2. 核心权重或 nuScenes 数据缺失，且本机/共享盘均不存在；
3. 发现 future-GT 泄漏进入 generated-rollout 正式评价；
4. 单卡 OOM 且无法在预注册缩减方案内运行；
5. 关键代码事实与本文严重冲突，无法在不破坏历史的情况下继续；
6. 所有候选路线均触发预注册 rejection；
7. 环境或数据损坏导致正式证据无法建立。

若 human review 是唯一未决项：

```text
状态写 awaiting_reviews
继续完成其他独立 gate
最后一次性汇报 review material 和机器结论
```

## 0.3 禁止事项

本轮禁止：

- 继续 SVD denoising-prefix perturbation sibling 生成；
- 搜索新的 fork fraction、\(\rho\)、CFG branch；
- 使用旧 53 pair 或旧 local labels 训练；
- 实现旧 DrivePO tube-DPO；
- 训练大型 reward model；
- 完整 25-step sampling-chain 反传；
- 在线 PPO/GRPO；
- 切换双卡；
- 重建或覆盖历史 run；
- 自动 push；
- 将训练 scorer 同时作为唯一正式 evaluator；
- 用 future ego/box/track 评价自由生成 rollout；
- 将 image-plane acceleration 称为真实世界加速度。

---

# 1. 当前已确认的历史结论

## 1.1 Endpoint 路线

已否定：

```text
Base rollout
→ trajectory correction
→ RGB crop/resize/paste
→ VAE/hybrid latent
→ shared temporal LoRA endpoint regression
```

关键问题：

- counterfactual RGB target 不合法；
- 亚像素 correction 可在整数 compositor 中变成零 RGB 变化；
- source removal、disocclusion、depth order 不成立；
- shared temporal LoRA 无局部 parameter subspace。

## 1.2 Sibling preference 路线

已否定：

```text
同一 SVD 去噪前缀
→ latent perturbation siblings
→ automatic physics preference
→ DPO
```

正式证据：

- 24 个 P1 sibling 人审中 22 tie、0 decisive；
- 修复 common support 和 false-strict 后，96 个未审 condition 只有 2 strict、94 incomparable；
- 唯一 earlier-fork fallback 为 1 strict、0 tie、7 incomparable；
- earlier fork 还出现 first-frame 偏差和 temporal-jump quality failure；
- 路线已按预注册门禁 `rejected`。

必须在新文档中明确：

> 被拒绝的是 SVD 内部扰动造 preference candidate 的机制，不是所有 motion representation learning 或所有 reward alignment。

## 1.3 可复用资产

| 资产 | 可复用范围 |
|---|---|
| `svd_official_v1` | 所有新生成和 conditioning 的唯一协议 |
| `NuScenesFutureVideoDataset` | 真实视频、内参、cam2ego、ego2global、instance_token、2D boxes、稀疏 LiDAR depth |
| `compute_ego_flow` / geometry utils | 真实训练视频的 ego-induced image motion |
| `NuScenesTrackProvider` | 按 instance_token 关联真实标注轨迹 |
| RAFT | 真实视频 flow baseline、生成视频训练侧 scorer |
| CoTracker3 E0 v3 | 独立 generated-rollout evaluator |
| SVD raw-\(v\) / \(x_0\) API | 新辅助训练与正式生成 |
| 7 个 SVD feature hook | 冻结表示审计 |
| temporal-only rank-16 LoRA | 第一版轻量训练参数 |
| manifest/fingerprint/review/runtime | 所有新 gate 的可追溯基础设施 |
| sibling/UPO 代码 | 负结果、common-support 工具；不得作为新训练数据 |

## 1.4 与 research 踩坑账本的防重复映射

本计划不覆盖 `docs/RESEARCH_FAILURES.md`。各路线必须显式关闭以下风险：

| 路线 | 必须回应的条目 | 本计划中的控制 |
|---|---|---|
| R1 | `RF-02`、`RF-08`、`UR-06` | 区分真实时间尺度、SVD micro-conditioning 与 evaluator nuisance |
| A0 | `RF-02`–`RF-05`、`UR-01`、`UR-03` | 只在真实训练视频用 GT；generated evaluator 禁止 future-GT；先做 target legality |
| A1 | `RF-07`、`UR-01` | 完整 ego/actor signal、scene-disjoint holdout、shuffle 与 single-frame controls |
| A2 | `RF-01`、`RF-06`、`UR-02`、`UR-05` | real loss、gradient audit、locality/first-frame 与 anti-collapse capacity gate |
| B0 | `RF-04`、`RF-08`–`RF-12`、`UR-02`–`UR-04` | 自然独立 seeds、独立 evaluator、共同 condition、human gate 与 low-motion 拒绝 |
| Route C | `UR-06` | 只读审计显式 action/trajectory-conditioned backbone，不偷换 SVD 结论 |

任何 gate 若只能通过降低旧阈值、复用旧 labels、减少运动、丢弃困难 support 或事后筛样本，应按对应
`RF-*` 条目 fail closed，而不是修改本计划追求通过。

---

# 2. 文献核查要求

如果环境联网，先对以下一手来源做定向核查，禁止只依赖博客：

| 工作 | 需要核查的点 |
|---|---|
| Track4Gen，CVPR 2025 | feature layer、point correlation、refiner、zero-conv、训练模块与计算成本 |
| MoAlign，ICLR 2026 | motion-specific subspace、flow-predictive teacher、relation alignment 与训练细节 |
| Geometry Forcing，2025/2026 | intermediate representation alignment、angular/scale loss、是否修改 inference |
| SHIFT，2026 | natural rollout reward、AWR/SFT hybrid、noise alignment、motion collapse |
| EOT-WM，2025 | ego 与 other-vehicle trajectory 的表示；与本项目“训练侧监督、不加 inference condition”的区别 |
| OpenDWM | nuScenes 支持、模型规模、预训练权重、单/双 4090 适配性 |
| DrivingGen，2026 | driving-world-model evaluation 的 trajectory plausibility、agent consistency 与视觉指标 |
| PhysAlign / 相关 physics representation alignment | synthetic physics supervision 与 feature alignment 的边界 |

优先来源：

```text
CVF Open Access
OpenReview
arXiv 原文
官方项目页
官方 GitHub
```

输出：

```text
docs/ROUTE_PIVOT_LITERATURE_MATRIX.md
```

表格至少包括：

```text
Paper
Venue/year
Backbone
Training data
Motion/geometry teacher
Supervision layer
Trainable modules
Inference conditions
Compute
Direct overlap
Remaining novelty
Official code availability
```

不得因为新论文与本计划接近就忽略；相反，必须主动收紧创新边界。

---

# 3. 新路线总览

```text
旧路线归档
       │
       ├── R1：真实视频时间采样 / fps / conditioning audit
       │
       ├── Route A
       │      A0：真实 ego–actor target validity
       │       ↓
       │      A1：frozen SVD motion-feature probe
       │       ↓ pass
       │      A2：小规模 auxiliary alignment capacity
       │
       └── Route B
              B0：natural independent-rollout best-of-N ceiling
               ↓ human + machine pass
              B1：AWR/SFT 计划，不在本轮自动长训

A/B 证据汇总
       ↓
D0：选择主线 / fallback / 停止 SVD
```

稳定任务 ID 与依赖如下：

| ID | 当前状态 | Gate | 依赖 |
|---|---|---|---|
| `RP-R0-00` | done | 仓库、环境、资产、文档基线 | 无 |
| `RP-LIT-01` | done | 一手文献与创新边界矩阵 | R0 |
| `RP-R1-02` | done | 时间采样与 SVD fps audit | R0 |
| `RP-A0-03` | awaiting_reviews | 真实 ego–actor target legality | R0 |
| `RP-A1-SCAN-04A` | running | frozen SVD feature scan | A0 machine evidence |
| `RP-A1-CONFIRM-04B` | pending | scene-disjoint confirm | A1-SCAN 有合法候选 |
| `RP-B0-05` | pending | natural-rollout best-of-N ceiling | R0；与 A 路线独立 |
| `RP-A2-06` | pending | auxiliary-alignment capacity | A1-CONFIRM pass |
| `RP-C0-07` | pending | action-conditioned backbone 迁移审计 | A 与 B 均 rejected |
| `RP-D0-08` | pending | 路线决策与最终报告 | 所有已解锁 gate 结束 |

阶段状态只按正式门禁更新；工程失败使用新 run ID 修复，不把任务直接标为 research `rejected`。

Route A 和 Route B 在数据构造与机器诊断上独立。不要因为其中一条失败而停止另一条。

---

# 4. R0 — 仓库、环境与历史归档

## 4.1 开始前

通过指定远端连接后执行：

```bash
pwd
git status --short
git branch --show-current
git rev-parse HEAD
git log -10 --oneline
git remote -v
nvidia-smi
df -h /root/autodl-tmp
ps aux | grep -E 'python|train|motion_proj' | grep -v grep
```

读取：

```text
AGENTS.md
docs/RESEARCH_STATUS.md
docs/RESEARCH_FAILURES.md
docs/EXPERIMENTS.md
docs/archive/2026-07/PHYSICS_DPO_AUTORESEARCH_PLAN_V4_AC_REVISED.md
docs/archive/2026-07/AUTORESEARCH_RETROSPECTIVE_2026-07.md
docs/archive/2026-07/AUTORESEARCH_ROUTE_DECISION.md
configs/data/nuscenes_trainval.yaml
configs/model/svd.yaml
```

激活环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/motionproj
export PYTHONPATH=.
```

运行 baseline：

```bash
PYTHONPATH=. pytest -q
```

最近一次 V4 收口报告为 208 passed，但必须记录当前实际结果。

## 4.2 资产检查

确认：

```text
SVD-XT 权重
nuScenes trainval metadata
CAM_FRONT keyframes
LIDAR_TOP keyframes
RAFT 权重
Depth-Anything 权重或本地 cache
CoTracker3 repo commit + checkpoint SHA
现有 scene split manifest
可用磁盘
```

只做存在性与 fingerprint 检查，不自动下载大型新模型。

## 4.3 归档新路线决定

当前计划文件：

```text
docs/MOTION_ROUTE_PIVOT_AUTORESEARCH_PLAN_V5.md
```

更新：

```text
docs/RESEARCH_STATUS.md
docs/EXPERIMENTS.md
```

明确：

```text
DrivePO / SVD internal sibling route = rejected
旧 PA3–PA8 = rejected / not run by dependency
当前新问题 = real-motion representation vs natural-rollout alignment
```

建议提交：

```text
docs(research): 归档 sibling 路线并启动真实运动表征研究
```

---

# 5. R1 — 时间采样与 SVD conditioning 审计

这是新路线的首个必要 gate。

## 5.0 执行结论（2026-07-18，done）

正式 run `route-pivot-r1-temporal-s20260718-v1` 已在 clean commit `f4b4cd5` 完成：

- 32 个 scene-distinct 真实 clips 的中位 timestamp delta 为 `0.5000 s`，有效 fps 为 `2.0000 Hz`，
  8 帧中位覆盖 `3.5000 s`；
- 8 conditions × 3 fps × 2 seeds 共 48/48 生成与打分有效，16/16 paired groups 有效，
  same-initial-noise trace 通过且生成/评估均不使用 future GT；
- 相对 `fps=7`，`fps=2/4` 的 dynamic degree 分别增加 `24.74%/10.05%`，image velocity
  分别增加 `77.97%/110.71%`，paired bootstrap 均显著；
- 但 `fps=2` 的首帧、锐度、闪烁、track survival 与 acceleration safeguard 全部失败；`fps=4`
  的 survival ratio `0.859 < 0.90`、acceleration p95 ratio `1.950 > 1.25`，同样不合格；
- 冻结 V5 后续生成协议为 `generation.fps=7`。这不表示 7 与真实采样率一致，而是否定“直接降低
  SVD fps micro-conditioning 就能安全修复时间 mismatch”。

32 个 blinded pairs 已生成且 verdict 模板保持空值，作为 evaluator nuisance 的补充诊断；不阻塞独立的
A0 machine gate。完整协议、reviewer 质疑与证据见
[`ROUTE_PIVOT_TEMPORAL_AUDIT.md`](ROUTE_PIVOT_TEMPORAL_AUDIT.md)。

当前代码事实：

- nuScenes keyframes 约 2 Hz；
- `frame_stride=1`；
- 8 帧约覆盖 3.5 秒；
- 14 帧约覆盖 6.5 秒；
- SVD generation 默认 `fps=7`；
- C0 只证明实现与官方 pipeline 一致，不证明 `fps=7` 与训练数据的真实采样间隔匹配。

## 5.1 研究问题

> 当前真实训练视频的时间尺度与 SVD micro-conditioning 是否存在系统性 mismatch？这一 mismatch 是否足以影响 Base motion 和后续 motion supervision？

## 5.2 实现

新增：

```text
motion_proj/diagnostics/temporal_sampling_audit.py
configs/diagnostics/route_pivot_temporal_sampling.yaml
tests/test_temporal_sampling_audit.py
```

对官方 train split 抽取 32 clips，统计：

```text
timestamp delta
effective fps
clip duration
ego translation / rotation per frame
box center displacement per second
valid actor track length
```

必须使用真实 `timestamps`，所有速度/加速度以后按实际 \(\Delta t\) 计算。

## 5.3 Base generation 对照

固定 8 个 validation conditions、相同 initial noise，比较：

```text
fps input = 2
fps input = 4
fps input = 7
```

其他参数：

```text
svd_official_v1
25 steps
8 frames
同 motion_bucket_id
同 guidance
同 noise_aug_strength
2 generation seeds
```

评价：

- CoTracker track survival；
- dynamic degree；
- image-plane velocity；
- acceleration outlier；
- first-frame fidelity；
- visual quality；
- pairwise human review material。

不使用 GT future 选择最佳生成。

## 5.4 决策

若 `fps=2` 或 `fps=4` 在不损害质量下显著改变 motion：

- 后续训练/生成必须将 `generation.fps` 版本化；
- 选值只能依据真实采样一致性和 Base diagnostics，不依据训练结果；
- 旧 V5/DrivePO run 保持历史协议，不重写。

若三档差异很小，保持 7 并记录。

输出：

```text
runs/route-pivot-temporal-<unique-id>/
docs/ROUTE_PIVOT_TEMPORAL_AUDIT.md
```

---

# 6. Route A — 真实 Ego–Actor Motion Representation Alignment

## 6.1 精确研究假设

视频扩散模型的 motion failure 可能不是缺少一个可回归的 RGB target，而是内部表示没有显式区分：

```text
由相机自车运动产生的全局视差
vs.
由周围交通参与者运动产生的 residual motion
```

目标不是给模型增加 future trajectory condition，而是：

> 在真实训练视频上，用标定几何和 instance annotations 对 SVD 的中间表示提供 training-only auxiliary supervision；推理输入保持第一帧图像不变。

工作名称暂定：

```text
EgoActor-Align
```

只有 A2 rollout 结果成立后才允许作为正式方法名。

---

# 7. A0 — 真实运动 target 合法性

## 7.0 执行结论（2026-07-18，machine pass / awaiting reviews）

当前事实源为 `route-pivot-a0-real-motion-s20260718-v3`（clean commit `45cb279`）：

- 修复了 `min_box_visibility` 读取但未应用的真实 bug，并以 additive schema 加入 annotation token、
  attributes、camera-frame center/corners、global velocity 与逐帧 calibration；
- 16 个 scene-distinct clips 上 420/421 actor pairs finite；进一步要求 actual_t、actual_t+1 和
  static-if-world-fixed_t+1 都在图内后，仍有 392/421 pairs、89 条 unique tracks；
- localizable support 中 moving/stationary pairs 为 181/208，residual AUC `0.8600`；velocity direction
  positive fraction `0.9725`；residual 与 ego speed 的 Spearman `0.2226`；
- 157,394 个 confident GT-box 外 LiDAR 点上，sparse ego flow 与 RAFT 夹角不超过 45° 的比例为
  `0.9870`；schema、visibility、projection、calibration、LiDAR 与 leakage checks 全过；
- v1 曾把 34 个部分可见但 center 已出画的 observations 错纳入 clipped-xyxy 分母；34/34 failures
  均出画、图内 failure 为 0。v1 原样保留为 checker bug，v2 修正 eligibility，v3 再前置可局部化共同支持。

因此 A0 machine evidence 解锁 A1-SCAN；12 个 review panels 和空 verdict 模板已生成，最终 Route A
promotion 仍受 human gate 约束。完整边界与证据见
[`ROUTE_PIVOT_REAL_MOTION_TARGET_AUDIT.md`](ROUTE_PIVOT_REAL_MOTION_TARGET_AUDIT.md)。

## 7.1 先审计当前数据代码

必须核实：

- `min_box_visibility` 当前是否实际过滤；
- `boxes` 是否只保存 xyxy/depth，而没有 camera-frame center/corners；
- `instance_token` 跨帧关联是否完整；
- annotation attributes 是否可读取；
- `timestamps` 是否为每个 camera keyframe 的真实微秒；
- LiDAR depth coverage；
- `cam2ego` 是否在 clip 内恒定；
- camera ego pose 是否逐帧对应。

当前代码若未应用 `min_box_visibility`，应将其视为真实 bug并修复，但要保证 schema 向后兼容。

## 7.2 Additive box schema

在不破坏旧 cache 的前提下，真实数据 box 建议新增：

```text
annotation_token
instance_token
category
attributes
visibility
xyxy
center_cam             # [3]
corners_cam            # [8,3]
center_depth
size3d
velocity_global        # [3]，若 devkit 提供
```

旧字段保持。

新增测试：

```text
tests/test_nuscenes_motion_schema.py
tests/test_box_visibility_filter.py
tests/test_instance_track_continuity.py
tests/test_timestamp_units.py
```

## 7.3 Actor residual 的几何定义

对实例 \(i\)、相邻帧 \(t,t+1\)：

实际投影中心：

\[
p^i_t=\pi(KX^i_t),
\qquad
p^i_{t+1}=\pi(KX^i_{t+1}).
\]

假设该 actor 在世界坐标静止时的预测位置：

\[
\tilde X^i_{t+1}
=
T_{\mathrm{cam}_{t+1}\leftarrow\mathrm{cam}_t}
X^i_t,
\]

\[
\tilde p^i_{t+1}=\pi(K\tilde X^i_{t+1}).
\]

actor residual displacement：

\[
r^i_t
=
p^i_{t+1}-\tilde p^i_{t+1}.
\]

time-normalized residual velocity：

\[
v^i_t=\frac{r^i_t}{\Delta t_t}.
\]

所有 target 必须：

- 使用真实训练视频；
- 使用真实 annotation；
- 只在训练/representation probe 中使用；
- 不进入 generated-rollout evaluator；
- 不被描述为 future condition；
- 不被描述为真实物理加速度。

## 7.4 Background ego supervision

第一版 primary background target 只使用：

```text
真实投影 LiDAR 点
+ intrinsics
+ cam2ego
+ ego2global
```

计算 sparse ego-induced flow。

Depth-Anything + LiDAR scale calibration 只能作为 dense-proxy ablation，不作为 A0 primary truth。

原因：

- 减少单目 depth teacher 噪声；
- 保持几何 target 的可解释性；
- 避免将 depth foundation model 的误差误认成 SVD motion failure。

## 7.5 Actor strata

只使用可运动类别：

```text
vehicle.*
human.pedestrian.*
cycle.*
```

记录 nuScenes attributes：

```text
vehicle.moving
vehicle.parked
vehicle.stopped
pedestrian.moving
...
```

A0 不把 attribute 当生成条件，只用于 sanity/noise-floor 分析。

## 7.6 A0 machine sanity

至少使用：

```text
16 clips
来自不同 scenes
8 frames
真实 timestamp
```

检查：

1. box center 投影回 xyxy 内；
2. actor track continuity；
3. frame pair \(\Delta t\) 合法；
4. static expected projection finite；
5. moving actor residual显著大于 parked/stopped actor；
6. `velocity_global` 投影方向与 residual direction 基本一致；
7. sparse ego flow 与 RAFT 在 GT-box 外背景区域方向基本一致；
8. actor residual 不由 ego motion magnitude单独决定；
9. visibility/occlusion截断正确；
10. 无 source-future 信息进入 generated evaluator。

建议指标：

```text
moving vs stationary residual AUC >= 0.75
box-center projection-in-box >= 0.98
valid paired actor tracks >= 32
background ego-vs-RAFT angular agreement >= 0.70
finite target fraction >= 0.95
```

阈值须在正式 run 前写入 YAML；如当前数据分布不支持，允许基于 synthetic unit scale调整一次，但必须记录。

## 7.7 Synthetic geometry tests

构造：

- 纯 ego translation；
- 纯 yaw rotation；
- 世界静止 actor；
- actor 横向移动；
- actor 纵向移动；
- parked actor；
- visibility断裂；
- behind-camera；
- variable \(\Delta t\)。

验证 actor residual 和 ego flow 符号、尺度与单位。

## 7.8 A0 人工 audit

输出 12 个 panel：

```text
RGB frames
3D box projection
actual center track
static-if-world-fixed track
actor residual arrows
sparse ego flow
RAFT flow
category/attribute/visibility
timestamps
```

生成完整 review prompt 和 `reviews.template.jsonl`。

如果 review 未完成：

```text
A0 = machine_pass / awaiting_reviews
```

不要等待；继续 A1 machine probe，但最终 promotion 保留 human gate。

## 7.9 代码与产物

新增：

```text
motion_proj/data/real_motion_targets.py
motion_proj/diagnostics/real_motion_target_audit.py
configs/diagnostics/route_pivot_real_motion_targets.yaml

tests/test_real_motion_targets.py
tests/test_ego_actor_residual.py
tests/test_real_motion_no_eval_leakage.py
```

run：

```text
runs/route-pivot-a0-real-motion-<unique-id>/
```

---

# 8. A1 — Frozen SVD Motion-Feature Probe

## 8.1 为什么旧 F1 不适用

旧 F1 测试的是：

```text
projected track correction - observed track
```

该 correction 常低于 1 px。

A1 测试的是：

```text
完整 ego-induced displacement
完整 actor residual displacement/velocity
```

信号尺度不同。不得引用旧 F1 直接否定 A1。

## 8.2 Probe 目标

回答：

1. 哪个 SVD layer/噪声尺度包含 ego motion；
2. 哪个 layer包含 actor correspondence；
3. ego-centered residual parameterization 是否比 absolute flow 更易学习；
4. motion signal 是否来自 temporal representation，而不是单帧 appearance；
5. 是否存在跨 scene holdout 泛化。

## 8.3 Feature layers

复用已经验证的 7 个 hook：

```text
down_s8
down_s16
down_s32
mid_s64
up_s32
up_s16
up_s8
```

sigma 初扫：

```text
0.05
0.2
1.0
```

输入使用真实视频 latent 和 official conditioning。

## 8.4 不缓存完整 feature map

为节省磁盘，只保存 sampled feature records：

```text
sample_id
scene_id
layer
sigma
query_type
query_position
feature_vector_t
local_target_features_or_cost_window
ego_expected_position
actual_position
actor_residual
dt
category
attribute
visibility
validity
```

必须 fingerprint：

- backbone；
- layer paths；
- sigma；
- dataset split；
- target builder；
- query set。

## 8.5 两阶段 probe

### A1-SCAN

```text
24 train clips
8 dev clips
7 layers × 3 sigmas
```

只训练极小 probe，筛选 top-2 layer/sigma。

### A1-CONFIRM

```text
64 train clips
16 dev clips
16 scene-disjoint holdout clips
top-2 configurations
```

不得用 holdout 选 layer 或 threshold。

## 8.6 Probe tasks

### Task E：Sparse ego-flow regression

在 GT box 外 sparse LiDAR anchor 上：

\[
\hat u^{ego}=h_{ego}(F_t(x)).
\]

指标：

```text
EPE
angular error
relative EPE
valid coverage
```

### Task A-ABS：Absolute actor next-position

从 source actor feature预测：

\[
p_{t+1}.
\]

作为 generic tracking baseline。

### Task A-RES：Ego-centered actor residual

先用几何得到 static expected position：

\[
\tilde p_{t+1},
\]

再在其附近提取 local correlation/cost volume，预测：

\[
r_t=p_{t+1}-\tilde p_{t+1}.
\]

这是核心 driving-specific probe。

### Control

必须包括：

```text
zero residual
mean residual per category
time-shuffled feature
frame-order shuffled feature
single-frame spatial feature
instance-token shuffled target
```

## 8.7 Probe 模型

只允许：

- linear head；
- two-layer MLP；
- 小型 local cost-volume head。

禁止在 A1 训练 diffusion backbone、LoRA、refiner 或 zero-conv。

所有模型参数与训练 steps匹配。

## 8.8 A1 gate

A1 通过必须同时满足：

1. ego-flow probe 在 holdout 上比 zero/mean baseline EPE 至少改善 20%；
2. moving actor A-RES 在 holdout 上比 zero-residual baseline至少改善 15%；
3. A-RES 比相同容量的 A-ABS normalized error至少改善 10%；
4. time-shuffled/frame-shuffled control明显变差；
5. parked/stopped actor 不被系统性预测为大 residual；
6. top layer在两个 sigma附近结论稳定；
7. 结果跨 scene，而非仅 train；
8. compact feature cache和probe完全可复现。

若只 ego 通过、actor失败：

```text
Route A actor hypothesis rejected
可保留 ego-only representation baseline
不得包装为 EgoActor-Align
```

若 actor通过、ego弱：

- 检查 sparse depth coverage；
- 允许 dense calibrated-depth ablation一次；
- 仍弱则只保留 actor-residual路线。

若两者都失败：

```text
Route A rejected
```

## 8.9 代码与产物

新增：

```text
motion_proj/diagnostics/motion_feature_probe.py
motion_proj/data/motion_feature_records.py
configs/diagnostics/route_pivot_motion_feature_probe.yaml

tests/test_motion_feature_sampling.py
tests/test_actor_residual_probe.py
tests/test_probe_scene_split.py
tests/test_probe_controls.py
```

run：

```text
runs/route-pivot-a1-feature-scan-<id>/
runs/route-pivot-a1-feature-confirm-<id>/
```

---

# 9. A2 — Auxiliary Alignment Capacity Test（条件执行）

只有 A1 通过才执行。

## 9.1 目标

验证：

> 真实 ego/actor auxiliary loss 能否更新 SVD temporal LoRA，使 held-out representation 和少量完整 rollout 指标方向改善，而不破坏 real denoising、first frame和运动量。

这不是正式训练，只是单卡 feasibility。

## 9.2 第一版架构

固定：

```text
SVD VAE frozen
image encoder frozen
UNet base frozen
temporal-only rank-16 LoRA trainable
ego head trainable
actor residual head trainable
```

不增加 inference condition。

暂不加入 zero-conv/refiner；只有 auxiliary gradient完全无法影响 rollout时才作为未来 fallback，不在本轮自动扩方法。

## 9.3 Loss

真实视频 denoising：

\[
\mathcal L_{real}.
\]

ego sparse flow：

\[
\mathcal L_{ego}
=
\frac{\sum w\,
\mathrm{SmoothL1}(\hat u^{ego},u^{ego})}
{\sum w+\epsilon}.
\]

actor residual：

\[
\mathcal L_{actor}
=
\frac{\sum w\,
\mathrm{SmoothL1}(\hat r,r)}
{\sum w+\epsilon}.
\]

总损失：

\[
\mathcal L
=
\mathcal L_{real}
+
\lambda_e\mathcal L_{ego}
+
\lambda_a\mathcal L_{actor}.
\]

权重只通过初始 gradient RMS 校准：

\[
0.5
\le
\frac{\|\lambda_e g_e\|}
{\|g_{real}\|+\epsilon}
\le2,
\]

\[
0.5
\le
\frac{\|\lambda_a g_a\|}
{\|g_{real}\|+\epsilon}
\le2.
\]

禁止根据 rollout 结果搜索权重。

## 9.4 Gradient audit

训练前记录：

```text
real vs ego gradient cosine
real vs actor gradient cosine
ego vs actor gradient cosine
per-module gradient norm
temporal vs spatial trainable tensors
```

要求：

- gradient finite；
- motion gradient到达 temporal LoRA；
- spatial LoRA为0；
- 无 unexpected trainable parameter；
- 若 median cosine < -0.5，停止自动训练并分类为 severe objective conflict。

只允许一个 fallback：

```text
alternating real/motion updates
```

不自动引入 PCGrad/复杂多任务优化。

## 9.5 Capacity 规模

```text
16 train clips
8 held-out clips
50 updates
若有改善，再到100 updates
1 seed
```

每 25 step 记录：

- held-out ego/actor probe；
- real denoising；
- adapter norm；
- first-frame prediction drift；
- Base-reference drift。

## 9.6 Rollout sanity

固定：

```text
8 validation conditions
2 matched generation seeds
25 steps
official protocol
Base vs adapter
```

独立 CoTracker：

```text
track survival
active coverage
velocity distribution
acceleration/curvature outlier
dynamic degree
```

训练侧真实 target不得用于 generated评价。

## 9.7 A2 gate

全部满足：

1. held-out actor residual error改善至少10%；
2. held-out ego error不恶化超过5%；
3. real denoising不恶化超过10%；
4. frame-0 visual drift不明显；
5. CoTracker primary motion metric至少方向改善；
6. dynamic degree、survival无>5%退化；
7. 两 generation seeds方向一致；
8. 无 NaN/OOM。

若 representation改善但 rollout不变：

```text
classification = representation_only_no_transfer
```

不要长训；将 short-chain/feedback architecture列为后续研究，不在本轮扩展。

若 rollout方向恶化：

```text
Route A training mechanism rejected
```

---

# 10. Route B — Natural Rollout Best-of-N / Reward Alignment Ceiling

该路线不再制造内部 sibling。

## 10.1 研究问题

> Frozen SVD 的自然独立采样分布中，是否已经存在人工和独立 evaluator 都认可的更优 motion rollout？如果没有，任何 AWR/DPO 都缺乏可利用的 support。

## 10.2 Candidate generation

固定：

```text
16 preference-dev conditions
svd_official_v1
25 steps
相同 condition
独立 initial seeds
先4 candidates / condition
```

如果少于 12/16 condition存在至少两个合法、非重复、可比较 candidate：

```text
同一16 conditions扩到8 candidates
```

禁止改 CFG、fork、\(\rho\)、scheduler或后处理。

总上限：

```text
16 × 8 = 128 videos
```

## 10.3 Scorers

训练侧候选排序诊断：

```text
RAFT + P-UNC
generic motion smoothness
quality guard
motion floor
track survival
```

独立 evaluation：

```text
CoTracker3
```

P-UNC 不得作为唯一 best-of-N 结论。

## 10.4 Best-of-N 对照

对每 condition比较：

```text
Base fixed seed
random candidate
P-UNC-best
generic-smoothness-best
CoTracker-best（仅 oracle upper bound，不可训练）
```

报告：

- candidate diversity；
- scorer rank correlation；
- winner branch/seed偏差；
- dynamic degree；
- track survival；
- quality；
- best-of-N upper bound；
- P-UNC 与 CoTracker direction agreement。

## 10.5 Anti-collapse

best candidate不得通过：

- freeze；
- slow-down；
- object disappearance；
- low track coverage；
- blur/flicker；
- first-frame损坏；

获胜。

复用现有 shortcut stress 思路，但对象是自然 rollout。

## 10.6 Human review

生成 24 个盲审 cases：

```text
12: P-UNC-best vs random
12: P-UNC-best vs Base
```

Stage A 不展示 scorer。

Verdict：

```text
A better / B better / tie / both invalid
motion plausibility
motion amount
visual quality
identity consistency
reason
```

Codex不得代填。

不等待 review，继续 Route A 和 machine aggregation。

## 10.7 B0 machine gate

机器侧至少要求：

1. ≥12/16 conditions有有效 candidate diversity；
2. P-UNC-best 相对 random 的 CoTracker win rate ≥60%；
3. P-UNC-best 相对 Base 的 CoTracker win rate ≥55%；
4. dynamic degree/survival无系统下降；
5. P-UNC 与 generic smoothness不是完全等价排序；
6. scorer-best不是由极少数 conditions主导；
7. 不存在 seed ID垄断。

人工最终 gate：

```text
decisive P-UNC-best preference >=65%
low-motion winner = 0
catastrophic winner = 0
```

若 machine gate失败：

```text
Route B rejected
```

若 machine通过但人审未完成：

```text
Route B = awaiting_reviews
```

## 10.8 后续定位

若 B0 通过，下一阶段可考虑：

```text
condition-relative AWR + real SFT
```

不是 DPO。

原因：

- 自然 candidates不需要可靠二元 strict pair；
- continuous advantage 更适合 candidate pool；
- SHIFT 已证明 AWR/SFT hybrid 是强 baseline；
- 本计划不自动进入 B1长训。

## 10.9 代码与产物

新增：

```text
motion_proj/diagnostics/natural_rollout_ceiling.py
motion_proj/eval/natural_rollout_ranking.py
configs/diagnostics/route_pivot_natural_rollout.yaml

tests/test_natural_rollout_pool.py
tests/test_best_of_n_selection.py
tests/test_natural_rollout_anti_collapse.py
```

run：

```text
runs/route-pivot-b0-natural-rollout-<unique-id>/
```

---

# 11. Route C — Backbone Migration Audit（仅在 A/B 都失败时）

若：

```text
Route A rejected
AND
Route B rejected
```

不要继续修 SVD scorer 或 loss。

执行只读迁移审计：

## 11.1 候选

优先：

```text
OpenDWM
其他有官方权重、nuScenes支持的 action/trajectory-conditioned driving world model
```

## 11.2 审计内容

- pretrained checkpoint；
- nuScenes data schema；
- ego/action/trajectory condition；
- other-agent representation；
- model parameter count；
- single 4090 inference；
- two-4090 LoRA或adapter feasibility；
- official evaluation；
- license；
- cache/dataset复用比例；
- 预计迁移工作量。

不得自动下载超大权重或开始训练。

输出：

```text
docs/BACKBONE_MIGRATION_AUDIT.md
```

最终可以选择：

```text
C1：迁移到 action-conditioned driving backbone
C2：停止 CVPR 2027 当前生成模型方向
```

---

# 12. 路线自动决策

完成 R1、A0、A1、B0 后：

## Decision A

```text
A1 pass
```

选择：

```text
Main = Route A
Fallback = Route B（若人审通过）
```

继续 A2。

## Decision B

```text
A1 fail
B0 machine + human pass
```

选择：

```text
Main = natural-rollout AWR/SFT
Route A rejected
```

本轮只输出 B1 预注册计划，不自动长训。

## Decision C

```text
A1 pass
B0 pass
```

选择：

```text
Main = Route A representation alignment
Post-training baseline = Route B AWR
```

A2 通过后再考虑组合。

## Decision D

```text
A1 fail
B0 fail
```

选择：

```text
SVD motion-post-training support insufficient
进入 Route C 迁移审计
```

## Decision E

```text
A1 probe pass
A2 representation improves
A2 rollout no transfer
```

结论：

```text
one-step representation alignment insufficient
```

后续候选才是：

```text
short-chain feature feedback
refiner/zero-conv
controlled backbone
```

不得直接长训。

---

# 13. 评估与统计规范

## 13.1 真实 target/probe

统计单位：

```text
scene
clip
instance
frame pair
```

不能把所有点当独立样本。

报告：

```text
per-scene mean/median
moving/stationary strata
category strata
visibility strata
ego-motion magnitude strata
valid coverage
worst 10%
```

## 13.2 Generated rollout

主 evaluator：

```text
CoTracker3
```

训练侧：

```text
RAFT/P-UNC
```

报告：

- camera-compensated image-plane velocity；
- acceleration/curvature outlier；
- track survival；
- active coverage；
- dynamic degree；
- subject/background consistency；
- temporal flicker；
- first-frame fidelity；
- invalid rate。

## 13.3 Bootstrap

- generation seeds先在condition内聚合；
- clips在scene内聚合；
- scene-level paired bootstrap；
- 10,000 samples；
- invalid不填0；
- 报告实际 \(n\)；
- 小于32 clips只称screening；
- FVD只在至少256 clips或full val报告。

---

# 14. 资源边界

## 单卡

本轮所有 gate使用单张4090：

```text
R1
A0
A1
A2 capacity
B0 128-video上限
```

每个 run记录：

```text
peak VRAM
seconds/clip
seconds/video
seconds/feature extraction
seconds/update
disk usage
```

## 磁盘

不缓存全feature maps。

优先保存：

```text
sampled feature records
targets
manifest
mp4 review material
metrics
```

至少保留30 GB安全空间。

## 双卡

本计划禁止切双卡。

只有最终主路线通过单卡 feasibility 后，由下一份正式 scale plan 决定是否切换。

---

# 15. Git、运行与证据规范

每个阶段开始：

```bash
git status --short
git rev-parse HEAD
```

每个正式 run：

```text
unique run ID
resolved config
manifest
git state
environment fingerprint
data/model fingerprint
metrics.jsonl
summary.json
COMPLETE / FAILED / REJECTED / awaiting_reviews
```

失败 run：

- 不删除；
- 不覆盖；
- 不复用 ID；
- 区分 engineering failure 与 research failure。

提交前：

```bash
PYTHONPATH=. pytest -q
git diff --cached --check
git diff --cached
```

建议 commits：

```text
docs(research): 启动真实运动表征路线审计
fix(data): 完善 nuScenes 运动标注 schema
research(data): 验证 ego 与 actor residual target
research(features): 审计 SVD 真实运动表示
research(eval): 测量自然 rollout best-of-N 上限
research(train): 验证真实运动辅助对齐容量
docs(research): 固化路线切换结论
```

不自动 push。

---

# 16. 输出文档

必须生成：

```text
docs/MOTION_ROUTE_PIVOT_AUTORESEARCH_PLAN_V5.md
docs/ROUTE_PIVOT_LITERATURE_MATRIX.md
docs/ROUTE_PIVOT_TEMPORAL_AUDIT.md
docs/ROUTE_PIVOT_FINAL_REPORT.md
```

更新：

```text
docs/RESEARCH_STATUS.md
docs/RESEARCH_FAILURES.md
docs/EXPERIMENTS.md
```

`docs/archive/` 中的历史复盘和计划不再修改。新的负结论追加到 `RESEARCH_FAILURES.md`，当前 gate 与唯一
下一步只写入 `RESEARCH_STATUS.md`，原始 run 事实追加到 `EXPERIMENTS.md`。

`ROUTE_PIVOT_FINAL_REPORT.md` 必须包括：

1. Executive decision；
2. 当前 repo/环境；
3. sibling路线关闭证据；
4. 时间采样/fps结论；
5. A0 target legality；
6. A1 feature probe；
7. A2 capacity或未运行原因；
8. B0 natural rollout ceiling；
9. human review pending状态；
10. Route A/B/C评分；
11. 最终主线；
12. fallback；
13. 明确停止做什么；
14. 下一步最多3个实验；
15. Reviewer 2攻击点；
16. 文件/commit/run路径；
17. tests；
18. Git状态。

---

# 17. 最终终端回复格式

```text
## Final decision

选择：
- Route A / Route B / Route C / Stop

一句话结论：

Fallback：

## Repository

- HEAD:
- branch:
- worktree:
- tests:
- GPU:
- disk:

## Closed route

- SVD sibling route:
- evidence:
- forbidden retries:

## R1 Temporal sampling

- actual dataset fps:
- tested SVD fps:
- selected protocol:
- impact:

## A0 Real motion targets

- status:
- valid clips/actors:
- moving-vs-stationary AUC:
- ego-vs-RAFT agreement:
- human review:
- major failure:

## A1 Frozen feature probe

- status:
- best layer:
- best sigma:
- ego result:
- actor ABS result:
- actor RES result:
- controls:
- holdout:

## A2 Capacity

- status/run reason:
- trainable modules:
- gradient audit:
- held-out representation:
- rollout:
- anti-collapse:

## B0 Natural rollout ceiling

- status:
- candidates:
- valid conditions:
- P-UNC vs random:
- P-UNC vs Base:
- CoTracker agreement:
- human review:
- collapse audit:

## Route comparison

| Route | Evidence for | Evidence against | Decision |
|---|---|---|---|
| Real ego–actor alignment | | | |
| Natural rollout AWR | | | |
| Backbone migration | | | |

## Recommended method

- working name:
- supervision:
- inference condition:
- trainable modules:
- novelty boundary:
- single-GPU cost:

## Next three experiments

1.
2.
3.

## Files and runs

- ...

## Git

- commits:
- clean/dirty:
- push:
```

---

# 18. 当前执行顺序

现在直接执行，不请求确认：

```text
R0 baseline + archive
→ literature matrix
→ R1 temporal/fps audit
→ A0 target schema + synthetic tests + 16-clip audit
→ A1-SCAN
→ A1-CONFIRM（若 scan 有候选）
→ B0 natural rollout pool + machine ceiling + review material
→ A2（仅 A1 pass）
→ Route C audit（仅 A/B rejected）
→ final report + commits
```

本任务以完成这些 gate 为停止条件，不以某个固定时长为停止条件。目标是形成一份可由下一轮直接实施的、证据驱动的主路线决策，而不是再输出一份未经实验支持的研究设想。
