# Motion-Proj 新路线 Autoresearch 计划 V7

> **工作名称**：**OccGS-Resim — Occupancy-Anchored Object-Centric Gaussian Resimulation for Driving Data Generation**  
> **中文名称**：基于占用约束与对象中心高斯场景图的驾驶反事实重仿真  
> **目标投稿**：CVPR 2027  
> **计划状态**：`approved_for_autonomous_execution`（用户已授权：清空 context 后按本计划自主连续执行至终态）  
> **当前任务**：V7 自主执行已收尾 —— 见 `docs/OCCGS_FINAL_REPORT.md`（D1=`modify_method_then_scale`）。  
> **硬件基线**：单张 RTX 4090 24 GB；数据盘 128 GB、不可扩容。  
> **当前代码基线**：Motion-Proj V6；原 SVD/Projection/Preference/ReSim 路线均已关闭。  
> **执行原则**：先验证可编辑三维场景和反事实传感器数据是否合法，再考虑生成补全和下游数据效用。  
> **核心禁令**：不再把通用 2D 视频扩散 latent 当作唯一世界状态；不重开 RF-01–RF-18 中已拒绝的配方。  
> **自主执行契约**：完整授权、上下文回填、终止与关机规则见 **§A0**；本计划已可在清空 context 后独立接续。

---

# A0. 自主执行协议（Autonomous Execution Contract）

> **本节为本轮最高优先级。** 用户已授权：清空 context 后，直接把本文件交给一个新 agent 自主连续执行，
> 无需逐 gate 人工确认。任何新会话开机后，**第一步永远是执行 §A0.6 恢复协议**。

## A0.1 授权范围

用户明确授予以下权力，可自行判断使用，无需再确认：

1. **自主连续执行**：按 §13 里程碑顺序自动推进 `G0 → D0 → B0 → O0 → S0 → C0 → L0 → U0 → D1`；
   某个 machine gate **通过**即自动进入下一 gate，**不再停下等待人工审阅**（这一点覆盖旧 §20/§21
   中“第一轮结束即停机待审”的措辞）。
2. **任意调用 subagent**：允许自由使用 `explore`（代码库/文献探查）、`generalPurpose`（多步子任务）、
   `shell`（命令执行）、`best-of-n-runner`（隔离 worktree 做并行/对照实验）等。长搜索、并行基线、
   隔离试验都应优先交给 subagent，以节省主 context。
3. **临时落盘长会话**：过长的中间输出/日志/草稿可写入 `/root/autodl-tmp/motion_proj/tmp/*.txt`
   （已 gitignore）。这些是**非正式草稿**；正式证据仍按 §16 写入 `runs/`。

## A0.2 上下文保鲜与回填规则（强制）

> 目的：**任何时刻清空 context 都能无缝接续**。因此“完成一个小 todo 就必须回填”，而不是批量补写。

每完成一个 todo / 子里程碑，**在开始下一件事之前**必须依次回填：

1. **本节 §A0.5 执行进度日志**：append 一行（UTC 时间戳、gate/todo、结论、证据路径、下一步）。
2. **§13 里程碑表**：更新对应行的 `状态` 列（`pending`/`in_progress`/`done`/`rejected`/`blocked`）。
3. **`docs/RESEARCH_STATUS.md`**：更新“当前计划 / 当前状态 / 当前任务 / 当前 gate”。
4. 有正式数值时更新 **`docs/EXPERIMENTS.md`**（append-only）；出现新的研究负结论时新增
   **`docs/RESEARCH_FAILURES.md`** 的 `RF-*` 条目并引用其重开条件。
5. 阶段产出文档（§17）按需创建/更新：`OCCGS_THIRD_PARTY_AUDIT.md`、`OCCGS_DATA_PREPARATION.md`、
   `OCCGS_RECONSTRUCTION_BASELINE.md`、`OCCGS_OCCUPANCY_STATE.md`、`OCCGS_COUNTERFACTUAL_PROTOCOL.md`、
   `OCCGS_FINAL_REPORT.md`。

回填后再用 `TodoWrite` 推进内部 todo。禁止“先跑完很多步、最后一起补文档”。

## A0.3 硬约束（不得违反）

- **单卡**：所有阶段限单张 RTX 4090；单场景峰值显存目标 `< 22 GB`（硬顶 24 GB）。不默认 DDP，
  不擅自切双卡（仅当 §14.2 全部条件满足时，暂停并在进度日志记录“建议双卡”后按 §A0.4 处理）。
- **磁盘 ≥ 30 GiB**：任何训练/预处理/大批产物写盘前先 `df -h /root/autodl-tmp`，确保
  `avail − 预估峰值 ≥ 30 GiB`；否则先缩协议或清理**可重建** cache（不得删受保护材料）。
  raw nuScenes 只用 symlink，不复制。
- **镜像装包**：pip=`mirrors.aliyun.com`，torch wheels=`mirrors.aliyun.com/pytorch-wheels`，
  conda=`mirrors.tuna.tsinghua.edu.cn`；禁止 `download.pytorch.org` / `pypi.org`。
- **环境隔离**：只用 `/root/autodl-tmp/envs/drivestudio`；禁止污染 `motionproj` / `resim`。
- **防重复**：不重开 `RF-01`–`RF-18` 已拒绝的配方；不把 dataset 拥有的 future state 当作自由生成条件。
- **Git**：可 `commit`（用 §16.4 前缀），**禁止自动 `push`**；不删除受保护 run/manifest/review 材料。

## A0.4 终止条件与关机（Termination & Shutdown）

**“全部工作完成”** 指下列任一成立：

- **(A) 成功走完全流程**：`U0` 通过且 `D1` 得出最终决策（accept / scale-up / 修改方法）；或
- **(B) 命中停止条件而 reject**：任一阶段命中 §15 停止条件，按 §16 归档为 `REJECTED` 终态；或
- **(C) 授权范围内无法解决的 blocker**：环境/数据/资源阻塞在本协议授权范围内无法自动修复。

无论 (A)/(B)/(C)，收尾**必须按顺序**完成：

1. 写 / 定稿 `docs/OCCGS_FINAL_REPORT.md`（结论、通过与失败的 gate、证据路径、后续建议）。
2. 回填 §A0.5、§13、`RESEARCH_STATUS.md`、`EXPERIMENTS.md`、（如有）`RESEARCH_FAILURES.md`。
3. `git add -A && git commit`（不 push）；确认 `git status` 干净。
4. **确认上述 1–3 全部完成后**，在终端执行关机：

```bash
sync
/usr/bin/shutdown -h now   # 当前为 root；如非 root 用 sudo /usr/bin/shutdown -h now
```

关机是**最后一个动作**；执行前必须再次核对所有文档已回填、commit 已完成。若因权限/环境无法关机，
在进度日志记录“shutdown_failed + 原因”后结束。

## A0.5 执行进度日志（append-only）

> 新会话读这里即可知道“已经做到哪、下一步做什么”。只追加、不改写历史行；时间用 UTC。

| UTC 时间 | gate / todo | 结论 | 证据 / 产出路径 | 下一步 |
|---|---|---|---|---|
| 2026-07-21T16:34Z | `E0-ENV-01` | done（smoke_passed） | `docs/OCCGS_E0_ENV_MANIFEST.md`；env `/root/autodl-tmp/envs/drivestudio` | 进入 `G0` 第三方审计 |
| 2026-07-21T16:40Z | `A0` 自主执行协议 | 计划已固化为可自主执行 | 本文件 §A0 | 由新 context 从 `G0` 开始 |
| 2026-07-21T17:02Z | `G0-THIRDPARTY-00` | **PASS**：DriveStudio MIT/单卡默认/可限 scene+camera+time；SplatAD Apache-2.0 仅参考；Occ3D MIT 可按 scene 子集取；本地 raw 审计发现只有 v1.0-mini 10 scenes 有完整前向 sweep → D0 场景池限于 mini | `docs/OCCGS_THIRD_PARTY_AUDIT.md`、`docs/OCCGS_LICENSE_AND_DATA_POLICY.md` | 进入 `D0-DATA-02`：从 mini 10 scenes 选 S0/S1/S2 并冻结标准 |
| 2026-07-21T17:50Z | `D0-DATA-02` | **PASS**：冻结 S0=0655/003、S1=0796/005、S2=0757/004；10Hz 预处理+SegFormer-B5 sky mask；完整性三 scene OK；清理已关闭 V6 的 ReSim CogVideoX 权重 37G（源码保留）后 avail≈70GiB | `docs/OCCGS_DATA_PREPARATION.md`；`data/occgs/scene_specs/d0_frozen_picks_v2.json`；`data/occgs/reviews/d0_integrity/` | 进入 `B0-RECON-03`：先 B0-1 单相机 4s StreetGS smoke |
| 2026-07-21T19:20Z | `B0-RECON-03` | **PASS**：B0-1 smoke + B0-2/3/4 冻结 StreetGS 30k×3scene 完成；S0/S2 test PSNR≈25.5；S1 test 20.2（动态更强，可识别）；无 NaN/OOM；peak VRAM≈5GB；RigidNodes 分离可见 | `docs/OCCGS_RECONSTRUCTION_BASELINE.md`；`runs/occgs_resim/b0_recon/occgs_b0/b0_{2,3,4}_*` | 进入 O0（occupancy 已并行构建）→ S0 → C0 |
| 2026-07-21T19:30Z | `O0-OCC-04` | **PASS**：per-frame ego LiDAR+box occupancy；unknown 保留；不下载 Occ3D | `docs/OCCGS_OCCUPANCY_STATE.md`；`data/occgs/occupancy/{003,004,005}/` | 进入 S0 |
| 2026-07-21T19:40Z | `S0-EDIT-05` | **PASS**：raised-cosine cut-in editor；V4 负例拒绝；actor 与 RigidNodes true_id 对齐（003/a35，005/a34，004/a8） | `data/occgs/scene_specs/s0_edits/`；`docs/OCCGS_COUNTERFACTUAL_PROTOCOL.md` §1 | 进入 C0 |
| 2026-07-21T19:42Z | `C0-CF-06` | **PASS**：无扩散反事实渲染；编辑局部可见；机器合法 top-24=24/24；≥2 scene（003/004） | `runs/occgs_resim/c0_cf/{s0,s1,s2}/`；`data/occgs/reviews/c0_legality/` | 进入 L0 |
| 2026-07-21T19:43Z | `L0-COMP-07` | **PASS***：Telea+hard composition；outside-mask L1=0；视觉弱不宣称 >GS | `runs/occgs_resim/l0_comp/`；`docs/OCCGS_L0_COMPLETION.md` | 进入 U0 |
| 2026-07-21T19:44Z | `U0-UTILITY-08` | **partial**：proxy PASS（OccGS accept、naive V4 reject、有 RGB 信号）；**未跑** camera 3D mAP | `runs/occgs_resim/u0_screen/u0_proxy_v1.json` | 进入 D1 |
| 2026-07-21T19:45Z | `D1-DECIDE-09` | **done**：决策 `modify_method_then_scale`；路线不 reject；完整下游待扩 | `docs/OCCGS_FINAL_REPORT.md` | 收尾 commit + shutdown |

## A0.6 恢复协议（新会话第一步）

清空 context 后，新 agent 必须按此顺序接续，不得跳过：

1. 读本文件 **头部状态块 + §A0（尤其 §A0.5 进度日志）+ §13 里程碑表**。
2. 读 `docs/RESEARCH_STATUS.md` 的“当前任务 / 当前 gate”。
3. 定位 §13 中第一个非 `done` 的里程碑 = 当前 gate；读其对应正文小节与 Gate 条件。
4. 若该 gate 有半成品证据（`runs/occgs_resim/...` 或 `tmp/*.txt`），先核对再续跑，避免重复劳动。
5. 继续自主执行，遵守 §A0.2 回填规则与 §A0.3 硬约束，直至命中 §A0.4 终止条件并关机。

---

# 0. Executive Decision

## 0.1 为什么切换到这条路线

Motion-Proj 已验证的主要失败具有一致结构：

```text
隐式 2D 视频状态
→ 尝试恢复/修改物理运动
→ 缺少显式动作、几何、遮挡和实例状态
→ target、preference 或 action response 不可辨
```

新的因果顺序改为：

```text
真实驾驶日志
→ 显式静态场景 + object-centric dynamic actors
→ occupancy / map / kinematic constraints
→ 修改 ego / actor trajectory
→ 重新渲染 RGB、depth、semantic、instance、box
→ 只在真实缺失区域做局部生成补全
→ 验证合成数据对下游任务的价值
```

这里的“world model”不再等同于一个视频扩散模型，而是：

\[
S_{t+1}=f_{\mathrm{dyn}}(S_t,a_t,\eta_t),
\qquad
Y_t=g_{\mathrm{sensor}}(S_t),
\]

其中：

- \(S_t\)：显式三维世界状态；
- \(f_{\mathrm{dyn}}\)：运动学、轨迹或学习式状态转移；
- \(g_{\mathrm{sensor}}\)：camera/LiDAR/depth/semantic renderer；
- 视频生成模型只负责 renderer 无法观察到的外观补全。

## 0.2 当前研究问题

> 对真实驾驶日志进行 actor 轨迹编辑时，能否利用 occupancy 约束和 object-centric Gaussian scene graph，生成在几何、遮挡、实例、传感器标签和时间上都一致的反事实驾驶数据，并证明这些数据比纯 3DGS 编辑或纯视频生成更适合下游自动驾驶训练？

## 0.3 工作假设

### H1：Occupancy anchor 能提高 actor edit 的合法性

相对纯 object-centric 3DGS：

- 更少碰撞和道路越界；
- 更正确的遮挡/深度顺序；
- 更一致的 RGB、depth、instance mask 和 2D/3D box；
- 更准确的可见区域与 disocclusion mask。

### H2：显式 disocclusion mask 能解决局部生成的作用域问题

将生成模型限制在：

```text
新暴露区域
+
低几何置信区域
```

而非重生成整帧，可以避免旧路线中：

- mask 外变化；
- source duplication；
- identity drift；
- 首帧和背景泄漏。

### H3：经过合法性过滤的反事实数据能产生可测的下游收益

至少在一个明确长尾场景上，例如：

```text
vehicle cut-in / lane merge
```

合成数据应提升：

- camera 3D detection；
- occupancy prediction；
- 或场景事件识别/挖掘；

而不是只提升 PSNR/SSIM。

## 0.4 论文创新边界

以下内容已经不是创新：

- 用 3DGS 重建驾驶场景；
- 静态背景与动态车辆分解；
- 移动车辆、删除车辆或改变相机轨迹；
- 用 occupancy 生成视频/LiDAR；
- 用视频扩散增强 novel-view rendering；
- 实时 RGB/LiDAR Gaussian rendering。

本项目只可能在以下组合上形成贡献：

1. **Panoptic Occupancy–Gaussian Binding**  
   occupancy、实例状态、Gaussian actor node 和监督标签绑定为同一可追溯世界状态。

2. **Constraint-Certified Counterfactual Editing**  
   actor trajectory edit 在渲染前通过 road/free-space/collision/ground-contact/kinematic gate。

3. **Visibility-Consistent Sensor and Label Regeneration**  
   每次编辑同步生成 RGB、depth、semantic、instance mask、2D/3D box、可见性和 provenance。

4. **Occupancy-Guided Localized Disocclusion Completion**  
   生成模型只补充由 ray tracing 确认的新暴露区域，并 hard-preserve 已知几何区域。

5. **Downstream Data Utility**  
   用长尾反事实数据验证真实任务收益。

如果最终只得到：

```text
DriveStudio / Street Gaussians 复现
+
手工移动车辆
```

则方法贡献不足，应作为基础设施结果而非 CVPR 主论文。

---

# 1. 最近邻工作与基线定位

## 1.1 Reconstruction-first

### UniSim

核心思想：

```text
单条真实日志
→ 静态背景 + 动态 actor 重建
→ 改变 ego/actor 状态
→ 重新渲染 camera/LiDAR
```

本项目继承：

- 先修改三维场景，再渲染；
- 动态 actor 独立表示；
- 新视角和 actor edit。

区别候选：

- occupancy/panoptic state 作为编辑合法性与标签一致性层；
- 更明确的 localized disocclusion completion；
- 基于公开 nuScenes/DriveStudio 的轻量实现。

### Street Gaussians / DrivingGaussian / OmniRe

共同能力：

- static/dynamic decomposition；
- object-centric Gaussian nodes；
- tracked actor poses；
- novel-view rendering；
- scene editing。

本项目不能把这些写成原创。

### SplatAD

提供 camera 和 LiDAR Gaussian rasterization，并显式考虑 rolling shutter、LiDAR intensity 和 ray dropout。

本项目第一阶段不立即实现完整 SplatAD，但将其作为：

- 多传感器 renderer reference；
- 后续 LiDAR 输出基线；
- sensor realism 对照。

## 1.2 Occupancy-first

### Occ3D / OpenOccupancy

本项目优先将 occupancy 用作：

- static/free-space reference；
- visibility和occlusion约束；
- edit validity；
- semantic/geometry evaluation。

第一阶段不训练 occupancy predictor。

### UniScene

UniScene 证明 occupancy 可以作为 semantic occupancy、video 和 LiDAR 的统一中间状态。

本项目不会从头复现其 Occ-VAE/DiT/video/LiDAR 全链路；只借鉴：

```text
layout / occupancy
→ multimodal sensor generation
```

## 1.3 Hybrid geometry + generation

### StreetCrafter / DriveDreamer4D / ReconDreamer

共同思想：

- 显式几何提供相机控制；
- 视频生成模型补充未观测视角；
- 生成数据反哺 3D/4D reconstruction。

本项目只在显式 Gaussian edit 已通过后，研究：

```text
occupancy/ray-tracing uncertainty
→ localized video completion
```

不重新让视频扩散生成整段视频。

## 1.4 Primary Open-Source Stack

### 主框架：DriveStudio

选择理由：

- 原生支持 nuScenes；
- 统一多数据集处理；
- 支持多相机；
- 支持 background、vehicle 和 non-rigid Gaussian representations；
- 包含官方 OmniRe 与 Street-Gaussians-style 实现；
- 基于现代 gsplat；
- 支持 camera pose 和 GT box refinement。

### 第一基线：DriveStudio 中的 Street-Gaussians-style multi-representation

只建模：

```text
static background
+
vehicle Gaussian nodes
```

暂不启用 SMPL 或复杂 non-rigid humans。

原因：

- 最接近当前“车辆轨迹反事实”研究问题；
- 环境和数据准备更轻；
- 先验证车辆 edit，不把 pedestrian body model 引入主问题。

### 第二基线：OmniRe

仅在第一基线通过后启用，用于：

- 更复杂 actor；
- 多类动态场景；
- 检查结果是否依赖简化表示。

### Renderer Reference：SplatAD / neurad-studio

第一阶段仅做代码和接口审计；不下载 PandaSet、不另起完整训练路线。

---

# 2. 与 Motion-Proj V6 的关系

## 2.1 保留内容

从 Motion-Proj 复用：

```text
nuScenes raw data位置与devkit经验
scene-level split和fingerprint
真实timestamp/pose/box schema
geometry utilities
CoTracker/RAFT evaluator
run manifest和atomic state
人工review协议
RESEARCH_FAILURES防重复账本
```

## 2.2 不直接复用

不把以下模块接入新训练：

```text
SVDBackbone
projection target
P-UNC renderer
physics preference pair
DrivePO
ReSim action screen
temporal LoRA
```

这些模块只作为：

- 失败证据；
- evaluator工具；
- 未来 localized completion 的历史风险提醒。

## 2.3 新仓库组织

推荐保持两个独立仓库：

```text
/root/autodl-tmp/motion_proj
/root/autodl-tmp/third_party/drivestudio
```

新研究控制代码可以先放：

```text
motion_proj/resim/
motion_proj/occupancy/
motion_proj/gaussian_eval/
```

但不 fork/复制 DriveStudio 核心训练器到 Motion-Proj。

通过 wrapper、manifest 和 adapter 层连接：

```text
Motion-Proj:
  research orchestration
  scene specification
  occupancy/edit validation
  evaluation
  downstream utility

DriveStudio:
  data preprocessing
  Gaussian reconstruction
  rendering
```

---

# 3. 总体阶段图

```text
G0 事实与第三方审计
 ↓
E0 DriveStudio单卡环境初始化
 ↓
D0 nuScenes小规模数据准备
 ↓
B0 object-centric Gaussian重建基线
 ↓
O0 occupancy/panoptic anchor构建
 ↓
C0 无生成模型的counterfactual actor edit
 ↓
C1 multimodal label一致性和人工合法性
 ↓
L0 局部disocclusion completion feasibility
 ↓
U0 下游数据效用screening
 ↓
D1 最终路线：扩规模 / 修改方法 / 停止
```

所有阶段必须 machine gate 通过后才进入下游。

---

# 4. G0 — 事实、代码与许可证审计

## 4.1 目标

在安装和下载前，确认：

- DriveStudio 当前 commit；
- NuScenes loader的输入结构；
- Street-Gaussians-style和OmniRe config；
- 单卡显存入口；
- edit/render接口现状；
- license；
- 第三方模型和数据使用限制。

## 4.2 必须核查

```text
DriveStudio:
  repository commit
  requirements
  gsplat version
  PyTorch/CUDA compatibility
  NuScenes preprocess
  available configs
  checkpoint/resume
  render outputs
  scene edit APIs
  object nodes and transforms

SplatAD:
  rasterizer API
  neurad-studio integration
  license
  supported outputs

Occ3D:
  annotation format
  grid convention
  visibility mask
  disk footprint
  license
```

## 4.3 输出

```text
docs/OCCGS_THIRD_PARTY_AUDIT.md
docs/OCCGS_LICENSE_AND_DATA_POLICY.md
```

## 4.4 Gate

必须满足：

- DriveStudio license允许学术研究；
- nuScenes processing可限定scene/camera/time range；
- 单卡可配置；
- 无需提前下载另一完整数据集；
- 不要求将 future actor状态输入自由生成模型。

失败时：

```text
评估 neurad-studio / Street Gaussians official作为fallback
```

---

# 5. E0 — 环境初始化计划

> **状态：已完成（`done`）。** 实测清单见 `docs/OCCGS_E0_ENV_MANIFEST.md`（torch 2.1.2+cu118 /
> gsplat 1.3.0 / pytorch3d 0.7.5 / nvdiffrast 0.4.0，smoke 全通过）。本节保留为操作记录与复现依据；
> 新会话无需重跑，只在环境损坏时按此重建。

## 5.1 环境隔离

保留：

```text
motionproj env
resim env
```

新增：

```text
drivestudio env
drivestudio-seg env（仅在必须自行提取mask时）
```

禁止在 `motionproj` 中直接安装 DriveStudio 依赖。

## 5.2 目录

```text
/root/autodl-tmp/
├── motion_proj/
├── third_party/
│   ├── drivestudio/
│   └── splatad/              # 仅代码审计阶段，可后装
├── envs/
│   ├── motionproj/
│   ├── resim/
│   ├── drivestudio/
│   └── drivestudio-seg/
├── data/
│   ├── nuscenes/             # 现有raw数据
│   └── occgs/
│       ├── processed/
│       ├── occupancy/
│       ├── scene_specs/
│       └── reviews/
└── runs/
    └── occgs_resim/
```

## 5.3 DriveStudio安装流程

执行前固定commit并写manifest。

参考流程：

```bash
cd /root/autodl-tmp/third_party
git clone --recursive https://github.com/ziyc/drivestudio.git
cd drivestudio
git checkout <PINNED_COMMIT>

conda create -p /root/autodl-tmp/envs/drivestudio python=3.9 -y
conda activate /root/autodl-tmp/envs/drivestudio

pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/NVlabs/nvdiffrast
```

最终版本必须以仓库requirements和当前CUDA兼容性为准，不允许盲目复制旧torch版本。

## 5.4 Environment smoke

必须通过：

```text
Python import
PyTorch CUDA
gsplat rasterization
PyTorch3D import
nvdiffrast import
DriveStudio config parse
one-camera dataloader
one forward render
one backward step
checkpoint save/load
```

记录：

```text
CUDA
driver
torch
torchvision
gsplat
pytorch3d
nvdiffrast
DriveStudio commit
peak VRAM
```

## 5.5 Mask环境

DriveStudio官方NuScenes流程要求sky mask，并可选fine dynamic mask。

优先级：

1. 复用官方/已提供的预处理mask；
2. 只对选定scene运行官方SegFormer流程；
3. 若旧mmcv/torch无法在当前机器可靠安装，使用现代分割模型建立替代mask，但必须：
   - 单独版本化；
   - 人工检查；
   - 与官方mask在mini subset对比；
   - 不静默改变baseline。

第一阶段不处理SMPL human poses。

## 5.6 E0 Gate

- full import通过；
- CUDA extension通过；
- rasterizer forward/backward通过；
- 显存峰值 < 22 GB；
- checkpoint可恢复；
- worktree和environment fingerprint落盘。

---

# 6. D0 — 数据集准备计划

## 6.1 数据策略

第一阶段只使用现有nuScenes，不下载Waymo或PandaSet。

理由：

- 当前项目已有nuScenes；
- DriveStudio原生支持nuScenes；
- nuScenes包含6相机、LiDAR、ego pose、3D boxes和instance identity；
- 降低磁盘和迁移成本。

## 6.2 数据频率

nuScenes：

```text
标注keyframe：2 Hz
camera原始频率：约12 Hz
LiDAR：20 Hz
```

DriveStudio支持插值到10 Hz。

本项目必须区分：

```text
真实camera frame
插值actor annotation
真实keyframe annotation
```

禁止将10Hz插值box写成新的人工GT。

## 6.3 第一批scene

通过只读扫描选3个互补scene：

### Scene S0：static-heavy

- 少量动态vehicle；
- 良好LiDAR覆盖；
- 小ego rotation；
- 用于环境和背景重建。

### Scene S1：vehicle-dynamic

- 至少2个持续可见moving vehicles；
- 明确instance token；
- 中等遮挡；
- 用于object-centric重建和简单轨迹编辑。

### Scene S2：cut-in / merge候选

- 车辆横向相对运动；
- lane topology可解释；
- actor保持至少4–6秒可见；
- 用于最终长尾scenario原型。

选择标准必须在查看训练结果前冻结。

## 6.4 第一阶段时间窗

每scene：

```text
8秒
10 Hz processed timeline
3个前向相机：
  CAM_FRONT_LEFT
  CAM_FRONT
  CAM_FRONT_RIGHT
```

约：

```text
80 timesteps × 3 cameras = 240 images / scene
```

这样：

- 比2Hz keyframe更适合3DGS；
- 低于DriveStudio对450+图像建议使用extended config的范围；
- 单卡和磁盘更可控。

## 6.5 预处理

目标命令形式：

```bash
python datasets/preprocess.py \
  --data_root <NUSCENES_RAW> \
  --target_dir /root/autodl-tmp/data/occgs/processed \
  --dataset nuscenes \
  --split v1.0-trainval \
  --start_idx <SCENE_INDEX> \
  --num_scenes 1 \
  --interpolate_N 4 \
  --workers <SAFE_WORKERS> \
  --process_keys images lidar calib objects
```

实际参数以固定DriveStudio commit为准。

之后：

```text
sky mask
dynamic/fine dynamic mask（按baseline需要）
```

## 6.6 数据完整性检查

每个scene：

- image count；
- timestamp monotonic；
- camera intrinsics/extrinsics；
- ego pose；
- LiDAR pose；
- object IDs；
- box interpolation；
- camera/box synchronization；
- mask coverage；
- LiDAR depth coverage；
- corrupt files；
- disk bytes。

可视化12个时刻：

```text
3-camera RGB
3D boxes
instance IDs
LiDAR projection
sky/dynamic masks
ego trajectory
```

## 6.7 磁盘门槛

环境和预处理前先测：

```text
现有可用空间
raw数据是否重复
conda cache
third_party size
processed scene size
checkpoint size
```

硬规则：

- 始终保留至少30 GB；
- 不复制raw nuScenes，只使用symlink；
- 第一阶段只处理3 scene；
- 不下载完整Occ3D、UniScene和SplatAD checkpoint；
- 可重建cache与正式证据分开。

## 6.8 D0 Gate

每scene：

- 3相机数据完整；
- 时间窗有效；
- 至少1个vehicle actor有连续轨迹；
- S1/S2至少有2个moving actors；
- mask与box人工抽查通过；
- 处理后总磁盘仍保留30 GB。

---

# 7. B0 — Object-Centric Gaussian Baseline

## 7.1 目标

先回答：

> 单卡和当前nuScenes数据能否稳定重建可编辑的静态背景和车辆节点？

不做任何occupancy创新。

## 7.2 训练阶梯

### B0-1：单场景、单相机、短窗

```text
S0
CAM_FRONT
4秒
Street-Gaussians-style config
```

验证：

- 环境；
- rasterizer；
- checkpoint；
- render；
- camera trajectory。

### B0-2：单场景、三相机、8秒

```text
S0
3 front cameras
8秒
```

验证多视角背景。

### B0-3：动态车辆场景

```text
S1
3 front cameras
8秒
background + vehicle nodes
```

验证：

- actor decomposition；
- tracked pose；
- actor render；
- occlusion。

### B0-4：3-scene repeatability

对S0/S1/S2使用完全冻结配置，禁止scene-specific调参。

## 7.3 基线比较

至少：

```text
Static GS / single representation
Street-Gaussians-style multi representation
OmniRe（仅在资源允许时）
```

目的不是刷SOTA，而是确认object decomposition真实有用。

## 7.4 重建指标

### RGB

```text
PSNR
SSIM
LPIPS
```

### Geometry

```text
LiDAR-depth L1/RMSE
depth ordering error
sky leakage
background/actor mask IoU
```

### Dynamic actor

```text
projected center error
box-mask consistency
actor identity consistency
pose adherence
occlusion correctness
```

### Temporal

```text
flicker
actor texture consistency
background stability
```

## 7.5 Held-out协议

禁止只在训练帧评价。

每个scene冻结：

```text
train frames
held-out time frames
held-out camera frames / novel pose
```

统计以scene为单位。

## 7.6 人工review

12个case：

- reconstruction；
- actor close-up；
- occlusion；
- held-out view；
- failure case。

## 7.7 B0 Gate

必须满足：

1. 3/3 scenes训练完成；
2. 无NaN/OOM；
3. actor与background基本分离；
4. held-out render可识别；
5. actor center/box误差在预注册阈值内；
6. 人工review无系统性ghosting；
7. 单scene训练时间和磁盘可接受。

失败分类：

```text
environment_failure
data_sparsity
pose_error
mask_failure
dynamic_decomposition_failure
single-GPU limitation
```

只有dynamic decomposition失败时才考虑更换到OmniRe或其他框架。

---

# 8. O0 — Occupancy / Panoptic Anchor

## 8.1 第一阶段不训练occupancy模型

Occupancy来源按优先级：

1. LiDAR + map + 3D boxes生成的scene-local occupancy；
2. 可选Occ3D-nuScenes annotation作验证；
3. 后续才考虑预测occupancy。

## 8.2 世界状态

定义：

\[
S_t=
\left(
O^{static},
\{N_i,T_{i,t},G_i,O_i^{canon}\}_{i=1}^{N},
T_t^{ego}
\right),
\]

其中：

- \(O^{static}\)：静态占用/free-space；
- \(N_i\)：actor node和instance ID；
- \(T_{i,t}\)：actor pose；
- \(G_i\)：actor Gaussian parameters；
- \(O_i^{canon}\)：canonical actor occupancy；
- \(T_t^{ego}\)：ego pose。

## 8.3 Static occupancy

来自：

```text
multi-sweep LiDAR
nuScenes map drivable area
ground/road separation
static/dynamic masks
```

输出：

```text
occupied
free
unknown
semantic class（可用时）
observed mask
```

必须保留unknown，不将未观测区域标为空闲。

## 8.4 Dynamic panoptic occupancy

对每个vehicle actor：

```text
instance ID
canonical occupancy
current SE(3) pose
velocity
visibility
Gaussian node IDs
```

第一版canonical occupancy可由：

- 3D box；
- actor LiDAR points；
- actor Gaussian alpha hull；

三者构建并比较。

## 8.5 Gaussian–occupancy binding

每个Gaussian必须标记：

```text
background / actor instance
canonical/world coordinates
occupancy cell support
semantic
confidence
```

编辑actor时：

```text
pose transform
→ Gaussian transform
→ instance occupancy transform
→ box/mask/depth重新生成
```

## 8.6 Occupancy一致性指标

- Gaussian mass outside occupancy；
- occupied voxel with no Gaussian support；
- free-space violation；
- actor–actor collision；
- actor–static collision；
- road-support violation；
- ground-contact error；
- rendered depth vs occupancy ray-depth；
- instance label mismatch。

## 8.7 O0 Gate

相对无occupancy anchor基线：

- collision/free-space错误显著减少；
- depth/visibility一致性提高；
- actor render质量无明显退化；
- unknown区域不被误当free；
- 结果不依赖Occ3D future信息。

---

# 9. S0 — Scenario State and Trajectory Editor

## 9.1 第一scenario

固定：

```text
vehicle cut-in / lane merge
```

选择原因：

- 与用户现有RoadMerge/LaneMerge知识强相关；
- 横向actor motion清晰；
- 可定义road/lane constraints；
- 适合数据挖掘、感知和规划下游。

## 9.2 轨迹表示

使用Frenet或road-aligned spline：

\[
\tau_i(t)=
(s_i(t),d_i(t),v_i(t),a_i(t)).
\]

第一版不训练神经dynamics。

## 9.3 显式约束

### Kinematic

```text
speed
acceleration
jerk
yaw rate
curvature
lateral acceleration
```

### Road

```text
drivable area
lane boundaries
road surface
heading alignment
```

### Interaction

```text
actor–actor collision
time-to-collision
minimum distance
ego occupancy
visibility
```

### Edit magnitude

```text
小扰动
中扰动
拒绝大幅超出重建support的轨迹
```

## 9.4 Counterfactual variants

每个source scene只生成有限、离散variants：

```text
V0 原始轨迹
V1 小幅提前cut-in
V2 小幅延后cut-in
V3 更强横向位移但仍合法
V4 rejected unsafe trajectory（仅作validator测试，不渲染训练数据）
```

禁止连续搜索生成“最好看”的轨迹。

## 9.5 Provenance

每个variant保存：

```text
source scene
actor ID
original trajectory
edited trajectory
constraint result
occupancy before/after
Gaussian transform
camera trajectory
renderer commit/config
all labels
```

---

# 10. C0 — 无生成模型的Counterfactual Rendering

## 10.1 目标

先证明显式scene edit本身合法。

输出：

```text
RGB
depth
semantic
instance mask
2D boxes
3D boxes
visibility
occupancy
uncertainty
```

第一阶段不做diffusion修复。

## 10.2 编辑类型

依次执行：

```text
actor pose perturbation
actor trajectory change
actor removal
```

actor insertion放到后续，因为需要外部canonical asset和更复杂appearance匹配。

## 10.3 可见性与遮挡

基于Gaussian/occupancy ray tracing计算：

```text
known visible
known occluded
newly disoccluded
unknown
actor overlap
```

不得用2D paste。

## 10.4 Counterfactual legality metrics

### Trajectory adherence

渲染actor中心、mask和box应与输入轨迹一致。

### Label synchronization

RGB/depth/instance/box应来自同一state snapshot。

### Occlusion

前后景顺序应与depth/occupancy一致。

### Identity

未编辑actor和背景不应变化。

### Locality

场景变化应主要发生在：

```text
edited actor tube
source footprint
new disocclusion
new occlusion
```

## 10.5 Human review

至少24 cases：

```text
original vs edited
RGB
depth
instance
occupancy BEV
trajectory
disocclusion mask
```

Verdict：

```text
geometry legal
actor motion plausible
occlusion correct
identity preserved
labels synchronized
render usable
failure reason
```

## 10.6 C0 Gate

- 轨迹投影误差通过；
- 碰撞/道路约束为0；
- 未编辑区域变化在阈值内；
- occlusion/depth一致；
- 2D/3D标签同步；
- 20/24人工case合法；
- 至少2个scene通过。

失败时不进入生成补全。

---

# 11. L0 — Occupancy-Guided Localized Completion

> 仅C0通过后执行。

## 11.1 研究问题

> 对移动/删除actor后新暴露、但原始日志从未观察到的区域，能否只在occupancy确定为unknown/disoccluded的区域做视频补全，并保持其他几何和实例状态不变？

## 11.2 Completion mask

\[
M_{\mathrm{comp}}=
M_{\mathrm{disocc}}
\cup
M_{\mathrm{lowconf}},
\]

必须排除：

```text
高置信Gaussian区域
edited actor可见区域
其他actor
已知road/structure边界
frame 0保护区域（按任务定义）
```

## 11.3 输入条件

至少：

```text
Gaussian RGB render
depth
semantic
instance
occupancy/ray status
completion mask
neighbor frames
```

## 11.4 模型选择

不在本计划阶段固定具体大模型。

选择顺序：

1. 现有视频inpainting/completion模型；
2. StreetCrafter类geometry-conditioned video diffusion；
3. 最后才考虑复用SVD。

禁止：

- 无mask整帧生成；
- 将completion模型输出作为新几何真值；
- 自动修改actor或depth；
- 在未知区域之外接受变化。

## 11.5 Hard composition

最终输出：

\[
I^{final}
=
(1-M)I^{GS}
+
MI^{gen}.
\]

边界可soft blend，但必须报告：

- outside-mask error；
- boundary artifact；
- depth/semantic conflict；
- temporal flicker。

## 11.6 L0 Gate

相对纯GS：

- disocclusion视觉质量提高；
- outside-mask变化接近0；
- depth order不变；
- actor identity不变；
- temporal consistency提高；
- 人工review通过。

若无法局部保持，则归档为：

```text
completion_locality_failure
```

不得回到整视频扩散路线。

---

# 12. U0 — 下游数据效用

> 仅C0通过；L0可选。

## 12.1 第一任务选择

优先级：

1. camera 3D detection；
2. occupancy prediction；
3. lane-merge/cut-in event classifier或retrieval；
4. planning（资源允许后）。

## 12.2 数据组

```text
R: 真实训练数据
R + GS: 未经occupancy过滤的Gaussian编辑数据
R + OccGS: occupancy-certified编辑数据
R + OccGS + completion: 局部补全数据
```

所有组：

- 数据量匹配；
- scene不泄漏；
-训练预算匹配；
- label provenance完整。

## 12.3 长尾测试集

构建固定cut-in/merge test strata：

```text
near/far
low/high occlusion
small/large relative speed
day/night
straight/turning road
```

## 12.4 指标

根据下游任务：

```text
mAP / NDS
occupancy mIoU / RayIoU
event precision/recall
worst-strata performance
calibration
```

## 12.5 U0 Gate

完整方法必须超过：

```text
real-only
real + naive GS edit
```

否则occupancy/edit方法没有数据价值。

---

# 13. Autoresearch里程碑

> 状态由 §A0.2 强制回填；本轮 V7 已全部收尾（见 `OCCGS_FINAL_REPORT.md`）。

| ID | 状态 | 内容 | 主要Gate |
|---|---|---|---|
| `E0-ENV-01` | done | DriveStudio环境（见 `OCCGS_E0_ENV_MANIFEST.md`） | rasterizer forward/backward ✅ |
| `G0-THIRDPARTY-00` | done | DriveStudio/SplatAD/Occ3D审计 | stack合法且单卡可行 ✅ |
| `D0-DATA-02` | done | 3-scene nuScenes子集 | 数据/box/mask完整 ✅ |
| `B0-RECON-03` | done | object-centric GS baseline | 3/3 scene可重建 ✅ |
| `O0-OCC-04` | done | occupancy/panoptic anchor | unknown保留；不依赖Occ3D ✅ |
| `S0-EDIT-05` | done | cut-in trajectory editor | 约束和provenance通过 ✅ |
| `C0-CF-06` | done | 无生成模型反事实渲染 | top-24机器合法24/24 ✅ |
| `L0-COMP-07` | done | 局部disocclusion补全 | outside-mask L1=0（协议）✅ |
| `U0-UTILITY-08` | partial | 下游数据效用 | proxy PASS；全量mAP未跑 |
| `D1-DECIDE-09` | done | 最终决策 | `modify_method_then_scale` |

---

# 14. 单卡与双卡策略

## 14.1 单卡阶段

以下必须单张4090完成：

```text
环境smoke
3-scene数据处理
B0 reconstruction
occupancy构建
单scene actor edit
C0 legality screening
L0小规模completion
```

## 14.2 双卡触发

只有以下同时满足：

- B0、O0、C0全部通过；
- 单卡瓶颈确认为吞吐而非显存；
- 方法对naive GS有明确收益；
- 需要扩到10+ scenes或并行下游实验；

才允许用户停机切双卡。

双卡优先用于：

```text
两个scene并行训练
两个baseline并行
多个下游seed
```

不默认DDP。

---

# 15. 资源预算与停止规则

## 15.1 初始规模

```text
3 scenes
3 cameras
8 seconds
10 Hz
vehicles only
```

## 15.2 禁止提前扩展

不在以下通过前：

- 下载Waymo/PandaSet；
- 下载全UniScene权重；
- 下载完整Occ3D（除非subset/annotation体积已审计）；
- 开启6 camera；
- 开启pedestrian SMPL；
- 训练diffusion；
- 做10+scene scale。

## 15.3 停止条件

### Environment

- CUDA extension持续无法编译；
- 单场景峰值显存>24GB且无官方轻量配置；
- 环境需破坏现有Motion-Proj。

### Data

- 10Hz标注/图像严重错位；
- 选定scene动态actor不足；
- 磁盘无法保留30GB。

### Reconstruction

- 3/3 scene均出现严重ghosting；
- object nodes无法稳定分解；
- held-out视角不可用；
- actor pose误差无法通过GT refinement修复。

### Occupancy

- anchor不减少任何一致性错误；
- occupancy噪声大于其约束收益；
- unknown/free混淆严重。

### Counterfactual

- 轨迹合法但render不遵循；
- source footprint/disocclusion仍非法；
- 未编辑区域变化过大；
- 人工合法率不足。

### Completion

- outside-mask变化；
- depth/instance冲突；
- temporal identity漂移。

### Utility

- 不超过naive GS；
- 收益只来自数据量；
- 标签噪声抵消long-tail收益。

---

# 16. 工程与Git规范

## 16.1 第三方仓库

固定：

```text
repo URL
commit
submodule commits
license
environment lock
```

不修改其历史。

自定义patch：

```text
/root/autodl-tmp/motion_proj/patches/drivestudio/
```

或独立fork branch；每个patch有原因和测试。

## 16.2 Run规范

每个run：

```text
manifest.json
resolved.yaml
git/third-party fingerprints
scene IDs
camera/time range
dataset checksums
environment
metrics.jsonl
summary.json
COMPLETE / FAILED / REJECTED
```

## 16.3 正式输出

```text
RGB
depth
semantic
instance
occupancy
boxes
trajectory
uncertainty
provenance
```

## 16.4 Commit建议

```text
docs(resim): 启动OccGS反事实重仿真路线
chore(env): 固定DriveStudio单卡环境
data(resim): 准备nuScenes三场景基线
research(gs): 建立object-centric Gaussian基线
feat(occupancy): 绑定panoptic occupancy与Gaussian节点
feat(resim): 增加约束轨迹编辑与多模态重渲染
research(completion): 验证局部disocclusion补全
research(data): 评估长尾合成数据效用
docs(resim): 固化最终路线决策
```

不自动push。

---

# 17. 输出文档

必须生成：

```text
docs/OCCGS_RESIM_AUTORESEARCH_PLAN_V7.md
docs/OCCGS_THIRD_PARTY_AUDIT.md
docs/OCCGS_ENVIRONMENT.md
docs/OCCGS_DATA_PREPARATION.md
docs/OCCGS_RECONSTRUCTION_BASELINE.md
docs/OCCGS_OCCUPANCY_STATE.md
docs/OCCGS_COUNTERFACTUAL_PROTOCOL.md
docs/OCCGS_FINAL_REPORT.md
```

更新：

```text
docs/RESEARCH_STATUS.md
docs/RESEARCH_FAILURES.md
docs/EXPERIMENTS.md
docs/THIRD_PARTY.md
docs/ARTIFACT_RETENTION.md
```

---

# 18. 最终论文实验矩阵

## Reconstruction baselines

```text
Static GS
Street-Gaussians-style
OmniRe
```

## State constraints

```text
No occupancy
Box-only occupancy
LiDAR/map occupancy
Occ3D-assisted occupancy（可选）
```

## Editing

```text
Naive actor transform
Kinematic-only
Kinematic + road
Full occupancy-certified
```

## Completion

```text
No completion
Image inpainting
Video completion
Occupancy-guided localized completion
```

## Utility

```text
Real only
Real + naive GS
Real + OccGS
Real + OccGS + completion
```

核心消融必须回答：

1. Occupancy是否真的提高编辑合法性？
2. Panoptic binding是否提高标签同步？
3. Localized completion是否只改变unknown区域？
4. 合成数据是否优于naive GS augmentation？

---

# 19. Reviewer 2 最可能的攻击

1. **“这只是DriveStudio/Street Gaussians加几个规则。”**  
   必须用panoptic occupancy binding、visibility consistency和下游utility证明方法价值。

2. **“Occupancy来自GT，不能叫world model。”**  
   准确称为controllable resimulation；GT state是scenario specification，不声称free-form forecasting。

3. **“已有UniSim可以添加和移动车辆。”**  
   必须比较多模态标签一致性、occupancy certificate和localized completion。

4. **“Occupancy只是后处理filter。”**  
   需要证明它进入actor node transform、ray visibility、label regeneration和completion mask。

5. **“3DGS新视角仍有ghosting。”**  
   先限制edit support，再用显式disocclusion mask做局部补全。

6. **“生成数据没有实际价值。”**  
   必须有下游长尾任务，不只报PSNR。

7. **“只在3个scene上。”**  
   3 scene只做feasibility；论文阶段必须扩到scene-disjoint规模和多seed。

8. **“nuScenes 2Hz/插值不可信。”**  
   所有真实和插值状态明确区分，评估使用真实keyframe/held-out camera作为锚点。

---

# 20. 执行范围与自主推进

> **已被 §A0 覆盖为自主连续执行。** 用户已授权：不再“第一轮结束即停机待审”，而是按 §13
> 顺序自动推进，直到命中 §A0.4 终止条件后自行关机。下述里程碑内的“禁止提前扩展”仍然有效。

自主执行仍必须遵守（与 §15.2 一致）：

```text
不download Occ3D全量（除非subset/annotation体积已审计）
不download Waymo/PandaSet
不download全UniScene权重
gate未过不实现下游occupancy方法/actor edit/diffusion
不开6 camera、不开pedestrian SMPL
不切双卡（除非§14.2条件全满足→按§A0.3暂停记录）
不扩到10+scene正式规模
```

每个 gate 完成后（无论 pass/fail）必须按 §A0.2 回填，并在 §A0.5 记录：

```text
DriveStudio commit和license / 环境版本 / CUDA extension状态
nuScenes raw路径与scene清单 / 3个选定scene及理由 / 处理数据大小
单场景smoke显存/时间 / 首个render和失败项 / 下一Gate
```

---

# 21. Autoresearch执行顺序

自主 agent 按以下顺序连续执行（E0 已完成，从 G0 起）：

```text
[恢复协议 §A0.6] 读取V7头部+§A0进度日志+§13里程碑与RESEARCH_STATUS
→ 读取Motion-Proj状态与RF账本
→ G0 第三方一手审计（固定commit/license/单卡可行）
→ (E0 已完成，跳过；损坏才重建)
→ D0 扫描nuScenes scene → 冻结S0/S1/S2 → 处理scene → 数据完整性可视化
→ B0-1 单相机4秒baseline → B0-2 三相机8秒 → B0-3 动态 → B0-4 repeatability
→ O0 occupancy/panoptic anchor
→ S0 cut-in trajectory editor
→ C0 无生成反事实渲染（人工gate）
→ L0 局部disocclusion补全（可选，C0过后）
→ U0 下游数据效用
→ D1 最终决策与终报
→ [§A0.4] 收尾commit + 关机
```

每过一个 gate 都按 §A0.2 回填。gate 通过则自动进入下一 gate；gate 失败按 §15/§16 归档并进入
§A0.4 终止流程。全流程走完（或命中停止条件/blocker）后写终报、commit、执行 `/usr/bin/shutdown -h now`。

---

# 22. 主要外部参考

```text
DriveStudio / OmniRe:
https://github.com/ziyc/drivestudio

Street Gaussians:
https://github.com/zju3dv/street_gaussians

SplatAD:
https://github.com/carlinds/splatad

neurad-studio:
https://github.com/georghess/neurad-studio

UniScene:
https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation

Occ3D:
https://github.com/Tsinghua-MARS-Lab/Occ3D

OpenOccupancy:
https://github.com/JeffWang987/OpenOccupancy
```

所有实现和引用必须优先查看论文原文与官方代码，不从二手博客复制关键算法。
