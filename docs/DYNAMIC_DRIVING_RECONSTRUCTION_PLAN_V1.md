# 动态驾驶场景重建与反事实编辑研究计划 V1

- 版本：V1
- 日期：2026-07-27；最终更新：2026-07-29
- 当前里程碑状态：`research terminal / M7 rejected`；M4 done，M5 blocked，M6 done，M8/M9 未授权
- 当前授权：本轮只允许证据闭合；禁止在 M7 novelty 失败后启动 M8 方法/消融或 M9 盲审
- 主基线：AD-GS exact reproduction
- 前馈对照：DGGT inference-only
- 编辑参考：DrivingEditor
- 必查几何对照：VAD-GS（仅当后续创新涉及补密/可见性/缺失几何）
- 研究主张候选：`rejected`；身份/actor binding/轨迹编辑主机制与现有工作直接重合
- 历史路线：nuScenes cut-in mining 已 `rejected / frozen`

## 0. 一页执行摘要

新路线不再先找 cut-in，也不再用事件池数量决定项目能否开始。执行顺序固定为：

```text
官方协议与资产闭合
→ AD-GS scene-0230 exact smoke
→ AD-GS 官方六场景 exact reproduction
→ DGGT 同片段推理级对照
→ 对象重建、轨迹编辑、遮挡/去遮挡与噪声压力测试
→ 只根据跨场景可复现失败选择一个创新假设
→ matched ablation、下游感知评测与盲审
```

实际执行在 M7 novelty gate 结束：M8/M9 未授权，最后一行没有启动。完整终态见第 17–19 节。

在 AD-GS 六场景复现门禁通过前，禁止集成 Motion-Proj/StreetGS/OccGS、occupancy、扩散、物理约束、感知损失或
轨迹编辑。cut-in 最多在方法成熟后作为一个演示，不再要求官方占比或事件召回率。

预注册时的候选论文问题是：

> 当对象轨迹被反事实修改后，如何显式重算遮挡/去遮挡，在观测不足区域给出可校准的真实性与拒绝信号，并在非目标
> 区域保持跨视角、跨时间和下游感知一致性？

M6 证明冻结身份不可用，M7 又证明候选核心机制缺少独立 novelty；因此它没有成为本轮贡献。

## 1. 路线裁决与失败继承

### 1.1 旧路线最终裁决

- nuScenes 官方数据结构没有事件级 cut-in 真值或公开总体占比，不能构造可信召回率分母。
- 冻结 strict-v2 在 675 个 prospective scenes 上只有 `1 PASS / 1 scene`。
- 继续调阈值主要增加事件挖掘、地图、接收车和审核工程，而不是动态重建或反事实编辑研究。
- 准确结论是“当前 cut-in 可验证事件池过稀”，不是“nuScenes 没有 cut-in”。

历史证据与清理记录：

- [`archive/2026-07/cutin-mining-closed/README.md`](archive/2026-07/cutin-mining-closed/README.md)
- [`archive/2026-07/cutin-mining-closed/CLEANUP_MANIFEST.md`](archive/2026-07/cutin-mining-closed/CLEANUP_MANIFEST.md)
- [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md) 中 `N1-F24`、`N1-F26`、`PIVOT-F01` 至 `PIVOT-F05`

### 1.2 新路线必须继承的失败约束

| 既有记录 | 对新路线的硬约束 |
|---|---|
| `RF-05/06` | 先证明监督和比较对象存在；不能用代理信号循环证明自己 |
| `RF-08/09` | 必须有 matched baseline、全分布、coverage、per-scene 与 worst-case |
| `RF-16/18` | RGB 局部差异和工程可运行不等于几何、时序或下游收益 |
| `V7-RISK-03/04` | hard composition 与极端负例只能做 smoke，不能证明 completion 或任务效用 |
| `V7-RISK-06` | 不得只挑成功场景；官方六场景结果必须全部报告 |
| `V7-RISK-07/17` | RGB、depth、mask、instance、pose 必须有 typed provenance，不能混写 truth tier |
| `V7-RISK-10/16` | UNKNOWN/未观测必须保留为显式状态；不得靠阈值变成 PASS |
| `N1-F24/PIVOT-F05` | 内存或 GPU 不足立即停机、留证据、等用户授权，不死磕 |
| `N1-F26/PIVOT-F01` | 不再恢复 cut-in threshold tuning |
| `PIVOT-F03` | AD-GS exact reproduction 前禁止集成式改进 |
| `PIVOT-F04` | visibility-aware densification 和基本对象平移已有先例，不能重命名为新贡献 |

## 2. 调研结论：为什么这样选基线

### 2.1 主基线：AD-GS

AD-GS 是 ICCV 2025 的自监督对象级动态驾驶场景高斯方法。官方实现公开了 nuScenes 的固定六场景、帧范围、
预处理脚本、训练入口和评测入口；方法用伪 2D 分割做对象/背景分解，以局部 B 样条加全局三角函数拟合对象运动，
并建模双向时间可见性。它不需要人工 3D 轨迹框，和本项目“对象级重建后再编辑”的目标最接近。

官方 nuScenes 协议：

- scenes：`0230, 0242, 0255, 0295, 0518, 0749`
- 每个 scene：camera chain 的第 `10..69` 帧，含首尾，共 60 帧
- cameras：`CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT`
- 分辨率：`900×1600`
- 测试：每 4 张取 1 张，严格沿用 upstream 实现
- 训练：官方配置默认 60,000 iterations
- 论文六场景均值：PSNR `31.06`、SSIM `0.925`、LPIPS `0.164`
- 论文硬件：单张 RTX 3090

选择理由不是“它永远是最新 SOTA”，而是它当前具备最完整、最可审计的 exact reproduction 路径，且对象级表示
可以自然暴露编辑失败。

### 2.2 前馈对照：DGGT

DGGT 从未标定位姿 RGB 直接单次前向预测相机、动态 3D Gaussian 和生命周期，不做逐场景优化。官方代码、训练代码
和 nuScenes 权重已经公开，适合回答速度/泛化与对象可控性/精细几何之间的差异。

限制：

- 官方默认推理窗口较短，不能把它与使用完整 60 帧优化的 AD-GS 当成严格 matched leaderboard；
- 当前 `inference.py` 源码审计发现参数名 `difix`/`diffusion` 不一致，必须先过 upstream smoke；
- nuScenes checkpoint 约 5.41 GB，模型仓库为非商业许可；运行前必须做许可证与哈希登记；
- 本阶段只做 inference，不训练 DGGT，不把兼容性补丁当方法贡献。

### 2.3 编辑参考：DrivingEditor

DrivingEditor 将动态前景和静态背景分支解耦，在不依赖人工 3D 框的条件下展示对象删除、添加等编辑。官方代码主要
面向 Waymo/KITTI，环境为 Python 3.9、PyTorch 2.2.2、CUDA 11.8，数据接口不是 nuScenes。因此它首先用于定义
编辑能力与比较边界；只有在 AD-GS 复现后、且 nuScenes adapter 成本通过门禁时才运行。

### 2.4 2026 工作带来的 novelty 边界

- VAD-GS（CVPR 2026）已经公开代码并做 visibility-aware densification、跨相机/跨帧 MVS 和缺失几何补密。
  若后续主张涉及几何补密，它是强制对照，不得只和 naive 3DGS 比。
- DenoiseGS（AAAI 2026）已明确研究相机参数噪声和动态标注噪声，并用 B 样条优化轨迹；本计划因此把噪声压力测试
  设为必做项，不能把“对噪声更稳”当无对照的新发现。
- Perception-aware 3DGS（ICLR 2026）表明高 PSNR/SSIM 不保证感知输出一致；本计划将对象区与感知一致性列为
  primary/guardrail，而不是可选附录。
- ReconDrive 已公开 4.6 GB checkpoint，但其官方 README 在 2026-07-26 仍把 base-model code/config/checkpoint
  列为 TODO；因此只列入 watchlist，不替代当前可执行的 DGGT。
- Real2Sim（2026-05 arXiv）已展示对象级平移/旋转/复制和 MPM 物理交互，但也承认未见视角对象不完整、渲染与物理
  解耦。基本对象变换或“加入物理”本身不足以成为当前贡献。
- GA-GS 的生成区域真实性权重与本计划的“真实性/置信度”方向接近；若实现该方向，必须做逐模块差异审计。

## 3. 数据集与资产方案

### 3.1 主数据集：nuScenes 官方 AD-GS 六场景

本阶段不再做事件筛选。主数据单元就是 upstream 固定的六个 scene 和固定 60 帧。

选择 nuScenes 的理由：

- AD-GS 和 DGGT 都支持；
- 服务器已有 metadata、部分原始传感器文件和本地只读官方 tar shards；
- 现有 DriveStudio/StreetGS 基础可在 baseline 锁定后复用；
- 3 个前向相机足够对齐 AD-GS 论文协议。

nuScenes 官方提供对象类别、属性、3D 框、传感器和地图等标注，但没有事件级 cut-in 真值；本计划不再使用或推断
事件召回率。

### 3.2 当前资产缺口

2026-07-26 只读 metadata 审计结果如下。`expected/present` 是完整 scene sensor chain 的 metadata 条目数与当前
磁盘实际文件数，不代表 AD-GS 最终会复制全部条目。

| scene | CAM_FRONT | CAM_FRONT_LEFT | CAM_FRONT_RIGHT | LIDAR_TOP |
|---|---:|---:|---:|---:|
| 0230 | 234 / 40 | 230 / 0 | 230 / 0 | 387 / 40 |
| 0242 | 233 / 40 | 235 / 0 | 233 / 0 | 391 / 40 |
| 0255 | 233 / 40 | 233 / 0 | 233 / 0 | 390 / 40 |
| 0295 | 234 / 40 | 235 / 0 | 231 / 0 | 391 / 40 |
| 0518 | 235 / 41 | 237 / 0 | 235 / 0 | 395 / 41 |
| 0749 | 240 / 41 | 239 / 0 | 240 / 0 | 399 / 41 |

结论：当前 raw 主要只有 keyframe 资产；AD-GS 需要的左右前相机和中间 sweep 不完整。不能直接开训练。

2026-07-27 的 M2 已在版本化目录中闭合该缺口：六场景各 180 RGB + 60 个最近 LiDAR，共 1,440 个唯一
sensor payload；另补齐 nuScenes devkit 初始化必需的 4 个静态 map masks。所有文件均为非 symlink、非空且
SHA-256 与 manifest 一致。旧 `/root/autodl-tmp/data/nuscenes` 仍未原地改写。

### 3.3 选择性提取，不复制 294 GB 全量 tar

本机只读公共目录：

```text
/autodl-pub/data/nuScenes/Fulldatasetv1.0/Trainval/
```

其中有 `v1.0-trainval01_blobs.tgz` 至 `v1.0-trainval10_blobs.tgz`，合计约 294 GB。执行阶段采用：

1. 只从 metadata 解析六场景、三相机 `10..69` 帧及其最近 LiDAR 的精确相对路径；
2. 写出 `required_members.txt`、scene/frame/sensor 对照表和 SHA-256；
3. 流式扫描 tar 目录索引，建立 `member → shard` 映射；
4. 只提取清单成员到版本化数据目录，不全量解压；
5. 校验每 scene 应有 180 张 RGB、对应 LiDAR、时间戳单调、标定与 pose 可解析；
6. 原始数据只读，所有派生文件写入新目录。

禁止用缺失的 side cameras 静默降级为单相机，也禁止降低分辨率后与论文指标直接比较。

### 3.4 数据目录

```text
/root/autodl-tmp/data/dynamic_recon/
├── manifests/
│   ├── adgs_nuscenes_v1_required_members.txt
│   ├── adgs_nuscenes_v1_member_shards.tsv
│   └── adgs_nuscenes_v1_manifest.json
├── raw_subset/adgs_nuscenes_v1/
└── processed/adgs_nuscenes_v1/
    ├── scene-0230/
    ├── scene-0242/
    ├── scene-0255/
    ├── scene-0295/
    ├── scene-0518/
    └── scene-0749/
```

现有 `/root/autodl-tmp/data/nuscenes` 不原地改写。

### 3.5 暂不使用的数据

- WOD-E2E：保留为以后长尾视频/策略评测候选，不作为对象级 3DGS 首个数据源。
- Waymo：只有当 DrivingEditor/VAD-GS 必须做正式可运行对照时再申请下载或使用已授权资产。
- nuScenes cut-in pool：只作历史回归与可选 demo，不进入主实验选择。
- 自选“动态更强”scene：禁止在看过方法输出后替换官方六场景。

## 4. 环境配置与复现锁定

### 4.1 隔离原则

不污染现有 `motionproj` 环境。upstream 代码放在 `/root/autodl-tmp/third_party/`，环境放在
`/root/autodl-tmp/envs/`，每个仓库固定 commit。

```text
/root/autodl-tmp/third_party/AD-GS
/root/autodl-tmp/third_party/Depth-Anything-V2
/root/autodl-tmp/third_party/Grounded-SAM-2
/root/autodl-tmp/third_party/dggt
/root/autodl-tmp/third_party/VAD-GS        # 条件启用
/root/autodl-tmp/third_party/DrivingEditor # 条件启用

/root/autodl-tmp/envs/adgs
/root/autodl-tmp/envs/adgs-dpt
/root/autodl-tmp/envs/adgs-sam
/root/autodl-tmp/envs/dggt
/root/autodl-tmp/envs/vadgs                # 条件启用
```

### 4.2 AD-GS 官方环境

先原样尝试 upstream `environment.yaml`：

- Python `3.7.16`
- PyTorch `1.13.1`
- torchvision `0.14.1`
- torchaudio `0.13.1`
- CUDA 11.7 runtime packages
- COLMAP `3.7`
- nuscenes-devkit `1.1.10`
- Open3D `0.17.0`
- 自定义 `simple-knn`
- 自定义 `depth-diff-gaussian-rasterization`
- PyTorch3D

伪监督使用独立环境：

- Depth Anything V2 Large：Python `3.11`，固定 checkpoint SHA-256；
- Grounded-SAM-2：Python `3.10`，`CUDA_HOME` 指向 12.1 toolkit；
- CoTracker3：固定本地 repo commit 与离线权重哈希，禁止运行时追随在线 `main`；
- COLMAP/SfM：记录 COLMAP 版本、命令行和数据库哈希。

若 Python 3.7/旧包 channel 已不可求解，只允许最小 compatibility patch：

1. 保存原始求解失败日志；
2. 新建 `compatibility/AD-GS-<date>.patch`；
3. 只改安装/ABI，不改损失、数据、模型或评测；
4. patch 单独 commit，并在 baseline 报告中标记；
5. 无法证明等价时 exact reproduction 状态为 `blocked`，不得进入方法研究。

### 4.3 DGGT 环境

- Python `3.10`
- PyTorch `2.4.1`
- torchvision `0.19.1`
- torchaudio `2.4.1`
- 编译 `third_party/pointops2`
- 首轮只下载 nuScenes 主 checkpoint；不开 DiFix
- 只有 interpolation 实验需要时才下载 TAPIP3D

源码 smoke 必须先验证：

- `--help` 与 import；
- `difix`/`diffusion` 参数名；
- checkpoint key 和模型结构；
- 1 view / 3 view、sequence length 4；
- 输入 resize 到 width 518 的真实行为；
- 不启用 diffusion 时是否仍强制 import/加载相关依赖。

### 4.4 可选环境

- VAD-GS：官方 dev branch，Python 3.8、PyTorch 1.12 + CUDA 11.3、COLMAP 3.10-dev；
- DrivingEditor：Python 3.9、PyTorch 2.2.2、CUDA 11.8、DEVA/Grounded-SAM、nvdiffrast。

可选环境只有在对应实验门禁解锁后才安装，避免一次构建五套栈。

### 4.5 每个环境必须输出

```text
repo_url
git_commit
git_diff
submodule_commits
conda_explicit.txt
pip_freeze.txt
python_version.txt
torch_cuda_versions.txt
nvcc_version.txt
gcc_version.txt
gpu_driver.txt
checkpoint_sha256.txt
license_audit.md
smoke.log
```

AD-GS 仓库当前未在顶层清晰展示许可证文件，执行前必须确认代码再分发边界；nuScenes 数据遵守
CC BY-NC-SA 4.0 与附加条款。DGGT 代码与模型许可证分别登记，不能混写。

## 5. 资源合同

### 5.1 本轮合同

- GPU：`NVIDIA GeForce RTX 3090, 24576 MiB`，driver `580.105.08`；
- `memory.max=96,636,764,160` bytes（90 GiB）；
- 当前换机 M2 实例峰值 `memory.current=9,685,876,736` bytes；
- scene-0242 1,000-step train/render 峰值
  `23,832,678,400 / 25,567,031,296` bytes；
- 当前实例截至 1,000-step 全程 `oom=0 / oom_kill=0`；
- 数据盘在新合同启动时约 141 GiB 可用。

该合同满足 24 GB GPU、至少 80 GiB cgroup 和 60 GiB 启动磁盘门槛，允许恢复 scene-0242 并继续串行 M4。

### 5.2 下一轮最低申请

- GPU：1× RTX 4090 24 GB 或不低于论文 RTX 3090 24 GB 的单卡；
- 系统内存：最低 32 GB，推荐 64 GB；
- CPU：至少 8 cores，COLMAP/解压阶段推荐 16 cores；
- 磁盘：当前约 67 GiB 可用；启动前要求可用空间 ≥60 GiB，并始终保留 20 GiB 安全余量；
- 运行时：先给 scene-0230 smoke 窗口，完整六场景预算由 smoke 实测外推后再确认。

### 5.3 停机条件

任一条件触发即保存现场并等待用户：

- cgroup `memory.current / memory.max ≥ 0.90` 持续两个采样周期；
- `memory.events` 的 `oom`/`oom_kill` 增加；
- RC 137、SIGKILL、CUDA OOM；
- GPU 峰值显存超过可用量；
- 磁盘可用空间低于 20 GiB；
- 需要降低官方分辨率、删相机、缩短正式 split 才能继续。

禁止杀 Cursor/Jupyter/TensorBoard 等用户服务，禁止静默降配后继续声称 exact reproduction。

## 6. Run contract 与目录

所有新 run 使用稳定 ID 和唯一实例目录：

```text
/root/autodl-tmp/runs/dynamic_recon/<RUN_ID>/<instance_id>/
├── manifest.json
├── resolved.yaml
├── environment/
├── inputs.json
├── stdout.log
├── stderr.log
├── metrics.json
├── artifacts.json
├── summary.md
└── terminal.json
```

`terminal.json.status` 只允许：

```text
pending | running | blocked | done | rejected
```

每个 run 必须记录 commit、upstream commit、config fingerprint、数据 manifest hash、seed、GPU、峰值 RAM/VRAM、
wall time、退出码、产物字节数与 SHA-256。失败 run 不覆盖，重跑使用新 instance ID。

## 7. 里程碑与实验注册表

| 里程碑 | Run ID | 状态 | 交付物 | 解锁条件 |
|---|---|---|---|---|
| M0 旧路线封存与清理 | `DR-M0-ARCHIVE-01` | done | 归档、清理清单、恢复路径 | 保留项验证、无 OOM |
| M1 官方调研与方案 | `DR-M1-PLAN-01` | done | 本计划、失败追加、人工审核包 | 用户审核并开放资源 |
| M2 环境与资产 smoke | `DR-M2-ENV-ASSET-01` | done | 锁定环境、六场景精确资产 manifest | 所有结构门禁通过 |
| M3 AD-GS 0230 | `DR-M3-ADGS-0230-01` | done | 预处理、60k 训练、渲染、指标、资源画像 | pipeline 完整、无协议修改 |
| M4 AD-GS 六场景 | `DR-M4-ADGS-6SCENE-01` | done | 六场景 per-scene/aggregate 结果 | 三项论文复现带宽全过 |
| M5 DGGT 推理对照 | `DR-M5-DGGT-NUSC-01` | blocked | native 与 common-input 诊断报告 | 明确 upstream packaging blocked 证据 |
| M6 编辑/噪声压力测试 | `DR-M6-STRESS-01` | done | failure matrix、typed metrics、完整 ABSTAIN coverage | 6/6 scenes 持久身份失败 |
| M7 创新假设预注册 | `DR-M7-HYPOTHESIS-01` | rejected | 唯一候选 A 与官方 novelty matrix | 与 InstDrive 等直接重合 |
| M8 方法与消融 | `DR-M8-METHOD-01` | rejected | 未授权、0 seeds、0 proposed metrics | M7 novelty gate 失败 |
| M9 人工审核与最终包 | `DR-M9-HUMAN-01` | rejected | 未触发说明、machine termination pack | 不存在可盲审的 M8 方法结果 |

每完成一个里程碑，必须在同一工作回合更新本表、[`RESEARCH_STATUS.md`](RESEARCH_STATUS.md) 和
[`EXPERIMENTS.md`](EXPERIMENTS.md)，再决定下一步。不得一次越过两个未通过门禁。

## 8. M2：环境、资产和伪监督闭合

M2 冻结环境、权重、原始资产和各伪监督模型的最小 GPU smoke；scene-0230 的整段 depth/mask/flow/COLMAP
生成属于 M3 的预处理 stages。这个执行拆分不改变 upstream 顺序、输入、分辨率或模型，只避免在原始资产门禁通过前
生成不可审计的派生产物。

### 8.1 环境 smoke

顺序固定：

1. 记录 GPU/driver/CUDA/gcc/cgroup/disk；
2. clone 并 pin AD-GS commit；
3. 原样创建 AD-GS 环境；
4. 编译两个 rasterizer 和 simple-knn；
5. 运行 import、单 Gaussian forward/backward、LPIPS 与 PyTorch3D smoke；
6. 分别创建 DPT、SAM 环境并做单图 smoke；
7. 只有以上通过后才提取 scene-0230。

### 8.2 数据结构门禁

每个 scene：

- RGB 文件计数、分辨率、时间戳、相机顺序；
- camera pose 与 intrinsics 都是 finite；
- 最近 LiDAR 匹配时间差分布；
- train/test 索引与 upstream 逐项一致；
- M3 `prepare_raw` 后生成的 `meta.npz`、`points3d.ply` 可解析；
- 不允许 symlink 指向会被清理的旧 N1 cache。

### 8.3 伪监督流水线

按 upstream 顺序：

```text
Depth Anything V2 depth
→ Grounded-SAM-2 object/sky masks
→ segmented point cloud
→ CoTracker3 flow
→ COLMAP SfM
```

结构门禁：

- 每个输入都有对应输出；
- shape、dtype、坐标系、时间戳和 camera ID 一致；
- 无 NaN/Inf，mask ID 不冲突；
- flow 正反向索引一致；
- COLMAP 注册相机数与预期一致；
- 产物哈希与生成命令完整。

诊断抽样固定使用每 scene 的帧索引：

```text
10, 18, 26, 34, 42, 50, 58, 66
```

三相机全部展示，共 144 个 frame-camera 样本。人工诊断只标记明显空 mask、漏掉大车、ID 断裂、depth 崩溃或
COLMAP 失配；不得根据诊断结果手修某个 scene 的 mask 后仍称 exact。

## 9. M3/M4：AD-GS exact reproduction

### 9.1 scene-0230 小步执行

1. 100-iteration smoke：验证 forward/backward、checkpoint、render、metrics；
2. 1,000-iteration profile：记录 VRAM、RAM、单迭代时间、checkpoint 增长；
3. 资源投影通过后运行官方 60,000 iterations；
4. 用官方命令渲染 validation/test；
5. 保存全帧指标、对象区诊断、视频和资源曲线。

100/1,000 iteration 只用于工程画像，不进入论文指标。

### 9.2 scene-0230 通过条件

官方没有公开 scene-0230 单场景目标值，因此本门禁只判断协议完整性：

- 60,000 iterations 正常终止；
- 输入为官方 scene/帧/三相机/原分辨率；
- 训练、checkpoint、render、metrics 产物完整；
- 无 compatibility patch 改变模型或指标；
- 随机抽取帧没有全黑、全空、严重错相机或静态/动态分支完全崩溃；
- 资源画像可支持继续六场景。

不得用 scene-0230 的单点 PSNR 宣称复现论文。

### 9.3 六场景数值门禁

严格使用官方六场景均值。预注册工程复现带宽：

| 指标 | 论文值 | 通过带宽 |
|---|---:|---:|
| PSNR ↑ | 31.06 | `≥ 30.56` |
| SSIM ↑ | 0.925 | `≥ 0.915` |
| LPIPS(VGG) ↓ | 0.164 | `≤ 0.184` |

三个指标必须同时通过，且报告六个 per-scene 值、均值、最差场景和缺失率。该带宽只定义“足够接近以继续研究”，
不允许把落在带宽内表述为逐位复现。

若失败：

1. 只检查资产、upstream commit、官方 config、metric backbone、相机顺序和 compatibility patch；
2. 最多做一次有明确根因的重跑；
3. 仍失败则 M4=`blocked`，提交复现失败包，等待用户决定；
4. 不调模型超参、不删困难 scene、不降分辨率、不直接开始创新。

## 10. M5：DGGT 推理级对照

DGGT 只回答“前馈速度/泛化换来了什么、失去了什么”，不做不公平的统一排行榜。

### 10.1 两层协议

`native protocol`：

- 使用官方 nuScenes checkpoint、默认 sequence length 4；
- 先 1-view，再在显存允许且 checkpoint 支持时做 3-view；
- diffusion 关闭；
- 报告官方 resize、输入帧、预测 pose、PSNR/SSIM/LPIPS、速度和 VRAM。

`common-observation diagnostic`：

- 从同一 AD-GS 六场景和同一 60 帧中取固定窗口；
- AD-GS 使用已完成的场景模型渲染相同 target；
- DGGT 只看到窗口规定的 RGB；
- 显式报告两者实际输入帧数、是否用 pose、是否逐场景优化；
- 结果只用于 failure characterization，不写“同预算优于”。

### 10.2 固定窗口

每 scene 使用三个不重叠 4-frame 窗口：

```text
[10,11,12,13]
[34,35,36,37]
[66,67,68,69]
```

如果 upstream 的 frame index 在预处理后重新编号，adapter 必须保存 raw↔processed 映射，不得事后换窗口。

### 10.3 对照维度

- wall time、GPU time、peak VRAM；
- camera translation/rotation error（有官方 pose 真值时）；
- 全图与对象区 PSNR/SSIM/LPIPS；
- depth/flow consistency；
- 动态对象边界、生命周期与拖尾；
- 是否存在可分离、可直接编辑的对象表示；
- 跨窗口 identity 与时间连续性。

若 DGGT 因官方代码 bug 需要 patch，原始失败和 patch 后结果分开；若 24 GB 仍 OOM，按资源规则停止，不静默缩小
窗口后替代正式结果。

## 11. M6：压力测试

### 11.1 对象选择

不使用 cut-in 语义。每 scene 从训练完成前冻结的 pseudo-object tracks 中选两个车辆：

- `high-support`：满足有效轨迹后，可见帧数最高；
- `boundary-support`：满足最低有效轨迹后，可见帧数最低。

最低有效轨迹：

- 在 60 帧中同一 ID 有效至少 20 帧；
- 至少一个相机中可见至少 10 帧；
- native resolution 下中位 mask area ≥500 px；
- 不存在同帧 ID 冲突。

不足两个时如实记 coverage，不用下一名“看起来更好”的对象顶替。

### 11.2 原始重建压力

对全部六场景、全部有效对象报告：

- 轮廓重影与边界 LPIPS；
- 拖尾、车体/车轮拉伸；
- 遮挡后重现的 ID 连续性；
- 对象高斯污染静态背景；
- 刚体 pairwise-distance drift；
- 多相机深度排序和 mask 一致性。

### 11.3 参数化轨迹编辑

每个对象使用 actor-local 坐标，语义中性的固定操作：

| 编辑 | 参数 |
|---|---|
| 横向平移 | `+0.5 m, +1.0 m, +1.5 m` |
| 沿原轨迹提前/延后 | `Δt=-0.5 s, +0.5 s` |
| 速度缩放 | `0.75×, 1.25×` |
| 局部停止再起步 | 中段停止 `1.0 s`，随后平滑恢复 |
| 删除 | 全轨迹删除 |

横向正方向由 actor-local `+y` 定义，不声称道路合法、cut-in 或 merge。操作超出证据支持时不能偷偷裁剪，必须输出
support/confidence/ABSTAIN。

检查：

- 对象是否沿新轨迹稳定；
- 几何与外观是否保持刚体；
- 原位置是否残影；
- 新位置的遮挡和深度排序是否更新；
- 阴影是否与对象解耦或留下残影；
- 新暴露背景是否跨视角/时间一致；
- 非目标区域是否保持。

### 11.4 去遮挡真值分层

反事实视频没有天然真值，因此禁止只靠主观视频或自洽规则评分。

| Truth tier | 定义 | 允许指标 |
|---|---|---|
| A：held-out observed | 同一静态表面在另一时间/相机有真实观测，并通过 pose/depth 对齐 | RGB、LPIPS、depth、perception |
| B：geometric support | 有几何/多视角支持，但没有同视角真实 RGB | depth、reprojection、temporal consistency |
| C：unsupported | 没有可验证观测 | uncertainty、risk-coverage、ABSTAIN、人审 |

构造 `pseudo-hole benchmark`：

1. 用真实动态对象 footprint 或合成 occluder 定义洞区；
2. 从训练输入中隐藏本可见的静态区域；
3. 保留另一个时间/相机的真实观测只作评测；
4. 比较 no-completion、2D framewise diagnostic、observed-only Gaussians 和候选 3D completion；
5. 2D inpaint 永远只作诊断，不作为世界状态真值。

### 11.5 噪声压力测试

先在固定前三个官方 scenes `0230/0242/0255` 做 one-factor-at-a-time：

| 噪声 | 级别 |
|---|---|
| camera translation | σ=`0.02/0.05/0.10 m` |
| camera rotation | σ=`0.1/0.3/1.0°` |
| mask dropout | `5/10/20%` frames |
| mask erosion/dilation | `3/7/15 px` |
| flow noise | σ=`1/3/5 px` |
| ID switch | 连续 `1/3/5` frames |
| prior missing | `5/10/20%` frames |

另做一个预注册 medium-combined corruption。噪声 seed 固定；每个级别至少 3 seeds。若以后声称鲁棒性，扩到六场景。

## 12. 评测矩阵

### 12.1 重建质量

- 全图：PSNR、SSIM、LPIPS(VGG/Alex，分开命名)；
- 对象区：object-mask PSNR/SSIM/LPIPS、boundary LPIPS；
- 背景：static-region 指标、动态污染率；
- 几何：LiDAR/held-out depth MAE、reprojection error；
- 时序：flow-warp error、temporal LPIPS、flicker spectrum；
- 对象：rigidity drift、track continuity、ID switches。

### 12.2 编辑质量

- target-region expected change；
- non-target RGB/LPIPS preservation；
- 原 footprint residual；
- 新 footprint occlusion correctness；
- Tier-A disocclusion RGB/depth；
- Tier-B cross-view/temporal reprojection；
- Tier-C uncertainty、coverage、AURC/risk-coverage；
- edit magnitude 对失败率的曲线，不只报平均值。

### 12.3 下游感知一致性

先审计并 pin ICLR 2026 Perception-aware 3DGS 官方 evaluator。若其模型/数据接口不能直接复用，选择一个冻结、
公开、可哈希的 2D vehicle detector 与 tracker，且在看结果前写入 resolved config。

报告：

- 原图与重建图的 detector box matching IoU；
- 类别置信度差；
- recall/false disappearance；
- tracker IDF1 与 ID switches；
- 编辑目标的预期变化；
- 非目标对象输出保持率。

感知模型输出只能作为一个 task-aligned evaluator，不是真实世界安全证书。

### 12.4 统计与报告

- exact reproduction 使用 upstream seed，并记录确定性设置；
- 方法比较与噪声实验使用至少 3 seeds；
- 同 scene、frame、camera、actor、edit magnitude、seed 和预算 matched；
- 报 mean/std、bootstrap 95% CI、per-scene、per-object、worst-case、coverage；
- UNKNOWN/ABSTAIN 单独报告，不从分母中消失；
- 任何缺失 run 进入 coverage，不只展示成功视频。

## 13. M7：如何根据失败选择唯一创新

M6 完成前不写方法代码。用以下决策表：

| 稳定失败 | 候选主假设 | 必须对照 |
|---|---|---|
| 原轨迹内也出现刚体/身份/生命周期错误 | A：可编辑运动表示与轨迹不确定性 | AD-GS B-spline、DenoiseGS 边界 |
| 移动物体后原位置洞区和新遮挡错误最稳定 | B：编辑诱发可见性重算 + 置信度高斯 | no completion、VAD-GS、GA-GS 边界 |
| RGB 改善但 detector/tracker 明显退化 | C：感知保持的反事实编辑 | Perception-aware 3DGS evaluator/loss |
| DGGT 快但几何粗，AD-GS 好但慢 | D：前馈初始化 + 约束精修 | DGGT、ReconDrive watchlist、AD-GS |
| 没有跨 ≥3 scenes 的一致失败 | 不立方法 claim | 扩充诊断或停止 |

当前推荐优先级是 `B + C`，但只有当两类失败在至少 3 个官方 scenes、high/boundary-support 两类对象上重复出现时才
允许合并成一个主假设。若失败只出现在单场景，不得以 cut-in 或“长尾”命名放大。

## 14. 候选方法框架与 matched ablation

若 M7 选择 `B + C`，候选框架暂名：

> Uncertainty-Aware Object-Centric Counterfactual Gaussian Re-Simulation

模块候选：

1. **Editable trajectory state**：保留原 AD-GS B-spline，编辑只施加显式 SE(3)/时间重参数残差；
2. **Visibility recomputation**：从同一 world state 重算 source footprint、edited footprint、first-hit depth 与相机可见性；
3. **Evidence-typed background Gaussians**：
   - observed-real；
   - multi-view-supported；
   - generated-unsupported；
4. **真实性/置信度**：每个 Gaussian/像素输出 evidence tier 与 uncertainty，允许 ABSTAIN；
5. **Perception preservation**：只在非目标区约束冻结感知特征/输出；目标区约束预期变化；
6. **跨视角/时间一致性**：生成内容不能逐帧独立成为世界状态。

最小消融：

```text
A0 AD-GS original reconstruction
A1 AD-GS + naive trajectory transform
A2 A1 + visibility recomputation
A3 A2 + observed-only background completion
A4 A3 + confidence-aware generated Gaussians
A5 A4 + perception preservation
```

若研究点涉及缺失几何，加入 VAD-GS 或其可匹配子模块；若 DrivingEditor adapter 可在冻结预算内完成，加入其可执行
删除基线。2D Telea/逐帧扩散只作 diagnostic lower baseline，不能与 3D world-state 方法等价表述。

primary endpoint 在 M7 预注册，建议：

- Tier-A disocclusion LPIPS；
- non-target perception consistency；
- temporal warp error；
- risk-coverage/AURC；

guardrails：

- 官方重建 PSNR/SSIM/LPIPS 不显著退化；
- target object rigidity 与 ID 不退化；
- runtime/VRAM/coverage 完整报告。

## 15. 通过、拒绝与停止

### 15.1 进入方法研究的必要条件

- M4 AD-GS 六场景复现带宽三项全过；
- DGGT smoke 可运行或有明确 upstream blocked 证据；
- 压力测试在至少 3 scenes 重复同一失败；
- novelty 审计确认没有把 VAD-GS/DrivingEditor/Real2Sim/GA-GS 已有能力重命名；
- primary endpoint 有真实 held-out 或明确 truth tier。

### 15.2 方法通过条件

- 预注册 primary endpoint 在 3 seeds 上改善，95% CI 不跨过零或满足预注册 effect-size；
- 所有 guardrails 通过；
- 六场景和 worst-case 均报告；
- coverage 不因 ABSTAIN 被隐藏；
- 人工盲审不出现系统性身份、遮挡、残影或时序退化。

具体 effect-size 必须在看到 proposed 结果前，依据 M6 baseline 方差写入 M7；现在不凭空设百分比。

### 15.3 拒绝条件

- exact reproduction 未过却靠自定义 config 获得“更好”结果；
- 失败只在挑选场景或单个 actor 出现；
- 方法只提升全图 PSNR，目标区/感知/时序无改善；
- completion 只在 unsupported 区域看起来好，没有 Tier-A/B 证据或置信度；
- 降低 coverage 才改善均值；
- 与 VAD-GS/DrivingEditor/GA-GS 的实质差异无法说明；
- 资源不足却静默缩协议。

## 16. 人工审核设计

### 16.1 计划审核（本轮）

本轮人工包只审核：

- cut-in 是否正确封存且失败未删除；
- 删除项是否仅为可再生产物；
- baseline 选择、数据、环境、门禁和停止规则是否完整；
- 2026 新工作是否改变 novelty 边界；
- 下一轮资源申请是否合理。

人工结论由用户/指定评审填写，Codex 不代填。

### 16.2 结果审核（M9）

固定 blind pairwise 包：

- 六场景；
- high-support 与 boundary-support 两类对象；
- 原始、横移 1.0 m、速度 0.75×、stop/restart、删除五类；
- target crop、full frame、depth/ID overlay、三相机同步视频；
- method label 随机化；
- 每个样本评价身份、刚体、残影、遮挡、洞区、时序、非目标保持；
- `PASS / FAIL / UNCERTAIN` 与原因码；
- 全部样本和失败样本都保留。

## 17. 里程碑更新日志

### 2026-07-26 — M0 `done`

- 完整备份：`/root/autodl-tmp/motion_proj_backups/docs-before-direction-pivot-2026-07-26/`
- cut-in 文档移入 `docs/archive/2026-07/cutin-mining-closed/`
- 删除约 5 GiB 可再生缓存/中间 checkpoint/失败渲染副本和临时 worktree
- 保留 raw data、final checkpoints、正式指标、人工结论、失败记录
- 执行后 `oom=0 / oom_kill=0`
- 下一步：M1 官方调研、完整计划与人工审核包

### 2026-07-26 — M1 `done`

- 已完成 AD-GS、DGGT、DrivingEditor、VAD-GS、DenoiseGS、Perception-aware 3DGS、ReconDrive、Real2Sim 官方来源核对
- 已完成六场景本地资产缺口审计
- 文档导航、失败总账、实验台账和空白人工审核包已完成
- 下一步：停止计算，等待用户审核并开放 ≥32 GB RAM 与 24 GB GPU

### 2026-07-27 — M2 `done`

- 正式实例：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M2-ENV-ASSET-01/20260727T180733__e49a4e-4080s-r3/`
- 项目 commit：`2d46f4c1c79708451081f291a267a7acd26a3236`；未提交工作树由 fingerprint 和 source snapshots 固化
- AD-GS、DPT、Grounded-SAM-2、固定 revision 的 Grounding DINO HF 与固定 CoTracker3 权重 smoke 全部通过
- 六场景共 1,440 个 sensor payload 与 4 个 map masks 通过文件、哈希、尺寸、时间戳、标定、pose 和 LiDAR 审计
- 数据 manifest SHA-256：`64c68972a25834757168cd8fdc11c64b134b6ae0d9206a9ebde4064891c16092`
- AD-GS compatibility patch SHA-256：`efbed2eb888d2e77238e99ea84423435cc5d241b3dbb0dc55443e4967eb1c98a`
- 前一实例因遗漏 devkit 所需 map masks 以 `blocked` 保留；修复没有覆盖失败证据
- 正式实例峰值 cgroup memory `30,123,261,952` bytes，`oom=0 / oom_kill=0`
- 下一步：M4 依次执行其余五个官方 scenes，并与已完成的 scene-0230 聚合六场景指标

### 2026-07-27 — M3 `done`

- 第一个 blocked 实例：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M3-ADGS-0230-01/20260727T181617__scene0230__s0/`
- blocked 根因：AD-GS nuScenes prepare 固定输出 PNG，而 Grounded-SAM-2 video loader 只按 JPEG 扩展名枚举
- 第二个 blocked 实例：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M3-ADGS-0230-01/20260727T182247__scene0230__s0-r2/`
- 第二次 blocked 根因：upstream COLMAP 使用全部 128 个 CPU threads，feature extraction 峰值
  cgroup memory `62,265,835,520` bytes，连续两个采样高于 90% 停止线；`oom=0 / oom_kill=0`
- 当前正式实例：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M3-ADGS-0230-01/20260727T195611__scene0230__s0-r3/`
- 启动配置：scene-0230，frames 10..69，三相机 upstream 顺序，900×1600，seed 0
- r3 逐文件验证并复用 r2 的 860 个 pre-COLMAP 产物，复用指纹：
  `836b8f1480b083e5d9180f235e06d290c2ed7c5eb6f4de791d920fe945282891`
- compatibility patch 新增 byte-preserving `.jpg` staging alias，并只把 COLMAP 并发限制为 16 threads；
  不转码、不改像素、模型、SIFT 参数、匹配、损失或指标；patch SHA-256：
  `49b4c06ecec6c30f1e80b5abf4d46970920f9d71952acbda273774d9b5b34f48`
- COLMAP 已通过：138/138 images registered、70,933 points；阶段峰值 cgroup memory
  `35,117,174,784` bytes，`oom=0 / oom_kill=0`
- processed audit 已通过：180 images/depth/sky/semantic、138/138 flow；meta、points 与 COLMAP 均可解析
- 100-step official test render：
  `SSIM 0.722992 / PSNR 20.259757 / LPIPS(VGG) 0.508387`
- 1,000-step official test render：
  `SSIM 0.817581 / PSNR 24.365907 / LPIPS(VGG) 0.376104`；checkpoint 为 216,705 points
- 1,000-step train/render 峰值 cgroup memory：
  `54,966,005,760 / 57,229,574,144` bytes；峰值 VRAM `6,723 / 3,957` MiB；无 OOM
- 100/1,000 工程门禁证据：
  `gate_train100.json`、`gate_train1000.json`
- 正式 60,000-step train/render 均 `rc=0`；test：
  `SSIM 0.905364 / PSNR 29.902695 / LPIPS(VGG) 0.212178 / FPS 48.0888`
- train：
  `SSIM 0.939280 / PSNR 33.639803 / LPIPS(VGG) 0.181345 / FPS 42.3325`
- 最终 1,315,757 points；60k train/render 峰值 cgroup memory：
  `59,136,491,520 / 59,530,678,272` bytes；峰值 VRAM `16,039 / 6,407` MiB；无 OOM
- 180 个 train/test render 全部 1600×900、非空、非全黑；代表性 GT/render 对抽查未发现错相机或分支整体崩溃，
  动态车辆与路侧信息牌仍存在可见拖影，保留为 baseline 局限
- 最终审计：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M3-ADGS-0230-01/20260727T195611__scene0230__s0-r3/m3_final_audit.json`
- 单场景无预注册数值目标，M3 只裁决 pipeline 完整性，不能冒充六场景论文复现

### 2026-07-28 — M4 `blocked`

- scene-0230 直接采用 M3 冻结的正式 60k 结果，不重复训练
- scene-0242 正式实例：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260727T235743__scene0242__s0/`
- 通用逐场景 runner：
  `scripts/run_dr_adgs_scene.py`，SHA-256
  `b6d91f6986828dc8d23d0a6384bfe5bc5f9c3f466f91c189f167b0d2e8d52be8`
- scene-0242 preprocess 完整通过：180 images/depth/sky/semantic、138/138 flow、COLMAP 138/138
  registered images；SfM 只有 1,639 points，但主点云有 392,177 points，结构审计无失败
- flow 耗时约 2 小时 45 分，峰值 cgroup memory `44,934,205,440` bytes、VRAM `19,473` MiB；
  COLMAP 峰值 cgroup memory `44,902,842,368` bytes，均无 OOM
- 100-step train 已完成：test PSNR `16.817851`、train PSNR `17.162207`、79,320 points；
  峰值 cgroup memory `59,359,428,608` bytes
- 随后的 official render 在第 2/138 帧触发注册硬停止：连续两个采样达到 cgroup 90%，峰值
  `59,996,393,472` bytes，超过停止线约 81,599,693 bytes；runner 以 `SIGTERM` 结束本 stage，
  `oom=0 / oom_kill=0`，没有杀其他服务或继续重跑
- M4 当前需要外部资源变更后以新 instance 恢复；不得在现有合同下重跑、清空全局缓存、降分辨率或删相机

### 2026-07-28 — M4 `running`

- 新资源合同：RTX 3090 24 GB，cgroup 90 GiB，数据盘启动时约 141 GiB 可用
- 换机 smoke：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M2-ENV-ASSET-01/20260728T131221__wm-clone-3090-r4/`
- 五项环境 smoke、1,440/1,440 sensor payload 与 4 个 map masks 均通过；峰值 cgroup memory
  `9,685,876,736` bytes，`oom=0 / oom_kill=0`
- 当前 scene-0242：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T131642__scene0242__s0-r3-wm3090/`
- 已冻结 processed scene 的逐文件复用 fingerprint：
  `32bf9ccaa108273b69286625a0c7aaacb04fd9d76f243daff976206d0b7ef4f6`
- 独立 preprocess audit 再次通过；100-step test：
  `SSIM 0.771688 / PSNR 16.814014 / LPIPS(VGG) 0.453852`
- 1,000-step test：
  `SSIM 0.857157 / PSNR 24.363341 / LPIPS(VGG) 0.356590`
- 1,000-step train/render 峰值 cgroup memory：
  `23,832,678,400 / 25,567,031,296` bytes；峰值 VRAM `6,647 / 4,145` MiB；无 OOM
- 60k train 已启动；完成后先冻结 scene-0242，再串行执行 0255、0295、0518、0749
- 剩余场景 fail-closed sequencer：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T134226__remaining-sequencer-wm3090/`
- sequencer 只在 scene-0242 `done` 且 launcher `rc=0` 后逐场景启动，任何 source hash、资源或终态异常即停止
- 首个换机复用实例因完整性校验误拒合法的空 COLMAP 占位文件而在训练前 `blocked`：
  `20260728T131533__scene0242__s0-r2-wm3090/`；修复没有改变模型、数据或评测协议

### 2026-07-28 — M4 聚合守护与 M5 readiness

- 六场景聚合 finalizer：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T141204__aggregate6-s0-wm3090/`
- finalizer 只读等待 sequencer，逐场景复核合同、60k 产物、render 数量、checkpoint、OOM 与 official metrics；
  只有 coverage=6/6 且三项门禁同时通过才写 M4=`done`
- finalizer SHA-256：
  `64306bcb952d7753ef5799d6bce0a9b5aafbb975d672abd504035ac80ec1b8d4`
- DGGT official repo 已固定到 commit `a3276d2bbe4cbb03bcc117830b1836110a27adeb`；该动作只做
  readiness，不改变 M5=`pending`
- DGGT model repo revision 固定为 `735ac9a6486057b1eb886c33a8c6dc79e0b43214`；
  nuScenes checkpoint 远端大小 `5,411,266,466` bytes，尚未下载
- 静态审计再次确认 upstream `diffusion`/`difix` 参数错配，并发现 mode 2 硬编码 `start_idx=0`；
  M5 必须保存原始失败、使用最小兼容性 patch，并以可哈希 staging 实现固定窗口
- DGGT 代码许可证 Apache-2.0，模型权重 CC BY-NC 4.0；两者不得混写
- M5 post-gate controller：
  `/root/autodl-tmp/runs/dynamic_recon/_controllers/20260728T143042__m4-to-m5-wm3090/`
- controller 只在 M4 aggregate `done`、`all_gates_passed=true`、launcher `rc=0` 后创建正式 M5 run；
  M4 失败时不安装环境、不下载权重、不占 GPU
- M5 runner 固定 SHA-256：
  `3be81eef40d2062b9a8000ed086a5d9fbbb99e81e7aa25d3345dc90b4c07f445`；
  controller SHA-256：
  `31a90fb574b5dc886cc106086beaa4890ba850acda0bd5a8fd989696effdcbbf`
- 固定窗口 adapter SHA-256：
  `e8a629583eeb26ea6d60149c8340a38119dbfcff73270dcd6b2da32de295dfcf`；
  DGGT 单行 compatibility patch SHA-256：
  `a433785a84fffe44e5a84354b2aacf3bb3c21b308186fb88e52848b3476cb3a1`

### 2026-07-29 — M4 `done`

- 正式聚合实例：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T141204__aggregate6-s0-wm3090/`
- 六场景 coverage `6/6`，全部 60k checkpoint、42 test renders、138 train renders 与 official metrics 完整；
- official test arithmetic mean：
  `PSNR 31.174515 / SSIM 0.927661 / LPIPS(VGG) 0.163489`；
- 三项门禁 `PSNR≥30.56 / SSIM≥0.915 / LPIPS≤0.184` 全过；
- worst scene：PSNR scene-0295 `29.355150`、SSIM scene-0230 `0.905364`、LPIPS scene-0230
  `0.212178`；
- 六场景 train/render 均 `oom=0 / oom_kill=0`，没有通过缩协议获得结果。

### 2026-07-29 — M5 `blocked`

- 首个实例在 `env_torch` 期间因外部容器实例重建中断，已保留为 blocked：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T094923__native-nusc-s0-wm3090/`；
- 恢复实例：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/`；
- requirements 成功解析为 `rerun-sdk 0.23.1 / opencv-python 4.11.0.86 / numpy 1.26.4`；
- pointops2 的 PEP 517 隔离 build env 因 `ModuleNotFoundError: torch` 以 `rc=1` 失败，
  `oom=0 / oom_kill=0`、GPU 0 MiB；没有事后加入 `--no-build-isolation` 覆盖正式失败；
- checkpoint、untouched `difix` smoke、1-view/3-view 均未启动；common-observation 正式指标也未触发；
- 216-target AD-GS↔固定窗口像素映射只完成只读预审，不冒充 M5 正式 run；明确 blocked 证据满足 M6 前置分支。

### 2026-07-29 — M6 `done`

- 正式实例：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M6-STRESS-01/20260729T145645__identity-audit-s0-wm3090/`；
- 冻结 SAM pseudo ID 在六场景最长仅支持 `1/6/1/1/2/1` 帧，均低于 `20/60`；
- 六个 60k checkpoint 都只保存二值 `obj∈{0,1}`，持久 instance ID 数为 0；
- `persistent_object_identity_unavailable` 在 6/6 scenes 重复，满足跨场景失败门禁；
- 0/12 eligible object slots，全部对象编辑、pseudo-hole 与三 seeds 噪声行按协议记为 ABSTAIN 并保留
  coverage，没有事后几何重关联回填 baseline。

### 2026-07-29 — M7 `rejected`，M8/M9 未触发

- 正式 M7：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M7-HYPOTHESIS-01/20260729T145748__novelty-audit-s0-wm3090/`；
- 只考察决策表 A；InstDrive、Director、OmniRe、HorizonForge 与 G²Editor 已直接覆盖其持久实例身份、
  actor-centric Gaussian、轨迹编辑与遮挡补全主机制；
- novelty gate 失败，禁止事后注册 primary endpoint，M8 为 0 seeds/0 proposed metrics；
- 没有可供 M9 盲审的方法结果，因此不生成伪造 clips 或 verdict；human verdict 保持 `null`；
- 结果审核未触发说明：
  `docs/human-review/dynamic-reconstruction-results-v1/`。

## 18. 官方一手来源

- [nuScenes 官方数据说明](https://www.nuscenes.org/nuscenes)
- [nuScenes 官方 devkit](https://github.com/nutonomy/nuscenes-devkit)
- [nuScenes 使用条款](https://www.nuscenes.org/terms-of-use)
- [AD-GS ICCV 2025 论文](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Xu_AD-GS_Object-Aware_B-Spline_Gaussian_Splatting_for_Self-Supervised_Autonomous_Driving_ICCV_2025_paper.pdf)
- [AD-GS 官方代码](https://github.com/JiaweiXu8/AD-GS)
- [AD-GS 官方项目页](https://jiaweixu8.github.io/AD-GS-web/)
- [DGGT 官方代码](https://github.com/xiaomi-research/dggt)
- [DGGT 官方模型](https://huggingface.co/xiaomi-research/dggt)
- [DrivingEditor 论文条目](https://pubmed.ncbi.nlm.nih.gov/41650405/)
- [DrivingEditor 官方代码](https://github.com/WangXu-xxx/DrivingEditor)
- [VAD-GS CVPR 2026 论文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_VAD-GS_Visibility-Aware_Densification_for_3D_Gaussian_Splatting_in_Dynamic_Urban_CVPR_2026_paper.html)
- [VAD-GS 官方代码](https://github.com/YikangZhang1641/VAD-GS)
- [DenoiseGS AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37640)
- [Perception-aware 3DGS ICLR 2026](https://openreview.net/forum?id=PmQlMTBmpa)
- [Perception-aware 3DGS 官方代码](https://github.com/Shanicky-RenzhiWang/Perception-aware-3DGS)
- [ReconDrive 官方代码](https://github.com/TuojingAI/ReconDrive)
- [Real2Sim 论文](https://arxiv.org/abs/2605.13591)
- [GA-GS 论文](https://arxiv.org/abs/2604.04331)
- [InstDrive 论文](https://arxiv.org/abs/2508.12015)
- [Director 论文](https://arxiv.org/abs/2604.01678)
- [OmniRe OpenReview](https://openreview.net/forum?id=9cwxZxJixB)
- [HorizonForge 论文](https://arxiv.org/abs/2602.21333)
- [G²Editor 论文](https://arxiv.org/abs/2508.20471)
- [Waymo E2E 官方页](https://waymo.com/open/data/e2e/)

## 19. 当前下一条动作

本计划已按预注册停止规则走到研究终态。保留 M4 exact reproduction、M5 DGGT upstream 对照与 M6 跨场景
身份负结果；M7 novelty gate 失败后，不启动 M8/M9，不把 AD-GS 适配工程重命名为方法创新。后续若提出新路线，
必须使用新的任务 ID、独立 novelty delta 与前瞻 primary endpoint，不能回填或覆盖本轮 ABSTAIN/rejected 证据。
