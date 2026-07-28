# Experiments

- 更新时间：2026-07-28
- 活跃路线：动态驾驶场景重建与反事实编辑
- 权威方案：[`DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md`](DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md)
- 历史快照：[`archive/2026-07/cutin-mining-closed/EXPERIMENTS_CUTIN_FINAL_SNAPSHOT.md`](archive/2026-07/cutin-mining-closed/EXPERIMENTS_CUTIN_FINAL_SNAPSHOT.md)

本文件只登记当前路线。V1–V7.1、OccGS 和 cut-in 的完整实验历史已归档；失败事实继续由
[`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md) 约束，不因精简活动台账而删除。

## 1. 状态词

只使用：

```text
pending | running | blocked | done | rejected
```

`done` 表示预注册门禁满足；`blocked` 表示需要外部资源/权限或 upstream 修复；`rejected` 表示研究门禁失败，不能靠
重命名、挑场景或放宽阈值继续。

## 2. 活跃注册表

| Run ID | 状态 | 目标 | 输入 | Primary gate | 证据路径 |
|---|---|---|---|---|---|
| `DR-M0-ARCHIVE-01` | done | 封存 cut-in、清理可再生产物 | 历史 docs/data/runs | 保留项完整、删除项精确、无 OOM | `docs/archive/2026-07/cutin-mining-closed/` |
| `DR-M1-PLAN-01` | done | 官方调研与下一阶段完整方案 | 官方论文/代码、本地只读资产 | baseline/data/env/experiments/stops/review 全部闭合 | `docs/DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md` |
| `DR-M2-ENV-ASSET-01` | done | AD-GS 环境与六场景资产 smoke | 官方六 scenes、独立 envs | 编译/单步/文件计数/typed provenance 全过 | `/root/autodl-tmp/runs/dynamic_recon/DR-M2-ENV-ASSET-01/20260727T180733__e49a4e-4080s-r3/` |
| `DR-M3-ADGS-0230-01` | done | scene-0230 exact pipeline | frames 10..69、三相机、900×1600 | 60k 正常结束、官方 render/metrics 完整 | `/root/autodl-tmp/runs/dynamic_recon/DR-M3-ADGS-0230-01/20260727T195611__scene0230__s0-r3/` |
| `DR-M4-ADGS-6SCENE-01` | blocked | AD-GS 六场景数值复现 | 官方六 scenes | PSNR≥30.56、SSIM≥0.915、LPIPS≤0.184 | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260727T235743__scene0242__s0/` |
| `DR-M5-DGGT-NUSC-01` | pending | DGGT 推理级对照 | 同六 scenes 固定窗口 | upstream smoke；完整输入预算与速度/质量报告 | 同上 |
| `DR-M6-STRESS-01` | pending | 重建/编辑/去遮挡/噪声压力测试 | 六 scenes、冻结对象与编辑幅度 | ≥3 scenes 重复同一失败 | 同上 |
| `DR-M7-HYPOTHESIS-01` | pending | 唯一创新假设预注册 | M6 failure matrix | novelty、truth tier、primary、baseline 冻结 | `docs/` |
| `DR-M8-METHOD-01` | pending | 方法/消融/统计 | 六 scenes、3 seeds | primary 改善且 guardrails 不退化 | `/root/autodl-tmp/runs/dynamic_recon/` |
| `DR-M9-HUMAN-01` | pending | 盲审与最终人工包 | 冻结全量 clips/metrics | 用户/指定评审完成 verdict | `docs/human-review/` |

## 3. `DR-M0-ARCHIVE-01`

### 结果

- 状态：`done`
- 文档快照：
  `/root/autodl-tmp/motion_proj_backups/docs-before-direction-pivot-2026-07-26/`
- 回收：约 5 GiB；
- 删除：3 个 N1 可再生 10 Hz cache、3 个 B0 中间 checkpoint、2 个 H1C 失败渲染副本、
  5 个干净 worktree、41 个冗余 `.codexbak.*`；V7.1 审计目录内 20 个已索引备份继续保留；
- 保留：raw nuScenes、mini comparator、trainval annotations、final checkpoints、正式指标/日志、审核证据、
  `RESEARCH_FAILURES.md`；
- `memory.events`: `oom=0`, `oom_kill=0`。

### 证据

- [`archive/2026-07/cutin-mining-closed/CLEANUP_MANIFEST.md`](archive/2026-07/cutin-mining-closed/CLEANUP_MANIFEST.md)
- [`archive/2026-07/cutin-mining-closed/README.md`](archive/2026-07/cutin-mining-closed/README.md)

## 4. `DR-M1-PLAN-01`

### 结果

- 状态：`done`
- 主基线：AD-GS；
- 前馈对照：DGGT inference-only；
- 编辑参考：DrivingEditor；
- conditional geometry comparator：VAD-GS；
- primary data：AD-GS 官方 nuScenes 六 scenes；
- M1 当时资产结论：左右前相机和 sweeps 不完整，必须选择性提取；该缺口已由 M2 闭合；
- 当前资源结论：RTX 4080 SUPER 32 GB、cgroup 约 62 GiB、数据盘约 149 GiB 可用，可继续串行 M4。

### 复现锚点

```text
scenes = 0230,0242,0255,0295,0518,0749
frames = 10..69 inclusive
cameras = CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT
resolution = 900x1600
iterations = 60000
paper_mean = PSNR 31.06 / SSIM 0.925 / LPIPS 0.164
```

### 下一步

M2 已完成；M3 的冻结预处理与结构审计已通过，正在按 100 → 1,000 → 60,000 iterations 的门禁顺序推进。

## 5. `DR-M2-ENV-ASSET-01`

### 结果

- 状态：`done`
- 正式实例：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M2-ENV-ASSET-01/20260727T180733__e49a4e-4080s-r3/`
- blocked 前置实例：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M2-ENV-ASSET-01/20260727T165549__e49a4e-4080s-r2/`
- 项目 commit：`2d46f4c1c79708451081f291a267a7acd26a3236`
- 正式数据 manifest：
  `/root/autodl-tmp/data/dynamic_recon/manifests/adgs_nuscenes_v1_manifest.json`
- manifest SHA-256：`64c68972a25834757168cd8fdc11c64b134b6ae0d9206a9ebde4064891c16092`
- 资产：1,440/1,440 sensor payload；4/4 map masks；无 symlink、空文件或哈希不匹配
- 每场景：180 RGB + 60 unique nearest LiDAR；图像均为 1600×900；时间戳、标定、pose 和 LiDAR 均通过审计
- 环境：AD-GS / DPT / SAM2 / pinned Grounding DINO HF / pinned CoTracker3 smoke 全部 `rc=0`
- 资源：正式实例峰值 cgroup memory `30,123,261,952` bytes，OOM/OOM-kill delta `0/0`
- compatibility patch：
  `efbed2eb888d2e77238e99ea84423435cc5d241b3dbb0dc55443e4967eb1c98a`

第一次审计暴露 raw subset 缺少 nuScenes devkit 初始化必需的 4 个 `maps/*.png`；该实例保留为 `blocked`，
补齐静态 map masks 并纳入 manifest/hash 审计后以 r3 新实例通过。M2 只闭合环境、权重、原始资产与单样本
伪监督 smoke；scene-0230 的完整派生预处理按计划在 M3 stages 中执行。

### 输入

- official AD-GS commit；
- official environment.yaml；
- scene-0230 首先；
- 本机只读 nuScenes tar shards。

### 固定顺序

1. resource preflight；
2. pin source/license；
3. create AD-GS env；
4. compile rasterizers；
5. one-step forward/backward；
6. DPT/SAM single-image smoke；
7. exact asset member manifest；
8. scene-0230 selective extraction；
9. upstream preprocess structural audit。

### 单卡停止

- memory ≥90% limit；
- OOM/RC137；
- GPU OOM；
- disk free <20 GiB；
- 需要少相机/低分辨率才能继续。

任一触发后状态写 `blocked`，不重跑。

## 6. `DR-M3/M4` 预注册

### M3

scene-0230 先 100 iterations、再 1,000 iterations 做资源画像；只有投影满足资源合同才跑官方 60k。
M3 不以单场景数字对齐论文，只验证 exact pipeline 完整。

首个实例 `20260727T181617__scene0230__s0` 在 sky mask 阶段按合同 `blocked`：AD-GS 生成 PNG，
Grounded-SAM-2 loader 只枚举 JPEG。失败 processed scene 已原样移至
`/root/autodl-tmp/data/dynamic_recon/processed_failed/20260727T181617__scene0230__s0/`。

第二个实例 `20260727T182247__scene0230__s0-r2` 完成 depth/mask/segmentation/flow 后，在 COLMAP
feature extraction 因 upstream 默认使用全部 128 个 CPU 线程触发内存停止门禁：峰值
`62,265,835,520` bytes，连续两个采样高于 cgroup 的 90%，但 `oom=0 / oom_kill=0`。其失败
COLMAP 派生目录已移至
`/root/autodl-tmp/data/dynamic_recon/processed_failed/20260727T182247__scene0230__s0-r2/colmap/`。

当前实例：
`/root/autodl-tmp/runs/dynamic_recon/DR-M3-ADGS-0230-01/20260727T195611__scene0230__s0-r3/`。
runner 在校验输入哈希和 860 个已完成派生文件后复用 r2 的 pre-COLMAP 结果；复用指纹为
`836b8f1480b083e5d9180f235e06d290c2ed7c5eb6f4de791d920fe945282891`。COLMAP 只限制并发为
16 threads，不改图像、SIFT 参数、匹配方式或几何结果；该阶段已通过，峰值 cgroup memory
`35,117,174,784` bytes。完整 processed audit 通过：180 images/depth/sky/semantic、138/138 flow、
COLMAP 138/138 注册图像和 70,933 points。

compatibility patch 同时包含 byte-preserving `.jpg` staging alias 与 COLMAP 并发上限；不转码、不改模型、
阈值、损失或指标。当前 patch SHA-256 为
`49b4c06ecec6c30f1e80b5abf4d46970920f9d71952acbda273774d9b5b34f48`。
100-step 与 1,000-step 工程门均已通过。官方 test render 指标从
`SSIM 0.722992 / PSNR 20.259757 / LPIPS(VGG) 0.508387` 改善至
`SSIM 0.817581 / PSNR 24.365907 / LPIPS(VGG) 0.376104`。1,000-step checkpoint 含
216,705 points；train/render 峰值 cgroup memory 分别为 `54,966,005,760` /
`57,229,574,144` bytes，峰值 VRAM `6,723 / 3,957` MiB，均无 OOM，磁盘仍有约 150 GiB。
门禁证据保存在当前实例的 `gate_train100.json` 与 `gate_train1000.json`。

正式 60,000-step train/render 均 `rc=0`。official test：
`SSIM 0.905364 / PSNR 29.902695 / LPIPS(VGG) 0.212178 / FPS 48.0888`；official train：
`SSIM 0.939280 / PSNR 33.639803 / LPIPS(VGG) 0.181345 / FPS 42.3325`。最终 1,315,757 points；
train/render 峰值 cgroup memory `59,136,491,520 / 59,530,678,272` bytes，峰值 VRAM
`16,039 / 6,407` MiB，均无 OOM。结构、完整渲染统计与代表性视觉抽查通过，M3 状态为 `done`。

### M4

六场景全部运行，三项必须同时通过：

```text
mean PSNR >= 30.56
mean SSIM >= 0.915
mean LPIPS(VGG) <= 0.184
```

报告所有 per-scene、mean、worst-case、coverage。失败只允许一次有明确根因的重跑；仍失败则 `blocked`。

scene-0230 使用 M3 已冻结的正式结果。scene-0242 正式实例
`/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260727T235743__scene0242__s0/`
的 preprocess 已完整通过；138/138 flow、138/138 registered images 与结构审计均完成。100-step train
完成后，official render 在第 2/138 帧连续两个采样达到 cgroup 90% 停止线，峰值
`59,996,393,472` bytes，超过停止线约 81.6 MB；stage `rc=-15`，runner `rc=1`，
`oom=0 / oom_kill=0`。实例已按合同保留为 `blocked`，不立即重跑。恢复要求：提高 cgroup 内存额度后使用
新 instance；建议至少 80 GiB、推荐 96 GiB。

## 7. `DR-M5` 预注册

DGGT 不做训练。固定三个 4-frame windows/scene：

```text
10..13
34..37
66..69
```

先官方 native protocol，再做 common-observation diagnostic。必须报告 DGGT 与 AD-GS 不同的输入帧数、pose 使用、
逐场景优化和 resize，禁止写成 matched leaderboard。

## 8. `DR-M6` 预注册

### 对象

每 scene 最多两个：

- 可见支持最高的 `high-support`；
- 仍满足最低门槛但支持最低的 `boundary-support`。

不足两个如实记 coverage。

### 编辑

```text
lateral +0.5/+1.0/+1.5 m
time shift -0.5/+0.5 s
speed 0.75x/1.25x
stop 1.0 s then restart
delete
```

不赋予 cut-in/merge 语义。

### 真值

- Tier A：held-out real observation；
- Tier B：geometric support；
- Tier C：unsupported，只评 uncertainty/ABSTAIN/人审。

### 噪声

固定 `0230/0242/0255`，one-factor-at-a-time，3 seeds；完整级别见权威计划第 11.5 节。

## 9. `DR-M7/M8` 预注册

只有 M6 在 ≥3 scenes 重复失败才选方法。当前预期优先考察：

```text
编辑诱发 visibility recomputation
+ evidence-typed/confidence-aware Gaussians
+ non-target perception preservation
```

但 VAD-GS、GA-GS、DrivingEditor、DenoiseGS、Perception-aware 3DGS 的 claim 边界必须先审计。若无稳定失败，
M7=`rejected`，不硬造模块。

方法实验至少 3 seeds，matched scene/frame/camera/actor/edit/seed/budget，报告 CI、worst-case 与 coverage。

## 10. 历史路线入口

- cut-in 最终封存：[`archive/2026-07/cutin-mining-closed/README.md`](archive/2026-07/cutin-mining-closed/README.md)
- cut-in 结束时状态：[`archive/2026-07/cutin-mining-closed/RESEARCH_STATUS_CUTIN_FINAL_SNAPSHOT.md`](archive/2026-07/cutin-mining-closed/RESEARCH_STATUS_CUTIN_FINAL_SNAPSHOT.md)
- cut-in 结束时实验台账：[`archive/2026-07/cutin-mining-closed/EXPERIMENTS_CUTIN_FINAL_SNAPSHOT.md`](archive/2026-07/cutin-mining-closed/EXPERIMENTS_CUTIN_FINAL_SNAPSHOT.md)
- OccGS V7/V7.1：[`archive/2026-07/v7-feasibility/`](archive/2026-07/v7-feasibility/)
- 所有失败：[`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)
