# Research Status

- 更新时间：2026-07-28
- 当前路线：动态驾驶场景重建与反事实编辑
- 当前里程碑：`DR-M4-ADGS-6SCENE-01`
- 状态：`blocked`
- 当前门禁：scene-0242 official render 触发 cgroup 90% 硬停止
- 当前资源合同：RTX 4080 SUPER 32 GB，cgroup `memory.max=66,571,993,088` bytes，数据盘约 149 GiB 可用
- 本轮执行起点：`2d46f4c1c79708451081f291a267a7acd26a3236`
- 本轮交付 commit：尚未提交；正式 run 已记录 worktree fingerprint 和 source snapshots
- 权威计划：[`DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md`](DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md)

## 当前裁决

nuScenes cut-in mining 已 `rejected / frozen`：

- 官方没有事件级 cut-in 真值或公开总体占比，无法验证召回率；
- strict-v2 在 675 个 prospective scenes 上只有 `1 PASS / 1 scene`；
- 继续投入主要产生事件挖掘、地图、接收车与审核工程，不再服务于重建/编辑核心问题；
- 结果只说明当前可验证事件池过稀，不说明 nuScenes 没有 cut-in；
- cut-in 以后最多作为可选演示，不再承担数据入口、方法定义或论文成立条件。

历史文档、最终报告、审核包和清理清单：

- [`archive/2026-07/cutin-mining-closed/README.md`](archive/2026-07/cutin-mining-closed/README.md)
- [`archive/2026-07/cutin-mining-closed/CLEANUP_MANIFEST.md`](archive/2026-07/cutin-mining-closed/CLEANUP_MANIFEST.md)

## 已完成

### `DR-M0-ARCHIVE-01` — done

- 建立归档前完整 docs 快照；
- 将 cut-in 路线文档移入单一历史目录；
- 保留原始数据、final checkpoints、正式指标、配置、人工结论与失败教训；
- 删除约 5 GiB 可再生缓存、中间 checkpoint、失败渲染副本和 5 个临时 worktree；
- 删除 41 个冗余 `.codexbak.*`，保留 V7.1 审计目录内 20 个已索引备份；
- 执行后磁盘约 `62G used / 67G avail`；
- `memory.events: oom=0, oom_kill=0`。

证据：

```text
/root/autodl-tmp/motion_proj/docs/archive/2026-07/cutin-mining-closed/
/root/autodl-tmp/motion_proj_backups/docs-before-direction-pivot-2026-07-26/
```

### `DR-M1-PLAN-01` — done

- 核对 AD-GS、DGGT、DrivingEditor、VAD-GS、DenoiseGS、Perception-aware 3DGS、ReconDrive、
  Real2Sim 与 GA-GS 的官方论文/代码状态；
- 确定 AD-GS exact reproduction 为主基线；
- 确定 DGGT 为 inference-only 前馈对照；
- 将 VAD-GS 设为补密/可见性方向的强制 novelty 对照；
- 审计 AD-GS 官方六个 nuScenes scenes 的本地资产缺口；
- 写成完整环境、数据、实验、统计、停止与人工审核方案；
- 将路线转向失败与防重复项加入 [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)。

### `DR-M2-ENV-ASSET-01` — done

- 正式实例：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M2-ENV-ASSET-01/20260727T180733__e49a4e-4080s-r3/`
- AD-GS、DPT、Grounded-SAM-2、固定 revision Grounding DINO HF 和固定 CoTracker3 权重 smoke 全部通过；
- 六场景精确提取 1,440 个 sensor payload，并补齐 4 个 nuScenes map masks；
- 每场景 180 RGB + 60 unique LiDAR；1600×900、相机顺序、时间戳、标定、pose、LiDAR shape 全部通过；
- manifest SHA-256：
  `64c68972a25834757168cd8fdc11c64b134b6ae0d9206a9ebde4064891c16092`；
- compatibility patch SHA-256：
  `efbed2eb888d2e77238e99ea84423435cc5d241b3dbb0dc55443e4967eb1c98a`；
- 正式实例峰值 cgroup memory `30,123,261,952` bytes，`oom=0 / oom_kill=0`；
- 首次审计因缺少 4 个 map masks 失败，已以独立 `blocked` 实例保留，没有覆盖失败证据。

### `DR-M3-ADGS-0230-01` — done

- 正式实例：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M3-ADGS-0230-01/20260727T195611__scene0230__s0-r3/`
- 180 images/depth/sky/semantic、138/138 flow、COLMAP 138/138 registered images 全部通过；
- 60k official test：
  `SSIM 0.905364 / PSNR 29.902695 / LPIPS(VGG) 0.212178 / FPS 48.0888`；
- 60k official train：
  `SSIM 0.939280 / PSNR 33.639803 / LPIPS(VGG) 0.181345 / FPS 42.3325`；
- checkpoint：1,315,757 points；
- 60k train/render 峰值 cgroup memory：
  `59,136,491,520 / 59,530,678,272` bytes；
- 60k train/render 峰值 VRAM：`16,039 / 6,407` MiB；
- `oom=0 / oom_kill=0`，渲染阶段距 90% 停止线仅 384,115,507 bytes；
- 全量结构/像素统计与代表性视觉抽查通过；动态目标拖影作为 baseline 局限保留；
- 最终审计：`m3_final_audit.json`；M3 无单场景数值目标，不能冒充论文六场景结果。

## 当前资产事实

AD-GS 官方 nuScenes 协议固定为：

```text
scene-0230, scene-0242, scene-0255,
scene-0295, scene-0518, scene-0749
frames 10..69 inclusive
CAM_FRONT / CAM_FRONT_LEFT / CAM_FRONT_RIGHT
900×1600
```

M1 发现的左右前相机和中间 sweeps 缺口已经闭合。正式 raw subset：

```text
/root/autodl-tmp/data/dynamic_recon/raw_subset/adgs_nuscenes_v1/
```

来源是 `/autodl-pub/data/nuScenes/Fulldatasetv1.0/Trainval/` 的只读 tar shards；只提取精确 member，
没有全量解压 294 GB。

## 当前执行：`DR-M4-ADGS-6SCENE-01`

scene-0230 使用 M3 已冻结的正式 60k 结果。当前 scene-0242 实例：

```text
/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260727T235743__scene0242__s0/
```

通用 runner `scripts/run_dr_adgs_scene.py` 把原 M3 runner 只做 scene/task 参数化；preprocess、模型、
阈值、分辨率、帧、相机、seed、训练和指标命令均未改变。runner SHA-256：
`b6d91f6986828dc8d23d0a6384bfe5bc5f9c3f466f91c189f167b0d2e8d52be8`。

scene-0242 preprocess 已通过：180 images/depth/sky/semantic、138/138 flow、COLMAP 138/138
registered images；主点云 392,177 points，SfM 1,639 points。100-step train 完成，test PSNR
`16.817851`、train PSNR `17.162207`、79,320 points；训练峰值 cgroup memory
`59,359,428,608` bytes。

official render 随后在第 2/138 帧连续两个采样达到 90% 停止线，峰值
`59,996,393,472` bytes，stage `rc=-15`、runner `rc=1`；`oom=0 / oom_kill=0`。该实例已保留为
`blocked`，没有立即重跑、放宽门禁、降分辨率、删相机或全局清缓存。

固定执行：

恢复条件：

1. 提高容器 cgroup 内存；建议至少 80 GiB、推荐 96 GiB；
2. 新建 M4 scene-0242 instance，复用已冻结 processed scene，但不覆盖 blocked 证据；
3. 保持 frames 10..69、三相机、900×1600、seed 0、模型、损失与指标不变；
4. 再继续 scene-0255、0295、0518、0749，最后聚合六场景门禁。

任何内存/显存不足、RC137 或需要缩减官方协议的情况，立即写 `blocked` 并等待用户，不反复重跑。

## 禁止事项

- 不恢复 cut-in mining 或调低 strict-v2 门槛；
- 不在 AD-GS 六场景 exact reproduction 前合并 Motion-Proj/OccGS/StreetGS；
- 不提前加 occupancy、物理、扩散、感知损失或轨迹编辑；
- 不删困难 scene、不降分辨率、不缺相机后仍对齐论文指标；
- 不把兼容性 patch、工程运行或单个好视频写成方法贡献；
- 不把 unsupported 生成内容伪装成真实观测；
- 不杀用户服务争抢内存。

## 文档入口

- [完整研究计划](DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md)
- [实验注册表](EXPERIMENTS.md)
- [失败与防重复](RESEARCH_FAILURES.md)
- [文档导航](README.md)
- [本轮人工审核包](human-review/dynamic-reconstruction-plan-v1/README.md)
- [历史归档](archive/2026-07/README.md)
