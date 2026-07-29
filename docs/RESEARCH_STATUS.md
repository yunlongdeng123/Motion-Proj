# Research Status

- 更新时间：2026-07-29
- 当前路线：动态驾驶场景重建与反事实编辑
- 当前里程碑：`DR-M7-HYPOTHESIS-01`
- 状态：`rejected / route stopped by preregistered novelty gate`
- 当前门禁：M7 novelty 未通过；M8/M9 均未授权，不再启动方法、消融或人工盲审
- 当前资源合同：RTX 3090 24 GB，cgroup `memory.max=96,636,764,160` bytes（90 GiB）
- 本轮执行起点：`d90226cbba3854fe67cf32e6cb6be323a106e778`
- 本轮结果代码 commit：`460124664629f0b7bbea1f3509b7721f9d8cfe7d`
- 文档收口 commit：本文件所在提交（交付时以 `git log -1` 为准）
- 权威计划：[`DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md`](DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md)

## 最终裁决

本路线完成了强基线复现与负结果审计，但没有形成可注册的新方法：

- M4 `done`：AD-GS 官方六场景 exact reproduction 为
  `PSNR 31.174515 / SSIM 0.927661 / LPIPS(VGG) 0.163489`，三项带宽全过；
- M5 `blocked`：正式恢复实例
  `/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/`；
  requirements 成功后，pointops2 的 PEP 517 隔离构建环境因没有 `torch` 而失败，无 OOM；权重、native inference
  和 common-observation 正式指标均未运行，216-target 像素映射只完成只读预审；
- M6 `done`：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M6-STRESS-01/20260729T145645__identity-audit-s0-wm3090/`；六场景冻结 pseudo ID 最长仅 `1/6/1/1/2/1` 帧，checkpoint 又只保留
  二值对象标记，`persistent_object_identity_unavailable` 在 6/6 scenes 重复；
- M7 `rejected`：
  `/root/autodl-tmp/runs/dynamic_recon/DR-M7-HYPOTHESIS-01/20260729T145748__novelty-audit-s0-wm3090/`；候选 A 与 InstDrive、Director、OmniRe、HorizonForge、G²Editor
  直接重合，novelty gate 不通过；
- M8 `rejected / not authorized`：0 seeds、0 proposed metrics，没有事后注册 primary endpoint；
- M9 `rejected / not triggered`：没有可盲审的 method clips，human verdict=`null`，Codex 未代填。

结果审核未触发说明与证据清单：
[`human-review/dynamic-reconstruction-results-v1/`](human-review/dynamic-reconstruction-results-v1/)。

准确 claim 边界是：保留 AD-GS 六场景复现、DGGT upstream 对照和 AD-GS identity collapse 的负结果；不得把
适配工程、0 coverage 或已有 instance-aware/editing 机制重命名为创新。

## 历史 cut-in 裁决

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

## M4/M5 执行历史（已由上方最终裁决取代）

以下内容保留长跑期间的逐阶段事实和旧时点状态；不再表示当前任务仍为 running。

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

新实例资源授权后，先完成换机 smoke：

```text
/root/autodl-tmp/runs/dynamic_recon/DR-M2-ENV-ASSET-01/20260728T131221__wm-clone-3090-r4/
```

- AD-GS / DPT / Grounded-SAM-2 / pinned Grounding DINO HF / CoTracker3 全部通过；
- 1,440/1,440 sensor payload 与 4/4 map masks 重新审计通过；
- 峰值 cgroup memory `9,685,876,736` bytes，`oom=0 / oom_kill=0`；
- config fingerprint：
  `61ed109016f16a74ea0b53b175436d840cb33eae63dbf76e9b9da729fe82d2a5`。

当前 scene-0242 新实例：

```text
/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T131642__scene0242__s0-r3-wm3090/
```

- 逐文件复用已冻结 processed scene，output fingerprint：
  `32bf9ccaa108273b69286625a0c7aaacb04fd9d76f243daff976206d0b7ef4f6`；
- 独立 preprocess audit 再次通过；
- 100-step test：
  `SSIM 0.771688 / PSNR 16.814014 / LPIPS(VGG) 0.453852`；
- 1,000-step test：
  `SSIM 0.857157 / PSNR 24.363341 / LPIPS(VGG) 0.356590`；
- 1,000-step train/render 峰值 cgroup memory：
  `23,832,678,400 / 25,567,031,296` bytes；峰值 VRAM `6,647 / 4,145` MiB；
- `oom=0 / oom_kill=0`，60k train 已于 2026-07-28 13:34 +08:00 后启动；
- runner SHA-256：
  `3fe0b746ff442085d3b0b40bb64a30c8fe2f05fae90842541af37872de150653`；
- config fingerprint：
  `3c5a332e0ae4324565cb5b93ba34f20d34655d0cb909ba5688a05f9b3f5a185b`。

剩余场景 sequencer：

```text
/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T134226__remaining-sequencer-wm3090/
```

它只在 scene-0242 `done` 且 launcher `rc=0` 后，按
`0255 → 0295 → 0518 → 0749` 严格串行调用同一冻结 runner；任一 source hash、资源或终态异常立即
`blocked`。sequencer SHA-256：
`a3b150f04a988feacd96781b590806ec5589ec3a1f0673c08a90540c3f644dfd`。

复用校验的首个新实例因合法的 0-byte COLMAP 占位文件被误判而在训练前 `blocked`，证据保留于
`20260728T131533__scene0242__s0-r2-wm3090/`；修复只调整完整性校验，不改变资产、模型或协议。

六场景聚合 finalizer 已启动并只读等待 sequencer：

```text
/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T141204__aggregate6-s0-wm3090/
```

finalizer SHA-256：
`64306bcb952d7753ef5799d6bce0a9b5aafbb975d672abd504035ac80ec1b8d4`。它逐场景复核
terminal、协议、seed、upstream commit、60k train/render、OOM、42/138 张 test/train 渲染、checkpoint 与
official `results.json`，再写 per-scene、mean、worst-case、coverage 和三项门禁；任何缺失或数值不通过都
`blocked`，不会解锁 M5。

M5 仅完成不占 GPU 的 upstream readiness 审计，里程碑仍为 `pending`：

- DGGT 已固定到 `/root/autodl-tmp/third_party/dggt` commit
  `a3276d2bbe4cbb03bcc117830b1836110a27adeb`，worktree clean；
- Hugging Face 模型 revision 固定为 `735ac9a6486057b1eb886c33a8c6dc79e0b43214`；
  nuScenes 权重远端大小 `5,411,266,466` bytes，尚未下载；
- 代码许可证 Apache-2.0；模型卡许可证 CC BY-NC 4.0，必须分开登记；
- upstream `inference.py` 仍定义 `args.diffusion` 却访问 `args.difix`，且 mode 2 忽略 CLI
  `start_idx`；正式 M5 必须先保存原始失败，再以最小 patch 和显式窗口 staging 继续。

M5 的 post-gate controller 已只读等待 M4：

```text
/root/autodl-tmp/runs/dynamic_recon/_controllers/20260728T143042__m4-to-m5-wm3090/
```

只有 M4 aggregate `done`、`all_gates_passed=true` 且 launcher `rc=0` 才创建正式 M5 run。M5 runner 会按顺序
创建隔离环境、准备并哈希 18 个固定窗口、下载固定 revision 的 nuScenes 权重、保存 untouched upstream
`difix` 失败、应用单行 compatibility patch、完成 18/18 1-view，并在 24 GiB 支持时尝试 3-view。若 M4
`blocked`，controller 也立即 `blocked`，不会下载权重或占用 GPU。

冻结 SHA-256：

```text
prepare_dr_m5_dggt_inputs.py e8a629583eeb26ea6d60149c8340a38119dbfcff73270dcd6b2da32de295dfcf
DGGT-2026-07-28.patch         a433785a84fffe44e5a84354b2aacf3bb3c21b308186fb88e52848b3476cb3a1
run_dr_m5_dggt.py             3be81eef40d2062b9a8000ed086a5d9fbbb99e81e7aa25d3345dc90b4c07f445
run_dr_m5_after_m4.py         31a90fb574b5dc886cc106086beaa4890ba850acda0bd5a8fd989696effdcbbf
```

60k 完成后先裁决 scene-0242，再严格串行继续 scene-0255、0295、0518、0749，最后由 finalizer 聚合。
任何内存/显存不足、RC137、合同缺失或需要缩减官方协议的情况，立即写 `blocked`，不反复重跑。

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
