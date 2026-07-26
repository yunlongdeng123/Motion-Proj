# Research Status

- 更新时间：2026-07-26
- 当前路线：动态驾驶场景重建与反事实编辑
- 当前里程碑：`DR-M1-PLAN-01`
- 状态：`done`
- 下一门禁：用户审核计划并开放资源后，启动 `DR-M2-ENV-ASSET-01`
- 当前资源合同：无 GPU，cgroup `memory.max=2,147,483,648` bytes
- 本轮执行起点：`beee1de`
- 本轮交付 commit：以远端 `git rev-parse HEAD` 为准
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

## 当前资产事实

AD-GS 官方 nuScenes 协议固定为：

```text
scene-0230, scene-0242, scene-0255,
scene-0295, scene-0518, scene-0749
frames 10..69 inclusive
CAM_FRONT_LEFT / CAM_FRONT / CAM_FRONT_RIGHT
900×1600
```

当前磁盘缺少左右前相机和中间 sweeps，不能直接训练。所需文件可从本机只读官方 tar shards 选择性提取：

```text
/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/
```

禁止全量解压 294 GB；下一阶段先生成精确 member manifest。

## 下一步：`DR-M2-ENV-ASSET-01`

用户开放资源后，第一步只做资源 preflight：

- 至少 1×24 GB GPU；
- 系统 RAM 最低 32 GB，推荐 64 GB；
- 磁盘启动时可用 ≥60 GiB，并保留 20 GiB 安全余量；
- 记录 GPU/driver/CUDA/gcc/cgroup/process inventory。

资源满足后：

1. pin AD-GS upstream commit；
2. 创建隔离的 AD-GS/DPT/SAM 环境；
3. 编译并做最小 forward/backward smoke；
4. 选择性提取 scene-0230；
5. 完成伪监督与数据结构门禁；
6. 先 100/1,000 iteration profile，再决定是否启动官方 60k。

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
