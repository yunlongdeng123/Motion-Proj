# Motion-Proj 运行环境

- 更新时间：2026-07-29
- 当前事实源：本文件记录机器/路径；研究授权只看 [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)
- 当前资源合同：RTX 3090 24 GB，cgroup 内存上限 90 GiB
- 新路线环境：AD-GS/DPT/Grounded-SAM-2 已安装；DGGT 代码、隔离环境与正式终态由 M5 固定

## 1. 当前资源

2026-07-28 换机后现场：

```text
cgroup memory.max     96,636,764,160 B (90 GiB)
memory.events         oom 0 / oom_kill 0
GPU                   NVIDIA GeForce RTX 3090, 24,576 MiB
driver                580.105.08
data disk             250G total / 112G used / 139G avail / 45%
```

M4 已完成，M5 blocked 证据位于
`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/`。任何连续两个 5 秒采样达到 cgroup 90%、RC137、SIGKILL、GPU OOM、
`memory.events` 增加或数据盘低于 20 GiB，都按 `N1-F24/PIVOT-F05` 停止当前 stage；不杀其他服务，
不降分辨率或删相机绕过门禁。

## 2. 现有环境

环境都在数据盘，禁止混装：

| 环境 | 路径 | 大小约 | 用途/状态 |
|---|---|---:|---|
| motionproj | `/root/autodl-tmp/envs/motionproj` | 7.8G | 主仓库轻量审计与测试，保留 |
| drivestudio | `/root/autodl-tmp/envs/drivestudio` | 7.0G | 历史 StreetGS/OccGS，保留 |
| resim | `/root/autodl-tmp/envs/resim` | 6.4G | 历史 ReSim V6，非活跃 |
| adgs | `/root/autodl-tmp/envs/adgs` | 7.1G | AD-GS train/render/COLMAP，M4 done |
| adgs-dpt | `/root/autodl-tmp/envs/adgs-dpt` | 5.8G | Depth Anything V2，已通过 smoke |
| adgs-sam | `/root/autodl-tmp/envs/adgs-sam` | 6.0G | Grounded-SAM-2，已通过 smoke |
| dggt | `/root/autodl-tmp/envs/dggt` | 以 M5 manifest 为准 | requirements 已安装；pointops2 build isolation 失败，M5 blocked |

主仓库命令的标准激活：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motionproj
```

DriveStudio 历史环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/drivestudio
export CUDA_HOME=/usr/local/cuda-11.8
export PYTHONPATH=/root/autodl-tmp/third_party/drivestudio:$PYTHONPATH
```

历史细节：

- ReSim V6：[`archive/2026-07/v6/C1_V6_FINAL_REPORT.md`](../v6/C1_V6_FINAL_REPORT.md)
- OccGS E0：[`archive/2026-07/v7-feasibility/OCCGS_E0_ENV_MANIFEST.md`](../v7-feasibility/OCCGS_E0_ENV_MANIFEST.md)
- 当前保留规则：[`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md)

## 3. 新路线环境

| 环境 | 路径 | 关键 upstream 版本 / 当前状态 |
|---|---|---|
| AD-GS | `/root/autodl-tmp/envs/adgs` | 已安装；Python 3.7 / torch 1.13.1 compatibility 环境 |
| Depth Anything V2 | `/root/autodl-tmp/envs/adgs-dpt` | 已安装；Python 3.11 |
| Grounded-SAM-2 | `/root/autodl-tmp/envs/adgs-sam` | 已安装；Python 3.10 |
| DGGT | `/root/autodl-tmp/envs/dggt` | Python 3.10 / torch 2.4.1；requirements 完成，pointops2 未完成 |
| VAD-GS | `/root/autodl-tmp/envs/vadgs` | 未创建；M7 rejected 后未触发 |

完整顺序、兼容性 patch 规则和输出 manifest 见
[`DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md`](DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md) 第 4–5 节。

## 4. 代码与第三方路径

现有：

```text
/root/autodl-tmp/motion_proj
/root/autodl-tmp/third_party/drivestudio
/root/autodl-tmp/third_party/co-tracker
/root/autodl-tmp/third_party/gsplat
/root/autodl-tmp/third_party/pytorch3d
/root/autodl-tmp/third_party/AD-GS
/root/autodl-tmp/third_party/Depth-Anything-V2
/root/autodl-tmp/third_party/Grounded-SAM-2
/root/autodl-tmp/third_party/dggt
```

条件启用、当前不存在：

```text
/root/autodl-tmp/third_party/VAD-GS
/root/autodl-tmp/third_party/DrivingEditor
```

固定 commits：

```text
AD-GS             9a208512e49c8ddbaa20387921d9648adcd21cb4
Depth-Anything-V2 a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
Grounded-SAM-2    b7a9c29f196edff0eb54dbe14588d7ae5e3dde28
DGGT              a3276d2bbe4cbb03bcc117830b1836110a27adeb
```

每个新增仓库必须登记 commit、submodules、license、local diff 和 checkpoint SHA-256，不能只记分支名。

## 5. 数据路径

```text
# AD-GS 官方六场景精确 raw subset，含三前相机、LiDAR、metadata 与 maps
/root/autodl-tmp/data/dynamic_recon/raw_subset/adgs_nuscenes_v1

# 六场景已生成/正在生成的 AD-GS processed assets
/root/autodl-tmp/data/dynamic_recon/processed/adgs_nuscenes_v1

# 本机只读 nuScenes trainval tar shards，10 个 blobs 合计约 294G
/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval

# 历史 OccGS 数据
/root/autodl-tmp/data/occgs

# 历史本地子集
/root/autodl-tmp/data/nuscenes
```

AD-GS 官方六场景缺口已由选择性 tar member 提取闭合，1,440/1,440 sensor payload 与 4/4 map masks
在换机 M2 smoke 中再次通过。没有全量解压 294 GB，也没有原地修改历史 `/root/autodl-tmp/data/nuscenes`。

## 6. Run 与备份路径

```text
# 历史 runs
/root/autodl-tmp/runs/occgs_resim
/root/autodl-tmp/runs/event_first

# 新路线计划 runs
/root/autodl-tmp/runs/dynamic_recon

# 本轮归档前完整 docs 恢复点
/root/autodl-tmp/motion_proj_backups/docs-before-direction-pivot-2026-07-26
```

正式 run 使用唯一 ID，不覆盖失败实例。config、metrics、summary、terminal、hash 与人工 verdict 永久保护。

## 7. 磁盘策略

2026-07-28：

```text
/dev/md0  250G total  112G used  139G available  45%
```

新路线规则：

- 安装/训练启动前可用空间必须 ≥60 GiB；
- 运行中始终保留 20 GiB；
- 不复制 294 GB 公共 tar；
- 不在 AD-GS exact reproduction 前下载 Waymo、PandaSet、大型视频生成模型或全部可选 baseline；
- 环境、权重和输出先由 scene-0230 100/1,000-iteration profile 估算，再批准六场景；
- 空间不足时按清单评估可再生 cache，不从 raw、final checkpoint、正式指标或人工证据开始删。

## 8. 网络与下载

M4 exact 已完成；DGGT 环境/权重下载与 upstream smoke 由 M5 正式 run 执行：

- conda/pip 优先使用已配置镜像；
- Hugging Face checkpoint 必须固定 revision/文件 SHA-256；
- 下载前记录 license 和远端大小；
- 下载后立即哈希，不保留重复 cache；
- 任何网络/权重差异都写入 run manifest。

## 9. 换机 preflight

换机后第一条动作只读：

```text
memory.max/current/events
nvidia-smi GPU/VRAM/driver
nvcc/gcc
CPU cores
df/inode
process inventory
```

最低目标：

```text
RAM >= 32 GB (64 GB recommended)
GPU >= 1 x 24 GB
disk available >= 60 GiB
```

当前实例已通过该 preflight 和换机 M2 smoke。DGGT 环境、5.41 GB nuScenes 权重与终态以
`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/`
的 manifest/stages/summary 为准；checkpoint 未下载，native inference 未执行。
