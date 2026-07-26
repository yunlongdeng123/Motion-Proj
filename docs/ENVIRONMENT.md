# Motion-Proj 运行环境

- 更新时间：2026-07-26
- 当前事实源：本文件记录机器/路径；研究授权只看 [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)
- 当前资源合同：无 GPU，cgroup 内存上限 2 GiB
- 新路线环境：尚未安装

## 1. 当前资源

2026-07-26 现场：

```text
cgroup memory.max     2,147,483,648 B
metadata audit peak   2,129,526,784 B
memory.events         oom 0 / oom_kill 0
GPU                   当前未开放；不以历史 4090 配置推断本轮可用
data disk             128G total / 62G used / 67G avail / 49%
```

当前只允许轻量文本、Git 和精确文件操作。禁止 conda 求解、权重下载、tar 全量扫描、预处理、训练、推理和 GPU
测试。任何持续接近 `memory.max`、RC137、SIGKILL 或 OOM 都按 `N1-F24/PIVOT-F05` 停机并等待用户。

## 2. 现有环境

环境都在数据盘，禁止混装：

| 环境 | 路径 | 大小约 | 用途/状态 |
|---|---|---:|---|
| motionproj | `/root/autodl-tmp/envs/motionproj` | 7.8G | 主仓库轻量审计与测试，保留 |
| drivestudio | `/root/autodl-tmp/envs/drivestudio` | 7.0G | 历史 StreetGS/OccGS，保留 |
| resim | `/root/autodl-tmp/envs/resim` | 6.4G | 历史 ReSim V6，非活跃 |

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

- ReSim V6：[`archive/2026-07/v6/C1_V6_FINAL_REPORT.md`](archive/2026-07/v6/C1_V6_FINAL_REPORT.md)
- OccGS E0：[`archive/2026-07/v7-feasibility/OCCGS_E0_ENV_MANIFEST.md`](archive/2026-07/v7-feasibility/OCCGS_E0_ENV_MANIFEST.md)
- 当前保留规则：[`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md)

## 3. 新路线计划环境

只有用户开放资源并通过 `DR-M2-ENV-ASSET-01` 后才创建：

| 环境 | 计划路径 | 关键 upstream 版本 |
|---|---|---|
| AD-GS | `/root/autodl-tmp/envs/adgs` | Python 3.7.16 / torch 1.13.1 / CUDA 11.7 runtime / COLMAP 3.7 |
| Depth Anything V2 | `/root/autodl-tmp/envs/adgs-dpt` | Python 3.11 |
| Grounded-SAM-2 | `/root/autodl-tmp/envs/adgs-sam` | Python 3.10 / CUDA_HOME 12.1 |
| DGGT | `/root/autodl-tmp/envs/dggt` | Python 3.10 / torch 2.4.1 |
| VAD-GS | `/root/autodl-tmp/envs/vadgs` | 条件启用；Python 3.8 / torch 1.12 + cu113 |

这些只是计划值，不是已安装事实。完整顺序、兼容性 patch 规则和输出 manifest 见
[`DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md`](DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md) 第 4–5 节。

## 4. 代码与第三方路径

现有：

```text
/root/autodl-tmp/motion_proj
/root/autodl-tmp/third_party/drivestudio
/root/autodl-tmp/third_party/co-tracker
/root/autodl-tmp/third_party/gsplat
/root/autodl-tmp/third_party/pytorch3d
```

计划新增但当前不存在/未锁定：

```text
/root/autodl-tmp/third_party/AD-GS
/root/autodl-tmp/third_party/Depth-Anything-V2
/root/autodl-tmp/third_party/Grounded-SAM-2
/root/autodl-tmp/third_party/dggt
/root/autodl-tmp/third_party/VAD-GS
/root/autodl-tmp/third_party/DrivingEditor
```

每个新增仓库必须登记 commit、submodules、license、local diff 和 checkpoint SHA-256，不能只记分支名。

## 5. 数据路径

```text
# 当前本地 raw 子集，约 35G；主要是 CAM_FRONT、LIDAR_TOP 与 metadata
/root/autodl-tmp/data/nuscenes

# 本机只读 nuScenes trainval tar shards，10 个 blobs 合计约 294G
/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval

# 历史 OccGS 数据
/root/autodl-tmp/data/occgs

# 新路线计划目录
/root/autodl-tmp/data/dynamic_recon
```

AD-GS 官方六场景的左右前相机和中间 sweeps 当前不完整。下一轮只从 tar shards 选择性提取精确 member，不全量
解压，不原地修改 `/root/autodl-tmp/data/nuscenes`。

六场景资产缺口表见
[`DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md`](DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md) 第 3.2 节。

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

本轮清理后：

```text
/dev/md0  128G total  62G used  67G available  49%
```

新路线规则：

- 安装/训练启动前可用空间必须 ≥60 GiB；
- 运行中始终保留 20 GiB；
- 不复制 294 GB 公共 tar；
- 不在 AD-GS exact reproduction 前下载 Waymo、PandaSet、大型视频生成模型或全部可选 baseline；
- 环境、权重和输出先由 scene-0230 100/1,000-iteration profile 估算，再批准六场景；
- 空间不足时按清单评估可再生 cache，不从 raw、final checkpoint、正式指标或人工证据开始删。

## 8. 网络与下载

当前不下载。资源开放后：

- conda/pip 优先使用已配置镜像；
- Hugging Face checkpoint 必须固定 revision/文件 SHA-256；
- 下载前记录 license 和远端大小；
- 下载后立即哈希，不保留重复 cache；
- 任何网络/权重差异都写入 run manifest。

## 9. 下一轮资源 preflight

用户开放资源后的第一条动作只读：

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

满足后才创建 AD-GS 环境；不满足则 `DR-M2-ENV-ASSET-01=blocked` 并等待。
