# V7.3 数据盘清理记录

日期2026-09-07；task=`WS-V73-M0-RISK-STORAGE-02`；base commit=`63626e8d`。

数据盘700GiB。清理前约568.33GiB已用、131.67GiB可用；实际释放 **54.63GiB**，清理后约513.71GiB已用、**186.30GiB可用**。以文件系统实际空闲字节差计算；目录du之和与释放量可因硬链接不同。

## 已执行

| 对象 | 目录占用GiB | 判断 |
|---|---:|---|
| `/root/autodl-tmp/envs/worldsim-v32-asset-harvester` | 13.27 | 已退役依赖或可重下缓存，已删除 |
| `/root/autodl-tmp/envs/worldsim-v33-sam2` | 5.29 | 已退役依赖或可重下缓存，已删除 |
| `/root/autodl-tmp/envs/worldsim-v61-gaussianworld` | 7.16 | 已退役依赖或可重下缓存，已删除 |
| `/root/autodl-tmp/envs/worldsim-v61-hy3d-omni` | 5.80 | 已退役依赖或可重下缓存，已删除 |
| `/root/autodl-tmp/envs/worldsim-v61-irwm` | 6.09 | 已退役依赖或可重下缓存，已删除 |
| `/root/autodl-tmp/third_party/worldsim_v32/asset-harvester/checkpoints` | 7.47 | 已退役依赖或可重下缓存，已删除 |
| `/root/autodl-tmp/hf_cache/worldsim_v32_asset_harvester` | 3.63 | 已退役依赖或可重下缓存，已删除 |
| `/root/autodl-tmp/pip_cache/http-v2` | 3.28 | 已退役依赖或可重下缓存，已删除 |
| `/root/autodl-tmp/conda_pkgs/cache` | 1.64 | 已退役依赖或可重下缓存，已删除 |

另清理 conda 包下载归档，保留 extracted package cache，避免破坏可能的环境链接。删除环境前导出 conda 包名/版本/build/channel 及 Python distribution 版本；记录不包含新校验和。环境恢复需要重新安装，有编译扩展时按保留的第三方源码重新构建；不是完整二进制环境备份。

## 保留及后续候选

| 对象 | 约GiB | 判断 |
|---|---:|---|
| data 总计 | 214.54 | 原始LiDAR/RGB/标定、AV2、规范输入保留，当前数据恢复代价大 |
| runs 总计 | 161.49 | 全部保留，确保逐实验回溯；不以整个旧版本目录作为删除单位 |
| models/eas_vggt | 14.51 | 当前VGGT/Pi3X/MapAnything必需，保留 |
| motionproj / v72-pointr | 8.10 / 7.68 | 原生解码/强补全基线，保留 |
| adgs / drivestudio / v72-lidar4d | 6.96 / 6.96 / 5.97 | 背景/已跑外部基线，保留 |
| data/worldsim_v4/adgs_processed_v4 | 15.87 | 后续可逐场景判断，但背景拼接可能复用，当前不删 |
| runs/worldsim_v6 与 v64 的 sensor_worker/sidecar | 52.55 / 33.26（各版本总量） | 有重建中间payload清理潜力，先明确每个结果对原数组依赖，当前不删 |

未新增哈希/校验和/指纹；保留已有历史记录原样。清理脚本、plan/result、5套环境版本清单见同目录 `autoresearch/worldsim_v73/storage/`；服务器原件在 `/root/autodl-tmp/cleanup_manifests/worldsim-v73-20260907/`。三本研究台账已同里程碑同步。磁盘暂不构成研究阻塞，继续原生DPT训练；GPU资源需实测再判断。
