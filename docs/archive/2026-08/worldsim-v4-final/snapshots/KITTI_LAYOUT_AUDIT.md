# WorldSim V4 KITTI Layout Audit

- Task：`WS-V4-D1-KITTI-ADAPTER-01`
- 状态：`blocked`
- 原因：`blocked_local_dataset_missing`
- canonical blocked run：`20260811T085210Z__d1-kitti-layout-formal-s0-r2`
- 请求路径：`/root/autodl-pub/KITTI`
- 物理路径：`/autodl-pub/data/KITTI`
- 下载 / training / quality read：`0 / 0 / 0`

## 1. 实际 Layout

`/root/autodl-pub` 是指向 `/autodl-pub/data` 的符号链接。请求路径与解析后的物理路径当前都不存在；D1 没有创建
空目录，也没有使用 `wget/curl/aria2/kaggle/gdown` 或任何替代下载。因此本机不能执行两 sequence adapter smoke，
更不能执行 10-sequence cross-domain quality。

这属于外部数据挂载阻塞，不是 adapter 算法失败。P0 的 [`WS_V4_KITTI_AUDIT.md`](WS_V4_KITTI_AUDIT.md) 保留为
启动审计；本文件记录 D1 实现后的 formal terminal。

## 2. 已实现 Adapter 合同

D1 新增 tracking-first / raw-fallback 自动发现：

- Tracking：`image_02`, `image_03`, `velodyne`, `label_02`, `calib`，并发现可用 `poses/oxts/timestamps`；
- Raw：`date_drive_sync/image_02|image_03|velodyne_points|oxts`、tracklets 与 date-level calibration；
- 相机只使用 KITTI 原生两路彩色 `image_02/image_03`，不伪造第三相机；
- 解析 tracking calibration、track ID/3D box、LiDAR，支持 LiDAR→rectified camera→image 与 3D box projection；
- sequence 内 frame 按确定性规则划分 train/development/heldout，保持互斥；
- 两 sequence 只标记 `adapter_smoke`，目标 formal 数为 10；KITTI 不允许重搜方法阈值。

synthetic tracking/raw fixtures 覆盖 12 项冻结门：meter/axis/handedness、`T_velo_cam`、rectification、intrinsics、
timestamp order、track ID、3D box projection、LiDAR projection、object/world/camera chain、stereo association、
heldout leak 与 deterministic manifest hash。D1 定向测试=`7 passed`。
D1/D0/P0 与 V3.3/V3.2 联合回归=`113 passed`。

## 3. Canonical Evidence

目录：

```text
/root/autodl-tmp/runs/worldsim_v4/WS-V4-D1-KITTI-ADAPTER-01/
  20260811T085210Z__d1-kitti-layout-formal-s0-r2
```

| Artifact | SHA-256 |
|---|---|
| `resolved.yaml` | `bffe6eaad29e352ccd31812bc58d634fc656bb5e33f7b5535b0d7e64ebf893d4` |
| `summary.json` | `05eeb26542c761033afc0cae4e7c42ac861ea02cc22941e76d4f9df65df39ca9` |
| `manifest.json` | `7c9ca256486968d36001bd7a10bfeb283e2157186689356e15e54deab06af582` |
| `status.json` | `598eac510f0d19edc2851202802257fc3f96874e343e93c572a145aee779d54b` |
| `fingerprint.json` | `233e464e8bb4c7bcc1c225b1e4f6e2395267358f947a2914968f766e75e8fdc5` |
| `artifacts/layout_audit.json` | `9641705e514098a01d1e833de585a9db936dbeefc5d81cdbc2b43137d1191e28` |
| `artifacts/kitti_manifest.json` | `f90d95500d386ffa92b6f1d7d5fd32c6199af44f9f650e2882f17411c45ffcf6` |

run 含 `14` 个文件、`45,214 bytes`；5 份 source snapshot、requested/resolved root、root existence 与 no-download
标记均被 fingerprint。r1 正确 blocked，但只记录物理路径；r2 同时记录请求路径与 symlink 解析路径，故为 canonical。

## 4. 合法复开条件

仅当用户把真实 KITTI 挂载到 `/root/autodl-pub/KITTI` 后，以新 run ID 重新执行：

1. tracking training 优先；若不完整再审计 raw；
2. 12/12 adapter gates 与两 sequence smoke 全通过；
3. 冻结 exact manifest SHA；
4. 不改 nuScenes 已冻结的方法阈值；
5. 通过后才解锁 10-sequence cross-domain formal。

在此之前，D1 保持 `blocked`，E1 保持 `pending`；该阻塞不妨碍先完成 nuScenes B0/M1/M2/M3 development 与
validation，但 single-card closure 仍不能宣称完成。
