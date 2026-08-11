# WorldSim V4 P0：KITTI 本地布局审计

- 日期：2026-08-11
- Task：`WS-V4-P0-SCOPE-PAPER-FREEZE-01`
- 预期根目录：`/root/autodl-pub/KITTI`
- 审计方式：只执行本地 `test -d` / `find` / `du` 类元数据检查
- 网络下载：`false`
- 终态：`blocked_local_dataset_missing`

## 1. 实际事实

P0 在 AutoDL 主机上检查 `/root/autodl-pub/KITTI`，目录不存在。因此当前无法判定公共盘内容属于 tracking、raw、
odometry 或 object layout，也不能生成 sequence/camera/LiDAR/calibration/track manifest。

这不是算法质量失败，也不是 adapter 投影失败。它只表示计划中“公共盘已经挂载”的外部前提与当前机器状态不一致。
P0 不创建空目录冒充数据、不从网络下载 KITTI，也不把其他数据复制到该路径。

## 2. 恢复条件

只有 `/root/autodl-pub/KITTI` 重新可见后，`WS-V4-D1-KITTI-ADAPTER-01` 才能启动，并按顺序：

1. 自动发现 tracking training 或 raw sync layout；
2. 冻结 `image_02/image_03`，不伪造第三相机；
3. 验证 meter/axis/handedness、`T_velo_cam`、rectification、intrinsics、timestamp、track ID、3D box/LiDAR
   投影、object-local→world→camera、stereo association、heldout leak 与 deterministic manifest hash；
4. 只对两个 sequence 做 adapter smoke；
5. 所有方法参数保持 nuScenes frozen，不在 KITTI 调 calibration/risk/router/B-spline/threshold。

任一 12 项 adapter exact gate 失败时，终态必须是 `blocked_dataset_adapter`，不得输出质量表。当前缺目录状态不会阻塞
`WS-V4-D0-NUSCENES-COHORT-01`，但会阻塞 single-card closure 中的 KITTI 部分，直到外部数据挂载恢复。
