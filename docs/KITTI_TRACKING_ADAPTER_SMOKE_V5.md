# KITTI Tracking V5 adapter smoke 元数据与使用说明

状态：`done`

任务：`WS-V5-D1-KITTI-ADAPTER-01`

日期：2026-08-14

## 数据位置与冻结范围

- 原始 ZIP：`/root/autodl-tmp/data_tracking_{calib,oxts,label_2,image_2,image_3,velodyne}.zip`；原 ZIP 未删除、未覆盖。
- 选择性抽取根目录：`/root/autodl-tmp/data/worldsim_v5/kitti_tracking_smoke`。
- raw manifest：`/root/autodl-tmp/data/worldsim_v5/manifests/kitti_tracking_smoke_raw_v1.json`。
- 本轮只冻结 labeled training sequences `0000/0001`，用途是 result-blind adapter smoke，不是方法质量评估或 cross-domain claim。
- 文件分母=`1805`，解压字节=`2,104,258,586`；raw manifest file/content SHA=`a3c77ab82bb29d2b34f615743a4e0393d611fb634826900c3756193127450928 / 98685653d4488b53a6ab94890629347b11cc43793b78b05c6b05738fcf46d83f`。

## 正式产物

- selective extraction run：`/root/autodl-tmp/runs/worldsim_v5/WS-V5-D1-KITTI-ADAPTER-01/20260814T121500Z__kitti-smoke-extract-0000-0001-s0-r001`；source=`5855bb0ee812b5d280e8a8d2ffc4512cfcf4d68b` clean。
- extraction summary/status/fingerprint/manifest SHA=`0ba90a7496d8f8a41dd147ef13579c9ad8d0a25aea7ee829d275ca162df1c363 / 091645899fa8b1f725d709e695fec664b44e79161c2846b8ad295f47718ca680 / 644845e4c3ae8ef05a49ed6daf399fbfed156a5a7bd569a6d65eb961bf2a1026 / 96585bf46127f2fd5eca0a123afe068be6f3922bc73e0362d3019af4c25bc8b3`。
- adapter run：`/root/autodl-tmp/runs/worldsim_v5/WS-V5-D1-KITTI-ADAPTER-01/20260814T122500Z__kitti-adapter-smoke-0000-0001-s0-r003`；source=`43fe090db160d6c9bceb6974937a4c20a2d7a760` clean。
- adapter summary/status/fingerprint/manifest SHA=`3b27cb9fa9b06f563b690cc44b1466e622b578bc88294b450ed254e8192a970b / 404b204ed1a7ff26a1bd6f277e80d6a2e4c66690ef29f0452678cb1506b76dd2 / 2df0f6535f63509f61fe6f72c483955a98c01a9010bd0b71bdfff7364ab5be56 / 099f136af9f519820412d0d3f25fbaaacb969706dd8e4536c7864b27c7fb90ec`。

## 序列合同

| sequence | image_02 | image_03 | LiDAR | full denominator | multimodal evaluable | coverage | explicit abstain |
|---|---:|---:|---:|---:|---:|---:|---|
| `0000` | 154 | 154 | 154 | 154 | 154 | 1.0000000000 | 无 |
| `0001` | 447 | 447 | 443 | 447 | 443 | 0.9910514541 | frames `177–180` 缺 LiDAR |

- `0000`：OXTS=`154`，label rows=`1089`，nonnegative track IDs=`15`。
- `0001`：OXTS=`447`，label rows=`4271`，nonnegative track IDs=`98`。
- 两序列 stereo baseline 均为 `0.5327254279298227 m`。
- 缺 LiDAR 帧禁止补值、删除 denominator 或静默取交集；调用 `freeze_sensor_frame_policy()` 获取完整 denominator、evaluable 集合与 abstain 明细。

## 坐标与解析合同

- calibration 同时接受 `P0:`–`P3:` 冒号行和无冒号的 `R_rect / Tr_velo_cam / Tr_imu_velo` 行，并验证 shape、finite 与旋转合同。
- OXTS 按官方 30-field 记录解析 latitude/longitude/altitude/roll/pitch/yaw，首帧为局部 Mercator 原点；不得当作 12-value `3×4` pose。
- LiDAR 到 rectified camera：`R_rect @ Tr_velo_cam`；camera pose 使用 `world_from_imu @ inverse(Tr_imu_velo) @ inverse(R_rect)`。
- smoke 已验证采样 PNG decode、全部 LiDAR 文件 `N×4 float32` 字节布局、采样点对 `P2/P3` 的前向有限投影、pose rotation determinant 与 label frame 范围。

## 实验使用边界

- adapter smoke 只证明数据布局、分母、坐标链和基础解码可用；`quality_read/method_training/method_inference/parameter_search=false`。
- 官方 testing split 无 `label_02`，不能用于需要 GT track/box 的质量主表。
- 后续若冻结 10-sequence cross-domain pool，必须从 21 个 labeled training sequences 结果前选择，并另建 manifest；不得依据本 smoke 的 0000/0001 指标选场景。
- `20260814T122000Z__...r002` 在 payload 读取前因直接脚本入口缺少 project import root 失败，未创建 run 目录；修复提交=`43fe090...`，r003 为新 ID，不覆盖该失败尝试。
