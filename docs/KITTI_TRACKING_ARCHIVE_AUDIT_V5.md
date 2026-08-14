# WorldSim V5 KITTI Tracking 压缩包与 Metadata 审计

- Task：`WS-V5-D1-KITTI-ARCHIVE-AUDIT-01`
- 日期：`2026-08-14`
- 状态：`blocked_dataset_adapter`
- archive root：`/root/autodl-tmp`
- manifest SHA-256：`56388fc64e36c77ebac5a6ee761aa1a17297faeb876347715e8c6e9d52ec23a7`
- download / extraction / quality / training / parameter search：`0 / 0 / 0 / 0 / 0`

## 1. 数据结论

7 个压缩包的 central directory 均可读，内容布局对应 KITTI Tracking。解压后应形成 `training|testing/image_02|image_03|velodyne`、`label_02`、`oxts`、`calib`；原生相机合同固定为 `image_02/image_03`，不构造第三相机。训练序列=`21`，测试序列=`29`，训练/测试 frame=`8008/11095`。

官方 testing split 不含 `label_02`，因此 V5 的可量化 cross-domain pool 只能从 21 个 training sequences 中冻结；testing split 可用于无标签 adapter/engineering smoke，不能进入带 GT 的主表。

## 2. 压缩包清单

| Component | Archive GiB | Files | Uncompressed GiB | SHA-256 | Payload CRC |
|---|---:|---:|---:|---|---|
| `velodyne` | `34.300` | `19099` | `34.297` | `15dfa06f23b75a8ca3f31d17820f64b3a485abac9e2f6346fcc83731a9ab72c0` | `not_run_large_archive` |
| `image_02` | `14.727` | `19103` | `14.724` | `9206250fbcd0074328bf589ec81ab0b0681608db59537f024b1b9ff44e81d53c` | `not_run_large_archive` |
| `image_03` | `14.056` | `19103` | `14.053` | `146a7c6416da734df797100f2813fe37bb0314b950b4593153800101380cb913` | `not_run_large_archive` |
| `label_02` | `0.002` | `21` | `0.009` | `6ad1fa0125e3cddcbe4d182e41cd12a854a11b0c733b26c706595f6db5c254e0` | `verified` |
| `oxts` | `0.008` | `50` | `0.008` | `0b93e47a01d9479b7aeffcad50c0a08566f723119b6dfb7a11da3466acaf02de` | `verified` |
| `calib` | `0.000` | `50` | `0.000` | `281b8824e82a0710685d0ab641d3f243cffb7af05aebfda550ecfa225742c463` | `verified` |
| `devkit` | `0.000` | `20` | `0.000` | `17091016def85820bc18a7a350b625f3431b585b6903f778beaf49cbf444270c` | `verified` |

压缩包总计约 `63.094 GiB`（`67.747 GB`），未额外执行全量 ZIP entry CRC 解码；本次读取了全部 archive bytes 生成 SHA-256，并对小包执行 `ZipFile.testzip()`。解压 staging 后仍需按 member size、frame alignment 和抽样 payload 再审计。

## 3. Gate

| Gate | Result |
|---|---|
| `all_archives_present` | `PASS` |
| `central_directories_readable` | `PASS` |
| `archive_sha256_recorded` | `PASS` |
| `no_duplicate_members` | `PASS` |
| `safe_member_paths` | `PASS` |
| `unencrypted_members` | `PASS` |
| `expected_component_paths` | `PASS` |
| `expected_sequence_sets` | `PASS` |
| `sensor_frame_alignment` | `FAIL` |
| `label_frames_within_sensor_frames` | `PASS` |
| `oxts_row_count_matches_frames` | `PASS` |
| `calibration_keys_present` | `PASS` |
| `devkit_seqmap_matches_sensor_frames` | `PASS` |
| `small_archive_payload_crc_verified` | `PASS` |
| `native_stereo_contract` | `PASS` |
| `disk_space_for_extract_with_margin` | `PASS` |

### 严格 frame alignment blocker

- `training/0001`：frame counts=`{'image_02': 447, 'image_03': 447, 'velodyne': 443}`，missing relative to image_02=`{'image_03': [], 'velodyne': ['000177', '000178', '000179', '000180']}`，extra=`{'image_03': [], 'velodyne': []}`。

在确认这是否为官方已知缺帧并冻结 common-frame/abstain 规则前，真实 adapter smoke 保持 blocked；不得静默取交集后把 coverage 写成完整序列。

## 4. 逐序列 Metadata

完整逐序列记录见同目录 `KITTI_TRACKING_ARCHIVE_METADATA_V5.json`。训练序列字段包含 stereo/LiDAR frame 计数、label 行数、annotated frame、track ID 与 class 分布、OXTS 行宽和 calibration keys；测试序列不虚构 label。

| Split | Seq | Frames | Label rows | Tracks | OXTS rows | Sensor exact |
|---|---:|---:|---:|---:|---:|---|
| `training` | `0000` | `154` | `1089` | `15` | `154` | `yes` |
| `training` | `0001` | `447` | `4271` | `98` | `447` | `no` |
| `training` | `0002` | `233` | `2098` | `20` | `233` | `yes` |
| `training` | `0003` | `144` | `861` | `9` | `144` | `yes` |
| `training` | `0004` | `314` | `2012` | `41` | `314` | `yes` |
| `training` | `0005` | `297` | `2148` | `36` | `297` | `yes` |
| `training` | `0006` | `270` | `1446` | `15` | `270` | `yes` |
| `training` | `0007` | `800` | `3722` | `63` | `800` | `yes` |
| `training` | `0008` | `390` | `2088` | `28` | `390` | `yes` |
| `training` | `0009` | `803` | `5301` | `89` | `803` | `yes` |
| `training` | `0010` | `294` | `1323` | `28` | `294` | `yes` |
| `training` | `0011` | `373` | `4359` | `60` | `373` | `yes` |
| `training` | `0012` | `78` | `354` | `4` | `78` | `yes` |
| `training` | `0013` | `340` | `2410` | `68` | `340` | `yes` |
| `training` | `0014` | `106` | `798` | `17` | `106` | `yes` |
| `training` | `0015` | `376` | `3495` | `26` | `376` | `yes` |
| `training` | `0016` | `209` | `4145` | `28` | `209` | `yes` |
| `training` | `0017` | `145` | `1499` | `11` | `145` | `yes` |
| `training` | `0018` | `339` | `1794` | `21` | `339` | `yes` |
| `training` | `0019` | `1059` | `11192` | `106` | `1059` | `yes` |
| `training` | `0020` | `837` | `8896` | `134` | `837` | `yes` |
| `testing` | `0000` | `465` | `N/A` | `N/A` | `465` | `yes` |
| `testing` | `0001` | `147` | `N/A` | `N/A` | `147` | `yes` |
| `testing` | `0002` | `243` | `N/A` | `N/A` | `243` | `yes` |
| `testing` | `0003` | `257` | `N/A` | `N/A` | `257` | `yes` |
| `testing` | `0004` | `421` | `N/A` | `N/A` | `421` | `yes` |
| `testing` | `0005` | `809` | `N/A` | `N/A` | `809` | `yes` |
| `testing` | `0006` | `114` | `N/A` | `N/A` | `114` | `yes` |
| `testing` | `0007` | `215` | `N/A` | `N/A` | `215` | `yes` |
| `testing` | `0008` | `165` | `N/A` | `N/A` | `165` | `yes` |
| `testing` | `0009` | `349` | `N/A` | `N/A` | `349` | `yes` |
| `testing` | `0010` | `1176` | `N/A` | `N/A` | `1176` | `yes` |
| `testing` | `0011` | `774` | `N/A` | `N/A` | `774` | `yes` |
| `testing` | `0012` | `694` | `N/A` | `N/A` | `694` | `yes` |
| `testing` | `0013` | `152` | `N/A` | `N/A` | `152` | `yes` |
| `testing` | `0014` | `850` | `N/A` | `N/A` | `850` | `yes` |
| `testing` | `0015` | `701` | `N/A` | `N/A` | `701` | `yes` |
| `testing` | `0016` | `510` | `N/A` | `N/A` | `510` | `yes` |
| `testing` | `0017` | `305` | `N/A` | `N/A` | `305` | `yes` |
| `testing` | `0018` | `180` | `N/A` | `N/A` | `180` | `yes` |
| `testing` | `0019` | `404` | `N/A` | `N/A` | `404` | `yes` |
| `testing` | `0020` | `173` | `N/A` | `N/A` | `173` | `yes` |
| `testing` | `0021` | `203` | `N/A` | `N/A` | `203` | `yes` |
| `testing` | `0022` | `436` | `N/A` | `N/A` | `436` | `yes` |
| `testing` | `0023` | `430` | `N/A` | `N/A` | `430` | `yes` |
| `testing` | `0024` | `316` | `N/A` | `N/A` | `316` | `yes` |
| `testing` | `0025` | `176` | `N/A` | `N/A` | `176` | `yes` |
| `testing` | `0026` | `170` | `N/A` | `N/A` | `170` | `yes` |
| `testing` | `0027` | `85` | `N/A` | `N/A` | `85` | `yes` |
| `testing` | `0028` | `175` | `N/A` | `N/A` | `175` | `yes` |

## 5. 存储与 staging

- 当前 filesystem free：`99.800 GiB`；
- 预计新增解压占用：`63.090 GiB`；
- 预留安全余量：`20.000 GiB`；
- 预计解压后 free：`36.710 GiB`；
- 推荐目标：`/root/autodl-tmp/data/kitti_tracking_v5`。

本次未解压。后续 staging 必须先写入同文件系统 `.partial` 目录，合并 6 个 data archives，运行完整 layout/frame/calibration/OXTS audit 后再原子发布；不得直接覆盖 `/root/autodl-pub/KITTI`，也不得删除原 zip。

## 6. 后续实验合同

1. 先做 2-sequence adapter smoke，只验证坐标、pose、track ID、stereo/LiDAR 对齐和确定性 manifest；
2. nuScenes V5 的 M1/M2/M3 参数完全冻结后，才从 training pool 冻结 10-sequence formal；
3. KITTI 禁止重搜 Bayesian、graph、geometry、router、temporal 或 kinematic 参数；
4. adapter/pose/calibration 失败写 `blocked_dataset_adapter`，方法在合格 adapter 上失败才写 cross-domain method failure；
5. testing split 无 label，不进入带 GT 的 cross-domain 质量主表。

## 7. 已知实现风险

V4 adapter 把 `oxts` 当作 3×4 pose 文本读取。metadata 中已记录实际 OXTS 每行字段宽度；V5 staging smoke 必须先验证其语义并显式转换为世界位姿，不能把行数对齐等同于 pose chain 已正确。官方 calibration 的 `R_rect`/`Tr_velo_cam` 行没有冒号，V4 parser 会忽略；V5 adapter 必须同时接受 colon 与 whitespace 两种 key 格式。
