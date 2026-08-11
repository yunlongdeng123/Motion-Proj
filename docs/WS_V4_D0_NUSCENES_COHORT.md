# WorldSim V4 D0 nuScenes Cohort Freeze

- Task：`WS-V4-D0-NUSCENES-COHORT-01`
- 状态：`done`
- 冻结日期：2026-08-11
- canonical run：`20260811T084108Z__d0-cohort-formal-s40117-r4`
- cohort SHA-256：`eda9f6847d2d9d01ce813c06f550aa2a0f5cf9a23ee8ab3ba766911acb144578`
- selection seed：`40117`
- 训练 / 模型推理 / test quality 读取：`0 / 0 / 0`

## 1. 结论

D0 从官方 nuScenes `v1.0-trainval` 元数据枚举 `850` 个候选 scene，仅使用结果前可观测的地点、时间、天气、
道路形态、actor 支持、遮挡、运动、距离和 donor metadata proxy，冻结出 scene-disjoint 的 `6 development + 6
validation + 18 test`。development/validation 全来自官方 `train`，test 全来自官方 `val`；任何方法结果、模型质量
或 test quality 均未进入选择。

冻结配置保存全部 30 scene 的 actor、三类 edit、连续片段、相机/LiDAR 合同和逐帧 train/development/heldout
划分。正式构建不仅比较 scene 名单与 cohort SHA，还逐字段比较完整 `scene_records`，漂移即失败。

## 2. 冻结 Scene

| Role | Scene |
|---|---|
| development | `scene-0230`, `scene-0242`, `scene-0255`, `scene-0048`, `scene-0994`, `scene-0139` |
| validation | `scene-0071`, `scene-1089`, `scene-0317`, `scene-0862`, `scene-1012`, `scene-0450` |
| test | `scene-0919`, `scene-0100`, `scene-0520`, `scene-0634`, `scene-1062`, `scene-0626`, `scene-0015`, `scene-0552`, `scene-0924`, `scene-0906`, `scene-0519`, `scene-0781`, `scene-1072`, `scene-0554`, `scene-0911`, `scene-0966`, `scene-0800`, `scene-0632` |

前三个 development scene 只作为 pre-V4 已有 preprocess infrastructure anchor；选择理由没有读取其 V3/V3.3
质量结果。test scene 在 M1/M2/M3 之前已冻结并哈希，后续不得因结果替换。

## 3. 分层覆盖

- 时间：day=`21`、dusk=`4`、night=`5`；天气：rain=`6`、dry/unspecified=`24`。
- 道路：intersection=`6`、turn/curve=`4`、roundabout=`3`、road segment=`5`、general urban=`12`。
- actor：car=`14`、bus=`4`、bicycle=`3`、motorcycle=`3`、construction=`2`、trailer=`2`、truck=`1`、
  emergency=`1`。
- 速度：stationary=`8`、low-speed=`4`、normal-speed=`18`；距离：near=`14`、mid=`14`、far=`2`。
- occlusion：heavy=`28`、normal=`2`；donor proxy：strong=`3`、medium=`4`、weak=`23`。

30/30 scene 都找到 high-support actor 与 difficult actor；ABSTAIN 合同仍由 validator 覆盖，未来重建如找不到 actor
必须保留 denominator 并显式写 `ABSTAIN_NO_ACTOR`。连续片段均为 7 个关键帧、`2.899163–3.150218 s`，满足
冻结的 2–4 秒合同。

## 4. Frame 与 Sensor 合同

- 相机：`CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`；LiDAR：`LIDAR_TOP`。
- 帧按 scene 内时间顺序确定性划分：`index % 5 == 4` 为 heldout，`index % 5 == 2` 为 development，其余 train。
- 每个 scene 的三个 frame 集合完整、互斥；scene name/token 在 cohort 内唯一。
- 每个 scene 冻结 `remove / lateral / insert` 与同一个 high-support actor，并冻结一条 continuous clip。

## 5. 两 Scene Preprocess Smoke

smoke 只复用既有 DriveStudio 10 Hz 产物，不训练、不推理、不读取质量：

| Scene | Existing index | Frames | Images | LiDAR | Result |
|---|---:|---:|---:|---:|---|
| `scene-0230` | 179 | 196 | 1,176 | 196 | passed |
| `scene-0242` | 191 | 196 | 1,176 | 196 | passed |

检查覆盖首尾图像、首尾 LiDAR、`instances_info.json` 与 `frame_instances.json` 的存在、非空、JSON 可读及 SHA-256。
这只证明 adapter/preprocess 产物布局可用，不是 baseline 或方法质量证据。

## 6. Canonical Evidence

目录：

```text
/root/autodl-tmp/runs/worldsim_v4/WS-V4-D0-NUSCENES-COHORT-01/
  20260811T084108Z__d0-cohort-formal-s40117-r4
```

| Artifact | SHA-256 |
|---|---|
| `resolved.yaml` | `ed47c0da2c76e14b3b0a0e7a8b4d9b580bdf37e4c69a1d5a389b965e88c667a1` |
| `summary.json` | `ec96970d8733e99b206048baf463fce69cae99db916c15b1e4fd777a74d4f276` |
| `manifest.json` | `3349a63667988c61596494506c67cf4d3b7f36e934ab4fac5d0935974c0d6b30` |
| `status.json` | `1dfd5db4e71566c344aa382e9f8e464c0b512cb01ff8a6053a03123bd3cb4461` |
| `artifacts/nuscenes_cohort.json` | `eda9f6847d2d9d01ce813c06f550aa2a0f5cf9a23ee8ab3ba766911acb144578` |
| `artifacts/nuscenes_candidates.jsonl` | `5be022825b7eb98bfc9ddbd1b22e85e1bdf9b9b9d23e8fda9b647b05bf73079f` |
| `artifacts/smoke.json` | `ae6ae8dedb0633b3a88d0aa5b341ce27ec34aea3f491985f7fece08e194d8e76` |

run 含 `12` 个文件、`1,221,825 bytes`。metadata 的 11 个官方 JSON 表逐文件记录 bytes/SHA-256；诊断 r1
保留为 noncanonical。r2 因提交门禁将冻结 YAML 从 CRLF 规范成 LF 而失去 source-byte canonical 身份；r3 的
未排序 set 浮点求和导致 cohort SHA 漂移，被 freeze gate 拦截并补录 immutable `blocked` terminal；r4 固定 tag
排序、跨三个 `PYTHONHASHSEED` 验证后重建出原 cohort SHA。D0 定向测试=`8 passed`，D0/P0 与 V3.3/V3.2 联合
回归=`106 passed`。

## 7. 边界与下一门禁

- D0 没有展开 549 GB sensor archive；仅从 public disk 解压官方 metadata archive。
- donor support 是 metadata proxy，不冒充真实重建 donor 质量。
- D0 不证明 30 scene 已训练或可渲染，不产生 PSNR/SSIM/LPIPS 等质量结论。
- `/root/autodl-pub/KITTI` 仍缺失；D1 必须保持 `blocked_local_dataset_missing`，禁止下载绕过。
- 下一步实现 D1 layout-detecting adapter 的 fail-closed 合同，然后推进 6-development-scene matched baseline B0；M1
  必须等 B0 evaluator 稳定复现后才解锁。
