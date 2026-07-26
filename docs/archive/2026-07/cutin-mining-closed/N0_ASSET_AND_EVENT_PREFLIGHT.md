# N0 资产与事件上限预检

> **日期**：2026-07-24
> **状态**：`N0_ASSET_COMPLETE / N1_MINI_EVENT_POOL_REJECTED`
> **范围**：现有 nuScenes mini、DriveStudio 10 Hz 派生数据、H1 冻结 actor pool。
> **执行说明**：用户提供 map-expansion v1.3；未修改标注、未使用 GPU。

## 1. 结论

只读预检时的资产缺口已解除，N0 正式通过；随后的 N1 event gate 正式拒绝：

- 3 个处理场景实际对应 2 张地图：`boston-seaport` 与 `singapore-queenstown`；
- map-expansion v1.3 四图均可加载并查询 lane、lane_connector、connectivity 与 drivable polygon；
- 冻结 6 actors 的轨迹都连续，但其中 3 个在完整 track span 内位移不足 1 m；
- 因此“track 很长/可见帧多”不等于“适合产生 cut-in/merge 事件”；
- N1 扩展到所有 source-only eligible actors 后仍为 positive=0、same-actor pair=0，mini pool 已 reject。

## 2. scene→map provenance

DriveStudio converter 直接使用 `self.nusc.scene[scene_idx]`，并把 `scene_idx` 格式化为三位处理目录。因此
`003/005/004` 是 nuScenes mini `scene.json` 的零基索引，不是 `scene-003/005/004` 名称。

| processed scene | nuScenes scene | token | samples | location | log |
|---|---|---|---:|---|---|
| `003` | `scene-0655` | `bebf5f5b2a674631ab5c88fd1aa9e87a` | 41 | `boston-seaport` | `n008-2018-08-27-11-48-51-0400` |
| `005` | `scene-0796` | `c5224b9b454b4ded9b5d2d2634bbda8a` | 40 | `singapore-queenstown` | `n015-2018-10-02-10-50-40+0800` |
| `004` | `scene-0757` | `2fc3753772e241f2ab2cd16a784cc680` | 41 | `boston-seaport` | `n008-2018-08-30-15-16-55-0400` |

事实源：

- `/root/autodl-tmp/data/nuscenes/v1.0-mini/scene.json`；
- `/root/autodl-tmp/data/nuscenes/v1.0-mini/log.json`；
- `/root/autodl-tmp/third_party/drivestudio/datasets/nuscenes/nuscenes_preprocess.py`。

## 3. 冻结 actor continuity

`instances_info.json` 的 `frame_idx` 与 `obj_to_world` 只读审计：

| scene | actor | class | frames | frame span | gaps | first→last world displacement |
|---|---:|---|---:|---|---:|---:|
| 003 | 38 | `vehicle.construction` | 191 | 10–200 | 0 | 0.88 m |
| 003 | 35 | `vehicle.car` | 131 | 5–135 | 0 | 0.29 m |
| 005 | 23 | `vehicle.car` | 151 | 5–155 | 0 | 0.76 m |
| 005 | 20 | `vehicle.car` | 191 | 5–195 | 0 | 194.83 m |
| 004 | 4 | `vehicle.bus.rigid` | 171 | 0–170 | 0 | 97.28 m |
| 004 | 8 | `vehicle.car` | 46 | 15–60 | 0 | 30.88 m |

场景 inventory：

| scene | total actors | `vehicle.*` actors | tracks ≥20 frames |
|---|---:|---:|---:|
| 003 | 117 | 110 | 110 |
| 005 | 50 | 39 | 40 |
| 004 | 24 | 22 | 23 |

这些数字只证明 annotation 连续性和粗位移；没有 lane graph 时，不能把移动 actor 标成 lane-change/cut-in，
也不能把静止 actor 自动标为 negative。它们足以说明 H1 的 source-only 排序
`visible_frame_count / track_frame_count / lidar_point_count` 会偏向长而清晰的 track，却没有事件相关性。

## 4. 最小 map 资产

官方 `NuScenesMap` 从：

```text
<dataroot>/maps/expansion/<map_name>.json
```

读取 map，并要求 map version 不低于 `1.3`。当前 pilot 只需要：

```text
/root/autodl-tmp/data/nuscenes/maps/expansion/boston-seaport.json
/root/autodl-tmp/data/nuscenes/maps/expansion/singapore-queenstown.json
```

需要的层：

- `polygon`、`line`、`node`；
- `drivable_area`、`road_segment`、`road_block`；
- `lane`、`lane_connector`；
- `arcline_path_3`、`connectivity`；
- 可选诊断：`lane_divider`、`road_divider`、`stop_line`、`traffic_light`。

官方实现与路径：
[nuScenes map API](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/map_expansion/map_api.py)。

## 5. N0 正式结果

用户提供：

`/root/autodl-tmp/nuScenes-map-expansion-v1.3.zip`

压缩包测试无错误，大小 `398,535,531 bytes`，SHA256：

`9dbc80a095b6b28d9b79fc9a43471a750dc92ca78c6d0db288fd92b34be5a144`

只安装 `expansion/`、`prediction/` 与 `LICENSE`，未覆盖 raster PNG。正式 run：

`/root/autodl-tmp/runs/event_first/N0-ASSET-01/v71_n0-asset-01__map-v1-3__s0__20260723T232427413355Z__e250cccd/`

| gate | observed | verdict |
|---|---:|---|
| 四图 version | 4/4 为 `1.3` | pass |
| 必需 map layers | 4/4 完整 | pass |
| scene→map exact | 3/3 | pass |
| raw/processed keyframe pose translation | max `0 m` | pass |
| raw/processed keyframe pose rotation | max `1.01065e-7 rad` | pass |
| ego closest-lane | 121/121 | diagnostic pass |
| selected actor closest-lane | 003 `266/322`；005 `342/342`；004 `154/217` | diagnostic |

asset manifest SHA 为 `48e8ace8…c60a7`，scene-map registry SHA 为 `7c83e936…560c1`，唯一终态
`COMPLETE`。

selected actor lane coverage 不作为 N0 fail gate；它暴露的是旧 actor pool 含静止/非道路位置对象。N1
因此重新使用预注册 source-only eligibility 扫描全部 vehicle tracks，没有通过挑 actor 修补旧结果。

## 6. N1 停止边界

N1 正式 run 得到 45 eligible actors、71 transitions、22 topology-pass candidates，但 interaction
PASS 为 0；`REJECTED / reject_mini_event_pool`。完整分解见
[`N1_MINI_EVENT_POOL_REPORT.md`](N1_MINI_EVENT_POOL_REPORT.md)。

不得在这三个 scenes 上降低 front/rear、gap、稳定帧或 map-match 门槛翻案。下一次数据动作应是新的明确授权：
优先 nuScenes `v1.0-trainval` annotations/metadata；传感器 sweeps、Waymo、nuPlan 与大模型仍不在授权内。
