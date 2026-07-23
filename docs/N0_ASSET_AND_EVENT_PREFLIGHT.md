# N0 资产与事件上限预检

> **日期**：2026-07-24
> **状态**：`READ_ONLY_PREFLIGHT_COMPLETE / VECTOR_MAP_MISSING`
> **范围**：现有 nuScenes mini、DriveStudio 10 Hz 派生数据、H1 冻结 actor pool。
> **未执行**：未下载数据、未修改标注、未启动 GPU run。

## 1. 结论

现有数据足以确认 H1 proposal bank 的上游弱点，但不足以运行正式 event gate：

- 3 个处理场景实际对应 2 张地图：`boston-seaport` 与 `singapore-queenstown`；
- 本机缺 `maps/expansion/*.json`，所以不能正式查询 lane、lane_connector、connectivity 或 drivable polygon；
- 冻结 6 actors 的轨迹都连续，但其中 3 个在完整 track span 内位移不足 1 m；
- 因此“track 很长/可见帧多”不等于“适合产生 cut-in/merge 事件”；
- 下一步需要的最小外部资产是两张相关地图的官方 map-expansion JSON，而不是全量 nuScenes/Waymo/nuPlan。

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

## 5. 获得授权后的 N0 smoke

1. 下载到临时目录，核对来源、许可、大小、version 和 SHA256；
2. 只读安装到 `maps/expansion/`，不覆盖当前 raster PNG；
3. 对两张地图实例化 `NuScenesMap`；
4. 对三场景所有 ego/selected-actor poses 查询 closest lane、drivable layer 与 incoming/outgoing connectivity；
5. 验证 world/map 坐标不需隐式变换；保留 round-trip fixture；
6. 生成 immutable `asset_manifest.json` 和 `scene_map_registry.json`；
7. 若任何 selected actor 长时间没有附近 lane，先审计坐标/场景语义，不放宽搜索半径救结果。

## 6. 停止与授权边界

当前已到明确外部边界：官方 map-expansion 文件尚未存在，本轮没有数据下载授权。没有该资产时可以继续写
N1 schema/tests，但不能给出正式事件标签或运行 lane-based proposal。

若用户授权，建议只批准 nuScenes 官方 map expansion 包/上述两张 JSON，不批准全量 Waymo、nuPlan 或大型
模型权重。后者只有在 mini event pool 被正式 reject 后才重新评估。
