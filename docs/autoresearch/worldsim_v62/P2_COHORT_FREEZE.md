# WorldSim V6.2 P2 development cohort 冻结

- Task：`WS-V62-P2-EVIDENCE-QUERY-DATASET-01`
- Hypothesis：`WS-V62-H-P2-001`
- Substage：`cohort_freeze`
- 状态：`done_metadata_only_cohort_frozen`

## 选择规则

完整采用 `configs/worldsim_v4/nuscenes_cohort_v1.yaml` 在任何 V6.2 结果之前冻结的 validation 六场景，不从中删除或
增补场景。该 V4 cohort 的选择仅使用 nuScenes metadata；本次额外确认 6/6 都在官方 train split、processed 传感器完整，
并排除 legacy scene-0048/0242。没有读取 Occupancy quality、proposal、render 或 evaluator outcome。

## Cohort

| scene | index | frames | location | time/weather | metadata actors |
|---|---:|---:|---|---|---:|
| scene-0071 | 68 | 196 | boston-seaport | day / dry | 32 |
| scene-0317 | 251 | 191 | singapore-queenstown | day / dry | 6 |
| scene-0450 | 364 | 196 | boston-seaport | day / rain | 71 |
| scene-0862 | 652 | 196 | singapore-queenstown | dusk / dry | 11 |
| scene-1012 | 770 | 196 | singapore-queenstown | night / dry | 8 |
| scene-1089 | 829 | 196 | singapore-hollandvillage | night / dry | 7 |

每帧处理目录具备六相机 image/extrinsics、六个 intrinsics、LiDAR 与 LiDAR pose。该检查只看存在性与计数。

## Target 与 evidence roles

每场固定 12 targets：

```text
17, 32, 47, 62, 77, 92, 107, 122, 137, 152, 167, 182
```

每个 target：

```text
method candidates  = [-6, -4, -2, 0]
dropout target     = 按 target ordinal 在四个 method candidates 中轮换一个
visible E_input    = 剩余三个 method candidates
independent target = [-5, -3, -1, 1]
```

因此 72 target units 的 method/dropout/target source role 在单元内互斥；target stride=15，也避免相邻单元复用同一
source frame。confirmation/test 没有解析、列举或读取。

## 下一步

按 `configs/worldsim_v62/p2_development_cohort_v1.yaml` 实现 geometry/evidence query materializer。首轮只构建训练所需的
sparse evidence/query sidecar；不加入 prior 或质量选择，P4 再为同一坐标附加 frozen IR-WM logits/features。
