# 多视角道路来源与静态透明度交接

这条正式代码通路先生成固定道路资产，再训练选中静态行的透明度门。原生30K检查点、道路几何与颜色、车辆运动及环境参数均不更新。来源认证完整并不等于编辑成功；本轮结果仍是负结果。

| 模块 | 输入与输出职责 |
|---|---|
| `v44_road_annotations` | 从原始TFRecord导出全部类型的3D框和真实FRONT相机2D框；核对场景、时间戳和帧合同。 |
| `v44_road_geometry` | 从显式测量种子、相机、轨迹、标签及测距生成有限范围查询地面网格；不读取RGB或外观ROI。输出 `geometry.npz`。 |
| `v44_road_observation` | 局部测量面、父格三角证据认证、子格资格继承、颜色采样及独立推断的通用算子。 |
| `v44_road_multiview` | 将输入帧接入上述算子，输出固定 `reference.npz`、逐帧父格证据 `parents/`、来源身份和训练域 `views/`。 |
| `v44_road_gate_fit` | 读取显式场景manifest、训练域与原生KEEP/DELETE基线；仅优化无车辆语义归属的静态候选透明度门。输出候选索引、完整checkpoint、覆盖统计及恢复记录。 |

几何准备已独立重做；`keys`、`local_xyz`、`world_xyz`、`model_from_world` 与本轮使用的原几何种子逐值一致。几何模块不依赖旧外观ROI。多视角来源认证另从归一化初始平面15厘米范围内的测距点取地面，剔除全部3D物体框内点，按10厘米XY去重并拟合局部测量面。查询范围由全部目标地面射线及显式限距决定；来源图像质量按全部199帧的中位数归一化。这些是当前输入适配，不宣称与旧场景逐位一致。

8厘米父格以真实测距三角、局部法向、投影尺度、2D前景保护和3D框射线遮挡共同认证来源。2厘米子格继承对应逻辑父格的来源资格，重新检查投影与二维保护，再从经Lanczos缩放的sRGB图像双线性取色并转为线性RGB；子格没有独立的测距三角认证。参考颜色至少需要3个合格视角一致。缺失处仅允许可靠沥青颜色的8近邻推断：最近邻距离不超过0.6米，置信权重0.25；不是要求8个邻居都在0.6米内。真实认证、推断和未知分别保存，未知不产生隐藏RGB监督，也不使用手工颜色ROI补源。

原生KEEP渲染只用于受限的干净路面颜色对齐及保持损失，不充当被遮挡道路真值。标线、其他车辆、剩余物体框射线遮挡及独立动态贡献不低于0.25的区域参与保护。DELETE使用道路参考与外围保持，KEEP另对目标车辆施加权重2的原生RGB保持损失。

## 显式命令入口

从仓库根目录运行，配置与manifest均须使用对应数据的明确路径。`BANK`为图像、测距、相机与轨迹输入；`ANN`为原始注释导出；几何配置声明种子及范围，多视角配置声明几何、注释、来源帧和保护基线；场景manifest声明冻结模型、实例掩码、帧和道路资产。

```bash
export PYTHONPATH="$PWD/scripts/worldsim_v82"
python -m drivingeditor_dynamic_edit.v44_road_annotations \
  --bank "$BANK" --parser-dir "$PARSER_DIR" --output "$ANN"
python -m drivingeditor_dynamic_edit.v44_road_geometry \
  --config "$GEOMETRY_CONFIG" --output "$GEOMETRY_OUT"
python -m drivingeditor_dynamic_edit.v44_road_multiview \
  --config "$SOURCE_CONFIG" --output "$ROAD_OUT"

fit_args=(--manifest "$MANIFEST" --views "$VIEWS" \
  --baseline "$DELETE_BASELINE" --keep-baseline "$KEEP_BASELINE" \
  --delete-track-id "$TRACK_ID")
python -m drivingeditor_dynamic_edit.v44_road_gate_fit \
  "${fit_args[@]}" --output "$FIT32" --steps 32
python -m drivingeditor_dynamic_edit.v44_road_gate_fit \
  "${fit_args[@]}" --output "$FIT1000" --steps 1000 \
  --resume "$FIT32/checkpoint000032.pt"
```

多视角配置的 `raw_annotations` 指向 `$ANN`，`geometry_seed` 指向 `$GEOMETRY_OUT/geometry.npz`；场景manifest的 `road` 指向 `$ROAD_OUT/reference.npz`，`$VIEWS` 指向 `$ROAD_OUT/views`。`views/manifest.json`必须将全部场景帧唯一绑定到道路视图和DELETE基线。训练按固定种子打乱后完整遍历199帧；无道路颜色监督的帧仍参与KEEP与外围保持。1000步中每帧训练5或6次。恢复检查候选身份、输入摘要、优化器、随机状态、剩余帧队列及轮次；上述32步加精确恢复968步合计1000次更新，`--steps`表示最终累计步数。最终场景manifest通过原有 `static_gates.indices` 与 `static_gates.checkpoint` 显式引用导出的索引和checkpoint，再使用既有渲染入口。

原factory和scene已从备份精确恢复为原HEAD。RGB混合、单位缩放包装器及其测试已移入实验归档，没有作为正式默认行为。正式CPU检查26项通过（18项交接策略、8项几何与可见性）。原生光栅器近裁剪及物理深度问题尚未修复。

## 本轮结果与边界

以下为目标区域内的线性RGB来源代理MAE，计数单位是像素帧。各行的前后对比固定同一参考及同一域；新旧域之间不能直接比较以宣称整体提升。

| 评价域 | 像素帧 | 更新前 | 更新后 |
|---|---:|---:|---:|
| 新来源认证域 | 1,923 | 0.030195 | 0.012179 |
| 新来源推断域 | 274,606 | 0.047930 | 0.026604 |
| 固定旧来源认证域 | 70,853 | 0.024741 | 0.043199 |
| 固定旧来源推断域 | 436,293 | 0.029968 | 0.045930 |

新来源小范围误差下降，固定旧域误差反而上升，KEEP局部仍出现车辆切片。来源重建与局部来源误差改善不能作为编辑成功的证据，也没有独立的被遮挡道路真值。原生30K模型保持原样，029场景未改动；该候选未替换原平台结果。
