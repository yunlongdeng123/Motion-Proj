# 同一对象高斯场共同初始化

![Architecture components](architecture_components.svg)

`joint_actor_init.npz` 包含清洁旧场 8,925 行及完整 TRELLIS 410,080 行，合计 419,005 行。五项初始参数均按父源原值保留；两个来源的后续位置、尺度、旋转、透明度和颜色可按置信度联合更新。

旧场仅清退旧粗下半 624 行、精确 3σ 椭球与固定对象 OBB 不相交的 15,332 行，以及相交但来源混合其它车辆的 179 行。17 行真实不透明表面关联均保留为高几何置信度；其余 8,908 行保留为低几何置信度。没有用中心必须在框内或贴外壳作为选择条件。来源清退不删除原场景对应的道路、背景或其它车辆。

| 接口 | 约定 |
|---|---|
| 原始参数 | `xyz[N,3]` 米，`scales[N,3]` 激活标准差，`rotation[N,4]` wxyz，`coefficients[N,16,3]`，`opacity[N]` 激活值 |
| 来源 | `source_id`：旧场 0 / TRELLIS 1；`source_row_id` 为父 NPZ 行号；`native_global_row_id` 为原场景行号，TRELLIS 为 -1 |
| 置信度 | `geometry_confidence`、`appearance_confidence`、`opacity_confidence` 分离，均为先验约束权重 |
| 观测 | `is_observed` 为直接来源证据；TRELLIS `observed_surface` 为 0.075 米三维观测邻域 |
| 冻结教师 | 从初始化 TRELLIS 分区克隆不可变参数；`teacher_prior_weight=1-observed_surface_confidence` |
| 真实监督 | `real_V_supervision_weight` 和 `high_frequency_permission` 全行等于 1 |

TRELLIS 原直接观测支持为 12,865 行，三维邻域为 24,410 行，未知区域为 385,670 行。连续置信度为 `1-smoothstep(clamp(distance/0.075,0,1))`；未知区域教师先验权重为 1。该邻域只控制教师先验过渡，所有真实 V 保持正常监督。

`old_source_cleanup.npz` 保存每个旧行的保留标记、重叠清退原因、精确原生行号及应冻结保留的场景上下文行号。仅清洁旧场 8,925 个原生行号列为重复源拷贝候选；混合其它车辆与清退外部点保留在原始冻结上下文中。道路、相机、轨迹和其它车辆保持冻结。

`manifest.json` 给出父源 SHA256、字段、置信度值和完整策略；`cpu_exact_source_verification.json` 逐项核验所有 TRELLIS 参数及清洁旧场参数与父源字节完全一致。生成命令：`python build_joint_initialization.py`。本步骤仅使用 CPU。
