# WorldSim V5 M3 Constraint-Projected Temporal Development

- Task：`WS-V5-M3-CONSTRAINT-PROJECTED-TEMPORAL-01`
- 日期：`2026-08-14`
- 当前状态：`running`
- 当前阶段：`trajectory_mechanism_insufficient_no_renderer_unlock`
- 数据范围：fresh development 8 scenes；validation/test/KITTI quality 均未读取

## 1. 当前结论

M3 的 result-blind 协议、8-scene clip inventory 和 T2–T5 轨迹投影实现已经冻结，但当前 LATERAL/INSERT 请求没有足够的 T2 物理违例信号，不能支持 constraint projection，也没有解锁 renderer、collision gate、method selection 或 validation。

测量修正后的 r005 是当前权威机制结果：`15/16` T2 请求本来就是 safe，剩余 `1/16` 只有 `2` 项违例；T5 将其降为 `1` 项，总体 reduction=`50%`，但低于预注册的最小 `8` 个 T2-violation evaluable requests。正式结论为 `m3_constraint_projection_insufficient_t2_violation_signal`。

## 2. 协议边界

- M2 已 rejected，所以 REMOVE 只保留 exact bypass、语义重引入与 rollback 检查，不进入 M3 trajectory physics denominator。
- trajectory primary operations 为 `LATERAL/INSERT`，单位=`one_scene_one_operation_one_clip`；8 scenes × 2 operations=`16 requests`。
- T2 必须在 fresh development 同输入上重跑 V4 frozen SE(3) B-spline。V4 canonical r238/r335 的 baseline 是 `FRAME_INDEPENDENT`，其统计不能当作 V5 T2 comparator statistics。
- T3=minimum-jerk；T4=T3+road contact；T5=T4+vehicle kinematics。optional local residual 因 M1 rejected 保持关闭。
- r001 只解锁 implementation；r003 只解锁 trajectory mechanism metrics；直到机制门通过前，不允许读取 render quality。

## 3. 运行时间线

| Run | 状态 | 结果 |
|---|---|---|
| r001 `20260814T223000Z__m3-protocol-audit-s0-r001` | done | 冻结 T2 comparator、T3–T5、物理 caps、REMOVE 隔离与 quality locks；implementation only unlocked |
| r002 `20260814T224500Z__m3-development-clip-inventory-s0-r002` | blocked | YAML 缺 `protocol_audit.conclusion`，在 annotation streaming 前 `KeyError`；无数据/质量/GPU 读数，补 terminal 后保留 |
| r003 `20260814T225000Z__m3-development-clip-inventory-s0-r003` | done | 8/8 fresh development scenes 各冻结 1 个七 keyframe vehicle clip；0 abstain；只读 annotation metadata |
| r004 `20260814T230000Z__m3-constraint-mechanism-s0-r004` | done | T2 violations=`38`，T5=`34`，但全部 T2 违例均为 heading-velocity；T5 产生 `20` yaw-rate + `14` heading，门为 insufficient |
| r005 `20260814T231500Z__m3-constraint-mechanism-measurement-v2-s0-r005` | done | 低速 heading unobservable、允许 reverse、convergence=zero violations 后 exact replay；T2/T5=`2/1`，evaluable=`1/16`，仍 insufficient |

## 4. r004→r005 测量修正

r004 的 `38` 个 T2 违例全部来自 heading-velocity mismatch；T5 的逐帧 heading correction 与 yaw-rate projection形成固定循环，虽然更新量收敛，却残留 `20` 个 yaw-rate 违例。这里有三项测量/实现错误，而不是阈值可调机会：

1. 低速位置抖动不能定义 velocity heading，heading metric 改为仅在 speed>`1 m/s` 时可观测；
2. 倒车运动是合法运动，mismatch 取 forward/reverse heading 的较小值；
3. POCS 的“更新量固定点”不等于 feasible-set membership，只有剩余物理违例为 0 才允许 `converged=true`。

该修正是看过 r004 后的 result-aware development correction，r005 不具 confirmatory 身份，也不能单独解锁 renderer/validation。修正后 T2 违例从 `38→2`，说明原机会主要是 measurement artifact。

## 5. Fresh clip inventory

8/8 selected clips 全为 `vehicle.car`，每个 `7` keyframes / `3.0s`；只使用 category、visibility、annotation LiDAR count、instance token 与 sample order。eligible windows 范围=`1–224`，selected minimum LiDAR points 范围=`2–2351`；没有读取图片、LiDAR blob、reconstruction/edit quality。

## 6. 复开条件

当前禁止为了凑足 `8` 个 evaluable 而降低 heading speed floor、禁止取消 reverse 语义、禁止降低物理 caps 或改 r005 分母。下一步若继续，只能注册与计划一致的新 desired-motion hypothesis（例如明确的加速、制动、lane-change stress request），使用新 task/config/run，并明确标为 development hypothesis；或者停止 physics branch，不得用 V4 `FRAME_INDEPENDENT` 统计替代 T2 证据。

机器元数据：[`archive/2026-08/worldsim-v5-m3/M3_R001_R005_METADATA.json`](archive/2026-08/worldsim-v5-m3/M3_R001_R005_METADATA.json)。
