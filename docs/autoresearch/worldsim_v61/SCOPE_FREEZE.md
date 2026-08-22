# WorldSim V6.1 P0 范围冻结

- Task：`WS-V61-P0-SCOPE-FREEZE-01`
- Hypothesis：`WS-V61-H-P0-001`
- 状态：`active_pre_registered`
- 开始时间：`2026-08-22T09:58:17Z`

## 唯一北极星

构建 Occupancy-authoritative、Gaussian-rendered、task-verifiable 的四维驾驶世界编译器。

第一阶段只回答：在同一 28-case development denominator 上，Occupancy-controlled 3D proposal 能否以
`0 false-safe` 把安全通过数从 R10 的 `3/28` 提升到至少 `5/28`。

## 冻结边界

- V6 selector 研究族保持关闭，不继续 threshold、新 actor 或新 selector。
- `O_method` 使用 target frame 的偶数偏移 raw LiDAR sweeps；`O_eval` 使用不重叠的奇数偏移 sweeps。
- native boxes 只作为 observed actor geometry/lifecycle 来源；预测或生成 Occupancy 不得写成观测 GT。
- 高斯只拥有 appearance；collision state 由 Occupancy/SDF/mesh 独立拥有。
- confirmation 保持锁定；先完成 SceneIR-O、ME-0 与 ME-1 oracle upper bound。

## 防重复引用

`V6-F25`、`V6-F26`、`V6-F65`、`V6-F71`、`V6-F78`、`V6-F79`。

## 停止规则

- ME-1 oracle `<5/28`：停止 Hunyuan3D、GaussianWorld、OccWorld 接入，修 compiler/evaluator。
- ME-2 voxel-controlled actor `<2/6`：停止 prompt/seed/texture 调参。
- predicted Occupancy 出现 false-safe：保持 UNKNOWN/abstain，不降阈值。
