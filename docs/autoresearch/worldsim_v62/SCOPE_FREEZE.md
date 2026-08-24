# WorldSim V6.2 P0 范围冻结

- Task：`WS-V62-P0-SCOPE-FREEZE-01`
- Hypothesis：`WS-V62-H-P0-001`
- 状态：`done_scope_frozen`
- 分支：`research/worldsim-v6.2-cpsc`
- 起点：`main@c8e9dee`

## 冻结问题

V6.2 只回答：能否把真实 FREE/OCC 作为方法前向中的硬约束，把冻结的 IR-WM Occupancy 作为可推翻软先验，并通过
FREE/OCC/UNKNOWN 三态选择性补全，在 legacy28 上恢复至少 `5/28` 安全 ACCEPT、保持 `0 false-safe`，之后再在 fresh
scene-disjoint 数据上完成 calibration、one-shot confirmation 与 exact-once test。

## 不可变起点

- V6.1 终态保持 `v61_minimum_experiment_closed_negative`，不重跑或后修补。
- oracle：`10/28 ACCEPT, 0 false-safe`。
- GaussianWorld：`10/28 ACCEPT, 10/10 false-safe`。
- IR-WM：`10/28 ACCEPT, 10/10 false-safe`。
- 不搜索第三 Occupancy backend，不做 confidence/grid/history/checkpoint/verifier sweep。

## 方法与评测边界

- `E_input` 可见真实传感器硬证据；`E_eval` 只用于独立评测，永不进入 method、candidate selection 或阈值拟合。
- observed FREE/OCC 在 forward 后必须成立；冲突证据输出 UNKNOWN。
- IR-WM 全程冻结，只训练轻量 adapter/query/evidential/projection residual。
- all-UNKNOWN 由 safe-OCC retention 和 UNKNOWN 上限拒绝。
- legacy28 只作机制基准；正式主张由 fresh calibration、one-shot confirmation 和 exact-once test 决定。

## 用户约束覆盖

V6.2 新产物不增加哈希、校验和或指纹，也不建立重型完整性检查、重复 smoke 或大范围回归门。产物身份使用逻辑路径、
语义版本、task/run ID 与 Git 提交记录；每个实质机制变化只运行一个能够证伪当前风险的窄验证。

## 资源快照

- RTX 3090 24GB 空闲；无研究 GPU 进程。
- `/root/autodl-tmp` 剩余约 65GB。
- CPSC training 目标峰值不超过 18GiB；22GiB 为硬停止线；增量磁盘不超过 20GiB。

## 下一步

执行 `WS-V62-P1-NOVELTY-AUDIT-01`。在确认无直接重合前不编码；P1 通过后按 operator-first 原则先实现 P3 的最小硬投影
核心，再让 P2 evidence-query dataset 围绕已冻结的 operator contract 产出训练数据。
