# M43 算子命名勘误

日期：2026-09-06

任务：`WS-V72-P0-OPERATOR-ERRATUM-01`

基线提交：`79910be13e1f4740dbcd793fbfd45ee6d66020f8`

失败账本引用：`V71-F52`、`V71-F43`

## 结论

M43 canonical run 中 `m8_point_surface` 的 early/hit 描述性统计混用了 categorical baseline 与 literal surface output，因此不能解释为匹配算子的几何迁移结果。canonical run 及其原始 `summary.json` 保持不可变。

M39 与单位权重 categorical baseline 使用同一算子，其跨传感器判定不受影响：all/hazard/clear early delta 为 `+0.229/+0.542/-0.036pp`，hit delta 为 `+5.888/+6.302/+5.537pp`，冻结判定仍为 `m39_development_only_cross_sensor_rejected`。

## 根因与代码修正

`scripts/run_worldsim_v71_m43_m39_av2_zero_shot.py::_evaluate_bundle` 先由 `evaluate_actor_surface` 写入 literal baseline/output 计数，随后曾用 categorical baseline 覆盖通用的 `baseline_early_count` 与 `baseline_hit_count`。修正后：

- literal 几何计数继续使用 `baseline_*` 与 `output_*`，只交给 `summarize_surface_rows`；
- categorical 回波计数使用 `categorical_baseline_*` 与 `m39_*`，只交给 `_surface_return_summary`；
- Actor retention 由输入与输出的 ID、轨迹和尺寸逐字段比较，hazard retention 另含 hazard 标签比较；不再在该 runner 中仅依赖常数约定。

微型语义检查必须证明两个命名空间不会互相覆盖。

## 已有 artifact 的边界

已保存的 `EXTERNAL_ACTORS.jsonl` 只保留了覆盖后的 categorical baseline 计数，没有保留原 literal baseline 计数，也没有保存可直接重算的完整输出表面。因此无法只靠现有 JSONL 恢复匹配 literal baseline。

这项重算属于 `legacy_diagnostic`，需要重新加载冻结模型并编译已消费的 20 个 AV2 logs。当前无 GPU，不启动重算，也不改变 M39 判定。GPU 恢复后可在不训练、不调参、不读取新 cohort 的前提下生成独立勘误 artifact；它不构成 V7.2 新泛化证据。

## 允许与禁止的表述

允许：M43 的同算子 categorical 比较显示 hit 增加，同时 all/hazard early 恶化；描述性 Chamfer 配对仍为 `+0.162mm`。

禁止：引用 `16.907%→45.271%` 或 `16.556%→50.149%` 作为匹配算子的 M8 几何迁移失败，也禁止由此单独归因于传感器线束或几何—传感器耦合。
