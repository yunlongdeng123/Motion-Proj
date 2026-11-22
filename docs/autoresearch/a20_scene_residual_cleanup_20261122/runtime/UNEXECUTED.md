# 未执行的连续透明度方案

用户于 2026-11-22 要求停止追加实验。以下代码仅为 CPU 准备稿，未启动 GPU，不构成实验结果，也不进入当前可用恢复入口：

- `optimize_context_visibility.py`：120 步局部透明度优化草稿。
- `cold_reload_continuous.py`：对应连续透明度资产的冷重载草稿。
- `query_rgb_residual.lowalpha_cpu_only.py`：未运行的低覆盖域诊断扩展。

当前实际执行的 RGB 来源查询是 `query_rgb_residual.py`，SHA256 为 `2c0afb36c8a2bb5dc9292ecac25ce10bdc07d063b4d33058d435fb26c64be1f3`。当前所有已跑清理阶段的优化器更新数均为 0。
