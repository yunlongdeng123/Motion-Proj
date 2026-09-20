# SparseDrive 固定队列执行记录

task/run：`WS-SIM-SPARSE-NATIVE-CLOSEOUT-01 / 20260920-r1`。

固定队列在首阶段导入时报 `ModuleNotFoundError: No module named 'terminaltables'`，实际保存前向 **0/40**，其余依赖阶段跳过。环境错误不计作模型或科学失败。

- [终态快照](terminal_state.json)：阶段命令、时间及退出码。
- [导入错误日志](real_policy_tail.txt)。
- [已保存结果清单](evidence.json)：无模型输出。
- [当时的队列计划](../../../archive/2026-09/v74-0920/WORLDSIM_SIMULATION_CLOSEOUT_20260920.md)：是运行前计划，不是完成报告。

```mermaid
flowchart LR
    A[冻结六相机输入] --> B[导入 SparseDrive]
    B --> C[缺 terminaltables]
    C --> D[终止 / 保存日志]
    D --> E[0 前向 / 无评价]
```

运行随后归档并按当时授权请求关机；此目录不表达当前电源状态，也不授权重启。当前执行范围只见 [RESEARCH_STATUS](../../../RESEARCH_STATUS.md)。
