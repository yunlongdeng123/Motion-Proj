# WorldSim V7.2 贡献—证据映射

| 计划贡献 | 必需证据 | 当前状态 |
|---|---|---|
| target-free 动态 LiDAR 查询契约 | 四项语义检查与接口测试 | P0 已实现，待最终回归 |
| 统一 surface-to-return 算子 | G0--G3 共用查询、深度与状态口径 | P0 已实现接口，D0 待跑 |
| object surface 与 full neural LiDAR 路线判定 | 几何、条件回波、全回波、代价和失败态证据 | D0 待跑 |
| 学习式 evidential surface 方法 | 跨外部数据集的冻结测试与消融 | 路线未锁定，不作声明 |

所有数值宏只能由包含 resolved config、manifest、fingerprint、JSONL 和 summary 的正式运行更新。
