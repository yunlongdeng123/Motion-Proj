# WorldSim V7.2 贡献—证据映射

| 计划贡献 | 必需证据 | 最终状态 |
|---|---|---|
| target-free 动态 LiDAR 查询契约 | 接口检查、独立 targets、角色门控 | 已实现；clean dev 501 Actors 正式运行 |
| 统一 surface-to-return 评价 | G0/G1/G2/G3/A1 共用密度与射线口径 | 已完成；实际点数与几何/回波分表报告 |
| 对象补全与 full neural LiDAR 路线判定 | 干净 dev D1 + 完整 LiDAR capability | 已完成；D1=`close_method_claim` |
| 学习式观测约束 surface 方法 | 超过 TSDF/AdaPoinTr 且副作用受控 | 未支持；V71-F63 关闭 |
| 投稿级完整应用与独立 source test | D1 通过后才解锁 | 未解锁；不读取 final roles |

最终交付是可证伪诊断与基线技术报告。数值来自正式 run summary 与机械 `D1_DEV_GATE.json`；路线 A/B 均未晋级。
