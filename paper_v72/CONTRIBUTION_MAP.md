# WorldSim V7.2 贡献—证据映射

> 2026-09-07 更新：此表仅描述旧补全路线报告；新主线为 [EAS-VGGT](../docs/WORLDSIM_V7_2_EAS_VGGT_RECOVERY_PLAN.md)，新方法贡献尚待 E1–E5。A1 负结果保留，缺 B 比较的整体收口解释由 `V71-F64` 更正，路线漂移由 `V71-F65` 更正。

| 计划贡献 | 必需证据 | 最终状态 |
|---|---|---|
| target-free 动态 LiDAR 查询契约 | 接口检查、独立 targets、角色门控 | 已实现；clean dev 501 Actors 正式运行 |
| 统一 surface-to-return 评价 | G0/G1/G2/G3/A1 共用密度与射线口径 | 已完成；实际点数与几何/回波分表报告 |
| 旧对象补全与 full neural LiDAR 路线判定 | A/B 同协议候选比较 | A1 负结果成立；B 比较未完成，旧 D1 不足以整体判定；当前选路协议已退役 |
| 学习式观测约束 surface 方法 | 超过 TSDF/AdaPoinTr 且副作用受控 | 未支持；V71-F63 关闭 |
| 投稿级完整应用与独立 source test | D1 通过后才解锁 | 未解锁；不读取 final roles |

本目录的历史交付为诊断与基线技术报告。数值来自正式 run summary；机械 `D1_DEV_GATE.json` 的 B=false 只是当时实现输出，不能当作科学拒绝。当前 EAS-VGGT 计划与后续实验另行记录。
