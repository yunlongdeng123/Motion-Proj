# 实验索引

本页按实验定位报告和原始证据；当前执行状态只见 [RESEARCH_STATUS](RESEARCH_STATUS.md)。完整历史过程保留在 [V7.4 实验历史](archive/2026-09/v74-0920/EXPERIMENT_HISTORY.md)及[V7.3 完整台账](archive/2026-09/pre-v74/V73_EXPERIMENTS.md)。

| task / run 或阶段 | 记录内容 | 入口 |
|---|---|---|
| WS-V75-QUALIFY-01 / 20260920-r1 | 官方接口契约与资源来源核对 | [问题协议](v75/PROBLEM.md)、[V75-F01](research_failures/entries/V75-F01.md) |
| WS-V75-PREFLIGHT-01 / 20260920-r1 | 环境、真实初帧与条件、输入编码和权重预检 | [运行准备](v75/PREFLIGHT.md)、[证据](autoresearch/worldsim_v75/preflight) |
| WS-V75-BASELINE-01 / 20260920-single3090-r1 | 单卡首段、完整clean及同seed完整重复；资源与视频核验 | [单卡基线](v75/BASELINE.md)、[证据](autoresearch/worldsim_v75/baseline) |
| WS-V75-LOCALIZE-01 / 20260920-r1 | 单目标±0.5m条件干预与恢复；两seed有限实验 | [定位实验](v75/LOCALIZATION.md) |
| WS-V75-COHORT-01；NATIVE-COHORT-01 / 20260920-r1 | 来源目录与访问核验、3开发+3保留冻结、三例原生clean | [原生样例基线](v75/NATIVE_COHORT.md) |
| WS-V75-AV2-BRIDGE-01 / 20260920-r1 | Argoverse显式三维条件适配、真实未来RGB与单卡clean | [输入桥接](v75/AV2_BRIDGE.md) |
| WS-V74-P0-* / 20260910 | 数据、资源与输入角色准备 | [P0](archive/2026-09/v74-0920/WORLDSIM_V74_P0_HANDOFF.md) |
| WS-V74-METHOD-TOURNAMENT-01 | H1 WEX / RIF / DCS 与强控制 | [结果](archive/2026-09/v74-0920/WORLDSIM_V7_4_RESULTS.md)、[失败](archive/2026-09/v74-0920/WORLDSIM_V7_4_FAILURES.md) |
| H2 CPU / GPU P1 | 数据与学习能力实验、实现关闭 | [CPU](archive/2026-09/v74-0920/WORLDSIM_V7_4_H2_CPU_HANDOFF.md)、[GPU](archive/2026-09/v74-0920/WORLDSIM_V7_4_H2_GPU_P1_REPORT.md) |
| WS-V74-H2-P15-01 / switching-margin-r1 | 失效轨迹审计 | [P1.5](archive/2026-09/v74-0920/WORLDSIM_V7_4_H2_P15_FORENSICS_REPORT.md) |
| WS-V74-H2-P16-01 / 20260913__physical-metric-r2 | 普通几何控制 | [P1.6](archive/2026-09/v74-0920/WORLDSIM_V7_4_H2_P16_DIAGNOSTIC_REPORT.md) |
| WS-V74-H2-REDISCOVERY-01 / 20260915-cpu-r1 | 旧案例 CPU 回放与来源核对 | [重新取证](archive/2026-09/v74-0920/WORLDSIM_V7_4_H2_FIRST_RETURN_REDISCOVERY.md) |
| WS-V74-MAINFIG-01 / 20260915-first-return-r1；SECONDARY-01 / 20260915-secondary-r1 | 4主模型48前向、第二批24前向及表面参照 | [官方模型主图](archive/2026-09/v74-0920/WORLDSIM_V7_4_MAIN_FIGURES.md) |
| WS-SIM-IMPACT-01 / 20260915-r1 | HUGSIM-LTF 与碰撞表示 | [报告](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_IMPACT.md) |
| WS-SIM-LIDAR-01；INTERACTION-01 / 20260915-r1 | LiDAR、策略/PDM 与六日志交互 | [LiDAR](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_LIDAR_IMPACT.md)、[交互](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_INTERACTION_FINDINGS.md) |
| Pi3X 局部资产审计 | 路面修复未消除残余 | [因果审计](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_PI3X_CAUSAL_AUDIT.md) |
| WS-SIM-NATIVE-CLOSEDLOOP-01 / scene0004-step030000-full1 | SplatAD完整预算、反馈与感知控制 | [完整预算](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_FULL_BUDGET_PERCEPTION.md) |
| WS-SIM-FF-LOCAL-ASSET-01 / r2 | Ω 与 Pi3X 网格编辑、重新投射与检测 | [局部资产](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_LOCAL_ASSET_CAUSAL_AUDIT.md) |
| WS-SIM-FF-COHORT-PERCEPTION-01 / 20260915-r1 | 82新增+20复用检测；全部分母 | [六日志](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_SIX_LOG_PERCEPTION.md) |
| WS-SIM-FRESH-NATIVE-01 / 20260915-r1 | 16组真实驾驶基线/路线控制 | [新来源](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_FRESH_BASELINE.md) |
| WS-SIM-SPARSE-NATIVE-CLOSEOUT-01 / 20260920-r1 | 导入缺 terminaltables；0/40前向；依赖任务跳过 | [终态与日志](autoresearch/worldsim_simimpact/closeout_20260920/README.md) |
| WS-V74-DOCS-CLEANUP-01 / 20260920 | 文档职责、旧 failure 检索与归档链接修复；无新实验 | [整理记录](archive/2026-09/v74-0920/CLEANUP.md) |

每个 run 的真实完成数量、配置、seed、数据角色、成本和失败边界以该报告及对应 manifest 为准。不同类型执行次数不合并为独立样本。更早版本的实验从[历史归档](archive/README.md)进入。
