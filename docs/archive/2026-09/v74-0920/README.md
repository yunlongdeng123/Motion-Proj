# V7.4 报告与过程归档

承接用户 2026-09-20 的 `v74-0920` 归档目录。这里的计划、阶段状态和电源指令属于成文时历史，不产生新执行授权；当前状态只见 [RESEARCH_STATUS](../../../RESEARCH_STATUS.md)。

- [旧版本 failure](../../../research_failures/VERSIONS.md) · [全量 ID](../../../research_failures/IDS.md) · [全部 V7.4 卡](../../../research_failures/ENTRIES.md)。
- [原状态历史](STATUS_HISTORY.md) · [原实验历史](EXPERIMENT_HISTORY.md)。
- [整理记录](CLEANUP.md) · [原路径→归档路径](PATH_MAP.json) · [历史正文完整性核对](FAILURE_AUDIT.json)。
- [大资产恢复](../worldsim_v74_closeout_20260913/README.md)：此前已经外移的资产，未在本次重新删除。

## 下游仿真与感知

- [2026-09-20 进展与资源收口](WORLDSIM_SIMULATION_CLOSEOUT_20260920.md)
- [以仿真退化为起点的发现协议](WORLDSIM_SIMULATION_DOWNSTREAM_FIRST.md)
- [新来源真实驾驶基线：一次路线控制后仍未准入](WORLDSIM_SIMULATION_FRESH_BASELINE.md)
- [完整仿真与局部恢复：感知退化已确认，几何主因尚未成立](WORLDSIM_SIMULATION_FULL_BUDGET_PERCEPTION.md)
- [重建几何是否实质影响端到端仿真：第一轮实际结果](WORLDSIM_SIMULATION_IMPACT.md)
- [自然近车交互补证：发现路锥接触，但主要差距由观测范围解释](WORLDSIM_SIMULATION_INTERACTION_FINDINGS.md)
- [重建几何是否实质影响仿真：LiDAR 策略与官方车辆模型实测](WORLDSIM_SIMULATION_LIDAR_IMPACT.md)
- [局部网格→仿真 LiDAR→车辆定位：有限因果证据与删除对照](WORLDSIM_SIMULATION_LOCAL_ASSET_CAUSAL_AUDIT.md)
- [原生 RGB＋LiDAR 闭环试跑：差距主要经 RGB 通道传递，几何因果仍未确认](WORLDSIM_SIMULATION_NATIVE_CLOSED_LOOP_PILOT.md)
- [原生 RGB＋LiDAR 仿真推进记录](WORLDSIM_SIMULATION_NATIVE_SENSOR_STAGE.md)
- [Pi3X 残余的有限因果审计](WORLDSIM_SIMULATION_PI3X_CAUSAL_AUDIT.md)
- [代码核查更新：新来源基线未通过，准备原生nuScenes规划器（2026-09-15）](WORLDSIM_SIMULATION_PLANNER_INTERFACE_REVIEW.md)
- [六日志扩大复核：框精度退化可见，前车严重丢失未重复](WORLDSIM_SIMULATION_SIX_LOG_PERCEPTION.md)

## H1、P0 与版本收尾

- [V74 P0 完成与有卡开机交接](WORLDSIM_V74_P0_HANDOFF.md)
- [V7.4 收尾：没有形成经过验证的主方法](WORLDSIM_V7_4_CLOSEOUT.md)
- [V74 失败过程与反例](WORLDSIM_V7_4_FAILURES.md)
- [V74 方法实现记录](WORLDSIM_V7_4_METHODS.md)
- [V74 方法竞争最终报告](WORLDSIM_V7_4_RESULTS.md)
- [Motion-Proj / WorldSim V7.4：方法创新并行淘汰研究计划](WorldSim_V74_Method_Tournament_Plan.md)

## H2 学习器与诊断

- [V74 下半场：无卡阶段交接](WORLDSIM_V7_4_H2_CPU_HANDOFF.md)
- [V74-H2 计划1.1：GPU P1执行报告](WORLDSIM_V7_4_H2_GPU_P1_REPORT.md)
- [GPU P1执行补充：计划1.1 / 2026-09-12](WORLDSIM_V7_4_H2_METHOD_CONTRACT.md)
- [V74-H2 P1.5：Switching-margin 失败审计定义](WORLDSIM_V7_4_H2_P15_FORENSICS_PLAN.md)
- [V74-H2 P1.5：薄结构失败来自连续深度漂移](WORLDSIM_V7_4_H2_P15_FORENSICS_REPORT.md)
- [V74-H2 P1.6：法向误差、累积漂移与普通物理度量控制](WORLDSIM_V7_4_H2_P16_DIAGNOSTIC_PLAN.md)
- [V74-H2 P1.6：普通度量解释旧漂移，有限支撑问题仍开放](WORLDSIM_V7_4_H2_P16_DIAGNOSTIC_REPORT.md)
- [计划1.1执行后论文状态：主方法证据尚未建立](WORLDSIM_V7_4_H2_PAPER_DRAFT.md)
- [Motion-Proj / WorldSim V7.4 下半场完整研究计划](WorldSim_V74_Second_Half_Generative_Surface_Plan.md)

## 官方模型取证与主图

- [V7.4-H2：首次回波 badcase 重新取证](WORLDSIM_V7_4_H2_FIRST_RETURN_REDISCOVERY.md)
- [V7.4-H2：GPU 接续与可证伪比较](WORLDSIM_V7_4_H2_REDISCOVERY_GPU_HANDOFF.md)
- [V7.4：从重建表面到物理首回波的证据与主图](WORLDSIM_V7_4_MAIN_FIGURES.md)
- [Figure captions](WORLDSIM_V7_4_MAIN_FIGURE_CAPTIONS.md)
