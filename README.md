# 当前：V7.4-H2 首次回波重新取证完成（2026-09-15）

已回到 `research/worldsim-v7.4-h2-generative-surface`，V8.1 分支保留。task `WS-V74-H2-REDISCOVERY-01 / 20260915-cpu-r1`，状态 **CPU_DONE_WAIT_GPU**。恢复7个旧案例、8364束双精度回放、3对象×7系统配对、3个RGB上下文和15张图；0新模型推理、0训练。白色轿车两个留出时刻Early59/557、18/195，0.5m仍32/752；旧VGGT-native实际为微调DPT＋LiDAR融合，不能代表官方VGGT。旧20日志r7−native融合15/20同增Hit/Early，12/20四项同增；强r6对照完整保留。

官方VGGT/Ω、Pi3X、MapAnything、DVGT、DGGT的共同失效仍待原生输出和尺度/读出控制；Early不等于false-safe。只准备有限GPU验证，不恢复已关闭H2训练或旧调度。旧20日志已曝光，10日志reserve质量未读；人工verdict=null。failure_ledger_refs=V73-F02/F03/F04/F09、V74-F06、V74-H2-F11；delta=V74-H2-F12。当前不继承历史shutdown授权，未关机、无自动续跑。

[本轮报告与组件图](docs/WORLDSIM_V7_4_H2_FIRST_RETURN_REDISCOVERY.md)；[GPU接续](docs/WORLDSIM_V7_4_H2_REDISCOVERY_GPU_HANDOFF.md)；[F12](docs/research_failures/entries/V74-H2-F12.md)。以下为历史。

# 当前：V7.4-H2 存量 badcase 重新取证（2026-09-15）

用户已授权回到 `research/worldsim-v7.4-h2-generative-surface`，当前无卡。task `WS-V74-H2-REDISCOVERY-01` / `20260915-cpu-r1`：追查覆盖与首次回波冲突的旧真实案例，审计 VGGT 与后继方法的实测边界。只执行存量 CPU 分析、可视化与后续实验准备，不自动恢复已关闭 H2 学习器或旧调度，不继承历史关机授权。V8.1 独立分支保留。旧20日志确认现为已曝光资料，10日志新 reserve 继续封存。人工 verdict=null。以下是历史。

# WorldSim V7.4：已收尾

最终精简：GitHub 实际下载 ZIP 约 **22.2 MB**；同口径本地打包105.25→21.45 MB，减少约80%，跟踪文件展开约53.6 MB（原375.9 MB）；两批共307个资产、原文件322.49 MB已归档。全部代码/配置/Markdown、失败查询索引和V7.4报告图保留。旧下载包不会自动变小，需重新下载当前分支；完整 Git 历史保持，未强推改写。

当前状态：**V7.4 已结束，未形成经过验证的论文主方法。** H1 NO_SURVIVOR；H2 当前 ordered 实现关闭，整体 witness-native 假说未被全面证伪。后续遵循 failure discovery 优先原则；本轮不自动续跑。

- [版本收尾与研究教训](docs/WORLDSIM_V7_4_CLOSEOUT.md)
- [P1.6 病因诊断、普通控制与架构](docs/WORLDSIM_V7_4_H2_P16_DIAGNOSTIC_REPORT.md)
- [P1.5 失败审计、架构与证据](docs/WORLDSIM_V7_4_H2_P15_FORENSICS_REPORT.md)
- [GPU执行报告与架构](docs/WORLDSIM_V7_4_H2_GPU_P1_REPORT.md)
- [当前研究状态](docs/RESEARCH_STATUS.md)
- [失败资产渐进读取](docs/RESEARCH_FAILURES.md)
- [原修订计划](docs/WorldSim_V74_Second_Half_Generative_Surface_Plan.md)
- [CPU数据与文档交接](docs/WORLDSIM_V7_4_H2_CPU_HANDOFF.md)

H1 NO_SURVIVOR与V73均为冻结历史。

[大资产归档与轻量获取方法](docs/archives/worldsim_v74_closeout_20260913/README.md)：当前分支保留代码、核心文档及必要证据；旧批量资产按需恢复。
