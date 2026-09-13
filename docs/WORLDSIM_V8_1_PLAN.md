# V8.1 执行入口

用户原计划：[WorldSim_V81_SparseView_LowTexture_Failure_Discovery_Plan.md](WorldSim_V81_SparseView_LowTexture_Failure_Discovery_Plan.md)。主方向是稀疏视角×低纹理的科学发现和 badcase/goodcase atlas；V8.2 才研究解法。V7.4 已关闭路线不重开。

当前 task：WS-V81-CPU-01；状态与资源交接见 [CPU handoff](WORLDSIM_V8_1_CPU_HANDOFF.md)。

按顺序读取：[方法审计](WORLDSIM_V8_1_METHOD_AUDIT.md) → [科学报告与架构](WORLDSIM_V8_1_SCIENTIFIC_REPORT.md) → [Failure Atlas](WORLDSIM_V8_1_FAILURE_ATLAS.md) → [V8.2入口](WORLDSIM_V8_1_TO_V8_2_DECISION.md)。

原始配置 `configs/worldsim_v81/cpu.json`；所有本轮自然候选均为DISCOVERY，合成干预独立标注。正式结果不允许只看好看的图片，必须同时显示简单控制、参考局限、goodcase和独立日志分母。
