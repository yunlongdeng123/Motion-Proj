# 当前：V8.1 四环证据有限补证（2026-09-13）

用户要求其他方法纳入 badcase report、现有结果继续收口。task WS-V81-CLOSE-01 / 20260913-bounded-r1；四环准入为可靠残余、预指定真实证据干预与恢复、普通控制及可恢复空间、新日志与对应重建任务。完整自然四格不再作为所有推进的先决条件。不重跑440项，不训练，不开启V8.2，不继承V74关机授权。

10个既有ROI统一内部复核；8个预先冻结AV2新发现日志完成840格筛查，16候选先看RGB/参考。模型前固定3日志的有效后一帧干预，rich→removed→restored，共18项；DVGT/GPU0与VGGT/GPU1独立执行。双侧设计因前帧出画/遮挡改为单个实际有效后一帧，修改发生在所有新输出之前并留存原设计。旧10日志确认reserve继续封存。人工verdict=null。

[方法族badcase report](WORLDSIM_V8_1_METHOD_BADCASE_REPORT.md)；外部证据`/root/autodl-tmp/runs/worldsim_v81/WS-V81-CLOSE-01`。failure_ledger_refs=V81-F01/F02/F03/F04；failure_ledger_delta=待本轮收口。以下为历史。

# V8.1执行入口

原计划：[稀疏视角×低纹理失效发现](WorldSim_V81_SparseView_LowTexture_Failure_Discovery_Plan.md)。当前P2主模型阶段已完成，未进入V8.2。

读取[科学报告](WORLDSIM_V8_1_SCIENTIFIC_REPORT.md) → [图谱](WORLDSIM_V8_1_FAILURE_ATLAS.md) → [V8.2决策](WORLDSIM_V8_1_TO_V8_2_DECISION.md) → [GPU交接](WORLDSIM_V8_1_GPU_HANDOFF.md)。CPU r2和旧交接为历史，无需重复无卡准备。
