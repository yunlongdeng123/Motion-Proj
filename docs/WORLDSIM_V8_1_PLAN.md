# 当前：V8.1 有限补证收口（2026-09-13）

WS-V81-CLOSE-01完成：10旧候选内部复核；8新AV2日志840格模型前筛查，3日志18项新推理。旧440项未重跑、无训练。木板墙最强法向候选参考不稳；灰墙297点/0.103m好例保留。VGGT两个新日志有0.429/0.290m可恢复差距，单图度量锚点后MAE0.018/0.472/0.337m；DVGT两个后向单相机坐标/覆盖合同未建立，不计科学failure或稳健负例。

四环证据不足，V8.2 NO_GO；当前共同失效子命题降低优先级，停止本批次。其他8方法已纳入方法族badcase report，DGGT等方法级实证仍缺，条件下游扩展未触发；不能宣称整个V8.1已测完。无自动续跑、确认reserve封存、未关机。人工verdict=null；failure_ledger_refs=V81-F01/F02/F03/F04；failure_ledger_delta=V81-F05。

[有限收口报告](WORLDSIM_V8_1_BOUNDED_CLOSEOUT_REPORT.md)；[方法族badcase report](WORLDSIM_V8_1_METHOD_BADCASE_REPORT.md)；[推进判定](WORLDSIM_V8_1_TO_V8_2_DECISION.md)。以下为历史。

# 当前：V8.1 四环证据有限补证（2026-09-13）

用户要求其他方法纳入 badcase report、现有结果继续收口。task WS-V81-CLOSE-01 / 20260913-bounded-r1；四环准入为可靠残余、预指定真实证据干预与恢复、普通控制及可恢复空间、新日志与对应重建任务。完整自然四格不再作为所有推进的先决条件。不重跑440项，不训练，不开启V8.2，不继承V74关机授权。

10个既有ROI统一内部复核；8个预先冻结AV2新发现日志完成840格筛查，16候选先看RGB/参考。模型前固定3日志的有效后一帧干预，rich→removed→restored，共18项；DVGT/GPU0与VGGT/GPU1独立执行。双侧设计因前帧出画/遮挡改为单个实际有效后一帧，修改发生在所有新输出之前并留存原设计。旧10日志确认reserve继续封存。人工verdict=null。

[方法族badcase report](WORLDSIM_V8_1_METHOD_BADCASE_REPORT.md)；外部证据`/root/autodl-tmp/runs/worldsim_v81/WS-V81-CLOSE-01`。failure_ledger_refs=V81-F01/F02/F03/F04；failure_ledger_delta=待本轮收口。以下为历史。

# V8.1执行入口

原计划：[稀疏视角×低纹理失效发现](WorldSim_V81_SparseView_LowTexture_Failure_Discovery_Plan.md)。当前P2主模型阶段已完成，未进入V8.2。

读取[科学报告](WORLDSIM_V8_1_SCIENTIFIC_REPORT.md) → [图谱](WORLDSIM_V8_1_FAILURE_ATLAS.md) → [V8.2决策](WORLDSIM_V8_1_TO_V8_2_DECISION.md) → [GPU交接](WORLDSIM_V8_1_GPU_HANDOFF.md)。CPU r2和旧交接为历史，无需重复无卡准备。
