# 当前研究状态

更新：2026-10-12（新加坡）。分支 `research/worldsim-v8.2-drivingeditor-debug`；继承v8.1 `c21547193d765e8f8123b27b73e31e033295a316`。当前范围仅是保存已有DrivingEditor调试证据并交付GitHub代码，task/run `WS-V82-DRIVINGEDITOR-DEBUG-HANDOFF-20261012/r1`；没有新增训练、推理或实验。用户已要求停止研究，`drivingeditor-50k`自动化PAUSED，其他场景队列保持暂停；不从历史ACTIVE、diagnosing、50k计划或旧prompt恢复任务。

银色车Waymo FRONT raw0–48共49帧的B1/B2/V/S均已从同一saved30k完成2000步至32000并保存、配对验证。B1/B2全图PSNR恢复36.43076/36.43047dB，静态道路仍差、动态层仍含背景。V/S车体外static PSNR为27.33124/31.76342dB，但composite降至35.90406/35.44799dB；V actor MSE全部49帧更高，S对B1/B2分别46/47帧更高，动态背景和边缘车形残片仍在。两项是诊断干预，不是作者公式认证、从零训练或完整50k成功；单次B2重复不是通用噪声界。**整个debug未完成，完整paper复现及编辑效果未通过。** [流程、组件图与交接](v82/DRIVINGEDITOR_DEBUG_HANDOFF.md) · [轻量结果](v82/DRIVINGEDITOR_DEBUG_HANDOFF.json)。

已确认reset/保存顺序、稀疏深度重复像素顺序依赖、SGCM始终不可见重复clone及原目标分配歧义；它们不能合并为唯一彩噪根因。逆深度明确来自paper，harmonic融合/sky900只是未认证实现选择。固定pose时间控制确认static随时间变化；初始化有169行native目标点留在static的见证下界，但90.7967%身份未知，不能推全49帧污染率或彩噪因果。M-detector native core/adapter已在私有依赖下隔离编译成功、20个vendor文件未改，尚无native构造、扫描或标签；时间合同`author_equivalence=false`，scan-start与mid-pose等价性未验，shadow仍缺失。I/H/T是未执行设计。相关历史资产边界[V76-F03](research_failures/entries/V76-F03.md)保留，原receipt和本次整理`failure_ledger_delta=none`，人工verdict=null。

代码按仓库第三方checkout外置规则交付：`compatibility/DrivingEditor-v82/`存兼容与V/S patch，`scripts/worldsim_v82/drivingeditor_debug/`存排错工具；轻量证据位于 `docs/autoresearch/worldsim_v82/drivingeditor_debug_20261012/r1/evidence/{debug,provenance}`。完整源码快照在 `drivingeditor_v82_remote_code_evidence.tar.gz`；大型数据、checkpoint、FP32、全量媒体及原深色HTML仍保留本地/远端，以资产目录为恢复入口，轻量Git不冒称完整备份。旧3090是原2024代码21帧30k，32.7914dB为reset前历史输出，不能与新49帧指标合并。没有新GPU队列或关机操作；旧“整个排错完成后关机”条件尚未满足。

v8.1 DGGT四帧编辑、Seen-to-Scene暂停与历史结果继续保留在[v8.1报告](v81/DGGT_WAYMO_INFERENCE_R1.md)、[P1周期记录](v81/P1_REVIEW_CYCLES_R1.md)及[实验索引](EXPERIMENTS.md)，不因v8.2整理恢复或重写其研究结论。后续研究、实验与电源操作等待新的用户指令；本次仅完成证据和代码交付。
