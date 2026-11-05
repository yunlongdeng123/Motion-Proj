# 当前研究状态

更新：2026-11-05。

当前主线为Scene008车辆外观恢复。道路暂停，PVD及外观模型权重冻结；原M2保留，研究候选没有自动采用。

已完成冻结外观先验、原始1920分辨率观测拟合，以及对象坐标表面重采样的正式对照。最清楚的完整a6资产可独立渲染并统一插回场景，三套49帧图库和保留／删除a37成对检查已交付。参考图写实目标尚未达到，隐藏结构无独立真值；局部外域成像与深度仍有代价。完整指标、预算和来源见[本轮报告](autoresearch/scene008_a6_appearance_formal_20261105/README.md)。

下一步先做真实观测约束下的表面几何拟合与多视图一致性，不因灰色材质回退直接启动PVD后训练。当前没有已启动的后续训练控制器。

执行边界：subagent仅gpt-6.1-sol/max；AutoDL按本会话明确要求保持开机，不自动关机。failure_ledger_refs=[]，failure_ledger_delta=none；人工画质verdict未代填。

RCR-1支线已收尾归档：补回RGB约束压住局部蓝色，但完整候选新增亮片未达floor，不采用并保留当前M2；没有自动后续GPU队列。HELD9只对道路RGB校准target留出，不是独立native泛化验证。见[交互验收、实际源码与研究归档](autoresearch/scene008_rcr1_20261105/index.html)。本支线failure_ledger_refs=[V82-F13]，failure_ledger_delta=V82-F13；不改本页车辆外观主线安排。

历史：此前状态全文保存在[更新前状态归档](autoresearch/scene008_a6_appearance_formal_20261105/prior_research_status.md)，逐实验入口继续保留在[实验索引](EXPERIMENTS.md)。
