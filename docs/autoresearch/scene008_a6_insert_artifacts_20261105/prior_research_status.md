# 当前研究状态

更新：2026-11-05。

当前主线为Scene008车辆外观恢复。道路暂停，PVD及外观模型权重冻结；原M2保留，研究候选没有自动采用。

已完成冻结外观先验、原始1920分辨率观测拟合，以及对象坐标表面重采样的正式对照。最清楚的完整a6资产可独立渲染并统一插回场景，三套49帧图库和保留／删除a37成对检查已交付。参考图写实目标尚未达到，隐藏结构无独立真值；局部外域成像与深度仍有代价。完整指标、预算和来源见[本轮报告](autoresearch/scene008_a6_appearance_formal_20261105/README.md)。

下一步先做真实观测约束下的表面几何拟合与多视图一致性，不因灰色材质回退直接启动PVD后训练。当前没有已启动的后续训练控制器。

执行边界：subagent仅gpt-6.1-sol/max；AutoDL按本会话明确要求保持开机，不自动关机。failure_ledger_refs=[]，failure_ledger_delta=none；人工画质verdict未代填。

RCR-1支线当前由用户采用：2026-11-05明确选择RCR-1 delete用于异常蓝色RGB修复（scope=abnormal_blue_RGB_fix）；固定完整M2资产不变，原管线仅作reference。raw130/137亮片继续开放，完整画质无回退未作新结论，没有自动后续GPU队列。HELD9只对道路RGB校准target留出，不是独立native泛化验证。见[当前交互报告](autoresearch/scene008_rcr1_20261105/index.html)及[用户采用决定](autoresearch/scene008_rcr1_20261105/evidence/user_acceptance_20261105.json)。本支线failure_ledger_refs=[V82-F13]为历史评审，本次failure_ledger_delta=none；车辆外观主线安排不变。

a5补充只读诊断已归档，当前优先依据[两份实际报告直接复核](autoresearch/scene008_rcr1_20261105/followup_a5/index.html#report-crosscheck)：旧Full5 B30K保留a5独立动态身份，仅mask动态索引3/4（a6/a37）时a5仍完整；新资产报告的ownership6+37 Native10K在M0/M2前已缺损。配方提交3fefd98仅为首次归档，不是唯一causal commit；具体引入提交与单项机制仍未确定。这次追加只读图像/源码复核0GPU、0SGD，不是同预算单变量实验。既存原生重放为4帧×（旧2＋当前7）＝36次渲染，单列原有开销；RCR-1账本仍为26,609更新，采用范围保持异常蓝色RGB修复。failure_ledger_delta=none；车辆外观主线安排不变。

历史：此前状态全文保存在[更新前状态归档](autoresearch/scene008_a6_appearance_formal_20261105/prior_research_status.md)，逐实验入口继续保留在[实验索引](EXPERIMENTS.md)。
