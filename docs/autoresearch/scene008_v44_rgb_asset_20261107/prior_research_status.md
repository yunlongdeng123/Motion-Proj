# 当前研究状态

更新：2026-11-06。

当前主线为 Scene008 a6 的实际几何与外观恢复。PVD、预训练模型及背景参数冻结，原 M2 与旧 projective 资产保留，研究候选没有自动采用。

本轮已完成删 a37 后黑斑的实际颜色载体与几何位置诊断、观测锚定的 C2 连续深度后验、固定前景 matte RGBA 组合，以及相同 BUILD40 像素域的颜色拟合和固定49帧检查。新 chart 几何确实改变；原生480的 RGB 退化、HR/深度改善与外域局部变化须分别解释，不能把工程 PASS 当画质通过。详细证据见[本轮报告](autoresearch/scene008_a6_insert_artifacts_20261105/README.md)。

同一连续 surface / 固定 texture 的 SurfaceLOD 已完成固定49帧和必要的覆盖率读出；RGB与深度对照改善，同时保留少量低alpha及大幅深度前移的限制。报告、可重放入口与取回验收已完成，结果保留为研究候选，无后续训练控制器，没有启动 PVD 后训练或自由参数搜索。HELD9 未进入本次颜色目标，但此前已用于表示诊断与内部开发，不是独立未触碰泛化集。

展示部分已补齐49帧原生双状态PNG、可拖动对照和无JavaScript静态fallback；本次仅补图与展示，不改变科学结论。

执行边界：subagent 仅 gpt-6.1-sol/max；本会话明确保持 AutoDL 开机，不自动关机。failure_ledger_refs=[]，failure_ledger_delta=none；人工画质 verdict 未代填。

RCR-1支线当前由用户采用：2026-11-05明确选择RCR-1 delete用于异常蓝色RGB修复（scope=abnormal_blue_RGB_fix）；固定完整M2资产不变，原管线仅作reference。raw130/137亮片继续开放，完整画质无回退未作新结论，没有自动后续GPU队列。HELD9只对道路RGB校准target留出，不是独立native泛化验证。见[当前交互报告](autoresearch/scene008_rcr1_20261105/index.html)及[用户采用决定](autoresearch/scene008_rcr1_20261105/evidence/user_acceptance_20261105.json)。本支线failure_ledger_refs=[V82-F13]为历史评审，本次failure_ledger_delta=none；车辆外观主线安排不变。

a5补充只读诊断已归档，当前优先依据[两份实际报告直接复核](autoresearch/scene008_rcr1_20261105/followup_a5/index.html#report-crosscheck)：旧Full5 B30K保留a5独立动态身份，仅mask动态索引3/4（a6/a37）时a5仍完整；新资产报告的ownership6+37 Native10K在M0/M2前已缺损。配方提交3fefd98仅为首次归档，不是唯一causal commit；具体引入提交与单项机制仍未确定。这次追加只读图像/源码复核0GPU、0SGD，不是同预算单变量实验。既存原生重放为4帧×（旧2＋当前7）＝36次渲染，单列原有开销；RCR-1账本仍为26,609更新，采用范围保持异常蓝色RGB修复。failure_ledger_delta=none；车辆外观主线安排不变。

历史状态保存在[更新前快照](autoresearch/scene008_a6_insert_artifacts_20261105/prior_research_status.md)，实验导航见[索引](EXPERIMENTS.md)。
