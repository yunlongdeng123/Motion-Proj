# 当前研究状态

更新：2026-11-07（Asia/Singapore）。

Scene008 当前车体阶段保留 C2 连续表面、固定 RGBA 与同一 S/F SurfaceLOD；PVD 与预训练权重不更新。已有49帧双状态和拖动报告保留，几何前移、少量低alpha及内部开发域边界仍以原报告为准，不扩为360度或隐藏真值准确性结论。

当前工作为 v44：把 RCR-1 已保存的 RGB×1 原生10K及其 freshroad1K/gates 接入当前冻结车体。独立代码和实际远端CPU输入检查已完成（552项）；原11×9世界色场仅为来自旧M2联合场景的冻结迁移，未重标定。新组合未运行GPU、未得到画质verdict；当前CUDA不可用，下一步带GPU后先raw105/113/130/137成对pilot，通过工程检查再完整49帧。详见[v44接线与异常评估](autoresearch/scene008_v44_rgb_asset_20261107/index.html)。

银亮片与暗斑分别保留诊断：历史亮片净增亮由static主导、环境×T仍有车形，不能称为PVD车体产生或去蓝已修复。v44将输出同排序四组RGB/alpha贡献；新组合来源待实际渲染核对。无新native/road/asset训练，无待启动控制器。

执行边界：subagent仅gpt-6.1-sol/max；本会话保持AutoDL开机，不自动关机。failure_ledger_refs=[]，本次failure_ledger_delta=none；人工画质verdict未代填。

RCR-1支线当前由用户采用：2026-11-05明确选择RCR-1 delete用于异常蓝色RGB修复（scope=abnormal_blue_RGB_fix）；固定完整M2资产不变，原管线仅作reference。raw130/137亮片继续开放，完整画质无回退未作新结论，没有自动后续GPU队列。HELD9只对道路RGB校准target留出，不是独立native泛化验证。见[当前交互报告](autoresearch/scene008_rcr1_20261105/index.html)及[用户采用决定](autoresearch/scene008_rcr1_20261105/evidence/user_acceptance_20261105.json)。本支线failure_ledger_refs=[V82-F13]为历史评审，本次failure_ledger_delta=none；车辆外观主线安排不变。

a5补充只读诊断已归档，当前优先依据[两份实际报告直接复核](autoresearch/scene008_rcr1_20261105/followup_a5/index.html#report-crosscheck)：旧Full5 B30K保留a5独立动态身份，仅mask动态索引3/4（a6/a37）时a5仍完整；新资产报告的ownership6+37 Native10K在M0/M2前已缺损。配方提交3fefd98仅为首次归档，不是唯一causal commit；具体引入提交与单项机制仍未确定。这次追加只读图像/源码复核0GPU、0SGD，不是同预算单变量实验。既存原生重放为4帧×（旧2＋当前7）＝36次渲染，单列原有开销；RCR-1账本仍为26,609更新，采用范围保持异常蓝色RGB修复。failure_ledger_delta=none；车辆外观主线安排不变。

历史状态保存在[更新前快照](autoresearch/scene008_a6_insert_artifacts_20261105/prior_research_status.md)，实验导航见[索引](EXPERIMENTS.md)。
