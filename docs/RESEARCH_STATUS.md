# 当前研究状态

更新：2026-11-07（Asia/Singapore）。

Scene008 当前保留 C2 连续表面、固定 RGBA 与同一 S/F SurfaceLOD 车体。v44已完成RCR-1背景接入的4帧pilot和完整49帧双状态：同一车体，背景为已保存RGB×1 native10K及其匹配road1K/gates，11×9色场仅冻结迁移，未重标定。PVD、车体和所有预训练权重本轮无更新。

本轮优先道路蓝紫异常；用户明确将银残片留到下一轮。固定原观测道路1075281像素的绝对蓝误差keep .0165024→.0081845、delete .0143270→.0084500，49帧两状态逐帧均改善。全49帧delete道路proxy RGB .0186062→.0096190；该proxy不是隐藏道路独立真值。详见[v44逐帧报告](autoresearch/scene008_v44_rgb_asset_20261107/index.html)。

a6整体delete RGB .0549473→.0549451、深度 .171958→.170030m，无新增低alpha洞；局部并非零回退：a6最坏raw137，深度最坏raw101，a37 keep最坏raw122，均在报告披露。raw105真实蓝车局部超蓝尾部仍存在，不做整图去色。原预登记蓝ROI为空，另存有真实RGB支持的独立背景域与全49原观测道路统计，未改正式分母。HELD9仍为内部开发，不声称独立泛化。

工程与来源复核完成：pilot16＋full49 196＋必要分组诊断3＝215次native前向；独立CPU全49 1550项、跨阶段42项、分组90项通过。无新训练或求解。PNG/JSON/报告已取回；大float与components保留原campaign。银片主要为static增量和environment×T减量抵消的成像问题，训练根因未证；本轮不处理银片。

执行边界：subagent仅gpt-6.1-sol/max。按用户要求AutoDL保持开机，不自动关机；无待启动训练控制器。人工画质verdict仍为null，未替用户接受整体画质。failure_ledger_refs=[]；failure_ledger_delta=none。旧报告及CPU阶段保存。

RCR-1支线当前由用户采用：2026-11-05明确选择RCR-1 delete用于异常蓝色RGB修复（scope=abnormal_blue_RGB_fix）；固定完整M2资产不变，原管线仅作reference。raw130/137亮片继续开放，完整画质无回退未作新结论，没有自动后续GPU队列。HELD9只对道路RGB校准target留出，不是独立native泛化验证。见[当前交互报告](autoresearch/scene008_rcr1_20261105/index.html)及[用户采用决定](autoresearch/scene008_rcr1_20261105/evidence/user_acceptance_20261105.json)。本支线failure_ledger_refs=[V82-F13]为历史评审，本次failure_ledger_delta=none；车辆外观主线安排不变。

a5补充只读诊断已归档，当前优先依据[两份实际报告直接复核](autoresearch/scene008_rcr1_20261105/followup_a5/index.html#report-crosscheck)：旧Full5 B30K保留a5独立动态身份，仅mask动态索引3/4（a6/a37）时a5仍完整；新资产报告的ownership6+37 Native10K在M0/M2前已缺损。配方提交3fefd98仅为首次归档，不是唯一causal commit；具体引入提交与单项机制仍未确定。这次追加只读图像/源码复核0GPU、0SGD，不是同预算单变量实验。既存原生重放为4帧×（旧2＋当前7）＝36次渲染，单列原有开销；RCR-1账本仍为26,609更新，采用范围保持异常蓝色RGB修复。failure_ledger_delta=none；车辆外观主线安排不变。

历史状态保存在[更新前快照](autoresearch/scene008_a6_insert_artifacts_20261105/prior_research_status.md)，实验导航见[索引](EXPERIMENTS.md)。
