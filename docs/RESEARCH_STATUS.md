# 当前研究状态

更新：2026-11-07（Asia/Singapore）。

Scene008 当前保留 C2 连续表面、固定 RGBA 与同一 S/F SurfaceLOD 车体。v44已完成RCR-1背景接入的4帧pilot和完整49帧双状态：同一车体，背景为已保存RGB×1 native10K及其匹配road1K/gates，11×9色场仅冻结迁移，未重标定。PVD、车体和所有预训练权重本轮无更新。

本轮优先道路蓝紫异常；用户明确将银残片留到下一轮。固定原观测道路1075281像素的绝对蓝误差keep .0165024→.0081845、delete .0143270→.0084500，49帧两状态逐帧均改善。全49帧delete道路proxy RGB .0186062→.0096190；该proxy不是隐藏道路独立真值。详见[v44逐帧报告](autoresearch/scene008_v44_rgb_asset_20261107/index.html)。

a6整体delete RGB .0549473→.0549451、深度 .171958→.170030m，无新增低alpha洞；局部并非零回退：a6最坏raw137，深度最坏raw101，a37 keep最坏raw122，均在报告披露。raw105真实蓝车局部超蓝尾部仍存在，不做整图去色。原预登记蓝ROI为空，另存有真实RGB支持的独立背景域与全49原观测道路统计，未改正式分母。HELD9仍为内部开发，不声称独立泛化。

工程与来源复核完成：pilot16＋full49 196＋必要分组诊断3＝215次native前向；独立CPU全49 1550项、跨阶段42项、分组90项通过。无新训练或求解。PNG/JSON/报告已取回；大float与components保留原campaign。银片主要为static增量和environment×T减量抵消的成像问题，训练根因未证；本轮不处理银片。

执行边界：subagent仅gpt-6.1-sol/max。按用户要求AutoDL保持开机，不自动关机；无待启动训练控制器。人工画质verdict仍为null，未替用户接受整体画质。failure_ledger_refs=[]；failure_ledger_delta=none。旧报告及CPU阶段保存。

RCR-1支线当前由用户采用：2026-11-05明确选择RCR-1 delete用于异常蓝色RGB修复（scope=abnormal_blue_RGB_fix）；固定完整M2资产不变，原管线仅作reference。raw130/137亮片继续开放，完整画质无回退未作新结论，没有自动后续GPU队列。HELD9只对道路RGB校准target留出，不是独立native泛化验证。见[当前交互报告](autoresearch/scene008_rcr1_20261105/index.html)及[用户采用决定](autoresearch/scene008_rcr1_20261105/evidence/user_acceptance_20261105.json)。本支线failure_ledger_refs=[V82-F13]为历史评审，本次failure_ledger_delta=none；车辆外观主线安排不变。

a5支线当前进行身份集合耦合研究，见[原生重建、初始化与监督对照](autoresearch/scene008_a5_regression_20261106/index.html)。既存Full5与仅a37的原生缓存双方均10K时已存在车顶/挡风玻璃差异，不能仅归因旧版30K；缓存PNG与fresh float指标分表。a5从完整21,358源点动态分支变为静态1,496采样行，初始化和目标同时改变。补齐19,916缺失unique xyz/time真实源点在5K仅降低部分误差，车顶未恢复；time-shift和screen-radius两次11977工程候选也未修复，均未采用。当前原生single从既有11977完整状态续至30K（15/20/30K节点），另以两初始化各做staticRGB1→5的5K消融；其他先验及训练49帧采样不变。新候选优先97/105/109 pilot后再扩49，已启动流程继续。闭合至Round1账目20,000正式＋6,599工程＝26,599，新研究按实际收据另计；failure_ledger_refs=[V82-F06]，本次failure_ledger_delta=updated:V82-F06。人工画质verdict保持null；AutoDL按最新要求保持开机，不使用重置卡。RCR-1/a6主线采用范围不变。

历史状态保存在[更新前快照](autoresearch/scene008_a6_insert_artifacts_20261105/prior_research_status.md)，实验导航见[索引](EXPERIMENTS.md)。
