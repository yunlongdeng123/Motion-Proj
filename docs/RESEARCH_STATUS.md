# 当前研究状态

更新：2026-11-07（Asia/Singapore）。

Scene008 当前保留 C2 连续表面、固定 RGBA 与同一 S/F SurfaceLOD 车体。v44已完成RCR-1背景接入的4帧pilot和完整49帧双状态：同一车体，背景为已保存RGB×1 native10K及其匹配road1K/gates，11×9色场仅冻结迁移，未重标定。PVD、车体和所有预训练权重本轮无更新。

本轮优先道路蓝紫异常；用户明确将银残片留到下一轮。固定原观测道路1075281像素的绝对蓝误差keep .0165024→.0081845、delete .0143270→.0084500，49帧两状态逐帧均改善。全49帧delete道路proxy RGB .0186062→.0096190；该proxy不是隐藏道路独立真值。详见[v44逐帧报告](autoresearch/scene008_v44_rgb_asset_20261107/index.html)。

a6整体delete RGB .0549473→.0549451、深度 .171958→.170030m，无新增低alpha洞；局部并非零回退：a6最坏raw137，深度最坏raw101，a37 keep最坏raw122，均在报告披露。raw105真实蓝车局部超蓝尾部仍存在，不做整图去色。原预登记蓝ROI为空，另存有真实RGB支持的独立背景域与全49原观测道路统计，未改正式分母。HELD9仍为内部开发，不声称独立泛化。

工程与来源复核完成：pilot16＋full49 196＋必要分组诊断3＝215次native前向；独立CPU全49 1550项、跨阶段42项、分组90项通过。无新训练或求解。PNG/JSON/报告已取回；大float与components保留原campaign。银片主要为static增量和environment×T减量抵消的成像问题，训练根因未证；本轮不处理银片。

执行边界：subagent仅gpt-6.1-sol/max。按用户要求AutoDL保持开机，不自动关机；无待启动训练控制器。人工画质verdict仍为null，未替用户接受整体画质。failure_ledger_refs=[]；failure_ledger_delta=none。旧报告及CPU阶段保存。

RCR-1支线当前由用户采用：2026-11-05明确选择RCR-1 delete用于异常蓝色RGB修复（scope=abnormal_blue_RGB_fix）；固定完整M2资产不变，原管线仅作reference。raw130/137亮片继续开放，完整画质无回退未作新结论，没有自动后续GPU队列。HELD9只对道路RGB校准target留出，不是独立native泛化验证。见[当前交互报告](autoresearch/scene008_rcr1_20261105/index.html)及[用户采用决定](autoresearch/scene008_rcr1_20261105/evidence/user_acceptance_20261105.json)。本支线failure_ledger_refs=[V82-F13]为历史评审，本次failure_ledger_delta=none；车辆外观主线安排不变。

a5本轮已完成[完整五身份原生10K重建与身份保持](autoresearch/scene008_identity_edit_interface_20261107/index.html)：a5车顶和挡风玻璃恢复连续可辨结构，原生细节仍偏软；49帧已评价、24个a5可见帧已由代理复核，不宣称GT级清晰或人工质量验收。建模集合固定dynamic_actor_ids=[5,6,18,26,37]，所有车辆使用各自完整真实源点；查询目标改变不再把保留车辆转到静态分支。新增逐高斯actor与来源谱系随clone/split/prune及完整状态恢复保持，原生输入/训练数学不变；首步13项原始loss与B相同，元数据有/无时真实拓扑、参数/Adam/RNG等价。旧四处原生Full5权重缺可靠动态身份，未猜标签，而是新训带身份检查点。

同原生10K固定121,128像素帧，旧Single/new Full5身份RGB MAE .06209628→.03032921；固定上部25,906像素帧 .11551418→.03964854。上部owner5 raw alphaT均值.916079、主owner5像素率.980506仅作身份一致性证据。新正式10,000＋真实smoke/完整恢复4,683更新独立记账；连同[旧研究收口](autoresearch/scene008_a5_regression_20261106/index.html)44,622，累计59,305。旧Single止20K，30K未执行，不下30K结论。新完整训练现场保留，推理包去Adam后RGB/alpha/owner三帧逐值等价并提供本机备份与加载说明。静态车形残影与编辑质量归v44，本分支不处理。GPU按用户要求保持开机；本轮未用重置卡，human_verdict=null。failure_ledger_refs=[V82-F06]，failure_ledger_delta=updated:V82-F06。

历史状态保存在[更新前快照](autoresearch/scene008_a6_insert_artifacts_20261105/prior_research_status.md)，实验导航见[索引](EXPERIMENTS.md)。
