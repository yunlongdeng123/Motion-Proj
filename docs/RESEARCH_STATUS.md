# 当前研究状态

更新：2026-11-07（Asia/Singapore）。

Scene008本轮已完成[路面接缝与车轮遮挡分项验证](autoresearch/scene008_road_seam_contact_20261107/index.html)。红侧得到可复现的道路支撑改进候选，蓝侧跨帧接缝仍开放；不合并为整场景通过/失败，原V44正式版本保留，human_verdict=null。

主候选保持C2/SurfaceLOD车辆、父cellfit1000道路颜色、原生10K、旧道路1K/gates、环境与轨迹不变。10098个世界道路父格在同一切平面细分为40392子格；三轴sigma减半，intrinsic opacity仍.99。仅新道路使用parent-relative Mip幅度归一化，按相机协方差确定有效幅值，与删除指令无关；所有层仍共同进入原生Gaussian排序。没有车体前景优先级、逐像素挖路、车辆整体抬升或新静态退役。

红侧固定raw128的23条车面在前射线，车辆贡献均值.566785→.666535；raw130固定24条.622210→.734828，后者逐条增加且无>.01退化。两帧共23条地面在前射线仍不露出车体。相对V44仍有6/7条严重chart-first缺失，近接触另列；这是当前车面/实测支持地面的成像诊断，不是隐藏轮胎准确率。原生滤波地板与中心深度排序分开解释，不宣称完整Mip-Splatting、2DGS或MOG不透明首交。

蓝侧raw128/delete近似蓝框的1385个原观测道路像素，RGB MAE .031119→.029598；固定两侧6bins的残差跳变.012529→.009794。二者均非整条边界验收。独立的5关键帧世界patch副表显示跨帧回退：raw105/delete/lateral-y 8bins .011334→.019934；raw137/delete/lateral-y 15bins .010415→.015789。KEEP可有相反趋势；主蓝框与世界patch不同分母明确分表，far边界和无配对分桶保持unsupported，不记为0。

分组表显示蓝圈既有新/旧道路交接，也有新道路与暗蓝静态层混合，不能统一归成两套道路颜色不一致。固定6世界basis/18RGB系数的源颜色约束QP只略降蓝框RGB误差，没有实质改善固定接缝指标；该颜色控制和5帧冻结组合仅保留对照，不并入主资产，不继续调权重/步数。下一步应验证实测道路对冲突旧静态表示的持久接管，以及车地查询算子的兼容性，而不是继续补自由颜色。

主AA候选完成49帧双状态；相对父R0的98个状态帧，道路与车外保护MAE均下降，但局部接缝回退及少量a6逐帧微小RGB退化仍披露。raw137固定720像素新道路solo alpha最小.999794，0个低于.98。40392行资产和父来源映射已保存，5关键帧×双状态独立10次重放的全部保存浮点字段及PNG与最终结果逐值相同。删除只移除a37，两种状态共用同一份资产和滤波规则。

本轮1095次native前向，0反向；含92次蓝侧真实短程/正式/线性验证与10次独立重放。正式80F及smoke/正常方程新进程恢复分别记账；颜色QP的smoke与formal三通道各解一次，不能写成没有拟合。原模型、车辆与预训练网络权重更新为0；R1未扩，学习型道路生成先验未训练，R4决策泛化未验证。代码、资产、报告与必要状态已取回核验。failure_ledger_refs=[V82-F13]，failure_ledger_delta=updated:V82-F13。

subagent仅gpt-6.1-sol/max；本轮无待启动GPU队列。按用户要求AutoDL保持开机，不自动关机。RCR-1异常蓝色RGB的原限定采用决定与原V44回退入口保持；本轮几何支撑候选不代表整场景已采用。

v44＋a5车顶恢复已完成四帧接线、全49 BUILD身份统计与完整49帧KEEP/DELETE诊断，作为[可审阅候选](autoresearch/scene008_v44_full5_integration_20261107/index.html)交付，原v44仍保留默认且可切回，未替用户接受整体画质。复用50d6daa8的完整五身份10K参数，把初始来源谱系与当前可见语义分开：按原BUILD a5掩码下的逐高斯joint alpha×T贡献，固定多数归属选择232个Full5静态行和120722个动态行，其中动态来源5有118133行、来源37有2589行，共同作为估计语义a5接入；v44其余static/road1K gates/fine11×9/C2/a37/env和原行号均不改。新增a5行选择只用固定三维归属sidecar，不依据当帧掩码或GT；原v44输入读取依赖保持，两删除状态共用相同场景表示。Full5推理备份与身份包已取回校验，公共CLI105 DELETE与配对产物PNG逐字节相同。

同原固定121128像素帧，a5 KEEP MAE .0454184441→.0348554260、上部25906像素帧 .0773441193→.0515276369；DELETE .0454534998→.0348808236。全部24个a5可见帧由代理复核，车顶/风挡连续结构恢复，细节仍软。v44保护门均过：逐帧a6 RGB/深度、KEEP a37和DELETE road/blue指标Δ0。不宣称逐帧RGB更好：原25门21过，94全域/上部两状态四项严格改善门仍FAIL；12/24可见帧全车RGB回退，最坏90 Δ+.0134514。source-only加法与严格旧static来源退役失败均保留；不修改原阈值，不扩亮片/道路研究。本轮303次native前向＋49次dummy-feature贡献反向，Gaussian/颜色/opacity优化更新0；全49身份统计使用原训练BUILD掩码，不是独立泛化。所有自有worker已退出且无未来控制器，GPU按用户要求保持开机。human_verdict=null、quality_accepted=false；failure_ledger_refs=[V82-F06]，failure_ledger_delta=updated:V82-F06。

a5本轮已完成[完整五身份原生10K重建与身份保持](autoresearch/scene008_identity_edit_interface_20261107/index.html)：a5车顶和挡风玻璃恢复连续可辨结构，原生细节仍偏软；49帧已评价、24个a5可见帧已由代理复核，不宣称GT级清晰或人工质量验收。建模集合固定dynamic_actor_ids=[5,6,18,26,37]，所有车辆使用各自完整真实源点；查询目标改变不再把保留车辆转到静态分支。新增逐高斯actor与来源谱系随clone/split/prune及完整状态恢复保持，原生输入/训练数学不变；首步13项原始loss与B相同，元数据有/无时真实拓扑、参数/Adam/RNG等价。旧四处原生Full5权重缺可靠动态身份，未猜标签，而是新训带身份检查点。

同原生10K固定121,128像素帧，旧Single/new Full5身份RGB MAE .06209628→.03032921；固定上部25,906像素帧 .11551418→.03964854。上部owner5 raw alphaT均值.916079、主owner5像素率.980506仅作身份一致性证据。新正式10,000＋真实smoke/完整恢复4,683更新独立记账；连同[旧研究收口](autoresearch/scene008_a5_regression_20261106/index.html)44,622，累计59,305。旧Single止20K，30K未执行，不下30K结论。新完整训练现场保留，推理包去Adam后RGB/alpha/owner三帧逐值等价并提供本机备份与加载说明。静态车形残影与编辑质量归v44，本分支不处理。GPU按用户要求保持开机；本轮未用重置卡，human_verdict=null。failure_ledger_refs=[V82-F06]，failure_ledger_delta=updated:V82-F06。

历史状态保存在[更新前快照](autoresearch/scene008_a6_insert_artifacts_20261105/prior_research_status.md)，实验导航见[索引](EXPERIMENTS.md)。
