# 当前研究状态

更新：2026-11-07（Asia/Singapore）。

Scene008已完成[道路融合目标→持久三维写回](autoresearch/scene008_road_fusion_bake_20261107/index.html)。保留C作为“已观测道路外观／横向边界改进候选”，近端仍开放；原V44及前轮AA保留，human_verdict=null，整场景未采用。

使用190个非HELD FRONT实测道路见证构成共享chart，固定道路几何与原生场景。A未融合atlas、B固定Poisson、C固定10px真实RGB边缘混合，三组均fresh基线400paired steps、同2053父格RGB/6159标量、同mask/Adam.01；8045无监督父格原色不变，40392子格继承同一父色。C是A/B固定49帧回退后，依据投影有效mask硬切的额外梯度证据登记的唯一fallback，未扩带宽/权重网格，未执行legacy静态资格或退役。

raw128 DELETE原蓝框1385道路像素RGB MAE .029598→C .015966；near6bins却.009794→.011725。全49横向277bins、40帧支持，DELETE .016715→.009164，KEEP .015085→.008788；近端248bins、47帧支持，DELETE .009340→.009739、KEEP .008399→.008709仍回退；far全部unsupported。C横向DELETE 147bins改善、107回退、23相等，不由平均数代替逐域验收。保护均值下降但KEEP7/DELETE4帧有小回退，最坏均值Δ+4.42e-5/+2.81e-5。

C主要是观测RGB锚定：BUILD atlas权重均值.1866且无w=1，C目标依视角有效mask；HELD目标本身含queryO仅作诊断，不叫预测质量。3D拟合只用BUILD40，native颜色不condition queryO；继承场景已预曝，非独立泛化。A/B负例与C强控制均保留，不单独归因Poisson或学习道路先验。

三组294个材料状态的几何/alpha/分组贡献与AA逐值一致，车体预乘颜色保持；固定71红侧射线W不变，原6/7严重chart-first缺失未解决。三份完整40392行资产已导出，30次fresh独立重放的所有浮点字段/PNG逐字节一致；无编辑/删除共用同一资产，删除只关a37。

本轮3080F/2528B，含正式1200optimizer steps与单列24次smoke/续步工程更新；原native/车辆/几何/opacity/预训练网络更新0。完整恢复起点allstate exact；续步参数exact而少量Adam moment存在极小差异，实际收据保留。failure_ledger_refs=[V82-F13]，failure_ledger_delta=updated:V82-F13。R1盲化生成、完整MOG首交与R4泛化未新增验证。

下一步锁定近端边界的分层残差与跨帧对应，核实同一实测表面在新旧层间的材质衔接，再建立保护／接管账本。本轮不继续调带宽或扩大旧高斯关闭范围。源码、状态、资产、报告已归档取回核验；子代理仅sol6.1/max，AutoDL按用户要求保持开机。

发布核验：完整交付逐文件与PNG静态检查通过；自动浏览器交互／移动布局验收被本地file协议策略拦截，未执行，也未宣称通过。

v44＋a5车顶恢复已完成四帧接线、全49 BUILD身份统计与完整49帧KEEP/DELETE诊断，作为[可审阅候选](autoresearch/scene008_v44_full5_integration_20261107/index.html)交付，原v44仍保留默认且可切回，未替用户接受整体画质。复用50d6daa8的完整五身份10K参数，把初始来源谱系与当前可见语义分开：按原BUILD a5掩码下的逐高斯joint alpha×T贡献，固定多数归属选择232个Full5静态行和120722个动态行，其中动态来源5有118133行、来源37有2589行，共同作为估计语义a5接入；v44其余static/road1K gates/fine11×9/C2/a37/env和原行号均不改。新增a5行选择只用固定三维归属sidecar，不依据当帧掩码或GT；原v44输入读取依赖保持，两删除状态共用相同场景表示。Full5推理备份与身份包已取回校验，公共CLI105 DELETE与配对产物PNG逐字节相同。

同原固定121128像素帧，a5 KEEP MAE .0454184441→.0348554260、上部25906像素帧 .0773441193→.0515276369；DELETE .0454534998→.0348808236。全部24个a5可见帧由代理复核，车顶/风挡连续结构恢复，细节仍软。v44保护门均过：逐帧a6 RGB/深度、KEEP a37和DELETE road/blue指标Δ0。不宣称逐帧RGB更好：原25门21过，94全域/上部两状态四项严格改善门仍FAIL；12/24可见帧全车RGB回退，最坏90 Δ+.0134514。source-only加法与严格旧static来源退役失败均保留；不修改原阈值，不扩亮片/道路研究。本轮303次native前向＋49次dummy-feature贡献反向，Gaussian/颜色/opacity优化更新0；全49身份统计使用原训练BUILD掩码，不是独立泛化。所有自有worker已退出且无未来控制器，GPU按用户要求保持开机。human_verdict=null、quality_accepted=false；failure_ledger_refs=[V82-F06]，failure_ledger_delta=updated:V82-F06。

a5本轮已完成[完整五身份原生10K重建与身份保持](autoresearch/scene008_identity_edit_interface_20261107/index.html)：a5车顶和挡风玻璃恢复连续可辨结构，原生细节仍偏软；49帧已评价、24个a5可见帧已由代理复核，不宣称GT级清晰或人工质量验收。建模集合固定dynamic_actor_ids=[5,6,18,26,37]，所有车辆使用各自完整真实源点；查询目标改变不再把保留车辆转到静态分支。新增逐高斯actor与来源谱系随clone/split/prune及完整状态恢复保持，原生输入/训练数学不变；首步13项原始loss与B相同，元数据有/无时真实拓扑、参数/Adam/RNG等价。旧四处原生Full5权重缺可靠动态身份，未猜标签，而是新训带身份检查点。

同原生10K固定121,128像素帧，旧Single/new Full5身份RGB MAE .06209628→.03032921；固定上部25,906像素帧 .11551418→.03964854。上部owner5 raw alphaT均值.916079、主owner5像素率.980506仅作身份一致性证据。新正式10,000＋真实smoke/完整恢复4,683更新独立记账；连同[旧研究收口](autoresearch/scene008_a5_regression_20261106/index.html)44,622，累计59,305。旧Single止20K，30K未执行，不下30K结论。新完整训练现场保留，推理包去Adam后RGB/alpha/owner三帧逐值等价并提供本机备份与加载说明。静态车形残影与编辑质量归v44，本分支不处理。GPU按用户要求保持开机；本轮未用重置卡，human_verdict=null。failure_ledger_refs=[V82-F06]，failure_ledger_delta=updated:V82-F06。

历史状态保存在[更新前快照](autoresearch/scene008_a6_insert_artifacts_20261105/prior_research_status.md)，实验导航见[索引](EXPERIMENTS.md)。
