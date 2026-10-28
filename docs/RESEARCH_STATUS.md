# 当前研究状态

当前任务WS-V43-TARGET-DETECTION-20261028，run recall-ceiling-r1。用户最新要求先解决召回上界，背景误选只记录、不否决扩候选；车底与关联阴影立即纳入，独立紫色effect。训练更新0，第三场景仍停止，旧队列/心跳未恢复。

第一场景raw98/113/114的原门槛重放、同域全开门槛、扩域一次、删除后全量重算并再剥一轮均完成。固定body/effect/broad域最终Gaussian alpha为数值0；原生RGB仍有斑块，分量和环境cube texel伴随闭合说明剩余暗像素由environment回填提供。不能把负值PNG截黑全当未删Gaussian，也不能宣称语义召回100%。[算法、组件和结果](autoresearch/v43_ownership_20261028/RECALL_CEILING_V1.md)。

逐行台账分检测域未覆盖、规则排除、after新贡献（exact0与阈值穿越分列）；13列旧规则重放零差异。background_veto=false，训练标签未写入，剥离后查询仅作诊断。暗像素查询0.1/0.2/0.3阈值同时记录；环境原参数、颜色、checkpoint均未改，无补景。

共享工作台及第一场景已增加召回上界入口。3帧×5阶段原生RGB/Gaussian/环境/alpha、逐行gate原因、旧历史双目标报告和AI初审均保留。AI不是人工GT，human_verdict=null；没有跨场景泛化验证。浏览器验证未采纳，交付以真实数据分量、PNG、静态接线检查为据。

下一步：先独立判断环境方向/texel中的目标关联污染与真实阴影，不能继续把这些残留当Gaussian漏删；候选生成中保留宽域，把深度/时序/运动冲突存作后续学习特征。获得独立车体/阴影/环境伪影评审后，再逐项加回背景保护，记录召回损失。当前未自动开新训练或扩大case。

现场：/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/unknown_detection_v3；当前资源/快照见[STATE](autoresearch/v43_ownership_20261028/STATE.json)。带GPU实例仅在结果取回、代码发布与活动进程排查完成后按长期授权关闭。failure_ledger_refs=[V82-F01]；failure_ledger_delta=none。

上一阶段[独立阴影证据](autoresearch/v43_ownership_20261028/TARGET_DETECTION_V3.md)和旧Top3均作为历史资料保留，不作为当前验收结论。
