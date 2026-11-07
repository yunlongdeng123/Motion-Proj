# 当前研究状态

更新：2026-11-07（Asia/Singapore）。

Scene008沿用[已交付v44＋a5车顶恢复](autoresearch/scene008_v44_full5_integration_20261107/index.html)累计21K组件（v44 10K＋道路1K＋恢复a5 Full5 10K，身份接入0K）。用户撤回端到端20K／纯Full5替换；车顶、车辆、身份、环境、背景、几何及intrinsic opacity冻结，新增仅道路RGB。用户明确GPU保持开机，不关机或停止其他服务；子代理仅sol6.1/max。

[C49材质完整对照](autoresearch/scene008_v44_a5_c49_rgb_20261107/index.html)新正式2K已完成，不续5K/10K。49均BUILD（含旧HELD9），C400历史初始化，新Adam/正式0；2119父6357RGB可训练、7979父冻结。真实600.208秒smoke及恢复共1996工程更新单列；保存/评价/全恢复/next2覆盖，起点exact、next2 Adam差2.84e-14不冒称全部exact。正式2K为4016F/4000B，完整49双状态Cwarm vs2K再196F；末5对前5完整epoch MSE仅−.0009545%而RGB仍漂移，C2遮挡与侧缝仍在，按证据止2K。13包599,316,804B已本机取回逐SHA验证。主线正式累计23K，不是单模型连续23K。

[四帧接入与控制](autoresearch/scene008_v44_a5_road_fusion_integration_20261107/index.html)：H2用冻结模型前景贡献保守裁道路child，减大银片并保a5/a6，薄边/非前景守卫仍失败；I固定真实世界0.4m coverage在128/137 KD四次前向更差，停止、不搜宽度。J在H2＋2K固定几何测2119父RGB仿射算子，24F/16dummyB验证，一次LP＋8F后6池过，但局部bin回退。

K在同算子/变量/目标下加106逐bin不回退约束，一次CPU LP＋真实8F，6汇总与106原容差门全部过。主128 KEEP/DELETE=.00377605/.00629795（21K .00548139/.00800314）；worldnear=.00500314/.00560753，lateral=.00368891/.00343041。0Adam/0原模型更新，仅40父RGB变化；7979冻结父及全部非RGBexact，相对H2 alpha/depth/8组件/语义a5色逐值同。逐bin并非均严格改善，最小原门余量7.54e-8，额外工程reserve有2.46e-8超出，原1e-6门未改。

尚未整体通过：原94门55通过/31失败/8空域，失败为protectedNR RGB7、originalroad RGB8、alpha/depth16。真图仍有128细黑缝、137色边，128(301,221)/137(300,227)自行车道白点H2/J/K完全同值。最大几何差多数在被删37旧足迹，但有真实protectedNR及a26/a18各1投影尾，不全豁免；137坏near bin与H3地先像素无交集，不能归因于C2白尾。

两处白点已真实2F/2dummyB测全父贡献并重构四版本RGB，0优化：>99%贡献来自2119可训练父，否定冻结父锁死。Cwarm灰色到正式2K显著变白，说明材质学习诱发，J/K未改掉。唯一12父回滚L已真实8F完成，实际仅10父色变、0B/0优化/0新solve；106 signed与K exact、原6+106全过，geometry/身份exact。128白点MAE .47224→.03038（21K .01314仍优），137 .42410→.01774（21K .02492），总RGB同Cwarm。真图刺眼白点消除，旁边暗点、细缝/色边仍在；原94仍55PASS/31FAIL/8空域。L只是因果控制，不作为通用修复。K/L保留4帧，未扩49，不把完整C控制49当K49。已定位实际资格仅aggregate fit>0、损失仅fit_mask RGB，没有protect监督，未发现target错路径；下一步修这项覆盖接口，避免靠新坏点逐一回滚或加步数。同步完成源码/结果归档、失败卡和白名单Git提交推送；不相关E3脏文件保留。

默认v44＋a5、源C与中央viewer未改。quality_accepted=false、human_verdict=null。failure_ledger_refs=[V82-F13,V82-F06]，failure_ledger_delta=updated:V82-F13,updated:V82-F06。历史状态见[原件](autoresearch/scene008_v44_a5_road_fusion_integration_20261107/prior_research_status_before_road_integration.md)，run见[索引](EXPERIMENTS.md)。
