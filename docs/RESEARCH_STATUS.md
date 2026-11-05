# 当前研究状态

2026-11-05，Scene008完整下部C/CM资格与剂量匹配五臂验证未采用；mode增量很小，raw97略超累计outside预算。预检显示corridor仅覆盖7.97%原生前方责任；1502个双侧观测候选覆盖40.26%，转向完整重表达无退役控制。[实验/预检/下一步合同](autoresearch/scene008_a6_depth_consistency_20261105/README.md)。

2026-11-05，Scene008完成近/同/远物理法向支撑9个对照，0个通过全部方法验收；324个保存指标重放及81条实际选择状态重载。物理支撑位置与原生中心排序深度需区分。AutoDL保持开机。[实验与代码](autoresearch/scene008_a6_layered_handoff_20261105/README.md)。

2026-11-05，Scene008完成三组实测表面薄壳/贡献登记成对接管。两套规则均未通过，216分支重放及独立审计通过；全部登记行mixed且法线范围近2–3m，下一步先区分深度层。保留父资产，AutoDL开机。[实验与代码](autoresearch/scene008_a6_coupled_handoff_20261105/README.md)。

2026-11-05，Scene008 完成启发式规则首轮验证。旧 free 全为膨胀框外项，原始 199 帧新增 1,003 条候选体积内第一回波自由线段。独立叠加式实测代理面在留出正对照中恶化 RGB 与深度，未采用。[证据、对照和下一步耦合接管设计](autoresearch/scene008_a6_heuristic_validation_20261105/README.md)。

2026-11-05，Scene008 a6 在实际光栅射线上按逐行遮挡影响分配持久衰减。黑底显露 50.85% → 50.93%，删 a37 街景 48.78% → 48.86%。49 帧双状态保护与独立重放通过。[实验、排序对照与账本](autoresearch/scene008_a6_depth_order_20261105/README.md)。

2026-11-05，Scene008 a6 采用LiDAR 裁剪后重分配。黑底下部显露 50.00% → 50.85%，删 a37 街景显露 47.93% → 48.78%。49 帧保护与道路复核通过。[实验、代码与逐帧指标](autoresearch/scene008_a6_continuous_takeover_20261105/README.md)。

2026-11-05，Scene008 采用共享 11×9 世界网格校正道路颜色。无编辑道路 MAE 0.032733 → 0.031688，删 a37 道路 MAE 0.022207 → 0.018912，两种状态接缝带改善。49 帧几何和可见贡献重放一致，a6 下部 PVD 街景显露率仍为 47.93%。[逐帧验收与代码](autoresearch/scene008_ray_road_quality_20261105/README.md)。

2026-11-05，Scene008 a6 以射线积分裁剪部分高斯支撑。通过保护的同位替换将黑底下部显露率从 48.90% 提至约 50.00%，删 a37 的街景从 46.84% 提至 47.93%。[原生一致性、逐帧指标与代码](autoresearch/scene008_a6_ray_switch_20261105/README.md)

全量同位替换有 5 帧非目标保护失败。现有空射线证据与受保护的非目标像素无交集，中央混合支撑仍需确认。整体道路和可见车身质量基本持平。AutoDL 保持开机。

[上一轮保留侧高斯重表达](autoresearch/scene008_a6_truncated_support_20261105/README.md)。


2026-11-05，Scene008旧裁剪分支收尾，转向单一完整a6资产。完整2048 PVD固定几何＋BUILD可见RGB拟合，原生a6退出成像，49帧保存重载和统一遮挡验证通过；画质未验收，native/new黑底RGB .01873/.04951，空间与SH容量差异待独立拆分。AutoDL保持开机。[报告、完整资产与下一合同](autoresearch/scene008_a6_complete_asset_20261105/README.md)。


2026-11-05，Scene008完整单一资产完成材质/统一采样/实测法向校正三组49帧对照。M1受限SH3小幅增益，M2全车细采样RGB .04951→.03812、深度 .32646→.27498m；M3仅234中心支持，深度微改善但RGB略差。局部颜色诊断进一步确认：左道路/右路缘窗口49/49帧相对原图额外偏蓝，原生方向背景与静态颜色补偿未完全匹配；道路插入又增加边界跳变，既有颜色场仅部分缓解。[49帧冻结对照与分层证据](autoresearch/scene008_local_colour_diag_20261105/index.html)。AutoDL保持开机。[报告、资产与复核](autoresearch/scene008_a6_asset_controls_20261105/README.md)。


2026-11-05，Scene008车辆先验输入输出审计：PVD只输出XYZ，M2的10557/18432点无RGB损失支持，低于全199帧实测最低高度的8178/8407点保持全局蓝灰回退色；原图、单位、坐标、颜色解码链未发现确定bug。完成4x1000步冻结几何输入敏感性及同输入复现对照；PVCNN数值非逐位确定，结果不采用。资产新增显式外观来源合同，三帧黑底/街景渲染不变。暂不训练、不启动道路分支、AutoDL保持开机。[报告与证据](autoresearch/scene008_a6_io_audit_20261105/README.md)。


2026-11-05，Scene008道路先验前置诊断与计划完成：raw137独立720px道路alpha0，主要因旧road envelope不到目标world范围；295可信query中294在raw154–175有真实FRONT见证，优先实测恢复。当前199仅FRONT，未证明全相机nevervisible。M2冻结，未修路/训练，AutoDL开机。[报告与下一阶段合同](autoresearch/scene008_road_prior_assessment_20261105/README.md)。
