# 实验索引

本页按实验定位报告和原始证据；当前执行状态只见 [RESEARCH_STATUS](RESEARCH_STATUS.md)。完整历史过程保留在 [V7.4 实验历史](archive/2026-09/v74-0920/EXPERIMENT_HISTORY.md)及[V7.3 完整台账](archive/2026-09/pre-v74/V73_EXPERIMENTS.md)。

| task / run 或阶段 | 记录内容 | 入口 |
|---|---|---|
| WS-V81-DGGT-WAYMO-INFERENCE-20261011 / r1 | 官方基线、四支高斯与16帧Difix保留；24张单/双图参考对照及8组配对完成，三支仍有残影/黑斑、无参考收益；四帧CPU足迹仅几何估计，后续深度顺序CPU首帧alpha校准未过1灰阶门限即停止，无贡献结果；深色页44张RGB+4张分类图、107链接完整 | [流程、组件图与停止证据](v81/DGGT_WAYMO_INFERENCE_R1.md) · [轻量结果](v81/DGGT_WAYMO_INFERENCE_R1.json) |
| WS-V81-STORAGE-20261010 / r1 | 授权清理释放105.95GB；旧优化器退役、推理权重保留；当前P1继续运行 | [清单与恢复边界](v81/STORAGE_CLEANUP_20261010.md) · [实测](v81/STORAGE_CLEANUP_20261010.json) |
| WS-V81-SEEN-TO-SCENE-P1-20261009 / r1 | 用户暂停并转DGGT；最新完整14000已验证保留，最后日志14386，386尾部更新未落盘；父子退出、GPU释放，禁止自动续训/清理。最新已审结果仍12500六窗，旧20000计划停用 | [周期与暂停记录](v81/P1_REVIEW_CYCLES_R1.md)、[7500传播评估](v81/P1_STEP7500_EVALUATION_R1.md)、[协议](v81/YOUTUBE_VOS_P1_R1.md) |
| WS-V81-SEEN-TO-SCENE-20261009 / r1 | 原始nuScenes 6 train+2 val/200连续帧；两步真实优化+checkpoint恢复、独立val 25帧生成完成；显存/推理mask维度工程修复，18审核视频，非P1指标复现 | [协议与证据](v81/NUSCENES_P0_R1.md) · [实测](v81/P0_R1_RESULTS.json) |
| WS-V81-SEEN-TO-SCENE-20261009 / r0 | v7.7继承分支；Seen-to-Scene方法核对，CPU数据与传播原语准备；真实训练0步／推理0窗 | [报告](v81/SEEN_TO_SCENE_P0.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r53 | SAM3 ModelScope权重与真实轮廓CPU入口；旧矩形接口排查，推理0窗/训练0步 | [报告](v77/REAL_INSTANCE_MASK_R53.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r52 | R001原生主干3次64步；best1全尺寸回退，矩形proxy/伪标签全部退役，仅保留失败诊断 | [报告](v77/SINGLE_CASE_FINETUNE_R52.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r51 | 52景72例官方/r46/r47对照；104新窗，AI六列抽帧评分，504视频解码；训练0 | [报告](v77/REAL_DELETE_NATIVE_R51.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r50 | 20景40个固定真实DELETE完成；2例SAM吞邻车拒绝；独立f05结构粗分类及5.6-sol四列小数评分，训练0 | [报告](v77/REAL_DELETE_STRUCTURE_R50.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r49 | 实例/位置软绑定；64步/20新窗；两阶段5.6 Sol直接看图未见稳定升级；保留r46，GPU已空 | [报告](v77/TARGET_PROTECTED_SPATIAL_BINDING_R49.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r48 | 同指令消融；128步/20新窗；指定5.6 Sol独立图像review，保留r46；人工待评 | [报告](v77/TARGET_PROTECTED_SHORT_CYCLE_R48.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r47 | RGB/BEV条件接口；320步、48新窗＋12基线；合成条件增量，真实固定帧局部改善但仍失败；待人工 | [报告](v77/TARGET_PROTECTED_MULTI_PRIOR_R47.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r46 | SDK时间修复；160步、25新窗＋11复用，GPU已停 | [报告](v77/TARGET_PROTECTED_ONUQ_TIME_FIX_R46.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r45 | 160步、32窗；时间语义bug中止，保留旧结果 | [报告](v77/TARGET_PROTECTED_ONUQ_ADAPTER_R45.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r44 | 显式主B必要关系前置；4固定失败候选提前拦3，Q060保留；4测试通过，新数据／模型收益0 | [记录](autoresearch/worldsim_v77/target_protected_20260929/r44/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r43 | 取消固定16m距离代理后6个新scene满足元数据过程；保留全部实际空间门槛并前置到SAM之前；有界来源对照已完成；4候选独立QA为0/0/1/0，均未准入；新训练0 | [记录](autoresearch/worldsim_v77/target_protected_20260929/r43/progress.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r42 | 取消每scene前三截断后440→612窗口；原440及r41过滤逐条重现，但固定过程筛选仍0来源；停止将截断作为主因；新训练0 | [记录](autoresearch/worldsim_v77/target_protected_20260929/r42/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r41 | 旧440候选池中静止清楚B的7条可解析窗口，相机三秒行程均不足1m；固定视差过程筛选0来源，不提RGB/不放宽阈值；新增训练0 | [收口/状态](autoresearch/worldsim_v77/target_protected_20260929/r41/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r40 | 完成固定23例提示帧对照：{"nonvehicle_or_incomplete_protection": 8, "instance_identity_or_visibility_uncertain": 6, "insufficient_actual_occlusion": 1, "insufficient_other_frame_evidence": 2, "static_foreground": 1, "ordinary_background": 3, "actual_size": 2}；新增训练0 | [收口/状态](autoresearch/worldsim_v77/target_protected_20260929/r40/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r39 | 真实过去2秒位姿作A的20次候选尝试，2条空间可行，分别缺完整几何/其他帧证据，0准入；新增训练0 | [收口/状态](autoresearch/worldsim_v77/target_protected_20260929/r39/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r38 | 21空间轨迹/6scene，补齐39实例评价标签后仅3条同scene普通背景技术候选；冻结代表S016独立QA0，未准入；新增训练0 | [收口/状态](autoresearch/worldsim_v77/target_protected_20260929/r38/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r37 | 新18scene预登记后15scene/450帧可用；有序抽样与多地图分派修复，17条SAM轨迹15条技术通过；新增训练0 | [收口/状态](autoresearch/worldsim_v77/target_protected_20260929/r37/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r36 | 合法轮廓反证数值改善，但Q046仍QA1且损失正确投影；不推广、不调网格；训练0 | [收口](autoresearch/worldsim_v77/target_protected_20260929/r36/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r35 | 150投影拒绝中0例仅由水平截边导致；不放宽ego/尺度约束；训练0 | [收口](autoresearch/worldsim_v77/target_protected_20260929/r35/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r34 | Q046主B身份可用、邻车边缘溢出，条件1；修复审核图纵横比；训练0 | [收口](autoresearch/worldsim_v77/target_protected_20260929/r34/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r33 | 372固定静止A解只得同scene P001，独立QA0，停止同批位置搜索；训练0 | [收口](autoresearch/worldsim_v77/target_protected_20260929/r33/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r32 | 同87轨迹技术1→3；Q046输入QA2，Q035因小而模糊QA0；旧Q060保留；训练0 | [收口](autoresearch/worldsim_v77/target_protected_20260929/r32/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r31 | 同87候选拆分主显露B与轻触邻车，定位旧门槛语义错误；训练0 | [收口](autoresearch/worldsim_v77/target_protected_20260929/r31/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r30 | 同87轨迹20帧质量控制，仅增1未准入候选，不全局推广 | [收口](autoresearch/worldsim_v77/target_protected_20260929/r30/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r29 | Q060最终H遮后合法状态，30帧独立条件QA2；训练0步 | [收口](autoresearch/worldsim_v77/target_protected_20260929/r29/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r28 | 固定87候选补实例标签，1条输入QA2；修复首因与位姿元数据 | [收口](autoresearch/worldsim_v77/target_protected_20260929/r28/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r27 | 43条车道长度拒绝的拓扑/观测地面对照；实际轮廓仍遇未核验实例，训练0 | [报告](v77/TARGET_PROTECTED_VISIBLE_INPUT_R23_R27.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r26 | 8真实DEV×2臂；可见洞裁减导致A048/A034/A061退化，不推广，48视频验证 | [报告](v77/TARGET_PROTECTED_VISIBLE_INPUT_R23_R27.md) · [固定帧观察](autoresearch/worldsim_v77/target_protected_20260929/r26/assistant_output_review.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r25 | 3例输入90帧合同与独立AI2；不等于下游通过，12视频验证 | [报告](v77/TARGET_PROTECTED_VISIBLE_INPUT_R23_R27.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r24 | 空间Adapter30帧零等价/梯度；稀疏证据缩放丢失修复，训练0 | [报告](v77/TARGET_PROTECTED_VISIBLE_INPUT_R23_R27.md) · [探针](autoresearch/worldsim_v77/target_protected_20260929/r24/resampling_fixed/result.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r23 | 固定61scene/1830帧只1普通背景候选，reveal0；50条配方未形成，训练0 | [报告](v77/TARGET_PROTECTED_VISIBLE_INPUT_R23_R27.md) · [分母](autoresearch/worldsim_v77/target_protected_20260929/r23/factory_summary.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r22 | 遮后输入的actor-state合法性／可用性探针；不训练、不启surfel | [预案](autoresearch/worldsim_v77/target_protected_20260929/r22/plan.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r21 | 入口修复、去重与单次20/30帧工程探针；简单actor-state先于surfel | [预案](autoresearch/worldsim_v77/target_protected_20260929/r21/plan.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r20 | 两例四窗零训练舍入对照；未复现P019新增车，停止该工程诊断 | [诊断与证据](v77/TARGET_PROTECTED_ENGINEERING_AUDIT_R19.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r19 | 610帧合同；GT硬裁漏目标、初始化舍入、4train＋1val重复；CPU修复与人工记录 | [诊断与证据](v77/TARGET_PROTECTED_ENGINEERING_AUDIT_R19.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r18 | 160步完成／100步保护项，用户要求0新评价窗时停止；权重保留、效果未知 | [诊断与证据](v77/TARGET_PROTECTED_ENGINEERING_AUDIT_R19.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r17 | 训练未传B，87步工程中止；非有效三项损失对照，全部保留 | [纠正](autoresearch/worldsim_v77/target_protected_20260929/r17/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r16 | 40来源1200曝光；10独立AI2三秒候选、40视频交付；val扫过0、训练0 | [报告](v77/TARGET_PROTECTED_LONG_DATA_R16.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r15 | 相对合法路径，12独立AI2；6train扫过/4world、val扫过0，训练0 | [报告](v77/TARGET_PROTECTED_PATH_PROCESS_R15.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r14 | 同80张量/160步时间更新；新GT改善，真实8DEV无稳定收益，228视频验收 | [报告](v77/TARGET_PROTECTED_TEMPORAL_SCOPE_R14.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r13 | 同r7固定CFG1：10窗完成，P006额外生车/误差增加，真实无跨例收益；60视频交付，不推广 | [报告](v77/TARGET_PROTECTED_SAMPLING_R13.md) · [收口](autoresearch/worldsim_v77/target_protected_20260929/r13/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r12 | 同世界真实轨迹平移32来源0候选，raw地面修复未改善产量，训练0并停止该搜索 | [记录](autoresearch/worldsim_v77/target_protected_20260929/r12/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r11 | 新过程来源32世界，端点拟合0候选；保留拒绝、训练0，停止中点放置搜索 | [零产出](autoresearch/worldsim_v77/target_protected_20260929/r11/closeout.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r10 | 50train/25world，4扫过/3world，同80张量160步；19例四臂76窗与278视频交付；有限过程结果不推广 | [报告](v77/TARGET_PROTECTED_TEMPORAL_CONTROL_R10.md) · [结果](autoresearch/worldsim_v77/target_protected_20260929/r10/results_summary.json) · [审核](autoresearch/worldsim_v77/target_protected_20260929/r10/review_link.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r9 | 实测r7/r8扫过0；22候选18通过；覆盖缺额按预案训练0步，保留全部反例 | [计划与图](autoresearch/worldsim_v77/target_protected_20260929/r9/plan.md) · [独立QA](autoresearch/worldsim_v77/target_protected_20260929/r9/independent_data_reviews.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r8 | 50train/25scene/max3；同80张量160步，7合成+8真实三臂45窗完成；对r7洞MAE−1.05%、保护车+4.78%，真实固定帧未见明确迁移收益；不推广 | [报告](v77/TARGET_PROTECTED_DATA_CONTROL_R8.md) · [结果](autoresearch/worldsim_v77/target_protected_20260929/r8/results_summary.json) · [审核](autoresearch/worldsim_v77/target_protected_20260929/r8/review_link.md) |
| WS-V77-TARGET-PROTECTED-20260929 / r7 | 查r6训练缺106目标encoder；严格恢复，同80张量160步修复；数据/模块科学结论重新建立 | [报告](v77/TARGET_PROTECTED_DIAGNOSIS_R7.md) · [证据](autoresearch/worldsim_v77/target_protected_20260929/r7/diagnosis_summary.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r6 | 52技术AI2例/48train+4val；原结构80张量160步；4留出保护车MAE均变差，原模型保留；8有效任务+1空窗 | [报告](v77/TARGET_PROTECTED_FINETUNE_R6.md) · [证据](autoresearch/worldsim_v77/target_protected_20260929/r6/pilot_summary.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r5 | CPU连续规划4scene/40帧/16视频；精确准入4例waiting，推理/训练0 | [报告](v77/TARGET_PROTECTED_CPU_R5.md) · [证据](autoresearch/worldsim_v77/target_protected_20260929/r5/summary.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r4 | CPU固定10帧29/32；几何预案4例；SAM2队列4实例，未推理/训练 | [报告](v77/TARGET_PROTECTED_CPU_R4.md) · [证据](autoresearch/worldsim_v77/target_protected_20260929/r4/summary.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r3 | 连续来源3段；新合成24例/5场景；独立{'pass': 24}；训练0 | [GPU造数报告](v77/TARGET_PROTECTED_GPU_R3.md) · [证据](autoresearch/worldsim_v77/target_protected_20260929/r3/summary.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r2 | 37例训练输入重评{'train_usable_pending_human': 27, 'uncertain': 9, 'reject': 1}；630帧RGB泄漏0；2实例4视图待GPU；训练0 | [CPU合同与造数入口](v77/TARGET_PROTECTED_CPU_R2.md) · [证据](autoresearch/worldsim_v77/target_protected_20260929/r2/summary.json) |
| WS-V77-TARGET-PROTECTED-20260929 / r1 | 49合成case/11个receiver scene；独立{'reject': 7, 'pass': 28, 'uncertain': 14}，仅通过例人工逐帧全检；密集类缺额、零训练 | [GPU试产](v77/TARGET_PROTECTED_GPU.md) · [CPU来源](v77/TARGET_PROTECTED_CPU.md) · [质量协议v2](v77/TARGET_PROTECTED_QUALITY.md) |
| WS-V77-DELETE-AUDIT-20260928 / r1 | 45 val scene/70单车DELETE，280视频已交付；用户CSV已归档，human_score 0/1/2/未填=26/26/15/3；原助手抽帧观察保留 | [用户视频评分](v77/DELETE_AUDIT_USER_REVIEW.md) · [最终工程结果](v77/DELETE_AUDIT_RESULTS.md) |
| WS-V77-NINE-FULL-20260928 / r1 | 九例统一r30迁移配置完整DELETE完成，待用户视频审核；无训练/逐例修补 | [报告](v77/NINE_FULL_RESULTS.md) · [证据](autoresearch/worldsim_v77/nine_full_20260928/review_link.md) |
| WS-V77-EXPAND-EDIT-20260928 / r1–r7 | 固定新增六例，五例30帧生成/一例输入停止；r30回接Ω、INSERT、原位/接地控制，MOVE未放行；78监督候选、零训练 | [报告与组件图](v77/EXPANSION_EDIT_CHECKPOINT.md)、[登记/证据](autoresearch/worldsim_v77/expand_edit_20260928/review_link.md) |
| WS-V77-HYBRID-BG-20260927 / r36–r41 | 实测道路算子与后向证据控制；固定邻车修复车型、PBR失败停止，保留r18/r21 | [资产与证据报告](v77/HYBRID_RETAINED_ASSET.md)、[登记/观察](autoresearch/worldsim_v77/hybrid_bg_20260927/retained_asset/review_link.md) |
| WS-V77-HYBRID-BG-20260927 / r30–r35 | 首窗范围/羽化/邻车保护改善，长时序与来源几何反例；实际28帧 | [时序报告](v77/HYBRID_THIRD_TEMPORAL.md)、[证据](autoresearch/worldsim_v77/hybrid_bg_20260927/temporal_review/review_link.md) |
| WS-V77-HYBRID-BG-20260927 / r25–r29 | 第三例矩形条件改善；路面支撑与Poisson写回反例；保留r18/r28 | [第三例报告](v77/HYBRID_THIRD_SCENE.md)、[实际证据](autoresearch/worldsim_v77/hybrid_bg_20260927/third_scene/review_link.md) |
| WS-V77-HYBRID-BG-20260927 / r23–r24 | 可见对象2倍尺度与实际DELETE逐帧外观条件；保留r18，明确第三例协议 | [外观控制报告](v77/HYBRID_APPEARANCE_CONTROLS.md)、[实际输入与结果](autoresearch/worldsim_v77/hybrid_bg_20260927/appearance_controls/review_link.md) |
| WS-V77-HYBRID-BG-20260927 / r19–r22 | 保留用户优选r18；真实围栏来源失败控制、局部alpha写回与后续两个时间窗 | [修复报告](v77/HYBRID_FENCE_WRITEBACK.md)、[用户评价/证据](autoresearch/worldsim_v77/hybrid_bg_20260927/fence_writeback/review_link.md) |
| WS-V77-HYBRID-BG-20260927 / r14–r18 | 邻车真实证据正控制、首帧条件拆解、SV3D中间图与对应视角锚点修复 | [条件报告](v77/HYBRID_NEIGHBOR_CONDITIONS.md)、[实际来源/重放验证](autoresearch/worldsim_v77/hybrid_bg_20260927/neighbor_conditions/evidence_validation.json) |
| WS-V77-HYBRID-BG-20260927 / r10–r13 | 纠正隐藏邻车判据；首帧matting失败；局部深度改善但真实支持不足 | [身份/证据报告](v77/HYBRID_IDENTITY_EVIDENCE.md)、[来源验证](autoresearch/worldsim_v77/hybrid_bg_20260927/identity_audit/evidence_validation.json) |
| WS-V77-HYBRID-BG-20260927 / r3–r9 | 分段捕获与上下文/写回控制；再生车减少，结构与前景残留未通过；持续目标未完成 | [分阶段报告](v77/HYBRID_STAGE_AUDIT.md)、[证据](autoresearch/worldsim_v77/hybrid_bg_20260927/stage_audit/summary.json) |
| WS-V77-HYBRID-BG-20260927 / r2 | 两scene三mask + factual evidence + DiffuEraser各30帧；再生车/残影失败，停止候选、不进入Ω；原默认保留 | [实测与architecture](v77/HYBRID_BACKGROUND_RESULTS.md)、[收口](autoresearch/worldsim_v77/hybrid_bg_20260927/r2/closeout.json) |
| WS-V77-HYBRID-BG-20260927 / r1 | DiffuEraser最小权重、独立环境与三mask CPU合同准备；零模型前向，等待用户GPU；两开发scene | [方案与architecture](v77/HYBRID_BACKGROUND_PREP.md)、[登记](autoresearch/worldsim_v77/hybrid_bg_20260927/registration.json) |
| WS-V77-DELETE-FULL-20260927 / r1 | 完整5s/10s六相机DELETE：24补景窗、150次Ω、181原GLB层；后向再生成车形、长序列漂移、错误遮挡独立留证；22视频，人工null | [协议与architecture](v77/DELETE_FULL.md)、[登记](autoresearch/worldsim_v77/delete_full_20260927/) |
| WS-V77-DRIVEEDITOR-COMPARE-20260927 / r1,r2 | 官方训练后deletion零训练，3次/30帧；0230扩大mask失败、唯一范围控制去掉主要车形残影；0255直接改善；原生/合成/失败/标注11视频，人工null | [实测与architecture](v77/DRIVEEDITOR_RESULTS.md)、[登记](v77/DRIVEEDITOR_RUN.md)、[证据](autoresearch/worldsim_v77/driveeditor_run_20260927/)、[V77-F02](research_failures/entries/V77-F02.md) |
| WS-V77-DRIVEEDITOR-ASSESS-20260926 / r1 | 用户指定DriveEditor补景评估；官方deletion/blank数据分支核对、20帧CPU mask预检、checkpoint缺失确认；无生成/训练，优先现成权重再决定数据适配 | [评估与architecture](v77/DRIVEEDITOR_ASSESSMENT.md)、[预检证据](autoresearch/worldsim_v77/driveeditor_assess_20260926/) |
| WS-V77-PIPELINE-R3-20260926 / r1 | 131/181帧×6相机背景观测检查；0230悬浮GT框产生168个假可见核心查询，贴地后0；两例联合背景候选均0，停止伪观测复制；零模型前向、12测试、人工null | [报告与architecture](v77/PIPELINE_R3.md)、[run证据](autoresearch/worldsim_v77/pipeline_r3_20260926/)、[V77-F02](research_failures/entries/V77-F02.md) |
| WS-V77-PIPELINE-R2-20260926 / r1 | 静态LiDAR补充命令拒绝：0255旧2m及前移2m均100/100帧有静态占据；两路原RGB/mask冻结，196帧ProPainter长上下文仍留主要残影；零训练，六段完整原时间窗视频，人工null | [报告与architecture](v77/PIPELINE_R2.md)、[run证据](autoresearch/worldsim_v77/pipeline_r2_20260926/)、[V77-F02](research_failures/entries/V77-F02.md) |
| WS-V77-ACTOR-COMMAND-AUDIT-20260926 / r1 | 两目标完整50/100帧指令扫掠检查；0230碰邻车且车头反向，0255最小8.4cm、投影/缩放纠错；撤回资产失败/路线关闭归因，零新模型前向，人工null | [审计与architecture](v77/ACTOR_COMMAND_AUDIT.md)、[新run证据](autoresearch/worldsim_v77/actor_command_audit_20260926/)、[V77-F02更正](research_failures/entries/V77-F02.md) |
| WS-V77-EXPLICIT-POC-20260926 / r1 | 两开发场景 SAM2/ProPainter/冻结Ω背景/官方 Hunyuan3D-2.1 单图GLB；各10时刻×6相机、20次新增Ω前向、零训练；背景残影独立保留；旧资产失败/关闭归因经后续指令与放置审计撤回，人工null | [报告与architecture](v77/EXPLICIT_POC.md)、[run轻量证据](autoresearch/worldsim_v77/explicit_poc_20260926/)、[V77-F02](research_failures/entries/V77-F02.md) |
| WS-V77-VIDEO-REVIEW-20260926 / r1 | 用户要求人眼时序复核：三个完整连续场景191/196/196帧，六相机，原始/factual/固定2m平移九视频；新增574冻结前向、复用9时刻，零训练；人工判定空白 | [视频合同与architecture](v77/VIDEO_REVIEW.md)、[逐帧轻量证据](autoresearch/worldsim_v77/video_review_20260926/) |
| WS-V77-P0-24ACTOR-20260926 / r1 | 3场景24固定对象，冻结Ω六相机；预测相机与GT标定/背景尺度两读出，解析编辑672次检查、630次非空；完整评价采用纠错后的evaluation_v2 | [结果与architecture](v77/P0_RESULTS.md)、[登记/指标/来源](autoresearch/worldsim_v77/p0_20260926/)、[V77-F01](research_failures/entries/V77-F01.md) |
| WS-V77-P0-TEMPORAL-20260926 / r1 | 原24对象的0/20/40帧GT规范坐标并集，新增6次冻结前向，336次非空编辑；召回随并集不下降，缺面/重影仍在；后验开发控制 | [三时刻控制与边界](v77/P0_RESULTS.md)、[完整逐对象指标](autoresearch/worldsim_v77/p0_20260926/metrics_temporal.json) |
| WS-V77-BOOTSTRAP-20260926 | 从V7.6收口建立v77；VGGT系列重建基座、冻结Ω零训练结构化编辑协议及权重准备，P0实验未运行 | [P0协议与architecture](v77/P0_PROTOCOL.md)、[checkpoint来源与检查](v77/checkpoint_manifest.json) |
| WS-V76-CLOSEOUT-20260926 | 用户关闭V7.6：当前HUGSIM/VAD-GS对象资产不足以支撑高保真编辑；保留baseline，后续重建基座使用VGGT系列 | [收口与architecture](v76/CLOSEOUT.md)、[V76-F03](research_failures/entries/V76-F03.md) |
| VADGS-P0R1-000-TEST-30000 / 20260926 | 30k官方75视图25.2378/0.8028/0.1717；Camera5共61视图单列；横移actor变换断言失败，未完成且不再续跑 | [最终记录](v76/CLOSEOUT.md)、[原始指标与状态副本](autoresearch/worldsim_v76/closeout_20260926/) |
| VADGS-P0R1-000-TEST-16000 / 20260926 | 同一75官方test：22.8532 / 0.7496 / 0.2303；保存张量与75张原始RGB/depth/acc均有限，继续30k | [执行报告](v76/P0R1_EXECUTION.md)、[指标/图像/有限值审核](autoresearch/worldsim_v76/p0r1_exhaustive/official_test_16k/) |
| VADGS-BACKGROUND-COMPLETE-20260926 | 两场景各366背景SAM完成，732张逐一解码/编码与文件名核验，CPU进程退出；动态身份仍未准入 | [输入报告](v76/MATCHED_SCENE_PRIORS.md)、[文件清单/日志](autoresearch/worldsim_v76/matched-scene-priors/background-completion/) |
| VADGS-P0R1-000-TEST-8000 / 20260926 | 同一75官方test：22.7086/0.7520/0.2348，较4k三项改善；固定五相机图和9k日志风险留证，继续30k | [报告与architecture](v76/P0R1_EXECUTION.md)、[指标/图像/审核](autoresearch/worldsim_v76/p0r1_exhaustive/official_test_8k/) |
| VADGS-P0R1-000-GATE-4000 / 20260926 | 4k新权重75官方test：21.7392/0.7300/0.2727；六横移×32actor世界变换零变化；近树遮挡方向对照，工程通过但非30k结论 | [报告与architecture](v76/P0R1_EXECUTION.md)、[指标/图像/几何](autoresearch/worldsim_v76/p0r1_exhaustive/engineering_4k/) |
| VADGS-VISIBLE-SAM-CROSSFRAME-20260926 | 固定36视图47投影对象关联28个；可见公交车因truck检测漏标，唯一fallback扩展按停止规则收口，完整动态输入未准入 | [适配边界](v76/MATCHED_SCENE_PRIORS.md)、[全部结果与审核](autoresearch/worldsim_v76/matched-scene-priors/crossframe-gate/)、[V76-F02](research_failures/entries/V76-F02.md) |
| VADGS-P0R1-000-INITIALIZATION-CONTROL；VISIBLE-DETECTION-GATE-20260926 | 穷举完成并开始全新训练；15,475个COLMAP点的55,557条保存可见性边全部正确；同场景可见检测→SAM固定6正/4遮挡控制通过，未准入完整场景训练 | [初始化核验](v76/P0R1_EXECUTION.md)、[可见身份控制](v76/MATCHED_SCENE_PRIORS.md) |
| VADGS-MATCHED-PRIORS-20260926；SAM-OCCLUSION-20260926 | 两场景各366深度/法线，官方六视图验证编码；SAM固定12视图发现V76-F02遮挡身份错误，动态标签保留并退出训练入口 | [数据合同与architecture](v76/MATCHED_SCENE_PRIORS.md)、[生成代码与门禁证据](autoresearch/worldsim_v76/matched-scene-priors/)、[V76-F02](research_failures/entries/V76-F02.md) |
| VADGS-P0R1-000 / 20260926 | 修复track映射，独立exhaustive匹配/新初始化/从零30k；官方test与Camera5外推分开，有限自动队列和最终收口关机授权 | [执行记录与architecture](v76/P0R1_EXECUTION.md)、[配置与控制器](autoresearch/worldsim_v76/p0r1_exhaustive/) |
| VADGS-P0-000-COLMAP-AUDIT；TEST-16000 / 20260926 | 两项复现诊断：COLMAP <0.6px剩33,930/138,944点；官方时间test75视图22.7010/0.7486/0.2313；发现并修复V76-F01图像ID错绑，旧run停止并保留 | [复现审计与architecture](v76/P0_REPRODUCTION_AUDIT.md)、[证据与修复](autoresearch/worldsim_v76/reproduction-audit/)、[V76-F01](research_failures/entries/V76-F01.md) |
| VADGS-P0-000 / 早期诊断 | 官方nuScenes 000，VAD-GS从零训练；4k/8k几何及相机5外推；相机0的61帧是混合训练/留出对照。旧初始化后证实有工程错误，不作方法判决 | [V7.6工程记录与architecture](v76/P0_ENGINEERING.md)、[4k/8k数据](autoresearch/worldsim_v76/p0-8k-evidence.json) |
| WS-V75-OMNI-REVIEW-02 / 20260923-proposal-r9-all-approved | 24个10秒case获人工批准，6个ego、18个对象；获批清单与原始参考保留。r9批准快照不是新的OmniDreams生成成绩 | [V7.5 r9收尾与architecture](autoresearch/worldsim_v75/downstream_bench/r9-closeout/README.md)、[机器可读索引](autoresearch/worldsim_v75/downstream_bench/r9-closeout/summary.json) |
| WS-V75-FIVE-BASELINES-01 / 20260923-r1，HUGSIM r9 | 7场景ground/scene各30k与导出；24/24 case各100帧事实/反事实和自动指标；10Hz适配与两个场景质量风险单列；训练checkpoint后已退役 | [收尾数量、证据与恢复边界](autoresearch/worldsim_v75/downstream_bench/r9-closeout/README.md) |
| WS-V75-FIVE-BASELINES-01 / 20260923-r1，DriveEditor r9 | 18对象case原生10帧事实/反事实；11个完成10秒迭代反事实；6个ego接口不支持、7个对象迭代未完成，不合并分母 | [收尾](autoresearch/worldsim_v75/downstream_bench/r9-closeout/README.md)、[V75-F02](research_failures/entries/V75-F02.md) |
| V7.5 r9 证据保留 / 20260925 | 保留119个MP4、逐case指标、配置和两份人工评分工作簿，119/119哈希核验；旧模型、七场景checkpoint及中间输入退役，可用空间约141→378GiB | [保留与重建限制](autoresearch/worldsim_v75/downstream_bench/r9-closeout/README.md) |
| WS-V75-OMNI-REVIEW-02 / 20260923-proposal-r1至r7-insertion-review | 历史待审提案：r7替换INSERT-02/04/05，原始参考有视频，当时生成与评分留空。后续以r9获批清单为准，不将旧待审状态当现状 | [r7历史提案](autoresearch/worldsim_v75/downstream_bench/review-20260923/insertion-review-r7.md)、[r6历史提案](autoresearch/worldsim_v75/downstream_bench/review-20260923/human-feedback-r6.md) |
| WS-V75-DOWNSTREAM-FULL-01 / 20260923-r1 | 我们固定24case：OmniDreams 24对/48段3696帧，原始参考24段、自动读出与固定五帧AI初评；ReSim单卡chunk17+offload真实ego队列；三个pilot高斯重建输入准备。静帧诊断非闭环评测，人工null、六维无总分 | [记录与architecture](autoresearch/worldsim_v75/downstream_bench/full-20260923/README.md)、[轻量证据](autoresearch/worldsim_v75/downstream_bench/full-20260923/summary.json) |
| WS-V75-DOWNSTREAM-GPU-SMOKE-01 / 20260923-r1 | 单 RTX3090：OmniDreams ego 减速2×61帧，GaussianDWM QA兼容版1条合成回答，HUGSIM移除2×9帧×6相机；ReSim VAE OOM停止，StreetGS checkpoint缺失；工程 smoke，正式bench 0 case | [报告与architecture](autoresearch/worldsim_v75/downstream_bench/gpu-smoke-20260923/README.md)、[日志/结果](autoresearch/worldsim_v75/downstream_bench/gpu-smoke-20260923/summary.json) |
| WS-V75-DOWNSTREAM-CFBENCH-PREP-02 / 20260923-r1 | GPU-free 全量准备：24/24 CPU geometry pass、24张人工审阅图；ReSim/DriveEditor/GaussianDWM/HUGSIM 输入与公开资产物化；GaussianDWM loader 合同问题、DriveEditor reference mask 风险及 HUGSIM release 路径问题；10项本地测试与55项上游测试通过；0模型前向 | [设计、发现与边界](v75/DOWNSTREAM_COUNTERFACTUAL_BENCH.md)、[manifest/qualification/assets/env/preflight/plan](autoresearch/worldsim_v75/downstream_bench/) |
| WS-V75-DOWNSTREAM-CFBENCH-PREP-01 / 20260922-r1 | 下游导向 paired-edit pilot：四类各6个共24个候选；六模型能力/资源 registry、CPU preflight、fail-closed planner、六维无总分 evaluator；0模型调用 | [设计与边界](v75/DOWNSTREAM_COUNTERFACTUAL_BENCH.md)、[manifest/preflight/plan](autoresearch/worldsim_v75/downstream_bench/) |
| WS-STORAGE-OLD-RUNS-RETIRE-01 / 20260922-r1 | 历史研究脉络与产物退役；保留轻量证据、当前 V7.5 依赖核验；0 模型调用 | [清单与恢复边界](autoresearch/old_runs_retirement_20260922/README.md) |
| WS-V75-CF-DEFINITION-01 / 20260921-r1 | WorldSim反事实质量定义、参考层级与有限实验设计；0模型调用，非新科学结果 | [定义](v75/PROBLEM.md)、[评价协议](v75/COUNTERFACTUAL_EVALUATION.md)、[审阅图](autoresearch/worldsim_v75/counterfactual_definition/) |
| WS-V75-ACTOR-REMOVAL-QUALIFY-01；GENERATION-01 / 20260921-r2；r1 | 两个曝光开发源中冻结首个A遮挡B对象对；三状态×未编辑/移除共6段702帧；reference通过，类别先验残留候选 | [对象移除报告](v75/ACTOR_REMOVAL_COUNTERFACTUAL.md)、[轻量证据](autoresearch/worldsim_v75/actor_removal_counterfactual/summary.json) |
| WS-V75-ACTOR-REMOVAL-CONFIRM-SOURCES-01 / 20260921-r1 | 固定8个未曝光日志、每个一个+6.5s窗口；第5个首次通过后停止；24次CPU检测、0重建/生成 | [对象移除报告](v75/ACTOR_REMOVAL_COUNTERFACTUAL.md)、[来源结果](autoresearch/worldsim_v75/actor_removal_counterfactual/confirmation-source-result.json) |
| WS-V75-ACTOR-REMOVAL-CONFIRM-QUALIFY-01 / 20260921-r1,r2,r3 | r1/r2为raster前环境失败、0生成；r3复用同一单次DVGT，三状态raster通过，科学输入不变 | [工程与资格边界](v75/ACTOR_REMOVAL_COUNTERFACTUAL.md)、[资格摘要](autoresearch/worldsim_v75/actor_removal_counterfactual/confirmation-qualification-summary.json) |
| WS-V75-ACTOR-REMOVAL-CONFIRM-GENERATION-01 / 20260921-r1 | 独立源三状态×两编辑6段702帧；reference/DVGT/class均A=10/10、B=10/10，reference gate失败，禁止重建排名 | [对象移除报告](v75/ACTOR_REMOVAL_COUNTERFACTUAL.md)、[评价](autoresearch/worldsim_v75/actor_removal_counterfactual/confirmation-evaluation.json) |
| WS-V75-ACTOR-REMOVAL-ONSET-01 / 20260921-r1 | A从f=0状态缺失的单段117帧机制诊断；A仍10/10，支持initial-image anchor；无闭环 | [对象移除报告](v75/ACTOR_REMOVAL_COUNTERFACTUAL.md)、[机制评价](autoresearch/worldsim_v75/actor_removal_counterfactual/onset-evaluation.json) |
| WS-V75-ACTOR-REMOVAL-G1-SCREEN-01 / 20260921-r1 | 下一批8个未曝光日志的中远距/小占比输入窗口；第5个首次通过后停止，8次CPU检测、0重建/生成 | [显著性候选](v75/ACTOR_REMOVAL_COUNTERFACTUAL.md)、[来源摘要](autoresearch/worldsim_v75/actor_removal_counterfactual/g1-salience-screen.json) |
| WS-V75-ACTOR-REMOVAL-G1-QUALIFY-01；G1-GENERATION-01 / 20260921-r1 | reference raster通过；仅未编辑/移除2段234帧，A=10/10→7/10仍未通过G1，DVGT未准入 | [门控结果](v75/ACTOR_REMOVAL_COUNTERFACTUAL.md)、[评价](autoresearch/worldsim_v75/actor_removal_counterfactual/g1-reference-evaluation.json) |
| WS-V75-ACTOR-REMOVAL-G1-FAR-SCREEN-01 / 20260921-r1 | 下一批8个未曝光日志的远距小投影窗口；0个合格A→B对、0 detector/重建/生成，门槛未放宽 | [对象移除收口](v75/ACTOR_REMOVAL_COUNTERFACTUAL.md)、[空窗口](autoresearch/worldsim_v75/actor_removal_counterfactual/g1-far-screen.json) |
| WS-V75-ACTOR-SLOWDOWN-G1-QUALIFY-01；G1-GENERATION-01 / 20260921-r1 | OmniDreams中距来源未来0.5×减速；前5帧输入严格一致，reference-state未编辑/减速各1段，后段5/5服从编辑，G1通过 | [轨迹报告](v75/ACTOR_TRAJECTORY_COUNTERFACTUAL.md)、[主图](autoresearch/worldsim_v75/actor_trajectory_counterfactual/main-state-error-figure.png) |
| WS-V75-ACTOR-SLOWDOWN-STATE-QUALIFY-01 / 20260921-r1,r2；STATE-GENERATION-01 / r1 | r1在状态误差样本读出前缺iopath结束；r2同输入完成1次官方DVGT读出及三状态资格；同一个OmniDreams新增depth-shift/depth+shape未编辑与减速4段468帧，13.27m输入误差对应55.61/32.08px生成响应误差 | [完整结果](v75/ACTOR_TRAJECTORY_COUNTERFACTUAL.md)、[轻量摘要](autoresearch/worldsim_v75/actor_trajectory_counterfactual/summary.json) |
| WS-V75-ACTOR-SLOWDOWN-CONTROL-QUALIFY-01；CONTROL-G1-01；CONTROL-STATE-01 / 20260921-r1 | 复用独立小误差源与既有未编辑序列；同一个OmniDreams新增reference/depth-shift/depth+shape减速3段351帧，0.380m输入误差下生成响应未退化 | [正反对照](v75/ACTOR_TRAJECTORY_COUNTERFACTUAL.md)、[响应图](autoresearch/worldsim_v75/actor_trajectory_counterfactual/response-error-contrast.png) |
| WS-V75-ACTOR-SLOWDOWN-CONFIRM-SCREEN-01 / 20260921-r1,r2；CONFIRM-QUALIFY-01 / r1 | r1为投影/场景字段工程错误、0检测；r2同一8日志在第6个选首个目标、1次CPU检测；reference raster后段变化139/64/0/0/0，G0失败，0重建/生成且不替换来源 | [停止结论](v75/ACTOR_TRAJECTORY_COUNTERFACTUAL.md)、[来源结果](autoresearch/worldsim_v75/actor_trajectory_counterfactual/confirmation-screen.json) |
| WS-V75-VELOCITY-PRIOR-CONTROL-01 / 20260920-r1 | 两日志20流CPU回放、世界速度零均值普通控制；主项退化、未准入生成 | [收口与全部证据](autoresearch/worldsim_v75/velocity_prior_control/README.md) |
| WS-V75-QUALIFY-01 / 20260920-r1 | 官方接口契约与资源来源核对 | [问题协议](v75/PROBLEM.md)、[V75-F01](research_failures/entries/V75-F01.md) |
| WS-V75-PREFLIGHT-01 / 20260920-r1 | 环境、真实初帧与条件、输入编码和权重预检 | [运行准备](v75/PREFLIGHT.md)、[证据](autoresearch/worldsim_v75/preflight) |
| WS-V75-BASELINE-01 / 20260920-single3090-r1 | 单卡首段、完整clean及同seed完整重复；资源与视频核验 | [单卡基线](v75/BASELINE.md)、[证据](autoresearch/worldsim_v75/baseline) |
| WS-V75-LOCALIZE-01 / 20260920-r1 | 单目标±0.5m条件干预与恢复；两seed有限实验 | [定位实验](v75/LOCALIZATION.md) |
| WS-V75-COHORT-01；NATIVE-COHORT-01 / 20260920-r1 | 来源目录与访问核验、3开发+3保留冻结、三例原生clean | [原生样例基线](v75/NATIVE_COHORT.md) |
| WS-V75-AV2-BRIDGE-01 / 20260920-r1 | Argoverse显式三维条件适配、真实未来RGB与单卡clean | [输入桥接](v75/AV2_BRIDGE.md) |
| WS-V75-NATURAL-01；NATURAL-SOURCES-01 / 20260920-r1 | 旧目标参考排除；八日志冻结、四个清楚可见目标的DVGT距离读出与普通控制 | [自然状态读出](v75/NATURAL_STATE.md) |
| WS-V75-NATURAL-ROLLOUT-01 / 20260920-r1 | 一个自然残余发现候选的五组生成、真实视频检测与额外观测修复 | [生成比较](v75/NATURAL_STATE.md)、[证据](autoresearch/worldsim_v75/natural_state/) |
| WS-V75-NATURAL-ROLLOUT-01 / 20260920-seed43 | 同一自然候选、固定输入的四组随机性复核；三项对照与实际状态链条图 | [有限复核](v75/NATURAL_STATE.md)、[逐时刻结果](autoresearch/worldsim_v75/natural_state/replication/replication_result.json) |
| WS-V75-OBSERVATION-01 / 20260920-r1 | 已有视频的固定点观测、缺失支持与普通投影控制 | [观测边界](v75/NATURAL_STATE.md)、[结果](autoresearch/worldsim_v75/natural_state/observation/result.json) |
| WS-V75-CONFIRM-SOURCES-01 / 20260920-r1 | 四日志前瞻来源窗口；真实可观测性未通过，无模型推理 | [来源与排除](v75/NATURAL_STATE.md)、[记录](autoresearch/worldsim_v75/natural_state/observation/confirmation_visual_review.json) |
| WS-V75-VISIBLE-DEV-01 / 20260920-r1 | 四日志可见性前置、八候选两读出、完整排除与普通尺度控制 | [可见性窗口](v75/VISIBLE_COHORT.md) |
| WS-V75-VISIBLE-ROLLOUT-01 / 20260920-r1 | 一例四组生成、实际二维正反结果、条件刚体比较不合格 | [报告](v75/VISIBLE_COHORT.md)、[证据](autoresearch/worldsim_v75/visible_cohort/) |
| WS-V75-CLOSEDLOOP-CONTRACT-01 / 20260920-r1 | 真实场景39帧动作→相机→条件契约；无世界模型生成或策略结论 | [闭环接口](v75/CLOSED_LOOP.md) |
| WS-V75-FOLLOWING-BASELINE-01 / 20260920-r1；20260920-natural4 | 任务来源与真实RGB策略；两个通过、一个相机不可观测 | [反馈报告](v75/FOLLOWING_CLOSED_LOOP.md) |
| WS-V75-FOLLOWING-CLOSEDLOOP-01 / 20260920-r1 | 一任务四组真实生成反馈，468帧/60决策，普通尺度与参考修复 | [报告](v75/FOLLOWING_CLOSED_LOOP.md)、[证据](autoresearch/worldsim_v75/following/) |
| WS-V75-BRAKING-TASKS-01 / 20260920-r1 | 固定12日志48起点，9个记录减速窗口，原持续前车入口0合格 | [制动任务](v75/BRAKING_TASKS.md)、[分母](autoresearch/worldsim_v75/braking_tasks/task_sources.json) |
| WS-V75-BRAKING-INTERFACE-AUDIT-01 / 20260920-r1 | 一次事后真实RGB入口诊断，5次感知；距离/速度状态误差通过同一IDM抵消，无生成 | [报告](v75/BRAKING_TASKS.md)、[证据](autoresearch/worldsim_v75/braking_tasks/interface-audit/) |
| WS-V75-RASTER-POLICY-01；APPROACH-BASELINE-01 / 20260920-r1；20260920-association-r2 | 官方高程＋真实历史的普通基线；实际关联反例修复与保存输入回放 | [接近任务](v75/APPROACH_CLOSED_LOOP.md)、[工程证据](autoresearch/worldsim_v75/approach/) |
| WS-V75-APPROACH-CLOSEDLOOP-01；APPROACH-REFERENCE-01 / 20260920-r1；20260920-association-r2 | 5组585帧含原GT工程失败；最终四组完整反馈、一次DVGT与42点截止前参照，几何修正未稳定恢复动作 | [报告](v75/APPROACH_CLOSED_LOOP.md)、[四组结果](autoresearch/worldsim_v75/approach/comparison.json) |
| WS-V75-APPROACH-STATE-CONTROL-01 / 20260920-r1 | 四组CPU直接状态反馈；468帧旧轨迹精确复现、468帧新控制，几何响应与生成/策略接口边界 | [报告](v75/APPROACH_STATE_CONTROL.md)、[结果](autoresearch/worldsim_v75/approach_state_control/result.json) |
| WS-V75-IOU-ASSOCIATION-CONTROL-01 / 20260920-r1 | 原策略175次回放访问精确复现；固定IoU关联平均改善但额外制动恶化，未进入新生成 | [控制报告](v75/APPROACH_STATE_CONTROL.md)、[结果](autoresearch/worldsim_v75/approach_state_control/iou_control/result.json) |
| WS-V75-BRAKING-DEV2-01；APPROACH-BASELINE-02 / 20260920-r1 | 新有限6×3窗口18→2→1，真实35次检测与固定策略基线通过 | [任务发现](v75/BRAKING_TASKS.md)、[分母](autoresearch/worldsim_v75/approach_dev2/screen/result.json) |
| WS-V75-APPROACH-CLOSEDLOOP-02；APPROACH-RECONSTRUCTION-02；APPROACH-REFERENCE-02 / 20260920-r1 | 第二制动任务四组468帧反馈、1次DVGT与55点额外参照；保留自然误差影响小的好案例 | [接近任务](v75/APPROACH_CLOSED_LOOP.md)、[结果](autoresearch/worldsim_v75/approach_dev2/comparison.json) |
| WS-V75-APPROACH-STATE-CONTROL-02 / 20260920-r1 | 第二任务468帧旧动作精确回放与468帧直接状态控制；双任务统一对比图 | [接近任务](v75/APPROACH_CLOSED_LOOP.md)、[状态控制](autoresearch/worldsim_v75/approach_dev2/state_control/result.json) |
| WS-V75-TEMPORAL-STATE-AUDIT-01 / 20260920-r1 | 官方时间接口、两任务过去观测与静止/CV强控制；两组234帧精确复现、四组468帧新CPU反馈，无模型调用 | [时间状态范围](v75/APPROACH_CLOSED_LOOP.md#时间状态审计近静止任务的范围边界)、[证据](autoresearch/worldsim_v75/temporal_state_audit/result.json) |
| WS-V75-MOVING-FOLLOWING-SCREEN-01 / 20260920-r1 | 复用固定6日志18起点的真实运动跟车资格；18→15→2→1近静止→0，无模型调用，窗口关闭 | [任务覆盖边界](v75/BRAKING_TASKS.md#同一固定来源的真实运动跟车资格)、[完整分母](autoresearch/worldsim_v75/moving_following_screen/result.json) |
| WS-V75-SHAPE-PRIOR-AUDIT-01 / 20260920-r1 | 两任务五组普通形状/距离适配；1170帧CPU控制，目标GT形状移出拟合、额外LiDAR单列，0次新模型调用 | [形状输入边界](v75/APPROACH_CLOSED_LOOP.md#普通形状先验与生成闭环相近间距不保证相同制动)、[完整审计](autoresearch/worldsim_v75/shape_feedback/audit/result.json) |
| WS-V75-SHAPE-FEEDBACK-01 / 20260920-r1；20260920-conditioning-r2 | 首轮目录预检错误保留；两任务两普通形状导出共468帧实际反馈，936帧新旧精确回放；2.757/0.301m配对进度差、先验部分缓解非完整恢复 | [结果与真实对比图](v75/APPROACH_CLOSED_LOOP.md#四段实际生成反馈)、[结果](autoresearch/worldsim_v75/shape_feedback/comparison.json)、[完整来源](autoresearch/worldsim_v75/shape_feedback/provenance.json) |
| WS-V75-SHAPE-CONFIRM-01 / 20260920-r1 | 唯一额外seed43：两GT通过后四配对分支，702帧生成/精确回放；主候选2.757→−0.324m、三个判据反向而关闭；第二任务小效应由普通先验恢复 | [复核结论](v75/APPROACH_CLOSED_LOOP.md#唯一额外seed复核主候选反向小效应可由普通先验恢复)、[事前规则](autoresearch/worldsim_v75/shape_confirmation/protocol.json)、[完整比较](autoresearch/worldsim_v75/shape_confirmation/review/comparison.json) |
| WS-V75-DVGT-CONTRACT-01 / 20260920-r1 | 两个旧调用的CPU契约检查：官方像素、坐标代数、原生方向与已有PnP；0次模型调用 | [接口范围](v75/NATURAL_STATE.md#原生点图契约与有限时间上下文控制)、[结果](autoresearch/worldsim_v75/native_contract/single_frame_audit/result.json) |
| WS-V75-DVGT-TEMPORAL-CONTRACT-01 / 20260920-r1 | 两次固定3时刻2Hz前向；14视角13改善1退化，严重单帧方向错位大幅减轻，余差未通过原生使用筛查；0次生成 | [完整对比](v75/NATURAL_STATE.md#两次冻结的普通历史观测控制)、[事前协议](autoresearch/worldsim_v75/native_contract/protocol.json)、[结果](autoresearch/worldsim_v75/native_contract/result.json) |
| WS-V74-P0-* / 20260910 | 数据、资源与输入角色准备 | [P0](archive/2026-09/v74-0920/WORLDSIM_V74_P0_HANDOFF.md) |
| WS-V74-METHOD-TOURNAMENT-01 | H1 WEX / RIF / DCS 与强控制 | [结果](archive/2026-09/v74-0920/WORLDSIM_V7_4_RESULTS.md)、[失败](archive/2026-09/v74-0920/WORLDSIM_V7_4_FAILURES.md) |
| H2 CPU / GPU P1 | 数据与学习能力实验、实现关闭 | [CPU](archive/2026-09/v74-0920/WORLDSIM_V7_4_H2_CPU_HANDOFF.md)、[GPU](archive/2026-09/v74-0920/WORLDSIM_V7_4_H2_GPU_P1_REPORT.md) |
| WS-V74-H2-P15-01 / switching-margin-r1 | 失效轨迹审计 | [P1.5](archive/2026-09/v74-0920/WORLDSIM_V7_4_H2_P15_FORENSICS_REPORT.md) |
| WS-V74-H2-P16-01 / 20260913__physical-metric-r2 | 普通几何控制 | [P1.6](archive/2026-09/v74-0920/WORLDSIM_V7_4_H2_P16_DIAGNOSTIC_REPORT.md) |
| WS-V74-H2-REDISCOVERY-01 / 20260915-cpu-r1 | 旧案例 CPU 回放与来源核对 | [重新取证](archive/2026-09/v74-0920/WORLDSIM_V7_4_H2_FIRST_RETURN_REDISCOVERY.md) |
| WS-V74-MAINFIG-01 / 20260915-first-return-r1；SECONDARY-01 / 20260915-secondary-r1 | 4主模型48前向、第二批24前向及表面参照 | [官方模型主图](archive/2026-09/v74-0920/WORLDSIM_V7_4_MAIN_FIGURES.md) |
| WS-SIM-IMPACT-01 / 20260915-r1 | HUGSIM-LTF 与碰撞表示 | [报告](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_IMPACT.md) |
| WS-SIM-LIDAR-01；INTERACTION-01 / 20260915-r1 | LiDAR、策略/PDM 与六日志交互 | [LiDAR](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_LIDAR_IMPACT.md)、[交互](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_INTERACTION_FINDINGS.md) |
| Pi3X 局部资产审计 | 路面修复未消除残余 | [因果审计](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_PI3X_CAUSAL_AUDIT.md) |
| WS-SIM-NATIVE-CLOSEDLOOP-01 / scene0004-step030000-full1 | SplatAD完整预算、反馈与感知控制 | [完整预算](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_FULL_BUDGET_PERCEPTION.md) |
| WS-SIM-FF-LOCAL-ASSET-01 / r2 | Ω 与 Pi3X 网格编辑、重新投射与检测 | [局部资产](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_LOCAL_ASSET_CAUSAL_AUDIT.md) |
| WS-SIM-FF-COHORT-PERCEPTION-01 / 20260915-r1 | 82新增+20复用检测；全部分母 | [六日志](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_SIX_LOG_PERCEPTION.md) |
| WS-SIM-FRESH-NATIVE-01 / 20260915-r1 | 16组真实驾驶基线/路线控制 | [新来源](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_FRESH_BASELINE.md) |
| WS-SIM-SPARSE-NATIVE-CLOSEOUT-01 / 20260920-r1 | 导入缺 terminaltables；0/40前向；依赖任务跳过 | [终态与日志](autoresearch/worldsim_simimpact/closeout_20260920/README.md) |
| WS-V74-DOCS-CLEANUP-01 / 20260920 | 文档职责、旧 failure 检索与归档链接修复；无新实验 | [整理记录](archive/2026-09/v74-0920/CLEANUP.md) |

每个 run 的真实完成数量、配置、seed、数据角色、成本和失败边界以该报告及对应 manifest 为准。不同类型执行次数不合并为独立样本。更早版本的实验从[历史归档](archive/README.md)进入。

- `WS-V77-DELETE-REPAIR-20260927/r1`：三场景入口试验失败并回退；实际数量与边界见[报告](v77/DELETE_REPAIR.md)、[登记](autoresearch/worldsim_v77/delete_repair_20260927/registration.json)、[收口](autoresearch/worldsim_v77/delete_repair_20260927/closeout.json)。
