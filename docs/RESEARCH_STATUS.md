# 当前研究状态

更新：2026-10-18。当前 task=WS-V82-R15-BASELINE-ENGINEERING-20261018；无subagent，两个正式实验直接执行。

用户最新要求：直接完成两阶段身份一致的五身份 GN11 30K＋新实例30K，取消两阶段之间的消融；额外补齐所有可见车辆（含停驶车辆）的掩码、真实初始化和统一身份表后，自动跑第一阶段30K。完整RGB保留，车辆由动态分支重建。本轮全车辆实验不跑第二阶段；最终同时检查逐ID删除和删除全部车辆后的完整静态街景。所有视频及显示使用完整画面，无ROI裁剪。failure_ledger_refs=[V82-F06]，不增加R16/R17，原45K暂停，历史资产保留。

五身份GN11实际继续训练，最近快照22100/30000。等待中的中间消融controller18165及贡献追踪27418已核对无子进程后取消，未消费新增训练步数；旧4×300保留，原five_raw_body误标更正继续有效。direct controller39526已登记：等待当前GN11真正30000后绑定实际背景/环境及静态点数，直接fresh五实例30000，然后实际完整49帧分支、独立资产、逐ID与全部删除。没有1K质量停等或新的中间对照。第二阶段恢复孤立raw-body alpha100，同时保留visible-ID100，按2000步SH/49相机无放回采样，并接入原生静态及保身份刚体空间clone/split/prune。空间参考9项和epoch sampler完整恢复/未来150次序列通过CPU；GPU训练仍待执行。刚体不采用PVG时间split，SGCM沿用第一阶段结果；原生append重置屏幕半径历史的次序被忠实保留，不冒称有效的持续屏幕大点约束。

全车辆输入CPU清查49帧共45条车辆轨迹，其中20条在本相机有投影LiDAR，32条按中位速度<0.5m/s计停驶，这些统计不代表45辆均可见。现有R15的43身份不能等于完整覆盖。已用原始1920×1280完整49帧运行原生SAM3 vehicle预测，推理峰值11.2704GiB，在14GiB自身分配器上限与核实主进程下并行完成；未改旧mask。低分辨率旧缓存和新全分辨率提议仍有远处车辆关联/覆盖缺口，不能把未知当背景。all-vehicle30K controller41828已登记，等待all_vehicle_ready.json的实际覆盖/初始化/同一身份集验收，然后自动30K和完整49帧删除全部车辆评估。输入准备及峰值准入通过才并行，未证明显存足够时排队；未观测遮挡区域没有新补全算法，不声称自动恢复。

[双实验组件、当前状态和完整画面说明](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/index.html)。human_verdict=null。全部两项实验、评估、报告及Git交付完成后才按授权关机，automation-2每30分钟跟进。

五身份分支同步视频已交付：旧五身份30K与原R15冻结4K，a6/a37/a5/a26/a18，共49帧raw89—137。主入口改为同一时间轴的删除前后整景、全部Actor动态分支、静态分支，可切换alpha、独立资产、观测图与物理阴影；另导出五段合成总览MP4，所有分支在同一视频内逐帧对齐。默认关闭物理阴影，删除前后整景保持相同开关；开阴影时目标从实际太阳caster中移除，静态接收面独立光栅化。旧5重放full逐像素匹配既有PNG，所有删除target-alpha=0；R15静态几何/外观不改。浏览器安全策略拒绝file://验证，未宣称实际浏览器验收；帧数/编码/帧序和异步解码同步逻辑另有验证。[同步播放器与组件图](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/baseline_complete.html)。三维中心页仅作补充，human_verdict=null。

旧GN11 30K＋五实例30K已完成，a5/a26/a18/a6/a37都有真实独立owner，全49帧245身份帧的独立RGB、alpha、全景可见alpha与删除完成。历史旧3的147身份帧与原冻结R15的245身份帧也已完成，总637身份帧；本地全PNG矩阵及31视频交付，Git保存视频、固定锚帧和完整三值支撑证据。[旧基线完成对照与组件](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/baseline_complete.html) · [旧5/原R15三维中心与真实支撑](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/tristate/index.html)。

以同一真实车框、物理米坐标及1e-5m容差，旧5/原R15的框外中心：a6 3.73%/24.09%，a37 6.35%/17.53%，a5 1.97%/4.22%，a26 2.45%/6.85%，a18 34.97%/46.44%。旧5无增密且初始化不同；a6最大σ约3.29m，在raw105/113/121可见大面积下部支撑、亮白/蓝色条带。a37两版删除均残留车形；旧5不是干净资产真值，框外率低也不能证明资产干净。旧5没有三值证据/概率，绿色只表示硬owner。原始历史旧3检查点仍只有三个owner，不能把新5归属倒推到它。

修复R15实际完成4000步，训练总用时63.11分钟（含周期审计/保存，非同点数单步基准）；1K/2K/3K/4K检查点保留。三次46帧当前证据审计实际owner变化分别为[86, 39, 30]。五身份49帧独立/删除及真实动态/静态分支完成，已加入同步播放器第三模型；该轮R15队列已结束；当前新增基线工程重跑仍在执行，不能据旧队列状态关机。实际raw113中a6下部白带、a37关阴影删除后的静态车形仍在，整体资产质量未通过。[完整实际结果与组件](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/repair_4k_20261018/index.html)。human_verdict=null。

修复使用Actor不可重置来源锚点、累计0.25m、真实框+0.10m容许域（保留初始异常包络）、σ增长1.5倍、2%每次/4倍总增密预算；当前证据审计在1500/2500/3500，source-tag不挡候选，反证撤销确定状态，重复真实3D才改owner，真实道路/大支撑可否决整点转车。FP64因子阴影接入模型/恢复。原30K学习率、loss、SGCM、容量保留，外部池不重放旧459点补入。组合修复不能归因单项，需检查保体、a6碎片与a37外观残影。26项CPU针对测试通过；4项旧reference失败在原版相同复现，已披露。

新旧五身份实际分支已经交付为同步逐帧视频。原GN11、旧3、旧5、原R15及新修复4K检查点和原始输入保留，用户其他dirty不动。同一重建窗口，无独立编辑GT，human_verdict=null，problem_solved=null。

此前R15 4K和三版分支渲染结束；当前用户新增的基线两阶段工程重跑优先。全部新任务、controller、评估、报告与Git交付真正完成后才按既有授权关机，记录实际SSH观察。


补充修复4K三维归属对照：[原4K／修复4K同视角中心与证据](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/repair_4k_20261018/tristate/index.html)，可切相同初始化／旧5对照，五身份/背景/状态筛选及点属性。CPU只读提取真实4000步检查点，a6/a37/a5/a26/a18内部unknown=50/20/53/27/0，全部稀有unknown保留于显示样本；背景视图同时保留目标本体。中心不等于完整Gaussian支撑，新增修复三值GPU支撑着色投影未另做；真实49帧RGB/alpha/Delete视频已交付。控制逻辑与状态计数验证通过，未冒称浏览器file://实际验收。
