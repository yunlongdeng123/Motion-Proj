# 当前研究状态

更新：2026-10-18。当前 task=WS-V82-R15-BASELINE-ENGINEERING-20261018；用户要求撤出旧五实例干净基线结论，先排工程再比较算法。failure_ledger_refs=[V82-F06]，不新建failureID、R16/R17，无subagent；原45K暂停保留。

用户最新授权下已重新安排完整五身份GN11 30K＋五实例30K，旧结果保留为历史诊断而非合格基线。确认原GN11动态集合只有a5/a26/a18；a6/a37共45418/555950个五身份正像素在原canonical动态mask中覆盖均0，两车直到实例阶段才退出静态RGB监督。raw128原始/GN11/旧3/旧5同帧链路确认旧5蓝白条带显著恶化；49帧native静态renderer与适配器最大RGB差约2.4e-7。实际CUDA第二批通道、单ID、换序及梯度通过。较早本地快照导致的学习率10倍断言已撤回，实际归档调用路径会补乘0.1。

原raw动态body α100在刚体适配中被0权重及可见ID α替代，遮挡语义不等价；固定拓扑也不能称完整继承原增密/剪枝。已完成4×300步诊断；实际调用复核发现five_raw_body仍用全场景可见union alpha，只恢复了100权重，撤回“恢复原孤立body”的描述，前四臂opacity日志均不能当孤立body指标。旧臂与检查点保留，另排真正使用package.dynamic_alpha的five_raw_body_isolated 300步。五身份一致的独立原始测量输入已完成实际49相机、.1米尺度及SGCM CPU检查；新五身份GN11 30K已经实际启动，最近原生状态已超过10K；两阶段实际坐标变换差0、时间差1.11e-16且49帧身份/二值union一致。后续controller18165已排队等待当前GN11释放GPU，再顺序执行纠正300步、两臂损失梯度与完整恢复/下一更新检查、五臂全49帧独立/删除渲染；不并发抢卡、正式实例更新仍0。独立run为campaign/r15_baseline_owner5_20261018，完成后核对原约束、实际损失梯度、Adam/RNG恢复及编辑表现再启动fresh五实例30K。原GN11/旧3/旧5/原R15/修复R15检查点和原始输入不改，不因旧队列complete提前关机。[本轮组件图、逐帧溯源与固定协议](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/index.html)。human_verdict=null。

五身份分支同步视频已交付：旧五身份30K与原R15冻结4K，a6/a37/a5/a26/a18，共49帧raw89—137。主入口改为同一时间轴的删除前后整景、全部Actor动态分支、静态分支，可切换alpha、独立资产、观测图与物理阴影；另导出五段合成总览MP4，所有分支在同一视频内逐帧对齐。默认关闭物理阴影，删除前后整景保持相同开关；开阴影时目标从实际太阳caster中移除，静态接收面独立光栅化。旧5重放full逐像素匹配既有PNG，所有删除target-alpha=0；R15静态几何/外观不改。浏览器安全策略拒绝file://验证，未宣称实际浏览器验收；帧数/编码/帧序和异步解码同步逻辑另有验证。[同步播放器与组件图](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/baseline_complete.html)。三维中心页仅作补充，human_verdict=null。

旧GN11 30K＋五实例30K已完成，a5/a26/a18/a6/a37都有真实独立owner，全49帧245身份帧的独立RGB、alpha、全景可见alpha与删除完成。历史旧3的147身份帧与原冻结R15的245身份帧也已完成，总637身份帧；本地全PNG矩阵及31视频交付，Git保存视频、固定锚帧和完整三值支撑证据。[旧基线完成对照与组件](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/baseline_complete.html) · [旧5/原R15三维中心与真实支撑](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/tristate/index.html)。

以同一真实车框、物理米坐标及1e-5m容差，旧5/原R15的框外中心：a6 3.73%/24.09%，a37 6.35%/17.53%，a5 1.97%/4.22%，a26 2.45%/6.85%，a18 34.97%/46.44%。旧5无增密且初始化不同；a6最大σ约3.29m，在raw105/113/121可见大面积下部支撑、亮白/蓝色条带。a37两版删除均残留车形；旧5不是干净资产真值，框外率低也不能证明资产干净。旧5没有三值证据/概率，绿色只表示硬owner。原始历史旧3检查点仍只有三个owner，不能把新5归属倒推到它。

修复R15实际完成4000步，训练总用时63.11分钟（含周期审计/保存，非同点数单步基准）；1K/2K/3K/4K检查点保留。三次46帧当前证据审计实际owner变化分别为[86, 39, 30]。五身份49帧独立/删除及真实动态/静态分支完成，已加入同步播放器第三模型；该轮R15队列已结束；当前新增基线工程重跑仍在执行，不能据旧队列状态关机。实际raw113中a6下部白带、a37关阴影删除后的静态车形仍在，整体资产质量未通过。[完整实际结果与组件](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/repair_4k_20261018/index.html)。human_verdict=null。

修复使用Actor不可重置来源锚点、累计0.25m、真实框+0.10m容许域（保留初始异常包络）、σ增长1.5倍、2%每次/4倍总增密预算；当前证据审计在1500/2500/3500，source-tag不挡候选，反证撤销确定状态，重复真实3D才改owner，真实道路/大支撑可否决整点转车。FP64因子阴影接入模型/恢复。原30K学习率、loss、SGCM、容量保留，外部池不重放旧459点补入。组合修复不能归因单项，需检查保体、a6碎片与a37外观残影。26项CPU针对测试通过；4项旧reference失败在原版相同复现，已披露。

新旧五身份实际分支已经交付为同步逐帧视频。原GN11、旧3、旧5、原R15及新修复4K检查点和原始输入保留，用户其他dirty不动。同一重建窗口，无独立编辑GT，human_verdict=null，problem_solved=null。

此前R15 4K和三版分支渲染结束；当前用户新增的基线两阶段工程重跑优先。全部新任务、controller、评估、报告与Git交付真正完成后才按既有授权关机，记录实际SSH观察。


补充修复4K三维归属对照：[原4K／修复4K同视角中心与证据](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/repair_4k_20261018/tristate/index.html)，可切相同初始化／旧5对照，五身份/背景/状态筛选及点属性。CPU只读提取真实4000步检查点，a6/a37/a5/a26/a18内部unknown=50/20/53/27/0，全部稀有unknown保留于显示样本；背景视图同时保留目标本体。中心不等于完整Gaussian支撑，新增修复三值GPU支撑着色投影未另做；真实49帧RGB/alpha/Delete视频已交付。控制逻辑与状态计数验证通过，未冒称浏览器file://实际验收。
