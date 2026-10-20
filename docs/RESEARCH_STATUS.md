# 当前研究状态

更新：2026-10-20T19:41:45+08:00。task=WS-V82-R15-GN11-STAGE1-V3-20261020。v3 已按用户要求撤销背景统一时间冻结，改为原生背景 PVG + 三维证据选择性时间正则；代码、四臂计划、架构图与CPU证据已部署。GPU 未授权、控制器未启动、训练更新0、human_verdict=null。

v3 共同修复四条机制：在真实联合前向上恢复车身最终合成 RGB，并把梯度路由到全部动态参数和静态几何/opacity，排除静态 SH 颜色与 EnvMap；用车身有效 LiDAR 深度处罚非法静态前遮挡，同时保留 v2 的三维中心体积排斥；对 EnvMap 混入前的独立动态预乘颜色增加严格掩码外 mean+top-1% 与低 alpha/高颜色比值约束；显式返回静态/动态分支屏幕梯度，与联合统计逐 Gaussian 取 max 且每视图只计一次。完整动态 mask alpha MSE 权重 100 无条件保留，不增加显示阶段颜色截断。

本次时间修订：背景 `_t/_scaling_t/_velocity` 和原生 SHM/时间显隐、时间梯度/原生时间细化恢复；稳定表面权重由同49帧重复LiDAR与有支撑路面几何决定，所有车辆/行人/骑行者三维体积否决稳定标签。稳定区运动/显隐各0.1，上下文/未知各0.001；直接约束raw velocity位移上界，避免feature velocity随寿命衰减的漏洞。全图绝对时间/速度项由选择性目标替代，NPS仍关闭。五车排斥使用本次光栅化实际位置及有效opacity，保护合法道路薄层；车身联合梯度包含背景时间参数但继续排除颜色。CPU体素并集82,973，与物体否决区交集0；原585,943点中心查询有59,151个上下文/未知，实际训练还按尺度筛选稳定资格。

四组均为 a6/a37/a5/a26/a18、raw89–137 共49帧、seed0、fresh 第一阶段30K、第二阶段0、5K间隔六节点；共同使用上述 v3 修复，只改变道路来源权重：full、real_only、prior_only、ownership_only。固定单路顺序执行，每组先做真实 CUDA 前向/反向零更新 gate；失败保留现场，不自动重试或覆盖。

CPU 验证：14项测试通过，覆盖原有目标/路由、背景时间前向与梯度、raw运动不被宽寿命藏住、当前几何/拓扑重查、时间增密计数、append/prune与checkpoint恢复、OmegaConf政策解析。模型生命周期测试只替换CUDA设备常量为CPU；另以真实49帧metadata和模型桩验证配置构造。编译、政策/四臂实际输入预检通过。真实CUDA准入与质量仍待授权后验证；缺GPU审批时控制器实际拒绝，无训练进程。

资源准入：本轮只读采样同盘可用106,154,893,312B，已超过四臂控制器70,000,000,000B门槛。此前空间阻塞已不适用；本轮未清理既有资产。campaign状态更新为 `WAITING_USER_REVIEW_AND_GPU_APPROVAL`，用户审核并明确开放GPU后才创建独立审批文件。

[v3 方法、组件图、四臂与限制](../scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v3_20261020/README.md) · [固定计划](../scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v3_20261020/plan.json) · [CPU 部署凭据](../scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v3_20261020/cpu_validation_receipt.json)。failure_ledger_refs=[V82-F01,V82-F06]，failure_ledger_delta=none；本轮不新增失败ID，不宣称GPU、30K或质量结果。

## 已完成的原A/B基线与历史证据

A同一批a6/a37/a5/a26/a18完成fresh GN11逻辑30K＋fresh实例30K。B按最新范围完成LiDAR∩FRONT相机20身份原生第一阶段30K，第二阶段0。两项均保留原49帧完整RGB；13纯相机轨迹只作上下文，B删除全部纳入车辆不等于全画面无车。原45K暂停，所有历史输入/checkpoint保留；未加入窗外训练帧、生成补全或新消融，human_verdict=null。

A五身份两阶段30K＋30K、B交集20身份第一阶段30K及最终49帧零更新删除评估已完成；两项均未达到干净街景目标。A／B静态分支和删除图仍保留明显车形、亮白／彩色条带；A的a6独立资产有白色带状碎片。不能把训练完成、低损失或目标alpha清零当作分离质量通过。

技术完成凭据：A真正30K完整单文件3047836844B，91项模型/Adam张量有限、RNG/采样器30000与终点策略合同已CPU核验，终点检查明确不授权30001步；最终评估另实际CUDA恢复全部模型/Adam/RNG/采样状态精确相等，零更新。A完整49帧×5身份共245记录、1568 PNG；每ID删除目标可见alpha精确0，静态分支逐像素不变。B四组件30K/Adam/RNG/SGCM/sampler及native49评估已CPU核验，最终另49帧343 PNG删除纳入动态集合，独立static回放通过。两项最终评估不改checkpoint。177558879B仅PNG/JSON归档已一次下载并逐文件SHA核对，未下载checkpoint/NPZ。46路当前完整480×320/49帧/10fps/all-I视频已发布（新A32、新B7、A第一阶段7），默认新A，可逐身份选择；全部显示/导出完整画面，无ROI。技术和文件通过不代表背景质量通过。

原GN11曾在25847读取未完成CUDA事件中断，用户仅授权5846必要重放；该历史额外预算单列。后续所有原生/实例完整状态每5K保存；原不存在的GN11 5/15K不补造。实例1999完整保存后切换同预算共享准入、从2000精确恢复，额外重放0。后来宿主机md0写入等待时保留原进程，02:45UTC确认同PID恢复继续，额外重放0；具体平台根因和恢复时刻未获确认。30K只读检查脚本首次遇到终点续训限制和终态精简字段，现用独立终点合同/真实最后4000字节日志解决；未修改训练实现、目标、上限或已保存模型。

两阶段身份/mask/init输入绑定已验证。实例适配为GT刚体、原孤立raw-body100＋visible-ID100、SH2000、49无放回、原生static与保owner空间clone/split/prune，staticLR.1/环境冻结、第二阶段无新SGCM；不复现PVG时间split。静态原生时间位移与衰减正则不一致的继承问题、缺道路观测与外观阴影证据分别保留，不能将所有残留归因于补道路，也未对具体残留Gaussian完成因果归因。学习率10倍误判和短测body域误标撤回继续保留。

[完整画面同步播放器与质量报告](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/index.html) · [原始输入覆盖](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/intersection_inputs.html)。文件/PNG/实际视频解码与JS合同验证已完成；file URL安全限制未绕过，未宣称浏览器UI交互验收。13份真实训练源码原字节ZIP和licenses哈希核对，源ZIP SHA256=331f837f2f0fd36c9d58875d1027d1e878aa84f957110f9621a5f618e3cb1c58。failure_ledger_refs=[V82-F06]，不新增ID。全部A/B训练、最终零更新49帧删除评估、一次素材下载、完整画面实际复核和Git交付后，已按授权执行AutoDL关机。实际请求2026-10-19T04:55:43.986900+00:00，SSH观察完成2026-10-19T04:56:11.660350+00:00；入口/usr/bin/shutdown，执行前sync。关机前交付提交696a05c00f9409e9c9e18cbe3f235e411d9c18d3已推送并核对remote HEAD。当前A/B-aware preflight核对guardian79622、A79624、B79625、旧磁盘helper252862/252851及全部campaign worker/待启动controller均无活动，GPU计算进程为空，最终完整检查点和归档保留。SSH连续三次连接失败；这是实际SSH不可达观察，不声称已核实AutoDL平台计费。A/B干净街景质量仍未通过；没有再训练或重放，human_verdict=null。

五身份分支同步视频已交付：旧五身份30K与原R15冻结4K，a6/a37/a5/a26/a18，共49帧raw89—137。主入口改为同一时间轴的删除前后整景、全部Actor动态分支、静态分支，可切换alpha、独立资产、观测图与物理阴影；另导出五段合成总览MP4，所有分支在同一视频内逐帧对齐。默认关闭物理阴影，删除前后整景保持相同开关；开阴影时目标从实际太阳caster中移除，静态接收面独立光栅化。旧5重放full逐像素匹配既有PNG，所有删除target-alpha=0；R15静态几何/外观不改。浏览器安全策略拒绝file://验证，未宣称实际浏览器验收；帧数/编码/帧序和异步解码同步逻辑另有验证。[同步播放器与组件图](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/baseline_complete.html)。三维中心页仅作补充，human_verdict=null。

旧GN11 30K＋五实例30K已完成，a5/a26/a18/a6/a37都有真实独立owner，全49帧245身份帧的独立RGB、alpha、全景可见alpha与删除完成。历史旧3的147身份帧与原冻结R15的245身份帧也已完成，总637身份帧；本地全PNG矩阵及31视频交付，Git保存视频、固定锚帧和完整三值支撑证据。[旧基线完成对照与组件](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/baseline_complete.html) · [旧5/原R15三维中心与真实支撑](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/tristate/index.html)。

以同一真实车框、物理米坐标及1e-5m容差，旧5/原R15的框外中心：a6 3.73%/24.09%，a37 6.35%/17.53%，a5 1.97%/4.22%，a26 2.45%/6.85%，a18 34.97%/46.44%。旧5无增密且初始化不同；a6最大σ约3.29m，在raw105/113/121可见大面积下部支撑、亮白/蓝色条带。a37两版删除均残留车形；旧5不是干净资产真值，框外率低也不能证明资产干净。旧5没有三值证据/概率，绿色只表示硬owner。原始历史旧3检查点仍只有三个owner，不能把新5归属倒推到它。

修复R15实际完成4000步，训练总用时63.11分钟（含周期审计/保存，非同点数单步基准）；1K/2K/3K/4K检查点保留。三次46帧当前证据审计实际owner变化分别为[86, 39, 30]。五身份49帧独立/删除及真实动态/静态分支完成，已加入同步播放器第三模型；该轮R15队列已结束；当前新增基线工程重跑仍在执行，不能据旧队列状态关机。实际raw113中a6下部白带、a37关阴影删除后的静态车形仍在，整体资产质量未通过。[完整实际结果与组件](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/repair_4k_20261018/index.html)。human_verdict=null。

修复使用Actor不可重置来源锚点、累计0.25m、真实框+0.10m容许域（保留初始异常包络）、σ增长1.5倍、2%每次/4倍总增密预算；当前证据审计在1500/2500/3500，source-tag不挡候选，反证撤销确定状态，重复真实3D才改owner，真实道路/大支撑可否决整点转车。FP64因子阴影接入模型/恢复。原30K学习率、loss、SGCM、容量保留，外部池不重放旧459点补入。组合修复不能归因单项，需检查保体、a6碎片与a37外观残影。26项CPU针对测试通过；4项旧reference失败在原版相同复现，已披露。

新旧五身份实际分支已经交付为同步逐帧视频。原GN11、旧3、旧5、原R15及新修复4K检查点和原始输入保留，用户其他dirty不动。同一重建窗口，无独立编辑GT，human_verdict=null，problem_solved=null。

此前R15 4K和三版分支渲染结束；当前用户新增的基线两阶段工程重跑优先。全部新任务、controller、评估、报告与Git交付真正完成后才按既有授权关机，记录实际SSH观察。


补充修复4K三维归属对照：[原4K／修复4K同视角中心与证据](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/repair_4k_20261018/tristate/index.html)，可切相同初始化／旧5对照，五身份/背景/状态筛选及点属性。CPU只读提取真实4000步检查点，a6/a37/a5/a26/a18内部unknown=50/20/53/27/0，全部稀有unknown保留于显示样本；背景视图同时保留目标本体。中心不等于完整Gaussian支撑，新增修复三值GPU支撑着色投影未另做；真实49帧RGB/alpha/Delete视频已交付。控制逻辑与状态计数验证通过，未冒称浏览器file://实际验收。
