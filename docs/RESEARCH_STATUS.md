# 当前研究状态

更新：2026-10-21 06:04 SGT，主task=WS-V82-R15-GN11-STAGE1-V3-20261020。当前campaign=/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_serial_r1_20261021，单槽run_v3_four_arms.py/four_arm_registry.json已自动由full切到real_only，completed_arms=[full]；物理并发1，顺序full→real_only→prior_only→ownership_only，逻辑两批不变。49训练/49道路参考、seed0、各fresh stage1 30K/stage2 0、每5K六节点及原生点数/分辨率/损失/调度/SGCM/NPS均不变。full于约05:08完成训练段，逻辑30000且三个optimizer.step实际调用各30000；六个5K/10K/15K/20K/25K/30K事件全部complete、24件checkpoint和各49帧七通道/道路浮点文件实际存在，各节点均已分别做有限值核验。30K四件共2,634,337,175字节，343张PNG+49份NPZ逐件验证480×320、有限值与alpha最大0.99990058；CPU加载模型/状态有限，14,034永久道路ID唯一且747条容量记录守恒，SGCM/道路/真实CUDA RNG齐全、sampler.steps=30000。30K的49训练视角重建PSNR=34.6138、SSIM=0.96554、LPIPS=0.03892，不是最终编辑质量或科学收益判定。full的每参数Adam时钟分别29856/29989/30000，原生optimizer之前144/11次通用拓扑换Parameter后grad=None，与实际step调用计数不是同一口径；不得为凑时钟改原生调度，也未声称实际非零中断精确续训。full已退出且不重启；控制器10144/start_ticks850738395、当前real_only训练17779/start_ticks851761976已按/proc身份核验，不按这些历史PID恢复。real_only真实CUDA零更新前后向/数值准入和0步四件/49帧七通道/道路浮点发布及严格状态恢复通过，实际文件亦CPU核验；0步不是正式5K。real_only正式5K和10K两个事件均complete：本轮10K四件共2,123,294,423字节、49帧×7=343张PNG及49份道路浮点NPZ逐件核验尺寸/有限值/alpha范围，CPU加载模型有限且14,034永久道路ID唯一守恒，sidecar的SGCM/道路/真实CUDA RNG齐全、sampler.steps=10000；内部每参数Adam时钟9905/9989/10000另报，不冒充step调用数或实际非零续训验证。该10K训练视角PSNR=31.4195、SSIM=0.93797、LPIPS=0.07777，不作质量判定；最近status11600逻辑步、三step调用各11600，目标有限、道路ID守恒，observed_rgb/depth=1/0.1、prior_rgb/depth=0/0符合配置；prior_only和ownership_only尚未启动。同一心跳v3-gpu自05:32起ACTIVE、每30分钟。本轮为第一次真实30分钟定时检查：06:03快照monitor/check_20261020T220310348944340Z.json相较05:31基准real_only由5200推进到11500（随后status11600），三step调用一致、healthy_now=true、both_current_arms_have_5k_node=true，未见保存/数值/显存/磁盘风险；30分钟稳定计数=1/2，维持30分钟，不把此前10分钟或本轮CPU补充核验当第二次30分钟检查。再一次真实30分钟检查稳定且无风险才可改同一任务为60，保留其他字段，不新建重复任务。新arm/批次或风险立即回10。full整步profile最大allocated/reserved约19.24/25.55GiB，real_only截至本轮约13.05/16.06GiB，profile不含评价保存，单节点和瞬时低显存不能保证后期容量；当前盘余约65GiB、未见OOM或数值错误。full在29999次更新时的道路独立/删除五车背景/联合两个锚视角贡献诊断有限且原alpha重放通过，合法遮路不当非法遮挡GT、来源梯度不当因果收益。当前只完成full训练段，正式49帧删除五车/道路/联合收口评价仍待实现/验证并执行，不得把训练节点评价当最终完整删除评价；人审待定，不会为制造好结果重调方法。原publish_r2 full8400/real_only7752调用后OOM现场和5K、原发布失败10,000均保留，累计失败逻辑更新26,152另记。CPU79项与full/real_only实际零更新CUDA准入通过，调度修复35e87f9c已推送。用户授权确认工程bug自主诊断、备份、测试、提交推送并必要恢复或新目录fresh重跑；只有四组均30K、六节点、全部正式49帧评价/资产核验、报告代码索引推送完成，且全机无未结束或待启动训练/评价/数据/渲染/控制器队列，才正常关机，不需再审批、不释放销毁实例或删数据。该最新授权覆盖旧仅监控/无关机限制，不是现在关机；GPU空闲、异常退出或train返回EVALUATION_PENDING均不算全完成，不强杀不相关工作。无subagent、无199扩展/反事实、未执行关机；quality_accepted=false、human_verdict=null，failure_ledger_refs=[V82-F01,V82-F06]、delta=none。[资源恢复组件、测试、失败原始状态与授权](../scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v3_20261020/resource_recovery_r1/README.md)。

199曝光审计真实读取同一TFRecord的raw0–198，固定原26,088个世界单元与13,720查询并集，不重建曲面/法线/支撑。窗外没有原准备SAM mask，因此49/199均按全类别相机标注框、投影标签框、近面裁剪三维框包络的同口径重算，另用原有LiDAR/射线等门限；原V3/SAM49仅作独立参考。raw89/a6的确认单元0→225/493（来源raw149–198；仍42零来源、47单来源、179无曲面支撑）；a37为283→840/1170（新增557，仍56零来源、76单来源、198无支撑）。查询并集1013→4013，新增3000=2897零来源转确认+103单来源转确认。至少2帧才确认；几何证据不是语义GT、纹理质量通过或因果收益。原49全RGB/pose/标定/LiDAR/时间戳实测一致；主道路包未晋升199候选。不能把a6在49中无曝光外推成199中全部只能先验，也不能把剩余未知补成真实纹理。[组件、真实结果与复现](../scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v3_20261020/exposure199/README.md)。

v3共同修复保持真实联合前向、不裁错误遮挡。R2车身RGB仅更新非道路 `xyz/opacity/scaling/rotation` 与全部动态参数，排除背景 `t/scaling_t/velocity`、SH、明确道路行和EnvMap；其余原有效路径仍可更新背景时间参数，不是全背景冻结。车身L1主分母改为3HW，仅这一部分与V1等效，掩码DSSIM不等于V1整图DSSIM；旧面积归一只留作独立备选。静态平均深度前遮挡、三维中心体积排斥和动态严格掩码外mean+top1%/低alpha高颜色损失的权重与阈值不变。完整动态mask alpha MSE100与独立RGB继续保留，动态独立RGB环境填充仍stop-gradient。

确定性屏幕重复计数已修复：路由autograd.grad前后保存/恢复三套代理.grad，隔离非叶子retain_grad的自动留存副作用，再把base与指定路由梯度各计一次。CPU原式复现7→14；修复覆盖非叶子/叶子、既有缓冲、body/road/base叠加和道路行屏蔽。分支/联合增密仍逐Gaussian取max、每视图单计数。道路额外来源有自己的光栅代理；固定道路容量不走通用增密，不误称其全部进入当前训练视图的静态代理。

新增只读诊断：每100步记录车内质量、车外颜色/top1%、边界带误差、联合/独立动态/静态与LiDAR深度覆盖/MAE。准入、第一步与5K节点更新前保存浮点图，并抽最多14个有效车身LiDAR像素；用原生feature通道保留全部高斯与原alpha*T，分别计算静态T/联合T中的车前静态贡献、框外中心贡献与道路行贡献，同时对照框内中心计数。最多4次额外光栅/节点、样本alpha与原图不等即失败；缺LiDAR不伪造零误差。5m/20m平均抵消用CPU构造验证，未宣称真实场景已测出漏检；不是连续椭球first-return，也不是全图贡献统计。[组件、责任和诊断工具](../scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v3_20261020/README.md#监督责任)。

本次正则修订：静态寿命项 `1/mean(abs(map))` 权重0.1，静态速度图L1权重0.01；两者保留联合alpha归一化。NPS重新启用、权重0.01，保留独立背景alpha归一化、原五身份mask、边界距离和四邻域双奇偶求和。训练直接传原配置给原生混合损失；上一轮选择性raw运动／显隐正则从活跃路径移除，不叠加、不加载三维时间证据。旧证据和离线工具保留，不是本轮方法依赖。原生 `_t/_scaling_t/_velocity` 仍可训练；五车排斥仍用实际渲染三维位置、有效opacity与合法道路薄层保护。V1 feature velocity的寿命衰减继承不改；全图软正则可能抑制非目标运动，不能称为选择性保真已解决。

道路新路径：五车全部49帧掩码射线自动查询同一实测曲面，永久0.2m世界单元与1.2m上下文邻域；每个来源保留原始RGB/uv/深度/9项硬门限，不融合颜色、不优先late帧、不用人工供体或周期纹理。至少2个硬通过来源才能确认曝光，单来源仍未知；有曲面支持但零来源才是先验。26,088个世界单元中14,034个有支撑，新增同数固定薄层高斯计入原300万静态点数上限，不另加Adam步。道路行用同一DC纹理和几何参加正常联合渲染；曲面/法线/薄层/锚点约束共同启用，固定ID不参与通用拓扑与SGCM替换，也不在opacity reset节点清成透明。普通背景机制不变。额外每步2个来源共享优化同一参数，可靠性只能对硬通过观测减权；无曝光区用连续先验与真实邻区边界，不是真实隐藏纹理。

真实覆盖分母为世界单元：raw89/a37=1170（351确认曝光、528先验、291未知，其中93单来源未确认）；raw89/a6=493（0确认曝光、314先验、179未知）。a6仍不能宣称恢复唯一真实纹理；不能与V2像素统计直接比成功率。已生成两车逐单元来源审计页，并在浏览器实际切换查看原图裁片/拒绝原因；现在训练日志开始记录实际可靠性、访问和采样梯度，不以示意值代填，梯度触达不代表因果收益。实际某单元raw131被拒绝，不照搬用户示例数值。

四组仍为五身份、同49帧、seed0、fresh第一阶段30K/第二阶段0、5K间隔六节点。共同包含新道路表示/几何和原v3修复，只改变查询真实多视角与连续先验来源权重：full、real_only、prior_only、ownership_only。最后一组现在是共同道路表示/几何条件下两类新增外观来源关闭，不等同旧V2同名组。额外来源光栅开销另报，不声称与原生30K相同计算量。每组先真实CUDA零更新gate，失败保留现场不自动重试。正常四臂审批还需匹配新的road_version。

反事实仅登记：`road_cf_raw89_a37_drop_raw133`，真实几何权重自动选择raw133，覆盖126个确认查询单元。reference与drop_source同初始化/seed/30K，移除指定单元的额外道路来源，不移除原生整帧重建；执行入口需独立GPU审批，主四臂审批不能授权额外两组。如果均fresh运行增加60K，精确同合同full臂可审核后复用。当前来源权重/采样梯度不能证明帮助，两个真实终态和49帧质量检查仍待执行。

渲染审计新增：gate在真实图逐项求导对照三套屏幕缓冲，再在临时统计核对max/可见性并集/单次分母；本轮只CPU桩回归。固定道路行不计通用增密统计。CUDA events采样gate/1/100步及全步光栅调用累计，主机wall/整步显存峰值、来源访问/权重/梯度触达、原生cap超额和拓扑/SGCM净容量变化分别记录。gate/1/5K更新前增加3次贡献光栅，核对原alpha并输出道路单独/背景/联合T下的全部道路贡献，按世界单元anchor采样；合法车遮路不标成非法遮挡。新render版本和真实数值证据加入准入闸门。[组件、边界和复现](../scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v3_20261020/README.md#渲染增密与道路审计)。

CPU验证现63项通过、0失败/错误/跳过，包含先前57项与覆盖孔洞/边界图/固定子单元/三路mask/两种入口授权及CLI绑定回归。V1两份损失源码不变；新49包只修复mask门限，固定查询/几何/原始RGB相等。中途新增边界诊断曾误索引已reduce标量，CPU捕获后修复并全量重跑，未进入GPU；旧expand测试桩修复证据保留。首组实际CUDA数值/alpha重放通过并开始开销/容量日志，最终场景质量未验收。199审计不重跑、不晋升。新凭据见[road_review_r2/validation.json](../scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v3_20261020/road_review_r2/validation.json)与同目录gpu_full_gate_001；历史render_audit_r1和supervision_r2保持原记录。failure_ledger_refs=[V82-F01,V82-F06]；failure_ledger_delta=none。

短程控制仅登记背景时间车身路由开/关配对建议，各1000步、额外2000步；两者共同采用本轮屏幕bug修复、道路行保护和V1尺度L1，面积归一不混入。需要独立runner审核与GPU批准，没有可执行队列、没有消耗预算；该建议不能冒充已验证时间路由是泄漏原因。主审批和gate新增监督版本匹配，旧版本准入不授权新配置。

资源准入：199审计后只读采样同盘可用105,957,957,632B，超过主四臂70,000,000,000B门槛。旧V2包、checkpoint与旧审计保留，本轮未清理既有资产；早期001–004不完整CPU工程运行现场保留，未计入曝光结果。状态仍为 `WAITING_USER_REVIEW_AND_GPU_APPROVAL`；真实CUDA新路径、显存/时间和训练质量待用户审核授权。

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
