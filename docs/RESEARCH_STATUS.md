# 当前研究状态

更新：2026-09-23。分支：`research/worldsim-v7.5-generative-state`。本文件是唯一当前快照；历史见[V7.4归档](archive/2026-09/v74-0920/STATUS_PRE_V75.md)，实验按[索引](EXPERIMENTS.md)查阅。

## 当前方向与授权

按用户最新要求，V7.5继续研究**基于高斯重建的反事实编辑**，但主验证转为下游导向的自建 paired-edit benchmark：冻结 factual/counterfactual 分支，覆盖速度变化、横向重定位/换道、移除和插入，以 A/P/E/O、trajectory adherence、object/background preservation 六维分别报告，pilot 不合成总分。统一的是 case、输出和评价合同；OmniDreams、ReSim、DriveEditor、GaussianDWM、Street Gaussians、HUGSIM 的原生/适配/consumer-only 能力不作伪等价。完整定义、能力矩阵和架构见[下游反事实 benchmark](v75/DOWNSTREAM_COUNTERFACTUAL_BENCH.md)，此前“什么重建状态足以支持可信反事实生成”的[问题定义](v75/PROBLEM.md)继续作为上位研究问题。

当前 AutoDL 已开机但无 GPU，本轮已把能做的 GPU-free 准备推进到底：六系统源码和能力 registry 冻结；24 个 case 全部通过 source-track、目标可见与初始碰撞 CPU gate，状态为 `geometry_pass_manual_pending`，道路与人工 verdict 仍为 null；ReSim 的 6 个 ego paired adapter、DriveEditor 的 18 个逐 case official-format pickle、GaussianDWM 三个场景共 715 帧兼容 Gaussian、HUGSIM 官方 scene-0383 导出场景/地图/5 scenarios/6 cars 均已物化。OmniDreams、ReSim、GaussianDWM、HUGSIM 为 `ready_for_gpu_preflight`，Street Gaussians 复用历史证据，DriveEditor 仅剩官方 Google Drive 12.1GB 模型的 24h quota。0次模型前向、无训练、无旧队列恢复。单张RTX3090正式推理既有授权保留；任何OOM立即停止，不自动降配置、重试或把不支持的任务换成 proxy。

2026-09-22 按用户最新授权完成历史存储退役：本轮旧 runs 大产物释放 166.8 GiB，当前可用约 356.3 GiB（已用 343.7 GiB）。历史配置、指标、日志、源码及报告图保留；旧训练 checkpoint、固定表面和大型数组已按清单退役，重建产物需要恢复输入并重新训练/推理，文档不是完整备份。当前 V7.5 的 4,038 个文件/链接核验未变，data/models/envs/external 和论文原文未清理。[历史 runs 研究脉络、删除与恢复清单](autoresearch/old_runs_retirement_20260922/README.md)。此前 V4 原始数据已整体退出常驻存储，预处理权重归位到 models/legacy_v4_preprocess；[V4 数据恢复](autoresearch/v4_data_retirement_20260922/README.md)、[前轮旧 runs 去重](autoresearch/storage_cleanup_20260922_r2/README.md)、[V7.5 数组压缩](autoresearch/storage_cleanup_20260922/README.md)记录保留。存储工作均 0 次模型调用，未改研究结论。

## 已有证据及保留边界

- [单卡基线](v75/BASELINE.md)、[原生开发样例](v75/NATIVE_COHORT.md)和[AV2桥接](v75/AV2_BRIDGE.md)证明公开单视图链路可运行；工程失败[V75-F01](research_failures/entries/V75-F01.md)已绕过，第三方源码未修改。
- [定位](v75/LOCALIZATION.md)、[自然读出](v75/NATURAL_STATE.md)和[可见性窗口](v75/VISIBLE_COHORT.md)保留全部好坏表现、参考排除和普通尺度控制。未建立稳定额外放大、普遍SOTA失效或驾驶危害，不再追加长时搜索。
- 两个[接近任务](v75/APPROACH_CLOSED_LOOP.md)已接通生成RGB→策略→动作→ego/相机→下一段生成。真实输入与GT生成分别门控；中心/尺度/目标LiDAR没有同时恢复动作与执行。固定6日志×3起点的运动跟车窗口18→0，平移、尺度、关联、运动和时长扩展均关闭。
- 形状主候选02678d04在唯一额外seed43中，配对行进差由+2.757m反为−0.324m，平均误差和欠制动差也反向，已关闭，不追加seed44。24642607保留两seed同向的小效应，普通类别先验已改善；承认普通解，不升级为严重危害或复杂主方法必要性。
- [原生DVGT契约与三时刻控制](v75/NATURAL_STATE.md#原生点图契约与有限时间上下文控制)：两次固定官方前向，14视角中13个方向误差改善，严重单帧错位大部分由普通历史观测解释。两例仍未通过事前原生投影筛查，不能把残余直接当生成badcase；不继续长度/坐标搜索。
- 已完成[速度先验控制](autoresearch/worldsim_v75/velocity_prior_control/README.md)：20条既有流CPU回放，0新模型调用。主项平均误差下降但额外制动恶化，同场景两个GT seed也未通过，未准入新生成；全部正反结果保留，不扫描速度/噪声。
- [对象移除反事实](v75/ACTOR_REMOVAL_COUNTERFACTUAL.md)完成。曝光开发源中 reference/DVGT removed 为 A=`1/10`、B=`9/10`/`8/10`，类别先验为 A=`4/10`、B=`4/10`，形成状态可编辑性候选；冻结的独立源三状态均为 A=`10/10`、B=`10/10`，reference gate 失败，候选未复现。把 A 从 f=0 起移出全部状态条件后仍为 A=`10/10`，支持 initial-image anchor 压过结构化编辑。该源禁止重建排名，也未启动闭环。
- 新的中距 G1 来源窗口筛到 A=`5859px²/41.88m`，只跑 reference 两段：removed A=`7/10`、B=`10/10`，仍未通过 A≤2 的门槛，故0次DVGT。三个固定来源的 A 残留为`1/10→7/10→10/10`，与初始投影面积增大、距离减小同向；当前仅是显著性 envelope 候选，面积/距离/场景/运动混杂，不声称因果或单调。
- 下一批8日志的远距小投影窗口为0个合格A→B对象对，0 detector/重建/生成；未放宽面积、距离、时间或拓扑。这是数据覆盖边界，不是模型通过。对象即时移除作为独立重建比较入口到此关闭。
- [未来轨迹反事实](v75/ACTOR_TRAJECTORY_COUNTERFACTUAL.md)已完成，被测模型是OmniDreams single-view 2B。两个来源的reference减速门控都通过。大误差来源给OmniDreams输入reference/depth-shift/depth+shape三种状态，响应误差为`2.11/55.61/32.08px`，编辑成功为`5/5、2/5、3/5`；小误差独立对照为`3.59/1.65/2.84px`，未退化。DVGT只提供一次现实来源的深度误差样本，不是此实验的生成模型主语。当前支持“OmniDreams会把足以改变未来占据与可见性的大状态误差传播到生成响应”，不支持任意误差有害、普遍发生率或严格阈值。
- 新的固定8日志确认窗口在第6个日志找到首个输入合格目标（`49.55m/3047px²`），但reference轨迹编辑在后段raster中只有`139/64/0/0/0`个变化像素，G0失败。按协议不替换第7/8个来源，0次DVGT、0次生成。现有大误差结果保留为机制badcase，不升级prevalence或策略危害主张。

这些实验支持接口能力与有限误差边界，**不证明重建普遍成熟，也不证明反事实正确**。DVGT距离读出使用GT尺寸/朝向、LiDAR诊断有额外信息、交通非反应式，原有信息边界继续有效。当前只有单一IDM，没有成熟驾驶多策略排名证据。

## 本轮交付与下一步

`WS-V75-DOWNSTREAM-CFBENCH-PREP-02 / 20260923-r1`完成 GPU-free bench 全量准备：10项本地单元测试通过；GaussianDWM 官方测试55 passed、1 skipped；六环境核心 import 5/6 通过，StreetGS 因旧环境缺 CUDA 扩展而准确 fail-closed。preflight 最终为 OmniDreams/ReSim/GaussianDWM/HUGSIM=`ready_for_gpu_preflight`、Street Gaussians=`prior_evidence_reusable`、DriveEditor=`source_only_missing_weights`。下载与物化脚本拒绝 `.aria2`/尺寸不足文件；GaussianDWM 官方裸 Tensor 样例与 loader 的合同不匹配已用官方环境复现，兼容包装输出 `[16000,14] float32`；HUGSIM sample input 与导出 scene 的区别及旧 scenario 路径后缀已修正；DriveEditor 18 个输入已结构验证但参考 crop 仍须人工审阅。生成物见[bench证据目录](autoresearch/worldsim_v75/downstream_bench/)，设计、问题发现与停止规则见[报告](v75/DOWNSTREAM_COUNTERFACTUAL_BENCH.md)。本轮0次模型前向，`failure_ledger_delta:none`。

无 GPU 阶段剩余的唯一外部下载阻塞是 DriveEditor 权重配额；道路有效与最终 case verdict 必须人工完成，脚本不会代填。GPU恢复后固定按 OmniDreams → ReSim → DriveEditor → GaussianDWM consumer → Street Gaussians → HUGSIM 推进，每个方法先跑1个合法 paired smoke；DriveEditor 只有在满足官方显存配置时进入，HUGSIM 完整闭环还需 AD client，StreetGS 若要本轮新 render 还需恢复 checkpoint/编译扩展。unsupported、consumer-only、历史复用和本轮新推理分栏，不把旧图冒充新推理。

`WS-V75-CF-DEFINITION-01 / 20260921-r1`完成质量定义；对象移除开发发现、独立确认、f=0机制诊断和来源收口均完成。未来减速新增8段、936帧、120次生成前向；定义后累计17段、1,989帧、255次生成前向，单张RTX3090无OOM。大误差源新增一次官方DVGT-1前向；资格r1因缺`iopath`在前向前结束，r2保持同一科学输入完成，不计重复或failure卡。图和轻量结果见[轨迹反事实证据](autoresearch/worldsim_v75/actor_trajectory_counterfactual/)及[对象移除证据](autoresearch/worldsim_v75/actor_removal_counterfactual/)。

此前有限来源确认已按停止规则关闭。原先 cuboid→raster 演员覆盖审计并入新 bench 的 case qualification gate，使“几何投影可见”与“状态条件真实承载目标”一致；不再作为单独主线继续筛日志、调用DVGT或启动策略反馈。若以后用新冻结协议独立复现大误差与material effect，再只对该具体状态错误接入一次实际反馈。当前不追加同源seed、事件帧、减速系数或阈值扫描。

保留V7.4 [F20](research_failures/entries/V74-H2-F20.md)、[F21](research_failures/entries/V74-H2-F21.md)、[F22](research_failures/entries/V74-H2-F22.md)边界，不恢复失败的TransFuser域基线。人工verdict均null；本轮`failure_ledger_delta:none`，不新增失败卡，不在AGENTS或FAILURES重复状态。
