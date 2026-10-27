# 当前研究状态

当前任务：WS-V43-OWNERSHIP-STRUCTURE-20261028。沿用v42框架，先推进语义/实例归属诊断和车辆结构；再研究静态车辆表达分解及显露背景修复。当前诊断范围仅第一、第二开发场景；第二已完成10K。第三已在用户新指示到达前完成10K及原生/固定后端诊断，保留现场，此后禁止继续训练、编辑或研究分析。原输入三个V42 30K场景各49帧。旧队列和自动心跳未恢复，原V42 checkpoint与操作结果保留。用户本轮明确覆盖额度20%停止线，额度只记录，收到用户停止通知后保存边界。

最新用户已开GPU并授权继续第一场景诊断与算法改进。仅第一场景追加原生语义剥离310图，以及同Actor10K起点、冻结几何/DC的四组opacity控制，各新增10K，smoke+恢复额外各110更新单列；最多两个已登记子进程并行，NVML采样峰约3647MiB。当前状态与证据见[第一场景GPU报告](autoresearch/v43_ownership_20261028/FIRST_GPU_REVIEW.md)。第三场景仍暂缓，第二未新增训练，旧队列/心跳没有恢复。

内外分区监督在本开发场景改善核心覆盖并降低外泄漏；共享MLP没有额外收益证据，保留实验接口，不设为默认。原生V1无编辑通路不变，实际编辑接口与保存RGB字节差0。原生去车辆+unknown仍有暗残留及背景缺损，目标区域保留贡献多带road标签；静态外观污染/混合支撑没有解决，不整批迁移road到车辆。归属学习、补全与跨场景泛化尚未证明。

实际GPU/进程/节点只看[当前快照](autoresearch/v43_ownership_20261028/STATE.json)。已完成19类+unknown的147帧冻结SegFormer-B0推理，以及完整Gaussian行的alpha*T可见贡献、同像素实际目标LiDAR深度支持。类别预测不是GT，掩码贡献不是正确身份。大量静态目标像素贡献位于与车身不相容的中心深度，支持背景外观污染或混合支撑的候选解释，不能全部整颗搬入车辆。

结构探针从带来源的实测目标点重建Actor局部资产，复用冻结static/environment；每case fresh10K，完整fresh300s+restore300s smoke单列。原百万行自由动态分支继续作为对照。此探针改变表示、容量、颜色、损失与RGB缩放，不能冒称完整GN11继承或把原30K与新10K直接当单因素比较；同初始化无约束10K控制隔离几何预算。还没有unknown分类器学习或背景联合优化的优越性证明。

实验合同、明确差异、验收与下一步见[本轮协议](autoresearch/v43_ownership_20261028/EXPERIMENT_CONTRACT.md)；[组件图](autoresearch/v43_ownership_20261028/architecture.svg)。每case一个离线HTML，原重建、原编辑、新结构节点与19类联合着色共用帧轴，目标橙框、其他车辆蓝框；CPU/GPU标签编码分别版本化，原owner不覆盖。

逐项badcase当前状态见[六项诊断](autoresearch/v43_ownership_20261028/BADCASES.md)。已恢复默认无编辑V1原生通路，新增3米偏移、3/5/8像素unknown带与opacity控制。静态残影仍未修净，整行抑制有背景副作用；不将诊断候选冒充归属已解决。车底/阴影按用户要求暂缓。

历史完整正式10K场景数：3 / 3。这是执行完成数，quality_accepted=false，human_verdict=null。v43自身物理编辑只做目标相关诊断，旧八类物理请求及其固定ProPainter结果保留；不把新增checkpoint/版本/诊断帧重复计正式案例。

固定后端仍为同一预训练ProPainter，输入/权重/参数不变；新局部资产移动/转向/删除诊断按原请求执行，原生与生成结果分开。历史阶段二规模和封存协议见[原v42计划](autoresearch/persistent_world_20261027/PLAN.md)，未运行封存编辑选方法。

代码：scripts/worldsim_v82/v43_ownership_20261028；新campaign：/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028。模型、数据、日志不入Git。保留用户桥接脚本未提交；保留旧checkpoint spillover物理依赖。failure_ledger_refs=[V82-F01]，failure_ledger_delta=none；当前只是增加同口径诊断，不新造普遍失败结论。
