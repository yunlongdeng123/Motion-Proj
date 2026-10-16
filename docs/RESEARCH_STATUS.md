# 当前研究状态

更新：2026-10-17（Asia/Singapore）。最高优先⑥动态资产混入道路（P1归属／边界），解决后①道路细节②车辆关联阴影④车身外观；禁止subagent。用户授权自主调研、实验、自检，完成全部工作并保存推送、确认GPU及相关任务退出后才关闭AutoDL。暂时空闲不能关机。

任务WS-V82-DRIVINGEDITOR-SHADOW-REBUILD-20261017。E0@5000和E1归一化短对照@6000均未通过资产干净度；后者车外残片变弱，但部分车身误差增加。原始测量初始化谱系不读取GN11或旧18k。已做全场景alpha*T的46训练帧逐点审计，补齐旧OBB guard空缺区域的真实LiDAR道路表面正证据。62跨边界父点拆为124保留子点；完整43身份复审后，426道路点锚定实际世界LiDAR测量，混合/未知保留，不做车底高度裁剪。详细[归属组件及证据](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_ownership_r6/)。

E2_ownership_s0使用implementation_r6、data_r3_ownership，保留原图/mask/轨迹，仅增加认证训练道路像素。完整外部域监督改为全场景可见贡献，以保护被遮挡真实车体；内域仍raw alpha，未知403对无mask梯度；数值权重与变更均显式登记。25CPU检查通过，未代表画质通过。当前只从global6000优化1000步至7000后停，控制器E2_geometry_controller.json，日志E2_geometry.log，active_experiment.json可读。到7000根代理检查同7帧、完整49帧65流动态/编辑与归属残留，再决定后续，不能直接进入材质。

delete准入仅7个索引5、6、12、18、25、26、37，须同时清楚RGB与同帧真实LiDAR；不能扩至43。旧三0/1/2对应5/26/18，连续同轴±1m平移及15/30度旋转；另四如无旧资产仅旧场景外观参考。动态独立RGB/rawalpha、全场景动态可见RGB/alpha、逐Actor及编辑动态视频必须交付。静态分支单独渲染仅诊断。

主预算45k包含已用6000；其余几何/材质/联合分阶段，消融接口保留但不自动排七臂。b-c-15先10分钟，稳定后30/60分钟，查PID/日志/数值与阶段，不重复启动，其他GPU任务优先。仍需正式完整视频、旧共同相机/GT只读配对渲染、阴影阶段和最终报告。相机协议r2已CPU核验：旧native-0.125像素、新RGB实际Lanczos与旧CV2均差2.4097/255；原生结果保留。

人工verdict=null，problem_solved=null；无独立无影或反事实编辑GT，不伪造。failure_ledger_refs=[V82-F01]、failure_ledger_delta=none。原dirty用户工作与所有历史失败证据保留。[实验索引](EXPERIMENTS.md)。
