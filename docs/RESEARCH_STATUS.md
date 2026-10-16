# 当前研究状态

更新：2026-10-17（Asia/Singapore）。禁止subagent，优先⑥动态资产道路归属／边界，再①②④。用户授权自主实验、自检；全部工作完成、保存推送且GPU及相关任务均退出后关闭AutoDL，临时空闲不能关机。

任务WS-V82-DRIVINGEDITOR-SHADOW-REBUILD-20261017。E3@8000源点修复后，根代理同7帧复查看到6/7车身误差下降、a37略升，车底残片未完全消除。全49帧65流及逐点审计完成。此前E1/E2两组130条49帧H264已本地交付并解码验证，统一入口有同步比较页；均为canonical几何诊断，材质／阴影未开始。

8k诊断揭示a37约96.6%实测道路动态贡献在未知实例mask帧。R8没有补造mask，使用独立真实LiDAR道路正证据：世界>=3时刻、自身局部平面法线、真实车体3D接触保护、深度及无已知body重叠，得到252652训练像素。原RGB/masks/known/road/depth/seed/轨迹不变，只追加road_certified。原完整mask内外项保留，新增道路域mean((sum43可见Actorα)^2)权重0.5；未认证未知不贴标签。28CPU通过。静态alpha均值0.9815，95.87%>0.9，仅0.2719%<0.5，允许短探测并继续查孔洞／车体。详见[独立道路证据与组件](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_road_identity_r8/)。

当前E4_road_identity_s0/implementation_r8/data_r4_road_identity，E4_geometry_controller.json、E4_geometry.log，仅8000→9000。随后自动E4_preview_009000、E4_geometry49_009000完整49帧、ownership_009000_r8全场景43身份审计，停止等待根代理复查。不得重复启动或盲进材质。几何累计只到30000（含对照／修复），材质10000、联合5000，总45000不变；不自动跑七消融。b-c-15先10分钟，稳定后30/60，非必要不通知。

delete准入仍仅7辆5/6/12/18/25/26/37，旧0/1/2→5/26/18；另四无旧资产仅旧场景外观。辅助403未知mask无原maskloss，旧三147完整域保留。原GN11/v2/18k不进入v3初始化，各父checkpoint和测量/失败证据不删。原dirty用户工作保留。

仍需⑥资产干净度、①②④后续阶段、完整正式阴影与连续编辑视频、旧共同相机只读GPU对照和最终中文图文报告。camera_pairing_audit_r2已CPU核验：旧native-0.125像素、新实际Lanczos与旧CV2差2.4097/255，GPU配对重渲染待做。人工verdict=null，problem_solved=null；无独立无影/反事实GT，不伪造收益。failure_ledger_refs=[V82-F01]，failure_ledger_delta=none。[实验索引](EXPERIMENTS.md)。
