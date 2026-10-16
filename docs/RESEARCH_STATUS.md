# 当前研究状态

更新：2026-10-17（Asia/Singapore）。禁止subagent，优先⑥动态资产归属／边界，再①②④。自主实验与自检已授权；全部工作完成、保存推送且GPU/相关任务退出后关闭AutoDL，临时空闲不能关机。

任务WS-V82-DRIVINGEDITOR-SHADOW-REBUILD-20261017。R8独立道路身份短测在E4@9000产生新车尾斑块，a37 rawbodyalpha0.9676→0.9428；未采用该学习状态，全部失败证据保留。发现6921条道路投影可能被相机车体遮挡。旧“96.6%在未知mask帧”统计含该域，不能当真实可见路面污染比例，更不是真假黑影比例。详见[可见性修复与共同域](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_camera_visibility_r9/)。

R9数据data_r5_ray_visible_road拒绝潜在遮挡证据，保留245731训练道路正标签，另修正7410个新增road域像素。真实跨帧body LiDAR点只作稀疏相机深度遮挡拒绝，不补造body mask；原RGB/masks/known/depth/seed/轨迹不变。未认证／遮挡仍未知，原完整mask内外项保留。共享implementation_r8、28CPU已过。

主方法E5_ray_visible_from8k从E3@8000安全参数/Adam/RNG恢复，4731道路参考重新计算。E5_geometry_controller.json/E5_geometry.log，仅工作8000→9000，再同7预览、E5_geometry49_009000完整49帧65流、ownership_009000_r9旧域诊断并停止。common_r9_metric_controller另在锁后只读统一重评E3@8k、E4@9k、E5@9k到相同新道路域，必须一起检查，不能把过滤指标的收益当模型收益。不得误启动未训练的E5_ray_visible_s0备用fork。资产干净度尚未验收，材质未开始。

预算明确计废弃1000：工作模型几何到29000（＋废弃1000=30000资源）、材质10000至39000、联合5000至44000；工作计数＋废弃1000=总45000。后续active_experiment须保留discarded_updates_charged=1000；不自动排七消融。根代理自决无需重复授权，明显归属/几何问题先诊断，不盲跑。b-c-15先10分钟，稳定后30/60，未变化安静。

delete仅7准入车5/6/12/18/25/26/37，旧0/1/2→5/26/18；另外4无旧资产仅旧场景外观。E1–E4现有260条49帧H264已本地逐帧验证，统一入口多版本同步比较。均为canonical几何诊断，未验证物理阴影；完整动态raw/可见RGB/alpha、逐Actor及delete、旧三平移/15/30旋转均保留。原GN11/v2/18k不加载，所有父快照保留。

仍需⑥资产验收、①②④后续阶段、正式阴影/编辑视频、旧共同相机GPU只读重渲染和最终中文组件图报告。相机r2 CPU核验旧native-0.125像素、实际新Lanczos与旧CV2差2.4097/255，GPU配对待做。人工verdict=null，problem_solved=null；无无影/编辑GT，不伪造收益。failure_ledger_refs=[V82-F01]，failure_ledger_delta=none。原dirty用户工作不改。[实验索引](EXPERIMENTS.md)。
