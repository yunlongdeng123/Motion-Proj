# 当前研究状态

更新：2026-10-17（Asia/Singapore）。禁止subagent／新chat。先⑥动态资产归属和边界，后①道路细节②阴影④车身。自主实验与自检已授权，全部完成保存推送且GPU/所有任务退出后AutoDL关机，临时空闲不能关闭。[实验索引](EXPERIMENTS.md)。

任务WS-V82-DRIVINGEDITOR-SHADOW-REBUILD-20261017。当前E6_visible_identity_s0／implementation_r10／data_r5_ray_visible_road，从E5@9000模型/Adam/RNG与相同数据严格fork，只启用instance_balanced_visible_identity。已知车体可见身份内部改为逐Actor域内均值再平均，权重0.1；原完整known-HWK身份外部项公式完全不变，raw内部0.05、完整图外部0.5、认证道路0.5保留。未知mask不补标签。29CPU检查通过。[机制、公式与证据](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_visible_identity_r10/)。

E6_geometry_controller.json／E6_geometry.log，仅工作9000→10000，再E6_preview_010000、E6_geometry49_010000（49帧65流）、commonR9_E6_010000相同道路域全43身份归因与commonR10_comparison，停根代理复查。检查白车/部分轮廓可见性、a37后半段车底残片与车尾、a5/a26道路泄漏和删除背景孔洞。检查安全后推进较大几何段，不能无限1000短测，也不能盲进材质。

既有共同245731道路证据域：a37安全8k→R9@9k道路动态可见质量208.003→29.472，已知车身可见alpha0.9533→0.9484；R8失败对照车身0.9006，因新增车尾斑块未作主状态续训。R9的相机LiDAR深度代理拒绝6921潜在遮挡标签、7410新增road域像素并刷新4731参考，原RGB/body/known/depth/seed/轨迹保持，不生成mask，稀疏代理不是完整可见真值。原96.6%旧R5域统计不是真实道路污染比例。a18 rawalpha约0.97但fullscene自身贡献0.815；静态body域贡献1928（道路363），加权尺度相对锚点0.642，不能宣称泛化道路尺度膨胀或高度裁车。

预算含废弃R8的1000更新：工作G29000＋废弃1000=资源30000，M10000至工作39000，J5000至工作44000；工作＋废弃=45000。future active_experiment保留discarded_updates_charged=1000，不伪改globalstep、不自动排七消融。10分钟检查，稳定后30/60，未变化安静。材质／物理阴影尚未开始。

delete仅相机清楚且同帧真实LiDAR充分的7车5/6/12/18/25/26/37，旧0/1/2→5/26/18；其他4若无旧资产只作旧场景外观。E1–E5五版本325条49帧H264本地解码，统一入口可同步比较全动态独立/可见RGB/alpha、逐Actor与delete、旧三平移/15/30旋转。仍是canonical几何诊断，原GN11/v2/18k与所有父快照不改。camera_pairing_audit_r2已CPU核对旧native-.125px、新Lanczos与旧CV2差2.4097/255，共同相机/GT GPU只读重渲染仍待做。原dirty用户工作保留。

尚需⑥资产检查后①②④正式三阶段、物理阴影与多Actor编辑连续视频、旧版本公平对照、中文组件图报告及最终收口。无独立无影/编辑GT，不伪造收益；human_verdict=null，problem_solved=null，failure_ledger_refs=[V82-F01]，failure_ledger_delta=none。
