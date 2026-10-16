# 当前研究状态

更新：2026-10-17（Asia/Singapore）。禁止subagent／新chat。先⑥动态资产归属和边界，后①道路细节②阴影④车身。自主运行与自检已授权；全部完成保存推送且GPU/所有训练/评价/数据/渲染/控制器/锁排队任务退出后AutoDL关机，临时空闲不能关闭。[实验索引](EXPERIMENTS.md)。

任务WS-V82-DRIVINGEDITOR-SHADOW-REBUILD-20261017。主方法E6_visible_identity_s0／implementation_r10／data_r5_ray_visible_road，已完成10000步复查，继续相同几何10000→20000。continue_geometry_gate.py／E6_gate20000_controller.json／E6_gate20000.log，launch flockPID21404，train21407（须实际核验）。父E6/gate_010000.pt保留，预计约77分钟训练＋3分钟完整49帧和归因，之后E6_preview_020000、E6_geometry49_020000、commonR9_E6_020000停根代理复查，不能自动进材质。稳定后巡检已由10改30分钟，异常缩短，状态不变安静。29CPU检查通过，代码97dee0a已推且引用源码核验。[机制、10k复查与来源诊断](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_visible_identity_r10/)。

10k实际49帧6/7车身训练MAE改善；共同245731真实道路域a18body可见alpha0.815→0.9194，a37为0.9484→0.9605，道路质量29.47→22.27。a12为0.8244→0.8061且MAE0.00809→0.00846略退，仍部分/污染；白车锯齿、a5/a26条片与a37后段环片未全解决，资产未验收。新监督仅平衡已知身份内部域，原完整known-HWK身份外部公式EXACT保持、权重0.1；raw内部0.05／完整图可见外部0.5／认证道路0.5不变，未知403不补标签、旧三147完整内外监督保留。

冻结10k只读CPU来源复查完成：25个重复body/road局部可分混合体积、353个精确真实道路seed源且世界>=3时刻支持、4sigma车体排除/重复RGB车体保护后的检查提名。尚未修改模型，也非归属真值。到20k须重新计算，不使用旧行号；检查源帧/点、车体轮胎和零更新拆分/实际世界道路重锚对照，防止只改善原视角而编辑后仍带路面。没有来源支持或不可见不能判背景，不按高度剪。此检查可与继续29k几何结合，材质前先资产质量关卡。

此前R8E4@9k出现新车尾斑块，失败参数未作主状态续训，证据全保留；R9从E3@8k恢复Adam/RNG，拒绝6921潜在相机遮挡道路标签、7410新增road域像素，刷新4731参考。世界道路LiDAR不是相机可见道路；稀疏body点遮挡代理不是完整真值。原96.6%旧投影域统计不是真实道路污染或黑影真假比例。原GN11/v2/18k不载入，新模型原测量谱系。

预算含废弃R8的1000更新：工作G29000＋废弃1000=资源30000，M10000至工作39000，J5000至工作44000，工作＋废弃=45000。future active_experiment保留discarded_updates_charged=1000，不伪改globalstep、不自动排七消融。M/J吞吐需按450k点真实更新；初始测速仅参考。

delete仅相机清楚且同帧LiDAR充分的7车5/6/12/18/25/26/37，旧0/1/2→5/26/18，其余4无旧独立资产仅旧场景外观。E1–E6六版本390条49帧H264本地逐条解码，统一入口可同步比较全动态独立/可见RGB/alpha、逐Actor/delete、旧三同轴平移/15/30旋转。仍为canonical几何诊断，物理阴影未训练。旧共同相机GT实际GPU重渲染仍待，旧native-.125px、新Lanczos与旧CV2差2.4097/255已CPU核对，不冒称公平像素收益。原dirty用户文件保留。

尚需⑥资产检查后①②④三阶段、物理阴影与多Actor编辑连续视频、公平旧版本对照、中文组件图报告和最终收口。无独立无影/编辑GT；human_verdict=null，problem_solved=null；failure_ledger_refs=[V82-F01]，failure_ledger_delta=none。
