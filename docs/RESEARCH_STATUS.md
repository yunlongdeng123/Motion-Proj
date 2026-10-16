# 当前研究状态

更新：2026-10-17（Asia/Singapore）。用户禁止subagent/新chat。先⑥车辆资产带道路，明确改善后才①道路细节②阴影④车身。任务WS-V82-DRIVINGEDITOR-SHADOW-REBUILD-20261017，当前run ownership_controls_r12_sam3，immutable implementation_r12。[输入、机制、组件图和协议](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_sam3_frontend_r12/)。[实验索引](EXPERIMENTS.md)。

three_control_controller_r12.py outer26840已启动有限A/B/C各3000实际更新，每组1500分段；最近快照A2250、312307点、0.359秒/步、余训练约40分钟（不含49x65渲染、C审计与传输）。此为有时间戳快照，实际用远端monitor_sam3_controls_r12.py核对PID和pipeline_state，不重复启动，不恢复旧E6或SAM2队列。每arm结束释放campaign/gpu_slot.lock，其他任务优先。C_source ready已完成真实BUILD源/坐标和远端4identity CPU检查，本地全33通过；正式C1500先全场景alpha*T审计、保留两侧的实际3D混合拆分、再次审计，再世界道路锚重分owner/精确独立motion支持的static->Actor、真实车体待定池恢复，余未知另存。只改相应owner/坐标/几何/Adam，owner和恢复步骤不减旧opacity；无关Actor邻域/旧道路参考保持，split分配光学密度。三组完整49帧后停WAITING_ROOT_FINITE_THREE_CONTROL_GATE，未资产验收，后续材质未启动。

用户更正特征DINOv3 ViT-B/16，SAM3替代SAM2/GroundedSAM2。22资产校验和实际GPU推理已通过：raw107 SAM3检出18车辆，DINOv3层7去CLS+4register得600x768有限patch；对应精度未验。原SAM3venv缺Torch，采用任务内runtime_sam3_dinov3 (Py3.12/Torch2.7.1cu126/Transformers4.56.2)，首次psutil缺包在该隔离环境修复并保留失败；高斯仍用envs/drivingeditor (Py3.9/Torch2.2)，共享环境未改。SAM3原生train46视频跟踪和3隔离QUERY全49预测保存。inputpipeline25859初次合并原标签回退出现重叠链而退出，原失败保留；CPU finalizer_r12b用同批49预测联合迭代回退已修好、root已看8行证据图并放行输入，读input_pipeline_resolution和shared_SAM3_root_review，不重跑输入管线。

冻结bundle shared_sam3_masks_r2，三组相同。a37全49 SAM3，其余显式DEVA/冲突回退和未知；1748known/2107，359unknown只排maskloss，原三147内外监督完整。R10固定损失不再调权重。A288526点/B257214点，52741训练认证道路像素，两个CPU预检过。BUILD=train46且rawrow%5!=0；raw101/113/125及其余帧20%返回只QUERY，参考和训练跟踪不看QUERY。复用原几何cache和7074独立RGB LK+真LiDAR三时刻轨迹，忽略旧语义字段。未从已绑定高斯运动自证，停车/小运动/遮挡/冲突保留待定，阴影光流不是车体。禁随机框补点、高度裁车底、只最大簇。C原未知池59208，接纳需真实body/depth或精确motion、重复渲染body父点与实际体积、无正道路冲突；未解决不作背景。

初始化B仍未通过a12保体：body0.9559→0.7059、下部0.8947→0.4737（204体点全是训练帧保留返回，testframe body分母0）；a26下部B0.9117；a37道路seedproxy3835→736但body0.9971→0.9936。此开发QUERY已看过，不能冒称盲测；B输入未事后更改，C依既定分阶段纠错设计验证。种子均值proxy/alpha或平均误差不代表正确资产，须全49动态/编辑与真实体覆盖同时过关。

旧E6@11000和旧SAM2A@1500完整保存停止，旧全部owned进程已退出，B/C SAM2未启，旧学参数不复用。历史成本含失败R8一千及未记录0..49，为13500..13549。三控9000；若⑥通过选中模型共G15k/M7.5k/J2.5k，累计上界44549。三控未明确改善时，只再一次IDSplat官方前端e3e08fc+infra：SAM3/真实LiDAR深度、保细部聚类、DINOv3真对应+RANSAC，与Waymo GT对照，GTSAM可选不盖GT；保留停车、禁随机补种，适配{局部点云,实例ID,轨迹,可信度,原始来源}及Nerfstudio相机y/z、米制/时刻、对象管理和SplatAD专有LiDAR监督差异。该最后路线先G3k，准入后总G13k/M7k/J2.4k=22400，全部资源上界44949。最后也失败则保存推送、核验所有训练/渲染/数据/评价/控制器/锁排队退出且无其他GPU任务，然后AutoDL关机并暂停心跳，不无限调参。临时空闲不关机。

delete仅准入5/6/12/18/25/26/37，相机清楚+同帧真实LiDAR充分；旧0/1/2→5/26/18，另4无旧独立资产只比旧场景外观。全49帧65流：全动态独立RGB/rawalpha、全场景可见RGB/alpha、逐Actor独立/可见、7车delete/剩余动态、旧三同轴±1m/15/30连续编辑。旧E1–E6历史390条49帧H264已逐条解码，均canonical诊断；物理阴影未训练。共同相机/GT旧版GPU重渲染仍待（native-.125px、Lanczos/CV2差2.4097/255 CPU核对），旧GN11/v2/18k只读。输入报告/旧49帧已接本地统一入口，原dirty用户工作保持。每10min检查，稳定后30/60，不变安静，仅阶段/失败/异常/需操作通知。

无独立无影/编辑GT；human_verdict=null，problem_solved=null，failure_ledger_refs=[V82-F01]，failure_ledger_delta=none。代码、下载、推理和CPU检查不是画质验收。尚需三组正式资产gate、条件最后前端，成功后①②④及公平旧版对照和最终中文交付。
