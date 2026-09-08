# Q-v2：观测查询驱动共享顶点表面

2026-09-08 23:10 UTC。联合r1（code bfc181b4、PID96997）与同网格LiDAR r2（code95050522、PID98643）均在正式训练，质量结果pending。r2在23:08快照为第4轮、1459次真实更新；两支资源正常。三角结果见[完整报告](WORLDSIM_V7_3_TRIANGLE_RESULTS.md)：R12显著降低R10的侵入却增加miss，未同时恢复正确命中与覆盖。本轮检验共享几何支持，保持可训练原生DPT和原物理目标，不声称连通即正确。

![Q-v2实际组件](autoresearch/worldsim_v73/qv2/V73_QV2_ARCHITECTURE.png)

## 一手依据与迁移边界

[Pixel2Mesh，ECCV2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Nanyang_Wang_Pixel2Mesh_Generating_3D_ECCV_2018_paper.html)通过图网络与图像特征变形椭球。[官方实现](https://github.com/nywang16/Pixel2Mesh)是旧TensorFlow环境；本轮迁移共享顶点和逐阶段几何特征读取思想，不移植旧环境、不宣称复现其完整模型或结果。

[Mesh R-CNN，ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Gkioxari_Mesh_R-CNN_ICCV_2019_paper.html)结合粗体素拓扑与网格细化；[官方mesh_head](https://raw.githubusercontent.com/facebookresearch/meshrcnn/main/meshrcnn/modeling/roi_heads/mesh_head.py)明确按顶点对齐图像、沿边传递特征并更新顶点，训练使用完整GT网格的采样Chamfer/法向/边损失。这里只借鉴显式顶点细化，不移植体素分类或将稀疏LiDAR当完整表面。

[PyTorch3D解析球面细分源码](https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/pytorch3d/utils/ico_sphere.py)说明共享边细分与单位化的标准构造。本机采用已有SciPy与Torch自主实现解析二十面体细分，不新增依赖。ConvexHull仅用于12个解析顶点建立初始20面，不对LiDAR/预测点取凸包。与之前DMTet/FlexiCubes审计相比，直接网格避免依赖有符号场的活动支持出生；代价是固定genus-zero拓扑，不能生成任意孔洞或断开的车轮等部件，也仍会折叠/自交/塌缩。

## 实际实现

`ActorSharedMeshQueryDecoder`继承现有三层192维Query组件及4层×4局部采样，替换最终表面表示。每Actor仍有最多1024个build LiDAR与512个原生深度支持查询；原生候选不足时沿用LiDAR/coarse fallback，这些查询只提供观测隐状态，不各自产生输出小片。

新增642个具有真实三维位置的表面顶点，以只读Actor尺寸缩放解析球面初始化；1280个三角面的顶点索引共享且固定。每层沿网格边聚合，并独立读取最近16个观测查询，再与精确投影的多尺度局部视觉读取联合更新顶点。观测查询也继续更新；离散近邻选索引不反传，但所选原生坐标、特征与消息保留梯度。两类消息各取平均后等权组合，避免网格顶点挤出稀疏观测。输出同一组顶点/三角面给coverage、beam free、框包络和硬首交点，没有opacity/existence/半径头、筛掉违规面的后处理或训练/导出换三角化。

几何监督仍为稀疏测量→表面单向coverage、原始束beam free与直接build轴向深度Huber；未知表面不惩罚为不存在，UNKNOWN不标FREE。固定闭合拓扑是一项可失败的形状先验，不能作为内部占据真值。原始束只监督有效首返回前；不添加体积正厚度。首轮不叠加ARAP、Laplacian或新loss，以先识别参数化的实际效果。

可训练Query参数1668393（旧1670517），原生DPT32654562不变，早期/跨视图聚合仍冻结。输出面数1280，旧片为8×(512+min(build,1024))，约4104–12288；总隐查询数为642+512+min(build,1024)，最高2178，旧最高1536。面数、连续支持、初始化与邻接同时属于此次参数化迁移，不能说是面密度/初始化/纯拓扑完全匹配的单模块因果实验。同解码器LiDAR控制现已提前安排，便于主候选结束时区分视觉通路与参数化本身；必要的预算/来源判别依据结果再定，不以无止境分辨率网格拖延。

## 正式运行登记

- task：`WS-V73-Q-V2-01`，run：`20260908T221500Z__shared-mesh-full-track-beam-s7304-r1`；状态running（PID96997，初始化评价完成，30轮正式训练继续；逐步状态见run/status.json）。
- 依据提交07e98727；启动manifest记录实际实现commit。M1r3 fresh DPT、Query seed7304，30轮、371 FIT Actor/20日志；489对象完整评价、75 DEV/5日志含51总不可用对象。原窗口24视图与分辨率不变。
- 同R12的full_track FIT标签、native1/free.5/beam width.03m/res32/event0、AdamW1e-5与clip1；不从R12 checkpoint恢复、不改visual-only输入cohort、不加入上层LoRA或新局部对应/near-boundary目标。
- 新网格initial必须实际评价；旧LiDAR PCA baseline复用R10保存结果。最终主要比较Q-v2−R12，再以R8/R14定位物理控制边界。仍按独立日志配对，报告hit/early/miss/free/距离/召回与表面预算；未成功返回也保留。FIT不是泛化证据，移动子集2日志的稀疏限制继续。
- 启动：`scripts/run_worldsim_v73_qv2_shared_mesh_r1.sh`；沿用现有训练器、逐轮checkpoint与状态原子写入。20新日志质量不提前读取。资源依据真实训练峰值判断，不能把下面合成检查当完整DPT资源估计。

## 必要接口验证

一次合成检查验证642顶点/1280面、每条拓扑边归属2个面；轴向首交点[2,2,1.2]m符合解析边界。使用相同最近面算子反传后，四尺度视觉梯度范数为[1.45348e-5,8.12089e-6,5.26674e-6,1.04275e-5]，原生种子梯度1.73559e-6，说明新顶点确实可接收两类几何信息。此为可微接口证据，不是真实数据质量，也不是禁止未来无有效视图/零梯度的门控。

初次合成数据使用4×4最粗特征图，既有约3.05特征像素初始offset使候选全部越界，因此“所有尺度都必须正梯度”断言失败；核对现有valid mask及[PyTorch grid_sample边界说明](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html)后，改用12/24/48/96合成图重试通过。模型mask/offset未修改，没有放宽真实数据输入条件；记录为检查输入修复，无新增科学失败ID。只重试这一失败检查，未扩展smoke/回归。

代码：`motion_proj/worldsim_v73/shared_mesh_queries.py`；证据：`autoresearch/worldsim_v73/qv2/shared_mesh_path_check.jsonl`、组件图及源码。failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F04,V73-F06,V73-F09]；failure_ledger_delta=none（候选结果pending，F02未解决），下一V73-F10。

## 后续判别

如果连通表面减少missing却增加early，检查实际网格位置/折叠与certified-free边界梯度，而非以闭合性当真值；如果训练几何仍难恢复，区分上下游支持与视觉对应，按已准备的边界独立适配VGGT上层。若物理/覆盖同时改善，继续同信息控制、完整输入与背景合成，再选择新日志确认。用户提出的near-surface certified-free与calibration-guided consistency可行性已写计划16节，均保持独立机制，不混入首轮。

## 同共享网格的 LiDAR-only 控制（2026-09-08 23:00 UTC）

主联合Q-v2 r1已到第3轮，物理/覆盖结果仍pending。为防止把椭球初始支持或共享网格本身的收益误归于视觉，补齐同一`ActorSharedMeshQueryDecoder`的LiDAR-only r2；这是计划中的同解码器控制，提前并行训练以在主候选结束时取得可解释比较。旧R8采用独立patch，不能代替此项。

task `WS-V73-Q-V2-01` / run `20260908T230000Z__shared-mesh-lidar-full-track-beam-s7304-r2`，已从95050522启动，PID98643；23:08 UTC第4轮1459次更新，allocated峰值0.252920GiB。仍用原371 FIT/67 DEV可用输入、完整489对象、seed7304、30轮/full_track/AdamW1e-5/clip1、642顶点1280面、相同尺寸椭球与共享边。build LiDAR的最多1024证据查询+512支持查询输入不变；512支持来自同build LiDAR，关闭图像/DPT/视觉特征读取，native-data-weight=0。coverage、beam free .5/.03m/res32、envelope .05、event0保留。

此对照同时移除DPT适配、原生深度支持、视觉读取和native深度辅助监督，是视觉几何整通路控制，不是仅一个attention token的消融。相同seed不意味着不同分支消耗的逐步CUDA随机数相同，故不称逐步采样完全匹配。两支具有相同网格输出与Query参数定义；LiDAR模式未调用的视觉模块虽仍存于state_dict，不会接收梯度或Adam更新。其summary的requires_grad参数数不能冒充每个参数都参与训练，依据实际梯度路径解释。

新initial必须实际评价，不能复用joint初始网格预测（观测隐状态不同），固定LiDAR PCA复用R10保存结果。先汇总r2−R8判断LiDAR参数化变化，再在两者都完成后汇总r1−r2，比较hit/early/miss/free/测量距离/召回；完整分母与独立日志配对保持。R12/R14已保存的基线继续作为外部通路锚点。分析脚本只新增保存真实surface_vertices/faces/parameterization字段及对照路径元数据；不改变任何指标、聚合、bootstrap或旧结果，不回算历史。

前述“不等主候选结果做完整矩阵”的取舍仍成立：当前仅增加这一必要控制，不启动其他拓扑/密度/损失网格。可训练视觉几何r1主线继续，LiDAR控制不替代它。23:00前资源为GPU14412/24576MiB，主allocated11.591GiB、RSS34.08GiB、cgroup无OOM、磁盘68GiB可用；LiDAR路径不加载DPT/744前缀，沿用已完成R7/R8的低显存执行方式，实际新增峰值在启动后记录。两个现有6线程进程在14CPU配额内；耗时均披露并行竞争。

证据和启动入口`scripts/run_worldsim_v73_qv2_lidar_mesh_r2.sh`，源机制复用本页已核实Pixel2Mesh/Mesh R-CNN与原项目LiDAR路径，无新卡点、无新依赖/单独smoke/回归。failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F04,V73-F05,V73-F06,V73-F09]；failure_ledger_delta=none，正式结果pending，下一V73-F10。20新日志未读、30分钟ACTIVE、完成不关机。

## 固定推理与收口边界（2026-09-08 23:10 UTC）

`evaluate_worldsim_v73_fixed_actors.py`现按checkpoint配置选择共享网格或旧patch，保持相同build-only输入、无梯度、真实三角面硬读出与全部不可用对象分母。新网格保留642顶点/1280面原输出，不能重解释为642个独立patch，也不重网格化。已仅在CPU成功加载r2真实完成轮次checkpoint（全部keys匹配）；完整固定推理尚未执行，20新日志质量未读。checkpoint类加载只是接口证据，不能替代正式模型效果。

收口命令为`bash scripts/summarize_worldsim_v73_qv2.sh lidar_r2`与`... joint_r1`；分别在相应最终489对象评价完成后各执行一次，后者须等两支都完成。前者比较r2−R8；后者比较r1−r2/R12/R14/R8。原指标与统计脚本复用，不回算历史比较；保存训练曲线、配对图、原summary及最终manifest。

旧`analyze_worldsim_v73_query_provenance.py`与`analyze_worldsim_v73_surface_failures.py`按每patch8面/9顶点归属来源，不能用于本表示。Q-v2的source=2只说明输出来自共享网格顶点，不能给混合观测状态硬分LiDAR/视觉因果来源。`analyze_worldsim_v73_surface_support.py`仅依赖顶点/面，可按final失败模式使用，但闭合表面常有正常入/出两个交点；深度层数不是自交证明。主评价仍报告真实early/hit/miss/free与测量覆盖。

23:08运行证据归档`autoresearch/worldsim_v73/qv2/shared_mesh_lidar_r2_training_start.json`，r2完成初始化489对象，1459/1459更新，非零Query梯度、DPT梯度0；原views字段仅元数据，不代表LiDAR分支读取图像。总GPU15163/24576MiB、无OOM，两支继续。failure_ledger_delta=none，F02等待最终对照，不新增失败ID、不额外做smoke/回归。
