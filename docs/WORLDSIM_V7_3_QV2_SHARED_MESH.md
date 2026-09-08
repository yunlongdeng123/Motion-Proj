# Q-v2：观测查询驱动共享顶点表面

2026-09-08 22:22 UTC。首候选已从bfc181b4启动，PID96997，已完成489对象initial并进入第1轮shared_train；正式质量结果pending。三角结果见[完整报告](WORLDSIM_V7_3_TRIANGLE_RESULTS.md)：R12显著降低R10的侵入却增加miss，未同时恢复正确命中与覆盖。本轮检验共享几何支持，保持可训练原生DPT和原物理目标，不声称连通即正确。

![Q-v2实际组件](autoresearch/worldsim_v73/qv2/V73_QV2_ARCHITECTURE.png)

## 一手依据与迁移边界

[Pixel2Mesh，ECCV2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Nanyang_Wang_Pixel2Mesh_Generating_3D_ECCV_2018_paper.html)通过图网络与图像特征变形椭球。[官方实现](https://github.com/nywang16/Pixel2Mesh)是旧TensorFlow环境；本轮迁移共享顶点和逐阶段几何特征读取思想，不移植旧环境、不宣称复现其完整模型或结果。

[Mesh R-CNN，ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Gkioxari_Mesh_R-CNN_ICCV_2019_paper.html)结合粗体素拓扑与网格细化；[官方mesh_head](https://raw.githubusercontent.com/facebookresearch/meshrcnn/main/meshrcnn/modeling/roi_heads/mesh_head.py)明确按顶点对齐图像、沿边传递特征并更新顶点，训练使用完整GT网格的采样Chamfer/法向/边损失。这里只借鉴显式顶点细化，不移植体素分类或将稀疏LiDAR当完整表面。

[PyTorch3D解析球面细分源码](https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/pytorch3d/utils/ico_sphere.py)说明共享边细分与单位化的标准构造。本机采用已有SciPy与Torch自主实现解析二十面体细分，不新增依赖。ConvexHull仅用于12个解析顶点建立初始20面，不对LiDAR/预测点取凸包。与之前DMTet/FlexiCubes审计相比，直接网格避免依赖有符号场的活动支持出生；代价是固定genus-zero拓扑，不能生成任意孔洞或断开的车轮等部件，也仍会折叠/自交/塌缩。

## 实际实现

`ActorSharedMeshQueryDecoder`继承现有三层192维Query组件及4层×4局部采样，替换最终表面表示。每Actor仍有最多1024个build LiDAR与512个原生深度支持查询；原生候选不足时沿用LiDAR/coarse fallback，这些查询只提供观测隐状态，不各自产生输出小片。

新增642个具有真实三维位置的表面顶点，以只读Actor尺寸缩放解析球面初始化；1280个三角面的顶点索引共享且固定。每层沿网格边聚合，并独立读取最近16个观测查询，再与精确投影的多尺度局部视觉读取联合更新顶点。观测查询也继续更新；离散近邻选索引不反传，但所选原生坐标、特征与消息保留梯度。两类消息各取平均后等权组合，避免网格顶点挤出稀疏观测。输出同一组顶点/三角面给coverage、beam free、框包络和硬首交点，没有opacity/existence/半径头、筛掉违规面的后处理或训练/导出换三角化。

几何监督仍为稀疏测量→表面单向coverage、原始束beam free与直接build轴向深度Huber；未知表面不惩罚为不存在，UNKNOWN不标FREE。固定闭合拓扑是一项可失败的形状先验，不能作为内部占据真值。原始束只监督有效首返回前；不添加体积正厚度。首轮不叠加ARAP、Laplacian或新loss，以先识别参数化的实际效果。

可训练Query参数1668393（旧1670517），原生DPT32654562不变，早期/跨视图聚合仍冻结。输出面数1280，旧片为8×(512+min(build,1024))，约4104–12288；总隐查询数为642+512+min(build,1024)，最高2178，旧最高1536。面数、连续支持、初始化与邻接同时属于此次参数化迁移，不能说是面密度/初始化/纯拓扑完全匹配的单模块因果实验。若观察到收益，后续再做同解码器LiDAR控制及必要的预算/来源判别，不以无止境分辨率网格拖延。

## 正式运行登记

- task：`WS-V73-Q-V2-01`，run：`20260908T221500Z__shared-mesh-full-track-beam-s7304-r1`；状态running（PID96997，初始化评价完成，现第1轮真实训练）。
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
