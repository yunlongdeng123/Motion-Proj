# V7.3 AV2逐点时空几何接口

## V7.3 旧AV2完整Actor固定推理结果（2026-09-08）

code0a610555下已完成3项旧开发窗口真实评价，每项均覆盖21Actor/1日志/4移动对象、保留2个空输入和1个无自有留出返回，不训练、不卡点删对象、不读新20日志作方法选择。任务WS-V73-M4-AV2-FIXED-01：原生融合run `20260908T013000Z__old-development-native-fusion-r1`，53.7060s/GPU allocated1.29079GiB/RSS3.47623GiB；LiDAR PCA run `20260908T013100Z__old-development-lidar-pca-r1`，29.7731s/0.02101GiB/1.00754GiB；固定LiDAR-only r9 run `20260908T013200Z__old-development-lidar-event-r9`，43.3979s/0.05542GiB/1.17406GiB。三个进程全部正常结束，完整surface/frames保存在各run。

| 固定方法 | hit | early | miss | free m | target→surface m | recall@0.2m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LiDAR PCA | 32.7578% | 3.6863% | 62.5793% | 0.103598 | 0.122136 | 75.5924% |
| M1r3原生融合 | 29.1706% | 8.5989% | 60.3500% | 0.489094 | 0.115922 | 77.2792% |
| 固定LiDAR-only r9 | 42.0909% | 7.7867% | 43.4908% | 0.180803 | 0.080236 | 82.6710% |

原生融合略增覆盖/降低一向距离但显著增加已观测自由空间侵入；LiDAR-only r9降低缺失/增加命中同样付出侵入代价。仅一条旧开发日志，不能给出有意义的日志bootstrap区间，不能当独立新日志或跨域成功结论。unknown区域没有按稀疏target惩罚；本表距离是一向target→surface，不是完整Chamfer。证据 `coverage/av2_old_fixed_r1_analysis.json` 保留原始Actor统计、共有输入/移动分层；摘要中的单日志bootstrap退化值不作不确定性推断。

原生路径已经真实读取七相机×四时刻和完整有效画幅，固定DPT实时解码、padding排除及规范表面读出可执行。此处native_fusion不消费相机ID嵌入，LiDAR-only也不消费视觉；七相机embedding插值和视觉空间交互仍需完成joint权重在旧窗口上验证，不能把这三项称为已验证完整视觉query跨域迁移。新20日志仍无模型推理/评价。

当前三项长期训练正常：r5恢复PID39009在epoch22，峰值allocated10.19917GiB；native-only full_track r11 PID38460进入epoch1真实反向，原生头梯度有效/query梯度0，峰值2.34640GiB；Ada Y-up r2 PID37887进入epoch7，峰值1.00312GiB。mmap恢复进程持续推进；cgroup OOM计数仍0。r10同full_track joint在恢复主模型结束后调度，尚未启动，无后台启动队列。后续将完整主模型/强基线的表面放回同一背景比较，再做选定模型的新日志确认，不提前关闭研究。

failure_ledger_delta=update F01/F02/F04/F05/F08执行与旧域失败证据；F01/F02/F03/F04/F05/F07仍active，F06直接数据配置缓解，F08恢复执行有效但退出原因未明，下一编号V73-F09。整个V7.3未完成，shutdown=false；全部研究收尾保存/push且无训练、评估、数据和待启动任务才关机。

---



## 完整28视图前缀已完成（2026-09-08）

`WS-V73-M4-AV2-PREFIX-01/20260908T011000Z__old-development-28view-prefix-r1` 成功完成，code9541ac7d，33.9146s，GPU allocated峰值7.64359GiB、RSS7.60567GiB，冻结四层缓存2118584384字节。全部28视图672×672共同进入跨视图聚合层，仅逐图patch编码按7图分批。原始头与build325203个对应估计固定IRLS scale26.60637，之后装入M1r3适配头，优化更新0。只缓存完全冻结前缀，没有将其误称上层PEFT。

官方及当前源码已经选择保存DPT层4/11/17/23，复用这一既有能力，不能把它列成新增创新或本次独立内存增益。每视图CPU token clone后保存，避免序列化整窗口storage。依据[VGGT官方aggregator](https://raw.githubusercontent.com/facebookresearch/vggt/main/vggt/models/aggregator.py)。

新增固定Actor评价入口 `scripts/evaluate_worldsim_v73_fixed_actors.py`，供已完成共享模型、原生融合或LiDAR PCA使用；输入角色、空输入对象、7相机权重和padding矩形均显式保留，无优化或目标初始化。尚未运行固定模型神经评价；冻结前缀成功不能证明query迁移、跨域几何或新20日志效果。先在此旧开发窗口运行，再用最终选定方法处理外部确认集。


2026-09-08。实现 `motion_proj/worldsim_v73/av2_geometry.py`，仅在旧开发日志02678d04-cc9f-3148-9f95-1ba66347dff9执行真实数据诊断。20条新确认日志的760文件已传输完成（977.74s，s5cmd返回0，下载PID34904退出），没有运行其网络推理或读取其中传感器数值进行方法选择。

依据[官方Sweep接口](https://github.com/argoverse/av2-api/blob/main/src/av2/structures/sweep.py)及[官方SE3读取约定](https://github.com/argoverse/av2-api/blob/main/src/av2/utils/io.py)，每点时间为扫描参考纳秒时间加offset_ns；发布的坐标已补偿到参考时刻ego系。因此世界端点只乘一次参考ego姿态，不能再次按逐点ego变换移动端点；束原点则取逐点时刻的ego姿态与其物理LiDAR外参：

\[
x_w=T_{w\leftarrow ego}(t_0)x_{ego(t_0)},\qquad
o_w(t_i)=T_{w\leftarrow ego}(t_i)o_{sensor,ego},\qquad
d_i=\|x_w-o_w(t_i)\|.
\]

Actor规范坐标用只读轨迹在同一个t_i的逆变换处理端点、原点与方向；相机投影用该相机实际曝光t_v的ego与Actor位姿。`TimedPoses`先用int64减去时间原点，再转相对秒做平移插值和Slerp；不将大绝对纳秒先转float。越界位姿返回独立known掩码，数值占位虽夹到边界但不能作为已知pose使用。Actor标注原本在对应时刻ego系，先转city系再插值；num_interior_pts不参与输入或选择。

旧开发log扫描序号5有94860个原始返回，实际offset为0.615–106.315ms，ego位姿覆盖全部返回。忽略逐点时间、只用参考扫描时刻的两个原点，与逐点原点之差中位0.59106m、p95 1.07697m、最大1.15741m。这是此旧日志的几何诊断，不是全AV2误差估计，也不是模型精度增益。

官方文档/API确认laser_number为0–63、传感器有up/down，但本次检索未找到明确的编号组映射声明。为避免默默假设，诊断在相同原始返回和固定标定下比较两种映射的逐ring仰角离散度：0–31→up时，各ring角度MAD的中位为0.011554°；交换后为0.126340°。据此选0–31为up、32–63为down，记录为旧开发数据支持的推断，代码仍显式暴露lower_ids_sensor参数。没有按新确认日志单独重估映射或按模型质量选传感器约定。

七路图像全部保留，不裁剪视场；当前接口按最长边672缩放并放入672×672画布，内参同步应用缩放/平移。旧日志前相机有效矩形[81,0,590,672]，其他六台[0,81,672,590]；记录实际时间和光轴方向。七台光轴方位约0.20°、45.11°、−44.91°、99.48°、−98.89°、153.31°、−152.94°，不能把nuScenes的六项相机ID直接索引成七项。完整扫描、每ring及每camera结果在 `docs/autoresearch/worldsim_v73/coverage/av2_geometry_development_r1.json`。CPU诊断0.99s、RSS0.22056GiB，没有GPU/网络模型测试。

## 共同输入导出与七路接口（2026-09-08后续里程碑）

`scripts/prepare_worldsim_v73_av2_actors.py` 已导出同一旧开发日志的完整四build/两留出窗口：run `WS-V73-M4-AV2-DATA-01/20260908T003500Z__old-development-window-r1`，41.84s、RSS0.87773GiB，28视图、562494原始返回（375000 build、187494留出），全部有传感器姿态。依据build时刻已知轨迹选21辆车，其中19个ready、2个空输入仍保存case；4辆速度>2m/s。一个空输入对象有3个留出自有返回，不能在评价时因缺build输入删除。一个仅单时刻轨迹对象没有相机时刻/逐点时刻可用pose，也保留缺失状态。全部21对象和每scan概况在 `coverage/av2_old_window_r1_index.json`。

build_observations与case保存原始点、逐点真实束范围/归属/规范原点、只读相机姿态、内参和完整view_indices。表面输入只来自4个build扫描；两次留出射线独立保存，不进入输入支持。归属按每点真实时间的全部已知类别框判断，重叠剔除为ambiguous；非刚体返回不作为车辆视觉监督。已知轨迹的每camera姿态不依赖该scan是否碰巧命中该Actor。

针对相机数量改变先检索[CVPR2023 CAPE](https://openaccess.thecvf.com/content/CVPR2023/html/Xiong_CAPE_Camera_View_Position_Embedding_for_Multi-View_3D_Object_Detection_CVPR_2023_paper.html)与[ICCV2023 PETRv2](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_PETRv2_A_Unified_Framework_for_3D_Perception_from_Multi-Camera_Images_ICCV_2023_paper.pdf)，保留相机局部坐标及显式跨时刻姿态对应的方向。此处为现有模型接口迁移，并非复现其检测方法或借其结果证明重建有效。

新增可选camera_embedding_weights：仅根据标定方位在原六项已训练嵌入之间作周期线性插值，保留七路独立图像、投影、方向与时间，不加第七项随机参数，不对图像特征先跨视角平均。原六项方位为[0.45,−56.99,−111.71,179.64,108.59,55.24]度，由nuScenes标定循环均值获得。插值只处理离散ID域接口，不能保证学习到的ID嵌入天然随方位平滑；这项假设需旧AV2开发真实推理检验，最终外部数据不用于选择映射。

`ProjectedLocalRead` 对query投影和每个偏移采样都使用有效像素矩形，`native_surface_points/seeds` 同时排除padding像素。共同训练器与简单fusion入口已接入这两个可选字段；旧nuScenes case没有字段时维持原计算路径。汇总器保留external_confirmation为独立角色，不会默默落掉新域或冒称development。当前没有新增GPU测试，运行中的r5/CAPA/r9仍使用已载入代码；完整28视图神经路径的资源和效果尚未测得。

目前新20日志仍只有原始载荷，没有神经推理或外部评分。下一步准备旧AV2窗口的冻结前缀与固定模型推理，实际验证padding/七相机/几何梯度使用，再进行最终选定方法的新日志应用。F04/F05继续active，整个V7.3未完成。
