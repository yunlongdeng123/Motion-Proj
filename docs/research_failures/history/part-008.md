# 历史原始记录 008

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V7.3 构建束自由空间背景雕刻对照登记（2026-09-08）

三项长期训练正常：r5恢复已进入epoch23，native-only r11进入epoch2，Ada Y-up r2进入epoch14；未新增GPU并发，未读新20日志评价。现有场景背景自身的cohort early约27%且free约1m，足以遮蔽Actor增益，F04需直接处理构建侧几何一致性。

先检索[经典范围图空间雕刻](https://lightfield.stanford.edu/papers/volrange/)、[Voxblox自由空间/全束积分](https://voxblox.readthedocs.io/en/latest/pages/The-Voxblox-Node.html)及[Open3D多交点API](https://www.open3d.org/docs/latest/cpp_api/classopen3d_1_1t_1_1geometry_1_1_raycasting_scene.html)，再迁移为当前显式三角面的轻量背景对照，不声称复现TSDF或新方法。

新增 `scripts/carve_worldsim_v73_background.py`：以相同r2近传感器修订背景为输入，只读取原四个build扫描的有效原始首返回和已知扫描时刻标定，查询每束全部返回交点；若0<t<d_build−0.2m则删除被矛盾证据命中的背景三角面。首回波之后和未观测方向仍为未知，不移动Actor，不改轨迹，不通过opacity隐藏错表面，也不按heldout值删支持。保留所有heldout文件/归属/位姿为原文件链接。删除整片是离散近似，可能损失覆盖；必须同时报告hit/miss/early/free和边界分层。

Open3D官方实现会合并部分等深重合交点，单次list_intersections并不保证所有重合三角面被清除。只作一次雕刻，并报告剩余build early/free实际读数，不以循环门控或假定零违规代替结果。分块8192射线只控制内存，全部build束保留；不人为扩张光束宽度或把unknown变成free。代码未运行，效果待测。

登记数据 `WS-V73-M4-SCENE-DATA-01/20260908T014500Z__development-build-free-carved-r3`。配套同模型场景比较：原r2背景run `WS-V73-M4-SCENE-COMPOSITION-01/20260908T014500Z__development-event-capa-original-bg-r4`，雕刻背景run `20260908T014500Z__development-event-capa-carved-bg-r5`；包含同一PCA、r8、r9、CAPA r2、Ada r1和native fusion r2显式Actor表面。两边不重新网络推理，用相同416704原始heldout束，CPU BVH完成全局首交点；先回答背景变化是否减少侵入及代价，再观察event/强基线趋势。Ada r1轴限制仍明确，不替代正在训练的r2。均尚未启动。

failure_ledger_delta=update F04的构建证据对策；F01/F02/F03/F04/F05/F07仍active，F06直接数据配置缓解，F08恢复推进但退出原因未知，下一编号V73-F09。整个V7.3未完成，shutdown=false；正常训练继续，不把此CPU对照结束当作整体结束。

---



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



## V7.3 joint已恢复真实训练与固定Actor评价入口（2026-09-08）

恢复run `WS-V73-M2-GLOBAL-ACTOR-01/20260908T012500Z__population-joint-r5-epoch21-resume-r1` 已实际运行，code914d582d、PID39009，日志 `/root/autodl-tmp/controller_logs/v73_population_joint_r5_resume.log`。完整载入epoch21原生头、query及168项Adam状态，从epoch22继续真实反向与更新，峰值allocated10.19840GiB。继承7791完整呈现，原第22轮107个未保存更新保留在parent现场且不计入恢复模型。原运行status已标为interrupted，interruption.json保留末状态与原因未知证据。尚未完成余9轮或最终评价。

仅冻结前缀和build图像改用mmap文件页，原生DPT与query仍实时可训练；无视图/空间支持缩减。旧checkpoint缺随机数状态，resume.json记录seed7304重启CUDA抽样、重放Python shuffle顺序的限制，不能宣称逐比特续跑。后续checkpoint同时保存fit顺序和Python/CPU/CUDA RNG。

固定模型评价脚本 `scripts/evaluate_worldsim_v73_fixed_actors.py` 已实现：完整载入已训练共享头/query，或选择固定原生融合/PCA；所有Actor保留，non-ready为空预测，输入角色原样进入最终汇总，七路相机插值权重与有效画幅传入同一解码/曲面支持路径。输出共同显式三角曲面及字面首交点评价，heldout仅交给最终评价器，无优化/目标初始化。此入口尚未实际运行，不把源码接通当作跨域效果验证；下一步旧AV2完整窗口真实推理，然后才能处理新20日志。

event/CAPA/AV2前缀及native控制专题报告同步更新。native-only r11 PID38460仍做全队列初始评价，Ada Y-up r2 PID37887已进入epoch2训练，joint恢复PID39009正常；r10仍未启动，无自动启动队列。F08当前为已恢复执行、原因未明和最终结果待完成；F01/F02/F03/F04/F05/F07仍active，F06直接数据配置缓解，下一编号V73-F09。整个V7.3未完成，shutdown=false；全部研究收尾保存/push且确认无任何训练、评估、数据或待启动任务后才关机。

---



## V7.3 event/CAPA完整结果、AV2前缀完成及joint中断恢复（2026-09-08）

r9（code c2fbdccf，30epoch/11130更新、5632.23s）完整75开发Actor/5日志：hit32.4688%、early6.0060%、miss54.5686%、free0.038543m、target→surface0.227363m、recall@0.2m72.4665%。相对r8仅增加event：hit+1.466pp，日志bootstrap95%[−0.418,+3.510]pp；miss−2.265pp，[−4.321,−0.476]pp；early+0.415pp，[−0.257,+1.050]pp；free+0.004223m，[−0.001748,+0.012530]m；距离−0.002190m，[−0.008942,+0.003087]m；recall+0.521pp，[−0.578,+1.641]pp。缺失减少，但未形成覆盖/自由空间一致改善，不能直接选为最终胜者。移动9Actor/2日志hit10.4026%、early4.2654%、miss76.8620%、free0.017647m、距离0.146308m、recall81.8736%，样本量限制保留。证据m2/global/population_r9_analysis.json。

训练event全部自有抽样束：epoch1为38087束/no_support14745/capped15315/字面miss22187；epoch30为38156/14723/15505/23575。随机抽样不相同，不能当配对泛化指标；它表明近39%的训练监督束仍无有限宽度表面支持，F03未解决，不能声称event似然自动创造缺失支持。

CAPA r2（code eb42f835）31窗口×100步=3100次build-only TTA完成，全部744视图最终全24联合推理，489Actor已评价，PID31466退出。8277.17s，GPU allocated峰值8.08370GiB、RSS13.74811GiB。完整75开发/5日志：hit17.8032%、early10.9508%、miss58.4555%、free0.120341m、距离0.203133m、recall76.8964%；空表面8→3，仍全部保留。相对原生M1融合，hit−4.974pp，[−13.315,+1.334]pp；free−0.029963m，[−0.061784,+0.011995]m；距离+0.027708m，[+0.000880,+0.075590]m。此任务迁移未显示一致优势，不等于CAPA原论文任务失败；CAPA每开发窗口用build适配，原生模型为共享训练，信息/优化预算差别明确。移动9Actor/2日志hit11.2711%、early4.8243%、miss69.1382%、free0.037237m、距离0.139751m、recall88.2605%。证据m2/global/capa_r2_analysis.json。

旧AV2前缀 `WS-V73-M4-AV2-PREFIX-01/20260908T011000Z__old-development-28view-prefix-r1` 已成功，code9541ac7d，PID38283退出。完整28视图672×672跨视图聚合，独立图编码chunk7，33.9146s，GPU allocated峰值7.64359GiB、RSS7.60567GiB，4层缓存2118584384字节。原始头/build325203个尺度对应，固定IRLS scale26.60637；之后装入M1r3适配头，优化更新0。没有新20日志评分。冻结前缀资源问题在该旧窗口可行，完整query神经评价尚未做，不能外推所有跨域资源或效果。

native-only r11 `20260907T233000Z__population-native-only-full-track-s7304-r11` 已真实启动，code9541ac7d、PID38460，日志 `/root/autodl-tmp/controller_logs/v73_population_native_r11.log`；744共享冻结视图、371fit/67dev/51空输入，当前初始评价。Ada Y-up r2 PID37887继续。

新增V73-F08：主joint r5 PID18843在epoch22第107个呈现后意外消失，末条elapsed22180.14s、无traceback/完成输出，status仍显示running。第21轮模型/优化器已保存，7791完整呈现；余107步没有进入该checkpoint。容器memory.events的oom=0/oom_kill=0，宿主MemAvailable约681GiB，内核日志无读取权限；无法据此确定退出原因，也不能将其写成OOM或模型失败。保留全部原始日志、checkpoint与末状态，记录中断；整个研究未完成，不能关机。

已先检索PyTorch官方mmap加载与恢复指南：https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html 。冻结tokens/build图像改为只读torch.load(mmap=True)，进程可共享文件页，避免各自复制整个CPU前缀；不改变任何输入视图或可训练路径。新增--resume-from恢复模型和优化器并写新run，保留parent未完成epoch现场；以后保存Python/CPU/CUDA RNG和fit顺序。旧r5缺RNG，恢复明确为seed7304重启CUDA抽样、重放Python shuffle顺序，不声称逐比特连续。

登记恢复run `WS-V73-M2-GLOBAL-ACTOR-01/20260908T012500Z__population-joint-r5-epoch21-resume-r1`，同r5配置从epoch21继续到30，原initial/PCA复用；尚未启动，代码入库后立即执行，最终报告区分7791继承与新3339呈现，旧未保存107步单列。r10同全轨迹joint仍未运行，不能用本恢复替代r10。

failure_ledger_delta=update F01/F02/F03/F05/F07，并新增F08训练意外中断active；F01/F02/F03/F04/F05/F07仍active，F06直接数据配置缓解。下一编号V73-F09。当前不是已证实不可恢复的资源不足，先执行恢复和内存映射对策。整个V7.3未完成，shutdown=false；全部研究收尾保存/push且无训练、评估、数据和待启动任务后才关机。

---



## V7.3 首事件训练收尾、Ada轴修订启动与28视图前缀准备登记（2026-09-08）

r9 `20260907T230000Z__population-lidar-track-beam-event-s7304-r9` 已完成30epoch/11130更新和全489对象最终评价，PID33444退出。code c2fbdccf，5632.23s、GPU allocated峰值0.42522GiB、RSS3.00089GiB；独立日志配对统计正在汇总，尚不提前判断event增益。训练记录保留所有自有束的no_support/capped/字面miss，缺支持的零位置梯度限制仍在。

释放该进程后实际启动AdaPoinTr r2 `WS-V73-M2-ADAPOINTR-01/20260907T231000Z__population-full-track-pcn-yup-s7307-r2`，code acd91103，PID37887，日志 `/root/autodl-tmp/controller_logs/v73_population_adapointr_r2.log`。相对r1仅输入[x,z,y]转PCN Y-up并逆变换全部输出；完整模型、初始化、30epoch/全轨迹fit标签和共同曲面评价保持。r2最终结果尚未产生，F07仍active。主joint r5与CAPA r2继续，无重复启动或后台队列。

登记旧AV2开发窗口冻结前缀准备 `WS-V73-M4-AV2-PREFIX-01/20260908T011000Z__old-development-28view-prefix-r1`，脚本 `scripts/prepare_worldsim_v73_native_prefix.py`。依据官方VGGT源码，只分批独立逐图patch编码器（7图/批），其后28视图672×672共同进入全部跨视图聚合层。官方和本地aggregator已经只保留DPT所需4/11/17/23层，复用已有能力，不能将其写成新增内存优化。每视图缓存clone后保存以免复制整窗口底层storage；只缓存完全冻结前缀，原生适配头仍从M1r3权重实时解码。尺度以原始预训练头和build传感器对应估计固定窗口IRLS，随后装入已适配头；新域无共享梯度更新。

该28视图任务尚未启动，等待CAPA真正结束释放峰值预算；不削减第七路、时间跨度或视场。其目的是旧开发数据接口与真实资源测量，不是新20日志确认结果。源码依据：https://raw.githubusercontent.com/facebookresearch/vggt/main/vggt/models/aggregator.py 、https://raw.githubusercontent.com/facebookresearch/vggt/main/vggt/heads/dpt_head.py 。

failure_ledger_delta=update F01/F03/F05/F07进度；F01/F02/F03/F04/F05/F07仍active，F06直接数据配置缓解，下一编号V73-F08。joint full_track r10/native-only r11继续待资源。整个V7.3未完成，shutdown=false；整体收尾保存/push且确认无训练、评估、数据和待启动任务后才关机。

---



## V7.3 AV2共同Actor窗口导出与七相机读取接口（2026-09-08）

`WS-V73-M4-AV2-DATA-01/20260908T003500Z__old-development-window-r1` 已完成旧开发日志02678d04-cc9f-3148-9f95-1ba66347dff9的数据准备。实际执行基点75dd260d加本里程碑工作树源码，随后一并入库；无网络模型推理。41.84s、RSS0.87773GiB、28个完整七相机×四时刻视图、6扫描562494原始返回（375000 build/187494留出），全体传感器pose有效。依据build轨迹选21车辆，19个有输入、2个空输入均保存case，4辆已知速度>2m/s。一个空输入对象有3个留出自有返回，仍保留缺失；单时刻轨迹在逐点/相机时刻没有可用pose者也保留，不用外推伪装已知轨迹。

每点Actor规范位置/原点/方向与归属按实际发射时间，所有已知框参与唯一归属；图像关联按各曝光时间移动同一规范点，只读取四个build扫描。留出射线与模型输入支持分开保存。build_observations.pt与21个case已按共同格式导出；coverage/av2_old_window_r1_index.json保留各Actor与scan统计。

对七相机接口已先检索CVPR2023 CAPE、ICCV2023 PETRv2的相机局部位置与时空对齐处理，再补充可选calibration-azimuth camera_embedding_weights：周期插值现有六项嵌入，保留七路图像、投影、方向和时间，无新增随机camera参数。该插值是工程假设而非文献已验证的跨域保证。局部query和偏移采样、原生深度支持生成均接入valid_image_rect排除padding；共同训练/fusion入口可传递，现有nuScenes输入无这些字段时维持原路径。统计器新增角色保持独立，external_confirmation不会遗漏或混作开发集。

完整28视图神经推理、其资源峰值及插值有效性尚未运行，不能称外部方法验证完成。下一步在该旧AV2开发窗口提取真实冻结前缀并运行固定模型；当前三项GPU正常占用，不叠加作业，不削减第七视角或视场。新20日志未做模型推理和外部评价。报告 `WORLDSIM_V7_3_AV2_GEOMETRY.md` 已更新。

failure_ledger_delta=update F01/F04/F05对应/资源/新域接口进展；F01/F02/F03/F04/F05/F07仍active，F06直接数据配置缓解，下一编号V73-F08。主r5、CAPA r2、event r9继续，后续Ada轴r2、joint full_track r10、native-only r11待资源，无启动队列。整个V7.3未完成，shutdown=false，最终需保存/push并确认无训练/评估/数据/待启动任务后才关机。

---



## V7.3 外部载荷下载完成与AV2逐点几何接口（2026-09-08）

20条新AV2确认日志的760文件已下载完成，343260344字节；s5cmd返回0，977.74s，PID34904正常退出。传输状态保存在 /root/autodl-tmp/controller_logs/v73_av2_confirmation/status.json，原始760行copy.log保留。仅复制原始文件，没有读取新日志的图像/点云数值进行方法选择，没有外部网络推理；既有登记名单不变。

按AV2官方Sweep及SE3接口实现 `motion_proj/worldsim_v73/av2_geometry.py`：int64相对纳秒插值、逐点发射时间、参考ego补偿端点、两LiDAR逐点原点、已知Actor/camera时刻变换、保留七路全视場图像及padding有效矩形。输入端点只应用一次参考ego→city变换，不能重复运动补偿；Actor坐标/原点使用各点时间，图像使用自身曝光时间。尚未完成Actor case/build导出、六相机ID到七路接口与padding排除，不能称外部模型已接通。

一次旧开发真实扫描诊断已完成：02678d04-cc9f-3148-9f95-1ba66347dff9序号5，94860返回、offset0.615–106.315ms，全部有ego姿态。只用扫描时刻原点会造成原点位置差中位0.59106m/p95 1.07697m/最大1.15741m。laser0–31→up的ring仰角MAD中位0.011554°，反向映射0.126340°，采用前者但明确为旧开发数据的几何推断，未检索到官方编号组声明；不在新确认日志重估。七相机原始曝光/光轴/内参与有效矩形均记录，672×672仅完整画布而非有效视场；padding不能形成表面支持。CPU0.99s、RSS0.22056GiB，无模型smoke或新确认质量读取。证据coverage/av2_geometry_development_r1.json，报告 `WORLDSIM_V7_3_AV2_GEOMETRY.md`。

主joint r5、CAPA r2（最近17/31窗口完成）和event r9（最近epoch14）正常运行，不增加GPU并发。AdaPoinTr轴修订r2、joint full_track r10和native-only r11待现有资源释放，无自动启动队列。下一步在旧AV2开发数据导出共同Actor/射线接口并处理七路相机/有效画幅，之后继续完整对照和最终新日志应用。

failure_ledger_delta=update F04/F05数据对应与外部格式进度；F01/F02/F03/F04/F05/F07仍active，F06直接数据配置缓解，下一编号V73-F08。整个V7.3未完成，shutdown=false；保存/push和确认无训练、评估、数据及待启动任务之后才执行最终关机。

---



## V7.3 二十条新AV2日志的元数据选择与精确载荷准备（2026-09-08）

针对nuScenes trainval无未记录新日志的F05，已依据AV2官方数据源迁移：只读S3目录得到Sensor train700条日志，排除仓库configs/docs/scripts中已引用身份及本地已有目录，共1条；699候选按日志名字典序选前20条，在读取每条对象元数据前登记身份。既有80条AV2 val legacy日志可用于格式开发。新train日志在本研究中是external_confirmation，不能用于共享训练或结构/超参选择；跨日志与传感器域改变需单列，不能混称同分布nuScenes确认。

`configs/worldsim_v73/av2_external_confirmation_r1.json` 保存完整20身份/时间/载荷表，`scripts/prepare_worldsim_v73_av2_confirmation.py` 复用既有名单并按需复制；未新增哈希/校验和/指纹。元数据准备98.49s，20×(7周视相机×4build+6LiDAR+4标定/轨迹/标注)=760文件，343260344字节/327.36MiB，元数据无缺项。build扫描序号[5,15,20,30]、留出[10,25]；相机最近实际时间差−23.10至+18.11ms，必须各自时间投影，不能视作同步真值。图像、点云、标注值和模型分数均未用于选择，缺输入和缺返回必须保留。

精确载荷下载已实际启动：codec00020fb，PID34904，日志 /root/autodl-tmp/controller_logs/v73_av2_confirmation_download.log，子目录v73_av2_confirmation/status.json及copy.log记录760文件的传输进度；当前copying，s5cmd四并发正常写入所选文件。数据进程也纳入关机前任务检查。此时外部推理/评价尚未开始；AV2双LiDAR来源、ego补偿、轨迹插值、纵向相机内参和6项nuScenes相机嵌入→7周视接口需在旧开发日志先迁移，不削减第7路输入，不给新增随机相机嵌入后冒称已可靠迁移。详见 `WORLDSIM_V7_3_CONFIRMATION_DATA.md`。主joint r5、CAPA r2与event r9正常运行，Ada r1已退出；后续r10/r11、Ada r2仍待可用资源。

failure_ledger_delta=update F05数据来源与格式对策；F01/F02/F03/F04/F05/F07仍active，F06直接数据配置缓解，下一编号V73-F08。整个V7.3未完成，shutdown=false，自动研究继续；最终必须保存/push且无训练、评估、数据或待启动任务才关机。

---



## V7.3 AdaPoinTr首轮完整结果、event真实启动与独立日志对策（2026-09-08）

AdaPoinTr r1 `WS-V73-M2-ADAPOINTR-01/20260907T221000Z__population-full-track-pcn-s7307-r1` 已完成并正常退出：code23981936，完整模型32494657个可训练参数，30epoch/11130 Actor呈现/2790优化更新，4926.63s含489对象初始/最终评价，GPU allocated峰值1.00658GiB、RSS2.11761GiB。最终75开发对象/5日志：hit16.05%、early11.31%、miss48.92%、free0.14656m、target→surface距离0.09758m、recall@0.2m85.66%。相对自身初始化，距离−0.11974m、日志bootstrap95%[−0.27337,−0.02175]m，recall+18.54pp、[+7.71,+30.93]pp，hit+6.68pp、[+3.16,+10.20]pp；early+6.38pp、[+2.47,+10.82]pp，free+0.03040m、[−0.01929,+0.07727]m。几何覆盖可学习，但不是一致的物理改善。

相对同full_track标签的r7，Ada r1 hit−15.70pp、[−26.16,−2.97]pp；free+0.07499m、[−0.02406,+0.17408]m；相对r8，free+0.11224m、[+0.02594,+0.20383]m，recall+13.71pp、[+2.70,+27.37]pp。移动开发9对象/2日志的hit7.14%、early4.39%、miss18.56%、free0.08765m、距离0.15679m、recall45.61%，大量返回较晚，不能只看miss降低。8个空输入开发对象及23个无自有留出回波均保留。全体、共有输入、移动分层和配对见m2/global/adapointr_r1_analysis.json。r1未迁移PCN Y-up坐标的F07限制保留，修订r2尚未运行；不能仅凭r1否定完整补全基线。

Ada退出后已在释放的GPU预算内启动r9 `WS-V73-M2-GLOBAL-ACTOR-01/20260907T230000Z__population-lidar-track-beam-event-s7304-r9`，codec2fbdccf，PID33444，日志 `/root/autodl-tmp/controller_logs/v73_population_lidar_event_r9.log`。与r8仅增加固定geometry first-event项，weight0.01/σ0.2m/cap28，完整30epoch、全轨迹fit标签和相同原始build输入。已进入真实反向与优化，首epoch峰值allocated0.40113GiB；no_support/capped/原字面miss分别保留，不把有界event代理称为已解决缺支持。主joint r5和CAPA r2继续，r10/r11与Ada轴修订r2仍待资源，无自动启动队列。

F05调查：本地35场景/27日志的七传感器载荷足够形成窗口，但当前25日志外仅0139/0379，均有旧版实际使用记录。进一步比对发现nuScenes trainval全部850场景/68日志均已有V7.2角色，无未分配日志。公共挂载有完整294GiB归档，能补载荷不能补独立性，未大规模解压。已先查AV2官方论文/格式/下载源并成功匿名列举Sensor train目录；下一步从未出现过身份选外部确认候选，使用旧AV2开发日志解决格式问题，最终候选不用于结构或超参。跨传感器域限制必须单列；当前无新候选载荷/质量读取。详见 `WORLDSIM_V7_3_CONFIRMATION_DATA.md` 和 coverage/log_payload_inventory_r1.json。

failure_ledger_delta=update F02/F03/F05/F07；F01/F02/F03/F04/F05/F07仍active，F06直接数据配置缓解，下一编号V73-F08。整个V7.3仍未完成，shutdown=false；三项正常GPU研究继续。仅全部研究收尾后保存/push并确认无训练、评估、数据和待启动任务，才执行已授权关机。

---



## V7.3 同全轨迹标签原生DPT控制实现与登记（2026-09-08）

现有M1r3/fusion与r7/r8/Ada的标签预算不同，不能直接归因架构。已补齐共享训练器native_only模式：直接微调完整原生DPT，规范坐标native+LiDAR融合与同预算PCA曲面，query模块冻结且不参与预测；同full_track surface/free和build depth监督，不增加外挂小头。训练每步真实重算原生头；仅固定权重评价按scene/view复用深度，避免多个Actor重复解码。原始输入、视图和多尺度原生路径均保留。

完整371fit Actor每epoch都呈现；无相机/无任何有效梯度的native-only样本保留LiDAR输出并记录无优化，而不伪称一次DPT更新。PCA方向是逐步更新、当前步固定的条件位置梯度近似。51个原输入空对象仍按主模型协议空预测，全部队列和缺失分母保持。实现尚未实际训练，不新增重复smoke。协议详见 `WORLDSIM_V7_3_MATCHED_NATIVE_CONTROL.md`。

登记r10 `20260907T233000Z__population-joint-full-track-s7304-r10`（相对正在运行r5仅扩展full_track fit标签）及r11 `20260907T233000Z__population-native-only-full-track-s7304-r11`（同标签/原生初始化/30epoch的保守几何适配）；二者均未启动，无后台队列。r9 event对照也未启动。当前r5、AdaPoinTr r1与CAPA r2正常推进，需等实际资源释放后调度，不能从暂时空闲的瞬时显存忽略已知峰值。

failure_ledger_delta=update F02/F05比较完整性与主路线推进；F01/F02/F03/F04/F05/F07仍active，F06直接数据配置缓解，下一编号V73-F08。整个V7.3未完成，shutdown=false；长训练结束后继续按真实结果改进，全部研究收尾并确认无任务后才关机。

---


