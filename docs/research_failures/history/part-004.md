# 历史原始记录 004

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V7.3 / Q-v2 持续研究授权与计划revision6（2026-09-08 17:15 UTC / 新加坡2026-09-09）

用户最新明确要求持续V7.3，随后Q-v2；若全部完成且没有明确下一步，参考其方案继续auto research，而不是关机。旧“V7.3完成后shutdown/暂停自动跟进”安排已取消，AGENTS和30分钟ACTIVE heartbeat同步覆盖。真正不可避免的资源不足仍按原约定保存/push、确认无训练/评估/数据任务及启动队列后shutdown并通知加卡；当前未发现这类资源出口条件。

用户文件原样归档 `docs/references/V73_FOLLOWUP_REFERENCE_20260909.txt`，当前解释与执行顺序见计划revision6第17节。文件是后续建议参考，不是自动改动当前配置的指令；其中新基座/论文主张待一手核实。`WS-V73-Q-V2-01`登记pending，表示三角后下一轮Query显式表面生成研究，具体表示未选，没有新增实现/run/seed/结果。若coverage/physics冲突持续，优先参数化，保留可训练几何与物理监督；随后按证据分别研究近边界free、局部对应及上层聚合器适配，不要求穷尽全部后端或强制四格矩阵。主联合R10/R12/R14 event=0，R9是LiDAR-only，不能声称联合event已测。

17:11 UTC实读：R12/PID81766第10/30轮，R14/PID68108第27/30轮，均按原配置运行；R10已收口，三角尚未完成。当前代码依据ada1e773，本次仅授权/计划与参考归档，无新实验、未读取20新日志质量、未变更训练/表面/损失/cohort。现有缓存只有冻结前缀含义，未来适配聚合器需重算对应通路。参考中的完整几何基座/上层LoRA是待核实候选，不是已成功实现。

failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F04,V73-F05,V73-F06,V73-F09]；failure_ledger_delta=none，旧风险保持，下一编号V73-F10。下一步持续R12/R14，完整三角配对后推进Q-v2；每个实质里程碑同步三本台账、小步push、报告附architecture components图。完成不会触发shutdown，30分钟自动跟进ACTIVE。

---

## V7.3 表面参数化候选的源码迁移准备（2026-09-08 16:00 UTC）

变更类型为一手源码/设计分析，无新训练或评价run。项目依据944d6904，复核Kaolin marching_tetrahedra、NVIDIA nvdiffrec DMTet提取、FlexiCubes核心与官方优化示例；详见`docs/WORLDSIM_V7_3_SURFACE_PARAMETERIZATION.md`及计划16.4。候选尚未实现、训练或选定，三角收口后的条件决策保持。

新明确的迁移边界：DMTet/FlexiCubes按符号离散选择活跃单元，再对已选几何插值反向；仅接提取后网格损失不保证空支持恢复。原生build深度有梯度也不能代替生成参数获得支持梯度。FlexiCubes训练四分片/导出两分片在非共面时可能改变实际表面，若采用应统一训练与物理读出的显式三角划分，或单列转换误差；grad_func/QEF非可微分支不作为主训练路径。官方例子含完整参考网格/内部SDF条件，不能迁成未知区真值。相同grid resolution不等价于同输出密度或成本，当前没有候选资源实测。

对应迁移准备：共享格点隐状态可由保留的三维查询局部传递，避免默认所有稠密格点都读24视图；若需要直接场监督或新初始化，单独披露其作用，UNKNOWN不补FREE/正厚度。若无法合理提供场的支持恢复条件，直接共享顶点显式曲面仍是候选替代，不退回native-only。源码/推论/本机实现/收益状态已分开，未下载依赖或权重，未运行测试，实际architecture components图已附报告。

failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F04,V73-F06]；failure_ledger_delta=none，无新失败编号，下一V73-F10。15:47 UTC R12/PID81766第5轮、R14/PID68108第22轮正常，GPU14759/24576MiB、cgroup oom/oom_kill=0；本轮未改训练、表面、损失或cohort。先完成R10/R12/R14与R11锚点，再决定参数化；near-surface与深度/方向/遮挡一致性保持独立待测。20新日志质量未读，30分钟跟进ACTIVE，全V7.3未完成、shutdown=false。

---

## V7.3 同监督表面支持诊断完成（2026-09-08 15:20 UTC）

`WS-V73-M2-SURFACE-SUPPORT-01/20260908T151000Z__development-full-track-fixed-surfaces-r2`已done，code6920c120；只对R10/R7已保存表面读取CPU首/全交点及无符号最近距离，wall.788708s、RSS.631866GiB，正常退出。R11完整复用r1保存计数，无重复训练/神经推理/测试。全部75开发Actor/5日志/11886 owned返回、23无owned对象和每方法8空表面保留；不读取新20日志质量。

日志等权近邻覆盖/任意正确沿束交点/正确首交点：R10为82.6047/34.4466/27.1463%，R7为74.5468/39.3379/31.7549%，R11为75.1499/18.3920/17.6111%。邻近但无正确沿束交点分别48.1581/35.2089/56.7579%；R10−R7差+12.9492pp，五日志全增加。R10−R7 early中，无正确交点但邻近+8.8477pp、五日志全增加，后方有正确交点−.2827pp、3增2减；late且邻近+11.6434pp、五日志全增加。新增错误并非主要是早片挡住已存在的正确沿束片。

R10−R11任意正确沿束支持+16.0546pp、五日志全增加，说明Query确实比原生控制生成更多正确位置的沿束支持；同时early且后方正确+6.5194pp、五日志全增加。不能把这一正支持增量外推成物理优势，也不能否定Query生成能力。R10/R7/R11多个early层出现率10.8324/8.2673/1.8116%，全部层数均值2.400/3.798/.642；层数和出现率不同，不称Query总是生成更多层。

三个模型共享full_track标签、旧cohort和hard-free语义，模型与实际优化成本仍有差异；归因不能只落在attention。只做保存统计的逐日志差值、没有新增bootstrap。八类计数不是逐射线训练转移轨迹，owned束诊断也不能完整解释全部near-box束free或背景F04。原始1276条R10 early中577条后方正确、641条无正确但邻近、58条两者无，原始计数与日志等权分母不混写。

证据已归档`docs/autoresearch/worldsim_v73/m2/global/surface_support_r2_{summary,manifest}.json`、`surface_support_matched_r2_analysis.json`和`V73_SURFACE_SUPPORT_MATCHED_R2.png/pdf`；R11来源为r1 summary。专项`WORLDSIM_V7_3_SURFACE_SUPPORT.md`保留实际architecture components图、同监督分类表与新图，以及r1历史。图复用既有Matplotlib并视觉核对，结果归并/绘图脚本保存；未改动训练代码。

failure_ledger_refs=[V73-F02,V73-F03,V73-F04]；failure_ledger_delta=update V73-F02 support evidence, no new failure ID，下一编号V73-F10。同一已研究卡点复用计划revision5及r1的一手文献/接口依据；三角若仍冲突，优先Query surface parameterization，分别再检验near-surface certified-free与局部深度/方向/遮挡一致性。不以移除早片或选后方交点冒充正确几何，UNKNOWN不标FREE。R12/PID81766和R14/PID68108按原配置运行，最新15:06 UTC为第3/19轮、GPU14759/24576MiB、cgroup无OOM、磁盘69GiB可用；当前全V7.3未完成，30分钟跟进ACTIVE，shutdown=false。

---

## V7.3 同监督固定表面诊断登记（2026-09-08 15:10 UTC）

任务`WS-V73-M2-SURFACE-SUPPORT-01`新增run `20260908T151000Z__development-full-track-fixed-surfaces-r2`，pending。R10最终产物现在可用，本次只对R10 joint与R7 LiDAR的已保存表面复用原有CPU诊断；R11直接复用r1已保存的Actor/帧/日志计数，不重算。新问题是同full_track/hard-free条件下的邻近覆盖、正确沿束支持与早层遮挡差异，补足r1中R5/R9/R11训练条件不匹配的边界；不改主评价或当前三角配置。

沿用全部75开发Actor/5日志/11886 owned返回、.2m距离容差与1e-4m数值交点合并。空表面和无owned对象继续保留，八类定义/聚合不变；不读取20新日志，不训练或神经推理，不改表面、删除Query或返回后方交点。分析脚本保持`scripts/analyze_worldsim_v73_surface_support.py`，新代码只涉及保存结果归并/绘图；原解析束验证已覆盖该读取方法，不再重复测试。

15:06 UTC R12/PID81766第3轮、R14/PID68108第19轮正常运行；GPU14759/24576MiB、cgroup oom/oom_kill=0、磁盘69GiB可用。表面诊断复用已有CPU环境、线程2，无新增GPU负担。代码基于cc6b3a3e，manifest记录执行提交；failure_ledger_refs=[V73-F02,V73-F03,V73-F04]，登记failure_ledger_delta=none。仍先完成三角后决定是否修改Query surface parameterization，近边界监督/对应一致性分别研究。V7.3未完成、30分钟跟进ACTIVE、shutdown=false。

---


## V7.3 R10最终配对完成：生成提高命中并增加侵入（2026-09-08 14:30 UTC）

任务`WS-V73-M2-GLOBAL-ACTOR-01`，run `20260907T233000Z__population-joint-full-track-s7304-r10`，训练code26a7e509、seed7304；分析基于1dad4dd8后工作树的保存结果。R10 done、30轮11130呈现/11130真实更新、零跳步/恢复，371 FIT训练输入与67可预测DEV不变、489对象全记录/51零LiDAR缺失保留；DPT32654562＋Query1670517参数，native_project变化.005422188，wall33811.146305s、allocated10.213784GiB、RSS34.030704GiB，412133346字节latest.pt与最终surface保存，PID53472已退出。

完整75开发Actor/5日志、23无owned返回、8空表面：hit.271463、early.202637、miss.326175、free.271026m、单向target→surface.113193m、recall@.2m .826047。R10−同标签同hard-free原生R11：hit+9.535pp [95% +4.616,+16.268]pp、miss−27.477pp [−37.927,−20.665]pp；early+12.620pp [+5.904,+18.968]pp、free+.106062m [+.013990,+.209804]m。hit/missing在5日志均改善，early在5日志均变差。distance−.055311m [−.129487,+.001635]m、recall+7.455pp [−4.861,+23.965]pp，不能仅据均值宣称稳定几何优势。

相对同full_track LiDAR R7：hit−4.609pp [−13.684,+7.766]pp、miss−17.280pp [−30.451,−7.635]pp、early+8.425pp [+1.889,+16.084]pp、free+.199452m [+.077589,+.317941]m。相对短窗R5：miss−5.092pp [−10.279,−.610]pp，其余所列指标区间跨0；R5恢复缺完整RNG的执行差异保留。相对自身初始化：missing−18.525pp，early+7.989pp；不把更多表面命中等同正确首表面。

移动9 Actor仅2日志、6对象共6657 owned返回：R10移动hit均值.630446，但其中日志ddc03471df3e4c9bb9663629a4097743仅1条owned返回，另一日志ca6d14b008ed4e0bb6b1eaaedadbd6c1有6656条、hit.260892且比R7低9.379pp。这个分母事实削弱动态泛化解释，不删除对象/改聚合。FIT20日志的hit+.053166、distance−.020973m来自full_track标签覆盖时刻，不是泛化。

训练native测量加权Huber .781005→.365007m，采样coverage .304005→.199457m、hard free .363581→.215236m；DPT/Query均有梯度。只是优化过程，不将不同采样均值当固定射线配对。分析只读取已有结果，配对一次5日志bootstrap10000/seed7304；绘图复用保存区间和日志，没有重跑神经推理/训练或测试套件。

证据归档`docs/autoresearch/worldsim_v73/m2/global/population_joint_r10_summary.json`、`population_joint_r10_analysis.json`、`population_joint_r10_training.json`、`V73_JOINT_R10_TRAINING.png/pdf`、`V73_JOINT_R10_PAIRS.png/pdf`；两图已视觉检查。`WORLDSIM_V7_3_MATCHED_NATIVE_CONTROL.md`与`WORLDSIM_V7_3_POPULATION_RESULTS.md`已同步实际architecture components图、原生对照、区间与移动分母限制；比较协议也反映R10完成/R12启动。paper r2保持其10:30 UTC历史快照，下一完整三角里程碑再修订。

failure_ledger_refs=[V73-F02,V73-F03,V73-F04,V73-F05,V73-F09]；failure_ledger_delta=update V73-F02 evidence, no new failure ID，下一编号V73-F10。F02覆盖/物理冲突得到同目标R10−R11证据，非所有视觉几何适配的负结论；F03/F04/F05/F09均未解除。已查DMTet/FlexiCubes、ViGT与GNT/TransFusion一手来源的迁移边界见计划revision5及SUPPORT报告，不机械重复检索同一已记录卡点。

R12/code dc6fe427/PID81766和R14/PID68108按原登记训练；本次结果分析未修改运行配置。继续R12−R10、R12−R14、R14−R11；三角若仍冲突，优先改Query表面参数化，随后分别评估near-surface certified-free与候选深度/方向/遮挡一致性，不以loss扫描或native-only退路替代。20新日志质量未读。最后资源实测14:15 UTC无OOM，完整V7.3未完成、30分钟跟进ACTIVE、shutdown=false。

---

## V7.3 R10完成与R12有限宽束联合训练启动（2026-09-08 14:15UTC）

`WS-V73-M2-GLOBAL-ACTOR-01/20260907T233000Z__population-joint-full-track-s7304-r10` 已done，PID53472退出、GPU已释放；30轮11130呈现/11130真实更新、零跳步/零恢复，489个Actor的最终评价与412133346字节latest.pt已保存。wall33811.146305s（9.392h），allocated峰值10.213784GiB、RSS34.030704GiB，原生project参数最大变化.005422188。371 FIT/67可预测DEV、旧489 cohort/51零LiDAR缺失保留、744份冻结前缀/full_track标签；当前正在从保存的结果计算R10−R5/R7/R9/R11与初始化的日志配对，不提前宣布科学结论。

R12按此前登记接续启动：`WS-V73-M2-GLOBAL-ACTOR-01/20260908T050000Z__population-joint-full-track-beam-range-s7304-r12`，code dc6fe427，PID81766，日志`/root/autodl-tmp/controller_logs/v73_population_joint_beam_r12.log`。M1r3重新初始化、seed7304、30轮、原旧输入cohort/full_track；native1、free .5，只有free目标相对R10改为beam_tube_range .03m/res32，event0。复用R5同初始化评价与PCA，非resume；没有加入新表面参数化、额外LoRA、visual-only cohort或用户新候选损失/对应模块。

14:15UTC R12第1轮已有optimizer更新，DPT与Query梯度均正、allocated10.228380GiB；GPU进程11494MiB，R14/PID68108仍运行、进程3066MiB。cgroup oom/oom_kill均0，没有确证资源不足；原始24视图未缩减，仅冻结前缀缓存，DPT继续真实反向。seed存于manifest顶层并在trainer固定7304，不在config子字典，不能把子字典缺值读成未设置seed。

R12用已保存静态启动脚本直接启动，无自动GPU队列；没有中断R14。R10分析仅消费既有结果/训练日志，不重复训练或推理。调度里程碑failure_ledger_delta=none，R10科学结论待汇总；F02/F03/F04/F05/F09继续有效，下一失败编号V73-F10。三角完整比较尚未收口，20新日志质量未读；全V7.3未完成，30分钟跟进继续，shutdown=false。

---


## V7.3 固定表面支持诊断完成（2026-09-08）

`WS-V73-M2-SURFACE-SUPPORT-01/20260908T120000Z__development-fixed-surfaces-r1` code866ed28f已done、PID78078退出；全部75开发Actor/5日志/11886个owned返回，23对象无归属返回仍保留，R5/R9/R11各8空表面。CPU1.050194s、峰值RSS0.633244GiB，无神经推理/优化或表面修改。一次3条解析束检查已在登记前通过，没有重跑训练、smoke或回归套件。

日志等权：R5近邻覆盖79.7126%、任意正确沿束交点31.3134%、字面hit22.2773%，有48.3992个百分点表面邻近但沿束无正确交点。其early20.9807%中9.0361pp后方有正确交点、11.7509pp无正确交点但表面邻近、0.1938pp两者都无；不是所有错误均为“早片遮住正确片”。R9三项为72.4665/36.0040/32.4688%，R11为75.1499/18.3920/17.6111%；R11也存在严重覆盖/沿束支持差距，不能归因于Query独有因素。

多个early深度层出现率R5/R9/R11为14.3220/3.6001/1.8116%；平均全部层数却为2.420/3.570/0.642，出现率和层数不等价，不据此宣称Query总是生成更多层。深度合并只处理数值重合，不是拓扑计数。八类互斥分量与每Actor/帧/日志完整归档`m2/global/surface_support_r1_summary.json`及manifest；统计按Actor owned分母再日志等权，原始束计数另存，后方正确交点不替代主评价首返回。

专题`docs/WORLDSIM_V7_3_SURFACE_SUPPORT.md`包含实际architecture components图及本次统计图`m2/V73_SURFACE_SUPPORT.png/pdf`。图使用远端既有Matplotlib生成并视觉核对，首次本地运行缺Matplotlib后直接复用已有远端环境，无安装/数据重跑。R5/R9/R11训练条件不同，结果仅为机制诊断，不是匹配消融或独立确认；阶段paper r2历史快照保持。

failure_ledger_delta=none，F02/F03/F04持续、下一编号V73-F10。本证据支持继续区分前后表面冲突与近邻覆盖未转化为正确支持，不能确定单一根因。先收口R10/R12/R14；若冲突持续，优先改表面参数化，再独立评估用户近边界free与局部对应建议。11:53UTC R10第24轮、R14第7轮均正常；本轮未改变其配置。R12静态脚本`/root/autodl-tmp/controller_logs/launch_joint_beam_r12.sh`已备好但未启动、无自动GPU队列；等待R10完成和释放完整Query资源。20新日志质量仍未读，V7.3未完成，30分钟跟进继续，shutdown=false。

---


## V7.3 固定表面的沿束支持诊断登记（2026-09-08 11:58UTC）

R10/PID53472第24轮、R14/PID68108第7轮继续正常训练，R12等待完整Query显存。本次登记`WS-V73-M2-SURFACE-SUPPORT-01/20260908T120000Z__development-fixed-surfaces-r1`，对全部75旧开发Actor的R5恢复完成表面、R9与R11最终表面做一次CPU诊断，尚未执行正式数据；不改动三角训练或选择新参数化。

问题：错误首交点之后是否仍有距原始真实返回±0.2m的交点；如果没有，测量点是否仍在表面0.2m邻域。采用已有SurfaceBVH的cast_rays、Open3D list_intersections及unsigned compute_distance，将所有owned heldout返回分为正确首交点、early且后方正确、early无正确但邻近、early无正确且不邻近、late邻近/不邻近、missing邻近/不邻近八类；空表面保留missing，无owned返回对象保留零分母，不选择高误差样例。

同时报告沿束多个数值分离深度层与多个free违规层；1e-4m相邻深度合并仅处理数值重合，不解释为精确拓扑/重复片计数。后方正确交点只是诊断，绝不替换字面首返回。统计先汇合Actor内原heldout帧、按owned束归一，再日志内Actor及5日志等权；与主评价聚合细节不同，不用来替换主表、宣称因果增益或独立新日志确认。R5/R9/R11目标、标签、模型不同，三者只作固定产物机制比较。

实现`scripts/analyze_worldsim_v73_surface_support.py`；[Open3D官方0.19文档](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html)核实全部交点/无符号距离接口，避免对不闭合片使用occupancy/SDF符号。一次三条解析束检查通过：early+后方命中、邻近但missing、远离且missing分别为1；未发起回归套件。本机CPU环境复用，线程2，不训练/神经推理、不改表面/opacity、不下载权重。

failure_ledger_refs=[V73-F02,V73-F03,V73-F04]；登记阶段failure_ledger_delta=none，下一失败编号V73-F10。证据从abd80411工作树增加本诊断；run将记录实际代码提交。完成后保存全部Actor/帧计数与5日志统计，结合三角最终结果再讨论参数化及近边界监督；不将当前诊断代替三角收口。20新日志质量未读，整个V7.3未完成，30分钟跟进继续，shutdown=false。

---


## V7.3 后续边界监督与对应建议的风险复核（2026-09-08）

本次为计划revision5条件设计，failure_ledger_delta=none，不创建V73-F10。F02：更密负采样不等于更好显式面位置梯度，防止通过丢失coverage降低free；认证只沿有依据的返回前自由域，不扩散到未知邻域。F03：分段相交处罚仍不能凭空创造缺失支持；保留直接coverage恢复。F04：frustum有效性与attention不是遮挡/Actor归属真值。F06：预测深度不应通过硬mask或自适应置信度关闭全部几何训练入口。

这些是既有风险对应的新方案边界，不是本轮实测负结果或风险解除。先完成三角比较，持续冲突时优先改Query参数化；两个用户建议分开评价。自动跟进已改30分钟并保留关机约定。

代码阅读基于ce685c1b工作树的`spatial_queries.py::ProjectedLocalRead`与`surface_visibility.py::BeamTubeFreeSpaceLoss`；证据与一手链接见计划revision5第16节。R10/PID53472与R14/PID68108在本次读取仍运行，R12尚未启动；未修改模型/损失/数据/训练配置，未新增进程、环境、权重下载或测试。20个新日志质量仍未读，V7.3未完成，shutdown=false。

---

## V7.3 用户方向确认：三角收口后优先表面参数化（2026-09-08）

用户明确保持“可训练几何基座 + 显式3D surface generation + 物理约束”。已写入项目AGENTS与计划revision4第15节，并同步比较协议；这是一项条件研究决策，不是新实验结果或新增失败。当前代码证据截至7ca07029，R11/F09原生负结果、F02覆盖/free冲突、F03缺支持、F04场景读出与F05输入条件边界继续有效。

先收口R10/R12/R14：R12−R10比较同Query参数化下finite-beam free目标，R12−R14比较同finite-beam目标下Query生成与原生PCA融合；已有R11补足原生hard-free锚点，R14−R11看原生路径目标变化。R10−R14有两项因素变化，不作单因素归因。R10/R14现有训练不中断、不改配置；R12按原登记启动，不混入新cohort/额外适配或新表面。

若收口后仍是Query coverage强但physics差，下一轮首要改变Query surface parameterization，保留可训练原生几何路径与物理约束；先检索顶会/优秀官方开源，再结合当前片形状、方向、范围、重叠和连接等实际失败选择一个迁移候选，不预定SDF/网格或同时维护多套表示。不继续以纯loss扫描为主，不把native-only默认升为最终主线；输入cohort扩展与适配范围另作独立因素，不在这一轮混改。现有覆盖/硬首交点/free/缺失及日志配对仍用于判断，不加晋级门控。

10:44UTC R10/PID53472 epoch21、R14/PID68108 epoch3正常；R12未启动，无新进程或GPU队列。比较协议同时修正旧“R11/R13未完成”表述：R11已10550更新完成，R13仅一轮真实反向、完整412 fit新cohort训练仍未做。用户方向已同步到原15分钟自动跟进prompt，保留全部资源/无任务后关机约定。failure_ledger_delta=none，下一失败编号V73-F10；20新日志质量未读，V7.3未完成，shutdown=false。

---


## V7.3 阶段论文r2与简明组件图完成（2026-09-08）

基于已提交bf04ef32/966bed9e的R11/R13证据，更新`paper_v73/main.tex`与README为10:30UTC阶段快照。R11完整原生控制加入11方法开发主表，并新增同run初始化的6项配对区间表；明确10550真实更新/580跳步、5日志hit下降、FIT标签内改善与不同native_fusion条件距离分母。R13仅为41次一轮反向，其中5次envelope-only；唯一开发归属束从miss变early、误差3.462215m，不能写成输入覆盖即质量成功。R10/R14继续运行、R12/完整新cohort/20新日志模型质量尚未完成，原interim_r1作为历史快照保留。

新增`scripts/plot_worldsim_v73_architecture.py`，以现有Matplotlib绘制实际VGGT冻结前缀→可训练DPT→深度/多尺度特征→局部3D查询→patch head→规范表面→已知运动/背景/硬首交点组件图。明确只读标定/尺度/轨迹及原生控制的PCA替换路径，灰色中间表示不暗示梯度截断；没有将用户示例的BEV/occupancy组件当本项目设计。图为可编辑SVG、矢量PDF及PNG，论文Figure1实际嵌入；后续报告沿用用户要求的这种简明组件表达。

表格导出仅读取已归档JSON，未重新推理/训练或重新bootstrap；新增LoRA3D引用已核实官方ICLR2025 Spotlight项目与准确题名，并修正CAPA题名为单数Adaptation。20新日志TSDF构建已完成的状态也同步正文，同时保留model_quality未读边界。没有使用这些输入构建结果选择外部日志或参数。

本地现有TinyTeX编译7页成功，全部页面视觉检查通过、图表可读、无overfull/underfull或未定义引用；无新环境、无smoke/回归。归档`docs/autoresearch/worldsim_v73/paper/interim_r2.pdf`，源`paper_v73/`，图`paper_v73/figures/architecture_components.{pdf,svg,png}`；可读交付另存本地outputs的V73_INTERIM_REPORT_R2.pdf、源码zip和组件图。

10:26UTC实际进程R10/PID53472 epoch20、R14/PID68108 epoch2均正常，未重复启动、无新GPU队列。failure_ledger_delta=none；F02/F03/F04/F05/F09及既有边界继续，下一失败编号V73-F10。下一步分析完成后的R10，与R11同标签/目标比较；资源可用后执行已登记R12，独立安排完整visual-only新cohort训练，最终再读20新日志。整个V7.3未完成，自动跟进active，shutdown=false。

---


## V7.3 零LiDAR真实训练结果与R14启动（2026-09-08）

`WS-V73-M2-GLOBAL-ACTOR-01/20260908T085500Z__visual-only-full-track-one-epoch-s7304-r13` code11433ba4已done、PID67103退出。原metadata全部51对象/41 fit真实更新/5 dev有表面/5缺两类输入者保留缺失；41次呈现均有DPT/query正梯度，native_project变化.000271443，完整DPT32654562/query1670517参数、195.286933s、allocated8.170046GiB/RSS14.146515GiB、288份原冻结前缀。不是旧checkpoint的固定推理，也不是完整新cohort训练或充分拟合。

34个有正目标对象共54416点，7个无正目标者coverage=null且均有原始采样束，其中2次free>0、5次采样free已满足而只有box envelope梯度；不能把所有41更新写成真实传感器残差驱动。此零build LiDAR组的native_observed_points均0，DPT通过特征/生成表面接收梯度，native像素数据项实际不激活。6次训练使用coarse退路；最终fit41/43、dev5/8有表面，coarse为4/1，初始化8/0。

开发只有一条归属返回：初始miss变最终early，误差3.462215m、侵入3.262215m；表面距离.011909m/recall1不能覆盖错误首交点。开发near-box free .107936→.164967m、仅59个Actor–ray实例/2日志，不据此主张泛化或用退化bootstrap。FIT窗口hit.005556→.175000、free1.235780→.023596m，但recall.580556→.511111；其24返回/14对象/6日志位于训练标签范围内。完整412 fit新cohort训练仍待目标比较后独立纳入；不把R12/R14目标因素与输入cohort混改，不重复此一轮smoke。

R13原始summary/analysis/41行train及计数分析归档`docs/autoresearch/worldsim_v73/m2/global/visual_only_r13_*`，专项`WORLDSIM_V7_3_EMPTY_INPUTS.md`更新到真实执行状态。既有F02/F03/F05继续，无新增失败ID；R11负结果F09已由bf04ef32记录，下一编号F10。

R14已按bf04ef32登记后实际启动：`20260908T100000Z__population-native-full-track-beam-range-s7304-r14`，PID68108，日志`/root/autodl-tmp/controller_logs/v73_population_native_beam_r14.log`。原生30轮/full_track/旧489 cohort/M1r3 seed7304，只有free从R11的hard range改为beam_tube_range .03m/res32；真实DPT更新已产生，query固定。10:07UTC epoch1 allocated2.358664GiB、进程3066MiB；R10/PID53472 epoch19、进程12570MiB，合计15636MiB、oom/oom_kill均0。R12未启动、无自动GPU队列，待完整query资源；当前无确证资源不足。

用户新增写作偏好已保存用户级与项目AGENTS并push133e9a29：技术报告/paper必须有简单组件图，以模块、箭头和少量标签说明输入/组件/数据流/输出。下一稿paper同步R11/R13及此结构图，interim_r1保留原时间快照。20新日志模型质量仍未读取，整个V7.3未完成；自动跟进active，shutdown=false。

---


