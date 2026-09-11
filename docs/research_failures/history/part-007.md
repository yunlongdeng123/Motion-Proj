# 历史原始记录 007

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V7.3 主联合训练30轮完成与观测域双向指标登记（2026-09-08）

r5恢复已完成30轮、11130次有效更新/呈现：7791来自完整恢复epoch，3339为新增，原中断107次未保存更新独立留档，合计实际执行至少11237次。当前仍在最终表面评价，不是整个run或V7.3完成。训练首轮→第30轮：native Huber测量加权0.804565→0.387529m，有监督Actor均值3.370621→1.069023m；采样target→surface均值0.050599→0.040186m；采样hard free0.332841→0.277535m，仍有明显波动。357个有视图对象中的native支持回退17→9；14个无相机姿态对象保留LiDAR路径。每轮原生监督346对象/173972测量。

两组梯度中位数DPT334.9837→175.9464、query12.9341→12.5636，均为全局裁剪前范数；非零梯度不等于解决物理优化，不能从不同模块范数直接推断梯度冲突或学习率失效。采样/权重随epoch变化，训练曲线没有开发泛化含义。图`m2/global/V73_JOINT_R5_TRAINING.png/pdf`已由完整30轮历史生成并检查可读，标明epoch21恢复及并行资源影响耗时，没有重复模型测试。

核对[SS3DM NeurIPS2024官方评价](https://github.com/THU-LYJ-Lab/SS3DM-Benchmark)，明确稀疏GT不支持完整表面precision/Chamfer。新增`scripts/evaluate_worldsim_v73_observed_points.py`和`docs/WORLDSIM_V7_3_OBSERVED_METRICS.md`：在原heldout首返回明确属于Actor的固定束域，将每个方法字面首交点P与原测量G进行双向点集precision/recall/F-score(0.2m)及非平方Chamfer两均值之和；不截掉远错误，不把未观测表面当空。不替代连续表面单向召回或逐束early/free/miss，NN可以掩盖邻束匹配错误，完整表面GT限制仍保留。

登记`WS-V73-M4-OBSERVED-POINTS-01/20260908T043000Z__development-fixed-surfaces-r1`，全75旧开发Actor/5日志，PCA/r5/r6/r9/native fusion/CAPA2/Ada Y-up2；r5完成后读取保存表面做CPU实际评价，无神经重推理。无测量对象保留不可用；有测量无预测时P/R/F=0、距离不可定义并单列缺失/有效样本数，不能只按成功返回距离挑方法。不同标签、TTA和空输入路径差异不混为同预算，完整说明在新文档。

上述观测域指标是F05评价边界的补充，没有新增一套验收或新失败编号；F01/F02/F03/F04/F05继续active，F06缓解，F07接口修正但物理问题保留，F08恢复训练完成但需最终评价且原原因未知，下一编号V73-F09。r11继续训练；旧AV2 joint、外部prefix、r10仍未启动，无等待启动队列。整个V7.3未完成，shutdown=false。

---


## V7.3 主联合训练过程汇总与旧AV2联合接口登记（2026-09-08）

r5恢复已进入epoch30收尾，r11原生full_track继续训练。新增`scripts/summarize_worldsim_v73_training.py`，直接整理既有train.jsonl的逐epoch实际呈现/更新、可用视图、native支持回退、原生监督计数、几何损失、两模块梯度及耗时。原生Huber同时按有效测量加权和有监督Actor等权，缺监督对象不记为零误差。恢复历史中的完整epoch只计一次，另记原中断107次未保存呈现；资源同时保留原进程最后观测耗时与恢复耗时，不把恢复时长冒充完整训练成本。新增训练过程图入口，最终评价完成后才归档完整曲线与最终质量结果。训练损失变化不承担泛化或物理成功主张。

共同Actor统计器在只有一个独立日志时不再生成退化的bootstrap区间，保留实际配对差值并标注跨日志不确定性无法估计；5日志/20日志定义不变。此次是已有统计语义的修正，无模型重跑或额外smoke/回归。

登记旧AV2固定主模型任务 `WS-V73-M4-AV2-FIXED-01/20260908T041500Z__old-development-joint-r5-r1`：等r5完成并退出后，使用其固定恢复最终权重、既有旧AV2 28视图前缀与原固定米制对齐，评价原21Actor/两heldout扫描。第一次让主joint实际经过全部七相机embedding插值、letterbox有效矩形和3D投影查询；先前native fusion/r9没有检验这个完整视觉查询接口。仅旧开发窗口，无优化、不使用新20日志选择实现约定。之后在同一旧背景上进行统一场景读出。

登记外部前缀 `WS-V73-M4-AV2-PREFIX-01/20260908T043000Z__external20-28view-prefix-r1`：原固定20日志、每日志28视图672×672，独立图像patch encoder分块7，但全部28视图共同进入聚合器；仅缓存完全冻结的4个官方DPT输入层。原预训练头与build测量估计固定IRLS米制尺度，再安装M1r3头；不读heldout质量、不优化新日志、不缓存可训练头输出。旧窗口实测prefix allocated7.64359GiB，每日志缓存约1.97GiB，20日志预计约39.5GiB；当前数据盘112GiB可用。须在r5退出后按实际空闲显存启动，保持全部相机/时间，不因24GiB设计削减输入。

两项仅登记，尚未启动，无后台等待启动队列。r10完整轨迹joint仍按已登记配置从原M1r3/seed训练，不拿r5恢复权重冒充同初始化标签对照。F01/F02/F03/F04/F05仍active，F06缓解，F07轴修正已训练但物理问题保留，F08待恢复最终结果且原退出原因未知，下一编号V73-F09。整个V7.3未完成，shutdown=false；所有训练、评价、数据和控制进程正常结束及结果保存/push后才可能最终关机。

---


## V7.3 轨迹编辑组合完成与外部20场景构建完成（2026-09-08）

`WS-V73-M4-TRAJECTORY-EDIT-01/20260908T033000Z__old-av2-r9-lateral2m-r1` 已以codee45118ef正常完成，CPU5.5590s/RSS0.95334GiB。旧AV2开发窗口全部21个固定r9 Actor规范BVH与原build雕刻背景共同读出；分别编辑build/轨迹元数据确定的全部4个移动Actor，按每束实际时刻施加局部+Y 2m，其他层不变。四个独立编辑情景各使用同187494条原有返回束，未重训、未重新生成形状、未读旧实测距离作为编辑GT。

| 编辑Actor前缀 | 原先Actor首交点 | 编辑后Actor首交点 | 新增遮挡 | 释放束 | 释放到未知 | 释放到其他Actor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6247b383 | 6 | 4 | 3 | 5 | 5 | 0 |
| 9054f80c | 2 | 2 | 2 | 2 | 2 | 0 |
| 94dede14 | 19 | 0 | 0 | 19 | 19 | 0 |
| f696fa0d | 38 | 22 | 21 | 37 | 31 | 6 |

四个独立情景合计63次束释放，57次没有已存表面接续、6次接续其他Actor，接续背景0；不是同时移动四辆车的单一场景统计。保留未知暴露，没有补洞。该演示证明固定规范表面与时变轨迹的机械组合，不能证明反事实真实性、碰撞安全、完整no-return传感器仿真或视觉主方法优越。第三辆编辑后在原有束子集上没有首交点也完整保留。图`m4/V73_TRAJECTORY_EDIT.png/pdf`展示全部4车的首个登记扫描，区分灰色规范中心与彩色真实逐束交点；修正了最初图中中心重叠可能与交点混淆的符号，未重跑模型。逐情景/帧统计在`m4/trajectory_edit_r1_summary.json`。

外部场景 `WS-V73-M4-AV2-SCENE-DATA-01/20260908T031000Z__external20-per-return-build-background-r1` code8a0cf6a4 已完整完成20日志，1519.9831s，CPU父RSS0.12975GiB、子最大0.91083GiB；父47224、外层shell47223和子任务均已退出。936Actor、105755个只读轨迹姿态、80build/40heldout扫描、7815152/3908250原始返回，全部原身份与空输入保留。6839807个背景点生成54718456个三角面，按预先确定build-free单次雕刻删除2159469（3.9465%），保留52558987面；build冲突束906573→23，剩余侵入和183.0328m。未因外部残差改变参数或循环删面，不声称严格零违规；无heldout模型质量评分或外部网络推理。

完整索引含原始轨迹数组76684364字节，保留于source run，不在Git重复巨大轨迹。新增`scripts/summarize_worldsim_v73_scene_construction.py`仅整理构建计数和既有build诊断，归档`m4/av2_external20_scene_construction.json`，保留源路径和各日志执行资源；不是新的验收门控。原始背景、未雕刻控制、heldout束、轨迹与全部构建日志持续可回溯。

F04有了轨迹组合的实际证据，但已存背景稀疏/未知暴露和错误前表面仍未解决；F01/F02/F03/F05继续active，F06直接数据约束缓解，F07轴接口落实，F08恢复推进但原退出原因未知，下一编号V73-F09。主r5恢复最近epoch29、native-only full_track r11 epoch8，实际GPU峰值仍10.19917/2.34640GiB，cgroup oom/oom_kill=0，数据盘约112GiB可用。当前仅这两项主训练在运行，无训练启动队列。

下一步在r5完成最终评价并退出后汇总主视觉几何结果、同信息LiDAR/原生融合/强基线差值，实际运行旧AV2七相机joint接口，然后安排已登记full_track joint r10及外部冻结前缀。最终确认仍等待方法选择，不能将本轮应用演示或数据完成算作整个V7.3完成。shutdown=false；继续15分钟自动推进，真正研究收尾保存/push且确认所有任务/控制器结束后关机。

---


## V7.3 只读规范表面的轨迹编辑演示登记（2026-09-08）

主r5恢复当前epoch28、native-only full_track r11 epoch7，训练仍正常；外部场景CPU构建最近15/20日志，无新外部模型质量结果。按计划13.4补充组合应用，先阅读[Street Gaussians ECCV2024官方代码](https://github.com/zju3dv/street_gaussians)及[NeuRAD CVPR2024论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Tonderski_NeuRAD_Neural_Rendering_for_Autonomous_Driving_CVPR_2024_paper.pdf)。迁移Actor/背景分解与刚体actor shift演示思想，使用现有不透明三角面硬排序；不复现其外观渲染、传感器概率或以编辑展示证明其/本方法真实性。

新增`scripts/demonstrate_worldsim_v73_trajectory_edit.py`，登记 `WS-V73-M4-TRAJECTORY-EDIT-01/20260908T033000Z__old-av2-r9-lateral2m-r1`，当前尚未执行。使用已完成旧AV2开发窗口02678d04…的完整21Actor和固定LiDAR-only r9表面，每个Actor规范BVH只加载一次；编辑对象为build/轨迹元数据中速度>2m/s的全部4辆，未按预测或heldout质量筛选。对每辆分别施加局部+Y 2m偏移，其他Actor/背景/射线不变，在两次原始heldout扫描每束真实时刻组合，不修改规范形状、尺寸、法向、连接或模型参数。

记录同一187494条原始有返回束子集上的first-owner变化、新增遮挡、释放到背景/其他Actor/未知的计数；旧实测深度不作为编辑真值，不计算编辑accuracy、free违规或置信区间。未观测no-return束没有凭空补成完整扫描，露出的未知背景也不补洞。偏移未检验交通可行性或碰撞安全，因此本演示仅证明固定表面+轨迹的机械组合能力。随后可在完成的主模型表面上复用入口，当前r9示例不承担主视觉方法的增益主张。

F04组合/背景未知边界继续active，F01/F02/F03/F05也保留；F06直接数据约束缓解，F07轴接口已落实，F08恢复推进且原退出原因未知，下一编号V73-F09。无新增hash、门控或模型smoke/回归。整个V7.3未完成，shutdown=false；主模型与数据控制器全部正常继续。

---


## V7.3 固定点集密度揭示覆盖与物理冲突；外部场景构建登记（2026-09-08）

code5dfa3948完成 `WS-V73-M2-ADAPOINTR-DENSITY-01/20260908T030000Z__development-native-vs-matched-r1`，CPU10.6413s/RSS0.66930GiB，所有75旧开发Actor/5日志，无训练或重新神经推理。Ada r2原生16384点的target→point0.088114m、recall86.9506%；匹配中心0.162732m、77.7732%。相同0.06m/20邻居PCA从匹配预算扩大到全部点：hit14.2906%→39.0141%，early8.5389%→39.3882%，miss54.2423%→18.2534%，free0.122057→0.341008m，target→surface0.113749→0.053369m，recall84.6032%→87.9661%。

密度变化的5日志配对：hit+24.724pp、bootstrap95%[+14.343,+36.328]pp；early+30.849pp、[+16.340,+42.835]pp；miss−35.989pp、[−45.407,−25.494]pp；free+0.218951m、[+0.108226,+0.334258]m。移动9对象/2日志全密度hit63.2750%、early34.8396%、miss1.0245%、free0.168564m。采样/显式化影响已证实，加密并未解决错误前表面。此为不同输出密度的诊断，不能放入匹配预算主表当方法增益；原GPU匹配结果与CPU BVH还存在数值实现差异。依据APSS/2DGS的表面定义思路分离评价，不引入opacity。F02/F03保留；若扩大Ada训练，必须同步约束实际最终表面，不能只在评价时加密。

同code完成 `WS-V73-M4-SCENE-COMPOSITION-01/20260908T030000Z__development-adapointr-yup-carved-r6`，CPU3.8597s/RSS0.77354GiB，同雕刻背景/原始416704束/6场景5日志，使用r2匹配曲面。cohort11886束的等日志hit11.0410%、early33.6000%、miss36.5152%、free0.951842m。相对同背景Ada r1，cohort hit−0.682pp、[−2.892,+1.941]pp，free+0.001128m、[−0.016595,+0.014165]m；相对r8，hit−11.418pp、[−15.326,−8.430]pp，free+0.042653m、[+0.011555,+0.089395]m。相对PCA miss−8.009pp、[−14.314,−1.704]pp，但hit−10.874pp、[−21.286,−0.461]pp及free+0.048074m、[+0.011879,+0.098823]m。全密度表面尚未做场景读出。F04仍在，坐标修订不能作为场景成功证据。

汇总器支持跨已保存run的方法配对，角色和真实日志数来自数据，不再固定“5开发日志”；单日志不输出退化bootstrap区间。仅汇总本次真实结果，无重复模型测试。Ada方法报告改为当前完整方法/预算/结果/密度/场景与限制，历史执行证据保留在三账本。`docs/autoresearch/worldsim_v73/m2/global/adapointr_density_r1_summary.json`、`V73_ADAPOINTR_DENSITY.png/pdf`与m4/scene_composition_r6_summary.json、scene_composition_r6_paired.json均已保存；图已检查排版。

旧AV2逐束场景方案已完成真实开发验证，现登记 `WS-V73-M4-AV2-SCENE-DATA-01/20260908T031000Z__external20-per-return-build-background-r1`，新增CPU串行构建器 `scripts/prepare_worldsim_v73_av2_scene_batch.py`，已以code8a0cf6a4实际启动，CPU父进程PID47224（外层等待shell47223），第一日志已完成构建；日志`/root/autodl-tmp/controller_logs/v73_av2_external20_scene_data_r1.log`。复用原20身份/936Actor/4build+2heldout时刻、0.06m PCA、+.1m已知框排除及首回波前.2m的build-only删面；不按外部质量换日志或参数。输入坐标/背景构建与heldout束格式保存可先完成，外部网络和heldout质量评分继续等待方法选择。其父进程及子进程属于关机前必须结束的任务。

主r5恢复最近epoch26、native-only full_track r11最近epoch5，GPU allocated峰值仍10.19917/2.34640GiB；cgroup oom/oom_kill仍0。继续两训练，r5完成后汇总真实主结果并验证旧AV2七相机joint路径，再安排full_track joint r10和外部前缀；不并发挤入已知需7.64GiB的28视图前缀。F01/F02/F03/F04/F05 active，F06直接数据约束缓解，F07轴接口落实，F08恢复推进/原退出原因未知，下一编号V73-F09。整个V7.3未完成，shutdown=false，无训练启动队列，15分钟自动推进保持。

---


## V7.3 AdaPoinTr坐标修订结果、AV2外部输入完成与旧域场景实测（2026-09-08）

AdaPoinTr Y-up r2 `WS-V73-M2-ADAPOINTR-01/20260907T231000Z__population-full-track-pcn-yup-s7307-r2` 已完整结束，code acd91103，30epoch/11130呈现/2790更新，5109.12s，GPU allocated峰值1.00658GiB、RSS2.07924GiB；PID37887已退出。保持r1输入/全轨迹标签/seed/预算，仅迁移官方PCN轴接口。75开发Actor/5日志：hit14.2906%、early8.5389%、miss54.2423%、free0.122057m、target→surface距离0.113749m、recall@0.2m84.6032%。相对自身初始化，距离−0.057385m、日志bootstrap95%[−0.086634,−0.037315]m，recall+11.792pp、[+4.380,+18.482]pp；early+5.573pp、[+2.968,+8.172]pp。相对r1，free−0.024503m、[−0.086788,+0.032201]m，hit−1.761pp、[−6.766,+3.705]pp，距离+0.016169m、[+0.000341,+0.035096]m，不能宣称坐标修复带来一致物理改善。F07接口修订已执行并完成训练，原r1局限保留；物理缺陷归入仍active的F02/F03，不扩大成完整补全路线失败。

相对r8，Ada r2 hit−16.712pp、[−23.434,−8.407]pp，free+0.087737m、[+0.038663,+0.136812]m，recall+12.658pp、[+2.480,+26.112]pp；相对r9，hit−18.178pp、[−26.080,−8.322]pp，free+0.083514m、[+0.030030,+0.136998]m。移动9Actor/2日志hit4.1164%、early8.0231%、miss76.3141%、free0.057109m、距离0.104197m、recall98.8670%，再次说明0.2m邻近覆盖不能代替字面命中。8个原空输入和23个无自有留出返回仍在全体报告；原生16384点与匹配曲面分别保存。完整配对见 `docs/autoresearch/worldsim_v73/m2/global/adapointr_r2_analysis.json`。

针对覆盖/首交点脱节，先查阅[APSS SIGGRAPH2007](https://cgl.ethz.ch/research/past_projects/apss/)与[2DGS SIGGRAPH2024官方实现](https://github.com/hbb1/2d-gaussian-splatting)。前者体现局部拟合对稀疏点集表面定义的作用，后者显式区分有向surfel求交与网格提取；不迁移可学习opacity替错误前景变透明。先登记一次固定输出诊断 `WS-V73-M2-ADAPOINTR-DENSITY-01/20260908T030000Z__development-native-vs-matched-r1`：只读r2最终点集，在全部75旧开发Actor上对比原生点集覆盖、匹配中心覆盖和全部16384点同0.06m/20邻居PCA显式化的硬交点。它改变输出密度，作为敏感性分析而非匹配预算主表，不按结果挑Actor/半径。另登记r2匹配表面的雕刻背景组合 `WS-V73-M4-SCENE-COMPOSITION-01/20260908T030000Z__development-adapointr-yup-carved-r6`。两项当前尚未执行；不预先归因于支持稀疏或法向。

外部数据 `WS-V73-M4-AV2-DATA-01/20260908T020000Z__external20-common-windows-r1` code f70d099c 已完整导出/合并，2061.27s，父RSS3.66828GiB、子最大RSS1.26955GiB，PID41941及子任务已退出。原20日志全部保留，936刚体Actor：878 ready、58 unavailable_input，221移动>2m/s、108无速度；1个无可用Actor相机时刻，119个无自有heldout回波。560个完整相机输入，原始11723402返回=build7815152+heldout3908250，姿态未知返回0、框归属重叠7077。输入/标注值已处理，但未做外部网络推理、模型质量评分、共享优化或按覆盖更换身份。新域最终确认尚未发生，训练仍仅旧fit；索引归档至coverage/av2_external20_index.json。

旧AV2逐束场景三项登记均以code ad457f3c成功执行：数据34.5845s/RSS0.85305GiB，原背景评价11.3142s/RSS0.96797GiB，雕刻背景11.0018s/RSS0.94923GiB。21Actor、187494原始heldout束，cohort5257、移动cohort88、框边界2708，单条历史开发日志，不能用射线量或单日志bootstrap当独立确认。2916480背景三角面中仅按build自由空间删除211100（7.24%）；71498条build冲突束降至0，此结论仅此窗口中央束/当前容差，不能外推所有几何无侵入。

旧AV2仅背景全束free0.733877→0.513045m、early20.5169%→14.1498%、hit44.3774%→48.5205%，miss31.1071%→32.1333%。雕刻背景cohort：PCA hit62.9446%/early9.5872%/miss21.6663%/free0.060124m；native fusion 42.6479%/33.3080%/18.2043%/0.359073m；r9 54.2134%/9.2068%/17.2912%/0.074238m。移动仅88束时PCA hit21.59%/miss73.86%，r9 44.32%/37.50%，native fusion22.73%/64.77%；不从这一旧单日志选外部赢家。背景对cohort的提前遮挡仅约0.38%，与nuScenes约25.19%明显不同，需要分别报告数据与背景误差；F04仍active。详见m4/av2_old_scene_r1_summary.json、av2_old_scene_r2_summary.json、av2_old_scene_data_index.json。

主joint r5恢复PID39009与native-only full_track r11 PID38460仍运行，占GPU约13810/2920MiB；未启动r10。磁盘约116GiB可用，当前没有真实资源不足证据。F01/F02/F03/F04/F05 active，F06直接数据约束缓解，F07坐标接口已落实，F08已恢复但原退出原因未知，下一编号V73-F09。整个V7.3未完成，shutdown=false；继续自动推进，最终必须保存/push并确认训练、评价、数据及启动控制进程全部结束才关机。

---


## V7.3 AV2逐束时刻场景组合实现登记（2026-09-08）

依据AV2官方Sweep时间/ego补偿约定和Open3D显式BVH接口，新增 `SurfaceBVH.cast_per_time`：一个Actor只建一次固定规范表面BVH，每块8192条束按实际纳秒时间插值已知刚体位姿，变换原点和方向，保持米制距离并参与背景/全部Actor的统一最近正交点。缺失轨迹或传感器姿态的束预测为缺失并计入原始分母；没有把整扫描单姿态称作逐点查询。既有nuScenes单姿态路径保留。来源 https://github.com/argoverse/av2-api/blob/main/src/av2/structures/sweep.py ，https://www.open3d.org/docs/latest/cpp_api/classopen3d_1_1t_1_1geometry_1_1_raycasting_scene.html 。

新增 `scripts/prepare_worldsim_v73_av2_scene_geometry.py`，从同一Actor数据登记的四build/两heldout扫描构建场景：build原始点在所有有效时刻已知框+.1m之外形成背景支持；世界端点只应用一次参考ego变换，束原点和归属使用各自发射时间。保存未雕刻背景和相同build-only自由空间单次雕刻背景，heldout束及归属完全一致，range<1m仅分层、不删支持/评价束。已知轨迹按原始timestamps/矩阵保存，不为每个模型重估。

共同场景评价器接入逐束轨迹、可选背景文件、移动cohort返回组和按输入角色分别统计；external_confirmation不会硬编码成“旧5日志开发”，未知pose也单列。CPU评价环境为既有Python3.9/Torch2.1/Open3D0.19，补装pyarrow21.0.0用于读取Feather，复用现有环境，无新环境或模型测试；主训练环境pyarrow24.0.0保持原状。

登记旧开发数据 `WS-V73-M4-AV2-SCENE-DATA-01/20260908T024000Z__old-development-per-return-r1`；同模型两背景比较 `WS-V73-M4-AV2-SCENE-01/20260908T024000Z__old-development-uncarved-r1` 与 `20260908T024000Z__old-development-carved-r2`，使用完整21Actor、原始两次heldout扫描、已完成的固定native fusion/r9和PCA。源码尚未执行本场景组合，不把已实现接口当成功证据；先在旧开发窗口跑完整数据与实际查询。

新20日志CPU转换仍在原PID41941下进行，主r5恢复和native-only r11继续正常训练，Ada Y-up r2处于最终评价收尾。F01/F02/F03/F04/F05/F07仍active，F06直接数据配置缓解，F08恢复推进但退出原因未明，下一编号V73-F09。整个V7.3未完成，shutdown=false；未启动r10，无训练启动队列，数据父进程及子任务须完成后才可能执行最终关机。

---



## V7.3 外部20日志通用输入已启动与场景结果图（2026-09-08）

`WS-V73-M4-AV2-DATA-01/20260908T020000Z__external20-common-windows-r1` 已实际启动，codef70d099c，CPU父进程PID41941，日志 `/root/autodl-tmp/controller_logs/v73_av2_external20_export_r1.log`。最新3/20日志导出完成，原身份顺序、四build/两heldout时间和全部七相机保留；后续日志由该父进程顺序执行，属于关机前必须结束的数据控制进程。尚未完成整个20日志合并。

原始图像/点云和标注数值已用于格式转换、逐点坐标、已知轨迹归属与输入/heldout分离，不能再写“新数据数值未读取”。没有运行新域网络推理、重建质量评分或共享梯度更新；方法选择仍依据既有开发日志。exporter没有按新输入覆盖或预测好坏挑选/删除身份，空输入也保留。每日志处理日志、input_protocol和case均在注册run下，真实质量确认仍待最终模型选择后进行。

背景雕刻结果图由已保存r4/r5汇总生成，未重复推理：`docs/autoresearch/worldsim_v73/m4/V73_BUILD_FREE_BACKGROUND.png/pdf`，脚本 `scripts/plot_worldsim_v73_background_carving.py`。上排显示5日志背景free/early/miss的配对变化，下排显示同背景组合的cohort free/hit/miss；均值与个别日志都保留，明确Ada r1轴限制及不同训练预算。已检查排版并修正一处长标题截断，未新增模型smoke/回归。

后续AV2全场景组合需保持逐点时刻：当前nuScenes场景入口使用每扫描一个Actor刚体矩阵，不能直接冒充AV2逐点已知轨迹。下一实现应按每束时间把原点/方向转入同一Actor规范系，再通过其固定BVH求交并全局硬排序；背景使用参考ego已补偿端点和逐点束原点。先在旧AV2开发窗口接通，20新日志不用于选择几何时序约定。

三项训练正常，最近Ada Y-up r2为epoch25；r5恢复和native-only full_track继续，未启动r10。F01/F02/F03/F04/F05/F07仍active，F06直接数据配置缓解，F08恢复推进但退出原因未知，下一编号V73-F09。整个V7.3未完成，shutdown=false；必须完成研究、保存/push并确认训练、评价、数据及其启动控制进程全部结束后才关机。

---



## V7.3 构建自由空间雕刻的完整场景结果与外部数据整理登记（2026-09-08）

code8723cfc7完成背景数据r3及场景r4/r5：构建雕刻34.2274s、RSS6.70612GiB；原背景7方法组合5.5153s/RSS0.73833GiB，雕刻背景同7方法5.2191s/RSS0.73386GiB。6场景/5日志、每方法416704条原始heldout束全部保留，无GPU或网络模型推理。原4850928个背景三角面中移除193273个（3.984%）；原四build扫描有85138条束与背景free冲突，雕刻后剩4条、侵入总和32.7904m。等深重合BVH/数值边界等可能留残差，未进一步循环删面，也未假称严格零违规。新旧数据和全部输出保留。

仅背景的5日志等权全束free 0.319106→0.265644m，差−0.053463m、bootstrap95%[−0.080883,−0.026042]m，5/5降低；early12.2215%→10.6776%，差−1.544pp、[−2.634,−0.576]pp；miss52.2716%→53.2793%，差+1.008pp、[+0.491,+1.415]pp。hit+0.122pp、[−0.080,+0.430]pp。侵入改善伴随覆盖代价，必须并列。cohort返回被仅背景提前挡住的比例27.0786%→25.1910%，free1.035546→0.893276m；仍很大，F04没有解决。

| 雕刻背景上的Actor方法 | 全束hit | 全束early | 全束miss | 全束free m | cohort hit | cohort early | cohort miss | cohort free m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LiDAR PCA | 26.7218% | 10.7892% | 52.5204% | 0.268128 | 21.9148% | 29.8001% | 44.5240% | 0.903769 |
| r8 | 26.4275% | 10.8267% | 52.5444% | 0.268368 | 22.4588% | 30.0254% | 39.2326% | 0.909189 |
| r9 event | 26.4905% | 10.8402% | 52.5048% | 0.268768 | 22.6124% | 29.7411% | 39.0463% | 0.912482 |
| CAPA r2 | 26.5422% | 10.9154% | 52.4881% | 0.273695 | 18.0507% | 34.8447% | 39.3287% | 0.959045 |
| AdaPoinTr r1（轴限制） | 26.2901% | 10.9687% | 52.3971% | 0.278756 | 11.7231% | 35.4275% | 33.9002% | 0.950714 |
| native fusion | 26.5554% | 10.9088% | 52.2796% | 0.276713 | 20.0435% | 33.4639% | 39.2864% | 0.928830 |

r9−r8的cohort hit+0.154pp [−2.449,+2.347]，early−0.284pp [−1.622,+0.773]，miss−0.186pp [−2.276,+2.411]，free+0.003293m [−0.000556,+0.009766]m；事件项仍未形成明确场景级优势。r9−PCA的cohort free+0.008713m [+.000336,+.018847]m，不能用returned MAE或降低miss掩盖侵入。

PCA组合的注释框边界带（代理非真接触边界）free差−0.149531m [−0.411545,−0.010161]m，early−2.314pp [−4.623,−0.336]，miss+1.813pp [+0.317,+3.836]。支持删除使边界缺失增加；没有解决边界完整性。完整各组计数、逐日志及配对保存在m4/scene_data_carved_r3_index.json、scene_composition_r4_summary.json、scene_composition_r5_summary.json、scene_composition_r5_paired.json。

后续场景主比较采用此build一致性更强的r3背景，同时保留原r2背景敏感性证据；这是开发集方法选择，所有Actor方法必须同背景，不使用新20日志挑背景。该雕刻仅显式三角面近似，不是TSDF复现或独立论文贡献。

登记 `WS-V73-M4-AV2-DATA-01/20260908T020000Z__external20-common-windows-r1`，使用新脚本 `prepare_worldsim_v73_av2_cohort.py` 顺序调用已在旧日志跑通的相同exporter，完全复用先前20个身份/时间/载荷清单。仅CPU坐标、轨迹归属和输入/heldout分离整理，无模型推理、共享拟合或重建质量选样；会读取新传感器数值用于格式转换，不能再写成“新数据数值从未读取”。所有身份和空输入保留，heldout仅保存为评价射线。此时仅登记，尚未启动。最终模型/结构选择继续只用既有开发日志；新域指标未计算。

failure_ledger_delta=update F04已实现build-only几何对策且量化覆盖代价，F05准备外部通用输入；F01/F02/F03/F04/F05/F07仍active，F06直接数据配置缓解，F08恢复推进但原因未知，下一编号V73-F09。三项正常训练继续，整个V7.3未完成，shutdown=false；整体完成并保存/push、确认无研究/数据/待启动任务后才关机。

---



