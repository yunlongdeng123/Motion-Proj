# 历史原始记录 005

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V7.3 原生强控制R11完成与同目标R14登记（2026-09-08）

`WS-V73-M2-GLOBAL-ACTOR-01/20260907T233000Z__population-native-only-full-track-s7304-r11` code9541ac7d已完成，PID38460退出。30轮11130呈现/10550实际更新/580无梯度跳步；580中420无相机、160有相机但原生支持为空而退回固定LiDAR，均无native build对应、两组梯度均0，涉及22个Actor。完整DPT32654562参数更新，query0可训练；native_project最大变化.005223576，wall30855.513131s、allocated2.346402GiB、RSS34.607677GiB；无恢复、无OOM。所有489对象保留，371 fit更新候选、67 dev可预测、51零LiDAR缺失；开发无梯度。

开发75/5日志R11最终hit.176111/early.076432/miss.600949/free.164965m/单向surface distance.168504m/recall.751499。相对同run初始化：hit−5.166pp，95%[-10.920,-1.125]pp，5/5日志下降；early−1.512pp[-2.872,-.660]pp，5/5下降；free+.014661m[-.006742,+.031388]m，miss+2.741pp及recall−2.979pp区间跨0。FIT日志hit+2.276pp[+.925,+3.701]pp、early−3.707pp[-6.431,-1.301]pp；FIT是训练标签范围内读数，不是泛化成功。

同full_track的LiDAR R7相比，R11 hit−14.144pp[-18.582,-8.623]pp、miss+10.197pp[+4.691,+16.870]pp；R9相比free+.126422m[+.046265,+.199240]m，但R9同时使用beam/event，不能归因于适配架构本身。R11与旧native_fusion的条件surface距离不能直接减两个均值：旧融合额外对5个零LiDAR开发对象生成支持，R11保留缺失；同ready67对象的初始化与融合完全相同，配对距离差采用共同可评价对象。R10仍运行，不能提前宣称native或query赢得匹配比较。

新增科学候选负结果V73-F09：本配置原生完整适配改善训练拟合却降低开发硬hit，归因尚未解决。先检索[LoRA3D ICLR2025官方](https://520xyxyzq.github.io/lora3d/)与[CAPA官方](https://research.nvidia.com/labs/dvl/projects/capa/)的场景校准/有限适配，二者不证明当前共享DPT失败原因，已有CAPA R2负结果保留。直接迁移优先完成既定R10/R12几何目标对照，不从一个控制结果否定视觉适配、不叠加confidence屏蔽free、不重跑原CAPA。更受限的DPT适配或build侧场景校准仅在匹配证据支持后作为独立因素研究，不读20新日志调参。

登记R14：`WS-V73-M2-GLOBAL-ACTOR-01/20260908T100000Z__population-native-full-track-beam-range-s7304-r14`。将此前“R12若采用beam目标须有匹配native控制”提前并行实施，仍只有原生融合/空间查询两个主候选。R14与R11共同M1r3/seed7304/full_track/30轮/AdamW1e-5/native1/free.5/event0/旧489 cohort，仅free改为beam_tube_range、width.03m/res32；复用R11同输入初始化与原PCA，重新训练而非resume。与R12目标一致，避免用hard-free R11冒充匹配控制。冻结前缀使用现有mmap执行方式，不减少24视图/分辨率，当前显存允许原生控制与R10并行；R12仍待完整query训练资源。登记时R14未启动，无自动GPU队列。

R13一轮真实visual-only检查亦已完成：code11433ba4，全部51原metadata对象、41更新、195.286933s、allocated8.170046GiB/RSS14.146515GiB；完整新输入路径已发生真实DPT/query反向，专项结果尚在整理，不当作完整训练或质量成功。R10继续，20日志输入已就绪且模型质量未读。R11原始summary/analysis/training归档m2/global/population_native_r11_*，过程图已按真实模式生成检查；failure_ledger_delta=add V73-F09，F02/F03/F04/F05持续，下一失败编号V73-F10；整个V7.3未完成，shutdown=false。

---


### V73-F09：共享原生DPT强控制的开发硬命中退化

- 最终独立确认补充（2026-09-09 21:30 UTC）：最终r7开发的联合增量与旧native-only负结果分开保留；但20日志AV2确认中Joint r7不及匹配LiDAR r6，不能把开发正结果升级为跨域有效主张。此最终联合对照不等于本条旧native-only控制，也不证明所有DPT适配无效。 证据docs/autoresearch/worldsim_v73/final_confirmation/{analysis,models}.json及配对图；run WS-V73-FINAL-CONFIRMATION-01/20260909T135000Z__fixed-r7-r6-external20-r1，执行code6499d34d，全部五方法936对象/20日志/0优化更新。878 ready、58缺输入、119无owned heldout对象均保留；含221移动>2m/s、108运动未知、556 build<100对象。模型/尺度/配置固定在读取外部质量之前，未按AV2再训练、调参或挑场景。

- 最终r7边界更新：开放chart/首面目标下，整条Joint相对同表示LiDAR r6的hit、miss、distance已有区间不跨0的开发增益。F09描述旧native-only强控制，不能据此宣称所有DPT适配无效；r7仍有free风险并待外部确认。

- 分类/状态：科学候选负结果；active（R11执行done，机制归因未定）。
- 观察：R11在旧完整489对象、20 FIT日志的full_track标签下完成30轮/10550实际更新，原生DPT真实变化.005223576。FIT hit+2.276pp，而5个开发日志hit均下降，平均−5.166pp、95%日志bootstrap[-10.920,-1.125]pp；开发early−1.512pp，但free与recall未显示整体改善。未用新20日志选模型。
- 推翻范围：本R11配置没有成为更好的开发物理表面重建模型。完整微调和更多FIT测量本身不保证硬首返回提高；没有证明全部视觉几何适配、LoRA或原生表示失败。
- 根因边界：训练/开发差异是事实，尚不能区分观测污染、共享参数漂移、离散支持/PCA读出和目标语义。580跳步的原因是无有效原生梯度，不是低loss被当作已优化。原生融合与R11条件距离的不同可评价对象不可混减均值。
- 对策/复开：保留R10同标签空间查询与R11的匹配比较；R14仅将R11的hard free替换为与R12同一beam_tube_range，其他信息/初始化/标签/优化条件不变。LoRA3D/CAPA官方有限适配和场景校准作为后续迁移依据，不把预测confidence用于逃避已观测free，也不直接重跑已完成CAPA配置。
- R14补充（2026-09-08 19:05 UTC）：同原生路径仅改beam_tube_range，30轮/10550更新完成。R14−R11开发hit+2.945pp、95%[−.370,+7.506]pp，free+.001454m、[−.022062,+.026683]m，6项区间均跨0；未建立稳定修复。相对同beam的LiDAR R8，hit−10.446pp、free+.132099m、recall+5.253pp，三项区间均不跨0。R14相对初始化hit差区间跨0，不把R11的显著下降转抄到R14。根因仍未唯一定位；覆盖/物理冲突不限于Query，表面和上游表示分别研究。新增证据`m2/global/population_native_r14_*`及同一专项报告，不新增失败编号、不关闭F09。
- 证据：`WS-V73-M2-GLOBAL-ACTOR-01/20260907T233000Z__population-native-only-full-track-s7304-r11`，code9541ac7d；`docs/autoresearch/worldsim_v73/m2/global/population_native_r11_summary.json`、`population_native_r11_analysis.json`、`population_native_r11_training.json`；专项报告`docs/WORLDSIM_V7_3_MATCHED_NATIVE_CONTROL.md`。


---

## V7.3 原生控制的训练过程报告入口修正（2026-09-08）

R11已到最后一轮，但尚未完成最终评价。原`plot_worldsim_v73_training.py`固定写“Joint”“371”与“epoch21恢复/107丢弃更新”，直接用于原生控制或visual-only新输入会错误归属实验。本次仅修正报告入口：训练摘要读取manifest的mode/include_visual_only；图题按真实模式选择，epoch呈现数、实际optimizer更新/跳步及恢复信息从已存日志计算。没有正中位数梯度的参数组明确标注，不在对数轴制造一条正梯度曲线；没有build native-depth监督时标记缺失，不记作零误差。

训练、模型和现存checkpoint未改动，无新神经推理或smoke/回归。代码已静态检查；R11最终summary产生后才用其真实完整日志生成和检查原生训练图，尚不声明该图已完成。原R5阶段图保留其正确恢复事实。此为同一已知硬编码报告问题的直接修复，无新的研究机制或文献卡点。

09:25UTC R11/PID38460完成10936/11130个计划呈现，仍运行epoch30；R10/PID53472继续，R12尚未启动。R13的51对象子集已准备，待R11进程实际退出再启动一轮真实反向；没有额外等待控制器或GPU队列。新20日志TSDF背景全部完成并已归档3295403f，质量仍未读。failure_ledger_delta=none（已有F02/F03/F04/F05持续，下一失败编号V73-F09）；整个V7.3未完成，shutdown=false。

---


## V7.3 新20日志逐返回TSDF背景输入准备完成（2026-09-08）

`WS-V73-M4-AV2-SCENE-DATA-01/20260908T084000Z__external20-vdb-per-return-r2` code75ac0563已完成；单CPU进程wall1249.572386s/RSS4.307499GiB、0 GPU/optimizer更新，PID64413已退出。全部预选20日志/936 Actor/80 build扫描；原7815152束，积分6839969背景返回，父PCA去重中心6839807，相差162个重复原测量。4149271个真实float64原点组，不平均或量化逐返回原点、不二次变换AV2已补偿端点、不限制返回数。

固定VDBFusion0.1.6 voxel.1m/trunc.3m、uniform unit weight、space_carving=true、fill_holes=false/min_weight0。原生6962322三角面，沿所有原build束在首返回前.2m单次雕刻146991片（2.111%），保留6815331；20日志剩余build矛盾束0/侵入距离和0m。这个零值是构建约束下的读数，不能表示新时刻质量或场景完整性，也不能据此改变参数。原PCA及两份TSDF mesh均保留，未另存大VDB体积。

3908250条原heldout返回仅链接，owner、逐返回sensor-known与只读Actor轨迹沿用父数据。model_quality_computed=false；20新日志还没有模型质量评价。最终方法/背景选择仍须先在开发结果上固定，不按新域成绩挑日志。紧凑逐日志证据`docs/autoresearch/worldsim_v73/m4/av2_external20_vdb_construction_r2.json`，专项报告`docs/WORLDSIM_V7_3_AV2_VDB_BACKGROUND.md`；没有把76MB轨迹index复制进git。

09:20UTC R11/PID38460进入epoch30但尚未结束，R10/PID53472 epoch15正常、合计GPU15505MiB、oom/oom_kill均0；R12未启动，R13真实反向仍待R11实际退出释放显存。R13的51原metadata对象输入链接已准备，不需要重新构建。LaTeX interim_r1 PDF保留其09:04阶段快照，新主训练结果完成后统一推进下一稿。failure_ledger_delta=none（F02/F03/F04/F05持续，下一失败编号V73-F09）。整个V7.3未完成，自动跟进保持active，shutdown=false。

---


## V7.3 LaTeX阶段证据稿同步（2026-09-08）

计划要求同步V7.3 LaTeX的任务、方法假设与证据表；原仓库只有历史paper/paper_v72，本次新增`paper_v73/main.tex`和中文README，保留旧论文。英文草稿写明当前原生DPT/空间查询实现、不同标签与预算的对照条件、全489分母/开发5日志、R5的coverage/free负结果、事件缺支持、输出密度、Actor TSDF适用性、背景构建覆盖成本及未完成实验。没有预写最终方法胜出、完整表面真值、已完成新域确认或arXiv发布。

新增`scripts/export_worldsim_v73_interim_tables.py`，只读取已归档population_joint_r5_analysis与actor_tsdf_r1_analysis JSON，生成10方法主表及6项配对差值/既有95%日志区间；没有重跑神经推理/统计bootstrap，没有按表格成绩改分母。明确R5完成30有效epoch/11130更新，同时披露107个丢弃更新、旧RNG未保存及恢复边界；CAPA TTA与AdaPoinTr的预算/预训练/标签差异不写成单因素结论。官方VGGT、AdaPoinTr、CAPA、DS-NeRF、Deformable DETR与nvdiffrast来源已核对，VDBFusion沿用此前核实资料。

草稿使用本地现有TinyTeX编译，移除本机未装的可选microtype排版包后通过；没有新增环境或下载大依赖。5页PDF及全部页面排版检查完成，主表和引用可读，没有溢出或未定义引用。研究代码没有新smoke/回归。PDF为阶段可读稿，最终结果尚待R10/R11/R12、visual-only训练与已登记20新日志确认，不能当成整个V7.3完成。

R13输入子集已由codebd71fcd1实际准备：`WS-V73-M2-EMPTY-INPUTS-01/20260908T085500Z__metadata-zero-lidar-training-inputs-r3`，51链接、43 fit/9日志、8 development/2日志、CPU.011242s；原metadata输入状态全部保留，零新标签选择/推理/训练。R13一轮真实反向仍等待R11退出，无自动GPU队列。外部20日志TSDF背景继续CPU构建，质量未读；R10/R11继续。failure_ledger_delta=none，F02/F03/F04/F05持续、下一失败编号V73-F09；shutdown=false。

---


## V7.3 登记零LiDAR完整子集的一次真实训练检查（2026-09-08）

现有显式visual-only推理已完成，但603eac54的新训练入口仅做过语法检查。针对已有F05输入条件缺口，登记一次覆盖全部原metadata零build LiDAR对象的真实反向实验；沿用已调研VGGT/SparseNeuS迁移依据，不重复网络调研或新增世界表示。不是按成绩挑选少量可拟合Actor，也不通过多次smoke替代认真训练。

数据准备`WS-V73-M2-EMPTY-INPUTS-01/20260908T085500Z__metadata-zero-lidar-training-inputs-r3`：新增`scripts/prepare_worldsim_v73_visual_only_cohort.py`，仅按原489对象index中build_points==0选择全部51对象（43 fit/9日志、8 development/2日志），原case文件直接链接；没有相机位姿者仍保留，不读取标签或预测质量用于筛选。该小index只服务明确的输入子集诊断，不替代全489主表。

登记`WS-V73-M2-GLOBAL-ACTOR-01/20260908T085500Z__visual-only-full-track-one-epoch-s7304-r13`：从原M1r3初始化DPT/查询、seed7304、joint/native_surface、native_data_weight1、full_track FIT标签、hard range free.5、event0、lr1e-5，一轮完整遍历41个有相机fit对象。初始/PCA/最终均在51对象上重新评价，不复用旧initial/baseline，不恢复R5/R10权重。34个有正目标与7个无正目标但有真实近框束的fit对象全部保留；无正目标的coverage记null，不能把未知区域当自由空间。检查记录真实DPT/查询梯度、coarse支持路径、跳步原因及峰值资源，含3个无相机开发对象的缺失分母。

这一次短训练仅回答新入口是否能实际学习和正确处理空目标；不据此选择主架构、声称充分拟合或跨日志质量成功，也不修改R10/R11/R12的cohort、目标或顺序。更大cohort的完整训练需作为独立变化与主候选结果衔接。登记时R13尚未启动；待R11实际退出后再直接启动，无等待轮询控制器/自动GPU队列。

08:50UTC外部20日志逐返回TSDF构建已启动，PID64413、code75ac0563，按原20日志顺序处理build输入，未运行模型质量评价。R10/PID53472 epoch13、R11/PID38460 epoch28仍训练；没有OOM，R12未启动。failure_ledger_delta=none（F02/F03/F04/F05持续，下一失败编号V73-F09）；整个V7.3未完成，shutdown=false。

---


## V7.3 旧AV2 TSDF场景比较完成，登记20新日志背景候选（2026-09-08）

`WS-V73-M4-AV2-SCENE-01/20260908T083000Z__old-development-vdb-r4` codee93c8b94完成，CPU13.407617s/RSS.704468GiB、0新推理/训练。固定21 Actor的已存R5/R9/native与PCA/仅背景，原187494束、5257 cohort、88 moving束，单日志不做跨日志bootstrap。

仅背景TSDF+雕刻相对PCA+雕刻全束hit48.520→39.175%、early14.150→4.299%、miss32.133→45.543%、free.513045→.107526m；hit−9.346pp、early−9.850pp、miss+13.409pp、free−.405518m。覆盖成本必须同时报告。同TSDF背景cohort：PCA hit62.907/early9.397/miss22.656/free.044827m，native42.534/33.137/18.889/.343803m，R9 54.194/9.017/18.528/.058941m，R5 53.472/30.455/10.101/.299177m。R5−R9 early+21.438pp/free+.240236m、miss−8.427pp/hit−.7228pp；不同FIT标签/适配通路的整体候选差异，非单纯空间交互因果结论。

同一R5只换背景，cohort hit不变、early−.1332pp、miss+.2092pp、free−.015140m；moving88束的free.855627m及返回指标不变。降低背景侵入没有消除Actor问题，F02不能归因于拼接背景；旧单日志不承担20新日志确认。报告`docs/WORLDSIM_V7_3_AV2_VDB_BACKGROUND.md`，原构建/场景/配对归档m4/av2_old_vdb_*，所有构建/评价进程均已退出。

登记`WS-V73-M4-AV2-SCENE-DATA-01/20260908T084000Z__external20-vdb-per-return-r2`，父数据为`20260908T031000Z__external20-per-return-build-background-r1`；全部20预先选定日志/936 Actor/80 build扫描，固定相同TSDF参数与逐返回积分、同一build雕刻。新20日志只准备背景候选，heldout文件直接链接，不读其质量评价，不依据新域质量调参/换日志。原PCA背景保留，该候选不是最终优越性判定。登记时尚未执行，拟由一个CPU进程依次构建所有日志，无新的GPU训练队列。

R10/R11继续原进程，R12未启动，visual-only新反向等待显存释放。F02/F04/F05保持active，下一失败编号V73-F09；整个V7.3未完成，shutdown=false，任何数据构建任务执行期间也不关机。

---


## V7.3 旧AV2逐返回TSDF背景构建完成，登记固定表面比较（2026-09-08）

`WS-V73-M4-AV2-SCENE-DATA-01/20260908T083000Z__old-development-vdb-per-return-r2` code7dadd661完成，CPU38.345745s/RSS1.715073GiB、0 GPU/更新。旧日志02678d04的4个build扫描共375000返回，实际积分364570背景返回，分别与父数据四扫描保留数91095/89913/90813/92749一致；父PCA去重后为364560中心，差10个重复返回不能混作选点变化。218766个真实原点组逐组积分，未采用统一扫描原点或近似时刻。

原生393002三角面，沿所有原build束单次雕刻11499片(2.926%)，保留381503；9523条build矛盾束→0、剩余build侵入距离和0。原heldout文件与21 Actor已知逐返回轨迹均沿用父数据。构建阶段没有模型评价，也不表示新时刻几何提高；原生和雕刻两份表面均保留。

登记同一旧日志固定表面场景比较`WS-V73-M4-AV2-SCENE-01/20260908T083000Z__old-development-vdb-r4`，使用已存joint R5、LiDAR R9、native fusion和共同PCA/仅背景，原187494束/5257 cohort束保持。仅改变背景为上述TSDF+build雕刻，零新神经推理/训练；比较须同时报告命中、early/free及缺失，不把一日志当独立确认或为其生成跨日志bootstrap。登记时评价尚未执行。

新增紧凑构建摘要脚本，归档每日志参数/返回数/原点组/三角规模，不复制大规模Actor轨迹数组到工作区或git。新20日志模型质量仍未读；是否继续准备其TSDF候选以旧开发结果及既定F04决策为依据，不按新域质量挑背景。R10/R11继续，R12未启动，visual-only真实反向仍待显存释放。F04/F05保持active，下一失败编号V73-F09；整个V7.3未完成，shutdown=false。

---


## V7.3 AV2逐返回TSDF背景迁移登记（2026-09-08）

当前AV2已存背景是PCA，尚未接入nuScenes开发侧的TSDF候选。针对VDBFusion单次integrate只接受一个sensor origin的接口卡点，本轮先查[官方仓库](https://github.com/PRBonn/vdbfusion)及[原生积分源码](https://raw.githubusercontent.com/PRBonn/vdbfusion/main/src/vdbfusion/vdbfusion/VDBVolume.cpp)。AV2已补偿到reference ego的端点只能转world一次；逐返回原点不能替换成扫描统一原点，否则会破坏已记录的米制射线。

新增`motion_proj/worldsim_v73/vdb_integration.py`按实际相同float64原点分组，排序后一次分段，原点不取整/平均、不改变时刻、不截断点数；保留组内输入顺序。避免逐个原点反复扫描整点集造成O(N×原点数)工作。使用现有vdbfusion0.1.6 wheel和CPU环境，不新增环境或改第三方原生算法。

登记`WS-V73-M4-AV2-SCENE-DATA-01/20260908T083000Z__old-development-vdb-per-return-r2`，新增`scripts/prepare_worldsim_v73_av2_vdb_background.py`；登记时未执行。先在旧开发日志02678d04完成实际背景构建：与原父数据相同4个build扫描，逐返回已知sensor pose、全部有效时刻已知框+.1m外端点，无near-sensor删点。uniform weight/voxel.1/trunc.3/space_carving=true/fill_holes=false/min_weight0；保留原生mesh与按所有原build束一次雕刻后的mesh。原父PCA端点做float32去重，而TSDF逐原返回加权，二者计数差异需披露。

只链接父数据原heldout束、owner、sensor-known标记与只读Actor轨迹，不为构建读取heldout值或评价模型质量。新20日志尚不运行质量评价；旧开发接口完成后才将同实现用于其背景准备，最终背景选择仍在质量确认之前完成。原生体积可由已有输入重建，本次保留两份显式表面与完整构建参数，不另存大体积文件。

08:18UTC R10/PID53472 epoch12、R11/PID38460 epoch26继续原训练；R12未启动，visual-only新反向仍待释放显存。本项仅CPU数据构建，执行期间同样不得shutdown。F04/F05保持active，下一失败编号V73-F09；整个V7.3未完成，shutdown=false。

---


## V7.3 实验可比条件整理与原生强控制报告更新（2026-09-08）

新增`docs/WORLDSIM_V7_3_COMPARISON_PROTOCOLS.md`，基于实际manifest、实现和已存结果整理主路径、CAPA/AdaPoinTr/融合、零LiDAR及场景/新域比较。没有新训练、推理或重复评价；这是一份后续技术报告的证据解释文档，不是新增门控或验收清单。

CAPA每窗口重置并用development的build做TTA、每张图重新估计affine尺度/偏移；共享joint/native控制的development不反传且build尺度固定，两者不能写成完全相同训练/校准条件。AdaPoinTr的全16384点与匹配PCA片、更新累积及优化器不同，输出密度/呈现数/optimizer步数也不能混写。R5恢复缺完整RNG、107个丢弃更新与实际11130有效更新保留；不为整理报告自动重跑。

R10−R11才是同full_track/目标下的当前原生强控制比较，但同时改变生成参数化和片形状，不自动证明attention收益；完整population同容量pointwise尚未运行。若采用R12的finite-beam目标，也需要匹配目标的原生控制。R12和visual-only新cohort仍分开安排，不合并归因。现有query读取mask为投影/裁剪有效性，不冒充密集遮挡真值；局部片之间未显式焊接，不声称闭合拓扑。

`WORLDSIM_V7_3_MATCHED_NATIVE_CONTROL.md`已改写为当前状态，清除同页仍被写作现状的“R10/R11未启动”等旧叙述，恢复历史在三账/git与R5报告保留。07:48UTC实际R10/PID53472 epoch10、R11/PID38460 epoch24，均正常反向，allocated峰值10.213784/2.346402GiB。二者最终结果均未完成；R12未启动，无后台启动队列，visual-only真实反向尚待资源释放。

单体Actor等权、场景按原束加权、observed点集与完整表面、旧AV2开发日志与20新域确认均分别注明；外部20日志输入就绪但质量仍未读，不保证预训练语料级未见，也不冒充nuScenes IID或同推理预算。F01–F08既有状态不因文档整理改变，下一失败编号V73-F09；整个V7.3未完成，shutdown=false。

---


## V7.3 Actor TSDF对照完成：低侵入伴随严重缺失（2026-09-08）

`WS-V73-M2-TSDF-FUSION-01/20260908T072000Z__population-build-vdb-r1` code6f753dab完成全部489 Actor/25日志，CPU9.081212s/RSS.649452GiB，0 GPU/训练更新。原生TSDF与同mesh单次build雕刻两份表面全部保存，融合、读出和配对分析进程均已退出。

fit414对象积分194680原build返回，144形成mesh，131713三角面删2677；development75积分18593返回，18形成mesh，18982面删398。原build矛盾束2624/352→雕刻后侵入距离和0；观察时段一致不等于新时刻正确。有LiDAR但无mesh的fit227/dev49仍保留，不能偷偷退回PCA。全489按原build计数：0点51→0非空、1–15点179→0、16–63点88→16、64–255点81→56、≥256点90→90，未按质量选对象。

完整75开发对象/5日志，TSDF+雕刻hit3.279%、early.258%、miss96.330%、free.000773m、recall21.508%；原TSDF为3.320%/.266%/96.283%/.000856m/21.512%。相对同输入PCA：hit−17.966pp，95%[−32.317,−5.663]；miss+22.919pp [7.264,41.174]；recall−43.910pp [−58.409,−29.412]；free−.016935m [−.035062,−.004568]。低侵入伴随大量缺失，不能称重建改善。表面距离只有18对象/4日志可定义，配对PCA增加.366072m [.280899,.487521]，不以未配对均值相减。移动9对象/2日志6无表面，miss91.237%、free.001030m，表面距离只1日志。

此结果限定于voxel.1/trunc.3、不补unknown洞的官方原生mesh配置，不外推所有TSDF或较密静态背景。保留为经典融合适用性诊断，不以超过它充当主方法胜出；继续强native/LoRA、LiDAR同解码器与AdaPoinTr对照，不围绕本配置开参数网格或添加主表示。报告`docs/WORLDSIM_V7_3_ACTOR_TSDF.md`；构建和完整配对归档m2/global/actor_tsdf_r1_*，全原表面/逐帧结果保留在run，图V73_ACTOR_TSDF.png/pdf直接读取已存结果。

R10/R11继续原进程，R12未启动，visual-only新训练入口尚待真实反向验证；外部20日志质量仍未读取。F02/F03/F04/F05保持active，下一失败编号V73-F09；整个V7.3未完成，shutdown=false。

---


## V7.3 同输入Actor TSDF非学习基线登记（2026-09-08）

主表尚缺当前完整489 cohort的TSDF融合强参照，不能用旧V7.2 legacy cohort的带上限射线/zero-crossing点集与anchors并集代替当前字面表面比较。本轮先查询[Curless–Levoy SIGGRAPH1996原文](https://lightfield.stanford.edu/papers/volrange/)及[VDBFusion官方积分与提取源码](https://github.com/PRBonn/vdbfusion)，利用已安装vdbfusion0.1.6的现有CPU环境迁移，不新增环境、主候选或GPU任务。

登记`WS-V73-M2-TSDF-FUSION-01/20260908T072000Z__population-build-vdb-r1`，新增`scripts/evaluate_worldsim_v73_actor_tsdf.py`，尚未执行。完整489 Actor/25日志、414 fit/75 development，所有零输入与无owned返回对象保留；只将原4个build扫描的明确Actor归属端点及当时传感器原点在已有规范坐标中积分，不读FIT full_track标签、图像或heldout质量构建表面。无输入点数截断、位姿重估或按效果选对象。

固定voxel=.1m/trunc=.3m，与既有背景基线尺度相同；官方均匀权重/space_carving=true，fill_holes=false/min_weight=0，8个cell角均需观测权重，不在unknown边界闭合表面。不把TSDF空输出退回PCA再冒充TSDF完成补全。保留uncarved原生mesh，并以所有原build近框束（包括非当前Actor归属）执行已有one-pass首回波前0.2m整三角面雕刻；后方仍unknown。两种表面构建完成后才进入heldout硬评价，没有目标时刻删面、opacity或凸包。

原生TSDF mesh和固定query patch密度不同，分别报告顶点、三角数、面积与缺失，不能称同输出预算。兼容字段surface_patches在该基线仅计原生三角元素，绝不能读成模型的8三角query片。保存全部两种表面及逐帧结果，通过现有CPU BVH统一首交点/表面距离、Actor→日志等权配对；CPU与旧GPU算子有浮点/共面细节边界。此项检验经典融合在当前稀疏输入上是否适用，不预定成功，也不把低侵入但高缺失算作胜出。

R10/R11继续原进程，当前epoch8/22；R12未启动，visual-only训练入口实现尚待真实反向验证。外部20日志质量确认仍未读取。F02/F04/F05保持active，下一失败编号V73-F09；整个V7.3未完成，shutdown=false。

---


