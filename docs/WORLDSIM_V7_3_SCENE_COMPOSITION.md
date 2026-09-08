# V7.3 场景级背景与Actor组合

## 最新背景对照：TSDF减少错误遮挡并降低覆盖（2026-09-08）

官方VDBFusion .1m/.3m、fill_holes=false与同一build雕刻已完成，使用原6开发场景/5日志/416704束。背景导致cohort提前返回由PCA+雕刻的25.191%降为5.839%、free .893276→.166678m；全束背景hit26.020→22.714%、miss53.279→62.802%，覆盖代价明确。详见[完整构建、场景对照与日志配对](WORLDSIM_V7_3_VDB_BACKGROUND.md)。TSDF+雕刻作为后续开发的主背景候选，原PCA+雕刻保留固定敏感性对照。

同TSDF背景下r5联合模型cohort hit26.905%、early29.099%、miss27.565%、free.316813m；r9为32.872%/13.768%/44.601%/.208408m。r5−r9 early+15.331pp、95%[+7.145,+25.152]，free+.108405m [+.039898,+.180905]，5日志都更差；F02仍存在。仅换背景带来的r5 hit改善不能归为Actor架构收益。全原束、轨迹、已训练表面均固定，未做新神经推理或外观重加权。

两项新场景run为`WS-V73-M4-SCENE-COMPOSITION-01/20260908T054500Z__development-vdb-background-r8`与`20260908T054500Z__development-vdb-build-carved-r9`，已完成，报告包含CAPA2/Ada2/r8/native fusion/PCA共同背景结果。新20日志仅完成输入/原PCA背景/冻结前缀构建，未读模型质量；该TSDF对照不是新域确认。R10/R11训练继续，整个V7.3未完成，shutdown=false。

---



## 主联合r5同背景组合结果（2026-09-08）

nuScenes r7 `20260908T044000Z__development-joint-r5-carved-r7` code26a7e509完成，3.7789s/RSS.78560GiB。同r3雕刻背景、75开发Actor/6场景/5日志、416704原束；cohort11886束，r5 hit21.4071%、early43.0467%、miss23.1791%、free1.007616m。

相对同背景r8：cohort early+13.021pp [95% +5.801,+23.188]，free+.098427m [+.036532,+.167488]，miss−16.054pp [−21.983,−9.849]，hit−1.052pp [−9.228,+9.222]。相对r9 early+13.306pp [+5.937,+23.406]、free+.095135m [+.033168,+.167766]。相对PCA early+13.247pp [+6.464,+23.108]、free+.103847m [+.040282,+.169700]；物理代价明确，returned MAE改善不能替代原始束指标。

旧AV2同背景逐束r3也完成，21对象/1旧日志/187494原束，cohort5257束r5 hit53.4716%、early30.5878%、miss9.8916%、free.314317m，r9为54.2134%/9.2068%/17.2912%/.074238m。该单旧日志没有bootstrap区间，不是外部确认。两种任务均只读取固定表面，未重新神经推理。源摘要m4/scene_composition_r7_summary.json/r7_paired.json与av2_old_scene_r3_summary.json/r3_paired.json。

## AdaPoinTr Y-up在相同雕刻背景的组合结果（2026-09-08）

`WS-V73-M4-SCENE-COMPOSITION-01/20260908T030000Z__development-adapointr-yup-carved-r6`已完成，code5dfa3948，CPU3.8597s/RSS0.77354GiB。与下方r5使用同一build-free雕刻背景、6场景/5日志和每方法416704条原始heldout束，只增加已完成AdaPoinTr Y-up r2的固定匹配表面，不重新运行网络或训练。cohort11886束、移动cohort6657束/2日志，所有空表面/其他物体/边界和传感器近区返回保留。

| Ada r2组合 | hit | early | miss | free m | 有返回MAE m |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全原始束 | 26.2558% | 10.9824% | 52.3892% | 0.275415 | 1.284493 |
| cohort返回 | 11.0410% | 33.6000% | 36.5152% | 0.951842 | 2.394034 |
| 移动cohort，2日志 | 6.0021% | 5.0856% | 74.9249% | 0.025530 | 2.571467 |

同背景cohort配对，Ada r2−r1 hit−0.682pp、日志bootstrap95%[−2.892,+1.941]pp，early−1.828pp、[−6.443,+2.788]pp，miss+2.615pp、[−1.592,+9.478]pp，free+0.001128m、[−0.016595,+0.014165]m；有返回MAE+0.237263m、[+0.040702,+0.473580]m。轴接口修复没有产生一致的完整场景改善。

Ada r2−r8 cohort hit−11.418pp、[−15.326,−8.430]pp，free+0.042653m、[+0.011555,+0.089395]m。相对PCA miss−8.009pp、[−14.314,−1.704]pp，hit−10.874pp、[−21.286,−0.461]pp，free+0.048074m、[+0.011879,+0.098823]m。MAE只在有返回时定义，必须和原始分母内的miss并列，不能单独当成功。

结果`m4/scene_composition_r6_summary.json`与`scene_composition_r6_paired.json`保存全部原始分层和日志差值。r2原生16384点与匹配表面的密度影响已经另做Actor级分析，见`WORLDSIM_V7_3_ADAPOINTR_BASELINE.md`；全密度结果没有混入本场景表。背景本身仍造成较大cohort提前遮挡，F04继续active。

AV2旧开发日志的逐束时刻完整场景接口也已完成，使用同一静态背景/规范Actor BVH和逐束只读姿态；结果与nuScenes不同，应单列，见`WORLDSIM_V7_3_AV2_GEOMETRY.md`。单条旧日志不提供有意义的独立日志置信区间，不能用于宣布新20日志外部确认。外部数据目前只做格式与固定build场景构建，尚无模型质量评价。


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



## V7.3 场景配对收尾与CAPA分块后真实训练（2026-09-08）

复用场景r3保存结果完成r8−r7配对：cohort free−0.04382m、日志bootstrap95%[−0.07655,−0.01284]m，4日志降低、1相同；early−5.98pp、[−9.44,−2.52]pp；hit−3.52pp、[−9.62,+1.79]pp；miss+7.39pp、[+0.22,+14.55]pp。场景侵入降低仍伴随覆盖代价，且背景自身误差占比大，不宣告F04解决。全部原始束、cohort、背景/其他对象、边界代理与近传感器分层保留在m4/scene_composition_r3_summary.json，配对在scene_composition_r3_paired.json。没有新增模型推理。

完整Actor开发队列的PCA/r6/r7/r8六项结果图已生成 `docs/autoresearch/worldsim_v73/m2/global/V73_POPULATION_FREE_TRADEOFFS.png/pdf`；柱为日志等权均值、点为5条独立日志，图中明确r6短窗标签与r7/r8全轨迹标签差别。图不使用旧25Actor诊断子集替代当前75个开发对象，运动样本仍仅2日志的限制保留在正文。

CAPA修订r2已实际启动：run `WS-V73-M2-CAPA-01/20260907T225000Z__population-build-tta-chunked-s7305-r2`，codeeb42f835，PID31466，日志 `/root/autodl-tmp/controller_logs/v73_population_capa_r2.log`。原始完整VGGT和393216个LoRA参数成功加载，首窗口已越过原OOM位置进入真实反向/优化，官方step0/10/20/30/40的L1读数为3.8186/1.8611/2.0940/1.4381/1.2993。随机视图子集不同，不能把这五个数当成同样本学习曲线或开发效果。随后首个scene-0015窗口已实际完成全部100步、全24视图联合推理、LoRA/深度保存及其Actor评价；适配与保存267.033s，累计GPU allocated峰值8.08180GiB，现进入scene-0071。首窗口已跨过完整优化/推理路径，其余30窗口仍待完成，不称整个基线完成或所有规模资源问题已解决。

最近GPU进程占用r5 12358MiB、Ada r1 1582MiB、CAPA r2 9880MiB，不能再叠加新GPU作业。当前GPU余量限制属于并发调度；正常训练继续，先等现有作业释放资源，再启动已准备的event单因素对照与Ada轴修订r2，不为关机强行结束正常任务。Ada r1最近epoch17/1567次优化更新，原完整初始化输出已保留；其轴未迁移限制仍按F07报告。

下一event run登记为 `WS-V73-M2-GLOBAL-ACTOR-01/20260907T230000Z__population-lidar-track-beam-event-s7304-r9`：相对r8仅event-weight=0.01，固定σ0.2m/C28/width0.03m/res32，保持371fit/完整489队列、全轨迹fit标签、原build输入、seed7304、30epoch/11130更新与free配置。复用相同r6 initial和r5 PCA；无支持不从event分母删除。此时未启动，无后台自动启动队列，待有实测资源后执行。后续主joint同full_track适配、强视觉/补全控制、独立新日志及完整应用仍需推进。

failure_ledger_delta=update F01/F02/F03/F04/F05/F07状态；F01/F02/F03/F04/F05/F07继续active，F06直接数据配置缓解，下一编号V73-F08。三项长训练/适配均正常，整个V7.3未完成，shutdown=false；15分钟自动研究继续，真正完成后保存/push并确保无任务/队列再关机。

---


## V7.3 场景配对结果与运行状态（2026-09-08）

复用既有结果完成独立日志配对（`scripts/summarize_worldsim_v73_scene_composition.py`；10000次日志bootstrap、seed7306，不重跑模型）：背景近点修订使5/5日志的全束free降低，平均差−0.20861m、95%区间[−0.39920,−0.07872]m；early差−1.74pp、区间[−4.03,−0.34]pp。miss在5/5日志增加，平均+1.69pp、区间[+0.29,+4.02]pp。真实场景误差降低和覆盖代价须同时报告，仍为旧5日志开发证据。

修订背景上，r6对PCA的cohort hit差+7.09pp，[+4.05,+10.18]pp，5/5日志改善；cohort free却增加0.04469m，[+0.01348,+0.08405]m。native fusion对PCA的cohort hit差−1.84pp，[−6.90,+3.22]pp；free增加0.02376m，[+0.01274,+0.03450]m。未形成完整物理优势。全部组、逐日志差值和分母保存在 `docs/autoresearch/worldsim_v73/m4/scene_composition_r2_paired.json` 与原summary；不同fit标签预算不混称同监督方法胜负。

r8已实际启动：code d50c9eeb，PID26975，日志 `/root/autodl-tmp/controller_logs/v73_population_lidar_beam_range_r8.log`，真实free_mode=beam_tube_range、full_track标签，已到epoch2、GPU allocated峰值0.33034GiB，非零query梯度。joint r5 PID18843至epoch11，GPU峰值10.196GiB；r7 PID23188在final_evaluation，已写423/489个最终surface文件，仍有正常进展，不重复启动或中断。场景数据/评价任务均已结束。

failure_ledger_delta=update V73-F02/F04证据与运行状态；F01/F02/F03/F04/F05仍active，F06直接数据配置缓解，下一编号V73-F07。r7最终summary未生成前不下结果结论。下一步完成r7→r6完整/共有输入/运动分层对比，继续主模型、强基线、event与新日志。整个V7.3未完成，shutdown=false。


---

## V7.3 背景近传感器对照完成（2026-09-08）

code7fb7e5a7完成r2背景与场景比较：数据 `WS-V73-M4-SCENE-DATA-01/20260907T211500Z__development-background-sensor-close-r2`，41.72s、RSS7.650GiB；比较 `WS-V73-M4-SCENE-COMPOSITION-01/20260907T211500Z__development-pca-lidar-native-r2`，3.953s、RSS0.751GiB、纯CPU。数据PID26177与评价PID26331均结束。

仅在每个build扫描按官方nuScenes采样LiDAR坐标abs(x)<1且abs(y)<1排除静态背景候选，共194536条原始近点；背景精确坐标去重支持723130→606366。416704条留出原始束全部保留，Actor预测、轨迹及读出尺度不变。新增分层显示96339条评价返回落在近传感器区、320365条在其外；这是几何范围定义，不是精确自车语义标签，也不把传感器壳体承诺为静态世界。

仅背景的开发5日志均值（统计单位scene→log，与Actor表不同）：全束free0.52771→0.31911m，early13.96%→12.22%，同时hit26.31%→25.90%、miss50.59%→52.27%。近点静态累积解释了部分严重侵入，但不能把支持删减伴随的覆盖下降隐藏。框外背景代理free0.53447→0.32445m；cohort返回被背景提前阻挡的日志平均仍27.08%、free1.03555m；近传感器区外全束仍有15.68%early、45.22%miss、free0.40830m。因此F04仍active，背景不能视为已经正确的固定真值。

使用修订后的同一背景，完整组合结果：

| Actor方法 | 全束hit | 全束early | 全束miss | 全束free m | cohort hit | cohort early | cohort miss | cohort free m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LiDAR PCA | 26.60% | 12.33% | 51.52% | 0.32154 | 21.45% | 31.68% | 43.14% | 1.04603 |
| LiDAR-only r6 | 26.72% | 12.50% | 51.18% | 0.32652 | 28.54% | 38.45% | 27.59% | 1.09073 |
| native fusion r2 | 26.43% | 12.45% | 51.28% | 0.32993 | 19.61% | 35.17% | 38.05% | 1.06979 |

所有组合的相对Actor趋势仍未形成物理优势；全束差值很小不能替代cohort分项。固定背景会显著影响整体数值，后续主模型须使用相同已登记背景，并继续研究build支持与已观测free的一致性；不能用开发GT删背景支持、把错表面变透明或删掉困难评价束。Voxblox/TSDF与基于free的静态一致性已有官方参考，尚未声称实现这些新背景方法。

原始结果=`docs/autoresearch/worldsim_v73/m4/scene_composition_r2_summary.json`，数据记录=`scene_data_r2_index.json`；对比研究图与复现脚本为 `V73_SCENE_BACKGROUND_COMPOSITION.png/.pdf`、`scripts/plot_worldsim_v73_scene_composition.py`。图为已有5日志的点估计，不是新日志确认或显著性结论。

主joint r5 PID18843至epoch10，全轨迹LiDAR-only r7 PID23188至epoch28，正常训练，无资源不足。下一步待r7最终surface/summary完成后，用同一完整cohort配对r6并分列原build LiDAR-ready与运动子集；不将中途checkpoint当成最终结果。CAPA、AdaPoinTr、主模型完整结果、首事件与新日志确认继续待推进。failure_ledger_delta=update V73-F04；F01/F02/F03/F04/F05仍active，F06直接数据配置缓解，下一编号V73-F07。整个V7.3未完成，shutdown=false。


---

## V7.3 真实场景组合完成与背景近点对策（2026-09-08）

code3f77f087完成原6开发scene/5日志场景数据与全部4种组合读出。数据run `WS-V73-M4-SCENE-DATA-01/20260907T211000Z__development-build-background-r1`：42.57s、RSS7.652GiB，723130个build背景支持/5785040三角面，12个留出扫描416704条原始束。比较run `WS-V73-M4-SCENE-COMPOSITION-01/20260907T211000Z__development-pca-lidar-native-r1`：4.409s、RSS0.785GiB、纯CPU、48个方法×扫描记录；没有训练/数据作业残留。每个方法都用同一背景，未将未重建对象的束删除。

以下是scene内束加权→scene/log等权的开发均值，与旧Actor均值统计单位不同：

| 组合 | 全部束hit | 全部束early | 全部束miss | 全部束free m | cohort返回hit | cohort返回early | cohort返回miss | cohort返回free m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 仅背景诊断 | 26.31% | 13.96% | 50.59% | 0.52771 | 0.73% | 28.04% | 67.25% | 1.25900 |
| LiDAR PCA＋背景 | 27.00% | 14.07% | 49.84% | 0.53015 | 21.36% | 32.63% | 42.82% | 1.26948 |
| LiDAR-only r6＋背景 | 27.13% | 14.23% | 49.51% | 0.53509 | 28.19% | 39.07% | 27.58% | 1.31358 |
| native fusion r2＋背景 | 26.84% | 14.19% | 49.61% | 0.53853 | 19.53% | 36.11% | 37.75% | 1.29321 |

cohort返回仅11886束，框外背景代理397071束，其他注释对象7452束、重叠歧义295束；注释框边界带15687束是与以上组重叠的代理。场景总体被背景主导。cohort返回数在scene0359/0919/1089只有25/169/74，日志等权均值因此不能解释成全体射线比例。PCA→r6的背景代理返回被Actor提前抢占246→629束，native fusion为689束；这些是原始池化计数，不是独立样本数。当前方法未达到物理优势，固定背景也不够可靠，不能把所有组合错误归因于Actor。

F04新增定位：先查[nuScenes官方remove_close](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py)、[Voxblox官方](https://github.com/ethz-asl/voxblox)与[Open3D TSDF官方](https://open3d.org/docs/release/tutorial/t_reconstruction_system/integration.html)，再追踪背景BVH首交点的支持。27.51s CPU诊断显示背景56902个early中10974个命中“距任一build传感器xy均小于1m”的支持，该重叠组占侵入距离总和45.47%；掠射abs(cos)<0.1组只有1138个early。原始build扫描的传感器xy近点合计194536/833216，约23.35%。近点几何条件不是精确自车语义标签，任一build传感器邻域也不是精确采样来源，当前归因仍为定位线索。

因此下一对照只修改静态背景构建：沿用官方devkit的采样LiDAR坐标abs(x)<1且abs(y)<1近点规则，在各自build扫描去除这部分背景候选。评价原始束一条不删，另列近传感器/其外区域，不能靠删评价错误制造改善。Actor训练输入、已训练表面、轨迹、曲面尺度和模型参数均不变。登记r2数据 `WS-V73-M4-SCENE-DATA-01/20260907T211500Z__development-background-sensor-close-r2` 与比较 `WS-V73-M4-SCENE-COMPOSITION-01/20260907T211500Z__development-pca-lidar-native-r2`；待完成才能判断此对策解释多少错误，未知背景与归属问题仍独立存在。

结果证据=`docs/autoresearch/worldsim_v73/m4/scene_*r1*.json`、`background_attribution_r1.json`；更完整边界见 `WORLDSIM_V7_3_SCENE_COMPOSITION.md`。汇总工具同时加入通用已完成run配对引用与依据原build输入状态的共有LiDAR-ready分层，不重跑相同模型，可用于r7对r6与fusion的后续比较。

joint r5 PID18843、全轨迹LiDAR-only r7 PID23188继续正常训练；r7已至epoch22，未中断正常任务。failure_ledger_delta=update V73-F04（完成真实全局读出，发现背景近点问题，修复效果待测）；F01/F02/F03/F04/F05仍active，F06直接数据配置缓解，下一编号V73-F07。CAPA、AdaPoinTr、主模型完成与新日志确认仍待推进；整个V7.3未完成，shutdown=false。


---

以下保留首次实现和实验登记，最新实测结果见上方。研究风险F04需要在同一背景中比较全局首交点，Actor单独读出不足以判断遮挡和边界。

采用[Street Gaussians（ECCV 2024）作者实现](https://github.com/zju3dv/street_gaussians/)的背景与规范Actor分解方式，保留已知刚体轨迹；不迁移外观透明度作为物理读出。按[Open3D0.19官方RaycastingScene](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html)，为各规范三角面建立CPU加速结构，将原始束逆变换到该Actor坐标，求交后与静态背景统一取最近的正距离及owner。刚体旋转不改变单位方向长度，参数t仍是米制距离。固定曲面不需要梯度，复用现有worldsim-v72-lidar4d环境（Torch2.1.0+cu121、Open3D0.19.0、SciPy1.13.1），仅补装129kB的ijson3.5.1；当前训练环境不变。

背景只来自原4个build扫描，在各自扫描时刻剔除所有已知注释框+0.1m内的点，包括非刚体和未参加重建的已知物体；不是仅在最后位置裁剪。剩余点世界坐标累积，固定0.06m间距、9顶点/8三角面的PCA局部片，使用全部build背景点，不做凸包、自动补洞、opacity或闭合假设。未被采样的背景仍可缺失。注释不完整和框归属错误仍会产生残留/误剔除；不能声称语义真实背景已经完备。

评价读取同一短窗口内未输入的原始首回波束，保留所有有效距离，包括非刚体和未重建对象。各方法共享同一背景、轨迹、原始束和硬三角读出。单列all raw returns、cohort自有返回、框外背景代理返回、其他注释对象、重叠歧义、注释框表面0.2m边界带。此边界带由真实回波端点与已知框定义，是诊断代理，不能称为真实Actor接触边界。报告literal hit/early/late/miss、直接free侵入、带returned计数的条件MAE、背景返回被Actor抢先、Actor返回被背景抢先、归属错配和近深度重叠。

聚合先在scene内按真实束计数，再scene等权到log、独立log等权；与此前Actor内加权→log均值的统计单位不同，不能直接横比数值。全场景平均可能被背景主导，cohort与背景分项必须并列。原始GT只有首返回，后方表面不被额外标为错误或自由空间；没有反事实GT的轨迹编辑仅作组合能力验证。

入口：`prepare_worldsim_v73_scene_geometry.py`（build背景/评价束）；`evaluate_worldsim_v73_scene_composition.py`（已完成各方法surface文件）；`scene_readout.py`（BVH/只读刚体组合）。缺artifact直接报未完成，合法空表面保留为miss，避免将工程缺文件伪装为模型缺失。

首次数据登记：`WS-V73-M4-SCENE-DATA-01/20260907T211000Z__development-build-background-r1`，原有6开发scene/5日志；不使用新source作为开发，也不新增优化。首次比较登记：`WS-V73-M4-SCENE-COMPOSITION-01/20260907T211000Z__development-pca-lidar-native-r1`，background-only诊断、LiDAR PCA、已完成完整LiDAR-only r6、native+LiDAR fusion r2。r6比融合有更多短窗fit标签，监督预算差异保留；尚未完成的joint r5不读取中途结果冒充最终预测。

只进行一次解析遮挡/刚体读出核验，预期两Actor先后位于5/7m、背景10m；平移前后的全局owner变化有解析真值。它检验逆变换和遮挡读出实现，不等于真实场景验证。随后直接运行真实开发队列，不新增连续门控或回归套件。

F04仍active：全局排序解决读出语义，不自动修复背景ghost、注释误差、曲面片撕裂或未知背景；等实际分项结果后再选择迁移对策。F03首事件、强基线与新日志确认也未完成，shutdown=false。
