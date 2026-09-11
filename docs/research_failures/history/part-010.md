# 历史原始记录 010

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V7.3 全轨迹训练结束与米制射线管free比较（2026-09-08）

LiDAR-only全轨迹标签r7 PID23188已完成30epoch/11130更新，正在完整489Actor最终曲面评价；训练结束记录elapsed3105.48s，最终wall time和开发结果待summary完成。不把最后一次loss或中间checkpoint当成最终性能。主joint r5 PID18843继续正常训练。

现在进行已实现的米制射线管free目标比较：`WS-V73-M2-GLOBAL-ACTOR-01/20260907T212500Z__population-lidar-track-beam-range-s7304-r8`。相对r7，仅把free-mode由range改为beam_tube_range，仍为米制深度侵入、权重0.5；固定宽度0.03m、32×32栅格，通过同一显式三角面的最近深度及轮廓反传。输入、全轨迹fit标签、371fit对象、67dev可输入对象、51空输入对象、seed7304、AdamW lr1e-5、30epoch/11130更新及query架构均保持一致。复用完全相同的r6 initial与r5固定PCA评价，不重做相同模型读出。

本比较检验“原硬相交free缺少轮廓位移梯度时，保留米制严重程度的有限宽度几何目标能否改善真实侵入”；不混入event、额外opacity、自由半径或结构改动。原25Actor的beam_tube仅覆盖比例目标曾降低early数量却恶化侵入距离；此次range版本已在解析场景验证位置和轮廓梯度，但尚没有真实训练收益证据。它仍是优化代理：有限支持外可无梯度，仍需coverage吸引；宽度不是已校准的真实激光光束，不能将解析梯度结果写成全局收敛保证。官方nvdiffrast来源及已有解析证据见 `WORLDSIM_V7_3_FREE_VISIBILITY_DESIGN.md`。

r8使用现有motionproj训练环境与已编译nvdiffrast0.3.3，CUDA12.1编译器来自保留的v72-pointr环境；不升级Torch、不重建大环境。LiDAR-only作业不加载DPT，启动后以真实峰值记录资源；与r5和r7末尾评价并发的wall time不作单作业速度比较。r7评价结果一旦完成即用新增--reference配对r6，并列full cohort、原build LiDAR-ready、运动日志分层。

背景r2已完成，数据/评价进程均退出，scene结果和图已push3cb18bba。failure_ledger_delta=update V73-F02的单因素比较登记；F01/F02/F03/F04/F05仍active，F06直接数据配置缓解，下一编号V73-F07。CAPA、AdaPoinTr、主joint最终结果、event与新日志确认未完成，shutdown=false。


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

## V7.3 场景组合实现与实验登记（2026-09-08）

完整原生融合r2结果已经归档并push（ae295229）；joint r5 PID18843至epoch9、全轨迹LiDAR-only r7 PID23188至epoch19，正常运行。接下来并行做CPU场景物理评价，当前训练配置不变。

F04全局读出实现完成：复用Open3D0.19 CPU BVH，以每Actor的只读刚体逆变换读取原始束，和同一build背景全局取最近交点及owner。背景在每次build扫描排除所有已知注释框+0.1m内测量，采用固定0.06m PCA曲面片；不补未知背景、不引入opacity。现有v72-lidar4d环境仅补装ijson3.5.1，未新建大环境或改变训练Torch。

一次解析真值检查已经执行：run `WS-V73-M4-SCENE-READOUT-01/20260907T210500Z__analytic-rigid-occlusion-r1`，原先首距离[5,10,miss]、owner[1,0,-1]；平移前Actor后变为[7,5,miss]、owner[2,1,-1]，数值误差小于1微米。它支持刚体转换/全局遮挡实现，不作为真实几何或反事实真实性证据。未新增训练smoke/回归套件。

登记开发场景数据 `WS-V73-M4-SCENE-DATA-01/20260907T211000Z__development-build-background-r1` 与比较 `WS-V73-M4-SCENE-COMPOSITION-01/20260907T211000Z__development-pca-lidar-native-r1`。对象为原6开发scene/5日志，固定背景上比较PCA、已完成r6 LiDAR-only与fusion r2；background-only为诊断项。全部原始回波保留，cohort/背景代理/其他对象/歧义/注释框边界带分列，按scene→独立log聚合，不能与Actor-only均值混比。真实数据作业提交后启动，结果待实际完成。

方案、官方依据、统计域与未知区域边界见 `docs/WORLDSIM_V7_3_SCENE_COMPOSITION.md`。failure_ledger_delta=update V73-F04实现进展；F01/F02/F03/F04/F05仍active，F06直接数据配置缓解，下一编号V73-F07。尚未完成CAPA模型、AdaPoinTr、新日志确认及完整主方法对比，shutdown=false。


---

## V7.3 完整原生融合参考完成（2026-09-08）

`WS-V73-M2-GLOBAL-FUSION-01/20260907T203500Z__population-native-lidar-fusion-r2` 已完成，code d78e99ae，489 Actor、671.62s、GPU allocated峰值2.687GiB、RSS6.364GiB。每个24视图窗口固定M1r3 DPT解码一次，再做Actor规范融合和同尺度PCA三角片；没有新增优化或全轨迹标签，不能与更多标签训练的模型声称同预算架构胜负。

开发75对象/5日志的LiDAR PCA→native+LiDAR：hit21.24%→22.78%，early4.35%→9.16%，miss73.41%→57.35%，free0.01771m→0.15030m，观测target到surface距离0.30479m→0.17542m，0.2m覆盖65.42%→78.13%。相对PCA，hit差+1.53pp，日志bootstrap95%[-4.10,+6.35]pp，3/5日志改善；free差+0.13260m，[+0.05887,+0.20172]m，5/5日志变差。距离、覆盖、miss在5/5日志改善。覆盖与物理侵入的冲突仍在，原生融合不是已成立的物理胜者。

dev无预测对象8→3，fit43→10，说明原生视觉支持可覆盖部分无build LiDAR对象；这不能直接解释为其表面正确。dev仍23对象没有留出自有回波，保留未知GT语义。当前主模型的visual-only路径尚未接通，51个空输入对象仍明确计入；后续比较须同时列完整队列及共有可输入对象，不把输入可用性差异偷偷归于交互结构。运动dev仅9对象/2日志：fusion hit9.02%、early9.92%、miss67.06%、free0.20818m；尚不足以确认动态泛化。

结果及配对/运动分层已归档 `docs/autoresearch/worldsim_v73/m2/global/population_fusion_r2_summary.json` 与 `population_fusion_r2_analysis.json`。当前joint r5 PID18843正常训练至epoch8；全轨迹标签LiDAR-only r7 PID23188至epoch14，真实fit_label_times=full_track、GPU allocated峰值0.1945GiB。融合PID23189已结束。没有改动在跑配置，没有发生资源不足。

下一独立工作为F04的场景组合：先查阅[Open3D官方RaycastingScene](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html)与[Street Gaussians作者代码](https://github.com/zju3dv/street_gaussians/)，采用静态背景与只读轨迹Actor各自BVH求交，再统一取最近深度及owner。现有v72-lidar4d环境已安装Open3D0.19，可复用CPU读出，不为场景暴力全对全占用训练显存。静态背景只用build测量，并在每次采样排除已知动态归属；边界、背景未知和非刚体缺失必须分列。该方案尚未产生场景结果，不把Actor-only评价称为场景验证。

failure_ledger_delta=update V73-F02/F04/F05；F01/F03继续active，F06直接数据配置缓解，下一编号V73-F07。CAPA模型、AdaPoinTr、全局组合、新日志确认尚未完成。整个V7.3未完成，shutdown=false；继续研究。


---

## V7.3 完整LiDAR控制与全轨迹监督结果（2026-09-08）

完整cohort LiDAR-only r6完成（code98d9fa32，run `20260907T193500Z__population-lidar-only-extra-time-s7304-r6`）：371fit训练Actor、67dev可输入Actor，30epoch/11130更新，2904.27s含自身initial/final评价，峰值0.181GiB、RSS2.964GiB。固定PCA基线从r5复用，其计算成本不包括在r6 wall time中；该run与joint r5并发，不用于单作业速度排名。489对象全部记录，43fit/8dev无输入对象保留空预测；dev75对象中23个无留出自有回波，不把它们当作有几何GT，miss/覆盖与无输出计数同时报告。

dev5日志均值：LiDAR PCA→训练LiDAR-only为 hit21.24%→31.01%，early4.35%→14.02%，miss73.41%→49.31%，free0.01771m→0.08375m，观测target到surface距离0.30479m→0.24118m，0.2m覆盖65.42%→72.42%。hit/覆盖/距离在5/5日志改善，但early/free没有日志改善（4差、1相同）。相对PCA，hit配对差+9.76pp、日志bootstrap95%区间[+4.79,+14.73]pp；free差+0.06605m、区间[+0.03014,+0.09334]m。纯LiDAR生成器同样有覆盖与侵入冲突，因此不能把F02全部归因为高维视觉。当前joint r5还在训练，不能用其25Actor旧结果与完整r6直接比较。

新增按已知build窗口平均速度>2m/s的运动子集汇总（只分析已有结果，不重跑模型）：dev9Actor仅2日志，其中3个无留出自有回波、1个无预测。r6运动子集hit18.70%、miss71.85%、free0.04875m、距离0.11049m；该独立样本量不足以形成动态泛化主张。总体5日志也仍为旧开发集，不是新日志确认。完整结果与配对/运动分层=`docs/autoresearch/worldsim_v73/m2/global/population_lidar_r6_*.json`。

fit全轨迹标签构建也完成（code1c19f2c2，run `20260907T200000Z__fit-track-measurements-r2`）：175.86s、RSS8.553GiB、磁盘356MiB。414对象中407有正观测，合计1690284个Actor内精确坐标去重点（非独立表面样本），7748633条相关原始束，11214个Actor×扫描、9782个标签专用时刻；原build点合计194680。43个无build LiDAR对象中36个在更长时段有标签，仍不将这些后来测量作为输入。更充分观测不是完整表面GT，未知区域语义不变。索引=`docs/autoresearch/worldsim_v73/coverage/fit_track_targets_r2_index.json`，目标Tensor留在单独run目录。

后续r7登记：`WS-V73-M2-GLOBAL-ACTOR-01/20260907T203500Z__population-lidar-only-track-labels-s7304-r7`。相对r6仅将fit surface/free标签替换为独立全轨迹目标；build输入、seed7304、同decoder、lr1e-5、range free0.5、30epoch/11130更新保持相同。新参数--fit-targets仅在fit损失加载target_points_actor_m/target_rays，predict仍只读原case.points_actor_m；native数据辅助项（未来joint使用）也只读build像素。开发集不加载此目录。有效label_times记full_track，fit原留出短窗口已属训练标签；无输入、未参加优化对象不再标记training_labels。复用r6相同输入/同seed模型的initial评价和r5固定PCA结果，避免相同算子重复运行；复用路径写入manifest。

同时登记完整cohort原生保守参考：`WS-V73-M2-GLOBAL-FUSION-01/20260907T203500Z__population-native-lidar-fusion-r2`。使用已认真训练的M1r3 DPT，固定其参数，每个24视图窗口解码一次后供所有Actor共享米制深度；这是评价期固定输出缓存，不是训练中缓存可训练路径。按原规范轨迹融合native+LiDAR、同0.06m PCA曲面片；没有Actor build LiDAR但有native支持时允许native-only，完全无支持则明确缺失。所有489对象均纳入，不能再沿用旧25Actor融合结果代表全队列。其fit标签预算仍为M1 build深度监督、没有全轨迹标签，差异单列，不宣称同标签架构胜负；原生强控制后续仍可用相同几何标签优化。

r5正常运行；r6和标签生成已结束。r7及固定融合评价提交后启动，预计额外GPU开销小于另一套24视图反向，仍以实际占用为准，不把并发争抢当成单作业资源不足。CAPA数据已备、模型未运行；AdaPoinTr接同真实标签与统一表面、场景组合、独立新日志仍待推进。failure_ledger_delta=update V73-F02/F05；F01/F03/F04仍active，F06直接数据配置缓解，下一编号V73-F07。整个V7.3未完成，shutdown=false。

---

## V7.3 fit全轨迹标签可用性与构建（2026-09-08）

fit全轨迹载荷盘点完成：run `WS-V73-M2-EXTENDED-TARGETS-01/20260907T195000Z__fit-track-payload-inventory-r1`，code856aa525，38.90s。25fit场景合计1004个关键帧LiDAR均在本地；414个窗口内已知Actor中，403个在当前短窗口之外还有可插值轨迹与可用LiDAR，合计9064个额外Actor×关键帧机会。该数量不是新增正点数，尚须实际提取归属观测；11个无额外机会对象保留。没有新增日志或dev标签，输入Actor选择仍依据原build/metadata。记录=`docs/autoresearch/worldsim_v73/coverage/fit_track_payload_inventory_r1.json`。

真实fit标签生成已启动（code1c19f2c2，PID21982，日志 `/root/autodl-tmp/controller_logs/v73_fit_track_targets_r2.log`；已生成首批25对象，进程RSS约8.1GiB，最终数量待完成汇总）：`WS-V73-M2-EXTENDED-TARGETS-01/20260907T200000Z__fit-track-measurements-r2`，入口 `scripts/prepare_worldsim_v73_track_targets.py`。使用每次LiDAR扫描的只读Actor姿态，将该轨迹内全部可用关键帧正点累积到规范坐标，同时保留原始束首回波/歧义归属及已观测free。仍为box+0.1m、重叠排除的归属代理和scan级时间近似，不声称成为完整表面GT或精确实例分割。非build记录明确为fit_label_time。

标签保存为单独目录的target_points_actor_m与target_rays，原始actor-data/build输入/图像/metric scale/query seed一律不替换，当前r5/r6保持短窗训练配置直至完成。414个fit对象包括43个当前无build LiDAR者，标签文件保留其输入缺失身份，不能将后来测量偷偷拿来当输入。有更多观测支持仍不等于未知区域已知：后续训练不得将所有预测到稀疏标签距离都解释为几何错误。主模型与AdaPoinTr等控制需要同一新标签预算后才可比较，不能把较多监督单独包装成架构收益。

为避免每Actor重复读取同一扫描并计算全场景box归属，数据构建按场景共享world points、传感器原点与membership，再按当前Actor姿态投影；处理完一个场景释放缓存。默认短窗读取语义保持不变，include_track仅用于新的fit标签入口，不新增校验门控、hash或重复回归。

当前r5 PID18843和r6 PID20389正常训练；本标签生成用CPU并保留已有raw载荷，不需要下载或新环境。CAPA数据桥已完成，但CAPA模型优化未启动。failure_ledger_delta=update V73-F05（从短窗口到更充分真实训练标签的准备，不是已测性能提升）；F01/F02/F03/F04/F05仍active，F06在直接数据配置缓解，下一编号V73-F07。资源充足，shutdown=false，继续研究。

---

## V7.3 CAPA实际输入条件完成（2026-09-08）

CAPA build条件桥已在全部31个现有窗口实际执行完成：run `WS-V73-M2-CAPA-DATA-01/20260907T194500Z__surround-build-conditions-r1`，codebbd438fa，CPU构建3.537s。744视图共有2516975个有效稀疏轴向深度像素（fit1993519/dev523456），无零条件视图；该数量按视图计数，同一物理点可投影到多个视图，不是独立3D点数。只保留每像素最近的正测量，未读取额外时刻标签，未重复保存RGB或稠密条件张量。记录=`docs/autoresearch/worldsim_v73/coverage/capa_build_conditions_r1.json`。

这只确认实际数据桥已执行；CAPA权重载入、LoRA优化及物理评价尚未运行，不能记成完成基线。后续运行入口与协议差异见 `docs/WORLDSIM_V7_3_CAPA_BASELINE.md`。完整队列r5 PID18843已到epoch3，LiDAR-only r6 PID20389已到epoch5，二者保持原配置，峰值分别10.196/0.181GiB，磁盘约130GiB可用，无OOM。尚未完成的正常长训练继续运行，不因阶段结束关机。

下一独立数据工作：只在fit日志检查同一已知Actor整个可用轨迹内是否还有额外LiDAR记录，判断能否为主模型与AdaPoinTr提供更充分的真实训练目标。目前r4/r5的额外目标只含短窗口内2个未输入时刻；官方PCN使用完整表面目标，不能把稀疏目标强行解释为完整GT。此检查依据已有build Actor身份、只读轨迹和载荷可用性，不依据模型分数；dev不扩展训练标签，当前两条训练不改输入或目标。发现资源/数据缺口时再先检索官方优秀方案并迁移，不从短窗监督负结果外推整个路线。

failure_ledger_delta=update baseline data preparation + V73-F05；F01/F02/F03/F04/F05继续active，F06直接数据配置缓解，下一编号V73-F07。整个V7.3未完成，shutdown=false。

---

## V7.3 LiDAR控制运行与CAPA桥接准备（2026-09-08）

LiDAR-only完整队列r6已启动（code98d9fa32，PID20389，run `20260907T193500Z__population-lidar-only-extra-time-s7304-r6`），通过初始表面评价后进入真实训练。观测query梯度非零，native组为0，峰值0.181GiB；没有视觉前缀或DPT参与。r5 PID18843同时正常运行，GPU进程占用约12.1GiB+r6约0.53GiB，当前无OOM或资源不足。两条run的输入/标签协议保持登记配置，等待完整结果再比较。

CAPA基线桥接代码已准备，模型优化尚未启动/验证。入口=`scripts/evaluate_worldsim_v73_capa.py`，数据桥=`motion_proj/worldsim_v73/capa_inputs.py`。直接调用本地官方CAPAProtocol和VGGT LoRA配置，不复制v72含checksum/固定旧split的包装器；依赖peft0.19.1、omegaconf2.3.1、colorlog6.10.1、huggingface_hub0.36.2及原始VGGT本地权重均已在motionproj环境，不需要新环境、下载或升级Torch。

具体迁移边界：保留原378×672图像和K，build投影深度量化到像素时取最近正测量；不读取额外fit/heldout时刻。采用官方100步/rank4/alpha8/qkv设置；官方实际trainable选择是patch_embed内LoRA，其他注入LoRA冻结，这与主候选可训练DPT不同。官方每步随机取10%帧，本窗口为3帧；最终全24帧联合推理。为了保留反向通路并控制激活，启用上游aggregator已有的nonreentrant checkpointing，不缓存可训练路径最终特征。官方逐图像scale+shift从build测量拟合，作为CAPA协议差异明确记录，不能冒充主方法的窗口共享固定scale。

CAPA每窗口重置后适配，包括dev窗口自身的build输入；其额外时刻始终只评价，不能将这种TTA称为“dev完全不反传的共享模型”。这属于部署时允许的稀疏输入适配，而不是读取dev留出标签调参。输出米制深度按相机曝光时刻与已知Actor轨迹规范化，接同固定大小PCA三角片融合与原始首回波评价；有图像native支持但无Actor build LiDAR时允许native-only表面，完全无支持明确miss。输出密度与观测可用比例单独记录，不把点到面转换结果当成CAPA论文原任务结果。

先执行31个窗口的CPU条件构建，保存计数与时间/视图来源，不把已有RGB再复制到磁盘。模型优化等待当前GPU长训练完成或有足够实测余量后启动，不因人为并发争抢造成OOM而宣称需要加卡。后续按实际单作业资源选择保留信息的执行优化；这不改变用户不以24GB限制研究的要求。CAPA运行仍需原始基座及完整反向；若正常单作业仍不足，再按已授权流程保存、无任务关机并提示加卡。

failure_ledger_delta=update V73-F05（同cohort控制已训练）及baseline preparation；F01/F02/F03/F04/F05仍active，F06直接数据配置缓解但不宣称彻底解决，下一编号V73-F07。整个V7.3持续进行，shutdown=false。

---

## V7.3 完整队列首轮与LiDAR-only强控制（2026-09-08）

完整队列joint r5（code8831def5，PID18843）已完成首个371 Actor更新并进入epoch2，非smoke/短回归。首轮峰值10.196GiB；14次无Actor相机位姿为预期LiDAR路径，17次有视图但预测native支持为空，分别记录，不能合并成F06塌缩率。357有视图更新的native组梯度norm中位数354.00、query组12.85，逐样本norm比例中位数29.51。该证据只反映参数组量级；Adam对尺度有适应性，不能据此声称query没学习，更不能等同逐损失梯度冲突。当前保持r5优化配置不变。首轮诊断=`docs/autoresearch/worldsim_v73/m2/global/population_r5_epoch1_diagnostics.json`。

登记同一完整cohort的LiDAR-only控制：`WS-V73-M2-GLOBAL-ACTOR-01/20260907T193500Z__population-lidar-only-extra-time-s7304-r6`。371fit/67dev可输入、51无输入对象均与r5相同；只读build LiDAR、尺寸和已知轨迹，使用同一个3层局部空间查询解码器和固定三角patch，视觉读取关闭、completion改为LiDAR surface seed、不训练或读取DPT。真实fit额外时刻surface/free标签、query seed7304、lr1e-5、range free0.5、30epoch/11130更新与r5一致；native depth辅助项仅存在于视觉候选。它用于测量多模态候选的整体增量，不能把差异仅归因于某个视觉中间层。

复用r5已经完成的同一cohort LiDAR PCA基线JSON，避免重新计算完全相同的固定算子；r6仍正常评价自己的initial/final表面。复用路径入manifest，基线计算时间不算r6训练成本。r5继续运行；LiDAR-only不持有DPT激活及冻结视觉前缀，先启动真实完整实验并观察资源，而不削减当前视觉候选输入。pointwise控制需同样完整cohort，待GPU空间合适时启动，不盲目叠加两套24视图反向。

强基线代码已查：保留的官方AdaPoinTr PCN为512query/16384输出、600epoch默认训练，预训练权重和可运行v72-pointr环境仍在；旧v72 wrapper含固定旧split及checksum逻辑，本轮不复用该wrapper，后续直接接官方模型与V7.3数据。官方PCN完整GT的双向Chamfer不能直接被解释为稀疏LiDAR的完整表面监督，需明确未知区边界。CAPA官方仓库也已在本机，VGGT LoRA默认rank4、qkv、100steps；应认真适配到build稀疏测量与统一规范融合，而非声称当前native DPT已经复现CAPA。依据：[AdaPoinTr作者实现](https://github.com/yuxumin/PoinTr)、[CAPA项目](https://research.nvidia.com/labs/dvl/projects/capa/)。

failure_ledger_delta=update V73-F05（完整数据真正训练中）及F06（按缺观测原因分开统计）；F01/F02/F03/F04/F05仍active，F06直接测量配置缓解但仍有短时空支持事件，下一编号V73-F07。资源尚足，shutdown=false。整个V7.3未完成，继续训练、分析与基线迁移，最终无任务关机授权不变。

---

