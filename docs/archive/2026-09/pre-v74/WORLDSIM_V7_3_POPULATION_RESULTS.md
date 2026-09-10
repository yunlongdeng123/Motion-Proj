> 历史归档（2026-09-10）：以下为原版本事实与指令快照；当前执行以 docs/RESEARCH_STATUS.md 为准。

# V7.3 完整窗口Actor队列结果

**当前（2026-09-08 22:05 UTC）：三角已全部收口。** R10/R12/R14三角及R11锚点已收口。R12相对R10降低early与自由空间侵入，但增加miss、降低hit与召回；并未同时恢复物理质量与覆盖。相对同beam的R14，R12 free更低、miss更高，hit/距离/召回区间跨0。相对同beam LiDAR控制R8，R12在全部5开发日志降低hit、增加miss。转入Q-v2显式表面参数化研究，保留可训练DPT与物理监督；不继续单纯loss网格，也不退回native-only。 完整更新见[三角报告](WORLDSIM_V7_3_TRIANGLE_RESULTS.md)。以下日期较早的等待/调度状态保留为历史。

## 当前：R14原生beam对照收口，等待R12（2026-09-08 19:05 UTC）

R14完成30轮和完整489对象最终评价，PID68108正常退出。相对R11，hit均值增加2.945pp，但6项开发指标的95%日志配对区间全部跨0，不能声称finite-beam free稳定改善原生路径。相对同beam/free/full_track的LiDAR控制R8，R14召回提高5.253pp，却在5日志全降低hit，free也明显增加。覆盖/物理冲突并不只存在于Query候选，尚不能把根因唯一归到Query曲面片。R12仍在运行，三角未收口。

| 方法 | hit | early | miss | free（m） | 单向distance（m） | recall@.2m |
|---|---:|---:|---:|---:|---:|---:|
| R11 native/hard free | 0.176111 | 0.076432 | 0.600949 | 0.164965 | 0.168504 | 0.751499 |
| R14 native/beam free | 0.205563 | 0.099932 | 0.575200 | 0.166419 | 0.166946 | 0.771985 |
| R8 LiDAR/beam free | 0.310028 | 0.055909 | 0.568336 | 0.034320 | 0.229553 | 0.719452 |

R14−R11 hit+2.945pp、95%[−.370,+7.506]pp，free+.001454m、[−.022062,+.026683]m；六项区间全跨0。R14−R8 hit−10.446pp、[−17.429,−5.270]pp，recall+5.253pp、[+3.180,+8.258]pp，free+.132099m、[+.053660,+.203728]m。相对初始化仍无稳定开发增益，训练native/coverage拟合改善不能替代物理结果。

R14为30轮/10550实际更新/580无梯度跳步，wall8.245h、allocated2.361161GiB；完整489对象和旧输入cohort保持。完整配对、训练图、资源与移动子集边界见[原生控制报告](WORLDSIM_V7_3_MATCHED_NATIVE_CONTROL.md)。原有architecture components图见下文。三角未收口；Q-v2条件决策和独立20日志未曝光状态保持，三本台账更新V73-F09证据，无新失败ID。以下历史结果的调度文字以文首为准。

## 历史：R10 full_track联合训练完成，三角当时仍待R12/R14（2026-09-08 14:30 UTC）

R10完成30轮、11130次真实更新、零恢复/跳步，原371 FIT训练输入、67开发可预测输入与完整489 cohort不变；DPT32654562＋Query1670517参数实际训练，allocated峰值10.213784GiB、wall9.392h。以下是完整75开发Actor/5日志的等权日志统计，8个空表面和23个无owned返回对象均保留。

![当前主路径与原生融合控制](../../../../paper_v73/figures/architecture_components.png)

| 方法 | hit | early | miss | free（m） | 单向distance（m） | recall@.2m |
|---|---:|---:|---:|---:|---:|---:|
| R10初始化 | 0.221164 | 0.122749 | 0.511425 | 0.302697 | 0.156341 | 0.791305 |
| **R10 joint/full_track/hard free** | 0.271463 | 0.202637 | 0.326175 | 0.271026 | 0.113193 | 0.826047 |
| R11 native/full_track/hard free | 0.176111 | 0.076432 | 0.600949 | 0.164965 | 0.168504 | 0.751499 |
| R5 joint/短窗/hard free | 0.222773 | 0.209807 | 0.377094 | 0.351717 | 0.145289 | 0.797126 |
| R7 LiDAR/full_track/hard free | 0.317549 | 0.118390 | 0.498974 | 0.071575 | 0.209524 | 0.745468 |
| R9 LiDAR/full_track/beam+event | 0.324688 | 0.060060 | 0.545686 | 0.038543 | 0.227363 | 0.724665 |
| LiDAR PCA | 0.212448 | 0.043527 | 0.734104 | 0.017708 | 0.304790 | 0.654182 |

同full_track/hard-free目标下，R10−R11：hit+9.535pp、95%[+4.616,+16.268]pp；miss−27.477pp、[−37.927,−20.665]pp；同时early+12.620pp、[+5.904,+18.968]pp、free+.106062m、[+.013990,+.209804]m。距离/召回均值较好但5日志区间跨0。相对R7，missing降低17.280pp，但early+8.425pp、free+.199452m；尚未超过同监督LiDAR控制的物理表现。R10相对R5的missing降低5.092pp、[−10.279,−.610]pp，其他所列指标区间跨0；保留R5恢复RNG差异。

移动子集只有2日志/9 Actor；其中1日志仅1条owned返回，R10的63.0446%移动hit均值不能当稳健动态结论。另一日志6656条owned返回的hit为26.0892%，比R7低9.379pp。保留原分母与聚合，完整配对、训练曲线和分母说明见[同目标联合/原生控制报告](WORLDSIM_V7_3_MATCHED_NATIVE_CONTROL.md)。单向target→surface不是完整表面精度或对称Chamfer。

R12 code dc6fe427/PID81766已于14:13 UTC启动，R14/PID68108继续训练；配置不混入新的near-surface项、对应模块或visual-only cohort。先收口R10/R12/R14与R11锚点；若冲突持续，首要改动仍为Query surface parameterization。F02证据更新，无新失败ID。所有R10结果归档`m2/global/population_joint_r10_*.json`及训练/配对图，20新日志质量未读，整个V7.3未完成。

以下按原研究阶段保留历史结果；其中“已登记/待运行”等调度描述只对应各节当时状态，以本文当前节与三本台账为准。

## 历史：主joint r5最终结果与full_track对照（2026-09-08）

r5完整30轮/11130有效更新和489对象评价已完成。全75开发对象/5日志hit22.2773%、early20.9807%、miss37.7094%、free.351717m、单向距离.145289m、recall79.7126%。相对同短窗r6：hit−8.728pp [−14.097,−4.478]，free+.267963m [+.055532,+.503369]，距离−.095886m [−.221717,−.013341]，recall+7.289pp [+2.275,+14.295]。覆盖增益没有兑现为正确第一表面，不能宣布主假设成立。

完整初始化/强基线表、恢复成本、移动分组和机制判断见`WORLDSIM_V7_3_JOINT_R5_RESULTS.md`，图`autoresearch/worldsim_v73/m2/global/V73_JOINT_POPULATION_RESULTS.png/pdf`。新增观测域点集F-score同样显示r5比r6低7.778pp，严格区别于连续表面召回及完整表面GT。

r10 full_track joint已以code26a7e509/PID53472启动，保持原M1r3/seed初始化和hard free/native/30轮，仅扩fit标签；r11同标签原生强控制继续运行。r12仅改有限宽束free的联合比较已登记未运行。以下保留先前PCA/r6/r7/r8的完整监督与物理目标证据。

## V7.3 米制射线管free完整结果与CAPA资源对策（2026-09-08）

r8 `WS-V73-M2-GLOBAL-ACTOR-01/20260907T212500Z__population-lidar-track-beam-range-s7304-r8` 完成，coded50c9eeb，30epoch/11130更新，4384.52s含完整评价，GPU allocated峰值0.33034GiB、RSS3.00688GiB；PID26975退出。与r7相同原build输入、全轨迹fit标签、架构、seed、优化预算，仅free改为米制有限射线管目标，权重仍0.5。

| 完整开发75 Actor/5日志 | literal hit | early | miss | free m | target到surface m | recall@0.2m |
|---|---:|---:|---:|---:|---:|---:|
| 固定LiDAR PCA | 21.24% | 4.35% | 73.41% | 0.01771 | 0.30479 | 65.42% |
| r7，硬相交range free | 31.75% | 11.84% | 49.90% | 0.07157 | 0.20952 | 74.55% |
| r8，beam_tube_range free | 31.00% | 5.59% | 56.83% | 0.03432 | 0.22955 | 71.95% |

r8−r7日志配对：free−0.03725m、95%[−0.06910,−0.01236]m，4日志降低、1相同；early−6.25pp、[−10.46,−2.40]pp，4降低、1相同。hit−0.75pp、[−5.51,+4.00]pp；miss+6.94pp、[−0.61,+14.49]pp。距离+0.02003m、[−0.02263,+0.07133]m，recall−2.60pp、[−5.01,−0.39]pp。代理带来的侵入改善已反映在中心束硬读出，但有覆盖代价且free仍高于PCA，不称全面优势。

原build LiDAR-ready分层67对象/5日志（16无留出自有回波）：hit33.41%、early5.82%、miss53.65%、free0.03647m、距离0.22955m、recall76.34%。完整主报告保留23无留出自有回波及8空预测。运动>2m/s仅9对象/2日志：hit9.19%、early4.14%、miss81.57%、free0.02204m、距离0.14144m、recall84.82%；相对r7，侵入降低但命中/定位退化，不能将整体free改善外推为动态重建成立。fit旧窗口时刻已属训练标签：hit37.32%、early4.72%、miss52.19%、free0.02283m、距离0.16420m、recall76.49%，不是独立确认。所有分层和逐日志配对保存在population_lidar_r8_summary/analysis.json。

同已修订r2 build背景的场景r3也已实际完成（run `WS-V73-M4-SCENE-COMPOSITION-01/20260907T224500Z__development-full-track-free-r3`，code277c9771，4.051s、RSS0.758GiB、纯CPU，全部416704束保留）。cohort返回：PCA hit21.45%/early31.68%/miss43.14%/free1.04603m；r7为25.07%/37.88%/31.26%/1.09527m；r8为21.55%/31.90%/38.65%/1.05145m。全原始束free为PCA0.32154m、r7 0.32539m、r8 0.32183m。背景已有的大量早面/未知覆盖仍在，场景结果不能只归因于Actor生成；F04仍active。完整数据与配对归档m4/scene_composition_r3_*.json。

CAPA r1实际载入了原始本地VGGT及393216个可训练LoRA参数，但首窗口在官方仿射对齐sort处OOM，尚未完成第一个优化step。失败code277c9771、PID30545已退出、峰值allocated8.60038GiB；当时r5约12.07GiB、Ada r1约1.54GiB并发占用，不能据此宣告单作业必须加卡。r1/status.json与traceback完整保留，未影响另两项正常训练。

按要求先检索[CAPA官方对齐](https://github.com/nv-dvl/capa/blob/main/capa/utils/alignment.py)、[MoGe官方分块求解](https://github.com/microsoft/MoGe/blob/main/moge/utils/alignment.py)和[PyTorch显存文档](https://docs.pytorch.org/docs/stable/notes/cuda.html)，定位全锚点×全观测的中间矩阵。已将仿射锚点按128分块，保留官方所有点、所有锚点、同一weighted-median求解和全局scatter_min选择，GPU抽样seed和12000点上限不变，不缩小RGB/视图。一次CPU数值对比（2例×129点、噪声/离群/零权重、234有效锚点、chunk7）scale/shift最大差均0；这是执行优化对比，不代表完整CAPA已成功。代码=`capa_alignment.py`；记录=`capa_chunked_alignment_comparison.json`。修订r2 `20260907T225000Z__population-build-tta-chunked-s7305-r2` 待提交后重新执行实际100步/31窗口。

后续event机制比较采用r8 free配置：保持surface覆盖、full_track标签和全部预算，加入一个0.01权重的截断首事件项，检验是否能在保留free改善时恢复命中/覆盖；不把r8预选为最终胜出方法。该真实训练尚未启动，等CAPA实际首窗口资源明确后调度。主路线B joint r5继续原配置，后续仍需同full_track监督及强视觉控制。failure_ledger_delta=update F01/F02/F04/F05；F01/F02/F03/F04/F05/F07继续active，F06直接数据配置缓解，下一编号V73-F08。整个V7.3未完成，shutdown=false，不因并发OOM中断正常作业关机。

---


## V7.3 全轨迹LiDAR-only控制最终完成（2026-09-08）

`WS-V73-M2-GLOBAL-ACTOR-01/20260907T203500Z__population-lidar-only-track-labels-s7304-r7` 已完成，code d78e99ae，30epoch/11130更新，3519.80s包含最终完整评价（initial与固定PCA复用），GPU allocated峰值0.19448GiB、RSS2.987GiB；PID23188已退出。输入仍为原build，只有fit surface/free标签扩展到全轨迹；51个原输入不可用对象继续保留空预测，没有把后续测量作为推理输入。开发75对象/5日志、23个无留出自有回波、8个无预测。

| 同一完整开发队列 | literal hit | early | miss | free m | 观测target到surface m | recall@0.2m |
|---|---:|---:|---:|---:|---:|---:|
| 固定LiDAR PCA | 21.24% | 4.35% | 73.41% | 0.01771 | 0.30479 | 65.42% |
| LiDAR-only r6，短窗标签 | 31.01% | 14.02% | 49.31% | 0.08375 | 0.24118 | 72.42% |
| LiDAR-only r7，全轨迹标签 | 31.75% | 11.84% | 49.90% | 0.07157 | 0.20952 | 74.55% |

r7−r6独立日志配对：hit+0.75pp，bootstrap95%[−2.10,+3.37]pp（3/5日志改善）；early−2.18pp，[−4.70,−0.37]pp（4改善、1相同）；free−0.01218m，[−0.02226,−0.00210]m（3改善、1变差、1相同）；target距离−0.03165m，[−0.08340,−0.00251]m（4/5改善）；recall+2.12pp，[−0.03,+4.44]pp。miss+0.58pp，[−4.10,+5.27]pp。更充分的训练观测缓解部分侵入/定位问题，尚未形成全面优势；尤其free仍明显高于PCA。此收益属于标签范围变化，不能归因于视觉或空间交互结构。

原build LiDAR-ready开发分层为67对象/5日志，其中16无留出自有回波：r7 hit33.77%、early12.47%、miss46.77%、free0.07548m、target距离0.20952m、recall78.89%。该分层由原输入元数据定义，与模型质量无关；完整75对象仍为主报告。已知速度>2m/s子集仅9对象/2日志（3无留出自有回波、1空预测）：r7 hit17.73%、early6.84%、miss73.64%、free0.03667m、距离0.08915m、recall85.42%；相对r6有物理侵入改善和命中/缺失退化，不能凭2日志宣称动态泛化成立。

fit旧短窗口时刻也属于训练标签：r7 hit40.30%、early9.50%、miss45.01%、free0.04752m、距离0.15537m、recall79.40%；这些不是独立确认，也不是对整条全轨迹所有标签的完整评价。所有标签仍是稀疏真实观测与框归属代理，不称完整表面GT。

完整summary、原PCA/初始化/final逐Actor记录、r6与native fusion配对、运动及共有输入分层均已归档 `docs/autoresearch/worldsim_v73/m2/global/population_lidar_r7_summary.json` 与 `population_lidar_r7_analysis.json`，采用既有结果汇总，不重跑基线。native fusion与r7标签预算和无LiDAR输入处理有差异，结果表不混为同监督架构比较。

当前主joint r5 PID18843（最近epoch11）、米制beam_tube_range r8 PID26975（最近epoch2、GPU峰值0.33034GiB）正常训练。r8与r7输入/全轨迹标签/架构/seed相同，仅free目标变化；等待真实最终结果决定下一对策。完整主模型后还需用与r7/r8相同标签预算训练joint/原生保守控制，并推进CAPA、AdaPoinTr、event和新日志；不从本LiDAR控制推导视觉路线失败。

failure_ledger_delta=update V73-F02/F05（更多fit观测有局部作用，动态独立样本仍不足）；F01/F02/F03/F04/F05继续active，F06直接数据配置缓解，下一编号V73-F07。场景背景r2及配对已完成，scene无残留作业。整个V7.3未完成，shutdown=false；按既有自动研究持续推进，真正完成且确认无训练/评价/数据/任务队列后再关机。


---

## V7.3 全轨迹训练结束与米制射线管free比较（2026-09-08）

LiDAR-only全轨迹标签r7 PID23188已完成30epoch/11130更新，正在完整489Actor最终曲面评价；训练结束记录elapsed3105.48s，最终wall time和开发结果待summary完成。不把最后一次loss或中间checkpoint当成最终性能。主joint r5 PID18843继续正常训练。

现在进行已实现的米制射线管free目标比较：`WS-V73-M2-GLOBAL-ACTOR-01/20260907T212500Z__population-lidar-track-beam-range-s7304-r8`。相对r7，仅把free-mode由range改为beam_tube_range，仍为米制深度侵入、权重0.5；固定宽度0.03m、32×32栅格，通过同一显式三角面的最近深度及轮廓反传。输入、全轨迹fit标签、371fit对象、67dev可输入对象、51空输入对象、seed7304、AdamW lr1e-5、30epoch/11130更新及query架构均保持一致。复用完全相同的r6 initial与r5固定PCA评价，不重做相同模型读出。

本比较检验“原硬相交free缺少轮廓位移梯度时，保留米制严重程度的有限宽度几何目标能否改善真实侵入”；不混入event、额外opacity、自由半径或结构改动。原25Actor的beam_tube仅覆盖比例目标曾降低early数量却恶化侵入距离；此次range版本已在解析场景验证位置和轮廓梯度，但尚没有真实训练收益证据。它仍是优化代理：有限支持外可无梯度，仍需coverage吸引；宽度不是已校准的真实激光光束，不能将解析梯度结果写成全局收敛保证。官方nvdiffrast来源及已有解析证据见 `WORLDSIM_V7_3_FREE_VISIBILITY_DESIGN.md`。

r8使用现有motionproj训练环境与已编译nvdiffrast0.3.3，CUDA12.1编译器来自保留的v72-pointr环境；不升级Torch、不重建大环境。LiDAR-only作业不加载DPT，启动后以真实峰值记录资源；与r5和r7末尾评价并发的wall time不作单作业速度比较。r7评价结果一旦完成即用新增--reference配对r6，并列full cohort、原build LiDAR-ready、运动日志分层。

背景r2已完成，数据/评价进程均退出，scene结果和图已push3cb18bba。failure_ledger_delta=update V73-F02的单因素比较登记；F01/F02/F03/F04/F05仍active，F06直接数据配置缓解，下一编号V73-F07。CAPA、AdaPoinTr、主joint最终结果、event与新日志确认未完成，shutdown=false。


---

## V7.3 完整原生融合参考完成（2026-09-08）

`WS-V73-M2-GLOBAL-FUSION-01/20260907T203500Z__population-native-lidar-fusion-r2` 已完成，code d78e99ae，489 Actor、671.62s、GPU allocated峰值2.687GiB、RSS6.364GiB。每个24视图窗口固定M1r3 DPT解码一次，再做Actor规范融合和同尺度PCA三角片；没有新增优化或全轨迹标签，不能与更多标签训练的模型声称同预算架构胜负。

开发75对象/5日志的LiDAR PCA→native+LiDAR：hit21.24%→22.78%，early4.35%→9.16%，miss73.41%→57.35%，free0.01771m→0.15030m，观测target到surface距离0.30479m→0.17542m，0.2m覆盖65.42%→78.13%。相对PCA，hit差+1.53pp，日志bootstrap95%[-4.10,+6.35]pp，3/5日志改善；free差+0.13260m，[+0.05887,+0.20172]m，5/5日志变差。距离、覆盖、miss在5/5日志改善。覆盖与物理侵入的冲突仍在，原生融合不是已成立的物理胜者。

dev无预测对象8→3，fit43→10，说明原生视觉支持可覆盖部分无build LiDAR对象；这不能直接解释为其表面正确。dev仍23对象没有留出自有回波，保留未知GT语义。当前主模型的visual-only路径尚未接通，51个空输入对象仍明确计入；后续比较须同时列完整队列及共有可输入对象，不把输入可用性差异偷偷归于交互结构。运动dev仅9对象/2日志：fusion hit9.02%、early9.92%、miss67.06%、free0.20818m；尚不足以确认动态泛化。

结果及配对/运动分层已归档 `docs/autoresearch/worldsim_v73/m2/global/population_fusion_r2_summary.json` 与 `population_fusion_r2_analysis.json`。当前joint r5 PID18843正常训练至epoch8；全轨迹标签LiDAR-only r7 PID23188至epoch14，真实fit_label_times=full_track、GPU allocated峰值0.1945GiB。融合PID23189已结束。没有改动在跑配置，没有发生资源不足。

下一独立工作为F04的场景组合：先查阅[Open3D官方RaycastingScene](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html)与[Street Gaussians作者代码](https://github.com/zju3dv/street_gaussians/)，采用静态背景与只读轨迹Actor各自BVH求交，再统一取最近深度及owner。现有v72-lidar4d环境已安装Open3D0.19，可复用CPU读出，不为场景暴力全对全占用训练显存。静态背景只用build测量，并在每次采样排除已知动态归属；边界、背景未知和非刚体缺失必须分列。该方案尚未产生场景结果，不把Actor-only评价称为场景验证。

failure_ledger_delta=update V73-F02/F04/F05；F01/F03继续active，F06直接数据配置缓解，下一编号V73-F07。CAPA模型、AdaPoinTr、全局组合、新日志确认尚未完成。整个V7.3未完成，shutdown=false；继续研究。


---

完整cohort LiDAR-only r6完成（code98d9fa32，run `20260907T193500Z__population-lidar-only-extra-time-s7304-r6`）：371fit训练Actor、67dev可输入Actor，30epoch/11130更新，2904.27s含自身initial/final评价，峰值0.181GiB、RSS2.964GiB。固定PCA基线从r5复用，其计算成本不包括在r6 wall time中；该run与joint r5并发，不用于单作业速度排名。489对象全部记录，43fit/8dev无输入对象保留空预测；dev75对象中23个无留出自有回波，不把它们当作有几何GT，miss/覆盖与无输出计数同时报告。

dev5日志均值：LiDAR PCA→训练LiDAR-only为 hit21.24%→31.01%，early4.35%→14.02%，miss73.41%→49.31%，free0.01771m→0.08375m，观测target到surface距离0.30479m→0.24118m，0.2m覆盖65.42%→72.42%。hit/覆盖/距离在5/5日志改善，但early/free没有日志改善（4差、1相同）。相对PCA，hit配对差+9.76pp、日志bootstrap95%区间[+4.79,+14.73]pp；free差+0.06605m、区间[+0.03014,+0.09334]m。纯LiDAR生成器同样有覆盖与侵入冲突，因此不能把F02全部归因为高维视觉。当前joint r5还在训练，不能用其25Actor旧结果与完整r6直接比较。

新增按已知build窗口平均速度>2m/s的运动子集汇总（只分析已有结果，不重跑模型）：dev9Actor仅2日志，其中3个无留出自有回波、1个无预测。r6运动子集hit18.70%、miss71.85%、free0.04875m、距离0.11049m；该独立样本量不足以形成动态泛化主张。总体5日志也仍为旧开发集，不是新日志确认。完整结果与配对/运动分层=`docs/autoresearch/worldsim_v73/m2/global/population_lidar_r6_*.json`。

fit全轨迹标签构建也完成（code1c19f2c2，run `20260907T200000Z__fit-track-measurements-r2`）：175.86s、RSS8.553GiB、磁盘356MiB。414对象中407有正观测，合计1690284个Actor内精确坐标去重点（非独立表面样本），7748633条相关原始束，11214个Actor×扫描、9782个标签专用时刻；原build点合计194680。43个无build LiDAR对象中36个在更长时段有标签，仍不将这些后来测量作为输入。更充分观测不是完整表面GT，未知区域语义不变。索引=`docs/autoresearch/worldsim_v73/coverage/fit_track_targets_r2_index.json`，目标Tensor留在单独run目录。

后续r7登记：`WS-V73-M2-GLOBAL-ACTOR-01/20260907T203500Z__population-lidar-only-track-labels-s7304-r7`。相对r6仅将fit surface/free标签替换为独立全轨迹目标；build输入、seed7304、同decoder、lr1e-5、range free0.5、30epoch/11130更新保持相同。新参数--fit-targets仅在fit损失加载target_points_actor_m/target_rays，predict仍只读原case.points_actor_m；native数据辅助项（未来joint使用）也只读build像素。开发集不加载此目录。有效label_times记full_track，fit原留出短窗口已属训练标签；无输入、未参加优化对象不再标记training_labels。复用r6相同输入/同seed模型的initial评价和r5固定PCA结果，避免相同算子重复运行；复用路径写入manifest。

同时登记完整cohort原生保守参考：`WS-V73-M2-GLOBAL-FUSION-01/20260907T203500Z__population-native-lidar-fusion-r2`。使用已认真训练的M1r3 DPT，固定其参数，每个24视图窗口解码一次后供所有Actor共享米制深度；这是评价期固定输出缓存，不是训练中缓存可训练路径。按原规范轨迹融合native+LiDAR、同0.06m PCA曲面片；没有Actor build LiDAR但有native支持时允许native-only，完全无支持则明确缺失。所有489对象均纳入，不能再沿用旧25Actor融合结果代表全队列。其fit标签预算仍为M1 build深度监督、没有全轨迹标签，差异单列，不宣称同标签架构胜负；原生强控制后续仍可用相同几何标签优化。

r5正常运行；r6和标签生成已结束。r7及固定融合评价提交后启动，预计额外GPU开销小于另一套24视图反向，仍以实际占用为准，不把并发争抢当成单作业资源不足。CAPA数据已备、模型未运行；AdaPoinTr接同真实标签与统一表面、场景组合、独立新日志仍待推进。failure_ledger_delta=update V73-F02/F05；F01/F03/F04仍active，F06直接数据配置缓解，下一编号V73-F07。整个V7.3未完成，shutdown=false。
