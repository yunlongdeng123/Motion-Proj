# 历史原始记录 009

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V7.3 场景配对收尾与CAPA分块后真实训练（2026-09-08）

复用场景r3保存结果完成r8−r7配对：cohort free−0.04382m、日志bootstrap95%[−0.07655,−0.01284]m，4日志降低、1相同；early−5.98pp、[−9.44,−2.52]pp；hit−3.52pp、[−9.62,+1.79]pp；miss+7.39pp、[+0.22,+14.55]pp。场景侵入降低仍伴随覆盖代价，且背景自身误差占比大，不宣告F04解决。全部原始束、cohort、背景/其他对象、边界代理与近传感器分层保留在m4/scene_composition_r3_summary.json，配对在scene_composition_r3_paired.json。没有新增模型推理。

完整Actor开发队列的PCA/r6/r7/r8六项结果图已生成 `docs/autoresearch/worldsim_v73/m2/global/V73_POPULATION_FREE_TRADEOFFS.png/pdf`；柱为日志等权均值、点为5条独立日志，图中明确r6短窗标签与r7/r8全轨迹标签差别。图不使用旧25Actor诊断子集替代当前75个开发对象，运动样本仍仅2日志的限制保留在正文。

CAPA修订r2已实际启动：run `WS-V73-M2-CAPA-01/20260907T225000Z__population-build-tta-chunked-s7305-r2`，codeeb42f835，PID31466，日志 `/root/autodl-tmp/controller_logs/v73_population_capa_r2.log`。原始完整VGGT和393216个LoRA参数成功加载，首窗口已越过原OOM位置进入真实反向/优化，官方step0/10/20/30/40的L1读数为3.8186/1.8611/2.0940/1.4381/1.2993。随机视图子集不同，不能把这五个数当成同样本学习曲线或开发效果。随后首个scene-0015窗口已实际完成全部100步、全24视图联合推理、LoRA/深度保存及其Actor评价；适配与保存267.033s，累计GPU allocated峰值8.08180GiB，现进入scene-0071。首窗口已跨过完整优化/推理路径，其余30窗口仍待完成，不称整个基线完成或所有规模资源问题已解决。

最近GPU进程占用r5 12358MiB、Ada r1 1582MiB、CAPA r2 9880MiB，不能再叠加新GPU作业。当前GPU余量限制属于并发调度；正常训练继续，先等现有作业释放资源，再启动已准备的event单因素对照与Ada轴修订r2，不为关机强行结束正常任务。Ada r1最近epoch17/1567次优化更新，原完整初始化输出已保留；其轴未迁移限制仍按F07报告。

下一event run登记为 `WS-V73-M2-GLOBAL-ACTOR-01/20260907T230000Z__population-lidar-track-beam-event-s7304-r9`：相对r8仅event-weight=0.01，固定σ0.2m/C28/width0.03m/res32，保持371fit/完整489队列、全轨迹fit标签、原build输入、seed7304、30epoch/11130更新与free配置。复用相同r6 initial和r5 PCA；无支持不从event分母删除。此时未启动，无后台自动启动队列，待有实测资源后执行。后续主joint同full_track适配、强视觉/补全控制、独立新日志及完整应用仍需推进。

failure_ledger_delta=update F01/F02/F03/F04/F05/F07状态；F01/F02/F03/F04/F05/F07继续active，F06直接数据配置缓解，下一编号V73-F08。三项长训练/适配均正常，整个V7.3未完成，shutdown=false；15分钟自动研究继续，真正完成后保存/push并确保无任务/队列再关机。

---


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


## V7.3 首事件解析语义完成并接入可选训练项（2026-09-08）

`WS-V73-M3-FIRST-EVENT-01/20260907T224000Z__geometry-first-event-r2` 完成一次解析实验，code518268eb，0.927s。σ0.2m/C28/footprint0.03m/32²：正确5m面NLL0；4.6m早面遮住5m正确后面时NLL2.000001、早面沿深度梯度−10.000003、后面梯度0；复制早面仍NLL2.000001。无支持NLL28、质量0、梯度0；轮廓偏移0.025m时质量0.839767、NLL0.174630、横向梯度+6.97729。重复面没有增益，被遮挡的正确后面不能绕过早面，缺失支持的死梯度也确实仍存在，不能写成已解决F03。

首个r1启动在解析场景执行前因shell PATH未包含已有ninja可执行文件而失败；使用现有motionproj/bin和保留的CUDA12.1编译器修正PATH后执行同一组案例，未安装包、未改变方法或扩展回归。r1/failure.json保留错误与对策，r2完整结果归档 `docs/autoresearch/worldsim_v73/m3/analytic_first_event_r2.json`。

现已将该代理接入共享训练器的可选--event-weight（默认0）。event读取与free同一随机原始束子集中的positive_actor，避免把后方Actor或背景返回当作当前Actor应生成的表面；没有额外随机抽样改变其余控制的样本顺序。完整owned子集包含无支持束，训练日志保存supervised/no_support/capped、几何返回质量与字面owned miss；原surface coverage和独立free保持。当前仍无真实event训练结果。r8最终评价完成后固定free目标再登记/启动一个event权重比较，不能从解析案例推断真实收益。

failure_ledger_delta=update F03解析与实现证据；F01/F02/F03/F04/F05/F07仍active，F06直接数据配置缓解，下一编号V73-F08。r5与AdaPoinTr r1正常训练，CAPA等待r8退出。整个V7.3未完成，shutdown=false。

---


## V7.3 几何首事件代理实现（2026-09-08）

依据已登记的首事件设计，实现 `motion_proj/worldsim_v73/first_event.py`：固定footprint内最近三角面、米制深度插值、预乘likelihood轮廓梯度、log平移求和及全束统一截断。far裁剪由当前几何决定，不读取target；没有opacity或target挑面。返回supervised/no_support/capped计数与几何返回质量，明确缺失支持处仍可能零梯度。当前未接入训练器，不改变r5/r8或Ada作业。

登记一次解析实验 `WS-V73-M3-FIRST-EVENT-01/20260907T224000Z__geometry-first-event-r1`，检查正确面、早面遮住正确后面、重复早面、无支持和轮廓位置梯度；相应脚本为 `scripts/diagnose_worldsim_v73_first_event.py`。此时仅代码完成、实验尚未执行，不能声称代理正确或真实有效。r8仍在最终完整评价；完成后再依据r7/r8硬结果选free目标并接入单因素event训练。

failure_ledger_delta=update F03实现状态；F01/F02/F03/F04/F05/F07仍active，F06直接数据配置缓解，下一编号V73-F08。主joint r5与AdaPoinTr r1正常训练，CAPA待r8退出再启动。整个V7.3未完成，shutdown=false。

---


## V7.3 首事件缺失支持的可实施边界（2026-09-08）

检索DS-NeRF CVPR2022、终止分布EMD ECCV2024和nvdiffrast官方实现后，整理 `docs/WORLDSIM_V7_3_FIRST_EVENT_DESIGN.md`。候选分布来自固定footprint内同一显式表面的几何首交点，无独立opacity、无重复支持概率累加、无提前裁掉错误早面。只对当前Actor实际拥有的首返回做event；其他原始束保留free。

严格s=0的NLL无穷和位置零梯度没有被log-domain“解决”。拟考虑全owned束上的显式截断NLL代理，同时保留直接surface coverage与独立free；missing取同一最大损失、不能从分母删除，截断比例与hard miss另报。此方案尚未实现/训练，不称严格似然、新物理模型或已经消除F03；该设计解释了为什么不能把数值ε或代理下降当作支持恢复。待r7/r8完整硬结果后固定free配置，再做一次有归因的event比较。

当前r8完成30epoch/11130更新，进入完整489对象最终评价；r5与AdaPoinTr r1正常训练。CAPA已登记但未启动，等待r8退出后的显存余量。failure_ledger_delta=update F03方法边界；F01/F02/F03/F04/F05/F07仍active，F06直接数据配置缓解，下一编号V73-F08。整个V7.3未完成，shutdown=false。

---


## V7.3 CAPA全窗口实际适配运行登记（2026-09-08）

准备运行 `WS-V73-M2-CAPA-01/20260907T223500Z__population-build-tta-s7305-r1`，沿用已记录的官方100步rank4/alpha8、patch_embed qkv LoRA、每步3/24视图随机采样、最终全24视图联合推理；31个build窗口、完整489 Actor队列。所有744视图已有有效稀疏深度条件，fit/dev均只在各自build输入上窗口内TTA，逐窗口重置；额外时刻仅评价。协议、对齐及native-only边界见 `docs/WORLDSIM_V7_3_CAPA_BASELINE.md`。

已静态核对模块路径：主工程NativeGeometryPyramid只在实例化时加载普通VGGT，而该CAPA入口未实例化它；实际CAPA wrapper优先加载其VGGT_VPT。因此没有发现此前担心的普通VGGT抢先导入问题，不添加清模块之类补丁。上游DINO与跨视图aggregator均有nonreentrant checkpoint路径，现有train()覆盖这些路径；保留全部图像与视图信息。开启官方已有每10步loss日志并记录窗口累计显存峰值，不额外建重复测试。

本登记不代表权重加载或LoRA训练已成功。计划在r8训练/最终评价退出并有显存余量后直接执行真实完整运行，避免并发争抢被误判为单作业资源不足；当前不建立后台自动启动队列。AdaPoinTr r1与主joint r5保持各自配置。F01/F02/F03/F04/F05/F07继续active，F06直接数据配置缓解，下一编号V73-F08。整个V7.3未完成，shutdown=false。

---


## V7.3 AdaPoinTr官方坐标接口修订（2026-09-08）

V73-F07：首轮AdaPoinTr r1使用本项目Z-up Actor轴输入PCN预训练模型，尚未迁移其车辆坐标约定。依据适配前物理误差继续查阅官方源码，确认[PoinTr NormalizeObjectPose](https://github.com/yuxumin/PoinTr/blob/master/datasets/data_transforms.py)及[PCN test_kitti](https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py)在车辆规范化后显式交换Y/Z，输出再逆变换。该证据说明现有预训练接口有可修正的域差异，但并不单独证明r1全部误差由此造成。

已为后续训练加入input_frame=pcn_y_up，输入(x,y,z)→(x,z,y)，所有coarse/fine/denoised输出变回Actor米制坐标后计算同一损失与表面。保持已知box最大维度的各向同性尺度、原输入/全轨迹fit标签/seed/网络/预算不变；不依据target拟合轴或尺度。矩阵仅定义模型坐标接口，不修改Actor轨迹和相机标定。保留input_frame=actor用于明确复现原r1。

r1 PID28663继续原已加载代码，记录不覆盖，作为未迁移预训练轴的对照；不把其最终结果独自作为认真迁移的强基线。登记修订版 `WS-V73-M2-ADAPOINTR-01/20260907T231000Z__population-full-track-pcn-yup-s7307-r2`，与r1相同30epoch/全参数微调，待当前Ada作业完成后启动，不设后台自动重启或重复队列。r2当前仅实现/登记，未训练；F07状态active，下一failure编号V73-F08。F01/F02/F03/F04/F05仍active，F06直接数据配置缓解。主joint r5与米制free r8照常，整个V7.3未完成，shutdown=false。

---


## V7.3 完整AdaPoinTr已实际微调，初始结果归档（2026-09-08）

`WS-V73-M2-ADAPOINTR-01/20260907T221000Z__population-full-track-pcn-s7307-r1` 已成功加载官方epoch353 checkpoint，准确可训练参数32494657。运行代码23981936，PID28663，日志 `/root/autodl-tmp/controller_logs/v73_population_adapointr_r1.log`。489对象的初始化评价已全部完成，随后开始完整模型任务微调；最近快照为epoch4、optimizer_update290、非零梯度范数1.6422、GPU allocated峰值1.00239GiB。该快照证明真实优化已进行，不是最终拟合或效果结论；最新状态以run/status.json为准。

| 完整开发75对象/5日志，适配前 | literal hit | early | miss | free m | target到surface m | recall@0.2m |
|---|---:|---:|---:|---:|---:|---:|
| 固定LiDAR PCA | 21.24% | 4.35% | 73.41% | 0.01771 | 0.30479 | 65.42% |
| 官方预训练AdaPoinTr，尚未本轮微调 | 9.37% | 4.94% | 54.97% | 0.11617 | 0.21732 | 67.11% |

适配前相对PCA，独立日志配对hit−11.87pp、95%区间[−21.74,−2.01]pp，4日志下降、1相同；free+0.09846m、[+0.02770,+0.16252]m，4日志增加。miss−18.44pp、[−30.88,−3.52]pp，4日志降低。表面距离−0.08747m、[−0.19164,+0.01970]m；recall+1.70pp、[−11.13,+10.12]pp。覆盖和字面交点并未同步改善，不能据此宣告完整适配后的AdaPoinTr失败。此处只评价匹配数量的PCA曲面片；完整16384点输出已保存，原生密度与显式化差异还需分开分析。

初始完整报告包含23个无留出自有回波和8个无输入空预测；共有LiDAR输入分层67对象/5日志，hit10.18%、early5.23%、miss51.80%、free0.12335m、距离0.21732m、recall71.71%。运动子集仍仅9对象/2日志，不能作为动态泛化确认。逐Actor、所有分层与配对保存在 `docs/autoresearch/worldsim_v73/m2/global/adapointr_r1_initial_analysis.json`，运行manifest另存同目录。仅汇总已有初始化结果，未重复推理或训练。

主joint r5 PID18843继续运行；同全轨迹标签的米制beam free r8 PID26975最近到epoch26，尚未完成最终评价。当前三项训练均有正常进展，不关机。failure_ledger_delta=update F01/F02/F05（真实完整强基线启动及适配前起点）；F01/F02/F03/F04/F05仍active，F06直接数据配置缓解，下一编号V73-F07。CAPA、全轨迹joint、event及新日志确认仍待完成，整个V7.3未完成，shutdown=false。

---


## V7.3 完整AdaPoinTr任务适配实现与登记（2026-09-08）

主joint r5与米制free r8正常训练，当前不重复启动。独立完成官方AdaPoinTr强控制的实现，登记 `WS-V73-M2-ADAPOINTR-01/20260907T221000Z__population-full-track-pcn-s7307-r1`。使用现有官方完整PCN权重与v72-pointr环境，512query/16384点、6层encoder/8层decoder、384维、全部模型参数微调；不使用旧V7.2包装器或更小输出头。

先核对官方模型/训练配置，再迁移到本任务的稀疏真实标签：保留补全和去噪结构，统一物理曲面coverage/free/envelope，另有0.1 coarse覆盖和0.1局部稀疏去噪项；不把未采样表面惩罚为错误。所有原build点均输入；精确kNN按查询分块保留全keys，输入不足512时只重复已有点。米制归一化仅用已知box最大维度，target不进入输入或尺度估计。

首轮30epoch/11130 Actor呈现、预计2790优化更新（每4个Actor累积），AdamW1e-4、wd5e-4、21epoch乘0.9、clip10、seed7307。fit使用与r7/r8相同全轨迹观测，dev无梯度，371fit/67dev可输入者优化/评价，51个无输入对象仍空预测并计入分母。未完成真实模型加载/训练前不能称强基线已经完成；拟合不足时再按证据扩大预算，不把30epoch当成足够性的门控。

物理主评价用min(build,1024)+512个FPS中心和相同0.06m PCA三角片，完整16384原生点输出另存。每步邻域/PCA架重新估计但固定其本步梯度，曲面位置梯度传回预测中心；因此属于条件位置梯度近似，不能隐藏为完全相同原生几何参数化。既有PCA只读基线的算子数值不变。协议差异、来源、保存与资源策略详见 `docs/WORLDSIM_V7_3_ADAPOINTR_BASELINE.md`。

failure_ledger_delta=update强补全基线实现/登记，F01/F02/F03/F04/F05仍active，F06直接数据配置缓解，下一编号V73-F07。CAPA、主joint完整与全轨迹标签适配、event、新日志确认仍待推进。资源按真实执行记录，整个V7.3未完成，shutdown=false。


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

## V7.3 场景配对结果与运行状态（2026-09-08）

复用既有结果完成独立日志配对（`scripts/summarize_worldsim_v73_scene_composition.py`；10000次日志bootstrap、seed7306，不重跑模型）：背景近点修订使5/5日志的全束free降低，平均差−0.20861m、95%区间[−0.39920,−0.07872]m；early差−1.74pp、区间[−4.03,−0.34]pp。miss在5/5日志增加，平均+1.69pp、区间[+0.29,+4.02]pp。真实场景误差降低和覆盖代价须同时报告，仍为旧5日志开发证据。

修订背景上，r6对PCA的cohort hit差+7.09pp，[+4.05,+10.18]pp，5/5日志改善；cohort free却增加0.04469m，[+0.01348,+0.08405]m。native fusion对PCA的cohort hit差−1.84pp，[−6.90,+3.22]pp；free增加0.02376m，[+0.01274,+0.03450]m。未形成完整物理优势。全部组、逐日志差值和分母保存在 `docs/autoresearch/worldsim_v73/m4/scene_composition_r2_paired.json` 与原summary；不同fit标签预算不混称同监督方法胜负。

r8已实际启动：code d50c9eeb，PID26975，日志 `/root/autodl-tmp/controller_logs/v73_population_lidar_beam_range_r8.log`，真实free_mode=beam_tube_range、full_track标签，已到epoch2、GPU allocated峰值0.33034GiB，非零query梯度。joint r5 PID18843至epoch11，GPU峰值10.196GiB；r7 PID23188在final_evaluation，已写423/489个最终surface文件，仍有正常进展，不重复启动或中断。场景数据/评价任务均已结束。

failure_ledger_delta=update V73-F02/F04证据与运行状态；F01/F02/F03/F04/F05仍active，F06直接数据配置缓解，下一编号V73-F07。r7最终summary未生成前不下结果结论。下一步完成r7→r6完整/共有输入/运动分层对比，继续主模型、强基线、event与新日志。整个V7.3未完成，shutdown=false。


---

