# 历史原始记录 003

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## 用户条件策略覆盖：先等Q-v2结果（2026-09-09 01:47 UTC）

联合r1/PID96997正在第17/30轮，继续原bfc181b4配置。结果出来前暂停新候选训练和实现扩展；3caffc7a已实现的upper入口保留待用，不自动启动。完整r1−r2按原75 DEV/5日志配对收口，再依用户两条策略推进：joint明确帮助时研究open/structured surface + constructive ray-support与独立upper PEFT；几乎无帮助时优先surface representation + constructive ray supervision，不继续投入DINOv3/full FT。证据不确定时不宣称视觉无用，先聚焦已知表面/沿束支持问题。near-boundary free暂后置。

六项判断在计划revision7第18节逐项区分事实与假设：固定genus-zero是事实，闭合与UNKNOWN冲突、局部collapse/stretch的原因仍未证明；已有target→surface吸引，尚缺明确的正确沿束支持构造约束。r1−r2检验整条视觉几何路径及其辅助监督，不唯一定位某层表示。没有新质量结果、训练或评价run；20新日志未读。关联V73-F01/F02/F03/F04/F05/F06/F09，failure_ledger_delta=none，下一V73-F10。

此决策在3caffc7a基础上写入AGENTS、计划与专项报告，三本台账同步提交/push；30分钟自动跟进同步新优先级并保持ACTIVE，正常无变化时安静，完成不关机。

---

## 上层LoRA训练/恢复入口接入，未启动正式实验（2026-09-09 01:24 UTC）

基于3c824f96，训练入口新增默认关闭的`--upper-lora`；开启时把18–23组qkv LoRA加入DPT/Query优化器，逐步重算全窗口后选择Actor输入。checkpoint保存upper_config/upper_adapter并支持相同定义的resume，固定推理按checkpoint恢复同一通路。旧配置走原路径；正在运行的Q-v2 r1仍使用bfc181b4已加载代码与原配置，没有开启上层适配。

一次新CPU保存—恢复/真实前缀构造检查done：人为赋值的LoRA参数精确恢复（最大差0，adapter文件1581418字节），scene-0015完整24视图4/11/17各[1,24,1301,2048]，两个Actor共享前缀/DPT，旧23不读取。脚本4.518705s/RSS3.002346GiB，无optimizer更新、无聚合/传感器前向；这不验证Adam恢复、完整反传或GPU峰值。先核对官方LoRA及PyTorch optimizer保存顺序，证据`docs/autoresearch/worldsim_v73/upper_tail/checkpoint_check_r1.json`，实现/命令/组件图见上层适配报告；此前小型梯度检查未重复。

01:15 UTC，联合r1/PID96997正在第15/30轮，GPU14436/24576MiB、100%利用率，分配峰值11.591165GiB，RSS约34.08GiB，盘余68GiB。继续训练及完整r1−r2收口，再登记独立机制实验；没有新增正式run、没有读20新日志质量。关联V73-F01/F02/F03/F06/F09，failure_ledger_delta=none，下一V73-F10；30分钟跟进ACTIVE、完成不关机。代码、报告和三本台账同提交并push。

---

## 上层LoRA独立接口落地，真实训练待进行（2026-09-09 00:40 UTC）

在3eeca1e8基础上新增`upper_aggregation.py`及一次CPU接口脚本，代码/证据/本次记录同提交。实现VGGT第18–23组qkv LoRA（默认rank8，393216参数）、4/11/17冻结前缀与重新生成23层的DPT接口；保留全窗口24视图顺序，之后才选择Actor输入，不跨优化步缓存适配输出。未接入运行中的Q-v2 r1或正式优化器，完整传感器梯度与GPU峰值均pending。

真实12个尾部Block仅CPU加载，冻结参数151182336；随机3视图/11token/32维/rank2检查对官方交替处理输出最大误差0，首frame/末global LoRA B梯度非零，冻结输入及原权重无梯度，一次合成更新改变输出，第三视图能影响第一视图。脚本1.779956s/RSS1.508625GiB，不能当成24视图资源或科学收益。首次位置张量非连续错误按官方PositionGetter的clone修复后，同一检查通过；保留首错与通过证据，不新增科学失败ID。

报告及组件图：`docs/WORLDSIM_V7_3_UPPER_AGGREGATION_ADAPTATION.md`、`docs/autoresearch/worldsim_v73/upper_tail/V73_UPPER_TAIL_INTERFACE.png`；原始`interface_check_r1.jsonl`及`interface_check_r1_attempt1.txt`同目录。官方VGGT、LoRA与PyTorch源码接入依据见报告。20新日志质量未读，未下载环境/权重。

Q-v2联合r1/PID96997于00:26 UTC第11轮正常：GPU分配峰值11.591165GiB、RSS约34.08GiB，盘余约68GiB；无OOM事件。继续原配置并等待完整r1−r2，随后分别决定表面局部形状、对应、certified-free或上层适配，不能把接口可用当成上层已训练。关联V73-F01/F02/F03/F06/F09，failure_ledger_delta=none；原风险保持，下一V73-F10。30分钟ACTIVE、完成不关机。

---

## Q-v2 r2固定表面分解完成（2026-09-08 23:55 UTC）

support-r3 `WS-V73-M2-SURFACE-SUPPORT-01/20260908T235000Z__qv2-lidar-r2-vs-r8-support-r3`已done，执行基于a587bd6b与已登记收口工作树；75原DEV/5日志、11886 owned束、CPU.612499s/RSS.631134GiB，无神经推理或更新。联合r1继续训练，不改变其配置或读取20新日志。

r2日志等权early19.4901%中6.5867个百分点有后方正确交点，12.9034个百分点缺正确沿束支持；late24.4222%（R8为6.5727%）。missing且有邻近表面由30.1786%降至4.0498%，不能当成正确hit改善。池化2092条early中只有290条有后方正确交点，和日志加权各自报告。两层交点可能只是闭合表面入/出，不认定自交；该诊断只含owned束，不归因全部背景侵入。

下一步等待r1−r2：当前证据不支持仅靠更多free排斥解决几何，也不拒绝视觉适配。根据主联合结果区分位置/局部形状/视觉表示与前面遮挡，按既定条件独立研究；不扩loss网格。Q-v2报告、八类图与`qv2/support_r3/{summary,manifest}.json`同步。failure_ledger_delta=update V73-F02 evidence; no new failure ID，下一V73-F10；30分钟ACTIVE、完成不关机。

---

## Q-v2 LiDAR r2收口：低missing未兑现正确表面（2026-09-08 23:50 UTC）

共享网格LiDAR r2已完成，但未同时改善物理与覆盖：相对同beam R8，miss−31.295pp，early+13.899pp、free+.106203m，三项95%日志配对区间均不跨0；hit−.454pp、单向distance−.067188m、recall−7.120pp区间均跨0。更多束有交点不等于表面更准确；该冲突在无视觉输入时仍存在，不能归咎于背景视觉token或因此拒绝可训练几何基座。联合r1继续，最终r1−r2整通路比较尚pending。

task `WS-V73-Q-V2-01` / r2 run `20260908T230000Z__shared-mesh-lidar-full-track-beam-s7304-r2`，code95050522、分析a587bd6b；30轮11130实际更新、零跳步/恢复、完整489对象最终评价done。耗时2505.097482s、GPU.252920GiB、RSS1.876888GiB、checkpoint11658264字节。75 DEV/5日志含23无owned与8空表面；hit/early/miss/free/distance/recall=.305489/.194901/.255388/.140523/.162366/.648249。r2−R8 miss区间[−40.344,−22.374]pp、early[+7.212,+22.236]pp、free[+.041433,+.175709]m；其余三项跨0。20新日志未读。

已先核对Mesh R-CNN、Point2Mesh官方源码和Open3D交点接口，详见Q-v2报告文末。下一步登记固定r2/R8表面支持判别 `WS-V73-M2-SURFACE-SUPPORT-01/20260908T235000Z__qv2-lidar-r2-vs-r8-support-r3`（pending），区分错误前表面、后方有效支持与支持缺失；不重推理或复制闭合/双向完整表面真值，不凭两层交点判自交。联合r1/PID96997第7轮正常，等待r1−r2后再改训练因素，保留可训练几何基座主线。

三本台账、计划与`WORLDSIM_V7_3_QV2_SHARED_MESH.md`同步；归档`qv2/shared_mesh_lidar_r2_{summary,final_manifest,analysis,training}.json`和两图，主分析只执行一次，图中native选择对LiDAR标不适用。failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F04,V73-F05,V73-F06,V73-F09]；failure_ledger_delta=update V73-F02 evidence; no new failure ID，下一V73-F10。30分钟ACTIVE、完成不关机。

---

## Q-v2两支真实训练；固定网格评估入口就绪（2026-09-08 23:10 UTC）

LiDAR r2 `WS-V73-Q-V2-01/20260908T230000Z__shared-mesh-lidar-full-track-beam-s7304-r2`已从95050522启动，PID98643。23:08:11 UTC快照：完成489对象initial，第4轮，已记录1459呈现/1459实际更新，Query梯度非零、DPT梯度0，allocated峰值0.252920GiB；GPU总15163/24576MiB，cgroup oom/oom_kill=0，磁盘仍68GiB可用。联合r1/PID96997仍正常训练（其allocated峰值11.591165GiB）；不改两支配置。LiDAR日志的views=24/6只是相机位姿元数据，不代表读取图像，744前缀未加载。两支正式质量均pending。

固定checkpoint评估器按保存的query_surface/mesh_level构造共享网格；旧checkpoint缺字段仍走原patch类。原脚本硬编码独立patch，不能直接加载新source embedding和mesh buffers，此处在正式固定推理前完成接入。保存真实顶点/面数与parameterization，surface_patches保持兼容字段但明确shared_mesh计数是顶点，移除“所有方法0.06m patches”的错误通用描述。仅CPU加载一次r2已完成轮次的真实latest.pt，全部state keys匹配、642顶点1280面；未启动额外推理、回归或读取20新日志，原训练入口/参数不变。

新增收口入口`scripts/summarize_worldsim_v73_qv2.sh lidar_r2`与`... joint_r1`：各在相应完整final结束后执行一次。r2对R8；r1对r2/R12/R14/R8，沿用已有六指标、完整489分母/75 DEV/5日志配对与10000次seed7304 bootstrap，不回算旧对照。固定推理准备不等于候选已获准进入新日志；仍在方法选择结束后确认。

Q-v2的source=2标记共享网格顶点，不能复用旧patch的primitive_id//8或vertex_id//9来源归因。既有只读vertices/faces的BVH支持诊断可在final后按失败需要使用；闭合网格一入一出本来可能有两个深度层，不能据此断言自交或重复patch。此为预先明确表示语义，不是新增科学失败或修订已发表指标。

证据`qv2/shared_mesh_lidar_r2_manifest.json`、`qv2/shared_mesh_lidar_r2_training_start.json`与当前代码差异；报告Q-v2固定评价节同步。failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F04,V73-F05,V73-F06,V73-F09]；failure_ledger_delta=none（运行与新表示接入，无新质量结论），下一V73-F10。三本台账/报告同步，小步push；30分钟ACTIVE、完成不关机，下一步等待正式final并推进已登记Q-v2机制判别。

---

## Q-v2继续；登记同网格LiDAR控制（2026-09-08 23:00 UTC）

22:55 UTC主r1/PID96997第3轮正常，DPT/Query梯度均非零、allocated峰值11.591165GiB、RSS约34.08GiB；无OOM，磁盘68GiB可用。根据F02/F09的参数化与视觉归因风险，提前训练同网格LiDAR控制，而不等主r1完成才补齐。r1保留可训练DPT+显式共享网格+physics，配置/进程不变。

task `WS-V73-Q-V2-01` / run `20260908T230000Z__shared-mesh-lidar-full-track-beam-s7304-r2`登记pending，基于a74b1dda，启动manifest记录实际代码。原cohort、seed7304、30轮/full_track、642顶点1280面、相同Query类与尺寸椭球；mode=lidar_only，512支持种子来自build LiDAR，native-data-weight=0。coverage/free.5/beam.03/res32/event0/envelope.05/AdamW1e-5保持。此为移除整个视觉/native通路及其辅助监督的控制，不称attention-only因果实验。未调用视觉参数无梯度，不把summary的requires_grad总数误报为全部实际更新。

新initial实际评价、固定PCA复用；优先r2−R8参数化差异，再r1−r2同网格通路比较，保持489完整分母/5 DEV日志与未读20新日志。只增加这一必要控制，不展开矩阵。详见Q-v2报告新增节；分析脚本仅保留输出顶点/面数、表示与路径元数据，指标/聚合/bootstrap不变，不重算旧对照。6+6线程在14CPU配额内，实际新作业资源待启动记录；运行耗时披露并行竞争。

failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F04,V73-F05,V73-F06,V73-F09]；failure_ledger_delta=none（对照登记，无新结果），下一V73-F10。源码依据复用已核实Pixel2Mesh/Mesh R-CNN与现有LiDAR训练路径，未出现新卡点、不加smoke/回归/新依赖。三本台账与计划同步并push后启动，30分钟ACTIVE、完成不关机。

---

## Q-v2进入真实共享训练（2026-09-08 22:24 UTC）

首轮`WS-V73-Q-V2-01/20260908T221500Z__shared-mesh-full-track-beam-s7304-r1`（code bfc181b4、PID96997）完成489对象初始化评价，现为shared_train第1轮，快照已记录116次呈现/116次真实更新。首步DPT/Query裁剪前梯度分别544.504028/11.606274，当前allocated峰值11.591165GiB；输出642共享顶点/1280面，24视图实例已正常反传。此为实际执行与梯度证据，不作质量结论；正式30轮继续，不改变配置或提前读取新日志。

证据`qv2/shared_mesh_r1_training_start.json`与原run的initial/train/status，论文r3在此前22:20快照基础上保持质量pending；failure_ledger_delta=none，F02/F03/F04/F05/F06/F09保持，下一V73-F10。30分钟跟进ACTIVE、保持开机，完成后按已登记R12/R8/R14比较继续研究。

---

## Q-v2正式作业运行；阶段论文r3同步三角（2026-09-08 22:22 UTC）

Q-v2 `WS-V73-Q-V2-01/20260908T221500Z__shared-mesh-full-track-beam-s7304-r1`已从提交bfc181b4启动，PID96997，目标30轮。22:20 UTC状态initial_evaluation，438可用Actor/51不可用、371 FIT与744全窗口冻结前缀已加载；实际完成更新数尚未产生，不把初始化评价说成已完成训练。沿用R12 full_track/native1/free.5/beam.03/res32/event0与M1r3 fresh初始化、seed7304。进程正常、cgroup oom/oom_kill=0，服务器保持运行。原run保存manifest/cohort/status和复用PCA结果；后续checkpoint、train.jsonl、完整final与配对结果按原训练器保存，不新增重复检查。

阶段论文`paper_v73/main.tex`更新为r3（22:20UTC），已完整纳入R10/R12/R14/R11结论、14方法主表、12项三角配对与Q-v2实际architecture components图。明确R12的free收益伴随missing增加、相对R8仍无正确命中优势；Q-v2只是已实现且运行中的候选，固定拓扑、面密度、初始化差异均明示。R5旧结果/恢复、R13一轮边界、移动2日志稀疏性、输入覆盖与20新日志未读状态保留。未新增模型推理或bootstrap。

现有TinyTeX编译9页/344398字节成功，9页视觉检查完成、图表可读，无overfull或未定义引用；末条参考文献有一条轻微underfull提示，无内容溢出，不为消除提示反复编译。论文归档`docs/autoresearch/worldsim_v73/paper/interim_r3.pdf`，本地outputs另存可读PDF和源zip；旧r1/r2保持历史。图源/四表导出脚本与README同步，未新增环境、smoke或回归。

三角后的两项建议仍可独立研究：near-boundary只能增强经认证返回前区间中违规面的梯度，R12已有更低free但更多miss，因此不单独扩大排斥来替代表面支持；新局部对应项补候选深度/偏移射线/可靠归属，现有可学习offset与attention继续保留。Q-v2首轮不混入这两项或上层LoRA；根据显式网格的真实失败再分别接入。详见计划16–17节、Q-v2报告和上层接口报告。

failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F04,V73-F05,V73-F06,V73-F09]；failure_ledger_delta=none（运行与论文里程碑，无新科学结果），下一V73-F10。30分钟ACTIVE、完成不关机；正式作业结束后先按相同日志比较R12/R8/R14，持续推进Q-v2、必要的上层表示/输入与场景验证，不等待用户确认。

---

## Q-v2共享顶点表面已实现，正式训练登记（2026-09-08 22:12 UTC）

依据三角收口07e98727与V73-F02/F03/F04/F06/F09，检索Pixel2Mesh ECCV2018、Mesh R-CNN ICCV2019官方论文/源码及PyTorch3D细分。实现`ActorSharedMeshQueryDecoder`：642共享顶点/1280面，从只读尺寸椭球先验初始化；最多1024 LiDAR+512原生深度查询提供隐状态，经网格边、最近观测和既有局部视觉读取更新顶点。固定拓扑不保证无自交，不对观测取凸包，UNKNOWN不标FREE。DPT32654562+Query1668393可训练；原24视图冻结前缀保持。详细来源、预算差异与architecture components图见`WORLDSIM_V7_3_QV2_SHARED_MESH.md`。

task `WS-V73-Q-V2-01` / run `20260908T221500Z__shared-mesh-full-track-beam-s7304-r1` 登记pending，代码提交后立即启动30轮。保留R12的原cohort、M1r3初始化、seed7304/full_track/native1/free.5/beam.03/res32/event0/AdamW1e-5。新initial实际评价，旧PCA baseline复用；不resume旧patch、不混入新loss/上层LoRA/visual-only。输出1280面与旧最多12288不同，比较属于参数化方案而非等密度纯拓扑因果控制；最终先对R12，再R8/R14。20新日志未读。

唯一合成路径检查通过：共享边归属2面、轴向首交点[2,2,1.2]m、四层视觉及原生种子梯度均非零。初次4×4合成粗图使既有offset全部越界，导致全尺度正梯度断言失败；核对grid_sample与valid mask后仅扩大合成图重试通过，未改模型、未改真实输入门槛，不另分科学失败ID。归档`qv2/shared_mesh_path_check.jsonl`，该检查不是质量或全DPT资源证据；没有回归/多轮smoke。

failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F04,V73-F06,V73-F09]；failure_ledger_delta=none，参数化效果pending，旧风险保持，下一V73-F10。运行入口`scripts/run_worldsim_v73_qv2_shared_mesh_r1.sh`，正式manifest记录实际commit。空闲GPU24GB，数据盘68GiB可用，资源不足未触发；完成不关机，30分钟ACTIVE。后续分别研究certified-free边界、局部对应、上层几何适配，训练期间先同步技术报告与无需修改模型的后续准备。

---

## V7.3 三角收口：转入 Q-v2（2026-09-08 22:05 UTC）

R10/R12/R14三角及R11锚点已收口。R12相对R10降低early与自由空间侵入，但增加miss、降低hit与召回；并未同时恢复物理质量与覆盖。相对同beam的R14，R12 free更低、miss更高，hit/距离/召回区间跨0。相对同beam LiDAR控制R8，R12在全部5开发日志降低hit、增加miss。转入Q-v2显式表面参数化研究，保留可训练DPT与物理监督；不继续单纯loss网格，也不退回native-only。

R12 run `20260908T050000Z__population-joint-full-track-beam-range-s7304-r12` / task `WS-V73-M2-GLOBAL-ACTOR-01` / code dc6fe427，分析d9637d0e：30轮11130更新/0跳步/0恢复，完整489最终评价done、PID81766退出。DPT32654562 + Query1670517可训练，M1r3初始化、full_track/native1/free.5/beam.03/res32/event0；原cohort与744前缀保留。wall27059.008429s、GPU10.235226GiB、RSS34.104195GiB、native_project变化.005098347。

75 DEV/5日志全部保留23无owned与8空表面。R12 hit/early/miss/free/distance/recall=.215934/.056423/.651456/.060573/.154347/.762202。R12−R10：hit−5.553pp [−13.635,−.239]、early−14.621pp [−19.532,−9.688]、miss+32.528pp [+27.867,+38.476]、free−.210453m [−.315461,−.110354]、distance+.041154m、recall−6.385pp，六项区间不跨0。early/free改善伴随支持覆盖与正确命中损失，不能单列free当成功。

R12−R14：free−.105845m [−.177679,−.036656]、early−4.351pp [−7.986,−.992]，miss+7.626pp [+5.122,+10.203]；hit/距离/召回区间跨0。R14−R11六项跨0结论复用。R12−R8：hit−9.409pp [−12.540,−6.705]、miss+8.312pp [+5.322,+11.592]，5日志全退化；其余四项跨0。通路比较包含多因素，不唯一归因Query或视觉token。

训练native加权Huber .707176→.388714m，采样coverage .307158→.224648m，hard free .208722→.029032m；两组梯度真实非零。FIT含训练标签、移动2日志其中1条owned返回限制保留。20新日志质量未读。详见`WORLDSIM_V7_3_TRIANGLE_RESULTS.md`（实际architecture图、完整均值/配对/训练/资源）；4份R12 JSON与两图归档，汇总只执行一次，未重推理/回归。

failure_ledger_refs=[V73-F02,V73-F03,V73-F04,V73-F06,V73-F09]；failure_ledger_delta=update V73-F02 evidence; no new failure ID，下一V73-F10。Q-v2任务`WS-V73-Q-V2-01`进入实现准备，首要因素为显式表面参数化；目标、数据与DPT适配范围保持，后续局部对应/near-boundary/上层适配独立研究。资源正常，GPU现空闲供下一轮；30分钟ACTIVE、完成不关机。

---

## V7.3 上层跨视图适配接口准备（2026-09-08 19:50 UTC）

基于96709415及R14结果，独立核对VGGT官方/本机aggregator、DPT、attention及PyTorch2.4.1 checkpoint/SDPA。没有改训练或提前选择Q-v2；详见`WORLDSIM_V7_3_UPPER_AGGREGATION_ADAPTATION.md`，含明确标注未实现的architecture components图。若以18–23组适配为例，可以复用冻结17组global半部及4/11/17的DPT输入；23组必须重算。层号为0起，6组是12个frame/global block，不把旧最终缓存当适配结果。

关键接入约束：当前Actor视图裁取在全窗口冻结聚合之后；上层可训练时须先保持完整24视图和原顺序/特殊token/RoPE，再为Actor选择DPT输入。不能在global attention之前裁视图；不能跨optimizer步缓存已适配输出。若改变为同窗口多Actor合并更新，预算与loss权重必须单独披露。非重入checkpoint保留缓存输入上新参数的梯度；其间冻结FFN也不能no_grad切断输入梯度。

只CPU读取一份M1r3缓存与权重shape：4层[1,1,1301,2048]为FP32，patch_start5；qkv3072×1024、projection1024×1024。24视图全局31224token，单状态FP32 121.969MiB、4拼接输入975.750MiB；显式16头score为BF16 29.055GiB/FP32 58.111GiB，仅理论反例、不是实测OOM。rank8/末6组qkv LoRA 393216参数，加proj589824，不含DPT/Query/激活。SDPA后端与实际反向峰值待首次实施时记录，不因理论score申请加卡或减少视图。

证据`m2/global/upper_tail_metadata_r1.json`、提案图与对应源码/绘图脚本；无新模型/训练/评价run，无测试、无下载新权重或依赖，20新日志质量未读。failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F06,V73-F09]；failure_ledger_delta=none，旧风险保持，下一V73-F10。19:38 UTC R12/PID81766第20轮正常，GPU12660/24576MiB、cgroup oom/oom_kill=0；先完成三角再定Q-v2，30分钟ACTIVE、完成不关机。

---

## V7.3 R14原生beam对照完成：未形成稳定物理增益（2026-09-08 19:05 UTC）

`WS-V73-M2-GLOBAL-ACTOR-01/20260908T100000Z__population-native-full-track-beam-range-s7304-r14`已done，codebf04ef32，PID68108退出。30轮/11130呈现/10550真实更新、580次no_surface_gradient（420无相机、160有相机无有效原生梯度）、零恢复。DPT32654562参数、Query冻结，native_project变化.004810881；wall29683.499341s（8.245h）、allocated2.361161GiB、RSS34.697498GiB。744份冻结前缀保留，398679588字节checkpoint与完整489最终记录/表面保存；无上层聚合器适配。

全部75开发Actor/5日志、23无owned/8空表面保留。R14 hit/early/miss/free/distance/recall为.205563/.099932/.575200/.166419/.166946/.771985。R14−R11只改free目标：hit+2.945pp、95%[−.370,+7.506]pp，free+.001454m、[−.022062,+.026683]m，六项区间全部跨0，未建立稳定收益。不能把R8−R7的LiDAR free改善外推到原生路径。

同beam目标的R14−R8：hit−10.446pp、[−17.429,−5.270]pp，5日志全下降；recall+5.253pp、[+3.180,+8.258]pp，5日志全提高；free+.132099m、[+.053660,+.203728]m。覆盖/物理冲突亦见原生路径，不足以唯一定位到Query曲面片。该比较包含基座/支持/PCA等通路差异，不作单模块因果归因。相对初始化hit−2.221pp区间跨0，不能称稳定改善或显著退化。

训练native测量加权Huber .805201→.469110m、采样coverage .318582→.242145m；hard free .134530→.152671m、beam目标 .135196→.151870m。损失/过程不等于泛化；FIT距离−.011413m、recall+3.213pp区间不跨0，但full_track标签包含该评价时刻。移动仍9Actor/2日志，其中一日志仅1条owned返回，不能支持稳健动态主张。

一次原聚合/10000次seed7304日志配对已保存，未重推理/重测旧基线。`m2/global/population_native_r14_{summary,manifest,analysis,training}.json`及`V73_NATIVE_R14_{TRAINING,PAIRS}.png/pdf`归档，图已视觉核对；MATCHED_NATIVE_CONTROL、POPULATION_RESULTS、COMPARISON_PROTOCOLS同步，保留实际architecture components图。执行复用汇总/绘图脚本，未新增测试或独立科学run。

failure_ledger_refs=[V73-F02,V73-F03,V73-F04,V73-F09]；failure_ledger_delta=update V73-F09 evidence; no new failure ID，下一V73-F10。19:01 UTC R12/PID81766第17轮正常，两组梯度为正、cgroup oom/oom_kill=0；继续R12后收口三角，Q-v2 pending。旧已核实问题复用VGGT/LoRA3D/CAPA与参数化源码依据，不发起loss网格/重跑。20新日志质量未读，30分钟ACTIVE、完成不关机，资源不足出口未触发。

---

## V7.3 R14训练完成，最终评价运行中（2026-09-08 18:13 UTC）

`WS-V73-M2-GLOBAL-ACTOR-01/20260908T100000Z__population-native-full-track-beam-range-s7304-r14`完成30轮、11130次呈现和10550次实际更新，580次无有效梯度跳步；PID68108仍在运行最终评价，尚无final/summary，不登记为done或推断质量。训练阶段elapsed29061.685493s，最终含评价wall待收口。保持M1r3/seed7304、full_track、native1/free.5/event0，唯一目标因素为beam_tube_range。18:07 UTC R12/PID81766第13轮正常；GPU15731/24576MiB、cgroup oom/oom_kill=0、磁盘69GiB可用，无资源不足或shutdown条件。

下一步直接复用既有`summarize_worldsim_v73_global_results.py`与`summarize_worldsim_v73_training.py`读取R14完成产物，做R14−R11原生目标对照及R14−R8同beam目标的通路比较；仍用全部75开发Actor/5日志、原聚合与一次10000次seed7304配对bootstrap，不再推理或重测旧模型。R14−R8不是纯视觉特征单模块因果对照。三角完整结论仍待R12；20新日志质量未读。

配对图入口`scripts/plot_worldsim_v73_joint_r10_pairs.py`新增可选model-label/reference/protocol-note，以同一绘图方法呈现R14及后续三角结果，默认R10配置保留；只读取保存的区间，不新增bootstrap。此次仅静态审阅绘图差异，尚未生成/核对R14图。执行脚本暂存`/root/autodl-tmp/controller_logs/summarize_native_r14.sh`，未启动，没有后台等待队列；下次看到R14 summary与done后执行一次。改动不涉及训练或现有模型。

failure_ledger_refs=[V73-F02,V73-F03,V73-F04,V73-F09]；failure_ledger_delta=none，最终科学判断pending，下一V73-F10。用户revision6继续生效：Q-v2 pending、30分钟跟进ACTIVE，完成不关机；若三角coverage/physics冲突持续，优先Query surface parameterization，再分别检验近边界监督、局部对应与必要的上层适配。

---

