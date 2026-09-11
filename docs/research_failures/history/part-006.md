# 历史原始记录 006

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V7.3 visual-only训练入口已实现，尚未执行新cohort训练（2026-09-08）

F05的固定推理已证明41 fit/5 development个零LiDAR但有相机位姿的对象可生成表面，尚未证明物理改善。本次将同一输入规则接入共享训练器：显式`--include-visual-only`、默认关闭；只依据build点数与已知相机位姿纳入，不依赖原生候选数、target质量或heldout误差。完整主cohort仍489，开启后训练412 fit、预测72 development，2 fit/3 development无任何输入对象仍保留缺失。R10/R11/R12既定371 fit/67 development可用输入对照均保持默认关闭。

读取已完成input audit的既有统计：新增41 fit对象中34个共有54416个full_track正目标，另外7个无正目标但有真实full_track近框首回波；41个合计349221个Actor–ray实例，完全无目标/无原束者0。该计数是FIT标签可用性，不是新域测试；原束可跨Actor重复，不能当独立样本。不能把7个无正目标对象的未知表面标空，它们仅受到原始首回波之前的free约束及既定弱框正则。

原生支持与LiDAR均为空时，joint/pointwise用已有coarse查询，仍读取四层可训练DPT特征；native_only保留真实空表面，不能伪造它具备查询生成能力。训练中空target/空surface的coverage记录null和原因，避免空均值NaN；无原束时free为零但不声称有观测。没有任何实际测量监督时跳过optimizer，不靠框正则计成数据训练；真实有标签但无几何梯度/无表面另记原因。原始owned miss与event absent-support语义保留。

新增cohort不能复用旧initial预测或通过resume悄悄排除新增对象：新实验从M1开始、重新计算initial；固定推理自动继承checkpoint的visual-only训练配置，旧checkpoint仍需显式override且标为未训练输入条件。汇总保留全489主表，新增原metadata零LiDAR分组、coarse fallback、不可用coverage与跳步原因；不按非空预测选择分母。

四个修改脚本的单次py_compile语法检查通过；尚未运行新的真实反向/完整新cohort训练，不把语法检查或固定推理称为训练验证。下一步在现有训练释放显存后执行一次必要的真实零LiDAR梯度验证，再根据R10/R12目标对照选择单独新cohort训练；不同时混改当前目标隔离实验。本轮没有第三个训练进程或等待启动队列，R10/R11继续；R12仍未启动。F02/F03/F05保持active，下一失败编号V73-F09，外部20日志质量确认仍未读取。整个V7.3未完成，shutdown=false。

---


# Motion-Proj 统一失败、风险与防重复账本

## V7.3 全51零LiDAR固定推理完成：接口可运行、未训练退路仍侵入free（2026-09-08）

`WS-V73-M2-EMPTY-INPUTS-01/20260908T063000Z__fixed-joint-r5-visual-only-r2` code6ebf1289完成，r5最终固定checkpoint、0更新、51/51对象；wall75.879384s、allocated GPU2.686777GiB、RSS13.917145GiB，PID58903退出。与R10/R11并行时合计GPU20282MiB，未发生资源阻断。相机/标定/轨迹只读，未读外部20日志，未重训或增加opacity。

fit43对象中41输出512片表面：32个当前r5 native种子、9个coarse退路；2无相机对象保留空。开发8对象中5输出512片且均为native种子，3无相机保留空。先前M1r3 native fusion fit33有支持/8相机对象无支持与当前不同，因为原生头不同；不能把所有变化归因于查询。r5的coarse退路此前主要随原生种子小残差训练，本次仅是输入条件迁移。

已有逐帧硬读出按Actor/独立日志汇总：fit9日志，43894 build近框实例/22932 heldout实例，r5 free .344801/1.201042m，M1r3 native fusion .073563/.024463m；增加支持同时明显违反已观测free。fit heldout24归属束/14对象/6日志，r5 hit5.556%、miss57.222%，native .556%/86.667%；主体已适配过这些fit日志，不作新日志泛化主张。开发2日志70 build/59 heldout近框实例，r5 free0/0，native .179105/0，但唯一归属回波(scene0919/94ffa142)两方法都缺失；不以零free或更多表面宣布几何收益，不为单束生成bootstrap。

报告`docs/WORLDSIM_V7_3_EMPTY_INPUTS.md`已更新；原始摘要`m2/global/empty_fixed_r2_summary.json`及分组`empty_fixed_r2_analysis.json`保留分母、缺失与每Actor记录。新增`scripts/summarize_worldsim_v73_empty_predictions.py`只处理已存结果，无重复推理或射线读出。主训练器visual-only入口尚未增加，下一实现应显式支持空点集、有正fit目标/仅free负约束/无监督三个情形，保留coarse退路的真实梯度与无支持边界，再作为独立cohort变化认真训练；不得暗改R10/R11/R12。

R10/R11仍在原PID53472/38460训练，R12仍仅登记；外部20日志质量未读。F05输入能力问题已定位但训练与质量未解决，F02/F03仍在，其他风险状态不变，下一失败编号V73-F09。整个V7.3未完成，shutdown=false。

---



## V7.3 零LiDAR核查完成与固定视觉入口登记（2026-09-08）

`WS-V73-M2-EMPTY-INPUTS-01/20260908T062000Z__population-build-availability-r1` codeb9690161完成，CPU1.249847s/RSS.362228GiB、零推理/更新。43个fit零LiDAR对象分布9日志，其中41有相机位姿和保守框视锥重叠、33有既有native支持、36有full_track正目标共58710点。41相机对象内34有正目标、8无既有native候选、9无build近框束。8个开发零LiDAR对象分布2日志，5有相机且已有native支持；全部8对象合计只有1条heldout归属回波，不能用此子组承载几何精度主张。fit/development原build近框束总数43894/70，缺失束和无相机对象均保留。

详见`docs/WORLDSIM_V7_3_EMPTY_INPUTS.md`，原始核查归档m2/global/empty_inputs_r1_summary.json。相机视锥仅可能性、native候选不是正确性。F05中“无LiDAR即无可预测输入”属于当前主接口限制；不改变已有完成结果，也不暗中修改当前R10/R11或R12的训练cohort。

固定推理器新增`--include-visual-only`显式入口与metadata-only的`--input-subset zero_lidar`。两类表面支持均空时，seed函数返回None并使用现有coarse位置查询；有原生支持则沿原路径，不伪造LiDAR点。原生融合模式无两类支持仍空，双模态均缺失仍空。所有旧默认运行保持原协议，R10/R11已载入的训练代码不变。r5的coarse参数此前主要随原生seed小残差训练，此输入退路未充分训练，不声称已有质量收益。

登记`WS-V73-M2-EMPTY-INPUTS-01/20260908T063000Z__fixed-joint-r5-visual-only-r2`，r5最终固定checkpoint、全51零LiDAR对象、0更新/无外部数据。登记时尚未执行；此轮只接固定推理入口，训练器visual-only入口仍待独立变更。F05继续，下一失败编号V73-F09；整个V7.3未完成，shutdown=false。

---



## V7.3 F05零LiDAR输入的实际视觉支持核查登记（2026-09-08）

R10/R11在06:10UTC继续原PID53472/38460、epoch5/18，实际native反向正常，GPU15503MiB；未启动重复训练。主训练器仍根据旧`unavailable_input`排除43 fit/8 development零LiDAR对象，而已有population native fusion允许零LiDAR但非空原生支持输出。需要把输入能力与实现限制分开，不能把“无LiDAR”自动解释为“无图像几何支持”。

先查[VGGT官方模型](https://github.com/facebookresearch/vggt)、[原生头forward](https://raw.githubusercontent.com/facebookresearch/vggt/main/vggt/models/vggt.py)及[SparseNeuS官方实现](https://github.com/xxlong0/SparseNeuS)：前者为CVPR2025，后者ECCV2022，均支持由图像建立几何，但这不保证当前动态Actor在米制对齐或遮挡条件下重建正确。本轮不另装新基座/表示；优先利用已有原生深度支持路径。

登记`WS-V73-M2-EMPTY-INPUTS-01/20260908T062000Z__population-build-availability-r1`，新增`scripts/analyze_worldsim_v73_empty_inputs.py`：完整51个metadata零LiDAR对象，读取现有case的标定/框/原build近框束与已完成native-fusion结果；仅fit侧统计full_track标签可用性。相机框与5个视锥平面的重叠是保守几何可能性，不是实际可见性；已有native框内候选也不是精度真值。零新推理、零重训、无新域读取，不按预测质量挑对象。

登记时尚未执行；是否为后续训练增加显式visual-only入口需先看实际输入与目标覆盖。当前R10/R11及R12既定对照保持原配置，避免同时修改free目标与训练cohort。F05仍active，其他风险不变，下一失败编号V73-F09；整个V7.3未完成，shutdown=false。

---



## V7.3 TSDF场景对照完成：背景遮挡改善，联合模型前表面仍错误（2026-09-08）

场景`WS-V73-M4-SCENE-COMPOSITION-01/20260908T054500Z__development-vdb-background-r8`与`20260908T054500Z__development-vdb-build-carved-r9` codebe34e7fc完成，CPU3.86073/3.83265s、RSS.68997/.68205GiB；两项构建、顺序读出脚本及3份配对分析均已退出。固定75 Actor表面、6场景/5日志/416704原束/11886 cohort束，共同TSDF背景，无新网络推理/训练。

仅背景TSDF+build雕刻相对PCA+build雕刻：全束hit26.020→22.714%、early10.678→4.896%、miss53.279→62.802%、free.265644→.082163m。日志配对free−.183481m、95%[−.274646,−.111157]，early−5.782pp [−8.896,−2.720]，miss+9.522pp [+5.340,+13.772]，hit−3.307pp [−4.795,−1.935]；全5日志都减少侵入，也都降低命中并增加缺失。cohort被背景提前遮挡25.191→5.839%、free.893276→.166678m，分别差−19.352pp [−35.229,−3.475]与−.726597m [−1.527483,−.149778]。F04解释了大量原组合错误，但背景覆盖成本仍在，不能称已经解决。

同TSDF+雕刻下cohort r5 hit26.905%、early29.099%、miss27.565%、free.316813m；r9为32.872%/13.768%/44.601%/.208408m，native fusion24.465%/17.043%/49.525%/.208454m，PCA27.833%/12.247%/57.210%/.189840m。r5−r9 early+15.331pp [+7.145,+25.152]、free+.108405m [+.039898,+.180905]，5/5差；miss−17.036pp [−25.065,−9.008]，hit−5.967pp [−17.113,+5.874]。r5−native early+12.056pp [+5.436,+18.838]、free+.108359m [+.044972,+.163334]，5/5差。主联合模型仍以错误前表面换取较低缺失，F02不由背景修复消除。

仅r5换背景cohort hit+5.498pp [+0.334,+10.663]、early−13.948pp [−24.603,−3.449]、free−.690803m [−1.477539,−.149474]，但miss+4.386pp [+2.733,+6.242]；不归因于Actor架构。移动6657原束/2日志r5 early52.840%、miss21.800%、free.301967m，r9 early2.674%、miss78.733%、free.011671m，样本与覆盖权衡不能省略。CAPA2/Ada2/r8及全组完整表见`docs/WORLDSIM_V7_3_VDB_BACKGROUND.md`。

原始summary及3份配对归档m4/vdb_scene_*，构建index亦已保存；图V73_VDB_BACKGROUND.png/pdf已检查。TSDF+build雕刻作为后续开发主背景候选，PCA+build雕刻继续固定敏感性对照，不靠更少支持宣布完整场景成功。新20日志质量确认仍未读，最终背景需在读取前固定。R10/R11在05:51UTC继续原PID53472/38460、epoch4/17，GPU15503MiB；R12尚未启动、无GPU等待队列。F02/F04持续，F03/F05边界保留，下一失败编号V73-F09；整个V7.3未完成，shutdown=false。

---



## V7.3 VDBFusion背景已构建，登记同表面场景比较（2026-09-08）

背景数据r4 `WS-V73-M4-SCENE-DATA-01/20260908T054000Z__development-vdbfusion-background-r4` code061cef64完成，CPU62.0251s/RSS6.90116GiB、磁盘457MiB；原PID56321已退出。相同6场景/5日志/24个build扫描，实际积分606366背景测量，生成727557三角面，保留官方TSDF体积。未读heldout值，无GPU训练/新查询网络推理。

随后r5 `20260908T054000Z__development-vdbfusion-build-carved-r5`同code完成，CPU31.6688s/RSS6.73702GiB、PID56605退出。所有原build束one-pass排除17781片(2.444%)，保留709776片；12017条build矛盾束→0，剩余build侵入距离和0。该结果仅属于构建时段，不能称新时刻精度提升，也不保证未知射线正确。所有原heldout束、owner、Actor轨迹仍链接原数据；不按开发错误删面。

构建原始index归档`m4/vdb_background_r4_construction.json`与`vdb_background_r5_construction.json`。登记场景比较`WS-V73-M4-SCENE-COMPOSITION-01/20260908T054500Z__development-vdb-background-r8`、`20260908T054500Z__development-vdb-build-carved-r9`，固定r5 joint、r8/r9 LiDAR、native fusion、CAPA2、AdaPoinTr2，另含相同LiDAR-PCA及仅背景读出。仅改变背景表示/是否使用既定build雕刻，仍评价全部416704原束及同一11886 cohort束，不删近传感器困难组。登记时尚未执行，两项由短CPU脚本顺序运行，不是训练队列；运行期间不得shutdown。

背景体素分辨率与三角数不同于旧PCA，这正是此背景方法对照，不能混成Actor输出密度控制。需要同时看缺失、hit、early、free和cohort分项，不能凭零build侵入或面数下降宣布F04解决。R10/R11继续，R12未启动，外部20日志质量确认仍未读。F04持续，下一失败编号V73-F09；整个V7.3未完成，shutdown=false。

---



## V7.3 F04迁移：官方VDBFusion静态背景对照登记（2026-09-08）

针对原PCA背景在同一build雕刻后仍会提前遮挡Actor的F04，重新查询优秀开源并读取[PRBonn VDBFusion官方仓库](https://github.com/PRBonn/vdbfusion)、原生积分及MarchingCubes源码。官方CITATION列为Sensors2022、DOI10.3390/s22031296，不能误标为ICRA。该实现直接接收世界点云和传感器原点，适用于现有LiDAR输入，不需要构造近似针孔深度图。已在现有CPU/Open3D环境worldsim-v72-lidar4d安装vdbfusion0.1.6官方769.9kB wheel、--no-deps；没有升级Torch/CUDA或新增大环境。

登记同一6开发场景/5日志的背景构建`WS-V73-M4-SCENE-DATA-01/20260908T054000Z__development-vdbfusion-background-r4`。固定voxel=.1m、trunc=.3m、space_carving=true、统一权重；与PCA r2相同4个build扫描，排除每时刻全部已知注释框+.1m内点及采样LiDAR abs(x),abs(y)<1m近点，无输入点数截断。只读build测量，旧heldout束/owner/位姿链接原文件。fill_holes=false、min_weight=0使mesh单元8角都需观测权重，不让未知体素的默认正值闭合虚构表面；这是已安装官方API的参数语义。

由于排除了动态端点的射线不会通过VDBFusion默认积分产生它们的自由前缀，随后同样使用既有one-pass全原build射线雕刻，登记`20260908T054000Z__development-vdbfusion-build-carved-r5`。这样TSDF与PCA两类表面都能通过所有已观测build首回波约束；不向动态端点后方刻空、不用heldout质量删三角面。分别保存雕刻前后结果及体积/表面规模，后续固定相同Actor比较literal hit/early/free/miss；TSDF变化不能归为Actor架构收益，也不预先保证更好。

新增`scripts/prepare_worldsim_v73_vdb_background.py`；以上两项在登记时尚未执行。R10/R11继续既有训练，R12仍只登记，外部20日志模型质量仍未读取。F04继续active而非修复完成，其他风险不变，无重复smoke/回归、无新增校验机制。整个V7.3未完成，shutdown=false。

---



## V7.3 查询来源诊断完成：新增支持贡献大部分自由空间错误（2026-09-08）

`WS-V73-M2-QUERY-PROVENANCE-01/20260908T052000Z__development-fixed-joint-lidar-r1` code05db8db9完成，CPU0.693519s/RSS0.634186GiB、零训练更新。全部75旧开发Actor/5日志，43832近框Actor–ray实例/11886原归属束；BVH真实首交点primitive ID映射每query的8三角面。原始实例可能跨Actor重复，来源标签仅描述初始化，双方后续都共享空间/视觉更新，不能称纯模态因果贡献。

r5 build来源10419片、中心到最近build均距.048996m、free贡献.029052m、early4.538pp、hit19.990pp；completion34304片、均距1.087350m、free.322666m、early16.443pp、hit2.287pp。上述为Actor内归一→日志内Actor平均→独立日志等权，completion占总free91.740%、总early78.370%。r6 build来源均距.448301/free.010546，completion均距.066475/free.073208。来源贡献相加恢复已有硬评价；中心到最近build距离不是自身初始位移，稀疏支持之外本身不是错误或自由空间。r5原始owned early计数build622/completion364与日志等权占比不同，必须保留统计口径。

全部9元数据速度>2m/s对象均保存、可视化，3个无heldout归属返回者仍保留。移动组仅2有观测日志：r5 completion free.188118m/early58.396pp，build free.026976m/early4.373pp。scene0359/c07f236a只有1条归属束，r5 completion提前.580631m、r6缺失，对该日志移动均值影响很大，不夸大动态样本量。67非空r5表面顶点到片中心最大.103816m，未靠无界半径或opacity扩大支持。图线段是同束原测量到实际首交点，不是最近邻；全9对象按owner排序且两方法坐标范围一致，地面投影不替代三维评价。

报告`docs/WORLDSIM_V7_3_QUERY_PROVENANCE.md`，原始摘要归档`m2/global/query_provenance_r1_summary.json`，来源PNG/PDF及全部移动Actor三页PDF/PNG已生成检查。发现不支持对所有查询加强零位移先验；继续既定R10全轨迹joint/R11强原生头及仅改finite-beam free的R12。R10/R11在05:27UTC仍实际训练(epoch3/16)，GPU合计15503MiB；R12仅登记、未启动、无后台等待队列。新域20日志模型质量确认仍未执行。F02/F04继续，F03/F05边界保留，未新增重复失败编号，下一V73-F09；整个V7.3未完成，shutdown=false。

---



## V7.3 固定首交点的查询来源诊断登记（2026-09-08）

r10 full_track joint继续epoch2、r11 native-only继续epoch15，均正常反向；当前合计显存约15.5GiB，无新资源阻断。为区分“观测支持被移离”与“新增补全造成错误前表面”，登记`WS-V73-M2-QUERY-PROVENANCE-01/20260908T052000Z__development-fixed-joint-lidar-r1`，覆盖全部75旧开发Actor、r5与同短窗r6的已存最终表面，尚未执行。

新增`scripts/analyze_worldsim_v73_query_provenance.py`，通过CPU BVH的真实首交点primitive ID对应每query的8个三角面，关联保存的source标签：0为初始化于build LiDAR的查询，1为新增completion查询。两者后续都参与同一空间/视觉更新，因此这是初始化来源诊断，不是“纯LiDAR贡献/纯视觉贡献”的因果消融。没有删除查询、重排遮挡、重新神经推理或训练，也不选择错误较大的对象。

对全部heldout原始近框束分解free侵入，对明确归属首返回的束分解early/hit/late；缺失束无来源标签且保留。各来源free和事件贡献先按Actor原始束数归一，再日志内Actor及独立日志等权；另外保留原始Actor–ray实例计数，它们可能跨Actor重复同一物理束，不能当独立样本。各来源中心到build测量的最近距离及0.2m外比例单独报告，离开build支持本身不等于错误/幻觉。

保存全部9个元数据速度>2m/s对象的原测量、实际首交点、首片来源及返回mask，用于后续可视化；3个没有heldout归属返回者也保留，不按效果挑例。补全错误与观测查询漂移的区分有助于决定是否需要对可靠build支持加直接几何约束，但在读到结果前不改r10/r11，不预设强零位移正则。r12有限宽束free仍仅登记。F02/F04持续，其他风险状态不变，下一编号V73-F09；整个V7.3未完成，shutdown=false。

---


## V7.3 full_track联合训练实际启动与主模型场景读出完成（2026-09-08）

`WS-V73-M2-GLOBAL-ACTOR-01/20260907T233000Z__population-joint-full-track-s7304-r10` 已实际启动，code26a7e509、PID53472，日志`/root/autodl-tmp/controller_logs/v73_population_joint_r10.log`。从原M1r3/seed7304初始化，复用相同PCA/initial评价；无resume-from，不继承r5已训练权重。唯一相对r5的目标修改是fit标签扩为既定full_track；hard range free.5/native1/event0/30轮保持。已进入epoch1实际反向，native/query梯度均非零，当前allocated峰值10.19725GiB；不是仅登记或smoke。r11原生full_track继续原PID38460。r12有限宽束比较仅登记、尚未启动，无后台等待启动队列。

nuScenes场景`WS-V73-M4-SCENE-COMPOSITION-01/20260908T044000Z__development-joint-r5-carved-r7` code26a7e509已完成，CPU3.7789s/RSS.78560GiB。固定r5最终Actor表面与同一r3 build雕刻背景，6场景/5日志/416704原束/11886cohort束。全束hit26.6942%、early11.0288%、miss52.0292%、free.285150m；cohort hit21.4071%、early43.0467%、miss23.1791%、free1.007616m。相对同背景r8：cohort early+13.021pp、95%[+5.801,+23.188]，free+.098427m [+.036532,+.167488]，miss−16.054pp [−21.983,−9.849]，hit−1.052pp [−9.228,+9.222]。相对r9 early+13.306pp [+5.937,+23.406]、free+.095135m [+.033168,+.167766]。因此减少缺失仍以更多前表面为代价，不能把returned MAE下降当主结果成功。

r5−同背景PCA cohort early+13.247pp [+6.464,+23.108]、free+.103847m [+.040282,+.169700]；全5日志都更差。移动返回组6657束/仅2日志，r5 hit17.0222%、early53.3954%、miss20.5379%、free.304949m；旧r5背景比较run未存此分组，不虚造跨run移动配对。原始按日志/组统计、精确计数及配对归档m4/scene_composition_r7_summary.json与r7_paired.json。此场景按日志内真实束加权，与Actor等权单体表不可直接数值相减。

旧AV2场景`WS-V73-M4-AV2-SCENE-01/20260908T044000Z__old-development-joint-r5-carved-r3` code26a7e509完成，CPU8.5166s/RSS.94763GiB；一个旧日志、21固定Actor、逐束真实时刻、187494原束/5257cohort束。cohort r5 hit53.4716%、early30.5878%、miss9.8916%、free.314317m；同背景r9为54.2134%/9.2068%/17.2912%/.074238m，PCA为62.9446%/9.5872%/21.6664%/.060124m。4移动Actor对应88束，r5 hit22.7273%、miss45.4545%、free.855627m。接口及组合已实际运行，但没有跨域物理优势；单日志不提供bootstrap区间。证据m4/av2_old_scene_r3_summary.json/r3_paired.json。

外部20日志数据/背景/完整冻结前缀都已生成，尚未进行新域模型质量确认；不能因旧AV2运行或构建结束宣称独立确认完成。数据盘当前约72GiB可用，两个训练并行时cgroup oom/oom_kill仍0；缓存/并行执行压力需按实际记录，不能把cgroup reclaim计数称为OOM。已完成CPU场景进程全部退出。

F02/F04由完整联合模型和同背景读出得到进一步负结果，F03缺支持梯度仍待后续联合物理目标研究，F05新域/完整GT边界保留；F06直接头监督缓解、F07轴接口完成而物理未解决、F08恢复结果完整且原退出原因未知。F01没有触发真实资源不足停机，下一编号V73-F09。图m2/global/V73_JOINT_POPULATION_RESULTS.png/pdf已检查并修正过长横轴标签，无重复网络推理/回归。整个V7.3未完成，shutdown=false，继续自动研究。

---



## V7.3 主联合r5最终完成：覆盖改善伴随物理退化（2026-09-08）

r5恢复run `20260908T012500Z__population-joint-r5-epoch21-resume-r1` code914d582d正常完成全部30轮/489cohort最终评价，原PID39009已退出。11130有效更新=7791继承+3339新增，原107未保存更新独立记账；DPT32654562/query1670517可训练参数，恢复wall11359.0603s、GPU10.19917GiB/RSS34.05680GiB，原观测wall22180.1436s，合计下界33539.2039s。DPT project变化.0018500仅相对恢复epoch21。F08恢复执行完成，退出原因仍未知，无OOM归因。

全75开发Actor/5日志：r5 hit22.2773%、early20.9807%、miss37.7094%、free.351717m、target→surface.145289m、recall79.7126%。相对同短窗r6：hit−8.728pp、bootstrap95%[−14.097,−4.478]，5/5差；free+.267963m [+.055532,+.503369]；距离−.095886m [−.221717,−.013341]、recall+7.289pp [+2.275,+14.295]。相对initial命中无明确改善，early+8.706pp [+1.221,+18.493]；相对native fusion free+.201413m [+.036521,+.399522]、early+11.826pp [+5.400,+23.016]。移动9对象仅2日志early62.7680%、recall91.5425%，不能用覆盖掩盖前表面。fit标签读出free也.340482→.432083m，表明不只是开发过拟合。

完整表、配对、恢复边界和下一实验在`docs/WORLDSIM_V7_3_JOINT_R5_RESULTS.md`；证据`m2/global/population_joint_r5_analysis.json`及`population_joint_r5_training.json`。主假设当前不成立于该配置，不能外推为视觉几何路线全面失败。已查[nvdiffrast SIGGRAPH Asia2020官方说明](https://nvlabs.github.io/nvdiffrast/)，将既有r8的有限宽束米制free作为下一机制迁移，保留r10标签对照与r11强原生控制。登记r12 `20260908T050000Z__population-joint-full-track-beam-range-s7304-r12`，同r10只改free为beam_tube_range .03m/32，event仍0，尚未启动。

观测点集任务`WS-V73-M4-OBSERVED-POINTS-01/20260908T043000Z__development-fixed-surfaces-r1` code36b22287完成，CPU16.9374s/RSS.64511GiB，75对象/5日志/11886原束。r5 P/R/F=35.967/36.065/35.412%，r6=50.561/39.916/43.190%；配对F−7.778pp [−11.735,−3.820]，共同有预测33对象的双向距离+.386073m [+.163915,+.696813]。全缺失距离不可定义/有效分母明确保留，不把此观测域指标冒充完整表面GT。PID51413已退出，原样点集/返回mask保存在run，摘要归档m4/observed_points_r1_summary.json。

旧AV2 joint `WS-V73-M4-AV2-FIXED-01/20260908T041500Z__old-development-joint-r5-r1` code36b22287完成，21对象/1旧日志，七相机/28×672²完整联合接口实际执行；0更新，69.6078s/GPU5.43790GiB/RSS3.67498GiB。hit30.3075%、early17.0071%、miss43.9941%、free.968468m、距离.096616m、recall79.5175%；接口可运行但物理未成功，单旧日志无跨日志区间。PID51411已退出，证据coverage/av2_old_joint_r5_analysis.json。

外部20日志前缀`WS-V73-M4-AV2-PREFIX-01/20260908T043000Z__external20-28view-prefix-r1`以code36b22287完整完成，407.9406s/GPU7.64359GiB/RSS10.63879GiB，560视图、42371687680字节缓存（约39.46GiB），6722839条build尺度对应，固定IRLS尺度范围11.0605–55.1629。全部28视图联合聚合，只分块独立patch编码；没有新域优化、heldout质量评分或模型选择。原PID52753任务已结束，摘要归档coverage/av2_external20_prefix_summary.json，原全部日志保留。r11继续训练，r10接下来按原登记启动，当前无等待启动队列。

登记r5场景CPU读出：nuScenes `WS-V73-M4-SCENE-COMPOSITION-01/20260908T044000Z__development-joint-r5-carved-r7`，同既有r3雕刻背景；旧AV2 `WS-V73-M4-AV2-SCENE-01/20260908T044000Z__old-development-joint-r5-carved-r3`，同逐束时刻背景。均读取固定表面，不重新推理网络，尚未执行。F01/F02/F03/F04/F05继续，F06缓解，F07轴修正但物理风险保留，F08恢复完成原因未知，下一编号V73-F09。整个V7.3未完成，shutdown=false。

---


