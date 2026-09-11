# 历史原始记录 012

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V7.3 环视原生结果与跨Actor共享训练（2026-09-08）

M1六相机r3已完成（code cf039715，run `20260907T161500Z__native-dpt-surround25-dev6-s7301-r3`）：25fit/6dev场景，每窗口4时刻×6相机=24views，60epochs/1500更新，2287.82s，原生DPT 32,654,562参数。project最大变化0.004401，前缀峰值约5.510GiB，训练峰值1.350GiB，RSS9.853GiB。

独立日志等权Actor轴向MAE：fit 20日志 3.926→0.771m；development 5日志 3.493→3.321m，配对变化-0.173m，bootstrap95%区间[-1.595,1.664]m，4/5日志改善。全部仍为build内保留点插值诊断，不能代替未输入时刻表面/首交点。scene0519的轴向early=0.801，原生深度优化不保证物理表面改善。F05继续active。

原始结果与日志配对表：`docs/autoresearch/worldsim_v73/m1/r3_summary.json`、`r3_log_analysis.json`。环视前后覆盖不同，不把r2/r3的聚合数字直接当作同样本收益。

下一任务 `WS-V73-M2-GLOBAL-ACTOR-01`，run `20260907T165500Z__surround-shared-native-s7304-r1`：20fit日志/5dev日志各选一个Actor，按build/轨迹信息优先速度>2m/s，再选日志内build支持数中位数；不是按预测质量或heldout误差挑选。无可用输入的选中对象保留在cohort并注明，不静默剔除。现有dev的运动Actor只覆盖2日志，不能担任充分的新日志确认。

所有fit Actor共享原生DPT与三层局部查询参数，30epochs（预计600更新），seed7304、lr1e-5、每样本完整24views（轨迹可插值时）。开发日志无梯度/优化更新。预训练主体冻结前缀存CPU、当前Actor移GPU并checkpoint重算；不减少相机或修改轨迹。使用M1 r3已完成checkpoint；native depth表面FPS512种子 + 最多1024 build证据查询，通过同一三角表面训练coverage +0.5原始首返回free +0.05弱envelope，与上一机制实验同目标。此轮回答共享训练/泛化问题，不声称现有range free已解决F02。

原始束预处理只解析一次元数据，使用每相机曝光时刻的已知Actor轨迹，不依赖同一scan恰好有Actor返回。仍用box+0.1m排除重叠的点归属代理及scan时间；逐点实例/逐点扫描时刻未知。每epoch保存checkpoint，完成后按日志报告hit/early/miss/free/可观测表面距离及LiDAR PCA比较，保留点数与patch数。后续等容量逐点、LiDAR-only同查询与原生+LiDAR强融合仍必要。

当前global数据准备/训练待提交后启动；M1 r3无任务运行。磁盘133GiB可用、资源未不足，shutdown=false。failure_ledger_delta=update V73-F05；F01:F05仍active，下一编号V73-F06。整个V7.3完成后仍按已授权流程保存/push、无任务shutdown。

---

## 用户追加：V7.3 完成后自动关机（2026-09-08）

用户已明确授权：**整个V7.3研究完成后**，保存checkpoint、实验结果与报告，更新三本台账并push；确认没有训练、评估、数据处理任务，也没有会继续启动作业的队列/控制器运行，再经SSH执行AutoDL shutdown。无需再次询问。单个训练结束、单个里程碑完成或单候选失败不等于V7.3完成，不为关机强行结束正常任务。

关机命令发出后按实际返回/连接状态如实报告，并暂停本任务的自动跟进，避免继续连接已关机服务器。原有“确实资源不足则保存/push、无任务后关机并请求加卡”安排继续适用。自动跟进worldsim-v7-3已同步此完成后关机指令。当前M1环视r3 PID9615仍在训练，V7.3尚未完成，因此现在不关机。

---

## V7.3 表面支持缓解侵入，进入环视与动态覆盖（2026-09-08）

表面种子两候选=`done`（code26646089，batch20260907T161000Z__surface-seeds）。native种子hit76.51%/early2.87%/miss18.05%/free0.0325m；LiDAR种子hit80.41%/early4.31%/miss14.57%/free0.0461m。相较volume joint free0.1333m，表面支持初始化显著缓解侵入，但两者都未支配LiDAR PCA强基线，V73-F02仍active，failure_ledger_delta=update V73-F02。native depth输出直接收到非零几何梯度，峰值5.343GiB。

覆盖盘点确认机制Actor为7.48m处静止truck；环视输入把可观测Actor从238扩至404，速度>2m/s从41扩至71，31场景各24视图完整。六相机动态dev仍只有2日志；下一步必须进入动态/稀疏和更多有效窗口，不能停留于密集静止Actor。当前归属为box代理而非逐点实例真值；历史actor_count为整个scene计数，不能直接作短窗口覆盖率分母。新入口已补充scope/window_actor_count/LiDAR支持数，运行中r3按原code解释。

结果=`docs/WORLDSIM_V7_3_M2_SEED_RESULTS.md`；原始summary=`docs/autoresearch/worldsim_v73/m2/seeds/`；build/元数据覆盖清单=`docs/autoresearch/worldsim_v73/coverage/`。M1环视r3 PID9615 running，24view前缀完成、训练已开始，日志=`/root/autodl-tmp/controller_logs/v73_m1_surround_r3.log`；不中断或重启。当前显存/内存/磁盘足够，shutdown=false。

下一任务：共享跨Actor查询训练及同信息控制，按build观测/只读轨迹组织运动和稀疏队列；同步检验局部free几何方向和连贯性。已先查DRC CVPR2017与nvdiffrast可见性梯度，range深度罚与退出射线管的几何方向可能不同，尚属待测机制。event与场景拼接仍pending，四风险及F05未解除；下一编号V73-F06。无需用户确认，自动研究继续。

---

## V7.3 扩大环视原生几何训练（2026-09-08）

M2表面种子批次PID9230 running（native先行、lidar随后），native depth输出已获得非零梯度，12view当前峰值5.343GiB；完整曲面结果尚未完成。

同时将M1强基线扩大到六个环视相机，避免把早期三前向相机配置当作研究上限。配置=`configs/worldsim_v73/m1_native_geometry_surround.yaml`，run=`20260907T161500Z__native-dpt-surround25-dev6-s7301-r3`，25fit/6dev、4时刻×6相机=24视图、378×672、60epochs/1500更新、seed7301，从预训练DPT初始化。冻结aggregator重新联合提取24视图的多层前缀；不能复用旧12view最终上下文。现有dataset含六相机payload目录，各场景实际可用窗口/Actor观测数由cohort完整报告，不凭目录存在宣称覆盖足够。

已知资源90GiB cgroup、3090 24GiB、磁盘约160GiB可用；M2自身峰值5.343GiB，原生逐视图训练峰值约1.35GiB，24view前缀峰值由r3实测。若并发引发压力，优先串行调度完整窗口，不删相机来适配显存；只有完整任务在合理执行优化后仍不足才保存/push并无任务shutdown。当前未出现资源不足，不停机。

该扩展针对F05视觉覆盖风险，不承担独立确认；保持相同fit/dev身份且不读source/external。旧M1结果只说明前向12view配置，不能替代环视基线。每个里程碑同步三本台账与push，禁止新增哈希/门控/重复smoke。

---

## V7.3 原生depth直接生成表面查询种子（2026-09-08）

`WS-V73-M2-PHYSICAL-SURFACE-01` 表面种子两候选已实现，待提交后启动：`20260907T161000Z__surface-seeds-native-r1`、`20260907T161000Z__surface-seeds-lidar-r1`，均joint/120steps/seed7303/free权重0.5，复用r1原始束缓存与M1 r2冻结前缀/已微调head。原生候选从当前可训练DPT depth输出反投影至Actor坐标，按只读框归属选择候选，用FPS选512个种子；选点离散但保留位置至depth的梯度，另外继续读取四级refinenet特征。无native支持时回退LiDAR并记录逐视图支持数。

LiDAR种子候选同样512 FPS。两者初始仅加5cm范围的可学习抖动，后续仍为无小位移硬上限的三层位置更新；没有删除completion表面、学习置信度/透明度或缩小patch半径。除种子和原生depth直接路径外保持原训练设计；将记录native depth输出梯度、支持数、真实曲面/free结果和资源。种子仅用build图像/点、标定、轨迹，未读heldout target做选择。

前一完整控制组已证实volume completion阻挡远背景为主要卡点；本轮迁移On-Surface Prior/AdaPoinTr的表面支持思想，旨在检验初始化而非宣称新理论。V73-F02/F03保持active，failure_ledger_delta=none，下一编号V73-F06；暂不加入event或新正则。当前资源充足、shutdown=false。

---

## V7.3 控制组完成：过剩补全曲面是主要卡点（2026-09-08）

控制批次20260907T154500Z__physical-controls=`done`，native融合与4组120steps均完成，无运行研究进程。joint-r2 hit75.43%/early5.24%/miss16.51%/free0.133m，pointwise hit74.60%/free0.150m，LiDAR-only hit72.76%/free0.106m，joint-no-free hit76.16%/free0.090m。空间交互和视觉的净优势均未成立；不把free变差草率外推监督无用。

已按面来源定位：joint-r2 heldout累计侵入的71.58%来自completion面遮挡原始非Actor返回（952.42/1330.58m），build亦同。少量不受覆盖监督的体积内曲面挡住远背景是主要卡点，LiDAR-only同样存在。failure_ledger_delta=update V73-F02，下一编号V73-F06。结果和来源表=`docs/WORLDSIM_V7_3_M2_CONTROL_RESULTS.md`、`docs/autoresearch/worldsim_v73/m2/controls/`。

已先查On-Surface Prior CVPR2022及AdaPoinTr作者代码；下一轮用可训练原生DPT点图的规范表面种子初始化completion，增加LiDAR表面种子控制，保持现有监督与容量。原生depth输出接入几何位置梯度，仍允许查询生成新支持，不学习opacity/存在概率/半径避罚。当前问题是方法效果，资源充足，shutdown=false。实现和下一训练继续推进。

---

## V7.3 首轮真实曲面负结果与后续比较（2026-09-07）

M2真实曲面r1=`done`（code45737d33、run20260907T154000Z__joint-physical-scene0100-s7303-r1）；native融合=`done`（code512048bb、run20260907T154500Z__native-fusion-scene0100-r1）。统一三角读出结果见`docs/WORLDSIM_V7_3_M2_PHYSICAL_RESULTS.md`及`docs/autoresearch/worldsim_v73/m2/physical_r1_comparison.json`。

单fit Actor两个未输入时刻，joint约69.62% hit/5.42% early/22.21% miss，对照LiDAR约69.58% hit/0.70% early/29.34% miss；已知原始束free平均侵入约0.078m，对照约0.0066m。native微调改善原生表面读出，但native-only未并入原始LiDAR表面，尚不能当最强同信息融合基线。joint未超过LiDAR强基线；正观测点均距约2.1cm不能掩盖提前相交。

V73-F02更新为已有实测负证据，其他风险保持active；failure_ledger_delta=update V73-F02，下一编号V73-F06。已先检索AdaPoinTr官方自适应查询与LaS-Comp CVPR2026的观测保持/边界一致性，候选迁移详见报告。不能把当前单侧coverage下的过剩表面外推成路线B失败。

同信息控制批次PID6922正在继续：`/root/autodl-tmp/runs/worldsim_v73/WS-V73-M2-CONTROL-BATCH-01/20260907T154500Z__physical-controls/status.json`；native-fusion已done，joint-r2 running，随后pointwise-r1/lidar-only-r1/joint-no-free-r1。批次控制器=`/root/autodl-tmp/codex_tmp/control_v73_batch.py`；不中断/重复现有任务。待成组结果后再调整支持初始化或几何监督，不同时堆叠正则。当前卡为方法效果，不是资源；shutdown=false。

---

## V7.3 同信息控制与原生表面比较注册（2026-09-07）

`WS-V73-M2-PHYSICAL-SURFACE-01` r1当前running，PID6424，code=`45737d33`，120steps已过半；输入包含4build时刻10222个原始归属点（最终unique数以run为准），2个heldout时刻，所有近框原始束共约3.2万。训练峰值5.306GiB，原生project与法向梯度非零；完整结果待落盘，不提前判胜。

接入控制组：joint、pointwise（同message/update容量、只读自身查询）、lidar_only（同query解码器关闭视觉读出），以及joint无free。全组统一seed7303，并在原生头构造后重置query初始化种子，因此它们彼此共享初值；早期r1没有这次重置，只作先导，不能与新组冒称相同初值。各120steps、lr1e-5、相同build点/束抽样量和显式patch尺度，先在同一fit Actor作机制比较。M1的冻结/已微调原生点图也用相同1536 PCA曲面片读出，5cm规范体素平均，记录每视图入Actor框点数，不能按heldout质量选点。

计划批次=`20260907T154500Z__physical-controls`：joint-r2、pointwise-r1、lidar-only-r1、joint-no-free-r1；原生融合run=`20260907T154500Z__native-fusion-scene0100-r1`。只在build范围内indices2,5诊断，尚非独立日志确认；同时报告真实首交点hit/early/miss、已知free侵入、正观测点到同曲面距离/recall，未知区域不被当成负表面。原生点图按已知box归属存在背景污染风险，单独披露。

当前四风险及F05继续active，failure_ledger_delta=none；下一编号V73-F06。

---

## V7.3 M1 扩展结果与真实曲面实验（2026-09-07）

M1 r2=`done`，code=`424743fc`，run=`20260907T151000Z__native-dpt-fit25-dev6-s7301-r2`。25fit/6dev场景、60epochs/1500更新，1130.19s。原生DPT 32,654,562参数，project最大变化0.004067；训练峰值1.348GiB、RSS8.817GiB。

按独立日志先平均场景MAE再等权汇总：fit 20日志 3.669→0.849m；development 5日志 2.870→2.485m，配对变化-0.384m，日志bootstrap95%区间[-0.889,0.395]m，4/5日志改善。6dev场景中5改善、scene1089 1.176→2.318m退化；scene0519 MAE改善而轴向early比例0.258→0.598。scene0994零Actor诊断点保留且不进入MAE均值；不能静默删除。

仍只是build点插值诊断，非新时刻表面或字面首交点。区间跨零、支持量不均、部分early变差，F05继续active；不把这次结果写成跨日志方法胜利。详表=`docs/autoresearch/worldsim_v73/m1/r2_summary.json`与`r2_log_analysis.json`。

`WS-V73-M2-PHYSICAL-SURFACE-01` 已实现待启动，run=`20260907T154000Z__joint-physical-scene0100-s7303-r1`，120steps，seed7303，native=M1 r2。用同一三角曲面最近点coverage +0.5原始首回波前free +0.05弱envelope；build随机1024点/512原始束每步，保留12视图1536查询。所有build原始Actor点可训练，本run改以build范围内未输入时刻（indices2,5）诊断；目标点只进入损失，不用于推理查询出生。提供同密度1536个LiDAR PCA曲面片比较；完整同信息原生融合、等容量逐点、LiDAR-only训练控制仍待接入。

硬相交先无梯度确定首三角面，再重算所选相交位置梯度；最近曲面查询按最优重心点反传，避免保留全体pair计算图。当前分块全扫描仍为O(PF)/O(RF)计算，不宣称BVH加速。解析例验证了前后表面归属、miss、free梯度方向与面内/边缘最近点。资料依据：[Open3D射线](https://open3d.org/html/tutorial/geometry/ray_casting.html)、[最近曲面](https://open3d.org/docs/latest/tutorial/geometry/distance_queries.html)、[nvdiffrast](https://nvlabs.github.io/nvdiffrast/)；后者明确区分支持内位置与轮廓可见性梯度。没有安装新环境依赖，也没有添加哈希/校验和/指纹。

当前failure_ledger_delta=none，沿用F02/F03/F05解释尚未解决的表面/自由空间/泛化问题。最近曲面coverage可牵引缺失支持，但不自动保证完整拓扑或远距离未知区补全；硬free可经移开表面规避，因此必须与coverage联合并报告miss。下一编号V73-F06。

---

## V7.3 M2 联合通路实测完成（2026-09-07）

`WS-V73-M2-JOINT-GEOMETRY-PATH-01` r1=`done`，run=`20260907T151700Z__joint-dpt-query-s7302-r1`，code=`c0e23d28`，seed7302。12视图、1536查询、13824顶点/12288三角面，8步联合训练17.61s；峰值5.358GiB、RSS2.016GiB。原生DPT project梯度每步非零，参数最大变化7.758e-5；build center coverage从0.07640降至0.04688m。详表=`docs/autoresearch/worldsim_v73/m2/r1_summary.json`。

本结果只确认几何位置→局部视觉读取→预训练DPT的有效联合通路和当前配置资源；未训练patch法向/弯曲、无free/event、无held-out曲面比较，不能声明补全/主方法成功。`visual_observed_fraction=0.985`仅指投影在图内，尚非遮挡可见性。cKDTree避免全体两两建图，checkpoint保留12视图；不据此外推更多视图或上层LoRA显存。

V73-F01在当前12view配置下得到资源缓解；F02/F03/F04/F05仍active。failure_ledger_delta=none，下一编号V73-F06。原生DPT可训练的结论来自真实梯度/参数变化，不来自输出命名。

---

## V7.3 M2 联合通路注册补记（2026-09-07）

`WS-V73-M2-JOINT-GEOMETRY-PATH-01` 将直接检测原生DPT→多尺度投影局部采样→查询几何的有效梯度和激活成本，迁移依据为Deformable DETR、AdaPoinTr与PyTorch non-reentrant checkpoint官方文档。尚未产出模型比较，failure_ledger_delta=none；不解除V73-F01:F05。稀疏coverage只用于本次center通路，不能解释成完整曲面片/未知区补全已学习；下一编号V73-F06。

---

## V73-F05 — 原生 DPT 小样本可拟合但开发支持与泛化不足（2026-09-07）

- 最终独立确认补充（2026-09-09 21:30 UTC）：最终固定AV2跨域确认失败：Joint r7相对匹配LiDAR r6的hit−2.9419pp、early+5.3637pp、free+.325997m、target→surface距离+.013259m、recall−2.4204pp，五项95%日志配对区间均排除0且方向更差；free在20/20日志变差。miss−.9625pp的区间[−1.9366,+.0576]pp跨0，不构成可靠改善或等效。开发5日志hit+10.5363pp等正结果仍成立，但未通过20个独立AV2日志的跨数据集确认。本轮V7.3联合方法的可泛化物理表面主张实验失败/未获支持，不外推所有视觉基座无用。 已证实的失败表现是跨域联合增量反转及early/free冲突；开发取证表明规则局部片也会错误延伸并遮挡后方正确支持，严重sliver并非这些early面的解释。域差异、视图/标定嵌入与米制对齐残差、局部视觉对应污染、支持域外扩是可能机制，未做因果隔离，不能写成已定位唯一根因。Joint同时改变native种子、可训练DPT特征及build-depth辅助项，故不是纯视觉特征消融。无OOM或资源阻断，不能用算力不足解释本次负结果。 证据docs/autoresearch/worldsim_v73/final_confirmation/{analysis,models}.json及配对图；run WS-V73-FINAL-CONFIRMATION-01/20260909T135000Z__fixed-r7-r6-external20-r1，执行code6499d34d，全部五方法936对象/20日志/0优化更新。878 ready、58缺输入、119无owned heldout对象均保留；含221移动>2m/s、108运动未知、556 build<100对象。模型/尺度/配置固定在读取外部质量之前，未按AV2再训练、调参或挑场景。

category=`scientific/generalization_and_support`；status=`active`；task=`WS-V73-M1-NATIVE-GEOMETRY-ADAPT-01`；code=`29595e20`；run=`20260907T150300Z__native-dpt-fit4-dev2-s7301-r1`。

观察：原生32.65M参数DPT真实更新，4fit场景Actor诊断MAE全部下降；scene0048开发MAE3.594→2.867m，scene0359仅11点且5.043→5.095m。train早于target0.2m比例增加，不能以深度MAE替代free/硬表面收益。两dev样本不支持泛化结论，也不否定视觉几何适配。

先查CAPA官方稀疏几何TTA与CVPR2024 TTA depth completion，再迁移为全部25fit/6dev日志训练、记录实际Actor支持，必要时比较build-only局部适配/显式传感器融合。当前首选增加有效日志数据，保持同一损失/lr，不把结构和优化同时更改。解除条件为充分观测下稳定的开发几何/硬读出收益；不能因训练loss下降解除。r2=`20260907T151000Z__native-dpt-fit25-dev6-s7301-r2`；报告=`docs/WORLDSIM_V7_3_M1_NATIVE_RESULTS.md`。

V73-F01 outcome：12视图冻结前缀4.578GiB/DPT训练1.348GiB，当前native-DPT候选无资源阻塞；上层LoRA/空间查询尚未测，风险仅局部缓解。V73-F02:F04仍active。下一V73编号=`V73-F06`。

---

## V7.3 M1 启动前风险复核（2026-09-07）

`WS-V73-M1-NATIVE-GEOMETRY-ADAPT-01` 的原生 DPT 入口与12视图数据路径已实现并通过语法编译，尚无新模型结果；`failure_ledger_delta=none`，V73-F01:F04仍active。保留逐点LiDAR采集时间缺失、稀疏像素遮挡和box归属近似的边界；不把build点插值误差当真实首交点指标。配置引用四风险及V71-F54/F66/F68/F69，下一V73编号仍F05。

M0首次push遇到本次开机后旧LocalTUN远端端口消失（connection refused）；复用既有V7-F05网络恢复，重建当前LocalTUN并以新端口40923单命令代理push成功。无数据/模型影响，不新增科学failure；M0交付commit=`dbe98c91`。

---

## V7.3 当前风险总览（修订 2，2026-09-07）

`WS-V73-M0-RISK-STORAGE-02` 完成磁盘清理与方法修订，没有新增算法失败。四项 active 风险见下表，方法已缓解但实证待 M1–M4；下一 V7.3 风险/失败编号=`V73-F05`。既有 V71 编号不重写。

| ID | 分类/状态 | 具体机制与下一判别 |
|---|---|---|
| V73-F01 | resource/architecture；active | 局部多尺度采样和分块；实测原生 DPT/上层适配内存，保留完整输入 |
| V73-F02 | objective/support；active | coverage + 几何 free；禁止 opacity/existence/半径塌缩逃避 |
| V73-F03 | optimization/support_birth；active | 几何吸引持续生支持，event 仅修顺序；推理只读 build |
| V73-F04 | composition/ownership；active | 静态去动态归属，遮挡后 UNKNOWN，统一硬交点和 owner |

### V73-F01 — 目标时空配置的激活与局部查询成本

观察：目前单卡 RTX3090 24GB；cgroup 内存90GiB、CPU14核，旧文首宿主755GB不能当训练预算。尚未有 V7.3 OOM。来源/迁移：VGGT 官方多层 DPT 与分块、Deformable DETR 稀疏采样；禁止全图 dense query attention，kNN 建图另报成本。最小辨别：真实 DPT 更新的峰值与耗时，再测必要 LoRA；证据=`docs/WORLDSIM_V7_3_RESEARCH_PLAN.md` 13.1，base=`63626e8d`。不因3090存在就提前认定失败或退回路线A。

