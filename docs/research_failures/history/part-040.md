# 历史原始记录 040

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F64 — P81 prep r1错误假设blob shard按scene index百位切分

- 分类：`engineering/dataset-archive-routing`；状态：`resolved_r2_active_before_target_read`。
- 症状：r1只在`scene-0016→01`命中389与`scene-0523→05`命中391；错误绑定03/07/08/09均0，最终缺
  3,120/3,900 LIDAR files。0 scene preprocess、0 Actor rows、0 target/model evaluation。
- 卡点调研：nuScenes官方/devkit只要求十个blob archives合并解压，不承诺metadata index分包；读取archives开头真实
  session members后，按采集session冻结`04:{0344,0330,0923,0963}`、`06:{0627,0784}`、
  `10:{1059,1071}`，保留已证实`01:{0016}`、`05:{0523}`。
- r2复用780已提取members，只扫描04/06/10；不改cohort、target、models、gates或claim。下一编号=`V67-F65`。

P87/P88分别以Deep Sets与set attention填充纯IO等待，均在target read前冻结；当前无新增失败，下一编号仍为
`V67-F65`。

P89多阈值ordinal trajectory reliability同样在target read前冻结；当前无新增失败，下一编号仍为`V67-F65`。

P90 plain continuous trajectory max-error Huber同样在target read前冻结，用于检验既有source结果中plain regression
相对复合rank supervision的迁移优势；不修改P81 cohort、target或coverage，不进行loss/threshold sweep。

### V67-F65 — P90 r1直接脚本入口缺少仓库级Python import path

- attempted entry：从repo root直接执行`python scripts/run_worldsim_v67_p90_plain_trajectory_max_error.py ...`；
- symptom：入口导入`motion_proj.worldsim_v67`时抛`ModuleNotFoundError: No module named 'motion_proj'`；
- exposure：发生在argument parse、run创建、source load、epoch与P81/P85 target read之前，故r1不计scientific trial；
- root cause/literature response：Python官方command-line文档规定直接执行文件时把脚本目录而非当前repo root放在
  `sys.path`首位，并说明`PYTHONPATH`用于扩展module search path；与既有V65-F18同类；
- resolution：仅为r2进程设置`PYTHONPATH=.`，不改代码、model、loss、cohort、target、coverage或gate；r2已恢复GPU训练；
- prevention：后续repo-root脚本统一显式进程级`PYTHONPATH=.`，不修改全局shell环境。

下一可用编号：`V67-F66`。

P91固定q=.90 conditional max-error quantile在P81 target read前冻结，作为P90 mean-oriented Huber的单一tail-risk
对照；不扫quantile或coverage，不改变cohort/gates。当前无新增失败，下一编号仍为`V67-F66`。

### V67-F66 — P81 prep r2把部分scene session过早绑定到单个archive

- 分类：`engineering/dataset-archive-routing`；状态：`resolved_r3_active_before_target_read`；
- 症状：r2扫描04/06/10并复用01/05的780 files后仍缺1,175/3,900 LIDAR files；逐scene缺失为
  `0923:388`、`0784:398`、`0963:389`，其余7 scenes完整；0 scene preprocess、0 Actor rows、0 target read；
- root cause：用少量archive header把scene假设为同一候选包，没有查询已有完整member-shard manifest；scene table、
  capture session与十个download blob的分组不存在可推导的一一映射；
- literature/open-source response：nuScenes官方/devkit只要求合并十个blob archives；工程恢复转用项目已有、由完整
  archive扫描生成的V4 test member-shard manifest，而不继续猜scene index或单包session范围；
- exact recovery：manifest给出`n008-2018-08-30-15-31-50-0400→08`以及两条`n015` sessions
  `2018-10-08-15-44-23/2018-09-25-13-17-43→09`；故r3冻结`0784→08`、`0923/0963→09`，只扫描08/09并复用
  已提取2,725 files；
- claim impact：只修复输入路由，不换scene、不改H3.5、model、target、coverage、gate或claim；r2不计scientific trial。

下一可用编号：`V67-F67`。

P92 heteroscedastic Gaussian trajectory failure probability在P81 target read前冻结；不修改P81合同、不扫分布或variance
bound。当前无新增失败，下一编号仍为`V67-F67`。

P93 direct trajectory any-failure BCE同样在P81 target read前冻结；阈值固定为正式1m endpoint，不扫class weight或loss
组合。当前无新增失败，下一编号仍为`V67-F67`。

P94三成员direct-probability deep ensemble在P81 target read前冻结：seed0复用P93，seed1/2同协议独立初始化，最终只取
算术均值且不选member/subset。P90--P92 checkpoint已确认落盘后安全退出等待进程以释放4.5GiB，后续evaluation-only
恢复；无artifact/quality损失，不登记failure。下一编号仍为`V67-F67`。

### V67-F67 — visited Actor最大位移误差不产生稳定的task-conditioned trajectory增益

- 分类：`scientific/target-definition`；状态：`closed_negative_migrated_to_occupancy_flip`；
- independent signal retained：P81全row primary在9,559 rows上query/Actor/P73 selected events=`26/57/45`，相对
  Actor减少54.39%，10/10 scenes nonincreasing，3/3 gates通过；P82/P83 secondary均为0 events；
- faithful visited failure：P84在2,113 visited rows上candidate/P75=`235/208`；P85在1,089 trajectories上
  candidate/P75=`203/199`，均拒绝；
- direct family：P86--P94全部拒绝。最佳P86为query/Actor/P75=`187/193/199`，但query增益仅3.11%<10%，absolute
  reduction 37.48%<50%；P90=`191/201/199`，P94 ensemble=`204/189/199`；
- root cause：原label是Actor endpoint constant-velocity error，只依赖Actor history/future；τ只决定membership。于是
  Actor-only已获得目标的主要可预测部分，query features既不改变label也不能稳定提供预注册增益；all-row成功还可由
  separation>6m的未访问rows贡献，不能外推为visited-state reliability；
- literature/open-source response：ICCV 2021 safety-aware motion prediction将planner critical region表示为earliest
  occupancy，CVPR 2023 IMPLICITO只在candidate trajectory附近的spatiotemporal query points预测occupancy/flow；两者都
  指向“路径上的occupancy结论”而不是路径外的Actor位移误差；
- resolution：关闭max-error visited trajectory family，不降gate、不挑P86/P90、不继续architecture/loss/seed sweep；
  P95将target改为predicted-vs-observed Actor path对同一τ的occupancy decision flip。P81 cohort仅development，剩余10
  test-role scenes继续未读，只有development支持才做one-shot confirmation；
- claim impact：V6.7当前只支持全row event triage，不支持visited Actor reliability、planner/policy/closed-loop/safety。

下一可用编号：`V67-F68`。

P95 occupancy-flip迁移已冻结radius/time samples/width/coverage/model，不增加hash/checksum/fingerprint；当前无新增失败，
prep得到source/development row-level flips=`2,273/96`及false-safe=`925/32`，非空且未触发target恢复；当前无新增失败，
下一编号仍为`V67-F68`。

P95 development以query/Actor/P75 selected flips=`7/28/13`通过4/4 gates；由于P81 cohort已消费，只触发P96 remaining
10-scene one-shot confirmation，不记作独立成功。P96 cohort/shards/model/target/gates均已在sensor read前冻结；当前无新增
失败，下一编号仍为`V67-F68`。

P97只从冻结P95 artifacts派生false-safe target并与P96 archive IO重叠；0 new sensor/target read，且明确禁止替换P96
endpoint或model。

### V67-F68 — one-sided false-safe reliability不优于冻结P75且排序接近反向

- run：`run://worldsim_v67/WS-V67-P97-TRAJECTORY-FALSE-SAFE-01/20260830T005000Z__trajectory-false-safe-s0-r1`；
- symptom：development 1,791 trajectories/31 false-safe中，fixed50 query/Actor/P75 selected=`11/16/10`；query虽相对
  Actor减少31.25%、相对all减少28.99%，但仍比P75多1，query/Actor AUROC仅`.44692/.45555`；2/4 gates；
- root cause：row-level source/development positives只有925/32，trajectory endpoint更少；单独拆出false-safe破坏P95 total
  flip中由较多false-alarm提供的可学习排序，source BCE接近0也未迁移；
- literature response：NeurIPS 2021 rare-event工作指出信息量受positive数量约束且negative subsampling需log-odds校正；
  ICCV 2021 safety-aware occupancy使用专门hard/soft/unseen losses。当前项目没有独立校准cohort，不能事后引入这些
  weight/sampling corrections并把同一development调成成功；
- resolution：关闭standalone false-safe candidate，不加focal/class weight、不改threshold/radius；P98仅补齐互补
  false-alarm attribution，P96保持冻结total-flip endpoint/model；
- claim impact：无false-safe独立可靠度claim，不影响P95 development或尚未读取的P96。

下一可用编号：`V67-F69`。

P98 false-alarm attribution从冻结rows派生且不修改P96；development fixed50 query/Actor/P75=`0/25/3`、AUROC=
`.92312/.67813`，4/4 gates通过。冻结P95的真实selected subtype另为query/Actor/P75=`3+4 / 17+11 / 10+3`
（false-safe+false-alarm），所以P98不能被用来声称P95主要移除false alarms，也不能覆盖V67-F68或产生safety claim；
当前无新增失败，下一编号仍为`V67-F69`。

P99 equal-weight two-head multi-task在development选择8 flips（6 false-safe+2 false-alarm），优于Actor 39/P75 13但未超过
P95的7；它是V67-F68后的唯一shared-representation recovery，不替换已冻结P96 model，也不新增failure。下一编号仍为
`V67-F69`。

P100 temporal-clearance query augmentation从冻结P95 rows解析派生3维、0 new read；development fixed50
query/Actor/P75=`9/41/13`，4/4 gates但未超过P95的7。它是positive mechanism result，不登记failure、不替换P96，
也不扫feature/loss；下一编号仍为`V67-F69`。

P101针对P100压缩时间交互的表示瓶颈，一次性迁移为与occupancy target同构的9-step signed-clearance/boundary-distance
profile；development fixed50 query/Actor/P75=`13/29/13`，4/4 gates但只追平P75且不及P95。它不是正式失败，且不以
profile length/threshold sweep补救。P102只做一次hierarchical temporal-token→Actor-set结构恢复，P96仍只确认冻结P95；
P102得到`4/27/13`并刷新development best。P103 checkpoint/protocol在P96 target前冻结为prospective secondary，
不改变primary。至此无新增scientific failure，下一编号仍为`V67-F69`。

### V67-F69 — P96 scene-0556 session的相邻日期shard推断错误

- 分类：`engineering/archive-routing`；状态：`resolved_pre_target_exact_shard_recovery`；
- symptom：冻结cohort所需`scene-0556` session=`n008-2018-08-31-11-37-23-0400`事前按archive03相邻session推断，
  03完整扫描对390 candidates命中0；发生时仅08已精确命中397，10个processed scenes未ready，P96/P103 target rows不存在；
- root cause：10个nuScenes trainval blob parts不是按scene index或简单session日期连续分桶；相邻archive header不足以外推；
- literature/open-source response：nuScenes官方论坛与开源dataset setup确认10 parts提取后合并为一个dataroot，但公开文档
  不提供session→part索引。恢复因此不继续猜日期：02/04/05/07 exact locators先全部排除；随后对r1虽完整扫描、但其
  candidate filter未包含0556的01/06/08/09/10作第二轮exact-session locator，最终在06精确命中并停止其余workers；
- frozen recovery：只修`0556:03→06`、以exact-session tar补390 files并复用其他3,511 files，prep r2 active；不换scene/cohort/model/
  target/radius/width/coverage/gates，P95 primary与target read仍exact-once；不增加hash/checksum/fingerprint；
- claim impact：纯pre-target I/O failure，不改变P95/P102 development结果，也不产生confirmation evidence。

P96 prep r2最终3,901/3,901 mapped、newly extracted=350、10/10 preprocess，wall=`448.10s`；0556 exact shard=06。
恢复过程中未读取target、未换cohort/model/gate，故`V67-F69`关闭。

下一可用编号：`V67-F70`。

### V67-F70 — time-local-only监督使Actor-only固定覆盖排序优于query

- 分类：`scientific/objective-factorization`；状态：`resolved_by_single_multitask_recovery`；
- run：`run://worldsim_v67/WS-V67-P104-TEMPORAL-FLIP-SUPERVISION-01/20260830T030500Z__temporal-flip-supervision-s0-r1`；
- symptom：development 1,791 trajectories/95 flips上fixed50 query/Actor/P75=`1/0/13`；query absolute reduction
  `97.89%`且AUROC `.90726`，但Actor-only选0，因此query-vs-Actor=`-100%`，只过3/4 gates；
- root cause：5,336 positives/5,180,364 time tokens极稀疏，balanced time-local classification学到Actor motion shortcut；
  固定time max→Actor max又放大单token risk，丢失P102 trajectory-level set objective中的candidate-relative排序约束；
- literature response：CVPR 2023 IMPLICITO在连续spatiotemporal points联合表示occupancy/flow，CVPR 2024 Cam4DOcc对
  多future steps的occupancy与flow使用联合损失，而非用逐时辅助头替换最终forecast objective；
- resolution：不删relative gate、不把1 event包装成功、不扫sampling/aggregation/weight。唯一P105保留P102正式
  trajectory BCE与hierarchical model，将P104逐时flip只作equal-weight auxiliary；失败即关闭local-supervision family；
- claim impact：P104没有独立或task-conditioned success claim，不影响P102 development或P96/P103冻结confirmation。

P105 canonical r2以trajectory BCE为primary、equal-weight time-local BCE为auxiliary，fixed50 query/Actor/P75=
`6/27/13`，absolute/query-vs-Actor reduction=`87.36%/77.78%`，AUROC=`.89704/.64036`，4/4 gates。它解决
P104的relative failure，但未超过P102的4；因此只关闭`V67-F70`，不替换P103，不继续扫auxiliary weight或sampling。

下一可用编号：`V67-F71`。

### V67-F71 — P105 r1使用不存在的torch.flatnonzero入口

- 分类：`engineering/framework-compatibility`；状态：`resolved_before_first_optimizer_step`；
- symptom：P105 r1在首次构造temporal auxiliary indices时抛`AttributeError: torch has no attribute flatnonzero`；
  run目录已创建并完成source tensor装载，但0 optimizer step、0 development/confirmation evaluation；
- literature response：PyTorch官方文档规定`torch.nonzero(input, as_tuple=False)`返回nonzero坐标，`torch.flatten`
  保持元素顺序；恢复先flatten boolean mask，再取nonzero并flatten index tensor，与原意等价；
- resolution：只替换index API并以新run-id r2重启；cohort/data/architecture/loss weight/batch/epochs/selection/gates不变，
  r1保留不覆盖；
- claim impact：纯入口失败，无scientific evidence，不计P105 trial，也不影响P96/P103。

下一可用编号：`V67-F72`。

