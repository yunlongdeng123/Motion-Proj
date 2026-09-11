# 历史原始记录 052

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### P242 milestone note — L1目标对齐恢复连续预算compiler，无新增failure

- canonical：`run://worldsim_v67/WS-V67-P242-L1-INTEGRATED-MONOTONE-BUDGET-SURFACE-01/20260831T053000Z__l1-integrated-monotone-budget-surface-s0-r1`；
- 结果：P201 surface/final MAE=`.007271/.009869`，quality composite通过，budget/horizon violations=`0/0`，3/3；
- 因果边界：相对P241唯一科学变化是MSE改L1；没有width/quadrature/budget/step/seed sweep；
- next：P243全新10-scene/10-log cohort已在read前冻结并启动；P244只在其IO期间训练解析rate-spline successor，
  不读取P243 rows或quality。

下一可用编号仍为：`V67-F187`。

### P246 milestone note — 有限两侧budget扩展通过，无新增failure

- canonical：`run://worldsim_v67/WS-V67-P246-EXTENDED-BUDGET-RATE-SPLINE-01/20260831T070000Z__extended-budget-rate-spline-s0-r1`；
- 结果：P201八budget surface/final MAE=`.006580/.008776`，quality composite通过、双轴violations=`0/0`，3/3；
- 边界：只支持`.025--6.4`有限区间，不推断无界tail或formal calibration；没有range/knot/point-count sweep；
- next：P247在fresh rows前冻结；P248改变对象为reliability-level-conditioned inverse budget，不继续range优化。

下一可用编号仍为：`V67-F187`。

### V67-F187 — inverse budget精度通过但未保持forward probability fidelity

- canonical：`run://worldsim_v67/WS-V67-P248-INVERSE-RELIABILITY-BUDGET-COMPILER-01/20260831T073000Z__inverse-reliability-budget-compiler-s0-r1`；
- 观察：P201 inverse normalized-log-budget MAE=`.019881<=.075`，但在冻结P246重构的probability MAE=
  `.018624>.015`；1/2；source/P183重构MAE也为`.016876/.017074`；level/horizon violations=`0/0`；
- 解释：P201约`15.06%` target在budget下界裁剪，且冻结P246的局部CDF斜率不均；单纯平均budget L1会在平坦区和
  陡峭区等权，因此小坐标误差不能保证response误差；
- literature/open-source response：tandem neural-network inverse design冻结forward surrogate，并以重构response约束inverse；
  NeurIPS 2020 PCGrad在多目标梯度冲突时作投影。P249将冻结P246 cycle probability loss迁入训练，同时保留budget L1；
- response：P249只作一次tandem recovery；cycle gradient冲突时投影并norm-match primary，不扫人工权重；P248结构、
  levels/data/steps/seed/decisions不变；
- 防重复：不放宽`.015`、不增加bisection steps或level/knot/width sweep，不把inverse-budget pass包装为整体成功。

下一可用编号：`V67-F188`。

### P250 freeze note — inverse compiler prospective secondary，无新增failure

- P250在P243 rows和P249 outcome出现前冻结，仅等待两个atomic artifacts；
- 复用P243同一次fresh read，P249/P246/levels/MC/两门均不变，不把它表述为第二个独立cohort；
- 不因P249 development结果改变evaluator或gate；当前下一可用failure id保持`V67-F188`。

下一可用编号仍为：`V67-F188`。

### V67-F188 — PCGrad tandem混合目标只带来微小response改善

- canonical：`run://worldsim_v67/WS-V67-P249-TANDEM-CYCLE-INVERSE-BUDGET-01/20260831T080000Z__tandem-cycle-inverse-budget-s0-r1`；
- 观察：P201 inverse-budget MAE=`.019667`通过，但重构probability MAE=`.018257>.015`；相对P248仅改善
  `.000367`；source/P183重构=`.016490/.016759`；level/horizon violations=`0/0`；
- diagnosis：12,000 steps只有155次budget/cycle gradient conflict；PCGrad很少触发，norm-matched budget primary仍让模型
  主要拟合坐标误差，不能充分强调冻结forward的陡峭response区；
- literature/open-source response：经典tandem inverse design及其PyTorch实现把冻结forward surrogate直接作为唯一可微
  response loss，以避免非唯一或坐标loss牵制；
- response：P251从头训练，唯一loss为冻结P246重构probability L1；P248/P249结构、data、levels、steps、seed与两门不变；
- 防重复：不继续扫PCGrad、手工权重或budget/cycle组合；P251失败则关闭这一inverse student family。

下一可用编号：`V67-F189`。

### V67-F189 — direct tandem response-only训练仍停在相同误差带

- canonical：`run://worldsim_v67/WS-V67-P251-DIRECT-TANDEM-INVERSE-BUDGET-01/20260831T083000Z__direct-tandem-inverse-budget-s0-r1`；
- 观察：P201 inverse-budget MAE=`.020305`通过，但冻结P246重构probability MAE=`.018065>.015`；source/P183为
  `.016283/.016624`；结构违规仍为`0/0`；
- 解释：相对P249只改善`.000192`，且P248 budget-only、P249 mixed、P251 response-only三种语义都收敛在约`.018`；
  这不再支持继续扫描loss或优化器，瓶颈更像当前单一amortized inverse representation；
- literature response：marginal-utility planning把资源决策表达为增加一单位资源的value，不要求显式神经inverse；
  单调网络则允许从已冻结的monotone forward value取得结构一致的非负导数；
- response：关闭inverse student family；P252改为蒸馏冻结P246解析`dP/dlog-budget`的非负elasticity head；
- 防重复：不再试inverse width/knot/level/step/loss/weight/optimizer或warm-start；P250仅完成此前冻结的same-read报告。

下一可用编号：`V67-F190`。

### P252/P253 marginal-value milestone note — 新对象通过且fresh secondary冻结，无新增failure

- P252 canonical=`run://worldsim_v67/WS-V67-P252-MARGINAL-RELIABILITY-ELASTICITY-01/20260831T090000Z__marginal-reliability-elasticity-s0-r1`；
- P201 elasticity MAE=`.019285`、mean within-query Spearman=`.925247`，2/2，非负违规0；
- P253在P243 rows前冻结为same-read secondary，P252/P246/budgets/MC/两门不变；
- P254只推进冻结surrogate下的shadow-price budget policy，不重开inverse student family；下一failure id保持F190。

下一可用编号仍为：`V67-F190`。

### V67-F190 — P243 scene-shard推定漏掉同log跨archive LIDAR members

- run：`run://worldsim_v67/WS-V67-P243-CONTINUOUS-BUDGET-FRESH-CONFIRMATION-PREP-01/20260831T060000Z__continuous-budget-fresh-prep-s0-r1`；
- symptom：八个推定shards全部结束并命中3,518/3,914 required LIDAR，但缺396个同一`n015-2018-11-14-19-09-14`
  log members；prep在preprocess/row materialization/quality前退出；
- root cause：archive part不是scene-exclusive；scene编号映射可定位大多数文件，但一个log的sensor members可跨part；
- literature/open-source response：nuScenes官方devkit setup明确要求下载并在同一root合并全部archives且不要覆盖公共目录；
- resolution：cohort、目标members和已提取文件不变；只并行扫描此前未触碰的01/06补精确missing set。初始recovery入口
  仍会先重扫09，发现后立即取消且0新增output；r3以`recovery_only`跳过primary scan；
- claim impact：0 target/quality read，不换scene、不改P242/P244/P246/P249/P252/P254或任何decision。

下一可用编号：`V67-F191`。

### V67-F191 — P256 evaluation重复附加已有group-member轴

- run：`run://worldsim_v67/WS-V67-P256-GROUP-BUDGET-DUAL-COMPILER-01/20260831T094500Z__group-budget-dual-compiler-s0-r1`；
- symptom：12k训练完成且末段train price MAE约`.007`，首次source evaluation的`_reward`将`G×F×S` budget shape
  再拼接`S`，请求把`G×1×S×36` broadcast到`G×F×S×S×36`而抛ValueError；P183/P201 metric=0；
- root cause：helper按原先`G×F` dual-price输入书写，但调用方已先通过P254展开成per-member budgets；
- literature/open-source response：NumPy官方`broadcast_to`要求原shape与目标shape按广播规则兼容；已有member轴应直接
  保留，不能再次添加；
- resolution：feature target改为`budget_z.shape+(36,)`，budget直接flatten；r2从头训练，data/group/fraction/
  bisection/model/seed/steps/decisions全部不变；
- residual/recovery：r2的return reshape仍把`S`附加一次，在同一quality前位置退出；继续归入F191。r3把结果直接
  reshape回既有`G×F×S`再沿S求均值；P201 fraction MAE=`.016420`、regret=`.00001163`，2/2，F191关闭；
- prevention：不为单一shape错误增加smoke/regression matrix，仅在正式入口修复调用合同。

下一可用编号：`V67-F192`。

### P256 milestone note — fixed-group dual compiler恢复并通过，无新增failure

- canonical r3=`run://worldsim_v67/WS-V67-P256-GROUP-BUDGET-DUAL-COMPILER-01/20260831T103000Z__group-budget-dual-compiler-s0-r3`；
- P201 26 groups的attained fraction MAE=`.016420`、frozen Lagrangian regret=`.00001163`、price violations=0，2/2；
- F191两次入口均在任何P201 metric前退出，r3科学合同与r1完全相同；
- next：P257改变utility为单次冻结的log reliability，不扫P256 group/architecture；下一failure id保持F192。

下一可用编号仍为：`V67-F192`。

### P258 milestone note — log-utility fixed-group dual compiler通过，无新增failure

- canonical=`run://worldsim_v67/WS-V67-P258-LOG-UTILITY-GROUP-DUAL-01/20260831T110000Z__log-utility-group-dual-s0-r1`；
- P201 fraction MAE=`.015445`、frozen log-Lagrangian regret=`-.00000421`、price violations=0，2/2；
- P256与P258说明linear/log两种冻结surrogate utility均能被fixed-group共享dual price摊销；不外推真实scheduler；
- next：P259连续条件化alpha，不再分别训练更多离散risk utilities；下一failure id保持F192。

下一可用编号仍为：`V67-F192`。

### V67-F192 — P243冻结scene-1011的396帧不在本机全部十个官方trainval blobs中

- runs：r1已扫推定02/03/04/05/07/08/09/10；r3=`run://worldsim_v67/WS-V67-P243-CONTINUOUS-BUDGET-FRESH-CONFIRMATION-PREP-01/20260831T101500Z__continuous-budget-fresh-prep-s0-r3`补扫01/06；
- symptom：最后01/06分别约`950.7/900.2s`且均0命中；缺失集合仍是396个
  `n015-2018-11-14-19-09-14` LIDAR，元数据映射表明全部只属于`scene-1011`；
- external evidence：nuScenes官方devkit要求下载全部archives并合并到同一root；本机已覆盖10/10 trainval blob parts，
  因而不能再靠猜测shard或重复扫描恢复；不增加checksum/hash/fingerprint；
- scientific impact：prep在preprocess/row/quality前退出，P243及所有same-read secondary仍未读新target；
- response：保留其余九个冻结unused val scenes，不用任何历史验证scene补位；仅移除不可用`scene-1011`且不替换，
  cohort边界改为9 scenes/9 logs、location=`5/2/1/1`。P242/P244/P246/P249/P252/P254 artifacts、budgets、MC、
  metrics与gates均不变；r4 prep与r2 confirmation接替；
- prevention：本地archive不可用作为明确resource exception记录，不做分卷重扫、完整数据审计或冗长回归。

下一可用编号：`V67-F193`。

### V67-F193 — P249 inverse compiler在P243 fresh same-read上仍未过response重构门

- canonical：`run://worldsim_v67/WS-V67-P250-INVERSE-BUDGET-SAME-READ-CONFIRMATION-01/20260831T121200Z__inverse-budget-same-read-confirmation-s0-r2`；
- observation：9-scene/1,710-trajectory same-read secondary上inverse normalized log-budget MAE=`.016419≤.075`，
  但冻结P246 reconstructed probability MAE=`.015582>.015`；level/horizon monotonic violations=`0/0`；
- context：development P249为`.018257`，fresh误差更低但仍未跨过事前冻结response门；lower-censored target占
  `20.62%`，与F187--F189的陡峭/截断响应解释一致；
- response：维持inverse student family关闭；不因接近阈值而放宽gate、换scene、增加loss/width/knots或再次训练；
  同一次read的P243/P245/P247/P253/P255均按原冻结合同报告；
- claim impact：只拒绝clipped reliability level到minimum budget的amortized inverse response fidelity；不否定
  P243/P245/P247 continuous forward surfaces、P253 marginal ranking或P255 shadow-price policy。

下一可用编号：`V67-F194`。

### P259/P260 milestone note — alpha-fair连续risk条件化推进，无新增failure

- P259 canonical=`run://worldsim_v67/WS-V67-P259-ALPHA-FAIR-SHADOW-PRICE-POLICY-01/20260831T111500Z__alpha-fair-shadow-price-policy-s0-r1`；
- P201 budget MAE=`.011076`、frozen alpha-fair regret=`.0001333`、price violations=0，2/2；
- P260 P201 fraction MAE=`.017168`、frozen alpha-fair Lagrangian regret=`.00001031`、violations=0，2/2；
- P261只把冻结P259推进到train sizes 32/64/128、heldout 48/96的permutation-invariant set compiler，
  alpha/fraction/steps/两门一次锁定，不扫attention或pooling family；
- next failure id保持F193。

### P261/P262 milestone note — variable-set通过并进入task-horizon preference，无新增failure

- P261 canonical=`run://worldsim_v67/WS-V67-P261-VARIABLE-SET-ALPHA-FAIR-DUAL-01/20260831T120000Z__variable-set-alpha-fair-dual-s0-r1`；
- P201 sizes 48/96 aggregate fraction MAE=`.014662`、regret=`-.000000447`、violations=0，2/2；
- P262增加连续horizon preference这一新任务条件，不回头扫P261 pooling/attention/width；五train与四heldout
  preference、alpha交叉一次冻结；
- P262 P201 budget MAE=`.012221`、task-alpha utility regret=`.0002099`、violations=0，2/2；P263仅推进其
  fixed-group shared dual，不修改preference scalarization；
- next failure id保持F194。

