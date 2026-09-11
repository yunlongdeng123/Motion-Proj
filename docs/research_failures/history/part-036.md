# 历史原始记录 036

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F05 — free trajectory residual学习source-scene action shortcut

- 分类：`algorithm/domain-generalization`；状态：`active_single_recovery_frozen`。
- 观察：P15 train Spearman/pairwise/selected reduction=`0.8633/1/0.5015`，selection仅=`0.2102/0.5866/0.1099`；
  unsafe AUROC=`0.6464`，显著低于qmean `0.9727`；0/6 gates。
- 根因：8-D free MLP residual可覆盖qmean base并编码source cohort的action/context shortcut；不是trajectory prediction
  object本身失败。
- 检索/恢复：CVPR 2026 ResAD用deterministic reference上的normalized residual抑制spurious correlation；UAI 2025
  constrained monotonic calibration强调保留base ranking。唯一P15R只训练12 action biases，case-centered，score residual
  bounded `±0.02`；qmean dominant，data/lattice/gates不变。
- 防重复：不扫residual bound/action lattice/loss/gates，不做第二P15R；下一编号=`V67-F06`。

P15R exact-once：bounded adapter使selection Spearman/AUROC/pairwise均略高于qmean，但selected reduction仅
`0.170481`（qmean=`0.163836`，delta=`+0.006645`），未达`0.25/+0.05`两门；4/6 gates，`V67-F05`更新为
`closed_negative_after_single_recovery`。禁止继续在P10X调adapter。P16把P10V/P10X降格为two-domain development，
模型冻结后才做P9 fresh action-task confirmation；下一编号仍为`V67-F06`。

### V67-F06 — P16 domain variance变量未在adapter作用域绑定

- 分类：`implementation/training-entry`；状态：`resolved_pre_confirmation_entry`。
- 观察/根因：r1在首个loss构造时报`NameError: domain_variance`；domain-balanced patch只在free head中构造变量，却在
  bounded adapter loss引用。model freeze与P9 action target materialization均未发生，无scientific metric。
- 恢复：在adapter内按`domain_index`计算equal-domain Huber mean/variance；r2不改数据、模型、loss权重或gates。
- 防重复：保留r1失败目录，以r2继续；下一编号=`V67-F07`。

P16 r2按原合同完成并在模型冻结后首次materialize P9 action targets；入口问题已关闭。r2 learned adapter相对qmean的
Spearman/pairwise/selected reduction delta=`-0.007213/-0.044718/-0.001150`，是独立科学负结果而非F06延续。

### V67-F07 — 学习替代trajectory qmean持续破坏跨域决策排序

- 分类：`algorithm/representation-and-authority`；状态：`closed_negative_after_independent_confirmation`。
- 观察：P16 two-domain bounded action-ID adapter在P9为3/6 gates；P17 monotone quantile pool为1/6 gates。P17 learned/qmean
  Spearman=`0.645502/0.658731`、pairwise=`0.749190/0.779650`、selected reduction=`0.387839/0.418184`。
- 根因：trajectory corridor内qmean已是强、低容量且可迁移的充分排序统计；action identity和自由分布混合分别引入
  cohort shortcut与低分位偏置。P17学到distribution mix=`0.497368`，没有产生独立可迁移增益。
- 推翻项：推翻“在现有三cohort上学习替代qmean scorer可提升authority”的假设；不推翻given-trajectory visited-state
  reliability，也不推翻P9 qmean本身的0.4182 selected-cost reduction。
- 外部检索/迁移：NeurIPS 2023 PlanCP与NeurIPS 2025 conformal risk training把uncertainty用于规划risk control而非任意
  改写base score；CVPR 2024 online-map uncertainty也把UQ送入downstream task。P18因此冻结qmean ranking，只训练
  case-level benefit/abstention compiler。
- 防重复：不扫quantile levels、distribution mix、action ID、score residual、gate或selection fraction；P9已消费，只作
  P18 method selection。若P18成立，再到独立cohort一次确认；下一编号=`V67-F08`。

P18 fixed-qmean selective compiler在P9以固定49.30% coverage把selected-cost reduction从`0.418184`提高到`0.487876`
（delta=`+0.069693`），6/6 scenes不退化、4/4 gates通过。该结果使F07进入independent confirmation，尚不因consumed
selection单独关闭；P19保持compiler/coverage/gates冻结，不在P9继续试验。

P19 independent结果保留relative gain：qmean `0.295088→0.350899`（`+0.055811`）、authorized positive rate=
`0.805556`且6/6 scenes不退化；但冻结absolute reduction `0.45`门失败，故3/4正式拒绝并关闭F07，不降门、不改coverage、
不做第二P19。下一方法P20改为四域listwise decision-focused action ordering，不是selective-gate recovery。

### V67-F08 — P20 listwise action compiler候选

- 分类：`algorithm/decision-focused-ranking`；状态：`resolved_by_independent_listwise_confirmation`。
- 动机：F07显示case authority可稳定提高relative benefit，但冻结qmean ordering的cohort ceiling使P19绝对门失败。
- 检索/迁移：ICML 2022将decision-focused learning表述为learning-to-rank；NeurIPS 2019/2021提供可微排序/top-k代理。
  P20以四开发域、case-centered `±0.02` residual和soft top-k target cost直接训练action set。
- 防重复：不扫residual bound、architecture、temperature、loss weight、selected fraction或gate；V67 P1 action target在模型
  freeze后一次读取。下一编号=`V67-F09`。

P20 exact-once在P1 action-task confirmation通过4/4 gates：selected reduction `0.460084`，相对qmean `+0.030723`；
pairwise `0.826230`，5/5 eligible scenes不退化。scene-1046的12 units均低于16-point footprint，保留为coverage边界，
不删scene或降minimum footprint。F08关闭；P21只在冻结P20排序上训练selective authority。

P21在P2V action-task confirmation同样4/4 gates：固定49.30% coverage下，P20/qmean reduction=
`0.404135/0.345130`，selective authority=`0.450102`；35/35 authorized cases nonnegative，5/5 covered scenes不退化。
无新failure。下一候选P22从mean benefit推进到unsafe tail proxy；下一编号仍为`V67-F09`。

### V67-F09 — P22 tail-risk-aware listwise compiler候选

- 分类：`algorithm/risk-sensitive-ranking`；状态：`closed_negative_after_first_trial`。
- 动机：P20/P21已支持mean visited-state cost selection，但`any hidden-FREE` unsafe event仍只作AUROC描述，尚未进入
  differentiable selected-set objective。
- 检索/迁移：ICML 2024 risk-sensitive reward-free RL和NeurIPS 2021 distributional CVaR强调tail outcome；P22新增soft
  selected unsafe-rate损失，但明确不把binary proxy包装成CVaR或safety guarantee。
- 防重复：unsafe weight=`0.25`、六开发域、architecture/residual/temperature/fraction/gates预先固定；V64 P10R4 action
  target在模型freeze后一次读取，不扫weight或tail definition。下一编号=`V67-F10`。

P22 exact-once仅1/4 gates：unsafe reduction比P20增加`+0.004719`，不足冻结`+0.02`，且mean reduction
`0.329362 <0.35`并比P20低`0.003501`；8/8 scene support通过。禁止扫unsafe weight/threshold；binary any-event family关闭。

### V67-F10 — P23 continuous entropic selected-cost候选

- 分类：`algorithm/continuous-risk-sensitive-ranking`；状态：`closed_negative_after_first_trial`。
- 动机/检索：NeurIPS 2022 Efficient Risk-Averse RL指出离散tail会形成tail barrier；NeurIPS 2020 OCE risk learning覆盖
  entropic/CVaR等连续风险。P23用连续target cost的entropic soft selected risk绕开binary plateau。
- 合同：risk aversion=`10`、weight=`0.25`、七开发域；确认用V64 P10R2八场景；与冻结P20/P22/qmean同分母比较。
- 防重复：不扫risk aversion/weight/tail fraction/model/temperature/gate；不把objective称为OCE/CVaR保证。下一编号=`V67-F11`。

P23在mean reduction/pairwise/scene support三门通过，但top-10% tail mean与P20几乎相同（ratio=`0.999450 >0.95`）；
3/4 gates正式拒绝。连续entropic objective提升mean `+0.013006`，但未产生tail独立增益；不扫risk aversion/weight，
tail auxiliary研究关闭。

### V67-F11 — P24 adaptive fixed-total action budget候选

- 分类：`algorithm/task-conditioned-budget-allocation`；状态：`resolved_by_fixed_total_budget_confirmation`。
- 动机：P20/P21已经证明ranking与selective authority；P24研究在总action数完全一致时，能否按case难度分配1--5个actions。
- 方法：P20 within-case order冻结；八域16-hidden bounded `±0.05` case calibration只改变跨case slot priority。
- 防重复：不扫max actions、offset bound、architecture、fraction或gate；P6R action target在模型freeze后一次读取。下一编号=
  `V67-F12`。

### V67-F12 — P24 evaluator未与offset dataset的single-action exclusion对齐

- 分类：`implementation/evaluator-indexing`；状态：`resolved_pre_metric_evaluator_alignment`。
- 观察/根因：r1训练和cache完成后，offset dataset按既有selection合同跳过`<2` eligible actions cases，但adaptive evaluator
  仍遍历全部unique cases，row 78访问size 78 offset数组外索引；Python/NumPy官方文档定义该情况为`IndexError`。
- 科学暴露：model已在confirmation target materialization前冻结；失败发生在任何metric/gate计算前，无scientific result。
- 恢复：evaluator改用与`_within_case_selection`一致的`>=2 actions` cases/all-action denominator；r2复用r1 frozen artifact
  和cache，不重训、不改合同。下一编号=`V67-F13`。

P24 r2 4/4 gates通过：exact budget=`222/222`，adaptive reduction=`0.758380`，相对fixed P20=`+0.161610`，7/7
evaluable scenes不退化。F12工程入口关闭，F11科学候选关闭为positive。r2只读r1 artifact/cache，未重复训练。

### V67-F13 — P25 coverage-constrained fixed-total budget候选

- 分类：`algorithm/selective-budget-authority`；状态：`resolved_by_coverage_constrained_confirmation`。
- 动机：P24要求每case至少1 action；P25在总budget相同下允许部分case abstain，同时冻结至少50% case coverage。
- 方法：九域bounded offset；P20 within-case order不变；每case0--6 actions；总action数等于fixed quarter baseline。
- 防重复：不扫coverage/max actions/offset/model/fraction/gate；P4C action target在model freeze后一次读取。下一编号=
  `V67-F14`。

P25 5/5 gates通过：exact budget=`243/243`、case coverage=`0.606742`、reduction=`0.694998`，相对fixed P20 /
P24=`+0.382792/+0.100552`，8/8 scenes不退化。该结果说明增益来自在固定总预算下联合abstention与跨case分配，
不是减少动作总数；不升级为collision/planning/safety claim。

### V67-F14 — P26 large-cohort coverage transfer候选

- 分类：`algorithm/large-cohort-transfer`；状态：`resolved_by_large_cohort_transfer`。
- 动机：P25在八场景成立后，将其cohort滚入development并重新训练；在从未用于V6.7 allocation的P6E 16场景/
  192-case分层cohort检验迁移规模与场景支持。
- 方法：十域bounded offset；冻结P20 within-case order；exact fixed-quarter total budget、minimum 50% case coverage、
  0--6 actions/case；架构和训练超参原样继承P25。
- 防重复：不扫coverage/max actions/offset/model/fraction/gate；P6E target只读一次。下一编号=`V67-F15`。

P26 5/5 gates通过：exact budget=`511/511`、case coverage=`0.644444`、reduction=`0.792541`，相对fixed P20 /
P24=`+0.391952/+0.108614`，15/15 evaluable scenes不退化。16个输入scenes中scene 0按冻结footprint无evaluable
case，故场景支持分母为15，不是事后排除。

### V67-F15 — P27 stratum-balanced authority候选

- 分类：`algorithm/context-balanced-authority`；状态：`resolved_by_stratum_balanced_confirmation`。
- 动机：P25/P26证明全局coverage有效；P27防止authority集中到容易context，在四个既定strata内各自保证至少50% cases。
- 方法：十一域重新训练同一bounded offset；P20 order、exact total budget、global 50% coverage与0--6 actions不变；
  P6R只作已消费legacy confirmation。
- 防重复：不扫group、coverage、model或gate；若失败关闭group-floor候选。下一编号=`V67-F16`。

P27 6/6 gates通过：budget=`222/222`，global coverage=`0.628205`，minimum stratum coverage=`0.50`；reduction=
`0.800447`，相对fixed P20/P24=`+0.203678/+0.042068`，6/6 evaluable scenes不退化。该positive只在已消费P6R上
支持context-balanced mechanism，不恢复fresh confirmation。

### V67-F16 — P28 budget-conditioned unseen-fraction候选

- 分类：`algorithm/budget-conditioned-authority`；状态：`resolved_by_unseen_budget_transfer`。
- 动机：P20--P27均固定25% budget；P28把requested budget作为显式条件，检验一个模型能否插值到未训练fraction。
- 方法：P10R4从P28训练域移除；其余十域在0.25/0.50联合训练，confirmation fraction固定1/3；exact total与四strata
  coverage约束保持。
- 防重复：不扫fraction、architecture、offset、coverage、group或gate；P10R4是consumed legacy，不作fresh claim。下一编号=
  `V67-F17`。

P28 6/6 gates通过：1/3 exact budget=`363/363`，global/minimum stratum coverage=`0.708333/0.583333`；reduction=
`0.674930`，相对fixed P20=`+0.393479`，8/8 scenes不退化。支持heldout-budget interpolation mechanism；因cohort
全局已消费，不称fresh generalization。

### V67-F17 — P29 nested low/high authority候选

- 分类：`algorithm/budget-path-consistency`；状态：`resolved_by_nested_budget_confirmation`。
- 动机：budget-conditioned priorities可能随requested fraction变化；runtime预算变化需要low-budget authority严格嵌套于high。
- 方法：P4C从P29训练移除；十域0.25/0.50训练；confirmation同时输出exact 25%/50%，high集合以low为mandatory
  prefix再按high-budget priority扩展；两预算各自保持四group coverage。
- 防重复：不扫fractions、nesting rule、group、model或gate；失败即关闭nested candidate。下一编号=`V67-F18`。

P29 7/7 gates通过：low/high exact=`243/243,494/494`、nested count=`243`；low/high reduction=
`0.758868/0.387925`，相对各自fixed P20=`+0.446663/+0.182809`；两预算8/8 scenes不退化。支持budget-path
consistency mechanism，但P4C全局已消费，不称fresh。

