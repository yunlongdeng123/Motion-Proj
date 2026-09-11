# 历史原始记录 037

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F18 — P30 horizon-conditioned H插值候选

- 分类：`algorithm/horizon-conditioned-authority`；状态：`resolved_by_heldout_horizon_transfer`。
- 动机：此前visited-state对象均固定H=2s；P30显式条件化H，响应“给定tau，未来H秒被访问states是否可靠”。
- 方法：P10V H=1s新cache与H=2s development联合训练；P10X从训练移除，confirmation H=1.5s；25% exact budget、
  三context groups各50% coverage、P20 order冻结。
- 输入结果：H=1s materialization 72 cases、733/864 eligible actions，无失败。
- 防重复：不扫H、footprint、model、offset、coverage、group或gate；P10X globally consumed，不作fresh claim。下一编号=
  `V67-F19`。

P30 6/6 gates通过：H=1.5s source/eligible=`864/717`，66 cases，budget=`176/176`，global/minimum group coverage=
`0.636364/0.50`；reduction=`0.740743`，相对fixed P20=`+0.505189`。Scene support 5/6，唯一正delta=
`1.86e-9`是浮点均值差，仍按预注册`<=0`规则计higher；minimum 5门通过，不做epsilon改写。

### V67-F19 — P31 joint budget-H unseen-condition候选

- 分类：`algorithm/joint-task-conditioned-authority`；状态：`resolved_by_joint_condition_transfer`。
- 动机：P28和P30分别证明budget/H插值；P31检验同一模型对未见联合条件`(1/3,1.5s)`的组合泛化。
- 方法：四domains，H=`1/2s`、budget=`.25/.50`联合case rows；P10X训练排除；确认cache复用P30 H=1.5s，
  budget改为1/3，三context groups coverage保底。
- 防重复：不扫条件点、model、offset、coverage、group或gate；失败不回退到单轴结果。下一编号=`V67-F20`。

P31 6/6 gates通过：unseen pair `(1/3,1.5s)` budget=`236/236`，global/minimum group coverage=
`0.727273/0.583333`，reduction=`0.690636`，相对fixed=`+0.499918`，5/6 scenes不退化。

### V67-F20 — P32 joint-condition nested budget候选

- 分类：`algorithm/joint-condition-budget-path-consistency`；状态：`resolved_by_joint_nested_confirmation`。
- 方法/结果：H=1.5s上low/high exact=`176/176,352/352`，176 low actions全部保留；low/high reduction=
  `0.811047/0.404128`，相对fixed=`+0.575493/+0.263991`；minimum group=`0.50/0.791667`，7/7 gates。
- 边界：P10X globally consumed，只支持联合条件下nested mechanism；无safety claim。下一编号=`V67-F21`。

### V67-F21 — P33 second-cohort joint condition transfer候选

- 分类：`algorithm/second-cohort-joint-condition-transfer`；状态：`resolved_by_second_cohort_joint_transfer`。
- 方法：同一四development domains、budget/H conditions、architecture/loss/gates；P4C不进训练，首次物化H=1.5s并在1/3
  budget读取一次；四context groups各coverage>=0.50。
- 防重复：不扫条件/model/gate，不以P10X结果调整P4C门；globally consumed但joint-H task未读。下一编号=`V67-F22`。

P33 6/6 gates通过：P4C H=1.5s `973/1152` eligible、89 cases，budget=`315/315`，global/minimum group=
`0.696629/0.50`；reduction=`0.698243`、相对fixed=`+0.439588`，8/8 scenes不退化。

### V67-F22 — P34 bounded heteroscedastic authority候选

- 分类：`algorithm/aleatoric-case-uncertainty`；状态：`closed_negative_after_first_trial`。
- 调研：NeurIPS 2023支持异方差Gaussian regression稳定化；NeurIPS 2024指出单网络evidential epistemic可能不可靠。
- 方法：P31同一joint-condition features上训练bounded mean与`0.005..0.10` scale；P10X consumed selection固定
  conservative offset=`mean+1.0*scale`，与冻结P31 mean compiler比较。
- 结果：scale-error Spearman=`0.190272`通过`>=0.15`，但conservative/mean reduction=
  `0.610037/0.690636`，delta=`-0.080599`，未达`+0.01`；其余exact total、minimum group与scene support通过。
- 结论：scale与误差相关不等于其保守加法能改善有限预算决策；不以弱UQ诊断替代selection utility。
- 边界：仅aleatoric error scale；不声称epistemic/calibrated interval/OOD；P22/P23 action-tail family保持关闭。
- 防重复：不扫scale bound、sigma weight、loss、model或gate；aleatoric priority family关闭。下一编号=`V67-F23`。

### V67-F23 — P35 fixed deep-ensemble authority候选

- 分类：`algorithm/model-disagreement-authority`；状态：`closed_negative_after_first_trial`。
- 调研迁移：Deep Ensembles（NeurIPS 2017）以独立初始化成员分歧作为实用uncertainty基线；结合NeurIPS 2024对
  evidential epistemic可靠性的批评，P35显式使用三个独立模型，而不从单头scale外推epistemic结论。
- 方法：seed `0/1/2`顺序训练同一P31 joint-condition head；P4C consumed selection固定priority=
  `ensemble_mean+1.0*ensemble_std`，与冻结P33 mean compiler在相同exact budget/group constraints下比较。
- 结果：disagreement-error Spearman=`0.144178`通过`.10`，但mean disagreement仅`3.53e-6`，三个成员的
  residual RMS均饱和到`0.05`；conservative与P33选择完全相同，reduction=`0.698243`，delta=`0.0`。
- 结论：相同小模型/完整批次在强边界解上发生成员塌缩；有弱误差相关仍没有可消费的排序变化。
- 边界：disagreement仅为epistemic proxy；不声称posterior calibration、OOD、collision/planning/closed-loop/safety。
- 防重复：不扫成员数、seed、uncertainty weight、model、loss或gate；uncertainty-for-decision路线关闭。
  下一编号=`V67-F24`。

### V67-F24 — P36 conditioned differentiable top-k action compiler候选

- 分类：`algorithm/conditioned-decision-focused-action-ranking`；状态：`resolved_by_direct_conditioned_topk`。
- 调研：ICLR 2019 NeuralSort与ICML 2020 SoftSort将离散排序替换为可微连续松弛，使最终top-k目标能反向训练。
- 方法迁移：在P20 action features上增加budget与H两个条件；用`.25/.50` budgets、`H=1/2s`四domains训练，
  主损失为soft top-k权重下的真实visited-state cost，并保留小权重pairwise/regression稳定项。
- 判定：P4C consumed `(1/3,1.5s)`直接输出action scores，经同一exact-total/group constraints选择；相对冻结P33
  reduction至少`+0.005`且6 scenes不退化。
- 结果：reduction=`0.719901`，相对P33=`+0.021658`、相对fixed=`+0.461246`；exact `315/315`、minimum
  group=`0.50`、8/8 scenes、4/4 gates。
- 防重复：固定temperature/residual bound/model/loss/gates；不扫参。下一编号=`V67-F25`。

### V67-F25 — P37 frozen conditioned-action second-cohort transfer候选

- 分类：`algorithm/cross-cohort-conditioned-action-transfer`；状态：`closed_negative_after_frozen_read`。
- 方法：P36 model/normalizer完全冻结，P10X consumed `(1/3,1.5s)`单次读取；与冻结P31在相同exact/group约束比较。
- 判定：reduction delta `>=+0.005`、minimum group `.50`、至少5 scenes不退化；无训练/refit/sweep。
- 结果：coverage/minimum group提升到`0.787879/0.666667`，但reduction=`0.656886`低于P31 `0.690636`，
  delta=`-0.033750`；exact与scene gates通过，decision gain失败。
- 结论：P36的P4C增益不是跨cohort稳定增益；增加coverage不能替代selected-cost质量。
- 边界：第二consumed cohort的method transfer，不是fresh population confirmation。下一编号=`V67-F26`。

### V67-F26 — P38 smooth worst-domain conditioned top-k候选

- 分类：`algorithm/worst-domain-decision-focused-training`；状态：`closed_negative_after_first_trial`。
- 调研：ICLR 2020 GroupDRO关注最坏group风险；ICML 2021 REx通过训练域风险关系改善OOD generalization。
- 方法迁移：P36仅将domain mean+variance聚合换为temperature `.02` log-sum-exp smooth maximum；其余数据、
  architecture、soft top-k、pairwise/regression、residual bound与epochs完全不变。
- 判定：P10X consumed `(1/3,1.5s)`相对冻结P31 reduction `>=+0.005`、minimum group `.50`、scene support `5`。
- 结果：coverage/minimum group=`0.803030/0.708333`，但reduction=`0.629974`，比P31低`-0.060662`；
  worst-domain objective比P36 transfer再退`-0.026913`。
- 结论：平滑最坏域目标鼓励更广coverage，但没有提高跨cohort action ordering；objective-reweighting路线关闭。
- 防重复：不扫temperature/aggregation/model/loss/gate。下一编号=`V67-F27`。

### V67-F27 — P39 expanded-domain conditioned top-k候选

- 分类：`algorithm/training-domain-diversity`；状态：`resolved_by_expanded_domain_training`。
- 方法：恢复P36 mean+variance objective；加入已消费P4C/P10X H=1.5作为两个development domains，训练budgets=
  `.25/1/3/.50`，共6 domains；P3C action targets从训练排除。
- 判定：P3C H=2/budget=1/3相对冻结P31 reduction `>=+0.005`、minimum group `.50`、至少4/5 scenes不退化。
- 结果：exact=`238/238`、coverage/min group=`0.766667/0.708333`；reduction=`0.724052`，相对P31=
  `+0.013217`、相对fixed=`+0.355925`，5/5 scenes，4/4 gates。
- 防重复：只改训练domain/budget denominator，不改模型、loss、temperature或gate；不扫cohort组合。下一编号=`V67-F28`。

### V67-F28 — P40 frozen expanded-domain fourth-cohort transfer候选

- 分类：`algorithm/fourth-cohort-transfer`；状态：`closed_negative_after_frozen_read`。
- 方法：冻结P39 artifact；P10R4 action targets未进入P20/P31/P39训练，H=2/budget=1/3一次读取；四scene-pair groups。
- 判定：相对冻结P31 reduction `>=+0.005`、minimum group `.50`、至少6/8 scenes不退化。
- 结果：coverage/minimum group=`0.760417/0.583333`、8/8 scenes，但reduction=`0.654575`，比P31低
  `-0.020355`；更广coverage再次未转为更低cost。
- 边界：cohort全局已消费，故不是fresh confirmation；无训练/refit/sweep。下一编号=`V67-F29`。

### V67-F29 — P41 terminal eight-domain conditioned top-k候选

- 分类：`algorithm/terminal-domain-expansion`；状态：`closed_negative_terminal`。
- 调研：DomainBed报告在严格统一条件下，carefully implemented ERM可达到强domain-generalization表现；因此不再追加
  新DG正则，而测试有限数据域扩展的终点。
- 方法：P39加入已消费P3C/P10R4为development，形成8 domains×3 budgets；P10R2 action targets完全排除。
- 判定：P10R2 H=2/budget=1/3相对P31 `>=+0.005`、minimum group `.50`、至少6/8 scenes。
- 结果：P41/P31 reduction=`0.747149/0.743093`，delta=`+0.004055`，仅比冻结门低`0.000945`；exact、
  group、8/8 scenes通过，但AND verdict仍拒绝。
- stop：不降门、不扫cohort组合/model/loss/temperature；action-scorer跨cohort family关闭。下一编号=`V67-F30`。

### V67-F30 — P42 frozen allocator + trained action refinement hybrid候选

- 分类：`algorithm/compositional-case-action-authority`；状态：`resolved_by_hybrid_composition`。
- 调研迁移：ICLR 2020 blackbox combinatorial differentiation强调学习模型与既有组合优化结构的端到端组合；项目内不引入
  外部solver，而保留已支持的P31 allocator并学习P20之上的case-centered action residual。
- 训练：9 consumed domains×3 budgets；base score=P20，head结构/soft top-k同P41；P31 case offset全程冻结。
- 判定：P6R action-target-untouched H2/budget1/3相对P31 `>=+0.005`、minimum group `.50`、至少5/7 scenes。
- 结果：exact=`294/294`、coverage/min group=`0.705128/0.50`；reduction=`0.800132`，相对P31=
  `+0.007912`、相对fixed=`+0.250012`，7/7 scenes，4/4 gates。
- 防重复：融合权重固定1:1加法，不扫模型/loss/temperature/gate。下一编号=`V67-F31`。

### V67-F31 — P43 frozen hybrid nested-budget结构候选

- 分类：`algorithm/hybrid-nested-budget-structure`；状态：`closed_negative_low_budget_regression`。
- 方法：P42/P31/P20全冻结；P6R quarter/half两预算分别条件化score，low集合强制作为high前缀扩展。
- 判定：两端exact、low subset high、minimum group `.50`、相对对应P31 nested baseline不退化、两端至少5 scenes。
- 结果：exact与222-action strict nesting通过；hybrid low/high reduction=`0.808732/0.641285`，P31=
  `0.833218/0.638464`，deltas=`-0.024486/+0.002821`；低预算非退化失败。
- 结论：P42 gain是budget-dependent；不以high gain覆盖low regression。
- 边界：同一consumed P6R结构读取，不新增泛化claim；不扫budget/weight/gate。下一编号=`V67-F32`。

### V67-F32 — P44 low-budget anchored hybrid候选

- 分类：`algorithm/budget-anchored-hybrid`；状态：`resolved_by_low_budget_anchor`。
- 调研迁移：Nested Dropout学习有序子结构；Once-for-All用progressive shrinking服务多resource constraints。P44不迁移
  其网络规模机制，而迁移“低资源子系统应是高资源系统前缀/锚点”的结构原则。
- 方法：P42 residual乘固定amplitude=`clip((budget-.25)/(.50-.25),0,1)`；quarter严格回退P20，half完整hybrid。
- 数据：9 consumed domains训练；P6R H1.5 task target新物化，与训练并行；budget1/3一次判定。
- 启动记录：首次materializer CLI缺`--run-id`在argparse退出，0 cache写入/target read；补同一run-id后继续，非科学失败。
- 结果：H1.5 eligible=`881/1152`、76 cases；exact=`292/292`、coverage/min group=`0.671053/0.50`；
  reduction=`0.809547`，相对P31=`+0.020361`、相对fixed=`+0.306632`，6/6 scenes，4/4 gates。
- 防重复：不扫anchor/full fraction/amplitude/model/loss/gate。下一编号=`V67-F33`。

### V67-F33 — P45 frozen anchored-hybrid nested-budget候选

- 分类：`algorithm/anchored-hybrid-nested-structure`；状态：`resolved_by_anchored_nested_nonregression`。
- 方法：P44/P31/P20全冻结；新H1.5 cache quarter/half strict nesting。Quarter action score由anchor严格等于P20。
- 判定：两端exact、strict nesting、minimum group `.50`、相对各自P31 nested baseline非退化、至少5 scenes。
- 结果：low/high exact=`220/436`且strict nested；deltas over P31=`0/+0.005463`，minimum group=
  `.50/.625`，scene=`5/7,7/7`，5/5 gates。
- 边界：同一新H task condition的结构read；无训练/refit/budget/weight sweep。下一编号=`V67-F34`。

