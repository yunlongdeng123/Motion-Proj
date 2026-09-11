# 历史原始记录 043

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F85 — source ranked-range supervision未改变跨cohort fixed50 tail ordering

- 分类：`scientific/selective-tail-objective`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P119-RANKED-RANGE-TAIL-01/20260830T074500Z__ranked-range-tail-s0-r1`；
- method：冻结P109 distribution/crossing score，只在source per-scene `.35--.65` operating range训练bounded hidden32 residual；
  79,478 trajectories中ranked-range positives仅65，optimizer不读P81/P96/P113；
- symptom：P81/P96保持0 events，但P113仍为6，未达到clearance limit5；三个consumed cohorts AUROC相对P109均下降
  `.003836/.004586/.001217`；
- interpretation：binary occupancy flip在source fixed50邻域极稀疏，局部pairwise loss没有足够稳定的跨scene order signal；
  partial-AUC动机不能替代实际迁移结果；
- resolution：不扫percentile band、residual bound、head、loss、seed或coverage；关闭binary fixed50 recovery。下一对象改为
  连续的τ-conditioned boundary-normal state cost/selective regression，保留P113 negative；
- claim impact：无ranked-range/partial-AUC/selective-risk improvement claim，无P113 recovery或safety claim。

下一可用编号：`V67-F86`。

### V67-F86 — continuous cost regressor未超过冻结P109 sufficient score

- 分类：`scientific/continuous-selective-regression`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P120-CONTINUOUS-BOUNDARY-STATE-COST-01/
  20260830T075000Z__continuous-boundary-state-cost-s0-r1`；
- new object：Actor residual沿candidate τ boundary normal的absolute projection除以predicted absolute clearance，再作trajectory max；
  它是τ-conditioned continuous reliability cost，不是旧endpoint error或binary flip；
- symptom：learned/P109 selected cost在P81=`.2032/.1863`、P96=`.1850/.1788`、P113=`.2237/.2247`；learned
  Spearman相对P109 gain=`-.0123/-.0515/+.0173`，两项冻结decision全失败；
- interpretation：top16+clearance回归头在source拟合continuous target，但没有超过P109 standardized crossing score；P109本身已是
  跨三个consumed cohorts的强低容量 sufficient ranking statistic；
- retained evidence：P109 continuous-cost Spearman=`.8065/.7183/.7921`，fixed50 cost reduction=
  `89.75%/77.05%/83.37%`，明显高于clearance-only ranking；这只支持冻结P121候选，不把P120写成成功；
- resolution：不扫cost definition/floor/head/loss/seed/coverage，不训练第二regressor。P121在全新target-unread cohort只确认冻结
  P109 continuous object；
- claim impact：无learned continuous head或independent continuous reliability claim，仍无collision/safety claim。

下一可用编号：`V67-F87`。

### V67-F87 — full-covariance排序增量未保证continuous fixed50 cost nonregression

- 分类：`scientific/full-covariance-continuous-selection`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P122-FULL-COVARIANCE-CONTINUOUS-SELECTION-01/
  20260830T081000Z__full-covariance-continuous-selection-s0-r1`；
- protocol：P121 target仍未物化时，只读已消费P81/P96/P113；冻结P117/P109 checkpoints、continuous cost、`.05m` floor与
  fixed50，不训练、不refit。只有三cohort selected cost均不回退且mean Spearman gain≥`.005`才可成为P121 secondary；
- symptom：full covariance的Spearman gain=`+.011488/+.004767/+.011976`，均值`.009410`通过；但selected cost在P96
  `.178783→.184867`、P113 `.224742→.225542`，故nonregression失败；
- interpretation：P117 full-likelihood训练带来的整体排序改善仍会移动fixed50边界，global continuous rank gain不足以保证
  operating-point cost不退化；与P118一起说明不能把收益简化为conditional rho推理项；
- resolution：不把P117追加到P121同读、不扫rho/score/coverage、不训练combiner；P121 primary继续只用冻结P109；
- claim impact：无full-covariance continuous-selection或独立迁移claim；P117 consumed AUROC mechanism support仍保留。

下一可用编号：`V67-F88`。

### V67-F88 — continuous operating-range rank residual仍产生跨cohort排序漂移

- 分类：`scientific/continuous-selective-ranking`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P123-CONTINUOUS-RANK-RESIDUAL-01/
  20260830T081500Z__continuous-rank-residual-s0-r1`；
- method：在source每scene的P109 score `.25--.75` operating band内，以continuous cost percentile `<=.35/>=.65`构造
  13,123个within-scene pairs；hidden32 bounded residual、6,000 steps，0 development/P121 target optimizer read；
- symptom：P81/P96/P113 Spearman gain=`-.019849/-.056155/+.008165`；P96 selected cost由`.178783`退化至`.183085`，
  所以no-regression与mean-rank-gain两项均失败；
- interpretation：稠密continuous target消除了P119的binary positive scarcity，但source operating-range residual仍改变已较强的
  P109 sufficient ordering并跨cohort漂移；问题不是简单换成continuous pair label即可解决；
- resolution：不扫band、bound、architecture、loss、seed或coverage；关闭P119/P120/P123 downstream head family，不创建P121
  secondary。下一机制若继续，必须改变Actor residual distribution而不是再接selection head；
- claim impact：无continuous pairwise/selective-ranking improvement claim；P121冻结P109 primary不变。

下一可用编号：`V67-F89`。

### V67-F89 — 固定重尾Student-t Actor residual在P96显著过宽

- 分类：`scientific/heavy-tailed-actor-uncertainty`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P124-CORRELATED-STUDENT-T-UNCERTAINTY-01/
  20260830T082000Z__correlated-student-t-uncertainty-s0-r1`；
- literature response：NeurIPS Student-t robust regression支持以重尾likelihood降低outlier影响；CVPR long-tail trajectory工作支持
  直接建模tail distribution。迁移时只把P117 Gaussian NLL改为固定`df=4` multivariate Student-t；
- symptom：P81/P96/P113 AUROC gain=`+.000248/-.054071/-.005427`，fixed50 events=`0/7/7`，相对P109的`0/0/6`
  在P96/P113均退化；两项decision失败；
- interpretation：统一重尾分布把outlier与结构性motion modes混合吸收，训练likelihood更低但boundary-relevant ranking在P96明显
  过宽；NLL改善不能替代跨cohort selective evidence；
- resolution：不扫degrees-of-freedom、scale floor、correlation、loss、seed或coverage；关闭Student-t family。若继续Actor
  distribution，只允许固定低组件数的显式multimodal residual，而不是调重尾参数；
- claim impact：无heavy-tailed Actor uncertainty improvement、P121 secondary或safety claim。

下一可用编号：`V67-F90`。

### V67-F90 — 两组件mixture modes未形成可迁移的boundary-relevant分解

- 分类：`scientific/multimodal-actor-uncertainty`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P125-TWO-MODE-ACTOR-UNCERTAINTY-01/
  20260830T082500Z__two-mode-actor-uncertainty-s0-r1`；
- literature response：CVPR/ICCV multimodal trajectory工作支持显式mode specialization；CVPR 2022同时指出GMM hard-to-optimize/
  overfit风险。迁移只固定K=2，并用mixture-weighted boundary CDF，不扫组件数；
- symptom：mean maximum component weights约`.79--.82`，非完全collapse，但P81/P96/P113 AUROC gain=
  `-.002116/-.024330/-.006968`，fixed50 events=`0/4/7`也劣于P109=`0/0/6`；
- interpretation：组件确实分工，但source中形成的modes不等于跨cohort candidate-τ boundary modes；更低mixture NLL不能保证
  task-conditioned selection。该失败与F89共同关闭单模型tail/multimodal output扩展；
- resolution：不扫K、entropy regularizer、scale floor、projection、seed或coverage；下一步若继续不确定性机制，只允许用独立
  模型间disagreement分解epistemic/aleatoric，而不是再改同一输出分布；
- claim impact：无GMM/multimodal Actor uncertainty improvement、P121 secondary或safety claim。

下一可用编号：`V67-F91`。

### V67-F91 — ensemble全局AUROC一致提升但P96 fixed50出现一个事件

- 分类：`scientific/epistemic-aleatoric-ensemble`；状态：`closed_negative_binary_composite`；
- canonical：`run://worldsim_v67/WS-V67-P126-ACTOR-DEEP-ENSEMBLE-01/
  20260830T083000Z__actor-deep-ensemble-s0-r1`；
- literature response：NeurIPS deep ensembles与ICML uncertainty decomposition支持用独立成员mean disagreement补充aleatoric variance；
  P126固定三成员，复用seed0、只新训seed1/2，law-of-total-variance无可调权重；
- symptom：P81/P96/P113 AUROC gain=`+.001968/+.010012/+.006259`且mean `.006080`通过，P113 events `6→4`；但P96
  events `0→1`，因此event noninferiority失败；
- interpretation：约2.2--2.6%的projected epistemic fraction提供一致global ordering信息，却仍会移动rare-event fixed50边界；
  该结果不能写成binary selective success，但与此前单模型输出族全退化不同；
- resolution：binary composite严格拒绝；不删P96、不调member/seed/epistemic weight。因P121已预注册continuous endpoint，只允许
  P127在同一consumed rows上按事前continuous nonregression/rank decisions一次迁移；
- claim impact：无binary ensemble/fixed50 independent claim；P121 primary不变。

下一可用编号：`V67-F92`。

### P127 freeze note — ensemble continuous-cost迁移

- candidate：冻结P126三成员与total-variance score；P109、continuous cost、`.05m` floor、fixed50均不变；
- role：P81/P96/P113 consumed development；P121 target未物化，P127 optimizer/read均不接触P121；
- decisions：selected continuous cost三cohort全不回退；mean Spearman gain≥`.005`；
- prevention：失败登记F92并关闭ensemble，不扫member/weight/score/cost/coverage；成功才可在P121 rows前另冻same-read secondary。

### P127 outcome / P128 freeze note — continuous迁移成立并冻结same-read secondary

- P127 consumed result：三cohort selected cost全低于P109；Spearman gain=`+.046981/+.135078/+.075514`，两门均通过；
- P128在P121 rows物化前冻结，直接复用P121 primary同一NPZ；P121仍是唯一primary；
- P128 decisions：Spearman gain over P109≥`.005`且selected cost≤P109；失败使用F92并关闭ensemble，不做第二secondary/
  member/weight/score recovery；
- failure delta：P127无新增failure；下一可用编号仍为F92。

### P121/P128 outcome note — independent primary与same-read ensemble增量均成立

- P121 primary：P109 Spearman `.761472`、超过clearance `+.288235`；fixed50 cost reduction `77.36%`且cost低于clearance，
  2/2 decisions，scene-level independent support；
- P128 secondary：ensemble relative P109 Spearman gain `+.047211`、selected cost `.270506<=.277957`，2/2 decisions；
- timing disclosure：08:34:24确认rows absent后P128内容冻结并复制，rows在Git commit guard前的传输窗口内物化；commit
  `572f7d5`因此晚于materialization。内容在读取outcome前冻结且未改，但论文只称prospective-content same-read secondary；
- resolution：无scientific failure，不把P128冒充commit-before-read prereg或独立cohort；下一可用failure仍为F92。

### P129 freeze note — ensemble continuous increment独立确认

- metadata-only cohort：`0017/0345/0962/0095/0522/0625/0798/0921/0927/1063`；target-unread、四location 3/3/3/1、
  内部10 distinct log sessions；只称scene-level independent；
- candidate/decisions：冻结P126 total variance、P109、continuous cost/floor/H3.5/fixed50；Spearman gain≥`.005`且selected
  cost≤P109；
- prevention：只允许target前exact locator correction；scientific failure使用当时下一可用编号并关闭ensemble independent increment，不换
  scene/member/weight/score/cost/coverage/gate或第二cohort。
- outcome：11,406 rows/1,681 trajectories；ensemble relative P109 Spearman gain=`+.042572`、selected cost
  `.308669<.329340`，2/2 decisions。无scientific failure；支持scene-level independent ensemble increment，claim不外推到
  session-level/collision/calibrated probability/closed-loop/safety。

### V67-F92 — P129异步evaluator在错误工作目录解析相对入口

- 分类：`engineering/launcher-entry`；状态：`resolved_pre_run_pre_target`。
- 观察：首次waiting evaluator由含后台async list的shell命令启动，实际从`/root`寻找`./scripts/...p129...py`并立即退出；
  没有创建primary run leaf、没有读取P129 rows、没有计算metric。独立prep及其7个shard workers持续正常运行。
- 根因/外部检索：GNU Bash官方manual说明`&`形成asynchronous list，工作目录/grouping边界需显式控制；组合命令没有把
  预期`cd`可靠约束到该后台evaluator入口。
- 恢复：只用absolute script/config/PYTHONPATH与`setsid`重启相同canonical evaluator；cohort、checkpoints、score、cost、
  coverage与decisions全部不变。
- 防重复：长IO waiting evaluator使用absolute entry；不为此增加smoke/regression matrix。下一可用编号=`V67-F93`。

### P130 freeze note — ensemble moment distribution distillation

- motivation：P126/P127/P128表明ensemble total covariance带来连续排序增量，但三模型推理成本可压缩；P129 archive IO期间
  允许只访问source/consumed development的GPU研究。
- method：冻结P126三成员，按law of total covariance形成单个full-cov Gaussian teacher；P117结构student以闭式Gaussian KL
  训练6,000 steps。只做seed0单trial，不扫蒸馏loss、权重、结构或coverage。
- decisions/outcome：P81/P96/P113 selected cost逐cohort不劣于P126，mean Spearman difference≥`-.005`；实际mean
  Spearman difference=`-.002024`通过，但P113 cost `.225324>.218791`，故single-student moment方案拒绝，不改P129。
- references：UAI 2022/2023 ensemble distribution distillation、NeurIPS 2022 functional ensemble distillation。

