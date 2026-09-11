# 历史原始记录 042

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### P108/P110/P111 outcome note — independent factorization成立但clearance baseline限制贡献归因

- P108 canonical：`run://worldsim_v67/WS-V67-P108-UNCERTAINTY-TUBE-CONFIRMATION-01/20260830T063500Z__uncertainty-tube-confirmation-s0-r1`；
  fresh fixed50 P107/Actor/P75=`5/35/20`，AUROC `.95107/.77605`，2/2 decisions，scene-level independent primary支持；
- P110 same-read prospective secondary：directional/Actor/P75=`1/53/20`，AUROC `.96027/.69142`，2/2 decisions；
- P111 frozen no-learning comparator：clearance-only=`1` event，AUROC `.91644`；它优于P107 fixed50 event count并与P110持平，
  但全排序低于P110；
- interpretation：证据支持“Actor uncertainty distribution与candidate τ boundary解析分离”相对Actor-only/P75稳定迁移，
  也支持direction-aware score的ranking增量；但强geometry baseline阻止把全部事件收益写成learned uncertainty贡献；
- claim boundary：P108只scene-level independent；P110/P111同一read；无session-level、calibrated probability、collision、
  planner、policy、closed-loop或safety claim。无新failure，下一编号保持`V67-F79`。

下一可用编号仍为：`V67-F79`。

### P113 freeze note — 独立检验learned directional uncertainty是否超过纯clearance

- motivation：P108 scalar primary虽超过Actor/P75，但P111 clearance fixed50更少；P110 directional与clearance同为1 event、
  AUROC多`.04383`。需要新cohort检验全排序增量，不能在P108同read上升级claim；
- target-unread cohort：`0094/0331/0521/0003/0013/0038/0797/0920/0926/1061`，四location、10 distinct sessions；
- frozen decision：directional selected events不多于clearance，并且AUROC gain≥`.02`；P109 checkpoint/projection、baseline floor、
  H3.5、time/Actor max、fixed50全部冻结，不引入Actor/P75 gate matrix；
- stop rule：一次read；失败登记下一可用编号（P116完成后为`V67-F82`）并关闭uncertainty-over-geometry claim，不换cohort/
  metric/floor/model或做recovery。
- failure-ID note：P114/P115/P116占用`V67-F79/F80/F81`；P113 prep locator已占F82，P118机制负结果占F83，因此P113若
  scientific decision失败使用`V67-F84`。编号变化不改变任何scientific decision。

下一可用编号：`V67-F84`。

### P114 freeze note — downstream tail aggregation不占用P113确认

- motivation：P109的Actor/time max没有显式传播多个crossing probabilities；task-relevant failure detection提示应在downstream
  cost空间聚合预测分布，而不是继续扩raw query classifier；
- fixed method：冻结P109 Gaussian，只训练top-16 crossing probabilities加independent-union proxy的正权重monotone pool；
  P81/P96均已消费，只作development，P113 rows不会被读取或用于model selection；
- prevention：不扫top-k/pool/loss/seed/coverage，不以P114覆盖P113 primary；若development失败登记`V67-F79`并关闭该tail-pool
  形式，若成功下一failure ID仍为`V67-F79`且只能另取未来target-unread cohort；
- execution：P113 archive IO期间运行6,000-step GPU训练，避免把I/O等待变成研究停顿；不增加hash/checksum/fingerprint或测试矩阵。

### V67-F79 — learned monotone tail pooling稀释directional boundary maximum

- 分类：`scientific/downstream-tail-aggregation`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P114-MONOTONE-TAIL-RISK-01/20260830T071000Z__monotone-tail-risk-s0-r1`；
- symptom：79,478 source trajectories上的balanced BCE降至`.304205`，但consumed P81 AUROC从P109 max的`.967639`
  降至`.951378`；P96从`.904345`降至`.902976`且fixed50 selected events由0增至1。三项冻结decision全失败；
- interpretation：occupancy flip由最接近boundary的Actor/time局部tail主导；将top-16 crossing probabilities与independent-union
  proxy作正权重混合，会把弱、强相关的time/Actor probabilities累积进去。source discrimination改善不能替代跨cohort排序；
- literature response：task-relevant failure detection支持把预测分布传播到downstream cost，但不保证independence-style pooling适合
  强时序/多Actor相关的occupancy boundary。当前结果保留直接对齐decision boundary的P109 maximum；
- resolution：不扫top-k、union公式、model、loss、seed或coverage；P114不进入P113或未来confirmation。P113冻结候选不变；
- claim impact：无trajectory tail calibrator、collision probability或safety claim；不影响P108 factorization primary与P109/P110
  directional evidence。

下一可用编号：`V67-F80`。

### P115 freeze note — 从marginal pooling转向coherent Actor residual sequence

- card point：`V67-F79`证明top-k/independent-union downstream pool在两个consumed cohorts都劣于P109 max，不允许回扫pool；
- literature response：ICCV 2023 Joint Metrics Matter指出marginal metrics/forecasts会遗漏多Agent联合一致性；PRECOG显式建模
  multi-agent conditional futures；CVPR 2026 FoSS使用frequency-domain trajectory结构建模长时依赖；
- migration：P115只改Actor-only uncertainty representation，一次输出9-step residual的固定前4 DCT coefficients和scale，候选τ
  仍只通过P109解析boundary projection进入；P81/P96 consumed development，不读取P113；
- prevention：不扫coefficient count/architecture/loss/seed/projection/coverage；若失败登记`V67-F80`，P113若随后失败顺延F81。

### V67-F80 — 低频spectral Actor sequence在P96过度平滑boundary-relevant residual

- 分类：`scientific/actor-uncertainty-representation`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P115-SPECTRAL-ACTOR-UNCERTAINTY-01/20260830T071500Z__spectral-actor-uncertainty-s0-r1`；
- symptom：P81 spectral AUROC `.977092`较P109 `.967639`提高`.009453`且均0 selected events；P96却从P109的
  0 events/`.904345`退化到7 events/`.847123`，三项冻结decision全失败；
- interpretation：前4个DCT modes提供低频trajectory coherence，但source/P81收益不能迁移到P96；截断会抹掉影响局部
  occupancy boundary crossing的末端或高频残差。joint/spectral representation并不自动等于更可靠的task ranking；
- literature response：Joint Metrics Matter与PRECOG说明联合future重要，FoSS说明频域结构可建模长时依赖；本结果限定了其在
  当前constant-velocity residual UQ上的直接低频迁移，不能借论文动机忽略独立development反转；
- resolution：不扫DCT coefficient count、hidden width、loss、seed或projection；P115不进入P113或未来confirmation，保留
  P109 pointwise directional Gaussian；
- claim impact：无spectral/joint uncertainty或collision claim；P108/P109/P110证据不变。

下一可用编号：`V67-F81`。

### P116 freeze note — 用directional q90 field替代Gaussian分布假设

- card point：P115表明低频joint Gaussian在P81/P96方向反转，禁止扫DCT coefficient/structure；
- literature response：AISTATS 2022 Multivariate Quantile Function Forecaster直接参数化多元quantile，NeurIPS 2021强调full
  quantile function可表达distribution-free uncertainty；当前迁移只取固定q90与8 directions，不声称完整quantile function；
- method：Actor-time residual沿固定unit direction作pinball监督，推理时boundary normal只作为directional query，clearance只在
  解析ratio中出现；因此不回到raw end-to-end query classifier；
- prevention：P81/P96 consumed development，不读P113；不扫direction count/quantile/model/loss/seed/coverage。失败登记
  `V67-F81`并关闭该形式，P113若随后失败使用F82。

下一可用编号仍为：`V67-F81`。

### V67-F81 — distribution-free directional q90仍不及Gaussian standardized crossing margin

- 分类：`scientific/actor-uncertainty-distribution`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P116-DIRECTIONAL-QUANTILE-FIELD-01/20260830T072000Z__directional-quantile-field-s0-r1`；
- symptom：P81 q90/P109 AUROC=`.963841/.967639`且均0 selected events；P96 q90/P109=`.889318/.904345`，
  selected events=`6/0`。两cohort AUROC gain均负，三项冻结decision全失败；
- interpretation：直接预测adverse residual q90移除了Gaussian shape假设，却也丢失P109 mean与scale组成的standardized margin；
  `q90/clearance`在当前source跨域不足，不能因distribution-free动机包装为更可靠；
- literature response：multivariate quantile function与quantile-UQ工作支持灵活分布建模，但本项目固定8-direction/q90的窄迁移
  没有超过Gaussian；这是否定当前实现对象，不是否定一般quantile方法；
- resolution：不扫direction count、quantile、network、loss、seed或coverage；与P114/P115一起关闭P109替代model family；
- claim impact：无directional quantile/conformal/collision probability claim；P109/P113保持唯一冻结主线。

下一可用编号：`V67-F82`。

### V67-F82 — scene-0003 session跨public archive part导致冻结locator不完整

- 分类：`engineering/data-locator`；状态：`resolved_pre_target_exact_locator_recovery`；
- failed run：`run://worldsim_v67/WS-V67-P113-DIRECTIONAL-VS-CLEARANCE-CONFIRMATION-PREP-01/
  20260830T070000Z__directional-vs-clearance-prep-s0-r1`；
- symptom：初始六shard extraction得到其余9 scenes的3,517 files，但scene-0003的384个LIDAR members全部缺失；runner在
  preprocess、target row materialization和任何metric前按exact extraction合同退出，P113 evaluator保持等待；
- root cause：scene-0003与P81 scene-0344共享session，但先前scene-0344命中的shard04不能推出同session早期scene所在part；
  nuScenes公开10-part archive可跨session切分，官方devkit只要求合并全部parts且不提供session→part index；
- recovery：不换scene/model/decision。完整排除02/03/05/06/08/09，在01命中384/384并直接提取，停止尚未完成的07/10；
  locator期间先预处理9个ready scenes。仅修正`scene-0003:04→01`，prep r2映射3,894/3,894、复用9 scenes并在60.63s
  完成scene-0003，总wall71.41s。无hash/checksum/fingerprint或额外quality gate；
- claim impact：纯pre-target engineering failure，不影响P113冻结scientific protocol，也不产生任何confirmation result。

下一可用编号：`V67-F83`。

### P117 positive mechanism note — full bivariate covariance improves consumed directional ranking

- card point：P114--P116三种替代均未超过P109，但P109的diagonal Gaussian仍强制纵/横Actor residual条件独立；
- literature response：CVPR 2023 IPCC-TP显式学习joint Gaussian means/covariances，支持把相关结构作为独立变量，而不是继续扩大
  raw query classifier或扫downstream aggregation；
- frozen migration：P117仅新增一个bounded correlation输出和完整bivariate Gaussian NLL，source/features/network width/
  optimizer/steps/seed/boundary projection/coverage均沿用P109，只读consumed P81/P96且不读P113；
- result：P81/P96都保持0 selected events，AUROC相对P109分别`+.004903/+.009320`，平均`+.007111`通过`.005`门；
  verdict=`supported_development_correlated_actor_uncertainty`；
- claim/prevention：该结果不是failure，也不占用failure ID；不扫rho bound/loss/width/seed/projection。它只能作为未来全新
  target-unread cohort的候选，不能在P113 target已冻结后替换primary或改变decision。

下一可用编号仍为：`V67-F83`。

### V67-F83 — conditional rho推理项未解释P117 full-covariance训练收益

- 分类：`scientific/mechanism-ablation`；状态：`closed_negative_after_single_ablation`；
- canonical：`run://worldsim_v67/WS-V67-P118-CORRELATION-ABLATION-01/20260830T073000Z__correlation-ablation-s0-r1`；
- frozen comparison：同一P117 checkpoint、mean/scale/rows/projection/fixed50，只比较conditional rho与rho=0；不训练、不refit；
- symptom：P81 AUROC gain仅`+.000304`，P96为`-.000115`，平均`+.000094 < .003`；两cohort event count虽均为0，
  但两cohort AUROC正增益门和mean-gain门失败；
- interpretation：P117相对P109的开发增益来自完整bivariate likelihood训练package，不能定位为inference-time conditional
  correlation项的直接贡献；mean/scale在joint NLL下的重塑是未分离因素；
- resolution：不扫rho bound、loss、seed或重复训练，不用事后门解释P117。保留P117为完整package的consumed-development候选，
  但论文明确报告P118 negative mechanism；不影响P113冻结P109 primary。

下一可用编号：`V67-F84`。

### V67-F84 — independent AUROC增量未转化为fixed50事件优势

- 分类：`scientific/selective-tail-transfer`；状态：`closed_negative_after_one_shot_confirmation`；
- canonical：`run://worldsim_v67/WS-V67-P113-DIRECTIONAL-VS-CLEARANCE-CONFIRMATION-01/
  20260830T070500Z__directional-vs-clearance-s0-r1`；
- frozen read：P109 checkpoint/normalization/linear projection、`.05m` clearance、H3.5、time/Actor max、per-scene fixed50与
  10-scene cohort全在target前冻结；只比较directional和clearance两门；
- symptom：7,206 rows/1,525 trajectories/79 flips上，directional AUROC `.920155`较clearance `.875291`提高`.044864`
  并通过`.02`门，但selected events=`6 vs 5`，严格noninferiority门失败；1/2 decisions；
- interpretation：learned Actor uncertainty改善全局排序，却没有稳定控制固定coverage处的rare-event tail；强clearance geometry
  在单一operating point仍可更好。AUROC与fixed-budget selective risk不可互相替代；
- literature response：NeurIPS 2022 partial-AUC optimization与fixed-coverage selective prediction工作说明全局AUC可掩盖相关
  FPR/coverage区间。本结果支持下一路线直接研究ranked range/selective tail objective，而不是继续堆Gaussian/quantile结构；
- resolution：关闭当前P109 uncertainty-over-clearance claim；不降coverage/floor/gate，不换P113 cohort，不在本read上试P117，
  不做第二P113 recovery。P108相对Actor/P75的scene-level factorization primary保留；
- claim impact：可写independent AUROC ranking gain，但必须同时写composite verdict rejected；无calibrated probability、collision、
  planner、policy、closed-loop或safety claim。

下一可用编号：`V67-F85`。

