# WorldSim V6.7 ArXiv 技术报告：从端到端可靠性捷径到 Actor uncertainty × trajectory boundary

- 分支：`research/worldsim-v6.7-anisotropic-surface`
- 报告状态：`P147 continuous selection + P183 density CDF independent support; P192 refinement boundary documented`
- 主要硬件：单张 RTX 3090 24GB
- 证据角色：development、consumed cross-cohort、scene-level independent confirmation 严格分开

## 摘要

WorldSim V6.7研究的问题是：给定候选 Ego 轨迹 `τ`，未来 `H` 秒中它实际访问的 world/Actor states 是否可靠，以及这种
可靠度能否用于固定覆盖率排序。研究首先证明了一个重要但有限的信号：P81在10个未读场景的全 Actor-query row 上，冻结
query scorer将fixed-50% selected unreliable events从Actor-only的57降至26。然而，当评价对象收紧为真正被轨迹访问的Actor
states或trajectory-level any-failure时，P84--P94的fixed summary、Deep Sets、attention、ordinal、continuous、quantile、
Gaussian、BCE与ensemble全部不能稳定超过Actor-only。其原因不是简单容量不足，而是endpoint Actor prediction error本身不依赖
`τ`，候选轨迹只改变membership。

P95据此将endpoint改为task-conditioned occupancy-decision flip，但端到端query classifier在development很强、独立cohort却
反转；hierarchical temporal/Actor encoder和更多source data都未恢复。最终成功的迁移不是继续扩大query网络，而是将问题因子化为：
网络只预测Actor未来位置误差分布，candidate `τ`只通过signed boundary clearance进入解析查询。P107的scalar q90 uncertainty
tube在两个consumed cohorts一致成立，并在P108的新10-scene cohort将fixed-50% occupancy flips从Actor-only的35降至5；P109的
directional diagonal-Gaussian进一步以boundary-normal projection获得更强排序。P111表明纯clearance geometry已经是强baseline，
因此最终论文贡献必须区分“factorization超过Actor-only”与“learned uncertainty超过geometry”。P113表明learned score有排序
增量但fixed50 cost未稳定占优；转向continuous cost后，P121和P129先后给出独立支持。最终P147在新10-scene cohort的
`.8/1.5/2.5/3.0/3.5s`五个horizon上均得到正Spearman增量和负selected-cost差，macro=`+.17419/-.01777`，
把贡献推进为scene-level independent multi-horizon continuous selection。

在概率接口上，P173单调CDF虽有fresh Brier discrimination，却未超过horizon-only的marginal calibration。P182因此改为直接拟合
`log1p(continuous boundary-state cost)`的5-component conditional density，再以解析CDF查询七个预算。P183在完全不同的
10 scenes/10 logs、五个horizon上确认：相对P173的integrated Brier macro降低`28.48%`，mean absolute reliability error macro降低
`69.38%`，两项预注册gate均通过。P192进一步表明source scene等权采样可在四个已消费development cohort上继续改善P182，
但P193在已消费P183 rows上的冻结次级诊断显示H`.8/1.5s` Brier回退`.93%/1.13%`、macro calibration无净改善，
因此该改进不能升级；P182仍是唯一具有fresh支持的density。P194/P195的全局与horizon-conditional sampler均未消除跨cohort
回退。P196冻结两个专家后在旧development通过，但P197 consumed-secondary仍复制短H退化；因此这些结果都不进入P183主结论。

在此边际结论上，P199首次把同一candidate trajectory的四个horizon对齐并学习Gaussian-copula dependence；source-heldout
joint-all-H reliability相对P182 marginal independence product的Brier改善`16.97%`、calibration error改善`71.85%`。P200在已消费
P183 rows上的secondary改善`17.40%/66.72%`。P201进一步在10个未读official-val scenes/10 logs、1,846 joint trajectories上
确认：相对independent marginals的Brier改善`17.52%`、calibration error改善`53.37%`，两门均通过。因此joint-horizon
dependence现获得fresh scene-level支持；不外推到session/population或formal calibration。P202说明直接joint CDF虽将calibration error再降
`32.51%`，却使Brier恶化`10.32%`；相比之下，P203对冻结P199输出施加共享rank-preserving beta map，在source dev把Brier和
calibration error分别改善`1.32%/54.32%`，P204 consumed-secondary也保持`2.57%/34.51%`改善。该校准层需等待P205同一次
fresh read的前瞻次级结果；P205最终把P199 Brier再改善`3.70%`、calibration error改善`49.12%`，2/2，但仍严格标作
same-read prospective secondary。P206的全局常数copula相对P199使Brier与calibration error分别退化`1.02%/24.94%`，说明
P199提升来自输入条件化依赖，而非仅加入静态相关矩阵。P207据此测试一次冻结的rank-2-plus-diagonal条件结构，作为full
conditional Cholesky的结构化消融：Brier微增`.077%`但calibration退化`1.52%`，因此拒绝。P208进一步冻结P199与
independence成分，只训练逐实例linear mixture gate；模型给P199平均`.9828`权重但Brier/校准仍退化，故关闭local shrinkage。
P209转而固定`nu=4`训练conditional Student-t copula，检验Gaussian相关未表达的尾依赖。这些development trials均不改变
P201 raw P199 primary合同。P209的Brier/校准分别退化`.56%/2.20%`，因此copula变体终止。P210进一步把prediction
object改写为四H maximum continuous cost的条件密度；其解析CDF与all-H reliable event严格等价，用以检验P182式density
inductive bias能否优于dependence factorization。P210把calibration改善`28.42%`但Brier退化`1.30%`；P211的proper-score
scalar pool在source几乎全选P210，却在heldout仍使Brier退化`.96%`，表明flat representation发生refinement shift。P212据此用
共享horizon token encoder与mean/max pooling重建相同maximum density，不改变P201 primary。
P212在source dev同时改善Brier`3.84%`与calibration`17.80%`，但P213在P183 secondary的Brier退化`2.74%`，所以该
maximum-density分支不进入主方法；它作为结构与transfer负结果解释为何最终采用P199 dependence factorization加P203校准层。
P214随后将maximum对象扩展为四个prefix survival curves：宏平均Brier改善`1.16%`、最终四H Brier改善`5.21%`，但宏平均
calibration error退化`30.33%`，因此拒绝。P215以互斥的density-fit、calibration和development scene sets训练低自由度
monotone beta层，在source dev同时改善Brier `.895%`和calibration error `38.31%`；但P216冻结迁移到P183后Brier反而
退化`1.89%`，虽校准仍改善`19.87%`。因此prefix-survival density尚无跨cohort正证据，不能替代P199/P203。P217仅作为
受AISTATS covariate-shift calibration启发的post-hoc unlabeled-target importance-weighting development；domain accuracy仅
`.52915`且P183 Brier仍退化`1.93%`，故该family关闭。P218进一步把对象从maximum/first-passage改为四H时间加权的累计
visited-state exposure density；当前只处于source development，尚不构成论文结论。
P218相对P182+P199连续分解control的Brier仅改善`.387%`且calibration error退化`26.25%`。P219正在用互斥source
density/calibration/dev scenes与一个shared monotone beta map作最后一次P183 consumed transfer；在该read完成前累计exposure
也不构成论文正结论。
P219虽在source dev改善Brier/calibration `1.37%/22.11%`，在P183却退化`4.49%/25.27%`，因此累计exposure也关闭。
P220开始把稳定的P199概率接口视作frozen predictor，另学其event-level proper loss并在固定50% coverage下选择性授权；
这是authority compiler的development object，尚无跨cohort或risk-control保证。
P220在source fixed50相对confidence把selected Brier/calibration改善`23.61%/36.43%`，P221在P183仍改善
`13.71%/27.69%`；但P222在更难P201上分别退化`1.31%/9.67%`，所以absolute loss authority未获跨cohort稳定支持。
P223正以source-only same-budget pairwise proper-loss ranking作一次恢复；P201已消费，任何正结果也只能授权新的fresh test，
不能本身写作confirmation。
P223把P201 Brier/calibration退化缩小到`.43%/2.14%`但仍0/2，故逐budget learned authority关闭。P224将selective object
改为整条trajectory的七预算reliability curve，以integrated realized Brier训练单一fixed50 authority；当前仍是post-hoc
development，不能视为confirmation或formal risk control。
P224在P201进一步退化`7.48%/22.25%`，learned authority因此关闭。P225/P226表明P203 calibration可让P201整曲线
selected Brier改善，但该选择在P183/source的selection-conditional calibration不一致，不能形成selective claim。P227转而
蒸馏已有fresh支持的P203(P199) teacher：以P182 marginal surface和P199 features输入，输出结构上单调的七预算CDF，目标是
用single forward保留teacher质量而消除1024-sample copula MC。P227在P201的teacher MAE=`.007633`，student/teacher Brier=
`.090272/.090478`、calibration=`.023727/.024630`，两项冻结门均通过；当前batched student forward=`.000940s`，teacher
MC stage=`.007007s`。这仍是观察P201后的post-hoc development；P228已在target read前冻结全新10-scene/10-log cohort，
archive/preprocess与确认并行运行。等待IO期间P229只训练一次`64x64` compact student，不做结构或超参扫描。
P229把参数从22,280降至7,048（`-68.37%`），P201 teacher MAE=`.008252`、Brier degradation仅`.159%`且
calibration改善`.000945`，2/2通过。P230进一步固定该训练合同，只移除8个P199 condition features，测试28个marginal
CDF是否足以编译teacher；它不把distillation等同为horizon independence，也不读取P228 fresh quality。
P230在P201获得MAE=`.009653`，Brier/calibration相对teacher改善`.571%/.001233`，说明conditional teacher可被
marginal-only runtime interface近似，但仍需fresh确认。P231随后固定64x64 full-input student，以half teacher/half source
outcome proper loss训练一次，研究distillation是否能从纯emulation推进到quality refinement；不扫mixing weight。
P231虽在P183/P201改善Brier `2.02%/1.29%`，P201 teacher MAE却升到`.027831>.02`，说明静态half/half loss
改变了compiler semantics（F178）。P232依据NeurIPS 2020 PCGrad与ICLR 2026 DTO-KD，改为冲突时投影truth gradient并
匹配teacher-gradient norm；这是单次gradient-level恢复，不做loss-weight sweep。
P232在9,672/10,000 steps触发conflict projection，最终P201 MAE=`.009108`，Brier/calibration相对teacher改善
`.428%/.000491`，3/3通过。P233据此从final curve推进到4-prefix×7-budget surface：用base budget CDF和三个
budget-monotone retention curves同时保证budget monotonicity与horizon-prefix monotonicity；不依赖post-hoc penalty。
P233在P201获得surface/final MAE=`.006982/.009186`，surface Brier/calibration相对teacher改善`.433%/.000927`，
结构violations=`0/0`，3/3通过。P235固定该surface合同，只移除8个P199 condition features，检验marginal-only
runtime interface；P234将使用P228首次fresh rows作同读secondary确认，不另开cohort。

本研究不提供collision probability calibration、planner/policy authority、closed-loop性能或safety guarantee。可辩护贡献是一个
任务条件化但分层的可靠性接口：`Actor uncertainty distribution × candidate-trajectory boundary query`。

## 1. 研究对象与评价合同

对场景中的Actor `i`、未来离散时间 `t` 和候选 Ego 轨迹 `τ`，我们区分三个对象：

```text
E_i,t       = Actor constant-velocity forecast error
F_i,t(τ)    = predicted-vs-observed occupancy decision flip near τ
C_i,t(τ)    = Actor-to-τ signed boundary clearance
```

早期模型直接学习 `score(Actor history, τ) -> reliability`。最终模型改为：

```text
p(Actor future residual | Actor history, t)
                    ×
analytic boundary query(C_i,t(τ))
                    ↓
trajectory-level reliability ranking
```

主要评价采用每scene固定50% coverage，报告被选择trajectory中的occupancy-flip event count与全排序AUROC。固定覆盖率避免
事后阈值优化；event count是任务选择指标，AUROC用于区分同一operating point下的整体排序。Interaction radius固定为
`Actor half-width + 1.0m Ego half-width`，预测horizon为3.5s、9个future samples。

## 2. 递进研究链

### 2.1 全row信号与visited-state对象失败

P81在9,559 Actor-query rows、735 unreliable endpoint events上取得query/Actor/P73 selected events=`26/57/45`，query相对
Actor-only减少54.39%，10/10 scenes不增，AUROC=`0.94147/0.92947/0.93666`。但这允许模型优先选择远离Ego轨迹的Actor rows。

收紧对象后，P84在2,113个visited rows上为`235` events，劣于P75的`208`；P85在1,089条visited trajectories上为
`203`，劣于P75的`199`。P86--P94覆盖fixed summary、Deep Sets、Set Transformer、ordinal thresholds、Huber max-error、
q90 pinball、heteroscedastic Gaussian、direct BCE与三成员deep ensemble；最佳P86也只有query/Actor/P75=`187/193/199`，
query-over-Actor改善3.11%，未达冻结10%，absolute reduction仅37.48%。这条负链关闭“visited Actor最大位移误差”对象，而
不是触发architecture sweep。

### 2.2 真正task-conditioned occupancy flip仍发生跨cohort反转

P95把监督改为9个future samples上constant-velocity Actor path与observed Actor path相对`τ`的occupied/free decision是否
翻转。Development cohort上，P95 query/Actor/P75=`7/28/13`，AUROC=`0.83952/0.69366`，4/4 gates通过。P102进一步用
time-token encoder和Actor set pooling得到`4/27/13`，成为development best。

独立P96却发生反转：P95=`8/5/12`、AUROC=`0.65542/0.71181`；P102的prospective secondary P103=`9/7/12`，
relative reduction同样为负。P104 time-local auxiliary、P105 multi-task recovery和P106全source scaling没有恢复独立结论。
这证明端到端query classifier可以学习cohort-specific Actor/trajectory interaction shortcut，development内更强的结构和更多数据
不等于task-conditioned迁移更稳。

### 2.3 Actor uncertainty与candidate boundary解析因子化

P107网络完全不读取candidate `τ`。它从19维Actor history/dynamics和normalized future time预测9-step position-error q90；
trajectory score固定为：

```text
max_actor,time q90_error_i,t / max(abs(C_i,t(τ)), 0.05m)
```

在consumed P81/P96上，P107 query/Actor/P75分别为`2/36/13`与`2/9/12`，query AUROC分别为`0.92901/0.87305`。
两个异质cohort方向一致，区别于P95/P102反转。

P108在新的10-scene cohort进行一次scene-level independent primary read：8,766 rows形成1,531 trajectories和116 flips；
fixed-50%选择764条时，P107/Actor/P75=`5/35/20`，absolute reduction=91.36%，query-over-Actor reduction=85.71%，
AUROC=`0.95107/0.77605`。两项冻结decision均通过，支持Actor uncertainty与candidate boundary分离的主结论。

### 2.4 Directional uncertainty与geometry归因边界

P109将scalar tube升级为Actor-only Ego-frame longitudinal/lateral residual的diagonal Gaussian，训练目标为Gaussian NLL；推理时
只沿Actor-to-Ego boundary normal投影mean/variance，形成linearized standardized crossing margin。916,722个Actor-time tokens、
6,000 steps后final NLL=`-3.64128`。Consumed P81/P96均选择0 events，AUROC=`0.96764/0.90434`。

在P108同一read上，target出现前冻结的P110 directional secondary为`1/53/20`，AUROC=`0.96027/0.69142`。但P111
no-learning clearance-only也选择1 event，AUROC=`0.91644`。因此：

- learned directional score相对纯clearance有`+0.04383` AUROC的全排序增量；
- fixed-50% event count没有显示learned uncertainty超过geometry；
- P108仍然支持factorization相对Actor-only/P75，但不能把全部收益归因于学习到的uncertainty。

P112尝试用冻结P109 Gaussian做256-sample nonlinear Euclidean crossing。它在P81保持0 events，却在P96退化到3 events且
AUROC从linearized的`0.90434`降至`0.85852`，因此拒绝并关闭sample-count/nonlinear-sampling sweep。当前证据支持
与decision boundary对齐的linear projection，不支持将finite-sample nonlinear Monte Carlo包装为更可靠的collision probability。

P114进一步检验task-relevant downstream aggregation：冻结P109，把每trajectory top-16 crossing probabilities和
independent-union proxy送入只有正权重的monotone pool。虽然79,478 source trajectories上的balanced BCE降至`.304205`，
P81 AUROC却从`.967639`降至`.951378`；P96从`.904345`降至`.902976`，fixed50 events还从0增到1。该结果说明
occupancy flip更接近局部boundary-tail事件，累积多个强相关time/Actor probability会稀释signal；保留P109 max并关闭聚合扫参。

P115再把Actor uncertainty改成一次输出完整9-step residual sequence的前4个DCT modes。它在P81把AUROC从P109的
`.967639`提高到`.977092`并保持0 events，却在P96退化为7 events/`.847123`，而P109为0/`.904345`。低频
coherence在一个cohort有益，却会跨域抹掉boundary-relevant末端或高频残差；因此同样终局拒绝，不扫coefficient count或结构。

P116最后移除Gaussian假设，训练8-direction q90 residual projection field，并用adverse-direction q90/clearance排序。P81
q90/P109 AUROC=`.963841/.967639`且均0 events；P96=`.889318/.904345`且events=`6/0`。因此非参数
directional quantile也未超过P109 mean/scale standardized margin。P114--P116三条替代线全部关闭。

P117进一步检验P109唯一未表达的二维相关结构：保持P109全部输入、宽度、优化器、步数、seed、boundary projection与
fixed50不变，只新增bounded correlation并以完整bivariate Gaussian NLL训练。P81/P96都维持0 selected events，AUROC=
`.972542/.913665`，相对P109增益=`+.004903/+.009320`，平均`+.007111`达到冻结`.005`门；预测平均绝对相关系数=
`.39595/.43201`。该结果支持correlation-aware法向方差作为下一代候选，但只来自consumed development，不能事后替换
P113已冻结的diagonal P109，也不构成独立泛化或概率校准证据。

P118用同一checkpoint将推理时rho置零，保持mean/scale/rows与全部selection合同相同。Conditional rho相对zero rho的
P81/P96 AUROC gain仅`+.000304/-.000115`，平均`+.000094`，未达`.003`。因此P117正结果只能归因于完整bivariate
likelihood training package；不能进一步声称conditional correlation term自身已被跨cohort机制消融支持。

### 2.5 P113独立uncertainty-vs-clearance确认

P113在任何target read前冻结新的10-scene、四location cohort：`0094/0331/0521/0003/0013/0038/0797/0920/
0926/1061`。P109 checkpoint、normalization、linear projection、0.05m floor、H3.5、time/Actor max和fixed50均不变。
唯一decision为：

```text
directional selected events <= clearance-only selected events
AND
directional AUROC - clearance-only AUROC >= 0.02
```

P113 outcome：fresh 7,206 rows、1,525 trajectories、79 flips，fixed50 directional/clearance/Actor/P75=
`6/5/38/20`。Directional AUROC=`.920155`，clearance=`.875291`，增量`+.044864`通过`.02`门；但events
noninferiority `6<=5`失败。整体verdict=`rejected_independent_directional_uncertainty_gain_over_clearance`。这支持独立
全排序增量，却拒绝固定coverage tail上的learned-uncertainty-over-geometry claim；两者必须同时报告。

输入工程注：prep r1的scene-0003冻结shard locator缺失384/384 members，在任何preprocess/target read前退出；官方archive
无session→part index。exact locator在shard01命中全部members，config只改`04→01`；prep r2映射3,894/3,894，复用已并行
预处理的9 scenes并完成scene-0003，target前关闭`V67-F82`，cohort/scientific protocol不变。

P119随后按ranked-range objective在source fixed50邻域训练bounded tail residual。由于79,478 source trajectories中只有65个
positive落入冻结range，它在P81/P96/P113得到events=`0/0/6`，与P109完全相同，且AUROC三处都小幅下降。该负结果说明
改变binary ranking loss仍未解决tail transfer；下一研究应把对象改为连续τ-conditioned boundary-state cost，而不是扫range。

P120把target改为observed Actor residual沿τ boundary normal的absolute projection除以predicted clearance的trajectory max。
新增continuous regressor未超过P109并登记F86；但冻结P109对该cost在P81/P96/P113的Spearman达到
`.8065/.7183/.7921`，fixed50 cost reduction=`89.75%/77.05%/83.37%`，而clearance Spearman仅
`.5625/.3795/.6307`。因此论文当前可把它写成consumed-development mechanism，独立claim必须等P121。

P121在新target read前冻结10-scene scene-level independent cohort与两项composite decision，确认P109 continuous-cost ranking
及fixed50 selection，而不重新打开binary flip gate。14,554 rows/1,581 trajectories上，P109 Spearman=`.76147`，超过
clearance `+.28823`；fixed50 selected cost=`.27796`，相对全体降低`77.36%`且低于clearance `.32215`。2/2通过，形成
scene-level independent continuous reliability support。

P121 archive IO期间，P122在P121 rows出现前用已消费P81/P96/P113比较P117 full covariance与P109。Full covariance使
continuous-cost Spearman平均提升`.00941`，但P96/P113 fixed50 selected cost略退化，预先冻结的nonregression失败并登记F87。
因此P117不进入P121同读secondary；独立primary仍只确认P109，且不做covariance/coverage sweep。

P123进一步用13,123个source within-scene continuous-cost pairs训练fixed50附近的bounded rank residual。它消除了P119的
binary稀疏性，但P81/P96 Spearman分别下降`.01985/.05615`，P96 selected cost也回退，登记F88。P119/P120/P123共同表明：
在冻结P109后增加downstream selection head没有稳定收益，后续机制研究应回到Actor residual distribution本身。

P124据此把P117 Gaussian likelihood唯一替换为固定`df=4` correlated Student-t。916,722 tokens训练后，P81几乎不变，
P96/P113 AUROC下降`.05407/.00543`且events=`7/7`，登记F89。重尾NLL虽更低，但统一放宽tail会损害boundary ranking；
这支持下一步区分多种motion modes，而不是继续调Student-t自由度。

P125固定K=2 correlated Gaussian mixture并以weighted boundary CDF评分。组件没有完全collapse，但三cohort AUROC仍分别比P109
低`.00212/.02433/.00697`，events=`0/4/7`，登记F90。说明source likelihood modes不自动成为τ-boundary modes；因此
single-model Gaussian/full-cov/Student-t/GMM output family至此关闭。

P126转向三成员deep ensemble，以mean aleatoric variance加member-mean epistemic variance。三cohort AUROC均提升，平均
`+.00608`，P113 events `6→4`；但P96 `0→1`使binary composite仍拒绝并登记F91。epistemic fraction仅约2--3%，却是
single-model扩展后首个跨三cohort同向排序信号。P127已在P121 target前冻结，只迁移到continuous cost一次。

P127的continuous transfer三组全成立：selected cost均下降，Spearman gain=`+.04698/+.13508/+.07551`。因此P128在P121
rows出现前冻结为prospective same-read secondary，只检验相对P109的rank gain与cost noninferiority；它不替换P121 primary。

P128在同一P121 rows上得到ensemble Spearman `.80868`，相对P109 `+.04721`；selected cost `.27051<.27796`，两门通过。
时间边界必须披露：runner/config内容在08:34:24 rows-absent检查后冻结并复制，但rows在Git commit guard前物化，故commit
`572f7d5`晚于materialization；内容在读取outcome前未改。论文称其prospective-content same-read secondary，而非严格
commit-before-read prereg或第二独立cohort。

为确认ensemble increment本身，P129从official val metadata冻结另一组target-unread 10 scenes，四location分布3/3/3/1且
cohort内10 distinct logs。P126/P109、continuous cost与两项increment decisions完全不变。11,406 rows/1,681 trajectories上，
ensemble/P109 Spearman=`.82688/.78431`（gain=`+.04257`），fixed50 cost=`.30867/.32934`；两项均通过。因此ensemble
increment获得scene-level独立支持，但历史session overlap仍禁止session-level、collision、calibrated probability或safety外推。

P129首次waiting evaluator因Bash async-list的工作目录作用域从`/root`解析相对入口，在run/target前退出；改用absolute
entry和`setsid`后恢复，prep持续运行，故F92只属pre-run engineering。等待archive IO时，P130按UAI ensemble distribution
distillation思路，将P126 moment-matched full covariance以闭式Gaussian KL蒸馏到一个P117结构student；一次6,000-step
GPU训练，只在consumed P81/P96/P113判断相对P126的continuous selection/rank retention，不读取P129 target。

P130 final KL=`.07387`且mean Spearman difference=`-.00202`通过，但P113 selected cost `.22532>.21879`，所以moment
distillation严格拒绝（F93）。P131不调KL，而改成functional distillation：单query MLP直接拟合P126 task-conditioned
row boundary score，以显式signed-clearance profile和boundary normals保留decision geometry；仍只作consumed development。

P131 row Smooth-L1虽低至`.00635`，trajectory-max后的mean Spearman difference却为`-.36263`且三组cost全面退化（F94）。
这把失败定位为supervision granularity而非简单容量。P132据RD-Suite/PiRank/PLD把trajectory max置于训练图内，用同scene
trajectory pair ordering直接蒸馏P126；不再优化pointwise平均误差，也不增加temperature或top-k sweep。

P132把Spearman恢复到`.829--.851`，但相对P126 mean仍少`.02018`且三组selected cost均回退（F95）。因此compression
family关闭。P133转向BatchEnsemble原生高效表示：shared weights加三组rank-one member factors/independent bootstraps，
一次graph直接学习aleatoric+epistemic decomposition，再复用同一τ-boundary解析；这检验失败是否来自单student缺少member diversity。

P133 NLL虽为`-3.691`，epistemic fraction却仅`.13%--.34%`，mean rank差`-.01454`且P81/P113 cost回退（F96），
支持shared-rank-one diversity collapse解释。P134因此进入Packed independent regime：三套独立blocks以batched kernels置于
单graph，保留三成员容量/FLOPs；它检验parallel packaging而非parameter compression，不写成single-model加速结论。

P134把epistemic fraction恢复到`1.46%--2.23%`且mean rank delta=`+.00187`，但P96 cost `.17218>.16757`使严格
composite失败（F97）。由于P134每member batch只有P126的1/3，P135仅提高到P126-equivalent 65,536做一次compute parity；
不改模型或decision，也不将可能的成功写成compute reduction。

P135在P126-equivalent batch下mean rank差=`-.00127`，但三组selected cost仍小幅回退（F98），故packing route关闭。
P136转向单训练路径快照：3个固定cosine cycles在总计6,000 steps内产生2000/4000/6000三个solutions，以FGE/Snapshot
Ensemble方式估计member disagreement；这检验独立全训练是否必要，不主张校准posterior。

P136在P81有效，却在P96产生rank `-.02696`与cost回退，mean rank差=`-.00855`（F99）；cyclic endpoints不足以形成
跨cohort模式。P137改用20个low-LR iterates拟合SWAG diag+low-rank weight covariance并固定采3模型，仍只检验continuous
selection/rank retention；因optimizer沿用AdamW，论文不把它表述为严格Bayesian calibration。

P137达到mean rank `+.00231`并改善P113 cost，但P81/P96 cost微回退（F100），single-path posterior route关闭。P138不再
近似P126，而把P117 XY full covariance与deep-ensemble epistemic covariance联合：三个full-cov members的within/between
covariance同时投影到τ-boundary normal，要求相对diagonal P126取得明确`.005` mean rank gain且cost全不退。

P138在P81/P113改善，却在P96 rank/cost反转，mean gain=`+.00359<.005`（F101）。P139据此不再扩covariance，而把
global token sampling改为uniform source-scene sampling，三member模型/steps/seeds保持P126；它检验scene size sampling
shortcut，不等价于GroupDRO/Fishr，也不声称source scenes是semantic domains。

P139在三cohort cost全回退，mean Spearman gain=`-.01275`（F102），说明parameter-free scene balancing丢失了自然token
distribution的有效统计。P140保留自然token权重，但把scene改为member bootstrap unit：每个member固定有放回抽102个source
scenes，以系统性scene omission/duplication增强成员差异；P129 primary完成后才作为第四个consumed cohort参与development。

P140在P113/P129降低selection cost，却在P81/P96回退，mean rank gain=`-.00405`（F103）；structured scene diversity并未
稳定超过P126。P141因此回到natural-token训练，在精确复用P126三成员基础上只新增seeds3/4，检验5-member规模是否在四个
consumed cohorts产生一致increment；若支持才需要另一组fresh confirmation。

P141 mean rank gain仅`+.00056`且P96/P113 cost回退（F104），故三成员保持complexity-benefit optimum，停止member sweep。
P142转而采用conditional forecasting的对象定义：三成员直接拟合候选τ在每个future time的boundary-normal projected Actor
residual distribution；其监督仍是连续world-state residual而非decision cost/event，从模型层面把UQ条件化到被访问state。

P142在P129取得`+.01319` rank gain、P113降低cost，但P96显著回退且四cohort mean gain近0（F105）：task conditioning有
信息，却不能从头替代通用Actor distribution。P143据此采用frozen-base residual correction：P126提供projected mean/scale，
query-conditioned ensemble只拟合standardized residual distribution，从结构上保留P126跨cohort稳定性。

P143仍使四cohort cost全退、mean rank gain=`-.01344`（F106），说明source query-conditioned recalibration不可迁移，per-time
family关闭。P144只补最后一个未覆盖结构：以P126 trajectory score为anchor，完整Actor-query token set经Deep Sets聚合后学习
bounded rank residual；监督直接位于部署层级的source within-scene continuous-cost order。

P144在P129降低cost但P96再次反转，mean rank=`-.00096`（F107），下游compiler capacity路线关闭。检查上游后发现P126把
`.8/1.5/2.5/3.0s` source horizons只编码成normalized fraction，丢失absolute future time。P145追加`fraction×H`并重训
三成员，以H3.5 consumed cohorts检验显式time-varying uncertainty能否改善外推。

P145在P81/P113/P129 rank提升但P96=`-.01616`，mean仍为负且三组cost回退（F108）。P146隔离time-scale机制：完整冻结
P126 network/means，只训练每member每轴的bias与positive absolute-time slope，使time conditioning只能改变aleatoric scale。
P146的12 scalars在P129降低selected cost，但四cohort mean Spearman gain仍为`-.001607`且两组cost回退（F109），因此
absolute-time scalar growth不是稳定解释。后续P147转向新的10-scene五时域独立确认，P148直接学习未压缩的9步残差序列集成，
两者分别回答“跨H是否成立”和“跨时刻相关结构是否缺失”。P148四cohort rank一致下降，mean gain=`-.012380`（F110），
说明共同sequence decoder仍不等于联合分布收益。P149进一步把multimodality从P125的逐时刻components提升到整条trajectory modes，
并用未来任一时刻boundary crossing作为直接score；这是对P148 negative的机制递进，而非capacity sweep。P149四mode均active，
但四cohort rank mean下降`.098286`（F111），表明joint likelihood/any-crossing仍与continuous cost错位。P150因此不再间接预测
residual distribution，而在5.18M query-time tokens上直接学习normalized boundary-state cost分布并汇聚upper cost。P150在
P81/P129得到正rank增量，却在P96反转`-.028055`（F112），说明direct object优于generative proxy但ERM不稳定。P151保持对象/
architecture/score不变，以scene×horizon worst-quartile group NLL隔离检验domain-robust optimization；P96进一步恶化到
`-.115416`（F113），关闭direct-cost/DRO family。P152转而保持已支持的P126 residual object，唯一机制变化是为每个member
添加永久冻结的random function prior，以检验function-space epistemic extrapolation，而非继续增加downstream capacity。
P152四cohort rank全退（F114），说明forced diversity没有可靠方向。P153冻结P109预测，仅拟合full Bayesian last-layer
covariance；它在部分cohort有正信号，但epistemic fraction约`10^-4`且mean gain仅`.000844`（F115）。P154因此不再用
token-count posterior concentration，而在frozen hidden space显式学习source density，只在低密度输入膨胀P126 variance。
P154确认所有target cohort都出现hidden-density shift，却使P81 operating cost明显回退（F116），表明rarity不是reliability。
P155将机制移到训练期：同time-fraction RegMixup在source Actor/residual空间平滑scene variation，同时保留原始ERM loss；
四cohort mean rank仍为`-.007045`（F117），故augmentation family关闭。P147 prep的scene0110 shard locator在任何target read前由
scan分母纠正为02（F118），不改变科学protocol或fresh cohort。
P156进一步把多时域建模改写为continuous-time residual increments：以真实`H/8`预测velocity error并积分mean/variance，检验
kinematic temporal coherence能否解除P148 direct position decoder的跨cohort退化；四cohort rank一致下降约`.029`（F119），
表明independent increment accumulation同样不迁移。P147 r1在0 target read下确认scene0110缺388 files，r2只补修正后的shard02。
P147 r2最终`3,909/3,909` mapped并完成10/10 scenes。唯一fresh read中，五个H的P126-vs-P109 Spearman gain=
`+.373741/+.240461/+.096388/+.086006/+.074345`，selected-cost差=
`-.014964/-.017759/-.015318/-.015881/-.024911`；macro=`+.174188/-.017767`，2/2决策通过。这是当前最强
multi-horizon evidence，但独立性只到scene level，不能写成session-level泛化或安全保证。
与此同时，P157不再增加共享模型容量，而把`.8/1.5/2.5/3.0s` source domains拆成四个独立三成员专家，并令H3.5固定
路由到最近的H3.0专家，直接检验shared-horizon negative transfer。12个member训练NLL均正常，但四个H3.5 consumed
cohorts的mean Spearman gain=`-.59434`、cost为P126的约3.9--5.8倍（F120）。因此拒绝nearest-lower extrapolation；该负结果
与P147正结果共同支持保留shared multi-horizon representation，但不能否定拥有exact target-horizon训练数据的专家。
P158因此不再拆horizon或增加网络容量，而保留P126 shared three-member architecture，只以closed-form marginal Gaussian
CRPS替代NLL训练。旧四cohort rank mean=`-.02371`且cost全退（F121）；P147 post-confirmation五H虽rank全正，但只有
`.8/1.5s` cost改善，中长H回退随H扩大。由此拒绝用marginal CRPS替换P126，也不按post-read horizon事后切换模型。
P159只递进一个机制：三members联合优化multivariate Energy Score，以ensemble-level sample distance同时约束accuracy与spread；
其shared architecture和downstream score保持P126不变。旧四cohort rank mean=`-.04251`且cost全退（F122），P147也只有
短H cost微降。因此proper-score retraining family关闭；下一步研究frozen P126 member distributions的聚合规则。
P160将moment matching换成逐member exact crossing CDF的等权linear pool；旧四cohort rank mean=`-.04330`且cost全退，
P147五H也全部退化，H0.8 rank下降`.31838`（F123）。这说明moment-margin aggregation整体不能被“更exact”的CDF mixture
自然替换。不过P161的直接zero-epistemic ablation进一步修正了归因：移除between-member
variance后P147五H几乎不变，explicit epistemic fraction仅约1.2%--1.8%（旧四约1.6%--2.6%），full-vs-control rank
mean=`-.00009`（F124）。因此真正支持的是member-averaged mean/aleatoric moment predictor，不是已被单独证明的epistemic addend。
P162据此不再调variance，而扩展world-state object：从processed Actor annotations学习future yaw residual，并把oriented-rectangle
support的一阶不确定性与P126 position field组合；position-only control共享同一oriented clearance。该结果将回答粒子近似之外的
Actor footprint state是否能为trajectory boundary reliability提供增量。结果yaw MAE随H增长，但旧四cohort rank mean=`-.000284`、
仅2/4 cost改善，P147 H3.5 cost回退`.02613`（F126）；因此拒绝yaw Gaussian的一阶support传播，不否定oriented box object本身。
P163保留oriented object但移除线性化：直接学习每个query normal上的exact support residual，并与冻结P126 position field组合。
5.18M query-time tokens上的三成员均正常收敛，但旧P81/P96/P113/P129 rank gain mean=`-.001156`，P96 cost回退；
P147五H post-confirmation诊断也没有稳定rank/cost方向（F127）。因此负结果不只来自yaw一阶传播：在当前轨迹边界成本中，
厘米级support residual没有超过position residual与clearance的主导作用。oriented-footprint family关闭，P126/P147 position-state
结论保持，论文不得将box/yaw state写成已支持增量。

P164据此不再扩展footprint state，而改进已支持的position reliability条件表示。多Actor forecasting文献普遍显式建模social/
pairwise interaction；本实验只迁移最小可归因模块：冻结P126三个member，以zero-init最近8 Actor set-attention adapter修正各member
的mean/log-scale。它保留P126 compiler与continuous target，因此任何增量都对应邻居条件，而非重新训练整个预测器；当前角色仍是
consumed development，P147只作post-confirmation描述。三adapter source NLL均降至约`-5.0`，但旧四cohort rank gain mean=
`-.04474`且cost全部回退；P147五H同样全面退化（F129）。这说明关系上下文并非无信息，而是把scene-specific interaction
composition注入单Actor marginal后破坏了跨场景稳定性。论文保留该negative mechanism，不把AgentFormer/IPCC式完整joint
forecasting的有效性与本最小adapter混为一谈。

P165将这次失败转为预测对象变化：不再用neighbors修正任一Actor marginal，而冻结P126 mean/scale，联合生成同一anchor下
所有Actors的9-step standardized residual innovations。Permutation-equivariant diffusion的samples被直接编译成P120同定义
continuous boundary cost q75，因而区别于P149的单Actor coherent modes与any-crossing proxy。该阶段仍为consumed development；
只有旧四cohort相对P126同时不退化才可能进入新的独立确认。P165 final noise MSE=`.31946`；旧四rank gain全正、mean=
`+.00811`，P147五H也全正且cost五组全降。可是旧P81/P96/P129 fixed50 cost分别回退`.00594/.00112/.00900`，
因此冻结composite仍拒绝（F130）。论文可以报告“joint residual samples提供跨9个切片一致的rank signal”，但不能写成q75
selection candidate成立，也不能用已读P147选择另一个quantile。

P166转向不改变排序的校准问题：固定P126 score，以horizon-conditioned monotone neural spline预测同定义continuous cost的
条件期望，并与horizon-only baseline比较。这响应回归不确定性文献中“rank/accuracy不等于calibration”的区分；当前只检验
point expected cost，不产生credible interval、coverage或conformal guarantee。若consumed development成立，才会冻结模型到全新场景。
P166旧四cohort raw MSE虽逐组小幅下降，但平均仅`4.09%<20%`，10-bin expected-cost error反而四组全差；P147 H0.8也
退化（F131）。因此论文结论维持“可排序、可降低fixed-coverage cost”，不升级为绝对cost calibrated prediction。

P167不继续修改模型，而将冻结P126-vs-P109/P147五时域协议转移到第二个target-unread 10-scene cohort。该cohort覆盖四个
location、9个distinct logs；由于仍有log overlap，只能增加scene-level独立重复的强度。执行采用shard-ready preprocess与
scene-ready GPU scoring的在线流水线，使archive IO与3090评分重叠；结果未产生前，本报告不预写支持或拒绝结论。

与P167 archive IO重叠的P168只改变P165 samples的风险泛函：用最高4/16 sampled costs的upper-tail mean替代单个q75，
`.75`水平完全继承且不扫描。该迁移依据是coherent risk对整个尾部的利用，以及joint diffusion sample-level reliability aggregation；
它不训练新模型。旧四rank gain mean仅`+.00460<.005`，selected cost四组全部回退，故0/2并登记F132；按冻结规则未读取
P167 prospective rows。这个负结果把问题从“q75估计噪声”进一步定位为“global rank与scene内fixed50 cutoff错配”，下一研究对象
必须直接训练coverage-conditioned selection objective，而不是继续选择手工sample statistic。

P169对这一定位作controlled training：沿用P144的P126-anchored trajectory-set representation与容量，只把within-scene pairwise
ranking surrogate替换为scene-list soft lower-50% selected-cost objective。16×128 source lists、`.20`温度及`.5` residual bound
均在训练前冻结；旧四development通过前不读取P167。该设计对应differentiable sorting文献关于“训练surrogate与最终ranking metric
存在gap”的直接修复。P169使旧四cost由P144的明显退化收缩为3组微降、P96仅`+.000306`微退，但mean rank gain仍只有
`+.00212<.005`，F133。结果说明objective alignment有效但新增表示不迁移；P126-anchored residual head family关闭，且未读P167。

P170据此从selection head转向可评估的one-sided cost upper bound：source scenes按mod-5拆分训练/校准，训练q90 log-cost
monotone spline并以held-out residual作一次split-conformal offset，同时保留horizon-only control。旧四需要每组经验覆盖≥`.88`
且mean upper-bound至少锐化10%才进入P167。由于跨scene exchangeability未证明，论文即使结果为正也只写empirical coverage，
不将NeurIPS CQR的一般有限样本结论越权为当前数据协议的formal guarantee。

P170 r1在完成source q90训练后、任何evaluation前暴露split entry错误：P109 source artifact已由旧protocol排除absolute
mod-5 scenes，导致calibration为空（F134）。恢复只在artifact实际unique scene groups上按固定顺序每5取1，保持group-disjoint；
r2从头训练，未使用r1 loss选择任何参数。因此F134是工程split恢复，不是coverage负结果。

P170 r2在旧P81/P96/P113/P129得到`.9257/.9541/.9102/.9060`经验coverage，并相对horizon-only把mean upper bound
分别缩短`15.94%/41.07%/23.02%/14.60%`（平均`23.66%`）；development 2/2。模型与offset在P167 rows前冻结，
当前等待五时域prospective secondary。论文在P167结果前只可写“development支持进入确认”，不能写跨scene formal coverage。

P171是P170的唯一效率增量：冻结base quantile model，以score+horizon条件的q90 residual rectifier替代constant conformity correction，
并保留最终held-out scene offset。它对应ICML 2025对conformity score conditional rectification的最小迁移；旧四保持coverage并比P170
再锐化5%前不读取P167。该阶段仍只允许empirical coverage/sharpness解释。

P171旧四coverage仍为`.9542/.9279/.9548/.9126`，但相对P170的mean sharpness恶化`19.48%`（F135）；只有P96
更窄。说明source residual conditional pattern未跨scene稳定迁移，constant P170 offset更可靠。P171未读P167并关闭，P170保持
唯一prospective cost-bound candidate。

P167 10/10 scene-ready流水线完成，五H rank gain=`+.4191/+.2774/+.1510/+.1285/+.0947`，selected-cost delta=
`-.01767/-.02199/-.01309/-.01515/-.01630`；macro rank/cost=`+.21412/-.0168403`，2/2，形成第二次scene-level
independent multi-horizon support。scene-1065 H3.5的P109局部score为常量，local Spearman按定义为undefined并以`null`报告；
不进入pooled/macro。r1 strict JSON failure与r2无科学合同差异（F137）。

P170在P167 rows前已冻结，五H prospective empirical coverage=`.8907/.8632/.8318/.8261/.8226`；虽然mean sharpness
reduction=`25.09%`，但4/5 horizon低于`.88`，F138。故只能写“score-conditioned bound更窄但跨scene under-cover”，不能
写formal或empirical 90% coverage support；不在P167上二次校准。

P172在P167 partial read后另行训练q10/q90 two-sided cost interval，仅用旧四作consumed development；P167明确禁止承担
该candidate的prospective confirmation。旧四mean width缩短`18.55%`，但P81 coverage只有`.732<.78`，F136；因此不建立
新confirmation，不改lower quantile。该负结果强化P170 one-sided upper bound比two-sided cost interval更稳的当前边界。

P173进一步直接学习`P(cost(visited state)<=budget | τ-score,H,budget)`的单调CDF，而非voxel correctness。七个固定budget
联合训练后，旧P81/P96/P113/P129相对horizon-only的integrated-Brier降低`34.49%/45.27%/46.20%/31.15%`，四组
全部不退，mean=`39.28%`，2/2 development support。其mean absolute reliability error只在P96改善，另外三组更差；因此
该结果只证明budget-conditioned distribution含新增判别信息，尚不是calibrated probability。P167已读，不能承担P173独立确认。

P174用21个held-out source scenes拟合保持单调的Beta map；P81/P96/P113/P129相对raw marginal calibration-error change=
`-10.77%/+16.63%/+11.05%/+5.80%`，mean只有`+5.68%`，F140。P176则直接用integrated Brier训练，proper-score
reduction提升为`38.07%/45.33%/48.26%/32.15%`，但四组marginal error仍全高于horizon-only，F141。这组结果区分了
refinement/discrimination与probability calibration：论文可以写P173/P176显著降低proper score，但必须同时写跨scene概率刻度未建立。

P175为P173冻结了新的10-scene/10-log五H确认，archive→scene→GPU流水线正在运行。P177只在已消费source上检验scene-uniform
Brier training，严格排除P175。P177 mean Brier reduction=`40.82%`，但四组marginal error仍全高于control，F142；因此关闭
source-only probability-calibration路线。它不能替换P175的事前candidate，也不会改变新cohort门。

随后两项机制诊断同样严格排除P175。P178把absolute inverse-clearance作为第二个单调risk coordinate，旧四Brier与calibration
error均4/4改善，但mean calibration gain只有`5.08%<10%`（F143）；这说明几何条件有效但普通additive conditioning不足。
P179改用top-16 Actor-query DeepSet context residual，Brier在3/4 cohort回退、mean calibration change=`-8.45%`（F144），表明
高容量场景上下文在source拟合后形成interaction shortcut。P180据此不再增加自由context，而把P120定义中的
`projected error / clearance <= budget`重写为`projected error <= budget × minimum clearance`，训练effective-threshold单调CDF；
但旧四Brier全部回退`1.78%--27.11%`、mean calibration change=`-4.38%`（F145）。根因是trajectory minimum clearance
丢失逐Actor/time的error-clearance配对，因此关闭该压缩。P181进一步采用5个scene-bootstrap低容量单调CDF进行概率平均，
以模型边际化模拟source covariate shifts；但member probability deviation仅约`1.5%`，四cohort Brier变化接近零、mean
calibration change=`-0.25%`（F146），说明同构低维单调模型缺少function diversity。P182据此改为直接拟合continuous
`log1p(boundary cost)`的5-component conditional density，并从解析CDF查询七budget。旧四相对P173 Brier改善
`24.48%/18.46%/16.46%/31.17%`，calibration-error改善`53.06%/60.11%/50.45%/81.07%`，mean=`61.17%`，
2/2 development support。这是迄今同时改善refinement与marginal probability scale的最强结果，但仍不是独立证据。

P183因此在任何新quality read前另冻10-scene/10-log、四location=`3/3/3/1`确认，与P175完全分离；只保留相对P173 mean Brier
和mean calibration各改善10%的两项macro gate。P183最终五H Brier reduction=`17.14%/32.26%/30.54%/31.27%/31.18%`，
macro=`28.48%`；calibration-error reduction=`38.62%/91.68%/79.86%/73.29%/63.46%`，macro=`69.38%`，2/2通过。
这是不同10-scene/10-log的scene-level fresh support。P184仅用已消费旧四训练scene-bootstrap density ensemble以利用等待期GPU，
不读取P183，也没有改变冻结P182/P183 candidate。

P175最终独立结果为五H Brier reduction=`24.60%/33.42%/38.16%/38.41%/36.11%`、mean=`34.14%`，但model/control
macro calibration error=`.07102/.06101`，F147。这将P173论文结论明确限定为“跨scene proper-score discrimination支持，
calibrated probability拒绝”。这也说明P182同时改善Brier与calibration的development结果具有必要性，而不是重复优化同一指标。

P184相对P182 mean calibration进一步改善`20.57%`，且3/4 cohort Brier改善，但P81 Brier回退`2.18%`（F148）；故uniform density
ensemble不升级。P185改为五个连续source-scene environment的worst-NLL训练，mean calibration改善`13.02%`，但P81 Brier仍回退
`2.64%`（F149）；因此source bootstrap/DRO支线关闭。P186只用fixed condition/target noise平滑P182 likelihood，mean calibration虽改善
`19.60%`，但四cohort Brier全部回退`1.63%--27.51%`（F150），说明边际刻度改善来自conditional refinement损失，source-noise
smoothing关闭。P187据heavy-tail CDE文献冻结ν=`3` Student-t component，只检验Gaussian family misspecification；P183 candidate仍保持
P182 single density，任何development rescue均不回流改变prospective protocol。P187最终只改善P81/P113，P96/P129 Brier与calibration
均回退，mean calibration improvement仅`3.85%`（F151），所以“全局加重尾部”被关闭。P188进一步按NeurIPS Neural Spline Flows迁移
固定8-bin conditional RQ spline，以区分重尾不足和随condition变化的偏态/局部形状；仍是consumed-development机制试验。
同时，P189保留P182 Gaussian mixture representation，只把训练目标替换为七个既有预算上的mean Brier（离散CRPS）；它与P188并发，
用于区分distribution-family misspecification和NLL/objective mismatch，且不改变P183冻结的P182 candidate。
P188最终把source NLL降到`-1.39293`，但相对P182只有P96 Brier改善；其余三个回退且mean calibration恶化`23.89%`（F152）。
因此论文可把它作为“density likelihood与decision reliability不等价”的直接负证据，而不是把更低NLL误写为不确定性质量提升。
P189直接优化七预算Brier后，P96/P113 Brier改善`6.94%/9.02%`且mean calibration改善`11.09%`，但P81/P129回退（F153）。
这把问题缩小为refinement与calibration的梯度冲突；P190从P182 checkpoint采用norm-balanced PCGrad联合NLL与Brier，不做loss-weight sweep。
P190最终让P81/P113/P129 Brier改善且mean calibration改善`7.19%`，但P96仍回退`.62%`（F154）；严格拒绝说明仅优化目标不足。
P191因此把P126 crossing ratio解压为aleatoric、ensemble epistemic和projected-mean magnitude三个冻结context proxy，检验condition sufficiency。
P191只改善P113，P129 Brier反而回退`16.70%`（F155），表明这些分量不是稳定shift proxy。P192不再增加feature或loss，转而检验
source sampling measure：102 scenes等权的environment-balanced ERM对比P182按trajectory pooled ERM。P192四cohort Brier均改善
`1.66%/2.87%/5.06%/.80%`且mean calibration改善`16.65%`，因此development支持；但它没有参与P183 frozen candidate选择，仍需
另一个future cohort才能升级为独立结论。P193先在冻结后读取已消费P183 rows作非独立secondary，发现H`.8/1.5s`回退而长H
改善，macro calibration improvement=`-.03%`，故停止P192晋升，不浪费新的fresh cohort。该结果把问题定位为sampling measure
的horizon trade-off。P194全局50/50与P195线性horizon sampler均被拒绝；P196冻结density后以source NLL学习两标量router，
旧四cohort Brier全改善且mean calibration +18.73%，但router退化为55.7%常数pool，并在P197 consumed P183上再次使短H回退。
P198进一步把short/long parameters完全隔离仍在P81/P113回退`4.20%/5.56%`，据此sampling/pooling/expert family终止；
P182仍是唯一fresh-supported marginal density。后续研究转向冻结marginals后的joint-horizon dependence，不再改写P183结论。

## 3. 核心结果表

| 阶段 | 数据角色 | query / Actor / P75 selected events | query / Actor AUROC | 结论 |
|---|---|---:|---:|---|
| P81 | independent all-row endpoint | `26 / 57 / 45` | `.94147 / .92947` | 窄全row triage支持 |
| P86 best | same-read visited-trajectory | `187 / 193 / 199` | 见canonical summary | task-conditioned增量不足 |
| P95 | consumed development occupancy flip | `7 / 28 / 13` | `.83952 / .69366` | development支持 |
| P96 | independent P95 | `8 / 5 / 12` | `.65542 / .71181` | relative反转 |
| P102 | consumed development hierarchical | `4 / 27 / 13` | `.87161 / .68922` | development best |
| P103 | independent P102 secondary | `9 / 7 / 12` | `.74385 / .67973` | relative反转 |
| P107/P81 | consumed scalar uncertainty | `2 / 36 / 13` | `.92901 / .56826` | development支持 |
| P107/P96 | consumed scalar uncertainty | `2 / 9 / 12` | `.87305 / .61786` | development支持 |
| P108 | independent scalar uncertainty | `5 / 35 / 20` | `.95107 / .77605` | primary支持 |
| P110 | same-read directional secondary | `1 / 53 / 20` | `.96027 / .69142` | secondary支持 |
| P111 | same-read clearance-only | `1 / n/a / n/a` | `.91644 / n/a` | 强geometry baseline |
| P114 | consumed downstream tail pool | P81/P96=`0 / 1` | `.95138 / .90298` | reject；均低于P109 max |
| P115 | consumed spectral Actor sequence | P81/P96=`0 / 7` | `.97709 / .84712` | reject；P96强退化 |
| P116 | consumed directional q90 field | P81/P96=`0 / 6` | `.96384 / .88932` | reject；均低于P109 |
| P117 | consumed full-covariance Gaussian | P81/P96=`0 / 0` | `.97254 / .91366` | development support；mean gain `+.00711` |
| P118 | same-checkpoint rho ablation | P81/P96均`0 / 0` | rho gain=`+.00030 / -.00012` | reject direct-rho mechanism |
| P113 | independent directional vs clearance | directional/clearance=`6 / 5` | `.92016 / .87529` | reject composite；AUROC gate pass |
| P119 | consumed ranked-range tail | P81/P96/P113=`0 / 0 / 6` | gain=`-.00384/-.00459/-.00122` | reject tail recovery |
| P120 | consumed continuous boundary cost | P109 selected reduction=`.8975/.7705/.8337` | Spearman=`.8065/.7183/.7921` | base candidate；new head reject |
| P122 | consumed full-cov continuous selection | selected cost=`.1854/.1849/.2255` | Spearman gain=`+.0115/+.0048/+.0120` | reject；P96/P113 cost回退 |
| P123 | consumed continuous rank residual | selected cost=`.1783/.1831/.2241` | Spearman gain=`-.0198/-.0562/+.0082` | reject downstream head |
| P124 | consumed correlated Student-t | P81/P96/P113 events=`0/7/7` | AUROC gain=`+.0002/-.0541/-.0054` | reject uniform heavy tail |
| P125 | consumed K2 Gaussian mixture | P81/P96/P113 events=`0/4/7` | AUROC gain=`-.0021/-.0243/-.0070` | reject learned modes |
| P126 | consumed deep ensemble | P81/P96/P113 events=`0/1/4` | AUROC gain=`+.0020/+.0100/+.0063` | binary composite reject；rank signal retained |
| P127 | consumed ensemble continuous | selected cost全降 | Spearman gain=`+.0470/+.1351/+.0755` | support；freeze P128 |
| P121 | independent continuous primary | selected cost=`.27796`，reduction=`.7736` | Spearman=`.76147`，gain over clearance=`+.28823` | 2/2 support |
| P128 | P121 same-read ensemble secondary | selected cost=`.27051<.27796` | Spearman gain over P109=`+.04721` | support with timing caveat |
| P129 | independent ensemble increment | selected cost=`.30867<.32934` | Spearman gain=`+.04257` | 2/2 scene-level support |
| P147 | independent five-horizon ensemble increment | mean selected-cost delta=`-.01777` | mean Spearman gain=`+.17419` | 2/2 scene-level multi-H support |
| P164 | nearest-neighbor residual adapter over frozen P126 | cost 4/4 regress | mean Spearman gain=`-.04474` | reject marginal interaction context |
| P165 | joint multi-Actor residual diffusion | old cost 1/4 improve；P147 5/5 improve | old mean rank=`+.00811`；P147 all positive | rank mechanism positive, q75 selection reject |

## 4. 失败如何推动研究对象变化

| failure family | 被否定的假设 | 递进迁移 |
|---|---|---|
| `V67-F67` | visited Actor max-error可稳定产生τ-conditioned增益 | 改为真实occupancy-decision flip |
| `V67-F68--F73` | subtype、local auxiliary、层级结构或更多source可修复development shortcut | 保留独立拒绝，不扫参 |
| `V67-F74--F75` | end-to-end occupancy query classifier可独立迁移 | Actor distribution与candidate query解耦 |
| `V67-F76--F77` | launcher与NPZ交付工程事件 | 绝对入口与partial→atomic replace；不加校验门控 |
| `V67-F78` | nonlinear finite-sample crossing必优于linear boundary projection | 保留directional linearized score |
| `V67-F79` | monotone top-k/union tail pooling可提升P109 max | 终局拒绝；后继P115占用F80 |
| `V67-F80` | 低频joint Actor residual sequence可稳定提升P109 | P96反转；保留pointwise directional model |
| `V67-F81` | distribution-free directional q90可超过Gaussian | 两cohort均无增益；保留P109 standardized margin |
| `V67-F82` | inferred session shard可直接定位scene-0003 | pre-target exact locator失败；扫描公开parts恢复 |
| `V67-F83` | P117收益由conditional rho推理项直接产生 | zero-rho消融几乎不变且P96反向 |
| `V67-F84` | 全局AUROC增量可保证fixed50 rare-event优势 | AUROC gain `+.04486`但events `6>5` |
| `V67-F85` | source ranked-range loss可修复fixed50 transfer | P113仍6 events且三cohort AUROC都退化 |
| `V67-F86` | continuous regressor可超过P109 base | P81/P96退化；保留P109 continuous object |
| `V67-F87` | full covariance全局排序增量可保证fixed50 continuous cost不退化 | P96/P113 selected cost略升；不进入P121 secondary |
| `V67-F88` | 稠密continuous operating-range pairs可消除跨cohort selection漂移 | P81/P96 rank退化且P96 cost回退 |
| `V67-F89` | 统一重尾likelihood可稳定改善Actor boundary uncertainty | P96 scale过宽、AUROC与events均退化 |
| `V67-F90` | 显式K2 residual modes可替代单Gaussian并跨cohort迁移 | components active但三cohort AUROC全退化 |
| `V67-F91` | ensemble全局增益可直接保证binary fixed50 noninferiority | P96多1 event；只允许continuous transfer |

## 5. 系统与资源

- 全部训练与推理使用1x RTX 3090；没有多卡需求。
- P107/P109各训练916,722个Actor-time tokens、6,000 steps；P107 wall约30.01s，P109训练与P108归档I/O重叠。
- P108并行流式扫描7个shards、精确提取3,877/3,877 files并预处理10/10 scenes，wall约2,439.39s。
- P112 nonlinear试验只做固定256 samples、seed0的一次read，不做sample/seed/distribution sweep。
- P114在P113归档I/O期间完成6,000-step GPU训练，wall约13.66s，未读取P113 target。
- P115同样在P113 I/O期间完成101,858 Actor sequences的6,000-step GPU训练，wall约36.41s。
- P116训练916,722 Actor-time tokens×随机8-direction q90 queries，6,000 steps，wall约30.07s。
- P117训练916,722 Actor-time tokens的correlated bivariate Gaussian，6,000 steps，wall约45.49s。
- P118冻结checkpoint的conditional-vs-zero-rho消融wall约1.03s，不重训、不读取P113。
- P119 source-only ranked-range GPU训练6,000 steps，wall约43.50s。
- P120 source-only continuous cost regression 6,000 steps，wall约27.30s。
- P122在P121 archive IO期间做冻结checkpoint GPU inference，wall约1.01s；没有训练或新target read。
- P123与同一archive IO重叠训练13,123个continuous pairs、6,000 steps，wall约44.36s。
- P124同样在IO窗口训练916,722 Actor-time tokens、6,000 steps，wall约53.55s。
- P125训练K2 mixture 6,000 steps，wall约63.01s；没有component/entropy sweep。
- P126新训seed1/2各6,000 steps并复用P109 seed0；单卡顺序执行，与P121 IO重叠。
- P127冻结checkpoint推理wall数秒；P128 runner/config内容在P121 rows物化前冻结并复制，Git commit晚于materialization，见timing caveat。
- P121 prep/primary wall=`1829.91/1816.03s`，主要是并行archive scan与等待；P128 inference wall=`.665s`。
- 未新增hash、checksum或fingerprint；没有smoke/regression matrix。

## 6. 有效性与claim边界

本报告支持：

- task-conditioned occupancy-decision reliability应将Actor uncertainty与candidate trajectory geometry因子化；
- scalar q90 tube相对Actor-only/P75在一个scene-level independent cohort成立；
- directional boundary projection在development和一个prospective secondary read上提供强排序；
- full bivariate covariance在两个consumed cohorts上进一步提升directional AUROC，但尚无独立确认；
- clearance-only是必须正视的强baseline，learned uncertainty的独立增量必须由P113单独裁决。
- P113确认了learned score的独立AUROC增量，但拒绝其fixed50 event noninferiority；后续应研究ranked-range/selective-tail
  objective，而不是把全局AUROC当作固定预算tail保证。
- P183在不同10-scene/10-log、五时域确认了continuous-cost conditional density相对P173的proper-score与marginal-reliability增益；
  这是scene-level支持，不是session/population generalization或formal calibration guarantee。
- P192--P197的scene weighting、conditional sampler与frozen-expert pool均在旧development或长H有局部收益，但consumed P183
  短H持续回退；它们构成sampling-measure negative chain，P182仍是唯一fresh-supported density。

本报告不支持：

- session-level或population-level generalization；
- calibrated collision probability、conformal coverage或formal risk bound；
- planner、policy、control、closed-loop或deployment authority；
- safety improvement或false-safe guarantee；
- 用P81/P96等已消费cohort冒充新的独立确认。

## 7. 结论

V6.7最重要的递进结论不是某个更大的query network，而是预测对象和条件化位置的改变。让`τ`直接进入端到端classifier，会在
development中产生很强、但跨cohort不稳定的interaction shortcut；让网络只预测Actor uncertainty，再让`τ`通过明确的boundary
geometry进入解析计算，才获得稳定的scene-level independent evidence。与此同时，clearance-only baseline提醒我们不能把几何贡献
误写成learned uncertainty。进一步把连续boundary-state cost建模为条件分布、而不是分别校准若干二值事件，使P183在不同10-scene/
10-log五时域上同时改善Brier与marginal reliability。当前最稳妥结论是“可查询的visited-state cost distribution获得scene-level fresh
support”；它仍不等价于collision probability、formal coverage或deployment safety authority。

## References

- Yu et al., [Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html), NeurIPS 2020.
- Hayder et al., [DTO-KD: Dynamic Trade-off Optimization for Effective Knowledge Distillation](https://openreview.net/pdf?id=QMItTyQW92), ICLR 2026.

- Chai et al., [MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction](https://proceedings.mlr.press/v100/chai20a.html), CoRL 2019.
- Farid et al., [Task-Relevant Failure Detection for Trajectory Predictors in Autonomous Vehicles](https://proceedings.mlr.press/v205/farid23a/farid23a.pdf), CoRL 2022.
- Weng et al., [Joint Metrics Matter: A Better Standard for Trajectory Forecasting](https://openaccess.thecvf.com/content/ICCV2023/html/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.html), ICCV 2023.
- Rhinehart et al., [PRECOG: PREdiction Conditioned on Goals in Visual Multi-Agent Settings](https://openaccess.thecvf.com/content_ICCV_2019/html/Rhinehart_PRECOG_PREdiction_Conditioned_on_Goals_in_Visual_Multi-Agent_Settings_ICCV_2019_paper.html), ICCV 2019.
- Huang et al., [FoSS: Modeling Long-Range Dependencies and Multimodal Uncertainty in Trajectory Prediction](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_FoSS_Modeling_Long-Range_Dependencies_and_Multimodal_Uncertainty_in_Trajectory_Prediction_CVPR_2026_paper.html), CVPR 2026.
- Kan et al., [Multivariate Quantile Function Forecaster](https://proceedings.mlr.press/v151/kan22a.html), AISTATS 2022.
- Chung et al., [Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification](https://proceedings.neurips.cc/paper/2021/hash/5b168fdba5ee5ea262cc2d4c0b457697-Abstract.html), NeurIPS 2021.
- Casas et al., [Implicit Latent Variable Model for Scene-Consistent Motion Forecasting](https://openaccess.thecvf.com/content/ICCV2023/html/Casas_Implicit_Latent_Variable_Model_for_Scene-Consistent_Motion_Forecasting_ICCV_2023_paper.html), ICCV 2023.
- [Open-source collision-probability estimation with stochastic boundary crossing](https://github.com/TUM-AVS/Collision-Probability-Estimation), 2025.
