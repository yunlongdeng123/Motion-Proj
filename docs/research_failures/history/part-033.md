# 历史原始记录 033

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F220 — P343补齐插值task strata导致保守化且未改善P201 tail risk

- canonical=`run://worldsim_v67/WS-V67-P343-INTERPOLATED-TASK-WORST-GROUP-AUTHORITY-01/
  20260901T125500Z__interpolated-task-worst-group-authority-s0-r2`；
- symptom：P201 non-anchor coverage `.331239 < P337 .339800`，max nominal-risk excess
  `.004217 > .003571`，仅q90 risk/coverage通过，2/4 rejected；
- retained evidence：q85 unsafe `.074803`很低且q90 exact `.304645/.049020`，但q75 coverage下降、q95
  unsafe `.054217`，39组约束把模型推向保守而没有稳定tail transfer；
- literature/migration：trajectory-conditioned occupancy world models（Drive-OccWorld、UniOcc）直接把ego
  trajectory作为future-state query；P344因此改成visited-state ceiling exceedance probability BCE，而非继续
  调quantile risk warp；
- forbidden rescue：不扫group、dual、margin、interpolation grid、capacity或seed；resolution=
  `closed family, prediction object moved to P344 direct visited-state reliability`。

### V67-F221 — P344全局概率校准无法控制各ceiling的direct reliability风险

- canonical=`run://worldsim_v67/WS-V67-P344-DIRECT-VISITED-RELIABILITY-AUTHORITY-01/
  20260901T131500Z__direct-visited-reliability-authority-s0-r1`；
- symptom：P201 q90 coverage `.280055 < .30`且max unsafe `.120879 > .10`，monotonicity两门通过，2/4；
- heterogeneity：q90 strict/mid/high unsafe=`.120879/.067073/.024675`，全局bias若压strict会进一步压低已经不足的
  mean coverage，而放宽mid/high又不能单独进行；
- literature/migration：NeurIPS 2024 multicalibration与官方empirical implementation支持按subpopulation做
  post-hoc calibration；P345使用预定义ceiling×set-size groups与continuous-H slope；
- forbidden rescue：不扫global temperature/bias、risk threshold、capacity、steps或seed；resolution=
  `closed by P345 group calibration attempt; residual probability-bin issue moved to F222`。

### V67-F222 — P345分组仿射校准仍未校准预测概率区间

- canonical=`run://worldsim_v67/WS-V67-P345-MULTIGROUP-CALIBRATED-VISITED-RELIABILITY-01/
  20260901T133000Z__multigroup-calibrated-visited-reliability-s0-r1`；
- symptom：P201 q90 max unsafe从P344 `.120879`降至`.107143`，但仍高于`.10`；mean coverage从
  `.280055`降至`.278689`，仍低于`.30`，故仅两个monotonicity gates通过，2/4 rejected；
- retained evidence：strict/mid/high q90 unsafe改善为`.107143/.058511/.021390`，说明semantic grouping方向有效，
  但source probability frontier仍显示同一group内部的score-dependent miscalibration；
- literature/migration：AISTATS probability-calibration work将isotonic regression定义为独立校准集上的无参数单调
  piecewise-constant map，NeurIPS verified calibration指出histogram方法的sample-efficiency代价；P346因此使用
  无需预选bin数的PAV，并为其分配独立scene fold2；
- forbidden rescue：不微调temperature/bias/horizon slope、`.10` probability threshold、steps或seed；resolution=
  `open via P346 disjoint-fold isotonic multicalibration`。

### P346 r1 operational note — 仓库搜索路径缺失，不计科学failure

- r1在导入`from scripts...`时抛`ModuleNotFoundError`，发生于模型和quality载入前，0 training；
- 这是V65-F18已知入口环境约束，不改变方法或实验合同；仅设置`PYTHONPATH=.`后以r2重启；
- 未分配新failure id，下一可用编号保持`V67-F223`。

### P346 supported note — PAV关闭P345 decision failure，但不关闭跨分布校准问题

- canonical r2的P201四门通过，故不登记negative-result failure；
- source heldout-H q90 unsafe=`.267108`以及`[0,1]`饱和映射表明独立fold PAV仍对horizon/covariate shift敏感；
- P201 task-condition strict unsafe均不超过`.10`，因此该现象不撤销预注册development verdict，但禁止写成general
  calibration、formal multicalibration或safety guarantee；
- 下一步按AISTATS 2020 covariate-shift calibration，只使用target unlabeled covariates估计importance weights，不读取
  target reliability labels；下一可用failure id保持`V67-F223`。

### V67-F223 — P347 importance weighting因source/target support mismatch退化

- canonical=`run://worldsim_v67/WS-V67-P347-SHIFT-WEIGHTED-ISOTONIC-RELIABILITY-01/
  20260901T140000Z__shift-weighted-isotonic-reliability-s0-r1`；
- symptom：P201 q90 coverage=`.298361 < .30`、max unsafe=`.105882 > .10`，两项monotonicity通过，2/4 rejected；
- cause：domain discriminator final BCE=`6.15e-5`，self-normalized odds范围`.01012--382.42`，5,490 source
  calibration rows的ESS仅`69.62`；importance-weighted PAV实际由极少数source rows支配；
- literature/migration：AISTATS 2020同一工作明确限定importance weighting需要source/target足够接近，并用
  adversarial feature map使分布更不可分；DANN/DomainBed/TLlib提供gradient reversal迁移；P348先对latent做
  target-unlabeled alignment，再恢复unweighted P346 PAV；
- forbidden rescue：不裁剪、温度化或平滑weights，不扫domain classifier/threshold/seed；resolution=
  `open via P348 domain-adversarial reliability representation`。

### V67-F224 — P348无条件domain alignment破坏reliability判别与校准

- canonical=`run://worldsim_v67/WS-V67-P348-DOMAIN-ADVERSARIAL-VISITED-RELIABILITY-01/
  20260901T141500Z__domain-adversarial-visited-reliability-s0-r1`；
- symptom：P201 q90 coverage升至`.373497`，但max unsafe升至`.173228`；只有risk gate失败，3/4 rejected；
- diagnosis：domain BCE=`.676887`接近随机，说明alignment确实降低domain可分性；同时source calibration BCE从P346
  `.283113`恶化到`.406183`，说明encoder也抹平了与unsafe event相关的多模态结构，而非alignment没发生；
- literature/migration：NeurIPS 2018 CDAN指出unconditional adversarial alignment不适合分类的多模态分布，提出
  feature与classifier prediction的multilinear conditioning；P349将latent×predicted set-risk输入domain head；
- forbidden rescue：不减GRL coefficient或回调capacity/steps，不将高coverage包装成成功；resolution=
  `open via P349 conditional domain adversarial reliability`。

### P349 supported-but-unstable note — 不把endpoint pass包装为稳定训练

- canonical P349的P201 q90 coverage/risk=`.300546/.093750`且两项monotonicity=0，按预注册门为supported；
- 但step6k后domain BCE升到`4.76--6.10`，final=`3.311811`，base BCE也从约`.18`退化到`.255622`；
- 不选择5.5k checkpoint、不缩短steps、不调GRL；参照CDAN+E官方实现，P350统一引入entropy conditioning与
  warm-start reversal，检验能否保留endpoint frontier并消除late collapse；
- 该限制不登记negative failure，下一可用编号保持`V67-F225`。

### V67-F225 — P350稳定CDAN+E仍未保留unsafe-event语义

- canonical=`run://worldsim_v67/WS-V67-P350-ENTROPY-WARMSTART-CDAN-RELIABILITY-01/
  20260901T144500Z__entropy-warmstart-cdan-reliability-s0-r1`；
- symptom：P201 q90 coverage=`.299727 < .30`、max unsafe=`.186667 > .10`，两项monotonicity通过，2/4；
- retained evidence：final domain BCE=`.679949`且无P349 late explosion，base BCE=`.151812`，说明CDAN+E解决
  optimization collapse；失败来自稳定但错误的target frontier，不是训练发散；
- literature/migration：CVPR 2022 Adaptive Teacher与CVPR 2023 consistency teacher方法用teacher/student consistency
  约束domain-adversarial student的判别语义；P351冻结P346并在source+target-unlabeled上锚定soft risk predictions；
- forbidden rescue：不选P349/P350中间checkpoint、不改entropy/GRL或gate；resolution=
  `open via P351 teacher-anchored CDAN+E`。

### V67-F226 — P351 soft teacher anchor不足以恢复P346 risk frontier

- canonical=`run://worldsim_v67/WS-V67-P351-TEACHER-ANCHORED-CDAN-RELIABILITY-01/
  20260901T150000Z__teacher-anchored-cdan-reliability-s0-r1`；
- symptom：P201 q90 coverage=`.345628`充足，但max unsafe=`.135135 > .10`，3/4 rejected；
- retained evidence：相对P350 unsafe `.186667→.135135`且group calibration BCE `.325300→.301715`，teacher
  anchor方向有效；但soft consistency BCE末端`.186978`仍允许CDAN student偏离P346 calibrated frontier；
- literature/migration：NeurIPS 2019 uncertainty-under-shift与NeurIPS 2021/UAI 2022 calibrated ensembles支持
  model marginalization和pool-then-calibrate；P352停止target adaptation，改用3个source scene-fold heads；
- forbidden rescue：不增加teacher weight、不硬拷贝P346 endpoint、不调CDAN；resolution=
  `closed target-unlabeled adaptation family; open P352 crossfold ensemble`。

### V67-F227 — P352 mean probability ensemble放大authority但未迁移风险控制

- canonical=`run://worldsim_v67/WS-V67-P352-CROSSFOLD-ENSEMBLE-VISITED-RELIABILITY-01/
  20260901T151500Z__crossfold-ensemble-visited-reliability-s0-r1`；
- symptom：source fold5 q90 coverage/unsafe=`.332778/.090164`，但P201=`.395355/.175182`；P201仅risk失败，3/4；
- diagnosis：三个成员均收敛且PAV in-fold BCE=`.271313`，mean pooling明显提高coverage；成员均值消除了epistemic
  disagreement，导致shifted strict-ceiling样本仍被准入；
- literature/migration：NeurIPS 2024 Credal Deep Ensembles以概率上下界保留epistemic uncertainty；P353冻结成员，
  用maximum unsafe probability作为upper pool再source-only校准；
- forbidden rescue：不改member数/seed/folds、不选择weighted mean或事后pool quantile；resolution=
  `open via P353 upper-unsafe probability pooling`。

### V67-F228 — P353 upper pool改善shift risk但pool后校准仍越过硬风险线

- canonical=`run://worldsim_v67/WS-V67-P353-UPPER-POOL-CROSSFOLD-RELIABILITY-01/
  20260901T153000Z__upper-pool-crossfold-reliability-s0-r1`；
- symptom：P201 q90 coverage=`.357650`充分，但max unsafe=`.131148 > .10`；两项monotonicity通过，3/4 rejected；
- retained evidence：相对P352 mean-pool，q90 unsafe从`.175182`降到`.131148`，同时coverage仅从`.395355`降到
  `.357650`；source fold5 q90=`.306423/.077568`。成员上包络确实提供有效保守性，并非无效方向；
- diagnosis：P353先对raw unsafe取maximum，再用单一fold3 affine map和fold4 PAV重新拟合pooled score；该共同映射
  重新吸收source平均频率，无法保持各成员在shift样本上的独立高风险证据；
- literature/migration：NeurIPS 2021 *Uncertainty Quantification and Deep Ensembles*明确比较individual-calibrate-
  then-pool与pool-then-calibrate，并表明顺序改变最终校准；P354为每个冻结成员单独source-calibrate，再取maximum；
- forbidden rescue：不调risk gate、pool quantile、member/fold/seed或PAV bins；resolution=
  `open via P354 memberwise-calibrated upper pooling`。

### V67-F229 — P354 memberwise upper calibration控制风险但过度拒绝

- canonical=`run://worldsim_v67/WS-V67-P354-MEMBERWISE-CALIBRATED-UPPER-POOL-RELIABILITY-01/
  20260901T154500Z__memberwise-calibrated-upper-pool-reliability-s0-r1`；
- symptom：P201 q90 max unsafe=`.047945`显著低于`.10`，但mean coverage=`.244536 < .30`；3/4 rejected；
- retained evidence：相对P353，风险`.131148→.047945`且四个q的unsafe均低于`.088`，说明memberwise calibration
  成功保留了成员高风险证据；失败来自authority过少，不是risk transfer失败；
- diagnosis：对每个样本/ceiling要求三个independently calibrated members全部低于`.10`，把member disagreement
  全部当成同等强度的epistemic danger；source fold5 q90 risk仅`.007586`也显示大量未用风险预算；
- literature/migration：NeurIPS 2023分析deep ensemble selective classification收益集中在top-ambiguity samples；
  NeurIPS 2025进一步指出单调rescaling不能修复ranking，需feature-aware/non-monotone calibrator；P355用source-only
  learned disagreement calibrator重排，而非在P353/P354之间手选常数权重；
- forbidden rescue：不降低coverage gate、不改q90 threshold、不扫mean/max convex coefficient；resolution=
  `open via P355 feature-aware ensemble selective reliability`。

### V67-F230 — P355 feature-aware BCE恢复coverage但未学习选择性条件风险

- canonical=`run://worldsim_v67/WS-V67-P355-FEATURE-AWARE-ENSEMBLE-SELECTIVE-RELIABILITY-01/
  20260901T160000Z__feature-aware-ensemble-selective-reliability-s0-r1`；
- symptom：P201 q90 coverage=`.409290`，但max unsafe=`.205479 > .10`；只有risk gate失败，3/4 rejected；
- source evidence：feature BCE/PAV=`.271707/.273126`，但source fold5 q90 unsafe也为`.104854`，说明endpoint错配
  已在source出现，不需要用target shift解释全部失败；
- diagnosis：BCE优化所有样本的平均proper score，PAV只做单调重标定；两者都没有优化最终largest-feasible-set
  选择后unsafe/coverage比率，feature-aware MLP把排序能力主要用于恢复authority；
- literature/migration：NeurIPS 2020 PACC给出约束学习primal-dual形式；AISTATS 2021 one-sided prediction直接寻找
  false-positive受控的最大decision set；ICLR 2021显示group DRO可改善worst-group selective behavior。P356显式训练
  source task×ceiling worst-group conditional risk；
- forbidden rescue：不缩P355 width/steps、不加class weight、不调PAV或q90 threshold；resolution=
  `open via P356 worst-group selective-risk primal-dual calibration`。

