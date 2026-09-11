# 历史原始记录 044

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F93 — moment-matched Gaussian蒸馏未保持P113 selection boundary

- 分类：`algorithm/distillation-object`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P130-ENSEMBLE-DISTRIBUTION-DISTILLATION-01/
  20260830T091000Z__ensemble-distribution-distillation-s0-r1`。
- 观察：final Gaussian KL=`.073871`；三cohort mean Spearman difference=`-.002024`满足retention，P81/P96 selected cost改善，
  但P113 `.225324`高于P126 `.218791`，因此cost nonregression失败。
- 解释：全局moment KL可保留整体排序，却没有优先保存fixed50 decision boundary附近的teacher function；不是GPU/容量故障。
- 防重复：不扫KL权重、student width/depth或seed；后续P131改变预测对象为task-conditioned boundary score functional
  distillation，而不是P130调参恢复。

### P131 freeze note — task-conditioned functional score distillation

- method：冻结P126 row boundary score作为teacher，单query MLP读取既有query features、signed-clearance profile和boundary
  normals，以一次6,000-step Smooth-L1直接拟合teacher function；只用source和consumed P81/P96/P113。
- decisions/outcome：相对P126三cohort selected cost nonregression与mean Spearman difference≥`-.005`；实际mean Spearman
  difference=`-.362629`且三组cost全退化，两门全失败；P129 rows隔离。
- prevention：seed/architecture/loss/input/coverage一次冻结；关闭direct pointwise functional student，不做蒸馏sweep。

### V67-F94 — pointwise teacher-score回归未保持trajectory max排序

- 分类：`algorithm/supervision-granularity`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P131-TASK-CONDITIONED-SCORE-DISTILLATION-01/
  20260830T091500Z__task-conditioned-score-distillation-s0-r1`。
- 观察：row Smooth-L1降至`.006349`，但trajectory-max三cohort Spearman=`.380515/.436117/.669918`，mean relative
  ensemble=`-.362629`；selected cost三组全面退化，0/2 decisions。
- 根因：source中大量普通row主导pointwise loss，小tail误差经max放大并改变每条trajectory的代表row；训练层级与部署聚合错配。
- 防重复：不扫Huber/temperature/width/seed或加权tail；P132把trajectory max放入训练图并直接优化同scene pair ordering。

### P132 freeze note — trajectory-max rank distillation

- method：P131相同inputs/MLP，但先对trajectory rows取student max，再用同source scene内uniform pairs与P126 teacher order做
  pairwise logistic；6,000 steps、pair batch4096、seed0一次，无temperature/top-k/pointwise auxiliary。
- decisions/outcome：相对P126三cohort selected cost nonregression、mean Spearman difference≥`-.005`；实际mean Spearman
  difference=`-.020183`且三组cost均回退，两门全失败；P129 rows隔离。
- prevention：关闭single-query distillation family，不做ranking-loss sweep。

### V67-F95 — aggregation-aligned rank student仍未保留deep-ensemble增量

- 分类：`algorithm/teacher-compression-capacity`；状态：`closed_negative_after_aggregation_aligned_trial`。
- canonical：`run://worldsim_v67/WS-V67-P132-TRAJECTORY-RANK-DISTILLATION-01/
  20260830T092000Z__trajectory-rank-distillation-s0-r1`。
- 观察：pairwise logistic降至`.117696`，三cohort Spearman恢复到`.829--.851`，远好于P131；但相对P126 mean仍
  `-.020183`，selected cost三组全回退，0/2 decisions。
- 解释：监督层级修复了pointwise→max错配，却无法让单query function重建independent Actor member disagreement；真正增量
  更可能需要原生多成员表示，而非teacher score近似。
- 防重复：P130 moment、P131 pointwise function、P132 trajectory ranking三种compression object均关闭；不扫loss/temperature/
  width/seed。P133转向一次native BatchEnsemble。

### P133 freeze note — rank-one native efficient ensemble

- method：3个BatchEnsemble members共享每层weight，保留member-specific rank-one input/output factors与独立bootstrap；
  diagonal Gaussian NLL、total variance projection、source与continuous evaluation均继承P126/P127。
- decisions/outcome：相对P126三cohort selected cost nonregression、mean Spearman difference≥`-.005`；实际mean Spearman
  difference=`-.014545`，P81/P113 cost回退，两门全失败；P129 rows隔离。
- prevention：固定3 members/seed0/factor init/6,000 steps，不扫rank/member/structure；rank-one shared-weight trial关闭。

### V67-F96 — BatchEnsemble shared weights压缩了epistemic diversity

- 分类：`algorithm/embedded-ensemble-diversity`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P133-BATCHENSEMBLE-ACTOR-UNCERTAINTY-01/
  20260830T092500Z__batchensemble-actor-uncertainty-s0-r1`。
- 观察：NLL=`-3.69116`良好，但三cohort projected epistemic fraction仅`.13%--.34%`，明显低于P126的约`2.2%--2.6%`；
  mean Spearman difference=`-.014545`，P81/P113 selected cost回退。
- 外部证据/解释：2026 controlled BatchEnsemble study报告rank-one members在function/parameter space近似相同；本结果同向，
  表明shared-weight collective regime而非训练不足。
- 防重复：不扫factor init/rank/member/seed；P134只作一次独立block packed regime，若仍失败则不再追efficient ensemble结构。

### P134 freeze note — packed independent member blocks

- method：3套独立weights/biases用member batched kernels在单graph执行，各自bootstrap；不共享P133 weights/factors，保留
  P126三成员容量与FLOPs。
- decisions/outcome：相对P126三cohort selected cost nonregression、mean Spearman difference≥`-.005`；实际rank mean=`+.001874`
  通过，P81/P113 cost改善，但P96 `.172184>.167572`，cost gate失败；P129 rows隔离。
- prevention：不扫packing/group/width；只允许P135一次P126 per-member compute-parity recovery。

### V67-F97 — reduced per-member packed budget未保持P96 selection cost

- 分类：`algorithm-compute/ensemble-training-budget`；状态：`active_single_compute_parity_recovery`。
- canonical：`run://worldsim_v67/WS-V67-P134-PACKED-INDEPENDENT-ACTOR-ENSEMBLE-01/
  20260830T093000Z__packed-independent-actor-ensemble-s0-r1`。
- 观察：independent blocks恢复epistemic fraction到`1.46%--2.23%`，mean rank delta=`+.001874`且P81/P113 cost改善；
  唯一失败为P96 cost `.172184>.167572`。final NLL=`-3.56197`弱于P126 members。
- 预算根因候选：P134为了aggregate batch parity设每member 21,845，而P126每member 65,536；6,000 steps下每member sample
  exposure只有1/3，不能把剩余差异直接归因于packed representation。
- 唯一恢复：P135仅把member batch改到65,536做compute parity；结构/seed/steps/score/decisions不变。不论结果不再扫budget。

### P135 freeze note — full-budget packed compute parity

- method：exact P134 runner与三独立blocks；唯一变化member batch `21845→65536`，匹配P126 per-member 6,000-step exposure。
- decisions/outcome：仍为三cohort cost nonregression与mean Spearman difference≥`-.005`；实际mean rank=`-.001273`通过，
  但三组cost均略回退；P129 rows隔离。
- prevention：packed route关闭；不再扫batch/steps/member/seed/packing。

### V67-F98 — full per-member compute仍未保持P126 fixed50 cost boundary

- 分类：`algorithm/ensemble-solution-variance`；状态：`closed_negative_after_compute_parity`。
- canonical：`run://worldsim_v67/WS-V67-P135-FULL-BUDGET-PACKED-ACTOR-ENSEMBLE-01/
  20260830T093500Z__full-budget-packed-actor-ensemble-s0-r1`。
- 观察：compute parity后NLL=`-3.62334`、epistemic fraction=`1.51%--2.63%`、mean rank difference=`-.001273`均接近
  P126；但P81/P96/P113 selected cost分别高`.003705/.001170/.004970`，cost gate全回退。
- 解释：P134的主要rank差异确受budget影响，但fixed50 selection对independent solution/seed边界敏感；不能把near-parity
  重写成noninferiority成功。
- 防重复：不继续packed seed/budget/member sweep。P136换成single-path cyclic snapshots，检验低成本posterior path samples。

### P136 freeze note — cyclic snapshot Actor ensemble

- method：一个P109结构训练6,000 steps，3个固定2,000-step cosine cycles，LR `.001→.00001`，只存2000/4000/6000；
  三snapshot按total variance评分。
- decisions/outcome：相对P126三cohort selected cost nonregression、mean Spearman difference≥`-.005`；实际mean rank=
  `-.008549`，P96/P113 cost回退，两门全失败；P129 rows隔离。
- prevention：固定cycles/LR/snapshot/seed；snapshot first trial关闭，不扫schedule。

### V67-F99 — cyclic snapshots未形成可迁移的P96 functional diversity

- 分类：`algorithm/single-path-mode-diversity`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P136-SNAPSHOT-ACTOR-ENSEMBLE-01/
  20260830T094000Z__snapshot-actor-ensemble-s0-r1`。
- 观察：三个cycle NLL逐步改善；P81 cost/rank优于P126，但P96 rank差`-.026964`且P96/P113 costs回退，mean rank=
  `-.008549`，0/2 decisions。
- 解释：同一路径cycle endpoints未覆盖独立seed的function modes；最后snapshot本身也尚弱于P126 members。
- 防重复：不扫cycle length/LR range/snapshot count。P137改为SWAG covariance而非挑snapshot。

### P137 freeze note — low-rank-plus-diagonal weight posterior

- method：单P109路径6,000 steps；4000后固定LR `.0001`、每100 steps收集20 iterates；拟合SWAG diag+low-rank
  covariance并以seed137采3 models。
- decisions/outcome：相对P126三cohort selected cost nonregression、mean Spearman difference≥`-.005`；实际mean rank=
  `+.002315`通过，P113 cost改善，但P81/P96微回退；P129 rows隔离。
- prevention：固定collection window/LR/rank/sample count/seed；single-path posterior route关闭。

### V67-F100 — SWAG近似rank保留但未满足逐cohort cost nonregression

- 分类：`algorithm/approximate-posterior-boundary`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P137-SWAG-ACTOR-ENSEMBLE-01/
  20260830T094500Z__swag-actor-ensemble-s0-r1`。
- 观察：20 iterates/3 samples使mean rank delta=`+.002315`，P113 cost改善；但P81/P96 costs分别高`.002115/.000274`，
  strict nonregression失败。
- 解释：low-rank weight posterior比cycle snapshots更接近P126，但fixed sampling仍不能复制每cohort selection boundary；
  不是用更多samples/调collection LR的授权理由。
- 防重复：不扫posterior scale/rank/sample/seed。P138改变uncertainty family为full-cov aleatoric+epistemic deep ensemble。

### P138 freeze note — full-covariance deep ensemble

- method：P117 seed0 + 同协议seed1/2；total projected covariance=mean member full covariance + variance member means。
- decisions/outcome：相对P126三cohort selected cost nonregression且mean Spearman gain≥`.005`；实际mean gain=`+.003590`，
  P96 rank/cost回退，两门全失败；P129 rows隔离。
- prevention：不扫correlation/member/weight/projection；full-cov ensemble first trial关闭。

### V67-F101 — full-cov aleatoric+epistemic ensemble仍在P96反转

- 分类：`algorithm/covariance-transfer`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P138-FULL-COVARIANCE-DEEP-ENSEMBLE-01/
  20260830T095000Z__full-covariance-deep-ensemble-s0-r1`。
- 观察：P81/P113 rank/cost改善，P96 rank gain=`-.004597`、cost `.170009>.167572`；mean gain=`+.003590<.005`，
  0/2 decisions。新member NLL显著收敛，不是训练入口失败。
- 解释：local XY correlation对不同cohort方向不一致；within-member structured covariance不能自动解决source weighting/transfer。
- 防重复：不扫correlation parameterization/weight/member/seed。P139保持diagonal P126结构，仅改uniform scene sampling。

### P139 freeze note — uniform source-scene sampling

- method：三diagonal members完全匹配P126，只把global token-uniform改为scene-uniform→token-uniform；scene不是semantic domain label。
- decisions/outcome：相对P126三cohort selected cost nonregression、mean Spearman gain≥`.005`；P129 rows隔离。实际三cohort
  cost全回退，Spearman gain=`-.009921/-.013609/-.014711`，mean=`-.012747`，两门全失败。
- prevention：不加GroupDRO/Fishr penalty、不扫scene weights/subsets；失败登记F102并关闭simple balancing route。

### V67-F102 — uniform source-scene sampling一致削弱continuous ordering

- 分类：`algorithm/source-sampling`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P139-SCENE-BALANCED-DEEP-ENSEMBLE-01/
  20260830T095500Z__scene-balanced-deep-ensemble-s0-r1`。
- 观察：P81/P96/P113 selected cost=`.180687/.173277/.232300`均高于P126；mean Spearman gain=`-.012747`。
  三member NLL均收敛，失败不是launcher或训练中断。
- 解释：对小scene过采样改变了Actor residual训练分布，但没有提供semantic group信息，反而丢失自然token distribution的有效统计。
- 防重复：关闭uniform scene weighting，不扫scene权重/penalty。P140只迁移bootstrap unit以增加member diversity，并恢复scene内
  natural token weighting。

