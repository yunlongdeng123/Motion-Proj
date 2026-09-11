# 历史原始记录 046

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F113 — worst-group dense cost放大P96反转

- 分类：`algorithm/group-DRO-domain-transfer`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P151-GROUP-DRO-BOUNDARY-COST-01/
  20260830T110000Z__group-dro-boundary-cost-s0-r1`。
- 观察：408 scene×horizon environments；P96 rank gain从P150的`-.028055`恶化至`-.115416`，四cohort mean=
  `-.046683`；仅P81 selected cost改善。
- 解释：worst-quartile NLL重压hard/noisy groups，未提取稳定invariant；direct cost对象的domain-robust训练恢复失败。
- 防重复：关闭direct-cost/scene-DRO family，不扫group fraction或IRM penalty。P152返回P126 Actor residual ensemble并改变prior机制。

### P152 freeze note — randomized function-prior Actor ensemble

- method：3 P109-shaped trainable Gaussian members，各加独立冻结random mean prior，scale固定1；aleatoric scale不来自prior。
- constants：P109 normalization/source、Gaussian NLL、6,000 steps/member、P126 continuous score与四cohort decisions。
- prevention：不扫prior scale/architecture/loss/member/seed/score/coverage；不以P151作teacher或head。

### V67-F114 — randomized function prior制造的差异未改善排序

- 分类：`algorithm/randomized-prior-epistemic-transfer`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P152-RANDOMIZED-PRIOR-ACTOR-ENSEMBLE-01/
  20260830T110500Z__randomized-prior-actor-ensemble-s0-r1`。
- 观察：四cohort cost全退，Spearman gain全负，mean=`-.006852`。
- 解释：冻结random function prior能保持成员差异，但差异方向未对齐continuous boundary cost；diversity不等于useful epistemic。
- 防重复：不扫prior scale/seed/architecture；转P153 exact last-layer feature leverage。

### V67-F115 — exact token posterior在大样本下epistemic过度集中

- 分类：`algorithm/Bayesian-last-layer-underdispersion`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P153-BAYESIAN-LAST-LAYER-ACTOR-01/
  20260830T111000Z__bayesian-last-layer-actor-s0-r1`。
- 观察：P81/P113 rank增、P96/P113 cost改善，但mean rank仅`+.000844`；P81/P129 cost回退。epistemic fraction仅
  `1.2e-4--2.0e-4`，远低于P126 useful member spread。
- 解释：916,722 token Fisher使last-layer posterior高度集中；不按结果事后降低effective N或prior precision。
- 防重复：不扫posterior temperature/prior/effective samples。P154改用source hidden density的parameter-free standardized distance。

### P154 freeze note — hidden-density-aware P126 variance

- density：P109 frozen 128D hidden；4-layer RealNVP、6,000 steps、916,722 source tokens。
- integration：P126 predictions全冻结；variance multiplier=`1+ReLU(source-standardized NLL)`；两decision不变。
- prevention：不扫flow depth/scale/inflation/weight/score/coverage。

### V67-F116 — hidden density shift不能区分低密度可靠性

- 分类：`algorithm/density-aware-OOD-transfer`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P154-DENSITY-AWARE-ACTOR-ENSEMBLE-01/
  20260830T111500Z__density-aware-actor-ensemble-s0-r1`。
- 观察：四cohort mean inflation=`1.84--2.24`，故flow识别到source-support shift；但mean rank=`-.001727`，P81 cost从
  `.176665`恶化至`.196679`，仅P96 operating cost改善。
- 解释：feature rarity不是reliability充分统计量；blind density inflation会把低密度但可靠Actor/query错误提升。
- 防重复：不扫flow/inflation threshold/weight。P155把generalization机制移到same-fraction RegMixup训练，而非test-time OOD分数。

### P155 freeze note — time-fraction matched RegMixup ensemble

- training：原始/Mixup Gaussian NLL各`.5`；pair同time fraction，Beta alpha`.2`，feature+2D target同lambda。
- constants：P126 shape/3 members/6,000 steps/source/score和四cohort decisions。
- prevention：不扫alpha/loss weight/pairing/architecture/member/seed/score/coverage。

### V67-F117 — same-fraction RegMixup未保持P126跨cohort排序

- 分类：`algorithm/train-time-domain-interpolation`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P155-REGMIXUP-ACTOR-ENSEMBLE-01/
  20260830T112000Z__regmixup-actor-ensemble-s0-r1`。
- 观察：仅P129 selected cost改善；四cohort rank gain=`-.010876/-.014890/-.002785/+.000369`，mean=`-.007045`。
- 解释：同time-role convex interpolation仍平滑掉对continuous boundary cost有用的residual tail/order；source interpolation不是P126
  independent transfer的恢复机制。
- 防重复：不扫Mixup alpha/loss weight/pairing；关闭train-time augmentation family，回到P147 fresh multi-horizon evidence。

### V67-F118 — P147 scene0110 shard locator错误（pre-target engineering）

- 分类：`engineering/dataset-archive-routing`；状态：`resolved_before_target_read`。
- evidence：r1 shard01冻结需求774，scan found386；386精确覆盖scene0018，scene0110未命中。scene0110 official index92属于shard02。
- recovery：唯一改`scene-0110 01→02`；复用所有existing LIDAR，r2在02精确found388，总计`3,909/3,909` mapped并完成
  10/10 preprocess，wall=`516.63s`。
- scientific impact：修复前0 target rows/metrics；不换scene/cohort/H/model/score/decision，不重启P147 evaluator。修复后P147
  自动完成并通过2/2 macro decisions，证明locator问题未改变科学协议。

### P156 freeze note — continuous-time integrated increment ensemble

- object：source exact-zero初始residual；8个`Δresidual/(H/8)` velocity Gaussian，以absolute midpoint+H conditioning。
- integration：position mean累加increment，aleatoric variance累加independent increment variance，epistemic取member integrated mean variance。
- decisions：P81/P96/P113/P129相对P126 cost全不退、mean rank gain≥`.005`。
- prevention：不扫architecture/loss/member/seed/integration/variance/score/coverage；不重复P148 direct position decoder。

### V67-F119 — independent increment integration造成全cohort排序退化

- 分类：`algorithm/continuous-time-increment-transfer`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P156-INTEGRATED-INCREMENT-ACTOR-ENSEMBLE-01/
  20260830T112500Z__integrated-increment-actor-ensemble-s0-r1`。
- 观察：四cohort cost全退；rank gain集中为`-.027-- -.032`，mean=`-.028923`；epistemic fraction`.051--.087`。
- 解释：velocity representation具kinematic coherence，但把8个increment variance独立累加导致长期不确定性结构过宽/错序；P148失败
  不是简单改成increment即可恢复。
- 防重复：不扫correlation/integration/variance weight；关闭temporal sequence family，等待P147 frozen P126/P109 primary。

### P157 freeze note — horizon-specialist Actor ensemble

- object：`.8/1.5/2.5/3.0s`分别训练P109-shaped三成员专家，各自normalization；H3.5固定路由H3.0。
- rationale：只检验shared-horizon negative transfer是否压低P126；不引入learned router，也不改变directional boundary query。
- decisions：P81/P96/P113/P129相对P126 cost全不退、mean Spearman gain≥`.005`；失败才使用F120。
- prevention：不扫expert count/routing/architecture/loss/member/seed/score/coverage；P147 primary保持原冻结P126/P109。

### V67-F120 — nearest-lower horizon expert在H3.5严重外推失配

- 分类：`algorithm/horizon-specialist-extrapolation`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P157-HORIZON-SPECIALIST-ACTOR-ENSEMBLE-01/
  20260830T113500Z__horizon-specialist-actor-ensemble-s0-r1`。
- 观察：12个member训练NLL正常，但四个H3.5 consumed cohorts路由H3.0后cost约为P126的3.9--5.8倍；mean Spearman gain=
  `-.594342`，0/2 decisions。
- 解释：完全拆分去掉了shared horizon support，而H3.0 expert的time input/target normalization未覆盖H3.5，nearest-lower
  routing形成真实外推；P147同时证明shared P126在五H独立cohort上全方向支持。
- 防重复：不扫expert count/nearest router或用P147 fresh target训练专家；不把本失败外推为exact-horizon expert普遍无效。

### P158 freeze note — CRPS shared Actor ensemble

- object：保留P126 shared three-member diagonal Gaussian与total variance boundary score，只以closed-form marginal Gaussian CRPS训练。
- decisions：P81/P96/P113/P129相对P126 cost全不退、mean Spearman gain≥`.005`；P147五H仅post-confirmation描述。
- prevention：不扫NLL/CRPS mixture、loss weight、architecture/member/seed/score/coverage；失败才使用F121。

### V67-F121 — marginal CRPS改善P147 rank但破坏跨cohort fixed50 cost

- 分类：`algorithm/proper-scoring-rule-transfer`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P158-CRPS-ACTOR-ENSEMBLE-01/20260830T114500Z__crps-actor-ensemble-s0-r1`。
- 观察：旧四cohort rank mean=`-.023715`且cost全退；P147 post-confirmation五H rank全正，但仅`.8/1.5s` cost改善，
  中长H cost回退随H放大到`+.04723`。
- 解释：marginal CRPS可改善新cohort全局ordering，但不保证固定覆盖率tail operating point；且逐轴CRPS不建模multivariate/
  ensemble-level dependence。P147数字是post-read diagnosis，不是prospective selection证据。
- 防重复：不扫CRPS weight或事后按H切换P126/P158；若继续proper-score路线，只允许joint multivariate ensemble objective。

### P159 freeze note — joint multivariate Energy Score ensemble

- object：三P126-shaped members联合优化`E||X-y||-.5E||X-X'||`，每步两组独立samples；推理score不变。
- decisions：P81/P96/P113/P129相对P126 cost全不退、mean rank gain≥`.005`；P147仍post-confirmation descriptive。
- prevention：不扫Energy/Variogram混合、sample count、weight、architecture/member/seed/score/coverage；失败才使用F122。

### V67-F122 — joint Energy Score进一步放大旧cohort排序退化

- 分类：`algorithm/joint-ensemble-proper-score-transfer`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P159-JOINT-ENERGY-SCORE-ACTOR-ENSEMBLE-01/
  20260830T115000Z__joint-energy-score-actor-ensemble-s0-r1`。
- 观察：旧四cohort cost全退，rank mean=`-.042511`；P147只有`.8/1.5s` cost微降，中长H继续回退。
- 解释：joint multivariate sample distance没有转化为boundary-tail fixed50 authority，且比marginal CRPS在旧cohort退化更大；
  问题不再归因于“CRPS缺少ensemble dependence”。
- 防重复：关闭proper-score training family；不做Energy/Variogram混合、更多samples或NLL+CRPS权重扫。下一步只改frozen P126
  distribution aggregation，不重训同一architecture/loss family。

### V67-F123 — exact member probability linear pool破坏P126 moment-margin排序

- 分类：`algorithm/deep-ensemble-distribution-aggregation`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P160-EXACT-ENSEMBLE-MIXTURE-BOUNDARY-01/
  20260830T115500Z__exact-ensemble-mixture-boundary-s0-r1`。
- 观察：旧四cohort cost全退、rank mean=`-.043302`；P147五H cost全退，短H rank最大下降`-.31838`。
- 解释：平均bounded CDF压缩了far-from-boundary差异，且没有像moment score一样把between-member mean variance显式写入
  standardized margin；“exact mixture”不等于当前selection objective更合适。
- 防重复：保留P126 moment matching；不扫temperature/member weights/log/product/quantile pool，distribution aggregation关闭。

### V67-F124 — P126显式epistemic variance不是独立增益主因

- 分类：`mechanism/epistemic-variance-attribution`；状态：`closed_negative_ablation`。
- canonical：`run://worldsim_v67/WS-V67-P161-EPISTEMIC-VARIANCE-ABLATION-01/
  20260830T120000Z__epistemic-variance-ablation-s0-r1`。
- 观察：置零between-member variance后旧四cohort rank mean只变`+.000090`（full-minus-control=`-.000090`），cost几乎相同；
  P147中长H selected set/cost exact相同。epistemic fraction仅约1.2%--2.6%。
- 解释：P126相对P109的增益来自independent members对mean/aleatoric predictions的averaging，而非explicit epistemic addend。
- 防重复：不调epistemic multiplier/floor；论文禁写“P147证明epistemic uncertainty带来增益”，改写为deep-ensemble moment
  predictor / member averaging，且保留P126/P147现有方法支持。

### V67-F125 — P162 yaw source/eval processed-root解析错误（pre-training engineering）

- 分类：`engineering/processed-scene-routing`；状态：`resolved_r3_before_training`。
- evidence：r1按四位scene id寻找`0001`，实际processed dirs为三位；r2改`001`后发现source rows来自V4 processed root，
  fresh evaluation rows来自V67 root。两次均在构造source yaw entries时退出，0 optimizer step/metric。
- recovery：r3固定按三位scene id依次解析V4、V67两个既有roots；不改yaw target、P126、oriented support、cohort或decisions。
- scientific impact：无方法结果污染；r3已进入3×6,000-step GPU训练。

### P162 freeze note — oriented-footprint Actor reliability

- object：current yaw+yaw-rate forecast的wrapped residual Gaussian；length/width rectangle support沿predicted boundary normal线性传播。
- control：同一oriented predicted clearance与P126 position field，但zero yaw residual；actual cost包含support error。
- prevention：不扫class/box scale/yaw representation/support derivative/score/coverage；失败才使用F126。

