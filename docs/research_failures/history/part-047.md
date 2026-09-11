# 历史原始记录 047

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F126 — yaw uncertainty一阶footprint传播无稳定selection增量

- 分类：`algorithm/oriented-footprint-yaw-linearization`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P162-ORIENTED-FOOTPRINT-ACTOR-ENSEMBLE-01/
  20260830T121000Z__oriented-footprint-actor-ensemble-s0-r3`。
- 观察：训练NLL正常且yaw MAE随H增长，但旧四cohort rank mean=`-.000284`、仅2/4 cost改善；P147 H3.5 cost回退`.02613`。
- 解释：yaw residual是可预测状态，但rectangle support包含absolute trigonometric非线性；以predicted heading处的一阶导数传播
  Gaussian不能稳定捕获turning/axis-switch边界，且position residual仍主导cost。
- 防重复：不扫yaw scale/class/box inflation或导数权重。若恢复，只允许直接预测actual-minus-predicted support residual，
  不重复yaw Gaussian linearization。

### P163 freeze note — direct query-normal footprint support residual

- object：exact actual-minus-predicted rectangle support，input=Actor/time/query normal/predicted heading sincos，三成员Gaussian。
- control/decisions：冻结P126 position field；同一oriented clearance position-only control；旧四cohort cost全不退+mean rank≥`.005`。
- prevention：不扫class/box scale/model/loss/score/coverage；P147 post-confirmation only。

### V67-F127 — direct footprint support residual无稳定trajectory-selection增量

- 分类：`algorithm/direct-oriented-support-residual`；状态：`closed_negative_after_single_recovery`。
- canonical：`run://worldsim_v67/WS-V67-P163-DIRECT-FOOTPRINT-SUPPORT-ENSEMBLE-01/
  20260830T121500Z__direct-footprint-support-ensemble-s0-r1`。
- 观察：5.18M tokens、三成员NLL正常；旧四cohort rank gain mean=`-.001156`，P96 cost回退，0/2 decisions；P147五H
  也没有稳定rank/cost方向。
- 解释：直接support residual本身幅度很小（P147 mean absolute约`.0020--.0117m`），position residual及clearance主导
  当前continuous cost；移除yaw一阶近似仍不能提供可迁移的排序信息。
- 防重复：关闭yaw/box-scale/support-loss/normal-conditioned footprint变体，不以局部H微增恢复结论。下一步只允许改变
  Actor position reliability的条件表示或迁移到新预测对象，不再修补footprint支线。

### P164 freeze note — nearest-neighbor interaction residual over frozen P126

- object：保持P126二维position residual与continuous boundary compiler，只增加same-anchor最近8 Actor关系条件。
- isolation：P126 member完全冻结；adapter zero-init且只输出mean/log-scale residual，避免把全模型重训差异归因为interaction。
- decisions：旧四cohort相对P126 cost全不退+mean rank gain≥`.005`；P147 post-confirmation only。
- prevention：不扫neighbor count/graph radius/attention width/adapter/loss/score/coverage；算法失败才使用F129并关闭当前interaction adapter。

### V67-F128 — P164非登录launcher缺仓库根目录（pre-training engineering）

- 分类：`engineering/python-entrypoint`；状态：`resolved_before_data_and_training`。
- evidence：首次远端命令在import `motion_proj`时退出，0 run directory、0 data read、0 optimizer step、0 metric。
- recovery：只增加进程级`PYTHONPATH=.`并以相同config/run id启动；科学对象、cohort、model与decisions不变。
- 防重复：后续远端research launcher显式带repo-level `PYTHONPATH=.`；不把入口错误计为算法trial。

### V67-F129 — nearest-neighbor marginal interaction adapter跨cohort全面退化

- 分类：`algorithm/interaction-conditioned-marginal-position`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P164-INTERACTION-CONTEXT-ACTOR-ENSEMBLE-01/
  20260830T123000Z__interaction-context-actor-ensemble-s0-r1`。
- 观察：三adapter NLL约`-5.0`且source收敛；旧四cohort rank gain mean=`-.04474`、cost 4/4回退，P147五H也rank/cost全退。
- 解释：same-anchor neighbors对source residual likelihood有强解释力，但其交通构成/关系模式跨scene漂移；把context注入每个Actor
  marginal mean/scale破坏了P126已经稳定的局部运动学表示。结果不否定joint multi-Actor forecasting，只否定当前marginal adapter。
- 防重复：不扫neighbor count、distance radius、attention width或解冻P126；若继续相关结构，只能改变预测对象为joint event/
  joint residual dependency，或转入新独立证据，不再重训marginal context adapter。

### P165 freeze note — joint multi-Actor diffusion around frozen P126 marginals

- object：same scene/horizon/anchor Actor set的9-step standardized residual innovation联合分布；P126 mean/scale冻结。
- compiler：16 joint samples、8 DDIM steps，直接计算同定义continuous boundary cost q75；不复用P149 any-crossing proxy。
- decisions：旧四cohort相对P126 cost全不退+mean rank gain≥`.005`；P147 post-confirmation only。
- prevention：不扫diffusion/sample/q75/max actors/architecture/loss/coverage；算法失败才使用F130并关闭joint diffusion trial。

### V67-F130 — joint diffusion rank一致改善但q75 fixed50 cost未跨旧cohort不退

- 分类：`algorithm/joint-residual-operating-point`；状态：`closed_composite_negative_with_positive_rank_signal`。
- canonical：`run://worldsim_v67/WS-V67-P165-JOINT-MULTI-ACTOR-DIFFUSION-01/
  20260830T124000Z__joint-multi-actor-diffusion-s0-r1`。
- 观察：旧四cohort rank gain全正、mean=`+.00811`，rank门通过；P81/P96/P129 selected cost小幅回退，non-regression门失败。
  P147五H rank/cost全部改善，但它们已被primary读取，只能作post-confirmation diagnosis。
- 解释：joint samples比marginal adapter更稳定地刻画相对风险，但sampled cost q75与per-scene fixed50 cutoff仍有operating-point
  mismatch；这不是P164式整体interaction shortcut，也不足以恢复candidate。
- 防重复：不扫q50/q90/sample count/DDIM steps/coverage或用P147选择quantile；保留“9/9 rank slices positive”为机制结果，
  关闭当前q75 joint-diffusion selection candidate。后续不得以放宽cost gate进入独立确认。

### P166 freeze note — monotone expected-cost calibration of frozen P126 rank

- object：`E[P120 continuous cost | frozen P126 score,H]`，不是P165 quantile recovery或selection head。
- model/control：5 fixed knots的positive-increment monotone spline vs horizon-only linear calibration；source only training。
- decisions：旧四cohort MSE逐组不退+mean reduction≥20%；P147 post-confirmation only。
- prevention：不扫knots/bin/loss/architecture/metric，不改变P126 rank/coverage；失败才使用F131并关闭point-calibration trial。

### V67-F131 — monotone expected-cost calibration只有约4% MSE改善且bin error恶化

- 分类：`algorithm/expected-cost-calibration`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P166-MONOTONE-EXPECTED-COST-CALIBRATION-01/
  20260830T125000Z__monotone-expected-cost-calibration-s0-r1`。
- 观察：旧四raw MSE逐组小幅下降但mean reduction仅`4.09%<20%`；10-bin expected-cost error全高于horizon-only；
  P147 H0.8 MSE退化`7.68%`。
- 解释：单调score能保留rank并解释少量均值变化，但P120 cost强重尾、跨scene scale漂移；source log-cost fit不等于raw expected-cost
  calibration。结果不否定P126 ranking，只禁止把它写成calibrated magnitude。
- 防重复：不扫knots/hidden/loss/bin或改用更低MSE门；关闭point expected-cost calibration。若未来做interval/conformal，必须
  作为新预测对象且用新独立校准数据，不能复用本结果降门槛。

下一可用编号为：`V67-F132`。

### P167 freeze note — second scene-level multi-horizon confirmation with live IO→GPU pipeline

- candidate：完全冻结P126、P109与P147五时域continuous score/cost定义；本阶段不训练、校准或修改模型。
- unread cohort：`0269/0346/0968/0524/0557/0904/0802/0928/0930/1065`；四location、9 distinct logs，全部在
  sensor/target read前从official val metadata选定。因log overlap，独立性只到scene level。
- execution：shard完成释放scene preprocessing，scene marker释放GPU scoring；允许与剩余archive IO重叠，不等待全cohort ready。
- decisions：五个H的mean rank gain `>=.005`且mean selected-cost delta `<=0`；不增加子群门或回归矩阵。
- prevention：只允许target read前修正exact archive locator；不换scene、删scene、改H/score/cost/coverage/decision，失败才使用
  `V67-F132`并保留完整per-H负结果；不加hash/checksum/fingerprint。

下一可用编号仍为：`V67-F132`。

### P168 freeze note — coherent upper-tail mean over frozen P165 joint samples

- reason：P165的单个q75 order statistic保留rank但selection operating point不稳；coherent risk文献支持用完整upper tail而非单点VaR。
- compiler：16个冻结joint samples中最高4个P120同定义cost的均值；`.75`从P165继承，非事后quantile选择。
- decisions：旧四cost全不退+mean rank gain≥`.005`；通过后才读P167 prospective secondary，并复用同两门。
- prevention：不训练/解冻P165、不扫alpha/sample/DDIM/coverage/cost或decision；development失败才使用`V67-F132`并立即关闭。

### V67-F132 — coherent upper-tail mean仍使旧四fixed50 selected cost全部回退

- 分类：`algorithm/joint-sample-risk-operating-point`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P168-JOINT-TAIL-MEAN-COMPILER-01/
  20260830T131000Z__joint-tail-mean-compiler-s0-r1`。
- 观察：旧四rank gain mean=`+.00460<.005`，且selected-cost delta=`+.01020/+.00153/+.00208/+.00924`，0/2 decisions。
- 解释：CVaR式tail aggregation减少单个order statistic噪声，但P165 joint sample的global ordering变化仍没有对齐scene内fixed50 cutoff；
  问题不是只需把q75换成另一个手工risk functional。
- 防重复：不扫alpha/quantile/sample/DDIM；按冻结规则未读取P167 prospective rows。若继续，只能直接训练scene/coverage-conditioned
  selection objective，而不是继续手选sample statistic。

下一可用编号为：`V67-F133`。

### P169 freeze note — direct scene-list soft fixed50 training

- migration：F132表明global rank与scene cutoff错配；P169复用P144 representation/anchor，只把pairwise surrogate换成soft selected-cost。
- training：source-only 16 scenes×128 list，median/MAD detached，temperature `.20`，6,000 steps，residual penalty `.10`。
- decisions：旧四cost全不退+mean rank gain≥`.005`；通过才读P167 prospective secondary。
- prevention：不扫temperature/list/model/bound/loss/coverage；algorithm failure才用`V67-F133`并关闭direct soft-cutoff trial。

### V67-F133 — direct soft fixed50训练只回到P126邻域且P96仍微回退

- 分类：`algorithm/fixed-coverage-residual-transfer`；状态：`closed_negative_after_controlled_objective_change`。
- canonical：`run://worldsim_v67/WS-V67-P169-SOFT-FIXED-COVERAGE-COMPILER-01/
  20260830T131500Z__soft-fixed-coverage-compiler-s0-r1`。
- 观察：3/4 cohort selected cost微降，但P96 delta=`+.000306`；mean rank gain=`+.00212<.005`，0/2 decisions。
- 解释：direct scene-list objective修复了P144 pairwise surrogate的大幅错配，但最优残差接近0；P126强anchor之外的token pattern未稳定迁移。
- 防重复：不扫temperature/list/residual bound/architecture或放宽门；未读取P167。关闭P126-anchored learned selection residual family。

下一可用编号为：`V67-F134`。

### P170 freeze note — split-conformal one-sided continuous-cost upper bound

- object：q90 upper bound of P120 cost conditioned on frozen P126 score+horizon；horizon-only同目标control。
- split/training：source scene `%5==0` calibration-only；其余8,000-step pinball；每个model只有一次q90 residual offset。
- decisions：旧四coverage每组≥`.88`且mean sharpness reduction≥10%；通过才读取P167。
- prevention：不扫quantile/split/knots/loss/threshold；失败才用`V67-F134`。跨scene只写empirical coverage，不写formal guarantee。

### V67-F134 — source artifact已排除absolute mod-5 scenes导致calibration split为空

- 分类：`implementation/group-split-entry`；状态：`resolved_pre_evaluation`。
- failed run：`run://worldsim_v67/WS-V67-P170-CONFORMAL-COST-UPPER-BOUND-01/
  20260830T132000Z__conformal-cost-upper-bound-s0-r1`。
- exposure：8,000-step source-only q90 loss可见；0 old-cohort/P167 evaluation、0 conformal offset、0 coverage/sharpness/verdict。
- root cause：P109 source artifact由更早protocol构造时已排除absolute `scene_index%5==0`，P170重复使用同一条件得到空calibration。
- recovery：遵循group-disjoint split，改为artifact内ordered unique scene position每5取1；不随机、不读target、不改模型或门；r2从头训练。
- prevention：今后对派生source artifact按实际group集合分割，不假设原metadata整数空间仍完整；不增加split sweep或验证矩阵。

下一可用编号为：`V67-F135`。

### P170 r2 development outcome note — F134后原合同2/2通过

- r2未复用r1权重；ordered unique source scene split产生非空、scene-disjoint calibration并完成single q90 offsets。
- 旧四coverage minimum=`.90601`，mean sharpness reduction=`23.66%`；2/2 development decisions。
- 当前模型冻结等待P167 prospective rows；没有quantile/split/threshold recovery，也没有新failure。

下一可用编号仍为：`V67-F135`。

### P171 freeze note — learned conditional rectification of frozen P170 conformity residual

- base：P170 model/norms/global offset冻结；rectifier只用score+horizon预测q90 log-cost residual correction。
- split：P170同一ordered scene group holdout；6,000-step source training后single final q90 offset。
- decisions：old4 coverage每组≥`.88`+mean sharpness over P170≥5%；通过才读P167。
- prevention：不扫hidden/quantile/split/loss/threshold；algorithm failure才用`V67-F135`并关闭rectifier trial。

