# 历史原始记录 049

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F148 — bootstrap density ensemble牺牲P81 Brier以换取不均匀calibration增益

- 分类：`algorithm/density-ensemble-environment-tradeoff`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P184-SCENE-BOOTSTRAP-LOG-COST-DENSITY-ENSEMBLE-01/
  20260830T153000Z__scene-bootstrap-log-cost-density-ensemble-s0-r1`。
- 观察：mean calibration-error reduction vs P182=`20.57%`，且P96/P113/P129 Brier改善；但P81 Brier回退`2.18%`、calibration退`17.05%`。
- 解释：density ensemble产生约`1.7%--2.0%`概率分歧并改善部分环境，但uniform averaging不是环境稳健目标，仍可牺牲一个cohort。
- 防重复：不调ensemble weight/member count/bootstrap；下一步直接优化固定source environments的worst NLL。

下一可用编号为：`V67-F149`。

### V67-F149 — source worst-environment NLL仍复现P81 trade-off

- 分类：`algorithm/source-environment-DRO-tradeoff`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P185-WORST-ENVIRONMENT-LOG-COST-DENSITY-01/
  20260830T154000Z__worst-environment-log-cost-density-s0-r1`。
- 观察：P96/P113/P129 Brier改善`5.37%/2.14%/.79%`，mean calibration improvement=`13.02%`；但P81 Brier回退`2.64%`、
  calibration退`17.51%`。
- 解释：ordered-source环境最坏NLL不等价于未知P81 shift；它重新分配source likelihood，却与P184一样牺牲P81概率刻度。
- 防重复：关闭source bootstrap/group-DRO rescue；不扫环境划分、数量、temperature或loss，P182/P183保持冻结。

下一可用编号为：`V67-F150`。

### V67-F150 — source noise smoothing以conditional refinement换取边际校准

- 分类：`algorithm/noise-regularized-density-refinement-loss`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P186-NOISE-REGULARIZED-LOG-COST-DENSITY-01/
  20260830T155000Z__noise-regularized-log-cost-density-s0-r1`。
- 观察：mean calibration-error reduction vs P182=`19.60%`，P81/P96/P113三者边际误差改善；但四个cohort Brier均回退
  `20.82%/14.13%/1.63%/27.51%`，P129 calibration也回退`17.13%`。
- 解释：fixed input/target smoothing改变了source prevalence刻度，但同时抹平P182在score/horizon/clearance上的条件分辨率；
  marginal calibration改善不足以补偿proper-score refinement损失。
- 防重复：关闭source-noise smoothing；不扫condition/target noise scale。下一步只试正交的fixed heavy-tail density family。

下一可用编号为：`V67-F151`。

### V67-F151 — fixed heavy tails改善两cohort但不能统一概率刻度

- 分类：`algorithm/heavy-tail-density-cohort-tradeoff`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P187-STUDENT-T-LOG-COST-MIXTURE-DENSITY-01/
  20260830T160000Z__student-t-log-cost-mixture-density-s0-r1`。
- 观察：相对P182，P81/P113 Brier改善`4.15%/3.88%`且calibration改善`43.36%/30.51%`；P96/P129 Brier回退
  `3.14%/.45%`且calibration回退`20.31%/38.16%`，mean calibration improvement仅`3.85%`。
- 解释：ν=`3`重尾能修复部分cohort的Gaussian tail misspecification，却同时扩散另两个cohort的中心概率质量；固定全局tail family
  不能表达随condition变化的偏态/局部形状。
- 防重复：关闭Student-t/单纯heavy-tail family rescue，不扫ν或component count；下一步只试一次conditional monotone spline。

下一可用编号为：`V67-F152`。

### V67-F152 — 更强source spline likelihood没有转化为跨cohort可靠性

- 分类：`algorithm/flexible-density-source-overfit`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P188-CONDITIONAL-SPLINE-LOG-COST-DENSITY-01/
  20260830T161000Z__conditional-spline-log-cost-density-s0-r1`。
- 观察：8-bin RQ spline final source NLL=`-1.39293`，显著低于P182约`-1.09`；但仅P96 Brier改善`7.78%`，P81/P113/P129
  分别回退`7.42%/3.47%/.20%`，mean calibration change=`-23.89%`。
- 解释：模型容量成功拟合source log-cost的偏态/局部形状，却放大了source特有概率刻度；NLL提升不等同于proper-score跨cohort迁移。
- 防重复：关闭RQ-spline bin/tail/flow-depth sweep；不因source NLL更低放宽可靠性gate。下一机制只检验objective mismatch。

下一可用编号为：`V67-F153`。

### V67-F153 — pure budget-Brier目标改善两cohort但损失NLL refinement

- 分类：`algorithm/proper-score-objective-tradeoff`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P189-BUDGET-BRIER-LOG-COST-CDF-01/
  20260830T161500Z__budget-brier-log-cost-cdf-s0-r2`。
- 观察：P96/P113 Brier改善`6.94%/9.02%`，mean calibration improvement=`11.09%`；但P81/P129 Brier回退`2.06%/.56%`，
  calibration回退`3.28%/21.89%`。r1 pre-step bool-cast错误无quality，不是本算法failure。
- 解释：直接proper-score训练证实NLL/objective mismatch，但完全替换NLL会丢掉P182在P81/P129保留的conditional refinement。
- 防重复：不扫budget weights或阈值；不再做pure Brier from-scratch。下一步只做一次无手调权重的NLL+Brier PCGrad。

下一可用编号为：`V67-F154`。

### V67-F154 — PCGrad把冲突压缩到单一P96残差但仍未全通过

- 分类：`algorithm/multi-objective-residual-cohort-conflict`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P190-PCGRAD-LOG-COST-CDF-01/
  20260830T162500Z__pcgrad-log-cost-cdf-s0-r1`。
- 观察：P81/P113/P129 Brier改善`.15%/2.27%/.44%`，mean calibration improvement=`7.19%`；P96 Brier仍回退`.62%`、
  calibration回退`2.42%`，因此逐cohort noninferiority失败。
- 解释：norm-balanced PCGrad有效缓和NLL/Brier冲突，但共享的3D condition仍不能区分P96概率刻度；这是condition sufficiency残差，
  不是继续调loss weight的授权。
- 防重复：不放宽P96 gate，不扫PCGrad weight/projection/step/lr；下一步显式解压冻结boundary evidence context。

下一可用编号为：`V67-F155`。

### V67-F155 — 解压boundary evidence component加剧跨cohort不稳定

- 分类：`algorithm/context-proxy-nontransport`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P191-DECOMPOSED-BOUNDARY-EVIDENCE-DENSITY-01/
  20260830T163000Z__decomposed-boundary-evidence-density-s0-r1`。
- 观察：仅P113 Brier改善`5.49%`；P81/P96/P129回退`5.62%/4.00%/16.70%`，mean calibration improvement=`1.79%`。
- 解释：aleatoric/epistemic/projected-mean magnitude在source提高NLL但不是稳定shift proxy；显式解压P126 ratio反而让density利用不可迁移刻度。
- 防重复：关闭这三个context及其子集/aggregation sweep；不使用location/target标签救结果。下一步只改source scene sampling measure。

下一可用编号为：`V67-F156`。

### V67-F156 — 纯scene等权sampling产生短时域可靠性回退

- 分类：`algorithm/sampling-measure-horizon-tradeoff`；状态：`closed_negative_after_post_confirmation_secondary`。
- canonical：`run://worldsim_v67/WS-V67-P193-SCENE-BALANCED-POST-CONFIRMATION-01/
  20260830T170000Z__scene-balanced-post-confirmation-s0-r2`。
- 观察：冻结P192相对P182在已消费P183 rows上，H`.8/1.5s` Brier回退`.93%/1.13%`，H`2.5/3.0/3.5s`
  改善`.32%/1.30%/4.75%`；macro Brier虽改善`.86%`，macro calibration improvement为`-.03%`，0/2 gates。
- 解释：source scene等权不是无条件稳健改进；它改变trajectory measure，对长时域稀疏scene有利，却削弱短时域高密度状态的
  概率刻度。P192 development四cohort正结果保留，但不能晋升或再占用fresh cohort。
- 工程边界：r1缺仓库级`PYTHONPATH`，在import阶段退出且未读quality/未占GPU；r2仅修进程环境。此启动失败不另分算法号。
- 防重复：不做scene-weight/horizon-weight网格，不在P183 rows调gate或后处理；P194只允许事前固定half pooled/half scene-balanced
  的一次折中训练。若仍失败，关闭sampling-measure refinement并保留P182为唯一fresh-supported density。

下一可用编号为：`V67-F157`。

### V67-F157 — 全局50/50 sampler折中未继承两端优势

- 分类：`algorithm/global-mixture-negative-transfer`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P194-MIXED-SCENE-EMPIRICAL-LOG-COST-DENSITY-01/
  20260830T171000Z__mixed-scene-empirical-log-cost-density-s0-r1`。
- 观察：相对P182仅P96 Brier改善`.24%`；P81/P113/P129回退`1.19%/.46%/.93%`，mean calibration improvement
  `-12.98%`，0/2 gates。source NLL=`-1.10084`不能挽救transfer判定。
- 解释：pooled与scene-balanced risk不是可用一个全局凸混合消除的偏差；P193已显示效应随horizon变号，P194把不同H继续共享同一
  sampling weight，因而仍产生negative transfer。
- 防重复：不扫25/50/75%或连续mixture weight，不在P183选择权重。P195只允许一次由source horizon端点事前确定的线性
  conditional sampler；模型容量、NLL、训练预算保持不变。

下一可用编号为：`V67-F158`。

### V67-F158 — horizon条件sampler仍受共享density参数干扰

- 分类：`algorithm/conditional-sampling-shared-parameter-interference`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P195-HORIZON-CONDITIONED-SCENE-SAMPLING-01/
  20260830T172000Z__horizon-conditioned-scene-sampling-s0-r1`。
- 观察：相对P182，P96/P129 Brier改善`2.45%/.94%`，P81/P113回退`3.85%/.52%`；mean calibration改善`9.35%`，
  但逐cohort noninferiority失败。结果优于P194 calibration，却仍不是可晋升candidate。
- 解释：已知H条件的采样只改变共享参数收到的梯度比例，不能防止不同measure对同一density weights的覆盖；这解释了calibration
  可改善但P81 proper score显著退化。
- 防重复：关闭sampling schedule family，不扫非线性schedule或端点概率。P196冻结两个完整专家，只训练source-only两标量单调
  horizon router；若仍失败，关闭P182/P192 refinement并回到fresh-supported P182。

下一可用编号为：`V67-F159`。

### V67-F159 — source-NLL固定density pool仍复制短时域回退

- 分类：`algorithm/source-router-target-horizon-nontransport`；状态：`closed_negative_after_consumed_secondary`。
- canonical：`run://worldsim_v67/WS-V67-P197-ROUTED-DENSITY-POST-CONFIRMATION-01/
  20260830T174000Z__routed-density-post-confirmation-s0-r1`。
- 观察：P196在旧四cohort 2/2 development通过，但冻结后在已消费P183五H上，H`.8/1.5` Brier回退`.49%/.55%`，
  macro calibration只改善`1.13%`，0/2 gates。router slope近零导致所有H约55.7%使用P192。
- 解释：source likelihood没有识别target-domain的horizon-dependent expert suitability；完整专家pool解决共享训练干扰，却仍把scene-balanced
  expert过多混入短H。该结果只是否定P196升级，不推翻P183/P182 fresh density。
- 防重复：不在P183 rows拟合router/threshold，不扫constant weight或增加router feature。P198只允许一次source horizon参数隔离：
  short/long experts分别训练，边界锁定相邻source horizons；失败即关闭P182/P192 sampling/expert refinement family。

下一可用编号为：`V67-F160`。

### V67-F160 — short/long参数隔离放大长时域跨cohort偏差

- 分类：`algorithm/horizon-specialist-nontransport`；状态：`closed_negative_family_terminal`。
- canonical：`run://worldsim_v67/WS-V67-P198-SHORT-LONG-DENSITY-EXPERTS-01/
  20260830T175000Z__short-long-density-experts-s0-r1`。
- 观察：short/long分别8,000-step专训，P96/P129 Brier改善`.79%/.38%`，但P81/P113回退`4.20%/5.56%`；mean
  calibration improvement=`-.87%`，0/2 gates。更低的horizon-subset NLL没有形成迁移优势。
- 解释：scene-measure shift不等于可由source horizon partition识别的task差异；参数隔离减少gradient interference，却牺牲跨H共享
  statistical strength并使long expert对P81/P113失配。
- 防重复：P192--P198 sampling、global mix、conditional sampler、frozen router与horizon expert路线全部关闭。不扫boundary、expert
  count、init、schedule或loss；P182保持唯一fresh-supported marginal density。下一步改变预测对象为joint-horizon dependence。

下一可用编号为：`V67-F161`。

P199 r1 engineering note（不占算法编号）：冻结`scene_index % 5 == 0`与P109既有“排除mod5=0”的4/5 source构造
完全互补，导致6,000-step训练后、任何development metric前发现0-row split并退出。只读scene-index计数确认mod5=
`[0,20,28,22,32]`；r2只把remainder恢复为1，模型/数据/seed/horizons/budgets/MC/decisions全不变。不把0-row当科学失败，
不试多个split；下一可用编号仍为`V67-F161`。

