# 历史原始记录 034

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F231 — P356 soft worst-group primal-dual未进入可行域且未迁移

- canonical=`run://worldsim_v67/WS-V67-P356-WORST-GROUP-SELECTIVE-RISK-CALIBRATOR-01/
  20260901T163000Z__worst-group-selective-risk-calibrator-s0-r3`；
- symptom：训练末端soft max-group risk=`.147173 > .10`，source fold5/P201 q90 unsafe=
  `.106481/.210191`；P201 coverage=`.406284`充分但risk失败，3/4 rejected；
- optimization evidence：dual max仅增至`8.19968`且soft coverage=`.673433`，coverage reward持续推动激进准入；独立
  fold PAV也不会改变错误排序。继续增dual/降temperature属于超参救援而非新研究；
- throughput note：r1/r2只在训练中因逐组CPU调度/小batch GPU低利用率中止，0 PAV/P201 quality read；r3通过
  scatter reduction和等总样本大batch完成，故只有r3是科学结果，不新增failure id；
- literature/migration：cumulative event risk应随预测horizon单调；NeurIPS 2019 cumulative-hazard network、AISTATS
  2022 SuMo与CHIL 2023 Neural Fine-Gray用positive monotone network建模时间风险。P357回到P346 direct head，
  结构化修复heldout-H外推；
- forbidden rescue：不扫dual/coverage/BCE weight/temperature或选r1/r2 checkpoint；resolution=
  `closed feature-calibrator family; open P357 monotone cumulative-risk head`。

### V67-F232 — P357 cumulative-hazard单调先验与visited-state cost label不匹配

- canonical=`run://worldsim_v67/WS-V67-P357-MONOTONE-HORIZON-CUMULATIVE-RISK-01/
  20260901T164500Z__monotone-horizon-cumulative-risk-s0-r1`；
- symptom：source fold3 q90 coverage/unsafe=`.288125/.261733`，P201=`.327049/.163636`；只有P201 coverage与两项
  monotonicity通过，2/4 rejected；
- diagnosis：positive base/rate/curvature保证随H累计，但当前target是各固定H下visited-state selected cost exceedance，
  不是“首次event在H前发生”的cumulative incidence；强制所有context风险随H增加会错误合并不同future-state切片；
- retained evidence：训练BCE仍收敛到`.198605`，但PAV BCE=`.338803`且source heldout risk极差，说明失败不是OOM、
  发散或capacity，而是结构先验错误；
- literature/migration：CVPR 2019 multi-step occupancy、CVPR 2024 Cam4DOcc都在预设future time grid评价状态；
  NeurIPS 2022 multi-task工作支持shared trunk+task-specific heads。P358把horizon当离散支持任务，并只保留task shift；
- forbidden rescue：不改hazard order/activation、放松单调性或继续套survival loss；resolution=
  `closed monotone-horizon family; open P358 fixed-grid multi-horizon reliability`。

### V67-F233 — P358跨horizon共享校准破坏固定网格多头的任务尺度

- canonical=`run://worldsim_v67/WS-V67-P358-FIXED-GRID-MULTIHORIZON-RELIABILITY-01/
  20260901T170000Z__fixed-grid-multihorizon-reliability-s0-r1`；
- symptom：P201 q90 coverage/unsafe=`.306284/.112360`，coverage已过`.30`但unsafe高于`.10`；source fold3
  q90=`.264609/.340974`，故3/4 rejected；
- retained evidence：训练BCE=`.116441`显著低于P357，证明四个task heads有拟合能力；问题出现在共享group-cal
  BCE=`.325290`与把四个horizon拼接后拟合的PAV BCE=`.345041`，不是OOM、发散或GPU资源不足；
- literature/migration：ICML 2024 multicalibration强调在预定义交叉group内校准，NeurIPS 2024也指出post-processing
  对原本不校准的模型最有帮助。P359把horizon作为显式group，为每个`H×ceiling×set`独立拟合校准参数与PAV；
- forbidden rescue：不扫head sharing、grid、阈值、width、steps或seed，不读取P201 labels调map；resolution=
  `retain fixed-grid representation; open P359 horizon-group conditional calibration`。

### V67-F234 — P359纯分组校准停在risk--coverage边界两侧

- canonical=`run://worldsim_v67/WS-V67-P359-HORIZON-GROUP-CALIBRATED-RELIABILITY-01/
  20260901T171500Z__horizon-group-calibrated-reliability-s0-r1`；
- symptom：P201 q90 coverage/unsafe=`.293169/.105263`，分别距离`.30/.10`仅`.006831/.005263`，但AND合同下
  两门均失败；source fold3 risk仍为`.296380`；
- retained evidence：相对P358，P201 risk `.112360→.105263`，说明horizon-group map方向正确；同时coverage
  `.306284→.293169`，证明它只在既有risk--coverage排序上移动 operating point，没有改善排序本身；
- literature/migration：ICML 2019 SelectiveNet联合优化预测与拒绝，AISTATS 2021 one-sided prediction在高准确率端
  显式压低false positive。P360据此在表示训练中固定unsafe positive weight=9，再用无权重校准恢复probability；
- forbidden rescue：不扫temperature、PAV、threshold或class weight，不把P201的边界差用于调参；resolution=
  `close calibration-only family; open P360 one-sided representation training`。

### V67-F235 — P360点式one-sided BCE恢复coverage但未消除严格ceiling排序尾差

- canonical=`run://worldsim_v67/WS-V67-P360-ONE-SIDED-MULTIHORIZON-RELIABILITY-01/
  20260901T173000Z__one-sided-multihorizon-reliability-s0-r1`；
- symptom：P201 q90 coverage/unsafe=`.342896/.107527`，覆盖充分但strict ceiling仍有10/93 unsafe，故3/4 rejected；
- retained evidence：相对P359，coverage `.293169→.342896`，source risk `.296380→.258403`，说明fixed one-sided
  weight改善了representation；P201 risk的小幅回升说明点式loss仍容许少数critical pair inversion；
- literature/migration：COLT 2013把AUC regret归约为bipartite pairwise classification，NeurIPS 2025 selective-gap
  分解也明确指出单调校准不改变ranking。P361在同horizon/ceiling内直接优化unsafe-vs-safe RankNet loss；
- forbidden rescue：不扫positive weight、pairwise weight/count/margin或threshold；resolution=
  `retain one-sided anchor; open P361 pairwise ranking refinement`。

### V67-F236 — P361全局pairwise排序与低风险selective尾部目标不对齐

- canonical=`run://worldsim_v67/WS-V67-P361-PAIRWISE-ONE-SIDED-RELIABILITY-01/
  20260901T174500Z__pairwise-one-sided-reliability-s0-r2`；
- symptom：P201 q90 coverage/unsafe=`.343169/.127660`，覆盖与两项单调门通过但risk比P360的`.107527`更差；
  source q90=`.292887/.218504`，仍不支持source scene transfer；
- retained evidence：pairwise loss收敛到`.088043`、无权重calibration/PAV=`.315332/.321800`，证明训练执行有效；
  失败来自优化全分布unsafe--safe inversions未强调最终`.10`低风险工作点，而非入口、OOM或未收敛；
- literature interpretation：COLT AUC reduction针对全局bipartite ranking；NeurIPS 2024 AUGRC与NeurIPS 2025 selective-gap
  工作均提示全曲线/全局ranking与特定低风险operating point存在错位。本结果实证复制该边界；
- forbidden rescue：不扫pair loss weight/count/margin、positive weight或threshold，不挑checkpoint；resolution=
  `close pairwise fixed-grid recovery; retain P346 as stable best under explicit heldout-H limitation`。

### V6.7 quota closeout failure boundary

- F233--F236依次关闭shared calibration、horizon-group post-hoc、pointwise one-sided与global pairwise恢复；
- P361 r1是step1前API兼容退出，0科学read，不另占failure id；r2是唯一P361科学结果；
- 当前无active experiment，下一可用编号`V67-F237`；停止原因是用户额度要求，不是资源不足或多卡需求。

### V67-F207 — P319只投影task不能关闭严格ceiling authority空集

- canonical=`run://worldsim_v67/WS-V67-P319-MINIMAL-TASK-PROJECTION-AUTHORITY-01/
  20260901T061500Z__minimal-task-projection-authority-s0-r1`；
- symptom：13个task-conditioned sets中按P317 score做最小task偏移，P201 mean projected/exact coverage=
  `.31913/.26066`，gain=`.05847 < .10`；strict ceiling两者均0，coverage-gain gate失败；
- retained evidence：max unsafe projected rate=`.04444`、monotonicity violations=0；中/高ceiling gain=
  `.06557/.10984`，说明projection机制有效，但可行集对象太窄；
- literature check/migration：OptNet action projection与NeurIPS 2022 Safety Editor均把编辑施加于action而非只改上层task；
  P320据此保持task不变，枚举六动作的15个top-2 pairs并训练pair-level editor；
- forbidden rescue：不降`.10` gate、不删除strict ceiling、不增加task grid、不在P201训练或校准；
- resolution status：`open via P320 action-pair editor`。

### V67-F208 — P320逐pair校准在事后pair选择前产生保守margin叠加

- canonical=`run://worldsim_v67/WS-V67-P320-ACTION-PAIR-SAFETY-EDITOR-01/
  20260901T063000Z__action-pair-safety-editor-s0-r1`；
- symptom：217,890 pair examples训练收敛到loss=`.19281`，但15个candidate逐个q90校准要求margin=`.51617`；
  P201 mean edited/nominal coverage=`.25738/.26066`，gain=`-.00328 < .10`；
- retained evidence：max unsafe=`.05263`、monotonicity=0；strict/mid gain=`.00328/.01148`，说明pair object有信号，
  但高ceiling gain=`-.02459`暴露per-pair calibration与事后选择不匹配；
- literature check/migration：ICML 2022 decision-focused learning-to-rank支持直接优化候选集合排序，set encoder保证
  permutation equivariance；P321联合选15 pairs，随后只对hard-selected outcome训练/校准q90；
- forbidden rescue：不调margin/q90/gate、不删除high ceiling、不只报告strict gain、不在P201重训；
- resolution status：`open via P321 groupwise selected-outcome authority`。

### V67-F209 — P321 groupwise选择仍不能提升六动作lattice总体authority coverage

- canonical=`run://worldsim_v67/WS-V67-P321-GROUPWISE-PAIR-AUTHORITY-EDITOR-01/
  20260901T064500Z__groupwise-pair-authority-editor-s0-r1`；
- symptom：selector/risk losses=`.21701/.01266`，selected-outcome margin降到`.29468`，但P201 mean
  groupwise/nominal coverage=`.25956/.26066`，gain=`-.00109 < .10`；
- retained evidence：max unsafe=`.01774`、monotonicity=0，高ceiling gain=`.03934`；strict仍0，mid gain=
  `-.04262`。失败不是逐pair multiple-selection margin单一问题，而是当前六动作candidate family的capacity边界；
- literature check/migration：多horizon uncertainty工作强调uncertainty随预测时域增长且应显式建模；P322不再编辑
  task/pair，直接学习requested set在continuous H下的visited-state cost quantile surface；
- forbidden rescue：不调q90/utility weight/selector architecture/gate，不扩同一pair family，不在P201训练；
- resolution status：`closed repair family; migrated to P322 horizon-conditioned authority`。

### V67-F210 — P322冻结P199 artifact run ID少reliability

- failed run=`run://worldsim_v67/WS-V67-P322-HORIZON-CONDITIONED-ADMITTED-SET-QUANTILES-01/
  20260901T070000Z__horizon-conditioned-admitted-set-quantiles-s0-r1`；
- symptom：P199 task目录正确，但run ID误写`20260830T181000Z__joint-horizon-copula-s0-r2`，实际为
  `20260830T181000Z__joint-horizon-reliability-copula-s0-r2`；`torch.load`前FileNotFound；
- scientific status：0 training、0 calibration、0 P201 quality，不产生horizon-method verdict；
- minimal recovery：r2只修run path；模型、source/P201 rows、H split、quantiles、seed、steps和gates不变；
- resolution status：`closed by P322 r2 supported result`。

### V67-F211 — P324全局raw residual margin在heldout horizon欠缺极少量coverage

- canonical=`run://worldsim_v67/WS-V67-P324-HORIZON-CEILING-SELECTIVE-AUTHORITY-01/
  20260901T074000Z__horizon-ceiling-selective-authority-s0-r1`；
- symptom：43,578 examples、6k steps、final loss=`.23806`；source q90 raw margin=`.29441`。P201
  mean coverage=`.19836 < .20`，仅差`.00164`，coverage gate失败；
- retained evidence：max/mean unsafe=`.04762/.02385`远低于`.15`，ceiling monotonicity=0；highest-ceiling
  coverage=`.48115`，冻结H-q95 baseline mean coverage=`.13361`，说明task×H score有选择价值；
- diagnosis：跨`.8/1.5/3.0s`共用单个raw residual margin忽略residual spread随H变化；heldout H=`2.5s`
  上安全但略过保守。差距虽小，仍不得舍入为通过或用P201调margin；
- literature check/migration：NeurIPS 2025 time-varying nonconformity normalization与NeurIPS 2022 temporal
  quantile adjustment均显式让不确定性校准随时间变化；P325据此学习source-only task/H-conditioned positive
  scale，再用source calibration冻结一个normalized q90阈值；仅作经验风险校准，不宣称formal guarantee；
- forbidden rescue：不降低`.20`、不改q90、不删strict ceiling、不扫scale/gate/seed、不用P201训练或校准；
- resolution status：`closed by P325 heldout task+H supported result`；P325 source strict低样本不稳定仍作为
  claim boundary保留；下一可用failure id=`V67-F212`。

### V67-F212 — P331全正多轴交互使continuous-risk nested authority过度保守

- canonical=`run://worldsim_v67/WS-V67-P331-RISK-SIZE-HORIZON-AUTHORITY-SURFACE-01/
  20260901T093000Z__risk-size-horizon-authority-surface-s0-r1`；
- symptom：130,734 examples、9k steps、final pinball=`.02573`，但source q90 offset=`.07846`；P201
  mean any-authority coverage=`.24672 < .30`，coverage gate失败，strict coverage=0；
- retained evidence：max/mean unsafe=`.03764/.02198`、size/ceiling violations=0；high-ceiling coverage=
  `.65328`、mean selected size=`2.80`，说明nested-size object有效但q/H/k joint parameterization保守；
- diagnosis：沿k累积positive base/H/q/H×q coefficients不仅保证partial monotonicity，还额外强制所有高阶
  interactions为正（supermodular）；这比问题需要的局部序约束更强，造成q90 offset与strict abstention；
- literature check/migration：NeurIPS 2017 Deep Lattice Networks以interpolated lookup-table局部序约束表达
  flexible partial-monotone functions；ICML 2023 Constrained Monotonic Networks证明更一般单调函数逼近可行。
  P332采用context-conditioned H×q×k lattice vertices与逐轴monotone envelope；
- forbidden rescue：不降`.30`、不改q90/range/k/gate、不在P201 fit、不扫capacity/seed；
- resolution status：`open via P332 partial-monotone lattice`；下一可用failure id=`V67-F213`。

