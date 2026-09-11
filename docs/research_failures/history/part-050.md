# 历史原始记录 050

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F161 — 直接joint CDF以calibration换取refinement退化

- 分类：`algorithm/joint-proper-score-refinement-loss`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P202-DIRECT-MONOTONE-JOINT-CDF-01/
  20260830T185000Z__direct-monotone-joint-cdf-s0-r1`。
- 观察：七预算direct monotone CDF把mean calibration error从P199 `.022017`降至`.014859`（`32.51%`），但integrated Brier
  从`.075012`升至`.082756`（恶化`10.32%`），1/2 gates。
- 解释：直接Brier head更贴近budget-wise总体发生率，却比“冻结marginals + dependence”丢失instance-level refinement；与proper-score
  calibration/refinement分解一致，不能把更低平均误差包装成更好概率预测。
- 防重复：不扫BCE/Brier mix、head、budget或容量。P203只允许一次共享三参数rank-preserving beta map作用于P199输出；P201仍
  确认raw P199。若P203失败，关闭joint post-hoc calibration family。

### P201/P206 background-entry recovery note — 不占算法failure编号

- P201 evaluator首次后台入口因shell后台命令的工作目录未保留，Python尝试打开`/root/scripts/...`后退出；P201 preparation
  六个archive workers未受影响，0 processed-row/quality read；改用绝对script/config路径后按原合同重启r2。
- P206 r1/r2分别在import/argparse阶段因缺项目`PYTHONPATH`及遗漏`--runs-root`退出；0 source-row load、0 optimizer step、
  0 metric/gate。r3只修正进程入口参数后进入训练，模型、数据、split、seed和decisions全不变。
- 这些事件不改变P201/P206科学trial计数，也不触发额外smoke、cohort、gate、hash/checksum/fingerprint。

下一可用编号仍为：`V67-F162`。

### V67-F162 — 全局常数copula不足以替代输入条件化dependence

- canonical：`run://worldsim_v67/WS-V67-P206-CONSTANT-JOINT-COPULA-ABLATION-01/20260830T193000Z__constant-joint-copula-ablation-s0-r3`；
- 观察：constant/P199 integrated Brier=`.075778/.075012`，相对退化`1.02%`；mean calibration error=
  `.027508/.022017`，相对退化`24.94%`，0/2 gates；
- 解释：冻结相同P182 marginals、PIT、split、joint event、budgets和MC后，只有相关结构是否依赖score/clearance不同；因此
  P199增益不能仅解释为一个静态跨horizon correlation matrix；
- 文献响应：NeurIPS 2013 conditional copula指出covariates显著影响dependence时静态copula会失真；NeurIPS 2019/2024
  支持时变低秩协方差。P207只迁移一次rank-2-plus-diagonal conditional structure，不扫rank/width/loss；
- 防重复：不再训练第二个constant copula，不改P201 primary，不以source ablation宣称fresh generalization。

下一可用编号：`V67-F163`。

P207 r1 engineering note（不占算法编号）：rank-2 covariance参数化为`U U^T + D`，但final factor head被全零初始化，
使`U=0`处factor梯度严格为零；8,000 steps后NLL仍约identity baseline，结果不可用于否定低秩结构。参考ICML 2017 strict-saddle
与NeurIPS low-rank factorization使用随机/扰动初始化，r2只改为seed0的小随机factor初始化；data/rank/width/steps/lr/MC/门
完全不变。r2已呈现有效NLL下降，因此下一可用算法编号仍为`V67-F163`。

### V67-F163 — rank-2条件copula的微小Brier增益伴随calibration回退

- canonical：`run://worldsim_v67/WS-V67-P207-LOW-RANK-CONDITIONAL-JOINT-COPULA-01/20260830T195500Z__low-rank-conditional-joint-copula-s0-r2`；
- 观察：low-rank/P199 Brier=`.074955/.075012`，严格改善仅`.077%`；calibration error=`.022352/.022017`，
  退化`1.52%`，故1/2 gates；
- 解释：rank-2-plus-diagonal结构保留大部分条件dependence并轻微正则化refinement，但不足以同时维持总体可靠度；
- 文献响应：AISTATS mixture-of-copulas允许在不改变边际的情况下混合依赖成分。P208只训练P199-vs-independence的单线性
  conditional shrinkage gate，保留P199 refinement，不再替换完整相关结构；
- 防重复：不扫rank、factor scale、width、loss或初始化；r1 zero-gradient不计算法trial，r2是唯一有效P207 verdict。

下一可用编号：`V67-F164`。

### V67-F164 — P199向independence的条件收缩无development空间

- canonical：`run://worldsim_v67/WS-V67-P208-CONDITIONAL-COPULA-SHRINKAGE-01/20260830T200500Z__conditional-copula-shrinkage-s0-r1`；
- 观察：gate把平均`.98283`权重留给P199，最小`.95130`、最大`1.0`；混合后Brier退化`.084%`，calibration error
  退化`4.40%`，0/2 gates；
- 解释：合法P199/independence mixture的likelihood optimum靠近P199边界，简单逐实例减弱dependence既未改善refinement也未改善
  marginal joint frequency；
- 文献响应：conditional Student-t copula允许尾依赖结构区别于Gaussian相关。P209固定`nu=4`作一次full conditional t-copula
  trial，不继续增加mixture components或gate深度；
- 防重复：关闭P206 static、P207 low-rank replacement与P208 shrinkage；不扫initial weight/component/gate/loss。

下一可用编号：`V67-F165`。

### V67-F165 — fixed-nu Student-t尾依赖不优于Gaussian P199

- canonical：`run://worldsim_v67/WS-V67-P209-CONDITIONAL-STUDENT-T-COPULA-01/20260830T201500Z__conditional-student-t-copula-s0-r1`；
- 观察：Student-t/P199 Brier=`.075435/.075012`，退化`.56%`；calibration error=`.022503/.022017`，
  退化`2.20%`，0/2 gates；
- 解释：在相同P182 marginals/full conditional correlation下，`nu=4`共享重尾并未解释四H joint-event，Gaussian P199更适合；
- 文献响应：UAI multivariate extreme-value与conditional-density工作允许直接建模maximum functional。P210转为
  `log1p(max_H cost_H)`连续density，使目标CDF与joint event严格对齐，不继续换copula；
- 防重复：关闭P206--P209 static/low-rank/shrinkage/t-copula；不扫df或试Gaussian/t mixtures。

下一可用编号：`V67-F166`。

### V67-F166 — direct maximum-cost density改善校准但损失refinement

- canonical：`run://worldsim_v67/WS-V67-P210-JOINT-MAX-COST-DENSITY-01/20260830T203000Z__joint-max-cost-density-s0-r1`；
- 观察：相对P199，mean calibration error改善`28.42%`，但integrated Brier退化`1.30%`，1/2 gates；
- 解释：一维maximum density能拟合joint总体频率，却把四H输入flat拼接后没有保留P199的instance refinement；与P202的
  calibration/refinement tradeoff方向一致，但幅度明显更小；
- 响应：先用一次proper-score global linear pool检验互补性，不改P210模型或门。

### V67-F167 — P199/P210 proper-score linear pool未跨scene迁移

- canonical：`run://worldsim_v67/WS-V67-P211-JOINT-PROBABILITY-LINEAR-POOL-01/20260830T204000Z__joint-probability-linear-pool-s0-r1`；
- 观察：source training给P210 `98.12%`权重，但dev pool Brier仍比P199退化`.96%`，calibration改善`27.87%`，1/2 gates；
- 解释：training/development对max-density refinement的偏好发生反转，单scalar不能恢复；问题更像flat representation的scene shift；
- 文献响应：Deep Sets/Set Transformer用共享element encoder与pooling编码集合结构，且maximum-value regression直接支持max-aware
  aggregation。P212只试一次mean/max DeepSet density；
- 防重复：不扫pool weight、conditional gate、第三成分或budget-wise权重；P210/P211关闭。

下一可用编号：`V67-F168`。

### V67-F168 — DeepSet maximum density未迁移到已消费P183 joint rows

- canonical：`run://worldsim_v67/WS-V67-P213-DEEPSET-MAX-DENSITY-POST-CONFIRMATION-01/20260830T210000Z__deepset-max-density-post-confirmation-s0-r1`；
- 观察：P212在source dev同时改善Brier`3.84%`和calibration`17.80%`，但冻结后在P183 rows的Brier相对P199退化
  `2.74%`；calibration仍改善`19.31%`，故1/2 gates；
- 解释：set encoder修复了source split内flat representation，但maximum-density的calibration/refinement transfer冲突仍存在；
  P199 dependence factorization在P183与P201上更稳定；
- 响应：不为P212开启fresh cohort，不以P201已读rows作事后promotion；保留P212为结构消融，P199/P203为可迁移链；
- 防重复：关闭P210--P213 maximum density、scalar pool与DeepSet recovery；不试attention/pooling/capacity/mixture。

下一可用编号：`V67-F169`。

### V67-F169 — prefix survival density改善refinement但概率尺度失准

- canonical：`run://worldsim_v67/WS-V67-P214-PREFIX-SURVIVAL-MAX-COST-DENSITY-01/20260830T213000Z__prefix-survival-max-cost-density-s0-r1`；
- 观察：四prefix宏平均Brier相对P199改善`1.16%`，最终四H Brier改善`5.21%`，但宏平均calibration error退化
  `30.33%`，只过Brier门；
- 解释：前缀最大cost对象及共享set representation具有refinement增量，但训练density的source mixture不能直接给出稳定概率尺度；
- 文献响应：proper calibration/refinement分解与beta calibration允许保序地修正概率尺度。P215只作一次disjoint-scene
  monotone beta recovery，使density fit、calibrator fit和development read不共享scene；
- 防重复：不扫prefix数、survival loss、mixture components、network width、budget、split或calibrator family；若P215失败即关闭。

下一可用编号：`V67-F170`。

F169 recovery closure：P215把source scenes按remainder拆成9,730 density-fit、5,043 calibration与3,742 development
trajectories；低自由度monotone beta层在未触碰dev上同时改善P199 Brier `.895%`和calibration error `38.31%`，2/2。
这只解决source probability-scale失配；是否迁移由P216决定。

### V67-F170 — disjoint-calibrated prefix survival仍发生跨cohort refinement回退

- canonical：`run://worldsim_v67/WS-V67-P216-CALIBRATED-PREFIX-SURVIVAL-POST-CONFIRMATION-01/20260830T220000Z__calibrated-prefix-survival-post-confirmation-s0-r1`；
- 观察：冻结P215在P183 consumed-secondary中calibration error改善`19.87%`，但macro Brier退化`1.89%`，1/2；
- 解释：disjoint beta校准可迁移概率尺度，却不能修复source-to-P183的conditional refinement/covariate shift；
- 文献响应：AISTATS 2020 calibrated prediction under covariate shift使用unlabeled-target density ratio做importance weighting。
  P217只试一次目标特征加权的proper density/calibration训练，P183标签不参与优化；
- 防重复：P217若失败即关闭prefix survival/density/calibration/importance-weighting家族；不扫weight clip、domain net或loss。

下一可用编号：`V67-F171`。

### V67-F171 — unlabeled-target importance weighting不能修复prefix refinement shift

- canonical：`run://worldsim_v67/WS-V67-P217-TARGET-WEIGHTED-PREFIX-SURVIVAL-01/20260830T221500Z__target-weighted-prefix-survival-s0-r1`；
- 观察：domain classifier accuracy仅`.52915`，加权后P183 macro Brier仍退化`1.93%`，calibration改善`19.60%`，1/2；
- 解释：score/clearance/horizon covariates上source与P183近乎不可分，P216失败更符合conditional mechanism/refinement shift，
  而非能由unlabeled density ratio纠正的covariate shift；
- 响应：关闭prefix survival density、beta calibration与importance-weighting；P218更换prediction object为时间加权累计
  visited-state exposure，降低maximum/first-passage对单个极值的敏感性；
- 防重复：不扫domain classifier、weight bounds、scene split、calibrator、prefix hazard或density结构。

下一可用编号：`V67-F172`。

### V67-F172 — direct cumulative-exposure density仅有微弱refinement且校准退化

- canonical：`run://worldsim_v67/WS-V67-P218-CUMULATIVE-EXPOSURE-DENSITY-01/20260830T224000Z__cumulative-exposure-density-s0-r1`；
- 观察：相对P182 marginals + P199 copula连续采样control，Brier改善`.387%`，但calibration error退化`26.25%`，1/2；
- 解释：从maximum改成time-weighted sum降低了tail sensitivity，但direct aggregate density仍存在概率尺度偏差；
- 文献响应：ICML distribution calibration与calibrated-sharp density支持在独立校准集上对CDF作单调map。P219只拟合一个
  shared beta map并直接以P183 consumed transfer决定，不进行第二source-only迭代；
- 防重复：若P219未跨P183同时过Brier/calibration，则关闭cumulative-exposure density，不扫budget/dt/mixture/map。

下一可用编号：`V67-F173`。

P219 engineering recovery note（不占算法编号）：r1在任何rows load或optimizer step前因`_metrics`一处缺失右括号发生
`SyntaxError`；只补齐括号，以r2运行，config、split、model、budgets、MC、control、metrics和gates完全不变。

### V67-F173 — calibrated cumulative exposure在P183同时损失refinement与校准

- canonical：`run://worldsim_v67/WS-V67-P219-CALIBRATED-CUMULATIVE-EXPOSURE-TRANSFER-01/20260830T230500Z__calibrated-cumulative-exposure-transfer-s0-r2`；
- 观察：source dev同时改善Brier`1.37%`与calibration`22.11%`，但P183 Brier退化`4.49%`、calibration退化
  `25.27%`，0/2；
- 解释：聚合target的direct density在source内可拟合且可校准，但conditional distribution跨cohort不稳定；maximum与sum对象
  均复现，故不是单一极值functional的问题；
- 响应：关闭direct aggregate density，保留P182+P199 factorization；P220改学其realized proper loss并作固定覆盖率selective authority；
- 防重复：不扫exposure intervals/budgets/density/calibrator/importance weights或新aggregate functional。

下一可用编号：`V67-F174`。

