# 历史原始记录 048

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F135 — conditional conformity rectifier保持coverage但显著恶化跨scene sharpness

- 分类：`algorithm/conditional-conformity-transfer`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P171-RECTIFIED-CONFORMAL-COST-BOUND-01/
  20260830T133000Z__rectified-conformal-cost-bound-s0-r1`。
- 观察：旧四coverage均≥`.9125`，但mean sharpness reduction over P170=`-19.48%`；仅P96锐化，1/2 decisions。
- 解释：source in-sample residual trend经final offset后仍跨scene漂移；conditional correction扩大三组upper bound，constant P170 offset更稳。
- 防重复：不扫hidden/split/quantile或对scene选择rectifier；按规则未读P167。P170仍是唯一prospective upper-bound candidate。

下一可用编号为：`V67-F136`。

### P167 partial pipeline note — 2/10 scenes scored while remaining shards scan

- shards03/08完整命中冻结members并释放`0269/0802`；preprocess与GPU scoring均未等待全部cohort。
- 两scene五H local rank gain全正，但partial result不得触发stop、scene replacement或decision修改。
- 无locator/资源/算法failure；下一编号保持`V67-F136`。

下一可用编号仍为：`V67-F136`。

### P172 freeze note — two-sided conformal cost interval after P167 partial read

- candidate：q10/q90 score-conditioned monotone models vs horizon-only，80% interval，P170同ordered-scene split。
- decisions：old4 coverage≥`.78` each + mean width reduction≥10%；P167禁止作为prospective confirmation。
- prevention：不扫quantile/split/knots/threshold；失败才用`V67-F136`，支持也必须新target-unread cohort。

### V67-F136 — two-sided cost interval虽更窄但P81 coverage仅.732

- 分类：`algorithm/two-sided-cost-interval-transfer`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P172-CONFORMAL-COST-INTERVAL-01/
  20260830T133500Z__conformal-cost-interval-s0-r1`。
- 观察：mean width reduction=`18.55%`通过，但P81 coverage=`.73199<.78`，1/2 decisions。
- 解释：跨scene时score-conditioned q10 lower edge比one-sided q90 upper更脆弱；效率提升不足以覆盖undercoverage。
- 防重复：不降低coverage门、不扫q10/q90或改split；关闭two-sided interval，保留P170 one-sided候选。

下一可用编号为：`V67-F137`。

### V67-F137 — P167局部常量输入产生undefined Spearman并阻止strict JSON收口

- 分类：`implementation/undefined-local-diagnostic-serialization`；状态：`resolved_without_metric_change`。
- failed run：`run://worldsim_v67/WS-V67-P167-PIPELINED-MULTI-HORIZON-CONFIRMATION-01/
  20260830T130500Z__pipelined-multi-horizon-confirmation-s0-r1`。
- exposure：10/10 rows、五H aggregate metrics和P170 prospective read已完成；只有scene-1065 H3.5局部描述性rank gain为NaN。
- root cause：53 trajectories上的P109 score为常量；按SciPy定义Spearman未定义并返回NaN，而RFC JSON不允许NaN且runner使用
  `allow_nan=false`。该local value不参与pooled per-H或macro decision。
- recovery：只把非有限local diagnostic写为JSON `null`；P126/P109、rows、scene、H、cost、coverage和decisions均不变。
  r2 macro rank/cost=`+.21412/-.0168403`，2/2 supported。

下一可用编号为：`V67-F138`。

### V67-F138 — P170新场景upper bound虽更窄但四个中长horizon under-cover

- 分类：`algorithm/one-sided-cost-bound-scene-shift`；状态：`closed_negative_after_prospective_read`。
- canonical：`run://worldsim_v67/WS-V67-P170-CONFORMAL-COST-UPPER-BOUND-01/
  20260830T132500Z__conformal-cost-upper-bound-s0-r2`。
- 观察：P167五H coverage=`.89073/.86316/.83184/.82614/.82257`，4/5低于`.88`；mean sharpness reduction仍为
  `25.09%`，1/2 prospective decisions。
- 解释：source-scene single offset在新location/horizon mixture下不够保守；模型锐化随H增强，但coverage同步下降。
- 防重复：不在P167上追加offset、不降低`.88`门、不扫quantile/split；关闭P170，保留P167 relative ranking/selection主结论。

下一可用编号为：`V67-F139`。

### V67-F139 — P173 direct-script launcher未把repo root加入module path

- 分类：`implementation/non-login-python-entry`；状态：`resolved_pre_training`。
- failed run：`run://worldsim_v67/WS-V67-P173-MONOTONE-VISIT-RELIABILITY-CDF-01/
  20260830T134000Z__monotone-visit-reliability-cdf-s0-r1`；0 optimizer step、0 cohort evaluation。
- root cause：Python按官方语义将direct script目录而非current repo root置于`sys.path[0]`，`scripts.*` package import失败。
- recovery：只在launcher进程增加`PYTHONPATH=.`；不改代码模型/预算/数据/seed/steps/decision，r2从头训练并2/2支持。
- prevention：后续remote direct-script launcher沿用进程级repo-root path，不修改全局shell profile，也不增加入口测试矩阵。

下一可用编号为：`V67-F140`。

### V67-F140 — scene-held-out Beta map未稳定改善P173跨cohort概率刻度

- 分类：`algorithm/post-hoc-reliability-calibration-shift`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P174-GROUP-SPLIT-BETA-RELIABILITY-CALIBRATION-01/
  20260830T140000Z__group-split-beta-reliability-calibration-s0-r1`。
- 观察：calibrated CDF的Brier在旧四均优于calibrated horizon-only；但相对raw calibration-error change为
  `-10.77%/+16.63%/+11.05%/+5.80%`，mean=`+5.68%<10%`，P81反向，1/2 decisions。
- 解释：source-held-out global monotone map纠正总体skew，却不能适配P81 scene prevalence；post-hoc source calibration在shift下不稳定。
- 防重复：不扫scene split、Beta/temperature/isotonic map或降低10%门；P173只保留discriminative proper-score claim。

下一可用编号为：`V67-F141`。

### V67-F141 — direct integrated-Brier训练提升proper score但未修复marginal calibration

- 分类：`algorithm/proper-score-refinement-vs-calibration`；状态：`closed_negative_after_controlled_loss_change`。
- canonical：`run://worldsim_v67/WS-V67-P176-INTEGRATED-BRIER-VISIT-RELIABILITY-CDF-01/
  20260830T142000Z__integrated-brier-visit-reliability-cdf-s0-r1`。
- 观察：旧四Brier reduction mean=`40.95%`且逐组非退化；但model marginal error在四组均高于horizon-only，2/3 checks。
- 解释：Brier proper score同时包含calibration与refinement；P126 score带来的conditional refinement足以显著降Brier，但scene-level
  reliability prevalence偏移仍保留，所以低proper score不能单独升级为calibrated probability。
- 防重复：不混合BCE/Brier或扫loss weight；只允许一次scene-uniform source sampling检验grouping根因，P175候选仍为P173。

下一可用编号为：`V67-F142`。

### P175/P177 freeze note — fresh P173 confirmation and scene-uniform development remain separated

- P175 cohort在P174/P176结果之后、任何新sensor/target read之前冻结，10 scenes来自10 logs、四location；P173 artifact不变。
- P175只有mean Brier gain与mean marginal-error noninferiority两门；P174/P176/P177不得读取或改变P175。
- P177只改P176 source sampler为scene-uniform；若失败使用F142并关闭source-only calibration-training，不影响P175执行。

### V67-F142 — scene-uniform Brier仍不能消除跨cohort marginal calibration偏移

- 分类：`algorithm/scene-balanced-proper-score-calibration`；状态：`closed_negative_after_single_sampler_change`。
- canonical：`run://worldsim_v67/WS-V67-P177-SCENE-UNIFORM-BRIER-VISIT-RELIABILITY-CDF-01/
  20260830T142500Z__scene-uniform-brier-visit-reliability-cdf-s0-r1`。
- 观察：旧四mean Brier reduction=`40.82%`、逐组全优于control；但marginal calibration error仍4/4高于horizon-only，2/3 checks。
- 解释：按scene均衡source prior没有消除新cohort reliability prevalence差异；P173 signal稳定改善conditional refinement，但绝对概率
  刻度需要target-side information，不能由source sampler保证。
- 防重复：关闭source-only post-hoc/Brier/scene-balance calibration支线；不扫DRO/group weights。P175仍只确认冻结P173的
  proper-score discrimination，不升级probability claim。

下一可用编号为：`V67-F143`。

### V67-F143 — absolute clearance条件带来一致小增量但未达到冻结门

- 分类：`algorithm/mechanism-conditioned-reliability-calibration`；状态：`closed_negative_after_single_feature_change`。
- canonical：`run://worldsim_v67/WS-V67-P178-CLEARANCE-CONDITIONED-RELIABILITY-CDF-01/
  20260830T143000Z__clearance-conditioned-reliability-cdf-s0-r1`。
- 观察：旧四Brier相对P173全部改善`1.07%--5.20%`，calibration error也全部改善`3.85%--6.03%`；但mean=`5.08%<10%`。
- 解释：absolute inverse-clearance提供跨cohort一致的几何信息，却以budget-independent additive risk进入logit，不能表达P120代价中
  `budget × clearance`的乘法事件结构。
- 防重复：不降低10%门、不扫clearance变换/knots；下一步只允许把物理乘法直接写入预测对象。

下一可用编号为：`V67-F144`。

### V67-F144 — Actor set-context residual形成跨scene shortcut并破坏P173可靠性刻度

- 分类：`algorithm/set-context-residual-shift`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P179-SET-CONTEXT-RELIABILITY-CDF-01/
  20260830T143500Z__set-context-reliability-cdf-s0-r1`。
- 观察：P81/P96/P113 Brier分别回退`2.60%/13.79%/10.85%`，仅P129改善`2.60%`；mean calibration-error reduction=`-8.45%`。
- 解释：top-16 Actor token能降低source BCE，但mean+max context产生约`.92--1.03`的持续logit偏移，在location/scene prevalence变化时
  成为shortcut；预算单调性仍在，但绝对概率刻度恶化。
- 防重复：不扫DeepSet/attention pooling、token cap、residual bound或depth；回到显式cost factorization，P175候选不变。

下一可用编号为：`V67-F145`。

### V67-F145 — minimum-clearance有效阈值压缩破坏Actor/time误差配对

- 分类：`algorithm/mechanism-factorization-overcompression`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P180-EFFECTIVE-ERROR-THRESHOLD-RELIABILITY-CDF-01/
  20260830T144500Z__effective-error-threshold-reliability-cdf-s0-r1`。
- 观察：旧P81/P96/P113/P129 Brier相对P173回退`8.10%/27.11%/1.78%/11.17%`；mean calibration-error reduction=`-4.38%`。
- 解释：真实P120事件逐Actor/time计算`error_i / clearance_i`再取max；以全轨迹minimum clearance乘budget构成单一threshold，
  会把某一Actor的最小净空错误配给另一Actor的最大误差，导致系统性保守且损失判别信息。
- 防重复：不扫min/quantile/harmonic clearance聚合或threshold knots；转向scene-bootstrap模型边际化，保留原P173事件表示。

下一可用编号为：`V67-F146`。

### V67-F146 — scene-bootstrap CDF ensemble缺少function diversity且概率刻度不变

- 分类：`algorithm/bootstrap-ensemble-low-diversity`；状态：`closed_negative_after_single_trial`。
- canonical：`run://worldsim_v67/WS-V67-P181-SCENE-BOOTSTRAP-RELIABILITY-CDF-ENSEMBLE-01/
  20260830T145500Z__scene-bootstrap-reliability-cdf-ensemble-s0-r1`。
- 观察：四cohort Brier相对P173变化仅`-.58%/+ .14%/+ .08%/-.29%`，mean calibration-error reduction=`-.25%`；
  member probability deviation只有`.0128--.0166`。
- 解释：P173低维score/H/budget单调结构在64--69 unique-scene bootstrap环境收敛到几乎相同函数；权重边际化没有产生可利用的
  epistemic diversity，因此平均概率基本等于single P173。
- 防重复：不增加member count、随机seed、bootstrap size或temperature；转向对continuous cost本身建条件密度。

下一可用编号为：`V67-F147`。

### V67-F147 — P173 fresh confirmation保留proper-score优势但概率刻度未通过

- 分类：`algorithm/fresh-scene-reliability-calibration-shift`；状态：`closed_negative_primary_confirmation`。
- canonical：`run://worldsim_v67/WS-V67-P175-VISIT-RELIABILITY-CDF-CONFIRMATION-01/
  20260830T141500Z__visit-reliability-cdf-confirmation-s0-r1`。
- 观察：五H Brier reduction=`24.60%/33.42%/38.16%/38.41%/36.11%`、mean=`34.14%`通过；但model/control
  macro marginal calibration error=`.07102/.06101`，第二门失败。
- 解释：P126 trajectory score在全新scene仍显著增加conditional refinement，但P173 source prevalence刻度不能稳定迁移；这与旧四诊断一致。
- 防重复：不在P175 target上拟合Beta/temperature/isotonic、不降低门；P182已在P175结果完成前冻结不同density路线，P183另用全新cohort确认。

下一可用编号为：`V67-F148`。

