# 历史原始记录 045

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### P140 freeze note — source-scene bootstrap member diversity

- method：每member固定从102 source scenes有放回抽102次，重复scene完整保留其Actor-time tokens；三独立models/NLL/steps/batch/
  projection匹配P139/P126。
- evaluation/decisions：primary完成后才消费P129，并与P81/P96/P113共同相对P126检验四cohort cost nonregression及mean
  Spearman gain≥`.005`。实际P113/P129 cost改善，但P81/P96 cost微退；mean Spearman gain=`-.004052`，两门失败。
- prevention：不扫bootstrap fraction/member/seed/weight/coverage；只作一次scene-bagged development。

### V67-F103 — scene bootstrap局部改善cost但未保持四cohort排序

- 分类：`algorithm/ensemble-diversity-unit`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P140-SCENE-BAGGED-DEEP-ENSEMBLE-01/
  20260830T100000Z__scene-bagged-deep-ensemble-s0-r1`。
- 观察：P113/P129 cost改善到`.216172/.303464`，但P81/P96为`.179070/.167830`而回退；四cohort mean rank
  gain=`-.004052`。所有members收敛，非执行失败。
- 解释：scene omission增加了diversity，却降低部分常见source-mode的排序精度；单靠bootstrap unit不能稳定超过natural-token P126。
- 防重复：不扫bootstrap fraction或seed。P141恢复natural-token训练，仅检验独立支持后增加member count。

### P141 freeze note — five-member natural-token ensemble scaling

- method：复用P126 seeds0/1/2，只按exact P126 protocol新增seeds3/4；形成5-member total variance。
- evaluation/decisions：consumed P81/P96/P113/P129相对P126，四cohort cost nonregression且mean Spearman gain≥`.003`。
  实际mean gain=`+.000557`，P96/P113 cost回退，两门失败。
- prevention：固定5 members与seeds3/4，只跑一次；不扫member/seed/weight/projection/coverage。

### V67-F104 — 五成员规模增加未形成稳定continuous increment

- 分类：`algorithm/ensemble-size-scaling`；状态：`closed_negative_after_one_fixed_scale_trial`。
- canonical：`run://worldsim_v67/WS-V67-P141-FIVE-MEMBER-DEEP-ENSEMBLE-01/
  20260830T100500Z__five-member-deep-ensemble-s0-r1`。
- 观察：P81/P96/P113/P129 rank gain=`+.000445/+.002049/-.000543/+.000275`，mean=`+.000557<.003`；
  P96/P113 cost回退。新增两members训练收敛。
- 解释：更多自然分布members只降低Monte Carlo noise，未改变task alignment；三成员已经捕获主要increment。
- 防重复：不再扫member count/seed。P142改变预测对象为query-conditioned projected residual distribution。

### P142 freeze note — task-conditioned projected residual ensemble

- method：直接训练`p(n(τ)^T e | τ, Actor, t)`三成员异方差Gaussian；输入24 query features+time fraction+normal，
  target是真实projected residual，不是teacher/cost/event。
- decisions/outcome：consumed P81/P96/P113/P129相对P126四cohort cost nonregression、mean Spearman gain≥`.005`；实际
  mean rank gain=`+.000037`，仅P113 cost改善，P81/P96/P129回退，两门失败。
- prevention：一次固定3-member trial，不扫input/loss/member/seed/weight/coverage；支持才做fresh confirmation。

### V67-F105 — direct task-conditioned projection在P129增益但跨cohort不稳

- 分类：`algorithm/prediction-object-transfer`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P142-TASK-CONDITIONED-PROJECTED-ENSEMBLE-01/
  20260830T101000Z__task-conditioned-projected-ensemble-s0-r1`。
- 观察：P129 rank gain=`+.013189`、P113 cost改善，但P96 rank=`-.009093`且cost回退；四cohort mean rank接近0，
  P81/P96/P129 cost不满足nonregression。三models均完成6,000 steps。
- 解释：query-conditioned projected residual含有效新信息，但从头替换通用2D Actor distribution造成source query geometry shortcut。
- 防重复：不扫P142 inputs/architecture。P143只学P126 standardized residual correction，保留通用base。

### P143 freeze note — P126-based conditional residual correction

- method：冻结P126 `μ0/σ0`，三member学习`z=(n^T e-μ0)/σ0` conditional distribution；input追加`μ0/logσ0`，
  final mean/variance以base scale重构。
- decisions/outcome：consumed P81/P96/P113/P129相对P126四cohort cost nonregression、mean Spearman gain≥`.005`；实际
  mean gain=`-.013445`且四组cost全回退，两门失败。
- prevention：不扫correction mixing/weight/input/loss/member/seed/coverage；失败关闭conditional residual route。

### V67-F106 — P126-based standardized residual correction仍破坏跨cohort排序

- 分类：`algorithm/conditional-recalibration-transfer`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P143-CONDITIONAL-RESIDUAL-ENSEMBLE-01/
  20260830T101500Z__conditional-residual-ensemble-s0-r1`。
- 观察：四cohort selected cost全高于P126，rank gain=`-.022649/-.013583/+.001054/-.018600`；训练NLL正常收敛。
- 解释：source conditional residual不是可迁移calibration map；即使以P126 scale标准化，query distribution shift仍改变排序。
- 防重复：关闭per-time conditional distribution/correction family，不扫mixing weight。P144迁移到trajectory set和direct cost rank。

### P144 freeze note — P126-anchored trajectory-set rank compiler

- method：top16 P126-risk Actor-query tokens，经Deep Sets mean+max输出bound `.5` residual并加回P126 trajectory score；
  source scene内trajectory pairs用真实continuous cost ordering监督。
- decisions/outcome：consumed P81/P96/P113/P129相对P126四cohort cost nonregression、mean Spearman gain≥`.005`；实际
  mean rank=`-.000962`，仅P129 cost改善，P81/P96/P113回退，两门失败。
- prevention：唯一set compiler trial；不扫top-k/architecture/pairs/bound/loss/seed/coverage。

### V67-F107 — full trajectory set compiler仍未解决P96 transfer

- 分类：`algorithm/downstream-set-authority-transfer`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P144-TRAJECTORY-SET-RANK-COMPILER-01/
  20260830T102000Z__trajectory-set-rank-compiler-s0-r1`。
- 观察：P81/P96/P113/P129 rank gain=`+.001499/-.007292/+.001589/+.000355`；仅P129 cost改善，其他三组回退。
- 解释：完整set features/aggregation不能消除source→P96 shift；下游capacity不是主要瓶颈。
- 防重复：关闭trajectory residual compiler，不扫top-k/bound/architecture。P145修复上游absolute-time alias。

### P145 freeze note — absolute future-time conditioned Actor ensemble

- method：P126 actor inputs只追加`fraction×H` absolute seconds并保留fraction；其余3-member diagonal Gaussian协议完全匹配。
- source/evaluation：source H=`.8/1.5/2.5/3.0`，四consumed evaluation均H3.5 extrapolation。
- decisions/outcome：相对P126四cohort cost nonregression、mean Spearman gain≥`.005`；实际三cohort rank改善但P96
  `-.016165`，mean=`-.001551`，仅P81 cost改善，两门失败。

### V67-F108 — absolute-time signal未抵消从头重训的P96 representation drift

- 分类：`algorithm/horizon-conditioning-transfer`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P145-ABSOLUTE-TIME-ACTOR-ENSEMBLE-01/
  20260830T102500Z__absolute-time-actor-ensemble-s0-r1`。
- 观察：P81/P113/P129 rank增益为正，但P96=`-.016165`；仅P81 cost改善，mean rank=`-.001551`。
- 解释：absolute time有跨三cohort信息，但从头训练同时改变P126 mean representation，无法归因到time-varying scale。
- 防重复：不扫time input。P146冻结P126 mean/network，只训练monotone absolute-time scale adapter。

### P146 freeze note — frozen-P126 monotone time-scale adapter

- method：每个P126 member/axis只训练bias与positive absolute-time slope，共12 scalars；mean/features全部冻结。
- decisions：consumed H3.5 P81/P96/P113/P129相对P126四cohort cost nonregression、mean Spearman gain≥`.005`。
- prevention：固定adapter form/steps/LR；不扫slope、loss、seed、weight或coverage。

### V67-F109 — monotone absolute-time scale未跨四cohort迁移

- 分类：`algorithm/horizon-conditioned-scale-transfer`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P146-MONOTONE-TIME-SCALE-ADAPTER-01/
  20260830T103000Z__monotone-time-scale-adapter-s0-r1`。
- 观察：learned slopes全部为正且P129 selected cost从`.308669`降至`.296310`，但P81/P113 cost回退；四cohort
  Spearman gain=`-.004452/+.002360/-.001189/-.003149`，mean=`-.001607`，0/2 decisions。
- 解释：冻结P126 mean后排除了representation drift，但单调axis-wise scale growth仍不足以表达跨时序的相关残差结构。
- 防重复：关闭scalar time adapter，不扫slope/form。P147改做新scene多时域确认；P148改预测完整9步残差序列分布。

### P147/P148 freeze note — multi-horizon confirmation + full-sequence training

- P147：新10-scene scene-level independent cohort、五个H、P126/P109、fixed50/`.05m`以及两个macro decisions已冻结；
  只有pre-target exact shard locator可修正。
- P148：三member、Actor+H输入、完整`9×2` diagonal sequence输出、6,000 steps/member和四consumed cohort decisions冻结；
  不做DCT/architecture/loss/seed/coverage sweep。
- operations：P147 IO/preprocess/evaluator与P148 3090训练并行；不新增hash/checksum/fingerprint或回归矩阵。

### V67-F110 — full-resolution sequence mean未恢复continuous ranking

- 分类：`algorithm/temporal-sequence-transfer`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P148-FULL-SEQUENCE-ACTOR-ENSEMBLE-01/
  20260830T104500Z__full-sequence-actor-ensemble-s0-r1`。
- 观察：四cohort Spearman均下降，gain=`-.013157/-.011868/-.012388/-.012105`，mean=`-.012380`；仅P96
  selected cost微降。member projected epistemic fraction仅`.018--.045`。
- 解释：完整9步共同decoder与absolute H并未优于P126逐时刻tokens；问题不只是P115 DCT压缩，单峰diagonal sequence仍不足。
- 防重复：不扫P148 hidden/steps/member。检索trajectory-set、multi-scale VAE与uncertainty-aware diffusion后，P149只迁移
  coherent sequence-level mixture和any-time boundary event score。

### P149 freeze note — coherent trajectory mixture

- method：4 sequence modes，各mode完整9步mean/scale；整序列mixture NLL，score为mode-weighted any-time boundary crossing。
- decisions：consumed P81/P96/P113/P129相对P126 cost全不退、mean Spearman gain≥`.005`。
- prevention：固定4 modes/8,000 steps/seed0；不扫component/architecture/loss/weight/coverage，不重复P125 per-time K2。

### V67-F111 — coherent trajectory modes未转化为continuous reliability ranking

- 分类：`algorithm/coherent-multimodal-transfer`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P149-COHERENT-TRAJECTORY-MIXTURE-01/
  20260830T105000Z__coherent-trajectory-mixture-s0-r1`。
- 观察：mean max component weight约`.53--.56`，故四mode实际active；但四cohort cost全退，Spearman gain=
  `-.099259/-.191974/-.053506/-.048404`，mean=`-.098286`。
- 解释：整序列multimodality虽然提高source likelihood，但mode-weighted any-time crossing与continuous normalized error ranking严重错位；
  失败不是单纯mode collapse。
- 防重复：不扫component/diversity/any-crossing aggregation。P150直接训练continuous reliability cost的query-time稠密局部项。

### P150 freeze note — dense task-conditioned boundary-cost distribution

- target/input：`log1p(|n·residual|/clearance)`；24 query features+fraction+normal+log clearance；5.18M source tokens。
- model/decision：3 Gaussian members、fixed 1σ upper score；四consumed cohort cost全不退且mean rank gain≥`.005`。
- prevention：固定upper sigma/architecture/loss/member/seed/coverage；不重复P120 P109-summary post-hoc head。

### V67-F112 — direct dense cost对象仍在P96发生ERM transfer反转

- 分类：`algorithm/direct-cost-domain-transfer`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P150-DENSE-BOUNDARY-COST-ENSEMBLE-01/
  20260830T105500Z__dense-boundary-cost-ensemble-s0-r1`。
- 观察：P81/P129 Spearman gain=`+.005119/+.005795`，说明对象对齐有信号；但P96=`-.028055`、P113=`-.004568`，
  P129 selected cost回退至`.343517`，mean gain=`-.005427`，0/2 decisions。
- 解释：稠密直接监督解决了P149 any-crossing严重错位，却未解决跨scene/domain排序稳定性；P96是主要反转点。
- 防重复：不扫1σ/architecture/loss。P151只改训练risk aggregation为scene×horizon worst-group NLL。

### P151 freeze note — scene-horizon group-DRO dense cost

- environments/object：source scene×horizon；P150 target/input/network/1σ score全部不变。
- objective：每batch 64 groups×1,024 tokens，优化最差25% group NLL均值；3 members、6,000 steps/member。
- prevention：不扫group fraction/environment/IRM penalty/architecture/loss/member/seed/upper sigma/coverage。

