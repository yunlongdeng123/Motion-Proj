# 历史原始记录 051

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F174 — absolute proper-loss authority未迁移到更难P201 cohort

- canonical：`run://worldsim_v67/WS-V67-P222-SELECTIVE-AUTHORITY-P201-TERTIARY-01/20260830T234000Z__selective-authority-p201-tertiary-s0-r1`；
- 观察：P220在source/P183分别改善selected Brier `23.61%/13.71%`，但P201相对confidence Brier退化`1.31%`、
  calibration退化`9.67%`，0/2；P201全量Brier`.09397`明显高于P183`.07212`；
- 解释：MSE预测absolute realized loss在source difficulty regime有效，但固定覆盖率只需要稳定次序；难度基线变化会损害绝对回归排序；
- 文献响应：selective uncertainty margin ranking及ranking-aligned decision losses直接优化相对次序。P223只试一次同budget
  pairwise logistic，不使用P183/P201 labels训练；
- 防重复：P223若未在P201过原两门即关闭learned authority head，保留confidence control；不扫coverage/margin/auxiliary/group。

下一可用编号：`V67-F175`。

### V67-F175 — pairwise loss ranking缩小但未消除P201 selective reversal

- canonical：`run://worldsim_v67/WS-V67-P223-PAIRWISE-SELECTIVE-AUTHORITY-RECOVERY-01/20260830T235500Z__pairwise-selective-authority-recovery-s0-r1`；
- 观察：相对P220，P201 Brier退化由`1.31%`缩至`.43%`、calibration退化由`9.67%`缩至`2.14%`，但仍0/2；
  source/P183保持正向，说明ranking alignment有帮助但逐budget ordering仍跨cohort不稳；
- 解释：compiler真正授权的是trajectory reliability interface，逐budget独立top-50会产生七个不同接受集，并放大budget-specific shift；
- 响应：关闭逐budget authority；P224只试一次structured trajectory-curve selection，把七预算proper loss聚合后作单一授权；
- 防重复：不再试逐budget MSE/pairwise/listwise/margin/group loss或coverage sweep。

下一可用编号：`V67-F176`。

### V67-F176 — 整条trajectory curve的learned authority在P201更强反转

- canonical：`run://worldsim_v67/WS-V67-P224-TRAJECTORY-CURVE-AUTHORITY-01/20260831T001000Z__trajectory-curve-authority-s0-r1`；
- 观察：source/P183均正向，但P201 selected Brier退化`7.48%`、calibration退化`22.25%`，0/2；
- 解释：聚合七预算并未消除cohort difficulty shortcut，反而使一个错误排序同时影响完整curve；confidence control更稳；
- 响应：关闭所有learned authority heads；只允许冻结P203 calibrated confidence的一次selection/output attribution；
- 防重复：不再训练authority MSE/pairwise/listwise/curve/group/ensemble/router或扫coverage。

下一可用编号：`V67-F177`。

### V67-F177 — calibrated-confidence selection改善Brier但未跨cohort保持选择后校准

- canonical：`run://worldsim_v67/WS-V67-P226-CALIBRATED-CONFIDENCE-SELECTION-ONLY-01/20260831T003000Z__calibrated-confidence-selection-only-s0-r1`；
- 观察：仅改selection时P183/P201 raw-P199 selected Brier均改善`2.02%/1.31%`，但P201 calibration退化`1.63%`，
  source Brier/calibration也退化`.82%/12.11%`，cross-cohort composite 1/2；
- 解释：P203共享单调map只改变约3.6% trajectory membership；其小Brier收益不足以形成稳定selection-conditional calibration；
- 响应：关闭selective authority并保留raw confidence作为描述性control；P227转向不改变teacher semantics的单调curve distillation；
- 防重复：不扫coverage、map、confidence functional、mixing或selector；不申请selective fresh cohort。

下一可用编号：`V67-F178`。

### P227 milestone note — 单调curve distillation通过，无新增failure

- canonical：`run://worldsim_v67/WS-V67-P227-MONOTONE-RELIABILITY-CURVE-DISTILLATION-01/20260831T004500Z__monotone-reliability-curve-distillation-s0-r1`；
- 结果：P201 teacher MAE=`.007633`，Brier相对退化=`-.229%`，calibration absolute increase=`-.000903`，2/2；
- 边界：这是P201已观察后的post-hoc development，不消除P220--P226 authority failures，也不授予selective/planner authority；
- next：P228已冻结全新10-scene/10-log cohort确认；P229只在其IO期间训练单个compact候选，不读P228 quality。

下一可用编号仍为：`V67-F178`。

### V67-F178 — half-teacher/half-truth改善proper score但越过teacher fidelity边界

- canonical：`run://worldsim_v67/WS-V67-P231-TRUTH-REGULARIZED-MONOTONE-CURVE-01/20260831T014500Z__truth-regularized-monotone-curve-s0-r1`；
- 观察：P183/P201 Brier改善`2.02%/1.29%`、calibration均改善，但P201 teacher MAE=`.027831`，超过冻结`.02`；
- 解释：固定50/50 output-space混合把student从teacher compiler推成新的source-supervised predictor；quality改善不能覆盖语义失败；
- 文献响应：NeurIPS 2020 PCGrad和ICLR 2026 DTO-KD均从gradient conflict/dynamic balance处理多目标，而不是继续扫静态权重；
- response：P232只试一次conflict projection + gradient norm matching；P231不降gate、不扫mixing、不进P228 fresh；
- 防重复：不再试静态teacher/truth ratio、temperature、truth auxiliary或label smoothing sweep。

下一可用编号：`V67-F179`。

### P232 milestone note — gradient-level balance恢复teacher fidelity，无新增failure

- 9,672/10,000 training steps检测到冲突；projection + norm matching没有静态loss-weight；
- P201 teacher MAE=`.009108`，Brier/calibration均改善，3/3；P183仍在容差内；
- P233改变输出对象为结构单调的budget×prefix surface，不继续扫truth/teacher balancing。

下一可用编号仍为：`V67-F179`。

### V67-F179 — marginal-only surface在最终曲线fidelity上极小但严格失败

- canonical：`run://worldsim_v67/WS-V67-P235-MARGINAL-ONLY-PREFIX-SURFACE-01/20260831T023500Z__marginal-only-prefix-surface-s0-r1`；
- 观察：P201 surface MAE与quality两门通过，Brier/calibration改善`.509%/.000614`，但final MAE=`.010090>.01`；
- 解释：28 marginal values足以拟合surface平均行为，但直接删除8个condition features仍损失最终joint-curve细节；
- 文献响应：privileged/missing-modality distillation建议恢复缺失representation后复用teacher head；P236训练feature hallucinator；
- response：不放宽`.01`、不round pass、不扫feature subset；P235不进入fresh P234；
- 防重复：不再直接训练marginal-only surface width/depth/penalty/seed或budget/horizon变体。

下一可用编号：`V67-F180`。

### V67-F180 — deterministic privileged-feature hallucination放大最终曲线偏差

- canonical：`run://worldsim_v67/WS-V67-P236-PRIVILEGED-FEATURE-HALLUCINATION-SURFACE-01/20260831T031000Z__privileged-feature-hallucination-surface-s0-r1`；
- 观察：P201 condition-feature RMSE=`.3433`，surface quality仍改善，但final MAE=`.014007>.01`；
- 解释：marginal CDF对raw condition是many-to-one，确定性point alignment不能恢复copula conditioning细节；
- 文献响应：NeurIPS 2024 PCD同样指出missing-modality information asymmetry使deterministic alignment过严；
- response：关闭hallucination，不引入概率feature sampling/conformal machinery；P237从真实上游raw conditions端到端amortize；
- 防重复：不试hallucinator width/depth/adversary/output loss/probabilistic family或sample-count sweep。

下一可用编号：`V67-F181`。

### P228/P234 fresh milestone note — 两个冻结compiler均通过，无新增算法failure

- P228 full-curve primary：MAE `.007443`、Brier改善`.316%`、calibration increase仅`.000020`，3/3；
- P234 prefix-surface same-read secondary：surface/final MAE `.007101/.009483`、Brier/calibration均改善、violations=`0/0`；
- delayed wrong-root launcher note：最初PowerShell复合SSH中的相对`--runs-root runs`子命令在prep退出后延迟启动，
  只在repo-local隔离目录重复物化rows并于artifact load前失败；canonical绝对runs-root P228/P234早已完成。失败副本移至
  `/tmp/p228_delayed_wrong_runs_root_196944`，未改变scene/model/metric/verdict，不占算法failure编号；
- preparation canonical已移至`/root/autodl-tmp/runs`，repo worktree保持clean；未添加hash/checksum/fingerprint。

下一可用编号仍为：`V67-F181`。

### V67-F181 — raw-condition end-to-end surface无法恢复conditional-density刻度

- canonical：`run://worldsim_v67/WS-V67-P237-RAW-CONDITION-PREFIX-SURFACE-01/20260831T033000Z__raw-condition-prefix-surface-s0-r1`；
- 观察：P201 surface MAE和quality composite过门，但final MAE=`.016343>.01`；P183 final MAE=`.015716`；
- 解释：固定source cohort中从8个conditions重新学习P182 density+C199 dependence的复合映射，丢失了显式marginal刻度；
- response：关闭input reduction，保留fresh-supported P233输入；P238改研究连续budget CDF能力，不再删输入；
- 防重复：不试raw-only width/depth/loss/seed或混合少量marginal subsets。

下一可用编号：`V67-F182`。

### V67-F182 — 全局logistic-mixture连续CDF在heldout预算失真

- canonical：`run://worldsim_v67/WS-V67-P238-CONTINUOUS-BUDGET-PREFIX-SURFACE-01/20260831T040000Z__continuous-budget-prefix-surface-s0-r1`；
- 观察：P201 heldout surface/final MAE=`.015630/.021162`，calibration increase=`.005166`，0/3；
- 解释：4-component全局shape即使单调，也无法保留七个已确认knots之间的局部曲率与calibration刻度；
- 文献响应：NeurIPS 2019 monotone rational-quadratic splines强调通过局部bins与knots提升单调transform灵活性；
- response：P239直接保留P233 knots并作local log-budget interpolation；不增加components或再训全局CDF；
- 防重复：不扫mixture K、temperature、scale floor、width/depth或heldout budget定义。

下一可用编号：`V67-F183`。

### V67-F183 — knot-preserving局部插值仍无法满足最终曲线fidelity

- canonical：`run://worldsim_v67/WS-V67-P239-KNOT-PRESERVING-BUDGET-INTERPOLATION-01/20260831T042000Z__knot-preserving-budget-interpolation-s0-r1`；
- 观察：P201 heldout surface MAE=`.013768`过门，Brier改善`1.082%`且calibration increase仅`.000371`，但final
  teacher MAE=`.017072>.01`；2/3；
- 解释：保留P233七knots消除了P238的全局shape误差，却不能从端点恢复full-prefix在P203校准前后的区间曲率；
- response：P240只检验一次“逆P203—raw插值—重施P203”，不引入spline degree/derivative/knot sweep；
- 防重复：不降低final gate、不把更好Brier替代teacher fidelity、不扫linear/cubic/RQ spline family。

下一可用编号：`V67-F184`。

### V67-F184 — 校准感知插值仍失真并破坏prefix层级单调性

- canonical：`run://worldsim_v67/WS-V67-P240-CALIBRATION-AWARE-BUDGET-INTERPOLATION-01/20260831T044000Z__calibration-aware-budget-interpolation-s0-r1`；
- 观察：P201 surface/final MAE=`.013514/.016057`，quality composite通过但final gate失败；budget violations=0，
  horizon violations=2,041；
- 解释：只变换full-prefix可略减final误差，却与前三个prefix的线性轨迹失配；误差主要不是单一P203 link造成；
- literature response：NeurIPS 2019 UMNN通过正导数积分提供比固定全局mixture或事后link插值更灵活的单调函数；ICML
  2023进一步指出朴素正权重结构的凸性限制；
- response：关闭无训练连续预算后处理，P241直接在稠密source budget targets训练双轴结构单调integral surface；
- 防重复：不再试probit/logit/beta-link、分prefix map、projection或spline-form sweep。

下一可用编号：`V67-F185`。

### V67-F185 — P241 r1将dense target预算数误耦合为feature维度

- run：`run://worldsim_v67/WS-V67-P241-INTEGRATED-MONOTONE-BUDGET-SURFACE-01/20260831T050000Z__integrated-monotone-budget-surface-s0-r1`；
- symptom：旧`_dataset`按传入budget数量拼接marginal feature；31个target budgets产生`8+4x31=132`输入，而冻结
  P233-compatible encoder为36维，首次forward抛`8192x132`与`36x128`矩阵维度错误；
- exposure：teacher arrays已物化，但optimizer step=0、parameter update=0、P183/P201 metric/gate read=0；算法假设未判；
- resolution：输入独立调用七个anchor budgets固定为36维；31点调用只取teacher target，六heldout midpoints仍完全排除；
  model/seed/steps/batch/width/quadrature/decision均不变，r2继续；
- prevention：连续query模型必须区分context anchor grid与teacher query grid；不为此增加smoke/regression matrix。

下一可用编号：`V67-F186`。

### V67-F186 — 正导数积分surface显著收敛但final MAE窄幅失败

- canonical：`run://worldsim_v67/WS-V67-P241-INTEGRATED-MONOTONE-BUDGET-SURFACE-01/20260831T051000Z__integrated-monotone-budget-surface-s0-r2`；
- 观察：P201 heldout surface MAE=`.007950`、Brier/calibration均优于teacher、双轴violations=`0/0`，但final MAE=
  `.010320>.01`；2/3；P183/source final MAE=`.010869/.010666`；
- 解释：相比P238的`.021162`，positive-rate conditional integral已把final误差降低51.2%，剩余边界与训练MSE和decision
  MAE不一致相符，而非明显shape/monotonicity失败；
- literature response：ICCV 2019 regression distillation给出absolute-error/Laplace imitation；CVPR 2024 KD-DETR也用
  L1对齐回归teacher/student outputs；
- response：P242从头单次训练，仅把MSE改为L1；不改结构、数据、budget、steps、seed或decision；
- 防重复：不round pass、不放宽`.01`、不扫L1/Huber/log-cosh混合、prefix weights、quadrature或width。

下一可用编号：`V67-F187`。

