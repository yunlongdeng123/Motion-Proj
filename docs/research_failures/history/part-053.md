# 历史原始记录 053

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### P263/P264 milestone note — task-conditioned group dual通过并进入variable-set组合，无新增failure

- P263 canonical=`run://worldsim_v67/WS-V67-P263-TASK-HORIZON-GROUP-DUAL-01/20260831T124500Z__task-horizon-group-dual-s0-r1`；
- P201 fraction MAE=`.016964`、task-alpha Lagrangian regret=`-.00000144`、violations=0，2/2；
- P264只组合此前分别成立的P261 cardinality interpolation与P263 task preference，不增加attention/teacher/gate sweep；
- P264 P201 sizes48/96 aggregate fraction MAE=`.015254`、regret=`.00000191`、violations=0，2/2；
- P265改变task preference对象为四horizon simplex，P201 budget MAE=`.034543`、utility regret=`.0011145`、
  violations=0，2/2；P266只推进fixed-group dual，不扫P264 set architecture；next failure id保持F194。

### P266/P267 milestone note — simplex group dual通过，冻结最后一次variable-set组合，无新增failure

- P266 canonical=`run://worldsim_v67/WS-V67-P266-SIMPLEX-HORIZON-GROUP-DUAL-01/20260831T134500Z__simplex-horizon-group-dual-s0-r1`；
- P201 fraction MAE=`.015670`、simplex Lagrangian regret=`-.00001355`、violations=0，2/2；
- P267只组合已分别成立的variable set与simplex preference，预先声明其后关闭同轴叠加，不继续attention/size/vector sweep；
- P267 P201 sizes48/96 fraction MAE=`.013630`、regret=`-.00000606`、violations=0，2/2；同轴叠加关闭；
- P268切换为soft final-reliability-floor Lagrangian对象，固定unit penalty与双单调结构，不声称hard risk constraint；
- P268 P201 budget MAE=`.010533`、regret=`.0001539`、price/floor violations=0，2/2，candidate与teacher
  shortfall同为`.003167`；P269只推进fixed-group dual；next failure id保持F194。

### P257 milestone note — log-utility shadow-price policy通过，无新增failure

- canonical=`run://worldsim_v67/WS-V67-P257-LOG-UTILITY-SHADOW-PRICE-POLICY-01/20260831T104500Z__log-utility-shadow-price-policy-s0-r1`；
- P201 budget MAE=`.010997`、frozen log-utility regret=`.0002719`、price violations=0，2/2；
- epsilon固定`.05`且只运行一次，不做risk/price/grid/architecture sweep；
- next：P258在P256原样fixed-group合同下编译P257 dual price；下一failure id保持F192。

下一可用编号仍为：`V67-F192`。

### P254/P255 shadow-price milestone note — 单轨迹预算policy通过，无新增failure

- P254 canonical=`run://worldsim_v67/WS-V67-P254-SHADOW-PRICE-BUDGET-POLICY-01/20260831T091500Z__shadow-price-budget-policy-s0-r1`；
- P201 budget MAE=`.011423`、frozen utility regret=`.0001782`，2/2，price monotonicity violations=0；
- P255在P243 rows前冻结同读secondary，P254/P246/prices/grid/MC/两门不变；
- P256推进为固定64-row groups的dual-price compiler，仍只优化冻结surrogate；下一failure id保持F190。

下一可用编号仍为：`V67-F190`。

### P244 milestone note — 解析rate spline改善跨cohort fidelity，无新增failure

- canonical：`run://worldsim_v67/WS-V67-P244-MONOTONE-RATE-SPLINE-SURFACE-01/20260831T063000Z__monotone-rate-spline-surface-s0-r1`；
- 结果：P201/P183 final MAE=`.008973/.009665`，两者quality均改善、双轴violations=`0/0`；3/3；
- 边界：显存降低但当前小batch计时未显示加速，因此不把analytic integration包装成latency成功；
- next：P245在P243首次rows前冻结为same-read secondary；P246只扩有限budget范围，不改P243/P245合同。

下一可用编号仍为：`V67-F187`。

### P233 milestone note — 双轴结构单调surface通过，无新增failure

- P201 surface/final MAE均过门，surface Brier/calibration均优于teacher，两轴violations=`0/0`；
- P235只作一次marginal-only runtime interface ablation；P234将在P228首次quality前冻结同读fresh secondary；
- 不增加surface penalty、结构、budget/horizon或MC sweep。

下一可用编号仍为：`V67-F179`。

### P230 milestone note — marginal-only compiler通过，无新增failure

- P201 teacher MAE=`.009653`，Brier/calibration均优于teacher；P183轻微退化仍在冻结容差；
- 结果只支持student运行时接口简化，不证明horizon independence，也不改变P199/P203 teacher的语义；
- P231只作一次half-teacher/half-truth proper-loss训练；P228 quality仍未读取。

下一可用编号仍为：`V67-F178`。

### P229 milestone note — 64x64 compact student通过，无新增failure

- P201 teacher MAE=`.008252`，Brier relative degradation=`.159%`，calibration absolute increase=`-.000945`，2/2；
- 参数由P227的22,280降至7,048（`-68.37%`），没有width/depth sweep；
- P230只做一次marginal-only interface ablation；P228 fresh quality仍未读取。

下一可用编号仍为：`V67-F178`。

P205 locator recovery note（不占算法编号）：P205原waiter仍指向P201 cwd失败的r1目录，而P201 canonical已恢复为r2。
发现时P201 rows尚未生成；只将`frozen_rows.run`改到r2并以新run-id重启，P203 map、P199 comparator、budgets、MC、metrics、
gates均不变。P205 r2是唯一quality read，结果2/2；下一可用算法编号仍为`V67-F169`。

### P121 freeze note — continuous τ-conditioned boundary-state cost独立确认

- candidate：冻结P109 score，不使用已失败P120 learned head；continuous target、`.05m` floor、H3.5、fixed50全冻结；
- cohort：target-unread `0093/0332/0519/0014/0036/0221/0794/0916/0924/1062`，四location、内部10 distinct sessions；
  历史session overlap使证据只scene-level independent；
- decisions：ranking composite=`Spearman>=.70`且比clearance高≥`.10`；selection composite=`cost reduction>=.70`且cost不高于
  clearance。只保留两门，不加binary flip/AUROC/gate matrix；
- outcome：冻结protocol未改，P121 2/2 decisions通过；P128也通过且未占用F92。没有第二P121 recovery或新增gate；
  `V67-F92`恢复为下一可用编号。

下一可用编号仍为：`V67-F92`。

### V6.6 当前边界（2026-08-28）

- V6.6已终态`v66_research_complete_arxiv_report_ready`。P3C/P6/P8R分别支持independent legacy local ranking、
  Actor-preserving package与synthetic bounded response；`V66-F02`记录natural physical repair terminal negative。
  Plan规定P7 FAIL不得进入RL，所以P9/P10/P11未执行；任何physical repair或matched RL必须在新版本开始。
- V6.6 已从合入 V6.5 终态 `288fa9f` 的 `main` 建立；当前解锁P3L低容量local geometry head，仍只在
  consumed legacy train/selection角色内执行。
- 直接继承 `V65-F19`：visited-state reliability 不得重命名为 direct action authority；V6.6 validity 轴必须与
  hazard 轴分离。继承 `V64-F28`：手工低维 collision critic 不是正式 RL。继承 `V1-F06`：稀疏 cut-in pool
  不得成为主数据入口或论文成立条件。
- P1-D 的 synthetic hazard attribute 不是物理 cut-in/collision edit；P10V 六场景已消费，只能作 mechanism，
  不得替代 fresh selection/confirmation。`V66-F01 resolved`记录deterministic injected certificate对natural local
  geometry conflict的0 recall，以及P3L/P3C两级certificate恢复。`V66-F02 closed_negative_after_single_recovery`
  记录P7 triage无法迁移为physical repair。`V66-F03 resolved_by_single_implementation_recovery`记录P8低速
  stop-state重复jerk update及single-rate-limiter修复；
  下一可用编号=`V66-F04`。
- 本次 P0 只做最小研究冻结，无 smoke/regression matrix、无新 hash/checksum/fingerprint；failure ledger delta=`none`。
- P1-D evaluator实现与`py_compile`通过，未创建formal run、未读quality，未出现工程或算法失败；下一编号仍为
  `V66-F01`。q0在representation-level paired corruption中保持原score是预注册的actor-blind baseline语义，不能解释为
  对重新渲染artifact的实测不敏感。
- P1-D formal一次完成且4/4 development gates通过，无新failure。构造性certificate满分只解除“factor接口是否存在
  signal”的开发前置，不解除natural artifact truth、fresh generalization或真实hazard edit风险；下一编号仍为`V66-F01`。
- P2-D独立certificate入口已实现但尚未formal执行；无新failure。它必须从observable factors重算reason codes，禁止复用
  P1预写decision冒充独立证书；下一编号仍为`V66-F01`。
- P2-D一次完成且8/8 gates通过，无新failure。P3在deterministic injected development上因P2 AUROC/AUPRC已经1而无
  预注册相对增量headroom，保持locked/not executed；这不是learned model失败，也不得用同数据训练后声称超过P2。
  下一编号仍为`V66-F01`。
- P4-D matched repair入口已实现但未formal执行；无新failure。DROP消除violation时必须同时报告hazard event loss，
  ABSTAIN不得把不可发出的geometry记作已修复，REPAIR不得改变Actor ID/track/trajectory/hazard attribute；下一编号仍为
  `V66-F01`。
- P4-D一次完成且R2 6/6 gates通过，无新failure。R0 DROP的hazard retention=0.5说明“删掉artifact actor”会在配对构造中
  明确制造easier-world；但这是positive comparator observation，不单列failure。R2满分仍只限deterministic paired factor，
  natural/fresh边界未解除；下一编号仍为`V66-F01`。
- P2N natural-conflict诊断已实现但尚未读target；无新failure。observed-FREE actor boundary只定义local geometry
  conflict，禁止升级成Actor existence artifact；若deterministic证书transfer失败，必须登记`V66-F01`并先检索外部方案。

<a id="detail-v66"></a>

## V6.6 HARP-Compiler 详细账本（2026-08-28）

### V66-F01 — deterministic injected certificate不覆盖natural Actor-owned local geometry conflict

- 分类：`algorithm/evaluation`；状态：`resolved_by_two_level_certificate`。
- 观察：P2N独立legacy cohort含891 actor-unit，498个存在至少一个target observed-FREE boundary point；P2 injected
  certificate对这些local conflicts的recall=`0`、AUROC=`0.5`、AUPRC=`0.558923`。q0有弱signal但不足以单独作为
  authority：AUROC/AUPRC=`0.543745/0.612874`，rate Spearman=`0.267650`。
- 根因：P2 factor只覆盖support缺失、duplicate、lifecycle、kinematic/identity与shape injection。natural conflict常发生在
  已有Actor hit/current/swept support的局部owned geometry，Actor existence证据不能替代primitive geometry validity。
- 推翻项：推翻“deterministic injected certificate可直接迁移到natural local geometry”的假设；不推翻Actor existence
  protection，也不把local conflict升级为whole-Actor artifact。
- 防重复/复开：禁止降低hidden-FREE label、扫count/rate threshold、把conflict Actor直接DROP或在P10X反复调模型。
  合法恢复必须使用instance-evidence local geometry head，train/selection scene-disjoint；P10X只允许一次selection，之后需
  另一独立cohort确认。
- 外部检索迁移：CVPR 2024 Symphonies的instance query/context、GaussianFormer object-centric sparse representation、
  Cam4DOcc 4D instance occupancy以及CVPR evidential occupancy的unknown/contradiction建模，落地为两级certificate而非
  替换backbone。详见`P3L_INSTANCE_EVIDENCE_MIGRATION_FREEZE.md`。
- 证据：`WS-V66-P2N-NATURAL-ACTOR-CONFLICT-DIAGNOSTIC-01` /
  `20260828T090228Z__natural-actor-conflict-s0-r1`；恢复证据=`WS-V66-P3L-ACTOR-LOCAL-GEOMETRY-HEAD-01` /
  `20260828T091036Z__local-geometry-head-s0-r1`与`WS-V66-P3C-INDEPENDENT-LOCAL-GEOMETRY-CONFIRM-01` /
  `20260828T091611Z__independent-local-geometry-confirm-s0-r1`。恢复不改变deterministic certificate的原始0 recall。

P3L固定8维/2x32/seed0 single selection已完成：P10X AUROC/AUPRC=`0.652365/0.692384`，相对deterministic
增加`+0.152365/+0.133461`，6/6 scenes above chance。`V66-F01`状态转为`recovering`，尚需独立cohort no-refit
确认才可关闭；下一编号仍为`V66-F02`。禁止在P10X继续扫参或解析threshold。

P3C已在读取local-conflict label前固定V65 P2V六场景独立cohort、P3L checkpoint及三项ranking/support gates；
实现尚未创建formal run，无新failure，`V66-F01`保持recovering、下一编号仍为`V66-F02`。

P3C exact-once 4/4 gates通过：581 actor-unit上AUROC/AUPRC=`0.761644/0.767165`，相对deterministic
增加`+0.261644/+0.238766`，6/6 scenes above chance。`V66-F01`由“deterministic existence protection + learned
local geometry ranking”恢复关闭；不赋予Actor drop authority。下一可用编号仍为`V66-F02`。

P6 bake实现已就绪但未创建formal run。continuous score没有被事后阈值化，所有Actor保留；runtime不加载模型/
hidden target，hazard不控制existence。没有新增failure，下一可用编号仍为`V66-F02`。

P6 formal 6/6 gates通过：581 actor states全部保留，metadata complete=1，actor removed=0，hidden-target fields=0。
这只证明runtime package capability，不把未执行的physical repair记成artifact下降；无新failure，下一可用编号仍为
`V66-F02`。

P7 fixed-budget audit已在读取本阶段指标前冻结，formal run尚未创建。动作只作triage estimand，禁止把预测处理数冒充
物理修复数；无新failure，下一可用编号仍为`V66-F02`。

