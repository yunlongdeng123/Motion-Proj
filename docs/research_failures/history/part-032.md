# 历史原始记录 032

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F206 — P306等价执行图被额外fine-tune破坏跨场景fidelity

- canonical=`run://worldsim_v67/WS-V67-P306-SHARED-CONTEXT-PIECEWISE-AUTHORITY-COMPILER-01/
  20260901T023000Z__shared-context-piecewise-authority-s0-r1`；
- result：P201 attained MAE=`.0306373`未过 P301 `.0285229`门；regret=`5.6335e-5`、violations=0、forward=
  `.5659s`均通过，3/4；source attained=`.0220001`；
- root cause：shared-context graph在同权重下与P302等价，失败来自不必要的6k继续训练再次放大source→P201 shift；
  不是 context factoring数值错误；
- recovery：P306R按 Network Morphism原则锁死 P302 weights、0 steps，仅改执行图；恢复 P302 attained=
  `.02565857`并把forward降到`.56142s`，4/4 supported，F206关闭；
- next object/result：P307改为精确6-action task-conditioned sets；沿用P306R但不再训练任意scene chunks。P201
  attained MAE=`.02973484`、regret=`2.78998e-5`、violations=`0`，3/3 supported；F206恢复路径由执行图优化继续推进到
  task-conditioned authority且没有新增failure；下一步仅做冻结模型跨cohort确认；
- P307后续：P308以P307 weights在P243九场景285个六动作组上0-step确认，attained MAE=`.02415646`、regret=
  `3.04462e-5`、violations=0，3/3 supported；未触发failure，也未refit/换cohort/放宽门；
- next object：authority fidelity已跨cohort成立，转向预算到显式action admission/selection的decision层；失败才使用
  `V67-F207`，不再以P307/P308同指标微调补救；
- P309 freeze：P307三点authority base + `.25` bounded residual + differentiable top-2，真实visited-state cost两门；
  P201 reduction=`60.90%`、pairwise=`.83580`，2/2 supported，未触发F207；
- P310 freeze：P309/P307在P277 180个六动作组上0-step action-admission-family untouched确认；失败才登记F207，
  result reduction=`63.72%`、pairwise=`.83069`、6/6 scenes降低，2/2 supported；未触发F207；
- P311 result：为排除低progress混杂，冻结P309并学习正progress-preference rate；P201 reduction=`39.26%`、
  pairwise=`.89740`、full-progress `.8459→.9164→.9459`且violations=0，3/3 supported；
- P312 freeze：P311/P309/P307在P243 285个六动作组上0-step family-untouched确认；失败才登记F207，不refit、
  result reduction=`37.65%`、pairwise=`.87384`、full-progress `.8070→.9000→.9474`且violations=0，3/3
  supported；未触发F207；下一步进入二维task condition而非重复同轴确认；
- P313 result：冻结P311/P309/P307，训练progress×continuous lateral command二维task condition；P201 reduction=
  `35.66%`、pairwise=`.88795`、command response=`.89590`、progress violations=0，3/3 supported；未触发F207；
- P314 freeze：P313全链在P277 180个六动作组上0-step two-axis family-untouched确认；失败才登记F207，不refit、
  result reduction=`36.70%`、pairwise=`.91799`、command response=`.98889`、progress violations=0，3/3
  supported；未触发F207；下一步转向set-level quantile certificate而非继续ranking；
- P315 result：冻结P313 top-2，训练q50/q80/q95 admitted-set max visited-state cost head；P201 coverage=
  `.64016/.85574/.97869`、max undercoverage=`-.02869`、median log-cost MAE=`.17716`、order violations=0，
  2/2 supported；未触发F207，不扫quantile/width/loss/gate；下一步仅做冻结证书跨cohort确认；
- P316 result：P315完整证书在P277上0-step确认，coverage=`.66944/.95000/.98750`、max undercoverage=
  `-.03750`、median log-cost MAE=`.12101`，2/2 supported；未触发F207；
- P317 result：冻结P315 q95 base训练bounded residual ceiling-conditioned selective authority；P201 mean/highest
  coverage=`.26066/.70000`（baseline `.04672/.14016`）、max unsafe=`.00937`、monotonicity violations=0，
  3/3 supported；未触发F207；最严格ceiling选择0，记录为预期abstention边界而非failure；
- P318 result：P317全链在P243上0-step确认，mean/highest coverage=`.31257/.67719`、max unsafe=`.02332`、
  monotonicity violations=0，3/3 supported；未触发F207，不再增加同family confirmation；
- P319 result：13-task minimum-deviation projection的risk/monotonicity通过，但mean coverage gain=
  `.05847 < .10`，1/3失败，登记F207；严格ceiling仍无feasible task，不降门/删ceiling/扩task grid；
- P320 result：逐pair bounded residual editor的q90 margin=`.51617`，P201 mean coverage gain=`-.00328 < .10`，
  risk/monotonicity通过但coverage失败，登记F208；不调margin/quantile/gate；
- P321 result：groupwise selector + selected-outcome q90训练收敛，但P201 mean coverage gain=`-.00109 < .10`，
  risk/monotonicity通过，登记F209；关闭task/pair repair family，不调selected q90/weight/architecture/gate；
- P322 r1：P199 run ID少`reliability`，artifact load FileNotFound，0 training/0 quality，登记F210；r2只修
  路径后P201 heldout task+2.5s coverage=`.54754/.82131/.95492`、median MAE=`.15154`、H/quantile
  violations=0，3/3 supported，F210关闭；
- P323 result：P322冻结certificate在P277上0-step horizon-family untouched确认，coverage=
  `.56806/.90694/.98472`、median MAE=`.09370`、H/quantile violations=0，3/3 supported；未触发F211，
  不再增加同family confirmation；
- P324 result：P322 q95 base上训练task×H×ceiling selective authority；P201 risk/monotonicity通过，但mean
  coverage=`.19836 < .20`，以`.00164`未过，登记F211；不舍入通过、不降门、不在P201 recalibrate；
- P325 result：normalized q90 time-varying scale使P201 coverage=`.21831`、max unsafe=`.08333`、
  monotonicity=0，3/3 supported，F211关闭；source strict仅20 admissions且unsafe=`.55`作为限制保留，
  不事后新增gate；
- P326 result：q85 continuous-H decision boundary使P201 coverage=`.32077`、max unsafe=`.04072`、
  monotonicity=0，source heldout-H max unsafe=`.03902`，3/3 supported；未触发F212；
- P327 result：risk-conditioned H×q surface在P201 q90得到coverage=`.25847`、max unsafe=`.03175`、
  monotonicity=0，source heldout H+q90 max unsafe=`.02763`，3/3 supported；未触发F212；
- P328 result：continuous-q implicit surface在P201 q90得到coverage=`.24945`、max unsafe=`.02217`、
  monotonicity=0，3/3 supported；未触发F212；但q90 offset=`.06317`且strict coverage=0作为capacity限制保留；
- P329 result：P201 coverage=`.23415`、max unsafe=`.01761`、monotonicity=0，3/3 gate supported；未触发
  F212，但q90 offset=`.08060`、strict coverage=0且被P326/P327支配，关闭spline tuning；
- P330 result：nested k=`1/2/3`在P201得到mean any-authority coverage=`.35082`、max unsafe=`.07006`、
  size/ceiling violations=0，3/3 supported；strict coverage=`.03934`，fixed-top2空集被cardinality缩放解决；
- P331 result：max unsafe=`.03764`与monotonicity=0通过，但mean any-authority coverage=
  `.24672 < .30`，2/3 rejected，登记F212；不降门或扫q/range/capacity/seed；
- P332 result：train pinball=`.01629`，但q90 offset=`.08820`；P201 coverage=`.24153 < .30`，risk/
  monotonicity通过、coverage失败，登记F213；不扫lattice knots/capacity；
- P333 result：P201 mean/high coverage=`.30464/.68197`、max unsafe=`.04902`、size/ceiling violations=0，
  3/3 supported；相对P332 mean coverage=`.24153`且strict=0，局部尺度恢复strict=`.06475`并关闭F213；
  source scale range=`[1.15e-9,3.38496]`作为限制保留；
- P334 result：固定`.02`下界解决near-zero scale，P201 risk/monotonicity通过但coverage=
  `.28989 < .30`，2/3 rejected，登记F214；不扫floor/steps/range；
- P335 result：q90逐值保持P333，P201 coverage/max unsafe=`.30464/.04902`、monotonicity=0，3/3
  supported；q75/q85/q90/q95 quantile-order violations=0，F214关闭；
- P336 result：source heldout mean pinball=`.057986 <= P335 .058153`，P201 q90三门与q-order通过，
  4/4 supported；增益仅`.29%`且q95 unsafe=`.05405`未改善，关闭高阶curvature sweep；
- P337 r1：numeric scene ID `%5`使fit remainder0为空，首个sample前退出，0 training/quality，登记F215；
- P337 r2：改为sorted unique-scene rank folds后，source pinball改善`1.17%`，P201 q90三门与q-order通过，
  4/4 supported；q95 coverage/unsafe=`.29262/.04819`，F215关闭；
- P338 result：P201 expected-size MAE/accuracy=`.02280/.99788`且q90 teacher risk/coverage通过，3/3
  supported；但mean temperature=`.0050018`塌到下界，只支持fidelity，不声称宽梯度；
- P339 result：P201 mean temperature=`.0200016`、expected-size MAE/accuracy=`.09824/.99092`，q90 teacher
  risk/coverage通过，4/4 supported；停止temperature sweep；
- P340：non-anchor coverage改善但risk excess `.003892 > P337 .003571`，3/4 rejected，登记F216；
- P341：条件风险primal-dual使coverage升至`.34954`、q95改善，但q85 risk excess扩大到`.00534`，
  3/4 rejected，登记F217；
- P342：source最坏组风险改善，但P201 q95 risk excess `.003892 > .003571`，3/4 rejected，登记F218；
- P343 r1：插值rows接错single-set路径，训练实际重复P342；step1001主动中止，登记F219；r2 active；
- P343 r2：P201 coverage与risk excess两门均比P337差，2/4 rejected，登记F220；
- P344：P201 q90 risk/coverage `.12088/.28005`，两门失败，登记F221；
- P345：group calibration将P201 q90 risk降到`.10714`，但coverage降到`.27869`，2/4 rejected，登记F222；
- P346：P201 q90 coverage/risk `.32842/.09091`，四门supported；source heldout-H仍漂移，仅保留development claim；
- P347：domain近完美可分导致importance ESS=`69.62`，q90 coverage/risk `.29836/.10588`，登记F223；
- P348：unconditional alignment的q90 coverage/risk `.37350/.17323`，risk显著恶化，登记F224；
- P349：q90 `.30055/.09375`四门supported，但step6k后adversarial collapse，保留unstable limitation；
- P350：稳定训练但q90 `.29973/.18667`，risk semantics丢失，登记F225；
- P351：teacher anchor改善coverage但q90 risk仍`.13514`，登记F226；target-unlabeled adaptation family关闭；
- P352：source q90通过但P201 mean-pool q90 risk=`.17518`，登记F227；
- P353：frozen upper-pool将P201 q90 risk降至`.13115`但仍失败，登记F228；
- P354：memberwise upper-pool q90 risk=`.04795`但coverage=`.24454`，登记F229；
- P355：feature-aware BCE q90 coverage=`.40929`但risk=`.20548`，登记F230；
- P356：primal-dual q90 coverage=`.40628`但risk=`.21019`，登记F231；feature calibrator family关闭；
- P357：monotone-horizon q90 source/P201 risk=`.26173/.16364`，登记F232；
- P358：fixed-grid multi-horizon q90 source/P201 risk=`.34097/.11236`，登记F233；
- P359：horizon-group q90 coverage/risk=`.29317/.10526`，登记F234；
- P360：one-sided q90 coverage/risk=`.34290/.10753`，登记F235；
- P361：pairwise q90 coverage/risk=`.34317/.12766`，登记F236；
- 下一可用 failure id 为 `V67-F237`；当前按用户额度收口，无active experiment。

P361 r1 operational note（不占用新failure id）：当前PyTorch无`torch.flatnonzero`，故在step1 loss反传前退出，
无checkpoint、PAV或P201 scientific read。官方API确认以`torch.nonzero(..., as_tuple=False).flatten()`返回等价1-D
索引；r2只做该兼容修复，全部冻结参数不变。下一可用failure id仍为`V67-F236`。

### V67-F216 — P340固定risk odds优化目标与hard conditional unsafe endpoint错位

- canonical=`run://worldsim_v67/WS-V67-P340-DECISION-FOCUSED-DIFFERENTIABLE-AUTHORITY-01/
  20260901T120000Z__decision-focused-differentiable-authority-s0-r1`；
- symptom：P201 non-anchor coverage `.343260 > P337 .339800`，但最大nominal-risk excess
  `.003892 > .003571`；q90 exact与risk/coverage门均通过，总计3/4 rejected；
- cause：`q/(1-q)`乘soft unsafe mass属于固定标量化，未除以soft admission coverage，故与最终准入集合内
  unsafe rate不等价；
- literature/migration：NeurIPS 2020 PAC constrained learning与NeurIPS 2023 resilient constrained
  learning支持显式dual替代固定penalty；P341直接优化条件风险；
- forbidden rescue：不扫odds coefficient、multiplier bound、temperature或seed；resolution=`closed by P341
  objective alignment, transfer issue moved to F217`。

### V67-F217 — P341平均source条件风险约束未迁移到heldout task/horizon

- canonical=`run://worldsim_v67/WS-V67-P341-CONDITIONAL-RISK-PRIMAL-DUAL-AUTHORITY-01/
  20260901T121500Z__conditional-risk-primal-dual-authority-s0-r1`；
- symptom：P201 non-anchor coverage `.349545`显著高于P337，q95 unsafe `.050279 < .053571`，但q85
  unsafe=`.155340`，最大risk excess=`.005340 > .003571`，3/4 rejected；
- diagnosis：q85三个source平均dual均归零，说明训练均值约束未看见heldout 3.0s/插值task condition的最坏风险；
- literature/migration：NeurIPS/ICLR group DRO把目标改为有限组最大风险；P342用27个training horizon×task
  condition组的最坏conditional risk；
- forbidden rescue：不调dual learning rate、augmented penalty、margin、capacity或seed；resolution=
  `open via P342 worst-group conditional-risk training`。

### V67-F218 — P342 source最坏组约束未覆盖目标插值task support

- canonical=`run://worldsim_v67/WS-V67-P342-WORST-GROUP-CONDITIONAL-RISK-AUTHORITY-01/
  20260901T123000Z__worst-group-conditional-risk-authority-s0-r1`；
- symptom：source q95 unsafe `.050891 < P337 .053030`且P201 non-anchor coverage `.345993 > .339800`，
  但P201 q95 unsafe `.053892`使risk excess `.003892 > P337 .003571`，3/4 rejected；
- diagnosis：27组dual全部按原training task grid构造；target的progress `.25/.75`和command `-.5/.5`位于组间，
  source worst-group成立不蕴含这些插值条件成立；
- literature/migration：ICLR 2024 conformal risk control的shift extension与group-conditional calibration均要求
  calibration distribution/strata匹配目标；P343在source trajectory上生成相同插值task strata，无需P201标签；
- forbidden rescue：不扫group definition、dual LR、margin、temperature或seed；resolution=
  `open via P343 interpolated-task worst-group training`。

### V67-F219 — P343 r1插值task rows未接入multi-set authority路径

- run=`run://worldsim_v67/WS-V67-P343-INTERPOLATED-TASK-WORST-GROUP-AUTHORITY-01/
  20260901T124500Z__interpolated-task-worst-group-authority-s0-r1`；
- symptom：step1/501/1001 loss与conditional risk和P342逐位相同，说明新task strata未改变batch；
- cause：interpolated source concat实现位于single-set materialization else，而P343使用multi-set size prefix分支；
- response：step1001主动SIGINT避免继续浪费GPU；在每个set-size local source生成后拼接相同interpolated rows，
  r2 step1 risk `.236447 != r1 .229960`；
- scope：r1没有完成summary或读取quality，不构成科学negative；不更改P343 hypothesis/gates/hyperparameters；
  resolution=`closed by multi-set wiring fix 41e22b8 and P343 r2`。

