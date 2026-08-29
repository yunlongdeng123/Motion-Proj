# Motion-Proj 统一失败、风险与防重复账本

## V6.7 P269/P270 latest failures（2026-08-31）

### V67-F194 — P269 训练完成后的 final-reliability shortfall 轴切片错误

- run：`run://worldsim_v67/WS-V67-P269-RELIABILITY-FLOOR-GROUP-DUAL-01/20260831T145000Z__reliability-floor-group-dual-s0-r1`；
- symptom：12,000-step GPU training 完成后，`_utility` 将形状 `(group,price,member,horizon)` 的 final reliability 误切为
  `(group,price,horizon)`，与 alpha utility 的 `(group,price,member)` 无法广播，正式 source/P183/P201 summary 未生成；
- root cause：NumPy 从 trailing axes 对齐广播；原 `probability[:,ai,li,:,-1]` 的 `-1` 落在 member 轴而非 horizon 轴；
- response：参考 NumPy 官方 broadcasting 规则与 ICLR 2022 einops 的显式轴思想，唯一修复为
  `probability[:,ai,li,:,:, -1]`。不引入依赖、不增加 gate/test，不改训练合同；commit=`020726a`；
- impact：r1=`implementation_failed_after_training/no_verdict`，不解读训练 loss 为科学支持；同合同 r2 的 P201
  budget-fraction MAE=`.012541`、regret=`4.19e-6`，2/2 通过，状态=`resolved_by_minimal_axis_fix`。

### V67-F195 — P270 两次 pre-science 启动环境错误

- r1 在 import 阶段因直接脚本入口未含 repo root，报 `ModuleNotFoundError: scripts`；r2 加 `PYTHONPATH=.` 后因误传
  repo-local `runs`，在首个冻结 artifact load 前报 `FileNotFoundError`；
- exposure：两次都未完成 frozen artifact load、teacher target、optimizer step 或 metric read，不构成科学 trial；
- resolution：r3 只修启动命令为 `PYTHONPATH=.` 与 absolute runs-root `/root/autodl-tmp/runs`；实验配置、seed、模型、
  teacher、门和 claim 不变；
- impact：canonical scientific run 迁移到 `20260831T151000Z__tail-cvar-equivariant-allocator-s0-r3`；该 run P201 budget
  MAE=`.011861`、regret=`9.24e-5`，2/2 通过，状态=`resolved_before_scientific_trial`。

下一可用编号：`V67-F196`。

### P269 r2 / P270 r3 / P271 r1 outcome note — 无新增 failure

- P269 最小轴修复后 2/2；P270 fixed64 tail-CVaR allocator 2/2；P271 held-out sizes 48/96 亦 2/2；
- 三者均保持预冻结模型、条件轴、训练步数与 decision gates，没有以重复 smoke/regression 或阈值扫描补救；
- P272 现只把冻结 P271 primal 连接到 attainable-budget dual，当前运行中；下一 failure id 仍为 `V67-F196`。

### P272/P273 outcome note — 无新增 failure

- P272 在 P201 held-out cardinality 上 2/2，随后冻结 primal/dual；P273 在 allocation-family untouched 的 P243 九场景
  reuse 上按原两门一次 confirmation，attained fraction MAE=`.016783`、regret=`8.44e-6`，再次 2/2；
- 没有更换 cohort、重训、refit、gate/threshold/cardinality sweep；P243 曾用于 surface confirmation 的消费边界已显式保留；
- tail-CVaR compiler family 至此关闭为 supported；下一可用 failure id 仍为 `V67-F196`。

### P274 bootstrap epistemic surface note — active / 无新增 failure

- 新对象不改 P270--P273 verdict；5-member scene-bootstrap monotone ensemble现正训练；
- P201只用两个预冻结决策，P243明确 descriptive-only；无 hash/checksum/fingerprint 或 smoke/regression matrix；
- 下一可用 failure id 仍为 `V67-F196`。

### P274 outcome / P275 start note — 无新增 failure

- P274 P201 final MAE=`.009363`、epistemic-error Spearman=`.693292`，2/2；P243仅描述且方向一致；
- P275 只在 P274 成立后启动，固定蒸馏 `mean-beta*std` 并用解析正 rates 编码 beta/budget 单调，不更改 P274 门；
- 下一可用 failure id 仍为 `V67-F196`。

### P275 outcome / P276 start note — 无新增 failure

- P275 P201 surface/final MAE=`.002235/.002861`、三轴 violations=0，2/2；P243继续只描述；
- P276 以冻结 P275 为唯一 teacher，soft-floor allocator当前运行；不将 LCB score包装为 credible interval/hard constraint；
- 下一可用 failure id 仍为 `V67-F196`。

### P276 outcome / P277 IO / P278 GPU overlap note — 无新增 failure

- P276 P201 budget MAE=`.011545`、regret=`.0001335`，2/2；
- P277六个新场景在 sensor/target read 前冻结，archive/preprocess 与 P278 fixed-group dual GPU training 并行；
- 当前没有资源不足或多卡需求；下一可用 failure id 仍为 `V67-F196`。

### P278 outcome / P279 composite-risk start note — 无新增 failure

- P278 P201 attained fraction MAE=`.016307`、regret=`8.11e-6`，2/2；
- P279 只在 P274--P278 均支持后组合 epistemic LCB 与 Actor-tail CVaR；预先保留“经验 composite score、非 coherence
  proof/credible interval/posterior-predictive risk”的 claim boundary；
- P277 archive IO 仍与 P279 GPU 并行；下一可用 failure id 仍为 `V67-F196`。

### P279 outcome / P280 composite-risk dual start note — 无新增 failure

- P279 P201 budget MAE=`.0117673`、composite-risk regret=`.000105044`，两门通过，解析 price/floor violations=`0/0`；
- P280 只把冻结 P279 primal 接到 attainable-fraction dual，不改变 composite-risk 定义、P279 verdict 或 claim boundary；
- 一次本地 PowerShell 启动命令漏写 `ssh`，在本机 `Set-Location` 阶段退出，未接触远端、未创建 run、未加载模型，
  属于控制端命令失误而非科学/实现 trial；随后正确 SSH 命令已启动 canonical r1；
- P277 IO 继续并行；无资源不足或多卡需求，下一可用 failure id 仍为 `V67-F196`。

### P280 outcome / P281 variable-cardinality start note — 无新增 failure

- P280 P201 attained fraction MAE=`.0162331`、composite regret=`2.67e-6`，2/2，fraction-price violations=0；
- P281 仅将冻结 P279 warm-start到 sizes32/64/128，并在未参与训练的48/96上一次评估，不改变风险对象或门；
- P277 IO 与 P281 GPU 继续并行；下一可用 failure id 仍为 `V67-F196`。

### V67-F196 — P282 frozen P199 run ID复制时漏词

- run：`run://worldsim_v67/WS-V67-P282-VARIABLE-SET-EPISTEMIC-TAIL-CVAR-DUAL-01/
  20260831T193000Z__variable-set-epistemic-tail-cvar-dual-s0-r1`；
- symptom：配置将 canonical `...__joint-horizon-reliability-copula-s0-r2` 误写成少一个 `reliability` 的路径，
  在 `torch.load` 首个 P199 artifact 时 `FileNotFoundError`；
- exposure：P126/P182虽加载，但 P199/copula、P203、P275/P281、teacher targets、optimizer step和任何 metric 均未发生；
- research response：Hydra defaults composition 与 OmegaConf interpolation均建议复用共享配置值，避免变体配置重复；考虑当前
  项目已冻结 PyYAML 合同且禁止扩大工程范围，本轮不迁移框架，只从已通过 P281/P279 配置恢复同一 canonical ref；
- resolution：commit=`e806de4`；r2=`20260831T193500Z__variable-set-epistemic-tail-cvar-dual-s0-r2` 已按原 seed、
  data、model、steps、gates启动；状态=`resolved_before_scientific_trial`。

下一可用编号：`V67-F197`。

### P282 r2 outcome / P283 conformalized surface start note — 无新增 failure

- P282 r2 P201 attained fraction MAE=`.0137322`、composite regret=`4.17e-6`，2/2；F196 已按相同合同关闭；
- P283 参考 conformal risk control / risk-averse calibration，把 P274 ensemble disagreement校准为 tolerance-conditioned
  multiplier；由于 unit scene-correlated，只预注册 empirical coverage，明确禁止 distribution-free/conditional guarantee；
- P283 GPU 与 P277 archive IO 并行；下一可用 failure id 保持 `V67-F197`。

### V67-F197 — P283 ensemble-std multiplier在 P201 严重欠覆盖

- run：`run://worldsim_v67/WS-V67-P283-CONFORMALIZED-EPISTEMIC-LCB-SURFACE-01/
  20260831T194500Z__conformalized-epistemic-lcb-s0-r1`；
- symptom：student对 frozen conformal teacher 的 P201 surface MAE=`.004687`、三轴 violations=0，但四个 heldout delta 的
  simultaneous coverage仅 `.666/.647/.592/.471`，maximum undercoverage=`.2590`，超过冻结 `.12`；
- diagnosis：校准 multipliers 高达 `4.35--9.77` 仍欠覆盖；P243 max undercoverage=`.2923` 同向，说明 P274 std 在
  low-disagreement biased rows 上是错误尺度，不能靠 multiplier 蒸馏容量或更多 steps 修复；
- literature response：dependent-data conformal工作强调 exchangeability破坏会损失 coverage；ICML/UAI 的 group/cluster
  conformal与 sequential residual方法改用/分组 residual score。当前先做最小对象迁移：P284 使用未标准化 additive
  one-sided residual quantile；不调 P283 gate、不扫 multiplier、不把 P283 1/2 包装为成功；
- claim impact：关闭 standardized ensemble multiplier 的 empirical coverage claim；P274 disagreement-error association 与
  P275 beta-conditioned score fidelity仍保留，但都不升级为 coverage guarantee。

下一可用编号：`V67-F198`。

### P284 additive residual recovery start note

- P284 同一 calibration split/delta/student/steps/gates，仅替换 nonconformity score，canonical r1 已启动；
- 即使 P284 通过，也只支持 observed-cohort empirical coverage，不支持 iid/exchangeable、conditional或 safety guarantee；
- P277 IO 继续并行；下一可用 failure id 保持 `V67-F198`。

### P277 fresh confirmation terminal note — 无新增 failure

- targeted shard扫描、2,341-member extraction和6/6 preprocess全部完成，无 recovery cohort、scene替换或质量预读；
- formal one-shot 在1,080 trajectories上 budget MAE=`.0153153`、regret=`.0008650`，2/2通过；
- 该 positive result只确认 frozen P275/P276 surrogate transfer，不能修复或覆盖 P283 的 standardized coverage failure；
- 下一可用 failure id 保持 `V67-F198`。

### P284 outcome / P285 allocator start note — 无新增 failure

- P284 P201 fidelity MAE=`.005598`、max undercoverage=`.06208`，2/2；P243描述性结果同向；
- additive residual structural recovery支持 empirical tolerance-conditioned LCB surface，但不反向改写 P283 standardized
  multiplier failure，也不升级为 distribution-free/conditional guarantee；
- P285 只在 P284 成立后冻结 teacher 并训练 soft-floor allocator；下一可用 failure id 保持 `V67-F198`。

### P285 outcome / P286 group dual start note — 无新增 failure

- P285 P201 budget MAE=`.0113088`、regret=`5.01e-5`，2/2；price/floor violations=0；
- P286 仅将冻结 P285 primal 接到 attainable-fraction dual，不更改 P284 empirical coverage或 P285 allocation verdict；
- 下一可用 failure id 保持 `V67-F198`。

### P286 outcome / P287 calibrated composite-risk start note — 无新增 failure

- P286 P201 attained fraction MAE=`.0202496`、group regret=`9.55e-6`，2/2；
- P287 仅在 P284--P286 成立后组合 additive empirical LCB 与 Actor-tail CVaR，预先保留 no coherent-risk/formal-
  coverage/hard-constraint claim boundary；
- 下一可用 failure id 保持 `V67-F198`。

### P287 outcome / P288 variable-cardinality start note — 无新增 failure

- P287 P201 budget MAE=`.0130764`、composite regret=`8.71e-5`，2/2，price/floor violations=0；
- P288 仅 warm-start同一架构到 train sizes32/64/128并在48/96评估，不改变 P284 coverage或 P287 risk定义；
- 下一可用 failure id 保持 `V67-F198`。

### P288 outcome / P289 variable-set dual start note — 无新增 failure

- P288 P201 budget MAE=`.0108336`、composite regret=`3.39e-5`，heldout sizes48/96 2/2；
- P289 仅编译冻结 P288 的 attainable-budget dual，不改变 additive coverage、tail-risk对象或 cardinality protocol；
- 下一可用 failure id 保持 `V67-F198`。

### P290 additive-family confirmation freeze note — 无新增 failure

- P277 rows虽然已被旧 beta-LCB family消费，但 additive P284--P289 从未训练/选模/读取其 quality，复用边界已显式写入；
- P290 只在 P289 通过后做一次 frozen chain read，不以 confirmation 反向调 P284/P288/P289；
- 下一可用 failure id 保持 `V67-F198`。

### P289/P290 outcome note — 无新增 failure

- P289 P201 price/attained-fraction MAE=`.0267769/.0143254`、composite regret=`5.28e-6`、violations=0，2/2；
- 冻结后一次读取的 P290 P277-reuse confirmation 为 attained-fraction MAE=`.0122735`、composite regret=
  `1.85e-5`、violations=0，2/2；未据此反向调参或更换 cohort；
- P290 仅是 additive-family untouched reuse；P277 cohort 已被旧 beta-LCB P277 消费，不记为全项目 fresh；
- 下一可用 failure id 保持 `V67-F198`。

### V67-F198 — P291 symmetric L1 student破坏 adaptive teacher coverage

- canonical=`run://worldsim_v67/WS-V67-P291-CONTEXT-ADAPTIVE-ADDITIVE-LCB-SURFACE-01/
  20260831T214500Z__context-adaptive-additive-lcb-s0-r1`；
- symptom：teacher P201 coverage=`.92829/.84311/.73917/.65513` 接近 desired `.925/.85/.75/.65`，但 symmetric L1
  student coverage=`.79963/.73361/.67050/.59656`，max undercoverage=`.125366>.12`；fidelity 与 efficiency 两门通过，
  coverage门失败；
- root cause：MAE=`.0042137` 很小但未限制误差方向，LCB student 小幅系统上偏会被 simultaneous-horizon coverage
  放大；这与 NeurIPS 2023 报告的 distillation confidence exaggeration 一致；
- closed candidate：不再继续 symmetric L1、step/width/lr/threshold 扫描；P292 不得引用 rejected P291；
- structural recovery：P291R 冻结全部 adaptive teacher与三门，仅用 fixed lower-quartile pinball 做单侧蒸馏；
- 下一可用 failure id 为 `V67-F199`。

### V67-F198 resolution / P292 start note — 无新增 failure

- P291R P201 max undercoverage=`.051625`、mean conservatism=`.0158992 < P284 .0172214`，3/3；说明单侧
  distillation 在不放弃 adaptive efficiency 的情况下恢复 empirical coverage；
- P292 只编译冻结 P291R 到 P285 同构 soft-floor allocator，不改变 coverage对象、student loss或 P285两门；
- 下一可用 failure id 保持 `V67-F199`。

### P292 outcome / P293 group-dual start note — 无新增 failure

- P292 P201 budget MAE=`.0215751`、LCB-floor regret=`.0003612`，2/2，price/floor violations=0；
- P293 只编译冻结 P292 的 attainable-budget fixed64 dual，不改变 P291R coverage或 P292 utility定义；
- 下一可用 failure id 保持 `V67-F199`。

### P293 outcome / P294 adaptive tail-CVaR start note — 无新增 failure

- P293 P201 attained-fraction MAE=`.0255409`、group regret=`9.79e-5`，2/2，violations=0；
- P294 将冻结 P291R 作为 coverage layer，并以 P287 同一 empirical tail-CVaR定义训练 fixed64 primal；不修改
  P291R coverage、tail mass、eta/grid或 gates；
- 下一可用 failure id 保持 `V67-F199`。

### P294 outcome / P295 variable-cardinality start note — 无新增 failure

- P294 P201 budget MAE=`.0220461`、composite regret=`.00023215`，2/2，price/floor violations=0；
- P295 仅 warm-start同一架构到 sizes32/64/128并在48/96评估，不改变 P291R coverage或 P294 tail-risk定义；
- 下一可用 failure id 保持 `V67-F199`。

### P295 outcome / P296 variable-set dual start note — 无新增 failure

- P295 P201 budget MAE=`.0204663`、composite regret=`.00021336`，heldout sizes48/96 2/2，violations=0；
- P296 只编译冻结 P295 attainable-budget dual，不改变 P291R coverage、P294 tail-risk或 cardinality protocol；
- 下一可用 failure id 保持 `V67-F199`。

### P297 direct compiler freeze note — 无新增 failure

- P297 不是绕过 empirical constraint：teacher budgets仍由冻结 P295 + 同一 bisection target定义；变化只在推理
  路径从 dual→primal 两调用改为 fraction→budgets 单调用；
- fraction取负后进入 P295 positive price-rate axis，固定结构保证更多 attainable budget不产生更低 budgets；
- 仅在 P296 通过后训练，不以 P296/P297 结果调整 gate；下一可用 failure id 保持 `V67-F199`。

### P296 outcome / P297 direct training start note — 无新增 failure

- P296 P201 attained-fraction MAE=`.0240720`、composite regret=`1.58e-5`，heldout sizes48/96 2/2，violations=0；
- P297 按冻结协议启动，训练 teacher仍为 P295+bisection；P296仅作两阶段 baseline，不作为 P297训练组件；
- 下一可用 failure id 保持 `V67-F199`。

### P297 outcome / P298 interaction-aware start note — 无新增 failure

- P297 P201 attained-fraction MAE=`.0297261`、regret=`6.62e-5`、violations=0，2/2；direct compiler成立，
  但 constraint fidelity弱于 P296 two-stage `.0240720`，准确记录为可行性而非质量提升；
- P298 只加入 zero-gated one-block 4-head set attention并预注册必须严格优于 P297 attained MAE；不进行
  head/depth/width/step/gate sweep；
- 下一可用 failure id 保持 `V67-F199`。

### V67-F199 — P298 self-attention未改善 direct constraint fidelity

- canonical=`run://worldsim_v67/WS-V67-P298-ATTENTIVE-DIRECT-AUTHORITY-COMPILER-01/
  20260901T000000Z__attentive-direct-authority-s0-r1`；
- symptom：P201 attained MAE=`.0334487`，高于冻结 P297 baseline `.0297261`；通用 fidelity/regret门仍通过，
  comparative improvement门失败；size48/96=`.0310884/.0358091`；
- root cause：新增 pairwise capacity未直接约束 group mean budget，且 forward `.3904s` 高于 P297 `.3188s`；
  当前证据不支持 attention 是 direct authority fidelity 的必要结构；
- closed candidate：不扫 attention heads/depth/width/normalization/steps；保留 P297 direct baseline；
- structural pivot：P299 回到 P297 架构，等权加入 group-mean budget distillation loss，直接对齐 attained constraint；
- 下一可用 failure id 为 `V67-F200`。

### V67-F200 — P299 group-mean soft loss未跨 cohort改善 direct constraint

- canonical=`run://worldsim_v67/WS-V67-P299-CONSTRAINT-AWARE-DIRECT-AUTHORITY-COMPILER-01/
  20260901T001500Z__constraint-aware-direct-authority-s0-r1`；
- symptom：source attained MAE=`.0225111 > P297 .0204608`，P201=
  `.0317413 > P297 .0297261`；regret与通用 fidelity仍过门，但 comparative门失败；
- root cause：soft mean matching仍只是 teacher imitation，并未在推理结构中强制 fraction与 group mean的关系；
  source/P201差异表明固定 loss weight不能保证跨 cohort constraint；
- closed candidate：不扫 group-loss weight、Actor/group loss比例、steps/lr；保留 P297 direct baseline；
- structural pivot：P300 采用同模型0/1端点的自一致可微 mean projection；
- 下一可用 failure id 为 `V67-F201`。

### V67-F201 — P300 uniform-shift projection破坏逐 Actor fraction单调性

- canonical=`run://worldsim_v67/WS-V67-P300-PROJECTED-DIRECT-AUTHORITY-COMPILER-01/
  20260901T003000Z__projected-direct-authority-s0-r1`；
- registered gates：P201 attained MAE=`.0120266`、regret=`.00014114`、严格优于 P297，3/3；预注册 verdict不反改；
- latent structural defect：P201 fraction-budget monotonicity violations=`10,754`（size48/96=`4,744/6,010`），
  source=`20,746`。uniform mean shift + box clip可保 aggregate mean，却不能保每个 Actor budget顺序；
- accurate disposition：记录 supported constraint fidelity，但 P300不进入最终 compiler，不写 monotone authority claim；
- structural recovery：P301 同一 model 0/1 endpoints逐元素 convex combination已通过，P201 attained MAE=
  `.0285229`、regret=`7.7359e-5`、violations=`0`，F201关闭；P302固定三锚点与 P303归一化正积分 warp
  分别研究非线性表达力/单调用校准；P302进一步达到 attained MAE=`.0256586`、violations=`0`，不重新打开 F201；
- 下一可用 failure id 为 `V67-F207`。

### V67-F202 — P303 warp积分错误复用 base knot count

- failed run=`run://worldsim_v67/WS-V67-P303-NORMALIZED-MONOTONE-WARP-AUTHORITY-COMPILER-01/
  20260901T011500Z__normalized-monotone-warp-authority-s0-r1`；
- symptom：独立 warp输出8 knots，但调用 base `_integral`时索引宽度使用 base rate knot count，首个 forward触发
  CUDA gather index out-of-bounds；0 completed training steps、0 quality read、无科学 verdict；
- research check：PyTorch gather官方合同要求所有 index落在 source dim范围内；UMNN开源实现也将积分器与自身
  integrand参数绑定。问题是局部张量维度所有权，不是 positive-integral假设失败；
- minimal recovery：r2用 warp rates自己的 `shape[-1]`计算 width/index/cumulative，模型、knots、hidden、steps、
  seed、teacher、gates全部不变；
- concurrent branch：P304只复用已支持 P302 的 base做单调用压缩训练，不依赖 F202失败路径；
- 下一可用 failure id 为 `V67-F206`。

### V67-F203 — P303 context warp未跨规模保持 attained-fraction fidelity

- canonical=`run://worldsim_v67/WS-V67-P303-NORMALIZED-MONOTONE-WARP-AUTHORITY-COMPILER-01/
  20260901T013000Z__normalized-monotone-warp-authority-s0-r2`；F202修复后合同原样完成；
- result：P201 budget/attained MAE=`.0066260/.0288235`、regret=`6.0888e-5`、violations=`0`、forward=`.7006s`；
  attained门及严格优于 P301门失败，2/4；size48/96=`.0269137/.0307334`；
- root cause：source attained=`.0200754`且训练 element loss很低，但 group-conditioned warp依赖 raw pooled moments，
  在 P201/size96 shift下 group mean calibration失配；这是 OOD group calibration问题，不是单调结构或预算拟合失败；
- literature response：NeurIPS 2024 multicalibration/OOD工作要求跨重叠groups显式校准；COLT 2022指出普通 DRO
  不保证 shift下统一低regret。因此不靠增加 knots/hidden补救，也不把 source改善外推为 P201 claim；
- disposition：关闭 normalized-warp结构扫描；保留其 `.7006s`单调用/零违例结果为 Pareto descriptive negative，
  独立 P304从 P302 anchor-trained base做单调用压缩，不复用 context warp；
- 下一可用 failure id 为 `V67-F206`。

### V67-F204 — P304 anchor-trained base单调用压缩未保持P201 fidelity

- canonical=`run://worldsim_v67/WS-V67-P304-ANCHOR-INITIALIZED-SINGLE-CALL-AUTHORITY-COMPILER-01/
  20260901T014500Z__anchor-initialized-single-call-s0-r1`；
- result：P201 attained MAE=`.0290092 > P301 .0285229`，但 regret=`6.5460e-5`、violations=0、forward=`.3070s`；
  3/4拒绝；source attained=`.0192548`，size96=`.0310086`；
- interpretation：NeurIPS 2021指出 student即使容量足够也常无法高保真匹配 teacher；本项目中移除 P302 piecewise
  decision rule后，anchor initialization保留了速度但没有保留跨场景 group calibration；
- disposition：不做 distillation weight/EMA/step sweep；与 F203共同关闭 single-call compression，选择 P302为
  fidelity/monotonicity端的 Pareto compiler；
- 下一可用 failure id 为 `V67-F206`。

### V67-F205 — P305 confirmation rows指向prep run

- failed run=`run://worldsim_v67/WS-V67-P305-FRESH-REUSE-PIECEWISE-AUTHORITY-CONFIRMATION-01/
  20260901T020000Z__fresh-reuse-piecewise-authority-confirmation-s0-r1`；
- symptom：`P277_FRESH_EPISTEMIC_ROWS.npz`实际位于 P277 confirmation run，配置误指 prep run；`np.load`前
  FileNotFound，0 training、0 quality read、无科学 verdict；
- minimal recovery：r2只替换 source/p201 artifact run path；模型、rows文件、conditions、sizes、teacher、gates不变；
- resolution：r2 3/3 supported，attained MAE=`.0167472`、regret=`-.0001084`、violations=0；
- next research：P306保持 P302三锚点规则，只共享重复 context computation；不依赖 P303/P304失败权重；
- 下一可用 failure id 为 `V67-F207`。

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
- P307后续：P308已冻结为P307 weights在P243九场景六动作组上的0-step family-untouched确认；若失败才使用
  `V67-F207`，不以refit、换cohort或放宽门恢复；
- 下一可用 failure id 为 `V67-F207`。

> **最后更新**：2026-08-29
> **唯一活跃失败事实源**：本文件 `docs/RESEARCH_FAILURES.md`
> **覆盖范围**：V1–V6.7、V7/V7.1、N1/cut-in 与跨路线工程/资源/协议教训
> **事实边界**：失败事实以 canonical run、`docs/EXPERIMENTS.md`、`docs/RESEARCH_STATUS.md` 和冻结证据为准

本文件是仓库中唯一持续维护的 failure ledger。`docs/archive/**/RESEARCH_FAILURES*.md` 只是对应 commit 的不可变
历史快照；`WS_*_FAILURE_FORENSICS.md` 是专项诊断报告，不是第二本账。V6.3/V6.4报告索引
`docs/autoresearch/worldsim_v63/ARXIV_EVIDENCE_INDEX.md`与`docs/autoresearch/worldsim_v64/ARXIV_EVIDENCE_INDEX.md`只导航本账与canonical evidence，也不是第二本失败账。新路线、新版本和新实验不得再创建并行的
`*_FAILURES.md` 事实源。

### V6.7 当前边界（2026-08-28）

- 新分支从V6.6 terminal `c05ca27`建立；V6.6 `V66-F02`不复开。
- P1/P2/P3已支持task-untouched legacy ranking、Actor package与固定action set；不等于fresh或physical repair。
- P4的source `behind_hit`交集让conflict reduction=`0.678963`通过，但overall/clean retention=
  `0.392368/0.396519`失败；登记`V67-F01 active`。
- P4R单次motion-compensated inward-ray结构恢复9/9 gates通过，`V67-F01 resolved_by_single_structural_recovery`；
  下一可用编号=`V67-F02`。
- 禁止radius/gate/budget sweep、target-dependent retention、Actor deletion与未通过physical repair前的RL claim。

<a id="detail-v67"></a>

## V6.7 Ray-Terminated Actor Surface 详细账本（2026-08-28）

### V67-F01 — aggregated source behind-hit与motion-compensated Actor hit失去ray/Actor对应

- 分类：`algorithm/representation`；状态：`resolved_by_single_structural_recovery`。
- 观察：P4在72 units / 517 Actor states / 258 acted states上把conflict points从`1,003`降至`322`
  （reduction=`0.678963`），但overall/clean retention仅`0.392368/0.396519 < 0.40`；7/9 gates通过仍拒绝。
- 根因：`behind_hit`由原始source-frame endpoint ray生成并跨帧聚合；Actor hit随后被motion-compensate到target Actor frame。
  在query端将二者相交无法恢复“哪个Actor hit、哪条ray”的对应，导致方向支持过稀。
- 推翻项：推翻“聚合behind-hit栅格可直接约束补偿后same-Actor邻域”的假设；不推翻P1 ranking、P2 package、P3 actions，
  也不改变Actor存在性与hazard保护。
- 防重复/复开：不降0.40 retention gate，不扫radius/threshold/action budget，不把全radius结果重报，不读target调rule。
- 外部检索迁移：ALSO与evidence-theory occupancy支持sensor termination前free/后unknown的分离；continuous occlusion与
  GPOcc支持ray-wise inward geometry。唯一P4R在target frame对nearest compensated same-Actor hit构造解析inward ray。
- 证据：`WS-V67-P4-RAY-TERMINATED-SURFACE-01/20260828T105253Z__ray-surface-s0-r1`；恢复冻结=
  `docs/autoresearch/worldsim_v67/P4R_MOTION_COMPENSATED_INWARD_RAY_FREEZE.md`。

P4R canonical=`WS-V67-P4R-MOTION-COMPENSATED-INWARD-RAY-01/20260828T105920Z__inward-ray-s0-r1`：
conflict reduction=`0.517448`、overall/clean retention=`0.529225/0.531941`，Actor/shell/identity-trajectory完全保持，
9/9 gates通过。该关闭只支持task-untouched legacy capability；P5-P8另用V65 P2六场景做独立surface confirmation。

P5独立legacy transfer 4/4 gates通过且无新failure；但head AUROC/AUPRC=`0.665176/0.676612`低于q0的
`0.695177/0.706467`。这不是P5预注册gate失败，作为negative comparator observation保留，并禁止声称learned head跨cohort
dominance。下一可用编号仍为`V67-F02`。

P6独立Actor package 6/6 gates通过，无Actor removal、hidden target或hazard-existence coupling；无新failure，下一编号仍为
`V67-F02`。

P7固定L0 actions 6/6 gates通过，pooled conflict reduction=`0.612179`；q0 pooled=`0.650641`更高但两场景低于0.5，
L0六场景均高于0.5。保持为negative comparator/robustness observation，不改P8 arm或gate；无新failure，下一编号仍为
`V67-F02`。

P8独立legacy physical confirmation以conflict reduction=`0.501469`、overall/clean=`0.553679/0.556700`通过9/9 gates；
无新failure。P4R/P8仍是globally consumed legacy，不能代替fresh population；P9已冻结新六场景输入链，下一编号仍为
`V67-F02`。

### V67-F02 — fresh native launcher入口合同未显式展开

- 分类：`engineering/entrypoint`；状态：`resolved_pre_quality_entry_contract`。
- 观察：scene-0348正式worker前依次遇到task parent不存在、V65-style `base_config`未由PyYAML自动展开、单卡配置误写
  device index `1`；随后旧失败run目录按exact-once拒绝覆盖。另一次evidence `--help`漏`PYTHONPATH=.`，未创建run。
- 根因：旧runner直接对单文件`yaml.safe_load`，并在创建run前查询parent disk；`resources.gpu`语义是device index。
- 恢复：创建task parent、显式展开冻结V6.3 native runtime字段、恢复GPU index 0、保留r1/r2失败目录并以r3完成0348。
- 数据边界：所有失败均在成功worker/quality read/scientific metric前；scene/cohort/model/targets未变，native inference未重复。
- 证据：prep/native/evidence canonical见`docs/autoresearch/worldsim_v67/P9_FRESH_INPUT_PIPELINE_RESULT.md`。
- 下一可用编号：`V67-F03`。

P10 fresh transfer 4/4 gates通过且无新failure；head与q0 AUROC近似相同、AUPRC低`0.046332`，作为negative
comparator保留，不改变冻结fresh chain。下一编号仍为`V67-F03`。

P11 fresh Actor package 6/6 gates通过；938 states与metadata完整保留、Actor removed/hidden target=`0/0`，无新failure；
下一编号仍为`V67-F03`。

### V67-F03 — 本地PowerShell提前解释远端动态run-id

- 分类：`execution/run-locator`；状态：`resolved_by_atomic_run_directory_rename`。
- 观察/根因：P12命令中的`$(date ...)`被PowerShell提前解释，成功evaluation写入literal backslash目录。
- 恢复：确认唯一completed目录和目标不存在后原子rename为`20260828T114900Z__fresh-actions-s0-r1`；未重跑evaluation、未改artifact。
- P12科学结果：L0在469/938固定budget处理341/563 conflicts，reduction=`0.605684`，6/6 gates通过。
- 防重复：后续使用显式run-id，不在PowerShell字符串内嵌远端命令替换；下一编号=`V67-F04`。

P13 fresh inward-ray physical confirmation 9/9 gates通过：conflict reduction=`0.529249`、overall/clean=
`0.554522/0.559808`，Actor contracts exact；无新failure，下一编号仍为`V67-F04`。后续直接进入P14 GPU训练，
不增加legacy确认或审计矩阵。

### V67-F04 — in-sample rescue threshold跨场景过度自信

- 分类：`algorithm/selective-risk`；状态：`active_single_recovery_frozen`。
- 观察：P14 train residual AUROC/AUPRC=`1/1`且train conflict rescue=0，但selection 0.5 threshold rescue
  `7,612 clean + 420 conflict`，使clean retention大升而conflict reduction从analytic `0.4924`降到`0.0939`；5/6 gates。
- 附带评估问题：selection包含654个不在action rows内的points，使analytic comparator低于P4R canonical；不是learned collapse根因。
- 检索/迁移：NeurIPS 2017 selective classification以abstention控制risk，ICLR 2024 Conformal Risk Control以heldout
  monotone loss校准，ICCV 2021 SENTRY用跨域consistency筛选。唯一P14R采用leave-one-training-scene-out conflict score
  的冻结1% quantile，并只评估exact action-eligible denominator。
- 防重复：不改architecture/loss/quantile/gates，不用selection拟合threshold，不做第二P14R；证据=
  `WS-V67-P14-DIRECTIONAL-SURFACE-TRAIN-01/20260828T120000Z__directional-surface-s0-r1`；下一编号=`V67-F05`。

P14R exact-once结果：LOSO threshold=`0.999919`且denominator修正后，analytic comparator恢复到P4R canonical；但learned
仍rescue`6,382 clean + 284 conflict`，conflict reduction=`0.234297`，5/6 gates。`V67-F04`状态更新为
`closed_negative_after_single_recovery`；禁止继续point threshold/ensemble/loss/model sweep。按预定后备路线更换prediction
object为Ego trajectory visited-state reliability；下一编号仍为`V67-F05`。

### V67-F05 — free trajectory residual学习source-scene action shortcut

- 分类：`algorithm/domain-generalization`；状态：`active_single_recovery_frozen`。
- 观察：P15 train Spearman/pairwise/selected reduction=`0.8633/1/0.5015`，selection仅=`0.2102/0.5866/0.1099`；
  unsafe AUROC=`0.6464`，显著低于qmean `0.9727`；0/6 gates。
- 根因：8-D free MLP residual可覆盖qmean base并编码source cohort的action/context shortcut；不是trajectory prediction
  object本身失败。
- 检索/恢复：CVPR 2026 ResAD用deterministic reference上的normalized residual抑制spurious correlation；UAI 2025
  constrained monotonic calibration强调保留base ranking。唯一P15R只训练12 action biases，case-centered，score residual
  bounded `±0.02`；qmean dominant，data/lattice/gates不变。
- 防重复：不扫residual bound/action lattice/loss/gates，不做第二P15R；下一编号=`V67-F06`。

P15R exact-once：bounded adapter使selection Spearman/AUROC/pairwise均略高于qmean，但selected reduction仅
`0.170481`（qmean=`0.163836`，delta=`+0.006645`），未达`0.25/+0.05`两门；4/6 gates，`V67-F05`更新为
`closed_negative_after_single_recovery`。禁止继续在P10X调adapter。P16把P10V/P10X降格为two-domain development，
模型冻结后才做P9 fresh action-task confirmation；下一编号仍为`V67-F06`。

### V67-F06 — P16 domain variance变量未在adapter作用域绑定

- 分类：`implementation/training-entry`；状态：`resolved_pre_confirmation_entry`。
- 观察/根因：r1在首个loss构造时报`NameError: domain_variance`；domain-balanced patch只在free head中构造变量，却在
  bounded adapter loss引用。model freeze与P9 action target materialization均未发生，无scientific metric。
- 恢复：在adapter内按`domain_index`计算equal-domain Huber mean/variance；r2不改数据、模型、loss权重或gates。
- 防重复：保留r1失败目录，以r2继续；下一编号=`V67-F07`。

P16 r2按原合同完成并在模型冻结后首次materialize P9 action targets；入口问题已关闭。r2 learned adapter相对qmean的
Spearman/pairwise/selected reduction delta=`-0.007213/-0.044718/-0.001150`，是独立科学负结果而非F06延续。

### V67-F07 — 学习替代trajectory qmean持续破坏跨域决策排序

- 分类：`algorithm/representation-and-authority`；状态：`closed_negative_after_independent_confirmation`。
- 观察：P16 two-domain bounded action-ID adapter在P9为3/6 gates；P17 monotone quantile pool为1/6 gates。P17 learned/qmean
  Spearman=`0.645502/0.658731`、pairwise=`0.749190/0.779650`、selected reduction=`0.387839/0.418184`。
- 根因：trajectory corridor内qmean已是强、低容量且可迁移的充分排序统计；action identity和自由分布混合分别引入
  cohort shortcut与低分位偏置。P17学到distribution mix=`0.497368`，没有产生独立可迁移增益。
- 推翻项：推翻“在现有三cohort上学习替代qmean scorer可提升authority”的假设；不推翻given-trajectory visited-state
  reliability，也不推翻P9 qmean本身的0.4182 selected-cost reduction。
- 外部检索/迁移：NeurIPS 2023 PlanCP与NeurIPS 2025 conformal risk training把uncertainty用于规划risk control而非任意
  改写base score；CVPR 2024 online-map uncertainty也把UQ送入downstream task。P18因此冻结qmean ranking，只训练
  case-level benefit/abstention compiler。
- 防重复：不扫quantile levels、distribution mix、action ID、score residual、gate或selection fraction；P9已消费，只作
  P18 method selection。若P18成立，再到独立cohort一次确认；下一编号=`V67-F08`。

P18 fixed-qmean selective compiler在P9以固定49.30% coverage把selected-cost reduction从`0.418184`提高到`0.487876`
（delta=`+0.069693`），6/6 scenes不退化、4/4 gates通过。该结果使F07进入independent confirmation，尚不因consumed
selection单独关闭；P19保持compiler/coverage/gates冻结，不在P9继续试验。

P19 independent结果保留relative gain：qmean `0.295088→0.350899`（`+0.055811`）、authorized positive rate=
`0.805556`且6/6 scenes不退化；但冻结absolute reduction `0.45`门失败，故3/4正式拒绝并关闭F07，不降门、不改coverage、
不做第二P19。下一方法P20改为四域listwise decision-focused action ordering，不是selective-gate recovery。

### V67-F08 — P20 listwise action compiler候选

- 分类：`algorithm/decision-focused-ranking`；状态：`resolved_by_independent_listwise_confirmation`。
- 动机：F07显示case authority可稳定提高relative benefit，但冻结qmean ordering的cohort ceiling使P19绝对门失败。
- 检索/迁移：ICML 2022将decision-focused learning表述为learning-to-rank；NeurIPS 2019/2021提供可微排序/top-k代理。
  P20以四开发域、case-centered `±0.02` residual和soft top-k target cost直接训练action set。
- 防重复：不扫residual bound、architecture、temperature、loss weight、selected fraction或gate；V67 P1 action target在模型
  freeze后一次读取。下一编号=`V67-F09`。

P20 exact-once在P1 action-task confirmation通过4/4 gates：selected reduction `0.460084`，相对qmean `+0.030723`；
pairwise `0.826230`，5/5 eligible scenes不退化。scene-1046的12 units均低于16-point footprint，保留为coverage边界，
不删scene或降minimum footprint。F08关闭；P21只在冻结P20排序上训练selective authority。

P21在P2V action-task confirmation同样4/4 gates：固定49.30% coverage下，P20/qmean reduction=
`0.404135/0.345130`，selective authority=`0.450102`；35/35 authorized cases nonnegative，5/5 covered scenes不退化。
无新failure。下一候选P22从mean benefit推进到unsafe tail proxy；下一编号仍为`V67-F09`。

### V67-F09 — P22 tail-risk-aware listwise compiler候选

- 分类：`algorithm/risk-sensitive-ranking`；状态：`closed_negative_after_first_trial`。
- 动机：P20/P21已支持mean visited-state cost selection，但`any hidden-FREE` unsafe event仍只作AUROC描述，尚未进入
  differentiable selected-set objective。
- 检索/迁移：ICML 2024 risk-sensitive reward-free RL和NeurIPS 2021 distributional CVaR强调tail outcome；P22新增soft
  selected unsafe-rate损失，但明确不把binary proxy包装成CVaR或safety guarantee。
- 防重复：unsafe weight=`0.25`、六开发域、architecture/residual/temperature/fraction/gates预先固定；V64 P10R4 action
  target在模型freeze后一次读取，不扫weight或tail definition。下一编号=`V67-F10`。

P22 exact-once仅1/4 gates：unsafe reduction比P20增加`+0.004719`，不足冻结`+0.02`，且mean reduction
`0.329362 <0.35`并比P20低`0.003501`；8/8 scene support通过。禁止扫unsafe weight/threshold；binary any-event family关闭。

### V67-F10 — P23 continuous entropic selected-cost候选

- 分类：`algorithm/continuous-risk-sensitive-ranking`；状态：`closed_negative_after_first_trial`。
- 动机/检索：NeurIPS 2022 Efficient Risk-Averse RL指出离散tail会形成tail barrier；NeurIPS 2020 OCE risk learning覆盖
  entropic/CVaR等连续风险。P23用连续target cost的entropic soft selected risk绕开binary plateau。
- 合同：risk aversion=`10`、weight=`0.25`、七开发域；确认用V64 P10R2八场景；与冻结P20/P22/qmean同分母比较。
- 防重复：不扫risk aversion/weight/tail fraction/model/temperature/gate；不把objective称为OCE/CVaR保证。下一编号=`V67-F11`。

P23在mean reduction/pairwise/scene support三门通过，但top-10% tail mean与P20几乎相同（ratio=`0.999450 >0.95`）；
3/4 gates正式拒绝。连续entropic objective提升mean `+0.013006`，但未产生tail独立增益；不扫risk aversion/weight，
tail auxiliary研究关闭。

### V67-F11 — P24 adaptive fixed-total action budget候选

- 分类：`algorithm/task-conditioned-budget-allocation`；状态：`resolved_by_fixed_total_budget_confirmation`。
- 动机：P20/P21已经证明ranking与selective authority；P24研究在总action数完全一致时，能否按case难度分配1--5个actions。
- 方法：P20 within-case order冻结；八域16-hidden bounded `±0.05` case calibration只改变跨case slot priority。
- 防重复：不扫max actions、offset bound、architecture、fraction或gate；P6R action target在模型freeze后一次读取。下一编号=
  `V67-F12`。

### V67-F12 — P24 evaluator未与offset dataset的single-action exclusion对齐

- 分类：`implementation/evaluator-indexing`；状态：`resolved_pre_metric_evaluator_alignment`。
- 观察/根因：r1训练和cache完成后，offset dataset按既有selection合同跳过`<2` eligible actions cases，但adaptive evaluator
  仍遍历全部unique cases，row 78访问size 78 offset数组外索引；Python/NumPy官方文档定义该情况为`IndexError`。
- 科学暴露：model已在confirmation target materialization前冻结；失败发生在任何metric/gate计算前，无scientific result。
- 恢复：evaluator改用与`_within_case_selection`一致的`>=2 actions` cases/all-action denominator；r2复用r1 frozen artifact
  和cache，不重训、不改合同。下一编号=`V67-F13`。

P24 r2 4/4 gates通过：exact budget=`222/222`，adaptive reduction=`0.758380`，相对fixed P20=`+0.161610`，7/7
evaluable scenes不退化。F12工程入口关闭，F11科学候选关闭为positive。r2只读r1 artifact/cache，未重复训练。

### V67-F13 — P25 coverage-constrained fixed-total budget候选

- 分类：`algorithm/selective-budget-authority`；状态：`resolved_by_coverage_constrained_confirmation`。
- 动机：P24要求每case至少1 action；P25在总budget相同下允许部分case abstain，同时冻结至少50% case coverage。
- 方法：九域bounded offset；P20 within-case order不变；每case0--6 actions；总action数等于fixed quarter baseline。
- 防重复：不扫coverage/max actions/offset/model/fraction/gate；P4C action target在model freeze后一次读取。下一编号=
  `V67-F14`。

P25 5/5 gates通过：exact budget=`243/243`、case coverage=`0.606742`、reduction=`0.694998`，相对fixed P20 /
P24=`+0.382792/+0.100552`，8/8 scenes不退化。该结果说明增益来自在固定总预算下联合abstention与跨case分配，
不是减少动作总数；不升级为collision/planning/safety claim。

### V67-F14 — P26 large-cohort coverage transfer候选

- 分类：`algorithm/large-cohort-transfer`；状态：`resolved_by_large_cohort_transfer`。
- 动机：P25在八场景成立后，将其cohort滚入development并重新训练；在从未用于V6.7 allocation的P6E 16场景/
  192-case分层cohort检验迁移规模与场景支持。
- 方法：十域bounded offset；冻结P20 within-case order；exact fixed-quarter total budget、minimum 50% case coverage、
  0--6 actions/case；架构和训练超参原样继承P25。
- 防重复：不扫coverage/max actions/offset/model/fraction/gate；P6E target只读一次。下一编号=`V67-F15`。

P26 5/5 gates通过：exact budget=`511/511`、case coverage=`0.644444`、reduction=`0.792541`，相对fixed P20 /
P24=`+0.391952/+0.108614`，15/15 evaluable scenes不退化。16个输入scenes中scene 0按冻结footprint无evaluable
case，故场景支持分母为15，不是事后排除。

### V67-F15 — P27 stratum-balanced authority候选

- 分类：`algorithm/context-balanced-authority`；状态：`resolved_by_stratum_balanced_confirmation`。
- 动机：P25/P26证明全局coverage有效；P27防止authority集中到容易context，在四个既定strata内各自保证至少50% cases。
- 方法：十一域重新训练同一bounded offset；P20 order、exact total budget、global 50% coverage与0--6 actions不变；
  P6R只作已消费legacy confirmation。
- 防重复：不扫group、coverage、model或gate；若失败关闭group-floor候选。下一编号=`V67-F16`。

P27 6/6 gates通过：budget=`222/222`，global coverage=`0.628205`，minimum stratum coverage=`0.50`；reduction=
`0.800447`，相对fixed P20/P24=`+0.203678/+0.042068`，6/6 evaluable scenes不退化。该positive只在已消费P6R上
支持context-balanced mechanism，不恢复fresh confirmation。

### V67-F16 — P28 budget-conditioned unseen-fraction候选

- 分类：`algorithm/budget-conditioned-authority`；状态：`resolved_by_unseen_budget_transfer`。
- 动机：P20--P27均固定25% budget；P28把requested budget作为显式条件，检验一个模型能否插值到未训练fraction。
- 方法：P10R4从P28训练域移除；其余十域在0.25/0.50联合训练，confirmation fraction固定1/3；exact total与四strata
  coverage约束保持。
- 防重复：不扫fraction、architecture、offset、coverage、group或gate；P10R4是consumed legacy，不作fresh claim。下一编号=
  `V67-F17`。

P28 6/6 gates通过：1/3 exact budget=`363/363`，global/minimum stratum coverage=`0.708333/0.583333`；reduction=
`0.674930`，相对fixed P20=`+0.393479`，8/8 scenes不退化。支持heldout-budget interpolation mechanism；因cohort
全局已消费，不称fresh generalization。

### V67-F17 — P29 nested low/high authority候选

- 分类：`algorithm/budget-path-consistency`；状态：`resolved_by_nested_budget_confirmation`。
- 动机：budget-conditioned priorities可能随requested fraction变化；runtime预算变化需要low-budget authority严格嵌套于high。
- 方法：P4C从P29训练移除；十域0.25/0.50训练；confirmation同时输出exact 25%/50%，high集合以low为mandatory
  prefix再按high-budget priority扩展；两预算各自保持四group coverage。
- 防重复：不扫fractions、nesting rule、group、model或gate；失败即关闭nested candidate。下一编号=`V67-F18`。

P29 7/7 gates通过：low/high exact=`243/243,494/494`、nested count=`243`；low/high reduction=
`0.758868/0.387925`，相对各自fixed P20=`+0.446663/+0.182809`；两预算8/8 scenes不退化。支持budget-path
consistency mechanism，但P4C全局已消费，不称fresh。

### V67-F18 — P30 horizon-conditioned H插值候选

- 分类：`algorithm/horizon-conditioned-authority`；状态：`resolved_by_heldout_horizon_transfer`。
- 动机：此前visited-state对象均固定H=2s；P30显式条件化H，响应“给定tau，未来H秒被访问states是否可靠”。
- 方法：P10V H=1s新cache与H=2s development联合训练；P10X从训练移除，confirmation H=1.5s；25% exact budget、
  三context groups各50% coverage、P20 order冻结。
- 输入结果：H=1s materialization 72 cases、733/864 eligible actions，无失败。
- 防重复：不扫H、footprint、model、offset、coverage、group或gate；P10X globally consumed，不作fresh claim。下一编号=
  `V67-F19`。

P30 6/6 gates通过：H=1.5s source/eligible=`864/717`，66 cases，budget=`176/176`，global/minimum group coverage=
`0.636364/0.50`；reduction=`0.740743`，相对fixed P20=`+0.505189`。Scene support 5/6，唯一正delta=
`1.86e-9`是浮点均值差，仍按预注册`<=0`规则计higher；minimum 5门通过，不做epsilon改写。

### V67-F19 — P31 joint budget-H unseen-condition候选

- 分类：`algorithm/joint-task-conditioned-authority`；状态：`resolved_by_joint_condition_transfer`。
- 动机：P28和P30分别证明budget/H插值；P31检验同一模型对未见联合条件`(1/3,1.5s)`的组合泛化。
- 方法：四domains，H=`1/2s`、budget=`.25/.50`联合case rows；P10X训练排除；确认cache复用P30 H=1.5s，
  budget改为1/3，三context groups coverage保底。
- 防重复：不扫条件点、model、offset、coverage、group或gate；失败不回退到单轴结果。下一编号=`V67-F20`。

P31 6/6 gates通过：unseen pair `(1/3,1.5s)` budget=`236/236`，global/minimum group coverage=
`0.727273/0.583333`，reduction=`0.690636`，相对fixed=`+0.499918`，5/6 scenes不退化。

### V67-F20 — P32 joint-condition nested budget候选

- 分类：`algorithm/joint-condition-budget-path-consistency`；状态：`resolved_by_joint_nested_confirmation`。
- 方法/结果：H=1.5s上low/high exact=`176/176,352/352`，176 low actions全部保留；low/high reduction=
  `0.811047/0.404128`，相对fixed=`+0.575493/+0.263991`；minimum group=`0.50/0.791667`，7/7 gates。
- 边界：P10X globally consumed，只支持联合条件下nested mechanism；无safety claim。下一编号=`V67-F21`。

### V67-F21 — P33 second-cohort joint condition transfer候选

- 分类：`algorithm/second-cohort-joint-condition-transfer`；状态：`resolved_by_second_cohort_joint_transfer`。
- 方法：同一四development domains、budget/H conditions、architecture/loss/gates；P4C不进训练，首次物化H=1.5s并在1/3
  budget读取一次；四context groups各coverage>=0.50。
- 防重复：不扫条件/model/gate，不以P10X结果调整P4C门；globally consumed但joint-H task未读。下一编号=`V67-F22`。

P33 6/6 gates通过：P4C H=1.5s `973/1152` eligible、89 cases，budget=`315/315`，global/minimum group=
`0.696629/0.50`；reduction=`0.698243`、相对fixed=`+0.439588`，8/8 scenes不退化。

### V67-F22 — P34 bounded heteroscedastic authority候选

- 分类：`algorithm/aleatoric-case-uncertainty`；状态：`closed_negative_after_first_trial`。
- 调研：NeurIPS 2023支持异方差Gaussian regression稳定化；NeurIPS 2024指出单网络evidential epistemic可能不可靠。
- 方法：P31同一joint-condition features上训练bounded mean与`0.005..0.10` scale；P10X consumed selection固定
  conservative offset=`mean+1.0*scale`，与冻结P31 mean compiler比较。
- 结果：scale-error Spearman=`0.190272`通过`>=0.15`，但conservative/mean reduction=
  `0.610037/0.690636`，delta=`-0.080599`，未达`+0.01`；其余exact total、minimum group与scene support通过。
- 结论：scale与误差相关不等于其保守加法能改善有限预算决策；不以弱UQ诊断替代selection utility。
- 边界：仅aleatoric error scale；不声称epistemic/calibrated interval/OOD；P22/P23 action-tail family保持关闭。
- 防重复：不扫scale bound、sigma weight、loss、model或gate；aleatoric priority family关闭。下一编号=`V67-F23`。

### V67-F23 — P35 fixed deep-ensemble authority候选

- 分类：`algorithm/model-disagreement-authority`；状态：`closed_negative_after_first_trial`。
- 调研迁移：Deep Ensembles（NeurIPS 2017）以独立初始化成员分歧作为实用uncertainty基线；结合NeurIPS 2024对
  evidential epistemic可靠性的批评，P35显式使用三个独立模型，而不从单头scale外推epistemic结论。
- 方法：seed `0/1/2`顺序训练同一P31 joint-condition head；P4C consumed selection固定priority=
  `ensemble_mean+1.0*ensemble_std`，与冻结P33 mean compiler在相同exact budget/group constraints下比较。
- 结果：disagreement-error Spearman=`0.144178`通过`.10`，但mean disagreement仅`3.53e-6`，三个成员的
  residual RMS均饱和到`0.05`；conservative与P33选择完全相同，reduction=`0.698243`，delta=`0.0`。
- 结论：相同小模型/完整批次在强边界解上发生成员塌缩；有弱误差相关仍没有可消费的排序变化。
- 边界：disagreement仅为epistemic proxy；不声称posterior calibration、OOD、collision/planning/closed-loop/safety。
- 防重复：不扫成员数、seed、uncertainty weight、model、loss或gate；uncertainty-for-decision路线关闭。
  下一编号=`V67-F24`。

### V67-F24 — P36 conditioned differentiable top-k action compiler候选

- 分类：`algorithm/conditioned-decision-focused-action-ranking`；状态：`resolved_by_direct_conditioned_topk`。
- 调研：ICLR 2019 NeuralSort与ICML 2020 SoftSort将离散排序替换为可微连续松弛，使最终top-k目标能反向训练。
- 方法迁移：在P20 action features上增加budget与H两个条件；用`.25/.50` budgets、`H=1/2s`四domains训练，
  主损失为soft top-k权重下的真实visited-state cost，并保留小权重pairwise/regression稳定项。
- 判定：P4C consumed `(1/3,1.5s)`直接输出action scores，经同一exact-total/group constraints选择；相对冻结P33
  reduction至少`+0.005`且6 scenes不退化。
- 结果：reduction=`0.719901`，相对P33=`+0.021658`、相对fixed=`+0.461246`；exact `315/315`、minimum
  group=`0.50`、8/8 scenes、4/4 gates。
- 防重复：固定temperature/residual bound/model/loss/gates；不扫参。下一编号=`V67-F25`。

### V67-F25 — P37 frozen conditioned-action second-cohort transfer候选

- 分类：`algorithm/cross-cohort-conditioned-action-transfer`；状态：`closed_negative_after_frozen_read`。
- 方法：P36 model/normalizer完全冻结，P10X consumed `(1/3,1.5s)`单次读取；与冻结P31在相同exact/group约束比较。
- 判定：reduction delta `>=+0.005`、minimum group `.50`、至少5 scenes不退化；无训练/refit/sweep。
- 结果：coverage/minimum group提升到`0.787879/0.666667`，但reduction=`0.656886`低于P31 `0.690636`，
  delta=`-0.033750`；exact与scene gates通过，decision gain失败。
- 结论：P36的P4C增益不是跨cohort稳定增益；增加coverage不能替代selected-cost质量。
- 边界：第二consumed cohort的method transfer，不是fresh population confirmation。下一编号=`V67-F26`。

### V67-F26 — P38 smooth worst-domain conditioned top-k候选

- 分类：`algorithm/worst-domain-decision-focused-training`；状态：`closed_negative_after_first_trial`。
- 调研：ICLR 2020 GroupDRO关注最坏group风险；ICML 2021 REx通过训练域风险关系改善OOD generalization。
- 方法迁移：P36仅将domain mean+variance聚合换为temperature `.02` log-sum-exp smooth maximum；其余数据、
  architecture、soft top-k、pairwise/regression、residual bound与epochs完全不变。
- 判定：P10X consumed `(1/3,1.5s)`相对冻结P31 reduction `>=+0.005`、minimum group `.50`、scene support `5`。
- 结果：coverage/minimum group=`0.803030/0.708333`，但reduction=`0.629974`，比P31低`-0.060662`；
  worst-domain objective比P36 transfer再退`-0.026913`。
- 结论：平滑最坏域目标鼓励更广coverage，但没有提高跨cohort action ordering；objective-reweighting路线关闭。
- 防重复：不扫temperature/aggregation/model/loss/gate。下一编号=`V67-F27`。

### V67-F27 — P39 expanded-domain conditioned top-k候选

- 分类：`algorithm/training-domain-diversity`；状态：`resolved_by_expanded_domain_training`。
- 方法：恢复P36 mean+variance objective；加入已消费P4C/P10X H=1.5作为两个development domains，训练budgets=
  `.25/1/3/.50`，共6 domains；P3C action targets从训练排除。
- 判定：P3C H=2/budget=1/3相对冻结P31 reduction `>=+0.005`、minimum group `.50`、至少4/5 scenes不退化。
- 结果：exact=`238/238`、coverage/min group=`0.766667/0.708333`；reduction=`0.724052`，相对P31=
  `+0.013217`、相对fixed=`+0.355925`，5/5 scenes，4/4 gates。
- 防重复：只改训练domain/budget denominator，不改模型、loss、temperature或gate；不扫cohort组合。下一编号=`V67-F28`。

### V67-F28 — P40 frozen expanded-domain fourth-cohort transfer候选

- 分类：`algorithm/fourth-cohort-transfer`；状态：`closed_negative_after_frozen_read`。
- 方法：冻结P39 artifact；P10R4 action targets未进入P20/P31/P39训练，H=2/budget=1/3一次读取；四scene-pair groups。
- 判定：相对冻结P31 reduction `>=+0.005`、minimum group `.50`、至少6/8 scenes不退化。
- 结果：coverage/minimum group=`0.760417/0.583333`、8/8 scenes，但reduction=`0.654575`，比P31低
  `-0.020355`；更广coverage再次未转为更低cost。
- 边界：cohort全局已消费，故不是fresh confirmation；无训练/refit/sweep。下一编号=`V67-F29`。

### V67-F29 — P41 terminal eight-domain conditioned top-k候选

- 分类：`algorithm/terminal-domain-expansion`；状态：`closed_negative_terminal`。
- 调研：DomainBed报告在严格统一条件下，carefully implemented ERM可达到强domain-generalization表现；因此不再追加
  新DG正则，而测试有限数据域扩展的终点。
- 方法：P39加入已消费P3C/P10R4为development，形成8 domains×3 budgets；P10R2 action targets完全排除。
- 判定：P10R2 H=2/budget=1/3相对P31 `>=+0.005`、minimum group `.50`、至少6/8 scenes。
- 结果：P41/P31 reduction=`0.747149/0.743093`，delta=`+0.004055`，仅比冻结门低`0.000945`；exact、
  group、8/8 scenes通过，但AND verdict仍拒绝。
- stop：不降门、不扫cohort组合/model/loss/temperature；action-scorer跨cohort family关闭。下一编号=`V67-F30`。

### V67-F30 — P42 frozen allocator + trained action refinement hybrid候选

- 分类：`algorithm/compositional-case-action-authority`；状态：`resolved_by_hybrid_composition`。
- 调研迁移：ICLR 2020 blackbox combinatorial differentiation强调学习模型与既有组合优化结构的端到端组合；项目内不引入
  外部solver，而保留已支持的P31 allocator并学习P20之上的case-centered action residual。
- 训练：9 consumed domains×3 budgets；base score=P20，head结构/soft top-k同P41；P31 case offset全程冻结。
- 判定：P6R action-target-untouched H2/budget1/3相对P31 `>=+0.005`、minimum group `.50`、至少5/7 scenes。
- 结果：exact=`294/294`、coverage/min group=`0.705128/0.50`；reduction=`0.800132`，相对P31=
  `+0.007912`、相对fixed=`+0.250012`，7/7 scenes，4/4 gates。
- 防重复：融合权重固定1:1加法，不扫模型/loss/temperature/gate。下一编号=`V67-F31`。

### V67-F31 — P43 frozen hybrid nested-budget结构候选

- 分类：`algorithm/hybrid-nested-budget-structure`；状态：`closed_negative_low_budget_regression`。
- 方法：P42/P31/P20全冻结；P6R quarter/half两预算分别条件化score，low集合强制作为high前缀扩展。
- 判定：两端exact、low subset high、minimum group `.50`、相对对应P31 nested baseline不退化、两端至少5 scenes。
- 结果：exact与222-action strict nesting通过；hybrid low/high reduction=`0.808732/0.641285`，P31=
  `0.833218/0.638464`，deltas=`-0.024486/+0.002821`；低预算非退化失败。
- 结论：P42 gain是budget-dependent；不以high gain覆盖low regression。
- 边界：同一consumed P6R结构读取，不新增泛化claim；不扫budget/weight/gate。下一编号=`V67-F32`。

### V67-F32 — P44 low-budget anchored hybrid候选

- 分类：`algorithm/budget-anchored-hybrid`；状态：`resolved_by_low_budget_anchor`。
- 调研迁移：Nested Dropout学习有序子结构；Once-for-All用progressive shrinking服务多resource constraints。P44不迁移
  其网络规模机制，而迁移“低资源子系统应是高资源系统前缀/锚点”的结构原则。
- 方法：P42 residual乘固定amplitude=`clip((budget-.25)/(.50-.25),0,1)`；quarter严格回退P20，half完整hybrid。
- 数据：9 consumed domains训练；P6R H1.5 task target新物化，与训练并行；budget1/3一次判定。
- 启动记录：首次materializer CLI缺`--run-id`在argparse退出，0 cache写入/target read；补同一run-id后继续，非科学失败。
- 结果：H1.5 eligible=`881/1152`、76 cases；exact=`292/292`、coverage/min group=`0.671053/0.50`；
  reduction=`0.809547`，相对P31=`+0.020361`、相对fixed=`+0.306632`，6/6 scenes，4/4 gates。
- 防重复：不扫anchor/full fraction/amplitude/model/loss/gate。下一编号=`V67-F33`。

### V67-F33 — P45 frozen anchored-hybrid nested-budget候选

- 分类：`algorithm/anchored-hybrid-nested-structure`；状态：`resolved_by_anchored_nested_nonregression`。
- 方法：P44/P31/P20全冻结；新H1.5 cache quarter/half strict nesting。Quarter action score由anchor严格等于P20。
- 判定：两端exact、strict nesting、minimum group `.50`、相对各自P31 nested baseline非退化、至少5 scenes。
- 结果：low/high exact=`220/436`且strict nested；deltas over P31=`0/+0.005463`，minimum group=
  `.50/.625`，scene=`5/7,7/7`，5/5 gates。
- 边界：同一新H task condition的结构read；无训练/refit/budget/weight sweep。下一编号=`V67-F34`。

### V67-F34 — P46 anchored hybrid cross-cohort new-H replication候选

- 分类：`algorithm/cross-cohort-task-condition-replication`；状态：`resolved_by_cross_cohort_horizon_replication`。
- 方法：P44结构/anchor/loss不变；新增P6R-H1.5 development domain；P10R4-H1.5 target首次物化并作budget1/3 read。
- 判定：相对P31 `>=+0.005`、exact、minimum group `.50`、至少6/8 scenes。
- 结果：H1.5 eligible=`1077/1152`；exact=`353/353`、coverage/min group=`.666667/.583333`；
  reduction=`.683908`，相对P31=`+.011489`、相对fixed=`+.431285`，8/8 scenes，4/4 gates。
- 边界：source cohort曾在H2训练，但H1.5 task target未见；不作fresh population claim，不扫参。下一编号=`V67-F35`。

### V67-F35 — P47 frozen cross-cohort anchored nested replication候选

- 分类：`algorithm/cross-cohort-nested-replication`；状态：`closed_negative_high_budget_regression`。
- 方法：P46/P31/P20全冻结；P10R4-H1.5 quarter/half strict nesting，quarter anchor回退P20。
- 判定：两端exact、strict nesting、minimum group `.50`、相对各自P31非退化、至少6 scenes。
- 结果：low delta=`0`，high delta=`-0.032572`；exact/nesting/group/8 scenes通过，高预算非退化失败。
- 结论：low anchor可迁移，但full-residual high endpoint不能跨cohort稳定；单侧anchor high family关闭。
- 边界：第二cohort新H结构复制，无训练/refit/sweep。下一编号=`V67-F36`。

### V67-F36 — P48 double-anchored interior-budget adapter候选

- 分类：`algorithm/endpoint-preserving-residual-adapter`；状态：`closed_negative_below_frozen_gain_gate`。
- 调研迁移：Residual Adapters以共享主干+小域适配器服务多domain；Net2Net强调function-preserving变换。P48保持两端
  冻结函数不变，只在budget内部激活adapter。
- 方法：amplitude在`.25/.50`为0，在`1/3`为1，分段线性；11 domains×3 budgets训练；P31 allocator冻结。
- 数据：新增P10R4-H1.5 development；P10R2-H1.5 target首次物化并作peak read。
- 结果：P48/P31 reduction=`.742759/.740902`，delta=`+.001857`，exact/group/8 scenes通过但未达冻结`+.005`。
- 结论：双端function-preserving合同成立，内部adapter增益不足；不降门、不扫两端/peak/amplitude/model/loss/gate。
  下一编号=`V67-F37`。

### V67-F37 — P49 Fishr-inspired domain-gradient consistency候选

- 分类：`algorithm/domain-gradient-consistency`；状态：`resolved_by_new_horizon_task_condition`。
- 调研迁移：Fishr（ICML 2022）以跨域梯度统计一致性提升domain generalization；P49只迁移到小adapter末层的
  normalized domain update direction dispersion，不声称实现完整per-sample gradient-variance Fishr。
- 方法：P48双端anchor/model/P31 allocator不变；加入固定`.01`末层方向离散惩罚，12 domains×3 budgets训练。
- 数据：新增已消费P10R2-H1.5 development；P3C-H1.5首次物化=`710/864` eligible并作一次budget1/3 read。
- 结果：exact=`236/236`、group=`.666667`；P49/P31 reduction=`.710322/.695815`，delta=`+.014506`；5/5
  scenes，4/4 gates。固定轻量gradient一致性训练得到正式支持。
- 防重复：不扫gradient weight、layer、anchor、peak、model、loss或gate。下一编号=`V67-F38`。

### V67-F38 — P50 frozen second task-condition replication候选

- 分类：`confirmation/frozen-cross-condition-transfer`；状态：`resolved_by_second_new_horizon_condition`。
- 方法：P49/P31/P20冻结；P2V-H1.5首次物化=`774/864` eligible、72 cases，budget1/3一次读取。
- 判定：exact、minimum group `.50`、相对P31 `+.005`、至少5 scenes；无训练/refit/weight/anchor/gate sweep。
- 结果：exact=`252/252`、group=`.541667`；P49/P31 reduction=`.789696/.739907`，delta=`+.049789`；
  6/6 scenes、4/4 gates。冻结方法跨第二新H条件复制。
- 边界：source cohort已消费且H2进入development，但H1.5 target未用于P49训练；不作fresh population claim。
  下一编号=`V67-F39`。

### V67-F39 — P51 large-cohort new-horizon gradient replication候选

- 分类：`algorithm/large-cohort-gradient-consistency`；状态：`resolved_by_large_cohort_new_horizon`。
- 方法：P49 gradient penalty/anchor/model/loss/gates不变，只将已消费P2V-H1.5滚入第13个development domain。
- 数据：P6E 16-scene H1.5 target首次物化=`2049/2304` eligible、192 cases，与GPU训练并行；budget1/3一次read。
- 判定：exact、四scene-group minimum coverage `.50`、相对P31 `+.005`、至少12 scenes。
- 结果：exact=`673/673`、group=`.50`；P51/P31 reduction=`.806000/.796088`，delta=`+.009912`；
  15/15 evaluable scenes，4/4 gates。
- 防重复：不扫gradient weight/anchor/peak/model/loss/group/gate。下一编号=`V67-F40`。

### V67-F40 — P52 frozen below-range horizon extrapolation候选

- 分类：`confirmation/frozen-horizon-extrapolation`；状态：`resolved_by_short_horizon_extrapolation`。
- 方法：P51/P31/P20冻结；P10V-H0.8首次物化=`694/864` eligible，低于所有训练H。
- 结果：exact=`229/229`、group=`.521739`；P51/P31 reduction=`.761914/.680754`，delta=`+.081161`；
  6/6 scenes，4/4 gates。无训练/refit/sweep。
- 边界：source cohort已消费；只称task-condition extrapolation。下一编号=`V67-F41`。

### V67-F41 — P53 jointly unseen budget + horizon gradient hybrid候选

- 分类：`algorithm/joint-budget-horizon-generalization`；状态：`resolved_by_joint_unseen_condition`。
- 方法：P51 gradient/anchor/model/loss不变；training budgets加入固定`.40`，形成14 domains×4 budgets。
- 数据：P10X-H0.8首次物化=`662/864` eligible、72 cases，与GPU训练重叠；formal budget=`.375`未见。
- 判定：exact、minimum group `.50`、相对P31 `+.005`、至少5 scenes。
- 结果：exact=`218/218`、group=`.541667`；P53/P31 reduction=`.733916/.724912`，delta=`+.009004`；
  6/6 scenes，4/4 gates。
- 防重复：不扫budget集合、gradient weight、anchor、model、loss或gate。下一编号=`V67-F42`。

### V67-F42 — P54 frozen second-cohort joint-condition replication候选

- 分类：`confirmation/frozen-joint-condition-transfer`；状态：`resolved_by_second_joint_cohort`。
- 方法：P53/P31/P20冻结；P4C-H0.8=`861/1152` eligible；formal budget `.375`，无训练/refit。
- 结果：exact=`282/282`、group=`.50`；P53/P31 reduction=`.830087/.775806`，delta=`+.054281`；
  8/8 scenes，4/4 gates。联合条件跨第二cohort复制。
- 边界：globally consumed source cohort；不作fresh population claim。下一编号=`V67-F43`。

### V67-F43 — P55 fixed-tail weight-averaged gradient hybrid候选

- 分类：`algorithm/flat-minimum-weight-averaging`；状态：`closed_negative_after_single_trial`。
- 调研迁移：SWAD（NeurIPS 2021）以flat minima缩小domain generalization gap；SWA（UAI 2018）沿训练轨迹平均权重。
- 方法：P53完全不变，只固定平均最后20%=1,200 checkpoints；不用validation选窗口，不改学习率或训练长度。
- 数据：P10R4-H0.8=`984/1152` eligible、96 cases；同一formal read加载冻结P53作method baseline。
- 判定：exact/group/scenes、相对P31 `+.005`、相对P53 `+.002`；不扫averaging start/schedule/gate。
- 结果：P55/P53/P31 reduction=`.688694/.698266/.694007`；相对P53/P31=`-.009572/-.005313`；
  exact/group/8 scenes通过但两decision gates失败，3/5 gates。
- 结论：固定末20% averaging退化；不改window/schedule，不继续weight-averaging family。下一编号=`V67-F44`。

### V67-F44 — P57 fixed-radius SAM gradient hybrid候选

- 分类：`algorithm/sharpness-aware-optimization`；状态：`closed_negative_after_single_recovery`。
- 调研迁移：SAM（ICLR 2021）优化邻域worst loss；ASAM（ICML 2021）提供scale-aware扩展。P57只用标准SAM，
  固定`rho=.05`，不尝试ASAM或radius sweep。
- 方法：P53 data/model/gradient/budgets/anchor/loss/seed/epochs不变；每epoch标准两步SAM。
- 数据：P10R2-H0.8=`1034/1152` eligible、96 cases；同read比较冻结P53。
- 判定：exact/group/scenes、相对P31 `+.005`、相对P53 `+.002`。失败则关闭sharpness优化family。
- 结果：P57/P53/P31 reduction=`.731922/.723709/.727373`；相对P53=`+.008213`，相对P31=`+.004549`，
  后者差`.000451`未过冻结门；4/5 gates。
- 结论：严格拒绝，不降门/扫rho/试ASAM；flat/sharpness optimization family关闭。下一编号=`V67-F45`。

### V67-F45 — P58 differentiable case-selective residual expert失败

- 分类：`algorithm/case-selective-mixture-of-experts`；状态：`closed_negative_after_single_trial`。
- 调研迁移：DSelect-k（NeurIPS 2021）提供连续可微expert selection；sparse MoE（ICLR 2017）按输入路由专家。
- 方法：P20 frozen base expert + P53形式residual expert；新增固定width8 sigmoid case gate；其余P53合同不变。
- 数据：P6R-H0.8=`868/1152` eligible、96 cases；同read比较P53/P31。
- 异步复现输入：P3C-H0.8=`695/864` eligible、72 cases；仅物化，P58通过前selection read=false。
- 判定：exact/group/scenes、相对P31 `+.005`、相对P53 `+.002`；不扫gate width/expert count/temperature。
- 结果：P58/P53/P31 reduction=`.777488/.774840/.797323`；相对P53=`+.002649`通过，但相对P31
  `-.019835`，scene non-increasing=`5/6`；3/5 gates。gate mean=`.904328`，多数case近全开。
- 结论：不扫gate结构/温度/top-k，不读取P59 replication quality；case-selective expert关闭。检索PRECOG、MotionLM与
  uncertainty-aware actor prediction后，下一步改变预测对象为given-`tau` Actor-state reliability。下一编号=`V67-F46`。

### V67-F46 — P60 trajectory-conditioned Actor-state reliability结果

- 分类：`prediction-object/trajectory-conditioned-actor-state-reliability`；状态：`supported_with_ranking_limitation`。
- 调研迁移：PRECOG（ICCV 2019）按受控Ego goal条件化多Actor forecast；MotionLM（ICCV 2023）支持条件rollout；
  CVPR 2018证明trajectory uncertainty可指示实际预测误差。P60迁移为当前单卡可训练的小型reliability estimator。
- 对象：Actor常速度H-step endpoint error乘`exp(-predicted tau separation/6m)`；只使用logged dense tracks监督，
  不声称counterfactual response。
- Protocol：scene `%5!=0`、H `.8/1.5s`训练；scene `%5==0`、H `2.0s`confirmation；query MLP与同容量
  Actor-only baseline同read。三门=`Spearman .55 / MAE gain 10% / AUROC .75`。
- 锁：不扫width、history、exposure radius、error threshold、split或horizon；失败即改变Actor target/forecast family，
  不回到world-state authority补救。
- 结果：unseen-scene/H2 query Spearman=`.756794`、MAE=`.093437`、AUROC=`.960804`；相对Actor-only MAE
  降低`25.45%`，但Spearman低`.014367`。3/3原门通过；准确结论限于校准/事件识别，排序增益未成立。
- 下一步：P61只加固定pairwise term恢复排序；不改预测对象或P60门。下一编号=`V67-F47`。

### V67-F47 — P61 fixed pairwise Actor-reliability recovery失败

- 分类：`algorithm/pairwise-ranked-actor-reliability`；状态：`closed_negative_after_single_trial`。
- 方法：P60 query head增加weight `.10`、temperature `.05`、gap `.02`、shifts `[1,17,257]`的pairwise loss；
  Actor-only head仍只做Huber，其余特征/架构/epoch exact。
- Protocol：development split改为scene `%5==1`、H2；P60三门加Spearman delta over Actor-only `+.01`。
- 结果：Spearman `.755004/.740135`、delta=`+.014869`，AUROC `.945763`；但MAE
  `.156003/.117128`，query退化`33.19%`；3/4 gates。
- 结论：排序恢复但尺度失配；不扫pair配置。调研后只允许一次train-only monotone calibration恢复。下一编号=`V67-F48`。

### V67-F48 — P62 train-only monotone calibration恢复失败

- 分类：`algorithm/order-preserving-regression-calibration`；状态：`closed_negative_after_single_recovery`。
- 调研迁移：ICML 2018 calibrated regression支持model-agnostic recalibration；UAI 2025强调instance-wise monotonic map
  保持ranking。P62使用最小正斜率affine map，不使用非参数binning/isotonic complexity。
- 方法：P61模型/排序配置exact；训练完成后仅用training prediction/target最小二乘拟合`slope>0,bias`，confirmation
  只apply。第三development split为scene `%5==2`、H2，四门保持exact。
- 锁：不扫map/slope/pair配置；失败即关闭pairwise+calibration recovery。下一编号=`V67-F49`。
- 结果：map=`1.016524*x-.002156`；query/Actor-only Spearman=`.742362/.751193`、MAE
  ` .084206/.081360`、AUROC=`.951511/.950829`；rank和MAE两门失败，2/4 gates。
- 结论：校准近恒等，跨split conflict不是scale-only；pairwise+calibration family关闭。

### V67-F49 — P63 Rank-N-Contrast-inspired representation失败

- 分类：`algorithm/continuous-rank-contrastive-representation`；状态：`closed_negative_after_single_trial`。
- 调研迁移：Rank-N-Contrast（NeurIPS 2023）把continuous target order编码进representation，报告更好的regression与shift
  generalization。P63用固定三shift triplet surrogate预训练query encoder，不向scalar output直接加ranking loss。
- 方法：500 contrastive epochs；冻结encoder后1000 Huber head epochs；Actor-only baseline总计1500 Huber epochs。
  第四split scene `%5==3`、H2；四门exact，不扫contrastive配置。
- 结果：query/Actor-only Spearman=`.242097/.799512`、MAE=`.136035/.098264`、AUROC=`.820529/.961380`；
  1/4 gates，冻结representation不能被linear scalar head可靠读出。
- 结论：不解冻encoder/延长head/扫contrastive配置；family关闭。下一编号=`V67-F50`。

### V67-F50 — P64 plain-Huber Actor-reliability replication支持

- 分类：`replication/plain-actor-reliability`；状态：`supported_second_split`。
- 调研迁移：WACV 2020 actor motion uncertainty采用可部署的feed-forward预测并强调state、velocity、acceleration、heading；
  当前资产没有raster/map supervision，因此P64不虚构大backbone，直接复现P60的低容量state-query MLP。
- 方法：P60特征、plain Huber、架构、1500 epochs、三门exact；第五split scene `%5==4`、H2。
- 结果：query/Actor-only Spearman=`.769725/.424100`、MAE=`.092651/.107035`（降低`13.44%`）、
  AUROC=`.957408/.850937`；3/3 gates，27/32 scenes rank noninferior。
- 结论：P60 plain-Huber在第二split复现；不加辅助loss/calibrator。该population不是fresh confirmation。下一编号=`V67-F51`。

### V67-F51 — P65 uncertainty-native quantile Actor reliability失败

- 分类：`prediction/ordered-quantile-actor-reliability`；状态：`closed_negative_after_single_trial`。
- 方法：同一query representation输出q10/q50/q90，用pinball loss并结构保证顺序；q50承担P60 reliability read，
  q10-q90提供empirical central interval。
- Protocol：train scene `%5!=0`、H `.8/1.5s`；remainder0 scene复用但confirmation H改为未见2.5s。
- 判定：P60三门+80% interval coverage `[.75,.85]`；不扫quantile/coverage/loss/horizon，不声称conformal coverage。
  结果：q50/Actor-only Spearman=`.717354/.800656`、MAE=`.151378/.153607`、AUROC
  `=.962478/.932027`；coverage=`.672228`，2/4 gates。
- 结论：CQR文献明确neural quantile可undercover且finite coverage需独立calibration；当前没有冻结calibration split，
  不做conformal或quantile sweep。下一编号=`V67-F52`。

### V67-F52 — P66 plain-Huber H2.5 isolation支持

- 分类：`ablation/long-horizon-vs-quantile-loss`；状态：`supported_isolation`。
- 方法：P60 plain-Huber exact；P65 scene split/H2.5 exact；唯一变化是去掉quantile head/pinball。
- 判定：P60三门exact。若通过，P65主要归因于uncertainty head/loss；若失败，plain方法边界为H2附近。
- 结果：query/Actor-only Spearman=`.759251/.773758`、MAE=`.149455/.209832`（降低`28.77%`）、
  AUROC=`.945655/.939205`；3/3 gates。
- 结论：H2.5 plain method成立；P65负结果定位于quantile head/loss与未校准interval。下一编号=`V67-F53`。

### V67-F53 — P67 direct binary Actor reliability失败

- 分类：`prediction/direct-binary-actor-reliability`；状态：`closed_negative_after_single_trial`。
- 对象：冻结定义`raw actor state error>1m AND predicted tau separation<=6m`，直接输出unreliable probability。
- 同read：binary query、continuous query、binary Actor-only；正类权重=train negative/positive exact ratio。
- 结果：AUROC=`.939174/.940707/.911397`；binary相对continuous=`-.001533`、相对Actor-only=`+.027777`；2/3 gates。
- 结论：不扫threshold/loss/class weight；direct binary关闭，continuous expected-error score保留。下一编号=`V67-F54`。

### V67-F54 — P68/P69 fixed-coverage selective Actor reliability支持

- 分类：`consumer/selective-regression-triage`；状态：`supported_two_splits`。
- 调研迁移：SelectiveNet（ICML 2019）与selective regression（ICML 2022）将abstention表述为risk-coverage tradeoff，
  并警告group风险可能随coverage下降而恶化；因此固定per-scene 50%，不做global threshold sweep。
- P68：cost -78.93%、unreliable prevalence -88.58%、32/32 scenes nonincreasing，continuous cost低于binary。
- P69：cost -68.80%、unreliable prevalence -87.83%、23/23 scenes nonincreasing，continuous cost低于Actor-only continuous。
- 结论：只支持reliability triage/abstention；不删除Actor、不改geometry、不称authority/planning/safety。下一编号=`V67-F55`。

### V67-F55 — P70 fresh-population绝对尺度迁移失败，selective ordering保留

- canonical：`run://worldsim_v67/WS-V67-P70-FRESH-ACTOR-RELIABILITY-01/20260829T153500Z__fresh-actor-reliability-s0-r2`；
  worldsim-v5六个fresh scene、H2.5共5,471 rows，source-overlap 276/756明确排除。
- 症状：query/Actor-only Spearman=`.808522/.777160`，但MAE=`.186774/.186914`，改善仅`.0753%`而非冻结10%；
  4门中3门通过，全面fresh transfer拒绝。AUROC=`.952704/.967216`也不支持全面优于Actor-only。
- 保留信号：50% selective cost -89.78%、unreliable prevalence -99.40%、6/6 scenes nonincreasing，query selection
  cost仍比Actor-only低`.001823`；因此不是ordering/triage失败，而是absolute scale未跨root获得相对优势。
- 启动故障：r1在数据读取前因非交互SSH shell缺项目`PYTHONPATH`退出；r2只补环境变量，未改科学合同。
- 文献迁移：ICLR 2025 regression TTA指出naive全特征alignment可能恶化回归；ICCV 2021 calibration under covariate
  shift支持以独立calibration domains学习迁移。P71只用重叠scene 276/756训练冻结backbone residual adapter，六个fresh
  scenes继续隔离；不扫adapter/epoch/lr。下一编号=`V67-F56`。

### V67-F56 — P71 target-only hidden-feature residual calibration负迁移

- canonical：`run://worldsim_v67/WS-V67-P71-RESIDUAL-ACTOR-CALIBRATION-01/20260829T160000Z__residual-actor-calibration-s0-r1`；
  276/756共1,875 H2.5 rows训练等容量query/Actor-only linear residual adapters。
- 症状：fresh query MAE `.186774→.266719`（+42.80%）、Spearman `.808522→.554614`；adapted Actor-only
  MAE `.177064`，query相对恶化50.63%。只有selective cost gate通过，1/4拒绝。
- 原因定位：calibration-domain supervised loss下降不代表跨scene transfer；hidden embedding上的instance-dependent residual拥有
  足够自由度重排分数，276/756不能代表其余六scene。
- 保留边界：adapted score的50% triage仍cost -70.99%、unreliable prevalence -84.89%、6/6 scenes不增，但显著弱于frozen P70。
- 文献迁移：CVPR 2021区分accuracy/ranking-preserving calibrator，UAI 2025强调monotonic calibration保序；P72只训练
  positive-slope affine，base score完全冻结。不扫P71 lr/epoch/width。下一编号=`V67-F57`。

### V67-F57 — P72 monotone target calibration未恢复query相对MAE优势

- canonical：`run://worldsim_v67/WS-V67-P72-MONOTONE-ACTOR-CALIBRATION-01/20260829T163000Z__monotone-actor-calibration-s0-r1`；
  query/Actor-only各只训练positive-slope affine，base score冻结。
- 结果：query scale/bias=`.906696/.004084`，MAE `.186774→.185193`（仅+.846%）；Actor-only scale/bias
  `=1.649869/.023415`，MAE `.159585`，所以query相对差16.05%。1/3 gates拒绝。
- 保留：query Spearman `.808522`、AUROC `.952704`和50% selection完全保持，cost -89.78%、unreliable prevalence -99.40%。
- 结论：关闭target calibration family，不扫非线性map/权重/epoch；P70跨root贡献限于ordering/triage。
- 文献迁移：multi-horizon direct forecasting与long/short-term trajectory modeling支持扩大训练horizon support；P73在source
  H `.8/1.5/2.5`联合训练并评估H3，而非继续处理同一H2.5尺度。下一编号=`V67-F58`。

### V67-F58 — P73 multi-horizon训练改善frozen误差但未建立query pointwise优势

- canonical：`run://worldsim_v67/WS-V67-P73-MULTI-HORIZON-ACTOR-RELIABILITY-01/20260829T170000Z__multi-horizon-actor-s0-r1`；
  299,103 base + 141,295 H2.5 rows，GPU warmup/joint训练与两段materialization重叠。
- 结果：H3 query MAE `.228784`，比frozen P66 `.289723`改善21.03%；但Actor-only `.229099`，query只低`.137%`
  而非10%。Spearman query/Actor=`.778665/.746849`，AUROC=`.974031/.966363`，2/3 gates拒绝。
- 保留：50% query selection cost `.034406`，优于Actor `.038821`和frozen `.041978`，相对all降低91.09%；
  unreliable prevalence降低97.76%。
- 结论：不降pointwise gate、不扫horizon mix/epochs；multi-horizon pointwise claim拒绝，但selective ordering进一步增强。
- 文献迁移：SelectiveNet/Selective Regression/Regression Deferral支持直接训练accept/reject函数。P74用scene-horizon内
  最低cost半集作为固定coverage admission监督，在H3.5比较query、Actor-only与P73 continuous。下一编号=`V67-F59`。

### V67-F59 — P74 direct low-risk-half admission弱于continuous expected-cost selector

- canonical：`run://worldsim_v67/WS-V67-P74-FIXED-COVERAGE-ACTOR-ADMISSION-01/20260829T173000Z__fixed-coverage-admission-s0-r1`；
  440,398 source rows训练，H3.5六scene 5,049 rows。
- 结果：query/Actor admission AUROC=`.835296/.817442`；50% selected cost query/Actor=`.067374/.067681`，
  query只低`.45%`，未达5%；P73 continuous=`.047491`，direct admission高41.87%。1/3 gates拒绝。
- 保留：query相对all `.524624`仍降低cost 87.16%，unreliable prevalence `.074668→.007927`，6/6 scenes不增；
  但这些不能覆盖对更强continuous baseline的失败。
- 结论：关闭binary admission family，不扫label quantile/BCE/coverage；保留multi-horizon continuous expected-cost selector。
- 数据对策：按DriveStudio官方modular process keys，只抽取V5 validation role的LIDAR并生成`lidar/calib/objects` 10Hz数据；
  P75在新8-scene cohort H3.5做一次fixed-coverage read，准备IO与GPU训练重叠。下一编号=`V67-F60`。

### V67-F60 — P75 validation LIDAR generic shard routing造成无效全包扫描

- 分类：`engineering/data-routing`；状态：`resolved_by_exact_scene_shards_and_scene_ready_preprocess`。
- attempted run：`WS-V67-P75-FRESH-ACTOR-COHORT-PREP-01/20260829T180000Z__validation-actor-prep-s0-r1`；
  generic extractor在member index尚未建立时把全部3,128 candidates交给10个约30GB gzip shards，2 workers长时间只扫描
  01/02；0/8 processed scenes ready，未读H3.5 target/metric。
- 根因：该通用逻辑适合未知member位置，但本cohort的scene顺序与官方trainval shard分段已经由metadata确定；继续全包扫描只浪费IO，
  不增加科学信息。P75 GPU四horizon训练已完成且主进程保持，不重训。
- 外部迁移：DriveStudio官方NuScenes流程支持按`process_keys`只生成`lidar/calib/objects`；恢复进一步按scene metadata冻结
  `0129/0170→02,0364/0384→04,0640→06,0977/0997→09,1053→10`，五包并发且每包只接所属members。
- 边界：r1保留为工程失败；r2不换scene/horizon/model/gate，不加入图像、mask、quality、hash/checksum/fingerprint。
  下一编号=`V67-F61`。

P76 dense percentile-rank与P77 group-balanced ListNet均只读source rows，并在P75 fresh validation rows产生前冻结；它们用于
填充P75输入IO期间的GPU空档，不改变P75首次fresh read。当前无新增失败，下一编号仍为`V67-F61`。

P78 fixed-coverage boundary-pair目标也在同一blind时段预注册；只有P77结束且P75数据仍未ready才执行。它不复开P74 BCE，
也不更改P75 cohort/gates。当前无新增失败，下一编号仍为`V67-F61`。

P79将horizon作为domain做固定risk-variance regularization，只在P78结束且fresh rows仍未ready时运行；明确不包装为完整
Fishr。当前无新增失败，下一编号仍为`V67-F61`。

P80预注册为一次linear horizon feature modulation，不做高阶interaction/scale sweep；仍只填充fresh IO等待期。
当前无新增失败，下一编号仍为`V67-F61`。

P75 model r1的2,400s readiness上限将早于已观测tar/preprocess wall；r2在fresh read前仅复用r1 source-H3 cache并提前
持久化模型，scientific合同不变。r1若按预期timeout则登记`V67-F61`，不得把它计作算法失败。

恢复结果：五包在`1409.8--1683.5s`完成，3,128 members全找到；scene-ready手工feeder先完成8个最小processed scenes，
父协调器恢复后8/8 `reused_ready`，prep wall=`1982.66s`。P75 r1在旧timeout前完成，故没有timeout failure；预备r2在
joint epoch1001终止且0 fresh read。F60关闭，下一编号仍为`V67-F61`。

### V67-F61 — P75 fresh H3.5未建立相对Actor-only的fixed-coverage mean-cost优势

- canonical：`run://worldsim_v67/WS-V67-P75-FRESH-VALIDATION-MULTI-HORIZON-01/20260829T180000Z__fresh-validation-actor-s0-r1`；
  8 scenes、8,000 rows、280 unreliable events，首次fresh read。
- 结果：query pointwise MAE比Actor低13.12%，AUROC高`.003494`；但50% selected mean cost `.038723`高于Actor
  `.037013`（退化4.62%），且只比P73 `.039619`低2.26%而非5%。1/3 selector gates，严格拒绝。
- 保留：相对all cost降低84.04%，8/8 scenes nonincreasing；query selected unreliable prevalence `.00175`低于
  Actor/P73 `.0025`。后者只生成新的独立可靠性hypothesis，不覆盖mean-cost失败。
- 防重复：不降5%门、不在该cohort调coverage/score fusion/threshold；全面selector claim关闭。下一编号=`V67-F62`。

### V67-F62 — P76--P80 blind source-ranking恢复均未超过P75

- 所有模型在P75 fresh rows/metrics出现前冻结，随后同cohort development read；不是五次独立confirmation。
- selected cost：P75/P76/P77/P78/P79/P80=`.038723/.043334/.053605/.052137/.049604/.045743`；五个恢复均更差。
- P76/P80相对各自Actor-only仍改善8.31%/13.83%，说明τ特征并非完全无效；但dense percentile rank、ListNet、
  boundary pairs、horizon risk variance与linear horizon-FiLM均未建立相对当前best selector的增益。
- 防重复：不扫rank temperature、pairing、V-REx weight、FiLM order、epochs或coverage；该source-ranking恢复族关闭。
  下一编号=`V67-F63`。

P81--P83已在新test-role sensor/target read前冻结：P81把primary endpoint改为P75留下的fixed-coverage unreliable-event
prevalence，P82/P83分别训练全source pairwise AUC与horizon-balanced pairwise AUC；当前tar IO与GPU training并行，尚无
新增scientific/engineering failure。P75 mean-cost负结论保持不变，下一编号仍为`V67-F63`。

P84在同一target-read前处理一个潜在对象错配：全row event metric可能被`separation>6m`的未访问状态主导，因此新增
visited-region-only denominator与去τ重复的Actor failure因子化模型。它不修改P81 protocol/verdict，当前仍无新增失败；
下一编号仍为`V67-F63`。

P85进一步把visited Actor rows按`scene/anchor/τ`聚合为“任一访问状态不可靠”的trajectory-level对象；显式
`anchor_frame`只是group字段，不是hash/checksum/fingerprint。协议在target read前冻结，当前无新增失败，下一编号仍为
`V67-F63`。

P86进一步冻结direct trajectory set-summary model；source anchor rows物化与P84 GPU训练并行。它不读取P81 target来选
aggregation/loss/radius，当前无新增失败，下一编号仍为`V67-F63`。

### V67-F63 — P86 r1 trajectory aggregation逐group全表扫描

- 分类：`engineering/cpu-group-aggregation-complexity`；状态：`resolved_before_training_or_test_read`。
- 症状：576,032 source rows使用`for group -> flatnonzero(inverse==group)`，复杂度`O(N×G)`；r1只写resolved，GPU训练、
  model artifact与test target read均为0。
- 调研/修复：按NumPy官方`unique(return_inverse)`与stable `argsort`语义，一次排序后遍历连续group slices；baseline
  group-max也用相同线性遍历。未改source、features、aggregation定义、model、loss、epochs、gates或fresh cohort。
- r2已进入GPU训练；r1不计scientific trial。下一编号=`V67-F64`。

### V67-F64 — P81 prep r1错误假设blob shard按scene index百位切分

- 分类：`engineering/dataset-archive-routing`；状态：`resolved_r2_active_before_target_read`。
- 症状：r1只在`scene-0016→01`命中389与`scene-0523→05`命中391；错误绑定03/07/08/09均0，最终缺
  3,120/3,900 LIDAR files。0 scene preprocess、0 Actor rows、0 target/model evaluation。
- 卡点调研：nuScenes官方/devkit只要求十个blob archives合并解压，不承诺metadata index分包；读取archives开头真实
  session members后，按采集session冻结`04:{0344,0330,0923,0963}`、`06:{0627,0784}`、
  `10:{1059,1071}`，保留已证实`01:{0016}`、`05:{0523}`。
- r2复用780已提取members，只扫描04/06/10；不改cohort、target、models、gates或claim。下一编号=`V67-F65`。

P87/P88分别以Deep Sets与set attention填充纯IO等待，均在target read前冻结；当前无新增失败，下一编号仍为
`V67-F65`。

P89多阈值ordinal trajectory reliability同样在target read前冻结；当前无新增失败，下一编号仍为`V67-F65`。

P90 plain continuous trajectory max-error Huber同样在target read前冻结，用于检验既有source结果中plain regression
相对复合rank supervision的迁移优势；不修改P81 cohort、target或coverage，不进行loss/threshold sweep。

### V67-F65 — P90 r1直接脚本入口缺少仓库级Python import path

- attempted entry：从repo root直接执行`python scripts/run_worldsim_v67_p90_plain_trajectory_max_error.py ...`；
- symptom：入口导入`motion_proj.worldsim_v67`时抛`ModuleNotFoundError: No module named 'motion_proj'`；
- exposure：发生在argument parse、run创建、source load、epoch与P81/P85 target read之前，故r1不计scientific trial；
- root cause/literature response：Python官方command-line文档规定直接执行文件时把脚本目录而非当前repo root放在
  `sys.path`首位，并说明`PYTHONPATH`用于扩展module search path；与既有V65-F18同类；
- resolution：仅为r2进程设置`PYTHONPATH=.`，不改代码、model、loss、cohort、target、coverage或gate；r2已恢复GPU训练；
- prevention：后续repo-root脚本统一显式进程级`PYTHONPATH=.`，不修改全局shell环境。

下一可用编号：`V67-F66`。

P91固定q=.90 conditional max-error quantile在P81 target read前冻结，作为P90 mean-oriented Huber的单一tail-risk
对照；不扫quantile或coverage，不改变cohort/gates。当前无新增失败，下一编号仍为`V67-F66`。

### V67-F66 — P81 prep r2把部分scene session过早绑定到单个archive

- 分类：`engineering/dataset-archive-routing`；状态：`resolved_r3_active_before_target_read`；
- 症状：r2扫描04/06/10并复用01/05的780 files后仍缺1,175/3,900 LIDAR files；逐scene缺失为
  `0923:388`、`0784:398`、`0963:389`，其余7 scenes完整；0 scene preprocess、0 Actor rows、0 target read；
- root cause：用少量archive header把scene假设为同一候选包，没有查询已有完整member-shard manifest；scene table、
  capture session与十个download blob的分组不存在可推导的一一映射；
- literature/open-source response：nuScenes官方/devkit只要求合并十个blob archives；工程恢复转用项目已有、由完整
  archive扫描生成的V4 test member-shard manifest，而不继续猜scene index或单包session范围；
- exact recovery：manifest给出`n008-2018-08-30-15-31-50-0400→08`以及两条`n015` sessions
  `2018-10-08-15-44-23/2018-09-25-13-17-43→09`；故r3冻结`0784→08`、`0923/0963→09`，只扫描08/09并复用
  已提取2,725 files；
- claim impact：只修复输入路由，不换scene、不改H3.5、model、target、coverage、gate或claim；r2不计scientific trial。

下一可用编号：`V67-F67`。

P92 heteroscedastic Gaussian trajectory failure probability在P81 target read前冻结；不修改P81合同、不扫分布或variance
bound。当前无新增失败，下一编号仍为`V67-F67`。

P93 direct trajectory any-failure BCE同样在P81 target read前冻结；阈值固定为正式1m endpoint，不扫class weight或loss
组合。当前无新增失败，下一编号仍为`V67-F67`。

P94三成员direct-probability deep ensemble在P81 target read前冻结：seed0复用P93，seed1/2同协议独立初始化，最终只取
算术均值且不选member/subset。P90--P92 checkpoint已确认落盘后安全退出等待进程以释放4.5GiB，后续evaluation-only
恢复；无artifact/quality损失，不登记failure。下一编号仍为`V67-F67`。

### V67-F67 — visited Actor最大位移误差不产生稳定的task-conditioned trajectory增益

- 分类：`scientific/target-definition`；状态：`closed_negative_migrated_to_occupancy_flip`；
- independent signal retained：P81全row primary在9,559 rows上query/Actor/P73 selected events=`26/57/45`，相对
  Actor减少54.39%，10/10 scenes nonincreasing，3/3 gates通过；P82/P83 secondary均为0 events；
- faithful visited failure：P84在2,113 visited rows上candidate/P75=`235/208`；P85在1,089 trajectories上
  candidate/P75=`203/199`，均拒绝；
- direct family：P86--P94全部拒绝。最佳P86为query/Actor/P75=`187/193/199`，但query增益仅3.11%<10%，absolute
  reduction 37.48%<50%；P90=`191/201/199`，P94 ensemble=`204/189/199`；
- root cause：原label是Actor endpoint constant-velocity error，只依赖Actor history/future；τ只决定membership。于是
  Actor-only已获得目标的主要可预测部分，query features既不改变label也不能稳定提供预注册增益；all-row成功还可由
  separation>6m的未访问rows贡献，不能外推为visited-state reliability；
- literature/open-source response：ICCV 2021 safety-aware motion prediction将planner critical region表示为earliest
  occupancy，CVPR 2023 IMPLICITO只在candidate trajectory附近的spatiotemporal query points预测occupancy/flow；两者都
  指向“路径上的occupancy结论”而不是路径外的Actor位移误差；
- resolution：关闭max-error visited trajectory family，不降gate、不挑P86/P90、不继续architecture/loss/seed sweep；
  P95将target改为predicted-vs-observed Actor path对同一τ的occupancy decision flip。P81 cohort仅development，剩余10
  test-role scenes继续未读，只有development支持才做one-shot confirmation；
- claim impact：V6.7当前只支持全row event triage，不支持visited Actor reliability、planner/policy/closed-loop/safety。

下一可用编号：`V67-F68`。

P95 occupancy-flip迁移已冻结radius/time samples/width/coverage/model，不增加hash/checksum/fingerprint；当前无新增失败，
prep得到source/development row-level flips=`2,273/96`及false-safe=`925/32`，非空且未触发target恢复；当前无新增失败，
下一编号仍为`V67-F68`。

P95 development以query/Actor/P75 selected flips=`7/28/13`通过4/4 gates；由于P81 cohort已消费，只触发P96 remaining
10-scene one-shot confirmation，不记作独立成功。P96 cohort/shards/model/target/gates均已在sensor read前冻结；当前无新增
失败，下一编号仍为`V67-F68`。

P97只从冻结P95 artifacts派生false-safe target并与P96 archive IO重叠；0 new sensor/target read，且明确禁止替换P96
endpoint或model。

### V67-F68 — one-sided false-safe reliability不优于冻结P75且排序接近反向

- run：`run://worldsim_v67/WS-V67-P97-TRAJECTORY-FALSE-SAFE-01/20260830T005000Z__trajectory-false-safe-s0-r1`；
- symptom：development 1,791 trajectories/31 false-safe中，fixed50 query/Actor/P75 selected=`11/16/10`；query虽相对
  Actor减少31.25%、相对all减少28.99%，但仍比P75多1，query/Actor AUROC仅`.44692/.45555`；2/4 gates；
- root cause：row-level source/development positives只有925/32，trajectory endpoint更少；单独拆出false-safe破坏P95 total
  flip中由较多false-alarm提供的可学习排序，source BCE接近0也未迁移；
- literature response：NeurIPS 2021 rare-event工作指出信息量受positive数量约束且negative subsampling需log-odds校正；
  ICCV 2021 safety-aware occupancy使用专门hard/soft/unseen losses。当前项目没有独立校准cohort，不能事后引入这些
  weight/sampling corrections并把同一development调成成功；
- resolution：关闭standalone false-safe candidate，不加focal/class weight、不改threshold/radius；P98仅补齐互补
  false-alarm attribution，P96保持冻结total-flip endpoint/model；
- claim impact：无false-safe独立可靠度claim，不影响P95 development或尚未读取的P96。

下一可用编号：`V67-F69`。

P98 false-alarm attribution从冻结rows派生且不修改P96；development fixed50 query/Actor/P75=`0/25/3`、AUROC=
`.92312/.67813`，4/4 gates通过。冻结P95的真实selected subtype另为query/Actor/P75=`3+4 / 17+11 / 10+3`
（false-safe+false-alarm），所以P98不能被用来声称P95主要移除false alarms，也不能覆盖V67-F68或产生safety claim；
当前无新增失败，下一编号仍为`V67-F69`。

P99 equal-weight two-head multi-task在development选择8 flips（6 false-safe+2 false-alarm），优于Actor 39/P75 13但未超过
P95的7；它是V67-F68后的唯一shared-representation recovery，不替换已冻结P96 model，也不新增failure。下一编号仍为
`V67-F69`。

P100 temporal-clearance query augmentation从冻结P95 rows解析派生3维、0 new read；development fixed50
query/Actor/P75=`9/41/13`，4/4 gates但未超过P95的7。它是positive mechanism result，不登记failure、不替换P96，
也不扫feature/loss；下一编号仍为`V67-F69`。

P101针对P100压缩时间交互的表示瓶颈，一次性迁移为与occupancy target同构的9-step signed-clearance/boundary-distance
profile；development fixed50 query/Actor/P75=`13/29/13`，4/4 gates但只追平P75且不及P95。它不是正式失败，且不以
profile length/threshold sweep补救。P102只做一次hierarchical temporal-token→Actor-set结构恢复，P96仍只确认冻结P95；
P102得到`4/27/13`并刷新development best。P103 checkpoint/protocol在P96 target前冻结为prospective secondary，
不改变primary。至此无新增scientific failure，下一编号仍为`V67-F69`。

### V67-F69 — P96 scene-0556 session的相邻日期shard推断错误

- 分类：`engineering/archive-routing`；状态：`resolved_pre_target_exact_shard_recovery`；
- symptom：冻结cohort所需`scene-0556` session=`n008-2018-08-31-11-37-23-0400`事前按archive03相邻session推断，
  03完整扫描对390 candidates命中0；发生时仅08已精确命中397，10个processed scenes未ready，P96/P103 target rows不存在；
- root cause：10个nuScenes trainval blob parts不是按scene index或简单session日期连续分桶；相邻archive header不足以外推；
- literature/open-source response：nuScenes官方论坛与开源dataset setup确认10 parts提取后合并为一个dataroot，但公开文档
  不提供session→part索引。恢复因此不继续猜日期：02/04/05/07 exact locators先全部排除；随后对r1虽完整扫描、但其
  candidate filter未包含0556的01/06/08/09/10作第二轮exact-session locator，最终在06精确命中并停止其余workers；
- frozen recovery：只修`0556:03→06`、以exact-session tar补390 files并复用其他3,511 files，prep r2 active；不换scene/cohort/model/
  target/radius/width/coverage/gates，P95 primary与target read仍exact-once；不增加hash/checksum/fingerprint；
- claim impact：纯pre-target I/O failure，不改变P95/P102 development结果，也不产生confirmation evidence。

P96 prep r2最终3,901/3,901 mapped、newly extracted=350、10/10 preprocess，wall=`448.10s`；0556 exact shard=06。
恢复过程中未读取target、未换cohort/model/gate，故`V67-F69`关闭。

下一可用编号：`V67-F70`。

### V67-F70 — time-local-only监督使Actor-only固定覆盖排序优于query

- 分类：`scientific/objective-factorization`；状态：`resolved_by_single_multitask_recovery`；
- run：`run://worldsim_v67/WS-V67-P104-TEMPORAL-FLIP-SUPERVISION-01/20260830T030500Z__temporal-flip-supervision-s0-r1`；
- symptom：development 1,791 trajectories/95 flips上fixed50 query/Actor/P75=`1/0/13`；query absolute reduction
  `97.89%`且AUROC `.90726`，但Actor-only选0，因此query-vs-Actor=`-100%`，只过3/4 gates；
- root cause：5,336 positives/5,180,364 time tokens极稀疏，balanced time-local classification学到Actor motion shortcut；
  固定time max→Actor max又放大单token risk，丢失P102 trajectory-level set objective中的candidate-relative排序约束；
- literature response：CVPR 2023 IMPLICITO在连续spatiotemporal points联合表示occupancy/flow，CVPR 2024 Cam4DOcc对
  多future steps的occupancy与flow使用联合损失，而非用逐时辅助头替换最终forecast objective；
- resolution：不删relative gate、不把1 event包装成功、不扫sampling/aggregation/weight。唯一P105保留P102正式
  trajectory BCE与hierarchical model，将P104逐时flip只作equal-weight auxiliary；失败即关闭local-supervision family；
- claim impact：P104没有独立或task-conditioned success claim，不影响P102 development或P96/P103冻结confirmation。

P105 canonical r2以trajectory BCE为primary、equal-weight time-local BCE为auxiliary，fixed50 query/Actor/P75=
`6/27/13`，absolute/query-vs-Actor reduction=`87.36%/77.78%`，AUROC=`.89704/.64036`，4/4 gates。它解决
P104的relative failure，但未超过P102的4；因此只关闭`V67-F70`，不替换P103，不继续扫auxiliary weight或sampling。

下一可用编号：`V67-F71`。

### V67-F71 — P105 r1使用不存在的torch.flatnonzero入口

- 分类：`engineering/framework-compatibility`；状态：`resolved_before_first_optimizer_step`；
- symptom：P105 r1在首次构造temporal auxiliary indices时抛`AttributeError: torch has no attribute flatnonzero`；
  run目录已创建并完成source tensor装载，但0 optimizer step、0 development/confirmation evaluation；
- literature response：PyTorch官方文档规定`torch.nonzero(input, as_tuple=False)`返回nonzero坐标，`torch.flatten`
  保持元素顺序；恢复先flatten boolean mask，再取nonzero并flatten index tensor，与原意等价；
- resolution：只替换index API并以新run-id r2重启；cohort/data/architecture/loss weight/batch/epochs/selection/gates不变，
  r1保留不覆盖；
- claim impact：纯入口失败，无scientific evidence，不计P105 trial，也不影响P96/P103。

下一可用编号：`V67-F72`。

### V67-F72 — P106 r1漏做occupancy-flip target adapter而误训Actor endpoint error

- 分类：`engineering/target-adapter`；状态：`resolved_by_exact_adapter_restore`；
- symptom：r1 development出现1,636/1,791 events、query/Actor/P75=`828/829/753`，与冻结occupancy-flip cohort的
  95 events明显不一致；检查表明P104 raw rows保留原`raw_actor_state_error_m`，P106 prep未执行P95的field rebinding；
- exposure：r1完成了错误target的训练/评估，但没有读取P96/P103，也未改变任何冻结checkpoint；该run整体作废，不作为
  scientific data-scale trial或negative result；
- literature response：CVPR 2023 IMPLICITO对query-point occupancy ground truth直接使用BCE；P106必须监督明确的
  predicted-vs-observed occupancy decision flip，而非Actor位移误差；
- resolution：r2 prep显式保留`actor_position_error_m`后，将`raw_actor_state_error_m`和`target_cost`设置为
  `occupancy_decision_flip.float`，新prep/model run-id重跑；source volume/model/loss/epochs/development/gates不变；
- claim impact：纯adapter failure，不计P106科学试验，不影响P102/P103/P96。

下一可用编号：`V67-F73`。

### V67-F73 — all-source scale-up产生negative transfer并劣于P75

- 分类：`scientific/source-domain-transfer`；状态：`closed_negative_no_target_weighting`；
- canonical：`run://worldsim_v67/WS-V67-P106-ALL-SOURCE-HIERARCHICAL-01/20260830T042000Z__all-source-hierarchical-s0-r2`；
- symptom：正确95-event development上fixed50 query/Actor/P75=`16/25/13`；query虽相对Actor减少36%、absolute减少
  66.30%，但比P75多3，prevalence ratio=`1.2308`，2/4 gates；AUROC=`.76717/.69915`；
- interpretation：原4/5 source训练的P102为`4/27/13`；一次性加入23 scenes/114,575 rows后，source trajectories
  79,478→97,441却外推退化，属于可观测negative transfer，不是容量或训练不足证据；
- literature response：CVPR 2019 Characterizing and Avoiding Negative Transfer指出弱相关source会伤害target，并以
  source filtering/weighting缓解；但这里唯一development已消费、P96未读，事后按development筛scene或学weights会污染结论；
- resolution：关闭all-source scaling，不挑remainder、不扫source subset/epoch/normalization、不引入target-dependent
  domain weights；保留P102原4/5 source模型为best与P103唯一frozen checkpoint；
- claim impact：不支持“更多source data单调改善reliability”，不影响P102 development或P96/P103 confirmation。

下一可用编号：`V67-F74`。

### V67-F74 — P95 occupancy-flip primary无法独立超过Actor-only

- 分类：`scientific/independent-generalization`；状态：`closed_negative_one_shot_primary`；
- canonical：`run://worldsim_v67/WS-V67-P96-OCCUPANCY-FLIP-CONFIRMATION-01/20260830T004000Z__occupancy-flip-confirmation-s0-r1`；
- symptom：fresh 1,720 trajectories/36 flips上fixed50 query/Actor/P75=`8/5/12`；query absolute reduction=55.50%且
  优于P75，但比Actor多60%，query/Actor AUROC=`.65542/.71181`，relative gate失败，3/4 gates；
- subtype：query=`7 false-safe+1 false-alarm`，Actor false-safe=0，P75 false-safe=10；不能把相对P75改善写成
  task-conditioned或safety gain，因为Actor-only更优；
- root cause：P95 development的τ features增益未跨cohort稳定；Actor dynamics本身携带主要可迁移risk，end-to-end query
  classifier把clearance/context相关性学成cohort-specific shortcut；
- resolution：按one-shot rule关闭P95 primary，不换scene、不降query-over-Actor gate、不用P103覆盖primary、不在同read
  选择P104--P106；
- claim impact：不支持independent trajectory-conditioned occupancy-decision reliability、collision或safety claim。

下一可用编号：`V67-F75`。

### V67-F75 — P102 hierarchical temporal secondary仍无法独立超过Actor-only

- 分类：`scientific/representation-generalization`；状态：`closed_negative_prospective_secondary`；
- canonical：`run://worldsim_v67/WS-V67-P103-HIERARCHICAL-CONFIRMATION-01/20260830T024000Z__hierarchical-confirmation-s0-r1`；
- symptom：同一fresh cohort fixed50 query/Actor/P75=`9/7/12`，absolute reduction=49.94%、query-vs-Actor=-28.57%，
  AUROC=`.74385/.67973`，3/4 gates；query false-safe/false-alarm=`7/2`；
- interpretation：hierarchical temporal tokens把development从P95的7改善到4，但independent仍落后Actor-only；这排除
  “只需更强query temporal encoder”作为P95失败的充分解释；
- literature response：CoRL MultiPath把intent/control uncertainty分层并支持closed-form space-time collision queries；
  下一对象应先估Actor uncertainty distribution，再解析投影到τ clearance，而非继续端到端query分类器；
- resolution：关闭P102/P103 representation family，不扫seed/width/pooling/auxiliary/data scale；若继续P107，P81/P96
  只作consumed development，新confirmation必须另冻target-unread cohort；
- claim impact：无independent hierarchical task-conditioned claim，不影响P81 all-row triage窄结论。

下一可用编号：`V67-F76`。

### P107 launch note — 不再训练end-to-end query classifier

- P95/P102在development强、P96/P103独立relative失败，Actor-only在fresh cohort更稳；因此不把卡点解释为继续扩模型的理由；
- literature response：MultiPath将Actor intent/control uncertainty与candidate trajectory的space-time collision query分开，
  P107迁移为Actor q90 time-local error tube加固定clearance解析投影；网络完全不读candidate τ；
- prevention：P81/P96只作已消费development，不换gate包装独立成功；q90、`.05m` floor、max聚合与fixed50一次冻结，
  禁止quantile/floor/aggregation/seed/width sweep；若失败登记`V67-F76`并换研究对象；
- execution：source materialization一交付即启动GPU训练，同时继续两个development cohort的CPU/IO；未增加hash、
  checksum、fingerprint或smoke/regression matrix，单3090资源足够。

下一可用编号仍为：`V67-F76`。

### V67-F76 — P107首次model launcher在后台分组后丢失仓库工作目录

- 分类：`engineering/process-launch`；状态：`resolved_before_run_creation`；
- symptom：同一SSH命令中prep后使用`&`，shell把带`cd`的前段置入后台子shell，model随后从`/root`解析相对
  `scripts/run_worldsim_v67_p107_actor_uncertainty_tube.py`并立即报文件不存在；
- exposure：0 model run directory、0 source/development load、0 optimizer step、0新target read；prep进程正常继续；
- resolution：不改代码/数据/quantile/floor/aggregation/steps/coverage，只以绝对script、config与PYTHONPATH启动原定
  canonical r1；model等待source artifact后自动接管GPU；
- prevention：后续独立后台作业均显式使用绝对入口，避免依赖同一复合shell中的`cd`作用域；不增加校验或测试矩阵；
- claim impact：纯launcher failure，不计科学trial，不改变P107 verdict边界。

下一可用编号：`V67-F77`。

### V67-F77 — P107 r1在source压缩写完前读取final NPZ

- 分类：`engineering/artifact-delivery-race`；状态：`resolved_before_first_optimizer_step`；
- symptom：producer创建`SOURCE_ACTOR_UNCERTAINTY_ROWS.npz`后仍在`np.savez_compressed`写zip，等待方只判断文件存在，
  r1遂在`np.load`报`BadZipFile`；
- exposure：run目录/resolved config已创建，但0 optimizer step、0 development evaluation、0新target read；producer随后
  正常完成575,596 source rows以及P81/P96两个development artifacts；
- resolution：r2只复用完整artifact立即训练；producer今后写`.partial.npz`，完成后由同目录`Path.replace`原子交付final；
  不增加hash、checksum、fingerprint或内容门控；
- claim impact：纯producer-consumer race，不计科学trial；q90/model/steps/clearance/aggregation/coverage与cohorts不变。

下一可用编号：`V67-F78`。

### P107 outcome note — V67-F77后因子化development在两个consumed cohorts一致成立

- canonical：`run://worldsim_v67/WS-V67-P107-ACTOR-UNCERTAINTY-TUBE-01/20260830T061000Z__actor-uncertainty-tube-s0-r2`；
- P81 fixed50解析τ-risk/Actor/P75=`2/36/13`，P96=`2/9/12`；两者query-over-Actor分别减少94.44%/77.78%，
  query AUROC `.92901/.87305`；
- interpretation：Actor-only q90 tube保留可迁移dynamics uncertainty，τ只通过物理boundary clearance解析进入排序，避免
  P95/P102的cohort-specific query shortcut；这是跨两个已消费cohort的机制证据，不是独立确认；
- next：冻结P107 checkpoint、normalization、q90、`.05m` floor、time/Actor max、H3.5和fixed50，另取target-unread
  cohort作一次primary confirmation；不在新target read后改score/model/gate。

下一可用编号仍为：`V67-F78`。

### P108 freeze note — 新confirmation只在scene level独立

- target-unread cohort固定为`0092/0329/0555/0012/0035/0268/0795/0917/0925/1060`，四location覆盖且cohort内
  10个distinct sessions；选择只用official split/order、location/session metadata与既有processed-path absence，不读target；
- frozen primary是P107 r2 checkpoint；q90、normalization、`.05m` clearance floor、time/Actor max、H3.5、fixed50
  全不变，只比较是否严格少于Actor-only且不多于P75，不复制P96的冗长gate matrix；
- limitation：部分session在历史cohort有相邻scene，因此只声称scene-level independent；不会写session-level、collision、
  planning或safety generalization；
- route miss只可在target materialization前定位exact shard并修locator，不换scene；target一旦读取，任何失败直接登记
  `V67-F78`并关闭primary recovery。

下一可用编号仍为：`V67-F78`。

### P109 launch note — 用方向投影检验P107 isotropic tube近似

- motivation：P107 scalar q90除以absolute clearance忽略Actor误差方向；MultiPath的分层uncertainty与closed-form query启发
  将Actor signed residual distribution沿candidate boundary normal解析投影；
- fixed method：diagonal Gaussian mean/scale、Gaussian NLL、9 time samples、linearized signed-clearance margin、time/Actor max；
  网络不读τ，且不扫full covariance、scale floor、loss、projection、aggregation、seed或coverage；
- evidence boundary：只先读已消费P81/P96；通过后才允许在P108 target rows出现前冻结prospective secondary。P108 frozen
  P107 primary、cohort和decision均不改变；
- execution：P109 source producer先原子交付source，GPU随后训练，同时producer继续development IO并与P108 archive IO重叠。

下一可用编号仍为：`V67-F78`。

### P109 outcome note — directional projection在两个consumed cohorts均为0 selected events

- canonical：`run://worldsim_v67/WS-V67-P109-DIRECTIONAL-ACTOR-UNCERTAINTY-01/20260830T062500Z__directional-actor-uncertainty-s0-r1`；
- outcome：P81 query/Actor/P75=`0/44/13`、P96=`0/5/12`，query AUROC `.96764/.90434`；6,000-step final
  Gaussian NLL=`-3.64128`，无工程/资源失败；
- literature response：最新开源semi-analytic collision研究同样将stochastic boundary crossing作为独立于spatial overlap的
  高效估计路线；这里保留更窄的linearized occupancy-flip ranking解释，不宣称Monte Carlo等价或calibrated probability；
- resolution：P109 development执行条件满足，P110 checkpoint/config在P108 target rows出现前冻结并只作同read prospective
  secondary；P108 P107-scalar primary的cohort、score和decision不改变。

下一可用编号仍为：`V67-F78`。

### P107/P109 mechanism note — clearance-only不能解释跨cohort增益

- frozen baseline：每个Actor/time取`1/max(abs(predicted separation - interaction radius), .05m)`，再time/Actor max与
  per-scene fixed50；不训练、不调floor/aggregation/coverage；
- result：consumed P81选择1/95 events、AUROC `.91404`，consumed P96选择13/36、AUROC `.79879`；对照P107=`2/2`、
  P109=`0/0`；
- interpretation：P81上的强结果有显著boundary-distance成分，但clearance-only跨到P96明显退化；Actor uncertainty与方向
  投影提供了不能由纯几何解释的稳定性证据；
- protocol impact：P108/P110在target read前仅增加同一descriptive comparator，不改变P108 frozen primary或P110 decision，
  不创建gate matrix、threshold sweep或新claim。

下一可用编号仍为：`V67-F78`。

### V67-F78 — nonlinear Gaussian occupancy sampling跨cohort劣于linearized boundary projection

- 分类：`scientific/uncertainty-query-approximation`；状态：`closed_negative_no_sampling_sweep`；
- canonical：`run://worldsim_v67/WS-V67-P112-NONLINEAR-GAUSSIAN-CROSSING-01/20260830T065000Z__nonlinear-gaussian-crossing-s0-r1`；
- symptom：固定256-sample nonlinear recomputation在P81保持0 selected events且AUROC `.97228`，但P96变为3 events/
  AUROC `.85852`，劣于P109 linearized projection的0/`.90434`；
- interpretation：unimodal diagonal Gaussian在完整2D distance非线性下的finite-sample tail会放大scale/mean误差；boundary-normal
  projection更直接对齐occupancy decision boundary，跨cohort反而稳定；
- literature response：2025 open-source semi-analytic work分别研究spatial overlap与stochastic boundary crossing；本结果支持在
  当前数据/模型保留boundary-crossing approximation，而不是假设更“精确”的spatial sampling必然更好；
- resolution：关闭sample-count/full-covariance/distribution/seed sweep，保留P109 frozen linear score；P108/P110不变；
- claim impact：不支持nonlinear sampled collision probability或calibration claim，不影响P109 development机制结果。

下一可用编号：`V67-F79`。

### P108/P110/P111 outcome note — independent factorization成立但clearance baseline限制贡献归因

- P108 canonical：`run://worldsim_v67/WS-V67-P108-UNCERTAINTY-TUBE-CONFIRMATION-01/20260830T063500Z__uncertainty-tube-confirmation-s0-r1`；
  fresh fixed50 P107/Actor/P75=`5/35/20`，AUROC `.95107/.77605`，2/2 decisions，scene-level independent primary支持；
- P110 same-read prospective secondary：directional/Actor/P75=`1/53/20`，AUROC `.96027/.69142`，2/2 decisions；
- P111 frozen no-learning comparator：clearance-only=`1` event，AUROC `.91644`；它优于P107 fixed50 event count并与P110持平，
  但全排序低于P110；
- interpretation：证据支持“Actor uncertainty distribution与candidate τ boundary解析分离”相对Actor-only/P75稳定迁移，
  也支持direction-aware score的ranking增量；但强geometry baseline阻止把全部事件收益写成learned uncertainty贡献；
- claim boundary：P108只scene-level independent；P110/P111同一read；无session-level、calibrated probability、collision、
  planner、policy、closed-loop或safety claim。无新failure，下一编号保持`V67-F79`。

下一可用编号仍为：`V67-F79`。

### P113 freeze note — 独立检验learned directional uncertainty是否超过纯clearance

- motivation：P108 scalar primary虽超过Actor/P75，但P111 clearance fixed50更少；P110 directional与clearance同为1 event、
  AUROC多`.04383`。需要新cohort检验全排序增量，不能在P108同read上升级claim；
- target-unread cohort：`0094/0331/0521/0003/0013/0038/0797/0920/0926/1061`，四location、10 distinct sessions；
- frozen decision：directional selected events不多于clearance，并且AUROC gain≥`.02`；P109 checkpoint/projection、baseline floor、
  H3.5、time/Actor max、fixed50全部冻结，不引入Actor/P75 gate matrix；
- stop rule：一次read；失败登记下一可用编号（P116完成后为`V67-F82`）并关闭uncertainty-over-geometry claim，不换cohort/
  metric/floor/model或做recovery。
- failure-ID note：P114/P115/P116占用`V67-F79/F80/F81`；P113 prep locator已占F82，P118机制负结果占F83，因此P113若
  scientific decision失败使用`V67-F84`。编号变化不改变任何scientific decision。

下一可用编号：`V67-F84`。

### P114 freeze note — downstream tail aggregation不占用P113确认

- motivation：P109的Actor/time max没有显式传播多个crossing probabilities；task-relevant failure detection提示应在downstream
  cost空间聚合预测分布，而不是继续扩raw query classifier；
- fixed method：冻结P109 Gaussian，只训练top-16 crossing probabilities加independent-union proxy的正权重monotone pool；
  P81/P96均已消费，只作development，P113 rows不会被读取或用于model selection；
- prevention：不扫top-k/pool/loss/seed/coverage，不以P114覆盖P113 primary；若development失败登记`V67-F79`并关闭该tail-pool
  形式，若成功下一failure ID仍为`V67-F79`且只能另取未来target-unread cohort；
- execution：P113 archive IO期间运行6,000-step GPU训练，避免把I/O等待变成研究停顿；不增加hash/checksum/fingerprint或测试矩阵。

### V67-F79 — learned monotone tail pooling稀释directional boundary maximum

- 分类：`scientific/downstream-tail-aggregation`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P114-MONOTONE-TAIL-RISK-01/20260830T071000Z__monotone-tail-risk-s0-r1`；
- symptom：79,478 source trajectories上的balanced BCE降至`.304205`，但consumed P81 AUROC从P109 max的`.967639`
  降至`.951378`；P96从`.904345`降至`.902976`且fixed50 selected events由0增至1。三项冻结decision全失败；
- interpretation：occupancy flip由最接近boundary的Actor/time局部tail主导；将top-16 crossing probabilities与independent-union
  proxy作正权重混合，会把弱、强相关的time/Actor probabilities累积进去。source discrimination改善不能替代跨cohort排序；
- literature response：task-relevant failure detection支持把预测分布传播到downstream cost，但不保证independence-style pooling适合
  强时序/多Actor相关的occupancy boundary。当前结果保留直接对齐decision boundary的P109 maximum；
- resolution：不扫top-k、union公式、model、loss、seed或coverage；P114不进入P113或未来confirmation。P113冻结候选不变；
- claim impact：无trajectory tail calibrator、collision probability或safety claim；不影响P108 factorization primary与P109/P110
  directional evidence。

下一可用编号：`V67-F80`。

### P115 freeze note — 从marginal pooling转向coherent Actor residual sequence

- card point：`V67-F79`证明top-k/independent-union downstream pool在两个consumed cohorts都劣于P109 max，不允许回扫pool；
- literature response：ICCV 2023 Joint Metrics Matter指出marginal metrics/forecasts会遗漏多Agent联合一致性；PRECOG显式建模
  multi-agent conditional futures；CVPR 2026 FoSS使用frequency-domain trajectory结构建模长时依赖；
- migration：P115只改Actor-only uncertainty representation，一次输出9-step residual的固定前4 DCT coefficients和scale，候选τ
  仍只通过P109解析boundary projection进入；P81/P96 consumed development，不读取P113；
- prevention：不扫coefficient count/architecture/loss/seed/projection/coverage；若失败登记`V67-F80`，P113若随后失败顺延F81。

### V67-F80 — 低频spectral Actor sequence在P96过度平滑boundary-relevant residual

- 分类：`scientific/actor-uncertainty-representation`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P115-SPECTRAL-ACTOR-UNCERTAINTY-01/20260830T071500Z__spectral-actor-uncertainty-s0-r1`；
- symptom：P81 spectral AUROC `.977092`较P109 `.967639`提高`.009453`且均0 selected events；P96却从P109的
  0 events/`.904345`退化到7 events/`.847123`，三项冻结decision全失败；
- interpretation：前4个DCT modes提供低频trajectory coherence，但source/P81收益不能迁移到P96；截断会抹掉影响局部
  occupancy boundary crossing的末端或高频残差。joint/spectral representation并不自动等于更可靠的task ranking；
- literature response：Joint Metrics Matter与PRECOG说明联合future重要，FoSS说明频域结构可建模长时依赖；本结果限定了其在
  当前constant-velocity residual UQ上的直接低频迁移，不能借论文动机忽略独立development反转；
- resolution：不扫DCT coefficient count、hidden width、loss、seed或projection；P115不进入P113或未来confirmation，保留
  P109 pointwise directional Gaussian；
- claim impact：无spectral/joint uncertainty或collision claim；P108/P109/P110证据不变。

下一可用编号：`V67-F81`。

### P116 freeze note — 用directional q90 field替代Gaussian分布假设

- card point：P115表明低频joint Gaussian在P81/P96方向反转，禁止扫DCT coefficient/structure；
- literature response：AISTATS 2022 Multivariate Quantile Function Forecaster直接参数化多元quantile，NeurIPS 2021强调full
  quantile function可表达distribution-free uncertainty；当前迁移只取固定q90与8 directions，不声称完整quantile function；
- method：Actor-time residual沿固定unit direction作pinball监督，推理时boundary normal只作为directional query，clearance只在
  解析ratio中出现；因此不回到raw end-to-end query classifier；
- prevention：P81/P96 consumed development，不读P113；不扫direction count/quantile/model/loss/seed/coverage。失败登记
  `V67-F81`并关闭该形式，P113若随后失败使用F82。

下一可用编号仍为：`V67-F81`。

### V67-F81 — distribution-free directional q90仍不及Gaussian standardized crossing margin

- 分类：`scientific/actor-uncertainty-distribution`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P116-DIRECTIONAL-QUANTILE-FIELD-01/20260830T072000Z__directional-quantile-field-s0-r1`；
- symptom：P81 q90/P109 AUROC=`.963841/.967639`且均0 selected events；P96 q90/P109=`.889318/.904345`，
  selected events=`6/0`。两cohort AUROC gain均负，三项冻结decision全失败；
- interpretation：直接预测adverse residual q90移除了Gaussian shape假设，却也丢失P109 mean与scale组成的standardized margin；
  `q90/clearance`在当前source跨域不足，不能因distribution-free动机包装为更可靠；
- literature response：multivariate quantile function与quantile-UQ工作支持灵活分布建模，但本项目固定8-direction/q90的窄迁移
  没有超过Gaussian；这是否定当前实现对象，不是否定一般quantile方法；
- resolution：不扫direction count、quantile、network、loss、seed或coverage；与P114/P115一起关闭P109替代model family；
- claim impact：无directional quantile/conformal/collision probability claim；P109/P113保持唯一冻结主线。

下一可用编号：`V67-F82`。

### V67-F82 — scene-0003 session跨public archive part导致冻结locator不完整

- 分类：`engineering/data-locator`；状态：`resolved_pre_target_exact_locator_recovery`；
- failed run：`run://worldsim_v67/WS-V67-P113-DIRECTIONAL-VS-CLEARANCE-CONFIRMATION-PREP-01/
  20260830T070000Z__directional-vs-clearance-prep-s0-r1`；
- symptom：初始六shard extraction得到其余9 scenes的3,517 files，但scene-0003的384个LIDAR members全部缺失；runner在
  preprocess、target row materialization和任何metric前按exact extraction合同退出，P113 evaluator保持等待；
- root cause：scene-0003与P81 scene-0344共享session，但先前scene-0344命中的shard04不能推出同session早期scene所在part；
  nuScenes公开10-part archive可跨session切分，官方devkit只要求合并全部parts且不提供session→part index；
- recovery：不换scene/model/decision。完整排除02/03/05/06/08/09，在01命中384/384并直接提取，停止尚未完成的07/10；
  locator期间先预处理9个ready scenes。仅修正`scene-0003:04→01`，prep r2映射3,894/3,894、复用9 scenes并在60.63s
  完成scene-0003，总wall71.41s。无hash/checksum/fingerprint或额外quality gate；
- claim impact：纯pre-target engineering failure，不影响P113冻结scientific protocol，也不产生任何confirmation result。

下一可用编号：`V67-F83`。

### P117 positive mechanism note — full bivariate covariance improves consumed directional ranking

- card point：P114--P116三种替代均未超过P109，但P109的diagonal Gaussian仍强制纵/横Actor residual条件独立；
- literature response：CVPR 2023 IPCC-TP显式学习joint Gaussian means/covariances，支持把相关结构作为独立变量，而不是继续扩大
  raw query classifier或扫downstream aggregation；
- frozen migration：P117仅新增一个bounded correlation输出和完整bivariate Gaussian NLL，source/features/network width/
  optimizer/steps/seed/boundary projection/coverage均沿用P109，只读consumed P81/P96且不读P113；
- result：P81/P96都保持0 selected events，AUROC相对P109分别`+.004903/+.009320`，平均`+.007111`通过`.005`门；
  verdict=`supported_development_correlated_actor_uncertainty`；
- claim/prevention：该结果不是failure，也不占用failure ID；不扫rho bound/loss/width/seed/projection。它只能作为未来全新
  target-unread cohort的候选，不能在P113 target已冻结后替换primary或改变decision。

下一可用编号仍为：`V67-F83`。

### V67-F83 — conditional rho推理项未解释P117 full-covariance训练收益

- 分类：`scientific/mechanism-ablation`；状态：`closed_negative_after_single_ablation`；
- canonical：`run://worldsim_v67/WS-V67-P118-CORRELATION-ABLATION-01/20260830T073000Z__correlation-ablation-s0-r1`；
- frozen comparison：同一P117 checkpoint、mean/scale/rows/projection/fixed50，只比较conditional rho与rho=0；不训练、不refit；
- symptom：P81 AUROC gain仅`+.000304`，P96为`-.000115`，平均`+.000094 < .003`；两cohort event count虽均为0，
  但两cohort AUROC正增益门和mean-gain门失败；
- interpretation：P117相对P109的开发增益来自完整bivariate likelihood训练package，不能定位为inference-time conditional
  correlation项的直接贡献；mean/scale在joint NLL下的重塑是未分离因素；
- resolution：不扫rho bound、loss、seed或重复训练，不用事后门解释P117。保留P117为完整package的consumed-development候选，
  但论文明确报告P118 negative mechanism；不影响P113冻结P109 primary。

下一可用编号：`V67-F84`。

### V67-F84 — independent AUROC增量未转化为fixed50事件优势

- 分类：`scientific/selective-tail-transfer`；状态：`closed_negative_after_one_shot_confirmation`；
- canonical：`run://worldsim_v67/WS-V67-P113-DIRECTIONAL-VS-CLEARANCE-CONFIRMATION-01/
  20260830T070500Z__directional-vs-clearance-s0-r1`；
- frozen read：P109 checkpoint/normalization/linear projection、`.05m` clearance、H3.5、time/Actor max、per-scene fixed50与
  10-scene cohort全在target前冻结；只比较directional和clearance两门；
- symptom：7,206 rows/1,525 trajectories/79 flips上，directional AUROC `.920155`较clearance `.875291`提高`.044864`
  并通过`.02`门，但selected events=`6 vs 5`，严格noninferiority门失败；1/2 decisions；
- interpretation：learned Actor uncertainty改善全局排序，却没有稳定控制固定coverage处的rare-event tail；强clearance geometry
  在单一operating point仍可更好。AUROC与fixed-budget selective risk不可互相替代；
- literature response：NeurIPS 2022 partial-AUC optimization与fixed-coverage selective prediction工作说明全局AUC可掩盖相关
  FPR/coverage区间。本结果支持下一路线直接研究ranked range/selective tail objective，而不是继续堆Gaussian/quantile结构；
- resolution：关闭当前P109 uncertainty-over-clearance claim；不降coverage/floor/gate，不换P113 cohort，不在本read上试P117，
  不做第二P113 recovery。P108相对Actor/P75的scene-level factorization primary保留；
- claim impact：可写independent AUROC ranking gain，但必须同时写composite verdict rejected；无calibrated probability、collision、
  planner、policy、closed-loop或safety claim。

下一可用编号：`V67-F85`。

### V67-F85 — source ranked-range supervision未改变跨cohort fixed50 tail ordering

- 分类：`scientific/selective-tail-objective`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P119-RANKED-RANGE-TAIL-01/20260830T074500Z__ranked-range-tail-s0-r1`；
- method：冻结P109 distribution/crossing score，只在source per-scene `.35--.65` operating range训练bounded hidden32 residual；
  79,478 trajectories中ranked-range positives仅65，optimizer不读P81/P96/P113；
- symptom：P81/P96保持0 events，但P113仍为6，未达到clearance limit5；三个consumed cohorts AUROC相对P109均下降
  `.003836/.004586/.001217`；
- interpretation：binary occupancy flip在source fixed50邻域极稀疏，局部pairwise loss没有足够稳定的跨scene order signal；
  partial-AUC动机不能替代实际迁移结果；
- resolution：不扫percentile band、residual bound、head、loss、seed或coverage；关闭binary fixed50 recovery。下一对象改为
  连续的τ-conditioned boundary-normal state cost/selective regression，保留P113 negative；
- claim impact：无ranked-range/partial-AUC/selective-risk improvement claim，无P113 recovery或safety claim。

下一可用编号：`V67-F86`。

### V67-F86 — continuous cost regressor未超过冻结P109 sufficient score

- 分类：`scientific/continuous-selective-regression`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P120-CONTINUOUS-BOUNDARY-STATE-COST-01/
  20260830T075000Z__continuous-boundary-state-cost-s0-r1`；
- new object：Actor residual沿candidate τ boundary normal的absolute projection除以predicted absolute clearance，再作trajectory max；
  它是τ-conditioned continuous reliability cost，不是旧endpoint error或binary flip；
- symptom：learned/P109 selected cost在P81=`.2032/.1863`、P96=`.1850/.1788`、P113=`.2237/.2247`；learned
  Spearman相对P109 gain=`-.0123/-.0515/+.0173`，两项冻结decision全失败；
- interpretation：top16+clearance回归头在source拟合continuous target，但没有超过P109 standardized crossing score；P109本身已是
  跨三个consumed cohorts的强低容量 sufficient ranking statistic；
- retained evidence：P109 continuous-cost Spearman=`.8065/.7183/.7921`，fixed50 cost reduction=
  `89.75%/77.05%/83.37%`，明显高于clearance-only ranking；这只支持冻结P121候选，不把P120写成成功；
- resolution：不扫cost definition/floor/head/loss/seed/coverage，不训练第二regressor。P121在全新target-unread cohort只确认冻结
  P109 continuous object；
- claim impact：无learned continuous head或independent continuous reliability claim，仍无collision/safety claim。

下一可用编号：`V67-F87`。

### V67-F87 — full-covariance排序增量未保证continuous fixed50 cost nonregression

- 分类：`scientific/full-covariance-continuous-selection`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P122-FULL-COVARIANCE-CONTINUOUS-SELECTION-01/
  20260830T081000Z__full-covariance-continuous-selection-s0-r1`；
- protocol：P121 target仍未物化时，只读已消费P81/P96/P113；冻结P117/P109 checkpoints、continuous cost、`.05m` floor与
  fixed50，不训练、不refit。只有三cohort selected cost均不回退且mean Spearman gain≥`.005`才可成为P121 secondary；
- symptom：full covariance的Spearman gain=`+.011488/+.004767/+.011976`，均值`.009410`通过；但selected cost在P96
  `.178783→.184867`、P113 `.224742→.225542`，故nonregression失败；
- interpretation：P117 full-likelihood训练带来的整体排序改善仍会移动fixed50边界，global continuous rank gain不足以保证
  operating-point cost不退化；与P118一起说明不能把收益简化为conditional rho推理项；
- resolution：不把P117追加到P121同读、不扫rho/score/coverage、不训练combiner；P121 primary继续只用冻结P109；
- claim impact：无full-covariance continuous-selection或独立迁移claim；P117 consumed AUROC mechanism support仍保留。

下一可用编号：`V67-F88`。

### V67-F88 — continuous operating-range rank residual仍产生跨cohort排序漂移

- 分类：`scientific/continuous-selective-ranking`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P123-CONTINUOUS-RANK-RESIDUAL-01/
  20260830T081500Z__continuous-rank-residual-s0-r1`；
- method：在source每scene的P109 score `.25--.75` operating band内，以continuous cost percentile `<=.35/>=.65`构造
  13,123个within-scene pairs；hidden32 bounded residual、6,000 steps，0 development/P121 target optimizer read；
- symptom：P81/P96/P113 Spearman gain=`-.019849/-.056155/+.008165`；P96 selected cost由`.178783`退化至`.183085`，
  所以no-regression与mean-rank-gain两项均失败；
- interpretation：稠密continuous target消除了P119的binary positive scarcity，但source operating-range residual仍改变已较强的
  P109 sufficient ordering并跨cohort漂移；问题不是简单换成continuous pair label即可解决；
- resolution：不扫band、bound、architecture、loss、seed或coverage；关闭P119/P120/P123 downstream head family，不创建P121
  secondary。下一机制若继续，必须改变Actor residual distribution而不是再接selection head；
- claim impact：无continuous pairwise/selective-ranking improvement claim；P121冻结P109 primary不变。

下一可用编号：`V67-F89`。

### V67-F89 — 固定重尾Student-t Actor residual在P96显著过宽

- 分类：`scientific/heavy-tailed-actor-uncertainty`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P124-CORRELATED-STUDENT-T-UNCERTAINTY-01/
  20260830T082000Z__correlated-student-t-uncertainty-s0-r1`；
- literature response：NeurIPS Student-t robust regression支持以重尾likelihood降低outlier影响；CVPR long-tail trajectory工作支持
  直接建模tail distribution。迁移时只把P117 Gaussian NLL改为固定`df=4` multivariate Student-t；
- symptom：P81/P96/P113 AUROC gain=`+.000248/-.054071/-.005427`，fixed50 events=`0/7/7`，相对P109的`0/0/6`
  在P96/P113均退化；两项decision失败；
- interpretation：统一重尾分布把outlier与结构性motion modes混合吸收，训练likelihood更低但boundary-relevant ranking在P96明显
  过宽；NLL改善不能替代跨cohort selective evidence；
- resolution：不扫degrees-of-freedom、scale floor、correlation、loss、seed或coverage；关闭Student-t family。若继续Actor
  distribution，只允许固定低组件数的显式multimodal residual，而不是调重尾参数；
- claim impact：无heavy-tailed Actor uncertainty improvement、P121 secondary或safety claim。

下一可用编号：`V67-F90`。

### V67-F90 — 两组件mixture modes未形成可迁移的boundary-relevant分解

- 分类：`scientific/multimodal-actor-uncertainty`；状态：`closed_negative_after_single_trial`；
- canonical：`run://worldsim_v67/WS-V67-P125-TWO-MODE-ACTOR-UNCERTAINTY-01/
  20260830T082500Z__two-mode-actor-uncertainty-s0-r1`；
- literature response：CVPR/ICCV multimodal trajectory工作支持显式mode specialization；CVPR 2022同时指出GMM hard-to-optimize/
  overfit风险。迁移只固定K=2，并用mixture-weighted boundary CDF，不扫组件数；
- symptom：mean maximum component weights约`.79--.82`，非完全collapse，但P81/P96/P113 AUROC gain=
  `-.002116/-.024330/-.006968`，fixed50 events=`0/4/7`也劣于P109=`0/0/6`；
- interpretation：组件确实分工，但source中形成的modes不等于跨cohort candidate-τ boundary modes；更低mixture NLL不能保证
  task-conditioned selection。该失败与F89共同关闭单模型tail/multimodal output扩展；
- resolution：不扫K、entropy regularizer、scale floor、projection、seed或coverage；下一步若继续不确定性机制，只允许用独立
  模型间disagreement分解epistemic/aleatoric，而不是再改同一输出分布；
- claim impact：无GMM/multimodal Actor uncertainty improvement、P121 secondary或safety claim。

下一可用编号：`V67-F91`。

### V67-F91 — ensemble全局AUROC一致提升但P96 fixed50出现一个事件

- 分类：`scientific/epistemic-aleatoric-ensemble`；状态：`closed_negative_binary_composite`；
- canonical：`run://worldsim_v67/WS-V67-P126-ACTOR-DEEP-ENSEMBLE-01/
  20260830T083000Z__actor-deep-ensemble-s0-r1`；
- literature response：NeurIPS deep ensembles与ICML uncertainty decomposition支持用独立成员mean disagreement补充aleatoric variance；
  P126固定三成员，复用seed0、只新训seed1/2，law-of-total-variance无可调权重；
- symptom：P81/P96/P113 AUROC gain=`+.001968/+.010012/+.006259`且mean `.006080`通过，P113 events `6→4`；但P96
  events `0→1`，因此event noninferiority失败；
- interpretation：约2.2--2.6%的projected epistemic fraction提供一致global ordering信息，却仍会移动rare-event fixed50边界；
  该结果不能写成binary selective success，但与此前单模型输出族全退化不同；
- resolution：binary composite严格拒绝；不删P96、不调member/seed/epistemic weight。因P121已预注册continuous endpoint，只允许
  P127在同一consumed rows上按事前continuous nonregression/rank decisions一次迁移；
- claim impact：无binary ensemble/fixed50 independent claim；P121 primary不变。

下一可用编号：`V67-F92`。

### P127 freeze note — ensemble continuous-cost迁移

- candidate：冻结P126三成员与total-variance score；P109、continuous cost、`.05m` floor、fixed50均不变；
- role：P81/P96/P113 consumed development；P121 target未物化，P127 optimizer/read均不接触P121；
- decisions：selected continuous cost三cohort全不回退；mean Spearman gain≥`.005`；
- prevention：失败登记F92并关闭ensemble，不扫member/weight/score/cost/coverage；成功才可在P121 rows前另冻same-read secondary。

### P127 outcome / P128 freeze note — continuous迁移成立并冻结same-read secondary

- P127 consumed result：三cohort selected cost全低于P109；Spearman gain=`+.046981/+.135078/+.075514`，两门均通过；
- P128在P121 rows物化前冻结，直接复用P121 primary同一NPZ；P121仍是唯一primary；
- P128 decisions：Spearman gain over P109≥`.005`且selected cost≤P109；失败使用F92并关闭ensemble，不做第二secondary/
  member/weight/score recovery；
- failure delta：P127无新增failure；下一可用编号仍为F92。

### P121/P128 outcome note — independent primary与same-read ensemble增量均成立

- P121 primary：P109 Spearman `.761472`、超过clearance `+.288235`；fixed50 cost reduction `77.36%`且cost低于clearance，
  2/2 decisions，scene-level independent support；
- P128 secondary：ensemble relative P109 Spearman gain `+.047211`、selected cost `.270506<=.277957`，2/2 decisions；
- timing disclosure：08:34:24确认rows absent后P128内容冻结并复制，rows在Git commit guard前的传输窗口内物化；commit
  `572f7d5`因此晚于materialization。内容在读取outcome前冻结且未改，但论文只称prospective-content same-read secondary；
- resolution：无scientific failure，不把P128冒充commit-before-read prereg或独立cohort；下一可用failure仍为F92。

### P129 freeze note — ensemble continuous increment独立确认

- metadata-only cohort：`0017/0345/0962/0095/0522/0625/0798/0921/0927/1063`；target-unread、四location 3/3/3/1、
  内部10 distinct log sessions；只称scene-level independent；
- candidate/decisions：冻结P126 total variance、P109、continuous cost/floor/H3.5/fixed50；Spearman gain≥`.005`且selected
  cost≤P109；
- prevention：只允许target前exact locator correction；scientific failure使用当时下一可用编号并关闭ensemble independent increment，不换
  scene/member/weight/score/cost/coverage/gate或第二cohort。
- outcome：11,406 rows/1,681 trajectories；ensemble relative P109 Spearman gain=`+.042572`、selected cost
  `.308669<.329340`，2/2 decisions。无scientific failure；支持scene-level independent ensemble increment，claim不外推到
  session-level/collision/calibrated probability/closed-loop/safety。

### V67-F92 — P129异步evaluator在错误工作目录解析相对入口

- 分类：`engineering/launcher-entry`；状态：`resolved_pre_run_pre_target`。
- 观察：首次waiting evaluator由含后台async list的shell命令启动，实际从`/root`寻找`./scripts/...p129...py`并立即退出；
  没有创建primary run leaf、没有读取P129 rows、没有计算metric。独立prep及其7个shard workers持续正常运行。
- 根因/外部检索：GNU Bash官方manual说明`&`形成asynchronous list，工作目录/grouping边界需显式控制；组合命令没有把
  预期`cd`可靠约束到该后台evaluator入口。
- 恢复：只用absolute script/config/PYTHONPATH与`setsid`重启相同canonical evaluator；cohort、checkpoints、score、cost、
  coverage与decisions全部不变。
- 防重复：长IO waiting evaluator使用absolute entry；不为此增加smoke/regression matrix。下一可用编号=`V67-F93`。

### P130 freeze note — ensemble moment distribution distillation

- motivation：P126/P127/P128表明ensemble total covariance带来连续排序增量，但三模型推理成本可压缩；P129 archive IO期间
  允许只访问source/consumed development的GPU研究。
- method：冻结P126三成员，按law of total covariance形成单个full-cov Gaussian teacher；P117结构student以闭式Gaussian KL
  训练6,000 steps。只做seed0单trial，不扫蒸馏loss、权重、结构或coverage。
- decisions/outcome：P81/P96/P113 selected cost逐cohort不劣于P126，mean Spearman difference≥`-.005`；实际mean
  Spearman difference=`-.002024`通过，但P113 cost `.225324>.218791`，故single-student moment方案拒绝，不改P129。
- references：UAI 2022/2023 ensemble distribution distillation、NeurIPS 2022 functional ensemble distillation。

### V67-F93 — moment-matched Gaussian蒸馏未保持P113 selection boundary

- 分类：`algorithm/distillation-object`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P130-ENSEMBLE-DISTRIBUTION-DISTILLATION-01/
  20260830T091000Z__ensemble-distribution-distillation-s0-r1`。
- 观察：final Gaussian KL=`.073871`；三cohort mean Spearman difference=`-.002024`满足retention，P81/P96 selected cost改善，
  但P113 `.225324`高于P126 `.218791`，因此cost nonregression失败。
- 解释：全局moment KL可保留整体排序，却没有优先保存fixed50 decision boundary附近的teacher function；不是GPU/容量故障。
- 防重复：不扫KL权重、student width/depth或seed；后续P131改变预测对象为task-conditioned boundary score functional
  distillation，而不是P130调参恢复。

### P131 freeze note — task-conditioned functional score distillation

- method：冻结P126 row boundary score作为teacher，单query MLP读取既有query features、signed-clearance profile和boundary
  normals，以一次6,000-step Smooth-L1直接拟合teacher function；只用source和consumed P81/P96/P113。
- decisions/outcome：相对P126三cohort selected cost nonregression与mean Spearman difference≥`-.005`；实际mean Spearman
  difference=`-.362629`且三组cost全退化，两门全失败；P129 rows隔离。
- prevention：seed/architecture/loss/input/coverage一次冻结；关闭direct pointwise functional student，不做蒸馏sweep。

### V67-F94 — pointwise teacher-score回归未保持trajectory max排序

- 分类：`algorithm/supervision-granularity`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P131-TASK-CONDITIONED-SCORE-DISTILLATION-01/
  20260830T091500Z__task-conditioned-score-distillation-s0-r1`。
- 观察：row Smooth-L1降至`.006349`，但trajectory-max三cohort Spearman=`.380515/.436117/.669918`，mean relative
  ensemble=`-.362629`；selected cost三组全面退化，0/2 decisions。
- 根因：source中大量普通row主导pointwise loss，小tail误差经max放大并改变每条trajectory的代表row；训练层级与部署聚合错配。
- 防重复：不扫Huber/temperature/width/seed或加权tail；P132把trajectory max放入训练图并直接优化同scene pair ordering。

### P132 freeze note — trajectory-max rank distillation

- method：P131相同inputs/MLP，但先对trajectory rows取student max，再用同source scene内uniform pairs与P126 teacher order做
  pairwise logistic；6,000 steps、pair batch4096、seed0一次，无temperature/top-k/pointwise auxiliary。
- decisions/outcome：相对P126三cohort selected cost nonregression、mean Spearman difference≥`-.005`；实际mean Spearman
  difference=`-.020183`且三组cost均回退，两门全失败；P129 rows隔离。
- prevention：关闭single-query distillation family，不做ranking-loss sweep。

### V67-F95 — aggregation-aligned rank student仍未保留deep-ensemble增量

- 分类：`algorithm/teacher-compression-capacity`；状态：`closed_negative_after_aggregation_aligned_trial`。
- canonical：`run://worldsim_v67/WS-V67-P132-TRAJECTORY-RANK-DISTILLATION-01/
  20260830T092000Z__trajectory-rank-distillation-s0-r1`。
- 观察：pairwise logistic降至`.117696`，三cohort Spearman恢复到`.829--.851`，远好于P131；但相对P126 mean仍
  `-.020183`，selected cost三组全回退，0/2 decisions。
- 解释：监督层级修复了pointwise→max错配，却无法让单query function重建independent Actor member disagreement；真正增量
  更可能需要原生多成员表示，而非teacher score近似。
- 防重复：P130 moment、P131 pointwise function、P132 trajectory ranking三种compression object均关闭；不扫loss/temperature/
  width/seed。P133转向一次native BatchEnsemble。

### P133 freeze note — rank-one native efficient ensemble

- method：3个BatchEnsemble members共享每层weight，保留member-specific rank-one input/output factors与独立bootstrap；
  diagonal Gaussian NLL、total variance projection、source与continuous evaluation均继承P126/P127。
- decisions/outcome：相对P126三cohort selected cost nonregression、mean Spearman difference≥`-.005`；实际mean Spearman
  difference=`-.014545`，P81/P113 cost回退，两门全失败；P129 rows隔离。
- prevention：固定3 members/seed0/factor init/6,000 steps，不扫rank/member/structure；rank-one shared-weight trial关闭。

### V67-F96 — BatchEnsemble shared weights压缩了epistemic diversity

- 分类：`algorithm/embedded-ensemble-diversity`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P133-BATCHENSEMBLE-ACTOR-UNCERTAINTY-01/
  20260830T092500Z__batchensemble-actor-uncertainty-s0-r1`。
- 观察：NLL=`-3.69116`良好，但三cohort projected epistemic fraction仅`.13%--.34%`，明显低于P126的约`2.2%--2.6%`；
  mean Spearman difference=`-.014545`，P81/P113 selected cost回退。
- 外部证据/解释：2026 controlled BatchEnsemble study报告rank-one members在function/parameter space近似相同；本结果同向，
  表明shared-weight collective regime而非训练不足。
- 防重复：不扫factor init/rank/member/seed；P134只作一次独立block packed regime，若仍失败则不再追efficient ensemble结构。

### P134 freeze note — packed independent member blocks

- method：3套独立weights/biases用member batched kernels在单graph执行，各自bootstrap；不共享P133 weights/factors，保留
  P126三成员容量与FLOPs。
- decisions/outcome：相对P126三cohort selected cost nonregression、mean Spearman difference≥`-.005`；实际rank mean=`+.001874`
  通过，P81/P113 cost改善，但P96 `.172184>.167572`，cost gate失败；P129 rows隔离。
- prevention：不扫packing/group/width；只允许P135一次P126 per-member compute-parity recovery。

### V67-F97 — reduced per-member packed budget未保持P96 selection cost

- 分类：`algorithm-compute/ensemble-training-budget`；状态：`active_single_compute_parity_recovery`。
- canonical：`run://worldsim_v67/WS-V67-P134-PACKED-INDEPENDENT-ACTOR-ENSEMBLE-01/
  20260830T093000Z__packed-independent-actor-ensemble-s0-r1`。
- 观察：independent blocks恢复epistemic fraction到`1.46%--2.23%`，mean rank delta=`+.001874`且P81/P113 cost改善；
  唯一失败为P96 cost `.172184>.167572`。final NLL=`-3.56197`弱于P126 members。
- 预算根因候选：P134为了aggregate batch parity设每member 21,845，而P126每member 65,536；6,000 steps下每member sample
  exposure只有1/3，不能把剩余差异直接归因于packed representation。
- 唯一恢复：P135仅把member batch改到65,536做compute parity；结构/seed/steps/score/decisions不变。不论结果不再扫budget。

### P135 freeze note — full-budget packed compute parity

- method：exact P134 runner与三独立blocks；唯一变化member batch `21845→65536`，匹配P126 per-member 6,000-step exposure。
- decisions/outcome：仍为三cohort cost nonregression与mean Spearman difference≥`-.005`；实际mean rank=`-.001273`通过，
  但三组cost均略回退；P129 rows隔离。
- prevention：packed route关闭；不再扫batch/steps/member/seed/packing。

### V67-F98 — full per-member compute仍未保持P126 fixed50 cost boundary

- 分类：`algorithm/ensemble-solution-variance`；状态：`closed_negative_after_compute_parity`。
- canonical：`run://worldsim_v67/WS-V67-P135-FULL-BUDGET-PACKED-ACTOR-ENSEMBLE-01/
  20260830T093500Z__full-budget-packed-actor-ensemble-s0-r1`。
- 观察：compute parity后NLL=`-3.62334`、epistemic fraction=`1.51%--2.63%`、mean rank difference=`-.001273`均接近
  P126；但P81/P96/P113 selected cost分别高`.003705/.001170/.004970`，cost gate全回退。
- 解释：P134的主要rank差异确受budget影响，但fixed50 selection对independent solution/seed边界敏感；不能把near-parity
  重写成noninferiority成功。
- 防重复：不继续packed seed/budget/member sweep。P136换成single-path cyclic snapshots，检验低成本posterior path samples。

### P136 freeze note — cyclic snapshot Actor ensemble

- method：一个P109结构训练6,000 steps，3个固定2,000-step cosine cycles，LR `.001→.00001`，只存2000/4000/6000；
  三snapshot按total variance评分。
- decisions/outcome：相对P126三cohort selected cost nonregression、mean Spearman difference≥`-.005`；实际mean rank=
  `-.008549`，P96/P113 cost回退，两门全失败；P129 rows隔离。
- prevention：固定cycles/LR/snapshot/seed；snapshot first trial关闭，不扫schedule。

### V67-F99 — cyclic snapshots未形成可迁移的P96 functional diversity

- 分类：`algorithm/single-path-mode-diversity`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P136-SNAPSHOT-ACTOR-ENSEMBLE-01/
  20260830T094000Z__snapshot-actor-ensemble-s0-r1`。
- 观察：三个cycle NLL逐步改善；P81 cost/rank优于P126，但P96 rank差`-.026964`且P96/P113 costs回退，mean rank=
  `-.008549`，0/2 decisions。
- 解释：同一路径cycle endpoints未覆盖独立seed的function modes；最后snapshot本身也尚弱于P126 members。
- 防重复：不扫cycle length/LR range/snapshot count。P137改为SWAG covariance而非挑snapshot。

### P137 freeze note — low-rank-plus-diagonal weight posterior

- method：单P109路径6,000 steps；4000后固定LR `.0001`、每100 steps收集20 iterates；拟合SWAG diag+low-rank
  covariance并以seed137采3 models。
- decisions/outcome：相对P126三cohort selected cost nonregression、mean Spearman difference≥`-.005`；实际mean rank=
  `+.002315`通过，P113 cost改善，但P81/P96微回退；P129 rows隔离。
- prevention：固定collection window/LR/rank/sample count/seed；single-path posterior route关闭。

### V67-F100 — SWAG近似rank保留但未满足逐cohort cost nonregression

- 分类：`algorithm/approximate-posterior-boundary`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P137-SWAG-ACTOR-ENSEMBLE-01/
  20260830T094500Z__swag-actor-ensemble-s0-r1`。
- 观察：20 iterates/3 samples使mean rank delta=`+.002315`，P113 cost改善；但P81/P96 costs分别高`.002115/.000274`，
  strict nonregression失败。
- 解释：low-rank weight posterior比cycle snapshots更接近P126，但fixed sampling仍不能复制每cohort selection boundary；
  不是用更多samples/调collection LR的授权理由。
- 防重复：不扫posterior scale/rank/sample/seed。P138改变uncertainty family为full-cov aleatoric+epistemic deep ensemble。

### P138 freeze note — full-covariance deep ensemble

- method：P117 seed0 + 同协议seed1/2；total projected covariance=mean member full covariance + variance member means。
- decisions/outcome：相对P126三cohort selected cost nonregression且mean Spearman gain≥`.005`；实际mean gain=`+.003590`，
  P96 rank/cost回退，两门全失败；P129 rows隔离。
- prevention：不扫correlation/member/weight/projection；full-cov ensemble first trial关闭。

### V67-F101 — full-cov aleatoric+epistemic ensemble仍在P96反转

- 分类：`algorithm/covariance-transfer`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P138-FULL-COVARIANCE-DEEP-ENSEMBLE-01/
  20260830T095000Z__full-covariance-deep-ensemble-s0-r1`。
- 观察：P81/P113 rank/cost改善，P96 rank gain=`-.004597`、cost `.170009>.167572`；mean gain=`+.003590<.005`，
  0/2 decisions。新member NLL显著收敛，不是训练入口失败。
- 解释：local XY correlation对不同cohort方向不一致；within-member structured covariance不能自动解决source weighting/transfer。
- 防重复：不扫correlation parameterization/weight/member/seed。P139保持diagonal P126结构，仅改uniform scene sampling。

### P139 freeze note — uniform source-scene sampling

- method：三diagonal members完全匹配P126，只把global token-uniform改为scene-uniform→token-uniform；scene不是semantic domain label。
- decisions/outcome：相对P126三cohort selected cost nonregression、mean Spearman gain≥`.005`；P129 rows隔离。实际三cohort
  cost全回退，Spearman gain=`-.009921/-.013609/-.014711`，mean=`-.012747`，两门全失败。
- prevention：不加GroupDRO/Fishr penalty、不扫scene weights/subsets；失败登记F102并关闭simple balancing route。

### V67-F102 — uniform source-scene sampling一致削弱continuous ordering

- 分类：`algorithm/source-sampling`；状态：`closed_negative_after_first_trial`。
- canonical：`run://worldsim_v67/WS-V67-P139-SCENE-BALANCED-DEEP-ENSEMBLE-01/
  20260830T095500Z__scene-balanced-deep-ensemble-s0-r1`。
- 观察：P81/P96/P113 selected cost=`.180687/.173277/.232300`均高于P126；mean Spearman gain=`-.012747`。
  三member NLL均收敛，失败不是launcher或训练中断。
- 解释：对小scene过采样改变了Actor residual训练分布，但没有提供semantic group信息，反而丢失自然token distribution的有效统计。
- 防重复：关闭uniform scene weighting，不扫scene权重/penalty。P140只迁移bootstrap unit以增加member diversity，并恢复scene内
  natural token weighting。

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

### P242 milestone note — L1目标对齐恢复连续预算compiler，无新增failure

- canonical：`run://worldsim_v67/WS-V67-P242-L1-INTEGRATED-MONOTONE-BUDGET-SURFACE-01/20260831T053000Z__l1-integrated-monotone-budget-surface-s0-r1`；
- 结果：P201 surface/final MAE=`.007271/.009869`，quality composite通过，budget/horizon violations=`0/0`，3/3；
- 因果边界：相对P241唯一科学变化是MSE改L1；没有width/quadrature/budget/step/seed sweep；
- next：P243全新10-scene/10-log cohort已在read前冻结并启动；P244只在其IO期间训练解析rate-spline successor，
  不读取P243 rows或quality。

下一可用编号仍为：`V67-F187`。

### P246 milestone note — 有限两侧budget扩展通过，无新增failure

- canonical：`run://worldsim_v67/WS-V67-P246-EXTENDED-BUDGET-RATE-SPLINE-01/20260831T070000Z__extended-budget-rate-spline-s0-r1`；
- 结果：P201八budget surface/final MAE=`.006580/.008776`，quality composite通过、双轴violations=`0/0`，3/3；
- 边界：只支持`.025--6.4`有限区间，不推断无界tail或formal calibration；没有range/knot/point-count sweep；
- next：P247在fresh rows前冻结；P248改变对象为reliability-level-conditioned inverse budget，不继续range优化。

下一可用编号仍为：`V67-F187`。

### V67-F187 — inverse budget精度通过但未保持forward probability fidelity

- canonical：`run://worldsim_v67/WS-V67-P248-INVERSE-RELIABILITY-BUDGET-COMPILER-01/20260831T073000Z__inverse-reliability-budget-compiler-s0-r1`；
- 观察：P201 inverse normalized-log-budget MAE=`.019881<=.075`，但在冻结P246重构的probability MAE=
  `.018624>.015`；1/2；source/P183重构MAE也为`.016876/.017074`；level/horizon violations=`0/0`；
- 解释：P201约`15.06%` target在budget下界裁剪，且冻结P246的局部CDF斜率不均；单纯平均budget L1会在平坦区和
  陡峭区等权，因此小坐标误差不能保证response误差；
- literature/open-source response：tandem neural-network inverse design冻结forward surrogate，并以重构response约束inverse；
  NeurIPS 2020 PCGrad在多目标梯度冲突时作投影。P249将冻结P246 cycle probability loss迁入训练，同时保留budget L1；
- response：P249只作一次tandem recovery；cycle gradient冲突时投影并norm-match primary，不扫人工权重；P248结构、
  levels/data/steps/seed/decisions不变；
- 防重复：不放宽`.015`、不增加bisection steps或level/knot/width sweep，不把inverse-budget pass包装为整体成功。

下一可用编号：`V67-F188`。

### P250 freeze note — inverse compiler prospective secondary，无新增failure

- P250在P243 rows和P249 outcome出现前冻结，仅等待两个atomic artifacts；
- 复用P243同一次fresh read，P249/P246/levels/MC/两门均不变，不把它表述为第二个独立cohort；
- 不因P249 development结果改变evaluator或gate；当前下一可用failure id保持`V67-F188`。

下一可用编号仍为：`V67-F188`。

### V67-F188 — PCGrad tandem混合目标只带来微小response改善

- canonical：`run://worldsim_v67/WS-V67-P249-TANDEM-CYCLE-INVERSE-BUDGET-01/20260831T080000Z__tandem-cycle-inverse-budget-s0-r1`；
- 观察：P201 inverse-budget MAE=`.019667`通过，但重构probability MAE=`.018257>.015`；相对P248仅改善
  `.000367`；source/P183重构=`.016490/.016759`；level/horizon violations=`0/0`；
- diagnosis：12,000 steps只有155次budget/cycle gradient conflict；PCGrad很少触发，norm-matched budget primary仍让模型
  主要拟合坐标误差，不能充分强调冻结forward的陡峭response区；
- literature/open-source response：经典tandem inverse design及其PyTorch实现把冻结forward surrogate直接作为唯一可微
  response loss，以避免非唯一或坐标loss牵制；
- response：P251从头训练，唯一loss为冻结P246重构probability L1；P248/P249结构、data、levels、steps、seed与两门不变；
- 防重复：不继续扫PCGrad、手工权重或budget/cycle组合；P251失败则关闭这一inverse student family。

下一可用编号：`V67-F189`。

### V67-F189 — direct tandem response-only训练仍停在相同误差带

- canonical：`run://worldsim_v67/WS-V67-P251-DIRECT-TANDEM-INVERSE-BUDGET-01/20260831T083000Z__direct-tandem-inverse-budget-s0-r1`；
- 观察：P201 inverse-budget MAE=`.020305`通过，但冻结P246重构probability MAE=`.018065>.015`；source/P183为
  `.016283/.016624`；结构违规仍为`0/0`；
- 解释：相对P249只改善`.000192`，且P248 budget-only、P249 mixed、P251 response-only三种语义都收敛在约`.018`；
  这不再支持继续扫描loss或优化器，瓶颈更像当前单一amortized inverse representation；
- literature response：marginal-utility planning把资源决策表达为增加一单位资源的value，不要求显式神经inverse；
  单调网络则允许从已冻结的monotone forward value取得结构一致的非负导数；
- response：关闭inverse student family；P252改为蒸馏冻结P246解析`dP/dlog-budget`的非负elasticity head；
- 防重复：不再试inverse width/knot/level/step/loss/weight/optimizer或warm-start；P250仅完成此前冻结的same-read报告。

下一可用编号：`V67-F190`。

### P252/P253 marginal-value milestone note — 新对象通过且fresh secondary冻结，无新增failure

- P252 canonical=`run://worldsim_v67/WS-V67-P252-MARGINAL-RELIABILITY-ELASTICITY-01/20260831T090000Z__marginal-reliability-elasticity-s0-r1`；
- P201 elasticity MAE=`.019285`、mean within-query Spearman=`.925247`，2/2，非负违规0；
- P253在P243 rows前冻结为same-read secondary，P252/P246/budgets/MC/两门不变；
- P254只推进冻结surrogate下的shadow-price budget policy，不重开inverse student family；下一failure id保持F190。

下一可用编号仍为：`V67-F190`。

### V67-F190 — P243 scene-shard推定漏掉同log跨archive LIDAR members

- run：`run://worldsim_v67/WS-V67-P243-CONTINUOUS-BUDGET-FRESH-CONFIRMATION-PREP-01/20260831T060000Z__continuous-budget-fresh-prep-s0-r1`；
- symptom：八个推定shards全部结束并命中3,518/3,914 required LIDAR，但缺396个同一`n015-2018-11-14-19-09-14`
  log members；prep在preprocess/row materialization/quality前退出；
- root cause：archive part不是scene-exclusive；scene编号映射可定位大多数文件，但一个log的sensor members可跨part；
- literature/open-source response：nuScenes官方devkit setup明确要求下载并在同一root合并全部archives且不要覆盖公共目录；
- resolution：cohort、目标members和已提取文件不变；只并行扫描此前未触碰的01/06补精确missing set。初始recovery入口
  仍会先重扫09，发现后立即取消且0新增output；r3以`recovery_only`跳过primary scan；
- claim impact：0 target/quality read，不换scene、不改P242/P244/P246/P249/P252/P254或任何decision。

下一可用编号：`V67-F191`。

### V67-F191 — P256 evaluation重复附加已有group-member轴

- run：`run://worldsim_v67/WS-V67-P256-GROUP-BUDGET-DUAL-COMPILER-01/20260831T094500Z__group-budget-dual-compiler-s0-r1`；
- symptom：12k训练完成且末段train price MAE约`.007`，首次source evaluation的`_reward`将`G×F×S` budget shape
  再拼接`S`，请求把`G×1×S×36` broadcast到`G×F×S×S×36`而抛ValueError；P183/P201 metric=0；
- root cause：helper按原先`G×F` dual-price输入书写，但调用方已先通过P254展开成per-member budgets；
- literature/open-source response：NumPy官方`broadcast_to`要求原shape与目标shape按广播规则兼容；已有member轴应直接
  保留，不能再次添加；
- resolution：feature target改为`budget_z.shape+(36,)`，budget直接flatten；r2从头训练，data/group/fraction/
  bisection/model/seed/steps/decisions全部不变；
- residual/recovery：r2的return reshape仍把`S`附加一次，在同一quality前位置退出；继续归入F191。r3把结果直接
  reshape回既有`G×F×S`再沿S求均值；P201 fraction MAE=`.016420`、regret=`.00001163`，2/2，F191关闭；
- prevention：不为单一shape错误增加smoke/regression matrix，仅在正式入口修复调用合同。

下一可用编号：`V67-F192`。

### P256 milestone note — fixed-group dual compiler恢复并通过，无新增failure

- canonical r3=`run://worldsim_v67/WS-V67-P256-GROUP-BUDGET-DUAL-COMPILER-01/20260831T103000Z__group-budget-dual-compiler-s0-r3`；
- P201 26 groups的attained fraction MAE=`.016420`、frozen Lagrangian regret=`.00001163`、price violations=0，2/2；
- F191两次入口均在任何P201 metric前退出，r3科学合同与r1完全相同；
- next：P257改变utility为单次冻结的log reliability，不扫P256 group/architecture；下一failure id保持F192。

下一可用编号仍为：`V67-F192`。

### P258 milestone note — log-utility fixed-group dual compiler通过，无新增failure

- canonical=`run://worldsim_v67/WS-V67-P258-LOG-UTILITY-GROUP-DUAL-01/20260831T110000Z__log-utility-group-dual-s0-r1`；
- P201 fraction MAE=`.015445`、frozen log-Lagrangian regret=`-.00000421`、price violations=0，2/2；
- P256与P258说明linear/log两种冻结surrogate utility均能被fixed-group共享dual price摊销；不外推真实scheduler；
- next：P259连续条件化alpha，不再分别训练更多离散risk utilities；下一failure id保持F192。

下一可用编号仍为：`V67-F192`。

### V67-F192 — P243冻结scene-1011的396帧不在本机全部十个官方trainval blobs中

- runs：r1已扫推定02/03/04/05/07/08/09/10；r3=`run://worldsim_v67/WS-V67-P243-CONTINUOUS-BUDGET-FRESH-CONFIRMATION-PREP-01/20260831T101500Z__continuous-budget-fresh-prep-s0-r3`补扫01/06；
- symptom：最后01/06分别约`950.7/900.2s`且均0命中；缺失集合仍是396个
  `n015-2018-11-14-19-09-14` LIDAR，元数据映射表明全部只属于`scene-1011`；
- external evidence：nuScenes官方devkit要求下载全部archives并合并到同一root；本机已覆盖10/10 trainval blob parts，
  因而不能再靠猜测shard或重复扫描恢复；不增加checksum/hash/fingerprint；
- scientific impact：prep在preprocess/row/quality前退出，P243及所有same-read secondary仍未读新target；
- response：保留其余九个冻结unused val scenes，不用任何历史验证scene补位；仅移除不可用`scene-1011`且不替换，
  cohort边界改为9 scenes/9 logs、location=`5/2/1/1`。P242/P244/P246/P249/P252/P254 artifacts、budgets、MC、
  metrics与gates均不变；r4 prep与r2 confirmation接替；
- prevention：本地archive不可用作为明确resource exception记录，不做分卷重扫、完整数据审计或冗长回归。

下一可用编号：`V67-F193`。

### V67-F193 — P249 inverse compiler在P243 fresh same-read上仍未过response重构门

- canonical：`run://worldsim_v67/WS-V67-P250-INVERSE-BUDGET-SAME-READ-CONFIRMATION-01/20260831T121200Z__inverse-budget-same-read-confirmation-s0-r2`；
- observation：9-scene/1,710-trajectory same-read secondary上inverse normalized log-budget MAE=`.016419≤.075`，
  但冻结P246 reconstructed probability MAE=`.015582>.015`；level/horizon monotonic violations=`0/0`；
- context：development P249为`.018257`，fresh误差更低但仍未跨过事前冻结response门；lower-censored target占
  `20.62%`，与F187--F189的陡峭/截断响应解释一致；
- response：维持inverse student family关闭；不因接近阈值而放宽gate、换scene、增加loss/width/knots或再次训练；
  同一次read的P243/P245/P247/P253/P255均按原冻结合同报告；
- claim impact：只拒绝clipped reliability level到minimum budget的amortized inverse response fidelity；不否定
  P243/P245/P247 continuous forward surfaces、P253 marginal ranking或P255 shadow-price policy。

下一可用编号：`V67-F194`。

### P259/P260 milestone note — alpha-fair连续risk条件化推进，无新增failure

- P259 canonical=`run://worldsim_v67/WS-V67-P259-ALPHA-FAIR-SHADOW-PRICE-POLICY-01/20260831T111500Z__alpha-fair-shadow-price-policy-s0-r1`；
- P201 budget MAE=`.011076`、frozen alpha-fair regret=`.0001333`、price violations=0，2/2；
- P260 P201 fraction MAE=`.017168`、frozen alpha-fair Lagrangian regret=`.00001031`、violations=0，2/2；
- P261只把冻结P259推进到train sizes 32/64/128、heldout 48/96的permutation-invariant set compiler，
  alpha/fraction/steps/两门一次锁定，不扫attention或pooling family；
- next failure id保持F193。

### P261/P262 milestone note — variable-set通过并进入task-horizon preference，无新增failure

- P261 canonical=`run://worldsim_v67/WS-V67-P261-VARIABLE-SET-ALPHA-FAIR-DUAL-01/20260831T120000Z__variable-set-alpha-fair-dual-s0-r1`；
- P201 sizes 48/96 aggregate fraction MAE=`.014662`、regret=`-.000000447`、violations=0，2/2；
- P262增加连续horizon preference这一新任务条件，不回头扫P261 pooling/attention/width；五train与四heldout
  preference、alpha交叉一次冻结；
- P262 P201 budget MAE=`.012221`、task-alpha utility regret=`.0002099`、violations=0，2/2；P263仅推进其
  fixed-group shared dual，不修改preference scalarization；
- next failure id保持F194。

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

### V66-F02 — fixed-budget conflict triage不等于natural physical repair

- 分类：`algorithm/evaluation`；状态：`closed_negative_after_single_recovery`。
- 观察：P7 L0在固定290 action budget下处理210/307 conflict states，exposure reduction=`68.40%`且Actor/hazard proxy
  保留；但`physical_geometry_mutated=false`，所以handled只是候选，不是实际修复。Q0还出现scene yield=`5/6`，说明简单
  actor-blind ranking可能把一个scene local geometry全部送入action并造成easier-world风险。
- 推翻项：推翻“ranking/triage通过即可称P7 physical distribution成功”；不推翻P3L/P3C ranking或P6 package capability。
- 防重复/复开：禁止把action count写成repaired artifact count，禁止调50% budget/score threshold，禁止删Actor或用target
  label决定primitive retention。
- 外部检索迁移：NeuRAD sensor-aware dynamic Actor、Neural Scene Graphs track-based static/dynamic decomposition、
  Cam4DOcc instance 4D occupancy，迁移为“canonical Actor collision shell保持 + motion-compensated sensor-supported local
  surface repair”。详见`P7R_SENSOR_SUPPORTED_REPAIR_MIGRATION_FREEZE.md`。
- 合法恢复：沿用P7 L0 action set；acted boundary只保留same-Actor motion-compensated hit，target仅评估；同时过conflict
  reduction、clean/overall geometry yield、Actor/shell/track/hazard preservation gates。
- 证据：`WS-V66-P7-HAZARD-PRESERVING-DISTRIBUTION-01` /
  `20260828T092919Z__fixed-budget-distribution-s0-r1`；首次physical recovery reject=
  `WS-V66-P7R-SENSOR-SUPPORTED-ACTOR-REPAIR-01/20260828T093710Z__sensor-surface-repair-s0-r1`。
- recovery：exact hit保留过稀；P7R2依据PoinTr/SnowflakeNet/RFNet冻结Actor-local 0.512m support expansion，
  详见`P7R2_RADIUS_SUPPORT_RECOVERY_FREEZE.md`。唯一恢复仍未同时满足冲突下降与干净几何保留，family终止。

P7R point-level repair当时按冻结规则实现；target只参与post-repair metric，未进入L0 action或same-Actor hit retention
rule。该阶段`V66-F02`进入唯一恢复，最终终态见下。

P7R formal被拒绝：conflict reduction=`0.847660`，但overall/clean boundary retention=`0.383588/0.395715 < 0.40`；
7/9 gates通过不改变verdict。根因是0.2m exact evidence-voxel hit过稀；随后只执行预先冻结的P7R2 fixed
one-native-voxel `0.512m` same-Actor support neighborhood，原九门不变。

P7R2 formal同样被拒绝：overall/clean boundary retention=`0.617684/0.619549`通过，但conflict reduction=
`0.417872 < 0.50`；8/9 gates通过不改变verdict。固定支持半径增加后同时保留了过多conflict points，暴露sensor proximity
无法区分合法表面支持与Actor-local free-space conflict。按预冻结规则不试中间radius、不降gate、不改budget、不训练
completion model；`V66-F02`以`closed_negative_after_single_recovery`终止。P7 triage正结果保留，但physical repair、
RL-ready distribution与P9继续锁定。完整证据：`P7R2_RADIUS_SUPPORT_RESULT.md`。

### V66-F03 — P8低速stop-state重复应用jerk update

- Pre-run operational note：P8首次shell invocation因未设置repo `PYTHONPATH`在module import前退出；没有进入runner、
  创建run directory或读取scientific metric。只修正launcher环境后执行同一代码/config，因此不分配failure ID，也不构成
  第二次scientific read。
- 分类：`implementation/numerical`；状态：`resolved_by_single_implementation_recovery`。
- 观察：P8六场景X1 collision steps均为0，但scene-0001/0219 command jerk分别为`9.637574/7.400627m/s^3`，
  超过固定`6m/s^3`；仅4/6 scenes全门通过，formal verdict拒绝。
- 根因：零速边界先通过正常rate limiter更新acceleration，随后stop分支再次增加一个jerk step；同一离散步对command
  应用了两次rate update。不是IDM参数或Actor selection证据。
- 防重复：禁止换Actor/scene、调headway/IDM/AV/horizon、放宽jerk gate或只删低速场景。
- 外部检索迁移：Autoware longitudinal controller的DRIVE/STOPPING/STOPPED state与vehicle command longitudinal
  jerk limiter，迁移为明确stopped desired acceleration=0且每步只过一次原rate limiter。
- 唯一恢复：P8R只移除第二次increment，其余输入、参数、轨迹、指标与gates exact不变；失败关闭P8 family。
- 证据：`WS-V66-P8-REACTIVE-ACTOR-01/20260828T095440Z__reactive-actor-s0-r1`；冻结：
  `docs/autoresearch/worldsim_v66/P8R_STOP_STATE_JERK_RECOVERY_FREEZE.md`。

P8R保持全部实验参数与gates exact，只把stopped desired acceleration置0并让command每步通过原rate limiter一次。
六场景全部支持，pooled X0/X1 collision steps=`306/0`，minimum X1 gap=`1.948192m`，maximum command jerk=
`6.000000m/s^3`；`V66-F03`恢复关闭。该恢复只修复numerical update，不把synthetic response扩展到natural interaction，
也不改变P7 terminal negative或解锁P9/RL。证据：`WS-V66-P8R-STOP-STATE-JERK-RECOVERY-01/
20260828T095839Z__stop-state-jerk-recovery-s0-r1`。

下一可用编号：`V66-F04`。

### V6.5 终态边界（2026-08-28）

- `V65-F19` 为 terminal algorithm negative：P10X 5/6 gates通过，但 direct selected-action cost reduction
  `16.38% < 25%`。支持给定 `tau` 的 visited-state reliability ranking/calibration，不支持 direct action
  authority、planner、policy、closed-loop、RL或safety。
- V6.6 只可把 q0/visited-state score当 diagnostic；不得以新阈值、第二 confirmation cohort或新 critic
  复开 V6.5 family。

### V6.4 当前边界（2026-08-27）

- V6.4已终态`v64_research_complete_report_ready`。正证据边界为P6R/P4C独立选择性校准和P10R4 untouched
  fixed-opportunity exact empirical route-local risk；不支持population、physical collision、planning、closed-loop、RL或
  safety claim。P11/P11R collision critic以`V64-F28 closed_negative_after_single_recovery`终止，P11D确认unsafe prior与
  ranking同时跨cohort漂移，不授权第二次threshold recovery、大型NWM/RL或复开V6.4。
- 版本收口与arXiv索引没有新增实验或failure ID。`V64-F25`只在P10R4独立固定机会分母层面解除；P10T/current-M0
  `V64-F21`负结论及P10R2 selected-denominator relative non-improvement保持。详细终态见
  `docs/autoresearch/worldsim_v64/V64_RESEARCH_FAMILY_CLOSEOUT.md`与`ARXIV_EVIDENCE_INDEX.md`。
- shutdown前最终报告审计只读确认7个关键canonical run的目录与`summary.json/status.json`可用，并按既有`V64-F03`
  合同在非登录shell激活`motionproj`后解析；没有重跑、重算、改结果或新增failure。arXiv写作交接见
  `docs/autoresearch/worldsim_v64/V64_ARXIV_REPORT_HANDOFF.md`，本次failure ledger delta=`none`。

- V6.4 从`research/worldsim-v6.3-surface-tail@c192955`直接建立；`V63-F24`仍关闭 Surface family，新的合法路线只能研究
  native aleatoric/epistemic uncertainty、scene/stratum conditional risk 与独立 case-level calibration。
- 首个核心假设已冻结为原生 U0 对比 geometry-conditioned feature-density U2；旧 4+2 scene 只作机制诊断，禁止作为
  fresh V6.4 claim，也不允许由该结果读取 calibration/confirmation/test。
- retrospective U2 已在两个旧 evaluation scene 都优于 U0，但 FPR@95TPR 仍高，只授权建立 fresh cohort，不授权
  authority/calibration claim。当前 `V64-F01--F03`均为 resolved engineering/operations，不得写成算法负结论。
- compact fresh cohort 已从 V6.1–V6.3 未读 quality 的 scene 中按 metadata-only 冻结。r2 证明候选还必须存在于冻结的
  train temporal metadata；恢复队列在任何 fresh quality read 前改冻并登记`V64-F05`。不得把更早版本曾出现过但未进入
  V6.1–V6.3 UQ路线的scene错判为legacy，也不得用本轮后续质量回改cohort或复用r2部分产物。
- r3 已完整生成6-scene/72-target native sidecar，单卡资源通过；这只是capability，不得提前写成fresh UQ成立。
- fresh UQ已在target quality读取前冻结为同一PCA-16/GMM-4和两条晋级门；同数据换seed、PCA/GMM或改scene均禁止。
- fresh U2虽过相对门，但两scene内AUROC都约0.498且FPR95约0.96，登记`V64-F10 active`。后续只允许已冻结的
  fit-only PCA-16 logistic risk head执行一次；不得把监督标签用于evaluation拟合、扫描超参或扩展split。
- U3已通过两fresh scene绝对AUROC门，但高FPR95保持`V64-F11 active`。独立calibration/confirmation已按metadata-only冻结为
  `16+8 scenes`；当前与旧evaluation score均不得回流修改cohort、risk rule或head。
- P6整批准备在共享盘扫描`>1 h`后仍为`9/10 shards`且GPU空闲，登记`V64-F12 active`；恢复采用scene-ready有界
  producer-consumer，不重复已完成shard、不做无关GPU filler，confirmation仍保持锁定。
- P6 prep r1又暴露固定`1176/196`不适用于`nbr_samples=41`的scene-1045；登记`V64-F13
  recovery_frozen_pre_quality`，恢复只使用metadata派生帧数并复用已完成raw/scene。
- P6 r2已完成24场景并删除临时raw，`V64-F12/V64-F13`分别由producer-consumer与variable-length恢复；短SSH命令继承
  stdin导致本地编排不退出，登记`V64-F14 resolved_operations`并按OpenSSH官方`-n`修复。
- 独立192-case校准在最低5% coverage仍为`41/192` failure、risk/UCB=`0.2135/0.2929`，登记`V64-F15 active`；
  confirmation target保持未读，禁止放宽risk合同或删stratum。

### V6.3 报告使用边界（2026-08-26）

- active scientific negative evidence=`V63-F02, V63-F24`：原生特征没有解除逐点路线的`4/4 false-safe`，而B3
  Surface-Mean随后在P6两scene均输Native B2并按Stop2关闭surface family。
- recovered scientific failure=`V63-F19`：P5 positive-authority collapse已由P5D确认并由P5R primal-dual恢复训练侧可行性；
  该恢复不能覆盖后续P6 stage rejection。
- `V63-F01/F03--F18/F20--F23`均为resolved或resolved_preexecution工程、协议、数据表示、数值、metadata或operations
  记录；论文附录可用作复现教训，但不得当作算法negative count。
- B4/B5/M0和P7--P11是`not executed/locked`，不是失败attempt；本次文档审计没有新增failure ID，也没有重分类旧失败。

## 0. 使用合同与渐进式导航

### 0.1 渐进式读取

1. 每次研究任务先读本节和“V1–V5 版本总览”，确认当前问题属于算法、数据、评测、协议、工程、资源还是治理风险。
2. 根据 task、模块和关键词用 `rg` 定位 failure ID；只展开命中的完整条目及其相邻门禁，不默认一次加载全部长账本。
3. 新计划至少引用一个直接相关 failure ID；涉及跨路线复用时，再读取对应版本详细章和 canonical evidence。
4. 只有在做全局路线审计、迁移或报告附录时才通读全文件；归档快照仅在核对当时字节或历史编号时读取。

### 0.2 渐进式写入

1. 新坑先在当前版本详细章追加一个唯一 ID，再更新本文件顶部版本总览；不要复制整章或另建版本 failure 文档。
2. 旧结论被新证据解除时不得删除原条目；在原条目追加 `resolved/superseded`、新证据、仍然成立的边界和新 ID。
3. 写入只包含可复用的失败事实与防重复门禁；逐步日志、完整 stdout 和大表留在 run，实验结果表留在
   `EXPERIMENTS.md`。
4. 每个正式实验在启动前登记 `failure_ledger_refs`，收口时登记 `failure_ledger_delta`。任何 `blocked/rejected`、
   前提被推翻、工程恢复、门禁失败或旧风险解除，都必须在同一逻辑提交中更新本文件；若确无新增，只在实验台账写
   `failure_ledger_delta=none`，避免向失败账本灌入成功流水账。

### 0.3 单条记录最小 schema

| 字段 | 必填内容 |
|---|---|
| ID / 分类 | 唯一 `<路线>-FNN`；分类为 `algorithm/data/evaluation/protocol/engineering/resource/governance` 之一或组合 |
| 状态 | `active/resolved/superseded`；实验 task 状态仍只用 `pending/running/blocked/done/rejected` |
| 观察事实 | 分母、错误、指标、资源或 terminal；不把推断写成事实 |
| 根因与推翻项 | 已确认根因，以及它推翻了哪个假设、实现合同或旧结论 |
| 防重复与复开 | 禁止的事后调参/删分母/覆盖 run，以及合法复开需要的新证据 |
| 证据 | task/run ID、commit、summary/manifest/文档路径；工程失败与算法 reject 分开 |

### 0.4 目录

- [V6.6 详细账本](#detail-v66)
- [V1–V6 版本总览与 V1/V2 汇总](#1-v1v6-版本总览与-v1v2-汇总)
- [V6.4 详细账本](#detail-v64)
- [V6.3 详细账本](#detail-v63)
- [V6.2 详细账本](#detail-v62)
- [V6.1 详细账本](#detail-v61)
- [V6 详细账本](#detail-v6)
- [V5.1 详细账本](#detail-v51)
- [V5 详细账本](#detail-v5)
- [V4 详细账本](#detail-v4)
- [V3.3/V3.2/V3.1 详细账本](#detail-v3)
- [V2 继承门禁](#detail-v2)
- [V7/V7.1、N1/cut-in 与历史路线](#detail-legacy)
- [跨路线原则与新实验检查表](#detail-cross-route)

## 1. V1–V6 版本总览与 V1/V2 汇总

| 版本 | 终态/核心推翻 | 主要工程坑 | 详细证据入口 |
|---|---|---|---|
| V1 | AD-GS 六场景复现成立，但 persistent identity 不存在；唯一候选与已有工作直接重合，M7 rejected，M8/M9 未授权 | DGGT `pointops2` PEP 517 隔离缺 torch；训练完成、评估依赖和容器生命周期必须分开 | 下方 `V1-F01`–`V1-F06`、`PIVOT-F14B/F15/F16`、V1 frozen archive |
| V2 | M0–M4 闭环；M5 三场景压力测试只完成部分资产，不能写成路线完成；局部保持不等于删除后背景真实 | 空 shell PATH、CUDA/headers/包版本、devkit schema、SM 8.6、render 累积内存、子进程回收、空 actor slice | 下方 `V2-F01`–`V2-F09`、V2 继承门禁、`EXPERIMENTS.md` V2 注册表 |
| V3/V3.1 | A1 保持 off；A2 是 boundary/全局/成本 tradeoff；A3 局部精修不晋级；P1 pruning rejected；P2/P3 只支持存储/资产拆分 claim | 相机标签、随机 CUDA 初始化、分辨率三层合同、runtime state key、资源 ceiling、前馈前置条件 | `V3-F01`–`V3-F25` |
| V3.2/V3.3 | S4 temporal 未完成、S5 语义生产链回退；RoadPatch/asset/release 只在冻结场景和协议成立，不构成跨场景 dominance | frozen base identity、empty-target、模型/视图可用性、类型/枚举严格比较、确定性 archive | `V32-*`、`V33-*` 详细章 |
| V4 | M1 scene-disjoint validation rejected；M2 selective routing 成立但 geometry MAE 退化 `+3.3908 m`；M3 仅在冻结 18-scene exact-once test confirmed | cohort 非确定性、split leak、SSH 断管、解释器分层、CUDA arch、immutable run/staging、完整 denominator | live canonical `V4-F01`–`V4-F49` |
| V5 | M1/M2/M3 全部 rejected；structured graph 不稳定、无 absolute geometry-safe candidate、constraint projection 信号不足 | KITTI calib/OXTS 语义、缺 LiDAR 帧、provenance enum、launcher 原子目录、heading metric 和 long-run stdout | `V5-F01`–`V5-F59` |
| V5.1 | M1-only 已收尾、无 promoted candidate；U2/B3 保留为 V5.2 comparator。LUDVIG uplift/raw graph、progressive、simple voxel node、Gaussian Grouping 与 exact faithful Trace3D operator 均按各自冻结门 rejected；Stage H 未运行，保持 pending 并由 V5.2 observation-source scope 取代 | uplift 无 actor margin；progressive/node elevation 的 IoU/FN 跨场失稳；identity coverage/persistence 不足；Trace3D alpha 跨 fresh process 非确定；另有零长 KNN、跨 shell、解释器/helper/CUDA 初始化、PDF/CLI、partial staging、solver/license/stdout、bytecode/cache、SAM 显存、batch sensitivity 与 CUBLAS 恢复边界 | `V51-F01`–`V51-F66` |
| V5.2 | 18-case 人工复核冻结 `9 BASE_FAILURE + 8 M123_ELIGIBLE + 1 unresolved`；M1/M3 症状匹配增强但 causal bridge 未通过，M2 降级为 safety/abstention | 原 census 的 actor/boundary 指标可被 global collapse 污染；eligible case 必须保持 `5 Discovery design + 3 one-shot Confirmation` | `V52-F01`–`V52-F02` |
| V6 | V5.2 TrackBayes 主线已由 world-compiler direction reset 取代；G0–G3 与 R1 capability gate 已通过，尚无方法质量结论 | pytest import/runtime profile 不能混用；历史外部资产可能只剩 manifest；冻结 plan 的 exact allowlist 必须显式纳入 terminal closeout；formal capability run 禁止 dirty source | `V6-F01`–`V6-F05`；G0–G3/R1 governance artifacts；V6 plan |
| V6.1 | 最小实验负结论收口：oracle `10/28, 0 false-safe`，GaussianWorld/IR-WM 均恢复 `10/28` 表面支持但各自 `10/10 false-safe`；ME-4 未解锁 | predicted argmax Occupancy 不能升级为安全 authority；第三 backend、threshold/grid/history/verifier sweep 均冻结 | `V61-F01`–`V61-F13`；`V61_MINIMUM_EXPERIMENT_CLOSEOUT.md` |
| V6.2 | CPSC-Lite family负结论收口：P6与唯一P6R均为`4/28,4/4 false-safe`，P7/P8未解锁 | evidence dropout把source-valid UNKNOWN从82.7%降到63.9%但未改变四个unsafe accepts；query-wise projection不能提供hidden surface authority；第二recovery、O_eval调参、backbone/backend/sweep冻结；未来复开需native logits/features、独立calibration与hidden-surface risk supervision；不新增哈希/校验和/指纹 | `V62-F01`–`V62-F07`；`P6R_EVIDENCE_DROPOUT_CLOSEOUT.md` |
| V6.3 | P2D native pointwise rejected；P3/P4 passed；P5/P5D objective collapse；P5R恢复训练candidate；P6 B3在两scene均输Native B2，surface family closed negative，P7锁定 | 训练内feasible不得冒充stage candidate；B3 tail与area同时失败后禁止继续B4/B5/M0、换seed/模型/门或读取legacy/H/T；未来复开必须是fresh uncertainty representation与conditional-coverage新版本 | `V63-F01`–`V63-F24`；`ARXIV_EVIDENCE_INDEX.md`；`P6_SURFACE_FAMILY_CLOSEOUT.md`；各P2D/P3/P4/P5/P5D/P5R/P6 prereg |
| V6.4 | full-native MLP与conditional M0通过独立exact-once；M1在untouched fixed-opportunity denominator相对支持；P11 collision critic经一次独立threshold recovery仍rejected，版本终态report-ready | U2绝对弱、U3高FPR、PCA calibration失败；selected与fixed denominator必须分开；共享盘I/O以restricted-shard、per-scene staging、ready-first GPU queue恢复；critic unsafe prior与ranking跨cohort漂移 | `V64-F01`–`V64-F28`；`V64_RESEARCH_FAMILY_CLOSEOUT.md`；`ARXIV_EVIDENCE_INDEX.md`；各P6R/P4C/P10/P11 closeout |
| V6.5 | visited-state reliability ranking与单调校准可迁移；one-shot direct action selection benefit失败并终止 | q0只可作给定轨迹访问状态的reliability diagnostic；禁止第二confirmation、阈值/lattice/critic救援 | `V65-F01`–`V65-F19`；`V65_ARXIV_TECHNICAL_REPORT.md`；`ARXIV_EVIDENCE_INDEX.md` |
| V6.6 | terminal/report-ready：两级certificate/HARP bake/P8R synthetic response支持；P7 surface repair终局负结果；P9未执行 | `V66-F02 closed_negative_after_single_recovery`；`V66-F03 resolved_by_single_implementation_recovery`；下一failure id=`V66-F04` | `WORLDSIM_V6_6_HARP_COMPILER_PLAN.md`；`docs/autoresearch/worldsim_v66/V66_ARXIV_TECHNICAL_REPORT.md` |

P4C conditional compiler freeze没有新增failure：它只把已读calibration中“50%的3个failure全部在rain”迁移为单一固定
coverage map，并在任何新quality read前冻结新8-scene confirmation。若formal replay不满足预注册coverage/risk gate，直接登记
algorithm failure并关闭该candidate，不改mapping或扫描第二版本。
新confirmation入口前free disk仅29 GiB；只回收精确的13 GiB pip下载缓存后为41 GiB，未删除formal run、模型、环境或processed
资产。这是预防性可恢复空间管理，不新增failure ID。
P4C确认执行继续直接复用V64-F16的scene-ready迁移并将新temporary raw改为独立精确路径；在target read前已冻结所有run ID、
单preprocess/双GPU并发和controller cleanup ownership。入口未出现新failure。

### 1.1 V1 汇总条目

- `V1-F01`（`algorithm/evaluation`, `active`）：AD-GS 六场景 exact reproduction 只证明 frozen baseline 可复现，
  不证明对象级编辑、未知背景恢复或新方法成立。证据为 V1 M4 aggregate；禁止把复现分数重命名为贡献。
- `V1-F02`（`engineering`, `resolved in V2`）：DGGT V1 在 input staging 前被 `pointops2` 的 PEP 517 隔离构建
  缺少 torch 阻塞，没有质量/速度数字。V2 用固定 compiler/runtime/headers 和 upstream 非隔离安装解除工程前置；
  V1 terminal 仍保持 blocked，详见 `PIVOT-F14B`。
- `V1-F03`（`algorithm/data`, `active`）：六场景 pseudo ID 最长支持仅 `1/6/1/1/2/1` 帧，checkpoint 只有
  二值 `obj`，`0/12` object slots 可评；`persistent_object_identity_unavailable` 不能靠事后几何关联回填。
- `V1-F04`（`algorithm/governance`, `active`）：候选“恢复持久身份并绑定 actor 后做轨迹编辑”与 InstDrive、Director、
  OmniRe、HorizonForge、G²Editor 直接重合；适配 AD-GS 是工程，不足以通过 novelty gate。M7 保持 rejected。
- `V1-F05`（`protocol/evaluation`, `active`）：M7 拒绝后 M8/M9 未授权，0 seeds、0 proposed metrics、human verdict
  为 `null`。禁止事后补 endpoint、把 0 coverage 写成提升或由 Codex 代填人评。
- `V1-F06`（`data/governance`, `active`）：早期 cut-in 路线没有官方召回率分母，strict-v2 在 675 scenes 仅
  `1 PASS / 1 scene`；这说明当前可验证事件池过稀，不说明 nuScenes 没有 cut-in。cut-in 只能作可选演示，不能再
  承担主数据入口或论文成立条件。

V1 canonical 状态、实验和专项报告保存在 `docs/archive/2026-07/dynamic-reconstruction-v1/`；其中的
`RESEARCH_FAILURES.md` 是冻结快照，不再单独维护。

### 1.2 V2 汇总条目

- `V2-F01`（`engineering`, `resolved`）：非登录 shell 中裸 `python` 不在 PATH；runner 必须显式绑定解释器，
  不运行 `conda init`，也不能把 PATH 错误写成网络/依赖失败。
- `V2-F02`（`engineering`, `resolved`）：DGGT r1–r7 依次暴露 pip backtracking、CUDA compiler/runtime mismatch、
  cusparse headers、transformers/diffusers/torch schema、`flow_vis` 和 retry schema；每次修复必须新 run，native 已完成
  的阶段不被后续 common-eval blocked 覆盖。
- `V2-F03`（`data/protocol`, `resolved`）：磁盘 `sample.json` 没有 devkit runtime `anns` 反向索引，Decimal、
  invalid projection schema 和 nearest-sweep 也会破坏 exact mapping；最终必须以 exact `sample_token` 为主键。
- `V2-F04`（`resource/protocol`, `active`）：30k checkpoint 完成不等于累积 full render 完成；后者在 577/588 时
  越过 90% cgroup 合同并安全停止。训练与 post-render 必须分开裁决，不得删 checkpoint 或写成 OOM/方法失败。
- `V2-F05`（`engineering`, `resolved`）：CUDA 扩展 import 成功不代表包含 RTX 3090 SM 8.6 kernel；必须从冻结源码
  按目标 arch 重编并运行真实 forward/backward，不能只做 import smoke。
- `V2-F06`（`data/evaluation`, `active`）：训练可把非目标 actor Gaussian slice 裁为空；registry 必须显式 unavailable，
  选定 actor 必须非空，禁止静默删 denominator 或把空 slice 当成功删除。
- `V2-F07`（`engineering/resource`, `resolved`）：外层 timeout 不会回收 `start_new_session=True` 的子进程；必须按
  精确 PGID 清理并保留 interrupted terminal，长任务使用 detached controller 和独立日志。
- `V2-F08`（`algorithm/protocol`, `active`）：M5 只完成 0230/0242 checkpoint 与 0255 诊断；三场景×两 actor×四编辑、
  pseudo-hole/perception/final matrix 未完成。空 tensor `torch.cat` 是工程阻塞，不是 3DGS 方法失败，也不允许把部分资产
  写成 M5 done。
- `V2-F09`（`evaluation/algorithm`, `active`）：lateral/delete non-target PSNR 93/95 dB 主要是硬局部保持构造，
  不能证明 source footprint 后背景、边界或时序真实；后续必须把 outside preservation 与 hole/depth/boundary/temporal
  指标分开。

### 1.3 V4 历史编号冲突校正

2026-08-17 统一账本时发现，旧追加段落重复使用了 `V4-F30`–`V4-F33`。为保证后续引用唯一，本文件将 live canonical
编号校正如下；archive 快照保持原字节和旧编号，不回写：

- B0/D0 段的 `V4-F17`–`V4-F33` 保持不变；
- 历史 M1 development/validation `V4-F30`–`V4-F34` → live `V4-F34`–`V4-F38`；
- 历史 M1 rejection/M2 validation `V4-F35`–`V4-F39` → live `V4-F39`–`V4-F43`；
- 历史 M3 `V4-F40`–`V4-F45` → live `V4-F44`–`V4-F49`。

新文档、代码和 run manifest 只引用 live canonical ID；核对旧 commit/归档时同时记录“historical ID → live ID”。

<a id="detail-v64"></a>

## V6.4 原生不确定性编译器详细账本（2026-08-26）

- `V64-F01`（`engineering/runtime`, `resolved_pre_quality_read`）：P0 定向投影测试首次使用控制台入口
  `pytest -q tests/worldsim_v62/test_projection.py`，测试收集阶段即因仓库根目录未进入 `sys.path`触发
  `ModuleNotFoundError: motion_proj`。失败发生在 formal run、GPU context、数据与 V6.4 quality read 之前；没有科学结果。
  依据 pytest 官方 import-path 合同，仅把入口改为
  `python -m pytest -q tests/worldsim_v62/test_projection.py`，结果 `1 passed in 1.59s`。防重复：仓库内测试统一用
  `python -m pytest`；不为入口差异改包结构、污染环境或扩展回归矩阵。证据=
  `WS-V64-P0-SCOPE-GIT-01`、`docs/autoresearch/worldsim_v64/P0_SCOPE.md`、
  `https://docs.pytest.org/en/stable/explanation/pythonpath.html`。

- `V64-F02`（`engineering/operations`, `resolved_pre_formal_run`）：UQ prereg commit 首次两次普通
  `git push`在 30 秒窗口内无输出，远端 ref 保持 `04b343f`，本地 `f1764de`未丢失，formal run 尚未启动。GitHub
  官方状态页显示 Git Operations operational；本地 `localtun sessions`确认当前 AutoDL session 的 remote proxy 后，
  仅为该 push 显式设置 HTTP/HTTPS/ALL proxy，普通 push 成功推进远端到 `f1764de`。防重复：远端网络慢时先检查
  GitHub 状态和当前 LocalTUN session；端口是会话级，禁止复用旧记忆值，也不 force push。证据=`f1764de`、
  `https://www.githubstatus.com/`。

- `V64-F03`（`engineering/runtime`, `resolved_post_run_read`）：canonical UQ run 已成功结束后，第一次只读 summary
  命令在非登录 shell 直接调用裸 `python`，因环境未激活在读取文件前返回 `command not found`。按仓库环境合同 source
  `conda.sh`并激活 `motionproj`后，同一只读程序完成逐 scene 指标读取；run、summary 与模型均未修改。防重复：任何
  非登录 Python 命令显式激活环境；该错误不登记为算法或 formal run failure。证据=
  `run://worldsim_v64/WS-V64-P3-NATIVE-UQ-01/20260826T080200Z__uq-retrospective-s0-r1`、
  `https://docs.conda.io/projects/conda/en/25.1.x/dev-guide/deep-dives/activation.html`。

- `V64-F04`（`engineering/runtime`, `resolved_pre_data_read`）：fresh sidecar 首次 formal 入口
  `20260826T081300Z__fresh-native-s0-r1`在 wrapper 调用继承 runner 后，因 task parent 尚不存在而对
  `shutil.disk_usage`触发 `FileNotFoundError`。run leaf 未创建，GPU、processed scene、IR-WM 与 quality 均未触达，
  canonical run=`null`。恢复只在 wrapper 中先 `mkdir` task parent，再由未改的 runner 创建 exclusive run leaf；
  cohort、seed、资源门和 denominator 不变。防重复：disk probe 必须绑定已存在的挂载内路径，不把前置目录缺失写成磁盘
  不足或算法失败。证据=`WS-V64-P2-FRESH-NATIVE-SIDECAR-01`、
  `https://docs.python.org/3/library/shutil.html#shutil.disk_usage`。

- `V64-F05`（`data/interface`, `resolved_pre_quality_read`）：fresh sidecar r2=
  `20260826T081500Z__fresh-native-s0-r2`启动冻结 IR-WM worker 后，初始 cohort 中 val-split 的`scene-0100`与
  `scene-0632`在`nuscenes_temporal_infos_train.pkl`查询处触发`KeyError`。`scene-0230`已完成12个native units，
  worker wall=`35.8975 s`、peak GPU=`4.1305 GiB`，blocked run leaf总计`528 MiB`；其余scene未形成完整denominator，
  canonical=`null`，target evidence与任何fresh quality均未读。根因是selector只核对processed/raw可用性，却漏掉冻结
  extractor的train temporal metadata membership。检索IR-WM、BEVFormer与DriveStudio官方数据准备合同后，恢复仅在
  pre-quality 阶段改冻为六个均在train temporal metadata、且未进入V6.1–V6.3 quality ledger的scene；evaluation两scene
  从本机raw数据用官方DriveStudio流程物化。防重复：保留r2，不把12个部分unit混入r3，不生成val temporal metadata救旧
  cohort，不改变seed/targets/model/UQ/门；r3必须使用全新exclusive leaf完成72-unit denominator。证据=
  `docs/autoresearch/worldsim_v64/P2_FRESH_COHORT_FREEZE.md`、
  `https://github.com/ziyc/drivestudio/blob/main/docs/NuScenes.md`、
  `https://github.com/fundamentalvision/bevformer`、`https://github.com/APRIL-ZJU/IR-WM/blob/ir-wm/README.md`。

- `V64-F06`（`engineering/operations`, `resolved_post_run_read`）：r3 已成功输出正式summary后，首次只读收口程序
  假定文件名为`P2_NATIVE_SUMMARY.json`，对不存在路径调用`Path.read_text()`触发`FileNotFoundError`；run、GPU、
  artifacts与quality均未修改。实际继承的V6.3 extractor写出`P2_SUMMARY.json`。按Python pathlib官方合同先枚举run根目录，
  再读取实际文件；canonical r3与全部指标不变。防重复：继承runner的consumer不得根据task或版本猜文件名，先用明确目录
  枚举或读取runner源码中的输出合同；不为只读路径错误重跑formal。证据=
  `run://worldsim_v64/WS-V64-P2-FRESH-NATIVE-SIDECAR-01/20260826T082600Z__fresh-native-s0-r3`、
  `docs/autoresearch/worldsim_v64/P2_FRESH_SIDECAR_CLOSEOUT.md`、
  `https://docs.python.org/3/library/pathlib.html`。

- `V64-F07`（`engineering/operations`, `resolved_pre_run`）：fresh evidence首次launcher在Windows PowerShell传给SSH的
  双引号字符串中使用`$(git status --porcelain)`；PowerShell按官方解析规则在本地提前执行subexpression，因本地工作目录
  不是repo返回fatal。远端evidence run目录仍不存在、数据/quality/GPU均未触达。恢复仅删除嵌入式subexpression，在单独
  只读命令已确认远端branch clean与目标路径不存在后，使用同一config和固定r1启动。防重复：PowerShell到SSH的双引号
  参数不嵌入`$()`或`$var`远端shell表达式；状态检查拆成独立命令，不把本地解析错误写成formal failure。证据=
  `run://worldsim_v64/WS-V64-P2E-FRESH-EVIDENCE-01/20260826T084000Z__fresh-evidence-s0-r1`、
  `https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_parsing`。

- `V64-F08`（`resource/protocol`, `resolved_by_native_voxel_recovery`）：继承的V6.3 surface compiler历史formal为
  `72 units / 47,568.47 s wall / 3,334.28 s max unit`。fresh surface r1运行约4分钟时仍为`0/72 units`、4 KiB，两个worker
  CPU各约100%且无资源异常；照旧执行预计浪费约13小时，并生成当前UQ不消费的signed-distance、patch、normal、actor与
  proposal registry。检索OCCUQ ICRA 2025的原生`200×200×16` voxel-level feature GMM，以及CuPy/cuCIM exact EDT后，
  选择前者：在UQ score读取前按精确PGID停止并保留partial，预注册唯一native-boundary-voxel r1；不安装CuPy、不继续旧
  full-stack。防重复：不得把partial写成算法失败，不得删除/覆盖r1；native r1后禁止回旧surface、换EDT/denominator或
  sweep救结果。证据=`docs/autoresearch/worldsim_v64/P4N_NATIVE_VOXEL_UQ_RECOVERY_FREEZE.md`、
  `https://github.com/ika-rwth-aachen/OCCUQ`、
  `https://docs.cupy.dev/en/latest/reference/generated/cupyx.scipy.ndimage.distance_transform_edt.html`。

- `V64-F09`（`algorithm/interface`, `resolved_pre_evaluation`）：native-voxel r1在四个fit scene采样后，冻结occupied-boundary
  denominator的预测FREE geometry组仅`43`点，小于4-component GMM固定最低`80`点，`NativeFeatureDensityUQ.fit`抛错；
  r1仅8 KiB resolved/status，无model、evaluation score或gate verdict。根因是把retrospective全surface上的双预测geometry
  条件化机械搬到几乎全occupied的region。OCCUQ官方`gmm_utils.py`按真实voxel类别收集feature，并在推理时对类密度做
  `logsumexp`边缘化。恢复因此固定为region整体一个boundary-global GMM-4；不是降低样本门或扫描组件数。防重复：不得复制
  43点、降80门、把evaluation并入fit或回到双组；v2只执行r2一次，其他输入/gate不变。证据=
  `docs/autoresearch/worldsim_v64/P4N_NATIVE_VOXEL_UQ_RECOVERY_FREEZE.md`、
  `https://raw.githubusercontent.com/ika-rwth-aachen/OCCUQ/main/tools/gmm_utils.py`。

- `V64-F10`（`algorithm/evaluation`, `active`）：native-voxel r2按冻结协议完成并通过相对门：pooled U2 AUROC=
  `0.518545`、较最佳U0增`0.083047`，scene support=`2/2`。但两个scene内U2 AUROC仅`0.498387/0.498295`，
  FPR@95TPR=`0.965465/0.960623`；scene-0359 AP低于prevalence，scene-0998的50% coverage risk高于prevalence。
  pooled改善可能部分来自scene-level prevalence/score shift，不能包装成可靠场景内ranking、authority或calibration。
  顶会迁移依据：OCCUQ将dense UQ supervision与feature GMM分工；ReliOcc采用plug-and-play hybrid voxel uncertainty；EvOcc
  用evidence supervision显式建模unobserved/contradicting evidence。恢复只允许先冻结一个用四fit scenes hidden-FREE标签训练的
  轻量risk head，再在相同两scene分母执行一次；禁止扫描GMM/PCA/seed/denominator/gate或读取更多split救结果。证据=
  `docs/autoresearch/worldsim_v64/P4N_FRESH_UQ_CLOSEOUT.md`、`https://github.com/ika-rwth-aachen/OCCUQ`、
  `https://doi.org/10.24963/ijcai.2025/220`、
  `https://openaccess.thecvf.com/content/CVPR2025/papers/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.pdf`。

- `V64-F11`（`algorithm/evaluation`, `active`）：P5固定监督risk head按预注册通过pooled和两scene AUROC门，pooled U3
  AUROC/AUPRC=`0.658118/0.148720`，scene AUROC=`0.640682/0.636266`。但FPR@95TPR仍为pooled`0.867738`、
  scene`0.859069/0.907021`；logistic输出也未在独立calibration set上校准。故本run只支持ranking，不支持低误报authority、
  calibrated probability、conditional coverage或safety。检索ICLR 2024 Conformal Risk Control后，合法恢复必须先冻结新的
  scene-disjoint calibration/confirmation cohort，以scene/unit为交换单元选择单调selective set，并在untouched confirmation
  一次验证；不得用已读scene-0359/0998选threshold、扫描risk/coverage或把voxel当独立样本制造虚假样本量。证据=
  `docs/autoresearch/worldsim_v64/P5_SUPERVISED_RISK_CLOSEOUT.md`、
  `https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html`、
  `https://github.com/aangelopoulos/conformal-risk`。

- `V64-F12`（`resource/operations`, `resolved_by_pipeline`）：P6正式准备入口扫描共享盘官方tar超过一小时后，已完成
  `9/10`个shard、临时raw约`14 GiB`、盘余量约`45 GiB`，但旧入口必须等全部raw和24个processed scene完成才启动
  IR-WM，观测GPU=`0% / 1 MiB`。24-scene native按既有6-scene输出外推约`13.3 GiB`，再叠加processed和临时raw，
  证明此前`~21.6 GiB`整批持久化估算缺少足够余量。该事实是I/O/调度阻塞，不是模型或数据质量结论。检索NVIDIA
  DALI异步pipelined execution、bounded prefetch queue及WebDataset shard streaming后，恢复保留已完成shard工作，并在
  DriveStudio scene达到冻结的`1176 images + 196 lidar`后立即按scene送入IR-WM，最多两个GPU worker；先处理16-scene
  calibration，模型冻结后再处理锁定confirmation。禁止重复扫描已完成shard、启动无关GPU filler、降低quality边界、把
  confirmation提前读入校准或用多卡掩盖共享盘瓶颈。证据=
  `run://worldsim_v64/WS-V64-P6-CALIBRATION-SIDECAR-01/20260826T100000Z__calibration-prep-s0-r1`、
  `https://docs.nvidia.com/deeplearning/dali/archives/dali_190/user-guide/docs/advanced_topics_performance_tuning.html`、
  `https://github.com/webdataset/webdataset`。

- `V64-F13`（`data/interface`, `resolved_pre_quality_read`）：prep r1在全部tar扫描完成、首个scene-1045官方
  DriveStudio转换成功后，以`images=1206, lidar=201`对旧六场景硬编码的`1176/196`做比较并抛错。r1未读
  Occupancy/UQ/hidden-FREE或calibration/confirmation质量；临时raw和完整首场景均保留。根因是nuScenes scene记录有
  `nbr_samples`，而`interpolate_N=4`的官方DriveStudio时间表长度为`(nbr_samples-1)*5+1`；scene-1045的
  `nbr_samples=41`故应为201帧、六相机1206图，不是文件缺失或重复。恢复只从冻结metadata派生每scene期望数，并让新r2
  显式复用现有临时raw和完整scene；不删除额外合法帧、不重扫tar、不改变12 target、cohort、seed或backend。首场景已独立
  完成`12/12` native targets，证明201帧接口可供IR-WM消费，但不构成质量结论。证据=
  `run://worldsim_v64/WS-V64-P6-CALIBRATION-SIDECAR-01/20260826T100000Z__calibration-prep-s0-r1`、
  `run://worldsim_v64/WS-V64-P6-CALIBRATION-SIDECAR-01/20260826T111500Z__calibration-native-scene-1045-s0-r1`、
  `https://github.com/ziyc/drivestudio`、`https://www.nuscenes.org/nuscenes?frame=0&sceneId=scene-0011&view=regular`。

- `V64-F14`（`engineering/operations`, `resolved_pre_quality_read`）：Windows PowerShell中的两个长驻feed lane反复调用
  短`ssh` readiness/publish命令；远端命令已经结束，但client继承PTY stdin后未退出，导致下一scene不推进。一次lane恢复时
  scene-0810的原远端worker仍在运行，新wrapper只在发现run leaf已存在时抛`FileExistsError`，没有覆盖或重复GPU计算；
  原worker随后正常完成`12/12`。OpenSSH官方手册明确后台/编排调用用`-n`禁止读取stdin；所有短检查、publish和wrapper调用
  加`-n`后lane连续推进，双worker达到100% GPU。防重复：不得因client挂起杀未知远端进程或新建重复run；先查remote PID/
  summary，再恢复缺失scene。证据=`https://man.openbsd.org/ssh`及P6逐scene run leaves。

- `V64-F15`（`algorithm/evaluation`, `resolved_by_new_version`）：冻结U3在16个独立calibration scene的192个case上没有任何正coverage
  通过case risk合同。最低5% coverage已有`41/192` failure，empirical risk=`0.213542`、simultaneous UCB=
  `0.292860`；night/vulnerable-transit分别`16/48`与`13/48`，所以不是Bonferroni或Clopper-Pearson过严。10%到50%
  coverage的failure继续增至`54,62,74,80,93`。根因边界是PCA16线性risk ranking不能跨新night/rain/construction/
  vulnerable场景提供case-level hidden-FREE控制；P5两scene AUROC通过不再足以解锁calibration/authority。confirmation target仍
  未读。禁止降低epsilon=0.05、提高conflict threshold=0.05、删stratum、读confirmation选策略或添加<5%事后coverage。
  合法复开必须是新模型版本：16个已消费scene只作development training，当前8个untouched scene作独立calibration，并先
  metadata-only冻结新confirmation。迁移依据=`https://proceedings.mlr.press/v97/geifman19a`、
  `https://proceedings.neurips.cc/paper/2019/hash/0c4b1eeb45c90b52bfb9d07943d855ab-Abstract.html`、
  `https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html`；closeout=
  `docs/autoresearch/worldsim_v64/P6_CASE_CALIBRATION_CLOSEOUT.md`。
  迁移在读取原confirmation前已冻结：16个已读scene仅作development，原8个quality-unread scene转独立calibration；新
  confirmation按剩余metadata-only pool/seed1固定为`1023,1105,0903,0451,0981,0537,0789,0157`。模型固定为完整
  273D的`128/64` focal-loss MLP且不做超参扫描。此冻结没有读取新quality、没有产生新failure ID；详见
  `docs/autoresearch/worldsim_v64/P6R_SELECTIVE_MLP_FREEZE.md`。
  第一阶段正式训练已完成：`786054` points，loss=`0.0337864->0.0251443`，development AUROC=`0.8811503`
  （仅描述），GPU fit=`10.1545 s`。模型现已冻结且原8-scene calibration仍未读；这既不关闭V64-F15，也不产生新
  failure。下一判定只来自预注册的96-case独立校准。
  独立证据现已在模型冻结后一次完成`8 scenes/96 units`，source-role overlap与query均为0；尚未读取模型分数或选择
  coverage，故V64-F15状态不变且没有新增failure ID。
  P6R独立评分随后以0.05--0.40 coverage全部得到`0/96` failure和simultaneous UCB=`0.048647`，选择最大通过40%；
  50%为`3/96`、UCB=`0.103218`而正确拒绝。故失败以“新模型版本解决”关闭；原PCA16线性U3负结论不改写，且新
  confirmation仍未读。证据=`run://worldsim_v64/WS-V64-P6R-CALIBRATION-01/20260826T141500Z__case-calibration-s0-r1`。
  exact-once confirmation只在冻结40%上读分一次，得到`1/96` failure；四strata分别`0/24,1/24,0/24,0/24`，
  总体和分层gate均通过。V64-F15因此以独立校准加新确认的完整新版本证据收口，但不产生现实安全声明。

- `V64-F16`（`resource/operations`, `resolved_by_scene_ready_streaming_and_catalog_finalize`）：exact-once新8 scene在现有raw cache均无payload；需要约24.8k sensor member。
  前一P6整批扫描虽已学习43033个member->shard映射，但`scan_shards`写回时只保留当前batch，故unseen batch仍会触发10个
  `.tgz`全扫，预计重现>1h GPU空转屏障。WIDS明确把index用于稀疏random access，ratarmount为compressed tar持久化SQLite
  index；结合当前代码不新增依赖，迁移为superset member->shard catalog，并以scene raw-ready为边界并发DriveStudio preprocess
  和最多两个IR-WM consumer。本批不可避免的一次scan完成后目录可复用；禁止重新裁catalog、等待整批processed才启GPU或用
  多卡掩盖I/O。证据=`https://github.com/webdataset/wids`、`https://github.com/mxmlnkn/ratarmount`和
  `docs/autoresearch/worldsim_v64/P6R_CONFIRMATION_EXECUTION_FREEZE.md`。
  下游exact-once合同已在target read前固定为40% policy、overall最多4/96且每stratum最多1/24 loss；不以本I/O失败
  改变科学gate。
  scene-ready priority scheduling已完成全部`8 scenes/96 blind targets`：先完成单shard/相关shard组，DriveStudio与IR-WM按
  scene流水，最大worker显存`4.1314 GiB`；没有等待整批processed或启用多卡。故GPU-idle/全批屏障部分已恢复；superset
  catalog的剩余EOF写回和可重建临时raw删除仍由原prep controller收口，完成后再把V64-F16标为resolved。
  exact-once评分后剩余scanner恢复并正常EOF：prep=`20260826T143000Z__confirmation-prep-s0-r1`完成8 scene，superset
  catalog=`57338 entries / 6880063 bytes`，temporary raw由controller删除。至此资源failure正式关闭；总wall `5872.4206 s`
  保留为首次稀疏scan成本证据，不把它误写成GPU wall。

- `V64-F17`（`data/interface`, `resolved_pre_score`）：exact-once evidence r1完成33/96 units后，scene-1105 frame62
  在`load_frame_boxes`用直接dict索引触发`KeyError`。processed审计显示该scene缺0--9、56--64的frame_instances键；这些
  frame在instances_info中逐一为0 annotation，`missing_with_annotations=[]`，不是sensor缺失或hidden target异常。nuScenes
  官方devkit对non-keyframe box使用相邻sample annotation插值，没有annotation时返回空/当前集合；故common loader把缺键解释为
  empty actor list。r2以hardlink复用33个完整NPZ、只算剩63；NPZ未存储的三个summary字段显式null，不伪造。禁止重算33、
  改scene/policy/gate或用target score挑恢复。证据=`https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py`
  和`docs/autoresearch/worldsim_v64/P6R_CONFIRMATION_EVIDENCE_RECOVERY_FREEZE.md`。
  r2 canonical=`20260826T152500Z__confirmation-evidence-s0-r2`按上述合同完成`96/96`，其中33 hardlink复用、63新算，
  query/role overlap均0，wall=`74.6360 s`；模型分数仍未读，故以pre-score状态关闭。
  随后的exact-once评分成功消费该证据一次且未触发第二次恢复，确认本interface failure没有改变冻结策略或coverage。

- `V64-F18`（`data/interface`, `resolved_pre_quality`）：P4C v1 metadata-only selection只检查nuScenes scene table、
  sample count与used-scene ledger，漏掉IR-WM train temporal pickle membership。7个scene完成blind native；`scene-0276`
  DriveStudio完成后在worker读取`payload["infos"][scene]`处`KeyError`，native output/target/model score均未读。官方BEVFormer/
  IR-WM使用生成的temporal train/val infos，不能用未分割scene table替代membership。恢复保留7个valid leaf并只替换无效scene：
  从commit `4813438`重建seed2 fallback；`scene-0572`是`flipped`误命中substring `ped`，首个token-valid且temporal-member
  的vulnerable候选为`scene-0813(631)`。不得重选其余7 scene、改policy/model/gate或读取quality挑替换。
  因v1 controller持有旧catalog snapshot，replacement写独立JSON并在两者结束后union，避免两个`os.replace` writer互相丢更新；
  证据=`https://github.com/APRIL-ZJU/IR-WM/blob/ir-wm/README.md`、`https://github.com/fundamentalvision/BEVFormer`和
  `docs/autoresearch/worldsim_v64/P4C_TEMPORAL_MEMBERSHIP_RECOVERY_FREEZE.md`。
  冻结replacement随后完成raw准备、DriveStudio与blind IR-WM native；corrected aggregate复用7个valid leaf并加入
  `scene-0813`，得到`8 scenes/96 targets/4423846027 bytes`，maximum worker peak=`4.1314 GiB`。全过程未读取confirmation
  target/quality/model score，未改变C0/M0、模型、gate或96-case denominator，故本interface failure在quality read前关闭。
  canonical=`run://worldsim_v64/WS-V64-P4C-CONDITIONAL-CONFIRMATION-SIDECAR-01/20260826T170000Z__native-aggregate-s0-r1`。
  corrected evidence随后一次完成`96/96 units`、query/source-role overlap=`0/0`且未触发同类membership或actor-frame错误；
  model score仍未读，没有新增failure ID。evidence=`run://worldsim_v64/WS-V64-P4C-CONDITIONAL-CONFIRMATION-EVIDENCE-01/20260826T171500Z__confirmation-evidence-s0-r1`。
  exact-once scorer随后只读冻结C0/M0一次，两臂均为`0/96` failure且M0 coverage uplift=`0.0750164`，三项gate全部通过；
  没有触发第二次恢复、mapping选择或新failure。本条保持`resolved_pre_quality`，并由fresh positive result确认恢复没有改变合同。
  下游P10M state-bake在该结果后冻结为target-free materialization：只读METHOD/native/model并让package-only consumer读取结果；
  未发现新blocker或新增failure ID，不回开本条。
  P10M formal随后一次完成96个package，M0比C0新增`74499`个emitted voxels且两项gate通过；state bake target read=false、
  runtime model/evidence access=false，没有新增failure ID。该结果不把voxel materialization外推为GS/sensor/collision authority。

- `V64-F19`（`integration/resource`, `resolved_by_sparse_gaussian_adapter`）：P10M fresh cohort的8个nuScenes scene均无同场景StreetGS/
  SceneIR checkpoint；旧V6 GS runtime只绑定其他scene，并以manifest SHA256/双bake bit-exact为入口，直接复用既不满足same-scene
  语义，也违反V6.4禁止新增hash/checksum/fingerprint的约束。不得把跨scene checkpoint硬接到fresh package、恢复旧hash治理，或把
  voxel package直接称为photorealistic GS。检索GaussianFormer的sparse semantic Gaussians、GaussianWorld的
  `{position,scale,rotation,semantic,feature}`表示与GaussianOcc的voxel-grid uniform scale/identity rotation后，冻结P10G最小迁移：
  每个M0 emitted voxel一个Gaussian，fixed `scale=0.256m/opacity=0.95/identity rotation`，GPU probabilistic BEV splat；只读P10M
  package，不读target/model/StreetGS。证据=`https://github.com/huang-yh/GaussianFormer`、
  `https://openaccess.thecvf.com/content/CVPR2025/papers/Zuo_GaussianWorld_Gaussian_World_Model_for_Streaming_3D_Occupancy_Prediction_CVPR_2025_paper.pdf`、
  `https://openaccess.thecvf.com/content/ICCV2025/papers/Gan_GaussianOcc_Fully_Self-supervised_and_Efficient_3D_Occupancy_Estimation_with_Gaussian_ICCV_2025_paper.pdf`。
  formal P10G一次完成`96/96` package，生成`534581`个M0 semantic Gaussians并在GPU BEV splat获得相对C0
  `+41016` support cells；target/model/StreetGS access均false，两项gate通过。因此“无法在no-hash/same-scene约束下进入任何
  Gaussian consumer”的integration blocker关闭；photorealistic StreetGS/sensor binding仍不在该恢复声明内。
  后续P10R直接冻结为logged future-lidar route corridor semantic consumer；只读P10G package与pose，不产生新failure ID，也不
  用route overlay冒充photorealistic或collision recovery。
  P10R formal在36/96 cases得到`+375` route support cells并通过冻结门，但C0/M0 binary intercept均为96/96、additional intercept
  cases=0；这是明确的metric saturation边界而非新implementation failure。不得把support gain包装成更多collision case被拦截。

- `V64-F20`（`evaluation/metric`, `resolved_by_route_local_cell_severity`）：P10R binary route intercept对C0/M0均为`96/96`，使case-level
  hit metric完全饱和；虽然M0在36 cases新增375 support cells，但不能回答这些新增state是否把hidden FREE写成OCCUPIED。不得事后缩短
  horizon、缩小corridor或提高density threshold制造未饱和case。Waymo Occupancy Flow以固定current-ego grid做cell-level occupancy
  metric，Implicit Occupancy Flow允许planner在连续时空点query，soft collision optimization使用连续势能而非binary hit；迁移为
  同一冻结2s/1.5m corridor上的route-local target hidden-FREE rate。policy/model/route均不改，只允许一次target audit；pooled M0
  conflict门保持原0.05，case failure只描述。证据=`https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/occupancy_flow_metrics.proto`、
  `https://openaccess.thecvf.com/content/CVPR2023/html/Agro_Implicit_Occupancy_Flow_Fields_for_Perception_and_Prediction_in_Self-Driving_CVPR_2023_paper.html`、
  `https://openaccess.thecvf.com/content/CVPR2023W/E2EAD/papers/Kedia_Integrated_Perception_and_Planning_for_Autonomous_Vehicle_Navigation_An_Optimization-Based_CVPRW_2023_paper.pdf`。
  P10C一次读取冻结target后得到M0 route emitted=`10013`、conflict=`43`、pooled rate=`0.004294`，并相对C0新增563 state；
  cell-level severity因此成功打破binary metric saturation，本条关闭。但5/96局部case仍超0.05，转入V64-F21，不由pooled pass覆盖。

- `V64-F21`（`evaluation/tail-risk`, `closed_negative_tail_authority`）：P10C pooled M0 route hidden-FREE rate虽仅
  `0.004294`，仍有5/96 case局部超过0.05，最高`scene-0895/f152=0.106383`；其余包括`0876/f047=0.076923`、
  `0876/f182=0.069444`、`0454/f122=0.063492`、`0895/f137=0.057692`。这阻止把pooled severity提升为route/collision
  authority。不得在已读target上调policy、改route或挑scene。CVaR对最坏尾部而非均值进行风险汇总，且PAC-Bayesian CVaR工作
  明确区分empirical tail与generalization bound；故只冻结alpha0.10/worst10 empirical audit，M0门仍0.05，不做优化或population
  声明。证据=`https://proceedings.neurips.cc/paper/2015/hash/64223ccf70bbb65a3a4aceac37e21016-Abstract.html`、
  `https://proceedings.neurips.cc/paper/2020/hash/d02e9bdc27a894e882fa0c9055c99722-Abstract.html`。
  P10T rows-only formal得到C0/M0 worst10 empirical CVaR=`0.0504298/0.0517085`，M0比C0高`0.0012787`且超过0.05；
  verdict=`rejected_empirical_route_tail`。本次未重读target、未改policy或tail fraction。故current M0 route/collision tail authority
  正式关闭并锁定P11；不得用P4C pooled/fresh pass、P10G support或P10R exposure覆盖该负结论。合法恢复必须是新版本、独立
  calibration与新confirmation，不能回调本次frozen M0。
  新版本恢复已冻结为P10R2/M1：把已消费P6R confirmation降格为development/calibration cohort，保持每case总selected count与
  M0相同，仅把route corridor名义覆盖限制到独立C0=`0.40`，并按原冻结risk score把释放预算重分配到non-route；模型、M0
  stratum coverage、2s/1.5m route与worst10 tail均不变。M1只可在该consumed cohort形成candidate，不能关闭本条；若两项冻结门
  通过，仍需metadata-only冻结全新temporal-member confirmation并exact-once确认。不得扫route cap、改尾部比例、重训模型，或用
  M1 calibration结果改写current M0 negative closeout。
  P10R2 formal一次完成：总coverage delta=`0`，route selected `5912->3826`、hidden-FREE conflict `23->9`，route worst10
  empirical CVaR `0.0220499->0.0114783`，M1最大case rate=`0.0454545`，两项candidate gate通过。该结果来自已消费cohort，
  因此只支持进入fresh confirmation；V64-F21仍保持`closed_negative_tail_authority`且P11仍锁定。下一步不得复用该cohort作确认，
  也不得因margin较大再扩route cap；只允许按metadata冻结未读质量的新temporal-member cohort并exact-once检验固定M1。
  fresh confirmation现已在任何target/model-score read前按seed3冻结为`1020,1016,0596,0590,0006,0472,0070,0371`，
  8/8均为IR-WM train temporal member且>=40 samples。选择只用description/name/count/index与当前124-scene排除集合；固定
  M1、2s/1.5m route、worst10和两项gate均未变。该prereg不关闭V64-F21；只有新96-case exact-once结果才能决定M1是否获得
  bounded fresh empirical route-tail authority，且无论结果如何都不改写历史M0负结论。

- `V64-F22`（`resource/operations`, `resolved_by_io_reassignment`）：P4C科学链已完成并推送后，两套cleanup controller仍为
  不同required-member集合重复顺序扫描同一10个official tar，持续占用NVMe且不产生新科学证据；这会把P10R2 fresh
  confirmation的scene-ready GPU feed推迟到可选catalog union之后。再次确认P4C native aggregate、evidence与exact-once summary
  均完整后，终止两个scanner tree；不把未完成union写成成功，而是明确放弃optional catalog enrichment。仅删除预注册为official
  tar可恢复的`worldsim_v64_p4c_raw_batch`与`worldsim_v64_p4c_replacement_raw_batch`（约6.8GiB），保留全部processed、native、
  evidence、model、run artifacts及已有`57338-entry/6880063-byte`catalog。I/O随后只服务P10R2一套新扫描与scene-ready feeder；
  不新增hash/checksum/fingerprint，也不改变任何科学policy/gate/result。

- `V64-F23`（`resource/scheduling`, `recovery_frozen_pre_target`）：P10R2 10-shard scan完成并成功流水化`0590/0596/0070`
  三个native leaf后，`scene-1020(778)`已由第二producer写成canonical processed，但对应feeder线程仍排在另一长耗时
  preprocess mutex后，形成head-of-line blocking并让GPU空闲。不得增加无关GPU filler或重算有效leaf。NVIDIA DALI明确以
  asynchronous pipelined execution和分离CPU/GPU prefetch queues隐藏阶段时延；迁移为同一feeder prefix的可恢复调度：启动先复用
  `passed=true,target_count=12` leaf，canonical processed直接绕过preprocess lock进入GPU semaphore。只丢弃当前可从raw重建的
  staging partial；cohort/model/policy/targets/gates/canonical IDs完全不变，target与model score仍未读。证据=
  `https://docs.nvidia.com/deeplearning/dali/user-guide/docs/pipeline.html`。
  ready-first恢复最终复用6个complete leaf，并只对`0006/0371`启动两个GPU worker；8 leaf全部12/12通过，本条关闭为
  `resolved_by_ready_first_resume`。

- `V64-F24`（`resource/operations`, `resolved_by_producer_single_owner`）：全shard scan结束后，prep主循环与feeder各自成为
  DriveStudio producer，先对不同scene并行有利，但随后同时开始`scene-0371(288)`，若均完成会竞争同一canonical目录。
  在任何duplicate canonical write前终止较晚的prep producer/tree，保留feeder较早staging、已落盘catalog和全部complete outputs。
  feeder第一次恢复还遇到`scene-0006`仅有run目录无summary的中断partial；确认无进程占用且仅1个未完成文件后精确删除并从
  complete canonical processed重建。最终8/8 processed、8/8 native、96 targets全部通过。prep以新r2和
  `--reuse-temporary-raw`只读8个complete canonical scene，`0.8171s`写summary并删除raw，不重扫tar/重做preprocess。
  防重复：scene-ready阶段只有feeder拥有producer写权；prep在stream结束后只作reuse finalize。科学cohort/policy/target lock未变。
  后续fresh evidence一次完成96/96 units、0 reuse与0 source-role overlap，未复发temporal membership、producer或partial问题；
  V64-F23/F24保持关闭，不新增failure ID。target现已读，故此后只允许预注册exact-once scorer，不得再改M1或cohort。

- `V64-F25`（`evaluation/generalization`, `resolved_exact_empirical_cohort_relative_confirmation`）：P10R2 prereg的绝对M1 route CVaR门在fresh
  96 cases通过（`0.0403133<=0.05`），总coverage严格保持，故formal verdict按合同为supported。但calibration中的相对改善没有
  确认：fresh M0 CVaR=`0.0391815`，M1-M0=`+0.0011318`；M1 pointwise failures从1增至2、maximum从`0.06818`升至
  `0.08333`。同时M1 route selected/conflicts从`8117/54`降到`4971/20`，说明绝对冲突质量下降但case-rate尾部受更小分母与
  稀疏离散事件支配。不得用absolute gate pass声称相对改善，也不得在已读confirmation上调route cap、tail fraction或挑case。
  合法下一步必须先检索denominator-stable sparse risk / occupancy-flow severity方法，再冻结rows-only诊断或新版本；P11 comparative
  authority保持锁定。current M0 P10T负结论与M1 absolute fresh pass分别保留，互不覆盖。
  检索Waymo Occupancy Flow fixed ego-grid cell metrics、Occupancy Flow Fields与Implicit Occupancy Flow后，冻结P10R3为
  `conflict count / route-eligible voxel count`的fixed-opportunity rows-only诊断，在consumed calibration与fresh confirmation分别
  使用同一worst10。该post-hoc诊断无confirmatory gate，不可关闭本条；只用于判断selected-only可变分母是否解释方向反转，且
  不得借结果回调M1或解锁P11。证据=`https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/occupancy_flow_metrics.proto`、
  `https://waymo.com/research/occupancy-flow-fields-for-motion-forecasting-in-autonomous-driving/`、
  `https://openaccess.thecvf.com/content/CVPR2023/papers/Agro_Implicit_Occupancy_Flow_Fields_for_Perception_and_Prediction_in_Self-Driving_CVPR_2023_paper.pdf`。
  P10R3 canonical rows-only结果在consumed calibration与fresh confirmation的固定分母worst10均为M1更低：
  `0.0132351->0.00455240`与`0.0216470->0.0149832`；pooled density也分别下降`0.00143870/0.00265563`。
  这使“selected-only可变分母导致方向反转”成为一致的描述性诊断，但P10R3是在读过fresh confirmation后冻结，不能作为独立
  confirmation；本条继续active，禁止据此回写P10R2 formal verdict或解锁P11。下一合法动作是先检索paired sparse-event
  confirmatory设计，再决定是否在从未读quality的test cohort冻结一次固定分母exact-once，不能复用已读cohort做显著性包装。
  检索NeurIPS 2021 `rliable`与ICLR 2024 Conformal Risk Control后，P10R4冻结一次untouched 96-case test：保留个体成对
  方向作描述，不做bootstrap/significance；CRC因目标是单调expected loss而不迁移为tail gate。三项confirmatory gate只包含
  coverage保持、fixed-denominator worst10不劣、pooled fixed density不劣。test未读前不设最小effect、不改M1；若失败则本条
  terminal rejected并保持P11锁定，若通过也只关闭exact empirical cohort层面的relative问题。输入I/O改为单遍metadata与
  raw-only producer/单feeder，避免GPU因重复8次`sample_data.json`扫描或duplicate preprocess owner空等。
  P10R4唯一untouched exact-once最终三门全过：coverage delta=`0`，fixed-opportunity worst10 M0/M1=
  `0.020725740/0.010821074`（delta=`-0.009904666`），pooled fixed density=`0.004944667/0.002001413`
  （delta=`-0.002943254`）；paired M1 lower/equal/higher=`18/78/0`。因此本条只在独立96-case exact empirical cohort的相对
  fixed-opportunity层面关闭，并解除由本条造成的P11 bounded-design锁。不得把它写成P10R2 selected-denominator formal重判，
  也不覆盖V64-F21对current M0的负结论；population、physical collision、planning、closed-loop与safety仍无authority。

- `V64-F26`（`io/execution`, `resolved_by_restricted_shards_and_dual_queue`）：P10R4首个raw-only入口发现`14437` required members均不在持久catalog，
  10个`.tgz`并发扫描约4分钟仅到`4--10%`，workers主要处于page wait且GPU尚无完整scene。CPython tarfile对gzip selected
  members仍需顺序流；ratarmount/rapidgzip可建seek-point index，但为一次性cohort新建10份index仍先消耗全量扫描。
  现有`71555`条semantic member→shard catalog显示七scene的capture prefix唯一落在05/06/08/10，`scene-0668`由相邻
  temporal range与已经原子落盘的exact-prefix files冻结到07。恢复只扫描`05,06,07,08,10`，保留all-shard尝试已完成文件，
  原workers停后只删除其`.partial.<pid>`；继续使用已运行的唯一feeder，不启动第二preprocess producer。若任一member找不到，
  restricted scan必须失败并回到未猜测的全量扫描，不得换scene或读quality。科学合同与test unread状态不变。
  restricted r1在进入scan前因既有resume目录仍执行`mkdir(exist_ok=false)`退出；该run不含新增archive读、preprocess、GPU或
  target read。恢复只把显式`--resume-raw-scan`的mkdir改为`exist_ok=true`，默认新run防覆盖不变；以新r2继续，不新增failure ID。
  r2使scene-0598 native以`45.4004s/4.1314GiB`完成，但单preprocess mutex下一scene转换超过2分钟，GPU再次出现供给缺口。
  按既有DALI分离CPU/GPU queue依据，停止feeder parent但让唯一in-flight scene-0462预处理完成，保留0598 native；同prefix
  feeder恢复为两个独立per-scene staging与`2 preprocess / 2 native` slots。不得对同scene启动第二owner；完整canonical/native
  必须reuse。科学合同与test unread不变，本恢复仍归V64-F26。
  canonical r2最终在`1807.8114s`扫描05/06/07/08/10并找齐`14437/14437`；per-shard命中
  `5401/1824/1818/1783/3611`也揭示capture prefix会跨archive boundary，但冻结五分片union完整。catalog增至
  `85992 entries`。raw完成时双队列已完成0598/0462 native且GPU峰值均`4.1314GiB`，故本条关闭；若后续native/evidence
  出现科学或独立工程故障应另记，不得重开全量tar scan。

- `V64-F27`（`io/execution`, `resolved_by_exact_stage_path_and_reuse`）：双preprocess独立target已完成scene-1084/1081，但DriveStudio实际将
  `..._processed_824`重写到`..._processed_10Hz_824/trainval/824`；feeder按常规append查找`..._824_10Hz`，因此在
  canonical install/native前抛出。824/821 stage分别有完整`1206/201`与`1176/196` images/lidar且无native partial。
  parent已停，唯一in-flight 424/522不终止、不重复；修复只镜像DriveStudio既有字符串重写。进程退出后四scene原子安装，
  night两scene用冻结underlying native command与原计划run dirs直接供GPU，patched feeder随后同prefix复用。若任一stage计数
  不完整则只重建该scene；不得删除完整stage、换scene或读test quality。
  恢复最终复用4个complete native leaf，并对其余4 scene完成同prefix native；最后两scene从stage ready到native启动仅等待
  `0.0646/0.0625s`。aggregate为`8 scenes / 96 targets / 4423846058 bytes / passed`，峰值worker显存`4.1314GiB`；
  test target/quality/model score仍未读。finalizer只登记8个complete canonical scene并删除可重建raw，故本条关闭；后续evidence或
  exact-once若失败必须按其实际阶段登记，不能重跑native或改cohort。

- `V64-F28`（`algorithm/evaluation`, `closed_negative_after_single_recovery`）：P11 bounded critic formal run按预注册只以selected-policy
  false-safe不劣与progress/stuck作gate，Real-only/naive/verified分别为`13/12/12` false-safe、progress均`1.0`、stuck均`0`，
  因此formal verdict合法为supported。但完整1248-action主指标揭示三臂unsafe recall仅`2.17%/0/1.09%`，false-safe=
  `180/184/182`；verified与naive的policy false-safe和reward完全相同，Brier/ECE反而更差。训练正例为`3/384`、
  `191/1152`、`96/768`，说明固定0.5输出在长尾与跨cohort下没有形成violation critic authority；这不是全刹车作弊，
  而是稀有unsafe识别塌缩。不得只引用三门PASS声称collision improvement，也不得在已读P10R4上调threshold、改lattice、
  重训或另跑同test。参考CVPR 2019 class-balanced loss、ICLR 2021 logit adjustment与Recovery RL后，唯一有界复开是保留
  已冻结模型，用从未生成action label的独立cohort解析选择一次unsafe-recall threshold，再在另一未读action-label cohort
  exact-once；若progress/stuck或recall失败则关闭P11，不训练大型NWM/RL。证据=
  `run://worldsim_v64/WS-V64-P11-BOUNDED-COLLISION-CRITIC-01/20260827T033000Z__bounded-collision-critic-s0-r1`。
  恢复已冻结为P11R：三critic不重训；P10R2 action labels只作独立calibration，每臂用unsafe score的20%分位解析选择
  target recall=0.80的单一threshold；threshold落盘后才允许生成P4C从未读取的action labels并exact-once。P10R4 labels、
  threshold grid、lattice/feature/model修改和第二evaluation均禁止。门只包含recall、policy false-safe不劣与progress/stuck。
  P11R最终threshold=`4.25e-18/0.191678/0.084891`；离散20%分位使naive/verified calibration recall均为`70/88=0.79545`，
  未静默改quantile或重跑。P4C evaluation中verified recall进一步降至`85/137=0.62044`；其policy false-safe/progress/stuck=
  `2/0.87240/0.11458`。Real-only threshold把全部action判unsafe，`96/96` fallback stop带来false-safe0但progress0/stuck1，说明
  recall与anti-trivial progress不能靠单一operating-point同时恢复。四门仅progress/stuck通过，P11R rejected，本条以negative
  terminal关闭P11；大型NWM/RL、再校准、换loss/model/lattice、第二evaluation均不解锁。后续只允许rows-only failure
  characterization和V6.4报告收口，不得创建新的P11科学attempt。
  P11D rows-only诊断进一步显示calibration→evaluation unsafe prior=`0.07051->0.10978`，verified unsafe q20/median score却下移
  `0.05328/0.13745`且safe median近乎不变；AP/AUROC=`0.24710/0.71165 -> 0.13740/0.56274`。这同时存在prior shift与
  unsafe ranking degradation，不支持“只换一个threshold即可恢复”的解释。该诊断无gate、无native/evidence reread，不改P11
  terminal；未来复开必须是新版本、新的可迁移violation representation与独立cohort，而不是本版本校准修补。

<a id="detail-v63"></a>

## V6.3 SurfNCC 防重复结论（2026-08-24）

- `V63-F01`（`engineering`, `resolved`）：P0 integration branch 的首次定向验证错误指向不存在的
  `tests/test_worldsim_v62_projection.py`，第二次改到真实文件后又因 pytest 进程未包含 repo-local `PYTHONPATH` 而在
  import collection 阶段触发 `ModuleNotFoundError: motion_proj`。两次均未读取数据/quality、未运行 GPU、未改变源码，
  不是 V6.2 projection 回归或 V6.3 算法失败。唯一恢复是在同一 conda 环境用
  `PYTHONPATH=. pytest -q tests/worldsim_v62/test_projection.py`，结果 `1 passed`。防重复：远端 repo 的定向 pytest
  必须使用真实 `tests/<version>/...` 路径并显式提供 repo-local import path；不为这一入口错误增加 smoke/regression
  矩阵。证据=`WS-V63-P0-SCOPE-GIT-01` 与 P0 shell terminal。

- `V63-F02`（`algorithm/evaluation`, `active`）：P2D canonical=
  `20260824T145924Z__native-pointwise-s0-r1` 用冻结 P5 best 与真实 per-cell IR-WM logits/BEV 执行 unchanged legacy28
  gate，Native B2仍为`4/28 ACCEPT,4/4 false-safe`，接受集合仍是scene-0242四个missing-route-support cases；R10=
  `2/3`、Actor gain=`0`、static/disocclusion gain=`2`、mask-area=`0.094024`、accepted FREE conflict mean/worst=
  `0.045783/0.092105`。source-valid UNKNOWN=`0.639211`，safe-OCC retention=`1.0`，hard violations=`0/939206`。
  与prototype P6/P6R相同的接受集合和false-safe说明V62-F05的feature bridge是加重因素而非主因；已推翻“只要恢复
  native feature，逐voxel CPSC即可获得hidden-surface authority”。防重复：不得对P2D重训、调threshold/seed/grid、
  用legacy O_eval选epoch或把mean conflict过门包装成安全。按预注册迁移到P3：Point Transformer V3官方实现支持
  efficient serialized point neighborhoods，visibility-aware reconstruction明确把FREE visibility作为surface约束，CVaR
  直接优化局部尾部；V6.3只迁移已在P1冻结的deterministic surface topology + patch CVaR，不改变alpha/cohort/gates。
  证据=`docs/autoresearch/worldsim_v63/P2D_NATIVE_POINTWISE_PREREG.md`、
  `https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.pdf`、
  `https://pmc.ncbi.nlm.nih.gov/articles/PMC4897344/`。

- `V63-F03`（`engineering`, `resolved`）：P3 probe r1=`20260824T150842Z__surface-probe-s20260824-r1`
  在`_native_occupied_target_grid`对长度`300/300/40`的三个target-grid axis arrays调用`numpy.stack`时触发
  `ValueError: all input arrays must have the same shape`。该返回值未被调用方消费，失败发生在surface extraction、
  target supervision与任何quality gate之前，run仅4 KB，不能写成surface方法失败。NumPy官方`stack`合同要求每个输入
  shape相同；恢复为返回三个独立axis arrays，并在同轮pre-run audit中把route-support类型更新限定到对应local surface、
  法向量统计限定为finite unit vectors、target grid超出native z范围的点显式标为invalid而非用`100% valid`作错误门禁。
  这些都是接口/统计修复，不改变proposal volume、6-connected topology、patch参数、cohort或科研门槛。r1不可覆盖；
  revision 2复用冻结配置。防重复：不同长度的坐标轴不得stack；native coverage必须作为显式映射事实交给后续模型处理，
  不得偷偷clip或删除proposal。证据=`docs/autoresearch/worldsim_v63/P3_SURFACE_CORPUS_PREREG.md`、
  `https://numpy.org/doc/2.0/reference/generated/numpy.stack.html`。

- `V63-F04`（`engineering`, `resolved`）：P3 probe r2=`20260824T151429Z__surface-probe-s20260824-r2`
  在runner入口触发`FileExistsError`：外层launcher先`mkdir`了叶run directory，runner为保护不可变run又显式拒绝已存在
  路径。0 unit、0 surface、0 quality read；不能解释为F03恢复失败或科研结果。Python官方`Path.mkdir`说明默认
  `exist_ok=False`时目标存在即抛`FileExistsError`。恢复只让launcher确保task父目录存在、把叶目录留给runner原子创建；
  不修改源码、配置或科研合同。r2目录和console保留，revision 3使用新路径。防重复：带immutable-run自建语义的runner
  不得由外层预建叶目录。证据=`docs/autoresearch/worldsim_v63/P3_SURFACE_CORPUS_PREREG.md`、
  `https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir`。

- `V63-F05`（`engineering/data-representation`, `resolved`）：P3 probe r3=
  `20260824T151618Z__surface-probe-s20260824-r3`首次完整产出`191 surfaces/498 patches/152226 points`，但101个
  微小static components的至少一点法向量无效（85个singleton，其余component size 3–11），令minimum normal-valid=
  `0`、probe未过。根因不是surface缺失：对称孤立voxel的六个外露面法向量相消，centroid fallback在离散medial-axis点
  也为零。Gradient-SDF说明SDF梯度给出normal但medial axis因最近surface不唯一而奇异；Open3D法向量接口要求需要时
  显式按camera location定向。恢复只在face-sum和centroid fallback都为零时，用target sensor viewpoint给出确定性单位
  方向，最后仅为sensor恰与点重合保留固定轴退路；不删除tiny proposal、不改变volume/topology/patch/cohort/gate。
  r3及其完整诊断保持不可变，r4使用新run。防重复：不得把tiny components静默过滤来换取normal-valid=1，也不得使用
  随机法向量。证据=`docs/autoresearch/worldsim_v63/P3_SURFACE_CORPUS_PREREG.md`、
  `https://openaccess.thecvf.com/content/CVPR2022/papers/Sommer_Gradient-SDF_A_Semi-Implicit_Surface_Representation_for_3D_Reconstruction_CVPR_2022_paper.pdf`、
  `https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html`。

- `V63-F06`（`engineering/protocol`, `resolved`）：P3 r4=`20260824T152300Z__surface-probe-s20260824-r4`虽以
  `minimum normal-valid=1.0`和8/8 negative contracts得到runner `passed=true`，但formal前对照P1冻结point encoder
  schema发现payload缺signed FREE/OCC distance、patch-local coordinate、method/target behind-hit与第四个temporal support，
  且`ray_hit_order`字段实际保存raw metric distance。该问题不会改变r4的geometry capability，却使其不足以喂给冻结
  SurfNCC，故不得把r4写成完整P3 pass。恢复使用SciPy exact Euclidean distance transform按0.2m sampling生成仅依赖
  method-visible evidence的signed distances；patch coordinate减冻结patch centroid；hit order在每个surface ray bundle内按
  distance+lexicographic tie-break归一化，并另存raw distance；同时显式补behind-hit、temporal UNKNOWN与actor observed-hit。
  r5=`20260824T152843Z__surface-probe-s20260824-r5`验证上述aggregate字段后，P4 loader审计继续发现aggregate counts无法
  执行冻结的整段temporal-window dropout；同一恢复因此再补每个method sweep的state/contradiction `[point,sweep]`
  矩阵，并用配置中的单一required-field清单防止再次静默漏项。VideoMAE与MaST-Pre支持结构化时间mask应保留时间维，
  但V6.3不迁移其高mask ratio/预训练目标。无target信息进入proposal/feature decision、无新超参或quality选择。
  防重复：capacity前必须逐字段对齐P1 schema；不能用字段名掩盖语义错位、用aggregate冒充per-sweep，或事后删掉冻结输入
  以让loader先跑。证据=
  `docs/autoresearch/worldsim_v63/P3_SURFACE_CORPUS_PREREG.md`、
  `https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html`、
  `https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf`、
  `https://proceedings.neurips.cc/paper_files/paper/2022/file/416f9cb3276121c42eebb86352a4354a-Paper-Conference.pdf`、
  `https://openaccess.thecvf.com/content/ICCV2023/papers/Shen_Masked_Spatio-Temporal_Structure_Prediction_for_Self-supervised_Learning_on_Point_Cloud_ICCV_2023_paper.pdf`。

- `V63-F07`（`engineering`, `resolved`）：r6 pre-run per-sweep窄检查首次凭scene名猜测processed path为
  `trainval/000`，在读取`instances_info.json`时触发`FileNotFoundError`；没有创建run或读取quality。冻结cohort是该映射的
  唯一事实源，实际`scene-0071 processed_index=68`；改用`trainval/068`后检查通过，state与contradiction均为
  `[3,300,300,40]`且逐voxel FREE+OCC+UNKNOWN count恒等于3。防重复：raw processed目录只按cohort metadata中的
  `processed_index`解析，不从scene display name猜目录；这一入口错误不扩展smoke/regression。证据=P3 r6 pre-run shell与
  `configs/worldsim_v62/p2_development_cohort_v1.yaml`。

- `V63-F08`（`engineering`, `resolved`）：P4尚未解锁执行时的temporary synthetic AMP interface r1在
  `binary_cross_entropy(sigmoid(hidden_free/authority))`触发PyTorch RuntimeError；128个随机点、无真实surface/quality、
  未创建正式run。PyTorch官方AMP文档明确说明BCELoss backward梯度可能无法用FP16表示，autocast因此主动拒绝，并要求
  使用`binary_cross_entropy_with_logits`。恢复同时输出hidden-FREE/authority logits供loss使用，推理概率仍为sigmoid；
  r2合成forward/backward finite，proposal-token gradient存在。防重复：FP16训练的二元head必须保留logits并在autocast
  下用BCE-with-logits，不通过禁用AMP或转FP32绕开冻结precision。证据=`docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://docs.pytorch.org/docs/stable/amp.html#prefer-binary-cross-entropy-with-logits-over-binary-cross-entropy`。

- `V63-F09`（`engineering`, `resolved`）：未来P5的packed-proposal synthetic r1把NumPy API名迁到PyTorch，调用不存在的
  `torch.flatnonzero`而在token selection前失败；128随机点、无真实surface/quality、无正式P5 run。PyTorch官方提供
  `torch.nonzero(input, as_tuple=False)`返回二维索引；对一维mask用`.squeeze(1)`得到所需索引。替换后r2完成2个proposal、
  4个patch的FP16 forward/backward，proposal CVaR shape=`[2]`且Transformer/proposal-token gradient非零。防重复：
  NumPy的`flatnonzero`不得假设存在于torch namespace；CUDA mask索引统一使用官方`torch.nonzero`/`torch.where`接口。
  证据=temporary packed-interface terminal、`https://docs.pytorch.org/docs/stable/generated/torch.nonzero.html`。

- `V63-F10`（`engineering/protocol`, `resolved_preexecution`）：P4/P5正式执行前对首40个P3完成单元做method-only结构读取，
  发现每个单元恰有一个surface超过冻结的8192-point microbatch（`40/40`，最大`173488` points）；这些大surface的完整
  patch set平均`297.45`、最大`417` tokens。H-P4-001只取最大proposal的首chunk，既未覆盖全部点，也只让每个chunk
  独立生成proposal token；这只能证明局部point memory，不能证明P1冻结的complete-proposal interaction。没有创建P4 run、
  启动GPU或读取target quality。H-P4-001因此在执行前withdrawn；H-P4-002保持同一两个unit、模型宽度、两步AdamW、
  accumulation4、CVaR/gate/resource不变，改为先按8192点编码完整patch，再把当前proposal的全部patch token汇合后运行
  两层attention与唯一proposal token。Set Transformer与Perceiver支持用小型set/latent bottleneck承接大输入；本迁移
  不增加learned token、删点、改分辨率或改denominator。未切块路径的12项输出模块化等价审计max abs diff=`0.0`。
  防重复：capacity不得用首chunk冒充完整proposal；point microbatch只能切point graph，proposal identity与patch context
  必须在上层重组。证据=`docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://proceedings.mlr.press/v97/lee19d.html`、`https://proceedings.mlr.press/v139/jaegle21a.html`。

- `V63-F11`（`engineering/protocol`, `resolved_preexecution`）：同一次P5执行前审计发现packed chunk曾各自计算actor/safe/
  unsafe标签、各自抽structural dropout，并在移除hard/temporal/actor-observed evidence后仍保留原`authority_bits`输入与
  原authority标签；selection还把各chunk hidden-FREE CVaR的最大值当完整proposal CVaR。这会让同一proposal跨chunk
  标签/selector漂移、从辅助authority通道看见已mask支持，并改变tail统计。没有真实P5 run、checkpoint或quality read。
  恢复把actor/safe/unsafe/full point count绑定完整proposal；每epoch/proposal只抽一个semantic selector并由所有chunk
  消费；遮蔽后从剩余method/temporal支持重算authority输入和监督，保留合法Actor current/swept与closure支持；selection
  汇合全部hidden-FREE点后精确计算alpha0.90 proposal CVaR，并统一为hard projection优先、仅learned low-authority OCC
  转UNKNOWN的最终decision后再统计coverage/UNKNOWN/accuracy。训练端明确保留内存受限的packed stochastic CVaR surrogate，
  不冒充exact full-batch optimizer。MAE支持“encoder不可见mask输入、原semantic target仍监督”；minibatch risk文献提示
  tail functional在小batch上可能有偏。防重复：任何evidence-derived辅助通道必须与mask同步；chunk-local统计不得冒充
  proposal统计。证据=`docs/autoresearch/worldsim_v63/P5_TRAIN_PREREG.md`、
  `https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html`、
  `https://arxiv.org/abs/2301.11724`、
  `https://openaccess.thecvf.com/content/CVPR2025/papers/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.pdf`。

- `V63-F12`（`engineering/protocol`, `resolved_preexecution`）：完整proposal context恢复后继续逐loss审计，发现ranking仍只在
  当前packed chunk恰好共现的safe/unsafe proposals间配对；跨chunk的大proposal会多次占用同一safe对照，而不共现的
  nearest-size pair永远没有loss。这违反P1冻结的complete-proposal、同actor/static stratum、nearest full-point-count
  一对一匹配。没有真实P5 run、checkpoint或quality read。Cross-Batch Memory证明小batch pair mining可由历史embedding
  扩展，但其stale queue与额外memory state在本项目无必要：每个unit的完整patch token set本来就小。恢复先从完整unit
  metadata一次性生成一对一pair，再用当前权重的detached完整patch-token cache运行可微proposal attention/risk head，
  每unit只施加一次全proposal ranking；point losses仍按8192 chunks有界。margin=`0.10`、weight=`0.25`、labels、cohort、
  optimizer与denominator均不变，也没有新增queue/momentum/hyperparameter。防重复：proposal-level pair loss不得由chunk
  共现关系定义；batching只能限制point graph，不能改变匹配集合。聚焦语义审计把safe/unsafe proposals置于两个不同chunk，
  仍得到冻结unit pair=`[(0,1)]`。证据=
  `docs/autoresearch/worldsim_v63/P5_TRAIN_PREREG.md`、
  `https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html`。

- `V63-F13`（`engineering/protocol`, `resolved_preexecution`）：batch invariance审计发现6-neighbor edge曾按surface identity
  构建，但一个大surface按patch边界进入多个point chunks时，跨chunk patch边会被静默删除；同一对patch偶尔共处一个chunk
  时该边又存在，因此输出依赖packing而非冻结几何。没有真实P4/P5 run、GPU结果或quality read。GraphSAINT明确指出induced
  subgraph minibatch丢失外部边会产生sampling bias；Point-BERT则提供local patch先编码为token、再由Transformer组合的成熟
  分层结构。V6.3不引入随机subgraph sampling、归一化估计或halo超参，而是把两层deterministic 6-neighbor aggregation的
  local neighborhood明确绑定到已冻结的完整patch；patch最大2048且从不切分，所以边集合与8192 packing无关，跨patch交互
  由完整proposal patch attention承担。proposal surface/6-connectivity、patch membership、模型层数、point features、cohort
  与denominator不变。防重复：任何point microbatch必须保持local encoder的计算单元完整；不能让edge存在性取决于邻patch
  是否碰巧同batch。两完整patch的聚焦语义审计得到full/split有向边数=`4/4`。证据=
  `docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://openreview.net/pdf?id=BJe8pkHFwS`、
  `https://openaccess.thecvf.com/content/CVPR2022/html/Yu_Point-BERT_Pre-Training_3D_Point_Cloud_Transformers_With_Masked_Point_Modeling_CVPR_2022_paper.html`。

- `V63-F14`（`engineering/protocol`, `resolved_preexecution`）：训练端已把matched ranking限定为完整scene/frame unit，
  但selection汇总曾把24个selection units的proposal rows一次交给全局nearest-size matcher，因此safe/unsafe可跨scene/frame
  配对；完整proposal risk虽正确，checkpoint objective仍会受跨案例规模巧合影响。没有真实P5 run、checkpoint或quality
  read。CVPR 2016 lifted structured embedding与CVPR 2022 graph sampling都说明pair mining的候选关系/采样边界是目标的一部分，
  不能把扩大候选池当作中性的batch实现。恢复仅按`(scene,target_frame)`分组执行原有actor/static、nearest full-point-count、
  one-to-one matcher，再对有pair的unit loss等权平均；margin=`0.10`、weight=`0.25`、proposal risk、cohort、threshold与gate
  均不变。聚焦synthetic把safe/unsafe分别置于两个unit得到`0 pair`。防重复：train/selection的proposal matching边界必须
  同为完整unit；不得跨case挖pair或引入memory queue。证据=`docs/autoresearch/worldsim_v63/P5_TRAIN_PREREG.md`、
  `https://openaccess.thecvf.com/content_cvpr_2016/html/Song_Deep_Metric_Learning_CVPR_2016_paper.html`、
  `https://openaccess.thecvf.com/content/CVPR2022/html/Liao_Graph_Sampling_Based_Deep_Metric_Learning_for_Generalizable_Person_Re-Identification_CVPR_2022_paper.html`。

- `V63-F15`（`engineering/evaluation`, `resolved_preexecution`）：P4已有`cvar_gradient_nonzero` gate曾在total loss backward后
  检查`hidden_free_head.weight.grad`，但该head同时受BCE-with-logits监督；即使proposal CVaR图断开，BCE也足以让flag
  为true，造成capacity假阳性。没有真实P4 run、quality read或scientific denominator。CVaR优化的一手工作明确把risk
  objective的梯度作为优化对象，PyTorch官方`autograd.grad`提供指定outputs到inputs的直接VJP。恢复在原forward图上对
  `proposal_cvar.mean()`分别向state/hidden-free/authority heads求梯度并只接受finite nonzero direct path；聚焦synthetic
  三条head路径均通过。原gate名称、阈值、units、steps、模型与资源合同不变，也未新增回归矩阵。防重复：多项loss共享
  parameter时，不得以总梯度证明某个特定loss已连通。证据=`docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html`、
  `https://proceedings.mlr.press/v235/kim24x.html`。

- `V63-F16`（`data/evaluation`, `resolved`）：P3 formal终态前语义审计发现surface registry与summary字段
  `hidden_free_count/hidden_free_point_count`实际只计算`target_state==FREE`，没有同时要求
  `method_state==UNKNOWN && !method_contradiction`，所以该描述统计不能按hidden-FREE引用。point NPZ中的method/target/
  contradiction、proposal/patch/native features均正确；P4不消费此计数，P5 training/selection从point arrays使用正确布尔
  条件，因此不是标签污染或语料重建失败。NeurIPS dataset documentation实践强调保留旧版本并显式记录metadata限制。
  恢复边界：不改写formal artifact；terminal后从72个原始NPZ一次重算得到target FREE/OCC/UNKNOWN=
  `1545584/335050/9702367`、correct hidden-FREE=`688837`，已在三本账与P3 prereg登记勘误；未来materializer
  以additive v2把`target_free/occupied/unknown`与正确`hidden_free`分字段输出。
  canonical r6 probe一次重算得到target FREE/OCC/UNKNOWN=`19609/3891/128726`、correct hidden-FREE=`8311`，确认旧值
  `19609`只是target-FREE。
  禁止因描述字段误名重跑13.213小时正确point corpus，也禁止继续引用旧summary的hidden-FREE数字。证据=
  `docs/autoresearch/worldsim_v63/P3_SURFACE_CORPUS_PREREG.md`、
  `https://arxiv.org/abs/1803.09010`、
  `https://papers.neurips.cc/paper_files/paper/2024/file/605bbd006beee7e0589a51d6a50dcae1-Supplemental-Datasets_and_Benchmarks_Track.pdf`。

- `V63-F17`（`engineering/numerics`, `resolved`）：H-P4-002 r1 canonical=
  `20260825T045854Z__capacity-h002-s0-r1`在11.181s完成全部2 train/2 selection complete proposals，peak仅
  `0.196070 GiB`，finite loss、direct CVaR三head gradient、proposal-token gradient、hard violations=`0`、checkpoint
  reload与selection finite均成立；但unscale后的total gradients含nonfinite，且same-model/reloaded FP16 forward max abs
  difference均为`9.059906e-6`，所以冻结finite与exact-zero determinism gate诚实未过。没有quality conclusion、calibration、
  confirmation或test read，不是资源/算法失败。PyTorch官方AMP文档说明默认initial scale可能使FP16 gradient overflow；
  reproducibility文档说明CUDA SDPA不同backend/backward确定性不同，math backend配合deterministic algorithms可确定执行。
  唯一有界r2恢复保留FP16但固定GradScaler initial scale=`1024`，禁用flash/memory-efficient SDPA、只启用math SDPA并开启
  deterministic algorithms。模型、units、dropout、loss、optimizer LR/WD、2 steps、accum4、gate与22GiB ceiling均不变。
  r3=`20260825T051200Z__capacity-h002-s0-r3`在同一合同下以finite gradient与exact-zero repeat/reload正式通过。
  防重复：不得放宽exact-zero阈值、忽略nonfinite flag、增加steps或把r1写成quality negative。
  证据=`docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://docs.pytorch.org/docs/stable/amp.html`、
  `https://docs.pytorch.org/docs/stable/notes/randomness.html`。

- `V63-F18`（`engineering/runtime`, `resolved`）：H-P4-002 r2 canonical=
  `20260825T050400Z__capacity-h002-s0-r2`在第一个CUDA math attention forward处被PyTorch deterministic-algorithm runtime
  拒绝：CUDA>=10.2的cuBLAS矩阵运算只有在进程启动前设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`或`:16:8`后才允许确定执行。
  r2在任何optimizer step、capacity summary、quality/calibration/confirmation/test read之前终止，run leaf为空；因此它没有检验
  F17的AMP-scale或exact-zero恢复，不是第二个科研/数值尝试。NVIDIA cuBLAS官方结果可重复性说明`:4096:8`会固定workspace
  配置且增加约24 MiB，PyTorch deterministic文档对CUDA matmul给出同一前置条件。恢复把`:4096:8`同时绑定到launcher和
  runner的pre-torch-import环境，并由P4/P5配置显式记录；新增开销远低于22 GiB ceiling。r3继续F17的同一次有界恢复，
  model/data/FP16/AMP scale/SDPA backend/dropout/loss/optimizer/steps/accum/gates均不变。防重复：不得关闭determinism、放宽
  exact-zero门或把入口异常写成capacity/quality失败。r3实际peak=`0.256589 GiB`、wall=`11.863s`并正式passed，说明环境
  恢复闭合而无需新增资源或协议变化。证据=`docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility`、
  `https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html`。

- `V63-F19`（`algorithm/evaluation`, `resolved_by_constrained_recovery_p6_unlocked`）：P5 canonical=
  `run://worldsim_v63/WS-V63-P5-SURFNCC-TRAIN-01/20260825T051530Z__surfncc-train-s0-r1`完成全部48个train与24个
  scene-disjoint selection units，`7 epochs/1792 steps`、finite training、peak=`0.403084 GiB`且累计hard violations=`0`；
  runner据此正确报告capacity/training `passed=true`。但冻结lexicographic objective选择的epoch 3仅是best
  training-objective checkpoint：safe-OCC retention=`0`、emitted-OCC coverage=`0.0371977<0.10`、source-valid UNKNOWN=
  `0.861807>0.60`。该checkpoint把有正向OCC支持的安全曲面与危险/缺证据曲面一起拒绝，不能作为SurfNCC candidate，
  不能用低false-safe或低tail掩盖零仿真效用。这是positive-authority collapse症状；现有证据尚不足以区分representation/
  supervision重叠、raw/post-projection decision composition或weighted-objective optimization collapse，故不得提前把根因写成
  ordinary underfit或任一优化结论。

  P5D H002 canonical=`run://worldsim_v63/WS-V63-P5D-AUTHORITY-COLLAPSE-DIAGNOSTIC-01/20260825T084844Z__authority-diagnostic-s0-r2`
  已把根因收敛：safe-OCC raw/projected/post-authority decision均为实际`[FREE,OCCUPIED,UNKNOWN]=[153,0,62301]`且
  authority veto=`0`，排除hard projection与decision composition；raw `P(OCC)`虽以AUC=`0.722684`保留弱排序，绝对
  mean仅`0.006459`，`q_AUTH` AUC也仅`0.578070`。weighted tail/retention gradient mean比=`5.531x`，direct-tail与
  state-head比分别=`1.715x/1.732x`，tail-retention cosine mean=`-0.411568`；retention loss mean=`0.968547`。
  因此primary root确认为weighted-objective optimization collapse，evidence-authority supervision弱对齐为次级机制，
  而不是solver或authority veto失败。

  P5R canonical=
  `run://worldsim_v63/WS-V63-P5R-CONSTRAINED-SURFNCC-TRAIN-01/20260825T091631Z__constrained-train-s0-r1`
  以同一SurfNCC representation、数据、hard projection、seed0与P5 epoch3 model-only warm-start运行proxy primal-dual；
  retention/emitted-OCC/non-UNKNOWN改为约束，旧weighted retention term置0。formal完成`10 epochs/2560 steps`、finite、
  hard violations=`0`，未读P6/calibration/H/T。best feasible epoch6的retention=`0.721226`、coverage=`0.114148`、
  non-UNKNOWN=`0.686101`，四项exact gate全过，tail+rank=`0.520541`，因此`candidate_promotable=true`并解锁P6。
  epoch 7–9连续三轮没有更优feasible candidate后按patience停止；尤其epoch 8/9虽tail更低但coverage/UNKNOWN失门，未覆盖
  epoch6。由此F19的positive-authority collapse已由约束优化闭合，而不是靠降低gate或回改solver闭合。

  防重复：不得增加epoch、换seed、加大模型、改变CVaR alpha、降低retention/coverage/UNKNOWN gate、提高
  `lambda_ret`，也不得回改已连续零违反的FREE/OCC projection、ray hard constraint、lifecycle或V6.2 solver。P5R不再追加
  recovery/sweep；合法下一步仅为冻结best candidate进入原P6 fresh matched AB。P6必须保留Native B2、surface encoder、
  CVaR与authority消融及原晋级门；P5R的candidate pass不能冒充P6/校准/confirmation/deployment结论。
  证据=`docs/autoresearch/worldsim_v63/P5_TRAIN_PREREG.md`、
  `docs/autoresearch/worldsim_v63/P5D_AUTHORITY_COLLAPSE_DIAGNOSTIC_PREREG.md`、
  `configs/worldsim_v63/p5d_authority_collapse_diagnostic_v1.yaml`、
  `scripts/run_worldsim_v63_p5d_authority_diagnostic.py`、
  `configs/worldsim_v63/p5r_constrained_surfncc_train_v1.yaml`、
  `scripts/run_worldsim_v63_p5r_constrained_train.py`、
  `https://proceedings.mlr.press/v97/geifman19a.html`、`https://proceedings.mlr.press/v98/cotter19a.html`、
  `https://proceedings.mlr.press/v97/cotter19b.html`。

- `V63-F20`（`engineering/runtime`, `resolved`）：H-P5D-001第一次formal入口在创建run leaf、读取P5
  checkpoint/train arrays或建立CUDA context前，将`shutil.disk_usage`直接调用于尚不存在的
  `/root/autodl-tmp/runs/worldsim_v63/WS-V63-P5D-AUTHORITY-COLLAPSE-DIAGNOSTIC-01`，触发`FileNotFoundError`。
  新task namespace按设计尚未由runner创建，所以canonical run=`null`，没有任何分布、梯度或科学结论。Python官方
  `shutil.disk_usage(path)`要求path指向已有filesystem位置；`Path.mkdir(parents=True)`才负责创建缺失父目录。
  H-P5D-002只在disk check前向上寻找最近已存在父目录并执行相同20 GiB资源检查，之后仍由formal runner创建唯一leaf；
  checkpoint、48-unit分布、4-unit gradient probe、模型/FP16、threshold/gate、零optimizer与P6/H/T locks全部不变。
  防重复：新task resource check不得假设namespace已存在，也不得为了通过检查预创建并冒充failed/canonical run；本恢复不
  增加smoke或质量读取。证据=`scripts/run_worldsim_v63_p5d_authority_diagnostic.py`、
  `https://docs.python.org/3/library/shutil.html#shutil.disk_usage`、
  `https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir`。

- `V63-F21`（`evaluation/metadata`, `resolved`）：P5D H002 canonical的`DECISION_STAGE_COUNTS.json`把
  `class_order`描述文字写成`UNKNOWN/FREE/OCCUPIED`，但生成counts的`torch.bincount`直接以argmax class index为bin，冻结
  project constants实际为`FREE_INDEX=0/OCCUPIED_INDEX=1/UNKNOWN_INDEX=2`。因此三个counts数组本身、raw/projected/
  post-authority等值关系、authority veto=`0`、全部distribution/gradient/summary均正确；错误只在数组标签文字。正确
  safe-OCC counts为`FREE/OCCUPIED/UNKNOWN=153/0/62301`，不是旧文字顺序的解释。canonical artifact保持不可变，runner
  future label已改成`FREE/OCCUPIED/UNKNOWN`；不为13分钟正确诊断重跑，也不改写artifact。防重复：任何class-count数组
  必须从同模块index constants生成或明确按constants记录order，不凭tri-state自然语言习惯手写顺序。证据=
  `motion_proj/worldsim_v62/projection.py`、`scripts/run_worldsim_v63_p5d_authority_diagnostic.py`、
  `https://docs.pytorch.org/docs/stable/generated/torch.argmax.html`、
  `https://docs.pytorch.org/docs/stable/generated/torch.bincount.html`。

- `V63-F22`（`engineering/operations`, `resolved`）：P5R terminal文档收口的首次SSH备份命令在本地PowerShell双引号中
  使用远端`$b`，变量在发送前被本地展开，导致远端备份在复制前失败并在项目外创建`/docs`重复副本树；随后一次包含
  `$(realpath /docs)`的保护命令也先被本地PowerShell解释并在任何删除动作前失败。两次均未修改
  `/root/autodl-tmp/motion_proj`、canonical run、checkpoint或Git工作树。只读`find`确认`/docs`全部是可由原仓库恢复的
  重复副本后，以显式绝对目标删除该树；随后不用变量或命令替换，以显式
  `/tmp/worldsim_v63_pre_p5rclose_20260825T1440Z`成功备份七个文档。P6 prereg同步后的inline `python -c` YAML检查又因
  同一PowerShell→SSH引号层在读文件前`SyntaxError`；改为将只读Python源码经stdin传给远端解释器后验证通过，项目仍未变。
  防重复：从PowerShell发送SSH文件操作时，不在双引号命令中使用远端`$var`、`$(...)`或嵌套inline Python字符串；备份、
  清理目标使用已解析的显式绝对路径并拆成独立步骤，结构验证统一经stdin发送。

- `V63-F23`（`engineering/runtime`, `resolved_pre_quality_read`）：P6 B0/B1/B2首次formal入口直接执行
  `python scripts/run_worldsim_v63_p6_development_ab.py`，解释器按官方合同只把输入脚本所在`scripts/`目录置于module search
  path首位，因而在首个project import触发`ModuleNotFoundError: motion_proj`。失败发生在run leaf创建、P3/native数据、B2
  checkpoint与CUDA context之前，canonical run=`null`，没有P6 quality或科学结果。恢复不改源码/配置/denominator/gate，
  只从repo root改用`python -m scripts.run_worldsim_v63_p6_development_ab`，使当前目录进入module path；同解释器`--help`
  入口验证通过。防重复：repo-local runner若import project package或兄弟`scripts` module，formal launcher统一用`python -m`
  或已安装console entry point，不把direct-file import失败登记为算法reject，也不为此扩展smoke矩阵。证据=
  `https://docs.python.org/3/library/sys_path_init.html`、
  `https://packaging.python.org/en/latest/guides/creating-and-packaging-command-line-tools/`。

- `V63-F24`（`algorithm/evaluation`, `active route-closed`）：P6 B3 Surface-Mean虽在训练内冻结epoch1 feasible checkpoint
  （hard0、retention=`0.636863`、OCC coverage=`0.285326`、UNKNOWN=`0.550411`），但统一逐scene stage evaluator在两scene
  都不优于冻结Native B2。scene-0450 common surface hidden-FREE CVaR=`0.596685 vs 0.497850`，相对改善=
  `-19.852%`，accepted area ratio=`0.406270`且source-valid UNKNOWN=`0.651678>0.60`；scene-1089 tail=
  `0.655861 vs 0.465122`，改善=`-41.008%`，area ratio=`0.499323`。两scene hard0、retention、case、actor/static过门，
  说明失败不是hard solver回归或all-UNKNOWN，而是surface architecture在保留一定OCC后仍同时放大hidden-FREE tail并显著
  收缩相对Native B2的写入面积。supporting scenes=`0/2`，H-P6-001 rejected。

  主计划Stop2因此关闭surface architecture family：B4 Surface-Max、B5 Surface-CVaR和M0 authority均不执行，H-P6-002/
  H-P6-003关闭未读，P7没有frozen P6 M0输入而保持locked；legacy/calibration/confirmation/test均未读。不得用pooled
  retention、训练内candidate、较高accuracy或hard0掩盖逐scene tail/area失败，也不得换seed、加大模型、改CVaR alpha、
  降低area/UNKNOWN/2%门或先读legacy/H/T复开。未来合法复开必须在新版本预注册feature-level aleatoric/epistemic
  uncertainty与scene/stratum-conditional coverage约束，并使用fresh development denominator；相关候选仅为EvOcc
  （CVPR 2025）、ReliOcc（IJCAI 2025）、OCCUQ（ICRA 2025开源）及UAI 2024 conditional robust optimization，不构成
  V6.3 recovery授权。证据=`docs/autoresearch/worldsim_v63/P6_SURFACE_FAMILY_CLOSEOUT.md`、
  `run://worldsim_v63/WS-V63-P6-DEVELOPMENT-AB-01/20260826T014500Z__b3-eval-s0-r1`、
  `https://openaccess.thecvf.com/content/CVPR2025/html/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.html`、
  `https://www.ijcai.org/proceedings/2025/220`、`https://github.com/ika-rwth-aachen/OCCUQ`、
  `https://proceedings.mlr.press/v244/chenreddy24a.html`。

<a id="detail-v62"></a>

## V6.2 CPSC 防重复结论（2026-08-24）

- `V62-F01`（`algorithm/evaluation`, `active`）：V6.1 oracle Occupancy 在 legacy28 得到 `10/28 ACCEPT` 且
  `0 false-safe`，说明物理状态补全存在真实上界；GaussianWorld 和 IR-WM 的 learned argmax Occupancy 都得到同一
  `10/28` 表面支持，但各自接受项全部为 `10/10 false-safe`。已确认的共同根因是 dense learned prior 能覆盖 proposal，
  却没有把真实 observed FREE 当作不可违反的前向约束；这推翻“感知型 argmax Occupancy 可直接成为 world compiler
  物理权威”，不推翻 learned prior 作为软信息源。防重复：不得以第三 backend、confidence threshold、entropy、grid、
  history window、checkpoint、verifier 放宽或 observed-FREE 事后全 veto 复开 V6.1。合法复开仅限 V6.2 CPSC：方法输入内
  hard FREE/OCC、contradiction→UNKNOWN、可推翻 prior、anti-trivial coverage 和独立 false-safe 评测；若 B1 hard clip
  已达 `>=5/28, 0 false-safe`，应诚实转为 projection-only compiler。证据=`V61-F11,V61-F13`、
  `docs/autoresearch/worldsim_v61/V61_MINIMUM_EXPERIMENT_CLOSEOUT.md`、
  `docs/WORLDSIM_V6_2_CONSTRAINT_AWARE_PHYSICAL_STATE_COMPLETION_PLAN.md`。
- `V62-F02`（`data/protocol`, `resolved`）：P2 r1 query probe 的 `method/target_state` 沿用 V6.1 evidence 编码
  `UNKNOWN/FREE/OCCUPIED=0/1/2`，而 P3 model distribution 固定 `FREE/OCCUPIED/UNKNOWN=0/1/2`；字段名未显式区分，
  若直接训练会静默互换 UNKNOWN/FREE/OCC 标签。r1 只做 CPU 资源/池探针，未训练、未产出科学结果或 formal dataset。
  r2 在任何 formal materialization 前把字段拆成 `*_evidence_state` 与 remapped `*_class_index`，并确认两者范围0..2；
  canonical probe=`20260824T082318Z__query-probe-s20260824-r2`。防重复：loader 只能把 `target_class_index` 送入
  three-state loss，把 `*_evidence_state` 作为 hard-evidence feature；禁止依赖裸整数碰巧相同或在 loader 中无名 remap。

- `V62-F03`（`data/algorithm`, `resolved`）：P2 formal r1 在 `scene-1012/f152` 因 instantaneous
  `actor_envelope` pool=`0` 停止；该帧并非没有 actor，而是4个当前 actor 全在冻结 ROI 外，其中一个 actor 在可见
  method sweep `f146` 仍穿过 ROI。只按 target-frame box 构造 actor pool 推翻了“每个冻结 target 当前 ROI 都含 actor”
  的隐含假设，也会诱使实现删除固定的15k actor query。参考 QueryOcc 的相邻时刻独立4D查询以及动态稀疏 query 的
  时序传播，恢复方案把 actor query support 固定为 `current target envelope ∪ visible method-sweep envelopes`；时序包络
  只定义 query support，不升级为 hard OCC evidence、不读取 dropout/target evidence，也不挪用 actor quota。定点复现
  `20260824T083403Z__actor-sweep-repro-s20260824-r5`：current=`0`、visible swept=`450` voxels、actor-type query=
  `15000/15000`、total=`100000`、exit=`0`。防重复：不得因某个 target 当前 ROI 无 actor 而删 unit、删 actor query、
  改 ROI 或把 target evidence 当 method input；若 visible method sweep 也无 actor support，必须登记新的 cohort-level
  事实并重新审视 actor-query定义，不能静默转采 EASY FREE。formal r1=
  `20260824T082601Z__query-dataset-s20260824-r1`，未完成 manifest、未用于训练或质量结论。恢复后的 formal r2=
  `20260824T083654Z__query-dataset-s20260824-r2` 已完成72/72 units、7.2M queries，combined actor pool 0空、
  source-role overlap=0；该成功不新增 failure ID。

- `V62-F04`（`engineering`, `resolved`）：P4 probe r1 在 official IR-WM plugin import 阶段、GPU forward 与任何
  sidecar 写入前失败；隔离 Python 可执行文件虽来自 `worldsim-v61-irwm`，controller 却继承外层 shell PATH，导致
  PyTorch `cpp_extension.load()` 的 `verify_ninja_availability()` 找不到 env 内已安装的 `bin/ninja`。这不是缺依赖、
  CUDA 不兼容、IR-WM 方法失败或数据失败。PyTorch 官方实现明确通过 PATH 调用 `ninja --version`，V6.1 已成功的
  IR-WM controller 也显式 prepend env `bin` 并冻结 `TORCH_CUDA_ARCH_LIST=8.6`。恢复仅复用同一环境合同：prepend
  env bin，设置 `PYTHONNOUSERSITE=1`、OMP/MKL threads、CUDA device与SM 8.6；不安装包、不改模型/输入/query或门槛。
  failed probe=`20260824T085711Z__prior-sidecar-probe-s1-r1`，无科学输出；防重复：用隔离 Python 启动 native/CUDA
  worker时不得假设其 bin 自动进入 PATH，也不得把 loader import failure记成方法 rejection。恢复后的同输入 r2=
  `20260824T085956Z__prior-sidecar-probe-s1-r2` 已产生100k query-aligned sidecar，peak=`4.05GiB`、target evidence
  read=`false`；因此 F04 保持 resolved，不新增 recovery。

- `V62-F05`（`data/protocol`, `resolved for artifact-bounded P6`）：P6 接口审计推翻“V6.1 已冻结P5可直接消费的
  prior sidecar”。canonical ME3R 的四个IR-WM输出仅有argmax `class_label[200,200,16]`、occupied mask和网格/pose，
  没有17 logits或256D BEV；禁止重跑IR-WM，故逐cell uncertainty/latent不可精确恢复。阶段表还有第二处冲突：B2需要
  Tier-C校准阈值、B4需要未训练的no-evidence-dropout checkpoint、full M0需要P8 grouped conformal，三者在P6均不存在。
  防重复：不得从硬label伪造逐cell置信度、用legacy O_eval拟合adapter/threshold、重跑backbone、补训多臂或把B5冒充
  conformal M0。参考CVPR 2022 ProtoSeg的非参数训练特征均值，恢复只用P5 train split按17 class求query-weighted
  logits/BEV prototype并查表，P5保持frozen；24-unit只读失真审计agreement=`0.896898`、bridge hidden-FREE=
  `0.399349`、safe-OCC=`0.872897`、hard violation=`0`。合法P6只比较B0/B1/B3/B5，明确B2/B4 unavailable、M0 defer
  P8；bridge claim始终是lossy artifact transfer，不是native sidecar parity。证据=`P6_LEGACY_INTERFACE.md`、
  `configs/worldsim_v62/p6_legacy28_v1.yaml`、`motion_proj/worldsim_v62/legacy_bridge.py`。

- `V62-F06`（`algorithm/evaluation`, `active; recovery exhausted`）：P6 canonical=`20260824T095529Z__legacy28-s0-r1` 在同一28-case上
  B5仅`4/28 ACCEPT`且`4/4 false-safe`，mask-area=`0.09402`、R10=`2/3`、Actor新增=`0`；source-valid UNKNOWN=
  `0.82735` 超过0.50。B3与B5 case decision完全相同；B5 hard projection仍是`0/939206`违规，oracle accepted surface
  safe-OCC retention=`1.0`。B1虽把accepted FREE conflict从B0 mean/worst=`0.26748/0.57057`降到
  `0.05058/0.11722`，但仍`10/10 false-safe`，所以简单hard clip既不安全也未触发Stop 1。根因边界：argmax-only
  prototype input造成严重missing-feature shift，evidential head高UNKNOWN仍保留4个hidden-unsafe route surface；局部
  projection只能保证观测cell，不能恢复丢失特征或证明隐藏表面。禁止用本次O_eval调threshold/prototype、改grid/window、
  重跑IR-WM、删case、放松UNKNOWN/FREE gate或改选第二backend。唯一合法复开=`P6R evidence dropout`：依据CVPR 2022
  Modality-Agnostic Learning，只用P2/P4 train模拟`p=0.5` prototype feature loss，并由frozen full-view P5 teacher做
  `0.25 KL`一致性；相同P6 gate一次性复评。若P6R失败，CPSC-Lite关闭，不再换projection/set-valued recovery。
  P6R formal r2=`20260824T101705Z__feature-dropout-train-s0-r2` 已按冻结复合目标选best epoch2；objective改善但
  prototype hidden-FREE false-OCC为`0.41441`，尚未解除本条。只有未改门槛的legacy28 recovery可以裁决F06，训练
  selection不能替代false-safe结果。
  唯一P6R legacy recovery=`20260824T102709Z__feature-dropout-legacy28-s0-r1` 仍为`4/28 ACCEPT,4/4 false-safe`，
  接受集合完全相同；UNKNOWN虽从`0.827351`降到`0.638518`，仍超过0.50，mask-area=`0.094024`、R10=`2/3`、Actor
  gain=`0`、worst FREE conflict=`0.087379`。因此missing-feature exposure缓解abstention但没有建立hidden-surface
  authority，本条从“允许唯一recovery”更新为“recovery exhausted / family closed”。后续不得选择projection architecture
  或set-valued head作为第二recovery，也不得绕行P7/P8 calibration。未来新版本复开至少要求native per-voxel logits/features、
  独立calibration cohort与直接hidden-surface false-safe risk supervision，并在任何legacy评分前重新scope-freeze。

- `V62-F07`（`engineering`, `resolved`）：P6R首次formal入口
  `20260824T101047Z__feature-dropout-train-s0-r1` 从source=`d8f69d0`创建run后，在pure-prototype baseline selection
  的首个`compute_cpsc_losses`调用触发`KeyError: prior_tristate`；optimizer steps=`0`、checkpoint=`0`、legacy O_eval
  read=`0`，因此不是训练不稳定或机制rejection。根因是recovery runner自定义mapping batch漏传原P5 prior-preserve
  loss需要的字段。PyTorch官方`torch.utils.data`说明mapping sample/batch由collation保留键，调用方必须完整提供约定字段；
  恢复前已核对loss的全部batch访问，确认没有第二个遗漏键。修复不能把full-view先验静态塞入所有路径：selection传
  `bridge_prior[:,18:21]`，训练传逐query混合后的`corrupt_prior[:,18:21]`，使loss与student实际证据视图一致。
  failed r1保持不可变；revision 2只改batch合同与run revision，不改模型、数据、p/KL、loss权重、seed、资源或legacy
  gate，不增probe/smoke/回归矩阵。证据：PyTorch DataLoader官方文档
  `https://docs.pytorch.org/docs/stable/data.html`、P6R terminal/config/runner。

<a id="detail-v61"></a>

## V6.1 Occupancy-verified world compiler 防重复结论（2026-08-22）

- `V61-F01`（`engineering/protocol`, `resolved`）：`WS-V61-H-P0-001` 首次正式启动在创建 run
  directory、读取 R9/R10/raw evidence、GPU、训练或生成器之前，对尚不存在的
  `/root/autodl-tmp/runs/worldsim_v61` 调用 `shutil.disk_usage`，触发 `FileNotFoundError`。没有 canonical run，
  也没有方法结果；不得把它记成 Occupancy/SceneIR-O rejection。`WS-V61-H-P0-002` 只在资源审计前以
  `mkdir(parents=True, exist_ok=True)` 创建精确 namespace，并增加缺失目录单测；R9/R10 hashes、28-case、scene mapping、
  truth tiers、threshold/stop rules、资源门与 confirmation lock 全部不变。H002 从干净提交 `6247fd8` 运行并使全部
  P0 gate PASS；canonical=`run://worldsim_v61/WS-V61-P0-SCOPE-FREEZE-01/20260822T100812Z__scope-freeze-s20260822-r1`，
  gate SHA=`fb2a416a...ae40`。仍然成立的边界：任何新路线 runner 都必须先创建自己的精确 namespace，禁止把父目录的
  可用空间当成子 namespace 已存在。

<a id="detail-v6"></a>

## V6 可验证世界编译器新增防重复结论（2026-08-21）

- `V6-F01`（`governance/research-direction`, `active`）：V5.2.1 的 badcase census、人工归因、Base Validity、
  immutable run、exact-once、UNKNOWN/abstention 与 failure-ledger 纪律继续有效，但它们没有建立 TrackBayes/M3 的
  causal bridge，也未解决偏离 logged trajectory 后的大场景扩展、生成内容可信固化和闭环复用。Stage H/BKI 从未执行，
  不得写成算法 reject；V5.2 M123 autoresearch 主线状态为 `superseded_by_v6_direction_reset`，M1/M2/M3 只作为
  SceneIR provenance、factorized validity 和 dynamics verifier 的子系统证据。V6 的合法复开边界是跨 frontend 的
  SceneIR/support/provenance/verify/bake/deterministic-runtime 问题，禁止把 V6 再退化为 StreetGS repair、TrackBayes-only、
  KNN/Graph/BKI 或 cut-in mining 主线。证据=`WS-V6-G0-REPO-CONVERGENCE-01`、
  `docs/WORLDSIM_V6_VERIFIABLE_WORLD_COMPILER_AUTORESEARCH_PLAN.md`、
  `docs/autoresearch/worldsim_v6/governance/REPO_PREFLIGHT.json`；方法质量 run=`0`。
- `V6-F02`（`engineering/protocol`, `resolved`）：G2 首次裸 `pytest -q` 在 collection 阶段出现 `12` 个
  `motion_proj/scripts` import error，显式 `PYTHONPATH=$PWD` 后才执行测试；随后 motionproj interpreter 下的 4 个
  V5.1 frozen-runtime tests 因预期 `/root/autodl-tmp/envs/drivestudio/bin/python / torch 2.1.2+cu118` 而失败，使用合同
  指定解释器后对应四文件 `15 passed`。根因是 integration runner 把 repo import root 与历史 runtime profile 当成单一
  环境默认值，不是 merge、算法或冻结结果漂移。以后全量 gate 必须显式注入 repo root，并按 config runtime 分组；不得
  删除 exact-runtime tests 或放宽版本字段。证据=`WS-V6-G2-BRANCH-CONVERGENCE-01`、最终 motionproj
  `1443 passed / 1 skipped` + DriveStudio `15 passed`。
- `V6-F03`（`engineering/asset-integrity`, `resolved with retained asset boundary`）：G2 frozen-asset regression 发现
  Instant NuRec official checkout、P2 selected checkpoint 和 P3 的 `158` 个 chunk payload 缺失，但 manifest、source
  checkpoint 与冻结 hash 仍在；这会造成 5 个 asset-dependent tests 失败，不能倒写历史方法 reject。Instant NuRec 按 exact
  public commit/tree 恢复；P2 从 immutable source 确定性重建并命中 `432,111,754 bytes / 7be87e8b...7448`；P3 重建
  payload 对旧 manifest `158/158` bytes/hash exact 后只补缺失文件，未覆盖旧 manifest。P2 recovery r1 因传相对 protocol
  path 在 snapshot 前 blocked，r2 改为绝对路径后成功；旧 r1 保留。以后清理 canonical selected asset 必须同步保留可执行
  materializer 与 exact source，恢复只能新 run→逐 hash 比较→补缺失字节，禁止生成近似资产或改旧 manifest。证据=
  `20260821T073335Z__g2-p2-asset-recovery-s0-r1`、`20260821T073353Z__g2-p2-asset-recovery-s0-r2`、
  `20260821T073459Z__g2-p3-asset-recovery-s0-r1`、R0/P3 `23 passed`。
- `V6-F04`（`protocol/governance`, `resolved`）：V5.1 `_validate_normative_plan_binding` 只允许 P0 base hash 与
  `b359541` Stage-B append hash，但当前 canonical 文档已在 `a9dede0` 冻结 terminal closeout，SHA-256=
  `a0e764f3...fe1d`；因此 4 个 protocol tests 在到达各自语义断言前统一被旧 allowlist 拦截。修复不改历史 config、plan
  或 gate，只把该 exact terminal commit/hash 加入 fail-closed allowlist，并把回归期望指向 terminal hash；任何未知第四种
  状态仍拒绝。禁止用“任意后继 commit”或跳过 hash 来救测试。证据=`WS-V6-G2-BRANCH-CONVERGENCE-01`、
  `tests/test_worldsim_v51_protocol.py=9 passed`。
- `V6-F05`（`engineering/provenance`, `resolved with noncanonical run retained`）：R1 首个实例
  `20260821T081500Z__r1-capability-s0-r1` 在 capability runner/config/tests 仍未提交时执行，summary 如实记录
  `source_dirty=true`；手工指定的目录时间标签还晚于真实完成时间。其能力事实虽通过，但不能作为 canonical closeout，旧目录
  保留且不覆盖。修复是在 runner 创建 run 前读取 `git status --porcelain`，dirty 即 fail-closed；先提交
  `d981df7fdde5458eb3878193c4a76f6dcf926ad4`，再由 runner 自动生成真实 UTC 标签的新实例
  `20260821T080610Z__r1-capability-s0-r1`，其 `source_dirty=false`、gate PASS。以后不得用“内容看起来正确”绕过
  source cleanliness 或手工修正旧 terminal；工程实例与 canonical evidence 必须分开登记。

<a id="detail-v52"></a>

## V5.2 人工归因与 M123 causal bridge 防重复结论（2026-08-20）

- `V52-F01`（`evaluation/attribution`, `active`）：V5.2.1 的 `GLOBAL_RGB / ACTOR_RGB / BOUNDARY` failure label 是合法
  census 结果，但不能自动解释为 M1/M2/M3 的模块失败。用户指定评审者对代表性 18-case package 完成 `18/18` 逐图复核后，
  冻结 `9 BASE_FAILURE + 8 M123_ELIGIBLE + 1 ATTRIBUTION_UNRESOLVED`：AD-GS 的多条 actor/boundary case 实际由白屏、
  单色、全局 smear 主导；即使 ownership 完美也无法恢复这些画面。不得用 BASE_FAILURE case 评价 TrackBayes、M3 delta 或
  M2 router，也不得删除这些 case 来改善基座 aggregate。所有后续 M123 run 必须先执行 P0 Base Validity Gate，并在完整报告中
  单独保留 base sentinels。证据=`WS-V521-P11-HUMAN-ATTRIBUTION-01`、canonical run=
  `/root/autodl-tmp/runs/worldsim_v521/20260820T130000Z__p11-human-review-attribution-s0-r001`、cases SHA=
  `d89f4a4b...381f`。
- `V52-F02`（`evaluation/causal-identification`, `active`）：8 个 StreetGS eligible case 的视觉症状与 M1 observation scarcity、
  M3 actor trajectory/visibility 高度相容，但 panel 不能证明失败 pixel 恰好来自 low-observation/uncertain Gaussian，也不能证明
  ghost 会被 actor-pose warp 解释。因此状态只能是 `DIRECTION_SUPPORTED_CAUSAL_BRIDGE_PENDING` /
  `SYMPTOM_OVERLAP_STRONG_EXACT_TEMPORAL_BRIDGE_PENDING`，不得从人工诊断直接晋级 TrackBayes 或修改 M3。合法复开必须保持
  Discovery design `5`（#05/#10/#11/#16/#17）与 one-shot Confirmation `3`（#06/#12/#18）分离，先冻结并执行 exact
  pixel→Gaussian/U2-B3 observability bridge 与 `unwarped/flow-warped/pose-warped` temporal bridge；Confirmation 不得选 arm、
  threshold 或 metric。M2 只消费已通过 candidate 的 uncertainty/validity 并决定 execute/abstain；geometry undefined 时不得
  写 geometry-safe。证据=`docs/run_manifests/worldsim-v5.2.1-human-review-attribution-v1/` 与
  `configs/worldsim_v52/m123_autoresearch_v1.yaml`。

<a id="detail-v51"></a>

## V5.1 M1-only 新增防重复结论（2026-08-17）

- `V51-F01`（`engineering`, `resolved`）：首轮执行
  `pytest -q tests/test_worldsim_v51_protocol.py tests/test_audit_worldsim_v51_start.py` 在 collection 阶段因测试文件直接
  import `motion_proj.worldsim_v51`、但未显式把仓库根加入 `sys.path` 而报 `ModuleNotFoundError: motion_proj`；同轮
  `python scripts/audit_worldsim_v51_start.py --help` 已通过，所以该 terminal 推翻的是“pytest 启动环境总会自动注入
  repo root”的工程假设，不是 P0 协议或算法失败。修复在测试 import 前按绝对 `ROOT` 注入路径并以原命令回归；失败时
  没有运行方法、读取 validation/test/KITTI quality 或产生质量数字。后续直接脚本与测试入口都必须有独立 import smoke，
  禁止把 collection error 计入方法分母。证据：`WS-V51-P0-M1-SCOPE-FREEZE-01`、
  `tests/test_worldsim_v51_protocol.py`、`tests/test_audit_worldsim_v51_start.py`。
- `V51-F02`（`engineering/evaluation`, `resolved`）：A0 runner 的 metric 单测把输入 `float32(0.1)` 产生的
  `0.10000000149011612` 与 Python 十进制 `0.1` 做严格相等，导致 `1 failed / 6 passed`；这推翻的是“人工十进制常数
  可以作为 bit-exact 浮点 oracle”的测试假设，不是 frozen metric 定义或 A0 canonical replay 失败。修复只把人工常数
  断言改为 `pytest.approx`；正式 A0 仍把同一实现重算值与 canonical JSON float 做 `delta == 0.0`，posterior/statistics
  仍逐 bit 比较。禁止为了让 exact gate 通过而对 canonical metric 使用容差、舍入或字符串截断。失败时未启动正式 run、
  GPU renderer、方法推理或 validation/test/KITTI quality read。证据：`WS-V51-M1-A-UNARY-OBSERVABILITY-01`、
  `tests/test_replay_worldsim_v51_v5_unary.py`。
- `V51-F03`（`engineering/evaluation`, `resolved`）：A1 规定 `visibility >= 0.01` 为 inclusive gate，但冻结 NPZ 中
  visibility 是 `float32`；若直接与 Python double `0.01` 比较，存储的 `float32(0.01)` 会因表示略小而被误判为
  false，首轮测试得到 `[False,False,False]` 而非 `[True,False,False]`。这会真实改变 observation denominator，不能靠
  放宽单测解决。修复是在比较前把配置阈值量化为 observation dtype，同时在诊断中分别记录 configured/applied value；
  门仍是 inclusive，未读取 evaluation quality 或搜索阈值。禁止用 epsilon、容差或事后改 threshold 隐式改变分母。
  证据：`WS-V51-M1-A-UNARY-OBSERVABILITY-01`、`motion_proj/worldsim_v51/evidence/visibility.py`、
  `tests/test_worldsim_v51_visibility.py`。
- `V51-F04`（`algorithm/protocol`, `resolved before A2 quality read`）：A2 首轮只读 evidence-statistics 检查发现，
  若直接在“全部 Gaussian”上取 effective-count 下分位数、entropy/disagreement 上分位数并用 OR 组成 UNKNOWN，
  scene-0379/1087 的 count 与 disagreement 分位数会同时退化为 `0`；inclusive `disagreement >= 0` 会把全部 Gaussian
  判为 UNKNOWN，Gaussian coverage 直接变成 `0`。根因是未观测但由冻结 base-model prior 明确赋类的 Gaussian 占
  `67.39%/97.20%`，它们不能和真正有 semantic observation 的校准总体混在一起。A2 在任何 evaluation artifact 或
  quality metric 读取前，把阈值总体冻结为“三个 H scene 中 effective-count>0 的 A1 Gaussian pooled population”，并用
  `high entropy AND (low count OR high disagreement)` 保留 entropy 作为必要条件；三个阈值固定为该总体的
  `Q25(count)=0.19274792820215225`、`Q75(entropy)=0.005402358970383594`、
  `Q75(disagreement)=8.494543610182426e-12`。禁止把全量分位数退化误写成 A2 方法负结果，也禁止在看到 A2 evaluation
  quality 后改总体、分位点、布尔规则或图像 abstain threshold。证据：
  `configs/worldsim_v51/m1_unary_unknown_v1.yaml` 与 A1 posterior SHA binding；S/validation/test/KITTI 仍未读取。
- `V51-F05`（`algorithm/protocol`, `rejected by A3 r005`）：计划 A3-1 提议以
  `n_eff=(sum r)^2/(sum r^2+epsilon)` 限制 A3-0 的 fractional concentration `sum r`，并解释为
  correlation-aware。但 A1 reliability 逐 observation 严格在 `[0,1]`，因此忽略仅用于数值稳定的 epsilon 时恒有
  `sum r^2 <= sum r`，进而 `n_eff >= sum r`：作为上限必然是 no-op，直接替换则会提高而非降低 posterior
  concentration。更根本的是该式只见单 observation 权重及其平方和，没有任何 view-pair correlation observable；
  调换 view 顺序或相关结构而保持权重集合时结果不变，不能支持“10 个高度相关 view 不等于 10 个独立证据”的命题。
  r005 用 v2 evidence-only audit 对 `944,443` 个 positive-count Gaussian 复现：无 epsilon 时
  `n_eff<sum(r)` 为 `0`，absolute cap change>`1e-9` 为 `0`；若直接 replacement，`940,762/944,443=99.6102%`
  Gaussian concentration 被放大。结论=`a3_kish_cap_rejected_structural_noop_not_correlation_aware`；A3 不启动 GPU
  quality arm，按原计划只解锁独立 A4。禁止为挽救 A3 事后加入相关系数/时间核/feature similarity；那将是新机制，
  不是原 A3-1 的修复。r005 未读 evaluation artifact/quality、validation/test/KITTI，failure delta=`V51-F05/F06`。
- `V51-F06`（`engineering/evaluation`, `resolved without quality read`）：A3 audit v1/r004 用相对 cap change 判断
  epsilon 是否产生“有意义修正”，但 0471/0379 存在 reliability=`1.401298464324817e-45` 的 float32 最小次正规数；
  `epsilon=1e-12` 使这些近零 mass 的相对变化达到 `1.0`，尽管三个场景最大绝对 cap reduction 仅约
  `2.5e-13`。r004 因此合法保留为 `done/inconclusive`，不是 A3 得到有效 concentration reduction，也不是方法质量
  失败；它没有读取 evaluation artifact/quality、启动 GPU renderer 或改变 posterior。v2 新配置绑定 r004 与 v1 hash，
  在新结果前把 meaningful gate 改为 absolute cap change>`1e-9`，同时继续报告相对量作诊断。禁止用近零分母的巨大
  相对数宣称机制有效，也禁止覆盖 r004 terminal；只能用新 r005 重放同一 45 份 evidence observation。
- `V51-F07`（`algorithm/novelty`, `rejected by A4 r006`）：CIF 原论文将 occupancy probability 与 conditional
  instance distribution 分开，并明确针对 appearance opacity 与 occupancy 混淆；其完整方法还包含 learned deformable
  Gaussian instance field、identity calibration 与 semantic resampling。V5.1 计划明确不引入后三类机制，而当前 renderer
  已把 appearance `base_opacity` 与 conditional ownership sidecar 相乘，A1 已分离 visibility eligibility，A2 已分离
  UNKNOWN。因而 A4 若把 `base_opacity` 当 occupancy，会在 renderer 中二次乘 alpha 且违反参考机制；若用
  visibility/effective-count，会再次把不可见误作不存在；若对已实例化 Gaussian 设 occupancy=1，则与现有 renderer
  bit-exact no-op。r006 绑定三个 A2 posterior 与 renderer/visibility/abstention 源码，确认 occupancy field=`0/3`、
  constant-one 对现有 renderer=`3/3 bit exact`，而复用 appearance opacity=`3/3 non-exact` 且会二次缩放。结论=
  `a4_cif_decoupling_rejected_no_independent_occupancy_observable`；未读 evaluation artifact/quality、未启动 GPU/training。
  禁止把已有 A1/A2 分解重新命名为 CIF 增益，或在结果后偷偷解锁完整 CIF 训练、校准/重采样。参考：
  [CVPR 2026 official paper](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_Consistent_Instance_Field_for_Dynamic_Scene_Understanding_CVPR_2026_paper.html)。
- `V51-F08`（`engineering/resource`, `resolved without duplicate run`）：首次尝试并行启动 S unary materialization 时，
  PowerShell→SSH→远端 Bash 的多层转义把 `$!/$?` 保留成字面量，导致外层 `wait/test` 解析失败。只读 PID、GPU、run 目录
  与 status 审计确认 scene-0998 只有一个正式进程，scene-0359 根本没有启动；因此保留 r049 单实例自然完成，再以独立前台
  命令串行执行 r050。两者均为 `done`、checkpoint 前后 SHA exact，且无重复 scene/candidate 分母。3090 实测后段显存保留
  分别达到约 `22 GiB/20 GiB`，也推翻了“两个 unary materialization 可安全共卡并发”的资源假设。以后 Windows 发起的远端
  后台编排不得内联依赖 `$!/$?`；优先每个长 run 使用独立前台 SSH session，确需后台时使用远端脚本/控制器并单独审计
  PID、日志与 terminal。wrapper 失败不得写成方法失败，也不得因外层 exit code 重跑已封口的 immutable run。证据：
  r049/r050、source commit `6950597`、`configs/worldsim_v51/stage_a_screening_v1.yaml`。
- `V51-F09`（`algorithm/evaluation`, `rejected by Stage A r007`）：A1 visibility 在 H 上通过的 scene-balanced
  `ΔBoundary-F1=+0.001155713` 没有在预注册 S=`0998/0359` 上复现。S 两场 delta 分别为
  `-0.0000904944/+0.0000574359`，只有 `1/2` 非负、`0/2` 达到 clearly-positive `+0.001`，均值为
  `-0.0000165293`；尽管 mean IoU/Brier/ECE 略改善且 FN 增量仍在门内，冻结 gate 是合取，A1 必须 rejected。
  这推翻“hard visibility eligibility 的 H 小效应可跨开发场景稳定复制”，不是 Bayesian U2/B3 基线失败。禁止根据 IoU
  或 calibration 的微小正向分量保留 A1、删除 0998、放宽 clearly-positive 门或在同一 S 上重选 visibility threshold。
  合法复开需要新机制、新任务和未读场景；V5.1 当前冻结 U2/B3，不再继续复杂化 Bayesian family。证据：r007、
  `configs/worldsim_v51/stage_a_closeout_v1.yaml`。
- `V51-F10`（`algorithm/evaluation`, `rejected by Stage A r007`）：A2 UNKNOWN 在 S 上仍能集中错误：scene-balanced
  accepted/abstained error=`0.0148416/0.134393`，两场均有非空 denominator；但 coverage 在 0998/0359 为
  `0.250105/0.864765`，scene-balanced mean=`0.557435<0.60`，未通过冻结 selective gate。0998 的 UNKNOWN Gaussian
  ratio=`34.5512%`，0359 仅=`0.7891%`，说明 H 分位数规则的场景依赖很强；同时 A2 conditional posterior 与 A1 相同，
  继承 `V51-F09` 的 conditional gate 失败。因此 A2 rejected，不能用较高 unknown recall 或 error separation 掩盖可用覆盖率
  不足，也不能在看到 S 后调整 Q25/Q75、布尔规则或 image threshold。证据：r007；failure delta=`V51-F09/F10`。
- `V51-F11`（`protocol/governance`, `resolved by explicit user authorization on 2026-08-17`）：normative plan 对 Stage A 全失败后的解锁规则内部冲突。§10.8
  明写“所有 Stage A arm 都失败”时保留 U1/U2 并进入 Stage B；附录“八、Stage A 后如何解锁”却只在 Stage A
  candidate 通过 S 时允许 `WS-V51-M1-B-LUDVIG-UPLIFT-01`。r007 的真实状态正是 A1–A4 全 rejected、fallback=
  `U2/B3`，因此执行者不能静默挑选有利条款，也不能把“进入 Stage B”的研究顺序当成独立授权。Stage B 保持
  `pending/locked`；合法复开必须由用户明确选择“授权 U2/B3 fallback 进入 Stage B”或“按 candidate-pass 条款关闭
  M1”，再用 freeze-only commit 统一 normative/short plan/config。用户于 2026-08-17 明确选择“授权 U2/B3 fallback
  继续 M1”，并要求单 arm/scene/工程/paper failure 留档后自动进入下一冻结路线；因此本治理阻塞解除，但原条款冲突和
  r007 结论不删除。解法采用 executable authorization overlay 绑定原 normative/P0/Stage A/proposal SHA，不原地改写
  冻结字节；M2/M3 与 validation/test/KITTI 锁保持。该问题不是算法负结果，解除时仍未读取 C/validation/test/KITTI
  quality。证据：`docs/WORLDSIM_V5_1_M1_TOPCONF_PLAN.md`、`configs/worldsim_v51/stage_a_closeout_v1.yaml`、
  `configs/worldsim_v51/stage_b_authorization_v1.yaml`、`docs/WS_V51_STAGE_B_PREFLIGHT.md`。
- `V51-F12`（`engineering/resource/governance`, `resolved by r003 asset + r004 resource smoke`）：Stage B 的 faithful 第一版要求官方 DINOv2
  ViT-g/14 registers，但 2026-08-17 只读审计只找到 Depth-Anything-V2 内部 DINOv2 模块，torch/HuggingFace cache
  均无对应官方 checkpoint；“存在 DINOv2 源文件”不能写成 LUDVIG 资产已冻结。官方 LUDVIG README 记录的测试平台是
  A6000 48GB，当前 RTX 3090 只有 24576 MiB，且 Stage A 单个 unary materialization 已实测约 20–22 GiB，因此
  DINO extraction 与 DriveStudio renderer 禁止同进程或同卡并发常驻。授权后必须先冻结 upstream commit、官方模型来源、
  checkpoint SHA/license、preprocess 与 PCA population/seed，再采用“离线 DINO sidecar→释放进程→renderer uplift”分段
  执行。官方 checkpoint HEAD bytes=`4,546,140,349`，multipart ETag 不是 SHA；官方 extractor 又会把全 camera raw
  feature 预分配在 GPU。V5.1 只允许先下载后全 SHA，再用 CPU memmap/40-D patch-grid streaming 与 dense-parity test
  保持语义；缺资产或 OOM 只记工程/resource terminal，不得写成 feature uplift 失败，也不得临时换小模型、降分辨率后
  仍称 faithful port。
  资产缺失子项由 r003 解除：official bytes=`4,546,140,349`、SHA-256=`746ecb8c...a283`，本地重算
  8 MiB×542-part ETag exact。r004 随后在 clean source=`935d2b2` 上以官方 commit=`7764ea0f...25fc8`、原始分辨率
  预处理、ViT-g/14 registers 与 strict state dict 完成 one-image forward：params=`1,136,486,912`、keys=`568`、
  missing/unexpected=`0/0`，4 个输出 shape 全 exact；GPU sampled/Torch reserved peak=`6,702/6,376 MiB`、cgroup peak=
  `15,701,860,352 bytes`，显著低于预注册门，资源风险因此 resolved。该解除不代表 feature uplift 有效，也不解除
  DINO→释放进程→renderer 的顺序合同；后续仍禁止同卡并发、临时换小模型/分辨率，且必须先过 operator parity。
  证据：`configs/worldsim_v51/stage_b_preflight_v1.yaml`、`docs/WS_V51_STAGE_B_PREFLIGHT.md`、
  `configs/worldsim_v51/stage_b_dinov2_resource_freeze_v1.yaml`、r004 canonical run。
- `V51-F13`（`engineering/protocol`, `resolved without method execution`）：P0 scope config 已把 normative plan
  SHA-256 冻结为 `3d7f7481...`，但 Stage A closeout commit `3d33262` 曾直接向该长计划加入 5 行执行进展，使当前
  HEAD SHA 漂到 `b119cd56...`。Stage B preflight 运行 `pytest -q tests/test_worldsim_v51_protocol.py` 时因此得到
  `2 failed / 1 passed`，即使本轮新增注记撤回后仍复现，证明这是 inherited drift。这推翻“冻结后的 normative plan
  仍可作普通活文档窄改”的工程假设，不是 P0、Stage B 算法或数据失败。修复用新提交移除这 5 行，把当前状态继续保留
  在 short plan/status/experiments，恢复原 plan SHA exact；不改写历史，也不只改 expected SHA 掩盖漂移。若用户授权后
  确需统一解锁规则，必须建立显式 supersession/migration 并同步 P0 binding。失败时没有下载 checkpoint、启动
  method/GPU run 或读取 C/validation/test/KITTI quality。证据：`3d33262`、
  `configs/worldsim_v51/p0_m1_scope_v1.yaml`、`tests/test_worldsim_v51_protocol.py`、
  `docs/WS_V51_STAGE_B_PREFLIGHT.md`。
- `V51-F14`（`engineering/protocol`, `resolved by r010`）：LUDVIG DINO extractor 的 PCA 路径不是天然确定性合同。
  `PCA(n_components=40)` 没有 `random_state`，大矩阵会走 randomized solver；当 patch 数超过 500,000 时还用未设 seed
  的 `np.random.choice` subsample。更隐蔽的是 GPU path 用 PyTorch `std`（默认 correction=1），CPU path 用 NumPy
  `std`（correction=0），所以为省显存切到 CPU 会改变标准化与全部 feature。V5.1 proposal 冻结 H evidence=
  `45 views×7,296 patches=328,320`，明确不触发 subsample；固定 std correction=1、randomized PCA
  random_state=`20260814`、40-D、whiten=false，并把 scaler/PCA state 持久化后只 transform S/C。不得把 solver/seed/std
  差异当作 backbone 增益或在 S/C refit；这是 reproducibility hardening，不是参数搜索。本轮未下载模型、提取 feature 或
  读取质量。证据：LUDVIG `predictors/dino.py`、`configs/worldsim_v51/stage_b_freeze_proposal_v1.yaml`、
  `docs/WS_V51_STAGE_B_FREEZE_PROPOSAL.md`。r010 已对 45 H views exact 执行该 hardened contract：首图 raw feature
  repeat bit-exact，PCA state deterministic NPZ repeat byte-exact，45 个 sidecar 的 file/content SHA 全 exact，raw memmap
  成功后删除，PCA state SHA=`fe9eea72...3231c8`；因此本条 resolved。该解决只证明 feature/PCA 可复现，不证明
  LUDVIG uplift 或方法质量有效，S/C 仍只准 transform、不得 refit。
- `V51-F15`（`evaluation/governance`, `resolved by r015 without promoting proxy to method input`）：Stage B 的 same-actor/actor-background metric 可从
  frozen `RigidNodes.points_ids[:,0]` 与 Background row 构造，但这是 base-model membership proxy，不是真实 ownership GT。
  若把该 proxy 输入 DINO/PCA/uplift/权重会形成标签泄漏；若只凭 proxy margin 解锁 Graph，则会把模型自身表示循环证明为
  语义正确。proposal 将其限制为 evaluation-only stratum，强制写
  `model_membership_proxy_not_ground_truth`，并同时报告不消费 membership 的 same-Gaussian repeatability 与 heldout DINO
  reprojection。无 eligible actor 的 scene 必须保留 abstain；不得降低 32-Gaussian eligibility、删 1087/0379 或只报大
  Rigid 场景。Stage B 未获授权，本轮没有产生 metric。证据：V5 formal30k r027–r034 metadata、
  `configs/worldsim_v51/stage_b_freeze_proposal_v1.yaml`、`docs/WS_V51_STAGE_B_FREEZE_PROPOSAL.md`。r015 已严格按
  evaluation-only 声明执行：proxy 未进入 method/PCA/uplift，1087 因无 eligible actor 保留 abstain，同时报告不消费
  membership 的 repeatability 与 heldout reprojection；治理风险因此 resolved，但方法因 margin 失败另记 `V51-F31`。
- `V51-F16`（`engineering/resource`, `resolved by parallel r003`）：DINOv2 asset r002 在已 source network turbo、official URL、
  fixed target 与 curl resume 合同下运行约 `106 s`，连续 prefix 仅增长到 `26,566,656 / 4,546,140,349 bytes`；
  按稳定窗口外推需数小时。执行者精确核对并 `TERM` 唯一 curl PID，runner 以 `exit=-15` 写入 blocked terminal，
  final asset 不存在，`.partial` 及其 SHA=`934ef5aa...e2265` 保留。该事实推翻“代理单连接足以在合理实验窗口完成 4.5 GB
  official asset”的工程假设，不是 checkpoint 损坏、DINO/LUDVIG 方法或 GPU 失败。合法恢复必须新 run ID，冻结 prefix
  bytes/SHA，以互不重叠的 fixed HTTP ranges 并行下载；每段验证 range bytes/SHA，assembly 后同时通过 full SHA-256 与
  S3 multipart ETag=`3d1b...-542`（8 MiB×542 parts）才可原子发布。禁止覆盖 r002、删除 prefix 后假装首次下载、
  使用镜像/不同权重或只凭 total bytes/remote ETag 宣称完成。证据：
  `20260817T141600Z__m1-stage-b-dinov2-asset-s20260814-r002`、
  `configs/worldsim_v51/stage_b_dinov2_download_parallel_v1.yaml`。r003 以 14 ranges 在 `1504.935 s` 完成；逐段
  bytes/SHA、assembled full SHA=`746ecb8c...a283`、multipart ETag=`3d1b...-542` 与 terminal/manifest 二次复核全 exact，
  final 原子发布后精确删除 `15 files / 4,546,140,349 bytes` staging。本条工程恢复因此 resolved；当时仍 active 的
  ViT-g 24GB resource smoke 风险 `V51-F12` 后由 r004 独立解除。
- `V51-F17`（`engineering/protocol`, `resolved before formal r005`）：synthetic operator parity 首轮 unit regression=
  `2 failed / 8 passed`，两处 failure 都来自同一个 below-Gaussian-view-mass 夹具：它把 24 个 intersection 全设为
  minimum contribution=`1e-4`，所以聚合 mass 实际为 `0.0024≥0.001`，被测算子正确保留该 group，而测试错误地要求
  drop。这推翻的是“逐 intersection 在 floor 上就能构造低于 group floor”的夹具假设，不是 B0/B1 公式、dense oracle、
  lazy bilinear 或 LUDVIG 方法失败。修复只把该 group 改成 5 个 `1e-4`、其余为 0，使 mass=`0.0005`；不改阈值、
  operator、seed 或通过标准。失败发生在 formal r005 创建前，未加载 DINO/renderer、未读真实 feature 或质量；禁止用
  降低 group floor 掩盖夹具错误。证据：`tests/test_worldsim_v51_feature_uplift.py`、
  `scripts/audit_worldsim_v51_stage_b_operator_parity.py` 的 pre-formal regression。修正后 19/19 regression PASS；formal
  r005 又以 11/11 checks PASS，并真实观测 `8 Gaussian-view → 7 kept + 1 dropped`，确认本条 resolved。
- `V51-F18`（`engineering`, `resolved before r005 result-freeze commit`）：新增 result-freeze test 先通过 canonical run
  文件存在/SHA、summary status/checks/checkpoint immutable，随后因把局部变量 `validate_freeze` 简化为 `freeze` 时漏改
  两条 parity 断言，得到 `NameError` 与 `1 failed / 19 passed`。这推翻“机械重命名后所有引用自然一致”的测试维护假设，
  不推翻 r005 artifact、operator parity 或任一质量结论。修复只替换两处旧变量名，并重跑同一 20-test regression；
  禁止重跑/覆盖 r005 或修改 freeze 数字来绕过测试。证据：`tests/test_worldsim_v51_feature_uplift.py`、
  `configs/worldsim_v51/stage_b_operator_parity_freeze_v1.yaml`。
- `V51-F19`（`engineering/protocol`, `resolved by v2/r007 reaching renderer`）：one-H-view formal r006 在
  `_build_runtime()` 导入 DriveStudio `models.gaussians.basics` 时因 `ModuleNotFoundError: pytorch3d` blocked；v1 config 错把
  入口冻结为 motionproj Python，而历史 DriveStudio 运行合同使用独立 `/root/autodl-tmp/envs/drivestudio/bin/python`。
  terminal 发生在 dataset/trainer 构造和 renderer 启动前，0 intersection、0 denominator、0 quality；这推翻“主项目环境可
  直接承载 DriveStudio CUDA 依赖”的工程假设，不是 renderer/LUDVIG/资源失败。合法恢复必须保留 r006，以 v2 + 新 r007
  只替换 interpreter，并在 formal 内 exact 核对 executable、torch=`2.1.2+cu118`、CUDA=`11.8`、`pytorch3d/gsplat`
  imports；不得安装包污染 motionproj env 或改 view/floor/resource gate。证据：r006 status SHA=`06b74ec9...b4be3`、
  `configs/worldsim_v51/stage_b_one_view_contribution_v1.yaml` 与 v2 recovery config。v2/r007 已 exact 使用该环境并完成
  dataset/trainer/checkpoint/renderer 启动，本条因此 resolved；r007 的后续尺寸阻塞另记 `V51-F20`。
- `V51-F20`（`engineering/protocol`, `resolved by v3/r008 reaching post-render resource gate`）：r007 到达真实单视图 renderer 后，v2 把 sensor
  JPEG `1600×900` 错冻为 renderer width/height，触发 fail-closed。冻结 r027 source config 已明确三路
  `downscale_when_loading=[2,2,2]`，现有 V5 SAM/actor configs 与历史 `V3-F18` 也记录 model-native=`800×450`；这是重复
  违反三层尺寸合同，不是 renderer 或 contribution 质量失败。r007 在 intersection inventory 前停止，且旧错误文本未写出
  observed/expected 数值。合法恢复必须保留 r007，以 v3/r008 显式同时冻结 sensor/downscale/model-native 三层尺寸并增强
  错误文本；不得改 checkpoint、view、support floor 或把 800×450 写成降分辨率调参。loader 会基础设施性物化
  image/mask/LiDAR，但 runner 不消费其值；二者须分开记录。证据：r007 status SHA=`da279515...8d3c9`、r027
  `config.yaml` SHA=`eb22faea...9c6d`、`configs/worldsim_v51/stage_b_one_view_contribution_v3.yaml`。v3/r008 已按
  `800×450` 完成 renderer 并进入 post-render 资源门，因此本条 resolved；后续资源阻塞另记 `V51-F21`。
- `V51-F21`（`engineering/resource/protocol`, `resolved by v4/r009`）：r008 在真实单视图 renderer 和
  contribution 汇总完成后，NVIDIA peak=`14,234 MiB` 超过预注册 ceiling=`12,288 MiB`，故 status=`blocked`；
  cgroup peak 仅 `9,598,074,880 bytes`，89 个采样无错误，进程正常退出且 GPU 已释放，不能误写成 OOM、renderer
  或算法质量失败。原 runner 又在资源门通过后才写 contribution/resource artifact，使 blocked run 只保留
  status/events/resolved/resource-samples；这会降低失败诊断可审计性。禁止覆盖 r008，合法恢复只能新建 v4/r009：
  保持 scene/view/checkpoint/renderer/two-floor/quality locks 不变，仅把 NVIDIA/Torch ceiling 提升为 `16,384 MiB`
  （仍低于 24 GiB），并在资源判定前先持久化只读 denominator/resource 诊断。r008 status/resource-samples SHA=
  `8b8ebe17...b2118bf / fc0f9788...a90932`；不得用这次工程资源事实选择或调节算法质量。v4/r009 已在
  `14,234 MiB NVIDIA / 13,882 MiB Torch reserved / 9,593,946,112 bytes cgroup` 下通过，诊断 artifacts 也在 gate 前
  持久化，因此本条 resolved；冻结结果见 `stage_b_one_view_contribution_freeze_v1.yaml`。
- `V51-F22`（`engineering/audit`, `resolved immediately`）：r009 完成后的第一次独立逐文件 verifier 把 manifest
  payload key 硬编码为 `files`，但该 runner 的冻结 schema 使用 `inventory`，只读命令因此 `KeyError: 'files'`；
  formal r009 及任何 artifact 均未改变。修正 verifier 按冻结 schema 读取 `inventory` 后，8/8 manifest entries 的
  SHA/bytes exact，run=`10 files / 28,156 bytes`。后续 verifier 必须先读 schema/key，不能跨 runner 猜测 manifest 字段。
- `V51-F23`（`engineering/resource/protocol`, `resolved by v2/r012`）：H 45-view sparse uplift r011 已完整处理
  3 scenes×15 views、写出 6 个 Gaussian feature sidecar 并证明 3 个 base checkpoint before/after exact，但 post-compute
  NVIDIA peak=`20,554 MiB` 与 Torch reserved=`20,202 MiB` 超过 v1 预注册的共同 ceiling=`18,432 MiB`，故 formal status
  必须保持 `blocked`。cgroup peak 仅 `13,328,011,264 bytes`，799 个 resource samples 无 monitor error，runner 正常释放 GPU；
  这推翻的是“单-view 14,234 MiB 足以外推三场景 streaming full-run 低于 18 GiB”的资源假设，不推翻 sparse transpose、
  B0/B1、DINO/PCA 或任何 method-quality 结论。合法恢复必须保留 r011 及其 gate 前诊断，以 v2 + 新 r012 从冻结原输入
  完整重跑；唯一改变为 NVIDIA/Torch ceiling=`22,528 MiB`（仍低于 24 GiB），不得复用 blocked sidecar、改 scene/view/floor、
  调整 operator 或读取 membership/quality。r011 失败证据已独立验证 6/6 NPZ file/content identity、manifest chain 与
  checkpoint immutability exact；status/resources/report/manifest/resource-samples SHA=
  `a450cdaf...0eee6/0312e190...8a37/98140571...99c9/88956448...dae6/76422821...ae9c`。配置证据为
  `configs/worldsim_v51/stage_b_h_uplift_v1.yaml` 与 v2 recovery config。v2/r012 从原冻结输入完整重跑，在相同 observed
  NVIDIA/Torch reserved=`20,554/20,202 MiB` 下通过 22 GiB 门；6/6 sidecar、19/19 manifest、3 checkpoint 与全部 locks
  独立审计 exact，因此本条 resolved。该解决不证明 B0/B1 方法质量有效，`V51-F15` 仍须由预注册 evaluation-only 门处理。
- `V51-F24`（`engineering/protocol`, `resolved before formal r013`）：H heldout feature 预注册回归中，新 runner/sidecar
  测试已在 frozen motionproj interpreter 得到 `8 passed`，但随后一次聚合命令又在同一 interpreter 调用依赖 DriveStudio
  runtime 的 `test_worldsim_v51_h_uplift.py`，该测试按合同报告 runtime mismatch，汇总为 `1 failed / 9 passed`。这只是测试
  调用环境错误，未创建 formal run、未启动 GPU、未读 membership/uplift quality，不能写成 H uplift 或 heldout transform 失败。
  合法修复是按 `V51-F19` 的环境分层分别执行：DINO/sidecar/heldout tests 使用 motionproj Python，renderer/uplift tests
  使用 drivestudio Python；禁止为让聚合命令通过而改任一冻结 runtime config 或向主环境安装 DriveStudio CUDA 依赖。
  r013 config 将本条纳入 `failure_ledger_refs`，并须在 clean prereg commit 前保留两个解释器各自的 PASS 证据。
- `V51-F25`（`engineering`, `resolved before formal r013`）：从 Windows PowerShell 发出的第二次聚合 SSH 命令在
  双引号内包含 `$(find ...)`；本地 shell 在 SSH 前提前解释该命令替换，并把远端 `find -name` 片段误当成 PowerShell
  命令，最终本地报 `-name is not recognized`、远端 bash 报 unmatched quote。测试没有启动、GPU 未使用、仓库未修改、
  quality 未读。防重复门禁：跨 PowerShell→SSH 的命令不得嵌套未转义命令替换；本次改为独立执行测试、`git diff --check`
  与只读 `find`，不使用 shell substitution 控制流。不得把 launcher quoting 失败计入算法或测试 verdict。
- `V51-F26`（`engineering`, `resolved during r013 post-run audit`）：首次只读 r013 inspection 又把多语句 Python
  `-c` 嵌入 PowerShell 的双引号 SSH 字符串，本地 parser 在远端执行前把 Python 括号误解释为 PowerShell，报
  `An expression was expected after '('`；没有命令抵达远端、没有 artifact/repo/GPU/quality 状态变化。该问题与
  `V51-F25` 同属跨 shell 引号边界，但触发面是嵌套 Python source。合法修复为用 `apply_patch` 创建独立只读 auditor、
  `scp` 到精确 `/tmp` 路径后以固定参数执行；后续禁止在 PowerShell→SSH 双层命令中内嵌多语句 Python source。
- `V51-F27`（`engineering`, `resolved during r013 post-run audit`）：同步 auditor 与三份台账时，命令工作目录是
  local staging 根而三份 source path 误写成 `docs/...`，因此 auditor 已成功传到精确 `/tmp/audit_r013.py`，随后三个
  docs scp 在本地以 `stat local ... No such file` 失败；远端 repo 和 run 均未被部分修改。修复只把三份 source path
  写成 `motion_proj/docs/...` 并逐项同步；防重复要求多 source scp 前先按当前 workdir 解析 source，且不能把部分成功
  的前序传输误当成整条命令成功。
- `V51-F28`（`engineering/resource/protocol`, `resolved by v2/r015`）：formal H evaluation r014 已完整处理
  3 scenes、90/90 evidence/evaluation views并先持久化 3 scene reports 与完整只读 report，但 terminal resource gate
  观测 NVIDIA peak=`22,570 MiB > 22,528 MiB`、Torch reserved=`23,354 MiB > 22,528 MiB`，因此 status 必须保持
  `blocked`。cgroup peak=`14,305,161,216 bytes`，1,208 samples、0 monitor errors、duration=`897.647 s`，GPU 已释放；
  这推翻的是“r012 的 22 GiB uplift ceiling 足以覆盖双向 evaluation sparse projection”的资源外推，不是 H gate verdict。
  禁止读取 blocked r014 的 scene/aggregate quality、覆盖 run 或据其数值改 metric/pair/proxy/gate。合法恢复只允许新 v2/r015
  把 NVIDIA/Torch ceiling 同时提高到 `24,000 MiB`（仍低于 24,576 MiB 卡容量），其余 base config 逐字继承，并从
  r012/r010/r013 原冻结输入完整重跑。r014 status/resources/report/progress/resource-samples SHA=
  `6409545b...f6d1/ffc98a00...674e/510f82ec...227c/61475cb1...06ed/8fae05eb...7cf`；10 files，无 partial。
  v2/r015 从原 freeze 完整重跑，在同一 NVIDIA/Torch=`22,570/23,354 MiB` 下通过 `24,000 MiB` 门，故本条 resolved；
  r015 的 H verdict 由独立质量门决定，不能倒写 r014 为成功。
- `V51-F29`（`engineering/audit`, `resolved immediately`）：r014 blocked metadata 首次只读 hash 命令把实际
  `events.jsonl` 误写成 `events.json`，`sha256sum` 因该单项不存在返回非零，使后续 `&& find` 没有执行；前面其余 hash
  已正常输出，run/repo/quality 均未改变。修正为冻结 schema 的 `events.jsonl` 后 SHA=`31fff013...a7ce`，并完成 10-file
  bytes inventory。后续 verifier 必须从 runner/schema 读取精确 artifact 名，不凭相邻 runner 猜扩展名。
- `V51-F30`（`engineering/audit`, `resolved during r015 closeout`）：独立 auditor 首先要求 blocked r014 与 recovery r015
  report 在删除 `seconds` 后逐 Python float exact，得到 assertion failure。递归定位显示差异均为并行 CPU sparse/BLAS reduction
  的末位浮点扰动；离散字段、denominator、checkpoint、gate verdict 全 exact。修正审计合同为离散字段 exact、float
  absolute tolerance=`1e-12`，共 241 个差异，最大仅 `4.9760e-13`，复核 PASS。禁止把非 bit-exact reduction 误写成方法
  不可复现，也不得用宽松相对容差掩盖 gate 翻转；任何离散/gate 差异或 float 超过 `1e-12` 仍须 fail。
- `V51-F31`（`algorithm/evaluation`, `active rejected-route prevention`）：canonical r015 的 H gate 为 rejected。3 scenes
  中仅 0471/0379 evaluable，B1 actor-background margins=`-0.121280/-0.098618`，正场景=`0/2`，scene-balanced=
  `-0.109949<0`；1087 按冻结 32-Gaussian rule 无 eligible active actor，必须 abstain。Rigid coverage mean=`0.842910`
  与 heldout reprojection `B1-B0=+0.022777` 均过门，说明 normalized transpose 能改善 2D feature reconstruction，
  但不能证明 actor 内 feature 比最近 Background 更紧致；0471/0379 的 B0 margin 也已为负，且 B1 没有救回该前提。
  这推翻 LUDVIG uplift 可直接支撑 driving Gaussian semantic graph 的核心假设，因此按预注册同时 reject raw LUDVIG graph，
  不得降低 actor minimum、删除 1087、只报 reprojection 或先加 Bayesian/SAM/motion edge 救 graph。下一合法路线是独立
  faithful progressive propagation；S/C/validation/test/KITTI 仍未读。
- `V51-F32`（`engineering/launcher`, `resolved before D0 preregistration`）：首次 D0 只读 inventory 把含
  `U2|B3|Bayesian|...` 的正则直接嵌入 PowerShell→SSH 命令，外层 shell 抢先解释 `|`，导致远端 `sed` address
  截断并把各 regex 分支误当命令；没有文件、run、GPU 或 quality 状态变化。后续统一把多行只读脚本 UTF-8 base64
  编码后交给远端 `bash`，避免跨 shell parser 改写。不得把 launcher quoting failure 计入 D0 方法 verdict。
- `V51-F33`（`engineering/runtime`, `resolved before D0 preregistration`）：第二次 artifact inventory 在远端使用裸
  `python`，但主机 PATH 按合同没有该命令；三个 YAML path 已被 `rg` 只读打印，内嵌解析均未运行，也未改状态。
  修正为显式 `/root/autodl-tmp/envs/motionproj/bin/python` 后完成 NPZ identity/count/quantile 审计。后续所有 V5.1
  runner/auditor 必须使用冻结解释器绝对路径，禁止把 shell PATH 差异写成数据或算法失败。2026-08-20 收尾清理预审
  首次又假设 `/usr/bin/python3` 存在并在任何 inventory 写入前失败；随后显式使用 `/root/miniconda3/bin/python` 运行同一
  审计，候选集合未变化。该复发没有研究资产状态变化，进一步要求一次性维护的 cleanup 工具也必须绑定已探测的绝对解释器。
- `V51-F34`（`engineering/runtime`, `resolved before D0 preregistration`）：D0 扩大回归时又用 motionproj Python
  调用了依赖 DriveStudio runtime identity 的 H evaluation config test，得到该项 runtime drift；D0 新测试本身已
  `4/4 PASS`，没有 formal run、GPU 或 quality read。这是 `V51-F24` 的重复触发，说明仅在文档记环境分层不足。
  r020 freeze 的扩大回归再次用 motionproj 聚合 26 个 V5.1 test files，结果 `95 passed / 3 runtime-drift failed`；失败项
  正是三个已冻结为 DriveStudio Python 的旧 H tests，本次 E0a 定向 8 项均 PASS。按 suite 拆分后再用 frozen DriveStudio
  interpreter 补跑 3 项通过。后续回归命令必须在生成 file list 时就按 runtime 分组并分别记录结果；不得改 runtime
  freeze 来迁就聚合命令，也不得把这一环境错误写成算法 regression。
- `V51-F35`（`engineering/protocol`, `resolved before D0 preregistration`）：同一扩大回归发现 P0 scope 与 Stage-B
  authorization 仍保留最初 top-plan SHA=`3d7f7481...`，而 commit `b359541` 为预注册 H heldout contract 对计划做了
  17 行 append-only 更新，当前 SHA=`b4888476...`；旧 validator 只允许单 hash，导致 4 个既有 protocol tests 在真实
  route assertion 前 fail。禁止改写两份历史 freeze 的 recorded SHA。修复在 validator 中显式固定
  `base hash → authorized append hash` 两状态链，第三种状态仍 fail-closed，并增加 current/historical 双 hash 回归。
- `V51-F36`（`engineering/test`, `resolved before formal D0 operator run`）：首个 progressive expansion unit fixture
  预期 `p=0.5` 节点最终 UNKNOWN，却把该节点直接连到 `p=0.01` Background seed。两者 L2-normalized binary
  distribution cosine 约 `0.714`，高于冻结最低 threshold=`0.5`，故算子把它合法扩张为 Background；失败的是 fixture
  的“无支持”假设，不是算法。修复只删除这条边，使该节点真正孤立；其余阈值/公式/实现不变，5/5 operator tests PASS。
  禁止为满足错误预期而提高阈值、加入 confidence gate 或改变 UNKNOWN 语义。
- `V51-F37`（`algorithm/evaluation`, `active; D0 rejected by r018`）：faithful SAI3D-style raw-Gaussian progressive
  propagation 在 frozen H matched 12 views 上只通过 BF1 两项门：positive scene=`2/3`、scene-balanced BF1=
  `+0.0002196`；IoU=`-0.0714543<0` 与 FN semantic mass=`+0.1694766>+0.02` 同时 FAIL。0471 的
  BF1/IoU 有改善，但 FN 仍 `+0.080830`；1087/0379 的 IoU=`-0.159417/-0.220397`、FN=
  `+0.218146/+0.209454`，说明减少 FP/提高部分 calibration 并不能补偿 actor 漏检和跨场不稳定。这推翻“在 raw
  Gaussian 上按冻结 KNN 与多视图 SAM affinity 做 progressive growing 可稳定优于 U2/B3”的假设，不推翻所有 graph
  或 super-primitive 路线。D1 永久跳过；禁止按 r018 调 thresholds/hops/seeds/affinity 或重读 H。合法后续只有按冻结
  顺序进入 Stage E，先以 no-quality E0 检查 node elevation 是否提高 observation density，再按其门禁决定 E1/E2。
  证据：r018，source=`2cd98b3`，summary/manifest=`b08c7276...62d6/792660e3...010c`，independent metric/gate
  replay=`18c12f4d...0d2`，freeze=`configs/worldsim_v51/stage_d_progressive_h_evaluation_freeze_v1.yaml`。
- `V51-F38`（`engineering/shell`, `resolved during r019 monitoring`）：旁路进度查询再次在 PowerShell→SSH 边界使用
  `$run`，本地 shell 先展开变量并破坏远端引号，得到 `unexpected EOF`；正式 r019 进程独立运行、未受影响，也没有
  repo/run 写入。后续监控只能使用绝对字面路径或仓库内 CLI，禁止跨 shell 传未编码变量。这是 `V51-F32` 的复发，
  说明“已知坑”仍需由可执行入口而非记忆约束。
- `V51-F39`（`engineering/data-contract`, `resolved by v2/r020`）：formal r019 在 0471/1087 完成后，
  因 0379 frozen KNN 含 `34/7,123,746` zero-length edges 而 blocked；v1 把“用于 voxel scale 的 edge length”错误
  写成全量严格正值。三场 nonfinite edge 均为 0，0379 仍有 `7,123,712` positive edges，因此这是 quantile 输入合同
  过强，不是算法质量、OOM 或 corrupted geometry。r019 terminal/13 files 保留，partial assignments 禁止晋级/复用。
  合法 recovery v2 只排除零长 edge 的 scale statistic，保留全部 Gaussian，其他 quantiles/gate/views/locks byte-semantic
  继承，并以新 r020 完整重跑；r020 三场/九档 assignment、metrics 与 gate 独立复算 exact，report=`8df03b2a...5d34`。
  防重复边界仍成立：不得把 zero edge 直接删除出后续 topology，也不得借 recovery 改 voxel level。
- `V51-F40`（`engineering/shell`, `resolved with enforced command boundary during r020 freeze`）：量化 F39 时首次 base64 远端 Python 命令仍错误嵌入双引号，
  PowerShell 把 `base64.b64decode(...)` 当作本地命令，远端再次未执行。修复不是继续堆转义，而是新增可测试的只读
  `scripts/audit_worldsim_v51_e0a_edges.py`；CLI test PASS，三场 edge identity/zero/nonfinite/positive quantile 审计完成，
  report=`30493d5d...bc5`。r020 freeze 时又误用一次跨 PowerShell/SSH inline `python -c`，只产生远端 SyntaxError、未写入
  run；随即改为本地解析 YAML、远端只运行仓库 CLI/pytest。后续需要多语句远端分析时必须先落地仓库内 auditor，
  禁止临时内嵌脚本；单语句也不得跨两层 shell 手写嵌套引号。r022 审计前的旁路摘要查询再次因 heredoc 嵌套引号
  得到 `unexpected EOF`，随后发现远端没有 `jq`；两者均未写 run。改为 scp 冻结 JSON 后在本地只读解析，并由仓库
  auditor 完成正式审计。进入 Stage F 后又有一次含 `$f` 的远端 loop 被 PowerShell 提前展开，以及一次嵌套
  `python -c` 验证命令 SyntaxError；均在 formal r023 前、无 run/asset/repo 状态变化。该复发进一步说明远端临时解析
  不是证据入口；正式 source audit 必须由仓库 runner 完成。r025 freeze 上传后的 YAML smoke 又因 PowerShell 中手写
  `python -c` 反斜杠转义得到 `unterminated string literal`，紧接着尝试 stdin 单层命令仍被本地 quote stripping 破坏；
  两次测试链都在该点停止且未修改 run/repo 文件。最终不再修补 shell quoting，改为仓库 pytest 直接加载 freeze YAML，
  并配合 `git diff --check` 复核。该 recurrence 不影响此前已 PASS 的 r025 独立 auditor。2026-08-20 cleanup inventory 又有
  两次 inline `awk`/shell-loop 因 PowerShell→SSH quoting 失败；两次均为只读、没有创建/修改/删除研究资产。最终把审计和
  fail-closed deletion 放入固定 Python 文件，以 exact JSON plan 执行。禁止再为临时汇总跨两层 shell 拼循环、变量或 awk。
- `V51-F41`（`engineering/environment`, `resolved during r020 audit`）：本地 `autodl-stage/motion_proj` 只是按约束用于
  `apply_patch` 的 partial staging tree，不包含完整 `motion_proj.worldsim_v5` package；误在该目录收集 E0a 联合测试时触发
  `ModuleNotFoundError`。这不是 canonical repo、r020 或 auditor 失败。修复为只在本地做语法检查/编辑，将新增文件同步到
  远端完整 clean checkout 后运行同一测试，结果 `8 passed`；随后 r020 独立审计通过。后续不得把 partial staging 当作
  可运行 checkout，也不得为迎合该环境复制缺失 package 或修改 import path。r022 审计阶段在同一 staging tree 误跑
  `git diff --check`，因它不含 `.git` 只打印 usage；命令没有修改文件，正式 CLI test 与审计仍在远端完整 checkout PASS。
- `V51-F42`（`algorithm/evaluation`, `active; E0B rejected by r022`）：simple voxel super-primitive control 的
  `fine_q50 + member-unary mean + visibility-weighted SAM mean + max visibility + frozen D0 propagation` 在 frozen H matched
  12 views 上未能优于 U2/B3，也未能稳定优于 raw D0。相对 U2/B3 虽有 BF1 positive scenes=`2/3`，scene-balanced
  BF1=`-0.0002566`、IoU=`-0.0925468`、FN=`+0.1899473` 全部 FAIL；相对 D0 的 BF1 nonnegative scenes 仅
  `1/3`，mean BF1=`-0.0004762`、IoU=`-0.0210926`、FN=`+0.0204707`，四项机制门全 FAIL。0379 相对 D0
  `ΔIoU=-0.064618 / ΔFN=+0.067752`，说明确定性 voxel 合并与 member evidence 平均会扩散弱/错误证据，结构密度提升
  不能推出语义质量提升；1087 的近 no-op 也未形成可泛化收益。该结果只推翻这套 simple node-elevation 实例，不推翻
  faithful Gaussian Grouping 或所有 graph/node 方法。E1 PanoGS 与 E2 AG²aussian 按预注册停止，禁止依据 r022 调
  voxel level、node aggregation、seed/threshold/hop、删除 0379 或重读 H；下一合法路线是 Gaussian Grouping faithful
  source audit 与 no-quality preflight。证据：r022 summary/manifest=`4964a2f0...3d4/3c5a2fbe...7aa`，independent dual-gate
  replay=`5ced73db...104f`，freeze=`configs/worldsim_v51/stage_e_e0b_h_evaluation_freeze_v1.yaml`。
- `V51-F43`（`engineering/tooling`, `resolved before F0 preregistration`）：下载并哈希 Gaussian Grouping official PDF 后，
  远端没有 `pdfinfo/pdftotext`；桌面依赖清单给出的 Poppler override/fallback 也不可执行。改用 bundled Python 的
  `pdfplumber` 读 18 页，但首次输出受 Windows GBK 限制，遇到作者脚注符号触发 `UnicodeEncodeError`；只设置任务级
  `PYTHONIOENCODING=utf-8` 后完成全文提取，并用已存在的 `pypdfium2` 渲染方法第 6–8 页检查公式与图示。没有安装
  系统包、没有改 PDF、也未触及方法数据。后续 PDF source audit 优先复用 bundled Python 并显式 UTF-8，不假设远端或
  dependency catalog 中声明的 Poppler binary 实际存在；工具缺失不得写成论文或算法证据。
- `V51-F44`（`engineering/runner`, `resolved by r024`）：F0 source preflight runner 复用 Stage-B `_git`
  helper 时写成 `_git("rev-parse", "HEAD")`，但 helper 签名是 `_git(project, *args)`，实际调用变成
  `git -C rev-parse HEAD` 并在 source commit 读取处失败。r023 此时只创建 run 目录和 `resolved_config.yaml=7,796`
  bytes，尚未写 status、启动 resource monitor、读取 source/data/schema、运行 CUDA smoke 或读取任何 quality；它是不可晋级
  的 incomplete shell。合法 recovery 只新增 `repository_source_identity(PROJECT)` 显式绑定与参数顺序回归，用新 clean
  commit/r024 从头运行；r024 已越过 source identity 并执行到 adapter smoke 前，证明本项修复有效。不得删除 r023、
  手补 terminal 或借机改变 F0 source/method/data contract。
- `V51-F45`（`engineering/runtime`, `resolved by r025`）：r024 完成 official source identity、代码语义与三场
  train-only metadata/observation schema 的内存检查后，在 16D adapter smoke 前直接调用
  `torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))`；当前进程尚未初始化 CUDA context，PyTorch 返回
  `Invalid device argument 0: did you call init?`。r024 status=`blocked`，只有 resolved/status/events/resource samples
  四文件=`9,449 bytes`，没有 source report、CUDA render、SAM/DEVA/identity training 或 quality read。合法 recovery 只按
  `set_device → one-scalar allocation/context init → reset peak → smoke` 顺序执行并加入 call-order regression，新 r025 从头
  重跑；r025 与独立 replay 均得到 `[1,32,32,16]` render、`189` positive-alpha pixels、`48/48` identity gradients、
  base gradients absent，GPU peak=`310 MiB`，证明初始化顺序修复有效。禁止放宽 resource ceiling 或把已在 blocked run
  内存检查过的数据结果晋级；canonical 只认 r025。
- `V51-F46`（`data-contract/algorithm`, `active prerequisite after r025`）：Gaussian Grouping official identity mechanism
  需要 SAM everything masks 经 DEVA semionline 关联后的跨视图一致 short IDs；r025 独立核对三场各 15 个 train-only
  observation 后，45/45 只有 binary actor-union probability 等同一套 23 fields，没有任何
  `instance/identity/object_id/mask_id/class_id` label。`instances_info/frame_instances` 中的 stable actor token 只描述场景级
  track metadata，不提供每像素 mask；三个 formal checkpoint 也只能提供已训练 Gaussian state，不能反推出监督标签。
  同时 official DEVA propagation 与 SAM ViT-H weights 均 absent，现有 SAM2.1 Hiera Large 虽有 checkpoint，但不符合上游
  SAM-v1 everything-mode source contract。该缺口不否定 source core 或 frozen-base 16D adapter（两者 r025 PASS），但
  `current_training_input_ready=false /identity_training_authorized=false`。合法下一步只能先预注册并执行 train-only F0a
  asset acquisition + SAM/DEVA identity-mask materialization，冻结 URL/SHA、输出 schema、确定性、资源与 partial recovery；
  禁止用 metadata、binary U2/B3、SAM2 或 evaluation target 代替，禁止在 materialization 冻结前启动 F0 training。
  证据：r025 summary=`da4890d...988`、audit=`14d2b78b...8b64`、
  `configs/worldsim_v51/stage_f_f0_source_preflight_freeze_v1.yaml`。
- `V51-F47`（`engineering/orchestration/tooling`, `resolved during r026`）：首次启动 r026 时把远端 Linux command 与
  `/root/autodl-tmp/motion_proj` workdir 直接交给桌面本地 `bash` tool，Windows 在建立 SSH 前返回
  `CreateProcess ... 目录名称无效`；远端 run/path/assets 均尚不存在，因此这不是 r026 blocked terminal，修复为从本地
  PowerShell 显式 `ssh wm-3090-0811` 后按原 prereg run ID 启动。r026 完成后只读汇总文件字节时又假设远端存在 `bc`，
  在已打印 auditor hash 与 status/manifest size 后报 `bc: command not found`；它没有修改 run，完整字节数改由已审计的
  manifest inventory 加 status/manifest exact size 得到 `51,021`。这两次都推翻“tool shell/workdir 与常用 CLI 可跨本地/
  远端默认存在”的工程假设，不影响 r026 的 `done` 或 asset hashes。后续远端命令必须从 Windows 使用 SSH alias，run
  证据计算由仓库 runner/auditor 完成，禁止临时依赖未冻结的 `bc/jq/python -c`。证据：r026 audit=`5a360f42...817c`，
  freeze=`configs/worldsim_v51/stage_f_f0a_asset_source_acquisition_freeze_v1.yaml`。
- `V51-F48`（`engineering/data-contract`, `resolved before formal r027`）：r027 prereg config 首稿手录
  scene-0471/frame0/camera0 SHA 时，把 canonical `093d38e8d8d8f12...819e` 漏写一个 `d8` 成
  `093d38e8d8f12...819e`。formal-config pytest 在任何 r027 run 目录、wheel/env mutation、GPU/model/image decode 前
  fail-closed；远端 `sha256sum` 与 r026 已独立审计的 selected manifest 一致，证明是配置转录错误而非图像漂移。修复只从
  r026 canonical record 恢复完整 SHA，并以原测试重跑；不得改图、重选 view、放宽 hash 或把该失败写成 SAM/DEVA 结果。
  后续长 identity 必须由 manifest 机器传递并保留 config-validation test，禁止凭聊天摘要/截断 hash 手工补全。
- `V51-F49`（`engineering/license/runtime`, `resolved by r028/r029`）：r027 已 atomic 构建 isolated venv 并安装
  exact `supervision=0.14.0/PuLP=2.7.0/gurobipy=10.0.3`，但第一个 Gurobi tiny MILP 在创建 model 时报告
  `License expired 2024-10-28`，因此 status=`blocked`。失败发生在 one-view upstream CLI 前：没有加载 DEVA/SAM 权重、
  没有 GPU model forward、没有 decode input/mask、没有 quality；4 files=`12,861 bytes`，不得把它写成 identity mechanism
  或 association 失败。根因是把上游 `gurobipy>=10.0.3` 的最低版本误冻结成 exact 10.0.3，而该 wheel 的内置 restricted
  runtime 已过期；服务器没有另一个 `gurobi.lic` 可续用。合法 recovery v2 只将 Gurobi 提升到当前 index 可得的
  `12.0.3`（仍满足上游版本下界），使用全新 wheelhouse/venv/r028 重跑，其他 source/assets/input/CLI/resource/locks
  不变；仍要求 Gurobi tiny optimum，禁止直接跳过 gate 或静默采用 PuLP 后声称 faithful。r028/r029 的 Gurobi 12.0.3
  tiny model 均得到 status=`2`、solution=`1.0`，因此 license 前置已解除；后续 stdout 与 GPU failures 分别独立记账。
- `V51-F50`（`engineering/source-provenance`, `resolved before r028`）：r027 environment path verification 用子 Python
  import frozen DEVA source，默认 bytecode policy 在 Gaussian Grouping checkout 内生成 4 个未跟踪 `__pycache__` 目录；
  tracked diff 为 0，但后续 v2 config 的 clean-source gate 正确 fail-closed。该污染不是 upstream 修改、算法失败或 r028 run；
  在确认 exact paths 后只删除这 4 个由本任务生成的 cache，并让 runner 的全部子进程继承
  `PYTHONDONTWRITEBYTECODE=1`。第一次清理 wrapper 又因 PowerShell 提前处理 `$p` 而出现 quote EOF，未删除或修改任何
  文件；随后改用 4 个显式绝对 target 完成清理，两个 external repo 恢复 clean。禁止把 `git status` 放宽为忽略 untracked，
  也不得把 source tree 内 cache 纳入冻结；后续 import smoke 必须同时核对 commit/tree/clean。
- `V51-F51`（`engineering/runner`, `resolved by r029`）：v2/r028 成功获取并安装
  `gurobipy=12.0.3`，Gurobi 不再抛 license-expired/CalledProcessError；但 restricted-license banner 与 runner 自己打印的
  JSON 同时写 stdout，v2 对整段 `json.loads` 而得到 `JSONDecodeError line 1 column 1`。r028 因此在 solver output parse
  blocked，4 files=`10,642 bytes`；one-view CLI、DEVA/SAM load、input/mask decode 与 quality 仍全部未发生，不能把它写成
  solver、association 或算法失败。v3/r029 唯一修复是解析最后一个非空 stdout 行为 JSON，并把前置 banner 原样存档；
  solver status=`OPTIMAL` 与 solution=`1` 的门不放宽，环境/权重/view/CLI/resource/locks 全继承 v2。禁止粗暴丢弃全部
  stdout、用正则猜数字或因 parser failure 绕过 Gurobi gate。r029 已越过两种 solver gate 并启动 official CLI，证明
  terminal-line parser 恢复有效；r029 后续 OOM 另记 `V51-F52`，不得倒写本项未解决。
- `V51-F52`（`engineering/resource`, `active; allocator-only recovery disproved by r030`）：r029 首次完整越过 environment、Gurobi/PuLP 和
  official model-load gate，在唯一 scene-0471/frame0/camera0 输入上执行 SAM ViT-H everything mask；上游默认
  points-per-side/batch=`64/64`，在 `BatchMaskData.cat` 尝试再分配 `6.74 GiB` 时 CUDA OOM。错误现场为 GPU total/free=
  `23.56/6.72 GiB`、process=`16.83 GiB`、PyTorch allocated/reserved-unallocated=`10.74/5.77 GiB`，独立 resource samples
  peak=`17,246 MiB`。r029 status=`blocked`，6 files=`22,458 bytes`，没有 mask、`pred.json`、identity quality 或 cross-view
  association denominator；这推翻“24GB 可直接运行 official default one-view”的资源假设，不推翻 Gaussian Grouping 算法。
  第一合法 recovery/r030 只设置 allocator `max_split_size_mb:128`，source/view/size/IoU/grid/batch/ceilings 均不变；若仍
  OOM，必须保留 r030 后另行预注册 `SAM_NUM_POINTS_PER_BATCH` 单变量 batching adaptation，并补 parity/repeatability，禁止
  同轮缩图、改 grid/阈值或把工程失败写成 algorithm reject。r030 已证明该 allocator-only recovery 不足，后续事实续记
  `V51-F55`。证据：r029 stderr=`205266c9...403`、resource=`a0d825a8...01a`、status=`ff75badc...f66`。
- `V51-F53`（`engineering/source-provenance`, `resolved by r030 asset publish`）：r029 模型构造首次触发冻结 DEVA source
  `deva/model/resnet.py` 中的 `model_zoo.load_url`，隐式从 PyTorch URL 下载 ResNet50/18 到用户级 `/root/.cache`。两份
  资产分别为 `102,502,400 bytes /19c8e357...097` 与 `46,827,520 bytes /5c106cde...13f8`；此前 F0a asset freeze 未枚举
  这项 transitive dependency，故“官方 DEVA+SAM 两权重已经覆盖全部模型资产”的来源合同被推翻。合法 recovery 固定 source
  literal URL、bytes/full SHA，把 exact cache 原子复制到专用
  `/root/autodl-tmp/models/gaussian_grouping_v51_stage_f/torch_home/hub/checkpoints`，并让 subprocess 固定 `TORCH_HOME`；禁止
  继续依赖用户 cache、重新下载未验 hash、修改 frozen upstream 或把 ResNet 权重混称 DEVA checkpoint。canonical 目标独立
  审计前不得删除原 cache；审计后只允许精确清理由本次生成且 hash 匹配的两个源文件。r030 前置已把两份资产原子发布到
  dedicated target；独立复核 bytes/full SHA、无 `.partial`，且 official CLI stderr 没有 download 行，证明 `TORCH_HOME`
  生效。本项来源缺口已解除。r032 canonical audit PASS 后，两份原 cache 源副本已按 exact path 精确删除；dedicated
  `TORCH_HOME` copies full SHA 保持，后续不再依赖用户 cache 或网络。
- `V51-F54`（`engineering/orchestration`, `resolved before prereg commit`）：第一次提交 v4 时把包含括号与多段正文的
  `git commit -m` 放进 Windows PowerShell→SSH→bash 双层命令；本地层提前剥掉远端参数引号，bash 在 Conventional Commit
  标题的 `(` 前直接 syntax error。失败发生在 Git 创建 commit、push、r030 run 或 canonical asset publish 之前；staged diff
  保持不变，不是 config、test 或算法失败。这是 `V51-F47` 跨 shell 合同的复发：恢复改为在本地用 patch 生成独立 commit
  message 文件、scp 到远端临时路径，再以 `git commit -F` 单参数读取；禁止继续手工嵌套长 `-m`、省略正文或覆盖 staged
  内容。提交后必须精确删除临时 message，并重查 branch/status/commit。
- `V51-F55`（`engineering/resource`, `active; batch-only recovery disproved by r031`）：r030 在 source=`33c013d` 上保持同一 input、官方
  point grid/batch=`64/64` 与全部方法参数，仅增加 `max_split_size_mb:128`。reserved-unallocated 已由 r029 的 `5.77 GiB`
  降到 `578.72 MiB`，说明碎片明显减少，但同一 `BatchMaskData.cat` 仍尝试分配 `9.49 GiB`，free=`9.16 GiB` 而 OOM；
  sampled GPU peak=`24,098 MiB`，超过 prereg `24,000 MiB`，cgroup=`18,035,429,376 bytes`，101 samples/0 errors。
  6 files=`25,873 bytes`，没有 mask、metadata 或 quality。该结果推翻 allocator-only 足以让 official default batch 在 3090
  上运行的假设，也提示主要约束已不是 allocator fragmentation；仍不构成算法 reject。下一合法 recovery/r031 只把上游
  文档明确称为 parallel point prompts 的 `SAM_NUM_POINTS_PER_BATCH=64→32`，保留 points-per-side=`64`，并要求成功后另做
  batch parity/repeatability；禁止同轮改 grid、size、IoU、模型或资源 gate。证据：r030 stderr=`15d9bd12...bef`、resource=
  `39060722...6ad`、status=`d38fd753...0f7`。r031 已证明 batch32 仍不足，后续累计规模事实续记 `V51-F57`。
- `V51-F56`（`engineering/orchestration`, `resolved before r031 prereg`）：为核对 batch 参数来源所发的只读 `rg` 命令在
  Windows PowerShell→SSH→bash 双层字符串内包含 alternation `|`，引号被提前剥离后 bash 把后半段当命令，返回
  `points_per_batch: command not found`。没有 repo/run/asset 状态变化；恢复为单关键词、无 pipe 的 `rg`，定位到 DEVA
  `docs/DEMO.md`、`ext_eval_args.py` 和 `automatic_sam.py`：参数默认 64，定义为每批并行 point prompts，并直接传给 SAM
  `points_per_batch`。今后临时只读 SSH 查询也必须避开嵌套 alternation/pipe，复杂查询落到本地或正式 auditor。
- `V51-F57`（`engineering/resource`, `resolved for 24GB by r032 grid32; default-grid boundary remains`）：r031 精确执行 v5 的唯一变化
  `SAM_NUM_POINTS_PER_BATCH=64→32`，stdout 确认 side/batch=`64/32`，但仍在同一 `MaskData.cat` 累积点 OOM：request/free=
  `9.32/9.31 GiB`、allocated/reserved-unallocated=`13.34 GiB/599.11 MiB`，GPU peak=`24,066 MiB`、cgroup peak=
  `18,052,734,976 bytes`、119 samples/0 errors。6 files=`28,677 bytes`，mask/metadata/quality 均 absent。相对 r030 的
  `9.49 GiB` request 仅下降约 `0.17 GiB`，推翻“缩并行 batch 可解决最终累计 masks 峰值”的假设；继续 batch16 是重复
  调参，未授权。源码表明每批 full-resolution masks 在 NMS 前累积，规模主要受 points-per-side² 控制；DEVA 官方文档又明确
  建议降低 `SAM_NUM_POINTS_PER_SIDE` 来减少 automatic queries。下一合法 recovery/r032 只设 side=`32`（1024 prompts），
  batch=`32` 与其他参数/门禁不动；它必须标作 documented resource adaptation，成功后需同-grid batch parity、3-view
  association/repeatability 和后续质量门，禁止把 resource PASS 冒充 default-grid parity。证据：r031 stderr=
  `b822aab6...692`、resource=`0a06475d...af4`、status=`99d081ee...e23`。r032 以 side/batch=`32/32` 在 GPU peak
  `23,954 MiB` 内完成 output schema，解除当前 24GB execution prerequisite；但 default grid64 仍不可运行，grid32 quality/
  association 尚未证明，不能删除 r029–r031 或声称 exact-default parity。
- `V51-F58`（`engineering/orchestration`, `resolved after r032 audit`）：按 `V51-F53` 清理门禁删除两份用户 cache 前，首次
  wrapper 用嵌套 `$(sha256sum ... | cut -d " " -f1)`；PowerShell/SSH/bash 再次破坏 delimiter 引号，`cut` 在第一个
  `&&` 前退出，两份文件均未删除。已有 r032 independent audit 保存 source/canonical full SHA，恢复改为无 pipe/无命令替换
  的两个显式 `rm -f`，随后验证源路径 absent 且 canonical SHA 分别保持 `5c106cde...13f8/19c8e357...0097`。禁止在双层
  shell 中拼 checksum parser；以后先由 auditor 落证据，再用 exact path 单动作清理并独立验证。
- `V51-F59`（`protocol/data-contract`, `resolved by r033 association subgate; one-view boundary remains`）：r032 的 mask 是合法 `900×1600 uint8`，但 histogram=
  `{0:1,440,000}`。这不是 SAM grid32 quality reject：唯一输入少于 semionline `num_voting_frames=3`，upstream flush 没有形成
  cross-view consensus，因此 all-background 正是预先声明的 one-view 边界。它同时推翻“one-view resource PASS 可证明
  identity masks ready”的隐含推断。下一步必须在冻结 grid32 上做 same-grid batch parity，并用至少 3 个按时序排序的
  train-only views 检查 non-empty masks、stable short IDs、repeatability 与资源；在此之前禁止 full materialization、identity
  training 或把 annotation_count=1 当成实例覆盖。证据：r032 mask=`0bf854a1...59d`、audit=`cebe07fd...cd5`、freeze=
  `configs/worldsim_v51/stage_f_f0a_environment_one_view_smoke_freeze_v1.yaml`。r033 在同一 grid32/batch32 上用冻结的
  frame=`0/40/80` 完成 3-frame voting：三张 mask 全 non-empty，19 个 positive short IDs 至少跨 2 帧，且 batch32 repeat
  三 mask/metadata bit-exact，因此“one-view 不能证明 identity input”的前置边界已解除；该证据不读质量，也不解除 r033
  的 batch/resource 失败（`V51-F60/F61`）。
- `V51-F60`（`algorithm/implementation-contract`, `resolved for current method selection by r034; sensitivity boundary remains`）：r033 预注册把同 grid32 的
  `SAM_NUM_POINTS_PER_BATCH=32→16` 视为 execution-memory parity 臂，但 batch16 与 batch32 的三张短 ID mask 和 `pred.json`
  全不 exact。逐帧不同 label pixels=`208,647/288,527/244,696`，exact fraction=`0.855106/0.799634/0.830072`，binary
  foreground IoU=`0.961177/0.995201/0.969622`；batch32 IDs 含 `36/62/95`，batch16 含 `13/63/96`。与此同时
  batch32 primary↔repeat 的三张 mask 和 metadata 均 bit-exact，association/non-empty 也通过，故差异不能归因于无约束随机
  重跑。已确认的推翻项是“batch 只改变显存、不改变输出”；更深机制可能涉及 AMP batch-shape 数值路径与候选/NMS 边界，
  但 r033 没有证明具体源码根因。禁止放宽 exact 门、用高 foreground IoU 冒充 identity parity，或从 batch16/32 中按结果
  挑一臂。合法恢复只允许在新 run 恢复 upstream default batch64、保持 grid32/输入/阈值不变并独立检查 repeatability、
  association 与资源；若 batch64 不可运行，则该 faithful-input 路径必须保持 blocked 而不是继续调 batch。证据：r033 source=
  `191d3e4...12f`、parity=`7a6db15f...7ae`、audit=`a5a7d5c8...fa7d`。r034 恢复 upstream default batch64，保持
  grid32/三帧/其他参数，primary↔repeat 三 mask/metadata bit-exact、association PASS；当前方法选择因此固定 batch64，不再
  依赖被证伪的 batch32 execution-only 解释。本条的 batch-sensitive 事实仍是 active anti-regression boundary：后续任何
  batch 改动都是方法变化，必须新协议且不能用 r034 质量作等价担保。
- `V51-F61`（`engineering/resource/protocol`, `resolved by r034 physical-headroom contract; r033 remains failed`）：r033 三臂串行而非并发，independent audit 从 241 条
  resource samples 重算 NVIDIA peak=`24,116 MiB > 24,000 MiB`，超过预注册门 116 MiB；cgroup peak=
  `17,956,044,800 bytes`、event wall=`78.917s`、monitor errors=`0`。runner 先因 parity fail-closed，故没有执行后置 resource
  adjudication 或发布 `resources.json/summary/manifest`；这不能让已记录的 GPU 越门消失。它推翻“r032 one-view peak
  23,954 MiB 可直接外推到三视图三臂合同”的资源假设，不是 mask quality reject。禁止倒写 r033 为资源 PASS、事后把旧门
  提到 24,116，或把无 OOM 等同有安全余量。新 batch64 smoke 若修改 ceiling，必须在启动前绑定卡总显存、明确保留 headroom
  与复开理由；若 OOM/越新门则停止该 recovery，不得继续缩 batch 回到已证伪的 execution-only 解释。证据：r033 resource=
  `db4e6d17...8e9c`、status=`e027888e...7234`、audit=`22,939 bytes /a5a7d5c8...fa7d`。r034 没有修改 r033 门，
  而是新预注册 card total=`24,576 MiB`、minimum headroom=`256 MiB`、peak ceiling=`24,320 MiB`；upstream batch64 两臂
  实测 peak=`24,092 MiB`、headroom=`484 MiB`、cgroup=`17,957,322,752 bytes`、142 samples/0 errors，全门通过并由
  audit=`e0988f50...5258` 重放。因此当前三视图 upstream-batch64 resource prerequisite resolved；45-view materialization
  仍需独立总时长/磁盘/输出分母门，不能从两臂 smoke 外推。
- `V51-F62`（`engineering/CUDA-runtime/resource-boundary`, `resolved_recovery; root_cause_unproven`）：r035 依冻结顺序串行做三场 45-view train-only
  materialization；0471 已完成 `15 masks +pred.json`，但 1087 official grid32/upstream-batch64/AMP subprocess 处理前两张
  后，在第三张触发 three-frame vote，于 `consensus_associated.py:58 spatial_alignment` 的 `value @ affinity` 返回
  `CUDA CUBLAS_STATUS_INTERNAL_ERROR / cublasGemmStridedBatchedExFix`，因此 1087 没有 canonical mask/pred/report，0379
  未启动。该错误不是显式 PyTorch OOM；resource samples 重放 peak=`24,124/24,576 MiB`、headroom=`452 MiB`、cgroup=
  `17,961,271,296 bytes`、174 samples/0 errors，仍通过 r035 预注册数值门，所以现阶段既不能武断归因 OOM，也不能用
  headroom 数值排除 allocator/CUBLAS workspace/driver 异常。它推翻了“r034 同 batch64 三视图 PASS 可直接外推任意场景
  的 45-view execution stability”，但没有推翻 Gaussian Grouping identity 算法或证明 mask quality 失败。禁止把 0471 的
  `15/45` partial 写成 full materialization、原地续跑/覆盖 r035、跳过 1087、改场景顺序、缩 batch 回到已证明会改变输出
  的配置，或读取 partial quality 再选 recovery。合法下一步只能新预注册 exact 1087 `000_0/000_1/000_2` 三视图，保持
  grid32/batch64/AMP/size480/thresholds 并启用 `CUDA_LAUNCH_BLOCKING=1` 定位是否可重放；diagnostic 输出不得进入质量或
  training。证据：r035 source=`e4d64d3...1424`、status/events/resource-samples=`c3f917bd...f61/7d2221b5...0b7/
  d46d632d...4e2`、stderr=`f626efc6...8a5`、audit=`25,311 bytes /6d217a7e...13e1 /PASS`。r036 在 exact
  1087 首三视图、相同 method 且 `CUDA_LAUNCH_BLOCKING=1` 下串行 fresh-process replay 两次：第一次在相同 GEMM
  位置复现、第二次成功并生成 3 个 schema-valid mask/pred，故 deterministic input/shape 必现假设被推翻；与此同时一成一败
  证明 runtime 还不具备 materialization 所需的 repeatable execution。r036 resource 仍 PASS，成功输出未读质量且不能补写
  r035。下一步先做预注册 runtime health/control-vs-target reproducibility gate；禁止用第二次偶然成功直接重启 45-view。
  r036 evidence=`summary 32e59c85...3ea /audit 5,077 bytes, ec7cfa36...34f6 /freeze
  configs/worldsim_v51/stage_f_f0e_scene1087_cuda_fault_localization_freeze_v1.yaml`。r037 用同卡、同 frozen method、同
  `CUDA_LAUNCH_BLOCKING=1` 做 A–B–A–B：两次 0471 known-good control 均成功、mask/pred 彼此 exact 且与 r034 SHA
  相同；夹在其间的两次 1087 target 均在相同 CUBLAS site 失败。故“整张 GPU/所有三视图已普遍失效”不成立，失败收窄为
  target-path process instability；但 r036 曾有一次 target success，仍不能倒写 deterministic data failure。ECC/page/row 对
  RTX3090 为 N/A，dmesg 又无权限，二者都不能冒充健康证明。合法下一步是 source-neutral trace，在不改 upstream 文件/
  tensor 内容/方法参数的前提下记录 control/target matmul tensor metadata 与 allocator 状态；任何 trace 输出仍不得参与质量或
  training。r037 evidence=`summary 5fd4a4e8...df8 /audit 8,245 bytes, 2fb76f32...d50d /freeze
  configs/worldsim_v51/stage_f_f0f_cuda_runtime_health_reproducibility_freeze_v1.yaml`。r038 source-neutral trace 的 control/
  target 都成功且输出分别 exact 对齐 r034/r036 success；control 两个 matmul 是 `26/36 objects`，target 是 `3/52`，两侧
  affinity 都为 `[1,1620,1620]`，故“target 首个 matmul 更大所以必败”被直接推翻。更关键的 allocator observation 是：
  control pre-matmul driver-free 仅约 `35/55 MiB`、allocator retry=`0`；target success process 已发生一次 allocator retry，
  cache 被释放后 pre-matmul free 约 `18.15/17.23 GiB`。这使 allocator-cache/CUBLAS-workspace state 成为有证据的 active
  hypothesis，但不是根因证明：trace timing 可能扰动执行，且 control 在低 free 下仍成功。合法 recovery 只允许预注册在
  frozen line58 matmul 前执行 `torch.cuda.empty_cache()`，不改 tensor/operator/grid/batch/AMP，并要求 control/target 双 repeat
  对既有 success hashes bit-exact；禁止把 cache observation 写成 OOM 或跳过 parity 直接 full materialization。r038 evidence=
  `summary e9db6152...8f46 /audit 16,025 bytes, a8cbdb5b...4047 /freeze
  configs/worldsim_v51/stage_f_f0g_target_tensor_allocator_instrumentation_freeze_v1.yaml`。r039 在每个 frozen matmul 前调用
  `torch.cuda.empty_cache()`，A–B–A–B 四个 fresh process 全部成功；8 次 intervention 均有 before/after allocator 证据，
  control/target 双 repeat 与 r034、r036/r038 success hashes 全部 exact，且资源门 PASS。因此 empty-cache 是当前合法的
  execution recovery candidate，并推翻“释放 cache 必然改变 identity 输出”的担忧；但三视图 parity 不能外推 1087
  15-view/全 45-view，`V51-F62` 仍 active。下一步只允许单场 1087 15-view recovery，禁止直接把 r039 写成 full
  materialization ready。r039 evidence=`summary d720af4e...9505 /audit 8,625 bytes, fda57ee4...88ab /freeze
  configs/worldsim_v51/stage_f_f0h_pre_matmul_empty_cache_parity_freeze_v1.yaml`。r040 把同一 intervention 扩到 exact
  scene-1087 15-view：单 fresh process 完成 `15 uint8 900×1600 masks +pred.json`，6/6 observed matmul 均有
  empty-cache before/after 证据，resource gate 与独立审计全部 PASS；未读取 mask 内容质量。因此“该 recovery 只对三视图
  probe 有效、扩到 1087 15-view 必然失败”已被推翻，但它仍不能外推 fresh 三场 45-view，`V51-F62` 保持 active。
  下一步只允许预注册新目录、按 `0471→1087→0379` 串行的 45-view recovery；禁止续写 r035、复用 r035 partial、先读
  r040 quality，或直接放行 identity training。r040 evidence=`summary 312a0277...a65 /audit 9,254 bytes,
  1393c664...67c /freeze configs/worldsim_v51/stage_f_f0i_scene1087_15_view_empty_cache_recovery_freeze_v1.yaml`。
  r041 再从 exact r026 manifest 新建三组 scene-local 输入，按 `0471→1087→0379` 三个 fresh process 串行执行；三场
  全部成功，45/45 schema masks、3/3 pred、18/18 pre-matmul empty-cache evidence、output record chain 与资源门均由
  独立审计重放。因此 r035 暴露的 full-materialization execution failure 已在 frozen empty-cache intervention 下解除，
  `V51-F62` 改为 resolved recovery；但 trace timing/allocator-cache/CUBLAS workspace 中哪一个是唯一根因仍未证明，不能
  写成 OOM root cause。该结论也完全不包含 mask quality、actor identity alignment 或 training readiness；下一步必须新预
  注册质量/对齐门，禁止把“45 个文件存在”写成算法有效。r041 evidence=`summary f3ee3ad1...c183 /materialization
  32b5d8d3...1b7f /audit 18,462 bytes, acd5a91b...31d2 /freeze configs/worldsim_v51/stage_f_f0j_fresh_45_view_
  empty_cache_materialization_freeze_v1.yaml`。
- `V51-F63`（`algorithm/instance-quality-alignment`, `rejected`）：faithful Gaussian Grouping 的 45-view materialization 在
  F0l 首次读取 frozen train-only weak support 后，只通过 scene-1087，scene-0471/0379 均失败。0471 的 foreground
  coverage/one-to-one identity recall/persistent-track fraction=`0.122784/0.080747/0`，0379=`0.238278/0.202933/0`，分别远低于
  读像素前冻结的 `0.70/0.35/0.50`；两场 assignment efficiency=`0.937/1.0`，说明问题不是主要由 short-ID collision
  造成，而是 actor support 大量未覆盖且同一 3D track 的 assigned ID 没有跨两个 eligible views 持续。1087 的
  `0.859091/0.505009/0.5` 全门通过不能覆盖 all-three-scene contract。该负结论推翻“只要 full materialization 稳定，
  faithful DEVA short IDs 就足以作为当前 driving identity training 输入”；它不是 CUDA/资源 blocked，也不允许训练后再救。
  限制：DriveStudio dynamic union 只作 foreground weak support，3D projected boxes 只作 track attribution，不是真值 instance
  segmentation；因此结论严格限于当前三场、frame→camera view order 与 adapter，不能外推为 Gaussian Grouping 普遍无效。
  F1/F2/identity training 关闭，下一步按冻结路线进入 Trace3D source/method/immutable-base adapter preflight。证据：r043
  summary=`f13c8094...da8a`、report=`b1e4bb40...95ed`、audit=`4,210 bytes /f478fbd9...4320`、freeze=
  `configs/worldsim_v51/stage_f_f0l_train_only_quality_identity_alignment_freeze_v1.yaml`。
- `V51-F64`（`engineering/tool-availability`, `resolved`）：Trace3D G0 r044 已成功原子发布 official PDF 与 exact
  repo commit/tree，随后仅在读取 PDF 页数时因系统没有 `pdfinfo` executable 抛出 `FileNotFoundError`。异常发生在任何
  repository semantic report、source execution、submodule initialization、model download 或 image/mask/quality read 前；run
  只留下 running status 与 start event，必须保留且不能事后补成 blocked/done。该失败推翻“AutoDL 默认具备 Poppler CLI”
  的工程假设，不涉及 Trace3D 方法可行性。合法恢复只在新 run exact 复核现有 paper=`2,390,825 bytes /d50eda07...47e4`
  与 repo=`7465ad94...c442/tree 22d30d19...a05d/clean`，用标准库 PDF `/Type /Page` marker 计数替代 external tool；不得
  删除或重下已发布资产，也不得借恢复 init submodules/执行源码/读质量。closeout=
  `configs/worldsim_v51/stage_g_g0_trace3d_source_method_preflight_r044_closeout_v1.yaml`。r045 以原资产 exact reuse、标准库
  page-marker count=`11` 完成恢复；独立 audit `053cf574...d3b` PASS，故该工程失败关闭。r044 历史终态不倒写；本次修复不构成
  Trace3D 方法成功或训练授权。
- `V51-F65`（`engineering/algorithm-determinism`, `resolved_method_rejection`）：Trace3D exact unpatched CUDA extension 在 r046 的
  preregistered synthetic class-response gates 上 PASS，但相同 config/input/extension 的独立 fresh-process audit 将 foreground
  alpha weight 从 `0.0267562941` 重放为 `0.0056084292`（absolute diff=`0.0211478649`）；hard class vector 均为 `[0,1]`。
  official `id_trace.cu` 在多个 pixel threads 面向同一 per-Gaussian/class global weight 时使用普通 `+=` 而不是原子累加，
  这是与漂移相容的 source-level hazard，但一次跨进程差异尚不足以把根因写死。不得因 hard argmax 一致就进入真实 U2/B3
  adapter，也不得事后给 r046 增加失败门或直接 patch upstream。合法下一步是在全新预注册 run 中以多个 fresh processes 同时
  冻结 hard/alpha exact determinism；FAIL 时 faithful Trace3D operator rejected 并按路线转 BKI/graph-free，PASS 才允许预注册
  real tensor/camera adapter。r046 capability PASS 与本风险并存，且都不构成质量证据。
  r047 按上述规则执行 8 个 fresh processes、每进程 hard 两次与 alpha 两次；16 个 hard vectors 均为 `[0,1]`，但
  alpha exact vectors 为 `0.0056084292/0.0267562941` 两种，故唯一数 `2 > 1` 并被独立 audit `98c72ba7...d31`
  重算确认。该结果足以拒绝当前 exact unpatched faithful operator，但仍不把普通 `+=` 写成已证明的唯一根因，也不外推
  所有 Trace3D 实现。按预注册 failover 不 patch、不进入 real adapter，路线转 `WS-V51-M1-H-GRAPHFREE-01`。
- `V51-F66`（`algorithm/governance`, `superseded`）：Stage H 的 faithful BKI/graph-free fallback 在 V5.1 内没有启动，
  task status 保持 `pending`、execution=`false`，并由 V5.2 scope 取代；因此它既不是 done，也不是 empirical rejected。
  收口依据是累计而非新增质量读数：r018 的 progressive propagation 只有 ΔBoundary-F1=`+0.0002196`，同时
  ΔIoU=`-0.0714543`、ΔFN=`+0.1694766`；r022 的 simple voxel node 虽提高 observation density，相对 U2/B3 的
  BF1/IoU/FN 仍为 `-0.0002566/-0.0925468/+0.1899473`；r043 的 0471/0379 identity recall 仅
  `0.080747/0.202933` 且 persistence=`0/0`；r047 的 faithful Trace3D alpha 又不能跨 fresh process exact 重放。
  这些事实共同推翻“在不改变 evidence source 的情况下继续替换传播器仍有较高边际收益”的 V5.1 资源分配假设；它们
  支持当前瓶颈为 effective observation structural missing，但不证明 BKI 或所有空间 kernel 普遍无效。禁止在 V5.1
  继续做 BKI source preflight、kernel/threshold 调参，或把 `superseded` 改写成 BKI reject。合法复开只能在 V5.2 先引入
  独立新观测源，并在首次方法质量读取前冻结 coverage、identity persistence、fresh-process reproducibility 与跨场分母；
  证据=`configs/worldsim_v51/m1_closeout_v1.yaml`、`docs/archive/2026-08/worldsim-v51-m1-closeout/README.md`，
  authoring base=`fc07b99`，failure delta=`V51-F66`。

<a id="detail-v5"></a>

## V5 M3 constraint projection 新增防重复结论（2026-08-14）

- `V5-F52`：V4 M3 canonical r238/r335 的 baseline 是 `FRAME_INDEPENDENT`，不是 V5 要求的 `T2_V4_FROZEN_SE3_BSPLINE`。V4 的 `30.41%/34.39%` warp 改善不能直接成为 V5 T2 comparator 证据；V5 必须在同一 fresh clip 上重跑 T2–T5。
- `V5-F53`：M2 rejected 后，M3 REMOVE 不能隐式复用 geometry-safe repair。REMOVE 只保留 exact bypass、semantic reintroduction 与 rollback checks，不进入 trajectory physics denominator；M3 正结果永远不得倒写 M2 成功。
- `V5-F54`：r002 是 config identity blocked，不是 clip inventory 质量失败。缺少 `protocol_audit.conclusion` 在 annotation streaming 前触发 KeyError；无图片/LiDAR blob/quality/GPU 读数。terminal 保留，只以新 r003 修复。
- `V5-F55`：低速位置抖动不能定义 velocity heading，倒车也不是 heading inconsistency。r004 的 T2 38 项违例全部来自旧 heading metric；禁止用 speed>`0.1m/s` 且 forward-only 的结果宣称 projection 有大幅物理改善。当前修正规则是 speed>`1m/s` 且 forward/reverse mismatch 取小者。
- `V5-F56`：POCS 更新量固定点不等于物理可行。r004 T5 的逐帧 heading correction 产生 `20` yaw-rate + `14` heading violations，却曾报告 converged；现在只有剩余 total violations=`0` 才允许 convergence。禁止用旧 converged flag 支持方法。
- `V5-F57`：r005 exact replay 后 T2=`15/16 safe`，仅 `1/16` request 有 `2` 项违例；T5 降到 `1` 项，但预注册最小 evaluable 是 `8`。当前结论必须是 insufficient signal，不能因相对 reduction=`50%` 而解锁 renderer、选择 arm 或进入 validation。
- `V5-F58`：不得通过降低 heading speed floor、取消 reverse、降低物理 caps、删除 T2-safe requests 或改 minimum-evaluable gate 来“修复”r005。复开只能注册新的 desired-motion hypothesis 与新 run，明确保留 result-aware development 身份；collision/render/validation 仍需独立门。
- `V5-F59`：r006 已把 V5 constraint-projection M3 正式收口为 `rejected`。禁止事后扩大 lane shift/acceleration stress template 来人为制造 T2 violations；V4 M3 时序正结果继续成立，但只能写作历史 baseline，不能倒写 V5 constraint projection 成功。未来复开必须是独立新路线与新冻结协议。

## V5 M2 cross-view scaffold 与拒绝收口新增防重复结论（2026-08-14）

- `V5-F46`：repair asset provenance 必须来自已有不可变枚举。r012 使用新字符串 `cross_view_background_depth_scaffold`，在首个 asset、GPU 与方法质量读取前被拒绝；该 terminal 是工程 blocked，不是 G4 质量结果。修复只改为既有 `native_scene_donor` 并新建 r013，不覆盖 r012。
- `V5-F47`：G4 的 Gaussianization 后改善 `17/22` 不能替代 raw 相对门。r013 raw 仅 `12/22`，低于冻结的 `14/22`，且 raw/post absolute-safe 均为 `0/22`；G4 必须保持 rejected。其 direct projection mean/median 仅约 `4.73%/0.78%`，不得靠 Gaussianization 或 fallback 隐藏覆盖不足。
- `V5-F48`：G5 的相对支持不能覆盖绝对几何安全失败。r014 raw/post 改善=`15/22`、`19/22`，mean delta=`-3.270320/-4.023966m`，但 raw/post absolute-safe 仅 `1/22`、`0/22`；正式写法只能是“model proxy 上相对改善，未形成 geometry-safe candidate”，不得写成 M2 成功或跨视角恢复真实背景。
- `V5-F49`：多相机投影覆盖不是独立真值。r014 any/direct/extrapolation/fallback mean=`60.40%/15.57%/47.99%/36.44%`，LiDAR projected mean≈`0.8%`；大量信息来自 bounded extrapolation 与 G0 fallback。禁止把 source 数量、投影覆盖率或 proxy MAE 当作 same-view hidden-background GT confidence。
- `V5-F50`：G5 绝对门失败触发结果前 hard stop。禁止事后搜索 absolute threshold、camera/time source grid、fusion、disagreement、extrapolation radius、Gaussian stride/opacity，亦不得自动解锁神经 surface；任何复开必须是新科研假设、新 task、新冻结协议与独立 evidence source。
- `V5-F51`：r015 已把 `WS-V5-M2-GEOMETRY-FIRST-REPAIR-01` 正式收口为 `rejected`，method/router/validation 均未解锁。后续 M3 是独立任务，任何 M3 正结果不得倒写 M2 成功；`WS-V5-M2-GEOMETRY-FEASIBLE-ROUTER-01` 保持 locked，不能在没有 geometry-safe candidate 时单独运行。

## V5 M2 geometry-first 新增防重复结论（2026-08-14）

- `V5-F34`：同一 view 的 actor-union mask 不是合法的一次修复 request。r002/r003 把多个 actor 合成一个 hole，最大 union 达 `152,410` pixels，导致大洞支配均值；r004 已恢复 `one_actor_one_view_one_hole`，得到 `23=22 accepted+1 rejected` 且 union pixel-exact replay。后续方法选择不得复用 union-mask 分母。
- `V5-F35`：base renderer `background_depth` 是维护一致性 proxy，不是 same-view hidden-background GT。r005 的 reference confidence mean/median 只有 `0.0585/0.0582`，范围 `0–0.1557`；任何 raw/post MAE 必须与 `model_proxy_not_ground_truth` 同表，不得写成真实道路深度恢复。
- `V5-F36`：G0 在逐 actor r005 上 raw absolute fail=`22/22`，raw MAE mean/median=`8.5872/8.7151m`；同时 Gaussianization primary=`16/22`。不得因为 G0 是最简单 arm 或相对复杂 surface 较稳，就把它写成 geometry-safe candidate。
- `V5-F37`：G1 piecewise plane 只有 `5/22` 请求改善 `>=0.5m`，candidate−G0 mean/median=`+3.658565/+2.300282m`。它在当前 model proxy 上正式 rejected；禁止根据局部 side 拟合直觉继续调 piece 分区后复活同一 arm。
- `V5-F38`：G2 MLS 的收益由少数洞驱动。它在一个 `118,022`-pixel actor 上改善 `11.194128m`，但仅 `8/22` 请求达到改善门，mean/median delta=`+3.005506/+1.620793m`。不得引用最大改善或自适应带宽完成度掩盖广泛稳定性失败。
- `V5-F39`：G3 quadratic 的 median delta=`-1.489037m` 不能覆盖 mean=`+0.103693m` 与 improvement count=`11/22<14/22`。r009 已按冻结 gate rejected；不得事后将判据改成 median-only，也不得沿用 request-unit 错误的 r003 作为相反证据。
- `V5-F40`：blocked terminal 必须与方法失败分离。r001 是 unavailable-view denominator 合同错误；r007 是 artifact serializer 局部变量遮蔽导致 `KeyError: 0`。两者没有可用质量 summary，不能进入 arm 均值，也不能覆盖目录或改 terminal；修复只允许新 run ID。
- `V5-F41`：G1/G2/G3 全部 rejected 且 G0 raw `22/22` fail，当前不存在 safe candidate。按照 V5 causal order，feasibility-first router、validation 和神经 surface 都不得解锁；下一步只能先诊断/修复 Gaussianization representation 与 alpha compositing，之后重新经过独立 development gate。
- `V5-F42`：formal run 的目标目录必须由 runner 原子创建，不能为了 stdout 重定向提前 `mkdir`。r010 在任何模型加载/GPU/质量读取前被 overwrite guard 拒绝；其 blocked terminal、events 和 run.log 保留。修复只改变 launcher，把日志放到 run 外部并用新 r011；不得删除 r010 或把它计入方法分母。
- `V5-F43`：提高 Gaussian asset opacity 不能修复当前 representation gap。r011 的 OPAQUE−BASE 在 `0/22` 请求改善 `>=0.1m`，mean/median post-MAE delta=`+0.059686/+0.065773m`；在 dense 条件下继续提高 opacity 也退化 `+0.035533m`。禁止继续提高 opacity、改 alpha 阈值或把 background mixing 写成已支持机制。
- `V5-F44`：stride `2→1` 的 DENSE arm 在 `20/22` 请求改善，mean/median delta=`-0.424179/-0.480927m`，但这是 frozen scene0471 model proxy 上的机制取证，不是 geometry-safe method selection。G0 raw 仍 `22/22` absolute fail，validation/KITTI/独立 GT 均未读取；不得直接把 stride-1 送入 validation、改 router，或把 post-render 改善写成真实道路恢复。
- `V5-F45`：factorial arm 通过数必须解释为因素对比，不能按臂名投票。r011 中 DENSE 与 DENSE_OPAQUE 都过门，正式 summary 因此保守写作 `multiple_gaussianization_factors_have_broad_mechanism_support`；但 OPAQUE 在 sparse/dense 两个条件都退化，描述性 density/opacity main effect=`-0.436256/+0.047609m`。后续只允许冻结 density representation repair 并重新过独立 gate，不能把组合臂通过误写成 opacity 或 interaction 获支持。

## V5 M1 structured unary 新增防重复结论（2026-08-14）

- `V5-F20`：renderer 的逐 pixel intersection 不是独立多视角证据；同一 Gaussian 在同一 view 覆盖更多像素时，若逐行更新 Beta，会把屏幕面积伪装成 view count。V5 必须先按 `Gaussian × view` 聚合 contribution，再以 `1-exp(-mass)` 形成饱和 visibility，B0/B1 每 Gaussian 每 view 最多一票；不得以原始 intersection 行数扩大 evidence denominator。
- `V5-F21`：只改 arm 名称不能形成 Bayesian ablation。若 B0/B1/B3 共享同一 soft signal 与同一 reliability，它们在代数上会退化为同一方法。结果读取前已冻结为 `B0=hard unweighted`、`B1=hard reliability-weighted`、`B3=soft SAM probability × reliability fractional count`；B2 继续延后，禁止在读到 scene0471 指标后改定义。
- `V5-F22`：ownership posterior 不是新的 Gaussian opacity。若直接把 posterior 当 opacity，会给原 base 中近透明的 Gaussian 注入虚假 semantic mass。所有 2D ownership evaluation 必须渲染 `immutable base opacity × ownership probability`，并在运行前后复核 base checkpoint SHA；不得用 posterior-only rasterization 形成虚假 IoU/Boundary F1 改善。
- `V5-F23`：scene0471 的 annotation prompt 集与 checkpoint RigidNodes 表示没有天然一一对应。冻结规则选出 `17` 个轨迹大于 1 m 的非自行车 vehicle（`15 car + 2 construction`），而 formal checkpoint 只有 `15` 个 RigidNodes instances；差异可能来自无足够 LiDAR 的标注 actor。2D SAM union 与 3D base-model proxy 必须分别报告，不得把表示缺口静默算成 unary FN、删除 prompt 或补造 Gaussian。
- `V5-F24`：直接调用 `process_camera/collect_gaussians` 会绕过 `SceneGraphTrainer.forward()` 中的 timeline 设置；若不显式按 `normed_time` 更新 `trainer.cur_frame` 与各 Gaussian model 的 `set_cur_frame`，动态 actor 会被错误地固定在旧帧。V5 sidecar runner 必须复现该状态迁移并做 nearest-timestamp 回归测试；仅图像 frame ID 正确不足以证明动态 Gaussian pose 正确。
- `V5-F25`：scene0471 r037 的 B1/B3 虽同时改善 IoU、Boundary F1、Brier、ECE、NLL 与 FP semantic mass，但 2D FN semantic mass 相对 B0 分别增加 `+0.0915315/+0.0954773`，明显超过计划 validation gate 的 `+0.01` 容忍量。不得只摘录正指标把 r037 写成 M1 成功，也不得以 aggregate calibration 改善掩盖漏检代价；后续 graph 诊断必须逐臂保留 FN、per-view denominator 与 abstain。
- `V5-F26`：r037 是单个 development scene、固定 `0.5` 阈值、`8 accepted + 7 abstained` evaluation views 上的 SAM-proxy 机制诊断；它能推翻“reliability-aware unary 完全无方向信号”，但不能证明 topology graph 必要、不能选择 B1/B3、不能代表 8-scene validation。graph 只能在单独预注册协议后启动，禁止根据 r037 结果补调 unary 参数、读取 validation 或直接扩展 Transformer/semantic split。
- `V5-F27`：r038 的 G3 在 scene0471 2D SAM-proxy 上只带来 `+0.008585/+0.006245` Boundary F1（B1/B3），虽然方向一致且 FN 增量小于 `0.002`，但 Gaussian membership proxy 的 IoU 与 Boundary F1 同时退化。不得只摘录 2D 正指标把 graph 写成已通过，也不得只看 proxy 负指标否定全部图机制；两套口径都必须保留，并在 result-blind development replication 后再决定 formal arm。
- `V5-F28`：r038 的 `cross_proxy_affinity_ratio` 使用 Background/RigidNodes membership 仅做事后 leakage 审计；graph candidate/affinity 明确不消费该字段。G1→G3 从 `0.0083646` 降到 `0.0040198` 证明物理 affinity 更少跨越 base proxy，不等于真实语义边界 GT 或 graph 必要性。禁止把 proxy 反馈进建图、据此调 k/扩散率，或直接解锁 semantic split/validation。
- `V5-F29`：长时 sidecar 不能依赖 SSH stdout 生命周期。scene1087 unary r041 已完成主要计算，却在关闭的输出管道上写日志触发 `BrokenPipeError` 并合法标记 `blocked`；不得把它写成方法失败、覆盖目录或复用编号。正式长任务必须在启动时脱离 SSH 并把 stdout/stderr 重定向到独立日志；同一冻结配置以新编号 r042 完成，r041 继续作为基础设施失败证据。
- `V5-F30`：scene0471 的 `8 accepted + 7 abstain` 不能硬编码成 graph 的全场景分母。r044 在读取 scene1087 的绑定 unary 后被该常量 fail-closed；修复 `d55a067` 改为 summary/diagnostics 双重验证 accepted、abstain、总分母与 B1/B3 `(frame,camera)` 键，并要求 accepted>0。r044 是通用化合同失败，不是数据或 graph 质量失败；修复后只能使用新 run r045。
- `V5-F31`：三场景 frozen SAM 的可用视图为 scene0471/1087/0379=`18/2/6`（各 30），对应 unary 可评估分母=`8+7 / 1+14 / 3+12`。scene1087 的负方向只来自 1 个可评估视图，不能扩大为总体失败；但 result-blind replication 必须保留该稀疏场景和全部 abstain，禁止删场景、补 prompt、补 mask 或只报告可评估视图较多的场景。
- `V5-F32`：G3 三场景复制门正式失败。六个 `scene × unary` 单元只有 `3/6` 个 Boundary F1 为正（门槛 `>=4/6`）；虽然 mean ΔBoundary-F1=`+0.0016107723`、mean ΔFN-mass=`+0.0025676789` 单项通过，但 scene1087 的 G1 cross-proxy affinity 已为 `0`，G3=`1.2800523e-29`，逐场严格下降也失败。不得用正均值覆盖稳定性门、选择 G3、读取 validation 或直接堆 Transformer。semantic split 仍是条件任务，必须先用独立 boundary-residual forensic 证明 boundary ambiguity 是主要残差；当前不自动解锁。
- `V5-F33`：boundary error enrichment 高不等于 boundary 是主要残差。r001 六个单元的 enrichment=`3.83×–280.98×`，但 boundary-primary=`0/6`，mean boundary classification/semantic-error share 只有 `0.402095/0.248353`。尤其 scene0379 虽约 68% threshold error 位于极小边界带，boundary semantic-error mass 仍只有 26%–36%；不得只引用 enrichment 解锁 split。M1B 条件未成立，semantic split/Transformer/validation 继续禁止，M1 structured ownership 收口为 rejected。

## V5 KITTI archive / adapter 新增防重复结论（2026-08-14）

- `V5-F01`：官方 KITTI Tracking calibration 不是统一的 `key: values` 语法。实际 `P0`–`P3` 行带冒号，`R_rect`、`Tr_velo_cam`、`Tr_imu_velo` 行不带冒号；V4 `_read_numeric_table()` 会静默忽略后三类行，真实 adapter 将缺失 rectification/extrinsic。V5 必须同时解析 colon/whitespace 两种格式，并对矩阵 shape、finite、handedness 和投影做 2-sequence smoke；不得把 zip layout ready 写成 calibration gate 已通过。
- `V5-F02`：官方 tracking OXTS 每行是 `30` 个导航/IMU 字段，不是 12-value `3×4` world pose。V4 `_load_pose_matrices()` 会截取前 12 个值并错误解释为位姿，行数与 sensor frame 相等也不能证明 object/world/camera chain 正确。V5 必须按官方语义从 latitude/longitude/altitude/roll/pitch/yaw 构造 pose，并结合 `Tr_imu_velo` 验证坐标链；禁止直接复用 V4 OXTS path 当 pose matrix。
- `V5-F03`：central directory 可读、成员集合对齐和全 archive SHA-256 只证明压缩包可冻结、可进入 staging，不等于真实 adapter 已完成。合法晋级仍需要独立 `.partial` 解压、post-extract member/frame audit、2-sequence 坐标/pose/track-ID smoke 和新 manifest；不得从 archive metadata 直接写 `WS-V5-D1-KITTI-ADAPTER-01=done`。
- `V5-F04`：KITTI Tracking 官方 testing split 没有 `label_02`。testing sequences 可用于无标签 adapter/engineering smoke，但不能进入需要 track/box GT 的 cross-domain 质量主表。V5 10-sequence formal 必须从 21 个 labeled training sequences 中在结果前冻结；不得用 testing split 扩大带 GT denominator。
- `V5-F05`：实际 archive 的 `training/0001` 不是三传感器全帧严格对齐：`image_02/image_03/velodyne=447/447/443`，LiDAR 缺 `000177`–`000180`。这不是 ZIP 损坏或全 KITTI 缺失，但会使“每帧都有 stereo+LiDAR”的 adapter 合同失败。V5 必须在结果前冻结 common-frame/abstain 与 coverage denominator，逐序列记录被排除帧；不得静默 `set` 取交集、补造 LiDAR、删掉 0001 或写成 447/447 完整 multimodal coverage。
- `V5-F06`：V4 M2 geometry risk 使用 `clip(hole_geometry_mae_m / 0.5, 0, 1)`；r219–r221 的 `214` 个 candidates 中 `192` 个饱和为 1，且所有 MAE `>=0.5 m` 的 `192/192` candidates 都相同。`57/130` 个有候选 request 存在“未归一化 rendered MAE 不同、geometry risk 全相同”的碰撞。V5 不得只改 geometry 权重或阈值后宣称解决；任何 mapping 必须保持 tail rank，并报告 saturation ratio、unique risk、rank correlation 与 bad-tail distinguishability。
- `V5-F07`：M2 `+3.3908096237 m` 是保留 abstain 的 policy-level scene-balanced delta，不等于 83 个 accepted repair candidate 相对 TELEA 退化。accepted-only 同请求 router/TELEA=`1.62295/2.01453 m`，而 47 个 risk-abstain 的 atomic no-op/TELEA request mean=`16.58283/2.60817 m`。V4 caveat 不得删除，但后续必须同时报告 accepted geometry、abstain geometry、coverage 和 full-denominator valid yield；不得把两种口径互相替代。
- `V5-F08`：V4 M1 canonical state 已把 observation 压成正/负 count 与乘积 weight，未持久化 per-view observation、投影 boundary distance、Gaussian center/covariance 或 neighborhood/topology disagreement。仅凭 r200 state 不能证明 SAM 错、graph 必然有效或 topology 是唯一根因。M1-D0 必须先生成带 provenance 的 per-Gaussian/per-view diagnostic；缺字段时保持 `running/blocked_evidence_missing`，不能从 aggregate Boundary F1 直接跳到完整 graph 实现。
- `V5-F09`：`WS-V5-M1-D0-BAYES-FORENSICS-01=done` 只表示历史分母已机器重算、缺失字段采集合约已冻结。canonical conclusion=`blocked_evidence_missing_contract_frozen`；不得把 task done 改写成 evidence complete、graph 已验证、M1 rejection 被推翻或可直接训练 full structured ownership。
- `V5-F10`：M2 的 retrospective geometry oracle 只按现有 rendered `hole_geometry_mae_m` 相对排序；其 reference 是 immutable base `Background_depth`，不是 same-view hidden-background GT。即使 `62/83` accepted 与该 oracle 一致，也不能证明 candidate 物理正确。必须先补 `reference_source/confidence` 与 raw→pre-Gaussian→post-render 三段误差，随后才允许在 fresh development 拟合 non-saturating mapping。
- `V5-F11`：P0 freeze-only commit=`dfe7526c7a83ca12d7fa9f6c5a11a29ea7b27b19` 只冻结 scope、historical bindings、missing-evidence schema 与审计器。它不包含 fresh scene selection、模型实现或质量结果；任何后续工作必须通过 P0 formal audit，并继续保持 fresh/test/KITTI quality 未读与 parameter search=false。
- `V5-F12`：fresh 8/8/20 的冻结只使用官方 split、scene context、actor annotation/LiDAR-count metadata proxy 与 sensor-keyframe completeness；没有展开图像/LiDAR blob，也没有读取 reconstruction/edit/M1/M2/M3 quality。20 个 test scene 的身份出现在 freeze manifest 不等于 test quality 已读；`V5_TEST_FREEZE.json` 与 exact-once ledger 形成前，禁止加载其内容或指标。
- `V5-F13`：fresh development cohort 冻结后，8 个 scene 的 processed 为 `0/8`；三前向相机+LiDAR keyframe 的 `0/1280` 只是早期粗审计，不能作为 DriveStudio 10Hz preprocess 的完整分母。核对上游后，真实合同是六相机+`LIDAR_TOP` 的完整 keyframe/sweep 时间链，metadata-only 精确分母为 `14,220` files、当前 `0` present。这是 selective extraction/preprocess 工程前置，不是 M1 质量失败；不得退回 V4 scenes、替换 frozen cohort、提前读 validation，或为省事解压全部约 294 GB blobs。必须一次扫描 metadata、按 member→archive 选择性抽取，并保留逐 scene/sensor/file denominator 与内容哈希。
- `V5-F14`：V4 semantic mask NPZ 实际保留 SAM2 `logits/raw_binary/binary`，因此 V5 不得把最终 binary 当作唯一 confidence，也不得为补 confidence 重新运行或更换 SAM。V5 使用 frozen logit 的 sigmoid 作为 observation probability；quality gate rejected mask 仍保留 raw logit 供诊断，但必须把 positive/negative/reliability 全部置零并显式记录 availability，禁止把拒绝样本误当作背景负证据。
- `V5-F15`：Gaussian 最小 covariance 主轴只能作为 renderer-native surface-normal proxy，不是 LiDAR/mesh ground-truth normal；其符号还具有本征向量二义性。V5 必须用 reference camera 定向、验证 covariance 正定与 available normal 单位范数，并把 `normal_is_ground_truth=false` 固化进 config。若 graph 改善，不能据此宣称已恢复真实表面法线。
- `V5-F16`：DriveStudio 原生 nuScenes preprocess 完成不等于 StreetGS 训练输入已闭合；它生成 images/calibration/LiDAR/object/dynamic masks，但不生成 StreetGS loader 必需的三训练相机 `sky_masks`。V5 首个 profile r003 在任何训练迭代前因 `sky_masks/000_0.png` 缺失合法 `blocked`，summary SHA=`a2802430984ab369143be609088df514e3ed0943563b23ee0a5b3bee02e214f7`。这不是 reconstruction 质量失败，不得覆盖 r003、伪造空 mask 或把 preprocess 8/8 改写成失败；必须先用已冻结本地 SegFormer revision、offline/atomic 协议派生每 scene `frames×3` masks，绑定独立 manifest/SHA，再以新 run ID 重跑 profile。

- `V5-F19`：Python 包内单测通过不等于脚本可从仓库根直接启动。KITTI audit r002 attempt 在读取任何 payload 前因 `ModuleNotFoundError: motion_proj` 失败；import 发生在 runner main 前，因此没有生成 run 目录。不得伪造 r002 terminal、复用该 ID 或把它写成数据/坐标质量失败。修复必须在 package import 前显式加入 project root，并增加 `script.py --help` 直接入口回归测试；提交 `43fe090...` 后以 r003 新 ID 完成真实 smoke。
- `V5-F18`：单场或部分场景的 100-step 成功不能解锁全量 formal，也不能解释成 reconstruction 质量成功。必须保留 8-scene denominator，每场验证 step-100 checkpoint、finite means、summary/status/fingerprint/run-manifest、clean source 与 checkpoint bytes/SHA；formal runner 必须再次读取已提交的 cohort binding。r019–r026 的 `8/8 done` 只证明训练链路与资源门可用，尚未读取 development quality，也不允许跳过 30k base、改用 profile checkpoint 做 structured ownership 结论。
- `V5-F17`：sky mask 文件存在不等于训练输入已合法绑定。V5 必须同时验证 8 个独立 run 的 summary、run manifest、sky-mask manifest、冻结 SegFormer revision、`frames×3` denominator 与全部 PNG bytes/SHA，并把这些 identity 通过新 overlay 绑定到不可变 reconstruction base 配置；不得回写被 r003 引用的 base 配置、只数文件名后开训，或把 segmentation inference 误写成 method inference。r011–r018 已按该协议闭合 `4704/4704`，只解锁 `profile100`，不直接解锁 30k formal 或质量结论。

<a id="detail-v4"></a>

## V4 M3 / 18-scene exact-once 防重复结论（2026-08-13）

- `V4-F44`：r258 因 18 场 sky masks 尚未齐全而 fail-closed，r277 因上游假定 instance timeline 稠密而在 scene-0919 暴露稀疏时间轴合同错误；两者都是资产/兼容性失败，不是模型质量失败。只允许以提交 `d5a4794e` 的稀疏 timeline 兼容修复及 r278 100-step smoke 解锁正式训练，不得覆盖失败 run 或提前读 test quality。
- `V4-F45`：M3 validation r238 的完整 denominator 是 `3 evaluable + 3 abstain = 6`。30.4106% warp L1 改善与 2.6470% temporal LPIPS 改善只来自可评场景；不得删除 abstain、写成 6/6 质量成功，或外推到长时序/非三前向相机。
- `V4-F46`：REMOVE 使用 exact bypass，零时序增益是冻结组合合同的结果；M3 通过依赖预注册的 across-operation temporal gate，不代表每个 operation 都严格改善。不得事后取消 bypass、改 operation 权重或只报告 LATERAL/INSERT。
- `V4-F47`：M2 晋级不消除 geometry 风险。hole geometry MAE 的 signed improvement 为 `-3.3908096237 m`（即误差退化 `+3.3908096237 m`）；18-scene 时序结论无论为 `confirmed`，都不得改写成 repair geometry dominance。
- `V4-F48`：18-scene test 使用 committed freeze 与 exact-once ledger；每场 attempt marker 在任何 test content/quality read 前以 exclusive create 写入，已消费 attempt 禁止重跑。canonical ledger=`/root/autodl-tmp/runs/worldsim_v4/WS-V4-M3-TEMPORAL-DELTA-01/20260813T222011Z__m3-test-exact-once-ledger-s0`，attempt/completion=`18/18`；聚合器只读 run evidence，未重读 test source content。
- `V4-F49`：test 的 abstain 必须留在 18-scene denominator。canonical `20260813T225624Z__m3-test-aggregate18-s0-r335` 为 `12 evaluable + 6 abstain`、conclusion=`confirmed`；不得把 evaluable-only gate 写成全 18 场成功，也不得因 `not_confirmed` 复用同一 test 调参或因 `confirmed` 扩大声明边界。

## V4 M1 rejection / M2 validation 新增防重复结论（2026-08-13）

- `V4-F39`：M1 的 development 正结果不能覆盖 scene-disjoint validation 负结果。validation 只有
  `3/6` scenes 可评，方向支持=`0/6`，Boundary F1/Brier/ECE 均反向；base/checkpoint exact 且没有 validation
  重搜。M1 必须保持 `rejected`，不得继续加 feature、transformer、改 threshold，或把 M2 成功倒写成 M1 成功。
- `V4-F40`：M2 validation 的完整 denominator 是 `6 scenes / 154 requests`。scene-1089/0862/1012 的
  `ABSTAIN_NO_ACTOR` 和 scene-0317 的 24 个 `ABSTAIN_NO_ROLE_MATCHED_ERASE_PACKAGE` 都必须保留；不得只用
  130 个具备 role asset 的请求或 3 个可评场景改写 coverage。canonical coverage 固定为 `83/154=0.5389610390`。
- `V4-F41`：validation 不允许重新选择 baseline、risk weights 或 threshold。matched baseline 必须沿用 development
  冻结的 `TELEA`，router 必须沿用 `uncertainty_forward/threshold=1.0`。即使 validation 上其他 arm 的 composite
  error 更低，也不得事后改 comparator 或路由 operating point。
- `V4-F42`：M2 通过的是预注册的合取门，不是所有 repair 轴支配。相对 TELEA，router 的 global PSNR/SSIM/LPIPS、
  hole PSNR、static LiDAR 和 selective-risk separation 通过，但 hole geometry MAE 从 `2.1435024986 m` 退化到
  `5.5343121223 m`。不得把 `hole_any_endpoint` 通过写成 geometry 改善、真值背景恢复或全面优于 Telea。
- `V4-F43`：selective-risk 成立只表示 frozen uncertainty 排序在当前 validation 请求上有误差分离：abstained
  counterfactual error 比 accepted 高 `0.1241311528`。它不证明 71 个 abstain 已被成功修复，也不允许把 abstain
  从 usable-yield 分母删除。M3 与 18-test 必须继续同时报告 coverage、abstain、blocked 和 worst-case。

## V4 M1 / validation 新增防重复结论（2026-08-12）

- `V4-F34`：同一 scene 的历史 V3.3 train mask 不能自动视为符合 V4 冻结的
  `sample_index mod 5` partition。scene-0230 的 development target 审计发现真实 train/evaluation
  frame overlap，因此必须 `ABSTAIN_LEGACY_SPLIT_LEAK`；不得放宽 split、删除该 scene denominator，
  或把旧 heldout 结果改名为 development。
- `V4-F35`：M1 六场景质量均值只允许在可评 scenes 上计算，但 coverage denominator 必须保持全部六场。
  r124 的 `2 evaluable + 4 abstain` 是协议事实，不得把 2/2 改写成 6/6 成功或静默删除 abstain。
- `V4-F36`：validation 只能复用 development 冻结的 evidence arm、calibrator、mask threshold 与 temporal
  retention；禁止在六个 validation scenes 上再次执行 arm search、calibration fit 或 threshold search。
- `V4-F37`：长时间 archive scan 不能依附会超时断开的 SSH stdout。r128 的 10 个 worker 在扫描约
  58 分钟后因外层 SSH 断管触发 `BrokenPipeError`；这不是数据缺失，也不得覆盖该 run。重试必须使用
  stdin=`/dev/null`、stdout/stderr 文件重定向、parent PID=1 的 detached 进程，并复用已提取的非空文件。
- `V4-F38`：Python 环境必须按 stage 显式区分。validation raw extraction 需要
  `/root/autodl-tmp/envs/motionproj/bin/python`（含 `ijson`）；StreetGS/V3.3 GPU runtime 使用
  `/root/autodl-tmp/envs/drivestudio/bin/python`。r127/r129 分别保留缺依赖与错误解释器路径证据，
  不通过临时安装或删除失败记录掩盖环境错误。


> **历史合并注记（2026-08-12）**：以下 V4 D0/B0 与更早内容在当日从旧账本合入；当前权威元数据、目录和写入合同
> 以上方 2026-08-17 统一入口为准。完整 `RF-01`–`RF-18` 原文见
> [`archive/2026-07/v7-feasibility/RESEARCH_FAILURES_RF01_RF18.md`](archive/2026-07/v7-feasibility/RESEARCH_FAILURES_RF01_RF18.md)。

本文件保留仍约束后续路线的历史结论，并把 H1-11D 的失败严格分为“观察到的事实、合理推断、尚未知、
复开条件”。归档不会使旧失败失效；任何新计划复用旧机制时仍须满足原 RF 的重开条件。

## V4 D0 防重复结论（2026-08-11）

- `V4-F08`：nuScenes metadata 的实际可用入口是 `/root/autodl-pub/nuScenes` 中的官方 archive，不是历史代码里的
  `/root/autodl-tmp/data/nuscenes` 或空的 DriveStudio data 目录。D0 只展开 `v1.0-trainval_meta.tgz`，不得为 cohort
  选择展开 549 GB sensor blobs、下载副本或把空目录写成数据集失败。
- `V4-F09`：nuScenes log filename 不能用未锚定的三组 `NN-NN-NN` 正则取小时。首版把
  `n015-2018-11-14-19-09-14+0800` 的月份 `11` 当成小时，测试将夜间误分为白天。修复必须解析完整
  `YYYY-MM-DD-HH-MM-SS±ZZZZ` 尾部；不得以放宽测试或手工改 scene 标签绕过。
- `V4-F10`：三个既有 processed scene 是 infrastructure anchors，不是结果前随机样本，也不能用其 V3/V3.3 质量选择
  test。D0 只把 0230/0242/0255 固定在 development，并以 metadata-only diversity 补足其余 scenes；后续不得把
  anchor smoke 写成 baseline 或方法质量结论。
- `V4-F11`：只冻结 scene name 列表不足以复验 cohort。D0 config 同时冻结 30 scene 的 actor/edit/clip/frame/sensor
  完整记录，formal builder 逐字段比对并锁 cohort SHA；任何 metadata 更新或构建逻辑变化都必须新建 run，不能静默
  沿用 `eda9f684...44578`。
- `V4-F12`：metadata donor support 是场景级 proxy，不是重建后 donor 的图像/几何质量。D0 可以用它做结果前分层，
  但 B0/M2 不得把 strong/medium/weak 标签直接当作 repair 成功、真实性或 quality ground truth。
- `V4-F13`：确定性 greedy selection 不能对无序 `set` 直接做普通浮点求和。r1/r2 恰好重建一致，r3 在另一
  `PYTHONHASHSEED` 下因同分 score 的末位舍入选择了不同 scene，freeze gate 以 `af36f51a...3447 !=
  eda9f684...44578` 拦截。r4 必须对 tag 排序并用 `math.fsum`，测试须跨多个 hash seed；不得固定环境变量掩盖算法
  非确定性，也不得把 r3 候选倒写成新 cohort。

## V4 D1 防重复结论（2026-08-11）

- `V4-F14`：requested dataset path 与 symlink-resolved physical path 必须同时记录。D1 r1 只写了
  `/autodl-pub/data/KITTI`，容易被误读为审计了不同根；r2 明确 `/root/autodl-pub/KITTI ->
  /autodl-pub/data/KITTI` 且两者均缺失。不得用 `Path.resolve()` 后的单一路径抹掉用户合同。
- `V4-F15`：synthetic 12-gate pass 只证明 adapter schema/坐标/投影/track-ID 检查可执行，不等于真实 KITTI adapter
  smoke。真实目录缺失时任务必须保持 `blocked`，不能将 unit fixture 写成两 sequence evidence 或 10-sequence
  cross-domain 结果。
- `V4-F16`：KITTI 原生合同只有 `image_02/image_03` 两路彩色相机。不得为了匹配 nuScenes 三相机伪造第三视角；
  tracking 缺失时才审计 raw，且 pose/tracklet/calibration 任一门失败都必须 `blocked_dataset_adapter`。

## V4 B0 防重复结论（2026-08-11）

- `V4-F17`：历史 summary/metrics 记录 checkpoint 曾经存在，不等于当前 checkpoint 可执行。B0 首次磁盘审计中，
  scene-0230/0242/0255 的历史 StreetGS 文件均已不在路径；只能保留 bytes/hash provenance，不能把历史
  `exists=true` 或旧质量数值登记为当前 executable。正式 matrix 必须在每次 run 重新检查真实文件。
- `V4-F18`：AD-GS historical aggregate 的 6-scene metrics 只覆盖旧 cohort，且其历史 source/env/checkpoint 路径已缺失。
  与 V4 development 重叠的 0230/0242/0255 也只能记 `historical_metrics_only`；2026-08-12 新恢复的 exact official
  source/env 只解除执行前置，不使历史 metric executable。不得把旧三场景数值拼接新三场景、用 aggregate mean 代替
  scene rows，或把环境 smoke 写成 V4 same-split checkpoint。
- `V4-F19`：baseline inventory 的 `blocked` terminal 是当前资产前置条件未齐，不是 B0 task 的永久 blocked 或方法失败。
  B0 继续 `running`，通过新 run 补齐资产；旧 inventory 不覆盖。只有 V3.3/StreetGS/AD-GS 各 6/6 且统一 evaluator
  完整后才可收口，不能因 M1 实现更有趣而跳过 matched baseline。
- `V4-F20`：DriveStudio preprocess 会把输出根再追加 `_10Hz`，并按零填充 scene index 写目录。r4 的上游命令成功
  不等于 runner 目录合同成功；必须验证真实输出 root、scene dir、`1,176 RGB / 196 LiDAR` 后才能登记 done，不能移动
  或重命名一个未审计路径来掩盖合同错误。
- `V4-F21`：远端网络不可达不能用镜像、floating revision 或未校验模型绕过。sky-model r10 因 `Errno 101` 保留
  blocked；本机只从官方固定 revision URL 获取三文件，传输前后均按 bytes/SHA256 exact 校验，远端恢复经临时目录
  原子发布且 generation 保持 offline。任何 staging 漂移都必须 fail-closed。
- `V4-F22`：preprocess 预建的空 `sky_masks/` 与已有推理产物不是同一状态。r12 因旧 runner 只看目录存在而 blocked；
  修复只允许非 symlink 的空目录，发布前再次 `rmdir`，已有 mask、非目录或 partial 一律拒绝，不能覆盖正式产物。
- `V4-F23`：StreetGS 100-step profile 只证明训练链、checkpoint schema 和单卡资源门可执行。r16 不能计入 6-scene
  30k formal coverage，也不能读取/登记质量或据此宣称 baseline 已完成；每场 formal 必须新建不可变 run。
- `V4-F24`：六场景 StreetGS 的 Gaussian 数、wall 与 peak GPU 差异很大；scene-0255/0048 sampled peak 达
  `24,092/24,000 MiB`，scene-0994 final RigidNodes 仅 `1,029`。不能用一个 scene 的 profile 外推所有资源，不能因
  actor 稀疏补点或删 scene，也不能把无 OOM 的近上限运行倒写成资源失败。主表保留每场分母与工程行。
- `V4-F25`：StreetGS 原生 `test_image_stride=10` 不是冻结的 `sample_index mod 5` 三分区。r17/r20/r22/r24/r26/r28
  即使完成 30k、checkpoint finite 且未主动读 test quality，余数 4 的 heldout 输入仍可能进入训练，因此只能保留为
  protocol-mismatch provenance；r29 的 `StreetGS=6` 被 corrected inventory r33 明确推翻。不得用“训练成功”替代
  matched-contract 合规，也不得覆盖旧 run 来修正历史。
- `V4-F26`：不读取 test quality 还不够，训练进程也必须在 I/O 层隔离 development/heldout。AD-GS adapter 正式训练
  只物化 `train` 的 354 张图；兼容补丁增加 `--disable_test_evaluation`，避免上游在 final iteration 自动将 test
  iterations 加入评测。审计/统一 evaluator 可以显式物化三分区，但训练 runner 不得复用该全量目录。
- `V4-F27`：source checkout、权重和 Python 包必须分别固定 commit/bytes/SHA，环境恢复成功不等于 baseline scene
  executable。r34 从冻结本地环境离线复制，编译 `simple_knn` 与 `diff_gaussian_rasterization` 并通过真实 CUDA
  forward/backward smoke；在 strict preprocess + checkpoint 完成前，AD-GS coverage 仍为 `0/6`。
- `V4-F28`：传输/命令包装失败与模型失败必须分开。CoTracker/plyfile 的首次远端校验受 shell quoting 影响，DPT 下载
  曾出现两个进程指向同一 partial，发布后的附带 `stat` 也曾因 quoting 失败；这些尝试均未被登记为 canonical。
  只有停止冲突进程、临时路径原子发布并对最终 bytes/SHA 做独立复验后才可使用，且不得把 wrapper failure 写成
  权重、CUDA 或算法失败。
- `V4-F29`：preprocess 失败后的目标目录不是可静默复用的 canonical。r35 因启动包装器预建 run 目录而被不可变门拒绝；
  r36 因环境构建留下的未跟踪目录被 source-audit 拒绝；r37 完成 adapter/depth/segment 后因可选诊断依赖
  `flow_vis` 缺失而在 flow 启动时 blocked。三者均未训练或读取 dev/heldout；r37 partial 移入
  `work/codex-backups/2026-08-12-adgs-r37-partial-scene0230`，不覆盖、不伪装 resume。正式 flow 通过显式
  no-visualization 合同移除诊断视频依赖，CUDA extension 后续从 run-local source copy 构建，避免再次污染 official checkout。
- `V4-F30`：修掉 preprocess 可视化依赖不等于训练 import graph 已解除同名依赖。r39 尚未进入 iteration，
  `loss_utils -> flow_utils` 就因全局 `flow_vis` import blocked；正确修复是只在 TensorBoard flow 图真正调用时 lazy import，
  并把 `utils/flow_utils.py` 纳入 exact compatibility patch。不得安装非必要诊断包来掩盖正式无评测训练合同。
- `V4-F31`：Python import 成功不证明 CUDA extension 包含当前 GPU kernel。r41 能加载 PyTorch3D 0.7.5，
  但 inherited `_C.so` 在 RTX3090 KNN 首次执行时报 `no kernel image`；必须从 clean frozen source 在 run-local
  目录以 `TORCH_CUDA_ARCH_LIST=8.6` 重编，并在环境 smoke 中真实调用 `knn_points`。r42 完成该合同后 r43 才进入
  100/100 iteration；r39/r41 继续保留 blocked，不倒写为成功。
- `V4-F32`：进度条达到 60k 或路径上存在 checkpoint 不等于 scene executable。r44 只有在 formal step、run 内
  `point_cloud/deform/env` 三文件 bytes/SHA、fingerprint/manifest、source HEAD、六修改文件与兼容补丁全部精确后，
  才由 r45 从 `AD-GS 0/6` 更新为 `1/6`；StreetGS 同样从“存在即计数”收紧为 runtime+bytes+SHA 精确。
  `AD-GS-2026-07-27.patch` 是 zero-context patch，reverse-check 必须显式传 `--unidiff-zero`，否则会产生审计假阴性。
- `V4-F33`：单个方法达到 6/6 不等于 B0 完成。StreetGS r32/r46/r48/r50/r52/r54 已按 strict mod5、
  checkpoint bytes/SHA 与 clean r55 inventory 收口为 6/6，但 V3.3/AD-GS 仍各为 1/6，统一 evaluator 也尚未生成
  完整 scene rows；不得据此启动 M1、读取 test quality，或把 inventory 的 `matched_baseline_assets_incomplete`
  倒写成 StreetGS 失败。后续只补缺失方法/场景并保留旧 inventory。

## V4 P0 防重复结论（2026-08-11）

- `V4-F01`：计划草案记录的 HEAD 不是执行时事实。草案写 `main@144ed19`，P0 实查为 `main@2108430`，且 V3.3
  收口 `e6663e1` 已进入其历史。V4 必须从真实 `main` 建分支，不得回退旧 HEAD 或把草案 provenance 写成 canonical。
- `V4-F02`：计划写“`/root/autodl-pub/KITTI` 已在公共盘”不等于当前机器已挂载。P0 实查目录不存在，状态固定为
  `blocked_local_dataset_missing`。不得创建空目录、下载 KITTI、借其他 layout 冒充，或把外部缺盘写成 adapter/算法失败。
- `V4-F03`：H0 在计划表中写了 `conditional`，但同一计划规定任务状态只允许
  `pending/running/blocked/done/rejected`。V4 注册表将 H0 规范化为 `pending`，条件授权单独记录；不得引入第六种状态。
- `V4-F04`：一手论文、项目页或官方源码存在不等于 baseline 已在本机 single RTX 3090 + same split 执行。SplatAD、
  IDSplat、HorizonForge、RecEdit-Drive 等必须分开记录 paper/source/executable 状态；没有 matched run 不填数值。
- `V4-F05`：KITTI 缺失不阻塞 D0 nuScenes cohort，但会阻塞 single-card closure 的 KITTI adapter smoke。不得因此提前
  读取 nuScenes test、跳过跨数据集条件，或用多卡/新下载掩盖外部前提。
- `V4-F06`：V4 的公式必须同时对应 config、代码、ablation 和可计算指标。P0 只冻结 schema，不代表 M1/M2/M3
  已实现或有效；后续失败必须按预注册早停，不得继续堆 evidence feature、diffusion 或数学包装。
- `V4-F07`：formal run 通过不代表可跳过提交前 whitespace gate。P0 r1 使用的方法合同正确，但未跟踪计划的参考文献
  含 Markdown 行尾空格，`git diff --check` 拒绝提交；规范引用格式后 plan/config SHA 改变。r1 保留 noncanonical
  done，r2 对最终字节重新审计并成为 canonical；不得倒写或覆盖 r1。

<a id="detail-v3"></a>

## V3.3 R0 防重复结论（2026-08-11）

- `V33-F41`：R0 不能把 JSON 中“语义相近”的类型或枚举视为相同。diagnostic 前三次分别把 S2 的空列表写成
  数值 0、把 S3 `heldout` 写成 `heldout_confirmation`、把 S4 `real_renderer_evaluation` 写成
  `evaluation`，均 fail-closed。以后 verifier 必须比较原始类型/枚举；不得用字符串归一化掩盖 schema 漂移。
- `V33-F42`：正式 instance-field validator 通过不代表 NPZ 必须有未约定的 `schema_version`。r4 在报告层
  冗余读取该字段而 failed；修复只移除报告假设，仍执行完整 validator。以后不能把“自己希望存在的字段”变成
  canonical 资产失败，也不能因此跳过正式 schema 校验。
- `V33-F43`：RoadPatch 成为 V3.3 主方法不等于已在 matched 协议下胜过 V3.2 Telea。两者 base、空间语义与
  评测协议不同；R0 答案固定为 `not_directly_ranked`。不得用 B1 相对 B0 的 heldout gate 写成 Telea head-to-head，
  也不得因缺直接排名否定 RoadPatch 的 3D-native/provenance/heldout 成立结论。
- `V33-F44`：内容寻址 release 不允许为“完整”而复制 579 MB base checkpoint。R0 package 含 O1 sidecar、
  RoadPatch/A4/S4 delta、production renders 与 external reference，forbidden model suffix count=`0`；离线 verifier
  同时锁 file set/bytes/SHA。任何新增 `.pth/.pt/.ckpt/.safetensors` 或未登记文件都必须拒绝。
- `V33-F45`：deterministic archive 不能包含当前 run timestamp、绝对输出路径或可变 ZIP metadata。R0 release
  ledger 只引用固定 canonical 输入，ZIP entry 排序/1980 timestamp/permission/compression 固定；diagnostic、
  formal 和同 run replay SHA 均为 `cffaad16...44a7`。以后新增 release 字段必须先证明跨 run byte-exact。
- `V33-F46`：R0 的 `v33_supported` 只覆盖 scene-0230 主链、冻结确认视图和单 RTX 3090。它不证明
  scene-0242/0255 完整 V3.3 transfer、生成 actor GT、相邻视频时序、闭环安全或传感器真实性。F0 LiDAR-EVS
  仍是 conditional 新任务，不能倒写进 R0 的 4/4 success criteria。

## V3.3 S5 防重复结论（2026-08-11）

- `V33-F35`：unconstrained Harmonizer 不能因“只做视觉润色”进入删除生产链。canonical r4 的 edit target
  delete candidate 让冻结 SAM2 semantic mass/fraction 增加 `+0.126399/+0.133885`；production raw fallback
  两项均为 `0`。以后不得关闭 detector、改用候选图作 delete，或只展示另外四个未触发视图。
- `V33-F36`：跨视图平均改善不能掩盖单视图确认失败。五视图 contact 平均为改善，但 heldout f060/c1 的
  contact L1 delta=`+0.422686`，超过冻结逐视图上限 `+0.25`；因此 G1 必须 rejected、production=G0。
  不得改成 aggregate gate、放宽上限、删除该视图或用 edit target 的强改善抵消它。
- `V33-F37`：heldout confirmation 不是第二个开发集。S5 先只用 f091/c1、f005/c0、f065/c1 选择 G1，随后才
  读取 f020/c0、f060/c1；看到 F36 后不得调 contact/shadow 区域、权重、cap 或阈值并继续在同一 heldout 上
  宣称泛化。合法复开需要新假设、新 task 和未读 confirmation 数据；R0 只登记当前负结果。
- `V33-F38`：冻结推理环境不应为共享模块的无关顶层依赖而污染。r1 的 Harmonizer 已完成，SAM2 启动时因
  `semantic_gate.py` 顶层 SciPy import failed；SAM2 只消费 semantic mass/decision，并不构建 gate。修复是将
  SciPy 限在 builder 内 lazy import，不是往冻结 SAM2 环境临时安装包；r1 terminal 保持 failed。
- `V33-F39`：R3D2 official code、Apache license 与 clean commit 不等于存在作者 pretrained pipeline。
  canonical S5 只登记 `blocked_pretrained_model_unavailable`，`model_loaded=false/training=false`。不得拿
  SD-Turbo/TAESD base、Harmonizer 或自训权重冒充 R3D2-fast。
- `V33-F40`：S5 的五个冻结视图不是相邻视频帧，不能从 deterministic image SHA、跨 run exact 或五视图
  quality 推出 temporal consistency。canonical 明确 `not_evaluated_non_temporal_frozen_five_view_protocol`；
  时序 claim 需要独立连续视频协议、新指标与新 run。

## V3.3 S4 防重复结论（2026-08-11）

- `V33-F30`：S1 `hard_instance_id` 是候选身份集合，不等于所有 Background 候选都应被硬 ERASE。r2 将
  high actor 的 `36,736 Background + 4,525 Rigid` 全部设为零，虽然 target coverage=`0.999741`，但目标外
  L1=`0.821965>0.5`，方法按冻结门 rejected。不得通过放宽 L1 门、扩大 target mask 或只报告 coverage 复活该臂。
  合法修复是使用 S1 已训练 instance opacity 的 MAP 正类 `p>=0.5` 选择 Background，同时保留全部物理 Rigid core；
  r8 以 `1,614+4,525` 行将目标外 L1 降到 `0.225349`，门与视角未变。
- `V33-F31`：S2 canonical `roadpatch_delta.npz` 的 104 行同时服务 high/boundary 两个 actor，不是任一单 edit
  都应加载的整体。S4 high package 必须按冻结 `target_role=high_support` 只取 25 行，并保留 parent delta SHA；
  把 boundary 的 79 行混入 high edit 会产生无关场景变化，不能用“同属 Background repair”掩盖。
- `V33-F32`：恢复同一批 Python/Parameter 对象是必要条件，但不足以证明可回滚。renderer 可能持有缓存或顺序状态；
  每个 stack 卸载后必须重新执行 source render 并比较 tensor schema+bytes SHA。canonical r8 为 5 视角×4 stack=
  `20/20` exact，另有 full deterministic replay 和 replay rollback；以后不能只比较 checkpoint SHA 或 object id。
- `V33-F33`：immutable base + delta package 不能把 579 MB checkpoint 复制到 base 目录后仍称“小型 delta”。
  canonical package 的 base 只含 checkpoint/registry reference descriptor，完整 checkpoint copy=`0`，最大 payload
  `3,942,422 bytes`；任何 materialized deployment checkpoint 必须作为独立 deployment factor，不得改写 authoring state。
- `V33-F34`：S4 r3/r4 的方法和指标有效，但最终核心增加 fail-closed policy validator 后其 source snapshot 不再是
  提交态；r5/r6 与最终方法一致，但 builder 后续补齐异常 terminal。最终只认完整重跑 r7/r8 为 canonical；不得以
  “只是校验/失败处理代码”绕过源码 byte-exact 合同。

## V3.3 S3 防重复结论（2026-08-11）

- `V33-F22`：DriveStudio `instances_info.obj_to_world` 在冻结数据中可直接是 list，也可能是历史 JSON string。
  selector r0 只接受 string，在首个候选 fail-closed，未发布 selection。修复必须对两种 schema 都验证 4×4 shape
  与 finite，不得无校验 `json.loads` 或续写 r0；r1/r2 已以新 run 收口。
- `V33-F23`：view selector 的确定性不能用复制 manifest 冒充。high r1→r2、boundary r7→r8 都重新执行全部
  D2 original/delete 候选渲染，formal 传入 diagnostic 的 expected selection/input SHA 并得到 byte-exact；候选池
  必须继续排除 heldout 与 reserved development，不能让评估帧参与 view selection。
- `V33-F24`：更多输入视图不自动等于更好 actor。high development 中 A4 被选择是因为冻结 metric order 下
  IoU/boundary 改善且 LPIPS/PSNR/背景漂移/横移碎裂 retention 全过；heldout 只允许 A0 与 frozen A4 一次确认。
  不得从 heldout 重选 A1/A2/A4，或把生成背面写成 GT accuracy。
- `V33-F25`：V3.2 manual A0 只绑定 high-support token `af663...c5c29`，不是 boundary actor 基线。boundary
  评估必须使用同 identity 的 immutable D2 native actor，不能复用 high A0、换 actor 或仅凭 class 相同跳过三元核对。
- `V33-F26`：Asset Harvester 对 boundary A4 成功生成 PLY/NPZ 不等于 production override 可接受。r12 中 A4
  相对 native 的 IoU/boundary F1 从 `0.666562/0.555343` 降到 `0.624832/0.492141`，LPIPS/PSNR 也失败；
  决策为 `ABSTAIN_GENERATED_OVERRIDE`，未读 boundary heldout。不得通过放宽门、读取 heldout 或只展示 orbit
  render 复活该资产。
- `V33-F27`：CLI 编排提供错误 inference manifest SHA 时，importer 必须在物化前 fail-closed。boundary r10
  因传错 SHA 保留失败证据；正确 SHA 只用于新 r11。不得删除 r10 后续写、绕过 SHA，或把编排错误写成模型失败。
- `V33-F28`：canonical eval 的 source snapshot 必须等于最终提交源码。high r5/r6 的指标与 r13/r14 完全相同，
  但 evaluator 后来增加 native baseline 支持，旧快照不再等于提交态；因此只把 r13/r14 作为 canonical。不得用
  “逻辑没变”绕过 byte-exact 合同。
- `V33-F29`：scene-0242/0255 有旧 V3 checkpoint 不等于存在本任务冻结的 V3.3 S1/S2 mask/actor 输入链。
  在 boundary transfer 已拒绝的情况下更不能混用旧资产补齐跨场景表。合法复开需要新 task、每场景 exact identity/
  mask/checkpoint 协议、冻结 high actor policy 和新 run；旧 S3 terminal 不续写。

## V3.3 S2 防重复结论（2026-08-11）

- `V33-F15`：正式 run 目录不能在 runner 注册前由 `nohup ... > run.log` 预创建。r0 因 shell 先创建
  `run.log` 而触发 non-empty run-directory fail-closed；不得删除日志后续写。正式托管应把 launcher 日志写到
  run 目录外，或由 runner 创建目录后再写；r10/r11 使用新 run 收口。
- `V33-F16`：V3.1 P3 package 的空间网格是 `(x,y)` 且绑定 V3.2 P2 FP16 mixed checkpoint，不是当前 D2 FP32
  原生 Background 的道路索引。DriveStudio 首个 CAM_FRONT 是 OpenCV `x-right/y-down/z-forward`，道路 BEV
  必须使用 `(x,z)`；r1 因错误要求 P3 manifest 绑定 D2 exact SHA 而失败。不得为复用旧索引而放宽 checkpoint hash。
- `V33-F17`：相机内参在当前 DriveStudio 输入中是 9 个值（`fx/fy/cx/cy + 5 distortion`），不是只含 4 个值；
  r2 在正式物化前 fail closed。所有 adapter 必须显式接受已冻结 schema、校验前四项和 distortion 长度，不能静默切片
  后假称已适配其他相机模型。
- `V33-F18`：对整格直接取 `max_scale/max_plane_residual` 会被一个天空、立面或跨层 Gaussian 污染。r3 得到
  `53,541` patches 但 `0` valid；这不证明场景没有道路 donor。修复应先逐行排除 actor/generated/低 support/
  scale outlier，再确定性选择 `<=0.75 m` 的 densest vertical slab，最后做 plane/normal 门。r10 由此得到
  `822` valid patches；不得回到 whole-cell 放宽阈值。
- `V33-F19`：cross-view sidecar 的 `visible_view_count` 与 front-camera frustum observation 是两个合同。
  r4/r5 已有 valid 4 m patch，但把 `minimum_multi_camera_count=2` 当作当前六相机逐相机观测计数，导致两个真实
  target 的 top-5 手工门失败；当前实现明确要求 sidecar `visible_view_count>=5` 且至少一个 front-camera frustum
  observation，不虚构不可得的逐相机 visibility。
- `V33-F20`：donor 几何合格不等于新增 Gaussian 数量可以无限。r8 的 2,150-row dense delta 在 development
  selection 中可见，但 heldout PSNR/SSIM 退化 `-0.8553 dB/-0.00619`，保持 rejected。修复不是从 heldout 选
  top-K，而是在候选资格阶段冻结 `maximum_rows_per_target=512`，让搜索选择最小可见 delta；r11 最终为
  `25+79=104` rows，并通过全部 heldout 门。不得复活 r8 或事后改写其 terminal。
- `V33-F21`：官方 Inpaint360GS source clean、Apache-2.0 不等于当前 StreetGS/3090 上已复现。官方声明
  RTX 4090/CUDA 11.8，并需要主环境、独立 LaMa 环境、CropFormer/Big-LaMa/SAM/DeAOT/GroundingDINO 权重；
  官方代码没有 DriveStudio/StreetGS checkpoint adapter。r12 因这些前置条件 fail-closed 为
  `blocked_single_3090`、`official_execution_attempted=false`。该状态不是 B2 质量负结论，也不得用 Telea、base
  SAM 或自写 RoadPatch 输出冒充官方 Inpaint360GS。

## V3.3 S1 防重复结论（2026-08-11）

- `V33-F07`：磁盘可用空间在 P0 后被外部流程扩大，同时 V3.2 SAM2 checkout/weight/runtime 被删除；不能把
  canonical train masks 仍存在误判为 heldout 推理环境仍可用。普通 clone 又被大型 demo checkout 拖住，已终止本任务
  PID 并保留 `sam2.incomplete-20260811T0138`。恢复只能 sparse checkout exact commit、下载 exact weight，另建
  隔离环境并冻结 package list；不得修改 DriveStudio 环境或复用不完整 checkout。
- `V33-F08`：heldout-target r2 在旧 `/root/autodl-tmp/envs/worldsim-v32-sam/bin/python` 不存在时 exit=`127`；
  prompts 虽生成但 run terminal=`failed`，不得续写。新环境必须复原 V3.2 记录的 Python/torch/torchvision 版本并
  新建 run；r4 已按此收口。
- `V33-F09`：SAM2 singleton predictor 在当前 exact runtime 返回 `[object,1,H,W]`，而旧兼容路径也可能给
  `[object,H,W]`。r3 无条件 `unsqueeze` 产生 5D interpolation 错误；修复必须显式接受 rank 3/4、拒绝其他 rank，
  并新建 r4。r3 未发布正式 mask，不得当质量证据。
- `V33-F10`：更宽的 O3 ambiguous reassignment 并不自动改善边界。100-step development smoke 中 O3 的
  boundary F1/IoU=`0.123499/0.160504`，低于 O1 的 `0.149382/0.181752`，且 FP 更高；O3 已排除。除非提出
  新的几何邻域/身份证据并使用新 task/run，否则不得因“候选更多”重开。
- `V33-F11`：O1 在 heldout 显著改善 boundary/IoU/NBD/FP，但 FN mass 从 `0.061278` 增到 `0.109356`，
  identity presence 仍为 `0.972973`。因此只能声明对象边界与 false-positive 抑制突破，不能声明全面支配或完整
  召回；S2 delete mask 必须继续报告 FN/残留语义，不能用 O1 的高 precision 掩盖漏删。
- `V33-F12`：`np.savez_compressed` 默认把当前时间写入 ZIP entry header；即使 r6/r7 的 O0 全部数组 exact，文件
  SHA 也会漂移。r6 因此保留为 done noncanonical。r7 writer 固定 entry 排序、1980 ZIP timestamp、权限与压缩
  参数，并用同一 field 二次写入 byte-exact 测试锁定容器确定性。该合同不等于宣称 CUDA 训练位级确定性：
  r6→r7 O1 最大 logit/opacity 漂移为 `0.001357 / 8.918e-05`，但 heldout aggregate exact。
- `V33-F13`：正式 run 的 source snapshot 必须与最终提交源码 byte exact，纯 EOF 空白也不能例外。r7 方法与门禁
  均通过，但提交前 `git diff --check` 清理了 4 个新文件的多余 EOF 空行，导致其中 2 个冻结快照不再与待提交源码
  exact；r7 因此降为 done noncanonical，不能仅凭“空白不影响算法”继续引用为 canonical。
- `V33-F14`：长 GPU 任务不能把前台 SSH 生命周期当作任务托管。r8 已完成模型与 finalizer，但 124 秒调用超时关闭
  stdout，外层 `tee` 收到 SIGPIPE，terminal 按预注册 trap 写成 `failed / exit 141`；不得把已有 summary 反推成 done。
  r9 改用 `nohup` 后台托管并以只读 SSH 轮询，最终正常 `done`、GPU 释放、9 个 source snapshots 全 exact。

## V3.3 P0 防重复结论（2026-08-11）

- `V33-F01`：官方 SAM3.1 source 可 checkout 不等于 checkpoint 可执行。当前代码固定为 `96914d2`，但
  `hf auth whoami` 为未登录且 cache 无 SAM3.1；不得绕过 gated access、猜权重 revision/hash，或让该门阻塞
  dual-opacity 主假设。S1 必须 exact fallback 到 V3.2 SAM2.1 canonical masks；未来解锁需新 task/protocol/run。
- `V33-F02`：论文写 code available、GitHub 仓库存在或项目页可访问，不等于存在 runnable implementation。
  GS-RoadPatching `468f812` 只有 HTML/CSS/JS/图片、无算法源码和根 LICENSE；OP2GS、3D-GIMP、FocusGS、
  LiDAR-EVS 也没有可固定官方 runnable source。后续只能称 `*-inspired` 或 `audit_only`，不得写 reproduction。
- `V33-F03`：R3D2 `3fc6e31` 已公开 Apache-2.0 训练/export/eval 代码，但仓库只声明下载 `sd-turbo/taesd`
  base，没有作者训练并导出的 R3D2 pipeline。单卡从零训练不是“补齐 inference”，且被计划禁止；S5 保持
  `weights_blocked`，不能拿 base diffusion 输出冒充 R3D2。
- `V33-F04`：GOR-IS source release 不消除许可与运行合同。根许可证只允许 non-commercial research/evaluation，
  torch/CUDA 未 pin，且要求 nvdiffrast、CUDA rasterizer 和 OptiX gtracer；没有 pretrained manifest。它只能作
  optional audit，不能抢占 RoadPatch 主线或被写成单卡已验证 baseline。
- `V33-F05`：Inpaint360GS 的官方 source/Apache-2.0 只支持进入 adapter/preflight。上游验证环境是 RTX 4090 /
  CUDA 11.8，并依赖外部 CropFormer/LaMa 权重；在 StreetGS split、相机、分辨率和输入 schema 冻结前，不得
  安装/训练。若 24 GiB 下必须静默降正式分辨率、改 heldout 或改相机数，必须 `blocked_single_3090`。
- `V33-F06`：P0 重新 hash 的 D2/S2/S3/mixed/chunk 五资产 exact 与 V3.2 `36 passed` 只证明 immutable baseline
  仍成立，不证明 V3.3 方法有效。P0 全程无训练/模型推理；S1 必须另建 protocol、run 和指标证据。

## V3.2 终局处置与复开门禁（2026-08-11）

V3.2 已以 `WS-V32-R0-INTEGRATION-01=done`、整体 `none_plan_complete` 收口。归档位于
[`archive/2026-08/worldsim-v3.2/`](archive/2026-08/worldsim-v3.2/README.md)。下面的 `V3-F34`–`V3-F46`
继续约束任何后续路线，但不构成继续执行 V3.2 的任务清单。

| 分支 | 终局处置 | 禁止的延续方式 | 合法复开条件 |
|---|---|---|---|
| S1 semantic lift | `done`，canonical r6 | 复用 identity-invalid r5；绕过 ID/token/rigid 三元核对 | 新数据或新语义假设；新 task/protocol/run；继续 fail-closed identity |
| S2 background inpaint | `done`，canonical r3 | 把 Telea unseen RGB 当作 geometry/GT；复活退化的 r2 | 独立深度或多视图证据；预注册未观测 3D 门；新 run |
| S3 actor harvest | `done`，canonical r3 | 把生成背面写成 GT；倒写 CUDA preflight 失败的 r2 | 新 actor/方法假设；固定 source/weight/license；新 task/run |
| S4 harmonizer | task `done`，non-temporal `excluded diagnostic`；temporal `blocked` | 仅凭锐化或全图指标把删除区重生车辆纳入生产链；绕过 gated 权重 | 合法取得 gated base；显式 semantic preservation + temporal gate；新 task/protocol/run |
| S5 multiview upper bound | `blocked`，未授权 | 猜测许可证、移植无根许可证代码或把未执行写成质量结论 | 明确可执行许可证与权重；独立资源审计；用户重新授权的新 task |
| R0 integration | `done`，canonical r4 | 从 exact package 外推 streaming、跨场景、闭环安全或 GT correctness | 为对应 claim 增加独立数据、协议、测量与 run；不得续写 r4 |

统一复开规则：外部门禁解除只改变“是否可提出新任务”，不会把 S4 temporal 或 S5 自动变成当前任务。任何复开都
必须引用相关失败 ID，使用新 task ID、新冻结 protocol 和不可复用 run ID；旧 `blocked/rejected/done` terminal
保持不可改写。当前 `next_action=none_plan_complete`。

## V3.2 防重复结论（2026-08-10）

- `V3-F34`：actor role 必须同时绑定 dataset instance ID、`instances_info.id`/instance token 与 checkpoint
  rigid model index。只分别验证 class、token registry 和 core count 会允许“2D mask 属于 actor A、D2 core 属于 actor B”
  的静默错配。所有 prompt、semantic lift、asset generation 在运行前必须 fail-closed 核对三元 identity；旧 r5
  因 ID `5`/token `af663…` 错配而失效，不得通过后续 adapter 或人工选图补救。

- `V3-F35`：AutoDL 根 `.condarc` 可把 `nvidia` channel 重写到缺包镜像；第三方官方 setup 的 CUDA channel
  不能只凭 channel 名复现。Asset Harvester 必须使用 `--override-channels` 和明确的官方 NVIDIA/defaults URL，
  同时记录 setup 日志；TUNA 对应 404 不能被误判成官方包不存在。
- `V3-F36`：复制第三方 setup 脚本到 `/tmp` 后，基于脚本位置计算的 `REPO_DIR` 会静默变成 `/tmp`。
  transport-only patch 必须冻结真实 checkout 绝对路径，并让恢复 wrapper 显式接收 formal `RUN_DIR`；不得把
  环境完成结果写入已拒绝的旧 run。
- `V3-F37`：`gsplat` 的浅克隆或 transport-only 复制不会自动带上 GLM submodule。Asset Harvester
  环境不能只以 `pip install` 成功为准；必须固定 `gsplat` commit、初始化 GLM 到 exact commit，
  并在当前 GPU 上 import CUDA extension。
- `V3-F38`：第三方 setup 子进程里的 conda activation 不会传回父 wrapper。恢复脚本后续记录、
  校验或 formal 推理必须使用明确的环境 Python 绝对路径，不得依赖裸 `python`
  或父 shell 的隐式 PATH。
- `V3-F39`：PyTorch 2.10 下 `torch.cuda.manual_seed_all` 不会立即建立 CUDA context；在此前调用
  `reset_peak_memory_stats(0)` 可以在模型加载前报 `Invalid device argument`。资源监控 runner 必须先
  `set_device` + `cuda.init`，再清零峰值计数器。S3 r2 因此在 GPU peak=`0 MiB`、无部分输出时
  `rejected`；修复后必须新建 formal run，不得改写 r2。
- `V3-F40`：边界目标的时间邻近帧不等于存在静态世界几何重叠。S2 r1 的固定支持集合跨过 frame `31` 后，
  boundary mask 的有效跨视图覆盖低于 `32` 像素；放宽深度门也不能修复零几何重叠。后续只能在 train-only
  视图上冻结 exhaustive camera/frame geometry audit，再新建 run；不得读取 held-out 来选支持帧，也不得单纯
  放宽深度容差掩盖视锥不重叠。
- `V3-F41`：2D unseen completion 可生成不等于其深度足以作为全时段静态 Background。S2 r2 把全部未观测
  Telea 区域写入 3D checkpoint 后，四路 held-out 平均 PSNR/SSIM 退化 `0.495842 dB / 0.007160`，形成后续帧
  灰色遮挡；候选必须拒绝。S2 r3 保留完整 2D unseen artifact/provenance，但高支持 checkpoint 只持久化
  cross-view observed geometry，并重新通过未放宽的 held-out 门。后续不得把 inpainted RGB 自动升级为
  geometry-grounded world state；若要持久化 unseen 3D，必须增加独立深度/多视图证据和新预注册门。
- `V3-F42`：Harmonizer 导出 JIT 不是脱离官方 NGC runtime 即可原样执行的普通 TorchScript。当前权重包含
  `tex_ts::rmsnorm_fwd_inf_ts`，PyTorch 2.10 还会把两个 einops shape scalar 随 `map_location` 移到 CUDA，
  造成 shape tensor CPU/CUDA 冲突。当前适配只允许使用独立公式验证为 BF16 exact 的 RMSNorm 回退，并将整数
  1/2 shape scalar 放回 CPU；必须记录 runtime deviation 和测试，不能写成 untouched official runtime。
- `V3-F43`：生成式 final-render enhancer 可以恢复外观，同时破坏明确的 counterfactual 语义。S4 r2/r3 在
  G1 remove+inpaint 区域重新生成 actor-like 黑色车辆外观；r3 的 mask 内 L1=`14.217278`、changed fraction
  `0.541750`，失败冻结 `12.0 / 0.40` 门。全图 PSNR、outside drift 或“看起来更锐利”都不能覆盖 actor deletion
  失败；non-temporal Harmonizer 仅保留 optional diagnostic，不得默认进入 remove 输出链。未来复开必须增加
  显式 semantic conditioning/preservation 和连续帧 temporal gate，并取得 gated Cosmos base 的合法授权。
- `V3-F44`：DriveStudio 只读 forward 使用 `torch.inference_mode()` 不等于 trainer 已切换到 eval。训练态
  renderer 会对 `means2d` 调用 `retain_grad()`，与 inference tensor 冲突；R0 r2 因此在首个 forward 明确
  `rejected`。所有只读 replay 必须在每次 state load 后显式 `trainer.set_eval()`；失败 run 不得补写结果，
  只能修复代码后新建唯一 run。
- `V3-F45`：质量门必须冻结指标、区域、单位和范数；`MAE<=1 uint8` 不能被实现成逐像素 L∞/max-error
  `<=1`。R0 r3 的 source→mixed PSNR=`67.24–68.43 dB`、MAE=`0.0093–0.0123`，但两视角存在极少量
  max error=`2`，因错误合同仍保持 `rejected`。修复必须更新 protocol/runner hash 后新建 run，不能在旧结果上
  改 gate 或把 max error 隐藏。
- `V3-F46`：R0 `done` 只证明当前 scene 的 V3.2 资产可追踪集成与固定三视角 storage/package 等价。
  `GENERATED_BACKGROUND` 和 `GENERATED_ACTOR` 仍不是 GT；S4 仍被排除；S5 仍阻塞。432 MB mixed checkpoint、
  444 MB chunk payload 与 8.36 GiB 峰值也不证明 streaming、load、render、跨场景泛化或闭环安全收益；未来相关
  claim 必须有独立 protocol、数据与测量，不能由 R0 exact reassembly 外推。

- `V3-F18`：A3 R1 的工程链可逐位重放，但 heldout 评测越过冻结 GPU ceiling，且资源无效 diagnostic 为
  geometry 改善与 RGB safeguard 退化并存。后续不得提高旧 ceiling、替换旧 renderer 或继续调同一四步配方。
- `V3-F19`：传感器原始尺寸、checkpoint 原生加载尺寸和评测输出尺寸是三个不同合同。A4-P0 v1 的
  `1600×900` 误记不能在后续路线重现；每次 profile 必须同时记录三层尺寸和 downscale 来源。
- `V3-F20`：checkpoint `state_dict` 键不等于加载后的 runtime attribute。任何新资产注册、恢复或 streaming
  代码必须同时审计保存端、加载端赋值和 live object，不能从序列化 schema 猜运行时 API。
- `V3-F21`：A4-P1 最小预注册剪枝臂 b05 已违反 global/non-target 质量门。不得事后增加 b01/b02、放宽
  `0.10 dB` 或只挑 actor/boundary 指标，把同一结果改写成剪枝成功。
- `V3-F22`：A4-P2 只证明选择性 FP16 参数存储在冻结质量门内可把 checkpoint 减少 `25.35%`；它没有证明
  renderer、load、FPS 或 peak VRAM 加速，且 Gaussian means 不能安全降为 FP16。
- `V3-F23`：A4-P3 只证明 159-file static/actor chunk package 可 exact 重组。package 比 source 大
  `2.79%`、load/reassembly 更慢；没有 demand loading、cache policy 和驻留集测量时，不能声称 streaming/LOD 收益。
- `V3-F24`：F0 只完成 Instant NuRec 官方能力与本机前置审计；本机没有执行 inference，standalone CLI 只导出
  static PLY。不得把它写成前馈基线质量失败，也不得把 static PLY 当完整 dynamic WorldSim checkpoint。
- `V3-F25`：R0 的 63 inputs、23 decisions、12 deliverables 与 P3 package exact 只证明 V3.1 证据链闭环。
  它不证明 D2 dominance、R1/P1 有效、P2/P3 加速、完整 world model、跨场景泛化或闭环安全。

## V3 启动时必须先读的结论（2026-08-05）

- `V3-F01`：M4 的 non-target PSNR 93/95 dB 是硬局部编辑的构造/保持性证据，不是编辑后视觉质量。
- `V3-F02`：DriveStudio 已有 Affine、CamPose 与 LiDAR 初始化；A1 必须做 off/native/enhanced 消融，
  不得把上游能力改名为新增模块。
- `V3-F03`：V2 M5 未完成。0230/0242 checkpoint、Tier A/B/C 和 0255 诊断可复用，但不得把部分资产
  写成三场景压力测试通过。
- `V3-F04`：scene-0255 是小输入 CUDA `torch.cat` 工程阻塞且无 OOM 证据，不能写成 3DGS 方法失败。
- `V3-F05`：三个 scene 只支撑模型消融和工程判断；不得外推 trainval、夜间、长时或复杂交互。
- `V3-F06`：Instant NuRec 等工作已经改变前馈基线边界；DGGT 只作历史范式对照，不做跨分辨率、跨输入、
  跨训练预算的 leaderboard。
- `V3-F07`：persistent identity、actor binding、scene graph 和基础 trajectory edit 已由上游与 V2 覆盖，
  不能作为 V3 模型贡献。
- `V3-F08`：rolling shutter 需要真实 readout direction/time；没有 metadata 时必须 `not_supported`，不得
  从帧时间或相机顺序推测行曝光时间。
- `V3-F09`：actor-aware densification 必须分 D0–D3 小步消融；不得一次加入 boundary、LiDAR、visibility、
  residual 后只报告一个合并结果。
- `V3-F10`：编辑后 local refinement 的 unknown background 仍是 unknown；只允许 Tier-A、多视图或 LiDAR
  支持监督，Tier B/C 不得当伪真值回传。
- `V3-F11`：全图 PSNR/SSIM 不能代替 actor/边界质量；counterfactual mask 也不是真值分割，必须同时报告
  visible-image/pixel coverage，避免目标未渲染时通过缩小分母得到虚高指标。
- `V3-F12`：nuScenes processed camera ID 必须以数据加载器事实源映射；显示标签写错会把非相邻相机当成
  预注册相机对，已有 formal 必须 rejected 后重跑，不能只改图标题。
- `V3-F13`：seed=0 不保证 CUDA visibility filter 后的随机背景初始化逐点/逐计数复现。记录的 LiDAR/actor
  tensor exact 可作门禁，重建初始化 depth 只能作 witness，不能冒充源训练初始化 exact residual。
- `V3-F14`：局部 role、全图画质或 learned correction 稳定性改善不能替代预注册阶段主端点。C2/C3 未通过
  E1/E2 合同就不能为了保留增强模块而成为 C*。
- `V3-F15`：确认场景的原始端点方向可以与开发场景相反。不得把完整 Pareto 合同的 `done_off` 改写成
  “C0 在所有场景、所有指标都最好”，也不得只挑 0255 E1/E2 error 改写 C*。
- `V3-F16`：A2-D2 的边界改善、global/non-target 退化与更高训练成本构成严格 Pareto tradeoff。不得用新增
  事后标量权重把它改写成 D2 dominance；后续采用 D2 必须同时登记 D1 fallback 和完整退化轴。
- `V3-F17`：Gaussian ancestry、counterfactual footprint 和未提交 V2 M5 产物都不能自动升级为 A3 真值。
  ancestry 只证明来源，paired mask 只是模型诊断；A3 必须使用已提交输入和 typed support，S-C 保持 unsupported。

### V3-F01：局部保持不等于编辑质量

M4 的 lateral/delete non-target PSNR=`93.394483/95.598042`，主要来自编辑器只改变目标 actor 并保留其他
Gaussian。它证明实现没有意外改动非目标区域，但不能证明 source footprint 后方背景正确、actor 边界自然或
连续帧无闪烁。V3 必须把 outside preservation 与 Tier-A hole、depth ordering、boundary 和 temporal 指标分开。

### V3-F02：原生校准和初始化不能重复发明

DriveStudio `e59bda4` 的 `AffineTransform` 已输出 RGB affine，`CameraOptModule` 已学习 3D 平移和 6D 旋转
残差，数据集也已从 LiDAR 初始化背景/实例。A1 的合法动作是关闭/原生/增强的受控消融，以及 support provenance
审计；不能把启用原生 config 写成新成像、位姿或 LiDAR 模块。

### V3-F03/F04：V2 M5 部分证据与 scene-0255 工程阻塞必须分开

V2 M5 没有生成预注册的 24 条有效序列和 final matrix。scene-0230/0242 checkpoint 是有效训练资产；
scene-0255 训练则阻塞于 `datasets/driving_dataset.py` 实例点列表的 CUDA `torch.cat`。r27 观察到 166 个
CUDA float32 tensors、152 个 `(0, 3)` 空 tensor、177 scalars，且 `oom/oom_kill=0`。V3 A0 可以基于此做
最小 compatibility fix，但必须使用新 task/run，不能改写 M5 terminal，也不能由诊断完成推断训练完成。

V3 A0 已用 `436cfc1` 实现配对过滤：点与颜色按同一个 empty-row 条件过滤，全空时返回 prototype view。
canonical smoke `20260805T161656Z__scene0255-catfix-s0-r2` 在原生错误复现后完成真实 dataset init、1-step
优化与 checkpoint，说明该工程阻塞已在 smoke 范围解除。随后新 30k run
`20260805T162355Z__scene0255-native30k-s0-r1` 完成 checkpoint、registry 与 held-out 评估；0230/0242 通过
严格等价合同复用。该兼容问题现已闭环，但只证明工程修复和 A0 基线成立，不证明任何 A1/A2 方法提升。

### V3-F11：全图质量与模型差分 mask 都有明确边界

A0 中 scene-0242 全图 PSNR=`29.107`，高于 0230/0255，但其 high actor 区域 PSNR=`19.788`，反而是三场景
最低。scene-0255 boundary actor 区域 SSIM=`0.526`，也没有被全图 SSIM=`0.743` 反映。后续 A1/A2 不得只用
全图指标判断动态对象提升。

A0 actor mask 来自同一 checkpoint 的 original 与 actor-delete 配对渲染差分，是模型 counterfactual
diagnostic，不是 nuScenes 真值 segmentation。如果模型没有画出 actor，mask 会缩小；因此每个结果必须同时报告
candidate/visible image、effect pixel coverage 和 `ABSTAIN`。tight-crop LPIPS 用固定 8px padding 与 256px 输出，
不能和全图 DriveStudio LPIPS 混为同一指标。

A0 finalizer r1 因复用 checkpoint run 使用 `source_training_resources`、原生 run 使用 `train_resources` 而
`blocked`。这是汇总 schema 兼容失败，不是模型失败；`00ba4e8` 增加显式 provenance 归一化，r2 为唯一完成矩阵。

### V3-F05/F06/F07：结论规模与研究边界

三个固定 scene 足以比较相同数据、预算和实现下的 A0–A4，但不构成数据规模、天气、城市或交互分布覆盖。
Instant NuRec、OmniRe、IDSplat、SplatAD、ADGaussian、Real2Sim、RoVES 等工作分别覆盖前馈分层重建、
实例场景图、传感器和物理方向；V3 的价值来自完整复现、窄模型改动、负结果和工程 Pareto，而不是重新命名
已公开能力。只有 A2/A3 在至少 2/3 场景方向一致且资源稳定，才讨论扩展场景。

### V3-F08/F09/F10：禁止不可归因或无真值捷径

rolling shutter 没有 row timing 就不能实现；actor-aware densification 必须从 actor/background threshold 与
quota 开始，再分别增加 boundary/residual 和 LiDAR/visibility；local refinement 必须冻结 affected set 外参数，
并区分 expected/first-hit/measured depth。不得用 hard-composition outside=0、原图 actor 像素或未知区域的
自洽渲染作为方法成功证据。

### V3-F12：相机标签错误会污染跨相机端点

A1-E0 初版沿用了错误的显示顺序 `0=FRONT_LEFT / 1=FRONT / 2=FRONT_RIGHT`，但 DriveStudio nuScenes
事实源明确为 `0=FRONT / 1=FRONT_LEFT / 2=FRONT_RIGHT`。结果是名义上的相邻相机对可能实际落到
左右两侧非相邻画面，零支持也会被错误解释为模型现象。首次 formal
`20260806T140703Z__scene0230-c0-a1-e0-formal-full-s0-r1` 因此已标记
`rejected / INVALID_CAMERA_ID_LABEL_MAPPING`，原 terminal/manifest/summary 以 `*.original_done.json` 保留；
`d85ef27` 修复后 C0/C1 使用新唯一 run 回填。

防重复门禁：相机 ID/name 映射必须来自训练数据加载器或预处理权威列表，写入 resolved config 并纳入 hash；
QA 必须验证投影落在实际重叠的建筑/路面。若映射错误，所有受影响正式结果必须 rejected，不得通过重命名
已有 JSON、图片或曲线继续使用。

### V3-F13：随机 CUDA 可见性筛选不等于 exact 初始化 replay

A1 最小 LiDAR provenance 的 strict smoke
`20260806T142900Z__scene0230-a1-lidar-provenance-smoke1-s0-r1` 观察到：800,000 个背景 LiDAR 点、全部 24 个
actor point/color tensor、75,002 个 RigidNodes 初始点均 exact match，但随机 near/far 球面候选经过 CUDA
visibility filter 后，背景初始 Gaussian 数从源运行 946,484 变为 replay 的 946,597；后续 replay 又得到
946,309 和 946,291。这不是 LiDAR 输入变化，也没有训练或 checkpoint 修改。

防重复门禁：冻结的 `a1_lidar_provenance_v1.yaml` 要求记录 LiDAR/actor tensor exact match，并记录随机球面
候选、visibility mask SHA 和计数；背景 exact replay 固定为 `report_not_gate`，不允许事后设置“接近即可”的计数
容差。正式初始 depth residual 必须标为
`seed0_reconstructed_initialization_witness_not_exact_source_initialization`。要获得源训练初始化的 exact depth，未来
必须在训练创建时直接持久化 post-filter 初始化 tensors；A2 的逐 Gaussian ancestry 仍需独立 instrumentation。

### V3-F14：局部改善不能替代阶段主端点

scene-0230 中，C2 的 boundary-support E2 mean/P90 从 C0 的 `0.003547/0.006353` 改善为
`0.003346/0.005447`，但 high-support E2 P90 退化到 `0.011734`，actor/boundary LPIPS 也整体退化；因此不能把
单个 role 的改善提升为整个 E2 端点改善。C3 的全图 PSNR/LPIPS、boundary actor 质量和 learned pose correction
稳定性均最好，但 E1 median/P90 与两个 E2 role 仍未严格优于 C0。

A1-S0-v1 在结果已可见后、确认场景前把 V3.1 7.5 操作化为无容差严格 Pareto，并如实披露该时点；没有新增
事后数值阈值。正式结论必须是 `C*=C0-off / done_off`。不得更换 role、放宽端点、只引用 C3 全图画质或把
learned correction 幅值写成 pose GT，以强行保留增强模块。

### V3-F15：完整合同通过不等于每项指标方向一致

scene-0242 的 C0 在 global、E1 和 high E2 上优于 C1；scene-0255 则相反，C1 的 E1 median/P90 和两个 E2
role error 都更低。但 0255 C1 的 high E2 coverage 从 `23.529%` 降至 `21.569%`，boundary/high actor LPIPS
均退化，因而仍未通过冻结的“主端点改善、另一端点不退化、appearance LPIPS 可接受”完整合同。

A1 finalizer 的合法表述是：C1 在两个确认场景均不 eligible，C*=C0 保持 `done_off`，同时原始端点方向具有
scene dependence。禁止写成“C0 普遍校准更优”，也禁止忽略 coverage/appearance 只引用 0255 error 重选 C1。

### V3-F16：边界优先分支选择不等于全面方法提升

A2 formal 中，D2 相对 D1 的 boundary-support boundary-band PSNR/SSIM/LPIPS 从
`25.770024/.821572/.048382` 改善到 `26.171399/.828868/.044568`；但 global 从
`27.770024/.850915/.177704` 退化到 `27.703188/.850333/.178344`，non-target PSNR/SSIM 也下降，训练
wall time 从 `2099.33 s` 增至 `2720.82 s`。fixed 与 matched strict-quality、quality-cost 裁决都为
`tradeoff_non_dominated`，且 matched D2 只是 fixed 30k 的 exact alias，不是独立复现。

A3 采用 D2 是因为 A2 的预注册靶点包含 actor boundary，并且 D2 在该边界带三项指标同时改善；这是完整结果
可见后的工程资产路由，不是新增数值门槛、统计显著性或 D2 对 D1 的支配结论。任何后续报告都必须同时保留
D1 quota-only fallback，披露 D2 的 global/部分 actor/non-target/cost 退化，并禁止只摘录边界带结果宣称 A2
“全面提升”。单场景 scene-0230 也不能支持跨场景泛化结论。

### V3-F17：来源账本与 paired mask 不等于局部精修监督真值

D2 final checkpoint 的 Background ancestry 完整对齐 `1,205,164` 个 Gaussian，其中 `240,528` 个
`init_source=LIDAR` direct roots，其余含 random、split 与 clone；`nearest_lidar_distance` 对部分 lineage 有限，
但它是出生/父子来源记录，不是当前 target ray 的 T0 measured depth。A3 只能把 calibrated LiDAR projection 的
`depth_lidar_measured` 当 T0，把 first-hit 当 T1 ordering，把 expected depth 保持 diagnostic。

同样，source/edited footprint 来自同一 checkpoint 的 paired RGB difference，只能定位干预区域，不能充当真值
segmentation 或删除后的背景 RGB。S-A RGB 监督必须来自排除 target view 的 alternate camera/time 真实观测并有
calibrated reprojection；S-B 只使用 measured LiDAR 或至少两视图 geometry，禁止 RGB loss；S-C 不更新、不 seed、
不进入 loss，只报告 coverage/uncertainty/ABSTAIN。

当前工作树中的 V2 M5 protocol、`stress_metrics.py` 和 stress runner 均未提交且属于被冻结的用户工作，A3 不得
通过 import 或复制其结果建立隐式依赖。只能复用已提交并按 SHA 冻结的 M4 edit、paired mask、typed-depth 与
registry 接口；否则无法形成 clean source commit，也会把 V2 未闭环事实倒写成 V3 证据。

### V3-F18：A3 工程可重放不等于局部精修可晋级

A3 R1 已证明四个 S-B/T0 unit 的 opacity/scale 更新可以逐位重放，但可变集合只有 `51` 个 Background rows、
四个 unit 合计只有 `8` 个 T0 geometry pixels，S-A/RGB 为 `0/ABSTAIN`。heldout r2/r4/r5 的单 view 峰值稳定在
`14,241–14,245 MiB`，超过结果前冻结的 `12,288 MiB` ceiling；r5 的资源无效 diagnostic 同时出现 depth-order
改善和 non-target/original-global RGB MSE 严格退化，exact Pareto 为 `tradeoff_non_dominated`。

因此后续若复开局部精修，研究变量必须先变为“可观测支持如何获得、分层和拒绝”，而不是继续调 R1 的 step、
LR、alpha、mask dilation 或旧 renderer。合法复开需要新任务、新协议、与 heldout 隔离的支持审计，以及冻结前
证明 S-A 或更充分 T0/多视图证据确实存在；否则保持 `A3*=R0/D2 exact alias`。

### V3-F19：部署尺寸必须分为 sensor、model-native 与 evaluation 三层

A4-P0 v1 把 nuScenes sensor `1600×900` 写成 checkpoint 原生分辨率，但 source config 已固定
`downscale_when_loading=[2,2,2]`，真实模型加载和 render 均为 `800×450`。v1 保持 `blocked`，v2 只纠正输入
合同并在新 run 完整重跑；不能使用 v1 的性能数字关闭 P0。

后续所有 runtime、质量和资源协议必须显式记录 sensor resolution、source-config downscale、model-native
resolution 和最终 evaluation/output resolution。任何一层变化都是新的实验因子，不能用“native”一词静默折叠。

### V3-F20：序列化 schema 不能替代 live runtime API 审计

A4-P5 r1 已生成合法 registry，却把 checkpoint key `points_ids` 当作加载后的 `RigidNodes.points_ids`；实际
`load_state_dict` 会把它写入 `self.point_ids`。r1 因此保留 `blocked`，修复后的 r2 才通过 14/14 audits。

后续做 lazy loading、分块恢复、资产注册或 checkpoint 迁移时，必须分别验证 state key、load hook、live attribute
和调用方，并用真实 fresh-process reload 测试锁定。不能因为 checkpoint 中存在字段，就推断运行时对象暴露同名接口。

### V3-F21：预注册最小剪枝臂失败后不能事后缩小 fraction

A4-P1 的 b05/b10/b20 均通过结构、reload、count 和资源审计，但最小 b05 已使 global occupied PSNR、global PSNR
和 non-target PSNR 分别退化 `0.117684/0.110926/0.125462 dB`，超过冻结 `0.10 dB` 门；更大 fraction 失败更多
端点。局部 actor/boundary 指标保持或改善不能覆盖全局与非目标区失败。

后续不得在同一 ranking、视图和结果上新增 b01/b02、改变阈值或只报局部轴。若重新研究压缩，必须有不同的、
结果前可解释的结构假设与新预注册；单纯把 fraction 调小不是新的研究问题。

### V3-F22：FP16 存储压缩不等于端到端加速

A4-P2 把 Background/RigidNodes 的 10 个 scale/quat/feature/opacity tensors 转为 FP16，checkpoint 从
`578,819,674` 降到 `432,111,754` bytes，31/31 quality safeguards 通过。但 candidate 的 load、P50 和 FPS 没有
形成一致加速，renderer 输入仍显式转回 FP32；source audit 还表明 Background means 若做 FP16 roundtrip，最大
空间误差接近 `1 m`。

因此当前合法 claim 仅为 `mixed_precision_parameter_storage_fp32_render`。后续若研究低精度执行，必须单独冻结
renderer dtype、数值误差、质量、peak resident memory 和 latency 合同；不得从文件变小推断 Tensor Core、VRAM
或实时收益，也不得把 means、trajectory 或 provenance 一并降精度。

### V3-F23：exact chunk package 不等于 streaming/LOD 系统

A4-P3 的 133 static + 24 actor + skeleton + manifest 共 159 files 可 exact 重组，57 RGB SHA、31 endpoints、
85 tensor paths 和 source/registry immutability 全通过。但 package 比 source checkpoint 大 `2.792171%`，全量读取的
load/reassembly 与 render 均未加速，filesystem cache 也未控制。

后续只有在实现真实 demand loading、明确 working set/cache/eviction、记录首帧与稳态 latency、peak resident bytes、
I/O bytes 和 exact fallback 后，才可研究 streaming/LOD。继续把同一 159-file package 全量读入内存，只能叫资产
分离，不能叫部署加速。

### V3-F24：F0 前置失败不是前馈方法质量失败

Instant NuRec canonical audit 只通过 4/11 prerequisites；Python 3.11、uv、30 GiB VRAM、100 GB free disk、exact
weights、licensed NCore input 与 terms record 未同时满足，所以 `inference_command_constructed=false`。官方 standalone
CLI 的实际输出又只含 static PLY，不含 dynamic/sky/ISP/actor registry/trajectory/depth。

后续不得把“本机未运行”写成 upstream quality reject，也不得用 static PLY 与 StreetGS 完整 checkpoint 做假等价
比较。只有硬件、许可、数据和 converter 接口全部独立满足后，才能用新任务做窄范围前馈 pilot。

### V3-F25：证据链 exact 不等于研究主张自动成立

R0 canonical 的 63/63 inputs、23/23 decisions、12/12 deliverables、26/26 manifest files 与 P3 159-file package
全部 exact，证明 V3.1 可以从冻结事实恢复同一结论与生产链。它没有增加场景、seed、真值、闭环控制或新的方法臂。

后续研究必须从一个明确、可证伪的新问题出发，并说明新增证据解除哪一条失败约束；不能把 R0 的可复现性重新命名为
完整 world model、跨场景泛化、物理真实性或安全性。若主张涉及 A2/A3 方法，至少需要独立场景确认；若主张涉及
部署收益，必须直接测量对应 runtime/working-set 端点。

<a id="detail-v2"></a>

## V2 启动时必须先读的结论（2026-08-02）

- `PIVOT-F03`：AD-GS exact reproduction 已完成；V2 只读最终 checkpoint/render/metrics，不重复训练。
- `PIVOT-F04`：可见性建模不等于未观测背景真值；M5 必须保留 Tier A/B/C。
- `PIVOT-F05/F14`：资源、外部实例与方法失败分开；OOM/重启不能写成模型质量结论。
- `PIVOT-F06/F07/F08`：换机、非登录 shell 与浮动权重必须重新审计；镜像不能改变固定版本。
- `PIVOT-F10/F11/F12/F13`：PNG/JPEG、COLMAP 并发、cgroup 90% 和合法空占位均已有失败证据。
- `PIVOT-F14B`：V1 pointops2 的直接根因是 PEP 517 隔离构建缺少 torch；V2 先按 upstream
  `python setup.py install`，不重复原 `pip install .`。
- `PIVOT-F15`：AD-GS camera-local pseudo ID 与二值 `obj` 不能支持对象级编辑；V2 以 nuScenes
  `instance_token` 只做评测真值，不注入 AD-GS 训练。
- `PIVOT-F16`：持久身份、actor binding 与基础轨迹编辑本身已不新；V2 必须先产生跨三场景真实失败，
  再做新的 novelty gate。

存储清理只使历史环境和中间 checkpoint non-resident，不撤销上述失败，也不允许重新运行已关闭路线。

<a id="detail-legacy"></a>

## N1 kinematics-first 第三次 reject 与第四版约束（2026-07-25）

### N1-F12：地图分支收敛不是车辆横向机动

**观察**

- 第三次人审文件：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-01/v71_n1-event-kinematic-01__kinematic-v1__s0__20260725T092427030639Z__8c2247b6/audit/review_working.jsonl`；
- review SHA256：
  `005cd74b874833808435fd2f47387d1d8e446cdea2d3a5cae6146e34bf331e96`；
- 12/12 已审，`TRUE_POSITIVE=0`、`FALSE_POSITIVE=12`、`UNCERTAIN=0`，precision=`0`；
- subject maneuver 为 `INVALID` 12/12；failure code 为
  `SUBJECT_NO_LATERAL_MANEUVER=12`、`ROUTE_CONTINUATION=11`、`NORMAL_TURN=1`、
  `MAP_MATCH_JITTER=1`；
- 第三版 12/12 机器候选都是 `converging_branch_merge`。规则只验证 source/target
  地图分支在几何上汇合，却没有验证车辆中心/车身相对接收车道发生 outside→inside 横移；
- target corridor 人审 12/12 为 `VALID` 并不能挽救 subject maneuver。地图画对了，不等于事件成立。

**根因**

第三版仍把“actor 沿一条会汇入 target 的道路行驶”当成“actor 主动切入 target 车流”。车辆可以保持正常
转向/道路中心线跟随，而道路本身向另一分支收敛；仅比较 source/target approach heading 或地图 token
变化仍会把路形变化误写成车辆运动学。

**防重复**

- 必须直接从原始 2 Hz annotation 计算 subject 相对接收 corridor 的连续横向状态；
- 至少观察目标车道中心外→中心内，并在进入后保持名义 1 s；10 Hz 插值不参与物理门；
- 进入前还必须与接收 corridor 近似同向，避免把大角度路口/主路续接的几何距离收敛当作 cut-in；
- 不再把 `merge` 地图类别、multiple incoming、route token change 或道路弯曲本身当正例。

### N1-F13：接收车必须来自独立目标车流，不能复用 subject 后车

**观察**

- 第三次 review 中 rear 为 `INVALID` 2/12、front 为 `INVALID` 1/12；
- 第三版 corridor 构造会贪心选择与 subject source 最顺的 incoming，再在该 corridor 上找 rear；
- 因而所谓 rear 往往就是 subject 原队列中的后车，而不是被切入目标车流的接收车；
- K3-004 选错 front branch；K3-007、K3-010 选错 rear branch。其余多项虽被人审写成 corridor
  `VALID`，也只说明地图链连续，不证明 receiver 角色语义成立。

**第四版硬约束**

1. parallel lane change 的 target chain 显式排除 source token；
2. merge 只枚举 `target` 的 direct incoming 中不同于 subject source 的分支；
3. RECEIVER 必须在进入前后保持同一 identity、同向、最近后车次序与 `[0.5,40] m` bumper gap；
4. subject/receiver 之间不得遗漏更近同 corridor 车辆；
5. negative control 也必须存在持续 receiver，不能用孤车普通直行冒充交互密度等价 control。

### N1-F14：第三次裁决的研究失败与工程失败必须分开

**研究裁决**

- clean adjudication commit：`1fbbbc1`；
- 成功 run：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-AUDIT-01/v71_n1-event-kinematic-audit-01__human-audit-reject-v1__s0__20260725T155754010881Z__4c51f0d9`；
- 唯一终态 `REJECTED`，`n2_authorized=false`。

**保留的工程失败**

第一次 formal adjudication 使用了错误的 audit-manifest 指纹键，在写入研究产物前失败：
`.../v71_n1-event-kinematic-audit-01__human-audit-reject-v1__s0__20260725T155523736677Z__4c51f0d9/`。
该目录保留 `FAILED/failure.json`，原因是 `engineering_manifest_key_mismatch`。修复将
`artifact_set_sha256` 更正为实际 schema 的 `immutable_artifact_set_sha256`，并把所有输入校验提前到
run 目录创建之前。不得删除失败尝试或把它统计成 research reject。

### N1-F15：四图全常驻与重型 map API 会触发 2 GiB cgroup 峰值

**观察**

- 第四版首个 development smoke 在算法开始前以 `RC=137` 被杀；
- 容器 `memory.max=2147483648`，当时常驻服务已占约 `1.85 GiB`；
- 官方 `nuscenes.map_expansion.map_api` 的导入会连带 OpenCV、Matplotlib、Shapely 和渲染 API；
  单是 import probe RSS 就从约 `58 MiB` 增至约 `212 MiB`；
- 同时常驻四张 `NuScenesMap`、完整 sample/instance JSON 行和 128-scene dense batch 会进一步放大峰值。

**工程修复**

- 新增只读取 `lane`、`lane_connector`、`arcline_path_3`、`connectivity` 的轻量 map reader；
- arcline 离散化与官方 devkit reference 在单测中逐点一致；
- map index 改为一次只缓存一个 location，calibration/evaluation 按 location 排序；
- `sample.json`、`instance.json` 改为 ijson 流式最小字段投影，scene builder 复用同一 metadata source；
- scene batch 冻结为 32；不得通过杀死用户编辑器进程、修改容器上限或跳过地图证据来“解决”。

### N1-F16：负对照配置契约缺项导致首个正式 K4 工程失败

**失败事实**

- 失败 run：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T170948229629Z__46186120`；
- 失败代码提交：`f5c9bbe4c819abce42e1cca0b8800e16a77af680`；
- 正式日志：
  `/root/autodl-tmp/runs/event_first/n1k4_formal_f5c9bbe.log`，SHA256
  `b9ac6d3cce2e731f16aad7bc6a068eaf09c54439ad654bb0a3a9d0c58f63a487`；
- calibration 已运行，进入首个 formal evaluation batch 后，在构造 same-actor 30-frame
  lane-keeping negative control 时抛出 `KeyError: min_median_speed_mps`；
- `lane_keeping_features` 实际接收 `kinematics_control`，但该字段只写在 `cutin` 下；同一调用下一步还会需要
  `max_acceleration_mps2`，原 YAML 也遗漏；
- 失败发生在 `event_pool.json`、`summary.json` 与任何研究裁决写入前。因此它是工程失败，不是机器 gate
  reject，更不是第四次人工评测结论；`n2_authorized=false`。

**保留与修复**

- 旧目录不删除、不改造成成功 run；写入结构化 `failure.json` 与 `FAILED`，原 `RUNNING` 被保留为
  `RUNNING.invalidated`；
- 修复提交 `8581d4dcd1bf9a4f92b426c601e1149c804afc5a` 同时补入
  `kinematics_control.min_median_speed_mps=0.5` 和 `max_acceleration_mps2=12.0`；
- 新增启动前 `_validate_config_contract`，在加载 nuScenes metadata 前检查 delayed runtime dependency、
  `receiver_cutin` 审核 schema 与 `never_start_n2_from_this_run=true`；
- 新增 post-run-directory 异常处理：后续未捕获异常自动写 `FAILED/failure.json`、清除活动
  `RUNNING`，并强制 `n2_authorized=false`；
- 27 项相关测试通过后才从新 run ID 完整重跑。

**防重复**

1. development pilot 必须覆盖至少一个 positive actor 的 negative-control 搜索；“positive=1、
   negative=0”不能被误读为该分支已执行；
2. 所有按候选稀疏触发的配置依赖必须在启动时校验，不能等全量运行数分钟后才由 `KeyError` 暴露；
3. 任何残留 `RUNNING` 的异常目录必须先结构化归档，再开始新 run；禁止覆盖、续跑或统计为 research
   reject；
4. 修复配置/异常落盘不授权改变冻结 K4 阈值、评估 scene 或候选排序。

### N1-F17：重复扫描 583 MB 标注文件产生 cgroup 页缓存压力与外部 SIGKILL

**失败事实**

- 失败 run：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T171746938858Z__5b1634e3`；
- 失败代码提交：`8581d4dcd1bf9a4f92b426c601e1149c804afc5a`；
- 正式日志：
  `/root/autodl-tmp/runs/event_first/n1k4_formal_8581d4d.log`，SHA256
  `7a89e5f5ab88c53a6d9531dedc56a9db302cef8dab144ade7da75a91f3c09191`；
- calibration 与前 96/685 个 evaluation scenes 已执行，随后 shell 报 `Killed`；没有
  `event_pool.json`、`summary.json` 或研究裁决；
- 本层 cgroup `memory.max=2147483648`，事件计数仍为 `oom=0`、`oom_kill=0`，因此不能把信号来源伪写成
  kernel OOM；终态登记为 `external_sigkill_under_cgroup_memory_pressure`；
- `sample_annotation.json` 大小为 `583417244` bytes。失败后进程已消失时，
  `memory.current=1704148992`、file cache=`639545344` bytes；
- 对该只读标注文件执行 `POSIX_FADV_DONTNEED` 后，在没有停止编辑器/Jupyter/TensorBoard 等用户服务的前提下，
  `memory.current` 立即降为 `1169817600`、file cache 降为 `102739968` bytes。

**根因边界**

证据支持以下工程推断：每个 32-scene batch 都顺序扫描 583 MB 标注表，读页长期计入 2 GiB cgroup；
进程 RSS、既有服务和文件页缓存共同逼近硬上限，外部管理层随后发送 SIGKILL。由于无内核日志权限且
`oom_kill=0`，不能声称已证明具体 killer；但页缓存释放的前后差值直接证明了主要可控压力源。

**修复与复验**

- 修复提交 `f13eb0f1e39b608de1c5e698cd678c2dfd8365a4`；
- 所有大型顺序输入在读取前标记 `POSIX_FADV_SEQUENTIAL`，读取后标记
  `POSIX_FADV_DONTNEED`；
- per-scene JSON 改为 `json.load(file_handle)`，不再由 `read_text` 同时常驻字符串和解析对象；
- 每批显式删除 dense scene payload、执行 `gc.collect` 与 glibc `malloc_trim`；
- 每批日志新增 process RSS 与 cgroup current；正式启动若缺 POSIX page-cache control 则 fail closed；
- 30 项相关测试通过。新正式 run 在 96 scenes 的同一死亡点记录
  RSS `602673152`、cgroup current `1707110400` bytes，且继续运行，证明修复覆盖了原路径；
- 成功 run 最终完成 685/685 scenes；最后一批 RSS `510734336`、cgroup current
  `1612763136` bytes，`oom=0`、`oom_kill=0`。它以独立 run ID
  `...T173015103731Z__5b1634e3` 和唯一 `AWAITING_HUMAN_REVIEW` 结束。

**防重复**

1. Python RSS 不是 2 GiB 容器的完整内存分母；必须同时记录 anon、file cache、cgroup current 与
   `memory.events`；
2. 流式解析只限制 Python 对象，不自动释放内核页缓存；反复全表扫描必须有 cache-pressure 策略；
3. 不得以杀死用户服务、跳过正式场景、降低地图分辨率或减少校准标签来换取“成功”；
4. SIGKILL 无法触发 Python exception handler，因此监控器必须把残留 `RUNNING` 另行结构化封存；
5. 该修复只改变 I/O/内存生命周期，不改变 K4 候选、阈值、排序、scene split 或人工门槛。

### 第四版 calibration 冻结结果与禁止矩阵

第四版只用第二、三次全部 49 条人工标签调试阈值，所有 26 个已审 scene 从 formal evaluation 排除。
截至冻结前 development replay：

- 第三次 FP 拒绝 `12/12`；
- 第二次 FP 拒绝 `35/35`；
- 第二次 TP 保留 `1/2`；
- 被保留真例同时满足目标车道中心外→中心内、进入后稳定、进入前近似同向和 RECEIVER 前后身份连续；
- 另一个旧 TP 因没有独立 RECEIVER 的 pre identity support 被拒绝，不用旧 overall 标签覆盖新事件定义。

| 快捷做法 | 为什么无效 | 第四版合法替代 |
|---|---|---|
| 降低分支/多 incoming 门槛 | 仍把地图属性当车辆行为 | 原始 2 Hz center/box outside→inside |
| 复用 subject source-stream rear | 重演 K3 rear 污染 | 独立 direct incoming / target lane RECEIVER |
| 只要求进入后有 rear | 无法证明被切入车流在事件前已存在 | 同一 RECEIVER pre/post identity |
| 因 0.999 s 拒绝名义三帧 1 s | nuScenes 时间戳有毫秒抖动 | 冻结 20 ms timestamp tolerance，仍需 3 个 2 Hz 帧 |
| 在正式 train 结果上再调阈值/scene | evaluation 泄漏 | 阈值只由 49 条旧审标签冻结 |
| 缩短 30-frame negative 或允许 overlap | 改写 matched-control 问题 | physical event window + 0.5 s guard，control 仍 30 frames |
| 自动启动 N2 | 三次 reject 后边界更严格 | 第四次用户裁决 + 新授权前 `n2_authorized=false` |

## N1 kinematics-first 第三版（2026-07-25，已人工 REJECTED）

### N1-F09：候选真实性与 matched-control 支持是两个独立门槛

**观察**

- clean commit `aa162ef4dea808ad28ca7e56f1273f106e9c0e49` 上的 official train 694-scene
  formal run 完成 8,631 transitions → 1,879 topology-pass → 244 physical-motion-pass →
  12 interaction candidates；
- 12 candidates 覆盖 9 scenes，达到 candidate `≥12` 与 scene `≥6`；
- same-actor lane-keeping negative 只有 2，same-actor pair 只有 2，均低于冻结阈值 4；
- 因此 parent `machine_gate_passed=false`；parent 的唯一 terminal 保持
  `AWAITING_HUMAN_REVIEW`，后续独立 adjudication 已按 12/12 FP 写成 `REJECTED`；
- `AWAITING_HUMAN_REVIEW` 只是当时的审计就绪状态，从未表示 machine pass 或 N1/N2 授权。

**Pair 失败的冻结诊断**

对 12 个 positive actor 重放原 30-frame negative 搜索，不改 event pool：

| 主阻塞 | actor 数 | 观察 |
|---|---:|---|
| paired | 2 | 仅 `scene-0870` 两个 actor |
| 无 30-frame stable run | 1 | actor 轨迹支持太短 |
| 所有窗口与 positive overlap | 5 | 4–27 个候选窗口全部重叠 |
| non-overlap lane-keeping 存在，但 interaction 全失败 | 4 | 共 25 个 lane-keeping PASS windows，全部缺 center front/rear |

其中 6/10 未配对 actor 没有可用的非重叠长控制窗口；另外 4/10 没有等价的双侧 interaction control。
这不是把 gap 或速度阈值稍微放宽就能解决的问题。

**禁止快捷修补**

- 不把 30-frame 缩短到刚好得到 4 pairs；
- 不允许 negative 与 positive event overlap；
- 不用不同 actor 冒充 same-actor control，也不只挑有 pair 的两个 actor 报告；
- 不把普通 lane-keeping 但缺 front/rear 的窗口当成与正例等价的 interaction negative；
- 不因人工可能判真而把 `machine_research_support` 改成 true。

**可能突破**

第三次人工已表明 12/12 merge 候选均不真实，因此先修 subject/receiver 语义，再谈 control 扩展。
若第四版人审真实性通过但 same-actor control 仍不足，可新预注册“更长日志中的同 actor control”或
“matched-other-actor control”；后者必须显式匹配 scene、类别、速度、道路与交互密度。二者都不能回写
第三版 run。

### N1-F10：第三版最终候选只覆盖 converging-branch merge

**观察**

- 244 个 physical-motion-pass 包含 181 merge、63 parallel lane change；
- interaction 层有 215 个在中心关键帧缺 front/rear、17 个 temporal identity/bumper-gap 失败；
- 最终 12/12 candidates 全是 `converging_branch_merge`，parallel lane-change 为 0。

**能下的结论**

第三次人审只能估计这 12 个 converging-branch merge 的真实性。即使全部为真，也不能声称第三版已经覆盖
一般 lane change/cut-in；同时也不能断言 63 个 physical lane-change 都是假事件，因为它们是在更严格的
双侧 interaction 层失败。

**复开条件**

先把 `subject maneuver authenticity` 与 `front+rear gap-insertion interaction` 拆成两个预注册层。
可对 63 个 physical lane-change 建独立 diagnostic audit，但不得事后补进当前 12 条、降低当前 machine gate
或把 subject-only event 当 interaction positive。

### N1-F11：完整审核包不等于每项都有前视相机可见证据

**观察**

- 正式包包含 12/12 panels、evidence、topdown、checklist、prompt 和逐文件 SHA256；
- 本机 full train 数据完整覆盖 CAM_FRONT，但其他五个相机目录只有 mini 规模，不能为这些 formal scenes
  提供稳定六相机视图；
- 首尾面板 QA 正常；部分 subject/front/rear 不在 CAM_FRONT 视野，但 2 Hz annotation topdown 仍存在；
- 40 个 immutable audit files 复算 0 hash mismatch；空白 review validator 按预期 fail closed。

**审核边界**

不在 CAM_FRONT 中的角色不能被猜测为 VALID。评审先使用 topdown、vector centerline 和跨时刻 identity；
若相机与 annotation 冲突或证据仍不足，必须判 `UNCERTAIN` 并记录
`INSUFFICIENT_VISUAL_EVIDENCE`。补六相机或 raw LiDAR 需要独立资产/用途授权，不能偷偷进入本轮或 N2。

### N1 第三版历史禁止重试矩阵

| 快捷做法 | 为什么无效 | 合法后续 |
|---|---|---|
| 把 parent `AWAITING_HUMAN_REVIEW` 写成 pass | negative/pair 失败且人工 12/12 FP | 独立 adjudication 已 `REJECTED` |
| 用人工结果覆盖 pair gate | authenticity 与 comparison support 是不同问题 | 两类 gate 均保留 |
| 缩短/重叠 negative window | 事后改变 matched-control 定义 | 新任务、新 split、新预注册 |
| 用其他 actor 补足 same-actor pair | 混入 actor/scene confound | 预注册 matched-other-actor 设计 |
| 把 63 个 physical lane changes 加入 positive | 它们没有通过冻结 interaction | 单独 subject-only diagnostic |
| 用单侧 front 或 rear 算 interaction | 改变“插入双侧 gap”的研究对象 | 另立事件 subtype |
| 没有相机框也猜 TRUE/FALSE | 把证据缺失转成标签 | `UNCERTAIN` |
| 审核后自动启动 N2 | 本 run 明确 `n2_authorized=false` | 新授权 + 新 gate |

## N1 full-domain 第二次 reject（2026-07-25）

### N1-F05：把 target 多 incoming 的地图类别误当成 subject 行为

**观察**

- 父机器 run `N1-EVENT-FULL-01` 在 val 146 上报 37 个 positive，其中 topology 为 35 merge + 2 lane change；
- 完成人审文件 SHA256 为
  `ae71b31e02faf1d783c36748e629e85acf32a132f35ce2f98102a5f62201dd05`；
- 用户确认的逐项结果为 `TRUE_POSITIVE=2`、`FALSE_POSITIVE=35`、`UNCERTAIN=0`，机器候选精度
  `2/37=0.054054`；
- 多数 reviewer notes 明确指出：subject 沿与 target 共线的主路 lane/connector 正常直行，真正汇入
  target 的是另一条 incoming branch；旧规则却只因 `target_incoming_count>=2` 就把 subject 标为 merge；
- 独立 audit adjudication：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-FULL-AUDIT-01/v71_n1-event-full-audit-01__human-audit-reject-v1__s0__20260725T083929632491Z__6507cbac/`，
  唯一终态 `REJECTED`。

**能下的结论**

graph-corridor 修复了邻车跨 token fragmentation，却没有证明 subject 本身执行了 lateral maneuver。
“target 有多个 incoming”是地图节点属性，不是 actor-specific merge 证据；第二次 N1 不能进入 N2。

**不能下的结论**

不能据此断言 full nuScenes 没有真实 lane change/merge，也不能把 2 个标注 TP 当成已验证的完整事件池。
旧 audit panel 没有把 subject/front/rear 的 3D identity 投影到图像，且 reviewer 字段包含多个来源；
用户已整体确认 reject，但单条 TP 仍只能作为第三版 calibration 标签，不得直接进入 formal evaluation。

### N1-F06：10 Hz 插值 cadence 被错误提升为物理证据

**观察**

nuScenes `sample` 是 2 Hz 标注关键帧。第二版把 2 Hz box 线性/SLERP 插值到 10 Hz 后，用连续 lane-token run
寻找 transition；该做法对齐了 DriveStudio cadence，却没有产生新的物理观测。第二次人审 notes 多次指出
短 token 切换、轨迹插值或 map assignment 假象。

**防重复**

- 第三版速度、加速度、yaw-rate、lane preference、front/rear persistence 只能用原始 2 Hz keyframe；
- 10 Hz 只用于 frame 对齐、可视化和复现旧 transition 候选，不得计算导数或宣称 0.1 s 观测；
- 至少 3 个 pre 和 3 个 post keyframes；不足时为 `UNKNOWN`，不得靠插值补齐。

### N1-F07：单时刻中心距不是持续物理交互

**观察**

第二版在单一 relation frame 上用中心线 `s` 与中心距 `[2,60] m` 选择 front/rear，没有扣除 box extent，
也不要求同一 front/rear identity 跨时刻持续。37 个 machine positive 中 36 个至少依赖一个跨-token 邻车，
因此 branch 选择错误会直接翻转结果。

**第三版解除条件**

1. target corridor 每个 graph edge 同时满足方向连续和 endpoint 连续，并只选单一最连续分支；
2. 使用 oriented box 在 lane tangent 上的投影半长，报告 bumper gap 与 center gap；
3. 至少 2/3 个连续 2 Hz keyframe 保持同一 front/rear identity、方向和次序；
4. 同时报告 longitudinal speed、closing speed、headway/TTC；它们是诊断，不得替代人审。

### N1-F08：第二版审核合同与 provenance 不足

**观察**

- 父机器 run 诚实记录 `code_dirty=true`；它可定位但不是 clean-commit formal baseline；
- 旧人审清单给了逐项 verdict 定义，却未预注册聚合阈值；
- 因此第二次 reject adjudication 没有查看结果后补造阈值，只登记用户明确决定；
- 旧 CAM_FRONT 清单没有身份 box overlay，容易把画面中“真正并道的另一辆车”认成 subject。

**复开条件**

第三版必须在 clean commit 上运行；正式 audit pack 同时提供盲序、subject/front/rear 颜色框、2 Hz 俯视轨迹、
逐项 component verdict、failure codes、完整提示词、immutable file hashes、预注册统计阈值和独立
adjudication 命令。Agent 不得填写人工 verdict。

### N1 full-domain 禁止重试矩阵

| 快捷做法 | 为什么无效 | 第三版允许替代 |
|---|---|---|
| 继续调 `graph_hops` 或 gap | 35/37 误报的主体事件本身不成立 | actor-specific 2 Hz kinematics 先行 |
| target 有多个 incoming 就叫 merge | 把地图节点类别当行为 | 比较 source 与主路 incoming 的 approach geometry |
| 用已审 val 37 条挑最终阈值并在同一 split 报结果 | calibration/evaluation 泄漏 | val 只 calibration；official train formal evaluation |
| 从 10 Hz 插值计算速度/横移 | 人造高频证据 | 原始 sample timestamp + 2 Hz boxes |
| 单帧 front/rear 中心距 | branch/identity 易跳变，忽略车长 | branch-safe corridor + temporal identity + bumper gap |
| 把 2 个旧 TP 直接当第三版正例 | panel identity 仍有未决风险 | 只作 calibration；第三版候选重新盲审 |
| 机器候选一出现就启动 N2 | 人工真实性与样本支持尚未通过 | `AWAITING_HUMAN_REVIEW`，`n2_authorized=false` |

## N1 mini event-pool reject（2026-07-24）

### N1-F01：interaction-support failure

**观察**

- N0 map-expansion、scene→map 与 pose contract 已通过，不再是资产缺失；
- 45 个 source-only eligible actors 产生 71 个 stable token transitions；
- topology taxonomy：39 route continuations、19 merges、3 lane changes、10 unresolved；
- 19 merges + 3 lane changes 共 22 个 topology-pass candidates；
- 22/22 的 exact-target-token front/rear relation 为 FAIL；
- 18 个没有 target-token 邻车，4 个只有 front、没有 rear；0 个同时满足 2–60 m front/rear；
- positive=0、negative pairing=0、same-actor pair=0、positive scenes=0，唯一终态 `REJECTED`。

**能下的结论**

冻结 mini split 不支持可比较 interaction event pool，N2–N5 不触发。地图缺失不是旧 H1 的唯一根因；
补地图后 mini interaction support 仍为零。

**不能下的结论**

不能写成“人类绝对看不到任何交互”或“full nuScenes 也没有事件”。exact target token 可能把同一
longitudinal corridor 上的 actor 分到相邻 lane/connector token；该表示风险尚未独立校准。

**复开条件**

mini run 不复开。新的路线必须：

1. 使用不同 run/task ID；
2. 以 22 topology-pass mini cases 仅作 calibration/audit，不作 formal evaluation；
3. 在 graph corridor 上定义 route-aligned curvilinear front/rear，而非后验放宽欧氏半径；
4. calibration 与 evaluation scenes 分离；
5. 优先在 full nuScenes trainval annotations/metadata 上冻结并评估。

### N1-F02：exact-token corridor fragmentation

**观察**

71 transitions 中 39 个只是 directed route continuation，说明官方 lane graph 将连续道路划分为多个
lane/lane_connector token。当前 interaction 只接受 relation frame 上与 subject 完全相同的 target token。

**推断**

该规则高精度但可能低 recall，尤其在 lane→connector→lane 或短 lane segment 附近。它是 0 interaction
PASS 的一个可能贡献因素，但不是已证实的唯一原因；mini 本身也可能确实缺少前后车。

**禁止快捷修补**

- 不把“相邻 token”全部并入；
- 不把只有 front 或只有 rear 改成 positive；
- 不把 82–89 m front 后验纳入 60 m；
- 不在同一 22 cases 上调 graph hops、gap 或 heading 直到出现 positive。

允许的修复是先定义有向 corridor、route-aligned `s` 和 branch disambiguation，再由独立 calibration
审计冻结；formal evaluation 必须 scene-disjoint。

### N1-F03：mini scale 与静止对象密度

**观察**

- 003/005/004 eligible actors 为 7/22/16；
- 因首尾位移不足 5 m 被拒的 actor 为 107/17/5；
- eligible pose map-match coverage 为 88.89% / 95.60% / 93.36%；
- 官方 full nuScenes 有 1,000 个约 20 秒 scenes，850 个为 train/val，而当前 formal pool 只有 3 scenes。

**结论**

mini 三场景对多 scene interaction event pool 的统计支持不足。下一步应扩数据底座，不应换 actor 或删场景。
优先同域 `v1.0-trainval` annotations/metadata，只有其 event gate 仍失败才评估 nuPlan/Waymo。

### N1-F04：negative=0 的语义

N1 只为已经有 positive 的 actor 构造 same-actor comparable negative。因此 `negative=0` 是
`positive actor set=∅` 的结构结果，不证明没有稳定非事件窗口。后续报告必须同时给出 positive actor 分母，
不得把 negative=0 解释为数据中全是事件或完全无普通驾驶。

### N1 禁止重试矩阵

| 快捷做法 | 为什么无效 | 允许替代 |
|---|---|---|
| 删除 rear requirement | 改变冻结 interaction claim | corridor calibration + scene-disjoint evaluation |
| 扩大 60 m 到覆盖 82–89 m | 看结果后调阈值 | 在新 calibration pool 依据任务时间窗冻结 |
| exact token 改成任意相邻 token | 可能跨 branch/对向车道误配 | directed corridor + route-aligned `s` |
| 从 22 cases 挑“看起来像”的 positive | 人工/后验标签泄漏 | 完整盲审协议；calibration 不进入 eval |
| 在 005 单 scene 继续 | 删除失败 scene、失去多 scene gate | full trainval scene-disjoint split |
| 直接启动 N2/N3/render | 没有 comparable event | 新 N1 先通过 |

## 0. H1 reject 执行摘要

### 0.1 为什么 reject

| ID | 层级 | 观察到的事实 | 能下的结论 | 不能下的结论 |
|---|---|---|---|---|
| `H1-F01` | 事件存在性 | 30 proposals：0 positive、25 negative、5 source-positive/non-event、0 same-actor pair | 冻结 proposal bank 不支持 H3 或配对因果比较 | “occupancy 一定无效”或“换几个 actor 就会成功” |
| `H1-F02` | certificate 精度 | D1 TP=15、FP=5、precision=0.75 < 0.80 | H1-CERT 按预注册 reject | 仅因 recall=0.8824 就称 certificate 通过 |
| `H1-F03` | certificate 覆盖 | D1 UNKNOWN=10/30、PASS=0、PASS coverage=0 | 当前证据无法给出足够确定的正判定 | 把 UNKNOWN 排除或并入 PASS 后重算 |
| `H1-F04` | repair 吞吐 | D2 reject=30/30、export=0、usable yield=0 | H1-PROJ 按预注册 reject；外部 rate 不可定义 | “导出集 0/0 违规，所以修复完美” |
| `H1-F05` | 数据效用 | 无 positive pair，H1 已拒绝 | H3 不触发 | 以 RGB 差分、accept rate 或 proxy 代替下游任务 |
| `H1-F06` | 高成本阶段 | H1 前置 gate 失败 | H2/render audit/blind pack 不实例化是正确停止 | “没跑 H2，所以 H1 结论不完整” |
| `H1-F07` | 统计实现 | 首版 aggregate 把 rejection 计成零违例 | 聚合 bug 已修复且不影响方法输出 | 用首版 aggregate 支持方法 claim |
| `H1-F08` | 资产/证据 | 本机 map 只有 raster PNG；base UNKNOWN 约 96–98% | lane/road support 与独立覆盖存在硬缺口 | 从 raster 或 learned occupancy 静默补成真值 |

### 0.2 冻结证据

- 正式 run：
  `/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11D/v71_v7-h1-11d__pilot-3-matched__s0__20260723T155755269940Z__cf8d5ebc/`；
- proposal-bank SHA256：
  `f8986915f8d2be0cddddfa6be86f4d2d1ece456c12bf9a962cafec78fd058cd7`；
- config SHA256：
  `cf8d5ebc1429e076fc5142aa6a759a18f54b7f3f937c8423d51505a094bc9fe3`；
- C/D1 realized trajectory 30/30 identity；
- C external hard violation 17/30：003=5/10、005=7/10、004=5/10；
- D1：15 TP、5 FP、2 FN（含 abstention）、20 FAIL、10 UNKNOWN、0 PASS；
- D2：0 accept/export、0 usable yield；
- 唯一 terminal marker：`REJECTED`。

### 0.3 逐失败点根因、卡点与复开要求

#### `H1-F01`：proposal-support failure

**观察**

- source-only eligibility 固定为 3 scenes × 2 actors；
- 每 actor 固定 P1–P5，共 30 个 proposal；
- scenario-effect 没有产生任何 `0→1` positive；
- 5 个 source-positive case 在 proposal 后成为 non-event，25 个为 negative；
- 没有 same-actor positive/negative pair。
- 后续只读 continuity 审计发现，冻结 actor 003:38、003:35、005:23 在完整连续 track 内的 world
  displacement 仅 0.88 / 0.29 / 0.76 m；原 source-only 排序偏好长、清晰 track，但没有事件相关性。

**推断**

固定横向位移只满足几何“移动过”，没有以 lane topology、corridor crossing、target-lane front/rear gap、
duration 或 interaction 定义事件。该设计与“cut-in/merge 正例”的目标错位。这是由结果和 schema 支持的
最强解释，但尚未通过 vector map 重标注，所以不能断言每个 case 的唯一失败原因。

**卡点**

- 本机缺 nuScenes map-expansion vector JSON；
- mini 三场景的真实事件上限未知；
- 没有先冻结 natural-event pool，就不能知道是 proposal family 失败还是场景本身无事件。

**复开条件**

不是改 P1–P5。必须新建 event-first 路线：先冻结 map/track 事件定义和 actor pool，证明存在预定数量的
positive/negative 与 same-actor pair，然后才允许生成候选。若 mini 事件池不足，应 reject mini pool 或
请求新数据授权。

#### `H1-F02`：certificate precision failure

**观察**

5 个 FP 全来自 scene 004 actor 8；certificate 报告 5 个 static-overlap voxels，而独立 raw LiDAR
检查为 0 points。D1 precision 为 0.75，低于冻结门槛 0.80。

**推断**

结果与 coarse voxel quantization、box-to-voxel 接触或证据层不一致相符；尚不能证明是哪一个机制，也不能
从“0 raw points”推出空间一定安全，因为 LiDAR 可能受遮挡和采样稀疏影响。

**卡点**

- `0.4m` 离散 grid 将连续几何压成二值接触；
- 单一 voxel overlap 缺少距离、置信度和观测支持；
- static/dynamic 分层仍可能受历史 sweep 和运动补偿影响；
- raw point absence 不是 free-space ground truth。

**复开条件**

在 scene-disjoint calibration pool 上比较 coarse voxel 与 motion-compensated raw sweeps 的连续
point-to-OBB/swept-volume distance；逐类报告量化、动态残影、遮挡、地图边界和标注误差。门槛必须在
冻结评估前预注册，不能用 actor 004:8 调到通过。

#### `H1-F03`：coverage/abstention failure

**观察**

三场景 base unknown 约为 97.10% / 96.04% / 97.57%；D1 10/30 UNKNOWN，PASS coverage 为 0。
两个 FN 位于 005 的 P3/P5，D1 known fraction 为 0，而 raw LiDAR 只有 3/2 points。

**推断**

当前 single/coarse observation 无法支持大部分 free/occupied 判定。两个 FN 说明“极少 raw points”
也不能自动解决判定；具体是遮挡、采样、时序或标注问题仍未知。

**卡点**

- raw LiDAR 稀疏；
- 缺 vector drivable/lane polygons；
- 多 sweep 若不做动态/ego motion compensation 会制造 ghost；
- learned completion 会提高表面 coverage，却失去独立真值身份。

**复开条件**

增加独立 evidence，而不是调低 known-fraction：官方 vector map、ego/dynamic compensated sweeps、
显式 truth tier 与 uncertainty。继续报告 PASS/FAIL coverage 和 abstention；任何 learned occupancy
只能是附加证据层，不能作为外部 evaluator。

#### `H1-F04`：repair all-reject failure

**观察**

D2 没有接受或导出任何 proposal；usable yield=0，外部 violation rate 无分母。

**推断**

当前 projection/repair 约束组合没有可用工作区，或者 proposal 全都离可行域过远。因为 0 export，无法
区分“repair 算法差”与“输入候选全不可修复”各自贡献。

**卡点**

- 没有成功样本用于 paired outcome；
- 先验 proposal 不由 lane-reachable set 生成；
- 二值 certificate 既可能过严又可能不准；
- H2/H3 都依赖 D2 产出，故被同时锁死。

**复开条件**

先通过 N1 证明事件存在，再以 lane graph/target state 生成 reachable proposal；冻结 minimum usable yield、
comparable export 数和外部 evaluator。若仍为 all-reject，直接 reject proposal/repair family。

#### `H1-F05`：metric aggregation bug

**观察**

首版 summary 把 rejection 计为零违例，使 0 export 看起来像 0% external violation。唯一允许的
`metric_aggregation_bug` 修复保留了旧 aggregate；修复后无 export 时 fail closed。修复提交为
`b82c540`，不改变 proposal、trajectory、certificate 或 D2 输出。

**防重复**

- 所有 rate 必须同时报告 numerator、denominator、rejected、unknown；
- denominator=0 时写 `undefined`，不能写 0；
- terminal decision 必须读取 comparable export 和 usable yield；
- 原始 aggregate 不覆盖，修复生成新版本并记录 migration。

#### `H1-F06`：地图资产与证据缺口

**观察**

`/root/autodl-tmp/data/nuscenes/maps/` 只有 4 个 PNG，没有 vector JSON；本机没有 Waymo/nuPlan 数据。
`/root/autodl-tmp` 约有 65G 可用空间。

**卡点**

官方 lane graph/drivable polygon 暂不可查询；不能可靠地定义 target lane、connectivity、off-road 或
corridor crossing。DriveStudio adapter 代码的存在不等于数据和许可就绪。

**复开条件**

先生成最小资产清单并取得下载授权；保存来源、许可、大小、SHA256 和 scene→map 映射。不得从 raster PNG
反推正式 lane graph，也不得静默下载全量 Waymo/nuPlan。

### 0.4 禁止重试矩阵

| 快捷做法 | 为什么无效 | 允许的替代 |
|---|---|---|
| 降 known-fraction / coverage | 把无证据改名为有证据 | 增加独立 map/raw evidence |
| UNKNOWN 并入 PASS/FAIL | 改变预注册语义和分母 | 继续三态并单列 coverage |
| 删除 S1、005 或 004 actor 8 | 后验删难例 | scene-disjoint 新 pool |
| 换 actor、方向、P1–P5 幅度 | 用结果挑 proposal | 先冻结 event definition 与 actor pool |
| 0 export 报 0 violation | denominator=0 | 报 undefined + yield=0 |
| multi-sweep 直接堆叠 | 动态物体会 ghost | ego/dynamic motion compensation |
| learned occupancy 当 GT | 方法与 evaluator 循环 | raw/map 独立 evaluator + calibration |
| GS floaters/画质当安全证据 | renderer 不是物理传感器 | GS 只在 N4 导出 |
| 先做 H2/H3/scale | 没有 comparable positive | N1–N3 先过门 |
| 重命名 OccGS 复开 | 没有解除原失败 | 新路线必须满足复开条件 |

### 0.5 可复用资产

失败不否定以下工程资产：

- coordinate contract、`WorldState`、typed label/depth；
- run contract、artifact index、terminal marker 和 fail-closed aggregate；
- object-centric GS reconstruction/renderer；
- D1/D2 接口和 `PASS/FAIL/UNKNOWN` schema；
- 冻结 H1 bank 作为负对照与回归 fixture。

复用这些资产不能继承 H1 claim；新路线必须有新 preregistration、独立 event pool 和 evaluator。

## 1. 仍直接约束 V7 的历史结论

| ID | 状态 | 对 V7 的约束 |
|---|---|---|
| `RF-05` | rejected | 合法轨迹/点或局部像素变化不等于 RGB、遮挡、source removal、depth、identity 与标签都合法 |
| `RF-06` | rejected | 局部 loss 或 mask 不保证参数/输出只在局部改变；必须测 outside、boundary、frame-0 与 held-out |
| `RF-08` | limitation | 可复现的机器 evaluator 不等于绝对物理真值，更不能替代人工 verdict |
| `RF-09` | rejected | same-scene、shared identity 或结构合法不等于人类能辨别方法收益 |
| `RF-16` | limitation | layout/trajectory controllability 不等于 action-disentangled actor physics 或数据效用 |
| `RF-18` | rejected | ReSim `exp0_no_carla` 的 E-vs-F action response 不足；V7 不得借归档重开 C1P/C1S |

其他 RF 仍完整有效，但当前 OccGS 计划不直接复用对应的 SVD projection/preference 配方。

## 2. V7 风险索引

| ID | 状态 | 风险 | 禁止的快捷修补 |
|---|---|---|---|
| `V7-RISK-01` | rejected_v71 | occupancy 已接入 11D，但 certificate precision 与 repair yield 均未过预注册 gate | 因为 occupancy 文件存在或 D2 无 export 就宣称 H1 通过 |
| `V7-RISK-02` | limitation | C0 24/24 是按效应 top-k 的机器筛选，不是用户人工评测 | 写成 human pass，或只报 top-k 隐藏 46/62 全分布 |
| `V7-RISK-03` | open_risk | L0 mask 来自 RGB 差分，outside=0 由 hard composition 构造保证 | 用 0 leakage 宣称 occupancy-guided completion 有质量收益 |
| `V7-RISK-04` | open_risk | U0 以极端 V4 为 naive 对照且没有下游任务 | 把 accept rate / RGB signal 写成优于 naive GS 或 mAP 收益 |
| `V7-RISK-05` | legacy_limitation | V7 既有 run 缺正式 manifest、resolved config 与终态标记；V7.1 新 run 已由 EV-10 fail closed | 事后猜 seed/fingerprint 或伪造 immutable provenance |
| `V7-RISK-06` | open_risk | 只覆盖 mini 三场景，S1 held-out 质量偏弱 | 先扩规模、只筛容易场景或把三场景外推为论文结论 |
| `V7-RISK-07` | interface_mitigated_v71 | 11C 已闭合 WorldState→renderer→typed-label 工程链；occupancy repair 的方法增益仍未验证 | 把 label-sync 工程通过写成 occupancy certificate/projection 通过 |
| `V7-RISK-08` | legacy_risk_mitigated_v71 | O0 坐标注释、metadata 与实际变换含义不一致；11A 已冻结显式 frame 合同 | 沿用含义不明的 `pose/T`，或在 round-trip 前计算 H1 指标 |
| `V7-RISK-09` | confirmed_mitigated_v71 | 旧 rotated-corner AABB 使 PILOT-3 动态体素量膨胀 1.72–2.83 倍；扁平语义不能诚实移除 actor | 把旧 O0 AABB 当正式安全几何，或移除 actor 后把体积恢复为 free |
| `V7-RISK-10` | confirmed_failure_v71 | 高 UNKNOWN 在 11D 导致 10/30 D1 abstain、D2 30/30 拒绝与 0 usable yield | 把 UNKNOWN 并入 PASS/FAIL，或降低观测门槛追求 yield |
| `V7-RISK-15` | architecture_mitigated_v71 | certificate detection 与 trajectory projection 若混组会混淆检测和修复收益 | D1 修改 C trajectory，或把 D1/D2 合成单一 validity 数字 |
| `V7-RISK-16` | confirmed_failure_v71 | 冻结 30-proposal bank 得到 0 个 0→1 positive 和 0 个 same-actor pair | 用位移幅度或 RGB 差分代替 scenario-effect gate，或事后换 actor |
| `V7-RISK-17` | confirmed_mitigated_v71 | 单一 `depth` 名称会混淆 expected、first-hit 与 LiDAR measured truth tier；11C 已强制分名和 sidecar | 把 expected depth 登记为 measured GT，或省略 validity/truth-tier |

## 3. 风险详情与解除条件

### V7-RISK-01：occupancy 尚未进入方法

**观察**

- `occupancy/build_scene_occupancy.py` 独立写出 per-frame grid；
- `resim/s0_trajectory_editor.py` 只检查横向运动学、yaw、actor/ego 距离和粗横向范围；
- `resim/c0_counterfactual_render.py` 改写 RigidNodes pose，但没有查询 occupancy；
- `resim/l0_local_completion.py` 用 V0/edited RGB 差分构 mask。

**边界**

O0 是有用的世界状态基础设施，但当前不能支持“occupancy 提高合法性”或“occupancy-guided completion”主张。

**解除条件**

按 `V7-H1-11` 建立统一 actor/state mapping，让 occupancy 进入 edit certificate、visibility 与标签重生，并对
matched kinematic-only/naive baselines 做非循环消融。只添加一次 occupancy lookup 或 post-hoc filter 不足以解除。

### V7-RISK-02：机器 top-k 不等于人工合法率

**观察**

- C0 全部可见 case 为 46/62 machine legal；
- 24/24 是按 mean edit effect 排序后的 top-24；
- 当前 `reviews/` 目录是机器面板与机器 JSON，没有用户填写的 verdict。

**边界**

可表述为“机器筛选 top-24 均满足当前规则”，不得表述为“24/24 人工合法”或用其估计全候选分布。

**解除条件**

先冻结 blind sample、逐项 rubric、失败优先级、JSONL schema 与聚合阈值，再由用户或指定评审者完成 verdict。
agent 不代填，也不以机器字段映射成人工答案。

### V7-RISK-03：L0 primary metric 目前是构造不变量

**观察**

hard composition 直接复制 mask 外的 edited GS，因此 outside-mask L1 必然为 0；当前 12 帧结果只验证实现遵守
公式。mask 由 RGB 差分阈值和膨胀获得，不包含 ray visibility、unknown/free 或 source footprint geometry。

**边界**

L0 只证明 local composition 工程可行。没有证据表明 Telea 改善视觉、时序、depth 或 identity。

**解除条件**

使用 geometry-derived disocclusion mask，并在有真值的 pseudo-hole 上比较 no completion、Telea 与局部生成；
primary 必须包含 inside quality、boundary、temporal、depth/instance，而不是继续调阈值追 outside=0。

### V7-RISK-04：U0 proxy 不识别数据效用

**观察**

`naive_V4` 是约 39–50 m 的强制横移负例；它被拒绝只能证明 validator 能识别一个极端错误。当前没有训练
detector、occupancy model 或 event classifier，JSON 明确记录 `u0_full_map_pass=false`。

**边界**

不能声称 OccGS 优于 matched naive GS、real-only 或提供下游增益。

**解除条件**

对相同 proposal、相同样本量和相同训练预算比较 R / R+naive / R+OccGS / R+OccGS+completion，并使用
scene-disjoint split、至少 3 seeds 和任务指标。三场景只可用于 pipeline smoke。

### V7-RISK-05：既有 run provenance 不完整

**观察**

`runs/occgs_resim/` 现有 B0/C0/L0/U0 目录未发现 `manifest.json`、`resolved.yaml` 或终态标记。B0 仍有
`config.yaml`、metrics、checkpoint；其他阶段有 JSON 报告，但不足以满足正式 run contract。

**边界**

现有数值可作为 retrospective evidence，不能声称是完整、不可变、可从 manifest 一键复现的正式 run。

**解除条件**

`V7-EV-10` 为既有证据生成显式缺失项索引；所有新 run 通过 fail-closed wrapper 产生完整协议。禁止事后补造
未知字段或覆盖旧目录。

**2026-07-23 缓解结果**

- `V7_EVIDENCE_INDEX.json` 已逐文件索引 B0/O0/S0/C0/L0/U0 的 1,610 个文件，并保留正式字段的
  `missing/unknown_not_inferred`；
- V7.1 run contract 对 run ID 复用、三层 hash、artifact bytes、summary、冲突终态标记和 optional
  `not_triggered` 分支 fail closed；
- 正式 smoke 在 commit `3590558` 上以唯一 `COMPLETE` 结束，25 项相关测试通过。

该缓解只约束 V7.1 新 run；V7 旧 run 的 provenance 缺口不可逆，仍保持 retrospective/legacy limitation。

### V7-RISK-06：场景覆盖与质量

**观察**

本机只有 mini 10 scenes 具备前向完整 sweep；feasibility 只使用 3 scenes。S1 test PSNR/SSIM 为 20.18/0.472，
明显弱于 S0/S2。

**边界**

当前结果不能外推到 trainval、长时、多相机、夜间或复杂交互；也不能只删掉 S1 后报告更好均值。

**解除条件**

H1 先在冻结三场景与 worst-case 上通过，再审计可获得的 scene-disjoint 数据。扩展必须保留困难场景分层、
真实/插值 provenance 与相同门禁。

### V7-RISK-07：标签链未闭环

**观察**

C0 已改写 RigidNodes pose 并输出 RGB/depth/rigid 分量，但尚未形成统一的 semantic、instance、2D/3D box、
occupancy 与 visibility regeneration 流水线。

**边界**

“label synchronization”当前只可称 proxy/interface 可行，不是完整传感器与标签一致性。

**解除条件**

同一 world-state record 驱动 renderer 与所有标签 writer，逐帧验证 pose、depth、mask、box 和 occupancy 共位；
对缺失/不可见标签 fail closed。

**2026-07-23 缓解结果**

- 11C 在 PILOT-3 的 V0/V1、三场景、三前向相机上生成 18 个样本和 432 个 typed sidecar；
- 独立审计验证 18/18 样本、6/6 WorldState hash、temporal identity、三相机覆盖、instance-depth z-order 与
  state-specific safety/observation/render-support 引用；
- expected、first-hit、LiDAR measured depth 分名，有限 semantic scope 和 visibility provenance 均写入 sidecar；
- S1 保留，正式 run 以唯一 `COMPLETE` 结束。

该结果只解除 renderer/label 工程接口风险；11D 之前仍不能声称 occupancy certificate 或 repair 有方法收益。

### V7-RISK-08：O0 坐标框架歧义已确认

**观察**

- `occupancy/build_scene_occupancy.py` 文件头将 grid 描述为首帧 ego-centric；
- `meta.json` 将同一产物描述为 per-frame ego-centric；
- 实际实现每帧读取 `lidar_pose/{t}.txt`，以其逆矩阵把 world box 变换到 grid，同时直接使用 sensor-local
  LiDAR 点。因此产物实际是 per-frame LiDAR-sensor grid，而不是首帧固定 grid，也不能在未审计 LiDAR-to-ego
  外参前简称 ego frame；
- DriveStudio 则以起始 `CAM_FRONT` 的 `camera_to_world` 逆矩阵定义 model frame。

**边界**

现有 O0 数值仍可作为 coarse retrospective evidence，但在显式记录 `T_grid_world`、`T_model_world`、
`T_world_camera` 并通过 world→model/grid→world round trip 前，不得用于 H1 合法性指标。

**解除条件**

`V7-H1-11A` 统一使用 `T_dst_src` 命名，修正新 schema/adapter 的 frame 声明，以 synthetic fixtures 和
PILOT-3 原始标定验证 translation、yaw、box corners、camera projection 及 checkpoint pose round trip。
旧 O0 文件不原地改写；正式 H1 evidence 产生新版本与新 fingerprint。

**2026-07-23 缓解结果**

- 11A 将 annotation/model/grid/camera/LiDAR frame 分别冻结为 world、start-CAM_FRONT、per-frame-LiDAR、
  `T_world_camera` 与 `T_world_lidar`；
- 三场景 1,679 个 actor poses 的 translation、rotation、box 和三前向相机投影 round-trip gate 通过；
- registry 跨独立进程重建 hash 完全一致，正式 run 以唯一 `COMPLETE` 结束。

旧 O0 metadata 不原地改写，故该风险仍是 retrospective artifact 的 legacy limitation；V7.1 后续模块必须引用
11A coordinate contract 和新 fingerprint。

### V7-RISK-09/10：AABB 膨胀与高 UNKNOWN 已确认

**观察**

- 在完全相同的 PILOT-3 raw annotation、grid 和 240 帧上，旧 rotated-corner AABB 相对 oriented-box
  center-inclusion 的动态体素量比分别为 003 `1.721×`、005 `2.249×`、004 `2.833×`；
- 分离 dynamic instance layers 后，base unknown 比例仍为 `97.10% / 96.04% / 97.57%`；
- source actor removal 后原体积恢复 UNKNOWN，不会恢复 FREE；edited layer 可独立 remove/insert，三场景未出现
  layer overlap；
- 缺少 nuScenes map-expansion polygons 时 road-support 与 off-road control 保持 UNKNOWN。

**边界**

11B 已消除 AABB 作为正式动态几何和扁平 layer 删除污染，但没有降低 observation sparsity。30 条可测真实
controls 的 retention 为 100%，collision/teleport 可检测负例为 2/2；然而加入 road-support 后 32 条完整
certificate 全为 UNKNOWN。这是诚实 abstention，不是 H1-CERT pass。

**后续约束**

D1 必须报告 precision、recall、abstention 和 PASS coverage；UNKNOWN 不进入 TP/FP/FN。只有独立观测或 map
证据能把 UNKNOWN 变为可判定状态，禁止通过调大 unknown threshold、把 box 当 background surface 或用 Gaussian
floaters 补 safety evidence。

### V7-RISK-15/16：certificate/projector 与 scenario effect 必须继续拆分

11B 已冻结 `scenario-effect-v1` 的纯 3D 0→1/0→0 gate、same-actor pair schema 和
`certificate-calibration-v1` 三态接口。11D 必须让 D1 逐字节复用 C trajectory，D2 才允许修改轨迹；位移 proposal
若未形成冻结的 corridor crossing、duration、gap 与 TTC/headway 条件，只能标为 non-event，不能靠命名成为
cut-in/merge positive。

### V7-RISK-17：typed depth 语义混淆已缓解

11C 把 depth 冻结为三个不同产品：diagnostic expected depth、T1 Gaussian first-hit depth、T0 LiDAR measured
depth；每个产品有独立 validity、definition、truth tier 与 artifact sidecar。独立审计确认三类各 18 个，且没有
expected-as-measured 混写。后续 export/evaluator 必须继续按产品名和 truth tier 消费，不能重新折叠成无类型
`depth`。

### V7-H1-11D：H1-CERT / H1-PROJ 预注册拒绝

**冻结事实**

- source-only eligibility 覆盖 3 scenes × 2 actors，P1–P5 共 30 proposals；S1 未删除；
- C/D1 realized trajectory hash 30/30 完全相同；
- D1：precision `0.75`、recall `0.8824`、abstention `0.3333`、PASS coverage `0`；
- C external hard violation `17/30`；
- D2：0/30 export、0 usable yield，external rate 不可定义；
- scenario-effect：0 positive、25 negative、5 source-positive/non-event，0 same-actor pair。

**裁决**

H1-CERT 因 precision 低于 `0.80` 拒绝；H1-PROJ 因拒绝全部 proposal、无 comparable export、usable yield
低于 `70%` 拒绝。按路线转向规则停止 OccGS 方法 claim，只保留 object-centric GS、WorldState、typed label、
certificate/evaluator 与 run-contract 基础设施。

**唯一修复与防重复**

首版聚合把 rejection 计成零违规，已作为 `metric_aggregation_bug` 唯一修复，旧 aggregate 保留。修复未改变
方法输出；第二版对无 export 的 rate fail closed。不得继续：

- 调低 known-evidence/coverage 门槛把 UNKNOWN 改成 PASS；
- 删除 005/S1 或 004 actor 8；
- 根据现有结果重选 actor、方向、proposal 或 event threshold；
- 用固定-pool `0/30 violation` 隐藏 D2 的 `30/30 reject`；
- 因 recall 达标而隐藏 precision fail，或把 UNKNOWN 排除后重算；
- 在当前配方上继续 H2/H3/scale。

## N1 receiver-centric cut-in final：第四轮后新增防重复项（2026-07-26）

### N1-F18：receiver branch merge 的 13/13 历史假阳性不能靠阈值微调挽回

第四版旧 parent 的 18 个 machine candidates 中有 13 个 receiver-branch merge；第四轮人工裁决表明这类
历史 branch-merge 候选均为 `FALSE_POSITIVE`。因此 branch topology、`target_incoming_count`、shared successor
或 token change 永远不能单独证明 cut-in。final v2 把该类别固定为
`ABSTAIN/UNSUPPORTED_BRANCH_MERGE_MODE`；不得为了候选数量重新放宽它。

### N1-F19：support-count 不能替代完整 receiver identity 时序

K4-012 暴露了“support 数量足够”仍可能跨 raw 帧切换接收车身份的问题。legacy fixture 的 `1→38` 与 v2
raw map 重放中观测到的 `9→1→9` 是不同窗口/枚举证据，二者都不能被静默等同为连续 receiver。final v2
要求 required raw frame 全窗唯一 non-null identity、last-post anchor 和每帧 rank/gap/path-clear；任一身份
切换必须 FAIL 或 ABSTAIN，不能被总 support count 抵消。

### N1-F20：弯道 map jitter 的 post heading 不能借由宽松窗口穿透

K4-015 证明 source/target 局部不平行或 post heading 过大时，几何横向收敛可以伪装成切入。final v2 使用
local parallel overlap、raw post-heading、累计 yaw 和 raw-only provenance；它不是针对 scene/token 的黑名单。
禁止为了保留 K4-015 或增加 PASS 数而放宽这些 hard gate。

### N1-F21：CAM_FRONT 的五帧截图不能承担角色/时序的完整证明

单相机可见性和五帧页面截断无法可靠展示 SUBJECT、RECEIVER、source/target corridor 的完整 raw 窗口。
审核 V2 因而以逐 raw-frame topdown、2 Hz signals、actor-ID switch 标注和固定 camera-unavailable 警告为主；
相机只作可选证据。看不清必须 `UNCERTAIN`，不得肉眼猜身份或通过下载未授权传感器补洞。

### N1-F22：final v2 不是旧阈值的第五次微调

本轮只吸收第四轮已完成的校准信息，变更的是事件语义和证据链：parallel-only subject body entry、独立
receiver 全时序、raw 2 Hz hard evidence、三态 first-failure、streaming worker 与 blind/debug 分离。K4 只做
固定 regression；Resource Contract V1 在任何 final scene 前失败，用户复开后的 V2 已按 N1-F25 修正为
675 scenes 并完整运行，但同样没有用于调参。后续若研究 branch merge 或新资产，必须是新的任务 ID、
预注册与 scene-disjoint 评估。

### N1-F23：共享 cgroup 的 start 合同是研究终止门，而非可绕过的工程告警

final formal 在 clean commit `7104f5c` 的 preflight 已将 runner 自身 RSS 降至 `20,705,280` bytes，仍记录
`cgroup_memory_current_bytes=1,523,929,088`，超过冻结上限 `1,350,000,000`。它在任何 evaluation scene 前
安全失败；独立裁决冻结证据并以 `REJECTED/stop_nuscenes_cutin_mining` 结束。development override 的 32/96
smoke、清页缓存或 K4 回归均不能替代正式 start 合同。禁止杀死 Cursor/Jupyter/TensorBoard 等用户服务、修改
正式阈值、截断正式 split（当时 expected 常量误写为 669，见 N1-F25）或把这次结果说成“nuScenes 没有
cut-in”；`n2_authorized=false` 保持不变。

**2026-07-26 用户复开授权**

上述 `REJECTED` 仍是 Resource Contract V1 下不可改写的历史裁决，但不再代表任务永久停滞。用户随后显式
扩大容器内存并授权继续本次 final：现场复核 `memory.max=128,849,018,880` bytes（120 GiB），原
`2,147,483,648` bytes（2 GiB）资源前提已改变。因此必须保留失败 parent 与独立拒绝裁决，同时使用新的
Resource Contract V2、全新 config fingerprint 和不可复用 run ID 恢复 scene-disjoint formal
（经 N1-F25 确认为 675 scenes）；不得覆盖或续写
V1 失败目录，也不得把 V2 成功倒写成 V1 当时没有失败。

### N1-F24：内存不足时停止并等待资源授权，不继续死磕

**新执行规则**

1. 任何正式或开发任务若触发启动/运行 stop 阈值、`RC=137`、SIGKILL，或观察到持续逼近 cgroup
   `memory.max`，立即停止启动新 batch，并尽最大可能写入结构化 `FAILED/failure.json`、最后完成 scene、
   process RSS、cgroup current、anon/file cache 与 `memory.events`；
2. 不通过反复重跑、杀死 Cursor/Jupyter/TensorBoard 等用户服务、缩短正式 split、降低证据质量、跳过
   audit、修改研究阈值或清理不属于本任务的缓存来争抢资源；
3. 把失败点、最低所需资源和恢复命令回报用户，然后等待用户开放资源；没有新的明确授权时不得自行恢复；
4. 用户开放资源后，先记录新的 `memory.max/current` 和授权时间，版本化 resource contract，使用新 run ID
   从冻结研究配置重新运行。资源合同变化只允许调整资源阈值，不允许调整 cut-in taxonomy、hard gate、
   calibration/evaluation split、抽样或人工聚合门槛；
5. 资源暂停与研究拒绝分开登记。未读取 prospective evaluation scene 的资源失败不能被写成方法精度失败，
   后续在新授权下完成的结果也不能删除或覆盖先前工程失败证据。

### N1-F25：final 的 669-scene 预期是 split 算术错误，不是 evaluation 集合定义

Resource Contract V2 的首次 clean formal run：
`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-01/v71_n1-event-cutin-final-01__receiver-cutin-final-v1__s0__20260726T142634031503Z__5c8c65d7`
在 K4 regression 通过后、任何 evaluation scene 或 candidate 读取前 fail closed，错误为
`final evaluation scene 数不匹配: 675 != 669`。

独立复算表明：nuScenes official `train` 为 700 scenes；冻结的 42 个 calibration scenes 中，25 个属于
`train`、17 个属于 `val`，且没有 split 外 scene。因此 scene-disjoint evaluation 的确定数量是
`700 - 25 = 675`。`_resolve_evaluation_scenes` 已正确执行
`set(train) - set(all_calibration_scenes)`；错误只在 YAML 的 `expected_scene_count` 常量把不属于 train 的
calibration scene 也错误计入了减法。

合法修复仅为把 Resource Contract V2 配置中的 assertion 从 669 改为 675，并生成新 config fingerprint、
新 clean commit 和新 run ID。不得借此增删 calibration scene、显式挑选 evaluation scene、查看 candidate 后
改 split，或修改 taxonomy、strict gate、K4、抽样与人工门槛。失败 run 必须保留为工程契约失败，不能统计成
research reject。

### N1-F26：strict v2 在 675 scenes 上只有 1 个 PASS，不能靠人审单例或放宽规则扩池

Resource Contract V2 的 clean formal run：
`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-01/v71_n1-event-cutin-final-01__receiver-cutin-final-v1__s0__20260726T142941598714Z__883fae9a`
在 commit `beee1de`、seed 0、config fingerprint
`883fae9a6514c0bff5bba8bcaf81a22c79e6d719586221596a7d4b5364c337da` 上完成 675/675 scenes。

结果为 `ABSTAIN=1,556`、`FAIL=200`、`PASS=1`，唯一 PASS 只覆盖 1 scene；冻结 machine-readiness 要求
至少 3 candidates / 3 scenes，故 parent 以
`REJECTED / stop_nuscenes_cutin_mining_too_sparse` 结束。K4、raw-only 和资源合同检查均通过，峰值 batch
process RSS 为 `337,154,048` bytes、cgroup current 为 `4,556,898,304` bytes；这次拒绝不再是资源失败。

独立稀疏终局人工包保留唯一 PASS 和 3 条 diagnostic，但人工结果不能改变数量门失败：即使唯一 PASS 被判为
TP，仍只有 1 TP / 1 scene，低于 sparse 的 3/3。禁止为了形成池而把 ABSTAIN 提升为 PASS、恢复
`receiver_branch_merge`、放宽 raw/parallel/receiver 时序门、事后改 scene 或把单例人工真实性外推为总体
precision。准确结论是“当前冻结 strict v2 的 prospective pool 过稀”，不是“nuScenes 没有 cut-in”。

<a id="detail-cross-route"></a>

## 4. 跨路线必须保留的原则

1. 先证明监督/比较对象存在，再训练或扩量。
2. occupancy、编辑、渲染和标签必须共享同一显式状态，不允许旁路文档绑定。
3. matched baseline 使用相同 proposal、scene、actor、幅度、seed 与预算。
4. top-k 只用于诊断，不替代全分布、coverage 与 worst-case。
5. machine pass 只解锁下一门禁，不自动成为 human verdict、论文 claim 或 scale 授权。
6. hard composition 的局部性与 completion 的质量是两个独立门禁。
7. 下游效用必须由任务指标证明，不能由约束 accept rate、RGB 差分或 PSNR 代替。
8. 工程失败与 research reject 分开登记；既有 provenance 缺失必须诚实标记。
9. 失败范围不能过度外推，但也不能通过改名、放宽阈值或只挑成功场景重复旧问题。

## 5. 新实验防重复检查表

- [ ] 是否明确引用了相关 `RF-*` 与 `V7-RISK-*`？
- [ ] occupancy 是否真正进入决策/状态链，而非只在磁盘上存在？
- [ ] baseline 是否 matched，而非故意构造的极端负例？
- [ ] primary endpoint 是否避免“方法规则自己定义方法成功”的循环论证？
- [ ] 是否同时报告全分布、coverage、per-scene 与 worst case？
- [ ] completion 是否测 inside quality/temporal/depth，而非只测 outside exact？
- [ ] human verdict 是否只由用户/指定评审者填写？
- [ ] run 是否有唯一 ID、resolved config、fingerprint、metrics、summary 与终态标记？
- [ ] 哪个单卡门禁失败时停止，什么条件才允许 scale？

## 6. 2026-07-26 路线转向新增失败与防重复项

### PIVOT-F01：nuScenes cut-in 没有可验证的召回率分母

**观察**

nuScenes 官方公开的是场景、样本、对象实例、类别、属性、3D 框、传感器与地图等结构；`scene.description`
是自由文本，不是事件级 cut-in 真值。官方没有发布 cut-in 场景占比、逐事件标签或可直接用于召回率计算的全集分母。
四轮挖掘和最终 675-scene prospective run 最多只能测量“冻结规则产出的候选质量”，不能测量“数据集中所有
cut-in 被召回了多少”。

**最终证据**

- `N1-F26`：strict v2 在 675 个 scene 上只有 `1 PASS / 1 scene`；
- 最终人工稀疏包即使把唯一 PASS 判为真阳性，也仍低于预注册的 `3 candidates / 3 scenes`；
- 该结果不能外推为“nuScenes 没有 cut-in”，也不能用事后放宽规则伪造召回。

**裁决**

`cut-in mining` 状态固定为 `rejected / frozen`。以后 cut-in 只允许作为已经具备重建与编辑能力后的可选演示，
不再承担数据集入口、方法定义、训练前置条件或论文成立条件。

**解除条件**

只有新的、独立的数据源提供事件级真值及明确分母，或新的任务本身不需要宣称事件召回率，才允许创建全新任务 ID
重新讨论；不得恢复当前 strict-v2 阈值调参。

### PIVOT-F02：贡献漂移——工程系统吞噬了重建与编辑研究

**观察**

过去路线的主要投入逐步变成事件挖掘、地图匹配、接收车身份、规则校准、候选审核与资源合同。它们改善了审计性，
却没有自然回答动态对象几何、连续运动表示、遮挡/去遮挡、反事实轨迹编辑或下游感知一致性。

**边界**

这不否定已形成的 WorldState、typed label、run contract、审计与人工审核基础设施；它只否定把“更好的 cut-in
挖掘器”作为 3DGS/4D 重建论文的核心贡献。

**后续约束**

新路线必须先复现公开强基线，再通过重建/编辑压力测试选择创新点。数据工程模块只能服务于冻结实验，不得重新成为
论文主任务。每个里程碑都必须说明它直接回答的重建或编辑问题。

### PIVOT-F03：未完成 exact reproduction 前禁止集成式“改进”

**观察**

`RF-05/06/08/09/16/18` 与 `V7-RISK-03/04/06/07/10/16/17` 共同表明：输入、状态、比较对象、覆盖率和真值定义
未冻结时，模块堆叠会把工程可运行误当成方法收益。AD-GS 的公开 nuScenes 协议提供了固定 scene、帧区间、预处理、
训练和评测入口，适合作为新的事实锚点。

**禁令**

在 AD-GS exact reproduction 门禁通过前，不得：

- 合并 Motion-Proj/StreetGS/OccGS 模块；
- 加 occupancy、物理约束、扩散补全、感知损失或轨迹编辑；
- 更换为自选事件场景、调低分辨率后对齐论文指标或只展示成功帧；
- 把兼容性补丁、预处理修复或运行成功表述为方法改进。

任何 unavoidable compatibility patch 必须独立提交、最小化、附 upstream diff 与消融；原始基线结果必须保留。

### PIVOT-F04：不能把“可见性建模”泛化成未观测背景已经解决

**观察**

AD-GS 的双向时间可见性用于动态对象生命周期和已观察运动建模；VAD-GS（CVPR 2026）的 visibility-aware
densification 已覆盖稀疏观测下的几何补密；DrivingEditor 支持对象删除/添加；Real2Sim 进一步展示对象级编辑与
物理交互。这意味着“增加一个 visibility 模块”或“支持平移对象”本身已经不足以构成新意。

**仍未闭合的问题**

反事实轨迹编辑会同时制造原位置去遮挡、新位置遮挡、跨相机深度排序和证据外外观。当前项目只有在下列内容形成
联合、可验证方案时才可提出方法 claim：

- 编辑诱发的显式可见性重计算；
- 未观测区域的真实性/置信度与拒绝机制；
- 跨视角、跨时间一致的背景恢复；
- 目标区预期变化与非目标区感知保持。

**防重复**

创新选择前必须把 AD-GS、VAD-GS、DrivingEditor、DGGT/ReconDrive 和当时最新工作重新做一次代码可用性与
claim 边界审计；不得把已有 visibility-aware densification 或基本对象变换重新命名为贡献。

### PIVOT-F05：资源不足时研究停机规则跨路线继续生效

`N1-F24` 是项目级规则，不属于 cut-in 专属逻辑。本轮 cgroup 为 `memory.max=2,147,483,648` bytes，轻量元数据
审计后 `memory.current` 一度达到 `2,129,526,784` bytes，因此立即停止 Python 扫描、conda 求解、下载、
预处理和训练，只继续轻量文本/文件操作。后续任何新路线任务遇到相同条件时，必须保存失败/现场证据并等待用户开放
资源；不得反复重跑、杀用户服务或偷偷缩减正式协议。

### PIVOT-F06：旧机器 smoke 证据不能替代新实例复验

迁移到 RTX 4080 SUPER 新容器后，已有环境目录和旧 RTX 4090 日志仍然存在，但它们不能证明当前 driver、CUDA、
扩展 ABI、显存与 cgroup 合同可用。M2 因此在新机器上重新执行 AD-GS forward/backward、DPT、SAM2、
Grounding DINO HF 和 CoTracker3 smoke，并为每项保存独立退出码。

后续换机或容器重建时，即使复用同一 env/checkpoint，也必须生成新的 instance 级环境证据；旧日志只能作为历史，
不能复制为当前 PASS。

### PIVOT-F07：非 login shell 的 PATH 不能作为 CUDA provenance

M2 首次当前机器采集因非 login shell 找不到 `nvcc` 提前失败，而 `/usr/local/cuda/bin/nvcc` 实际存在。环境报告
已改为显式设置 `CUDA_HOME` 并调用绝对路径，同时传播 smoke 的真实退出码。

后续自动任务必须显式记录并使用 toolkit 路径；“命令不在 PATH”与“机器没有 CUDA toolkit”必须分开裁决。

### PIVOT-F08：在线浮动模型不能进入 exact reproduction

upstream 的 CoTracker `torch.hub` 在线 `main` 与未固定 revision 的 Hugging Face 模型都会随时间变化。M2 将
CoTracker repo、离线 checkpoint、Grounding DINO HF revision 与 snapshot fingerprint 全部固定并哈希，
运行时使用 offline mode。

后续 baseline 不得在正式 run 中联网追随 `main`、latest 或未固定 snapshot；若必须升级，使用新 config
fingerprint 和新 run instance。

### PIVOT-F09：tar 页缓存与 nuScenes auxiliary 都属于资源/资产合同

并行流式扫描约 294 GB tar 时，文件页缓存计入本容器 cgroup，首个扫描实例峰值达到 `57,001,484,288` bytes。
本任务只对自己已读过的 tar 文件区间调用 `POSIX_FADV_DONTNEED`，没有全局 `drop_caches`、杀用户服务或清理
其他任务缓存。

第一次结构审计还发现 1,440 个 RGB/LiDAR payload 齐全并不足以初始化 nuScenes devkit；`map.json` 引用的
4 个静态 map masks 同样是必需资产。失败实例
`20260727T165549__e49a4e-4080s-r2` 保留为 `blocked`，补齐并哈希登记 maps 后由新实例
`20260727T180733__e49a4e-4080s-r3` 通过。以后 selective extraction 必须同时审计运行库隐式依赖的 auxiliary
文件，不能只按训练脚本直接打开的传感器路径计数。

### PIVOT-F10：AD-GS 的 PNG 输出与 SAM2 的 JPEG-only 枚举不兼容

M3 首个 scene-0230 实例完成 `prepare_raw` 和 180 张 depth 后，在 sky mask 初始化时报
`no images found`。AD-GS `scripts/nuscene/nuscene.py` 固定写 `000000.png`，其 `semantic.py` 自己也会枚举
PNG；但 Grounded-SAM-2 `load_video_frames_from_jpg_images` 只按 `.jpg/.jpeg` 扩展名建立 video frame 列表。
这不是空数据、模型失败或资源 OOM。

失败实例 `20260727T181617__scene0230__s0` 保留为 `blocked`。最小兼容性修复只在 instance work dir 中为每个
PNG 建立相同字节内容的 `.jpg` 硬链接，跨文件系统时复制原始字节；PIL 已验证按内容可无损读取，不做 JPEG 转码，
不改 Grounding/SAM 模型、box/text 阈值、mask、帧序或评测。修复后的 AD-GS patch SHA-256 为
`114c3976af2c80d1da5581b401b3a099f22a7483347fc401113c8439bc991eb9`，必须由新 M3 instance 复验。

### PIVOT-F11：COLMAP 默认全核并发会越过本机 cgroup 内存门禁

M3 第二个实例 `20260727T182247__scene0230__s0-r2` 已完成 180 张 depth/object/sky/semantic 与
138/138 flow，在 COLMAP feature extraction 阶段发现 upstream 未指定线程数，COLMAP 自动使用容器可见的
128 个 CPU threads。cgroup memory 峰值达到 `62,265,835,520` bytes，并连续两个采样超过
`memory.max=66,571,993,088` 的 90% 停止线；runner 只终止本 run 的进程组，`oom=0 / oom_kill=0`，
失败实例与部分 COLMAP 目录均保留。

这不是图像、SIFT、匹配或几何协议失败。最小资源兼容修复显式传入
`SiftExtraction.num_threads=16` 与 `SiftMatching.num_threads=16`；不改分辨率、相机、帧、SIFT 参数、
exhaustive matching 或评测。r3 的 COLMAP 已完成 138/138 图像注册与 70,933 points，阶段峰值降至
`35,117,174,784` bytes。当前完整 compatibility patch SHA-256 为
`49b4c06ecec6c30f1e80b5abf4d46970920f9d71952acbda273774d9b5b34f48`。

以后在 CPU 核数远大于内存预算的容器内运行 COLMAP，必须显式登记并发数并纳入资源合同；不得把减少图像、
降低分辨率或删相机伪装成等价的资源修复。

### PIVOT-F12：跨场景连续执行时，official render 可先于 OOM 触发 cgroup 90% 合同

M3 scene-0230 的 60k render 峰值已达到 `59,530,678,272` bytes，距离注册的 90% 停止线仅
384,115,507 bytes。M4 scene-0242 严格串行完成全部 preprocess 后，100-step train 峰值为
`59,359,428,608` bytes；随后的 official render 在第 2/138 帧连续两个采样达到 90%，峰值
`59,996,393,472` bytes，比停止线高约 81.6 MB。runner 只向本 stage 进程组发送 `SIGTERM`，
stage `rc=-15`、runner `rc=1`，`oom=0 / oom_kill=0`，没有影响其他服务。

该结果说明当前 `memory.max=66,571,993,088` bytes 对六场景连续 exact reproduction 没有足够安全余量；
“尚未 OOM”不能用来绕过预注册停止线。blocked 实例
`/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260727T235743__scene0242__s0/`
必须保留，不在同一资源合同下立即重跑，也不得降分辨率、删相机、调模型或全局 `drop_caches`。

恢复时应先提高 cgroup 内存额度，建议至少 80 GiB、推荐 96 GiB，再创建新 instance；允许复用逐文件冻结的
processed scene，但必须记录来源、哈希和新资源合同。若无法增加资源，则 M4 保持 `blocked`，不得将只有
scene-0230 的结果写成六场景论文复现。

### PIVOT-F13：processed scene 复用校验必须区分关键产物与合法空占位

RTX 3090 换机后的首个 scene-0242 复用实例
`/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T131533__scene0242__s0-r2-wm3090/`
在训练前 fail closed：新增递归哈希校验把 COLMAP 的合法 0-byte `created/sparse/model/points3D.txt`
占位文件误判为损坏。旧实例以 `blocked` 保留，没有修改 processed scene 或启动训练。

合法修复允许 COLMAP 非关键占位文件为 0 bytes，同时继续强制 `database.db`、`cameras.txt`、
`images.txt`、`colmap.ply` 和所有训练直接消费的 image/depth/mask/flow/meta/point cloud 非空；
复用后必须重新运行独立 processed audit。修复后的新实例 output fingerprint 为
`32bf9ccaa108273b69286625a0c7aaacb04fd9d76f243daff976206d0b7ef4f6`，138/138 registered images
审计通过。不得删除空占位文件、伪造非空内容或因此重跑昂贵预处理。

### PIVOT-F14：容器实例重建必须与 OOM/方法失败分开

M5 首个正式 DGGT 实例
`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T094923__native-nusc-s0-wm3090/`
在 `env_torch` 下载期间停止。日志没有 stage 终态、OOM 或 `oom_kill` 增量；当前容器 PID 1 的启动时间为
2026-07-29 13:13:43 +08:00，晚于日志停止时间，故裁决为外部容器实例重建，而不是 DGGT 精度、显存或方法失败。

旧 run 与旧 controller 已原子标为 `blocked`，部分环境移动到
`/root/autodl-tmp/envs/dggt.interrupted-20260729T094923/`，没有覆盖或删除。恢复使用新 run
`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/`；以后同类中断必须先核对 PID 1 启动时间、stage marker、launcher 与 OOM 证据，再决定是否创建
新实例，禁止把 stale `running` 当成任务仍存活。

### PIVOT-F14B：pointops2 的 PEP 517 build isolation 没有继承已安装 torch

恢复实例完成 Python 3.10、torch 2.4.1 和全部 requirements；resolver 最终选择
`rerun-sdk 0.23.1 / opencv-python 4.11.0.86 / numpy 1.26.4`。随后 upstream pointops2 执行普通
`pip install .` 时，PEP 517 临时 build env 在读取 setup requirements 阶段报
`ModuleNotFoundError: No module named 'torch'`。正式 stage `rc=1`，峰值 cgroup memory
`16,839,843,840` bytes、GPU 0 MiB，`oom=0 / oom_kill=0`。

正式 blocked 证据：
`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/`。
checkpoint、native inference 和 common-observation metrics 均未启动；没有在同一实例事后加入
`--no-build-isolation` 覆盖失败。该结果属于明确 upstream packaging blocked，不是 DGGT 质量、显存或方法裁决，
并按权威计划第 15.1 节满足继续 M6 的替代前置证据。

### PIVOT-F15：AD-GS 冻结 pseudo ID 与 checkpoint 都不能支持单对象编辑

M6 直接审计训练前冻结的 `semantic/mask_*.npy`。按 camera-local ID 统计，六个官方场景最长支持帧为
`1 / 6 / 1 / 1 / 2 / 1`，全部低于预注册 `≥20/60`；processed scene 也没有冻结的 vehicle track artifact。
与此同时，六个 60k checkpoint 的 `point_cloud.ply` 均只有二值 `obj∈{0,1}`，没有持久 instance ID。

正式证据：
`/root/autodl-tmp/runs/dynamic_recon/DR-M6-STRESS-01/20260729T145645__identity-audit-s0-wm3090/`。稳定失败
`persistent_object_identity_unavailable` 在 6/6 scenes 重复。对象编辑、pseudo-hole 和噪声行全部保留为
`ABSTAIN`，0/12 object slots 没有从 coverage 分母删除。禁止在看到 checkpoint/场景结果后用几何 Hungarian
轨迹回填 M6 baseline；这类重关联只能作为新方法候选，并必须先过 novelty。

### PIVOT-F16：instance-aware 与 driving edit 已被 2025–2026 工作直接覆盖

M7 只沿决策表考察 A“可编辑运动表示与轨迹不确定性”。重新核对官方来源后：InstDrive 已用 SAM pseudo masks
学习动态驾驶场景 2D/3D instance identity；Director 已做 4D Gaussian identity consistency；OmniRe 已用
actor scene graph/canonical vehicle nodes 做仿真；HorizonForge 与 G²Editor 已覆盖车辆轨迹操作、删除和遮挡区
恢复。

正式 evidence：
`/root/autodl-tmp/runs/dynamic_recon/DR-M7-HYPOTHESIS-01/20260729T145748__novelty-audit-s0-wm3090/`。候选的持久身份、actor-centric Gaussian binding、时序一致性和轨迹/对象编辑
核心机制均为 direct overlap；confidence/ABSTAIN 是评测与安全护栏，不构成独立技术 delta，剩余差异只是
AD-GS 适配工程。因此 M7=`rejected`，不注册事后 primary endpoint；M8/M9 均
`rejected / not authorized`，不得通过改名、挑场景或把 0 coverage 写成改进继续。

### PIVOT-F17：DGGT 扩展构建必须同时固定 compiler、headers 和 Python 依赖上界

V2 M1 表明，“已安装 torch cu121”并不足以证明 CUDA extension 可构建。宿主只有 CUDA 11.8
toolkit，会在 pointops2 编译时与 torch 2.4.1+cu121 硬失配；只补 `nvcc` 又会缺
cusparse 等 headers。正确合同是在前缀环境固定 NVIDIA CUDA 12.1 compiler/runtime/headers，
传播 `CUDA_HOME/CPATH/LD_LIBRARY_PATH`，再按 upstream `python setup.py install`。

同一里程碑还暴露了浮动 Python 树的独立风险：transformers 5.x 使用 torch 2.4.1 未提供的
DTensor API，diffusers 0.39 触发 torch schema 不兼容。最终固定
`transformers 4.48.3 / tokenizers 0.21.0 / diffusers 0.32.2 / numpy 1.26.4 /
opencv-python 4.11.0.86 / rerun-sdk 0.23.1 / flow-vis 0.1`。

对应 blocked runs 为
`20260802T120027Z__native-nusc-s0`、`120943Z__...-r2`、`122213Z__...-r3`、
`122904Z__...-r4`、`124347Z__...-r5`。这些失败是构建/依赖证据，不能推断 DGGT
方法质量。

### PIVOT-F18：原生阶段完成不应被后续评估依赖失败覆盖

M1 r6 已完成 18/18 1-view 和 18/18 3-view，但 common evaluator 导入 AD-GS 冻结
`loss_utils` 时因 `flow_vis` 未安装而 blocked。原生输出本身未损坏，但主 terminal 已转为
blocked，禁止为了“好看的 done”改写。

恢复方式是新建 r8，对 r6 `native_summary.json/metrics.json`、每个 stage 和输出哈希做
fail-closed 引用后只执行 common diagnostic。r7 中重试封装自身的 `KeyError` 也以新的
blocked run 保留，再由 r8 完成。后续所有 multi-stage run 必须把“可复用的完成阶段”与
“整个 instance 的 terminal 终态”分开；重试不得修改旧 terminal。

### PIVOT-F19：nuScenes devkit 反向索引与磁盘 metadata 不是同一 schema

M2 r1 直接读取官方磁盘 `sample.json` 时发现其中没有 `anns`；该字段是 nuScenes devkit
初始化后才注入的反向索引，不是原始 JSON 合同。正式适配器改为流式扫描
`sample_annotation.sample_token`；由于这个外键非唯一，不得用单值 dict 覆盖同一 sample
的多个 annotation。`ijson` 还必须以 `use_float=True` 读取，否则 Decimal 会污染严格 JSON
运行合同。

同一里程碑还表明，“时间最近”不足以建立 raw annotation 到 camera sweep 的真值映射。r4
中 scene-0242 boundary actor 命中更近的 sweep，但 sweep 所属 `sample_token` 与 raw 2 Hz annotation
不同，因此在 QA 前即 blocked。正确规则是先限定 exact sample token，再在候选内最小化
timestamp delta；正式 r5 达到 `4356/4356` exact mappings。后续不得仅按文件名或时间
猜测 raw/processed/render 映射。

### PIVOT-F20：CUDA 扩展 import 成功不等于包含当前 GPU 架构

M3 的 DriveStudio 环境能正常 import `gsplat` 和 `nvdiffrast`，但旧二进制没有 RTX 3090 的
SM 8.6 kernel：前者在 SH rasterization 报 `no kernel image`，后者在 EnvLight 路径报 CUDA 209。
只做 import smoke 无法发现此类错误。恢复时分别固定官方源码 commit，以
`TORCH_CUDA_ARCH_LIST=8.6+PTX` 重建，并执行真实 CUDA forward/backward；旧 `.so` 先备份，
没有修改算子语义或模型配置。

对应 blocked runs 为 r4、r6、r7；正式 binary SHA-256 为 gsplat
`6d7c8e5a...dd6131`、nvdiffrast `0d18f767...96499`。以后 CUDA 扩展 readiness 必须包含
目标 GPU 上的实际 kernel forward/backward，不能只看包版本和 import。

### PIVOT-F21：训练完成 checkpoint 与累积式 post-render 必须分开裁决

M3 r8 的 30k 原生训练已保存 `step=30000` checkpoint，但上游随后将 588 个 full-render 结果累积在
内存中；在 `577/588` 时 cgroup memory 连续两次超过 90%，资源守卫发送 SIGTERM。`oom=0 /
oom_kill=0`，checkpoint 字节数与 step 完整。r8 仍保持 `blocked`，不得改写为 done；r12 通过新的
不可变 run 对 checkpoint step/bytes/hash 和原失败 terminal 做窄范围复核，再执行流式 27-image
edit smoke 完成 M3。

同一恢复链还发现，正式训练会把某个非目标 rigid model 的全部 Gaussian 裁剪掉。token、dataset column
和 model index 仍是一一映射，但 checkpoint slice 为空。registry v2 因此将其明确标成
`unavailable_empty_checkpoint_slice`，同时对正式选中 actor 继续要求非空。禁止为了全 registry 看起来
完整而伪造 slice，也禁止因一个非目标空 slice 丢弃 23 个真实非空映射。

### PIVOT-F22：外层 timeout 不会自动回收独立 session 的 GPU 子进程

M4 controller 用 `subprocess.Popen(..., start_new_session=True)` 隔离正式渲染，使 SSH/tmux 断开不应
误杀长任务；相应地，用外层 `timeout` 调试 controller 时，SIGINT 只终止父进程，子进程会以 PPID 1
继续占用 GPU。`debug_controller_s0_r5` 复现了该行为；残留子进程通过已核实的精确 PGID 发送 SIGTERM
回收，GPU 从约 `8.1 GiB` 回到 `0 MiB`，没有终止用户服务。

以后不得用外层 timeout 探测会派生独立 session 的 controller。正式运行应直接由 nohup/tmux 托管，
同时监控 controller PID、child PID、terminal 和 resource.jsonl；确需中止时必须核实 process tree 后
显式回收 child process group。`r5/r6` 的 running terminal 保留为中断证据，不改写成 done。

### PIVOT-F23：SE(3) 一致性容差必须覆盖 float32 往返误差

M4 单帧 r1 的 actor transform 先由 checkpoint float32 tensor 变换，再写入 JSON 并读回，最大平移误差
略高于 `1e-6 m`；其余 15 项检查均通过。把该值当几何失败会制造假阴性。协议在查看正式全量结果前
固定为 `1e-4 m`，r2/r3 冒烟通过，正式 196 帧实测最大误差为
`3.814697265625e-06 m`，rotation/size/canonical drift 均为零。容差变更只反映数值精度，不降低
1 m 编辑幅度，也不得据此为真正的轨迹偏差放宽门禁。

### PIVOT-F24：冻结 heldout 资源门失败不能靠事后更换 renderer 或提高阈值挽回

A3 R1 在结果前冻结 `12,288 MiB` PyTorch allocated GPU ceiling。三条完成全部 R0/R1 指标计算的只读路径
`r2/r4/r5` 分别达到 `14,241.777 / 14,244.924 / 14,241.399 MiB`；wall、cgroup、run bytes 与 OOM delta
均通过。资源审计、CPU checkpoint staging、Rigid quota device 兼容和逐 view `trainer.info` 释放都没有改变这一
单 view 峰值。继续把 renderer 改为 `packed=true`、分块/降分辨率，或把 ceiling 提高到观测值以上，会在看到结果后
改变 source-render 路径或预算，不再是预注册评测。

r5 的资源无效 diagnostic 也不能救回方法：S-B depth-order violation 从 `0.915792` 降至 `0.908173`，但
non-target 与 original-global RGB MSE 都按 exact comparator 严格变差，故为 `tradeoff_non_dominated`。正确分层是：
r5 run 保留 `blocked`，R1 方法臂登记 `rejected_resource_gate_and_diagnostic_tradeoff`，A3 任务以负结果 `done`；
生产路由回退到 R0/D2 immutable exact alias。以后若研究 packed/分块渲染，只能在 A4 作为新的部署因子另行冻结，
不得倒写 A3 heldout 结论或解锁 R2–R4。

### PIVOT-F25：部署 profile 必须区分传感器原始尺寸与 checkpoint 原生加载尺寸

A4-P0 v1 在新测量前把 scene-0230 的 nuScenes 传感器尺寸 `1600×900` 冻结为“原生分辨率”。formal r1 实际
完成 2 次 warm-up 与 9 次 measured render 后，11 行全部为 `800×450`，因此只在 finalize 的
`native_resolution_exact` audit 失败；资源、输入 hash、无训练/无 checkpoint、同步矩阵与无 torch resume audit
均通过。source config 事后审计确认三路相机已冻结 `data.pixel_source.downscale_when_loading=[2,2,2]`，故当前
checkpoint 的模型原生加载/渲染尺寸本来就是 `800×450`。

不得修改 v1 或把 r1 改写为 done，也不得用 r1 性能数字关闭 P0。正确处理是保留 r1=`blocked`，冻结其 protocol、
manifest、runtime stage/rows、resource audit 与 terminal hash；创建 v2 只纠正分辨率语义，再从新目录完整重跑。
后续协议必须同时记录 sensor resolution、source-config downscale 与 model-native render resolution；“native”一词
不能在这三层之间无来源转换。该纠错不授权降低分辨率、切换 renderer、改变资源 ceiling 或开启 P1/P2/P3/P5。

### PIVOT-F26：checkpoint state key 不能冒充加载后的模型运行时属性

A4-P5 formal r1 已通过 9 项输入审计并生成 `14,729-byte` reference-only deployment registry，但 fresh DriveStudio
worker 在 checkpoint 成功加载后读取 `RigidNodes.points_ids` 时报 `AttributeError`。源码事实是：checkpoint
`state_dict` 以 `points_ids` 为序列化键，`load_state_dict` 将它弹出并写入运行时属性 `self.point_ids`。两者语义
相关但接口层不同；直接把 checkpoint 键拼成对象属性会使恢复链在资产已经物化后失败。

r1=`20260809T155209Z__a4-p5-registry-resume-s0-r1` 保留 `blocked`，terminal SHA=`61d30a11...773e`；其已生成
registry SHA=`e48bccdf...9039d` 不覆盖。修复 `0e899b2` 只通过 fail-closed helper 读取 runtime `point_ids`，并用
两条测试分别锁定有效属性和拒绝 `points_ids` 别名；没有修改 P5 protocol、输入、资源 ceiling 或审计口径。
新目录 r2 以相同 registry SHA 完成 14/14 audits，证明问题属于 runner/runtime contract，不是资产或方法失败。

以后凡从 checkpoint 结构推断 live module API，必须同时核对 `state_dict` 保存端、加载端赋值和加载后的真实对象，
并用回归测试锁定层级；旧失败 run 维持 `blocked`，不得因修复后的新 run 成功而倒写。

### PIVOT-F27：最小预注册剪枝臂失败后不能事后补更小 fraction 或放宽质量门

A4-P1 在结果前固定 source/b05/b10/b20 四臂，并要求 global、actor、boundary 与 non-target 的全部 safeguard 同时
通过。canonical r1=`20260809T165058Z__a4-p1-contribution-prune-s0-r1` 完成 36-view contribution、三臂物化、
四臂 57-view 质量和 9-view runtime，21/21 audits 全 true，资源门通过；因此它不是工程或资源 `blocked`。

最小候选 b05 已将 checkpoint 减少 `23,881,368 bytes`，全部 row/invariant/reload/count 审计 exact，但 global
occupied PSNR、global PSNR 与 non-target PSNR 分别退化 `0.117684/0.110926/0.125462 dB`，超过冻结的
`0.10 dB` 上限；b10/b20 分别失败 12/15 个端点。局部 actor/boundary 指标的保持或改善不能覆盖全局与非目标区
失败。运行时 P50/FPS 也非随 fraction 单调，且 filesystem cache 未控制，只能报告，不能充当事后选择理由。

正确裁决是 P1 experiment=`done`、method=`rejected_quality_or_integrity_gate`、生产资产 exact fallback 到 source。
不能在看到 b05 失败后新增 b01/b02、改排名视图、放宽 `0.10 dB` 或只保留通过的局部端点；这些都属于新的预注册
实验，而不是当前 P1 的修复。该负结果只约束 scene-0230/seed-0/冻结视图矩阵，不外推为所有贡献度剪枝均失败。

### PIVOT-F28：顶层 `named_parameters()` 不保证覆盖普通映射中的子模型参数

A4-P2 formal r1=`20260809T174337Z__a4-p2-mixed-precision-s0-r1` 已成功完成 10-field checkpoint conversion、
source/candidate 57-view quality、两臂 runtime、aggregate 与 no-torch resume；aggregate 也按冻结门选择 mixed arm。
但 finalizer 的 `checkpoint_reduction_and_runtime_matrix_exact` 唯一失败：账本只调用
`trainer.named_parameters()`，而 DriveStudio 把 Gaussian 子模型保存在普通 `trainer.models` 映射中，并未注册成
顶层 `ModuleDict`。因此账本只看见 LPIPS 等 `9,883,392 bytes`，把 source/candidate 错记为相同总量且没有 FP16
bucket，尽管候选 checkpoint 已实际从 `578,819,674` 降到 `432,111,754 bytes`。

这属于 evidence collection defect，不是 conversion、质量、资源或方法失败。r1 保留 `blocked`，terminal SHA=
`5ef3dab60ff934af19ff547c0f7e7cd0fe74b83000476888a630341ee39474c0`；不得手改 r1 runtime stage 或把它倒写为
done。修复 `dcf2822` 显式遍历 `trainer.models` 中每个 module 的 parameters，并按 Parameter identity 去重；回归 fixture
故意使用未注册的普通映射，锁定 `models.Background._scales` 等字段必须进入账本。协议、字段、阈值、renderer 与
selection 均未改变。

新目录 canonical r2=`20260809T174850Z__a4-p2-mixed-precision-s0-r2` 完成 19/19 audits，正确记录 source/candidate
persistent bytes=`394,641,424 / 247,936,208` 与 candidate FP16 bucket=`146,705,216 bytes`，并选择 mixed arm。
以后审计复合模型时，必须同时核对容器是否已注册为 `nn.Module`、顶层 traversal 覆盖范围与逐字段预期集合；
证据账本缺失不得用 checkpoint 文件变小或 runtime 成功来推断补齐。

### PIVOT-F29：无卡实例必须以 cgroup 内存为资源合同，不能读取宿主机 `free`

V3.2 S0 在 AutoDL 无卡开机模式中观察到 `free -b` 暴露宿主机约 810 GB 内存，但
`/sys/fs/cgroup/memory.max=2,147,483,648`，实际只有 2 GiB。CoIn 的完整 partial-clone checkout 在大量 blob
物化时把 `memory.current` 推近上限且长时间无进展；继续并发创建环境或校验大权重会把可回收 page cache 与真实
匿名内存混在一起，增加无意义的 OOM 风险。

正确处理是停止该 checkout，把残留移到仓库外备份，并改用 `--no-checkout --filter=blob:none` 固定 commit/tree；
所有下载使用流式落盘，依赖安装和 hash 校验串行执行。GPU、VRAM 与 driver 则必须在有卡重启后重新审计，不能把
无卡模式的 `nvidia-smi: permission denied` 写成硬件不存在。后续任何资源 preflight 都必须同时记录
`memory.max`、`memory.current`、`memory.events` 和数据盘余量；宿主机 `free` 只作诊断，不作授权。

### PIVOT-F30：原子发布目录不能把 `.partial` 绝对路径写进 manifest

S1 prompt preparer 首次在 `s1_prompt_v1.partial` 内生成绝对 `video_dir`，发布时将目录改名为
`s1_prompt_v1`，manifest 内部却仍指向已经不存在的 `.partial` 路径。SAM2 因此把它判定为既非 MP4 也非 JPEG
目录，r1 在真实 GPU 启动后立即失败。修复是 manifest 只保存相对 `sam_inputs/...`，消费者相对 manifest
父目录解析；原子 rename 后重新验证每个目录和 JPEG 链接。任何会整体 rename 的 staged asset 都不得在内部保存
staging 绝对路径。

### PIVOT-F31：SAM2 `reverse=True` 默认从最早 prompt 开始，可能合法地产生零帧

首次双向传播实现只设置 `reverse=True`，但官方 predictor 默认 `start_frame_idx=min(condition frames)`；train-only
block 的首个 prompt 常在 local frame 0，反向 processing order 因此为空。调用成功和进度条 `0it` 不能证明反向
覆盖。正确做法是显式用 block 内最晚 prompt 作为 reverse start，并按每个 object 自己的 prompt frame 过滤输出；
r5 中实际产生 13 个 prompt 之前的 mask，才构成双向证据。

### PIVOT-F32：mask QC 必须在同一像素坐标系比较

r4 将 SAM logits 从源图 `1600×900` resize 到模型原生 `800×450`，却直接与源图坐标的逐帧 3D box 比较，
造成 `235/263` 假拒绝。修复后 box 按 exact x/y 比例映射到 800×450，r5 为 `212 accepted / 51
fail-closed`，其中 43 个是近空 mask。以后任何 IoU、centroid、boundary 或 area ratio 门禁都必须同时记录 source
size、target size 与变换；不同尺度间的数值不得直接进入裁决。

### PIVOT-F33：大规模 Gaussian 重复索引累加不得使用逐元素 `np.add.at`

S1 r2 在每个视图的数百万 ray/Gaussian intersections 上多次使用 `np.add.at`，CPU 单核成为瓶颈；同时该版本
仍缺计划要求的 negative views、depth-consistency rate 和 boundary score，因此保留 250 个 mask 后以 exit 143
终止，不得作为完成证据。r5 改用 `np.bincount(minlength=total)` 和向量化 view count，263-view lift wall
`770.733s`，并保存完整 posterior schema。研究 runner 必须输出阶段进度；“CPU 持续运行”不能替代复杂度审计。

### V6-F06：数据 adapter 的运行环境必须覆盖写出阶段依赖

R3 首次正式目录
`20260821T085802Z__support-deviation-s20260821-r1` 在完成 scene-0242 图像、标定与点云聚合后，
于 `store_ply` 写出阶段因主环境缺少 `plyfile` 失败。该 run 保留 `failed` terminal，不改写为完成；
失败发生在任何 checkpoint 推理、质量读取或确认集读取之前，因此不是方法负结果，也不是 GPU/数据资源不足。
修复只把冻结的 adapter 命令路由到已经具备 AD-GS 依赖的 `/root/autodl-tmp/envs/adgs/bin/python`，
不改变数据分区、场景、checkpoint、support 假设、指标或门槛。以后环境 readiness 必须覆盖 adapter 的最终序列化依赖，
不能以脚本启动和主体循环成功代替端到端环境兼容性。

### V6-F07：只读 renderer 的数据 loader 仍可能强制读取训练期辅助字段

R3 第二次正式目录
`20260821T090109Z__support-deviation-s20260821-r1` 已完成 scene-0242 adapter 和 StreetGS 全部冻结渲染，
随后 AD-GS `Scene` 构造因 adapter 没有 `depth/000000.npy` 失败。V4 adapter 按设计只生成图像、语义、天空、
位姿与点云，而 AD-GS nuScenes loader 即使在只读 checkpoint 渲染时仍无条件加载每张训练期 depth；源码检索确认
`gaussian_renderer` 不读取 `viewpoint_camera.depth`，R3 指标也只使用 DriveStudio 导出的真实稀疏 LiDAR。
因此该 run 保留 `failed` terminal，不改写为方法负结果。

修复是在每个不可变新 run 的 adapter 内生成全零 float32 depth 占位文件，并写出独立 audit，明确标注
`loader_field_only=true`、renderer/指标不消费以及真实几何证据来源；不修改 AD-GS checkout、checkpoint、
开发/确认分区、指标或假设。以后复用训练代码做只读渲染时，必须区分 loader 的强制 schema 字段与实际计算依赖，
占位值只能用于经源码证明不被实验结果消费的字段。

### V6-F08：通用场景点云不能替代 checkpoint 对应的 object-aware loader 资产

R3 第三次正式目录
`20260821T090552Z__support-deviation-s20260821-r1` 已再次完成 scene-0242 adapter 与 StreetGS 渲染，
AD-GS 随后在 `readnuScenesInfo` 对 `obj_id[..., 0]` 索引时失败。通用 V4 adapter 生成的 `points3d.ply`
只有 xyz/rgb/time，没有 AD-GS 训练 adapter 的 `obj` property；即使 checkpoint 加载随后会覆盖 Gaussian 初始化，
`Scene` loader 仍先强制构造 object-aware point cloud。该 run 保留 `failed`，不是方法或资源负结果。

修复不伪造 object id，而是把同一场景、同一冻结 checkpoint 训练时使用的
`adgs_processed_v4/train/<scene>/points3d.ply` 复制进新 development adapter；绑定前验证 PLY header 的
`property float obj`，记录通用点云 hash、冻结训练点云路径/hash 和复制后 hash。开发图像、位姿与分区继续来自
新 adapter，checkpoint 与指标不变。以后给冻结模型换 evaluation camera 集时，应复用训练时与模型结构耦合的
初始化/registry 资产，只替换经协议允许的观测与相机字段。

### V6-F09：lazy camera 偏移前必须整体迁移设备，不能只新建 CUDA 外参

R3 第四次正式目录已成功完成 scene-0242 的 object-aware `Scene` 构造与 checkpoint restore，
在首个 novel camera 的 `full_proj_transform` 计算处失败：冻结 AD-GS 配置启用 `lazy_load_to_gpu`，
原 camera 的 `projection_matrix` 留在 CPU，而 R3 worker 直接把新 `world_view_transform` 建在 CUDA，导致 BMM
设备不一致。该 run 保留 `failed`，资源峰值远低于门槛，不属于方法或资源负结果。

修复在任何偏移或编辑前调用上游 `Camera.cuda()`，一次性迁移 image/depth/semantic/sky 与全部变换矩阵，
再深拷贝和修改外参；这与上游 `Camera.to()` 合同一致，不改变相机数值、checkpoint、renderer、指标或门槛。
以后 lazy evaluation path 必须把 camera 作为一个设备一致的整体处理，不能只迁移新创建的 tensor。

### V6-F10：structured array 拼接后必须使用字段索引，不能依赖 recarray 属性

R3 第五次正式目录
`20260821T091503Z__support-deviation-s20260821-r1` 已完成 2 场景 × 2 frontend 的全部 adapter、
checkpoint restore、横向/前向/actor-edit 渲染和 worker audit，共 80 个 render；汇总阶段把 `np.rec.fromarrays`
结果经 `np.concatenate` 拼接后得到 structured `ndarray`，代码仍以 `values.y/values.x` 访问字段，触发
`AttributeError`。该 run 的 terminal 保留 `failed`，不得倒写为 done；渲染与 checkpoint 证据本身完整。

修复统一改用 `values["y"]` 等 structured-array 字段索引。为避免无意义重跑冻结 renderer，建立独立
analysis-only recovery run：先逐个重算全部 render SHA、核对每个 worker 的 render count、checkpoint 前后 hash、
无训练/无确认集 audit 与 adapter 分区，再从只读失败目录计算指标；新目录记录原 run/commit/terminal hash、
分析 commit 与聚合 content hash，原目录不修改。以后 post-render 工程失败可复用已验证的不可变证据，
但必须新建 terminal 和完整 provenance，不能在原失败目录续写。

### V6-F11：Gaussian `source_indices` 是 chunk-local 身份，不能跨模型直接判全局唯一

R5 v0 正式目录 `20260821T094101Z__provenance-s20260821-r1` 已生成 provenance package，并达到
chunk=`24/24`、actor=`23/23`、primitive=`1,267,870/1,267,870` 覆盖；但 raw `source_indices` 的全局 unique
count 只有 `1,095,606`，因此 100% identity gate fail-closed。原因是 StreetGS Background 与各 Rigid model
分别维护局部 source-index 空间，actor 数值可与 Background 重叠；这不表示 primitive 丢失。

v0 run 与 config 保持 failed/frozen。v1 不放宽全局唯一门，而把 primitive identity 改为
`(chunk_id, source_index)` 复合键：先要求每个 chunk 内 source index 唯一，再要求 chunk id 唯一，二者合取形成
全局唯一身份。provenance 字段、source-type 分离、覆盖率和无 confirmation/训练边界均不变。以后任何跨 chunk
primitive registry 都必须显式携带命名空间，不能把上游局部索引误当全局主键。

### V6-F12：R7 source manifest 必须绑定实测完整 SHA，不能使用手工摘录值

R7 首次正式入口在创建 run 目录与读取 render payload 前 fail-closed：预注册配置中的 R3 recovery manifest
SHA 使用了手工摘录值 `e866dae3b84c...`，而冻结文件实测 SHA 为
`e866dae35a4ff17fb75791ff395f45504f8f779d57e75121354b8be388595acc`。两者不一致，因此程序按 source
identity contract 立即拒绝；没有 pseudo-hole、proposal、质量指标、训练或 confirmation 读取，也没有可写成
`rejected` 的方法结果。

修复只替换为 `sha256sum` 实测的完整 source manifest SHA；R7 hypothesis、cohort、hole 定义、verifier 阈值、
decoy、gate 与资源合同均不改变。以后冻结跨 run source identity 时，必须从机器可读 artifact 或现场哈希复制完整值，
不得从状态文档里的短写或人工记忆还原。

### V6-F13：R7 verifier 必须显式归一 frontend 的 singleton-channel 维度

R7 首个有 run 目录的正式实例
`20260821T100107Z__oracle-missing-world-s20260821-r1` 在第一个 pseudo-hole mask 构造时失败：StreetGS/AD-GS
冻结渲染对 depth/dynamic opacity 保留了 `H×W` 与 `H×W×1` 两种合法 singleton-channel 表示，初版代码无条件
取 `[...,0]`，把二维 opacity 错切成长度 `H` 的向量，随后与 `H×W` mask 广播失败。该目录 terminal 保持
`failed`；尚未形成完整 denominator、gate 或方法结论。

修复新增唯一的 plane normalization：只接受 `H×W`、`H×W×1` 或 `1×H×W`，统一返回二维数组，其他形状
继续 fail-closed；所有 depth/semantic verifier 和 usable-region 路径共同使用它。hypothesis、pseudo-hole、decoy、
阈值和 source render 均不改变。

### V6-F14：actor/disocclusion pseudo-hole 不能假设每个 frontend 都导出非空 dynamic opacity

R7 第二个有 run 目录的正式实例
`20260821T100228Z__oracle-missing-world-s20260821-r1` 已完成 scene-0242 与部分 scene-0048 cases，随后在
scene-0048/StreetGS 的 disocclusion mask 上 fail-closed：该冻结 renderer 的 `dynamic_opacity` 在此帧为空，初版
mask 得到 0 pixels，低于预注册 `256` denominator。该目录保持 `failed`，不汇报不完整 gate。

修复不降低 minimum pixels，也不读取 confirmation，而是使用本实验本来就冻结的 `base` 与
`actor_remove_all` 配对渲染：以 RGB 变化或同 frontend 可比 depth 变化，加上可用的 dynamic opacity，形成 actor
evidence；disocclusion 对其确定性膨胀，actor-removal hole 使用原 evidence。这样 StreetGS/AD-GS 都以“实际 actor
edit effect”而不是可选 buffer 的存在作为 denominator，其他 hole、verifier、decoy 与 gate 不变。

### V6-F15：actor pseudo-hole denominator 必须先满足非空 actor-effect 证据

R7 第三个失败目录 `20260821T100348Z__oracle-missing-world-s20260821-r1` 在使用 RGB/depth actor-edit evidence 后，
scene-0048/StreetGS/frame-52 的 disocclusion mask 仍严格为 0。R3 冻结 `ACTOR_EDIT_EFFECTS.jsonl` 证实该 frontend
在 scene-0048 的 frame 52/57 上，`actor_remove_all`、translate、time-shift 的 global effect 与 nonzero fraction
全部为 0；因此这四个 actor-removal/disocclusion Cartesian cases 没有可构造的真实 pseudo-hole denominator。

这不是 oracle verifier 的负结果。预注册 `WS-V6-H-R7-001` 因 32-case minimum experiment 无法实例化而标记
`invalidated_pre_gate`，不倒写 gate。替代假设 `WS-V6-H-R7-002` 在 proposal 评分前冻结 eligibility：route/side
仍保留全部 16 cases；actor/disocclusion 要求原冻结 evidence 至少 256 pixels，不合格的四项显式记录 structural
ABSTAIN；剩余 28 oracle + 28 decoy 才进入完全相同的 verifier/bake 门禁。

### V6-F16：冻结 LaMa 配置必须用 OmegaConf 解析字段引用

R8 首轮正式目录 `20260821T103805Z__frozen-generator-s20260821-r1` 中，Big-LaMa 在构造
FFC generator 时失败。官方 `config.yaml` 的 `downsample_conv_kwargs` 与
`resnet_conv_kwargs` 使用 `${generator...}` 字段引用；直接用 PyYAML 读取会把引用保留为
字符串，最终在通道比例转整数时触发 `ValueError`。该候选被如实记为 `failed`，首轮 gate
保持 `rejected`，不改写为方法结论。

修复只把 generator 子树改为 `OmegaConf.load` 后 `resolve=True`，仍使用同一官方配置、源码、
checkpoint、输入、seed、阈值与选择规则；不解析或消费任何训练/确认数据。以后复用带插值的
冻结配置时，必须在进入模型构造前解析并审计最终标量，不能把 YAML 语法读取成功当成配置已实例化。

### V6-F17：扩散 inpaint 输出尺寸必须显式绑定冻结输入尺寸

同一 R8 首轮中，SD-v1.5 pipeline 虽完成模型加载和推理，但未显式传入 `height/width`，输出采用
默认 `512×512`，与冻结输入及 mask 的 `512×288` 不一致，overlay 布尔索引因此触发
`IndexError`。该候选同样保留 `failed`，不是 GPU、权重或方法负结论。

修复是在冻结 pipeline 调用中显式设置 `height=image.shape[0]`、`width=image.shape[1]`；不 resize
实验输入、不改变 mask、prompt、steps、guidance、seed、候选或门槛。后续所有生成器 adapter
必须把空间尺寸当作调用合同，并在 compositing 前 fail-closed 核对 image/mask/proposal 三者形状。

### V6-F18：推理 adapter 不得反序列化 checkpoint 的训练器对象

R8 第二轮正式目录 `20260821T104107Z__frozen-generator-s20260821-r1` 中，SD-v1.5 已完整通过
4 cases × 2 repeats 的 capability/resource gate，但 Big-LaMa 在 `torch.load` 时尝试恢复 checkpoint
中未参与推理的 Lightning callback，因轻量推理环境没有完整训练框架而失败。冻结规则要求两个 ungated
候选都被实际执行，所以该轮仍如实 `rejected`；SD 的通过不能事后放松这条规则。

修复改用 PyTorch `weights_only=True`，只读取 tensor/state_dict，再按冻结 generator 结构严格加载；
不安装或执行训练器，不改变 checkpoint 字节、候选、案例、输入、seed、阈值、资源合同或选择规则。
以后第三方训练 checkpoint 的推理入口必须默认最小反序列化面，训练回调、优化器和日志对象不得成为
部署环境的隐式依赖。

### V6-F19：weights-only 仍需显式白名单化归档中的非张量全局类型

R8 第三轮正式目录 `20260821T104241Z__frozen-generator-s20260821-r1` 中，Big-LaMa 的
`weights_only=True` 正确拒绝了归档内未白名单化的
`pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`；SD-v1.5 再次完整通过，但
双候选执行门仍未满足，所以该轮保持 `rejected`。这不是显存或模型能力失败。

修复为该精确全局名注册无方法、无训练行为的占位类型，并只把该类型加入 PyTorch safe globals；
checkpoint 继续以 `weights_only=True` 读取，后续仍只消费 `state_dict`。不引入 Lightning 训练栈，
不执行 callback，也不改变任何正式实验变量。若归档再暴露未预注册类型则继续 fail-closed，禁止切回
不受限反序列化来绕过门控。

### V6-F20：safe-global 修复必须基于静态完整清单，不能逐异常猜测

R8 第四轮正式目录 `20260821T104432Z__frozen-generator-s20260821-r1` 在白名单化
`ModelCheckpoint` 后继续 fail-closed，下一项未注册类型为 `omegaconf.dictconfig.DictConfig`；
SD-v1.5 仍完整通过而双候选执行门仍失败。逐次靠异常暴露类型会无意义地重复正式运行。

修复先用 `zipfile + pickletools` 静态读取官方 checkpoint 的 `data.pkl` GLOBAL 指令，不执行
反序列化；完整非内建清单只有 OmegaConf 的 `ContainerMetadata/Metadata/DictConfig/ListConfig/AnyNode`、
`typing.Any` 和已知 `ModelCheckpoint`，其余为 PyTorch tensor/标准容器重建函数。adapter 一次性白名单化
这些精确类型后仍使用 `weights_only=True`，实验变量与选择规则不变。以后同类归档应先做静态类型清单，
再建立最小安全白名单，避免把正式 run 当依赖探针。

### V6-F21：Python 2 pickle 内建名迁移后也必须进入 safe-global 清单

R8 第五轮正式目录 `20260821T104627Z__frozen-generator-s20260821-r1` 中，OmegaConf 类型完成
白名单后，weights-only loader 继续拒绝由旧归档 `__builtin__.dict` 迁移得到的 `builtins.dict`；
SD-v1.5 再次通过，双候选执行门仍保持 `rejected`。静态 GLOBAL 清单此前列出了旧模块名，但未把
Python 3 运行时映射后的内建类型显式加入 safe globals。

修复补入清单中出现的标准容器及迁移类型：`dict/list/int/OrderedDict/defaultdict`；仍不对白名单外
类型开放，不切换为 unrestricted pickle。模型、checkpoint、输入、seed、阈值与选择规则不变。
以后审计跨 Python 版本的 checkpoint 时，GLOBAL 清单必须同时记录归档名与当前运行时解析后的类型名。

### V6-F22：跨阶段 source manifest 必须复制机器实测完整 SHA

R9 首次正式入口在创建 run 目录前 fail-closed：冻结配置把 R7 manifest 的
`73dbb2ba11bc12bc...` 手工误抄为 `73dbb2ba11bc4e22...`。现场 `sha256sum` 与 R7 closeout 均确认
源文件仍为 `73dbb2ba11bc12bc4e22ca13765af10d14ac5d183e1d529add5e6d619f2a4d0c`；因此没有 proposal、
verifier、hidden target、训练、confirmation 或方法结果产生。

修复只替换为机器实测的完整 source manifest SHA，不改变 R9 hypothesis、模型、cohort、arm、阈值、
gate 或资源合同。这是与 V6-F12 同类的 provenance 抄录错误；后续跨 run 配置应由 manifest artifact
自动生成，禁止再次从缩写或人工记忆恢复完整哈希。

### V6-F23：冻结 semantic checkpoint 必须 strict 重建 auxiliary head

R9 首个有 run 目录的正式实例 `20260821T110446Z__independent-arms-s20260821-r1` 已先生成全部
28 个 Big-LaMa proposal，随后 semantic worker 在 strict load 时拒绝 checkpoint 中的
`aux_classifier.*`：初版 adapter 以 `aux_loss=False` 构造 DeepLabV3，遗漏了训练 checkpoint 保留的官方
auxiliary head。run 保持 `failed`，未产生 arm verdict、融合或 bake，也不是模型质量/资源负结论。

修复只以 `aux_loss=True` 重建同一 19-class DeepLabV3-ResNet50，并继续 strict load 全部参数；正式推理
仍只消费主输出 `out`，aux head 不参与 arm score。权重字节、cohort、proposal、threshold、gate 与资源合同
均不改变。以后冻结视觉 checkpoint 的结构审计必须覆盖所有 state-dict head，不能以“推理不消费”为由
在 strict identity 前删除参数。

### V6-F24：第三方 checkpoint 的主 head 与 auxiliary head 类数可能不一致

R9 第二个正式 run `20260821T110616Z__independent-arms-s20260821-r1` 在启用 aux 结构后继续由
strict load 拒绝：归档主 classifier 为 Cityscapes 19 类，但 `aux_classifier.4` 仍是 torchvision 默认
21 类（权重 `21×256×1×1`），而统一 `num_classes=19` 构造出的 aux head 为 19 类。run 保持
`failed`；28 proposals 已生成，但无 verifier verdict、融合或 bake。

修复精确重建归档结构：主 head 保持 19 类，aux 最后一层单独恢复为 21 类，然后 strict load 全 state dict；
aux 输出仍不被正式分数消费。模型权重、P3 动态类定义、cohort、threshold、gate 和资源合同均不改变。
以后第三方 segmentation checkpoint 必须逐 head 审计 shape，不能假设所有 classifier 共用同一 label count。

### V6-F25：capability 最优的 generator 不等于 verifier-arm 质量最优

R9 canonical rejected run `20260821T110743Z__independent-arms-s20260821-r1` 在 28 个 matched
development pseudo-holes 上证明 Big-LaMa 虽是 R8 的资源最优候选，但没有 verifier arm 可进入 R10：P1/P2
均 `0/28` ACCEPT；P3 在 12 个 actor-evidence cases 中接受 6 个，却有 1 个 false-safe，率 `1/6=0.1667`
高于冻结 `0.10`；P4 正确 `28/28` ABSTAIN。P0 photo/geometry false-safe 均为 `1.0`，P3 为
`0.5833`。outside-mask exact、无融合/无 bake/无 confirmation 均通过，峰值仅 `428 MiB`，不是资源失败。

H-R9-001 正确裁决为 `rejected`，不得放宽 photo/depth/semantic 阈值或把全拒绝写成有效 verifier。
新 H-R9-002 只切换到 R8 已完成 capability gate 的第二候选 SD-v1.5；沿用完全相同的 28-case cohort、
hidden observations、P1–P4、truth 定义、threshold、gate、模型和资源上限。Big-LaMa 与 SD 结果必须分属
独立不可变 run，禁止在看到 SD 结果后混合选择 per-case generator。

### V6-F26：单图生成候选无法把不可观测 missing-world 内容变成可验证事实

H-R9-002 canonical rejected run `20260821T111228Z__independent-arms-s20260821-r1` 用冻结
SD-v1.5 替换 Big-LaMa，并保持所有 verifier 与 gate 不变。P1 仍 `0/28` ACCEPT；P2 仅 `2/28=0.0714`
且低于冻结 `0.10` coverage；P3 与 Big-LaMa 同为 `6/12` ACCEPT、`1/6=0.1667` false-safe；P4
`28/28` ABSTAIN。outside-mask exact，峰值 `2696 MiB`，无融合、bake、训练或 confirmation。因此
H-R9-002 同样是方法质量 `rejected`，不是工程/资源 blocked。

两种单图 inpainting 都失败后，不得调松阈值、按案例混选生成器或转向 gated 23.8GB FLUX 权重来规避
负结果。新 H-R9-003 将唯一变量改为冻结 cross-frontend reconstructed proposal：同 scene/frame/edit variant
使用另一 frontend 的对齐 RGB 填入 mask；P1–P4、truth、threshold、gate 和 denominator 原样保留。该 proposal
仍标记 reconstructed，两个 frontend 来自同一传感器支持，不能解释为新增观测或独立 ground truth。

### V6-F27：正式入口必须显式建立仓库模块搜索路径

R12 第一次启动命令在创建 run 目录、加载模型或执行 GPU 推理前失败：直接运行
`python scripts/worldsim_v6/run_logsim.py` 时，Python 只把脚本目录加入模块搜索路径，因而无法导入仓库根目录下的
`motion_proj` 包并抛出 `ModuleNotFoundError`。该失败没有产生样本、指标或方法结论，也不是资源失败。

修复只在入口脚本中根据 `__file__` 把仓库根目录加入 `sys.path`，不改 R12 hypothesis、cohort、输入哈希、模型、
阈值、gate 或资源合同。后续以完全相同命令重跑；任何模型或指标失败仍独立登记，不能用本次入口错误掩盖。

### V6-F28：启用 CUDA 确定性算法前必须冻结 cuBLAS workspace 配置

R12 首个有 run 目录的正式实例 `20260821T114117Z__logsim-s20260821-r1` 已完成两项静态 chunk 的 CPU
重放构造并成功严格加载冻结 DeepLab checkpoint，但第一次 GPU forward 被 PyTorch fail-closed：代码启用了
`torch.use_deterministic_algorithms(True)`，CUDA 10.2+ 的 cuBLAS 路径还要求进程启动前设置
`CUBLAS_WORKSPACE_CONFIG=:4096:8`。该 run 保持 `blocked`，没有感知输出、完整 gate 或方法结论；也没有发生 OOM。

修复只在启动感知子进程前加入这一确定性环境变量，继续保留 deterministic algorithms、同一 checkpoint、4 个输入、
同一 cohort、阈值与资源上限。不得关闭确定性检查来换取通过；修复后新建独立 run 重试。

### V6-F29：世界空间 z-buffer 必须把空视锥投影作为有效零覆盖结果

R13 首个正式 run `20260821T120059Z__worldspace-route-s20260821-r1` 已完成两个 verified chunk 的世界坐标
提升，但部分大幅路线偏离没有任何点落入目标视锥。初版 z-buffer 仍为零长度索引构造了长度 1 的首元素布尔 mask，
触发 `IndexError`。该 run 保持 `blocked`，没有完整 baseline matrix、gate 或方法结论，也不是资源失败。

修复只在 z-buffer 中对零个可见点直接返回空的 x/y/z/source-index；下游按预注册协议记录
`projected_pixel_count=0`、指标不可用且 route-support fail。不得删掉这些偏离、降低分母或将空投影改写成 ABSTAIN。
其余输入、深度、标定、四方法、阈值、gate 和资源合同均不变，并以独立 run 重试。

### V6-F30：WorldSim evaluator 必须统一合法的 singleton-channel depth plane

R13 第二个正式 run `20260821T120208Z__worldspace-route-s20260821-r1` 在加载偏离路线的 StreetGS depth 时
fail-closed。该 renderer 保存合法的 `H×W×1` float depth，而 evaluator 的 PIL resize 入口只接收 `H×W`，
因此抛出 `TypeError`。run 保持 `blocked`，尚无完整 48-row baseline matrix 或 gate；没有 GPU/内存问题。

修复在 resize 前只接受 `H×W` 或 `H×W×1`，后者显式去掉最后 singleton channel；其他形状继续拒绝。
这与 V6-F13 的 plane normalization 原则一致，但本次记录覆盖独立的 R13 evaluator。不得改 depth 数值、插值模式、
样本、四方法、阈值或 gate，修复后以新 run 重试。

### V6-F31：verifier 相对深度不得直接解释为 WorldSim 米制相机 z

H-R13-001 canonical rejected run `20260821T120310Z__worldspace-route-s20260821-r1` 成功把两个 R11 chunk
封装为 58,273 个所谓世界点，但估计总面积仅 `0.01993 m²`，且 12/12 个非零路线偏离均为零投影覆盖，
usable lateral route 为 `0.0 m`。V6 的 matched false-safe 仍为 `0/3`，相对 naive 的降低为 `0.8214`，
所以拒绝原因不是安全 gate，而是 R9 depth 只为仿射对齐后的 verifier 几何比较服务，不能直接当作米制 z 做相机平移。

H-R13-001 按预注册门槛正式 `rejected`，不得通过放宽 256-pixel、0.12 photo 或 0.30 geometry 门槛恢复。
H-R13-002 只替换深度来源：使用同帧冻结 logged LiDAR metric depth，并限定最近填充距离不超过 8 个
512×288 像素；世界提升、四方法、12 个偏离、评估阈值、false-safe gate 和资源合同全部保持不变。

### V6-F32：无可见性约束的关键帧点云 union 会放大遮挡错误而非扩展路线

H-R13-003 canonical rejected run `20260821T121112Z__worldspace-fusion-s20260821-r1` 将两个 metric world chunk
按冻结 5cm voxel 做无目标视角过滤的 union。57,997 个输入点因近表面重复被折叠为 5,868 点，没有形成预期 densification；
共同 lateral route 从 `3.0m` 退化到 `2.0m`，5m mean geometry MRE 从 `9.0572` 升至 `10.6579`，
相对变化为 `-17.67%`，两帧 5m 继续失败。因此假设正式 `rejected`，不是工程或资源 blocked。

不得靠扫描更小 voxel 或放宽 photo/geometry 门槛复活该 union。H-R13-004 转向不同机制：保持 H-R13-002 的
逐帧 world points 和 RGB，不做 union/densification；只用同一冻结 support 中三相机 logged LiDAR 投影到目标视角，
在 4 像素邻域和 0.30 相对深度差内保留可见点。StreetGS truth proxy 只用于最终评估，不进入过滤。

### V6-F33：跨阶段安全摘要必须读取冻结 schema 的完整方法键

H-R13-004 首个正式 run `20260821T121825Z__worldspace-visibility-s20260821-r1` 已完成两次目标视角
LiDAR 投影和全部 12 个偏移的指标计算，但在汇总继承 H-R13-002 的 V6 false-safe 时 fail-closed。冻结摘要使用
`baseline_safety.v6_generate_verify_bake.false_safe_rate`，初版 runner 却读取了不存在的缩写键
`baseline_safety.v6.joint_false_safe_rate`，因此抛出 `KeyError`，run 保持 `blocked`，没有形成 gate 或方法结论。

修复只按冻结摘要的实际 schema 读取完整方法键和 `false_safe_rate` 字段；不改 world points、目标视角 LiDAR、4 像素/
0.30 visibility 合同、12 个偏移、质量阈值、false-safe 数值或资源合同。以后跨阶段消费结构化摘要时，必须把完整方法标识和
字段名纳入配置/manifest 合同，禁止用人工缩写推断 schema。

### V6-F34：删除式可见性筛选不能补齐远路线新暴露表面，且 5m 深度代理失去米制有效性

H-R13-004 canonical rejected run `20260821T121956Z__worldspace-visibility-s20260821-r1` 使用冻结三相机
logged LiDAR 在每个目标视角执行 4 像素/0.30 相对深度筛选。它保持共同 lateral route `3.0m`、精确复跑、源不可变和
V6 false-safe `0.0`，但两帧 5m 几何 MRE 仍为 `9.0323/9.4071`，photo MAE 为 `0.1404/0.1495`，
两帧均失败；5m 仍保留约 `78%` 的旧点，说明仅删除矛盾点并未提供新暴露表面。

独立诊断还显示，5m 目标视角投影后的实测 LiDAR 中位 z 约 `13.17/13.34m`，而 StreetGS 目标 depth proxy 中位数仅
`1.30/1.22`，该 proxy 在此外推距离不能作为米制几何真值。不得扫描 visibility 阈值或放宽 photo/geometry gate 来追逐
这个失效代理。路线偏移结论固定为 H-R13-002 已验证的 lateral `3m`、forward `2m`；后续直接进入计划尚未覆盖的 actor
add/remove、trajectory modification 与 traffic-density typed edit 实验。

### V6-F35：跨 run 回放内容等价比较必须排除非语义 repeat 序号

H-R13-005 首个正式 run `20260821T122830Z__dynamic-edits-s20260821-r1` 的三个 V6 typed edit 均通过全部
编辑、依赖闭包、时序和精确复跑检查，但总 gate 因 `base_matches_frozen_r12_replay=false` 保持 `rejected`。逐字段定位确认
唯一差异是当前重新加载调用使用 `repeat_index=0`，而冻结文件 `DYNAMIC_REPLAY_REPEAT1.json` 记录 `repeat_index=1`；
`replay_content_sha256` 及 actor、trajectory、semantic、collision、sensor、event 全部内容一致。

修复只在跨 run 内容等价比较的两侧移除非语义 `repeat_index`，仍严格比较冻结内容 hash 和所有功能字段；不改三个 edit、
四方法臂、actor/时间戳分母、碰撞计算、false-safe、资源合同或任何阈值。repeat 序号继续保留在各自运行记录内，但不得被当作
compiled-world 内容漂移。

### V6-F36：浮点 renderer RGB 归一化必须容忍轻微大于 1 的辐射 overshoot

H-R13-006 首个正式 run `20260821T123716Z__actor-sensor-perception-s20260821-r1` 完成 16 次 DeepLab
推理且精确复跑，但 AD-GS 两个 case 的 target RGB effect 被错误压到约 `0.00035`，继而使感知变化近零。StreetGS 两例保持
约 `0.09` target effect。根因是 AD-GS 的归一化浮点 RGB 存在轻微 `>1` overshoot，初版主 runner 把它误判为 0--255
后再除 255；worker 同样没有把它放大到 uint8 动态范围。该 run 的 AD-GS 指标无效，不能作为方法 rejection。

修复把浮点 renderer 合同明确为最大值 `<=2.0` 时仍按归一化辐射值处理：主 runner 直接 clip 到 `[0,1]`，perception
worker 乘 255 后 round/clip 为 uint8；只有明显大于 2 的数组才按 0--255 输入。冻结 render 字节、模型、case、mask、
阈值、repeat、资源和 gate 全部不变。以后不同 frontend 的 RGB 必须由显式 range contract 归一化，禁止用严格 `max<=1`
启发式区分编码。

### V6-F37：局部 RGB 编辑不保证全帧感知输出局部，宽感受野必须进入 verifier 因子设计

H-R13-006 canonical rejected run `20260821T123835Z__actor-sensor-perception-s20260821-r1` 在修复 RGB
range 后确认 StreetGS/AD-GS 两帧共 4 个 actor-remove case 均具有强 sensor locality：target RGB MAE
`0.0872--0.0954`、outside RGB MAE `0.000175--0.000399`、locality enrichment `224--498x`，16 次
DeepLab 推理精确复跑且峰值仅 `688MiB`。但全帧 DeepLab 在 target 外仍改变 `6.34%--10.31%` 标签，4 个 case
只有 AD-GS frame57 达到 2x perception locality，故假设按预注册 gate 正式 `rejected`。

不得把 outside 2% 或 enrichment 2x 阈值调松，也不得因 RGB 局部就宣称 perception failure 已解决。H-R13-007 改用
factorized ROI 机制：固定 256px tile/128px candidate stride，只根据 logged dynamic opacity 选择最高 actor fraction target
和与其不重叠的最低 actor fraction static tile，对两者独立执行冻结模型；full-frame rejection 永久保留，不被 ROI 结果覆盖。

### V6-F38：remove-all 编辑不能为 factorized perception 提供无 actor 的静态 ROI 对照

H-R13-007 canonical rejected run `20260821T124443Z__factorized-perception-s20260821-r1` 按冻结
256px tile/128px stride，在每个 frontend/frame 选择最高 actor fraction target 与不重叠的最低 actor fraction static。
四个 target 的 actor fraction 为 `0.753--0.936`，但四个所谓 static 仍为 `0.099--0.373`，全部超过预注册
`0.01` 上限；static RGB MAE 也为 `0.0095--0.0267`，证明 remove-all 操作本身横跨全图，而不是 tile 选择偶然失败。

不得扫描 tile 大小/stride 或放宽 static denominator。H-R13-008 改变真正的因果变量：利用 StreetGS 冻结 checkpoint 的
per-Gaussian `point_ids` 与 `instances_fv`，只删除在两帧均可见且 Gaussian 数最多的单个 model actor；actor 选择不读取
RGB/semantic outcome。冻结 AD-GS 不保留可审计 per-actor ID，必须 ABSTAIN，不得伪造跨 frontend 单 actor 对齐。

### V6-F39：actor Gaussian 数量不等于下游感知敏感度，单启发式选择必须扩展为完整分母

H-R13-008 canonical rejected run `20260821T125151Z__single-actor-perception-s20260821-r1` 从两帧均可见的
12 个 StreetGS model actor 中，按最大 Gaussian 数且最小 index 的预注册规则选中 index `2`（`13,490` Gaussians）。
logged rerender 对冻结 R3 RGB 的 MAE 为精确 `0`，单 actor 删除的 effect pixel 为 `10,271/9,047`，target RGB
MAE 为 `0.0210/0.0286`，outside RGB MAE 仅约 `1e-6`；但 target DeepLab label change 只有
`0.00068/0.0`，两帧都未达到冻结 2% 感知效应门。

该负结论拒绝“Gaussian 最多 actor 最能触发 perception”的启发式，不得改选第二大 actor 当作同一假设 recovery，也不得放宽
2% threshold。H-R13-009 一次性评估冻结 metadata 定义的全部 12 个 eligible actor，以固定完整分母报告 ACCEPT/ABSTAIN
覆盖率；只有两帧都通过完全相同门槛的 actor 才可被 V6 接受，其余必须显式 abstain。

## 7. 历史新路线启动前附加检查

- [ ] 是否明确说明该步骤直接服务于重建、编辑或可信评测，而不是重新做事件挖掘？
- [ ] AD-GS exact reproduction 是否已经通过冻结门禁？
- [ ] 是否把 upstream 原始结果与 compatibility patch 结果分开？
- [ ] 是否对 VAD-GS 等已公开的 visibility/completion 工作做 novelty 边界核对？
- [ ] 反事实无真值指标是否有真实 held-out/pseudo-hole 证据，而不是自洽规则？
- [ ] 是否同时评估目标区变化、非目标区保持、几何/时序一致性和下游感知？
- [ ] 遇到内存/GPU不足时是否按 `N1-F24/PIVOT-F05` 停机并等待授权？

## 8. WorldSim 后续正式消融前检查

- [ ] 是否使用 V3 task ID、新 run 和冻结 config/source hash，而不是续写 V2 terminal？
- [ ] 是否保持 scene-0230/0242/0255、split、seed、相机、步数和 actor cohort 不变？
- [ ] 是否把原生 Affine/CamPose/LiDAR init 与新增实现分开？
- [ ] rolling-shutter 路径是否有真实 row timing；没有时是否显式 `not_supported`？
- [ ] actor-aware 变化是否只增加一个可归因因子，并保留 module-off 原生等价测试？
- [ ] 是否同时报告 actor/boundary 质量、GS 数、训练时间、VRAM 和 non-target 保持？
- [ ] local refinement 是否冻结 affected set 外参数，并只使用 Tier-A/多视图/LiDAR 可观测证据？
- [ ] expected/first-hit/measured depth 是否继续保持 typed separation？
- [ ] 工程 `blocked`、方法负结果 `rejected` 和任务完成 `done` 是否没有混写？
- [ ] 结论是否明确限制在三场景消融，不写成大规模泛化或闭环安全结论？
- [ ] 新路线是否只选择一个 primary hypothesis，并说明它具体解除 `V3-F18`–`V3-F25` 中哪一项？
- [ ] 是否在任何训练、推理或新结果读取前冻结 matched baseline、主端点、资源门、停止条件和确认场景？
- [ ] 是否避免把更小剪枝 fraction、提高旧资源 ceiling、全量读取 chunk 或继续调 R1 配方伪装成新研究？

### V6-F40：可见 actor cohort 分母不得替代 SceneIR 全量 actor 分母

H-R13-010 canonical run `20260821T130810Z__sceneir-sensor-binding-s20260821-r1` 正确完成同一 scene-0242 checkpoint 的 SceneIR 编译、model index `0` 到 `actor_0000` / `streetgs_actor_0000` / `12,390` primitives 的绑定、actor remove、两次 fresh replay，以及未受影响 actor state、trajectory、semantic label 和 collision pair 的精确保持；继承的 H-R13-009 V6 false-safe 仍为 `0`。但是 preregistration 把 H-R13-009 中“两帧均可见”的 `12` 个 actor cohort 错当成 checkpoint 转换后的 SceneIR 全量 actor 数，实际冻结 converter 输出为 `27` 个 actor；因此预注册的 `15→14`、`2940→2744`、`20580→17836` 与实际 `27→26`、`5292→5096`、`68796→63700` 不符，typed dependency-closure gate 按约定拒绝。

该 run 保持 `rejected`，不能用其余检查通过来覆盖错误分母。恢复假设 H-R13-011 只把全量分母来源改为编辑前冻结 checkpoint 的 deterministic SceneIR converter 输出，并预注册上述实际总量；不修改 checkpoint、actor mapping、edit、replay、quality threshold、继承 verdict、资源合同或 unsupported claim。以后必须把 visibility/evaluation cohort 与 compiled-world total denominator 分别命名和冻结。

### V6-F41：下游 regression consumer 必须冻结上游 gate 的完整 decision 值

H-PT1-001 首个 formal run `20260821T132127Z__regression-utility-s20260821-r1` 在读取 scene-0230 development contract 时 fail-closed。上游 H-R13-005 gate 的实际 decision 是 `accept_typed_dynamic_edit_dependency_closure`，初版 consumer 却硬编码了缩写 `accept_typed_dynamic_edits`，因此在读取 heldout replay、构造四类 mutation 或计算三臂 quality metric 前即抛出 `PT1RegressionError`；该 run 只有 `TERMINAL.json`，不得产生方法结论。

修复把 scene-0230 与 scene-0242 两个上游 gate 的完整 accepted decision 值写入冻结 config，consumer 只按 config 精确比较。不得改变四类 stale-factor mutation、三方法臂、false-safe/detection gate、source hash、资源合同或 unsupported claim；修复后以新 run 重试 H-PT1-001。跨阶段消费者以后不得从 task 名或人类缩写猜测 structured gate value。

### V6-F42：policy 输入不得直接包含 verifier 的 decision statistic

H-PT2-001 canonical run `20260821T133158Z__risk-policy-s20260821-r1` 的数值 gate 全部为真，V6 arm 在 scene-0255 heldout 上达到 balanced accuracy `1.0`、false-safe `0`、safe-route completion `1.0`，并把 naive stale-label arm 的 false-safe 从 `1.0` 降为 `0`。但是 Real-only arm 同样达到 balanced accuracy `1.0` 和 false-safe `0`，尽管其 98 条训练行的 positive fraction 为 `0`。原因是 policy 直接接收 signed AABB clearance，而 hazard label 正是 `clearance<=0`；这等于把 verifier 判定边界编码进输入，Real-only 只需把阈值放在最小 logged clearance 以下就会偶然分开固定 synthetic offsets。

因此该 run 只保留“naive stale labels 有害”的诊断，不晋级为 incremental post-training utility；初版 gate 缺少对 Real-only 的增益约束也是方法治理缺口。H-PT2-002 保持三场景、frame 分母、clone offsets、label、heldout 与任务指标不变，移除 signed clearance/AABB extent/hazard verdict，只向 policy 提供原始绝对 ego-relative forward/lateral position；固定 axis-aligned rectangle ERM 候选网格，并新增相对 Real-only 与 naive 两者 false-safe 至少降低 `0.50` 的 gate。以后任何 learned verifier/policy 实验必须审计 feature 是否直接重编码 label rule。

### V6-F43：声明值经坐标变换后必须做远小于物理阈值间隔的数值 canonicalization

H-PT2-002 canonical run `20260821T133626Z__risk-policy-s20260821-r1` 在移除 signed-clearance feature leakage 后，使 Real-only 与 naive arm 都成为 constant CONTINUE，heldout false-safe 均为 `1.0`；V6 raw-position rectangle policy 把 false-safe 降至 `0.1122449`，safe-route completion 保持 `1.0`，但 balanced accuracy `0.9438776` 与 false-safe 仍未达到冻结的 `0.95` / `0.05` 门，故方法正式 `rejected`。

逐行诊断确认 11 个漏检全部是配置声明的 `3.0m` clone：`inv(T_ego) @ (T_ego @ T_offset)` 后 raw forward feature 变成 `3.00000000000011–3.00000000000045`，严格 `<=3.0` 比较失败；没有其他 hazard 漏检，也没有 false brake。H-PT2-003 只在 policy raw feature 写入前按 9 位小数 canonicalize（`1e-9m`），label 仍按未取整 signed clearance 计算；该尺度高于浮点漂移、但比最小候选阈值间隔小至少九个数量级。不得改样本、offset、threshold grid、label 或质量门。以后由声明变换生成的控制量必须把表示 canonicalization 与方法容差分开冻结。

### V6-F44：单一 zero-lateral intervention 训练不能支持二维风险泛化

H-PT3-001 canonical run `20260821T134426Z__intervention-robustness-s20260821-r1` 冻结 H-PT2-003 的全部 policy 参数，在未参与 PT2 训练或选择的 scene-0048 上评估 forward `1.5/4.5/7.5m` × nonzero lateral `0.75/1.5/2.5m` 的 441 个 edit rows，加 49 个 clean rows。分母含 232 hazards / 258 safe；冻结 V6 policy 的 lateral threshold 为 `0.0m`，因此对 232 hazards 检出 `0`，false-safe `1.0`，与 Real-only/naive constant policies 完全相同。source immutability、repeat exact、safe-route completion 均通过，所以是方法性 `rejected`。

不得把 PT2 的跨场景同 intervention pass 写成二维 policy generalization，也不得只把冻结 lateral threshold 放宽。H-PT3-002 改变训练 evidence：在 scene-0230/0242 使用 forward `0/2/4/6/8m` × lateral `0/1/2/3m` 的完整 factorized typed-edit grid，V6 重算标签、naive 保留 stale labels；scene-0048 使用与训练离散的 half-offset grid，所有质量门不变。以后 policy coverage 必须按 intervention factor denominator 报告，不能只报 scene denominator。

### V6-F45：最近 actor 的位置不足以表达 box overlap，必须保留 factorized extent

H-PT3-002 canonical run `20260821T134843Z__factorized-policy-training-s20260821-r1` 在 scene-0230/0242 完成 1,960 条二维 typed clone train rows，并在 scene-0048 的 980 条离散 half-offset edits 上评估。V6 相对两基线把 false-safe 从 `1.0` 降至 `0.4505208`、safe-route completion 保持 `1.0`，但 balanced accuracy 仅 `0.7747396`，未过冻结门，正式 `rejected`。

诊断显示 position-only policy 为每条 episode 只保留 signed-clearance 最小 actor 的 `|x|/|y|`，却丢掉 projected box half-extent 与 yaw；远处 safe clone 会由更近但窄或旋转的真实 actor 取代特征，同一位置因此对应不同 overlap label。训练 lateral `2.0m` 恰在默认 box 边界，未 canonicalize 的 factor label 还出现 66/32 等浮点混合。H-PT3-003 保持数据、grid、三臂、heldout 和质量门，新增 raw projected half-extents，按 1e-9m canonicalize forward/lateral factor labels，分别训练两个 logistic overlap heads 后 AND；不得输入 signed clearance 或最终 hazard verdict。以后几何 policy 的 factor representation 必须保留决定接触边界的尺寸/朝向信息。

### V6-F46：单一 synthetic box size/yaw 无法识别可迁移的 extent 系数

H-PT3-003 canonical run `20260821T135421Z__factorized-policy-training-s20260821-r1` 使用 raw `|x|/|y|` 与 projected half-extents，分别训练 forward/lateral logistic overlap heads。V6 train overall balanced accuracy 为 `1.0`，forward/lateral factor train accuracy 为 `1.0/0.9986`，但在 disjoint scene-0048 grid 上 balanced accuracy `0.7963`、false-safe `0.22135`、safe-route completion `0.81395`，未过冻结门，正式 `rejected`；两基线 false-safe 均为 `1.0`。

这不是优化未收敛，而是 synthetic train 全部采用默认 `4.5x2.0m`、relative yaw `0°`，projected extent 几乎常数，position 与 extent 的相反系数无法由 synthetic positives/negatives 识别，只能依赖稀疏真实 box 分布，换场景即漂移。H-PT3-004 保留 factor-head 架构、loss、task gates 与场景，扩展训练 denominator 为三种 size × 四种 yaw × 原 position grid；heldout 使用完全离散的三种 size × 三种 yaw。不得输入 signed gap 或固定解析碰撞公式来伪装 learning gain。以后 intervention coverage 必须同时报告 position、size 与 orientation factors。

### V6-F47：factor head 的训练目标必须与冻结的 balanced 指标对齐

H-PT3-004 canonical run `20260821T135942Z__factorized-policy-training-s20260821-r1` 在 23,520 条 multi-size/multi-yaw train interventions 与 8,820 条完全离散 heldout interventions 上运行。V6 把 false-safe 从 Real-only 的 `0.97133`、naive 的 `1.0` 降至 `0.14821`，但 balanced accuracy `0.85613`、safe-route completion `0.86047`，仍未通过冻结的 `0.90/0.10/0.90` 门，因此正式 `rejected`。

训练的 lateral factor 正例比例为 `0.88174`，原始 unweighted BCE 按出现频率主导梯度，与 task 从一开始冻结的 balanced accuracy/false-safe/completion 三目标不一致；这会同时留下 `14.82%` false-safe 与 `13.95%` false brake。H-PT3-005 保持场景、position/size/yaw 分母、raw feature、标签、三臂、步数和所有 gate 不变，只让每个 factor head 的正负类别在 BCE 梯度中各占一半。不得通过调决策阈值、改 heldout 或放宽门来恢复。

### V6-F48：类别平衡 BCE 不是可分 factor boundary 的充分机制

H-PT3-005 canonical run `20260821T140408Z__factorized-policy-training-s20260821-r1` 只把两个 logistic factor heads 改为正负类别总权重各半，其余数据、特征、三臂和 gate 均不变。V6 heldout balanced accuracy 降为 `0.82294`，false-safe 升为 `0.24746`，completion 为 `0.89334`；naive false-safe 也由 `1.0` 变为 `0.67863`，导致相对 naive 的 reduction 只有 `0.43116`。所有冻结质量门仍未通过，正式 `rejected`。

两个 V6 head 的 train balanced accuracy 仍只有 `0.97503/0.96014`，说明仅重加权有限步 smooth BCE 并没有形成稳定分离 margin；它改变了错误权衡，却没有消除训练边界错误。H-PT3-006 保持相同 denominator、raw feature 和原 gate，改用 deterministic class-balanced linear max-margin heads，并保留两 head AND。不得再通过类别权重或阈值扫描恢复。

### V6-F49：可分训练集上的最大间隔仍受离散 intervention 支持分辨率约束

H-PT3-006 canonical run `20260821T140847Z__factorized-policy-training-s20260821-r1` 使用相同 raw features 与 multi-size/multi-yaw 分母，将两个 factor heads 换为 class-balanced linear max-margin。V6 的 forward、lateral 与 joint train balanced accuracy 均为 `1.0`；heldout false-safe 降到 `0.05226`，相对 Real-only/naive 的 reduction 为 `0.66264/0.62637`，三项均过门。但 safe-route completion 只有 `0.84790`，balanced accuracy `0.89782`，仍未通过冻结门，正式 `rejected`。

原 train position 只有 forward `0/2/4/6/8` × lateral `0/1/2/3`，即使训练完全可分，最大间隔也可落在相邻采样位置之间；heldout 恰使用离散半步位置，暴露 `15.21%` false brake。H-PT3-007 只加密 train position denominator，新增位置全部与 heldout position set 离散，保持 size/yaw、heldout、features、SVM 和 gate 不变。不得调 SVM threshold 或删除 near-boundary heldout rows。

### V6-F50：set aggregation 下不能只用 false-safe 单轴增益比较会过度刹车的基线

H-PT6-001 canonical run `20260821T142626Z__compositional-risk-s20260821-r1` 在 scene0450 的 784 个双 clone episodes 上，用同一 frozen per-actor policy 对 logged actors 与两 clone 做 Boolean OR。V6 balanced accuracy/false-safe/completion 为 `1.0/0/1.0`；Real-only 为 `0.5/1.0/1.0`；naive 为 `0.52211/0.08844/0.13265`。V6 的全部绝对质量门通过，但相对 naive 的 false-safe reduction 最大只能是 `0.08844`，无法达到从 single-clone task 继承的 `0.50`，所以 H-PT6-001 仍正式 `rejected`。

naive 并未获得可用策略，而是 set OR 后以 `86.73%` false-brake 换取较低 false-safe。H-PT6-002 不追认旧 run，保持 frozen policy、scene、episode、label 和绝对门不变，预注册 paired Pareto gate：V6 对每个 baseline 的 false-safe 不更差、completion 不更差，并且 balanced accuracy 至少高 `0.20`。以后 actor-set policy 不得只用某一风险轴的下降评价会全刹车的 baseline。

### V6-F51：multi-actor development 的完美结果未在 one-shot confirmation 保持 false-safe

H-PT7-001 canonical run `20260821T143359Z__compositional-risk-confirmation-s20260821-r1` 在 attempt 先于质量读取、policy 与 H-PT6-002 gate 均冻结、scene0862 与八个 clone case tuples 全新的合同下，评估 784 个双 clone episodes。V6 balanced accuracy `0.92333`、safe-route completion `1.0`，且仍 Pareto 支配两基线；但 600 个 hazard 中漏掉 92 个，false-safe `0.15333`，超过冻结的 `0.10` 绝对门，因此 one-shot confirmation 正式 `rejected` 并消费。

不得用 scene0862 的漏检分布为该 multi-actor candidate 调 policy、case、threshold 或 gate，也不得用其余 Pareto checks 通过覆盖 false-safe failure。该 family 关闭。后续 H-PT8-001 来自此前一直显式保留的 `ABSTAIN_NO_LONGITUDINAL_CONTROLLER`，使用预注册 kinematic scenario grid 开始独立的 closed-loop utility family，不读取 PT7 badcase 生成参数。

### V6-F52：closed-loop collision 分母必须区分 policy 可避免与动力学不可避免，并审计 Real-only 监督

H-PT8-001 canonical run `20260821T143902Z__closed-loop-utility-s20260821-r1` 在 360 个静止单 actor、五秒、jerk-limited 纵向 scenarios 上，将三个 frozen policy arms 作为相同三秒 preview controller 的碰撞信号。V6 collision rate `0.25`、safe completion `0.9375`、comfort `1.0`、balanced accuracy `0.84375`，未过冻结门，正式 `rejected`；Real-only 与 V6 完全相同，naive balanced accuracy `0.69554`。

70 个 V6 collisions 按 6/8/10m/s 为 `3/22/45`，按 15/20/25/30/35m 初距为 `41/22/7/0/0`，并随 actor size 增大，说明原 denominator 把 t=0 已无法在同一 decel/jerk contract 下停车的场景也算成 policy false-safe。同时，Real-only equality 提醒后续必须审计它继承的 factor-label supervision；不得通过调 preview horizon 掩盖 baseline equality。H-PT8-002 只加入相同 dynamics 的 t=0 full-brake oracle，把 hazards 分为 avoidable/unavoidable，保留全部计数并只在 avoidable stratum 评价 policy collision；其余均不变。

### V6-F53：真实 box factor supervision 已解释静止 AABB preview，不能人为削弱 Real-only 制造增益

H-PT8-002 canonical run `20260821T144333Z__closed-loop-utility-s20260821-r1` 用同一 jerk/decel contract 的 t=0 full-brake oracle 将 280 个 uncontrolled hazards 分为 210 个 avoidable 与 70 个 unavoidable。V6 在 avoidable stratum collision `0`、safe completion `0.9375`、balanced accuracy `0.96875`，说明 H-PT8-001 的绝对 collision failure 来自可行性分母；但 Real-only 的三项指标完全相同，增量门仍失败，H-PT8-002 正式 `rejected`。

Real-only factor heads 虽无最终 clean collision positives，但合法读取了 logged box 的 forward/lateral overlap factor labels，已经足以学习静止 AABB 边界。不得删除这些真实监督、重命名 baseline 或只比较更弱 naive 来制造 V6 closed-loop gain；该静止单 actor family 关闭。H-R14-001 转回 core compiler 的既有 H-C 缺口，测试邻帧 temporal evidence-conditioned proposal，不由 PT8 结果选择参数。

### V6-F54：独立 verifier arms 合格不等于 factor conjunction 有足够 usable coverage

H-R15-001 canonical run `20260821T145504Z__factorized-verification-s20260821-r1` 只消费 H-R14-001 独立通过的 P1 photo 与 P2 geometry decisions，按双 ACCEPT 才 ACCEPT、双 REJECT 才 REJECT、disagreement 全 ABSTAIN。28 cases 中只有 1 个 joint ACCEPT、13 个 ABSTAIN、14 个 REJECT；joint false-safe 为 `0`、相对 P0 reduction `0.89286`，但 accept coverage `0.03571` 未达到冻结 `0.05`，因此正式 `rejected`。

不得用 OR、放宽 verifier threshold、把 disagreement 计作 ACCEPT 或把门改成 1 case 来恢复。RGB-only ECC proposal 对 photo 优化，却没有约束 geometry，导致两个独立有效 arm 的接受集合错位。H-R16-001 保留同一 temporal source、28-case denominator、P0-P4 thresholds 与后续 conjunction，仅把 outside-mask alignment image 改为等权 RGB gray + robust normalized render inverse-depth；render depth 只用于 proposal alignment，不充当 P2 truth。

### V6-F55：无类型约束的双模型语义共识会形成相关性共同错误

H-R20-001 canonical run `20260821T152456Z__semantic-consensus-s20260821-r1` 在冻结的 12 个 R16 semantic-evidence cases 上，以 DeepLabV3-ResNet50 与 SegFormer-B0 的 hole 内 dynamic-mask IoU `>=0.70` 作为唯一决策证据。该机制成功拒绝了 R16 原先唯一的边缘 false-safe `scene-0048__ad_gs__f057__actor_removal_hole`，但两个模型在 `scene-0048__ad_gs__f052__disocclusion` 上以 IoU `0.88267` 共同预测了错误动态内容，使 2 个 ACCEPT 中 1 个为 false-safe，false-safe rate `0.50`、相对 P0 reduction 仅 `0.08333`，正式 `rejected`。

不得把架构不同等同于错误独立，也不得扫描 consensus threshold；相关模型会在编辑语义不同的 hole 上共同犯错。H-R21-001 保留相同模型、12-case denominator、truth 与质量门，新增由 compiler edit type 决定的 typed semantic contract：`actor_removal_hole` 沿用冻结的 `0.50` 双模型 dynamic IoU 门；`disocclusion` 只有两个模型在 hole 内都预测零 dynamic pixel 才可 ACCEPT。决策仍不读取 target dynamic truth，P4 保持 ABSTAIN。

### V6-F56：actor edit-mask 的轴对齐二阶矩不足以恢复可接受的纹理对应

H-R22-001 canonical run `20260821T153531Z__independent-arms-s20260821-r1` 对六个 `actor_removal_hole` 只使用源 actor-edit mask 与目标已知 hole mask 的中心和轴向二阶矩，估计 axis-aligned affine 后搬运邻帧 actor RGB。actor 子集 P2 geometry 得到 `1/6` ACCEPT、false-safe `0` 并通过独立 gate，但 P1 photo 为 `0/6` ACCEPT，导致要求 P1/P2 同时独立合格的正式 gate 拒绝。四个 photo truth-safe cases 的 masked RGB MAE 仍为 `0.06299–0.07655`，高于冻结 `0.05`；整体 P1/P2 通过只来自未改变的非 actor cases，不能覆盖 actor 子集失败。

不得放宽 P1 阈值或用整体分母掩盖 actor failure。轴对齐中心/尺度对齐无法表示邻帧 actor 的旋转、剪切和透视轮廓变化。H-R23-001 保留相同源/目标 masks、六个 actor 分母、非 actor RGB-D ECC、verifier 与全部 gates，只把 actor 配准改为：以二阶矩 affine 为初始化，在两个二值 edit-mask 的 signed-distance field 上执行一次冻结 homography ECC；仍不读取 target RGB/depth/dynamic。

### V6-F57：edit-mask 轮廓 homography 仍不能建立跨时刻 actor 纹理对应

H-R23-001 canonical run `20260821T154110Z__independent-arms-s20260821-r1` 在相同六个 actor cases 上，以 R22 moment affine 初始化 signed-distance-field homography ECC。actor P1 仍为 `0/6` ACCEPT，P2 仍只有 `1/6` ACCEPT；SDF warp 让若干 photo MAE 从 R22 的 `0.063–0.076` 恶化到 `0.069–0.116`，仅一个 case 改善至 `0.05748`，仍未过冻结 `0.05`。因此整体 P1/P2 合格仍不能覆盖 actor subset，正式 `rejected`。

不得继续扫描 SDF ROI、iteration 或 photo threshold。二值 actor-edit support 约束的是轮廓，不携带姿态、可见面与光照的跨时刻纹理对应；更灵活的 homography 会扭曲错误纹理。既有 H-R9-003 已独立证明 same-time cross-frontend proposal 在 actor 子集上 P1 `4/6`、P2 `1/6` 且 false-safe 均为 `0`。H-R24-001 因此不再拟合 mask，而按 typed asset route 复用已验证来源：static/disocclusion 保持 R16 temporal RGB-D，actor_removal 使用同帧 cross-frontend；所有 verifier 与 actor gates 不变。

### V6-F58：小 actor 分母上的固定 false-safe rate delta 可以数学上不可达

H-R24-001 canonical run `20260821T154603Z__independent-arms-s20260821-r1` 按 hole type 复用已接受的 R16 temporal RGB-D static source 与 H-R9-003 same-time cross-frontend actor source。actor 子集 P1 为 `4/6` ACCEPT、false-safe `0`，P2 为 `1/6` ACCEPT、false-safe `0`；两臂的绝对 coverage/risk 都通过。但 actor P1 的 P0 只有 `1/6` false-safe，最大可能 rate reduction 为 `0.16667`，无法达到从 28-case 全局臂继承的固定 `0.25`，因此要求 P1/P2 均合格的 H-R24-001 按原合同正式 `rejected`。

不得追认 R24 gate，也不得删除唯一 P0 错误或降低风险要求。H-R25-001 保持 R24 proposals、decisions、truth 与 P1/P2 阈值完全冻结，只预注册适合六例小分母的 actor Pareto gate：每臂 coverage `>=0.10`、false-safe rate `<=0.10`，并且 false-safe count 相对 P0 至少严格减少 `1`。该离散计数门在两个臂上均可实现且仍要求实际消除错误；旧 R24 run 继续保持 rejected。

### V6-F59：下游 factorized consumer 必须从拥有字段的冻结 artifact 读取 case metadata

H-R27-001 首个 formal run `20260821T160452Z__three-factor-s20260821-r1` 在组装六个 actor factorized decision 时因 `KeyError: mask_pixel_count` 失败，仅产生 failed `TERMINAL.json`，未构造 gate 或方法结论。R24 的 `verifier_worker/PER_CASE_ARMS.jsonl` 拥有 factor decisions、truth 与 proposal hash，但 `mask_pixel_count` 的 schema owner 是同一 run 的 `CASES.jsonl`；初版 consumer 错把该 metadata 当作 arm row 字段。

修复把冻结 R24 `CASES.jsonl` 及其 SHA256 加入 R27 source contract，并仅从 case-id index 读取 `mask_pixel_count`。不得改变 P1/P2/P3 decisions、truth、三因子合取规则、coverage/risk/count gate、R24 rejected 状态或确认集锁。该失败属于 artifact binding plumbing，不否定 H-R27-001，修复后以新 run 重试。

### V6-F60：factorized validity 的各因子 truth 应做逻辑积，不能要求逐例标签相等

H-R27-001 canonical retry `20260821T160717Z__three-factor-s20260821-r1` 在六个 actor cases 上得到预期决策：`1` ACCEPT、`1` REJECT、`4` ABSTAIN，全部 factor decision disagreement 都 ABSTAIN，ACCEPT 的 joint truth-safe 为真，false-safe 为 `0`，coverage 为 `1/6`。但预注册合同额外要求 photo、geometry、semantic 三种 truth label 逐例相等；实际有两个 case 的 factor truths 不同，因此 run 按合同正式 `rejected`。

不得删除这两个不一致 case 或追认 R27。photo truth 衡量 RGB 恢复、geometry truth 衡量 depth、semantic truth 衡量动态语义，它们本就可以独立真假。H-R28-001 保持全部 proposal、factor decisions、fusion 与风险门冻结，将 joint truth 明确定义为三种 factor truth 的逻辑 AND，并要求精确保留两个 cross-factor truth disagreements；R27 的唯一实质失败项必须仍是错误的 truth identity 假设。以后 factorized evaluator 必须区分 decision disagreement 与 truth-factor diversity。

### V6-F61：未展开的聚合布尔失败不能用于猜测精确 factor-diversity 分母

H-R28-001 canonical run `20260821T161052Z__three-factor-s20260821-r1` 正确使用 factor-truth product，并保持 `1` ACCEPT、`1` REJECT、`4` ABSTAIN、false-safe `0`；coverage、strict error removal、所有 disagreement ABSTAIN、source immutability 与 R27 rejection retention 全部通过。唯一失败是预注册把 cross-factor truth disagreement count 猜成 `2`，完整逐因子展开后实际为 `3`。

R27 只给出了 truth-identity 聚合布尔失败，不能推出失败 case 的精确数量。不得追认 R28 或把实际值 `3` 再硬编码成安全门。H-R29-001 保持全部六例、factor decisions、factor truths、product、fusion 与质量门冻结，要求 factor truth diversity 非零且逐例透明报告；精确 diversity count 作为描述性输出，不参与资格。以后只能从已冻结的逐例 artifact 预注册精确计数，不能从 aggregate failure 反推。

### V6-F62：aggregate actor-layer validity 不能下放给单 actor identity component

H-R32-001 canonical run `20260821T163038Z__identity-factor-s20260821-r1` 用此前接受的 H-R13-009 model-index-0 removal support 与 H-R13-011 `actor_0000` binding，在 R30/R31 layer 内得到 `4,792` 个 identity pixels，覆盖 resized actor effect support 的 `91.02%`。P1 photo 在该 support 上仍 ACCEPT（MAE `0.043739`），且 photo/geometry/semantic 三种 truth evaluation 均 safe；但 P2 geometry mean relative error 为 `0.212180`，超过冻结 `0.20`，P3 DeepLab/SegFormer dynamic IoU 仅 `0.098330`，低于冻结 `0.50`，因此 identity-specific conjunction ABSTAIN，正式 `rejected`。

不得用 aggregate R29 ACCEPT、truth-safe、接近 geometry 门或 target semantic IoU `0.9946` 覆盖独立 decision failure，也不得放宽阈值。R7/R30 的 actor layer 由 all-actor edit evidence 构成，整体可用不推出任一 identity component 可用。H-R33-001 不再修复 generated layer，而提取 H-R13-011 已接受的 observed-support SceneIR `actor_0000` Gaussian chunk 与 logged trajectory，作为明确标注的 baseline/runtime asset；generated identity route 保持 rejected。

### V6-F63：预注册记录的声明时间不得晚于正式 run，即使 source commit 已先冻结

H-R37-001 首个 formal run `20260821T170543Z__trajectory-edit-s20260821-r1` 的方法、阈值与源代码已在 run 前 commit/push，数值上也使两个 1m actor translations 都通过 compiled/native sensor equivalence；但 `HYPOTHESES.jsonl` 内手填的 `recorded_at_utc=2026-08-21T17:15:00Z` 晚于 run directory 的 `17:05:43Z`。该自相矛盾时间戳破坏了预注册审计的机器可验证顺序，因此该 run 不得作为 canonical acceptance，数值只可用于 failure diagnosis。

不得回写旧记录、追认首 run 或仅凭 Git commit 顺序忽略结构化时间字段。H-R37-002 保持相同代码路径、两个 interventions、thresholds、source denominator 与资源合同，在服务器 `date -u` 实时时钟下追加新的预注册记录并重新 commit/push 后复跑；只有 retry 可成为 R37 canonical authority。

### V6-F64：factor consumer 记录 intervention metadata 时必须从冻结 owner 绑定完整字段

H-R39-001 首个 formal attempt `20260821T172447Z__static-contact-s20260821-r1` 在 static KD-tree 和任何 decision 生成前因 `KeyError: translation_delta_m` 失败，仅产生 failed `TERMINAL.json`。R39 config 为两个 intervention 写了预期 contact decision，但 consumer 在输出 decision row 时还读取 delta；delta 的事实 owner 是冻结 R38 payload/decision，初版 config 没有显式重复绑定。

修复只在 R39 config 中补入与 R38 完全相同的 `[1,0,0]` 与 `[0,1,0]`，不得改动 static query、0.80 coverage、0.90 retention、directional control、资源合同或 source denominator。该失败属于 metadata plumbing，不读取或改变实验结果；H-R39-001 在新 commit/push 后按同一假设重试。

### V6-F65：background Gaussian 密度与 actor AABB 极值不能充当 ground-contact evidence

H-R39-001 canonical retry `20260821T172656Z__static-contact-s20260821-r1` 在 824,583 个 observed background Gaussians 上按冻结的 1.5m horizontal / 0.35m vertical / 3-point contract 查询 actor AABB bottom。logged support coverage 只有 `0.17857`，远低于 `0.80`；x+1m coverage `0.16837` 因绝对覆盖不足而 REJECT，y+1m coverage 反而为 `0.19388`，方向控制失败。run 正式 `rejected`。

不得放宽 coverage、vertical tolerance 或把 retention 单独当 ACCEPT。background splats 混合道路、立面与其他表面，Gaussian AABB minimum 又受少量低端 primitives 支配，两者组合不是 ground-contact 语义。H-R40-001 转向冻结的同前端三相机 logged LiDAR：排除 dynamic pixels 后提升到 world frame，以 actor world-Gaussian y 轴 5% 分位作为 robust support anchor、局部 LiDAR y 轴 10% 分位作为 ground proxy，仅评估 R37 实际执行的 frame57；x/y directional controls 与物理/semantic-road abstention保留。

### V6-F66：独立 runner 必须显式绑定仓库根目录后再导入项目包

H-R41-001 首次正式启动在创建 run directory 或读取任何冻结 artifact 前因 `ModuleNotFoundError: No module named 'motion_proj'` 退出。R41 runner 缺少其他 WorldSim V6 runner 已使用的仓库根目录 `sys.path` 引导；实验主体、预注册 factor decisions 与 fusion contract 均未执行，因而这不是方法拒绝。

修复只把 `scripts/worldsim_v6/run_r41_actor_edit_factor_fusion.py` 的仓库根目录插入 `sys.path`，不得修改 R37/R38/R40 hashes、两个 intervention、四因子 decisions、reject-dominates fusion、资源合同或 claim boundary。修复后必须新 commit/push，并以同一 H-R41-001 重跑；首次启动不得被追认为 canonical run。

### V6-F67：手工绑定 source digest 必须逐字符比对实际 SHA256

H-R41-001 第二次正式启动已进入 source verification，但在创建 run directory 或读取 factor decision rows 前拒绝 R37 `MANIFEST.json`。诊断显示配置值第 38 个字符误抄为 `f`：`...e8f42f5946...`，实际冻结 SHA256 为 `...e8f42c5946...`；两者长度均为 64，因此肉眼概览未能发现单字符漂移。源 artifact 本身未改变。

不得跳过或放宽 `_verify`。修复只把 R37 manifest digest 的错误字符改为实际 `c`，其余 R37/R38/R40 hashes、interventions、factor decisions、fusion contract、资源与 claim boundary 全部冻结不变。新 commit/push 后仍按 H-R41-001 重跑，第二次启动不具 canonical authority。

### V6-F68：负值向量 CLI 参数必须用 `--option=value` 绑定，避免 argparse 将其解释为新选项

H-R43-001 首个 formal run `20260821T175434Z__selected-sensor-s20260821-r1` 已完成全部 source/proposal 绑定并创建 run，但 native worker 在加载 checkpoint 前以 rc=2 退出。日志明确为 `argument --translation-delta-m: expected one argument`：选中 proposal 以负数开头的字符串 `-1.0,0.0,-0.5` 被 argparse 解释成新的 option；此前 R37 的正值向量没有暴露这个入口问题。run 只有 failed `TERMINAL.json` 与 worker log，没有 sensor 或 gate。

修复仅把调用形式从两个 argv token `--translation-delta-m`, `<negative-vector>` 改为单 token `--translation-delta-m=<negative-vector>`。不得修改 R42 proposal、translation、renderer worker、任何 sensor threshold、GPU 预算或 claim boundary。该错误不否定 H-R43-001；必须在新 commit/push 后按原假设重试，首 run 永不追认为 canonical。

### V6-F69：verified translation 不应通过破坏性 float32 world-means 重写来拥有 trajectory edit

H-R44-001 canonical run `20260821T180210Z__verified-bake-s20260821-r1` 成功生成自包含 68MB package，所有非 translation actor fields byte-exact、shifted means content-addressed、manifest 完整、双次 bake byte-exact，且 typed validity/abstention 全部保留。但把 `[-1,0,-0.5]` 直接加到原始 float32 world means 后，反算 translation 的最大误差为 `1.9073486328125e-6m`，超过预注册 `1e-6m`，因此 run 正式 `rejected`。

不得把阈值放宽到 2e-6，也不得用舍入后的数组冒充精确 trajectory ownership。H-R45-001 改变表示机制：R35 的全部 actor arrays（包括 base world means）原样 byte-exact 保存，proposal translation 由独立 content-addressed float64 `T_delta_world` trajectory 拥有；runtime 明确按齐次变换组合 base world means。这样 edit 是显式、持久、可验证的，又不迫使高精度 transform 被吸收到 float32 geometry。R44 rejected package 仅保留为失败证据，不得供 runtime 使用。

### V6-F70：trajectory event identity 不能要求每个 timestamp 的 state content hash 唯一

H-R46-001 canonical run `20260821T181019Z__detached-logsim-s20260821-r1` 从完整复制的 detached R45 package 独立加载，196 行/每行 12,390 primitives、组合误差 `0`、导数不变误差 `1.42e-14`、两次 replay aggregate SHA256 完全相同，且 source package 在 copy 后未被 loader 使用。但预注册错误要求 196 个 state content hashes 全部不同；实际只有 `142` 个唯一状态。诊断显示唯一重复组覆盖 `14.1s` 到 `19.5s` 共 `55` 个 timestamp，表示同一个 stationary geometry state 被多个合法 trajectory events 引用。run 因此正式 `rejected`。

不得给 state bytes 掺入 timestamp 以伪造不同 state，也不得删除静止尾段。H-R47-001 明确分离两种身份：`materialized_state_sha256` 继续只哈希几何内容、允许并精确报告 142 个唯一状态；`trajectory_event_sha256` 哈希 sequence index、timestamp、visibility、proposal id 与 state hash，必须对196个事件全部唯一。重复 state group 与55次静止尾段必须原样保留，detached replay、组合精度和 abstention 合同不变。

### V6-F71：actor geometry trajectory 必须同时拥有 native lifecycle，不能把 repeated terminal pose 当作 active actor

H-R49-001 canonical run `20260821T182444Z__multiframe-sensor-s20260821-r1` 在 frames `[0,57,140,141,195]` 上把 R47 detached package 与 R35+同一 delta 两条 runtime 路径逐数组比较，5/5 sensor NPZ 完全相同，runtime modes、event/state identity、repeat、state restoration、package/checkpoint immutability 与资源门均通过。但两条 compiled 路径共同遗漏 native `RigidNodes.instances_fv`：frames 141/195 的 native actor 已 inactive、opacity 为零，compiled package 仍使用固定 observed opacity，导致最大 opacity field error `0.99643`、RGB MAE `0.00501`、depth MAE `0.44633m`。因此 run 正式 `rejected`；cross-path equality 只能证明两个 consumer 同错。

不得删除141/195、放宽 sensor 阈值、把 actor effect 为0解释成无关，或继续把 stationary geometry state 等同于 active lifecycle。H-R50-001 从冻结 StreetGS native `instances_fv[:, actor_0000]` 提取完整196帧生命周期，预注册验证 frames0-140 active、141-195 inactive 的单次边界，并把 content-addressed bool lifecycle 作为独立字段 bake 进 transform-owned package。base geometry、proposal transform 与 R49 rejection 必须原样保留；后续 sensor runtime 必须用 lifecycle 乘 actor opacity。
### V6-F72：下游 perception adapter 必须显式绑定 sensor NPZ 的拥有字段

H-R53-001 首次正式启动 `20260821T185332Z__lifecycle-perception-s20260821-r1` 在产生任何感知输出前失败。冻结 R49/R51 sensor NPZ 使用 `native_rgb` 与 `compiled_rgb` 字段，而复用的旧 R13 worker 硬编码读取 `rgb`，因此抛出 `KeyError: rgb is not a file in the archive`；run 仅有输入 index 与错误日志，没有 label、gate 或方法结论。

修复新增 R53 专用隔离 worker，唯一变化是显式读取 `compiled_rgb`；冻结 R52/R49/R51/model hashes、frames57/141/195、双重复、active exact control、inactive label-change gate、资源和 claim boundary 均不变。不得把 `native_rgb` 偷换为输入或先读取标签结果调阈值。新 commit/push 后按原 H-R53-001 重试，首次启动不具 canonical authority。
### V6-F73：冻结 CUDA 感知 worker 必须在进程启动前绑定确定性 CuBLAS workspace

H-R53-001 第二次正式启动 `20260821T185611Z__lifecycle-perception-s20260821-r1` 已正确读取 `compiled_rgb` 并加载冻结 DeepLabV3，但在首个 forward、任何 label 输出前被 `torch.use_deterministic_algorithms(True)` 拒绝：CUDA>=10.2 的 CuBLAS 需要进程启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。run 仍只有输入 index 与错误日志，没有方法结果。

修复仅由 R53 主进程向隔离 worker 环境注入 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，保持 deterministic algorithms 开启；不得关闭确定性模式。冻结 sources、模型、12 次推理分母、active/inactive gates、资源与 claim boundary 均不变。新 commit/push 后仍按原 H-R53-001 重试，第二次启动不具 canonical authority。
### V6-F74：跨 actor 复用 renderer-conformant translation 不保证全轨迹 interaction 可接受

H-R59-001 canonical run `20260821T192557Z__actor2-interaction-s20260821-r1` 将 R58 已通过 native renderer 的 `[-1,0,-0.5]m` translation 应用于 actor2 的完整196帧轨迹。self-kinematics 精确保持，最大 velocity/acceleration invariance error 仅 `8.88e-15/1.78e-13`；但相对 logged baseline 新增7个 AABB overlap events：actor0 在5.3--5.6s共4个，actor5 在6.2--6.4s共3个。因此 interaction factor 正式 `REJECT`，renderer conformance 不能提升为 edit validity。

不得删除发生 overlap 的帧、放宽 AABB gate、用删除4个旧 overlap 抵消新增7个，或因为 R58 sensor 通过就覆盖 R59。H-R60-001 保留 R58/R59 与完整27 actor x196帧 denominator，冻结 x/z 各 `[-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]m` 的80个非零 translation 网格，逐候选要求 self-kinematics ACCEPT 且新增 overlap 为0；按与被拒候选的距离、再按字典序选择最近可接受方案。contact、road、physics 与 safety 继续 ABSTAIN。

### V6-F75：StreetGS 数据 support 提取必须由拥有完整前端依赖的冻结环境执行

H-R62-001 首次正式启动 `20260821T194222Z__actor2-lidar-contact-s20260821-r1` 在读取任何 frame98 support 或产生 contact decision 前失败，`TERMINAL.json` SHA256 为 `592ba630ce84d996ff478b7314a12bfe1d5e0aedb0762bc7270e99ceaa7565d7`。主实验从通用 `motionproj` 环境直接导入冻结 StreetGS `DrivingDataset`，其模型依赖链要求 `pytorch3d`，该环境未安装，因此抛出 `ModuleNotFoundError: No module named 'pytorch3d'`。这属于环境 ownership plumbing，不构成 contact 方法结果。

修复新增隔离的 LiDAR support worker，并用配置中冻结的 DriveStudio Python 环境执行两次 frame98 三相机提取；主实验只读取两个 worker artifact、核验逐数组 repeat-exact 后运行原 contact evaluator。不得改变 R60 proposal、R61/R56/R40 authority、frame98、13,490 primitive denominator、动态像素排除、R40 quantile/radius/0.35m 阈值、预期 ACCEPT 方向、资源上限或 claim boundary。新 commit/push 后按原 H-R62-001 重试，首次启动不具 canonical authority。

### V6-F76：隔离前端 worker 必须同时绑定项目包根目录

H-R62-001 第二次正式启动 `20260821T194557Z__actor2-lidar-contact-s20260821-r1` 已切换到拥有 `pytorch3d` 的冻结 DriveStudio Python，但仍在读取 frame98 前以 worker rc=1 失败，`TERMINAL.json` SHA256 为 `1dcea9a007b18df26c4ff420fd15e8e759a1b814a5ad09184a3a4d22001420b0`。独立复现显示冻结 `DrivingDataset` 还导入项目内 `motion_proj.worldsim_v3.drivestudio_compat`，而新 worker 只加入 checkpoint backup 与 upstream 路径，遗漏当前仓库根目录，触发 `ModuleNotFoundError: No module named 'motion_proj'`。

修复只给 worker 增加显式 `--repo-root` 并在导入冻结 dataset 前插入 `sys.path`，同时令主进程在检查 rc 前落盘 worker stderr。不得改变 Python 环境、数据配置、proposal、frame、接触协议、阈值、预期方向或任何方法分母。新 commit/push 后仍按原 H-R62-001 重试；前两次启动都不具 canonical authority。

### V6-F77：单帧投影 LiDAR 无局部点时不得把 contact 缺证据解释为 edit invalid

H-R62-001 canonical run `20260821T194813Z__actor2-lidar-contact-s20260821-r1` 在 frame98 从三相机重复提取出完全一致的 `7,183` 个静态 logged-LiDAR world points，actor2 的13,490 primitives、生命周期、R60 proposal 与全部 authority 均精确绑定。但冻结2m局部查询在 logged 与 `[-1,0,0]m` 编辑中心附近都得到0个候选，最近水平距离分别为 `4.1758m` 与 `3.9666m`，因此两者 contact error 都不可计算并正式 `REJECT`。这证明的是单帧稀疏 support 不足，不是编辑破坏地面接触。

不得放宽2m半径、降低32点分母、增大0.35m误差阈值、删除 logged baseline 控制，或把两个 REJECT 宣称为编辑无效。H-R63-001 固定使用 target frame98 前后各10帧的对称21帧窗口 `[88,108]`，逐帧排除 dynamic pixels、提升到同一 world frame，并沿用已接受 R13 的0.05m deterministic voxel union；随后完全复用 R40 的 quantile/radius/denominator/error 阈值评价同一个 logged/selected pair。semantic road、physics、planning 与 safety 继续 ABSTAIN。

### V6-F78：相机投影 LiDAR 子集的时间融合仍可能无法覆盖远距 actor 接触邻域

H-R63-001 canonical run `20260821T195625Z__temporal-lidar-contact-s20260821-r1` 精确保留 R62 frame98 support，并从冻结21帧三相机数据得到 `149,723` 个 raw projected static observations、0.05m 去重后 `25,798` 个 world points；worker 双提取、坐标、窗口和 source gates 全部通过。但 logged 与 selected actor2 的2m邻域仍各为0点，两个 contact 均正式 `REJECT`。actor2 是 `vehicle.car` 且88--108帧间移动约 `1.97m`，因此失败说明相机投影稀疏子集在远距/遮挡区域不能承担 contact map，而不是简单增加同类帧数即可修复。

不得扩大投影时间窗、放宽 contact gates 或用环带高度挑选靠近 actor anchor 的平面。H-R64-001 改用同一 processed scene 的21帧360度 raw LiDAR，按每帧全部标注3D box加0.10m固定边界排除动态点，再做0.05m world voxel union；45个 raw/pose/box 输入以预先计算的聚合 SHA256 冻结。contact evaluator、logged baseline、selected proposal与全部阈值保持不变，semantic road、physics、planning与safety继续ABSTAIN。

### V6-F79：中心圆查询与 Gaussian 低分位 anchor 不适用于大尺寸 actor 的 box-filtered contact

H-R64-001 canonical run `20260821T200653Z__raw-lidar-contact-s20260821-r1` 从21帧360度 LiDAR 获得 `729,568` 点，按全部标注 box 排除 `95,432` 个动态点并形成 `73,010` 个静态5cm voxels；source、变换、动态过滤与资源门均通过。但 actor2 logged 中心查询虽有37点，Gaussian y-5% anchor 与 ground proxy 相差 `2.415m`；selected 只有18点且误差 `0.934m`，两者正式 `REJECT`。审计发现 model actor2 唯一对应 processed instance7 `vehicle.truck`，其 local +z 映射到 world -y，所以 Gaussian y-5% 是上表面而非接触底面；同时12.153m长的 truck 在 box-filter 后中心区域本就形成观测空洞。

不得继续把 R40 针对 actor0 偶然通过的中心/低分位定义当作跨 actor ground owner，也不得降低32点门。H-R65-001 显式绑定最近且有大间隔的 instance7 box，使用标注 local -z 底面拥有 contact anchor，并在 oriented footprint 外固定1m边界环查询 raw static voxels；环内用 median world-y 抑制不同高度表面，要求每个 intervention 至少64点、误差仍不超过0.35m。box只拥有几何位置，不提供 ground 高度；logged 与 selected 均须独立通过。

### V6-F80：actor package schema 分母必须把独立 lifecycle 计入 base arrays

H-R67-001 canonical run `20260821T202013Z__actor2-transform-bake-s20260821-r1` 成功产生 transform-owned package：float64 composition error 为0、双 bake byte-exact、196 transforms、13,490 primitives、四因子与 abstention 均正确，且所有源数组 hash 实际逐项相等。唯一失败是预注册把“7个 Gaussian/trajectory arrays”误写成 package 的完整 base array count=7；R56 还拥有独立 `actor_frame_validity`，实际键为8个，因此 `all_seven_base_arrays_byte_exact` 按合同正式失败。

不得删除 lifecycle、追认 H-R67-001、忽略数量门或改变任何 package bytes。H-R67-002 保持相同 R66/R56、proposal、transform、composition、repeat、资源与 claim boundary，只把 schema 分母明确为8，并同时要求8个 hash 全等且 `actor_frame_validity` 必须存在；新 commit/push 后重跑，首 run 保持 rejected authority。

### V6-F81：本地 shell 不得展开远端 Git 预检的命令替换

H-R68-001 第一次启动尝试在创建 run directory、读取任何冻结 artifact 或启动 GPU worker 前退出。PowerShell 在 SSH 到达服务器前展开了双引号字符串中的 `$(git status --porcelain)` 与 `$(git rev-parse ...)`，本地当前目录不是 Git 仓库，继而使传给远端 bash 的引号不闭合。服务器只收到语法错误；没有 R68 run、sensor、gate 或方法结果产生。

修复只把 SSH 的远端命令改为 PowerShell 单引号字面量，并把可执行 runner 的物理文件模式同步为已提交的 `100755`；不改 R68 代码、配置、冻结 source、frame98、actor2、transform/lifecycle ownership、sensor exactness、阈值、资源或 claim boundary。H-R68-001 保持 active，在 clean 且已 push 的同一 source commit 后重新启动。该失败是 launcher plumbing，不构成假设 rejection。

### V6-F82：multi-actor sensor 证据帧必须让每个被编辑 actor 对相机具有独立可见支持

H-R71-001 canonical run `20260821T205641Z__two-actor-sensor-s20260821-r1` 在 frame98 正确加载 R70，向 native checkpoint 同时应用 actor0/actor2 两个 transform，并在 compiled path 替换两组 owned fields。两组 field error、共享 RGB/depth/opacity、repeat、state restore、package/checkpoint immutability、资源和全部 abstention 均通过；但 actor0 虽 lifecycle active 且12,390个 opacity primitives 非零，其 camera effect pixels 为0。联合 sensor SHA256 因而与 R61/R68 actor2-only sensor 完全相同，只有 actor2 的19,785 effect pixels，按预注册的双 actor 可见门正式 rejected。

不得删除 `both_actors_have_visible_effect`、把 active primitives 当作 camera evidence，或追认 frame98 为双 actor runtime 成功。冻结 R51 证据表明 actor0 在采样帧中只有 frame57 具有17,568 effect pixels；冻结 R57 表明 actor2 在相邻 frame49 和后续 frame98 分别具有17,290/19,700 effect pixels。H-R71-002 因此在任何新渲染前固定 frame57，并改用冻结 R36 frame57 logged sensor 作 counterfactual baseline；所有 runtime、field/sensor、每 actor>=32 pixels、joint>=256 pixels、资源和 abstention 门保持不变。若 actor2 在 frame57 仍不可见，则保留第二次 rejection 并转向独立的可见交集搜索，不得继续猜帧。

### V6-F83：手工复制 source SHA256 时不得遗漏重复的相邻字节组

H-R82-001 第一次正式启动在创建 run directory、复制 package 或开始任何 bake 前，被冻结输入校验拒绝。R82 配置把 R70 `MANIFEST.json` 的实际 SHA256 `1583baf70c760ab700992ef9573ceb6fe59f992527445ac5eb5eb99f7795e6fe` 误抄为少一个 `5e` 字节组的 `1583baf70c760ab700992ef9573ceb6fe59f992527445ac5eb99f7795e6fe`；实际 artifact 与 R71 已使用的冻结 authority 均未变化。本次没有 run、gate、package 或方法结果。

不得跳过 `_verify`、重新生成 R70 artifact、放宽 package denominator 或追认本次启动。H-R82-002 仅修正该 source digest，并保持 R70/R80/R81、三个 actor、45 payload、34,257 primitives、588 trajectory rows、双 bake exact、资源和 claim boundary 全部不变；必须在新 commit/push 后重试。

### V6-F84：retry 配置的 hypothesis_id 必须与追加的预注册记录一致

H-R82-002 run `20260821T215606Z__three-actor-package-s20260821-r1` 数值上通过全部 package gates：3个 actor、45个 payload、34,257 primitives、588 trajectory rows、195,658,443 bytes，三棵 actor package tree byte-exact且双 bake 完全一致。但 digest repair commit 只修正 source SHA256，遗漏把 YAML `hypothesis_id` 从已关闭的 `WS-V6-H-R82-001` 更新为 active 的 `WS-V6-H-R82-002`，因此 SUMMARY 错绑旧 hypothesis。该 run 不得作为 canonical acceptance。

不得回写 run、把数值通过覆盖 provenance mismatch 或重新编号旧记录。H-R82-003 只把 YAML hypothesis binding 更新为 `WS-V6-H-R82-003`，保持已验证的 R70/R80/R81 hashes、bake bytes、所有 denominators、资源与 claim boundary 不变；新 commit/push 后重跑。

### V6-F85：稀疏固定时点的 actor lifecycle 有效性不能替代前视相机可见性

H-R95-001 canonical run `20260821T234324Z__scene0048-actor-visibility-s20260821-r1` 在第二个独立 scene0048 matched-formal30k checkpoint 上完整枚举9个 RigidNodes actor、196帧 lifecycle 与15,717个 primitives；source、partition、checkpoint immutability、GPU 和全部 denominator gate 均通过。但在预注册的 frames `[0,49,98,147,195]` 前视相机中，所有9个候选的 actor-only effect pixels 均为0，最大值仍为0，低于非平凡可见门64，因此 H-R95-001 正式 `rejected`。这证明固定五时点 lifecycle-active 不能推出 camera-visible support，并不否定 scene0048 checkpoint 或 actor 表示。

不得删除64像素门、把非零 primitives/opacity 当作屏幕可见、改用事后选中的单帧，或追认 R95 为成功。H-R96-001 保持相同 checkpoint、9个候选、前视相机、opacity阈值0.01和选择规则，改为在单个冻结进程内穷举全部196帧的所有 lifecycle-active actor/frame 对；仅在完整分母上按最大 effect pixels、actor index、frame index确定性选择。若穷举仍为0，则保留第二次 rejection 并转向三相机覆盖实验，而不是继续猜前视帧。

### V6-F86：全时域 sensor conformance 不得要求生命周期外的 actor frame_valid 恒为真

H-R98-001 canonical run `20260822T001113Z__scene0048-selector-transfer-s20260821-r1` 完成196帧 logged/edited RGBD、784次冻结 DeepLab 推理与全部资源/immutability 分母。零校准 threshold45 在 scene0048 得到 TP=30、TN=166、FP=0、FN=0，precision/recall/F1=1、skip=84.69%，优于 fixed256 的 F1=0.9831 与原生36帧 lifecycle 的 F1=0.9091。但预注册 gate 把 `package_actor_frame_valid` 作为196帧均须为真的 sensor-conformance 合取项；actor8 的冻结 lifecycle 正确地仅在 frames160..195 为真、frames0..159 为假，因此唯一方法检查 `all196_compiled_native_sensor_conformant` 与总 `passed` 为假，run 按合同正式 `rejected`。逐项诊断确认196/196帧数值 conformance、repeat 与 native state restoration 全部通过，最大 RGB MAE `1.36e-8`、depth MAE `6.19e-7m`。

不得追认或回写 R98、不得把160个 inactive frame 改成 active、不得删除全196帧 sensor 数值门，也不得重跑昂贵推理来掩盖治理错误。H-R99-001 只把合同修正为“每帧 `package_actor_frame_valid` 必须与冻结 native lifecycle 精确相等”，从 R98 的内容寻址 sensor/perception artifacts 独立重算196帧 conformance、784输出重复性与 selector 指标；R98 保持 rejected，R99 作为新的 governance-repair authority。

### V6-F87：跨场景配置不得凭摘要转抄 checkpoint authority

H-R102-001 首次正式启动在创建 run directory、读取 sensor 或启动 GPU worker 前被冻结输入校验拒绝。配置误用了不存在的 scene0255 matched-baseline 路径 `20260812T132516Z__streetgs-scene0255-matched-formal30k-s0-r50`，并把 R101 摘要中截断/误记的 digest `dba249822f22317d926cc2953d0a433f6a95e6963d35e42750b8f7074dad6acd` 当作 authority；R101 实际已通过的冻结 checkpoint 是 `20260811T214009Z__streetgs-scene0255-matched-formal30k-s0-r48`，SHA256 为 `dba24982a3f25e162b5e293165258a588cf9bd7a49e54e05d0d052de703cb2d2`。本次没有 run、gate、sensor、perception 或方法结果。

不得跳过 `_verify`、制造不存在的 checkpoint、把 launcher 失败解释成 selector rejection，或继续从人工摘要抄 authority。H-R102-002 只从已接受 R101 配置复制确切 checkpoint 路径与 SHA256，并更新 hypothesis binding；R101/R90 artifact、actor34 edit、196/784 分母、threshold45、逐帧 lifecycle 合同、资源与 claim boundary 全部不变。必须在新 commit/push 后重试。

### V6-F88：零 AABB interaction 不意味着 RGB factorial interaction 必须非零

H-R111-001 canonical run `20260822T022113Z__scene0255-two-actor-factorial-s20260821-r1` 精确绑定 R102 actor34-only、R110 actor24-only 与 R109 joint 的 00/10/01/11 冻结 sensor/perception arrays。三个 source 的 196/784 分母、repeat、logged cell 与 hashes 全部一致；actor34/actor24 条件边际分别覆盖 19/161 帧，joint 与 single-target union 的帧级 F1=1，像素 F1=0.999086，single-selector OR 对 joint target 的 F1=1。但预注册错误要求至少 1 个 RGB pixel 的 `rgb11-rgb10-rgb01+rgb00` 绝对残差超过 1/255；实际 196 帧的最大残差、平均残差与超阈像素数全部严格为 0，因此唯一 `sensor_factorial_interaction_detected` gate 失败，run 正式 `rejected`。

不得删除该 gate、追认 R111 或把精确 superposition 描述成非线性 renderer evidence。H-R112-001 以 R111 rejection 为冻结诊断 authority，改测与观测一致的新机制：sensor 层必须逐值 exact affine superposition；下游冻结 DeepLab 允许有界的非线性 residual，但 joint/single-union 像素 F1 必须不低于 0.995、对称差比例不高于 0.005，两个 actor 条件边际与帧级/selector-OR exactness 仍须保留。semantic correctness、local causality、contact、dynamics、physics、planning 与 safety 继续 ABSTAIN。

### V6-F89：正式 runner 的模块归属必须与导入路径一致

H-R116-001 首次正式启动在创建 run directory、读取冻结 artifact 或启动 GPU worker 前，以 `ModuleNotFoundError: No module named 'motion_proj.worldsim_v6.r116_scene0255_fourth_actor_edit_compiler'` 退出。入口脚本正确从项目包 `motion_proj.worldsim_v6` 导入主体，但实现文件被错误提交到 `scripts/worldsim_v6`；因此本次没有 run、gate、sensor、proposal、GPU 结果或方法结论，H-R116-001 按实现合同记为 infrastructure rejection。

不得通过临时修改 `PYTHONPATH`、从未承诺的工作树文件导入、忽略失败启动或把它追认为 actor1 方法结果。H-R117-001 仅修复模块 ownership：主体放入 `motion_proj/worldsim_v6`，runner 仍位于 `scripts/worldsim_v6` 并从项目包导入；actor1、frame195、4,489 effect pixels、838 Gaussians、196-frame lifecycle、80 translations、所有 source hashes、阈值、资源上限和 claim boundary 保持不变。必须在新 commit/push 后从干净工作树重新正式运行。

### V6-F90：RGB-difference 邻域不能假定覆盖冻结感知的全局标签响应

H-R122-001 canonical run `20260822T034305Z__spatial-impact-locality-s20260821-r1` 精确绑定 R118/R121 两个四 actor 饱和方向的392帧 sensor 与 frozen-DeepLab arrays，所有 source hashes、逐文件 hashes、196/196 正帧和资源门均通过。但预注册把实际 `450x800` 图像误写为 `600x1200`，因此 shape gate 正式失败；更关键的是方法门也独立失败：RGB-diff mask 膨胀64px 后聚合标签召回仅 `0.604784`，最差帧仅 `0.027397`，不存在预注册网格内逐帧100%覆盖半径。R122 按合同正式 `rejected`，不得因 shape 书写错误而追认其空间局部性假设。

不得只修正 shape 后删除 exact per-frame coverage、把60.48%聚合召回解释为稀疏验证成功、依赖平均 ROI 掩盖最差帧，或宣称 crop inference 等价。非 canonical 恢复诊断显示392/392帧所需半径均大于128px、391/392帧大于256px，中位所需半径约412px、最大约684px；固定256px平均已覆盖73.56%画面却仍只有92.90%标签召回。H-R123-001 必须用正确450x800分母和扩展半径网格正式复算这些非局部性边界，接受的结论应是拒绝 RGB-diff 膨胀稀疏机制，而不是放宽成近似覆盖。semantic correctness、crop equivalence、speedup、physics、planning 与 safety 继续 ABSTAIN。

### V6-F91：跨实验 source digest 必须直接复制磁盘 SHA256，不能依赖人工转抄

H-R124-001 首次正式启动于 `2026-08-22T03:57:05Z` 前在创建 run directory、聚合任何向量或产生方法结果之前被 source verifier 拒绝。R109 `SELECTOR_TRANSFER.json` 的配置 digest 被人工转抄为 `0b972e5d0ff102c1eda06a2b077f769fe836f4b3d856242f4df58dbecafc6eafd91`，而冻结磁盘文件的实际 SHA256 是 `0b972e5d0ff102c1eda06a2b077f769fe836f4b3d856242f4df75406faeafd91`。本次没有 canonical run、gate、聚合向量、指标或科学结论，按 infrastructure/source-authority rejection 记录。

不得跳过 `_verify`、修改 R109 artifact、追认 H-R124-001，或调整 threshold45、11条件、2156帧、类别支持、分离间隔及资源门来掩盖该错误。H-R125-001 只把 R109 selector digest 改为磁盘实值并更新 task/hypothesis identity；其余 policy/source authorities、condition corpus、门限、预期方向、资源合同和 claim boundary 全部保持不变，必须在新 commit/push 后从干净工作树正式运行。

### V6-F92：语料内精确阈值不能未经新条件检验就提升为前瞻不变量

H-R128-001 canonical run `20260822T042521Z__scene0230-orthogonal-holdout-s20260821-r1` 在预注册后新生成 scene0230 actor12 `[0,0,+0.5]m` 的196帧 sensor 与784个冻结 DeepLab 输出；所有 source、proposal、0新增 overlap、38,541 primitives、196帧 lifecycle、compiled/native sensor、repeat、GPU、wall 与 abstention gate 均通过。但 threshold45 在78个正帧中漏掉 frame77：RGB changed pixels 为26而 frozen-label changed pixels 为5，得到 TP77、FN1、TN118、FP0、recall0.987179、F1=0.993548。run 按 zero-error 合同正式 `rejected`。

不得追认 R128、删除 frame77、把5个标签像素降为无关、放宽 F1/recall，或在同一 holdout 上改 threshold 后宣称前瞻成功。诊断显示全部118个负帧的最大 RGB feature 仍为0、78个正帧的最小值为26，使包含 R128 的开发并集精确阈值区间缩为 `[1,26]`。H-R129-001 必须把 R128 明确降格为 threshold-revision development evidence，与 R126 的2156行和 R127 的196行合并，按预注册 max-min margin 规则选择 threshold13并只声明开发集精确性；随后必须在另一个新条件上做独立前瞻检验。

### V6-F93：AD-GS 全时域实验必须区分 train、development 与锁定的 heldout camera 分区

H-R134-001 首次正式启动 `20260822T053238Z__adgs-cross-frontend-threshold13-s20260821-r1` 在任何 sensor 或 perception 输出产生前，于请求 frame0 时退出。冻结 R3 adapter 的196时间轴只物化了118个 train、39个 development 时间步并刻意排除39个 heldout 时间步；worker 仅从 `getTestCameras()` 建表，因此只看见 development，frame0 不存在。失败目录仅4 KiB、sensor 文件数为0，不构成 threshold13 或 cross-frontend 方法结论。

不得把 heldout 图像补入 adapter、把缺帧静默删除后仍声称196分母、追认 H-R134-001，或把本次启动失败解释为 AD-GS transfer rejection。H-R134-002 只能合并 `getTrainCameras()` 与 `getTestCameras()`，从冻结 `partition.json` 预先导出 camera0 的精确118+39=157帧，并保持39个 heldout 未读；AD-GS checkpoint/edit、threshold13、正负支持、0 FP/FN、skip、资源门与所有 abstention 不变，新 commit/push 后重新正式运行。

### V6-F94：StreetGS 上冻结的单一 RGB 像素阈值不能直接宣称跨 frontend 不变

H-R134-002 canonical run `20260822T053744Z__adgs-cross-frontend-threshold13-s20260821-r2` 在 heldout 未读的前提下完成 AD-GS scene0048 的118 train+39 development 帧、157组 logged/edited sensor 与628个重复精确 DeepLab 输出；checkpoint、adapter、aggregate actor state restoration、分区、GPU、wall 与所有 abstention gate 均通过。冻结 StreetGS threshold13 在131个正帧、26个负帧上得到 TP130、FN1、TN26、FP0、recall0.992366、F1=0.996169，唯一漏检为 train frame13：RGB changed pixels=1、label changed pixels=1；run 按0-error合同正式 `rejected`。

不得追认 R134、删除 frame13、把1个标签像素降为无关、放宽 recall/F1，或把 AD-GS 数据用于回改 StreetGS threshold13 后仍称全局策略。诊断显示26个 AD-GS 负帧的最大 feature 为0、131个正帧的最小 feature 为1，开发区间是脆弱的单点 `[1,1]`。H-R135-001 只能显式声明 frontend-conditioned router：StreetGS 保持13，AD-GS 用 R134 开发集按预注册规则拟合为1；R134 保持 rejected，且 AD-GS 的39个 heldout 时间步必须在 policy freeze 后一次性验证。
### V6-F95: AD-GS threshold-1 exact classification does not survive the sole heldout confirmation

H-R136-001 canonical run `20260822T055538Z__adgs-heldout-confirmation-s20260821-r1` consumed the one allowed confirmation attempt before reading heldout quality. All 39 camera-0 heldout frames, 78 AD-GS renders, and 156 frozen DeepLab outputs completed within contract. Source immutability, adapter partitioning, repeat exactness, actor-state restoration, positive/negative support, skip, GPU, wall, output budget, and all abstention gates passed. The frozen R135 AD-GS threshold 1 nevertheless produced TP=31, TN=7, FP=1, FN=0: frame 14 changed 11 RGB pixels but changed 0 label pixels. Precision was 0.96875, recall 1.0, F1 0.984127, and the run was correctly rejected.

Do not retune a scalar threshold on these heldout rows, rerun the consumed candidate, delete frame 14, or reinterpret conservative over-execution as exact classification. The next hypothesis changes method family and objective: an exact-input identity guard may reuse cached perception only for byte-identical RGB inputs and must execute otherwise. R137 evaluates that one-sided operational contract on R134 development data plus the already frozen R133 StreetGS execution authority; R136 remains rejected.
### V6-F96: A negative comma-separated translation must not be passed as a detached argparse value

H-R138-001 canonical failed run `20260822T061548Z__adgs-antithetic-exact-input-s20260821-r1` created the exact-once attempt and completed train+heldout adapter materialization. The sensor subprocess then exited in argument parsing before frame 0: the detached token `-0.5,0.0,0.0` was treated as an option, so `--translation-world` reported that its argument was missing. No sensor array or perception output was produced, and the exact-input method was not measured. The attempt is nevertheless consumed under the preregistered any-outcome rule.

Do not rerun the same antithetic condition, claim a method rejection, or edit its run into a success. R139 uses a distinct world-z +0.5m condition and binds the vector as `--translation-world=<csv>`, while preserving the exact-once, 39-frame heldout, full/reference selective execution, identity-only reuse, reconstruction, support, resource, and abstention gates.
### V6-F97: Python booleans in formal JSON closeout code must use `False`, not JSON `false`

H-R140-001 failed run `20260822T063253Z__end-to-end-utility-s20260821-r1` verified all immutable inputs and wrote the end-to-end certificate, gate, and summary, but exited before RESOURCE_AUDIT, MANIFEST, and TERMINAL. The resource dictionary used the undefined Python name `false` for `gpu_used`, raising `NameError`. No GPU, training, confirmation read, or source mutation occurred. Although the partial gate was written, it is not a canonical acceptance because terminal closeout is incomplete.

Do not hand-create missing success artifacts or promote the partial gate. H-R140-002 changes only `false` to `False` and updates the YAML hypothesis binding; all sources, formulas, three conditions, zero-error authorities, end-to-end thresholds, resources, and claim boundaries remain unchanged for a new clean-commit run.
### V6-F98: A literal recovery must search the whole Python module, not only the first failing line

H-R140-002 failed run `20260822T063601Z__end-to-end-utility-s20260821-r1` reproduced the same certificate and gate as H001, then failed on the immediately following `training_started: false` field. The first recovery changed only `gpu_used`, leaving two lowercase JSON booleans in Python source. Again, no GPU, training, confirmation read, or source mutation occurred, and the partial gate is not canonical.

Do not promote either partial run or continue one-line-at-a-time repair. H-R140-003 is preregistered after an exhaustive `true|false|null` token search. It changes the exactly two remaining resource-audit values (`training_started`, `confirmation_content_read`) to Python `False` and updates the hypothesis binding; all scientific inputs, formulas, thresholds, denominators, budgets, and claim boundaries remain fixed.

### V61-F02：下游 runner 必须读取上游 gate 的真实嵌套 authority

H-ME1-001 正式入口完成所有冻结文件 hash 校验后，把 `ME0_GATE.json` 的通过位误读为顶层 `passed`，而
`worldsim_v61.me0_gate.v1` 的真实 authority 是 `checks.passed`。因此触发 `KeyError`；异常发生在 run directory
创建、O_method/O_eval tensor 读取、GPU ray compiler、proposal 编译和任何方法计算之前。canonical run=`null`，
不存在 oracle upper-bound 科学结果，不能把本次记成 method rejected。

不得跳过 ME-0 authority、修改 canonical ME-0 artifact、放宽 ME-1 gate，或把 launcher failure 追认为科学 attempt。
H-ME1-002 只把读取路径修正为 `document["checks"]["passed"]` 并增加嵌套 schema 回归；28-case、五臂、source
hashes、0.2m voxel/0.1m ray step、50% coverage、20% depth consistency、false-safe/stop rule 与资源预算全部不变。

### V61-F03：合法 actor ID 0 不得与 raster 的空身份 sentinel 共用

ME-2 actor control 准备审计发现，ME-0 的 scene-0048 sparse identity layer 合法包含 actor ID `0`，但 ME-1
相机 raster 用零初始化 `actor_grid`，导致 actor0 与“该 voxel 无 actor”无法区分。ME-1 primary O2 的10个 ACCEPT
全部来自 scene-0242，scene-0048 两个 actor case 均已由冻结 P1 REJECT，因此 O2=`10/28`、false-safe=`0`、
mask yield 与 primary gate 不受影响；但 canonical ME-1 的 O3 scene-0048 identity/swept 诊断不能提升为完整 actor 结论。

不得把 actor0 改号、删除 scene-0048、追认 O3 actor safety，或为此重跑 ME-1 主臂。后续实现把 empty sentinel 改为
`-1` 并增加 actor0 回归；ME-2/ME-4 必须消费修复后的 identity raster，已落盘 ME-1 run 保持不可变。

### V61-F04：冻结 source digest 必须先满足 SHA-256 的 64 字符结构合同

P4 第一次正式入口在创建 run directory、载入模型或占用 GPU 前，被 VAE source gate 拦截。实际固定 revision
`70e803bfb4e127d534049d8ab8c8cb511780d485` 的 VAE 文件为 `1311145138` bytes，实际 SHA-256 与服务器
`X-Linked-ETag` 均为 `379995ca170d8a899019125f389ba8692b2e35625ff64ddc3fdaa8c9302ac340`；预注册配置在
末尾误多录一个 `2`，形成 65 字符值。模型字节没有漂移，canonical run=`null`，不存在 capability 科学结果。

不得跳过 source gate、改写模型文件或重复下载。修复只删除多录字符，并新增所有 model digest 必须是 64 位小写
十六进制的回归测试；官方 commit、model/DINO revision、demo、seed、50 steps、512 octree、资源门与 stop rule 全部不变。

### V61-F05：离线 Hugging Face repo-id 解析必须有显式 revision ref

P4 第二次正式入口通过全部 source gate 并创建 failed run
`20260822T111747Z__voxel-smoke-s1234-r1`；Omni DiT 与 VAE 均以 0 missing/0 unexpected 成功载入，随后
`Dinov2Model.from_pretrained("facebook/dinov2-large")` 在离线模式失败。固定 snapshot 与三个文件已完整存在，但按
exact commit 下载不会自动创建默认 `refs/main`；官方 encoder 只传 repo-id、未传 revision，因而无法把默认 main
解析到已缓存 snapshot。该 run 没有生成 mesh/points 或 capability gate，不是模型能力 rejection。

不得开启正式 run 网络、重复下载 DINO、改官方 encoder 或更换 backbone。修复只按 Hugging Face 标准 cache schema
创建 `refs/main`，内容精确绑定已冻结 commit `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`；runner 在模型载入前
验证 ref、snapshot、config 与 model SHA，之后仍保持 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。

### V61-F06：Hugging Face cache ref 是无换行 commit token，不是普通文本行

P4 第三次正式入口创建 failed run `20260822T112159Z__voxel-smoke-s1234-r1`，再次在相同 DINO 离线解析点失败。
运行时常量确认 `huggingface_hub.HF_HUB_CACHE` 与 `transformers.TRANSFORMERS_CACHE` 都精确指向预期 cache root，
排除了环境变量和根路径猜测。直接审计已安装 `huggingface_hub.file_download.try_to_load_from_cache` 发现，它对
`refs/main` 使用原样 `f.read()`，不执行 `strip()`；由普通文本 staging 上传的 ref 是 41 bytes，尾部 `0a` 使
revision token 与 40 字符 snapshot 目录不相等。该 run 仍未生成 mesh/points 或 capability 结果。

不得继续猜 cache roots 或重复完整 P4。修复把精确目标 ref 机械规范化为 40 bytes，并让 runner 也要求 byte-exact
40 字符内容；先单独执行一次 repo-id 的离线 DINO load smoke，只有它通过后才允许下一次正式 P4。模型与参数不变。
修复后孤立 smoke 由 repo-id 离线载入 `Dinov2Model` 的 `304368640` 个参数，故解析卡点已关闭。

### V61-F07：共享 shape 环境必须显式包含 image-only backend 的官方导入依赖

H-ME2-001 failed run `20260822T120008Z__hy3d-actor-s1234-r1` 已通过全部冻结 source gate 并构造4个 actor
inputs，随后 A0 worker 导入官方 Hunyuan3D-2.1 package 时失败。该 package 的 `__init__.py` 无条件导入
`postprocessors.py`，后者依赖官方 `requirements.txt` 精确固定的 `pymeshlab==2022.2.post3`；既有 Omni
shape-inference 环境没有这个 image-only backend 依赖。失败发生在模型载入、GPU inference、asset 生成和 method
decision 之前；canonical run=`null`，不是 A0 或 A3 能力结论。

不得跳过 A0、patch 官方 `__init__.py`、安装无关 texture/UI 全依赖、改变四臂或把本次追认为方法 rejected。
H-ME2-002 只安装官方固定的 `pymeshlab==2022.2.post3`，在配置/runner 增加 exact version gate，并执行一次
离线 base pipeline import smoke；该 smoke 已成功导入 `Hunyuan3DDiTFlowMatchingPipeline`。模型、权重、
4 units/6 cases、seed、batch、steps、octree、compiler、truth separation、thresholds、资源和 stop rule 全部不变。

### V61-F08：Omni diffusion 支持 batch>1，不代表默认 marching-cubes extractor 也支持

H-ME2-002 failed run `20260822T120519Z__hy3d-actor-s1234-r1` 完成4个有效 A0 mesh，并成功载入 Omni、完成首个
A1 的2-sample diffusion 与 VAE implicit query。runner 随后发现输出 mesh 数为1而输入数为2并 fail-closed。
官方源码显示 `extract_geometry_vanilla` 虽把 logits reshape 为 `(batch_size,X,Y,Z)`，但 marching cubes 固定读取
`grid_logits[0]`，wrapper 也固定返回一元素 list；因此第二份 latent 没有 mesh，不是随机空输出或 OOM。
本次没有 A1/A2/A3 asset、method decision 或科学结论；canonical run=`null`。

不得静默丢弃第二样本、把6-case缩为3-case、patch官方源码、把全部 diffusion 降为batch1后冒充并行，或改变生成
参数。H-ME2-003 保持昂贵 diffusion batch2与逐样本generator，令官方 pipeline 返回两份 latent，再逐份调用同一
官方 VAE 的 batch1 decode/export。H002 已完成的4个 A0 只在 plan、input hashes、report 与每个 asset hash 全部
精确后复用，不重复GPU计算。模型、controls、seed、steps、octree、guidance、compiler、truth、threshold与stop rule不变。

#### V6-F97/V6-F98 recovery 收口

H-R140-003 从干净且已推送的源提交 `a13759ba8db03e1f740ad93e246ca24f0ff2d7fa` 完成 canonical run `20260822T063937Z__end-to-end-utility-s20260821-r1`。Scientific certificate SHA256 为 `913833af47e4171e27707f71418b6625ed358b538d1c8a5a18bca5ac7f585363`，与两次 partial computation 完全一致；完整 gate、summary、manifest、resource audit 与 terminal 的 SHA256 依次为 `ac3c79c0e93f2932a076da8323b89a210ff2cbaac27ffa13079ce89ae9d07b51`、`50900ff99736055a10c32f4362176b7fc87862ae84667591077d6c17024e635b`、`1cc753b3c0a9489ced2a58b23035466ee26cba963d2abc56c84ebd4d057e5a62`、`06c110236591529d5fef5f4178bfed696b6c0ad0cfbce94c497896dc92230265` 与 `be263ba010cdb936fbd01dbfa0fe294b8022101aae348c95030d2a42d45fdb77`。

该 recovery 不删除或重分类 V6-F97/V6-F98：两个失败目录继续保持不可变，只有 H-R140-003 是 canonical。完整 account 报告 StreetGS、AD-GS development 与 AD-GS exact-once confirmation 的端到端 reduction 分别为 13.5337%、11.1434% 与 1.66365%（macro 8.78024%、worst 1.66365%），reconstruction error 为 0。Selector 研究族在此次 recovery 后冻结；R141 未执行，本收口不授权继续 threshold、actor 或方向实验。

### V61-F09：通用生成表面不能冒充场景观测一致的 Occupancy

H-ME2-003 canonical run `20260822T121848Z__hy3d-actor-s1234-r1` 完成固定四臂各 6 例。A0 image、A1 bbox、
A2 raw-LiDAR point 与 A3 O_method voxel 全部为 `0/6 ACCEPT`。主臂 A3 没有 false-safe，但每例都在 method 和
独立 O_eval 中占据已观测 FREE cell；method conflict=`6..246`，eval conflict=`8..273`。所有四臂均有同一失败，
而 A3 的 native coverage、hole coverage、silhouette 与 extent 多数已达到冻结下限，因此不是提示词、seed、纹理或
单一轮廓阈值问题。

该失败是科学机制 rejection：Hunyuan 输出的通用闭合 actor surface 没有把场景 FREE-space 作为硬约束，不能作为
Occupancy-authoritative proposal。按预注册规则永久停止本版本 Hunyuan actor 路线；不得靠 prompt/seed/steps/
octree sweep、放宽 FREE=0、事后 clipping 或 per-case 选择恢复。ME-3 学习式 occupancy 是计划中的独立机制，
仍按 GaussianWorld→OccWorld→Drive-OccWorld→IR-WM→OccSora 优先级审计，不把本失败无依据外推到该路线。

### V61-F10：tmux 正式入口必须由 wrapper 自举仓库根目录

H-ME3-GW-001 的第一次正式入口从 source=`16c0efd8d570eaa15c5c4757ddfb434af8b61ede` 启动，但 tmux
非登录环境没有仓库级 `PYTHONPATH`；Python 把 `scripts/` 而不是 repository root 放在 `sys.path[0]`，wrapper
因此在导入 `motion_proj.worldsim_v61.me3_predicted_experiment` 时立即触发 `ModuleNotFoundError`。失败发生在 run
directory 创建、source/artifact 读取、模型载入、GPU context、predicted occupancy 或 method decision 之前；
canonical run=`null`，没有科学结果，不能记为 GaussianWorld rejection。

H-ME3-GW-002 只在 wrapper 导入项目包前把 `Path(__file__).resolve().parents[1]` 加入 `sys.path`，并先用
`--help` 做无 run/GPU 的入口 smoke。GaussianWorld commit/weights、两个并行 scene workers、seed、2Hz frame schedule、
class mapping、UNKNOWN policy、28-case denominator、O_eval separation、thresholds、资源预算与 stop rule 全部不变。


### V61-F11：单卡 inference capability 通过不等于 predicted Occupancy 可以成为安全 authority

H-ME3-GW-002 canonical run `20260822T134559Z__predicted-occ-s1-r1` 完成两个 scene worker、24 次官方 streaming
inference、4 个 target occupancy、28 个 method decisions 和隐藏 O_eval 评分。预测臂与 oracle O2 得到相同的
`10/28 ACCEPT` 和 mask-area yield=`0.3983001361`，但这10例全部 false-safe；route-support 例的 hidden
observed-FREE conflict ratio=`0.766..0.958`，actor/disocclusion=`0.159..0.328`。run 正确以
`predicted_zero_false_safe=false` 拒绝。P6 的 weight/output/资源 capability 结论仍有效，但不能提升为安全性结论。

官方 GaussianWorld head、网格与类别源码，以及 DriveStudio nuScenes transform 源码和跨 metadata 数值对照都没有发现
x/y/z、class17 empty、camera order 或 lidar2img 错误。小幅前相机矩阵差异来自 nuScenes 异步相机 timestamp，后相机
在机器精度内一致。因此不得通过轴交换、投影修补或输入排列试错重开 GaussianWorld。

不得用 O_eval 选 confidence threshold、降低 UNKNOWN/verifier 门、做 grid/schedule/checkpoint sweep，或把 predicted
FREE 冒充 observed truth。已有 artifact 已证明把 observed O_method FREE 作为保守 veto 会让10个接受项全部 abstain；
无需创建零产出的重复 run。ReliOcc、α-OCC 与 OCCUQ 的可靠 uncertainty 需要训练/calibration，朴素 softmax/entropy
也不支持无校准安全声明。后续只允许先做一次 IR-WM truth-free current-state capability smoke；通过后才消耗唯一一次
ME-3 recovery，失败则停止 learned occupancy 并保留负结论。
### V61-F12：checkpoint 的零 missing gate 必须区分未使用的官方删除参数与有效 forward state

H-P7-IRWM-001 canonical run `20260822T143153Z__irwm-current-smoke-s1-r1` 已完成官方 IR-WM current-state GPU
forward，并写出 finite/nonempty occupancy。17项 gate 中15项通过；失败项只有环境版本字符串和模型零 missing。
Detectron2 使用官方 `0.6+cu111` wheel，而预注册只写 `0.6`；这不是不同 release。checkpoint 的唯一 missing keys
为 `pts_bbox_head.transformer.reference_points.weight/bias`。冻结官方 `WorldBEVFormerHead.init_weights()` 明确删除
整个 `transformer.reference_points`，其 detector decoder 在本次 `get_bev_features` current-state 路径不执行。

不得改写 H001 的 rejected terminal、直接手工把 gate 改成 PASS、给 missing 参数调值，或重复完整 GPU forward。
H-P7-IRWM-002 使用独立 P7R task，精确绑定 H001 gate/report/output/manifest/terminal 和官方删除源码，只允许
Detectron2 build suffix 与上述两项 source-proven unused missing keys；其余 H001 capability、truth-free、resource
合同全部原样要求通过。任何额外 missing/unexpected key 或 artifact 漂移都停止 learned occupancy。
### V61-F13：不同 predicted Occupancy capability 不能替代独立 observed-FREE safety authority

H-ME3-IRWM-001 canonical run `20260822T145543Z__irwm-predicted-occ-s1-r1` 完成两个并行 scene workers、
4 个 target occupancy、28 个 method decisions 和隐藏 O_eval 评分。IR-WM primary 与 oracle O2 都得到相同的
`10/28 ACCEPT`、accepted mask yield=`0.3983001361`，但全部10例 false-safe。route-support 的 hidden
FREE conflict=`0.344..0.571`，actor/disocclusion=`0.106..0.173`，均超过固定0.05；因此唯一顶层失败 gate 是
`predicted_zero_false_safe`。正式 run 无训练、calibration、threshold selection、confirmation read 或 truth 泄漏，
资源也在预算内，故这是科学机制 rejection，不是工程 blocked。

GaussianWorld 与 IR-WM 使用不同官方时序机制、类别合同和网格，却都复现 oracle 的10例接受集合且得到10/10 false-safe。
本证据拒绝在当前 development 协议中把 learned argmax occupancy 直接作为安全 authority；不否定两模型的 perception
capability，也不产生现实安全声明。不得再换 backend、选 confidence threshold、改 checkpoint/grid/history window、
放宽 verifier、用 O_eval 选阈值，或执行确定性零 yield 的 observed-FREE veto 冒充恢复。唯一 ME-3 recovery 已消费，
ME-4 不授权；V6.1 minimum experiment 以负结论收口。
## WorldSim V6.5 failure ledger 启动审计（2026-08-27）

`WS-V65-P0-INHERITANCE-PROTOCOL-01` 没有新增科学、数据或资源失败。首次新分支 `git push -u` 未建立
upstream，根因是远端 GitHub 出站未使用当前 LocalTUN；设置会话端口后同一分支成功推送，没有代码/run/quality
影响，因此不分配 `V65-Fxx`。下一可用编号仍为 `V65-F01`。

P1 stage preregistration 与实现落盘前审计：没有新增 failure；尚未创建 P1 run、尚未读取指标，下一可用编号仍为
`V65-F01`。

### V65-F01 — trajectory condition 被错误要求改善 task-agnostic 物理标签

- task/run：`WS-V65-P1-CONDITION-SIGNAL-ATLAS-01` / `20260827T074500Z__signal-atlas-s0-r1`；
- symptom：T0 AUROC/AUPRC 相对 q0 为 `-0.000183/-0.001443`，fixed-route density 相对恶化 `5%`，
  scene lower/equal/higher=`1/13/2`；只有真实 trajectory 相对 shuffle 的 AUROC 响应 `+0.009591`；
- root cause：T0 把 trajectory residual 训练成 task-agnostic hidden-FREE 分类器；trajectory 明明是任务查询，
  却没有 task outcome / utility / actor-time supervision，目标语义与 WoTE/UniAD/VAD 的 planning-oriented 用法不一致；
- preserved evidence：canonical run、173MiB compact cache、预注册配置和模型均保留；
- resolution：`WS-V65-H-P1-001` rejected，禁止 T1/seed/capacity rescue。新建 `WS-V65-H-P1R-001`，明确分离
  frozen `r_phys` 与 nonnegative relevance-scaled `r_task`，只以 fixed-opportunity task risk 判定；
- claim impact：当前没有 V6.5 trajectory-conditioned method claim；P1R 仍是 legacy train-only mechanism。

下一可用编号：`V65-F02`。

P1R canonical run 完成且四个预注册 gate 全过，没有新增 failure。该结果只减少 1 个 sampled route conflict，
已按弱 train-only signal 记录，不以小 denominator 夸大结论。下一可用编号仍为 `V65-F02`。

P2 metadata-only cohort freeze、preparation/native/evidence/evaluator 入口落盘前没有新增 failure；formal P2 quality
尚未读取。下一可用编号仍为 `V65-F02`。

### V65-F02 — fresh scene metadata 选择未覆盖冻结 IR-WM temporal-info capability

- task/runs：preparation r1 与 `scene-0520/0781/0800` 三个 native scene r1；
- symptom：preparation 全 shard 扫描尚未完成；并发 native workers 在 sidecar、model score 与 quality 生成前均于
  `payload["infos"][scene]` 触发 `KeyError`；`scene-0106` 经 key audit 同样不可用；
- root cause：首版 cohort 只审计 V6.1–V6.4 config exposure 与 processed availability，没有检查冻结
  `nuscenes_temporal_infos_train.pkl` 的 700-key capability boundary；
- literature/open-source response：官方 BEVFormer 要求通过 `tools/create_data.py nuscenes ...` 生成
  `nuscenes_infos_temporal_{train,val}.pkl`。当前 research 为避免重新生成全量 infos、CAN bus 依赖与 schema 漂移，
  采用更窄的项目迁移：只从冻结 pickle 已支持、且未被 v61–v64 使用的 scene 中重选；
- resolution：formal quality read 仍为 false；首版 cohort 标记为
  `superseded_pre_read_capability_ineligible`。最终 cohort 冻结为
  `0996/0443/0002/0043/0023/0072`，保留失败目录和 2.5GiB partial raw，并让 recovery 复用已抽取文件；
- claim impact：没有 P2 模型或质量结论，`WS-V65-H-P2-001` 保持 active；不消耗唯一正式 P2 read。

下一可用编号：`V65-F03`。

V65-F02 recovery 已完成：preparation r2 成功生成 6/6 processed scenes；6 个 pipelined native scene runs
均 `passed=true`，72/72 targets 完整；evidence run 72/72 units `passed=true`。全程未读取 P2 quality，未出现
新的失败，因此下一可用编号仍为 `V65-F03`。

### V65-F03 — train-only monotone trajectory risk 在 fresh ranking boundary 完全等序

- task/run：`WS-V65-P2-TRAJECTORY-CONDITIONED-RISK-01` /
  `20260827T093900Z__trajectory-selection-s0-r1`；
- symptom：q0 与 task arm 均为 fixed-route `18/6975`，relative reduction=`0%`，worst-tail 完全相同，
  scene lower/equal/higher=`0/6/0`；task arm 多 4 个 non-route conflicts（relative `+0.0881%`）；
- root cause：P1R 的 legacy train-only 信号只有 1/20 sampled conflict，未迁移到 fresh 40% selection boundary；
  nonnegative trajectory residual 改变 probability，但没有产生可泛化的 route ranking 变化；
- preserved evidence：唯一正式 72-case run、case metrics、canonical native/evidence inputs 全部保留；
- resolution：`WS-V65-H-P2-001` rejected；关闭 trajectory-only score-ranking family，不执行 T1/seed/capacity/
  threshold rescue，不做第二次 P2 read；
- literature response：PRECOG/M2I/GameFormer/VAD/Implicit Occupancy Flow 均将条件建模放在 ego goal/action 与
  multi-agent future response 或连续时空查询上。后续仅允许以新 train-only hypothesis 审计 actor-time/action-outcome，
  且不得使用已消费 P2 cohort 做选择。

下一可用编号：`V65-F04`。

P2R actor-time train-only hypothesis 已在读取 token outcome 统计前冻结；当前无新增 failure，下一可用编号仍为
`V65-F04`。

### V65-F04 — 正常驾驶 legacy Actor tokens 的 1.5m binary collision support 为空

- run：`run://worldsim_v65/WS-V65-P2R-ACTOR-TIME-TRAIN-ONLY-01/20260827T100000Z__actor-time-s0-r1`；
- symptom：train=`0/476 positives`、eval=`0/302 positives`，A0/A1/shuffle AUPRC 均为 0、AUROC undefined；
- root cause：冻结的 1.5m Actor swept-envelope/ego-route 硬碰撞事件在这些正常驾驶 scenes 中不存在；
- resolution：run 与 96KiB cache 保留；不扩大半径、不换 scenes、不重跑 binary label。新建 P2C continuous
  proximity cost，固定 `exp(-distance/6m)` 与 absent=60m；
- literature response：joint dynamics+cost map（CVPR 2021）、Occupancy Flow、DTPP 与 DiffStack 都支持连续
  时空 cost/flow，而不是依赖稀有硬碰撞标签；
- claim impact：P2R 没有 actor-time 机制结论；P2C 仍为 legacy train-only。

下一可用编号：`V65-F05`。

### V65-F05 — 连续 cost 共用物化器仍硬依赖二值半径字段

- run：`run://worldsim_v65/WS-V65-P2C-ACTOR-TIME-COST-01/20260827T101500Z__actor-time-cost-s0-r1`；
- symptom：run directory/status 创建后，`_materialize` 在任何 evidence unit 读取、GPU geometry、训练或评分前，
  对不存在的 `evidence_contract.route_corridor_radius_m` 抛出 `KeyError`；
- root cause：P2R/P2C 共用 Actor-token materializer，但二值 label 的 task-specific config field 被写成公共必需字段；
- open-source response：Hydra 的 config composition 将共享基础字段与任务组差异组合；项目内采用同一窄边界，
  让未被 continuous target 消费的 binary radius 成为可选字段，而不是向 P2C 科学合同补入伪依赖；
- resolution：失败目录保留，科学输入/gates/seed/model 未改变；修复后使用新 run-id r2，未覆盖失败现场；
- claim impact：没有科学 read 或指标，不构成 P2C 负结果。

### V65-F06 — Actor×time 连续代价对时间敏感但不优于 snapshot

- run：`run://worldsim_v65/WS-V65-P2C-ACTOR-TIME-COST-01/20260827T102000Z__actor-time-cost-s0-r2`；
- symptom：A1 相对 A0 Spearman `-0.014889`、MSE relative reduction `-34.59%`；matched 40% pooled cost 虽降低
  `10.37%`，两个 eval scenes 却都恶化（lower/equal/higher=`0/0/2`）；
- mechanism audit：真实 A1 比 scene-wise shuffled-time Spearman 高 `0.098817`，故网络使用了时间特征；失败不是
  条件完全未进入网络，而是该增量在冻结 split 上没有形成优于强 snapshot geometry 的可泛化排序；
- literature response：UniAD、DTPP、DiffStack 的有效增量来自 planning-oriented joint objective 或候选策略
  cost evaluation，不支持继续放大当前独立 Actor-token MLP；
- resolution：`WS-V65-H-P2C-001` rejected；关闭 Actor/time family，不做 scale/seed/capacity/split rescue，P3
  不解锁。后续只允许审计固定 V6.4 risk 上的独立 admission，不能重开已关闭的 representation family；
- claim impact：无 Actor-time method/planning/safety claim；formal V6.5 selection read 仍为 false。

下一可用编号：`V65-F07`。

### V65-F07 — continuous learned admission 用coverage换来了新的case/route风险

- run：`run://worldsim_v65/WS-V65-P4T-LEARNED-ADMISSION-TRAIN-ONLY-01/20260827T110000Z__learned-admission-s0-r1`；
- symptom：coverage绝对提升`6.64pp`且scene support=`7/8`，但case failures `0→1`、pooled fixed-route density
  `+8.82%`、worst-tail `+7.61%`；
- root cause：连续context预测在一个night case给出`0.521873` coverage，高于其oracle-safe `0.510822`，把
  hidden-FREE conflict从`0.047814`推到`0.050764`；弱coverage回归精度不能替代显式held-out risk control；
- literature response：CRC需要独立calibration选择风险参数；SOFT top-k是P5 allocator且只在P4有效后解锁；
  GroupDRO不能修复pooled selector自身新增failure。三者都不能在观察本结果后作为无代价后处理；
- resolution：`WS-V65-H-P4T-001` rejected，关闭learned admission，不调coverage上限/loss/seed/capacity，不准备
  fresh admission cohort，不解锁P5/CRC；
- claim impact：保留V6.4 M1；没有V6.5 differentiable-admission、calibration、planning或safety claim。

下一可用编号：`V65-F08`。

### V65-F08 — nuScenes map expansion v1.2 与当前devkit schema不兼容

- scope：R3 pre-run capability audit；没有run directory、quality read、model fit或指标；
- symptom：公共盘v1.2 expansion JSON可解压，但当前`NuScenesMap`明确拒绝`version < 1.3`；
- root cause：v1.2是prediction/arcline版本，当前devkit合同要求v1.3；官方v1.3增加lidar basemap支持并移除一条坏lane；
- literature/open-source response：nuScenes官方安装文档要求把map expansion解压到`maps/expansion`并使用匹配devkit；
- resolution：不降级devkit、不绕过版本检查、不使用PNG伪造11层语义。改用公共盘官方v1.3，独立目录能力调用得到
  `8×200×200`非空语义mask；
- claim impact：无科学结论；R3 hypothesis在恢复完成后才预注册。

下一可用编号：`V65-F09`。

### V65-F09 — 官方地图上下文被模型使用，但没有改善 voxel 风险或路线决策

- run：`run://worldsim_v65/WS-V65-P1R3-MAP-CONTEXT-TRAIN-ONLY-01/20260827T114500Z__map-context-s0-r1`；
- symptom：R3 相对 q0 的 AUROC/AUPRC 为 `-0.000496/-0.002280`，fixed-route density 完全相同
  `0.00299581→0.00299581`，scene lower/equal/higher=`1/14/1`；
- mechanism audit：真实地图相对 within-unit shuffled map 的 AUROC 为 `+0.000625`，non-route risk `-0.756%`，
  因而地图通路并非完全失效，但增量太弱且没有形成路线决策变化；
- root cause：冻结 q0 已编码大部分可由局部 road semantics 解释的物理边界；给逐 voxel 判别器追加静态地图，仍未把
  监督对象对齐到 Ego 真正执行轨迹后访问的未来 world/Actor outcome；
- literature response：CoRL 2022 Task-Relevant Failure Detection 把预测误差传播到 planning cost；PRECOG 对受控
  Ego goal 条件化其他 Actor future。二者支持改变预测对象，不支持继续扩大 map residual；
- resolution：`WS-V65-H-P1R3-001` rejected；关闭 per-voxel map/context residual，不做 seed/capacity/feature/radius
  rescue。下一步建立 trajectory-level visited-state reliability，直接预测 `(scene, unit, τ)` 的未来访问走廊 outcome；
- claim impact：无 V6.5 map-conditioned method、selection、planning 或 safety claim。

下一可用编号：`V65-F10`。

### V65-F10 — trajectory-level neural head 改善MSE却破坏决策排序

- run：`run://worldsim_v65/WS-V65-P1R4-TRAJECTORY-VISITED-STATE-01/20260827T121500Z__visited-state-s0-r1`；
- symptom：V1 MSE相对Qagg降低`87.35%`，但Spearman `0.751487→0.635127`，unsafe AUROC
  `0.978261→0.909420`，lowest-risk 40%实际cost `0.038137→0.057718`（恶化`51.35%`），scene
  lower/equal/higher=`2/7/6`；
- root cause：小样本连续回归把大量低幅cost压缩得更准，但这种校准目标没有保持安全选择所需的尾部排序；
- preserved positive：改变预测对象本身成功。直接Qagg的3/3 viability gates全过：Spearman=`0.751487`、unsafe
  AUROC=`0.978261`、selected实际cost相对全体降低`62.98%`；
- resolution：拒绝learned V1 head，不改loss/seed/capacity；保留确定性、可解释的trajectory-level Qagg，后续只做
  Actor companion与fresh prediction-object transfer，不重新训练voxel residual；
- claim impact：支持legacy train-only trajectory-level mechanism，不构成formal V6.5 selection/planning/safety claim。

下一可用编号：`V65-F11`。

### V65-F11 — per-actor 强相关不能经max聚合成trajectory-level Actor reliability

- run：`run://worldsim_v65/WS-V65-P1R5-ACTOR-FALSE-SAFE-01/20260827T123100Z__actor-false-safe-s0-r1`；
- symptom：A0 trajectory max-cost forecast Spearman=`0.626087`，低于冻结`0.70`；A1更低为`0.488696`。
  `relu(A1-A0)` monitor 对`relu(target-A0)` gap 的Spearman=`-0.054402`、AUROC=`0.522222`，lowest-monitor
  40% gap反而恶化`73.40%`；
- support audit：eval 24 trajectories/302 Actor tokens，9个positive gaps、6个zero monitors，不是零标签问题；
- root cause：per-actor P2C Spearman `0.872`被trajectory-level maximum的极值误差放大；A1时间通路本身较弱，
  其正向disagreement不代表A0 false-safe；
- resolution：`WS-V65-H-P1R5-001` rejected；不调gap threshold、不训练monitor、不改max/smooth-max聚合，Actor
  companion关闭。保留R4 world-state Qagg positive，资源转向其fresh transfer；
- claim impact：无trajectory-level Actor reliability、planning或safety claim。

下一可用编号：`V65-F12`。

### V65-F12 — smooth-tail改善any-error分离，却恶化visited-error rate与实际选择代价

- run：`run://worldsim_v65/WS-V65-P1R6-SMOOTH-TAIL-VISITED-STATE-01/20260827T124500Z__smooth-tail-s0-r1`；
- symptom：固定temperature=0.10的Qsoft-tail令unsafe AUROC `0.978261→1.000000`，但Spearman
  `0.751487→0.708230`，selected-40% actual cost `0.038137→0.048535`（恶化`27.27%`），scene
  lower/equal/higher=`4/6/5`；
- root cause：upper-tail强调了单个高q0 state，适合检测trajectory内“是否存在任何风险”，却与当前监督的visited
  hidden-FREE连续比例及matched-coverage期望代价不对齐；
- literature response：MIDAM的smoothed-max/attention需要bag-level AUC训练；RAP把风险预测耦合进robust planning；
  TAT依赖多条采样轨迹和历史。三者都不是当前单轨迹固定q0的无训练温度修复；
- resolution：`WS-V65-H-P1R6-001` rejected；关闭smooth-tail，不扫temperature/target/coverage。保留R4 Qmean并
  按冻结合同完成P2V fresh transfer；未来只有在真实candidate trajectory set存在时才可新开set-level aggregation；
- claim impact：无smooth-tail、CVaR、planner或safety claim；R4 prediction-object positive不受影响。

下一可用编号：`V65-F13`。

### V65-F13 — scene-ready native launcher未先创建task parent

- attempted run：`run://worldsim_v65/WS-V65-P2V-FRESH-NATIVE-SIDECAR-01/20260827T133000Z__fresh-visited-native-scene-0001-s0-r1`；
- symptom：launcher正确捕获`scene-0001`最终preprocess marker，但native runner在创建run directory前调用
  `shutil.disk_usage(run_dir.parent)`，因task parent尚不存在抛出`FileNotFoundError`；
- exposure audit：失败发生在model/native artifact/quality读取前，未创建failed run directory，不是科学负结果；
- root cause：旧批处理流程总由外层脚本预建task目录，新scene-ready launcher遗漏了该入口前置条件；
- literature/open-source response：Python官方`Path.mkdir(parents=True, exist_ok=True)`明确用于递归创建缺失父目录；
- resolution：launcher在创建executor前预建精确task parent；不改scientific config、scene、seed、run prefix或gates，
  从已经完成的scene marker继续；
- claim impact：无科学read，不改变`WS-V65-H-P2V-001`状态。

下一可用编号：`V65-F14`。

### V65-F14 — scene-ready launcher绕过base+overlay解析器

- attempted runs：`...fresh-visited-native-scene-0001-s0-r1`与`...scene-0219-s0-r1`；
- symptom：task/run dirs创建后，V6.3 generic runner访问`config["inputs"]`抛出`KeyError`；
- exposure audit：两个目录只含空`plans/reports/logs`，未构造worker plan、未加载IR-WM、未读native或quality；
- root cause：`p2v_native_sidecars_v1.yaml`与此前成功P2配置一样是`base_config + overlay`，但新launcher直接调用
  generic runner，绕过了负责合并的`run_worldsim_v64_fresh_sidecars.py`；
- literature/open-source response：Hydra Defaults List和OmegaConf structured config均把完整schema composition置于
  runtime access之前；项目迁移采用已有wrapper作为同一composition boundary，而非复制backend字段；
- resolution：launcher切换到已验证wrapper；空失败dirs改名保留，原r1 canonical path继续，scientific合同不变；
- claim impact：无科学read，不改变P2V hypothesis或fresh exposure计数。

下一可用编号：`V65-F15`。

### V65-F15 — formal P2V evaluator错误假设q0 logits为二维

- run：`run://worldsim_v65/WS-V65-P2V-VISITED-STATE-TRANSFER-01/20260827T141500Z__fresh-visited-transfer-s0-r1`；
- symptom：第1个unit的frozen q0 forward返回1-D logits，`.squeeze(1)`抛出`IndexError: Dimension out of range`；
- exposure audit：1个formal input/target unit加载到内存；没有Qagg、target value、aggregate metric、gate或verdict输出，
  compact cache未落盘；因此不是零input exposure，但没有可用于方法选择的科学反馈；
- root cause：保存的q0 wrapper已在forward中移除singleton output dim，evaluator重复按`[B,1]`做维度特定squeeze；
- literature/open-source response：PyTorch Linear保留除最后feature维外的batch形状，`squeeze(dim)`只适用于存在的指定维；
  对已冻结scalar scorer的batch contract使用`.reshape(-1)`兼容`[B]`与`[B,1]`；
- resolution：只改tensor view，不改数值、model、target、sampling、candidate或gates；r1标记failed，r2继续同一冻结read；
- claim impact：r1不产生科学结论；fresh input已部分暴露，必须在论文/ledger披露该engineering recovery。

下一可用编号：`V65-F16`。

### V65-F16 — P3C canonical evidence命令遗漏required processed-root

- attempted entry：`WS-V65-P3C-CALIBRATION-EVIDENCE-01/20260827T150000Z__calibration-evidence-s0-r1`命令；
- symptom：Python `argparse` 报`--processed-root` required并在解析阶段退出；
- exposure audit：未创建run directory，未读processed input、evidence或quality，未产生metric/gate/verdict；
- root cause：旧query-dataset runner将processed root作为required CLI option，手动启动只传入config与run-dir；
- literature/open-source response：Python官方`argparse`文档说明`required=True`选项缺失时`parse_args()`必须先报错，
  因此该入口不构成科学读取；
- resolution：仅补充冻结的standard processed root，同一canonical run-id成功产生72 units；不改scene、reuse、
  seed、target或gate；
- claim impact：无科学暴露、无方法选择信息；P3C仍保持formal calibration read=false。

下一可用编号：`V65-F17`。

### V65-F17 — P3C formal config指向不存在的冻结q0 artifact

- run：`run://worldsim_v65/WS-V65-P3C-MONOTONE-CALIBRATION-TRANSFER-01/20260827T154500Z__calibration-transfer-s0-r1`；
- symptom：`joblib.load` 在不存在的`WS-V64-P1-BASELINE-TRANSFER-01/.../models/full_native_selective_mlp.joblib`
  抛`FileNotFoundError`；
- exposure audit：run dir仅含resolved config和running status；0 unit/native/target read，0 q0 output，0 metric/gate，
  compact cache未产生；
- root cause：P3C config手工填入的run-relative locator没有沿用已成功P2V的artifact二元组；baseline-transfer
  run不拥有该model；
- literature/open-source response：`joblib.load` 以给定filename/path还原artifact；MLflow等实验跟踪系统同样将
  artifact path定义为run root的相对路径，run与relative path必须同步；
- resolution：更正为P2V实际使用的`WS-V64-P6R-SELECTIVE-MLP-01/.../RISK_MODEL/
  full_native_selective_mlp.joblib`；仅locator改变，同一冻结model与所有科学合同不变；
- claim impact：r1无科学结论；formal calibration read仍为false，r2允许继续。

下一可用编号：`V65-F18`。

### P3C outcome note — V65-F17后窄修复r2成功

- recovery run：`run://worldsim_v65/WS-V65-P3C-MONOTONE-CALIBRATION-TRANSFER-01/20260827T155000Z__calibration-transfer-s0-r2`；
- same-contract audit：仅q0 run-relative artifact locator改变；冻结slope/bias、scene、input、sampling、target、seed和
  gates与r1完全一致；
- outcome：60 eligible units，MSE -92.80%，5-bin calibration error -88.31%，5/5 evaluable scenes改善，
  ranking/AUROC/selected set精确不变；6/6 gates；
- interpretation：V65-F17确认为纯artifact entry failure，不会遮蔽也不夸大r2的independent calibration result。

下一可用编号仍为：`V65-F18`。

### P10X stop-rule note — combined confirmation不设恢复cohort

- candidate已由P2V/P3C/P10V冻结；P10X不产生新模型、参数、threshold或action generator；
- confirmation只有6个与已支持分量直接对应的核心gates，scene/case/subgroup其余数值仅报告；
- 若任一核心gate失败，关闭combined candidate，不换scene、不创建第二confirmation cohort、不以critic补救；
- 下一可用failure id保持`V65-F18`。

下一可用编号仍为：`V65-F18`。

### P10V outcome note — direct action ranking成功，未触发critic recovery

- run：`run://worldsim_v65/WS-V65-P10V-ACTION-VISITED-STATE-TRANSFER-01/20260828T003000Z__action-transfer-s0-r1`；
- outcome：Spearman `0.7402`、unsafe AUROC `0.8588`、pairwise concordance `0.7325`、selected cost -33.26%、
  scene `6/0/0`，6/6 gates；
- prevention result：V64-F28的learned collision critic没有被重开；无head、threshold、lattice或label recovery；
- exclusions：51/864 actions由事前16-point footprint rule排除，72 stop rows由事前合同排除，非失败或事后门控；
- claim impact：支持fixed-action visited-state ranking，但不恢复collision/planning/safety authority。

下一可用编号仍为：`V65-F18`。

### P10V input-pipeline outcome note — 无新failure

- targeted shards 2/6/9找到冻结cohort全部10,709 members，未触发same-cohort full-scan fallback；
- archive extractor使用已有partial-file atomic rename，scene-ready feeder不需哈希/校验和/指纹或内容验证；
- 6/6 preprocess、native及72-unit evidence全部完成，role overlap=0，过程未读action quality；
- P10V formal read前的下一可用failure id保持`V65-F18`。

下一可用编号仍为：`V65-F18`。

### P10V prevention note — 不重复V64-F28 collision-critic路径

- V6.4的10-feature linear critic在独立cohort发生unsafe ranking退化，已终止为`V64-F28`；
- P10V只复用质量读取前已冻结的13-action generator，不复用critic、threshold、collision label或policy gate；
- 新对象是trajectory-visited hidden-FREE rate排序，不将world-state reliability冒充collision/safety authority；
- 若直接Qmean无action ranking，立即关闭action family，不训练critic补救。

下一可用编号仍为：`V65-F18`。

### V65-F18 — P10X scene-ready feeder首次入口缺少仓库级Python import path

- attempted entry：从仓库根目录直接执行`python scripts/launch_worldsim_v65_scene_ready_preprocess.py --help`；
- symptom：入口在导入`from scripts.prepare_dr_v2_drivestudio_scene import ...`时抛
  `ModuleNotFoundError: No module named 'scripts'`；
- exposure audit：发生在参数解析、run创建、scene/member/processed/native/evidence/quality读取之前；正在运行的三个archive
  scan children未停止，formal confirmation read仍为false；
- root cause：Python直接执行文件时将脚本目录而非仓库根目录置于`sys.path[0]`；仓库内绝对`from scripts...`
  import需要显式仓库搜索路径；
- literature/open-source response：Python官方`sys.path`与命令行文档说明`python script.py`首先加入脚本所在目录，
  `PYTHONPATH`用于扩展module search path；
- resolution：不改代码和科学合同，只对该进程设置`PYTHONPATH=.`；scene-ready preprocess与native watcher均已正常运行；
- claim impact：纯入口环境失败，0科学暴露，不触发第二cohort或任何gate/threshold修改。

下一可用编号：`V65-F19`。

### P10X input-pipeline outcome note — V65-F18后无新failure

- targeted shards 3/7/8找到冻结cohort全部10,718 members，未触发same-cohort full-scan fallback；
- `PYTHONPATH=.`恢复后6/6 preprocess、native及72-unit evidence全部完成，role overlap=0；
- partial evidence的48 units与后两场preprocess/native重叠，canonical只补24 units；
- aggregate仅建立unit symlink，未复制数组、未重复native inference、未增加哈希/校验和/指纹；
- 过程未读combined quality，下一可用failure id保持`V65-F19`。

下一可用编号仍为：`V65-F19`。

### V65-F19 — P10X one-shot combined confirmation未达到direct action-selection benefit

- run：`run://worldsim_v65/WS-V65-P10X-COMBINED-CONFIRMATION-01/20260828T013000Z__combined-confirmation-s0-r1`；
- symptom：route ranking、frozen-map MSE、action ranking、unsafe AUROC和pairwise共5门通过，但lowest-Qmean 25%
  action selection将actual cost从`0.120215`降至`0.100520`，只降低`16.38%`，未达冻结`25%`；
- support retained：route Spearman=`0.6098`、frozen calibration MSE -81.39%、action Spearman=`0.7729`、unsafe
  AUROC=`0.9727`、pairwise=`0.6557`；这些支持reliability evaluation，不足以越权为direct action authority；
- heterogeneity：action selection在5 scenes改善、1 scene (`0817`)退化；`0718`虽无eligible nominal route，但其action
  selection改善，故失败不能归因于单一footprint exclusion，也不允许删scene；
- literature/open-source response：AISTATS 2025的excess-risk分解指出recalibration可消除miscalibration regret，仍可能
  留下需要更强post-training的grouping loss；UAI 2023也报告recalibration通常不直接改善downstream regret。本结果与该
  边界一致，因此不事后改metric或把5/6门包装为authority成功；
- resolution：按冻结stop rule关闭combined/direct-action candidate；不换scene、不降gate、不做第二confirmation、不训练
  critic或重开V64-F28；保留given-`tau` visited-state ranking/calibration这一窄结论；
- claim impact：V6.5不获得planner/policy/closed-loop/collision/safety authority；P10X终态为negative combined result。

下一可用编号：`V65-F20`。

### P10X terminal note — 不以新cohort修复V65-F19

- 这次是冻结candidate的唯一formal confirmation read，cache reused=false，未发生入口/资源错误；
- one-shot verdict必须由6门AND rule决定，selected-cost failure不可由其余5门覆盖；
- 任何local/multicalibration或decision-focused head都将是未来版本的新方法、新protocol和新分支，不能作为V6.5恢复；
- V6.5用于arXiv的准确贡献边界是task-conditioned visited-state reliability evaluator，而非action authority compiler。

下一可用编号仍为：`V65-F20`。

### ArXiv report handoff audit note — 无新failure

- documentation-only阶段只检查P2V/P3C/P10V/P10X canonical summary/status存在且JSON可读，并确认P10V action rows=813；
- 首次审计在非登录shell调用裸`python`时命中既有PATH约束；使用项目环境绝对路径后同一read-only检查通过；
- 未创建run、未重新计算metric、未读取新quality、未改变任何verdict，也未增加hash/checksum/fingerprint；
- 技术报告、证据索引、主计划终态与`FAILURE_ANALYSIS.md`更新不构成新科学trial。

下一可用编号仍为：`V65-F20`。
